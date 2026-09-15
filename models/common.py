# SPDX-License-Identifier: AGPL-3.0-only
# Portions derived from triple-Mu/YOLOv8-TensorRT (MIT); see THIRD_PARTY_NOTICES.md.

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor, Graph, Value

# 得到锚点表和步长表，锚点表的shape为(n_anchors, 2)，步长表的shape为(n_anchors, 1)
def make_anchors(feats: Tensor,
                 strides: Tensor,
                 grid_cell_offset: float = 0.5) -> Tuple[Tensor, Tensor]:
    # 初始化两个空列表，分别用于存储锚点和步长张量
    anchor_points, stride_tensor = [], []
    # 确保输入的特征图列表不为空
    assert feats is not None
    # 获取第一个特征图的dtype和device信息，以便后续创建张量时使用相同的类型和设备
    dtype, device = feats[0].dtype, feats[0].device
    # 遍历每个特征图和对应的步长
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        # 生成特征图的 x 坐标序列，并添加网格单元偏移，每个anchor位于网格中心，shift x
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        # shift y
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        # 生成网格坐标矩阵，sx 和 sy 的形状为 (h, w)，每个元素表示对应位置的坐标
        sy, sx = torch.meshgrid(sy, sx)
        # 将两个(h, w)堆叠成一个(h, w, 2)的张量，并调整形状为(h*w, 2)，表示所有锚点的坐标
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        # 创建一个形状为(h*w, 1)的张量，填充为当前步长值，并添加到步长列表中
        stride_tensor.append(
            torch.full((h * w, 1), stride, dtype=dtype, device=device))
        # 最后将所有特征图的锚点和步长张量拼接成一个大的张量，并返回
    return torch.cat(anchor_points), torch.cat(stride_tensor)

# 定义一个自定义的 PyTorch autograd 函数，用于实现 TensorRT 的非极大值抑制（NMS）操作      
class TRT_NMS(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx: Graph,                 # 上下文对象，用于在前向和后向传播之间传递信息
            boxes: Tensor,              # 输入的边界框张量，形状为 (N, 4)，其中 N 是边界框的数量，每个边界框由四个坐标值表示
            scores: Tensor,             # 输入的分数张量，形状为 (N, C)，其中 N 是边界框的数量，C 是类别的数量，每个元素表示对应边界框的类别分数
            iou_threshold: float = 0.65,        # IoU（Intersection over Union）阈值，用于决定是否抑制重叠的边界框
            score_threshold: float = 0.25,      # 分数阈值，用于过滤掉低分数的边界框
            max_output_boxes: int = 100,        # 最大输出边界框数量，限制最终返回的边界框数量
            background_class: int = -1,         # 背景类别索引，如果设置为 -1 则不考虑背景类别
            box_coding: int = 0,                # 边界框编码方式，0 表示 (x1, y1, x2, y2)，1 表示 (cx, cy, w, h)
            plugin_version: str = '1',          # 插件版本信息，可能用于选择不同版本的 NMS 实现
            score_activation: int = 0           # 分数激活函数类型，0 表示不使用激活函数，1 表示使用 sigmoid 激活函数
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size, num_boxes, num_classes = scores.shape
        num_dets = torch.randint(0,
                                 max_output_boxes, (batch_size, 1),
                                 dtype=torch.int32)
        boxes = torch.randn(batch_size, max_output_boxes, 4)
        scores = torch.randn(batch_size, max_output_boxes)
        labels = torch.randint(0, 
                               num_classes, (batch_size, max_output_boxes), 
                               dtype=torch.int32)
        return num_dets, boxes, scores, labels
        
    @staticmethod
    def symbolic(
            g,
            boxes: Value,
            scores: Value,
            iou_threshold: float = 0.45,
            score_threshold: float = 0.25,
            max_output_boxes: int = 100,
            background_class: int = -1,
            box_coding: int = 0,
            score_activation: int = 0,
            plugin_version: str = '1') -> Tuple[Value, Value, Value, Value]:
        out = g.op('TRT::EfficientNMS_TRT',
                   boxes, 
                   scores,
                   iou_threshold_f=iou_threshold,
                   score_threshold_f=score_threshold,
                   max_output_boxes_i=max_output_boxes,
                   background_class_i=background_class,
                   box_coding_i=box_coding,
                   plugin_version_s=plugin_version,
                   score_activation_i=score_activation,
                   outputs=4)
        nums_dets, boxes, scores, classes = out
        return nums_dets, boxes, scores, classes

# 定义一个新的类 C2f_TRT，继承自 nn.Module，用于实现优化后的 C2f 模块   
class C2f(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, x):
        x = self.cv1(x)
        x = [x, x[:, self.c:, ...]]
        x.extend(m(x[-1]) for m in self.m)
        x.pop(1)
        return self.cv2(torch.cat(x, dim=1)) 

# 定义一个新的类 PostDetect，继承自 nn.Module，用于实现优化后的 Detect 模块的后处理逻辑   
class PostDetect(nn.Module):
    export = True
    shape = None
    dynamic = False
    iou_thres = 0.65
    conf_thres = 0.25
    topk = 100
    
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    # x 是一个包含多个特征图的列表 
    def forward(self, x):
        shape = x[0].shape
        # batch，结果列表，回归信息维度，回归信息维度等于 reg_max * 4，因为每个边界框有4个坐标值，每个坐标值有 reg_max 个离散化的回归值
        b, res, b_reg_num = shape[0], [], self.reg_max * 4
        # nl: number of layers，num_anchors: 每个特征图上的锚点数量，num_classes: 类别数量
        for i in range(self.nl):
            # cv2:边界框回归，cv3:类别预测，拼接后得到一个(batch, b_reg_num + num_classes, h, w)的张量
            # dims=1表示在特征维度上进行拼接，得到的结果是每个anchor对应一个包含回归和分类信息的向量
            # [batch, b_reg_num = 64, h, w] + [batch, num_classes = 80, h, w] -> [batch, 64 + 80, h, w]
            res.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(
                0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        # 将每个特征图的结果调整形状为 (batch, b_reg_num + num_classes, num_anchors(h * w))，然后在第2个(0, 1, 2)维度上拼接
        x = [i.view(b, self.no, -1) for i in res]
        y = torch.cat(x, 2)
        # 将回归信息和分类信息分开，回归信息的形状为 (batch, b_reg_num, num_anchors)，分类信息经过形状为 (batch, num_classes, num_anchors)
        boxes, scores = y[:, :b_reg_num, ...], y[:, b_reg_num:, ...].sigmoid()
        # (b, 64, 8400) -> (b, 4, 16, 8400) -> (b, 4, 8400, 16)
        boxes = boxes.view(b, 4, self.reg_max, -1).permute(0, 1, 3, 2)
        # [1, 4, 8400, 16] @ [16] -> [1, 4, 8400]，通过softmax将离散化的回归值转换为连续的坐标值，得到每个边界框的坐标信息
        boxes = boxes.softmax(-1) @ torch.arange(self.reg_max).to(boxes)
        # dim[1] = [l, t, r, b]，分别表示边界框的左、上、右、下坐标值
        boxes0, boxes1 = -boxes[:, :2, ...], boxes[:, 2:, ...]
        boxes = self.anchors.repeat(b, 2, 1) + torch.cat([boxes0, boxes1], 1)
        boxes = boxes * self.strides
        
        return TRT_NMS.apply(boxes.transpose(1, 2), scores.transpose(1, 2), 
                             self.iou_thres, self.conf_thres, self.topk)
        

# 替换模型的类为优化后的类，以便在推理过程中使用更高效的实现
# 修改__class__属性不会重新初始化对象，因此原有的属性和方法仍然保留，但新的类可以覆盖或添加新的方法来实现优化后的功能   
def optim(module: nn.Module) -> nn.Module:
    s = str(type(module))[6:-2].split('.')[-1]
    #s = module.__class__.__name__
    if s == 'Detect':
        setattr(module, '__class__', PostDetect)
    elif s == 'C2f':
        setattr(module, '__class__', C2f)
