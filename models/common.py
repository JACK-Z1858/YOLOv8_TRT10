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
        anchor_points.append(torch.stack((sx, sy), dim=-1).view(-1, 2))
        # 创建一个形状为(h*w, 1)的张量，填充为当前步长值，并添加到步长列表中
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        # 最后将所有特征图的锚点和步长张量拼接成一个大的张量，并返回
    return torch.cat(anchor_points), torch.cat(stride_tensor)
        
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
            plugin_version: str = "1.0",        # 插件版本信息，可能用于选择不同版本的 NMS 实现
            score_activation: int = 0,          # 分数激活函数类型，0 表示不使用激活函数，1 表示使用 sigmoid 激活函数
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size, num_boxes, num_classes = scores.shape
        num_dets = torch.randint(0,
                                 max_output_boxes, (batch_size, 1),
                                 dtype=torch.int32)
        boxes = torch.rand(batch_size, num_boxes, 4)
        scores = torch.rand(batch_size, max_output_boxes)
        labels = torch.randint(0, 
                               num_classes, (batch_size, max_output_boxes), 
                               dtype=torch.int32)
        return num_dets, boxes, scores, labels
        
    @staticmethod
    def symbolic(
            g,
            boxes: Value,
            scores: Value,
            iou_threshold: float = 0.65,
            score_threshold: float = 0.25,
            max_output_boxes: int = 100,
            background_class: int = -1,
            box_coding: int = 0,
            plugin_version: str = "1.0",
            score_activation: int = 0,
    ) -> Tuple[Value, Value, Value, Value]:
        out = g.op('TRT::EfficientNM_TRT',
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
        num_dets, boxes, scores, labels = out
        return num_dets, boxes, scores, labels
    
class C2f(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, x):
        x = self.cv1(x)
        x = [x, x[:self.c:, ...]]
        x.extend(self.m(x) for m in self.m)
        x.pop(1)
        return self.cv2(torch.cat(x, dim=1)) 
    
class PostDetect(nn.Module):
    def __init__(self, num_classes: int, anchors: Tensor, strides: Tensor):
        super(PostDetect, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.strides = strides
    
    def forward(self, feats: Tensor) -> Tuple[Tensor, Tensor]:
        # Placeholder for the actual post-processing implementation
        # This function should take the raw output from the model and convert it into bounding boxes and class scores.
        pass
    
class PostSeg(nn.Module):
    def __init__(self, num_classes: int):
        super(PostSeg, self).__init__()
        
        self.num_classes = num_classes
    
    def forward(self, feats: Tensor) -> Tensor:
        # Placeholder for the actual post-processing implementation for segmentation
        pass
    
def optim(model: nn.Module) -> nn.Module:
    s = str(type(model)[6:-2].split('.')[-1])
    if s == 'Detect':
        setattr(model, '__class__', PostDetect)
    elif s == 'Segment':
        setattr(model, '__class__', PostSeg)
    elif s == 'C2f':
        setattr(model, '__class__', C2f)