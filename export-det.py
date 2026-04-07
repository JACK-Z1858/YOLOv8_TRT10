import argparse
from io import BytesIO

import onnx
import torch
from ultralytics import YOLO

from models.common import PostDetect, optim

try:
    import onnxsim
except ImportError:
    onnxsim = None
    
# 定义一个函数 parse_args()，用于解析命令行参数
def parse_args():
    # 创建一个 ArgumentParser 对象，用于处理命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-w',
                        '--weights',
                        type=str,
                        required=True,
                        help='PyTorch yolo weights')
    parser.add_argument('--iou-thres',
                        type=float,
                        default=0.65,
                        help='IOU threshold for NMS plugin')
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.25,
                        help='Confidence threshold for NMS plugin')
    parser.add_argument('--topk',
                        type=int,
                        default=1000,
                        help='Maximum number of detection bboxes')
    parser.add_argument('--opset',
                        type=int,
                        default=11,
                        help='ONNX opset version')
    parser.add_argument('--sim',
                        action='store_true',
                        help='simplify the ONNX model')
    parser.add_argument('--input-shape',
                        # number of arguments，组合为列表
                        nargs='+',
                        type=int,
                        default=[1, 3, 640, 640],
                        help='Model input shape only for api builder')
    parser.add_argument('device',
                        type=str,
                        default='cpu',
                        help='Export ONNX device')   
    args = parser.parse_args()
    assert len(args.input_shape) == 4
    PostDetect.conf_thres = args.conf_thres
    PostDetect.iou_thres = args.iou_thres
    PostDetect.topk = args.topk
    return args

def main(args):
    b = args.input_shape[0]
    YOLOv8 = YOLO(args.weights)
    # 将模型融合并设置为评估模式
    model = YOLOv8.model.fuse().eval()
    for m in model.modules():
        optim(m)
        m.to(device=args.device)
    model.to(device=args.device)
    # 预热模型，运行两次前向传播以确保模型已经加载到设备上并且所有的权重都已经准备好
    fake_input = torch.randn(args.input_shape).to(device=args.device)
    for _ in range(2):
        model(fake_input)
    save_path = args.weights.replace('.pt', '.onnx')
    with BytesIO() as f:
        torch.onnx.export(
            model,
            fake_input,
            f,
            opset_version=args.opset,
            input_names=['images'],
            output_names=['num_dets', 'boxes', 'scores', 'labels'],)
        f.seek(0)
        onnx_model = onnx.load(f)
    onnx.checker.check_model(onnx_model)
    # 手动修改输出张量的形状信息，以适应后续的 ONNX 模型简化和优化步骤
    shapes = [b, 1, b, args.topk, 4, b, args.topk, b, args.topk] 
    for i in onnx_model.graph.output:
        for j in i.type.tensor_type.shape.dim:
            j.dim_param = str(shapes.pop(0))
    if args.sim:
        try:
            # 函数返回元组，onnx_model 是简化后的模型，check 是一个布尔值，表示简化是否成功
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
        except Exception as e:
            print(f'simplify ONNX model failed: {e}')
    onnx.save(onnx_model, save_path)
    print(f'ONNX model has been saved to {save_path}')
    
if __name__ == '__main__':
    main(parse_args())