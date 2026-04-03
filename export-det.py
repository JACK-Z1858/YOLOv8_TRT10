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
    
def parse_args():
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
    args = parser.parse_args()
    return args

