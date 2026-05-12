import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

device = torch.device("cuda:0")

cap = cv2.VideoCapture("data/passby.mp4")
ret, im0 = cap.read()
cap.release()
if not ret:
    raise SystemExit("读帧失败")

# 与 predict(imgsz=640) 类似的 letterbox 输入
lb = LetterBox(new_shape=(640, 640), auto=True, stride=32)
im = lb(image=im0)
im = im.transpose((2, 0, 1))[::-1]  # HWC -> CHW, BGR -> RGB
im = np.ascontiguousarray(im)
im = torch.from_numpy(im).to(device).float() / 255.0
im = im.unsqueeze(0)  # 1x3x640x640

yolo = YOLO("yolov8s.pt").to(device)
m = yolo.model.eval()
# 若要 FP16 推理（和 half=True 类似），可取消下面两行之一：
# im = im.half()
# m.half()

with torch.inference_mode():
    for _ in range(20):
        m(im)
        torch.cuda.synchronize()

    n = 100
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        m(im)
        torch.cuda.synchronize()
    t1 = time.perf_counter()

print(f"avg forward {(t1 - t0) / n * 1000:.3f} ms (仅 model.model, 无 NMS)")