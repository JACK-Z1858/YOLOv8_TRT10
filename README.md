# YoloTRTFlow v0.1.0

YoloTRTFlow is a compact YOLOv8 inference pipeline built with TensorRT 10 and OpenCV CUDA.
It loads a TensorRT `.engine` containing EfficientNMS and accepts images, image
directories, videos, cameras, and video streams.

Runtime options are stored in a YAML file, including the CUDA device, worker
count, queue depth, display and video output, and benchmark range. Each worker
owns an independent TensorRT execution context, CUDA stream, and set of I/O
buffers. Completed frames are restored to their original order before output.

The repository also includes a minimal serial Python CPU demo for a simple
performance comparison.

## Project layout

```text
├─ src/                    C++ implementation
├─ include/yolo/           C++ headers
├─ config.example.yaml     Example runtime configuration
├─ cpu_demo.py             Serial Python CPU baseline
├─ export-det.py           YOLOv8 ONNX export utility
└─ models/common.py        EfficientNMS export support
```

## C++ TensorRT application

Requirements:

- CMake 3.18 or newer
- A C++17 compiler
- CUDA
- TensorRT 10
- OpenCV built with CUDA modules

Build and run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTensorRT_ROOT=/path/to/TensorRT
cmake --build build --config Release
./build/yolo_trt_flow config.example.yaml
```

With a multi-config Windows generator, the executable is commonly located at:

```powershell
.\build\Release\yolo_trt_flow.exe config.example.yaml
```

Copy and edit `config.example.yaml` to select the engine, input source, worker
count, and queue depth. Setting `workers` to `1` uses one inference worker.

## Python CPU baseline

```bash
python -m pip install -r requirements.txt
python cpu_demo.py yolov8s.pt data/passby.mp4
```

The demo runs on CPU with batch size 1 and processes frames serially. Use the
original `.pt` weights: the ONNX produced by `export-det.py` contains a
TensorRT-only EfficientNMS node and is not intended for CPU inference.

## Model export

```bash
python export-det.py --weights yolov8s.pt --sim
trtexec --onnx=yolov8s.onnx --saveEngine=yolov8s.engine
```

TensorRT engines are generally tied to the CUDA, TensorRT, and GPU environment
in which they are built. Rebuild the engine on the deployment machine when
those environments differ.

## Version history

### v0.1.0

- Provides one configurable C++ TensorRT inference application.
- Adds YAML configuration for workers, queues, input, output, and benchmarking.
- Adds a minimal serial Python CPU baseline.

### v0.0.9

- Experimental version before the project restructuring.
- Included separate OpenCV CPU, single-threaded TensorRT, and multi-worker
  TensorRT examples.
- Examples were built independently and used a mixture of hard-coded and
  command-line options.

## License and acknowledgments

Copyright (C) 2026 Jack-Z1858.

This project is licensed under the
[GNU Affero General Public License v3.0](LICENSE).

Portions are derived from
[triple-Mu/YOLOv8-TensorRT](https://github.com/triple-Mu/YOLOv8-TensorRT),
licensed under the MIT License. The Python utilities use
[Ultralytics](https://github.com/ultralytics/ultralytics), which has its own
licensing terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the
required notices and details.
