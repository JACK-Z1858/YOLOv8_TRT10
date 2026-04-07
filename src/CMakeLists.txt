cmake_minimum_required(VERSION 3.12)

set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(yolov8_trt10 LANGUAGES CXX CUDA)

