# SPDX-License-Identifier: AGPL-3.0-only

cmake_minimum_required(VERSION 3.18)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" YOLO_TRT_FLOW_VERSION)
string(STRIP "${YOLO_TRT_FLOW_VERSION}" YOLO_TRT_FLOW_VERSION)
project(yolo_trt_flow VERSION "${YOLO_TRT_FLOW_VERSION}" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TensorRT REQUIRED)
find_package(Threads REQUIRED)

add_executable(yolo_trt_flow
    src/main.cpp
    src/config.cpp
    src/trt_engine.cpp
)

target_include_directories(yolo_trt_flow PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
)

target_link_directories(yolo_trt_flow PRIVATE ${TensorRT_LIB_DIRS})

target_link_libraries(yolo_trt_flow PRIVATE
    CUDA::cudart
    ${OpenCV_LIBS}
    ${TensorRT_LIBRARIES}
    Threads::Threads
)

if(MSVC)
    target_compile_options(yolo_trt_flow PRIVATE /W4)
else()
    target_compile_options(yolo_trt_flow PRIVATE -Wall -Wextra -Wpedantic)
endif()
