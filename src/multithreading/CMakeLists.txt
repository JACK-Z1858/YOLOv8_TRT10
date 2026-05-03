
cmake_minimum_required(VERSION 3.12)

project(mCUDAslot LANGUAGES CXX) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# OpenCV
find_package(OpenCV REQUIRED)

# CUDA
find_package(CUDAToolkit REQUIRED)
get_filename_component(CUDA_LIB_DIR ${CUDAToolkit_LIBRARY_DIR} DIRECTORY)

# TensorRT
find_package(TensorRT REQUIRED)

if (TensorRT_VERSION_MAJOR GREATER_EQUAL 10)
    message(STATUA "Build with -DTRT_10")
    add_definitions(-DTRT_10)
endif()

list(APPEND ALL_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
)

list(APPEND ALL_LIB_DIRS
    ${CUDA_LIB_DIR}
    ${TensorRT_LIB_DIRS}
)

list(APPEND ALL_LIBS
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    ${TensorRT_LIBRARIES}
)

add_executable(
    ${PROJECT_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/thread.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/yolov8.cpp
)

target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
    ${ALL_INCLUDE_DIRS}
)

target_link_directories(
    ${PROJECT_NAME}
    PRIVATE
    ${ALL_LIB_DIRS}
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
    ${ALL_LIBS}
)