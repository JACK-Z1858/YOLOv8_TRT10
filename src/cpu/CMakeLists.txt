cmake_minimum_required(VERSION 3.12)

project(yolov8_cpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(Function)

# OpenCV
find_package(OpenCV 4.9 REQUIRED)
print_var(OpenCV_LIBS)
print_var(OpenCV_LIBRARIES)
print_var(OpenCV_INCLUDE_DIRS)

list(APPEND ALL_INCLUDE_DIRS
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

list(APPEND ALL_LIBS
    ${OpenCV_LIBS}
)

print_var(ALL_INCLUDE_DIRS)
print_var(ALL_LIBS)

add_executable(
    ${PROJECT_NAME}
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/yolov8.hpp   
)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${ALL_INCLUDE_DIRS}
)

target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
    ${ALL_LIBS}
)