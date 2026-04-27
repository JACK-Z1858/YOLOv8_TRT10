
cmake_minimum_required(VERSION 3.12)

project(multithreadingTest LANGUAGES CXX) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# OpenCV
find_package(OpenCV REQUIRED)

list(APPEND ALL_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

list(APPEND ALL_LIB_DIRS
    
)

list(APPEND ALL_LIBS
    ${OpenCV_LIBS}
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