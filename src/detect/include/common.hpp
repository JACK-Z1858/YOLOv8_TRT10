//
// Created by ubuntu2404 on 2026/3/21
//
#pragma once   
#include "NvInfer.h"    //TensorRT头文件 外部依赖
#include <iostream>
#include "opencv2/core/types.hpp"
//CUDA错误检查宏
#define CHECK(call)                                                                 \
    do {                                                                            \
        const cudaError_t error_code = call;                                        \
        if (error_code != cudaSuccess) {                                            \
            printf("CUDA ERROR:\n");                                                \
            printf("    File:   %s\n", __FILE__);                                   \
            printf("    Line:   %d\n", __LINE__);                                   \
            printf("    Error code: %d\n", error_code);                             \
            printf("    Error text: %s\n", cudaGetErrorString(error_code));         \
            exit(1);                                                                \
        }                                                                           \
    } while(0)

class Logger: public nvinfer1::ILogger {
public:
    nvinfer1::ILogger::Severity reportaleSeverity;
    
    explicit Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO) :
        reportaleSeverity(severity) {}
    
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        if (severity > reportaleSeverity) {
            return;
        }
        switch (severity) {
            case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
                std::cerr << "kINTERNAL_ERROR: ";
                break;
            case nvinfer1::ILogger::Severity::kERROR:
                std::cerr << "ERROR: ";
                break;
            case nvinfer1::ILogger::Severity::kWARNING:
                std::cerr << "WARNING: ";
                break;
            case nvinfer1::ILogger::Severity::kINFO:
                std::cerr << "INFO: ";
                break;
            default:
                std::cerr << "VERBOSE: ";
                break;
        }
        std::cerr << msg <<std::endl;
    }
};

//获取模型的输入形状
inline int get_size_by_dims(const nvinfer1::Dims& dims) {
    int size = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        size *= dims.d[i];
    }
    return size;
}

//获取数据精度
inline int type_to_size(const nvinfer1::DataType& dataType) {
    switch (dataType) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kBOOL:
            return 1;
        default:
            return 4;
    }
}

// 钳制在 min 和 max 之间
inline static float clamp(float val, float min, float max) {
    return val > min ? (val < max ? val : max) : min;
}

namespace det {

// 绑定 / 端口
struct Binding {
    size_t          size  = 1;
    size_t          dsize = 1;
    nvinfer1::Dims  dims;
    std::string     name;
};

// 检测到的目标
struct Object {
    cv::Rect_<float> rect;          // 检测框
    int              label = 0;     // 类别标签
    float            prob  = 0.0;   // 置信度
};

// 预处理参数表 Pre-processing parameters
struct PreParam {
    float ratio  = 1.0f;    // 缩放比例
    float dw     = 0.0f;    // delta width 宽度方向贴边
    float dh     = 0.0f;    // delta height 高度方向贴边
    float height = 0;       // 原始高度
    float width  = 0;       // 原始宽度
};
} // namespace det

const std::vector<std::string> CLASS_NAMES = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
    {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
    {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
    {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
    {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
    {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
    {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
    {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
    {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
    {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
    {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
    {80, 183, 189},  {128, 128, 0}};