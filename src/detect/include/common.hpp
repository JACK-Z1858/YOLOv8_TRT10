//
// Created by ubuntu2404 on 2026/3/21
//
#ifndef COMMON_HPP
#define COMMON_HPP
#include "NvInfer.h"    //TensorRT头文件 外部依赖
#include "filesystem.hpp"   //
#include "opencv2/opencv.hpp"   //OpenCV头文件 外部依赖

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

#endif