//
// Created by ubuntu2404 on 2026/3/23
//
#pragma once
#include "common.hpp"

class YOLOv8 {
public:
    explicit YOLOv8(const std::string& engine_file_path);
    ~YOLOv8();

    void        make_pipe(bool warmup = true);
    void        copy_from_Mat(const cv::Mat& image);
    void        copy_from_Mat(const cv::Mat& image, cv::Size& size);
    void        letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);
    void        infer();
    void        postprocess(std::vector<det::Object>& objs);
    static void draw_objects(const cv::Mat&                                image,
                             cv::Mat&                                      res,
                             const std::vector<det::Object>&                    objs,
                             const std::vector<std::string>&               CLASS_NAMES,
                             const std::vector<std::vector<unsigned int>>& COLORS);
    int                  num_bindings;
    int                  num_inputs  = 0;       // 输入 tensor 信息
    int                  num_outputs = 0;       // 输出 tensor 信息
    std::vector<det::Binding> input_bindings;        
    std::vector<det::Binding> output_bindings;       
    std::vector<void*>   host_ptrs;             // CPU 内存指针
    std::vector<void*>   device_ptrs;           // GPU 内存指针

    det::PreParam pparam;

private:
    nvinfer1::ICudaEngine*          engine  = nullptr;          // 加载 .engine 文件
    nvinfer1::IRuntime*             runtime = nullptr;          // TensorRT 运行时
    nvinfer1::IExecutionContext*    context = nullptr;          // 执行上下文
    cudaStream_t                    stream  = nullptr;          // CUDA 流，用于异步执行
    Logger                          gLogger{nvinfer1::ILogger::Severity::kERROR};   // 类内部声明成员变量时用"{}"，声明成员函数时用"()"
};
