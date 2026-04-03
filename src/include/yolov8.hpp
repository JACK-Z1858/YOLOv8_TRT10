//
// Created by ubuntu2404 on 2026/3/23
//

#ifndef YOLOV8_HPP
#define YOLOV8_HPP
#include "NvInferPlugin.h"
#include "include/common.hpp"
#include <fstream>
using namespace det;

class YOLOv8 {
public:
    explicit YOLOv8(const std::string& engine_file_path);
    ~YOLOv8();

    void        make_pipe(bool warmup = true);
    void        copy_from_Mat(const cv::Mat& image);
    void        copy_from_Mat(const cv::Mat& image, cv::Size& size);
    void        letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);
    void        infer();
    void        postprocess(std::vector<Object>& objs);
    static void draw_objects(const cv::Mat&                                image,
                             cv::Mat&                                      res,
                             const std::vector<Object>&                    objs,
                             const std::vector<std::string>&               CLASS_NAMES,
                             const std::vector<std::vector<unsigned int>>& COLORS);
    int                  num_bindings;
    int                  num_inputs  = 0;       // 输入 tensor 信息
    int                  num_outputs = 0;       // 输出 tensor 信息
    std::vector<Binding> input_bindings;        
    std::vector<Binding> output_bindings;       
    std::vector<void*>   host_ptrs;             // CPU 内存指针
    std::vector<void*>   device_ptrs;           // GPU 内存指针

    PreParam pparam;

private:
    nvinfer1::ICudaEngine*          engine  = nullptr;          // 加载 .engine 文件
    nvinfer1::IRuntime*             runtime = nullptr;          // TensorRT 运行时
    nvinfer1::IExecutionContext*    context = nullptr;          // 执行上下文
    cudaStream_t                    stream  = nullptr;          // CUDA 流，用于异步执行
    Logger                          gLogger{nvinfer1::ILogger::Severity::kERROR};   // 类内部声明成员变量时用"{}"，声明成员函数时用"()"
};

// 类外定义构造函数
YOLOv8::YOLOv8(const std::string& engine_file_path) {
    /* 加载 .engine 文件 */
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());                                        // 文件打开错误，终止程序
    file.seekg(0, std::ios::end);                               // 把文件读取指针移动到距离文件末尾 0 字节的位置
    auto size = file.tellg();                                   // 获取当前位置（文件大小）
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];                      // 动态分配内存（char刚好占 1 字节）
    assert(trtModelStream);                                     // 内存分配失败
    file.read(trtModelStream, size);                            // 把文件写进内存
    file.close();
    
    /* 反序列化创建运行时 */
    initLibNvInferPlugins(&this->gLogger, "");
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);
    
    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    delete[] trtModelStream;
    this->context = this->engine->createExecutionContext();

    assert(this->context != nullptr);
    cudaStreamCreate(&this->stream);

    // TRT_10
    this->num_bindings = this->engine->getNbIOTensors();

    for (int i = 0; i < this->num_bindings; ++i) {
        Binding         binding;
        nvinfer1::Dims  dims;

        // TRT_10
        std::string             name  = this->engine->getIOTensorName(i);
        nvinfer1::DataType      dtype = this->engine->getTensorDataType(name.c_str());
        // end TRT_10

        binding.name   = name;
        binding.dsize  = type_to_size(dtype);

        // TRT_10
        // 判断当前 tensor 是 input 还是 output
        bool IsInput = engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;

        if (IsInput) {
            this->num_inputs += 1;
            // TRT_10
            // 通过优化配置文件 profile 获取动态张量形状
            dims = this->engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
            // set max opt shape
            this->context->setInputShape(name.c_str(), dims);
            // end TRT_10
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
        } else {
            // TRT_10
            // 从 context 获取 tensor 形状
            dims = this->context->getTensorShape(name.c_str());

            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }    
}

// 类外定义析构函数
YOLOv8::~YOLOv8(){
    delete this->context;
    delete this->engine;
    delete this->runtime;

    cudaStreamDestroy(this->stream);
    for (auto& ptr : this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto& ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}

/* 初始化推理管道 */
void YOLOv8::make_pipe(bool warmup) {
    
}

void YOLOv8::copy_from_Mat(const cv::Mat& image) {

}

void YOLOv8::copy_from_Mat(const cv::Mat& iamge, cv::Size& size) {

}

/* 等比例缩放图片到 640x640 */
void YOLOv8::letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size) {

}

/* 执行推理 */
void YOLOv8::infer() {

}

/* 后处理输出结果 */
void YOLOv8::postprocess(std::vector<Object>& objs) {

}

/* 可视化 */
void YOLOv8::draw_objects(const cv::Mat&                                image,
                          cv::Mat&                                      res,
                          const std::vector<Object>&                    objs,
                          const std::vector<std::string>&               CLASS_NAMES,
                          const std::vector<std::vector<unsigned int>>& COLORS)
{

}

#endif