#pragma once

#include "common.hpp"
#include "thread/thread.hpp"
#include <NvInferRuntime.h>
#include <memory>


class YOLOv8 {
private:
    std::priority_queue<FrameData, std::vector<FrameData>, decltype(cmpFrameData)> buffer{cmpFrameData};
    uint32_t expected_id{0};
    uint32_t max_buffer_size{5};

    std::unique_ptr<nvinfer1::IRuntime>          runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine>       engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t                                 stream_;
    
    Logger  gLogger_;

private:
    void preProcess(cv::cuda::GpuMat &input);
    void postProcess(FrameData& frame);
    void infer();
    void drawObjects(FrameData& frame);
    bool reorderBuffer(const FrameData& frame, threadSafeQueue& out2show);

    cv::cuda::GpuMat blobFromGpuMat(cv::cuda::GpuMat& input);
    cv::cuda::GpuMat resizeKeepAspectRatioPadRightBottom(cv::cuda::GpuMat& input,
                                                         const float       inp_h,
                                                         const float       inp_W,
                                                         const float       r);

public:
    explicit YOLOv8(const std::string& engine_path);
    ~YOLOv8();

    void makepipe();
    void VideoReader(std::string inputVideo, threadSafeQueue& read2work);
    void Worker(threadSafeQueue&read2work, threadSafeQueue& work2out);
    void Outputer(threadSafeQueue& work2out, threadSafeQueue& out2show);

    uint32_t             num_bindings_{0};
    uint32_t             num_inputs_{0};
    uint32_t             num_outputs_{0};
    std::vector<Binding> i_bindings_;
    std::vector<Binding> o_bindings_;
    std::vector<void*>   host_ptrs_;
    std::vector<void*>   device_ptrs_;
    float                ratio_  = 1.0f;
    float                height_ = 0.0f;
    float                width_  = 0.0f;
};