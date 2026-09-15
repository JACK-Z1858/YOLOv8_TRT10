// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "yolo/types.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/core/cuda.hpp>
#include <memory>
#include <string>
#include <vector>

namespace yolo {

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override;
};

class TrtEngine;

class TrtWorker {
public:
    explicit TrtWorker(const TrtEngine& engine);
    ~TrtWorker();

    TrtWorker(const TrtWorker&) = delete;
    TrtWorker& operator=(const TrtWorker&) = delete;

    void process(Frame& frame);

private:
    struct Buffer {
        std::string name;
        nvinfer1::DataType type{};
        std::size_t bytes{};
        void* device{};
        void* host{};
        bool input{};
    };

    const TrtEngine& engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_{};
    std::vector<Buffer> buffers_;
    int input_height_{};
    int input_width_{};
    cv::cuda::GpuMat source_;
    cv::cuda::GpuMat resized_;
    cv::cuda::GpuMat padded_;
    std::vector<cv::cuda::GpuMat> channels_;

    Buffer& buffer(const std::string& name);
    void preprocess(const cv::Mat& image, float& scale, float& pad_x, float& pad_y);
    void postprocess(Frame& frame, float scale, float pad_x, float pad_y);
};

class TrtEngine {
public:
    explicit TrtEngine(const std::string& engine_path);

    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

private:
    friend class TrtWorker;
    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
};

}  // namespace yolo
