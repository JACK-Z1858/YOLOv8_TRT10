// SPDX-License-Identifier: AGPL-3.0-only

#include "yolo/trt_engine.hpp"

#include <NvInferPlugin.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

namespace yolo {
namespace {

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

std::size_t element_size(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL: return 1;
        default: throw std::runtime_error("unsupported TensorRT tensor data type");
    }
}

std::size_t volume(const nvinfer1::Dims& dims) {
    std::size_t result = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            throw std::runtime_error("unresolved dynamic tensor shape");
        }
        result *= static_cast<std::size_t>(dims.d[i]);
    }
    return result;
}

const std::vector<std::string> kCocoNames = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
};

cv::Scalar color_for(int label) {
    return cv::Scalar((37 * label + 80) % 255, (17 * label + 160) % 255, (29 * label + 40) % 255);
}

}  // namespace

void Logger::log(Severity severity, const char* message) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cerr << "TensorRT: " << message << '\n';
    }
}

TrtEngine::TrtEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("cannot open engine: " + engine_path);
    }
    const auto end = file.tellg();
    if (end <= 0) {
        throw std::runtime_error("engine file is empty: " + engine_path);
    }
    std::vector<char> bytes(static_cast<std::size_t>(end));
    file.seekg(0, std::ios::beg);
    if (!file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
        throw std::runtime_error("cannot read engine: " + engine_path);
    }

    initLibNvInferPlugins(&logger_, "");
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        throw std::runtime_error("cannot create TensorRT runtime");
    }
    engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (!engine_) {
        throw std::runtime_error("cannot deserialize TensorRT engine");
    }
}

TrtWorker::TrtWorker(const TrtEngine& engine) : engine_(engine) {
    context_.reset(engine_.engine_->createExecutionContext());
    if (!context_) {
        throw std::runtime_error("cannot create TensorRT execution context");
    }
    check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");

    try {
        const int count = engine_.engine_->getNbIOTensors();
        for (int i = 0; i < count; ++i) {
            Buffer item;
            item.name = engine_.engine_->getIOTensorName(i);
            item.type = engine_.engine_->getTensorDataType(item.name.c_str());
            item.input = engine_.engine_->getTensorIOMode(item.name.c_str()) == nvinfer1::TensorIOMode::kINPUT;

            nvinfer1::Dims dims = engine_.engine_->getTensorShape(item.name.c_str());
            if (item.input) {
                if (item.type != nvinfer1::DataType::kFLOAT) {
                    throw std::runtime_error("only FP32 engine input tensors are currently supported");
                }
                bool dynamic = false;
                for (int d = 0; d < dims.nbDims; ++d) dynamic = dynamic || dims.d[d] < 0;
                if (dynamic) {
                    dims = engine_.engine_->getProfileShape(item.name.c_str(), 0, nvinfer1::OptProfileSelector::kOPT);
                    if (!context_->setInputShape(item.name.c_str(), dims)) {
                        throw std::runtime_error("cannot set input shape for " + item.name);
                    }
                }
                if (dims.nbDims != 4 || dims.d[0] != 1 || dims.d[1] != 3) {
                    throw std::runtime_error("expected a 1x3xHxW input tensor");
                }
                input_height_ = dims.d[2];
                input_width_ = dims.d[3];
            } else {
                dims = context_->getTensorShape(item.name.c_str());
            }

            item.bytes = volume(dims) * element_size(item.type);
            buffers_.push_back(item);
            auto& stored = buffers_.back();
            check_cuda(cudaMalloc(&stored.device, stored.bytes), "cudaMalloc");
            if (!stored.input) {
                check_cuda(cudaMallocHost(&stored.host, stored.bytes), "cudaMallocHost");
            }
            if (!context_->setTensorAddress(stored.name.c_str(), stored.device)) {
                throw std::runtime_error("cannot bind tensor " + stored.name);
            }
        }

        if (input_height_ <= 0 || input_width_ <= 0) {
            throw std::runtime_error("engine has no supported image input");
        }
        buffer("num_dets");
        buffer("boxes");
        buffer("scores");
        buffer("labels");
        if (buffer("num_dets").type != nvinfer1::DataType::kINT32 ||
            buffer("labels").type != nvinfer1::DataType::kINT32 ||
            buffer("boxes").type != nvinfer1::DataType::kFLOAT ||
            buffer("scores").type != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error("unsupported EfficientNMS output data types");
        }
    } catch (...) {
        context_.reset();
        for (auto& item : buffers_) {
            if (item.device) cudaFree(item.device);
            if (item.host) cudaFreeHost(item.host);
        }
        if (stream_) cudaStreamDestroy(stream_);
        stream_ = nullptr;
        throw;
    }
}

TrtWorker::~TrtWorker() {
    if (stream_) cudaStreamSynchronize(stream_);
    context_.reset();
    for (auto& item : buffers_) {
        if (item.device) cudaFree(item.device);
        if (item.host) cudaFreeHost(item.host);
    }
    if (stream_) cudaStreamDestroy(stream_);
}

TrtWorker::Buffer& TrtWorker::buffer(const std::string& name) {
    const auto it = std::find_if(buffers_.begin(), buffers_.end(), [&](const Buffer& value) { return value.name == name; });
    if (it == buffers_.end()) {
        throw std::runtime_error("engine is missing required output tensor: " + name);
    }
    return *it;
}

void TrtWorker::preprocess(const cv::Mat& image, float& scale, float& pad_x, float& pad_y) {
    if (image.empty()) {
        throw std::runtime_error("cannot preprocess an empty frame");
    }

    scale = std::min(static_cast<float>(input_width_) / image.cols,
                     static_cast<float>(input_height_) / image.rows);
    const int resized_width = static_cast<int>(std::round(image.cols * scale));
    const int resized_height = static_cast<int>(std::round(image.rows * scale));
    const int left = (input_width_ - resized_width) / 2;
    const int top = (input_height_ - resized_height) / 2;
    pad_x = static_cast<float>(left);
    pad_y = static_cast<float>(top);

    auto cv_stream = cv::cuda::StreamAccessor::wrapStream(stream_);
    source_.upload(image, cv_stream);
    cv::cuda::resize(source_, resized_, cv::Size(resized_width, resized_height), 0, 0, cv::INTER_LINEAR, cv_stream);
    padded_.create(input_height_, input_width_, CV_8UC3);
    padded_.setTo(cv::Scalar(114, 114, 114), cv::noArray(), cv_stream);
    resized_.copyTo(padded_(cv::Rect(left, top, resized_width, resized_height)), cv_stream);

    cv::cuda::split(padded_, channels_, cv_stream);
    auto& input = *std::find_if(buffers_.begin(), buffers_.end(), [](const Buffer& value) { return value.input; });
    float* destination = static_cast<float*>(input.device);
    const int plane = input_height_ * input_width_;
    cv::cuda::GpuMat red(input_height_, input_width_, CV_32F, destination);
    cv::cuda::GpuMat green(input_height_, input_width_, CV_32F, destination + plane);
    cv::cuda::GpuMat blue(input_height_, input_width_, CV_32F, destination + 2 * plane);
    channels_[2].convertTo(red, CV_32F, 1.0 / 255.0, cv_stream);
    channels_[1].convertTo(green, CV_32F, 1.0 / 255.0, cv_stream);
    channels_[0].convertTo(blue, CV_32F, 1.0 / 255.0, cv_stream);
}

void TrtWorker::postprocess(Frame& frame, float scale, float pad_x, float pad_y) {
    const auto* count = static_cast<const int*>(buffer("num_dets").host);
    const auto* boxes = static_cast<const float*>(buffer("boxes").host);
    const auto* scores = static_cast<const float*>(buffer("scores").host);
    const auto* labels = static_cast<const int*>(buffer("labels").host);
    const int max_detections = static_cast<int>(buffer("scores").bytes / sizeof(float));
    const int detections = std::clamp(count[0], 0, max_detections);

    frame.detections.clear();
    frame.detections.reserve(static_cast<std::size_t>(detections));
    for (int i = 0; i < detections; ++i) {
        const float x0 = std::clamp((boxes[i * 4] - pad_x) / scale, 0.0f, static_cast<float>(frame.image.cols));
        const float y0 = std::clamp((boxes[i * 4 + 1] - pad_y) / scale, 0.0f, static_cast<float>(frame.image.rows));
        const float x1 = std::clamp((boxes[i * 4 + 2] - pad_x) / scale, 0.0f, static_cast<float>(frame.image.cols));
        const float y1 = std::clamp((boxes[i * 4 + 3] - pad_y) / scale, 0.0f, static_cast<float>(frame.image.rows));
        frame.detections.push_back({labels[i], scores[i], cv::Rect2f(x0, y0, x1 - x0, y1 - y0)});
    }
}

void TrtWorker::process(Frame& frame) {
    float scale = 1.0f;
    float pad_x = 0.0f;
    float pad_y = 0.0f;
    preprocess(frame.image, scale, pad_x, pad_y);
    check_cuda(cudaStreamSynchronize(stream_), "CUDA preprocessing");

    const auto start = std::chrono::steady_clock::now();
    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("TensorRT enqueueV3 failed");
    }
    for (auto& item : buffers_) {
        if (!item.input) {
            check_cuda(cudaMemcpyAsync(item.host, item.device, item.bytes, cudaMemcpyDeviceToHost, stream_), "cudaMemcpyAsync");
        }
    }
    check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    const auto end = std::chrono::steady_clock::now();
    frame.inference_ms = std::chrono::duration<double, std::milli>(end - start).count();
    postprocess(frame, scale, pad_x, pad_y);

    for (const auto& detection : frame.detections) {
        const auto color = color_for(detection.label);
        cv::rectangle(frame.image, detection.box, color, 2);
        const std::string name = detection.label >= 0 && static_cast<std::size_t>(detection.label) < kCocoNames.size()
            ? kCocoNames[static_cast<std::size_t>(detection.label)]
            : "class_" + std::to_string(detection.label);
        const std::string text = name + " " + cv::format("%.1f%%", detection.score * 100.0f);
        cv::putText(frame.image, text, detection.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
}

}  // namespace yolo
