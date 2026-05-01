#include "include/yolov8.hpp"
#include "include/common.hpp"
#include <NvInferPlugin.h>
#include <chrono>
#include <cstddef>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/cudaarithm.hpp>
#include <ratio>
#include <stdexcept>
#include <fstream>

YOLOv8::YOLOv8(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> trtModelStream(size);
    if (!file.read(trtModelStream.data(), size)) {
        throw std::runtime_error("Unable to read engine file.");
    }
    file.close();

    initLibNvInferPlugins(&gLogger_, "");
    runtime_ = std::unique_ptr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(gLogger_)};
    if (!runtime_) {
        throw std::runtime_error("Failed to create runtime.");
    }

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>{runtime_->deserializeCudaEngine(trtModelStream.data(), size)};
    if (!engine_) {
        throw std::runtime_error("Failed to deserialize CUDA engine.");
    }

    context_ = std::unique_ptr<nvinfer1::IExecutionContext>{engine_->createExecutionContext()};
    if (!context_) {
        throw std::runtime_error("Failed to create execution context.");
    }

    cudaStreamCreate(&stream_);

    num_bindings_ = engine_->getNbIOTensors();
    for (int i = 0; i < num_bindings_; ++i) {
        Binding        binding;
        nvinfer1::Dims dims;
        
        std::string        name  = engine_->getIOTensorName(i);
        nvinfer1::DataType dtype = engine_->getTensorDataType(name.c_str());
        binding.name = name;
        binding.dsize = type2dsize(dtype);

        bool isInput = engine_->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
        if (isInput) {
            ++num_inputs_;
            dims = engine_->getProfileShape(name.c_str(), i, nvinfer1::OptProfileSelector::kMAX);
            context_->setInputShape(name.c_str(), dims);
            binding.dims = dims;
            binding.size = dims2size(dims);
            i_bindings_.push_back(binding);
        } else {
            ++num_outputs_;
            dims = context_->getTensorShape(name.c_str());
            binding.dims = dims;
            binding.size = dims2size(dims);
            o_bindings_.push_back(binding);
        }
    }
}

YOLOv8::~YOLOv8() {
    cudaStreamDestroy(stream_);

    for (auto& ptr : device_ptrs_) {
        CHECK(cudaFree(ptr));
    }
    for (auto& ptr : host_ptrs_) {
        CHECK(cudaFreeHost(ptr));
    }
}

void YOLOv8::makepipe() {
    for (auto& binding : i_bindings_) {
        void *d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, binding.size * binding.dsize, stream_));
        device_ptrs_.push_back(d_ptr);
        
        auto name = binding.name.c_str();
        context_->setTensorAddress(name, d_ptr);
    }

    for (auto& binding : o_bindings_) {
        void *d_ptr, *h_ptr;
        CHECK(cudaMallocAsync(&d_ptr, binding.size * binding.dsize, stream_));
        CHECK(cudaHostAlloc(&h_ptr, binding.size * binding.dsize, 0));
        device_ptrs_.push_back(d_ptr);
        host_ptrs_.push_back(h_ptr);

        auto name = binding.name.c_str();
        context_->setTensorAddress(name, d_ptr);
    }
}

void YOLOv8::VideoReader(std::string inputVideo, threadSafeQueue& read2work) {
    cv::VideoCapture    cap;
    
    try {
        cap.open(std::stoi(inputVideo));
    } catch(const std::exception &e) {
        cap.open(inputVideo);
    }

   // Try to use HD resolution (or closest resolution)
    auto resW = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    auto resH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "Original video resolution: (" << resW << "x" << resH << ")" << std::endl;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    resW = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    resH = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "New video resolution: (" << resW << "x" << resH << ")" << std::endl; 

    if (!cap.isOpened()) {
        throw std::runtime_error("Uable to open video capture with input: " + inputVideo);
    }
    
    uint32_t frameCount{0};
    cv::Mat  img;

    while(true) {
        auto start = std::chrono::system_clock::now();
        ;
        if(!cap.read(img)) break;
        FrameData frame;
        frame.frame_id = frameCount++;
        frame.img      = img.clone();
        auto end = std::chrono::system_clock::now();
        frame.prof.t_read = (double)std::chrono::duration<double, std::milli>(end - start).count();
        frame.t_enqueue = std::chrono::steady_clock::now();
        if(!read2work.push(frame)) {
            break;
        }
    }
    if (!read2work.isStopPush()) {
        read2work.stopPush();
    }
}

void YOLOv8::Worker(threadSafeQueue& read2work, threadSafeQueue& work2out) {
    FrameData frame;
    while (read2work.pull(frame)) {   
        cv::cuda::GpuMat gpuMat(frame.img);

        auto start = std::chrono::steady_clock::now();
        preProcess(gpuMat);
        auto end = std::chrono::steady_clock::now();
        frame.prof.t_pre = (double)std::chrono::duration<double, std::milli>(end - start).count();

        start = std::chrono::steady_clock::now();
        cv::cuda::GpuMat blob = blobFromGpuMat(gpuMat);
        size_t bytes = static_cast<size_t>(blob.rows) * blob.step;
        CHECK(cudaMemcpyAsync(
            device_ptrs_[0], blob.cudaPtr(), bytes, cudaMemcpyDeviceToDevice, stream_
        ));
        infer();    // 推理失败则标记failed输出原图，保证每个id都会出现一次
        end = std::chrono::steady_clock::now();
        frame.prof.t_infer = (double)std::chrono::duration<double, std::milli>(end - start).count();
        
        start = std::chrono::steady_clock::now();
        postProcess(frame);
        end = std::chrono::steady_clock::now();
        frame.prof.t_post = (double)std::chrono::duration<double, std::milli>(end - start).count();

        if(!work2out.push(frame)) {
            break;
        }
        
        
    }
    if (!work2out.isStopPush()) {
        work2out.stopPush();
    }
}

void YOLOv8::Outputer(threadSafeQueue& work2out, threadSafeQueue& out2show) {
    FrameData frame;
    while (work2out.pull(frame)) {
        auto start = std::chrono::steady_clock::now();
        drawObjects(frame);
        auto end = std::chrono::steady_clock::now();
        frame.prof.t_draw = (double)std::chrono::duration<double, std::milli>(end - start).count();
        if (!reorderBuffer(frame, out2show)) {
            break;
        }
        
        
    }
    if (!out2show.isStopPush()) {
        out2show.stopPush();
    }
}

bool YOLOv8::reorderBuffer(const FrameData& frame, threadSafeQueue& out2show) {
    if (frame.frame_id == expected_id) {
        if (!out2show.push(frame)) {
            return false;
        } else {
            ++expected_id;
        }         
    } else {
        buffer.emplace(frame);
    }

    if (buffer.size() > max_buffer_size) {
        expected_id = buffer.top().frame_id;
    }

    while (!buffer.empty() && expected_id == buffer.top().frame_id) {
            if (!out2show.push(buffer.top())) {
                return false;
            } else {
                buffer.pop();
                ++expected_id;
            }    
        }
    return true;
}

void YOLOv8::preProcess(cv::cuda::GpuMat& gpuMat) {
    const auto& dim = i_bindings_[0].dims;
    const float inp_h = dim.d[2];
    const float inp_w = dim.d[3];
    height_  = gpuMat.rows;
    width_   = gpuMat.cols;
    float r  = std::min(inp_h / height_, inp_w / width_);
    ratio_   = 1 / r;
    
    cv::cuda::GpuMat rgbMat;
    cv::cuda::cvtColor(gpuMat, rgbMat, cv::COLOR_BGR2RGB);

    cv::cuda::GpuMat resized = rgbMat.clone();
    if (height_ != inp_h || width_ != inp_w) {
        resized = resizeKeepAspectRatioPadRightBottom(rgbMat, inp_h, inp_w, r);
    }
    gpuMat = resized;
}

cv::cuda::GpuMat YOLOv8::resizeKeepAspectRatioPadRightBottom(cv::cuda::GpuMat& input, 
                                                             const float       inp_h,
                                                             const float       inp_W,
                                                             const float       r) 
{
    int unpad_h = r * input.rows;
    int unpad_w = r * input.cols;
    cv::cuda::GpuMat re(unpad_h, unpad_w, CV_8UC3);
    cv::cuda::resize(input, re, re.size());
    cv::cuda::GpuMat out((int)inp_h, (int)inp_W, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, unpad_w, unpad_h)));
    return out;
}

cv::cuda::GpuMat YOLOv8::blobFromGpuMat(cv::cuda::GpuMat& input) {
    const int h = input.rows;
    const int w = input.cols;

    std::vector<cv::cuda::GpuMat> chw_u8(3);
    cv::cuda::split(input, chw_u8);

    cv::cuda::GpuMat blob(1, 3 * h * w, CV_32F);
    float* base = reinterpret_cast<float*>(blob.ptr<float>());
    cv::cuda::GpuMat c0(h, w, CV_32F, base + 0 * h * w);
    cv::cuda::GpuMat c1(h, w, CV_32F, base + 1 * h * w);
    cv::cuda::GpuMat c2(h, w, CV_32F, base + 2 * h * w);

    chw_u8[0].convertTo(c0, CV_32F, 1.f / 255.f);
    chw_u8[1].convertTo(c1, CV_32F, 1.f / 255.f);
    chw_u8[2].convertTo(c2, CV_32F, 1.f / 255.f);
    return blob;
}

void YOLOv8::infer() {
    context_->enqueueV3(stream_);

    for (int i = 0; i < num_outputs_; ++i) {
        size_t osize = o_bindings_[i].size * o_bindings_[i].dsize;
        CHECK(cudaMemcpyAsync(
            host_ptrs_[i], device_ptrs_[i + num_inputs_], osize, cudaMemcpyDeviceToHost, stream_
        ));
    }
    cudaStreamSynchronize(stream_);
}

void YOLOv8::postProcess(FrameData& frame) {
    Object obj;
    auto* num_dets = static_cast<int*>(host_ptrs_[0]);
    auto* boxes  = static_cast<float*>(host_ptrs_[1]);
    auto* scores = static_cast<float*>(host_ptrs_[2]);
    auto* labels   = static_cast<int*>(host_ptrs_[3]);
    
    for (int i = 0; i < num_dets[0]; ++i) {
        float* ptr = boxes + i * 4;
        float x0 = *ptr++;
        float y0 = *ptr++;
        float x1 = *ptr++;
        float y1 = *ptr;
        
        x0 = clamp(x0 * ratio_, 0.f, width_);
        x1 = clamp(x1 * ratio_, 0.f, width_);
        y0 = clamp(y0 * ratio_, 0.f, height_);
        y1 = clamp(y1 * ratio_, 0.f, height_);

        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.prob = *(scores + i);
        obj.label = *(labels + i);
        frame.objects.push_back(obj);
    }
}

void YOLOv8::drawObjects(FrameData& frame) {
    cv::Mat res = frame.img;
    for (auto& obj : frame.objects) {
        cv::Scalar color = cv::Scalar(COLORS[obj.label][0], COLORS[obj.label][1], COLORS[obj.label][2]);
        cv::rectangle(res, obj.rect, color, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[obj.label].c_str(), obj.prob * 100);

        int      baseLine    = 0;
        cv::Size label_size  = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y + 1;

        if (y > res.rows) {
            y = res.rows;
        }

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);

        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
}