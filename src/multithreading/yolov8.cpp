#include "include/yolov8.hpp"
#include "include/pinnedMemoryPool.hpp"
#include <NvInferPlugin.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/cudaarithm.hpp>
#include <ratio>
#include <stdexcept>
#include <fstream>
#include <utility>


YOLOv8::YOLOv8(const std::string& engine_path, 
               const std::string& video_path, 
               const uint32_t POOL_SIZE, 
               const size_t max_img_size)
                : input_video_path_(video_path), mem_pool_(POOL_SIZE, max_img_size)
{
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
            binding.dims = dims;
            binding.size = dims2size(dims);
            i_bindings_.push_back(binding);
        } else {
            ++num_outputs_;
            
            o_bindings_.push_back(binding);
        }
    }
}

YOLOv8::~YOLOv8() {
    
}


void YOLOv8::VideoReader(threadSafeQueue& read2work) {
    cv::VideoCapture    cap;
    
    try {
        cap.open(std::stoi(input_video_path_));
    } catch(const std::exception &e) {
        cap.open(input_video_path_);
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
        throw std::runtime_error("Uable to open video capture with input: " + input_video_path_);
    }
    
    uint32_t frameCount{0};
    cv::Mat  temp_img;

    while(true) {
        auto start = std::chrono::system_clock::now();
        if(!cap.read(temp_img)) {
            break;
        }
        void* ptr = mem_pool_.acquire();
        cv::Mat pinned_mat(temp_img.rows, temp_img.cols, temp_img.type(), ptr);
        temp_img.copyTo(pinned_mat);

        FrameData frame;
        frame.frame_id   = frameCount++;
        frame.img        = pinned_mat;
        frame.pinned_ptr = ptr;
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

std::vector<WorkContext> YOLOv8::initWorkContext(uint32_t nbSlots) {
    std::vector<WorkContext> slots;
    cudaStream_t init_stream;
    cudaStreamCreate(&init_stream);

    for (int i = 0; i < nbSlots; ++i) {
        WorkContext wc;
        wc.context.reset(engine_->createExecutionContext());
        
        for (auto& binding : i_bindings_) {
            void* d_ptr;
            CHECK(cudaMallocAsync(&d_ptr, binding.size * binding.dsize, init_stream));
            wc.device_ptrs.push_back(d_ptr);
            auto name = binding.name.c_str();
            wc.context->setInputShape(name, binding.dims);
            wc.context->setTensorAddress(name, d_ptr);
        }

        for (auto& binding : o_bindings_) {
            nvinfer1::Dims dims = wc.context->getTensorShape(binding.name.c_str());
            size_t osize = dims2size(dims) * binding.dsize;
            wc.o_sizes.push_back(osize);

            void *d_ptr, *h_ptr;
            CHECK(cudaMallocAsync(&d_ptr, osize, init_stream));
            CHECK(cudaHostAlloc(&h_ptr, osize, 0));
            wc.device_ptrs.push_back(d_ptr);
            wc.host_ptrs.push_back(h_ptr);
            auto name = binding.name.c_str();
            wc.context->setTensorAddress(name, d_ptr);
        }
        slots.push_back(std::move(wc));
    }
    cudaStreamSynchronize(init_stream);
    cudaStreamDestroy(init_stream);
    return slots;
}

void YOLOv8::runWorker(threadSafeQueue& read2work, threadSafeQueue& work2out) {
    uint32_t nbSlots = 1;
    std::vector<WorkContext> wc = initWorkContext(nbSlots);
    
    std::vector<cudaStream_t> stream_copy(nbSlots); 
    std::vector<cudaStream_t> stream_infer(nbSlots);
    
    std::vector<cv::cuda::Stream> cv_copy(nbSlots);  
    std::vector<cudaEvent_t> ev_prep(nbSlots);
    std::vector<cudaEvent_t> ev_done(nbSlots);
    for (int i = 0; i < nbSlots; ++i) {
        cudaStreamCreate(&stream_copy[i]);
        cudaStreamCreate(&stream_infer[i]);
        cudaEventCreate(&ev_prep[i]);
        cudaEventCreate(&ev_done[i]);
        cv_copy[i] = cv::cuda::StreamAccessor::wrapStream(stream_copy[i]);
    }

    std::deque<InFlight> q;
    InFlight oldest;
    FrameData frame;
    while (read2work.pull(frame)) {
        if (q.size() == 1) {
            oldest = q.front();
            cudaEventSynchronize(ev_done[oldest.slot]);
            oldest.frame.obj_ptr.clear();
            for (auto ptr : wc[oldest.slot].host_ptrs) {
                oldest.frame.obj_ptr.push_back(ptr);
            }

            postProcess(oldest.frame);
            if(!work2out.push(oldest.frame)) {
                break;
            }
            q.pop_front();
        }
        uint32_t s = frame.frame_id & (nbSlots - 1);
        wc[s].gpuMat.create(frame.img.rows, frame.img.cols, frame.img.type());   
        wc[s].gpuMat.upload(frame.img, cv_copy[s]);
        preProcess(wc[s], frame, cv_copy[s]);
        cudaEventRecord(ev_prep[s], stream_copy[s]);

        cudaStreamWaitEvent(stream_infer[s], ev_prep[s]);
        infer(wc[s], stream_infer[s]);    // 推理失败则标记failed输出原图，保证每个id都会出现一次
        cudaEventRecord(ev_done[s], stream_infer[s]);
        // 如何保证数据从 host 拷贝出了
        q.push_back({s, std::move(frame)});
    }

    while (!q.empty()) {
        oldest = q.front();
        cudaEventSynchronize(ev_done[oldest.slot]);
        oldest.frame.obj_ptr.clear();
        for (auto ptr : wc[oldest.slot].host_ptrs) {
            oldest.frame.obj_ptr.push_back(ptr);
        }

        postProcess(oldest.frame);
        if(!work2out.push(oldest.frame)) {
            break;
        }
        q.pop_front();
    }

    for (int i = 0; i < nbSlots; ++i) {
        cudaStreamDestroy(stream_copy[i]);
        cudaStreamDestroy(stream_infer[i]);
        cudaEventDestroy(ev_prep[i]);
        cudaEventDestroy(ev_done[i]);
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
        /*
        if (!reorderSkipBuffer(frame, out2show)) {
            break;
        }
        */
        if (!out2show.push(frame)) {
            break;
        }
    }
    if (!out2show.isStopPush()) {
        out2show.stopPush();
    }
}

bool YOLOv8::reorderSkipBuffer(const FrameData& frame, threadSafeQueue& out2show) {
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

void YOLOv8::preProcess(WorkContext& wc, FrameData& frame, cv::cuda::Stream cv_copy) {
    const auto& dim = i_bindings_[0].dims;
    const float inp_h = dim.d[2];
    const float inp_w = dim.d[3];
    frame.height  = wc.gpuMat.rows;
    frame.width   = wc.gpuMat.cols;
    float r  = std::min(inp_h / frame.height, inp_w / frame.width);
    frame.ratio   = 1 / r;
    
    resizeKeepAspectRatioPadRightBottom(wc, inp_h, inp_w, r, frame.height, frame.width, cv_copy);
    D2DBlobFromPadded(wc, cv_copy);
}

void YOLOv8::resizeKeepAspectRatioPadRightBottom(WorkContext&     wc, 
                                                 const float      inp_h,
                                                 const float      inp_W,
                                                 const float      r,
                                                 const float      h,
                                                 const float      w,
                                                 cv::cuda::Stream cv_copy) 
{
    int unpad_h = r * h;
    int unpad_w = r * w;
    wc.resized.create(unpad_h, unpad_w, CV_8UC3);
    cv::cuda::resize(wc.gpuMat, wc.resized, wc.resized.size(), 0, 0, cv::INTER_LINEAR, cv_copy);

    wc.padded.create((int)inp_h, (int)inp_W, CV_8UC3);
    wc.padded.setTo(cv::Scalar(114, 114, 114), cv::noArray(), cv_copy);
    wc.resized.copyTo(wc.padded(cv::Rect(0, 0, unpad_w, unpad_h)), cv_copy);
}

void YOLOv8::D2DBlobFromPadded(WorkContext& wc, cv::cuda::Stream cv_copy) {
    const int h = wc.padded.rows;
    const int w = wc.padded.cols;

    for (auto& channel : wc.chw_u8) {
        channel.create(wc.padded.rows, wc.padded.cols, CV_8UC1);
    }
    cv::cuda::split(wc.padded, wc.chw_u8, cv_copy);

    float* base = reinterpret_cast<float*>(wc.device_ptrs[0]);
    cv::cuda::GpuMat c0(h, w, CV_32F, base + 0 * h * w);
    cv::cuda::GpuMat c1(h, w, CV_32F, base + 1 * h * w);
    cv::cuda::GpuMat c2(h, w, CV_32F, base + 2 * h * w);

    // BGR -> RGB
    wc.chw_u8[2].convertTo(c0, CV_32F, 1.f / 255.f, cv_copy);
    wc.chw_u8[1].convertTo(c1, CV_32F, 1.f / 255.f, cv_copy);
    wc.chw_u8[0].convertTo(c2, CV_32F, 1.f / 255.f, cv_copy);
}

void YOLOv8::infer(const WorkContext& wc, cudaStream_t stream_infer) {
    wc.context->enqueueV3(stream_infer);

    for (int i = 0; i < num_outputs_; ++i) {
        size_t osize = wc.o_sizes[i];
        CHECK(cudaMemcpyAsync(
            wc.host_ptrs[i], wc.device_ptrs[i + num_inputs_], osize, cudaMemcpyDeviceToHost, stream_infer
        ));
    }
}

void YOLOv8::postProcess(FrameData& frame) {
    Object obj;
    auto* num_dets = static_cast<int*>(frame.obj_ptr[0]);
    auto* boxes  = static_cast<float*>(frame.obj_ptr[1]);
    auto* scores = static_cast<float*>(frame.obj_ptr[2]);
    auto* labels   = static_cast<int*>(frame.obj_ptr[3]);
    
    for (int i = 0; i < num_dets[0]; ++i) {
        float* ptr = boxes + i * 4;
        float x0 = *ptr++;
        float y0 = *ptr++;
        float x1 = *ptr++;
        float y1 = *ptr;
        
        x0 = clamp(x0 * frame.ratio, 0.f, frame.width);
        x1 = clamp(x1 * frame.ratio, 0.f, frame.width);
        y0 = clamp(y0 * frame.ratio, 0.f, frame.height);
        y1 = clamp(y1 * frame.ratio, 0.f, frame.height);

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

void YOLOv8::releasePinnedPtr(void* ptr) {
    if (ptr != nullptr) {
        mem_pool_.release(ptr);
    }
}