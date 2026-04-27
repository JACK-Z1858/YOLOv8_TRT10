#include "include/yolov8.hpp"
#include "include/thread/thread.hpp"
#include <cstdint>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

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
    while(cap.read(img)) {
        FrameData frame;
        frame.frame_id = frameCount++;
        frame.img      = img;
        if(!read2work.push(frame)) {
            break;
        }
    }
    read2work.stopPush();
}

void YOLOv8::Worker(threadSafeQueue& read2work, threadSafeQueue& work2out) {
    FrameData frame;
    while (read2work.pull(frame)) {
        preProcess();
        infer();    // 推理失败则标记failed输出原图，保证每个id都会出现一次
        postProcess();
        if(!work2out.push(frame)) {
            break;
        }
    }
    work2out.stopPush();
}

void YOLOv8::Outputer(threadSafeQueue& work2out, threadSafeQueue& out2show) {
    FrameData frame;
    while (work2out.pull(frame)) {
        drawObjects();
        if (!reorderBuffer(frame, out2show)) {
            break;
        }
    }
    out2show.stopPush();
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

YOLOv8::YOLOv8() {

}

YOLOv8::~YOLOv8() {}

void YOLOv8::preProcess() {}

void YOLOv8::infer() {}

void YOLOv8::postProcess() {}

void YOLOv8::drawObjects() {}