#pragma once

#include "thread/thread.hpp"
#include "opencv2/core/types.hpp"
#include <cstdint>

struct Object {
    int              label{};
    float            prob{};
    cv::Rect_<float> rect;
};

class YOLOv8 {
private:
    std::priority_queue<FrameData, std::vector<FrameData>, decltype(cmpFrameData)> buffer{cmpFrameData};

    uint32_t expected_id{0};
    uint32_t max_buffer_size{5};

private:
    void preProcess();

    void postProcess();

    void infer();

    void drawObjects();

    bool reorderBuffer(const FrameData& frame, threadSafeQueue& out2show);

public:
    YOLOv8();

    ~YOLOv8();

    void VideoReader(std::string inputVideo, threadSafeQueue& read2work);

    void Worker(threadSafeQueue&read2work, threadSafeQueue& work2out);

    void Outputer(threadSafeQueue& work2out, threadSafeQueue& out2show);
};