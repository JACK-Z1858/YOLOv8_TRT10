#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <string>
// #include <opencv2/core/mat.hpp>
#include <queue>

struct FrameData {
    uint32_t frame_id{};
    cv::Mat  img; 
};

class threadSafeQueue {
private:
    
    std::queue<FrameData>   queue_;
    const uint32_t          max_size_;
    const std::string       queue_name_;

    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    mutable std::mutex              mutex_;

    bool                    stopped_push_ = false;
    bool                    shutdowned_ = false;

public:
    threadSafeQueue(const std::string& name, const uint32_t max_size = 4);
    
    ~threadSafeQueue();

    // if need push ? push : stop
    bool push(const FrameData &frame);

    // if need pull ? pull : stop
    bool pull(FrameData &frame);

    void stopPush();

    void shutdown();

    bool stoppedPush () const;

    bool shutdowned() const;
};

inline auto cmpFrameData = [](FrameData left, FrameData right) {
    return left.frame_id > right.frame_id;
};