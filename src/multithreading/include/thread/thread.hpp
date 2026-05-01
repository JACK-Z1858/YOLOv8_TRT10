#pragma once

#include "../common.hpp"
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <queue>

struct queueStats {
    uint32_t push_count{0};
    uint32_t pull_count{0};
    double push_wait_sum{0.0};
    double pull_wait_sum{0.0};
    double push_wait_max{0.0};
    double pull_wait_max{0.0};
};

class threadSafeQueue {
private:
    
    std::queue<FrameData>   queue_;
    const uint32_t          max_size_;
    const std::string       queue_name_;

    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    mutable std::mutex              mutex_;

    bool                    is_stop_push_ = false;
    bool                    is_shutdown_ = false;

public:
    threadSafeQueue(const std::string& name, const uint32_t max_size = 4);
    
    ~threadSafeQueue();

    // if need push ? push : stop
    bool push(const FrameData &frame);

    // if need pull ? pull : stop
    bool pull(FrameData &frame);

    void stopPush();

    void shutdown();

    bool isStopPush () const;

    bool isShutdown() const;

    queueStats stats;
};

inline auto cmpFrameData = [](FrameData left, FrameData right) {
    return left.frame_id > right.frame_id;
};

