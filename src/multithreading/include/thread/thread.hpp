#pragma once

#include "../common.hpp"
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <queue>
#include <vector>
#include "atomic"

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

template <typename T>
class SPSCqueue {
private:
    size_t capacity_;
    size_t max_elements;
    size_t mask_;
    std::vector<T> buffer_;

    // 生产者独占
    alignas(64) std::atomic<size_t> head_{0};
    size_t cached_tail_{0};

    // 消费者独占
    alignas(64) std::atomic<size_t> tail_{0};
    size_t cached_head_{0};

public:
    explicit SPSCqueue(size_t capacity, size_t max_elements) {
        assert(capacity > 0 && (capacity & (capacity - 1)) == 0);
        assert(max_elements < capacity);
        capacity_ = capacity;
        mask_ = capacity - 1;
        buffer_.resize(capacity_);
    }
    
    // producer
    bool push(const T& item) {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        size_t current_size = (current_head - cached_tail_) & mask_;

        if (current_size >= max_elements) {
            cached_tail_ = tail_.load(std::memory_order_acquire);
            current_size = (current_head - cached_tail_) & mask_;
            if (current_size >= max_elements) {
                return false;
            }
        }
        buffer_[current_head] = item;

        const size_t next_head = (current_head + 1) & mask_;
        head_.store(next_head, std::memory_order_release);
        return true;
    }

    // consumer
    bool pop(T& item) {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);

        if (current_tail == cached_head_) {
            cached_head_ = head_.load(std::memory_order_acquire);
            if (current_tail == cached_head_) {
                return false;
            }
        }
        item = buffer_[current_tail];
        tail_.store((current_tail + 1) & mask_, std::memory_order_release);
        return true;
    }
};