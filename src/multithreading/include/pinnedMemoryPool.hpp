#pragma once
#include "common.hpp"
#include <mutex>
#include <queue>
#include <condition_variable>
#include <string>
#include "opencv2/videoio.hpp"

inline const size_t getMaxFrameBytes(const std::string& input_video_path) {
     // Initialize pinned memory pool
     cv::VideoCapture cap(input_video_path);
     if (!cap.isOpened()) {
         throw std::runtime_error("Failed to open video: " + input_video_path);
     }
     cv::Mat first_frame;
     cap >> first_frame;
     if (first_frame.empty()) {
         throw std::runtime_error("Failed to read video: " + input_video_path);
     }
     size_t max_img_size = first_frame.total() * first_frame.elemSize();
     std::cout << "Image resolution: " << first_frame.cols << "x" << first_frame.rows
               << ", Required memory per frame: " << max_img_size << "bytes." << std::endl;
     return max_img_size;
}

class PinnedMemoryPool {
private:
    std::queue<void*>       free_queue_;    // 存放当前可用指针
    std::vector<void*>      all_allocated_; // 记录所有分配的指针
    std::mutex              mtx_;
    std::condition_variable not_empty_;
    size_t                  buffer_size_;   // 每块内存的大小

public:
    PinnedMemoryPool(int pool_size, size_t max_img_size) : buffer_size_(max_img_size) {
        for (int i = 0; i < pool_size; ++i) {
            void* ptr = nullptr;
            CHECK(cudaMallocHost(&ptr, buffer_size_));
            free_queue_.push(ptr);
            all_allocated_.push_back(ptr);
        }
    }

    ~PinnedMemoryPool() {
        for (void* ptr : all_allocated_) {
            if (ptr) {
                cudaFreeHost(ptr);
            }
        }
    }

    void* acquire() {
        std::unique_lock<std::mutex> lock(mtx_);
        not_empty_.wait(lock, [this]() {
            return !free_queue_.empty();
        });
        void* ptr = free_queue_.front();
        free_queue_.pop();
        return ptr;
    }

    void release(void* ptr){
        std::lock_guard<std::mutex> lock(mtx_);
        free_queue_.push(ptr);
        not_empty_.notify_one();
    }

    size_t getBufferSize() const {
        return buffer_size_;
    }
};