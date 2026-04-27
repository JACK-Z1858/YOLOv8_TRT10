#include "include/thread/thread.hpp"
#include <mutex>
#include <iostream>
#include <ostream>

threadSafeQueue::threadSafeQueue(const std::string& name, const uint32_t max_size) 
    : queue_name_(name), max_size_(max_size) {
}

threadSafeQueue::~threadSafeQueue() {}

bool threadSafeQueue::push(const FrameData &frame) {
    // lock the shared data
    std::unique_lock<std::mutex> lock(mutex_);
    
    not_full_.wait(lock, [this]() {
        return queue_.size() < max_size_ || stopped_push_ || shutdowned_;
    });

    if (stopped_push_) {
        return false;
    }

    queue_.push(std::move(frame));
    lock.unlock();

    not_empty_.notify_one();
    std::cout<< queue_name_ << ": Pushed: " << frame.frame_id << std::endl; 
    return true;
}

bool threadSafeQueue::pull(FrameData &frame) {
    std::unique_lock<std::mutex> lock(mutex_);

    not_empty_.wait(lock, [this]() {
        return queue_.size() > 0 || stopped_push_ || shutdowned_;
    });

    if ((queue_.empty() && stopped_push_) || shutdowned_) {
        return false;   // pull finished, shutdown successfully
    }

    frame = std::move(queue_.front());
    queue_.pop();
    lock.unlock();

    not_full_.notify_one();
    std::cout << queue_name_ << ": Pulled: " << frame.frame_id << std::endl;
    return true;
}


bool threadSafeQueue::stoppedPush() const{
    std::lock_guard<std::mutex> lock(mutex_);
    return stopped_push_;
}

bool threadSafeQueue::shutdowned() const{
    std::lock_guard<std::mutex> lock(mutex_);
    return shutdowned_;
}

void threadSafeQueue::stopPush() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (stopped_push_) {
        lock.unlock();
        std::cout << queue_name_ << " already stopped pushing." << std::endl;
    } else {
        stopped_push_ = true;
        lock.unlock();
        not_empty_.notify_all();
        not_full_.notify_all();
        std::cout << queue_name_ << ": stop pushing" << std::endl;
    }
}

void threadSafeQueue::shutdown() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (shutdowned_) {
        lock.unlock();
        std::cout << queue_name_ << " already shutdowned." << std::endl;
    } else {
        stopped_push_ = true;
        shutdowned_ = true;
        
        std::queue<FrameData> empty_queue;
        std::swap(queue_, empty_queue);
        lock.unlock();
        not_full_.notify_all();
        not_empty_.notify_all();
        std::cout << queue_name_ << " is shutdowned" << std::endl;
    }
}