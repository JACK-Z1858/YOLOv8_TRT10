#include "include/thread/thread.hpp"

threadSafeQueue::threadSafeQueue(const std::string& name, const uint32_t max_size) 
    : queue_name_(name), max_size_(max_size) {
}

threadSafeQueue::~threadSafeQueue() {}

bool threadSafeQueue::push(const FrameData &frame) {
    // lock the shared data
    std::unique_lock<std::mutex> lock(mutex_);
    
    not_full_.wait(lock, [this]() {
        return queue_.size() < max_size_ || is_stop_push_ || is_shutdown_;
    });

    if (is_stop_push_) {
        return false;
    }

    queue_.push(std::move(frame));
    lock.unlock();

    not_empty_.notify_one();
    // std::cout<< queue_name_ << ": Pushed: " << frame.frame_id << std::endl; 
    return true;
}


bool threadSafeQueue::pull(FrameData &frame) {
    std::unique_lock<std::mutex> lock(mutex_);

    not_empty_.wait(lock, [this]() {
        return queue_.size() > 0 || is_stop_push_ || is_shutdown_;
    });

    if ((queue_.empty() && is_stop_push_) || is_shutdown_) {
        return false;   // pull finished, shutdown successfully
    }

    frame = std::move(queue_.front());
    queue_.pop();
    lock.unlock();

    not_full_.notify_one();
    // std::cout << queue_name_ << ": Pulled: " << frame.frame_id << std::endl;
    return true;
}


bool threadSafeQueue::isStopPush() const{
    std::lock_guard<std::mutex> lock(mutex_);
    return is_stop_push_;
}

bool threadSafeQueue::isShutdown() const{
    std::lock_guard<std::mutex> lock(mutex_);
    return is_shutdown_;
}

void threadSafeQueue::stopPush() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (is_stop_push_) {
        lock.unlock();
        // std::cout << queue_name_ << " already stopped pushing." << std::endl;
    } else {
        is_stop_push_ = true;
        lock.unlock();
        not_empty_.notify_all();
        not_full_.notify_all();
        // std::cout << queue_name_ << ": stop pushing" << std::endl;
    }
}

void threadSafeQueue::shutdown() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (is_shutdown_) {
        lock.unlock();
        // std::cout << queue_name_ << " already shutdowned." << std::endl;
    } else {
        is_stop_push_ = true;
        is_shutdown_ = true;
        
        std::queue<FrameData> empty_queue;
        std::swap(queue_, empty_queue);
        lock.unlock();
        not_full_.notify_all();
        not_empty_.notify_all();
        // std::cout << queue_name_ << " is shutdowned" << std::endl;
    }
}