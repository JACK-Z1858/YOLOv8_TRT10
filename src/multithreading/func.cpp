#include "include/thread/thread.hpp"
#include "include/func.hpp"
#include <chrono>
#include <thread>
#include <iostream>

void Producer(threadSafeQueue &p2c) {
    for (uint32_t i = 0; i < 30; ++i) {
        FrameData frame = {i};
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        p2c.push(frame);
    }
    p2c.stopPush();
}

void Consumer(threadSafeQueue &p2c) {
    FrameData frame;
    while(p2c.pull(frame)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    std::cout << p2c.queue_name_ << ": Consumer shutdown" << std::endl;
}