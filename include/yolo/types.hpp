// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <chrono>
#include <cstdint>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <vector>

namespace yolo {

struct Detection {
    int label{};
    float score{};
    cv::Rect2f box;
};

struct Frame {
    std::uint64_t id{};
    cv::Mat image;
    std::vector<Detection> detections;
    std::chrono::steady_clock::time_point enqueued_at;
    double inference_ms{};
    double end_to_end_ms{};
};

}  // namespace yolo
