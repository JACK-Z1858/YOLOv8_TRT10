// SPDX-License-Identifier: AGPL-3.0-only

#include "yolo/config.hpp"

#include <opencv2/core/persistence.hpp>
#include <stdexcept>

namespace yolo {
namespace {

template <typename T>
void read_if_present(const cv::FileNode& parent, const char* key, T& value) {
    const auto node = parent[key];
    if (!node.empty()) {
        node >> value;
    }
}

}  // namespace

Config load_config(const std::string& path) {
    cv::FileStorage file(path, cv::FileStorage::READ);
    if (!file.isOpened()) {
        throw std::runtime_error("cannot open config file: " + path);
    }

    Config config;
    const auto model = file["model"];
    const auto input = file["input"];
    const auto pipeline = file["pipeline"];
    const auto output = file["output"];
    const auto benchmark = file["benchmark"];

    read_if_present(model, "engine", config.engine_path);
    read_if_present(model, "device", config.device);
    read_if_present(input, "source", config.source);

    int workers = static_cast<int>(config.workers);
    int queue_depth = static_cast<int>(config.queue_depth);
    read_if_present(pipeline, "workers", workers);
    read_if_present(pipeline, "queue_depth", queue_depth);
    if (workers <= 0 || queue_depth <= 0) {
        throw std::runtime_error("workers and queue_depth must be greater than zero");
    }
    config.workers = static_cast<std::size_t>(workers);
    config.queue_depth = static_cast<std::size_t>(queue_depth);

    read_if_present(output, "show", config.show);
    read_if_present(output, "video", config.output_video);
    read_if_present(output, "fps", config.output_fps);

    double warmup_frames = static_cast<double>(config.warmup_frames);
    double max_frames = static_cast<double>(config.max_frames);
    read_if_present(benchmark, "warmup_frames", warmup_frames);
    read_if_present(benchmark, "max_frames", max_frames);
    if (warmup_frames < 0 || max_frames < 0) {
        throw std::runtime_error("frame counts cannot be negative");
    }
    config.warmup_frames = static_cast<std::uint64_t>(warmup_frames);
    config.max_frames = static_cast<std::uint64_t>(max_frames);

    if (config.engine_path.empty() || config.source.empty()) {
        throw std::runtime_error("model.engine and input.source are required");
    }
    if (config.output_fps <= 0) {
        throw std::runtime_error("output.fps must be greater than zero");
    }
    return config;
}

}  // namespace yolo
