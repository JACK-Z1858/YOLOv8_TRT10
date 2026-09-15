// SPDX-License-Identifier: AGPL-3.0-only

#include "yolo/blocking_queue.hpp"
#include "yolo/config.hpp"
#include "yolo/trt_engine.hpp"
#include "yolo/types.hpp"

#include <cuda_runtime_api.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

bool is_image(const fs::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return std::tolower(c); });
    return extension == ".jpg" || extension == ".jpeg" || extension == ".png" || extension == ".bmp";
}

bool is_camera(const std::string& source) {
    return !source.empty() && std::all_of(source.begin(), source.end(), [](unsigned char c) { return std::isdigit(c); });
}

struct LaterFrame {
    bool operator()(const yolo::Frame& left, const yolo::Frame& right) const { return left.id > right.id; }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " CONFIG.yaml\n";
        return 2;
    }

    try {
        const auto config = yolo::load_config(argv[1]);
        if (cudaSetDevice(config.device) != cudaSuccess) {
            throw std::runtime_error("cannot select CUDA device " + std::to_string(config.device));
        }

        yolo::TrtEngine engine(config.engine_path);
        yolo::BlockingQueue<yolo::Frame> input(config.queue_depth);
        yolo::BlockingQueue<yolo::Frame> output(config.queue_depth);
        std::mutex error_mutex;
        std::exception_ptr background_error;
        const auto record_error = [&](std::exception_ptr error) {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!background_error) background_error = error;
            input.close();
            output.close();
        };

        std::thread reader([&] {
            try {
                std::uint64_t id = 0;
                const auto submit = [&](const cv::Mat& image) {
                    if (image.empty()) throw std::runtime_error("input contains an unreadable frame");
                    yolo::Frame frame;
                    frame.id = id++;
                    frame.image = image.clone();
                    frame.enqueued_at = std::chrono::steady_clock::now();
                    return input.push(std::move(frame));
                };

                const fs::path source(config.source);
                if (fs::is_directory(source)) {
                    std::vector<fs::path> images;
                    for (const auto& entry : fs::directory_iterator(source)) {
                        if (entry.is_regular_file() && is_image(entry.path())) images.push_back(entry.path());
                    }
                    std::sort(images.begin(), images.end());
                    for (const auto& image_path : images) {
                        if (config.max_frames && id >= config.max_frames) break;
                        if (!submit(cv::imread(image_path.string()))) break;
                    }
                } else if (fs::is_regular_file(source) && is_image(source)) {
                    submit(cv::imread(source.string()));
                } else {
                    cv::VideoCapture capture;
                    if (is_camera(config.source)) capture.open(std::stoi(config.source));
                    else capture.open(config.source);
                    if (!capture.isOpened()) throw std::runtime_error("cannot open input source: " + config.source);
                    cv::Mat image;
                    while ((!config.max_frames || id < config.max_frames) && capture.read(image)) {
                        if (!submit(image)) break;
                    }
                }
                input.close();
            } catch (...) {
                record_error(std::current_exception());
            }
        });

        std::atomic<std::size_t> workers_left{config.workers};
        std::vector<std::thread> workers;
        workers.reserve(config.workers);
        for (std::size_t i = 0; i < config.workers; ++i) {
            workers.emplace_back([&] {
                try {
                    if (cudaSetDevice(config.device) != cudaSuccess) {
                        throw std::runtime_error("worker cannot select CUDA device " + std::to_string(config.device));
                    }
                    yolo::TrtWorker worker(engine);
                    yolo::Frame frame;
                    while (input.pop(frame)) {
                        worker.process(frame);
                        if (!output.push(std::move(frame))) break;
                    }
                } catch (...) {
                    record_error(std::current_exception());
                }
                if (workers_left.fetch_sub(1) == 1) output.close();
            });
        }

        std::priority_queue<yolo::Frame, std::vector<yolo::Frame>, LaterFrame> pending;
        std::uint64_t expected = 0;
        std::uint64_t measured = 0;
        double inference_sum = 0.0;
        double latency_sum = 0.0;
        cv::VideoWriter writer;
        std::chrono::steady_clock::time_point measurement_start;

        const auto consume = [&](yolo::Frame frame) {
            frame.end_to_end_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - frame.enqueued_at).count();
            if (frame.id >= config.warmup_frames) {
                if (measured == 0) measurement_start = frame.enqueued_at;
                ++measured;
                inference_sum += frame.inference_ms;
                latency_sum += frame.end_to_end_ms;
            }
            if (!config.output_video.empty()) {
                if (!writer.isOpened()) {
                    writer.open(config.output_video, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                                config.output_fps, frame.image.size());
                    if (!writer.isOpened()) throw std::runtime_error("cannot open output video: " + config.output_video);
                }
                writer.write(frame.image);
            }
            if (config.show) {
                cv::imshow("YoloTRTFlow", frame.image);
                if (cv::waitKey(1) == 'q') input.close();
            }
            std::cout << "\rProcessed: " << (frame.id + 1) << std::flush;
        };

        try {
            yolo::Frame frame;
            while (output.pop(frame)) {
                pending.push(std::move(frame));
                while (!pending.empty() && pending.top().id == expected) {
                    auto next = pending.top();
                    pending.pop();
                    consume(std::move(next));
                    ++expected;
                }
            }
        } catch (...) {
            record_error(std::current_exception());
        }

        reader.join();
        for (auto& worker : workers) worker.join();
        if (background_error) std::rethrow_exception(background_error);
        if (!pending.empty()) throw std::runtime_error("pipeline output is missing one or more frames");

        std::cout << "\nFrames: " << expected << '\n';
        if (measured) {
            const double elapsed_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - measurement_start).count();
            std::cout << "Average TensorRT + D2H time: " << inference_sum / measured << " ms\n"
                      << "Average end-to-end latency: " << latency_sum / measured << " ms\n"
                      << "Pipeline throughput: " << measured * 1000.0 / elapsed_ms << " FPS\n";
        } else {
            std::cout << "No frames remained after benchmark warmup.\n";
        }
        cv::destroyAllWindows();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
