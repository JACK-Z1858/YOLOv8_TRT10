//
// Created by ubuntu2404 0n 2026/4/1
//
#include "yolov8.hpp"
#include <memory>
#include <opencv2/core/utility.hpp>  // cv::glob
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include <chrono>
#include <ostream>
#include <ratio>

namespace fs = std::filesystem;

// argument count： 命令行参数数量
// argument vector：存储所有命令行参数的字符串数组
int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s [engine_path] [image_path/image_dir/video_path]\n", argv[0]);
        return -1;
    }

    int frameCount = 0;
    int measureCount = 0;
    int measureBeg = 300;
    int measureEnd = 1000;
    std::chrono::steady_clock::time_point TTstart;
    std::chrono::steady_clock::time_point TTend;
    std::chrono::steady_clock::time_point processBeg;
    std::chrono::steady_clock::time_point processEnd;
    double processSum{0.0};
    // cuda:0
    cudaSetDevice(0);
    
    const std::string engine_file_path{argv[1]};
    const fs::path    path{argv[2]};

    std::vector<std::string> imagePathList;
    bool                     isVideo{false};
    
    std::unique_ptr<YOLOv8> yolov8 = std::make_unique<YOLOv8>(engine_file_path);
    yolov8->make_pipe(true);

    if (fs::exists(path)) {
        std::string suffix = path.extension();
        if (suffix == ".jpg" || suffix == ".jpge" || suffix == ".png") {
            imagePathList.push_back(path);
        } 
        else if (suffix == ".mp4" || suffix == ".avi" || suffix == ".m4v" || suffix == ".mpeg" || suffix == ".mov" || suffix == ".mkv") {
            isVideo = true;
        } 
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            std::abort();
        }
    }
    else if (fs::is_directory(path)) {
        cv::glob(path.string() + "/*.jpg", imagePathList);
    }

    cv::Mat                  res, image;
    cv::Size                 size = cv::Size{640, 640};
    std::vector<det::Object> objs;

    // cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

    if (isVideo) {
        cv::VideoCapture cap(path);

        if (!cap.isOpened()) {
            printf("can not open %s\n", path.c_str());
            return -1;
        }
        
        while (true) {
            if (frameCount == measureBeg) TTstart = std::chrono::steady_clock::now();
            processBeg = std::chrono::steady_clock::now();
            if (!cap.read(image)) break;
            frameCount++;
        
            objs.clear();
            yolov8->copy_from_Mat(image, size);
            // auto start = std::chrono::system_clock::now();
            yolov8->infer();
            // auto end = std::chrono::system_clock::now();
            yolov8->postprocess(objs);
            yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
            processEnd = std::chrono::steady_clock::now();

            if (frameCount >= measureBeg && frameCount <= measureEnd) {
                processSum += std::chrono::duration<double, std::milli>(processEnd - processBeg).count();
                ++measureCount;
            }
            // auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            // printf("cost %2.4lf ms\n", tc);
            // cv::imshow("result", res);
            std::cout << "\rResult showed: " << frameCount << std::flush;
            if (cv::waitKey(1) == 'q' || frameCount == measureEnd) {
                TTend = std::chrono::steady_clock::now();
                break;
            }
        }
    }
    else {
        for (auto& p : imagePathList) {
            objs.clear();
            image = cv::imread(p);
            yolov8->copy_from_Mat(image, size);
            auto start = std::chrono::system_clock::now();
            yolov8->infer();
            auto end = std::chrono::system_clock::now();
            yolov8->postprocess(objs);
            yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS);
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);
            cv::imshow("result", res);
            cv::waitKey(0);
            // cv::imwrite("outputs/res.png", res);
        }
    }
    auto TTtc = std::chrono::duration<double, std::milli>(TTend - TTstart).count();
    std::cout << std::endl
              << "Measure count: " << measureCount << ": Total time cost: " << TTtc << "ms" <<std::endl
              << "Avg process: " << processSum / measureCount << "ms" << std::endl;
    cv::destroyAllWindows();
    return 0;
}