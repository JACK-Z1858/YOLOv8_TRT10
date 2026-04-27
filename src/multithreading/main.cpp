#include "include/thread/thread.hpp"
#include "include/yolov8.hpp"
#include <opencv2/highgui.hpp>
#include <string>
#include <thread>

int main(int argc, char *argv[]) {
    YOLOv8 yolo;

    const std::string videoPath{argv[1]};

    threadSafeQueue read2work("read2work");
    threadSafeQueue work2out("work2out");
    threadSafeQueue out2show("out2show");

    std::thread reader(&YOLOv8::VideoReader, &yolo, videoPath, std::ref(read2work));
    std::thread worker(&YOLOv8::Worker, &yolo, std::ref(read2work), std::ref(work2out));
    std::thread outer(&YOLOv8::Outputer, &yolo, std::ref(work2out), std::ref(out2show));

    FrameData res;
    cv::resizeWindow("result", 1280, 720);
    while (out2show.pull(res)) {
        cv::imshow("result", res.img);
        if (cv::waitKey(33) == 'q') {
            read2work.shutdown();
            work2out.shutdown();
            out2show.shutdown();
            break;
        }
    }
    cv::destroyAllWindows();
    reader.join();
    worker.join();
    outer.join();
    return 0;
}