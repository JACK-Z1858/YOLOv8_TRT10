#include "include/thread/thread.hpp"
#include "include/yolov8.hpp"
#include <chrono>
#include <cstdint>
#include <opencv2/highgui.hpp>
#include <ostream>
#include <ratio>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " [engine_path] [data_path] [number of workers]" << std::endl;
        return -1;
    }
    double t_read_sum{0.0};
    double t_pre_sum{0.0};
    double t_infer_sum{0.0};
    double t_post_sum{0.0};
    double t_draw_sum{0.0};
    double t_e2e_sum{0.0};
    uint32_t measureCount{0};
    std::chrono::steady_clock::time_point start, end;

    const std::string enginePath{argv[1]};
    const std::string videoPath{argv[2]};
    uint32_t nbWorkers = static_cast<uint32_t>(std::stoul(argv[3]));
    YOLOv8 yolo(enginePath);
    // yolo.makepipe();

    threadSafeQueue read2work("read2work",2);
    threadSafeQueue work2out("work2out",2);
    threadSafeQueue out2show("out2show", 2);

    std::thread reader(&YOLOv8::VideoReader, &yolo, videoPath, std::ref(read2work));
    std::vector<std::thread> workers;
    workers.reserve(nbWorkers);
    for (int i = 0; i < nbWorkers; ++i) {
        workers.emplace_back(std::thread(&YOLOv8::runWorker, &yolo, std::ref(read2work), std::ref(work2out)));
    }
    
    std::thread outer(&YOLOv8::Outputer, &yolo, std::ref(work2out), std::ref(out2show));

    FrameData res;
    while (out2show.pull(res)) {
        if (res.frame_id == 300) {
                start = std::chrono::steady_clock::now();
            }
        if (res.frame_id >= 300 && res.frame_id <= 1000) {
            
            auto final = std::chrono::steady_clock::now();
            res.prof.t_e2e = std::chrono::duration<double, std::milli>(final - res.t_enqueue).count();
            measureCount += 1;
            t_read_sum += res.prof.t_read;
            t_pre_sum += res.prof.t_pre;
            t_infer_sum += res.prof.t_infer;
            t_post_sum += res.prof.t_post;
            t_draw_sum += res.prof.t_draw;
            t_e2e_sum += res.prof.t_e2e;
            
        }
        //cv::imshow("result", res.img);
        std::cout << "\rResult showed: "<< res.frame_id << std::flush;
        if (cv::waitKey(1) == 'q' || res.frame_id == 1000) {
            end = std::chrono::steady_clock::now();
            read2work.shutdown();
            work2out.shutdown();
            out2show.shutdown();
            break;
        }
    }
    auto tc = std::chrono::duration<double, std::milli>(end - start).count();
    cv::destroyAllWindows();
    cv::waitKey(1);
    reader.join();
    for (int i = 0; i < nbWorkers; ++i) {
        workers[i].join();
    }
    
    outer.join();
    std::cout << std::endl
              << "Measure Count: " << measureCount << std::endl
              << "Avg read time: " << t_read_sum / measureCount << "ms" << std::endl
              << "Avg PreProcess time: " << t_pre_sum / measureCount << "ms" << std::endl
              << "Avg Inference time: " << t_infer_sum / measureCount << "ms" << std::endl
              << "Avg PostProcess time: " << t_post_sum / measureCount << "ms" << std::endl
              << "Avg Draw time: " << t_draw_sum / measureCount << "ms" << std::endl
              << "Avg End2End time: " << t_e2e_sum / measureCount << "ms" << std::endl
              << "Total time cost: " << tc << "ms" << std::endl;
    return 0;
}