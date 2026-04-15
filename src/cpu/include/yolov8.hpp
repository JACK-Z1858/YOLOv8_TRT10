//
// Created bu ubuntu2404 on 2026/04/15
//
#ifndef YOLOV8_HPP
#define YOLOV8_HPP
#include <opencv2/opencv.hpp>

float clamp(float val, float min, float max) {
    return val > min ? (val < max ? val : max) : min;
}

cv::Mat infer(cv::dnn::Net& net, cv::Mat& blob) {
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    std::vector<std::string> layerNames = net.getUnconnectedOutLayersNames();

    printf("Output layers count: %zu\n", layerNames.size());
    for (auto& name : layerNames) {
        printf("  layer: %s\n", name.c_str());
    }
    
    net.forward(outputs, layerNames);

    printf("outputs size: %zu\n", outputs.size());
    for (int k = 0; k < outputs.size(); k++) {
        printf("  output[%d] dims: %d\n", k, outputs[k].dims);
        for (int d = 0; d < outputs[k].dims; d++) {
            printf("    dim[%d] = %d\n", d, outputs[k].size[d]);
        }
    }
    return outputs[0];
}

void postprocess(const cv::Mat&        output,
                std::vector<cv::Rect>& boxes,
                std::vector<float>&    scores,
                std::vector<int>&      labels,
                const float            rw,
                const float            rh)
{
    cv::Mat ot        = output.reshape(1, output.size[1]);
    const float* data = (float*)ot.data;
    for (int i = 0; i < 8400; ++i) {
        float max_score = 0.0f;
        int best_class  = -1;

        for (int j = 0; j < 80; ++j) {
            int row     = j + 4;
            float score = data[row * 8400 + i];

            if(score > max_score) {
                max_score  = score;
                best_class = j;
            }
        }
        if (max_score > 0.5f) {
            float x = data[0 * 8400 + i] * rw;
            float y = data[1 * 8400 + i] * rh;
            float w = data[2 * 8400 + i] * rw;
            float h = data[3 * 8400 + i] * rh;

            // 中心点坐标转换为左上角坐标
            int left = int(x - 0.5f * w);
            int top  = int(y - 0.5f * h);

            boxes.emplace_back(cv::Rect(left, top, (int)w, (int)h));
            scores.push_back(max_score);
            labels.push_back(best_class);
        }
    }
}

void draw(cv::Mat& img,
        const std::vector<cv::Rect>&    boxes,
        const std::vector<float>&       scores,
        const std::vector<int>&         labels,
        const std::vector<int>&         indices,
        const std::vector<std::string>&               CLASS_NAMES,
        const std::vector<std::vector<unsigned int>>& COLORS)
{
    for (int index : indices) {
        cv::Rect box   = boxes[index];
        float    score = scores[index];
        int      label = labels[index];

        cv::Scalar color = cv::Scalar(COLORS[label][0], COLORS[label][1], COLORS[label][2]);
        cv::rectangle(img, box, color);
        
        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[label].c_str(), score * 100);
        int      baseLine   = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);
        
        cv::rectangle(img, cv::Rect(box.x, box.y, label_size.width, label_size.height + baseLine), 
                    {0, 0, 255}, -1);
        cv::putText(img, text, cv::Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX,
                    0.4, {255, 255, 255}, 1); 
    } 
}
#endif