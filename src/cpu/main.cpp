//
// Created bu ubuntu2404 on 2026/4/14
//
#include <yolov8.hpp>
#include <chrono>
#include <fstream>

const std::vector<std::string> CLASS_NAMES = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
    {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
    {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
    {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
    {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
    {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
    {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
    {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
    {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
    {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
    {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
    {80, 183, 189},  {128, 128, 0}};

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s [onnx_path] [image_path]\n", argv[0]);
        return -1;
    }
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    const std::string onnx_path{argv[1]};
    const std::string image_path{argv[2]};

    // 读取.onnx
    cv::dnn::Net net = cv::dnn::readNetFromONNX(onnx_path);
    cv::Mat img      = cv::imread(image_path);

    if (img.empty()) {
        fprintf(stderr, "Error: Cannot read image: %s\n", image_path.c_str());
        return -1;
    }
    if (net.empty()) {
        fprintf(stderr, "Error: Cannot load model: %s\n", onnx_path.c_str());
        return -1;
    }

    float width  = img.cols;
    float height = img.rows;
    float rw     = width / 640.0;
    float rh     = height / 640.0;
    
    cv::Mat blob = cv::dnn::blobFromImage(img, 1/255.0, cv::Size(640, 640), cv::Scalar(), true, false);

    // 执行推理
    auto start = std::chrono::system_clock::now();
    cv::Mat output = infer(net, blob);
    auto end = std::chrono::system_clock::now();
    auto tc  = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
    printf("cost %2.4lf ms\n", tc);

    // postprocess
    std::vector<cv::Rect> boxes;
    std::vector<float>    scores;
    std::vector<int>      labels;
    postprocess(output, boxes, scores, labels, rw, rh);
    
    //NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(
        boxes,
        scores,
        0.5f,
        0.4f,
        indices
    );

    // draw
    draw(img, boxes, scores, labels, indices, CLASS_NAMES, COLORS);
    cv::imshow("result", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}

    