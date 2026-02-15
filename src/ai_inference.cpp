/**
 * ai_inference.cpp (Restored Final Version)
 * 特性: Letterbox预处理 + 0.75高阈值 + GStreamer低延迟
 */

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

// ================= 配置参数 =================
const std::string MODEL_PATH = "../src/model.onnx";
const int INPUT_W = 320;
const int INPUT_H = 320;
const float CONF_THRESHOLD = 0.75f; // 高阈值，过滤误报
const float IOU_THRESHOLD = 0.45f;

// ================= 结构体定义 =================
struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

// ================= 全局变量 =================
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLO_Infer");
Ort::Session* session = nullptr;

// ================= Letterbox 预处理 =================
void letterbox(const cv::Mat& src, cv::Mat& dst, std::vector<float>& input_tensor_values) {
    int col = src.cols;
    int row = src.rows;
    int _max = std::max(col, row);
    
    cv::Mat canvas = cv::Mat::zeros(_max, _max, CV_8UC3);
    canvas.setTo(cv::Scalar(114, 114, 114)); 
    src.copyTo(canvas(cv::Rect(0, 0, col, row)));

    cv::resize(canvas, dst, cv::Size(INPUT_W, INPUT_H));
    cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
    dst.convertTo(dst, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(dst, channels);
    int pixel_count = INPUT_W * INPUT_H;
    input_tensor_values.assign(pixel_count * 3, 0);
    std::memcpy(input_tensor_values.data(), channels[0].data, pixel_count * sizeof(float));
    std::memcpy(input_tensor_values.data() + pixel_count, channels[1].data, pixel_count * sizeof(float));
    std::memcpy(input_tensor_values.data() + pixel_count * 2, channels[2].data, pixel_count * sizeof(float));
}

// ================= 后处理函数 =================
std::vector<Detection> postprocess(const cv::Mat& frame, float* output, const std::vector<int64_t>& shape) {
    std::vector<Detection> detections;
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    int rows = shape[2];       
    int dimensions = shape[1]; 

    int _max = std::max(frame.cols, frame.rows);
    float scale = (float)_max / INPUT_W; 

    for (int i = 0; i < rows; ++i) {
        float max_conf = 0.0f;
        int class_id = -1;
        for (int j = 4; j < dimensions; ++j) {
            float conf = output[j * rows + i];
            if (conf > max_conf) {
                max_conf = conf;
                class_id = j - 4;
            }
        }

        if (max_conf > CONF_THRESHOLD) {
            float cx = output[0 * rows + i];
            float cy = output[1 * rows + i];
            float w  = output[2 * rows + i];
            float h  = output[3 * rows + i];

            float left   = (cx - 0.5 * w) * scale;
            float top    = (cy - 0.5 * h) * scale;
            float width  = w * scale;
            float height = h * scale;

            boxes.push_back(cv::Rect((int)left, (int)top, (int)width, (int)height));
            confidences.push_back(max_conf);
            class_ids.push_back(class_id);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, IOU_THRESHOLD, indices);
    for (int idx : indices) {
        detections.push_back({class_ids[idx], confidences[idx], boxes[idx]});
    }
    return detections;
}

// ================= 主函数 =================
int main() {
    try {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(2);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = new Ort::Session(env, MODEL_PATH.c_str(), session_options);
    } catch (const std::exception& e) {
        std::cerr << "❌ 模型加载失败: " << e.what() << std::endl;
        return -1;
    }

    // GStreamer 管道：丢帧策略
    std::string pipeline = "libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! queue max-size-buffers=1 leaky=downstream ! appsink sync=false drop=true";
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    
    if (!cap.isOpened()) return -1;

    cv::Mat frame;
    std::vector<float> input_tensor_values;
    std::vector<int64_t> input_shape = {1, 3, INPUT_H, INPUT_W};
    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0"};

    std::cout << "🚀 AI 推理已恢复..." << std::endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        auto start = std::chrono::high_resolution_clock::now();

        cv::Mat processed_img;
        letterbox(frame, processed_img, input_tensor_values);

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

        auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_dims = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape(); 

        std::vector<Detection> results = postprocess(frame, output_data, output_dims);

        auto end = std::chrono::high_resolution_clock::now();
        float fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        for (const auto& det : results) {
            cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
            std::string label = "Class " + std::to_string(det.class_id) + ": " + std::to_string(det.confidence).substr(0, 4);
            if (det.class_id == 0) label = "Bag? " + std::to_string(det.confidence).substr(0, 4);
            cv::putText(frame, label, cv::Point(det.box.x, det.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        }

        cv::putText(frame, "FPS: " + std::to_string(fps), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        cv::imshow("C++ AI Car (Restored)", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    delete session;
    return 0;
}