/**
 * 视觉伺服小车 Phase 6.2 (逻辑闭环最终版)
 * 修复：UI 方向显示错误、旋转超时计数器异常
 * 增强：状态重置机制、成功奖励机制
 */

#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <csignal>
#include <iomanip>
#include <cstdlib>

// ================== 1. 配置区 ==================
const std::string MODEL_PATH = "./models/best.onnx"; 
const std::string CLASS_LABEL = "Red_Bag";

const cv::Size INPUT_SIZE(320, 320); 
const float CONF_THRESHOLD = 0.5f;
const float IOU_THRESHOLD = 0.4f;

// 动力参数
const int BASE_SPEED = 90;
const int MAX_SPEED = 180;
const int BACK_SPEED = 100;
const int DEAD_ZONE = 30; 

// ================== 2. 共享数据 ==================
cv::Mat shared_frame;
std::string shared_ui_message = "INIT";
std::string shared_ui_fps = "0.0";
std::string shared_state_time = "0.0s";
std::string shared_wheel_info = "L:0 R:0"; 
std::string shared_direction = "STOP"; 
std::vector<cv::Rect> shared_boxes; 

std::mutex data_mtx;
std::atomic<bool> is_running(true); 

// ================== 3. 通信模块 ==================
int serial_fd = -1; 
std::string last_sent_cmd = "";
std::chrono::steady_clock::time_point last_sent_time;

bool init_serial(const std::string& port_name) {
    serial_fd = open(port_name.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if (serial_fd == -1) {
        std::cerr << "[Hardware] Error: Unable to open UART " << port_name << std::endl;
        return false;
    }
    struct termios options;
    tcgetattr(serial_fd, &options);
    cfsetispeed(&options, B9600);
    cfsetospeed(&options, B9600);
    options.c_cflag |= (CLOCAL | CREAD);
    options.c_cflag &= ~PARENB; 
    options.c_cflag &= ~CSTOPB; 
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;     
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_oflag &= ~OPOST;
    tcsetattr(serial_fd, TCSANOW, &options);
    return true;
}

void raw_force_stop() {
    if (serial_fd != -1) {
        std::string stop_cmd = "{\"cmd\":\"move\",\"val\":[0,0]}\n";
        for(int i=0; i<5; i++) { 
            write(serial_fd, stop_cmd.c_str(), stop_cmd.length());
            tcdrain(serial_fd); 
            usleep(20000); 
        }
        std::cout << "[System] RAW STOP SENT." << std::endl;
    }
}

void signal_handler(int signum) {
    std::cout << "\n[System] Caught Ctrl+C! Stopping..." << std::endl;
    is_running = false;
    raw_force_stop();
    if (serial_fd != -1) close(serial_fd);
    exit(0);
}

void smart_send_cmd(const std::string& cmd) {
    if (serial_fd == -1) return;
    auto now = std::chrono::steady_clock::now();
    long ms_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_sent_time).count();

    bool changed = (cmd != last_sent_cmd);
    bool time_to_send = (ms_since_last > 100); 

    if (changed || time_to_send) {
        std::string full_cmd = cmd + "\n";
        write(serial_fd, full_cmd.c_str(), full_cmd.length());
        last_sent_cmd = cmd;
        last_sent_time = now;
    }
}

// ================== 4. 复杂状态机 ==================
enum RobotState {
    SEARCHING, TRACKING, OBSTACLE_STOP, BACKING_UP, 
    WAIT_AFTER_BACKUP, WAIT_AFTER_TURN, EMERGENCY_HALT
};

struct RobotContext {
    RobotState current_state = SEARCHING;
    RobotState prev_state = SEARCHING;
    std::chrono::steady_clock::time_point state_start_time = std::chrono::steady_clock::now();
    
    // 计数器
    int obstacle_retry_count = 0; 
    int turn_retry_count = 0;     
    
    // 计时相关
    std::chrono::steady_clock::time_point last_stop_trigger_time;
    std::chrono::steady_clock::time_point continuous_turn_start_time; 
    bool is_turning = false; // 是否正在连续旋转
    int stop_trigger_count = 0;

    // 参数 (ms)
    const double AREA_RATIO_THRESHOLD = 0.75; 
    const double AREA_RATIO_SAFE = 0.60;
    const int TIMEOUT_STOP_BEFORE_BACK = 6000; 
    const int TIMEOUT_BACKUP = 2000;           
    const int TIMEOUT_WAIT_IDLE = 5000;        
    const int TIMEOUT_MAX_TURN = 6000;         
    const int TIMEOUT_TURN_COOL = 3000;        
};

// ================== 5. UI 线程 ==================
void capture_thread_func() {
    std::string pipeline = "libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1 ! "
                           "videoconvert ! video/x-raw, format=BGR ! "
                           "appsink drop=true sync=false";
    
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) { is_running = false; return; }

    auto last_time = std::chrono::steady_clock::now();
    int frame_count = 0;

    while (is_running) {
        cv::Mat frame;
        cap >> frame; 
        if (frame.empty()) break;

        frame_count++;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();
        if (elapsed > 1000) { 
            double fps = frame_count * 1000.0 / elapsed;
            {
                std::lock_guard<std::mutex> lock(data_mtx);
                shared_ui_fps = std::to_string((int)fps);
            }
            frame_count = 0; last_time = now;
        }

        std::string status_text, fps_text, time_text, wheel_text, dir_text;
        std::vector<cv::Rect> boxes_to_draw;
        {
            std::lock_guard<std::mutex> lock(data_mtx);
            frame.copyTo(shared_frame); 
            status_text = shared_ui_message;
            fps_text = shared_ui_fps;
            time_text = shared_state_time;
            wheel_text = shared_wheel_info; 
            dir_text = shared_direction; 
            boxes_to_draw = shared_boxes; 
        }

        for (const auto& box : boxes_to_draw) {
            cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, CLASS_LABEL, cv::Point(box.x, box.y - 5), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }

        cv::Mat canvas;
        cv::copyMakeBorder(frame, canvas, 0, 80, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

        int font = cv::FONT_HERSHEY_SIMPLEX;
        double scale = 0.6; int thick = 2; cv::Scalar black(0, 0, 0);

        cv::putText(canvas, "FPS:" + fps_text, cv::Point(10, 510), font, scale, black, thick);
        cv::putText(canvas, "T:" + time_text, cv::Point(120, 510), font, scale, cv::Scalar(0,0,255), thick);
        cv::putText(canvas, wheel_text, cv::Point(230, 510), font, scale, cv::Scalar(255,0,0), thick);

        cv::Scalar st_color = (status_text.find("STOP") != std::string::npos) ? cv::Scalar(0,0,255) : black;
        cv::putText(canvas, status_text, cv::Point(10, 545), font, 0.7, st_color, thick);
        
        // 显示方向 (修正颜色)
        cv::putText(canvas, dir_text, cv::Point(350, 545), font, 0.8, cv::Scalar(0,150,0), 2);

        cv::imshow("Robot Vision (Phase 6.2)", canvas);
        if (cv::waitKey(1) == 'q') {
            is_running = false;
            raw_force_stop();
        }
    }
}

// ================== 6. AI 线程 ==================
void inference_thread_func() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YoloInference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    Ort::Session session(env, MODEL_PATH.c_str(), session_options);
    const char* input_names[] = {"images"}; 
    const char* output_names[] = {"output0"};

    RobotContext ctx;

    while (is_running) {
        cv::Mat work_frame;
        {
            std::lock_guard<std::mutex> lock(data_mtx);
            if (shared_frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            shared_frame.copyTo(work_frame);
        }

        // Preprocess & Inference
        int w = work_frame.cols; int h = work_frame.rows; int max_side = std::max(h, w);
        cv::Mat square_img = cv::Mat::zeros(max_side, max_side, CV_8UC3);
        work_frame.copyTo(square_img(cv::Rect(0, 0, w, h)));
        cv::Mat blob = cv::dnn::blobFromImage(square_img, 1.0/255.0, INPUT_SIZE, cv::Scalar(), true, false);

        std::vector<int64_t> input_shape = {1, 3, INPUT_SIZE.height, INPUT_SIZE.width};
        size_t input_tensor_size = 1 * 3 * INPUT_SIZE.height * INPUT_SIZE.width;
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), input_tensor_size, input_shape.data(), input_shape.size());
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

        float* float_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int proposals = output_shape[2]; 
        
        float* ptr_x = float_data; float* ptr_y = float_data + proposals;
        float* ptr_w = float_data + 2 * proposals; float* ptr_h = float_data + 3 * proposals;
        float* ptr_score = float_data + 4 * proposals;

        std::vector<cv::Rect> boxes; std::vector<float> confidences;
        float scale_factor = (float)max_side / INPUT_SIZE.width;

        for (int i = 0; i < proposals; ++i) {
            if (ptr_score[i] > CONF_THRESHOLD) {
                float x = ptr_x[i]; float y = ptr_y[i]; float w = ptr_w[i]; float h = ptr_h[i];
                int left = int((x - 0.5 * w) * scale_factor); int top = int((y - 0.5 * h) * scale_factor);
                int width = int(w * scale_factor); int height = int(h * scale_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back(ptr_score[i]);
            }
        }
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, IOU_THRESHOLD, indices);

        bool target_found = false; float target_area_ratio = 0.0f; int error_x = 0;
        std::vector<cv::Rect> valid_boxes;

        if (!indices.empty()) {
            target_found = true; int idx = indices[0]; cv::Rect box = boxes[idx];
            target_area_ratio = (float)(box.width * box.height) / (w * h); 
            error_x = (box.x + box.width / 2) - (w / 2);
            valid_boxes.push_back(box);
        }

        // === FSM 逻辑 ===
        auto now = std::chrono::steady_clock::now();
        if (ctx.current_state != ctx.prev_state) {
            ctx.state_start_time = now;
            ctx.prev_state = ctx.current_state;
        }
        double state_sec = std::chrono::duration_cast<std::chrono::milliseconds>(now - ctx.state_start_time).count() / 1000.0;
        
        std::string cmd_str = "";
        std::string ui_msg = "";
        std::string wheel_str = "L:0 R:0";
        std::string dir_str = "STOP";

        int speed_l = 0, speed_r = 0;

        switch (ctx.current_state) {
            case SEARCHING:
                ui_msg = "SEARCHING (Idle)";
                speed_l = 0; speed_r = 0;
                
                // [修复] 离开 SEARCHING 时，确保状态干净
                ctx.is_turning = false; 
                
                if (target_found) ctx.current_state = TRACKING;
                break;

            case TRACKING:
                ui_msg = "TRACKING " + std::to_string((int)(target_area_ratio * 100)) + "%";
                if (!target_found) {
                    ctx.current_state = SEARCHING;
                } else if (target_area_ratio >= ctx.AREA_RATIO_THRESHOLD) {
                    ctx.current_state = OBSTACLE_STOP;
                    ctx.stop_trigger_count = 0; 
                } else {
                    int turn = 0;
                    if (std::abs(error_x) > DEAD_ZONE) {
                        turn = error_x * 0.25; 
                    }
                    speed_l = std::min(std::max(BASE_SPEED + turn, -MAX_SPEED), MAX_SPEED);
                    speed_r = std::min(std::max(BASE_SPEED - turn, -MAX_SPEED), MAX_SPEED);

                    // 连续旋转检测
                    bool currently_turning = (std::abs(speed_l - speed_r) > 20); 
                    if (currently_turning) {
                        if (!ctx.is_turning) {
                            ctx.is_turning = true;
                            ctx.continuous_turn_start_time = now; // 记录开始旋转的时间
                        } else {
                            // 检查旋转持续时间
                            double turn_duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - ctx.continuous_turn_start_time).count() / 1000.0;
                            if (turn_duration > (ctx.TIMEOUT_MAX_TURN / 1000.0)) {
                                ctx.current_state = WAIT_AFTER_TURN; // 触发冷却
                            }
                        }
                    } else {
                        // [增强] 如果没有旋转（走直线），重置旋转标记和计数器
                        // 这就是“成功奖励机制”：只要你能走直线，之前的旋转账单清零
                        ctx.is_turning = false;
                        ctx.turn_retry_count = 0; 
                    }
                }
                break;

            case OBSTACLE_STOP:
                ui_msg = "OBSTACLE STOP (" + std::to_string(ctx.obstacle_retry_count) + "/3)";
                speed_l = 0; speed_r = 0;
                
                if (target_found && target_area_ratio >= ctx.AREA_RATIO_THRESHOLD) {
                    if (state_sec > (ctx.TIMEOUT_STOP_BEFORE_BACK / 1000.0)) {
                        ctx.current_state = BACKING_UP;
                    }
                } else {
                    ctx.current_state = TRACKING;
                    ctx.obstacle_retry_count = 0; 
                }
                break;

            case BACKING_UP:
                ui_msg = "BACKING UP";
                if (state_sec > (ctx.TIMEOUT_BACKUP / 1000.0)) {
                    ctx.current_state = WAIT_AFTER_BACKUP; 
                } else {
                    speed_l = -BACK_SPEED; speed_r = -BACK_SPEED;
                }
                break;

            case WAIT_AFTER_BACKUP:
                ui_msg = "WAITING (Retry " + std::to_string(ctx.obstacle_retry_count + 1) + "/3)";
                speed_l = 0; speed_r = 0;
                if (state_sec > (ctx.TIMEOUT_WAIT_IDLE / 1000.0)) {
                    ctx.obstacle_retry_count++;
                    if (ctx.obstacle_retry_count >= 3) {
                        ctx.current_state = EMERGENCY_HALT;
                    } else {
                        ctx.current_state = SEARCHING; 
                    }
                }
                break;

            case WAIT_AFTER_TURN:
                ui_msg = "SPIN COOLING (" + std::to_string(ctx.turn_retry_count + 1) + "/3)";
                speed_l = 0; speed_r = 0;
                
                // [修复] 关键点：进入这个状态时，必须重置 is_turning
                // 否则回到 TRACKING 时，时间差是旧的，会立即再次触发超时
                ctx.is_turning = false;

                if (state_sec > (ctx.TIMEOUT_TURN_COOL / 1000.0)) {
                    ctx.turn_retry_count++;
                    if (ctx.turn_retry_count >= 3) {
                        ctx.current_state = EMERGENCY_HALT;
                    } else {
                        ctx.current_state = SEARCHING;
                    }
                }
                break;

            case EMERGENCY_HALT:
                ui_msg = "EMERGENCY HALT (Manual)";
                speed_l = 0; speed_r = 0;
                break;
        }

        cmd_str = "{\"cmd\":\"move\",\"val\":[" + std::to_string(speed_l) + "," + std::to_string(speed_r) + "]}";
        wheel_str = "L:" + std::to_string(speed_l) + " R:" + std::to_string(speed_r);
        
        // [修复 1] UI 方向判断逻辑：先判断差速，再判断前进
        int diff = speed_l - speed_r;
        if (speed_l == 0 && speed_r == 0) dir_str = "STOP";
        else if (speed_l < -20 && speed_r < -20) dir_str = "BACK";
        else if (diff > 30) dir_str = "RIGHT"; // 左轮快=右转
        else if (diff < -30) dir_str = "LEFT";  // 右轮快=左转
        else dir_str = "FORWARD"; // 只有差速很小且速度>0时才是前进

        {
            std::lock_guard<std::mutex> lock(data_mtx);
            shared_ui_message = ui_msg;
            std::stringstream ss;
            ss << std::fixed << std::setprecision(1) << state_sec << "s";
            shared_state_time = ss.str();
            shared_boxes = valid_boxes;
            shared_wheel_info = wheel_str;
            shared_direction = dir_str; 
        }

        if (!cmd_str.empty()) {
            smart_send_cmd(cmd_str);
        }
    }
}

// ================== 7. Main ==================
int main() {
    atexit(raw_force_stop); 
    signal(SIGINT, signal_handler); 

    std::cout << "=== RPi Robot Vision (Phase 6.2 Logic Fixed) ===" << std::endl;

    if (!init_serial("/dev/ttyACM0")) { 
        std::cerr << "WARNING: Serial init failed!" << std::endl;
    }

    std::thread t1(capture_thread_func);
    std::thread t2(inference_thread_func);

    t1.join();
    t2.join();
    
    return 0;
}