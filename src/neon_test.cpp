#include <iostream>
#include <vector>
#include <chrono>
#include <arm_neon.h> // 必须引入这个头文件
#include <opencv2/opencv.hpp>

// 任务：将一张灰度图的所有像素值 + 50 (模拟亮度提升)

// 1. 普通 C++ 写法
__attribute__((optimize("no-tree-vectorize")))
void naive_brighten(const cv::Mat& input, cv::Mat& output, int value) {
    int rows = input.rows;
    int cols = input.cols;
    if (input.isContinuous()) {
        cols *= rows;
        rows = 1;
    }
    for (int i = 0; i < rows; ++i) {
        const uint8_t* ptr_in = input.ptr<uint8_t>(i);
        uint8_t* ptr_out = output.ptr<uint8_t>(i);
        for (int j = 0; j < cols; ++j) {
            // 简单的加法，但在 CPU 里是一次次算的
            int val = ptr_in[j] + value;
            ptr_out[j] = (val > 255) ? 255 : val; // 饱和运算
        }
    }
}

// 2. NEON SIMD 写法 (一次处理 16 个像素)
void neon_brighten(const cv::Mat& input, cv::Mat& output, int value) {
    int rows = input.rows;
    int cols = input.cols;
    int total = rows * cols;
    
    const uint8_t* src = input.data;
    uint8_t* dst = output.data;

    // 创建一个包含 16 个 value 的向量
    uint8x16_t v_add = vdupq_n_u8((uint8_t)value);

    int i = 0;
    // 每次步进 16
    for (; i <= total - 16; i += 16) {
        // 1. Load: 一次加载 16 个字节到寄存器
        uint8x16_t v_src = vld1q_u8(src + i);
        
        // 2. Add: 饱和加法 (Qadd)，自动处理溢出(>255变成255)，不用写 if 判断
        uint8x16_t v_dst = vqaddq_u8(v_src, v_add);
        
        // 3. Store: 存回内存
        vst1q_u8(dst + i, v_dst);
    }

    // 处理剩下的尾巴 (不足16个的部分)
    for (; i < total; ++i) {
        int val = src[i] + value;
        dst[i] = (val > 255) ? 255 : val;
    }
}

int main() {
    // 创建一张巨大的图 (4000x3000) 以便看出差距
    cv::Mat src = cv::Mat::zeros(3000, 4000, CV_8UC1);
    randu(src, cv::Scalar(0), cv::Scalar(200)); // 随机填充
    cv::Mat dst_naive = src.clone();
    cv::Mat dst_neon = src.clone();

    // --- 测试普通写法 ---
    auto start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<100; i++) // 跑100次取平均
        naive_brighten(src, dst_naive, 50);
    auto end = std::chrono::high_resolution_clock::now();
    double time_naive = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Naive C++ Time: " << time_naive << " ms" << std::endl;

    // --- 测试 NEON 写法 ---
    start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<100; i++)
        neon_brighten(src, dst_neon, 50);
    end = std::chrono::high_resolution_clock::now();
    double time_neon = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "NEON SIMD Time: " << time_neon << " ms" << std::endl;

    std::cout << "Speedup: " << time_naive / time_neon << "x" << std::endl;

    return 0;
}