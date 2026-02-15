# 🚗 Raspberry Pi Visual Servoing Robot (C++ & YOLOv11)

> 基于树莓派 4B 的高性能视觉伺服小车，实现了从 Python 原型到 C++ 重构的完整工程化落地。集成 ONNX Runtime 推理、SIMD 指令集加速与有限状态机 (FSM) 控制。

---

## 🌟 核心亮点 (Key Features)

* **高性能 AI 推理**: 将 YOLOv11 模型部署于 **ONNX Runtime C++ API**，相比 Python 版本推理延迟降低 **30%**。
* **底层性能优化**: 利用 **ARM NEON SIMD** 指令集重写图像预处理模块，实测加速比达 **4.6x**。
* **多线程架构**: 使用 `std::thread` 构建 **采集-推理-控制** 分离流水线，实现 30 FPS 实时响应。
* **鲁棒控制逻辑**: 设计包含 **死区控制 (Dead Zone)**、**防抖动 (Debounce)** 和 **状态自恢复** 的 FSM 状态机。
* **工程化交付**: 标准 CMake 构建系统，遵循 **代码与资源分离** 的标准目录结构。

## 🛠️ 技术栈 (Tech Stack)

* **硬件**: Raspberry Pi 4B (4GB), CSI Camera, L298N Driver, TT Motors.
* **语言**: C++14 (主要), Python (原型验证).
* **核心库**: OpenCV 4.x, ONNX Runtime, pthread.
* **工具**: CMake, GCC, Git/GitHub.

## 📂 项目结构 (Project Structure)

```text
├── src/                  # 核心源代码
│   ├── main.cpp          # 主程序入口 (多线程 & FSM)
│   └── neon_test.cpp     # SIMD 性能基准测试代码
├── models/               # 模型资源文件
│   └── best.onnx         # 训练好的 YOLOv11n 模型
├── CMakeLists.txt        # 构建脚本
└── README.md             # 项目说明

## 🚀 快速开始 (Quick Start)

### 1. 依赖安装
```bash
# 安装 OpenCV 和 编译工具
sudo apt-get install libopencv-dev cmake g++

### 3. 编译项目
mkdir build && cd build
cmake ..
make

## 3. 运行
# 运行 SIMD 基准测试
./neon_test

# 运行主程序 (自动加载 ../models/best.onnx)
./main_app

#性能数据（Benchmark）
模块,Naive C++,NEON SIMD,加速比
图像预处理,2141 ms,457 ms,4.68x

Author: Wisdomlinghb
