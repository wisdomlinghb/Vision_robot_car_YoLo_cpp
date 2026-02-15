cmake_minimum_required(VERSION 3.10)
project(rpi_robot_ai)

# 1. 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)

# 2. 寻找 OpenCV 包 (这是系统安装的，所以能自动找到)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# ============ 关键修改开始(2026.2.13 9pm) ============
# 3. 设置 ONNX Runtime 的路径 (指向我们刚才下载的文件夹)
set(ONNX_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime")

# 4. 告诉编译器：头文件(.h) 去哪里找？
# 相当于告诉厨师：菜谱在 "onnxruntime/include" 里面
include_directories(${ONNX_ROOT}/include)

# 5. 告诉链接器：库文件(.so) 去哪里找？
# 相当于告诉厨师：食材在 "onnxruntime/lib" 里面
link_directories(${ONNX_ROOT}/lib)
# ============ 关键修改结束 ============

# 6. 生成可执行文件
add_executable(main_app src/main.cpp)

# 7. 链接所有库
# 注意：onnxruntime 是库的名字 (libonnxruntime.so 去掉 lib 和 .so)
target_link_libraries(main_app PRIVATE 
    ${OpenCV_LIBS} 
    onnxruntime 
    Threads::Threads
)

#Phase 6:NEON test
# 添加 NEON 测试程序
add_executable(neon_test src/neon_test.cpp)
target_link_libraries(neon_test PRIVATE ${OpenCV_LIBS})

# 开启编译器的 NEON 优化选项 (针对 Cortex-A72)
target_compile_options(neon_test PRIVATE -O3 -mcpu=cortex-a72)