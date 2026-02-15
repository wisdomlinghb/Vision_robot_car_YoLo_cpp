/**
 * serial_test.cpp
 * C++ 串口通信测试：向 Arduino 发送控制指令
 * 原理：在 Linux 中，串口就是一个文件，用 write() 写入数据即可。
 */

#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <fcntl.h>      // File Control Definitions
#include <termios.h>    // POSIX Terminal Control Definitions
#include <unistd.h>     // UNIX Standard Definitions
#include <cstring>
#include <thread>
#include <chrono>

// ================= 配置 =================
// 请根据 ls /dev/tty* 的结果修改这里！
const char* PORT_NAME = "/dev/ttyACM0"; // 或者是 /dev/ttyUSB0
const int BAUD_RATE = B9600;          // 你的 Arduino 代码里设定的波特率

// ================= 串口初始化函数 =================
int open_serial_port(const char* port) {
    // O_RDWR: 读写模式
    // O_NOCTTY: 不作为控制终端 (防止键盘信号干扰)
    // O_NDELAY: 非阻塞模式
    int fd = open(port, O_RDWR | O_NOCTTY | O_NDELAY);
    
    if (fd == -1) {
        std::cerr << "❌ 无法打开串口: " << port << std::endl;
        return -1;
    }

    // 配置串口参数 (termios 结构体)
    struct termios options;
    tcgetattr(fd, &options); // 获取当前配置

    // 设置波特率
    cfsetispeed(&options, BAUD_RATE);
    cfsetospeed(&options, BAUD_RATE);

    // 控制模式 (CFLAG)
    options.c_cflag |= (CLOCAL | CREAD); // 忽略调制解调器状态线，启用接收
    options.c_cflag &= ~PARENB;          // 无校验
    options.c_cflag &= ~CSTOPB;          // 1位停止位
    options.c_cflag &= ~CSIZE;           // 清除数据位掩码
    options.c_cflag |= CS8;              // 8位数据位

    // 原始模式 (Raw Mode) - 这一点非常重要！
    // 禁用所有特殊的处理（如回车换行转换、信号字符等），我们要发的是纯数据
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_oflag &= ~OPOST;

    // 应用配置
    tcsetattr(fd, TCSANOW, &options);
    
    // 稍微等待一下串口稳定
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    tcflush(fd, TCIOFLUSH); // 清空缓冲区

    return fd;
}

// ================= 发送指令函数 =================
void send_command(int fd, int left_speed, int right_speed) {
    char buffer[64];
    
    // 构造 JSON 字符串
    // 假设你的 Arduino 解析格式是: {'cmd':'move', 'val':[L, R]}
    // 注意：JSON 格式必须严格匹配你 Arduino 的代码！
    // 如果你之前的协议不同，请在这里修改 sprintf 的格式
    int len = sprintf(buffer, "{\"cmd\":\"move\",\"val\":[%d,%d]}\n", left_speed, right_speed);
    
    // 写入串口
    int bytes_written = write(fd, buffer, len);
    
    if (bytes_written < 0) {
        std::cerr << "⚠️ 写入失败" << std::endl;
    } else {
        std::cout << "📤 发送: " << buffer; // buffer 自带 \n
    }
}

int main() {
    std::cout << "🔌 正在连接串口: " << PORT_NAME << "..." << std::endl;
    int serial_fd = open_serial_port(PORT_NAME);

    if (serial_fd == -1) return -1;

    std::cout << "✅ 串口连接成功！准备发送测试指令..." << std::endl;

    // 测试 1: 前进 (1秒)
    std::cout << "🚗 前进..." << std::endl;
    send_command(serial_fd, 100, 100);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // 测试 2: 后退 (1秒)
    std::cout << "🚗 后退..." << std::endl;
    send_command(serial_fd, -100, -100);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // 测试 3: 原地旋转 (1秒)
    std::cout << "🔄 旋转..." << std::endl;
    send_command(serial_fd, -100, 100);
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // 结束: 停车
    std::cout << "🛑 停车..." << std::endl;
    send_command(serial_fd, 0, 0);

    close(serial_fd);
    std::cout << "👋 测试结束。" << std::endl;

    return 0;
}