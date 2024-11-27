#include "hsv.h"
#include<iostream>

const int RUN_TIMES = 100;
int H_lo = 35, H_hi = 80;
int S_lo = 40, S_hi = 225;
int V_lo = 50, V_hi = 200;

int main() {
    // std::cout << ::cv::getBuildInformation() << std::endl;
    cv::TickMeter tm;

    auto image = cv::imread("../1080p.png");
    cv::Mat result;

    // warm up
    for (int i = 0; i < RUN_TIMES * 3; i++) {
        hsv1(image, result, H_lo, H_hi, S_lo, S_hi, V_lo, V_hi);
        hsv2(image, result, H_lo, H_hi, S_lo, S_hi, V_lo, V_hi);
        hsv3(image, result, H_lo, H_hi, S_lo, S_hi, V_lo, V_hi);
    }


    double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0, t8 = 0, t9 = 0, t10 = 0;

    // ========== METHOD 1 ==========
    for (int i = 0; i < RUN_TIMES; i++) {
        tm.start();
        hsv1(image, result, H_lo, H_hi, S_lo, S_hi, V_lo, V_hi);
        tm.stop();
        std::cout << "Method 1 Time: " << tm.getTimeMilli() << " ms\n";
        t1 += tm.getTimeMilli();
        tm.reset();
    }
    cv::imwrite("out/hsv1.png", result);
    std::cout << std::endl;

    // ========== METHOD 2 ==========
    for (int i = 0; i < RUN_TIMES; i++) {
        tm.start();
        hsv2(image, result, H_lo, H_hi, S_lo, S_hi, V_lo, V_hi);
        tm.stop();
        std::cout << "Method 2 Time: " << tm.getTimeMilli() << " ms\n";
        t2 += tm.getTimeMilli();
        tm.reset();
    }
    cv::imwrite("out/hsv2.png", result);
    std::cout << std::endl;

    // ========== METHOD 3 ==========
    for (int i = 0; i < RUN_TIMES; i++) {
        tm.start();
        hsv3(image, result, H_lo, H_hi, S_lo, S_hi, V_lo, V_hi);
        tm.stop();
        std::cout << "Method 3 Time: " << tm.getTimeMilli() << " ms\n";
        t3 += tm.getTimeMilli();
        tm.reset();
    }
    cv::imwrite("out/hsv3.png", result);
    std::cout << std::endl;

    // ========== METHOD 4 ==========
    for (int i = 0; i < RUN_TIMES; i++) {
        tm.start();
        hsv4(image, result, H_lo, H_hi, S_lo, S_hi, V_lo, V_hi);
        tm.stop();
        std::cout << "Method 4 Time: " << tm.getTimeMilli() << " ms\n";
        t4 += tm.getTimeMilli();
        tm.reset();
    }
    cv::imwrite("out/hsv4.png", result);
    std::cout << std::endl;

    // ========== END ==========
    std::cout << "[C++.HSV_1] Average Time: " << t1 / RUN_TIMES << " ms" << std::endl;
    std::cout << "[C++.HSV_2] Average Time: " << t2 / RUN_TIMES << " ms" << std::endl;
    std::cout << "[C++.HSV_3] Average Time: " << t3 / RUN_TIMES << " ms" << std::endl;
    std::cout << "[C++.HSV_4] Average Time: " << t4 / RUN_TIMES << " ms" << std::endl;
    return 0;
}
