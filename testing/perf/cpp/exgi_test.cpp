#include "exgi.h"
#include<iostream>

const int RUN_TIMES = 20;
int threshold = 25;

int main() {
    // std::cout << ::cv::getBuildInformation() << std::endl;
    cv::TickMeter tm;

    auto image = cv::imread("../1080p.png");
    cv::Mat result;

    // warm up
    for (int i = 0; i < RUN_TIMES; i++) {
        ExGI1(image, result, threshold);
        ExGI2(image, result, threshold);
        ExGI3(image, result, threshold);
        ExGI4(image, result, threshold);
        ExGI5(image, result, threshold);
        ExGI6(image, result, threshold);
        ExGI7(image, result, threshold);
        ExGI8(image, result, threshold);
        ExGI9(image, result, threshold);
        ExGI10(image, result, threshold);
    }


    double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0, t8 = 0, t9 = 0, t10 = 0;

    // ========== METHOD 1 ==========
    for (int i = 0; i < RUN_TIMES; i++) {
        tm.start();
        ExGI1(image, result, threshold);
        tm.stop();
        std::cout << "Method 1 Time: " << tm.getTimeMilli() << " ms\n";
        t1 += tm.getTimeMilli();
        tm.reset();
    }
    cv::imwrite("out/exgi1.png", result);
    std::cout << std::endl;

    // ========== METHOD 2 ==========
    for (int i = 0; i < RUN_TIMES; i++) {
        tm.start();
        ExGI2(image, result, threshold);
        tm.stop();
        std::cout << "Method 2 Time: " << tm.getTimeMilli() << " ms\n";
        t2 += tm.getTimeMilli();
        tm.reset();
    }
    cv::imwrite("out/exgi2.png", result);
    std::cout << std::endl;

    // ========== METHOD 3 ==========
    for (int i = 0; i < RUN_TIMES; i++) {
        tm.start();
        ExGI3(image, result, threshold);
        tm.stop();
        std::cout << "Method 3 Time: " << tm.getTimeMilli() << " ms\n";
        t3 += tm.getTimeMilli();
        tm.reset();
    }
    cv::imwrite("out/exgi3.png", result);

    // ========== METHOD 4 ==========
    std::cout << std::endl;
    for (int i = 0; i < RUN_TIMES; i++) {
        tm.start();
        ExGI4(image, result, threshold);
        tm.stop();
        std::cout << "Method 4 Time: " << tm.getTimeMilli() << " ms\n";
        t4 += tm.getTimeMilli();
        tm.reset();
    }
    cv::imwrite("out/exgi4.png", result);
    std::cout << std::endl;

    // ========== METHOD 5 ==========
    for (int i = 0; i < RUN_TIMES; i++) {
        tm.start();
        ExGI5_nosplit(image, result, threshold);
        tm.stop();
        std::cout << "Method 5 Time: " << tm.getTimeMilli() << " ms\n";
        t5 += tm.getTimeMilli();
        tm.reset();
    }
    cv::imwrite("out/exgi5.png", result);
    std::cout << std::endl;

    // ========== METHOD 6 ==========
    for (int i = 0; i < RUN_TIMES; i++) {
        tm.start();
        ExGI6(image, result, threshold);
        tm.stop();
        std::cout << "Method 6 Time: " << tm.getTimeMilli() << " ms\n";
        t6 += tm.getTimeMilli();
        tm.reset();
    }
    cv::imwrite("out/exgi6.png", result);
    std::cout << std::endl;

    // ========== METHOD 7 ==========
    for (int i = 0; i < RUN_TIMES; i++) {
        tm.start();
        ExGI7(image, result, threshold);
        tm.stop();
        std::cout << "Method 7 Time: " << tm.getTimeMilli() << " ms\n";
        t7 += tm.getTimeMilli();
        tm.reset();
    }
    cv::imwrite("out/exgi7.png", result);
    std::cout << std::endl;

    // ========== METHOD 8 ==========
    for (int i = 0; i < RUN_TIMES; i++) {
        tm.start();
        ExGI8(image, result, threshold);
        tm.stop();
        std::cout << "Method 8 Time: " << tm.getTimeMilli() << " ms\n";
        t8 += tm.getTimeMilli();
        tm.reset();
    }
    cv::imwrite("out/exgi8.png", result);
    std::cout << std::endl;

    // ========== METHOD 9 ==========
    for (int i = 0; i < RUN_TIMES; i++) {
        tm.start();
        ExGI9(image, result, threshold);
        tm.stop();
        std::cout << "Method 9 Time: " << tm.getTimeMilli() << " ms\n";
        t9 += tm.getTimeMilli();
        tm.reset();
    }
    cv::imwrite("out/exgi9.png", result);
    std::cout << std::endl;

    // ========== METHOD 10 ==========
    for (int i = 0; i < RUN_TIMES; i++) {
        tm.start();
        ExGI10(image, result, threshold);
        tm.stop();
        std::cout << "Method 10 Time: " << tm.getTimeMilli() << " ms\n";
        t10 += tm.getTimeMilli();
        tm.reset();
    }
    cv::imwrite("out/exgi10.png", result);
    std::cout << std::endl;

    // ========== END ==========
    std::cout << "[C++.ExGI_1] Average Time: " << t1 / RUN_TIMES << " ms" << std::endl;
    std::cout << "[C++.ExGI_2] Average Time: " << t2 / RUN_TIMES << " ms" << std::endl;
    std::cout << "[C++.ExGI_3] Average Time: " << t3 / RUN_TIMES << " ms" << std::endl;
    std::cout << "[C++.ExGI_4] Average Time: " << t4 / RUN_TIMES << " ms" << std::endl;
    std::cout << "[C++.ExGI_5] Average Time: " << t5 / RUN_TIMES << " ms" << std::endl;
    std::cout << "[C++.ExGI_6] Average Time: " << t6 / RUN_TIMES << " ms" << std::endl;
    std::cout << "[C++.ExGI_7] Average Time: " << t7 / RUN_TIMES << " ms" << std::endl;
    std::cout << "[C++.ExGI_8] Average Time: " << t8 / RUN_TIMES << " ms" << std::endl;
    std::cout << "[C++.ExGI_9] Average Time: " << t9 / RUN_TIMES << " ms" << std::endl;
    std::cout << "[C++.ExGI_10] Average Time: " << t10 / RUN_TIMES << " ms" << std::endl;
    return 0;
}
