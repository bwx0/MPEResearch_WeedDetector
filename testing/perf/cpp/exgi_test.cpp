#include "exgi.h"
#include "perf.h"
#include<iostream>

int threshold = 25;


int main() {
    // std::cout << ::cv::getBuildInformation() << std::endl;
    cv::TickMeter tm;

    auto image = cv::imread("../1080p.png");
    cv::Mat result;

    // warm up
    for (int i = 0; i < 10; i++) {
        ExGI1(image, result, threshold);
        cv::imwrite("out/exgi1.png", result);
        ExGI2(image, result, threshold);
        cv::imwrite("out/exgi2.png", result);
        ExGI3(image, result, threshold);
        cv::imwrite("out/exgi3.png", result);
        ExGI4(image, result, threshold);
        cv::imwrite("out/exgi4.png", result);
        ExGI5(image, result, threshold);
        cv::imwrite("out/exgi5.png", result);
        ExGI6(image, result, threshold);
        cv::imwrite("out/exgi6.png", result);
        ExGI7(image, result, threshold);
        cv::imwrite("out/exgi7.png", result);
        ExGI8(image, result, threshold);
        cv::imwrite("out/exgi8.png", result);
        ExGI9(image, result, threshold);
        cv::imwrite("out/exgi9.png", result);
        ExGI10(image, result, threshold);
        cv::imwrite("out/exgi10.png", result);
    }

    benchmark(ExGI1, 1000, 5, "C++.ExGI_1", image, result, threshold);
    benchmark(ExGI2, 1000, 5, "C++.ExGI_2", image, result, threshold);
    benchmark(ExGI3, 1000, 5, "C++.ExGI_3", image, result, threshold);
    benchmark(ExGI4, 1000, 5, "C++.ExGI_4", image, result, threshold);
    benchmark(ExGI5, 1000, 5, "C++.ExGI_5", image, result, threshold);
    benchmark(ExGI6, 1000, 5, "C++.ExGI_6", image, result, threshold);
    benchmark(ExGI7, 1000, 5, "C++.ExGI_7", image, result, threshold);
    benchmark(ExGI8, 1000, 5, "C++.ExGI_8", image, result, threshold);
    benchmark(ExGI9, 1000, 5, "C++.ExGI_9", image, result, threshold);
    benchmark(ExGI10, 1000, 5, "C++.ExGI_10", image, result, threshold);
    return 0;
}
