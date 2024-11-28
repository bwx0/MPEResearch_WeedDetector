#include "hsv.h"
#include "perf.h"

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
    for (int i = 0; i < 10; i++) {
        hsv1(image, result, H_lo, H_hi, S_lo, S_hi, V_lo, V_hi);
        cv::imwrite("out/hsv1.png", result);
        hsv2(image, result, H_lo, H_hi, S_lo, S_hi, V_lo, V_hi);
        cv::imwrite("out/hsv2.png", result);
        hsv3(image, result, H_lo, H_hi, S_lo, S_hi, V_lo, V_hi);
        cv::imwrite("out/hsv3.png", result);
        hsv4(image, result, H_lo, H_hi, S_lo, S_hi, V_lo, V_hi);
        cv::imwrite("out/hsv4.png", result);
    }

    benchmark(hsv1, 1000, 5, "C++.HSV_1", image, result, H_lo, H_hi, S_lo, S_hi, V_lo, V_hi);
    benchmark(hsv2, 1000, 5, "C++.HSV_2", image, result, H_lo, H_hi, S_lo, S_hi, V_lo, V_hi);
    benchmark(hsv3, 1000, 5, "C++.HSV_3", image, result, H_lo, H_hi, S_lo, S_hi, V_lo, V_hi);
    benchmark(hsv4, 1000, 5, "C++.HSV_4", image, result, H_lo, H_hi, S_lo, S_hi, V_lo, V_hi);

    return 0;
}
