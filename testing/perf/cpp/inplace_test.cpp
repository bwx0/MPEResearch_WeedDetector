#include <opencv2/opencv.hpp>
#include <opencv2/core/hal/intrin.hpp>

constexpr int low_thresh = 25;

cv::Mat method1(const cv::Mat image) {
    cv::Mat result;
    cv::threshold(image, result, 100, 255, cv::THRESH_BINARY);
    return result;
}


cv::Mat method2(const cv::Mat image) {
    cv::threshold(image, image, 100, 255, cv::THRESH_BINARY);
    return image;
}


const int RUN_TIMES = 300;

int main() {
    std::cout << ::cv::getBuildInformation() << std::endl;
    cv::TickMeter tm;

    auto image = cv::imread("../1080p.png");
    cv::Mat result;

    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    // warm up
    for (int i = 0; i < RUN_TIMES; i++) {
        method1(image);
        method2(image);
    }


    double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0, t8 = 0, t9 = 0, t10 = 0;

    // ========== METHOD 1 ==========
    for (int i = 0; i < RUN_TIMES; i++) {
        tm.start();
        auto r = method1(image);
        tm.stop();
        std::cout << "Method 1 Time: " << tm.getTimeMilli() << " ms\n";
        t1 += tm.getTimeMilli();
        tm.reset();
        cv::imwrite("img1.png", r);
    }
    std::cout << std::endl;

    // ========== METHOD 2 ==========
    for (int i = 0; i < RUN_TIMES; i++) {
        tm.start();
        auto r = method2(image);
        tm.stop();
        std::cout << "Method 2 Time: " << tm.getTimeMilli() << " ms\n";
        t2 += tm.getTimeMilli();
        tm.reset();
        cv::imwrite("img2.png", r);
    }
    std::cout << std::endl;

    // ========== END ==========
    std::cout << "Method 1 Average Time: " << t1 / RUN_TIMES << " ms" << std::endl;
    std::cout << "Method 2 Average Time: " << t2 / RUN_TIMES << " ms" << std::endl;
    return 0;
}
