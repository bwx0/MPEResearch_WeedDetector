#ifndef IMAGE_REASSEMBLER_UTIL_H
#define IMAGE_REASSEMBLER_UTIL_H

#include "Rect.h"
#include <vector>
#include <opencv2/opencv.hpp>

extern bool timing_log_enabled;

void set_timing_log_enabled(bool enabled);

inline bool is_timing_log_enabled() {
    return timing_log_enabled;
}

cv::Mat bgr2ExGI_cvfuncs(const cv::Mat &image, int low_threshold);

cv::Mat bgr2ExGI_loop(const cv::Mat &image, int low_threshold, int nThreads = 3);

cv::Mat bgr2ExGI_simd(const cv::Mat &image, int low_threshold, int nThreads = 1);

std::vector<Rect> merge_overlapping_rectangles(const std::vector<Rect> &rects);

std::vector<std::vector<double>> box_ioa1(const std::vector<Rect> &boxes1, const std::vector<Rect> &boxes2);

inline cv::Mat bgr2ExGI(const cv::Mat &image, int low_threshold) {
    return bgr2ExGI_simd(image, low_threshold, 1);
}

// A wrapper class for cv::TickMeter that simplifies logging execution time
class Stopwatch {
public:
    Stopwatch() {
        tickMeter.start();
    }

    double stop() {
        tickMeter.stop();
        double elapsed = tickMeter.getTimeMilli();
        tickMeter.reset();
        return elapsed;
    }

    double stop_and_print(const std::string &name, const std::string &msg = "") {
        double elapsed = stop();
        if (is_timing_log_enabled())
            printf("[%s]  %fms  %s\n", name.c_str(), elapsed, msg.c_str());
        return elapsed;
    }

private:
    cv::TickMeter tickMeter;
};

#endif //IMAGE_REASSEMBLER_UTIL_H
