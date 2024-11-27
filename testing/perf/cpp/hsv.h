#ifndef PERF_HSV_H
#define PERF_HSV_H
#include <opencv2/opencv.hpp>

void hsv1(const cv::Mat &image, cv::Mat &result, int H_lo, int H_hi, int S_lo, int S_hi, int V_lo, int V_hi);
void hsv2(const cv::Mat &image, cv::Mat &result, int H_lo, int H_hi, int S_lo, int S_hi, int V_lo, int V_hi);
void hsv3(const cv::Mat &image, cv::Mat &result, int H_lo, int H_hi, int S_lo, int S_hi, int V_lo, int V_hi);
void hsv4(const cv::Mat &image, cv::Mat &result, int H_lo, int H_hi, int S_lo, int S_hi, int V_lo, int V_hi);

#endif //PERF_HSV_H
