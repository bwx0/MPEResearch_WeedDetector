#ifndef PERF_EXGI_H
#define PERF_EXGI_H

#include <opencv2/opencv.hpp>

void ExGI1(const cv::Mat &image, cv::Mat &result, int threshold);
void ExGI2(const cv::Mat &image, cv::Mat &result, int threshold);
void ExGI3(const cv::Mat &image, cv::Mat &result, int threshold);
void ExGI4(const cv::Mat &image, cv::Mat &result, int threshold);
void ExGI5(const cv::Mat &image, cv::Mat &result, int threshold);
void ExGI6(const cv::Mat &image, cv::Mat &result, int threshold);
void ExGI7(const cv::Mat &image, cv::Mat &result, int threshold);
void ExGI8(const cv::Mat &image, cv::Mat &result, int threshold);
void ExGI9(const cv::Mat &image, cv::Mat &result, int threshold);
void ExGI10(const cv::Mat &image, cv::Mat &result, int threshold);
void ExGI5_nosplit(const cv::Mat& image, cv::Mat& result, int threshold);

#endif //PERF_EXGI_H
