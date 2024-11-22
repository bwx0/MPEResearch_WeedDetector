#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

std::string dst_path = "data/VegIndexTest";

double div(double a, double b) {
    return b != 0 ? a / b : 0;
}

cv::Mat apply_filter(const cv::Mat& rgb_image, const std::function<cv::Mat(const cv::Mat&, const cv::Mat&, const cv::Mat&)>& filter) {
    std::vector<cv::Mat> channels(3);
    cv::split(rgb_image, channels);
    
    cv::Mat R = channels[2]; // Note: OpenCV loads images in BGR format
    cv::Mat G = channels[1];
    cv::Mat B = channels[0];
    
    cv::Mat filtered = filter(R, G, B);
    cv::Mat normalized;
    cv::normalize(filtered, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    
    return normalized;
}

namespace opencv_impl{

cv::Mat ExGI(const cv::Mat& R, const cv::Mat& G, const cv::Mat& B) {
    return 2 * G - R - B;
}

cv::Mat GLI(const cv::Mat& R, const cv::Mat& G, const cv::Mat& B) {
    return (2 * G - R - B) / (2 * G + R + B + 1e-5); // Adding a small constant to avoid division by zero
}

cv::Mat MPRI(const cv::Mat& R, const cv::Mat& G, const cv::Mat& B) {
    return (G - R) / (G + R + 1e-5);
}

cv::Mat VARI(const cv::Mat& R, const cv::Mat& G, const cv::Mat& B) {
    return (G - R) / (G + R - B + 1e-5);
}

cv::Mat GRRI(const cv::Mat& R, const cv::Mat& G, const cv::Mat& B) {
    return 255 * G / (R + 1e-5);
}

cv::Mat VEG(const cv::Mat& R, const cv::Mat& G, const cv::Mat& B) {
    cv::Mat R_float, B_float;
    R.convertTo(R_float, CV_32F); // Convert R to floating point
    B.convertTo(B_float, CV_32F); // Convert B to floating point

    cv::Mat R_pow, B_pow;
    cv::pow(R_float, 2.0/3, R_pow); // Apply power operation
    cv::pow(B_float, 1.0/3, B_pow); // Apply power operation

    cv::Mat denominator = R_pow.mul(B_pow) + cv::Scalar(1e-5); // Ensure no division by zero
    cv::Mat result;
    cv::divide(255 * G, denominator, result, 1, CV_8U); // Perform division and scale

    return result;
}


cv::Mat MGRVI(const cv::Mat& R, const cv::Mat& G, const cv::Mat& B) {
    return (G.mul(G) - R.mul(R)) / (G.mul(G) + R.mul(R) + 1e-5);
}

cv::Mat RGVBI(const cv::Mat& R, const cv::Mat& G, const cv::Mat& B) {
    return (G - B.mul(R)) / (G.mul(G) + R.mul(B) + 1e-5);
}


}

namespace plain_impl{

cv::Mat ExGI(const cv::Mat& R, const cv::Mat& G, const cv::Mat& B) {
    cv::Mat result = cv::Mat::zeros(G.size(), G.type());
    for (int i = 0; i < G.rows; i++) {
        for (int j = 0; j < G.cols; j++) {
            result.at<uchar>(i, j) = (2 * G.at<uchar>(i, j) - R.at<uchar>(i, j) - B.at<uchar>(i, j));
        }
    }
    return result;
}

cv::Mat GLI(const cv::Mat& R, const cv::Mat& G, const cv::Mat& B) {
    cv::Mat result = cv::Mat::zeros(G.size(), CV_32F);
    for (int i = 0; i < G.rows; i++) {
        for (int j = 0; j < G.cols; j++) {
            float numerator = 2 * G.at<uchar>(i, j) - R.at<uchar>(i, j) - B.at<uchar>(i, j);
            float denominator = 2 * G.at<uchar>(i, j) + R.at<uchar>(i, j) + B.at<uchar>(i, j) + 1e-5;
            result.at<float>(i, j) = numerator / denominator;
        }
    }
    return result;
}

cv::Mat MPRI(const cv::Mat& R, const cv::Mat& G, const cv::Mat& B) {
    cv::Mat result = cv::Mat::zeros(G.size(), CV_32F);
    for (int i = 0; i < G.rows; i++) {
        for (int j = 0; j < G.cols; j++) {
            float numerator = G.at<uchar>(i, j) - R.at<uchar>(i, j);
            float denominator = G.at<uchar>(i, j) + R.at<uchar>(i, j) + 1e-5;
            result.at<float>(i, j) = numerator / denominator;
        }
    }
    return result;
}

cv::Mat VARI(const cv::Mat& R, const cv::Mat& G, const cv::Mat& B) {
    cv::Mat result = cv::Mat::zeros(G.size(), CV_32F);
    for (int i = 0; i < G.rows; i++) {
        for (int j = 0; j < G.cols; j++) {
            float numerator = G.at<uchar>(i, j) - R.at<uchar>(i, j);
            float denominator = G.at<uchar>(i, j) + R.at<uchar>(i, j) - B.at<uchar>(i, j) + 1e-5;
            result.at<float>(i, j) = numerator / denominator;
        }
    }
    return result;
}

cv::Mat GRRI(const cv::Mat& R, const cv::Mat& G, const cv::Mat& B) {
    cv::Mat result = cv::Mat::zeros(G.size(), CV_32F);
    for (int i = 0; i < G.rows; i++) {
        for (int j = 0; j < G.cols; j++) {
            float denominator = R.at<uchar>(i, j) + 1e-5;
            result.at<float>(i, j) = 255 * G.at<uchar>(i, j) / denominator;
        }
    }
    return result;
}

cv::Mat VEG(const cv::Mat& R, const cv::Mat& G, const cv::Mat& B) {
    cv::Mat result = cv::Mat::zeros(G.size(), CV_8U);
    for (int i = 0; i < G.rows; i++) {
        for (int j = 0; j < G.cols; j++) {
            float R_pow = std::pow(R.at<uchar>(i, j), 2.0/3);
            float B_pow = std::pow(B.at<uchar>(i, j), 1.0/3);
            float denominator = R_pow * B_pow + 1e-5;
            result.at<uchar>(i, j) = cv::saturate_cast<uchar>(255 * G.at<uchar>(i, j) / denominator);
        }
    }
    return result;
}

cv::Mat MGRVI(const cv::Mat& R, const cv::Mat& G, const cv::Mat& B) {
    cv::Mat result = cv::Mat::zeros(G.size(), CV_32F);
    for (int i = 0; i < G.rows; i++) {
        for (int j = 0; j < G.cols; j++) {
            float G2 = G.at<uchar>(i, j) * G.at<uchar>(i, j);
            float R2 = R.at<uchar>(i, j) * R.at<uchar>(i, j);
            float numerator = G2 - R2;
            float denominator = G2 + R2 + 1e-5;
            result.at<float>(i, j) = numerator / denominator;
        }
    }
    return result;
}

cv::Mat RGVBI(const cv::Mat& R, const cv::Mat& G, const cv::Mat& B) {
    cv::Mat result = cv::Mat::zeros(G.size(), CV_32F);
    for (int i = 0; i < G.rows; i++) {
        for (int j = 0; j < G.cols; j++) {
            float RB = R.at<uchar>(i, j) * B.at<uchar>(i, j);
            float G2 = G.at<uchar>(i, j) * G.at<uchar>(i, j);
            float numerator = G.at<uchar>(i, j) - RB;
            float denominator = G2 + RB + 1e-5;
            result.at<float>(i, j) = numerator / denominator;
        }
    }
    return result;
}

}

void test_filter(const cv::Mat& rgb_image, const std::function<cv::Mat(const cv::Mat&, const cv::Mat&, const cv::Mat&)>& filter, const std::string& name) {
    int nTests = 100;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < nTests; ++i) {
        cv::Mat filtered = apply_filter(rgb_image, filter);
    }
    
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    
    std::cout << name << "  elapsed: " << duration / 1000.0 << "s     speed: " << duration / double(nTests) << "ms/img" << std::endl;
}
using namespace plain_impl;
int main() {
    cv::Mat rgbimg = cv::imread("/home/pi/proj/camtest/data/pisize.png");
    
    test_filter(rgbimg, ExGI, "ExGI");
    test_filter(rgbimg, ExGI, "ExGI");
    test_filter(rgbimg, GLI, "GLI");
    test_filter(rgbimg, MPRI, "MPRI");
    test_filter(rgbimg, VARI, "VARI");
    test_filter(rgbimg, GRRI, "GRRI");
    test_filter(rgbimg, VEG, "VEG");
    test_filter(rgbimg, MGRVI, "MGRVI");
    test_filter(rgbimg, RGVBI, "RGVBI");
    
    return 0;
}