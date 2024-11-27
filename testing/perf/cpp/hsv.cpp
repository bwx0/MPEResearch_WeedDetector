#include "hsv.h"
#include <opencv2/core/hal/intrin.hpp>

void hsv1(const cv::Mat &image, cv::Mat &result, int H_lo, int H_hi, int S_lo, int S_hi, int V_lo, int V_hi) {
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> channels(3);
    cv::split(hsv, channels);

    cv::Mat maskH, maskS, maskV;

    cv::inRange(channels[0], H_lo, H_hi, maskH);
    cv::inRange(channels[1], S_lo, S_hi, maskS);
    cv::inRange(channels[2], V_lo, V_hi, maskV);

    result = maskH & maskS & maskV;
}


void hsv2(const cv::Mat &image, cv::Mat &result, int H_lo, int H_hi, int S_lo, int S_hi, int V_lo, int V_hi) {
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> channels(3);
    cv::split(hsv, channels);

    cv::inRange(channels[0], H_lo, H_hi, channels[0]);
    cv::inRange(channels[1], S_lo, S_hi, channels[1]);
    cv::inRange(channels[2], V_lo, V_hi, channels[2]);

    result = channels[0] & channels[1] & channels[2];
}

void hsv3(const cv::Mat &image, cv::Mat &result, int H_lo, int H_hi, int S_lo, int S_hi, int V_lo, int V_hi) {
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> channels(3);
    cv::split(hsv, channels);

    cv::inRange(channels[0], H_lo, H_hi, channels[0]);
    cv::inRange(channels[1], S_lo, S_hi, channels[1]);
    channels[0] &= channels[1];
    cv::inRange(channels[2], V_lo, V_hi, channels[1]);
    channels[0] &= channels[1];

    result = channels[0];
}

void hsv4(const cv::Mat &image, cv::Mat &result, int H_lo, int H_hi, int S_lo, int S_hi, int V_lo, int V_hi) {
    result.create(image.rows, image.cols, CV_8U);

    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    int width = image.cols, height = image.rows;

    for (int y = 0; y < height; ++y) {
        const uchar *srcRow = hsv.ptr<uchar>(y);
        uchar *dstRow = result.ptr<uchar>(y);

        int x = 0;

        const int pixels_per_iteration = 16;
        cv::v_uint8x16 v_h_lo = cv::v_setall_u8(H_lo), v_h_hi = cv::v_setall_u8(H_hi);
        cv::v_uint8x16 v_s_lo = cv::v_setall_u8(S_lo), v_s_hi = cv::v_setall_u8(S_hi);
        cv::v_uint8x16 v_v_lo = cv::v_setall_u8(V_lo), v_v_hi = cv::v_setall_u8(V_hi);

        for (; x <= width - pixels_per_iteration; x += pixels_per_iteration) {
            cv::v_uint8x16 h_vec, s_vec, v_vec;
            cv::v_load_deinterleave(srcRow + x * 3, h_vec, s_vec, v_vec);

            cv::v_uint8x16 h_mask = (v_h_lo <= h_vec) & (h_vec <= v_h_hi);
            cv::v_uint8x16 s_mask = (v_s_lo <= s_vec) & (s_vec <= v_s_hi);
            cv::v_uint8x16 v_mask = (v_v_lo <= v_vec) & (v_vec <= v_v_hi);

            cv::v_uint8x16 mask = h_mask & s_mask & v_mask;

            cv::v_store(dstRow + x, mask);
        }

        for (; x < width; x++) {
            int h = srcRow[3 * x + 0];
            int s = srcRow[3 * x + 1];
            int v = srcRow[3 * x + 2];

            dstRow[x] = (H_lo <= h && h <= H_hi &&
                         S_lo <= s && s <= S_hi &&
                         V_lo <= v && v <= V_hi
                        ) ? 255 : 0;
        }
    }

}