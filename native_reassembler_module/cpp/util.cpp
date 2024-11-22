#include "util.h"
#include <opencv2/core/hal/intrin.hpp>

bool timing_log_enabled = false;

void set_timing_log_enabled(bool enabled) {
    timing_log_enabled = enabled;
}

cv::Mat bgr2ExGI_cvfuncs(const cv::Mat &image, int low_threshold) {
    cv::Mat image_i32, result;
    image.convertTo(image_i32, CV_16S);

    std::vector<cv::Mat> channels(3);
    cv::split(image_i32, channels);
    cv::Mat B = channels[0];
    cv::Mat G = channels[1];
    cv::Mat R = channels[2];

    cv::Mat exg = 2 * G - R - B;

    cv::threshold(exg, image_i32, low_threshold, 255, cv::THRESH_BINARY);

    image_i32.convertTo(result, CV_8U);
    return result;
}

std::vector<Rect> merge_overlapping_rectangles(const std::vector<Rect> &rects0) {
    if (rects0.empty()) {
        return {};
    }

    std::vector<Rect> rects = rects0;
    std::sort(rects.begin(), rects.end(), [](const Rect &a, const Rect &b) -> bool {
        return a.x < b.x;
    });

    std::vector<Rect> q1, q2;
    q1.reserve(rects.size());
    q2.reserve(rects.size());

    for (Rect u: rects) {
        bool changed = true;
        while (changed) {
            changed = false;
            for (Rect v: q1) {
                if (u.intersects(v)) {
                    u = u.merge(v);
                } else {
                    q2.push_back(v);
                }
            }
            q1.clear();
            std::swap(q1, q2);
        }
        q1.push_back(u);
    }

    return q1;
}

std::vector<std::vector<double>> box_ioa1(const std::vector<Rect> &boxes1, const std::vector<Rect> &boxes2) {
    std::vector<std::vector<double>> result(boxes1.size(), std::vector<double>(boxes2.size(), 0.0));
    for (size_t i = 0; i < boxes1.size(); ++i) {
        const Rect &a = boxes1[i];
        for (size_t j = 0; j < boxes2.size(); ++j) {
            const Rect &b = boxes2[j];

            int x1 = std::max(a.x, b.x);
            int y1 = std::max(a.y, b.y);
            int x2 = std::min(a.x + a.w, b.x + b.w);
            int y2 = std::min(a.y + a.h, b.y + b.h);

            int interArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);
            int box1Area = a.area();

            double ioa = interArea / static_cast<double>(box1Area);
            result[i][j] = ioa;
        }
    }
    return result;
}

cv::Mat bgr2ExGI_loop(const cv::Mat &image, int low_threshold, int nThreads) {
    cv::Mat result;
    result.create(image.rows, image.cols, CV_8U);

#pragma omp parallel for num_threads(nThreads)
    for (int y = 0; y < image.rows; ++y) {
        const cv::Vec3b *srcRow = image.ptr<cv::Vec3b>(y);
        uchar *dstRow = result.ptr<uchar>(y);

        for (int x = 0; x < image.cols; ++x) {
            int b = srcRow[x][0];
            int g = srcRow[x][1];
            int r = srcRow[x][2];

            int value = 2 * g - r - b;

            dstRow[x] = (value >= low_threshold) ? 255 : 0;
        }
    }
    return result;
}

cv::Mat bgr2ExGI_simd(const cv::Mat &image, int low_threshold, int nThreads) {
    cv::Mat result;
    result.create(image.rows, image.cols, CV_8U);

    int width = image.cols;
    int height = image.rows;

#pragma omp parallel for num_threads(nThreads) shared(width, height, low_threshold, image, result) default(none)
    for (int y = 0; y < height; ++y) {
        const uchar *srcRow = image.ptr<uchar>(y);
        uchar *dstRow = result.ptr<uchar>(y);

        int x = 0;

        const int pixels_per_iteration = 16;

        for (; x <= width - pixels_per_iteration; x += pixels_per_iteration) {
            cv::v_uint8x16 b_vec, g_vec, r_vec;
            // srcRow[B G R B G R B G R B G R ...]  ---->  b_vec[B B B B ...] g_vec[G G G G ...] r_vec[R R R R ...]
//            __builtin_prefetch(srcRow + ((x * 3) & ~64) + 64);
//            __builtin_prefetch(srcRow + ((x * 3) & ~64) + 128);
//            __builtin_prefetch(srcRow + ((x * 3) & ~64) + 192);
//            __builtin_prefetch(srcRow + ((x * 3) & ~64) + 256);
//            __builtin_prefetch(dstRow + (x & ~64));
            cv::v_load_deinterleave(srcRow + x * 3, b_vec, g_vec, r_vec);

            // Expand to 16-bit integers
            cv::v_uint16 b_low, b_high, g_low, g_high, r_low, r_high;
            cv::v_expand(b_vec, b_low, b_high);
            cv::v_expand(g_vec, g_low, g_high);
            cv::v_expand(r_vec, r_low, r_high);

            // 2 * g - r - b
            cv::v_uint16x8 val_low = g_low + g_low - r_low - b_low;
            cv::v_uint16x8 val_high = g_high + g_high - r_high - b_high;

            cv::v_int16x8 s_low = cv::v_reinterpret_as_s16(val_low);
            cv::v_int16x8 s_high = cv::v_reinterpret_as_s16(val_high);

            cv::v_int16x8 mask_low = s_low >= cv::v_setall_s16(low_threshold);
            cv::v_int16x8 mask_high = s_high >= cv::v_setall_s16(low_threshold);

            // Pack the results back to 8-bit
            // We can stop at v_pack because values are already 255 by then.
            cv::v_uint8x16 result_vec = cv::v_pack(cv::v_reinterpret_as_u16(mask_low),
                                                   cv::v_reinterpret_as_u16(mask_high));

            cv::v_store(dstRow + x, result_vec);
        }

        // Process any remaining pixels
        for (; x < width; ++x) {
            int b = srcRow[3 * x + 0];
            int g = srcRow[3 * x + 1];
            int r = srcRow[3 * x + 2];

            int value = 2 * g - r - b;

            dstRow[x] = (value >= low_threshold) ? 255 : 0;
        }
    }
    return result;
}
