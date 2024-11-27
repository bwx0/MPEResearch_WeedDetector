#include <opencv2/opencv.hpp>
#include <opencv2/core/hal/intrin.hpp>

void ExGI1(const cv::Mat &image, cv::Mat &result, int threshold) {
    cv::Mat image16s;
    image.convertTo(image16s, CV_16SC3);

    cv::Matx13f transformMat(-1, 2, -1); // Coefficients for B, G, R
    cv::transform(image16s, result, transformMat);

    cv::threshold(result, result, threshold, 255, cv::THRESH_BINARY);

    result.convertTo(result, CV_8U);
}


void ExGI2(const cv::Mat &image, cv::Mat &result, int threshold) {
    result.create(image.rows, image.cols, CV_8U);

    for (int y = 0; y < image.rows; ++y) {
        const cv::Vec3b *srcRow = image.ptr<cv::Vec3b>(y);
        uchar *dstRow = result.ptr<uchar>(y);

        for (int x = 0; x < image.cols; ++x) {
            int b = srcRow[x][0];
            int g = srcRow[x][1];
            int r = srcRow[x][2];

            int value = 2 * g - r - b;

            dstRow[x] = (value > threshold) ? 255 : 0;
        }
    }
}


void ExGI3(const cv::Mat &image, cv::Mat &result, int threshold) {
    cv::Matx13f transformMat(-1.0f, 2.0f, -1.0f);
    cv::Mat imageFloat;

    image.convertTo(imageFloat, CV_32F);
    cv::transform(imageFloat, imageFloat, transformMat);
    cv::threshold(imageFloat, result, threshold, 255, cv::THRESH_BINARY);

    result.convertTo(result, CV_8U);
}


void ExGI4(const cv::Mat &image, cv::Mat &result, int threshold) {
    CV_Assert(image.type() == CV_8UC3);

    std::vector<cv::Mat> channels(3);
    cv::split(image, channels);

    cv::Mat b16s, g16s, r16s;
    channels[0].convertTo(b16s, CV_16S);
    channels[1].convertTo(g16s, CV_16S);
    channels[2].convertTo(r16s, CV_16S);

    result.create(image.rows, image.cols, CV_8U);

    const int simdWidth = 8;

    for (int y = 0; y < image.rows; ++y) {
        const short *bRow = b16s.ptr<short>(y);
        const short *gRow = g16s.ptr<short>(y);
        const short *rRow = r16s.ptr<short>(y);
        uchar *dstRow = result.ptr<uchar>(y);

        int x = 0;
        int width = image.cols;

        // SIMD processing
        for (; x <= width - simdWidth; x += simdWidth) {
            cv::v_int16x8 b_vec = cv::v_load(bRow + x);
            cv::v_int16x8 g_vec = cv::v_load(gRow + x);
            cv::v_int16x8 r_vec = cv::v_load(rRow + x);

            cv::v_int16x8 val = (g_vec << 1) - r_vec - b_vec;

            cv::v_int16x8 mask = val > cv::v_setall_s16(threshold);

            // Set result to 255 where mask is true, else 0
            cv::v_uint8x16 result_vec = cv::v_pack(cv::v_reinterpret_as_u16(mask & cv::v_setall_s16(255)),
                                                   cv::v_setzero_u16());

            cv::v_store(dstRow + x, result_vec);
        }

        // Process remaining pixels
        for (; x < width; ++x) {
            int b = bRow[x];
            int g = gRow[x];
            int r = rRow[x];

            int value = 2 * g - r - b;

            dstRow[x] = (value > threshold) ? 255 : 0;
        }
    }
}

void ExGI5(const cv::Mat &image, cv::Mat &result, int threshold) {
    cv::UMat uImage, uResult;
    image.copyTo(uImage);

    std::vector<cv::UMat> uChannels(3);
    cv::split(uImage, uChannels);

    cv::UMat b16s, g16s, r16s;
    uChannels[0].convertTo(b16s, CV_16S);
    uChannels[1].convertTo(g16s, CV_16S);
    uChannels[2].convertTo(r16s, CV_16S);

    cv::UMat temp;
    cv::add(g16s, g16s, temp);
    cv::subtract(temp, b16s, temp);
    cv::subtract(temp, r16s, temp);

    cv::threshold(temp, uResult, threshold, 255, cv::THRESH_BINARY);

    uResult.convertTo(result, CV_8U);
}


void ExGI6(const cv::Mat &image, cv::Mat &result, int threshold) {
    result.create(image.rows, image.cols, CV_8U);

#pragma omp parallel for num_threads(3)
    for (int y = 0; y < image.rows; ++y) {
        const cv::Vec3b *srcRow = image.ptr<cv::Vec3b>(y);
        uchar *dstRow = result.ptr<uchar>(y);

        for (int x = 0; x < image.cols; ++x) {
            int b = srcRow[x][0];
            int g = srcRow[x][1];
            int r = srcRow[x][2];

            int value = 2 * g - r - b;

            dstRow[x] = (value > threshold) ? 255 : 0;
        }
    }
}


void ExGI7(const cv::Mat &image, cv::Mat &result, int threshold) {
    result.create(image.rows, image.cols, CV_8U);

#pragma omp parallel for num_threads(4)
    for (int y = 0; y < image.rows; ++y) {
        const cv::Vec3b *srcRow = image.ptr<cv::Vec3b>(y);
        uchar *dstRow = result.ptr<uchar>(y);

        for (int x = 0; x < image.cols; ++x) {
            int b = srcRow[x][0];
            int g = srcRow[x][1];
            int r = srcRow[x][2];

            int value = 2 * g - r - b;

            dstRow[x] = (value > threshold) ? 255 : 0;
        }
    }
}

void ExGI8(const cv::Mat &image, cv::Mat &result, int threshold) {
    result.create(image.rows, image.cols, CV_8U);

    int width = image.cols;
    int height = image.rows;

    for (int y = 0; y < height; ++y) {
        const uchar *srcRow = image.ptr<uchar>(y);
        uchar *dstRow = result.ptr<uchar>(y);

        int x = 0;

        const int pixels_per_iteration = 16;

        for (; x <= width - pixels_per_iteration; x += pixels_per_iteration) {
            cv::v_uint8x16 b_vec, g_vec, r_vec;
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

            cv::v_int16x8 mask_low = s_low > cv::v_setall_s16(threshold);
            cv::v_int16x8 mask_high = s_high > cv::v_setall_s16(threshold);

            // Pack the results back to 8-bit
            // We can stop at v_pack because values are already 255 by then.
            cv::v_uint8x16 result_vec = cv::v_pack(cv::v_reinterpret_as_u16(mask_low), cv::v_reinterpret_as_u16(mask_high));

            cv::v_store(dstRow + x, result_vec);
        }

        // Process any remaining pixels
        for (; x < width; ++x) {
            int b = srcRow[3 * x + 0];
            int g = srcRow[3 * x + 1];
            int r = srcRow[3 * x + 2];

            int value = 2 * g - r - b;

            dstRow[x] = (value > threshold) ? 255 : 0;
        }
    }
}

void ExGI9(const cv::Mat &image, cv::Mat &result, int threshold) {
    result.create(image.rows, image.cols, CV_8U);

    int width = image.cols;
    int height = image.rows;

#pragma omp parallel for num_threads(3)
    for (int y = 0; y < height; ++y) {
        const uchar *srcRow = image.ptr<uchar>(y);
        uchar *dstRow = result.ptr<uchar>(y);

        int x = 0;

        const int pixels_per_iteration = 16;

        for (; x <= width - pixels_per_iteration; x += pixels_per_iteration) {
            cv::v_uint8x16 b_vec, g_vec, r_vec;
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

            cv::v_int16x8 mask_low = s_low > cv::v_setall_s16(threshold);
            cv::v_int16x8 mask_high = s_high > cv::v_setall_s16(threshold);

            // Pack the results back to 8-bit
            // We can stop at v_pack because values are already 255 by then.
            cv::v_uint8x16 result_vec = cv::v_pack(cv::v_reinterpret_as_u16(mask_low), cv::v_reinterpret_as_u16(mask_high));

            cv::v_store(dstRow + x, result_vec);
        }

        // Process any remaining pixels
        for (; x < width; ++x) {
            int b = srcRow[3 * x + 0];
            int g = srcRow[3 * x + 1];
            int r = srcRow[3 * x + 2];

            int value = 2 * g - r - b;

            dstRow[x] = (value > threshold) ? 255 : 0;
        }
    }
}


void ExGI10(const cv::Mat &image, cv::Mat &result, int threshold) {
    result.create(image.rows, image.cols, CV_8U);

    int width = image.cols;
    int height = image.rows;

#pragma omp parallel for num_threads(2)
    for (int y = 0; y < height; ++y) {
        const uchar *srcRow = image.ptr<uchar>(y);
        uchar *dstRow = result.ptr<uchar>(y);

        int x = 0;

        const int pixels_per_iteration = 16;

        for (; x <= width - pixels_per_iteration; x += pixels_per_iteration) {
            cv::v_uint8x16 b_vec, g_vec, r_vec;
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

            cv::v_int16x8 mask_low = s_low > cv::v_setall_s16(threshold);
            cv::v_int16x8 mask_high = s_high > cv::v_setall_s16(threshold);

            // Pack the results back to 8-bit
            // We can stop at v_pack because values are already 255 by then.
            cv::v_uint8x16 result_vec = cv::v_pack(cv::v_reinterpret_as_u16(mask_low), cv::v_reinterpret_as_u16(mask_high));

            cv::v_store(dstRow + x, result_vec);
        }

        // Process any remaining pixels
        for (; x < width; ++x) {
            int b = srcRow[3 * x + 0];
            int g = srcRow[3 * x + 1];
            int r = srcRow[3 * x + 2];

            int value = 2 * g - r - b;

            dstRow[x] = (value > threshold) ? 255 : 0;
        }
    }
}

void ExGI5_nosplit(const cv::Mat& image, cv::Mat& result, int threshold) {
    CV_Assert(image.type() == CV_8UC3);

    // Upload the image to UMat
    cv::UMat uImage;
    image.copyTo(uImage);

    // Convert to 16-bit signed integers to prevent overflow
    cv::UMat uImage16S;
    uImage.convertTo(uImage16S, CV_16SC3);

    // Define the transformation matrix for BGR channels
    cv::Matx13f transformMat(-1, 2, -1); // Coefficients for B, G, R

    // Apply the transformation
    cv::UMat temp;
    cv::transform(uImage16S, temp, transformMat);

    // Binarize at threshold 25
    cv::UMat uResult;
    cv::threshold(temp, uResult, threshold, 255, cv::THRESH_BINARY);

    // Download the result back to Mat if necessary
    uResult.convertTo(result, CV_8U);
}
