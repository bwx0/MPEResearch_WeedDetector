#include "roi_extractor.h"
#include "util.h"

ExGIGreenExtractor::ExGIGreenExtractor(int exgi_threshold_, int scale_factor_, int max_size_, int min_size_,
                                       bool merge_overlapping_rects_)
        : exgi_threshold(exgi_threshold_), scale_factor(scale_factor_), max_size(max_size_), size_threshold(min_size_),
          merge_overlapping_rects(merge_overlapping_rects_) {
}

std::vector<Rect> ExGIGreenExtractor::extract_roi(const cv::Mat &bgr_img) {
    return extract_green_regions_bgr(bgr_img);
}

std::vector<Rect> ExGIGreenExtractor::extract_green_regions_bgr(const cv::Mat &image) const {
    Stopwatch sw;
    cv::Mat scaled_image;
    if (scale_factor > 1) {
        cv::resize(image, scaled_image, {image.size[1] / scale_factor, image.size[0] / scale_factor}, 0, 0,
                   cv::INTER_NEAREST);
    } else {
        scaled_image = image;
    }
    sw.stop_and_print("resize " + std::to_string(scale_factor) + "x");

    cv::Mat exg = bgr2ExGI(scaled_image, exgi_threshold);
    sw.stop_and_print("bgr2ExGI");


    // Morphological closing
    // Use the close operation to clean up noise and connect leaves to the main stem, especially if they're
    // separated due to thin or faint connections in the picture.
    // In the Python implementation, the kernel size of 20 was chosen somewhat arbitrarily,
    // assuming the image resolution is 1080p.
    // However, in this C++ implementation, since we scale the original image at the beginning,
    // the kernel size should also be scaled by the same factor. scaled_img_height/50 would be 1080/50,
    // which gives a result close to 20.
    const int kernel_size = std::min(exg.size[0], exg.size[1]) / 50;
    cv::Mat b2;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    cv::morphologyEx(exg, b2, cv::MORPH_CLOSE, kernel);
    sw.stop_and_print("morphEx");

    // Connected components
    cv::Mat labels, stats, centroids;
    int num_labels = cv::connectedComponentsWithStats(b2, labels, stats, centroids, 4, CV_32S, cv::CCL_WU);
    sw.stop_and_print("cc");

    std::vector<Rect> bboxes;
    for (int i = 1; i < num_labels; ++i) {
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        x *= scale_factor, y *= scale_factor, w *= scale_factor, h *= scale_factor;

        bool is_too_small = w < size_threshold || h < size_threshold;
        bool is_crop_row = w > max_size || h > max_size;
        if (is_too_small || is_crop_row)
            continue;

        bboxes.emplace_back(x, y, w, h);
    }
    sw.stop_and_print("filter rects");

    std::vector<Rect> result;
    if (merge_overlapping_rects) {
        result = merge_overlapping_rectangles(bboxes);
        sw.stop_and_print("finalise rects",
                          std::to_string(bboxes.size()) + " -> " + std::to_string(result.size()) + "rects");
    } else {
        result = std::move(bboxes);
    }

    return result;
}


std::unique_ptr<ROIExtractor> createROIExtractor(ROIExtractorType type) {
    switch (type) {
        default:
        case ROIExtractorType::Default:
        case ROIExtractorType::ExGIExtractor:
            return std::make_unique<ExGIGreenExtractor>();
        case ROIExtractorType::HSVExtractor:
            return std::make_unique<HSVExtractor>();
    }
}