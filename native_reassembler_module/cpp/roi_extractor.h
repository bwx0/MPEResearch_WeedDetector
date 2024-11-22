#ifndef IMAGE_REASSEMBLER_ROI_EXTRACTOR_H
#define IMAGE_REASSEMBLER_ROI_EXTRACTOR_H

#include "Rect.h"
#include <opencv2/opencv.hpp>
#include <memory>

enum class ROIExtractorType {
    Default = 0,
    ExGIExtractor = 1,
    HSVExtractor = 2
};

class ROIExtractor {
public:
    virtual std::vector<Rect> extract_roi(const cv::Mat &image) = 0;

    virtual ~ROIExtractor() = default;
};

class ExGIGreenExtractor : public ROIExtractor {
public:
    explicit ExGIGreenExtractor(int exgi_threshold = 25, int scale_factor = 2,
                                int max_size = static_cast<int>(1080 * 0.4), int min_size = 5,
                                bool merge_overlapping_rects = true);

    std::vector<Rect> extract_roi(const cv::Mat &image) override;

private:
    std::vector<Rect> extract_green_regions_bgr(const cv::Mat &image) const;

    int exgi_threshold;
    int scale_factor;
    int max_size;
    int size_threshold;
    bool merge_overlapping_rects;
};

class HSVExtractor : public ROIExtractor {
public:
    explicit HSVExtractor() {}

    std::vector<Rect> extract_roi(const cv::Mat &image) override {
        throw std::runtime_error("Not implemented");
    }
};

std::unique_ptr<ROIExtractor> createROIExtractor(ROIExtractorType type);

#endif //IMAGE_REASSEMBLER_ROI_EXTRACTOR_H
