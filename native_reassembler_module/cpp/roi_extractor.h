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
    /**
     *
     * @param exgi_threshold
     * @param scale_factor Scale down the input image by the specified factor at the beginning,
     * apply the same ROI extraction process as the Python implementation, and scale the ROIs back
     * to the original size. This can reduce the
     * @param max_size The maximum width or height an ROI tile can have to still be considered valid;
     * otherwise, it will be considered as part of a crop row.
     * @param min_size The minimum width or height an ROI tile must have to still be considered valid;
     * otherwise, it will be considered as noise.
     * @param merge_overlapping_rects Merge overlapping ROIs or not. Merging overlapping ROIs can prevent the same plant
     * from showing up in multiple ROIs.
     */
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
