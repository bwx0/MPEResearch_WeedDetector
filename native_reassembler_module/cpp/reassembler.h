#ifndef IMAGE_REASSEMBLER_REASSEMBLER_H
#define IMAGE_REASSEMBLER_REASSEMBLER_H

#include "Rect.h"
#include "roi_extractor.h"
#include <vector>
#include <opencv2/opencv.hpp>

class Reassembler {
public:
    Reassembler();

    void addRect(const Rect &rect);

    void reassemble(const cv::Mat &srcImg,
                    cv::Mat &dstImg,
                    int initial_width = 640,
                    RectSortingMethod
                    sorting_method = RectSortingMethod::HEIGHT_DESC,
                    bool autosize = true,
                    int border = 3,
                    int margin = 8,
                    ROIExtractor *roi_extractor = default_roi_extractor.get()
    );

    std::vector<RectMapping> reverse_mapping(const std::vector<Rect> &rects) const;

    std::vector<RectMapping> mapping(const std::vector<Rect> &rects) const;

    std::vector<RectMapping> get_raw_mappings() const;

private:
    cv::Mat draw_rects(const cv::Mat &srcImg,
                       const std::vector<RectMapping> &rect_mappings,
                       int border = 0);

    std::vector<RectMapping> map0(const std::vector<Rect> &rect_mapped_space,
                                  const std::vector<Rect> &rect_reference,
                                  const std::vector<Rect> &rects,
                                  bool rev) const;

    std::vector<Rect> m_rects;
    std::vector<RectMapping> m_mappings;
    bool m_reassembled;
    static inline std::unique_ptr<ROIExtractor> default_roi_extractor = createROIExtractor(ROIExtractorType::Default);
};


#endif //IMAGE_REASSEMBLER_REASSEMBLER_H
