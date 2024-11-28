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

    /**
     * See the documentation in the Python implementation. Randomized margin size is not supported here.
     * @param srcImg The input image from which to extract ROIs and reassemble them.
     * @param dstImg The output image
     * @param packer_width The width for the fixed packer. A reasonable value is the square root of the (estimated)
     * total area of all ROIs. If using a resizable packer (`use_resizable_packer=True`), this can be ignored or
     * set to an arbitrary value.
     * @param sorting_method An appropriate ROI sorting method is required to achieve optimal space utilisation.
     * The default one (HEIGHT_DESC) is good enough for both fixed size and resizable packer.
     * @param use_resizable_packer Frame size are automatically adjusted during fitting. packer_width does not take effect when use_resizable_packer is true.
     * @param border_thickness The thickness (in pixels) of the black borders added to all sides of each ROI.
     * @param padding_size The number of pixels by which to expand each side of the rectangles outward from their center.
     * This enlarges the cropped area uniformly while keeping it centered around its original position.
     * @param roi_extractor
     */
    void reassemble(const cv::Mat &srcImg,
                    cv::Mat &dstImg,
                    int packer_width = 640,
                    RectSortingMethod
                    sorting_method = RectSortingMethod::HEIGHT_DESC,
                    bool use_resizable_packer = true,
                    int border_thickness = 3,
                    int padding_size = 8,
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
