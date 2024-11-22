#include "reassembler.h"
#include "packer.h"
#include "util.h"


Reassembler::Reassembler() : m_reassembled(false) {
}

void Reassembler::addRect(const Rect &rect) {
    if (m_reassembled)
        throw std::runtime_error("Cannot add more rectangles after reassembly.");
    m_rects.push_back(rect);
}

void Reassembler::reassemble(const cv::Mat &srcImg,
                             cv::Mat &dstImg,
                             int initial_width,
                             RectSortingMethod sorting_method,
                             bool autosize,
                             int border,
                             int margin,
                             ROIExtractor *roi_extractor) {
    if (m_reassembled)
        throw std::runtime_error("Already reassembled");

    Stopwatch sw_total_time;
    Stopwatch sw;

    // Step 0: Extract green regions
    if (roi_extractor != nullptr) {
        std::vector<Rect> green_rects = roi_extractor->extract_roi(srcImg);
        for (const Rect &rect: green_rects) {
            addRect(rect);
        }
    }
    sw.stop_and_print("extract roi");

    // Step 1: Finalise rectangles (add margins and borders)
    int imgw = srcImg.cols;
    int imgh = srcImg.rows;
    std::vector<Rect> expanded_rects;
    for (const Rect &rect: m_rects) {
        int x1 = rect.x;
        int y1 = rect.y;
        int w = rect.w;
        int h = rect.h;
        int x2 = x1 + w;
        int y2 = y1 + h;

        int extra = margin + border;
        x1 = std::max(x1 - extra, 0);
        y1 = std::max(y1 - extra, 0);
        x2 = std::min(x2 + extra, imgw);
        y2 = std::min(y2 + extra, imgh);
        expanded_rects.emplace_back(x1, y1, x2 - x1, y2 - y1);
        auto &&r = expanded_rects.back();
    }

    // Step 2: Sort rects and pack
    auto compare_func = getRectComparator(sorting_method);

    std::vector<Rect> sorted_rects = expanded_rects;
    std::sort(sorted_rects.begin(), sorted_rects.end(), compare_func);
    sw.stop_and_print("sort rects");

    std::vector<RectMapping> mappings;
    if (autosize) {
        ResizablePacker packer;
        mappings = packer.fit(sorted_rects);
    } else {
        Packer packer(initial_width, initial_width);
        mappings = packer.fit(sorted_rects);
    }
    this->m_mappings = mappings;
    sw.stop_and_print("fit");

    // Step 3: Construct the reassembled image
    cv::Mat result_img = draw_rects(srcImg, mappings, border);
    sw.stop_and_print("construct img");

    m_reassembled = true;

    // Collect stats
    int effective_area = 0;
    int n_fail = 0;
    for (const RectMapping &mapping: mappings) {
        if (mapping.dst) {
            effective_area += mapping.dst->area();
        } else {
            n_fail += 1;
        }
    }

    double total_time = sw_total_time.stop();

    int total_area = result_img.cols * result_img.rows;
    std::cout << "Result image size: " << result_img.cols << "x" << result_img.rows
              << "  effective_area=" << effective_area
              << "  total_area=" << total_area
              << "  utilisation=" << static_cast<double>(effective_area) / total_area
              << "  n_fail=" << n_fail
              << "  total_time=" << total_time << "ms:" << std::endl;

    dstImg = std::move(result_img);
}

cv::Mat Reassembler::draw_rects(const cv::Mat &srcImg,
                                const std::vector<RectMapping> &rect_mappings,
                                int border) {
    int canvas_width = 64;
    int canvas_height = 64;
    for (const RectMapping &rm: rect_mappings) {
        if (rm.dst) {
            canvas_width = std::max(canvas_width, rm.dst->x + rm.dst->w);
            canvas_height = std::max(canvas_height, rm.dst->y + rm.dst->h);
        }
    }

    cv::Mat canvas(canvas_height, canvas_width, srcImg.type(), cv::Scalar(0, 0, 0));

// TODO do we parallelize this?
#pragma omp parallel for
    for (int i = 0; i < rect_mappings.size(); i++) {
        const RectMapping &rm = rect_mappings[i];
        if (rm.dst) {
            int dst_x_start = rm.dst->x + border;
            int dst_y_start = rm.dst->y + border;
            int dst_x_end = rm.dst->x + rm.dst->w - border;
            int dst_y_end = rm.dst->y + rm.dst->h - border;

            int src_x_start = rm.src.x + border;
            int src_y_start = rm.src.y + border;
            int src_x_end = rm.src.x + rm.src.w - border;
            int src_y_end = rm.src.y + rm.src.h - border;

            cv::Rect dst_roi(dst_x_start, dst_y_start, dst_x_end - dst_x_start, dst_y_end - dst_y_start);
            cv::Rect src_roi(src_x_start, src_y_start, src_x_end - src_x_start, src_y_end - src_y_start);

            if (dst_roi.width > 0 && dst_roi.height > 0 && src_roi.width > 0 && src_roi.height > 0 &&
                src_roi.x >= 0 && src_roi.y >= 0 && src_roi.x + src_roi.width <= srcImg.cols &&
                src_roi.y + src_roi.height <= srcImg.rows &&
                dst_roi.x >= 0 && dst_roi.y >= 0 && dst_roi.x + dst_roi.width <= canvas.cols &&
                dst_roi.y + dst_roi.height <= canvas.rows) {
                srcImg(src_roi).copyTo(canvas(dst_roi));
            }
        }
    }

    return canvas;
}

std::vector<RectMapping> Reassembler::reverse_mapping(const std::vector<Rect> &rects) const {
    if (!m_reassembled)
        throw std::runtime_error("Not reassembled");

    std::vector<RectMapping> result = map0(rects, rects, rects, true);
    return result;
}

std::vector<RectMapping> Reassembler::mapping(const std::vector<Rect> &rects) const {
    if (!m_reassembled)
        throw std::runtime_error("Not reassembled");

    std::vector<RectMapping> result = map0(rects, rects, rects, false);
    return result;
}

std::vector<RectMapping> Reassembler::map0(const std::vector<Rect> &rect_mapped_space,
                                           const std::vector<Rect> &rect_reference,
                                           const std::vector<Rect> &rects,
                                           bool rev) const {
    std::vector<RectMapping> result;
    std::vector<std::vector<double>> ious = box_ioa1(rect_mapped_space, rect_reference);

    for (size_t i = 0; i < rect_mapped_space.size(); ++i) {
        const Rect &rect = rects[i];
        size_t ci = std::distance(ious[i].begin(), std::max_element(ious[i].begin(), ious[i].end()));
        double iou = ious[i][ci];

        for (size_t j = 0; j < m_mappings.size(); ++j) {
            const RectMapping &rm = m_mappings[j];
            if ((rev && rm.dst->contains_rect(rect)) || (!rev && rm.src.contains_rect(rect))) {
                iou = 1.0;
                ci = j;
                break;
            }
        }
        RectMapping candidate = m_mappings[ci];

        if (!candidate.dst || iou < 0.6) {
            result.emplace_back(rect, std::nullopt);
        } else {
            const Rect &dst = rev ? candidate.dst.value() : candidate.src;
            const Rect &src = rev ? candidate.src : candidate.dst.value();
            int dx = rect.x - dst.x;
            int dy = rect.y - dst.y;
            Rect mapped_rect(src.x + dx, src.y + dy, rect.w, rect.h);
            result.emplace_back(rect, mapped_rect);
        }
    }

    return result;
}

std::vector<RectMapping> Reassembler::get_raw_mappings() const {
    return m_mappings;
}
