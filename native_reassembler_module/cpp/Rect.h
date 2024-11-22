#ifndef IMAGE_REASSEMBLER_RECTS_H
#define IMAGE_REASSEMBLER_RECTS_H

#include <algorithm>
#include <functional>
#include <string>
#include <stdexcept>
#include <optional>

class Rect {
public:
    int x, y, w, h;

    Rect() : x(0), y(0), w(0), h(0) {}

    Rect(int x_, int y_, int w_, int h_) : x(x_), y(y_), w(w_), h(h_) {}

    bool contains(int px, int py) const {
        return (px >= x && px < x + w && py >= y && py < y + h);
    }

    bool contains_rect(const Rect &other) const {
        return (x <= other.x && y <= other.y && x + w >= other.x + other.w && y + h >= other.y + other.h);
    }

    bool intersects(const Rect &other) const {
        return !(x + w < other.x || other.x + other.w < x ||
                 y + h < other.y || other.y + other.h < y);
    }

    Rect merge(const Rect &other) const {
        int new_x = std::min(x, other.x);
        int new_y = std::min(y, other.y);
        int new_w = std::max(x + w, other.x + other.w) - new_x;
        int new_h = std::max(y + h, other.y + other.h) - new_y;
        return {new_x, new_y, new_w, new_h};
    }

    int area() const {
        return w * h;
    }
};

class RectMapping {
public:
    Rect src;
    std::optional<Rect> dst;

    RectMapping(const Rect &src_, const std::optional<Rect> &dst_) : src(src_), dst(dst_) {}
};


enum class RectSortingMethod {
    WIDTH_DESC = 1,
    HEIGHT_DESC = 2,
    AREA_DESC = 3,
    MAXSIDE_DESC = 4
};

inline std::function<bool(const Rect &a, const Rect &b)> getRectComparator(RectSortingMethod method) {
    switch (method) {
        case RectSortingMethod::WIDTH_DESC:
            return [](const Rect &a, const Rect &b) -> bool { return a.w > b.w; };
        case RectSortingMethod::HEIGHT_DESC:
            return [](const Rect &a, const Rect &b) -> bool { return a.h > b.h; };
        case RectSortingMethod::AREA_DESC:
            return [](const Rect &a, const Rect &b) -> bool { return a.area() > b.area(); };
        case RectSortingMethod::MAXSIDE_DESC:
            return [](const Rect &a, const Rect &b) -> bool { return std::max(a.w, a.h) > std::max(b.w, b.h); };
        default:
            throw std::runtime_error(std::string("Invalid Rect sorting method: ") + std::to_string((int) method));
    }
}


#endif //IMAGE_REASSEMBLER_RECTS_H
