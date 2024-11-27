#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include "exgi.h"
#include "hsv.h"

namespace py = pybind11;

std::function<void(const cv::Mat &, cv::Mat &, int)> exgi_impls[] = {
        ExGI1, ExGI2, ExGI3, ExGI4, ExGI5, ExGI6, ExGI7, ExGI8, ExGI9, ExGI10, ExGI5_nosplit
};
std::function<void(const cv::Mat &, cv::Mat &, int, int, int, int, int, int)> hsv_impls[] = {
        hsv1, hsv2, hsv3, hsv4
};

py::array_t<uint8_t> ExGI_binarize(const py::array_t<uint8_t> &py_srcImg, int threshold, int impl_index) {
    auto rows = py_srcImg.shape(0), cols = py_srcImg.shape(1);
    cv::Mat srcImg(rows, cols, CV_8UC3, (unsigned char *) py_srcImg.data());

    cv::Mat *dstImg = new cv::Mat();
    exgi_impls[impl_index](srcImg, *dstImg, threshold);

    py::capsule free_when_done(dstImg, [](void *v) {
        delete reinterpret_cast<cv::Mat *>(v);
    });
    py::array_t<uint8_t> py_result(
            {dstImg->rows, dstImg->cols},
            dstImg->data,
            free_when_done
    );

    return py_result;
}

py::array_t<uint8_t> RGB2HSV(const py::array_t<uint8_t> &py_srcImg) {
    auto rows = py_srcImg.shape(0), cols = py_srcImg.shape(1);
    cv::Mat srcImg(rows, cols, CV_8UC3, (unsigned char *) py_srcImg.data());

    cv::Mat *dstImg = new cv::Mat();
    cv::cvtColor(srcImg, *dstImg, cv::COLOR_RGB2HSV);

    py::capsule free_when_done(dstImg, [](void *v) {
        delete reinterpret_cast<cv::Mat *>(v);
    });
    py::array_t<uint8_t> py_result(
            {dstImg->rows, dstImg->cols, 3},
            dstImg->data,
            free_when_done
    );

    return py_result;
}

py::array_t<uint8_t> HSV_binarize(const py::array_t<uint8_t> &py_srcImg,
                                  int H_lo, int H_hi, int S_lo, int S_hi, int V_lo, int V_hi,
                                  int impl_index) {
    auto rows = py_srcImg.shape(0), cols = py_srcImg.shape(1);
    cv::Mat srcImg(rows, cols, CV_8UC3, (unsigned char *) py_srcImg.data());

    cv::Mat *dstImg = new cv::Mat();
    hsv_impls[impl_index](srcImg, *dstImg, H_lo, H_hi, S_lo, S_hi, V_lo, V_hi);

    py::capsule free_when_done(dstImg, [](void *v) {
        delete reinterpret_cast<cv::Mat *>(v);
    });
    py::array_t<uint8_t> py_result(
            {dstImg->rows, dstImg->cols},
            dstImg->data,
            free_when_done
    );

    return py_result;
}


PYBIND11_MODULE(perf_cpp, m) {
    m.def("ExGI_binarize", &ExGI_binarize, "ExGI_binarize()");
    m.def("RGB2HSV", &RGB2HSV, "RGB2HSV()");
    m.def("HSV_binarize", &HSV_binarize, "HSV_binarize()");
}
