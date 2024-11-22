#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include "Rect.h"
#include "reassembler.h"
#include "util.h"

namespace py = pybind11;

// Mock image processing function
std::pair<py::array_t<uint8_t>, std::vector<RectMapping>> test_to_grayscale(py::array_t<uint8_t> &img) {
    auto rows = img.shape(0);
    auto cols = img.shape(1);
    auto type = CV_8UC3;
    // Create processed image (for example, by converting to grayscale)
    cv::Mat frame(rows, cols, type, (unsigned char *) img.data());
    cv::Mat processedImage;
    cv::cvtColor(frame, processedImage, cv::COLOR_BGR2GRAY);

    // Create dummy RectMappings
    std::vector<RectMapping> mappings;
    mappings.emplace_back(Rect{0, 0, 50, 50}, Rect{10, 10, 50, 50});
    mappings.emplace_back(Rect{20, 20, 30, 30}, Rect{30, 30, 30, 30});

    py::array_t<uint8_t> py_processedImage(
            {processedImage.rows, processedImage.cols},  // shape
            processedImage.data                          // data pointer
    );

    return {py_processedImage, mappings};
}

std::pair<py::array_t<uint8_t>, std::vector<RectMapping>> reassemble(const py::array_t<uint8_t> &py_srcImg,
                                                                     int initial_width,
                                                                     int sorting_method,
                                                                     bool autosize,
                                                                     int border,
                                                                     int margin,
                                                                     ROIExtractor *roi_extractor) {
    // pass the data pointer to a cv::Mat object
    auto rows = py_srcImg.shape(0), cols = py_srcImg.shape(1);
    cv::Mat srcImg(rows, cols, CV_8UC3, (unsigned char *) py_srcImg.data());

    cv::TickMeter tm;
    tm.start();

    std::unique_ptr<ROIExtractor> tmp_extractor;
    if (!roi_extractor) {
        tmp_extractor.reset(new ExGIGreenExtractor());
        roi_extractor = tmp_extractor.get();
    }

    // do the reassembling work
    auto dstImg = new cv::Mat();
    Reassembler r;
    r.reassemble(srcImg,
                 *dstImg,
                 initial_width,
                 (RectSortingMethod) sorting_method,
                 autosize,
                 border, margin,
                 roi_extractor);

    // wrap the result with a python array, along with a capsule, which calls the destructor function when it is garbage collected
    py::capsule free_when_done(dstImg, [](void *v) {
        delete reinterpret_cast<cv::Mat *>(v);
    });
    py::array_t<uint8_t> py_reassembledImage(
            {dstImg->rows, dstImg->cols, 3},
            dstImg->data,
            free_when_done
    );

    tm.stop();
    // printf("reassemble total time: %fms\n", tm.getTimeMilli());

    return std::make_pair(py_reassembledImage, r.get_raw_mappings());
}


PYBIND11_MODULE(native_reassembler_module, m) {
    py::class_<Rect>(m, "Rect")
            .def(py::init<int, int, int, int>())
            .def_readwrite("x", &Rect::x)
            .def_readwrite("y", &Rect::y)
            .def_readwrite("w", &Rect::w)
            .def_readwrite("h", &Rect::h);

    py::class_<RectMapping>(m, "RectMapping")
            .def(py::init<Rect, Rect>())
            .def_readwrite("src", &RectMapping::src)
            .def_readwrite("dst", &RectMapping::dst);

    py::class_<ROIExtractor> roi_ex(m, "NativeROIExtractor");

    py::class_<ExGIGreenExtractor>(m, "NativeExGIExtractor", roi_ex)
            .def(py::init<int, int, int, int, bool>());

    m.def("test_to_grayscale", &test_to_grayscale, "Test function");
    m.def("reassemble_native", &reassemble, "The native image reassembling function.");
    m.def("set_timing_log_enabled", &set_timing_log_enabled, "Set whether the native reassembler logs detailed timings for each step.");
}
