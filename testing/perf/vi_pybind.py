import cv2
import numpy as np

from util import benchmark, add_dll_dirs

add_dll_dirs()
import perf_cpp as n

exgi_thr = 25
hsv_h_lo, hsv_h_hi = 35, 85
hsv_s_lo, hsv_s_hi = 40, 225
hsv_v_lo, hsv_v_hi = 50, 200


def check_results_equal():
    img = cv2.imread("1080p.png")

    # Some of the methods are problematic in that they do not produce the 100% accurate results.
    # Those methods are not included in the final design, but we still evaluate their performance.
    # Implementation 5,6,7,8,9 are the most efficient and are adopted in the final design. We will
    # check only their equal-ness here.
    for i in range(5, 10):
        img_i = n.ExGI_binarize(img, exgi_thr, i)
        img_i1 = n.ExGI_binarize(img, exgi_thr, i - 1)
        assert np.all(img_i == img_i1), f"exgi_{i - 1} != exgi_{i}"

    for i in range(1, 4):
        img_i = n.HSV_binarize(img, hsv_h_lo, hsv_h_hi, hsv_s_lo, hsv_s_hi, hsv_v_lo, hsv_v_hi, i)
        img_i1 = n.HSV_binarize(img, hsv_h_lo, hsv_h_hi, hsv_s_lo, hsv_s_hi, hsv_v_lo, hsv_v_hi, i - 1)
        assert np.all(img_i == img_i1), f"hsv_{i - 1} != hsv_{i}"

    print("Results are identical.")


def do_benchmarking():
    img = cv2.imread("1080p.png")

    # RGB2HSV
    def RGB2HSV(py_srcImg: np.ndarray):
        return n.RGB2HSV(py_srcImg)

    benchmark(RGB2HSV, test_name="bind.RGB2HSV", py_srcImg=img)

    # ExGI-based
    def ExGI_binarize(py_srcImg: np.ndarray, threshold: int, impl_index: int):
        return n.ExGI_binarize(py_srcImg, threshold, impl_index)

    benchmark(ExGI_binarize, test_name="bind.ExGI_binarize_1", py_srcImg=img, threshold=exgi_thr, impl_index=0)
    benchmark(ExGI_binarize, test_name="bind.ExGI_binarize_2", py_srcImg=img, threshold=exgi_thr, impl_index=1)
    benchmark(ExGI_binarize, test_name="bind.ExGI_binarize_3", py_srcImg=img, threshold=exgi_thr, impl_index=2)
    benchmark(ExGI_binarize, test_name="bind.ExGI_binarize_4", py_srcImg=img, threshold=exgi_thr, impl_index=3)
    benchmark(ExGI_binarize, test_name="bind.ExGI_binarize_5", py_srcImg=img, threshold=exgi_thr, impl_index=4)
    benchmark(ExGI_binarize, test_name="bind.ExGI_binarize_6", py_srcImg=img, threshold=exgi_thr, impl_index=5)
    benchmark(ExGI_binarize, test_name="bind.ExGI_binarize_7", py_srcImg=img, threshold=exgi_thr, impl_index=6)
    benchmark(ExGI_binarize, test_name="bind.ExGI_binarize_8", py_srcImg=img, threshold=exgi_thr, impl_index=7)
    benchmark(ExGI_binarize, test_name="bind.ExGI_binarize_9", py_srcImg=img, threshold=exgi_thr, impl_index=8)
    benchmark(ExGI_binarize, test_name="bind.ExGI_binarize_10", py_srcImg=img, threshold=exgi_thr, impl_index=9)
    benchmark(ExGI_binarize, test_name="bind.ExGI_binarize_5_nosplit", py_srcImg=img, threshold=exgi_thr, impl_index=10)

    # HSV-based
    def HSV_binarize(py_srcImg: np.ndarray,
                     H_lo: int, H_hi: int, S_lo: int, S_hi: int, V_lo: int, V_hi: int,
                     impl_index: int):
        return n.HSV_binarize(py_srcImg, H_lo, H_hi, S_lo, S_hi, V_lo, V_hi, impl_index)

    benchmark(HSV_binarize, test_name="bind.HSV_binarize_1", py_srcImg=img, impl_index=0,
              H_lo=hsv_h_lo, H_hi=hsv_h_hi, S_lo=hsv_s_lo, S_hi=hsv_s_hi, V_lo=hsv_v_lo, V_hi=hsv_v_hi)
    benchmark(HSV_binarize, test_name="bind.HSV_binarize_2", py_srcImg=img, impl_index=1,
              H_lo=hsv_h_lo, H_hi=hsv_h_hi, S_lo=hsv_s_lo, S_hi=hsv_s_hi, V_lo=hsv_v_lo, V_hi=hsv_v_hi)
    benchmark(HSV_binarize, test_name="bind.HSV_binarize_3", py_srcImg=img, impl_index=2,
              H_lo=hsv_h_lo, H_hi=hsv_h_hi, S_lo=hsv_s_lo, S_hi=hsv_s_hi, V_lo=hsv_v_lo, V_hi=hsv_v_hi)
    benchmark(HSV_binarize, test_name="bind.HSV_binarize_4", py_srcImg=img, impl_index=3,
              H_lo=hsv_h_lo, H_hi=hsv_h_hi, S_lo=hsv_s_lo, S_hi=hsv_s_hi, V_lo=hsv_v_lo, V_hi=hsv_v_hi)


if __name__ == "__main__":
    check_results_equal()
    do_benchmarking()
