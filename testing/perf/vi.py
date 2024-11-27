import cv2
import numpy as np

from util import add_dll_dirs

add_dll_dirs()
import perf_cpp as pb
import vi_python as py

exgi_thr = 25
hsv_h_lo, hsv_h_hi = 35, 85
hsv_s_lo, hsv_s_hi = 40, 225
hsv_v_lo, hsv_v_hi = 50, 200


def check_results_equal():
    img = cv2.imread("1080p.png")

    # The equal-ness of python implementations are already checked, so are the
    # pybind ones. So we will just pick one from each impl type as the representative
    # and check if they are equal.
    exgi_pb_5 = pb.ExGI_binarize(img, exgi_thr, 5)
    exgi_py_1 = py.ExGI_binarize1(img, exgi_thr)
    assert np.all(exgi_pb_5 == exgi_py_1)

    hsv_pb_1 = pb.HSV_binarize(img, hsv_h_lo, hsv_h_hi, hsv_s_lo, hsv_s_hi, hsv_v_lo, hsv_v_hi, 0)
    hsv_py_3 = py.HSV_binarize3(img, hsv_h_lo, hsv_h_hi, hsv_s_lo, hsv_s_hi, hsv_v_lo, hsv_v_hi)
    assert np.all(hsv_pb_1 == hsv_py_3)

    print("Binarized vegetation indices produced by all implementations are equal!")

if __name__ == '__main__':
    check_results_equal()
