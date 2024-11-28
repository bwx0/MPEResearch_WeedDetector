import cv2
import numpy as np

from util import benchmark
from vi_python import ExGI_binarize1

"""
Benchmark some image processing functions.
Most parameters are copied from this weed detection project or OWL.
So here I intend to evaluate the performance of certain steps in and my project.  
Since `cv2.***()` functions are basically wrappers for optimized C++ code,  
testing them in Python should be good enough.
"""

def ExGI(img: np.ndarray):
    b = img[:, :, 0].astype(np.int32)
    g = img[:, :, 1].astype(np.int32)
    r = img[:, :, 2].astype(np.int32)

    exg = g + g - r - b
    np.clip(exg, 0, 255, out=exg)

    return exg.astype(np.uint8)


def do_benchmarking():
    img = cv2.imread("1080p.png")
    exgi = ExGI(img)
    exgibin = ExGI_binarize1(img, threshold=20)

    print(f"number of contours={len(cv2.findContours(exgibin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])}")

    def find_contour(img):
        cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def threshold(img):
        cv2.threshold(img, 25, 255, cv2.THRESH_BINARY)

    def adaptiveThreshold(img):  # params copied from OWL
        cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 2)

    def morph_close(img, kernel):
        cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

    def find_cc(img, connectivity):
        cv2.connectedComponents(img, connectivity=connectivity)

    def find_cc_with_stats(img, connectivity):
        cv2.connectedComponentsWithStats(img, connectivity=connectivity)

    kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_7x7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_11x11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_15x15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_20x20 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    kernel_21x21 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

    benchmark(find_contour, test_name="cv.find_contour", img=exgibin)
    benchmark(threshold, test_name="cv.threshold", img=exgi)
    benchmark(adaptiveThreshold, test_name="cv.adaptiveThreshold", img=exgi)
    benchmark(morph_close, test_name="cv.morphologyEx(close,3x3)", img=exgibin, kernel=kernel_3x3)
    benchmark(morph_close, test_name="cv.morphologyEx(close,7x7)", img=exgibin, kernel=kernel_7x7)
    benchmark(morph_close, test_name="cv.morphologyEx(close,11x11)", img=exgibin, kernel=kernel_11x11)
    benchmark(morph_close, test_name="cv.morphologyEx(close,15x15)", img=exgibin, kernel=kernel_15x15)
    benchmark(morph_close, test_name="cv.morphologyEx(close,20x20)", img=exgibin, kernel=kernel_20x20)
    benchmark(morph_close, test_name="cv.morphologyEx(close,21x21)", img=exgibin, kernel=kernel_21x21)
    benchmark(find_cc, test_name="cv.connectedComponents(connectivity=4)", img=exgibin, connectivity=4)
    benchmark(find_cc, test_name="cv.connectedComponents(connectivity=8)", img=exgibin, connectivity=8)
    benchmark(find_cc_with_stats, test_name="cv.connectedComponentsWithStats(connectivity=4)", img=exgibin, connectivity=4)
    benchmark(find_cc_with_stats, test_name="cv.connectedComponentsWithStats(connectivity=8)", img=exgibin, connectivity=8)


if __name__ == "__main__":
    do_benchmarking()
