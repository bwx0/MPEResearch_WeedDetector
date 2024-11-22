import time

import cv2
import numpy as np

dst_path = "../data/VegIndexTest"


def apply_filter(rgb_image, filter):
    R, G, B = (np.array(rgb_image[:, :, 0], dtype=np.dtype(int)),
               np.array(rgb_image[:, :, 1], dtype=np.dtype(int)),
               np.array(rgb_image[:, :, 2], dtype=np.dtype(int)))
    filtered = filter(R, G, B)
    normalized = cv2.normalize(filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    normalized = np.uint8(normalized)
    return normalized


def div(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


ExGI = lambda R, G, B: (2 * G - R - B)
GLI = lambda R, G, B: div(2 * G - R - B, 2 * G + R + B)
MPRI = lambda R, G, B: div(G - R, G + R)
VARI = lambda R, G, B: div(G - R, G + R - B)
GRRI = lambda R, G, B: div(G, R) * 255
VEG = lambda R, G, B: div(G, (np.power(R, 0.6666666) * np.power(B, 0.3333333))) * 255
MGRVI = lambda R, G, B: div(G * G - R * R, G * G + R * R)
RGVBI = lambda R, G, B: div(G - B * R, G * G + R * B)


def test_filter(rgb_image, filter, name):
    nTests = 100
    start = time.time()
    for i in range(nTests):
        filtered = apply_filter(rgb_image, filter)
    elapsed_time = time.time() - start
    print(f"{name:5}  elapsed_time: {elapsed_time:.4f}s     speed: {1000 * elapsed_time / nTests:.2f}ms/img")


rgbimg = cv2.imread("../test_data/1200x800.png")
test_filter(rgbimg, ExGI, "ExGI")
test_filter(rgbimg, GLI, "GLI")
test_filter(rgbimg, MPRI, "MPRI")
test_filter(rgbimg, VARI, "VARI")
test_filter(rgbimg, GRRI, "GRRI")
test_filter(rgbimg, VEG, "VEG")
test_filter(rgbimg, MGRVI, "MGRVI")
test_filter(rgbimg, RGVBI, "RGVBI")


# results on my laptop
"""
ExGI   elapsed_time: 1.1134s     speed: 11.13ms/img
GLI    elapsed_time: 2.4847s     speed: 24.85ms/img
MPRI   elapsed_time: 2.6723s     speed: 26.72ms/img
VARI   elapsed_time: 2.4669s     speed: 24.67ms/img
GRRI   elapsed_time: 2.1795s     speed: 21.80ms/img
VEG    elapsed_time: 8.8046s     speed: 88.05ms/img
MGRVI  elapsed_time: 2.5258s     speed: 25.26ms/img
RGVBI  elapsed_time: 2.3860s     speed: 23.86ms/img
"""
