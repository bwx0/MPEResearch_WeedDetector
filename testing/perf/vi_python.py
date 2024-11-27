import cv2

import numpy as np

from util import benchmark


# As mentioned here, cv2.split() is more expensive compared to indexing.
# https://stackoverflow.com/questions/19181485/splitting-image-using-opencv-in-python

def ExGI_binarize1(img: np.ndarray, threshold: int = 25):
    """
    Type cast to avoid overflow/underflow: int32
    Channel split: array indices
    Threshold: cv2.threshold()
    """
    b = img[:, :, 0].astype(np.int32)
    g = img[:, :, 1].astype(np.int32)
    r = img[:, :, 2].astype(np.int32)

    exg = g + g - r - b
    np.clip(exg, 0, 255, out=exg)

    exg = exg.astype(np.uint8)
    return cv2.threshold(exg, threshold, 255, cv2.THRESH_BINARY)[1]


def ExGI_binarize2(img: np.ndarray, threshold: int = 25):
    """
    Type cast to avoid overflow/underflow: int32
    Channel split: cv2.split()
    Threshold: cv2.threshold()
    """
    b, g, r = cv2.split(img.astype(np.int32))

    exg = g + g - r - b
    np.clip(exg, 0, 255, out=exg)

    exg = exg.astype(np.uint8)
    return cv2.threshold(exg, threshold, 255, cv2.THRESH_BINARY)[1]


def ExGI_binarize3(img: np.ndarray, threshold: int = 25):
    """
    Type cast to avoid overflow/underflow: int32
    Channel split: array indices
    Threshold: np.where()
    """
    b = img[:, :, 0].astype(np.int32)
    g = img[:, :, 1].astype(np.int32)
    r = img[:, :, 2].astype(np.int32)

    exg = g + g - r - b

    return np.where(exg > threshold, 255, 0).astype(np.uint8)


def ExGI_binarize4(img: np.ndarray, threshold: int = 25):
    """
    Type cast to avoid overflow/underflow: int16
    Channel split: array indices
    Threshold: cv2.threshold()
    """
    b = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    r = img[:, :, 2].astype(np.int16)

    exg = g + g - r - b
    np.clip(exg, 0, 255, out=exg)

    exg = exg.astype(np.uint8)
    return cv2.threshold(exg, threshold, 255, cv2.THRESH_BINARY)[1]


def ExGI_binarize5(img: np.ndarray, threshold: int = 25):
    """
    Type cast to avoid overflow/underflow: int16
    Channel split: cv2.split()
    Threshold: cv2.threshold()
    """
    b, g, r = cv2.split(img.astype(np.int16))

    exg = g + g - r - b
    np.clip(exg, 0, 255, out=exg)

    exg = exg.astype(np.uint8)
    return cv2.threshold(exg, threshold, 255, cv2.THRESH_BINARY)[1]


def ExGI_binarize6(img: np.ndarray, threshold: int = 25):
    """
    Type cast to avoid overflow/underflow: int16
    Channel split: array indices
    Threshold: np.where()
    """
    b = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    r = img[:, :, 2].astype(np.int16)

    exg = g + g - r - b
    np.clip(exg, 0, 255, out=exg)

    return np.where(exg > threshold, 255, 0).astype(np.uint8)


def HSV_binarize1_1p(img: np.ndarray, H_lo: int = 35, H_hi: int = 80):
    """
    bounds for H
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hue = hsv[:, :, 0]
    result = cv2.inRange(hue, H_lo, H_hi)

    return result


def HSV_binarize2(img: np.ndarray, H_lo: int = 35, H_hi: int = 80,
                  S_lo: int = 40, S_hi: int = 225,
                  V_lo: int = 50, V_hi: int = 200):
    """
    bounds for HSV
    non-inplace
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    maskH = cv2.inRange(hue, H_lo, H_hi)
    maskS = cv2.inRange(saturation, S_lo, S_hi)
    maskV = cv2.inRange(value, V_lo, V_hi)

    return maskH & maskS & maskV


def HSV_binarize3(img: np.ndarray, H_lo: int = 35, H_hi: int = 80,
                  S_lo: int = 40, S_hi: int = 225,
                  V_lo: int = 50, V_hi: int = 200):
    """
    bounds for HSV
    partial inplace
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hue, saturation, value = cv2.split(hsv)  # hsv[:, :, n] is not a valid parameter for cv2.inRange

    cv2.inRange(hue, H_lo, H_hi, hue)
    cv2.inRange(saturation, S_lo, S_hi, saturation)
    cv2.inRange(value, V_lo, V_hi, value)

    return hue & saturation & value


def rgb2hsv(img):
    # The HSV conversion here is already optimized since cv2.cvtColor()
    # is essentially a wrapper around the underlying highly optimized C++ code,
    # and we aren't performing any operations other than the conversion itself.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv


def rgb2hsv_inplace(img):
    cv2.cvtColor(img, cv2.COLOR_BGR2HSV, img)


def preview():
    img = cv2.imread("1080p.png")
    exgi = ExGI_binarize1(img)
    img1 = HSV_binarize1_1p(img)
    img2 = HSV_binarize2(img)
    cv2.imshow("ExGI3", cv2.resize(ExGI_binarize3(img), dsize=(0, 0), fx=0.5, fy=0.5))
    cv2.imshow("ExGI", cv2.resize(exgi, dsize=(0, 0), fx=0.5, fy=0.5))
    cv2.imshow("HSV1p", cv2.resize(img1, dsize=(0, 0), fx=0.5, fy=0.5))
    cv2.imshow("HSV2", cv2.resize(img2, dsize=(0, 0), fx=0.5, fy=0.5))
    cv2.waitKey(0)


def check_results_equal():
    img = cv2.imread("1080p.png")

    # ExGI
    exg1 = ExGI_binarize1(img)
    exg2 = ExGI_binarize2(img)
    exg3 = ExGI_binarize3(img)
    exg4 = ExGI_binarize4(img)
    exg5 = ExGI_binarize5(img)
    exg6 = ExGI_binarize6(img)

    assert np.all(exg1 == exg2)
    assert np.all(exg2 == exg3)
    assert np.all(exg3 == exg4)
    assert np.all(exg4 == exg5)
    assert np.all(exg5 == exg6)

    # HSV
    hsv2 = ExGI_binarize2(img)
    hsv3 = ExGI_binarize3(img)
    assert np.all(hsv2 == hsv3)

    print("Results are identical.")


def do_benchmarking():
    img = cv2.imread("1080p.png")

    # ExGI-based
    benchmark(ExGI_binarize1, test_name="ExGI_binarize1", img=img)
    benchmark(ExGI_binarize2, test_name="ExGI_binarize2", img=img)
    benchmark(ExGI_binarize3, test_name="ExGI_binarize3", img=img)
    benchmark(ExGI_binarize4, test_name="ExGI_binarize4", img=img)
    benchmark(ExGI_binarize5, test_name="ExGI_binarize5", img=img)
    benchmark(ExGI_binarize6, test_name="ExGI_binarize6", img=img)

    # HSV-based
    benchmark(HSV_binarize1_1p, test_name="HSV_binarize1_1p", img=img)
    benchmark(HSV_binarize2, test_name="HSV_binarize2", img=img)
    benchmark(HSV_binarize3, test_name="HSV_binarize3", img=img)

    # RGB2HSV
    benchmark(rgb2hsv, test_name="rgb2hsv", img=img)
    benchmark(rgb2hsv_inplace, test_name="rgb2hsv_inplace", img=img)


if __name__ == "__main__":
    # preview()
    check_results_equal()
    do_benchmarking()
