import time

import cv2
import numpy as np

from util import print_platform_info, benchmark


def div(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


ExGI = lambda R, G, B: (2 * G - R - B)
GLI = lambda R, G, B: div(2 * G - R - B, 2 * G + R + B)
GNDVI = lambda R, G, B: div(G - R, G + R)
VARI = lambda R, G, B: div(G - R, G + R - B)
GRRI = lambda R, G, B: div(G, R) * 255
VEG = lambda R, G, B: div(G, (np.power(R, 0.6666666) * np.power(B, 0.3333333))) * 255
MGRVI = lambda R, G, B: div(G * G - R * R, G * G + R * R)
RGVBI = lambda R, G, B: div(G - B * R, G * G + R * B)

print_platform_info()
rgbimg = cv2.imread("1080p.png")

b, g, r = rgbimg[:, :, 0], rgbimg[:, :, 1], rgbimg[:, :, 2]

benchmark(ExGI, test_name="ExGI", R=r, G=g, B=b)
benchmark(GLI, test_name="GLI", R=r, G=g, B=b)
benchmark(GNDVI, test_name="GNDVI", R=r, G=g, B=b)
benchmark(VARI, test_name="VARI", R=r, G=g, B=b)
benchmark(GRRI, test_name="GRRI", R=r, G=g, B=b)
benchmark(VEG, test_name="VEG", R=r, G=g, B=b)
benchmark(MGRVI, test_name="MGRVI", R=r, G=g, B=b)
benchmark(RGVBI, test_name="RGVBI", R=r, G=g, B=b)

# results on my laptop
"""
[ExGI]	Total runtime: 4651ms   nRuns: 1000    4.651ms/call
[GLI]	Total runtime: 5035ms   nRuns: 198    25.429ms/call
[GNDVI]	Total runtime: 5035ms   nRuns: 260    19.365ms/call
[VARI]	Total runtime: 5028ms   nRuns: 243    20.691ms/call
[GRRI]	Total runtime: 5039ms   nRuns: 236    21.353ms/call
[VEG]	Total runtime: 5147ms   nRuns: 36    142.974ms/call
[MGRVI]	Total runtime: 5034ms   nRuns: 198    25.424ms/call
[RGVBI]	Total runtime: 5044ms   nRuns: 207    24.367ms/call
"""
