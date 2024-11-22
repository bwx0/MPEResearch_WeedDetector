import cv2
import numpy as np

from roiyolowd.util import rgba2ExGI

image = cv2.imread("../test_data/owl.png")
exg = rgba2ExGI(image)
gray = exg.copy()
_, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 10:
        cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)
pr = (np.hstack((
    cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR),
    image)))
cv2.imshow('owl', pr)
cv2.waitKey(1000000)
