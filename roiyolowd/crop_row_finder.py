import time

import cv2
import numpy as np


# This is garbage, just don't use it
class CropRowFinder:
    def __init__(self):
        self.ed_last = None
        self.resize_factor = 4

    def find(self, current_frame: np.ndarray) -> np.ndarray:
        original_shape = current_frame.shape
        frame = cv2.resize(current_frame, (0, 0), fx=1 / self.resize_factor, fy=1 / self.resize_factor)
        if self.ed_last is None:
            self.ed_last = np.zeros_like(frame)
        self.ed_last = np.add(np.multiply(self.ed_last, 0.96), np.multiply(frame, 0.12))
        self.ed_last = np.clip(self.ed_last, 0, 500)
        clipped = np.clip(self.ed_last, 0, 255).astype(np.uint8)

        _, b12 = cv2.threshold(clipped, 220, 255, cv2.THRESH_BINARY)
        # b13 = cv2.morphologyEx(b12, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        b13 = cv2.morphologyEx(b12, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        result = cv2.resize(b13, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
        return result

    def apply(self, current_frame: np.ndarray) -> np.ndarray:  # unfinished
        mask = self.find(current_frame)
        cv2.imshow("masksss", mask)
        ord = cv2.bitwise_or(current_frame, mask)
        print("================")
        st = time.time()
        non_zero_indices = np.nonzero(mask)
        first_non_zero_index = list(zip(non_zero_indices[0], non_zero_indices[1]))
        for i, j in first_non_zero_index:
            if mask[i, j] == 0:
                continue
            mask = cv2.floodFill(mask, mask=None, seedPoint=[j, i], newVal=0)
            ord = cv2.floodFill(ord, mask=None, seedPoint=[j, i], newVal=0)
            print(f"fill {i} {j}")
        print(f"end {time.time() - st}")
        return ord
