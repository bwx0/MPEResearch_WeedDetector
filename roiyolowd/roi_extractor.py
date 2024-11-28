from abc import ABC, abstractmethod
from typing import List

import cv2
import numpy as np

from roiyolowd.util import Rect, merge_overlapping_rectangles, bgr2ExGI


class ROIExtractor(ABC):
    @abstractmethod
    def extract_roi(self, image: np.array) -> List[Rect]:
        pass


class ExGIGreenExtractor(ROIExtractor):
    def __init__(self, exgi_threshold: int = 25, max_size: int = 1080 * 0.4, min_size: int = 5,
                 merge_overlapping_rects: bool = True):
        """
        Args:
            exgi_threshold:
            max_size: The maximum width or height an ROI tile can have to still be considered valid; otherwise, it will be considered as part of a crop row.
            min_size: The minimum width or height an ROI tile must have to still be considered valid; otherwise, it will be considered as noise.
            merge_overlapping_rects: Merge overlapping ROIs or not. Merging overlapping ROIs can prevent the same plant from showing up in multiple ROIs.
        """
        self.exgi_threshold = exgi_threshold
        self.max_size = max_size
        self.size_threshold = min_size
        self.merge_overlapping_rects = merge_overlapping_rects

    def extract_roi(self, bgr_img: np.ndarray) -> List[Rect]:
        return self.extract_green_regions_bgr(bgr_img)

    def extract_green_regions_bgr(self, image: np.ndarray) -> List[Rect]:
        exg = bgr2ExGI(image)

        _, b1 = cv2.threshold(exg, self.exgi_threshold, 255, cv2.THRESH_BINARY)

        # cv2.imshow("a", b1)
        # ba2=crf.apply(b1.copy())
        # cv2.imshow("a2", ba2)

        # cr = crf.find(b1)
        # b1 = cv2.bitwise_and(b1, cv2.bitwise_not(cr))

        # cv2.imshow("b1", b1)

        # Use the close operation to clean up noise and connect leaves to the main stem, especially if they're
        # separated due to thin or faint connections in the picture.
        # The kernel size of 20 was chosen somewhat arbitrarily, assuming the image resolution is 1080p.
        b2 = cv2.morphologyEx(b1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20)))
        # cv2.imshow("b2", b2)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(b2, connectivity=8)

        result: List[Rect] = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            is_too_small = w < self.size_threshold or h < self.size_threshold
            is_crop_row = w > self.max_size or h > self.max_size
            if is_too_small or is_crop_row:
                continue
            result.append(Rect(x, y, w, h))

        if self.merge_overlapping_rects:
            result = merge_overlapping_rectangles(result)
        return result
