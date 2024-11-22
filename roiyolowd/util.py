import time
from typing import Tuple, Optional, List

import cv2
import numpy as np


def rgba2ExGI(rgba):
    """
    Calculate Excess Green Index for an rgba image
    """
    r = rgba[:, :, 0].astype(np.int32)
    g = rgba[:, :, 1].astype(np.int32)
    b = rgba[:, :, 2].astype(np.int32)

    exg = g + g - r - b
    np.clip(exg, 0, 255, out=exg)

    return exg.astype(np.uint8)


def bgr2ExGI(bgr):
    """
    Calculate Excess Green Index for a bgr image
    """
    r = bgr[:, :, 2].astype(np.int32)
    g = bgr[:, :, 1].astype(np.int32)
    b = bgr[:, :, 0].astype(np.int32)

    exg = g + g - r - b
    np.clip(exg, 0, 255, out=exg)

    return exg.astype(np.uint8)


def box_ioa1(box1: np.ndarray, box2: np.ndarray, eps: float = 1e-7):
    """
    Calculate intersection-over-areaOfBox1 (IoA1) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (np.ndarray): An array of shape (N, 4) representing N bounding boxes.
        box2 (np.ndarray): An array of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        np.ndarray: An NxM array containing the pairwise IoA1 values for every element in box1 and box2.
    """
    if len(box1) == 0 or len(box2) == 0:
        return np.array([])

    # Expand dimensions to prepare for broadcasting
    a1, a2 = np.split(np.expand_dims(box1, axis=1), 2, axis=2)  # Shape: (N, 1, 2)
    b1, b2 = np.split(np.expand_dims(box2, axis=0), 2, axis=2)  # Shape: (1, M, 2)

    # Calculate the intersection area
    inter = np.maximum(0, np.minimum(a2, b2) - np.maximum(a1, b1))
    inter_area = np.prod(inter, axis=2)  # Shape: (N, M)

    # Calculate the area of box1
    area1 = np.prod(a2 - a1, axis=2) + eps  # Shape: (N, 1)

    # Compute Intersection over Area of box1
    ioa1 = inter_area / area1

    return ioa1


class Rect:
    __slots__ = ('x', 'y', 'w', 'h')

    def __init__(self, x: int, y: int, w: int, h: int):
        self.x: int = x
        self.y: int = y
        self.w: int = w
        self.h: int = h

    def contains(self, x0: int, y0: int) -> bool:
        return self.x <= x0 <= self.x + self.w and self.y <= y0 <= self.y + self.h

    def contains_rect(self, other: 'Rect') -> bool:
        return self.contains(other.x, other.y) and self.contains(other.x + other.w, other.y + other.h)

    def intersects(self, other: 'Rect') -> bool:
        return not (self.x + self.w < other.x or
                    other.x + other.w < self.x or
                    self.y + self.h < other.y or
                    other.y + other.h < self.y)

    def merge(self, other: 'Rect') -> 'Rect':
        new_x = min(self.x, other.x)
        new_y = min(self.y, other.y)
        new_w = max(self.right, other.right) - new_x
        new_h = max(self.bottom, other.bottom) - new_y
        return Rect(new_x, new_y, new_w, new_h)

    def __repr__(self):
        return f'({self.x}, {self.y}, {self.w}, {self.h})'

    @property
    def pt1(self) -> Tuple[int, int]:
        return self.x, self.y

    @property
    def pt2(self) -> Tuple[int, int]:
        return self.x + self.w, self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @classmethod
    def from_tuple(cls, t: Tuple[int, int, int, int]):
        return Rect(t[0], t[1], t[2], t[3])

    def __iter__(self):
        for e in [self.x, self.y, self.w, self.h]:
            yield e


class WeedLabel:
    __slots__ = ('rect', 'cls', 'conf')

    def __init__(self, rect: Rect, cls: int, conf: float = 1.0):
        self.rect = rect
        self.cls = cls
        self.conf = conf

    def __repr__(self):
        return f"[{self.rect} cls={self.cls} conf={self.conf}]"


class RectMapping:
    __slots__ = ('src', 'dst')

    def __init__(self, src: Rect, dst: Optional[Rect]):
        self.src = src
        self.dst = dst

    def __repr__(self):
        return f'({self.src} -> {self.dst})'


def draw_boxes(img: np.ndarray, rects: List[Rect], color=(255, 0, 255)):
    """
    Draw rectangles on an image using the specified color.
    """
    for rect in rects:
        p1 = (rect.x, rect.y)
        p2 = (rect.x + rect.w, rect.y + rect.h)
        cv2.rectangle(img, p1, p2, color, 2)


def draw_labels(img: np.ndarray, wls: List[WeedLabel], color=(255, 0, 255)):
    """
    Draw weed labels on an image using the specified color.
    """
    for wl in wls:
        rect = wl.rect
        p1 = (rect.x, rect.y)
        p2 = (rect.x + rect.w, rect.y + rect.h)
        cv2.putText(img, f"{wl.cls}", p1, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
        cv2.rectangle(img, p1, p2, color, 2)


def split_list_randomly(lst, n: int):
    """
    Randomly split a list into n sublists that do not overlap.
    Args:
        lst: The list to split
        n: how many sublists to split into

    Examples:
        [1, 2, 3, 4, 5, 6, 7] -> [[5, 1, 6], [4, 2], [7, 3]]
    """
    np.random.shuffle(lst)
    sublists = np.array_split(lst, n)
    sublists = [sublist.tolist() for sublist in sublists]
    return sublists


def sample_rects(image_width: int, image_height: int, forbidden_area: List[Rect], scale=20, num_samples=50):
    """
    Sample a number of rectangles without overlapping with the given ones.
    Args:
        image_width:
        image_height:
        forbidden_area: do not overlap with these rectangles
        scale: mean value of the length distribution
        num_samples:

    Returns:

    """
    sampled_rects = []

    while len(sampled_rects) < num_samples:
        w = max(min(int(np.random.exponential(scale=scale)), 120), 6)
        h = max(min(int(np.random.exponential(scale=scale)), 120), 6)

        # Ensure the rectangle fits within the image boundaries
        max_x = image_width - w
        max_y = image_height - h

        if max_x <= 0 or max_y <= 0:
            continue

        # Sample a random top-left corner within the allowed range
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        new_rect = Rect(x, y, w, h)

        # Check if the new rectangle intersects with any given rectangle
        if all(not new_rect.intersects(rect) for rect in forbidden_area):
            sampled_rects.append(new_rect)

    return sampled_rects


class Stopwatch:
    """
    A timer that prints something when it is stopped.
    This simplifies timing code sections.
    Usage:
        sw = Stopwatch()
        sw.start() # optional
        # your code
        sw.stop("label")
    """

    def __init__(self):
        self.startT = 0
        self.start()

    def start(self):
        self.startT = time.time()

    def stop(self, name: str = None):
        el_ms = int(1000 * (time.time() - self.startT))
        if name is not None:
            print(f"[{name}] {el_ms}ms")
        self.start()
        return el_ms


def intersection_area(bbox1: Rect, bbox2: Rect) -> float:
    """
    Intersection area between two rectangles.
    """
    inter_x_min = max(bbox1.x, bbox2.x)
    inter_y_min = max(bbox1.y, bbox2.y)
    inter_x_max = min(bbox1.x + bbox1.w, bbox2.x + bbox2.w)
    inter_y_max = min(bbox1.y + bbox1.h, bbox2.y + bbox2.h)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    return inter_area


def calculate_iou(bbox1: Rect, bbox2: Rect) -> float:
    """
    Intersection over union between two rectangles.
    """
    inter_area = intersection_area(bbox1, bbox2)
    bbox1_area = bbox1.w * bbox1.h
    bbox2_area = bbox2.w * bbox2.h

    iou = inter_area / (bbox1_area + bbox2_area - inter_area)
    return iou


def merge_overlapping_rectangles2(rects: List[Rect]) -> List[Rect]:
    """
    Any two overlapping rectangles are merged into their minimum bounding rectangle.
    This is the initial version written by ChatGPT, but it turned out to be faulty.
    After applying some fix it is now working correctly, but it is best-case O(N^3)
    which is really slow even if the input is just a few tens of rects.
    """
    if not rects:
        return []
    rects = rects.copy()
    merged: List[Rect] = []
    while rects:
        current = rects.pop(0)

        tmp_merged = []
        for i, rect in enumerate(merged):
            if rect.intersects(current):
                current = rect.merge(current)
            else:
                tmp_merged.append(rect)
        tmp_merged.append(current)
        merged = tmp_merged

        # After merging, recheck the merged list for any further overlaps
        changed = True
        while changed:
            i = 0
            changed = False
            while i < len(merged):
                j = i + 1
                while j < len(merged):
                    if merged[i].intersects(merged[j]):
                        merged[i] = merged[i].merge(merged[j])
                        merged.pop(j)
                        changed = True
                    else:
                        j += 1
                i += 1

    return merged


def merge_overlapping_rectangles(rects0: List[Rect]) -> List[Rect]:
    """
    Any two overlapping rectangles are merged into their minimum bounding rectangle.
    This method is best-case O(N^2) and worst-case O(N^2.5), good enough for most cases.
    """
    rects = sorted(rects0, key=lambda r: r.x)  # sort the array but idk if it helps at all.
    q1, q2 = [], []

    for u in rects:
        changed = True
        while changed:
            changed = False
            for v in q1:
                if u.intersects(v):
                    u = u.merge(v)
                    changed = True
                else:
                    q2.append(v)
            q1.clear()
            q1, q2 = q2, q1

        q1.append(u)
    return q1
