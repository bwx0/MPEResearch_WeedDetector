from abc import ABC
from typing import Tuple, List, Optional

import numpy as np

from roiyolowd.util import RectMapping


class NativeROIExtractor(ABC):
    ...


class NativeExGIExtractor(NativeROIExtractor):
    def __init__(self, exgi_threshold: int = 25, scale_factor: int = 2, max_size: int = 1080 * 0.4, min_size: int = 5, merge_overlapping_rects = True):
        ...


def test_to_grayscale(img: np.ndarray) -> Tuple[np.ndarray, List[RectMapping]]:
    """
    Process an image and return a tuple containing:
    - Processed image as a numpy array
    - List of RectMapping instances, each containing a source and destination Rect.

    Parameters:
        img (np.ndarray): Input image in the form of a NumPy array with dtype uint8.

    Returns:
        Tuple[np.ndarray, List[RectMapping]]
    """
    ...


def reassemble_native(srcImg: np.ndarray,
                      initial_width: int,
                      sorting_method: int,
                      autosize: bool,
                      border: int,
                      margin: int,
                      native_roi_extractor: NativeROIExtractor) -> Tuple[np.ndarray, List[RectMapping]]:
    ...

def set_timing_log_enabled(enabled: bool) -> None:
    """
    Set whether the native reassembler logs detailed timings for each step.
    """
    ...