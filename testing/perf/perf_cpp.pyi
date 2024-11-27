import os

import numpy as np


def ExGI_binarize(py_srcImg: np.ndarray, threshold: int, impl_index: int) -> np.ndarray:
    ...


def RGB2HSV(py_srcImg: np.ndarray) -> np.ndarray:
    ...


def HSV_binarize(py_srcImg: np.ndarray,
                 H_lo: int, H_hi: int, S_lo: int, S_hi: int, V_lo: int, V_hi: int,
                 impl_index: int) -> np.ndarray:
    ...
