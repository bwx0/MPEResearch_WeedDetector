import time
from enum import Enum

import numpy as np

import native_reassembler_module.native_reassembler_module as nr
from roiyolowd.reassembler import Reassembler
from roiyolowd.util import RectMapping, Rect


class NativeRectSortingMethod(Enum):
    WIDTH_DESC = 1
    HEIGHT_DESC = 2
    AREA_DESC = 3
    MAXSIDE_DESC = 4


class NativeReassembler(Reassembler):
    def __init__(self):
        super().__init__()

    def reassemble(self, srcImg: np.ndarray,
                   initial_width: int = 640,
                   sorting_method: NativeRectSortingMethod = NativeRectSortingMethod.HEIGHT_DESC,
                   autosize: bool = True,
                   border: int = 3,
                   margin: int = 8,
                   native_roi_extractor: nr.NativeROIExtractor = nr.NativeExGIExtractor(25, 2, 1080 // 2, 5, True)) -> np.ndarray:
        if self.reassembled:
            raise Exception("Already reassembled")
        if not isinstance(margin, int):
            raise Exception("Margin should be a single int")
        self.reassembled = True

        st = time.time()
        img, native_mappings = nr.reassemble_native(srcImg, initial_width, sorting_method.value, autosize, border, margin,
                                                    native_roi_extractor)
        # print(f"native time: {((time.time() - st) * 1000.0)}ms")
        self.__set_mappings(native_mappings)

        return img

    def __set_mappings(self, native_mappings):
        for native_rm in native_mappings:
            src = native_rm.src
            dst = native_rm.dst
            src = Rect(src.x, src.y, src.w, src.h)
            if dst is not None:
                dst = Rect(dst.x, dst.y, dst.w, dst.h)
            rm = RectMapping(src, dst)
            self.mappings.append(rm)
