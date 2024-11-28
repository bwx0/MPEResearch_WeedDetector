import time
from typing import List, Optional, Tuple

import numpy as np

from roiyolowd.roi_extractor import ROIExtractor, ExGIGreenExtractor
from roiyolowd.util import box_ioa1, Rect, RectMapping, Stopwatch


# Reference:
# https://codeincomplete.com/articles/bin-packing/demo/

class RectSorting:
    """
    A few comparators for sorting Rect
    """
    WIDTH_DESC = lambda rect: rect.w
    HEIGHT_DESC = lambda rect: rect.h
    AREA_DESC = lambda rect: rect.w * rect.h
    MAXSIDE_DESC = lambda rect: max(rect.w, rect.h)


class Block:
    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.used = False
        self.down = None
        self.right = None


class Packer:
    def __init__(self, w: int, h: int):
        self.root: Block = Block(0, 0, w, h)

    def fit(self, rects: List[Rect]):
        result: List[RectMapping] = []
        for rect in rects:
            node = self.find_node(self.root, rect.w, rect.h)
            if node:
                fit = self.split_node(node, rect.w, rect.h)
                fit_rect = Rect(fit.x, fit.y, rect.w, rect.h)
                mapping = RectMapping(rect, fit_rect)
                result.append(mapping)
            else:
                mapping = RectMapping(rect, None)
                result.append(mapping)
        return result

    def find_node(self, root: Block, w: int, h: int) -> Optional[Block]:
        if root.used:
            return self.find_node(root.right, w, h) or self.find_node(root.down, w, h)
        elif w <= root.w and h <= root.h:
            return root
        else:
            return None

    def split_node(self, node: Block, w: int, h: int) -> Block:
        node.used = True
        node.down = Block(node.x, node.y + h, node.w, node.h - h)
        node.right = Block(node.x + w, node.y, node.w - w, h)
        return node


class ResizablePacker:
    def __init__(self):
        self.root: Optional[Block] = None

    def fit(self, rects: List[Rect]):
        if not rects or len(rects) == 0:
            return []

        w = rects[0].w
        h = rects[0].h
        self.root = Block(0, 0, w, h)
        result: List[RectMapping] = []

        for rect in rects:
            node = self.find_node(self.root, rect.w, rect.h)
            fit_rect: Optional[Rect] = None
            if node:
                fit = self.split_node(node, rect.w, rect.h)
                fit_rect = Rect(fit.x, fit.y, rect.w, rect.h)
            else:
                fit = self.grow_node(rect.w, rect.h)
                if fit:
                    fit_rect = Rect(fit.x, fit.y, rect.w, rect.h)
            mapping = RectMapping(rect, fit_rect)
            result.append(mapping)
        return result

    def find_node(self, root: Block, w: int, h: int) -> Optional[Block]:
        if root.used:
            return self.find_node(root.right, w, h) or self.find_node(root.down, w, h)
        elif w <= root.w and h <= root.h:
            return root
        else:
            return None

    def split_node(self, node: Block, w: int, h: int) -> Block:
        node.used = True
        node.down = Block(node.x, node.y + h, node.w, node.h - h)
        node.right = Block(node.x + w, node.y, node.w - w, h)
        return node

    def grow_node(self, w: int, h: int) -> Optional[Block]:
        can_grow_down = w <= self.root.w
        can_grow_right = h <= self.root.h

        should_grow_right = can_grow_right and (self.root.h >= self.root.w + w)
        should_grow_down = can_grow_down and (self.root.w >= self.root.h + h)

        if should_grow_right:
            return self.grow_right(w, h)
        elif should_grow_down:
            return self.grow_down(w, h)
        elif can_grow_right:
            return self.grow_right(w, h)
        elif can_grow_down:
            return self.grow_down(w, h)
        else:
            return None

    def grow_right(self, w: int, h: int) -> Optional[Block]:
        root_old = self.root
        self.root = Block(0, 0, root_old.w + w, root_old.h)
        self.root.used = True
        self.root.down = root_old
        self.root.right = Block(root_old.w, 0, w, root_old.h)
        node = self.find_node(self.root, w, h)
        if node:
            return self.split_node(node, w, h)
        else:
            return None

    def grow_down(self, w: int, h: int) -> Optional[Block]:
        root_old = self.root
        self.root = Block(0, 0, root_old.w, root_old.h + h)
        self.root.used = True
        self.root.down = Block(0, root_old.h, root_old.w, h)
        self.root.right = root_old
        node = self.find_node(self.root, w, h)
        if node:
            return self.split_node(node, w, h)
        else:
            return None


class Reassembler:
    """
    Usage:
    r = Reassembler()
    r.addRect(...) // optional
    reassembled_img = r.reassemble(img)

    """
    __default_roi_extractor_sentinel = object()

    def __init__(self):
        self.rects: List[Rect] = []
        self.mappings: List[RectMapping] = []
        self.reassembled = False

    def addRect(self, rect: Rect) -> None:
        if self.reassembled:
            raise Exception("Cannot add more rectangles after reassembly.")
        self.rects.append(rect)

    def reassemble(self, srcImg: np.ndarray,
                   packer_width: int = 640,
                   sorting_method: RectSorting = RectSorting.HEIGHT_DESC,
                   use_resizable_packer: bool = True,
                   border_thickness: int = 3,
                   padding_size: int | Tuple[int, int] = 8,
                   roi_extractor: Optional[ROIExtractor] = __default_roi_extractor_sentinel) -> np.ndarray:
        """
        Args:
            srcImg: The input image from which to extract ROIs and reassemble them.
            packer_width: The width for the fixed packer. A reasonable value is the square root of the (estimated) total area of all ROIs.
            If using a resizable packer (`use_resizable_packer=True`), this can be ignored or set to an arbitrary value.
            sorting_method: An appropriate ROI sorting method is required to achieve optimal space utilisation. The default one (HEIGHT_DESC) is good enough for both fixed size and resizable packer.
            use_resizable_packer: Frame size are automatically adjusted during fitting. packer_width does not take effect when use_resizable_packer is True.
            border_thickness: The thickness (in pixels) of the black borders added to all sides of each ROI.
            padding_size: The number of pixels by which to expand each side of the rectangles outward from their center.
            This enlarges the cropped area uniformly while keeping it centered around its original position.
            If a 2-tuple is provided, the padding size is randomized within the specified lower and upper bounds (the 2 numbers in the tuple).
            roi_extractor:

        Returns:
            The reassembled image.

        """
        if self.reassembled:
            raise Exception("Already reassembled")

        sw = Stopwatch()
        # Step 0: Extract green regions
        if roi_extractor is not None:
            if roi_extractor == self.__default_roi_extractor_sentinel:
                roi_extractor = ExGIGreenExtractor()
            green_rects: List[Rect] = roi_extractor.extract_roi(srcImg)
            for rect in green_rects:
                self.addRect(rect)
        sw.stop("extract green")

        # Step 1: Finalise rectangles (add margins and borders)
        # This place is a bit messy, but I don't think it's worth a refactor, because
        # it's still a relatively simple task.
        imgh, imgw = srcImg.shape[:2]
        expanded_rects: List[Rect] = []
        if isinstance(padding_size, int):  # Constant size margin
            extra = padding_size + border_thickness
            for rect in self.rects:
                x1, y1, w, h = rect.x, rect.y, rect.w, rect.h
                x2, y2 = x1 + w, y1 + h
                x1, y1 = max(x1 - extra, 0), max(y1 - extra, 0)
                x2, y2 = min(x2 + extra, imgw), min(y2 + extra, imgh)
                expanded_rects.append(Rect(x1, y1, x2 - x1, y2 - y1))
                rect = expanded_rects[len(expanded_rects) - 1]
        else:  # Margin with size ranging from the given range
            margin_lo, margin_hi = padding_size
            for rect in self.rects:
                x1, y1, w, h = rect.x, rect.y, rect.w, rect.h
                x2, y2 = x1 + w, y1 + h
                mL, mR, mT, mB = np.random.randint(margin_lo, margin_hi, size=4)
                x1, y1 = max(x1 - mL, 0), max(y1 - mT, 0)
                x2, y2 = min(x2 + mR, imgw), min(y2 + mB, imgh)
                expanded_rects.append(Rect(x1, y1, x2 - x1, y2 - y1))
        sw.stop("margins")

        # Step 2: sort rects and pack
        start_t = time.time()
        sorted_rects = sorted(expanded_rects, key=sorting_method, reverse=True)
        if use_resizable_packer:
            packer = ResizablePacker()
        else:
            packer = Packer(packer_width, packer_width)
        mappings = packer.fit(sorted_rects)
        self.mappings = mappings
        fit_el = int((time.time() - start_t) * 1000)
        sw.stop("fit")

        # Step 3: construct the reassembled image
        start_t = time.time()
        result_img = self.__draw_rects(srcImg, mappings, border_thickness)
        draw_el = int((time.time() - start_t) * 1000)
        sw.stop("img")

        # Collect stats
        effective_area = np.sum([i.dst.w * i.dst.h if i.dst else 0 for i in mappings])
        n_fail = np.sum([0 if i.dst else 1 for i in mappings])
        total_area = result_img.shape[0] * result_img.shape[1]
        print(f"{result_img.shape}   areaR={total_area / 640 / 640}   utilisation={effective_area / total_area}"
              f"  n_fail={n_fail}   fit_time={fit_el}ms   draw_time={draw_el}ms")

        self.reassembled = True
        return result_img

    def reverse_map(self, rects: List[Rect]) -> List[RectMapping]:
        """
        Map a list of rects from mapped image space back to original image space
        Args:
            rects:

        Returns:

        """
        if not self.reassembled:
            raise Exception("Not reassembled")

        rect_mapped_space = [(r.x, r.y, r.x + r.w, r.y + r.h) for r in rects]
        rect_reference = [(r.dst.x, r.dst.y, r.dst.x + r.dst.w, r.dst.y + r.dst.h) for r in self.mappings]

        result: List[RectMapping] = self.__map0(rect_mapped_space, rect_reference, rects, rev=True)
        return result

    def forward_map(self, rects: List[Rect]) -> List[RectMapping]:
        """
        Map a list of rects from original image space to reassembled image space
        Args:
            rects:

        Returns:

        """
        if not self.reassembled:
            raise Exception("Not reassembled")

        rect_mapped_space = [(r.x, r.y, r.x + r.w, r.y + r.h) for r in rects]
        rect_reference = [(r.src.x, r.src.y, r.src.x + r.src.w, r.src.y + r.src.h) for r in self.mappings]

        result: List[RectMapping] = self.__map0(rect_mapped_space, rect_reference, rects, rev=False)
        return result

    def __draw_rects(self, srcImg: np.ndarray,
                     rect_mappings: List[RectMapping],
                     border: int = 0) -> np.ndarray:
        """
        Construct a reassembled image using rect_mappings.
        Args:
            srcImg:
            rect_mappings:
            border:

        Returns:

        """
        canvas_width = max([rm.dst.x + rm.dst.w for rm in rect_mappings if rm is not None] + [64])
        canvas_height = max([rm.dst.y + rm.dst.h for rm in rect_mappings if rm is not None] + [64])
        canvas = np.full((canvas_height, canvas_width, 3), 0, dtype=np.uint8)

        for rect_map in rect_mappings:
            if rect_map.dst:
                dst_x_start = rect_map.dst.x + border
                dst_x_end = rect_map.dst.x + rect_map.dst.w - border
                dst_y_start = rect_map.dst.y + border
                dst_y_end = rect_map.dst.y + rect_map.dst.h - border

                src_x_start = rect_map.src.x + border
                src_x_end = rect_map.src.x + rect_map.src.w - border
                src_y_start = rect_map.src.y + border
                src_y_end = rect_map.src.y + rect_map.src.h - border

                canvas[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = srcImg[src_y_start:src_y_end, src_x_start:src_x_end]

        return canvas

    def __map0(self, rect_mapped_space: List[Tuple], rect_reference: List[Tuple], rects: List[Rect], rev: bool):
        result: List[RectMapping] = []
        ious = box_ioa1(np.array(rect_mapped_space), np.array(rect_reference))
        for i in range(0, len(rect_mapped_space)):
            rect = rects[i]
            ci = np.argmax(ious[i])
            candidate = self.mappings[ci]
            iou = ious[i][ci]

            # Choose the ROI that has the most overlapping with the rectangle, and use that ROI as the reference point to
            # map the rectangle location back to the original image.
            for j, rm in enumerate(self.mappings):
                if (rev and rm.dst.contains_rect(rect)) or (not rev and rm.src.contains_rect(rect)):
                    candidate = rm
                    iou = 1
                    break

            if not candidate or iou < 0.6:
                result.append(RectMapping(rect, None))
            else:
                dst = candidate.dst if rev else candidate.src
                src = candidate.src if rev else candidate.dst
                dx = rect.x - dst.x
                dy = rect.y - dst.y
                mapped_rect = Rect(src.x + dx, src.y + dy, rect.w, rect.h)
                result.append(RectMapping(rect, mapped_rect))

        return result


def create_reassembler(use_native: bool = False) -> Reassembler:
    if use_native:
        from roiyolowd.native_reassembler import NativeReassembler
        return NativeReassembler()
    else:
        return Reassembler()
