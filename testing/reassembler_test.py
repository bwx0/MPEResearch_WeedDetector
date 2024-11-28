from typing import List

import cv2
import numpy as np

import native_reassembler_module.native_reassembler_module as nr
from roiyolowd.reassembler import RectSorting, ResizablePacker, Reassembler, create_reassembler
from roiyolowd.util import RectMapping, Rect

rects_str = "99x39\n16x16\n16x16\n59x57\n30x35\n17x22\n16x16\n96x257\n17x16\n66x79\n26x22\n16x22\n16x16\n16x16\n16x16\n24x26\n77x72\n16x16\n16x16\n16x16\n41x67\n19x19\n16x16\n16x16\n105x116\n50x44\n16x16\n17x16\n33x39\n26x26\n22x19\n16x16\n16x16\n123x149\n23x26\n16x16\n57x96\n16x16\n16x16\n16x16\n22x37\n16x16\n30x16\n59x127\n19x17\n16x16\n17x17\n57x57\n48x19\n16x16\n16x16\n68x50\n16x17\n16x16\n16x16\n61x22\n22x24\n35x28\n16x17\n19x24\n16x28\n16x16\n24x41\n16x16\n19x16\n16x16\n16x22\n28x39\n19x16\n26x37\n22x26\n24x19\n16x17\n16x16\n16x16\n26x41\n24x41\n46x57\n16x19\n16x24\n44x90\n20x16\n41x44\n30x132\n72x59\n28x24\n24x46\n16x16\n19x17\n16x17\n16x16\n26x28\n16x16\n17x17\n16x16\n19x24\n35x39\n88x92\n28x41\n37x35\n83x22\n26x36\n39x17\n17x16\n16x16\n16x16\n16x16\n16x16\n16x16\n16x16\n16x16\n16x16\n16x16\n16x24\n28x28\n44x66\n17x16\n16x16\n41x28\n16x16\n24x26\n16x30\n16x16\n16x16\n24x19\n19x24\n44x66\n16x39\n24x33\n16x17\n72x341\n16x16\n22x50\n16x22\n125x66\n16x16\n16x17\n16x16\n16x16\n17x24\n114x81\n16x24\n16x16\n35x50\n16x26\n16x16\n16x16\n16x16\n22x19\n41x24\n19x16\n17x16\n17x17\n48x33\n16x16\n30x41\n37x59\n24x16\n16x17\n16x16\n19x17\n16x16\n16x16\n22x30\n19x16\n70x33\n16x16\n16x16\n19x22\n39x44\n30x24\n17x19\n24x17\n30x33\n41x50\n16x17\n16x16\n26x17\n44x28\n19x26\n16x16\n16x16\n16x16\n16x16\n16x16\n16x16\n16x16\n17x16\n16x16\n16x16"


def draw_rects_bound(canvas_width: int, canvas_height: int, rect_mappings: List[RectMapping]) -> np.ndarray:
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    for rect_map in rect_mappings:
        if rect_map.dst:
            top_left = (rect_map.dst.x, rect_map.dst.y)
            bottom_right = (rect_map.dst.x + rect_map.dst.w, rect_map.dst.y + rect_map.dst.h)

            col = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

            cv2.rectangle(canvas, top_left, bottom_right, col, 1)

    return canvas


def testPacker():
    lns = [line.split("x") for line in rects_str.split("\n")]
    rects = [Rect(0, 0, int(wh[0]), int(wh[1])) for wh in lns]
    print(rects)
    sorted_rects = sorted(rects, key=RectSorting.HEIGHT_DESC, reverse=True)

    packer = ResizablePacker()
    mapping = packer.fit(sorted_rects)

    img = draw_rects_bound(640, 640, mapping)
    cv2.imshow("rects", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def testRev():
    ra = Reassembler()
    ra.addRect(Rect(0, 0, 25, 30))
    ra.addRect(Rect(80, 100, 10, 100))
    ra.addRect(Rect(30, 30, 20, 20))
    ra.addRect(Rect(150, 150, 80, 50))
    ra.reassemble(np.zeros((500, 500, 3)))
    img = draw_rects_bound(640, 640, ra.mappings)
    print(ra.reverse_mapping(
        [Rect(10, 10, 10, 10), Rect(5, 20, 2, 2), Rect(91, 5, 2, 2), Rect(118, 5, 2, 2), Rect(200, 5, 2, 2)]))
    cv2.imshow("rects", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def reassemble(image):
    ra = create_reassembler(use_native=True)
    rf = ra.reassemble(image, use_resizable_packer=True)
    cv2.imshow("ra", rf)
    cv2.waitKey(0)


if __name__ == "__main__":
    # testPacker()
    # testRev()
    nr.set_timing_log_enabled(False)

    image = cv2.imread(r"../data/d2_i_frames_0570.png")
    reassemble(image)
