import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt

from roiyolowd.util import merge_overlapping_rectangles, Rect


def draw(rects, red_rects):
    fig, ax = plt.subplots(figsize=(8, 8))
    r = np.array([[x, y, x + w, x + h] for x, y, w, h in rects])
    for x, y, w, h in rects:
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='blue', facecolor='lightblue')
        ax.add_patch(rect)
    for x, y, w, h in red_rects:
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='red', facecolor='#0000')
        ax.add_patch(rect)
    ax.grid(True)
    ax.set_xlim(np.min(r, axis=0)[0] - 1, np.max(r, axis=0)[2] + 1)
    ax.set_ylim(np.min(r, axis=0)[1] - 1, np.max(r, axis=0)[3] + 1)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()


def test(rects):
    result = merge_overlapping_rectangles(rects)
    print(result)
    draw(rects, result)


test([
    Rect(1, 3, 3, 1),
    Rect(3, 5, 3, 1),
    Rect(5, 7, 3, 1),
    Rect(7, 9, 3, 1),
    Rect(9, 11, 3, 1),
    Rect(11, 13, 3, 1),
    Rect(13, 15, 3, 1),

    Rect(1, 13, 1, 3),
    Rect(3, 11, 1, 3),
    Rect(5, 9, 1, 3),
    Rect(12, 4, 2, 3),
    Rect(11, 5, 1, 3),
    Rect(13, 3, 1, 3),
    Rect(15, 1, 1, 3),
])

test([
    Rect(1, 3, 3, 1),
    Rect(3, 5, 3, 1),
    Rect(5, 7, 3, 1),
    Rect(7, 9, 3, 1),
    Rect(9, 11, 3, 1),
    Rect(11, 13, 3, 1),
    Rect(13, 15, 3, 1),

    Rect(1, 13, 1, 3),
    Rect(3, 11, 1, 3),
    Rect(5, 9, 1, 3),
    Rect(9, 7, 1, 3),
    Rect(11, 5, 1, 3),
    Rect(13, 3, 1, 3),
    Rect(15, 1, 1, 3),
])


test([
    Rect(1, 3, 3, 1),
    Rect(3, 5, 3, 1),
    Rect(5, 7, 3, 1),
    Rect(7, 9, 3, 1),
    Rect(9, 11, 3, 1),
    Rect(11, 13, 3, 1),
    Rect(13, 15, 3, 1),

    Rect(1, 13, 1, 3),
    Rect(3, 11, 1, 3),
    Rect(5, 9, 1, 3),
    Rect(13, 7, 1, 3),
    Rect(11, 5, 1, 3),
    Rect(13, 3, 1, 3),
    Rect(15, 1, 1, 3),
])