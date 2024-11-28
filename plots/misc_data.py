import os
from pathlib import Path
from pprint import pprint

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from roiyolowd.evaluation import WeedDetectorEvaluator
from roiyolowd.reassembler import Reassembler
from roiyolowd.roi_extractor import ExGIGreenExtractor
from roiyolowd.util import intersection_area


def save_plot(path, img):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, img)


def save_plt_plot(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight')


def process_box_size(img_path, label_path):
    image = Image.open(img_path)

    print(f"Processing image: {img_path}")
    print(f"Image size: {image.size}")

    with open(label_path, 'r') as label_file:
        labels = label_file.readlines()

    result = []
    cls_distr = [0] * 10

    for label in labels:
        lbl = label.replace("\n", "").split(" ")
        lbl = [float(x) for x in lbl]
        c, x, y, w, h = lbl
        wpx = int(image.size[0] * w)
        hpx = int(image.size[1] * h)
        result.append((wpx, hpx))
        cls_distr[int(c)] += 1
    return result, cls_distr


def box_size(path):
    images_dir = Path(path, "images")
    labels_dir = Path(path, "labels")
    image_files = sorted([f for f in os.listdir(images_dir)])
    label_files = sorted([f for f in os.listdir(labels_dir)])

    result = []
    cls_distr = {}
    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]
        img_path = os.path.join(images_dir, image_file)
        src_vid_name = base_name.split("-")[1].split("_")[0]
        label_path = os.path.join(labels_dir, base_name + '.txt')
        if os.path.exists(label_path):
            r, cd = process_box_size(img_path, label_path)
            result.extend(r)
            old_cd = cls_distr.get(src_vid_name, np.zeros(len(cd)))
            cls_distr[src_vid_name] = old_cd + np.array(cd)
        else:
            print(f"Warning: No corresponding label for image {img_path}")
    # print(result)
    pprint(cls_distr)

    x, y = zip(*result)

    # Plot the heatmap
    plt.figure(figsize=(6, 5))
    plt.hist2d(x, y, cmap='Blues', bins=(80, 80))
    plt.xlim(8, 150)
    plt.ylim(8, 150)
    plt.colorbar(label='Frequency')
    plt.title('Distribution of Ground Truth Label Sizes')
    plt.xlabel('Width (in pixels)')
    plt.ylabel('Height (in pixels)')
    save_plt_plot("out/gt_lbl_sz_heat.png")
    plt.show()

    sz = [w * h for w, h in result]
    plt.figure(figsize=(6, 5))
    plt.hist(sz, bins=200)
    plt.title('Distribution of Ground Truth Label Sizes')
    plt.xlim(100, 13000)
    plt.xlabel('Area (in pixels)')
    plt.ylabel('Frequency')
    save_plt_plot("out/gt_lbl_sz_hist.png")
    plt.show()


def process_illustration(path):
    out_dir = "out/illustration"
    img = cv2.imread(path)
    os.makedirs(out_dir, exist_ok=True)

    # original
    save_plot(f"{out_dir}/image.png", img)

    # rgb
    b, g, r = cv2.split(img)
    save_plot(f"{out_dir}/rgb_r.png", r)
    save_plot(f"{out_dir}/rgb_g.png", g)
    save_plot(f"{out_dir}/rgb_b.png", b)

    # exgi
    b, g, r = cv2.split(img.astype(np.int32))
    exgi = np.clip(g + g - b - r, 0, 255).astype(np.uint8)
    save_plot(f"{out_dir}/exgi.png", exgi)

    # bin
    _, exgi_bin = cv2.threshold(exgi, 25, 255, cv2.THRESH_BINARY)
    save_plot(f"{out_dir}/exgi_bin.png", exgi_bin)

    # green regions
    rects = ExGIGreenExtractor().extract_roi(img)
    exgi_b_rgb = cv2.cvtColor(exgi_bin, cv2.COLOR_GRAY2RGB)
    for rect in rects:
        cv2.rectangle(exgi_b_rgb, rect.pt1, rect.pt2, (0, 0, 255), 3)
    save_plot(f"{out_dir}/exgi_bin_rect.png", exgi_b_rgb)

    # roi
    img_rect = img.copy()
    for rect in rects:
        cv2.rectangle(img_rect, rect.pt1, rect.pt2, (0, 0, 255), 3)
    save_plot(f"{out_dir}/img_rect.png", img_rect)

    # reassembly
    ra = Reassembler()
    reassembled = ra.reassemble(img, use_resizable_packer=True, padding_size=0)
    save_plot(f"{out_dir}/reassembled.png", reassembled)


def calculate_preproc_recall(ioa_threshold=0.5):
    evaluator = WeedDetectorEvaluator("../dataset/test22_relabelled")

    n_positive = 0
    n_total = 0
    ioa_vals = []
    for tc in evaluator.test_dataset:
        img = cv2.imread(tc.image_path)
        gt = tc.labels

        ra = Reassembler()
        ra.reassemble(img, use_resizable_packer=True)
        mp = ra.mappings

        n_total += len(gt)

        for wl in gt:
            label_rect = wl.rect
            maxioa = 0
            for m in mp:
                pp_rect = m.src
                area_i = intersection_area(label_rect, pp_rect)
                ioa = area_i / label_rect.area
                if ioa > maxioa:
                    maxioa = ioa
            if maxioa >= ioa_threshold:
                n_positive += 1
                ioa_vals.append(maxioa)

        # print(f"Preprocessing recall: {n_positive}/{n_total}={n_positive / n_total}")

    print(f"Preprocessing recall: {n_positive}/{n_total}={n_positive / n_total}")
    print(ioa_vals)
    print(len(ioa_vals))

    plt.figure(figsize=(6, 5))
    plt.hist(ioa_vals, bins=100)
    plt.title('Distribution of Recall')
    plt.xlabel('intersection')
    plt.ylabel('Frequency')
    save_plt_plot("out/recall.png")
    plt.show()

    pts = []
    sorted_vals = sorted(list(set(ioa_vals)), reverse=True)
    for threshold in sorted_vals:
        pts.append([threshold, np.sum([i >= threshold for i in ioa_vals]) / len(ioa_vals)])

    pts = np.array(pts)

    plt.figure(figsize=(5, 4))
    plt.plot(pts[:, 0], pts[:, 1])
    plt.title('Recall Rate vs. Intersection Threshold')
    plt.xlabel('Intersection Threshold')
    plt.ylabel('Recall')
    plt.xlim(0, 1)
    plt.ylim(0.75, 1.01)
    plt.grid(visible=True)
    save_plt_plot("out/recall_thresh.png")
    plt.show()


def main():
    process_illustration("../test_data/frame.png")
    box_size("../dataset/test3")
    calculate_preproc_recall(0.5)


if __name__ == '__main__':
    main()
