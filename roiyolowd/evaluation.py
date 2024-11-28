import os
from pprint import pprint
from typing import List, Iterable, Tuple

import cv2
import numpy as np
import psutil
from PIL import Image
from matplotlib import pyplot as plt
from ultralytics.models.yolo.model import YOLO

from roiyolowd.util import calculate_iou, draw_labels, WeedLabel, Rect
from roiyolowd.weed_detector import OWLDetector, VanillaYOLOv8Detector, YOLOv8WithROIDetector, WeedDetector, ExGIDetector


def match_predictions_to_ground_truth(preds: List[WeedLabel], ground_truth: List[WeedLabel], iou_threshold=0.2) \
        -> List[Tuple[float, bool, int, int]]:
    """

    Args:
        preds:
        ground_truth:
        iou_threshold:

    Returns:
        List[Tuple[float, int, int, int]]
        tuple[0]: confidence
        tuple[1]: is true positive
        tuple[2]: label class id
        tuple[3]: area of the ground truth bounding box

    """
    ground_truth = ground_truth.copy()
    matched = []
    for i, label in enumerate(preds):  # Enumerate each prediction
        best_iou = 0
        best_gt_index = -1
        for j, gt in enumerate(ground_truth):  # Compare it with each unassigned GT
            iou = calculate_iou(label.rect, gt.rect)
            if iou > best_iou and iou >= iou_threshold and gt.cls == label.cls:
                best_iou = iou
                best_gt_index = j

        if best_gt_index != -1:  # The best match is found for pred i (iou larger than the threshold)
            gt = ground_truth[best_gt_index]
            matched.append((label.conf, True, label.cls, gt.rect.area))  # True positive
            ground_truth.pop(best_gt_index)  # Assign the best-match GT to prediction i
        else:
            matched.append((label.conf, False, label.cls, 0))  # False positive

    # Add false negatives for remaining ground truth bboxes
    for gt in ground_truth:
        matched.append((0.0, False, gt.cls, gt.rect.area))  # False negative (not detected)

    return matched


def compute_ap(recall: Iterable[float], precision: Iterable[float]):
    """
    Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall: The recall curve.
        precision: The precision curve.

    Returns:
        (float): Average precision.
        (np.ndarray): Precision envelope curve.
        (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def calculate_metrics(matched_predictions: List[Tuple[float, bool, int, int]], ground_truth: List[WeedLabel],
                      conf: float = 0.3, plot=False):
    matched_predictions = sorted(matched_predictions, key=lambda x: x[0] * 10 ** 20 + x[3], reverse=True)
    tp = np.cumsum([x[1] for x in matched_predictions])
    fp = np.cumsum([1 - x[1] for x in matched_predictions])
    precision = tp / (tp + fp)
    recall = tp / len(ground_truth)

    if plot:
        plt.scatter(recall, precision)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()

    ap, mpre, mrec = compute_ap(recall, precision)

    tp_val = np.sum([x[1] for x in matched_predictions if x[0] >= conf])
    fp_val = np.sum([1 - x[1] for x in matched_predictions if x[0] >= conf])
    precision_val = tp_val / (tp_val + fp_val)
    recall_val = tp_val / len(ground_truth)

    return ap, mpre, mrec, precision_val, recall_val


def calculate_metrics_classified(matched_predictions: List[Tuple[float, bool, int, int]], ground_truth: List[WeedLabel],
                                 conf: float = 0.3, plot=False):
    all_cls = set([x[2] for x in matched_predictions])

    result = {}
    ap_all, mpre_all, mrec_all, precision_val_all, recall_val_all = calculate_metrics(matched_predictions, ground_truth,
                                                                                      conf, plot)
    result['all'] = {
        'ap': ap_all,
        'precision': precision_val_all,
        'recall': recall_val_all,
    }

    for cls in all_cls:
        subset_pred = [x for x in matched_predictions if x[2] == cls]
        subset_gt = [x for x in ground_truth if x.cls == cls]
        ap, mpre, mrec, precision_val, recall_val = calculate_metrics(subset_pred, subset_gt, conf)
        result[f'{cls}'] = {
            'ap': ap,
            'precision': precision_val,
            'recall': recall_val,
        }

    return result


class Testcase:
    __slots__ = ["image_path", "labels"]

    def __init__(self, image_path: str, labels: List[WeedLabel]):
        self.image_path: str = image_path
        self.labels: List[WeedLabel] = labels


def load_labels(label_path, image_width: int, image_height: int) -> List[WeedLabel]:
    with open(label_path, 'r') as f:
        labels: List[WeedLabel] = []
        for line in f.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])
            bbox = [float(x) for x in parts[1:]]
            x, y, w, h = bbox
            x *= image_width
            y *= image_height
            w *= image_width
            h *= image_height
            label = WeedLabel(Rect(int(x - w / 2), int(y - h / 2), int(w), int(h)), class_id)
            labels.append(label)
        return labels


class WeedDetectorEvaluator:
    def __init__(self, dataset_dir: str, is_splitted: bool = False):
        """
        Args:
            dataset_dir: Path to YOLOv8 dataset directory.
            is_splitted: if the dataset has been split into train and test sets.
        """
        self.dataset_dir: str = dataset_dir
        self.image_dir = f'{dataset_dir}/images'
        self.label_dir = f'{dataset_dir}/labels'
        self.is_splitted: bool = is_splitted
        self.test_dataset: List[Testcase] = []
        self.__load_dataset()

    def __load_dataset(self):
        self.test_dataset.clear()

        # Load validation and test set if the given dataset is split into such sets. Otherwise, assume all images and
        # labels are stored directly in self.image_dir and self.label_dir.
        for subset in ['val', 'test'] if self.is_splitted else ['']:
            image_subset_dir = os.path.join(self.image_dir, subset)
            label_subset_dir = os.path.join(self.label_dir, subset)

            if not os.path.exists(image_subset_dir) or not os.path.exists(label_subset_dir):
                print(f"{image_subset_dir} does not exist. skipping.")
                continue

            for image_file in os.listdir(image_subset_dir):
                if image_file.endswith('.jpg') or image_file.endswith('.png'):
                    image_path = os.path.join(image_subset_dir, image_file)
                    label_file = image_file.replace('.jpg', '.txt').replace('.png', '.txt')
                    label_path = os.path.join(label_subset_dir, label_file)
                    with Image.open(image_path) as img:
                        width, height = img.size
                    labels = load_labels(label_path, width, height)
                    self.test_dataset.append(Testcase(image_path, labels))
        print(f"Loaded dataset: {self.dataset_dir}   Total images: {len(self.test_dataset)}")

    def evaluate(self, weed_detector: WeedDetector):
        ap_list = []
        all_predictions = []
        all_groundtruth = []
        all_area = []
        for testcase in self.test_dataset:
            bgrimg = cv2.imread(testcase.image_path)
            print(testcase.image_path)
            predictions, img = weed_detector.detect_and_draw(bgrimg)
            draw_labels(img, testcase.labels)
            cv2.imshow(weed_detector.name, img)
            cv2.waitKey(1)
            matched_predictions = match_predictions_to_ground_truth(predictions, testcase.labels)
            metrics = calculate_metrics_classified(matched_predictions, testcase.labels)
            areas = [(int(x[1]), x[3]) for x in matched_predictions]
            all_area.extend(areas)
            map50 = metrics['all']['ap']
            print(f"mAP50: {map50:.4f}    {testcase.image_path}")
            ap_list.append(map50)
            all_predictions.extend(matched_predictions)
            all_groundtruth.extend(testcase.labels)
        overall_metrics = calculate_metrics_classified(all_predictions, all_groundtruth, plot=True)
        print(f"overall mAP50: {overall_metrics['all']['ap']:.4f}")
        pprint(overall_metrics)
        # print(all_area)
        return overall_metrics


if __name__ == "__main__":
    # model = YOLO("../yolo_tools/runs/detect/train13/weights/best.pt")
    # model = YOLO("../models/ablation/train3_split_ra10_fixedmargin/weights/best.pt")
    model = YOLO("../models/final_80/weights/best.pt")
    det_owl = OWLDetector()
    det_yolo = VanillaYOLOv8Detector(model)
    det_yolo_roi = YOLOv8WithROIDetector(model)
    det_vi = ExGIDetector()

    evaluator = WeedDetectorEvaluator("../dataset/test22_relabelled")
    m0 = evaluator.evaluate(det_owl)
    m1 = evaluator.evaluate(det_yolo)
    m2 = evaluator.evaluate(det_yolo_roi)
    m3 = evaluator.evaluate(det_vi)

    print(m0['all']['ap'], m1['all']['ap'], m2['all']['ap'], m3['all']['ap'])

    print(f"peak memory usage: {psutil.Process().memory_info().peak_wset // 1024 // 1024}MB")
