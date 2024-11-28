import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

import cv2
import numpy as np
from torch import Tensor
from typing_extensions import override
from ultralytics.engine.results import Results
from ultralytics.models.yolo.model import YOLO

from roiyolowd.reassembler import create_reassembler
from roiyolowd.roi_extractor import ExGIGreenExtractor
from roiyolowd.util import Rect, WeedLabel, bgr2ExGI


class WeedDetector(ABC):

    def __init__(self, name: str):
        self.name = name
        self.__default_palette = {}

    @abstractmethod
    def detect(self, bgrimage: np.ndarray) -> List[WeedLabel]:
        pass

    def name_of_cls(self, cls: int) -> str:
        return f'{cls}'

    def color_of_cls(self, cls: int) -> Tuple[int, int, int]:
        if cls not in self.__default_palette:
            self.__default_palette[cls] = (np.random.randint(100, 255),
                                           np.random.randint(100, 255),
                                           np.random.randint(100, 255))
        return self.__default_palette[cls]

    def detect_and_draw(self, rgbimage: np.ndarray) -> Tuple[List[WeedLabel], np.ndarray]:
        labels = self.detect(rgbimage)
        canvas = rgbimage.copy()
        for label in labels:
            cv2.rectangle(canvas, label.rect.pt1, label.rect.pt2, self.color_of_cls(label.cls), 2)
            # cv2.putText(canvas, self.name_of_cls(label.cls), label.rect.pt1, cv2.FONT_HERSHEY_PLAIN, 1.0,
            #             self.color_of_cls(label.cls))
        return labels, canvas


class OWLDetector(WeedDetector):

    def __init__(self, area_threshold: int = 1000, CLS_WEED: int = 1):
        super().__init__("OWL Detector")
        self.area_threshold = area_threshold
        self.CLS_WEED = CLS_WEED

    @override
    def detect(self, bgrimage: np.ndarray) -> List[WeedLabel]:
        result: List[WeedLabel] = []
        exg = bgr2ExGI(bgrimage)
        gray = exg.copy()
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.area_threshold:
                x, y, w, h = cv2.boundingRect(contour)
                rect = Rect(x, y, w, h)
                wl = WeedLabel(rect, self.CLS_WEED)
                result.append(wl)

        return result


class ExGIDetector(WeedDetector):
    """
    This detector uses ROIs directly as its prediction result.
    The bounding boxes it generates contains all the plants a YOLOv8Detector can potentially detect.
    However, this detector has two limitations:
        - It cannot differentiate between crops and weeds.
        - Multiple plants may be grouped into a single bounding box.
    YOLOv8Detectors can address the two issues.
    """

    def __init__(self, CLS_WEED: int = 1):
        super().__init__("ExGIDetector")
        self.CLS_WEED = CLS_WEED
        self.green_extractor = ExGIGreenExtractor(merge_overlapping_rects=False)

    @override
    def detect(self, bgrimage: np.ndarray) -> List[WeedLabel]:
        result: List[WeedLabel] = []
        green_rects: List[Rect] = self.green_extractor.extract_roi(bgrimage)
        for rect in green_rects:
            result.append(WeedLabel(rect, self.CLS_WEED))
        return result


class YOLOv8Detector(WeedDetector, ABC):
    def __init__(self, name: str, model: YOLO, confidence_threshold: float):
        super().__init__(name)
        self.model: YOLO = model
        self.names: Dict[int, str] = model.names  # mappings from cls id to cls name
        self.confidence_threshold: float = confidence_threshold
        self.label_colors: Dict[int, Tuple[int, int, int]] = {}
        self.__init_label_colors()

    @override
    def name_of_cls(self, cls: int) -> str:
        return self.names[cls]

    @override
    def color_of_cls(self, cls: int) -> Tuple[int, int, int]:
        return self.label_colors[cls]

    @staticmethod
    def yolopred2weedlabels(pred: Results) -> List[WeedLabel]:
        raw_boxes: Tensor = pred.boxes.data.cpu()  # [nlabels, 6]    (x1,y1,x2,y2,conf,cls)
        labels: List[WeedLabel] = []
        for x1, y1, x2, y2, conf, cls in raw_boxes:
            labels.append(WeedLabel(Rect(int(x1), int(y1), int(x2 - x1), int(y2 - y1)), int(cls), float(conf)))
        return labels

    def __init_label_colors(self):
        import hashlib
        for id, name in self.names.items():
            rng = random.Random(int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) + 125)
            self.label_colors[id] = (rng.randint(30, 220), rng.randint(30, 220), rng.randint(30, 220))


class VanillaYOLOv8Detector(YOLOv8Detector):
    def __init__(self, model: YOLO, confidence_threshold: float = 0.25):
        super().__init__("Vanilla YOLOv8 Detector", model, confidence_threshold)

    @override
    def detect(self, bgr_image: np.ndarray) -> List[WeedLabel]:
        pred = self.model.predict(bgr_image, conf=self.confidence_threshold)
        wls: List[WeedLabel] = YOLOv8Detector.yolopred2weedlabels(pred[0])
        return wls


class YOLOv8WithROIDetector(YOLOv8Detector):
    def __init__(self, model: YOLO, confidence_threshold: float = 0.25, use_native_reassembler: bool = True,
                 reassemble_size_limit: float = 0.7):
        """

        Args:
            model: The YOLOv8 model
            confidence_threshold:
            use_native_reassembler:
            reassemble_size_limit: stop using the reassembled image if its size exceed original_size * reassemble_size_limit
        """
        super().__init__("YOLOv8 Detector with ROI", model, confidence_threshold)
        self.use_native_reassembler = use_native_reassembler
        self.reassemble_size_limit = reassemble_size_limit

    @override
    def detect(self, bgr_image: np.ndarray) -> List[WeedLabel]:
        ra = create_reassembler(self.use_native_reassembler)
        rf = ra.reassemble(bgr_image, use_resizable_packer=True, border_thickness=3)
        # cv2.imshow("rf", rf)

        if rf.shape[0] * rf.shape[1] >= bgr_image.shape[0] * bgr_image.shape[1] * self.reassemble_size_limit:
            # This could also be done in the reassembler by calculating the total area of the ROIs, which is
            # actually better as it saves time spent on image reconstruction.
            print(f"Reassembled image too large {rf.shape}, falling back to regular detection.")
            pred = self.model.predict(bgr_image, conf=self.confidence_threshold)
            wls: List[WeedLabel] = YOLOv8Detector.yolopred2weedlabels(pred[0])
            return wls

        pred = self.model.predict(rf, conf=self.confidence_threshold)
        wls_raw = YOLOv8Detector.yolopred2weedlabels(pred[0])
        rects: List[Rect] = [wl.rect for wl in wls_raw]
        wls: List[WeedLabel] = []

        # canvas = rf.copy()
        # for label in wls_raw:
        #     cv2.rectangle(canvas, label.rect.pt1, label.rect.pt2, self.color_of_cls(label.cls), 2)
        # cv2.imshow("rf222222", canvas)

        back = ra.reverse_map(rects)
        n_fail = 0
        for i in range(len(back)):
            br = back[i]
            if not br.dst:
                n_fail += 1
            else:
                wls.append(WeedLabel(br.dst, wls_raw[i].cls, wls_raw[i].conf))

        if n_fail > 0:
            print(f"################### Unable to map {n_fail} rects back to the original image space")

        return wls
