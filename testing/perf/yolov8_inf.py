import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from util import benchmark


def exportncnn():
    model = YOLO('yolov8n.pt')
    # Export the model to NCNN format
    model.export(format='ncnn')  # creates 'yolov8n_ncnn_model'


if not Path("yolov8n_ncnn_model").exists():
    exportncnn()


def inference():
    st = time.time()
    ncnn_model = YOLO('yolov8n_ncnn_model')
    print(f"load model: {int((time.time() - st) * 1000)}ms")

    img = cv2.imread("1200x800.png")

    def do_inference(img):
        results = ncnn_model(img)

    benchmark(do_inference, test_name="YOLOv8n Inference", img=img)


inference()
