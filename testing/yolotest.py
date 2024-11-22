import time
from pathlib import Path
from threading import Thread

import psutil
from ultralytics import YOLO

process = psutil.Process()
print(process.memory_info().rss)


def exportncnn():
    model = YOLO('yolov8n.pt')
    # Export the model to NCNN format
    model.export(format='ncnn')  # creates 'yolov8n_ncnn_model'


if not Path("yolov8n_ncnn_model").exists():
    exportncnn()


def inference():
    print(process.memory_info().rss)
    st = time.time()
    # ncnn_model = YOLO('yolov5nu.pt')
    ncnn_model = YOLO('yolov8n_ncnn_model')
    print(f"load model: {int((time.time() - st) * 1000)}ms")

    st = time.time()
    for i in range(10):
        results = ncnn_model('../test_data/nano.png')
    print(f"total time: {int((time.time() - st) * 1000)}ms")
    print(process.memory_info())


Thread(target=inference, args=()).start()
# Thread(target=inference, args=()).start()
