import re
import sys

import cv2
import numpy as np

areaR_list = []
utilisation_list = []


def my_custom_print(x, *args, **kwargs):
    pattern = r"areaR=([\d.]+)\s+utilisation=([\d.]+)"
    matches = re.findall(pattern, x)
    for match in matches:
        areaR_list.append(float(match[0]))
        utilisation_list.append(float(match[1]))

from roiyolowd.reassembler import Reassembler

sys.modules[Reassembler.__module__].__dict__['print'] = my_custom_print


def video_process(input_video_path):
    cap = cv2.VideoCapture(input_video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ra = Reassembler()
        ra.reassemble(frame)

    cap.release()


video_process(r"../test_data/d1.mp4")
print(np.mean(utilisation_list))
print(np.mean(areaR_list))