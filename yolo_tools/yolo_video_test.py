import time
from typing import List

import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

from roiyolowd.reassembler import Reassembler
from roiyolowd.util import Rect, draw_boxes, Stopwatch
from roiyolowd.weed_detector import YOLOv8Detector


def yolopred2rects(pred: Results) -> List[Rect]:
    from torch import Tensor
    rects: Tensor = pred.boxes.xywh.cpu().numpy()
    result: List[Rect] = [Rect(int(x - w / 2), int(y - h / 2), int(w), int(h)) for x, y, w, h in rects]
    return result


def detect_vanilla(model, frame):
    pred = model.predict(frame, conf=0.3, imgsz=1920)
    YOLOv8Detector.yolopred2weedlabels(pred[0])
    rects = yolopred2rects(pred[0])
    draw_boxes(frame, rects, (255, 0, 255))
    cv2.imshow("det_vanilla", frame)

    return len(pred[0])


def detect_roi(model, frame, frame0):
    sw = Stopwatch()

    ra = Reassembler()
    rf = ra.reassemble(frame0, autosize=True)
    sw.stop("reassemble")
    cv2.imshow("reassembled frame", rf)

    pred = model.predict(rf, conf=0.3)
    sw.stop("pred")
    rects = yolopred2rects(pred[0])
    sw.stop("map")
    draw_boxes(rf, rects, (0, 150, 255))
    cv2.imshow("det_roi", rf)

    back = ra.reverse_mapping(rects)
    rects_original_space = []
    for br in back:
        if not br.dst:
            print("################### Unable to map rects back to original image space")
        else:
            rects_original_space.append(br.dst)
    draw_boxes(frame, rects_original_space, (0, 150, 255))
    cv2.imshow("det_roi_final", frame)
    sw.stop("revmap")

    return len(pred[0])


def video_preview(video_file, model):
    cap = cv2.VideoCapture(video_file)
    tot_proc_time = 0
    tot_proc_frames = 0
    tot_pred = 0

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    try:
        tot_v = 0
        tot_r = 0
        while True:
            capture_st = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Reached end of video or failed to read frame.")
                break
            capture_el = time.time() - capture_st

            proc_st = time.time()

            # result = pred[0].plot(font_size=0.1, line_width=1)
            frame0 = frame.copy()

            proc_st_v = time.time()
            pred_v = detect_vanilla(model, frame)
            print(f"proc_st_v={int((time.time() - proc_st_v) * 1000)}ms")

            proc_st_r = time.time()
            pred_r = detect_roi(model, frame, frame0)
            print(f"proc_st_r={int((time.time() - proc_st_r) * 1000)}ms")

            tot_v += pred_v
            tot_r += pred_r

            print(tot_v, tot_r)

            # pred = model.predict(frame)
            # tot_pred += len(pred[0])

            proc_el = time.time() - proc_st

            tot_proc_time += proc_el
            tot_proc_frames += 1
            stat = (f"capture={int(capture_el * 1000)}ms  proc={int(proc_el * 1000)}ms    "
                    f"avg={int(tot_proc_time * 1000 / tot_proc_frames)}ms   "
                    f"avg_pred={tot_pred / tot_proc_frames}labels")
            print(stat)

            # cv2.imshow('preview', result)
            if cv2.waitKey(1) == ord('q'):
                break
        cv2.waitKey(0)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    model = YOLO("../models/final_80/weights/best.pt")
    video_preview(r"../test_data/d2.mp4", model)
