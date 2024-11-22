import cv2
from ultralytics.models.yolo.model import YOLO

from roiyolowd.weed_detector import OWLDetector, VanillaYOLOv8Detector, YOLOv8WithROIDetector, ExGIDetector

model = YOLO("../models/final_80/weights/best.pt")
# model = YOLO("../models/CottonWeedDet12/medium.pt")
# model = YOLO("../models/ablation/train13_vanilla_training/weights/best.pt")

det_vi = ExGIDetector()
det_owl = OWLDetector()
det_yolo = VanillaYOLOv8Detector(model)
det_yolo_roi = YOLOv8WithROIDetector(model, use_native_reassembler=True)

# total number of detections
tot_vi = 0
tot_owl = 0
tot_yolo = 0
tot_yolo_roi = 0


# TODO skip reassembling if reassembled image is too big

def process(frame):
    global tot_vi, tot_owl, tot_yolo, tot_yolo_roi

    result_vi, img_vi = det_vi.detect_and_draw(frame)
    result_owl, img_owl = det_owl.detect_and_draw(frame)
    result_yolo, img_yolo = det_yolo.detect_and_draw(frame)
    result_yolo_roi, img_yolo_roi = det_yolo_roi.detect_and_draw(frame)

    fsize = 0.47  # window scale factor, increase/decrease this value if the preview window is too small/big.
    cv2.imshow('ExGI', cv2.resize(img_vi, (0, 0), fx=fsize, fy=fsize))
    cv2.imshow('OWL', cv2.resize(img_owl, (0, 0), fx=fsize, fy=fsize))
    cv2.imshow('YOLOv8', cv2.resize(img_yolo, (0, 0), fx=fsize, fy=fsize))
    cv2.imshow('YOLOv8+ROI', cv2.resize(img_yolo_roi, (0, 0), fx=fsize, fy=fsize))

    tot_vi += len(result_vi)
    tot_owl += len(result_owl)
    tot_yolo += len(result_yolo)
    tot_yolo_roi += len(result_yolo_roi)

    print(f"Total number of detected objects: {tot_vi} {tot_owl} {tot_yolo} {tot_yolo_roi}")


def video_preview(video_file):
    """
    Iterate over each frame in the video and pass it to the process() function defined above.
    """
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file}.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video or failed to read frame.")
            break

        process(frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_preview(r"D:\projects\data_topdown\d2.MP4")
