import argparse
import time

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_model", type=str)
    parser.add_argument("path_to_image", type=str)
    parser.add_argument("imgsz", type=str)
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.path_to_model)

    img = cv2.imread(args.path_to_image)

    tot_time = 0
    n_frames = 0
    while tot_time < 15:
        start = time.time()
        model.predict(img)
        duration = time.time() - start
        tot_time += duration
        n_frames += 1

    print(f"fps: {n_frames / tot_time}")


if __name__ == "__main__":
    main()
