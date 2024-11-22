import os.path

import cv2

from roiyolowd.crop_row_finder import CropRowFinder
from roiyolowd.util import bgr2ExGI

"""
Specify the path to a video, extract every 15th frame, and mask the crop rows within.
"""

video_path = r"../test_data/d11.MP4"

def mask_video_frames(video_file):
    cap = cv2.VideoCapture(video_file)
    crf = CropRowFinder()

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    tot_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    current_frame_count = 0

    out_dir = video_file + "_masked"
    os.makedirs(out_dir, exist_ok=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Reached end of video or failed to read frame.")
                break

            print(str(current_frame_count) + "/" + str(tot_frame_count))

            exg = bgr2ExGI(frame)
            _, b1 = cv2.threshold(exg, 25, 255, cv2.THRESH_BINARY)

            mask = crf.find(b1)
            if current_frame_count % 15 == 0:
                mask = cv2.bitwise_not(mask)
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                fr = cv2.bitwise_and(mask_rgb, frame)
                cv2.imwrite(os.path.join(out_dir, str(current_frame_count) + ".png"), fr)

            if cv2.waitKey(1) == ord('q'):
                break
            current_frame_count += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    mask_video_frames(video_path)
