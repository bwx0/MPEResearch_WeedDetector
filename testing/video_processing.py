import cv2

from transform_and_preview import process_crop


def process(frame):
    return process_crop(frame)

def video_process(input_video_path, output_video_path):
    """
    Read a video, process each frame, and write the processed frames to a new video.

    Args:
    input_video_path (str): The path to the input video.
    output_video_path (str): The path for the output processed video.
    """
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    codec  = cv2.VideoWriter_fourcc(*'MP4V')

    # Create a video writer for the output video
    out = cv2.VideoWriter(output_video_path, codec, fps, (width, height), True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_argb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        processed_frame = process(frame_argb)

        out.write(processed_frame)

    cap.release()
    out.release()
    print("Video processing complete and saved to", output_video_path)

video_process('G1.MP4', 'G1_out.mp4')
