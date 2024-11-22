import time

import cv2
import numpy as np
from libcamera import controls
from picamera2 import Picamera2

# Initialize the camera
picamera2 = Picamera2(1)
preview_config = picamera2.create_preview_configuration(main={"size": (1920//2, 1080//2)})
picamera2.configure(preview_config)
picamera2.startt()

print(picamera2.camera_controls)

# Set up autofocus if supported
if "AfMode" in picamera2.camera_controls:
    print("afmode=continuous!! ", controls.AfModeEnum.Continuous)
    picamera2.set_controls({"AfMode": controls.AfModeEnum.Continuous})  # Enable continuous autofocus


try:
    while True:
        capture_st=time.time()
        frame = np.array(picamera2.capture_array())
        capture_el=time.time()-capture_st
        print(frame.shape)
        
        proc_st=time.time()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
        edges = cv2.Canny(gray, 40, 100)
        
        edges_red = np.zeros_like(frame)
        edges_red[:, :, 0] = edges  # Set red channel
        
        # Combine the original frame with red edges
        result = cv2.addWeighted(frame, 1, edges_red, 0.8, 0)
        
        proc_el=time.time()-proc_st
        
        print(f"capture={int(capture_el*1000)}ms  proc={int(proc_el*1000)}ms")
        
        
        # Show the result
        cv2.imshow('Canny Edge Detection with Red Edges', cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA))
        if cv2.waitKey(1) == ord('q'):
            break
finally:
    picamera2.stop()
    cv2.destroyAllWindows()
