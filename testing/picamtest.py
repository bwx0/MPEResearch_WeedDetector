import sys
import time
import warnings
from threading import Thread, Event

import cv2
import libcamera
import numpy as np
from picamera2 import Picamera2

PICAMERA_VERSION = 'picamera2'



class PiCamera2Stream:
    def __init__(self, src=0, resolution=(1200, 800), exp_compensation=-2, **kwargs):
        self.name = 'Picamera2Stream'
        self.size = resolution  # picamera2 uses size instead of resolution, keeping this consistent
        self.frame_width = None
        self.frame_height = None

        # set the picamera2 config and controls. Refer to picamera2 documentation for full explanations:
        #
        self.configurations = {
            # for those checking closely, using RGB888 may seem incorrect, however libcamera means a BGR format. Check
            # https://github.com/raspberrypi/picamera2/issues/848 for full explanation.
            "format": 'RGB888',
            "size": self.size
        }

        self.controls = {
            "AeExposureMode": 1,
            "AwbMode": libcamera.controls.AwbModeEnum.Daylight,
            "ExposureValue": exp_compensation
        }

        # Update config with any additional/overridden parameters
        self.controls.update(kwargs)

        # Initialize the camera
        self.camera = Picamera2(src)
        self.camera_model = self.camera.camera_properties['Model']

        if self.camera_model == 'imx296':
            print('[INFO] Using IMX296 Global Shutter Camera')

        elif self.camera_model == 'imx477':
            print('[INFO] Using IMX477 HQ Camera')

        elif self.camera_model == 'imx708':
            print('[INFO] Using Raspberry Pi Camera Module 3. Setting focal point at 1.2 m...')
            self.controls['AfMode'] = libcamera.controls.AfModeEnum.Manual
            self.controls['LensPosition'] = 1.2

        else:
            print('[INFO] Unrecognised camera module, continuing with default settings.')

        try:
            self.config = self.camera.create_preview_configuration(main=self.configurations,
                                                                   controls=self.controls)
            self.camera.configure(self.config)
            self.camera.startt()

            # set dimensions directly from the video feed
            self.frame_width = self.camera.camera_configuration()['main']['size'][0]
            self.frame_height = self.camera.camera_configuration()['main']['size'][1]

            # allow the camera time to warm up
            time.sleep(2)

        except Exception as e:
            print(f"Failed to initialize PiCamera2: {e}")
            raise

        if self.frame_width != resolution[0] or self.frame_height != resolution[1]:
            message = (f"The actual frame size ({self.frame_width}x{self.frame_height}) "
                       f"differs from the expected resolution ({resolution[0]}x{resolution[1]}).")
            warnings.warn(message, RuntimeWarning)

        self.frame = None
        self.stopped = Event()

    def start(self):
        # Start the thread to update frames
        self.thread = Thread(target=self.update, name=self.name, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        try:
            last=0
            while not self.stopped.is_set():
                frame = self.camera.capture_array("main")
                if frame is not None:
                    el=time.time()-last
                    # print(f"t={el}  fps={1/el}")
                    last=time.time()
                    self.frame = frame
                time.sleep(0.01)  # Slow down loop a little
        except Exception as e:
            print(f"Exception in PiCamera2Stream update loop: {e}")
        finally:
            print("Stopping Picamera...")
            self.camera.stop()  # Ensure camera resources are released properly

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        self.stopped.set()
        self.thread.join()
        self.camera.stop()
        time.sleep(2)  # Allow time for the camera to be released properly


def sum_modified_laplacian(img):
    """
    Compute the Sum-modified Laplacian (SML) of an image.
    
    Args:
    img (numpy.ndarray): Input image (grayscale or color).
    
    Returns:
    float: The SML value, higher indicates a more in-focus image.
    """
    # Convert the image to grayscale if it is color
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply the Laplacian operator
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    
    # Calculate the absolute values of Laplacian and then sum them
    sml = np.sum(np.abs(laplacian))
    
    return sml

print("init")

camera_port = 0

cam = PiCamera2Stream()
cam.start()
time.sleep(1) 

# for i in range(ramp_frames):
#  temp = camera.read()
ind=0
while True:
    start=time.time()
    img = cam.read()

    start=time.time()
    #processing
    cimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # print(f"cvt={int((time.time()-start)*1000)}ms")

    start=time.time()
    edges = cv2.Canny(cimg, 1, 60000, apertureSize=7)
    circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1,1,param1=200,param2=80,minRadius=0,maxRadius=0)
    if circles is not None:
        x=circles[0][0][0]
        y=circles[0][0][1]
        print("x= ",x,"y= ",y)
    else:
        # print("no circles detected")
        pass
    # print(f"hough={int((time.time()-start)*1000)}ms")
    
    sml=sum_modified_laplacian(cimg)
    sys.stdout.write(f"\rsml={sml}")
    sys.stdout.flush()
    
    start=time.time()
    # cv2.imwrite(f"data/out/aa_{ind}.png",cimg)
    # print(f"save={int((time.time()-start)*1000)}ms")
    
    ind=ind+1