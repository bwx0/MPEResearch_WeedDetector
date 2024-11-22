import time

import cv2

print("init")

camera_port = 0
ramp_frames = 20  # throw away first few frames to make it adjust to light levels

camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
print("init2")
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
retval = camera.set(3, 640)
retval = camera.set(4, 480)
print("init3")
print(camera.get(cv2.CAP_PROP_FRAME_WIDTH), camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"isopen={camera.isOpened()}")

assert camera.isOpened()

# for i in range(ramp_frames):
#  temp = camera.read()

while True:
    start = time.time()
    _, img = camera.read()
    print(f"read={int((time.time() - start) * 1000)}ms")

    start = time.time()
    # processing
    cimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"cvt={int((time.time() - start) * 1000)}ms")

    start = time.time()
    circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, 1, 1, param1=200, param2=80, minRadius=0, maxRadius=0)
    if circles is not None:
        x = circles[0][0][0]
        y = circles[0][0][1]
        print("x= ", x, "y= ", y)
    else:
        print("no circles detected")
    print(f"hough={int((time.time() - start) * 1000)}ms")
