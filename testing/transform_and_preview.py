import time

import cv2
import numpy as np
from libcamera import controls
from picamera2 import Picamera2

from roiyolowd.util import rgba2ExGI


def skeletonize_image(binary_image):
    cross_shape = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    skeleton = np.zeros_like(binary_image)

    # Keep applying the morphological operations until the image is fully eroded
    while True:
        eroded = cv2.erode(binary_image, cross_shape)
        temp = cv2.dilate(eroded, cross_shape)
        temp = cv2.subtract(binary_image, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary_image = eroded.copy()

        if cv2.countNonZero(binary_image) == 0:
            break

    return skeleton

def draw_lines(img, lines, thickness=1):
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 5000 * (-b))
            y1 = int(y0 + 5000 * (a))
            x2 = int(x0 - 5000 * (-b))
            y2 = int(y0 - 5000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0, 0), thickness)
    return img

def filter_close_lines(lines, rho_threshold, theta_threshold):
    filtered_lines = []
    for i in range(len(lines)):
        flag=True
        for j in range(i + 1, len(lines)):
            rho_i, theta_i = lines[i][0]
            rho_j, theta_j = lines[j][0]
            if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold and abs(theta_i)<abs(theta_j):
                flag=False
                break
        if flag:
            filtered_lines.append(lines[i])
    return filtered_lines


def process_canny(rgba):
    gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
    edges = cv2.Canny(gray, 40, 100)
    
    edges_red = np.zeros_like(rgba)
    edges_red[:, :, 0] = edges
    
    result = cv2.addWeighted(rgba, 1, edges_red, 0.8, 0)
    return result


def process_hough(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    edges = cv2.Canny(gray, 50, 120)
    
    pic1 = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGBA)
    
    rho = 5           # Distance resolution in pixels
    theta = np.pi / 50  # Angular resolution in radians
    threshold = 300   # Minimum number of votes (intersections in Hough grid cell)
    lines = cv2.HoughLines(edges, rho, theta, threshold)
    pic2 = draw_lines(image, lines)
    
    return np.hstack((pic1,pic2))


def process_crop(image):
    pic1=cv2.cvtColor(image.copy(),cv2.COLOR_RGBA2RGB)
    exg = rgba2ExGI(image)
    pic2=cv2.cvtColor(exg.copy(),cv2.COLOR_GRAY2RGB)
    _, b1 = cv2.threshold(exg, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pic3=cv2.cvtColor(b1.copy(),cv2.COLOR_GRAY2RGB)
    b2=cv2.morphologyEx(b1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(15,15)))
    pic4=cv2.cvtColor(b2.copy(),cv2.COLOR_GRAY2RGB)
    b3=cv2.morphologyEx(b2, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT,(12,12)))
    #b3=cv2.subtract(b3.copy(),cv2.morphologyEx(b3.copy(), cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))))
    b3=skeletonize_image(b3)
    b3=cv2.morphologyEx(b3, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    pic5=cv2.cvtColor(b3.copy(),cv2.COLOR_GRAY2RGB)
    
    
    factor=5
    #reduced = block_reduce(b1, block_size=(factor, factor), func=np.max)
    reduced=cv2.resize(b1, (b1.shape[1]//factor,b1.shape[0]//factor), interpolation = cv2.INTER_LINEAR);
    lines = cv2.HoughLines(b3, 1, np.pi / 30, 100, min_theta=-np.pi/8, max_theta=np.pi/8)
    if lines is None:
        lines=[]
    #lines = filter_close_lines(lines, 50, 3)
    print(len(lines))
    reducedimage=cv2.resize(image, (b1.shape[1]//factor,b1.shape[0]//factor), interpolation = cv2.INTER_LINEAR);
    result_image = draw_lines(cv2.cvtColor(image.copy(),cv2.COLOR_RGBA2RGB), lines, 8)
    pic10=result_image
    
    #return pic10
    return np.vstack((np.hstack((pic1,pic2,pic3)),np.hstack((pic4,pic5,pic10))))


def process(rgba):
    return process_crop(rgba)
    return process_hough(rgba)
    return process_canny(rgba)





def pic_preview():
    # Initialize the camera
    picamera2 = Picamera2(0)
    preview_config = picamera2.create_preview_configuration(main={"size": (1920//2, 1440//2)})
    print(picamera2.camera_controls)
    picamera2.configure(preview_config)
    # Set up autofocus if supported
    if "AfMode" in picamera2.camera_controls:
        print("afmode=continuous!! ", controls.AfModeEnum.Continuous)
        picamera2.set_controls({"AfMode": controls.AfModeEnum.Continuous})  # Enable continuous autofocus
    picamera2.startt()
    
    try:
        while True:
            capture_st=time.time()
            frame = np.array(picamera2.capture_array())
            capture_el=time.time()-capture_st
        
            proc_st=time.time()
            result = process(frame)
            proc_el=time.time()-proc_st
        
            stat = f"capture={int(capture_el*1000)}ms  proc={int(proc_el*1000)}ms"
            print(stat)
        
            cv2.imshow('previ', cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA))
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        picamera2.stop()
        cv2.destroyAllWindows()

pic_preview()
