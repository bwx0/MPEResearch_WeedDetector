import time

import cv2
import numpy as np

image = cv2.imread('../data/rpi_size.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50000, 60000, apertureSize=7)

rho = 5  # Distance resolution in pixels
theta = np.pi / 50  # Angular resolution in radians
threshold = 220 * 4  # Minimum number of votes (intersections in Hough grid cell)

start_time = time.time()

runs = 100

for _ in range(runs):
    lines = cv2.HoughLines(edges, rho, theta, threshold)

end_time = time.time()
elapsed_time = end_time - start_time
avg_time = elapsed_time / runs

print(f"Average execution time per call: {avg_time:.6f} seconds")

lines = cv2.HoughLines(edges, rho, theta, threshold)
