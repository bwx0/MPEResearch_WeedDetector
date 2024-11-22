import time

import cv2
import numpy as np
from vidstab.VidStab import VidStab

from roiyolowd.crop_row_finder import CropRowFinder
from roiyolowd.reassembler import Reassembler
from roiyolowd.util import rgba2ExGI, Rect


def remove_small_components(image, area_threshold=0.002):
    """
    Remove connected components in a binarized image whose area is less than 1% of the total white area.

    Parameters:
    image (numpy.ndarray): Binarized image (single-channel, with white as the foreground).
    area_threshold (float): The area threshold as a fraction of the total white area.

    Returns:
    numpy.ndarray: The image with small components removed.
    """
    # Ensure the image is binary
    if len(image.shape) > 2:
        raise ValueError("The input image must be a binarized (single-channel) image.")

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

    total_white_area = np.sum(image == 255)
    threshold_area = area_threshold * total_white_area
    output_image = np.zeros_like(image)

    for i in range(1, num_labels):
        component_area = stats[i, cv2.CC_STAT_AREA]
        if component_area >= threshold_area:
            output_image[labels == i] = 255

    return output_image


def skeletonize_image(binary_image):
    # return np.where(skeletonize(binary_image,method="zhang"),np.full_like(binary_image,255),np.zeros_like(binary_image))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    skeleton = np.zeros_like(binary_image)

    # Keep applying the morphological operations until the image is fully eroded
    while True:
        eroded = cv2.erode(binary_image, se)
        temp = cv2.dilate(eroded, se)
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

def draw_linesp(img, lines, thickness=1):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
    return img

def filter_close_lines(lines, rho_threshold, theta_threshold):
    filtered_lines = []
    for i in range(len(lines)):
        flag = True
        for j in range(i + 1, len(lines)):
            rho_i, theta_i = lines[i][0]
            rho_j, theta_j = lines[j][0]
            if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold and abs(theta_i) < abs(
                    theta_j):
                flag = False
                break
        if flag:
            filtered_lines.append(lines[i])
    return filtered_lines


def find_white_pixels_near_line(binary_image, x1, y1, x2, y2, d):
    white_pixels = np.column_stack(np.where(binary_image == 255))

    # components of the line equation
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2

    distances = np.abs(A * white_pixels[:, 1] + B * white_pixels[:, 0] + C) / np.sqrt(A ** 2 + B ** 2)

    close_white_pixels = white_pixels[distances < d]
    return close_white_pixels

def line_coverage(close_white_pixels, x1, y1, x2, y2, accumulator_len=100):
    pass

def draw_ransac_lines(img, lines, thickness=1):
    height, width, _ = img.shape
    if lines is not None:
        for line in lines:
            m, c = line
            cv2.line(img, (0, int(m*0+c)), (width, int(width*m+c)), (0, 255, 0), thickness)
    return img


def draw_ROI(image, roi_mask, alpha = 0.5):
    red_overlay = np.zeros_like(image)
    red_overlay[:, :, 0] = 255
    overlay = cv2.bitwise_and(red_overlay, red_overlay, mask=roi_mask)
    blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return blended

def ransac_lines(img):
    height, width = img.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    results = []
    for i in range(1, num_labels):
        cc_pixels = np.column_stack(np.where(labels == i))
        for j in range(1, len(cc_pixels)):
            if np.random.rand() > 0.2:
                continue
            p1 = cc_pixels[j]
            p2 = cc_pixels[np.random.randint(j)]
            w = find_white_pixels_near_line(img, *p1, *p2, 40)
            if(len(w)<500):
                continue
            x = w[:, 0]
            y = w[:, 1]
            m, c = np.polyfit(x, y, 1)
            y_pred = m * x + c
            ss_total = np.sum((y - np.mean(y))**2)
            ss_res = np.sum((y - y_pred)**2)
            r_squared = 1 - (ss_res / ss_total)
            if(r_squared<0.9):
                continue
            print(r_squared)
            results.append((m, c))
    return results

def to_simple_polygons(binary_image, epsilon_factor=0.005):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_image = np.zeros_like(binary_image)

    for contour in contours:
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(output_image, [approx_polygon], -1, 255, thickness=cv2.FILLED)

    return output_image


ed_last = None
def exp_decay(image):
    global ed_last
    if ed_last is None:
        ed_last = np.zeros_like(image)

    ed_last = np.add(np.multiply(ed_last, 0.96), np.multiply(image, 0.12))
    ed_last = np.clip(ed_last, 0, 500)

    clipped = np.clip(ed_last, 0, 255).astype(np.uint8)

    return clipped


def create_ortho_image(image, camera_angle=45):
    h, w = image.shape[:2]
    src_points = np.float32([[0+300, 0+300], [w+300, 0+300], [w+300, h+300], [0+300, h+300]])
    dst_points = np.float32([
        [220, 220],
        [w-220, 220],
        [w, h],
        [0, h]
    ])
    H = cv2.getPerspectiveTransform(dst_points, src_points)
    warped_image = cv2.warpPerspective(image, H, (w*2, h*5))
    return warped_image

def compute_ortho_image(img):
    ortho_image = create_ortho_image(img)

    # Save or display the ortho image
    cv2.imshow('Ortho Image', cv2.cvtColor(ortho_image, cv2.COLOR_RGB2BGR))

def process_canny(rgba):
    gray = cv2.cvtColor(rgba, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 100)

    edges_red = np.zeros_like(rgba)
    edges_red[:, :, 0] = edges

    result = cv2.addWeighted(rgba, 1, edges_red, 0.8, 0)
    return result


def process_hough(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 120)

    pic1 = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGBA)

    rho = 5  # Distance resolution in pixels
    theta = np.pi / 50  # Angular resolution in radians
    threshold = 50  # Minimum number of votes (intersections in Hough grid cell)
    lines = cv2.HoughLines(edges, rho, theta, threshold)
    pic2 = draw_lines(image, lines)

    return np.hstack((pic1, pic2))



def process_owl_contour(image):
    image = image.copy()
    exg = rgba2ExGI(image)
    gray = exg.copy()
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        # print("Contour Area:", area)
        if area > 10:
            cv2.drawContours(image, [contour], -1,
                             (np.random.randint(150, 255), np.random.randint(0, 255), np.random.randint(0, 255)), 2)
    pr = (np.hstack((
        cv2.cvtColor(exg, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR),
        image)))
    cv2.imshow('owl', cv2.cvtColor(pr, cv2.COLOR_RGB2BGR))

    return image




def reassemble_green(image_fullres):
    exg = rgba2ExGI(image_fullres)
    _, b1 = cv2.threshold(exg, 20, 255, cv2.THRESH_BINARY)
    pic1 = image_fullres.copy()
    pic2 = cv2.cvtColor(exg.copy(), cv2.COLOR_GRAY2RGB)
    b2 = cv2.morphologyEx(b1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    b2=b1

    pic3 = cv2.cvtColor(b2.copy(), cv2.COLOR_GRAY2RGB)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(b2, connectivity=8)


    N_TILE_X, N_TILE_Y = 120, 60
    tile_w, tile_h = b1.shape[1] // N_TILE_X, b1.shape[0] // N_TILE_Y
    if b1.shape[1] % N_TILE_X > 0 or b1.shape[0] % N_TILE_Y > 0:
        raise ValueError(f"N_TILE_X={N_TILE_X}   N_TILE_Y={N_TILE_Y}   tile_w={b1.shape[1] / N_TILE_X}   tile_h={b1.shape[0] / N_TILE_Y}")
    print(tile_w, tile_h)

    mask = [[False for _ in range(N_TILE_X)] for _ in range(N_TILE_Y)]

    p3 = image_fullres.copy()
    for i in range(1, num_labels):  # Skip the first label (background)
        x, y, w, h, area = stats[i]
        cv2.rectangle(p3, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Determine which tiles the bounding box overlays
        tile_x_start = x // tile_w
        tile_x_end = (x + w) // tile_w
        tile_y_start = y // tile_h
        tile_y_end = (y + h) // tile_h

        # Set corresponding mask tiles to True
        for ty in range(tile_y_start, tile_y_end + 1):
            for tx in range(tile_x_start, tile_x_end + 1):
                if 0 <= tx < N_TILE_X and 0 <= ty < N_TILE_Y:
                    mask[ty][tx] = True
    pic4 = p3


    p4 = image_fullres.copy()
    for i in range(0, N_TILE_Y):
        for j in range(0, N_TILE_X):
            y, x, h, w = i * tile_h, j * tile_w, tile_h, tile_w
            cv2.rectangle(p4, (x, y), (x + w, y + h), (0, 0, 255), 2 if not mask[i][j] else -1)
    pic5 = p4

    print(np.sum(mask), np.sum(mask)*tile_w*tile_h/640/640)


    pic = np.vstack((np.hstack((pic1, pic2, pic3)), np.hstack((pic4, pic5, pic4))))
    pic = cv2.resize(pic, (pic.shape[1]//3, pic.shape[0]//3))
    cv2.imshow('reassemble', pic)

    reassembler = Reassembler()
    imgh, imgw = image_fullres.shape[:2]
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if w<6 or h<6:
            continue
        margin = 6
        border = 2
        extra = margin + border
        x = max(0, x - extra)
        y = max(0, y - extra)
        w = min(w + extra * 2, imgw - x)
        h = min(h + extra * 2, imgh - y)
        reassembler.addRect(Rect(x, y, w, h))
    img_re = reassembler.reassemble(image_fullres, autosize=True, border=border, roi_extractor=None)
    cv2.imshow('reassembled', img_re)

    return img_re

cf = CropRowFinder()


def process_crop(image, image_fullres):
    pic1 = image.copy()
    # compute_ortho_image(pic1)
    process_owl_contour(image)
    reassemble_green(image_fullres)
    exg = rgba2ExGI(image)
    pic2 = cv2.cvtColor(exg.copy(), cv2.COLOR_GRAY2RGB)
    _, b1 = cv2.threshold(exg, 25, 255, cv2.THRESH_BINARY)
    # b1 = cv2.adaptiveThreshold(exg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 3)
    pic3 = cv2.cvtColor(b1.copy(), cv2.COLOR_GRAY2RGB)
    b2 = cv2.morphologyEx(b1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    b2 = b1
    b3 = cv2.morphologyEx(b2, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    # b3=cv2.subtract(b3.copy(),cv2.morphologyEx(b3.copy(), cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))))
    # b3 = cv2.erode(b1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    b3 = skeletonize_image(b3)
    b3 = cv2.morphologyEx(b3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    pic4 = cv2.cvtColor(b3.copy(), cv2.COLOR_GRAY2RGB)

    print(np.count_nonzero(b3))
    b3 = remove_small_components(b3, 0.002)

    pic5 = cv2.cvtColor(b3.copy(), cv2.COLOR_GRAY2RGB)

    rr=cf.find(b1)
    cv2.imshow("aa",rr)

    b11 = exp_decay(b1)
    _, b12 = cv2.threshold(b11, 220, 255, cv2.THRESH_BINARY)
    # b13 = cv2.morphologyEx(b12, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    b13 = cv2.morphologyEx(b12, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
    b14 = to_simple_polygons(b13)
    pic6 = cv2.cvtColor(b13, cv2.COLOR_GRAY2RGB)

    factor = 1
    # reduced = block_reduce(b1, block_size=(factor, factor), func=np.max)
    reduced = cv2.resize(b1, (b1.shape[1] // factor, b1.shape[0] // factor), interpolation=cv2.INTER_LINEAR)

    b21=cv2.Canny(b11, threshold1=10, threshold2=150)
    pic21 = cv2.cvtColor(b21.copy(), cv2.COLOR_GRAY2RGB)

    houghp = False
    if not houghp:
        lines = cv2.HoughLines(b21, 15, np.pi / 60, 800, min_theta=-np.pi / 4, max_theta=np.pi / 4)
        if lines is None:
            lines = []
        # lines = filter_close_lines(lines, 50, 3)
        print(len(lines))

        reducedimage = cv2.resize(image, (b1.shape[1] // factor, b1.shape[0] // factor), interpolation=cv2.INTER_LINEAR)
        result_image = draw_lines(cv2.cvtColor(image.copy(), cv2.COLOR_RGBA2RGB), lines, 8)
    else:
        lines = cv2.HoughLinesP(b21, 1, np.pi / 30, 10, minLineLength=10)
        if lines is None:
            lines = []
        print(len(lines))

        reducedimage = cv2.resize(image, (b1.shape[1] // factor, b1.shape[0] // factor), interpolation=cv2.INTER_LINEAR)
        result_image = draw_linesp(cv2.cvtColor(image.copy(), cv2.COLOR_RGBA2RGB), lines, 8)

    ransac=False
    if ransac:
        lines=ransac_lines(b3)
        result_image = draw_ransac_lines(cv2.cvtColor(image.copy(), cv2.COLOR_RGBA2RGB), lines, 8)

    pic10 = result_image

    preview_ROI = np.vstack((np.hstack((pic1, draw_ROI(image, b14))),
                             np.hstack((cv2.cvtColor(b11, cv2.COLOR_GRAY2RGB), cv2.cvtColor(b14, cv2.COLOR_GRAY2RGB)))))
    cv2.imshow('ROI', cv2.cvtColor(preview_ROI, cv2.COLOR_RGB2BGR))


    # return pic10
    return np.vstack((np.hstack((pic1, pic2, pic3)), np.hstack((pic4, pic21, pic10))))


def process(rgba, rgba0):
    # return reassemble_green(rgba0)
    return process_crop(rgba, rgba0)
    return process_hough(rgba)
    return process_canny(rgba)


def to_reassembled_video(video_file):
    from ultralytics.data.augment import LetterBox
    cap = cv2.VideoCapture(video_file)
    tot_proc_time = 0
    tot_proc_frames = 0

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('../data/output_video.mp4', fourcc, fps, (640, 640))

    try:
        while True:
            capture_st = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Reached end of video or failed to read frame.")
                break
            capture_el = time.time() - capture_st

            proc_st = time.time()
            result = reassemble_green(frame)
            proc_el = time.time() - proc_st

            tot_proc_time += proc_el
            tot_proc_frames += 1
            stat = (f"capture={int(capture_el * 1000)}ms  proc={int(proc_el * 1000)}ms    "
                    f"avg={int(tot_proc_time * 1000 / tot_proc_frames)}ms")
            print(stat)

            letterbox = LetterBox()
            rr = letterbox(image=result)
            cv2.imshow('preview', rr)
            out.write(rr)
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        out.release()

def video_preview(video_file):
    cap = cv2.VideoCapture(video_file)
    tot_proc_time = 0
    tot_proc_frames = 0

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    stabilizer = VidStab()

    try:
        while True:
            capture_st = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Reached end of video or failed to read frame.")
                break
            capture_el = time.time() - capture_st

            proc_st = time.time()
            frame0 = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            frame = cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4))

            # frame = stabilizer.stabilize_frame(input_frame=frame,
            #                                    smoothing_window=10)

            result = process(frame, frame0)
            proc_el = time.time() - proc_st

            tot_proc_time += proc_el
            tot_proc_frames += 1
            stat = (f"capture={int(capture_el * 1000)}ms  proc={int(proc_el * 1000)}ms    "
                    f"avg={int(tot_proc_time * 1000 / tot_proc_frames)}ms")
            print(stat)

            cv2.imshow('preview', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def image_preview(img_file):
    frame = cv2.imread(img_file)
    frame = frame.copy()
    frame = cv2.resize(frame, (frame.shape[1] // 3, frame.shape[0] // 3))
    proc_st = time.time()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    frame = cv2.resize(frame, (frame.shape[1] // 3, frame.shape[0] // 3))
    result = process(frame, frame)
    proc_el = time.time() - proc_st

    stat = f"proc={int(proc_el * 1000)}ms"
    print(stat)

    cv2.imshow('preview', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    cv2.waitKey()



# image_preview("data/rgimg1.jpg")

# to_reassembled_video(r"D:\projects\data_topdown\d3.MP4")
# video_preview("data/rg1.mp4")
# video_preview(r"D:\projects\Wundowie\poor-quality-downhill.MP4")
video_preview(r"../test_data/d2.mp4")
# video_preview(r"D:\projects\Wundowie\high-position-linearFOV.MP4")
