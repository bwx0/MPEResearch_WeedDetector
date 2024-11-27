import cv2
import numpy as np
import matplotlib.pyplot as plt


def p(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) * 255 / (max_val - min_val)
    return normalized_data


def compute_exgi(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    # Convert image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Split the channels
    R, G, B = np.array(image[:, :, 0], dtype=np.dtype(int)), np.array(image[:, :, 1], dtype=np.dtype(int)), np.array(
        image[:, :, 2], dtype=np.dtype(int))

    # Calculate ExGI
    ExGI = (2 * G - R - B)
    GLI = (2 * G - R - B) / (2 * G + R + B + 0.01)
    GNDVI = (G - R) / (G + R + 0.01)
    VARI = (G - R) / (G + R - B + 0.01)
    GRRI = G / (R + 0.01)
    VEG = G / (np.power(R + 1, 0.6666666) * np.power(B + 1, 0.3333333) + 1)
    MGRVI = (G * G - R * R) / (G * G + R * R + 0.01)
    RGVBI = (G - B * R) / (G * G + R * B + 0.01)

    ExGI_normalized = cv2.normalize(MGRVI, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    ExGI_normalized = np.uint8(ExGI_normalized)

    GRRI_normalized = cv2.normalize(GRRI, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    GRRI_normalized = np.uint8(GRRI_normalized)

    # Show the original and the ExGI images
    plt.figure(figsize=(20, 10))
    plt.subplot(3, 4, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(3, 4, 2)
    plt.title('Red')
    plt.imshow(R, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 4, 3)
    plt.title('Green')
    plt.imshow(G, cmap='gray')
    plt.axis('off')

    plt.subplot(3, 4, 4)
    plt.title('Blue')
    plt.imshow(B, cmap='gray')
    plt.axis('off')

    imgs = [ExGI, GLI, GNDVI, VARI, GRRI, VEG, MGRVI, RGVBI]
    names = ["ExGI", "GLI", "GNDVI", "VARI", "GRRI", "VEG", "MGRVI", "RGVBI"]
    for idx, im in enumerate(imgs):
        im_normalized = (cv2.normalize(p(im), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
        im_normalized = np.uint8(im_normalized)
        plt.subplot(3, 4, 5 + idx)
        plt.title(f'{names[idx]}')
        plt.imshow(im_normalized, cmap='gray')
        plt.axis('off')

    plt.show()

    # cv2.imwrite("Picture_ExGI.png", ExGI_normalized)
    # cv2.imwrite("Picture_GRRI.png", GRRI_normalized)


image_path = '../test_data/small.png'
compute_exgi(image_path)
