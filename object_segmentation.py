# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Imports
import os
import cv2
import numpy as np
import shutil

# Dataset path
proc_path = "/content/drive/MyDrive/coil-20-proc/coil-20-proc"

# NEW output folder (final version)
output_base = "/content/drive/MyDrive/coil-20-segmented-final"
stages = ["blur", "edges_canny", "edges_sobel", "morph", "mask"]

# Create output folders
for stage in stages:
    os.makedirs(os.path.join(output_base, stage), exist_ok=True)

# Noise removal
def noise_removal(img):
    return cv2.GaussianBlur(img, (5,5), 0)

# Canny edge detection
def edge_canny(img):
    return cv2.Canny(img, 50, 150)

# Sobel edge detection
def edge_sobel(img):
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(grad_x, grad_y)
    edges = np.uint8(edges)
    _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    return edges

# Morphological closing
def morphology(img):
    kernel = np.ones((3,3), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)

# Create mask using contours
def contour_mask(img, original_img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(original_img)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    return mask

# Process images
for img_name in os.listdir(proc_path):
    if not img_name.lower().endswith(".png"):
        continue

    img_path = os.path.join(proc_path, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        continue

    # Blur
    blur_img = noise_removal(img)
    cv2.imwrite(os.path.join(output_base, "blur", img_name), blur_img)

    # Canny edges
    edges_canny_img = edge_canny(blur_img)
    cv2.imwrite(os.path.join(output_base, "edges_canny", img_name), edges_canny_img)

    # Sobel edges
    edges_sobel_img = edge_sobel(blur_img)
    cv2.imwrite(os.path.join(output_base, "edges_sobel", img_name), edges_sobel_img)

    # Morphology using Canny edges
    morph_img = morphology(edges_canny_img)
    cv2.imwrite(os.path.join(output_base, "morph", img_name), morph_img)

    # Final mask using Canny pipeline
    mask_img = contour_mask(morph_img, img)
    cv2.imwrite(os.path.join(output_base, "mask", img_name), mask_img)

print("Processing complete.")

# Zip output folders
for stage in stages:
    folder_path = os.path.join(output_base, stage)
    zip_path = os.path.join(output_base, stage)
    shutil.make_archive(zip_path, 'zip', folder_path)

