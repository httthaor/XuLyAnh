import cv2
import numpy as np

def apply_averaging(img, ksize=3):
    if ksize % 2 == 0:
        ksize += 1
    result = cv2.blur(img, (ksize, ksize))
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_gaussian(img, ksize=3):
    if ksize % 2 == 0:
        ksize += 1
    result = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_median(img, ksize=3):
    if ksize % 2 == 0:
        ksize += 1
    result = cv2.medianBlur(img, ksize)
    return np.clip(result, 0, 255).astype(np.uint8)