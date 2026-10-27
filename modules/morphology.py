import cv2
import numpy as np

def apply_opening(img, ksize=3):
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def apply_closing(img, ksize=3):
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def apply_dilation(img, ksize=3):
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.dilate(img, kernel, iterations=1)

def apply_erosion(img, ksize=3):
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.erode(img, kernel, iterations=1)