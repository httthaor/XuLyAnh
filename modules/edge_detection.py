import cv2
import numpy as np

def _to_gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def apply_roberts(img):
    gray = _to_gray(img)
    kernel_roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    gx = cv2.filter2D(gray, cv2.CV_32F, kernel_roberts_x)
    gy = cv2.filter2D(gray, cv2.CV_32F, kernel_roberts_y)
    mag = cv2.magnitude(gx, gy)
    result = cv2.convertScaleAbs(mag)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

def apply_prewitt(img):
    gray = _to_gray(img)
    kernel_x = np.array([[-1, 0, 1], 
                         [-1, 0, 1], 
                         [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1], 
                         [ 0,  0,  0], 
                         [ 1,  1,  1]], dtype=np.float32)

    gx = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
    gy = cv2.filter2D(gray, cv2.CV_32F, kernel_y)
    mag = cv2.magnitude(gx, gy)
    result = cv2.convertScaleAbs(mag)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

def apply_sobel(img, ksize=3):
    if ksize % 2 == 0:
        ksize += 1
    gray = _to_gray(img)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(sobel_x, sobel_y)
    result = cv2.convertScaleAbs(mag)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

def apply_canny(img, low_thresh=50, high_thresh=150):
    gray = _to_gray(img)
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)