import cv2
import numpy as np

def apply_negative(img):
    return 255 - img

def apply_log(img):
    if len(img.shape) == 2:
        img_float = img.astype(np.float32)
        c = 255.0 / np.log(1.0 + np.max(img_float) + 1e-5)
        return np.clip(c * np.log(1.0 + img_float), 0, 255).astype(np.uint8)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    c = 255.0 / np.log(1.0 + np.max(v) + 1e-5)
    hsv[:, :, 2] = np.clip(c * np.log(1.0 + v), 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def apply_gamma(img, gamma=1.0):
    if gamma <= 0:
        gamma = 0.1
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    
    if len(img.shape) == 2:
        return cv2.LUT(img, table)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = cv2.LUT(hsv[:, :, 2], table)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def apply_hist_equalization(img):
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

def apply_clahe(img, clip_limit=2.0, grid_size=8):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    if len(img.shape) == 2:
        return clahe.apply(img)
    
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)