"""Image enhancement applied before detection (from the original drone)."""
import cv2
import numpy as np

SHARPEN_KERNEL = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)


def equalize_luma(frame):
    """Equalise brightness only: histogram on the Y channel keeps colours intact."""
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


def sharpen(frame):
    return cv2.filter2D(frame, -1, SHARPEN_KERNEL)


def preprocess(frame, equalize=True, sharpen_edges=True):
    if equalize:
        frame = equalize_luma(frame)
    if sharpen_edges:
        frame = sharpen(frame)
    return frame
