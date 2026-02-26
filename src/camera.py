# Camera module
"""
camera.py

Purpose:
Handles webcam initialization and frame retrieval.

Responsibilities:
- Open camera device
- Set resolution and FPS
- Return frames in BGR format

Does NOT:
- Run face detection
- Perform feature extraction
"""

import cv2
from config import CameraConfig

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(CameraConfig.DEVICE_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CameraConfig.WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CameraConfig.HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CameraConfig.FPS)

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

    def get_fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)