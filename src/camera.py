# ============================================================================
# CAMERA MODULE
# ============================================================================
# Purpose: Abstracts webcam hardware interaction using OpenCV
# 
# Responsibilities:
# - Initialize webcam with configured resolution and FPS
# - Provide clean interface for frame capture
# - Handle resource cleanup
# 
# Design Philosophy:
# - Single responsibility: only handles camera I/O
# - No face detection or feature extraction logic
# - Configuration driven (uses CameraConfig)
# ============================================================================

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
        """
        Initialize camera with configured parameters.
        
        Opens video capture device and applies resolution/FPS settings.
        Uses CameraConfig for all parameters (device ID, width, height, FPS).
        """
        # Open camera device (0 = default webcam)
        self.cap = cv2.VideoCapture(CameraConfig.DEVICE_ID)
        
        # Configure camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CameraConfig.WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CameraConfig.HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CameraConfig.FPS)

    def read(self):
        """
        Capture a single frame from the camera.
        
        Returns:
            Tuple of (success_flag, frame)
            - success_flag: Boolean indicating if frame was captured successfully
            - frame: NumPy array containing image in BGR format, or None if failed
        """
        return self.cap.read()

    def release(self):
        """
        Release camera hardware resources.
        
        IMPORTANT: Always call this when done to free the camera device.
        Failure to release can cause camera lock issues.
        """
        self.cap.release()

    def get_fps(self):
        """
        Get the actual FPS value from the camera.
        
        Note: Actual FPS may differ from configured FPS depending on hardware.
        
        Returns:
            Float representing frames per second
        """
        return self.cap.get(cv2.CAP_PROP_FPS)