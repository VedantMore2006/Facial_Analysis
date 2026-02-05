import cv2
import time
import os

os.environ['QT_QPA_PLATFORM'] = 'xcb'

class VideoSource:
    def __init__(self, source=0, target_fps=10):
        """
        source: 0 for webcam, or path to video file
        target_fps: how many frames per second we want to process
        """
        self.source = source
        self.target_fps = target_fps
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError("❌ Unable to open video source")

        self.frame_interval = 1.0 / self.target_fps
        self.last_frame_time = 0.0

    def get_frame(self):
        """
        Returns a sampled frame based on target_fps.
        If frame is skipped, returns None.
        """
        current_time = time.time()

        # Enforce sampling interval
        if current_time - self.last_frame_time < self.frame_interval:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        self.last_frame_time = current_time
        return frame

    def release(self):
        self.cap.release()
