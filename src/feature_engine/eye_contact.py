# Eye contact detection module
from collections import deque


class EyeContact:

    def __init__(self, fps, yaw_threshold=0.01, window_seconds=5):
        self.yaw_threshold = yaw_threshold
        self.window_frames = int(window_seconds * fps)
        self.buffer = deque(maxlen=self.window_frames)

    def update(self, yaw_value):

        # Contact if yaw magnitude small
        contact = 1 if abs(yaw_value) < self.yaw_threshold else 0

        self.buffer.append(contact)

        if len(self.buffer) == 0:
            return contact, 0

        ratio = sum(self.buffer) / len(self.buffer)

        return contact, ratio