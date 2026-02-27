# Blink detection module
import numpy as np
from collections import deque


class BlinkDetector:

    def __init__(self, fps, threshold=0.22, min_frames=2, window_seconds=10):
        self.threshold = threshold
        self.min_frames = min_frames
        self.fps = fps

        self.counter = 0
        self.blink_count = 0

        self.window_frames = int(window_seconds * fps)
        self.event_buffer = deque(maxlen=self.window_frames)

    # ----------------------------
    # EAR Calculation
    # ----------------------------

    def compute_ear(self, eye_points):

        p1, p2, p3, p4, p5, p6 = eye_points

        vertical1 = np.linalg.norm(np.array(p2) - np.array(p6))
        vertical2 = np.linalg.norm(np.array(p3) - np.array(p5))
        horizontal = np.linalg.norm(np.array(p1) - np.array(p4))

        if horizontal == 0:
            return 0

        return (vertical1 + vertical2) / (2.0 * horizontal)

    # ----------------------------
    # Update Blink State
    # ----------------------------

    def update(self, subset):

        left_eye = [
            subset[33], subset[160], subset[158],
            subset[133], subset[153], subset[144]
        ]

        right_eye = [
            subset[362], subset[385], subset[387],
            subset[263], subset[373], subset[380]
        ]

        left_ear = self.compute_ear(left_eye)
        right_ear = self.compute_ear(right_eye)

        ear = (left_ear + right_ear) / 2.0

        blink_event = 0

        if ear < self.threshold:
            self.counter += 1
        else:
            if self.counter >= self.min_frames:
                blink_event = 1
                self.blink_count += 1
            self.counter = 0

        # Store blink events in sliding window
        self.event_buffer.append(blink_event)

        # Blinks per minute
        if len(self.event_buffer) == self.window_frames:
            blinks_in_window = sum(self.event_buffer)
            blink_rate = (blinks_in_window / len(self.event_buffer)) * self.fps * 60
        else:
            blink_rate = 0

        return ear, blink_event, blink_rate