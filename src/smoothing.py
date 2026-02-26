# Smoothing module
"""
smoothing.py

Purpose:
Provide moving average smoothing for frame-wise signals.

Responsibilities:
- Maintain rolling window
- Return smoothed value

Does NOT:
- Know about baseline
- Know about scaling
"""

from collections import deque
import numpy as np


class MovingAverage:
    def __init__(self, window_size=5):
        self.window = deque(maxlen=window_size)

    def update(self, value):
        self.window.append(value)
        return np.mean(self.window)