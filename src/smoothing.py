# ============================================================================
# TEMPORAL SMOOTHING
# ============================================================================
# Purpose: Reduce noise in per-frame feature signals using moving average
# 
# Why Smoothing?
# - Raw frame-to-frame features can be noisy due to:
#   * Minor detection jitter
#   * Natural micro-movements
#   * Lighting variations
# - Smoothing reveals true behavioral trends
# 
# Moving Average:
# - Maintains sliding window of recent values
# - Returns mean of window
# - Window size controls smoothness vs responsiveness trade-off
#   * Small window (e.g., 3): More responsive, noisier
#   * Large window (e.g., 10): Smoother, slower to respond
# - Default: 5 frames (good balance at 15 FPS)
# 
# Does NOT:
# - Know about baseline (smoothing happens before/after baseline)
# - Know about scaling (smooths raw or scaled values equally)
# - Change the range of values (preserves scale)
# ============================================================================

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
        """
        Initialize moving average filter.
        
        Uses a deque (double-ended queue) with maximum length for efficient
        rolling window implementation. When window is full, oldest value is
        automatically removed when new value is added.
        
        Args:
            window_size: Number of recent frames to average (default: 5)
                        Larger = smoother but slower response
                        Smaller = faster response but noisier
        """
        # Deque with maxlen automatically drops oldest value when full
        self.window = deque(maxlen=window_size)

    def update(self, value):
        """
        Add new value and return smoothed average.
        
        Appends value to window and computes mean of all values in window.
        During initial frames (before window is full), average uses only
        available values.
        
        Args:
            value: New feature value to add
        
        Returns:
            Mean of values in current window
        
        Example:
            smoother = MovingAverage(window_size=3)
            smoother.update(1.0)  # returns 1.0 (window: [1.0])
            smoother.update(2.0)  # returns 1.5 (window: [1.0, 2.0])
            smoother.update(3.0)  # returns 2.0 (window: [1.0, 2.0, 3.0])
            smoother.update(4.0)  # returns 3.0 (window: [2.0, 3.0, 4.0])
        """
        self.window.append(value)
        return np.mean(self.window)