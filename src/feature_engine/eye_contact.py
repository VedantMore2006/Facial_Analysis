# ============================================================================
# EYE CONTACT DETECTION
# ============================================================================
# Purpose: Estimate if person is looking at camera (making eye contact)
# 
# Detection Method:
# - Uses head yaw (horizontal rotation) as proxy for gaze direction
# - Small yaw magnitude = looking straight ahead = eye contact
# - Large yaw magnitude = looking left/right = no eye contact
# 
# Why Yaw Instead of True Gaze?
# - Simplified approach that works well in practice
# - True gaze requires iris tracking (more complex)
# - Head orientation strongly correlates with attention direction
# - Good enough for engagement/attention detection
# 
# Output Metrics:
# 1. Contact (binary): 1 if yaw < threshold, 0 otherwise
# 2. Contact Ratio: Percentage of recent frames with eye contact
# 
# Use Cases:
# - Attention monitoring (is person focused on screen?)
# - Engagement detection (looking at interviewer/camera?)
# - Distraction detection (looking away repeatedly?)
# ============================================================================

# Eye contact detection module
from collections import deque


class EyeContact:

    def __init__(self, fps, yaw_threshold=0.01, window_seconds=5):
        """
        Initialize eye contact detector.
        
        Uses sliding window to compute contact ratio over recent time period.
        
        Args:
            fps: Frames per second (from camera)
            yaw_threshold: Maximum absolute yaw for eye contact (default: 0.01)
                          Smaller = stricter (requires more direct gaze)
                          Larger = more lenient
            window_seconds: Time window for computing contact ratio (default: 5 sec)
        """
        self.yaw_threshold = yaw_threshold
        self.window_frames = int(window_seconds * fps)
        # Circular buffer storing 1s (contact) and 0s (no contact)
        self.buffer = deque(maxlen=self.window_frames)

    def update(self, yaw_value):
        """
        Update eye contact detection with current head yaw.
        
        Determines if person is making eye contact based on head orientation,
        and computes ratio of recent frames with eye contact.
        
        Args:
            yaw_value: Current head yaw (horizontal rotation)
                      - Near 0: Looking straight ahead (eye contact)
                      - Positive: Head turned right
                      - Negative: Head turned left
        
        Returns:
            Tuple of (contact, ratio)
            - contact: 1 if making eye contact this frame, 0 otherwise
            - ratio: Proportion of recent frames with eye contact [0-1]
                    * 0: No eye contact in window
                    * 0.5: Eye contact half the time
                    * 1: Constant eye contact
        """

        # Determine if current frame has eye contact
        # Contact if yaw magnitude is small (looking straight)
        contact = 1 if abs(yaw_value) < self.yaw_threshold else 0

        # Add to sliding window
        self.buffer.append(contact)

        # Handle edge case of empty buffer
        if len(self.buffer) == 0:
            return contact, 0

        # Compute ratio: (frames with contact) / (total frames in window)
        ratio = sum(self.buffer) / len(self.buffer)

        return contact, ratio