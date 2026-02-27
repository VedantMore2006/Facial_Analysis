# ============================================================================
# BLINK DETECTION ENGINE
# ============================================================================
# Purpose: Detect blinks and compute blink rate using Eye Aspect Ratio (EAR)
# 
# What is EAR (Eye Aspect Ratio)?
# - Geometric ratio that indicates eye openness
# - Formula: (vertical1 + vertical2) / (2 * horizontal)
# - Higher value = eye more open
# - Lower value = eye more closed
# - Typical threshold: ~0.22 (below this = eye closed)
# 
# Blink Detection Logic:
# 1. Compute EAR for each frame
# 2. Eye closed: EAR < threshold for N consecutive frames
# 3. Blink registered: Eye was closed, now opens again
# 4. Prevent false positives by requiring minimum closed duration
# 
# Blink Rate Calculation:
# - Counts blinks in sliding time window (e.g., last 10 seconds)
# - Converts to blinks per minute (BPM)
# - Normal rate: 15-20 BPM
# - High rate may indicate stress, fatigue, or nervousness
# 
# Landmarks Used (per eye):
# - 1 point: outer corner (horizontal reference)
# - 4 points: top and bottom eyelid (vertical measurements)
# - 1 point: inner corner (horizontal reference)
# ============================================================================

# Blink detection module
import numpy as np
from collections import deque


class BlinkDetector:

    def __init__(self, fps, threshold=0.22, min_frames=2, window_seconds=10):
        """
        Initialize blink detector.
        
        Args:
            fps: Frames per second (from camera)
            threshold: EAR value below which eye is considered closed (default: 0.22)
            min_frames: Minimum consecutive frames eye must be closed to count as blink
            window_seconds: Time window for computing blink rate (default: 10 sec)
        """
        self.threshold = threshold
        self.min_frames = min_frames
        self.fps = fps

        # Blink detection state
        self.counter = 0              # Consecutive frames with eye closed
        self.blink_count = 0          # Total blinks detected

        # Sliding window for blink rate calculation
        self.window_frames = int(window_seconds * fps)
        self.event_buffer = deque(maxlen=self.window_frames)  # Stores 1s and 0s

    # ----------------------------
    # EAR (Eye Aspect Ratio) Calculation
    # ----------------------------

    def compute_ear(self, eye_points):
        """
        Compute Eye Aspect Ratio (EAR) for one eye.
        
        EAR = (vertical1 + vertical2) / (2 * horizontal)
        
        Where:
        - vertical1: distance between upper and lower eyelid (outer side)
        - vertical2: distance between upper and lower eyelid (inner side)
        - horizontal: distance between eye corners
        
        Args:
            eye_points: List of 6 (x,y) tuples representing eye landmarks
                       [outer_corner, top_outer, top_inner, inner_corner, bottom_inner, bottom_outer]
        
        Returns:
            EAR value (float)
            - ~0.3: Eye open
            - ~0.2: Eye partially closed
            - <0.15: Eye closed
        """
        p1, p2, p3, p4, p5, p6 = eye_points

        # Compute vertical distances (eyelid openings)
        vertical1 = np.linalg.norm(np.array(p2) - np.array(p6))  # Outer side
        vertical2 = np.linalg.norm(np.array(p3) - np.array(p5))  # Inner side
        
        # Compute horizontal distance (eye width)
        horizontal = np.linalg.norm(np.array(p1) - np.array(p4))

        # Prevent division by zero
        if horizontal == 0:
            return 0

        # Compute and return EAR
        return (vertical1 + vertical2) / (2.0 * horizontal)

    # ----------------------------
    # Update Blink State (called per frame)
    # ----------------------------

    def update(self, subset):
        """
        Process one frame to detect blinks and compute blink rate.
        
        Steps:
        1. Extract eye landmark points
        2. Compute EAR for both eyes
        3. Average the two EAR values
        4. Detect blink events using state machine
        5. Update sliding window and compute blink rate
        
        Args:
            subset: Dictionary of landmarks {index: (x, y)}
        
        Returns:
            Tuple of (ear, blink_event, blink_rate)
            - ear: Current Eye Aspect Ratio (float)
            - blink_event: 1 if blink detected this frame, 0 otherwise
            - blink_rate: Blinks per minute in sliding window
        """

        # Extract left eye landmarks (6 points)
        left_eye = [
            subset[33],   # Outer corner
            subset[160],  # Top outer
            subset[158],  # Top inner
            subset[133],  # Inner corner
            subset[153],  # Bottom inner
            subset[144]   # Bottom outer
        ]

        # Extract right eye landmarks (6 points)
        right_eye = [
            subset[362],  # Outer corner
            subset[385],  # Top outer
            subset[387],  # Top inner
            subset[263],  # Inner corner
            subset[373],  # Bottom inner
            subset[380]   # Bottom outer
        ]

        # Compute EAR for each eye
        left_ear = self.compute_ear(left_eye)
        right_ear = self.compute_ear(right_eye)

        # Average of both eyes (reduces false positives from single-eye closure)
        ear = (left_ear + right_ear) / 2.0

        # Initialize blink event flag
        blink_event = 0

        # ================================================================
        # BLINK DETECTION STATE MACHINE
        # ================================================================
        if ear < self.threshold:
            # Eye is closed, increment counter
            self.counter += 1
        else:
            # Eye is open
            if self.counter >= self.min_frames:
                # Eye was closed for minimum duration, register blink
                blink_event = 1
                self.blink_count += 1
            # Reset counter
            self.counter = 0

        # ================================================================
        # BLINK RATE CALCULATION (using sliding window)
        # ================================================================
        # Store blink event (1 or 0) in sliding window
        self.event_buffer.append(blink_event)

        # Only compute rate once window is full
        if len(self.event_buffer) == self.window_frames:
            # Count blinks in window
            blinks_in_window = sum(self.event_buffer)
            # Convert to blinks per minute
            # (blinks / frames) * fps * 60 seconds
            blink_rate = (blinks_in_window / len(self.event_buffer)) * self.fps * 60
        else:
            # Window not full yet, return 0
            blink_rate = 0

        return ear, blink_event, blink_rate