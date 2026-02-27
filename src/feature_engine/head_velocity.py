# ============================================================================
# HEAD YAW VELOCITY DETECTION
# ============================================================================
# Purpose: Measure head rotation speed (horizontal turning)
# 
# What is Head Yaw?
# - Yaw = rotation around vertical axis (shaking head "no")
# - Positive yaw = head turned right
# - Negative yaw = head turned left
# - Zero yaw = facing camera directly
# 
# Yaw Computation:
# - Compares nose position relative to eye corners
# - If nose closer to left eye: head turned right (positive yaw)
# - If nose closer to right eye: head turned left (negative yaw)
# - Equal distances: head centered (zero yaw)
# 
# Velocity Computation:
# - Absolute change in yaw between consecutive frames
# - Higher velocity = faster head movement
# - Captures head turning dynamics
# 
# Use Cases:
# - Engagement indicator (head movements during conversation)
# - Attention shifts (turning to look at different things)
# - Fidgeting/restlessness detection
# - Disagreement detection (head shaking)
# ============================================================================

import numpy as np


class HeadYawVelocity:

    def __init__(self):
        """
        Initialize head yaw velocity tracker.
        
        Requires storage of previous yaw to compute velocity.
        """
        self.prev_yaw = None  # Previous frame's yaw value

    def compute_yaw(self, subset):
        """
        Compute head yaw angle using nose-to-eye distances.
        
        Method:
        - Calculate distance from nose to left eye outer corner
        - Calculate distance from nose to right eye outer corner
        - Yaw = dist_left - dist_right
        
        Interpretation:
        - Positive: Nose closer to left eye (head turned right)
        - Negative: Nose closer to right eye (head turned left)
        - Zero: Equal distances (head facing forward)
        
        Args:
            subset: Dictionary of landmarks {index: (x, y)}
        
        Returns:
            Yaw value (float)
            - ~0: Facing camera
            - >0: Turned right
            - <0: Turned left
        """
        # Nose tip (landmark 1)
        nx, ny = subset[1]
        # Left eye outer corner (landmark 33)
        lx, ly = subset[33]
        # Right eye outer corner (landmark 263)
        rx, ry = subset[263]

        # Distance from nose to left eye
        dist_left = np.sqrt((nx - lx)**2 + (ny - ly)**2)
        # Distance from nose to right eye
        dist_right = np.sqrt((nx - rx)**2 + (ny - ry)**2)

        # Difference in distances indicates head rotation
        return dist_left - dist_right

    def compute_velocity(self, subset):
        """
        Compute head yaw velocity (rate of change).
        
        Velocity = absolute difference in yaw between consecutive frames.
        
        On first frame, returns 0 (no previous yaw to compare).
        
        Args:
            subset: Dictionary of current frame landmarks {index: (x, y)}
        
        Returns:
            Velocity (float)
            - 0: No head movement (or first frame)
            - Small (~0.001-0.005): Slow/stable head position
            - Medium (~0.005-0.02): Normal head movements
            - Large (>0.02): Rapid head turning
        """
        # Compute current yaw
        yaw = self.compute_yaw(subset)

        # First frame: can't compute velocity yet
        if self.prev_yaw is None:
            self.prev_yaw = yaw
            return 0

        # Compute absolute change (velocity = rate of rotation)
        velocity = abs(yaw - self.prev_yaw)

        # Store current yaw for next iteration
        self.prev_yaw = yaw

        return velocity

    def get_current_yaw(self):
        """
        Get the most recent yaw value.
        
        Used by other modules (e.g., eye contact detection).
        
        Returns:
            Current yaw value, or 0 if not yet computed
        """
        return self.prev_yaw if self.prev_yaw is not None else 0