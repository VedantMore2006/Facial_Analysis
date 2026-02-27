# ============================================================================
# FACIAL EXPRESSIVITY DETECTION
# ============================================================================
# Purpose: Measure overall facial animation/movement variation
# 
# What is Expressivity?
# - Quantifies how much the face is moving/changing
# - Captures dynamic facial activity beyond specific expressions
# - Higher values = more animated, engaged, expressive
# - Lower values = flat affect, less engaged
# 
# Computation Method:
# 1. Track 16 key expressive points (eyebrows, eyes, mouth, cheeks)
# 2. For each point, measure displacement from previous frame
# 3. Average all displacements to get overall movement metric
# 4. Temporal: compares consecutive frames
# 
# Expressive Points Include:
# - Eyebrows (major expressivity indicators)
# - Mouth corners and lips (speech and emotion)
# - Cheeks (smiling, other expressions)
# - Nose reference points
# 
# Use Cases:
# - Engagement detection (animated vs withdrawn)
# - Emotion intensity (how much facial activity)
# - Conversational dynamics (active vs passive)
# ============================================================================

import numpy as np

# Define the specific landmarks that best capture facial expressivity
# Focuses on areas that move most during expressions (eyebrows, mouth, cheeks)
EXPRESSIVE_POINTS = [
    # Left eyebrow landmarks (major expressivity indicators)
    70, 63, 105, 66, 107,
    # Right eyebrow landmarks
    336, 296, 334, 293, 300,
    # Mouth corner landmarks (smiling, frowning, speech)
    61, 291,   # Left and right corners
    13, 14,    # Upper and lower lip center
    # Nose sides (additional reference points)
    50, 280
]

class Expressivity:

    def __init__(self):
        """
        Initialize expressivity tracker.
        
        Requires storage of previous frame's landmarks to compute
        frame-to-frame displacement.
        """
        self.prev_subset = None  # Previous frame's landmarks

    def compute(self, subset):
        """
        Compute facial expressivity as average displacement of key points.
        
        Measures how much expressive landmarks moved since last frame.
        On first frame, returns 0 (no previous frame to compare).
        
        Steps:
        1. For each expressive point, compute displacement from previous frame
        2. Calculate Euclidean distance for x,y movement
        3. Average all displacements
        
        Args:
            subset: Dictionary of current frame landmarks {index: (x, y)}
        
        Returns:
            Average displacement across all expressive points (float)
            - 0: No movement (or first frame)
            - ~0.001-0.005: Low expressivity (flat affect)
            - ~0.005-0.015: Moderate expressivity (normal conversation)
            - >0.015: High expressivity (animated, emphatic)
        """

        # First frame: can't compute displacement yet
        if self.prev_subset is None:
            self.prev_subset = subset
            return 0

        # List to store displacement values
        displacements = []

        # Compute displacement for each expressive point
        for idx in EXPRESSIVE_POINTS:
            # Current frame coordinates
            x, y = subset[idx]
            # Previous frame coordinates
            px, py = self.prev_subset[idx]

            # Compute displacement vector components
            dx = x - px
            dy = y - py

            # Calculate Euclidean distance (magnitude of movement)
            displacement = np.sqrt(dx*dx + dy*dy)
            displacements.append(displacement)

        # Store current frame for next iteration
        self.prev_subset = subset

        # Return average displacement across all points
        return np.mean(displacements)