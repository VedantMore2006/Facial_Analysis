# ============================================================================
# LANDMARK PROCESSOR
# ============================================================================
# Purpose: Extract minimal landmark subset from full 478-point MediaPipe mesh
# 
# Why Subset?
# - MediaPipe provides 478 landmarks (entire face)
# - Most features only need 20-30 specific points
# - Extracting subset reduces memory and processing overhead
# - Improves performance and clarity
# 
# Landmark Categories in Subset:
# - Mouth corners & lips: 61, 291, 13, 14 (for smile, speech detection)
# - Eye corners & eyelids: 33, 133, 362, 263, 160, 158, etc. (for blink, gaze)
# - Eyebrows: 70, 63, 105, 66, 107, 336, 296, 334, 293, 300 (for expressivity)
# - Nose & face center: 1, 152, 50, 280 (for head pose)
# - Iris landmarks: 468-477 (for precise gaze tracking)
# 
# Does NOT:
# - Compute features (just extracts raw coordinates)
# - Normalize by face size (that's done per-feature)
# - Apply smoothing
# ============================================================================

# Landmark processing module
"""
landmark_processor.py

Purpose:
Extract selected landmark subset from full 478 mesh.

Responsibilities:
- Receive MediaPipe landmarks
- Extract required indices
- Return normalized x,y coordinates

Does NOT:
- Compute features
- Normalize by IOD
- Apply smoothing
"""

# Define the minimal set of landmarks needed for all features
# Using sorted set to remove duplicates and maintain order
LANDMARK_SUBSET = sorted(set([
    # Mouth landmarks (for smile detection, response latency)
    61, 291,      # Left and right mouth corners
    13, 14,       # Upper and lower lip center
    
    # Eye outer corners (for IOD normalization)
    33, 263,      # Left and right eye outer corners
    
    # Left eye landmarks (for blink detection)
    133, 160, 158, 153, 144,
    
    # Right eye landmarks (for blink detection)
    362, 385, 387, 373, 380,
    
    # Eyebrow landmarks (for expressivity)
    70, 63, 105, 66, 107,     # Left eyebrow
    336, 296, 334, 293, 300,  # Right eyebrow
    
    # Nose & face center (for head pose)
    50, 280,      # Nose sides
    1, 152,       # Nose tip and chin
    
    # Iris landmarks (for precise eye gaze tracking)
    468, 469, 470, 471, 472,  # Left iris
    473, 474, 475, 476, 477   # Right iris
]))

def extract_subset(landmarks):
    """
    Extract required landmark subset from full MediaPipe mesh.
    
    Takes the 478-point landmark object and extracts only the points
    needed for feature computation, reducing data size and processing time.
    
    Args:
        landmarks: MediaPipe landmark object (478 points)
    
    Returns:
        Dictionary mapping landmark indices to (x, y) tuples
        {index: (x, y)}
        Coordinates are normalized [0,1] relative to frame dimensions
    
    Example:
        {33: (0.123, 0.456), 61: (0.234, 0.567), ...}
    """
    subset = {}

    # Extract each required landmark
    for idx in LANDMARK_SUBSET:
        lm = landmarks.landmark[idx]
        # Store normalized x,y coordinates (z is ignored for 2D analysis)
        subset[idx] = (lm.x, lm.y)

    return subset