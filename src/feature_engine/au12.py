# ============================================================================
# AU12 (SMILE) DETECTION
# ============================================================================
# Purpose: Compute smile intensity using Action Unit 12 (lip corner puller)
# 
# What is AU12?
# - Action Unit 12 from Facial Action Coding System (FACS)
# - Represents "lip corner puller" muscle activation
# - Primary indicator of genuine smiling
# 
# Computation Method:
# 1. Measure distance between left and right mouth corners
# 2. Normalize by Interocular Distance (IOD) to account for face size
# 3. Larger ratio = wider smile
# 
# Why Normalize by IOD?
# - Different people have different face sizes
# - Close vs far from camera changes apparent size
# - IOD (eye distance) is stable reference measurement
# - Makes smile metric comparable across people and distances
# 
# Landmarks Used:
# - 61: Left mouth corner
# - 291: Right mouth corner
# - 33: Left eye outer corner
# - 263: Right eye outer corner
# ============================================================================

"""
au12.py

Smile width normalized by Interocular Distance (IOD)
"""

import numpy as np


def compute_iod(subset):
    """
    Compute Interocular Distance (distance between outer eye corners).
    
    IOD is used as a normalizing factor for facial measurements.
    It's relatively stable across expressions and head poses.
    
    Args:
        subset: Dictionary of landmarks {index: (x, y)}
    
    Returns:
        Euclidean distance between eye corners (float)
    """
    # Left eye outer corner (landmark 33)
    x1, y1 = subset[33]
    # Right eye outer corner (landmark 263)
    x2, y2 = subset[263]

    # Euclidean distance between the two points
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def compute_au12(subset):
    """
    Compute AU12 (smile intensity) as normalized mouth width.
    
    Formula: smile_width / IOD
    
    Steps:
    1. Compute distance between mouth corners (smile width)
    2. Compute interocular distance (IOD)
    3. Divide smile width by IOD for normalized metric
    
    Args:
        subset: Dictionary of landmarks {index: (x, y)}
    
    Returns:
        Normalized smile intensity (float)
        - ~0.5-0.6: Neutral expression
        - >0.6: Smiling
        - >0.7: Wide smile
    """
    # Left mouth corner (landmark 61)
    x1, y1 = subset[61]
    # Right mouth corner (landmark 291)
    x2, y2 = subset[291]

    # Calculate distance between mouth corners
    smile_width = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    # Get normalizing factor (eye distance)
    iod = compute_iod(subset)

    # Prevent division by zero (shouldn't happen with valid face)
    if iod == 0:
        return 0

    # Return normalized smile intensity
    return smile_width / iod