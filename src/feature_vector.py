# ============================================================================
# PRIVACY-SAFE FEATURE VECTOR BUILDER
# ============================================================================
# Purpose: Construct final privacy-safe feature vector for logging
# 
# Safety Mechanisms:
# 1. Clips all values to [0, 1] range (ensures bounded output)
# 2. Rounds to 4 decimal places (prevents precision leakage)
# 3. Standardized ordering (consistent across all frames)
# 
# Feature Vector Contents (6 features):
# 1. S_AU12: Smile intensity (mouth corner distance)
# 2. S_AUVar: Expressivity (facial movement variation)
# 3. S_HeadVelocity: Head movement speed
# 4. S_EyeContact: Eye gaze alignment ratio
# 5. S_BlinkRate: Blink frequency
# 6. S_ResponseLatency: Reaction time to stimulus
# 
# Does NOT:
# - Compute features (receives pre-scaled values)
# - Access landmarks (completely isolated from raw data)
# ============================================================================

"""
feature_vector.py

Construct privacy-safe scaled feature vector.

Responsibilities:
- Accept scaled feature inputs
- Clip to [0,1]
- Round to 4 decimals
- Return ordered vector

Does NOT:
- Compute features
- Access landmarks
"""

import numpy as np


def clip_and_round(value):
    """
    Ensure value is bounded and has consistent precision.
    
    Steps:
    1. Clip to [0, 1] range to prevent out-of-bounds values
    2. Round to 4 decimal places for consistent formatting
    3. Convert to Python float (not NumPy type)
    
    Args:
        value: Raw scaled value (may be outside [0,1] due to extremes)
    
    Returns:
        Bounded and rounded float value
    """
    # Ensure value is within valid range
    value = np.clip(value, 0.0, 1.0)
    # Round to 4 decimal places for consistency
    return round(float(value), 4)


def build_feature_vector(
    s_au12,
    s_expressivity,
    s_head_velocity,
    s_eye_contact,
    s_blink_rate,
    s_response_latency
):
    """
    Build ordered feature vector from scaled inputs.
    
    Takes 6 scaled feature values and constructs a standardized vector.
    Each value is clipped and rounded for safety and consistency.
    
    Args:
        s_au12: Scaled smile intensity [0-1]
        s_expressivity: Scaled facial expressivity [0-1]
        s_head_velocity: Scaled head movement speed [0-1]
        s_eye_contact: Scaled eye contact ratio [0-1]
        s_blink_rate: Scaled blink frequency [0-1]
        s_response_latency: Scaled response time [0-1]
    
    Returns:
        List of 6 clipped and rounded feature values
    """
    vector = [
        clip_and_round(s_au12),
        clip_and_round(s_expressivity),
        clip_and_round(s_head_velocity),
        clip_and_round(s_eye_contact),
        clip_and_round(s_blink_rate),
        clip_and_round(s_response_latency),
    ]

    return vector
