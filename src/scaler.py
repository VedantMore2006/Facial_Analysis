# ============================================================================
# DEVIATION SCALER
# ============================================================================
# Purpose: Transform raw features into normalized [0,1] deviations from baseline
# 
# Scaling Process (3 steps):
# 1. Z-Score: (value - μ) / σ
#    - Measures how many standard deviations away from baseline mean
#    - Negative = below baseline, Positive = above baseline
#    - Zero = exactly at baseline
# 
# 2. Sigmoid: 1 / (1 + e^(-z))
#    - Maps z-score to [0,1] range
#    - 0.5 = at baseline
#    - >0.5 = above baseline (more animated/engaged)
#    - <0.5 = below baseline (less animated/engaged)
# 
# 3. Interpretation:
#    - 0.5: Neutral (at baseline)
#    - 0.0-0.5: Below baseline (withdrawn/flat)
#    - 0.5-1.0: Above baseline (animated/engaged)
# 
# Example:
# If baseline smile width μ=0.2, σ=0.05:
# - Current smile=0.2: z=0, sigmoid(0)=0.5   (neutral)
# - Current smile=0.3: z=2, sigmoid(2)=0.88  (much more smiley)
# - Current smile=0.1: z=-2, sigmoid(-2)=0.12 (much less smiley)
# ============================================================================

# Scaler module
"""
scaler.py

Purpose:
Apply deviation modeling:
Z-score + sigmoid bounding

Output:
Scaled value in [0,1]
"""

import numpy as np


def z_score(value, mu, sigma):
    """
    Compute standardized z-score.
    
    Measures how many standard deviations the value is from the mean.
    
    Args:
        value: Current feature value
        mu: Baseline mean
        sigma: Baseline standard deviation
    
    Returns:
        Z-score (can be any real number)
        - 0: at baseline
        - positive: above baseline
        - negative: below baseline
    """
    return (value - mu) / sigma


def sigmoid(z):
    """
    Apply sigmoid function to map z-score to [0,1].
    
    Properties:
    - sigmoid(0) = 0.5 (baseline)
    - sigmoid(+inf) → 1.0
    - sigmoid(-inf) → 0.0
    - Smooth S-curve transition
    
    Args:
        z: Z-score (any real number)
    
    Returns:
        Value in range [0,1]
    """
    return 1 / (1 + np.exp(-z))


def scale_value(value, mu, sigma):
    """
    Scale a feature value to [0,1] based on baseline statistics.
    
    Combines z-score normalization with sigmoid bounding to create
    an interpretable deviation metric.
    
    Args:
        value: Current raw feature value
        mu: Baseline mean from personal session baseline
        sigma: Baseline standard deviation
    
    Returns:
        Scaled value in [0,1] where:
        - 0.5 = at baseline (typical behavior)
        - >0.5 = above baseline (elevated behavior)
        - <0.5 = below baseline (reduced behavior)
    """
    # Step 1: Compute z-score
    z = z_score(value, mu, sigma)
    # Step 2: Apply sigmoid to bound to [0,1]
    return sigmoid(z)