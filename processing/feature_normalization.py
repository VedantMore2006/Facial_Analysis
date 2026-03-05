"""
Feature Normalization

Applies z-score normalization and sigmoid scaling to map features to [0, 1] range.

Formula:
    z = (value - μ) / σ
    scaled = 1 / (1 + e^(-z))

Output interpretation:
    0.0 - 0.3 : suppressed behavior
    0.5       : baseline
    0.7 - 1.0 : elevated behavior
"""

import numpy as np


def normalize_feature(value, mean, std):
    """
    Apply z-score normalization and sigmoid scaling with clipping.
    
    Args:
        value: Current feature value
        mean: Baseline mean (μ)
        std: Baseline standard deviation (σ)
        
    Returns:
        Scaled value in range [0, 1]
    """

    if std == 0 or std < 1e-6:
        z = 0
    else:
        z = (value - mean) / std
        # Clip extreme z-scores to prevent overflow
        z = np.clip(z, -10, 10)

    scaled = 1 / (1 + np.exp(-z))

    # Hard clip to [0, 1] to ensure valid range
    return float(np.clip(scaled, 0, 1))


class FeatureNormalizer:
    """
    Normalizes all features using baseline statistics.
    
    Applies z-score normalization followed by sigmoid scaling
    to map feature values to [0, 1] range.
    """

    def __init__(self, mean, std):
        """
        Initialize normalizer with baseline statistics.
        
        Args:
            mean: Dict of feature name -> mean (μ)
            std: Dict of feature name -> std (σ)
        """

        self.mean = mean
        self.std = std

    def normalize(self, feature_values):
        """
        Normalize all feature values.
        
        Args:
            feature_values: Dict of feature name -> raw value
            
        Returns:
            Dict of feature name -> normalized value [0, 1]
        """

        normalized = {}

        for name, value in feature_values.items():

            mu = self.mean.get(name, 0)
            sigma = self.std.get(name, 1)

            normalized[name] = normalize_feature(value, mu, sigma)

        return normalized
