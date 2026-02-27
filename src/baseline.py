# ============================================================================
# PERSONAL SESSION BASELINE (PSB) MANAGER
# ============================================================================
# Purpose: Collect and compute statistical baseline for personalized normalization
# 
# Key Concepts:
# - Collects feature samples during initial baseline window
# - Computes mean (μ) and standard deviation (σ) for each feature
# - Computes average landmark positions for face geometry baseline
# - Once finalized, baseline is locked and cannot be modified
# 
# Why Baseline Matters:
# - Everyone has different neutral expressions and movement patterns
# - Baseline captures individual's "normal" behavior
# - Allows deviation detection: how much current behavior differs from baseline
# 
# Does NOT:
# - Compute features (receives pre-computed features)
# - Apply scaling (just stores statistics)
# ============================================================================

"""
baseline.py

Purpose:
Personal Session Baseline (PSB) manager.

Responsibilities:
- Collect feature samples
- Collect landmark positions
- Compute μ and σ for each feature
- Compute baseline mean landmark positions
- Apply sigma floor
- Lock baseline

Does NOT:
- Compute features
- Apply scaling
"""

import numpy as np
from config import BaselineConfig


class BaselineManager:

    def __init__(self):
        # Storage for feature values during baseline collection
        self.feature_samples = {}      # {feature_name: [values]}
        
        # Storage for landmark coordinates during baseline collection
        self.landmark_sums = {}        # {landmark_idx: [sum_x, sum_y]}
        self.frame_count = 0           # Number of frames processed

        # Computed statistics after finalization
        self.feature_stats = {}        # {feature_name: {"mu": mean, "sigma": std}}
        self.baseline_positions = {}   # {landmark_idx: (mean_x, mean_y)}

        # Lock status - prevents further modifications after finalization
        self.locked = False


    # ----------------------------
    # Feature Collection Method
    # ----------------------------

    def collect_features(self, feature_dict):
        """
        Accumulate feature values during baseline phase.
        
        Stores each feature value in a list for later statistical computation.
        Ignores calls after baseline is locked.
        
        Args:
            feature_dict: Dictionary mapping feature names to their current values
                         e.g., {"au12": 0.23, "blink_rate": 15.2}
        """
        # Prevent modifications after baseline is finalized
        if self.locked:
            return

        # Store each feature value in its respective list
        for key, value in feature_dict.items():
            if key not in self.feature_samples:
                self.feature_samples[key] = []  # Initialize list if first occurrence

            self.feature_samples[key].append(value)


    # ----------------------------
    # Landmark Baseline Collection Method
    # ----------------------------

    def collect_landmarks(self, subset):
        """
        Accumulate landmark positions for computing average face geometry.
        
        Sums up x,y coordinates for each landmark across all baseline frames.
        These sums will be divided by frame count to get mean positions.
        
        Args:
            subset: Dictionary mapping landmark indices to (x, y) coordinates
        """
        # Prevent modifications after baseline is finalized
        if self.locked:
            return

        # Accumulate coordinate sums for each landmark
        for idx, (x, y) in subset.items():

            if idx not in self.landmark_sums:
                self.landmark_sums[idx] = [0.0, 0.0]  # [sum_x, sum_y]

            # Add current coordinates to running sum
            self.landmark_sums[idx][0] += x
            self.landmark_sums[idx][1] += y

        # Track total frames for computing averages
        self.frame_count += 1


    # ----------------------------
    # Finalize Baseline Method
    # ----------------------------

    def finalize(self):
        """
        Compute final baseline statistics and lock the baseline.
        
        Called once baseline collection period ends. Performs:
        1. Computes mean (μ) and std deviation (σ) for each feature
        2. Applies sigma floor to prevent division by zero
        3. Computes average landmark positions
        4. Locks baseline to prevent further modifications
        
        After this, baseline stats are used for deviation scaling.
        """
        # Already locked, nothing to do
        if self.locked:
            return

        # ----------------------------
        # Compute Feature Statistics
        # ----------------------------
        for key, values in self.feature_samples.items():

            values = np.array(values)

            # Calculate mean and standard deviation
            mu = np.mean(values)
            sigma = np.std(values)
            
            # Apply floor to sigma to prevent division by zero in scaling
            sigma = max(sigma, BaselineConfig.SIGMA_FLOOR)

            # Store computed statistics
            self.feature_stats[key] = {
                "mu": mu,
                "sigma": sigma
            }

        # ----------------------------
        # Compute Landmark Baseline Positions
        # ----------------------------
        for idx, (sx, sy) in self.landmark_sums.items():

            # Divide accumulated sums by frame count to get averages
            self.baseline_positions[idx] = (
                sx / self.frame_count,
                sy / self.frame_count
            )

        # Lock baseline to prevent further modifications
        self.locked = True

        print("Baseline locked.")
        print("Feature stats:", self.feature_stats)


    # ----------------------------
    # Accessor Methods
    # ----------------------------

    def get_feature_stats(self, key):
        """
        Retrieve baseline statistics for a specific feature.
        
        Args:
            key: Feature name (e.g., "au12", "blink_rate")
        
        Returns:
            Dictionary with "mu" and "sigma" keys, or None if not found
        """
        return self.feature_stats.get(key, None)

    def get_baseline_positions(self):
        """
        Retrieve baseline landmark positions.
        
        Returns:
            Dictionary mapping landmark indices to (mean_x, mean_y) tuples
        """
        return self.baseline_positions