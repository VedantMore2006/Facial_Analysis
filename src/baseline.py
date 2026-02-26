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
        self.feature_samples = {}
        self.landmark_sums = {}
        self.frame_count = 0

        self.feature_stats = {}
        self.baseline_positions = {}

        self.locked = False


    # ----------------------------
    # Feature collection
    # ----------------------------

    def collect_features(self, feature_dict):

        if self.locked:
            return

        for key, value in feature_dict.items():
            if key not in self.feature_samples:
                self.feature_samples[key] = []

            self.feature_samples[key].append(value)


    # ----------------------------
    # Landmark baseline collection
    # ----------------------------

    def collect_landmarks(self, subset):

        if self.locked:
            return

        for idx, (x, y) in subset.items():

            if idx not in self.landmark_sums:
                self.landmark_sums[idx] = [0.0, 0.0]

            self.landmark_sums[idx][0] += x
            self.landmark_sums[idx][1] += y

        self.frame_count += 1


    # ----------------------------
    # Finalize baseline
    # ----------------------------

    def finalize(self):

        if self.locked:
            return

        # Feature stats
        for key, values in self.feature_samples.items():

            values = np.array(values)

            mu = np.mean(values)
            sigma = np.std(values)
            sigma = max(sigma, BaselineConfig.SIGMA_FLOOR)

            self.feature_stats[key] = {
                "mu": mu,
                "sigma": sigma
            }

        # Landmark baseline positions
        for idx, (sx, sy) in self.landmark_sums.items():

            self.baseline_positions[idx] = (
                sx / self.frame_count,
                sy / self.frame_count
            )

        self.locked = True

        print("Baseline locked.")
        print("Feature stats:", self.feature_stats)


    # ----------------------------
    # Accessors
    # ----------------------------

    def get_feature_stats(self, key):
        return self.feature_stats.get(key, None)

    def get_baseline_positions(self):
        return self.baseline_positions