# Baseline module
"""
baseline.py

Purpose:
Manage personal session baseline (PSB).

Responsibilities:
- Collect feature samples during baseline window
- Compute μ and σ
- Apply sigma floor
- Lock baseline after window

Does NOT:
- Compute features
- Apply scaling
"""

import numpy as np
from config import BaselineConfig

class BaselineManager:
    def __init__(self):
        self.samples = {}
        self.baseline_stats = {}
        self.locked = False

    def collect(self, feature_dict):
        """
        feature_dict:
        {
            "au12": value,
            "head_velocity": value,
            ...
        }
        """

        if self.locked:
            return

        for key, value in feature_dict.items():
            if key not in self.samples:
                self.samples[key] = []

            self.samples[key].append(value)

    def finalize(self):
        """
        Compute μ and σ after baseline window.
        """

        for key, values in self.samples.items():
            values = np.array(values)

            mu = np.mean(values)
            sigma = np.std(values)

            # Sigma floor
            sigma = max(sigma, BaselineConfig.SIGMA_FLOOR)

            self.baseline_stats[key] = {
                "mu": mu,
                "sigma": sigma
            }

        self.locked = True
        print("Baseline locked.")
        print(self.baseline_stats)

    def get_stats(self):
        return self.baseline_stats