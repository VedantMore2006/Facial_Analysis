"""
Baseline Statistics Collection

Collects feature values for a specified duration and computes
mean (μ) and standard deviation (σ) for personalized normalization.
"""

import numpy as np


class BaselineCollector:
    """
    Collects baseline statistics for feature normalization.
    
    Accumulates feature values over a specified duration (default 10 seconds)
    and computes mean and standard deviation for each feature.
    """

    def __init__(self, duration=10):
        """
        Initialize baseline collector.
        
        Args:
            duration: Collection duration in seconds (default: 10)
        """

        self.duration = duration
        self.start_time = None

        self.data = {}

        self.mean = {}
        self.std = {}

        self.completed = False

    def update(self, feature_values, timestamp):
        """
        Add feature values at current timestamp.
        
        Args:
            feature_values: Dict of feature name -> value
            timestamp: Current timestamp in seconds
        """

        if self.start_time is None:
            self.start_time = timestamp

        elapsed = timestamp - self.start_time

        if elapsed > self.duration:
            self.compute_statistics()
            self.completed = True
            return

        for name, value in feature_values.items():

            if name not in self.data:
                self.data[name] = []

            self.data[name].append(value)

    def compute_statistics(self):
        """
        Compute mean (μ) and standard deviation (σ) for each feature.
        
        Uses outlier-resistant approach: clips extreme values before
        computing statistics to prevent baseline corruption.
        
        Fallback to μ=0, σ=1 if insufficient data.
        """

        for name, values in self.data.items():

            if len(values) < 2:
                self.mean[name] = 0
                self.std[name] = 1
                continue

            values_array = np.array(values)

            # Remove extreme outliers (beyond 0.1% and 99.9% percentiles)
            p1 = np.percentile(values_array, 0.1)
            p99 = np.percentile(values_array, 99.9)

            # Clip values to reasonable range
            clipped = np.clip(values_array, p1, p99)

            self.mean[name] = float(np.mean(clipped))
            self.std[name] = float(np.std(clipped))

            # Prevent division by zero in normalization
            if self.std[name] < 1e-6:
                self.std[name] = 1.0

    def get_baseline(self):
        """
        Get computed baseline statistics.
        
        Returns:
            Tuple of (mean_dict, std_dict)
        """

        return self.mean, self.std
