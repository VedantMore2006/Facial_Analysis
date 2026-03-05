"""
EMA Smoothing for Behavioral Features

Exponential Moving Average: EMA_t = α * x_t + (1 − α) * EMA_(t−1)
"""


class EMASmoother:
    """
    Exponential Moving Average Smoother
    """

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.previous = None

    def update(self, value):
        """
        Update EMA value.
        """

        if self.previous is None:
            self.previous = value
            return value

        ema = self.alpha * value + (1 - self.alpha) * self.previous

        self.previous = ema

        return ema


class FeatureEMAManager:
    """
    Manages EMA smoothers for multiple features.
    
    One smoother per feature, automatically created on first use.
    """

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.smoothers = {}

    def smooth(self, feature_name, value):
        """
        Smooth a feature value using its dedicated EMA smoother.
        
        Args:
            feature_name: Name of the feature
            value: Raw feature value
            
        Returns:
            Smoothed value
        """

        if feature_name not in self.smoothers:
            self.smoothers[feature_name] = EMASmoother(self.alpha)

        return self.smoothers[feature_name].update(value)
