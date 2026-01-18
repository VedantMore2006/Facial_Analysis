import numpy as np

class BaselineNormalizer:
    def __init__(self, warmup_frames=30):
        self.warmup_frames = warmup_frames
        self.buffer = []
        self.baseline = None

    def update(self, feature_dict):
        """
        Collect features until baseline is established.
        """
        if self.baseline is not None:
            return

        self.buffer.append(feature_dict)

        if len(self.buffer) >= self.warmup_frames:
            self._compute_baseline()

    def _compute_baseline(self):
        """
        Compute mean baseline per feature.
        """
        self.baseline = {}
        for key in self.buffer[0].keys():
            values = [f[key] for f in self.buffer]
            self.baseline[key] = np.mean(values)

    def normalize(self, feature_dict):
        """
        Normalize features relative to baseline.
        """
        if self.baseline is None:
            return None

        normalized = {}
        for key, value in feature_dict.items():
            base = self.baseline.get(key, 1e-6)
            normalized[key] = value / base

        return normalized
