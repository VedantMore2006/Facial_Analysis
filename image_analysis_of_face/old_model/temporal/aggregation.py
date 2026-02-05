import numpy as np
from collections import deque

class TemporalAggregator:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def update(self, feature_dict):
        self.buffer.append(feature_dict)

        if len(self.buffer) < self.window_size:
            return None

        return self.aggregate()

    def aggregate(self):
        aggregated = {}
        keys = self.buffer[0].keys()

        for key in keys:
            values = np.array([f[key] for f in self.buffer])

            aggregated[key] = {
                "mean": float(np.mean(values)),
                "variance": float(np.var(values)),
                "velocity": float(values[-1] - values[0])
            }

        return aggregated
