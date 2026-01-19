import numpy as np
from collections import deque

class FeatureExtractor:
    def __init__(self, window_size=10):
        self.window_size = window_size

        # rolling history of features
        self.feature_history = {
            "blink": deque(maxlen=window_size),
            "ear": deque(maxlen=window_size),
            "jaw": deque(maxlen=window_size),
            "au12": deque(maxlen=window_size),
            "au15": deque(maxlen=window_size),
            "au4_vel": deque(maxlen=window_size)
        }

        # baseline (first window)
        self.baseline = None

    # -------------------------------------------------
    # Utility
    # -------------------------------------------------
    def _zscore(self, values):
        mean = np.mean(values)
        std = np.std(values) + 1e-6
        return (values[-1] - mean) / std

    # -------------------------------------------------
    # Emotional Flags (Phase 4.5)
    # -------------------------------------------------
    def emotional_flags(self, feature_window):
        """
        feature_window: dict with latest features
        Returns: dict with stress, flat affect, arousal flags
        """

        # Update rolling buffers
        self.feature_history["blink"].append(feature_window["blink"])
        self.feature_history["ear"].append(feature_window["ear"])
        self.feature_history["jaw"].append(feature_window["jaw"])
        self.feature_history["au12"].append(feature_window["au12"])
        self.feature_history["au15"].append(feature_window["au15"])
        self.feature_history["au4_vel"].append(feature_window["au4_vel"])

        # Wait until window fills
        if len(self.feature_history["blink"]) < self.window_size:
            return None

        # Establish baseline once
        if self.baseline is None:
            self.baseline = {
                k: np.mean(v) for k, v in self.feature_history.items()
            }

        # ---------------------------
        # Z-score features
        # ---------------------------
        blink_z = self._zscore(np.array(self.feature_history["blink"]))
        ear_var = np.var(self.feature_history["ear"])
        jaw_rigidity = 1.0 - np.std(self.feature_history["jaw"])

        # ---------------------------
        # Flags (LOCKED FORMULAS)
        # ---------------------------

        # Stress proxy
        stress_flag = min(
            1.0,
            0.5 * blink_z +
            0.3 * jaw_rigidity +
            0.2 * ear_var
        )

        # Flat affect proxy
        au12_mean = np.mean(self.feature_history["au12"])
        au15_mean = np.mean(self.feature_history["au15"])
        flat_affect_flag = 1.0 - np.mean([au12_mean, au15_mean])

        # Arousal proxy
        arousal_flag = min(
            1.0,
            np.mean(self.feature_history["au4_vel"]) * 2.0
        )

        return {
            "stress_flag": float(np.clip(stress_flag, 0.0, 1.0)),
            "flat_affect_flag": float(np.clip(flat_affect_flag, 0.0, 1.0)),
            "arousal_flag": float(np.clip(arousal_flag, 0.0, 1.0))
        }
