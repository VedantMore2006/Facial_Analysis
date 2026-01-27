import numpy as np
from collections import deque

class FeatureExtractor:
    def __init__(self, window_size=10):
        self.window_size = window_size

        # rolling history for emotional flags
        self.feature_history = {
            "blink": deque(maxlen=window_size),
            "ear": deque(maxlen=window_size),
            "jaw": deque(maxlen=window_size),
            "au12": deque(maxlen=window_size),
            "au15": deque(maxlen=window_size),
            "au4_vel": deque(maxlen=window_size)
        }

        self.baseline = None

    # -------------------------------------------------
    # Geometry helpers
    # -------------------------------------------------
    def _euclidean(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _get_point(self, landmarks, idx, w, h):
        lm = landmarks[idx]
        return (lm.x * w, lm.y * h)

    # -------------------------------------------------
    # Phase 4.4 FEATURES
    # -------------------------------------------------
    def eye_aspect_ratio(self, landmarks, eye_indices, w, h):
        p = [self._get_point(landmarks, i, w, h) for i in eye_indices]
        v1 = self._euclidean(p[1], p[5])
        v2 = self._euclidean(p[2], p[4])
        hdist = self._euclidean(p[0], p[3])
        return (v1 + v2) / (2.0 * hdist + 1e-6)

    def mouth_opening_ratio(self, landmarks, w, h):
        upper = self._get_point(landmarks, 13, w, h)
        lower = self._get_point(landmarks, 14, w, h)
        left = self._get_point(landmarks, 78, w, h)
        right = self._get_point(landmarks, 308, w, h)
        return self._euclidean(upper, lower) / (self._euclidean(left, right) + 1e-6)

    def jaw_drop(self, landmarks, w, h):
        chin = self._get_point(landmarks, 152, w, h)
        upper = self._get_point(landmarks, 13, w, h)
        return self._euclidean(chin, upper) / h

    # -------------------------------------------------
    # AU PLACEHOLDERS (SAFE, NON-SEMANTIC)
    # -------------------------------------------------
    def action_unit_12(self, landmarks, w, h):
        # lip corner pull proxy
        left = self._get_point(landmarks, 78, w, h)
        right = self._get_point(landmarks, 308, w, h)
        return self._euclidean(left, right) / w

    def action_unit_15(self, landmarks, w, h):
        # lip corner depressor proxy
        return 1.0 - self.action_unit_12(landmarks, w, h)

    def action_unit_4_velocity(self, landmarks, w, h):
        # eyebrow movement proxy (simple, stable)
        brow = self._get_point(landmarks, 65, w, h)
        eye = self._get_point(landmarks, 33, w, h)
        return abs(brow[1] - eye[1]) / h

    # -------------------------------------------------
    # Phase 4.5 EMOTIONAL FLAGS
    # -------------------------------------------------
    def _zscore(self, values):
        mean = np.mean(values)
        std = np.std(values) + 1e-6
        return (values[-1] - mean) / std

    def emotional_flags(self, feature_window):
        self.feature_history["blink"].append(feature_window["blink"])
        self.feature_history["ear"].append(feature_window["ear"])
        self.feature_history["jaw"].append(feature_window["jaw"])
        self.feature_history["au12"].append(feature_window["au12"])
        self.feature_history["au15"].append(feature_window["au15"])
        self.feature_history["au4_vel"].append(feature_window["au4_vel"])

        if len(self.feature_history["blink"]) < self.window_size:
            return None

        blink_z = self._zscore(np.array(self.feature_history["blink"]))
        ear_var = np.var(self.feature_history["ear"])
        jaw_rigidity = 1.0 - np.std(self.feature_history["jaw"])

        stress = min(1.0, 0.5 * blink_z + 0.3 * jaw_rigidity + 0.2 * ear_var)
        flat = 1.0 - np.mean([
            np.mean(self.feature_history["au12"]),
            np.mean(self.feature_history["au15"])
        ])
        arousal = min(1.0, np.mean(self.feature_history["au4_vel"]) * 2.0)

        return {
            "stress_flag": float(np.clip(stress, 0.0, 1.0)),
            "flat_affect_flag": float(np.clip(flat, 0.0, 1.0)),
            "arousal_flag": float(np.clip(arousal, 0.0, 1.0))
        }
