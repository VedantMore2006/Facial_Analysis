import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import numpy as np

class FeatureExtractor:
    def __init__(self):
        pass

    # -----------------------------
    # Utility functions
    # -----------------------------
    def _euclidean(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _get_point(self, landmarks, idx, w, h):
        lm = landmarks[idx]
        return (lm.x * w, lm.y * h)

    # -----------------------------
    # Eye Aspect Ratio (EAR)
    # -----------------------------
    def eye_aspect_ratio(self, landmarks, eye_indices, w, h):
        p = [self._get_point(landmarks, i, w, h) for i in eye_indices]

        vertical_1 = self._euclidean(p[1], p[5])
        vertical_2 = self._euclidean(p[2], p[4])
        horizontal = self._euclidean(p[0], p[3])

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    # -----------------------------
    # Mouth Opening Ratio (MOR)
    # -----------------------------
    def mouth_opening_ratio(self, landmarks, w, h):
        upper = self._get_point(landmarks, 13, w, h)
        lower = self._get_point(landmarks, 14, w, h)
        left = self._get_point(landmarks, 78, w, h)
        right = self._get_point(landmarks, 308, w, h)

        vertical = self._euclidean(upper, lower)
        horizontal = self._euclidean(left, right)

        mor = vertical / horizontal
        return mor

    # -----------------------------
    # Eyebrow Displacement
    # -----------------------------
    def eyebrow_displacement(self, landmarks, brow_indices, eye_center_idx, w, h):
        brow_points = [self._get_point(landmarks, i, w, h) for i in brow_indices]
        brow_y = np.mean([p[1] for p in brow_points])

        eye_center = self._get_point(landmarks, eye_center_idx, w, h)

        displacement = (eye_center[1] - brow_y) / h
        return displacement

    # -----------------------------
    # Jaw Drop Ratio
    # -----------------------------
    def jaw_drop(self, landmarks, w, h):
        chin = self._get_point(landmarks, 152, w, h)
        upper_lip = self._get_point(landmarks, 13, w, h)

        jaw_drop = self._euclidean(chin, upper_lip) / h
        return jaw_drop
