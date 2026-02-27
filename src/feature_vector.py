"""
feature_vector.py

Construct privacy-safe scaled feature vector.

Responsibilities:
- Accept scaled feature inputs
- Clip to [0,1]
- Round to 4 decimals
- Return ordered vector

Does NOT:
- Compute features
- Access landmarks
"""

import numpy as np


def clip_and_round(value):
    value = np.clip(value, 0.0, 1.0)
    return round(float(value), 4)


def build_feature_vector(
    s_au12,
    s_expressivity,
    s_head_velocity,
    s_eye_contact,
    s_blink_rate,
    s_response_latency
):
    vector = [
        clip_and_round(s_au12),
        clip_and_round(s_expressivity),
        clip_and_round(s_head_velocity),
        clip_and_round(s_eye_contact),
        clip_and_round(s_blink_rate),
        clip_and_round(s_response_latency),
    ]

    return vector
