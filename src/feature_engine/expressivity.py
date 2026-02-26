# Expressivity module
"""
expressivity.py

Compute overall facial movement amplitude per frame.
Normalized by IOD.
"""

import numpy as np


EXPRESSIVE_POINTS = [
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
    61, 291, 13, 14,
    50, 280
]


def compute_expressivity(subset, baseline_positions):

    displacements = []

    for idx in EXPRESSIVE_POINTS:
        x, y = subset[idx]
        bx, by = baseline_positions[idx]

        dx = x - bx
        dy = y - by

        displacements.append(np.sqrt(dx*dx + dy*dy))

    return np.mean(displacements)