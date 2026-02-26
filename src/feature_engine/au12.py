"""
au12.py

Smile width normalized by Interocular Distance (IOD)
"""

import numpy as np


def compute_iod(subset):
    # Left eye outer (33) & right eye outer (263)
    x1, y1 = subset[33]
    x2, y2 = subset[263]

    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def compute_au12(subset):
    x1, y1 = subset[61]
    x2, y2 = subset[291]

    smile_width = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    iod = compute_iod(subset)

    if iod == 0:
        return 0

    return smile_width / iod