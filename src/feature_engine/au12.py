# AU12 Action Unit module
"""
au12.py

Raw AU12 proxy:
Distance between mouth corners (61, 291)

No normalization yet.
No smoothing.
"""

import numpy as np

def compute_au12(subset):
    x1, y1 = subset[61]
    x2, y2 = subset[291]

    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)