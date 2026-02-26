# Landmark processing module
"""
landmark_processor.py

Purpose:
Extract selected landmark subset from full 478 mesh.

Responsibilities:
- Receive MediaPipe landmarks
- Extract required indices
- Return normalized x,y coordinates

Does NOT:
- Compute features
- Normalize by IOD
- Apply smoothing
"""

LANDMARK_SUBSET = sorted(set([
    61, 291, 13, 14,
    33, 133, 362, 263,
    160, 158, 153, 144,
    385, 387, 373, 380,
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
    50, 280,
    1, 152,
    468, 469, 470, 471, 472,
    473, 474, 475, 476, 477
]))

def extract_subset(landmarks):
    """
    Returns dict:
    {
        index: (x, y)
    }
    Coordinates are normalized [0,1]
    """

    subset = {}

    for idx in LANDMARK_SUBSET:
        lm = landmarks.landmark[idx]
        subset[idx] = (lm.x, lm.y)

    return subset