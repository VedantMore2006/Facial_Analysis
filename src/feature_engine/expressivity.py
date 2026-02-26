import numpy as np

EXPRESSIVE_POINTS = [
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
    61, 291, 13, 14,
    50, 280
]

class Expressivity:

    def __init__(self):
        self.prev_subset = None

    def compute(self, subset):

        if self.prev_subset is None:
            self.prev_subset = subset
            return 0

        displacements = []

        for idx in EXPRESSIVE_POINTS:
            x, y = subset[idx]
            px, py = self.prev_subset[idx]

            dx = x - px
            dy = y - py

            displacements.append(np.sqrt(dx*dx + dy*dy))

        self.prev_subset = subset

        return np.mean(displacements)