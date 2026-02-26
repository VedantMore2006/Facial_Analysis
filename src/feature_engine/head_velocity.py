# Head velocity module
"""
head_velocity.py

Compute head movement speed using nose tip displacement.
"""

import numpy as np


class HeadVelocity:
    def __init__(self):
        self.prev_position = None

    def compute(self, subset):

        x, y = subset[1]  # Nose tip

        if self.prev_position is None:
            self.prev_position = (x, y)
            return 0

        px, py = self.prev_position

        velocity = np.sqrt((x-px)**2 + (y-py)**2)

        self.prev_position = (x, y)

        return velocity