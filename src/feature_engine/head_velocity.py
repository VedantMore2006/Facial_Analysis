import numpy as np


class HeadYawVelocity:

    def __init__(self):
        self.prev_yaw = None

    def compute_yaw(self, subset):

        nx, ny = subset[1]      # nose
        lx, ly = subset[33]     # left eye outer
        rx, ry = subset[263]    # right eye outer

        dist_left = np.sqrt((nx - lx)**2 + (ny - ly)**2)
        dist_right = np.sqrt((nx - rx)**2 + (ny - ry)**2)

        return dist_left - dist_right

    def compute_velocity(self, subset):

        yaw = self.compute_yaw(subset)

        if self.prev_yaw is None:
            self.prev_yaw = yaw
            return 0

        velocity = abs(yaw - self.prev_yaw)

        self.prev_yaw = yaw

        return velocity