import numpy as np

from features.base_feature import BaseFeature


def point_distance(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)

    return np.linalg.norm(p1 - p2)


class MeanHeadVelocity(BaseFeature):
    def __init__(self):
        super().__init__("MeanHeadVelocity")

    def compute(self, landmarks, frame_buffer, timestamp):
        frames = frame_buffer.get_recent_frames(2)

        if len(frames) < 2:
            return 0

        prev = frames[-2]["landmarks"]

        anchor_points = [1, 2, 234, 454, 152]

        velocities = []

        for idx in anchor_points:
            v = point_distance(landmarks[idx], prev[idx])

            velocities.append(v)

        velocity = float(np.mean(velocities))

        self.update_history(velocity)

        return velocity


class HeadVelocityPeak(BaseFeature):
    def __init__(self):
        super().__init__("HeadVelocityPeak")

    def compute(self, landmarks, frame_buffer, timestamp):
        frames = frame_buffer.get_recent_frames(2)

        if len(frames) < 2:
            return 0

        prev = frames[-2]["landmarks"]

        anchor_points = [1, 2, 234, 454, 152]

        velocities = []

        for idx in anchor_points:
            v = point_distance(landmarks[idx], prev[idx])

            velocities.append(v)

        peak = max(velocities)

        self.update_history(peak)

        return peak


class HeadMotionEnergy(BaseFeature):
    def __init__(self):
        super().__init__("HeadMotionEnergy")

    def compute(self, landmarks, frame_buffer, timestamp):
        frames = frame_buffer.get_recent_frames(2)

        if len(frames) < 2:
            return 0

        prev = frames[-2]["landmarks"]

        anchor_points = [1, 2, 234, 454, 152]

        energy = 0

        for idx in anchor_points:
            energy += point_distance(landmarks[idx], prev[idx])

        self.update_history(energy)

        return energy


class LandmarkDisplacementMean(BaseFeature):
    def __init__(self):
        super().__init__("LandmarkDisplacementMean")

    def compute(self, landmarks, frame_buffer, timestamp):
        frames = frame_buffer.get_recent_frames(2)

        if len(frames) < 2:
            return 0

        prev = frames[-2]["landmarks"]

        displacements = []

        for idx in landmarks.keys():
            d = point_distance(landmarks[idx], prev[idx])

            displacements.append(d)

        value = float(np.mean(displacements))

        self.update_history(value)

        return value


class PostureRigidityIndex(BaseFeature):
    def __init__(self):
        super().__init__("PostureRigidityIndex")

    def compute(self, landmarks, frame_buffer, timestamp):
        frames = frame_buffer.get_recent_frames(2)

        if len(frames) < 2:
            return 0

        prev = frames[-2]["landmarks"]

        anchor_points = [1, 2, 234, 454, 152]

        motions = []

        for idx in anchor_points:
            m = point_distance(landmarks[idx], prev[idx])

            motions.append(m)

        variance = np.var(motions)

        # Use bounded function instead of 1/variance to avoid explosion
        # This gives rigidity in [0, 1] where 1 = high rigidity (low variance)
        rigidity = 1 / (1 + variance)

        self.update_history(rigidity)

        return rigidity
