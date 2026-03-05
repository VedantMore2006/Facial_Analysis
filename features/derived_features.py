import numpy as np

from features.base_feature import BaseFeature
from processing.geometry import inter_ocular_distance, normalized_distance


class OverallAUVariance(BaseFeature):
    def __init__(self):
        super().__init__("OverallAUVariance")

    def compute(self, landmarks, frame_buffer, timestamp):
        if frame_buffer.size() == 0:
            return 0

        frames = frame_buffer.get_all_frames()

        values = []

        for frame in frames:
            f = frame["features"]

            if "AU12Mean" in f:
                values.append(f["AU12Mean"])

        if len(values) < 2:
            return 0

        return float(np.var(values))


class FacialEmotionalRange(BaseFeature):
    def __init__(self):
        super().__init__("FacialEmotionalRange")

    def compute(self, landmarks, frame_buffer, timestamp):
        frames = frame_buffer.get_all_frames()

        values = []

        for frame in frames:
            f = frame["features"]

            if "AU12Mean" in f:
                values.append(f["AU12Mean"])

        if not values:
            return 0

        return float(max(values) - min(values))


class FacialTransitionFrequency(BaseFeature):
    def __init__(self, threshold=0.05):
        super().__init__("FacialTransitionFrequency")
        self.threshold = threshold

    def compute(self, landmarks, frame_buffer, timestamp):
        frames = frame_buffer.get_all_frames()

        values = []

        for frame in frames:
            f = frame["features"]

            if "AU12Mean" in f:
                values.append(f["AU12Mean"])

        if len(values) < 2:
            return 0

        transitions = 0

        for i in range(1, len(values)):
            if abs(values[i] - values[i - 1]) > self.threshold:
                transitions += 1

        # Return rate (proportion) instead of count
        return transitions / max(len(values) - 1, 1)


class NearZeroAUActivationRatio(BaseFeature):
    def __init__(self, threshold=0.1):
        super().__init__("NearZeroAUActivationRatio")
        self.threshold = threshold

    def compute(self, landmarks, frame_buffer, timestamp):
        frames = frame_buffer.get_all_frames()

        values = []

        for frame in frames:
            f = frame["features"]

            if "AU12Mean" in f:
                values.append(f["AU12Mean"])

        if not values:
            return 0

        near_zero = [v for v in values if v < self.threshold]

        return len(near_zero) / len(values)


class MotionEnergyFloorScore(BaseFeature):
    def __init__(self, threshold=0.01):
        super().__init__("MotionEnergyFloorScore")
        self.threshold = threshold

    def compute(self, landmarks, frame_buffer, timestamp):
        frames = frame_buffer.get_all_frames()

        motion_values = []

        for frame in frames:
            f = frame["features"]

            if "HeadMotionEnergy" in f:
                motion_values.append(f["HeadMotionEnergy"])

        if not motion_values:
            return 0

        low_motion = [v for v in motion_values if v < self.threshold]

        return len(low_motion) / len(motion_values)


class GestureFrequency(BaseFeature):
    def __init__(self, threshold=0.05):
        super().__init__("GestureFrequency")
        self.threshold = threshold

    def compute(self, landmarks, frame_buffer, timestamp):
        frames = frame_buffer.get_all_frames()

        motion_values = []

        for frame in frames:
            f = frame["features"]

            if "HeadVelocityPeak" in f:
                motion_values.append(f["HeadVelocityPeak"])

        gestures = [v for v in motion_values if v > self.threshold]

        # Return rate (proportion) instead of count
        if len(motion_values) == 0:
            return 0
        return len(gestures) / len(motion_values)


class MicroMotionEnergy(BaseFeature):
    def __init__(self):
        super().__init__("MicroMotionEnergy")

    def compute(self, landmarks, frame_buffer, timestamp):
        frames = frame_buffer.get_all_frames()

        displacements = []

        for frame in frames:
            f = frame["features"]

            if "LandmarkDisplacementMean" in f:
                displacements.append(f["LandmarkDisplacementMean"])

        if not displacements:
            return 0

        return float(np.mean(displacements))


class ShoulderElevationIndex(BaseFeature):
    def __init__(self):
        super().__init__("ShoulderElevationIndex")

    def compute(self, landmarks, frame_buffer, timestamp):
        # Requires body pose landmarks (not implemented yet)

        return 0
