import numpy as np

from features.base_feature import BaseFeature
from processing.geometry import inter_ocular_distance, normalized_distance


class AU12Mean(BaseFeature):
    def __init__(self):
        super().__init__("AU12Mean")

    def compute(self, landmarks, frame_buffer, timestamp):
        iod = inter_ocular_distance(landmarks)

        value = normalized_distance(landmarks[61], landmarks[291], iod)

        self.update_history(value)

        return value


class AU12Variance(BaseFeature):
    def __init__(self):
        super().__init__("AU12Variance")

    def compute(self, landmarks, frame_buffer, timestamp):
        iod = inter_ocular_distance(landmarks)

        value = normalized_distance(landmarks[61], landmarks[291], iod)

        self.update_history(value)

        if len(self.history) < 5:
            return 0

        return float(np.var(self.history))


class AU12ActivationFrequency(BaseFeature):
    def __init__(self, threshold=0.45):
        super().__init__("AU12ActivationFrequency")
        self.threshold = threshold

    def compute(self, landmarks, frame_buffer, timestamp):
        iod = inter_ocular_distance(landmarks)

        value = normalized_distance(landmarks[61], landmarks[291], iod)

        self.update_history(value)

        activations = [v for v in self.history if v > self.threshold]

        # Return rate (proportion) instead of count
        if len(self.history) == 0:
            return 0
        return len(activations) / len(self.history)


class AU15MeanAmplitude(BaseFeature):
    def __init__(self):
        super().__init__("AU15MeanAmplitude")

    def compute(self, landmarks, frame_buffer, timestamp):
        iod = inter_ocular_distance(landmarks)

        left = normalized_distance(landmarks[61], landmarks[95], iod)
        right = normalized_distance(landmarks[291], landmarks[324], iod)

        value = (left + right) / 2

        self.update_history(value)

        return value


class AU4MeanActivation(BaseFeature):
    def __init__(self):
        super().__init__("AU4MeanActivation")

    def compute(self, landmarks, frame_buffer, timestamp):
        iod = inter_ocular_distance(landmarks)

        value = normalized_distance(landmarks[63], landmarks[336], iod)

        self.update_history(value)

        return value


class AU4DurationRatio(BaseFeature):
    def __init__(self, threshold=0.25):
        super().__init__("AU4DurationRatio")
        self.threshold = threshold

    def compute(self, landmarks, frame_buffer, timestamp):
        iod = inter_ocular_distance(landmarks)

        value = normalized_distance(landmarks[63], landmarks[336], iod)

        self.update_history(value)

        if not self.history:
            return 0

        active = [v for v in self.history if v > self.threshold]

        return len(active) / len(self.history)


class AU1AU2PeakIntensity(BaseFeature):
    def __init__(self):
        super().__init__("AU1AU2PeakIntensity")

    def compute(self, landmarks, frame_buffer, timestamp):
        iod = inter_ocular_distance(landmarks)

        value = normalized_distance(landmarks[70], landmarks[296], iod)

        self.update_history(value)

        # Clip to [0, 1] to handle numerical precision issues
        return min(max(self.history), 1.0)


class AU20ActivationRate(BaseFeature):
    def __init__(self, threshold=0.35):
        super().__init__("AU20ActivationRate")
        self.threshold = threshold

    def compute(self, landmarks, frame_buffer, timestamp):
        iod = inter_ocular_distance(landmarks)

        value = normalized_distance(landmarks[78], landmarks[308], iod)

        self.update_history(value)

        activations = [v for v in self.history if v > self.threshold]

        # Return rate (proportion) instead of count
        if len(self.history) == 0:
            return 0
        return len(activations) / len(self.history)


class LipCompressionFrequency(BaseFeature):
    def __init__(self, threshold=0.2):
        super().__init__("LipCompressionFrequency")
        self.threshold = threshold

    def compute(self, landmarks, frame_buffer, timestamp):
        iod = inter_ocular_distance(landmarks)

        value = normalized_distance(landmarks[95], landmarks[324], iod)

        self.update_history(value)

        compressions = [v for v in self.history if v < self.threshold]

        # Return rate (proportion) instead of count
        if len(self.history) == 0:
            return 0
        return len(compressions) / len(self.history)
