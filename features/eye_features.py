import numpy as np

from features.base_feature import BaseFeature
from processing.geometry import inter_ocular_distance, normalized_distance


class BlinkRate(BaseFeature):
    def __init__(self, threshold=0.18):
        super().__init__("BlinkRate")
        self.threshold = threshold

    def compute(self, landmarks, frame_buffer, timestamp):
        iod = inter_ocular_distance(landmarks)

        left_eye = normalized_distance(landmarks[159], landmarks[145], iod)

        right_eye = normalized_distance(landmarks[386], landmarks[374], iod)

        openness = (left_eye + right_eye) / 2

        self.update_history(openness)

        if len(self.history) < 2:
            return 0

        blinks = 0

        for i in range(1, len(self.history)):
            if self.history[i - 1] > self.threshold and self.history[i] < self.threshold:
                blinks += 1

        return blinks


class BlinkClusterDensity(BaseFeature):
    def __init__(self, threshold=0.18):
        super().__init__("BlinkClusterDensity")
        self.threshold = threshold

    def compute(self, landmarks, frame_buffer, timestamp):
        iod = inter_ocular_distance(landmarks)

        left_eye = normalized_distance(landmarks[159], landmarks[145], iod)
        right_eye = normalized_distance(landmarks[386], landmarks[374], iod)

        openness = (left_eye + right_eye) / 2

        self.update_history(openness)

        if len(self.history) < 10:
            return 0

        blink_events = []

        for i in range(1, len(self.history)):
            if self.history[i - 1] > self.threshold and self.history[i] < self.threshold:
                blink_events.append(i)

        clusters = 0

        for i in range(1, len(blink_events)):
            if blink_events[i] - blink_events[i - 1] < 5:
                clusters += 1

        return clusters


class BaselineEyeOpenness(BaseFeature):
    def __init__(self):
        super().__init__("BaselineEyeOpenness")

    def compute(self, landmarks, frame_buffer, timestamp):
        iod = inter_ocular_distance(landmarks)

        left_eye = normalized_distance(landmarks[159], landmarks[145], iod)

        right_eye = normalized_distance(landmarks[386], landmarks[374], iod)

        openness = (left_eye + right_eye) / 2

        self.update_history(openness)

        return float(np.mean(self.history))


class GazeShiftFrequency(BaseFeature):
    def __init__(self):
        super().__init__("GazeShiftFrequency")

    def compute(self, landmarks, frame_buffer, timestamp):
        gaze_x = (landmarks[468][0] + landmarks[472][0]) / 2
        gaze_y = (landmarks[468][1] + landmarks[472][1]) / 2

        gaze = (gaze_x, gaze_y)

        self.update_history(gaze)

        if len(self.history) < 2:
            return 0

        shifts = 0

        for i in range(1, len(self.history)):
            dx = abs(self.history[i][0] - self.history[i - 1][0])
            dy = abs(self.history[i][1] - self.history[i - 1][1])

            if dx + dy > 0.02:
                shifts += 1

        # Return rate (proportion) instead of count
        # Divide by (len-1) since we're counting transitions between frames
        return shifts / max(len(self.history) - 1, 1)


class EyeContactRatio(BaseFeature):
    def __init__(self, center_threshold=0.05):
        super().__init__("EyeContactRatio")
        self.center_threshold = center_threshold

    def compute(self, landmarks, frame_buffer, timestamp):
        gaze_x = (landmarks[468][0] + landmarks[472][0]) / 2

        centered = abs(gaze_x - 0.5) < self.center_threshold

        self.update_history(1 if centered else 0)

        if not self.history:
            return 0

        return sum(self.history) / len(self.history)


class DownwardGazeFrequency(BaseFeature):
    def __init__(self, threshold=0.55):
        super().__init__("DownwardGazeFrequency")
        self.threshold = threshold

    def compute(self, landmarks, frame_buffer, timestamp):
        gaze_y = (landmarks[468][1] + landmarks[472][1]) / 2

        downward = gaze_y > self.threshold

        self.update_history(1 if downward else 0)

        # Return rate (proportion) instead of count
        if len(self.history) == 0:
            return 0
        return sum(self.history) / len(self.history)
