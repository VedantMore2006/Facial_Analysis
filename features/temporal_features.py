import numpy as np

from features.base_feature import BaseFeature


class ResponseLatencyMean(BaseFeature):
    def __init__(self, max_latency=5.0):
        super().__init__("ResponseLatencyMean")
        self.response_times = []
        self.max_latency = max_latency  # Expected max latency in seconds

    def compute(self, landmarks, frame_buffer, timestamp):
        frames = frame_buffer.get_all_frames()

        motion_values = []

        for frame in frames:
            f = frame["features"]

            if "HeadVelocityPeak" in f:
                motion_values.append((frame["timestamp"], f["HeadVelocityPeak"]))

        if not motion_values:
            return 0

        threshold = 0.04

        for t, v in motion_values:
            if v > threshold:
                latency = t - frames[0]["timestamp"]
                self.response_times.append(latency)
                break

        if not self.response_times:
            return 0

        # Normalize to [0, 1] by dividing by max expected latency
        mean_latency = float(np.mean(self.response_times))
        return min(mean_latency / self.max_latency, 1.0)


class SpeechOnsetDelay(BaseFeature):
    def __init__(self, max_delay=5.0):
        super().__init__("SpeechOnsetDelay")
        self.max_delay = max_delay  # Expected max delay in seconds

    def compute(self, landmarks, frame_buffer, timestamp):
        frames = frame_buffer.get_all_frames()

        mouth_movements = []

        for frame in frames:
            lm = frame["landmarks"]

            distance = abs(lm[13][1] - lm[14][1])

            mouth_movements.append((frame["timestamp"], distance))

        if not mouth_movements:
            return 0

        threshold = 0.01

        for t, d in mouth_movements:
            if d > threshold:
                delay = t - frames[0]["timestamp"]
                # Normalize to [0, 1] by dividing by max expected delay
                return min(delay / self.max_delay, 1.0)

        return 0


class NodOnsetLatency(BaseFeature):
    def __init__(self, max_latency=5.0):
        super().__init__("NodOnsetLatency")
        self.max_latency = max_latency  # Expected max latency in seconds

    def compute(self, landmarks, frame_buffer, timestamp):
        frames = frame_buffer.get_all_frames()

        head_movements = []

        for frame in frames:
            f = frame["features"]

            if "MeanHeadVelocity" in f:
                head_movements.append((frame["timestamp"], f["MeanHeadVelocity"]))

        threshold = 0.03

        for t, v in head_movements:
            if v > threshold:
                latency = t - frames[0]["timestamp"]
                # Normalize to [0, 1] by dividing by max expected latency
                return min(latency / self.max_latency, 1.0)

        return 0


class PauseDurationMean(BaseFeature):
    def __init__(self, max_pause_frames=120):
        super().__init__("PauseDurationMean")
        self.max_pause_frames = max_pause_frames  # ~4 seconds at 30fps

    def compute(self, landmarks, frame_buffer, timestamp):
        frames = frame_buffer.get_all_frames()

        pauses = []

        current_pause = 0

        threshold = 0.01

        for frame in frames:
            f = frame["features"]

            if "HeadMotionEnergy" in f and f["HeadMotionEnergy"] < threshold:
                current_pause += 1
            else:
                if current_pause > 0:
                    pauses.append(current_pause)
                    current_pause = 0

        if not pauses:
            return 0

        # Normalize to [0, 1] by dividing by max expected pause length
        mean_pause = float(np.mean(pauses))
        return min(mean_pause / self.max_pause_frames, 1.0)


class ExtendedSilenceRatio(BaseFeature):
    def __init__(self):
        super().__init__("ExtendedSilenceRatio")

    def compute(self, landmarks, frame_buffer, timestamp):
        frames = frame_buffer.get_all_frames()

        silence_frames = 0

        threshold = 0.01

        for frame in frames:
            f = frame["features"]

            if "HeadMotionEnergy" in f and f["HeadMotionEnergy"] < threshold:
                silence_frames += 1

        if not frames:
            return 0

        return silence_frames / len(frames)


class ReactionTimeInstabilityIndex(BaseFeature):
    def __init__(self):
        super().__init__("ReactionTimeInstabilityIndex")
        self.latencies = []

    def compute(self, landmarks, frame_buffer, timestamp):
        frames = frame_buffer.get_all_frames()

        if len(frames) < 2:
            return 0

        latency = frames[-1]["timestamp"] - frames[-2]["timestamp"]

        self.latencies.append(latency)

        if len(self.latencies) < 2:
            return 0

        return float(np.var(self.latencies))
