"""
response_latency.py

Vision-only response latency detection.

Detects first sustained mouth opening after a stimulus trigger.

Responsibilities:
- Store stimulus timestamp
- Detect first response
- Compute latency
- Store latency list

Does NOT:
- Apply baseline scaling
- Plot
"""

import numpy as np


class ResponseLatency:

    def __init__(self, fps, consecutive_frames=3):
        self.fps = fps
        self.required_frames = consecutive_frames

        self.stimulus_time = None
        self.waiting_for_response = False

        self.counter = 0
        self.latencies = []

        self.baseline_mouth_mu = None
        self.baseline_mouth_sigma = None

    # ----------------------------
    # Mouth opening computation
    # ----------------------------

    def compute_mouth_open(self, subset):
        x1, y1 = subset[13]  # upper lip
        x2, y2 = subset[14]  # lower lip

        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # ----------------------------
    # Set baseline mouth stats
    # ----------------------------

    def set_baseline_stats(self, mu, sigma):
        self.baseline_mouth_mu = mu
        self.baseline_mouth_sigma = sigma

    # ----------------------------
    # Trigger stimulus
    # ----------------------------

    def set_stimulus(self, current_time):
        self.stimulus_time = current_time
        self.waiting_for_response = True
        self.counter = 0
        print("Stimulus registered.")

    # ----------------------------
    # Update per frame
    # ----------------------------

    def update(self, subset, current_time):

        if not self.waiting_for_response:
            return None

        if self.baseline_mouth_mu is None:
            return None

        mouth_open = self.compute_mouth_open(subset)

        threshold = self.baseline_mouth_mu + 2 * self.baseline_mouth_sigma

        if mouth_open > threshold:
            self.counter += 1
        else:
            self.counter = 0

        if self.counter >= self.required_frames:
            latency = current_time - self.stimulus_time
            self.latencies.append(latency)

            print(f"Response detected. Latency: {latency:.3f} sec")

            self.waiting_for_response = False
            self.counter = 0

            return latency

        return None

    # ----------------------------
    # Get results
    # ----------------------------

    def get_latencies(self):
        return self.latencies

    def get_mean_latency(self):
        if len(self.latencies) == 0:
            return None
        return np.mean(self.latencies)