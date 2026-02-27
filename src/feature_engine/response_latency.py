# ============================================================================
# RESPONSE LATENCY DETECTION (VISION-ONLY)
# ============================================================================
# Purpose: Measure time from stimulus to first verbal response
# 
# What is Response Latency?
# - Time delay between question/prompt and person's first response
# - Indicator of processing speed, confidence, engagement
# - Measured in seconds
# 
# Detection Method:
# 1. Experimenter presses 'S' key when question ends (stimulus)
# 2. System watches for sustained mouth opening (speech start)
# 3. Time difference = response latency
# 
# Mouth Opening Detection:
# - Measures distance between upper and lower lip
# - Compares to baseline (normal closed-mouth distance)
# - Response detected when opening exceeds threshold for N frames
# - Multiple frames required to avoid false positives (swallowing, etc.)
# 
# Interpretation:
# - Short latency (~0.1-0.5s): Quick, confident response
# - Medium latency (~0.5-2s): Normal processing time
# - Long latency (>2s): Hesitation, confusion, or uncertainty
# 
# Limitations:
# - Vision-only (doesn't detect audio)
# - Detects mouth opening, not actual speech
# - Requires manual stimulus trigger
# ============================================================================

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
        """
        Initialize response latency detector.
        
        Args:
            fps: Frames per second (from camera)
            consecutive_frames: How many consecutive frames mouth must be open
                               to register as response (default: 3)
                               Higher = fewer false positives, slower detection
        """
        self.fps = fps
        self.required_frames = consecutive_frames

        # Stimulus tracking
        self.stimulus_time = None         # When did stimulus occur?
        self.waiting_for_response = False # Are we currently waiting for response?

        # Response detection state
        self.counter = 0                  # Consecutive frames with mouth open
        self.latencies = []               # List of all detected latencies

        # Baseline statistics (set during baseline phase)
        self.baseline_mouth_mu = None     # Mean mouth opening during baseline
        self.baseline_mouth_sigma = None  # Std dev of mouth opening

    # ----------------------------
    # Mouth Opening Computation
    # ----------------------------

    def compute_mouth_open(self, subset):
        """
        Compute vertical distance between lips (mouth opening).
        
        Measures distance between upper and lower lip center points.
        Larger value = mouth more open.
        
        Args:
            subset: Dictionary of landmarks {index: (x, y)}
        
        Returns:
            Euclidean distance between lip points (float)
        """
        # Upper lip center (landmark 13)
        x1, y1 = subset[13]
        # Lower lip center (landmark 14)
        x2, y2 = subset[14]

        # Calculate vertical opening distance
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # ----------------------------
    # Set Baseline Statistics
    # ----------------------------

    def set_baseline_stats(self, mu, sigma):
        """
        Set baseline mouth opening statistics.
        
        Called after baseline phase completes.
        Used to determine threshold for response detection.
        
        Args:
            mu: Mean mouth opening during baseline (normal closed mouth)
            sigma: Standard deviation of mouth opening
        """
        self.baseline_mouth_mu = mu
        self.baseline_mouth_sigma = sigma

    # ----------------------------
    # Trigger Stimulus Event
    # ----------------------------

    def set_stimulus(self, current_time):
        """
        Mark the stimulus event (question/prompt ends).
        
        Called when user presses 'S' key to indicate stimulus occurred.
        Starts waiting for first response.
        
        Args:
            current_time: Timestamp when stimulus occurred (seconds)
        """
        self.stimulus_time = current_time
        self.waiting_for_response = True
        self.counter = 0
        print("Stimulus registered.")

    # ----------------------------
    # Update Per Frame (Response Detection)
    # ----------------------------

    def update(self, subset, current_time):
        """
        Check for response (mouth opening) each frame.
        
        Only active when waiting for response after stimulus.
        Detects sustained mouth opening as indication of speech start.
        
        Logic:
        1. Check if we're waiting for response
        2. Measure current mouth opening
        3. Compare to baseline threshold (mu + 2*sigma)
        4. Count consecutive frames above threshold
        5. If sustained for N frames, register response and compute latency
        
        Args:
            subset: Dictionary of landmarks {index: (x, y)}
            current_time: Current timestamp (seconds)
        
        Returns:
            Latency value (float) if response detected, None otherwise
        """

        # Not waiting for response, nothing to do
        if not self.waiting_for_response:
            return None

        # Baseline not set yet, can't detect responses
        if self.baseline_mouth_mu is None:
            return None

        # Compute current mouth opening
        mouth_open = self.compute_mouth_open(subset)

        # Threshold: 2 standard deviations above baseline mean
        # This ensures we only detect significant openings (speech)
        threshold = self.baseline_mouth_mu + 2 * self.baseline_mouth_sigma

        # ================================================================
        # RESPONSE DETECTION STATE MACHINE
        # ================================================================
        if mouth_open > threshold:
            # Mouth is open, increment counter
            self.counter += 1
        else:
            # Mouth closed, reset counter
            self.counter = 0

        # Check if mouth has been open for required consecutive frames
        if self.counter >= self.required_frames:
            # Response detected!
            # Compute latency (time from stimulus to now)
            latency = current_time - self.stimulus_time
            self.latencies.append(latency)

            print(f"Response detected. Latency: {latency:.3f} sec")

            # Stop waiting for response (one response per stimulus)
            self.waiting_for_response = False
            self.counter = 0

            return latency

        # No response detected yet
        return None

    # ----------------------------
    # Get Results (Post-Session Analysis)
    # ----------------------------

    def get_latencies(self):
        """
        Get list of all detected latencies.
        
        Returns:
            List of latency values (seconds) for each stimulus-response pair
        """
        return self.latencies

    def get_mean_latency(self):
        """
        Get average latency across all detected responses.
        
        Useful for session-level statistics.
        
        Returns:
            Mean latency (float), or None if no responses detected
        """
        if len(self.latencies) == 0:
            return None
        return np.mean(self.latencies)