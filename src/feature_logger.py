"""
feature_logger.py

Purpose:
Log ONLY privacy-safe scaled features.

Does NOT:
- Store raw landmarks
- Store coordinates
- Store identifiable data
"""

import csv
import os
import time


class FeatureLogger:
    def __init__(self, output_dir="data/"):
        os.makedirs(output_dir, exist_ok=True)

        timestamp = int(time.time())
        self.filepath = f"{output_dir}/features_{timestamp}.csv"

        self.file = open(self.filepath, "w", newline="")
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "S_AU12",
            "S_AUVar",
            "S_HeadVelocity",
            "S_EyeContact",
            "S_BlinkRate",
            "S_ResponseLatency"
        ])

    def log(self, feature_vector):
        self.writer.writerow(feature_vector)

    def close(self):
        self.file.close()
