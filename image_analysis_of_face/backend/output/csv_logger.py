import csv
import time
import os

class CSVLogger:
    def __init__(self, filename="facial_signals.csv", rate_hz=5):
        self.filename = filename
        self.interval = 1.0 / rate_hz
        self.last_write = 0.0

        # Create file + header if not exists
        if not os.path.exists(self.filename):
            with open(self.filename, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "ear_norm",
                    "mouth_norm",
                    "jaw_norm",
                    "stress_flag",
                    "flat_affect_flag",
                    "arousal_flag"
                ])

    def log(self, data):
        now = time.time()
        if now - self.last_write < self.interval:
            return

        self.last_write = now

        with open(self.filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                round(now, 2),
                round(data["ear"], 3),
                round(data["mouth"], 3),
                round(data["jaw"], 3),
                round(data["stress"], 3),
                round(data["flat"], 3),
                round(data["arousal"], 3)
            ])
