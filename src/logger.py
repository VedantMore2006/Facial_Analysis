# Logger module
"""
logger.py

Purpose:
Save per-frame landmark subset into CSV.

Responsibilities:
- Create session file
- Append frame data
- Handle header formatting

Does NOT:
- Compute features
- Modify data
"""

import csv
import os
import time
from config import CSVConfig

class LandmarkLogger:
    def __init__(self):
        os.makedirs(CSVConfig.OUTPUT_DIR, exist_ok=True)

        timestamp = int(time.time())
        self.filepath = f"{CSVConfig.OUTPUT_DIR}/session_{timestamp}.csv"

        self.file = open(self.filepath, "w", newline="")
        self.writer = csv.writer(self.file)

        self.header_written = False

    def log(self, frame_index, timestamp_ms, subset_dict):

        if not self.header_written:
            header = ["frame_index", "timestamp_ms"]

            for idx in subset_dict.keys():
                header.append(f"{idx}_x")
                header.append(f"{idx}_y")

            self.writer.writerow(header)
            self.header_written = True

        row = [frame_index, timestamp_ms]

        for idx in subset_dict.keys():
            x, y = subset_dict[idx]
            row.extend([x, y])

        self.writer.writerow(row)

    def close(self):
        self.file.close()