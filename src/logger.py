# ============================================================================
# LANDMARK LOGGER (DEBUG/RESEARCH ONLY)
# ============================================================================
# Purpose: Save raw landmark coordinates to CSV for debugging/research
# 
# PRIVACY WARNING:
# - This stores RAW landmark coordinates
# - Landmarks can be used to reconstruct face appearance
# - Use only for debugging, NOT for production/deployment
# - Consider using FeatureLogger instead for privacy-safe logging
# 
# Output Format:
# - CSV with columns: frame_index, timestamp_ms, then x,y for each landmark
# - One row per frame
# - All coordinates are normalized [0,1]
# 
# Use Cases:
# - Debugging landmark detection issues
# - Developing new features
# - Research analysis
# ============================================================================

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
        """
        Initialize landmark logger with timestamped CSV file.
        
        Creates a new session file with timestamp in filename.
        Header is written dynamically based on first frame's landmarks.
        """
        # Create output directory if it doesn't exist
        os.makedirs(CSVConfig.OUTPUT_DIR, exist_ok=True)

        # Generate unique filename with timestamp
        timestamp = int(time.time())
        self.filepath = f"{CSVConfig.OUTPUT_DIR}/session_{timestamp}.csv"

        # Open CSV file for writing
        self.file = open(self.filepath, "w", newline="")
        self.writer = csv.writer(self.file)

        # Header is written on first log call (when we know landmark indices)
        self.header_written = False

    def log(self, frame_index, timestamp_ms, subset_dict):
        """
        Log landmark coordinates for a single frame.
        
        Writes header row on first call, then writes data rows.
        Each row contains frame index, timestamp, and x,y for each landmark.
        
        Args:
            frame_index: Sequential frame number (0, 1, 2, ...)
            timestamp_ms: Timestamp in milliseconds
            subset_dict: Dictionary {landmark_idx: (x, y)}
        """

        # Write header row on first call
        if not self.header_written:
            header = ["frame_index", "timestamp_ms"]

            # Add column for each landmark (idx_x, idx_y)
            for idx in subset_dict.keys():
                header.append(f"{idx}_x")
                header.append(f"{idx}_y")

            self.writer.writerow(header)
            self.header_written = True

        # Build data row: [frame_index, timestamp, x1, y1, x2, y2, ...]
        row = [frame_index, timestamp_ms]

        # Append x,y coordinates for each landmark
        for idx in subset_dict.keys():
            x, y = subset_dict[idx]
            row.extend([x, y])

        # Write row to CSV
        self.writer.writerow(row)

    def close(self):
        """
        Close CSV file and flush data to disk.
        
        IMPORTANT: Always call this at session end.
        """
        self.file.close()