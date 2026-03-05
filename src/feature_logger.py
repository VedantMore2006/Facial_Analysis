# ============================================================================
# PRIVACY-SAFE FEATURE LOGGER
# ============================================================================
# Purpose: Log ONLY scaled features (no raw landmarks or identifiable data)
# 
# Privacy Design:
# - Stores ONLY normalized feature values (0-1 range)
# - NO raw landmark coordinates
# - NO facial geometry data
# - NO identifiable information
# 
# Why This Matters:
# - Raw landmarks can reconstruct face appearance
# - Scaled features are behavioral metrics without biometric info
# - Enables analysis while protecting privacy
# 
# Output Format:
# - CSV with columns: S_AU12, S_AUVar, S_HeadVelocity, etc.
# - One row per frame
# - All values clipped to [0,1] range
# ============================================================================

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
from datetime import datetime


class FeatureLogger:
    def __init__(self, output_dir="data/"):
        """
        Initialize feature logger with timestamped output file.
        
        Creates a new CSV file for this session with timestamp in filename.
        Writes header row with feature names.
        
        Args:
            output_dir: Directory where CSV files will be saved
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate unique filename with human-readable timestamp
        # Format: features_HH-MM_DD-MM.csv (e.g., features_14-30_04-03.csv)
        now = datetime.now()
        timestamp_str = now.strftime("%H-%M_%d-%m")
        self.filepath = f"{output_dir}/features_{timestamp_str}.csv"

        # Open CSV file for writing
        self.file = open(self.filepath, "w", newline="")
        self.writer = csv.writer(self.file)

        # Write header row defining all feature columns
        # S_ prefix means "Scaled" - these are normalized values
        self.writer.writerow([
            "S_AU12",              # Scaled smile intensity
            "S_AUVar",             # Scaled facial expressivity/variation
            "S_HeadVelocity",      # Scaled head movement speed
            "S_EyeContact",        # Scaled eye contact ratio
            "S_BlinkRate",         # Scaled blink frequency
            "S_ResponseLatency"    # Scaled response time
        ])

    def log(self, feature_vector):
        """
        Write a single frame's feature vector to CSV.
        
        Args:
            feature_vector: List of 6 scaled feature values [0-1]
        """
        self.writer.writerow(feature_vector)

    def close(self):
        """
        Close the CSV file and flush all data to disk.
        
        IMPORTANT: Always call this at the end of the session.
        """
        self.file.close()


# ============================================================================
# VALIDATION RAW LOGGER
# ============================================================================
# Purpose: Log RAW, UNPROCESSED values for validation analysis
# 
# Design:
# - Logs raw feature values WITHOUT smoothing
# - Logs raw values WITHOUT baseline correction
# - Logs raw values WITHOUT scaling
# - Captures frame index and timestamp for temporal alignment
# 
# This artifact enables:
# - Validation of feature computation correctness
# - Analysis of raw signal quality
# - Transparency in data processing
# 
# Output Format:
# - CSV with columns: frame_index, timestamp_ms, au12_raw, expressivity_raw, etc.
# - One row per frame
# - All values are raw, unprocessed, unscaled
# ============================================================================

class ValidationRawLogger:
    def __init__(self, output_dir="data/"):
        """
        Initialize validation raw logger with timestamped output file.
        
        Creates a new CSV file for this session with timestamp in filename.
        Writes header row with raw feature names.
        
        Args:
            output_dir: Directory where CSV files will be saved
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate unique filename with human-readable timestamp
        # Format: validation_raw_session_HH-MM_DD-MM.csv
        now = datetime.now()
        timestamp_str = now.strftime("%H-%M_%d-%m")
        self.filepath = f"{output_dir}/validation_raw_session_{timestamp_str}.csv"

        # Open CSV file for writing
        self.file = open(self.filepath, "w", newline="")
        self.writer = csv.writer(self.file)

        # Write header row defining all raw feature columns
        self.writer.writerow([
            "frame_index",           # Frame number in session
            "timestamp_ms",          # Milliseconds since session start
            "au12_raw",              # Raw smile intensity (unscaled, unsmoothed)
            "expressivity_raw",      # Raw facial expressivity/variation (unscaled, unsmoothed)
            "head_velocity_raw",     # Raw head movement speed (unscaled, unsmoothed)
            "blink_rate_raw",        # Raw blink frequency (unscaled, unsmoothed)
            "ear_raw",               # Raw Eye Aspect Ratio (unscaled, unsmoothed)
            "yaw_raw"                # Raw head yaw angle (unscaled, unsmoothed)
        ])

    def log(self, frame_index, timestamp_ms, au12_raw, expressivity_raw, 
            head_velocity_raw, blink_rate_raw, ear_raw, yaw_raw):
        """
        Write a single frame's raw feature values to CSV.
        
        Args:
            frame_index: Frame number in the session
            timestamp_ms: Milliseconds since session start
            au12_raw: Raw smile intensity value
            expressivity_raw: Raw facial expressivity value
            head_velocity_raw: Raw head velocity value
            blink_rate_raw: Raw blink rate value
            ear_raw: Raw Eye Aspect Ratio value
            yaw_raw: Raw head yaw angle value
        """
        self.writer.writerow([
            frame_index,
            timestamp_ms,
            au12_raw,
            expressivity_raw,
            head_velocity_raw,
            blink_rate_raw,
            ear_raw,
            yaw_raw
        ])

    def close(self):
        """
        Close the CSV file and flush all data to disk.
        
        IMPORTANT: Always call this at the end of the session.
        """
        self.file.close()
