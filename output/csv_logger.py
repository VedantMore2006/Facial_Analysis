"""
CSV Logger for Behavioral Features

Logs both raw and scaled features to separate CSV files.
Filename format: HH_MM_DD_MM.csv (hour_minute_day_month)
"""

import os
import csv
from datetime import datetime


def generate_filename():
    """
    Generate timestamped filename for CSV logging.
    
    Format: HH_MM_DD_MM.csv
    Example: 21_30_05_03.csv (21:30 on March 5th)
    
    Returns:
        Filename string
    """

    now = datetime.now()

    return now.strftime("%H_%M_%d_%m") + ".csv"


class CSVLogger:
    """
    Dual CSV logger for raw and scaled features.
    
    Creates two CSV files per session:
    - output/raw/HH_MM_DD_MM.csv     : Raw feature values
    - output/scaled/HH_MM_DD_MM.csv  : Scaled features (prefixed with S_)
    
    Each file contains: timestamp + 34 features = 35 columns
    """

    def __init__(self, feature_names):
        """
        Initialize CSV logger with feature names.
        
        Args:
            feature_names: List of feature names (34 features)
        """

        self.feature_names = feature_names

        os.makedirs("output/scaled", exist_ok=True)
        os.makedirs("output/raw", exist_ok=True)

        filename = generate_filename()

        self.scaled_path = os.path.join("output/scaled", filename)
        self.raw_path = os.path.join("output/raw", filename)

        self.initialize_files()

    def initialize_files(self):
        """
        Create CSV files with headers.
        
        Scaled header: timestamp, S_AU12Mean, S_BlinkRate, ...
        Raw header:    timestamp, AU12Mean, BlinkRate, ...
        """

        scaled_header = ["timestamp"] + [
            f"S_{name}" for name in self.feature_names
        ]

        raw_header = ["timestamp"] + self.feature_names

        with open(self.scaled_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(scaled_header)

        with open(self.raw_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(raw_header)

    def log(self, timestamp, raw_features, scaled_features):
        """
        Log feature values to both CSV files.
        
        Args:
            timestamp: Current timestamp
            raw_features: Dict of feature name -> raw value
            scaled_features: Dict of feature name -> scaled value [0, 1]
        """

        raw_row = [timestamp] + [
            raw_features.get(name, 0) for name in self.feature_names
        ]

        scaled_row = [timestamp] + [
            scaled_features.get(name, 0) for name in self.feature_names
        ]

        try:
            with open(self.raw_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(raw_row)

            with open(self.scaled_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(scaled_row)
        except Exception as e:
            print(f"ERROR writing to CSV: {e}")
