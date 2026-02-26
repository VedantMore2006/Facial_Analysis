# /home/vedant/Facial_analysis/config.py

# Configuration file for Facial Analysis
"""
Central configuration file.
All tunable runtime parameters live here.
"""

class CameraConfig:
    DEVICE_ID = 0
    WIDTH = 640
    HEIGHT = 480
    FPS = 15

class BaselineConfig:
    ENABLE_BASELINE = True
    DURATION_SECONDS = 30
    SIGMA_FLOOR = 1e-6

class CSVConfig:
    OUTPUT_DIR = "data/"
    WRITE_PER_FRAME = True

class DebugConfig:
    SHOW_LANDMARKS = True
    SHOW_FPS = True