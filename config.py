# ============================================================================
# CENTRAL CONFIGURATION FILE
# ============================================================================
# Purpose: Single source of truth for all runtime parameters
# All tunable settings are organized by functional area
# Modify these values to change system behavior without editing core code
# ============================================================================

"""
Central configuration file.
All tunable runtime parameters live here.
"""

# ============================================================================
# Camera Configuration
# ============================================================================
# Controls webcam initialization parameters
class CameraConfig:
    DEVICE_ID = 0      # Webcam device index (0=default camera)
    WIDTH = 640        # Frame width in pixels
    HEIGHT = 480       # Frame height in pixels
    FPS = 15           # Target frames per second

# ============================================================================
# Baseline Configuration
# ============================================================================
# Controls personal session baseline (PSB) collection and processing
class BaselineConfig:
    ENABLE_BASELINE = True       # Whether to collect baseline before deviation phase
    DURATION_SECONDS = 30        # How long to collect baseline data (in seconds)
    SIGMA_FLOOR = 1e-6          # Minimum standard deviation to prevent division by zero

# ============================================================================
# CSV Output Configuration
# ============================================================================
# Controls data logging behavior
class CSVConfig:
    OUTPUT_DIR = "data/"         # Directory where CSV files are saved
    WRITE_PER_FRAME = True       # Whether to write each frame immediately (vs buffering)

# ============================================================================
# Debug/Visualization Configuration
# ============================================================================
# Controls visual debugging features during runtime
class DebugConfig:
    SHOW_LANDMARKS = True        # Draw facial landmarks on video feed
    SHOW_FPS = True              # Display FPS counter on screen