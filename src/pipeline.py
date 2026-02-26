"""
pipeline.py

Purpose:
Orchestrates full runtime loop.

Responsibilities:
- Frame loop
- Landmark extraction
- Baseline management
- Logging

Does NOT:
- Compute feature math
"""

import time
import cv2
from src.camera import Camera
from src.face_mesh import FaceMeshDetector
from src.landmark_processor import extract_subset
from src.logger import LandmarkLogger
from config import BaselineConfig, DebugConfig
from src.feature_engine.au12 import compute_au12
from src.baseline import BaselineManager
from src.smoothing import MovingAverage
from src.scaler import scale_value
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

def run_pipeline():

    cam = Camera()
    detector = FaceMeshDetector()
    logger = LandmarkLogger()

    baseline_manager = BaselineManager()
    baseline_finalized = False

    au12_smoother = MovingAverage(window_size=5)

    frame_index = 0
    session_start = time.time()

    baseline_end_time = session_start + BaselineConfig.DURATION_SECONDS

    print("Session started.")
    print(f"Baseline window: {BaselineConfig.DURATION_SECONDS} seconds")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        current_time = time.time()
        timestamp_ms = int(current_time * 1000)

        landmarks = detector.process(frame)

        if landmarks:
            subset = extract_subset(landmarks)

            # Raw feature computation
            au12_raw = compute_au12(subset)
            au12_smoothed = au12_smoother.update(au12_raw)

            feature_dict = {
                "au12": au12_smoothed
            }

            # Baseline collection or deviation phase
            if not baseline_manager.locked:
                baseline_manager.collect(feature_dict)

                if current_time > baseline_end_time:
                    baseline_manager.finalize()
                
                au12_scaled = 0.5  # neutral during baseline
            else:
                # Deviation computation
                stats = baseline_manager.get_stats()
                mu = stats["au12"]["mu"]
                sigma = stats["au12"]["sigma"]

                au12_scaled = scale_value(au12_smoothed, mu, sigma)

            logger.log(frame_index, timestamp_ms, subset)

            if DebugConfig.SHOW_LANDMARKS:
                detector.draw(frame, landmarks)

            # Optional overlay
            phase = "BASELINE" if not baseline_manager.locked else "DEVIATION"
            cv2.putText(
                frame,
                phase,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0) if phase == "BASELINE" else (0, 0, 255),
                2
            )

            # Display scaled AU12 value
            cv2.putText(
                frame,
                f"AU12: {au12_scaled:.3f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

        cv2.imshow("Facial Analysis", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        frame_index += 1

    print("Session ended.")
    logger.close()
    print("Detected FPS:", cam.get_fps())
    cam.release()
    cv2.destroyAllWindows()