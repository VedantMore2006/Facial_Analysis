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
from src.feature_engine.expressivity import Expressivity
from src.feature_engine.head_velocity import HeadYawVelocity
from src.baseline import BaselineManager
from src.smoothing import MovingAverage
from src.scaler import scale_value
import os
from datetime import datetime
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "xcb"

def run_pipeline():

    cam = Camera()
    detector = FaceMeshDetector()
    logger = LandmarkLogger()

    baseline_manager = BaselineManager()
    baseline_finalized = False

    au12_smoother = MovingAverage(window_size=5)
    expressivity_smoother = MovingAverage(window_size=5)
    expressivity_engine = Expressivity()
    head_engine = HeadYawVelocity()
    head_smoother = MovingAverage(window_size=5)

    au12_raw_list = []
    au12_smooth_list = []
    au12_scaled_list = []
    
    expressivity_raw_list = []
    expressivity_smooth_list = []
    expressivity_scaled_list = []
    
    head_raw_list = []
    head_smooth_list = []
    head_scaled_list = []

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
            
            expressivity_raw = expressivity_engine.compute(subset)
            expressivity_smoothed = expressivity_smoother.update(expressivity_raw)
            
            head_raw = head_engine.compute_velocity(subset)
            head_smoothed = head_smoother.update(head_raw)

            feature_dict = {
                "au12": au12_smoothed,
                "expressivity": expressivity_smoothed,
                "head_velocity": head_smoothed
            }

            # Baseline collection or deviation phase
            if not baseline_manager.locked:
                baseline_manager.collect_features(feature_dict)
                baseline_manager.collect_landmarks(subset)

                if current_time > baseline_end_time:
                    baseline_manager.finalize()

                au12_scaled = 0.5  # neutral during baseline
                expressivity_scaled = 0.5  # neutral during baseline
                head_scaled = 0.5  # neutral during baseline
            else:
                # Deviation computation
                stats_au12 = baseline_manager.get_feature_stats("au12")
                mu_au12 = stats_au12["mu"]
                sigma_au12 = stats_au12["sigma"]
                au12_scaled = scale_value(au12_smoothed, mu_au12, sigma_au12)
                
                stats_expr = baseline_manager.get_feature_stats("expressivity")
                mu_expr = stats_expr["mu"]
                sigma_expr = stats_expr["sigma"]
                expressivity_scaled = scale_value(expressivity_smoothed, mu_expr, sigma_expr)
                
                stats_head = baseline_manager.get_feature_stats("head_velocity")
                mu_head = stats_head["mu"]
                sigma_head = stats_head["sigma"]
                head_scaled = scale_value(head_smoothed, mu_head, sigma_head)

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
            
            # Display scaled Expressivity value
            cv2.putText(
                frame,
                f"Expressivity: {expressivity_scaled:.3f}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )
            
            # Display scaled Head Velocity value
            cv2.putText(
                frame,
                f"Head Velocity: {head_scaled:.3f}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 255),
                2
            )

            au12_raw_list.append(au12_raw)
            au12_smooth_list.append(au12_smoothed)
            au12_scaled_list.append(au12_scaled)
            
            expressivity_raw_list.append(expressivity_raw)
            expressivity_smooth_list.append(expressivity_smoothed)
            expressivity_scaled_list.append(expressivity_scaled)
            
            head_raw_list.append(head_raw)
            head_smooth_list.append(head_smoothed)
            head_scaled_list.append(head_scaled)

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

    import matplotlib.pyplot as plt

    # Create plots directory if it doesn't exist
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot AU12
    au12_plot_file = plots_dir / f"au12_signal_{timestamp}.png"
    plt.figure(figsize=(12, 5))
    plt.plot(au12_raw_list, label="Raw")
    plt.plot(au12_smooth_list, label="Smoothed")
    plt.plot(au12_scaled_list, label="Scaled")
    plt.legend()
    plt.title("AU12 Signal")
    plt.savefig(au12_plot_file, dpi=100, bbox_inches="tight")
    print(f"AU12 plot saved: {au12_plot_file}")
    plt.show()
    
    # Plot Expressivity
    expr_plot_file = plots_dir / f"expressivity_signal_{timestamp}.png"
    plt.figure(figsize=(12, 5))
    plt.plot(expressivity_raw_list, label="Raw")
    plt.plot(expressivity_smooth_list, label="Smoothed")
    plt.plot(expressivity_scaled_list, label="Scaled")
    plt.legend()
    plt.title("Expressivity Signal")
    plt.savefig(expr_plot_file, dpi=100, bbox_inches="tight")
    print(f"Expressivity plot saved: {expr_plot_file}")
    plt.show()
    
    # Plot Head Velocity
    head_plot_file = plots_dir / f"head_velocity_signal_{timestamp}.png"
    plt.figure(figsize=(12, 5))
    plt.plot(head_raw_list, label="Raw")
    plt.plot(head_smooth_list, label="Smoothed")
    plt.plot(head_scaled_list, label="Scaled")
    plt.legend()
    plt.title("Head Velocity Signal")
    plt.savefig(head_plot_file, dpi=100, bbox_inches="tight")
    print(f"Head Velocity plot saved: {head_plot_file}")
    plt.show()