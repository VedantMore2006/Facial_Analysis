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
from src.feature_logger import FeatureLogger
from config import BaselineConfig, DebugConfig
from src.feature_engine.au12 import compute_au12
from src.feature_engine.expressivity import Expressivity
from src.feature_engine.head_velocity import HeadYawVelocity
from src.feature_engine.blink import BlinkDetector
from src.feature_engine.eye_contact import EyeContact
from src.baseline import BaselineManager
from src.smoothing import MovingAverage
from src.scaler import scale_value
import os
from datetime import datetime
from pathlib import Path
from src.feature_engine.response_latency import ResponseLatency
from src.feature_vector import build_feature_vector
from src.visualization import plot_signal, plot_heatmap, print_summary_statistics
from src.dashboard import create_dashboard
import numpy as np
os.environ["QT_QPA_PLATFORM"] = "xcb"

def run_pipeline():

    cam = Camera()
    detector = FaceMeshDetector()
    logger = LandmarkLogger()
    feature_logger = FeatureLogger()

    baseline_manager = BaselineManager()
    baseline_finalized = False

    au12_smoother = MovingAverage(window_size=5)
    expressivity_smoother = MovingAverage(window_size=5)
    expressivity_engine = Expressivity()
    head_engine = HeadYawVelocity()
    head_smoother = MovingAverage(window_size=5)
    
    fps = cam.get_fps()
    blink_engine = BlinkDetector(fps)
    eye_engine = EyeContact(fps)
    latency_engine = ResponseLatency(fps)
    mouth_baseline_values = []
    last_latency_scaled = 0.5  # Event-based: holds last detected scaled latency

    au12_raw_list = []
    au12_smooth_list = []
    au12_scaled_list = []
    
    expressivity_raw_list = []
    expressivity_smooth_list = []
    expressivity_scaled_list = []
    
    head_raw_list = []
    head_smooth_list = []
    head_scaled_list = []
    
    blink_raw_list = []
    blink_scaled_list = []
    ear_list = []
    
    eye_raw_list = []
    eye_scaled_list = []
    
    latency_per_frame_list = []  # Track latency per frame for heatmap

    frame_index = 0
    session_start = time.time()

    baseline_end_time = session_start + BaselineConfig.DURATION_SECONDS
    baseline_frames = int(BaselineConfig.DURATION_SECONDS * fps)

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
            
            yaw_value = head_engine.get_current_yaw()
            contact, contact_ratio = eye_engine.update(yaw_value)
            
            ear, blink_event, blink_rate = blink_engine.update(subset)

            feature_dict = {
                "au12": au12_smoothed,
                "expressivity": expressivity_smoothed,
                "head_velocity": head_smoothed,
                "blink_rate": blink_rate,
                "eye_contact": contact_ratio
            }

            # Baseline collection or deviation phase
            if not baseline_manager.locked:
                baseline_manager.collect_features(feature_dict)
                baseline_manager.collect_landmarks(subset)
                
                # Collect mouth baseline for latency
                mouth_open = latency_engine.compute_mouth_open(subset)
                mouth_baseline_values.append(mouth_open)

                if current_time > baseline_end_time:
                    baseline_manager.finalize()
                    
                    # Calculate and set mouth baseline stats
                    mouth_mu = np.mean(mouth_baseline_values)
                    mouth_sigma = np.std(mouth_baseline_values)
                    latency_engine.set_baseline_stats(mouth_mu, mouth_sigma)

                au12_scaled = 0.5  # neutral during baseline
                expressivity_scaled = 0.5  # neutral during baseline
                head_scaled = 0.5  # neutral during baseline
                blink_scaled = 0.5  # neutral during baseline
                eye_scaled = 0.5  # neutral during baseline
                last_latency_scaled = 0.5  # neutral during baseline
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
                
                stats_blink = baseline_manager.get_feature_stats("blink_rate")
                mu_blink = stats_blink["mu"]
                sigma_blink = stats_blink["sigma"]
                blink_scaled = scale_value(blink_rate, mu_blink, sigma_blink)
                
                stats_eye = baseline_manager.get_feature_stats("eye_contact")
                mu_eye = stats_eye["mu"]
                sigma_eye = stats_eye["sigma"]
                eye_scaled = scale_value(contact_ratio, mu_eye, sigma_eye)

            # Update latency engine per frame (event-based)
            latency_value = latency_engine.update(subset, current_time)
            
            # If a new response latency was detected, scale and store it
            if latency_value is not None:
                # Scale latency (you may need baseline stats for latency in future)
                # For now, using event value directly if already scaled by engine
                # Or you can add latency baseline stats here
                last_latency_scaled = scale_value(latency_value, 0.5, 0.2) if baseline_manager.locked else 0.5
            
            # Track latency per frame for heatmap visualization
            latency_per_frame_list.append(last_latency_scaled)
            
            # Build feature vector from all scaled values
            feature_vector = build_feature_vector(
                au12_scaled,
                expressivity_scaled,
                head_scaled,
                eye_scaled,
                blink_scaled,
                last_latency_scaled
            )
            
            # Confirm bounds (debug assertion)
            for v in feature_vector:
                assert 0.0 <= v <= 1.0, f"Feature value {v} out of bounds [0,1]"
            
            # Log privacy-safe scaled features only (NO raw landmarks)
            feature_logger.log(feature_vector)

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
            
            # Display scaled Blink Rate value
            cv2.putText(
                frame,
                f"Blink Rate: {blink_scaled:.3f}",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 100, 200),
                2
            )
            
            # Display scaled Eye Contact value
            cv2.putText(
                frame,
                f"Eye Contact: {eye_scaled:.3f}",
                (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (100, 200, 255),
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
            
            blink_raw_list.append(blink_rate)
            blink_scaled_list.append(blink_scaled)
            ear_list.append(ear)
            
            eye_raw_list.append(contact_ratio)
            eye_scaled_list.append(eye_scaled)

        cv2.imshow("Facial Analysis", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            # Stimulus trigger: question ends at this moment
            latency_engine.set_stimulus(current_time)

        frame_index += 1

    print("Session ended.")
    logger.close()
    feature_logger.close()
    print(f"Features saved to: {feature_logger.filepath}")
    print("Detected FPS:", cam.get_fps())
    cam.release()
    cv2.destroyAllWindows()

    import matplotlib.pyplot as plt

    # Create plots directory if it doesn't exist
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use new visualization functions with baseline marking
    print("\nGenerating signal visualizations...")
    
    # Plot AU12
    au12_plot_file = plots_dir / f"au12_signal_{timestamp}.png"
    plot_signal(au12_raw_list, au12_smooth_list, au12_scaled_list, "AU12 Signal", baseline_frames=baseline_frames)
    plt.savefig(au12_plot_file, dpi=100, bbox_inches="tight")
    print(f"AU12 plot saved: {au12_plot_file}")
    
    # Plot Expressivity
    expr_plot_file = plots_dir / f"expressivity_signal_{timestamp}.png"
    plot_signal(expressivity_raw_list, expressivity_smooth_list, expressivity_scaled_list, "Expressivity Signal", baseline_frames=baseline_frames)
    plt.savefig(expr_plot_file, dpi=100, bbox_inches="tight")
    print(f"Expressivity plot saved: {expr_plot_file}")
    
    # Plot Head Velocity
    head_plot_file = plots_dir / f"head_velocity_signal_{timestamp}.png"
    plot_signal(head_raw_list, head_smooth_list, head_scaled_list, "Head Velocity Signal", baseline_frames=baseline_frames)
    plt.savefig(head_plot_file, dpi=100, bbox_inches="tight")
    print(f"Head Velocity plot saved: {head_plot_file}")
    
    # Plot Blink Rate
    blink_plot_file = plots_dir / f"blink_rate_signal_{timestamp}.png"
    plot_signal(blink_raw_list, blink_raw_list, blink_scaled_list, "Blink Rate Signal", baseline_frames=baseline_frames)
    plt.savefig(blink_plot_file, dpi=100, bbox_inches="tight")
    print(f"Blink Rate plot saved: {blink_plot_file}")
    
    # Plot EAR (Eye Aspect Ratio)
    ear_plot_file = plots_dir / f"ear_signal_{timestamp}.png"
    plt.figure(figsize=(12, 5))
    plt.plot(ear_list, label="EAR")
    if baseline_frames is not None:
        plt.axvspan(0, baseline_frames, alpha=0.2, color='gray', label="Baseline")
    plt.title("EAR (Eye Aspect Ratio) Signal")
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ear_plot_file, dpi=100, bbox_inches="tight")
    print(f"EAR plot saved: {ear_plot_file}")
    plt.show()
    
    # Plot Eye Contact
    eye_plot_file = plots_dir / f"eye_contact_signal_{timestamp}.png"
    plot_signal(eye_raw_list, eye_raw_list, eye_scaled_list, "Eye Contact Signal", baseline_frames=baseline_frames)
    plt.savefig(eye_plot_file, dpi=100, bbox_inches="tight")
    print(f"Eye Contact plot saved: {eye_plot_file}")
    
    # Generate behavioral heatmap
    print("\nGenerating behavioral deviation heatmap...")
    feature_matrix = np.array([
        au12_scaled_list,
        expressivity_scaled_list,
        head_scaled_list,
        eye_scaled_list,
        blink_scaled_list,
        latency_per_frame_list
    ])
    
    feature_names = [
        "AU12",
        "Expressivity",
        "Head Velocity",
        "Eye Contact",
        "Blink Rate",
        "Response Latency"
    ]
    
    heatmap_file = plots_dir / f"behavioral_heatmap_{timestamp}.png"
    plot_heatmap(feature_matrix, feature_names, baseline_frames=baseline_frames)
    plt.savefig(heatmap_file, dpi=100, bbox_inches="tight")
    print(f"Heatmap saved: {heatmap_file}")
    
    # Generate summary statistics
    print("\nGenerating summary statistics...")
    print_summary_statistics(feature_names, [
        au12_scaled_list,
        expressivity_scaled_list,
        head_scaled_list,
        eye_scaled_list,
        blink_scaled_list,
        latency_per_frame_list
    ])
    
    # Generate comprehensive dashboard
    print("\nGenerating behavioral dashboard...")
    feature_dict_lists = {
        "AU12": au12_scaled_list,
        "Expressivity": expressivity_scaled_list,
        "Head Velocity": head_scaled_list,
        "Eye Contact": eye_scaled_list,
        "Blink Rate": blink_scaled_list,
        "Response Latency": latency_per_frame_list
    }
    
    dashboard_names = list(feature_dict_lists.keys())
    
    dashboard_file = plots_dir / f"behavioral_dashboard_{timestamp}.png"
    dashboard_fig = create_dashboard(feature_dict_lists, dashboard_names, baseline_frames)
    dashboard_fig.savefig(dashboard_file, dpi=100, bbox_inches="tight")
    print(f"Dashboard saved: {dashboard_file}")