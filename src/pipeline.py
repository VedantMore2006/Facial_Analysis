# ============================================================================
# MAIN RUNTIME PIPELINE ORCHESTRATOR
# ============================================================================
# Purpose: Coordinates the entire facial analysis workflow from start to finish
# 
# Pipeline Flow:
# 1. INITIALIZATION
#    - Setup camera, face detector, feature engines
#    - Initialize baseline manager and smoothers
#    - Create data structures for tracking
# 
# 2. BASELINE COLLECTION PHASE (first 30 seconds)
#    - Detect face and extract landmarks
#    - Compute raw features
#    - Collect samples for baseline statistics
#    - Display "BASELINE" indicator
# 
# 3. BASELINE FINALIZATION
#    - Compute μ and σ for all features
#    - Lock baseline (prevents further modifications)
# 
# 4. DEVIATION DETECTION PHASE (rest of session)
#    - Continue computing raw features
#    - Scale features using baseline statistics
#    - Log privacy-safe scaled features
#    - Display real-time metrics
# 
# 5. CLEANUP & REPORTING
#    - Close files
#    - Release camera
#    - Print summary statistics
# 
# Key Design Principles:
# - Modular: Each component has single responsibility
# - Privacy-first: Only logs scaled features, not landmarks
# - Real-time: Processes and displays at camera FPS
# - Configurable: All parameters from config.py
# ============================================================================

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
import numpy as np

# Force X11 backend for OpenCV on Linux (prevents Wayland issues)
os.environ["QT_QPA_PLATFORM"] = "xcb"

def run_pipeline():
    """
    Main pipeline execution function.
    
    Runs the complete facial analysis workflow:
    - Initializes all components
    - Runs baseline collection phase
    - Transitions to deviation detection phase
    - Logs privacy-safe features
    - Handles user interaction and cleanup
    """
    # ========================================================================
    # INITIALIZATION: Setup all components
    # ========================================================================
    
    # Core components
    cam = Camera()                          # Webcam interface
    detector = FaceMeshDetector()           # MediaPipe face landmark detector
    logger = LandmarkLogger()               # Raw landmark logger (debug)
    feature_logger = FeatureLogger()        # Privacy-safe feature logger

    # Baseline management
    baseline_manager = BaselineManager()    # Collects and computes baseline stats
    baseline_finalized = False              # Tracks if baseline has been locked

    # Smoothing filters for noise reduction (5-frame moving average)
    au12_smoother = MovingAverage(window_size=5)
    expressivity_smoother = MovingAverage(window_size=5)
    head_smoother = MovingAverage(window_size=5)
    
    # Feature computation engines
    expressivity_engine = Expressivity()    # Facial movement variation
    head_engine = HeadYawVelocity()         # Head rotation speed
    
    # Time-dependent engines (need FPS for rate calculations)
    fps = cam.get_fps()
    blink_engine = BlinkDetector(fps)       # Blink frequency detection
    eye_engine = EyeContact(fps)            # Eye gaze alignment
    latency_engine = ResponseLatency(fps)   # Response time measurement
    
    # Baseline collection for response latency
    mouth_baseline_values = []              # Stores mouth opening values during baseline
    last_latency_scaled = 0.5               # Event-based: holds last detected scaled latency

    # Data tracking lists for post-session analysis (optional)
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
    ear_list = []                           # Eye Aspect Ratio values
    
    eye_raw_list = []
    eye_scaled_list = []
    
    latency_per_frame_list = []             # Track latency per frame for heatmap

    # Frame counting and timing
    frame_index = 0
    session_start = time.time()

    # Calculate when baseline phase ends
    baseline_end_time = session_start + BaselineConfig.DURATION_SECONDS
    baseline_frames = int(BaselineConfig.DURATION_SECONDS * fps)

    print("Session started.")
    print(f"Baseline window: {BaselineConfig.DURATION_SECONDS} seconds")

    # ========================================================================
    # MAIN PROCESSING LOOP: Run until user quits
    # ========================================================================
    while True:
        # Capture frame from camera
        ret, frame = cam.read()
        if not ret:
            break  # Camera disconnected or error

        # Get current timestamp
        current_time = time.time()
        timestamp_ms = int(current_time * 1000)

        # ====================================================================
        # FACE DETECTION: Extract landmarks from current frame
        # ====================================================================
        landmarks = detector.process(frame)

        if landmarks:
            # Extract minimal landmark subset (reduces processing overhead)
            subset = extract_subset(landmarks)

            # ================================================================
            # RAW FEATURE COMPUTATION: Compute current frame's features
            # ================================================================
            
            # Smile intensity (mouth corner width normalized by face size)
            au12_raw = compute_au12(subset)
            au12_smoothed = au12_smoother.update(au12_raw)
            
            # Facial expressivity (how much face is moving)
            expressivity_raw = expressivity_engine.compute(subset)
            expressivity_smoothed = expressivity_smoother.update(expressivity_raw)
            
            # Head movement speed (yaw rotation velocity)
            head_raw = head_engine.compute_velocity(subset)
            head_smoothed = head_smoother.update(head_raw)
            
            # Eye contact detection (is person looking at camera?)
            yaw_value = head_engine.get_current_yaw()
            contact, contact_ratio = eye_engine.update(yaw_value)
            
            # Blink detection (EAR-based with rate calculation)
            ear, blink_event, blink_rate = blink_engine.update(subset)

            # Package features into dictionary for baseline collection
            feature_dict = {
                "au12": au12_smoothed,
                "expressivity": expressivity_smoothed,
                "head_velocity": head_smoothed,
                "blink_rate": blink_rate,
                "eye_contact": contact_ratio
            }

            # ================================================================
            # BASELINE vs DEVIATION PHASE
            # ================================================================
            if not baseline_manager.locked:
                # ============================================================
                # BASELINE COLLECTION PHASE
                # ============================================================
                # Collecting samples to establish personal baseline
                
                # Collect feature samples for statistical analysis
                baseline_manager.collect_features(feature_dict)
                baseline_manager.collect_landmarks(subset)
                
                # Collect mouth baseline for response latency detection
                mouth_open = latency_engine.compute_mouth_open(subset)
                mouth_baseline_values.append(mouth_open)

                # Check if baseline period has ended
                if current_time > baseline_end_time:
                    # Finalize baseline: compute μ and σ for all features
                    baseline_manager.finalize()
                    
                    # Calculate and set mouth baseline stats for latency engine
                    mouth_mu = np.mean(mouth_baseline_values)
                    mouth_sigma = np.std(mouth_baseline_values)
                    latency_engine.set_baseline_stats(mouth_mu, mouth_sigma)

                # During baseline, all scaled values set to neutral (0.5)
                au12_scaled = 0.5
                expressivity_scaled = 0.5
                head_scaled = 0.5
                blink_scaled = 0.5
                eye_scaled = 0.5
                last_latency_scaled = 0.5
            else:
                # ============================================================
                # DEVIATION DETECTION PHASE
                # ============================================================
                # Baseline is locked, now computing deviations from baseline
                
                # Scale AU12 (smile) using baseline statistics
                stats_au12 = baseline_manager.get_feature_stats("au12")
                mu_au12 = stats_au12["mu"]
                sigma_au12 = stats_au12["sigma"]
                au12_scaled = scale_value(au12_smoothed, mu_au12, sigma_au12)
                
                # Scale expressivity using baseline statistics
                stats_expr = baseline_manager.get_feature_stats("expressivity")
                mu_expr = stats_expr["mu"]
                sigma_expr = stats_expr["sigma"]
                expressivity_scaled = scale_value(expressivity_smoothed, mu_expr, sigma_expr)
                
                # Scale head velocity using baseline statistics
                stats_head = baseline_manager.get_feature_stats("head_velocity")
                mu_head = stats_head["mu"]
                sigma_head = stats_head["sigma"]
                head_scaled = scale_value(head_smoothed, mu_head, sigma_head)
                
                # Scale blink rate using baseline statistics
                stats_blink = baseline_manager.get_feature_stats("blink_rate")
                mu_blink = stats_blink["mu"]
                sigma_blink = stats_blink["sigma"]
                blink_scaled = scale_value(blink_rate, mu_blink, sigma_blink)
                
                # Scale eye contact using baseline statistics
                stats_eye = baseline_manager.get_feature_stats("eye_contact")
                mu_eye = stats_eye["mu"]
                sigma_eye = stats_eye["sigma"]
                eye_scaled = scale_value(contact_ratio, mu_eye, sigma_eye)

            # ================================================================
            # RESPONSE LATENCY: Event-based detection
            # ================================================================
            # Update latency engine per frame (checks for mouth opening)
            latency_value = latency_engine.update(subset, current_time)
            
            # If a new response latency was detected, scale and store it
            if latency_value is not None:
                # Scale latency using simple z-score (could use baseline in future)
                last_latency_scaled = scale_value(latency_value, 0.5, 0.2) if baseline_manager.locked else 0.5
            
            # Track latency per frame for heatmap visualization
            # (holds last detected value between events)
            latency_per_frame_list.append(last_latency_scaled)
            
            # ================================================================
            # BUILD FEATURE VECTOR: Create privacy-safe output
            # ================================================================
            feature_vector = build_feature_vector(
                au12_scaled,
                expressivity_scaled,
                head_scaled,
                eye_scaled,
                blink_scaled,
                last_latency_scaled
            )
            
            # Confirm all values are bounded [0,1] (safety assertion)
            for v in feature_vector:
                assert 0.0 <= v <= 1.0, f"Feature value {v} out of bounds [0,1]"
            
            # ================================================================
            # LOGGING: Write privacy-safe scaled features to CSV
            # ================================================================
            # NOTE: NO raw landmarks are logged, only scaled features
            feature_logger.log(feature_vector)

            # ================================================================
            # VISUALIZATION: Draw landmarks and overlay text
            # ================================================================
            if DebugConfig.SHOW_LANDMARKS:
                detector.draw(frame, landmarks)

            # Display current phase (BASELINE or DEVIATION)
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

            # Display all scaled feature values on screen
            cv2.putText(
                frame,
                f"AU12: {au12_scaled:.3f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )
            
            cv2.putText(
                frame,
                f"Expressivity: {expressivity_scaled:.3f}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )
            
            cv2.putText(
                frame,
                f"Head Velocity: {head_scaled:.3f}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 255),
                2
            )
            
            cv2.putText(
                frame,
                f"Blink Rate: {blink_scaled:.3f}",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 100, 200),
                2
            )
            
            cv2.putText(
                frame,
                f"Eye Contact: {eye_scaled:.3f}",
                (20, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (100, 200, 255),
                2
            )

            # ================================================================
            # DATA TRACKING: Store values for post-session analysis
            # ================================================================
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

        # ====================================================================
        # DISPLAY: Show video feed with overlays
        # ====================================================================
        cv2.imshow("Facial Analysis", frame)

        # ====================================================================
        # KEYBOARD INPUT: Handle user commands
        # ====================================================================
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break  # Quit application
        elif key == ord("s"):
            # Stimulus trigger: marks the end of a question/prompt
            # Used to measure response latency (time until first response)
            latency_engine.set_stimulus(current_time)

        frame_index += 1

    # ========================================================================
    # CLEANUP & REPORTING: Close files and print session summary
    # ========================================================================
    print("Session ended.")
    
    # Close all file handles
    logger.close()
    feature_logger.close()
    
    # Print results summary
    print(f"\n✅ Features saved to: {feature_logger.filepath}")
    print(f"   Detected FPS: {cam.get_fps()}")
    print("\n📊 Visualize results in Streamlit:")
    print("   streamlit run app.py")
    
    # Release camera and close windows
    cam.release()
    cv2.destroyAllWindows()