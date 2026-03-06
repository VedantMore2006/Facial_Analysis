"""
Real-Time Mental Health Classification

Tests trained model with live webcam feed:
- Captures 5-second behavioral windows
- Extracts features in real-time
- Classifies behavior using trained XGBoost model
- Displays predictions with confidence scores

Usage:
    python test_model_live.py

Controls:
    ESC : Exit
    SPACE : Reset window buffer
"""

import cv2
import time
import pickle
import json
import pandas as pd
import numpy as np
from collections import deque

from landmarks.mediapipe_detector import MediaPipeDetector
from landmarks.landmark_subset import extract_subset
from processing.head_pose import estimate_head_pose, reject_unstable_frames
from core.frame_buffer import FrameBuffer
from core.feature_registry import FEATURE_REGISTRY

import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'


class RealTimeClassifier:
    """
    Real-time behavioral classification from webcam.
    """
    
    def __init__(self, model_path='ml/mental_health_model.pkl', 
                 metadata_path='ml/model_metadata.json',
                 window_duration=5.0, fps=30):
        """
        Initialize classifier.
        
        Parameters
        ----------
        model_path : str
            Path to trained XGBoost model
        metadata_path : str
            Path to model metadata
        window_duration : float
            Window size in seconds
        fps : int
            Expected frames per second
        """
        self.window_duration = window_duration
        self.fps = fps
        self.frames_per_window = int(window_duration * fps)
        
        # Load model
        print(f"Loading model: {model_path}")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        self.label_map = {int(k): v for k, v in metadata['label_map'].items()}
        
        print(f"✓ Model loaded")
        print(f"✓ Classes: {len(self.label_map)}")
        print(f"✓ Features: {len(self.feature_names)}")
        print(f"✓ Window: {window_duration}s ({self.frames_per_window} frames)")
        
        # Frame buffer for windowing
        self.frame_features = deque(maxlen=self.frames_per_window)
        
        # Prediction history for smoothing
        self.prediction_history = deque(maxlen=3)
        
    def extract_frame_features(self, landmarks_3d):
        """
        Extract all features from one frame.
        
        Parameters
        ----------
        landmarks_3d : np.ndarray
            Facial landmarks (468, 3)
        
        Returns
        -------
        dict
            Feature values for this frame
        """
        features = {}
        
        for feature_obj in FEATURE_REGISTRY:
            try:
                value = feature_obj.compute(landmarks_3d)
                features[feature_obj.name] = value
            except Exception as e:
                features[feature_obj.name] = 0.0
        
        return features
    
    def aggregate_window_features(self):
        """
        Aggregate buffered frames into window-level features.
        
        Returns
        -------
        pd.DataFrame or None
            Single row with aggregated features
        """
        if len(self.frame_features) < self.frames_per_window:
            return None
        
        # Convert frame features to DataFrame
        df = pd.DataFrame(list(self.frame_features))
        
        # Compute aggregation statistics
        agg_features = {}
        
        for col in df.columns:
            agg_features[f'{col}_mean'] = df[col].mean()
            agg_features[f'{col}_std'] = df[col].std()
            agg_features[f'{col}_max'] = df[col].max()
            agg_features[f'{col}_min'] = df[col].min()
        
        # Create single-row DataFrame
        window_df = pd.DataFrame([agg_features])
        
        # Ensure feature order matches model training
        # Reorder columns to match self.feature_names
        missing_cols = [col for col in self.feature_names if col not in window_df.columns]
        if missing_cols:
            # Add missing columns with 0
            for col in missing_cols:
                window_df[col] = 0.0
        
        # Select and order columns
        window_df = window_df[self.feature_names]
        
        return window_df
    
    def predict(self, window_features):
        """
        Classify behavior from window features.
        
        Parameters
        ----------
        window_features : pd.DataFrame
            Single row with aggregated features
        
        Returns
        -------
        tuple
            (predicted_label, confidence, all_probabilities)
        """
        # Get prediction probabilities
        probs = self.model.predict_proba(window_features)[0]
        
        # Get predicted class
        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class]
        
        return predicted_class, confidence, probs
    
    def get_smoothed_prediction(self, current_pred):
        """
        Smooth predictions over recent history.
        
        Parameters
        ----------
        current_pred : int
            Current prediction
        
        Returns
        -------
        int
            Smoothed prediction (most common in recent history)
        """
        self.prediction_history.append(current_pred)
        
        if len(self.prediction_history) < 2:
            return current_pred
        
        # Return most common prediction in history
        counts = {}
        for pred in self.prediction_history:
            counts[pred] = counts.get(pred, 0) + 1
        
        return max(counts.items(), key=lambda x: x[1])[0]
    
    def draw_prediction_overlay(self, frame, prediction, confidence, probabilities):
        """
        Draw prediction information on frame.
        
        Parameters
        ----------
        frame : np.ndarray
            Video frame
        prediction : int
            Predicted class label
        confidence : float
            Confidence score
        probabilities : np.ndarray
            All class probabilities
        """
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay panel
        overlay = frame.copy()
        panel_height = 280
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        frame[:] = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Title
        cv2.putText(frame, "MENTAL HEALTH CLASSIFICATION", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Prediction
        condition_name = self.label_map[prediction]
        color = self._get_color_for_class(prediction)
        
        cv2.putText(frame, f"Detected: {condition_name}", 
                   (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        cv2.putText(frame, f"Confidence: {confidence:.1%}", 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # All class probabilities
        y_offset = 160
        cv2.putText(frame, "All Classes:", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        y_offset += 25
        sorted_indices = np.argsort(probabilities)[::-1]
        
        for idx in sorted_indices[:6]:  # Show all 6 classes
            class_name = self.label_map[idx]
            prob = probabilities[idx]
            
            # Color based on probability
            if prob > 0.5:
                bar_color = (0, 255, 0)  # Green
            elif prob > 0.2:
                bar_color = (0, 165, 255)  # Orange
            else:
                bar_color = (100, 100, 100)  # Gray
            
            # Draw probability bar
            bar_width = int(200 * prob)
            cv2.rectangle(frame, (150, y_offset - 10), 
                         (150 + bar_width, y_offset + 5), bar_color, -1)
            
            # Class name and percentage
            text = f"{class_name[:15]:15s} {prob:5.1%}"
            cv2.putText(frame, text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 20
        
        # Buffer status
        buffer_pct = len(self.frame_features) / self.frames_per_window
        buffer_text = f"Buffer: {len(self.frame_features)}/{self.frames_per_window} " \
                      f"({buffer_pct:.0%})"
        
        buffer_color = (0, 255, 0) if buffer_pct >= 1.0 else (0, 165, 255)
        cv2.putText(frame, buffer_text, (w - 350, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, buffer_color, 2)
        
        # Instructions
        cv2.putText(frame, "ESC: Exit | SPACE: Reset Buffer", 
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _get_color_for_class(self, class_label):
        """Get display color for each class."""
        colors = {
            0: (147, 112, 219),  # Depression - Purple
            1: (0, 165, 255),    # Anxiety - Orange
            2: (0, 255, 255),    # Stress - Yellow
            3: (0, 255, 0),      # Bipolar Mania - Green
            4: (255, 0, 255),    # Phobia - Magenta
            5: (255, 0, 0)       # Suicidal - Red
        }
        return colors.get(class_label, (255, 255, 255))


def main():
    """
    Main execution loop.
    """
    print("=" * 70)
    print("REAL-TIME MENTAL HEALTH CLASSIFICATION")
    print("=" * 70)
    
    # Initialize classifier
    try:
        classifier = RealTimeClassifier()
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nMake sure you've trained the model first:")
        print("  python ml/run_full_pipeline.py")
        return
    
    # Initialize components
    print("\nInitializing video capture...")
    detector = MediaPipeDetector()
    # Lower detection thresholds for better sensitivity
    detector.face_mesh = detector.mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,  # Lower threshold
        min_tracking_confidence=0.3,   # Lower threshold
    )
    
    frame_buffer = FrameBuffer()
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Error: Cannot open webcam")
        return
    
    print("✓ Webcam opened")
    print("\n" + "=" * 70)
    print("LIVE CLASSIFICATION")
    print("=" * 70)
    print("\nInstructions:")
    print("  1. Wait for buffer to fill (5 seconds / 150 frames)")
    print("  2. Predictions will update continuously")
    print("  3. Press SPACE to reset buffer")
    print("  4. Press ESC to exit")
    print("\n" + "=" * 70)
    
    frame_count = 0
    last_prediction = None
    last_confidence = 0.0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("✗ Error reading frame")
            break
        
        frame_count += 1
        
        # Detect landmarks
        landmarks = detector.detect(frame)
        
        # Debug: Check landmark detection
        if frame_count % 30 == 0:  # Every second at 30fps
            if landmarks is None:
                print(f"Frame {frame_count}: ✗ No landmarks detected")
            else:
                print(f"Frame {frame_count}: ✓ Landmarks detected, buffer: {len(classifier.frame_features)}/150")
        
        if landmarks is not None:
            try:
                # Extract landmark subset
                subset = extract_subset(landmarks)
                
                # Estimate head pose (returns yaw, pitch, roll)
                yaw, pitch, roll = estimate_head_pose(subset, frame.shape)
                
                # Head pose rejection DISABLED - accept all frames
                # if not reject_unstable_frames(yaw, pitch):
                if True:  # Always accept frames
                    # Convert to 3D coordinates
                    landmarks_3d = np.array([
                        [lm.x, lm.y, lm.z] 
                        for lm in landmarks
                    ])
                    
                    # Extract frame features
                    frame_features = classifier.extract_frame_features(landmarks_3d)
                    
                    # Add to buffer
                    classifier.frame_features.append(frame_features)
                    
                    # If buffer is full, make prediction
                    if len(classifier.frame_features) >= classifier.frames_per_window:
                        # Aggregate features
                        window_features = classifier.aggregate_window_features()
                        
                        if window_features is not None:
                            # Predict
                            prediction, confidence, probs = classifier.predict(window_features)
                            
                            # Smooth prediction
                            smoothed_pred = classifier.get_smoothed_prediction(prediction)
                            
                            last_prediction = smoothed_pred
                            last_confidence = confidence
                    
                    # Draw landmarks on frame (optional visualization)
                    h, w = frame.shape[:2]
                    for lm in landmarks:
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                # else:
                #     # Frame rejected due to unstable head pose
                #     if frame_count % 30 == 0:
                #         print(f"Frame {frame_count}: ⚠ Unstable head pose (yaw={yaw:.1f}°, pitch={pitch:.1f}°)")
            
            except Exception as e:
                if frame_count % 30 == 0:
                    print(f"Frame {frame_count}: ✗ Error processing: {e}")
        else:
            # Show "no face detected" indicator on frame
            h, w = frame.shape[:2]
            cv2.putText(frame, "No face detected", (w//2 - 100, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw prediction overlay
        if last_prediction is not None:
            # Get current probabilities (if available)
            if len(classifier.frame_features) >= classifier.frames_per_window:
                window_features = classifier.aggregate_window_features()
                if window_features is not None:
                    _, _, current_probs = classifier.predict(window_features)
                else:
                    current_probs = np.zeros(len(classifier.label_map))
            else:
                current_probs = np.zeros(len(classifier.label_map))
            
            classifier.draw_prediction_overlay(
                frame, last_prediction, last_confidence, current_probs
            )
        else:
            # Show waiting message
            h, w = frame.shape[:2]
            text = "Collecting data... Please wait"
            cv2.putText(frame, text, (w//2 - 200, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            buffer_pct = len(classifier.frame_features) / classifier.frames_per_window
            progress_text = f"{buffer_pct:.0%} ({len(classifier.frame_features)}/{classifier.frames_per_window} frames)"
            cv2.putText(frame, progress_text, (w//2 - 100, h//2 + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display
        cv2.imshow('Mental Health Classification', frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("\n✓ Exiting...")
            break
        elif key == 32:  # SPACE
            print("\n✓ Buffer reset")
            classifier.frame_features.clear()
            classifier.prediction_history.clear()
            last_prediction = None
            last_confidence = 0.0
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print("SESSION COMPLETE")
    print("=" * 70)
    print(f"Total frames processed: {frame_count}")


if __name__ == "__main__":
    main()
