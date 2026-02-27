# ============================================================================
# FACE MESH DETECTOR
# ============================================================================
# Purpose: Wrapper for MediaPipe FaceMesh to detect 478 facial landmarks
# 
# Key Features:
# - Processes video frames to detect face landmarks
# - Returns 478 3D points (x, y, z normalized coordinates)
# - Provides visualization utility for drawing landmarks
# 
# MediaPipe FaceMesh:
# - Google's pre-trained model for real-time face landmark detection
# - Detects 478 points covering entire face (eyes, mouth, contours, etc.)
# - refine_landmarks=True enables iris detection for better eye tracking
# 
# Does NOT:
# - Compute features (just provides raw landmarks)
# - Normalize geometry (that's done in feature computation)
# - Store or log data
# ============================================================================

"""
face_mesh.py

Purpose:
Runs MediaPipe FaceMesh and optionally draws landmarks.

Responsibilities:
- Convert frame to RGB
- Run MediaPipe inference
- Return 478 landmarks
- Provide clean visualization utility

Does NOT:
- Compute features
- Normalize geometry
- Store landmarks
"""

import mediapipe as mp
import cv2
from src.landmark_processor import LANDMARK_SUBSET

mp_face_mesh = mp.solutions.face_mesh

class FaceMeshDetector:
    def __init__(self):
        """
        Initialize MediaPipe FaceMesh detector.
        
        Configuration:
        - static_image_mode=False: Optimized for video (uses temporal info)
        - max_num_faces=1: Only detect one face (improves performance)
        - refine_landmarks=True: Enables iris landmarks for eye tracking
        """
        self.mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )

    def process(self, frame):
        """
        Detect face landmarks in a video frame.
        
        Args:
            frame: BGR image from camera (NumPy array)
        
        Returns:
            Landmark object with 478 points if face detected, None otherwise
            Each landmark has .x, .y, .z coordinates (normalized 0-1)
        """
        # Convert BGR to RGB (MediaPipe expects RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run face mesh inference
        results = self.mesh.process(rgb)

        # Return first detected face landmarks (or None if no face found)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None

    def draw(self, frame, landmarks):
        """
        Draw only the landmarks that are actually used in feature computation.
        Much cleaner than full 478-point tessellation.
        
        This draws only the subset of landmarks defined in LANDMARK_SUBSET,
        making it easier to see which points are being used for analysis.
        
        Args:
            frame: BGR image to draw on (modified in-place)
            landmarks: MediaPipe landmark object from process() method
        """
        # Get frame dimensions for coordinate conversion
        h, w, _ = frame.shape
        
        # Draw each landmark in the subset
        for idx in LANDMARK_SUBSET:
            lm = landmarks.landmark[idx]
            
            # Convert normalized coordinates to pixel coordinates
            x = int(lm.x * w)
            y = int(lm.y * h)
            
            # Draw each used landmark as a small green circle
            cv2.circle(frame, (x, y), 2, (0, 255, 0), 1)