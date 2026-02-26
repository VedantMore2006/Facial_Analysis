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
        self.mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mesh.process(rgb)

        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None

    def draw(self, frame, landmarks):
        """
        Draw only the landmarks that are actually used in feature computation.
        Much cleaner than full 478-point tessellation.
        """
        h, w, _ = frame.shape
        
        for idx in LANDMARK_SUBSET:
            lm = landmarks.landmark[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)
            
            # Draw each used landmark as a small circle
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)