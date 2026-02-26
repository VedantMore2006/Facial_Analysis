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

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

class FaceMeshDetector:
    def __init__(self):
        self.mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )

        # Clean drawing style
        self.landmark_style = mp_drawing.DrawingSpec(
            color=(0, 255, 0),
            thickness=1,
            circle_radius=1  # 👈 reduced radius
        )

        self.connection_style = mp_drawing.DrawingSpec(
            color=(0, 200, 0),
            thickness=1
        )

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mesh.process(rgb)

        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None

    def draw(self, frame, landmarks):
        mp_drawing.draw_landmarks(
            frame,
            landmarks,
            mp_face_mesh.FACEMESH_TESSELATION,  # Full mesh
            landmark_drawing_spec=self.landmark_style,
            connection_drawing_spec=self.connection_style
        )