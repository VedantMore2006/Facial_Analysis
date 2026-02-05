import cv2
import mediapipe as mp
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'
class FaceLandmarkDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def process(self, frame):
        """
        Input: BGR frame (OpenCV)
        Output: (landmarks, annotated_frame)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        annotated_frame = frame.copy()
        landmarks = None

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = face_landmarks

            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_tesselation_style()
            )

        return landmarks, annotated_frame
