import cv2
import mediapipe as mp


class MediaPipeDetector:
    def __init__(self):
        """
        Initialize MediaPipe FaceMesh detector.
        """
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame):
        """
        Detect facial landmarks from a frame.

        Returns:
            list of landmarks or None
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        return results.multi_face_landmarks[0].landmark
