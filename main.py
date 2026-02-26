# /home/vedant/Facial_analysis/main.py

# Main entry point for Facial Analysis
# Main entry point for Facial Analysis

from src.camera import Camera
from src.face_mesh import FaceMeshDetector
import cv2
import os
import mediapipe as mp

# Wayland compatibility
os.environ["QT_QPA_PLATFORM"] = "xcb"

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Initialize camera
cam = Camera()



# Initialize face mesh detector  👈 THIS WAS MISSING
detector = FaceMeshDetector()

while True:
    ret, frame = cam.read()
    if not ret:
        break

    landmarks = detector.process(frame)

    if landmarks is not None:
        print(len(landmarks.landmark))

    for lm in landmarks.landmark:
        x = int(lm.x * frame.shape[1])
        y = int(lm.y * frame.shape[0])
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    
    if landmarks:
        mp_drawing.draw_landmarks(
            frame,
            landmarks,
            mp_face_mesh.FACEMESH_CONTOURS
        )

    cv2.imshow("Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

print("Detected FPS:", cam.get_fps())

cam.release()
cv2.destroyAllWindows()