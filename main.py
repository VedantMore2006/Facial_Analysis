# from src.camera import Camera
# from src.face_mesh import FaceMeshDetector
# import cv2
# import os
# from config import DebugConfig
# from src.landmark_processor import extract_subset
# from src.logger import LandmarkLogger
# import time
# os.environ["QT_QPA_PLATFORM"] = "xcb"

# logger = LandmarkLogger()
# frame_index = 0

# cam = Camera()
# detector = FaceMeshDetector()

# saved = False  # prevent multiple saves

# while True:
#     ret, frame = cam.read()
#     if not ret:
#         break
    

#     landmarks = detector.process(frame)

#     current_time = int(time.time() * 1000)

#     if landmarks:
#         subset = extract_subset(landmarks)
#         logger.log(frame_index, current_time, subset)
#     frame_index += 1

#     if landmarks:
#         if DebugConfig.SHOW_LANDMARKS:
#             detector.draw(frame, landmarks)

#         # Save first valid frame with landmarks
#         if not saved:
#             os.makedirs("data", exist_ok=True)
#             cv2.imwrite("data/test_landmarks.jpg", frame)
#             print("Saved test frame to data/test_landmarks.jpg")
#             saved = True

#     cv2.imshow("Test", frame)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#         break

# print("Detected FPS:", cam.get_fps())
# logger.close()
# cam.release()
# cv2.destroyAllWindows()

"""
Main entry point.
No logic should live here.
"""

from src.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()