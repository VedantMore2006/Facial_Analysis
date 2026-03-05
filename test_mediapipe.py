import cv2
import os

os.environ['QT_QPA_PLATFORM'] = 'xcb'

from landmarks.mediapipe_detector import MediaPipeDetector
from landmarks.landmark_subset import extract_subset


def main():
    detector = MediaPipeDetector()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = detector.detect(frame)
        if landmarks is not None:
            print("Landmarks detected:", len(landmarks))
            lm = landmarks[61]
            print("Landmark 61:", lm.x, lm.y, lm.z)
            subset = extract_subset(landmarks)
            print("Subset size:", len(subset))

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
