import os
# Ensure Qt uses XCB on Linux to avoid Wayland-related issues when showing windows.
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import cv2
from input.video_source import VideoSource
from processing.landmark_detection import FaceLandmarkDetector
from processing.feature_extraction import FeatureExtractor

# Temporal processing: baseline normalization and windowed aggregation.
from temporal.baseline import BaselineNormalizer
from temporal.aggregation import TemporalAggregator

# Configure temporal modules:
# - BaselineNormalizer: learns a baseline over the first N frames
# - TemporalAggregator: aggregates normalized features over a sliding window
baseline = BaselineNormalizer(warmup_frames=30)
aggregator = TemporalAggregator(window_size=10)

# Landmark index groups used for feature calculations (MediaPipe Face Mesh indices)
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_BROW = [65, 66, 67]
RIGHT_BROW = [295, 296, 297]

def main():
    # Initialize video capture at target FPS and processing modules
    video = VideoSource(source=0, target_fps=10)
    detector = FaceLandmarkDetector()
    extractor = FeatureExtractor()

    try:
        while True:
            frame = video.get_frame()
            if frame is None:
                continue

            # Detect face landmarks and get an annotated frame for display
            landmarks, annotated = detector.process(frame)

            if landmarks:
                h, w, _ = frame.shape

                # Compute per-frame facial features
                ear_l = extractor.eye_aspect_ratio(landmarks.landmark, LEFT_EYE, w, h)
                ear_r = extractor.eye_aspect_ratio(landmarks.landmark, RIGHT_EYE, w, h)
                mor = extractor.mouth_opening_ratio(landmarks.landmark, w, h)
                jaw = extractor.jaw_drop(landmarks.landmark, w, h)

                # Prepare features dictionary for temporal processing
                features = {
                    "ear_l": ear_l,
                    "ear_r": ear_r,
                    "mor": mor,
                    "jaw": jaw,
                }

                # Update baseline model and normalize current features
                baseline.update(features)
                normalized = baseline.normalize(features)

                # If normalized features are available (after warmup), aggregate temporally
                if normalized:
                    aggregated = aggregator.update(normalized)
                    if aggregated:
                        # Print aggregated temporal features for downstream logging/analysis
                        print("Temporal Features:", aggregated)

                cv2.putText(
                    annotated,
                    f"EAR L/R: {ear_l:.2f} / {ear_r:.2f} | MOR: {mor:.2f} | Jaw: {jaw:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            # Display the annotated frame and overlay of instantaneous features
            cv2.imshow("Phase 4.4 - Feature Extraction", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Clean up resources on exit
        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
