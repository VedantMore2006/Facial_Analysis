"""
Behavioral Feature Extraction Pipeline

Complete pipeline integrating all 34 features with:
- MediaPipe landmark detection
- Head pose estimation and frame rejection
- Frame buffering for temporal features
- EMA smoothing
- Baseline collection (10 seconds)
- Z-score + sigmoid normalization
- Dual CSV logging (raw + scaled)

Usage:
    python run_pipeline.py

Controls:
    ESC : Exit
"""

import cv2
import time

from landmarks.mediapipe_detector import MediaPipeDetector
from landmarks.landmark_subset import extract_subset

from processing.head_pose import estimate_head_pose, reject_unstable_frames
from processing.ema_smoothing import FeatureEMAManager

from processing.baseline import BaselineCollector
from processing.feature_normalization import FeatureNormalizer

from core.frame_buffer import FrameBuffer
from core.feature_registry import FEATURE_REGISTRY

from output.csv_logger import CSVLogger

import os

os.environ['QT_QPA_PLATFORM'] = 'xcb'
def main():
    """
    Main pipeline execution.
    """

    print("=" * 60)
    print("BEHAVIORAL FEATURE EXTRACTION PIPELINE")
    print("=" * 60)
    print("\nInitializing components...")

    # Step 1: Initialize components
    detector = MediaPipeDetector()
    frame_buffer = FrameBuffer()
    ema_manager = FeatureEMAManager(alpha=0.3)
    baseline = BaselineCollector(duration=10)

    feature_names = [f.name for f in FEATURE_REGISTRY]

    # Normalizer created after baseline completes
    normalizer = None
    baseline_logged = False

    print(f"✓ {len(FEATURE_REGISTRY)} features loaded")
    print("✓ Components initialized")

    # Step 2: Start webcam
    print("\nStarting webcam...")
    cap = cv2.VideoCapture(0)  # Try default webcam first

    if not cap.isOpened():
        print("✗ Error: Cannot open webcam")
        return

    print("✓ Webcam opened")
    print("\n" + "=" * 60)
    print("BASELINE COLLECTION (10 seconds)")
    print("=" * 60)
    print("Please remain still and maintain neutral expression\n")
    print("Reading first frame...")

    start_time = time.time()
    frame_count = 0

    # Step 3: Create CSV logger
    logger = CSVLogger(feature_names)
    print(f"✓ CSV logging initialized")
    print("\nStarting main loop (Press ESC to exit)...")
    
    # Create window early
    cv2.namedWindow("Behavioral Analysis Pipeline", cv2.WINDOW_NORMAL)

    try:
        # Step 4: Main pipeline loop
        while True:

            ret, frame = cap.read()

            if not ret:
                print("✗ Error: Cannot read frame")
                break

            timestamp = time.time()
            frame_count += 1
            
            # Debug: Show we got a frame
            if frame_count == 1:
                print(f"✓ First frame received: {frame.shape}")
            
            # Show raw frame immediately for first few frames
            if frame_count <= 3:
                cv2.imshow("Behavioral Analysis Pipeline", frame)
                cv2.waitKey(1)
                # print(f"Frame {frame_count} displayed")  # Reduce terminal clutter

            # Step 5: Detect landmarks
            landmarks = detector.detect(frame)
            
            # Debug landmark detection
            if frame_count <= 3:
                if landmarks is None:
                    print(f"Frame {frame_count}: No landmarks detected")
                else:
                    print(f"Frame {frame_count}: Landmarks detected ✓")

            if landmarks is None:
                cv2.putText(
                    frame,
                    "No face detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Behavioral Analysis Pipeline", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            # Step 6: Extract subset
            subset = extract_subset(landmarks)

            # Step 7: Head pose estimation
            yaw, pitch, roll = estimate_head_pose(subset, frame.shape)
            
            # Debug head pose for first frame only
            if frame_count == 3:
                print(f"Frame 3 pose: yaw={yaw:.1f}°, pitch={pitch:.1f}°, roll={roll:.1f}°")
                print("(Head pose rejection temporarily DISABLED for testing)")
            
            # TODO: Fix head pose calculation - currently disabled to allow data collection
            # Temporarily accepting all frames regardless of pose
            unstable = False  # reject_unstable_frames(yaw, pitch)

            if unstable:
                # Count rejected frames
                if frame_count <= 10:
                    print(f"Frame {frame_count}: REJECTED (unstable)")
                cv2.putText(
                    frame,
                    f"Unstable frame (yaw={yaw:.1f}, pitch={pitch:.1f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                )
                cv2.imshow("Behavioral Analysis Pipeline", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            # Step 8: Compute features
            raw_features = {}

            for feature in FEATURE_REGISTRY:
                try:
                    value = feature.compute(subset, frame_buffer, timestamp)
                    raw_features[feature.name] = value
                except Exception as e:
                    raw_features[feature.name] = 0
            
            # Confirm feature computation for first frame
            if frame_count == 1:
                print(f"✓ Frame {frame_count}: {len(raw_features)} features computed")

            # Step 9: Apply EMA smoothing
            smoothed_features = {}

            for name, value in raw_features.items():
                smoothed = ema_manager.smooth(name, value)
                smoothed_features[name] = smoothed

            # Step 10: Add frame to buffer
            frame_buffer.add_frame(
                timestamp,
                subset,
                (yaw, pitch, roll),
                smoothed_features
            )

            # Step 11: Baseline collection
            if not baseline.completed:
                baseline.update(smoothed_features, timestamp)
                # Debug baseline progress
                if frame_count <= 10 or frame_count % 50 == 0:
                    elapsed = timestamp - (baseline.start_time or timestamp)
                    print(f"Frame {frame_count}: Baseline update | Elapsed: {elapsed:.1f}s / 10s")

            # Step 12: Normalize features
            if baseline.completed:
                # Initialize normalizer once after baseline completes
                if normalizer is None:
                    mu, sigma = baseline.get_baseline()
                    normalizer = FeatureNormalizer(mu, sigma)
                    if not baseline_logged:
                        print("\n" + "=" * 60)
                        print("BASELINE COMPLETE - STARTING FEATURE LOGGING")
                        print("=" * 60 + "\n")
                        baseline_logged = True

                scaled_features = normalizer.normalize(smoothed_features)
                status = f"Logging (Buffer: {frame_buffer.size()}/150)"
            else:
                scaled_features = smoothed_features
                elapsed = timestamp - start_time
                remaining = max(0, 10 - elapsed)
                status = f"Baseline: {remaining:.1f}s remaining"

            # Step 13: Log CSV
            logger.log(timestamp, smoothed_features, scaled_features)
            
            # Confirm logging for first few frames
            if frame_count <= 3:
                print(f"✓ Frame {frame_count}: Data logged to CSV")
            
            # Show progress every 50 frames
            if frame_count % 50 == 0:
                print(f"Processed {frame_count} frames | {status}")

            # Step 14: Display frame
            cv2.putText(
                frame,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Behavioral Analysis Pipeline", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                print("\nExiting pipeline...")
                break
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # Step 15: Cleanup
        cap.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Average FPS: {fps:.1f}")
        print(f"Baseline completed: {'Yes' if baseline.completed else 'No'}")

        print(f"\nCSV files saved:")
        print(f"  Raw:    {logger.raw_path}")
        print(f"  Scaled: {logger.scaled_path}")

        print("\n✓ Pipeline terminated successfully")


if __name__ == "__main__":
    main()
