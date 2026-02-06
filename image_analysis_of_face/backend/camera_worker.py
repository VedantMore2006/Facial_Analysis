import threading
import time

import cv2

from input.video_source import VideoSource
from processing.landmark_detection import FaceLandmarkDetector
from processing.feature_extraction import FeatureExtractor
from temporal.baseline import BaselineNormalizer
from temporal.aggregation import TemporalAggregator
from output.csv_logger import CSVLogger
import state
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'
# -----------------------------
# Internal worker loop
# -----------------------------
def _pipeline_loop():
    print("[camera_worker] Pipeline started")

    try:
        video = VideoSource(source=0, target_fps=10)
    except RuntimeError as e:
        print("[camera_worker] Camera unavailable:", e)
        state.pipeline_running = False
        return

    detector = FaceLandmarkDetector()
    extractor = FeatureExtractor()
    baseline = BaselineNormalizer(warmup_frames=30)
    aggregator = TemporalAggregator(window_size=10)
    logger = CSVLogger(filename="facial_signals.csv", rate_hz=5)

    try:
        while state.pipeline_running:
            frame = video.get_frame()
            if frame is None:
                continue

            landmarks, _ = detector.process(frame)
            if not landmarks:
                continue

            h, w, _ = frame.shape

            # -----------------------------
            # Phase 4.4 features
            # -----------------------------
            LEFT_EYE  = [33, 160, 158, 133, 153, 144]
            RIGHT_EYE = [362, 385, 387, 263, 373, 380]

            ear_l = extractor.eye_aspect_ratio(landmarks.landmark, LEFT_EYE, w, h)
            ear_r = extractor.eye_aspect_ratio(landmarks.landmark, RIGHT_EYE, w, h)
            ear = (ear_l + ear_r) / 2.0

            mor = extractor.mouth_opening_ratio(landmarks.landmark, w, h)
            jaw = extractor.jaw_drop(landmarks.landmark, w, h)

            # Continuous blink proxy (NO thresholds)
            blink_signal = max(0.0, 1.0 - ear)

            # AU proxies
            au12 = extractor.action_unit_12(landmarks.landmark, w, h)
            au15 = extractor.action_unit_15(landmarks.landmark, w, h)
            au4_vel = extractor.action_unit_4_velocity(landmarks.landmark, w, h)

            # -----------------------------
            # Emotional flags (Phase 4.5)
            # -----------------------------
            features_for_flags = {
                "ear": ear,
                "blink": blink_signal,
                "jaw": jaw,
                "au12": au12,
                "au15": au15,
                "au4_vel": au4_vel
            }

            flags = extractor.emotional_flags(features_for_flags)

            # -----------------------------
            # Temporal normalization
            # -----------------------------
            temporal_features = {
                "ear_l": ear_l,
                "ear_r": ear_r,
                "mor": mor,
                "jaw": jaw
            }

            baseline.update(temporal_features)
            normalized = baseline.normalize(temporal_features)

            if flags and normalized:
                log_row = {
                    "ear": normalized["ear_l"],
                    "mouth": normalized["mor"],
                    "jaw": normalized["jaw"],
                    "stress": flags["stress_flag"],
                    "flat": flags["flat_affect_flag"],
                    "arousal": flags["arousal_flag"]
                }
                logger.log(log_row)

            time.sleep(0.001)

    finally:
        video.release()
        print("[camera_worker] Pipeline stopped")


# -----------------------------
# Public control functions
# -----------------------------
def start_pipeline():
    if state.pipeline_running:
        print("[camera_worker] Pipeline already running")
        return

    state.pipeline_running = True
    state.pipeline_thread = threading.Thread(
        target=_pipeline_loop,
        daemon=True
    )
    state.pipeline_thread.start()


def stop_pipeline():
    if not state.pipeline_running:
        print("[camera_worker] Pipeline not running")
        return

    state.pipeline_running = False
    state.pipeline_thread.join(timeout=2)
    state.pipeline_thread = None
