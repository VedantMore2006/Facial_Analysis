# Facial Analysis Project Synopsis

**Generated on:** March 6, 2026  
**Workspace:** `/home/vedant/Facial_analysis`  
**Scope:** Complete technical synopsis of codebase, ML pipeline, outputs, and conversation-driven development changes.

---

## 1) Executive Summary

This project is a webcam-based behavioral feature extraction and classification system that:

1. Detects facial landmarks using MediaPipe.
2. Computes frame-level behavioral features (34 core features, then expanded to 136 model features via window aggregations).
3. Logs both raw and scaled CSVs per recording session.
4. Trains an XGBoost multi-class model for 6 mental-health-related behavioral classes.
5. Supports both live inference and offline CSV session analysis.

The workflow has evolved from a purely synthetic-data training pipeline to a **hybrid strategy** (real acted pattern extraction + synthetic generation), and now includes an **offline latest-session analyzer** to reduce live inference instability/noise.

---

## 2) Current High-Level Architecture

### A. Data Capture & Feature Pipeline
- **Entry point:** `run_pipeline.py`
- **Landmark detector:** `landmarks/mediapipe_detector.py`
- **Landmark subset extraction:** `landmarks/landmark_subset.py`
- **Feature computations:** `features/*.py` via `core/feature_registry.py`
- **Temporal buffering:** `core/frame_buffer.py`
- **Smoothing:** `processing/ema_smoothing.py`
- **Baseline and normalization:**
  - `processing/baseline.py`
  - `processing/feature_normalization.py`
- **Head pose utilities:** `processing/head_pose.py`
- **Session logging:** `output/csv_logger.py`

### B. ML Pipeline
- **End-to-end orchestrator:** `ml/run_full_pipeline.py`
- **Window aggregation:** `ml/window_aggregator.py`
- **Baseline distribution modeling:** `ml/baseline_stats.py`
- **Class behavior profiles:** `ml/disorder_profiles.py`
- **Synthetic/hybrid generation:** `ml/synthetic_generator.py`
- **Training:** `ml/train_model.py`
- **Evaluation/reporting:** `ml/evaluate_model.py`

### C. Inference Utilities
- **Live webcam inference:** `test_model_live.py`
- **Offline latest-session inference:** `analyze_latest_session.py` (newly added)

---

## 3) Project File Inventory (Functional)

### Root scripts
- `run_pipeline.py` — real-time feature extraction and CSV logging.
- `test_model_live.py` — webcam classification using trained model.
- `analyze_latest_session.py` — analyze newest recorded CSV without live camera.
- `analyze_raw_features.py`, `detailed_csv_samples.py` — feature/data diagnostics.
- `test_mediapipe.py` — detector validation.

### Core system
- `core/feature_registry.py` — registers feature objects.
- `core/frame_buffer.py` — temporal frame/history storage.
- `core/pipeline.py` — present but currently empty.

### Feature modules
- `features/facial_au_features.py`
- `features/eye_features.py`
- `features/head_motion_features.py`
- `features/derived_features.py`
- `features/temporal_features.py`
- `features/base_feature.py`

### Landmark and geometry stack
- `landmarks/mediapipe_detector.py`
- `landmarks/landmark_subset.py`
- `processing/geometry.py`
- `processing/head_pose.py`

### Processing
- `processing/baseline.py`
- `processing/ema_smoothing.py`
- `processing/feature_normalization.py`

### Output and visualization
- `output/csv_logger.py`
- `feature_plots/plot_features.py`

### ML package
- `ml/disorder_profiles.py`
- `ml/synthetic_generator.py`
- `ml/train_model.py`
- `ml/evaluate_model.py`
- `ml/window_aggregator.py`
- `ml/baseline_stats.py`
- `ml/record_acted_session.py`
- `ml/run_full_pipeline.py`

### Documentation package
- `ml/ML_TRAINING.md`
- `ml/DATA_SHARING_PACKAGE.md`
- `ml/SHARING_QUICK_REFERENCE.md`
- `ml/README.md`
- `CSV_INTERPRETATION_GUIDE.md`
- `FEATURE_INSIGHTS.md`
- `README_CSV_ANALYSIS.md`
- `SCALING_FIXES.md`

---

## 4) Data Model and Feature Semantics

### Recording outputs
Each session writes two CSV streams with matching timestamps:
- `output/raw/*.csv` — raw feature values.
- `output/scaled/*.csv` — normalized/scaled values prefixed by `S_`.

### Feature levels
1. **Frame-level features**: 34 behavioral features per frame (AU/eye/head/temporal/derived).
2. **Window-level features**: 136 model features generated using aggregation suffixes:
   - `_mean`, `_std`, `_max`, `_min`

### Typical window settings
- Frame rate: ~30 FPS
- Window duration: 5 seconds
- Frames per window: 150
- Optional stride for repeated windows: 30 frames (in offline analyzer)

---

## 5) Classification Taxonomy (Finalized)

From `ml/disorder_profiles.py` and `ml/model_metadata.json`:

- **0** → Depression
- **1** → Anxiety
- **2** → Stress
- **3** → Bipolar Mania
- **4** → Phobia Common
- **5** → Suicidal Tendency

Important training-label behavior:
- Labels are explicitly written into dataset rows (`label` column).
- Dataset rows are shuffled after generation for randomized training order.
- Model learns from labels in CSV, not row position semantics.

---

## 6) End-to-End Operational Workflow

### Stage 1: Capture baseline/session features
```bash
python run_pipeline.py
```
- Captures webcam frames.
- Detects landmarks.
- Computes/smooths features.
- Logs raw + scaled CSVs.

### Stage 2: Build ML artifacts
```bash
python ml/run_full_pipeline.py
```
Pipeline sequence:
1. Window aggregation.
2. Baseline statistics.
3. Synthetic sample generation.
4. XGBoost training.
5. Evaluation report generation.

### Stage 3: Inference options
- **Live:** `python test_model_live.py`
- **Offline latest CSV:** `python analyze_latest_session.py`

---

## 7) Conversation & Development History (Session Synopsis)

This section records the practical evolution and fixes implemented during the collaboration.

### Phase A — Training strategy expansion
- User requested detailed workflow for **Hybrid Path 2** (real acted sessions + synthetic generation).
- Comprehensive instructions were provided for:
  - baseline handling,
  - acted session recording for each class,
  - pattern extraction,
  - hybrid dataset generation,
  - retraining and evaluation.

### Phase B — Import path reliability fixes
- Initial execution errors occurred due to module resolution.
- Import paths were updated in ML scripts to ensure package-aware execution (using `ml.` paths where needed).

### Phase C — Label assignment clarification
- User asked how model knows class blocks (e.g., “first 6000 depression”).
- Clarified full chain:
  - label map in `DISORDER_LABELS`,
  - labels embedded in training CSV rows,
  - training consumes `df['label']`,
  - rows are shuffled before split/training.

### Phase D — Live model testing implementation
- Added `test_model_live.py` for real-time classification from webcam.
- Solved multiple runtime blockers:
  - OpenCV module missing (`cv2`) → package installation guidance.
  - Detector API mismatch (`process` vs `detect`) → corrected call path.
  - Head pose tuple unpacking mismatch → fixed `yaw, pitch, roll` handling.

### Phase E — Landmark and frame acceptance debugging
- Initially buffer stayed `0/150` due to frame rejection/pose instability.
- Added diagnostics and lowered detection thresholds.
- Temporarily disabled head-pose rejection in live script to allow buffer fill.
- Result: landmarks started detecting and buffer progressed (`29/150`, `59/150`, …).

### Phase F — UI/font and OpenCV constant issues
- Qt warnings observed (missing fonts/style override), non-fatal.
- Runtime crash fixed:
  - `cv2.FONT_HERSHEY_BOLD` (invalid in OpenCV Python build)
  - replaced with `cv2.FONT_HERSHEY_SIMPLEX`.

### Phase G — Shift to offline session-level evaluation
- User proposed avoiding live-only inference.
- Implemented `analyze_latest_session.py`:
  - auto-load latest CSV (`output/scaled` then `output/raw`),
  - build model-aligned window features,
  - classify each window,
  - summarize dominant class by vote + average probabilities.

---

## 8) Current Runtime Status

### Working
- Model loading works.
- Landmark detection works in live mode.
- Buffer filling works after disabling pose rejection.
- Offline analyzer script is present and syntax-valid.

### Known warnings/non-critical noise
- Qt style warning: invalid style override `kvantum` (falls back to available styles).
- OpenCV Qt font directory warning: missing `cv2/qt/fonts` path (non-fatal for core logic).

### Known technical debt
- `test_model_live.py` showed Pandas fragmentation warnings due to per-column insertion when adding missing features; can be optimized with a single `reindex(..., fill_value=0.0)` approach.
- Head-pose rejection currently intentionally bypassed for stability; pose estimator calibration remains pending.

---

## 9) Model and Metadata Snapshot

From `ml/model_metadata.json`:
- **Classes:** 6
- **Model objective:** `multi:softprob`
- **Primary algorithm:** XGBoost
- **Training feature count:** 136
- **Window features naming:** `<base_feature>_<mean|std|max|min>`
- **Label map:** aligned to six target conditions.

---

## 10) Key Documentation Summary

- `ml/ML_TRAINING.md`: full deep-dive on synthetic/hybrid training methodology.
- `ml/DATA_SHARING_PACKAGE.md`: packaging/sharing specification and feature catalog.
- `ml/SHARING_QUICK_REFERENCE.md`: practical handoff checklist.
- `CSV_INTERPRETATION_GUIDE.md` + `FEATURE_INSIGHTS.md`: per-feature behavior interpretation, expected zeros, correlation patterns, temporal behavior notes.

---

## 11) Recommended Next Steps

1. **Stabilize inference path**
   - Prefer `analyze_latest_session.py` for primary decisioning.
   - Use live inference only for interactive feedback.

2. **Refine pose logic**
   - Recalibrate `estimate_head_pose` thresholds/orientation.
   - Re-enable frame rejection only after validation on your camera setup.

3. **Optimize live feature assembly**
   - Replace repeated column insertions with a vectorized DataFrame alignment strategy.

4. **Improve robustness reporting**
   - Save offline analysis summaries (`JSON/CSV`) for trend tracking across sessions.

5. **Hybrid data maturation**
   - Continue collecting acted sessions per class and regenerate hybrid dataset periodically.

---

## 12) Quick Command Reference

```bash
# 1) Capture new session CSVs
python run_pipeline.py

# 2) Train/update model
python ml/run_full_pipeline.py

# 3) Live webcam inference
python test_model_live.py

# 4) Offline latest session inference
python analyze_latest_session.py

# 5) Offline inference for a specific file
python analyze_latest_session.py --csv output/scaled/<session_file>.csv
```

---

## 13) Notes on Scope and Safety

- This system detects **behavioral patterns** associated with class profiles in training data.
- Outputs are model classifications, **not clinical diagnosis**.
- Use as a research/decision-support signal with human oversight and contextual interpretation.

---

## 14) Changelog of Recent Additions

- Added: `test_model_live.py` (real-time classification utility).
- Updated: live script fixes for detector API, head pose handling, OpenCV font constant compatibility.
- Added: `analyze_latest_session.py` (offline latest-CSV session evaluator).
- Added docs in `ml/` package for training and data sharing workflows.

---

## 15) Closing Snapshot

The codebase now supports a complete loop:

**Capture → Feature Engineering → Dataset Generation → Model Training → Live/Offline Inference**,

with practical stability improvements and a fallback offline analysis path that directly addresses session-level prediction consistency concerns.
