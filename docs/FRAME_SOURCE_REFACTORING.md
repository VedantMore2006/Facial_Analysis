# Frame Source Refactoring Summary

## ✅ Refactoring Complete

The project has been successfully refactored to support both live webcam and offline video file input without modifying any feature computation, baseline, or scaling logic.

---

## 📁 Files Created

### `/home/vedant/Facial_analysis/src/frame_source.py`
**New abstraction layer for frame input sources**

#### Classes:
1. **`FrameSource` (Abstract Base Class)**
   - `read() -> (ret, frame)` - Get next frame
   - `release()` - Free resources
   - `get_fps()` - Get frames per second
   - `is_realtime()` - Indicate if source is live or recorded

2. **`CameraSource` (Live Webcam)**
   - Wraps existing Camera logic
   - Uses OpenCV VideoCapture with device ID
   - Configurable via CameraConfig
   - Returns `is_realtime() = True`

3. **`VideoFileSource` (Offline Video)**
   - Accepts video file path
   - Extracts FPS from video metadata
   - Provides deterministic playback
   - Returns `is_realtime() = False`
   - Displays video info on initialization (FPS, frame count, duration)

---

## 📝 Files Modified

### `/home/vedant/Facial_analysis/main.py`
**Changes:**
- Added `argparse` for command-line argument parsing
- Added `--video` optional parameter for video file path
- Created `parse_arguments()` function
- Created `create_frame_source()` function to instantiate appropriate source
- Updated `run_pipeline()` call to pass frame_source parameter

**Usage:**
```bash
# Live webcam (default)
python main.py

# Offline video file
python main.py --video recordings/session1.mp4
```

---

### `/home/vedant/Facial_analysis/src/pipeline.py`
**Changes:**
1. **Removed:** `from src.camera import Camera`
2. **Updated:** `run_pipeline()` now accepts `frame_source` parameter
3. **Replaced:** `cam = Camera()` with injected `frame_source`
4. **Added:** `is_realtime = frame_source.is_realtime()` flag
5. **Updated:** FPS retrieval to use `frame_source.get_fps()`
6. **Updated:** Frame reading to use `frame_source.read()`
7. **Updated:** Cleanup to use `frame_source.release()`

**Timestamp Logic (Critical Change):**
```python
# Calculate timestamp based on source type
if is_realtime:
    # Realtime source: use actual elapsed time
    elapsed_time = current_time - session_start
    timestamp_ms = int(elapsed_time * 1000)
else:
    # Recorded source: use deterministic frame-based time
    timestamp_ms = int(frame_index * (1000 / fps))
```

**Reporting Enhanced:**
- Shows "Source FPS" instead of "Detected FPS"
- Shows "Total frames processed"

---

## 🔒 Files Untouched (As Required)

### Feature Engines
- ✅ `src/feature_engine/au12.py` - NOT MODIFIED
- ✅ `src/feature_engine/blink.py` - NOT MODIFIED
- ✅ `src/feature_engine/expressivity.py` - NOT MODIFIED
- ✅ `src/feature_engine/eye_contact.py` - NOT MODIFIED
- ✅ `src/feature_engine/head_velocity.py` - NOT MODIFIED
- ✅ `src/feature_engine/response_latency.py` - NOT MODIFIED

### Core Logic
- ✅ `src/baseline.py` - NOT MODIFIED
- ✅ `src/scaler.py` - NOT MODIFIED
- ✅ `src/smoothing.py` - NOT MODIFIED
- ✅ `src/feature_logger.py` - NOT MODIFIED
- ✅ `src/face_mesh.py` - NOT MODIFIED
- ✅ `config.py` - NOT MODIFIED

### Legacy Support
- ✅ `src/camera.py` - Still intact (can be used independently if needed)

---

## 🎯 Architecture Benefits

### 1. **Abstraction Layer**
   - Pipeline is now source-agnostic
   - Easy to add new sources (RTSP streams, image sequences, etc.)
   - Clean separation of concerns

### 2. **Strategy Pattern**
   - Runtime selection of frame source
   - No code duplication or branching in pipeline
   - Polymorphic behavior through interface

### 3. **Deterministic Timestamps**
   - Video files use frame-based time calculation
   - Ensures reproducible analysis of recorded sessions
   - Live webcam still uses real-time timestamps

### 4. **Minimal Changes**
   - Only 3 files modified
   - Feature computation logic untouched
   - Pipeline logic preserved (only frame input changed)

---

## 🧪 Validation

### Syntax Check
```bash
python -m py_compile src/frame_source.py src/pipeline.py main.py
```
✅ **Result:** All files compile successfully

### Help Text
```bash
python main.py --help
```
Shows proper usage instructions with examples

---

## 📊 Behavior Comparison

| Aspect | Live Webcam | Offline Video |
|--------|-------------|---------------|
| **Command** | `python main.py` | `python main.py --video file.mp4` |
| **Frame Source** | CameraSource | VideoFileSource |
| **Timestamp** | Real-time (elapsed) | Frame-based (deterministic) |
| **FPS** | Camera-detected | Video metadata |
| **Stop Condition** | User presses 'q' | User presses 'q' OR video ends |
| **Feature Logic** | ✅ Same | ✅ Same |
| **Baseline Logic** | ✅ Same | ✅ Same |
| **Output Format** | ✅ Same | ✅ Same |

---

## 🚀 Future Extensibility

The abstraction makes it easy to add new frame sources:

```python
class RTSPStreamSource(FrameSource):
    """Network camera stream"""
    def __init__(self, rtsp_url):
        self.cap = cv2.VideoCapture(rtsp_url)
    # ... implement interface methods

class ImageSequenceSource(FrameSource):
    """Directory of images"""
    def __init__(self, image_dir):
        self.images = sorted(glob.glob(f"{image_dir}/*.jpg"))
    # ... implement interface methods
```

Simply add a new class and update `main.py` argument parsing.

---

## ✅ Requirements Checklist

- [x] Create FrameSource abstraction with `read()`, `release()`, `get_fps()`
- [x] Implement CameraSource wrapping existing Camera logic
- [x] Implement VideoFileSource for offline video files
- [x] Add `is_realtime()` method to distinguish source types
- [x] Use argparse in main.py for `--video` parameter
- [x] Update pipeline.py to accept injected frame source
- [x] Deterministic timestamps for video files: `frame_index * (1000 / fps)`
- [x] Real-time timestamps for camera: `elapsed_time * 1000`
- [x] NO modifications to feature engines
- [x] NO modifications to baseline.py
- [x] NO modifications to scaler.py
- [x] NO modifications to smoothing.py
- [x] Minimal pipeline.py changes (only frame input section)
- [x] Modular and clean architecture

---

## 🎉 Result

**Clean abstraction layer implemented with zero impact on feature computation logic.**

The pipeline now supports both live and offline analysis while maintaining identical feature extraction behavior.
