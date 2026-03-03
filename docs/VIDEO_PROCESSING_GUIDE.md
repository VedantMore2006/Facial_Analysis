# Quick Reference: Video File Processing

## 🎥 Using Video Files Instead of Webcam

### Basic Usage

#### Live Webcam (Default)
```bash
python main.py
```

#### Offline Video File
```bash
python main.py --video path/to/video.mp4
```

---

## 📋 Example Workflows

### 1. Process a Recorded Session
```bash
# Analyze a previously recorded video
python main.py --video recordings/interview_2024.mp4
```

### 2. Batch Processing Multiple Videos
```bash
# Process multiple videos in sequence
for video in recordings/*.mp4; do
    python main.py --video "$video"
done
```

### 3. Test Pipeline on Sample Video
```bash
# Download sample video
wget https://example.com/sample.mp4 -O test_video.mp4

# Run analysis
python main.py --video test_video.mp4
```

---

## 🔍 What's Different with Video Files?

### Timestamps
- **Webcam**: Uses real-time clock (actual elapsed seconds)
- **Video File**: Uses frame-based time (frame_index / fps)
  ```python
  timestamp_ms = int(frame_index * (1000 / fps))
  ```

### FPS
- **Webcam**: Detected from camera hardware
- **Video File**: Read from video file metadata

### Session End
- **Webcam**: User presses 'q' to quit
- **Video File**: Ends when video finishes OR user presses 'q'

### Reproducibility
- **Webcam**: Timestamps vary based on system load
- **Video File**: Timestamps are deterministic (same video = same timestamps)

---

## ⚡ Performance Notes

### Video File Advantages
- **Reproducible**: Same input always produces same timestamps
- **No camera required**: Can analyze videos offline
- **Batch processing**: Process multiple recordings automatically
- **Validation**: Verify pipeline behavior on known inputs

### Video File Considerations
- FPS is fixed (can't change during processing)
- Timestamps are frame-based (not wall-clock time)
- All features computed identically to live mode

---

## 📊 Output Files

Both modes produce identical output structure:

### Production Features
```
data/features_TIMESTAMP.csv
```
Contains scaled features (S_AU12, S_AUVar, etc.)

### Validation Raw Values
```
data/validation_raw_session_TIMESTAMP.csv
```
Contains unprocessed raw values:
- frame_index
- timestamp_ms
- au12_raw
- expressivity_raw
- head_velocity_raw
- blink_rate_raw
- ear_raw
- yaw_raw

---

## 🛠️ Technical Details

### Video File Requirements
- **Format**: Any format supported by OpenCV (MP4, AVI, MOV, MKV, etc.)
- **Codec**: H.264, MPEG-4, etc.
- **Resolution**: Any (will be processed as-is)
- **FPS**: Extracted automatically from metadata

### Supported Video Formats
- ✅ `.mp4` (most common)
- ✅ `.avi`
- ✅ `.mov`
- ✅ `.mkv`
- ✅ `.webm`
- ✅ `.flv`

### Validation
If video file fails to open, you'll see:
```
ValueError: Failed to open video file: path/to/video.mp4
```

---

## 🎯 Use Cases

### Research & Development
- Test feature computation on controlled inputs
- Compare different baseline durations
- Validate pipeline changes

### Data Analysis
- Process archived interview recordings
- Analyze historical video data
- Generate features for ML training datasets

### Quality Assurance
- Regression testing with known videos
- Verify consistent feature extraction
- Benchmark performance

---

## 💡 Pro Tips

### 1. Check Video Info
When processing starts, you'll see:
```
📹 Video file loaded: recordings/session1.mp4
   FPS: 30.0
   Total frames: 9000
   Duration: 300.00 seconds
```

### 2. Verify Timestamps
Check `validation_raw_session_*.csv` to verify timestamps:
- Should increment by `1000/fps` milliseconds per frame
- Example: 30 FPS → ~33.33ms per frame

### 3. Baseline Collection
- Video files still use baseline collection (first 30 seconds)
- Ensure video is long enough for baseline + analysis
- Minimum recommended: 60 seconds

---

## 🚨 Common Issues

### Issue: "Failed to open video file"
**Solutions:**
- Verify file path is correct
- Check file permissions
- Ensure video format is supported by OpenCV
- Try converting to MP4: `ffmpeg -i input.mov output.mp4`

### Issue: Video plays too fast/slow
**Note:** Pipeline processes frame-by-frame as fast as possible. This is normal for offline processing. Display window shows current frame regardless of playback speed.

### Issue: FPS seems wrong
**Solution:** Check video metadata:
```bash
ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 video.mp4
```

---

## 📚 Related Documentation

- [Frame Source Refactoring Details](FRAME_SOURCE_REFACTORING.md)
- [Feature Computation Guide](../README.md)
- [Configuration Reference](../config.py)

---

## ✅ Summary

**Video file mode provides identical feature extraction to live webcam mode, with deterministic timestamps for reproducibility.**

Use `python main.py --video file.mp4` to process any video file through the facial analysis pipeline.
