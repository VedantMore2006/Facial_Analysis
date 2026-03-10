# Landmark Preview Visualization

This tool visualizes the 29 facial landmarks used in the facial analysis system.

## Quick Start

### Option 1: Webcam Mode (Default - Points Only)
```bash
cd landmarks
python landmarks_preview.py
```

**Webcam Controls:**
- `q` - Quit the application
- `l` - Toggle landmark labels ON/OFF
- `c` - Toggle connections between landmarks
- `s` - Save screenshot

### Option 2: Show Landmark Numbers
```bash
python landmarks_preview.py --labels
```

### Option 3: Static Image Mode
```bash
python landmarks_preview.py --image path/to/image.jpg --output result.png
```

## Command-Line Options

```bash
python landmarks_preview.py [OPTIONS]

Options:
  --image, -i PATH      Process a static image instead of webcam
  --output, -o PATH     Output filename for image mode (default: landmark_preview_output.png)
  --labels, -l          Show landmark index numbers on the visualization
  --connections, -c     Show lines connecting nearby landmarks
  --no-legend          Hide the color legend
  --quiet, -q          Suppress landmark information output
  -h, --help           Show help message
```

## Examples

**Webcam with labels:**
```bash
python landmarks_preview.py --labels
```

**Image processing with labels and connections:**
```bash
python landmarks_preview.py -i photo.jpg -o annotated.png --labels --connections
```

**Clean view (points only, no legend):**
```bash
python landmarks_preview.py --no-legend
```

## What You'll See

The visualization shows:
- **32 selected landmarks** out of MediaPipe's 478 total landmarks
- **Color-coded regions**:
  - 🔵 Light Blue: Face Contour (7 landmarks)
  - 🟢 Green: Eyes (8 landmarks)
  - 🟡 Bright Yellow: Right Eyebrow End (2 landmarks)
  - 🟡 Yellow: Eye Region (13 landmarks)
  - 🔴 Red: Mouth (2 landmarks)
- **Numbered labels** (optional - use `--labels`)
- **Legend** showing landmark count and region breakdown

## Landmark Subset Details

**Selected Indices:**
```
1, 2, 13, 14, 33, 50, 61, 63, 70, 78, 95, 
133, 145, 152, 159, 234, 263, 285, 291, 296, 300,
308, 324, 334, 336, 362, 374, 386, 454, 468, 472
```

These landmarks were selected to capture:
- Head pose and orientation
- Eye movements and blinks
- Right eyebrow end movements
- Facial action units (AUs)
- Mouth movements

## For Your Mentor

This visualization demonstrates which landmarks are being tracked for the facial analysis pipeline. The system uses a reduced set of 32 landmarks (vs. 478) for:
- Performance optimization
- Focus on relevant facial features
- Minimal eyebrow tracking (2 points at right eyebrow end)
- Sufficient coverage of key facial regions

The landmarks are extracted in real-time at ~30 FPS and used to compute:
- Eye aspect ratios
- Head pose angles
- Eyebrow movements
- Facial action units
- Temporal features for mental state analysis
