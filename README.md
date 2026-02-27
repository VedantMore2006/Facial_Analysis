# Privacy-Safe Behavioral Analysis System

A **privacy-first** real-time facial behavior analysis system that extracts high-level behavioral features without storing identifiable biometric data. The system uses baseline normalization, temporal smoothing, and event-based response tracking to provide behavioral insights suitable for mental health, user experience research, and human-computer interaction studies.

---

## 🎯 Core Specialties

### **1. Privacy-by-Design Architecture**
- ✅ **NO raw landmark coordinates stored** — only behavioral abstractions
- ✅ **NO face geometry recorded** — prevents identity reconstruction
- ✅ **NO video/image persistence** — all processing is real-time
- ✅ **Baseline-relative scaling** — each person is their own reference point
- ✅ **Clipped & rounded features** — all values normalized to [0, 1] range

**Result:** Behavioral fingerprints that reveal engagement patterns WITHOUT compromising privacy.

---

### **2. Personal Session Baseline (PSB) Normalization**
Instead of comparing individuals to population norms, the system:
- Collects **30-second baseline** at session start (neutral state)
- Computes **per-person statistics** (mean, standard deviation)
- Scales **all subsequent features** relative to this baseline
- Ensures **cross-session consistency** and individual sensitivity

**Benefit:** A naturally expressive person won't be flagged as "abnormal" — deviations are relative to their own baseline.

---

### **3. Six Behavioral Feature Extraction**

| Feature | What It Measures | Indicator Of |
|---------|------------------|--------------|
| **AU12 (Smile)** | Lip corner elevation | Positive affect, engagement, genuine emotion |
| **Expressivity** | Total facial movement | Animation, emotional engagement vs. flat affect |
| **Head Velocity** | Horizontal head rotation speed | Restlessness, scanning behavior, attention shifts |
| **Eye Contact** | Frontal gaze maintenance | Social engagement, attention, avoidance |
| **Blink Rate** | Blinks per minute (BPM) | Cognitive load, stress, anxiety, concentration |
| **Response Latency** | Time from stimulus to mouth opening | Processing speed, hesitation, cognitive effort |

---

### **4. Event-Based Response Latency Detection**
Traditional systems log every frame. This system:
- **Manual stimulus trigger** — press "s" when question ends
- **Automatic mouth opening detection** — finds response start
- **Latency calculation** — time between stimulus and response
- **Baseline-scaled latency** — normalized for individual speech patterns

**Use Case:** Interview analysis, cognitive testing, conversational AI evaluation.

---

### **5. Real-Time Processing Pipeline**
- **MediaPipe Face Mesh** — 468 landmark detection at 15 FPS
- **Temporal smoothing** — 5-frame moving average reduces noise
- **Per-frame feature computation** — raw → smoothed → scaled
- **Bounds validation** — assertion ensures all features in [0, 1]
- **CSV logging** — only scaled features stored (privacy-safe)

---

## 📁 Project Structure

```
Facial_analysis/
├── main.py                      # Entry point - runs full pipeline
├── config.py                    # All tunable parameters (FPS, baseline duration, etc.)
├── app.py                       # Streamlit dashboard for interactive visualization
├── data/
│   └── features_*.csv          # Privacy-safe feature logs (timestamped)
├── src/
│   ├── pipeline.py             # Main processing orchestrator
│   ├── camera.py               # Webcam capture & FPS management
│   ├── face_mesh.py            # MediaPipe landmark detection
│   ├── landmark_processor.py   # Extract subset of key landmarks
│   ├── baseline.py             # Personal baseline collection & normalization
│   ├── scaler.py               # Z-score scaling with sigma floor
│   ├── smoothing.py            # Moving average temporal filter
│   ├── feature_vector.py       # Clip, round, and construct final vector
│   ├── feature_logger.py       # Privacy-safe CSV writer
│   ├── logger.py               # (Optional) Raw landmark logger for debug
│   └── feature_engine/
│       ├── au12.py             # Smile intensity (lip corner distance)
│       ├── expressivity.py     # Total facial movement magnitude
│       ├── head_velocity.py    # Yaw angle velocity computation
│       ├── head_pose.py        # Head orientation estimation
│       ├── eye_contact.py      # Gaze engagement ratio
│       ├── blink.py            # Blink detection & rate calculation
│       └── response_latency.py # Event-based latency tracker
└── docs/                        # (Optional) Documentation
```

---

## 🔬 Technical Architecture

### **Data Flow:**

```
Webcam Frame (640×480)
    ↓
MediaPipe Face Mesh (468 landmarks)
    ↓
Landmark Subset Extraction (17 key points)
    ↓
Raw Feature Computation (AU12, Expressivity, Head Yaw, EAR, etc.)
    ↓
Temporal Smoothing (5-frame moving average)
    ↓
Baseline Collection (first 30 seconds)
    ↓
--- BASELINE LOCK ---
    ↓
Baseline-Relative Scaling (z-score normalization)
    ↓
Feature Vector Construction [6 values, clipped to [0,1]]
    ↓
Bounds Assertion (ensures valid range)
    ↓
Privacy-Safe CSV Logger (data/features_*.csv)
    ↓
Streamlit Dashboard Visualization (app.py)
```

---

## 🚀 Getting Started

### **Prerequisites:**
```bash
pip install opencv-python mediapipe numpy streamlit plotly pandas
```

### **1. Run Real-Time Analysis:**
```bash
python main.py
```

**Controls:**
- Press **"q"** to quit session
- Press **"s"** to trigger stimulus (for response latency)

**Output:**
- Features saved to: `data/features_{timestamp}.csv`
- Real-time video feed with landmarks & phase indicator

---

### **2. Visualize Results in Streamlit:**
```bash
streamlit run app.py
```

**Dashboard Features:**
- 📊 **Interactive Time-Series Plot** — Multi-feature overlay with baseline window
- 🔥 **Behavioral Heatmap** — Color-coded deviation matrix (blue=suppressed, red=elevated)
- 📈 **Summary Statistics** — Mean, Std for each feature
- 🎛️ **Feature Selection** — Toggle individual features on/off
- ⚙️ **Baseline Configuration** — Adjust baseline duration & FPS

**Upload:** The CSV file generated by `main.py`

---

## 🧬 Feature Modules Explained

### **AU12 (Action Unit 12 — Smile Intensity)**
- **File:** `src/feature_engine/au12.py`
- **Method:** Euclidean distance between left/right lip corners
- **Scaling:** Baseline-relative (personal baseline smile width)
- **Interpretation:** 
  - **High (>0.6):** Active smiling, positive affect
  - **Low (<0.4):** Neutral or suppressed expression

---

### **Expressivity (Facial Animation)**
- **File:** `src/feature_engine/expressivity.py`
- **Method:** Sum of per-landmark movement velocities
- **Scaling:** Baseline-relative (personal baseline animation level)
- **Interpretation:**
  - **High (>0.6):** Animated, emotionally engaged
  - **Low (<0.4):** Flat affect, emotional suppression

---

### **Head Velocity (Scanning Behavior)**
- **File:** `src/feature_engine/head_velocity.py`
- **Method:** Yaw angle change per second (°/s)
- **Scaling:** Baseline-relative (personal baseline head movement)
- **Interpretation:**
  - **High (>0.6):** Active scanning, restlessness
  - **Low (<0.4):** Head fixation, sustained attention

---

### **Eye Contact (Gaze Engagement)**
- **File:** `src/feature_engine/eye_contact.py`
- **Method:** Proportion of time with yaw angle < threshold (frontal gaze)
- **Scaling:** Baseline-relative (personal baseline gaze patterns)
- **Interpretation:**
  - **High (>0.6):** Strong social engagement
  - **Low (<0.4):** Gaze aversion, distraction

---

### **Blink Rate (Cognitive Load)**
- **File:** `src/feature_engine/blink.py`
- **Method:** Eye Aspect Ratio (EAR) drops below threshold → blink count → BPM
- **Scaling:** Baseline-relative (personal baseline blink frequency)
- **Interpretation:**
  - **High (>0.6):** Elevated stress, anxiety, fatigue
  - **Low (<0.4):** Concentration or discomfort (staring)

---

### **Response Latency (Processing Speed)**
- **File:** `src/feature_engine/response_latency.py`
- **Method:** 
  1. Press "s" key when stimulus ends
  2. Detect mouth opening (above baseline threshold)
  3. Calculate time difference
- **Scaling:** Baseline-relative (personal baseline speech onset speed)
- **Interpretation:**
  - **High (>0.6):** Quick response, low hesitation
  - **Low (<0.4):** Delayed response, processing difficulty

---

## 🔐 Privacy Guarantees

### **What IS Stored:**
✅ Scaled behavioral features (6 numbers per frame)  
✅ Timestamp (for temporal analysis)  
✅ Session metadata (FPS, baseline duration)

### **What IS NOT Stored:**
❌ Raw landmark coordinates (x, y, z)  
❌ Video frames or images  
❌ Face geometry or mesh structure  
❌ Identifiable biometric data  
❌ Reconstructable facial information

**Compliance:** Suitable for GDPR, HIPAA, and privacy-sensitive applications.

---

## 📊 Output Format

### **features_{timestamp}.csv**

| Column | Description | Range |
|--------|-------------|-------|
| `S_AU12` | Scaled smile intensity | 0.0 - 1.0 |
| `S_AUVar` | Scaled expressivity (facial animation) | 0.0 - 1.0 |
| `S_HeadVelocity` | Scaled head rotation speed | 0.0 - 1.0 |
| `S_EyeContact` | Scaled gaze engagement ratio | 0.0 - 1.0 |
| `S_BlinkRate` | Scaled blink frequency | 0.0 - 1.0 |
| `S_ResponseLatency` | Scaled response timing | 0.0 - 1.0 |

**Interpretation:**
- **0.5** = Baseline (neutral state)
- **>0.5** = Elevated relative to baseline
- **<0.5** = Suppressed relative to baseline

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
class CameraConfig:
    DEVICE_ID = 0      # Webcam selection
    WIDTH = 640        # Frame width
    HEIGHT = 480       # Frame height
    FPS = 15           # Target frame rate

class BaselineConfig:
    ENABLE_BASELINE = True
    DURATION_SECONDS = 30    # Baseline collection time
    SIGMA_FLOOR = 1e-6       # Minimum std dev

class DebugConfig:
    SHOW_LANDMARKS = True    # Draw landmarks on video
    SHOW_FPS = True          # Display FPS counter
```

---

## 🧪 Use Cases

### **1. Mental Health Monitoring**
- Track affect patterns over therapy sessions
- Detect flat affect, emotional suppression
- Monitor engagement during video consultations

### **2. User Experience Research**
- Measure engagement during product demos
- Detect confusion or frustration (elevated blink, low smile)
- Assess response confidence (latency tracking)

### **3. Interview Analysis**
- Quantify hesitation patterns (response latency)
- Detect stress indicators (blink rate, expressivity)
- Compare candidate behavioral profiles

### **4. Educational Technology**
- Monitor student attention (eye contact, head velocity)
- Detect confusion or cognitive overload (blink rate)
- Assess engagement during lessons (expressivity, smile)

### **5. Conversational AI Evaluation**
- Measure user frustration during chatbot interactions
- Detect when explanation is insufficient (scanning behavior)
- Optimize response timing based on latency data

---

## 🛡️ Ethical Considerations

### **Informed Consent:**
Always obtain explicit consent before recording or analyzing facial behavior.

### **Purpose Limitation:**
Use data only for stated purposes (e.g., research, UX testing).

### **Data Minimization:**
This system already implements privacy-by-design — no excess data is collected.

### **Transparency:**
Inform users that behavioral patterns are being analyzed, and explain what features are extracted.

### **Bias Mitigation:**
Baseline normalization reduces cross-individual bias, but always validate on diverse populations.

---

## 🔧 Extending the System

### **Add New Features:**
1. Create feature engine in `src/feature_engine/your_feature.py`
2. Implement computation logic (raw value from landmarks)
3. Add to pipeline in `src/pipeline.py`
4. Update feature vector in `src/feature_vector.py`
5. Update Streamlit dashboard columns

### **Custom Baselines:**
- Modify `src/baseline.py` to persist baselines across sessions
- Store baseline stats in database for longitudinal studies

### **Real-Time Alerts:**
- Add thresholds in pipeline for extreme deviations
- Trigger alerts when multiple features exceed limits

---

## 📚 References

- **MediaPipe Face Mesh:** [Google MediaPipe](https://mediapipe.dev/)
- **Action Units (FACS):** Ekman & Friesen facial coding system
- **Eye Aspect Ratio:** Soukupová & Čech (2016)
- **Privacy-Preserving CV:** Behavioral abstraction techniques

---

## 📝 License

This project is for research and educational purposes. Ensure compliance with local privacy regulations when deploying.

---

## 🤝 Contributing

Contributions are welcome! Focus areas:
- Additional behavioral features (e.g., gaze tracking, micro-expressions)
- Cross-session baseline persistence
- Real-time alert system
- Multi-face tracking

---

## 📧 Contact

For questions, issues, or collaboration inquiries, please open a GitHub issue or contact the project maintainer.

---

**Built with privacy, powered by behavior analysis.** 🔐📊

