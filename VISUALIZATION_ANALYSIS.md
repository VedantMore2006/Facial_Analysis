# COMPLETE VISUALIZATION ANALYSIS FOR FACIAL ANALYSIS PROJECT

## 📊 ALL VISUALIZATIONS IN THE PROJECT

---

## **GROUP A: INDIVIDUAL SIGNAL PLOTS** (6 plots via `plot_signal()`)

### **1. AU12 Signal Plot** 
**File:** `au12_signal_{timestamp}.png`  
**Location:** [pipeline.py](src/pipeline.py#L333-L336)  
**Method:** `plot_signal(au12_raw_list, au12_smooth_list, au12_scaled_list, ...)`

**What it shows:**
- 3 lines: Raw (faint), Smoothed (medium), Scaled (bold)
- Gray baseline window overlay
- X-axis: Frame number | Y-axis: Value

**What it indicates:**
- **Raw**: Unprocessed smile intensity (lip corner distance)
- **Smoothed**: Noise-reduced smile signal
- **Scaled**: Normalized smile relative to baseline (0=suppressed, 1=elevated)
- Shows engagement/positive affect patterns over time

---

### **2. Expressivity Signal Plot**
**File:** `expressivity_signal_{timestamp}.png`  
**Location:** [pipeline.py](src/pipeline.py#L339-L342)  
**Method:** `plot_signal(expressivity_raw_list, expressivity_smooth_list, expressivity_scaled_list, ...)`

**What it shows:**
- Raw, smoothed, scaled expressivity over time
- Baseline window marked

**What it indicates:**
- **Raw**: Total facial movement magnitude (sum of all landmark movements)
- **Smoothed**: Filtered expressivity
- **Scaled**: Normalized facial animation intensity
- High values = animated/expressive, Low values = flat/suppressed affect
- Useful for detecting emotional engagement vs. withdrawal

---

### **3. Head Velocity Signal Plot**
**File:** `head_velocity_signal_{timestamp}.png`  
**Location:** [pipeline.py](src/pipeline.py#L345-L348)  
**Method:** `plot_signal(head_raw_list, head_smooth_list, head_scaled_list, ...)`

**What it shows:**
- Raw, smoothed, scaled head yaw velocity

**What it indicates:**
- **Raw**: Speed of horizontal head rotation (degrees/sec)
- **Smoothed**: Filtered head movement
- **Scaled**: Normalized head movement relative to baseline
- High values = active scanning/searching, Low values = head fixation
- Can indicate restlessness, avoidance, or engagement shifts

---

### **4. Blink Rate Signal Plot**
**File:** `blink_rate_signal_{timestamp}.png`  
**Location:** [pipeline.py](src/pipeline.py#L351-L354)  
**Method:** `plot_signal(blink_raw_list, blink_raw_list, blink_scaled_list, ...)`  
**NOTE:** Raw used twice (no separate smoothed version)

**What it shows:**
- Blink rate in blinks-per-minute (BPM)
- Scaled version normalized to baseline

**What it indicates:**
- **Raw**: Instantaneous blink frequency
- **Scaled**: Baseline-adjusted blink rate
- High values = stress/anxiety/cognitive load
- Low values = concentration or discomfort (staring)
- Spikes can indicate processing difficulty or emotional response

---

### **5. EAR (Eye Aspect Ratio) Signal Plot**
**File:** `ear_signal_{timestamp}.png`  
**Location:** [pipeline.py](src/pipeline.py#L357-L368)  
**Method:** Direct matplotlib plot (not using `plot_signal()`)

**What it shows:**
- Single line: EAR value over time
- Gray baseline window
- NO smoothing, NO scaling

**What it indicates:**
- **EAR**: Eye openness ratio (geometry-based)
- High = eyes wide open
- Low = eyes closing/closed
- Used internally to detect blinks
- Shows fatigue patterns, alertness, blink events
- Raw diagnostic signal (not a behavioral feature)

---

### **6. Eye Contact Signal Plot**
**File:** `eye_contact_signal_{timestamp}.png`  
**Location:** [pipeline.py](src/pipeline.py#L371-L374)  
**Method:** `plot_signal(eye_raw_list, eye_raw_list, eye_scaled_list, ...)`  
**NOTE:** Raw used twice (no smoothed version)

**What it shows:**
- Eye contact ratio over time
- Scaled version normalized

**What it indicates:**
- **Raw**: Proportion of time maintaining frontal gaze (based on yaw angle)
- **Scaled**: Baseline-adjusted eye contact ratio
- High = sustained attention/engagement
- Low = gaze aversion/distraction
- Critical for assessing social engagement and attention

---

## **GROUP B: AGGREGATE VISUALIZATIONS**

### **7. Behavioral Deviation Heatmap (Standalone)**
**File:** `behavioral_heatmap_{timestamp}.png`  
**Location:** [pipeline.py](src/pipeline.py#L377-L397)  
**Method:** `plot_heatmap(feature_matrix, feature_names, baseline_frames)`

**What it shows:**
- 2D color matrix (14x6 inches)
- Rows = 6 features (AU12, Expressivity, Head Velocity, Eye Contact, Blink Rate, Response Latency)
- Columns = frames
- Color: Blue (0) = suppressed, Red (1) = elevated
- White dashed line marks baseline end

**What it indicates:**
- **Visual behavioral fingerprint** of entire session
- See which features co-vary over time
- Identify stress periods (multiple red rows)
- Identify withdrawal periods (multiple blue rows)
- Compare baseline (left) vs. stimulus (right)
- Pattern recognition: e.g., high blink + low smile = stress

---

### **8. Console Summary Statistics** (NOT a plot)
**File:** None (console output)  
**Location:** [pipeline.py](src/pipeline.py#L400-L409)  
**Method:** `print_summary_statistics(feature_names, feature_lists)`

**What it shows:**
- Text table in console
- Columns: Feature, Mean, Std, Min, Max
- 6 rows (one per feature)

**What it indicates:**
- Aggregate session metrics
- Mean shows overall deviation from baseline
- Std shows variability/consistency
- Min/Max show range of behavioral expression
- Used for quantitative reporting

---

### **9. Behavioral Dashboard (UNIFIED VIEW)**
**File:** `behavioral_dashboard_{timestamp}.png`  
**Location:** [pipeline.py](src/pipeline.py#L412-L437)  
**Method:** `create_dashboard(feature_dict_lists, dashboard_names, baseline_frames)`

**What it shows:**
- **Single 14x10 inch figure with 3 panels:**

**Panel 1 (Top, full width): Time-Series Plot**
- All 6 scaled features on same axes
- Color-coded lines
- Gray baseline window
- Grid overlay

**Panel 2 (Bottom-left): Heatmap**
- Same as standalone heatmap (#7)
- 6 features × frames
- Coolwarm colormap

**Panel 3 (Bottom-right): Statistics Table**
- Feature names, Mean, Std
- Green header styling
- Embedded in plot

**What it indicates:**
- **ONE-STOP VISUALIZATION** for entire session
- Compare features side-by-side in time-series
- See heatmap patterns
- Read summary stats
- Perfect for presentations/reports
- Eliminates need to open multiple plots

---

## **GROUP C: LEGACY/UNUSED**

### **10. plot_utils.py (OBSOLETE)**
**File:** [src/plot_utils.py](src/plot_utils.py)  
**Method:** Old `plot_signal()` without baseline support

**Status:** 
- ❌ **NOT USED** anywhere in pipeline
- Replaced by `visualization.plot_signal()` which has baseline support
- **SHOULD BE DELETED**

---

## 📋 SUMMARY TABLE

| # | Visualization | File Saved | Function | Purpose | Keep/Delete |
|---|---------------|------------|----------|---------|-------------|
| 1 | AU12 Signal | `au12_signal_{timestamp}.png` | Individual feature detail | Smile intensity tracking | **KEEP** or merge into dashboard |
| 2 | Expressivity Signal | `expressivity_signal_{timestamp}.png` | Individual feature detail | Facial animation tracking | **KEEP** or merge into dashboard |
| 3 | Head Velocity Signal | `head_velocity_signal_{timestamp}.png` | Individual feature detail | Head movement tracking | **KEEP** or merge into dashboard |
| 4 | Blink Rate Signal | `blink_rate_signal_{timestamp}.png` | Individual feature detail | Blink frequency tracking | **KEEP** or merge into dashboard |
| 5 | EAR Signal | `ear_signal_{timestamp}.png` | Diagnostic signal | Low-level eye openness | **OPTIONAL** (debug only) |
| 6 | Eye Contact Signal | `eye_contact_signal_{timestamp}.png` | Individual feature detail | Gaze engagement tracking | **KEEP** or merge into dashboard |
| 7 | Standalone Heatmap | `behavioral_heatmap_{timestamp}.png` | Aggregate view | Visual behavioral fingerprint | **DELETE** (redundant with dashboard) |
| 8 | Console Statistics | (none) | Text output | Quick summary | **KEEP** (useful for logs) |
| 9 | **Behavioral Dashboard** | `behavioral_dashboard_{timestamp}.png` | **Unified view** | **All-in-one presentation** | **✅ KEEP (PRIMARY)** |
| 10 | plot_utils.py | (none) | Obsolete function | None | **❌ DELETE** |

---

## 🎯 RECOMMENDATIONS

### **Option A: Minimal (Production)**
**Keep only:**
- ✅ Behavioral Dashboard (#9) — shows everything
- ✅ Console Statistics (#8) — for logging
- ✅ Feature Logger CSV — for data analysis

**Delete:**
- ❌ All individual signal plots (1-6) — redundant with dashboard
- ❌ Standalone heatmap (#7) — redundant with dashboard
- ❌ EAR plot (#5) — debug signal only
- ❌ plot_utils.py (#10) — obsolete

**Result:** 1 plot file + 1 CSV file per session (clean, professional)

---

### **Option B: Detailed (Research/Debug)**
**Keep:**
- ✅ Individual plots (1-6) — for deep-dive analysis per feature
- ✅ Behavioral Dashboard (#9) — for overview
- ✅ Console Statistics (#8)
- ✅ Feature Logger CSV

**Delete:**
- ❌ Standalone heatmap (#7) — redundant with dashboard
- ❌ plot_utils.py (#10) — obsolete

**Result:** 7 plot files + 1 CSV file per session (comprehensive)

---

### **Option C: Hybrid (Recommended)**
**Keep:**
- ✅ Behavioral Dashboard (#9) — PRIMARY OUTPUT
- ✅ Console Statistics (#8)
- ✅ Feature Logger CSV
- ✅ Individual plots (1-6) — but save in `plots/detailed/` subfolder

**Delete:**
- ❌ Standalone heatmap (#7)
- ❌ plot_utils.py (#10)

**Organize:**
```
plots/
  behavioral_dashboard_20260227_123456.png   << Main output
  detailed/
    au12_signal_20260227_123456.png          << For deep dive
    expressivity_signal_20260227_123456.png
    ...
```

**Result:** Clean main folder, detailed analysis available if needed

---

## 💡 DECISION QUESTIONS FOR YOU

1. **Are you presenting to non-technical audience?**
   - YES → Use **Option A** (dashboard only)
   - NO → Use **Option C** (dashboard + details)

2. **Do you need per-feature raw/smooth/scaled comparison?**
   - YES → Keep individual plots (1-6)
   - NO → Delete them, dashboard shows everything

3. **Do you use EAR for debugging blink detection?**
   - YES → Keep EAR plot (#5)
   - NO → Delete it

4. **How many plot files per session is acceptable?**
   - 1-2 files → **Option A**
   - 7-8 files → **Option C**

---

## 🔧 WHAT TO DELETE IMMEDIATELY (NO DOUBT)

1. **[src/plot_utils.py](src/plot_utils.py)** — completely obsolete
2. **Standalone Heatmap** ([pipeline.py lines 377-397](src/pipeline.py#L377-L397)) — redundant with dashboard panel

These two provide ZERO additional value and create redundancy.

---

Let me know which option you prefer, and I'll clean up the code accordingly!
