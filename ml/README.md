# Machine Learning Pipeline for Mental Health Classification

## 🎯 Overview

This ML pipeline trains an XGBoost classifier to detect behavioral patterns associated with mental health states from facial features.

**Classes (requested):**
- 0: Depression
- 1: Anxiety
- 2: Stress
- 3: Bipolar Mania
- 4: Phobia (Common)
- 5: Suicidal Tendency

**Features:** 34 behavioral features across 5 domains:
- Facial Action Units (AU)
- Eye & Gaze Behavior
- Head Motion
- Derived Features
- Temporal Response

---

## 🚀 Quick Start

### Option 1: Run Complete Pipeline (Synthetic Data)

```bash
cd ml/
python run_full_pipeline.py
```

This will:
1. ✅ Aggregate baseline data into 5-sec windows
2. ✅ Analyze feature distributions
3. ✅ Generate 12,000 synthetic samples (2K per class, 6 classes)
4. ✅ Train XGBoost classifier
5. ✅ Generate evaluation reports

**Time:** ~2-5 minutes  
**Output:** Trained model + evaluation plots

---

### Option 2: Hybrid Approach (Real + Synthetic)

**Step 1:** Generate baseline synthetic model (above)

**Step 2:** Record acted sessions (2 minutes each)

```bash
# Run pipeline while acting depressed
python run_pipeline.py
# Save as: output/scaled/depression_session.csv

# Repeat for: anxiety, stress, bipolar_mania, phobia_common, suicidal_tendency
```

**Step 3:** Extract real patterns

```python
from synthetic_generator import generate_from_real_session

# Extract patterns from your acted session
real_patterns = generate_from_real_session(
    csv_path='output/scaled/depression_session.csv',
    disorder_name='depression'
)
# Saves to: ml/real_patterns_depression.json
```

**Step 4:** Generate hybrid dataset

```python
from synthetic_generator import generate_hybrid_dataset

dataset = generate_hybrid_dataset(
    baseline_stats_path='ml/baseline_stats.json',
    real_patterns_paths={
        'depression': 'ml/real_patterns_depression.json',
        # Add more as you record them
    },
    samples_per_class=2000,
    output_path='ml/training_dataset_hybrid.csv'
)
```

**Step 5:** Re-train with hybrid data

```python
from train_model import MentalHealthClassifier

classifier = MentalHealthClassifier()
classifier.train('ml/training_dataset_hybrid.csv')
classifier.save_model('ml/mental_health_model.pkl', 'ml/model_metadata.json')
```

---

## 📂 Pipeline Components

### 1. `window_aggregator.py`
Converts frame-level features → 5-second windows

**Input:** `output/scaled/23_51_05_03.csv` (7K frames)  
**Output:** `ml/baseline_windows.csv` (~24 windows)

**Usage:**
```python
from window_aggregator import WindowAggregator

aggregator = WindowAggregator(window_duration=5.0, fps=30)
windowed = aggregator.aggregate_csv(
    csv_path='output/scaled/23_51_05_03.csv',
    output_path='ml/baseline_windows.csv',
    label=None
)
```

### 2. `baseline_stats.py`
Analyzes baseline distributions (mean, std, min, max)

**Input:** `ml/baseline_windows.csv`  
**Output:** `ml/baseline_stats.json`

**Usage:**
```python
from baseline_stats import BaselineAnalyzer

analyzer = BaselineAnalyzer()
stats = analyzer.analyze_csv(
    csv_path='ml/baseline_windows.csv',
    save_path='ml/baseline_stats.json'
)
```

### 3. `disorder_profiles.py`
Defines behavioral modifiers for each disorder

**Example modifiers:**
```python
DISORDER_PROFILES = {
    'depression': {
        'S_AU12Mean': {'mean_mult': 0.60, 'std_mult': 0.80},  # ↓ smile
        'S_ResponseLatencyMean': {'mean_mult': 1.40, ...},    # ↑ latency
    },
    'anxiety': {
        'S_BlinkRate': {'mean_mult': 1.50, ...},              # ↑ blinks
        'S_AU4MeanActivation': {'mean_mult': 1.45, ...},      # ↑ tension
    }
}
```

### 4. `synthetic_generator.py`
Generates training samples from modified distributions

**Output:** `ml/training_dataset.csv` (10K rows)

**Usage:**
```python
from synthetic_generator import SyntheticGenerator

generator = SyntheticGenerator('ml/baseline_stats.json')
dataset = generator.generate_full_dataset(
    samples_per_class=2000,
    output_path='ml/training_dataset.csv'
)
```

### 5. `train_model.py`
Trains XGBoost classifier

**Model specs:**
- Algorithm: XGBoost (multi-class)
- Trees: 500
- Max depth: 6
- Learning rate: 0.05

**Usage:**
```python
from train_model import MentalHealthClassifier

classifier = MentalHealthClassifier()
results = classifier.train('ml/training_dataset.csv')
classifier.save_model('ml/mental_health_model.pkl', 'ml/model_metadata.json')
```

### 6. `evaluate_model.py`
Generates evaluation visualizations

**Outputs:**
- `ml/evaluation/confusion_matrix.png`
- `ml/evaluation/feature_importance.png`
- `ml/evaluation/class_performance.png`
- `ml/evaluation/confidence_distribution.png`
- `ml/evaluation/classification_report.txt`

**Usage:**
```python
from evaluate_model import ModelEvaluator

evaluator = ModelEvaluator(
    model_path='ml/mental_health_model.pkl',
    metadata_path='ml/model_metadata.json',
    dataset_path='ml/training_dataset.csv'
)
evaluator.generate_full_report()
```

---

## 📊 Expected Performance

With synthetic data:
- **Accuracy:** ~85-95%
- **Per-class F1:** ~0.80-0.95

With hybrid data (real patterns):
- **Accuracy:** Potentially higher
- **Generalization:** Better real-world performance

---

## 🎭 Recording Acting Sessions

When acting, exaggerate these behaviors for ~2 minutes:

**Depression:**
- Minimal facial movement
- Downward gaze
- Slow responses
- Flat expressions

**Anxiety:**
- Rapid blinking
- Tense brow
- Darting eyes
- Fidgety head movements

**Bipolar Mania:**
- Excessive smiling
- Rapid head motion
- Intense eye contact
- Fast responses

**Stress:**
- Sustained facial tension
- Increased micro-movements
- Elevated blink/gaze shifts
- Variable response timing

**Phobia (Common):**
- Vigilant scanning gaze
- Tension with avoidant eye contact
- Elevated blink rate
- Reactive facial shifts

**Suicidal Tendency:**
- Flat affect and reduced movement
- Reduced eye contact, more downward gaze
- Longer pauses and delayed responses
- Do not imitate self-harm actions

---

## 🔧 Troubleshooting

**Issue:** Not enough frames for windows  
**Fix:** Record longer sessions (need 150 frames minimum for 5-sec window @ 30fps)

**Issue:** Low model accuracy  
**Fix:** Increase `samples_per_class` or adjust disorder modifiers

**Issue:** Import errors  
**Fix:** Install dependencies:
```bash
pip install xgboost scikit-learn matplotlib seaborn
```

---

## 📝 File Structure

```
ml/
├── window_aggregator.py        # Frame → Window conversion
├── baseline_stats.py           # Distribution analysis
├── disorder_profiles.py        # Behavioral modifiers
├── synthetic_generator.py      # Sample generation
├── train_model.py             # XGBoost training
├── evaluate_model.py          # Performance evaluation
├── run_full_pipeline.py       # Orchestrator
├── README.md                  # This file
│
├── baseline_windows.csv       # Windowed baseline data
├── baseline_stats.json        # Feature distributions
├── training_dataset.csv       # 10K training samples
├── mental_health_model.pkl    # Trained classifier
├── model_metadata.json        # Model info
├── feature_importance.csv     # Feature rankings
│
├── real_patterns_*.json       # [Optional] Real patterns
└── evaluation/                # Evaluation outputs
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── class_performance.png
    ├── confidence_distribution.png
    └── classification_report.txt
```

---

## 🎯 Next Steps After Training

1. **Review Results:** Check evaluation plots in `ml/evaluation/`
2. **Test Live:** Integrate model into real-time pipeline
3. **Collect Real Data:** Record acted sessions to improve model
4. **Deploy:** Use trained model for behavioral monitoring
5. **Iterate:** Refine disorder modifiers based on domain knowledge

---

## ⚠️ Important Notes

- This model detects **behavioral patterns**, not clinical diagnoses
- Synthetic data is for proof-of-concept only
- Real clinical validation requires professional data
- Always consult mental health professionals for actual diagnosis

---

## 📚 References

Disorder behavioral patterns based on:
- DSM-5 criteria
- Clinical psychology literature
- Facial action coding system (FACS)
- Psychomotor research

---

**Ready to train? Run:**
```bash
python ml/run_full_pipeline.py
```
