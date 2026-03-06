# Quick Reference - What to Share

## Essential Files (Send These)

```bash
# Navigate to ml/ directory and package these files:

1. training_dataset.csv              # 27MB - Main dataset (12,000 samples)
2. model_metadata.json               # 6KB - Feature names & labels
3. DATA_SHARING_PACKAGE.md           # Documentation & instructions
```

## Optional Files (If They Need the Model)

```bash
4. mental_health_model.pkl           # 2.9MB - Pre-trained XGBoost model
5. baseline_stats.json               # 30KB - Reference distributions
```

---

## Packaging Commands

### Option 1: Create ZIP Archive

```bash
cd /home/vedant/Facial_analysis/ml

# Essential files only (27 MB)
zip dataset_package.zip \
    training_dataset.csv \
    model_metadata.json \
    DATA_SHARING_PACKAGE.md

# With model included (30 MB)
zip dataset_package_with_model.zip \
    training_dataset.csv \
    model_metadata.json \
    DATA_SHARING_PACKAGE.md \
    mental_health_model.pkl \
    baseline_stats.json
```

### Option 2: Create TAR Archive

```bash
cd /home/vedant/Facial_analysis/ml

# Essential files only
tar -czf dataset_package.tar.gz \
    training_dataset.csv \
    model_metadata.json \
    DATA_SHARING_PACKAGE.md

# With model included
tar -czf dataset_package_with_model.tar.gz \
    training_dataset.csv \
    model_metadata.json \
    DATA_SHARING_PACKAGE.md \
    mental_health_model.pkl \
    baseline_stats.json
```

---

## What They Get

### Dataset Overview
- **12,000 samples** (2,000 per class)
- **136 features** (facial, eye, head movement)
- **6 classes** (Depression, Anxiety, Stress, Mania, Phobia, Suicidal)
- **Format:** CSV (easy to load in any language)

### Complete Documentation
- Feature descriptions (all 136 features explained)
- Label definitions (what each class represents)
- Code examples (Python - loading, training, using model)
- Expected performance (accuracy benchmarks)
- Limitations & ethical considerations

---

## Quick Email Template

```
Subject: Behavioral Dataset - Mental Health Classification

Hi [Name],

Attached is the synthetic behavioral dataset for the mental health 
classification project. Here's what's included:

FILES:
- training_dataset.csv (12,000 samples, 6 classes, 136 features)
- model_metadata.json (feature names and label mappings)
- DATA_SHARING_PACKAGE.md (complete documentation)
[- mental_health_model.pkl (optional pre-trained XGBoost model)]

QUICK STATS:
- 12,000 balanced samples (2,000 per class)
- 6 mental health conditions (Depression, Anxiety, Stress, 
  Bipolar Mania, Phobia, Suicidal Tendency)
- 136 behavioral features (facial actions, eye activity, head movement)
- Expected accuracy: 87-92% with XGBoost

DOCUMENTATION:
Everything is explained in DATA_SHARING_PACKAGE.md including:
- Feature descriptions
- Label definitions
- Code examples (Python)
- Performance benchmarks
- Ethical considerations

USAGE:
```python
import pandas as pd
df = pd.read_csv('training_dataset.csv')
X = df.drop('label', axis=1)  # 136 features
y = df['label']                 # 0-5 class labels
```

Let me know if you have any questions!

Best,
[Your Name]
```

---

## File Sizes

```
training_dataset.csv              27 MB
model_metadata.json              6 KB
DATA_SHARING_PACKAGE.md          ~50 KB (documentation)
mental_health_model.pkl          2.9 MB (optional)
baseline_stats.json              30 KB (optional)

Total (essential only):           ~27 MB
Total (with model):               ~30 MB
```

---

## Verification Checklist

Before sending, verify:

- [ ] training_dataset.csv has 12,001 lines (12k samples + header)
- [ ] model_metadata.json contains 136 feature names
- [ ] DATA_SHARING_PACKAGE.md opens and renders correctly
- [ ] ZIP/TAR archive extracts without errors
- [ ] Total size is reasonable for email/transfer method

Quick verification:
```bash
wc -l training_dataset.csv       # Should show: 12001
head -1 training_dataset.csv     # Should show all feature columns
tail -1 training_dataset.csv     # Should have label column
```

---

## Alternative: Share via Cloud

If files are too large for email:

**Google Drive / Dropbox:**
```
1. Upload dataset_package.zip
2. Get shareable link
3. Send link with expiry date
```

**GitHub (if appropriate):**
```bash
git init dataset_repo
cp training_dataset.csv dataset_repo/
cp model_metadata.json dataset_repo/
cp DATA_SHARING_PACKAGE.md dataset_repo/README.md
cd dataset_repo
git add .
git commit -m "Initial dataset release"
git push
```

**Note:** Check if data contains sensitive info before public sharing!

---

## They Should Have These Installed

```bash
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install scikit-learn>=1.3.0

# Optional (if using pre-trained model):
pip install xgboost>=2.0.0
```

---

## Summary

**Send:**
1. `training_dataset.csv`
2. `model_metadata.json`  
3. `DATA_SHARING_PACKAGE.md`

**They can:**
- Load data in any language (CSV format)
- Train their own models
- Use provided code examples
- Understand all 136 features
- See expected performance benchmarks

**All documentation is self-contained in DATA_SHARING_PACKAGE.md**
