# Feature Visualization System

Comprehensive visualization and analysis tool for behavioral features.

## 📊 What It Does

This system generates beautiful, insightful visualizations for all 34 behavioral features:

- **Time Series Plots**: Shows both raw and scaled values over time
- **Distribution Analysis**: Histograms with mean/median indicators
- **Correlation Heatmaps**: Identifies feature relationships
- **Outlier Detection**: Box plots highlighting anomalies
- **Summary Reports**: Statistical overview of all features

## 🎯 Key Features

✓ **Automatic Scaling Issue Detection**
  - Identifies features outside [0,1] range
  - Detects constant/zero-valued features
  - Flags potential outliers

✓ **Organized by Feature Groups**
  - Facial AU Features (9)
  - Eye Features (6)
  - Head Motion Features (5)
  - Derived Features (8)
  - Temporal Features (6)

✓ **Beautiful, Publication-Ready Plots**
  - High-resolution (300 DPI)
  - Professional color schemes
  - Clear labels and legends

## 📁 Output Structure

```
feature_plots/
├── time_series/          # Time series for each feature group
│   ├── facial_au_time_series.png
│   ├── eye_time_series.png
│   ├── head_motion_time_series.png
│   ├── derived_time_series.png
│   └── temporal_time_series.png
├── distributions/        # Distribution plots
│   ├── facial_au_distributions.png
│   └── ...
├── heatmaps/            # Correlation analysis
│   └── correlation_heatmap.png
├── outliers/            # Outlier detection
│   ├── facial_au_outliers.png
│   └── ...
├── summary/             # Statistical summaries
│   ├── summary_statistics.png
│   └── feature_statistics.csv
└── plot_features.py     # Main plotting script
```

## 🚀 Usage

### Automatic Mode (Uses Latest CSV Files)

```bash
python feature_plots/plot_features.py
```

### Manual Mode (Specify CSV Files)

```bash
python feature_plots/plot_features.py output/raw/22_55_05_03.csv output/scaled/22_55_05_03.csv
```

### Quick Plot Latest Data

```bash
cd feature_plots
python plot_features.py
```

## 📦 Dependencies

```bash
pip install pandas numpy matplotlib seaborn
```

## 🔍 Scaling Issue Detection

The system automatically identifies:

1. **Values < 0 or > 1**: Features not properly normalized
2. **Constant Features**: std < 0.01 (no variation)
3. **Mostly Zeros**: >90% zero values
4. **Extreme Outliers**: Beyond expected range

## 📈 Plot Descriptions

### Time Series Plots
- **Blue line**: Scaled values [0,1]
- **Light blue**: Raw values (secondary y-axis)
- **Red dashed line**: Baseline (0.5)
- Shows temporal evolution of each feature

### Distribution Plots
- **Histogram**: Frequency distribution
- **KDE curve**: Smoothed probability density
- **Red line**: Mean value
- **Green line**: Median value

### Correlation Heatmap
- **Red**: Positive correlation
- **Blue**: Negative correlation
- **White**: No correlation
- Lower triangle only (no redundancy)

### Outlier Detection (Box Plots)
- **Box**: 25th to 75th percentile
- **Line in box**: Median
- **Whiskers**: 1.5 × IQR
- **Dots**: Outliers
- **Reference lines**: 0.0, 0.5, 1.0

## 💡 Tips

- Run after each data collection session
- Compare plots across sessions to track changes
- Use summary statistics CSV for numerical analysis
- Check outlier plots to identify problematic features

## 🐛 Troubleshooting

**No plots generated?**
- Ensure CSV files exist in `output/raw/` and `output/scaled/`
- Check that CSV files have data (not just headers)

**Empty plots?**
- Verify that features are being computed correctly
- Check for all-zero columns

**Import errors?**
- Install dependencies: `pip install pandas matplotlib seaborn`

## 📊 Example Output

After running, you'll see:

```
======================================================================
FEATURE VISUALIZATION PIPELINE
======================================================================
Loading CSV files...
✓ Loaded 450 frames
✓ Found 34 features

======================================================================
SCALING ANALYSIS
======================================================================

⚠ Found 3 features with potential issues:

  • AU12Variance                       | Almost constant (std=0.000124)
  • ShoulderElevationIndex            | Mostly zeros (437/450)
  • DownwardGazeFrequency             | Values above 1 (max=1.2341)

======================================================================
GENERATING TIME SERIES PLOTS
======================================================================
...

✓ ALL VISUALIZATIONS COMPLETE
```

## 🎨 Customization

Edit `plot_features.py` to:
- Change color schemes
- Adjust figure sizes
- Add custom plots
- Modify feature groupings

---

**Created for**: Facial Analysis Pipeline  
**Version**: 1.0  
**Date**: March 2026
