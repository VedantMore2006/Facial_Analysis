import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np

mp = pd.read_csv("mediapipe_raw_features.csv")
of = pd.read_csv("openface_raw_features.csv")

# Extract all OpenFace AU intensity columns (AU*_r)
au_cols = [col for col in of.columns if col.startswith("AU") and col.endswith("_r")]
print(f"Found {len(au_cols)} OpenFace AU columns: {au_cols[:5]}...")

# Compute variance of OpenFace AUs per frame (measure of expressivity)
# Higher variance = more expressive face
of["au_variance"] = of[au_cols].var(axis=1)

# keep only relevant columns
mp = mp[["frame_index", "expressivity_raw"]]
of = of[["frame", "au_variance"]]

# rename to align
of = of.rename(columns={"frame": "frame_index"})

# merge
df = pd.merge(mp, of, on="frame_index")
print("Merged frames:", len(df))
print(df.head())
print("\nDescriptive statistics:")
print(df.describe())

# normalize both columns using StandardScaler
scaler = StandardScaler()

df["mp_norm"] = scaler.fit_transform(df[["expressivity_raw"]])
df["of_norm"] = scaler.fit_transform(df[["au_variance"]])

# ============================================================================
# STEP 4: Compute Correlation
# ============================================================================
corr, p = pearsonr(df["mp_norm"], df["of_norm"])

print("\n" + "="*60)
print("EXPRESSIVITY COMPARISON")
print("="*60)
print("MediaPipe: expressivity_raw (landmark movement magnitude)")
print("OpenFace:  variance of AU intensities")
print("="*60)
print("Pearson correlation:", corr)
print("p-value:", p)
print("="*60)

# Interpretation guide
print("\nInterpretation:")
if corr >= 0.8:
    print("  → Extremely strong correlation (0.8+)")
elif corr >= 0.6:
    print("  → Strong correlation (0.6-0.8)")
elif corr >= 0.4:
    print("  → Moderate correlation (0.4-0.6)")
else:
    print("  → Weak correlation (<0.4)")

print("\nNote: These measure similar but not identical concepts:")
print("  - MP expressivity: total landmark movement velocity")
print("  - OF AU variance: facial feature activation diversity")
print("  → Expect moderate correlation (0.4-0.6 is reasonable)")

# ============================================================================
# STEP 5: Plot Overlay
# ============================================================================
plt.figure(figsize=(14, 5))

plt.plot(df["mp_norm"], label="MediaPipe Expressivity", alpha=0.8)
plt.plot(df["of_norm"], label="OpenFace AU Variance", alpha=0.8)

plt.title(f"Expressivity Comparison (r={corr:.2f})")
plt.xlabel("Frame")
plt.ylabel("Normalized expressivity")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("expressivity_comparison_overlay.png", dpi=150)
print("\n✅ Saved: expressivity_comparison_overlay.png")
plt.show()

# ============================================================================
# STEP 6: Optional Smoothing Plot
# ============================================================================
df["mp_smooth"] = df["mp_norm"].rolling(10).mean()
df["of_smooth"] = df["of_norm"].rolling(10).mean()

plt.figure(figsize=(14, 5))

plt.plot(df["mp_smooth"], label="MP Expressivity smoothed", linewidth=2)
plt.plot(df["of_smooth"], label="OF AU Variance smoothed", linewidth=2)

plt.legend()
plt.title("Smoothed Expressivity Signals (10-frame rolling average)")
plt.xlabel("Frame")
plt.ylabel("Normalized expressivity")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("expressivity_comparison_smoothed.png", dpi=150)
print("✅ Saved: expressivity_comparison_smoothed.png")
plt.show()

# ============================================================================
# STEP 7: Check Frame Alignment (Hidden Bug Detector)
# ============================================================================
corr_array = np.correlate(
    df["mp_norm"] - df["mp_norm"].mean(),
    df["of_norm"] - df["of_norm"].mean(),
    mode="full"
)

lag = corr_array.argmax() - (len(df) - 1)

print("\n" + "="*60)
print("Frame Alignment Check:")
print(f"Best lag (frames): {lag}")
print("="*60)

if abs(lag) <= 1:
    print("  → Perfect sync! (lag ≈ 0)")
elif abs(lag) <= 5:
    print("  → Minor misalignment (acceptable)")
else:
    print("  → WARNING: Significant temporal shift detected!")

print("\n✅ Expressivity analysis complete!")
