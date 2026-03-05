import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np

mp = pd.read_csv("mediapipe_raw_features.csv")
of = pd.read_csv("openface_raw_features.csv")

# keep only relevant columns
mp = mp[["frame_index", "ear_raw"]]
of = of[["frame", "AU45_r"]]

# rename to align
of = of.rename(columns={"frame": "frame_index"})

# merge
df = pd.merge(mp, of, on="frame_index")
df = df[df["AU45_r"] >= 0]
print("Merged frames:", len(df))
print(df.head())

# normalize both columns using StandardScaler
scaler = StandardScaler()

df["mp_norm"] = scaler.fit_transform(df[["ear_raw"]])
df["of_norm"] = scaler.fit_transform(df[["AU45_r"]])

# ============================================================================
# STEP 4: Compute Correlation
# ============================================================================
corr, p = pearsonr(df["mp_norm"], df["of_norm"])

print("\n" + "="*60)
print("EYE ASPECT RATIO (BLINK) COMPARISON")
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

print("\nNote: EAR and AU45 measure opposite directions:")
print("  - EAR: High = eyes open, Low = eyes closed")
print("  - AU45: High = eyes closed, Low = eyes open")
print("  → Expect NEGATIVE correlation if both are correct!")

# ============================================================================
# STEP 5: Plot Overlay
# ============================================================================
plt.figure(figsize=(14, 5))

plt.plot(df["mp_norm"], label="MediaPipe EAR (raw)", alpha=0.8)
plt.plot(df["of_norm"], label="OpenFace AU45_r (blink)", alpha=0.8)

plt.title(f"Eye Aspect Ratio / Blink Comparison (r={corr:.2f})")
plt.xlabel("Frame")
plt.ylabel("Normalized intensity")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ear_comparison_overlay.png", dpi=150)
print("\n✅ Saved: ear_comparison_overlay.png")
plt.show()

# ============================================================================
# STEP 6: Optional Smoothing Plot
# ============================================================================
df["mp_smooth"] = df["mp_norm"].rolling(10).mean()
df["of_smooth"] = df["of_norm"].rolling(10).mean()

plt.figure(figsize=(14, 5))

plt.plot(df["mp_smooth"], label="MP EAR smoothed", linewidth=2)
plt.plot(df["of_smooth"], label="OF AU45 smoothed", linewidth=2)

plt.legend()
plt.title("Smoothed Eye/Blink Signals (10-frame rolling average)")
plt.xlabel("Frame")
plt.ylabel("Normalized intensity")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ear_comparison_smoothed.png", dpi=150)
print("✅ Saved: ear_comparison_smoothed.png")
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

print("\n✅ Eye aspect ratio / blink analysis complete!")
