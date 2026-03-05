import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np

mp = pd.read_csv("mediapipe_raw_features.csv")
of = pd.read_csv("openface_raw_features.csv")

# keep only relevant columns
mp = mp[["frame_index", "yaw_raw"]]
of = of[["frame", "pose_Ry"]]

# rename to align
of = of.rename(columns={"frame": "frame_index"})

# merge
df = pd.merge(mp, of, on="frame_index")
print("Merged frames:", len(df))
print(df.head())

# normalize both columns using StandardScaler
scaler = StandardScaler()

df["mp_norm"] = scaler.fit_transform(df[["yaw_raw"]])
df["of_norm"] = scaler.fit_transform(df[["pose_Ry"]])

# ============================================================================
# STEP 4: Compute Correlation
# ============================================================================
corr, p = pearsonr(df["mp_norm"], df["of_norm"])

print("\n" + "="*60)
print("HEAD YAW COMPARISON")
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

print("\nNote: For geometric head pose features, 0.7+ is expected.")

# ============================================================================
# STEP 5: Plot Overlay
# ============================================================================
plt.figure(figsize=(14, 5))

plt.plot(df["mp_norm"], label="MediaPipe Yaw (raw)", alpha=0.8)
plt.plot(df["of_norm"], label="OpenFace pose_Ry", alpha=0.8)

plt.title(f"Head Yaw Comparison (r={corr:.2f})")
plt.xlabel("Frame")
plt.ylabel("Normalized yaw angle")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("yaw_comparison_overlay.png", dpi=150)
print("\n✅ Saved: yaw_comparison_overlay.png")
plt.show()

# ============================================================================
# STEP 6: Optional Smoothing Plot
# ============================================================================
df["mp_smooth"] = df["mp_norm"].rolling(10).mean()
df["of_smooth"] = df["of_norm"].rolling(10).mean()

plt.figure(figsize=(14, 5))

plt.plot(df["mp_smooth"], label="MP smoothed", linewidth=2)
plt.plot(df["of_smooth"], label="OF smoothed", linewidth=2)

plt.legend()
plt.title("Smoothed Head Yaw Signals (10-frame rolling average)")
plt.xlabel("Frame")
plt.ylabel("Normalized yaw angle")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("yaw_comparison_smoothed.png", dpi=150)
print("✅ Saved: yaw_comparison_smoothed.png")
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

print("\n✅ Head yaw analysis complete!")
