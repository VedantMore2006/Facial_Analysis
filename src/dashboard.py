"""
dashboard.py

Creates a combined behavioral dashboard visualization.

Displays:
- Time-series plot of all features
- Behavioral deviation heatmap
- Summary statistics table

All in one unified dashboard.
"""

import matplotlib.pyplot as plt
import numpy as np


def create_dashboard(
    feature_dict_lists,
    feature_names,
    baseline_frames
):
    """
    Create a comprehensive behavioral analysis dashboard.
    
    Args:
        feature_dict_lists: Dict mapping feature names to lists of values
        feature_names: List of feature names (determines display order)
        baseline_frames: Number of frames in baseline window (shaded gray)
    """
    fig = plt.figure(figsize=(14, 10))

    # ----------------------------
    # 1️⃣ Time-Series Plot
    # ----------------------------
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)

    for name in feature_names:
        ax1.plot(feature_dict_lists[name], label=name, linewidth=2)

    ax1.axvspan(0, baseline_frames, alpha=0.2, color='gray', label='Baseline')
    ax1.set_title("Scaled Behavioral Signals Over Time", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Scaled Value (0–1)")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    # ----------------------------
    # 2️⃣ Heatmap
    # ----------------------------
    ax2 = plt.subplot2grid((2, 2), (1, 0))

    matrix = np.array([feature_dict_lists[name] for name in feature_names])

    im = ax2.imshow(matrix, aspect='auto', cmap='coolwarm', vmin=0, vmax=1)
    ax2.set_yticks(range(len(feature_names)))
    ax2.set_yticklabels(feature_names)
    ax2.set_xlabel("Frame")
    ax2.set_title("Behavioral Deviation Heatmap", fontsize=12, fontweight='bold')
    ax2.axvline(x=baseline_frames, color='white', linewidth=2, linestyle='--', alpha=0.7)

    plt.colorbar(im, ax=ax2, fraction=0.046, label="Value")

    # ----------------------------
    # 3️⃣ Summary Statistics Table
    # ----------------------------
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    ax3.axis("off")

    summary_data = []

    for name in feature_names:
        mean_val = np.mean(feature_dict_lists[name])
        std_val = np.std(feature_dict_lists[name])
        summary_data.append([name, round(mean_val, 3), round(std_val, 3)])

    table = ax3.table(
        cellText=summary_data,
        colLabels=["Feature", "Mean", "Std"],
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax3.set_title("Session Summary Statistics", fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.show()
    
    return fig
