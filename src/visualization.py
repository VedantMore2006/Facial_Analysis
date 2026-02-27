"""
visualization.py

Handles signal visualization for analysis.

Responsibilities:
- Plot raw/smoothed/scaled signals
- Mark baseline window
- Generate heatmap of behavioral deviations
- Display summary statistics
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_signal(raw, smoothed, scaled, name, baseline_frames=None):
    """
    Plot raw, smoothed, and scaled signal with optional baseline window.
    
    Args:
        raw: Raw signal values
        smoothed: Smoothed signal values
        scaled: Scaled signal values
        name: Title of the plot
        baseline_frames: Number of frames in baseline (will be shaded gray)
    """
    plt.figure(figsize=(12, 5))

    plt.plot(raw, label="Raw", alpha=0.5)
    plt.plot(smoothed, label="Smoothed", linewidth=2)
    plt.plot(scaled, label="Scaled", linewidth=2)

    if baseline_frames is not None:
        plt.axvspan(0, baseline_frames, alpha=0.2, color='gray', label="Baseline")

    plt.title(name)
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_heatmap(feature_matrix, feature_names, baseline_frames=None):
    """
    Plot behavioral deviation heatmap.
    
    Each row = feature
    Each column = frame
    Value = scaled value (0=blue, 1=red)
    
    Args:
        feature_matrix: Numpy array of shape (n_features, n_frames)
        feature_names: List of feature names
        baseline_frames: Number of frames in baseline (will be marked)
    """
    plt.figure(figsize=(14, 6))
    
    im = plt.imshow(feature_matrix, aspect='auto', cmap='coolwarm', vmin=0, vmax=1)
    plt.colorbar(im, label="Scaled Value")

    plt.yticks(range(len(feature_names)), feature_names)
    plt.xlabel("Frame")
    plt.title("Behavioral Deviation Heatmap")
    
    # Mark baseline window
    if baseline_frames is not None:
        plt.axvline(x=baseline_frames, color='white', linewidth=2, linestyle='--', label="Baseline End")
        plt.legend()

    plt.tight_layout()
    plt.show()


def print_summary_statistics(feature_names, feature_lists):
    """
    Print summary statistics table.
    
    Args:
        feature_names: List of feature names
        feature_lists: List of feature value lists (same order as names)
    """
    try:
        import pandas as pd
        
        summary = pd.DataFrame({
            "Feature": feature_names,
            "Mean": [np.mean(fl) for fl in feature_lists],
            "Std": [np.std(fl) for fl in feature_lists],
            "Min": [np.min(fl) for fl in feature_lists],
            "Max": [np.max(fl) for fl in feature_lists],
        })
        
        print("\n" + "="*70)
        print("BEHAVIORAL FEATURE SUMMARY")
        print("="*70)
        print(summary.to_string(index=False))
        print("="*70 + "\n")
        
    except ImportError:
        # Fallback if pandas not available
        print("\n" + "="*70)
        print("BEHAVIORAL FEATURE SUMMARY (without pandas)")
        print("="*70)
        for name, fl in zip(feature_names, feature_lists):
            print(f"{name:20s} | Mean: {np.mean(fl):.4f} | Std: {np.std(fl):.4f} | Min: {np.min(fl):.4f} | Max: {np.max(fl):.4f}")
        print("="*70 + "\n")
