#!/usr/bin/env python3
# ============================================================================
# FEATURE VISUALIZATION TOOL
# ============================================================================
# Purpose: Comprehensive plotting utility for facial analysis features
# 
# Generates:
# 1. Individual time-series plots for each feature
# 2. Combined heatmap visualization
# 3. Statistical overlays (mean, std deviation)
# 4. Baseline phase indicators
# 
# Usage:
#   python plot_features.py <csv_file_path>
#   or
#   python plot_features.py  (uses latest CSV in data/)
# 
# Output:
#   Saves plots to plots/ directory
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import glob
import os
from datetime import datetime
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

# Feature metadata (names, descriptions, colors)
FEATURE_INFO = {
    'S_AU12': {
        'name': 'Smile Intensity (AU12)',
        'description': 'Mouth corner width (lip corner puller)',
        'color': '#FF6B6B',
        'unit': 'normalized'
    },
    'S_AUVar': {
        'name': 'Facial Expressivity',
        'description': 'Overall facial movement variation',
        'color': '#4ECDC4',
        'unit': 'normalized'
    },
    'S_HeadVelocity': {
        'name': 'Head Movement Speed',
        'description': 'Horizontal rotation velocity (yaw)',
        'color': '#95E1D3',
        'unit': 'normalized'
    },
    'S_EyeContact': {
        'name': 'Eye Contact Ratio',
        'description': 'Proportion looking at camera',
        'color': '#F38181',
        'unit': 'normalized'
    },
    'S_BlinkRate': {
        'name': 'Blink Frequency',
        'description': 'Blinks per minute',
        'color': '#AA96DA',
        'unit': 'normalized'
    },
    'S_ResponseLatency': {
        'name': 'Response Latency',
        'description': 'Reaction time to stimulus',
        'color': '#FCBAD3',
        'unit': 'normalized'
    }
}

# Plot styling
BASELINE_DURATION_SEC = 30  # Duration of baseline phase
FPS = 15                    # Frames per second
PLOT_DPI = 150              # Resolution for saved plots
FIGSIZE_INDIVIDUAL = (14, 5)
FIGSIZE_HEATMAP = (16, 8)

# Output directory
OUTPUT_DIR = Path('plots')


# ============================================================================
# CLI ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Facial Analysis Feature Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect latest CSV, save to 'plots/' (default)
  python plot_features.py
  
  # Specify CSV file
  python plot_features.py --csv data/features_1772545814.csv
  
  # Custom output directory name
  python plot_features.py -c data/features_1772545814.csv -o my_analysis
  
  # Verbose mode for detailed output
  python plot_features.py --csv data/features_1772545814.csv --verbose
  
  # Combined options
  python plot_features.py -c data/features_1772545814.csv -o results -v
        """
    )
    
    parser.add_argument(
        '--csv', '-c',
        type=str,
        default=None,
        metavar='PATH',
        help='Path to features CSV file (default: auto-finds latest in data/)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='plots',
        metavar='NAME',
        help='Output directory name for plots (default: plots)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed processing information'
    )
    
    return parser.parse_args()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def find_latest_csv(data_dir='data', verbose=False):
    """Find the most recent features CSV file."""
    pattern = f"{data_dir}/features_*.csv"
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No feature CSV files found in {data_dir}/")
    
    # Sort by modification time, get latest
    latest = max(files, key=os.path.getmtime)
    
    if verbose:
        print(f"  → Found {len(files)} CSV file(s) in {data_dir}/")
        print(f"  → Selected latest: {latest}")
    
    return latest


def load_feature_data(csv_path, verbose=False):
    """Load and validate feature CSV data."""
    # Validate file exists
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    if verbose:
        print(f"  → Loading: {csv_path}")
    else:
        print(f"Loading data from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")
    
    # Validate expected columns
    expected_cols = list(FEATURE_INFO.keys())
    missing = set(expected_cols) - set(df.columns)
    
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    
    if verbose:
        print(f"  → Loaded {len(df)} frames successfully")
    else:
        print(f"✓ Loaded {len(df)} frames")
    
    print(f"✓ Duration: {len(df) / FPS:.1f} seconds")
    
    return df


def compute_statistics(df):
    """Compute summary statistics for all features."""
    stats = {}
    
    for feature in FEATURE_INFO.keys():
        if feature in df.columns:
            values = df[feature].values
            stats[feature] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
    
    return stats


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_individual_feature(df, feature_name, baseline_frames, output_dir):
    """
    Create detailed time-series plot for a single feature.
    
    Includes:
    - Time-series line plot
    - Baseline phase shading
    - Mean and std deviation lines
    - Statistical annotations
    """
    # Setup plot
    fig, ax = plt.subplots(figsize=FIGSIZE_INDIVIDUAL, dpi=PLOT_DPI)
    
    # Get feature info
    info = FEATURE_INFO[feature_name]
    values = df[feature_name].values
    frames = np.arange(len(values))
    time_seconds = frames / FPS
    
    # Calculate statistics
    overall_mean = np.mean(values)
    overall_std = np.std(values)
    baseline_mean = np.mean(values[:baseline_frames])
    deviation_mean = np.mean(values[baseline_frames:]) if len(values) > baseline_frames else None
    
    # Plot main line
    ax.plot(time_seconds, values, color=info['color'], linewidth=2, 
            label=info['name'], alpha=0.8)
    
    # Shade baseline phase
    if baseline_frames > 0:
        baseline_time = baseline_frames / FPS
        ax.axvspan(0, baseline_time, alpha=0.15, color='gray', 
                   label='Baseline Phase')
        ax.axvline(baseline_time, color='gray', linestyle='--', 
                   linewidth=1.5, alpha=0.5)
    
    # Add mean line
    ax.axhline(overall_mean, color='red', linestyle='--', linewidth=1.5, 
               alpha=0.6, label=f'Mean: {overall_mean:.3f}')
    
    # Add std deviation bands
    ax.axhline(overall_mean + overall_std, color='red', linestyle=':', 
               linewidth=1, alpha=0.4)
    ax.axhline(overall_mean - overall_std, color='red', linestyle=':', 
               linewidth=1, alpha=0.4)
    ax.fill_between(time_seconds, 
                     overall_mean - overall_std, 
                     overall_mean + overall_std, 
                     alpha=0.1, color='red', label=f'±1 SD: {overall_std:.3f}')
    
    # Add neutral baseline reference at 0.5
    ax.axhline(0.5, color='black', linestyle=':', linewidth=1, alpha=0.3)
    
    # Formatting
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Scaled Value (0-1)', fontsize=12, fontweight='bold')
    ax.set_title(f'{info["name"]}\n{info["description"]}', 
                 fontsize=14, fontweight='bold', pad=15)
    
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Add statistics text box
    stats_text = f'Statistics:\n'
    stats_text += f'Baseline Mean: {baseline_mean:.3f}\n'
    if deviation_mean is not None:
        stats_text += f'Deviation Mean: {deviation_mean:.3f}\n'
    stats_text += f'Min: {np.min(values):.3f}\n'
    stats_text += f'Max: {np.max(values):.3f}\n'
    stats_text += f'Range: {np.max(values) - np.min(values):.3f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.8), fontsize=9, family='monospace')
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / f'{feature_name}_timeseries.png'
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    
    plt.close()


def plot_combined_heatmap(df, baseline_frames, output_dir):
    """
    Create heatmap visualization of all features over time.
    
    Shows intensity patterns across all features simultaneously.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP, dpi=PLOT_DPI)
    
    # Prepare data (transpose so features are rows, time is columns)
    features = list(FEATURE_INFO.keys())
    feature_labels = [FEATURE_INFO[f]['name'] for f in features]
    data = df[features].T.values
    
    # Create heatmap
    im = ax.imshow(data, aspect='auto', cmap='RdYlBu_r', 
                   vmin=0, vmax=1, interpolation='nearest')
    
    # Set ticks and labels
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(feature_labels, fontsize=11)
    
    # X-axis: show time in seconds
    num_frames = data.shape[1]
    time_points = [0, num_frames//4, num_frames//2, 3*num_frames//4, num_frames-1]
    time_labels = [f'{int(t/FPS)}s' for t in time_points]
    ax.set_xticks(time_points)
    ax.set_xticklabels(time_labels, fontsize=10)
    
    # Add baseline separator line
    if baseline_frames > 0:
        ax.axvline(baseline_frames, color='white', linewidth=3, linestyle='--', alpha=0.8)
        ax.text(baseline_frames/2, -0.5, 'BASELINE', ha='center', va='top',
                fontsize=10, fontweight='bold', color='gray')
        ax.text(baseline_frames + (num_frames-baseline_frames)/2, -0.5, 
                'DEVIATION', ha='center', va='top',
                fontsize=10, fontweight='bold', color='darkred')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Scaled Value', rotation=270, labelpad=20, fontsize=11, fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_title('Behavioral Feature Heatmap\nColor intensity indicates deviation from baseline',
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add interpretation guide
    guide_text = 'Blue: Below baseline | White: At baseline (0.5) | Red: Above baseline'
    fig.text(0.5, 0.02, guide_text, ha='center', fontsize=10, 
             style='italic', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    
    # Save plot
    output_path = output_dir / 'combined_heatmap.png'
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    
    plt.close()


def plot_all_features_grid(df, baseline_frames, output_dir):
    """
    Create a grid layout showing all features in one figure.
    
    Useful for quick overview of all features.
    """
    features = list(FEATURE_INFO.keys())
    n_features = len(features)
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), dpi=PLOT_DPI)
    axes = axes.flatten()
    
    time_seconds = np.arange(len(df)) / FPS
    baseline_time = baseline_frames / FPS
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        info = FEATURE_INFO[feature]
        values = df[feature].values
        
        # Plot line
        ax.plot(time_seconds, values, color=info['color'], linewidth=1.5, alpha=0.8)
        
        # Shade baseline
        if baseline_frames > 0:
            ax.axvspan(0, baseline_time, alpha=0.1, color='gray')
            ax.axvline(baseline_time, color='gray', linestyle='--', 
                       linewidth=1, alpha=0.4)
        
        # Add mean line
        mean_val = np.mean(values)
        ax.axhline(mean_val, color='red', linestyle='--', 
                   linewidth=1, alpha=0.5)
        ax.axhline(0.5, color='black', linestyle=':', linewidth=0.8, alpha=0.3)
        
        # Formatting
        ax.set_title(info['name'], fontsize=11, fontweight='bold')
        ax.set_ylabel('Value', fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2)
        
        # Add mean annotation
        ax.text(0.98, 0.95, f'μ={mean_val:.3f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Overall title
    fig.suptitle('All Behavioral Features - Time Series Overview', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Common x-label
    fig.text(0.5, 0.02, 'Time (seconds)', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    
    # Save
    output_path = output_dir / 'all_features_grid.png'
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    
    plt.close()


def print_summary_statistics(stats):
    """Print formatted summary statistics to console."""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    for feature, values in stats.items():
        info = FEATURE_INFO[feature]
        print(f"\n{info['name']}:")
        print(f"  Mean:   {values['mean']:.4f}")
        print(f"  Std:    {values['std']:.4f}")
        print(f"  Median: {values['median']:.4f}")
        print(f"  Range:  [{values['min']:.4f}, {values['max']:.4f}]")
    
    print("\n" + "="*70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    # Parse command-line arguments
    args = parse_arguments()
    
    print("\n" + "="*70)
    print("FACIAL ANALYSIS FEATURE VISUALIZATION")
    print("="*70 + "\n")
    
    # Determine CSV file path
    if args.csv:
        csv_path = args.csv
        if args.verbose:
            print("CLI Mode: Using provided CSV path")
    else:
        if args.verbose:
            print("CLI Mode: Auto-detecting latest CSV...")
        else:
            print("No CSV file specified, searching for latest...")
        csv_path = find_latest_csv(verbose=args.verbose)
    
    # Load data
    try:
        df = load_feature_data(csv_path, verbose=args.verbose)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    
    # Calculate baseline frames
    baseline_frames = int(BASELINE_DURATION_SEC * FPS)
    
    # Compute statistics
    stats = compute_statistics(df)
    
    # Create output directory with user-specified name
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    if args.verbose:
        print(f"\n[VERBOSE] Configuration:")
        print(f"  - Input CSV: {csv_path}")
        print(f"  - Output directory: {output_dir.absolute()}")
        print(f"  - Baseline duration: {BASELINE_DURATION_SEC}s ({baseline_frames} frames)")
        print(f"  - Total frames: {len(df)}")
        print(f"  - Total duration: {len(df) / FPS:.1f}s")
    else:
        print(f"\nOutput directory: {output_dir.absolute()}")
    
    # Generate plots
    print("\nGenerating plots...")
    print("-" * 70)
    
    print("\n1. Individual feature plots:")
    for feature in FEATURE_INFO.keys():
        plot_individual_feature(df, feature, baseline_frames, output_dir)
    
    print("\n2. Combined visualizations:")
    plot_combined_heatmap(df, baseline_frames, output_dir)
    plot_all_features_grid(df, baseline_frames, output_dir)
    
    # Print statistics
    print_summary_statistics(stats)
    
    # Success message
    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETE!")
    print(f"📊 Generated {len(FEATURE_INFO) + 2} plots")
    print(f"📁 Location: {output_dir.absolute()}")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    # Run main
    main()
