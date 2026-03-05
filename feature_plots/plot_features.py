"""
Feature Visualization System

Generates comprehensive visualizations for all 34 behavioral features:
- Time series plots
- Distribution analysis
- Outlier detection
- Scaling validation
- Correlation heatmaps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set beautiful style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class FeatureVisualizer:
    """
    Comprehensive feature visualization and analysis tool.
    """
    
    def __init__(self, raw_csv_path, scaled_csv_path):
        """
        Initialize visualizer with CSV file paths.
        
        Args:
            raw_csv_path: Path to raw features CSV
            scaled_csv_path: Path to scaled features CSV
        """
        self.raw_csv_path = raw_csv_path
        self.scaled_csv_path = scaled_csv_path
        
        # Output directories
        self.time_series_dir = Path("feature_plots/time_series")
        self.dist_dir = Path("feature_plots/distributions")
        self.heatmap_dir = Path("feature_plots/heatmaps")
        self.summary_dir = Path("feature_plots/summary")
        self.outlier_dir = Path("feature_plots/outliers")
        
        # Ensure directories exist
        for dir_path in [self.time_series_dir, self.dist_dir, self.heatmap_dir, 
                         self.summary_dir, self.outlier_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load data
        print("Loading CSV files...")
        self.df_raw = pd.read_csv(raw_csv_path)
        self.df_scaled = pd.read_csv(scaled_csv_path)
        
        # Remove timestamp column for feature analysis
        self.raw_features = self.df_raw.drop('timestamp', axis=1)
        self.scaled_features = self.df_scaled.drop('timestamp', axis=1)
        
        # Feature groups
        self.feature_groups = {
            'Facial AU': ['AU12Mean', 'AU12Variance', 'AU12ActivationFrequency', 
                          'AU15MeanAmplitude', 'AU4MeanActivation', 'AU4DurationRatio',
                          'AU1AU2PeakIntensity', 'AU20ActivationRate', 'LipCompressionFrequency'],
            'Eye': ['BlinkRate', 'BlinkClusterDensity', 'BaselineEyeOpenness',
                    'GazeShiftFrequency', 'EyeContactRatio', 'DownwardGazeFrequency'],
            'Head Motion': ['MeanHeadVelocity', 'HeadVelocityPeak', 'HeadMotionEnergy',
                           'LandmarkDisplacementMean', 'PostureRigidityIndex'],
            'Derived': ['OverallAUVariance', 'FacialEmotionalRange', 'FacialTransitionFrequency',
                       'NearZeroAUActivationRatio', 'MotionEnergyFloorScore', 'GestureFrequency',
                       'MicroMotionEnergy', 'ShoulderElevationIndex'],
            'Temporal': ['ResponseLatencyMean', 'SpeechOnsetDelay', 'NodOnsetLatency',
                        'PauseDurationMean', 'ExtendedSilenceRatio', 'ReactionTimeInstabilityIndex']
        }
        
        print(f"✓ Loaded {len(self.df_raw)} frames")
        print(f"✓ Found {len(self.raw_features.columns)} features")
    
    def analyze_scaling_issues(self):
        """
        Identify features with scaling problems.
        """
        print("\n" + "=" * 70)
        print("SCALING ANALYSIS")
        print("=" * 70)
        
        issues = []
        
        for col in self.scaled_features.columns:
            values = self.scaled_features[col]
            
            min_val = values.min()
            max_val = values.max()
            mean_val = values.mean()
            std_val = values.std()
            
            # Check for issues
            problem = None
            if min_val < 0:
                problem = f"Values below 0 (min={min_val:.4f})"
            elif max_val > 1:
                problem = f"Values above 1 (max={max_val:.4f})"
            elif std_val < 0.01:
                problem = f"Almost constant (std={std_val:.6f})"
            elif (values == 0).sum() / len(values) > 0.9:
                problem = f"Mostly zeros ({(values == 0).sum()}/{len(values)})"
            
            if problem:
                issues.append({
                    'feature': col,
                    'problem': problem,
                    'min': min_val,
                    'max': max_val,
                    'mean': mean_val,
                    'std': std_val
                })
        
        if issues:
            print(f"\n⚠ Found {len(issues)} features with potential issues:\n")
            for issue in issues:
                print(f"  • {issue['feature']:<35} | {issue['problem']}")
                print(f"    Range: [{issue['min']:.4f}, {issue['max']:.4f}], "
                      f"Mean: {issue['mean']:.4f}, Std: {issue['std']:.6f}")
        else:
            print("\n✓ All scaled features are within [0, 1] range")
        
        return issues
    
    def plot_time_series_by_group(self):
        """
        Create time series plots organized by feature group.
        """
        print("\n" + "=" * 70)
        print("GENERATING TIME SERIES PLOTS")
        print("=" * 70)
        
        for group_name, features in self.feature_groups.items():
            print(f"\nProcessing {group_name} features...")
            
            # Create subplot grid
            n_features = len(features)
            n_cols = 3
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
            axes = axes.flatten() if n_features > 1 else [axes]
            
            for idx, feature in enumerate(features):
                ax = axes[idx]
                
                # Plot both raw and scaled
                if feature in self.raw_features.columns:
                    raw_col = feature
                    scaled_col = f"S_{feature}"
                    
                    # Raw values
                    ax2 = ax.twinx()
                    ax2.plot(self.raw_features[raw_col], 
                            color='lightblue', alpha=0.6, linewidth=1, label='Raw')
                    ax2.set_ylabel('Raw Value', color='lightblue')
                    ax2.tick_params(axis='y', labelcolor='lightblue')
                    
                    # Scaled values
                    if scaled_col in self.scaled_features.columns:
                        ax.plot(self.scaled_features[scaled_col], 
                               color='darkblue', linewidth=1.5, label='Scaled')
                        ax.axhline(y=0.5, color='red', linestyle='--', 
                                  alpha=0.3, linewidth=1, label='Baseline (0.5)')
                        ax.set_ylim([-0.1, 1.1])
                    
                    ax.set_title(feature, fontsize=11, fontweight='bold')
                    ax.set_xlabel('Frame', fontsize=9)
                    ax.set_ylabel('Scaled Value [0,1]', fontsize=9, color='darkblue')
                    ax.tick_params(axis='y', labelcolor='darkblue')
                    ax.grid(True, alpha=0.3)
                    
                    # Add legend
                    lines1, labels1 = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2, 
                             loc='upper right', fontsize=8)
            
            # Hide empty subplots
            for idx in range(n_features, len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            filename = f"{group_name.replace(' ', '_').lower()}_time_series.png"
            filepath = self.time_series_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved: {filepath}")
    
    def plot_distributions(self):
        """
        Create distribution plots for scaled features.
        """
        print("\n" + "=" * 70)
        print("GENERATING DISTRIBUTION PLOTS")
        print("=" * 70)
        
        for group_name, features in self.feature_groups.items():
            print(f"\nProcessing {group_name} features...")
            
            n_features = len(features)
            n_cols = 3
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
            axes = axes.flatten() if n_features > 1 else [axes]
            
            for idx, feature in enumerate(features):
                ax = axes[idx]
                scaled_col = f"S_{feature}"
                
                if scaled_col in self.scaled_features.columns:
                    values = self.scaled_features[scaled_col]
                    
                    # Histogram + KDE
                    sns.histplot(values, kde=True, ax=ax, bins=30, 
                                color='steelblue', alpha=0.6)
                    
                    # Add vertical lines for mean and median
                    mean_val = values.mean()
                    median_val = values.median()
                    ax.axvline(mean_val, color='red', linestyle='--', 
                              linewidth=2, label=f'Mean: {mean_val:.3f}')
                    ax.axvline(median_val, color='green', linestyle='--', 
                              linewidth=2, label=f'Median: {median_val:.3f}')
                    
                    ax.set_title(feature, fontsize=11, fontweight='bold')
                    ax.set_xlabel('Scaled Value', fontsize=9)
                    ax.set_ylabel('Frequency', fontsize=9)
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3, axis='y')
            
            # Hide empty subplots
            for idx in range(n_features, len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            filename = f"{group_name.replace(' ', '_').lower()}_distributions.png"
            filepath = self.dist_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved: {filepath}")
    
    def plot_correlation_heatmap(self):
        """
        Create correlation heatmap for all features.
        """
        print("\n" + "=" * 70)
        print("GENERATING CORRELATION HEATMAP")
        print("=" * 70)
        
        # Compute correlation matrix
        corr_matrix = self.scaled_features.corr()
        
        # Create large heatmap
        fig, ax = plt.subplots(figsize=(20, 18))
        
        # Plot heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   annot=False, fmt='.2f', ax=ax, vmin=-1, vmax=1)
        
        ax.set_title('Feature Correlation Matrix (Scaled Features)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        filepath = self.heatmap_dir / 'correlation_heatmap.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {filepath}")
    
    def plot_outlier_detection(self):
        """
        Create box plots to identify outliers.
        """
        print("\n" + "=" * 70)
        print("GENERATING OUTLIER DETECTION PLOTS")
        print("=" * 70)
        
        for group_name, features in self.feature_groups.items():
            print(f"\nProcessing {group_name} features...")
            
            # Filter features that exist
            existing_features = [f"S_{f}" for f in features 
                                if f"S_{f}" in self.scaled_features.columns]
            
            if not existing_features:
                continue
            
            fig, ax = plt.subplots(figsize=(16, 6))
            
            # Create box plot
            data_to_plot = [self.scaled_features[col] for col in existing_features]
            labels = [col.replace('S_', '') for col in existing_features]
            
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                           showmeans=True, meanline=True)
            
            # Color boxes
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.6)
            
            # Add reference lines
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.3, linewidth=1)
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.3, linewidth=1)
            ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.3, linewidth=1)
            
            ax.set_title(f'{group_name} Features - Outlier Detection', 
                        fontsize=14, fontweight='bold')
            ax.set_ylabel('Scaled Value', fontsize=11)
            ax.set_ylim([-0.2, 1.2])
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            filename = f"{group_name.replace(' ', '_').lower()}_outliers.png"
            filepath = self.outlier_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved: {filepath}")
    
    def generate_summary_report(self):
        """
        Generate comprehensive summary statistics report.
        """
        print("\n" + "=" * 70)
        print("GENERATING SUMMARY REPORT")
        print("=" * 70)
        
        summary_data = []
        
        for col in self.scaled_features.columns:
            values = self.scaled_features[col]
            
            summary_data.append({
                'Feature': col.replace('S_', ''),
                'Mean': values.mean(),
                'Std': values.std(),
                'Min': values.min(),
                'Max': values.max(),
                'Q25': values.quantile(0.25),
                'Median': values.median(),
                'Q75': values.quantile(0.75),
                'Zeros (%)': (values == 0).sum() / len(values) * 100,
                'Out of Range': ((values < 0) | (values > 1)).sum()
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        csv_path = self.summary_dir / 'feature_statistics.csv'
        summary_df.to_csv(csv_path, index=False)
        print(f"✓ Saved statistics: {csv_path}")
        
        # Create visual summary
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Mean values
        ax = axes[0, 0]
        summary_df.plot(x='Feature', y='Mean', kind='bar', ax=ax, 
                       color='steelblue', alpha=0.7)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax.set_title('Mean Values (All Features)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Scaled Value')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=90)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Standard deviation
        ax = axes[0, 1]
        summary_df.plot(x='Feature', y='Std', kind='bar', ax=ax, 
                       color='coral', alpha=0.7)
        ax.set_title('Standard Deviation (All Features)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Standard Deviation')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=90)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Value range
        ax = axes[1, 0]
        x = np.arange(len(summary_df))
        ax.barh(x, summary_df['Max'] - summary_df['Min'], 
               left=summary_df['Min'], alpha=0.7, color='green')
        ax.set_yticks(x)
        ax.set_yticklabels(summary_df['Feature'], fontsize=7)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=1, color='red', linestyle='--', alpha=0.5)
        ax.set_title('Value Ranges (Min to Max)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Scaled Value')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Plot 4: Percentage of zeros
        ax = axes[1, 1]
        summary_df.plot(x='Feature', y='Zeros (%)', kind='bar', ax=ax, 
                       color='purple', alpha=0.7)
        ax.set_title('Percentage of Zero Values', fontsize=12, fontweight='bold')
        ax.set_ylabel('Zeros (%)')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=90)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = self.summary_dir / 'summary_statistics.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved summary plot: {filepath}")
    
    def generate_all_plots(self):
        """
        Generate all visualizations.
        """
        print("\n" + "=" * 70)
        print("FEATURE VISUALIZATION PIPELINE")
        print("=" * 70)
        print(f"Raw CSV: {self.raw_csv_path}")
        print(f"Scaled CSV: {self.scaled_csv_path}")
        
        # Run analysis
        issues = self.analyze_scaling_issues()
        
        # Generate plots
        self.plot_time_series_by_group()
        self.plot_distributions()
        self.plot_correlation_heatmap()
        self.plot_outlier_detection()
        self.generate_summary_report()
        
        print("\n" + "=" * 70)
        print("✓ ALL VISUALIZATIONS COMPLETE")
        print("=" * 70)
        print(f"\nPlots saved in:")
        print(f"  • Time Series:    {self.time_series_dir}")
        print(f"  • Distributions:  {self.dist_dir}")
        print(f"  • Heatmaps:       {self.heatmap_dir}")
        print(f"  • Outliers:       {self.outlier_dir}")
        print(f"  • Summary:        {self.summary_dir}")


def main():
    """
    Main execution function.
    """
    import sys
    import glob
    
    if len(sys.argv) < 2:
        # Auto-detect latest CSV files
        print("No CSV file specified. Searching for latest files...")
        
        raw_files = sorted(glob.glob("output/raw/*.csv"))
        scaled_files = sorted(glob.glob("output/scaled/*.csv"))
        
        if not raw_files or not scaled_files:
            print("❌ No CSV files found in output/raw/ and output/scaled/")
            print("\nUsage: python plot_features.py [raw_csv_path] [scaled_csv_path]")
            return
        
        raw_csv = raw_files[-1]
        scaled_csv = scaled_files[-1]
        
        print(f"Using latest files:")
        print(f"  Raw:    {raw_csv}")
        print(f"  Scaled: {scaled_csv}")
    else:
        raw_csv = sys.argv[1]
        scaled_csv = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].replace('raw', 'scaled')
    
    # Create visualizer and generate plots
    visualizer = FeatureVisualizer(raw_csv, scaled_csv)
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()
