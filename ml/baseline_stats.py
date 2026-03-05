"""
Baseline Statistics Analyzer

Analyzes baseline behavioral data to compute feature distributions.
These distributions represent "normal" behavior and serve as reference
for generating disorder-modified synthetic data.
"""

import pandas as pd
import numpy as np
import json


class BaselineAnalyzer:
    """
    Computes statistical distributions from baseline behavioral data.
    """

    def __init__(self):
        self.stats = {}

    def analyze_csv(self, csv_path, save_path=None):
        """
        Analyze baseline CSV and compute feature statistics.

        Parameters
        ----------
        csv_path : str
            Path to windowed baseline CSV
        save_path : str, optional
            Path to save statistics as JSON

        Returns
        -------
        dict
            Feature statistics (mean, std, min, max)
        """
        print(f"\n{'='*60}")
        print(f"BASELINE STATISTICAL ANALYSIS")
        print(f"{'='*60}")

        # Load data
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} samples")

        # Remove label column if present
        feature_cols = [col for col in df.columns if col != 'label']
        print(f"✓ Analyzing {len(feature_cols)} features")

        # Compute statistics for each feature
        print("\nComputing distributions...")

        for col in feature_cols:
            self.stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median()),
                'q25': float(df[col].quantile(0.25)),
                'q75': float(df[col].quantile(0.75))
            }

        print(f"✓ Statistics computed for {len(self.stats)} features")

        # Save if requested
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(self.stats, f, indent=2)
            print(f"✓ Saved statistics: {save_path}")

        # Print sample statistics
        print(f"\n{'='*60}")
        print("SAMPLE FEATURE DISTRIBUTIONS")
        print(f"{'='*60}")

        # Show first 5 features as examples
        for i, (feature, stats) in enumerate(list(self.stats.items())[:5]):
            print(f"\n{feature}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std:  {stats['std']:.4f}")
            print(f"  Min:  {stats['min']:.4f}")
            print(f"  Max:  {stats['max']:.4f}")

        if len(self.stats) > 5:
            print(f"\n... and {len(self.stats) - 5} more features")

        return self.stats

    def load_stats(self, json_path):
        """
        Load pre-computed statistics from JSON.

        Parameters
        ----------
        json_path : str
            Path to statistics JSON file

        Returns
        -------
        dict
            Feature statistics
        """
        with open(json_path, 'r') as f:
            self.stats = json.load(f)

        print(f"✓ Loaded statistics for {len(self.stats)} features")
        return self.stats

    def get_feature_stats(self, feature_name):
        """
        Get statistics for a specific feature.

        Parameters
        ----------
        feature_name : str
            Feature name

        Returns
        -------
        dict
            Statistics for that feature
        """
        return self.stats.get(feature_name)

    def print_summary(self):
        """
        Print summary of baseline statistics.
        """
        print(f"\n{'='*60}")
        print("BASELINE STATISTICS SUMMARY")
        print(f"{'='*60}")
        print(f"Total features: {len(self.stats)}")

        # Compute overall statistics
        all_means = [s['mean'] for s in self.stats.values()]
        all_stds = [s['std'] for s in self.stats.values()]

        print(f"\nOverall distribution:")
        print(f"  Mean range: [{min(all_means):.4f}, {max(all_means):.4f}]")
        print(f"  Std range:  [{min(all_stds):.4f}, {max(all_stds):.4f}]")

        # Count features by aggregation type
        aggregation_types = {'mean': 0, 'std': 0, 'max': 0, 'min': 0}

        for feature in self.stats.keys():
            for agg_type in aggregation_types.keys():
                if f'_{agg_type}' in feature:
                    aggregation_types[agg_type] += 1

        print(f"\nFeatures by aggregation type:")
        for agg_type, count in aggregation_types.items():
            print(f"  {agg_type}: {count}")


if __name__ == "__main__":
    """
    Example usage: Analyze baseline windows
    """
    analyzer = BaselineAnalyzer()

    # Analyze baseline data
    stats = analyzer.analyze_csv(
        csv_path='ml/baseline_windows.csv',
        save_path='ml/baseline_stats.json'
    )

    # Print summary
    analyzer.print_summary()

    print(f"\n✓ Baseline analysis complete")
    print(f"\nNext step: Run synthetic_generator.py to create training dataset")
