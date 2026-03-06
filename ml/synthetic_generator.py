"""
Synthetic Dataset Generator

Generates synthetic behavioral samples by sampling from modified distributions.
Creates balanced training dataset for XGBoost classification.
"""

import pandas as pd
import numpy as np
import json
from disorder_profiles import DISORDER_LABELS, apply_disorder_modifiers


class SyntheticGenerator:
    """
    Generates synthetic training samples from baseline distributions.
    """

    def __init__(self, baseline_stats_path):
        """
        Parameters
        ----------
        baseline_stats_path : str
            Path to baseline statistics JSON
        """
        with open(baseline_stats_path, 'r') as f:
            self.baseline_stats = json.load(f)

        print(f"✓ Loaded baseline statistics for {len(self.baseline_stats)} features")

    def generate_samples(self, disorder_name, num_samples, label=None):
        """
        Generate synthetic samples for a specific disorder.

        Parameters
        ----------
        disorder_name : str
            One of DISORDER_LABELS keys in disorder_profiles.py
        num_samples : int
            Number of samples to generate
        label : int, optional
            Class label (if None, uses DISORDER_LABELS)

        Returns
        -------
        pd.DataFrame
            Generated samples
        """
        print(f"\nGenerating {num_samples} samples for: {disorder_name}")

        # Get label
        if label is None:
            label = DISORDER_LABELS[disorder_name]

        # Apply disorder modifiers for requested condition
        stats = apply_disorder_modifiers(self.baseline_stats, disorder_name)

        # Generate samples
        samples = []

        for i in range(num_samples):
            sample = {}

            for feature, feature_stats in stats.items():
                mean = feature_stats['mean']
                std = feature_stats['std']
                min_val = feature_stats['min']
                max_val = feature_stats['max']

                # Sample from normal distribution
                value = np.random.normal(mean, std)

                # Clip to realistic range
                value = np.clip(value, min_val, max_val)

                sample[feature] = value

            # Add label
            sample['label'] = label

            samples.append(sample)

        df = pd.DataFrame(samples)
        print(f"  ✓ Generated {len(df)} samples (label={label})")

        return df

    def generate_full_dataset(self, samples_per_class=2000, output_path=None):
        """
        Generate complete balanced dataset for all disorders.

        Parameters
        ----------
        samples_per_class : int
            Number of samples per disorder class
        output_path : str, optional
            Path to save dataset

        Returns
        -------
        pd.DataFrame
            Full training dataset
        """
        print(f"\n{'='*60}")
        print("SYNTHETIC DATASET GENERATION")
        print(f"{'='*60}")
        print(f"Samples per class: {samples_per_class}")
        print(f"Total classes: {len(DISORDER_LABELS)}")
        print(f"Expected total: {samples_per_class * len(DISORDER_LABELS)}")

        all_samples = []

        # Generate samples for each disorder
        for disorder in DISORDER_LABELS.keys():
            samples = self.generate_samples(disorder, samples_per_class)
            all_samples.append(samples)

        # Combine all samples
        full_dataset = pd.concat(all_samples, ignore_index=True)

        # Shuffle dataset
        full_dataset = full_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"\n{'='*60}")
        print("DATASET SUMMARY")
        print(f"{'='*60}")
        print(f"Total samples: {len(full_dataset)}")
        print(f"Features: {len(full_dataset.columns) - 1}")

        print(f"\nClass distribution:")
        class_counts = full_dataset['label'].value_counts().sort_index()
        for label, count in class_counts.items():
            # Find disorder name
            disorder_name = [k for k, v in DISORDER_LABELS.items() if v == label][0]
            print(f"  [{label}] {disorder_name}: {count}")

        # Save if requested
        if output_path:
            full_dataset.to_csv(output_path, index=False)
            print(f"\n✓ Saved dataset: {output_path}")

        return full_dataset


def generate_from_real_session(csv_path, disorder_name, window_duration=5.0, fps=30):
    """
    Extract behavioral patterns from a real acted session.
    Alternative to synthetic generation - uses actual recording.

    Parameters
    ----------
    csv_path : str
        Path to frame-level CSV from acted session
    disorder_name : str
        Disorder being acted
    window_duration : float
        Window size in seconds
    fps : int
        Frames per second

    Returns
    -------
    dict
        Feature statistics extracted from real performance

    Example
    -------
    >>> # After acting depressed for 2 minutes
    >>> real_patterns = generate_from_real_session(
    ...     'output/scaled/depression_session.csv', 
    ...     'depression'
    ... )
    >>> # Use these patterns instead of synthetic
    """
    print(f"\n{'='*60}")
    print(f"EXTRACTING REAL PATTERNS: {disorder_name}")
    print(f"{'='*60}")

    from ml.window_aggregator import WindowAggregator

    # Create windows from session
    aggregator = WindowAggregator(window_duration=window_duration, fps=fps)
    windowed = aggregator.aggregate_csv(csv_path)

    if windowed is None or len(windowed) == 0:
        print("✗ Error: No windows generated")
        return None

    # Remove label if present
    feature_cols = [col for col in windowed.columns if col != 'label']

    # Compute statistics from this real session
    real_stats = {}

    for col in feature_cols:
        real_stats[col] = {
            'mean': float(windowed[col].mean()),
            'std': float(windowed[col].std()),
            'min': float(windowed[col].min()),
            'max': float(windowed[col].max()),
            'median': float(windowed[col].median()),
            'q25': float(windowed[col].quantile(0.25)),
            'q75': float(windowed[col].quantile(0.75))
        }

    print(f"✓ Extracted statistics from {len(windowed)} windows")
    print(f"✓ Real behavioral patterns for '{disorder_name}' captured")

    # Save real patterns
    save_path = f'ml/real_patterns_{disorder_name}.json'
    with open(save_path, 'w') as f:
        json.dump(real_stats, f, indent=2)
    print(f"✓ Saved real patterns: {save_path}")

    return real_stats


def generate_hybrid_dataset(baseline_stats_path, real_patterns_paths, 
                           samples_per_class=6000, output_path=None):
    """
    Generate dataset mixing synthetic and real patterns.

    Parameters
    ----------
    baseline_stats_path : str
        Path to baseline statistics
    real_patterns_paths : dict
        Maps disorder_name -> path to real pattern JSON
        e.g., {'depression': 'ml/real_patterns_depression.json'}
    samples_per_class : int
        Samples per class
    output_path : str, optional
        Save path

    Returns
    -------
    pd.DataFrame
        Hybrid dataset

    Example
    -------
    >>> # Mix synthetic and real patterns
    >>> dataset = generate_hybrid_dataset(
    ...     'ml/baseline_stats.json',
    ...     {'depression': 'ml/real_patterns_depression.json'},
    ...     samples_per_class=2000
    ... )
    """
    print(f"\n{'='*60}")
    print("HYBRID DATASET GENERATION")
    print(f"{'='*60}")
    print(f"Using real patterns for: {list(real_patterns_paths.keys())}")
    print(f"Using synthetic for remaining disorders")

    generator = SyntheticGenerator(baseline_stats_path)
    all_samples = []

    for disorder in DISORDER_LABELS.keys():
        label = DISORDER_LABELS[disorder]

        # Check if we have real patterns for this disorder
        if disorder in real_patterns_paths:
            print(f"\n[Real] {disorder}")
            # Load real patterns
            with open(real_patterns_paths[disorder], 'r') as f:
                real_stats = json.load(f)

            # Generate samples from real patterns
            samples = []
            for i in range(samples_per_class):
                sample = {}
                for feature, stats in real_stats.items():
                    mean = stats['mean']
                    std = stats['std']
                    min_val = stats['min']
                    max_val = stats['max']

                    value = np.random.normal(mean, std)
                    value = np.clip(value, min_val, max_val)
                    sample[feature] = value

                sample['label'] = label
                samples.append(sample)

            df = pd.DataFrame(samples)
            print(f"  ✓ Generated {len(df)} samples from real patterns")

        else:
            # Use synthetic generation
            print(f"\n[Synthetic] {disorder}")
            df = generator.generate_samples(disorder, samples_per_class)

        all_samples.append(df)

    # Combine and shuffle
    full_dataset = pd.concat(all_samples, ignore_index=True)
    full_dataset = full_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"Total samples: {len(full_dataset)}")
    print(f"Features: {len(full_dataset.columns) - 1}")

    if output_path:
        full_dataset.to_csv(output_path, index=False)
        print(f"✓ Saved hybrid dataset: {output_path}")

    return full_dataset


if __name__ == "__main__":
    """
    Example usage: Generate synthetic dataset
    """
    # Generate fully synthetic dataset
    generator = SyntheticGenerator('ml/baseline_stats.json')

    dataset = generator.generate_full_dataset(
        samples_per_class=2000,
        output_path='ml/training_dataset.csv'
    )

    print(f"\n✓ Synthetic dataset generation complete")
    print(f"\nNext step: Run train_model.py to train XGBoost classifier")
