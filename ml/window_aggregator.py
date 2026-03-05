"""
Window Aggregator

Converts frame-by-frame CSV data into 5-second behavioral windows.
Each window becomes a single training sample with aggregated features.
"""

import pandas as pd
import numpy as np


class WindowAggregator:
    """
    Aggregates frame-level features into fixed-duration windows.
    """

    def __init__(self, window_duration=5.0, fps=30):
        """
        Parameters
        ----------
        window_duration : float
            Window size in seconds (default: 5.0)
        fps : int
            Frames per second (default: 30)
        """
        self.window_duration = window_duration
        self.fps = fps
        self.frames_per_window = int(window_duration * fps)

    def aggregate_csv(self, csv_path, output_path=None, label=None):
        """
        Convert frame-level CSV to window-level dataset.

        Parameters
        ----------
        csv_path : str
            Path to input CSV with frame-level features
        output_path : str, optional
            Path to save windowed dataset
        label : int or str, optional
            Class label to assign (e.g., 0=Healthy, 1=Depression)

        Returns
        -------
        pd.DataFrame
            Windowed dataset with aggregated features
        """
        print(f"\n{'='*60}")
        print(f"WINDOW AGGREGATION: {csv_path}")
        print(f"{'='*60}")

        # Load data
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} frames")

        # Remove timestamp column if present
        feature_cols = [col for col in df.columns if col != 'timestamp']
        print(f"✓ Found {len(feature_cols)} features")

        # Calculate number of complete windows
        num_windows = len(df) // self.frames_per_window
        print(f"✓ Creating {num_windows} windows ({self.window_duration}s each)")

        if num_windows == 0:
            print(f"✗ Error: Not enough frames for a single window")
            print(f"  Need: {self.frames_per_window} frames, Got: {len(df)} frames")
            return None

        # Aggregate windows
        windows = []

        for i in range(num_windows):
            start_idx = i * self.frames_per_window
            end_idx = start_idx + self.frames_per_window

            window_data = df.iloc[start_idx:end_idx][feature_cols]

            # Compute aggregated features
            window_features = {}

            for col in feature_cols:
                window_features[f'{col}_mean'] = window_data[col].mean()
                window_features[f'{col}_std'] = window_data[col].std()
                window_features[f'{col}_max'] = window_data[col].max()
                window_features[f'{col}_min'] = window_data[col].min()

            # Add label if provided
            if label is not None:
                window_features['label'] = label

            windows.append(window_features)

        # Convert to DataFrame
        windowed_df = pd.DataFrame(windows)

        print(f"✓ Generated {len(windowed_df)} window samples")
        print(f"✓ Total features per window: {len(windowed_df.columns) - (1 if label is not None else 0)}")

        # Save if requested
        if output_path:
            windowed_df.to_csv(output_path, index=False)
            print(f"✓ Saved to: {output_path}")

        return windowed_df


def aggregate_multiple_sessions(session_configs, output_path, window_duration=5.0, fps=30):
    """
    Aggregate multiple recording sessions into single dataset.

    Parameters
    ----------
    session_configs : list of dict
        Each dict contains {'csv_path': str, 'label': int/str}
    output_path : str
        Path to save combined dataset
    window_duration : float
        Window size in seconds
    fps : int
        Frames per second

    Returns
    -------
    pd.DataFrame
        Combined windowed dataset

    Example
    -------
    >>> sessions = [
    ...     {'csv_path': 'baseline.csv', 'label': 0},
    ...     {'csv_path': 'depression.csv', 'label': 1},
    ...     {'csv_path': 'anxiety.csv', 'label': 2}
    ... ]
    >>> dataset = aggregate_multiple_sessions(sessions, 'training_data.csv')
    """
    aggregator = WindowAggregator(window_duration=window_duration, fps=fps)
    all_windows = []

    print(f"\n{'='*60}")
    print(f"MULTI-SESSION AGGREGATION")
    print(f"{'='*60}")
    print(f"Sessions to process: {len(session_configs)}\n")

    for i, config in enumerate(session_configs, 1):
        csv_path = config['csv_path']
        label = config.get('label')

        print(f"[{i}/{len(session_configs)}] Processing: {csv_path}")

        windowed = aggregator.aggregate_csv(csv_path, label=label)

        if windowed is not None:
            all_windows.append(windowed)

    if not all_windows:
        print("✗ Error: No windows generated")
        return None

    # Combine all sessions
    combined_df = pd.concat(all_windows, ignore_index=True)

    print(f"\n{'='*60}")
    print(f"COMBINED DATASET")
    print(f"{'='*60}")
    print(f"Total samples: {len(combined_df)}")

    if 'label' in combined_df.columns:
        print("\nClass distribution:")
        print(combined_df['label'].value_counts().sort_index())

    # Save
    combined_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved combined dataset: {output_path}")

    return combined_df


if __name__ == "__main__":
    """
    Example usage: Aggregate baseline data
    """
    aggregator = WindowAggregator(window_duration=5.0, fps=30)

    # Process baseline recording
    baseline_windowed = aggregator.aggregate_csv(
        csv_path='output/scaled/23_51_05_03.csv',
        output_path='ml/baseline_windows.csv',
        label=0  # 0 = Healthy/Baseline
    )

    print(f"\n✓ Baseline aggregation complete")
    print(f"  Windows created: {len(baseline_windowed)}")
    print(f"\nNext step: Run baseline_stats.py to analyze distributions")
