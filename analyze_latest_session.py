"""
Analyze latest recorded session CSV with trained mental health model.

This script avoids live webcam inference and instead:
1) Loads the latest CSV from output/scaled (or a user-provided CSV)
2) Builds window-level features (mean/std/max/min) exactly like training
3) Runs model prediction for each window
4) Reports per-window and overall pattern summary

Usage:
    python analyze_latest_session.py
    python analyze_latest_session.py --csv output/scaled/02_15_06_03.csv
    python analyze_latest_session.py --window-frames 150 --stride 30
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze latest session CSV using trained model"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file. If omitted, latest file from output/scaled is used.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="ml/mental_health_model.pkl",
        help="Path to trained model pickle",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="ml/model_metadata.json",
        help="Path to model metadata json",
    )
    parser.add_argument(
        "--window-frames",
        type=int,
        default=150,
        help="Frames per inference window",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=30,
        help="Window stride in frames",
    )
    return parser.parse_args()


def find_latest_csv() -> Path:
    candidate_dirs = [Path("output/scaled"), Path("output/raw")]

    csv_files: list[Path] = []
    for directory in candidate_dirs:
        if directory.exists():
            csv_files.extend([p for p in directory.glob("*.csv") if p.is_file()])

    if not csv_files:
        raise FileNotFoundError("No CSV files found in output/scaled or output/raw")

    return max(csv_files, key=lambda p: p.stat().st_mtime)


def load_model_and_metadata(model_path: str, metadata_path: str):
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    with open(metadata_path, "r") as metadata_file:
        metadata = json.load(metadata_file)

    feature_names = metadata["feature_names"]
    label_map = {int(k): v for k, v in metadata["label_map"].items()}
    return model, feature_names, label_map


def infer_base_features(feature_names: list[str]) -> list[str]:
    base_features: list[str] = []
    seen = set()
    for full_name in feature_names:
        if full_name.rsplit("_", 1)[-1] in {"mean", "std", "max", "min"}:
            base = full_name.rsplit("_", 1)[0]
        else:
            base = full_name

        if base not in seen:
            seen.add(base)
            base_features.append(base)

    return base_features


def build_window_features(
    frame_df: pd.DataFrame,
    feature_names: list[str],
    base_features: list[str],
) -> pd.DataFrame:
    agg: dict[str, float] = {}

    for base in base_features:
        if base in frame_df.columns:
            series = pd.to_numeric(frame_df[base], errors="coerce").fillna(0.0)
        else:
            series = pd.Series(np.zeros(len(frame_df), dtype=float))

        agg[f"{base}_mean"] = float(series.mean())
        agg[f"{base}_std"] = float(series.std())
        agg[f"{base}_max"] = float(series.max())
        agg[f"{base}_min"] = float(series.min())

    window_df = pd.DataFrame([agg])
    window_df = window_df.reindex(columns=feature_names, fill_value=0.0)
    return window_df


def make_window_starts(total_rows: int, window_frames: int, stride: int) -> list[int]:
    if total_rows <= window_frames:
        return [0]

    starts = list(range(0, total_rows - window_frames + 1, stride))
    last_possible = total_rows - window_frames
    if starts[-1] != last_possible:
        starts.append(last_possible)
    return starts


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("LATEST SESSION CSV ANALYSIS")
    print("=" * 70)

    model, feature_names, label_map = load_model_and_metadata(
        args.model_path, args.metadata_path
    )
    base_features = infer_base_features(feature_names)

    csv_path = Path(args.csv) if args.csv else find_latest_csv()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Model: {args.model_path}")
    print(f"Metadata: {args.metadata_path}")
    print(f"CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    numeric_df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    total_rows = len(numeric_df)

    if total_rows == 0:
        raise ValueError("CSV contains no rows")

    window_frames = max(1, args.window_frames)
    stride = max(1, args.stride)
    starts = make_window_starts(total_rows, window_frames, stride)

    print(f"Rows: {total_rows}")
    print(f"Window size: {window_frames} frames")
    print(f"Stride: {stride} frames")
    print(f"Windows: {len(starts)}")

    predictions = []
    all_probs = []

    for index, start in enumerate(starts, start=1):
        end = min(start + window_frames, total_rows)
        window_slice = numeric_df.iloc[start:end]

        window_features = build_window_features(
            window_slice,
            feature_names,
            base_features,
        )

        probs = model.predict_proba(window_features)[0]
        pred = int(np.argmax(probs))
        conf = float(probs[pred])

        predictions.append(pred)
        all_probs.append(probs)

        print(
            f"Window {index:02d} [{start}:{end}] -> "
            f"{label_map[pred]} ({conf:.1%})"
        )

    probs_matrix = np.array(all_probs)
    mean_probs = probs_matrix.mean(axis=0)

    vote_counts = pd.Series(predictions).value_counts().sort_index()
    dominant_by_vote = int(vote_counts.idxmax())
    dominant_by_mean = int(np.argmax(mean_probs))

    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)
    print(f"Dominant (vote): {label_map[dominant_by_vote]} ({vote_counts.max()}/{len(predictions)} windows)")
    print(f"Dominant (mean prob): {label_map[dominant_by_mean]} ({mean_probs[dominant_by_mean]:.1%})")

    print("\nAverage probability by class:")
    for class_idx in np.argsort(mean_probs)[::-1]:
        print(f"  - {label_map[int(class_idx)]:20s}: {mean_probs[int(class_idx)]:.1%}")


if __name__ == "__main__":
    main()
