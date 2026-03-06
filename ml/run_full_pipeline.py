"""
Complete ML Training Pipeline

Orchestrates the entire machine learning workflow:
1. Window aggregation
2. Baseline statistics
3. Synthetic dataset generation
4. XGBoost training
5. Model evaluation

Usage:
    python ml/run_full_pipeline.py
"""

import os
from glob import glob


def find_default_baseline_csv():
    candidates = sorted(glob('output/scaled/*.csv'), key=os.path.getmtime)
    if not candidates:
        raise FileNotFoundError('No CSV found in output/scaled/')
    return candidates[-1]


def run_full_pipeline():
    """
    Execute complete ML pipeline.
    """
    print("\n" + "="*70)
    print(" " * 15 + "MENTAL HEALTH ML PIPELINE")
    print("="*70)
    print("\nThis will:")
    print("  1. Aggregate your baseline data into 5-second windows")
    print("  2. Analyze baseline behavioral distributions")
    print("  3. Generate synthetic training samples for 6 classes")
    print("  4. Train XGBoost classifier")
    print("  5. Evaluate model performance")
    print("\nEstimated time: 2-5 minutes")
    print("="*70)

    input("\nPress ENTER to start...")

    # Step 1: Window Aggregation
    print("\n" + "="*70)
    print("STEP 1/5: Window Aggregation")
    print("="*70)
    
    from ml.window_aggregator import WindowAggregator
    
    baseline_csv = os.getenv('BASELINE_CSV', find_default_baseline_csv())
    print(f"Using baseline CSV: {baseline_csv}")

    aggregator = WindowAggregator(window_duration=5.0, fps=30)
    baseline_windowed = aggregator.aggregate_csv(
        csv_path=baseline_csv,
        output_path='ml/baseline_windows.csv',
        label=None
    )

    if baseline_windowed is None or len(baseline_windowed) == 0:
        print("\n✗ Error: Window aggregation failed")
        return

    print(f"\n✓ Step 1 Complete: {len(baseline_windowed)} baseline windows created")

    # Step 2: Baseline Statistics
    print("\n" + "="*70)
    print("STEP 2/5: Baseline Statistical Analysis")
    print("="*70)

    from ml.baseline_stats import BaselineAnalyzer

    analyzer = BaselineAnalyzer()
    stats = analyzer.analyze_csv(
        csv_path='ml/baseline_windows.csv',
        save_path='ml/baseline_stats.json'
    )

    print(f"\n✓ Step 2 Complete: Baseline distributions computed")

    # Step 3: Synthetic Dataset Generation
    print("\n" + "="*70)
    print("STEP 3/5: Synthetic Dataset Generation")
    print("="*70)

    from synthetic_generator import SyntheticGenerator

    generator = SyntheticGenerator('ml/baseline_stats.json')
    dataset = generator.generate_full_dataset(
        samples_per_class=2000,
        output_path='ml/training_dataset.csv'
    )

    print(f"\n✓ Step 3 Complete: {len(dataset)} training samples generated")

    # Step 4: XGBoost Training
    print("\n" + "="*70)
    print("STEP 4/5: XGBoost Training")
    print("="*70)

    from train_model import MentalHealthClassifier

    classifier = MentalHealthClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05
    )

    results = classifier.train('ml/training_dataset.csv', test_size=0.2)

    # Feature importance
    importance_df = classifier.get_feature_importance(top_n=20)
    importance_df.to_csv('ml/feature_importance.csv', index=False)

    # Save model
    classifier.save_model(
        model_path='ml/mental_health_model.pkl',
        metadata_path='ml/model_metadata.json'
    )

    print(f"\n✓ Step 4 Complete: Model trained and saved")

    # Step 5: Evaluation
    print("\n" + "="*70)
    print("STEP 5/5: Model Evaluation")
    print("="*70)

    from evaluate_model import ModelEvaluator

    evaluator = ModelEvaluator(
        model_path='ml/mental_health_model.pkl',
        metadata_path='ml/model_metadata.json',
        dataset_path='ml/training_dataset.csv'
    )

    evaluator.generate_full_report(output_dir='ml/evaluation')

    print(f"\n✓ Step 5 Complete: Evaluation report generated")

    # Final Summary
    print("\n" + "="*70)
    print(" " * 20 + "PIPELINE COMPLETE!")
    print("="*70)
    print("\n📊 Generated Files:")
    print(f"  ├─ ml/baseline_windows.csv         ({len(baseline_windowed)} windows)")
    print(f"  ├─ ml/baseline_stats.json          (Feature distributions)")
    print(f"  ├─ ml/training_dataset.csv         ({len(dataset)} samples)")
    print(f"  ├─ ml/mental_health_model.pkl      (Trained model)")
    print(f"  ├─ ml/model_metadata.json          (Model info)")
    print(f"  ├─ ml/feature_importance.csv       (Feature rankings)")
    print(f"  └─ ml/evaluation/                  (Plots & reports)")

    print("\n📈 Model Performance:")
    print(f"  ├─ Accuracy:  {results['accuracy']:.2%}")
    print(f"  ├─ Precision: {results['precision']:.2%}")
    print(f"  ├─ Recall:    {results['recall']:.2%}")
    print(f"  └─ F1 Score:  {results['f1']:.2%}")

    print("\n🎯 Next Steps:")
    print("  1. Review evaluation plots in ml/evaluation/")
    print("  2. [Optional] Record 2-min acted sessions for each disorder")
    print("  3. [Optional] Replace synthetic patterns with real patterns")
    print("  4. Deploy model for real-time prediction")

    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        run_full_pipeline()
    except KeyboardInterrupt:
        print("\n\n✗ Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
