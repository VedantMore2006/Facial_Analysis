"""
Acting Session Helper

Guides recording for requested mental-health condition names and
extracts real patterns for hybrid dataset generation.
"""

import os
import sys

from disorder_profiles import DISORDER_LABELS, CONDITION_IMPORTANCE_PRIORS, DISORDER_PROFILES


DISORDER_INSTRUCTIONS = {
    'depression': {
        'duration': 120,
        'instructions': """
DEPRESSION GUIDE (2 minutes)
- Flat/low facial expression
- Reduced smile and gesture frequency
- More downward gaze, less eye contact
- Slower response timing
"""
    },
    'anxiety': {
        'duration': 120,
        'instructions': """
ANXIETY GUIDE (2 minutes)
- Frequent blinking
- Brow tension / lip compression
- Frequent gaze shifts
- Fast but inconsistent reaction timing
"""
    },
    'stress': {
        'duration': 120,
        'instructions': """
STRESS GUIDE (2 minutes)
- Sustained tension in face
- Moderate fidgeting and micro-movements
- Faster gaze changes
- Slightly delayed but variable response timing
"""
    },
    'bipolar_mania': {
        'duration': 120,
        'instructions': """
BIPOLAR MANIA GUIDE (2 minutes)
- High expressivity and gesture rate
- Rapid head and face transitions
- High energy throughout
- Fast response timing
"""
    },
    'phobia_common': {
        'duration': 120,
        'instructions': """
PHOBIA (COMMON) GUIDE (2 minutes)
- Vigilant scanning gaze
- Elevated blink and tension
- Avoid sustained eye contact
- Reactive facial shifts
"""
    },
    'suicidal_tendency': {
        'duration': 120,
        'instructions': """
SUICIDAL TENDENCY GUIDE (2 minutes)
- Low affect and reduced movement
- Reduced eye contact with more downward gaze
- Longer pauses and delayed responses
- Do NOT imitate any self-harm actions
"""
    }
}


def print_condition_priors(disorder_name):
    priors = CONDITION_IMPORTANCE_PRIORS.get(disorder_name, {})
    profile = DISORDER_PROFILES.get(disorder_name, {})

    print("\nTop prior feature importances (design priors):")
    for feature, weight in sorted(priors.items(), key=lambda item: item[1], reverse=True):
        print(f"  - {feature}: {weight:.2f}")

    print("\nKey multiplier values in current profile:")
    for feature, mods in list(profile.get('modifiers', {}).items())[:12]:
        print(f"  - {feature}: mean_mult={mods['mean_mult']:.2f}, std_mult={mods['std_mult']:.2f}")


def record_session(disorder_name):
    if disorder_name not in DISORDER_INSTRUCTIONS:
        print(f"✗ Unknown disorder: {disorder_name}")
        print(f"  Valid options: {', '.join(DISORDER_INSTRUCTIONS.keys())}")
        return

    label = DISORDER_LABELS[disorder_name]
    info = DISORDER_INSTRUCTIONS[disorder_name]

    print("\n" + "=" * 70)
    print(f"RECORDING ACTED SESSION: {disorder_name.upper()} (label={label})")
    print("=" * 70)
    print(info['instructions'])
    print_condition_priors(disorder_name)

    print("\nManual recording flow:")
    print("  1. Run: python run_pipeline.py")
    print(f"  2. Act as {disorder_name} for ~{info['duration']//60} minutes")
    print("  3. Press ESC")
    print("  4. Take newest CSV from output/scaled/")

    print("\nThen extract pattern with:")
    print("python -c \"from ml.synthetic_generator import generate_from_real_session; "
          f"generate_from_real_session('output/scaled/YOUR_FILE.csv','{disorder_name}')\"")


def check_recorded_sessions():
    print("\n" + "=" * 70)
    print("RECORDED ACTING SESSIONS STATUS")
    print("=" * 70)

    for disorder in DISORDER_INSTRUCTIONS.keys():
        pattern_file = f'ml/real_patterns_{disorder}.json'
        status = "Real patterns available" if os.path.exists(pattern_file) else "Not recorded yet"
        print(f"{'✓' if os.path.exists(pattern_file) else '✗'} {disorder:20s} - {status}")


def generate_hybrid_dataset_now():
    print("\n" + "=" * 70)
    print("GENERATING HYBRID DATASET")
    print("=" * 70)

    real_patterns = {}
    for disorder in DISORDER_INSTRUCTIONS.keys():
        pattern_file = f'ml/real_patterns_{disorder}.json'
        if os.path.exists(pattern_file):
            real_patterns[disorder] = pattern_file
            print(f"✓ Using real patterns for: {disorder}")

    if not real_patterns:
        print("\n✗ No real patterns found. Record at least one session first.")
        return

    from synthetic_generator import generate_hybrid_dataset

    dataset = generate_hybrid_dataset(
        baseline_stats_path='ml/baseline_stats.json',
        real_patterns_paths=real_patterns,
        samples_per_class=6000,
        output_path='ml/training_dataset_hybrid.csv'
    )

    print("\n✓ Hybrid dataset created: ml/training_dataset_hybrid.csv")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Real classes: {len(real_patterns)}")
    print(f"  Synthetic classes: {len(DISORDER_INSTRUCTIONS) - len(real_patterns)}")
    print("\nRetrain:")
    print("python -c \"from ml.train_model import MentalHealthClassifier; c=MentalHealthClassifier(); c.train('ml/training_dataset_hybrid.csv'); c.save_model('ml/mental_health_model.pkl','ml/model_metadata.json')\"")


def main():
    if len(sys.argv) < 2:
        print("\nUsage: python ml/record_acted_session.py [command]")
        print("Commands:")
        for disorder in DISORDER_INSTRUCTIONS.keys():
            print(f"  {disorder}")
        print("  status")
        print("  hybrid")
        check_recorded_sessions()
        return

    command = sys.argv[1].lower()

    if command == 'status':
        check_recorded_sessions()
    elif command == 'hybrid':
        generate_hybrid_dataset_now()
    elif command in DISORDER_INSTRUCTIONS:
        record_session(command)
    else:
        print(f"✗ Unknown command: {command}")
        print(f"  Valid: {', '.join(list(DISORDER_INSTRUCTIONS.keys()) + ['status', 'hybrid'])}")


if __name__ == "__main__":
    main()
