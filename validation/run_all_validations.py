"""
Master validation script - runs all feature comparisons.

Compares MediaPipe vs OpenFace features:
1. AU12 (smile)
2. Head Yaw (pose)
3. Eye Aspect Ratio / Blink (AU45)
4. Expressivity (AU variance)

Usage:
    python run_all_validations.py
"""

import subprocess
import sys

print("="*70)
print("FACIAL FEATURE VALIDATION SUITE")
print("MediaPipe vs OpenFace Comparison")
print("="*70)

validation_scripts = [
    ("AU12 (Smile Intensity)", "compare_au12.py"),
    ("Head Yaw (Pose)", "compare_yaw.py"),
    ("Eye Aspect Ratio / Blink", "compare_ear.py"),
    ("Expressivity (Facial Animation)", "compare_expressivity.py"),
]

results = []

for name, script in validation_scripts:
    print(f"\n\n{'='*70}")
    print(f"Running: {name}")
    print(f"Script: {script}")
    print("="*70)
    
    try:
        result = subprocess.run(
            [sys.executable, script],
            capture_output=False,
            text=True,
            check=True
        )
        results.append((name, "✅ PASSED"))
        print(f"\n✅ {name} validation complete!")
    except subprocess.CalledProcessError as e:
        results.append((name, "❌ FAILED"))
        print(f"\n❌ {name} validation failed!")
    except FileNotFoundError:
        results.append((name, "⚠️  SCRIPT NOT FOUND"))
        print(f"\n⚠️  Script not found: {script}")

# Summary
print("\n\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)

for name, status in results:
    print(f"{status}  {name}")

print("\n" + "="*70)
print("All validations complete!")
print("="*70)
