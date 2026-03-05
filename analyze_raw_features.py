"""
Raw Feature CSV Analysis
========================
Deep dive analysis of raw feature values to understand:
1. What each feature represents
2. Why certain values are zero
3. Statistical patterns and distributions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob

def analyze_csv(csv_path):
    """Load and analyze the raw CSV file."""
    df = pd.read_csv(csv_path)
    return df

def print_separator(title):
    """Print a formatted section separator."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def analyze_feature_categories(df):
    """Analyze features by category."""
    features = df.columns[1:]  # Skip timestamp
    
    categories = {
        'Facial Action Units (AU)': [
            'AU12Mean', 'AU12Variance', 'AU12ActivationFrequency',
            'AU15MeanAmplitude', 'AU4MeanActivation', 'AU4DurationRatio',
            'AU1AU2PeakIntensity', 'AU20ActivationRate'
        ],
        'Lip & Mouth Features': [
            'LipCompressionFrequency'
        ],
        'Blink & Eye Metrics': [
            'BlinkRate', 'BlinkClusterDensity', 'BaselineEyeOpenness'
        ],
        'Gaze & Eye Contact': [
            'GazeShiftFrequency', 'EyeContactRatio', 'DownwardGazeFrequency'
        ],
        'Head Motion & Velocity': [
            'MeanHeadVelocity', 'HeadVelocityPeak', 'HeadMotionEnergy',
            'LandmarkDisplacementMean', 'PostureRigidityIndex'
        ],
        'Facial Dynamics': [
            'OverallAUVariance', 'FacialEmotionalRange', 'FacialTransitionFrequency',
            'NearZeroAUActivationRatio'
        ],
        'Motion & Gesture': [
            'MotionEnergyFloorScore', 'GestureFrequency', 'MicroMotionEnergy',
            'ShoulderElevationIndex'
        ],
        'Temporal & Response': [
            'ResponseLatencyMean', 'SpeechOnsetDelay', 'NodOnsetLatency',
            'PauseDurationMean', 'ExtendedSilenceRatio', 'ReactionTimeInstabilityIndex'
        ]
    }
    
    return categories

def explain_features():
    """Provide detailed explanation of each feature."""
    explanations = {
        'AU12Mean': 'Average cheek raiser (AU12) muscle activation. 0=relaxed, 1=strong smile.',
        'AU12Variance': 'Variability in AU12 activation over recent frames. 0=stable, 1=highly variable.',
        'AU12ActivationFrequency': 'How often AU12 crosses activation threshold. Converted to rate [0,1].',
        'AU15MeanAmplitude': 'Lip corner depressor amplitude. 0=none, 1=strong lateral lip pull.',
        'AU4MeanActivation': 'Brow lowerer (AU4) activation. 0=neutral, 1=strong frown.',
        'AU4DurationRatio': 'Proportion of time AU4 stays activated. 0=never, 1=always active.',
        'AU1AU2PeakIntensity': 'Peak intensity of inner/outer brow raising (AU1+AU2). 0=none, 1=maximum.',
        'AU20ActivationRate': 'Lip stretcher activation rate. 0=low frequency, 1=high frequency.',
        'LipCompressionFrequency': 'How often lips compress. Rate normalized to [0,1].',
        'BlinkRate': 'Eye blink frequency. 0=no blinks, 1=frequent blinking.',
        'BlinkClusterDensity': 'How clustered blinks are in time. 0=spread out, 1=rapid clusters.',
        'BaselineEyeOpenness': 'Baseline eye opening degree. 0=closed, 1=wide open.',
        'GazeShiftFrequency': 'How often eyes move to different locations. Rate normalized.',
        'EyeContactRatio': 'Proportion of time looking straight ahead. 0=never centered, 1=always centered.',
        'DownwardGazeFrequency': 'How often gaze points downward. Rate normalized.',
        'MeanHeadVelocity': 'Average head movement speed. 0=still, 1=moving fast.',
        'HeadVelocityPeak': 'Maximum head movement speed during window. 0=still, 1=very fast movement.',
        'HeadMotionEnergy': 'Total energy of head movement. 0=no motion, 1=high energy motion.',
        'LandmarkDisplacementMean': 'Average distance facial landmarks move. 0=no change, 1=large changes.',
        'PostureRigidityIndex': 'How rigid/stable head posture is. 0=flexible, 1=rigid.',
        'OverallAUVariance': 'Total variability across all AU activations. 0=stable, 1=highly variable.',
        'FacialEmotionalRange': 'Range of emotional expressions detected. 0=limited, 1=full range.',
        'FacialTransitionFrequency': 'How often facial expression changes. Rate normalized.',
        'NearZeroAUActivationRatio': 'Proportion of near-zero AU activations. 0=many active, 1=mostly inactive.',
        'MotionEnergyFloorScore': 'Baseline motion energy threshold. Low values indicate stable baseline.',
        'GestureFrequency': 'How often head/arm gestures occur. Rate normalized.',
        'MicroMotionEnergy': 'Subtle micro-expression energy. 0=none, 1=high micro-expression activity.',
        'ShoulderElevationIndex': 'Shoulder position elevation. 0=relaxed, 1=shrugged.',
        'ResponseLatencyMean': 'Avg time to respond to stimuli. Normalized by max expected latency.',
        'SpeechOnsetDelay': 'Time before mouth movement starts. Normalized by max expected delay.',
        'NodOnsetLatency': 'Time before head nod occurs. Normalized by max expected latency.',
        'PauseDurationMean': 'Average duration of motion pauses. Normalized by typical pause length.',
        'ExtendedSilenceRatio': 'Proportion of time in extended stillness.',
        'ReactionTimeInstabilityIndex': 'Variability in reaction times. 0=consistent, 1=highly variable.'
    }
    return explanations

def analyze_zeros(df):
    """Analyze why features have zero values."""
    print_separator("ZERO VALUE ANALYSIS")
    
    features = df.columns[1:]
    zero_info = {
        'features_with_zeros': [],
        'mostly_zeros': [],
        'sometimes_zeros': []
    }
    
    for feature in features:
        zero_count = (df[feature] == 0).sum()
        zero_pct = (zero_count / len(df)) * 100
        
        if zero_pct > 99:
            zero_info['mostly_zeros'].append((feature, zero_pct, zero_count))
        elif zero_pct > 50:
            zero_info['sometimes_zeros'].append((feature, zero_pct, zero_count))
        elif zero_pct > 0:
            zero_info['features_with_zeros'].append((feature, zero_pct, zero_count))
    
    # Print mostly zeros
    print("\n📊 FEATURES WITH MOSTLY ZEROS (>99% zeros):")
    print("-" * 80)
    for feature, pct, count in zero_info['mostly_zeros']:
        print(f"  {feature:30s} | {pct:6.2f}% zeros ({count:4d} frames)")
        if 'Blink' in feature:
            print(f"    └─ Reason: Blinks are rare events. Only 5-10% of time are humans blinking.")
        elif 'Gesture' in feature or 'Shoulder' in feature:
            print(f"    └─ Reason: Gestures are infrequent. Person was relatively still during recording.")
        elif 'Transition' in feature:
            print(f"    └─ Reason: Expression changes are sparse. Facial expressions were stable.")
        elif 'NearZero' in feature:
            print(f"    └─ Reason: Most facial muscles had some activation. Measure of steady state.")
    
    # Print sometimes zeros
    print("\n📊 FEATURES WITH SOME ZEROS (50-99% zeros):")
    print("-" * 80)
    for feature, pct, count in zero_info['sometimes_zeros']:
        print(f"  {feature:30s} | {pct:6.2f}% zeros ({count:4d} frames)")
        if 'Rate' in feature or 'Frequency' in feature:
            print(f"    └─ Reason: Activity-based feature. Zero when no activations detected.")
        elif any(x in feature for x in ['Compression', 'ClusterDensity', 'Shift']):
            print(f"    └─ Reason: Event-based feature. Zero when no events occur in window.")
    
    # Print occasional zeros
    print("\n📊 FEATURES WITH OCCASIONAL ZEROS (<50% zeros):")
    print("-" * 80)
    for feature, pct, count in zero_info['features_with_zeros'][:15]:  # Top 15
        print(f"  {feature:30s} | {pct:6.2f}% zeros ({count:4d} frames)")
    if len(zero_info['features_with_zeros']) > 15:
        print(f"  ... and {len(zero_info['features_with_zeros']) - 15} more features")

def analyze_distributions(df):
    """Analyze value distributions for each feature."""
    print_separator("FEATURE VALUE DISTRIBUTIONS")
    
    features = df.columns[1:]
    
    print("\nTop features by average activation level:")
    print("-" * 80)
    
    means = df[features].mean().sort_values(ascending=False)
    
    for i, (feature, mean_val) in enumerate(means.head(10).items(), 1):
        min_val = df[feature].min()
        max_val = df[feature].max()
        std_val = df[feature].std()
        non_zero = (df[feature] != 0).sum()
        non_zero_pct = (non_zero / len(df)) * 100
        
        print(f"\n{i}. {feature}")
        print(f"   Mean: {mean_val:.4f} | Std: {std_val:.4f} | Range: [{min_val:.4f}, {max_val:.4f}]")
        print(f"   Non-zero: {non_zero_pct:.1f}% ({non_zero} frames)")

def analyze_temporal_patterns(df):
    """Analyze how features change over time."""
    print_separator("TEMPORAL PATTERNS (HOW VALUES CHANGE OVER TIME)")
    
    features = df.columns[1:]
    
    # Calculate frame-to-frame changes
    changes = []
    
    for feature in features:
        # Skip features that are mostly zero (change calculation less meaningful)
        if (df[feature] == 0).sum() > len(df) * 0.8:
            continue
        
        diff = df[feature].diff().abs().mean()
        changes.append((feature, diff))
    
    changes.sort(key=lambda x: x[1], reverse=True)
    
    print("\n🔄 MOST VOLATILE FEATURES (Change frame-to-frame most):")
    print("-" * 80)
    for feature, change in changes[:10]:
        print(f"  {feature:30s} | Avg change per frame: {change:.6f}")
        print(f"    └─ Frame-to-frame variability is high (sensitive, non-smooth)")
    
    print("\n🔄 MOST STABLE FEATURES (Change frame-to-frame least):")
    print("-" * 80)
    for feature, change in changes[-10:]:
        print(f"  {feature:30s} | Avg change per frame: {change:.6f}")
        print(f"    └─ Frame-to-frame consistency is high (smooth, filtered)")

def analyze_correlations(df):
    """Analyze feature correlations."""
    print_separator("FEATURE CORRELATIONS & RELATIONSHIPS")
    
    features = df.columns[1:]
    
    # Calculate correlation matrix
    corr_matrix = df[features].corr()
    
    # Find strongest positive correlations
    print("\n🔗 STRONGLY CORRELATED FEATURES (r > 0.7):")
    print("-" * 80)
    
    found_pairs = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            feat_i = corr_matrix.columns[i]
            feat_j = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            
            if abs(corr_val) > 0.7 and (feat_i, feat_j) not in found_pairs:
                found_pairs.add((feat_i, feat_j))
                direction = "↑↑ Positive" if corr_val > 0 else "↓↑ Negative"
                print(f"  {feat_i:25s} ←→ {feat_j:25s} | r = {corr_val:+.3f} ({direction})")

def print_feature_summary(df):
    """Print comprehensive feature summary."""
    print_separator("ALL FEATURES SUMMARY")
    
    features = df.columns[1:]
    explanations = explain_features()
    
    for feature in features:
        mean_val = df[feature].mean()
        std_val = df[feature].std()
        min_val = df[feature].min()
        max_val = df[feature].max()
        zero_pct = (df[feature] == 0).sum() / len(df) * 100
        
        print(f"\n📌 {feature}")
        print(f"   Definition: {explanations.get(feature, 'N/A')}")
        print(f"   Statistics: μ={mean_val:.4f}, σ={std_val:.4f}, range=[{min_val:.4f}, {max_val:.4f}]")
        print(f"   Zeros: {zero_pct:.1f}% of frames")

def main():
    """Main analysis function."""
    # Find the latest raw CSV
    raw_files = glob.glob("output/raw/*.csv")
    if not raw_files:
        print("❌ No raw CSV files found in output/raw/")
        return
    
    csv_path = max(raw_files, key=lambda x: Path(x).stat().st_mtime)
    print(f"\n📂 Analyzing: {csv_path}")
    
    # Load and analyze
    df = analyze_csv(csv_path)
    
    print(f"\n📊 Dataset Overview:")
    print(f"   Total frames: {len(df)}")
    print(f"   Total features: {len(df.columns) - 1}")
    print(f"   Time range: {df['timestamp'].min():.2f} to {df['timestamp'].max():.2f}")
    print(f"   Duration: {(df['timestamp'].max() - df['timestamp'].min()):.2f} seconds")
    
    # Run analyses
    analyze_zeros(df)
    analyze_distributions(df)
    analyze_temporal_patterns(df)
    analyze_correlations(df)
    print_feature_summary(df)
    
    # Final summary
    print_separator("KEY INSIGHTS & RECOMMENDATIONS")
    print("""
1. ZERO VALUES ARE NORMAL:
   • Blinks, gestures, and expression changes are RARE events
   • Zero means "no event detected in that frame"
   • This is expected behavior, not a bug
   
2. FEATURES FALL INTO CATEGORIES:
   • ACTIVATION features: Proportion/rate of muscle engagement (0-1)
   • MOTION features: Speed/energy of head/facial movement (0-1)
   • TEMPORAL features: Timing/delay of events (normalized, 0-1)
   • STABILITY features: Consistency/rigidity of posture (0-1)
   
3. BASELINE (0.5) vs ACTIVE:
   • After normalization, 0.5 = "baseline/neutral" for most features
   • 0.0-0.3 = suppressed/low activity
   • 0.3-0.7 = moderate/normal activity
   • 0.7-1.0 = elevated/strong activity
   
4. CORRELATION PATTERNS:
   • Motion features are highly correlated (moving head affects landmarks)
   • AU features are correlated (facial muscles work together)
   • Temporal features independent (different events)
   
5. TEMPORAL STABILITY:
   • Head motion/velocity features are volatile (change each frame)
   • Baseline/postural features are stable (smoothed over time)
   • This is expected - rapid motion changes, baseline doesn't
    """)

if __name__ == "__main__":
    main()
