"""
Disorder Behavioral Profiles

Defines how each mental health condition modifies baseline behavior.
Based on clinical literature and behavioral research.

Modifiers are applied as statistical transformations:
- Mean multipliers (e.g., 0.7 = 30% reduction)
- Std multipliers (e.g., 1.2 = 20% increase in variability)
"""

# Label mappings
DISORDER_LABELS = {
    'healthy': 0,
    'depression': 1,
    'anxiety': 2,
    'mania': 3,
    'burnout': 4
}


# Behavioral modifiers for each disorder
DISORDER_PROFILES = {
    
    'depression': {
        'description': 'Reduced facial expressivity, decreased movement, increased latency',
        
        'modifiers': {
            # Facial AU Features - REDUCED
            'S_AU12Mean': {'mean_mult': 0.60, 'std_mult': 0.80},  # Less smiling
            'S_AU12Variance': {'mean_mult': 0.65, 'std_mult': 0.85},
            'S_AU12ActivationFrequency': {'mean_mult': 0.70, 'std_mult': 0.90},
            'S_AU15MeanAmplitude': {'mean_mult': 0.75, 'std_mult': 0.85},  # Flatter lips
            'S_AU4MeanActivation': {'mean_mult': 1.15, 'std_mult': 1.05},  # Slight brow tension
            'S_AU20ActivationRate': {'mean_mult': 0.80, 'std_mult': 0.90},
            
            # Eye Features - REDUCED
            'S_BlinkRate': {'mean_mult': 0.85, 'std_mult': 0.90},
            'S_GazeShiftFrequency': {'mean_mult': 0.70, 'std_mult': 0.85},  # Less gaze movement
            'S_EyeContactRatio': {'mean_mult': 0.65, 'std_mult': 0.80},  # Reduced eye contact
            'S_DownwardGazeFrequency': {'mean_mult': 1.30, 'std_mult': 1.10},  # More downward gaze
            
            # Head Motion - REDUCED
            'S_MeanHeadVelocity': {'mean_mult': 0.60, 'std_mult': 0.75},  # Less movement
            'S_HeadVelocityPeak': {'mean_mult': 0.65, 'std_mult': 0.80},
            'S_HeadMotionEnergy': {'mean_mult': 0.55, 'std_mult': 0.70},
            'S_LandmarkDisplacementMean': {'mean_mult': 0.65, 'std_mult': 0.80},
            'S_PostureRigidityIndex': {'mean_mult': 1.25, 'std_mult': 1.10},  # More rigid
            
            # Derived Features
            'S_FacialEmotionalRange': {'mean_mult': 0.50, 'std_mult': 0.70},  # Flat affect
            'S_FacialTransitionFrequency': {'mean_mult': 0.65, 'std_mult': 0.80},
            'S_GestureFrequency': {'mean_mult': 0.60, 'std_mult': 0.75},
            'S_MicroMotionEnergy': {'mean_mult': 0.70, 'std_mult': 0.85},
            
            # Temporal Features - INCREASED LATENCY
            'S_ResponseLatencyMean': {'mean_mult': 1.40, 'std_mult': 1.20},  # Slower responses
            'S_SpeechOnsetDelay': {'mean_mult': 1.35, 'std_mult': 1.15},
            'S_NodOnsetLatency': {'mean_mult': 1.30, 'std_mult': 1.15},
            'S_PauseDurationMean': {'mean_mult': 1.25, 'std_mult': 1.10},
            'S_ReactionTimeInstabilityIndex': {'mean_mult': 1.15, 'std_mult': 1.05}
        }
    },
    
    'anxiety': {
        'description': 'Increased tension, rapid movements, heightened alertness',
        
        'modifiers': {
            # Facial AU Features - INCREASED TENSION
            'S_AU4MeanActivation': {'mean_mult': 1.45, 'std_mult': 1.25},  # High brow tension
            'S_AU4DurationRatio': {'mean_mult': 1.35, 'std_mult': 1.20},
            'S_AU1AU2PeakIntensity': {'mean_mult': 1.30, 'std_mult': 1.15},  # Inner brow raise
            'S_AU12Mean': {'mean_mult': 0.85, 'std_mult': 1.10},  # Tense smiling
            'S_LipCompressionFrequency': {'mean_mult': 1.40, 'std_mult': 1.20},
            
            # Eye Features - HEIGHTENED
            'S_BlinkRate': {'mean_mult': 1.50, 'std_mult': 1.30},  # Rapid blinking
            'S_BlinkClusterDensity': {'mean_mult': 1.45, 'std_mult': 1.25},
            'S_GazeShiftFrequency': {'mean_mult': 1.55, 'std_mult': 1.30},  # Darting eyes
            'S_EyeContactRatio': {'mean_mult': 0.80, 'std_mult': 1.20},  # Avoidant
            
            # Head Motion - INCREASED
            'S_MeanHeadVelocity': {'mean_mult': 1.25, 'std_mult': 1.40},  # Fidgety
            'S_HeadVelocityPeak': {'mean_mult': 1.30, 'std_mult': 1.45},
            'S_HeadMotionEnergy': {'mean_mult': 1.35, 'std_mult': 1.50},
            'S_PostureRigidityIndex': {'mean_mult': 0.85, 'std_mult': 1.25},  # Less rigid, more movement
            
            # Derived Features
            'S_OverallAUVariance': {'mean_mult': 1.35, 'std_mult': 1.30},  # High variability
            'S_FacialTransitionFrequency': {'mean_mult': 1.40, 'std_mult': 1.35},
            'S_MicroMotionEnergy': {'mean_mult': 1.45, 'std_mult': 1.40},
            'S_GestureFrequency': {'mean_mult': 1.30, 'std_mult': 1.35},
            
            # Temporal Features - MIXED
            'S_ResponseLatencyMean': {'mean_mult': 0.85, 'std_mult': 1.30},  # Fast but inconsistent
            'S_ReactionTimeInstabilityIndex': {'mean_mult': 1.40, 'std_mult': 1.35}
        }
    },
    
    'mania': {
        'description': 'High energy, excessive expressivity, rapid responses',
        
        'modifiers': {
            # Facial AU Features - EXAGGERATED
            'S_AU12Mean': {'mean_mult': 1.40, 'std_mult': 1.30},  # Excessive smiling
            'S_AU12Variance': {'mean_mult': 1.50, 'std_mult': 1.40},
            'S_AU12ActivationFrequency': {'mean_mult': 1.45, 'std_mult': 1.35},
            'S_AU20ActivationRate': {'mean_mult': 1.35, 'std_mult': 1.25},
            
            # Eye Features - INTENSE
            'S_GazeShiftFrequency': {'mean_mult': 1.60, 'std_mult': 1.50},  # Rapid gaze
            'S_EyeContactRatio': {'mean_mult': 1.25, 'std_mult': 1.15},  # Intense staring
            'S_BlinkRate': {'mean_mult': 0.70, 'std_mult': 0.85},  # Less blinking
            
            # Head Motion - VERY HIGH
            'S_MeanHeadVelocity': {'mean_mult': 1.60, 'std_mult': 1.50},  # Hyperkinetic
            'S_HeadVelocityPeak': {'mean_mult': 1.70, 'std_mult': 1.60},
            'S_HeadMotionEnergy': {'mean_mult': 1.75, 'std_mult': 1.65},
            'S_LandmarkDisplacementMean': {'mean_mult': 1.50, 'std_mult': 1.40},
            'S_PostureRigidityIndex': {'mean_mult': 0.50, 'std_mult': 0.70},  # Very loose
            
            # Derived Features - HIGH ENERGY
            'S_FacialEmotionalRange': {'mean_mult': 1.60, 'std_mult': 1.50},  # Dramatic
            'S_FacialTransitionFrequency': {'mean_mult': 1.70, 'std_mult': 1.60},
            'S_OverallAUVariance': {'mean_mult': 1.55, 'std_mult': 1.45},
            'S_GestureFrequency': {'mean_mult': 1.65, 'std_mult': 1.55},
            'S_MicroMotionEnergy': {'mean_mult': 1.60, 'std_mult': 1.50},
            
            # Temporal Features - RAPID
            'S_ResponseLatencyMean': {'mean_mult': 0.55, 'std_mult': 0.70},  # Very fast
            'S_SpeechOnsetDelay': {'mean_mult': 0.50, 'std_mult': 0.65},
            'S_NodOnsetLatency': {'mean_mult': 0.60, 'std_mult': 0.75},
            'S_PauseDurationMean': {'mean_mult': 0.45, 'std_mult': 0.60},
            'S_ExtendedSilenceRatio': {'mean_mult': 0.40, 'std_mult': 0.55}
        }
    },
    
    'burnout': {
        'description': 'Exhaustion, reduced engagement, flat affect',
        
        'modifiers': {
            # Facial AU Features - DEPLETED
            'S_AU12Mean': {'mean_mult': 0.50, 'std_mult': 0.70},  # Minimal smiling
            'S_AU15MeanAmplitude': {'mean_mult': 0.65, 'std_mult': 0.75},
            'S_AU20ActivationRate': {'mean_mult': 0.70, 'std_mult': 0.80},
            
            # Eye Features - TIRED
            'S_BaselineEyeOpenness': {'mean_mult': 0.75, 'std_mult': 0.85},  # Droopy eyes
            'S_BlinkRate': {'mean_mult': 1.20, 'std_mult': 1.10},  # More blinking (tired)
            'S_GazeShiftFrequency': {'mean_mult': 0.60, 'std_mult': 0.75},  # Disengaged
            'S_EyeContactRatio': {'mean_mult': 0.55, 'std_mult': 0.70},
            'S_DownwardGazeFrequency': {'mean_mult': 1.40, 'std_mult': 1.20},
            
            # Head Motion - LOW ENERGY
            'S_MeanHeadVelocity': {'mean_mult': 0.50, 'std_mult': 0.65},  # Minimal movement
            'S_HeadMotionEnergy': {'mean_mult': 0.45, 'std_mult': 0.60},
            'S_PostureRigidityIndex': {'mean_mult': 1.35, 'std_mult': 1.15},  # Stiff posture
            
            # Derived Features - FLAT
            'S_FacialEmotionalRange': {'mean_mult': 0.40, 'std_mult': 0.60},  # Very flat
            'S_FacialTransitionFrequency': {'mean_mult': 0.55, 'std_mult': 0.70},
            'S_OverallAUVariance': {'mean_mult': 0.50, 'std_mult': 0.65},
            'S_GestureFrequency': {'mean_mult': 0.45, 'std_mult': 0.60},
            'S_MotionEnergyFloorScore': {'mean_mult': 1.30, 'std_mult': 1.15},
            
            # Temporal Features - SLOW
            'S_ResponseLatencyMean': {'mean_mult': 1.30, 'std_mult': 1.15},
            'S_PauseDurationMean': {'mean_mult': 1.35, 'std_mult': 1.20},
            'S_ExtendedSilenceRatio': {'mean_mult': 1.40, 'std_mult': 1.25}
        }
    }
}


def apply_disorder_modifiers(baseline_stats, disorder_name):
    """
    Apply disorder-specific modifiers to baseline statistics.

    Parameters
    ----------
    baseline_stats : dict
        Baseline feature statistics
    disorder_name : str
        Disorder name ('depression', 'anxiety', 'mania', 'burnout')

    Returns
    -------
    dict
        Modified feature statistics

    Example
    -------
    >>> depression_stats = apply_disorder_modifiers(baseline, 'depression')
    """
    if disorder_name not in DISORDER_PROFILES:
        raise ValueError(f"Unknown disorder: {disorder_name}")

    profile = DISORDER_PROFILES[disorder_name]
    modifiers = profile['modifiers']
    
    modified_stats = {}

    for feature, stats in baseline_stats.items():
        # Get modifiers for this feature (default to 1.0 if not specified)
        feature_mods = modifiers.get(feature, {'mean_mult': 1.0, 'std_mult': 1.0})
        
        mean_mult = feature_mods['mean_mult']
        std_mult = feature_mods['std_mult']

        # Apply multipliers
        modified_stats[feature] = {
            'mean': stats['mean'] * mean_mult,
            'std': stats['std'] * std_mult,
            'min': stats['min'],  # Keep bounds
            'max': stats['max'],
            'median': stats['median'] * mean_mult,
            'q25': stats['q25'] * mean_mult,
            'q75': stats['q75'] * mean_mult
        }

    return modified_stats


def print_disorder_summary():
    """
    Print summary of all disorder profiles.
    """
    print(f"\n{'='*60}")
    print("DISORDER BEHAVIORAL PROFILES")
    print(f"{'='*60}")

    for disorder, profile in DISORDER_PROFILES.items():
        label = DISORDER_LABELS[disorder]
        print(f"\n[{label}] {disorder.upper()}")
        print(f"    {profile['description']}")
        print(f"    Modified features: {len(profile['modifiers'])}")


if __name__ == "__main__":
    print_disorder_summary()
