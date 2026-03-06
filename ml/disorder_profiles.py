"""
Disorder Behavioral Profiles

Defines how each mental health condition modifies baseline behavior.
Modifiers are statistical priors for synthetic generation (not clinical diagnosis).
"""

DISORDER_LABELS = {
    'depression': 0,
    'anxiety': 1,
    'stress': 2,
    'bipolar_mania': 3,
    'phobia_common': 4,
    'suicidal_tendency': 5
}


DISORDER_PROFILES = {
    'depression': {
        'description': 'Reduced expressivity, reduced movement, increased latency',
        'modifiers': {
            'S_AU12Mean': {'mean_mult': 0.60, 'std_mult': 0.80},
            'S_AU12Variance': {'mean_mult': 0.65, 'std_mult': 0.85},
            'S_FacialEmotionalRange': {'mean_mult': 0.50, 'std_mult': 0.70},
            'S_GestureFrequency': {'mean_mult': 0.60, 'std_mult': 0.75},
            'S_MeanHeadVelocity': {'mean_mult': 0.60, 'std_mult': 0.75},
            'S_HeadMotionEnergy': {'mean_mult': 0.55, 'std_mult': 0.70},
            'S_EyeContactRatio': {'mean_mult': 0.65, 'std_mult': 0.80},
            'S_DownwardGazeFrequency': {'mean_mult': 1.30, 'std_mult': 1.10},
            'S_ResponseLatencyMean': {'mean_mult': 1.40, 'std_mult': 1.20},
            'S_SpeechOnsetDelay': {'mean_mult': 1.35, 'std_mult': 1.15},
            'S_PauseDurationMean': {'mean_mult': 1.25, 'std_mult': 1.10}
        }
    },

    'anxiety': {
        'description': 'High tension, vigilance, fast but unstable reactions',
        'modifiers': {
            'S_AU4MeanActivation': {'mean_mult': 1.45, 'std_mult': 1.25},
            'S_LipCompressionFrequency': {'mean_mult': 1.40, 'std_mult': 1.20},
            'S_BlinkRate': {'mean_mult': 1.50, 'std_mult': 1.30},
            'S_BlinkClusterDensity': {'mean_mult': 1.45, 'std_mult': 1.25},
            'S_GazeShiftFrequency': {'mean_mult': 1.55, 'std_mult': 1.30},
            'S_EyeContactRatio': {'mean_mult': 0.80, 'std_mult': 1.20},
            'S_MeanHeadVelocity': {'mean_mult': 1.25, 'std_mult': 1.40},
            'S_HeadMotionEnergy': {'mean_mult': 1.35, 'std_mult': 1.50},
            'S_MicroMotionEnergy': {'mean_mult': 1.45, 'std_mult': 1.40},
            'S_ReactionTimeInstabilityIndex': {'mean_mult': 1.40, 'std_mult': 1.35}
        }
    },

    'stress': {
        'description': 'Sustained strain with elevated tension and variability',
        'modifiers': {
            'S_AU4MeanActivation': {'mean_mult': 1.25, 'std_mult': 1.15},
            'S_LipCompressionFrequency': {'mean_mult': 1.30, 'std_mult': 1.20},
            'S_BlinkRate': {'mean_mult': 1.25, 'std_mult': 1.20},
            'S_GazeShiftFrequency': {'mean_mult': 1.20, 'std_mult': 1.15},
            'S_MeanHeadVelocity': {'mean_mult': 1.15, 'std_mult': 1.20},
            'S_HeadMotionEnergy': {'mean_mult': 1.20, 'std_mult': 1.25},
            'S_MicroMotionEnergy': {'mean_mult': 1.25, 'std_mult': 1.25},
            'S_OverallAUVariance': {'mean_mult': 1.20, 'std_mult': 1.20},
            'S_ReactionTimeInstabilityIndex': {'mean_mult': 1.30, 'std_mult': 1.25},
            'S_ResponseLatencyMean': {'mean_mult': 1.10, 'std_mult': 1.15}
        }
    },

    'bipolar_mania': {
        'description': 'Very high energy, expressivity, and fast responding',
        'modifiers': {
            'S_AU12Mean': {'mean_mult': 1.40, 'std_mult': 1.30},
            'S_AU12Variance': {'mean_mult': 1.50, 'std_mult': 1.40},
            'S_FacialEmotionalRange': {'mean_mult': 1.60, 'std_mult': 1.50},
            'S_FacialTransitionFrequency': {'mean_mult': 1.70, 'std_mult': 1.60},
            'S_GestureFrequency': {'mean_mult': 1.65, 'std_mult': 1.55},
            'S_MeanHeadVelocity': {'mean_mult': 1.60, 'std_mult': 1.50},
            'S_HeadMotionEnergy': {'mean_mult': 1.75, 'std_mult': 1.65},
            'S_GazeShiftFrequency': {'mean_mult': 1.60, 'std_mult': 1.50},
            'S_ResponseLatencyMean': {'mean_mult': 0.55, 'std_mult': 0.70},
            'S_SpeechOnsetDelay': {'mean_mult': 0.50, 'std_mult': 0.65},
            'S_PauseDurationMean': {'mean_mult': 0.45, 'std_mult': 0.60}
        }
    },

    'phobia_common': {
        'description': 'Threat-related vigilance, avoidant eye behavior, elevated reactivity',
        'modifiers': {
            'S_AU4MeanActivation': {'mean_mult': 1.35, 'std_mult': 1.20},
            'S_LipCompressionFrequency': {'mean_mult': 1.30, 'std_mult': 1.20},
            'S_BlinkRate': {'mean_mult': 1.35, 'std_mult': 1.25},
            'S_GazeShiftFrequency': {'mean_mult': 1.40, 'std_mult': 1.30},
            'S_EyeContactRatio': {'mean_mult': 0.70, 'std_mult': 1.15},
            'S_DownwardGazeFrequency': {'mean_mult': 1.20, 'std_mult': 1.10},
            'S_HeadMotionEnergy': {'mean_mult': 1.20, 'std_mult': 1.25},
            'S_OverallAUVariance': {'mean_mult': 1.25, 'std_mult': 1.20},
            'S_FacialTransitionFrequency': {'mean_mult': 1.30, 'std_mult': 1.20},
            'S_ReactionTimeInstabilityIndex': {'mean_mult': 1.35, 'std_mult': 1.30}
        }
    },

    'suicidal_tendency': {
        'description': 'Strong withdrawal profile with low expressivity and delayed response patterns',
        'modifiers': {
            'S_AU12Mean': {'mean_mult': 0.45, 'std_mult': 0.65},
            'S_AU12Variance': {'mean_mult': 0.50, 'std_mult': 0.70},
            'S_FacialEmotionalRange': {'mean_mult': 0.35, 'std_mult': 0.55},
            'S_FacialTransitionFrequency': {'mean_mult': 0.50, 'std_mult': 0.70},
            'S_GestureFrequency': {'mean_mult': 0.40, 'std_mult': 0.60},
            'S_EyeContactRatio': {'mean_mult': 0.45, 'std_mult': 0.65},
            'S_DownwardGazeFrequency': {'mean_mult': 1.50, 'std_mult': 1.25},
            'S_MeanHeadVelocity': {'mean_mult': 0.45, 'std_mult': 0.60},
            'S_HeadMotionEnergy': {'mean_mult': 0.40, 'std_mult': 0.55},
            'S_ResponseLatencyMean': {'mean_mult': 1.45, 'std_mult': 1.25},
            'S_PauseDurationMean': {'mean_mult': 1.50, 'std_mult': 1.30},
            'S_ExtendedSilenceRatio': {'mean_mult': 1.55, 'std_mult': 1.30}
        }
    }
}


CONDITION_IMPORTANCE_PRIORS = {
    'depression': {
        'S_FacialEmotionalRange': 0.22,
        'S_AU12Mean': 0.18,
        'S_ResponseLatencyMean': 0.16,
        'S_EyeContactRatio': 0.14,
        'S_HeadMotionEnergy': 0.12,
        'S_PauseDurationMean': 0.10,
        'S_DownwardGazeFrequency': 0.08
    },
    'anxiety': {
        'S_BlinkRate': 0.20,
        'S_GazeShiftFrequency': 0.18,
        'S_AU4MeanActivation': 0.16,
        'S_ReactionTimeInstabilityIndex': 0.14,
        'S_MicroMotionEnergy': 0.12,
        'S_LipCompressionFrequency': 0.10,
        'S_HeadMotionEnergy': 0.10
    },
    'stress': {
        'S_AU4MeanActivation': 0.20,
        'S_MicroMotionEnergy': 0.17,
        'S_ReactionTimeInstabilityIndex': 0.16,
        'S_LipCompressionFrequency': 0.14,
        'S_BlinkRate': 0.12,
        'S_HeadMotionEnergy': 0.11,
        'S_ResponseLatencyMean': 0.10
    },
    'bipolar_mania': {
        'S_HeadMotionEnergy': 0.20,
        'S_FacialTransitionFrequency': 0.18,
        'S_FacialEmotionalRange': 0.17,
        'S_AU12Variance': 0.15,
        'S_GazeShiftFrequency': 0.12,
        'S_ResponseLatencyMean': 0.10,
        'S_GestureFrequency': 0.08
    },
    'phobia_common': {
        'S_GazeShiftFrequency': 0.20,
        'S_AU4MeanActivation': 0.17,
        'S_BlinkRate': 0.15,
        'S_EyeContactRatio': 0.14,
        'S_ReactionTimeInstabilityIndex': 0.13,
        'S_LipCompressionFrequency': 0.11,
        'S_HeadMotionEnergy': 0.10
    },
    'suicidal_tendency': {
        'S_FacialEmotionalRange': 0.22,
        'S_EyeContactRatio': 0.18,
        'S_PauseDurationMean': 0.15,
        'S_ExtendedSilenceRatio': 0.14,
        'S_ResponseLatencyMean': 0.12,
        'S_AU12Mean': 0.11,
        'S_HeadMotionEnergy': 0.08
    }
}


def apply_disorder_modifiers(baseline_stats, disorder_name):
    """Apply disorder-specific modifiers to baseline statistics."""
    if disorder_name not in DISORDER_PROFILES:
        raise ValueError(f"Unknown disorder: {disorder_name}")

    profile = DISORDER_PROFILES[disorder_name]
    modifiers = profile['modifiers']
    modified_stats = {}

    for feature, stats in baseline_stats.items():
        feature_mods = modifiers.get(feature, {'mean_mult': 1.0, 'std_mult': 1.0})
        mean_mult = feature_mods['mean_mult']
        std_mult = feature_mods['std_mult']

        modified_stats[feature] = {
            'mean': stats['mean'] * mean_mult,
            'std': stats['std'] * std_mult,
            'min': stats['min'],
            'max': stats['max'],
            'median': stats['median'] * mean_mult,
            'q25': stats['q25'] * mean_mult,
            'q75': stats['q75'] * mean_mult
        }

    return modified_stats


def print_disorder_summary():
    """Print summary of all disorder profiles."""
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
