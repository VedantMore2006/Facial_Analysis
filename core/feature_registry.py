"""
Feature Registry

Central registry of all 34 behavioral features for pipeline execution.
"""

# Module A — Facial AU Features
from features.facial_au_features import (
    AU12Mean,
    AU12Variance,
    AU12ActivationFrequency,
    AU15MeanAmplitude,
    AU4MeanActivation,
    AU4DurationRatio,
    AU1AU2PeakIntensity,
    AU20ActivationRate,
    LipCompressionFrequency
)

# Module B — Eye Features
from features.eye_features import (
    BlinkRate,
    BlinkClusterDensity,
    BaselineEyeOpenness,
    GazeShiftFrequency,
    EyeContactRatio,
    DownwardGazeFrequency
)

# Module C — Head Motion Features
from features.head_motion_features import (
    MeanHeadVelocity,
    HeadVelocityPeak,
    HeadMotionEnergy,
    LandmarkDisplacementMean,
    PostureRigidityIndex
)

# Module D — Derived Features
from features.derived_features import (
    OverallAUVariance,
    FacialEmotionalRange,
    FacialTransitionFrequency,
    NearZeroAUActivationRatio,
    MotionEnergyFloorScore,
    GestureFrequency,
    MicroMotionEnergy,
    ShoulderElevationIndex
)

# Module E — Temporal Features
from features.temporal_features import (
    ResponseLatencyMean,
    SpeechOnsetDelay,
    NodOnsetLatency,
    PauseDurationMean,
    ExtendedSilenceRatio,
    ReactionTimeInstabilityIndex
)

FEATURE_REGISTRY = [

    # Module A — Facial AU Features (9)
    AU12Mean(),
    AU12Variance(),
    AU12ActivationFrequency(),
    AU15MeanAmplitude(),
    AU4MeanActivation(),
    AU4DurationRatio(),
    AU1AU2PeakIntensity(),
    AU20ActivationRate(),
    LipCompressionFrequency(),

    # Module B — Eye Features (6)
    BlinkRate(),
    BlinkClusterDensity(),
    BaselineEyeOpenness(),
    GazeShiftFrequency(),
    EyeContactRatio(),
    DownwardGazeFrequency(),

    # Module C — Head Motion Features (5)
    MeanHeadVelocity(),
    HeadVelocityPeak(),
    HeadMotionEnergy(),
    LandmarkDisplacementMean(),
    PostureRigidityIndex(),

    # Module D — Derived Features (8)
    OverallAUVariance(),
    FacialEmotionalRange(),
    FacialTransitionFrequency(),
    NearZeroAUActivationRatio(),
    MotionEnergyFloorScore(),
    GestureFrequency(),
    MicroMotionEnergy(),
    ShoulderElevationIndex(),

    # Module E — Temporal Features (6)
    ResponseLatencyMean(),
    SpeechOnsetDelay(),
    NodOnsetLatency(),
    PauseDurationMean(),
    ExtendedSilenceRatio(),
    ReactionTimeInstabilityIndex()
]

# Feature count verification
assert len(FEATURE_REGISTRY) == 34, "Feature count mismatch!"


def get_feature_names():
    """Return list of all feature names."""
    return [feature.name for feature in FEATURE_REGISTRY]


def compute_all_features(landmarks, frame_buffer, timestamp):
    """
    Compute all features for current frame.

    Returns dict of feature name -> value.
    """
    features = {}

    for feature in FEATURE_REGISTRY:
        features[feature.name] = feature.compute(landmarks, frame_buffer, timestamp)

    return features
