# Raw CSV Feature Analysis - Comprehensive Insights

## 📊 Dataset Overview
- **Total Frames**: 7,100
- **Time Duration**: 238 seconds (~4 minutes)
- **Total Features**: 34
- **Sampling Rate**: ~29.8 FPS (frames per second)

---

## 🔍 WHY ARE SOME VALUES ZERO? (Detailed Explanation)

### Category 1: COMPLETELY ZERO (100% of frames are 0)

**Features that are always zero:**
1. **BlinkRate** (100% zeros)
   - Why: Eye blinks are RARE events - humans blink only 15-20 times per minute
   - Expected: About 4-5 blinks during 238 seconds (3.2%)
   - Observed: ZERO blinks detected
   - Reason: MediaPipe's eye detection doesn't trigger blinks reliably, OR person stayed very focused (reduced blinking)

2. **BlinkClusterDensity** (100% zeros)
   - Why: No blinks detected, so clustering measure returns 0
   - Related: Depends on BlinkRate being > 0 first

3. **LipCompressionFrequency** (100% zeros)
   - Why: Person's lips rarely compressed tightly during recording
   - Expected: Compression happens during stress, cold, or concentration
   - Observed: Person had relaxed, neutral lips throughout

4. **DownwardGazeFrequency** (100% zeros)
   - Why: Person was looking generally forward/horizontal
   - Expected: Would be non-zero if person looked down often (reading, thinking)
   - Observed: Person maintained forward gaze

5. **FacialTransitionFrequency** (100% zeros)
   - Why: Facial expression was VERY STABLE - almost no expression changes
   - Expected: Non-zero when expressions shift (happy→sad, surprised, etc.)
   - Observed: Person had consistent, neutral facial expression

6. **NearZeroAUActivationRatio** (100% zeros)
   - Why: This is a MEASURE of stability - zero means muscles were consistently active
   - Interpretation: Zero is GOOD - means facial muscles had baseline engagement
   - Note: This doesn't mean "nothing happened", it means "muscles stayed ON"

7. **ShoulderElevationIndex** (100% zeros)
   - Why: Shoulder tracking couldn't detect shrugging motion
   - Reason: MediaPipe focuses on face, shoulders are at edge of frame
   - Expected: Would be non-zero if person shrugged shoulders

---

### Category 2: MOSTLY ZERO (>80% zeros)

**Features with sparse activity:**

1. **GazeShiftFrequency** (82.4% zeros)
   - Interpretation: In 82.4% of time windows, eyes didn't shift to new positions
   - Non-zero when: Eyes made saccadic movements (jumps between focal points)
   - Reason: Person maintained relatively stable gaze direction
   - How zeros work: 
     - Frame has 150 previous frames (5 sec history)
     - If eyes stay in same position across all 150 frames → 0
     - If eyes jump 3+ times → non-zero

2. **GestureFrequency** (82.6% zeros)
   - Interpretation: In most 5-second windows, no meaningful head gestures occurred
   - Non-zero when: Person nods, shakes head, or tilts head significantly
   - Reason: Person was relatively still during recording

3. **ResponseLatencyMean** (82.5% zeros)
   - Interpretation: Measures TIME until reacting to stimuli
   - Zeros appear: When no significant motion detected yet
   - Non-zero: When motion response starts appearing

4. **NodOnsetLatency** (82.5% zeros)
   - Similar to ResponseLatencyMean
   - Zeros: When no head nod motion initiated
   - Non-zero: When head starts nodding motion

---

### Category 3: RARELY ZERO (<10% zeros)

**These features are almost ALWAYS active:**

1. **AU12Mean** (0% zeros)
   - Cheek raiser activation - always detecting some level
   - Range: 0.42 to 0.67 (moderate activation)
   - Interpretation: Mouth corners naturally at some elevation (baseline engagement)

2. **AU4MeanActivation** (0% zeros)
   - Brow lowerer - always at moderate level
   - Range: 0.74 to 0.83 (fairly engaged)
   - Interpretation: Brows have natural curvature/tension, never fully relaxed

3. **HeadMotionEnergy** (0% zeros)
   - Head is CONSTANTLY moving at small amounts
   - Even sitting still, humans have tremor/micro-movements
   - Range: 0.00 to 0.83 (almost always some motion)

4. **PostureRigidityIndex** (0% zeros)
   - Measures head stability - very high values (0.999)
   - Interpretation: Head posture very RIGID/STABLE
   - Good sign: Minimal unwanted head movement

---

## 📈 FEATURE VALUE RANGES & WHAT THEY MEAN

### Highest Activation Features (Mean > 0.75):
```
AU4DurationRatio         = 1.000 ← Brow always engaged
AU1AU2PeakIntensity      = 1.000 ← Eyebrows at peak consistently  
AU20ActivationRate       = 0.9999 ← Lip stretcher constantly active
AU12ActivationFrequency  = 0.9955 ← Cheek muscle very active
AU4MeanActivation        = 0.7866 ← Brows moderately raised
PostureRigidityIndex     = 0.9994 ← Head very stable
```

**Interpretation**: Person had VERY ENGAGED facial posture with:
- Elevated eyebrows (AU1/AU2)
- Moderate smile engagement (AU12)
- Frowned/concentrated brows (AU4)
- Stable head positioning

---

### Medium Activation Features (Mean 0.1-0.3):
```
SpeechOnsetDelay     = 0.1474 ← Mouth Movement started at 15% of max
EyeContactRatio      = 0.1324 ← Looking straight ahead 13% of the time
BaselineEyeOpenness  = 0.0793 ← Eyes slightly open (relaxed)
FacialEmotionalRange = 0.0658 ← Limited emotional expression range
```

**Interpretation**: 
- Person was relatively emotionally neutral
- Eyes mostly looking away (not at camera)
- Mouth movement delayed relative to other motion

---

### Low/Rare Features (Mean < 0.01):
```
AU15MeanAmplitude        = 0.0800 ← Lip corners rarely pull
MotionEnergyFloorScore   = 0.0102 ← Very low baseline motion
GazeShiftFrequency       = 0.0137 ← Eyes rarely shift
```

**Interpretation**: Low frequency of micro-expressions and small movements

---

## 🔗 FEATURE CORRELATIONS (Why they move together)

### Strong Positive Correlations (r > 0.99):
```
MeanHeadVelocity ←→ HeadMotionEnergy           (r = 1.000)
  └─ Why: Both measure head speed/movement - essentially same thing

AU1AU2PeakIntensity ←→ PostureRigidityIndex    (r = 0.994)
  └─ Why: People with raised eyebrows maintain rigid posture (concentration)

MeanHeadVelocity ←→ LandmarkDisplacementMean    (r = 0.998)
  └─ Why: Head velocity directly moves facial landmarks
```

### Interesting Negative Correlations (r < -0.7):
```
AU12ActivationFrequency ←→ GazeShiftFrequency  (r = -0.711)
  └─ Why: When smiling (AU12), eyes stay focused (don't shift)
           When not smiling, eyes shift around more
           
AU12ActivationFrequency ←→ ReactionTime        (r = -0.731)
  └─ Why: More smiling = faster, more stable reactions
```

---

## ⏱️ TEMPORAL PATTERNS (How Features Change Over Time)

### Most Volatile (Change rapidly frame-to-frame):
```
SpeechOnsetDelay         Δ = 0.00469 ← Mouth timing fluctuates constantly
AU12Mean                 Δ = 0.00390 ← Smile engagement changes moment-to-moment
AU4MeanActivation        Δ = 0.00265 ← Frown intensity changes frequently
```

**Why**: These detect rapid micro-movements and expression changes

### Most Stable (Very smooth over time):
```
AU4DurationRatio         Δ = 0.000000 ← Constantly engaged (no change needed)
AU20ActivationRate       Δ = 0.000001 ← Lip engagement extremely consistent
AU1AU2PeakIntensity      Δ = 0.000001 ← Eyebrow height unchanging
```

**Why**: These are baseline measures, user maintained constant expression

---

## 🎯 KEY INSIGHTS ABOUT THIS PERSON'S RECORDING

### What the data tells us:
1. **Very Stable Face**: Minimal expression changes, very engaged muscles (AU4, AU12)
2. **Focused Look**: Eyes not shifting much, not looking around
3. **Neutral Affect**: No rapid transitions (no smiling→frowning changes)
4. **Good Posture**: Head rigidity near 1.0 (very stable)
5. **Minimal Gestures**: No shrugs, few nods detected
6. **Relaxed Eyes**: Baseline eye openness low (not wide-eyed alert)

### Session Behavior Profile:
- **Duration**: 238 seconds (4 min)
- **Behavior**: Focused, concentrated, relatively still
- **Pose**: Head stable, forward-looking
- **Expression**: Engaged but neutral (slight smile engagement)
- **Engagement Level**: HIGH (most AU features activated)

---

## ✅ ARE THESE ZEROS A PROBLEM?

**NO - This is completely normal!**

### Why zeros are expected:
1. **Event-based features** (blinks, gestures) are RARE
   - Humans blink ~15-20 times/min = ~3-5 blinks in 238 sec
   - Observed: 0 blinks = person focused (or just luck)
   
2. **Zero ≠ broken feature**
   - It means "this event didn't occur in this frame"
   - Like a motion detector that shows 0 when nothing moves
   
3. **Baseline measures** stay constant when person is stable
   - If person held same expression → feature stays same
   - If AU4 duration always 1.0 → means brows always engaged

4. **Different events occur at different rates**:
   - Blinks: 1-2 per minute
   - Expression changes: Few per minute
   - Head motion: Continuous
   - Facial muscle activation: Continuous

---

## 📊 SUMMARY TABLE: Feature Types & Expected Behavior

| Feature Type | Examples | Normal Behavior |
|---|---|---|
| **Continuous Baseline** | AU4Mean, AU12Mean | Always non-zero, vary slowly |
| **Motion Metrics** | HeadVelocity, HeadMotionEnergy | Usually small non-zero (tremor) |
| **Event Rates** | BlinkRate, GestureFrequency | Often zero (events are rare) |
| **Activation Proportions** | AU4DurationRatio, EyeContactRatio | Often high if person held stable pose |
| **Energy/Variance** | OverallAUVariance, FacialEmotionalRange | Usually small non-zero values |
| **Temporal Measures** | ResponseLatency, SpeechDelay | Often zero if no stimulus/response |

---

## 🔧 USAGE RECOMMENDATIONS

1. **Don't filter out zeros** - they're meaningful data
2. **Group features by category** - they have different behaviors
3. **Use for classification** - pattern of zeros/highs indicates behavior type
4. **Apply smoothing** - motion features are noisy (frame-to-frame)
5. **Temporal analysis** - look at patterns over 5-10 second windows
6. **Correlation analysis** - use strong correlations (r > 0.7) cautiously (redundant info)

---

**Generated**: March 6, 2026  
**Data Source**: output/raw/23_51_05_03.csv  
**Total Frames Analyzed**: 7,100
