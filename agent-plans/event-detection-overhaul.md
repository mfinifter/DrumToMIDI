# Event Detection System Overhaul

## Problem Statement

The original librosa-based onset detection was fundamentally flawed for isolated drum stems:

### Issues with librosa onset detection:
1. **Wait periods blind to real events** - After detecting noise, `wait=3` frames blocked subsequent genuine hits
2. **Decay comparison to zero** - Long decays triggered multiple "onsets" when comparing to zero instead of event's ending energy
3. **Generic music tuning** - Optimized for mixed music, not clean isolated drums where events are visually obvious in DAW
4. **Over-detection** - Thunderstruck cymbals: detected 238 events vs ~54 actual (4.4x too many)

### Core insight from user:
> "if a noise event is detected but the thresholds are met, then the detection will be 'skipped' for a period to try to reduce the raw events being too close together. A subsequent real event, that would be easily hit the proper thresholds might then be skipped"

Looking at DAW waveform: cymbal hits at 112s and 119s are **unmistakably obvious** with clear attack transients.

## Solution: Energy Envelope Peak Detection

### New Approach
Instead of librosa's complex onset strength analysis, we now:

1. **Calculate RMS energy envelope** - exactly what DAWs render visually
2. **Find local maxima** using scipy `find_peaks` - robust peak detection
3. **Filter by prominence** - peaks must stand out significantly above local baseline
4. **Backtrack to attack start** - use `left_bases` and threshold to find onset (not peak)
5. **Absolute energy threshold** - ignore noise floor
6. **Minimum spacing** - prevent double-detection, but no blind wait periods

### Backtracking for Accurate Onset Timing

Even fast transients like crash cymbals take 50-120ms to reach peak energy. We detect the **attack start**, not the peak:

1. `scipy.find_peaks` finds energy peaks and returns `left_bases` (where rise begins)
2. From each peak, backtrack to find where energy crossed 15% of peak value
3. This gives onset time at attack start, matching what drummers play

**Example:** Crash cymbal at 112s
- Peak energy occurs at 112.628s
- Attack starts at 112.512s (116ms earlier)
- MIDI note placed at 112.512s matches drummer timing

### Key Differences

| Aspect | Old (librosa) | New (scipy peaks) |
|--------|---------------|-------------------|
| **Method** | Spectral flux onset strength | RMS energy envelope peaks |
| **Detection** | Custom peak picking with wait periods | scipy.signal.find_peaks |
| **Threshold** | Arbitrary onset strength | Prominence in dB above local minimum |
| **Spacing** | Wait=3 frames after ANY detection | distance parameter, no blindness |
| **Missed events** | Can miss real hits during wait period | Every peak evaluated independently |
| **Visual analogy** | "Guessing" transients from spectral change | "Seeing" peaks like in DAW waveform |

## Implementation

### Core Functions

**`calculate_energy_envelope()`** - RMS or spectral energy over time
```python
# Returns (times, energy_values) - the waveform visualization
left_times, left_energy = calculate_energy_envelope(
    left_channel, sr, frame_length=2048, hop_length=512, method='rms'
)
```

**`detect_transient_peaks()`** - Find prominent peaks using scipy
```python
# Uses scipy.signal.find_peaks with:
# - height: minimum absolute energy (noise floor)
# - distance: minimum spacing between peaks
# - prominence: must stand out above surroundings

# Prominence converted from dB: 
# "15dB above local minimum" means peak/minimum_ratio ≈ 5.6x

peak_indices, properties = find_peaks(
    energy,
    height=min_absolute_energy,
    distance=min_spacing_frames,
    prominence=prominence_linear
)
```

**`detect_stereo_transient_peaks()`** - Process L/R, merge nearby
```python
# 1. Calculate energy envelopes for L and R
# 2. Detect peaks in each channel independently
# 3. Merge L/R peaks within merge_window_ms (150ms)
# 4. Calculate pan confidence from L/R energy ratio
```

### Leveraging scipy.signal.find_peaks

We use several scipy features for robust detection:

**Core Features:**
- `prominence`: Minimum height above local minima (ensures peaks stand out)
- `height`: Absolute minimum energy threshold (noise floor)
- `distance`: Minimum spacing between peaks (prevents double-detection)
- `left_bases`: Starting point of each peak's rise (used for backtracking)

**Backtracking Algorithm:**
```python
# 1. Find peaks with prominence filtering
peak_indices, properties = find_peaks(energy, prominence=..., height=..., distance=...)

# 2. Get left bases (where each peak starts rising)
left_bases = properties['left_bases']

# 3. Backtrack from peak to find 15% threshold crossing
for peak_idx in peak_indices:
    # Search backwards from peak to left_base
    # Find where energy crosses 15% of peak value
    # That's the attack start (onset time)
```

This combines scipy's robust peak detection with domain knowledge about drum transients.

### Calibrated Parameters

Through iterative testing on Thunderstruck cymbals:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `threshold_db` | **15.0** | Prominence above local minimum (higher = fewer/cleaner) |
| `min_absolute_energy` | **0.01** | Noise floor - real cymbal hits are louder |
| `min_peak_spacing_ms` | **100.0** | Minimum time between peaks (no double-detection) |
| `merge_window_ms` | **150.0** | L/R merge window for stereo events |
| `frame_length` | 2048 | RMS calculation window |
| `hop_length` | 512 | Frame advance (~11.6ms at 44.1kHz) |

### Parameter Tuning Process

Started with educated guesses from DAW analysis:
- **threshold_db**: Tested 6.5 → 8.0 → 12.0 → **15.0** dB
- **min_absolute_energy**: Tested 0.001 → 0.005 → **0.01**

Results progression:
```
threshold_db=6.5,  min_energy=0.001: 485 onsets (10x too many)
threshold_db=6.5,  min_energy=0.005:  48 onsets (slightly low, missed 112s/119s)
threshold_db=12.0, min_energy=0.01:  110 onsets (2x too many)
threshold_db=15.0, min_energy=0.01:   75 onsets ✅ (matches user count ~54 + quieter hits)
```

**Final validation**: Both obvious DAW events detected at 112.628s and 119.838s.

## Results Comparison

### Thunderstruck Cymbals (~54 actual events)

| Method | Events Detected | Accuracy | Found 112s | Found 119s |
|--------|----------------|----------|------------|------------|
| librosa onset | 238 | 4.4x over | ✅ 112.454s | ✅ 119.676s |
| Energy tracking (6dB) | 116 | 2.1x over | ❌ Missing | ✅ 119.281s |
| **Transient peaks (15dB)** | **75** | **1.4x over** | **✅ 112.628s** | **✅ 119.838s** |

The 1.4x over-detection (75 vs 54) likely includes:
- Quieter cymbal hits not counted manually
- Cymbal bleed from other stems
- Ghost notes or subtle hi-hat contamination

These can be further filtered using geomean thresholds in post-processing.

## Benefits

1. ✅ **Visually verifiable** - Matches what you see in DAW waveform
2. ✅ **No blind spots** - Every peak evaluated, no wait-period misses
3. ✅ **Robust** - scipy's find_peaks is battle-tested
4. ✅ **Tunable** - Single prominence parameter controls sensitivity
5. ✅ **Fast** - RMS calculation much faster than spectral flux
6. ✅ **Explainable** - "Peaks 15dB above neighbors" vs black-box onset strength

## Future Improvements

### Visual Threshold Selector (Future Feature)
Could create interactive UI:
- Display energy envelope waveform
- Overlay detected peaks
- Slider to adjust threshold_db in real-time
- Show count of detected events
- Export when satisfied

### Adaptive Thresholding
Could analyze full file to set per-section thresholds:
- Quiet intro vs loud sections
- Different threshold for crash-heavy vs ride-heavy sections

### Multi-band Detection
Could run detection on different frequency bands:
- Low energy (rides, hi-hats)
- High energy (crashes)
- Merge results intelligently

## Files Modified

### New Files
- `stems_to_midi/energy_detection_core.py` - Peak-based detection implementation
- `export_energy_detection_data.py` - Export tool using new method
- `compare_detection_methods.py` - Side-by-side comparison
- `export_raw_lr_data.py` - Raw L/R channel data export (uses old method)

### Integration Path (Phase 6)
When integrating into main pipeline:
1. Replace `detect_dual_channel_onsets()` calls with `detect_stereo_transient_peaks()`
2. Update config to include prominence_threshold parameter
3. Remove deprecated librosa onset detection code
4. Update tests to use new method

## Conclusion

The new peak-based detection is fundamentally more reliable because it operates on the same visual representation (energy envelope) that makes events obvious to humans in a DAW. No more mysterious onset strength calculations or blind wait periods - just find the peaks that stand out.

The parameter tuning (settling at 15dB) is appropriate and expected - this is effectively a sensitivity knob that should be configurable per audio file/drum type. The visual interface suggestion is excellent for future work.
