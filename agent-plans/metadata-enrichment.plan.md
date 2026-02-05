# Comprehensive Metadata Enrichment Plan

## Overview

This plan addresses four interconnected improvements to the detection and analysis pipeline:

1. **Event Duration in Analysis**: Pass full event duration to spectral analysis instead of fixed windows
2. **Stem Label Standardization**: Unify all stems to use consistent naming (primary/secondary/tertiary)
3. **Extended Metadata Collection**: Add attack sharpness, pitch, spectral features, gaps, etc.
4. **Complete JSON Output**: Export ALL metadata for post-processing and debugging

**Total estimated effort**: 3-4 phases over multiple sessions

---

## Phase 1: Foundation - Event Duration & Standardization

**Goal**: Establish architectural foundation for rich metadata collection

### 1.1: Calculate Event Duration During Detection

**Files**: `stems_to_midi/detection_core.py`

Currently onsets are just time points. Need to calculate duration (time to next onset or silence).

**Implementation**:
```python
def calculate_event_durations(onset_times, audio, sr, silence_threshold_db=-40):
    """
    Calculate duration for each onset event.
    
    Duration = min(time_to_next_onset, time_to_silence)
    
    Args:
        onset_times: Array of onset times in seconds
        audio: Full audio signal
        sr: Sample rate
        silence_threshold_db: dB threshold for silence detection
        
    Returns:
        durations: Array of durations in seconds for each onset
    """
    durations = np.zeros(len(onset_times))
    
    for i, onset_time in enumerate(onset_times):
        onset_sample = int(onset_time * sr)
        
        # Get next onset time
        if i < len(onset_times) - 1:
            next_onset_time = onset_times[i + 1]
        else:
            next_onset_time = len(audio) / sr  # End of file
        
        # Find silence threshold crossing
        segment_end = int(next_onset_time * sr)
        segment = audio[onset_sample:segment_end]
        
        # Calculate RMS in small windows
        window_ms = 10
        window_samples = int(window_ms * sr / 1000)
        
        silence_sample = None
        for j in range(0, len(segment), window_samples):
            window = segment[j:j+window_samples]
            if len(window) == 0:
                break
            rms_db = 20 * np.log10(np.sqrt(np.mean(window**2)) + 1e-10)
            if rms_db < silence_threshold_db:
                silence_sample = onset_sample + j
                break
        
        # Duration = whichever comes first: next onset or silence
        if silence_sample is not None:
            durations[i] = (silence_sample - onset_sample) / sr
        else:
            durations[i] = next_onset_time - onset_time
    
    return durations
```

**Integration point**: Call in `detect_onsets_energy_based()` and `detect_onsets()` (librosa mode), return durations alongside onset_times.

**Modified return signature**:
```python
return {
    'onset_times': onset_times,
    'onset_strengths': onset_strengths,
    'peak_amplitudes': peak_amplitudes,
    'durations': durations  # NEW
}
```

### 1.2: Pass Duration to Spectral Analysis

**Files**: `stems_to_midi/analysis_core.py` - `analyze_onset_spectral()`

**Current signature**:
```python
def analyze_onset_spectral(audio, onset_time, sr, stem_type, config):
```

**New signature**:
```python
def analyze_onset_spectral(audio, onset_time, sr, stem_type, config, duration=None):
    """
    ...
    
    Args:
        ...
        duration: Duration of event in seconds (if known). If None, uses fixed window.
    """
```

**Usage of duration**:
- Calculate `amplitude_at_end` using onset_time + duration
- Measure envelope shape over full event (attack_sharpness)
- Calculate spectral features over actual event length
- Compute sustain relative to actual duration

### 1.3: Standardize Stem Label Names

**Files**: `stems_to_midi/analysis_core.py` - `get_spectral_config_for_stem()`

**Current behavior**: Returns different labels per stem (FundE, BodyE, WireE, SizzleE, etc.)

**New behavior**: Return standardized labels + human-readable aliases

**Modified return structure**:
```python
return {
    'freq_ranges': {
        'primary': (40, 80),      # Always present
        'secondary': (80, 150),   # Always present
        'tertiary': (2000, 6000)  # Optional (None if not configured)
    },
    'energy_labels': {
        'primary': 'primary_energy',    # NEW: standardized
        'secondary': 'secondary_energy',
        'tertiary': 'tertiary_energy'   # Or None
    },
    'energy_aliases': {  # NEW: for human readability
        'primary': 'FundE',      # Kick fundamental
        'secondary': 'BodyE',    # Kick body
        'tertiary': 'AttackE'    # Kick attack
    },
    'geomean_threshold': 150.0,
    'min_sustain_ms': None  # or value
}
```

**Update all stem configs**:
- Kick: primary=fundamental, secondary=body, tertiary=attack
- Snare: primary=low, secondary=body, tertiary=wire
- Toms: primary=fundamental, secondary=body, tertiary=None
- Hihat: primary=body, secondary=sizzle, tertiary=None
- Cymbals: primary=body, secondary=brilliance, tertiary=None

**Benefit**: Generic code can now process all stems identically using primary/secondary/tertiary.

---

## Phase 2: Extended Metadata Collection

**Goal**: Implement all new metadata calculations

### 2.1: Attack Sharpness

**New function in `analysis_core.py`**:
```python
def calculate_attack_sharpness(audio, onset_sample, duration_samples, sr):
    """
    Measure steepness of envelope attack.
    
    Sharp attacks (kick, snare) have steep rise.
    Reverb tails have gradual/flat rise.
    
    Returns:
        sharpness: 0-1, higher = sharper attack
    """
    from scipy.signal import hilbert
    from scipy.ndimage import gaussian_filter1d
    
    # Get attack portion (first 20% of duration or 50ms, whichever smaller)
    attack_duration = min(int(duration_samples * 0.2), int(0.050 * sr))
    if attack_duration < 10:
        return 0.0  # Too short to analyze
    
    # Extract attack segment
    attack_segment = audio[onset_sample:onset_sample+attack_duration]
    
    # Calculate envelope
    analytic = hilbert(attack_segment)
    envelope = np.abs(analytic)
    
    # Smooth envelope
    smooth = gaussian_filter1d(envelope, sigma=3)
    
    # Calculate maximum slope (rate of rise)
    gradient = np.gradient(smooth)
    max_slope = np.max(gradient) if len(gradient) > 0 else 0.0
    
    # Normalize to 0-1 range
    # Typical sharp attacks: slope > 0.01 per sample
    # Gradual/reverb: slope < 0.002 per sample
    normalized = min(1.0, max_slope / 0.01)
    
    return normalized
```

### 2.2: Envelope Continuity

**New function in `analysis_core.py`**:
```python
def calculate_envelope_continuity(audio, onset_sample, sr):
    """
    Measure energy continuity before onset.
    
    High continuity = energy doesn't drop before onset (reverb tail)
    Low continuity = energy drops then rises (new hit)
    
    Returns:
        continuity: 0-1, higher = more continuous
    """
    # Window sizes
    pre_window_ms = 50  # 50ms before onset
    peak_window_ms = 10  # 10ms at peak
    
    pre_samples = int(pre_window_ms * sr / 1000)
    peak_samples = int(peak_window_ms * sr / 1000)
    
    # Extract segments
    pre_start = max(0, onset_sample - pre_samples)
    pre_segment = audio[pre_start:onset_sample]
    peak_segment = audio[onset_sample:onset_sample+peak_samples]
    
    if len(pre_segment) == 0 or len(peak_segment) == 0:
        return 0.0
    
    # Calculate RMS energies
    energy_before = np.sqrt(np.mean(pre_segment**2))
    energy_at_peak = np.sqrt(np.mean(peak_segment**2))
    
    # Ratio (clamped to 0-1)
    continuity = energy_before / (energy_at_peak + 1e-10)
    return min(1.0, continuity)
```

### 2.3: Gap from Previous Event

**New function in `analysis_core.py`**:
```python
def calculate_gaps_from_previous(onset_times, durations):
    """
    Calculate time gap from previous event's end to current event's start.
    
    Args:
        onset_times: Array of onset times
        durations: Array of durations
        
    Returns:
        gaps_ms: Array of gaps in milliseconds (first event = None)
    """
    gaps = np.full(len(onset_times), np.nan)
    
    for i in range(1, len(onset_times)):
        prev_end_time = onset_times[i-1] + durations[i-1]
        curr_start_time = onset_times[i]
        gap_sec = curr_start_time - prev_end_time
        gaps[i] = max(0.0, gap_sec * 1000)  # Convert to ms, clamp to 0
    
    return gaps
```

### 2.4: Amplitude at Start/End

**New function in `analysis_core.py`**:
```python
def calculate_amplitude_at_time(audio, time_sample, sr, window_ms=10):
    """
    Calculate RMS amplitude at a specific time point.
    
    Args:
        audio: Full audio signal
        time_sample: Sample index
        sr: Sample rate
        window_ms: Window size for RMS calculation
        
    Returns:
        amplitude: RMS amplitude (linear scale)
    """
    window_samples = int(window_ms * sr / 1000)
    start = max(0, time_sample - window_samples // 2)
    end = min(len(audio), time_sample + window_samples // 2)
    
    segment = audio[start:end]
    if len(segment) == 0:
        return 0.0
    
    return np.sqrt(np.mean(segment**2))
```

### 2.5: Spectral Centroid

**New function in `analysis_core.py`**:
```python
def calculate_spectral_centroid(audio_segment, sr):
    """
    Calculate spectral centroid (brightness).
    
    NOT pitch - this is weighted average of all frequencies.
    High centroid = bright (cymbals, snare wire)
    Low centroid = dark (kick body)
    
    Returns:
        centroid_hz: Spectral centroid in Hz
    """
    from scipy.signal import stft
    
    if len(audio_segment) < 512:
        return 0.0
    
    f, t, Zxx = stft(audio_segment, fs=sr, nperseg=512)
    magnitude = np.abs(Zxx)
    
    # Weighted average frequency
    centroid = np.sum(f[:, np.newaxis] * magnitude, axis=0) / (np.sum(magnitude, axis=0) + 1e-10)
    
    # Average over time
    return float(np.mean(centroid))
```

### 2.6: Spectral Flux

**New function in `analysis_core.py`**:
```python
def calculate_spectral_flux(audio_segment, sr):
    """
    Calculate spectral flux (rate of spectral change).
    
    High flux = spectrum changing rapidly (new hit)
    Low flux = spectrum stable (sustain, reverb)
    
    Returns:
        flux: Average spectral flux
    """
    from scipy.signal import stft
    
    if len(audio_segment) < 1024:
        return 0.0
    
    f, t, Zxx = stft(audio_segment, fs=sr, nperseg=512, noverlap=256)
    magnitude = np.abs(Zxx)
    
    # Calculate flux between consecutive frames
    flux = np.sum(np.diff(magnitude, axis=1)**2, axis=0)
    
    return float(np.mean(flux))
```

### 2.7: Pitch Detection

**New function in `analysis_core.py`**:
```python
def detect_pitch(audio_segment, sr, fmin=40, fmax=400):
    """
    Detect fundamental frequency (pitch) using librosa.pyin.
    
    Useful for toms, less reliable for kick/snare/cymbals.
    
    Args:
        audio_segment: Audio to analyze
        sr: Sample rate
        fmin: Minimum frequency to consider (Hz)
        fmax: Maximum frequency to consider (Hz)
        
    Returns:
        pitch_hz: Detected pitch in Hz, or None if no pitch detected
    """
    import librosa
    
    if len(audio_segment) < 2048:
        return None
    
    # Use pyin for probabilistic pitch detection
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio_segment,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=2048
    )
    
    # Get most confident pitch estimate
    if voiced_probs is not None and len(voiced_probs) > 0:
        # Filter to voiced frames only
        valid_indices = np.where(voiced_flag)[0]
        if len(valid_indices) > 0:
            # Weight by confidence
            valid_pitches = f0[valid_indices]
            valid_probs = voiced_probs[valid_indices]
            
            # Remove NaN
            mask = ~np.isnan(valid_pitches)
            if np.any(mask):
                valid_pitches = valid_pitches[mask]
                valid_probs = valid_probs[mask]
                
                # Weighted average
                pitch = np.average(valid_pitches, weights=valid_probs)
                return float(pitch)
    
    return None
```

### 2.8: Peak Prominence

**New function in `analysis_core.py`**:
```python
def calculate_peak_prominence(audio, onset_sample, sr):
    """
    Calculate how much the peak stands out from surroundings.
    
    Uses scipy.signal.find_peaks prominence calculation.
    
    Returns:
        prominence: Normalized 0-1, higher = more prominent
    """
    from scipy.signal import find_peaks, peak_prominences
    from scipy.ndimage import gaussian_filter1d
    
    # Get envelope around onset
    window_samples = int(0.2 * sr)  # 200ms window
    start = max(0, onset_sample - window_samples // 2)
    end = min(len(audio), onset_sample + window_samples // 2)
    segment = np.abs(audio[start:end])
    
    if len(segment) < 10:
        return 0.0
    
    # Smooth
    smooth = gaussian_filter1d(segment, sigma=10)
    
    # Find peaks
    peaks, _ = find_peaks(smooth, distance=10)
    
    if len(peaks) == 0:
        return 0.0
    
    # Calculate prominences
    prominences = peak_prominences(smooth, peaks)[0]
    
    # Find our onset peak (closest to center)
    center_idx = onset_sample - start
    closest_peak_idx = np.argmin(np.abs(peaks - center_idx))
    
    prominence = prominences[closest_peak_idx]
    
    # Normalize by segment max amplitude
    max_amp = np.max(segment)
    return float(min(1.0, prominence / (max_amp + 1e-10)))
```

---

## Phase 3: Integration & JSON Output

**Goal**: Wire up all metadata collection and export to JSON

### 3.1: Modify `filter_onsets_by_spectral()`

**File**: `stems_to_midi/analysis_core.py`

**Current signature**:
```python
def filter_onsets_by_spectral(onset_times, onset_strengths, peak_amplitudes, audio, sr, stem_type, config, learning_mode=False):
```

**New signature**:
```python
def filter_onsets_by_spectral(onset_times, onset_strengths, peak_amplitudes, durations, audio, sr, stem_type, config, learning_mode=False):
    # durations parameter added ^
```

**Inside loop, collect all metadata**:
```python
for i, onset_time in enumerate(onset_times):
    duration = durations[i]
    duration_samples = int(duration * sr)
    onset_sample = int(onset_time * sr)
    
    # Existing spectral analysis
    spectral_data = analyze_onset_spectral(audio, onset_time, sr, stem_type, config, duration=duration)
    
    if spectral_data is None:
        continue
    
    # NEW: Extended metadata
    metadata = {
        **spectral_data,  # Includes primary/secondary/tertiary energy, geomean, sustain
        
        # Transient characteristics
        'attack_sharpness': calculate_attack_sharpness(audio, onset_sample, duration_samples, sr),
        'peak_prominence': calculate_peak_prominence(audio, onset_sample, sr),
        
        # Temporal relationships
        'gap_from_previous_ms': gaps[i] if i > 0 else None,
        'duration': duration,
        
        # Amplitude tracking
        'amplitude_at_start': calculate_amplitude_at_time(audio, onset_sample, sr),
        'amplitude_at_end': calculate_amplitude_at_time(audio, onset_sample + duration_samples, sr),
        'previous_amplitude_at_end': prev_amplitude_end if i > 0 else None,
        
        # Spectral features
        'spectral_centroid_hz': calculate_spectral_centroid(audio[onset_sample:onset_sample+duration_samples], sr),
        'spectral_flux': calculate_spectral_flux(audio[onset_sample:onset_sample+duration_samples], sr),
        
        # Pitch (optional, may be None)
        'pitch_hz': detect_pitch(audio[onset_sample:onset_sample+duration_samples], sr),
        
        # Envelope characteristics
        'envelope_continuity': calculate_envelope_continuity(audio, onset_sample, sr),
        
        # Original detection data
        'onset_strength': onset_strengths[i],
        'peak_amplitude': peak_amplitudes[i],
        'time': onset_time
    }
    
    # Store for next iteration
    prev_amplitude_end = metadata['amplitude_at_end']
    
    # Decision: keep or reject
    is_kept = should_keep_onset(...)
    metadata['is_kept'] = is_kept
    
    all_onset_data.append(metadata)
    
    if is_kept or learning_mode:
        filtered_times.append(onset_time)
        filtered_metadata.append(metadata)
```

### 3.2: Update JSON Output

**File**: `stems_to_midi/midi.py` - `save_analysis_sidecar()`

**Current behavior**: Saves partial metadata with stem-specific labels

**New behavior**: Save ALL metadata with standardized labels

**Modified function**:
```python
def save_analysis_sidecar(events_by_stem, midi_path, tempo, analysis_by_stem=None):
    """
    Save comprehensive analysis data as JSON sidecar.
    
    Now includes:
    - All spectral energies (standardized labels)
    - Attack sharpness, envelope continuity
    - Gap analysis, amplitude tracking
    - Spectral features (centroid, flux)
    - Pitch detection results
    - Reverb classification flags (when implemented)
    """
    sidecar_data = {
        'version': '3.0',  # Increment version for new format
        'tempo': tempo,
        'generated_at': datetime.now().isoformat(),
        'events': []
    }
    
    for stem_type, events in events_by_stem.items():
        for event in events:
            # Include ALL metadata fields
            event_data = {
                'stem_type': stem_type,
                'note': event['note'],
                'time': event['time'],
                'velocity': event['velocity'],
                'duration': event.get('duration', 0.1),
                
                # Standardized spectral energies
                'primary_energy': event.get('primary_energy'),
                'secondary_energy': event.get('secondary_energy'),
                'tertiary_energy': event.get('tertiary_energy'),
                'geomean': event.get('geomean'),
                'sustain_ms': event.get('sustain_ms'),
                
                # Detection metadata
                'onset_strength': event.get('onset_strength'),
                'peak_amplitude': event.get('peak_amplitude'),
                
                # Transient characteristics
                'attack_sharpness': event.get('attack_sharpness'),
                'peak_prominence': event.get('peak_prominence'),
                
                # Temporal relationships
                'gap_from_previous_ms': event.get('gap_from_previous_ms'),
                
                # Amplitude tracking
                'amplitude_at_start': event.get('amplitude_at_start'),
                'amplitude_at_end': event.get('amplitude_at_end'),
                'previous_amplitude_at_end': event.get('previous_amplitude_at_end'),
                
                # Spectral features
                'spectral_centroid_hz': event.get('spectral_centroid_hz'),
                'spectral_flux': event.get('spectral_flux'),
                'pitch_hz': event.get('pitch_hz'),
                
                # Envelope characteristics
                'envelope_continuity': event.get('envelope_continuity'),
                
                # Classification (when reverb filter implemented)
                'is_likely_reverb': event.get('is_likely_reverb', False),
                'reverb_confidence': event.get('reverb_confidence', 0.0)
            }
            
            sidecar_data['events'].append(event_data)
    
    # Save to JSON
    sidecar_path = midi_path.with_suffix('.analysis.json')
    with open(sidecar_path, 'w') as f:
        json.dump(sidecar_data, f, indent=2)
```

---

## Phase 4: Reverb Filter Implementation

**Goal**: Use collected metadata to filter reverb tails

See detailed implementation in `reverb-tail-detection.plan.md`.

**Summary**:
1. Create `stems_to_midi/reverb_filter.py` module
2. Implement `filter_reverb_tails()` using metadata
3. Integrate into `processing_shell.py`
4. Add config options per stem
5. Add learning mode visualization (velocity=2 for reverb)

---

## Testing Strategy

### Unit Tests

**File**: `stems_to_midi/test_analysis_core.py`

Add tests for each new function:
- `test_calculate_attack_sharpness()`
- `test_calculate_envelope_continuity()`
- `test_calculate_gaps_from_previous()`
- `test_calculate_amplitude_at_time()`
- `test_calculate_spectral_centroid()`
- `test_calculate_spectral_flux()`
- `test_detect_pitch()`
- `test_calculate_peak_prominence()`
- `test_calculate_event_durations()`

### Integration Tests

**Test datasets**:
1. **AC/DC Thunderstruck toms**: 3 hits with reverb tails
2. **Fast kick rolls**: Ensure gaps detected correctly
3. **Sparse hihat patterns**: Verify attack sharpness vs librosa
4. **Cymbal crashes**: Check sustain + spectral features

### Validation

**JSON output**:
- Load generated `.analysis.json` files
- Verify all fields present and valid types
- Check standardized labels used (primary/secondary/tertiary)
- Confirm metadata aligns across stems

**Learning mode**:
- Generate MIDI with all metadata
- Import to DAW
- Verify velocity markers (1=false positive, 2=reverb, >40=kept)

---

## Migration Path

### Backward Compatibility

**Option 1**: Dual labels during transition
```python
# In JSON output, include both old and new labels
'primary_energy': 1250.5,
'FundE': 1250.5,  # Alias for compatibility
```

**Option 2**: Version flag
```json
{
  "version": "3.0",
  "schema": "standardized_labels",
  ...
}
```

### Config Updates

**Old format** (still supported):
```yaml
kick:
  fundamental_freq_min: 40
  fundamental_freq_max: 80
```

**New format** (recommended):
```yaml
kick:
  spectral_ranges:
    primary: [40, 80]
    secondary: [80, 150]
    tertiary: [2000, 6000]
```

---

## Success Criteria

- [ ] Event duration calculated and passed to all analysis functions
- [ ] All stems use standardized primary/secondary/tertiary labels
- [ ] All 8 new metadata functions implemented and tested
- [ ] JSON output includes complete metadata (20+ fields per event)
- [ ] Backward compatibility maintained (tests pass)
- [ ] Tom reverb filtering reduces false positives by >80%
- [ ] Fast rolls NOT filtered (gap detection works)
- [ ] Documentation updated with new schema
- [ ] Learning mode exports all metadata for debugging

---

## Timeline Estimate

**Phase 1**: 2-3 hours (foundation)
**Phase 2**: 3-4 hours (metadata collection functions)
**Phase 3**: 2-3 hours (integration & JSON)
**Phase 4**: 2-3 hours (reverb filter)
**Testing/Polish**: 2 hours

**Total**: 11-15 hours over multiple sessions

---

## Next Steps

1. **Review this plan** - Confirm approach and priorities
2. **Start Phase 1** - Event duration + label standardization
3. **Test incrementally** - Run comparison tests after each phase
4. **Iterate** - Adjust thresholds based on real data
5. **Document** - Update architecture docs as we go
