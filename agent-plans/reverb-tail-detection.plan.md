# Reverb Tail Detection & Post-Processing Plan

## Problem Statement

Tom hits (and potentially other drums) trigger false detections during reverb tails. Current detection finds legitimate transients but also spurious events during the decay phase where energy wavers and fades.

**Key observations from toms waveform**:
1. Events during reverb start immediately when previous event ends (no gap)
2. No sharp attack (gradual energy rise, not steep transient)
3. Continuous envelope energy (no significant drop before onset)
4. Amplitude at start matches previous event's end amplitude

## Additional Requirements

### 1. Event Duration in Spectral Analysis
**Current limitation**: `filter_onsets_by_spectral()` uses fixed-length windows for analysis, which may not capture the full event characteristics.

**Needed**: Pass event duration/length to spectral analysis so we can:
- Analyze the entire event from onset to decay
- Better distinguish short transients (kick, snare) from sustained events (cymbals, reverb tails)
- Calculate envelope characteristics over the actual event duration
- Improve attack sharpness measurement by knowing the full envelope shape

### 2. Stem Label Standardization
**Current state**: Stems have different frequency range names and capabilities:
- Kick: FundE, BodyE, AttackE (3-range)
- Snare: LowE, BodyE, WireE (3-range)
- Toms: FundE, BodyE (2-range)
- Hihat: BodyE, SizzleE (2-range)
- Cymbals: BodyE, BrillE (2-range, hardcoded ranges)

**Problem**: Inconsistent naming makes generic processing difficult, limits extensibility.

**Needed**: Standardize all stems to share:
- Consistent label naming (Primary, Secondary, Tertiary)
- Unified 3-range capability (or optional third range)
- Same metadata fields available for all stems
- Consistent configuration structure

## Proposed Architecture

**Principle**: Separate detection from classification. Collect rich metadata during detection, then intelligently filter in post-processing.

### Event Metadata Structure

Extend each detected MIDI event with reverb/artifact detection metadata:

```python
event = {
    # Existing data
    'time': 1.234,
    'velocity': 85,
    'note': 36,
    'duration': 0.1,              # IMPORTANT: Now used for full-event analysis
    
    # NEW: Reverb tail detection metadata
    'attack_sharpness': 0.87,           # 0-1, higher = sharper attack (envelope derivative)
    'gap_from_previous_ms': 15.3,       # Time since last event ended
    'envelope_continuity': 0.92,        # 0-1, higher = more continuous with previous
    'amplitude_at_start': 0.15,         # RMS amplitude at onset
    'amplitude_at_end': 0.08,           # RMS amplitude at event end (using duration)
    'previous_amplitude_at_end': 0.14,  # Previous event's end amplitude
    'peak_prominence': 0.45,            # How much peak stands out from surroundings (scipy)
    'spectral_centroid_hz': 3500,       # Brightness - weighted avg frequency (NOT pitch)
    'spectral_flux': 0.23,              # Rate of spectral change between frames
    'pitch_hz': 85.0,                   # Detected fundamental frequency (librosa.pyin)
    
    # Standardized spectral energies (ALL stems)
    'primary_energy': 1250.5,           # Renamed from stem-specific labels
    'secondary_energy': 850.3,
    'tertiary_energy': 2100.7,          # Optional third range
    'geomean': 1050.2,
    'sustain_ms': 45.0,                 # For all stems (not just hihat/cymbals)
    
    # Computed classification flags
    'is_likely_reverb': False,
    'reverb_confidence': 0.05           # 0-1 confidence this is reverb
}
```

## Detection Heuristics

### Reverb Tail Indicators
An event is likely a reverb tail if **all** are true:

1. **No sharp attack**: `attack_sharpness < 0.3`
   - Measured as max envelope derivative in 10ms before peak
   - Sharp attacks have steep rise (>0.7), reverb tails have gradual rise

2. **Continuous from previous**: `gap_from_previous_ms < 50ms`
   - Previous event end time to current event start time
   - Reverb tails start during/immediately after previous event decay
   - Legitimate fast rolls have gaps >50ms

3. **No energy drop**: `envelope_continuity > 0.7`
   - Compare RMS before onset to RMS at peak
   - High value = continuous energy (no significant drop/silence)
   - Legitimate hits show energy drop before transient

4. **Similar amplitude to previous end**: `|amplitude_at_start - previous_amplitude_at_end| / previous_amplitude_at_end < 0.3`
   - Reverb tails have similar amplitude to where previous left off
   - New hits typically have amplitude increase

5. **Low spectral change**: `spectral_flux < 0.2` (optional)
   - Reverb maintains similar spectral profile
   - New hits change spectral content significantly

## Implementation Phases

### Phase 1: Metadata Collection (modify detection core)

**Location**: `stems_to_midi/analysis_core.py` - `analyze_onset_spectral()` or similar

Add calculations:
```python
def calculate_attack_sharpness(audio_segment, onset_sample, sr):
    """Measure steepness of envelope attack"""
    envelope = np.abs(hilbert(audio_segment))
    smooth = gaussian_filter1d(envelope, sigma=50)
    attack_slope = np.gradient(smooth[:onset_sample]).max()
    return min(1.0, attack_slope / threshold)

def calculate_envelope_continuity(audio, onset_sample, sr):
    """Measure energy continuity before onset"""
    pre_window = int(0.050 * sr)  # 50ms before
    peak_window = int(0.010 * sr)  # 10ms at peak
    energy_before = rms(audio[onset_sample-pre_window:onset_sample])
    energy_at_peak = rms(audio[onset_sample:onset_sample+peak_window])
    return energy_before / (energy_at_peak + 1e-10)

def calculate_spectral_flux(audio_segment, sr):
    """Measure rate of spectral change"""
    f, t, Zxx = stft(audio_segment, fs=sr, nperseg=512)
    spec = np.abs(Zxx)
    flux = np.sum(np.diff(spec, axis=1)**2, axis=0)
    return np.mean(flux)
```

Store in onset analysis data returned by `analyze_onset_spectral()`.

### Phase 2: Post-Processor Module

**New file**: `stems_to_midi/reverb_filter.py`

```python
def filter_reverb_tails(events, stem_type, config=None):
    """
    Remove likely reverb tails based on event metadata.
    
    Args:
        events: List of event dicts with metadata
        stem_type: 'kick', 'snare', 'toms', 'hihat', 'cymbals'
        config: Optional per-stem thresholds
        
    Returns:
        filtered_events: Events with reverb tails removed
        removed_events: Events classified as reverb (for debugging)
    """
    if len(events) == 0:
        return events, []
    
    # Get thresholds from config or use defaults
    thresholds = get_reverb_filter_thresholds(stem_type, config)
    
    filtered = []
    removed = []
    
    for i, event in enumerate(events):
        if i == 0:
            filtered.append(event)  # First event can't be reverb
            continue
        
        prev_event = filtered[-1] if filtered else None
        
        if is_likely_reverb_tail(event, prev_event, thresholds):
            event['is_likely_reverb'] = True
            removed.append(event)
        else:
            event['is_likely_reverb'] = False
            filtered.append(event)
    
    return filtered, removed

def is_likely_reverb_tail(event, previous_event, thresholds):
    """Classify event as reverb tail using multiple heuristics"""
    if previous_event is None:
        return False
    
    # Calculate confidence score for each heuristic
    scores = []
    
    # 1. Attack sharpness (inverted - low = reverb)
    if event.get('attack_sharpness') is not None:
        scores.append(1.0 - event['attack_sharpness'])
    
    # 2. Gap from previous (small = reverb)
    if event.get('gap_from_previous_ms') is not None:
        gap_score = 1.0 - min(1.0, event['gap_from_previous_ms'] / 100.0)
        scores.append(gap_score)
    
    # 3. Envelope continuity (high = reverb)
    if event.get('envelope_continuity') is not None:
        scores.append(event['envelope_continuity'])
    
    # 4. Amplitude similarity to previous end
    if all(k in event for k in ['amplitude_at_start', 'previous_amplitude_at_end']):
        amp_diff = abs(event['amplitude_at_start'] - event['previous_amplitude_at_end'])
        amp_ratio = amp_diff / (event['previous_amplitude_at_end'] + 1e-10)
        amp_score = 1.0 - min(1.0, amp_ratio / 0.5)  # <50% diff = suspicious
        scores.append(amp_score)
    
    # Average confidence
    confidence = np.mean(scores) if scores else 0.0
    event['reverb_confidence'] = confidence
    
    # Threshold decision
    return confidence > thresholds.get('reverb_confidence_threshold', 0.7)
```

### Phase 3: Integration

**Modify**: `stems_to_midi/processing_shell.py` - `process_stem_to_midi()`

```python
# After detection, before creating MIDI events
if config.get(stem_type, {}).get('filter_reverb_tails', False):
    from stems_to_midi.reverb_filter import filter_reverb_tails
    
    events, removed = filter_reverb_tails(
        events=result['events'],
        stem_type=stem_type,
        config=config
    )
    
    result['events'] = events
    result['reverb_removed_count'] = len(removed)
    
    if learning_mode:
        # Include removed events with velocity=2 (reverb marker)
        for rev_event in removed:
            rev_event['velocity'] = 2  # Different from false positive (1)
        result['events'].extend(removed)
```

**Config addition** (`midiconfig.yaml`):
```yaml
toms:
  filter_reverb_tails: true
  reverb_filter_thresholds:
    attack_sharpness_max: 0.3
    gap_ms_max: 50
    envelope_continuity_min: 0.7
    amplitude_similarity_threshold: 0.3
    reverb_confidence_threshold: 0.7
```

## Benefits

1. **Debuggable**: Export all metadata to sidecar JSON, visualize in DAW
2. **Tunable**: Adjust thresholds per-stem without re-detecting
3. **Transparent**: Learning mode shows "rejected: reverb (velocity=2)" vs "false positive (velocity=1)"
4. **Non-destructive**: Reverb filter is optional, can be toggled
5. **Extensible**: Easy to add more heuristics (zero-crossing rate, cepstral analysis, etc.)
6. **Accurate**: Multi-factor scoring more robust than single threshold

## Testing Strategy

1. **Toms with reverb**: AC/DC Thunderstruck (3 clear hits + long tails)
2. **Fast rolls**: Should NOT be filtered (gaps >50ms, sharp attacks)
3. **Sparse cymbals**: Decay filter vs reverb filter comparison
4. **Dense patterns**: Ensure legitimate hits in busy sections not filtered

## Success Criteria
Architecture Improvements Required

Before implementing reverb detection, these foundation improvements are needed:

### 1. Pass Event Duration to Spectral Analysis
Currently `filter_onsets_by_spectral()` uses fixed windows. Need to:
- Calculate event duration during detection (time to next onset or decay threshold)
- Pass duration to `analyze_onset_spectral()` 
- Use duration for amplitude_at_end, full envelope analysis
- Enables accurate attack_sharpness over complete event shape

### 2. Standardize Stem Metadata Structure
Currently stems have different labels (FundE, BodyE, WireE, etc.). Need to:
- Rename all to primary_energy, secondary_energy, tertiary_energy
- Make tertiary_energy optional (None if not configured)
- Update JSON output to use standardized names
- Update visualization/debugging tools to expect consistent structure
- Maintain backward compatibility by keeping old labels as aliases during transition

## Future Enhancements

- **Adaptive thresholds**: Learn from user edits in learning mode
- **Stem-specific profiles**: Toms vs cymbals have different reverb characteristics  
- **Machine learning**: Train classifier on labeled reverb vs legitimate hits
- **Visual debugging**: Waveform overlay showing attack_sharpness, gaps, continuity
- **Pitch-based classification**: Use pitch detection to improve tom pitch classification accurac

## Future Enhancements

- **Adaptive thresholds**: Learn from user edits in learning mode
- **Stem-specific profiles**: Toms vs cymbals have different reverb characteristics  
- **Machine learning**: Train classifier on labeled reverb vs legitimate hits
- **Visual debugging**: Waveform overlay showing attack_sharpness, gaps, continuity
