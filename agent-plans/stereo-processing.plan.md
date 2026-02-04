# Stereo Processing Implementation Plan

## Objective
Enable stereo audio analysis for stem-to-MIDI conversion to leverage spatial information (pan position) for better instrument identification. Currently, all stems are converted to mono by averaging channels, discarding valuable spatial data.

## Problem Statement
Drums are often panned in stereo mixes:
- **Cymbals**: Spread across stereo field (e.g., crash left, ride right)
- **Toms**: Panned left to right (floor tom left, rack tom right)
- **Hi-hat**: Often slightly off-center
- **Kick/Snare**: Usually centered

The Thunderstruck test case has cymbals at distinct pan positions (left and right), but we currently average them to mono, losing the ability to distinguish between them.

## Current Architecture
1. **Audio Loading**: `stems_to_midi/processing_shell.py::_load_and_validate_audio()`
   - Loads audio with `soundfile.read()`
   - Checks global `config['audio']['force_mono']` setting
   - Calls `ensure_mono()` if stereo and force_mono=True
2. **Mono Conversion**: `stems_to_midi/analysis_core.py::ensure_mono()`
   - Averages left and right channels: `np.mean(audio, axis=1)`
3. **Detection**: All detection functions expect mono audio
4. **Configuration**: Single global `force_mono` boolean applies to all stems

## Requirements
1. **Per-stem stereo control**: Each stem type should have independent stereo/mono setting
2. **Default behavior**: Kick=mono (centered, doesn't benefit), others=stereo
3. **Optional feature**: Can be disabled per-stem via configuration
4. **Stereo analysis**: Extract pan position and channel separation metadata
5. **Backwards compatibility**: Existing mono processing should still work
6. **Test case**: Thunderstruck cymbals (left vs right pan positions)

## Implementation Phases

### Phase 1: Configuration Infrastructure (30 min)
**Goal**: Add per-stem stereo settings to configuration system

**Changes**:
1. **midiconfig.yaml**: Add `use_stereo` boolean to each stem section
   ```yaml
   kick:
     use_stereo: false  # Kick is centered, no benefit
   snare:
     use_stereo: true   # May have ghost notes panned
   toms:
     use_stereo: true   # Often panned left-right
   hihat:
     use_stereo: true   # May be off-center
   cymbals:
     use_stereo: true   # Frequently panned
   ```

2. **webui/settings_schema.py**: Add SettingDefinition for each stem's stereo setting
   - Type: BOOL
   - Category: Per-stem (KICK, SNARE, TOMS, HIHAT, CYMBALS)
   - UI Control: CHECKBOX
   - Default: kick=False, others=True
   - Description: "Process this stem in stereo to use pan position for instrument identification"

**Success Criteria**:
- [ ] Config loads with per-stem use_stereo values
- [ ] Settings schema validates and provides UI metadata
- [ ] Backwards compatible (missing values use sensible defaults)

---

### Phase 2: Stereo Analysis Core Functions (45 min)
**Goal**: Pure functions to analyze stereo audio characteristics

**New file**: `stems_to_midi/stereo_core.py` (functional core)

**Functions**:
1. **`calculate_pan_position(stereo_audio, onset_sample, sr, window_ms=10.0) -> float`**
   - Analyzes amplitude difference between L/R channels around onset
   - Returns pan value: -1.0 (full left) to +1.0 (full right), 0.0 (center)
   - Pure function: deterministic, no side effects

2. **`separate_channels(stereo_audio) -> tuple[np.ndarray, np.ndarray]`**
   - Extracts left and right channels from stereo array
   - Returns: (left_mono, right_mono)
   - Pure function

3. **`detect_stereo_onsets(stereo_audio, sr, **librosa_params) -> StereoOnsetData`**
   - Runs onset detection on L, R, and mono (averaged) separately
   - Returns dict with:
     - `left_onsets`: np.ndarray of onset times
     - `right_onsets`: np.ndarray of onset times
     - `mono_onsets`: np.ndarray of onset times
     - `left_strengths`: np.ndarray of onset strengths
     - `right_strengths`: np.ndarray of onset strengths
   - TypedDict defined in `midi_types.py`

4. **`classify_onset_by_pan(pan_position, center_threshold=0.15) -> str`**
   - Classifies onset as 'left', 'right', or 'center' based on pan
   - Pure function
   - Returns: "left" | "right" | "center"

**Success Criteria**:
- [ ] All functions are pure (no I/O, deterministic)
- [ ] Type hints and docstrings complete
- [ ] Unit tests for each function
- [ ] Pan calculation matches manual verification on test audio

---

### Phase 3: Audio Loading Changes (30 min)
**Goal**: Keep stereo data when configured, convert to mono when not

**Changes**:
1. **`stems_to_midi/processing_shell.py::_load_and_validate_audio()`**
   - Check stem-specific `use_stereo` setting instead of global `force_mono`
   - Only call `ensure_mono()` if `use_stereo=False` for this stem
   - Keep stereo data (shape: `(samples, 2)`) if `use_stereo=True`

**Implementation**:
```python
# Old: global force_mono
if config['audio']['force_mono'] and audio.ndim == 2:
    audio = ensure_mono(audio)

# New: per-stem use_stereo
stem_config = config.get(stem_type, {})
use_stereo = stem_config.get('use_stereo', False)  # Default to mono for safety

if not use_stereo and audio.ndim == 2:
    audio = ensure_mono(audio)
    print("    Converted stereo to mono")
elif use_stereo:
    print("    Keeping stereo for spatial analysis")
```

**Success Criteria**:
- [ ] Kick loads as mono (use_stereo=False)
- [ ] Cymbals load as stereo (use_stereo=True)
- [ ] Backwards compatible with existing configs

---

### Phase 4: Detection Pipeline Updates (60 min)
**Goal**: Detection functions handle stereo input and use pan information

**Strategy**: Keep existing mono detection as default path, add stereo path

**Changes**:
1. **`stems_to_midi/detection_shell.py`**:
   - Update `detect_onsets()` to accept stereo audio
   - If stereo, run detection on L, R, and mono
   - Merge results with pan position metadata

2. **`stems_to_midi/processing_shell.py::process_stem_to_midi()`**:
   - After onset detection, if stereo, calculate pan for each onset
   - Add pan metadata to onset data structure
   - Use pan position in classification (e.g., cymbals)

**Example Flow (cymbals)**:
```python
if audio.ndim == 2:  # Stereo
    from .stereo_core import detect_stereo_onsets, calculate_pan_position
    
    stereo_data = detect_stereo_onsets(audio, sr, **onset_params)
    
    # Use strongest channel's detections
    # Enrich with pan information
    for onset in onsets:
        onset_sample = int(onset * sr)
        onset['pan_position'] = calculate_pan_position(audio, onset_sample, sr)
        onset['pan_class'] = classify_onset_by_pan(onset['pan_position'])
else:  # Mono (existing path)
    onsets = detect_onsets(audio, sr, **onset_params)
```

**Success Criteria**:
- [ ] Mono path unchanged (backwards compatible)
- [ ] Stereo path detects onsets with pan metadata
- [ ] Pan position used for cymbal classification

---

### Phase 5: Cymbal Classification Enhancement (45 min)
**Goal**: Use pan position to distinguish between cymbals (crash vs ride)

**Changes**:
1. **`stems_to_midi/analysis_core.py`**: Add pan-aware cymbal classification
   ```python
   def classify_cymbal_by_pan(
       pan_position: float,
       spectral_features: dict,
       config: dict
   ) -> int:
       """
       Classify cymbal type using pan position and spectral features.
       
       Typical panning:
       - Crash: Often left (-0.5 to -1.0) or center (-0.2 to 0.2)
       - Ride: Often right (0.5 to 1.0) or center
       - Chinese: Variable, use spectral features
       
       Returns:
           MIDI note number (crash=49, ride=51, chinese=52)
       """
   ```

2. **`stems_to_midi/processing_shell.py::_process_cymbals()`**:
   - If stereo, use pan-aware classification
   - Fall back to existing spectral-only classification if mono

**Success Criteria**:
- [ ] Thunderstruck cymbals classified differently based on pan
- [ ] Manual verification: left cymbal → crash (49), right → ride (51)
- [ ] Existing mono classification still works

---

### Phase 6: Testing & Validation (60 min)
**Goal**: Comprehensive tests and validation on Thunderstruck

**Test Strategy**:
1. **Unit tests**: `stems_to_midi/test_stereo_core.py`
   - Test `calculate_pan_position()` with synthetic stereo audio
   - Test `separate_channels()`
   - Test `classify_onset_by_pan()`

2. **Integration tests**: `test_integration.py`
   - Load Thunderstruck cymbals stem with `use_stereo=True`
   - Verify onsets have pan metadata
   - Verify left/right cymbals get different MIDI notes

3. **Manual validation**:
   - Run full pipeline on Thunderstruck with stereo enabled
   - Listen to MIDI output, verify pan-based classification
   - Compare with mono processing (should see fewer false positives)

**Success Criteria**:
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Thunderstruck cymbals correctly classified by pan
- [ ] No regressions in mono processing

---

### Phase 7: Documentation (30 min)
**Goal**: Update documentation with stereo processing info

**Changes**:
1. **docs/ARCH_DATA_FLOW.md**: Update Stage 2 (Onset Detection)
   - Note stereo processing option
   - Show pan metadata in onset data structure

2. **README.md**: Update features list
   - Add "Stereo processing for spatial instrument identification"

3. **TODO.md**: Mark stereo task as complete

4. **midiconfig.yaml**: Add comments explaining stereo settings

**Success Criteria**:
- [ ] Documentation updated and accurate
- [ ] Examples clear and helpful

---

## Risk Mitigation

### Risk 1: Performance Impact
**Concern**: Processing stereo may be slower (3x onset detections: L, R, mono)

**Mitigation**:
- Make it optional (per-stem configuration)
- Only enable for stems that benefit (cymbals, toms)
- Profile and optimize if needed

### Risk 2: Backwards Compatibility
**Concern**: Breaking existing configs and workflows

**Mitigation**:
- Default to mono if `use_stereo` not specified
- Keep global `force_mono` as override (if True, ignore per-stem settings)
- All existing tests should pass without changes

### Risk 3: Detection Quality
**Concern**: Stereo processing might introduce noise or false positives

**Mitigation**:
- Conservative defaults (only cymbals and toms benefit)
- Extensive testing on multiple songs
- Compare mono vs stereo results systematically

### Risk 4: Pan Position Unreliable
**Concern**: Some recordings may have unusual panning or phase issues

**Mitigation**:
- Make pan classification thresholds configurable
- Fall back to spectral-only classification if pan is ambiguous (near center)
- Add debug output for pan values

---

## Success Metrics
1. **Functional**: Thunderstruck cymbals classified correctly by pan position
2. **Performance**: No significant slowdown for mono processing
3. **Quality**: Improved cymbal classification accuracy vs. mono baseline
4. **Compatibility**: All existing tests pass unchanged
5. **Configurability**: Users can enable/disable per stem

---

## Testing Protocol: Thunderstruck Cymbals

**Setup**:
1. Use any Thunderstruck project (all have same cymbals)
2. Extract cymbals.wav from separated stems
3. Verify in DAW: visual inspection shows L/R panning

**Test Cases**:
1. **Mono baseline**: Run with `use_stereo: false`
   - Measure: total detections, crash/ride ratio
2. **Stereo processing**: Run with `use_stereo: true`
   - Measure: total detections, crash/ride ratio, pan distribution
3. **Manual verification**: Load MIDI in DAW
   - Verify: crashes align with left-panned hits, rides with right-panned

**Expected Results**:
- Stereo: Clear separation between crash (left) and ride (right)
- Mono: May miss distinction or classify based only on spectral features
- Improved accuracy: More consistent classification across song

---

## Dependencies
- No new external libraries required
- Uses existing: numpy, librosa, soundfile
- Builds on existing functional core architecture

---

## Timeline Estimate
- **Phase 1**: 30 min (config)
- **Phase 2**: 45 min (core functions)
- **Phase 3**: 30 min (audio loading)
- **Phase 4**: 60 min (detection pipeline)
- **Phase 5**: 45 min (cymbal classification)
- **Phase 6**: 60 min (testing)
- **Phase 7**: 30 min (documentation)

**Total**: ~5 hours

---

## Notes
- This plan follows functional core / imperative shell architecture
- Stereo analysis functions are pure (in `stereo_core.py`)
- Detection coordinators are in imperative shell (in `detection_shell.py`)
- TypedDicts in `midi_types.py` for data contracts
- Config changes backwards compatible with defaults
