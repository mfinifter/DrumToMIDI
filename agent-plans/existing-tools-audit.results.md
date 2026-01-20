# Existing Tools Audit - Results

**Plan**: [existing-tools-audit.plan.md](existing-tools-audit.plan.md)  
**Started**: 2026-01-19  
**Status**: Complete

---

## Phase 1: Code Inventory

### Entry Points Found

| File | Purpose | Status |
|------|---------|--------|
| `stems_to_midi_cli.py` | CLI for audio→MIDI conversion | ✅ Working |
| Web UI endpoints | Flask API for conversion | ⚠️ Needs verification |

### Core Modules Found (stems_to_midi/ package)

| File | Purpose | Key Functions | Status |
|------|---------|---------------|--------|
| `config.py` | DrumMapping dataclass | DrumMapping | ✅ |
| `detection_shell.py` | Onset detection coordinator | `detect_onsets`, `detect_tom_pitch`, `detect_hihat_state` | ✅ |
| `analysis_core.py` | Pure spectral analysis (1370 lines!) | `filter_onsets_by_spectral`, `calculate_spectral_energies`, `analyze_onset_spectral` | ✅ |
| `processing_shell.py` | Main pipeline orchestrator | `process_stem_to_midi` | ✅ |
| `learning.py` | Learn thresholds from user edits | `learn_threshold_from_midi`, `save_calibrated_config` | ⚠️ Needs testing |
| `midi.py` | MIDI file I/O | `create_midi_file`, `read_midi_notes` | ✅ |

### Optimization Subpackage (stems_to_midi/optimization/)

| File | Purpose | Status |
|------|---------|--------|
| `extract_features.py` | Export ALL onsets with features to CSV | ⚠️ Exists, untested |
| `optimize.py` | Bayesian optimization from labeled CSV | ⚠️ Exists, untested |

### Configuration Found

| File | Purpose | Status |
|------|---------|--------|
| `midiconfig.yaml` | Global + per-stem thresholds | ✅ |
| `midiconfig_calibrated.yaml` | Example calibrated config | ✅ |
| Per-project `midiconfig.yaml` | Project-specific overrides | ✅ |

### Learning/Analysis Tools Found

| Feature | Location | How to Use | Status |
|---------|----------|------------|--------|
| Learning mode | `--learn` CLI flag | Ultra-sensitive detection, outputs all hits | ✅ Working |
| Threshold learning | `learning.py` | Compare original + edited MIDI → optimal threshold | ⚠️ Needs test |
| Feature extraction | `optimization/extract_features.py` | `python -m stems_to_midi.optimization.extract_features 4 --stem hihat` | ⚠️ Untested |
| Bayesian optimization | `optimization/optimize.py` | `python -m stems_to_midi.optimization.optimize 4 --stem hihat` | ⚠️ Untested |

---

## Phase 2: Data Flow

```
Audio File (wav/flac)
    ↓
_load_and_validate_audio() [processing_shell.py]
    ↓ np.ndarray, sr
detect_onsets() [detection_shell.py]
    ↓ onset_times: np.ndarray, onset_strengths: np.ndarray
filter_onsets_by_spectral() [analysis_core.py]
    ↓ filtered onset_times, stem_geomeans, sustain_durations
_create_midi_events() [processing_shell.py]
    ↓ List[Dict]
create_midi_file() [midi.py]
    ↓
MIDI File
```

### Intermediate Types Discovered

| Type/Structure | Where Defined | What It Contains |
|----------------|---------------|------------------|
| `np.ndarray` | numpy | onset_times (seconds), onset_strengths |
| `List[Dict]` | processing_shell.py | `{time, note, velocity, duration}` ← **Current contract** |
| `DrumMapping` | config.py | MIDI note numbers per instrument |
| `MidiNote`, `DrumNote` | midi_types.py | Rendering types (not used in detection) |

### Key Insight: Two Separate Type Systems

1. **Detection output**: `List[Dict]` with `{time, note, velocity, duration}`
2. **Rendering types**: `MidiNote`, `DrumNote`, `MidiSequence` in `midi_types.py`

These are NOT connected. The Detection Output Contract should bridge them.

---

## Phase 3: Status Assessment

| Feature | Status | Evidence | Notes |
|---------|--------|----------|-------|
| Basic onset detection | ✅ Working | Tests pass, CLI works | Uses librosa |
| Spectral filtering (snare) | ✅ Working | geomean_threshold in config | Body + wire frequencies |
| Spectral filtering (kick) | ⚠️ Partial | Config exists, needs testing | Fundamental + body + attack |
| Hi-hat open/closed | ✅ Working | detect_hihat_state | Sustain-based classification |
| Tom pitch detection | ✅ Working | detect_tom_pitch | YIN/pYIN algorithms |
| Learning mode | ⚠️ Unclear | Code exists, no recent tests | May have bugs per TODO |
| Bayesian optimization | ⚠️ Unknown | Code exists in optimization/ | Never verified |
| Feature extraction → CSV | ⚠️ Unknown | Code exists | Untested |

### What's Actually Tested (from test files)

| Test File | What It Tests |
|-----------|---------------|
| `test_analysis_core.py` | Pure spectral functions |
| `test_detection_shell.py` | Onset detection |
| `test_learning.py` | Learning threshold logic |
| `test_stems_to_midi.py` | Integration tests |

---

## Phase 4: Recommendations

### For Detection Output Contract

1. **Current output format**: `List[Dict]` with `{time, note, velocity, duration}`
   - Simple but loses all spectral data
   - No confidence score
   - No algorithm attribution

2. **Spectral data available but discarded**:
   - `stem_geomeans` calculated but not exported
   - `sustain_durations` calculated but not exported  
   - `onset_strengths` calculated but not exported
   - All this data COULD go into DetectedHit

3. **Integration point**: Replace `_create_midi_events()` return type
   - Currently: `List[Dict]`
   - New: `List[DetectedHit]` or `DetectionResult`
   - Add `.to_midi_events()` method for backwards compatibility

4. **Learning mode already does feature extraction**:
   - `learning.py:analyze_onset_spectral()` extracts all features
   - Could reuse this for DetectedHit population

### For Workflow Improvements

1. **Optimization tools exist but are disconnected**:
   - `extract_features.py` → CSV
   - `optimize.py` → reads CSV, does Bayesian optimization
   - These could be integrated into main workflow

2. **Learning mode workflow**:
   - `--learn` flag works
   - `learn_threshold_from_midi()` compares original vs edited
   - Missing: easy way to invoke from CLI/WebUI

### Priority Order

1. **Test existing tools first** - verify learning mode and optimization actually work
2. **Add spectral data to output** - DetectedHit with all the features
3. **Integrate optimization tools** - make them first-class CLI commands
4. **Add ensemble detection** - already using librosa, just add Superflux

---

## Phase 5: Empirical Per-Stem Output Capture

Ran actual detection on projects to capture real terminal output formats.

### Snare Columns
```
Time    Str    Amp    BodyE    WireE    Total  GeoMean     Status
```
- BodyE = 150-400 Hz (drum body)
- WireE = 2-8 kHz (snare wires)
- GeoMean = sqrt(BodyE * WireE)
- Default threshold: 40.0

### Kick Columns
```
Time    Str    Amp    FundE    BodyE  AttackE    Total  GeoMean     Status
```
- FundE = 40-80 Hz (fundamental thump)
- BodyE = 80-150 Hz (body resonance)  
- AttackE = 150-400 Hz (beater attack)
- GeoMean = cbrt(FundE * BodyE * AttackE)
- Default threshold: 70.0

### Hi-Hat Columns
```
Time    Str    Amp    BodyE  SizzleE    Total  GeoMean  SustainMs     Status
```
- BodyE = 500-2000 Hz (metal body)
- SizzleE = 6-12 kHz (high sizzle)
- SustainMs = decay duration for open/closed detection
- GeoMean = sqrt(BodyE * SizzleE)
- Default threshold: 20.0
- Open/Closed threshold: 100ms

### Toms Columns
```
Time    Str    Amp    FundE    BodyE    Total  GeoMean     Status
```
Plus pitch detection phase:
```
Time  Pitch(Hz)      Tom
```
- FundE = 60-150 Hz
- BodyE = 150-400 Hz
- GeoMean = sqrt(FundE * BodyE)
- Default threshold: 80.0
- Tom classification: Low/Mid/High

### Cymbals Columns
Same format as hi-hat (BodyE 1-4kHz, BrillE 4-10kHz, SustainMs)

### Contract Documentation Created

See [docs/DETECTION_OUTPUT_CONTRACT.md](../docs/DETECTION_OUTPUT_CONTRACT.md) for:
- Complete field definitions per stem type
- Extended output format proposal
- CLI commands for testing
- Migration path

---

## Decision Log

| Date | Finding | Implication |
|------|---------|-------------|
| 2026-01-19 | detection output is `List[Dict]` | Contract should wrap this or replace it |
| 2026-01-19 | Spectral data calculated but discarded | DetectedHit should capture it |
| 2026-01-19 | Two type systems (detection vs rendering) | Contract bridges them |
| 2026-01-19 | Bayesian optimization exists | May not need to rewrite, just test and integrate |
| 2026-01-19 | analysis_core.py is 1370 lines | Rich feature extraction already available |
| 2026-01-19 | Per-stem columns empirically captured | Contract now documents real output format |
| 2026-01-19 | GeoMean formulas vary by stem | sqrt for 2-band, cbrt for 3-band instruments |
