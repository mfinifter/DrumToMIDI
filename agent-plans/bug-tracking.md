## Bugs Migrated to GitHub Issues

Bugs are now tracked in GitHub Issues: https://github.com/EverlastEngineering/DrumToMIDI/issues

- [#2](https://github.com/EverlastEngineering/DrumToMIDI/issues/2) - Missing instrument labels (Low)
- [#3](https://github.com/EverlastEngineering/DrumToMIDI/issues/3) - Text lane legend overlap (Medium)
- [#4](https://github.com/EverlastEngineering/DrumToMIDI/issues/4) - Add ModernGL renderer test (follow-up)
- [#5](https://github.com/EverlastEngineering/DrumToMIDI/issues/5) - Stale references to stems_to_midi.py

---

## Open Bugs (Not Yet in GitHub)

### Missing pan_confidence data in analysis JSON output
- **Status**: Open
- **Priority**: Medium
- **Description**: Pan position data is calculated during energy-based detection but not saved to the analysis.json sidecar file
- **Details**: 
  - `energy_detection_core.py` calculates `pan_confidence` for each onset (R-L)/(R+L) ranging from -1.0 (left) to +1.0 (right)
  - `analysis_core.py::extract_onset_features()` includes pan_confidence in feature extraction
  - `midi.py::save_analysis_sidecar()` does NOT include pan_confidence in saved JSON fields
- **Expected Behavior**: Pan position data should be available in analysis JSON for visualization and analysis
- **Actual Behavior**: Pan data is calculated but discarded during JSON serialization
- **Impact**: Cannot visualize or analyze stereo positioning of detected hits
- **Suggested Fix**: Add 'pan_confidence' to the field list in `save_analysis_sidecar()` (line ~240-245)

### Missing pitch data in analysis JSON output  
- **Status**: Open
- **Priority**: Medium
- **Description**: Pitch detection work was implemented but pitch_hz is not populated in analysis JSON output
- **Details**:
  - `detection_shell.py` has `detect_tom_pitch()` and `detect_cymbal_pitch()` functions using librosa pYIN
  - Config has 'pitch' listed as a secondary feature for clustering
  - `midi.py::save_analysis_sidecar()` includes 'pitch_hz' in the field list (line 242)
  - However, pitch detection functions are not called during normal processing
- **Expected Behavior**: Pitch should be detected for toms and cymbals and saved to JSON
- **Actual Behavior**: pitch_hz field exists but remains null/undefined in JSON output
- **Impact**: Cannot distinguish tom pitches (high/mid/low) or analyze cymbal characteristics
- **Investigation Needed**: 
  - Where should pitch detection be called in the processing pipeline?
  - Should it be per-onset or per-stem?
  - Performance impact of adding pitch detection?

### MIDI file creation error: "pop from empty list" in midiutil
- **Status**: Fixed
- **Priority**: High
- **Description**: IndexError in midiutil.MidiFile.deInterleaveNotes() when writing MIDI files with energy-based detection
- **Root Cause**: 
  1. Energy detection creating duplicate onset times (3x duplicates at 197.242s in cymbals)
  2. Zero-duration MIDI notes when two onsets occur at nearly identical times
  3. midiutil's deInterleaveNotes() failing to match note_on/note_off pairs with duplicates
- **Steps to Reproduce**: 
  1. Run stems-to-midi on project 14 (Thunderstruck)
  2. Energy detection produces duplicate onset times within 1ms
  3. MIDI creation calculates duration = next_onset - current_onset = 0.0
  4. midiutil.writeFile() crashes with "IndexError: pop from empty list"
- **Expected Behavior**: Each detected onset creates one MIDI note with valid duration
- **Actual Behavior**: Duplicate onsets create multiple notes at same time with 0 duration, causing MIDI library error
- **Fixed**: 2026-01-27
- **Solution**: Two-part fix:
  1. **Duplicate removal** in `energy_detection_shell.py`:
     - Round onset times to nearest millisecond
     - Remove duplicates within 1ms threshold
     - Prevents duplicate detections from reaching MIDI creation
  2. **Minimum duration enforcement** in `analysis_core.prepare_midi_events_for_writing()`:
     - Set MIN_DURATION_BEATS = 0.01 (5ms at 120 BPM)
     - Ensures all MIDI notes have valid duration
     - Prevents midiutil deInterleaveNotes errors
- **Impact**: Energy-based detection now produces valid MIDI files without errors. Removed 6 duplicate events from project 14 (snare: 1, toms: 5)
- **Files Modified**: 
  - `stems_to_midi/energy_detection_shell.py` (deduplication)
  - `stems_to_midi/analysis_core.py` (minimum duration)
- **Fixed in Commit**: (pending commit)

---

### CPU underutilization during MDX23C stem separation on macOS
- **Status**: Fixed
- **Priority**: High
- **Description**: When using MDX23C model with CPU inference on macOS, only ~55% CPU utilization observed with work distributed to efficiency cores instead of performance cores
- **Root Cause**: PyTorch default threading configuration limited to 4 threads (likely defaulting to physical core count), while system has 8 cores (4 performance + 4 efficiency)
- **Expected Behavior**: Full CPU utilization across all cores, prioritizing performance cores
- **Actual Behavior**: Only 4 threads used, resulting in ~55% CPU usage with efficiency cores engaged
- **Steps to Reproduce**: 
  1. Run stem separation with MDX23C model on Mac with device=cpu
  2. Observe CPU usage in Activity Monitor
  3. Notice low utilization (~55%) and efficiency core usage
- **Fixed**: 2026-01-25
- **Solution**: Added `_configure_cpu_threading()` method to `OptimizedMDX23CProcessor.__init__()`:
  - Detects total CPU core count using `multiprocessing.cpu_count()`
  - On macOS, detects performance vs efficiency core split using `sysctl`
  - Configures PyTorch: `torch.set_num_threads()` to use all cores
  - Sets `OMP_NUM_THREADS` and `MKL_NUM_THREADS` environment variables
  - Now uses 8 threads (100% utilization) instead of 4 threads (55% utilization)
- **Impact**: Should significantly improve CPU-based stem separation performance on macOS
- **Files Modified**: `mdx23c_optimized.py`

---

## Open Bugs (Not Yet in GitHub - Original)

### Cymbals appeared missing due to --maxtime truncation
- **Status**: Closed (User Error / Testing Bug)
- **Priority**: N/A
- **Description**: Cymbals appeared silent when testing with `--maxtime 60`
- **Root Cause**: Thunderstruck cymbals don't start until ~90 seconds (intro is all hi-hat)
  - 0-90s: max amplitude 0.000488 (effectively silent)
  - 90s+: max amplitude 0.31-0.56 (actual cymbal content)
- **Resolution**: Running full conversion (no maxtime) detects 77 cymbal events
- **Lesson**: When troubleshooting, run full conversion first, then use maxtime for faster iteration only after confirming content exists
- **Note 27**: Still just a technical anchor note for DAW alignment (see `stems_to_midi/midi.py:64`)

---

## Fixed Bugs (Historical)

### Broken import after file rename - render_midi_video_shell.py
- **Fixed**: 2026-01-18
- **Root Cause**: Missing test coverage for `render_project_video()` with `use_moderngl=True`
