## Bugs Migrated to GitHub Issues

Bugs are now tracked in GitHub Issues: https://github.com/EverlastEngineering/DrumToMIDI/issues

- [#2](https://github.com/EverlastEngineering/DrumToMIDI/issues/2) - Missing instrument labels (Low)
- [#3](https://github.com/EverlastEngineering/DrumToMIDI/issues/3) - Text lane legend overlap (Medium)
- [#4](https://github.com/EverlastEngineering/DrumToMIDI/issues/4) - Add ModernGL renderer test (follow-up)
- [#5](https://github.com/EverlastEngineering/DrumToMIDI/issues/5) - Stale references to stems_to_midi.py

---

## Open Bugs (Not Yet in GitHub)

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
