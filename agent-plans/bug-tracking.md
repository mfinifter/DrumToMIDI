## Bugs Migrated to GitHub Issues

Bugs are now tracked in GitHub Issues: https://github.com/EverlastEngineering/DrumToMIDI/issues

- [#2](https://github.com/EverlastEngineering/DrumToMIDI/issues/2) - Missing instrument labels (Low)
- [#3](https://github.com/EverlastEngineering/DrumToMIDI/issues/3) - Text lane legend overlap (Medium)
- [#4](https://github.com/EverlastEngineering/DrumToMIDI/issues/4) - Add ModernGL renderer test (follow-up)
- [#5](https://github.com/EverlastEngineering/DrumToMIDI/issues/5) - Stale references to stems_to_midi.py

---

## Open Bugs (Not Yet in GitHub)

### Cymbals stem is empty for Thunderstruck
- **Status**: Closed (Not a Bug)
- **Priority**: N/A
- **Description**: Cymbals stem shows as silent (max amplitude 0.000443) for AC_DC_Thunderstruck_Drums track
- **Root Cause**: The source audio separation (MDX23C) didn't extract cymbal content for this track
  - Both `cleaned/` and `stems/` folders have empty cymbals
  - This is separation model behavior, not a detection bug
- **Note 27 Mystery**: MIDI files contain note 27 which is NOT cymbals - it's a technical anchor note added at time 0 for DAW alignment (see `stems_to_midi/midi.py:64`)
- **Resolution**: Track genuinely has no separable cymbal content

---

## Fixed Bugs (Historical)

### Broken import after file rename - render_midi_video_shell.py
- **Fixed**: 2026-01-18
- **Root Cause**: Missing test coverage for `render_project_video()` with `use_moderngl=True`
