## Bugs Migrated to GitHub Issues

Bugs are now tracked in GitHub Issues: https://github.com/EverlastEngineering/DrumToMIDI/issues

- [#2](https://github.com/EverlastEngineering/DrumToMIDI/issues/2) - Missing instrument labels (Low)
- [#3](https://github.com/EverlastEngineering/DrumToMIDI/issues/3) - Text lane legend overlap (Medium)
- [#4](https://github.com/EverlastEngineering/DrumToMIDI/issues/4) - Add ModernGL renderer test (follow-up)
- [#5](https://github.com/EverlastEngineering/DrumToMIDI/issues/5) - Stale references to stems_to_midi.py

---

## Open Bugs (Not Yet in GitHub)

### Cleaned vs Raw stems have different audio content
- **Status**: Open (needs investigation)
- **Priority**: Low
- **Description**: Project 10 uses "cleaned" folder where cymbals stem is silent, but project 1 uses raw "stems" folder where cymbals has content
- **Root Cause Analysis**:
  - CLI prefers `cleaned/` over `stems/` folders
  - Sidechain cleanup may redistribute audio between stems
  - Cymbals content may end up in hihat stem after cleaning
  - No cross-stem logic in detection - stems process independently
- **Not a Bug**: This is expected behavior - cleaned stems have different content
- **Action Items**:
  - [ ] Document which stem folder to use for troubleshooting
  - [ ] Consider CLI flag to force use of raw stems: `--use-raw-stems`
  - [ ] Verify WebUI and CLI use same stem source for given project

---

## Fixed Bugs (Historical)

### Broken import after file rename - render_midi_video_shell.py
- **Fixed**: 2026-01-18
- **Root Cause**: Missing test coverage for `render_project_video()` with `use_moderngl=True`
