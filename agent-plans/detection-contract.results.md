# Detection Output Contract - Results

**Plan**: [detection-contract.plan.md](detection-contract.plan.md)  
**Started**: 2026-01-19  
**Status**: Phase 1 Complete, Phase 2-3 Partial

---

## Phase 1: Define Types
- [x] Add types to `midi_types.py` - SpectralOnsetData TypedDict (7b40fb9)
- [x] Document type contracts - docs/DETECTION_OUTPUT_CONTRACT.md (2eaf87a)
- [ ] Add unit tests for type serialization

## Phase 2: Refactor Current Detection
- [x] Audit current detection code - documented in DETECTION_OUTPUT_CONTRACT.md
- [x] Preserve spectral data through filter_onsets_by_spectral (e6f609f)
- [x] Wire spectral data to MIDI events in processing_shell.py (e6f609f)
- [ ] Create `LibrosaDetector` implementing `DrumDetector` protocol (future)
- [x] Verify no regressions - 602 tests passing

## Phase 3: Update Consumers
- [x] Identify consumers: detect_hihat_state(), learning.py, processing_shell.py
- [x] Add typed imports to consumers (7b40fb9)
- [x] Update MIDI export to carry spectral fields (e6f609f)
- [x] Add JSON sidecar output for analysis tools (c448daf)

## Phase 4: Enable Extensions
- [ ] Add `SuperfluxDetector`
- [ ] Add ensemble voting function
- [ ] Add stereo analysis function
- [ ] Add pattern recognition module

---

## Audit Log

### Current Detection Code (Producers)

| File | Function | Output | Notes |
|------|----------|--------|-------|
| analysis_core.py | filter_onsets_by_spectral() | filtered_onset_data list | Primary producer of SpectralOnsetData |
| analysis_core.py | analyze_onset_spectral() | dict with energy bands | Per-onset spectral analysis |

### Consumers of Detection Output

| File | Function | Input | Notes |
|------|----------|-------|-------|
| detection_shell.py | detect_hihat_state() | spectral_data: List[SpectralOnsetData] | Uses primary_energy, secondary_energy |
| learning.py | learn_threshold_from_midi() | analysis dicts | Uses all spectral fields |
| processing_shell.py | _create_midi_events() | spectral_data param | Enriches MIDI events |
| midi.py | save_analysis_sidecar() | events_by_stem with spectral | Exports to JSON |

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-19 | Use TypedDict over dataclass | Preserves dict compatibility with existing code |
| 2026-01-19 | Add sidecar JSON output | Enables analysis without re-running detection |
| 2026-01-19 | Generic field names (primary/secondary_energy) | Stem-agnostic, meaning documented in contract |

---

## Commits

| Hash | Description |
|------|-------------|
| 2eaf87a | docs: add Detection Output Contract with per-stem spectral schemas |
| 18484c2 | docs: add learning mode and troubleshooting workflows |
| fb0ba56 | docs: clarify note 27 anchor and cymbals investigation |
| 9947aa1 | docs: fix --maxtime truncation pitfall documentation |
| e6f609f | feat(analysis): implement Detection Output Contract - spectral data preservation |
| 7b40fb9 | feat(types): add SpectralOnsetData TypedDict for bidirectional contract |
| c448daf | feat(midi): add JSON sidecar output for spectral analysis data |

---

## Metrics

- Tests passing: 602
- Detection regressions: 0
- Files modified: midi_types.py, analysis_core.py, detection_shell.py, learning.py, processing_shell.py, midi.py, stems_to_midi_cli.py
