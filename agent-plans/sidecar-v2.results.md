# Sidecar V2 - Results

**Plan**: [sidecar-v2.plan.md](sidecar-v2.plan.md)  
**Started**: 2026-01-20  
**Status**: ✅ COMPLETE

---

## Phase 1-5: Sidecar V2 Implementation
- [x] Changed process_stem_to_midi() return format to Dict with events, all_onset_data, spectral_config
- [x] Updated stems_to_midi_cli.py to collect analysis_by_stem
- [x] Completely rewrote save_analysis_sidecar() for v2 format
- [x] Added _round_value() helper function
- [x] Extract logic block from spectral_config per stem
- [x] Include ALL onsets (KEPT + FILTERED) from all_onset_data
- [x] Applied numeric rounding (times=4 decimals, features=2 decimals)
- [x] New structure: {version: "2.0", stems: {stem_name: {logic: {}, events: []}}}
- [x] Updated tests for dict return format
- [x] Fixed early return paths in processing_shell.py

## Phase 6: Status Field Fix
- [x] Added 'status' field to all_onset_data in analysis_core.py
- [x] Status set to 'KEPT' or 'FILTERED' based on filtering decisions
- [x] Updated decay filter to mark retriggered events as FILTERED
- [x] Updated statistical filter to mark rejected events as FILTERED
- [x] Verified on project 10 full track (3036 events, 1422 filtered)
- [x] All 186 tests passing

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-20 | Simplified design: logic once per stem (not per-event) | User preference for brevity, reduces JSON size |
| 2026-01-20 | Status field in all_onset_data, not nested decision objects | Cleaner format, easier to query |
| 2026-01-20 | Times=4 decimals, features=2 decimals | 0.1ms time precision sufficient, feature precision balanced with readability |
| 2026-01-20 | Changed process_stem_to_midi return to Dict | Breaking change but necessary for analysis data passthrough |
| 2026-01-20 | Added status field after filtering logic runs | Enables tracking of KEPT vs FILTERED events across all filter passes |

---

## Metrics

- Tests passing: 186/186 ✅
- File size: 806K for full track with 3036 events (reasonable)
- Precision: Times 4 decimals, features 2 decimals
- Coverage: All onsets included (KEPT + FILTERED)
