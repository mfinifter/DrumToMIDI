# Sidecar V2 Plan

**Created**: 2026-01-20  
**Status**: Planning  
**Goal**: Enhance analysis sidecar with decision metadata, filtered events, and cleaner output

---

## Problem Statement

Current sidecar output:
1. Only includes KEPT events - no visibility into FILTERED onsets
2. No decision metadata - can't see WHY an onset was kept/filtered
3. Excessive numeric precision (15+ decimals) - bloats file size
4. Flat structure resists extension with new tools

## Success Criteria

1. All onsets included (KEPT, FILTERED, LEARNING)
2. Each onset has `decision` block with rule/threshold/margin
3. Numbers rounded to 2-4 decimal places
4. Extensible `features` dict for future tools
5. Existing consumers still work (backward compatible access patterns)
6. All 602 tests pass

---

## Phases

### Phase 1: Restructure Sidecar Format
- [ ] Change stems from `List[event]` to `{logic: {}, events: []}`
- [ ] Add `logic` block with thresholds and pass list per stem
- [ ] Keep event fields flat (no nested features dict - brevity)
- [ ] Update `save_analysis_sidecar()` in midi.py

### Phase 2: Include All Onsets  
- [ ] Modify `filter_onsets_by_spectral()` to return ALL onset data
- [ ] Add status field: KEPT, FILTERED, LEARNING
- [ ] Track which pass filtered each onset (for status)
- [ ] Update `processing_shell.py` to collect all onsets

### Phase 3: Add Logic Block
- [ ] Extract logic/thresholds from spectral_config per stem
- [ ] Include: geomean_threshold, min_sustain_ms, decay settings, etc.
- [ ] List active passes: ["geomean", "sustain", "decay", "statistical"]
- [ ] Consumer calculates margin from data + logic

### Phase 4: Numeric Precision
- [ ] Add rounding helper function
- [ ] Round features to 2 decimals (energy values)
- [ ] Round times to 4 decimals (ms precision)
- [ ] Round margins to 2 decimals
- [ ] Update `save_analysis_sidecar()` to apply rounding

### Phase 5: Update Consumers
- [ ] Update `detect_hihat_state()` to use features dict
- [ ] Update `learning.py` to use features dict
- [ ] Update any tests that check specific field locations

---

## Output Format (Target)

Logic defined ONCE per stem. Events contain only data. Consumer calculates margin.

```json
{
  "version": "2.0",
  "tempo_bpm": 120,
  "stems": {
    "snare": {
      "logic": {
        "geomean_threshold": 100.0,
        "min_sustain_ms": 50.0,
        "passes": ["geomean"]
      },
      "events": [
        {"time": 1.50, "note": 38, "velocity": 100, "status": "KEPT",
         "geomean": 316.23, "primary_energy": 500.00, "secondary_energy": 200.00},
        {"time": 2.34, "status": "FILTERED",
         "geomean": 45.00, "primary_energy": 80.00, "secondary_energy": 25.00}
      ]
    },
    "cymbals": {
      "logic": {
        "geomean_threshold": 80.0,
        "min_sustain_ms": 100.0,
        "decay_filter_enabled": true,
        "decay_window_sec": 0.5,
        "passes": ["geomean", "sustain", "decay"]
      },
      "events": [...]
    },
    "kick": {
      "logic": {
        "geomean_threshold": 50.0,
        "statistical_outlier_enabled": true,
        "passes": ["geomean", "statistical"]
      },
      "events": [...]
    }
  }
}
```

Consumer calculates margin: `margin = event.geomean - stem.logic.geomean_threshold`

---

## Risks

1. **Breaking existing consumers** - Mitigate with backward-compatible access
2. **File size increase** - Including filtered events adds data, but rounding helps
3. **Performance** - More tracking in hot path, but minimal impact expected

---

## Estimated Effort

- Phase 1: 20 min (sidecar format)
- Phase 2: 30 min (all onsets tracking)
- Phase 3: 15 min (logic block extraction)
- Phase 4: 10 min (rounding)
- Phase 5: 20 min (consumer updates)

Total: ~1.5 hours (simplified from 2.5h - no per-event decision objects)
