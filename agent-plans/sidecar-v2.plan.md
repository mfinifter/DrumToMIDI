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

### Phase 1: Restructure Contract (features dict)
- [ ] Update `SpectralOnsetData` TypedDict in midi_types.py
- [ ] Add `features: Dict[str, Any]` for extensible tool output
- [ ] Add `decision: Dict` for rule metadata
- [ ] Keep core fields at top level: time, note, velocity, status
- [ ] Update SPECTRAL_REQUIRED_FIELDS

### Phase 2: Include All Onsets
- [ ] Modify `filter_onsets_by_spectral()` to track filtered onsets
- [ ] Add `filtered_reason` field for onsets that didn't pass
- [ ] Return both kept and filtered in separate lists
- [ ] Update `processing_shell.py` to pass all onsets through

### Phase 3: Add Decision Metadata
- [ ] Track which pass filtered each onset (geomean, decay, statistical)
- [ ] Record threshold value applied
- [ ] Calculate margin (actual - threshold)
- [ ] Per-stem logic tracking (different stems have different rules)

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

```json
{
  "version": "2.0",
  "contract": "SpectralOnsetData",
  "precision": {"time": 4, "features": 2},
  "stems": {
    "snare": [
      {
        "time": 1.5000,
        "note": 38,
        "velocity": 100,
        "status": "KEPT",
        "features": {
          "geomean": 316.23,
          "primary_energy": 500.00,
          "secondary_energy": 200.00,
          "sustain_ms": 150.00,
          "onset_strength": 0.90
        },
        "decision": {
          "pass": "geomean",
          "rule": "geomean >= threshold",
          "threshold": 100.00,
          "value": 316.23,
          "margin": 216.23
        }
      },
      {
        "time": 2.3400,
        "status": "FILTERED",
        "features": {
          "geomean": 45.00,
          "primary_energy": 80.00,
          "secondary_energy": 25.00
        },
        "decision": {
          "pass": "geomean",
          "rule": "geomean >= threshold",
          "threshold": 100.00,
          "value": 45.00,
          "margin": -55.00
        }
      }
    ]
  }
}
```

---

## Risks

1. **Breaking existing consumers** - Mitigate with backward-compatible access
2. **File size increase** - Including filtered events adds data, but rounding helps
3. **Performance** - More tracking in hot path, but minimal impact expected

---

## Estimated Effort

- Phase 1: 30 min (contract changes)
- Phase 2: 45 min (filter tracking)
- Phase 3: 30 min (decision metadata)
- Phase 4: 15 min (rounding)
- Phase 5: 30 min (consumer updates)

Total: ~2.5 hours
