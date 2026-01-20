# Sidecar V2 - Results

**Plan**: [sidecar-v2.plan.md](sidecar-v2.plan.md)  
**Started**: 2026-01-20  
**Status**: Not started

---

## Phase 1: Restructure Contract
- [ ] Update `SpectralOnsetData` TypedDict
- [ ] Add `features` and `decision` dicts
- [ ] Update SPECTRAL_REQUIRED_FIELDS

## Phase 2: Include All Onsets
- [ ] Track filtered onsets in filter_onsets_by_spectral()
- [ ] Return both kept and filtered lists
- [ ] Update processing_shell.py

## Phase 3: Add Decision Metadata  
- [ ] Track filter pass (geomean/decay/statistical)
- [ ] Record threshold and margin
- [ ] Per-stem logic

## Phase 4: Numeric Precision
- [ ] Add rounding helper
- [ ] Apply to sidecar output

## Phase 5: Update Consumers
- [ ] detect_hihat_state()
- [ ] learning.py
- [ ] Tests

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| | | |

---

## Metrics

- Tests passing: N/A
- File size reduction: N/A
