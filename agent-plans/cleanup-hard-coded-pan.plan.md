# Cleanup Plan: Remove Hard-Coded Pan Thresholds

**Created**: 2026-02-03  
**Status**: Planning  
**Context**: Phase 4 of clustering-based threshold optimization complete. Before proceeding to Phase 5-6, need to remove old hard-coded pan classification approach.

## Background

The old stereo processing implementation (completed Feb 2) added pan-aware cymbal classification using **hard-coded pan thresholds**:

```python
# stems_to_midi/analysis_core.py:1639-1640
LEFT_THRESHOLD = -0.25   # More negative = left
RIGHT_THRESHOLD = 0.25   # More positive = right

# If pan < -0.25 → Crash
# If pan > 0.25 → Ride
# Else → Use spectral/pitch as fallback
```

This approach was a stepping stone but has fundamental limitations:
- **Fixed thresholds** don't adapt to different recording styles
- **Produces 107 detections** on Thunderstruck when only 57 exist (too many false positives from hihat bleed)
- **Pan alone insufficient** to distinguish actual cymbals from hihat/tom bleed

## New Approach (Clustering-Based)

The new clustering infrastructure (Phases 1-4 complete) will replace this with:

1. **Dual-channel onset detection**: Detect L/R onsets separately, merge within window
2. **Feature extraction**: Pan + spectral + pitch + timing for each onset
3. **Clustering**: Group similar onsets using DBSCAN or k-means
4. **Threshold optimization** (Phase 5): Iteratively adjust thresholds until cluster count matches expected
5. **Pipeline integration** (Phase 6): Use cluster assignments instead of hard-coded rules

## Files to Clean Up

### 1. `stems_to_midi/analysis_core.py`

**Function**: `classify_cymbal_by_pan()`  
**Lines**: 1599-1676  
**Action**: Mark as deprecated, add comment pointing to new clustering approach

```python
# DEPRECATED: This function uses hard-coded pan thresholds which don't
# adapt to different recording characteristics. Will be replaced by
# clustering-based classification in Phase 6 of threshold optimization.
# See: agent-plans/clustering-threshold-optimization.plan.md
def classify_cymbal_by_pan(...):
    ...
```

**Reason**: Don't delete yet - still used by processing_shell.py. Will be removed after Phase 6 integration.

### 2. `stems_to_midi/processing_shell.py`

**Function**: `_detect_cymbal_pitches()`  
**Lines**: 300-316 (pan-aware classification block)  
**Action**: Mark with TODO comment

```python
# TODO: Replace with clustering-based classification (Phase 6)
# Current implementation uses hard-coded pan thresholds (-0.25/+0.25)
# which don't adapt to recording characteristics.
if pan_positions is not None and len(pan_positions) == len(onset_times):
    from .analysis_core import classify_cymbal_by_pan
    ...
```

**Reason**: This code path will be entirely replaced by Phase 6 integration (optimize_thresholds_by_clustering → cluster assignments).

### 3. Documentation Updates

**File**: `agent-plans/stereo-processing.results.md`  
**Action**: Add note that Phase 5 (cymbal classification) is superseded by clustering approach

**File**: `agent-plans/clustering-threshold-optimization.results.md`  
**Action**: Add cleanup checklist item in Phase 6

## Cleanup Checklist (Phase 6)

When implementing Phase 6 (Pipeline Integration):

- [ ] Remove `classify_cymbal_by_pan()` function from analysis_core.py
- [ ] Remove pan-aware classification block from `_detect_cymbal_pitches()`
- [ ] Replace with clustering-based approach:
  ```python
  # New approach (pseudo-code):
  if threshold_optimization_enabled:
      features = extract_onset_features(audio, sr, onset_times, pan_confidence)
      cluster_result = cluster_onsets(features, method, n_clusters)
      cymbal_classifications = cluster_result['labels']  # Direct cluster assignments
  else:
      # Fallback to pitch-only (existing mono behavior)
      cymbal_classifications = classify_cymbal_pitch(detected_pitches)
  ```
- [ ] Update tests to reflect clustering-based classification
- [ ] Remove hard-coded threshold constants (LEFT_THRESHOLD, RIGHT_THRESHOLD)
- [ ] Update architecture docs to reflect new approach

## Why Not Clean Up Now?

**Decision**: Keep deprecated code until Phase 6 implementation complete

**Rationale**:
1. **Working baseline**: Current stereo processing (even with hard-coded thresholds) is better than mono-only
2. **Regression prevention**: Keep existing behavior until new clustering approach is fully validated
3. **Incremental migration**: Phase 5 will implement the optimization loop, Phase 6 will swap out the old code
4. **Testing**: Can A/B test old vs new approaches during Phase 6 validation

## Timeline

- **Now (Phase 4 complete)**: Document deprecation, add TODO comments
- **Phase 5**: Implement threshold optimization loop (uses clustering but doesn't touch processing_shell yet)
- **Phase 6**: Replace hard-coded classification with clustering-based approach, remove deprecated code
- **Phase 7**: Update documentation, remove all TODO comments

## Success Criteria

After cleanup:
- [ ] No hard-coded pan threshold constants in codebase
- [ ] `classify_cymbal_by_pan()` function removed
- [ ] Cymbal classification uses cluster assignments from optimization loop
- [ ] All tests passing with new approach
- [ ] Thunderstruck test case shows <60 detections (vs 107 with old approach)
- [ ] Architecture docs updated to reflect clustering-based approach

## Related Files

- `agent-plans/clustering-threshold-optimization.plan.md` - Master plan for new approach
- `agent-plans/stereo-processing.plan.md` - Old stereo processing approach (Phase 5 superseded)
- `agent-plans/stereo-processing.results.md` - Implementation results (note Phase 5 status)
