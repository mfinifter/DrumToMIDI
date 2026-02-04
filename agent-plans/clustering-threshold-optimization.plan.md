# Clustering-Based Threshold Optimization Plan

**Created**: 2026-02-03  
**Objective**: Replace hard-coded pan-based cymbal classification with adaptive threshold discovery using cluster count optimization.

## Problem Statement

Current implementation:
- Detects 107 onsets when only 57 actual crashes exist
- Hard-coded pan thresholds (`pan > 0.25 → ride`) cannot distinguish crash from hihat bleed
- Pan position alone insufficient for instrument classification

Root cause: Threshold values are guessed, not discovered from audio characteristics.

## Core Insight

**Threshold → Onset Count → Feature Vectors → Cluster Count**

By iteratively adjusting detection thresholds until cluster count matches expected instrument count, we automatically discover optimal thresholds for each track's characteristics.

## Architecture Overview

### Dual-Channel Detection
Run onset detection separately on L/R channels:
- Left-panned instruments trigger stronger left-channel onsets
- Right-panned instruments trigger stronger right-channel onsets
- Center-panned instruments trigger both equally
- Merge nearby L/R detections (within configurable window) into unified onset list

### Feature Vector Construction
For each detected onset, extract:
- **Pan confidence**: `(R-L)/(R+L)` where R/L are onset strengths
- **Spectral features**: centroid, rolloff, flatness, etc.
- **Pitch**: Detected fundamental frequency
- **Timing**: Delta from previous onset

### Clustering Methods (Pluggable)
- **DBSCAN**: Density-based, finds arbitrary shapes, no preset cluster count
- **k-means**: Requires cluster count, finds centroid-based groups
- Both implemented as pure functions accepting feature matrix

### Threshold Optimization Loop
```
while cluster_count != expected_clusters:
    onsets = detect_with_thresholds(audio, current_thresholds)
    features = extract_features(onsets)
    clusters = cluster(features)
    
    if len(clusters) > expected:
        increase_thresholds  # fewer onsets
    else:
        decrease_thresholds  # more onsets
```

## Implementation Phases

### Phase 1: Configuration Schema
**Deliverables**:
- Add `onset_merge_window_ms` to stem configs (default: 100ms)
- Add `expected_clusters` to stem configs (cymbals: 2)
- Add `clustering.method` ('dbscan' or 'kmeans')
- Add `clustering.features` list (which features to include)
- Add `threshold_optimization.enabled` boolean
- Add `threshold_optimization.max_iterations` (default: 20)
- Add `threshold_optimization.tolerance` (stop when ±N clusters)

**Tests**: Schema validation, config loading

### Phase 2: Dual-Channel Onset Detection
**Deliverables**:
- `detect_dual_channel_onsets()` in `stereo_core.py`
  - Separate L/R channel analysis
  - Independent onset detection per channel
  - Merge nearby detections within window
  - Return onset times + L/R strengths
- TypedDict: `DualChannelOnsetData`

**Tests**:
- L-only signal produces L-strong onsets
- R-only signal produces R-strong onsets
- Center signal produces equal L/R strengths
- Merging logic (nearby onsets combined)

### Phase 3: Feature Extraction
**Deliverables**:
- `extract_onset_features()` in `analysis_core.py`
  - Pan confidence: `(R-L)/(R+L)`
  - Spectral features (reuse existing functions)
  - Pitch (reuse existing functions)
  - Timing delta (time since previous onset)
- TypedDict: `OnsetFeatures`
- Return: List[OnsetFeatures] for all onsets

**Tests**:
- Pan confidence calculation accuracy
- Feature vector completeness
- Edge cases (zero denominator, first onset timing)

### Phase 4: Pluggable Clustering
**Deliverables**:
- `cluster_dbscan()` in `clustering_core.py` (NEW)
- `cluster_kmeans()` in `clustering_core.py` (NEW)
- `cluster_onsets()` dispatcher function
- TypedDict: `ClusterResult` (labels, n_clusters, centroids)

**Tests**:
- DBSCAN finds correct cluster count on synthetic data
- k-means with k=2 separates bimodal distribution
- Dispatcher selects correct method from config

**Dependencies**: `sklearn` (already in environment)

### Phase 5: Threshold Optimization Loop
**Deliverables**:
- `optimize_thresholds_by_clustering()` in `optimization_core.py` (NEW)
  - Input: audio, initial thresholds, expected clusters, config
  - Loop: adjust thresholds based on cluster count
  - Output: optimized thresholds, final cluster assignments
- Adjustment strategy: binary search or gradient-based
- Convergence: cluster_count within tolerance of expected

**Tests**:
- Converges to correct thresholds on known data
- Handles no-convergence gracefully (max iterations)
- Threshold adjustment direction correct

### Phase 6: Pipeline Integration
**Deliverables**:
- Update `processing_shell.py::process_stem_to_midi()`
  - Check if `threshold_optimization.enabled`
  - If yes: call optimization loop
  - If no: use configured thresholds (backward compatible)
- Update `_detect_cymbal_pitches()` to use cluster assignments
- Logging: report discovered thresholds, cluster stats

**Tests**: Integration test with Thunderstruck cymbals

### Phase 7: Documentation & Testing
**Deliverables**:
- Update `docs/ARCH_DATA_FLOW.md` with optimization loop
- Update `midiconfig.yaml` with inline comments
- Update `README.md` with clustering feature description
- Add configuration example for 2-cymbal vs 3-cymbal tracks
- Comprehensive integration test with real audio

**Tests**: End-to-end validation with multiple tracks

## Risks & Mitigations

### Risk: Convergence Failure
**Scenario**: Loop never finds thresholds that produce expected cluster count  
**Mitigation**: 
- Max iterations limit (default: 20)
- Fallback to initial thresholds with warning
- Log cluster counts at each iteration for debugging

### Risk: Computational Cost
**Scenario**: Optimization loop slow on long tracks  
**Mitigation**:
- Cache spectral/pitch calculations between iterations
- Only recalculate onset detection (cheap)
- Consider using track preview (first 30s) for optimization

### Risk: Overfitting to Expected Clusters
**Scenario**: Forcing wrong cluster count on ambiguous data  
**Mitigation**:
- Report cluster quality metrics (silhouette score)
- Warn if optimization produces low-quality clusters
- Allow `expected_clusters: null` to disable optimization

### Risk: Breaking Existing Functionality
**Scenario**: Changes break mono or non-clustering workflows  
**Mitigation**:
- Make optimization opt-in via config flag
- Maintain backward compatibility with existing thresholds
- All existing tests must pass

## Success Criteria

1. **Accuracy**: Thunderstruck test case correctly identifies 27L + 27R + 3C crashes
2. **Performance**: Optimization completes in <30s for typical 4-minute track
3. **Compatibility**: All existing mono tests pass unchanged
4. **Convergence**: ≥90% of real-world tracks converge within 20 iterations
5. **Code Quality**: Test coverage ≥85% on new clustering code
6. **Documentation**: User can configure expected clusters without code changes

## Dependencies

- **Existing Code**:
  - `stereo_core.py`: Extend with dual-channel detection
  - `analysis_core.py`: Add feature extraction functions
  - `processing_shell.py`: Integrate optimization loop
  - `midi_types.py`: Add new TypedDicts

- **Libraries**:
  - `sklearn` (DBSCAN, k-means) - already in environment
  - `librosa` (onset detection) - already in use
  - `numpy` (feature matrices) - already in use

- **Configuration**:
  - `midiconfig.yaml`: Add optimization settings
  - `webui/settings_schema.py`: Add UI controls

## Out of Scope (Future Work)

- Multi-feature weighting/normalization strategies
- Automatic expected_clusters discovery (unsupervised)
- Real-time optimization during playback
- GPU-accelerated clustering for very long tracks
- Cross-track threshold transfer learning
