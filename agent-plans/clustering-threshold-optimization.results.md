# Clustering-Based Threshold Optimization Results

**Started**: 2026-02-03  
**Status**: Planning Complete, Ready for Implementation

## Phase Completion Tracking

- [x] Phase 1: Configuration Schema
- [x] Phase 2: Dual-Channel Onset Detection
- [x] Phase 3: Feature Extraction
- [x] Phase 4: Pluggable Clustering
- [ ] Phase 5: Threshold Optimization Loop
- [ ] Phase 6: Pipeline Integration
- [ ] Phase 7: Documentation & Testing

## Metrics

- **Tests Written**: 22 (Phase 4) + 13 (Phase 3) + 8 (Phase 2) + 18 (Phase 1) = 61 total
- **Tests Passing**: 61/61 (18 schema + 8 dual-channel + 13 features + 22 clustering)
- **Coverage**: N/A
- **Integration Tests**: 0
- **Documentation Files Updated**: 0

## Decision Log

### 2026-02-03: Plan Created
- Decided on iterative threshold optimization approach
- Cluster count as objective function for threshold discovery
- Dual-channel detection with configurable merge window (100ms)
- Pluggable clustering (DBSCAN + k-means)
- Opt-in via configuration flag for backward compatibility

### User Requirements Captured
- Onset merge window: 100ms (configurable)
- Clustering method: Both DBSCAN and k-means (externalized)
- Feature priorities: Pan confidence primary, spectral/pitch/timing secondary
- Convergence: Adjust thresholds until cluster count matches expected

### 2026-02-03: Phase 4 Complete - Cleanup Plan Created
- Identified hard-coded pan threshold code in analysis_core.py::classify_cymbal_by_pan()
- Hard-coded thresholds: LEFT_THRESHOLD=-0.25, RIGHT_THRESHOLD=0.25
- Created cleanup plan: agent-plans/cleanup-hard-coded-pan.plan.md
- Added deprecation warnings to classify_cymbal_by_pan() and processing_shell.py
- Will remove deprecated code in Phase 6 after clustering-based approach validated

## Implementation Notes

### Phase 1: Configuration Schema (COMPLETE)
**Files Modified**:
- midiconfig.yaml: Added onset_merge_window_ms (100ms) and expected_clusters to all stems
- webui/settings_schema.py: Added 10 new SettingDefinition entries + 2 new categories (CLUSTERING, THRESHOLD_OPTIMIZATION)

**Configuration Added**:
- Per-stem: onset_merge_window_ms (100ms default), expected_clusters (1-2 depending on stem)
- Global clustering: method ('dbscan' or 'kmeans'), features list (pan_confidence, spectral_centroid, pitch, timing_delta)
- Global threshold_optimization: enabled (false), max_iterations (20), tolerance (0), initial_threshold_step (0.05), convergence_patience (3)

**Tests**: 18/18 passing (all existing schema tests pass with new config)

### Phase 2: Dual-Channel Onset Detection (COMPLETE)
**Files Modified**:
- midi_types.py: Added DualChannelOnsetData TypedDict
- stems_to_midi/stereo_core.py: Added detect_dual_channel_onsets() function
- stems_to_midi/test_stereo_core.py: Added TestDetectDualChannelOnsets class with 8 tests

**Key Implementation Details**:
- Separate left/right channel onset detection using librosa.onset.onset_detect()
- Merge nearby onsets within configurable window (default 100ms)
- Peak strength lookup: Search ±5 frames around backtracked onset to find peak energy
- Pan confidence calculation: (R-L)/(R+L) for each merged onset
- Returns: onset_times, left_strengths, right_strengths, pan_confidence arrays

**Tests**: 8/8 passing
- test_left_only_signal: Detects left-panned onsets with pan < 0
- test_right_only_signal: Detects right-panned onsets with pan > 0
- test_center_signal: Detects centered onsets with pan ≈ 0
- test_onset_merging: Merges nearby L/R onsets within window
- test_no_merging_far_apart: Keeps distant onsets separate
- test_result_structure: Validates return type and array lengths
- test_empty_audio: Handles silent audio gracefully
- test_merge_window_parameter: Window size affects merging behavior

### Phase 3: Feature Extraction (COMPLETE)
**Files Modified**:
- midi_types.py: Added OnsetFeatures TypedDict
- stems_to_midi/analysis_core.py: Added extract_onset_features() function
- stems_to_midi/test_analysis_core_features.py: Added TestExtractOnsetFeatures class with 13 tests

**Key Implementation Details**:
- Extracts 7 features per onset: time, pan_confidence, spectral_centroid, spectral_rolloff, spectral_flatness, pitch, timing_delta
- Spectral features computed using librosa (centroid, rolloff, flatness)
- Pitch detection supports both 'yin' and 'pyin' methods with configurable Hz range
- Timing delta: time since previous onset (None for first onset)
- Configurable analysis window (default 50ms)
- Handles edge cases: boundaries, empty lists, invalid windows

**Tests**: 13/13 passing
- test_basic_feature_extraction: Validates all features computed
- test_multiple_onsets_timing_delta: Timing delta calculated correctly
- test_pan_confidence_preserved: Pan values preserved from input
- test_spectral_features_vary: Spectral features differ for different signals
- test_tonal_vs_noise_flatness: Flatness distinguishes tone from noise
- test_pitch_detection_tone: Detects pitch on clear tones
- test_pitch_none_for_noise: Handles noisy signals gracefully
- test_empty_onset_list: Handles empty input
- test_onset_at_audio_boundaries: Handles edge cases
- test_window_size_parameter: Configurable window size
- test_pitch_method_parameter: Both yin and pyin methods work
- test_feature_dict_structure: All required fields present
- test_feature_types: Correct types for all fields

### Phase 4: Pluggable Clustering (COMPLETE)
**Files Created**:
- stems_to_midi/clustering_core.py: Added features_to_array(), cluster_dbscan(), cluster_kmeans(), cluster_onsets() functions
- stems_to_midi/test_clustering_core.py: Added comprehensive test suite with 22 tests

**Key Implementation Details**:
- **features_to_array()**: Converts OnsetFeatures list to numpy array, handles None values (replaces with 0)
- **cluster_dbscan()**: Density-based clustering, good for arbitrary shapes, identifies outliers as noise (label=-1)
  - Parameters: eps (neighbor distance), min_samples (core point threshold), normalize (StandardScaler)
  - Returns: labels, n_clusters, n_noise, core_sample_indices
- **cluster_kmeans()**: Partitions data into k clusters with nearest mean assignment
  - Parameters: n_clusters (required), normalize, random_state (for reproducibility)
  - Returns: labels, n_clusters, centroids (in original space), inertia (sum of squared distances)
- **cluster_onsets()**: High-level dispatcher function
  - Accepts OnsetFeatures list, method ('dbscan'/'kmeans'), feature_names selection
  - Converts features to array and routes to appropriate clustering function

**Tests**: 22/22 passing
- TestFeaturesToArray (4 tests):
  - test_basic_conversion: Default features exclude 'time'
  - test_custom_feature_names: Select specific features
  - test_none_values_replaced: None pitch → 0.0
  - test_empty_features: Empty list → (0, 0) array
- TestClusterDBSCAN (5 tests):
  - test_two_distinct_clusters: Finds 2 well-separated clusters
  - test_noise_detection: Identifies outliers as noise (label=-1)
  - test_normalization_effect: Normalization changes behavior with scale-diverse features
  - test_single_cluster: All points in one cluster
  - test_empty_input: Graceful handling
- TestClusterKMeans (7 tests):
  - test_two_clusters: Correct partitioning
  - test_three_clusters: All 3 clusters used
  - test_centroids_in_correct_space: Centroids returned in original feature space
  - test_deterministic_with_random_state: Same seed → same results
  - test_more_clusters_than_samples: Auto-caps at n_samples
  - test_inertia_decreases_with_more_clusters: More clusters = better fit
  - test_empty_input: Graceful handling
- TestClusterOnsets (6 tests):
  - test_dbscan_method: Dispatcher routes to DBSCAN
  - test_kmeans_method: Dispatcher routes to k-means
  - test_custom_feature_selection: Feature subset clustering
  - test_kmeans_requires_n_clusters: ValueError if missing
  - test_invalid_method: ValueError for unknown method
  - test_empty_features: Empty input handling

## Test Results

*This section will be updated as tests are written and run*

## Issues Encountered

*This section will document any blockers or surprises during implementation*
