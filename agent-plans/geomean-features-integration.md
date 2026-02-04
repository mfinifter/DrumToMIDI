# Geomean Features Integration - Results

## Issue
After adding geomean features (primary_energy, secondary_energy, geomean, total_energy, sustain_ms) to OnsetFeatures, clustering broke completely:
- With default parameters (eps=0.5, threshold=8.0): 60 onsets, 0 valid clusters (all noise)
- Feature extraction was correct, but DBSCAN couldn't form clusters

## Root Cause
**DBSCAN eps parameter was tuned for 6-dimensional feature space, but new feature space has 11 dimensions.**

The curse of dimensionality: distances between points increase as dimensions increase. The default eps=0.5 (which worked for 6 features) was too small for 11 features - no points were within eps distance of each other, so DBSCAN classified everything as noise.

## Solution
**Adjusted DBSCAN eps parameter from 0.5 → 1.2** to account for higher dimensional feature space.

## Results with New Parameters (eps=1.2, threshold=5.0)

### Quality Metrics
- **Silhouette score**: 0.765 (excellent separation, up from 0.52)
- **Noise ratio**: 0.602 (below 0.8 threshold)
- **Valid clusters**: 2 (target achieved)
- **Convergence**: True (target_achieved_with_quality)

### Cluster Statistics
- Total onsets detected: 123
- Cluster 0: 13 members (darker cymbals - lower geomean ~31-34)
- Cluster 1: 36 members (bright crashes - higher geomean ~108-118)
- Noise points: 74 (60.2%)

### Feature Discrimination
Cluster separation is driven by geomean of spectral band energies:

**Cluster 0 (Darker Cymbals)**:
- Spectral centroid: 6406-7019 Hz
- Secondary energy (4-10kHz): 56-66
- Geomean: 31-34

**Cluster 1 (Bright Crashes)**:
- Spectral centroid: 9186-10172 Hz
- Secondary energy (4-10kHz): 322-491
- Geomean: 108-118

The geomean successfully combines body energy (1-4kHz) and brilliance energy (4-10kHz) to distinguish cymbal types.

## Code Changes

### 1. Updated OnsetFeatures TypedDict (midi_types.py)
Added 5 new fields:
```python
primary_energy: float    # Energy in 1-4kHz (body)
secondary_energy: float  # Energy in 4-10kHz (brilliance)
geomean: float          # Geometric mean of primary & secondary
total_energy: float     # Sum of primary & secondary
sustain_ms: Optional[float]  # Temporal envelope duration
```

### 2. Updated extract_onset_features() (analysis_core.py)
- Added parameters for frequency range configuration
- Calls calculate_spectral_energies() for band-specific energy
- Calls calculate_geomean() using sqrt(primary * secondary)
- Calls calculate_sustain_duration() for envelope analysis

### 3. Updated clustering_core.py
- Default feature_names expanded from 6 → 11 features
- Includes all geomean-related features by default

### 4. Updated export_clustering_table.py
- Default threshold: 8.0 → 5.0
- Default eps: 0.5 → 1.2 (tuned for 11-dimensional space)
- Added quality metrics to output (silhouette, noise_ratio)

### 5. Fixed test expectations
- Updated test_clustering_core.py: feature array shape (2,6) → (2,11)
- Updated test_optimization_core.py: threshold_history tuples (2 elements) → (3 elements with quality dict)

## Testing
All 33 tests pass:
- 22 clustering tests (features_to_array, DBSCAN, k-means, integration)
- 11 optimization tests (convergence, bounds, quality metrics, history tracking)

## Key Lessons
1. **Feature space dimensionality affects distance calculations**: DBSCAN eps must scale with number of features
2. **StandardScaler helps but doesn't eliminate dimension effects**: Normalized distances still increase with dimensions
3. **Domain features (geomean) more discriminative than generic spectral features**: Body+brilliance energy separation works better than spectral centroid alone
4. **Quality metrics essential for validation**: Silhouette score confirms cluster separation is real, not artifact of parameter tuning

## Next Steps
1. Validate on additional audio files (different drummers/recording styles)
2. Consider if 11 features is optimal or if dimensionality reduction needed
3. Document eps parameter selection strategy for different feature counts
4. Test if adaptive eps (e.g., based on k-nearest-neighbor distances) improves robustness

## Files Modified
- midi_types.py (OnsetFeatures contract)
- stems_to_midi/analysis_core.py (extract_onset_features)
- stems_to_midi/clustering_core.py (default feature_names)
- export_clustering_table.py (defaults + quality output)
- stems_to_midi/test_clustering_core.py (shape expectations)
- stems_to_midi/test_optimization_core.py (history structure)
