"""
Threshold Optimization Core - Pure Functional Core

Pure functions for optimizing onset detection thresholds using clustering.
All functions are deterministic with no side effects.

Architecture: Functional Core
- No I/O operations
- Deterministic (same input → same output, given fixed random seeds)
- No external state or side effects
- Testable in isolation

The optimization strategy:
1. Detect onsets with current threshold
2. Extract features from onsets
3. Cluster features
4. Compare cluster count to expected
5. Adjust threshold based on cluster count (binary search)
6. Repeat until convergence or max iterations
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Literal
import librosa  # type: ignore
from sklearn.metrics import silhouette_score

# Import from sibling modules
try:
    from midi_types import DualChannelOnsetData, OnsetFeatures
    from stems_to_midi.stereo_core import detect_dual_channel_onsets
    from stems_to_midi.analysis_core import extract_onset_features
    from stems_to_midi.clustering_core import cluster_onsets
except ImportError:
    # Running from stems_to_midi/ directly
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from midi_types import DualChannelOnsetData, OnsetFeatures
    from stereo_core import detect_dual_channel_onsets
    from analysis_core import extract_onset_features
    from clustering_core import cluster_onsets


__all__ = [
    'optimize_threshold_by_clustering',
    'filter_temporal_impossibilities',
    'calculate_cluster_quality',
]


def filter_temporal_impossibilities(
    features: List[OnsetFeatures],
    min_onset_interval_ms: float = 50.0,
) -> List[OnsetFeatures]:
    """
    Remove onsets that are temporally impossible for human performance.
    
    Filters out onsets that occur too close together (< min_onset_interval_ms)
    within the same cluster, keeping only the strongest onset in each group.
    
    Pure function - no side effects.
    
    Args:
        features: List of onset features to filter
        min_onset_interval_ms: Minimum time between onsets (milliseconds)
            Default 50ms = physically impossible for humans
    
    Returns:
        Filtered list of OnsetFeatures
    
    Examples:
        >>> # Two onsets 30ms apart (impossible) - keep strongest
        >>> features = [
        ...     {'time': 1.0, 'pan_confidence': 0.5, 'spectral_centroid': 2000, ...},
        ...     {'time': 1.03, 'pan_confidence': 0.5, 'spectral_centroid': 1800, ...},
        ... ]
        >>> filtered = filter_temporal_impossibilities(features, min_onset_interval_ms=50)
        >>> len(filtered) == 1  # Only one kept
    """
    if len(features) == 0:
        return []
    
    # Sort by time
    sorted_features = sorted(features, key=lambda f: f['time'])
    
    min_interval_sec = min_onset_interval_ms / 1000.0
    filtered = []
    
    i = 0
    while i < len(sorted_features):
        current = sorted_features[i]
        group = [current]
        
        # Find all onsets within min_interval of current
        j = i + 1
        while j < len(sorted_features):
            next_feature = sorted_features[j]
            if next_feature['time'] - current['time'] < min_interval_sec:
                group.append(next_feature)
                j += 1
            else:
                break
        
        # Keep the strongest onset in the group
        # Use spectral_centroid as proxy for strength (higher = stronger)
        strongest = max(group, key=lambda f: f.get('spectral_centroid', 0))
        filtered.append(strongest)
        
        i = j if j > i + 1 else i + 1
    
    return filtered


def calculate_cluster_quality(
    features: List[OnsetFeatures],
    cluster_result: Dict[str, any],
    feature_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calculate quality metrics for clustering results.
    
    Computes metrics that indicate how well-separated and meaningful the clusters are:
    - silhouette_score: How well-separated clusters are (-1 to 1, higher better)
    - noise_ratio: Fraction of onsets labeled as noise (0 to 1, lower better)
    - valid_clusters: Number of non-noise clusters
    
    Pure function - no side effects.
    
    Args:
        features: List of onset features that were clustered
        cluster_result: Result dict from cluster_onsets()
        feature_names: Which features to use for quality calculation
    
    Returns:
        Dict with keys: silhouette_score, noise_ratio, valid_clusters
    
    Examples:
        >>> features = [...]
        >>> cluster_result = cluster_onsets(features, 'dbscan')
        >>> quality = calculate_cluster_quality(features, cluster_result)
        >>> quality['silhouette_score']  # 0.7 = good separation
        >>> quality['noise_ratio']  # 0.1 = 10% noise
    """
    if len(features) == 0:
        return {
            'silhouette_score': 0.0,
            'noise_ratio': 1.0,
            'valid_clusters': 0,
        }
    
    labels = cluster_result['labels']
    
    # Count noise (label = -1) vs valid clusters
    noise_count = np.sum(labels == -1)
    noise_ratio = noise_count / len(labels) if len(labels) > 0 else 1.0
    
    valid_clusters = len(set(labels) - {-1})
    
    # Calculate silhouette score if we have enough valid clusters
    silhouette = 0.0
    if valid_clusters >= 2 and noise_count < len(labels):
        # Remove noise points for silhouette calculation
        valid_mask = labels != -1
        if np.sum(valid_mask) >= 2:
            from stems_to_midi.clustering_core import features_to_array
            X = features_to_array(features, feature_names=feature_names)
            X_valid = X[valid_mask]
            labels_valid = labels[valid_mask]
            
            # Only calculate if we have multiple unique labels
            if len(np.unique(labels_valid)) >= 2:
                try:
                    silhouette = silhouette_score(X_valid, labels_valid)
                except ValueError:
                    # Can fail if clusters are too small or poorly formed
                    silhouette = 0.0
    
    return {
        'silhouette_score': float(silhouette),
        'noise_ratio': float(noise_ratio),
        'valid_clusters': int(valid_clusters),
    }


def optimize_threshold_by_clustering(
    stereo_audio: np.ndarray,
    sr: int,
    expected_clusters: int,
    initial_threshold: float = 0.3,
    clustering_method: Literal['dbscan', 'kmeans'] = 'kmeans',
    merge_window_ms: float = 100.0,
    max_iterations: int = 20,
    tolerance: int = 0,
    threshold_min: float = 0.05,
    threshold_max: float = 0.9,
    threshold_step_initial: float = 0.1,
    convergence_patience: int = 3,
    feature_names: Optional[List[str]] = None,
    clustering_kwargs: Optional[Dict] = None,
    # Quality-based optimization parameters
    min_silhouette_score: float = 0.4,
    max_noise_ratio: float = 0.3,
    min_onset_interval_ms: float = 50.0,
    use_quality_optimization: bool = True,
) -> Dict[str, any]:
    """
    Optimize onset detection threshold to achieve target cluster count.
    
    Uses binary search to find threshold that produces onset count such that
    clustering yields expected number of clusters. This automatically discovers
    optimal threshold for each track's characteristics.
    
    Algorithm:
    1. Start with initial threshold
    2. Detect dual-channel onsets
    3. Extract features
    4. Cluster features
    5. Compare cluster count to expected
    6. If too many clusters → increase threshold (fewer onsets)
    7. If too few clusters → decrease threshold (more onsets)
    8. Repeat with binary search until convergence
    
    Pure function - deterministic for fixed random seeds in clustering.
    
    Args:
        stereo_audio: Stereo audio array (2, samples) or (samples, 2)
        sr: Sample rate in Hz
        expected_clusters: Target number of clusters (e.g., 2 for L/R cymbals)
        initial_threshold: Starting onset detection threshold (0-1)
        clustering_method: 'dbscan' or 'kmeans'
        merge_window_ms: Window for merging L/R onsets (ms)
        max_iterations: Maximum optimization iterations
        tolerance: Stop when cluster count within ±tolerance of expected
        threshold_min: Minimum allowed threshold
        threshold_max: Maximum allowed threshold
        threshold_step_initial: Initial step size for binary search
        convergence_patience: Stop if no improvement for N iterations
        feature_names: Which features to use for clustering (default: all)
        clustering_kwargs: Additional kwargs for clustering (e.g., eps, random_state)
        min_silhouette_score: Minimum acceptable silhouette score (0-1, default 0.4)
            Higher = clusters must be more well-separated
        max_noise_ratio: Maximum acceptable noise ratio (0-1, default 0.3)
            Lower = fewer onsets can be labeled as noise
        min_onset_interval_ms: Minimum time between onsets (ms, default 50)
            Filters physically impossible onset timing
        use_quality_optimization: If True, optimize for cluster quality + count
            If False, optimize only for cluster count (legacy behavior)
    
    Returns:
        Dict with:
            - optimized_threshold: Best threshold found
            - final_cluster_count: Number of clusters at optimal threshold
            - final_cluster_result: Full clustering result dict
            - final_features: List of OnsetFeatures at optimal threshold
            - final_quality: Cluster quality metrics dict
            - iterations: Number of iterations used
            - converged: Whether optimization converged
            - convergence_reason: Why optimization stopped
            - threshold_history: List of (threshold, cluster_count, quality) tuples
    
    Examples:
        >>> audio = np.random.randn(2, 44100)  # 1 second stereo
        >>> result = optimize_threshold_by_clustering(
        ...     audio, sr=22050, expected_clusters=2,
        ...     clustering_method='kmeans'
        ... )
        >>> result['converged']
        True
        >>> 0.05 <= result['optimized_threshold'] <= 0.9
        True
    """
    if clustering_kwargs is None:
        clustering_kwargs = {}
    
    # Binary search state
    threshold_low = threshold_min
    threshold_high = threshold_max
    current_threshold = initial_threshold
    
    # Track history
    threshold_history: List[Tuple[float, int, Optional[Dict]]] = []
    
    # Best result tracking
    best_threshold = initial_threshold
    best_cluster_count = 0
    best_cluster_result = None
    best_features = []
    best_quality = {}
    best_score = float('-inf')  # Composite score (higher = better)
    
    # Convergence tracking
    iterations_without_improvement = 0
    converged = False
    convergence_reason = "max_iterations_reached"
    
    for iteration in range(max_iterations):
        # Step 1: Detect onsets with current threshold
        onset_data = detect_dual_channel_onsets(
            stereo_audio,
            sr,
            merge_window_ms=merge_window_ms,
            threshold=current_threshold,
            delta=0.01,
            wait=3,
            hop_length=512,
        )
        
        # Check if we got any onsets
        if len(onset_data['onset_times']) == 0:
            # No onsets detected - threshold too high
            threshold_history.append((current_threshold, 0))
            
            # Lower threshold
            threshold_high = current_threshold
            current_threshold = (threshold_low + current_threshold) / 2
            
            # Check if we're stuck at minimum
            if current_threshold <= threshold_min + 0.01:
                convergence_reason = "no_onsets_at_minimum_threshold"
                break
            
            continue
        
        # Step 2: Extract features
        features = extract_onset_features(
            audio=stereo_audio if stereo_audio.ndim == 1 else np.mean(stereo_audio, axis=0 if stereo_audio.shape[0] == 2 else 1),
            sr=sr,
            onset_times=onset_data['onset_times'],
            pan_confidence=onset_data['pan_confidence'],
            window_ms=50.0,
            pitch_method='yin',
            min_pitch_hz=80.0,
            max_pitch_hz=10000.0
        )
        
        # Step 2.5: Apply temporal filtering (remove physically impossible onsets)
        if use_quality_optimization and min_onset_interval_ms > 0:
            features = filter_temporal_impossibilities(features, min_onset_interval_ms)
        
        # Recheck if we still have features after filtering
        if len(features) == 0:
            threshold_history.append((current_threshold, 0, None))
            threshold_high = current_threshold
            current_threshold = (threshold_low + current_threshold) / 2
            if current_threshold <= threshold_min + 0.01:
                convergence_reason = "no_onsets_after_filtering"
                break
            continue
        
        # Step 3: Cluster features
        cluster_result = cluster_onsets(
            features,
            method=clustering_method,
            n_clusters=expected_clusters if clustering_method == 'kmeans' else None,
            feature_names=feature_names,
            **clustering_kwargs
        )
        
        cluster_count = cluster_result['n_clusters']
        
        # Step 4: Calculate cluster quality
        quality = calculate_cluster_quality(features, cluster_result, feature_names) if use_quality_optimization else {}
        threshold_history.append((current_threshold, cluster_count, quality if quality else None))
        
        # Step 5: Evaluate this result
        if use_quality_optimization:
            # Multi-objective score: quality metrics + cluster count proximity
            distance = abs(cluster_count - expected_clusters)
            cluster_score = 1.0 / (1.0 + distance)  # 1.0 if perfect, decreases with distance
            
            # Quality gates: must pass all thresholds
            quality_passed = (
                quality['silhouette_score'] >= min_silhouette_score and
                quality['noise_ratio'] <= max_noise_ratio and
                quality['valid_clusters'] >= 1
            )
            
            # Composite score: quality + cluster proximity
            if quality_passed:
                composite_score = cluster_score + quality['silhouette_score']
            else:
                # Penalize failures to meet quality thresholds
                composite_score = cluster_score * 0.5
            
            # Update best if score improved
            if composite_score > best_score:
                best_threshold = current_threshold
                best_cluster_count = cluster_count
                best_cluster_result = cluster_result
                best_features = features
                best_quality = quality
                best_score = composite_score
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
        else:
            # Legacy: optimize only for cluster count
            distance = abs(cluster_count - expected_clusters)
            if distance < abs(best_cluster_count - expected_clusters):
                best_threshold = current_threshold
                best_cluster_count = cluster_count
                best_cluster_result = cluster_result
                best_features = features
                best_quality = quality if quality else {}
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
        
        # Check if we've converged
        if use_quality_optimization:
            # Quality-based convergence: cluster count + quality thresholds
            quality_met = (
                quality['silhouette_score'] >= min_silhouette_score and
                quality['noise_ratio'] <= max_noise_ratio
            )
            cluster_count_met = abs(cluster_count - expected_clusters) <= tolerance
            
            if cluster_count_met and quality_met:
                converged = True
                convergence_reason = "target_achieved_with_quality"
                break
        else:
            # Legacy: just cluster count
            if abs(cluster_count - expected_clusters) <= tolerance:
                converged = True
                convergence_reason = "target_achieved"
                break
        
        # Check patience
        if iterations_without_improvement >= convergence_patience:
            convergence_reason = "no_improvement"
            break
        
        # Step 5: Adjust threshold (binary search)
        if cluster_count > expected_clusters:
            # Too many clusters → need fewer onsets → increase threshold
            threshold_low = current_threshold
            new_threshold = (current_threshold + threshold_high) / 2
        else:
            # Too few clusters → need more onsets → decrease threshold
            threshold_high = current_threshold
            new_threshold = (threshold_low + current_threshold) / 2
        
        # Check if search space exhausted
        if abs(new_threshold - current_threshold) < 0.01:
            convergence_reason = "threshold_range_exhausted"
            break
        
        current_threshold = new_threshold
    
    return {
        'optimized_threshold': best_threshold,
        'final_cluster_count': best_cluster_count,
        'final_cluster_result': best_cluster_result,
        'final_features': best_features,
        'final_quality': best_quality,
        'iterations': len(threshold_history),
        'converged': converged,
        'convergence_reason': convergence_reason,
        'threshold_history': threshold_history,
    }


# Type alias for result
ThresholdOptimizationResult = Dict[str, any]
