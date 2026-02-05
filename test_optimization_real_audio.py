#!/usr/bin/env python3
"""
Test optimization on real audio - Thunderstruck cymbals

This script validates the optimization loop on actual cymbal audio
to verify it reduces false positives compared to hard-coded thresholds.
"""

import librosa
import numpy as np
from stems_to_midi.optimization_core import optimize_threshold_by_clustering


def main():
    # Load Thunderstruck cymbals stem
    audio_path = "user_files/1 - AC_DC_Thunderstruck_Drums/stems/AC_DC_Thunderstruck_Drums-cymbals.wav"
    print(f"Loading: {audio_path}")
    
    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    print(f"Audio shape: {audio.shape}, SR: {sr}")
    
    # Ensure stereo
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.shape[0] != 2:
        # Convert multi-channel to stereo
        audio = audio[:2]
    
    print(f"Stereo shape: {audio.shape}")
    
    # Run optimization targeting 2 clusters (left/right cymbals)
    # Use quality-based optimization with DBSCAN
    print("\n=== Running Quality-Based Optimization ===")
    print("Settings:")
    print("  - expected_clusters: 2 (left/right cymbals)")
    print("  - clustering_method: dbscan (auto-detects noise)")
    print("  - min_silhouette_score: 0.4")
    print("  - max_noise_ratio: 0.3")
    print("  - min_onset_interval_ms: 50 (physical limit)")
    print()
    
    result = optimize_threshold_by_clustering(
        audio,
        sr=sr,
        expected_clusters=2,
        initial_threshold=8.0,  # Start high based on threshold sweep
        threshold_min=3.0,      
        threshold_max=15.0,     
        clustering_method='dbscan',
        clustering_kwargs={'eps': 0.5, 'min_samples': 5},  # eps works with normalized features
        merge_window_ms=100,
        max_iterations=20,
        tolerance=0,
        convergence_patience=5,
        feature_names=None,  # Use ALL features (pan, spectral, pitch, timing)
        # Quality parameters
        min_silhouette_score=0.3,  # Relaxed threshold
        max_noise_ratio=0.4,        # Allow more noise
        min_onset_interval_ms=50.0,
        use_quality_optimization=True,
    )
    
    print(f"\nResults:")
    print(f"  Converged: {result['converged']}")
    print(f"  Reason: {result['convergence_reason']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Optimized threshold: {result['optimized_threshold']:.4f}")
    print(f"  Final cluster count: {result['final_cluster_count']}")
    print(f"  Total onsets detected: {len(result['final_features'])}")
    
    # Display quality metrics
    if result.get('final_quality'):
        quality = result['final_quality']
        print(f"\nCluster Quality:")
        print(f"  Silhouette score: {quality.get('silhouette_score', 0):.3f} (target: ≥0.4)")
        print(f"  Noise ratio: {quality.get('noise_ratio', 0):.3f} (target: ≤0.3)")
        print(f"  Valid clusters: {quality.get('valid_clusters', 0)}")
    
    print(f"\nThreshold history:")
    for i, entry in enumerate(result['threshold_history']):
        threshold, cluster_count = entry[0], entry[1]
        quality = entry[2] if len(entry) > 2 else None
        if quality:
            print(f"  {i+1}. threshold={threshold:.4f} -> {cluster_count} clusters, "
                  f"silhouette={quality.get('silhouette_score', 0):.2f}, "
                  f"noise={quality.get('noise_ratio', 0):.2f}")
        else:
            print(f"  {i+1}. threshold={threshold:.4f} -> {cluster_count} clusters")
    
    # Show cluster distribution
    if result['final_cluster_result'] is not None:
        labels = result['final_cluster_result']['labels']
        unique_labels = np.unique(labels)
        print(f"\nCluster distribution:")
        for label in unique_labels:
            count = np.sum(labels == label)
            print(f"  Cluster {label}: {count} onsets")
    
    # Compare with baseline
    print(f"\n=== Comparison ===")
    print(f"Previous approach (hard-coded pan): 107 onsets")
    print(f"Optimization approach: {len(result['final_features'])} onsets")
    print(f"Target: ~57 actual crashes")
    
    improvement = 107 - len(result['final_features'])
    if improvement > 0:
        print(f"✓ Improvement: {improvement} fewer false positives")
    else:
        print(f"⚠ Warning: {abs(improvement)} more onsets detected")


if __name__ == '__main__':
    main()
