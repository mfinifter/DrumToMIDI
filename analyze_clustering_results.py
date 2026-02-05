#!/usr/bin/env python3
"""
Analyze and visualize clustering results for human review.

Creates a detailed markdown report showing:
- All detected onsets with features
- Cluster assignments
- Quality metrics
- Recommendations for parameter tuning
"""

import librosa
import numpy as np
from stems_to_midi.optimization_core import optimize_threshold_by_clustering
from pathlib import Path


def format_feature_row(idx: int, feature: dict, label: int) -> str:
    """Format a single onset feature as markdown table row."""
    cluster_str = "NOISE" if label == -1 else f"C{label}"
    
    # Handle None values
    def fmt(val, default=0):
        return default if val is None else val
    
    return (
        f"| {idx+1:3d} | {fmt(feature.get('time'), 0):6.3f} | {cluster_str:>6} | "
        f"{fmt(feature.get('pan_confidence'), 0):6.3f} | {fmt(feature.get('spectral_centroid'), 0):8.1f} | "
        f"{fmt(feature.get('spectral_rolloff'), 0):8.1f} | {fmt(feature.get('spectral_flatness'), 0):6.4f} | "
        f"{fmt(feature.get('pitch'), 0):7.1f} | {fmt(feature.get('timing_delta'), 0):6.3f} |"
    )


def generate_clustering_report(
    audio_path: str,
    output_path: str,
    threshold: float = 8.0,
    threshold_min: float = 3.0,
    threshold_max: float = 15.0,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 5,
    min_silhouette: float = 0.3,
    max_noise: float = 0.8,
):
    """Generate detailed markdown report of clustering analysis."""
    
    print(f"Loading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    
    # Ensure stereo
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.shape[0] != 2:
        audio = audio[:2]
    
    print(f"Running optimization...")
    result = optimize_threshold_by_clustering(
        audio,
        sr=sr,
        expected_clusters=2,
        initial_threshold=threshold,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        clustering_method='dbscan',
        clustering_kwargs={'eps': dbscan_eps, 'min_samples': dbscan_min_samples},
        merge_window_ms=100,
        max_iterations=20,
        tolerance=0,
        convergence_patience=5,
        feature_names=None,  # All features
        min_silhouette_score=min_silhouette,
        max_noise_ratio=max_noise,
        min_onset_interval_ms=50.0,
        use_quality_optimization=True,
    )
    
    print(f"Generating report: {output_path}")
    
    features = result['final_features']
    labels = result['final_cluster_result']['labels']
    quality = result['final_quality']
    
    # Sort onsets by cluster, then by time
    onset_data = list(zip(features, labels))
    onset_data.sort(key=lambda x: (x[1], x[0]['time']))
    
    # Count per cluster
    cluster_counts = {}
    for label in labels:
        cluster_counts[label] = cluster_counts.get(label, 0) + 1
    
    # Generate markdown report
    lines = [
        "# Cymbal Clustering Analysis Report",
        "",
        f"**Audio File:** `{Path(audio_path).name}`",
        f"**Sample Rate:** {sr} Hz",
        f"**Duration:** {len(audio[0])/sr:.2f} seconds",
        "",
        "## Optimization Settings",
        "",
        f"- **Threshold:** {result['optimized_threshold']:.4f}",
        f"- **DBSCAN eps:** {dbscan_eps}",
        f"- **DBSCAN min_samples:** {dbscan_min_samples}",
        f"- **Min silhouette score:** {min_silhouette}",
        f"- **Max noise ratio:** {max_noise}",
        "",
        "## Results Summary",
        "",
        f"- **Total onsets detected:** {len(features)}",
        f"- **Converged:** {result['converged']}",
        f"- **Convergence reason:** {result['convergence_reason']}",
        f"- **Iterations:** {result['iterations']}",
        "",
        "### Quality Metrics",
        "",
        f"- **Silhouette score:** {quality.get('silhouette_score', 0):.3f} (target: ≥{min_silhouette})",
        f"- **Noise ratio:** {quality.get('noise_ratio', 0):.3f} (target: ≤{max_noise})",
        f"- **Valid clusters:** {quality.get('valid_clusters', 0)}",
        "",
        "### Cluster Distribution",
        "",
    ]
    
    for label in sorted(cluster_counts.keys()):
        count = cluster_counts[label]
        pct = (count / len(labels) * 100) if len(labels) > 0 else 0
        if label == -1:
            lines.append(f"- **NOISE:** {count} onsets ({pct:.1f}%)")
        else:
            lines.append(f"- **Cluster {label}:** {count} onsets ({pct:.1f}%)")
    
    lines.extend([
        "",
        "## Threshold History",
        "",
        "| Iteration | Threshold | Clusters | Silhouette | Noise Ratio |",
        "|-----------|-----------|----------|------------|-------------|",
    ])
    
    for i, entry in enumerate(result['threshold_history'], 1):
        threshold_val, cluster_count = entry[0], entry[1]
        qual = entry[2] if len(entry) > 2 and entry[2] else {}
        sil = qual.get('silhouette_score', 0)
        noise = qual.get('noise_ratio', 0)
        lines.append(f"| {i:3d} | {threshold_val:9.4f} | {cluster_count:8d} | {sil:10.3f} | {noise:11.3f} |")
    
    lines.extend([
        "",
        "## Detected Onsets (Sorted by Cluster)",
        "",
        "| # | Time(s) | Cluster | Pan | Spec.Cent | Spec.Roll | Spec.Flat | Pitch(Hz) | TimeDelta |",
        "|---|---------|---------|-----|-----------|-----------|-----------|-----------|-----------|",
    ])
    
    for idx, (feature, label) in enumerate(onset_data):
        lines.append(format_feature_row(idx, feature, label))
    
    # Add analysis section
    lines.extend([
        "",
        "## Analysis Notes",
        "",
        "### Feature Meanings",
        "",
        "- **Pan:** Stereo position (-1=left, 0=center, +1=right)",
        "- **Spec.Cent:** Spectral centroid (brightness, Hz)",
        "- **Spec.Roll:** Spectral rolloff (energy distribution, Hz)",
        "- **Spec.Flat:** Spectral flatness (0=tonal, 1=noise-like)",
        "- **Pitch:** Estimated fundamental frequency (Hz)",
        "- **TimeDelta:** Time since previous onset (seconds)",
        "",
        "### Interpreting Results",
        "",
        "**NOISE points:** Onsets that don't fit into dense clusters",
        "- May be false positives (hihat bleed, artifacts)",
        "- May be valid but sparse onsets (single crash hits)",
        "- High noise ratio suggests threshold is too low or clusters too tight",
        "",
        "**Cluster separation:** Measured by silhouette score",
        "- >0.7: Excellent separation",
        "- >0.5: Good separation",
        "- >0.3: Acceptable separation",
        "- <0.3: Poor separation, consider different features or eps",
        "",
        "### Parameter Tuning Guidance",
        "",
    ])
    
    # Add recommendations based on results
    if quality.get('noise_ratio', 0) > 0.5:
        lines.append("⚠️ **High noise ratio** - Consider:")
        lines.append("- Decrease DBSCAN eps (tighter clusters)")
        lines.append("- Increase DBSCAN min_samples (require more dense regions)")
        lines.append("- Increase threshold (fewer weak onsets)")
        lines.append("")
    
    if quality.get('silhouette_score', 0) < 0.3:
        lines.append("⚠️ **Low silhouette score** - Consider:")
        lines.append("- Different feature subset (try pan_confidence + spectral_centroid only)")
        lines.append("- Adjust DBSCAN eps")
        lines.append("- Check if clusters actually exist in this audio")
        lines.append("")
    
    if quality.get('valid_clusters', 0) != 2:
        lines.append(f"⚠️ **Found {quality.get('valid_clusters', 0)} clusters instead of 2** - Consider:")
        lines.append("- Adjust threshold to change onset count")
        lines.append("- Review DBSCAN parameters")
        lines.append("- Verify audio has expected stereo separation")
        lines.append("")
    
    lines.extend([
        "### Quick Actions",
        "",
        "To adjust parameters, run:",
        "```bash",
        f"python analyze_clustering_results.py \\",
        f"  --threshold {threshold} \\",
        f"  --eps {dbscan_eps} \\",
        f"  --min-samples {dbscan_min_samples}",
        "```",
    ])
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✓ Report written to {output_path}")
    print(f"  Total onsets: {len(features)}")
    print(f"  Valid clusters: {quality.get('valid_clusters', 0)}")
    print(f"  Noise points: {cluster_counts.get(-1, 0)}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze cymbal clustering results')
    parser.add_argument('--audio', default='user_files/1 - AC_DC_Thunderstruck_Drums/stems/AC_DC_Thunderstruck_Drums-cymbals.wav',
                        help='Path to audio file')
    parser.add_argument('--output', default='clustering_analysis.md',
                        help='Output markdown file')
    parser.add_argument('--threshold', type=float, default=8.0,
                        help='Initial onset detection threshold')
    parser.add_argument('--threshold-min', type=float, default=3.0,
                        help='Minimum threshold')
    parser.add_argument('--threshold-max', type=float, default=15.0,
                        help='Maximum threshold')
    parser.add_argument('--eps', type=float, default=0.5,
                        help='DBSCAN eps parameter')
    parser.add_argument('--min-samples', type=int, default=5,
                        help='DBSCAN min_samples parameter')
    parser.add_argument('--min-silhouette', type=float, default=0.3,
                        help='Minimum silhouette score')
    parser.add_argument('--max-noise', type=float, default=0.8,
                        help='Maximum noise ratio')
    
    args = parser.parse_args()
    
    generate_clustering_report(
        audio_path=args.audio,
        output_path=args.output,
        threshold=args.threshold,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        dbscan_eps=args.eps,
        dbscan_min_samples=args.min_samples,
        min_silhouette=args.min_silhouette,
        max_noise=args.max_noise,
    )
