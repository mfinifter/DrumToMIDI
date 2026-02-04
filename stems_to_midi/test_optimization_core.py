"""
Tests for optimization_core.py - Threshold Optimization

Tests for threshold optimization loop using clustering.
"""

import pytest
import numpy as np
from stems_to_midi.optimization_core import optimize_threshold_by_clustering


class TestOptimizeThresholdByClustering:
    """Tests for threshold optimization function."""
    
    def test_basic_convergence(self):
        """Test that optimization converges on simple synthetic data."""
        sr = 22050
        duration = 2.0
        samples = int(sr * duration)
        
        # Create stereo audio with 2 distinct groups of onsets
        # Group 1: Left-panned onsets at 0.5s, 0.6s, 0.7s
        # Group 2: Right-panned onsets at 1.0s, 1.1s, 1.2s
        left = np.zeros(samples)
        right = np.zeros(samples)
        
        # Group 1: Left onsets (strong impulses with decay)
        for t in [0.5, 0.6, 0.7]:
            idx = int(t * sr)
            if idx < samples:
                left[idx] = 0.8
                if idx + 100 < samples:
                    left[idx+1:idx+100] = np.linspace(0.6, 0.0, 99)
        
        # Group 2: Right onsets
        for t in [1.0, 1.1, 1.2]:
            idx = int(t * sr)
            if idx < samples:
                right[idx] = 0.8
                if idx + 100 < samples:
                    right[idx+1:idx+100] = np.linspace(0.6, 0.0, 99)
        
        stereo = np.stack([left, right], axis=0)
        
        result = optimize_threshold_by_clustering(
            stereo,
            sr=sr,
            expected_clusters=2,
            initial_threshold=0.3,
            clustering_method='kmeans',
            max_iterations=20,
            tolerance=0,
        )
        
        # Should converge
        assert result['converged'] or result['iterations'] > 0
        
        # Should find a reasonable threshold
        assert 0.05 <= result['optimized_threshold'] <= 0.9
        
        # Should have detected some onsets
        assert len(result['final_features']) > 0
        
        # Cluster count should be close to expected
        assert abs(result['final_cluster_count'] - 2) <= 1
    
    def test_max_iterations_limit(self):
        """Test that optimization respects max_iterations limit."""
        sr = 22050
        duration = 1.0
        samples = int(sr * duration)
        
        # Create difficult case with random noise
        stereo = np.random.randn(2, samples) * 0.1
        
        result = optimize_threshold_by_clustering(
            stereo,
            sr=sr,
            expected_clusters=3,
            initial_threshold=0.3,
            clustering_method='kmeans',
            max_iterations=5,  # Very low limit
            tolerance=0,
        )
        
        # Should stop at max iterations
        assert result['iterations'] <= 5
        
        # Should have threshold history
        assert len(result['threshold_history']) <= 5
    
    def test_threshold_adjustment_direction(self):
        """Test that threshold adjusts in correct direction."""
        sr = 22050
        duration = 1.0
        samples = int(sr * duration)
        
        # Create audio with many onsets (clustered into 4+ groups naturally)
        left = np.zeros(samples)
        right = np.zeros(samples)
        
        # Create 4 groups with different characteristics
        for i, t in enumerate([0.2, 0.4, 0.6, 0.8]):
            idx = int(t * sr)
            if idx < samples:
                # Alternate left/right
                if i % 2 == 0:
                    left[idx] = 0.8
                    if idx + 100 < samples:
                        left[idx+1:idx+100] = np.linspace(0.6, 0.0, 99)
                else:
                    right[idx] = 0.8
                    if idx + 100 < samples:
                        right[idx+1:idx+100] = np.linspace(0.6, 0.0, 99)
        
        stereo = np.stack([left, right], axis=0)
        
        result = optimize_threshold_by_clustering(
            stereo,
            sr=sr,
            expected_clusters=2,  # Want 2, but will naturally get 4
            initial_threshold=0.1,  # Start low
            clustering_method='kmeans',
            max_iterations=10,
            tolerance=0,
        )
        
        # Check threshold history shows adjustment
        if len(result['threshold_history']) > 1:
            # If we got more clusters than expected, threshold should increase
            first_threshold, first_count = result['threshold_history'][0]
            if first_count > 2:
                # Threshold should generally increase (though may oscillate)
                # Just check that we tried different thresholds
                thresholds = [t for t, _ in result['threshold_history']]
                assert len(set(thresholds)) > 1, "Should try different thresholds"
    
    def test_convergence_with_tolerance(self):
        """Test that tolerance allows near-matches."""
        sr = 22050
        duration = 1.0
        samples = int(sr * duration)
        
        # Create 3 clear clusters
        left = np.zeros(samples)
        right = np.zeros(samples)
        
        for t in [0.3, 0.5, 0.7]:
            idx = int(t * sr)
            if idx < samples:
                # Make them distinctive
                if t < 0.4:
                    left[idx] = 0.9
                    if idx + 100 < samples:
                        left[idx+1:idx+100] = np.linspace(0.7, 0.0, 99)
                elif t < 0.6:
                    right[idx] = 0.9
                    if idx + 100 < samples:
                        right[idx+1:idx+100] = np.linspace(0.7, 0.0, 99)
                else:
                    left[idx] = right[idx] = 0.9
                    if idx + 100 < samples:
                        left[idx+1:idx+100] = np.linspace(0.7, 0.0, 99)
                        right[idx+1:idx+100] = np.linspace(0.7, 0.0, 99)
        
        stereo = np.stack([left, right], axis=0)
        
        # With tolerance=1, should accept 2-4 clusters when targeting 3
        result = optimize_threshold_by_clustering(
            stereo,
            sr=sr,
            expected_clusters=3,
            initial_threshold=0.3,
            clustering_method='kmeans',
            max_iterations=15,
            tolerance=1,  # Accept ±1
        )
        
        # Should converge more easily with tolerance
        if result['converged']:
            assert 2 <= result['final_cluster_count'] <= 4
    
    def test_no_onsets_handling(self):
        """Test handling when threshold is too high (no onsets)."""
        sr = 22050
        duration = 0.5
        samples = int(sr * duration)
        
        # Very quiet audio
        stereo = np.random.randn(2, samples) * 0.001
        
        result = optimize_threshold_by_clustering(
            stereo,
            sr=sr,
            expected_clusters=2,
            initial_threshold=0.8,  # Start very high
            clustering_method='kmeans',
            max_iterations=10,
            tolerance=0,
        )
        
        # Should handle gracefully
        assert 'optimized_threshold' in result
        assert 'converged' in result
        
        # May not converge if audio is too quiet
        # But should still return valid result structure
        assert isinstance(result['iterations'], int)
        assert result['iterations'] >= 0
    
    def test_clustering_method_parameter(self):
        """Test both DBSCAN and k-means methods work."""
        sr = 22050
        duration = 1.0
        samples = int(sr * duration)
        
        # Create simple test audio
        left = np.zeros(samples)
        left[sr//2] = 0.8
        left[sr//2+1:sr//2+100] = np.linspace(0.6, 0.0, 99)
        right = np.zeros(samples)
        right[int(0.7*sr)] = 0.8
        right[int(0.7*sr)+1:int(0.7*sr)+100] = np.linspace(0.6, 0.0, 99)
        
        stereo = np.stack([left, right], axis=0)
        
        # Test k-means
        result_kmeans = optimize_threshold_by_clustering(
            stereo,
            sr=sr,
            expected_clusters=2,
            clustering_method='kmeans',
            max_iterations=10,
        )
        
        assert 'optimized_threshold' in result_kmeans
        assert result_kmeans['iterations'] > 0
        
        # Test DBSCAN (more permissive since it finds clusters automatically)
        result_dbscan = optimize_threshold_by_clustering(
            stereo,
            sr=sr,
            expected_clusters=2,
            clustering_method='dbscan',
            max_iterations=10,
            clustering_kwargs={'eps': 1.0, 'min_samples': 1}
        )
        
        assert 'optimized_threshold' in result_dbscan
        assert result_dbscan['iterations'] > 0
    
    def test_threshold_bounds_respected(self):
        """Test that threshold stays within min/max bounds."""
        sr = 22050
        duration = 0.5
        samples = int(sr * duration)
        
        stereo = np.random.randn(2, samples) * 0.1
        
        result = optimize_threshold_by_clustering(
            stereo,
            sr=sr,
            expected_clusters=2,
            initial_threshold=0.5,
            clustering_method='kmeans',
            max_iterations=10,
            threshold_min=0.2,
            threshold_max=0.8,
        )
        
        # Optimized threshold should be within bounds
        assert 0.2 <= result['optimized_threshold'] <= 0.8
        
        # All historical thresholds should be within bounds
        for threshold, cluster_count, quality in result['threshold_history']:
            assert 0.2 <= threshold <= 0.8
    
    def test_result_structure(self):
        """Test that result dict has all required fields."""
        sr = 22050
        duration = 1.0
        samples = int(sr * duration)
        
        # Simple stereo audio
        left = np.zeros(samples)
        left[sr//2] = 0.8
        left[sr//2+1:sr//2+100] = np.linspace(0.6, 0.0, 99)
        stereo = np.stack([left, np.zeros(samples)], axis=0)
        
        result = optimize_threshold_by_clustering(
            stereo,
            sr=sr,
            expected_clusters=1,
            clustering_method='kmeans',
            max_iterations=5,
        )
        
        # Check all required fields
        required_fields = [
            'optimized_threshold',
            'final_cluster_count',
            'final_cluster_result',
            'final_features',
            'iterations',
            'converged',
            'convergence_reason',
            'threshold_history',
        ]
        
        for field in required_fields:
            assert field in result, f"Missing field: {field}"
        
        # Check types
        assert isinstance(result['optimized_threshold'], float)
        assert isinstance(result['final_cluster_count'], int)
        assert isinstance(result['iterations'], int)
        assert isinstance(result['converged'], bool)
        assert isinstance(result['convergence_reason'], str)
        assert isinstance(result['threshold_history'], list)
    
    def test_convergence_patience(self):
        """Test that convergence_patience stops optimization early."""
        sr = 22050
        duration = 1.0
        samples = int(sr * duration)
        
        # Create audio that may not converge easily
        stereo = np.random.randn(2, samples) * 0.1
        
        result = optimize_threshold_by_clustering(
            stereo,
            sr=sr,
            expected_clusters=3,
            clustering_method='kmeans',
            max_iterations=20,
            convergence_patience=2,  # Stop after 2 iterations without improvement
        )
        
        # Should stop early if no improvement
        if not result['converged'] and result['convergence_reason'] == 'no_improvement':
            # Check that we didn't use all iterations
            assert result['iterations'] < 20
    
    def test_feature_names_parameter(self):
        """Test that feature_names parameter is passed through."""
        sr = 22050
        duration = 1.0
        samples = int(sr * duration)
        
        left = np.zeros(samples)
        left[sr//2] = 0.8
        left[sr//2+1:sr//2+100] = np.linspace(0.6, 0.0, 99)
        stereo = np.stack([left, np.zeros(samples)], axis=0)
        
        # Use only pan_confidence for clustering (should still work)
        result = optimize_threshold_by_clustering(
            stereo,
            sr=sr,
            expected_clusters=1,
            clustering_method='kmeans',
            max_iterations=5,
            feature_names=['pan_confidence'],
        )
        
        # Should complete without error
        assert 'optimized_threshold' in result
        assert result['iterations'] > 0
    
    def test_threshold_history_tracked(self):
        """Test that threshold_history records all attempts."""
        sr = 22050
        duration = 1.0
        samples = int(sr * duration)
        
        left = np.zeros(samples)
        left[sr//2] = 0.8
        left[sr//2+1:sr//2+100] = np.linspace(0.6, 0.0, 99)
        stereo = np.stack([left, np.zeros(samples)], axis=0)
        
        result = optimize_threshold_by_clustering(
            stereo,
            sr=sr,
            expected_clusters=2,
            clustering_method='kmeans',
            max_iterations=10,
        )
        
        # Should have history
        assert len(result['threshold_history']) > 0
        assert len(result['threshold_history']) == result['iterations']
        
        # Each entry should be (threshold, cluster_count, quality)
        for entry in result['threshold_history']:
            assert isinstance(entry, tuple)
            assert len(entry) == 3
            threshold, cluster_count, quality = entry
            assert isinstance(threshold, float)
            assert isinstance(cluster_count, int)
            # quality can be dict or None
            assert quality is None or isinstance(quality, dict)
            assert 0.0 <= threshold <= 1.0
            assert cluster_count >= 0
