"""
Tests for extract_onset_features() in analysis_core.py

Tests feature extraction for clustering-based threshold optimization.
"""

import pytest
import numpy as np
from .analysis_core import extract_onset_features


class TestExtractOnsetFeatures:
    """Tests for onset feature extraction."""
    
    def test_basic_feature_extraction(self):
        """Test that basic feature extraction works."""
        sr = 22050
        duration = 1.0
        samples = int(sr * duration)
        
        # Create simple audio with transients
        audio = np.random.randn(samples) * 0.1
        audio[sr//2] = 0.8  # Transient at 0.5s
        
        onset_times = [0.5]
        pan_confidence = [-0.5]  # Left-biased
        
        features = extract_onset_features(audio, sr, onset_times, pan_confidence)
        
        assert len(features) == 1
        assert features[0]['time'] == 0.5
        assert features[0]['pan_confidence'] == -0.5
        assert features[0]['spectral_centroid'] > 0
        assert features[0]['spectral_rolloff'] > 0
        assert 0 <= features[0]['spectral_flatness'] <= 1
        assert features[0]['timing_delta'] is None  # First onset
    
    def test_multiple_onsets_timing_delta(self):
        """Test that timing_delta is calculated correctly for multiple onsets."""
        sr = 22050
        duration = 2.0
        samples = int(sr * duration)
        
        audio = np.random.randn(samples) * 0.1
        onset_times = [0.5, 1.0, 1.3]
        pan_confidence = [0.0, 0.0, 0.0]
        
        features = extract_onset_features(audio, sr, onset_times, pan_confidence)
        
        assert len(features) == 3
        # First onset has no timing_delta
        assert features[0]['timing_delta'] is None
        # Second onset: 1.0 - 0.5 = 0.5s
        assert features[1]['timing_delta'] == pytest.approx(0.5, abs=0.001)
        # Third onset: 1.3 - 1.0 = 0.3s
        assert features[2]['timing_delta'] == pytest.approx(0.3, abs=0.001)
    
    def test_pan_confidence_preserved(self):
        """Test that pan confidence values are preserved in features."""
        sr = 22050
        audio = np.random.randn(sr) * 0.1
        
        onset_times = [0.1, 0.2, 0.3]
        pan_confidence = [-0.8, 0.0, 0.7]  # Left, center, right
        
        features = extract_onset_features(audio, sr, onset_times, pan_confidence)
        
        assert features[0]['pan_confidence'] == -0.8
        assert features[1]['pan_confidence'] == 0.0
        assert features[2]['pan_confidence'] == 0.7
    
    def test_spectral_features_vary(self):
        """Test that spectral features vary between different signals."""
        sr = 22050
        samples = 1024
        
        # Low frequency signal
        t = np.linspace(0, samples/sr, samples)
        low_freq = np.sin(2 * np.pi * 200 * t)
        
        # High frequency signal
        high_freq = np.sin(2 * np.pi * 4000 * t)
        
        onset_times = [0.0]
        pan = [0.0]
        
        low_features = extract_onset_features(low_freq, sr, onset_times, pan)
        high_features = extract_onset_features(high_freq, sr, onset_times, pan)
        
        # High frequency signal should have higher centroid
        assert high_features[0]['spectral_centroid'] > low_features[0]['spectral_centroid']
        # High frequency signal should have higher rolloff
        assert high_features[0]['spectral_rolloff'] > low_features[0]['spectral_rolloff']
    
    def test_tonal_vs_noise_flatness(self):
        """Test that spectral flatness distinguishes tonal from noisy signals."""
        sr = 22050
        samples = 2048
        
        # Pure tone (low flatness)
        t = np.linspace(0, samples/sr, samples)
        tone = np.sin(2 * np.pi * 440 * t)
        
        # White noise (high flatness)
        noise = np.random.randn(samples)
        
        onset_times = [0.0]
        pan = [0.0]
        
        tone_features = extract_onset_features(tone, sr, onset_times, pan)
        noise_features = extract_onset_features(noise, sr, onset_times, pan)
        
        # Noise should have higher flatness than tone
        assert noise_features[0]['spectral_flatness'] > tone_features[0]['spectral_flatness']
        # Tone should have low flatness (< 0.5)
        assert tone_features[0]['spectral_flatness'] < 0.5
        # Noise should have high flatness (> 0.5)
        assert noise_features[0]['spectral_flatness'] > 0.5
    
    def test_pitch_detection_tone(self):
        """Test pitch detection on a clear tone."""
        sr = 22050
        duration = 0.1
        samples = int(sr * duration)
        
        # Generate 440 Hz tone (A4)
        t = np.linspace(0, duration, samples)
        tone = np.sin(2 * np.pi * 440 * t)
        
        onset_times = [0.0]
        pan = [0.0]
        
        features = extract_onset_features(
            tone, sr, onset_times, pan,
            min_pitch_hz=200.0,
            max_pitch_hz=800.0
        )
        
        # Should detect pitch near 440 Hz (allow some tolerance)
        if features[0]['pitch'] is not None:
            assert 400 < features[0]['pitch'] < 480
    
    def test_pitch_none_for_noise(self):
        """Test that pitch is None or unreliable for noise."""
        sr = 22050
        samples = 4096
        
        # White noise (no clear pitch)
        noise = np.random.randn(samples) * 0.5
        
        onset_times = [0.0]
        pan = [0.0]
        
        features = extract_onset_features(noise, sr, onset_times, pan)
        
        # Pitch may be None or just not well-defined
        # (Librosa might detect spurious pitches in noise, so we can't strictly assert None)
        assert 'pitch' in features[0]
    
    def test_empty_onset_list(self):
        """Test with no onsets."""
        sr = 22050
        audio = np.random.randn(sr) * 0.1
        
        features = extract_onset_features(audio, sr, [], [])
        
        assert len(features) == 0
    
    def test_onset_at_audio_boundaries(self):
        """Test onsets at very start and end of audio."""
        sr = 22050
        samples = sr  # 1 second
        audio = np.random.randn(samples) * 0.1
        
        # Onsets at start and near end
        onset_times = [0.0, 0.95]
        pan = [0.0, 0.0]
        
        features = extract_onset_features(audio, sr, onset_times, pan)
        
        # Should handle boundary conditions gracefully
        assert len(features) == 2
        assert features[0]['time'] == 0.0
        assert features[1]['time'] == 0.95
    
    def test_window_size_parameter(self):
        """Test that window_ms parameter affects analysis."""
        sr = 22050
        samples = sr
        audio = np.random.randn(samples) * 0.1
        
        onset_times = [0.5]
        pan = [0.0]
        
        # Small window
        features_small = extract_onset_features(
            audio, sr, onset_times, pan, window_ms=10.0
        )
        
        # Large window
        features_large = extract_onset_features(
            audio, sr, onset_times, pan, window_ms=100.0
        )
        
        # Both should complete successfully
        assert len(features_small) == 1
        assert len(features_large) == 1
        # Features may differ slightly due to different analysis windows
        assert features_small[0]['time'] == features_large[0]['time']
    
    def test_pitch_method_parameter(self):
        """Test that pitch_method parameter is accepted."""
        sr = 22050
        samples = 4096
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, samples/sr, samples))
        
        onset_times = [0.0]
        pan = [0.0]
        
        # Test 'yin' method
        features_yin = extract_onset_features(
            audio, sr, onset_times, pan, pitch_method='yin'
        )
        
        # Test 'pyin' method
        features_pyin = extract_onset_features(
            audio, sr, onset_times, pan, pitch_method='pyin'
        )
        
        # Both should complete
        assert len(features_yin) == 1
        assert len(features_pyin) == 1
    
    def test_feature_dict_structure(self):
        """Test that returned features have all required fields."""
        sr = 22050
        audio = np.random.randn(sr) * 0.1
        
        onset_times = [0.5]
        pan = [-0.3]
        
        features = extract_onset_features(audio, sr, onset_times, pan)
        
        required_fields = [
            'time', 'pan_confidence', 'spectral_centroid',
            'spectral_rolloff', 'spectral_flatness', 'pitch', 'timing_delta'
        ]
        
        for field in required_fields:
            assert field in features[0], f"Missing required field: {field}"
    
    def test_feature_types(self):
        """Test that feature values have correct types."""
        sr = 22050
        audio = np.random.randn(sr) * 0.1
        
        onset_times = [0.3, 0.6]
        pan = [0.2, -0.5]
        
        features = extract_onset_features(audio, sr, onset_times, pan)
        
        for f in features:
            assert isinstance(f['time'], float)
            assert isinstance(f['pan_confidence'], float)
            assert isinstance(f['spectral_centroid'], float)
            assert isinstance(f['spectral_rolloff'], float)
            assert isinstance(f['spectral_flatness'], float)
            assert f['pitch'] is None or isinstance(f['pitch'], float)
            assert f['timing_delta'] is None or isinstance(f['timing_delta'], float)
