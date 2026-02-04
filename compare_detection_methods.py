#!/usr/bin/env python3
"""
Compare librosa onset detection vs energy-based detection.
"""

import librosa
import numpy as np
from stems_to_midi.stereo_core import detect_dual_channel_onsets
from stems_to_midi.energy_detection_core import detect_stereo_energy_onsets


def compare_methods(audio_path: str, threshold: float = 3.0):
    """Compare detection methods side-by-side."""
    
    print(f"Loading: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.shape[0] != 2:
        audio = audio[:2]
    
    print(f"\n{'='*70}")
    print("LIBROSA ONSET DETECTION (current method)")
    print(f"{'='*70}")
    
    librosa_result = detect_dual_channel_onsets(
        audio, sr,
        threshold=threshold,
        merge_window_ms=100,
        delta=0.01,
        wait=3,
        hop_length=512,
    )
    
    print(f"Detected: {len(librosa_result['onset_times'])} onsets")
    print(f"Time range: {min(librosa_result['onset_times']):.2f}s - {max(librosa_result['onset_times']):.2f}s")
    
    # Show first 10
    print(f"\nFirst 10 onsets:")
    for i in range(min(10, len(librosa_result['onset_times']))):
        t = librosa_result['onset_times'][i]
        pan = librosa_result['pan_confidence'][i]
        l_str = librosa_result['left_strengths'][i]
        r_str = librosa_result['right_strengths'][i]
        print(f"  {i+1:3d}. t={t:7.3f}s  pan={pan:+.3f}  L={l_str:.2f}  R={r_str:.2f}")
    
    print(f"\n{'='*70}")
    print("ENERGY-BASED DETECTION (new DAW-like method)")
    print(f"{'='*70}")
    
    # Try different threshold_db values
    for thresh_db in [3.0, 6.0, 9.0]:
        energy_result = detect_stereo_energy_onsets(
            audio, sr,
            threshold_db=thresh_db,
            baseline_window_ms=100.0,
            min_event_spacing_ms=50.0,
            merge_window_ms=100.0,
            method='rms',
        )
        
        print(f"\nWith threshold_db={thresh_db}dB:")
        print(f"Detected: {len(energy_result['onset_times'])} onsets")
        
        if len(energy_result['onset_times']) > 0:
            print(f"Time range: {min(energy_result['onset_times']):.2f}s - {max(energy_result['onset_times']):.2f}s")
            
            # Show first 10
            print(f"\nFirst 10 onsets:")
            for i in range(min(10, len(energy_result['onset_times']))):
                t = energy_result['onset_times'][i]
                pan = energy_result['pan_confidence'][i]
                l_e = energy_result['left_energies'][i]
                r_e = energy_result['right_energies'][i]
                dur = energy_result['durations'][i]
                print(f"  {i+1:3d}. t={t:7.3f}s  pan={pan:+.3f}  L={l_e:.4f}  R={r_e:.4f}  dur={dur*1000:.0f}ms")
    
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")
    
    # Look for obvious events around 112s and 119s mentioned by user
    def find_nearby_events(times, target_time, window=0.5):
        return [t for t in times if abs(t - target_time) < window]
    
    print(f"\nEvents near 112s (obvious cymbal in DAW):")
    librosa_near_112 = find_nearby_events(librosa_result['onset_times'], 112.0, 1.0)
    print(f"  Librosa: {len(librosa_near_112)} events - {librosa_near_112[:5] if librosa_near_112 else 'NONE'}")
    
    energy_result_6db = detect_stereo_energy_onsets(
        audio, sr, threshold_db=6.0, method='rms'
    )
    energy_near_112 = find_nearby_events(energy_result_6db['onset_times'], 112.0, 1.0)
    print(f"  Energy (6dB): {len(energy_near_112)} events - {energy_near_112[:5] if energy_near_112 else 'NONE'}")
    
    print(f"\nEvents near 119s (obvious cymbal in DAW):")
    librosa_near_119 = find_nearby_events(librosa_result['onset_times'], 119.0, 1.0)
    print(f"  Librosa: {len(librosa_near_119)} events - {librosa_near_119[:5] if librosa_near_119 else 'NONE'}")
    
    energy_near_119 = find_nearby_events(energy_result_6db['onset_times'], 119.0, 1.0)
    print(f"  Energy (6dB): {len(energy_near_119)} events - {energy_near_119[:5] if energy_near_119 else 'NONE'}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare detection methods')
    parser.add_argument('--audio', default='user_files/1 - AC_DC_Thunderstruck_Drums/stems/AC_DC_Thunderstruck_Drums-cymbals.wav')
    parser.add_argument('--threshold', type=float, default=3.0)
    
    args = parser.parse_args()
    
    compare_methods(args.audio, args.threshold)
