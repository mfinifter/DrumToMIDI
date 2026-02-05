#!/usr/bin/env python3
"""
Threshold sweep test - find optimal threshold for Thunderstruck cymbals

This script tests different threshold values to understand the relationship
between threshold and onset count.
"""

import librosa
import numpy as np
from stems_to_midi.stereo_core import detect_dual_channel_onsets


def main():
    # Load Thunderstruck cymbals stem
    audio_path = "user_files/1 - AC_DC_Thunderstruck_Drums/stems/AC_DC_Thunderstruck_Drums-cymbals.wav"
    print(f"Loading: {audio_path}")
    
    audio, sr = librosa.load(audio_path, sr=None, mono=False)
    print(f"Audio shape: {audio.shape}, SR: {sr}\n")
    
    # Ensure stereo
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.shape[0] != 2:
        audio = audio[:2]
    
    # Try different thresholds
    thresholds = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0]
    
    print("Threshold | Onset Count")
    print("-" * 30)
    
    results = []
    for threshold in thresholds:
        result = detect_dual_channel_onsets(
            audio,
            sr=sr,
            threshold=threshold,
            merge_window_ms=100,
        )
        onset_count = len(result['onset_times'])
        results.append((threshold, onset_count))
        print(f"{threshold:8.1f}  |  {onset_count:5d}")
    
    # Find threshold closest to target of 57 onsets
    target = 57
    best_threshold, best_count = min(results, key=lambda x: abs(x[1] - target))
    
    print(f"\nTarget: {target} onsets")
    print(f"Best threshold: {best_threshold} -> {best_count} onsets")
    print(f"Difference: {abs(best_count - target)} onsets")


if __name__ == '__main__':
    main()
