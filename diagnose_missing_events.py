#!/usr/bin/env python3
"""Diagnose why events around 112s and 119s are missing."""

import librosa
import numpy as np
from stems_to_midi.energy_detection_core import calculate_energy_envelope, detect_transient_peaks

audio, sr = librosa.load('user_files/1 - AC_DC_Thunderstruck_Drums/stems/AC_DC_Thunderstruck_Drums-cymbals.wav', sr=None, mono=False)
if audio.ndim == 1:
    audio = np.stack([audio, audio], axis=0)

# Calculate energy
left_times, left_energy = calculate_energy_envelope(audio[0], sr, 2048, 512, 'rms')
right_times, right_energy = calculate_energy_envelope(audio[1], sr, 2048, 512, 'rms')

# Check specific time windows
for target_time in [112.0, 119.0]:
    idx = np.argmin(np.abs(left_times - target_time))
    window = slice(max(0, idx-10), min(len(left_times), idx+10))
    
    print(f"\n{'='*60}")
    print(f"Around {target_time}s (frame {idx}):")
    print(f"{'='*60}")
    print(f"Time range: {left_times[window][0]:.2f}s - {left_times[window][-1]:.2f}s")
    print(f"\nLeft channel:")
    print(f"  At {target_time:.1f}s: {left_energy[idx]:.6f}")
    print(f"  Max in window: {left_energy[window].max():.6f} at {left_times[window][np.argmax(left_energy[window])]:.3f}s")
    print(f"  Min in window: {left_energy[window].min():.6f}")
    
    print(f"\nRight channel:")
    print(f"  At {target_time:.1f}s: {right_energy[idx]:.6f}")
    print(f"  Max in window: {right_energy[window].max():.6f} at {right_times[window][np.argmax(right_energy[window])]:.3f}s")
    print(f"  Min in window: {right_energy[window].min():.6f}")
    
    print(f"\nThresholds:")
    print(f"  min_absolute_energy: 0.005")
    print(f"  Left passes: {left_energy[idx] >= 0.005}")
    print(f"  Right passes: {right_energy[idx] >= 0.005}")

print(f"\n{'='*60}")
print("Detecting with different thresholds:")
print(f"{'='*60}")

for threshold_db in [4.0, 6.5, 8.0]:
    for min_abs_energy in [0.001, 0.005, 0.01]:
        left_peaks = detect_transient_peaks(
            left_times, left_energy, 
            threshold_db=threshold_db,
            min_absolute_energy=min_abs_energy
        )
        right_peaks = detect_transient_peaks(
            right_times, right_energy,
            threshold_db=threshold_db, 
            min_absolute_energy=min_abs_energy
        )
        
        # Check if we found events near 112 and 119
        left_near_112 = [p for p in left_peaks if abs(p['onset_time'] - 112.0) < 2.0]
        left_near_119 = [p for p in left_peaks if abs(p['onset_time'] - 119.0) < 2.0]
        right_near_112 = [p for p in right_peaks if abs(p['onset_time'] - 112.0) < 2.0]
        right_near_119 = [p for p in right_peaks if abs(p['onset_time'] - 119.0) < 2.0]
        
        print(f"\nthreshold_db={threshold_db}, min_energy={min_abs_energy}:")
        print(f"  Total: L={len(left_peaks)} R={len(right_peaks)}")
        print(f"  Near 112s: L={len(left_near_112)} R={len(right_near_112)}")
        print(f"  Near 119s: L={len(left_near_119)} R={len(right_near_119)}")
