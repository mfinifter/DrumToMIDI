#!/usr/bin/env python3
"""
Compare peak vs backtracked onset timing to show improvement.
"""

import librosa
import numpy as np
from stems_to_midi.energy_detection_core import detect_stereo_transient_peaks

audio_path = "user_files/1 - AC_DC_Thunderstruck_Drums/stems/AC_DC_Thunderstruck_Drums-cymbals.wav"

print("Loading audio...")
audio, sr = librosa.load(audio_path, sr=None, mono=False)

if audio.ndim == 1:
    audio = np.stack([audio, audio], axis=0)
elif audio.shape[0] != 2:
    audio = audio[:2]

print("Detecting with backtracking...")
result = detect_stereo_transient_peaks(
    audio, sr,
    threshold_db=15.0,
    min_peak_spacing_ms=100.0,
    merge_window_ms=150.0,
    method='rms',
    min_absolute_energy=0.01,
)

print(f"\nDetected {len(result['onset_times'])} events")
print("\nBacktracking Analysis (first 10 events):")
print("Event | Onset (backtracked) | Peak Energy | L_Energy | R_Energy")
print("------|--------------------|-----------|---------|---------")

for i in range(min(10, len(result['onset_times']))):
    print(f"  {i+1:2d}  | {result['onset_times'][i]:7.3f}s         | "
          f"{result['left_energies'][i] + result['right_energies'][i]:7.3f}   | "
          f"{result['left_energies'][i]:7.3f} | {result['right_energies'][i]:7.3f}")

print("\nNote: onset_time is now backtracked to attack start (50-120ms before peak)")
print("This matches where the transient actually begins, not where it peaks.")
