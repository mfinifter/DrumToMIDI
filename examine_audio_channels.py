#!/usr/bin/env python
"""Examine audio channel balance"""
import soundfile as sf
import numpy as np

# Load audio file
audio_path = 'user_files/14 - AC_DC_Thunderstruck_Drums/stems/AC_DC_Thunderstruck_Drums-hihat.wav'
audio, sr = sf.read(audio_path)

print('=== Audio File Info ===')
print(f'Shape: {audio.shape}')
print(f'Sample rate: {sr}')
print(f'Channels: {audio.shape[1] if audio.ndim > 1 else 1}')
print(f'Duration: {len(audio)/sr:.2f} seconds')

if audio.ndim > 1:
    left = audio[:, 0]
    right = audio[:, 1]
    
    print(f'\n=== Channel Statistics ===')
    print(f'Left channel:')
    print(f'  min={left.min():.6f}, max={left.max():.6f}')
    print(f'  mean amplitude={np.abs(left).mean():.6f}')
    print(f'  RMS={np.sqrt(np.mean(left**2)):.6f}')
    
    print(f'\nRight channel:')
    print(f'  min={right.min():.6f}, max={right.max():.6f}')
    print(f'  mean amplitude={np.abs(right).mean():.6f}')
    print(f'  RMS={np.sqrt(np.mean(right**2)):.6f}')
    
    left_rms = np.sqrt(np.mean(left**2))
    right_rms = np.sqrt(np.mean(right**2))
    
    print(f'\n=== Channel Balance ===')
    print(f'Right/Left amplitude ratio: {np.abs(right).mean()/np.abs(left).mean():.2f}x')
    print(f'Right/Left RMS ratio: {right_rms/left_rms:.2f}x')
    
    if right_rms > left_rms * 2:
        print('\n⚠️  RIGHT CHANNEL IS SIGNIFICANTLY LOUDER')
        print('   This could cause detection issues if not handled properly.')
    elif left_rms > right_rms * 2:
        print('\n⚠️  LEFT CHANNEL IS SIGNIFICANTLY LOUDER')
        print('   This could cause detection issues if not handled properly.')
    else:
        print('\n✓ Channels are relatively balanced')
    
    # Sample some peaks
    print(f'\n=== Peak Analysis (first 10 seconds) ===')
    sample_len = sr * 10
    left_sample = left[:sample_len]
    right_sample = right[:sample_len]
    
    # Find peaks
    from scipy.signal import find_peaks
    left_peaks, _ = find_peaks(np.abs(left_sample), height=0.1)
    right_peaks, _ = find_peaks(np.abs(right_sample), height=0.1)
    
    print(f'Left channel peaks > 0.1: {len(left_peaks)}')
    print(f'Right channel peaks > 0.1: {len(right_peaks)}')
