#!/usr/bin/env python3
"""
Generate MIDI file for cymbals using NEW energy-based detection.

One-time script to validate new transient peak detection by creating a MIDI
file that can be overlaid in DAW to visually verify timing accuracy.
"""

import librosa
import numpy as np
from pathlib import Path
from stems_to_midi.energy_detection_core import detect_stereo_transient_peaks
from stems_to_midi.config import load_config
from stems_to_midi.midi import create_midi_file


def generate_cymbal_midi_with_new_detection(
    cymbal_audio_path: str,
    output_midi_path: str,
    threshold_db: float = 15.0,
    default_pitch: int = 49,  # Crash cymbal 1 (GM)
    default_velocity: int = 100,
):
    """
    Generate MIDI file for cymbals using energy-based detection.
    
    Args:
        cymbal_audio_path: Path to cymbal audio stem
        output_midi_path: Path to save MIDI file
        threshold_db: Prominence threshold in dB (calibrated to 15.0)
        default_pitch: MIDI note number for cymbals
        default_velocity: MIDI velocity for all hits
    """
    print(f"Loading audio: {cymbal_audio_path}")
    audio, sr = librosa.load(cymbal_audio_path, sr=None, mono=False)
    
    # Ensure stereo
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.shape[0] != 2:
        audio = audio[:2]
    
    print(f"Detecting transient peaks (NEW METHOD)...")
    print(f"  - threshold_db: {threshold_db}")
    print(f"  - min_absolute_energy: 0.01")
    print(f"  - min_peak_spacing_ms: 100.0")
    
    onset_data = detect_stereo_transient_peaks(
        audio, sr,
        threshold_db=threshold_db,
        min_peak_spacing_ms=100.0,
        merge_window_ms=150.0,
        method='rms',
        min_absolute_energy=0.01,
    )
    
    # onset_data is a dict with arrays: onset_times, left_energies, right_energies, pan_confidence
    num_events = len(onset_data['onset_times'])
    print(f"Detected {num_events} events")
    
    # Convert onset data to MIDI events
    config = load_config()
    midi_events = []
    
    for i in range(num_events):
        event = {
            'time': onset_data['onset_times'][i],
            'note': default_pitch,
            'velocity': default_velocity,
            'duration': 0.1,  # Short duration for cymbal hits
        }
        midi_events.append(event)
    
    # Create MIDI file
    print(f"Writing MIDI: {output_midi_path}")
    events_by_stem = {'cymbal': midi_events}
    create_midi_file(
        events_by_stem,
        output_midi_path,
        tempo=120.0,
        track_name="Cymbals (New Detection)",
        config=config
    )
    
    print(f"\nDone! Generated MIDI with {len(midi_events)} notes")
    print(f"Import {output_midi_path} into your DAW to verify timing")
    
    # Print first few events for verification
    print("\nFirst 5 events:")
    for i in range(min(5, num_events)):
        print(f"  {i+1}. Time: {onset_data['onset_times'][i]:.3f}s, "
              f"Pan: {onset_data['pan_confidence'][i]:.2f}, "
              f"L_Energy: {onset_data['left_energies'][i]:.3f}, "
              f"R_Energy: {onset_data['right_energies'][i]:.3f}")


if __name__ == '__main__':
    import sys
    
    # Default to user_files example
    default_input = "user_files/example_stems/thunderstruck_cymbals.wav"
    default_output = "user_files/example_stems/thunderstruck_cymbals_NEW_DETECTION.mid"
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else input_path.replace('.wav', '_NEW_DETECTION.mid')
    else:
        input_path = default_input
        output_path = default_output
    
    if not Path(input_path).exists():
        print(f"Error: Input file not found: {input_path}")
        print(f"\nUsage: {sys.argv[0]} <cymbal_audio.wav> [output.mid]")
        sys.exit(1)
    
    generate_cymbal_midi_with_new_detection(input_path, output_path)
