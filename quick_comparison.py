#!/usr/bin/env python
"""Quick detection method comparison for all stems"""
import yaml
import subprocess
import shutil
from pathlib import Path
from mido import MidiFile

PROJECT = "14"
STEMS = ["kick", "snare", "toms", "hihat", "cymbals"]
CONFIG_PATH = Path("user_files/14 - AC_DC_Thunderstruck_Drums/midiconfig.yaml")
BACKUP_PATH = CONFIG_PATH.with_suffix(".yaml.backup")

results = {}

# Backup original config
shutil.copy(CONFIG_PATH, BACKUP_PATH)

try:
    for stem in STEMS:
        print(f"\n{'='*50}")
        print(f"Testing {stem}...")
        print('='*50)
        
        results[stem] = {}
        
        for mode in ["energy", "librosa"]:
            print(f"\n--- {stem} ({mode} detection) ---")
            
            # Load config
            with open(CONFIG_PATH, 'r') as f:
                config = yaml.safe_load(f)
            
            # Set detection mode explicitly
            if stem not in config:
                config[stem] = {}
            
            if mode == "librosa":
                config[stem]['use_librosa_detection'] = True
            else:
                # Explicitly disable for energy mode
                config[stem]['use_librosa_detection'] = False
            
            with open(CONFIG_PATH, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            # Run conversion
            cmd = f"python stems_to_midi_cli.py {PROJECT} --stems {stem}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            # Check which detection was used
            if "Using energy-based detection" in result.stdout:
                detection_used = "energy"
            elif "Using librosa onset detection" in result.stdout:
                detection_used = "librosa"
            else:
                detection_used = "UNKNOWN"
            
            # Find and count MIDI file
            midi_dir = Path("user_files/14 - AC_DC_Thunderstruck_Drums/midi")
            midi_files = list(midi_dir.glob("*.mid"))
            
            if midi_files:
                midi_file = midi_files[0]  # Should only be one
                mid = MidiFile(midi_file)
                count = sum(1 for track in mid.tracks for msg in track 
                          if msg.type == 'note_on' and msg.velocity > 0)
                
                # Rename with mode suffix
                new_name = midi_file.stem + f"_{stem}_{mode}.mid"
                new_path = midi_dir / new_name
                shutil.move(midi_file, new_path)
                
                results[stem][mode] = {
                    'count': count,
                    'detection': detection_used,
                    'file': new_name
                }
                print(f"Created: {count} events (detection={detection_used})")
            else:
                results[stem][mode] = {'count': 0, 'detection': 'ERROR', 'file': None}
                print("ERROR: No MIDI file created")
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Stem':<10} {'Energy':<15} {'Librosa':<15} {'Difference':<12}")
    print("-"*60)
    
    for stem in STEMS:
        e_count = results[stem].get('energy', {}).get('count', 0)
        l_count = results[stem].get('librosa', {}).get('count', 0)
        diff = l_count - e_count
        diff_str = f"{diff:+d}" if diff != 0 else "same"
        
        e_detection = results[stem].get('energy', {}).get('detection', '?')
        l_detection = results[stem].get('librosa', {}).get('detection', '?')
        
        print(f"{stem:<10} {e_count:>3} ({e_detection:<7}) {l_count:>3} ({l_detection:<7}) {diff_str:<12}")
    
    print("="*60)

finally:
    # Restore original config
    shutil.copy(BACKUP_PATH, CONFIG_PATH)
    BACKUP_PATH.unlink()
    print("\nOriginal config restored.")
