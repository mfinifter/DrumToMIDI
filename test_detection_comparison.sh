#!/bin/bash
# Test script to compare energy-based vs librosa detection for each stem
# Processes project 14 (AC/DC Thunderstruck) with each stem individually
# Runs in both detection modes and compares outputs

set -e  # Exit on error

PROJECT=14
STEMS=("kick" "snare" "toms" "hihat" "cymbals")
CONFIG="user_files/14 - AC_DC_Thunderstruck_Drums/midiconfig.yaml"
CONFIG_BACKUP="${CONFIG}.backup"
RESULTS_FILE="detection_comparison_results.txt"

echo "========================================" > "$RESULTS_FILE"
echo "Detection Method Comparison" >> "$RESULTS_FILE"
echo "Project: $PROJECT (AC/DC Thunderstruck)" >> "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "========================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Backup original config
echo "Backing up config..."
cp "$CONFIG" "$CONFIG_BACKUP"

# Function to count MIDI events in a file
count_midi_events() {
    local midi_file="$1"
    if [ -f "$midi_file" ]; then
        # Use Python to count note events
        conda run -n drumtomidi python -c "
from midiutil import MIDIFile
import midiutil.MidiFile as mf
import struct

try:
    midi = mf.MIDIFile()
    with open('$midi_file', 'rb') as f:
        midi.read(f)
    
    # Count note_on events (ignore note_off)
    note_count = 0
    for track in midi.tracks:
        for event in track.eventList:
            if event.type == 'note_on':
                note_count += 1
    print(note_count)
except Exception as e:
    print('ERROR')
" 2>/dev/null
    else
        echo "0"
    fi
}

# Function to run detection for a stem with a specific mode
run_detection() {
    local stem="$1"
    local mode="$2"  # "energy" or "librosa"
    
    echo ""
    echo "========================================" | tee -a "$RESULTS_FILE"
    echo "Testing: $stem ($mode detection)" | tee -a "$RESULTS_FILE"
    echo "========================================" | tee -a "$RESULTS_FILE"
    
    # Modify config based on mode
    if [ "$mode" == "librosa" ]; then
        # Add use_librosa_detection: true to this stem's config
        conda run -n drumtomidi python -c "
import yaml
import sys

stem = '${stem}'
config_path = '${CONFIG}'

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

if stem not in config:
    config[stem] = {}
config[stem]['use_librosa_detection'] = True

with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
"
    else
        # Restore original config (energy-based is default)
        cp "$CONFIG_BACKUP" "$CONFIG"
    fi
    
    # Run stems_to_midi_cli.py for this stem only
    echo "Running conversion..." | tee -a "$RESULTS_FILE"
    conda run -n drumtomidi python stems_to_midi_cli.py $PROJECT --stems $stem 2>&1 | tee -a "$RESULTS_FILE"
    
    # Find the generated MIDI file (project name only, no stem suffix in filename)
    MIDI_DIR="user_files/14*/midi"
    MIDI_FILE=$(find $MIDI_DIR -name "*.mid" -type f 2>/dev/null | head -1)
    
    if [ -f "$MIDI_FILE" ]; then
        # Count events
        EVENT_COUNT=$(count_midi_events "$MIDI_FILE")
        
        # Rename MIDI file to include stem and mode
        MIDI_DIR_PATH=$(dirname "$MIDI_FILE")
        MIDI_BASENAME=$(basename "$MIDI_FILE" .mid)
        NEW_NAME="${MIDI_DIR_PATH}/${MIDI_BASENAME}_${stem}_${mode}.mid"
        mv "$MIDI_FILE" "$NEW_NAME"
        
        echo "" | tee -a "$RESULTS_FILE"
        echo "Result: $EVENT_COUNT MIDI events" | tee -a "$RESULTS_FILE"
        echo "Saved as: $(basename $NEW_NAME)" | tee -a "$RESULTS_FILE"
    else
        echo "WARNING: MIDI file not found in $MIDI_DIR" | tee -a "$RESULTS_FILE"
    fi
    
    # Restore original config for next run
    cp "$CONFIG_BACKUP" "$CONFIG"
}

# Main execution
echo "Starting detection comparison test..."
echo ""

for stem in "${STEMS[@]}"; do
    # Run with energy-based detection (default/new method)
    run_detection "$stem" "energy"
    
    # Run with librosa detection (legacy/old method)
    run_detection "$stem" "librosa"
    
    echo "" | tee -a "$RESULTS_FILE"
    echo "----------------------------------------" | tee -a "$RESULTS_FILE"
done

# Restore original config
echo ""
echo "Restoring original config..."
cp "$CONFIG_BACKUP" "$CONFIG"
rm "$CONFIG_BACKUP"

echo ""
echo "========================================" | tee -a "$RESULTS_FILE"
echo "COMPARISON COMPLETE" | tee -a "$RESULTS_FILE"
echo "========================================" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Summary comparison
echo "SUMMARY:" | tee -a "$RESULTS_FILE"
MIDI_DIR=$(find user_files/14*/midi -type d 2>/dev/null | head -1)

for stem in "${STEMS[@]}"; do
    ENERGY_FILE=$(find "$MIDI_DIR" -name "*_${stem}_energy.mid" 2>/dev/null | head -1)
    LIBROSA_FILE=$(find "$MIDI_DIR" -name "*_${stem}_librosa.mid" 2>/dev/null | head -1)
    
    ENERGY_COUNT=$(count_midi_events "$ENERGY_FILE")
    LIBROSA_COUNT=$(count_midi_events "$LIBROSA_FILE")
    
    DIFF=$((ENERGY_COUNT - LIBROSA_COUNT))
    
    printf "%-10s Energy: %4d events | Librosa: %4d events | Diff: %+4d\n" \
        "$stem" "$ENERGY_COUNT" "$LIBROSA_COUNT" "$DIFF" | tee -a "$RESULTS_FILE"
done

echo "" | tee -a "$RESULTS_FILE"
echo "Results saved to: $RESULTS_FILE" | tee -a "$RESULTS_FILE"
echo "MIDI files saved with _energy.mid and _librosa.mid suffixes"
echo ""
echo "Done!"
