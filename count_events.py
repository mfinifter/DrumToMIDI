import sys
from mido import MidiFile

midi_path = sys.argv[1]
mid = MidiFile(midi_path)
note_ons = sum(1 for track in mid.tracks for msg in track if msg.type == 'note_on' and msg.velocity > 0)
print(note_ons)
