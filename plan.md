# Plan: Groove Hints — User-Guided Quantization

## Problem

DrumToMIDI currently preserves sample-accurate onset times with **no quantization**. When the detected times are converted to MIDI beats (via `prepare_midi_events_for_writing`), triplet-feel or shuffled grooves end up with complex, non-standard beat positions (e.g., 1.3847 instead of a clean 1.3333 for a triplet). This makes the MIDI unreadable in a DAW and doesn't capture the musical intent.

## Solution Overview

Add an optional **groove hint** system where the user specifies what rhythmic subdivisions appear in the song. The system then quantizes detected onset times to the nearest position on the appropriate grid, *after* detection but *before* MIDI file creation.

This is a post-detection quantization step — it doesn't change how onsets are detected, only how they're mapped to musical time.

## Design Decisions

1. **Quantization happens in beat-space** (after `seconds_to_beats`, before writing). This is simpler and musically correct — the grid is defined in terms of beat subdivisions.

2. **Multiple groove types can coexist** — a song might have straight 8th-note hi-hats with triplet-feel snare/kick. The hint applies globally but allows all specified subdivisions.

3. **Quantize strength is configurable** — users can set how aggressively notes snap to the grid (100% = hard quantize, 50% = move halfway, 0% = no change).

4. **The feature is purely additive and opt-in** — no quantization by default, preserving current behavior.

## Groove Types and Their Grids

Each groove type defines a set of valid beat positions within one beat (quarter note = 1.0 beat):

| Groove Hint | Subdivisions per beat | Grid positions (within 1 beat) |
|---|---|---|
| `straight_8` | 2 | 0, 0.5 |
| `straight_16` | 4 | 0, 0.25, 0.5, 0.75 |
| `triplet_8` | 3 | 0, 0.3333, 0.6667 |
| `triplet_16` | 6 | 0, 0.1667, 0.3333, 0.5, 0.6667, 0.8333 |
| `swing` | 3 (biased) | 0, 0.3333, 0.6667 (same as triplet_8 — swing = triplet feel on 8ths) |
| `quarter` | 1 | 0 |

Multiple hints can be combined. E.g., `groove_hints: [straight_16, triplet_8]` produces a merged grid with all positions from both, so notes snap to the *nearest* valid position from any specified groove.

## Implementation Steps

### Step 1: Add quantization core function — `quantize_core.py` (new file)

Create `stems_to_midi/quantize_core.py` — a pure functional module with:

```python
def build_quantize_grid(groove_hints: list[str]) -> list[float]:
    """Return sorted list of unique grid positions within one beat [0, 1)."""

def quantize_beat_time(time_beats: float, grid: list[float]) -> float:
    """Snap a beat time to the nearest grid position."""

def quantize_events(
    prepared_events: list[dict],
    groove_hints: list[str],
    strength: float = 1.0
) -> list[dict]:
    """Quantize all events' time_beats. strength=1.0 is full snap, 0.0 is no-op."""
```

The quantize logic:
- Decompose `time_beats` into `whole_beat + fractional`
- Find nearest grid position to `fractional`
- Apply strength: `new_frac = fractional + strength * (nearest_grid - fractional)`
- Return `whole_beat + new_frac`

### Step 2: Wire into MIDI creation pipeline

In `stems_to_midi/midi.py` → `create_midi_file()`:
- After `prepare_midi_events_for_writing()` (line 82), call `quantize_events()` if groove hints are present in config.
- Pass groove hints and strength from the config dict.

### Step 3: Add config support

In `midiconfig.yaml`, add a new top-level section:

```yaml
# === QUANTIZATION (Groove Hints) ===
quantize:
  enabled: false                    # Set to true to enable quantization
  groove_hints: []                  # List: straight_8, straight_16, triplet_8, triplet_16, swing, quarter
  strength: 1.0                     # 0.0 (no snap) to 1.0 (hard quantize)
```

### Step 4: Add CLI argument

In `stems_to_midi_cli.py`, add:
- `--groove` flag accepting one or more groove hint names
- `--quantize-strength` float argument (0.0-1.0, default 1.0)

When `--groove` is specified, it overrides the config file's `quantize` section and sets `enabled: true`.

### Step 5: Add tests

Create `test_quantize_core.py` with tests for:
- Each groove type produces correct grid
- Quantization snaps to nearest grid position correctly
- Combined grooves merge grids properly
- Strength parameter interpolates correctly (0%, 50%, 100%)
- Edge cases: time at exactly 0, time at beat boundary, time equidistant between two grid positions
- No-op when groove_hints is empty or enabled is false

## File Changes Summary

| File | Change |
|---|---|
| `stems_to_midi/quantize_core.py` | **NEW** — pure quantization functions |
| `stems_to_midi/midi.py` | Wire quantization into `create_midi_file()` |
| `midiconfig.yaml` | Add `quantize:` section |
| `stems_to_midi_cli.py` | Add `--groove` and `--quantize-strength` CLI args |
| `test_quantize_core.py` | **NEW** — unit tests |

## Example Usage

```bash
# Triplet-feel song (e.g., 12/8 blues, shuffle)
python stems_to_midi_cli.py 1 --groove triplet_8

# Straight 16th-note song
python stems_to_midi_cli.py 1 --groove straight_16

# Song with both straight and triplet sections (quantize to nearest of either)
python stems_to_midi_cli.py 1 --groove straight_16 triplet_8

# Soft quantize (move 75% toward grid)
python stems_to_midi_cli.py 1 --groove triplet_8 --quantize-strength 0.75
```

Or via `midiconfig.yaml`:
```yaml
quantize:
  enabled: true
  groove_hints: [triplet_8]
  strength: 1.0
```
