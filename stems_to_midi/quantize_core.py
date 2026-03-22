"""
Quantize Core - Groove-Aware Quantization

Pure functions for snapping MIDI beat times to rhythmic grids.
No side effects, no I/O - only calculations.

Groove hints let users specify what subdivisions appear in a song
(e.g., triplet_8 for shuffle/blues feel). Detected onset times are
then snapped to the nearest valid grid position.
"""

from typing import Dict, List

# Grid positions within one beat (quarter note = 1.0) for each groove type.
# Each list contains the valid fractional positions in [0, 1).
GROOVE_GRIDS: Dict[str, List[float]] = {
    'quarter':      [0.0],
    'straight_8':   [0.0, 0.5],
    'straight_16':  [0.0, 0.25, 0.5, 0.75],
    'triplet_8':    [0.0, 1/3, 2/3],
    'triplet_16':   [0.0, 1/6, 1/3, 0.5, 2/3, 5/6],
    'swing':        [0.0, 1/3, 2/3],
}

VALID_GROOVE_HINTS = set(GROOVE_GRIDS.keys())


def build_quantize_grid(groove_hints: List[str]) -> List[float]:
    """
    Build a merged quantization grid from one or more groove hints.

    Combines all grid positions from the specified groove types,
    removes duplicates, and returns sorted positions in [0, 1).

    Args:
        groove_hints: List of groove type names (e.g., ['triplet_8', 'straight_16'])

    Returns:
        Sorted list of unique grid positions within one beat.
        Empty list if no valid hints provided.

    Raises:
        ValueError: If any groove hint is not recognized.
    """
    if not groove_hints:
        return []

    positions = set()
    for hint in groove_hints:
        if hint not in GROOVE_GRIDS:
            raise ValueError(
                f"Unknown groove hint '{hint}'. "
                f"Valid options: {sorted(VALID_GROOVE_HINTS)}"
            )
        positions.update(GROOVE_GRIDS[hint])

    return sorted(positions)


def quantize_beat_time(time_beats: float, grid: List[float], strength: float = 1.0) -> float:
    """
    Snap a single beat time to the nearest grid position.

    Decomposes time into whole beats + fractional part, finds the nearest
    grid position for the fractional part, then interpolates based on strength.

    Args:
        time_beats: Time in beats (e.g., 3.667 = beat 3 + 2/3)
        grid: Sorted list of grid positions within one beat [0, 1)
        strength: Quantize strength (0.0 = no change, 1.0 = full snap)

    Returns:
        Quantized time in beats.
    """
    if not grid or strength == 0.0:
        return time_beats

    whole = int(time_beats)
    frac = time_beats - whole

    # Find nearest grid position, considering wraparound to next beat.
    # E.g., frac=0.95 with grid=[0, 0.333, 0.667] should snap to 1.0 (next beat).
    best_dist = float('inf')
    best_pos = frac  # fallback

    for pos in grid:
        dist = abs(frac - pos)
        if dist < best_dist:
            best_dist = dist
            best_pos = pos

    # Also check wrapping: distance to 0.0 of next beat (i.e., grid[0] + 1.0)
    wrap_dist = abs(frac - 1.0)  # distance to next downbeat
    if wrap_dist < best_dist:
        best_pos = 1.0

    # Apply strength interpolation
    quantized_frac = frac + strength * (best_pos - frac)

    # Handle wraparound: if quantized_frac rounds up to next beat
    if quantized_frac >= 1.0:
        whole += 1
        quantized_frac -= 1.0

    return whole + quantized_frac


def quantize_events(
    prepared_events: List[dict],
    groove_hints: List[str],
    strength: float = 1.0
) -> List[dict]:
    """
    Quantize all events' time_beats to the groove grid.

    Args:
        prepared_events: List of event dicts with 'time_beats' key
        groove_hints: List of groove type names
        strength: Quantize strength (0.0 = no change, 1.0 = full snap)

    Returns:
        New list of event dicts with quantized time_beats.
        Original list is not modified.
    """
    if not groove_hints or strength == 0.0:
        return prepared_events

    grid = build_quantize_grid(groove_hints)
    if not grid:
        return prepared_events

    result = []
    for event in prepared_events:
        quantized = dict(event)
        quantized['time_beats'] = quantize_beat_time(
            event['time_beats'], grid, strength
        )
        result.append(quantized)

    return result
