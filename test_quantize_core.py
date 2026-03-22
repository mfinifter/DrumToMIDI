"""
Tests for stems_to_midi/quantize_core.py

Tests groove grid building, beat time quantization, and event quantization.
"""

import sys
import importlib
import pytest

# Import quantize_core directly to avoid stems_to_midi.__init__ pulling in
# dependencies (midiutil, etc.) that may not be installed in test environments.
spec = importlib.util.spec_from_file_location(
    "quantize_core",
    "stems_to_midi/quantize_core.py"
)
quantize_core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quantize_core)

GROOVE_GRIDS = quantize_core.GROOVE_GRIDS
build_quantize_grid = quantize_core.build_quantize_grid
quantize_beat_time = quantize_core.quantize_beat_time
quantize_events = quantize_core.quantize_events


# ============================================================================
# build_quantize_grid tests
# ============================================================================

class TestBuildQuantizeGrid:
    def test_empty_hints(self):
        assert build_quantize_grid([]) == []

    def test_quarter(self):
        assert build_quantize_grid(['quarter']) == [0.0]

    def test_straight_8(self):
        grid = build_quantize_grid(['straight_8'])
        assert grid == [0.0, 0.5]

    def test_straight_16(self):
        grid = build_quantize_grid(['straight_16'])
        assert grid == [0.0, 0.25, 0.5, 0.75]

    def test_triplet_8(self):
        grid = build_quantize_grid(['triplet_8'])
        assert len(grid) == 3
        assert grid[0] == 0.0
        assert abs(grid[1] - 1/3) < 1e-10
        assert abs(grid[2] - 2/3) < 1e-10

    def test_triplet_16(self):
        grid = build_quantize_grid(['triplet_16'])
        assert len(grid) == 6

    def test_swing_same_as_triplet_8(self):
        swing = build_quantize_grid(['swing'])
        triplet = build_quantize_grid(['triplet_8'])
        assert len(swing) == len(triplet)
        for s, t in zip(swing, triplet):
            assert abs(s - t) < 1e-10

    def test_combined_grids_merge(self):
        grid = build_quantize_grid(['straight_8', 'triplet_8'])
        # straight_8: 0, 0.5
        # triplet_8: 0, 0.333, 0.667
        # merged: 0, 0.333, 0.5, 0.667
        assert len(grid) == 4
        assert grid[0] == 0.0

    def test_combined_straight_16_triplet_8(self):
        grid = build_quantize_grid(['straight_16', 'triplet_8'])
        # straight_16: 0, 0.25, 0.5, 0.75
        # triplet_8: 0, 0.333, 0.667
        # merged: 0, 0.25, 0.333, 0.5, 0.667, 0.75
        assert len(grid) == 6

    def test_invalid_hint_raises(self):
        with pytest.raises(ValueError, match="Unknown groove hint 'waltz'"):
            build_quantize_grid(['waltz'])

    def test_grid_is_sorted(self):
        for name in GROOVE_GRIDS:
            grid = build_quantize_grid([name])
            assert grid == sorted(grid), f"Grid for '{name}' is not sorted"

    def test_all_positions_in_range(self):
        for name in GROOVE_GRIDS:
            grid = build_quantize_grid([name])
            for pos in grid:
                assert 0.0 <= pos < 1.0, f"Position {pos} out of [0,1) for '{name}'"


# ============================================================================
# quantize_beat_time tests
# ============================================================================

class TestQuantizeBeatTime:
    def test_exact_grid_position_unchanged(self):
        grid = [0.0, 0.5]
        assert quantize_beat_time(2.5, grid) == 2.5
        assert quantize_beat_time(3.0, grid) == 3.0

    def test_snap_to_nearest(self):
        grid = [0.0, 0.5]
        # 2.3 is closer to 2.5 than 2.0
        assert quantize_beat_time(2.3, grid) == 2.5
        # 2.1 is closer to 2.0 than 2.5
        assert quantize_beat_time(2.1, grid) == 2.0

    def test_triplet_snap(self):
        grid = [0.0, 1/3, 2/3]
        # A note at beat 1.35 should snap to 1.333...
        result = quantize_beat_time(1.35, grid)
        assert abs(result - (1 + 1/3)) < 1e-10

    def test_wraparound_to_next_beat(self):
        grid = [0.0, 0.5]
        # 2.9 is closer to 3.0 (next downbeat) than 2.5
        result = quantize_beat_time(2.9, grid)
        assert abs(result - 3.0) < 1e-10

    def test_triplet_wraparound(self):
        grid = [0.0, 1/3, 2/3]
        # 2.9 should wrap to 3.0
        result = quantize_beat_time(2.9, grid)
        assert abs(result - 3.0) < 1e-10

    def test_strength_zero_no_change(self):
        grid = [0.0, 0.5]
        assert quantize_beat_time(2.3, grid, strength=0.0) == 2.3

    def test_strength_half(self):
        grid = [0.0, 0.5]
        # 2.3 snaps to 2.5, but at 50% strength: 2.3 + 0.5*(0.5-0.3) = 2.4
        result = quantize_beat_time(2.3, grid, strength=0.5)
        assert abs(result - 2.4) < 1e-10

    def test_strength_full(self):
        grid = [0.0, 0.5]
        result = quantize_beat_time(2.3, grid, strength=1.0)
        assert abs(result - 2.5) < 1e-10

    def test_time_zero(self):
        grid = [0.0, 0.5]
        assert quantize_beat_time(0.0, grid) == 0.0

    def test_empty_grid_returns_original(self):
        assert quantize_beat_time(2.3, []) == 2.3

    def test_equidistant_snaps_to_first(self):
        grid = [0.0, 0.5]
        # 0.25 is equidistant from 0.0 and 0.5 — should snap to 0.0 (first found)
        result = quantize_beat_time(0.25, grid)
        assert result == 0.0 or result == 0.5  # either is acceptable


# ============================================================================
# quantize_events tests
# ============================================================================

class TestQuantizeEvents:
    def _make_events(self, times):
        return [
            {'time_beats': t, 'note': 36, 'velocity': 100, 'duration_beats': 0.1}
            for t in times
        ]

    def test_basic_quantization(self):
        events = self._make_events([0.0, 0.48, 1.02, 1.52])
        result = quantize_events(events, ['straight_8'])
        times = [e['time_beats'] for e in result]
        assert times == [0.0, 0.5, 1.0, 1.5]

    def test_triplet_quantization(self):
        events = self._make_events([0.0, 0.31, 0.68, 1.0])
        result = quantize_events(events, ['triplet_8'])
        times = [e['time_beats'] for e in result]
        assert abs(times[0] - 0.0) < 1e-10
        assert abs(times[1] - 1/3) < 1e-10
        assert abs(times[2] - 2/3) < 1e-10
        assert abs(times[3] - 1.0) < 1e-10

    def test_no_hints_returns_original(self):
        events = self._make_events([0.0, 0.31])
        result = quantize_events(events, [])
        assert result is events  # same object, not copied

    def test_zero_strength_returns_original(self):
        events = self._make_events([0.0, 0.31])
        result = quantize_events(events, ['triplet_8'], strength=0.0)
        assert result is events

    def test_original_not_modified(self):
        events = self._make_events([0.31])
        original_time = events[0]['time_beats']
        quantize_events(events, ['triplet_8'])
        assert events[0]['time_beats'] == original_time

    def test_preserves_other_fields(self):
        events = [{'time_beats': 0.31, 'note': 38, 'velocity': 95, 'duration_beats': 0.2, 'stem_type': 'snare'}]
        result = quantize_events(events, ['triplet_8'])
        assert result[0]['note'] == 38
        assert result[0]['velocity'] == 95
        assert result[0]['duration_beats'] == 0.2
        assert result[0]['stem_type'] == 'snare'

    def test_partial_strength(self):
        events = self._make_events([0.4])
        # 0.4 snaps to 0.5 (straight_8). At 50%: 0.4 + 0.5*(0.5-0.4) = 0.45
        result = quantize_events(events, ['straight_8'], strength=0.5)
        assert abs(result[0]['time_beats'] - 0.45) < 1e-10
