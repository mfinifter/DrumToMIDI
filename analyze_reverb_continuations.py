#!/usr/bin/env python3
"""
Analyze reverb continuation patterns in analysis.json files.

Detects events where:
- Next event starts within 5ms of previous end
- Amplitude continuity (start amplitude matches previous end within 0.001)
- Velocity decreases (decay pattern)
"""

import json
from pathlib import Path
from typing import List, Dict
import sys


def analyze_reverb_continuations(
    events: List[Dict],
    time_margin_ms: float = 5.0,
    amplitude_margin: float = 0.001,
) -> List[Dict]:
    """
    Identify reverb continuation patterns in event sequence.
    
    Args:
        events: List of event dictionaries (must be sorted by time)
        time_margin_ms: Maximum gap between end of event[i-1] and start of event[i]
        amplitude_margin: Maximum difference between end/start amplitudes
    
    Returns:
        List of reverb continuation events with context
    """
    if len(events) < 2:
        return []
    
    time_margin_sec = time_margin_ms / 1000.0
    continuations = []
    
    for i in range(1, len(events)):
        prev = events[i - 1]
        curr = events[i]
        
        # Calculate timing
        prev_end_time = prev['time'] + prev['duration_sec']
        gap = curr['time'] - prev_end_time
        
        # Check continuation conditions
        is_adjacent = abs(gap) <= time_margin_sec
        
        # Amplitude continuity check
        prev_end_amp = prev.get('amplitude_at_end', 0)
        curr_start_amp = curr.get('amplitude_at_start', 0)
        amp_diff = abs(curr_start_amp - prev_end_amp)
        is_amplitude_continuous = amp_diff <= amplitude_margin
        
        # Amplitude decay check (use amplitude or velocity)
        prev_amp = prev.get('amplitude')
        curr_amp = curr.get('amplitude')
        prev_vel = prev.get('velocity')
        curr_vel = curr.get('velocity')
        
        if prev_amp is not None and curr_amp is not None:
            is_decaying = curr_amp < prev_amp
        elif prev_vel is not None and curr_vel is not None:
            is_decaying = curr_vel < prev_vel
        else:
            continue
        
        if is_adjacent and is_amplitude_continuous and is_decaying:
            continuations.append({
                'index': i,
                'prev_time': prev['time'],
                'prev_end_time': prev_end_time,
                'curr_time': curr['time'],
                'gap_ms': gap * 1000,
                'prev_velocity': prev.get('velocity', prev_amp * 100 if prev_amp else 0),
                'curr_velocity': curr.get('velocity', curr_amp * 100 if curr_amp else 0),
                'prev_end_amp': prev_end_amp,
                'curr_start_amp': curr_start_amp,
                'amp_diff': amp_diff,
                'prev_note': prev.get('note', '?'),
                'curr_note': curr.get('note', '?'),
                'prev_status': prev.get('status', 'UNKNOWN'),
                'curr_status': curr.get('status', 'UNKNOWN'),
            })
    
    return continuations


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze reverb continuation patterns')
    parser.add_argument('analysis_file', help='Path to analysis.json file')
    parser.add_argument('--time-margin-ms', type=float, default=5.0,
                        help='Time margin for continuation detection (ms)')
    parser.add_argument('--amplitude-margin', type=float, default=0.001,
                        help='Amplitude margin for continuity check')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed information for each continuation')
    
    args = parser.parse_args()
    
    # Load analysis file
    with open(args.analysis_file) as f:
        data = json.load(f)
    
    print(f"Analyzing: {args.analysis_file}")
    print(f"Time margin: {args.time_margin_ms}ms")
    print(f"Amplitude margin: {args.amplitude_margin}")
    print("=" * 80)
    print()
    
    # Analyze each stem
    total_events = 0
    total_continuations = 0
    
    # Handle nested structure: data['stems'][stem_type]['events']
    stems_data = data.get('stems', {})
    
    for stem_type, stem_info in stems_data.items():
        if not isinstance(stem_info, dict):
            continue
        
        events = stem_info.get('events', [])
        if not events:
            continue
        
        total_events += len(events)
        
        # Sort by time
        events_sorted = sorted(events, key=lambda e: e['time'])
        
        # Find continuations
        continuations = analyze_reverb_continuations(
            events_sorted,
            args.time_margin_ms,
            args.amplitude_margin,
        )
        
        if continuations:
            total_continuations += len(continuations)
            
            print(f"{stem_type.upper()}: {len(continuations)} reverb continuations found")
            print(f"  Total events: {len(events)}")
            print(f"  Continuation rate: {len(continuations) / len(events) * 100:.1f}%")
            
            # Note distribution  
            notes = {c['curr_note'] for c in continuations if isinstance(c['curr_note'], (int, float))}
            if notes:
                print(f"  Affected notes: {sorted(notes)}")
            else:
                print(f"  Affected notes: N/A")
            
            # Velocity ranges
            vel_drops = [c['prev_velocity'] - c['curr_velocity'] for c in continuations]
            print(f"  Velocity drops: {min(vel_drops)} to {max(vel_drops)} (avg: {sum(vel_drops)/len(vel_drops):.1f})")
            
            # Timing stats
            gaps_ms = [c['gap_ms'] for c in continuations]
            print(f"  Time gaps: {min(gaps_ms):.3f}ms to {max(gaps_ms):.3f}ms (avg: {sum(gaps_ms)/len(gaps_ms):.3f}ms)")
            
            # Status distribution
            kept_count = sum(1 for c in continuations if c['curr_status'] == 'KEPT')
            filtered_count = sum(1 for c in continuations if c['curr_status'] == 'FILTERED')
            reverb_count = sum(1 for c in continuations if c['curr_status'] == 'REVERB_CONTINUATION')
            rejected_count = sum(1 for c in continuations if c['curr_status'] == 'REJECTED')
            print(f"  Status: {kept_count} KEPT, {filtered_count} FILTERED, {reverb_count} REVERB_CONTINUATION, {rejected_count} REJECTED")
            
            if args.verbose:
                print("\n  Examples:")
                for i, c in enumerate(continuations[:3]):  # Show first 3
                    print(f"    [{i+1}] t={c['prev_time']:.3f}s → {c['curr_time']:.3f}s (gap={c['gap_ms']:.3f}ms)")
                    print(f"        vel={c['prev_velocity']} → {c['curr_velocity']} (Δ={c['prev_velocity']-c['curr_velocity']})")
                    print(f"        amp={c['prev_end_amp']:.4f} → {c['curr_start_amp']:.4f} (Δ={c['amp_diff']:.6f})")
                    print(f"        note={c['prev_note']} → {c['curr_note']}, status={c['curr_status']}")
            
            print()
    
    print("=" * 80)
    print(f"SUMMARY:")
    print(f"  Total events: {total_events}")
    print(f"  Total reverb continuations: {total_continuations}")
    if total_events > 0:
        print(f"  Overall rate: {total_continuations / total_events * 100:.1f}%")
    else:
        print(f"  Overall rate: N/A (no events found)")
    print()
    print(f"These events could be marked as 'REVERB_CONTINUATION' instead of removed,")
    print(f"preserving the data while allowing MIDI export to filter them.")


if __name__ == '__main__':
    main()
