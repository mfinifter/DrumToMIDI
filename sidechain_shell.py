"""
Sidechain compression to reduce bleed between stems - Imperative Shell

Uses the separated snare track as a sidechain trigger to duck the kick track
when the snare is playing, effectively removing snare bleed from the kick.

Uses project-based workflow: automatically detects projects with stems
and creates cleaned versions in the project/cleaned/ directory.

Architecture: Imperative shell (this file) using functional core (sidechain_core.py)

Usage:
    python sidechain_cleanup.py              # Auto-detect project
    python sidechain_cleanup.py 1            # Process specific project
"""

from pathlib import Path
import numpy as np  # type: ignore
import soundfile as sf  # type: ignore
import argparse
import sys
from typing import Union

# Import functional core
from sidechain_core import (
    envelope_follower as _envelope_follower,
    sidechain_compress as _sidechain_compress
)

# Import project manager
from project_manager import (
    select_project
)


def envelope_follower(audio: np.ndarray, sr: int, attack_ms: float = 5.0, release_ms: float = 50.0) -> np.ndarray:
    """
    Create an envelope follower for the audio signal.
    
    Wrapper around functional core with no added logging (core is already pure).
    
    Args:
        audio: Input audio (mono or stereo)
        sr: Sample rate
        attack_ms: Attack time in milliseconds
        release_ms: Release time in milliseconds
    
    Returns:
        Envelope of the audio signal
    """
    return _envelope_follower(audio, sr, attack_ms, release_ms)


def sidechain_compress(
    main_audio: np.ndarray,
    sidechain_audio: np.ndarray,
    sr: int,
    threshold_db: float = -30.0,
    ratio: float = 10.0,
    attack_ms: float = 1.0,
    release_ms: float = 100.0,
    makeup_gain_db: float = 0.0,
    knee_db: float = 3.0
) -> np.ndarray:
    """
    Apply sidechain compression to main audio based on sidechain audio.
    
    Wrapper around functional core with progress reporting (imperative shell).
    
    Args:
        main_audio: Audio to be compressed (kick track)
        sidechain_audio: Audio that triggers compression (snare track)
        sr: Sample rate
        threshold_db: Threshold in dB below which no compression occurs
        ratio: Compression ratio (higher = more aggressive ducking)
        attack_ms: How quickly compression kicks in when snare hits
        release_ms: How quickly compression releases after snare stops
        makeup_gain_db: Gain to apply after compression
        knee_db: Soft knee width in dB
    
    Returns:
        Compressed audio
    """
    print(f"Applying sidechain compression with threshold={threshold_db}dB, ratio={ratio}:1")
    
    # Call functional core (no side effects)
    compressed, stats = _sidechain_compress(
        main_audio,
        sidechain_audio,
        sr,
        threshold_db,
        ratio,
        attack_ms,
        release_ms,
        makeup_gain_db,
        knee_db
    )
    
    # Report results (imperative shell responsibility)
    print("Compression applied:")
    print(f"  - Max gain reduction: {stats['max_gain_reduction_db']:.1f} dB")
    print(f"  - Compressed {stats['compression_percentage']:.1f}% of samples")
    
    return compressed


def process_stems(
    stems_dir: Union[str, Path],
    output_dir: Union[str, Path],
    threshold_db: float = -30.0,
    ratio: float = 10.0,
    attack_ms: float = 1.0,
    release_ms: float = 100.0,
    dry_wet: float = 1.0,
    clean_cymbals: bool = True
):
    """
    Process separated stems to remove bleed using sidechain compression.
    
    Removes snare bleed from kick track.
    Optionally removes hihat bleed from cymbals track.
    
    Args:
        stems_dir: Directory containing separated stems (flat structure: trackname-kick.wav, etc.)
        output_dir: Directory to save cleaned stems
        threshold_db: Sidechain threshold in dB
        ratio: Compression ratio (higher = more aggressive)
        attack_ms: Attack time in milliseconds
        release_ms: Release time in milliseconds
        dry_wet: Mix between original (0.0) and processed (1.0)
        clean_cymbals: If True, also remove hihat bleed from cymbals track
    """
    stems_dir = Path(stems_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all kick files in the stems directory (flat structure)
    kick_files = list(stems_dir.glob("*-kick.wav"))
    
    if not kick_files:
        raise RuntimeError(f"No kick files found in {stems_dir}")
    
    # Find tracks that have stems to clean
    tracks_to_process = []
    for kick_file in kick_files:
        # Extract base name (e.g., "trackname-kick.wav" -> "trackname")
        base_name = kick_file.stem.replace("-kick", "")
        snare_file = stems_dir / f"{base_name}-snare.wav"
        cymbals_file = stems_dir / f"{base_name}-cymbals.wav"
        hihat_file = stems_dir / f"{base_name}-hihat.wav"
        
        has_kick_snare = snare_file.exists()
        has_cymbals_hihat = clean_cymbals and cymbals_file.exists() and hihat_file.exists()
        
        if has_kick_snare or has_cymbals_hihat:
            tracks_to_process.append({
                'base_name': base_name,
                'kick_file': kick_file if has_kick_snare else None,
                'snare_file': snare_file if has_kick_snare else None,
                'cymbals_file': cymbals_file if has_cymbals_hihat else None,
                'hihat_file': hihat_file if has_cymbals_hihat else None
            })
        else:
            print(f"Warning: Skipping {base_name} - missing required stem pairs")
    
    if not tracks_to_process:
        raise RuntimeError(f"No tracks found with required stem pairs in {stems_dir}")
    
    print(f"Processing {len(tracks_to_process)} track(s)...")
    print("Settings:")
    print(f"  Threshold: {threshold_db} dB")
    print(f"  Ratio: {ratio}:1")
    print(f"  Attack: {attack_ms} ms")
    print(f"  Release: {release_ms} ms")
    print(f"  Dry/Wet: {dry_wet * 100:.0f}% processed")
    print(f"  Clean cymbals: {'Yes' if clean_cymbals else 'No'}")
    print()
    
    for track_idx, track_info in enumerate(tracks_to_process, 1):
        base_name = track_info['base_name']
        print(f"Processing: {base_name}")
        
        sr = None
        stems_processed = []
        
        # Process kick-snare if both exist
        if track_info['kick_file'] and track_info['snare_file']:
            print("  Removing snare bleed from kick track...")
            
            # Load audio files
            kick_audio, sr = sf.read(str(track_info['kick_file']))
            snare_audio, sr_snare = sf.read(str(track_info['snare_file']))
            
            if sr != sr_snare:
                print(f"  Warning: Sample rate mismatch! Kick: {sr}Hz, Snare: {sr_snare}Hz")
            else:
                # Ensure same length
                min_length = min(len(kick_audio), len(snare_audio))
                kick_audio = kick_audio[:min_length]
                snare_audio = snare_audio[:min_length]
                
                # Apply sidechain compression
                kick_compressed = sidechain_compress(
                    kick_audio,
                    snare_audio,
                    sr,
                    threshold_db=threshold_db,
                    ratio=ratio,
                    attack_ms=attack_ms,
                    release_ms=release_ms
                )
                
                # Dry/wet mix
                kick_final = dry_wet * kick_compressed + (1 - dry_wet) * kick_audio
                
                # Save cleaned kick track
                output_file = output_dir / f'{base_name}-kick.wav'
                sf.write(str(output_file), kick_final, sr)
                print(f"  Saved cleaned kick: {output_file.name}")
                stems_processed.append('kick')
        
        # Process cymbals-hihat if both exist and cleaning enabled
        if track_info['cymbals_file'] and track_info['hihat_file']:
            print("  Removing hihat bleed from cymbals track...")
            
            # Load audio files
            cymbals_audio, sr_cymbals = sf.read(str(track_info['cymbals_file']))
            hihat_audio, sr_hihat = sf.read(str(track_info['hihat_file']))
            
            if sr is None:
                sr = sr_cymbals
            
            if sr_cymbals != sr_hihat:
                print(f"  Warning: Sample rate mismatch! Cymbals: {sr_cymbals}Hz, Hihat: {sr_hihat}Hz")
            else:
                # Ensure same length
                min_length = min(len(cymbals_audio), len(hihat_audio))
                cymbals_audio = cymbals_audio[:min_length]
                hihat_audio = hihat_audio[:min_length]
                
                # Apply sidechain compression
                cymbals_compressed = sidechain_compress(
                    cymbals_audio,
                    hihat_audio,
                    sr_cymbals,
                    threshold_db=threshold_db,
                    ratio=ratio,
                    attack_ms=attack_ms,
                    release_ms=release_ms
                )
                
                # Dry/wet mix
                cymbals_final = dry_wet * cymbals_compressed + (1 - dry_wet) * cymbals_audio
                
                # Save cleaned cymbals track
                output_file = output_dir / f'{base_name}-cymbals.wav'
                sf.write(str(output_file), cymbals_final, sr_cymbals)
                print(f"  Saved cleaned cymbals: {output_file.name}")
                stems_processed.append('cymbals')
        
        print("Status Update: Saving Files...")
        print("Progress: 0%")
        
        # Copy other stems unchanged
        stem_counter = 0
        for stem_name in ['snare', 'toms', 'hihat', 'cymbals', 'kick']:
            # Skip if we already processed this stem
            if stem_name in stems_processed:
                continue
                
            stem_file = stems_dir / f"{base_name}-{stem_name}.wav"
            if stem_file.exists():
                # Copy unchanged
                stem_audio, stem_sr = sf.read(str(stem_file))
                output_file = output_dir / f'{base_name}-{stem_name}.wav'
                sf.write(str(output_file), stem_audio, stem_sr)
                print(f"  Copied unchanged: {output_file.name}")
            
            stem_counter += 1
            print(f"Progress: {min(100, stem_counter * 20)}%")

    print("Status Update: Process complete!")
    print(f"Stems saved to: {output_dir}")


def cleanup_project_stems(
    project_number: int = None,
    threshold_db: float = -30.0,
    ratio: float = 10.0,
    attack_ms: float = 1.0,
    release_ms: float = 100.0,
    dry_wet: float = 1.0,
    clean_cymbals: bool = True
):
    """
    Clean up stems for a project using sidechain compression.
    
    Removes snare bleed from kick track.
    Optionally removes hihat bleed from cymbals track.
    
    Args:
        project_number: Specific project to process, or None for auto-select
        threshold_db: Sidechain threshold in dB
        ratio: Compression ratio (higher = more aggressive)
        attack_ms: Attack time in milliseconds
        release_ms: Release time in milliseconds
        dry_wet: Mix between original (0.0) and processed (1.0)
        clean_cymbals: If True, also remove hihat bleed from cymbals track
    """
    # Select project
    project_info = select_project(project_number)
    if not project_info:
        print("No project available for cleanup.")
        sys.exit(1)
    
    project_folder = project_info['path']
    song_name = project_info['name']
    
    print(f"\n{'='*60}")
    print(f"Sidechain Cleanup: {song_name}")
    print(f"Project: {project_folder.name}")
    print(f"{'='*60}\n")
    
    # Check for stems directory
    stems_dir = project_folder / 'stems'
    if not stems_dir.exists():
        print("❌ No stems directory found in project.")
        print(f"   Expected: {stems_dir}")
        print("   Run separate.py first to generate stems.")
        sys.exit(1)
    
    # Create cleaned directory
    cleaned_dir = project_folder / 'cleaned'
    cleaned_dir.mkdir(exist_ok=True)
    
    # Process the stems
    process_stems(
        stems_dir=stems_dir,
        output_dir=cleaned_dir,
        threshold_db=threshold_db,
        ratio=ratio,
        attack_ms=attack_ms,
        release_ms=release_ms,
        dry_wet=dry_wet,
        clean_cymbals=clean_cymbals
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Remove bleed between stems using sidechain compression.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process most recent project with default settings (cleans kick and cymbals)
  python sidechain_cleanup.py
  
  # Process specific project
  python sidechain_cleanup.py 1
  
  # Only clean kick track (skip cymbals)
  python sidechain_cleanup.py --no-clean-cymbals
  
  # Aggressive ducking
  python sidechain_cleanup.py -t -40 -r 20 --attack 0.5
  
  # Gentle/subtle ducking
  python sidechain_cleanup.py -t -25 -r 4 --dry-wet 0.5
        """
    )
    
    parser.add_argument('project_number', type=int, nargs='?', default=None,
                        help="Project number to process (auto-selects most recent if not provided).")
    parser.add_argument('-t', '--threshold', type=float, default=-30.0,
                        help="Sidechain threshold in dB. Lower = more sensitive (default: -30).")
    parser.add_argument('-r', '--ratio', type=float, default=10.0,
                        help="Compression ratio. Higher = more aggressive ducking (default: 10).")
    parser.add_argument('--attack', type=float, default=1.0,
                        help="Attack time in milliseconds. Lower = faster response (default: 1).")
    parser.add_argument('--release', type=float, default=100.0,
                        help="Release time in milliseconds. Lower = faster recovery (default: 100).")
    parser.add_argument('--dry-wet', type=float, default=1.0,
                        help="Mix between original (0.0) and processed (1.0). Default: 1.0 (fully processed).")
    parser.add_argument('--no-clean-cymbals', action='store_true',
                        help="Skip cleaning hihat bleed from cymbals track (default: clean both kick and cymbals).")
    
    args = parser.parse_args()
    
    # Validate
    if not (0.0 <= args.dry_wet <= 1.0):
        parser.error("--dry-wet must be between 0.0 and 1.0")
    if args.ratio < 1.0:
        parser.error("--ratio must be >= 1.0")
    
    cleanup_project_stems(
        project_number=args.project_number,
        threshold_db=args.threshold,
        ratio=args.ratio,
        attack_ms=args.attack,
        release_ms=args.release,
        dry_wet=args.dry_wet,
        clean_cymbals=not args.no_clean_cymbals
    )
