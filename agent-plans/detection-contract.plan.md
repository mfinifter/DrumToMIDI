# Detection Output Contract Plan

**Created**: 2026-01-19  
**Status**: Planning  
**Goal**: Define standard data types for drum hit detection, enabling algorithm swapping, ensemble voting, and consistent workflow tools.

---

## Problem Statement

Current detection code outputs hits in ad-hoc formats. Each tool has its own representation, making it impossible to:
- Swap detection algorithms
- Combine results from multiple algorithms
- Build reusable analysis/workflow tools
- A/B test different approaches

## Success Criteria

1. All detection algorithms output `DetectionResult` objects
2. All workflow tools consume `DetectionResult` objects
3. Existing functionality preserved (no regressions)
4. New algorithms can be added by implementing one function
5. Ensemble voting works by merging `DetectionResult` lists

---

## Proposed Types

### Core Types (add to `midi_types.py`)

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class InstrumentType(Enum):
    """Standardized instrument classification."""
    # Kicks
    KICK = "kick"
    
    # Snares
    SNARE = "snare"
    SNARE_RIMSHOT = "snare_rimshot"
    SNARE_SIDESTICK = "snare_sidestick"
    
    # Hi-hats
    HIHAT_CLOSED = "hihat_closed"
    HIHAT_OPEN = "hihat_open"
    HIHAT_FOOT = "hihat_foot"
    
    # Toms
    TOM_HIGH = "tom_high"
    TOM_MID = "tom_mid"
    TOM_LOW = "tom_low"
    TOM_FLOOR = "tom_floor"
    
    # Cymbals
    CRASH = "crash"
    RIDE = "ride"
    RIDE_BELL = "ride_bell"
    CHINA = "china"
    SPLASH = "splash"
    
    # Other
    CLAP = "clap"
    UNKNOWN = "unknown"


@dataclass
class SpectralFeatures:
    """Raw spectral analysis data for a single hit."""
    geomean: float | None = None
    fundamental_energy: float | None = None
    body_energy: float | None = None
    attack_energy: float | None = None
    brightness: float | None = None  # High freq ratio
    
    # Frequency band energies (for debugging/analysis)
    band_energies: dict[str, float] = field(default_factory=dict)


@dataclass
class DetectedHit:
    """Single detected drum hit with metadata."""
    time_seconds: float
    instrument: InstrumentType
    confidence: float  # 0.0-1.0
    velocity: int      # MIDI 1-127
    
    # Algorithm attribution
    algorithm: str | None = None
    
    # Analysis metadata (optional, for debugging/workflow)
    spectral: SpectralFeatures | None = None
    stereo_position: float | None = None  # -1.0 (L) to 1.0 (R), 0.0 = center
    
    # Pattern context (filled in by pattern analysis, not detection)
    beat_position: float | None = None  # Position within beat (0.0-1.0)
    pattern_role: str | None = None     # "downbeat", "backbeat", "ghost", "fill"
    
    def to_midi_note(self) -> int:
        """Map instrument to standard MIDI note."""
        # GM drum map
        mapping = {
            InstrumentType.KICK: 36,
            InstrumentType.SNARE: 38,
            InstrumentType.SNARE_RIMSHOT: 40,
            InstrumentType.SNARE_SIDESTICK: 37,
            InstrumentType.HIHAT_CLOSED: 42,
            InstrumentType.HIHAT_OPEN: 46,
            InstrumentType.HIHAT_FOOT: 44,
            InstrumentType.TOM_HIGH: 50,
            InstrumentType.TOM_MID: 47,
            InstrumentType.TOM_LOW: 45,
            InstrumentType.TOM_FLOOR: 41,
            InstrumentType.CRASH: 49,
            InstrumentType.RIDE: 51,
            InstrumentType.RIDE_BELL: 53,
            InstrumentType.CHINA: 52,
            InstrumentType.SPLASH: 55,
            InstrumentType.CLAP: 39,
            InstrumentType.UNKNOWN: 38,  # Default to snare
        }
        return mapping.get(self.instrument, 38)


@dataclass
class ThresholdReport:
    """Transparency data about threshold decisions."""
    threshold_used: float
    threshold_type: str  # "geomean", "amplitude", "spectral_flux", etc.
    
    # Distribution data
    total_onsets: int
    passed_threshold: int
    
    # Histogram buckets for visualization
    histogram: dict[str, int] = field(default_factory=dict)
    # e.g., {"0-50": 100, "50-100": 45, "100-150": 30, ...}
    
    # Suggestions
    suggested_threshold: float | None = None
    percentile_90: float | None = None
    percentile_75: float | None = None


@dataclass
class DetectionResult:
    """Complete output from any detection algorithm."""
    # Core data
    hits: list[DetectedHit]
    stem_type: str  # "snare", "kick", "toms", "hihat", "cymbals"
    
    # Audio metadata
    audio_path: str
    audio_duration_seconds: float
    sample_rate: int
    channels: int  # 1=mono, 2=stereo
    
    # Algorithm metadata
    algorithm_name: str
    algorithm_version: str
    parameters: dict = field(default_factory=dict)
    
    # Transparency data (optional but recommended)
    all_onsets: list[DetectedHit] | None = None  # Pre-threshold
    threshold_report: ThresholdReport | None = None
    
    # Timing
    processing_time_seconds: float | None = None
    
    def high_confidence_hits(self, threshold: float = 0.9) -> list[DetectedHit]:
        """Return only high-confidence hits."""
        return [h for h in self.hits if h.confidence >= threshold]
    
    def hits_in_range(self, start: float, end: float) -> list[DetectedHit]:
        """Return hits within a time range."""
        return [h for h in self.hits if start <= h.time_seconds <= end]
    
    def to_midi_events(self) -> list[tuple[float, int, int]]:
        """Convert to (time, note, velocity) tuples for MIDI export."""
        return [(h.time_seconds, h.to_midi_note(), h.velocity) for h in self.hits]
```

---

## Detector Interface

All detection algorithms implement this interface:

```python
from abc import ABC, abstractmethod
from pathlib import Path

class DrumDetector(ABC):
    """Base class for all drum detection algorithms."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Algorithm name for attribution."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Algorithm version."""
        pass
    
    @abstractmethod
    def detect(
        self,
        audio_path: Path,
        stem_type: str,
        parameters: dict | None = None,
    ) -> DetectionResult:
        """
        Run detection on audio file.
        
        Args:
            audio_path: Path to audio file (wav, flac, etc.)
            stem_type: Type of stem ("snare", "kick", etc.)
            parameters: Algorithm-specific parameters (thresholds, etc.)
        
        Returns:
            DetectionResult with all detected hits
        """
        pass
    
    def detect_stereo(
        self,
        audio_path: Path,
        stem_type: str,
        parameters: dict | None = None,
    ) -> tuple[DetectionResult, DetectionResult, DetectionResult]:
        """
        Run detection on L, R, and mono separately.
        
        Default implementation processes same file 3 times with channel selection.
        Override for more efficient implementations.
        
        Returns:
            (left_result, right_result, mono_result)
        """
        # Default: call detect() 3 times with channel parameter
        left = self.detect(audio_path, stem_type, {**(parameters or {}), 'channel': 'left'})
        right = self.detect(audio_path, stem_type, {**(parameters or {}), 'channel': 'right'})
        mono = self.detect(audio_path, stem_type, {**(parameters or {}), 'channel': 'mono'})
        return left, right, mono
```

---

## Implementation Phases

### Phase 1: Define Types
- [ ] Add types to `midi_types.py`
- [ ] Add unit tests for type serialization
- [ ] Document type contracts

### Phase 2: Refactor Current Detection  
- [ ] Audit current detection code (what functions, what outputs?)
- [ ] Create `LibrosaDetector` implementing `DrumDetector`
- [ ] Wrap existing logic, output `DetectionResult`
- [ ] Verify no regressions (existing tests pass)

### Phase 3: Update Consumers
- [ ] Find all code that consumes detection output
- [ ] Update to consume `DetectionResult`
- [ ] Update MIDI export to use `to_midi_events()`

### Phase 4: Enable Extensions
- [ ] Add `SuperfluxDetector` (quick win, same library)
- [ ] Add ensemble voting function
- [ ] Add stereo analysis function
- [ ] Add pattern recognition module

---

## Ensemble Voting

With standard types, ensemble voting becomes straightforward:

```python
def ensemble_detect(
    detectors: list[DrumDetector],
    audio_path: Path,
    stem_type: str,
    time_tolerance: float = 0.025,  # 25ms
) -> DetectionResult:
    """
    Run multiple detectors and merge results.
    
    Confidence = (agreeing_algorithms / total_algorithms) * avg_confidence
    """
    results = [d.detect(audio_path, stem_type) for d in detectors]
    
    # Collect all hits with algorithm attribution
    all_hits: list[DetectedHit] = []
    for result in results:
        for hit in result.hits:
            hit.algorithm = result.algorithm_name
            all_hits.append(hit)
    
    # Cluster hits by time (within tolerance)
    clusters = cluster_by_time(all_hits, tolerance=time_tolerance)
    
    # For each cluster, create merged hit
    merged_hits = []
    for cluster in clusters:
        algorithms_agreeing = len(set(h.algorithm for h in cluster))
        avg_confidence = sum(h.confidence for h in cluster) / len(cluster)
        avg_velocity = int(sum(h.velocity for h in cluster) / len(cluster))
        
        merged = DetectedHit(
            time_seconds=sum(h.time_seconds for h in cluster) / len(cluster),
            instrument=most_common_instrument(cluster),
            confidence=(algorithms_agreeing / len(detectors)) * avg_confidence,
            velocity=avg_velocity,
            algorithm=f"ensemble({algorithms_agreeing}/{len(detectors)})",
        )
        merged_hits.append(merged)
    
    return DetectionResult(
        hits=merged_hits,
        stem_type=stem_type,
        algorithm_name="ensemble",
        algorithm_version="1.0",
        # ... other fields
    )
```

---

## Risks

1. **Refactoring scope**: Current detection code may be intertwined with other concerns
2. **Performance**: Adding metadata increases memory usage
3. **Breaking changes**: Downstream code expects current format

## Mitigations

1. Start with audit to understand current structure
2. Make spectral features optional (only populated when needed)
3. Implement adapter pattern - new types wrap old outputs initially

---

## Next Steps

1. Audit current detection code: `stems_to_midi.py`, `midi_core.py`, `midi_shell.py`
2. Identify all output formats currently used
3. Design migration path from current → contract types
4. Implement Phase 1 (types only, no behavior change)
