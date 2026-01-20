# Existing Tools Audit Plan

**Created**: 2026-01-19  
**Status**: ✅ Complete  
**Goal**: Understand what audio-to-MIDI tools exist, what works, what's broken, and how data flows through them. This informs the Detection Output Contract design.

---

## Key Findings

1. **Well-structured package** - `stems_to_midi/` has clean separation of concerns
2. **Current output**: `List[Dict]` with `{time, note, velocity, duration}`
3. **Spectral data calculated but discarded** - geomeans, sustain, etc. not in output
4. **Two disconnected type systems** - detection vs rendering (midi_types.py)
5. **Optimization tools exist** - Bayesian optimization in `optimization/` subpackage
6. **Learning mode works** - `--learn` flag, `learning.py` module

See [existing-tools-audit.results.md](existing-tools-audit.results.md) for full details.

---

## Objectives

1. Map all detection/conversion code paths
2. Identify what formats/types are currently used
3. Document what's working vs broken vs unknown
4. Find the "learning mode" and analysis tools mentioned in TODO
5. Understand how user corrections flow back (if at all)

## Success Criteria

- Complete inventory of detection-related code
- Data flow diagram from audio → MIDI
- List of existing output formats
- Status of each tool (working/broken/partial)
- Recommendations for Detection Output Contract

---

## Phase 1: Code Inventory

Find all files involved in audio → MIDI conversion:

### Primary Entry Points
- [ ] `stems_to_midi.py` - CLI tool
- [ ] Web UI separation/MIDI endpoints
- [ ] Any other entry points?

### Core Modules
- [ ] `midi_core.py` - likely contains detection logic
- [ ] `midi_shell.py` - imperative shell?
- [ ] `midi_types.py` - existing type definitions
- [ ] `midi_parser.py` - MIDI reading?

### Configuration
- [ ] `midiconfig.yaml` - thresholds, settings
- [ ] Learning mode temp configs?

### Analysis/Learning Tools
- [ ] Learning mode (`--learning` flag?)
- [ ] Threshold analysis scripts
- [ ] Hit comparison tools
- [ ] Any Bayesian optimization code?

---

## Phase 2: Data Flow Analysis

Trace the path: Audio File → Detected Hits → MIDI File

Questions to answer:
- What intermediate representations exist?
- Where are thresholds applied?
- What metadata is preserved/lost?
- Where could we insert the new contract types?

---

## Phase 3: Status Assessment

For each tool/feature:
| Tool | Status | Last Known Working | Notes |
|------|--------|-------------------|-------|
| Basic detection | ? | ? | |
| Learning mode | ? | ? | |
| Spectral filtering | ? | ? | |
| Open/closed hihat | ? | ? | |
| Threshold analysis | ? | ? | |

---

## Phase 4: Contract Recommendations

Based on findings:
- What types need to be defined?
- Where should they be inserted?
- What's the migration path?

---

## Audit Log

(Findings recorded as we go)
