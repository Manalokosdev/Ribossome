# Refactoring Summary: Amino Acids to Organs

## Overview
Converted special features from amino acid-based to organ-based architecture, maintaining the principle that all special abilities must come from 2-codon organ sequences (promoter + modifier).

## Changes Made

### 1. Amino Acids Converted to Structural Only

**F (Phenylalanine) - Base Type 4**
- Before: Vampire mouth (red, thickness 8.0, is_mouth=true)
- After: Structural amino acid (brown, thickness 4.5, no special properties)

**G (Glycine) - Base Type 5**
- Before: Vampire mouth (red, thickness 8.0, is_mouth=true)
- After: Structural amino acid (white/grey, thickness 3.0, no special properties)

**H (Histidine) - Base Type 6**
- Before: Vampire mouth (red, thickness 8.0, is_mouth=true)
- After: Structural amino acid (light blue, thickness 4.0, no special properties)

**L (Leucine) - Base Type 9**
- Before: Agent alpha sensor (dark magenta, thickness 10.0, is_agent_alpha_sensor=true)
- After: Structural amino acid (yellow-brown, thickness 4.5, no special properties)

**Y (Tyrosine) - Base Type 19**
- Before: Agent beta sensor (grey, thickness 9.0, is_agent_beta_sensor=true)
- After: Structural amino acid (olive green, thickness 4.0, no special properties)

### 2. New Organ Types Added

**Organ Type 33: VMPRE (Vampire Mouth)**
- Created by: K/C promoter + F/G/H modifier
- Properties: Red color, thickness 8.0, energy draining capability
- Drains energy from nearby living agents (distance-based, 0-40 units)
- Activity controlled by enabler organs (inverted logic - active when no enablers)
- Rendering: Large red 8-point asterisk
- Inspector symbol: 'V'
- Inspector name: VMPRE (5 chars)

**Organ Type 34: AGSNA (Agent Alpha Sensor)**
- Created by: V/M promoter + L modifier
- Properties: Dark magenta color, thickness 10.0, directional agent detection
- Senses nearby agents via distance and dot product with sensor direction
- Contributes to alpha signal channel
- Inspector symbol: '<'
- Inspector name: AGSNA (5 chars)

**Organ Type 35: AGSNB (Agent Beta Sensor)**
- Created by: V/M promoter + Y modifier
- Properties: Grey color, thickness 9.0, directional agent detection
- Senses nearby agents via distance and dot product with sensor direction
- Contributes to beta signal channel
- Inspector symbol: '>'
- Inspector name: AGSNB (5 chars)

### 3. Genome Translation Logic Updated

**K/C Promoter (lines 2113-2120)**
- Added check for F/G/H modifiers (4u/5u/6u) → creates organ type 33
- Existing mouth creation (modifiers 0-6) preserved
- Existing enabler, slope sensor, and clock creation preserved

**V/M Promoter (lines 2121-2125)**
- Added check for L modifier (9u) → creates organ type 34
- Added check for Y modifier (19u) → creates organ type 35
- Existing alpha/beta/energy sensor creation preserved

### 4. Functional Code Updated

**drain_energy Kernel (lines 2552-2636)**
- Changed vampire mouth check from `base_type == 4u || 5u || 6u`
- To: `base_type == 33u`
- All vampire mouth draining logic now operates on organ type 33

**process_agents Kernel - Agent Sensors (lines 3118-3177)**
- Agent alpha sensor: Changed from `amino_props.is_agent_alpha_sensor`
- To: `base_type == 34u`
- Agent beta sensor: Changed from `amino_props.is_agent_beta_sensor`
- To: `base_type == 35u`

**Rendering (lines 2408-2421)**
- Regular mouths (organ 20): Small yellow asterisk
- Vampire mouths (organ 33): Large red 8-point asterisk (separate rendering)
- Agent sensors: Use standard organ rendering (perpendicular sensor visualization)

### 5. Inspector and Display Updates

**Inspector Names (lines 1290-1293)**
- Added case 33u: VMPRE (Vampire Mouth)
- Added case 34u: AGSNA (Agent Alpha Sensor)
- Added case 35u: AGSNB (Agent Beta Sensor)

**Genome Display Symbols (lines 6536-6538)**
- Organ 33: 'V' (ASCII 86)
- Organ 34: '<' (ASCII 60)
- Organ 35: '>' (ASCII 62)

## Testing Requirements

1. **Genome Creation**: Verify that K/C + F/G/H codons create organ type 33
2. **Genome Creation**: Verify that V/M + L creates organ type 34
3. **Genome Creation**: Verify that V/M + Y creates organ type 35
4. **Vampire Draining**: Test that agents with organ 33 drain nearby agents
5. **Agent Sensing**: Test that organs 34/35 detect nearby agents and affect alpha/beta signals
6. **F/G/H Amino Acids**: Verify they are structural only (no draining, normal colors)
7. **L/Y Amino Acids**: Verify they are structural only (no agent sensing, normal colors)
8. **Inspector Display**: Check that organ names show as VMPRE/AGSNA/AGSNB
9. **Genome Viewer**: Check that symbols V/</>appear for the new organs
10. **Rendering**: Verify vampire mouths show large red asterisks

## Example Genomes

**Vampire Mouth Examples**:
- `AAA AAA AAA AAA AAA` + `CUU` (K) + `UUC` (F) = Vampire mouth organ (KC+F)
- `AAA AAA AAA AAA AAA` + `UGC` (C) + `GGU` (G) = Vampire mouth organ (CC+G)
- `AAA AAA AAA AAA AAA` + `AAA` (K) + `CAU` (H) = Vampire mouth organ (KK+H)

**Agent Alpha Sensor Examples**:
- `AAA AAA AAA AAA AAA` + `GUG` (V) + `CUG` (L) = Agent alpha sensor (VV+L)
- `AAA AAA AAA AAA AAA` + `AUG` (M) + `UUA` (L) = Agent alpha sensor (MM+L)

**Agent Beta Sensor Examples**:
- `AAA AAA AAA AAA AAA` + `GUA` (V) + `UAU` (Y) = Agent beta sensor (VV+Y)
- `AAA AAA AAA AAA AAA` + `AUG` (M) + `UAC` (Y) = Agent beta sensor (MM+Y)

## Architecture Compliance

✓ All special features are now organs (2-codon sequences)
✓ Single amino acids are purely structural
✓ Promoter system is consistent (K/C for mouths, V/M for sensors)
✓ Modifier system allows variation (F/G/H for vampire variants, L/Y for sensor types)
✓ Existing organ architecture preserved and extended

## Backward Compatibility

⚠️ **Breaking Change**: Existing genomes with single F/G/H/L/Y amino acids will lose their special properties. To maintain functionality, genomes must be updated to use the new 2-codon organ sequences (promoter + modifier).

Example conversion:
- Old: `...UUU...` (F alone = vampire mouth)
- New: `...AAA UUU...` (K+F = vampire mouth organ)

## Files Modified

1. `shader.wgsl`:
   - Amino acid properties (F, G, H, L, Y)
   - Organ type definitions (cases 33u, 34u, 35u)
   - Genome translation logic (promoter + modifier combinations)
   - drain_energy kernel (vampire mouth check)
   - process_agents kernel (agent sensor checks)
   - Rendering logic (vampire mouth asterisk)
   - Inspector names
   - Genome display symbols

## No Rust Changes Required

All changes were shader-only. The Rust application code did not require modification as the organ system infrastructure (encoding, decoding, buffers, pipelines) already existed and supports the new organ types seamlessly.
