# Shader Refactoring Plan: Modular Architecture

## Goal
Split the monolithic `shader.wgsl` into modular files that can be shared between the simulator and a future genome editor app.

## File Structure

```
shaders/
├── shared.wgsl           # Core structs, constants, utilities (lines 1-1700)
├── rendering.wgsl        # All rendering functions (lines 1701-5927 + 5928-6700)
├── simulation.wgsl       # Simulation kernels (lines 2160-5700)
└── editor.wgsl           # NEW: Simple editor-only shader
```

## Detailed Breakdown

### 1. `shared.wgsl` - Core Definitions
**What**: Structs, constants, and pure utility functions with no side effects
**Lines**: ~1-1700
**Contents**:
- Constants (SIM_SIZE, GRID_SIZE, MAX_BODY_PARTS, etc.)
- Structs (Agent, BodyPart, AminoAcidProperties, SimParams, etc.)
- BodyPart encoding/decoding (pack_prev_pos, get_base_part_type, etc.)
- Amino acid properties array and get_amino_acid_properties()
- Grid utilities (grid_index, is_in_bounds, clamp_position)
- Math utilities (rotate_vec2, hash, noise functions)
- Genome utilities (codon translation, genome reading, etc.)
- Sampling functions (sample_stochastic_gaussian, etc.)
- Part names (get_part_name)

**No dependencies**: This is the foundation

---

### 2. `rendering.wgsl` - Pure Rendering Functions
**What**: Functions that draw but don't modify simulation state
**Lines**: ~1701-1915 (drawing primitives) + ~5928-6700 (inspector)
**Contents**:

#### Drawing Primitives (~1701-1915):
- InspectorContext struct
- draw_line_ctx()
- draw_thick_line_ctx()
- draw_filled_circle_ctx()
- draw_circle_ctx()
- draw_asterisk_8_ctx()
- draw_asterisk_6_ctx()
- draw_particle_jet_ctx()
- draw_crosshair_ctx()

#### Body Part Rendering (~1916-2159):
- render_body_part_ctx() - **CORE FUNCTION**
  - Handles all organ types (propellers, mouths, sensors, vampire mouths, etc.)
  - Uses InspectorContext for coordinate mapping
  - Reads from agents_out or selected_agent_buffer

#### Inspector Rendering (~5928-6700):
- BarLayout struct and calculate_bar_layout()
- render_inspector() - Background and bars
- draw_inspector_agent() - Agent preview
- Font rendering functions
- Text drawing functions

**Dependencies**: `shared.wgsl` only

---

### 3. `simulation.wgsl` - Simulation Compute Kernels
**What**: Kernels that modify agent state, grids, physics
**Lines**: ~2160-5927 (excluding rendering)
**Contents**:

#### Core Simulation:
- **drain_energy** (~2160-2400) - Vampire draining
- **process_agents** (~2400-3700) - Physics, morphology, energy, feeding
- **propagate_signals** (~3700-4200) - Alpha/beta signal propagation  
- **decay_grids** (~4200-4400) - Trail/energy decay
- **composite_agents** (~5700-5900) - Visual rendering kernel

#### Environment:
- init_alpha_grid
- init_beta_grid  
- init_gamma_grid
- apply_beta_rain

#### Spatial Grid:
- clear_agent_spatial_grid
- populate_agent_spatial_grid

#### Spawning:
- spawn_agent_kernel
- clear_spawn_queue

**Dependencies**: `shared.wgsl` + `rendering.wgsl` (for composite_agents)

---

### 4. `editor.wgsl` - NEW: Editor-Only Shader
**What**: Simple shader for genome editor preview
**Contents**:

```wgsl
// Include shared definitions
{shared.wgsl content}

// Include rendering functions
{rendering.wgsl content}

// Editor-specific bindings
@group(0) @binding(0) var<storage, read> editor_agent: Agent;
@group(0) @binding(1) var<storage, read_write> output_texture: texture_storage_2d<rgba8unorm, write>;

// Simple kernel that just renders one agent
@compute @workgroup_size(16, 16)
fn render_editor_preview(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Render the agent using shared rendering functions
    // Much simpler than full simulation
}
```

**Dependencies**: `shared.wgsl` + `rendering.wgsl`

---

## Migration Strategy

### Phase 1: Create Modular Files (Manual Split)
1. Copy `shared.wgsl` sections from current shader
2. Copy `rendering.wgsl` sections  
3. Copy `simulation.wgsl` sections
4. Create `editor.wgsl` template

### Phase 2: Update Rust Code
```rust
// src/main.rs
let shared = include_str!("../shaders/shared.wgsl");
let rendering = include_str!("../shaders/rendering.wgsl");
let simulation = include_str!("../shaders/simulation.wgsl");

let full_shader = format!("{}\n{}\n{}", shared, rendering, simulation);
```

### Phase 3: Create Editor App
```rust
// editor/src/main.rs
let shared = include_str!("../shaders/shared.wgsl");
let rendering = include_str!("../shaders/rendering.wgsl");
let editor = include_str!("../shaders/editor.wgsl");

let editor_shader = format!("{}\n{}\n{}", shared, rendering, editor);
```

---

## Benefits

✅ **Code Reuse**: Editor uses exact same rendering as simulator
✅ **Maintainability**: Changes to rendering apply everywhere
✅ **Separation of Concerns**: Rendering vs. simulation clearly separated
✅ **Editor Performance**: No simulation overhead for static preview
✅ **Testing**: Can test rendering independently
✅ **Future Extensions**: Easy to create new tools (genome analyzer, statistics viewer, etc.)

---

## Key Decision: Buffer Access

**Challenge**: `render_body_part_ctx()` currently accesses both:
- `agents_out` (main simulation)
- `selected_agent_buffer` (inspector)

**Solution**: Make it generic via special agent_id flag:
```wgsl
fn render_body_part_ctx(
    part: BodyPart,
    part_index: u32,
    agent_id: u32,  // 0xFFFFFFFF = use selected_agent_buffer
    ...
) {
    let agent = select(
        agents_out[agent_id],
        selected_agent_buffer[0],
        agent_id == 0xFFFFFFFFu
    );
    // Render using agent data
}
```

This works for editor too - just pass 0xFFFFFFFF and populate selected_agent_buffer.

---

## Timeline Estimate

- Phase 1 (File Split): 2-3 hours
- Phase 2 (Rust Integration): 30 minutes  
- Phase 3 (Editor App): 4-6 hours
- Testing: 1-2 hours

**Total**: ~8-12 hours of work

---

## Next Steps

1. ✅ Review this plan
2. Create `shaders/` directory
3. Extract `shared.wgsl` content
4. Extract `rendering.wgsl` content
5. Extract `simulation.wgsl` content
6. Update Rust to concatenate files
7. Test that simulator still works
8. Create basic editor app structure
9. Implement genome editing UI

---

## Questions to Address

1. **WGSL Includes**: WGSL doesn't have native `#include`. Use Rust string concatenation?
2. **Binding Numbers**: Keep same bindings or reorganize?
3. **Editor UI Framework**: egui (same as simulator) or web-based?
4. **Genome Format**: JSON? Custom format? Compatible with simulator's save files?

---

## Editor UI Mockup

```
┌─────────────────────────────────────────────────────────────┐
│ Genome Editor                                   [Save] [Load]│
├──────────────┬──────────────────────────┬───────────────────┤
│ Amino Acids  │   Agent Preview          │  Genome Text      │
│              │                          │                   │
│ [A] Alanine  │      ┌────────┐          │  AUGAAAUUUCCC... │
│ [C] Cysteine │      │  ●●●   │          │                   │
│ [D] Aspartic │      │ ●●●●●  │          │  Body Parts: 45   │
│ [E] Glutamic │      │●●●●●●● │          │  Energy Cap: 120  │
│ ...          │      │ ●●●●●  │          │  Mass: 2.5        │
│              │      │  ●●●   │          │                   │
│ Organs       │      └────────┘          │  [Translate]      │
│ ┌──────────┐ │                          │  [Validate]       │
│ │K+F Vamp  │ │  Rotation: [===|====]    │                   │
│ │L+A Prop  │ │  Zoom:     [=====|==]    │                   │
│ └──────────┘ │                          │                   │
└──────────────┴──────────────────────────┴───────────────────┘
```

