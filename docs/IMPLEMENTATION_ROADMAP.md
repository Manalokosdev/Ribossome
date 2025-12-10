# ALsimulatorv3 - Implementation Roadmap

## Current State Analysis

### ✅ COMPLETED FEATURES
1. **GPU Architecture** - wgpu compute pipelines, WGSL shaders
2. **Agent Structure** - 16×u32 genome, 32×BodyPart array, position/velocity/rotation/energy
3. **20 Amino Acid Alphabet** - Full properties with colors, sensitivities, multipliers
4. **Genome Translation** - Codons → amino acid types → body parts
5. **Signal Propagation** - Alpha/beta signals with weighted neighbor multipliers (sum=1.0), fade 0.95, endpoint decay 0.3
6. **Morphology System** - Angles = base_angle + alpha×Sa + beta×Sb, constant lengths
7. **Physics** - Velocity integration, drag, toroidal wrapping (5120×5120 world)
8. **Rendering** - Toroidal rendering with wrapped positions, thick body segments, debug overlay
9. **Environment Grids** - Alpha/beta fields (512×512), sensor sampling via grid_index()
10. **Special Amino Acids**:
    - S (Serine) = Alpha sensor (green)
    - C (Cysteine) = Beta sensor (red)
    - P (Proline) = Propeller (blue)
    - M (Methionine) = Mouth (yellow)
    - W (Tryptophan) = Storage (orange)
11. **Debug Overlay** - Per-segment signal visualization (toggleable)
12. **Camera System** - Pan, zoom, center-on-CoM
13. **Visual Buffer Stride** - 256-byte aligned for wgpu compliance

### 🔨 PARTIALLY IMPLEMENTED (needs work)
1. **Energy System** - Energy field exists but no consumption/gain logic
2. **Mouth Mechanics** - Properties defined but no absorption implementation
3. **Propeller Physics** - Thrust arrows drawn but force application unclear
4. **Environment Interaction** - Static alpha/beta grids, no deposition
5. **Diffusion** - diffuse_pipeline exists but unused (dead code warning)

### ❌ MISSING CORE FEATURES

## Priority 1: Make Energy System Functional

### 1.1 Energy Consumption (GPU)
**Location**: `shader.wgsl` - `process_agents()` kernel

**Reference**: sensor-test app shows per-amino-acid energy_consumption values

**Task**:
- Sum energy_consumption from all body parts
- Deduct total from agent.energy each frame
- Set alive=0 when energy <= 0.0

**Implementation**:
```wgsl
// After morphology rebuild, before physics
var total_energy_consumption = 0.0;
for (var i = 0u; i < body_count; i++) {
    let props = get_amino_acid_properties(agents_out[agent_id].body[i].part_type);
    total_energy_consumption += props.energy_consumption;
}
agent.energy -= total_energy_consumption * params.dt;

if (agent.energy <= 0.0) {
    agent.alive = 0u;
    agents_out[agent_id] = agent;
    return;
}
```

### 1.2 Mouth Energy Absorption (GPU)
**Location**: `shader.wgsl` - `process_agents()` kernel

**Reference**: Amino acid M (Methionine) has:
- `is_mouth = true`
- `energy_absorption_rate = 0.2`
- `beta_absorption_rate = 0.2`
- `beta_damage = 1.0`

**Task**:
- For each mouth part, sample alpha_grid at world position
- Gain energy = alpha_sample × energy_absorption_rate × dt
- Take beta damage = beta_sample × beta_damage × dt

**Implementation**:
```wgsl
// During body part iteration
if (props.is_mouth) {
    let world_pos = agent.position + rotate_vec2(part.pos, agent.rotation);
    let grid_idx = grid_index(world_pos);
    let alpha_val = alpha_grid[grid_idx];
    let beta_val = beta_grid[grid_idx];
    
    agent.energy += alpha_val * props.energy_absorption_rate * params.dt;
    agent.energy -= beta_val * props.beta_damage * params.dt;
}
```

### 1.3 Storage Capacity (GPU)
**Location**: `shader.wgsl` - `process_agents()` kernel

**Reference**: Each amino acid has `energy_storage` property (5.0 to 12.0)

**Task**:
- Sum energy_storage from all body parts
- Clamp agent.energy to max_storage

**Implementation**:
```wgsl
var max_energy = 0.0;
for (var i = 0u; i < body_count; i++) {
    let props = get_amino_acid_properties(agents_out[agent_id].body[i].part_type);
    max_energy += props.energy_storage;
}
agent.energy = min(agent.energy, max_energy);
```

## Priority 2: Environment Interaction

### 2.1 Agent Deposition to Environment (GPU)
**Location**: New kernel `agent_deposit` or extend existing

**Reference**: LIFE_GPU_PORT_PROMPT mentions "agents deposit to alpha/beta grids based on body parts"

**Task**:
- Each body part deposits to alpha or beta grid at its world position
- Use atomic operations or separate deposit pass
- Determine deposition rule (even/odd type? sensor-based?)

**Questions for User**:
- Which amino acids deposit to alpha vs beta?
- Deposition amount per body part?
- Should sensors deposit or only non-sensors?

### 2.2 Environment Diffusion (GPU)
**Location**: `shader.wgsl` - `diffuse_alpha` and `diffuse_beta` kernels (already exist!)

**Current State**: diffuse_pipeline created but never dispatched

**Task**:
- Dispatch diffuse passes each frame in update()
- Implement 3×3 blur + 99% decay as per README
- Ensure double-buffering for grids

**Implementation** (src/main.rs):
```rust
// In update() after agent processing
encoder.dispatch_workgroups(
    (512 + 15) / 16,
    (512 + 15) / 16,
    1,
);
```

### 2.3 Dynamic Environment Generation
**Location**: CPU initialization or GPU kernel

**Reference**: LIFE_GPU_PORT_PROMPT mentions Perlin noise generation

**Task**:
- Generate alpha/beta patterns (food/poison zones)
- Optional: regenerate over time or from agent activity

**Questions for User**:
- Static or dynamic food sources?
- Perlin noise patterns or random blobs?

## Priority 3: Reproduction System

### 3.1 Reproduction Trigger (CPU-side)
**Location**: `src/main.rs` - after GPU readback

**Reference**: LIFE_GPU_PORT_PROMPT Section 6 - "spawn offspring when energy > threshold"

**Task**:
- Read agent buffer from GPU
- Check which agents have energy > reproduction_threshold (e.g., 80% of max_storage)
- For each reproducing agent:
  - Split energy between parent and offspring
  - Mutate genome
  - Append new agent to buffer
  - Recreate buffers if needed (×2 growth strategy)

**Implementation Sketch**:
```rust
fn check_reproduction(&mut self) {
    // Read agents buffer (async or staging)
    for agent in agents {
        let max_energy = calculate_max_storage(&agent);
        if agent.energy > max_energy * 0.8 && agent.alive == 1 {
            let offspring = mutate_agent(&agent);
            self.spawn_agent(offspring);
            agent.energy = max_energy * 0.4; // Split energy
        }
    }
}
```

### 3.2 Genome Mutation (CPU)
**Location**: `src/main.rs` - helper function

**Reference**: LIFE_GPU_PORT_PROMPT - deterministic mutation

**Task**:
- Copy parent genome
- Flip random bits (low mutation rate, e.g., 1-3 bits)
- Ensure valid codons (0-19 for amino acids, 255 for STOP)

**Implementation**:
```rust
fn mutate_genome(genome: &[u32; 16], rng: &mut RngState) -> [u32; 16] {
    let mut new_genome = *genome;
    let mutation_count = rng.gen_range(1..=3);
    for _ in 0..mutation_count {
        let byte_idx = rng.gen_range(0..64);
        let word = byte_idx / 4;
        let bit = rng.gen_range(0..8) + (byte_idx % 4) * 8;
        new_genome[word] ^= 1 << bit;
    }
    new_genome
}
```

### 3.3 Agent Spawning (CPU)
**Location**: `src/main.rs` - buffer management

**Reference**: LIFE_GPU_PORT_PROMPT - "append-only with ×2 headroom"

**Task**:
- Maintain current_agent_count vs buffer_capacity
- When spawning, append to agents array
- If agents.len() >= capacity:
  - Finish GPU work
  - Allocate new buffer (×2 size)
  - Copy data
  - Recreate bind groups
  - Update params.agent_count

**Implementation**:
```rust
fn spawn_agent(&mut self, agent: Agent) {
    self.agents_cpu.push(agent);
    if self.agents_cpu.len() >= self.agent_buffer_capacity {
        self.reallocate_agent_buffer();
    }
    // Upload to GPU
    self.queue.write_buffer(&self.agent_buffer_in, 0, bytemuck::cast_slice(&self.agents_cpu));
}
```

### 3.4 Dead Agent Cleanup (CPU)
**Location**: `src/main.rs` - compaction pass

**Reference**: LIFE_GPU_PORT_PROMPT - "swap-remove compaction"

**Task**:
- Periodically (every N frames) read alive flags
- Remove dead agents (swap-remove to maintain compaction)
- Update params.agent_count

**Implementation**:
```rust
fn compact_dead_agents(&mut self) {
    self.agents_cpu.retain(|a| a.alive == 1);
    self.queue.write_buffer(&self.agent_buffer_in, 0, bytemuck::cast_slice(&self.agents_cpu));
    self.sim_params.agent_count = self.agents_cpu.len() as u32;
}
```

## Priority 4: Physics Refinement

### 4.1 Propeller Force Application (GPU)
**Location**: `shader.wgsl` - `process_agents()` physics section

**Current State**: Thrust arrows drawn visually but no force applied

**Task**:
- For each propeller part (P - Proline), apply tangential thrust
- Force direction: perpendicular to position (tangent)
- Magnitude: thrust_force property (5.0 for Proline)

**Implementation**:
```wgsl
// During body part iteration
if (props.is_propeller) {
    let rotated_pos = rotate_vec2(part.pos, agent.rotation);
    let tangent = vec2<f32>(-rotated_pos.y, rotated_pos.x);
    let tangent_normalized = normalize(tangent);
    
    let thrust = tangent_normalized * props.thrust_force;
    agent.velocity += thrust * params.dt;
}
```

### 4.2 Signal-Driven Angular Velocity (GPU)
**Location**: `shader.wgsl` - physics or separate kernel

**Reference**: Morphology changes angles, but rotation is static

**Task** (optional enhancement):
- Asymmetric propeller thrust creates torque
- Or: sum of (alpha - beta) signals drives rotation

**Questions for User**:
- Should body shape changes affect rotation?
- Pure physics or signal-driven steering?

## Priority 5: UI/UX Improvements

### 5.1 Statistics Display (egui)
**Location**: `src/main.rs` - egui panel

**Task**:
- Show: alive_count, avg_energy, avg_body_length, generation
- Read from GPU counter or CPU tracking

### 5.2 Simulation Controls (egui)
**Location**: `src/main.rs` - egui panel

**Task**:
- Pause/resume toggle
- Speed slider (dt multiplier)
- Manual spawn button
- Reset simulation

### 5.3 Agent Inspection (egui)
**Location**: `src/main.rs` - click-to-select

**Task**:
- Click on agent → show genome, energy, body composition
- Highlight selected agent in render

### 5.4 Performance Monitoring (egui)
**Location**: `src/main.rs` - timing

**Task**:
- Frame time graph
- GPU dispatch time
- Agent count over time

## Priority 6: Code Cleanup

### 6.1 Remove Dead Code
**Files**: `src/main.rs`

**Issues**:
- Unused `diffuse_pipeline` (should be used!)
- Unused `read_counter` method
- Deprecated winit API calls

### 6.2 Enable Diffusion
**Location**: `src/main.rs` - update()

**Task**:
- Actually dispatch diffuse_pipeline each frame
- Check that it's hooked up correctly

### 6.3 std430 Alignment Validation
**Location**: All structs

**Task**:
- Verify Agent, BodyPart, SimParams match WGSL exactly
- Add static_assertions or runtime checks

## Implementation Order (Step-by-Step)

### Phase 1: Energy Works (Weeks 1-2)
1. ✅ Implement energy consumption
2. ✅ Implement mouth absorption
3. ✅ Implement storage capacity
4. ✅ Test: agents die when energy runs out
5. ✅ Test: agents with mouths gain energy in alpha zones

### Phase 2: Environment Lives (Week 3)
1. ✅ Enable diffusion pipeline
2. ✅ Implement agent deposition
3. ✅ Generate initial alpha/beta patterns
4. ✅ Test: environment changes over time

### Phase 3: Reproduction Begins (Week 4)
1. ✅ Implement genome mutation
2. ✅ Implement reproduction trigger
3. ✅ Implement agent spawning
4. ✅ Test: population grows when food abundant
5. ✅ Implement dead agent cleanup
6. ✅ Test: population stabilizes

### Phase 4: Physics Polish (Week 5)
1. ✅ Implement propeller thrust
2. ✅ Tune drag/energy parameters
3. ✅ Test: agents move believably

### Phase 5: UI/Debug (Week 6)
1. ✅ Add statistics panel
2. ✅ Add simulation controls
3. ✅ Add agent inspection
4. ✅ Performance tuning

### Phase 6: Optimization (Week 7+)
1. ✅ Buffer growth strategy
2. ✅ Compact dead agents
3. ✅ Workgroup size tuning
4. ✅ Scale to 10k+ agents

## Questions for User Before Starting

1. **Energy System**:
   - What should be the reproduction threshold? (80% of max_storage?)
   - How much energy should parent/offspring get? (50/50 split?)

2. **Environment**:
   - Which amino acids deposit to alpha vs beta?
   - Static food zones or dynamic regeneration?
   - Should environment be visible in rendering?

3. **Reproduction**:
   - Mutation rate? (1-3 bits per genome?)
   - Maximum population cap?
   - Asexual only or sexual reproduction later?

4. **Physics**:
   - Should rotation be signal-driven or physics-only?
   - Current drag value (0.95) good?

5. **Priorities**:
   - Which phase should we start with?
   - Any features to skip or defer?

## Reference Files

- **Sensor Test App** (`main - Copy.rs`): Signal propagation logic reference
- **LIFE_GPU_PORT_PROMPT.txt**: Architecture spec, genome translation, buffer management
- **Current shader.wgsl**: Lines 115-600 = amino acid properties, 732-1141 = morphology/physics/rendering

## Next Action

**Awaiting user input**: Which phase should we tackle first?

Recommended starting point: **Phase 1 (Energy System)** since it's foundational for reproduction.
