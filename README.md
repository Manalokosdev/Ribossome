# GPU Artificial Life Simulator

A minimal, high-performance GPU-accelerated artificial life simulator built with Rust, wgpu, and WGSL.

## Features

### Agent Architecture
- **Agents store genomes**: 64 bytes (16 × u32 array) for genetic information
- **Agents store body parts**: Up to 32 body parts per agent
- **Agents draw themselves**: Each agent renders its own body via GPU compute shader
- Each body part has:
  - Position (relative to agent center)
  - Size (radius)
  - Sensor strength (how strongly it senses environment)
  - Color (RGB)
  - Part type (encoded amino acid)

### Environment
- **Alpha grid**: 512×512 float grid - environment field A
- **Beta grid**: 512×512 float grid - environment field B  
- **Direct modification**: Agents can modify grid values directly by index
- **Diffusion & decay**: Environment grids diffuse and decay each frame

### Energy & Maintenance
- **Tunable metabolism**: Adjust food/poison power and the per-amino maintenance cost slider in the UI to control how quickly complex organisms burn energy.

### GPU Performance
- Ping-pong buffers for agents (no CPU synchronization needed)
- All operations in compute shaders (@workgroup_size 64)
- Single WGSL file reduces shader compilation overhead
- Direct buffer-to-texture copy for visualization
- Atomic operations for alive counter

### Simulation Passes (executed each frame)
1. `update_agents` - Physics, sensing environment, energy management
2. `agent_modify_env` - Agents deposit to alpha/beta grids based on body parts
3. `diffuse_grids` - 3×3 blur + 99% decay on environment grids
4. `clear_visual` - Clear visual buffer
5. `draw_agents` - Each agent draws its body parts and connections
6. Render pass - Display visual grid to screen

## Running

```powershell
cargo run --release
```

## Files

- `Cargo.toml` - Dependencies (wgpu 22, winit 0.30, etc.)
- `shader.wgsl` - Single unified WGSL shader with all compute/render passes
- `src/main.rs` - GPU initialization, buffers, pipelines, and event loop

## Current State

- 100 agents with 3-part bodies
- Agents move around sensing alpha/beta environment grids
- Body parts colored red, green, blue
- Even-typed parts deposit to alpha, odd-typed to beta
- Energy decreases over time
- Real-time rendering at high FPS

## Next Steps

- Add reproduction system (spawn offspring when energy > threshold)
- Implement genome translation (genes → body parts)
- Add mutation system
- Dynamic agent count (grow/shrink population)
- Interactive controls (sliders for parameters)
- Performance metrics display
