# Resolution Settings

## Overview

The simulator now supports configurable world resolution settings that can be adjusted in `simulation_settings.json`. These settings control the resolution of different simulation grids.

## Settings

Add these fields to your `simulation_settings.json`:

```json
{
  ... other settings ...
  
  "env_grid_resolution": 2048,
  "fluid_grid_resolution": 512,
  "spatial_grid_resolution": 512
}
```

### Resolution Parameters

- **env_grid_resolution**: Environment grid resolution (alpha/beta/gamma chemistry and terrain)
  - Default: 2048
  - Valid range: 256 to 8192
  - This is the main simulation grid resolution
  
- **fluid_grid_resolution**: Fluid simulation grid resolution
  - Default: 512
  - Valid range: 64 to 2048
  - Should be env_grid_resolution / 4 for optimal performance
  
- **spatial_grid_resolution**: Spatial hash grid resolution (for agent collision detection)
  - Default: 512
  - Valid range: 64 to 2048
  - Should match fluid_grid_resolution

## Ratio Constraints

The system enforces a 4:1 ratio between `env_grid_resolution` and `fluid_grid_resolution`/`spatial_grid_resolution`. 

**Examples:**
- env_res = 2048 → fluid_res = 512, spatial_res = 512 (current default)
- env_res = 1024 → fluid_res = 256, spatial_res = 256
- env_res = 4096 → fluid_res = 1024, spatial_res = 1024
- env_res = 512 → fluid_res = 128, spatial_res = 128

If you manually set values that violate this ratio (ratio < 2:1 or > 16:1), the `sanitize()` function will automatically correct them to 4:1 on load.

## Applying Changes

**IMPORTANT**: Resolution changes require a **full restart** of the application to take effect, as they require:
1. Recompiling GPU shaders with new constants
2. Reallocating all GPU buffers
3. Reinitializing the simulation state

### How to Change Resolution:

1. Close the simulator completely
2. Edit `simulation_settings.json` manually
3. Set your desired `env_grid_resolution` (e.g., 1024, 2048, 4096)
4. Set `fluid_grid_resolution` to `env_grid_resolution / 4`
5. Set `spatial_grid_resolution` to the same value as `fluid_grid_resolution`
6. Save the file
7. Restart the simulator

## Performance Considerations

Higher resolutions provide more detailed simulations but require more VRAM and processing power:

| Resolution | Env Grid Memory | Fluid Memory | Approx Total VRAM |
|------------|-----------------|--------------|-------------------|
| 512x512    | 4 MB            | 1 MB         | ~150 MB           |
| 1024x1024  | 16 MB           | 4 MB         | ~300 MB           |
| 2048x2048  | 67 MB           | 17 MB        | ~800 MB           |
| 4096x4096  | 268 MB          | 67 MB        | ~2.5 GB           |

**Recommendations:**
- **Low-end GPUs** (2-4 GB VRAM): Use 512 or 1024
- **Mid-range GPUs** (4-8 GB VRAM): Use 1024 or 2048 (default)
- **High-end GPUs** (8+ GB VRAM): Use 2048 or 4096

## Current Implementation Status

✅ **Implemented:**
- Settings fields added to `SimulationSettings` struct
- Validation and ratio enforcement in `sanitize()`
- Backward compatibility with old settings files (uses defaults if missing)
- Documentation

⚠️ **Requires Restart:**
- Resolution changes currently require manual restart
- Changes take effect when shaders are compiled at startup
- GPU buffers are allocated at initialization

🔄 **Future Enhancement:**
- Runtime resolution switching with live shader recompilation
- UI controls with restart prompt
- Real-time buffer reallocation

## Technical Details

The resolution values are injected into WGSL shaders at compile time:

```rust
// In main.rs during shader compilation:
let shader_source = format!(
    "const SIM_SIZE: u32 = {}u;\n
     const ENV_GRID_SIZE: u32 = {}u;\n
     const GRID_SIZE: u32 = {}u;\n
     const SPATIAL_GRID_SIZE: u32 = {}u;\n
     const FLUID_GRID_SIZE: u32 = {}u;\n{}",
    SIM_SIZE,
    env_grid_resolution,
    env_grid_resolution,
    spatial_grid_resolution,
    fluid_grid_resolution,
    shader_code
);
```

This ensures all shader code uses the correct resolution constants consistently.

## Troubleshooting

**Q: I changed the resolution but nothing happened**
A: You must fully close and restart the application for changes to take effect.

**Q: The app crashes after changing resolution**
A: Check that your values are within valid ranges (256-8192 for env, 64-2048 for fluid/spatial).

**Q: Performance is poor after increasing resolution**
A: Higher resolutions require more GPU power. Try a lower resolution or upgrade your GPU.

**Q: My fluid/spatial resolution doesn't match my env resolution / 4**
A: The `sanitize()` function will automatically correct invalid ratios on load. Check your settings file after restarting.
