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

**Resolution changes now take effect on restart.** The process has been implemented:

### How to Change Resolution:

1. Close the simulator (if running)
2. Edit `simulation_settings.json`
3. Set your desired `env_grid_resolution` (e.g., 1024, 2048, 4096)
4. Set `fluid_grid_resolution` to `env_grid_resolution / 4`
5. Set `spatial_grid_resolution` to the same value as `fluid_grid_resolution`
6. Save the file
7. **Restart the simulator** - the new resolution will be loaded automatically

The application now:
- ✅ Loads settings before GPU initialization
- ✅ Injects resolution values into shaders at compile time
- ✅ Properly validates and enforces ratio constraints
- ✅ Uses settings values for all grid allocations

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

✅ **Fully Implemented:**
- Settings fields with validation and ratio enforcement
- Settings loaded before GPU initialization
- Resolution values injected into shaders at startup
- All buffers allocated with correct sizes
- Backward compatibility with old settings files
- Changes take effect on application restart
- Documentation

## Technical Details

The resolution values are loaded from settings and injected into WGSL shaders at compile time during app startup:

```rust
// In main.rs during initialization:
let loaded_settings = SimulationSettings::load_from_disk(&settings_path)?;
let env_grid_res = loaded_settings.env_grid_resolution;
let fluid_grid_res = loaded_settings.fluid_grid_resolution;
let spatial_grid_res = loaded_settings.spatial_grid_resolution;

// Passed to GpuState creation
let state = GpuState::new_with_resources(
    window, instance, surface, adapter, device, queue,
    env_grid_res, fluid_grid_res, spatial_grid_res
).await;

// During shader compilation:
let shader_source = format!(
    "const ENV_GRID_SIZE: u32 = {}u;\n
     const FLUID_GRID_SIZE: u32 = {}u;\n
     const SPATIAL_GRID_SIZE: u32 = {}u;\n{}",
    env_grid_res, fluid_grid_res, spatial_grid_res, shader_code
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
