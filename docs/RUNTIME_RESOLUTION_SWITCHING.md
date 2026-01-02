# Runtime Resolution Switching

## Overview

ALsimulatorv3 now supports changing simulation resolution at runtime through the UI, without requiring manual code changes or app restarts. This feature allows you to adjust the grid resolution to balance visual quality and performance.

## Features

### ✅ UI Resolution Selector
- **Three preset options**: 2048×2048, 1024×1024, 512×512
- **One-click switching**: Select a resolution and the simulation resets automatically
- **Visual feedback**: Current resolution displayed, pending changes highlighted

### ✅ Resolution-Aware Snapshots
- **Metadata storage**: Snapshots now store their resolution (version 1.2)
- **Cross-resolution loading**: Load snapshots saved at different resolutions
- **Warning system**: Notifies when loading from different resolution
- **Backward compatible**: Still loads old snapshots without resolution data

## How to Use

### Changing Resolution via UI

1. Open the simulation
2. Look for the **"Resolution"** section in the control panel
3. Current resolution is displayed (e.g., "Current: 2048×2048")
4. Click one of the three preset buttons:
   - **2048×2048** - High detail (default)
   - **1024×1024** - Medium detail
   - **512×512** - Low detail / high performance
5. A yellow warning will appear: "⚡ Will reset to XXX×XXX resolution"
6. Click **"Reset Simulation"** button to apply the change
7. The simulation will recreate with the new resolution

### What Happens During Resolution Change

When you change resolution:

1. **Settings saved**: `simulation_settings.json` is updated with new values
2. **Full recreation**: GPU state is completely recreated with new buffers
3. **Ratio enforcement**: Fluid/spatial grids are set to env_grid/4 (maintains 4:1 ratio)
4. **Clean slate**: All agents, chemical fields, and state are reset

Example: Switching to 1024×1024 sets:
```json
"env_grid_resolution": 1024,
"fluid_grid_resolution": 256,
"spatial_grid_resolution": 256
```

### Loading Snapshots from Different Resolutions

When you load a snapshot saved at a different resolution, the system will:

1. **Detect the mismatch** and print a warning:
```
⚠️  WARNING: Resolution mismatch detected!
   Snapshot resolution: env=2048, fluid=512, spatial=512
   Current resolution:  env=1024, fluid=256, spatial=256
   Agent positions use absolute coordinates (SIM_SIZE=61440.0) and do not scale.
   Chemical grids will be resampled. Some visual differences may occur.
```

2. **Load agents normally**: Agent positions are in world coordinates (0-61440), which don't need scaling
3. **Resample chemical grids**: Alpha/beta/gamma chemical fields are interpolated to the new resolution

**Note**: Some visual differences may occur in chemical field distributions due to resampling.

## World Size vs Grid Resolution

**Important distinction**:

- **World Size (SIM_SIZE)**: Fixed at 61440.0 units
  - Agent positions are in world coordinates
  - Doesn't change with resolution
  - Agents don't need position scaling

- **Grid Resolution**: Variable (512/1024/2048)
  - Controls simulation detail/quality
  - Affects memory usage and performance
  - Chemical fields are resampled when changed

This design means:
- ✅ Agents stay in the same absolute positions
- ✅ No coordinate conversion needed
- ✅ Only grid sampling density changes

## Performance Implications

Higher resolution = more detail but slower:

| Resolution | Env Cells | Memory Impact | Performance |
|------------|-----------|---------------|-------------|
| 512×512    | 262K      | Low           | Fast        |
| 1024×1024  | 1.05M     | Medium        | Medium      |
| 2048×2048  | 4.19M     | High          | Slow        |

**Memory usage**: Each resolution level multiplies cell count by 4×.

**Recommendation**: Start with 2048×2048 for visual quality. Drop to 1024×1024 if performance is an issue. Use 512×512 for very large populations or slow hardware.

## Technical Details

### Snapshot Version History

- **v1.0**: Original format (no resolution info)
- **v1.1**: Added settings storage
- **v1.2**: Added resolution fields (env_grid_resolution, fluid_grid_resolution, spatial_grid_resolution)

### File Format Changes

The `SimulationSnapshot` struct now includes:
```rust
#[serde(default)]
env_grid_resolution: u32,
#[serde(default)]
fluid_grid_resolution: u32,
#[serde(default)]
spatial_grid_resolution: u32,
```

Old snapshots (v1.0/1.1) default these fields to 0, which the loader interprets as "no resolution data available."

### Implementation Architecture

1. **UI State**: `pending_resolution_change: Option<u32>` tracks requested resolution
2. **Settings Update**: `simulation_settings.json` is updated before recreation
3. **Full Rebuild**: `GpuState::new()` is called to recreate all GPU resources
4. **Shader Recompilation**: Shaders are recompiled with new grid constants

### Code Locations

- Resolution UI: [src/main.rs](../src/main.rs) ~line 13066
- Reset handler: [src/main.rs](../src/main.rs) ~line 15061
- Snapshot fields: [src/main.rs](../src/main.rs) ~line 3543
- Version check: [src/main.rs](../src/main.rs) ~line 11290

## Limitations

1. **No Hot Reload**: Resolution changes require simulation reset (by design)
2. **Fixed Ratios**: Fluid/spatial grids are always env_grid/4 (maintains optimal performance)
3. **Chemical Field Resampling**: Not perfect - some visual artifacts possible when loading different resolutions
4. **Manual Edit Required for Custom Values**: UI only supports 3 presets; custom values require editing JSON

## Future Enhancements

Potential improvements (not yet implemented):

- [ ] Custom resolution input field in UI
- [ ] Resolution scaling slider
- [ ] Agent-preserving resolution changes (save + reload snapshot automatically)
- [ ] Resolution presets in simulation settings dialog
- [ ] Better chemical field interpolation algorithms

## Troubleshooting

**Q: I clicked a resolution button but nothing happened**

A: You must click "Reset Simulation" after selecting a resolution. The yellow warning text reminds you of this.

**Q: My snapshot loaded but looks different at a new resolution**

A: This is expected. Chemical fields are resampled, which can create visual differences. Agent genomes and positions are preserved exactly.

**Q: Can I use resolutions other than the 3 presets?**

A: Yes, but you must edit `simulation_settings.json` manually and restart the application (not use the UI buttons).

**Q: What happens to performance at higher resolutions?**

A: GPU computation scales quadratically with resolution. Doubling resolution = 4× more cells = significantly slower simulation. Monitor your FPS and epochs/sec.

## See Also

- [RESOLUTION_SETTINGS.md](RESOLUTION_SETTINGS.md) - Original resolution configuration documentation
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Overall project architecture
- [simulation_settings.json](../simulation_settings.json) - Configuration file
