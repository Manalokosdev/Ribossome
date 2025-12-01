# Perlin Noise Rain System

## Overview
The rain system has been upgraded from gaussian noise to **time-evolving Perlin noise** for more coherent and interesting spatial patterns.

## How It Works

### Perlin Noise Parameters
Both alpha (food) and beta (poison) rain now use multi-octave Perlin noise with:
- **Octaves**: 3 (combines 3 layers of noise at different frequencies)
- **Scale**: 50.0 (controls pattern size - smaller = more detail)
- **Contrast**: 1.2 (sharpens the patterns)
- **Seeds**: 
  - Alpha: 12345
  - Beta: 67890 (different seed ensures independent patterns)

### Time Evolution
The noise patterns evolve slowly over time by shifting the sampling coordinates:
- **Time factor**: `random_seed / 100000.0`
- **Alpha drift**: Moves right (X-direction) over time
- **Beta drift**: Moves down (Y-direction) over time

This creates organic, slowly-changing patterns instead of static or completely random distributions.

### Rain Probability
Each cell's rain probability is calculated as:
```
alpha_probability = alpha_multiplier * 0.05 * perlin_noise_value
beta_probability = beta_multiplier * 0.05 * perlin_noise_value
```

The actual rain event is still stochastic - each cell compares against a random value per frame.

## Visualization Mode

### Enabling Rain Probability Visualization
To see the Perlin noise probability maps:
1. Set `debug_mode = 2` in the UI or settings
2. The screen will show:
   - **Green**: Alpha (food) rain probability
   - **Red**: Beta (poison) rain probability  
   - **Yellow**: Areas where both overlap
   - **Black**: Low probability areas

### Debug Mode Values
- `0`: Normal operation (no debug visualization)
- `1`: Agent debug mode (existing feature)
- `2`: **Rain probability map visualization** (NEW)

### What You'll See
- Smooth, coherent patterns instead of random noise
- Patterns slowly drift/evolve as the simulation runs
- High-probability regions appear brighter
- You can watch the noise patterns move in real-time

## Tuning Parameters

If you want to adjust the rain patterns, modify these values in `shader.wgsl` (around line 3534):

```wgsl
// Alpha rain Perlin noise
let alpha_noise_coord = vec2<f32>(grid_coord) / params.grid_size + vec2<f32>(time_factor, 0.0);
let alpha_perlin = layered_noise(
    alpha_noise_coord, 
    12345u,     // seed (change for different pattern)
    3u,         // octaves (more = more detail, slower)
    50.0,       // scale (smaller = finer patterns)
    1.2         // contrast (higher = sharper transitions)
);
```

### Parameter Effects
- **Octaves** (1-5): More octaves = richer detail but slower computation
- **Scale** (10-200): Lower = smaller, more frequent patches
- **Contrast** (0.5-2.0): Higher = sharper boundaries between high/low probability
- **Time divisor** (100000.0): Lower = faster evolution, higher = slower drift

## Benefits Over Gaussian Noise

1. **Spatial Coherence**: Creates regions of high/low food instead of random scatter
2. **Evolutionary Pressure**: Rewards agents that can find and exploit food-rich areas
3. **Time Dynamics**: Patterns shift slowly, forcing agents to adapt to changing environment
4. **Navigation Challenge**: Makes gradient-following sensors more valuable
5. **Visual Interest**: More organic, natural-looking resource distribution

## Implementation Details

The implementation is in `shader.wgsl`:
- **Rain generation**: Lines 3530-3555 (compute shader)
- **Visualization**: Lines 3747-3764 (fragment shader)
- **Noise functions**: Lines 1126-1175 (`noise2d`, `layered_noise`)

The `rain_map` buffer is no longer used - Perlin noise is calculated on-the-fly each frame based on the current `random_seed` value.
