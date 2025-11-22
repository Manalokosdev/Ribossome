# Progress (Updated: 2025-11-11)

## Done

- Fixed spawn button functionality
- Ensured pause consistency
- Implemented consistent RNG advancement in queue_random_spawns using xorshift
- Fixed continuous spawning by clearing spawn queue after processing
- Fixed panic when draining spawn queue beyond available elements
- Changed spawn system to use GPU-based position randomization
- Fixed RNG inconsistency in replenish_population function
- **Implemented Auto Difficulty System**:
    - Added `AutoDifficultyParam` struct and logic in `src/main.rs`.
    - Implemented epoch-based cooldowns (instead of frame-based).
    - Added UI controls for Auto Difficulty settings.
    - Fixed persistence issues (resetting difficulty on simulation reset/startup).
- **Fixed Displacer Organ Bias**:
    - Identified and fixed a directional bias in `shader.wgsl` caused by `round()` on random floats.
    - Implemented uniform integer distribution using hash modulo for Displacer offsets.
- **Fixed Rain Map Thumbnails**:
    - Corrected vertical flip in `src/main.rs` by iterating rows normally.
- **Smoothed Slope Forces**:
    - Implemented `sample_gamma_slope_bilinear` in `shader.wgsl`.
    - Updated agent physics to use bilinear interpolation for slope forces, reducing jitter.

## Doing

- Monitoring simulation performance and correctness.

## Next

- Further optimization of simulation speed if needed.
