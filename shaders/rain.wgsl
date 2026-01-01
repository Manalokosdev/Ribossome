// Targeted rain dispatch: instead of checking every cell, spawn only the expected number of rain drops.
// This is much more efficient when rain is sparse (e.g., 0.01% probability = 420 drops vs 4.2M cell checks).

@compute @workgroup_size(256)
fn apply_rain_drops(@builtin(global_invocation_id) gid: vec3<u32>) {
    let drop_id = gid.x;

    // Each invocation represents one rain drop to place
    // params.rain_drop_count is set on CPU based on expected rain this frame
    if (drop_id >= params.rain_drop_count) {
        return;
    }

    // Generate a random grid position for this drop
    let seed = drop_id + params.random_seed + params.epoch;
    let hash1 = hash(seed);
    let hash2 = hash(seed ^ 0x9e3779b9u);

    let x = u32(f32(hash1 % GRID_SIZE));
    let y = u32(f32(hash2 % GRID_SIZE));
    let idx = y * GRID_SIZE + x;

    // Read rain map multipliers for this cell
    let rain_packed = read_rain_maps(idx);
    let alpha_rain_map = clamp(rain_packed.x, 0.0, 1.0);
    let beta_rain_map = clamp(rain_packed.y, 0.0, 1.0);

    // Optional: apply precipitation boost based on local fluid velocity
    // For now, skip fluid sampling to keep it fast - we pre-calculated expected drops on CPU

    // Determine which type of drop this is based on the drop_id range
    // CPU calculates: alpha_drops = [0, alpha_count), beta_drops = [alpha_count, alpha_count+beta_count)
    if (drop_id < params.alpha_rain_drop_count) {
        // Alpha rain drop (food) - weighted by rain map
        // Use another RNG check to apply the rain_map probability
        let accept_seed = seed * 1103515245u;
        let accept_chance = hash_f32(accept_seed);
        if (accept_chance < alpha_rain_map) {
            // Write saturated alpha
            write_chem_alpha(idx, 1.0);
        }
    } else {
        // Beta rain drop (poison) - weighted by rain map
        let accept_seed = seed * 2147483647u;
        let accept_chance = hash_f32(accept_seed);
        if (accept_chance < beta_rain_map) {
            // Write saturated beta
            write_chem_beta(idx, 1.0);
        }
    }
}
