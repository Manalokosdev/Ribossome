// Standalone Fluid Simulation
// Stable-fluids style: advection + external forces + projection
//
// ============================================================================
// RECOMMENDED PER-FRAME DISPATCH ORDER (Rust/wgpu)
// ============================================================================
//
// Forces are written directly by agents to the forces buffer (binding 2) with
// 100x boost already applied. No intermediate buffer or copy needed.
//
// 1. [Agent simulation writes propeller forces directly to forces buffer]
//
// 2. add_forces (velocity += forces * dt) ← Apply forces immediately!
//
// 3. diffuse_velocity (optional explicit viscosity smoothing)
//
// 4. advect_velocity (self-advect velocity field)
//
// 5. vorticity_confinement (adds swirly detail, optional)
//
// 6. compute_divergence (∇·v)
//
// 7. clear_pressure (both pressure buffers)
//
// 8. jacobi_pressure (20-40 iterations with ping-pong)
//
// 9. subtract_gradient (make velocity divergence-free)
//
// 10. enforce_boundaries (free-slip walls)
//
// 11. advect_display_texture (optional feedback visualization)
//
// Note: Agents write to forces buffer directly at binding 2 (shared with fluid group).
//       Ping-pong velocity_in/velocity_out between steps as needed.
//       Final result should be in velocity_a for visualization.
//
// ============================================================================
// CONSTANTS
// ============================================================================

// NOTE: FLUID_GRID_SIZE is injected by Rust at shader compile time.

// Stability limits
const MAX_DT: f32 = 0.02;         // Clamp dt spikes (e.g., window stalls)
const MAX_FORCE: f32 = 200000.0;  // Clamp injected force magnitude per cell
const MAX_VEL: f32 = 2000.0;      // Clamp velocity magnitude per cell

// Fumaroles are configured at runtime from the UI / settings.
// They are supplied via a flat f32 storage buffer (no struct mirroring).
// Layout:
//   fumaroles[0] = count (as f32)
//   for each i in [0..count):
//     base = 1 + i*FUM_STRIDE
//     0 enabled (0/1)
//     1 x_frac (0..1, fluid-grid fraction)
//     2 y_frac (0..1, fluid-grid fraction)
//     3 dir_x
//     4 dir_y
//     5 strength (force magnitude, written into force_vectors)
//     6 spread (radius in fluid cells)
//     7 alpha_dye_rate (1/sec)
//     8 beta_dye_rate  (1/sec)
//     9 variation (0..1: jitter applied per-cell to direction/strength/rate)
const MAX_FUMAROLES: u32 = 64u;
const FUM_STRIDE: u32 = 10u;

const FUM_ENABLED: u32 = 0u;
const FUM_X_FRAC: u32 = 1u;
const FUM_Y_FRAC: u32 = 2u;
const FUM_DIR_X: u32 = 3u;
const FUM_DIR_Y: u32 = 4u;
const FUM_STRENGTH: u32 = 5u;
const FUM_SPREAD: u32 = 6u;
const FUM_ALPHA_DYE_RATE: u32 = 7u;
const FUM_BETA_DYE_RATE: u32 = 8u;
const FUM_VARIATION: u32 = 9u;

// Slope-driven obstacles (porous steepness)
// If enabled, steeper terrain becomes less permeable (more like rock / barrier).
const OBSTACLES_ENABLED: bool = true;
// Treat cells with very low permeability as effectively solid.
const SOLID_PERM_THRESHOLD: f32 = 0.02;

// Optional: drive fluid using the gamma slope (heightmap-style downhill flow).
// This uses the gradient of gamma (in fluid-cell space) as a force vector.
// Slope->fluid coupling mode:
// - Old mode (SLOPE_FORCE_ENABLED): directly accelerates along slope vector.
// - New mode (SLOPE_STEER_ENABLED): steers existing flow direction relative to slope.
// User request: keep the old method deactivated and try the steering method.
const SLOPE_FORCE_ENABLED: bool = false;
const SLOPE_STEER_ENABLED: bool = true;

// Runtime-controlled parameters (flat params buffer):
// - FP_FLUID_SLOPE_FORCE_SCALE: How strongly terrain slope drives fluid flow (default: 100.0)
// - FP_FLUID_OBSTACLE_STRENGTH: Permeability model perm = 1 / (1 + k * |slope|), larger k => stronger blocking (default: 200.0)

// Environment chem ↔ dye exchange tuning
// Controlled at runtime via FP_SPLAT.x/y and FP_CHEM_OOZE_STILL_RATE.

// Dye persistence & diffusion tuning
// 0..1: how much we blend towards a 4-neighbor blur each frame.
const DYE_DIFFUSE_MIX: f32 = 0.01;

// ============================================================================
// BINDINGS
// ============================================================================

// Velocity ping-pong buffers (vec2 per cell)
@group(0) @binding(0) var<storage, read> velocity_in: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> velocity_out: array<vec2<f32>>;

// Gamma grid (terrain) sampled as an obstacle field
@group(0) @binding(2) var<storage, read> gamma_grid: array<f32>;

// Environment chemistry grid:
// - chem_grid[idx].x = alpha (food)
// - chem_grid[idx].y = beta  (poison)
@group(0) @binding(10) var<storage, read_write> chem_grid: array<vec2<f32>>;

// Fluid-driven lift/deposition of the environment chem layers.
// Model: the advected dye field acts as the "carried" reservoir.
// Runtime knobs:
// - FP_SPLAT.x (chem_speed_equil): speed where transfer is 0 (equilibrium)
// - FP_SPLAT.y (chem_speed_max_lift): speed where transfer reaches max lift
// NOTE: This is (chem + dye) mass-conserving, except for saturation at 1.0.
// Single continuous transfer curve driven by dye-grid speed:
// - speed = 0                      => max deposition
// - speed = chem_speed_equil        => equilibrium (no lift/deposit)
// - speed >= chem_speed_max_lift    => max lift
// Overall transfer rate is controlled by CHEM_TRANSFER_RATE (fraction per second)
// while the two UI sliders control only the curve shape (speed thresholds).
const CHEM_TRANSFER_RATE: f32 = 4.0;
const CHEM_MAX_STEP_FRAC: f32 = 0.25; // hard clamp per tick to avoid instability

// Intermediate force vectors buffer (vec2 per cell) - agents write propeller forces here
@group(0) @binding(16) var<storage, read_write> fluid_force_vectors: array<vec2<f32>>;

// Combined forces buffer (vec2 per cell) - inject_test_force writes combined forces here
@group(0) @binding(7) var<storage, read_write> fluid_forces: array<vec2<f32>>;

// Pressure ping-pong + divergence (f32 per cell)
@group(0) @binding(4) var<storage, read> pressure_in: array<f32>;
@group(0) @binding(5) var<storage, read_write> pressure_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> divergence: array<f32>;

// Dye concentration ping-pong buffers (f32 per cell) - for flow visualization
// Two-channel dye per cell: x = beta (red), y = alpha (green)
@group(0) @binding(8) var<storage, read> dye_in: array<vec2<f32>>;
@group(0) @binding(9) var<storage, read_write> dye_out: array<vec2<f32>>;

// Agent trail dye (RGBA) advected by the fluid velocity.
// trail_in is prepared by the simulation pass (copy/decay + agent deposits).
// advect_trail writes the advected result into trail_out for rendering/sensing.
@group(0) @binding(12) var<storage, read_write> trail_out: array<vec4<f32>>;
@group(0) @binding(13) var<storage, read> trail_in: array<vec4<f32>>;

// Fumarole list (flat float buffer). See layout comment above.
@group(0) @binding(17) var<storage, read> fumaroles: array<f32>;

// Parameters
// IMPORTANT: keep this as a flat float buffer (no Rust<->WGSL struct mirroring).
// Packed as vec4<f32> to avoid uniform-array scalar stride pitfalls.
// Layout (f32 indices):
//  0 time
//  1 dt
//  2 decay
//  3 grid_size (as f32)
//  4..7  mouse (xy = pos in grid coords, zw = velocity in grid units/sec)
//  8..11 splat (x=ALPHA chem speed equil, y=ALPHA chem speed max lift, z=vorticity strength, w=viscosity)
//  12 fluid_slope_force_scale
//  13 fluid_obstacle_strength
//  14 vector_force_x
//  15 vector_force_y
//  16 vector_force_power
//  17 chem_ooze_still_rate
//  18 BETA chem speed equil
//  19 BETA chem speed max lift
//  20 dye_escape_rate_alpha (1/sec)
//  21 dye_escape_rate_beta (1/sec)
// (padded to 6 vec4s / 24 floats)
const FP_TIME: u32 = 0u;
const FP_DT: u32 = 1u;
const FP_DECAY: u32 = 2u;
const FP_GRID_SIZE: u32 = 3u;
const FP_MOUSE: u32 = 4u;
const FP_SPLAT: u32 = 8u;
const FP_FLUID_SLOPE_FORCE_SCALE: u32 = 12u;
const FP_FLUID_OBSTACLE_STRENGTH: u32 = 13u;
const FP_VECTOR_FORCE_X: u32 = 14u;
const FP_VECTOR_FORCE_Y: u32 = 15u;
const FP_VECTOR_FORCE_POWER: u32 = 16u;
const FP_CHEM_OOZE_STILL_RATE: u32 = 17u;
const FP_CHEM_SPEED_EQUIL_BETA: u32 = 18u;
const FP_CHEM_SPEED_MAX_LIFT_BETA: u32 = 19u;
const FP_DYE_ESCAPE_RATE_ALPHA: u32 = 20u;
const FP_DYE_ESCAPE_RATE_BETA: u32 = 21u;
const FP_VEC4_COUNT: u32 = 6u;

@group(0) @binding(3) var<uniform> params: array<vec4<f32>, FP_VEC4_COUNT>;

fn fp_f32(i: u32) -> f32 {
    let v = params[i >> 2u];
    let lane = i & 3u;
    if (lane == 0u) { return v.x; }
    if (lane == 1u) { return v.y; }
    if (lane == 2u) { return v.z; }
    return v.w;
}

fn fp_vec4(base: u32) -> vec4<f32> {
    return vec4<f32>(
        fp_f32(base + 0u),
        fp_f32(base + 1u),
        fp_f32(base + 2u),
        fp_f32(base + 3u),
    );
}

fn fum_f32(base: u32, off: u32) -> f32 {
    return fumaroles[base + off];
}

fn rotate2(v: vec2<f32>, a: f32) -> vec2<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec2<f32>(v.x * c - v.y * s, v.x * s + v.y * c);
}

// Display/feedback texture (ping-pong) used for visualization.
// This is in group(1) so it can coexist with the buffer-only group(0) layout.
// NOTE: WebGPU disallows read-only storage textures unless using a native-only feature;
// so we read via a sampled texture + sampler and write via a storage texture.
@group(1) @binding(0) var display_tex_in: texture_2d<f32>;
@group(1) @binding(1) var display_tex_sampler: sampler;
@group(1) @binding(2) var display_tex_out: texture_storage_2d<rgba16float, write>;

// ============================================================================
// VELOCITY EXPORT (for visualization in main sim composite)
// ============================================================================

@compute @workgroup_size(16, 16)
fn copy_velocity_to_forces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);
    fluid_forces[idx] = velocity_in[idx];
}

// Copy dye concentration to forces buffer for visualization (dye in x, zero in y)
@compute @workgroup_size(16, 16)
fn copy_dye_to_forces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);
    // Dye buffers are GAMMA_GRID_DIM^2, so map the current fluid cell to an env cell.
    let env_idx = gamma_idx_for_fluid_cell(x, y);
    // Store dye (beta, alpha) directly for downstream visualization/debug.
    fluid_forces[idx] = dye_in[env_idx];
}

@compute @workgroup_size(16, 16)
fn clear_forces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);
    fluid_forces[idx] = vec2<f32>(0.0, 0.0);
}

// Clear intermediate force vectors buffer (called before agents write)
@compute @workgroup_size(16, 16)
fn clear_fluid_force_vectors(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);
    fluid_force_vectors[idx] = vec2<f32>(0.0, 0.0);
}

// Inject a single point force into the intermediate force buffer.
// This is meant as a deterministic test driver so you can see how the fluid affects agents.
@compute @workgroup_size(1, 1, 1)
fn inject_fumarole_force_vector(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Single-invocation kernel: only one thread writes the single-cell force.
    if (gid.x != 0u || gid.y != 0u || gid.z != 0u) {
        return;
    }

    let count_f = fumaroles[0u];
    let count = u32(clamp(floor(count_f + 0.0001), 0.0, f32(MAX_FUMAROLES)));
    if (count == 0u) {
        return;
    }

    let size_f = f32(FLUID_GRID_SIZE);
    // Use coarse time buckets so variation doesn't flicker every single frame.
    let time_bucket = u32(fp_f32(FP_TIME)) / 16u;

    for (var i = 0u; i < count; i = i + 1u) {
        let base = 1u + i * FUM_STRIDE;
        let enabled = fum_f32(base, FUM_ENABLED);
        if (enabled < 0.5) {
            continue;
        }

        let x_frac = clamp(fum_f32(base, FUM_X_FRAC), 0.0, 1.0);
        let y_frac = clamp(fum_f32(base, FUM_Y_FRAC), 0.0, 1.0);

        // Keep it away from edges to avoid boundary artifacts.
        let fx_i = clamp(i32(round(size_f * x_frac)), 1, i32(FLUID_GRID_SIZE) - 2);
        let fy_i = clamp(i32(round(size_f * y_frac)), 1, i32(FLUID_GRID_SIZE) - 2);

        var dir = vec2<f32>(fum_f32(base, FUM_DIR_X), fum_f32(base, FUM_DIR_Y));
        let dir_len = length(dir);
        dir = select(dir / max(dir_len, 1e-8), vec2<f32>(0.0, 1.0), dir_len < 1e-5);

        let base_strength = max(fum_f32(base, FUM_STRENGTH), 0.0);
        let spread_f = max(fum_f32(base, FUM_SPREAD), 0.0);
        let variation = clamp(fum_f32(base, FUM_VARIATION), 0.0, 1.0);

        // Variation should be per-cell (spatially randomized), not once-per-fumarole.
        let seed_base = hash_u32(i * 0x9E3779B9u ^ time_bucket * 0x85EBCA6Bu ^ 0xC2B2AE35u);

        // Spread in fluid-cell units.
        let r = i32(clamp(round(spread_f), 0.0, 64.0));
        if (r <= 0) {
            let idx = grid_index(u32(fx_i), u32(fy_i));

            let seed0 = hash_u32(seed_base ^ idx);
            let seed1 = hash_u32(seed0 ^ 0xA511E9B3u);
            let seed2 = hash_u32(seed0 ^ 0x63D83595u);
            let ang = (rand01(seed1) * 2.0 - 1.0) * 3.14159265 * variation;
            let strength_jitter = 1.0 + (rand01(seed2) * 2.0 - 1.0) * variation;

            let dir_j = normalize(rotate2(dir, ang));
            let strength = base_strength * max(strength_jitter, 0.0);
            fluid_force_vectors[idx] = fluid_force_vectors[idx] + dir_j * strength;
        } else {
            for (var dy: i32 = -r; dy <= r; dy = dy + 1) {
                for (var dx: i32 = -r; dx <= r; dx = dx + 1) {
                    let xi = clamp(fx_i + dx, 1, i32(FLUID_GRID_SIZE) - 2);
                    let yi = clamp(fy_i + dy, 1, i32(FLUID_GRID_SIZE) - 2);
                    let idx = grid_index(u32(xi), u32(yi));

                    let seed0 = hash_u32(seed_base ^ idx);
                    let seed1 = hash_u32(seed0 ^ 0xA511E9B3u);
                    let seed2 = hash_u32(seed0 ^ 0x63D83595u);
                    let ang = (rand01(seed1) * 2.0 - 1.0) * 3.14159265 * variation;
                    let strength_jitter = 1.0 + (rand01(seed2) * 2.0 - 1.0) * variation;
                    let dir_j = normalize(rotate2(dir, ang));
                    let strength = base_strength * max(strength_jitter, 0.0);

                    let dist = length(vec2<f32>(f32(dx), f32(dy)));
                    // Smooth falloff (stronger core): w in [0..1]
                    let w0 = clamp(1.0 - dist / (f32(r) + 0.001), 0.0, 1.0);
                    let w = w0 * w0;

                    fluid_force_vectors[idx] = fluid_force_vectors[idx] + dir_j * (strength * w);
                }
            }
        }
    }
}

// Inject dye at the fumarole location.
// IMPORTANT: This must run *after* inject_dye(), otherwise inject_dye() will overwrite dye_out.
@compute @workgroup_size(1, 1, 1)
fn inject_fumarole_dye(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Single-invocation kernel: only one thread writes the single-cell dye.
    if (gid.x != 0u || gid.y != 0u || gid.z != 0u) {
        return;
    }

    let count_f = fumaroles[0u];
    let count = u32(clamp(floor(count_f + 0.0001), 0.0, f32(MAX_FUMAROLES)));
    if (count == 0u) {
        return;
    }

    let dt = clamp(fp_f32(FP_DT), 0.0, MAX_DT);
    let size_f = f32(FLUID_GRID_SIZE);
    let fluid_to_dye = f32(GAMMA_GRID_DIM) / f32(FLUID_GRID_SIZE);
    let time_bucket = u32(fp_f32(FP_TIME)) / 16u;

    for (var i = 0u; i < count; i = i + 1u) {
        let base = 1u + i * FUM_STRIDE;
        let enabled = fum_f32(base, FUM_ENABLED);
        if (enabled < 0.5) {
            continue;
        }

        let x_frac = clamp(fum_f32(base, FUM_X_FRAC), 0.0, 1.0);
        let y_frac = clamp(fum_f32(base, FUM_Y_FRAC), 0.0, 1.0);

        let fx_i = clamp(i32(round(size_f * x_frac)), 1, i32(FLUID_GRID_SIZE) - 2);
        let fy_i = clamp(i32(round(size_f * y_frac)), 1, i32(FLUID_GRID_SIZE) - 2);

        // Convert to dye grid coordinates.
        let dye_x0 = i32(clamp(f32(fx_i) * fluid_to_dye, 0.0, f32(GAMMA_GRID_DIM - 1u)));
        let dye_y0 = i32(clamp(f32(fy_i) * fluid_to_dye, 0.0, f32(GAMMA_GRID_DIM - 1u)));

        let spread_f = max(fum_f32(base, FUM_SPREAD), 0.0);
        // Map fluid-cell radius to dye-cell radius.
        let r_dye = i32(clamp(round(spread_f * fluid_to_dye), 1.0, 128.0));

        let base_rate_alpha = max(fum_f32(base, FUM_ALPHA_DYE_RATE), 0.0);
        let base_rate_beta = max(fum_f32(base, FUM_BETA_DYE_RATE), 0.0);
        let variation = clamp(fum_f32(base, FUM_VARIATION), 0.0, 1.0);

        if (base_rate_alpha <= 0.0 && base_rate_beta <= 0.0) {
            continue;
        }

        let seed_base = hash_u32(i * 0x9E3779B9u ^ time_bucket * 0x85EBCA6Bu ^ 0x27D4EB2Fu);

        for (var dy: i32 = -r_dye; dy <= r_dye; dy = dy + 1) {
            for (var dx: i32 = -r_dye; dx <= r_dye; dx = dx + 1) {
                let xi = clamp(dye_x0 + dx, 0, i32(GAMMA_GRID_DIM) - 1);
                let yi = clamp(dye_y0 + dy, 0, i32(GAMMA_GRID_DIM) - 1);
                let dye_idx = dye_grid_index(u32(xi), u32(yi));

                let seed0 = hash_u32(seed_base ^ dye_idx);
                let seed1 = hash_u32(seed0 ^ 0xB7E15162u);
                let rate_jitter = 1.0 + (rand01(seed1) * 2.0 - 1.0) * variation;
                let jitter = max(rate_jitter, 0.0);

                // Frame-rate independent injection: amount = rate * dt.
                let injected_amount_alpha = base_rate_alpha * jitter * dt;
                let injected_amount_beta = base_rate_beta * jitter * dt;
                if (injected_amount_alpha <= 0.0 && injected_amount_beta <= 0.0) {
                    continue;
                }

                let dist = length(vec2<f32>(f32(dx), f32(dy)));
                let w0 = clamp(1.0 - dist / (f32(r_dye) + 0.001), 0.0, 1.0);
                let w = w0 * w0;

                let current = dye_out[dye_idx];
                let next_beta = min(current.x + injected_amount_beta * w, 1.0);
                let next_alpha = min(current.y + injected_amount_alpha * w, 1.0);
                dye_out[dye_idx] = vec2<f32>(next_beta, next_alpha);
            }
        }
    }
}

// Combine propeller forces with test force - reads from fluid_force_vectors, writes to fluid_forces
@compute @workgroup_size(16, 16)
fn inject_test_force(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);

    // Start with propeller forces from agents (simulation pass writes into fluid_force_vectors).
    // Clamp to avoid solver blow-ups from rare extreme values.
    let f = sanitize_vec2(fluid_force_vectors[idx]);
    fluid_forces[idx] = clamp_vec2_len(f, MAX_FORCE);
}

// Fill the forces buffer with deterministic pseudo-random vectors.
// Used to verify the "forces -> velocity" path independent of agent propellers.
@compute @workgroup_size(16, 16)
fn randomize_forces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);

    // Deterministic per-frame seed from time (ms-ish).
    let t = u32(fp_f32(FP_TIME) * 1000.0);
    let r1 = rand01(idx ^ t ^ 0xA511E9B3u);
    let r2 = rand01(idx ^ t ^ 0x63D83595u);
    let v = vec2<f32>(r1 * 2.0 - 1.0, r2 * 2.0 - 1.0);

    let strength = 1.0;
    fluid_forces[idx] = v * strength;
}

// Deprecated: legacy chem->dye injection pass.
// We keep this kernel to preserve pipeline structure, but it no longer transfers
// mass between chem and dye. All chem↔dye exchange is handled by advect_dye using
// the single continuous speed transfer function.
@compute @workgroup_size(16, 16)
fn inject_dye(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    // Dye is stored/updated at environment resolution.
    if (x >= GAMMA_GRID_DIM || y >= GAMMA_GRID_DIM) {
        return;
    }

    let idx = dye_grid_index(x, y);

    // Pass-through.
    dye_out[idx] = clamp(dye_in[idx], vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
}

// Advect dye concentration using semi-Lagrangian method (same as velocity advection)
@compute @workgroup_size(16, 16)
fn advect_dye(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= GAMMA_GRID_DIM || y >= GAMMA_GRID_DIM) {
        return;
    }

    let idx = dye_grid_index(x, y);
    // Position in dye-grid coordinates.
    let pos = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5);

    // Convert dye-grid position into fluid-grid coordinates for velocity sampling.
    let dye_to_fluid = f32(FLUID_GRID_SIZE) / f32(GAMMA_GRID_DIM);
    let fluid_to_dye = 1.0 / dye_to_fluid;
    let fluid_pos = pos * dye_to_fluid;

    // Read current velocity
    let dt = clamp(fp_f32(FP_DT), 0.0, MAX_DT);
    let vel_fluid = clamp_vec2_len(sanitize_vec2(sample_velocity(fluid_pos)), MAX_VEL);

    // Convert velocity to dye-grid units before tracing.
    let vel_dye = vel_fluid * fluid_to_dye;

    // Backward trace - follow the flow backwards to find where dye came from
    let trace_pos = pos - vel_dye * dt;

    // Sample dye concentration at traced position using bilinear interpolation
    let advected_dye = sample_dye(trace_pos);

    // Extra diffusion: blend advected dye towards a local 4-neighbor blur.
    // This makes the dye spread out more instead of staying filament-thin.
    let cx = i32(x);
    let cy = i32(y);
    let l = dye_clamp_coords(cx - 1, cy);
    let r = dye_clamp_coords(cx + 1, cy);
    let u = dye_clamp_coords(cx, cy - 1);
    let d = dye_clamp_coords(cx, cy + 1);
    let d_l = dye_in[dye_grid_index(l.x, l.y)];
    let d_r = dye_in[dye_grid_index(r.x, r.y)];
    let d_u = dye_in[dye_grid_index(u.x, u.y)];
    let d_d = dye_in[dye_grid_index(d.x, d.y)];
    let neighbor_blur = (d_l + d_r + d_u + d_d) * 0.25;
    let advected_diffused = mix(advected_dye, neighbor_blur, clamp(DYE_DIFFUSE_MIX, 0.0, 1.0));

    // Dye ignores obstacles: don't attenuate by permeability.
    var dye_val = clamp(advected_diffused, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));

    // === Chem erosion/deposition coupling (mass-conserving) ===
    // Use dye-grid velocity magnitude (in dye cells / sec) as the driver.
    let speed = length(vel_dye);

    let chem_speed_equil_alpha_in = max(fp_vec4(FP_SPLAT).x, 0.0);
    let chem_speed_max_lift_alpha_in = max(fp_vec4(FP_SPLAT).y, 0.0);
    let chem_speed_equil_beta_in = max(fp_f32(FP_CHEM_SPEED_EQUIL_BETA), 0.0);
    let chem_speed_max_lift_beta_in = max(fp_f32(FP_CHEM_SPEED_MAX_LIFT_BETA), 0.0);

    // Ensure sane monotonic curves even if UI values are inverted.
    let chem_speed_equil_alpha = clamp(chem_speed_equil_alpha_in, 1e-4, 1000.0);
    let chem_speed_max_lift_alpha = max(chem_speed_max_lift_alpha_in, chem_speed_equil_alpha + 1e-4);
    let chem_speed_equil_beta = clamp(chem_speed_equil_beta_in, 1e-4, 1000.0);
    let chem_speed_max_lift_beta = max(chem_speed_max_lift_beta_in, chem_speed_equil_beta + 1e-4);

    // Each dye cell maps 1:1 to the environment chem cell at the same resolution.
    let env_idx = gamma_index(x, y);
    var chem = clamp(chem_grid[env_idx], vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));

    // Baseline chem→dye ooze in still water, so sensors can detect signals without flow.
    // IMPORTANT: this is intentionally NON-depleting (does not remove chem from the grid).
    // Think of it as a sensing/bleed copy into the advected dye layer.
    let ooze_rate = max(fp_f32(FP_CHEM_OOZE_STILL_RATE), 0.0);
    let stillness_alpha = 1.0 - smoothstep(0.0, chem_speed_equil_alpha, speed);
    let stillness_beta = 1.0 - smoothstep(0.0, chem_speed_equil_beta, speed);
    let ooze_frac_alpha = clamp(ooze_rate * stillness_alpha * dt, 0.0, CHEM_MAX_STEP_FRAC);
    let ooze_frac_beta = clamp(ooze_rate * stillness_beta * dt, 0.0, CHEM_MAX_STEP_FRAC);
    if (ooze_frac_alpha > 0.0) {
        // Alpha: chem.x -> dye.y
        let alpha_headroom = 1.0 - dye_val.y;
        let ooze_alpha = min(chem.x * ooze_frac_alpha, max(alpha_headroom, 0.0));
        dye_val.y = min(dye_val.y + ooze_alpha, 1.0);
    }
    if (ooze_frac_beta > 0.0) {
        // Beta: chem.y -> dye.x
        let beta_headroom = 1.0 - dye_val.x;
        let ooze_beta = min(chem.y * ooze_frac_beta, max(beta_headroom, 0.0));
        dye_val.x = min(dye_val.x + ooze_beta, 1.0);
    }

    // Channel-specific continuous transfer functions based on speed.
    // transfer_* in [-1, 1]
    //  +1 => max deposition (dye -> chem)
    //   0 => equilibrium
    //  -1 => max lift (chem -> dye)
    let deposit_strength_alpha = 1.0 - smoothstep(0.0, chem_speed_equil_alpha, speed);
    let lift_strength_alpha = smoothstep(chem_speed_equil_alpha, chem_speed_max_lift_alpha, speed);
    let transfer_alpha = clamp(deposit_strength_alpha - lift_strength_alpha, -1.0, 1.0);

    let deposit_strength_beta = 1.0 - smoothstep(0.0, chem_speed_equil_beta, speed);
    let lift_strength_beta = smoothstep(chem_speed_equil_beta, chem_speed_max_lift_beta, speed);
    let transfer_beta = clamp(deposit_strength_beta - lift_strength_beta, -1.0, 1.0);

    let pickup_strength_alpha = max(-transfer_alpha, 0.0);
    let pickup_strength_beta = max(-transfer_beta, 0.0);
    let deposit_strength_alpha_pos = max(transfer_alpha, 0.0);
    let deposit_strength_beta_pos = max(transfer_beta, 0.0);

    // Pickup (chem -> dye)
    let pickup_frac_alpha = clamp(CHEM_TRANSFER_RATE * pickup_strength_alpha * dt, 0.0, CHEM_MAX_STEP_FRAC);
    let pickup_frac_beta = clamp(CHEM_TRANSFER_RATE * pickup_strength_beta * dt, 0.0, CHEM_MAX_STEP_FRAC);

    let pickup_alpha_raw = chem.x * pickup_frac_alpha;
    let pickup_beta_raw = chem.y * pickup_frac_beta;

    // Clamp by dye headroom to avoid losing mass due to saturation.
    let dye_headroom_alpha = max(1.0 - dye_val.y, 0.0);
    let dye_headroom_beta = max(1.0 - dye_val.x, 0.0);
    let pickup_alpha = min(pickup_alpha_raw, dye_headroom_alpha);
    let pickup_beta = min(pickup_beta_raw, dye_headroom_beta);

    chem.x = max(chem.x - pickup_alpha, 0.0);
    chem.y = max(chem.y - pickup_beta, 0.0);
    dye_val.y = min(dye_val.y + pickup_alpha, 1.0);
    dye_val.x = min(dye_val.x + pickup_beta, 1.0);

    // Deposit (dye -> chem)
    let deposit_frac_alpha = clamp(CHEM_TRANSFER_RATE * deposit_strength_alpha_pos * dt, 0.0, CHEM_MAX_STEP_FRAC);
    let deposit_frac_beta = clamp(CHEM_TRANSFER_RATE * deposit_strength_beta_pos * dt, 0.0, CHEM_MAX_STEP_FRAC);

    // Deposit only up to available headroom in chem to preserve mass.
    let chem_headroom_alpha = max(1.0 - chem.x, 0.0);
    let chem_headroom_beta = max(1.0 - chem.y, 0.0);
    let deposit_alpha = min(dye_val.y * deposit_frac_alpha, chem_headroom_alpha);
    let deposit_beta = min(dye_val.x * deposit_frac_beta, chem_headroom_beta);
    chem.x = chem.x + deposit_alpha;
    chem.y = chem.y + deposit_beta;
    dye_val.y = max(dye_val.y - deposit_alpha, 0.0);
    dye_val.x = max(dye_val.x - deposit_beta, 0.0);

    // === Dye escape/decay (non-precipitating sink) ===
    // This counteracts still-water ooze by letting dye fade away without
    // depositing back into the chem grids.
    let escape_rate_alpha = max(fp_f32(FP_DYE_ESCAPE_RATE_ALPHA), 0.0);
    let escape_rate_beta = max(fp_f32(FP_DYE_ESCAPE_RATE_BETA), 0.0);
    let escape_factor_alpha = exp(-escape_rate_alpha * dt);
    let escape_factor_beta = exp(-escape_rate_beta * dt);
    dye_val.y = dye_val.y * escape_factor_alpha;
    dye_val.x = dye_val.x * escape_factor_beta;

    chem_grid[env_idx] = chem;
    dye_out[idx] = clamp(dye_val, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
}

// Clear dye buffer (for reset)
@compute @workgroup_size(16, 16)
fn clear_dye(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= GAMMA_GRID_DIM || y >= GAMMA_GRID_DIM) {
        return;
    }

    let idx = dye_grid_index(x, y);
    dye_out[idx] = vec2<f32>(0.0, 0.0);
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

fn grid_index(x: u32, y: u32) -> u32 {
    return y * FLUID_GRID_SIZE + x;
}

// Dye is stored at environment resolution (GAMMA_GRID_DIM x GAMMA_GRID_DIM).
fn dye_grid_index(x: u32, y: u32) -> u32 {
    return y * GAMMA_GRID_DIM + x;
}

fn hash_u32(v: u32) -> u32 {
    var x = v;
    x = x ^ (x >> 16u);
    x = x * 0x7FEB352Du;
    x = x ^ (x >> 15u);
    x = x * 0x846CA68Bu;
    x = x ^ (x >> 16u);
    return x;
}

fn rand01(seed: u32) -> f32 {
    // [0, 1)
    return f32(hash_u32(seed)) / 4294967296.0;
}

fn is_bad_f32(x: f32) -> bool {
    // WGSL portability: detect NaN via (x != x). Detect Inf/overflow via a large threshold.
    return (x != x) || (abs(x) > 1e20);
}

fn sanitize_vec2(v: vec2<f32>) -> vec2<f32> {
    if (is_bad_f32(v.x) || is_bad_f32(v.y)) {
        return vec2<f32>(0.0, 0.0);
    }
    return v;
}

fn clamp_vec2_len(v: vec2<f32>, max_len: f32) -> vec2<f32> {
    let len = length(v);
    if (len > max_len) {
        return v * (max_len / max(len, 1e-12));
    }
    return v;
}

fn obstacles_enabled() -> bool {
    return OBSTACLES_ENABLED;
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / max(edge1 - edge0, 1e-12), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn gamma_index(x: u32, y: u32) -> u32 {
    return y * GAMMA_GRID_DIM + x;
}

// gamma_grid packs 3 layers: height + slope_x + slope_y, each of size GAMMA_GRID_DIM^2.
const GAMMA_LAYER_SIZE: u32 = GAMMA_GRID_DIM * GAMMA_GRID_DIM;
const GAMMA_SLOPE_X_OFFSET: u32 = GAMMA_LAYER_SIZE;
const GAMMA_SLOPE_Y_OFFSET: u32 = GAMMA_LAYER_SIZE * 2u;

fn gamma_height_at_idx(idx: u32) -> f32 {
    return gamma_grid[idx];
}

fn gamma_slope_at_idx(idx: u32) -> vec2<f32> {
    return vec2<f32>(gamma_grid[idx + GAMMA_SLOPE_X_OFFSET], gamma_grid[idx + GAMMA_SLOPE_Y_OFFSET]);
}

fn gamma_at_fluid_cell(x: u32, y: u32) -> f32 {
    let idx = gamma_idx_for_fluid_cell(x, y);
    return gamma_height_at_idx(idx);
}

fn gamma_slope_force(x: u32, y: u32) -> vec2<f32> {
    if (!SLOPE_FORCE_ENABLED) {
        return vec2<f32>(0.0, 0.0);
    }

    // Uses the simulation-provided slope vector field (gamma+alpha+beta contributions).
    let idx = gamma_idx_for_fluid_cell(x, y);

    var slope_grad = gamma_slope_at_idx(idx);

    // Add the global vector force (physics uses vector_force_x/y scaled by vector_force_power).
    let global_vec = vec2<f32>(fp_f32(FP_VECTOR_FORCE_X), fp_f32(FP_VECTOR_FORCE_Y)) * fp_f32(FP_VECTOR_FORCE_POWER);
    slope_grad = slope_grad + global_vec;

    // gamma_slope is a gradient (points uphill). Fluid should be driven downhill.
    return -sanitize_vec2(slope_grad) * fp_f32(FP_FLUID_SLOPE_FORCE_SCALE);
}

fn gamma_slope_vector(x: u32, y: u32) -> vec2<f32> {
    // Uses the simulation-provided slope vector field (gamma+alpha+beta contributions).
    let idx = gamma_idx_for_fluid_cell(x, y);
    var slope_grad = gamma_slope_at_idx(idx);

    // Match the old behavior: include the global vector force.
    let global_vec = vec2<f32>(fp_f32(FP_VECTOR_FORCE_X), fp_f32(FP_VECTOR_FORCE_Y)) * fp_f32(FP_VECTOR_FORCE_POWER);
    slope_grad = slope_grad + global_vec;

    // gamma_slope is a gradient (points uphill). We want the downhill direction.
    return -sanitize_vec2(slope_grad);
}

fn gamma_slope_steer_velocity(x: u32, y: u32, v_in: vec2<f32>, dt: f32) -> vec2<f32> {
    if (!SLOPE_STEER_ENABLED) {
        return v_in;
    }

    // Direction-only steering:
    // Keep |v| the same, but rotate the direction toward the slope direction.
    // This matches the intent: “magnitude should be the same as the original fluid vector, but on another direction”.
    let s = gamma_slope_vector(x, y);

    let v_len = length(v_in);
    let s_len = length(s);
    if (v_len < 1e-5 || s_len < 1e-5) {
        return v_in;
    }

    let v_dir = v_in / v_len;
    let s_dir = s / s_len;

    let align = clamp(dot(v_dir, s_dir), -1.0, 1.0);
    // More steering when misaligned, none when aligned.
    // Clamp to [0,1] so this acts like a rotation/lerp rather than an acceleration.
    let t = clamp((1.0 - align) * fp_f32(FP_FLUID_SLOPE_FORCE_SCALE) * dt, 0.0, 1.0);

    // Lerp directions then renormalize.
    let dir_raw = v_dir + (s_dir - v_dir) * t;
    let dir_len = length(dir_raw);
    if (dir_len < 1e-5) {
        return v_in;
    }

    return (dir_raw / dir_len) * v_len;
}

fn gamma_idx_for_fluid_cell(x: u32, y: u32) -> u32 {
    let fx = (f32(x) + 0.5) * f32(GAMMA_GRID_DIM) / f32(FLUID_GRID_SIZE);
    let fy = (f32(y) + 0.5) * f32(GAMMA_GRID_DIM) / f32(FLUID_GRID_SIZE);
    let max_idx_f = f32(GAMMA_GRID_DIM - 1u);
    let gx = u32(clamp(fx, 0.0, max_idx_f));
    let gy = u32(clamp(fy, 0.0, max_idx_f));
    return gamma_index(gx, gy);
}

fn slope_magnitude_at_fluid_cell(x: u32, y: u32) -> f32 {
    let idx = gamma_idx_for_fluid_cell(x, y);
    let s = sanitize_vec2(gamma_slope_at_idx(idx));
    return length(s);
}

fn slope_permeability(x: u32, y: u32) -> f32 {
    // Porous steepness: flat = ~1, steep = smaller.
    // Permeability model: perm = 1 / (1 + k * |slope|)
    // Larger k => stronger blocking.
    let m = slope_magnitude_at_fluid_cell(x, y);
    return 1.0 / (1.0 + fp_f32(FP_FLUID_OBSTACLE_STRENGTH) * m);
}

fn obstacle_strength(x: u32, y: u32) -> f32 {
    // 0..1 where 1 means fully solid.
    // In slope-only mode we derive obstacles from steepness (not gamma height).
    let perm = permeability(x, y);
    return 1.0 - perm;
}

fn permeability(x: u32, y: u32) -> f32 {
    // 1..0 where 0 means fully solid.
    if (!obstacles_enabled()) {
        return 1.0;
    }
    return clamp(slope_permeability(x, y), 0.0, 1.0);
}

fn is_effectively_solid(x: u32, y: u32) -> bool {
    return permeability(x, y) < SOLID_PERM_THRESHOLD;
}

fn raw_velocity_cell(x: u32, y: u32) -> vec2<f32> {
    if (is_effectively_solid(x, y)) {
        return vec2<f32>(0.0, 0.0);
    }
    return sanitize_vec2(velocity_in[grid_index(x, y)]);
}

fn solid_normal_from_neighbors(x: u32, y: u32) -> vec2<f32> {
    if (!obstacles_enabled()) {
        return vec2<f32>(0.0, 0.0);
    }

    // Build an approximate outward normal from adjacent solid cells.
    // If the left cell is solid, the normal points +x (away from the wall), etc.
    let xm = select(x - 1u, x, x == 0u);
    let xp = select(x + 1u, x, x + 1u >= FLUID_GRID_SIZE);
    let ym = select(y - 1u, y, y == 0u);
    let yp = select(y + 1u, y, y + 1u >= FLUID_GRID_SIZE);

    let s_l = select(0.0, 1.0, (x > 0u) && is_effectively_solid(xm, y));
    let s_r = select(0.0, 1.0, (x + 1u < FLUID_GRID_SIZE) && is_effectively_solid(xp, y));
    let s_b = select(0.0, 1.0, (y > 0u) && is_effectively_solid(x, ym));
    let s_t = select(0.0, 1.0, (y + 1u < FLUID_GRID_SIZE) && is_effectively_solid(x, yp));

    let n = vec2<f32>(s_l - s_r, s_b - s_t);
    let n_len = length(n);
    return select(vec2<f32>(0.0, 0.0), n / n_len, n_len > 1e-6);
}

fn reflect_if_into_solid(x: u32, y: u32, v_in: vec2<f32>) -> vec2<f32> {
    // Energy-preserving reflection against the nearest solid boundary (if any).
    let n = solid_normal_from_neighbors(x, y);
    let d = dot(v_in, n);
    if (length(n) > 1e-6 && d < 0.0) {
        // Reflect the component pointing into the wall.
        return v_in - 2.0 * n * d;
    }
    return v_in;
}

fn is_solid_at_pos(pos: vec2<f32>) -> bool {
    // pos is in fluid-grid coords with cell centers at (x+0.5, y+0.5)
    let ix = i32(floor(pos.x - 0.5));
    let iy = i32(floor(pos.y - 0.5));
    if (ix < 0 || ix >= i32(FLUID_GRID_SIZE) || iy < 0 || iy >= i32(FLUID_GRID_SIZE)) {
        return true;
    }
    return is_effectively_solid(u32(ix), u32(iy));
}

fn clamp_coords(x: i32, y: i32) -> vec2<u32> {
    let cx = clamp(x, 0, i32(FLUID_GRID_SIZE) - 1);
    let cy = clamp(y, 0, i32(FLUID_GRID_SIZE) - 1);
    return vec2<u32>(u32(cx), u32(cy));
}

fn dye_clamp_coords(x: i32, y: i32) -> vec2<u32> {
    let cx = clamp(x, 0, i32(GAMMA_GRID_DIM) - 1);
    let cy = clamp(y, 0, i32(GAMMA_GRID_DIM) - 1);
    return vec2<u32>(u32(cx), u32(cy));
}

// Bilinear interpolation for velocity sampling
fn sample_velocity(pos: vec2<f32>) -> vec2<f32> {
    // Solid-wall behavior for semi-Lagrangian advection:
    // - If the traced position leaves the domain, treat it as sampling zero.
    // This prevents “outside wind” artifacts that occur when out-of-bounds
    // positions are clamped back into the interior.
    let min_pos = 0.5;
    let max_pos = f32(FLUID_GRID_SIZE) - 0.5;
    if (pos.x < min_pos || pos.x > max_pos || pos.y < min_pos || pos.y > max_pos) {
        return vec2<f32>(0.0, 0.0);
    }
    let p = clamp(pos, vec2<f32>(min_pos), vec2<f32>(max_pos));

    let x = p.x - 0.5;
    let y = p.y - 0.5;

    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = fract(x);
    let fy = fract(y);

    let c00 = clamp_coords(x0, y0);
    let c10 = clamp_coords(x1, y0);
    let c01 = clamp_coords(x0, y1);
    let c11 = clamp_coords(x1, y1);

    // IMPORTANT: Do not scale by permeability here.
    // Obstacles are handled as reflective boundaries (non-dissipative) instead of porous damping.
    let v00 = raw_velocity_cell(c00.x, c00.y);
    let v10 = raw_velocity_cell(c10.x, c10.y);
    let v01 = raw_velocity_cell(c01.x, c01.y);
    let v11 = raw_velocity_cell(c11.x, c11.y);

    let v0 = mix(v00, v10, fx);
    let v1 = mix(v01, v11, fx);

    return mix(v0, v1, fy);
}

// Bilinear interpolation for dye concentration sampling
fn sample_dye(pos: vec2<f32>) -> vec2<f32> {
    // Dye is sampled in dye-grid coordinates (GAMMA_GRID_DIM).
    // Out-of-bounds contributes no dye.
    let min_pos = 0.5;
    let max_pos = f32(GAMMA_GRID_DIM) - 0.5;
    if (pos.x < min_pos || pos.x > max_pos || pos.y < min_pos || pos.y > max_pos) {
        return vec2<f32>(0.0, 0.0);
    }
    let p = clamp(pos, vec2<f32>(min_pos), vec2<f32>(max_pos));

    let x = p.x - 0.5;
    let y = p.y - 0.5;

    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = fract(x);
    let fy = fract(y);

    let c00 = dye_clamp_coords(x0, y0);
    let c10 = dye_clamp_coords(x1, y0);
    let c01 = dye_clamp_coords(x0, y1);
    let c11 = dye_clamp_coords(x1, y1);

    // Dye ignores obstacles: sample raw dye values.
    let d00 = dye_in[dye_grid_index(c00.x, c00.y)];
    let d10 = dye_in[dye_grid_index(c10.x, c10.y)];
    let d01 = dye_in[dye_grid_index(c01.x, c01.y)];
    let d11 = dye_in[dye_grid_index(c11.x, c11.y)];

    let d0 = mix(d00, d10, fx);
    let d1 = mix(d01, d11, fx);

    return mix(d0, d1, fy);
}

fn sample_trail(pos: vec2<f32>) -> vec4<f32> {
    // Trail is sampled in dye-grid coordinates (GAMMA_GRID_DIM).
    // Out-of-bounds contributes no trail.
    let min_pos = 0.5;
    let max_pos = f32(GAMMA_GRID_DIM) - 0.5;
    if (pos.x < min_pos || pos.x > max_pos || pos.y < min_pos || pos.y > max_pos) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let p = clamp(pos, vec2<f32>(min_pos), vec2<f32>(max_pos));

    let x = p.x - 0.5;
    let y = p.y - 0.5;

    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = fract(x);
    let fy = fract(y);

    let c00 = dye_clamp_coords(x0, y0);
    let c10 = dye_clamp_coords(x1, y0);
    let c01 = dye_clamp_coords(x0, y1);
    let c11 = dye_clamp_coords(x1, y1);

    let t00 = trail_in[dye_grid_index(c00.x, c00.y)];
    let t10 = trail_in[dye_grid_index(c10.x, c10.y)];
    let t01 = trail_in[dye_grid_index(c01.x, c01.y)];
    let t11 = trail_in[dye_grid_index(c11.x, c11.y)];

    let t0 = mix(t00, t10, fx);
    let t1 = mix(t01, t11, fx);
    return mix(t0, t1, fy);
}

@compute @workgroup_size(16, 16)
fn advect_trail(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    // Trails are stored/updated at environment resolution.
    if (x >= GAMMA_GRID_DIM || y >= GAMMA_GRID_DIM) {
        return;
    }

    let idx = dye_grid_index(x, y);
    let pos = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5);

    // Convert trail-grid position into fluid-grid coordinates for velocity sampling.
    let dye_to_fluid = f32(FLUID_GRID_SIZE) / f32(GAMMA_GRID_DIM);
    let fluid_to_dye = 1.0 / dye_to_fluid;
    let fluid_pos = pos * dye_to_fluid;

    let dt = clamp(fp_f32(FP_DT), 0.0, MAX_DT);
    let vel_fluid = clamp_vec2_len(sanitize_vec2(sample_velocity(fluid_pos)), MAX_VEL);
    let vel_dye = vel_fluid * fluid_to_dye;

    // Semi-Lagrangian backtrace
    let trace_pos = pos - vel_dye * dt;
    let advected = sample_trail(trace_pos);

    // Optional extra diffusion: blend towards a 4-neighbor blur from trail_in.
    // Keep subtle; most spreading should come from advection.
    let TRAIL_DIFFUSE_MIX: f32 = 0.005;
    let cx = i32(x);
    let cy = i32(y);
    let l = dye_clamp_coords(cx - 1, cy);
    let r = dye_clamp_coords(cx + 1, cy);
    let u = dye_clamp_coords(cx, cy - 1);
    let d = dye_clamp_coords(cx, cy + 1);
    let t_l = trail_in[dye_grid_index(l.x, l.y)];
    let t_r = trail_in[dye_grid_index(r.x, r.y)];
    let t_u = trail_in[dye_grid_index(u.x, u.y)];
    let t_d = trail_in[dye_grid_index(d.x, d.y)];
    let neighbor_blur = (t_l + t_r + t_u + t_d) * 0.25;

    let mixed = mix(advected, neighbor_blur, clamp(TRAIL_DIFFUSE_MIX, 0.0, 1.0));
    trail_out[idx] = mixed;
}

fn splat_falloff(dist: f32, radius: f32) -> f32 {
    if (dist >= radius) {
        return 0.0;
    }
    let x = 1.0 - (dist / radius);
    // Smooth-ish falloff.
    return x * x;
}

fn sample_display_bilinear_pos(pos: vec2<f32>) -> vec4<f32> {
    // pos is in texel space where the center of texel (x,y) is (x+0.5,y+0.5).
    // Manual bilinear using 4 exact texel loads.
    let x = pos.x - 0.5;
    let y = pos.y - 0.5;

    let x0i = i32(floor(x));
    let y0i = i32(floor(y));
    let x1i = x0i + 1;
    let y1i = y0i + 1;

    let fx = fract(x);
    let fy = fract(y);

    let x0 = clamp(x0i, 0, i32(FLUID_GRID_SIZE) - 1);
    let x1 = clamp(x1i, 0, i32(FLUID_GRID_SIZE) - 1);
    let y0 = clamp(y0i, 0, i32(FLUID_GRID_SIZE) - 1);
    let y1 = clamp(y1i, 0, i32(FLUID_GRID_SIZE) - 1);

    let c00 = textureLoad(display_tex_in, vec2<i32>(x0, y0), 0);
    let c10 = textureLoad(display_tex_in, vec2<i32>(x1, y0), 0);
    let c01 = textureLoad(display_tex_in, vec2<i32>(x0, y1), 0);
    let c11 = textureLoad(display_tex_in, vec2<i32>(x1, y1), 0);

    let c0 = mix(c00, c10, fx);
    let c1 = mix(c01, c11, fx);
    return mix(c0, c1, fy);
}

fn keys_cubic_weight(x: f32, a: f32) -> f32 {
    // Keys cubic kernel (a=-0.5 Catmull-Rom; a=-1 Mitchell-ish; smaller magnitude is softer)
    let t = abs(x);
    let t2 = t * t;
    let t3 = t2 * t;

    if (t < 1.0) {
        return (a + 2.0) * t3 - (a + 3.0) * t2 + 1.0;
    }
    if (t < 2.0) {
        return a * t3 - 5.0 * a * t2 + 8.0 * a * t - 4.0 * a;
    }
    return 0.0;
}

fn keys_cubic_weights(f: f32, a: f32) -> array<f32, 4> {
    // Weights for taps at offsets [-1, 0, 1, 2] relative to base cell.
    return array<f32, 4>(
        keys_cubic_weight(f + 1.0, a),
        keys_cubic_weight(f, a),
        keys_cubic_weight(1.0 - f, a),
        keys_cubic_weight(2.0 - f, a),
    );
}

fn sample_display_bicubic_pos(pos: vec2<f32>) -> vec4<f32> {
    // pos is in texel space where the center of texel (x,y) is (x+0.5,y+0.5).
    let p = pos - vec2<f32>(0.5);
    let base = vec2<i32>(i32(floor(p.x)), i32(floor(p.y)));
    let f = fract(p);

    // NOTE: This function is currently unused; kept for quick A/B testing.
    let a = -0.35;
    let wx = keys_cubic_weights(f.x, a);
    let wy = keys_cubic_weights(f.y, a);

    // Naga currently requires constant indices for arrays; unroll the 4x4 taps.
    let wx0 = wx[0];
    let wx1 = wx[1];
    let wx2 = wx[2];
    let wx3 = wx[3];
    let wy0 = wy[0];
    let wy1 = wy[1];
    let wy2 = wy[2];
    let wy3 = wy[3];

    let x0 = clamp(base.x - 1, 0, i32(FLUID_GRID_SIZE) - 1);
    let x1 = clamp(base.x + 0, 0, i32(FLUID_GRID_SIZE) - 1);
    let x2 = clamp(base.x + 1, 0, i32(FLUID_GRID_SIZE) - 1);
    let x3 = clamp(base.x + 2, 0, i32(FLUID_GRID_SIZE) - 1);
    let y0 = clamp(base.y - 1, 0, i32(FLUID_GRID_SIZE) - 1);
    let y1 = clamp(base.y + 0, 0, i32(FLUID_GRID_SIZE) - 1);
    let y2 = clamp(base.y + 1, 0, i32(FLUID_GRID_SIZE) - 1);
    let y3 = clamp(base.y + 2, 0, i32(FLUID_GRID_SIZE) - 1);

    var sum = vec4<f32>(0.0);

    // Row y0
    sum = sum + textureLoad(display_tex_in, vec2<i32>(x0, y0), 0) * (wx0 * wy0);
    sum = sum + textureLoad(display_tex_in, vec2<i32>(x1, y0), 0) * (wx1 * wy0);
    sum = sum + textureLoad(display_tex_in, vec2<i32>(x2, y0), 0) * (wx2 * wy0);
    sum = sum + textureLoad(display_tex_in, vec2<i32>(x3, y0), 0) * (wx3 * wy0);

    // Row y1
    sum = sum + textureLoad(display_tex_in, vec2<i32>(x0, y1), 0) * (wx0 * wy1);
    sum = sum + textureLoad(display_tex_in, vec2<i32>(x1, y1), 0) * (wx1 * wy1);
    sum = sum + textureLoad(display_tex_in, vec2<i32>(x2, y1), 0) * (wx2 * wy1);
    sum = sum + textureLoad(display_tex_in, vec2<i32>(x3, y1), 0) * (wx3 * wy1);

    // Row y2
    sum = sum + textureLoad(display_tex_in, vec2<i32>(x0, y2), 0) * (wx0 * wy2);
    sum = sum + textureLoad(display_tex_in, vec2<i32>(x1, y2), 0) * (wx1 * wy2);
    sum = sum + textureLoad(display_tex_in, vec2<i32>(x2, y2), 0) * (wx2 * wy2);
    sum = sum + textureLoad(display_tex_in, vec2<i32>(x3, y2), 0) * (wx3 * wy2);

    // Row y3
    sum = sum + textureLoad(display_tex_in, vec2<i32>(x0, y3), 0) * (wx0 * wy3);
    sum = sum + textureLoad(display_tex_in, vec2<i32>(x1, y3), 0) * (wx1 * wy3);
    sum = sum + textureLoad(display_tex_in, vec2<i32>(x2, y3), 0) * (wx2 * wy3);
    sum = sum + textureLoad(display_tex_in, vec2<i32>(x3, y3), 0) * (wx3 * wy3);

    return sum;
}

// ============================================================================
// COMPUTE KERNELS
// ============================================================================

// Advect velocity field (semi-Lagrangian)
@compute @workgroup_size(16, 16)
fn advect_velocity(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);
    if (is_effectively_solid(x, y)) {
        velocity_out[idx] = vec2<f32>(0.0, 0.0);
        return;
    }
    let pos = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5);

    // Read current velocity
    let dt = clamp(fp_f32(FP_DT), 0.0, MAX_DT);
    let vel0 = clamp_vec2_len(raw_velocity_cell(x, y), MAX_VEL);
    let vel = reflect_if_into_solid(x, y, vel0);

    // Backward trace
    // Scale velocity by grid size relative to 1.0 if needed, but here pixels = units.
    var trace_pos = pos - vel * dt;
    // If we traced into a solid cell, try a reflected trace instead (prevents energy loss at obstacles).
    if (obstacles_enabled() && is_solid_at_pos(trace_pos)) {
        let vel_reflect = reflect_if_into_solid(x, y, vel);
        trace_pos = pos - vel_reflect * dt;
    }

    // Sample velocity at traced position
    let advected_vel = sample_velocity(trace_pos);

    // Apply decay in a frame-rate independent way.
    // `FP_DECAY` is interpreted as a per-frame damping factor at 60 FPS.
    // For arbitrary dt, scale it as decay^(dt*60).
    let decay_per_frame = clamp(fp_f32(FP_DECAY), 0.0, 1.0);
    let decay_factor = pow(decay_per_frame, dt * 60.0);
    var out_v = sanitize_vec2(advected_vel * decay_factor);
    out_v = reflect_if_into_solid(x, y, out_v);
    velocity_out[idx] = clamp_vec2_len(out_v, MAX_VEL);
}

// 3. Compute divergence of a velocity field (reads velocity_in, writes divergence)
@compute @workgroup_size(16, 16)
fn compute_divergence(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);

    if (is_effectively_solid(x, y)) {
        divergence[idx] = 0.0;
        return;
    }

    // Solid boundary: treat boundary cells as non-divergent.
    if (x == 0u || x == FLUID_GRID_SIZE - 1u || y == 0u || y == FLUID_GRID_SIZE - 1u) {
        divergence[idx] = 0.0;
        return;
    }

    let xm = clamp(i32(x) - 1, 0, i32(FLUID_GRID_SIZE) - 1);
    let xp = clamp(i32(x) + 1, 0, i32(FLUID_GRID_SIZE) - 1);
    let ym = clamp(i32(y) - 1, 0, i32(FLUID_GRID_SIZE) - 1);
    let yp = clamp(i32(y) + 1, 0, i32(FLUID_GRID_SIZE) - 1);

    let v_l = reflect_if_into_solid(u32(xm), y, raw_velocity_cell(u32(xm), y));
    let v_r = reflect_if_into_solid(u32(xp), y, raw_velocity_cell(u32(xp), y));
    let v_b = reflect_if_into_solid(x, u32(ym), raw_velocity_cell(x, u32(ym)));
    let v_t = reflect_if_into_solid(x, u32(yp), raw_velocity_cell(x, u32(yp)));

    // Central differences; assume dx = 1.
    let div = 0.5 * ((v_r.x - v_l.x) + (v_t.y - v_b.y));
    divergence[idx] = div;
}

fn curl_at(x: u32, y: u32) -> f32 {
    let xm = clamp(i32(x) - 1, 0, i32(FLUID_GRID_SIZE) - 1);
    let xp = clamp(i32(x) + 1, 0, i32(FLUID_GRID_SIZE) - 1);
    let ym = clamp(i32(y) - 1, 0, i32(FLUID_GRID_SIZE) - 1);
    let yp = clamp(i32(y) + 1, 0, i32(FLUID_GRID_SIZE) - 1);

    let v_l = reflect_if_into_solid(u32(xm), y, raw_velocity_cell(u32(xm), y));
    let v_r = reflect_if_into_solid(u32(xp), y, raw_velocity_cell(u32(xp), y));
    let v_b = reflect_if_into_solid(x, u32(ym), raw_velocity_cell(x, u32(ym)));
    let v_t = reflect_if_into_solid(x, u32(yp), raw_velocity_cell(x, u32(yp)));

    let dvy_dx = 0.5 * (v_r.y - v_l.y);
    let dvx_dy = 0.5 * (v_t.x - v_b.x);
    return dvy_dx - dvx_dy;
}

// Vorticity confinement directly into velocity_out.
@compute @workgroup_size(16, 16)
fn vorticity_confinement(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);

    if (is_effectively_solid(x, y)) {
        velocity_out[idx] = vec2<f32>(0.0, 0.0);
        return;
    }

    // Skip a small border so curl sampling never uses clamped edge neighbors.
    // This avoids confinement injecting high-frequency noise along the walls.
    let border = 2u;
    if (x < border || x >= FLUID_GRID_SIZE - border || y < border || y >= FLUID_GRID_SIZE - border) {
        // IMPORTANT: still write output, otherwise border cells keep stale values.
        velocity_out[idx] = velocity_in[idx];
        return;
    }

    let w_l = abs(curl_at(x - 1u, y));
    let w_r = abs(curl_at(x + 1u, y));
    let w_b = abs(curl_at(x, y - 1u));
    let w_t = abs(curl_at(x, y + 1u));

    let grad = vec2<f32>(w_r - w_l, w_t - w_b) * 0.5;
    let mag = max(length(grad), 1e-5);
    let n = grad / mag;

    let w = curl_at(x, y);
    // Cut off tiny curl values to avoid amplifying numerical noise.
    if (abs(w) < 5e-4) {
        velocity_out[idx] = velocity_in[idx];
        return;
    }
    // Confinement force: epsilon * (n x w_k) in 2D
    // Equivalent: f = epsilon * vec2(n.y, -n.x) * w
    let epsilon = fp_vec4(FP_SPLAT).z;
    if (abs(epsilon) < 1e-6) {
        velocity_out[idx] = velocity_in[idx];
        return;
    }
    let f = vec2<f32>(n.y, -n.x) * (w * epsilon);

    // Clamp per-frame change to keep confinement from injecting high-frequency “static”.
    let dt = clamp(fp_f32(FP_DT), 0.0, MAX_DT);
    var dv = f * dt;
    let dv_len = length(dv);
    let max_dv = 2.0;
    if (dv_len > max_dv) {
        dv = dv * (max_dv / dv_len);
    }

    var v = sanitize_vec2(velocity_in[idx] + dv);
    v = reflect_if_into_solid(x, y, v);
    velocity_out[idx] = clamp_vec2_len(v, MAX_VEL);
}

// 4. Clear pressure (for init / each frame)
@compute @workgroup_size(16, 16)
fn clear_pressure(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);
    pressure_out[idx] = 0.0;
}

// 5. Jacobi iteration for pressure solve (Poisson): ∇²p = div
@compute @workgroup_size(16, 16)
fn jacobi_pressure(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);

    if (is_effectively_solid(x, y)) {
        pressure_out[idx] = 0.0;
        return;
    }

    // Neumann boundary (solid walls): dp/dn = 0.
    // Implemented by mirroring the edge pressure to the outside.
    var p_l = pressure_in[idx];
    if (x > 0u && !is_effectively_solid(x - 1u, y)) {
        p_l = pressure_in[grid_index(x - 1u, y)];
    }
    var p_r = pressure_in[idx];
    if (x + 1u < FLUID_GRID_SIZE && !is_effectively_solid(x + 1u, y)) {
        p_r = pressure_in[grid_index(x + 1u, y)];
    }
    var p_b = pressure_in[idx];
    if (y > 0u && !is_effectively_solid(x, y - 1u)) {
        p_b = pressure_in[grid_index(x, y - 1u)];
    }
    var p_t = pressure_in[idx];
    if (y + 1u < FLUID_GRID_SIZE && !is_effectively_solid(x, y + 1u)) {
        p_t = pressure_in[grid_index(x, y + 1u)];
    }

    // Jacobi: p = (sum_neighbors - div) / 4, assuming dx = 1.
    let div = divergence[idx];
    if (is_bad_f32(div) || is_bad_f32(p_l) || is_bad_f32(p_r) || is_bad_f32(p_b) || is_bad_f32(p_t)) {
        pressure_out[idx] = 0.0;
        return;
    }
    pressure_out[idx] = (p_l + p_r + p_b + p_t - div) * 0.25;
}

// 6. Subtract pressure gradient from velocity to make it divergence-free.
@compute @workgroup_size(16, 16)
fn subtract_gradient(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);

    if (is_effectively_solid(x, y)) {
        velocity_out[idx] = vec2<f32>(0.0, 0.0);
        return;
    }

    // Use the same Neumann pressure boundary treatment as the Jacobi solve.
    var p_l = pressure_in[idx];
    if (x > 0u && !is_effectively_solid(x - 1u, y)) {
        p_l = pressure_in[grid_index(x - 1u, y)];
    }
    var p_r = pressure_in[idx];
    if (x + 1u < FLUID_GRID_SIZE && !is_effectively_solid(x + 1u, y)) {
        p_r = pressure_in[grid_index(x + 1u, y)];
    }
    var p_b = pressure_in[idx];
    if (y > 0u && !is_effectively_solid(x, y - 1u)) {
        p_b = pressure_in[grid_index(x, y - 1u)];
    }
    var p_t = pressure_in[idx];
    if (y + 1u < FLUID_GRID_SIZE && !is_effectively_solid(x, y + 1u)) {
        p_t = pressure_in[grid_index(x, y + 1u)];
    }

    let grad = vec2<f32>(p_r - p_l, p_t - p_b) * 0.5;
    var v = sanitize_vec2(velocity_in[idx]) - grad;
    v = reflect_if_into_solid(x, y, v);
    velocity_out[idx] = clamp_vec2_len(sanitize_vec2(v), MAX_VEL);
}

// Add forces to velocity field
@compute @workgroup_size(16, 16)
fn add_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);

    if (is_effectively_solid(x, y)) {
        velocity_out[idx] = vec2<f32>(0.0, 0.0);
        return;
    }

    // Add user forces (clamped) scaled by dt (clamped to avoid instability on dt spikes)
    let dt = clamp(fp_f32(FP_DT), 0.0, MAX_DT);
    let v0 = sanitize_vec2(velocity_in[idx]);
    let f_user = clamp_vec2_len(sanitize_vec2(fluid_forces[idx]), MAX_FORCE);
    var v = sanitize_vec2(v0 + f_user * dt);

    // Then apply slope steering as a pure direction change (preserve |v|).
    v = gamma_slope_steer_velocity(x, y, v, dt);
    v = reflect_if_into_solid(x, y, v);
    velocity_out[idx] = clamp_vec2_len(v, MAX_VEL);
}

// Velocity diffusion / viscosity (explicit step): v <- v + nu*dt*∇²v
// This damps 1-cell oscillations (checkerboard / “TV static”) without adding new buffers.
@compute @workgroup_size(16, 16)
fn diffuse_velocity(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);

    if (is_effectively_solid(x, y)) {
        velocity_out[idx] = vec2<f32>(0.0, 0.0);
        return;
    }

    // Keep boundaries unchanged; boundary enforcement pass handles walls.
    if (x == 0u || x == FLUID_GRID_SIZE - 1u || y == 0u || y == FLUID_GRID_SIZE - 1u) {
        velocity_out[idx] = velocity_in[idx];
        return;
    }

    let v_c = raw_velocity_cell(x, y);
    let v_l = raw_velocity_cell(x - 1u, y);
    let v_r = raw_velocity_cell(x + 1u, y);
    let v_b = raw_velocity_cell(x, y - 1u);
    let v_t = raw_velocity_cell(x, y + 1u);

    let lap = (v_l + v_r + v_b + v_t) - 4.0 * v_c;

    // nu is in "cells^2 / second" with dx=1; lower values reduce blur.
    let nu = max(fp_vec4(FP_SPLAT).w, 0.0);
    let dt = clamp(fp_f32(FP_DT), 0.0, MAX_DT);
    let a = nu * dt;
    var v = sanitize_vec2(v_c + a * lap);
    v = reflect_if_into_solid(x, y, v);
    velocity_out[idx] = clamp_vec2_len(v, MAX_VEL);
}

// 4. Enforce boundary conditions (free-slip: zero normal component)
@compute @workgroup_size(16, 16)
fn enforce_boundaries(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);

    // Solid obstacles: no-slip (zero velocity)
    if (is_effectively_solid(x, y)) {
        velocity_out[idx] = vec2<f32>(0.0, 0.0);
        return;
    }

    // Free-slip solid walls using a simple “ghost cell” style update:
    // - normal component is zero at the wall
    // - tangential component is copied from the adjacent interior cell
    var v = sanitize_vec2(velocity_in[idx]);

    // Reflective boundaries: bounce the normal component to conserve energy.
    // User requested: no energy loss on obstacles.
    let bounce_damping = 1.0; // perfectly elastic

    // Left wall (x=0): reflect horizontal velocity (bounce inward)
    if (x == 0u) {
        v.x = -v.x * bounce_damping;
    }
    // Right wall (x=max): reflect horizontal velocity (bounce inward)
    if (x == FLUID_GRID_SIZE - 1u) {
        v.x = -v.x * bounce_damping;
    }
    // Bottom wall (y=0): reflect vertical velocity (bounce upward)
    if (y == 0u) {
        v.y = -v.y * bounce_damping;
    }
    // Top wall (y=max): reflect vertical velocity (bounce downward)
    if (y == FLUID_GRID_SIZE - 1u) {
        v.y = -v.y * bounce_damping;
    }

    // Also reflect against interior solid obstacles (non-dissipative).
    v = reflect_if_into_solid(x, y, v);
    velocity_out[idx] = clamp_vec2_len(sanitize_vec2(v), MAX_VEL);
}

// 4. Clear velocities (for initialization)
@compute @workgroup_size(16, 16)
fn clear_velocity(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);
    velocity_out[idx] = vec2<f32>(0.0);
}

// 5. Copy buffer (for ping-pong)
@compute @workgroup_size(16, 16)
fn copy_velocity(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);
    velocity_out[idx] = velocity_in[idx];
}

// Advect the display texture forward using the *current* velocity field.
// This creates a feedback loop: the texture evolves from the previous frame.
@compute @workgroup_size(16, 16)
fn advect_display_texture(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);
    let pos = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5);
    let vel = velocity_in[idx];

    // Semi-Lagrangian backtrace in grid/texel space.
    let dt = fp_f32(FP_DT);
    let trace_pos = pos - vel * dt;

    let current = textureLoad(display_tex_in, vec2<i32>(i32(x), i32(y)), 0);

    // Avoid repeatedly resampling (which can slowly introduce/boost high-frequency noise)
    // when the flow is nearly stationary: just copy the exact texel.
    let disp = length(vel) * dt;
    if (disp < 0.01) {
        textureStore(display_tex_out, vec2<i32>(i32(x), i32(y)), current);
        return;
    }

    // Clamp to texel centers to stay in-bounds for the 4x4 footprint.
    let min_pos = 1.5;
    let max_pos = f32(FLUID_GRID_SIZE) - 1.5;
    let p = clamp(trace_pos, vec2<f32>(min_pos), vec2<f32>(max_pos));

    let advected = sample_display_bilinear_pos(p);
    // Apply only 10% of the advection over the existing texture.
    let c = clamp(mix(current, advected, 0.1), vec4<f32>(0.0), vec4<f32>(1.0));
    textureStore(display_tex_out, vec2<i32>(i32(x), i32(y)), c);
}
