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
const MAX_FORCE: f32 = 20000.0;   // Clamp injected force magnitude per cell
const MAX_VEL: f32 = 200.0;       // Clamp velocity magnitude per cell

// Gamma-grid-driven obstacles
// NOTE: Keep these as constants for now to avoid expanding uniform layouts.
const OBSTACLES_ENABLED: bool = false;
// Obstacle mapping: use gamma directly in [0,1] (optionally shaped by a power curve).
// 0 = no obstacle, 1 = fully solid.
const OBSTACLE_GAMMA_POWER: f32 = 0.0;

// Optional: drive fluid using the gamma slope (heightmap-style downhill flow).
// This uses the gradient of gamma (in fluid-cell space) as a force vector.
const SLOPE_FORCE_ENABLED: bool = true;
const SLOPE_FORCE_SCALE: f32 = 1000000.0;

// ============================================================================
// BINDINGS
// ============================================================================

// Velocity ping-pong buffers (vec2 per cell)
@group(0) @binding(0) var<storage, read> velocity_in: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> velocity_out: array<vec2<f32>>;

// Gamma grid (terrain) sampled as an obstacle field
@group(0) @binding(2) var<storage, read> gamma_grid: array<f32>;

// Intermediate force vectors buffer (vec2 per cell) - agents write propeller forces here
@group(0) @binding(16) var<storage, read_write> fluid_force_vectors: array<vec2<f32>>;

// Combined forces buffer (vec2 per cell) - inject_test_force writes combined forces here
@group(0) @binding(7) var<storage, read_write> fluid_forces: array<vec2<f32>>;

// Pressure ping-pong + divergence (f32 per cell)
@group(0) @binding(4) var<storage, read> pressure_in: array<f32>;
@group(0) @binding(5) var<storage, read_write> pressure_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> divergence: array<f32>;

// Dye concentration ping-pong buffers (f32 per cell) - for flow visualization
@group(0) @binding(8) var<storage, read> dye_in: array<f32>;
@group(0) @binding(9) var<storage, read_write> dye_out: array<f32>;

// Parameters
struct FluidParams {
    time: f32,
    dt: f32,
    decay: f32,
    grid_size: u32,
    // mouse.xy in grid coords (0..grid), mouse.zw = mouse velocity in grid units/sec
    mouse: vec4<f32>,
    // splat.x = radius (cells), splat.y = force scale,
    // splat.z = vorticity confinement strength (0 disables), splat.w = mouse_down (0/1)
    splat: vec4<f32>,
}
@group(0) @binding(3) var<uniform> params: FluidParams;

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
    // Store dye concentration in x component, zero in y
    fluid_forces[idx] = vec2<f32>(dye_in[idx], 0.0);
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
    let t = u32(params.time * 1000.0);
    let r1 = rand01(idx ^ t ^ 0xA511E9B3u);
    let r2 = rand01(idx ^ t ^ 0x63D83595u);
    let v = vec2<f32>(r1 * 2.0 - 1.0, r2 * 2.0 - 1.0);

    let strength = 1.0;
    fluid_forces[idx] = v * strength;
}

// Inject dye at locations where propellers are active (where forces are non-zero)
@compute @workgroup_size(16, 16)
fn inject_dye(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);

    if (is_effectively_solid(x, y)) {
        dye_out[idx] = 0.0;
        return;
    }

    // Check if there's a propeller force at this location
    let force = fluid_force_vectors[idx];
    let force_magnitude = length(force);

    // If there's significant force, inject dye (concentration = 1.0)
    // Otherwise, let existing dye decay slightly
    var current_dye = dye_in[idx];

    if (force_magnitude > 1.0) {
        // Strong dye injection at propeller locations
        current_dye = min(current_dye + 1.0, 1.0);
    } else {
        // Slight decay over time
        current_dye *= 0.995;
    }

    let perm = permeability(x, y);
    dye_out[idx] = current_dye * perm;
}

// Advect dye concentration using semi-Lagrangian method (same as velocity advection)
@compute @workgroup_size(16, 16)
fn advect_dye(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);
    if (is_effectively_solid(x, y)) {
        dye_out[idx] = 0.0;
        return;
    }
    let pos = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5);

    // Read current velocity
    let dt = clamp(params.dt, 0.0, MAX_DT);
    let vel = clamp_vec2_len(sanitize_vec2(velocity_in[idx]), MAX_VEL);

    // Backward trace - follow the flow backwards to find where dye came from
    let trace_pos = pos - vel * dt;

    // Sample dye concentration at traced position using bilinear interpolation
    let advected_dye = sample_dye(trace_pos);

    // Apply slight decay to make dye fade over time
    let decay_factor = 0.998;

    let perm = permeability(x, y);
    dye_out[idx] = clamp(advected_dye * decay_factor * perm, 0.0, 1.0);
}

// Clear dye buffer (for reset)
@compute @workgroup_size(16, 16)
fn clear_dye(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);
    dye_out[idx] = 0.0;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

fn grid_index(x: u32, y: u32) -> u32 {
    return y * FLUID_GRID_SIZE + x;
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
    // Map fluid cell (0..FLUID_GRID_SIZE) into gamma grid (0..GAMMA_GRID_DIM)
    // Both cover the same world extents.
    let fx = (f32(x) + 0.5) * f32(GAMMA_GRID_DIM) / f32(FLUID_GRID_SIZE);
    let fy = (f32(y) + 0.5) * f32(GAMMA_GRID_DIM) / f32(FLUID_GRID_SIZE);
    let max_idx_f = f32(GAMMA_GRID_DIM - 1u);
    let gx = u32(clamp(fx, 0.0, max_idx_f));
    let gy = u32(clamp(fy, 0.0, max_idx_f));
    return gamma_height_at_idx(gamma_index(gx, gy));
}

fn gamma_slope_force(x: u32, y: u32) -> vec2<f32> {
    if (!SLOPE_FORCE_ENABLED) {
        return vec2<f32>(0.0, 0.0);
    }

    // Use the simulation-provided slope vector field (already includes gamma+alpha+beta contributions).
    // Interpreted as a height gradient (points uphill); apply downhill force by negating it.
    let fx = (f32(x) + 0.5) * f32(GAMMA_GRID_DIM) / f32(FLUID_GRID_SIZE);
    let fy = (f32(y) + 0.5) * f32(GAMMA_GRID_DIM) / f32(FLUID_GRID_SIZE);
    let max_idx_f = f32(GAMMA_GRID_DIM - 1u);
    let gx = u32(clamp(fx, 0.0, max_idx_f));
    let gy = u32(clamp(fy, 0.0, max_idx_f));
    let idx = gamma_index(gx, gy);

    let slope_grad = gamma_slope_at_idx(idx);
    return sanitize_vec2(slope_grad) * SLOPE_FORCE_SCALE;
}

fn obstacle_strength(x: u32, y: u32) -> f32 {
    // 0..1 where 1 means fully solid.
    if (!obstacles_enabled()) {
        return 0.0;
    }
    let g = gamma_at_fluid_cell(x, y);
    return pow(clamp(g, 0.0, 1.0), OBSTACLE_GAMMA_POWER);
}

fn permeability(x: u32, y: u32) -> f32 {
    // 1..0 where 0 means fully solid.
    return 1.0 - obstacle_strength(x, y);
}

fn is_effectively_solid(x: u32, y: u32) -> bool {
    return obstacle_strength(x, y) > 0.999;
}

fn clamp_coords(x: i32, y: i32) -> vec2<u32> {
    let cx = clamp(x, 0, i32(FLUID_GRID_SIZE) - 1);
    let cy = clamp(y, 0, i32(FLUID_GRID_SIZE) - 1);
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

    let v00 = velocity_in[grid_index(c00.x, c00.y)] * permeability(c00.x, c00.y);
    let v10 = velocity_in[grid_index(c10.x, c10.y)] * permeability(c10.x, c10.y);
    let v01 = velocity_in[grid_index(c01.x, c01.y)] * permeability(c01.x, c01.y);
    let v11 = velocity_in[grid_index(c11.x, c11.y)] * permeability(c11.x, c11.y);

    let v0 = mix(v00, v10, fx);
    let v1 = mix(v01, v11, fx);

    return mix(v0, v1, fy);
}

// Bilinear interpolation for dye concentration sampling
fn sample_dye(pos: vec2<f32>) -> f32 {
    // Match velocity advection boundary handling:
    // out-of-bounds sampling contributes no dye.
    let min_pos = 0.5;
    let max_pos = f32(FLUID_GRID_SIZE) - 0.5;
    if (pos.x < min_pos || pos.x > max_pos || pos.y < min_pos || pos.y > max_pos) {
        return 0.0;
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

    let d00 = dye_in[grid_index(c00.x, c00.y)] * permeability(c00.x, c00.y);
    let d10 = dye_in[grid_index(c10.x, c10.y)] * permeability(c10.x, c10.y);
    let d01 = dye_in[grid_index(c01.x, c01.y)] * permeability(c01.x, c01.y);
    let d11 = dye_in[grid_index(c11.x, c11.y)] * permeability(c11.x, c11.y);

    let d0 = mix(d00, d10, fx);
    let d1 = mix(d01, d11, fx);

    return mix(d0, d1, fy);
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
    let dt = clamp(params.dt, 0.0, MAX_DT);
    let vel = clamp_vec2_len(sanitize_vec2(velocity_in[idx]), MAX_VEL);

    // Backward trace
    // Scale velocity by grid size relative to 1.0 if needed, but here pixels = units.
    let trace_pos = pos - vel * dt;

    // Sample velocity at traced position
    let advected_vel = sample_velocity(trace_pos);

    // Apply decay in a frame-rate independent way.
    // `params.decay` is interpreted as a per-frame damping factor at 60 FPS.
    // For arbitrary dt, scale it as decay^(dt*60).
    let decay_factor = pow(0.999, dt * 60.0);  // Reduced from params.decay (0.9995)
    let perm = permeability(x, y);
    velocity_out[idx] = clamp_vec2_len(sanitize_vec2(advected_vel * decay_factor), MAX_VEL) * perm;
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

    let v_l = sanitize_vec2(velocity_in[grid_index(u32(xm), y)]) * permeability(u32(xm), y);
    let v_r = sanitize_vec2(velocity_in[grid_index(u32(xp), y)]) * permeability(u32(xp), y);
    let v_b = sanitize_vec2(velocity_in[grid_index(x, u32(ym))]) * permeability(x, u32(ym));
    let v_t = sanitize_vec2(velocity_in[grid_index(x, u32(yp))]) * permeability(x, u32(yp));

    // Central differences; assume dx = 1.
    let div = 0.5 * ((v_r.x - v_l.x) + (v_t.y - v_b.y));
    divergence[idx] = div;
}

fn curl_at(x: u32, y: u32) -> f32 {
    let xm = clamp(i32(x) - 1, 0, i32(FLUID_GRID_SIZE) - 1);
    let xp = clamp(i32(x) + 1, 0, i32(FLUID_GRID_SIZE) - 1);
    let ym = clamp(i32(y) - 1, 0, i32(FLUID_GRID_SIZE) - 1);
    let yp = clamp(i32(y) + 1, 0, i32(FLUID_GRID_SIZE) - 1);

    let v_l = velocity_in[grid_index(u32(xm), y)] * permeability(u32(xm), y);
    let v_r = velocity_in[grid_index(u32(xp), y)] * permeability(u32(xp), y);
    let v_b = velocity_in[grid_index(x, u32(ym))] * permeability(x, u32(ym));
    let v_t = velocity_in[grid_index(x, u32(yp))] * permeability(x, u32(yp));

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
    let epsilon = params.splat.z;
    if (abs(epsilon) < 1e-6) {
        velocity_out[idx] = velocity_in[idx];
        return;
    }
    let f = vec2<f32>(n.y, -n.x) * (w * epsilon);

    // Clamp per-frame change to keep confinement from injecting high-frequency “static”.
    let dt = clamp(params.dt, 0.0, MAX_DT);
    var dv = f * dt;
    let dv_len = length(dv);
    let max_dv = 2.0;
    if (dv_len > max_dv) {
        dv = dv * (max_dv / dv_len);
    }

    let perm = permeability(x, y);
    velocity_out[idx] = clamp_vec2_len(sanitize_vec2(velocity_in[idx] + dv), MAX_VEL) * perm;
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
    if (x > 0u) {
        let w = permeability(x - 1u, y);
        let pn = pressure_in[grid_index(x - 1u, y)];
        p_l = mix(pressure_in[idx], pn, w);
    }
    var p_r = pressure_in[idx];
    if (x + 1u < FLUID_GRID_SIZE) {
        let w = permeability(x + 1u, y);
        let pn = pressure_in[grid_index(x + 1u, y)];
        p_r = mix(pressure_in[idx], pn, w);
    }
    var p_b = pressure_in[idx];
    if (y > 0u) {
        let w = permeability(x, y - 1u);
        let pn = pressure_in[grid_index(x, y - 1u)];
        p_b = mix(pressure_in[idx], pn, w);
    }
    var p_t = pressure_in[idx];
    if (y + 1u < FLUID_GRID_SIZE) {
        let w = permeability(x, y + 1u);
        let pn = pressure_in[grid_index(x, y + 1u)];
        p_t = mix(pressure_in[idx], pn, w);
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
    if (x > 0u) {
        let w = permeability(x - 1u, y);
        let pn = pressure_in[grid_index(x - 1u, y)];
        p_l = mix(pressure_in[idx], pn, w);
    }
    var p_r = pressure_in[idx];
    if (x + 1u < FLUID_GRID_SIZE) {
        let w = permeability(x + 1u, y);
        let pn = pressure_in[grid_index(x + 1u, y)];
        p_r = mix(pressure_in[idx], pn, w);
    }
    var p_b = pressure_in[idx];
    if (y > 0u) {
        let w = permeability(x, y - 1u);
        let pn = pressure_in[grid_index(x, y - 1u)];
        p_b = mix(pressure_in[idx], pn, w);
    }
    var p_t = pressure_in[idx];
    if (y + 1u < FLUID_GRID_SIZE) {
        let w = permeability(x, y + 1u);
        let pn = pressure_in[grid_index(x, y + 1u)];
        p_t = mix(pressure_in[idx], pn, w);
    }

    let grad = vec2<f32>(p_r - p_l, p_t - p_b) * 0.5;
    let perm = permeability(x, y);
    var v = (sanitize_vec2(velocity_in[idx]) * perm) - grad;

    // No-slip solid boundary: zero velocity at the walls (stronger containment).
    if (x == 0u || x == FLUID_GRID_SIZE - 1u) {
        v = vec2<f32>(0.0, 0.0);
    }
    if (y == 0u || y == FLUID_GRID_SIZE - 1u) {
        v = vec2<f32>(0.0, 0.0);
    }

    velocity_out[idx] = clamp_vec2_len(sanitize_vec2(v), MAX_VEL) * perm;
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

    // Add force scaled by dt (clamped to avoid instability on dt spikes)
    let dt = clamp(params.dt, 0.0, MAX_DT);
    let perm = permeability(x, y);
    let v0 = sanitize_vec2(velocity_in[idx]) * perm;
    let f_user = sanitize_vec2(fluid_forces[idx]);
    let f_slope = gamma_slope_force(x, y);
    let f = clamp_vec2_len(sanitize_vec2(f_user + f_slope), MAX_FORCE);
    velocity_out[idx] = clamp_vec2_len(sanitize_vec2((v0 + f * dt) * perm), MAX_VEL);
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

    let perm = permeability(x, y);
    let v_c = sanitize_vec2(velocity_in[idx]) * perm;
    let v_l = sanitize_vec2(velocity_in[grid_index(x - 1u, y)]) * permeability(x - 1u, y);
    let v_r = sanitize_vec2(velocity_in[grid_index(x + 1u, y)]) * permeability(x + 1u, y);
    let v_b = sanitize_vec2(velocity_in[grid_index(x, y - 1u)]) * permeability(x, y - 1u);
    let v_t = sanitize_vec2(velocity_in[grid_index(x, y + 1u)]) * permeability(x, y + 1u);

    let lap = (v_l + v_r + v_b + v_t) - 4.0 * v_c;

    // nu is in "cells^2 / second" with dx=1; lower values reduce blur.
    let nu = 0.05;  // Reduced from 0.35 for less dissipation
    let dt = clamp(params.dt, 0.0, MAX_DT);
    let a = nu * dt;
    velocity_out[idx] = clamp_vec2_len(sanitize_vec2(v_c + a * lap), MAX_VEL) * perm;
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

    // Left / right walls: set v.x = 0 and copy v.y from interior neighbor.
    // At walls, enforce zero velocity completely (no-slip boundaries)
    if (x == 0u || x == FLUID_GRID_SIZE - 1u || y == 0u || y == FLUID_GRID_SIZE - 1u) {
        v = vec2<f32>(0.0, 0.0);
    }

    let perm = permeability(x, y);
    velocity_out[idx] = clamp_vec2_len(sanitize_vec2(v), MAX_VEL) * perm;
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
    let trace_pos = pos - vel * params.dt;

    let current = textureLoad(display_tex_in, vec2<i32>(i32(x), i32(y)), 0);

    // Avoid repeatedly resampling (which can slowly introduce/boost high-frequency noise)
    // when the flow is nearly stationary: just copy the exact texel.
    let disp = length(vel) * params.dt;
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
