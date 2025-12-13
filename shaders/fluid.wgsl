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

const FLUID_GRID_SIZE: u32 = 128u;
const FLUID_GRID_CELLS: u32 = 16384u; // 128 * 128

// ============================================================================
// BINDINGS
// ============================================================================

// Velocity ping-pong buffers (vec2 per cell)
@group(0) @binding(0) var<storage, read> velocity_in: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> velocity_out: array<vec2<f32>>;

// Force input (vec2 per cell) - will be written by test force generator
@group(0) @binding(2) var<storage, read_write> forces: array<vec2<f32>>;

// Pressure ping-pong + divergence (f32 per cell)
@group(0) @binding(4) var<storage, read> pressure_in: array<f32>;
@group(0) @binding(5) var<storage, read_write> pressure_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> divergence: array<f32>;

// Force vectors buffer (vec2 per cell) - populated by propeller forces each frame
@group(0) @binding(7) var<storage, read_write> force_vectors: array<vec2<f32>>;


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
    forces[idx] = velocity_in[idx];
}

@compute @workgroup_size(16, 16)
fn clear_forces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);
    forces[idx] = vec2<f32>(0.0, 0.0);
}

@compute @workgroup_size(16, 16)
fn clear_force_vectors(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);
    force_vectors[idx] = vec2<f32>(0.0, 0.0);
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

fn grid_index(x: u32, y: u32) -> u32 {
    return y * FLUID_GRID_SIZE + x;
}

fn clamp_coords(x: i32, y: i32) -> vec2<u32> {
    let cx = clamp(x, 0, i32(FLUID_GRID_SIZE) - 1);
    let cy = clamp(y, 0, i32(FLUID_GRID_SIZE) - 1);
    return vec2<u32>(u32(cx), u32(cy));
}

// Bilinear interpolation for velocity sampling
fn sample_velocity(pos: vec2<f32>) -> vec2<f32> {
    // Clamp sampling to the interior so advection never repeatedly samples the
    // outermost edge cells (which tends to create persistent high-frequency noise).
    let min_pos = 1.5;
    let max_pos = f32(FLUID_GRID_SIZE) - 1.5;
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

    let v00 = velocity_in[grid_index(c00.x, c00.y)];
    let v10 = velocity_in[grid_index(c10.x, c10.y)];
    let v01 = velocity_in[grid_index(c01.x, c01.y)];
    let v11 = velocity_in[grid_index(c11.x, c11.y)];

    let v0 = mix(v00, v10, fx);
    let v1 = mix(v01, v11, fx);

    return mix(v0, v1, fy);
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

// 1. Generate external forces from static force vectors buffer
@compute @workgroup_size(16, 16)
fn generate_test_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);

    // Read force from the static force vectors buffer
    // (Populated by agent propellers in simulation shader)
    // Boost forces 100x so they survive fluid dissipation
    forces[idx] = force_vectors[idx] * 100.0;
}

// 2. Advect velocity field (semi-Lagrangian, no forces yet)
@compute @workgroup_size(16, 16)
fn advect_velocity(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);
    let pos = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5);

    // Read current velocity
    let vel = velocity_in[idx];

    // Backward trace
    // Scale velocity by grid size relative to 1.0 if needed, but here pixels = units.
    let trace_pos = pos - vel * params.dt;

    // Sample velocity at traced position
    let advected_vel = sample_velocity(trace_pos);

    // Apply decay in a frame-rate independent way.
    // `params.decay` is interpreted as a per-frame damping factor at 60 FPS.
    // For arbitrary dt, scale it as decay^(dt*60).
    let decay_factor = pow(0.999, params.dt * 60.0);  // Reduced from params.decay (0.9995)
    velocity_out[idx] = advected_vel * decay_factor;
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

    // Solid boundary: treat boundary cells as non-divergent.
    if (x == 0u || x == FLUID_GRID_SIZE - 1u || y == 0u || y == FLUID_GRID_SIZE - 1u) {
        divergence[idx] = 0.0;
        return;
    }

    let xm = clamp(i32(x) - 1, 0, i32(FLUID_GRID_SIZE) - 1);
    let xp = clamp(i32(x) + 1, 0, i32(FLUID_GRID_SIZE) - 1);
    let ym = clamp(i32(y) - 1, 0, i32(FLUID_GRID_SIZE) - 1);
    let yp = clamp(i32(y) + 1, 0, i32(FLUID_GRID_SIZE) - 1);

    let v_l = velocity_in[grid_index(u32(xm), y)];
    let v_r = velocity_in[grid_index(u32(xp), y)];
    let v_b = velocity_in[grid_index(x, u32(ym))];
    let v_t = velocity_in[grid_index(x, u32(yp))];

    // Central differences; assume dx = 1.
    let div = 0.5 * ((v_r.x - v_l.x) + (v_t.y - v_b.y));
    divergence[idx] = div;
}

fn curl_at(x: u32, y: u32) -> f32 {
    let xm = clamp(i32(x) - 1, 0, i32(FLUID_GRID_SIZE) - 1);
    let xp = clamp(i32(x) + 1, 0, i32(FLUID_GRID_SIZE) - 1);
    let ym = clamp(i32(y) - 1, 0, i32(FLUID_GRID_SIZE) - 1);
    let yp = clamp(i32(y) + 1, 0, i32(FLUID_GRID_SIZE) - 1);

    let v_l = velocity_in[grid_index(u32(xm), y)];
    let v_r = velocity_in[grid_index(u32(xp), y)];
    let v_b = velocity_in[grid_index(x, u32(ym))];
    let v_t = velocity_in[grid_index(x, u32(yp))];

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
    var dv = f * params.dt;
    let dv_len = length(dv);
    let max_dv = 2.0;
    if (dv_len > max_dv) {
        dv = dv * (max_dv / dv_len);
    }

    velocity_out[idx] = velocity_in[idx] + dv;
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

    // Neumann boundary (solid walls): dp/dn = 0.
    // Implemented by mirroring the edge pressure to the outside.
    var p_l = pressure_in[idx];
    if (x > 0u) {
        p_l = pressure_in[grid_index(x - 1u, y)];
    }
    var p_r = pressure_in[idx];
    if (x + 1u < FLUID_GRID_SIZE) {
        p_r = pressure_in[grid_index(x + 1u, y)];
    }
    var p_b = pressure_in[idx];
    if (y > 0u) {
        p_b = pressure_in[grid_index(x, y - 1u)];
    }
    var p_t = pressure_in[idx];
    if (y + 1u < FLUID_GRID_SIZE) {
        p_t = pressure_in[grid_index(x, y + 1u)];
    }

    // Jacobi: p = (sum_neighbors - div) / 4, assuming dx = 1.
    pressure_out[idx] = (p_l + p_r + p_b + p_t - divergence[idx]) * 0.25;
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

    // Use the same Neumann pressure boundary treatment as the Jacobi solve.
    var p_l = pressure_in[idx];
    if (x > 0u) {
        p_l = pressure_in[grid_index(x - 1u, y)];
    }
    var p_r = pressure_in[idx];
    if (x + 1u < FLUID_GRID_SIZE) {
        p_r = pressure_in[grid_index(x + 1u, y)];
    }
    var p_b = pressure_in[idx];
    if (y > 0u) {
        p_b = pressure_in[grid_index(x, y - 1u)];
    }
    var p_t = pressure_in[idx];
    if (y + 1u < FLUID_GRID_SIZE) {
        p_t = pressure_in[grid_index(x, y + 1u)];
    }

    let grad = vec2<f32>(p_r - p_l, p_t - p_b) * 0.5;
    var v = velocity_in[idx] - grad;

    // Free-slip solid boundary: zero normal component at the walls.
    if (x == 0u || x == FLUID_GRID_SIZE - 1u) {
        v.x = 0.0;
    }
    if (y == 0u || y == FLUID_GRID_SIZE - 1u) {
        v.y = 0.0;
    }

    velocity_out[idx] = v;
}

// 3. Add forces to velocity field
@compute @workgroup_size(16, 16)
fn add_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);

    // Add force scaled by dt
    velocity_out[idx] = velocity_in[idx] + forces[idx] * params.dt;
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

    // Keep boundaries unchanged; boundary enforcement pass handles walls.
    if (x == 0u || x == FLUID_GRID_SIZE - 1u || y == 0u || y == FLUID_GRID_SIZE - 1u) {
        velocity_out[idx] = velocity_in[idx];
        return;
    }

    let v_c = velocity_in[idx];
    let v_l = velocity_in[grid_index(x - 1u, y)];
    let v_r = velocity_in[grid_index(x + 1u, y)];
    let v_b = velocity_in[grid_index(x, y - 1u)];
    let v_t = velocity_in[grid_index(x, y + 1u)];

    let lap = (v_l + v_r + v_b + v_t) - 4.0 * v_c;

    // nu is in "cells^2 / second" with dx=1; lower values reduce blur.
    let nu = 0.05;  // Reduced from 0.35 for less dissipation
    let a = nu * params.dt;
    velocity_out[idx] = v_c + a * lap;
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

    // Free-slip solid walls using a simple “ghost cell” style update:
    // - normal component is zero at the wall
    // - tangential component is copied from the adjacent interior cell
    var v = velocity_in[idx];

    // Left / right walls: set v.x = 0 and copy v.y from interior neighbor.
    if (x == 0u) {
        v.x = 0.0;
        v.y = velocity_in[grid_index(1u, y)].y;
    } else if (x == FLUID_GRID_SIZE - 1u) {
        v.x = 0.0;
        v.y = velocity_in[grid_index(FLUID_GRID_SIZE - 2u, y)].y;
    }

    // Bottom / top walls: set v.y = 0 and copy v.x from interior neighbor.
    if (y == 0u) {
        v.y = 0.0;
        v.x = velocity_in[grid_index(x, 1u)].x;
    } else if (y == FLUID_GRID_SIZE - 1u) {
        v.y = 0.0;
        v.x = velocity_in[grid_index(x, FLUID_GRID_SIZE - 2u)].x;
    }

    velocity_out[idx] = v;
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
