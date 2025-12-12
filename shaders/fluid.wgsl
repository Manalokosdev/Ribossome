// Standalone Fluid Simulation
// Stable-fluids style: advection + external forces + projection + dye

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

// Passive dye ping-pong buffers (vec4 per cell: rgb + a)
@group(0) @binding(4) var<storage, read> dye_in: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> dye_out: array<vec4<f32>>;

// Pressure ping-pong + divergence (f32 per cell)
@group(0) @binding(6) var<storage, read> pressure_in: array<f32>;
@group(0) @binding(7) var<storage, read_write> pressure_out: array<f32>;
@group(0) @binding(8) var<storage, read_write> divergence: array<f32>;


// Parameters
struct FluidParams {
    time: f32,
    dt: f32,
    decay: f32,
    grid_size: u32,
    // mouse.xy in grid coords (0..grid), mouse.zw = mouse velocity in grid units/sec
    mouse: vec4<f32>,
    // splat.x = radius (cells), splat.y = force scale, splat.z = dye amount, splat.w = mouse_down (0/1)
    splat: vec4<f32>,
}
@group(0) @binding(3) var<uniform> params: FluidParams;

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
    let x = pos.x - 0.5;
    let y = pos.y - 0.5;
    
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

fn sample_dye(pos: vec2<f32>) -> vec4<f32> {
    let x = pos.x - 0.5;
    let y = pos.y - 0.5;

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

    let d00 = dye_in[grid_index(c00.x, c00.y)];
    let d10 = dye_in[grid_index(c10.x, c10.y)];
    let d01 = dye_in[grid_index(c01.x, c01.y)];
    let d11 = dye_in[grid_index(c11.x, c11.y)];

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

// ============================================================================
// COMPUTE KERNELS
// ============================================================================

// 1. Generate external forces (mouse splat + optional moving source)
@compute @workgroup_size(16, 16)
fn generate_test_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }
    
    let idx = grid_index(x, y);
    let pos = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5);

    // Start with zero; later passes can accumulate additional forces.
    var f = vec2<f32>(0.0);

    // Mouse splat force.
    if (params.splat.w > 0.5) {
        let center = params.mouse.xy;
        let offset = pos - center;
        let dist = length(offset);
        let falloff = splat_falloff(dist, params.splat.x);
        // Force along mouse motion.
        f = f + params.mouse.zw * (params.splat.y * falloff);
    } else {
        // Keep a small moving source so the sim shows motion without interaction.
        let base_center = vec2<f32>(f32(FLUID_GRID_SIZE) * 0.5);
        let orbit = vec2<f32>(sin(params.time * 0.7), cos(params.time * 0.7)) * (f32(FLUID_GRID_SIZE) * 0.18);
        let center = base_center + orbit;
        let offset = pos - center;
        let dist = length(offset);
        let radius = 18.0;
        let falloff = splat_falloff(dist, radius);
        let orbit_dir = normalize(vec2<f32>(cos(params.time * 0.7), -sin(params.time * 0.7)));
        var orbit_force = orbit_dir * (350.0 * falloff);
        if (dist > 1.0) {
            let radial = offset / dist;
            let tangential = vec2<f32>(-radial.y, radial.x);
            orbit_force = orbit_force + tangential * (250.0 * falloff);
        }
        f = f + orbit_force;
    }

    forces[idx] = f;
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
    
    // Apply decay
    velocity_out[idx] = advected_vel * params.decay;
}

// 2b. Advect passive dye by the *projected* velocity field (velocity_in)
@compute @workgroup_size(16, 16)
fn advect_dye(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);
    let pos = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5);

    // Use the divergence-free velocity to advect dye.
    let vel = velocity_in[idx];
    let trace_pos = pos - vel * params.dt;
    let advected = sample_dye(trace_pos);

    // Dye decay
    var out_dye = advected * 0.995;

    // Inject dye at mouse while dragging; otherwise inject at moving source.
    if (params.splat.w > 0.5) {
        let center = params.mouse.xy;
        let dist = length(pos - center);
        let falloff = splat_falloff(dist, params.splat.x);
        let t = params.time;
        // Bright, time-varying color.
        let color = vec3<f32>(
            0.5 + 0.5 * sin(t * 1.7),
            0.5 + 0.5 * sin(t * 1.7 + 2.094),
            0.5 + 0.5 * sin(t * 1.7 + 4.188)
        );
        let inject = color * (params.splat.z * falloff) * params.dt;
        out_dye = vec4<f32>(clamp(out_dye.rgb + inject, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    } else {
        let base_center = vec2<f32>(f32(FLUID_GRID_SIZE) * 0.5);
        let orbit = vec2<f32>(sin(params.time * 0.7), cos(params.time * 0.7)) * (f32(FLUID_GRID_SIZE) * 0.18);
        let center = base_center + orbit;
        let dist = length(pos - center);
        let falloff = splat_falloff(dist, 20.0);
        let color = vec3<f32>(0.2, 0.7, 1.0);
        let inject = color * (6.0 * falloff) * params.dt;
        out_dye = vec4<f32>(clamp(out_dye.rgb + inject, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    }

    dye_out[idx] = out_dye;
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

    // Skip edges.
    if (x == 0 || x == FLUID_GRID_SIZE - 1 || y == 0 || y == FLUID_GRID_SIZE - 1) {
        return;
    }

    let idx = grid_index(x, y);

    let w_l = abs(curl_at(x - 1u, y));
    let w_r = abs(curl_at(x + 1u, y));
    let w_b = abs(curl_at(x, y - 1u));
    let w_t = abs(curl_at(x, y + 1u));

    let grad = vec2<f32>(w_r - w_l, w_t - w_b) * 0.5;
    let mag = max(length(grad), 1e-5);
    let n = grad / mag;

    let w = curl_at(x, y);
    // Confinement force: epsilon * (n x w_k) in 2D
    // Equivalent: f = epsilon * vec2(n.y, -n.x) * w
    let epsilon = 8.0;
    let f = vec2<f32>(n.y, -n.x) * (w * epsilon);

    velocity_out[idx] = velocity_in[idx] + f * params.dt;
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

    // Boundary condition: pressure = 0 at edges
    if (x == 0 || x == FLUID_GRID_SIZE - 1 || y == 0 || y == FLUID_GRID_SIZE - 1) {
        pressure_out[idx] = 0.0;
        return;
    }

    let xm = clamp(i32(x) - 1, 0, i32(FLUID_GRID_SIZE) - 1);
    let xp = clamp(i32(x) + 1, 0, i32(FLUID_GRID_SIZE) - 1);
    let ym = clamp(i32(y) - 1, 0, i32(FLUID_GRID_SIZE) - 1);
    let yp = clamp(i32(y) + 1, 0, i32(FLUID_GRID_SIZE) - 1);

    let p_l = pressure_in[grid_index(u32(xm), y)];
    let p_r = pressure_in[grid_index(u32(xp), y)];
    let p_b = pressure_in[grid_index(x, u32(ym))];
    let p_t = pressure_in[grid_index(x, u32(yp))];

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

    let xm = clamp(i32(x) - 1, 0, i32(FLUID_GRID_SIZE) - 1);
    let xp = clamp(i32(x) + 1, 0, i32(FLUID_GRID_SIZE) - 1);
    let ym = clamp(i32(y) - 1, 0, i32(FLUID_GRID_SIZE) - 1);
    let yp = clamp(i32(y) + 1, 0, i32(FLUID_GRID_SIZE) - 1);

    let p_l = pressure_in[grid_index(u32(xm), y)];
    let p_r = pressure_in[grid_index(u32(xp), y)];
    let p_b = pressure_in[grid_index(x, u32(ym))];
    let p_t = pressure_in[grid_index(x, u32(yp))];

    let grad = vec2<f32>(p_r - p_l, p_t - p_b) * 0.5;
    let v = velocity_in[idx];
    velocity_out[idx] = v - grad;
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

// 4. Enforce boundary conditions (no-slip: velocity = 0 at edges)
@compute @workgroup_size(16, 16)
fn enforce_boundaries(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }
    
    let idx = grid_index(x, y);
    
    // Set velocity to 0 at boundaries, pass through interior cells.
    if (x == 0 || x == FLUID_GRID_SIZE - 1 || y == 0 || y == FLUID_GRID_SIZE - 1) {
        velocity_out[idx] = vec2<f32>(0.0);
    } else {
        velocity_out[idx] = velocity_in[idx];
    }
}

// 4b. Clear dye (for initialization)
@compute @workgroup_size(16, 16)
fn clear_dye(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= FLUID_GRID_SIZE || y >= FLUID_GRID_SIZE) {
        return;
    }

    let idx = grid_index(x, y);
    dye_out[idx] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
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
