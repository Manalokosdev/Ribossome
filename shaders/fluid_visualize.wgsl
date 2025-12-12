// Fluid visualization shader
// Modes:
// 0 = advected display texture (feedback)
// 1 = velocity Y component (red for positive, blue for negative)
// 2 = speed (magnitude)
// 3 = pressure
// 4 = divergence
// 5 = vorticity

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    // Full-screen triangle
    let x = f32((vertex_index & 1u) << 2u);
    let y = f32((vertex_index & 2u) << 1u);

    out.position = vec4<f32>(x - 1.0, 1.0 - y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5, y * 0.5);

    return out;
}

@group(0) @binding(0) var<storage, read> velocity: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> pressure: array<f32>;
@group(0) @binding(2) var<storage, read> divergence: array<f32>;
@group(0) @binding(4) var tex: texture_2d<f32>;
@group(0) @binding(5) var tex_sampler: sampler;

struct Params {
    grid_size: u32,
    mode: u32,
    _pad: vec2<u32>,
    // x = velocity intensity scale, y = pressure scale, z = divergence scale, w = vorticity scale
    scale: vec4<f32>,
}
@group(0) @binding(3) var<uniform> params: Params;

fn clamp_u32(v: i32, lo: i32, hi: i32) -> u32 {
    return u32(clamp(v, lo, hi));
}

fn curl_from_velocity(gx: u32, gy: u32, grid_size: u32) -> f32 {
    let xm = clamp_u32(i32(gx) - 1, 0, i32(grid_size) - 1);
    let xp = clamp_u32(i32(gx) + 1, 0, i32(grid_size) - 1);
    let ym = clamp_u32(i32(gy) - 1, 0, i32(grid_size) - 1);
    let yp = clamp_u32(i32(gy) + 1, 0, i32(grid_size) - 1);

    let idx_l = gy * grid_size + xm;
    let idx_r = gy * grid_size + xp;
    let idx_b = ym * grid_size + gx;
    let idx_t = yp * grid_size + gx;

    let v_l = velocity[idx_l];
    let v_r = velocity[idx_r];
    let v_b = velocity[idx_b];
    let v_t = velocity[idx_t];

    let dvy_dx = 0.5 * (v_r.y - v_l.y);
    let dvx_dy = 0.5 * (v_t.x - v_b.x);
    return dvy_dx - dvx_dy;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let grid_size = params.grid_size;

    // Convert UV to grid coordinates
    let gx = u32(in.uv.x * f32(grid_size));
    let gy = u32(in.uv.y * f32(grid_size));

    if (gx >= grid_size || gy >= grid_size) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let idx = gy * grid_size + gx;

    switch params.mode {
        case 0u: {
            // Display the feedback/advected texture directly.
            let c = textureSample(tex, tex_sampler, in.uv);
            return vec4<f32>(c.rgb, 1.0);
        }
        case 1u: {
            // Velocity Y component (red for positive, blue for negative)
            let v = velocity[idx];
            let vy = v.y * params.scale.x;
            let pos = clamp(vy, 0.0, 1.0);
            let neg = clamp(-vy, 0.0, 1.0);
            return vec4<f32>(pos, 0.0, neg, 1.0);
        }
        case 2u: {
            // Speed (magnitude)
            let v = velocity[idx];
            let speed = length(v);
            let intensity = clamp(speed * params.scale.x, 0.0, 1.0);
            return vec4<f32>(intensity, intensity, intensity, 1.0);
        }
        case 3u: {
            // Pressure (signed, centered at 0.5)
            let p = pressure[idx] * params.scale.y;
            let val = clamp(0.5 + p, 0.0, 1.0);
            return vec4<f32>(val, val, val, 1.0);
        }
        case 4u: {
            // Divergence (signed, centered at 0.5)
            let d = divergence[idx] * params.scale.z;
            let val = clamp(0.5 + d, 0.0, 1.0);
            return vec4<f32>(val, val, val, 1.0);
        }
        default: {
            // Vorticity (signed) computed from velocity
            let w = curl_from_velocity(gx, gy, grid_size) * params.scale.w;
            // Blue for negative, red for positive.
            let pos = clamp(w, 0.0, 1.0);
            let neg = clamp(-w, 0.0, 1.0);
            return vec4<f32>(pos, 0.0, neg, 1.0);
        }
    }
}
