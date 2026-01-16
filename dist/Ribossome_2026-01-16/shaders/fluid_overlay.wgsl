// Fluid Visualization Overlay Shader
// Renders fluid velocity field as color overlay on top of the main simulation

@group(0) @binding(0)
var fluid_velocity_texture: texture_2d<f32>;

@group(0) @binding(1)
var fluid_sampler: sampler;

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample fluid velocity
    let velocity = textureSample(fluid_velocity_texture, fluid_sampler, in.uv).xy;

    // Convert velocity to color (HSV-like mapping)
    let speed = length(velocity);
    let angle = atan2(velocity.y, velocity.x);

    // Map angle to hue (0-360 degrees to 0-1)
    let hue = (angle + 3.14159265) / (2.0 * 3.14159265);

    // HSV to RGB conversion
    let c = speed * 2.0; // Saturation based on speed
    let x = c * (1.0 - abs((hue * 6.0) % 2.0 - 1.0));
    let m = 0.0;

    var rgb: vec3<f32>;
    let h6 = hue * 6.0;
    if (h6 < 1.0) {
        rgb = vec3<f32>(c, x, 0.0);
    } else if (h6 < 2.0) {
        rgb = vec3<f32>(x, c, 0.0);
    } else if (h6 < 3.0) {
        rgb = vec3<f32>(0.0, c, x);
    } else if (h6 < 4.0) {
        rgb = vec3<f32>(0.0, x, c);
    } else if (h6 < 5.0) {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }

    // Alpha based on speed (make slow areas more transparent)
    let alpha = clamp(speed * 0.3, 0.0, 0.7);

    return vec4<f32>(rgb + m, alpha);
}
