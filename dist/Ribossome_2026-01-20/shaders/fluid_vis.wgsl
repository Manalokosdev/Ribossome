// Fluid Visualization Compute Shader
// Composites fluid velocity field onto visual grid as colored overlay

@group(0) @binding(0)
var<storage, read> fluid_velocity: array<vec2<f32>>;  // 128x128 fluid velocity field

@group(1) @binding(0)
var<storage, read_write> visual_grid: array<vec4<f32>>;  // Screen-sized render target

@group(1) @binding(1)
var<uniform> params: FluidVisParams;

struct FluidVisParams {
    screen_width: u32,
    screen_height: u32,
    fluid_grid_size: u32,
    opacity: f32,
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let screen_x = global_id.x;
    let screen_y = global_id.y;

    if (screen_x >= params.screen_width || screen_y >= params.screen_height) {
        return;
    }

    // Map screen pixel to fluid grid cell
    let fluid_x = (screen_x * params.fluid_grid_size) / params.screen_width;
    let fluid_y = (screen_y * params.fluid_grid_size) / params.screen_height;
    let fluid_idx = fluid_y * params.fluid_grid_size + fluid_x;

    // Read fluid velocity
    let velocity = fluid_velocity[fluid_idx];
    let speed = length(velocity);

    if (speed < 0.01) {
        return;  // Skip very slow fluid
    }

    // Convert velocity to color
    let angle = atan2(velocity.y, velocity.x);
    let hue = (angle + 3.14159265) / (2.0 * 3.14159265);

    // Simple HSV to RGB (based on hue only, saturation=1, value=speed-scaled)
    let brightness = clamp(speed * 5.0, 0.0, 1.0);
    var rgb: vec3<f32>;
    let h6 = hue * 6.0;
    if (h6 < 1.0) {
        rgb = vec3<f32>(1.0, h6, 0.0);
    } else if (h6 < 2.0) {
        rgb = vec3<f32>(2.0 - h6, 1.0, 0.0);
    } else if (h6 < 3.0) {
        rgb = vec3<f32>(0.0, 1.0, h6 - 2.0);
    } else if (h6 < 4.0) {
        rgb = vec3<f32>(0.0, 4.0 - h6, 1.0);
    } else if (h6 < 5.0) {
        rgb = vec3<f32>(h6 - 4.0, 0.0, 1.0);
    } else {
        rgb = vec3<f32>(1.0, 0.0, 6.0 - h6);
    }

    rgb *= brightness;

    // Blend with existing visual grid (additive with opacity)
    let screen_idx = screen_y * params.screen_width + screen_x;
    let current_color = visual_grid[screen_idx];
    let alpha = params.opacity * brightness;
    visual_grid[screen_idx] = vec4<f32>(
        current_color.rgb + rgb * alpha,
        current_color.a
    );
}
