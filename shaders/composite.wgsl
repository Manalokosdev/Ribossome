// ============================================================================
// COMPOSITE.WGSL - COMPOSITING AND BLENDING
// ============================================================================
// This file contains the final compositing stage that blends the rendered
// agents (from agent_grid) onto the environment background (visual_grid).
//
// Supports multiple blend modes:
// - Mode 0: Normal (alpha blend)
// - Mode 1: Add (additive blending with color tint)
// - Mode 2: Subtract (subtractive blending with color tint)
// - Mode 3: Multiply (multiplicative blending with color tint)
// ============================================================================

// Standalone bindings for composite shader (not part of shared.wgsl)
const SIM_SIZE: u32 = 30720u;

struct SimParams {
    grid_size: f32,
    window_width: f32,
    window_height: f32,
    camera_zoom: f32,
    camera_pan_x: f32,
    camera_pan_y: f32,
    visual_stride: u32,
    draw_enabled: u32,
    agent_blend_mode: u32,
    agent_color_r: f32,
    agent_color_g: f32,
    agent_color_b: f32,
    fluid_show: u32,
    _pad: u32,
}

@group(0) @binding(0)
var<storage, read_write> visual_grid: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> agent_grid: array<vec4<f32>>;

@group(0) @binding(2)
var<uniform> params: SimParams;

@group(0) @binding(3)
var<storage, read> fluid_dye: array<f32>;

@compute @workgroup_size(16, 16)
fn composite_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (params.draw_enabled == 0u) { return; }
    let x = gid.x;
    let y = gid.y;

    let safe_width = max(params.window_width, 1.0);
    let safe_height = max(params.window_height, 1.0);
    let width = u32(safe_width);
    let height = u32(safe_height);

    if (x >= width || y >= height) {
        return;
    }

    let idx = y * params.visual_stride + x;
    let agent_pixel = agent_grid[idx];

    let base_color = visual_grid[idx].rgb;
    let agent_color_param = vec3<f32>(
        clamp(params.agent_color_r, 0.0, 1.0),
        clamp(params.agent_color_g, 0.0, 1.0),
        clamp(params.agent_color_b, 0.0, 1.0)
    );

    var result_color = base_color;

    if (agent_pixel.a > 0.0) {
        if (params.agent_blend_mode == 0u) {
            // Comp (normal) - alpha blend agent on top of base
            result_color = mix(base_color, agent_pixel.rgb, agent_pixel.a);
        } else if (params.agent_blend_mode == 1u) {
            // Add - add agent color tint to agent pixel, then composite
            let tinted_agent = clamp(agent_pixel.rgb + agent_color_param, vec3<f32>(0.0), vec3<f32>(1.0));
            result_color = clamp(base_color + tinted_agent * agent_pixel.a, vec3<f32>(0.0), vec3<f32>(1.0));
        } else if (params.agent_blend_mode == 2u) {
            // Subtract - subtract agent color tint from agent pixel, then composite
            let tinted_agent = clamp(agent_pixel.rgb - agent_color_param, vec3<f32>(0.0), vec3<f32>(1.0));
            result_color = mix(base_color, tinted_agent, agent_pixel.a);
        } else {
            // Multiply - multiply agent by color tint, then composite
            let tinted_agent = agent_pixel.rgb * agent_color_param;
            result_color = mix(base_color, tinted_agent, agent_pixel.a);
        }
    }

    // Overlay fluid dye visualization if enabled
    if (params.fluid_show > 0u) {
        // Convert screen pixel -> world coordinates using the same camera math as shared.wgsl
        let safe_zoom = max(params.camera_zoom, 0.0001);
        let aspect_ratio = safe_width / safe_height;
        let view_width = params.grid_size / safe_zoom;
        let view_height = view_width / aspect_ratio;
        let cam_min_x = params.camera_pan_x - view_width * 0.5;
        let cam_min_y = params.camera_pan_y - view_height * 0.5;

        // Pixel center in normalized [0..1]
        let norm_x = (f32(x) + 0.5) / safe_width;
        let norm_y = (f32(y) + 0.5) / safe_height;
        var world_pos = vec2<f32>(
            cam_min_x + norm_x * view_width,
            cam_min_y + norm_y * view_height
        );

        // Wrap into [0..SIM_SIZE) to match simulation space
        let ws = f32(SIM_SIZE);
        world_pos = world_pos - floor(world_pos / ws) * ws;

        // Map to fluid grid coordinates (128x128) using the same mapping as propeller injection
        let fluid_x = u32(clamp(world_pos.x / ws * 128.0, 0.0, 127.0));
        let fluid_y = u32(clamp(world_pos.y / ws * 128.0, 0.0, 127.0));
        let fluid_idx = fluid_y * 128u + fluid_x;

        // Sample dye concentration directly from fluid_dye buffer
        let dye_concentration = fluid_dye[fluid_idx];

        // Dye color: bright cyan/blue that fades to transparent
        // Use a color that contrasts well with the simulation background
        let dye_color = vec3<f32>(0.2, 0.7, 1.0); // Bright cyan-blue

        // Alpha based on dye concentration - visible where dye exists
        let fluid_alpha = clamp(dye_concentration * 0.8, 0.0, 0.8);
        result_color = mix(result_color, dye_color, fluid_alpha)*0.0;
    }

    visual_grid[idx] = vec4<f32>(result_color, 1.0);
}
