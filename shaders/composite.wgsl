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

    // Skip if no agent drawn here (transparent)
    if (agent_pixel.a == 0.0) {
        return;
    }

    let base_color = visual_grid[idx].rgb;
    let agent_color_param = vec3<f32>(
        clamp(params.agent_color_r, 0.0, 1.0),
        clamp(params.agent_color_g, 0.0, 1.0),
        clamp(params.agent_color_b, 0.0, 1.0)
    );

    var result_color = vec3<f32>(0.0);

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

    visual_grid[idx] = vec4<f32>(result_color, 1.0);
}
