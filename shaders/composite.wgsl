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
const SIM_SIZE: u32 = 61440u;
// Visualization-only gain for fluid dye overlay. This does not affect simulation/sensing,
// only how strongly the dye is displayed when params.fluid_show is enabled.
// Keep this modest; the overlay also applies a tone-map to avoid blowout.
const DYE_VIS_GAIN: f32 = 2.0;

// Must match SimParams layout from main.rs.
struct SimParams {
    dt: f32,
    frame_dt: f32,
    drag: f32,
    energy_cost: f32,
    amino_maintenance_cost: f32,
    spawn_probability: f32,
    death_probability: f32,
    grid_size: f32,
    camera_zoom: f32,
    camera_pan_x: f32,
    camera_pan_y: f32,
    prev_camera_pan_x: f32,
    prev_camera_pan_y: f32,
    follow_mode: u32,
    window_width: f32,
    window_height: f32,
    alpha_blur: f32,
    beta_blur: f32,
    gamma_diffuse: f32,
    gamma_blur: f32,
    gamma_shift: f32,
    alpha_slope_bias: f32,
    beta_slope_bias: f32,
    alpha_multiplier: f32,
    beta_multiplier: f32,
    _pad_rain0: u32,
    _pad_rain1: u32,
    rain_drop_count: u32,
    alpha_rain_drop_count: u32,
    dye_precipitation: f32,
    chemical_slope_scale_alpha: f32,
    chemical_slope_scale_beta: f32,
    mutation_rate: f32,
    food_power: f32,
    poison_power: f32,
    pairing_cost: f32,
    max_agents: u32,
    cpu_spawn_count: u32,
    agent_count: u32,
    random_seed: u32,
    debug_mode: u32,
    visual_stride: u32,
    selected_agent_index: u32,
    repulsion_strength: f32,
    agent_repulsion_strength: f32,
    gamma_strength: f32,
    prop_wash_strength: f32,
    prop_wash_strength_fluid: f32,
    gamma_vis_min: f32,
    gamma_vis_max: f32,
    draw_enabled: u32,
    gamma_debug: u32,
    gamma_hidden: u32,
    slope_debug: u32,
    alpha_show: u32,
    beta_show: u32,
    gamma_show: u32,
    slope_lighting: u32,
    slope_lighting_strength: f32,
    trail_diffusion: f32,
    trail_decay: f32,
    trail_opacity: f32,
    trail_show: u32,
    interior_isotropic: u32,
    ignore_stop_codons: u32,
    require_start_codon: u32,
    asexual_reproduction: u32,
    background_color_r: f32,
    background_color_g: f32,
    background_color_b: f32,
    alpha_blend_mode: u32,
    beta_blend_mode: u32,
    gamma_blend_mode: u32,
    slope_blend_mode: u32,
    alpha_color_r: f32,
    alpha_color_g: f32,
    alpha_color_b: f32,
    beta_color_r: f32,
    beta_color_g: f32,
    beta_color_b: f32,
    gamma_color_r: f32,
    gamma_color_g: f32,
    gamma_color_b: f32,
    grid_interpolation: u32,
    alpha_gamma_adjust: f32,
    beta_gamma_adjust: f32,
    gamma_gamma_adjust: f32,
    light_dir_x: f32,
    light_dir_y: f32,
    light_dir_z: f32,
    light_power: f32,
    agent_blend_mode: u32,
    agent_color_r: f32,
    agent_color_g: f32,
    agent_color_b: f32,
    agent_color_blend: f32,
    epoch: u32,
    vector_force_power: f32,
    vector_force_x: f32,
    vector_force_y: f32,
    inspector_zoom: f32,
    agent_trail_decay: f32,
    fluid_show: u32,
    fluid_wind_push_strength: f32,
    alpha_fluid_convolution: f32,
    beta_fluid_convolution: f32,
    fluid_slope_force_scale: f32,
    fluid_obstacle_strength: f32,

    dye_alpha_color_r: f32,
    dye_alpha_color_g: f32,
    dye_alpha_color_b: f32,
    _pad_dye_alpha_color: f32,
    dye_beta_color_r: f32,
    dye_beta_color_g: f32,
    dye_beta_color_b: f32,
    _pad_dye_beta_color: f32,
}

@group(0) @binding(0)
var<storage, read_write> visual_grid: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> agent_grid: array<vec4<f32>>;

@group(0) @binding(2)
var<uniform> params: SimParams;

@group(0) @binding(3)
// Three-channel dye per env cell stored as vec4:
// - x = beta (red)
// - y = alpha (green)
// - z = gamma (blue)
var<storage, read> fluid_dye: array<vec4<f32>>;

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

    // Compose background layers first (dye), then composite agents last.
    var under_color = base_color;

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

        // Respect hard simulation bounds (no torus wrap). Outside the world: no dye overlay.
        let ws = max(params.grid_size, 1e-6);
        if (world_pos.x >= 0.0 && world_pos.x < ws && world_pos.y >= 0.0 && world_pos.y < ws) {
            // Map to dye grid coordinates (GAMMA_GRID_DIM x GAMMA_GRID_DIM).
            let grid_f = f32(GAMMA_GRID_DIM);
            let max_idx_f = grid_f - 1.0;
            let dye_x = u32(clamp(world_pos.x / ws * grid_f, 0.0, max_idx_f));
            let dye_y = u32(clamp(world_pos.y / ws * grid_f, 0.0, max_idx_f));
            let dye_idx = dye_y * GAMMA_GRID_DIM + dye_x;

            // Sample dye (x = beta, y = alpha) directly from dye buffer.
            // Apply a visualization-only gain, then tone-map to avoid hard saturation.
            let dye_raw = max(fluid_dye[dye_idx].xy, vec2<f32>(0.0, 0.0));
            let dye_lin = dye_raw * DYE_VIS_GAIN;
            // Simple Reinhard tone map per-channel: x / (1 + x)
            let dye_tm = dye_lin / (vec2<f32>(1.0, 1.0) + dye_lin);

            // Composite dye using independent dye colors, but reusing alpha/beta blend mode + gamma controls.
            let alpha_color = vec3<f32>(
                clamp(params.dye_alpha_color_r, 0.0, 1.0),
                clamp(params.dye_alpha_color_g, 0.0, 1.0),
                clamp(params.dye_alpha_color_b, 0.0, 1.0)
            );
            let beta_color = vec3<f32>(
                clamp(params.dye_beta_color_r, 0.0, 1.0),
                clamp(params.dye_beta_color_g, 0.0, 1.0),
                clamp(params.dye_beta_color_b, 0.0, 1.0)
            );

            // Apply gamma adjustment to dye intensities for consistent look.
            let dye_alpha = pow(dye_tm.y, params.alpha_gamma_adjust);
            let dye_beta = pow(dye_tm.x, params.beta_gamma_adjust);

            // Alpha dye layer
            if (params.alpha_blend_mode == 0u) {
                // Additive
                under_color = under_color + alpha_color * dye_alpha;
            } else {
                // Multiply with inverted channel
                under_color = under_color * mix(vec3<f32>(1.0), vec3<f32>(1.0) - alpha_color, dye_alpha);
            }

            // Beta dye layer
            if (params.beta_blend_mode == 0u) {
                // Additive
                under_color = under_color + beta_color * dye_beta;
            } else {
                // Multiply with inverted channel
                under_color = under_color * mix(vec3<f32>(1.0), vec3<f32>(1.0) - beta_color, dye_beta);
            }

            under_color = clamp(under_color, vec3<f32>(0.0), vec3<f32>(1.0));
        }
    }

    var result_color = under_color;
    if (agent_pixel.a > 0.0) {
        if (params.agent_blend_mode == 0u) {
            // Comp (normal) - alpha blend agent on top of base
            result_color = mix(under_color, agent_pixel.rgb, agent_pixel.a);
        } else if (params.agent_blend_mode == 1u) {
            // Add - add agent color tint to agent pixel, then composite
            let tinted_agent = clamp(agent_pixel.rgb + agent_color_param, vec3<f32>(0.0), vec3<f32>(1.0));
            result_color = clamp(under_color + tinted_agent * agent_pixel.a, vec3<f32>(0.0), vec3<f32>(1.0));
        } else if (params.agent_blend_mode == 2u) {
            // Subtract - subtract agent color tint from agent pixel, then composite
            let tinted_agent = clamp(agent_pixel.rgb - agent_color_param, vec3<f32>(0.0), vec3<f32>(1.0));
            result_color = mix(under_color, tinted_agent, agent_pixel.a);
        } else {
            // Multiply - multiply agent by color tint, then composite
            let tinted_agent = agent_pixel.rgb * agent_color_param;
            result_color = mix(under_color, tinted_agent, agent_pixel.a);
        }
    }

    visual_grid[idx] = vec4<f32>(result_color, 1.0);
}
