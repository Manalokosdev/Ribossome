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
// Two-tone energy palette (shared with the deposit renderer): ACTIVATED
// monomers are GOLD (charged, alive), SPENT are the complementary cool
// BLUE — mixed nucleotide hues averaged out to grey and became
// indistinguishable from the spent tint. Per-channel identity remains
// available on debug keys 1-4.
// Saturated poles with a LUMINANCE gap too (activated bright, spent dim),
// so the state reads even for color-weak eyes.
const MONOMER_ACT_COLOR: vec3<f32> = vec3<f32>(1.0, 0.62, 0.05);
const MONOMER_SPENT_COLOR: vec3<f32> = vec3<f32>(0.05, 0.3, 0.85);
const MONOMER_VIS: f32 = 0.5;

const TAU: f32 = 6.2831853;

fn thin_film_rgb(opd: f32) -> vec3<f32> {
    // Approximate thin-film interference (2-beam) sampled at representative wavelengths.
    // We treat `opd` as an optical path difference in arbitrary units.
    // Intensity per wavelength: I = 0.5 + 0.5*cos(2π*opd/λ + φ)
    // Using shorter λ for blue makes it oscillate faster, producing the expected color order.

    // Representative wavelengths (arbitrary scale; only ratios matter).
    let lambda_r = 0.650;
    let lambda_g = 0.530;
    let lambda_b = 0.460;

    // Reflection phase shift. A constant π shift flips bright/dark ordering; using π makes
    // the early bands closer to typical thin-film photos for a single-interface approximation.
    let phi = 3.1415927;

    let r = 0.5 + 0.5 * cos(TAU * (opd / lambda_r) + phi);
    let g = 0.5 + 0.5 * cos(TAU * (opd / lambda_g) + phi);
    let b = 0.5 + 0.5 * cos(TAU * (opd / lambda_b) + phi);

    // Mild contrast boost (keeps mid-bands visible without crushing).
    let rgb = vec3<f32>(r, g, b);
    return clamp(pow(rgb, vec3<f32>(1.1)), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Must match SimParams layout from main.rs.
struct SimParams {
    dt: f32,
    frame_dt: f32,
    drag: f32,
    energy_cost: f32,
    amino_maintenance_cost: f32,
    morphology_change_cost: f32,
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
    suspend_full_speed: f32,
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
    population_count: u32,
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
    species_heatmap: u32,
    species_bucket_size: u32,
    nucleotide_debug_mask: u32,
    suspend_zero_speed: f32,
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

@group(0) @binding(4)
// Quantized monomer counts: one u32 per cell-channel packing ACTIVATED
// (low 16 bits) and SPENT (high 16 bits). Read here to render spent
// matter as grey under the colored (activated) dye.
var<storage, read> chem_grid: array<u32>;

@group(0) @binding(5)
// Fluid temperature field (f32 per fluid cell) for the key-7 heatmap view.
var<storage, read> fluid_temperature: array<f32>;

fn chem_spent_total_f(cell_idx: u32) -> f32 {
    let base = cell_idx * 4u;
    let t = (chem_grid[base] >> 16u)
        + (chem_grid[base + 1u] >> 16u)
        + (chem_grid[base + 2u] >> 16u)
        + (chem_grid[base + 3u] >> 16u);
    return f32(t) / 3.0;
}

// DYE RETIRED: the monomer layer is read straight from the quantized grid
// (activated counts, low 16 bits, in REF units of 3 quanta).
fn act_cell4(cell: u32) -> vec4<f32> {
    let b = cell * 4u;
    return vec4<f32>(
        f32(chem_grid[b] & 0xFFFFu),
        f32(chem_grid[b + 1u] & 0xFFFFu),
        f32(chem_grid[b + 2u] & 0xFFFFu),
        f32(chem_grid[b + 3u] & 0xFFFFu)
    ) / 3.0;
}

// Bilinear sampling of activated counts (Grid Interpolation smooth).
fn sample_dye_bilinear(world_pos: vec2<f32>, ws: f32) -> vec4<f32> {
    let grid_f = f32(GAMMA_GRID_DIM);
    let gx = clamp(world_pos.x / ws * grid_f - 0.5, 0.0, grid_f - 1.0);
    let gy = clamp(world_pos.y / ws * grid_f - 0.5, 0.0, grid_f - 1.0);
    let x0 = u32(floor(gx));
    let y0 = u32(floor(gy));
    let x1 = min(x0 + 1u, GAMMA_GRID_DIM - 1u);
    let y1 = min(y0 + 1u, GAMMA_GRID_DIM - 1u);
    let fx = fract(gx);
    let fy = fract(gy);
    let v00 = act_cell4(y0 * GAMMA_GRID_DIM + x0);
    let v10 = act_cell4(y0 * GAMMA_GRID_DIM + x1);
    let v01 = act_cell4(y1 * GAMMA_GRID_DIM + x0);
    let v11 = act_cell4(y1 * GAMMA_GRID_DIM + x1);
    return mix(mix(v00, v10, fx), mix(v01, v11, fx), fy);
}

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

            // AQUARIUM AMBIENT LIGHT: a faint glow showing the UV gradient
            // (screen top = low world y = lit surface; keep LIGHT_ATTEN in
            // sync with simulation.wgsl).
            let light_c = exp(-max(params.dye_alpha_color_g, 0.5) * (1.0 - clamp(world_pos.y / ws, 0.0, 1.0))) * clamp(params.dye_alpha_color_r, 0.0, 2.0);
            // Warm sunlight tint (so it never reads as the cool spent blue).
            under_color = under_color + vec3<f32>(0.06, 0.05, 0.035) * light_c;

            // Sample the per-monomer dye (x=A, y=U, z=G, w=C), honoring the
            // Grid Interpolation setting like the deposit renderer does.
            var dye_raw4 = act_cell4(dye_idx);
            if (params.grid_interpolation >= 1u) {
                dye_raw4 = max(sample_dye_bilinear(world_pos, ws), vec4<f32>(0.0));
            }
            let dye_lin4 = dye_raw4 * DYE_VIS_GAIN;
            // Simple Reinhard tone map per-channel: x / (1 + x)
            let dye_tm4 = dye_lin4 / (vec4<f32>(1.0) + dye_lin4);
            // Pool views for the 2-channel compositing below
            // (dye_raw/dye_tm keep the legacy meaning: x = GC pool, y = AU pool).
            let dye_raw = vec2<f32>(dye_raw4.z + dye_raw4.w, dye_raw4.x + dye_raw4.y);
            let dye_tm = clamp(vec2<f32>(dye_tm4.z + dye_tm4.w, dye_tm4.x + dye_tm4.y), vec2<f32>(0.0), vec2<f32>(1.0));

            // Unified palette: ratios come straight from the dye itself now —
            // full per-nucleotide identity riding the currents.
            let pool_au = dye_raw4.x + dye_raw4.y;
            let pool_gc = dye_raw4.z + dye_raw4.w;
            let ratio_u = select(0.5, dye_raw4.y / max(pool_au, 1e-6), pool_au > 1e-6);
            let ratio_c = select(0.5, dye_raw4.w / max(pool_gc, 1e-6), pool_gc > 1e-6);
            if (params.nucleotide_debug_mask != 0u) {
                // RGBW debug (keys 1-4) applies to the dye too: split each
                // pool's tracer by the local monomer ratio — A=red, U=green,
                // G=blue, C=white, matching the deposit debug view.
                // Keys 1-4: ACTIVATED monomers only, per-nucleotide palette
                // (A=red, U=yellow, G=green, C=blue — the dye tracks act).
                let m = params.nucleotide_debug_mask;
                let a_dye = dye_tm4.x;
                let u_dye = dye_tm4.y;
                let g_dye = dye_tm4.z;
                let c_dye = dye_tm4.w;
                var dbg = vec3<f32>(0.0);
                if ((m & 1u) != 0u) { dbg += vec3<f32>(1.0, 0.15, 0.1) * a_dye; }
                if ((m & 2u) != 0u) { dbg += vec3<f32>(1.0, 0.85, 0.1) * u_dye; }
                if ((m & 4u) != 0u) { dbg += vec3<f32>(0.15, 0.9, 0.25) * g_dye; }
                if ((m & 8u) != 0u) { dbg += vec3<f32>(0.2, 0.45, 1.0) * c_dye; }
                under_color = clamp(under_color + dbg, vec3<f32>(0.0), vec3<f32>(1.0));
            } else {
            // Per-nucleotide palette: A=red, U=yellow, G=green, C=blue —
            // four hues spread around the color wheel so every channel is
            // identifiable at a glance. MONOMER_VIS dims the whole layer so
            // agents stay clearly readable on top of the soup.
            let dye_gamma = max(params.alpha_gamma_adjust, 0.05);
            // Two-tone by FRACTION, not addition: cells almost always hold
            // BOTH states, and gold+blue added together washed out to grey
            // (tone-on-tone). Hue = blue<->gold by activated fraction,
            // brightness = total amount — every cell reads as one color on
            // the cold<->hot axis.
            let act_amt = clamp(dye_tm4.x + dye_tm4.y + dye_tm4.z + dye_tm4.w, 0.0, 1.5);
            let spent_amt = clamp(chem_spent_total_f(dye_idx), 0.0, 1.5);
            let total_amt = act_amt + spent_amt;
            // Contrast-stretched fraction: a linear blend of complementary
            // colors passes through muddy grey, so push cells toward one
            // pole or the other.
            let act_frac = smoothstep(0.25, 0.75, act_amt / max(total_amt, 1e-5));
            let hue = mix(MONOMER_SPENT_COLOR, MONOMER_ACT_COLOR, act_frac);
            let inten = pow(clamp(total_amt, 0.0, 1.0), dye_gamma);
            // Brightness slider (repurposed _pad_dye_alpha_color slot).
            let monomer_vis = clamp(params._pad_dye_alpha_color, 0.0, 1.0);
            // OPACITY blend (not additive): deposit and dye now show the same
            // counts, and adding them doubled the soup over the agents.
            under_color = mix(under_color, hue, clamp(inten * monomer_vis, 0.0, 1.0));
            under_color = clamp(under_color, vec3<f32>(0.0), vec3<f32>(1.0));
            }
        }
    }

    // KEY 7: temperature heatmap view (black -> red -> yellow -> white),
    // overrides the background layers.
    if ((params.nucleotide_debug_mask & 32u) != 0u) {
        let safe_zoom2 = max(params.camera_zoom, 0.0001);
        let safe_w2 = max(params.window_width, 1.0);
        let safe_h2 = max(params.window_height, 1.0);
        let aspect2 = safe_w2 / safe_h2;
        let vw2 = params.grid_size / safe_zoom2;
        let vh2 = vw2 / aspect2;
        let wx2 = params.camera_pan_x - vw2 * 0.5 + ((f32(x) + 0.5) / safe_w2) * vw2;
        let wy2 = params.camera_pan_y - vh2 * 0.5 + ((f32(y) + 0.5) / safe_h2) * vh2;
        let ws2 = max(params.grid_size, 1e-6);
        if (wx2 >= 0.0 && wx2 < ws2 && wy2 >= 0.0 && wy2 < ws2) {
            let fxc = min(u32(wx2 / ws2 * f32(FLUID_GRID_SIZE)), FLUID_GRID_SIZE - 1u);
            let fyc = min(u32(wy2 / ws2 * f32(FLUID_GRID_SIZE)), FLUID_GRID_SIZE - 1u);
            // Cold blue -> warm red, with the ACTIVATION ISOTHERM (T = 2.0,
            // TEMP_ACT_THRESHOLD in fluid.wgsl) drawn as a white contour:
            // inside it, hot water re-activates spent monomers.
            let traw = fluid_temperature[fyc * FLUID_GRID_SIZE + fxc];
            let tt = clamp(sqrt(traw / 12.0), 0.0, 1.0);
            under_color = mix(vec3<f32>(0.02, 0.08, 0.35), vec3<f32>(0.95, 0.12, 0.05), tt);
            if (abs(traw - 2.0) < 0.12) {
                under_color = vec3<f32>(0.95, 0.95, 0.95);
            }
        } else {
            under_color = vec3<f32>(0.0);
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
