// ============================================================================
// RENDER.WGSL - ALL RENDERING FUNCTIONS AND KERNELS
// ============================================================================
// This file contains all rendering-related code:
// - Body part rendering functions (render_body_part, draw_selection_circle)
// - Drawing primitives (InspectorContext, draw_thick_line, draw_filled_circle, etc.)
// - Text rendering (vector font system)
// - Rendering kernels (clear_agent_grid, render_inspector, draw_inspector_agent, render_agents, composite_agents)
// ============================================================================

// ============================================================================
// PART RENDERING FUNCTION
// ============================================================================

// Inspector-specific render function (uses selected_agent_buffer instead of agents_out)
// Render a single body part with all its visual elements
fn render_body_part(
    part: BodyPart,
    part_index: u32,
    agent_id: u32,
    agent_position: vec2<f32>,
    agent_rotation: f32,
    agent_energy: f32,
    agent_color: vec3<f32>,
    body_count: u32,
    morphology_origin: vec2<f32>,
    amplification: f32,
    in_debug_mode: bool
) {
    render_body_part_ctx(part, part_index, agent_id, agent_position, agent_rotation, agent_energy, agent_color, body_count, morphology_origin, amplification, in_debug_mode, InspectorContext(vec2<f32>(-1.0), vec2<f32>(0.0), 1.0, vec2<f32>(0.0)));
}

fn render_body_part_ctx(
    part: BodyPart,
    part_index: u32,
    agent_id: u32,
    agent_position: vec2<f32>,
    agent_rotation: f32,
    agent_energy: f32,
    agent_color: vec3<f32>,
    body_count: u32,
    morphology_origin: vec2<f32>,
    amplification: f32,
    in_debug_mode: bool,
    ctx: InspectorContext
) {
    let base_type = get_base_part_type(part.part_type);
    let amino_props = get_amino_acid_properties(base_type);
    let rotated_pos = apply_agent_rotation(part.pos, agent_rotation);
    let world_pos = agent_position + rotated_pos;

    // Special agent_id value 0xFFFFFFFFu indicates we're rendering from selected_agent_buffer
    let use_selected_buffer = (agent_id == 0xFFFFFFFFu);

    // Determine segment start position
    var segment_start_world = agent_position + apply_agent_rotation(morphology_origin, agent_rotation);
    if (part_index > 0u) {
        var prev_part: BodyPart;
        if (use_selected_buffer) {
            prev_part = selected_agent_buffer[0].body[part_index - 1u];
        } else {
            prev_part = agents_out[agent_id].body[part_index - 1u];
        }
        let prev_rotated = apply_agent_rotation(prev_part.pos, agent_rotation);
        segment_start_world = agent_position + prev_rotated;
    }

    let is_first = part_index == 0u;
    let is_last = part_index == body_count - 1u;
    let is_single = body_count == 1u;

    // 1. STRUCTURAL RENDERING: Base segment line
    if (!in_debug_mode) {
        let thickness = part.size * 0.5;
        let blended_color = mix(amino_props.color, agent_color, params.agent_color_blend);
        draw_thick_line_ctx(segment_start_world, world_pos, thickness, vec4<f32>(blended_color, 1.0), ctx);
        if (!is_single && (is_first || is_last)) {
            draw_filled_circle_ctx(world_pos, thickness, vec4<f32>(blended_color, 1.0), ctx);
        }
    }

    // 2. DEBUG MODE RENDERING: Signal visualization
    if (in_debug_mode) {
        let a = part.alpha_signal;
        let b = part.beta_signal;
        let r = max(b, 0.0);
        let g = max(a, 0.0);
        let bl = max(max(-a, 0.0), max(-b, 0.0));
        let dbg_color = vec4<f32>(r, g, bl, 1.0);
        let thickness_dbg = max(part.size * 0.25, 0.5);
        draw_thick_line_ctx(segment_start_world, world_pos, thickness_dbg, dbg_color, ctx);
        if (!is_single && (is_first || is_last)) {
            draw_filled_circle_ctx(world_pos, thickness_dbg, dbg_color, ctx);
        }
        draw_filled_circle_ctx(world_pos, 1.5, dbg_color, ctx);
    }

    // 3. SPECIAL STRUCTURAL: Leucine (chirality flipper) - perpendicular bar
    if (base_type == 9u) {
        var segment_dir = vec2<f32>(0.0);
        if (part_index > 0u) {
            var prev: vec2<f32>;
            if (use_selected_buffer) {
                prev = selected_agent_buffer[0].body[part_index-1u].pos;
            } else {
                prev = agents_out[agent_id].body[part_index-1u].pos;
            }
            segment_dir = part.pos - prev;
        } else if (body_count > 1u) {
            var next: vec2<f32>;
            if (use_selected_buffer) {
                next = selected_agent_buffer[0].body[1u].pos;
            } else {
                next = agents_out[agent_id].body[1u].pos;
            }
            segment_dir = next - part.pos;
        } else {
            segment_dir = vec2<f32>(1.0, 0.0);
        }
        let seg_len = length(segment_dir);
        let axis_local = select(segment_dir / seg_len, vec2<f32>(1.0, 0.0), seg_len < 1e-4);
        let perp_local = vec2<f32>(-axis_local.y, axis_local.x);
        let perp_world = apply_agent_rotation(perp_local, agent_rotation);

        let half_length = part.size * 0.8;
        let p1 = world_pos - perp_world * half_length;
        let p2 = world_pos + perp_world * half_length;
        let perp_thickness = part.size * 0.3;
        let blended_color_leucine = mix(amino_props.color, agent_color, params.agent_color_blend);
        draw_thick_line_ctx(p1, p2, perp_thickness, vec4<f32>(blended_color_leucine, 1.0), ctx);
    }

    // 4. ORGAN: Condenser (charge storage/discharge)
    if (amino_props.is_condenser) {
        let signed_alpha_charge = part._pad.x;
        let signed_beta_charge = part._pad.y;
        let alpha_charge = clamp(abs(signed_alpha_charge), 0.0, 10.0);
        let beta_charge = clamp(abs(signed_beta_charge), 0.0, 10.0);
        let alpha_ratio = clamp(alpha_charge / 10.0, 0.0, 1.0);
        let beta_ratio = clamp(beta_charge / 10.0, 0.0, 1.0);
        let is_alpha_discharging = (signed_alpha_charge > 0.0);
        let is_beta_discharging = (signed_beta_charge > 0.0);

        let radius = max(part.size * 0.5, 3.0);

        var fill_color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        if (is_alpha_discharging || is_beta_discharging) {
            fill_color = vec4<f32>(1.0, 1.0, 1.0, 1.0); // White flash
        } else {
            let red_component = beta_ratio;
            let green_component = alpha_ratio;
            let charge_ratio = max(alpha_ratio, beta_ratio);
            let low_tint = vec3<f32>(red_component, green_component, 0.0) * 0.25;
            let base_tint = vec3<f32>(red_component, green_component, 0.0);
            let fill_rgb = mix(low_tint, base_tint, charge_ratio);
            fill_color = vec4<f32>(fill_rgb, 1.0);
        }

        // Fill circle
        let fill_segments = 32u;
        for (var s = 0u; s < fill_segments; s++) {
            let ang1 = f32(s) / f32(fill_segments) * 6.28318530718;
            let ang2 = f32(s + 1u) / f32(fill_segments) * 6.28318530718;
            let p1 = world_pos + vec2<f32>(cos(ang1) * radius, sin(ang1) * radius);
            let p2 = world_pos + vec2<f32>(cos(ang2) * radius, sin(ang2) * radius);
            draw_thick_line_ctx(world_pos, p1, radius * 0.5, fill_color, ctx);
            draw_thick_line_ctx(p1, p2, 1.0, fill_color, ctx);
        }

        // White outline
        let segments = 24u;
        var prev = world_pos + vec2<f32>(radius, 0.0);
        for (var s = 1u; s <= segments; s++) {
            let t = f32(s) / f32(segments);
            let ang = t * 6.28318530718;
            let p = world_pos + vec2<f32>(cos(ang) * radius, sin(ang) * radius);
            draw_thick_line_ctx(prev, p, 0.5, vec4<f32>(1.0, 1.0, 1.0, 1.0), ctx);
            prev = p;
        }
    }

    // 5. ORGAN: Enabler field visualization
    if (amino_props.is_inhibitor && params.camera_zoom > 5.0) {
        let radius = 20.0;
        let segments = 32u;
        let zoom = params.camera_zoom;
        let fade = clamp((zoom - 5.0) / 10.0, 0.0, 1.0);
        let alpha = 0.15 * fade;
        let color = vec4<f32>(0.2, 0.3, 0.2, alpha);
        var prev = world_pos + vec2<f32>(radius, 0.0);
        for (var s = 1u; s <= segments; s++) {
            let t = f32(s) / f32(segments);
            let ang = t * 6.28318530718;
            let p = world_pos + vec2<f32>(cos(ang)*radius, sin(ang)*radius);
            draw_thick_line_ctx(prev, p, 1.5, color, ctx);
            prev = p;
        }
        let blended_color_enabler = mix(amino_props.color, agent_color, params.agent_color_blend);
        draw_filled_circle_ctx(world_pos, 2.0, vec4<f32>(blended_color_enabler, 0.95), ctx);
    }

    // 6. ORGAN: Propeller jet particles
    if (PROPELLERS_ENABLED && amino_props.is_propeller && agent_energy > 0.0 && params.camera_zoom > 2.0) {
        var segment_dir = vec2<f32>(0.0);
        if (part_index > 0u) {
            var prev: vec2<f32>;
            if (use_selected_buffer) {
                prev = selected_agent_buffer[0].body[part_index-1u].pos;
            } else {
                prev = agents_out[agent_id].body[part_index-1u].pos;
            }
            segment_dir = part.pos - prev;
        } else if (body_count > 1u) {
            var next: vec2<f32>;
            if (use_selected_buffer) {
                next = selected_agent_buffer[0].body[1u].pos;
            } else {
                next = agents_out[agent_id].body[1u].pos;
            }
            segment_dir = next - part.pos;
        } else {
            segment_dir = vec2<f32>(1.0, 0.0);
        }
        let seg_len = length(segment_dir);
        let axis_local = select(segment_dir / seg_len, vec2<f32>(1.0, 0.0), seg_len < 1e-4);
        let axis_world = apply_agent_rotation(axis_local, agent_rotation);
        let jet_dir = normalize(vec2<f32>(-axis_world.y, axis_world.x));
        let exhaust_dir = -jet_dir;
        let propeller_strength = part.size * 2.5 * amplification;
        let zoom_factor = clamp((params.camera_zoom - 2.0) / 8.0, 0.0, 1.0);
        let jet_length = propeller_strength * mix(0.6, 1.2, zoom_factor);
        let jet_seed = agent_id * 1000u + part_index * 17u;
        let particle_count = 1u + u32(round(amplification * 5.0)) + u32(round(zoom_factor * 3.0));
        draw_particle_jet_ctx(world_pos, exhaust_dir, jet_length, jet_seed, particle_count, ctx);
    }

    // 7. ORGAN: Mouth (feeding organ) - asterisk marker
    if (amino_props.is_mouth) {
        // Vampire mouths (F/G/H) get special big red 8-point asterisks
        let is_vampire = (base_type == 4u || base_type == 5u || base_type == 6u);
        let mouth_radius = select(max(part.size * 1.5, 4.0), max(part.size * 3.0, 8.0), is_vampire);
        let mouth_color = select(mix(amino_props.color, agent_color, params.agent_color_blend), vec3<f32>(1.0, 0.0, 0.0), is_vampire);
        draw_asterisk_8_ctx(world_pos, mouth_radius, vec4<f32>(mouth_color, 0.9), ctx);
    }

    // 8. ORGAN: Displacer (repulsion field) - diamond marker
    if (amino_props.is_displacer) {
        let blended_color_displacer = mix(amino_props.color, agent_color, params.agent_color_blend);
        let diamond_size = max(part.size * 1.2, 3.0);
        // Draw diamond as 4 lines forming a diamond shape
        let half_s = diamond_size;
        let top = world_pos + vec2<f32>(0.0, -half_s);
        let right = world_pos + vec2<f32>(half_s, 0.0);
        let bottom = world_pos + vec2<f32>(0.0, half_s);
        let left = world_pos + vec2<f32>(-half_s, 0.0);
        draw_thick_line_ctx(top, right, 1.0, vec4<f32>(blended_color_displacer, 0.9), ctx);
        draw_thick_line_ctx(right, bottom, 1.0, vec4<f32>(blended_color_displacer, 0.9), ctx);
        draw_thick_line_ctx(bottom, left, 1.0, vec4<f32>(blended_color_displacer, 0.9), ctx);
        draw_thick_line_ctx(left, top, 1.0, vec4<f32>(blended_color_displacer, 0.9), ctx);
    }

    // 9. ORGAN: Alpha/Beta Sensors - visual marker scaled by sensing radius
    if (amino_props.is_alpha_sensor || amino_props.is_beta_sensor) {
        // Extract organ parameters to calculate actual sensor radius
        let organ_param = get_organ_param(part.part_type);
        let modifier_index = u32((f32(organ_param) / 255.0) * 19.0);
        let promoter_props = get_amino_acid_properties(base_type);
        let modifier_props = get_amino_acid_properties(modifier_index);
        let combined_param = promoter_props.parameter1 + modifier_props.parameter1;

        // Calculate actual sensor radius (same formula as in sample_stochastic_gaussian)
        let base_radius = 100.0;
        let radius_variation = combined_param * 100.0;
        let sensor_radius = abs(base_radius + radius_variation);

        // Scale visual marker based on sensor radius (normalized to typical range)
        // Typical range: 0-300, so normalize and scale for visibility
        let visual_scale = clamp(sensor_radius / 200.0, 0.3, 4.0);
        let marker_size = part.size * 1.5 * visual_scale;

        // Choose color based on sensor type and signal polarity
        var sensor_color = vec3<f32>(0.0);
        if (amino_props.is_alpha_sensor) {
            // Green for positive polarity, cyan for negative polarity
            sensor_color = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 1.0), combined_param < 0.0);
        } else {
            // Red for positive polarity, magenta for negative polarity
            sensor_color = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 1.0), combined_param < 0.0);
        }
        let blended_sensor_color = mix(sensor_color, agent_color, params.agent_color_blend * 0.3);

        // Draw circle marker
        draw_filled_circle_ctx(world_pos, marker_size, vec4<f32>(blended_sensor_color, 0.8), ctx);

        // Draw outline circle to indicate sensing range at high zoom

        if (params.camera_zoom > 80.0) {
            let zoom_fade = clamp((params.camera_zoom - 8.0) / 12.0, 0.0, 1.0);
            let outline_alpha = 0.15 * zoom_fade;
            let outline_color = vec4<f32>(blended_sensor_color, outline_alpha);
            let segments = 32u;
            var prev_outline = world_pos + vec2<f32>(sensor_radius, 0.0);
            for (var s = 1u; s <= segments; s++) {
                let t = f32(s) / f32(segments);
                let ang = t * 6.28318530718;
                let p = world_pos + vec2<f32>(cos(ang) * sensor_radius, sin(ang) * sensor_radius);
                draw_thick_line_ctx(prev_outline, p, 0.3, outline_color, ctx);
                prev_outline = p;
            }
        }
    }

    // 10. ORGAN: Sine Wave Clock - large pulsating circle
    if (amino_props.is_clock) {
        // Get clock signal from _pad.x (stored during signal update pass)
        let clock_signal = part._pad.x; // Range: -1 to +1

        // Decode promoter type from part_type parameter (bit 7)
        let organ_param = get_organ_param(part.part_type);
        let is_C_promoter = ((organ_param & 128u) != 0u);

        // Dark green for K promoter (alpha), dark red for C promoter (beta)
        let clock_color = select(vec3<f32>(0.0, 0.5, 0.0), vec3<f32>(0.5, 0.0, 0.0), is_C_promoter);

        // Pulsate size based on signal output
        // Map sine output (-1 to +1) to size multiplier (0.7 to 1.3)
        let size_multiplier = 1.0 + clock_signal * 0.3;
        let pulsating_size = part.size * size_multiplier;

        // Draw large filled circle with full opacity
        draw_filled_circle_ctx(world_pos, pulsating_size, vec4<f32>(clock_color, 1.0), ctx);
    }
}

// Draw a selection circle around an agent
fn draw_selection_circle(center_pos: vec2<f32>, agent_id: u32, body_count: u32) {
    if (params.draw_enabled == 0u) { return; }
    // Calculate approximate radius based on body size from the agent's actual body
    var max_dist = 20.0; // minimum radius
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let part = agents_out[agent_id].body[i];
        let dist = length(part.pos) + part.size;
        max_dist = max(max_dist, dist);
    }

    let color = vec4<f32>(1.0, 1.0, 0.0, 1.0); // Yellow crosshair

    // Draw crosshair with fixed radius (4 long arms)
    let fixed_radius = 25.0;  // Fixed distance from center
    let arm_length = 30.0;    // Length of each arm

    // Top arm
    draw_line(center_pos + vec2<f32>(0.0, fixed_radius), center_pos + vec2<f32>(0.0, fixed_radius + arm_length), color);
    // Right arm
    draw_line(center_pos + vec2<f32>(fixed_radius, 0.0), center_pos + vec2<f32>(fixed_radius + arm_length, 0.0), color);
    // Bottom arm
    draw_line(center_pos + vec2<f32>(0.0, -fixed_radius), center_pos + vec2<f32>(0.0, -fixed_radius - arm_length), color);
    // Left arm
    draw_line(center_pos + vec2<f32>(-fixed_radius, 0.0), center_pos + vec2<f32>(-fixed_radius - arm_length, 0.0), color);
}

// ============================================================================
// HELPER FUNCTIONS FOR DRAWING
// ============================================================================

// Inspector rendering context (pass vec2(-1.0) for use_inspector_coords to disable)
struct InspectorContext {
    use_inspector_coords: vec2<f32>,  // if x >= 0, use inspector mode
    center: vec2<f32>,                // center of preview window
    scale: f32,                       // scale factor for inspector
    offset: vec2<f32>,                // offset to actual buffer position
}

// Helper function to draw a thick line in screen space
fn draw_thick_line(p0: vec2<f32>, p1: vec2<f32>, thickness: f32, color: vec4<f32>) {
    draw_thick_line_ctx(p0, p1, thickness, color, InspectorContext(vec2<f32>(-1.0), vec2<f32>(0.0), 1.0, vec2<f32>(0.0)));
}

fn draw_thick_line_ctx(p0: vec2<f32>, p1: vec2<f32>, thickness: f32, color: vec4<f32>, ctx: InspectorContext) {
    var screen_p0: vec2<i32>;
    var screen_p1: vec2<i32>;
    var screen_thickness: i32;

    if (ctx.use_inspector_coords.x >= 0.0) {
        // Inspector mode: direct coordinate mapping
        screen_p0 = vec2<i32>(i32(ctx.center.x + p0.x * ctx.scale), i32(ctx.center.y + p0.y * ctx.scale));
        screen_p1 = vec2<i32>(i32(ctx.center.x + p1.x * ctx.scale), i32(ctx.center.y + p1.y * ctx.scale));
        screen_thickness = clamp(i32(thickness * ctx.scale), 0, 50);  // Clamp to prevent overflow
    } else {
        // World mode: use world-to-screen conversion
        screen_p0 = world_to_screen(p0);
        screen_p1 = world_to_screen(p1);
        let world_to_screen_scale = params.window_width / (params.grid_size / params.camera_zoom);
        screen_thickness = clamp(i32(thickness * world_to_screen_scale), 0, 50);  // Clamp to prevent overflow
    }

    // Optimized capsule drawing: rectangle + endpoint circles
    let dx = f32(screen_p1.x - screen_p0.x);
    let dy = f32(screen_p1.y - screen_p0.y);
    let len = sqrt(dx * dx + dy * dy);

    if (len < 0.5) {
        // Degenerate case: just draw a circle
        draw_filled_circle_optimized(screen_p0, f32(screen_thickness), color, ctx);
        return;
    }

    // Normalized direction and perpendicular
    let dir_x = dx / len;
    let dir_y = dy / len;
    let perp_x = -dir_y;
    let perp_y = dir_x;

    // Calculate bounding box for the capsule
    let half_thick = f32(screen_thickness);
    let min_x = min(screen_p0.x, screen_p1.x) - screen_thickness;
    let max_x = max(screen_p0.x, screen_p1.x) + screen_thickness;
    let min_y = min(screen_p0.y, screen_p1.y) - screen_thickness;
    let max_y = max(screen_p0.y, screen_p1.y) + screen_thickness;

    // Iterate only over bounding box (much smaller than full screen)
    for (var py = min_y; py <= max_y; py++) {
        for (var px = min_x; px <= max_x; px++) {
            let pixel_x = f32(px);
            let pixel_y = f32(py);

            // Vector from p0 to pixel
            let to_pixel_x = pixel_x - f32(screen_p0.x);
            let to_pixel_y = pixel_y - f32(screen_p0.y);

            // Project onto line direction to get position along line (0 to len)
            let t = to_pixel_x * dir_x + to_pixel_y * dir_y;

            // Distance to capsule axis and gradient position
            var dist_sq: f32;
            var gradient_t: f32;

            if (t < 0.0) {
                // Before p0: distance to p0
                dist_sq = to_pixel_x * to_pixel_x + to_pixel_y * to_pixel_y;
                gradient_t = 0.0;
            } else if (t > len) {
                // After p1: distance to p1
                let to_p1_x = pixel_x - f32(screen_p1.x);
                let to_p1_y = pixel_y - f32(screen_p1.y);
                dist_sq = to_p1_x * to_p1_x + to_p1_y * to_p1_y;
                gradient_t = 1.0;
            } else {
                // Between p0 and p1: perpendicular distance to line
                let perp_dist = to_pixel_x * perp_x + to_pixel_y * perp_y;
                dist_sq = perp_dist * perp_dist;
                gradient_t = t / len;
            }

            // Check if pixel is within capsule radius
            if (dist_sq <= half_thick * half_thick) {
                // Cylindrical surface lighting with curved surface
                // Calculate the point on the cylinder axis closest to this pixel
                // Clamp t to [0, len] so normals always point perpendicular to cylinder axis
                let t_clamped = clamp(t, 0.0, len);
                let axis_point_x = f32(screen_p0.x) + t_clamped * dir_x;
                let axis_point_y = f32(screen_p0.y) + t_clamped * dir_y;

                // Calculate radial distance from axis
                let radial_offset_x = pixel_x - axis_point_x;
                let radial_offset_y = pixel_y - axis_point_y;
                let radial_dist_sq = radial_offset_x * radial_offset_x + radial_offset_y * radial_offset_y;
                let radial_factor_sq = radial_dist_sq / (half_thick * half_thick);  // normalized (0 to 1)

                // Calculate z-component for curved cylinder surface (like a sphere cross-section)
                let z_sq = 1.0 - radial_factor_sq;
                let z = sqrt(max(z_sq, 0.0));

                // Surface normal with curved profile
                let surface_normal = normalize(vec3<f32>(radial_offset_x / half_thick, radial_offset_y / half_thick, z));

                // Light direction from params
                let light_dir = normalize(vec3<f32>(params.light_dir_x, params.light_dir_y, params.light_dir_z));

                // Lambertian diffuse lighting
                let diffuse = max(dot(surface_normal, light_dir), 0.0);

                // Use material color as base, lighten lit areas (dodge-like)
                let highlight = diffuse * params.light_power;
                let lighting = 1.0 + highlight;  // 1.0 = base color, >1.0 = brightened

                let shaded_color = vec4<f32>(
                    color.rgb * lighting,
                    color.a
                );

                var screen_pos = vec2<i32>(px, py);
                var idx: u32;
                var in_bounds = false;

                if (ctx.use_inspector_coords.x >= 0.0) {
                    // Inspector mode: offset to actual buffer position and check inspector bounds
                    let buffer_pos = screen_pos + vec2<i32>(i32(ctx.offset.x), i32(ctx.offset.y));
                    // Allow drawing anywhere in the inspector area (300px wide, full height)
                    if (buffer_pos.x >= i32(ctx.offset.x) && buffer_pos.x < i32(ctx.offset.x) + i32(INSPECTOR_WIDTH) &&
                        buffer_pos.y >= 0 && buffer_pos.y < i32(params.window_height)) {
                        idx = u32(buffer_pos.y) * params.visual_stride + u32(buffer_pos.x);
                        in_bounds = true;
                    }
                } else {
                    // World mode: check screen bounds
                    // Exclude inspector area if inspector is active (selected_agent_index != u32::MAX)
                    let inspector_active = params.selected_agent_index != 0xFFFFFFFFu;
                    let max_x = select(i32(params.window_width),
                                       i32(params.window_width) - i32(INSPECTOR_WIDTH),
                                       inspector_active);
                    if (screen_pos.x >= 0 && screen_pos.x < max_x &&
                        screen_pos.y >= 0 && screen_pos.y < i32(params.window_height)) {
                        idx = screen_to_grid_index(screen_pos);
                        in_bounds = true;
                    }
                }

                if (in_bounds) {
                    agent_grid[idx] = shaded_color;
                }
            }
        }
    }
}

// Thin line version: draws 1-pixel wide lines using Bresenham-like algorithm (for technical overlays)
fn draw_thin_line_ctx(p0: vec2<f32>, p1: vec2<f32>, color: vec4<f32>, ctx: InspectorContext) {
    var screen_p0: vec2<i32>;
    var screen_p1: vec2<i32>;

    if (ctx.use_inspector_coords.x >= 0.0) {
        screen_p0 = vec2<i32>(i32(ctx.center.x + p0.x * ctx.scale), i32(ctx.center.y + p0.y * ctx.scale));
        screen_p1 = vec2<i32>(i32(ctx.center.x + p1.x * ctx.scale), i32(ctx.center.y + p1.y * ctx.scale));
    } else {
        screen_p0 = world_to_screen(p0);
        screen_p1 = world_to_screen(p1);
    }

    let dx = abs(screen_p1.x - screen_p0.x);
    let dy = abs(screen_p1.y - screen_p0.y);
    let sx = select(-1, 1, screen_p0.x < screen_p1.x);
    let sy = select(-1, 1, screen_p0.y < screen_p1.y);
    var err = dx - dy;

    var x = screen_p0.x;
    var y = screen_p0.y;

    // Bresenham's line algorithm
    for (var i = 0; i < 10000; i++) {
        var screen_pos = vec2<i32>(x, y);
        var idx: u32;
        var in_bounds = false;

        if (ctx.use_inspector_coords.x >= 0.0) {
            let buffer_pos = screen_pos + vec2<i32>(i32(ctx.offset.x), i32(ctx.offset.y));
            if (buffer_pos.x >= i32(ctx.offset.x) && buffer_pos.x < i32(ctx.offset.x) + i32(INSPECTOR_WIDTH) &&
                buffer_pos.y >= 0 && buffer_pos.y < i32(params.window_height)) {
                idx = u32(buffer_pos.y) * params.visual_stride + u32(buffer_pos.x);
                in_bounds = true;
            }
        } else {
            let inspector_active = params.selected_agent_index != 0xFFFFFFFFu;
            let max_x = select(i32(params.window_width),
                               i32(params.window_width) - i32(INSPECTOR_WIDTH),
                               inspector_active);
            if (screen_pos.x >= 0 && screen_pos.x < max_x &&
                screen_pos.y >= 0 && screen_pos.y < i32(params.window_height)) {
                idx = screen_to_grid_index(screen_pos);
                in_bounds = true;
            }
        }

        if (in_bounds) {
            agent_grid[idx] = color;
        }

        if (x == screen_p1.x && y == screen_p1.y) {
            break;
        }

        let e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x += sx;
        }
        if (e2 < dx) {
            err += dx;
            y += sy;
        }
    }
}

// Gradient version: interpolates between start_color and end_color based on position along line
fn draw_thick_line_gradient_ctx(p0: vec2<f32>, p1: vec2<f32>, thickness: f32,
                                 start_color: vec4<f32>, end_color: vec4<f32>, ctx: InspectorContext) {
    // Convert world coordinates to screen coordinates
    var screen_p0: vec2<i32>;
    var screen_p1: vec2<i32>;
    var screen_thickness: i32;

    if (ctx.use_inspector_coords.x >= 0.0) {
        // Inspector mode: direct coordinate mapping
        screen_p0 = vec2<i32>(i32(ctx.center.x + p0.x * ctx.scale), i32(ctx.center.y + p0.y * ctx.scale));
        screen_p1 = vec2<i32>(i32(ctx.center.x + p1.x * ctx.scale), i32(ctx.center.y + p1.y * ctx.scale));
        screen_thickness = clamp(i32(thickness * ctx.scale), 0, 50);
    } else {
        // World mode: use world-to-screen conversion
        screen_p0 = world_to_screen(p0);
        screen_p1 = world_to_screen(p1);
        let world_to_screen_scale = params.window_width / (params.grid_size / params.camera_zoom);
        screen_thickness = clamp(i32(thickness * world_to_screen_scale), 0, 50);
    }

    let dx = f32(screen_p1.x - screen_p0.x);
    let dy = f32(screen_p1.y - screen_p0.y);
    let len = sqrt(dx * dx + dy * dy);

    if (len < 0.5) {
        // Degenerate case: just draw a circle with average color
        let avg_color = mix(start_color, end_color, 0.5);
        draw_filled_circle_optimized(screen_p0, f32(screen_thickness), avg_color, ctx);
        return;
    }

    let dir_x = dx / len;
    let dir_y = dy / len;
    let perp_x = -dir_y;
    let perp_y = dir_x;
    let half_thick = f32(screen_thickness);

    // Calculate bounding box
    let bbox_min_x = min(screen_p0.x, screen_p1.x) - screen_thickness;
    let bbox_min_y = min(screen_p0.y, screen_p1.y) - screen_thickness;
    let bbox_max_x = max(screen_p0.x, screen_p1.x) + screen_thickness;
    let bbox_max_y = max(screen_p0.y, screen_p1.y) + screen_thickness;

    for (var py = bbox_min_y; py <= bbox_max_y; py++) {
        for (var px = bbox_min_x; px <= bbox_max_x; px++) {
            let pixel_x = f32(px);
            let pixel_y = f32(py);

            let to_pixel_x = pixel_x - f32(screen_p0.x);
            let to_pixel_y = pixel_y - f32(screen_p0.y);

            let t = to_pixel_x * dir_x + to_pixel_y * dir_y;

            var dist_sq: f32;
            var gradient_t: f32;

            if (t <= 0.0) {
                dist_sq = to_pixel_x * to_pixel_x + to_pixel_y * to_pixel_y;
                gradient_t = 0.0;
            } else if (t >= len) {
                let to_p1_x = pixel_x - f32(screen_p1.x);
                let to_p1_y = pixel_y - f32(screen_p1.y);
                dist_sq = to_p1_x * to_p1_x + to_p1_y * to_p1_y;
                gradient_t = 1.0;
            } else {
                let perp_dist = to_pixel_x * perp_x + to_pixel_y * perp_y;
                dist_sq = perp_dist * perp_dist;
                gradient_t = t / len;
            }

            if (dist_sq <= half_thick * half_thick) {
                // Interpolate color based on position along line
                let base_color = mix(start_color, end_color, gradient_t);

                // Cylindrical surface lighting with curved surface
                // Calculate the point on the cylinder axis closest to this pixel
                // Clamp t to [0, len] so normals always point perpendicular to cylinder axis
                let t_clamped = clamp(t, 0.0, len);
                let axis_point_x = f32(screen_p0.x) + t_clamped * dir_x;
                let axis_point_y = f32(screen_p0.y) + t_clamped * dir_y;

                // Calculate radial distance from axis
                let radial_offset_x = pixel_x - axis_point_x;
                let radial_offset_y = pixel_y - axis_point_y;
                let radial_dist_sq = radial_offset_x * radial_offset_x + radial_offset_y * radial_offset_y;
                let radial_factor_sq = radial_dist_sq / (half_thick * half_thick);  // normalized (0 to 1)

                // Calculate z-component for curved cylinder surface (like a sphere cross-section)
                let z_sq = 1.0 - radial_factor_sq;
                let z = sqrt(max(z_sq, 0.0));

                // Surface normal with curved profile
                let surface_normal = normalize(vec3<f32>(radial_offset_x / half_thick, radial_offset_y / half_thick, z));

                let light_dir = normalize(vec3<f32>(params.light_dir_x, params.light_dir_y, params.light_dir_z));
                let diffuse = max(dot(surface_normal, light_dir), 0.0);

                // Use material color as base, lighten lit areas (dodge-like)
                let highlight = diffuse * params.light_power;
                let lighting = 1.0 + highlight;  // 1.0 = base color, >1.0 = brightened

                let color = vec4<f32>(base_color.rgb * lighting, base_color.a);

                var screen_pos = vec2<i32>(px, py);
                var idx: u32;
                var in_bounds = false;

                if (ctx.use_inspector_coords.x >= 0.0) {
                    let buffer_pos = screen_pos + vec2<i32>(i32(ctx.offset.x), i32(ctx.offset.y));
                    if (buffer_pos.x >= i32(ctx.offset.x) && buffer_pos.x < i32(ctx.offset.x) + i32(INSPECTOR_WIDTH) &&
                        buffer_pos.y >= 0 && buffer_pos.y < i32(params.window_height)) {
                        idx = u32(buffer_pos.y) * params.visual_stride + u32(buffer_pos.x);
                        in_bounds = true;
                    }
                } else {
                    let inspector_active = params.selected_agent_index != 0xFFFFFFFFu;
                    let max_x = select(i32(params.window_width),
                                       i32(params.window_width) - i32(INSPECTOR_WIDTH),
                                       inspector_active);
                    if (screen_pos.x >= 0 && screen_pos.x < max_x &&
                        screen_pos.y >= 0 && screen_pos.y < i32(params.window_height)) {
                        idx = screen_to_grid_index(screen_pos);
                        in_bounds = true;
                    }
                }

                if (in_bounds) {
                    agent_grid[idx] = color;
                }
            }
        }
    }
}

// Helper for drawing optimized filled circles (used by optimized thick line)
fn draw_filled_circle_optimized(center: vec2<i32>, radius: f32, color: vec4<f32>, ctx: InspectorContext) {
    let radius_i = i32(ceil(radius));
    let radius_sq = radius * radius;

    for (var dy = -radius_i; dy <= radius_i; dy++) {
        for (var dx = -radius_i; dx <= radius_i; dx++) {
            let dist_sq = f32(dx * dx + dy * dy);
            if (dist_sq <= radius_sq) {
                // Radial gradient shading: lighter at center, darker at edges
                let dist_factor = sqrt(dist_sq) / radius;  // 0.0 at center, 1.0 at edge
                let shaded_color = vec4<f32>(
                    mix(color.rgb, color.rgb * 0.5, dist_factor),  // Interpolate brightness
                    color.a
                );

                var screen_pos = center + vec2<i32>(dx, dy);
                var idx: u32;
                var in_bounds = false;

                if (ctx.use_inspector_coords.x >= 0.0) {
                    // Inspector mode
                    let buffer_pos = screen_pos + vec2<i32>(i32(ctx.offset.x), i32(ctx.offset.y));
                    if (buffer_pos.x >= i32(ctx.offset.x) && buffer_pos.x < i32(ctx.offset.x + 280.0) &&
                        buffer_pos.y >= i32(ctx.offset.y) && buffer_pos.y < i32(ctx.offset.y + 280.0)) {
                        idx = u32(buffer_pos.y) * params.visual_stride + u32(buffer_pos.x);
                        in_bounds = true;
                    }
                } else {
                    // World mode
                    let max_x = select(i32(params.window_width), i32(params.window_width) - i32(INSPECTOR_WIDTH), params.selected_agent_index != 0xFFFFFFFFu);
                    // World mode
                    if (screen_pos.x >= 0 && screen_pos.x < max_x &&
                    // World mode
                        screen_pos.y >= 0 && screen_pos.y < i32(params.window_height)) {
                        idx = screen_to_grid_index(screen_pos);
                        in_bounds = true;
                    }
                }

                if (in_bounds) {
                    agent_grid[idx] = shaded_color;
                }
            }
        }
    }
}

// Helper function to draw a clean circle outline in screen space
fn draw_circle(center: vec2<f32>, radius: f32, color: vec4<f32>) {
    // Convert world position to screen coordinates
    let screen_center = world_to_screen(center);

    // Calculate screen-space radius (accounting for zoom)
    let world_to_screen_scale = params.window_width / (params.grid_size / params.camera_zoom);
    let screen_radius = radius * world_to_screen_scale;

    let radius_i = i32(ceil(screen_radius));
    let line_thickness = 1.0; // pixels

    for (var dy = -radius_i; dy <= radius_i; dy++) {
        for (var dx = -radius_i; dx <= radius_i; dx++) {
            let offset = vec2<f32>(f32(dx), f32(dy));
            let dist = length(offset);

            if (abs(dist - screen_radius) < line_thickness) {
                let screen_pos = screen_center + vec2<i32>(dx, dy);

                // Check if in visible window bounds
                // Exclude inspector area if inspector is active
                let max_x = select(i32(params.window_width), i32(params.window_width) - i32(INSPECTOR_WIDTH), params.selected_agent_index != 0xFFFFFFFFu);
                if (screen_pos.x >= 0 && screen_pos.x < max_x &&
                    screen_pos.y >= 0 && screen_pos.y < i32(params.window_height)) {

                    let idx = screen_to_grid_index(screen_pos);
                    agent_grid[idx] = color;
                }
            }
        }
    }
}

// Helper: draw a filled circle in screen space
fn draw_filled_circle(center: vec2<f32>, radius: f32, color: vec4<f32>) {
    draw_filled_circle_ctx(center, radius, color, InspectorContext(vec2<f32>(-1.0), vec2<f32>(0.0), 1.0, vec2<f32>(0.0)));
}

fn draw_filled_circle_ctx(center: vec2<f32>, radius: f32, color: vec4<f32>, ctx: InspectorContext) {
    var screen_center: vec2<i32>;
    var screen_radius: f32;

    if (ctx.use_inspector_coords.x >= 0.0) {
        // Inspector mode
        screen_center = vec2<i32>(i32(ctx.center.x + center.x * ctx.scale), i32(ctx.center.y + center.y * ctx.scale));
        screen_radius = clamp(radius * ctx.scale, 0.0, 50.0);  // Clamp to prevent overflow
    } else {
        // World mode
        screen_center = world_to_screen(center);
        let world_to_screen_scale = params.window_width / (params.grid_size / params.camera_zoom);
        screen_radius = clamp(radius * world_to_screen_scale, 0.0, 50.0);  // Clamp to prevent overflow
    }

    let radius_i = i32(ceil(screen_radius));

    for (var dy = -radius_i; dy <= radius_i; dy++) {
        for (var dx = -radius_i; dx <= radius_i; dx++) {
            let offset = vec2<f32>(f32(dx), f32(dy));
            let dist2 = dot(offset, offset);
            if (dist2 <= screen_radius * screen_radius) {
                // Spherical surface lighting (optimized: avoid extra sqrt)
                let radius_sq = screen_radius * screen_radius;
                let dist_factor_sq = dist2 / radius_sq;  // (dist/radius)Â²

                // Calculate 3D surface normal for a sphere
                // In 2D view, we see a circle; assume sphere extends in z-direction
                let z_sq = 1.0 - dist_factor_sq;  // xÂ² + yÂ² + zÂ² = 1
                let z = sqrt(max(z_sq, 0.0));
                let surface_normal = normalize(vec3<f32>(offset.x / screen_radius, offset.y / screen_radius, z));

                // Light direction from params
                let light_dir = normalize(vec3<f32>(params.light_dir_x, params.light_dir_y, params.light_dir_z));

                // Lambertian diffuse lighting
                let diffuse = max(dot(surface_normal, light_dir), 0.0);

                // Use material color as base, lighten lit areas (dodge-like)
                let highlight = diffuse * params.light_power;
                let lighting = 1.0 + highlight;  // 1.0 = base color, >1.0 = brightened

                let shaded_color = vec4<f32>(
                    color.rgb * lighting,
                    color.a
                );

                var screen_pos = screen_center + vec2<i32>(dx, dy);
                var idx: u32;
                var in_bounds = false;

                if (ctx.use_inspector_coords.x >= 0.0) {
                    // Inspector mode
                    let buffer_pos = screen_pos + vec2<i32>(i32(ctx.offset.x), i32(ctx.offset.y));
                    if (buffer_pos.x >= i32(ctx.offset.x) && buffer_pos.x < i32(ctx.offset.x + 280.0) &&
                        buffer_pos.y >= i32(ctx.offset.y) && buffer_pos.y < i32(ctx.offset.y + 280.0)) {
                        idx = u32(buffer_pos.y) * params.visual_stride + u32(buffer_pos.x);
                        in_bounds = true;
                    }
                } else {
                    // World mode
                    let max_x = select(i32(params.window_width), i32(params.window_width) - i32(INSPECTOR_WIDTH), params.selected_agent_index != 0xFFFFFFFFu);
                    if (screen_pos.x >= 0 && screen_pos.x < max_x &&
                        screen_pos.y >= 0 && screen_pos.y < i32(params.window_height)) {
                        idx = screen_to_grid_index(screen_pos);
                        in_bounds = true;
                    }
                }

                if (in_bounds) {
                    // Alpha blending: blend new color with existing background
                    let bg_color = agent_grid[idx];
                    let src_alpha = shaded_color.a;
                    let inv_alpha = 1.0 - src_alpha;
                    let blended = vec4<f32>(
                        shaded_color.rgb * src_alpha + bg_color.rgb * inv_alpha,
                        max(shaded_color.a, bg_color.a)
                    );
                    agent_grid[idx] = blended;
                }
            }
        }
    }
}

// Helper: draw a 5-pointed star in screen space
fn draw_star(center: vec2<f32>, radius: f32, color: vec4<f32>) {
    // Convert world position to screen coordinates
    let screen_center = world_to_screen(center);

    // Calculate screen-space radius (accounting for zoom)
    let world_to_screen_scale = params.window_width / (params.grid_size / params.camera_zoom);
    let screen_radius = radius * world_to_screen_scale;

    // 5-pointed star with 10 points (5 outer, 5 inner)
    let num_points = 5u;
    let inner_radius = screen_radius * 0.38; // Inner points at ~38% of outer radius

    // Draw star as lines connecting the points
    for (var i = 0u; i < num_points; i++) {
        // Calculate angles (starting from top, going clockwise)
        let angle_outer = -1.57079632679 + f32(i) * 6.28318530718 / f32(num_points);
        let angle_inner = angle_outer + 3.14159265359 / f32(num_points);
        let angle_next_outer = -1.57079632679 + f32((i + 1u) % num_points) * 6.28318530718 / f32(num_points);

        // Calculate positions
        let outer_x = screen_center.x + i32(cos(angle_outer) * screen_radius);
        let outer_y = screen_center.y + i32(sin(angle_outer) * screen_radius);
        let inner_x = screen_center.x + i32(cos(angle_inner) * inner_radius);
        let inner_y = screen_center.y + i32(sin(angle_inner) * inner_radius);
        let next_outer_x = screen_center.x + i32(cos(angle_next_outer) * screen_radius);
        let next_outer_y = screen_center.y + i32(sin(angle_next_outer) * screen_radius);

        // Draw lines: outer -> inner -> next_outer
        draw_line_pixels(vec2<i32>(outer_x, outer_y), vec2<i32>(inner_x, inner_y), color);
        draw_line_pixels(vec2<i32>(inner_x, inner_y), vec2<i32>(next_outer_x, next_outer_y), color);
    }
}

// Helper: draw an 8-point asterisk for vampire mouths
fn draw_asterisk_8_ctx(center: vec2<f32>, radius: f32, color: vec4<f32>, ctx: InspectorContext) {
    let angle_step = 0.39269908; // PI / 8 = 22.5 degrees

    for (var i = 0u; i < 4u; i++) {
        let angle = f32(i) * angle_step * 2.0;
        let dx = cos(angle) * radius;
        let dy = sin(angle) * radius;
        let start = center - vec2<f32>(dx, dy);
        let end = center + vec2<f32>(dx, dy);
        draw_thick_line_ctx(start, end, 1.5, color, ctx);
    }
}

// Helper: draw an asterisk (*) with 4 crossing lines (vertical, horizontal, 2 diagonals)
fn draw_asterisk(center: vec2<f32>, radius: f32, color: vec4<f32>) {
    draw_asterisk_ctx(center, radius, color, InspectorContext(vec2<f32>(-1.0), vec2<f32>(0.0), 1.0, vec2<f32>(0.0)));
}

fn draw_asterisk_ctx(center: vec2<f32>, radius: f32, color: vec4<f32>, ctx: InspectorContext) {
    // Draw 4 lines: vertical, horizontal, and two diagonals
    let diag_offset = radius * 0.70710678; // radius / sqrt(2)

    // Vertical line
    let up = center + vec2<f32>(0.0, -radius);
    let down = center + vec2<f32>(0.0, radius);
    draw_thick_line_ctx(up, down, 1.0, color, ctx);

    // Horizontal line
    let left = center + vec2<f32>(-radius, 0.0);
    let right = center + vec2<f32>(radius, 0.0);
    draw_thick_line_ctx(left, right, 1.0, color, ctx);

    // Diagonal 1 (top-left to bottom-right)
    let tl = center + vec2<f32>(-diag_offset, -diag_offset);
    let br = center + vec2<f32>(diag_offset, diag_offset);
    draw_thick_line_ctx(tl, br, 1.0, color, ctx);

    // Diagonal 2 (top-right to bottom-left)
    let tr = center + vec2<f32>(diag_offset, -diag_offset);
    let bl = center + vec2<f32>(-diag_offset, diag_offset);
    draw_thick_line_ctx(tr, bl, 1.0, color, ctx);
}

// Helper: draw a cloud-like shape (fuzzy circle with some random bumps)
fn draw_cloud(center: vec2<f32>, radius: f32, color: vec4<f32>, seed: u32) {
    draw_cloud_ctx(center, radius, color, seed, InspectorContext(vec2<f32>(-1.0), vec2<f32>(0.0), 1.0, vec2<f32>(0.0)));
}

fn draw_cloud_ctx(center: vec2<f32>, radius: f32, color: vec4<f32>, seed: u32, ctx: InspectorContext) {
    // Optimized: single-pass cloud rendering instead of 9 separate circle draws
    var screen_center: vec2<i32>;
    var screen_radius: f32;

    if (ctx.use_inspector_coords.x >= 0.0) {
        screen_center = vec2<i32>(i32(ctx.center.x + center.x * ctx.scale), i32(ctx.center.y + center.y * ctx.scale));
        screen_radius = clamp(radius * ctx.scale, 0.0, 50.0);
    } else {
        screen_center = world_to_screen(center);
        let world_to_screen_scale = params.window_width / (params.grid_size / params.camera_zoom);
        screen_radius = clamp(radius * world_to_screen_scale, 0.0, 50.0);
    }

    // Pre-calculate all puff centers and radii
    let num_puffs = 8u;
    var puff_centers: array<vec2<f32>, 9>;
    var puff_radii: array<f32, 9>;

    // Central puff (larger)
    puff_centers[0] = vec2<f32>(f32(screen_center.x), f32(screen_center.y));
    puff_radii[0] = screen_radius * 0.7;

    // Surrounding puffs
    for (var i = 0u; i < num_puffs; i++) {
        let angle = f32(i) * 6.28318530718 / f32(num_puffs);
        let hash_val = hash_f32(seed * (i + 1u) * 2654435761u);
        let offset_dist = screen_radius * 0.4 * hash_val;
        puff_centers[i + 1u] = vec2<f32>(
            f32(screen_center.x) + cos(angle) * offset_dist,
            f32(screen_center.y) + sin(angle) * offset_dist
        );
        puff_radii[i + 1u] = screen_radius * (0.5 + 0.3 * hash_val);
    }

    // Find bounding box for all puffs
    var min_x = screen_center.x;
    var max_x = screen_center.x;
    var min_y = screen_center.y;
    var max_y = screen_center.y;

    for (var i = 0u; i < 9u; i++) {
        let r = i32(ceil(puff_radii[i]));
        min_x = min(min_x, i32(puff_centers[i].x) - r);
        max_x = max(max_x, i32(puff_centers[i].x) + r);
        min_y = min(min_y, i32(puff_centers[i].y) - r);
        max_y = max(max_y, i32(puff_centers[i].y) + r);
    }

    // Single pass over bounding box, check distance to all puffs
    for (var py = min_y; py <= max_y; py++) {
        for (var px = min_x; px <= max_x; px++) {
            let pixel_pos = vec2<f32>(f32(px), f32(py));
            var inside_any_puff = false;

            // Check if pixel is inside any of the 9 puffs
            for (var i = 0u; i < 9u; i++) {
                let dx = pixel_pos.x - puff_centers[i].x;
                let dy = pixel_pos.y - puff_centers[i].y;
                let dist_sq = dx * dx + dy * dy;
                if (dist_sq <= puff_radii[i] * puff_radii[i]) {
                    inside_any_puff = true;
                    break;
                }
            }

            if (inside_any_puff) {
                var screen_pos = vec2<i32>(px, py);
                var idx: u32;
                var in_bounds = false;

                if (ctx.use_inspector_coords.x >= 0.0) {
                    let buffer_pos = screen_pos + vec2<i32>(i32(ctx.offset.x), i32(ctx.offset.y));
                    if (buffer_pos.x >= i32(ctx.offset.x) && buffer_pos.x < i32(ctx.offset.x + 280.0) &&
                        buffer_pos.y >= i32(ctx.offset.y) && buffer_pos.y < i32(ctx.offset.y + 280.0)) {
                        idx = u32(buffer_pos.y) * params.visual_stride + u32(buffer_pos.x);
                        in_bounds = true;
                    }
                } else {
                    let max_x = select(i32(params.window_width), i32(params.window_width) - i32(INSPECTOR_WIDTH), params.selected_agent_index != 0xFFFFFFFFu);
                    if (screen_pos.x >= 0 && screen_pos.x < max_x &&
                        screen_pos.y >= 0 && screen_pos.y < i32(params.window_height)) {
                        idx = screen_to_grid_index(screen_pos);
                        in_bounds = true;
                    }
                }

                if (in_bounds) {
                    agent_grid[idx] = color;
                }
            }
        }
    }
}

// Helper: draw a particle jet (motion-blurred particles in a cone)
fn draw_particle_jet(origin: vec2<f32>, direction: vec2<f32>, length: f32, seed: u32, particle_count: u32) {
    draw_particle_jet_ctx(origin, direction, length, seed, particle_count, InspectorContext(vec2<f32>(-1.0), vec2<f32>(0.0), 1.0, vec2<f32>(0.0)));
}

fn draw_particle_jet_ctx(origin: vec2<f32>, direction: vec2<f32>, length: f32, seed: u32, particle_count: u32, ctx: InspectorContext) {
    // Draw motion-blurred particles spread in a cone
    let num_particles = clamp(particle_count, 2u, 10u);
    if (num_particles < 2u) { return; }
    let particle_color = vec4<f32>(0.2, 0.5, 1.0, 0.8); // Semi-transparent blue
    let cone_angle = 0.4; // Cone spread angle in radians (~23 degrees)

    for (var i = 0u; i < num_particles; i++) {
        // Generate two hash values for this particle
        let hash_val1 = hash_f32(seed * (i + 1u) * 2654435761u);
        let hash_val2 = hash_f32(seed * (i + 1u) * 1103515245u);

        // Distance along the jet (0 to 1)
        let denom = max(num_particles - 1u, 1u);
        let t = f32(i) / f32(denom);
        let distance = length * t * (0.7 + 0.6 * hash_val1);

        // Angular spread in cone (using hash to distribute evenly in cone)
        let angle_offset = (hash_val2 - 0.5) * cone_angle * (1.0 + t * 0.5); // Wider spread further out

        // Rotate direction by angle_offset
        let cos_angle = cos(angle_offset);
        let sin_angle = sin(angle_offset);
        let rotated_dir = vec2<f32>(
            direction.x * cos_angle - direction.y * sin_angle,
            direction.x * sin_angle + direction.y * cos_angle
        );

        // Calculate particle position
        let particle_pos = origin + rotated_dir * distance;

        // Motion blur: draw a short streak instead of a dot
        let streak_length = 1.6 * (1.0 - t * 0.35); // Longer streaks at base
        let streak_end = particle_pos + rotated_dir * streak_length;
        let streak_thickness = 0.45 * (1.0 - t * 0.6); // Thinner as they move away

        // Draw motion-blurred particle as a thick line
    let fade_color = vec4<f32>(particle_color.xyz, particle_color.w * (0.6 * (1.0 - t * 0.5)));
        draw_thick_line_ctx(particle_pos, streak_end, streak_thickness, fade_color, ctx);
    }
}
// ============================================================================
// COMPACT VECTOR FONT DATA (packed u32 format)
// Each u32 packs 4 coordinates as bytes: (p0.x, p0.y, p1.x, p1.y)
// Coordinates are scaled 0.0-1.0 â†’ 0-255 (decode with /255.0)
// Negative values (e.g., comma tail) are clamped to 0
// ============================================================================

var<private> FONT_SEGMENTS: array<u32, 160> = array<u32, 160>(
    0x00CC0033u,0xFFCC00CCu,0xFF33FFCCu,0x0033FF33u,0xFFCC0033u,0xFF800080u,0xCC4CFF80u,0xFFCCFF33u,
    0x80CCFFCCu,0x803380CCu,0x00338033u,0x00CC0033u,0xFFCCFF33u,0x00CCFFCCu,0x003300CCu,0x806680CCu,
    0x6633FF33u,0x66CC6633u,0xFFB200B2u,0xFF33FFCCu,0x8033FF33u,0x80CC8033u,0x00CC80CCu,0x003300CCu,
    0xFF33FFCCu,0x0033FF33u,0x00CC0033u,0x80CC00CCu,0x803380CCu,0xFFCCFF33u,0x0066FFCCu,0x00CC0033u,
    0xFFCC00CCu,0xFF33FFCCu,0x0033FF33u,0x80CC8033u,0x803380CCu,0xFF338033u,0xFFCCFF33u,0x00CCFFCCu,
    0x003300CCu,0xFF80001Au,0x00E6FF80u,0x66BF6640u,0xFF330033u,0xFFB2FF33u,0xBFCCFFB2u,0x80B2BFCCu,
    0x803380B2u,0x40CC80B2u,0x00B240CCu,0x003300B2u,0xFF33FFCCu,0x0033FF33u,0x00CC0033u,0xFF330033u,
    0xFF99FF33u,0xCCCCFF99u,0x33CCCCCCu,0x009933CCu,0x00330099u,0xFF33FFCCu,0x0033FF33u,0x80B28033u,
    0x00CC0033u,0xFF330033u,0xFFCCFF33u,0x80B28033u,0xFF33FFCCu,0x0033FF33u,0x00CC0033u,0x80CC00CCu,
    0x808080CCu,0xFF330033u,0xFFCC00CCu,0x80CC8033u,0xFF800080u,0x00B2004Cu,0xFFB2FF4Cu,0x3399FF99u,
    0x00663399u,0x33330066u,0xFF330033u,0x8033FFCCu,0x00CC8033u,0x0033FF33u,0x00CC0033u,0xFF1A001Au,
    0x8080FF1Au,0xFFE68080u,0x00E6FFE6u,0xFF330033u,0x00CCFF33u,0xFFCC00CCu,0x00CC0033u,0xFFCC00CCu,
    0xFF33FFCCu,0x0033FF33u,0xFF330033u,0xFFB2FF33u,0xBFCCFFB2u,0x80B2BFCCu,0x803380B2u,0x00CC0033u,
    0xFFCC00CCu,0xFF33FFCCu,0x0033FF33u,0x00E64C99u,0xFF330033u,0xFFB2FF33u,0xBFCCFFB2u,0x80B2BFCCu,
    0x803380B2u,0x00CC8080u,0xFF33FFCCu,0x8033FF33u,0x80CC8033u,0x00CC80CCu,0x003300CCu,0xFFCCFF33u,
    0x0080FF80u,0x3333FF33u,0x004C3333u,0x00B2004Cu,0x33CC00B2u,0xFFCC33CCu,0x0080FF1Au,0xFFE60080u,
    0x0033FF1Au,0x99800033u,0x00CC9980u,0xFFE600CCu,0xFFCC0033u,0x00CCFF33u,0x8080FF33u,0x8080FFCCu,
    0x00808080u,0xFFCCFF33u,0x0033FFCCu,0x00CC0033u,0x008C0073u,0x0D8C0D73u,0x00660080u,0x4C8C4C73u,
    0xB28CB273u,0x80CC8033u,0xCC803380u,0x80CC8033u,0x66CC6633u,0x99CC9933u,0xE64CCC33u,0x33CC1AB2u,
    0xFFCC0033u,0xB266FF99u,0x4C66B266u,0x00994C66u,0xB299FF66u,0x4C99B299u,0x00664C99u,0xFFCC0033u);

// Decode a packed segment into vec2 coordinates
fn unpack_segment(packed: u32) -> vec4<f32> {
    let p0x = f32((packed) & 0xFFu) / 255.0;
    let p0y = f32((packed >> 8u) & 0xFFu) / 255.0;
    let p1x = f32((packed >> 16u) & 0xFFu) / 255.0;
    let p1y = f32((packed >> 24u) & 0xFFu) / 255.0;
    return vec4<f32>(p0x, p0y, p1x, p1y);
}

// Compact font character data (offset + count per character)
struct FontChar {
    offset: u32,
    count: u32,
}

var<private> FONT_CHARS: array<FontChar, 47> = array<FontChar, 47>(
    // 0â€“9: digits
    FontChar(0u,   5u),   // '0'
    FontChar(5u,   2u),   // '1'
    FontChar(7u,   5u),   // '2'
    FontChar(12u,  4u),   // '3'
    FontChar(16u,  3u),   // '4'
    FontChar(19u,  5u),   // '5'
    FontChar(24u,  5u),   // '6'
    FontChar(29u,  2u),   // '7'
    FontChar(31u,  5u),   // '8'
    FontChar(36u,  5u),   // '9'
    // 10â€“35: Aâ€“Z
    FontChar(41u,  3u),   // 'A'
    FontChar(44u,  8u),   // 'B'
    FontChar(52u,  3u),   // 'C'
    FontChar(55u,  6u),   // 'D'
    FontChar(61u,  4u),   // 'E'
    FontChar(65u,  3u),   // 'F'
    FontChar(68u,  5u),   // 'G'
    FontChar(73u,  3u),   // 'H'
    FontChar(76u,  3u),   // 'I'
    FontChar(79u,  3u),   // 'J'
    FontChar(82u,  3u),   // 'K'
    FontChar(85u,  2u),   // 'L'
    FontChar(87u,  4u),   // 'M'
    FontChar(91u,  3u),   // 'N'
    FontChar(94u,  4u),   // 'O'
    FontChar(98u,  5u),   // 'P'
    FontChar(103u, 5u),   // 'Q'
    FontChar(108u, 6u),   // 'R'
    FontChar(114u, 5u),   // 'S'
    FontChar(119u, 2u),   // 'T'
    FontChar(121u, 5u),   // 'U'
    FontChar(126u, 2u),   // 'V'
    FontChar(128u, 4u),   // 'W'
    FontChar(132u, 2u),   // 'X'
    FontChar(134u, 3u),   // 'Y'
    FontChar(137u, 3u),   // 'Z'
    // 36â€“46: symbols
    FontChar(140u, 0u),   // ' ' (space)
    FontChar(140u, 2u),   // '.'
    FontChar(142u, 1u),   // ','
    FontChar(143u, 2u),   // ':'
    FontChar(145u, 1u),   // '-'
    FontChar(146u, 2u),   // '+'
    FontChar(148u, 2u),   // '='
    FontChar(150u, 3u),   // '%'
    FontChar(153u, 3u),   // '('
    FontChar(156u, 3u),   // ')'
    FontChar(159u, 1u)    // '/'
);
fn char_index(c: u32) -> i32 {
    if (c >= 48u && c <= 57u) { return i32(c - 48u); }
    if (c >= 65u && c <= 90u) { return i32(c - 65u + 10u); }
    if (c == 32u) { return 36; }
    if (c == 46u) { return 37; }
    if (c == 44u) { return 38; }
    if (c == 58u) { return 39; }
    if (c == 45u) { return 40; }
    if (c == 43u) { return 41; }
    if (c == 61u) { return 42; }
    if (c == 37u) { return 43; }
    if (c == 40u) { return 44; }
    if (c == 41u) { return 45; }
    if (c == 47u) { return 46; }
    return -1;
}

// Get character width (relative to height=1.0)
fn get_char_width(c: u32) -> f32 {
    if (c == 32u) { return 0.5; } // space
    if (c == 46u || c == 44u || c == 58u) { return 0.3; } // punctuation
    if (c == 73u || c == 49u) { return 0.5; } // 'I' and '1'
    if (c == 77u || c == 87u) { return 1.2; } // 'M' and 'W'
    return 1.0; // default width
}

// Draw a single character at position with specified height
fn draw_char_vector(pos: vec2<f32>, c: u32, height: f32, color: vec4<f32>, ctx: InspectorContext) -> f32 {
    let idx = char_index(c);
    if (idx < 0) {
        return height * 0.4; // fallback spacing for unsupported chars
    }

    let ch = FONT_CHARS[u32(idx)];
    let base = ch.offset;
    let seg_count = ch.count;
    let char_width = get_char_width(c) * height;

    // Use ~1px lines (user request)
    let line_thickness = max(1.0, height * 0.1);

    for (var i = 0u; i < seg_count; i++) {
        let seg = unpack_segment(FONT_SEGMENTS[base + i]);
        // Flip Y so font baseline is at bottom and screen Y grows downward
        let p0 = pos + vec2<f32>(seg.x * char_width, (1.0 - seg.y) * height);
        let p1 = pos + vec2<f32>(seg.z * char_width, (1.0 - seg.w) * height);
        draw_thick_line_ctx(p0, p1, line_thickness, color, ctx);
    }

    return char_width + height * 0.2; // width plus a bit of spacing
}

// Distance from point to segment (in character-local space)
fn point_segment_distance(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let ab_len_sq = max(dot(ab, ab), 1e-6);
    let t = clamp(dot(p - a, ab) / ab_len_sq, 0.0, 1.0);
    let proj = a + ab * t;
    return length(p - proj);
}

// Vector font mask evaluated per pixel (avoids race/flicker from draw_thick_line writes)
fn char_vector_mask(local_px: vec2<u32>, c: u32, height: f32) -> bool {
    let idx = char_index(c);
    if (idx < 0) { return false; }

    let ch = FONT_CHARS[u32(idx)];
    let base = ch.offset;
    let seg_count = ch.count;
    let char_width = get_char_width(c) * height;
    let line_thickness = max(1.0, height * 0.1);
    let half_thick = line_thickness * 0.5;
    let p = vec2<f32>(f32(local_px.x) + 0.5, f32(local_px.y) + 0.5);

    for (var i = 0u; i < seg_count; i++) {
        let seg = unpack_segment(FONT_SEGMENTS[base + i]);
        let p0 = vec2<f32>(seg.x * char_width, (1.0 - seg.y) * height);
        let p1 = vec2<f32>(seg.z * char_width, (1.0 - seg.w) * height);
        let d = point_segment_distance(p, p0, p1);
        if (d <= half_thick) {
            return true;
        }
    }
    return false;
}

// Draw a string at position with specified height
fn draw_string_vector(pos: vec2<f32>, text: ptr<function, array<u32, 32>>, length: u32, height: f32, color: vec4<f32>, ctx: InspectorContext) -> f32 {
    var cursor_x = pos.x;

    for (var i = 0u; i < length && i < 32u; i++) {
        let char_code = (*text)[i];
        let width = draw_char_vector(vec2<f32>(cursor_x, pos.y), char_code, height, color, ctx);
        cursor_x += width;
    }

    return cursor_x - pos.x;
}

// Helper: Convert u32 number to string (max 10 digits)
fn u32_to_string(value: u32, out_str: ptr<function, array<u32, 32>>, start: u32) -> u32 {
    if (value == 0u) {
        (*out_str)[start] = 48u; // '0'
        return 1u;
    }

    var temp = value;
    var digit_count = 0u;
    var digits: array<u32, 10>;

    // Extract digits in reverse order
    while (temp > 0u && digit_count < 10u) {

        visual_grid[visual_idx] = vec4<f32>(blended_color, 1.0);
    }
}

// ============================================================================
// AGENT RENDER BUFFER MANAGEMENT
// ============================================================================

@compute @workgroup_size(16, 16)
fn clear_agent_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    // Skip clearing the inspector area only when an agent is selected (to preserve inspector UI)
    if (params.selected_agent_index != 0xFFFFFFFFu && x >= width - INSPECTOR_WIDTH) {
        return;
    }

    let agent_idx = y * params.visual_stride + x;
    // Fade the agent trail based on decay rate (1.0 = instant clear, 0.0 = no clear)
    let current_color = agent_grid[agent_idx];
    agent_grid[agent_idx] = current_color * (1.0 - params.agent_trail_decay);
}

// Inspector bar layout configuration
struct BarLayout {
    bars_y_start: u32,
    bars_y_end: u32,
    genome_x_start: u32,
    genome_x_end: u32,
    amino_x_start: u32,
    amino_x_end: u32,
    label_x_start: u32,
    label_x_end: u32,
    alpha_x_start: u32,
    alpha_x_end: u32,
    beta_x_start: u32,
    beta_x_end: u32,
}

// Calculate inspector bar positions based on anchor position
// anchor_x, anchor_y: top-left corner of the bar area
// available_height: height available for bars
fn calculate_bar_layout(anchor_x: u32, anchor_y: u32, available_height: u32) -> BarLayout {
    let bar_width = 40u;       // Width of genome and amino bars (narrower to fit in 300px)
    let label_width = 80u;     // Width of legend area (fits 3-letter amino codes)
    let signal_width = 25u;    // Width of signal bars
    let gap_large = 3u;        // Gap between major sections
    let gap_small = 1u;        // Gap between related elements (keep signals tight to amino)

    var bar_layout: BarLayout;

    // Vertical extent
    bar_layout.bars_y_start = anchor_y;
    bar_layout.bars_y_end = anchor_y + available_height;

    // Horizontal layout (left to right): genome, amino, alpha, beta, labels
    bar_layout.genome_x_start = anchor_x;
    bar_layout.genome_x_end = bar_layout.genome_x_start + bar_width;

    bar_layout.amino_x_start = bar_layout.genome_x_end + gap_large;
    bar_layout.amino_x_end = bar_layout.amino_x_start + bar_width;

    bar_layout.alpha_x_start = bar_layout.amino_x_end + gap_small;
    bar_layout.alpha_x_end = bar_layout.alpha_x_start + signal_width;

    bar_layout.beta_x_start = bar_layout.alpha_x_end + gap_small;
    bar_layout.beta_x_end = bar_layout.beta_x_start + signal_width;

    bar_layout.label_x_start = bar_layout.beta_x_end + gap_large;
    bar_layout.label_x_end = bar_layout.label_x_start + label_width;

    return bar_layout;
}

// Render inspector panel background (called after clear, before agent drawing)
@compute @workgroup_size(16, 16)
fn render_inspector(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (params.draw_enabled == 0u) { return; }

    // Only draw if we have a selected agent
    if (params.selected_agent_index == 0xFFFFFFFFu) {
        return;
    }

    let x = gid.x;
    let y = gid.y;

    let safe_width = max(params.window_width, 1.0);
    let safe_height = max(params.window_height, 1.0);
    let window_width = u32(safe_width);
    let window_height = u32(safe_height);
    let inspector_height = window_height;

    // Only render in the inspector region (rightmost 300 pixels)
    if (x >= INSPECTOR_WIDTH || y >= inspector_height) {
        return;
    }

    // Flip Y coordinate so y=0 is at bottom, increases upward
    let flipped_y = window_height - 1u - y;

    // Map to actual buffer position (rightmost area)
    let buffer_x = window_width - INSPECTOR_WIDTH + x;

    // Transparent background (no background drawn)
    var color = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    // No border drawn

    // Agent preview window: y between 0 and 300
    let preview_y_start = 0u;
    let preview_y_end = 300u;
    let preview_x_start = 10u;
    let preview_x_end = 290u;

    // No preview window background drawn

    // Gene bars: y between 300 and 800
    let bar_anchor_x = 5u;
    let bar_anchor_y = 300u;  // Start at y=300 to avoid overlapping preview window
    let bar_available_height = 500u;  // Height from 300 to 800
    let bars = calculate_bar_layout(bar_anchor_x, bar_anchor_y, bar_available_height);

    let in_genome_bar = x >= bars.genome_x_start && x < bars.genome_x_end && flipped_y >= bars.bars_y_start && flipped_y < bars.bars_y_end;
    let in_amino_bar = x >= bars.amino_x_start && x < bars.amino_x_end && flipped_y >= bars.bars_y_start && flipped_y < bars.bars_y_end;
    let in_label_area = x >= bars.label_x_start && x < bars.label_x_end && flipped_y >= bars.bars_y_start && flipped_y < bars.bars_y_end;
    let in_alpha_bar = x >= bars.alpha_x_start && x < bars.alpha_x_end && flipped_y >= bars.bars_y_start && flipped_y < bars.bars_y_end;
    let in_beta_bar = x >= bars.beta_x_start && x < bars.beta_x_end && flipped_y >= bars.bars_y_start && flipped_y < bars.bars_y_end;

    // Draw coordinate grid on inspector panel (100px squares) - HIDDEN
    // let grid_spacing = 100u;
    // let is_grid_line = (x % grid_spacing == 0u || flipped_y % grid_spacing == 0u);
    // if (is_grid_line && x < 300u && flipped_y < window_height) {
    //     // Draw grid lines in bright cyan
    //     color = vec4<f32>(0.0, 1.0, 1.0, 1.0);
    // }

    // Draw coordinate numbers at grid intersections (2x scale for bigger text) - HIDDEN
    // let grid_x = (x / grid_spacing) * grid_spacing;
    // let grid_y = (flipped_y / grid_spacing) * grid_spacing;

    // Draw X coordinate (above the intersection point) - 2x scale
    // if (x >= grid_x + 2u && x < grid_x + 36u && flipped_y >= grid_y + 2u && flipped_y < grid_y + 16u) {
    //     let px = (x - grid_x - 2u) / 2u;  // Scale down for 2x rendering
    //     let py = (flipped_y - grid_y - 2u) / 2u;
    //     if (draw_number(grid_x, grid_x, grid_y, px, py)) {
    //         color = vec4<f32>(1.0, 1.0, 0.0, 1.0);  // Yellow text
    //     }
    // }

    // Draw Y coordinate (below the X coordinate) - 2x scale - HIDDEN
    // if (x >= grid_x + 2u && x < grid_x + 36u && flipped_y >= grid_y + 18u && flipped_y < grid_y + 32u) {
    //     let px = (x - grid_x - 2u) / 2u;  // Scale down for 2x rendering
    //     let py = (flipped_y - grid_y - 18u) / 2u;
    //     if (draw_number(grid_y, grid_x, grid_y, px, py)) {
    //         color = vec4<f32>(0.0, 1.0, 0.0, 1.0);  // Green text
    //     }
    // }

    // Draw full genome bar (all nucleotides from first non-X triplet) - VERTICAL
    if (in_genome_bar) {
        let genome = selected_agent_buffer[0].genome;
        let body_count = selected_agent_buffer[0].body_count;
        let available_height = bars.bars_y_end - bars.bars_y_start;
        let genome_pixel_y = flipped_y - bars.bars_y_start;  // Pixel position in bar (0 to available_height)
        let pixels_per_base = 2u;  // 2 pixels per base
        let base_index = genome_pixel_y / pixels_per_base;  // Which base are we displaying

        // Always start from first non-X triplet (gene start)
        let gene_start = genome_find_first_coding_triplet(genome);

        // Find translation start to know where active region begins
        var translation_start = 0xFFFFFFFFu;
        if (params.require_start_codon == 1u) {
            translation_start = genome_find_start_codon(genome);
        } else {
            translation_start = gene_start;
        }

        // Find stop codon position by simulating translation
        var stop_codon_end = 0xFFFFFFFFu;
        if (gene_start != 0xFFFFFFFFu && translation_start != 0xFFFFFFFFu) {
            var pos_b = translation_start;
            // Skip start codon (AUG) - consumed for initiation, not translated
            if (params.require_start_codon == 1u) {
                pos_b = translation_start + 3u;
            }
            var part_count = 0u;
            let offset_bases = translation_start - gene_start;
            var cumulative_bases = offset_bases;

            for (var i = 0u; i < MAX_BODY_PARTS; i++) {
                // Use centralized translation function
                let step = translate_codon_step(genome, pos_b, params.ignore_stop_codons == 1u);

                if (!step.is_valid) {
                    if (step.is_stop && part_count >= body_count) {
                        stop_codon_end = gene_start + cumulative_bases + 3u;
                    }
                    break;
                }

                if (part_count >= body_count) {
                    break;
                }

                pos_b += step.bases_consumed;
                cumulative_bases += step.bases_consumed;
                part_count += 1u;
            }
        }

        // Find first run of 3+ consecutive 'X's to mark end of gene
        var first_x_position = GENOME_LENGTH;
        if (gene_start != 0xFFFFFFFFu) {
            for (var scan_pos = gene_start; scan_pos < GENOME_LENGTH - 2u; scan_pos++) {
                let ascii0 = genome_get_base_ascii(genome, scan_pos);
                let ascii1 = genome_get_base_ascii(genome, scan_pos + 1u);
                let ascii2 = genome_get_base_ascii(genome, scan_pos + 2u);
                // Found 3 consecutive X's
                if (ascii0 == 88u && ascii1 == 88u && ascii2 == 88u) {
                    first_x_position = scan_pos;
                    break;
                }
            }
        }

        if (gene_start != 0xFFFFFFFFu && base_index < GENOME_LENGTH - gene_start) {
            let actual_base_index = gene_start + base_index;
            // Only draw if before first 'X'
            if (actual_base_index < first_x_position && actual_base_index < GENOME_LENGTH) {
                let base_ascii = genome_get_base_ascii(genome, actual_base_index);

                var base_color = vec3<f32>(0.5, 0.5, 0.5);
                if (base_ascii == 65u) {  // 'A'
                    base_color = vec3<f32>(0.0, 1.0, 0.0);
                } else if (base_ascii == 85u) {  // 'U'
                    base_color = vec3<f32>(0.0, 0.5, 1.0);
                } else if (base_ascii == 71u) {  // 'G'
                    base_color = vec3<f32>(1.0, 1.0, 0.0);
                } else if (base_ascii == 67u) {  // 'C'
                    base_color = vec3<f32>(1.0, 0.0, 0.0);
                }

                // Dim inactive parts (before translation start or after stop codon) by 75%
                if (translation_start != 0xFFFFFFFFu && actual_base_index < translation_start) {
                    base_color *= 0.25;
                } else if (stop_codon_end != 0xFFFFFFFFu && actual_base_index >= stop_codon_end) {
                    base_color *= 0.25;
                }

                color = vec4<f32>(base_color, 1.0);
            }
        }
    }

    // Draw amino acid/organ bar (only translated parts up to body_count) - VERTICAL WITH LABELS
    if (in_amino_bar || in_label_area) {
        let genome = selected_agent_buffer[0].genome;
        let body_count = selected_agent_buffer[0].body_count;
        let available_height = bars.bars_y_end - bars.bars_y_start;
        let genome_pixel_y = flipped_y - bars.bars_y_start;  // Pixel position in bar
        let pixels_per_base = 2u;  // 2 pixels per base
        let base_index_in_bar = genome_pixel_y / pixels_per_base;

        // Gene always starts at first non-X triplet
        let gene_start = genome_find_first_coding_triplet(genome);

        // Translation starts at AUG (if required) or gene start
        var translation_start = 0xFFFFFFFFu;
        if (params.require_start_codon == 1u) {
            translation_start = genome_find_start_codon(genome);
        } else {
            translation_start = gene_start;
        }

        if (gene_start != 0xFFFFFFFFu && translation_start != 0xFFFFFFFFu) {
            // Calculate offset from gene start to translation start (in bases)
            let offset_bases = translation_start - gene_start;

            // Walk through genome following translation logic in base space
            var cumulative_bases = offset_bases;  // start accounting from gene_start
            var cumulative_pixels = cumulative_bases * pixels_per_base;
            var pos_b = translation_start;
            // Skip start codon (AUG) - consumed for initiation, not translated
            if (params.require_start_codon == 1u) {
                pos_b = translation_start + 3u;
                cumulative_bases += 3u;
                cumulative_pixels += 3u * pixels_per_base;
            }
            var part_count = 0u;

            for (var i = 0u; i < MAX_BODY_PARTS; i++) {
                // Use centralized translation function
                let step = translate_codon_step(genome, pos_b, params.ignore_stop_codons == 1u);

                // Handle invalid translation or stop
                if (!step.is_valid) {
                    // Draw stop codon if needed
                    if (step.is_stop && part_count >= body_count && in_amino_bar) {
                        let span_start_pixels = cumulative_pixels;
                        let span_end_pixels = cumulative_pixels + 3u * pixels_per_base;
                        if (genome_pixel_y >= span_start_pixels && genome_pixel_y < span_end_pixels) {
                            color = vec4<f32>(0.0, 0.0, 0.0, 1.0);  // black
                        }
                    }
                    break;
                }

                if (part_count >= body_count) {
                    break;
                }

                // Get decoded part type and organ status
                let base_type = get_base_part_type(step.part_type);
                let is_organ = (base_type >= 20u);

                // Map genome_pixel_y (in pixels) into this part's span
                let span_start_pixels = cumulative_pixels;
                let span_height = step.bases_consumed * pixels_per_base;
                let span_end_pixels = span_start_pixels + span_height;
                let span_mid_pixels = (span_start_pixels + span_end_pixels) / 2u;

                if (genome_pixel_y >= span_start_pixels && genome_pixel_y < span_end_pixels) {
                    if (in_amino_bar) {
                        // Regular amino acid or organ color bar
                        let props = get_amino_acid_properties(base_type);
                        var base_color = props.color;

                        // For clock organs, oscillate color based on clock_signal in _pad.x
                        if (is_organ && base_type == 31u && part_count < body_count) {
                            // Read clock_signal from this part's _pad.x (range -1 to +1)
                            let clock_signal = selected_agent_buffer[0].body[part_count]._pad.y;
                            // Modulate brightness: 0.5 to 1.5 range based on signal
                            let brightness = 1.0 + clock_signal * 0.5;
                            base_color = base_color * brightness;
                        }

                        // For magnitude sensors (38-41), use brighter color tones to differentiate from directional sensors
                        let is_magnitude_sensor = base_type == 38u || base_type == 39u || base_type == 40u || base_type == 41u;
                        if (is_magnitude_sensor) {
                            // Brighten the color by 30%
                            base_color = base_color * 1.3;
                        }

                        // For organs that need amplification (propeller, displacer, mouth, vampire mouth, agent sensors), calculate and apply
                        let needs_amplification = props.is_propeller || props.is_displacer || props.is_mouth || base_type == 33u || base_type == 34u || base_type == 35u;
                        if (part_count < body_count && needs_amplification) {
                            let part_pos = selected_agent_buffer[0].body[part_count].pos;

                            // Calculate amplification from nearby enablers
                            var amp = 0.0;
                            for (var e = 0u; e < body_count; e++) {
                                let e_base_type = get_base_part_type(selected_agent_buffer[0].body[e].part_type);
                                let e_props = get_amino_acid_properties(e_base_type);
                                if (e_props.is_inhibitor) { // enabler flag
                                    let enabler_pos = selected_agent_buffer[0].body[e].pos;
                                    let d = length(part_pos - enabler_pos);
                                    if (d < 20.0) {
                                        amp += max(0.0, 1.0 - d / 20.0);
                                    }
                                }
                            }
                            let amplification = min(amp, 1.0);

                            // Multiply color by amplification (brighter = more amplification)
                            base_color = base_color * (1.0 + amplification);
                        }

                        color = vec4<f32>(base_color, 1.0);
                    } else if (in_label_area) {
                        // Per-pixel vector mask: stable, no race with draw_thick_line writes
                        let local_pixel_y = genome_pixel_y - span_start_pixels;

                        // Only consider rows within text height
                        let text_height = 8.0;   // Slightly smaller legend text
                        let text_rows = u32(ceil(text_height));
                        if (local_pixel_y < text_rows) {
                            let name = get_part_name(base_type);

                            let local_x = x - bars.label_x_start;
                            let text_start_x = 20u;
                            var cursor = text_start_x;
                            let char_spacing = 1u;
                            var hit = false;

                            // Unrolled to avoid dynamic array indexing
                            let c0 = name.chars[0];
                            if (name.len > 0u) {
                                let cw0 = u32(ceil(get_char_width(c0) * text_height));
                                if (local_x >= cursor && local_x < cursor + cw0) {
                                    let px = local_x - cursor;
                                    let py = local_pixel_y;
                                    if (char_vector_mask(vec2<u32>(px, py), c0, text_height)) { hit = true; }
                                }
                                cursor += cw0 + char_spacing;
                            }

                            let c1 = name.chars[1];
                            if (name.len > 1u) {
                                let cw1 = u32(ceil(get_char_width(c1) * text_height));
                                if (local_x >= cursor && local_x < cursor + cw1) {
                                    let px = local_x - cursor;
                                    let py = local_pixel_y;
                                    if (char_vector_mask(vec2<u32>(px, py), c1, text_height)) { hit = true; }
                                }
                                cursor += cw1 + char_spacing;
                            }

                            let c2 = name.chars[2];
                            if (name.len > 2u) {
                                let cw2 = u32(ceil(get_char_width(c2) * text_height));
                                if (local_x >= cursor && local_x < cursor + cw2) {
                                    let px = local_x - cursor;
                                    let py = local_pixel_y;
                                    if (char_vector_mask(vec2<u32>(px, py), c2, text_height)) { hit = true; }
                                }
                                cursor += cw2 + char_spacing;
                            }

                            let c3 = name.chars[3];
                            if (name.len > 3u) {
                                let cw3 = u32(ceil(get_char_width(c3) * text_height));
                                if (local_x >= cursor && local_x < cursor + cw3) {
                                    let px = local_x - cursor;
                                    let py = local_pixel_y;
                                    if (char_vector_mask(vec2<u32>(px, py), c3, text_height)) { hit = true; }
                                }
                                cursor += cw3 + char_spacing;
                            }

                            let c4 = name.chars[4];
                            if (name.len > 4u) {
                                let cw4 = u32(ceil(get_char_width(c4) * text_height));
                                if (local_x >= cursor && local_x < cursor + cw4) {
                                    let px = local_x - cursor;
                                    let py = local_pixel_y;
                                    if (char_vector_mask(vec2<u32>(px, py), c4, text_height)) { hit = true; }
                                }
                                cursor += cw4 + char_spacing;
                            }

                            let c5 = name.chars[5];
                            if (name.len > 5u) {
                                let cw5 = u32(ceil(get_char_width(c5) * text_height));
                                if (local_x >= cursor && local_x < cursor + cw5) {
                                    let px = local_x - cursor;
                                    let py = local_pixel_y;
                                    if (char_vector_mask(vec2<u32>(px, py), c5, text_height)) { hit = true; }
                                }
                                cursor += cw5 + char_spacing;
                            }

                            if (hit) {
                                color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
                            }
                        }
                    }
                }

                pos_b += step.bases_consumed;
                cumulative_bases += step.bases_consumed;
                cumulative_pixels += span_height;
                part_count += 1u;

                // Stop if we've rendered beyond the visible area
                if (cumulative_pixels > available_height) { break; }
            }
        }
    }

    // Draw signal bars (alpha and beta signals for each body part) - VERTICAL
    if (in_alpha_bar || in_beta_bar) {
        let genome = selected_agent_buffer[0].genome;
        let body_count = selected_agent_buffer[0].body_count;
        let available_height = bars.bars_y_end - bars.bars_y_start;
        let genome_pixel_y = flipped_y - bars.bars_y_start;  // Pixel position in bar
        let pixels_per_base = 2u;  // 2 pixels per base (must match amino bar)

        // Gene always starts at first non-X triplet
        let gene_start = genome_find_first_coding_triplet(genome);

        // Translation starts at AUG (if required) or gene start
        var translation_start = 0xFFFFFFFFu;
        if (params.require_start_codon == 1u) {
            translation_start = genome_find_start_codon(genome);
        } else {
            translation_start = gene_start;
        }

        if (gene_start != 0xFFFFFFFFu && translation_start != 0xFFFFFFFFu) {
            // Calculate offset from gene start to translation start (in bases)
            let offset_bases = translation_start - gene_start;

            // Walk through genome following translation logic in pixel space
            var cumulative_bases = offset_bases;
            var cumulative_pixels = cumulative_bases * pixels_per_base;
            var pos_b = translation_start;
            // Skip start codon (AUG) - consumed for initiation, not translated
            if (params.require_start_codon == 1u) {
                pos_b = translation_start + 3u;
                cumulative_bases += 3u;
                cumulative_pixels += 3u * pixels_per_base;
            }
            var part_count = 0u;

            for (var i = 0u; i < MAX_BODY_PARTS; i++) {
                // Use centralized translation function
                let step = translate_codon_step(genome, pos_b, params.ignore_stop_codons == 1u);

                if (!step.is_valid) {
                    break;
                }

                if (part_count >= body_count) {
                    break;
                }

                let base_type = get_base_part_type(step.part_type);

                // Map genome_pixel_y (in pixels) into this part's span
                let span_start_pixels = cumulative_pixels;
                let span_height = step.bases_consumed * pixels_per_base;
                let span_end_pixels = span_start_pixels + span_height;

                if (genome_pixel_y >= span_start_pixels && genome_pixel_y < span_end_pixels) {
                    // Get signals from actual body part
                    if (part_count < body_count) {
                        let part = selected_agent_buffer[0].body[part_count];
                        let a = part.alpha_signal;
                        let b = part.beta_signal;

                        // Debug mode color scheme: r=+beta, g=+alpha, blue=-alpha OR -beta
                        let r = max(b, 0.0);
                        let g = max(a, 0.0);
                        let bl = max(max(-a, 0.0), max(-b, 0.0));

                        color = vec4<f32>(r, g, bl, 1.0);
                    }
                }

                pos_b += step.bases_consumed;
                cumulative_bases += step.bases_consumed;
                cumulative_pixels += span_height;
                part_count += 1u;

                // Stop after drawing this part if it was before a stop codon
                if (step.is_stop) {
                    break;
                }

                // Stop if we've rendered beyond the visible area
                if (cumulative_pixels > available_height) { break; }
            }
        }
    }

    // Write to agent_grid using visual_stride
    let idx = y * params.visual_stride + buffer_x;
    agent_grid[idx] = color;
}

// Draw inspector agent (called after render_inspector, draws agent closeup in preview)
@compute @workgroup_size(16, 16)
fn draw_inspector_agent(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (params.draw_enabled == 0u) { return; }
    if (params.selected_agent_index == 0xFFFFFFFFu) { return; }

    // This shader doesn't need to do pixel-level work anymore
    // Just call render_body_part_ctx for each part with inspector context
    // Only run once (use thread 0,0)
    if (gid.x != 0u || gid.y != 0u) { return; }

    let body_count = min(selected_agent_buffer[0].body_count, MAX_BODY_PARTS);
    if (body_count == 0u) { return; }

    // Preview window setup (now at bottom with same coordinates as render_inspector)
    let preview_size = 280u;
    let preview_x_start = 10u;
    let safe_width = max(params.window_width, 1.0);
    let safe_height = max(params.window_height, 1.0);
    let window_width = u32(safe_width);
    let window_height = u32(safe_height);
    let preview_y_start = window_height - preview_size - 10u;  // 10px from bottom

    // Calculate auto-scale to fit agent
    var max_extent = 0.0;
    for (var i = 0u; i < MAX_BODY_PARTS; i++) {
        if (i < body_count) {
            let part = selected_agent_buffer[0].body[i];
            let dist = length(part.pos);
            max_extent = max(max_extent, dist + part.size);
        }
    }
    let available_space = f32(preview_size) * 0.45; // Half width, 90% of that
    let auto_scale = select(available_space / max_extent, 1.0, max_extent > 1.0);
    let scale_factor = auto_scale * params.inspector_zoom;  // Apply user zoom

    // Preview center
    let preview_center_x = f32(preview_x_start + preview_size / 2u);
    let preview_center_y = f32(preview_y_start + preview_size / 2u);

    // Calculate buffer offset (rightmost area)
    let buffer_offset_x = f32(window_width - INSPECTOR_WIDTH);

    // Create inspector context
    let ctx = InspectorContext(
        vec2<f32>(0.0, 0.0),  // use_inspector_coords (x >= 0 enables inspector mode)
        vec2<f32>(preview_center_x, preview_center_y),  // center of preview
        scale_factor,  // scale
        vec2<f32>(buffer_offset_x, 0.0)  // offset to actual buffer position
    );

    // Calculate agent color
    var color_sum = 0.0;
    for (var i = 0u; i < MAX_BODY_PARTS; i++) {
        if (i < body_count) {
            let base_type = get_base_part_type(selected_agent_buffer[0].body[i].part_type);
            let part_props = get_amino_acid_properties(base_type);
            color_sum += part_props.beta_damage;
        }
    }
    let agent_color = vec3<f32>(
        sin(color_sum * 3.0) * 0.5 + 0.5,
        sin(color_sum * 5.0) * 0.5 + 0.5,
        sin(color_sum * 7.0) * 0.5 + 0.5
    );

    // Get morphology origin (where chain starts in local space)
    let morphology_origin = selected_agent_buffer[0].morphology_origin;

    // Render all body parts using the unified render function
    // Note: agent is unrotated (rotation=0) in selected_agent_buffer
    let in_debug_mode = params.debug_mode != 0u;
    for (var i = 0u; i < MAX_BODY_PARTS; i++) {
        if (i >= body_count) { break; }

        let part = selected_agent_buffer[0].body[i];

        // Use special agent_id value 0xFFFFFFFFu to indicate selected_agent_buffer access
        render_body_part_ctx(
            part,
            i,
            0xFFFFFFFFu,  // special value to use selected_agent_buffer instead of agents_out
            vec2<f32>(0.0, 0.0),  // agent_position (will be offset by ctx)
            0.0,  // agent_rotation (already unrotated)
            selected_agent_buffer[0].energy,
            agent_color,
            body_count,
            morphology_origin,
            1.0,  // amplification
            in_debug_mode,
            ctx
        );
    }

    // ============================================================================
    // TEXT LABELS - Display agent information with scalable vector font
    // ============================================================================
    let text_color = vec4<f32>(1.0, 1.0, 1.0, 1.0);

    // Direct pixel inspector context (origin at inspector top-left).
    // Use scale of 0.33 to make lines thinner (1px * 0.33 G?? 0.33px which rounds to 1px)
    let text_ctx = InspectorContext(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
        0.33,
        vec2<f32>(buffer_offset_x, 0.0)
    );

    let text_height = 14.0;  // Will be scaled down by ctx.scale
    let char_width = 9.0;    // Approximate character width at this size (will be scaled)
    let line_height = 18.0;  // Spacing between lines (will be scaled)
    let max_width = 280.0;   // 300 - 20px padding

    // Simple top-down layout with explicit Y positions
    // Bars are at screen_height - 80, so text should start below that
    // Offset from bottom: bars at -80px, so text starts at -80px + 80px (bar height) = top
    // Actually place text starting from a safe position below bars
    let energy_y = f32(window_height) - 280.0 - 10.0 + 280.0 + 90.0;  // Below bars (bars ~80px tall)
    let organs_y = energy_y + 30.0;  // 30px below energy line

    // ENERGY label
    var energy_text: array<u32, 32>;
    energy_text[0] = 69u;  // 'E'
    energy_text[1] = 78u;  // 'N'
    energy_text[2] = 69u;  // 'E'
    energy_text[3] = 82u;  // 'R'
    energy_text[4] = 71u;  // 'G'
    energy_text[5] = 89u;  // 'Y'
    energy_text[6] = 58u;  // ':'
    energy_text[7] = 32u;  // ' '
    let energy_prefix_len = 8u;
    let energy_value_len = f32_to_string(selected_agent_buffer[0].energy, &energy_text, energy_prefix_len);
    let energy_total_len = energy_prefix_len + energy_value_len;
    let energy_pos = vec2<f32>(10.0, energy_y);
    draw_string_vector(energy_pos, &energy_text, energy_total_len, text_height, text_color, text_ctx);

    // Organ letters wrapped across multiple lines
    var current_y = organs_y;
    var current_x = 10.0;

    for (var i = 0u; i < body_count; i++) {
        let part = selected_agent_buffer[0].body[i];
        let base_type = get_base_part_type(part.part_type);

        // Get organ properties for color
        let part_props = get_amino_acid_properties(base_type);
        let letter_color = vec4<f32>(part_props.color, 1.0);

        // Convert organ type to letter
        var letter = 63u; // '?' default
        if (base_type < 20u) {
            // Amino acids A-T (0-19)
            letter = 65u + base_type;  // 'A' + offset
        } else {
            // Organs - assign letters
            switch (base_type) {
                case 20u: { letter = 77u; }  // 'M' = Mouth
                case 21u: { letter = 80u; }  // 'P' = Propeller
                case 22u: { letter = 65u; }  // 'A' = Alpha Sensor
                case 23u: { letter = 66u; }  // 'B' = Beta Sensor
                case 24u: { letter = 69u; }  // 'E' = Energy Sensor
                case 25u: { letter = 68u; }  // 'D' = Displacer
                case 26u: { letter = 78u; }  // 'N' = eNabler
                case 28u: { letter = 83u; }  // 'S' = Storage
                case 29u: { letter = 82u; }  // 'R' = poison Resistance
                case 30u: { letter = 67u; }  // 'C' = Chiral Flipper
                case 31u: { letter = 88u; }  // 'X' = Clock
                case 32u: { letter = 47u; }  // '/' = Slope Sensor
                case 33u: { letter = 86u; }  // 'V' = Vampire mouth
                case 34u: { letter = 60u; }  // '<' = Agent alpha sensor
                case 35u: { letter = 62u; }  // '>' = Agent beta sensor
                default: { letter = 63u; }   // '?' = unknown
            }
        }

        // Check if we need to wrap to next line
        if (current_x + char_width > max_width) {
            current_x = 10.0;
            current_y += line_height;  // Move down
        }

        // Draw the character with organ color
        let char_pos = vec2<f32>(current_x, current_y);
        current_x += draw_char_vector(char_pos, letter, text_height, letter_color, text_ctx);
    }
}

// ============================================================================
// AGENT RENDERING KERNEL
// ============================================================================

// Dedicated kernel to render all agents to agent_grid
// This runs after process_agents and before composite_agents
@compute @workgroup_size(256)
fn render_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (params.draw_enabled == 0u) { return; }

    let agent_id = gid.x;
    if (agent_id >= params.max_agents) {
        return;
    }

    // Only render alive agents
    if (agents_out[agent_id].alive == 0u) {
        return;
    }

    // Skip if no body parts
    let body_count = agents_out[agent_id].body_count;
    if (body_count == 0u) {
        return;
    }

    // Calculate camera bounds with aspect ratio
    let aspect_ratio = params.window_width / params.window_height;
    let camera_half_height = params.grid_size / (2.0 * params.camera_zoom);
    let camera_half_width = camera_half_height * aspect_ratio;
    let camera_center = vec2<f32>(params.camera_pan_x, params.camera_pan_y);
    let camera_min = camera_center - vec2<f32>(camera_half_width, camera_half_height);
    let camera_max = camera_center + vec2<f32>(camera_half_width, camera_half_height);

    // Frustum culling - check if agent is visible
    let margin = 20.0; // Maximum body extent
    let center = agents_out[agent_id].position;
    if (center.x + margin < camera_min.x || center.x - margin > camera_max.x ||
        center.y + margin < camera_min.y || center.y - margin > camera_max.y) {
        return; // Not visible, skip rendering
    }

    // Agent is visible - render it
    let in_debug_mode = params.debug_mode != 0u;

    // Calculate agent color
    var color_sum = 0.0;
    for (var i = 0u; i < MAX_BODY_PARTS; i++) {
        if (i < body_count) {
            let base_type = get_base_part_type(agents_out[agent_id].body[i].part_type);
            let part_props = get_amino_acid_properties(base_type);
            color_sum += part_props.beta_damage;
        }
    }
    let agent_color = vec3<f32>(
        sin(color_sum * 3.0) * 0.5 + 0.5,
        sin(color_sum * 5.0) * 0.5 + 0.5,
        sin(color_sum * 7.0) * 0.5 + 0.5
    );

    // Get the morphology origin
    let morphology_origin = agents_out[agent_id].morphology_origin;

    // Draw all body parts using unified rendering function
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let part = agents_out[agent_id].body[i];
        // Calculate jet amplification for this part
        var jet_amplification = 1.0;
        if (get_base_part_type(part.part_type) == 23u) { // Jet organ
            let alpha_signal = part.alpha_signal;
            let beta_signal = part.beta_signal;
            jet_amplification = sqrt(alpha_signal * alpha_signal + beta_signal * beta_signal) * 3.0;
        }

        render_body_part(
            part,
            i,
            agent_id,
            center,
            agents_out[agent_id].rotation,
            agents_out[agent_id].energy,
            agent_color,
            body_count,
            morphology_origin,
            jet_amplification,
            in_debug_mode
        );
    }

    if (in_debug_mode) {
        // Draw center cross marker only in debug mode
        let cross_size = 3.0;
        draw_thick_line(
            center + vec2<f32>(-cross_size, 0.0),
            center + vec2<f32>(cross_size, 0.0),
            0.5,
            vec4<f32>(1.0, 1.0, 1.0, 1.0),
        );
        draw_thick_line(
            center + vec2<f32>(0.0, -cross_size),
            center + vec2<f32>(0.0, cross_size),
            0.5,
            vec4<f32>(1.0, 1.0, 1.0, 1.0),
        );

        // Check if agent has vampire mouth (organ 33) and mark with red circle
        var has_vampire_mouth = false;
        for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
            let base_type = get_base_part_type(agents_out[agent_id].body[i].part_type);
            if (base_type == 33u) {
                has_vampire_mouth = true;
                break;
            }
        }

        if (has_vampire_mouth) {
            // Draw red circle around agent with vampire mouth
            let circle_radius = 15.0;
            let segments = 24u;
            var prev = center + vec2<f32>(circle_radius, 0.0);
            for (var s = 1u; s <= segments; s++) {
                let t = f32(s) / f32(segments);
                let ang = t * 6.28318530718;
                let p = center + vec2<f32>(cos(ang) * circle_radius, sin(ang) * circle_radius);
                draw_thick_line(prev, p, 2.0, vec4<f32>(1.0, 0.0, 0.0, 0.9));
                prev = p;
            }
        }
    }

    // Debug: count visible agents
    atomicAdd(&debug_counter, 1u);

    // Draw selection circle if this agent is selected
    if (agents_out[agent_id].is_selected == 1u) {
        draw_selection_circle(center, agent_id, body_count);
    }
}

