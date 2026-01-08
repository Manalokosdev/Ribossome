// microswim_agents.wgsl
// Faster hybrid microswimming - tuned for good perceived speed at 60 FPS
// Key changes for speed:
// - Higher base thrust scaling (MSP_COUPLING boosted)
// - Lower base drag (faster from same deformation)
// - Stronger vortex (but still gated)
// - Reduced quadratic drag and damping (less resistance)
// - Tighter but higher velocity caps
// Stability preserved: quadratic drag + strict gating + caps

@compute @workgroup_size(256)
fn microswim_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent_id = gid.x;
    if (agent_id >= params.agent_count || !ms_enabled() || agents_out[agent_id].alive == 0u) {
        return;
    }

    let body_count_out = min(agents_out[agent_id].body_count, MAX_BODY_PARTS);
    let body_count_in = min(agents_in[agent_id].body_count, MAX_BODY_PARTS);
    let first_build = (body_count_in == 0u);

    // Skip if anchored last frame
    var anchor_active_prev = false;
    for (var i = 0u; i < body_count_in; i++) {
        let part_in = agents_in[agent_id].body[i];
        let base_type_in = get_base_part_type(part_in.part_type);
        if (base_type_in == 42u && part_in._pad.y > 0.5) {
            anchor_active_prev = true;
            break;
        }
    }
    if (anchor_active_prev) {
        return;
    }

    let morph_swim_strength = select(
        0.0,
        ms_f32(MSP_COUPLING),
        (!first_build)
    );

    if (morph_swim_strength <= 0.0 || body_count_out <= 1u) {
        return;
    }

    let agent_rot = agents_out[agent_id].rotation;
    let dt_safe = max(params.dt, 1e-3);

    var thrust_world = vec2<f32>(0.0);
    var total_weight = 0.0;
    var total_deformation_sq = 0.0;

    // Asymmetry torque variables
    var torque_asymmetry = 0.0;
    var inertia_estimate = 0.0;

    // Vortex bonus (gated)
    var vortex_bonus = 0.0;

    // Hybrid Re blend - kept conservative thresholds
    let mass = agents_out[agent_id].total_mass;

    // Rough surface-area estimate from part geometry (length * thickness).
    // Used to scale response by area/mass (mass-per-area).
    var total_area = 0.0;
    for (var i = 0u; i < body_count_out; i++) {
        let part = agents_out[agent_id].body[i];
        let base_type = get_base_part_type(part.part_type);
        total_area += get_part_area_estimate(part.part_type);

        // Open vampire mouths (33) and spikes (46) add extra effective area.
        if (base_type == 33u || base_type == 46u) {
            let open = clamp(part._pad.y, 0.0, 1.0);
            var r = 30.0;
            if (i + 1u < body_count_out) {
                let next_base = get_base_part_type(agents_out[agent_id].body[i + 1u].part_type);
                r = param1_to_defense_radius(get_amino_acid_properties(next_base).parameter1);
            }
            total_area += open * (r * r) * 0.02;
        }
    }
    total_area = max(total_area, 1e-3);
    let area_response = clamp(total_area / max(mass, 1e-6), 0.25, 4.0);

    let mass_threshold_low = 1.0;
    let mass_threshold_high = 5.0;
    var re_blend = clamp((mass - mass_threshold_low) / (mass_threshold_high - mass_threshold_low), 0.0, 1.0);

    // Lower base drag for overall faster motion from deformation
    let c_par_base  = ms_f32(MSP_BASE_DRAG) * 0.7;  // Reduced 30% - agents slip more = faster
    let c_perp_base = c_par_base * ms_f32(MSP_ANISOTROPY);

    // Less aggressive drag reduction in high-Re (keep some viscosity)
    let c_par  = mix(c_par_base, c_par_base * 0.6, re_blend);
    let c_perp = mix(c_perp_base, c_perp_base * 0.8, re_blend);

    var current_vel = agents_out[agent_id].velocity;
    let speed = length(current_vel);
    var forward_dir = vec2<f32>(cos(agent_rot), sin(agent_rot));
    if (speed > 0.01) {
        forward_dir = normalize(current_vel);
    }
    let lateral_dir = vec2<f32>(-forward_dir.y, forward_dir.x);

    for (var j = 1u; j < body_count_out; j++) {
        let seg_start_prev = agents_in[agent_id].body[j - 1u].pos;
        let seg_end_prev   = agents_in[agent_id].body[j].pos;
        let seg_start_new  = agents_out[agent_id].body[j - 1u].pos;
        let seg_end_new    = agents_out[agent_id].body[j].pos;

        let delta_start = seg_start_new - seg_start_prev;
        let delta_end   = seg_end_new   - seg_end_prev;

        total_deformation_sq += dot(delta_start, delta_start) + dot(delta_end, delta_end);

        var v_seg_local = (delta_start + delta_end) * 0.5;
        var v_seg_world = apply_agent_rotation(v_seg_local, agent_rot);

        // Filters
        let raw_displacement_len = length(v_seg_local);
        if (raw_displacement_len < ms_f32(MSP_MIN_SEG_DISPLACEMENT)) {
            continue;
        }
        if (raw_displacement_len > MORPHOLOGY_MAX_WORLD_DELTA * 0.5) {
            v_seg_world *= (MORPHOLOGY_MAX_WORLD_DELTA * 0.5) / max(raw_displacement_len, 1e-6);
        }

        v_seg_world /= dt_safe;

        let vlen = length(v_seg_world);
        if (vlen > MORPHOLOGY_MAX_WORLD_VEL * 0.5) {
            v_seg_world *= (MORPHOLOGY_MAX_WORLD_VEL * 0.5) / max(vlen, 1e-6);
        }

        let seg_vec_new = seg_end_new - seg_start_new;
        let seg_len = length(seg_vec_new);
        if (seg_len < 0.01) {
            continue;
        }

        let seg_vec_prev = seg_end_prev - seg_start_prev;
        let prev_len = length(seg_vec_prev);
        if (prev_len > 0.01) {
            let len_ratio = seg_len / prev_len;
            if (len_ratio < ms_f32(MSP_MIN_LENGTH_RATIO) || len_ratio > ms_f32(MSP_MAX_LENGTH_RATIO)) {
                continue;
            }
        }

        let tangent_local = seg_vec_new / seg_len;
        let tangent_world = normalize(apply_agent_rotation(tangent_local, agent_rot));

        let v_parallel = dot(v_seg_world, tangent_world) * tangent_world;

        // Low-Re thrust
        thrust_world += (-c_par * v_parallel) * seg_len;
        total_weight += seg_len;

        // Vortex bonus: slightly higher strength + lower threshold for more frequent bursts
        let deform_strength = length(v_seg_world);
        if (deform_strength > 1.0) {  // Lowered threshold
            vortex_bonus += deform_strength * seg_len * 0.08;  // Increased strength
        }

        // Asymmetry torque
        let seg_mid_local = (seg_start_new + seg_end_new) * 0.5;
        let seg_mid_world = apply_agent_rotation(seg_mid_local, agent_rot);

        let lateral_offset = dot(seg_mid_world, lateral_dir);
        let perp_drag = c_perp * speed * seg_len;

        torque_asymmetry += lateral_offset * perp_drag;
        inertia_estimate += lateral_offset * lateral_offset * seg_len;
    }

    if (total_deformation_sq < ms_f32(MSP_MIN_TOTAL_DEFORMATION_SQ) || total_weight <= 1e-6) {
        return;
    }

    thrust_world /= total_weight;

    // Higher overall thrust scaling via morph_swim_strength (assume MSP_COUPLING = 2.0-3.0 in params)
    var final_thrust = thrust_world * morph_swim_strength * area_response;

    // Stronger but gated vortex
    if (total_deformation_sq > ms_f32(MSP_MIN_TOTAL_DEFORMATION_SQ) * 1.8) {
        let normalized_vortex = (vortex_bonus / total_weight) * re_blend * 1.5;  // Boosted
        let max_vortex = sqrt(total_deformation_sq) * 0.8;  // Higher cap
        final_thrust += forward_dir * min(normalized_vortex, max_vortex) * area_response;
    }

    // Higher thrust cap for faster peak speeds
    let tl = length(final_thrust);
    if (tl > ms_f32(MSP_MAX_FRAME_VEL) * 1.2) {  // Raised cap
        final_thrust *= (ms_f32(MSP_MAX_FRAME_VEL) * 1.2) / max(tl, 1e-6);
    }

    var agent_vel = agents_out[agent_id].velocity + final_thrust;

    // Reduced quadratic drag for higher sustained speeds
    let quad_drag_coeff = 0.002 * re_blend;  // Halved - less speed limiting
    let quad_drag = speed * speed * quad_drag_coeff;
    if (speed > 0.01) {
        agent_vel -= normalize(agent_vel) * quad_drag;
    }

    // Lighter damping for snappier motion
    let base_damping = 0.92;  // Was 0.95 - 8% loss vs 5%
    let damping = mix(base_damping, 0.98, re_blend);
    agent_vel *= damping;

    // Higher global velocity cap
    let v_len = length(agent_vel);
    if (v_len > VEL_MAX * 1.1) {  // Raised from 0.85
        agent_vel *= (VEL_MAX * 1.1) / max(v_len, 1e-6);
    }
    agents_out[agent_id].velocity = agent_vel;

    // Asymmetry torque (stable across regimes)
    if (inertia_estimate > 1e-6 && speed > 0.1) {
        let normalized_torque = torque_asymmetry / inertia_estimate;
        let torque_scale = ms_f32(MSP_TORQUE_STRENGTH) * morph_swim_strength * dt_safe * area_response;

        var delta_rot = normalized_torque * torque_scale * min(speed / 3.0, 1.0);
        delta_rot = clamp(delta_rot, -ANGVEL_MAX, ANGVEL_MAX);

        agents_out[agent_id].rotation += delta_rot;
    }
}
