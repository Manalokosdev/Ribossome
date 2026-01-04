// microswim_agents.wgsl
// Stable hybrid low-Re / high-Re microswimming
// Fixes: Quadratic drag for terminal velocity, strict vortex gating, conservative tuning
// Small agents: Pure low-Re viscous wiggling
// Large agents: Controlled inertial coasting + deformation-triggered vortex bursts

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
        max(max(params.prop_wash_strength, 0.0), max(params.prop_wash_strength_fluid, 0.0)) * ms_f32(MSP_COUPLING),
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

    // Vortex bonus (now strictly gated on significant deformation)
    var vortex_bonus = 0.0;

    // Hybrid Re blend based on mass
    // Tuned for typical agent masses: 0.02*parts (amino) to 0.15*parts (enabler/organ mix)
    let mass = agents_out[agent_id].total_mass;
    let mass_threshold_low = .10;    // Higher: keep agents in low-Re longer (~50 amino or ~6-7 enabler-heavy)
    let mass_threshold_high = 2.0;   // Higher: delay full high-Re (~150 amino or ~20 enabler-heavy)
    var re_blend = clamp((mass - mass_threshold_low) / (mass_threshold_high - mass_threshold_low), 0.0, 1.0);

    let c_par_base  = ms_f32(MSP_BASE_DRAG);
    let c_perp_base = c_par_base * ms_f32(MSP_ANISOTROPY);

    // Minimal drag reduction (75% min parallel, 90% min perpendicular) - keep agents slow
    let c_par  = mix(c_par_base, c_par_base * 0.75, re_blend);
    let c_perp = mix(c_perp_base, c_perp_base * 0.9, re_blend);

    // Current velocity for direction and high-Re effects
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

        // Low-Re thrust (always active)
        let thrust_parallel = (-c_par * v_parallel) * seg_len;
        thrust_world += thrust_parallel;
        total_weight += seg_len;

        // Vortex bonus: only on significant per-segment deformation (prevents free acceleration)
        let deform_strength = length(v_seg_world);
        if (deform_strength > 1.5) {  // Higher threshold: must actively bend
            vortex_bonus += deform_strength * seg_len * 0.04;  // Reduced strength (was 0.05)
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

    var final_thrust = thrust_world * morph_swim_strength;

    // Vortex bonus only when body significantly deforms (strict gate)
    // CRITICAL: Normalize vortex bonus like low-Re thrust - prevents unbounded accumulation
    if (total_deformation_sq > ms_f32(MSP_MIN_TOTAL_DEFORMATION_SQ) * 2.0 && total_weight > 1e-6) {
        let normalized_vortex = (vortex_bonus / total_weight) * re_blend;
        // Further limit: vortex can't exceed the actual deformation magnitude
        let max_vortex = sqrt(total_deformation_sq) * 0.5;
        final_thrust += forward_dir * min(normalized_vortex, max_vortex);
    }

    let tl = length(final_thrust);
    if (tl > ms_f32(MSP_MAX_FRAME_VEL) * 0.6) {  // Tighter cap (was 0.7)
        final_thrust *= (ms_f32(MSP_MAX_FRAME_VEL) * 0.6) / max(tl, 1e-6);
    }

    var agent_vel = agents_out[agent_id].velocity + final_thrust;

    // Quadratic drag for high-Re terminal velocity (key stability fix)
    // This creates natural speed limit proportional to sqrt(thrust/drag_coeff)
    let quad_drag_coeff = 0.004 * re_blend;  // Doubled again - very strong speed limiting
    let quad_drag = speed * speed * quad_drag_coeff;
    if (speed > 0.01) {
        agent_vel -= normalize(agent_vel) * quad_drag;
    }

    // Very strong damping - aggressive velocity decay
    let base_damping = 0.95;  // Much higher - only 5% velocity loss per frame minimum
    let damping = mix(base_damping, 0.96, re_blend);  // Even high-Re heavily damped
    agent_vel *= damping;

    let v_len = length(agent_vel);
    if (v_len > VEL_MAX * 0.85) {  // Slightly tighter (was 0.9)
        agent_vel *= (VEL_MAX * 0.85) / max(v_len, 1e-6);
    }
    agents_out[agent_id].velocity = agent_vel;

    // Asymmetry torque (stable across regimes)
    if (inertia_estimate > 1e-6 && speed > 0.1) {
        let normalized_torque = torque_asymmetry / inertia_estimate;
        let torque_scale = ms_f32(MSP_TORQUE_STRENGTH) * morph_swim_strength * dt_safe;

        var delta_rot = normalized_torque * torque_scale * min(speed / 3.0, 1.0);
        delta_rot = clamp(delta_rot, -ANGVEL_MAX, ANGVEL_MAX);

        agents_out[agent_id].rotation += delta_rot;
    }
}
