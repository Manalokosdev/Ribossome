// microswim_agents.wgsl
// Microswimming pass: deformation-based thrust + morphology asymmetry-based torque
// Thrust: Parallel drag from segment motion (stable forward propulsion)
// Torque: Drag imbalance from body asymmetry relative to current velocity direction
// Fixes excessive spinning; natural alignment and turning from loops/asymmetry

@compute @workgroup_size(256)
fn microswim_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent_id = gid.x;
    if (agent_id >= params.agent_count) {
        return;
    }

    if (!ms_enabled()) {
        return;
    }

    if (agents_out[agent_id].alive == 0u) {
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

    let c_par  = ms_f32(MSP_BASE_DRAG);
    let c_perp = c_par * ms_f32(MSP_ANISOTROPY);  // Recommended: 4.0–6.0

    // Current velocity defines "forward" direction for asymmetry calculation
    var current_vel = agents_out[agent_id].velocity;
    let speed = length(current_vel);
    var forward_dir = vec2<f32>(cos(agent_rot), sin(agent_rot));  // Fallback if stationary
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

        // === Noise & stability filters ===
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

        // Filter structural growth/recentering
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

        // === Thrust: parallel-only for stable forward swimming ===
        let thrust_parallel = (-c_par * v_parallel) * seg_len;
        thrust_world += thrust_parallel;
        total_weight += seg_len;

        // === Asymmetry torque from current morphology ===
        let seg_mid_local = (seg_start_new + seg_end_new) * 0.5;
        let seg_mid_world = apply_agent_rotation(seg_mid_local, agent_rot);

        let lateral_offset = dot(seg_mid_world, lateral_dir);  // Signed offset from centerline
        let perp_drag = c_perp * speed * seg_len;             // Drag proportional to speed

        torque_asymmetry += lateral_offset * perp_drag;       // Torque = offset × force
        inertia_estimate += lateral_offset * lateral_offset * seg_len;
    }

    if (total_deformation_sq < ms_f32(MSP_MIN_TOTAL_DEFORMATION_SQ) || total_weight <= 1e-6) {
        return;
    }

    // Average thrust
    thrust_world /= total_weight;

    // Apply bounded thrust
    var final_thrust_vel = thrust_world * morph_swim_strength;

    // IMPORTANT: microswim injects velocity directly, bypassing the main physics force/drag path.
    // Without a mass normalization, heavier morphologies can gain a disproportionate advantage.
    // Match the intuition of dv = F / m by scaling down for total_mass > 1.
    let total_mass = max(agents_out[agent_id].total_mass, 0.05);
    let mass_scale = 1.0 / max(total_mass, 1.0);
    final_thrust_vel *= mass_scale;

    let tl = length(final_thrust_vel);
    if (tl > ms_f32(MSP_MAX_FRAME_VEL)) {
        final_thrust_vel *= ms_f32(MSP_MAX_FRAME_VEL) / max(tl, 1e-6);
    }

    var agent_vel = agents_out[agent_id].velocity + final_thrust_vel;
    let v_len = length(agent_vel);
    if (v_len > VEL_MAX) {
        agent_vel *= VEL_MAX / max(v_len, 1e-6);
    }
    agents_out[agent_id].velocity = agent_vel;

    // === Apply morphology asymmetry torque ===
    if (inertia_estimate > 1e-6 && speed > 0.1) {
        let normalized_torque = torque_asymmetry / inertia_estimate;
        let torque_scale = ms_f32(MSP_TORQUE_STRENGTH) * morph_swim_strength * dt_safe;

        var delta_rot = normalized_torque * torque_scale * min(speed / 3.0, 1.0);
        delta_rot = clamp(delta_rot, -ANGVEL_MAX, ANGVEL_MAX);

        agents_out[agent_id].rotation += delta_rot;
    }

    // Optional: clear torque_debug if used elsewhere
    // agents_out[agent_id].torque_debug = 0.0;
}
