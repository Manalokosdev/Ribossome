// Dedicated microswimming compute pass.
// Stable forward thrust (parallel-only) + realistic torque from asymmetric/sideways deformation.
// Anisotropy now meaningfully controls turning strength from loops or one-sided waving.

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
    var torque_world = 0.0;           // Positive = counterclockwise
    var total_weight = 0.0;
    var total_deformation_sq = 0.0;

    let c_par  = ms_f32(MSP_BASE_DRAG);
    let c_perp = ms_f32(MSP_BASE_DRAG) * ms_f32(MSP_ANISOTROPY);  // Now affects torque strongly

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
        let seg_vec_len = length(seg_vec_new);
        if (seg_vec_len < 0.01) {
            continue;
        }

        // Filter structural growth/recentering
        let seg_vec_prev = seg_end_prev - seg_start_prev;
        let prev_len = length(seg_vec_prev);
        if (prev_len > 0.01) {
            let len_ratio = seg_vec_len / prev_len;
            if (len_ratio < ms_f32(MSP_MIN_LENGTH_RATIO) || len_ratio > ms_f32(MSP_MAX_LENGTH_RATIO)) {
                continue;
            }
        }

        let seg_len_world = seg_vec_len;
        let tangent_local = seg_vec_new / seg_vec_len;
        let tangent_world = normalize(apply_agent_rotation(tangent_local, agent_rot));

        let v_parallel = dot(v_seg_world, tangent_world) * tangent_world;
        let v_perp     = v_seg_world - v_parallel;

        // === Linear thrust: parallel-only for rock-solid straight swimming ===
        // Perpendicular components cancel across symmetric/asymmetric noise
        let thrust_parallel = (-c_par * v_parallel) * seg_len_world;  // reaction force
        thrust_world += thrust_parallel;

        // === Torque: use FULL anisotropic drag for realistic turning from side loops ===
        // Sideways motion (v_perp) generates strong offset force → natural yaw
        let drag_per_len_full = -c_par * v_parallel - c_perp * v_perp;
        let segment_force_full = drag_per_len_full * seg_len_world;  // reaction on swimmer

        // Lever arm: midpoint of segment relative to agent center (local origin)
        let seg_mid_local = (seg_start_new + seg_end_new) * 0.5;
        let lever_arm_world = apply_agent_rotation(seg_mid_local, agent_rot);

        // 2D cross product: τ = r_x * F_y - r_y * F_x
        torque_world += lever_arm_world.x * segment_force_full.y - lever_arm_world.y * segment_force_full.x;

        total_weight += seg_len_world;
    }

    if (total_deformation_sq < ms_f32(MSP_MIN_TOTAL_DEFORMATION_SQ) || total_weight <= 1e-6) {
        return;
    }

    // Average by total length (scale-invariant)
    thrust_world /= total_weight;
    torque_world /= total_weight;

    // === Apply linear thrust ===
    var final_thrust_vel = thrust_world * morph_swim_strength;
    let max_frame_vel = ms_f32(MSP_MAX_FRAME_VEL);
    let tl = length(final_thrust_vel);
    if (tl > max_frame_vel) {
        final_thrust_vel *= max_frame_vel / max(tl, 1e-6);
    }

    var agent_vel = agents_out[agent_id].velocity + final_thrust_vel;
    let v_len = length(agent_vel);
    if (v_len > VEL_MAX) {
        agent_vel *= VEL_MAX / max(v_len, 1e-6);
    }
    agents_out[agent_id].velocity = agent_vel;

    // === Apply torque-based rotation ===
    // Only turn when actually swimming forward — prevents spinning in place
    let swim_speed = length(final_thrust_vel);
    let speed_scale = clamp(swim_speed / max(max_frame_vel, 1e-6), 0.0, 1.0);

    var delta_rot = torque_world
                  * ms_f32(MSP_TORQUE_STRENGTH)
                  * morph_swim_strength
                  * dt_safe
                  * speed_scale;

    // Respect global angular velocity limits
    delta_rot = clamp(delta_rot, -ANGVEL_MAX, ANGVEL_MAX);

    agents_out[agent_id].rotation += delta_rot;

    // Optional debug visualization
    // NOTE: torque_debug is repurposed as a packed RGB cache in simulation.wgsl.
    // Do not write torque into it from microswim.
}
