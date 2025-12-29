// Dedicated microswimming compute pass.
// Reads previous stable-frame body positions from `agents_in` and current positions from `agents_out`.
// Applies an RFT-like anisotropic-drag thrust based on pure deformation only.

@compute @workgroup_size(256)
fn microswim_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent_id = gid.x;
    if (agent_id >= params.agent_count) {
        return;
    }

    if (!MORPHOLOGY_SWIM_SEPARATE_PASS || !MORPHOLOGY_SWIM_ENABLED) {
        return;
    }

    // NOTE: process_agents has already produced the current-frame state in agents_out.
    // agents_in still contains the previous frame's stable, recentered local geometry.
    if (agents_out[agent_id].alive == 0u) {
        return;
    }

    let body_count_out = min(agents_out[agent_id].body_count, MAX_BODY_PARTS);
    let body_count_in = min(agents_in[agent_id].body_count, MAX_BODY_PARTS);
    let first_build = (body_count_in == 0u);

    // Anchor (type 42): if any anchor was active last frame, freeze the whole agent.
    // Read from agents_in to avoid same-frame ordering dependence.
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

    // Morphology swim strength (reuses the existing prop-wash UI knobs as a convenient control).
    let morph_swim_strength = select(
        0.0,
        max(max(params.prop_wash_strength, 0.0), max(params.prop_wash_strength_fluid, 0.0)) * MORPHOLOGY_SWIM_COUPLING,
        (!first_build)
    );

    if (morph_swim_strength <= 0.0 || body_count_out <= 1u) {
        return;
    }

    let agent_rot = agents_out[agent_id].rotation;
    let dt_safe = max(params.dt, 1e-3);

    var thrust_world = vec2<f32>(0.0);
    var total_weight = 0.0;

    let c_par = MORPHOLOGY_SWIM_BASE_DRAG;
    let c_perp = MORPHOLOGY_SWIM_BASE_DRAG * MORPHOLOGY_SWIM_ANISOTROPY;

    // Iterate true segments between consecutive parts.
    // RFT (low-Re swimming): net drag/thrust scales with filament length (ds), not mass.
    // We therefore weight each segment contribution by its current geometric length.
    for (var j = 1u; j < body_count_out; j++) {
        // Segment endpoints (local), previous and current.
        let seg_start_prev = agents_in[agent_id].body[j - 1u].pos;
        let seg_end_prev = agents_in[agent_id].body[j].pos;
        let seg_start_new = agents_out[agent_id].body[j - 1u].pos;
        let seg_end_new = agents_out[agent_id].body[j].pos;

        // Pure deformation displacement in the stable local frame.
        let delta_start_local = seg_start_new - seg_start_prev;
        let delta_end_local = seg_end_new - seg_end_prev;

        // Segment midpoint deformation velocity (world units per tick)
        let v_seg_local = (delta_start_local + delta_end_local) * 0.5;
        var v_seg_world = apply_agent_rotation(v_seg_local, agent_rot);

        // Clamp huge jumps (e.g. from abrupt morphology changes)
        let dlen = length(v_seg_world);
        if (dlen > MORPHOLOGY_MAX_WORLD_DELTA) {
            v_seg_world = v_seg_world * (MORPHOLOGY_MAX_WORLD_DELTA / max(dlen, 1e-6));
        }
        v_seg_world = v_seg_world / dt_safe;
        let vlen = length(v_seg_world);
        if (vlen > MORPHOLOGY_MAX_WORLD_VEL) {
            v_seg_world = v_seg_world * (MORPHOLOGY_MAX_WORLD_VEL / max(vlen, 1e-6));
        }

        // Segment tangent direction (defines parallel/perp axes)
        let seg_vec_new = seg_end_new - seg_start_new;
        let seg_vec_len = length(seg_vec_new);
        if (seg_vec_len < 1e-4) {
            continue;
        }
        let seg_len_world = seg_vec_len;
        let tangent_local = seg_vec_new / seg_vec_len;
        let tangent_world = normalize(apply_agent_rotation(tangent_local, agent_rot));

        // Decompose velocity into parallel and perpendicular components
        let v_parallel = dot(v_seg_world, tangent_world) * tangent_world;
        let v_perp = v_seg_world - v_parallel;

        // RFT sign convention:
        // `drag_per_len` is the drag force on the filament (opposes motion), with higher drag for
        // perpendicular motion (c_perp > c_par). The force on the swimmer is the opposite of the
        // filament's force on the swimmer, i.e. it points opposite the local deformation velocity.
        let drag_per_len = (-c_par * v_parallel) + (-c_perp * v_perp);
        let segment_thrust = drag_per_len * seg_len_world;

        thrust_world += segment_thrust;
        total_weight += seg_len_world;
    }

    if (total_weight <= 1e-6) {
        return;
    }

    // Average thrust per unit length (keeps behavior consistent across different body sizes).
    thrust_world = thrust_world / total_weight;

    // Convert to a desired velocity contribution and clamp.
    var final_thrust_vel = thrust_world * morph_swim_strength;
    let tl = length(final_thrust_vel);
    if (tl > MORPHOLOGY_SWIM_MAX_FRAME_VEL) {
        final_thrust_vel = final_thrust_vel * (MORPHOLOGY_SWIM_MAX_FRAME_VEL / max(tl, 1e-6));
    }

    // Apply as a velocity delta for the next frame.
    // (process_agents has already integrated position this frame.)
    var agent_vel = agents_out[agent_id].velocity + final_thrust_vel;
    let v_len = length(agent_vel);
    if (v_len > VEL_MAX) {
        agent_vel = agent_vel * (VEL_MAX / max(v_len, 1e-6));
    }
    agents_out[agent_id].velocity = agent_vel;

    // Optional: heading alignment, applied as a small rotation update for the next frame.
    if (MORPHOLOGY_SWIM_HEADING_ALIGN_ENABLED && !DISABLE_GLOBAL_ROTATION) {
        let spd = length(agent_vel);
        if (spd > MORPHOLOGY_SWIM_HEADING_ALIGN_MIN_SPEED) {
            let vel_dir = agent_vel / spd;
            let forward = vec2<f32>(cos(agent_rot), sin(agent_rot));
            let c = forward.x * vel_dir.y - forward.y * vel_dir.x;
            let d = forward.x * vel_dir.x + forward.y * vel_dir.y;
            let angle_err = atan2(c, d);
            let speed_scale = clamp(spd / MORPHOLOGY_SWIM_HEADING_ALIGN_FULL_SPEED, 0.0, 1.0);
            let delta_rot = clamp(
                angle_err * MORPHOLOGY_SWIM_HEADING_ALIGN_STRENGTH,
                -MORPHOLOGY_SWIM_HEADING_ALIGN_MAX_ANGVEL,
                MORPHOLOGY_SWIM_HEADING_ALIGN_MAX_ANGVEL
            ) * speed_scale * ANGULAR_BLEND;

            let clamped_delta_rot = clamp(delta_rot, -ANGVEL_MAX, ANGVEL_MAX);
            agents_out[agent_id].rotation = agent_rot + clamped_delta_rot;
        }
    }
}
