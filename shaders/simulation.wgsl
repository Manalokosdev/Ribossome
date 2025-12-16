// Vampire mouths drain energy from nearby living agents
// ============================================================================

@compute @workgroup_size(256)
fn drain_energy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent_id = gid.x;
    if (agent_id >= params.agent_count) {
        return;
    }

    let agent = agents_in[agent_id];

    // Skip dead agents
    if (agent.alive == 0u || agent.energy <= 0.0) {
        return;
    }

    // Newborn grace: don't attack or get attacked in the first few frames after spawning
    if (agent.age < VAMPIRE_NEWBORN_GRACE_FRAMES) {
        return;
    }

    // Check if this agent has any vampire mouth organs (type 33)
    let body_count = agent.body_count;
    var has_vampire_mouth = false;

    for (var i = 0u; i < MAX_BODY_PARTS; i++) {
        if (i >= body_count) { break; }
        let base_type = get_base_part_type(agents_in[agent_id].body[i].part_type);
        if (base_type == 33u) {
            has_vampire_mouth = true;
            break;
        }
    }

    if (!has_vampire_mouth) {
        return;
    }

    // Collect nearby agents using spatial grid
    let scale = f32(SPATIAL_GRID_SIZE) / f32(SIM_SIZE);
    let my_grid_x = u32(clamp(agents_in[agent_id].position.x * scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));
    let my_grid_y = u32(clamp(agents_in[agent_id].position.y * scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));

    var neighbor_count = 0u;
    var neighbor_ids: array<u32, 64>;

    for (var dy: i32 = -10; dy <= 10; dy++) {
        for (var dx: i32 = -10; dx <= 10; dx++) {
            if (dx == 0 && dy == 0) { continue; }

            let check_x = i32(my_grid_x) + dx;
            let check_y = i32(my_grid_y) + dy;

            if (check_x >= 0 && check_x < i32(SPATIAL_GRID_SIZE) &&
                check_y >= 0 && check_y < i32(SPATIAL_GRID_SIZE)) {

                let check_idx = u32(check_y) * SPATIAL_GRID_SIZE + u32(check_x);
                let raw_neighbor_id = atomicLoad(&agent_spatial_grid[check_idx]);
                // Unmask high bit to get actual agent ID
                let neighbor_id = raw_neighbor_id & 0x7FFFFFFFu;

                if (neighbor_id != SPATIAL_GRID_EMPTY && neighbor_id != SPATIAL_GRID_CLAIMED && neighbor_id != agent_id) {
                    let neighbor = agents_in[neighbor_id];

                    if (neighbor.alive != 0u && neighbor.energy > 0.0 && neighbor.age >= VAMPIRE_NEWBORN_GRACE_FRAMES) {
                        if (neighbor_count < 64u) {
                            neighbor_ids[neighbor_count] = neighbor_id;
                            neighbor_count++;
                        }
                    }
                }
            }
        }
    }

    // No neighbors to drain
    if (neighbor_count == 0u) {
        return;
    }

    // Process each vampire mouth organ (F=4u, G=5u, H=6u)
    var total_energy_gained = 0.0;

    for (var i = 0u; i < MAX_BODY_PARTS; i++) {
        if (i >= body_count) { break; }
        let part = agents_in[agent_id].body[i];
        let base_type = get_base_part_type(part.part_type);

        // Check if this is a vampire mouth organ (type 33)
        if (base_type == 33u) {
            // Cooldown timer (stored in _pad.x) should tick regardless of enable state
            var current_cooldown = agents_in[agent_id].body[i]._pad.x;
            if (current_cooldown > 0.0) {
                current_cooldown -= 1.0;
                agents_in[agent_id].body[i]._pad.x = current_cooldown;
            }

            // Disabler gating:
            // Vampire mouths act by default, but are suppressed when "inhibitor/enabler" organs
            // (type 26) are placed nearby on the same agent body.
            let part_pos = agents_in[agent_id].body[i].pos;
            var block = 0.0;
            for (var j = 0u; j < min(body_count, MAX_BODY_PARTS); j++) {
                let check_type = get_base_part_type(agents_in[agent_id].body[j].part_type);
                if (check_type == 26u) {  // "Enabler" used as disabler for vampire mouths
                    let disabler_pos = agents_in[agent_id].body[j].pos;
                    let d = length(part_pos - disabler_pos);
                    if (d < 20.0) {
                        block += max(0.0, 1.0 - d / 20.0);
                    }
                }
            }
            block = min(block, 1.0);
            let quadratic_block = block * block;
            let mouth_activity = 1.0 - quadratic_block;

            // Only work if not fully suppressed
            if (mouth_activity > 0.0001) {

                // Get mouth world position
                let part_pos = part.pos;
                let rotated_pos = apply_agent_rotation(part_pos, agents_in[agent_id].rotation);
                let mouth_world_pos = agents_in[agent_id].position + rotated_pos;

                // Find closest victim within drain range
                var closest_victim_id = 0xFFFFFFFFu;
                var closest_dist = 999999.0;
                let max_drain_distance = 50.0; // Fixed 50 unit radius

                for (var n = 0u; n < neighbor_count; n++) {
                    let victim_id = neighbor_ids[n];
                    // Only read victim position for distance check (not energy yet)
                    let victim_pos = agents_in[victim_id].position;

                    // Distance from mouth to victim center
                    let delta = mouth_world_pos - victim_pos;
                    let dist = length(delta);

                    if (dist < max_drain_distance && dist < closest_dist) {
                        closest_dist = dist;
                        closest_victim_id = victim_id;
                    }
                }

                // Drain from closest victim only
                if (closest_victim_id != 0xFFFFFFFFu) {
                    // Try to claim the victim's spatial grid cell atomically
                    // This prevents multiple DIFFERENT vampires from draining the same victim simultaneously
                    // But allows the same vampire to drain with multiple mouths
                    let victim_pos = agents_in[closest_victim_id].position;
                    let victim_scale = f32(SPATIAL_GRID_SIZE) / f32(SIM_SIZE);
                    let victim_grid_x = u32(clamp(victim_pos.x * victim_scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));
                    let victim_grid_y = u32(clamp(victim_pos.y * victim_scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));
                    let victim_grid_idx = victim_grid_y * SPATIAL_GRID_SIZE + victim_grid_x;

                    // Atomic claim: mark victim with high bit to indicate it's being drained this frame
                    // The high bit preserves the victim ID for physics (unmask with & 0x7FFFFFFF)
                    // Once marked, the same vampire can drain with multiple mouths
                    let current_cell = atomicLoad(&agent_spatial_grid[victim_grid_idx]);
                    var can_drain = false;

                    // Check if cell contains the victim (with or without high bit)
                    let cell_agent_id = current_cell & 0x7FFFFFFFu;
                    let is_claimed = (current_cell & 0x80000000u) != 0u;

                    if (cell_agent_id == closest_victim_id && !is_claimed) {
                        // Victim is unclaimed - try to mark with high bit
                        let claimed_victim_id = closest_victim_id | 0x80000000u;
                        let claim_result = atomicCompareExchangeWeak(&agent_spatial_grid[victim_grid_idx], closest_victim_id, claimed_victim_id);
                        can_drain = claim_result.exchanged;
                    } else if (cell_agent_id == closest_victim_id && is_claimed) {
                        // Victim is already marked (claimed this frame) - allow same vampire to drain again
                        can_drain = true;
                    }

                    // Only proceed if we can drain this victim AND cooldown is ready
                    if (can_drain && current_cooldown <= 0.0) {
                        // Vampire drain scales down with disabler suppression
                        let victim_energy = agents_in[closest_victim_id].energy;

                        if (victim_energy > 0.0001) {
                            // Absorb up to 50% of victim's energy
                            let absorbed_energy = victim_energy * 0.5 * mouth_activity;

                            // Victim loses 1.5x the absorbed energy (75% total damage)
                            let energy_damage = absorbed_energy * 1.5;
                            agents_in[closest_victim_id].energy = max(0.0, victim_energy - energy_damage);

                            total_energy_gained += absorbed_energy;

                            // Store absorbed amount in _pad.y for visualization
                            agents_in[agent_id].body[i]._pad.y = absorbed_energy;

                            // Set cooldown timer
                            agents_in[agent_id].body[i]._pad.x = VAMPIRE_MOUTH_COOLDOWN;
                        } else {
                            agents_in[agent_id].body[i]._pad.y = 0.0;
                        }

                    } else {
                        // Keep cell claimed (cleared next frame) - this prevents other vampires from attacking same victim
                    }
                } else {
                    // Failed to claim - another vampire got here first
                    agents_in[agent_id].body[i]._pad.y = 0.0;
                }
            } else {
                agents_in[agent_id].body[i]._pad.y = 0.0;
            }
        } else {
            // Not a vampire mouth
            agents_in[agent_id].body[i]._pad.y = 0.0;
        }
    }

    // Add gained energy to this agent
    if (total_energy_gained > 0.0) {
        agents_in[agent_id].energy += total_energy_gained;
    }
}

// ============================================================================

@compute @workgroup_size(256)
fn process_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent_id = gid.x;
    if (agent_id >= params.agent_count) {
        return;
    }

    var agent = agents_in[agent_id];

    // Skip dead agents
    if (agent.alive == 0u) {
        agents_out[agent_id] = agent;
        return;
    }
    // Copy intact agent to output; we'll only modify specific fields below
    agents_out[agent_id] = agent;

    // ====== MORPHOLOGY BUILD ======
    // Genome scan only happens on first frame (body_count == 0)
    // After that, we just use the cached part_type values in body[]
    var body_count_val = agent.body_count;
    var first_build = (agent.body_count == 0u);
    var start_byte = 0u;

    if (first_build) {
        // FIRST BUILD: Scan genome and populate body[].part_type
        var start = 0xFFFFFFFFu;
        if (params.require_start_codon == 1u) {
            start = genome_find_start_codon(agent.genome);
        } else {
            start = genome_find_first_coding_triplet(agent.genome);
        }

        if (start == 0xFFFFFFFFu) {
            // Non-viable genome (no start codon or no codons)
            agents_out[agent_id].alive = 0u;
            agents_out[agent_id].body_count = 0u;
            return;
        }
        start_byte = start;

        // Translate genome into body parts (part_type gets cached in body[])
        var count = 0u;
        // Skip the start codon itself (AUG) - it's consumed for initiation, not translated
        var pos_b = start_byte;
        if (params.require_start_codon == 1u) {
            pos_b = start_byte + 3u;  // Skip AUG start codon
        }

        for (var i = 0u; i < MAX_BODY_PARTS; i++) {
            // Use centralized translation function
            let step = translate_codon_step(agent.genome, pos_b, params.ignore_stop_codons == 1u);

            // Stop if we hit end of genome or stop codon
            if (!step.is_valid) {
                break;
            }

            // Store the translated part_type
            agents_out[agent_id].body[i].part_type = step.part_type;
            count += 1u;
            pos_b += step.bases_consumed;
        }

        count = clamp(count, 0u, MAX_BODY_PARTS);
        agents_out[agent_id].body_count = count;
        body_count_val = count;

        if (count == 0u) {
            agents_out[agent_id].alive = 0u;
            return;
        }
    }

    // REBUILD body positions every frame (enables dynamic shape changes from signals)
    // Now we just read the cached part_type from body[] instead of re-scanning genome

    // Initialize outside the if block so they're in scope for agent_color calculation
    var total_mass_morphology = 0.05; // Default minimum
    var color_sum_morphology = 0.0;

    if (body_count_val > 0u) {

        // Use poison resistance count from previous frame (stored in struct)
        // This value only changes when morphology changes, so it's safe to use the stored value
        let signal_angle_multiplier = pow(0.5, f32(agents_out[agent_id].poison_resistant_count));

        // Dynamic chain build - angles modulated by alpha/beta signals
        var current_pos = vec2<f32>(0.0);
        var current_angle = 0.0;
        var chirality_flip = 1.0;
        var sum_angle_mass = 0.0;
        var total_mass_angle = 0.0;
        var total_capacity = 0.0;
        // Reset to 0 before accumulation
        total_mass_morphology = 0.0;
        color_sum_morphology = 0.0;
        var poison_resistant_count = 0u;

        // Loop through existing body parts and rebuild positions
        for (var i = 0u; i < min(body_count_val, MAX_BODY_PARTS); i++) {
            // Read cached part_type (set during first build or from previous frame)
            let final_part_type = agents_out[agent_id].body[i].part_type;
            let base_type = get_base_part_type(final_part_type);

            // Check for chiral flipper
            if (base_type == 30u) {
                chirality_flip = -chirality_flip;
            }

            // Count poison-resistant organs (type 29) for signal angle modulation
            if (base_type == 29u) {
                poison_resistant_count += 1u;
            }

            let props = get_amino_acid_properties(base_type);
            total_capacity += props.energy_storage;

            // Read previous frame's signals
            let alpha = agents_out[agent_id].body[i].alpha_signal;
            let beta = agents_out[agent_id].body[i].beta_signal;

            // Modulate angle based on signals
            let alpha_effect = alpha * props.alpha_sensitivity * SIGNAL_GAIN * ANGLE_GAIN_ALPHA;
            let beta_effect = beta * props.beta_sensitivity * SIGNAL_GAIN * ANGLE_GAIN_BETA;
            var target_signal_angle = alpha_effect + beta_effect;
            target_signal_angle = target_signal_angle * signal_angle_multiplier;
            target_signal_angle = clamp(target_signal_angle, -MAX_SIGNAL_ANGLE, MAX_SIGNAL_ANGLE);

            var smoothed_signal = target_signal_angle;

            // Apply chirality flip to angles
            current_angle += (props.base_angle + smoothed_signal) * chirality_flip;

            // Accumulate for average angle
            let m = max(props.mass, 0.01);
            sum_angle_mass += current_angle * m;
            total_mass_angle += m;
            // Also accumulate total mass and color sum (only changes when morphology rebuilds)
            total_mass_morphology += m;
            color_sum_morphology += props.beta_damage;

            // Calculate new position
            current_pos.x += cos(current_angle) * props.segment_length;
            current_pos.y += sin(current_angle) * props.segment_length;
            agents_out[agent_id].body[i].pos = current_pos;

            // Update size
            var rendered_size = props.thickness * 0.5;
            let is_sensor = props.is_alpha_sensor || props.is_beta_sensor || props.is_energy_sensor || props.is_agent_alpha_sensor || props.is_agent_beta_sensor || props.is_trail_energy_sensor;
            if (is_sensor) {
                rendered_size *= 2.0;
            }
            if (props.is_condenser) {
                rendered_size *= 0.5;
            }
            agents_out[agent_id].body[i].size = rendered_size;

            // Persist smoothed angle in _pad.x for regular amino acids only
            // Organs (condensers, clocks) use _pad for their own state storage
            // EXCEPTION: Vampire mouths (type 33) need _pad.y preserved for visualization
            let is_organ = (base_type >= 20u);
            let is_vampire_mouth = (base_type == 33u);
            if (!is_organ) {
                let keep_pad_y = agents_out[agent_id].body[i]._pad.y;
                agents_out[agent_id].body[i]._pad = vec2<f32>(smoothed_signal, keep_pad_y);
            } else if (is_vampire_mouth) {
                // Vampire mouths: preserve both cooldown (_pad.x) and drain amount (_pad.y)
                let cooldown = agents_in[agent_id].body[i]._pad.x;
                let drain_amount = agents_in[agent_id].body[i]._pad.y;
                agents_out[agent_id].body[i]._pad = vec2<f32>(cooldown, drain_amount);
            }
        }

        // Energy capacity
        agents_out[agent_id].energy_capacity = total_capacity;

        // Store total mass (only changes when morphology rebuilds)
        agents_out[agent_id].total_mass = max(total_mass_morphology, 0.05);

        // Store poison resistance count (only changes when morphology rebuilds)
        agents_out[agent_id].poison_resistant_count = poison_resistant_count;

        // Center of mass recentering
        var com = vec2<f32>(0.0);
        let rec_n = body_count_val;
        if (rec_n > 0u) {
            var mass_sum = 0.0;
            for (var i = 0u; i < min(rec_n, MAX_BODY_PARTS); i++) {
                let base_type = get_base_part_type(agents_out[agent_id].body[i].part_type);
                let props = get_amino_acid_properties(base_type);
                let m = max(props.mass, 0.01);
                com += agents_out[agent_id].body[i].pos * m;
                mass_sum += m;
            }
            com = com / max(mass_sum, 0.0001);

            // Morphology origin after centering
            var origin_local = -com;
            for (var i = 0u; i < min(rec_n, MAX_BODY_PARTS); i++) {
                agents_out[agent_id].body[i].pos -= com;
            }

            // Calculate mass-weighted average angle
            let avg_angle = sum_angle_mass / max(total_mass_angle, 0.0001);

            // Counteract internal rotation
            if (!DISABLE_GLOBAL_ROTATION) {
                agents_out[agent_id].rotation += avg_angle;
            }

            // Rotate body parts by -avg_angle
            let c_inv = cos(-avg_angle);
            let s_inv = sin(-avg_angle);

            for (var i = 0u; i < min(rec_n, MAX_BODY_PARTS); i++) {
                let p = agents_out[agent_id].body[i].pos;
                agents_out[agent_id].body[i].pos = vec2<f32>(
                    p.x * c_inv - p.y * s_inv,
                    p.x * s_inv + p.y * c_inv
                );
            }

            // Rotate morphology origin
            let o = origin_local;
            agents_out[agent_id].morphology_origin = vec2<f32>(
                o.x * c_inv - o.y * s_inv,
                o.x * s_inv + o.y * c_inv
            );
        }
    }

    // Calculate agent color from color_sum accumulated during morphology rebuild
    let agent_color = vec3<f32>(
        sin(color_sum_morphology * 3.0) * 0.5 + 0.5,      // R: multiplier = 3.0
        sin(color_sum_morphology * 5.25) * 0.5 + 0.5,     // G: multiplier = 5.25
        sin(color_sum_morphology * 7.364) * 0.5 + 0.5     // B: multiplier = 7.364
    );

    let body_count = body_count_val; // Use computed value instead of reading from agent

    // ====== UNIFIED SIGNAL PROCESSING LOOP ======
    // Optimized passes: enabler discovery, amplification calculation, signal storage, and propagation

    var amplification_per_part: array<f32, MAX_BODY_PARTS>;
    var propeller_thrust_magnitude: array<f32, MAX_BODY_PARTS>; // Store thrust for cost calculation
    var old_alpha: array<f32, MAX_BODY_PARTS>;
    var old_beta: array<f32, MAX_BODY_PARTS>;

    // First pass: find all enablers and store their positions, store signals
    var enabler_positions: array<vec2<f32>, MAX_BODY_PARTS>;
    var enabler_count = 0u;

    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let part_i = agents_out[agent_id].body[i];
        let base_type = get_base_part_type(part_i.part_type);
        let props = get_amino_acid_properties(base_type);

        // Store old signals for propagation
        old_alpha[i] = part_i.alpha_signal;
        old_beta[i] = part_i.beta_signal;

        // Collect enabler positions
        if (props.is_inhibitor) { // enabler role
            enabler_positions[enabler_count] = part_i.pos;
            enabler_count += 1u;
        }
    }

    // ====== COLLECT NEARBY AGENTS ONCE (for sensors and physics) ======
    let scale = f32(SPATIAL_GRID_SIZE) / f32(SIM_SIZE);
    let my_grid_x = u32(clamp(agent.position.x * scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));
    let my_grid_y = u32(clamp(agent.position.y * scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));

    var neighbor_count = 0u;
    var neighbor_ids: array<u32, 64>; // Store up to 64 nearby agents

    for (var dy: i32 = -10; dy <= 10; dy++) {
        for (var dx: i32 = -10; dx <= 10; dx++) {
            let check_x = i32(my_grid_x) + dx;
            let check_y = i32(my_grid_y) + dy;

            if (check_x >= 0 && check_x < i32(SPATIAL_GRID_SIZE) &&
                check_y >= 0 && check_y < i32(SPATIAL_GRID_SIZE)) {

                let check_idx = u32(check_y) * SPATIAL_GRID_SIZE + u32(check_x);
                let raw_neighbor_id = atomicLoad(&agent_spatial_grid[check_idx]);
                // Unmask high bit to get actual agent ID (vampire claim bit)
                let neighbor_id = raw_neighbor_id & 0x7FFFFFFFu;

                if (neighbor_id != SPATIAL_GRID_EMPTY && neighbor_id != SPATIAL_GRID_CLAIMED && neighbor_id != agent_id) {
                    let neighbor = agents_in[neighbor_id];

                    if (neighbor.alive != 0u && neighbor.energy > 0.0) {
                        if (neighbor_count < 64u) {
                            neighbor_ids[neighbor_count] = neighbor_id;
                            neighbor_count++;
                        }
                    }
                }
            }
        }
    }

    // Second pass: calculate amplification and propagate signals (merged for efficiency)
    // Track cumulative chirality so directional env sensors can swap left/right.
    var chirality_flip_signal = 1.0;
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let part_pos = agents_out[agent_id].body[i].pos;
        let base_type = get_base_part_type(agents_out[agent_id].body[i].part_type);
        let amino_props = get_amino_acid_properties(base_type);

        if (base_type == 30u) {
            chirality_flip_signal = -chirality_flip_signal;
        }

        // Calculate amplification using enabler list (O(n +? e) instead of O(n-?))
        var amp = 0.0;
        for (var e = 0u; e < enabler_count; e++) {
            let d = length(part_pos - enabler_positions[e]);
            if (d < 20.0) {
                amp += max(0.0, 1.0 - d / 20.0);
            }
        }
        amplification_per_part[i] = min(amp, 1.0);
        propeller_thrust_magnitude[i] = 0.0; // Initialize

        // Propagate signals through chain

        let has_left = i > 0u;
        let has_right = i < body_count - 1u;

        var new_alpha = 0.0;
        var new_beta = 0.0;
        if (params.interior_isotropic == 1u) {
            // Isotropic: use immediate neighbors only
            let left_a = select(0.0, old_alpha[i - 1u], has_left);
            let right_a = select(0.0, old_alpha[i + 1u], has_right);
            let left_b = select(0.0, old_beta[i - 1u], has_left);
            let right_b = select(0.0, old_beta[i + 1u], has_right);
            let count = (select(0.0, 1.0, has_left) + select(0.0, 1.0, has_right));
            if (count > 0.0) {
                new_alpha = (left_a + right_a) / count;
                new_beta = (left_b + right_b) / count;
            } else {
                // Single-part edge case: carry previous value (no neighbors)
                new_alpha = old_alpha[i];
                new_beta = old_beta[i];
            }
        } else {
            // Anisotropic: use per-amino left/right multipliers
            let alpha_from_left = select(0.0, old_alpha[i - 1u] * amino_props.alpha_left_mult, has_left);
            let alpha_from_right = select(0.0, old_alpha[i + 1u] * amino_props.alpha_right_mult, has_right);
            let beta_from_left = select(0.0, old_beta[i - 1u] * amino_props.beta_left_mult, has_left);
            let beta_from_right = select(0.0, old_beta[i + 1u] * amino_props.beta_right_mult, has_right);
            new_alpha = alpha_from_left + alpha_from_right;
            new_beta = beta_from_left + beta_from_right;
        }

        // Sensors:
        // - Environment sensors now read from the (diffused) dye layer deterministically.
        //   The old "sensor radius" is now interpreted as a signal gain multiplier.
        //   Actual env-sensor strength is derived from promoter+modifier param1 inside the sampling helpers.
        // - Neighbor/trail sensors still use a true spatial search radius.
        let env_sensor_gain_mult = 1.0;
        let neighbor_search_radius = 500.0;

        // Calculate sensor perpendicular orientation (pointing direction)
        var segment_dir = vec2<f32>(0.0);
        if (i > 0u) {
            let prev = agents_out[agent_id].body[i-1u].pos;
            segment_dir = agents_out[agent_id].body[i].pos - prev;
        } else if (body_count > 1u) {
            let next = agents_out[agent_id].body[1u].pos;
            segment_dir = next - agents_out[agent_id].body[i].pos;
        } else {
            // Single-part body: use forward direction
            segment_dir = vec2<f32>(1.0, 0.0);
        }
        let seg_len = length(segment_dir);
        let axis_local = select(segment_dir / seg_len, vec2<f32>(1.0, 0.0), seg_len < 1e-4);
        // Perpendicular (right-hand) to segment axis
        let perpendicular_local = normalize(vec2<f32>(-axis_local.y, axis_local.x));
        // Rotate to world space
        let perpendicular_world = normalize(apply_agent_rotation(perpendicular_local, agent.rotation));

        if (amino_props.is_alpha_sensor) {
            let rotated_pos = apply_agent_rotation(agents_out[agent_id].body[i].pos, agent.rotation);
            let world_pos = agent.position + rotated_pos;
            let sensor_seed = agent_id * 1000u + i * 13u;

            // Extract promoter and modifier parameter1 values
            // The part_type encodes the modifier (0-19) as parameter (0-255)
            let organ_param = get_organ_param(agents_out[agent_id].body[i].part_type);

            // Convert organ_param (0-255) back to modifier index (0-19)
            let modifier_index = u32((f32(organ_param) / 255.0) * 19.0);

            // Alpha sensors use promoters V(17) or M(10), get their parameter1 values
            // For sensors: organ 22 can be from V(17)+mod or M(10)+mod
            // We need to determine which promoter was used - check agent's genome or use base approximation
            // For simplicity, we'll use the organ's own parameter1 as promoter baseline
            // Reuse amino_props which was already computed for base_type
            let promoter_param1 = amino_props.parameter1;

            // Get modifier amino acid parameter1
            let modifier_props = get_amino_acid_properties(modifier_index);
            let modifier_param1 = modifier_props.parameter1;

            let sensed_value = sample_stochastic_gaussian(world_pos, env_sensor_gain_mult, sensor_seed, 0u, params.debug_mode != 0u, perpendicular_world, chirality_flip_signal, promoter_param1, modifier_param1);
            // Apply sqrt to increase sensitivity to low signals (0.01 -> 0.1, 0.25 -> 0.5, 1.0 -> 1.0)
            let nonlinear_value = sqrt(clamp(abs(sensed_value), 0.0, 1.0)) * sign(sensed_value);
            // Add sensor contribution to diffused signal (instead of mixing)
            new_alpha = new_alpha + nonlinear_value;
        }
        if (amino_props.is_beta_sensor) {
            let rotated_pos = apply_agent_rotation(agents_out[agent_id].body[i].pos, agent.rotation);
            let world_pos = agent.position + rotated_pos;
            let sensor_seed = agent_id * 1000u + i * 13u;

            // Extract promoter and modifier parameter1 values
            let organ_param = get_organ_param(agents_out[agent_id].body[i].part_type);

            // Convert organ_param (0-255) back to modifier index (0-19)
            let modifier_index = u32((f32(organ_param) / 255.0) * 19.0);

            // Get promoter and modifier parameter1 values
            // Reuse amino_props which was already computed for base_type
            let promoter_param1 = amino_props.parameter1;

            let modifier_props = get_amino_acid_properties(modifier_index);
            let modifier_param1 = modifier_props.parameter1;

            let sensed_value = sample_stochastic_gaussian(world_pos, env_sensor_gain_mult, sensor_seed, 1u, params.debug_mode != 0u, perpendicular_world, chirality_flip_signal, promoter_param1, modifier_param1);
            // Apply sqrt to increase sensitivity to low signals (0.01 -> 0.1, 0.25 -> 0.5, 1.0 -> 1.0)
            let nonlinear_value = sqrt(clamp(abs(sensed_value), 0.0, 1.0)) * sign(sensed_value);
            // Add sensor contribution to diffused signal (instead of mixing)
            new_beta = new_beta + nonlinear_value;
        }

        // TRAIL ENERGY SENSOR - Senses nearby agent energies from trail
        if (amino_props.is_trail_energy_sensor) {
            let rotated_pos = apply_agent_rotation(agents_out[agent_id].body[i].pos, agent.rotation);
            let world_pos = agent.position + rotated_pos;

            // Get sensor orientation (perpendicular to organ)
            let axis_local = normalize(agents_out[agent_id].body[i].pos);
            let perpendicular_local = normalize(vec2<f32>(-axis_local.y, axis_local.x));
            let perpendicular_world = normalize(apply_agent_rotation(perpendicular_local, agent.rotation));

            let sensed_value = sample_neighbors_energy(world_pos, neighbor_search_radius, params.debug_mode != 0u, perpendicular_world, &neighbor_ids, neighbor_count);
            // Normalize by a scaling factor (energy can be large, scale to -1..1 range)
            let normalized_value = tanh(sensed_value * 0.01); // tanh for soft clamping to -1..1
            // Split into alpha and beta based on sign (positive energy -> alpha, negative -> beta)
            new_alpha += max(normalized_value, 0.0);
            new_beta += max(-normalized_value, 0.0);
        }

        // ALPHA MAGNITUDE SENSORS - Organ types 38, 39 (V/M + I/K)
        // Measure alpha signal strength without directional bias
        if (base_type == 38u || base_type == 39u) {
            let rotated_pos = apply_agent_rotation(agents_out[agent_id].body[i].pos, agent.rotation);
            let world_pos = agent.position + rotated_pos;
            let sensor_seed = agent_id * 1000u + i * 13u;

            let organ_param = get_organ_param(agents_out[agent_id].body[i].part_type);
            let modifier_index = u32((f32(organ_param) / 255.0) * 19.0);
            let promoter_param1 = amino_props.parameter1;
            let modifier_props = get_amino_acid_properties(modifier_index);
            let modifier_param1 = modifier_props.parameter1;

            let sensed_value = sample_magnitude_only(world_pos, env_sensor_gain_mult, sensor_seed, 0u, params.debug_mode != 0u, promoter_param1, modifier_param1);
            let nonlinear_value = sqrt(clamp(abs(sensed_value), 0.0, 1.0)) * sign(sensed_value);
            new_alpha = new_alpha + nonlinear_value;
        }

        // BETA MAGNITUDE SENSORS - Organ types 40, 41 (V/M + T/V)
        // Measure beta signal strength without directional bias
        if (base_type == 40u || base_type == 41u) {
            let rotated_pos = apply_agent_rotation(agents_out[agent_id].body[i].pos, agent.rotation);
            let world_pos = agent.position + rotated_pos;
            let sensor_seed = agent_id * 1000u + i * 13u;

            let organ_param = get_organ_param(agents_out[agent_id].body[i].part_type);
            let modifier_index = u32((f32(organ_param) / 255.0) * 19.0);
            let promoter_param1 = amino_props.parameter1;
            let modifier_props = get_amino_acid_properties(modifier_index);
            let modifier_param1 = modifier_props.parameter1;

            let sensed_value = sample_magnitude_only(world_pos, env_sensor_gain_mult, sensor_seed, 1u, params.debug_mode != 0u, promoter_param1, modifier_param1);
            let nonlinear_value = sqrt(clamp(abs(sensed_value), 0.0, 1.0)) * sign(sensed_value);
            new_beta = new_beta + nonlinear_value;
        }

        // ALPHA MAGNITUDE SENSORS - Organ types 38, 39 (V/M + I/K)
        // Measure alpha signal strength without directional bias
        if (base_type == 38u || base_type == 39u) {
            let rotated_pos = apply_agent_rotation(agents_out[agent_id].body[i].pos, agent.rotation);
            let world_pos = agent.position + rotated_pos;
            let sensor_seed = agent_id * 1000u + i * 13u;

            let organ_param = get_organ_param(agents_out[agent_id].body[i].part_type);
            let modifier_index = u32((f32(organ_param) / 255.0) * 19.0);
            let promoter_param1 = amino_props.parameter1;
            let modifier_props = get_amino_acid_properties(modifier_index);
            let modifier_param1 = modifier_props.parameter1;

            let sensed_value = sample_magnitude_only(world_pos, env_sensor_gain_mult, sensor_seed, 0u, params.debug_mode != 0u, promoter_param1, modifier_param1);
            let nonlinear_value = sqrt(clamp(abs(sensed_value), 0.0, 1.0)) * sign(sensed_value);
            new_alpha = new_alpha + nonlinear_value;
        }

        // BETA MAGNITUDE SENSORS - Organ types 40, 41 (V/M + T/V)
        // Measure beta signal strength without directional bias
        if (base_type == 40u || base_type == 41u) {
            let rotated_pos = apply_agent_rotation(agents_out[agent_id].body[i].pos, agent.rotation);
            let world_pos = agent.position + rotated_pos;
            let sensor_seed = agent_id * 1000u + i * 13u;

            let organ_param = get_organ_param(agents_out[agent_id].body[i].part_type);
            let modifier_index = u32((f32(organ_param) / 255.0) * 19.0);
            let promoter_param1 = amino_props.parameter1;
            let modifier_props = get_amino_acid_properties(modifier_index);
            let modifier_param1 = modifier_props.parameter1;

            let sensed_value = sample_magnitude_only(world_pos, env_sensor_gain_mult, sensor_seed, 1u, params.debug_mode != 0u, promoter_param1, modifier_param1);
            let nonlinear_value = sqrt(clamp(abs(sensed_value), 0.0, 1.0)) * sign(sensed_value);
            new_beta = new_beta + nonlinear_value;
        }

        // ALPHA MAGNITUDE SENSORS - Organ types 38, 39 (V/M + I/K)
        // Measure alpha signal strength without directional bias
        if (base_type == 38u || base_type == 39u) {
            let rotated_pos = apply_agent_rotation(agents_out[agent_id].body[i].pos, agent.rotation);
            let world_pos = agent.position + rotated_pos;
            let sensor_seed = agent_id * 1000u + i * 13u;

            let organ_param = get_organ_param(agents_out[agent_id].body[i].part_type);
            let modifier_index = u32((f32(organ_param) / 255.0) * 19.0);
            let promoter_param1 = amino_props.parameter1;
            let modifier_props = get_amino_acid_properties(modifier_index);
            let modifier_param1 = modifier_props.parameter1;

            let sensed_value = sample_magnitude_only(world_pos, env_sensor_gain_mult, sensor_seed, 0u, params.debug_mode != 0u, promoter_param1, modifier_param1);
            let nonlinear_value = sqrt(clamp(abs(sensed_value), 0.0, 1.0)) * sign(sensed_value);
            new_alpha = new_alpha + nonlinear_value;
        }

        // BETA MAGNITUDE SENSORS - Organ types 40, 41 (V/M + T/V)
        // Measure beta signal strength without directional bias
        if (base_type == 40u || base_type == 41u) {
            let rotated_pos = apply_agent_rotation(agents_out[agent_id].body[i].pos, agent.rotation);
            let world_pos = agent.position + rotated_pos;
            let sensor_seed = agent_id * 1000u + i * 13u;

            let organ_param = get_organ_param(agents_out[agent_id].body[i].part_type);
            let modifier_index = u32((f32(organ_param) / 255.0) * 19.0);
            let promoter_param1 = amino_props.parameter1;
            let modifier_props = get_amino_acid_properties(modifier_index);
            let modifier_param1 = modifier_props.parameter1;

            let sensed_value = sample_magnitude_only(world_pos, env_sensor_gain_mult, sensor_seed, 1u, params.debug_mode != 0u, promoter_param1, modifier_param1);
            let nonlinear_value = sqrt(clamp(abs(sensed_value), 0.0, 1.0)) * sign(sensed_value);
            new_beta = new_beta + nonlinear_value;
        }

        // AGENT ALPHA SENSOR - Organ type 34 (V/M + L)
        // Senses nearby agent colors from trail
        if (base_type == 34u) {
            let rotated_pos = apply_agent_rotation(agents_out[agent_id].body[i].pos, agent.rotation);
            let sensor_world_pos = agent.position + rotated_pos;

            // Get sensor orientation (perpendicular to organ, rotated 90 degrees)
            let axis_local = normalize(agents_out[agent_id].body[i].pos);
            let perpendicular_local = normalize(vec2<f32>(-axis_local.y, axis_local.x));
            let perpendicular_world = normalize(apply_agent_rotation(perpendicular_local, agent.rotation));

            // Use agent_color calculated from color_sum_morphology
            let sensed_value = sample_neighbors_color(sensor_world_pos, neighbor_search_radius, params.debug_mode != 0u, perpendicular_world, agent_color, &neighbor_ids, neighbor_count);

            // Add agent color difference signal to alpha
            new_alpha += sensed_value;
        }

        // AGENT BETA SENSOR - Organ type 35 (V/M + Y)
        // Senses nearby agent colors from trail
        if (base_type == 35u) {
            let rotated_pos = apply_agent_rotation(agents_out[agent_id].body[i].pos, agent.rotation);
            let sensor_world_pos = agent.position + rotated_pos;

            // Get sensor orientation (perpendicular to organ, rotated 90 degrees)
            let axis_local = normalize(agents_out[agent_id].body[i].pos);
            let perpendicular_local = normalize(vec2<f32>(-axis_local.y, axis_local.x));
            let perpendicular_world = normalize(apply_agent_rotation(perpendicular_local, agent.rotation));

            // Use agent_color calculated from color_sum_morphology
            let sensed_value = sample_neighbors_color(sensor_world_pos, neighbor_search_radius, params.debug_mode != 0u, perpendicular_world, agent_color, &neighbor_ids, neighbor_count);

            // Add agent color difference signal to beta
            new_beta += sensed_value;
        }

        // PAIRING STATE SENSOR - Organ type 36 (H/Q + I/K)
        // Emits alpha or beta based on genome pairing completion percentage
        if (base_type == 36u) {
            // Get pairing percentage (0.0 to 1.0)
            let pairing_percentage = f32(agent.pairing_counter) / f32(GENOME_BYTES);

            // Extract parameter (0-127) and promoter bit
            let organ_param = get_organ_param(agents_out[agent_id].body[i].part_type);
            let param_normalized = f32(organ_param & 127u) / 127.0;
            let is_beta_emitter = (organ_param & 128u) != 0u;

            // Signal strength = pairing_percentage * param_normalized
            let signal_strength = pairing_percentage * param_normalized;

            if (is_beta_emitter) {
                new_beta += signal_strength;
            } else {
                new_alpha += signal_strength;
            }
        }

    // Energy sensor contribution rate (now 1.0 as requested)
    let accumulation_rate = 1.0;
        if (amino_props.is_energy_sensor) {
            // Energy sensor: lerp from 0->50 energy to alpha(-0.5->1.3) and beta(0.5->-0.7)
            let energy_t = clamp(agent.energy / 50.0, 0.0, 1.0);
            let energy_alpha = mix(-0.5, 1.3, energy_t);
            let energy_beta = mix(0.5, -0.7, energy_t);
            new_alpha += energy_alpha * accumulation_rate;
            new_beta += energy_beta * accumulation_rate;
        }

        // SINE WAVE CLOCK ORGAN
        if (amino_props.is_clock) {
            // Determine which signal type this clock emits based on promoter
            // K (Lysine) = 8 ? emits Alpha, advances on Beta
            // C (Cysteine) = 1 ? emits Beta, advances on Alpha
            let base_type = get_base_part_type(agents_out[agent_id].body[i].part_type);
            let organ_param = get_organ_param(agents_out[agent_id].body[i].part_type);

            // Decode promoter type from bit 7: K=0 (alpha), C=1 (beta)
            let is_C_promoter = ((organ_param & 128u) != 0u);

            // Get modifier index from lower 7 bits
            let modifier_index = u32((f32(organ_param & 127u) / 127.0) * 19.0);

            // Get modifier parameter1 for clock frequency scaling
            let modifier_props = get_amino_acid_properties(modifier_index);
            let param1 = modifier_props.parameter1;
            let is_standalone_clock = (modifier_index == 14u || modifier_index == 15u); // R/S modifiers = standalone oscillators

            // Determine field to sense and emit based on promoter
            let is_alpha_emitter = !is_C_promoter; // K emits alpha, C emits beta

            // Compute clock signal using sin(ax)
            var clock_signal: f32 = 0.0;
            if (is_standalone_clock) {
                // Standalone clocks: x = agent.age * param1
                let x = f32(agents_out[agent_id].age) * param1;
                clock_signal = sin(x);
            } else {
                // Signal-driven clocks: x = internal_value * parameter1
                // K senses beta, C senses alpha (opposite of what they emit)
                let sensed_field = select(new_alpha, new_beta, is_alpha_emitter); // K?beta, C?alpha
                // Use _pad.x to accumulate sensed field over time (NOT used by prev_pos for organs)
                var internal_value = agents_out[agent_id].body[i]._pad.x;
                internal_value += sensed_field;
                agents_out[agent_id].body[i]._pad.x = internal_value;

                let x = internal_value * param1;
                clock_signal = sin(x);
            }

            // Store clock signal in _pad.y for rendering (avoids conflict with prev_pos)
            agents_out[agent_id].body[i]._pad.y = clock_signal;

            // Emit to appropriate signal type
            if (is_alpha_emitter) {
                new_alpha = new_alpha + clock_signal;
            } else {
                new_beta = new_beta + clock_signal;
            }
        }

        // SLOPE SENSOR ORGAN (type 32u)
        // K/C + M/N/P/Q modifiers: samples slope gradient and emits signal based on orientation dot product
        let base_type_slope = get_base_part_type(agents_out[agent_id].body[i].part_type);
        if (base_type_slope == 32u) {
            // Get slope gradient at this position
            let world_pos = agent.position + apply_agent_rotation(part_pos, agent.rotation);
            let slope_gradient = read_gamma_slope(grid_index(world_pos));

            // Calculate orientation vector (perpendicular to segment)
            var segment_dir = vec2<f32>(0.0);
            if (i > 0u) {
                let prev = agents_out[agent_id].body[i-1u].pos;
                segment_dir = part_pos - prev;
            } else if (agents_out[agent_id].body_count > 1u) {
                let next = agents_out[agent_id].body[1u].pos;
                segment_dir = next - part_pos;
            } else {
                segment_dir = vec2<f32>(1.0, 0.0);
            }
            let seg_len = length(segment_dir);
            let axis_local = select(segment_dir / seg_len, vec2<f32>(1.0, 0.0), seg_len < 1e-4);
            let orientation_local = vec2<f32>(-axis_local.y, axis_local.x);
            let orientation_world = apply_agent_rotation(orientation_local, agent.rotation);

            // Dot product of slope with orientation
            let slope_alignment = dot(slope_gradient, orientation_world);

            // Get modifier parameter (encoded in part_type)
            let organ_param = get_organ_param(agents_out[agent_id].body[i].part_type);

            // Decode promoter type from bit 7: K=0 (alpha), C=1 (beta)
            let is_C_promoter = ((organ_param & 128u) != 0u);

            // Get modifier parameter (lower 7 bits)
            let modifier_param1 = f32(organ_param & 127u) / 127.0;

            // Generate signal: slope alignment * (modifier_param1 + props.parameter1)
            let signal_strength = slope_alignment * (modifier_param1 + amino_props.parameter1);

            // Emit to appropriate signal channel based on promoter
            // K promoter ? alpha signal, C promoter ? beta signal
            if (is_C_promoter) {
                new_beta = new_beta + signal_strength;
            } else {
                new_alpha = new_alpha + signal_strength;
            }
        }

        // Apply decay to non-sensor signals
        // Sensors are direct sources, condensers output directly without accumulation
        if (!amino_props.is_alpha_sensor && !amino_props.is_trail_energy_sensor) { new_alpha *= 0.99; }
        if (!amino_props.is_beta_sensor && !amino_props.is_trail_energy_sensor) { new_beta *= 0.99; }

        // Smooth internal signal changes to prevent sudden oscillations (75% new, 25% old)
        let update_rate = 0.75;
        let smoothed_alpha = mix(old_alpha[i], new_alpha, update_rate);
        let smoothed_beta = mix(old_beta[i], new_beta, update_rate);

        // Clamp to -1.0 to 1.0 (allows inhibitory and excitatory signals)
        agents_out[agent_id].body[i].alpha_signal = clamp(smoothed_alpha, -1.0, 1.0);
        agents_out[agent_id].body[i].beta_signal = clamp(smoothed_beta, -1.0, 1.0);
    }

    // ====== PHYSICS CALCULATIONS ======
    // Agent already centered at local (0,0) after morphology re-centering
    let center_of_mass = vec2<f32>(0.0);
    let total_mass = agents_out[agent_id].total_mass; // Already calculated during morphology
    let morphology_origin = agents_out[agent_id].morphology_origin;

    let drag_coefficient = total_mass * 0.5;

    // Accumulate forces and torques (relative to CoM)
    var force = vec2<f32>(0.0);
    var torque = 0.0;

    // Agent-to-agent repulsion (simplified: once per agent pair, using total masses)
    for (var n = 0u; n < neighbor_count; n++) {
        let neighbor = agents_out[neighbor_ids[n]];

        let delta = agent.position - neighbor.position;
        let dist = length(delta);

        // Distance-based repulsion force (inverse square law with cutoff)
        let max_repulsion_distance = 500.0;

        if (dist < max_repulsion_distance && dist > 0.1) {
            // Inverse square repulsion: F = k / (d^2)
            let base_strength = params.agent_repulsion_strength * 100000.0;
            let force_magnitude = base_strength / (dist * dist);

            // Clamp to prevent extreme forces at very small distances
            let clamped_force = min(force_magnitude, 5000.0);

            let direction = delta / dist; // Normalize

            // Use reduced mass for proper two-body physics: Î¼ = (m1 * m2) / (m1 + m2)
            let neighbor_mass = max(neighbor.total_mass, 0.01);
            let reduced_mass = (total_mass * neighbor_mass) / (total_mass + neighbor_mass);

            force += direction * clamped_force * reduced_mass;
        }
    }

    // Now calculate forces using the updated morphology (using pre-collected neighbors)
    var chirality_flip_physics = 1.0; // Track cumulative chirality for propeller direction
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let part = agents_out[agent_id].body[i];

        // Get amino acid properties
        let base_type = get_base_part_type(part.part_type);
        let amino_props = get_amino_acid_properties(base_type);

        // Check if this part is Leucine (index 9) and flip chirality
        if (base_type == 9u) {
            chirality_flip_physics = -chirality_flip_physics;
        }

        // Calculate segment midpoint for force application and torque
        var segment_start_chain = vec2<f32>(0.0);
        if (i > 0u) {
            segment_start_chain = agents_out[agent_id].body[i - 1u].pos;
        }
        let segment_midpoint_chain = (segment_start_chain + part.pos) * 0.5;
        let segment_midpoint = morphology_origin + segment_midpoint_chain;

        // Use midpoint for physics calculations
        let offset_from_com = segment_midpoint - center_of_mass;
        let r_com = apply_agent_rotation(offset_from_com, agent.rotation);
        let rotated_midpoint = apply_agent_rotation(segment_midpoint, agent.rotation);
        let world_pos = agent.position + rotated_midpoint;

        let part_mass = max(amino_props.mass, 0.01);
        let part_weight = part_mass / total_mass;

        // Slope force per amino acid
        let slope_gradient = read_gamma_slope(grid_index(world_pos));
        let slope_force = -slope_gradient * params.gamma_strength * part_mass;
        force += slope_force;
        torque += (r_com.x * slope_force.y - r_com.y * slope_force.x);

        // Fluid force per amino acid (apply at the same point as slope force)
        if (params.fluid_wind_push_strength != 0.0) {
            // Default regular amino acids to coupling=1.0 unless explicitly overridden.
            let wind_coupling = select(
                amino_props.fluid_wind_coupling,
                1.0,
                (base_type < 20u) && (amino_props.fluid_wind_coupling == 0.0)
            );

            if (wind_coupling != 0.0) {
                // Avoid clamping OOB into edge cells (prevents artificial edge wind).
                if (world_pos.x >= 0.0 && world_pos.x < f32(SIM_SIZE) && world_pos.y >= 0.0 && world_pos.y < f32(SIM_SIZE)) {
                    let grid_f = f32(FLUID_GRID_SIZE);
                    let max_idx_f = grid_f - 1.0;
                    let fluid_grid_x = u32(clamp(world_pos.x / f32(SIM_SIZE) * grid_f, 0.0, max_idx_f));
                    let fluid_grid_y = u32(clamp(world_pos.y / f32(SIM_SIZE) * grid_f, 0.0, max_idx_f));
                    let fluid_idx = fluid_grid_y * FLUID_GRID_SIZE + fluid_grid_x;

                    // Fluid velocity is in fluid-cell units; convert to world-units per simulation tick.
                    let v = fluid_velocity[fluid_idx];
                    let cell_to_world = f32(SIM_SIZE) / f32(FLUID_GRID_SIZE);
                    let v_frame = v * (cell_to_world);

                    // Overdamped: choose force so resulting velocity contribution ~ v_frame scaled.
                    let wind_force = v_frame * (wind_coupling * part_weight * params.fluid_wind_push_strength) * drag_coefficient;
                    force += wind_force;
                    torque += (r_com.x * wind_force.y - r_com.y * wind_force.x);
                }
            }
        }

        // Cached amplification for this part (organs will use it, organs may ignore)
        let amplification = amplification_per_part[i];



    // Propeller force - check if this amino acid provides thrust
    // Propellers only work if agent has enough energy to cover their consumption cost
    if (amino_props.is_propeller && agent.energy >= amino_props.energy_consumption) {
            // Determine local segment axis using neighbour part to make thrust perpendicular to actual segment, not radial.
            var segment_dir = vec2<f32>(0.0);
            if (i > 0u) {
                let prev = agents_out[agent_id].body[i-1u].pos;
                segment_dir = part.pos - prev;
            } else if (agents_out[agent_id].body_count > 1u) {
                let next = agents_out[agent_id].body[1u].pos;
                segment_dir = next - part.pos;
            } else {
                // Single-part body fallback
                segment_dir = vec2<f32>(1.0, 0.0);
            }
            let seg_len = length(segment_dir);
            let axis_local = select(segment_dir / seg_len, vec2<f32>(1.0, 0.0), seg_len < 1e-4);
            // Perpendicular (right-hand) to the segment axis is our thrust direction in local space
            // Apply chirality flip to thrust direction
            let thrust_local = normalize(vec2<f32>(-axis_local.y, axis_local.x)) * chirality_flip_physics;
            // Rotate to world space
            let thrust_dir_world = apply_agent_rotation(thrust_local, agent.rotation);

            // Prop wash displacement: stochastic transfer in thrust direction
            let prop_dir_len = length(thrust_dir_world);
            if (prop_dir_len > 1e-5) {
                let prop_dir = thrust_dir_world / prop_dir_len;
                // Scale prop wash by amplification so stronger jets stir more environment
                let prop_strength = max(params.prop_wash_strength * amplification, 0.0);

                if (prop_strength > 0.0) {
                    let clamped_pos = clamp_position(world_pos);
                    let grid_scale = f32(SIM_SIZE) / f32(GRID_SIZE);
                    var gx = i32(clamped_pos.x / grid_scale);
                    var gy = i32(clamped_pos.y / grid_scale);
                    gx = clamp(gx, 0, i32(GRID_SIZE) - 1);
                    gy = clamp(gy, 0, i32(GRID_SIZE) - 1);

                    let center_idx = u32(gy) * GRID_SIZE + u32(gx);
                    let distance = clamp(prop_strength * 2.0, 1.0, 5.0);
                    let target_world = clamped_pos + prop_dir * distance * grid_scale;
                    let target_gx = clamp(i32(round(target_world.x / grid_scale)), 0, i32(GRID_SIZE) - 1);
                    let target_gy = clamp(i32(round(target_world.y / grid_scale)), 0, i32(GRID_SIZE) - 1);
                    let target_idx = u32(target_gy) * GRID_SIZE + u32(target_gx);

                    if (target_idx != center_idx) {
                        var center_gamma = read_gamma_height(center_idx);
                        var target_gamma = read_gamma_height(target_idx);
                        var center_alpha = alpha_grid[center_idx];
                        var target_alpha = alpha_grid[target_idx];
                        var center_beta = beta_grid[center_idx];
                        var target_beta = beta_grid[target_idx];

                        let transfer_amount = prop_strength * 0.05 * part_weight;
                        if (transfer_amount > 0.0) {
                            // Capacities adjusted for 0..1 range
                            let alpha_capacity = max(0.0, 1.0 - target_alpha);
                            let beta_capacity = max(0.0, 1.0 - target_beta);
                            let gamma_capacity = max(0.0, 1.0 - target_gamma);

                            let gamma_transfer = min(min(center_gamma, transfer_amount), gamma_capacity);
                            let alpha_transfer = min(min(center_alpha, transfer_amount), alpha_capacity);
                            let beta_transfer = min(min(center_beta, transfer_amount), beta_capacity);

                            if (gamma_transfer > 0.0) {
                                center_gamma = center_gamma - gamma_transfer;
                                target_gamma = target_gamma + gamma_transfer;
                                write_gamma_height(center_idx, center_gamma);
                                write_gamma_height(target_idx, target_gamma);
                            }

                            if (alpha_transfer > 0.0) {
                                center_alpha = clamp(center_alpha - alpha_transfer, 0.0, 1.0);
                                target_alpha = clamp(target_alpha + alpha_transfer, 0.0, 1.0);
                                alpha_grid[center_idx] = center_alpha;
                                alpha_grid[target_idx] = target_alpha;
                            }

                            if (beta_transfer > 0.0) {
                                center_beta = clamp(center_beta - beta_transfer, 0.0, 1.0);
                                target_beta = clamp(target_beta + beta_transfer, 0.0, 1.0);
                                beta_grid[center_idx] = center_beta;
                                beta_grid[target_idx] = target_beta;
                            }
                        }
                    }
                }
            }

            if (PROPELLERS_ENABLED) {
                // Propeller strength scaled by quadratic amplification (enabler effect)
                // Squaring amplification makes thrust grow sharply only when enablers are very close
                // amp=0.5 -> thrust multiplier = 0.25, amp=0.8 -> 0.64, amp=1.0 -> 1.0
                let quadratic_amp = amplification * amplification;
                let propeller_strength = amino_props.thrust_force * 3 * quadratic_amp;
                propeller_thrust_magnitude[i] = propeller_strength; // Store for cost calculation
                let thrust_force = thrust_dir_world * propeller_strength;
                force += thrust_force;
                // Torque from lever arm r_com cross thrust (scaled down to reduce perpetual spinning)
                torque += (r_com.x * thrust_force.y - r_com.y * thrust_force.x) * (6.0 * PROP_TORQUE_COUPLING);

                // INJECT PROPELLER FORCE DIRECTLY INTO FLUID FORCES BUFFER
                // Map world position to fluid grid (FLUID_GRID_SIZE x FLUID_GRID_SIZE)
                // IMPORTANT: do NOT clamp out-of-bounds positions into the edge cells.
                // That produces "wind coming from outside" when propellers are outside the valid world.
                if (world_pos.x >= 0.0 && world_pos.x < f32(SIM_SIZE) && world_pos.y >= 0.0 && world_pos.y < f32(SIM_SIZE)) {
                    let grid_f = f32(FLUID_GRID_SIZE);
                    let max_idx_f = grid_f - 1.0;
                    let fluid_grid_x = u32(clamp(world_pos.x / f32(SIM_SIZE) * grid_f, 0.0, max_idx_f));
                    let fluid_grid_y = u32(clamp(world_pos.y / f32(SIM_SIZE) * grid_f, 0.0, max_idx_f));
                    let fluid_idx = fluid_grid_y * FLUID_GRID_SIZE + fluid_grid_x;

                    // Write thrust force into the shared vec2 buffer.
                    // NOTE: Race condition possible with multiple agents, but the effect is additive so acceptable.
                    let scaled_force = -thrust_force * FLUID_FORCE_SCALE * 0.1;
                    fluid_forces[fluid_idx] = fluid_forces[fluid_idx] + scaled_force;
                }
            }
        }

    // Displacer organ - sweeps material from one side to the other (directional transfer)
        // Only works if agent has enough energy to cover the displacer's consumption cost
        if (amino_props.is_displacer && agent.energy >= amino_props.energy_consumption) {
            // Determine local segment axis using neighbour part
            var segment_dir = vec2<f32>(0.0);
            if (i > 0u) {
                let prev = agents_out[agent_id].body[i-1u].pos;
                segment_dir = part.pos - prev;
            } else if (agents_out[agent_id].body_count > 1u) {
                let next = agents_out[agent_id].body[1u].pos;
                segment_dir = next - part.pos;
            } else {
                // Single-part body fallback
                segment_dir = vec2<f32>(1.0, 0.0);
            }
            let seg_len = length(segment_dir);
            let axis_local = select(segment_dir / seg_len, vec2<f32>(1.0, 0.0), seg_len < 1e-4);
            // Perpendicular (right-hand) to the segment axis is our sweep direction in local space
            // Apply chirality flip to sweep direction
            let sweep_local = normalize(vec2<f32>(-axis_local.y, axis_local.x)) * chirality_flip_physics;
            // Rotate to world space
            let sweep_dir_world = apply_agent_rotation(sweep_local, agent.rotation);

            // Material transfer displacement: move from one side to the other
            let sweep_dir_len = length(sweep_dir_world);
            if (sweep_dir_len > 1e-5) {
                let sweep_dir = sweep_dir_world / sweep_dir_len;
                // Scale sweep by amplification
                let sweep_strength = max(params.prop_wash_strength * amplification, 0.0);

                if (sweep_strength > 0.0) {
                    let clamped_pos = clamp_position(world_pos);
                    let grid_scale = f32(SIM_SIZE) / f32(GRID_SIZE);
                    var gx = i32(clamped_pos.x / grid_scale);
                    var gy = i32(clamped_pos.y / grid_scale);
                    gx = clamp(gx, 0, i32(GRID_SIZE) - 1);
                    gy = clamp(gy, 0, i32(GRID_SIZE) - 1);

                    let center_idx = u32(gy) * GRID_SIZE + u32(gx);
                    let distance = clamp(sweep_strength * 2.0, 1.0, 5.0);
                    let target_world = clamped_pos + sweep_dir * distance * grid_scale;
                    let target_gx = clamp(i32(round(target_world.x / grid_scale)), 0, i32(GRID_SIZE) - 1);
                    let target_gy = clamp(i32(round(target_world.y / grid_scale)), 0, i32(GRID_SIZE) - 1);
                    let target_idx = u32(target_gy) * GRID_SIZE + u32(target_gx);

                    if (target_idx != center_idx) {
                        var center_gamma = read_gamma_height(center_idx);
                        var target_gamma = read_gamma_height(target_idx);
                        var center_alpha = alpha_grid[center_idx];
                        var target_alpha = alpha_grid[target_idx];
                        var center_beta = beta_grid[center_idx];
                        var target_beta = beta_grid[target_idx];

                        let transfer_amount = sweep_strength * 1.0 * part_weight;
                        if (transfer_amount > 0.0) {
                            // Capacities adjusted for 0..1 range
                            let alpha_capacity = max(0.0, 1.0 - target_alpha);
                            let beta_capacity = max(0.0, 1.0 - target_beta);
                            let gamma_capacity = max(0.0, 1.0 - target_gamma);

                            let gamma_transfer = min(min(center_gamma, transfer_amount), gamma_capacity);
                            let alpha_transfer = min(min(center_alpha, transfer_amount), alpha_capacity);
                            let beta_transfer = min(min(center_beta, transfer_amount), beta_capacity);

                            if (gamma_transfer > 0.0) {
                                center_gamma = center_gamma - gamma_transfer;
                                target_gamma = target_gamma + gamma_transfer;
                                write_gamma_height(center_idx, center_gamma);
                                write_gamma_height(target_idx, target_gamma);
                            }

                            if (alpha_transfer > 0.0) {
                                center_alpha = clamp(center_alpha - alpha_transfer, 0.0, 1.0);
                                target_alpha = clamp(target_alpha + alpha_transfer, 0.0, 1.0);
                                alpha_grid[center_idx] = center_alpha;
                                alpha_grid[target_idx] = target_alpha;
                            }

                            if (beta_transfer > 0.0) {
                                center_beta = clamp(center_beta - beta_transfer, 0.0, 1.0);
                                target_beta = clamp(target_beta + beta_transfer, 0.0, 1.0);
                                beta_grid[center_idx] = center_beta;
                                beta_grid[target_idx] = target_beta;
                            }
                        }
                    }
                }
            }
        }

    }

    // Persist torque for inspector debugging
    agents_out[agent_id].torque_debug = torque;

    // Apply global vector force (wind/gravity)
    if (params.vector_force_power > 0.0) {
        let vector_force = vec2<f32>(
            params.vector_force_x * params.vector_force_power,
            params.vector_force_y * params.vector_force_power
        );
        force += vector_force;
    }

    // Apply linear forces - overdamped regime (fluid dynamics at nanoscale)
    // In viscous fluids at low Reynolds number, velocity is directly proportional to force
    // No inertia: velocity = force / drag
    let new_velocity = force / drag_coefficient;

    // Mass-dependent velocity smoothing to prevent jitter in heavy agents on slopes
    // Higher mass = more smoothing (0.95 for mass=0.01, 0.7 for mass=0.1)
    // This filters high-frequency oscillations while preserving directed motion
    let mass_smoothing = clamp(1.0 - (total_mass * 2.5), 0.1, 0.95);
    agent.velocity = mix(agent.velocity, new_velocity, mass_smoothing);

    let v_len = length(agent.velocity);
    if (v_len > VEL_MAX) {
        agent.velocity = agent.velocity * (VEL_MAX / v_len);
    }

    // Apply torque - overdamped angular motion (no angular inertia)
    // In viscous fluids, angular velocity is directly proportional to torque
    // Calculate moment of inertia just for scaling the rotational drag
    // Use segment midpoints for proper rotational inertia calculation
    var moment_of_inertia = 0.0;
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let part = agents_out[agent_id].body[i];
        let base_type = get_base_part_type(part.part_type);
        let props = get_amino_acid_properties(base_type);
        let mass = max(props.mass, 0.01);

        // Calculate segment midpoint
        var segment_start_chain = vec2<f32>(0.0);
        if (i > 0u) {
            segment_start_chain = agents_out[agent_id].body[i - 1u].pos;
        }
        let segment_midpoint_chain = (segment_start_chain + part.pos) * 0.5;
        let segment_midpoint = morphology_origin + segment_midpoint_chain;

        let offset = segment_midpoint - center_of_mass;
        let r_squared = dot(offset, offset);
        moment_of_inertia += mass * r_squared;
    }
    moment_of_inertia = max(moment_of_inertia, 0.01);

    // Overdamped rotation: angular_velocity = torque / rotational_drag
    let rotational_drag = moment_of_inertia * 20.0; // Increased rotational drag for stability
    var angular_velocity = torque / rotational_drag;
    angular_velocity = angular_velocity * ANGULAR_BLEND;
    angular_velocity = clamp(angular_velocity, -ANGVEL_MAX, ANGVEL_MAX);

    // Update rotation
    if (!DISABLE_GLOBAL_ROTATION) {
        agent.rotation += angular_velocity;
    } else {
        agent.rotation = 0.0; // keep zero for disabled global rotation experiment
    }

    // Update position
    // Closed world: clamp at boundaries
    agent.position = clamp_position(agent.position + agent.velocity);

    // ====== UNIFIED ORGAN ACTIVITY LOOP ======
    // Process trail deposition, energy consumption, and feeding in single pass

    // Use the post-morphology capacity written into agents_out this frame
    let capacity = agents_out[agent_id].energy_capacity;

    // poison_resistant_count stored in agent struct during morphology
    // Each poison-resistant organ reduces poison/radiation damage by 50%
    let poison_multiplier = pow(0.5, f32(agents_out[agent_id].poison_resistant_count));

    // Initialize accumulators
    let trail_deposit_strength = 0.08; // Strength of trail deposition (0-1)
    var energy_consumption = params.energy_cost; // base maintenance (can be 0)
    var total_consumed_alpha = 0.0;
    var total_consumed_beta = 0.0;

    // Single loop through all body parts
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let part = agents_out[agent_id].body[i];
        let base_type = get_base_part_type(part.part_type);
        let props = get_amino_acid_properties(base_type);
        let rotated_pos = apply_agent_rotation(part.pos, agent.rotation);
        let world_pos = agent.position + rotated_pos;
        let idx = grid_index(world_pos);

        // Calculate actual mouth speed for mouth organs (distance moved since last frame)
        // Only non-vampire mouths need to track previous position for speed-based absorption
        // Vampire mouths use _pad.x for cooldown tracking instead
        var mouth_speed = 0.0;
        var speed_absorption_multiplier = 1.0;

        if (props.is_mouth && base_type != 33u) {
            let packed_prev = bitcast<u32>(agents_out[agent_id].body[i]._pad.x);
            let prev_pos = unpack_prev_pos(packed_prev);
            let displacement_vec = world_pos - prev_pos;
            let displacement_sq = dot(displacement_vec, displacement_vec);
            // If displacement is very small OR prev_pos is near origin, treat as first frame
            let looks_like_first = (displacement_sq < 0.01) || (dot(prev_pos, prev_pos) < 1.0);
            mouth_speed = select(sqrt(displacement_sq), 0.0, looks_like_first);
            let normalized_mouth_speed = mouth_speed / VEL_MAX;
            speed_absorption_multiplier = exp(-8.0 * normalized_mouth_speed);

            // Update previous world position for next frame (only for non-vampire mouths)
            agents_out[agent_id].body[i]._pad.x = bitcast<f32>(pack_prev_pos(world_pos));
        }

        // Debug output for first agent only
        if (agent_id == 0u && params.debug_mode != 0u && i == 0u) {
            agents_out[agent_id].body[63].pos.x = mouth_speed;
            agents_out[agent_id].body[63].pos.y = speed_absorption_multiplier;
        }

        // 1) Trail deposition: blend agent color + deposit energy trail
        // NOTE: write into trail_grid_inject; the displayed trail_grid is produced by fluid-pass advection.
        let current_trail = trail_grid_inject[idx].xyz;
        let blended = mix(current_trail, agent_color, trail_deposit_strength);

        // Deposit energy trail (unclamped) - scale by agent energy
        let current_energy_trail = trail_grid_inject[idx].w;
        let energy_deposit = agent.energy * trail_deposit_strength * 0.1; // 10% of energy deposited
        let blended_energy = current_energy_trail + energy_deposit;

        trail_grid_inject[idx] = vec4<f32>(clamp(blended, vec3<f32>(0.0), vec3<f32>(1.0)), blended_energy);

        // 2) Energy consumption: calculate costs per organ type
        // DISABLED: Vampire mouths always active, no need to calculate enabler sum
        var global_mouth_activity = 1.0; // Always active
        // if (base_type == 33u) { // Only calculate for vampire mouths
        //     var enabler_sum = 0.0;
        //     var disabler_sum = 0.0;
        //     for (var j = 0u; j < agents_out[agent_id].body_count; j++) {
        //         let check_type = get_base_part_type(agents_out[agent_id].body[j].part_type);
        //         if (check_type == 26u) { enabler_sum += 1.0; }      // Enabler
        //         else if (check_type == 27u) { disabler_sum += 1.0; } // Disabler
        //     }
        //     global_mouth_activity = clamp(enabler_sum - disabler_sum, 0.0, 1.0);
        // }

        // Minimum baseline cost per amino acid (always paid)
        let baseline = params.amino_maintenance_cost;
        // Organ-specific energy costs
        var organ_extra = 0.0;
        if (props.is_mouth) {
            organ_extra = props.energy_consumption;

            // 3) Feeding: mouths consume from alpha/beta grids
            // Get enabler amplification for this mouth
            let amplification = amplification_per_part[i];

            // Consume alpha and beta based on per-amino absorption rates
            // and local availability, scaled by speed (slower = more absorption)
            let available_alpha = alpha_grid[idx];
            let available_beta = beta_grid[idx];

            // Per-amino capture rates let us tune bite size vs. poison uptake
            // Apply speed effects and amplification to the rates themselves
            let alpha_rate = max(props.energy_absorption_rate, 0.0) * speed_absorption_multiplier * amplification;
            let beta_rate  = max(props.beta_absorption_rate, 0.0) * speed_absorption_multiplier * amplification;

            // Total capture budget for this mouth this frame
            let rate_total = alpha_rate + beta_rate;
            if (rate_total > 0.0 && (available_alpha > 0.0 || available_beta > 0.0)) {
                let max_total = rate_total;

                // Weight consumption toward whichever is present and allowed by its rate
                let weighted_alpha = available_alpha * alpha_rate;
                let weighted_beta  = available_beta * beta_rate;
                let weighted_sum   = max(weighted_alpha + weighted_beta, 1e-6);
                let alpha_weight   = weighted_alpha / weighted_sum;
                let beta_weight    = 1.0 - alpha_weight;

                let consumed_alpha = min(available_alpha, max_total * alpha_weight);
                let consumed_beta  = min(available_beta,  max_total * beta_weight);

                // Apply alpha consumption - energy gain now uses base food_power (speed already in consumption)
                if (consumed_alpha > 0.0) {
                    alpha_grid[idx] = clamp(available_alpha - consumed_alpha, 0.0, available_alpha);
                    // Reduce energy gain exponentially per vampiric mouth: 1 mouth = 50%, 2 mouths = 25%, 3 mouths = 12.5%
                    // Calculate vampiric mouth count on the fly
                    var vampiric_count = 0u;
                    for (var j = 0u; j < min(agents_out[agent_id].body_count, MAX_BODY_PARTS); j++) {
                        if (get_base_part_type(agents_out[agent_id].body[j].part_type) == 33u) {
                            vampiric_count += 1u;
                        }
                    }
                    let vampiric_multiplier = pow(0.5, f32(vampiric_count));
                    agent.energy += consumed_alpha * params.food_power * vampiric_multiplier;
                    total_consumed_alpha += consumed_alpha;
                }

                // Apply beta consumption - damage uses poison_power, reduced by poison protection
                if (consumed_beta > 0.0) {
                    beta_grid[idx] = clamp(available_beta - consumed_beta, 0.0, available_beta);
                    agent.energy -= consumed_beta * params.poison_power * poison_multiplier;
                    total_consumed_beta += consumed_beta;
                }
            }
        } else if (props.is_propeller) {
            // Propellers: base cost (always paid) + activity cost (linear with thrust)
            // Since thrust already scales quadratically with amp, cost should scale linearly with thrust
            let base_thrust = props.thrust_force * 3.0; // Max thrust with amp=1
            let thrust_ratio = propeller_thrust_magnitude[i] / base_thrust;
            let activity_cost = props.energy_consumption * thrust_ratio * 1.5;
            organ_extra = props.energy_consumption + activity_cost; // Base + activity
        } else if (props.is_displacer) {
            // Displacers: base cost (always paid) + activity cost (from amplification)
            let amp = amplification_per_part[i];
            let activity_cost = props.energy_consumption * amp * amp * 1.5;
            organ_extra = props.energy_consumption + activity_cost; // Base + activity
        } else if (base_type == 33u) {
            // Vampire mouths: 3x base mouth cost (increased maintenance)
            organ_extra = props.energy_consumption * 3.0;
        } else {
            // Other organs use linear amplification scaling
            let amp = amplification_per_part[i];
            organ_extra = props.energy_consumption * amp * 1.5;
        }
        energy_consumption += baseline + organ_extra;
    }

    // Cap energy by storage capacity after feeding (use post-build capacity)
    // Always clamp to avoid energy > capacity, and to zero when capacity == 0
    agent.energy = clamp(agent.energy, 0.0, max(capacity, 0.0));

    // 3) Maintenance: subtract consumption after feeding
    agent.energy -= energy_consumption;

    // 4) Energy-based death check - death probability inversely proportional to energy
    // High energy = low death chance, low energy = high death chance
    let death_seed = agent_id * 2654435761u + params.random_seed * 1103515245u;
    let death_rnd = f32(hash(death_seed)) / 4294967295.0;

    // Prevent division by zero and NaN: use max(energy, 0.01) as divisor
    // At energy=10: probability / 10 = very low death chance
    // At energy=1: probability / 1 = normal death chance
    // At energy=0.01: probability / 0.01 = 100x higher death chance (starvation)
    let energy_divisor = max(agent.energy, 0.01);
    let energy_adjusted_death_prob = params.death_probability / energy_divisor;

    if (death_rnd < energy_adjusted_death_prob) {
        // Deposit remains: stochastic decomposition into either alpha or beta
        // Fixed total deposit = 1.0 (in 0..1 grid units), spread across parts
        if (body_count > 0u) {
            let total_deposit = 1.0;
            let deposit_per_part = total_deposit / f32(body_count);
            for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
                let part = agents_out[agent_id].body[i];
                let rotated_pos = apply_agent_rotation(part.pos, agent.rotation);
                let world_pos = agent.position + rotated_pos;
                let idx = grid_index(world_pos);

                // Stochastic choice: 50% alpha (nutrient), 50% beta (toxin)
                let part_hash = hash(agent_id * 1000u + i * 100u + params.random_seed);
                let part_rnd = f32(part_hash % 1000u) / 1000.0;

                if (part_rnd < 0.5) {
                    alpha_grid[idx] = min(alpha_grid[idx] + deposit_per_part, 1.0);
                } else {
                    beta_grid[idx] = min(beta_grid[idx] + deposit_per_part, 1.0);
                }
            }
        }

        // If this was the selected agent, transfer selection to a random nearby agent
        if (agent.is_selected == 1u) {
            let transfer_hash = hash(agent_id * 2654435761u + params.random_seed);
            let target_id = transfer_hash % params.agent_count;
            if (target_id < params.agent_count && agents_in[target_id].alive == 1u) {
                agents_out[target_id].is_selected = 1u;
            }
            agent.is_selected = 0u;
        }

        agent.alive = 0u;
        agents_out[agent_id] = agent;
        return;
    }
    // Note: alive counting is handled in the compaction/merge passes

    // ====== SPAWN/REPRODUCTION LOGIC ======
    // Better RNG using hash function with time and agent variation
    let hash_base = (agent_id + params.random_seed) * 747796405u + 2891336453u;
    let hash2 = hash_base ^ (hash_base >> 13u);
    let hash3 = hash2 * 1103515245u;

    // ====== RNA PAIRING REPRODUCTION (probabilistic counter) ======
    // Pairing counter probabilistically increments; reproduce when it reaches gene_length

    // First, calculate the gene length (number of non-X bases) for this agent
    var gene_length = 0u;
    var first_non_x: u32 = GENOME_LENGTH;
    var last_non_x: u32 = 0xFFFFFFFFu;
    for (var bi = 0u; bi < GENOME_LENGTH; bi++) {
        let b = genome_get_base_ascii(agents_out[agent_id].genome, bi);
        if (b != 88u) {
            if (first_non_x == GENOME_LENGTH) { first_non_x = bi; }
            last_non_x = bi;
            gene_length += 1u;
        }
    }

    var pairing_counter = agents_out[agent_id].pairing_counter;
    var energy_invested = 0.0; // Track energy spent on pairing for offspring

    if (gene_length > 0u && pairing_counter < gene_length) {
        // Try to increment the counter based on conditions
        let pos_idx = grid_index(agent.position);
        let beta_concentration = beta_grid[pos_idx];
        // Beta acts as pairing inhibitor
        let radiation_factor = 1.0 / max(1.0 + beta_concentration, 1.0);
        let seed = ((agent_id + 1u) * 747796405u) ^ (pairing_counter * 2891336453u) ^ (params.random_seed * 196613u) ^ pos_idx;
        let rnd = f32(hash(seed)) / 4294967295.0;
        let energy_for_pair = max(agent.energy, 0.0);

        // Probability to increment counter
        // Apply sqrt scaling: diminishing returns for high energy (sqrt(1)=1, sqrt(10)=3.16, sqrt(50)=7.07)
        // This makes low energy more viable while still rewarding energy accumulation
        let energy_scaled = sqrt(energy_for_pair + 1.0);
        // Apply radiation_factor (beta acts as reproductive inhibitor)
        // Poison protection also slows pairing by the same amount
        let pair_p = clamp(params.spawn_probability * energy_scaled * 0.1 * radiation_factor * poison_multiplier, 0.0, 1.0);
        if (rnd < pair_p) {
            // Pairing cost per increment
            let pairing_cost = params.pairing_cost;
            if (agent.energy >= pairing_cost) {
                pairing_counter += 1u;
                agent.energy -= pairing_cost;
                energy_invested += pairing_cost;
            }
        }
    }

    if (pairing_counter >= gene_length && gene_length > 0u) {
        // Attempt reproduction: create complementary genome offspring with mutations
        let current_count = atomicLoad(&spawn_debug_counters[2]);
        if (current_count < params.max_agents) {
            let spawn_index = atomicAdd(&spawn_debug_counters[0], 1u);
            if (spawn_index < 2000u) {
                // Generate hash for offspring randomization
                // CRITICAL: Include agent_id to ensure each parent's offspring gets unique mutations
                let offspring_hash = (hash3 ^ (spawn_index * 0x9e3779b9u) ^ (agent_id * 0x85ebca6bu)) * 1664525u + 1013904223u;

                // Create brand new offspring agent (don't copy parent)
                var offspring: Agent;

                // Random rotation
                offspring.rotation = hash_f32(offspring_hash) * 6.28318530718;

                // Spawn near parent with a small jitter to avoid perfect overlap (prevents extreme repulsion impulses)
                {
                    let jitter_angle = hash_f32(offspring_hash ^ 0xBADC0FFEu) * 6.28318530718;
                    let jitter_dist = 5.0 + hash_f32(offspring_hash ^ 0x1B56C4E9u) * 10.0;
                    let jitter = vec2<f32>(cos(jitter_angle), sin(jitter_angle)) * jitter_dist;
                    offspring.position = clamp_position(agent.position + jitter);
                }
                offspring.velocity = vec2<f32>(0.0);

                // Initialize offspring energy; final value assigned after viability check
                offspring.energy = 0.0;

                offspring.energy_capacity = 0.0; // Will be calculated when morphology builds
                offspring.torque_debug = 0.0;

                // Initialize as alive, will build body on first frame
                offspring.alive = 1u;
                offspring.body_count = 0u; // Forces morphology rebuild
                offspring.pairing_counter = 0u;
                offspring.is_selected = 0u;
                // Lineage and lifecycle
                offspring.generation = agents_out[agent_id].generation + 1u;
                offspring.age = 0u;
                offspring.total_mass = 0.0; // Will be computed after morphology build
                offspring.poison_resistant_count = 0u; // Will be computed after morphology build

                // Child genome: reverse complement (sexual) or direct copy (asexual)
                if (params.asexual_reproduction == 1u) {
                    // Asexual reproduction: direct genome copy (mutations applied later)
                    for (var w = 0u; w < GENOME_WORDS; w++) {
                        offspring.genome[w] = agents_out[agent_id].genome[w];
                    }
                } else {
                    // Sexual reproduction: reverse complementary of parent
                    for (var w = 0u; w < GENOME_WORDS; w++) {
                        let rev_word = genome_revcomp_word(agents_out[agent_id].genome, w);
                        offspring.genome[w] = rev_word;
                    }
                }

                // Sample beta concentration at parent's location to calculate radiation-induced mutation rate
                let parent_idx = grid_index(agent.position);
                let beta_concentration = beta_grid[parent_idx];

                // Beta acts as mutagenic radiation - increases mutation rate with power-of-5 curve
                // This creates clear ecological zones: safe (beta 0-4), moderate (4-7), extreme (7-10)
                // At beta=0: 1x mutations, beta=5: ~2x, beta=7: ~6x, beta=10: ~11x
                // Beta grid is now in 0..1 range; normalize directly
                let beta_normalized = clamp(beta_concentration, 0.0, 1.0);
                // Gentler mutation amplification to reduce genome erosion in high-beta zones
                let mutation_multiplier = 1.0 + pow(beta_normalized, 3.0) * 4.0;
                var effective_mutation_rate = params.mutation_rate * mutation_multiplier;
                // Clamp mutation probability to 1.0 to avoid guaranteed mutation when rate>1
                effective_mutation_rate = min(effective_mutation_rate, 1.0);

                // Determine active gene region (non-'X' bytes) in offspring after reverse complement
                var first_non_x: u32 = GENOME_LENGTH;
                var last_non_x: u32 = 0xFFFFFFFFu;
                for (var bi = 0u; bi < GENOME_LENGTH; bi++) {
                    let b = genome_get_base_ascii(offspring.genome, bi);
                    if (b != 88u) {
                        if (first_non_x == GENOME_LENGTH) { first_non_x = bi; }
                        last_non_x = bi;
                    }
                }
                var active_start: u32 = 0u;
                var active_end: u32 = 0xFFFFFFFFu;
                if (last_non_x != 0xFFFFFFFFu) {
                    active_start = first_non_x;
                    active_end = last_non_x;
                }

                // Optional insertion mutation: with small probability, insert 1..k new random bases at begin/end/middle
                // Then re-center the active gene region within the fixed GENOME_BYTES buffer with 'X' padding
                {
                    let insert_seed = offspring_hash ^ 0xB5297A4Du;
                    let insert_roll = hash_f32(insert_seed);
                    let can_insert = (last_non_x != 0xFFFFFFFFu);
                    if (can_insert && insert_roll < (effective_mutation_rate * 0.20)) {
                        // Extract current active sequence into a local array
                        var seq: array<u32, GENOME_LENGTH>;
                        var L: u32 = 0u;
                        for (var bi = active_start; bi <= active_end; bi++) {
                            if (L < GENOME_LENGTH) {
                                seq[L] = genome_get_base_ascii(offspring.genome, bi);
                                L += 1u;
                            }
                        }
                        // Compute max insert size so we don't exceed GENOME_BYTES
                        let max_ins = select(GENOME_LENGTH - L, 0u, L >= GENOME_LENGTH);
                        if (max_ins > 0u) {
                            let k = 1u + (hash(insert_seed ^ 0x68E31DA4u) % min(5u, max_ins));
                            // Choose insertion position: 0..L
                            let mode = hash(insert_seed ^ 0x1B56C4E9u) % 3u; // 0=begin,1=end,2=middle
                            var pos: u32 = 0u;
                            if (mode == 0u) { pos = 0u; }
                            else if (mode == 1u) { pos = L; }
                            else { pos = hash(insert_seed ^ 0x2C9F85A1u) % (L + 1u); }
                            // Shift right by k from end to pos
                            var j: i32 = i32(L);
                            loop {
                                j = j - 1;
                                if (j < i32(pos)) { break; }
                                seq[u32(j) + k] = seq[u32(j)];
                            }
                            // Fill inserted k bases with random RNA
                            for (var t = 0u; t < k; t++) {
                                let nb = get_random_rna_base(insert_seed ^ (t * 1664525u + 1013904223u));
                                seq[pos + t] = nb;
                            }
                            L = min(GENOME_LENGTH, L + k);
                            // Re-center into a new buffer with 'X' padding
                            var out_bytes: array<u32, GENOME_LENGTH>;
                            for (var t = 0u; t < GENOME_LENGTH; t++) { out_bytes[t] = 88u; }
                            let left_pad = (GENOME_LENGTH - L) / 2u;
                            for (var t = 0u; t < L; t++) {
                                out_bytes[left_pad + t] = seq[t];
                            }
                            // Write back to offspring.genome words
                            for (var w = 0u; w < GENOME_WORDS; w++) {
                                let b0 = out_bytes[w * 4u + 0u] & 0xFFu;
                                let b1 = out_bytes[w * 4u + 1u] & 0xFFu;
                                let b2 = out_bytes[w * 4u + 2u] & 0xFFu;
                                let b3 = out_bytes[w * 4u + 3u] & 0xFFu;
                                let word_val = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
                                offspring.genome[w] = word_val;
                            }
                            // Update active region after insertion
                            active_start = left_pad;
                            active_end = left_pad + L - 1u;
                        }
                    }
                }

                // Optional deletion mutation: mirror insert behavior but remove bases from begin/end/middle
                {
                    let delete_seed = offspring_hash ^ 0xE7037ED1u;
                    let delete_roll = hash_f32(delete_seed);
                    let has_active = (active_end != 0xFFFFFFFFu);
                    if (has_active && delete_roll < (effective_mutation_rate * 0.35)) {
                        // Extract current active sequence into a local array
                        var seq: array<u32, GENOME_LENGTH>;
                        var L: u32 = 0u;
                        for (var bi = active_start; bi <= active_end; bi++) {
                            if (L < GENOME_LENGTH) {
                                seq[L] = genome_get_base_ascii(offspring.genome, bi);
                                L += 1u;
                            }
                        }
                        if (L > MIN_GENE_LENGTH) {
                            let removable = L - MIN_GENE_LENGTH;
                            let max_del = min(5u, removable);
                            if (max_del > 0u) {
                                let k = 1u + (hash(delete_seed ^ 0x68E31DA4u) % max_del);
                                var pos: u32 = 0u;
                                let mode = hash(delete_seed ^ 0x1B56C4E9u) % 3u; // 0=begin,1=end,2=middle
                                if (mode == 0u) {
                                    pos = 0u;
                                } else if (mode == 1u) {
                                    pos = L - k;
                                } else {
                                    pos = hash(delete_seed ^ 0x2C9F85A1u) % (L - k + 1u);
                                }
                                // Shift left to remove k bases starting at pos
                                var j = pos;
                                loop {
                                    if (j + k >= L) { break; }
                                    seq[j] = seq[j + k];
                                    j = j + 1u;
                                }
                                L = L - k;
                                // Re-center into buffer with 'X' padding
                                var out_bytes: array<u32, GENOME_LENGTH>;
                                for (var t = 0u; t < GENOME_LENGTH; t++) { out_bytes[t] = 88u; }
                                let left_pad = (GENOME_LENGTH - L) / 2u;
                                for (var t = 0u; t < L; t++) {
                                    out_bytes[left_pad + t] = seq[t];
                                }
                                for (var w = 0u; w < GENOME_WORDS; w++) {
                                    let b0 = out_bytes[w * 4u + 0u] & 0xFFu;
                                    let b1 = out_bytes[w * 4u + 1u] & 0xFFu;
                                    let b2 = out_bytes[w * 4u + 2u] & 0xFFu;
                                    let b3 = out_bytes[w * 4u + 3u] & 0xFFu;
                                    let word_val = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
                                    offspring.genome[w] = word_val;
                                }
                                active_start = left_pad;
                                active_end = left_pad + L - 1u;
                            }
                        }
                    }
                }

                // Probabilistic point mutations only within active region
                var mutated_count = 0u;
                if (active_end != 0xFFFFFFFFu) {
                    for (var bi = active_start; bi <= active_end; bi++) {
                        let mutation_seed = offspring_hash * (bi + 1u) * 2654435761u;
                        let mutation_chance = hash_f32(mutation_seed);
                        if (mutation_chance < effective_mutation_rate) {
                            let word = bi / 4u;
                            let byte_offset = bi % 4u;
                            let new_base = get_random_rna_base(mutation_seed * 1664525u);
                            let mask = ~(0xFFu << (byte_offset * 8u));
                            let current_word = offspring.genome[word];
                            let updated_word = (current_word & mask) | (new_base << (byte_offset * 8u));
                            offspring.genome[word] = updated_word;
                            mutated_count += 1u;
                        }
                    }
                }

                // New rule: offspring always receives 50% of parent's current energy.
                // Pairing costs are NOT passed to the offspring.
                let inherited_energy = agent.energy * 0.5;
                offspring.energy = inherited_energy;
                agent.energy -= inherited_energy;

                // Mutation diagnostics omitted from Agent; could be added to a dedicated debug buffer if needed

                // Initialize body array to zeros
                for (var bi = 0u; bi < MAX_BODY_PARTS; bi++) {
                    offspring.body[bi].pos = vec2<f32>(0.0);
                    offspring.body[bi].size = 0.0;
                    offspring.body[bi].part_type = 0u;
                    offspring.body[bi].alpha_signal = 0.0;
                    offspring.body[bi].beta_signal = 0.0;
                    offspring.body[bi]._pad.x = bitcast<f32>(0u); // Packed prev_pos will be set on first morphology build
                    offspring.body[bi]._pad = vec2<f32>(0.0);
                }

                new_agents[spawn_index] = offspring;
            }
        }
        // Reset pairing cycle after reproduction
        pairing_counter = 0u;
    }
    agents_out[agent_id].pairing_counter = pairing_counter;

    // Always write selected agent to readback buffer for inspector (even when drawing disabled)
    if (agent.is_selected == 1u) {
        // Publish an unrotated copy for inspector preview
        var unrotated_agent = agents_out[agent_id];
        unrotated_agent.rotation = 0.0;
        // Speed info now stored per-mouth in body[63].pos during loop above (for debugging)
        // Store the calculated gene_length (we already computed it above for reproduction)
        unrotated_agent.gene_length = gene_length;
        // Copy generation/age/total_mass (already in agents_out) unchanged
        selected_agent_buffer[0] = unrotated_agent;
    }

    // Always write output state (simulation must continue even when not drawing)
    agents_out[agent_id].position = agent.position;
    agents_out[agent_id].velocity = agent.velocity;
    agents_out[agent_id].rotation = agent.rotation;
    agents_out[agent_id].energy = agent.energy;
    agents_out[agent_id].alive = agent.alive;
    // Increment age for living agents
    if (agents_out[agent_id].alive == 1u) {
        agents_out[agent_id].age = agents_out[agent_id].age + 1u;
    }
    // Note: body[], genome[], body_count, generation already set correctly in agents_out
}

// ============================================================================
// Helper: draw_line_pixels (for star rendering)
fn draw_line_pixels(p0: vec2<i32>, p1: vec2<i32>, color: vec4<f32>) {
    let dx = p1.x - p0.x;
    let dy = p1.y - p0.y;
    let steps = max(abs(dx), abs(dy));

    for (var s = 0; s <= steps; s++) {
        let t = f32(s) / f32(max(steps, 1));
        let screen_x = i32(mix(f32(p0.x), f32(p1.x), t));
        let screen_y = i32(mix(f32(p0.y), f32(p1.y), t));

        // Check if in visible window bounds
        if (screen_x >= 0 && screen_x < i32(params.window_width) &&
            screen_y >= 0 && screen_y < i32(params.window_height)) {
            let idx = screen_to_grid_index(vec2<i32>(screen_x, screen_y));
            agent_grid[idx] = color;
        }
    }
}

// Helper function to draw a clean line in screen space
fn draw_line(p0: vec2<f32>, p1: vec2<f32>, color: vec4<f32>) {
    let screen_p0 = world_to_screen(p0);
    let screen_p1 = world_to_screen(p1);

    let dx = screen_p1.x - screen_p0.x;
    let dy = screen_p1.y - screen_p0.y;
    let steps = max(abs(dx), abs(dy));

    for (var s = 0; s <= steps; s++) {
        let t = f32(s) / f32(max(steps, 1));
        let screen_x = i32(mix(f32(screen_p0.x), f32(screen_p1.x), t));
        let screen_y = i32(mix(f32(screen_p0.y), f32(screen_p1.y), t));
        let screen_pos = vec2<i32>(screen_x, screen_y);

        // Check if in visible window bounds
        let max_x = select(i32(params.window_width), i32(params.window_width) - i32(INSPECTOR_WIDTH), params.selected_agent_index != 0xFFFFFFFFu);
        if (screen_pos.x >= 0 && screen_pos.x < max_x &&
            screen_pos.y >= 0 && screen_pos.y < i32(params.window_height)) {

            let idx = screen_to_grid_index(screen_pos);
            agent_grid[idx] = color;
        }
    }
}

// ============================================================================
// ENVIRONMENT DIFFUSION & DECAY
// ============================================================================

@compute @workgroup_size(16, 16)
fn diffuse_grids(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if (x >= GRID_SIZE || y >= GRID_SIZE) {
        return;
    }

    let idx = y * GRID_SIZE + x;

    // Fluid-vector convolution (decoupled from dye):
    // Instead of advecting (which tends to look like global translation), we do a
    // directionally-skewed blur kernel similar to the previous slope-based convolution.
    // The fluid force-vector field defines a preferred blur direction and magnitude.
    let current_alpha = alpha_grid[idx];
    let current_beta = beta_grid[idx];

    // Per-channel shift/blur strength.
    let alpha_strength = clamp(params.alpha_blur, 0.0, 1.0);
    let beta_strength = clamp(params.beta_blur, 0.0, 1.0);
    // Repurpose gamma_blur as a simple persistence/decay multiplier (1.0 = no decay).
    let persistence = clamp(params.gamma_blur, 0.0, 1.0);

    // Map env cell coords to fluid grid coords.
    // NOTE: sample_grid_bilinear expects world-space in [0, SIM_SIZE).
    let pos_cell = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5);
    let cell_to_world = f32(SIM_SIZE) / f32(GRID_SIZE);
    let pos_world = pos_cell * cell_to_world;
    let env_to_fluid = f32(FLUID_GRID_SIZE) / f32(GRID_SIZE);
    let fluid_pos = pos_cell * env_to_fluid;
    let fx = clamp(i32(fluid_pos.x), 0, i32(FLUID_GRID_SIZE) - 1);
    let fy = clamp(i32(fluid_pos.y), 0, i32(FLUID_GRID_SIZE) - 1);
    let fluid_idx = u32(fy) * FLUID_GRID_SIZE + u32(fx);

    // Use fluid forces as a direction field (converted to env-cell units).
    // Keep it bounded so it behaves like a convolution offset, not advection.
    let v = fluid_velocity[fluid_idx] / env_to_fluid;
    let vlen = length(v);
    let dir = select(vec2<f32>(0.0, 0.0), v / vlen, vlen > 1e-6);

    // Map magnitude to a small sub-cell offset (in env-cell units).
    // This keeps the effect local and prevents whole-field translation.
    // NOTE: The simulation tick is dt-free; scaling by params.dt makes this effect vanish
    // when dt is 0/unused, so we treat this as a per-tick offset.
    let max_offset = 1.25;
    let base_offset = clamp(vlen * 0.002, 0.0, max_offset);

    // Skewed 3-tap kernel along the direction (a directional blur).
    // Equivalent to a convolution where the "center" is shifted by the vector field.
    let o_a = dir * (base_offset * alpha_strength);
    let o_b = dir * (base_offset * beta_strength);

    // Convolution taps (along +/- direction).
    // We bias weights toward center to keep things stable.
    let w0 = 0.25;
    let w1 = 0.50;
    let w2 = 0.25;

    let a_m = sample_grid_bilinear(pos_world - o_a * cell_to_world, 0u);
    let a_0 = sample_grid_bilinear(pos_world, 0u);
    let a_p = sample_grid_bilinear(pos_world + o_a * cell_to_world, 0u);
    let b_m = sample_grid_bilinear(pos_world - o_b * cell_to_world, 1u);
    let b_0 = sample_grid_bilinear(pos_world, 1u);
    let b_p = sample_grid_bilinear(pos_world + o_b * cell_to_world, 1u);

    let a_blur = clamp(a_m * w0 + a_0 * w1 + a_p * w2, 0.0, 1.0);
    let b_blur = clamp(b_m * w0 + b_0 * w1 + b_p * w2, 0.0, 1.0);

    // Blend toward the convolution result; persistence applies as decay.
    var final_alpha = clamp(mix(current_alpha, a_blur, alpha_strength) * persistence, 0.0, 1.0);
    var final_beta = clamp(mix(current_beta, b_blur, beta_strength) * persistence, 0.0, 1.0);

    // Stochastic rain - randomly add food/poison droplets (saturated drops)
    // Use position and random seed to generate unique random values per cell
    let cell_seed = idx * 2654435761u + params.random_seed;
    let rain_chance = f32(hash(cell_seed)) / 4294967295.0;

    // Uniform alpha rain (food): remove spatial and beta-dependent gradients.
    // Each cell independently receives a saturated rain event with probability alpha_multiplier * 0.05.
    // (Scaling by 0.05 preserves prior expected value semantics.)
    // NOTE: rain_map disabled (binding 16 repurposed for fluid simulation)
    let alpha_rain_factor = 1.0;  // Was: clamp(rain_map[idx].x, 0.0, 1.0);
    let alpha_probability_sat = params.alpha_multiplier * 0.05 * alpha_rain_factor;
    if (rain_chance < alpha_probability_sat) {
        final_alpha = 1.0;  // Saturated drop
    }

    // Uniform beta rain (poison): also no vertical gradient. Probability = beta_multiplier * 0.05.
    let beta_seed = cell_seed * 1103515245u;
    let beta_rain_chance = f32(hash(beta_seed)) / 4294967295.0;
    // NOTE: rain_map disabled (binding 16 repurposed for fluid simulation)
    let beta_rain_factor = 1.0;  // Was: clamp(rain_map[idx].y, 0.0, 1.0);
    let beta_probability_sat = params.beta_multiplier * 0.05 * beta_rain_factor;
    if (beta_rain_chance < beta_probability_sat) {
        final_beta = 1.0;  // Saturated drop
    }

    // Apply the diffused values
    // Apply the updated values in-place.
    alpha_grid[idx] = final_alpha;
    beta_grid[idx] = final_beta;
    // Gamma is not modified by this pass.
}

@compute @workgroup_size(16, 16)
fn compute_gamma_slope(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if (x >= GRID_SIZE || y >= GRID_SIZE) {
        return;
    }

    let ix = i32(x);
    let iy = i32(y);
    let idx = y * GRID_SIZE + x;

    // 8-neighbor gradient with diagonal weighting by sqrt(2)
    // Cardinal neighbors (distance = 1.0)
    let left = read_combined_height(ix - 1, iy);
    let right = read_combined_height(ix + 1, iy);
    let top = read_combined_height(ix, iy - 1);
    let bottom = read_combined_height(ix, iy + 1);

    // Diagonal neighbors (distance = sqrt(2) G?? 1.414)
    let top_left = read_combined_height(ix - 1, iy - 1);
    let top_right = read_combined_height(ix + 1, iy - 1);
    let bottom_left = read_combined_height(ix - 1, iy + 1);
    let bottom_right = read_combined_height(ix + 1, iy + 1);

    // Weight: diagonals contribute with 1/sqrt(2) factor due to longer distance
    // Cardinal X gradient: (right - left) / 2
    // Diagonal X gradient: (top_right + bottom_right - top_left - bottom_left) / (4 * sqrt(2))
    let sqrt2 = 1.41421356237;
    let dx_cardinal = (right - left) * 0.5;
    let dx_diagonal = (top_right + bottom_right - top_left - bottom_left) / (4.0 * sqrt2);

    let dy_cardinal = (bottom - top) * 0.5;
    let dy_diagonal = (bottom_left + bottom_right - top_left - top_right) / (4.0 * sqrt2);

    // Combine cardinal and diagonal contributions
    let dx = dx_cardinal + dx_diagonal;
    let dy = dy_cardinal + dy_diagonal;

    let inv_cell_size = f32(GRID_SIZE) / params.grid_size;
    var gradient = vec2<f32>(dx, dy) * inv_cell_size;

    // Add global vector force (gravity/wind) to slope gradient
    // This makes slope sensors respond to both terrain slope AND gravity direction
    if (params.vector_force_power > 0.0) {
        let gravity_vector = vec2<f32>(
            params.vector_force_x * params.vector_force_power,
            params.vector_force_y * params.vector_force_power
        );
        gradient += gravity_vector;
    }

    write_gamma_slope(idx, gradient);
}

@compute @workgroup_size(16, 16)
fn diffuse_trails(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if (x >= GRID_SIZE || y >= GRID_SIZE) {
        return;
    }

    let idx = y * GRID_SIZE + x;

    // Prepare the injection buffer for dye-style trails:
    // - start from the currently displayed trail_grid
    // - apply optional mild diffusion
    // - apply decay
    // Agents then deposit into trail_grid_inject during process_agents.
    let current = trail_grid[idx];
    let strength = clamp(params.trail_diffusion, 0.0, 1.0);
    let decay = clamp(params.trail_decay, 0.0, 1.0);

    let xi = i32(x);
    let yi = i32(y);
    let x_l = u32(clamp(xi - 1, 0, i32(GRID_SIZE) - 1));
    let x_r = u32(clamp(xi + 1, 0, i32(GRID_SIZE) - 1));
    let y_u = u32(clamp(yi - 1, 0, i32(GRID_SIZE) - 1));
    let y_d = u32(clamp(yi + 1, 0, i32(GRID_SIZE) - 1));

    let l = trail_grid[y * GRID_SIZE + x_l];
    let r = trail_grid[y * GRID_SIZE + x_r];
    let u = trail_grid[y_u * GRID_SIZE + x];
    let d = trail_grid[y_d * GRID_SIZE + x];
    let neighbor_blur = (l + r + u + d) * 0.25;

    let mixed = mix(current, neighbor_blur, strength);
    let faded_rgb = clamp(mixed.xyz * decay, vec3<f32>(0.0), vec3<f32>(1.0));
    let faded_energy = mixed.w * decay;
    trail_grid_inject[idx] = vec4<f32>(faded_rgb, faded_energy);
}

// Helper function to draw a digit (0-9) at a position
fn draw_digit(digit: u32, px: u32, py: u32) -> bool {
    // 5x7 pixel font - returns true if pixel should be lit
    if (px >= 5u || py >= 7u) { return false; }

    var row: u32 = 0u;

    // Digit 0
    if (digit == 0u) {
        if (py == 0u) { row = 0x0Eu; }
        else if (py == 1u) { row = 0x11u; }
        else if (py == 2u) { row = 0x11u; }
        else if (py == 3u) { row = 0x11u; }
        else if (py == 4u) { row = 0x11u; }
        else if (py == 5u) { row = 0x11u; }
        else if (py == 6u) { row = 0x0Eu; }
    }
    // Digit 1
    else if (digit == 1u) {
        if (py == 0u) { row = 0x04u; }
        else if (py == 1u) { row = 0x0Cu; }
        else if (py == 2u) { row = 0x04u; }
        else if (py == 3u) { row = 0x04u; }
        else if (py == 4u) { row = 0x04u; }
        else if (py == 5u) { row = 0x04u; }
        else if (py == 6u) { row = 0x0Eu; }
    }
    // Digit 2
    else if (digit == 2u) {
        if (py == 0u) { row = 0x0Eu; }
        else if (py == 1u) { row = 0x11u; }
        else if (py == 2u) { row = 0x01u; }
        else if (py == 3u) { row = 0x02u; }
        else if (py == 4u) { row = 0x04u; }
        else if (py == 5u) { row = 0x08u; }
        else if (py == 6u) { row = 0x1Fu; }
    }
    // Digit 3
    else if (digit == 3u) {
        if (py == 0u) { row = 0x0Eu; }
        else if (py == 1u) { row = 0x11u; }
        else if (py == 2u) { row = 0x01u; }
        else if (py == 3u) { row = 0x0Eu; }
        else if (py == 4u) { row = 0x01u; }
        else if (py == 5u) { row = 0x11u; }
        else if (py == 6u) { row = 0x0Eu; }
    }
    // Digit 4
    else if (digit == 4u) {
        if (py == 0u) { row = 0x02u; }
        else if (py == 1u) { row = 0x06u; }
        else if (py == 2u) { row = 0x0Au; }
        else if (py == 3u) { row = 0x12u; }
        else if (py == 4u) { row = 0x1Fu; }
        else if (py == 5u) { row = 0x02u; }
        else if (py == 6u) { row = 0x02u; }
    }
    // Digit 5
    else if (digit == 5u) {
        if (py == 0u) { row = 0x1Fu; }
        else if (py == 1u) { row = 0x10u; }
        else if (py == 2u) { row = 0x1Eu; }
        else if (py == 3u) { row = 0x01u; }
        else if (py == 4u) { row = 0x01u; }
        else if (py == 5u) { row = 0x11u; }
        else if (py == 6u) { row = 0x0Eu; }
    }
    // Digit 6
    else if (digit == 6u) {
        if (py == 0u) { row = 0x06u; }
        else if (py == 1u) { row = 0x08u; }
        else if (py == 2u) { row = 0x10u; }
        else if (py == 3u) { row = 0x1Eu; }
        else if (py == 4u) { row = 0x11u; }
        else if (py == 5u) { row = 0x11u; }
        else if (py == 6u) { row = 0x0Eu; }
    }
    // Digit 7
    else if (digit == 7u) {
        if (py == 0u) { row = 0x1Fu; }
        else if (py == 1u) { row = 0x01u; }
        else if (py == 2u) { row = 0x02u; }
        else if (py == 3u) { row = 0x04u; }
        else if (py == 4u) { row = 0x08u; }
        else if (py == 5u) { row = 0x08u; }
        else if (py == 6u) { row = 0x08u; }
    }
    // Digit 8
    else if (digit == 8u) {
        if (py == 0u) { row = 0x0Eu; }
        else if (py == 1u) { row = 0x11u; }
        else if (py == 2u) { row = 0x11u; }
        else if (py == 3u) { row = 0x0Eu; }
        else if (py == 4u) { row = 0x11u; }
        else if (py == 5u) { row = 0x11u; }
        else if (py == 6u) { row = 0x0Eu; }
    }
    // Digit 9
    else if (digit == 9u) {
        if (py == 0u) { row = 0x0Eu; }
        else if (py == 1u) { row = 0x11u; }
        else if (py == 2u) { row = 0x11u; }
        else if (py == 3u) { row = 0x0Fu; }
        else if (py == 4u) { row = 0x01u; }
        else if (py == 5u) { row = 0x02u; }
        else if (py == 6u) { row = 0x0Cu; }
    }

    return ((row >> (4u - px)) & 1u) != 0u;
}

// Helper function to draw a number at a position
fn draw_number(num: u32, base_x: u32, base_y: u32, px: u32, py: u32) -> bool {
    if (num < 10u) {
        return draw_digit(num, px, py);
    } else if (num < 100u) {
        let tens = num / 10u;
        let ones = num % 10u;
        if (px < 5u) {
            return draw_digit(tens, px, py);
        } else if (px >= 6u && px < 11u) {
            return draw_digit(ones, px - 6u, py);
        }
    } else if (num < 1000u) {
        let hundreds = num / 100u;
        let tens = (num / 10u) % 10u;
        let ones = num % 10u;
        if (px < 5u) {
            return draw_digit(hundreds, px, py);
        } else if (px >= 6u && px < 11u) {
            return draw_digit(tens, px - 6u, py);
        } else if (px >= 12u && px < 17u) {
            return draw_digit(ones, px - 12u, py);
        }
    }
    return false;
}

// ============================================================================
// CLEAR VISUAL GRID
// ============================================================================

@compute @workgroup_size(16, 16)
fn clear_visual(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    let visual_idx = y * params.visual_stride + x;

    // Convert screen pixel to world coordinates (accounting for camera and aspect ratio)
    let safe_zoom = max(params.camera_zoom, 0.0001);
    let aspect_ratio = safe_width / safe_height;
    let view_width = params.grid_size / safe_zoom;
    let view_height = view_width / aspect_ratio;
    let cam_min_x = params.camera_pan_x - view_width * 0.5;
    let cam_min_y = params.camera_pan_y - view_height * 0.5;

    // Screen pixel to normalized [0,1]
    let norm_x = f32(x) / safe_width;
    let norm_y = f32(y) / safe_height;

    // Normalized to world coordinates
    let world_x = cam_min_x + norm_x * view_width;
    let world_y = cam_min_y + norm_y * view_height;
    let world_pos = vec2<f32>(world_x, world_y);

    // Check if outside simulation bounds - render black
    let sim_size = f32(SIM_SIZE);
    if (world_x < 0.0 || world_x >= sim_size || world_y < 0.0 || world_y >= sim_size) {
        visual_grid[visual_idx] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Sample environment grids with selected interpolation mode
    var alpha: f32;
    var beta: f32;
    var gamma: f32;

    if (params.grid_interpolation == 2u) {
        // Bicubic (smoothest)
        alpha = clamp(sample_grid_bicubic(world_pos, 0u), 0.0, 1.0);
        beta = clamp(sample_grid_bicubic(world_pos, 1u), 0.0, 1.0);
        gamma = clamp(sample_grid_bicubic(world_pos, 2u), 0.0, 1.0);
    } else if (params.grid_interpolation == 1u) {
        // Bilinear (smooth)
        alpha = clamp(sample_grid_bilinear(world_pos, 0u), 0.0, 1.0);
        beta = clamp(sample_grid_bilinear(world_pos, 1u), 0.0, 1.0);
        gamma = clamp(sample_grid_bilinear(world_pos, 2u), 0.0, 1.0);
    } else {
        // Nearest neighbor (pixelated)
        alpha = clamp(alpha_grid[grid_index(world_pos)], 0.0, 1.0);
        beta = clamp(beta_grid[grid_index(world_pos)], 0.0, 1.0);
        gamma = clamp(read_gamma_height(grid_index(world_pos)), 0.0, 1.0);
    }

    // Hide gamma if requested (treat gamma exactly like alpha/beta, no normalization)
    var gamma_display = gamma;
    if (params.gamma_hidden != 0u) {
        gamma_display = 0.0;
    } else {
        let vis_range = max(params.gamma_vis_max - params.gamma_vis_min, 0.0001);
        gamma_display = clamp((gamma - params.gamma_vis_min) / vis_range, 0.0, 1.0);
    }

    // Start with background color (normalize to 0..1)
    var base_color = vec3<f32>(
        clamp(params.background_color_r, 0.0, 1.0),
        clamp(params.background_color_g, 0.0, 1.0),
        clamp(params.background_color_b, 0.0, 1.0)
    );

    // Slope visualization with optional lighting
    if (params.slope_debug != 0u) {
        let slope = read_gamma_slope(grid_index(world_pos));
        if (params.slope_lighting != 0u) {
            // Lighting mode: compute normal and shade with directional light
            let normal = normalize(vec3<f32>(-slope.x * 10.0, -slope.y * 10.0, 1.0));
            let light_dir = normalize(vec3<f32>(0.5, 0.5, 0.5));
            let diffuse = max(dot(normal, light_dir), 0.0);
            let brightness = (diffuse - 0.5) * 2.0 * params.slope_lighting_strength;
            base_color = vec3<f32>(brightness, brightness, brightness);
        } else {
            // Raw slope mode: red=X, green=Y
            let red = slope.x * 100.0 + 0.5;
            let green = slope.y * 100.0 + 0.5;
            base_color = vec3<f32>(red, green, 0.0);
        }
    } else {
        // New visualization system: composite channels with blend modes

        // Alpha channel
        if (params.alpha_show != 0u) {
            let alpha_color = vec3<f32>(
                clamp(params.alpha_color_r, 0.0, 1.0),
                clamp(params.alpha_color_g, 0.0, 1.0),
                clamp(params.alpha_color_b, 0.0, 1.0)
            );

            // Apply gamma correction to alpha value
            let alpha_corrected = pow(alpha, params.alpha_gamma_adjust);

            if (params.alpha_blend_mode == 0u) {
                // Additive: add channel color scaled by intensity
                base_color = base_color + alpha_color * alpha_corrected;
            } else {
                // Multiply: darken with inverted channel color
                base_color = base_color * mix(vec3<f32>(1.0), vec3<f32>(1.0) - alpha_color, alpha_corrected);
            }
        }

        // Beta channel
        if (params.beta_show != 0u) {
            let beta_color = vec3<f32>(
                clamp(params.beta_color_r, 0.0, 1.0),
                clamp(params.beta_color_g, 0.0, 1.0),
                clamp(params.beta_color_b, 0.0, 1.0)
            );

            // Apply gamma correction to beta value
            let beta_corrected = pow(beta, params.beta_gamma_adjust);

            if (params.beta_blend_mode == 0u) {
                // Additive
                base_color = base_color + beta_color * beta_corrected;
            } else {
                // Multiply with inverted channel
                base_color = base_color * mix(vec3<f32>(1.0), vec3<f32>(1.0) - beta_color, beta_corrected);
            }
        }

        // Gamma channel
        if (params.gamma_show != 0u) {
            let gamma_color = vec3<f32>(
                clamp(params.gamma_color_r, 0.0, 1.0),
                clamp(params.gamma_color_g, 0.0, 1.0),
                clamp(params.gamma_color_b, 0.0, 1.0)
            );

            // Apply gamma correction to gamma_display value
            let gamma_corrected = pow(gamma_display, params.gamma_gamma_adjust);

            if (params.gamma_blend_mode == 0u) {
                // Additive
                base_color = base_color + gamma_color * gamma_corrected;
            } else {
                // Multiply with inverted channel
                base_color = base_color * mix(vec3<f32>(1.0), vec3<f32>(1.0) - gamma_color, gamma_corrected);
            }
        }

        // Slope-based lighting effects (applied after all channels)
        if (params.slope_lighting != 0u) {
            let slope = read_gamma_slope(grid_index(world_pos));
            let normal = normalize(vec3<f32>(-slope.x * 10.0, -slope.y * 10.0, 1.0));
            let light_dir = normalize(vec3<f32>(0.5, 0.5, 0.5));
            let diffuse = max(dot(normal, light_dir), 0.0);
            // Center brightness at 0.5 (neutral), scale by strength
            let brightness = 0.5 + (diffuse - 0.5) * params.slope_lighting_strength;
            // Multiply base color by brightness
            base_color = base_color * brightness;
        }

        // Legacy slope blend modes for backwards compatibility
        if (params.slope_blend_mode != 0u) {
            let slope = read_gamma_slope(grid_index(world_pos));
            let normal = normalize(vec3<f32>(-slope.x * 10.0, -slope.y * 10.0, 1.0));
            let light_dir = normalize(vec3<f32>(params.light_dir_x, params.light_dir_y, params.light_dir_z));
            let light_factor = max(dot(normal, light_dir), 0.0);

            if (params.slope_blend_mode == 1u) {
                // Hard Light
                let blend = vec3<f32>(light_factor);
                base_color = select(
                    2.0 * base_color * blend,
                    vec3<f32>(1.0) - 2.0 * (vec3<f32>(1.0) - base_color) * (vec3<f32>(1.0) - blend),
                    blend > vec3<f32>(0.5)
                );
            } else if (params.slope_blend_mode == 2u) {
                // Soft Light
                let blend = vec3<f32>(light_factor);
                base_color = select(
                    2.0 * base_color * blend + base_color * base_color * (vec3<f32>(1.0) - 2.0 * blend),
                    sqrt(base_color) * (2.0 * blend - vec3<f32>(1.0)) + 2.0 * base_color * (vec3<f32>(1.0) - blend),
                    blend > vec3<f32>(0.5)
                );
            }
        }

        // Legacy gamma_debug mode for backwards compatibility
        if (params.gamma_debug != 0u) {
            base_color = vec3<f32>(gamma_display, gamma_display, gamma_display);
        }
    }

    // Clamp to valid range before writing
    base_color = clamp(base_color, vec3<f32>(0.0), vec3<f32>(1.0));

    // Write base color (motion blur will be applied in separate pass)
    visual_grid[visual_idx] = vec4<f32>(base_color, 1.0);

    // ====== SPATIAL GRID DEBUG VISUALIZATION ======
    // Show which grid cells contain agents (when debug mode is enabled)
    if (params.debug_mode != 0u) {
        let scale = f32(SPATIAL_GRID_SIZE) / f32(SIM_SIZE);
        let grid_x = u32(clamp(world_x * scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));
        let grid_y = u32(clamp(world_y * scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));
        let grid_idx = grid_y * SPATIAL_GRID_SIZE + grid_x;
        let raw_agent_id = atomicLoad(&agent_spatial_grid[grid_idx]);
        // Unmask high bit to get actual agent ID (vampire claim bit)
        let agent_id = raw_agent_id & 0x7FFFFFFFu;
        let is_claimed = (raw_agent_id & 0x80000000u) != 0u;

        // If cell contains an agent, tint it
        if (agent_id != SPATIAL_GRID_EMPTY && agent_id != SPATIAL_GRID_CLAIMED) {
            // Hash the agent ID to get a unique color per agent
            let hash = agent_id * 2654435761u;
            let r = f32((hash >> 0u) & 0xFFu) / 255.0;
            let g = f32((hash >> 8u) & 0xFFu) / 255.0;
            let b = f32((hash >> 16u) & 0xFFu) / 255.0;
            var debug_color = vec3<f32>(r, g, b);

            // If claimed (vampire draining), tint it red
            if (is_claimed) {
                debug_color = mix(debug_color, vec3<f32>(1.0, 0.0, 0.0), 0.6);
            }

            // Blend debug color with base color (50% opacity)
            base_color = mix(base_color, debug_color, 0.5);
            visual_grid[visual_idx] = vec4<f32>(base_color, 1.0);
        }
    }

    // ====== RGB TRAIL OVERLAY ======
    // Sample trail grid and blend onto the visual output
    let trail_color = clamp(trail_grid[grid_index(world_pos)].xyz, vec3<f32>(0.0), vec3<f32>(1.0));

    // Trail-only mode: show just the trail on black background
    if (params.trail_show != 0u) {
        let trail_only = trail_color * clamp(params.trail_opacity, 0.0, 1.0);
        visual_grid[visual_idx] = vec4<f32>(trail_only, 1.0);
    } else {
        // Normal mode: additive blending with opacity control (controlled by trail_opacity parameter)
        let blended_color = clamp(base_color + trail_color * clamp(params.trail_opacity, 0.0, 1.0), vec3<f32>(0.0), vec3<f32>(1.0));
        visual_grid[visual_idx] = vec4<f32>(blended_color, 1.0);
    }
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    // Full screen quad
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0)
    );

    let pos = positions[vid];
    var out: VertexOutput;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = (pos + 1.0) * 0.5;
    return out;
}

@group(0) @binding(0)
var visual_tex: texture_2d<f32>;

@group(0) @binding(1)
var visual_sampler: sampler;

@group(0) @binding(2)
var<uniform> render_params: SimParams;

@group(0) @binding(3)
var<storage, read> agent_grid_render: array<vec4<f32>>;

@group(0) @binding(16)
var<storage, read> fluid_velocity_render: array<vec2<f32>>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample visual texture (already has environment + agents + fluid dye composited)
    let color = textureSample(visual_tex, visual_sampler, in.uv);

    // Fluid visualization is now handled in composite.wgsl compute shader
    // which directly blends dye markers onto visual_grid before rendering

    return vec4<f32>(color.rgb, 1.0);
}

// ============================================================================
// ENVIRONMENT INITIALIZATION
// ============================================================================

@compute @workgroup_size(16, 16)
fn initialize_environment(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= GRID_SIZE || y >= GRID_SIZE) {
        return;
    }

    let idx = y * GRID_SIZE + x;

    // Use constant values for fast startup (can be overridden by loading terrain images)
    alpha_grid[idx] = environment_init.alpha_range.x; // Use minimum alpha value
    beta_grid[idx] = environment_init.beta_range.x;   // Use minimum beta value
    gamma_grid[idx] = 0.0;

    gamma_grid[idx + GAMMA_SLOPE_X_OFFSET] = environment_init.slope_pair.x;
    gamma_grid[idx + GAMMA_SLOPE_Y_OFFSET] = environment_init.slope_pair.y;

    trail_grid[idx] = environment_init.trail_values;
    trail_grid_inject[idx] = environment_init.trail_values;
}

// ============================================================================
// SPAWN/DEATH MANAGEMENT SHADERS
// ============================================================================

// Merge spawned agents into main buffer
@compute @workgroup_size(64)
fn merge_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
    let spawn_id = gid.x;
    let spawn_count = atomicLoad(&spawn_debug_counters[0]);

    if (spawn_id >= spawn_count) {
        return;
    }

    // Append to end of compacted alive array using alive counter as running size
    let target_index = atomicAdd(&spawn_debug_counters[2], 1u);
    if (target_index < params.max_agents) {
        agents_out[target_index] = new_agents[spawn_id];
    }
}

@compute @workgroup_size(64)
fn process_cpu_spawns(@builtin(global_invocation_id) gid: vec3<u32>) {
    let request_id = gid.x;
    if (request_id >= params.cpu_spawn_count) {
        return;
    }

    let request = spawn_requests[request_id];

    let alive_now = atomicLoad(&spawn_debug_counters[2]);
    if (alive_now >= params.max_agents) {
        return;
    }

    var spawn_index: u32 = 0u;
    loop {
        let current_spawn = atomicLoad(&spawn_debug_counters[0]);
        if (current_spawn >= 2000u) {
            return;
        }
        if (alive_now + current_spawn >= params.max_agents) {
            return;
        }
        let result = atomicCompareExchangeWeak(&spawn_debug_counters[0], current_spawn, current_spawn + 1u);
        if (result.exchanged) {
            spawn_index = result.old_value;
            break;
        }
    }

    let world_span = f32(SIM_SIZE);
    var base_seed = request.seed ^ (request_id * 747796405u);
    var genome_seed = request.genome_seed ^ (request_id * 2891336453u);

    var spawn_pos = vec2<f32>(request.position);
    if (spawn_pos.x == 0.0 && spawn_pos.y == 0.0) {
        let rx = hash_f32(base_seed ^ 0xA3C59ACBu);
        let ry = hash_f32(base_seed ^ 0x1B56C4E9u);
        spawn_pos = vec2<f32>(rx * world_span, ry * world_span);
    }

    var rotation = request.rotation;
    if (rotation == 0.0) {
        rotation = hash_f32(base_seed ^ 0xDEADBEEFu) * 6.28318530718;
    }

    var agent: Agent;
    agent.position = clamp_position(spawn_pos);
    agent.velocity = vec2<f32>(0.0);
    agent.rotation = rotation;
    agent.energy = max(request.energy, 0.0);
    agent.energy_capacity = 0.0; // Will be calculated after morphology builds
    agent.torque_debug = 0.0;
    agent.alive = 1u;
    agent.body_count = 0u;
    agent.pairing_counter = 0u;
    agent.is_selected = 0u;
    agent.generation = 0u;
    agent.age = 0u;
    agent.total_mass = 0.0;
    agent.poison_resistant_count = 0u;

    // If flags bit 0 set, use provided genome_override (ASCII bytes)
    if ((request.flags & 1u) != 0u) {
        // Manual unroll to satisfy constant-indexing requirement
        for (var w = 0u; w < GENOME_WORDS; w++) {
            let override_word = genome_read_word(request.genome_override, w);
            agent.genome[w] = override_word;
        }
    } else {
    // Create centered variable-length genome with 'X' padding on both sides
    // Length in [MIN_GENE_LENGTH, GENOME_LENGTH]
        genome_seed = hash(genome_seed ^ base_seed);
    let gene_span = GENOME_LENGTH - MIN_GENE_LENGTH;
    let gene_len = MIN_GENE_LENGTH + (hash(genome_seed) % (gene_span + 1u));
        var bytes: array<u32, GENOME_LENGTH>;
        for (var i = 0u; i < GENOME_LENGTH; i++) { bytes[i] = 88u; } // 'X'
        let left_pad = (GENOME_LENGTH - gene_len) / 2u;
        for (var k = 0u; k < gene_len; k++) {
            genome_seed = hash(genome_seed ^ (k * 1664525u + 1013904223u));
            bytes[left_pad + k] = get_random_rna_base(genome_seed);
        }
        // Write into 16 u32 words (ASCII)
        for (var w = 0u; w < GENOME_WORDS; w++) {
            let b0 = bytes[w * 4u + 0u] & 0xFFu;
            let b1 = bytes[w * 4u + 1u] & 0xFFu;
            let b2 = bytes[w * 4u + 2u] & 0xFFu;
            let b3 = bytes[w * 4u + 3u] & 0xFFu;
            let word_val = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
            agent.genome[w] = word_val;
        }
    }

    for (var i = 0u; i < MAX_BODY_PARTS; i++) {
        agent.body[i].pos = vec2<f32>(0.0);
        agent.body[i].size = 0.0;
        agent.body[i].part_type = 0u;
        agent.body[i].alpha_signal = 0.0;
        agent.body[i].beta_signal = 0.0;
        agent.body[i]._pad.x = bitcast<f32>(0u); // Packed prev_pos will be set on first morphology build
        agent.body[i]._pad = vec2<f32>(0.0);
    }

    new_agents[spawn_index] = agent;
}

@compute @workgroup_size(256)
fn initialize_dead_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.max_agents) {
        return;
    }

    let alive_total = atomicLoad(&spawn_debug_counters[2]);
    if (idx < alive_total) {
        return;
    }

    var agent = agents_out[idx];
    agent.alive = 0u;
    agent.body_count = 0u;
    agent.energy = 0.0;
    agent.velocity = vec2<f32>(0.0);
    agent.pairing_counter = 0u;
    agents_out[idx] = agent;
}

// Compact living agents from input to output, producing a packed array at the front
@compute @workgroup_size(64)
fn compact_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent_id = gid.x;

    if (agent_id >= params.agent_count) {
        return;
    }

    let agent = agents_in[agent_id];
    if (agent.alive != 0u) {
        let idx = atomicAdd(&spawn_debug_counters[2], 1u);
        if (idx < params.max_agents) {
            agents_out[idx] = agent;
        }
    }
}

// Reset spawn counter for next frame
@compute @workgroup_size(1)
fn reset_spawn_counter(@builtin(global_invocation_id) gid: vec3<u32>) {
    atomicStore(&spawn_debug_counters[0], 0u);
}

// ============================================================================
// MAP GENERATION
// ============================================================================

@compute @workgroup_size(16, 16)
fn generate_map(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= GRID_SIZE || y >= GRID_SIZE) {
        return;
    }
    let idx = y * GRID_SIZE + x;

    let mode = environment_init.gen_params.x;
    let gen_type = environment_init.gen_params.y;
    let value = bitcast<f32>(environment_init.gen_params.z);
    let seed = environment_init.gen_params.w;

    var output_value = value;

    if (gen_type == 1u) { // Noise
        // Choose scale based on which channel we're generating
        var scale = environment_init.noise_scale;
        if (mode == 1u) { scale = environment_init.alpha_noise_scale; } // Alpha
        else if (mode == 2u) { scale = environment_init.beta_noise_scale; } // Beta
        else if (mode == 3u) { scale = environment_init.gamma_noise_scale; } // Gamma

        let contrast = environment_init.noise_contrast;
        let octaves = environment_init.noise_octaves;
        let power = environment_init.noise_power;

        let coord = vec2<f32>(f32(x), f32(y)) / f32(GRID_SIZE);
        output_value = layered_noise(coord, seed, octaves, scale, contrast);
        output_value = pow(clamp(output_value, 0.0, 1.0), power);
    }

    if (mode == 1u) {
        alpha_grid[idx] = output_value;
    } else if (mode == 2u) {
        beta_grid[idx] = output_value;
    } else if (mode == 3u) {
        gamma_grid[idx] = output_value;
    }
}

// ============================================================================
// MOTION BLUR (Applied after background render, before agents)
// ============================================================================

@compute @workgroup_size(16, 16)
fn apply_motion_blur(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (params.draw_enabled == 0u || params.follow_mode == 0u) { return; }

    let x = gid.x;
    let y = gid.y;

    let safe_width = max(params.window_width, 1.0);
    let safe_height = max(params.window_height, 1.0);
    let width = u32(safe_width);
    let height = u32(safe_height);

    if (x >= width || y >= height) {
        return;
    }

    let visual_idx = y * params.visual_stride + x;

    // Calculate motion vector from camera movement
    let camera_motion = vec2<f32>(
        params.camera_pan_x - params.prev_camera_pan_x,
        params.camera_pan_y - params.prev_camera_pan_y
    );

    // Scale motion by zoom and normalize by frame time so blur length is frame-rate independent
    let frame_dt = max(params.frame_dt, 0.0001);
    let time_scale = clamp(0.016 / frame_dt, 0.1, 10.0); // Normalize to ~60fps reference
    let motion_scale = params.camera_zoom * time_scale * 0.5; // Halve blur length
    let motion_vector = camera_motion * motion_scale;
    let motion_length = length(motion_vector);

    // Apply motion blur only if camera is moving significantly
    if (motion_length > 0.01) {
        let screen_pos = vec2<f32>(f32(x), f32(y));

        // Get current pixel color
        let base_color = visual_grid[visual_idx].xyz;

        // Take 8 samples in direction opposite to camera motion (backward blur)
        let sample_count = 8;
        var color_sum = base_color;

        // Simple hash for randomization
        let pixel_hash = hash(visual_idx * 73856093u + params.random_seed);

        for (var i = 1; i <= sample_count; i++) {
            // Sample in opposite direction to camera motion (0.0 to 1.0 range)
            let sample_hash = hash(pixel_hash + u32(i) * 1664525u);
            let random_t = f32(sample_hash % 1000u) / 1000.0;

            // Sample opposite to motion vector (negative direction)
            let offset = -motion_vector * random_t;
            let sample_screen_pos = screen_pos + offset;

            // Convert to pixel coordinates with clamping
            let sample_x = u32(clamp(sample_screen_pos.x, 0.0, f32(width - 1u)));
            let sample_y = u32(clamp(sample_screen_pos.y, 0.0, f32(height - 1u)));

            // Sample from visual grid
            let sample_visual_idx = sample_y * params.visual_stride + sample_x;
            if (sample_visual_idx < arrayLength(&visual_grid)) {
                let sample_color = visual_grid[sample_visual_idx].xyz;
                color_sum += sample_color;
            }
        }

        // Average all samples and write back
        let final_color = color_sum / f32(sample_count + 1);
        visual_grid[visual_idx] = vec4<f32>(final_color, 1.0);
    }
}

// ============================================================================
// AGENT SPATIAL GRID - For neighbor detection and collisions
// ============================================================================

// Clear the agent spatial grid (mark all cells as empty)
@compute @workgroup_size(16, 16)
fn clear_agent_spatial_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if (x >= SPATIAL_GRID_SIZE || y >= SPATIAL_GRID_SIZE) {
        return;
    }

    let idx = y * SPATIAL_GRID_SIZE + x;
    atomicStore(&agent_spatial_grid[idx], SPATIAL_GRID_EMPTY);
}

// Populate the agent spatial grid with agent indices
@compute @workgroup_size(256)
fn populate_agent_spatial_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent_id = gid.x;

    if (agent_id >= params.agent_count) {
        return;
    }

    let agent = agents_in[agent_id];

    // Skip dead agents
    if (agent.alive == 0u || agent.energy <= 0.0) {
        return;
    }

    // Convert agent position (in SIM_SIZE space) to grid coordinates (SPATIAL_GRID_SIZE resolution)
    let scale = f32(SPATIAL_GRID_SIZE) / f32(SIM_SIZE);
    let grid_x = u32(clamp(agent.position.x * scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));
    let grid_y = u32(clamp(agent.position.y * scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));
    let primary_idx = grid_y * SPATIAL_GRID_SIZE + grid_x;

    // Try to claim the primary cell atomically
    let primary_result = atomicCompareExchangeWeak(&agent_spatial_grid[primary_idx], SPATIAL_GRID_EMPTY, agent_id);

    if (!primary_result.exchanged) {
        // Primary cell is occupied - search for nearest empty cell in a spiral pattern
        // This ensures all agents are findable even in crowded areas
        var found = false;

        // Search in expanding square rings up to radius 5 (covers 11x11 area = 121 cells)
        for (var radius = 1u; radius <= 5u && !found; radius++) {
            // Top and bottom edges of the square
            for (var dx: i32 = -i32(radius); dx <= i32(radius) && !found; dx++) {
                // Top edge
                let check_x_top = i32(grid_x) + dx;
                let check_y_top = i32(grid_y) - i32(radius);
                if (check_x_top >= 0 && check_x_top < i32(SPATIAL_GRID_SIZE) &&
                    check_y_top >= 0 && check_y_top < i32(SPATIAL_GRID_SIZE)) {
                    let idx = u32(check_y_top) * SPATIAL_GRID_SIZE + u32(check_x_top);
                    let result = atomicCompareExchangeWeak(&agent_spatial_grid[idx], SPATIAL_GRID_EMPTY, agent_id);
                    if (result.exchanged) {
                        found = true;
                    }
                }

                // Bottom edge (skip if radius == 0 to avoid duplicate)
                if (!found && radius > 0u) {
                    let check_x_bot = i32(grid_x) + dx;
                    let check_y_bot = i32(grid_y) + i32(radius);
                    if (check_x_bot >= 0 && check_x_bot < i32(SPATIAL_GRID_SIZE) &&
                        check_y_bot >= 0 && check_y_bot < i32(SPATIAL_GRID_SIZE)) {
                        let idx = u32(check_y_bot) * SPATIAL_GRID_SIZE + u32(check_x_bot);
                        let result = atomicCompareExchangeWeak(&agent_spatial_grid[idx], SPATIAL_GRID_EMPTY, agent_id);
                        if (result.exchanged) {
                            found = true;
                        }
                    }
                }
            }

            // Left and right edges (excluding corners already covered)
            for (var dy: i32 = -i32(radius) + 1; dy < i32(radius) && !found; dy++) {
                // Left edge
                let check_x_left = i32(grid_x) - i32(radius);
                let check_y_left = i32(grid_y) + dy;
                if (check_x_left >= 0 && check_x_left < i32(SPATIAL_GRID_SIZE) &&
                    check_y_left >= 0 && check_y_left < i32(SPATIAL_GRID_SIZE)) {
                    let idx = u32(check_y_left) * SPATIAL_GRID_SIZE + u32(check_x_left);
                    let result = atomicCompareExchangeWeak(&agent_spatial_grid[idx], SPATIAL_GRID_EMPTY, agent_id);
                    if (result.exchanged) {
                        found = true;
                    }
                }

                // Right edge
                if (!found) {
                    let check_x_right = i32(grid_x) + i32(radius);
                    let check_y_right = i32(grid_y) + dy;
                    if (check_x_right >= 0 && check_x_right < i32(SPATIAL_GRID_SIZE) &&
                        check_y_right >= 0 && check_y_right < i32(SPATIAL_GRID_SIZE)) {
                        let idx = u32(check_y_right) * SPATIAL_GRID_SIZE + u32(check_x_right);
                        let result = atomicCompareExchangeWeak(&agent_spatial_grid[idx], SPATIAL_GRID_EMPTY, agent_id);
                        if (result.exchanged) {
                            found = true;
                        }
                    }
                }
            }
        }

        // If still not found after searching 5 rings, agent won't be in spatial grid this frame
        // This is acceptable as it will retry next frame - prevents infinite loops
    }
}

