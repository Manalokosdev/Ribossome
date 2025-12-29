// Vampire mouths drain energy from nearby living agents
// ============================================================================

// Helper: add a force into the low-res fluid force grid using bilinear splatting.
// This reduces axis-aligned artifacts vs. writing into a single nearest cell.
fn atomic_add_f32(idx: u32, v: f32) {
    // CAS loop: atomic add for f32 stored as u32 bits.
    // NOTE: naga currently disallows passing storage pointers into functions,
    // so we address the global buffer by index here.
    loop {
        let old_bits = atomicLoad(&fluid_forces[idx]);
        let old_val = bitcast<f32>(old_bits);
        let new_bits = bitcast<u32>(old_val + v);
        let res = atomicCompareExchangeWeak(&fluid_forces[idx], old_bits, new_bits);
        if (res.exchanged) {
            break;
        }
    }
}

fn add_fluid_force_splat(pos_world: vec2<f32>, f: vec2<f32>) {
    if (pos_world.x < 0.0 || pos_world.x >= f32(SIM_SIZE) || pos_world.y < 0.0 || pos_world.y >= f32(SIM_SIZE)) {
        return;
    }

    let ws = f32(SIM_SIZE);
    let grid_f = f32(FLUID_GRID_SIZE);

    // Continuous coords in fluid-cell space.
    let gx = (pos_world.x / ws) * grid_f;
    let gy = (pos_world.y / ws) * grid_f;

    // Match the fluid shader convention: cell centers at (x+0.5, y+0.5).
    let x = gx - 0.5;
    let y = gy - 0.5;

    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = fract(x);
    let fy = fract(y);

    let ix0 = clamp(x0, 0, i32(FLUID_GRID_SIZE) - 1);
    let iy0 = clamp(y0, 0, i32(FLUID_GRID_SIZE) - 1);
    let ix1 = clamp(x1, 0, i32(FLUID_GRID_SIZE) - 1);
    let iy1 = clamp(y1, 0, i32(FLUID_GRID_SIZE) - 1);

    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    let idx00 = u32(iy0) * FLUID_GRID_SIZE + u32(ix0);
    let idx10 = u32(iy0) * FLUID_GRID_SIZE + u32(ix1);
    let idx01 = u32(iy1) * FLUID_GRID_SIZE + u32(ix0);
    let idx11 = u32(iy1) * FLUID_GRID_SIZE + u32(ix1);

    // Atomic accumulate into interleaved (x,y) entries per cell.
    let base00 = idx00 * 2u;
    let base10 = idx10 * 2u;
    let base01 = idx01 * 2u;
    let base11 = idx11 * 2u;

    atomic_add_f32(base00 + 0u, f.x * w00);
    atomic_add_f32(base00 + 1u, f.y * w00);
    atomic_add_f32(base10 + 0u, f.x * w10);
    atomic_add_f32(base10 + 1u, f.y * w10);
    atomic_add_f32(base01 + 0u, f.x * w01);
    atomic_add_f32(base01 + 1u, f.y * w01);
    atomic_add_f32(base11 + 0u, f.x * w11);
    atomic_add_f32(base11 + 1u, f.y * w11);
}

fn is_bad_f32(x: f32) -> bool {
    // WGSL portability: detect NaN via (x != x). Detect Inf/overflow via a large threshold.
    return (x != x) || (abs(x) > 1e20);
}

fn sanitize_f32(x: f32) -> f32 {
    return select(x, 0.0, is_bad_f32(x));
}

// Experiment toggle:
// If false, propellers do NOT apply thrust/torque directly to agent movement,
// and instead only influence motion indirectly through the fluid.
const PROPELLERS_APPLY_DIRECT_FORCE: bool = true;

// Experiment toggle:
// Inject forces into the fluid based on body-part motion caused by morphology updates.
// This enables propulsion-by-undulation once fluids are enabled, even with propellers disabled.
const MORPHOLOGY_INJECT_FLUID_FORCE: bool = true;
// Strength of morphology-induced fluid coupling (keep small; morphology updates can move parts a lot per frame).
const MORPHOLOGY_FLUID_COUPLING: f32 = 1.0;
// Clamp to avoid huge impulses when morphology jumps between frames.
const MORPHOLOGY_MAX_WORLD_DELTA: f32 = 2.0;
const MORPHOLOGY_MAX_WORLD_VEL: f32 = 20.0;

// Option B experiment:
// Apply a direct reaction force/torque to the agent from morphology-driven part motion.
// This is a coarse "resistive force" swimmer model (anisotropic drag) that remains stable
// even when the fluid grid is low-res.
const MORPHOLOGY_SWIM_ENABLED: bool = true;
// Overall strength multiplier. Uses the existing UI knob `prop_wash_strength_fluid` as a temporary control.
const MORPHOLOGY_SWIM_COUPLING: f32 = 1.0;
// Drag anisotropy: perpendicular drag > tangential drag.
const MORPHOLOGY_SWIM_C_PAR: f32 = 0.15;
const MORPHOLOGY_SWIM_C_PERP: f32 = 0.6;
// Clamp per-tick contribution to avoid explosive drift.
const MORPHOLOGY_SWIM_MAX_FRAME_VEL: f32 = 3.0;

@compute @workgroup_size(256)
fn drain_energy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent_id = gid.x;
    if (agent_id >= params.agent_count) {
        return;
    }

    // Spatial grid uses an epoch stamp so we don't have to clear it every frame.
    // Store stamp = (epoch + 1) so a zero-initialized buffer is always treated as empty.
    let current_stamp = params.epoch + 1u;

    // NOTE: This pass is intended to run after process_agents so that it can
    // operate on the fully materialized per-agent state in agents_out.
    let agent = agents_out[agent_id];

    // Skip dead agents
    if (agent.alive == 0u || agent.energy <= 0.0) {
        return;
    }

    // Newborn grace: don't attack or get attacked in the first few frames after spawning
    if (agent.age < VAMPIRE_NEWBORN_GRACE_FRAMES) {
        return;
    }

    // Collect vampire mouth organ indices (type 33) and whether this agent has any disablers (type 26).
    // This lets us iterate only vampire mouths later, instead of scanning every body part.
    let body_count = min(agent.body_count, MAX_BODY_PARTS);
    var vampire_mouth_indices: array<u32, MAX_BODY_PARTS>;
    var vampire_mouth_count = 0u;
    var has_disabler = false;
    for (var i = 0u; i < body_count; i++) {
        let base_type = get_base_part_type(agents_out[agent_id].body[i].part_type);
        if (base_type == 33u) {
            vampire_mouth_indices[vampire_mouth_count] = i;
            vampire_mouth_count += 1u;
        } else if (base_type == 26u) {
            has_disabler = true;
        }
    }
    if (vampire_mouth_count == 0u) {
        return;
    }

    // Vampire reach and corresponding grid search radius.
    // NOTE: cap to 10 (previous behavior) to avoid pathological loop sizes.
    let max_drain_distance = 50.0; // Fixed 50 unit radius
    let scale = f32(SPATIAL_GRID_SIZE) / f32(SIM_SIZE);
    let cell_size = f32(SIM_SIZE) / f32(SPATIAL_GRID_SIZE);
    let grid_radius = clamp(i32(ceil(max_drain_distance / cell_size)), 1, 10);

    // Process each vampire mouth organ (F=4u, G=5u, H=6u)
    var total_energy_gained = 0.0;

    for (var mi = 0u; mi < vampire_mouth_count; mi++) {
        let i = vampire_mouth_indices[mi];
        let part = agents_out[agent_id].body[i];
        let mouth_local_pos = part.pos;

        // World position for this mouth. Keep this computed regardless of enable state so
        // we can always update the tracking data.
        let rotated_pos = apply_agent_rotation(mouth_local_pos, agent.rotation);
        let mouth_world_pos = agent.position + rotated_pos;

            // Cooldown timer (stored in _pad.x) should tick regardless of enable state
            var current_cooldown = agents_out[agent_id].body[i]._pad.x;
            if (current_cooldown > 0.0) {
                current_cooldown -= 1.0;
                agents_out[agent_id].body[i]._pad.x = current_cooldown;
            }

            // Disabler gating:
            // Vampire mouths act by default, but are suppressed when "inhibitor/enabler" organs
            // (type 26) are placed nearby on the same agent body.
            var block = 0.0;
            if (has_disabler) {
                for (var j = 0u; j < body_count; j++) {
                    let check_type = get_base_part_type(agents_out[agent_id].body[j].part_type);
                    if (check_type == 26u) {  // "Enabler" used as disabler for vampire mouths
                        let disabler_pos = agents_out[agent_id].body[j].pos;
                        let d = length(mouth_local_pos - disabler_pos);
                        if (d < 20.0) {
                            block += max(0.0, 1.0 - d / 20.0);
                        }
                    }
                }
            }
            block = min(block, 1.0);
            let quadratic_block = block * block;
            let mouth_activity = 1.0 - quadratic_block;

            // Only work if not fully suppressed
            if (mouth_activity > 0.0001) {
                // Mouth-centered spiral search over spatial grid.
                // We break early on the first in-range victim to save compute.
                var closest_victim_id = 0xFFFFFFFFu;

                let mouth_grid_x = u32(clamp(mouth_world_pos.x * scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));
                let mouth_grid_y = u32(clamp(mouth_world_pos.y * scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));

                var found_victim = false;
                for (var radius: i32 = 0; radius <= grid_radius; radius++) {
                    if (found_victim) {
                        break;
                    }
                    if (radius == 0) {
                        // Single center cell
                        let check_x = i32(mouth_grid_x);
                        let check_y = i32(mouth_grid_y);
                        let check_cell = u32(check_y) * SPATIAL_GRID_SIZE + u32(check_x);
                        let stamp = atomicLoad(&agent_spatial_grid[spatial_epoch_index(check_cell)]);
                        if (stamp == current_stamp) {
                            let raw_neighbor_id = atomicLoad(&agent_spatial_grid[spatial_id_index(check_cell)]);
                            let neighbor_id = raw_neighbor_id & 0x7FFFFFFFu;
                            if (neighbor_id != SPATIAL_GRID_EMPTY && neighbor_id != SPATIAL_GRID_CLAIMED && neighbor_id != agent_id) {
                                let neighbor = agents_out[neighbor_id];
                                if (neighbor.alive != 0u && neighbor.energy > 0.0 && neighbor.age >= VAMPIRE_NEWBORN_GRACE_FRAMES) {
                                    let dist = length(mouth_world_pos - neighbor.position);
                                    if (dist < max_drain_distance) {
                                        closest_victim_id = neighbor_id;
                                        found_victim = true;
                                    }
                                }
                            }
                        }
                    } else {
                        // Top and bottom edges
                        for (var dx: i32 = -radius; dx <= radius; dx++) {
                            if (found_victim) {
                                break;
                            }
                            // Top
                            var check_x = i32(mouth_grid_x) + dx;
                            var check_y = i32(mouth_grid_y) - radius;
                            if (check_x >= 0 && check_x < i32(SPATIAL_GRID_SIZE) && check_y >= 0 && check_y < i32(SPATIAL_GRID_SIZE)) {
                                let check_cell = u32(check_y) * SPATIAL_GRID_SIZE + u32(check_x);
                                let stamp = atomicLoad(&agent_spatial_grid[spatial_epoch_index(check_cell)]);
                                if (stamp == current_stamp) {
                                    let raw_neighbor_id = atomicLoad(&agent_spatial_grid[spatial_id_index(check_cell)]);
                                    let neighbor_id = raw_neighbor_id & 0x7FFFFFFFu;
                                    if (neighbor_id != SPATIAL_GRID_EMPTY && neighbor_id != SPATIAL_GRID_CLAIMED && neighbor_id != agent_id) {
                                        let neighbor = agents_out[neighbor_id];
                                        if (neighbor.alive != 0u && neighbor.energy > 0.0 && neighbor.age >= VAMPIRE_NEWBORN_GRACE_FRAMES) {
                                            let dist = length(mouth_world_pos - neighbor.position);
                                            if (dist < max_drain_distance) {
                                                closest_victim_id = neighbor_id;
                                                found_victim = true;
                                            }
                                        }
                                    }
                                }
                            }

                            // Bottom
                            if (!found_victim) {
                                check_x = i32(mouth_grid_x) + dx;
                                check_y = i32(mouth_grid_y) + radius;
                                if (check_x >= 0 && check_x < i32(SPATIAL_GRID_SIZE) && check_y >= 0 && check_y < i32(SPATIAL_GRID_SIZE)) {
                                    let check_cell = u32(check_y) * SPATIAL_GRID_SIZE + u32(check_x);
                                    let stamp = atomicLoad(&agent_spatial_grid[spatial_epoch_index(check_cell)]);
                                    if (stamp == current_stamp) {
                                        let raw_neighbor_id = atomicLoad(&agent_spatial_grid[spatial_id_index(check_cell)]);
                                        let neighbor_id = raw_neighbor_id & 0x7FFFFFFFu;
                                        if (neighbor_id != SPATIAL_GRID_EMPTY && neighbor_id != SPATIAL_GRID_CLAIMED && neighbor_id != agent_id) {
                                            let neighbor = agents_out[neighbor_id];
                                            if (neighbor.alive != 0u && neighbor.energy > 0.0 && neighbor.age >= VAMPIRE_NEWBORN_GRACE_FRAMES) {
                                                let dist = length(mouth_world_pos - neighbor.position);
                                                if (dist < max_drain_distance) {
                                                    closest_victim_id = neighbor_id;
                                                    found_victim = true;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Left and right edges (excluding corners already covered)
                        for (var dy: i32 = -radius + 1; dy <= radius - 1; dy++) {
                            if (found_victim) {
                                break;
                            }
                            // Left
                            var check_x = i32(mouth_grid_x) - radius;
                            var check_y = i32(mouth_grid_y) + dy;
                            if (check_x >= 0 && check_x < i32(SPATIAL_GRID_SIZE) && check_y >= 0 && check_y < i32(SPATIAL_GRID_SIZE)) {
                                let check_cell = u32(check_y) * SPATIAL_GRID_SIZE + u32(check_x);
                                let stamp = atomicLoad(&agent_spatial_grid[spatial_epoch_index(check_cell)]);
                                if (stamp == current_stamp) {
                                    let raw_neighbor_id = atomicLoad(&agent_spatial_grid[spatial_id_index(check_cell)]);
                                    let neighbor_id = raw_neighbor_id & 0x7FFFFFFFu;
                                    if (neighbor_id != SPATIAL_GRID_EMPTY && neighbor_id != SPATIAL_GRID_CLAIMED && neighbor_id != agent_id) {
                                        let neighbor = agents_out[neighbor_id];
                                        if (neighbor.alive != 0u && neighbor.energy > 0.0 && neighbor.age >= VAMPIRE_NEWBORN_GRACE_FRAMES) {
                                            let dist = length(mouth_world_pos - neighbor.position);
                                            if (dist < max_drain_distance) {
                                                closest_victim_id = neighbor_id;
                                                found_victim = true;
                                            }
                                        }
                                    }
                                }
                            }

                            // Right
                            if (!found_victim) {
                                check_x = i32(mouth_grid_x) + radius;
                                check_y = i32(mouth_grid_y) + dy;
                                if (check_x >= 0 && check_x < i32(SPATIAL_GRID_SIZE) && check_y >= 0 && check_y < i32(SPATIAL_GRID_SIZE)) {
                                    let check_cell = u32(check_y) * SPATIAL_GRID_SIZE + u32(check_x);
                                    let stamp = atomicLoad(&agent_spatial_grid[spatial_epoch_index(check_cell)]);
                                    if (stamp == current_stamp) {
                                        let raw_neighbor_id = atomicLoad(&agent_spatial_grid[spatial_id_index(check_cell)]);
                                        let neighbor_id = raw_neighbor_id & 0x7FFFFFFFu;
                                        if (neighbor_id != SPATIAL_GRID_EMPTY && neighbor_id != SPATIAL_GRID_CLAIMED && neighbor_id != agent_id) {
                                            let neighbor = agents_out[neighbor_id];
                                            if (neighbor.alive != 0u && neighbor.energy > 0.0 && neighbor.age >= VAMPIRE_NEWBORN_GRACE_FRAMES) {
                                                let dist = length(mouth_world_pos - neighbor.position);
                                                if (dist < max_drain_distance) {
                                                    closest_victim_id = neighbor_id;
                                                    found_victim = true;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if (!found_victim) {
                    agents_out[agent_id].body[i]._pad.y = 0.0;
                }

                // Drain from closest victim only
                if (closest_victim_id != 0xFFFFFFFFu) {
                    // Try to claim the victim's spatial grid cell atomically
                    // This prevents multiple DIFFERENT vampires from draining the same victim simultaneously
                    // But allows the same vampire to drain with multiple mouths
                    let victim_pos = agents_out[closest_victim_id].position;
                    let victim_scale = f32(SPATIAL_GRID_SIZE) / f32(SIM_SIZE);
                    let victim_grid_x = u32(clamp(victim_pos.x * victim_scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));
                    let victim_grid_y = u32(clamp(victim_pos.y * victim_scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));
                    let victim_cell = victim_grid_y * SPATIAL_GRID_SIZE + victim_grid_x;

                    // Atomic claim: mark victim with high bit to indicate it's being drained this frame.
                    // The high bit preserves the victim ID for physics (unmask with & 0x7FFFFFFF).
                    // IMPORTANT: this claim must be exclusive across vampires to avoid
                    // cross-invocation RMW races on victim energy.
                    let victim_stamp = atomicLoad(&agent_spatial_grid[spatial_epoch_index(victim_cell)]);
                    var can_drain = false;
                    if (victim_stamp == current_stamp) {
                        let current_cell = atomicLoad(&agent_spatial_grid[spatial_id_index(victim_cell)]);

                        // Check if cell contains the victim (with or without high bit)
                        let cell_agent_id = current_cell & 0x7FFFFFFFu;
                        let is_claimed = (current_cell & 0x80000000u) != 0u;

                        if (cell_agent_id == closest_victim_id && !is_claimed) {
                            // Victim is unclaimed - try to mark with high bit
                            let claimed_victim_id = closest_victim_id | 0x80000000u;
                            let claim_result = atomicCompareExchangeWeak(&agent_spatial_grid[spatial_id_index(victim_cell)], closest_victim_id, claimed_victim_id);
                            can_drain = claim_result.exchanged;
                        }
                    }

                    // Only proceed if we can drain this victim AND cooldown is ready
                    if (can_drain && current_cooldown <= 0.0) {
                        // Vampire drain scales down with disabler suppression AND mouth movement speed
                        let victim_energy = agents_out[closest_victim_id].energy;

                        if (victim_energy > 0.0001) {
                            // Linear speed-based absorption reduction using RELATIVE victim speed.
                            // IMPORTANT: don't use mouth-tip delta here; rotation can make the tip move far
                            // faster than VEL_MAX even when the agent is behaving normally, which would
                            // zero-out vampire drains permanently.
                            let agent_vel = agent.velocity;
                            let victim_vel = agents_out[closest_victim_id].velocity;
                            let rel_speed = length(agent_vel - victim_vel);
                            let normalized_rel_speed = min(rel_speed / VEL_MAX, 1.0);
                            let speed_multiplier = clamp(1.0 - normalized_rel_speed, 0.0, 1.0);

                            // Absorb up to 50% of victim's energy (reduced by mouth speed and disabler).
                            // Poison Resistance (type 29) also protects against vampire drain, using the
                            // same per-organ 50% multiplier as poison damage.
                            let vampire_protection = pow(0.5, f32(agents_out[closest_victim_id].poison_resistant_count));
                            let absorbed_energy = victim_energy * 0.5 * mouth_activity * speed_multiplier * vampire_protection;

                            // Victim loses 2x the absorbed energy.
                            let energy_damage = absorbed_energy * 2.0;
                            agents_out[closest_victim_id].energy = max(0.0, victim_energy - energy_damage);

                            total_energy_gained += absorbed_energy;

                            // Store absorbed amount in _pad.y for visualization
                            agents_out[agent_id].body[i]._pad.y = absorbed_energy;

                            // Set cooldown timer
                            agents_out[agent_id].body[i]._pad.x = VAMPIRE_MOUTH_COOLDOWN;
                        } else {
                            agents_out[agent_id].body[i]._pad.y = 0.0;
                        }
                    } else {
                        agents_out[agent_id].body[i]._pad.y = 0.0;
                    }
                } else {
                    agents_out[agent_id].body[i]._pad.y = 0.0;
                }
            } else {
                agents_out[agent_id].body[i]._pad.y = 0.0;
            }

            // Store current position for next frame's movement calculation
            agents_out[agent_id].body[i].data = pack_position_to_f32(mouth_world_pos, f32(SIM_SIZE));
    }

    // Add gained energy to this agent
    if (total_energy_gained > 0.0) {
        agents_out[agent_id].energy += total_energy_gained;
    }
}

// ============================================================================

@compute @workgroup_size(256)
fn process_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent_id = gid.x;
    if (agent_id >= params.agent_count) {
        return;
    }

    // Spatial grid uses an epoch stamp so we don't have to clear it every frame.
    let current_stamp = params.epoch + 1u;

    // IMPORTANT: Avoid copying the full Agent payload into a local variable.
    // In practice this can pull in the entire body array and become a bandwidth hotspot.
    let agent_alive = agents_in[agent_id].alive;

    // Skip dead agents
    if (agent_alive == 0u) {
        // Avoid copying the full Agent payload for dead entries.
        agents_out[agent_id].alive = 0u;
        agents_out[agent_id].body_count = 0u;
        agents_out[agent_id].energy = 0.0;
        agents_out[agent_id].velocity = vec2<f32>(0.0);
        agents_out[agent_id].pairing_counter = 0u;
        agents_out[agent_id].is_selected = 0u;
        return;
    }

    // Pull only the scalar fields we use frequently into locals.
    let agent_position = agents_in[agent_id].position;
    let agent_velocity = agents_in[agent_id].velocity;
    let agent_rotation = agents_in[agent_id].rotation;
    let agent_energy = agents_in[agent_id].energy;
    let agent_energy_capacity = agents_in[agent_id].energy_capacity;
    let agent_torque_debug = agents_in[agent_id].torque_debug;
    let agent_morphology_origin = agents_in[agent_id].morphology_origin;
    let agent_body_count = agents_in[agent_id].body_count;
    let agent_pairing_counter = agents_in[agent_id].pairing_counter;
    let agent_is_selected = agents_in[agent_id].is_selected;
    let agent_generation = agents_in[agent_id].generation;
    let agent_age = agents_in[agent_id].age;
    let agent_total_mass = agents_in[agent_id].total_mass;
    let agent_poison_resistant_count = agents_in[agent_id].poison_resistant_count;
    let agent_gene_length = agents_in[agent_id].gene_length;
    let agent_genome_offset = agents_in[agent_id].genome_offset;
    let agent_genome_packed = agents_in[agent_id].genome_packed;

    // Working state for this invocation.
    var agent_pos = agent_position;
    var agent_vel = agent_velocity;
    var agent_rot = agent_rotation;
    var agent_energy_cur = agent_energy;
    var pairing_counter = agent_pairing_counter;

    // Morphology-driven swim uses per-part drag inside the physics loop.

    // Initialize output scalars explicitly (avoid full Agent megacopy).
    agents_out[agent_id].position = agent_position;
    agents_out[agent_id].velocity = agent_velocity;
    agents_out[agent_id].rotation = agent_rotation;
    agents_out[agent_id].energy = agent_energy;
    agents_out[agent_id].energy_capacity = agent_energy_capacity;
    agents_out[agent_id].torque_debug = agent_torque_debug;
    agents_out[agent_id].morphology_origin = agent_morphology_origin;
    agents_out[agent_id].alive = agent_alive;
    agents_out[agent_id].body_count = agent_body_count;
    agents_out[agent_id].pairing_counter = agent_pairing_counter;
    agents_out[agent_id].is_selected = agent_is_selected;
    agents_out[agent_id].generation = agent_generation;
    agents_out[agent_id].age = agent_age;
    agents_out[agent_id].total_mass = agent_total_mass;
    agents_out[agent_id].poison_resistant_count = agent_poison_resistant_count;
    agents_out[agent_id].gene_length = agent_gene_length;

    // Persist genome metadata across ping-pong buffers.
    agents_out[agent_id].genome_offset = agent_genome_offset;

    // Keep genome synchronized across ping-pong buffers.
    // NOTE: This is required for correctness because next frame's agents_in is this frame's
    // agents_out. If we don't write genome every frame, the buffer we write into will retain
    // stale/garbage genome data for existing agents.
    //
    // IMPORTANT: Naga/WGSL validation can reject dynamic indexing into certain array expressions.
    // Use constant indices for the fixed-size packed genome.
    agents_out[agent_id].genome_packed[0u] = agent_genome_packed[0u];
    agents_out[agent_id].genome_packed[1u] = agent_genome_packed[1u];
    agents_out[agent_id].genome_packed[2u] = agent_genome_packed[2u];
    agents_out[agent_id].genome_packed[3u] = agent_genome_packed[3u];
    agents_out[agent_id].genome_packed[4u] = agent_genome_packed[4u];
    agents_out[agent_id].genome_packed[5u] = agent_genome_packed[5u];
    agents_out[agent_id].genome_packed[6u] = agent_genome_packed[6u];
    agents_out[agent_id].genome_packed[7u] = agent_genome_packed[7u];
    agents_out[agent_id].genome_packed[8u] = agent_genome_packed[8u];
    agents_out[agent_id].genome_packed[9u] = agent_genome_packed[9u];
    agents_out[agent_id].genome_packed[10u] = agent_genome_packed[10u];
    agents_out[agent_id].genome_packed[11u] = agent_genome_packed[11u];
    agents_out[agent_id].genome_packed[12u] = agent_genome_packed[12u];
    agents_out[agent_id].genome_packed[13u] = agent_genome_packed[13u];
    agents_out[agent_id].genome_packed[14u] = agent_genome_packed[14u];
    agents_out[agent_id].genome_packed[15u] = agent_genome_packed[15u];

    // ====== MORPHOLOGY BUILD ======
    // Genome scan only happens on first frame (body_count == 0)
    // After that, we just use the cached part_type values in body[]
    var body_count_val = agent_body_count;
    var first_build = (agent_body_count == 0u);
    var start_byte = 0u;

    // Cache previous-frame body geometry so we can estimate morphology-driven motion.
    // Only meaningful after the first build.
    var prev_body_pos: array<vec2<f32>, MAX_BODY_PARTS>;
    if (!first_build) {
        for (var i = 0u; i < min(body_count_val, MAX_BODY_PARTS); i++) {
            prev_body_pos[i] = agents_in[agent_id].body[i].pos;
        }
    }

    if (first_build) {
        // FIRST BUILD: Scan genome and populate body[].part_type
        var start = 0xFFFFFFFFu;
        if (params.require_start_codon == 1u) {
            start = genome_find_start_codon(agent_genome_packed, agent_genome_offset, agent_gene_length);
        } else {
            start = genome_find_first_coding_triplet(agent_genome_packed, agent_genome_offset, agent_gene_length);
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
            let step = translate_codon_step(agent_genome_packed, pos_b, agent_genome_offset, agent_gene_length, params.ignore_stop_codons == 1u);

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

        // Poison resistance affects only poison damage (handled later), not signal-controlled bending.

        // Dynamic chain build - angles modulated by alpha/beta signals
        var current_pos = vec2<f32>(0.0);
        var current_angle = 0.0;
        var chirality_flip = 1.0;
        var sum_angle_mass = 0.0;
        var total_mass_angle = 0.0;
        var com_sum = vec2<f32>(0.0);
        var mass_sum = 0.0;
        var total_capacity = 0.0;
        // Reset to 0 before accumulation
        total_mass_morphology = 0.0;
        color_sum_morphology = 0.0;
        var poison_resistant_count = 0u;

        // Loop through existing body parts and rebuild positions
        for (var i = 0u; i < min(body_count_val, MAX_BODY_PARTS); i++) {
            // Read cached part_type and persist it to output (ping-pong safety).
            // IMPORTANT: on the first build we just wrote part_type into agents_out above,
            // so we must read from agents_out here (agents_in is still empty).
            let final_part_type = select(
                agents_in[agent_id].body[i].part_type,
                agents_out[agent_id].body[i].part_type,
                first_build
            );
            agents_out[agent_id].body[i].part_type = final_part_type;
            let base_type = get_base_part_type(final_part_type);

            // Check for chiral flipper
            if (base_type == 30u) {
                chirality_flip = -chirality_flip;
            }

            // Count poison-resistant organs (type 29) for poison damage reduction.
            if (base_type == 29u) {
                poison_resistant_count += 1u;
            }

            let props = get_amino_acid_properties(base_type);
            total_capacity += props.energy_storage;

            // Read previous frame's signals from input (agents_out is not pre-copied).
            let alpha = agents_in[agent_id].body[i].alpha_signal;
            let beta = agents_in[agent_id].body[i].beta_signal;

            // Modulate angle based on signals
            let alpha_effect = alpha * props.alpha_sensitivity * SIGNAL_GAIN * ANGLE_GAIN_ALPHA;
            let beta_effect = beta * props.beta_sensitivity * SIGNAL_GAIN * ANGLE_GAIN_BETA;
            var target_signal_angle = alpha_effect + beta_effect;
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

            // Accumulate center-of-mass in the same loop to avoid a second pass.
            com_sum += current_pos * m;
            mass_sum += m;

            // Check organ type once for both data field and _pad handling
            let is_vampire_mouth = (base_type == 33u);

            // Preserve organ internal state across frames by default.
            // Non-organs and vampire mouths may overwrite _pad below.
            agents_out[agent_id].body[i]._pad = agents_in[agent_id].body[i]._pad;

            // Data field: for vampire mouths (type 33), store packed world position for movement tracking
            if (is_vampire_mouth) {
                // Store current world position (agent.position + rotated local offset)
                let rotated_pos = apply_agent_rotation(current_pos, agent_rotation);
                let mouth_world_pos = agent_position + rotated_pos;
                agents_out[agent_id].body[i].data = pack_position_to_f32(mouth_world_pos, f32(SIM_SIZE));
            } else {
                agents_out[agent_id].body[i].data = 0.0;
            }
        }

        // Energy capacity
        agents_out[agent_id].energy_capacity = total_capacity;

        // Store total mass (only changes when morphology rebuilds)
        agents_out[agent_id].total_mass = max(total_mass_morphology, 0.05);

        // Store poison resistance count (only changes when morphology rebuilds)
        agents_out[agent_id].poison_resistant_count = poison_resistant_count;

        // Center of mass recentering
        let rec_n = body_count_val;
        if (rec_n > 0u) {
            let com = com_sum / max(mass_sum, 0.0001);

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
                let p0 = agents_out[agent_id].body[i].pos - com;
                agents_out[agent_id].body[i].pos = vec2<f32>(
                    p0.x * c_inv - p0.y * s_inv,
                    p0.x * s_inv + p0.y * c_inv
                );
            }

            // Rotate morphology origin
            let o = -com;
            agents_out[agent_id].morphology_origin = vec2<f32>(
                o.x * c_inv - o.y * s_inv,
                o.x * s_inv + o.y * c_inv
            );
        }
    }

    // NOTE: morphology recentering may update agents_out.rotation (avg_angle compensation).
    // We intentionally do NOT sync `agent_rot` here: `agent_rot` is the agent's physical rotation
    // integrated from torque, while the recentering rotation is a morphology gauge correction.

    // Inject morphology-driven motion into the fluid.
    // This is intentionally simple: estimate per-part velocity from (new_world - old_world) / dt,
    // then splat a matching force into the fluid solver.
    if (MORPHOLOGY_INJECT_FLUID_FORCE && !first_build) {
        let strength = max(params.prop_wash_strength_fluid, 0.0) * MORPHOLOGY_FLUID_COUPLING;
        if (strength > 0.0) {
            let dt_safe = max(params.dt, 1e-3);
            let old_rot = agent_rotation;
            let new_rot = agents_out[agent_id].rotation;

            for (var i = 0u; i < min(body_count_val, MAX_BODY_PARTS); i++) {
                let old_local = prev_body_pos[i];
                let new_local = agents_out[agent_id].body[i].pos;

                let old_world = agent_position + apply_agent_rotation(old_local, old_rot);
                let new_world = agent_position + apply_agent_rotation(new_local, new_rot);

                var delta_world = new_world - old_world;
                let dlen = length(delta_world);
                if (dlen > MORPHOLOGY_MAX_WORLD_DELTA) {
                    delta_world = delta_world * (MORPHOLOGY_MAX_WORLD_DELTA / max(dlen, 1e-6));
                }
                let delta_len2 = dot(delta_world, delta_world);
                if (delta_len2 > 1e-6) {
                    // Convert displacement to a velocity-like vector.
                    var v = delta_world / dt_safe;
                    let vlen = length(v);
                    if (vlen > MORPHOLOGY_MAX_WORLD_VEL) {
                        v = v * (MORPHOLOGY_MAX_WORLD_VEL / max(vlen, 1e-6));
                    }

                    let scaled_force = -v * FLUID_FORCE_SCALE * 0.1 * strength;
                    add_fluid_force_splat(new_world, scaled_force);
                }
            }
        }
    }

    // Morphology-driven swimming is handled as per-part drag inside the physics pass.

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
    var propeller_thrust_magnitude: array<f32, MAX_BODY_PARTS>; // Store activity magnitude for cost calculation (propeller or displacer)
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
        old_alpha[i] = agents_in[agent_id].body[i].alpha_signal;
        old_beta[i] = agents_in[agent_id].body[i].beta_signal;

        // Collect enabler positions
        if (props.is_inhibitor) { // enabler role
            enabler_positions[enabler_count] = part_i.pos;
            enabler_count += 1u;
        }
    }

    // ====== COLLECT NEARBY AGENTS ONCE (for sensors and physics) ======
    let scale = f32(SPATIAL_GRID_SIZE) / f32(SIM_SIZE);
    let my_grid_x = u32(clamp(agent_position.x * scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));
    let my_grid_y = u32(clamp(agent_position.y * scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));

    var neighbor_count = 0u;
    var neighbor_ids: array<u32, 64>; // Store up to 64 nearby agents

    // When the local window contains more than 64 eligible neighbors, keeping the first 64
    // in a fixed dy/dx scan order creates a directional bias in repulsion (dense blobs).
    // Use reservoir sampling so the final set is an unbiased subset of all eligible neighbors.
    var neighbor_seen = 0u;

    // Spiral scan (square rings) around the agent's cell.
    // This is more symmetric than row-major dy/dx and reduces structured bias.
    for (var radius: i32 = 0; radius <= 10; radius++) {
        if (radius == 0) {
            // Center cell
            let check_x = i32(my_grid_x);
            let check_y = i32(my_grid_y);
            if (check_x >= 0 && check_x < i32(SPATIAL_GRID_SIZE) &&
                check_y >= 0 && check_y < i32(SPATIAL_GRID_SIZE)) {
                let check_cell = u32(check_y) * SPATIAL_GRID_SIZE + u32(check_x);
                let stamp = atomicLoad(&agent_spatial_grid[spatial_epoch_index(check_cell)]);
                if (stamp == current_stamp) {
                    let raw_neighbor_id = atomicLoad(&agent_spatial_grid[spatial_id_index(check_cell)]);
                    let neighbor_id = raw_neighbor_id & 0x7FFFFFFFu;
                    if (neighbor_id != SPATIAL_GRID_EMPTY && neighbor_id != SPATIAL_GRID_CLAIMED && neighbor_id != agent_id) {
                        let neighbor_alive = agents_in[neighbor_id].alive;
                        let neighbor_energy = agents_in[neighbor_id].energy;
                        if (neighbor_alive != 0u && neighbor_energy > 0.0) {
                            neighbor_seen++;
                            if (neighbor_count < 64u) {
                                neighbor_ids[neighbor_count] = neighbor_id;
                                neighbor_count++;
                            } else {
                                let seed = (agent_id * 747796405u) ^ (params.epoch * 2891336453u) ^ (params.random_seed * 196613u) ^ check_cell;
                                let r = hash(seed) % neighbor_seen;
                                if (r < 64u) {
                                    neighbor_ids[r] = neighbor_id;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Top and bottom edges
            for (var dx: i32 = -radius; dx <= radius; dx++) {
                // Top
                var check_x = i32(my_grid_x) + dx;
                var check_y = i32(my_grid_y) - radius;
                if (check_x >= 0 && check_x < i32(SPATIAL_GRID_SIZE) &&
                    check_y >= 0 && check_y < i32(SPATIAL_GRID_SIZE)) {
                    let check_cell = u32(check_y) * SPATIAL_GRID_SIZE + u32(check_x);
                    let stamp = atomicLoad(&agent_spatial_grid[spatial_epoch_index(check_cell)]);
                    if (stamp == current_stamp) {
                        let raw_neighbor_id = atomicLoad(&agent_spatial_grid[spatial_id_index(check_cell)]);
                        let neighbor_id = raw_neighbor_id & 0x7FFFFFFFu;
                        if (neighbor_id != SPATIAL_GRID_EMPTY && neighbor_id != SPATIAL_GRID_CLAIMED && neighbor_id != agent_id) {
                            let neighbor_alive = agents_in[neighbor_id].alive;
                            let neighbor_energy = agents_in[neighbor_id].energy;
                            if (neighbor_alive != 0u && neighbor_energy > 0.0) {
                                neighbor_seen++;
                                if (neighbor_count < 64u) {
                                    neighbor_ids[neighbor_count] = neighbor_id;
                                    neighbor_count++;
                                } else {
                                    let seed = (agent_id * 747796405u) ^ (params.epoch * 2891336453u) ^ (params.random_seed * 196613u) ^ check_cell;
                                    let r = hash(seed) % neighbor_seen;
                                    if (r < 64u) {
                                        neighbor_ids[r] = neighbor_id;
                                    }
                                }
                            }
                        }
                    }
                }

                // Bottom
                check_x = i32(my_grid_x) + dx;
                check_y = i32(my_grid_y) + radius;
                if (check_x >= 0 && check_x < i32(SPATIAL_GRID_SIZE) &&
                    check_y >= 0 && check_y < i32(SPATIAL_GRID_SIZE)) {
                    let check_cell = u32(check_y) * SPATIAL_GRID_SIZE + u32(check_x);
                    let stamp = atomicLoad(&agent_spatial_grid[spatial_epoch_index(check_cell)]);
                    if (stamp == current_stamp) {
                        let raw_neighbor_id = atomicLoad(&agent_spatial_grid[spatial_id_index(check_cell)]);
                        let neighbor_id = raw_neighbor_id & 0x7FFFFFFFu;
                        if (neighbor_id != SPATIAL_GRID_EMPTY && neighbor_id != SPATIAL_GRID_CLAIMED && neighbor_id != agent_id) {
                            let neighbor_alive = agents_in[neighbor_id].alive;
                            let neighbor_energy = agents_in[neighbor_id].energy;
                            if (neighbor_alive != 0u && neighbor_energy > 0.0) {
                                neighbor_seen++;
                                if (neighbor_count < 64u) {
                                    neighbor_ids[neighbor_count] = neighbor_id;
                                    neighbor_count++;
                                } else {
                                    let seed = (agent_id * 747796405u) ^ (params.epoch * 2891336453u) ^ (params.random_seed * 196613u) ^ check_cell;
                                    let r = hash(seed) % neighbor_seen;
                                    if (r < 64u) {
                                        neighbor_ids[r] = neighbor_id;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Left and right edges (excluding corners already covered)
            for (var dy: i32 = -radius + 1; dy <= radius - 1; dy++) {
                // Left
                var check_x = i32(my_grid_x) - radius;
                var check_y = i32(my_grid_y) + dy;
                if (check_x >= 0 && check_x < i32(SPATIAL_GRID_SIZE) &&
                    check_y >= 0 && check_y < i32(SPATIAL_GRID_SIZE)) {
                    let check_cell = u32(check_y) * SPATIAL_GRID_SIZE + u32(check_x);
                    let stamp = atomicLoad(&agent_spatial_grid[spatial_epoch_index(check_cell)]);
                    if (stamp == current_stamp) {
                        let raw_neighbor_id = atomicLoad(&agent_spatial_grid[spatial_id_index(check_cell)]);
                        let neighbor_id = raw_neighbor_id & 0x7FFFFFFFu;
                        if (neighbor_id != SPATIAL_GRID_EMPTY && neighbor_id != SPATIAL_GRID_CLAIMED && neighbor_id != agent_id) {
                            let neighbor_alive = agents_in[neighbor_id].alive;
                            let neighbor_energy = agents_in[neighbor_id].energy;
                            if (neighbor_alive != 0u && neighbor_energy > 0.0) {
                                neighbor_seen++;
                                if (neighbor_count < 64u) {
                                    neighbor_ids[neighbor_count] = neighbor_id;
                                    neighbor_count++;
                                } else {
                                    let seed = (agent_id * 747796405u) ^ (params.epoch * 2891336453u) ^ (params.random_seed * 196613u) ^ check_cell;
                                    let r = hash(seed) % neighbor_seen;
                                    if (r < 64u) {
                                        neighbor_ids[r] = neighbor_id;
                                    }
                                }
                            }
                        }
                    }
                }

                // Right
                check_x = i32(my_grid_x) + radius;
                check_y = i32(my_grid_y) + dy;
                if (check_x >= 0 && check_x < i32(SPATIAL_GRID_SIZE) &&
                    check_y >= 0 && check_y < i32(SPATIAL_GRID_SIZE)) {
                    let check_cell = u32(check_y) * SPATIAL_GRID_SIZE + u32(check_x);
                    let stamp = atomicLoad(&agent_spatial_grid[spatial_epoch_index(check_cell)]);
                    if (stamp == current_stamp) {
                        let raw_neighbor_id = atomicLoad(&agent_spatial_grid[spatial_id_index(check_cell)]);
                        let neighbor_id = raw_neighbor_id & 0x7FFFFFFFu;
                        if (neighbor_id != SPATIAL_GRID_EMPTY && neighbor_id != SPATIAL_GRID_CLAIMED && neighbor_id != agent_id) {
                            let neighbor_alive = agents_in[neighbor_id].alive;
                            let neighbor_energy = agents_in[neighbor_id].energy;
                            if (neighbor_alive != 0u && neighbor_energy > 0.0) {
                                neighbor_seen++;
                                if (neighbor_count < 64u) {
                                    neighbor_ids[neighbor_count] = neighbor_id;
                                    neighbor_count++;
                                } else {
                                    let seed = (agent_id * 747796405u) ^ (params.epoch * 2891336453u) ^ (params.random_seed * 196613u) ^ check_cell;
                                    let r = hash(seed) % neighbor_seen;
                                    if (r < 64u) {
                                        neighbor_ids[r] = neighbor_id;
                                    }
                                }
                            }
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
        let perpendicular_world = normalize(apply_agent_rotation(perpendicular_local, agent_rotation));

        if (amino_props.is_alpha_sensor) {
            let rotated_pos = apply_agent_rotation(agents_out[agent_id].body[i].pos, agent_rotation);
            let world_pos = agent_position + rotated_pos;
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
            let rotated_pos = apply_agent_rotation(agents_out[agent_id].body[i].pos, agent_rotation);
            let world_pos = agent_position + rotated_pos;
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
            let rotated_pos = apply_agent_rotation(agents_out[agent_id].body[i].pos, agent_rotation);
            let world_pos = agent_position + rotated_pos;

            // Get sensor orientation (perpendicular to organ)
            let axis_local = normalize(agents_out[agent_id].body[i].pos);
            let perpendicular_local = normalize(vec2<f32>(-axis_local.y, axis_local.x));
            let perpendicular_world = normalize(apply_agent_rotation(perpendicular_local, agent_rotation));

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
            let rotated_pos = apply_agent_rotation(agents_out[agent_id].body[i].pos, agent_rotation);
            let world_pos = agent_position + rotated_pos;
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
            let rotated_pos = apply_agent_rotation(agents_out[agent_id].body[i].pos, agent_rotation);
            let world_pos = agent_position + rotated_pos;
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
            let rotated_pos = apply_agent_rotation(agents_out[agent_id].body[i].pos, agent_rotation);
            let sensor_world_pos = agent_position + rotated_pos;

            // Get sensor orientation (perpendicular to organ, rotated 90 degrees)
            let axis_local = normalize(agents_out[agent_id].body[i].pos);
            let perpendicular_local = normalize(vec2<f32>(-axis_local.y, axis_local.x));
            let perpendicular_world = normalize(apply_agent_rotation(perpendicular_local, agent_rotation));

            // Use agent_color calculated from color_sum_morphology
            let sensed_value = sample_neighbors_color(sensor_world_pos, neighbor_search_radius, params.debug_mode != 0u, perpendicular_world, agent_color, &neighbor_ids, neighbor_count);

            // Add agent color difference signal to alpha
            new_alpha += sensed_value;
        }

        // AGENT BETA SENSOR - Organ type 35 (V/M + Y)
        // Senses nearby agent colors from trail
        if (base_type == 35u) {
            let rotated_pos = apply_agent_rotation(agents_out[agent_id].body[i].pos, agent_rotation);
            let sensor_world_pos = agent_position + rotated_pos;

            // Get sensor orientation (perpendicular to organ, rotated 90 degrees)
            let axis_local = normalize(agents_out[agent_id].body[i].pos);
            let perpendicular_local = normalize(vec2<f32>(-axis_local.y, axis_local.x));
            let perpendicular_world = normalize(apply_agent_rotation(perpendicular_local, agent_rotation));

            // Use agent_color calculated from color_sum_morphology
            let sensed_value = sample_neighbors_color(sensor_world_pos, neighbor_search_radius, params.debug_mode != 0u, perpendicular_world, agent_color, &neighbor_ids, neighbor_count);

            // Add agent color difference signal to beta
            new_beta += sensed_value;
        }

        // PAIRING STATE SENSOR - Organ type 36 (H/Q + I/K)
        // Emits alpha or beta based on genome pairing completion percentage
        if (base_type == 36u) {
            // Get pairing percentage (0.0 to 1.0)
            let pairing_percentage = f32(agent_pairing_counter) / f32(GENOME_BYTES);

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
            let energy_t = clamp(agent_energy / 50.0, 0.0, 1.0);
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
            let world_pos = agent_pos + apply_agent_rotation(part_pos, agent_rot);
            var slope_gradient = read_gamma_slope(grid_index(world_pos));

            // Preserve historical behavior: slope sensors respond to both terrain slope AND
            // the configured global vector force (gravity/wind).
            if (params.vector_force_power > 0.0) {
                let gravity_vector = vec2<f32>(
                    params.vector_force_x * params.vector_force_power,
                    params.vector_force_y * params.vector_force_power
                );
                slope_gradient += gravity_vector;
            }

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
            let orientation_world = apply_agent_rotation(orientation_local, agent_rot);

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

        // Apply signal decay (requested): global decay for all parts (amino acids + organs).
        new_alpha *= 0.997;
        new_beta *= 0.997;

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
    // total_mass was calculated during morphology, but Anchor (type 42) can temporarily
    // increase effective mass when active.
    var total_mass = agents_out[agent_id].total_mass;
    let morphology_origin = agents_out[agent_id].morphology_origin;

    // Anchor mass multiplier when active (stored in part._pad.y as 0/1).
    let ANCHOR_MASS_MULT = 100.0;
    var anchor_mass_extra = 0.0;
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let part_out = agents_out[agent_id].body[i];
        let base_type_out = get_base_part_type(part_out.part_type);
        if (base_type_out == 42u && part_out._pad.y > 0.5) {
            let props = get_amino_acid_properties(base_type_out);
            let base_mass = max(props.mass, 0.01);
            anchor_mass_extra += base_mass * (ANCHOR_MASS_MULT - 1.0);
        }
    }
    total_mass = max(total_mass + anchor_mass_extra, 0.05);
    agents_out[agent_id].total_mass = total_mass;

    // Anchor (type 42): if any anchor was active last frame, freeze the whole agent.
    // Read from agents_in to avoid same-frame ordering dependence.
    var anchor_active_prev = false;
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let part_in = agents_in[agent_id].body[i];
        let base_type_in = get_base_part_type(part_in.part_type);
        if (base_type_in == 42u && part_in._pad.y > 0.5) {
            anchor_active_prev = true;
            break;
        }
    }

    let drag_coefficient = total_mass * 0.5;

    // Morphology swim strength (reuses the existing prop-wash UI knobs as a convenient control).
    // This is applied as per-part drag later (no direct angular-velocity injection).
    let morph_swim_strength = select(
        0.0,
        max(max(params.prop_wash_strength, 0.0), max(params.prop_wash_strength_fluid, 0.0)) * MORPHOLOGY_SWIM_COUPLING,
        (MORPHOLOGY_SWIM_ENABLED && !first_build)
    );

    // Accumulate forces and torques (relative to CoM)
    var force = vec2<f32>(0.0);
    var torque = 0.0;

    // (No global morphology-swim impulse here; handled per-part.)

    // Agent-to-agent repulsion (simplified: once per agent pair, using total masses)
    for (var n = 0u; n < neighbor_count; n++) {
        // IMPORTANT: read neighbors from agents_in (stable snapshot).
        // Reading from agents_out here creates a cross-invocation race (other agents are
        // writing their own agents_out entries concurrently), which can introduce artifacts
        // like directional drift/banding.
        let neighbor_id = neighbor_ids[n];
        // Avoid copying full neighbor Agent (includes body array).
        let neighbor_pos = agents_in[neighbor_id].position;
        let neighbor_mass = max(agents_in[neighbor_id].total_mass, 0.01);

        let delta = agent_pos - neighbor_pos;
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
            let reduced_mass = (total_mass * neighbor_mass) / (total_mass + neighbor_mass);

            force += direction * clamped_force * reduced_mass;
        }
    }

    // Now calculate forces using the updated morphology (using pre-collected neighbors)
    var chirality_flip_physics = 1.0; // Track cumulative chirality for propeller direction
    // Accumulate rotational inertia and detect vampire mouths in this same pass
    // to avoid extra full body scans.
    var moment_of_inertia = 0.0;
    var has_vampire_mouth = false;
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let part = agents_out[agent_id].body[i];

        // Get amino acid properties
        let base_type = get_base_part_type(part.part_type);
        let amino_props = get_amino_acid_properties(base_type);

        if (base_type == 33u) {
            has_vampire_mouth = true;
        }

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

        // Moment of inertia (rotation-invariant; can be computed in local space)
        let anchor_active = (base_type == 42u && part._pad.y > 0.5);
        let base_part_mass = max(amino_props.mass, 0.01);
        let part_mass = select(base_part_mass, base_part_mass * ANCHOR_MASS_MULT, anchor_active);
        moment_of_inertia += part_mass * dot(offset_from_com, offset_from_com);
        let r_com = apply_agent_rotation(offset_from_com, agent_rot);
        let rotated_midpoint = apply_agent_rotation(segment_midpoint, agent_rot);
        let world_pos = agent_pos + rotated_midpoint;

        let part_weight = part_mass / total_mass;

        // Morphology-driven swimming via per-part resistive drag.
        // Apply a drag force opposing the part's relative motion vs (agent movement + local fluid flow).
        // Internal deformation (old_local -> new_local) creates relative motion that can generate thrust.
        if (morph_swim_strength > 0.0 && !anchor_active_prev) {
            // Segment tangent (local -> world). Use neighbor to define axis.
            var seg_local = vec2<f32>(1.0, 0.0);
            if (i > 0u) {
                seg_local = part.pos - agents_out[agent_id].body[i - 1u].pos;
            } else if (body_count > 1u) {
                seg_local = agents_out[agent_id].body[1u].pos - part.pos;
            }
            let seg_len = length(seg_local);
            let t_local = select(seg_local / seg_len, vec2<f32>(1.0, 0.0), seg_len < 1e-6);
            let t_world = normalize(apply_agent_rotation(t_local, agent_rot));

            // Fluid velocity at this part (world units per tick).
            // IMPORTANT: when fluids are disabled, the velocity buffer can hold stale values.
            // Gate sampling behind the same switch used elsewhere (fluid_wind_push_strength).
            var v_fluid_frame = vec2<f32>(0.0);
            if (params.fluid_wind_push_strength != 0.0) {
                if (world_pos.x >= 0.0 && world_pos.x < f32(SIM_SIZE) && world_pos.y >= 0.0 && world_pos.y < f32(SIM_SIZE)) {
                    let v = sample_fluid_velocity_bilinear(world_pos);
                    let cell_to_world = f32(SIM_SIZE) / f32(FLUID_GRID_SIZE);
                    v_fluid_frame = v * (cell_to_world * params.dt);
                }
            }

            // Per-tick deformation displacement.
            // IMPORTANT: use local delta rotated by the *current* agent rotation so that rigid-body
            // rotation between frames does NOT create fake deformation (this was making compact
            // "ball" agents drift even when they aren't waving).
            let old_local = prev_body_pos[i];
            let new_local = part.pos;
            let delta_local = new_local - old_local;

            // Deadzone: ignore tiny geometry jitter.
            let delta_len = length(delta_local);
            if (delta_len < 1e-4) {
                continue;
            }

            var v_def = apply_agent_rotation(delta_local, agent_rot);
            let vdef_len = length(v_def);
            if (vdef_len > MORPHOLOGY_MAX_WORLD_DELTA) {
                v_def = v_def * (MORPHOLOGY_MAX_WORLD_DELTA / max(vdef_len, 1e-6));
            }

            // Relative motion of this part vs the fluid.
            // NOTE: use only internal deformation here so agents don't "swim" from rigid translation
            // (they are already overdamped at the agent level).
            let v_rel = v_def - v_fluid_frame;

            // Anisotropic resistive drag: C = c_perp*I + (c_par-c_perp)*t*t^T
            let tx = t_world.x;
            let ty = t_world.y;
            let d = (MORPHOLOGY_SWIM_C_PAR - MORPHOLOGY_SWIM_C_PERP);
            let C_xx = MORPHOLOGY_SWIM_C_PERP + d * (tx * tx);
            let C_xy = d * (tx * ty);
            let C_yy = MORPHOLOGY_SWIM_C_PERP + d * (ty * ty);

            var Cv = vec2<f32>(C_xx * v_rel.x + C_xy * v_rel.y, C_xy * v_rel.x + C_yy * v_rel.y);
            let cv_len = length(Cv);
            if (cv_len > MORPHOLOGY_SWIM_MAX_FRAME_VEL) {
                Cv = Cv * (MORPHOLOGY_SWIM_MAX_FRAME_VEL / max(cv_len, 1e-6));
            }

            // Convert velocity-like drag into force consistent with overdamped integrator.
            // part_weight * drag_coefficient == ~part_mass*0.5.
            let swim_force = (-Cv) * (morph_swim_strength * part_weight) * drag_coefficient;
            force += swim_force;
            torque += (r_com.x * swim_force.y - r_com.y * swim_force.x);
        }

        // Slope force per amino acid
        // Anchor (type 42): when active, ignore slope/global-vector contribution so
        // the mass multiplier doesn't turn slope into extreme lateral sliding.
        if (!anchor_active) {
            var slope_gradient = read_gamma_slope(grid_index(world_pos));
            // Preserve historical behavior: add global vector force at usage site.
            if (params.vector_force_power > 0.0) {
                let gravity_vector = vec2<f32>(
                    params.vector_force_x * params.vector_force_power,
                    params.vector_force_y * params.vector_force_power
                );
                slope_gradient += gravity_vector;
            }
            let slope_force = -slope_gradient * params.gamma_strength * part_mass;
            force += slope_force;
            torque += (r_com.x * slope_force.y - r_com.y * slope_force.x);
        }

        // Fluid force per amino acid (apply at the same point as slope force)
        // Anchor (type 42): when active, ignore wind push so it doesn't drift.
        if (!anchor_active && params.fluid_wind_push_strength != 0.0) {
            // Default body parts (amino acids + organs) to coupling=1.0 unless explicitly overridden.
            let wind_coupling = select(
                amino_props.fluid_wind_coupling,
                0.5,
                (amino_props.fluid_wind_coupling == 0.0)
            );

            if (wind_coupling != 0.0) {
                // Avoid clamping OOB into edge cells (prevents artificial edge wind).
                if (world_pos.x >= 0.0 && world_pos.x < f32(SIM_SIZE) && world_pos.y >= 0.0 && world_pos.y < f32(SIM_SIZE)) {
                    // Fluid velocity is in fluid-cell units; convert to world-units per simulation tick.
                    let v = sample_fluid_velocity_bilinear(world_pos);
                    let cell_to_world = f32(SIM_SIZE) / f32(FLUID_GRID_SIZE);
                    // NOTE: fluid velocity is in (fluid-cells / second). Our agent integrator uses
                    // per-tick deltas, so multiply by params.dt to convert seconds -> tick.
                    let v_frame = v * (cell_to_world * params.dt);

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
    if (PROPELLERS_ENABLED && amino_props.is_propeller && agent_energy_cur >= amino_props.energy_consumption) {
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
            let thrust_dir_world = apply_agent_rotation(thrust_local, agent_rot);

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
                    // Throw distance in cells for the direct (fluidless) prop wash.
                    // Increased range to make jets move material farther.
                    let distance = clamp(prop_strength * 4.0, 1.0, 10.0);
                    let target_world = clamped_pos + prop_dir * distance * grid_scale;
                    // IMPORTANT: keep mapping consistent with center cell selection.
                    // Using round() here creates a systematic half-cell (+0.5,+0.5) bias -> diagonal (45°) artifacts.
                    let target_gx = clamp(i32(target_world.x / grid_scale), 0, i32(GRID_SIZE) - 1);
                    let target_gy = clamp(i32(target_world.y / grid_scale), 0, i32(GRID_SIZE) - 1);
                    let target_idx = u32(target_gy) * GRID_SIZE + u32(target_gx);

                    if (target_idx != center_idx) {
                        var center_gamma = read_gamma_height(center_idx);
                        var target_gamma = read_gamma_height(target_idx);
                        var center_alpha = chem_grid[center_idx].x;
                        var target_alpha = chem_grid[target_idx].x;
                        var center_beta = chem_grid[center_idx].y;
                        var target_beta = chem_grid[target_idx].y;

                        let transfer_amount = prop_strength * 0.05 * part_weight;
                        if (transfer_amount > 0.0) {
                            // Capacities adjusted for 0..1 range
                            let alpha_capacity = max(0.0, 1.0 - target_alpha);
                            let beta_capacity = max(0.0, 1.0 - target_beta);
                            let gamma_capacity = max(0.0, 10.0 - target_gamma);

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
                                let prev_c = chem_grid[center_idx];
                                chem_grid[center_idx] = vec2<f32>(center_alpha, prev_c.y);
                                let prev_t = chem_grid[target_idx];
                                chem_grid[target_idx] = vec2<f32>(target_alpha, prev_t.y);
                            }

                            if (beta_transfer > 0.0) {
                                center_beta = clamp(center_beta - beta_transfer, 0.0, 1.0);
                                target_beta = clamp(target_beta + beta_transfer, 0.0, 1.0);
                                let prev_c = chem_grid[center_idx];
                                chem_grid[center_idx] = vec2<f32>(prev_c.x, center_beta);
                                let prev_t = chem_grid[target_idx];
                                chem_grid[target_idx] = vec2<f32>(prev_t.x, target_beta);
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
                if (PROPELLERS_APPLY_DIRECT_FORCE) {
                    force += thrust_force;
                    // Torque from lever arm r_com cross thrust (scaled down to reduce perpetual spinning)
                    torque += (r_com.x * thrust_force.y - r_com.y * thrust_force.x) * (6.0 * PROP_TORQUE_COUPLING);
                }

                // INJECT PROPELLER FORCE DIRECTLY INTO FLUID FORCES BUFFER
                // NOTE: Race condition possible with multiple agents, but the effect is additive so acceptable.
                let scaled_force = -thrust_force * FLUID_FORCE_SCALE * 0.1 * max(params.prop_wash_strength_fluid, 0.0);
                add_fluid_force_splat(world_pos, scaled_force);
            }
        }


        // Displacer force: symmetric fluid displacement (net agent thrust ~0)
        // - Displacer A (base_type 25): aligned with propeller direction (perpendicular to segment axis)
        // - Displacer B (base_type 27): aligned with segment axis (90° relative)
        if (amino_props.is_displacer && agent_energy_cur >= amino_props.energy_consumption) {
            var segment_dir = vec2<f32>(0.0);
            if (i > 0u) {
                let prev = agents_out[agent_id].body[i-1u].pos;
                segment_dir = part.pos - prev;
            } else if (agents_out[agent_id].body_count > 1u) {
                let next = agents_out[agent_id].body[1u].pos;
                segment_dir = next - part.pos;
            } else {
                segment_dir = vec2<f32>(1.0, 0.0);
            }
            let seg_len = length(segment_dir);
            let axis_local = select(segment_dir / seg_len, vec2<f32>(1.0, 0.0), seg_len < 1e-4);

            // Choose displacement direction in local space
            var disp_local = vec2<f32>(-axis_local.y, axis_local.x) * chirality_flip_physics;
            if (base_type == 27u) {
                disp_local = axis_local * chirality_flip_physics;
            }

            let disp_dir_world = normalize(apply_agent_rotation(disp_local, agent_rot));
            let dir_len = length(disp_dir_world);
            if (dir_len > 1e-5) {
                let dir = disp_dir_world / dir_len;

                // Displacer "chem sweep": move a fraction of local chem (alpha/beta)
                // along the bipolar axis.
                    // This provides a *direct* displacer effect even when fluids are disabled.
                {
                    let clamped_pos = clamp_position(world_pos);
                    let grid_scale = f32(SIM_SIZE) / f32(GRID_SIZE);
                    var gx = i32(clamped_pos.x / grid_scale);
                    var gy = i32(clamped_pos.y / grid_scale);
                    gx = clamp(gx, 0, i32(GRID_SIZE) - 1);
                    gy = clamp(gy, 0, i32(GRID_SIZE) - 1);

                    let center_idx = u32(gy) * GRID_SIZE + u32(gx);
                    var center = chem_grid[center_idx];
                    var center_gamma = max(read_gamma_height(center_idx), 0.0);

                    let base_seed = hash(agent_id * 747796405u + i * 2891336453u + params.random_seed * 196613u);
                    // Throw distance in cells for the direct (fluidless) displacer sweep.
                    // Increased range to make displacers move material farther.
                    let dist_plus = 1 + (hash(base_seed ^ 0xA3C59ACBu) % 6u);  // 1..6
                    let dist_minus = 1 + (hash(base_seed ^ 0x1B56C4E9u) % 6u); // 1..6

                    // When fluids are enabled, keep the previous strong sweep (25% total).
                    // When fluids are disabled, use a smaller sweep that still depends on
                    // enabler amplification so it doesn't become a free always-on teleporter.
                    let wash = max(params.prop_wash_strength, 0.0);
                    let transfer_frac_each = select(
                        0.04 * (amplification * amplification),
                        0.125,
                        wash > 0.0
                    );
                    var moved_any_gamma = false;

                    // +dir transfer
                    {
                        let target_world = clamped_pos + dir * (f32(dist_plus) * grid_scale);
                        let target_gx = clamp(i32(target_world.x / grid_scale), 0, i32(GRID_SIZE) - 1);
                        let target_gy = clamp(i32(target_world.y / grid_scale), 0, i32(GRID_SIZE) - 1);
                        let target_idx = u32(target_gy) * GRID_SIZE + u32(target_gx);

                        if (target_idx != center_idx) {
                            var dst = chem_grid[target_idx];
                            var dst_gamma = max(read_gamma_height(target_idx), 0.0);

                            let a_move = min(center.x * transfer_frac_each, max(0.0, 1.0 - dst.x));
                            let b_move = min(center.y * transfer_frac_each, max(0.0, 1.0 - dst.y));
                            let g_move = center_gamma * transfer_frac_each;

                            if (a_move > 0.0) {
                                center.x = clamp(center.x - a_move, 0.0, 1.0);
                                dst.x = clamp(dst.x + a_move, 0.0, 1.0);
                            }
                            if (b_move > 0.0) {
                                center.y = clamp(center.y - b_move, 0.0, 1.0);
                                dst.y = clamp(dst.y + b_move, 0.0, 1.0);
                            }

                            if (g_move > 0.0) {
                                center_gamma = max(center_gamma - g_move, 0.0);
                                dst_gamma = dst_gamma + g_move;
                                write_gamma_height(target_idx, dst_gamma);
                                moved_any_gamma = true;
                            }

                            chem_grid[target_idx] = dst;
                        }
                    }

                    // -dir transfer
                    {
                        let target_world = clamped_pos - dir * (f32(dist_minus) * grid_scale);
                        let target_gx = clamp(i32(target_world.x / grid_scale), 0, i32(GRID_SIZE) - 1);
                        let target_gy = clamp(i32(target_world.y / grid_scale), 0, i32(GRID_SIZE) - 1);
                        let target_idx = u32(target_gy) * GRID_SIZE + u32(target_gx);

                        if (target_idx != center_idx) {
                            var dst = chem_grid[target_idx];
                            var dst_gamma = max(read_gamma_height(target_idx), 0.0);

                            let a_move = min(center.x * transfer_frac_each, max(0.0, 1.0 - dst.x));
                            let b_move = min(center.y * transfer_frac_each, max(0.0, 1.0 - dst.y));
                            let g_move = center_gamma * transfer_frac_each;

                            if (a_move > 0.0) {
                                center.x = clamp(center.x - a_move, 0.0, 1.0);
                                dst.x = clamp(dst.x + a_move, 0.0, 1.0);
                            }
                            if (b_move > 0.0) {
                                center.y = clamp(center.y - b_move, 0.0, 1.0);
                                dst.y = clamp(dst.y + b_move, 0.0, 1.0);
                            }

                            if (g_move > 0.0) {
                                center_gamma = max(center_gamma - g_move, 0.0);
                                dst_gamma = dst_gamma + g_move;
                                write_gamma_height(target_idx, dst_gamma);
                                moved_any_gamma = true;
                            }

                            chem_grid[target_idx] = dst;
                        }
                    }

                    chem_grid[center_idx] = center;

                    if (moved_any_gamma) {
                        write_gamma_height(center_idx, center_gamma);
                    }
                }

                if (PROPELLERS_ENABLED) {
                    let quadratic_amp = amplification * amplification;
                    let displacer_strength = amino_props.thrust_force * 3.0 * quadratic_amp * 4.0;
                    // Reuse propeller_thrust_magnitude storage for displacer activity (mutually exclusive).
                    propeller_thrust_magnitude[i] = displacer_strength;

                    // Offset along the displacement axis so the force pair does not cancel trivially.
                    let cell_to_world = f32(SIM_SIZE) / f32(FLUID_GRID_SIZE);
                    let offset_world = cell_to_world * 2.0;
                    let p_plus = world_pos + dir * offset_world;
                    let p_minus = world_pos - dir * offset_world;

                    let f = dir * displacer_strength;
                    let force_scale = FLUID_FORCE_SCALE * 0.1 * max(params.prop_wash_strength_fluid, 0.0);

                    // Inject +f at +offset and -f at -offset -> net force = 0.
                    add_fluid_force_splat(p_plus, (-f * force_scale));
                    add_fluid_force_splat(p_minus, (f * force_scale));
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
    agent_vel = mix(agent_vel, new_velocity, mass_smoothing);

    if (anchor_active_prev) {
        agent_vel = vec2<f32>(0.0);
    }

    let v_len = length(agent_vel);
    if (v_len > VEL_MAX) {
        agent_vel = agent_vel * (VEL_MAX / v_len);
    }

    // Apply torque - overdamped angular motion (no angular inertia)
    // In viscous fluids, angular velocity is directly proportional to torque
    moment_of_inertia = max(moment_of_inertia, 0.01);

    // Overdamped rotation: angular_velocity = torque / rotational_drag
    let rotational_drag = moment_of_inertia * 20.0; // Increased rotational drag for stability
    var angular_velocity = torque / rotational_drag;

    // Add spin from local fluid vorticity (curl). This makes vortices visibly rotate agents.
    if (params.fluid_wind_push_strength != 0.0) {
        // center_of_mass is local (0,0) in this phase; sample at agent world position.
        let com_world = agent_pos;
        if (com_world.x >= 0.0 && com_world.x < f32(SIM_SIZE) && com_world.y >= 0.0 && com_world.y < f32(SIM_SIZE)) {
            let curl = sample_fluid_curl(com_world);
            // Curl is in ~1/sec; convert to per-tick delta with params.dt.
            angular_velocity += curl * params.fluid_wind_push_strength * WIND_CURL_ANGVEL_SCALE * params.dt;
        }
    }
    angular_velocity = angular_velocity * ANGULAR_BLEND;
    angular_velocity = clamp(angular_velocity, -ANGVEL_MAX, ANGVEL_MAX);

    if (anchor_active_prev) {
        angular_velocity = 0.0;
    }

    // Update rotation
    if (!DISABLE_GLOBAL_ROTATION) {
        agent_rot += angular_velocity;
    } else {
        agent_rot = 0.0; // keep zero for disabled global rotation experiment
    }

    // Update position
    // Closed world: clamp at boundaries
    agent_pos = clamp_position(agent_pos + agent_vel);

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
    var local_alpha = 0.0;
    var local_beta = 0.0;

    // If an agent has any vampire mouth (type 33), deactivate normal mouths.
    // (Computed earlier during the physics pass.)

    // Single loop through all body parts
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let part = agents_out[agent_id].body[i];
        let base_type = get_base_part_type(part.part_type);
        let props = get_amino_acid_properties(base_type);
        let rotated_pos = apply_agent_rotation(part.pos, agent_rot);
        let world_pos = agent_pos + rotated_pos;
        let idx = grid_index(world_pos);

        // Local chem accumulation for reproduction gating: sample at each part position.
        // This includes amino acids and organs (mouths, etc.).
        local_alpha += chem_grid[idx].x;
        local_beta += chem_grid[idx].y;

        // Anchor organ (type 42): signal-thresholded latch state.
        // - L-promoter anchors (LW/LY): use alpha signal (param bit 7 = 0)
        // - P-promoter anchors (PW/PY): use beta signal (param bit 7 = 1)
        if (base_type == 42u) {
            let organ_param = get_organ_param(part.part_type);
            let is_beta_anchor = (organ_param & 128u) != 0u;
            let signal = select(part.alpha_signal, part.beta_signal, is_beta_anchor);

            // Requested behavior: purely sign-based.
            // Active iff signal is positive; inactive if signal is 0 or negative.
            let now_active = signal > 0.0;

            agents_out[agent_id].body[i]._pad.y = select(0.0, 1.0, now_active);
        }

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
        let current_energy_trail = sanitize_f32(trail_grid_inject[idx].w);
        let agent_energy_nonneg = max(sanitize_f32(agent_energy_cur), 0.0);
        // Typical agent energy ~20; scale down contribution so trails don't saturate.
        let energy_deposit = agent_energy_nonneg * trail_deposit_strength * (0.1 / 20.0); // 20x weaker than before
        let blended_energy = sanitize_f32(current_energy_trail + energy_deposit);

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
            // Mouth energy cost:
            // - Base maintenance is always paid.
            // - Activity cost scales with enabler amplification (mouth effectiveness).
            // - Vampire mouths (type 33) pay 3x the activity cost.
            let base_maintenance = select(props.energy_consumption, props.energy_consumption * 3.0, base_type == 33u);

            // Deactivate normal mouths if any vampire mouth is present.
            let normal_mouth_deactivated = has_vampire_mouth && (base_type != 33u);
            if (normal_mouth_deactivated) {
                organ_extra = base_maintenance;
            } else {

            // 3) Feeding: mouths consume from alpha/beta grids
            // Get enabler amplification for this mouth
            let amplification = amplification_per_part[i];

            let activity_mult = select(1.0, 3.0, base_type == 33u);
            let activity_cost = props.energy_consumption * amplification * 1.5 * activity_mult;
            organ_extra = base_maintenance + activity_cost;

            // Consume alpha and beta based on per-amino absorption rates
            // and local availability, scaled by speed (slower = more absorption)
            // Bilinear availability sampling at the mouth position.
            // NOTE: This is paired with bilinear (4-cell) absorption writes below.
            let clamped_pos = clamp_position(world_pos);
            let grid_scale = f32(SIM_SIZE) / f32(GRID_SIZE);
            let gx = (clamped_pos.x / grid_scale) - 0.5;
            let gy = (clamped_pos.y / grid_scale) - 0.5;

            let x0 = i32(floor(gx));
            let y0 = i32(floor(gy));
            let x1 = min(x0 + 1, i32(GRID_SIZE) - 1);
            let y1 = min(y0 + 1, i32(GRID_SIZE) - 1);

            let fx = fract(gx);
            let fy = fract(gy);

            let w00 = (1.0 - fx) * (1.0 - fy);
            let w10 = fx * (1.0 - fy);
            let w01 = (1.0 - fx) * fy;
            let w11 = fx * fy;

            let idx00 = u32(clamp(y0, 0, i32(GRID_SIZE) - 1)) * GRID_SIZE + u32(clamp(x0, 0, i32(GRID_SIZE) - 1));
            let idx10 = u32(clamp(y0, 0, i32(GRID_SIZE) - 1)) * GRID_SIZE + u32(clamp(x1, 0, i32(GRID_SIZE) - 1));
            let idx01 = u32(clamp(y1, 0, i32(GRID_SIZE) - 1)) * GRID_SIZE + u32(clamp(x0, 0, i32(GRID_SIZE) - 1));
            let idx11 = u32(clamp(y1, 0, i32(GRID_SIZE) - 1)) * GRID_SIZE + u32(clamp(x1, 0, i32(GRID_SIZE) - 1));

            let c00 = chem_grid[idx00];
            let c10 = chem_grid[idx10];
            let c01 = chem_grid[idx01];
            let c11 = chem_grid[idx11];

            let available_alpha = w00 * c00.x + w10 * c10.x + w01 * c01.x + w11 * c11.x;
            let available_beta  = w00 * c00.y + w10 * c10.y + w01 * c01.y + w11 * c11.y;

            // Per-amino capture rates let us tune bite size vs. poison uptake
            // Apply speed effects and amplification to the rates themselves
            // Vampire mouths absorb 50% of what a normal mouth would.
            let mouth_absorption_multiplier = select(1.0, 0.5, base_type == 33u);
            let alpha_rate = max(props.energy_absorption_rate, 0.0)
                * speed_absorption_multiplier
                * amplification
                * mouth_absorption_multiplier;
            let beta_rate  = max(props.beta_absorption_rate, 0.0)
                * speed_absorption_multiplier
                * amplification
                * mouth_absorption_multiplier;

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

                // Apply alpha consumption - energy gain uses base food_power (speed already in consumption)
                if (consumed_alpha > 0.0) {
                    // Distribute absorption bilinearly into the 4 neighbor cells.
                    // NOTE: This is intentionally non-atomic and can be racy under contention.
                    let da00 = consumed_alpha * w00;
                    let da10 = consumed_alpha * w10;
                    let da01 = consumed_alpha * w01;
                    let da11 = consumed_alpha * w11;

                    if (da00 > 0.0) { let prev = chem_grid[idx00]; chem_grid[idx00] = vec2<f32>(clamp(prev.x - da00, 0.0, prev.x), prev.y); }
                    if (da10 > 0.0) { let prev = chem_grid[idx10]; chem_grid[idx10] = vec2<f32>(clamp(prev.x - da10, 0.0, prev.x), prev.y); }
                    if (da01 > 0.0) { let prev = chem_grid[idx01]; chem_grid[idx01] = vec2<f32>(clamp(prev.x - da01, 0.0, prev.x), prev.y); }
                    if (da11 > 0.0) { let prev = chem_grid[idx11]; chem_grid[idx11] = vec2<f32>(clamp(prev.x - da11, 0.0, prev.x), prev.y); }

                    agent_energy_cur += consumed_alpha * params.food_power;
                    total_consumed_alpha += consumed_alpha;
                }

                // Apply beta consumption - damage uses poison_power, reduced by poison protection
                if (consumed_beta > 0.0) {
                    // Distribute absorption bilinearly into the 4 neighbor cells.
                    // NOTE: This is intentionally non-atomic and can be racy under contention.
                    let db00 = consumed_beta * w00;
                    let db10 = consumed_beta * w10;
                    let db01 = consumed_beta * w01;
                    let db11 = consumed_beta * w11;

                    if (db00 > 0.0) { let prev = chem_grid[idx00]; chem_grid[idx00] = vec2<f32>(prev.x, clamp(prev.y - db00, 0.0, prev.y)); }
                    if (db10 > 0.0) { let prev = chem_grid[idx10]; chem_grid[idx10] = vec2<f32>(prev.x, clamp(prev.y - db10, 0.0, prev.y)); }
                    if (db01 > 0.0) { let prev = chem_grid[idx01]; chem_grid[idx01] = vec2<f32>(prev.x, clamp(prev.y - db01, 0.0, prev.y)); }
                    if (db11 > 0.0) { let prev = chem_grid[idx11]; chem_grid[idx11] = vec2<f32>(prev.x, clamp(prev.y - db11, 0.0, prev.y)); }

                    agent_energy_cur -= consumed_beta * params.poison_power * poison_multiplier;
                    total_consumed_beta += consumed_beta;
                }
            }
            }
        } else if (PROPELLERS_ENABLED && props.is_propeller) {
            // Propellers: base cost (always paid) + activity cost (linear with thrust)
            // Since thrust already scales quadratically with amp, cost should scale linearly with thrust
            let base_thrust = props.thrust_force * 3.0; // Max thrust with amp=1
            let thrust_ratio = propeller_thrust_magnitude[i] / base_thrust;
            // Reduce operational (thrust) cost by 1/3.
            let activity_cost = props.energy_consumption * thrust_ratio * 1.0;
            organ_extra = props.energy_consumption + activity_cost; // Base + activity
        } else if (props.is_displacer) {
            // Displacers: base cost + activity cost (linear with displacement strength)
            // Displacer strength is intentionally boosted (currently 4x) for visibility.
            // Normalize against that boost so the activity cost matches propellers at the same amp.
            let base_strength = props.thrust_force * 3.0 * 4.0; // Max with amp=1 (including strength boost)
            let strength_ratio = propeller_thrust_magnitude[i] / base_strength;
            let activity_cost = props.energy_consumption * strength_ratio * 1.0;
            organ_extra = props.energy_consumption + activity_cost;
        } else {
            // Other organs use linear amplification scaling
            let amp = amplification_per_part[i];
            organ_extra = props.energy_consumption * amp * 1.5;
        }
        energy_consumption += baseline + organ_extra;
    }

    // Cap energy by storage capacity after feeding (use post-build capacity)
    // Always clamp to avoid energy > capacity, and to zero when capacity == 0
    agent_energy_cur = clamp(agent_energy_cur, 0.0, max(capacity, 0.0));

    // 3) Maintenance: subtract consumption after feeding
    agent_energy_cur -= energy_consumption;

    // 4) Energy-based death check - death probability inversely proportional to energy
    // High energy = low death chance, low energy = high death chance
    let death_seed = agent_id * 2654435761u + params.random_seed * 1103515245u;
    let death_rnd = f32(hash(death_seed)) / 4294967295.0;

    // Prevent division by zero and NaN: use max(energy, 0.01) as divisor
    // At energy=10: probability / 10 = very low death chance
    // At energy=1: probability / 1 = normal death chance
    // At energy=0.01: probability / 0.01 = 100x higher death chance (starvation)
    let energy_divisor = max(agent_energy_cur, 0.01);
    let energy_adjusted_death_prob = params.death_probability / energy_divisor;

    if (death_rnd < energy_adjusted_death_prob) {
        // Deposit remains: stochastic decomposition into either alpha or beta
        // Fixed total deposit = 1.0 (in 0..1 grid units), spread across parts
        if (body_count > 0u) {
            let total_deposit = 1.0;
            let deposit_per_part = total_deposit / f32(body_count);
            for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
                let part = agents_out[agent_id].body[i];
                let rotated_pos = apply_agent_rotation(part.pos, agent_rot);
                let world_pos = agent_pos + rotated_pos;
                let idx = grid_index(world_pos);

                // Stochastic choice: 50% alpha (nutrient), 50% beta (toxin)
                let part_hash = hash(agent_id * 1000u + i * 100u + params.random_seed);
                let part_rnd = f32(part_hash % 1000u) / 1000.0;

                if (part_rnd < 0.5) {
                    let prev = chem_grid[idx];
                    chem_grid[idx] = vec2<f32>(min(prev.x + deposit_per_part, 1.0), prev.y);
                } else {
                    let prev = chem_grid[idx];
                    chem_grid[idx] = vec2<f32>(prev.x, min(prev.y + deposit_per_part, 1.0));
                }
            }
        }

        // If this was the selected agent, transfer selection to a random nearby agent.
        // Also immediately invalidate the inspector buffer so it can't keep rendering a stale
        // "alive" snapshot when readbacks are throttled.
        if (agent_is_selected == 1u) {
            var dead_snapshot = agents_out[agent_id];
            dead_snapshot.alive = 0u;
            dead_snapshot.is_selected = 0u;
            dead_snapshot.rotation = 0.0;
            selected_agent_buffer[0] = dead_snapshot;

            // Robust selection transfer:
            // - First, try a handful of hashed candidates.
            // - If that fails (e.g. unlucky dead picks), do a short linear scan.
            // This avoids losing selection and freezing the inspector.
            let transfer_hash = hash(agent_id * 2654435761u + params.random_seed);
            var transferred = false;

            if (params.agent_count > 0u) {
                for (var t = 0u; t < 8u; t = t + 1u) {
                    let cand = hash(transfer_hash + t * 747796405u) % params.agent_count;
                    if (cand != agent_id && agents_in[cand].alive == 1u) {
                        agents_out[cand].is_selected = 1u;
                        transferred = true;
                        break;
                    }
                }

                if (!transferred) {
                    // Wraparound scan from a hashed start (bounded to avoid GPU timeouts).
                    let start = transfer_hash % params.agent_count;
                    let limit = min(params.agent_count, 2048u);
                    for (var i = 0u; i < limit; i = i + 1u) {
                        let idx = (start + i) % params.agent_count;
                        if (idx != agent_id && agents_in[idx].alive == 1u) {
                            agents_out[idx].is_selected = 1u;
                            transferred = true;
                            break;
                        }
                    }
                }
            }

            // Selection state is stored in agents_out; keep local scalars immutable.
            agents_out[agent_id].is_selected = 0u;
        }

        // Avoid copying the full Agent payload on death.
        agents_out[agent_id].alive = 0u;
        agents_out[agent_id].body_count = 0u;
        agents_out[agent_id].energy = 0.0;
        agents_out[agent_id].velocity = vec2<f32>(0.0);
        agents_out[agent_id].pairing_counter = 0u;
        agents_out[agent_id].is_selected = 0u;
        return;
    }
    // Note: alive counting is handled in the compaction/merge passes

    // ====== RNA PAIRING ADVANCEMENT ======
    // Pairing counter probabilistically increments based on energy and chemical signals.
    // When it reaches gene_length, reproduction.wgsl will spawn offspring.

    // CRITICAL: Handle reproduction completion BEFORE pairing advancement.
    // Reproduction shader writes offspring when pairing_counter >= gene_length,
    // but it doesn't update the parent (different buffer). We must reset here.
    if (pairing_counter >= agent_gene_length && agent_gene_length > 0u) {
        // Reproduction happened - deduct 50% energy and reset counter
        agent_energy_cur *= 0.5;
        pairing_counter = 0u;
    }

    if (agent_gene_length > 0u && pairing_counter < agent_gene_length) {
        // If the agent has no energy capacity (no storage), it cannot sustain pairing.
        if (agent_energy_capacity > 0.0) {
            // Average local chemical signals across all body parts
            let local_alpha_avg = select(0.0, local_alpha / f32(body_count), body_count > 0u);
            let local_beta_avg = select(0.0, local_beta / f32(body_count), body_count > 0u);

            // Probability to increment/decrement counter based on conditions
            let pos_idx = grid_index(agent_pos);
            let seed = ((agent_id + 1u) * 747796405u) ^ (pairing_counter * 2891336453u) ^ (params.random_seed * 196613u) ^ pos_idx;
            let rnd = f32(hash(seed)) / 4294967295.0;
            let energy_for_pair = max(agent_energy_cur, 0.0);

            // Use averaged local signals for pairing drive
            let energy_scaled = sqrt(energy_for_pair);
            let alpha_gate = clamp(local_alpha_avg, 0.0, 1.0);
            let beta_gate = clamp(local_beta_avg, 0.0, 1.0);
            // TEMPORARY: deactivate alpha/beta influence on pairing
            // let pairing_drive = alpha_gate - beta_gate; // [-1, +1]
            let pairing_drive = 1.0; // Always max pairing rate (no chemical influence)

            let base_p = params.spawn_probability * energy_scaled * 0.1;
            let pair_p = clamp(base_p * max(pairing_drive, 0.0), 0.0, 1.0);
            let unpair_p = clamp(base_p * max(-pairing_drive, 0.0), 0.0, 1.0);

            // Pairing/un-pairing are mutually exclusive in a frame
            if (rnd < pair_p) {
                // Pairing cost per increment
                let pairing_cost = params.pairing_cost;
                if (agent_energy_cur >= pairing_cost) {
                    pairing_counter += 1u;
                    agent_energy_cur -= pairing_cost;
                }
            } else if (rnd < (pair_p + unpair_p)) {
                // Un-pairing event: lose one paired base (no energy refund)
                pairing_counter = select(pairing_counter - 1u, 0u, pairing_counter == 0u);
            }
        }
    }

    agents_out[agent_id].pairing_counter = pairing_counter;

    // Always write selected agent to readback buffer for inspector (even when drawing disabled)
    if (agent_is_selected == 1u) {
        // Publish an unrotated copy for inspector preview
        var unrotated_agent = agents_out[agent_id];
        unrotated_agent.rotation = 0.0;
        // Speed info now stored per-mouth in body[63].pos during loop above (for debugging)
        // Store the calculated gene_length (we already computed it above for reproduction)
        unrotated_agent.gene_length = agent_gene_length;
        // Copy generation/age/total_mass (already in agents_out) unchanged
        selected_agent_buffer[0] = unrotated_agent;
    }

    // Always write output state (simulation must continue even when not drawing)
    agents_out[agent_id].position = agent_pos;
    agents_out[agent_id].velocity = agent_vel;
    agents_out[agent_id].rotation = agent_rot;
    agents_out[agent_id].energy = agent_energy_cur;
    agents_out[agent_id].gene_length = agent_gene_length;
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
fn diffuse_grids_stage1(@builtin(global_invocation_id) gid: vec3<u32>) {
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
    let current_alpha = chem_grid[idx].x;
    let current_beta = chem_grid[idx].y;
    let current_gamma = read_gamma_height(idx);

    // Per-channel strengths.
    // Classic isotropic blur uses alpha_blur/beta_blur.
    // Fluid-directed convolution uses alpha_fluid_convolution/beta_fluid_convolution.
    let alpha_diffuse_strength = clamp(params.alpha_blur, 0.0, 1.0);
    let beta_diffuse_strength = clamp(params.beta_blur, 0.0, 1.0);
    let gamma_diffuse_strength = clamp(params.gamma_diffuse, 0.0, 1.0);
    let alpha_conv_strength = clamp(params.alpha_fluid_convolution, 0.0, 1.0);
    let beta_conv_strength = clamp(params.beta_fluid_convolution, 0.0, 1.0);
    let gamma_strength = clamp(params.gamma_shift, 0.0, 1.0);
    // Repurpose gamma_blur as a simple persistence/decay multiplier (1.0 = no decay).
    let persistence = clamp(params.gamma_blur, 0.0, 1.0);

    // Classic isotropic blur (3x3) + slope-based flux shift.
    // This restores the previous diffusion/blur + slope shift behavior, then we layer
    // the fluid-directed convolution on top.
    var alpha_sum = 0.0;
    var beta_sum = 0.0;
    var gamma_sum = 0.0;
    for (var i = 0u; i < 9u; i++) {
        let dx = i32(i % 3u) - 1;
        let dy = i32(i / 3u) - 1;
        let nx_i = clamp(i32(x) + dx, 0, i32(GRID_SIZE) - 1);
        let ny_i = clamp(i32(y) + dy, 0, i32(GRID_SIZE) - 1);
        let nidx = u32(ny_i) * GRID_SIZE + u32(nx_i);
        alpha_sum += chem_grid[nidx].x;
        beta_sum += chem_grid[nidx].y;
        gamma_sum += read_gamma_height(nidx);
    }

    let alpha_avg = alpha_sum / 9.0;
    let beta_avg = beta_sum / 9.0;
    let gamma_avg = gamma_sum / 9.0;

    // Apply blur factor (0 = no blur/keep current, 1 = full blur)
    let alpha_iso = mix(current_alpha, alpha_avg, alpha_diffuse_strength);
    let beta_iso = mix(current_beta, beta_avg, beta_diffuse_strength);
    let gamma_iso = mix(current_gamma, gamma_avg, gamma_diffuse_strength);

    // Slope-based flux shift (mass-conserving advection along slopes).
    let slope_here = read_gamma_slope(idx);
    let xi = i32(x);
    let yi = i32(y);
    let max_index = i32(GRID_SIZE) - 1;
    let left_x = max(xi - 1, 0);
    let right_x = min(xi + 1, max_index);
    let up_y = max(yi - 1, 0);
    let down_y = min(yi + 1, max_index);

    let left_idx = u32(yi) * GRID_SIZE + u32(left_x);
    let right_idx = u32(yi) * GRID_SIZE + u32(right_x);
    let up_idx = u32(up_y) * GRID_SIZE + x;
    let down_idx = u32(down_y) * GRID_SIZE + x;

    let alpha_left = chem_grid[left_idx].x;
    let alpha_right = chem_grid[right_idx].x;
    let alpha_up = chem_grid[up_idx].x;
    let alpha_down = chem_grid[down_idx].x;
    let beta_left = chem_grid[left_idx].y;
    let beta_right = chem_grid[right_idx].y;
    let beta_up = chem_grid[up_idx].y;
    let beta_down = chem_grid[down_idx].y;

    let slope_left = read_gamma_slope(left_idx);
    let slope_right = read_gamma_slope(right_idx);
    let slope_up = read_gamma_slope(up_idx);
    let slope_down = read_gamma_slope(down_idx);

    let kernel_scale = 1.0 / 8.0;

    // Alpha flux
    let slope_here_alpha = slope_here * params.alpha_slope_bias;
    let slope_left_alpha = slope_left * params.alpha_slope_bias;
    let slope_right_alpha = slope_right * params.alpha_slope_bias;
    let slope_up_alpha = slope_up * params.alpha_slope_bias;
    let slope_down_alpha = slope_down * params.alpha_slope_bias;

    var alpha_flux = 0.0;
    if (right_x != xi) {
        let flow_out = max(slope_here_alpha.x, 0.0) * current_alpha;
        let flow_in = max(-slope_right_alpha.x, 0.0) * alpha_right;
        alpha_flux += flow_out - flow_in;
    }
    if (left_x != xi) {
        let flow_out = max(-slope_here_alpha.x, 0.0) * current_alpha;
        let flow_in = max(slope_left_alpha.x, 0.0) * alpha_left;
        alpha_flux += flow_out - flow_in;
    }
    if (down_y != yi) {
        let flow_out = max(slope_here_alpha.y, 0.0) * current_alpha;
        let flow_in = max(-slope_down_alpha.y, 0.0) * alpha_down;
        alpha_flux += flow_out - flow_in;
    }
    if (up_y != yi) {
        let flow_out = max(-slope_here_alpha.y, 0.0) * current_alpha;
        let flow_in = max(slope_up_alpha.y, 0.0) * alpha_up;
        alpha_flux += flow_out - flow_in;
    }

    // Beta flux
    let slope_here_beta = slope_here * params.beta_slope_bias;
    let slope_left_beta = slope_left * params.beta_slope_bias;
    let slope_right_beta = slope_right * params.beta_slope_bias;
    let slope_up_beta = slope_up * params.beta_slope_bias;
    let slope_down_beta = slope_down * params.beta_slope_bias;

    var beta_flux = 0.0;
    if (right_x != xi) {
        let flow_out = max(slope_here_beta.x, 0.0) * current_beta;
        let flow_in = max(-slope_right_beta.x, 0.0) * beta_right;
        beta_flux += flow_out - flow_in;
    }
    if (left_x != xi) {
        let flow_out = max(-slope_here_beta.x, 0.0) * current_beta;
        let flow_in = max(slope_left_beta.x, 0.0) * beta_left;
        beta_flux += flow_out - flow_in;
    }
    if (down_y != yi) {
        let flow_out = max(slope_here_beta.y, 0.0) * current_beta;
        let flow_in = max(-slope_down_beta.y, 0.0) * beta_down;
        beta_flux += flow_out - flow_in;
    }
    if (up_y != yi) {
        let flow_out = max(-slope_here_beta.y, 0.0) * current_beta;
        let flow_in = max(slope_up_beta.y, 0.0) * beta_up;
        beta_flux += flow_out - flow_in;
    }

    // Base result after classic blur + slope shift.
    let alpha_base = clamp(alpha_iso - alpha_flux * kernel_scale, 0.0, 1.0);
    let beta_base = clamp(beta_iso - beta_flux * kernel_scale, 0.0, 1.0);

    // NOTE: sample_grid_bilinear expects world-space in [0, SIM_SIZE).
    // Fluid-directed convolution has been removed; keep classic blur+slope behavior.
    // We still compute a single local fluid speed sample for the rain logic below.
    let pos_cell = vec2<f32>(f32(x) + 0.5, f32(y) + 0.5);
    let cell_to_world = f32(SIM_SIZE) / f32(GRID_SIZE);
    let pos_world = pos_cell * cell_to_world;

    let raw_v = sample_fluid_velocity_bilinear(pos_world);
    let raw_vlen = length(raw_v);

    // Persistence applies as decay to alpha/beta.
    var final_alpha = clamp(alpha_base * persistence, 0.0, 1.0);
    var final_beta = clamp(beta_base * persistence, 0.0, 1.0);
    // Gamma: classic diffusion only (separate from Env Persistence).
    var final_gamma = max(gamma_iso, 0.0);

    // Stochastic rain - randomly add food/poison droplets (saturated drops)
    // Use position and random seed to generate unique random values per cell
    let cell_seed = idx * 2654435761u + params.random_seed;
    let rain_chance = f32(hash(cell_seed)) / 4294967295.0;

    // Uniform alpha rain (food): remove spatial and beta-dependent gradients.
    // Each cell independently receives a saturated rain event with probability alpha_multiplier * 0.05.
    // (Scaling by 0.05 preserves prior expected value semantics.)
    // Rain factor is read from the dedicated rain map texture (independent of fluid dyes).
    // IMPORTANT: rain_map_tex is stored at environment resolution (GRID_SIZE^2).
    // RG channels: x=alpha, y=beta.
    let rain_packed = textureLoad(rain_map_tex, vec2<i32>(i32(x), i32(y)), 0).xy;
    let alpha_rain_map = clamp(rain_packed.x, 0.0, 1.0);

    // Make it rain more when local velocity magnitude is low (stagnant areas).
    // This is intentionally dt-free and uses a smooth threshold to avoid banding.
    let local_mag = raw_vlen;
    let low_mag = 1.0 - smoothstep(0.02, 0.15, local_mag);
    // Stronger precipitation in slow-flow regions.
    // Shape: quadratic for a gentler mid-range and stronger near-still boost.
    let precip_boost = mix(1.0, 8.0, low_mag * low_mag);

    let alpha_probability_sat = clamp(params.alpha_multiplier * 0.05 * alpha_rain_map * precip_boost, 0.0, 1.0);
    if (rain_chance < alpha_probability_sat) {
        final_alpha = 1.0;  // Saturated drop
    }

    // Uniform beta rain (poison): also no vertical gradient. Probability = beta_multiplier * 0.05.
    let beta_seed = cell_seed * 1103515245u;
    let beta_rain_chance = f32(hash(beta_seed)) / 4294967295.0;
    let beta_rain_map = clamp(rain_packed.y, 0.0, 1.0);
    let beta_probability_sat = clamp(params.beta_multiplier * 0.05 * beta_rain_map * precip_boost, 0.0, 1.0);
    if (beta_rain_chance < beta_probability_sat) {
        final_beta = 1.0;  // Saturated drop
    }

    // IMPORTANT: Do NOT write alpha/beta in-place.
    // This kernel samples neighboring cells (via sample_grid_bilinear), so doing an in-place
    // update introduces cross-invocation races and directionally-biased artifacts.
    // We stage results into trail_grid_inject (a scratch buffer that is overwritten later
    // by the trail preparation pass).
    trail_grid_inject[idx] = vec4<f32>(final_alpha, final_beta, final_gamma, 0.0);
}

@compute @workgroup_size(16, 16)
fn diffuse_grids_stage2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if (x >= GRID_SIZE || y >= GRID_SIZE) {
        return;
    }

    let idx = y * GRID_SIZE + x;
    let staged = trail_grid_inject[idx];
    chem_grid[idx] = vec2<f32>(staged.x, staged.y);
    write_gamma_height(idx, staged.z);
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
    let gradient = vec2<f32>(dx, dy) * inv_cell_size;

    // IMPORTANT: gamma_slope is *terrain-only*.
    // Do NOT bake global vector forces (gravity/wind) into this buffer, because other
    // systems (e.g. fluid permeability/obstacles) interpret slope magnitude as terrain steepness.
    // If you want gravity to affect a particular sensor/force, add it at the usage site.
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
    let current_raw = trail_grid[idx];
    let current = vec4<f32>(
        clamp(current_raw.xyz, vec3<f32>(0.0), vec3<f32>(1.0)),
        sanitize_f32(current_raw.w)
    );
    let strength = clamp(params.trail_diffusion, 0.0, 1.0);
    let decay = clamp(params.trail_decay, 0.0, 1.0);

    let xi = i32(x);
    let yi = i32(y);
    let x_l = u32(clamp(xi - 1, 0, i32(GRID_SIZE) - 1));
    let x_r = u32(clamp(xi + 1, 0, i32(GRID_SIZE) - 1));
    let y_u = u32(clamp(yi - 1, 0, i32(GRID_SIZE) - 1));
    let y_d = u32(clamp(yi + 1, 0, i32(GRID_SIZE) - 1));

    let l_raw = trail_grid[y * GRID_SIZE + x_l];
    let r_raw = trail_grid[y * GRID_SIZE + x_r];
    let u_raw = trail_grid[y_u * GRID_SIZE + x];
    let d_raw = trail_grid[y_d * GRID_SIZE + x];
    let l = vec4<f32>(clamp(l_raw.xyz, vec3<f32>(0.0), vec3<f32>(1.0)), sanitize_f32(l_raw.w));
    let r = vec4<f32>(clamp(r_raw.xyz, vec3<f32>(0.0), vec3<f32>(1.0)), sanitize_f32(r_raw.w));
    let u = vec4<f32>(clamp(u_raw.xyz, vec3<f32>(0.0), vec3<f32>(1.0)), sanitize_f32(u_raw.w));
    let d = vec4<f32>(clamp(d_raw.xyz, vec3<f32>(0.0), vec3<f32>(1.0)), sanitize_f32(d_raw.w));
    let neighbor_blur = (l + r + u + d) * 0.25;

    let mixed = mix(current, neighbor_blur, strength);
    let faded_rgb = clamp(mixed.xyz * decay, vec3<f32>(0.0), vec3<f32>(1.0));
    let faded_energy = sanitize_f32(mixed.w) * decay;
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
        gamma = max(sample_grid_bicubic(world_pos, 2u), 0.0);
    } else if (params.grid_interpolation == 1u) {
        // Bilinear (smooth)
        alpha = clamp(sample_grid_bilinear(world_pos, 0u), 0.0, 1.0);
        beta = clamp(sample_grid_bilinear(world_pos, 1u), 0.0, 1.0);
        gamma = max(sample_grid_bilinear(world_pos, 2u), 0.0);
    } else {
        // Nearest neighbor (pixelated)
        let c = chem_grid[grid_index(world_pos)];
        alpha = clamp(c.x, 0.0, 1.0);
        beta = clamp(c.y, 0.0, 1.0);
        gamma = max(read_gamma_height(grid_index(world_pos)), 0.0);
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
        let current_stamp = params.epoch + 1u;
        let scale = f32(SPATIAL_GRID_SIZE) / f32(SIM_SIZE);
        let grid_x = u32(clamp(world_x * scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));
        let grid_y = u32(clamp(world_y * scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));
        let grid_cell = grid_y * SPATIAL_GRID_SIZE + grid_x;

        // Skip stale cells from prior epochs.
        let stamp = atomicLoad(&agent_spatial_grid[spatial_epoch_index(grid_cell)]);
        if (stamp != current_stamp) {
            return;
        }
        let raw_agent_id = atomicLoad(&agent_spatial_grid[spatial_id_index(grid_cell)]);
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

    // ====== TRAIL OVERLAY ======
    // trail_grid stores RGB dye trail + energy trail in W (unclamped).
    let trail_cell = trail_grid[grid_index(world_pos)];
    let trail_color = clamp(trail_cell.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    let trail_energy = max(trail_cell.w, 0.0);

    // Trail visualization modes:
    // 0 = normal (add RGB trails onto base)
    // 1 = trails-only (RGB)
    // 2 = trails-only (energy)
    if (params.trail_show == 1u) {
        let trail_only = trail_color * clamp(params.trail_opacity, 0.0, 1.0);
        visual_grid[visual_idx] = vec4<f32>(trail_only, 1.0);
    } else if (params.trail_show == 2u) {
        // Energy is stored as an unclamped accumulator.
        // Do NOT clamp the opacity gain here (only clamp output via tone mapping).
        let w_raw = trail_cell.w;
        // WGSL portability: avoid isNan/isInf (not available on all validator versions).
        let w_is_nan = w_raw != w_raw;
        let w_is_inf = abs(w_raw) > 1.0e30;
        if (w_is_nan || w_is_inf) {
            // Bright magenta = bad/invalid energy values.
            visual_grid[visual_idx] = vec4<f32>(1.0, 0.0, 1.0, 1.0);
        } else {
            let exposure = max(params.trail_opacity, 0.0);
            let e = max(w_raw, 0.0);
            let x = sqrt(e) * exposure;
            // Exposure mapping: makes tiny values visible without hard clamping.
            let v = 1.0 - exp(-x);
            visual_grid[visual_idx] = vec4<f32>(vec3<f32>(v), 1.0);
        }
    } else {
        let blended_color = clamp(
            base_color + trail_color * clamp(params.trail_opacity, 0.0, 1.0),
            vec3<f32>(0.0),
            vec3<f32>(1.0)
        );
        visual_grid[visual_idx] = vec4<f32>(blended_color, 1.0);
    }
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct InspectorOverlayVertexOutput {
    @builtin(position) position: vec4<f32>,
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

@vertex
fn vs_inspector_overlay(@builtin(vertex_index) vid: u32) -> InspectorOverlayVertexOutput {
    // Quad covering only the inspector area (rightmost INSPECTOR_WIDTH pixels)
    let safe_width = max(params.window_width, 1.0);
    let inspector_w = f32(INSPECTOR_WIDTH);
    let x0 = 1.0 - 2.0 * (inspector_w / safe_width);

    var positions = array<vec2<f32>, 6>(
        vec2<f32>(x0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(x0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(x0, 1.0)
    );

    let pos = positions[vid];
    var out: InspectorOverlayVertexOutput;
    out.position = vec4<f32>(pos, 0.0, 1.0);
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

@fragment
fn fs_inspector_overlay(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    if (params.draw_enabled == 0u) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }
    if (params.selected_agent_index == 0xFFFFFFFFu) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }

    let safe_width = max(params.window_width, 1.0);
    let safe_height = max(params.window_height, 1.0);
    let window_width = u32(safe_width);
    let window_height = u32(safe_height);

    let x = u32(pos.x);
    let y = u32(pos.y);
    if (y >= window_height) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Map screen x into inspector-local x
    let start_x = select(window_width - INSPECTOR_WIDTH, 0u, window_width < INSPECTOR_WIDTH);
    if (x < start_x) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let local_x = x - start_x;

    return inspector_panel_pixel(local_x, y);
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
    chem_grid[idx] = vec2<f32>(environment_init.alpha_range.x, environment_init.beta_range.x);
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
    // Clamp spawn count to buffer size (2000) to avoid OOB reads if more agents tried to spawn than fit
    let spawn_count = min(atomicLoad(&spawn_debug_counters[0]), 2000u);

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

    // If flags bit 0 set, use provided genome override (packed)
    if ((request.flags & 1u) != 0u) {
        agent.gene_length = request.genome_override_len;
        agent.genome_offset = request.genome_override_offset;
        // Manual unroll: some backends (via naga) require constant indexing into
        // fixed-size arrays inside storage-structs.
        agent.genome_packed[0u] = request.genome_override_packed[0u];
        agent.genome_packed[1u] = request.genome_override_packed[1u];
        agent.genome_packed[2u] = request.genome_override_packed[2u];
        agent.genome_packed[3u] = request.genome_override_packed[3u];
        agent.genome_packed[4u] = request.genome_override_packed[4u];
        agent.genome_packed[5u] = request.genome_override_packed[5u];
        agent.genome_packed[6u] = request.genome_override_packed[6u];
        agent.genome_packed[7u] = request.genome_override_packed[7u];
        agent.genome_packed[8u] = request.genome_override_packed[8u];
        agent.genome_packed[9u] = request.genome_override_packed[9u];
        agent.genome_packed[10u] = request.genome_override_packed[10u];
        agent.genome_packed[11u] = request.genome_override_packed[11u];
        agent.genome_packed[12u] = request.genome_override_packed[12u];
        agent.genome_packed[13u] = request.genome_override_packed[13u];
        agent.genome_packed[14u] = request.genome_override_packed[14u];
        agent.genome_packed[15u] = request.genome_override_packed[15u];
    } else {
    // Create centered variable-length genome with implicit 'X' padding on both sides.
    // Active region length in [MIN_GENE_LENGTH, GENOME_LENGTH].
        genome_seed = hash(genome_seed ^ base_seed);
        let gene_span = GENOME_LENGTH - MIN_GENE_LENGTH;
        let gene_len = MIN_GENE_LENGTH + (hash(genome_seed) % (gene_span + 1u));
        let left_pad = (GENOME_LENGTH - gene_len) / 2u;

        agent.gene_length = gene_len;
        agent.genome_offset = left_pad;
        for (var w = 0u; w < GENOME_PACKED_WORDS; w++) {
            agent.genome_packed[w] = 0u;
        }

        for (var k = 0u; k < gene_len; k++) {
            genome_seed = hash(genome_seed ^ (k * 1664525u + 1013904223u));
            let base_ascii = get_random_rna_base(genome_seed);
            let code = genome_base_ascii_to_2bit(base_ascii);
            let bi = left_pad + k;
            let wi = bi / GENOME_BASES_PER_PACKED_WORD;
            let bit_index = (bi % GENOME_BASES_PER_PACKED_WORD) * 2u;
            agent.genome_packed[wi] = agent.genome_packed[wi] | (code << bit_index);
        }
    }

    for (var i = 0u; i < MAX_BODY_PARTS; i++) {
        agent.body[i].pos = vec2<f32>(0.0);
        agent.body[i].data = 0.0;
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
    let alive_total = atomicLoad(&spawn_debug_counters[2]);
    let max_agents = params.max_agents;
    if (alive_total >= max_agents) {
        return;
    }

    // This kernel is dispatched only for the dead tail; gid.x is relative to `alive_total`.
    let idx = alive_total + gid.x;
    if (idx >= max_agents) {
        return;
    }

    // IMPORTANT: Avoid copying the full Agent struct (genome + body arrays).
    // We only need to ensure a few fields are reset for dead slots.
    agents_out[idx].alive = 0u;
    agents_out[idx].body_count = 0u;
    agents_out[idx].energy = 0.0;
    agents_out[idx].velocity = vec2<f32>(0.0);
    agents_out[idx].pairing_counter = 0u;
}

// Compact living agents from input to output, producing a packed array at the front
@compute @workgroup_size(64)
fn compact_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent_id = gid.x;

    // IMPORTANT:
    // Compact must cover a bit beyond `params.agent_count` to avoid dropping newborns
    // when CPU-side `agent_count` readback lags by a frame.
    // However, scanning the full `params.max_agents` would also pick up stale/garbage tail
    // slots (e.g. after loading a snapshot with fewer agents), which can cause runaway
    // "ghost" agents and continuous reproduction.
    let scan_limit = min(params.agent_count + 2000u, params.max_agents);
    if (agent_id >= scan_limit) {
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
        let prev = chem_grid[idx];
        chem_grid[idx] = vec2<f32>(output_value, prev.y);
    } else if (mode == 2u) {
        let prev = chem_grid[idx];
        chem_grid[idx] = vec2<f32>(prev.x, output_value);
    } else if (mode == 3u) {
        write_gamma_height(idx, output_value);
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

    let cell = y * SPATIAL_GRID_SIZE + x;
    atomicStore(&agent_spatial_grid[spatial_id_index(cell)], SPATIAL_GRID_EMPTY);
    atomicStore(&agent_spatial_grid[spatial_epoch_index(cell)], 0u);
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
    let primary_cell = grid_y * SPATIAL_GRID_SIZE + grid_x;

    // Stamp is epoch+1 so 0 remains the "never written" value.
    let current_stamp = params.epoch + 1u;

    // IMPORTANT: the spatial grid stores stamp and id in two different atomics.
    // If we write the stamp first, readers can see `stamp == current_stamp` and
    // still read a stale id (from a previous epoch) before the id store lands.
    // To avoid that, we do a two-phase stamp:
    //   1) CAS stamp -> (current_stamp | IN_PROGRESS_BIT)
    //   2) store id
    //   3) store stamp -> current_stamp
    // Readers only accept `stamp == current_stamp`, so they ignore in-progress cells.
    let IN_PROGRESS_BIT: u32 = 0x80000000u;
    let STAMP_EPOCH_MASK: u32 = 0x7FFFFFFFu;

    // Try to claim the primary cell via epoch stamp.
    let primary_old_stamp_raw = atomicLoad(&agent_spatial_grid[spatial_epoch_index(primary_cell)]);
    var found = false;
    if ((primary_old_stamp_raw & STAMP_EPOCH_MASK) != current_stamp) {
        let primary_claim = atomicCompareExchangeWeak(
            &agent_spatial_grid[spatial_epoch_index(primary_cell)],
            primary_old_stamp_raw,
            current_stamp | IN_PROGRESS_BIT
        );
        if (primary_claim.exchanged) {
            atomicStore(&agent_spatial_grid[spatial_id_index(primary_cell)], agent_id);
            atomicStore(&agent_spatial_grid[spatial_epoch_index(primary_cell)], current_stamp);
            found = true;
        }
    }

    if (!found) {
        // Primary cell is occupied - search for nearest empty cell in a spiral pattern
        // This ensures all agents are findable even in crowded areas
        // found is already declared above.

        // Search in expanding square rings up to radius 5 (covers 11x11 area = 121 cells)
        for (var radius = 1u; radius <= 5u && !found; radius++) {
            // Top and bottom edges of the square
            for (var dx: i32 = -i32(radius); dx <= i32(radius) && !found; dx++) {
                // Top edge
                let check_x_top = i32(grid_x) + dx;
                let check_y_top = i32(grid_y) - i32(radius);
                if (check_x_top >= 0 && check_x_top < i32(SPATIAL_GRID_SIZE) &&
                    check_y_top >= 0 && check_y_top < i32(SPATIAL_GRID_SIZE)) {
                    let cell = u32(check_y_top) * SPATIAL_GRID_SIZE + u32(check_x_top);
                    let old_stamp_raw = atomicLoad(&agent_spatial_grid[spatial_epoch_index(cell)]);
                    if ((old_stamp_raw & STAMP_EPOCH_MASK) != current_stamp) {
                        let claim = atomicCompareExchangeWeak(
                            &agent_spatial_grid[spatial_epoch_index(cell)],
                            old_stamp_raw,
                            current_stamp | IN_PROGRESS_BIT
                        );
                        if (claim.exchanged) {
                            atomicStore(&agent_spatial_grid[spatial_id_index(cell)], agent_id);
                            atomicStore(&agent_spatial_grid[spatial_epoch_index(cell)], current_stamp);
                            found = true;
                        }
                    }
                }

                // Bottom edge (skip if radius == 0 to avoid duplicate)
                if (!found && radius > 0u) {
                    let check_x_bot = i32(grid_x) + dx;
                    let check_y_bot = i32(grid_y) + i32(radius);
                    if (check_x_bot >= 0 && check_x_bot < i32(SPATIAL_GRID_SIZE) &&
                        check_y_bot >= 0 && check_y_bot < i32(SPATIAL_GRID_SIZE)) {
                        let cell = u32(check_y_bot) * SPATIAL_GRID_SIZE + u32(check_x_bot);
                        let old_stamp_raw = atomicLoad(&agent_spatial_grid[spatial_epoch_index(cell)]);
                        if ((old_stamp_raw & STAMP_EPOCH_MASK) != current_stamp) {
                            let claim = atomicCompareExchangeWeak(
                                &agent_spatial_grid[spatial_epoch_index(cell)],
                                old_stamp_raw,
                                current_stamp | IN_PROGRESS_BIT
                            );
                            if (claim.exchanged) {
                                atomicStore(&agent_spatial_grid[spatial_id_index(cell)], agent_id);
                                atomicStore(&agent_spatial_grid[spatial_epoch_index(cell)], current_stamp);
                                found = true;
                            }
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
                    let cell = u32(check_y_left) * SPATIAL_GRID_SIZE + u32(check_x_left);
                    let old_stamp_raw = atomicLoad(&agent_spatial_grid[spatial_epoch_index(cell)]);
                    if ((old_stamp_raw & STAMP_EPOCH_MASK) != current_stamp) {
                        let claim = atomicCompareExchangeWeak(
                            &agent_spatial_grid[spatial_epoch_index(cell)],
                            old_stamp_raw,
                            current_stamp | IN_PROGRESS_BIT
                        );
                        if (claim.exchanged) {
                            atomicStore(&agent_spatial_grid[spatial_id_index(cell)], agent_id);
                            atomicStore(&agent_spatial_grid[spatial_epoch_index(cell)], current_stamp);
                            found = true;
                        }
                    }
                }

                // Right edge
                if (!found) {
                    let check_x_right = i32(grid_x) + i32(radius);
                    let check_y_right = i32(grid_y) + dy;
                    if (check_x_right >= 0 && check_x_right < i32(SPATIAL_GRID_SIZE) &&
                        check_y_right >= 0 && check_y_right < i32(SPATIAL_GRID_SIZE)) {
                        let cell = u32(check_y_right) * SPATIAL_GRID_SIZE + u32(check_x_right);
                        let old_stamp_raw = atomicLoad(&agent_spatial_grid[spatial_epoch_index(cell)]);
                        if ((old_stamp_raw & STAMP_EPOCH_MASK) != current_stamp) {
                            let claim = atomicCompareExchangeWeak(
                                &agent_spatial_grid[spatial_epoch_index(cell)],
                                old_stamp_raw,
                                current_stamp | IN_PROGRESS_BIT
                            );
                            if (claim.exchanged) {
                                atomicStore(&agent_spatial_grid[spatial_id_index(cell)], agent_id);
                                atomicStore(&agent_spatial_grid[spatial_epoch_index(cell)], current_stamp);
                                found = true;
                            }
                        }
                    }
                }
            }
        }

        // If still not found after searching 5 rings, agent won't be in spatial grid this frame
        // This is acceptable as it will retry next frame - prevents infinite loops
    }
}

