const MAX_SPAWN_REQUESTS: u32 = 2048u;

@compute @workgroup_size(256)
fn reproduce_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent_id = gid.x;
    if (agent_id >= params.agent_count) {
        return;
    }

    // Read from agents_out (which is the INPUT buffer, same as agents_in in this bind group)
    // Reproduction runs BEFORE process_agents, so we read the clean input buffer
    // and write pairing/energy updates back to it.
    var agent = agents_out[agent_id];

    if (agent.alive == 0u) {
        return;
    }

    // ====== REPRODUCTION LOGIC ======
    // NOTE: Pairing counter advancement happens in process_agents (simulation.wgsl).
    // This shader only checks if pairing is complete and spawns offspring.

    let gene_length = agent.gene_length;
    let genome_offset = agent.genome_offset;
    let agent_genome_packed = agent.genome_packed;
    let pairing_counter = agent.pairing_counter;
    var agent_energy_cur = agent.energy;
    let agent_generation = agent.generation;
    let agent_pos = agent.position;

    // Better RNG using hash function with time and agent variation
    let hash_base = (agent_id + params.random_seed) * 747796405u + 2891336453u;
    let hash2 = hash_base ^ (hash_base >> 13u);
    let hash3 = hash2 * 1103515245u;

    // Check if pairing is complete and ready to spawn
    if (pairing_counter >= gene_length && gene_length > 0u) {
        // Attempt reproduction: create complementary genome offspring with mutations
        let inherited_energy = agent_energy_cur * 0.5;
        if (inherited_energy > 0.0) {
            // Atomically reserve a spawn slot
            // spawn_debug_counters[0] is reset to 0 at frame start and tracks total spawns
            let spawn_index = atomicAdd(&spawn_debug_counters[0], 1u);

            // Only proceed if we have room in the spawn request buffer
            if (spawn_index < MAX_SPAWN_REQUESTS) {
                // Generate hash for offspring randomization
                let offspring_hash = (hash3 ^ (spawn_index * 0x9e3779b9u) ^ (agent_id * 0x85ebca6bu)) * 1664525u + 1013904223u;

                // Create brand new offspring agent (don't copy parent)
                var offspring: Agent;

                // Random rotation
                offspring.rotation = hash_f32(offspring_hash) * 6.28318530718;

                // Spawn near parent with a small jitter
                {
                    let jitter_angle = hash_f32(offspring_hash ^ 0xBADC0FFEu) * 6.28318530718;
                    let jitter_dist = 5.0 + hash_f32(offspring_hash ^ 0x1B56C4E9u) * 10.0;
                    let jitter = vec2<f32>(cos(jitter_angle), sin(jitter_angle)) * jitter_dist;
                    offspring.position = clamp_position(agent_pos + jitter);
                }
                offspring.velocity = vec2<f32>(0.0);

                // Initialize offspring energy; final value assigned after viability check
                offspring.energy = 0.0;

                offspring.energy_capacity = 0.0;
                offspring.torque_debug = 0.0;
                offspring.morphology_origin = vec2<f32>(0.0);

                // Initialize as alive, will build body on first frame
                offspring.alive = 1u;
                offspring.body_count = 0u; // Forces morphology rebuild
                offspring.pairing_counter = 0u;
                offspring.is_selected = 0u;
                // Lineage and lifecycle
                offspring.generation = agent_generation + 1u;
                offspring.age = 0u;
                offspring.total_mass = 0.0;
                offspring.poison_resistant_count = 0u;

                // Default genome metadata.
                offspring.gene_length = 0u;
                offspring.genome_offset = 0u;
                for (var w = 0u; w < GENOME_PACKED_WORDS; w++) {
                    offspring.genome_packed[w] = 0u;
                }

                // Child genome: materialize to a temporary ASCII buffer
                var offspring_ascii: array<u32, GENOME_ASCII_WORDS>;
                if (params.asexual_reproduction == 1u) {
                    // Asexual reproduction: direct copy of bases.
                    for (var w = 0u; w < GENOME_ASCII_WORDS; w++) {
                        let bi0 = w * 4u + 0u;
                        let bi1 = w * 4u + 1u;
                        let bi2 = w * 4u + 2u;
                        let bi3 = w * 4u + 3u;
                        let b0 = genome_get_base_ascii(agent_genome_packed, bi0, genome_offset, gene_length) & 0xFFu;
                        let b1 = genome_get_base_ascii(agent_genome_packed, bi1, genome_offset, gene_length) & 0xFFu;
                        let b2 = genome_get_base_ascii(agent_genome_packed, bi2, genome_offset, gene_length) & 0xFFu;
                        let b3 = genome_get_base_ascii(agent_genome_packed, bi3, genome_offset, gene_length) & 0xFFu;
                        offspring_ascii[w] = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
                    }
                } else {
                    // Sexual reproduction: reverse complement of parent
                    for (var w = 0u; w < GENOME_ASCII_WORDS; w++) {
                        offspring_ascii[w] = genome_revcomp_ascii_word(agent_genome_packed, genome_offset, gene_length, w);
                    }
                }

                // Sample beta concentration at parent's location to calculate radiation-induced mutation rate
                let parent_idx = grid_index(agent_pos);
                let beta_concentration = chem_grid[parent_idx].y;
                let beta_normalized = clamp(beta_concentration, 0.0, 1.0);
                let mutation_multiplier = 1.0 + pow(beta_normalized, 3.0) * 4.0;
                var effective_mutation_rate = params.mutation_rate * mutation_multiplier;
                effective_mutation_rate = min(effective_mutation_rate, 1.0);

                // Determine active gene region (non-'X' bytes) in offspring after reverse complement
                var first_non_x: u32 = GENOME_LENGTH;
                var last_non_x: u32 = 0xFFFFFFFFu;
                for (var bi = 0u; bi < GENOME_LENGTH; bi++) {
                    let word = bi / 4u;
                    let byte_offset = bi % 4u;
                    let b = (offspring_ascii[word] >> (byte_offset * 8u)) & 0xFFu;
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

                // Optional insertion mutation
                {
                    let insert_seed = offspring_hash ^ 0xB5297A4Du;
                    let insert_roll = hash_f32(insert_seed);
                    let can_insert = (last_non_x != 0xFFFFFFFFu);
                    if (can_insert && insert_roll < (effective_mutation_rate * 0.20)) {
                        var seq: array<u32, GENOME_LENGTH>;
                        var L: u32 = 0u;
                        for (var bi = active_start; bi <= active_end; bi++) {
                            if (L < GENOME_LENGTH) {
                                let word = bi / 4u;
                                let byte_offset = bi % 4u;
                                seq[L] = (offspring_ascii[word] >> (byte_offset * 8u)) & 0xFFu;
                                L += 1u;
                            }
                        }
                        let max_ins = select(GENOME_LENGTH - L, 0u, L >= GENOME_LENGTH);
                        let k = 3u;
                        if (max_ins >= k) {
                            let mode = hash(insert_seed ^ 0x1B56C4E9u) % 3u;
                            var pos: u32 = 0u;
                            if (mode == 0u) { pos = 0u; }
                            else if (mode == 1u) { pos = L; }
                            else { pos = hash(insert_seed ^ 0x2C9F85A1u) % (L + 1u); }
                            var j: i32 = i32(L);
                            loop {
                                j = j - 1;
                                if (j < i32(pos)) { break; }
                                seq[u32(j) + k] = seq[u32(j)];
                            }
                            for (var t = 0u; t < k; t++) {
                                let nb = get_random_rna_base(insert_seed ^ (t * 1664525u + 1013904223u));
                                seq[pos + t] = nb;
                            }
                            L = min(GENOME_LENGTH, L + k);
                            var out_bytes: array<u32, GENOME_LENGTH>;
                            for (var t = 0u; t < GENOME_LENGTH; t++) { out_bytes[t] = 88u; }
                            let left_pad = (GENOME_LENGTH - L) / 2u;
                            for (var t = 0u; t < L; t++) {
                                out_bytes[left_pad + t] = seq[t];
                            }
                            for (var w = 0u; w < GENOME_ASCII_WORDS; w++) {
                                let b0 = out_bytes[w * 4u + 0u] & 0xFFu;
                                let b1 = out_bytes[w * 4u + 1u] & 0xFFu;
                                let b2 = out_bytes[w * 4u + 2u] & 0xFFu;
                                let b3 = out_bytes[w * 4u + 3u] & 0xFFu;
                                let word_val = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
                                offspring_ascii[w] = word_val;
                            }
                            active_start = left_pad;
                            active_end = left_pad + L - 1u;
                        }
                    }
                }

                // Optional deletion mutation
                {
                    let delete_seed = offspring_hash ^ 0xE7037ED1u;
                    let delete_roll = hash_f32(delete_seed);
                    let has_active = (active_end != 0xFFFFFFFFu);
                    if (has_active && delete_roll < (effective_mutation_rate * 0.35)) {
                        var seq: array<u32, GENOME_LENGTH>;
                        var L: u32 = 0u;
                        for (var bi = active_start; bi <= active_end; bi++) {
                            if (L < GENOME_LENGTH) {
                                let word = bi / 4u;
                                let byte_offset = bi % 4u;
                                seq[L] = (offspring_ascii[word] >> (byte_offset * 8u)) & 0xFFu;
                                L += 1u;
                            }
                        }
                        if (L > MIN_GENE_LENGTH) {
                            let removable = L - MIN_GENE_LENGTH;
                            let k = 3u;
                            if (removable >= k) {
                                var pos: u32 = 0u;
                                let mode = hash(delete_seed ^ 0x1B56C4E9u) % 3u;
                                if (mode == 0u) { pos = 0u; }
                                else if (mode == 1u) { pos = L - k; }
                                else { pos = hash(delete_seed ^ 0x2C9F85A1u) % (L - k + 1u); }
                                var j = pos;
                                loop {
                                    if (j + k >= L) { break; }
                                    seq[j] = seq[j + k];
                                    j = j + 1u;
                                }
                                L = L - k;
                                var out_bytes: array<u32, GENOME_LENGTH>;
                                for (var t = 0u; t < GENOME_LENGTH; t++) { out_bytes[t] = 88u; }
                                let left_pad = (GENOME_LENGTH - L) / 2u;
                                for (var t = 0u; t < L; t++) {
                                    out_bytes[left_pad + t] = seq[t];
                                }
                                for (var w = 0u; w < GENOME_ASCII_WORDS; w++) {
                                    let b0 = out_bytes[w * 4u + 0u] & 0xFFu;
                                    let b1 = out_bytes[w * 4u + 1u] & 0xFFu;
                                    let b2 = out_bytes[w * 4u + 2u] & 0xFFu;
                                    let b3 = out_bytes[w * 4u + 3u] & 0xFFu;
                                    let word_val = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
                                    offspring_ascii[w] = word_val;
                                }
                                active_start = left_pad;
                                active_end = left_pad + L - 1u;
                            }
                        }
                    }
                }

                // Probabilistic point mutations
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
                            let current_word = offspring_ascii[word];
                            let updated_word = (current_word & mask) | (new_base << (byte_offset * 8u));
                            offspring_ascii[word] = updated_word;
                            mutated_count += 1u;
                        }
                    }
                }

                // Compute and cache gene_length once for the offspring.
                if (active_end != 0xFFFFFFFFu) {
                    offspring.gene_length = active_end - active_start + 1u;
                    offspring.genome_offset = active_start;
                } else {
                    offspring.gene_length = 0u;
                    offspring.genome_offset = 0u;
                }

                // Pack ASCII offspring genome into 2-bit packed representation.
                {
                    var packed: array<u32, GENOME_PACKED_WORDS>;
                    for (var w = 0u; w < GENOME_PACKED_WORDS; w++) {
                        packed[w] = 0u;
                    }
                    for (var bi = 0u; bi < GENOME_LENGTH; bi++) {
                        let word_ascii = bi / 4u;
                        let byte_offset = bi % 4u;
                        let b = (offspring_ascii[word_ascii] >> (byte_offset * 8u)) & 0xFFu;
                        if (b != 88u) {
                            let code = genome_base_ascii_to_2bit(b);
                            let wi = bi / GENOME_BASES_PER_PACKED_WORD;
                            let bit_index = (bi % GENOME_BASES_PER_PACKED_WORD) * 2u;
                            packed[wi] = packed[wi] | (code << bit_index);
                        }
                    }
                    for (var w = 0u; w < GENOME_PACKED_WORDS; w++) {
                        offspring.genome_packed[w] = packed[w];
                    }
                }

                // Offspring receives 50% of parent's current energy.
                offspring.energy = inherited_energy;
                agent_energy_cur -= inherited_energy;

                // Initialize body array to zeros
                for (var bi = 0u; bi < MAX_BODY_PARTS; bi++) {
                    offspring.body[bi].pos = vec2<f32>(0.0);
                    offspring.body[bi].data = 0.0;
                    offspring.body[bi].part_type = 0u;
                    offspring.body[bi].alpha_signal = 0.0;
                    offspring.body[bi].beta_signal = 0.0;
                    offspring.body[bi]._pad.x = bitcast<f32>(0u);
                    offspring.body[bi]._pad = vec2<f32>(0.0);
                }

                new_agents[spawn_index] = offspring;

                // Update agent state in the INPUT buffer after successful spawn
                // Energy was already deducted when creating offspring
                agent.energy = agent_energy_cur;
                agent.pairing_counter = 0u;  // Reset after successful spawn
                agents_out[agent_id] = agent;
            }
        }
    }
}
