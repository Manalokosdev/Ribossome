const MAX_SPAWN_REQUESTS: u32 = 2048u;

@compute @workgroup_size(256)
fn reproduce_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent_id = gid.x;
    if (agent_id >= params.agent_count) {
        return;
    }

    var agent = agents_out[agent_id];

    if (agent.alive == 0u) {
        return;
    }

    let gene_length = agent.gene_length;
    let genome_offset = agent.genome_offset;
    let agent_genome_packed = agent.genome_packed;
    let pairing_counter = agent.pairing_counter;
    var agent_energy_cur = agent.energy;
    let agent_generation = agent.generation;
    let agent_pos = agent.position;

    let hash_base = (agent_id + params.random_seed) * 747796405u + 2891336453u;
    let hash2 = hash_base ^ (hash_base >> 13u);
    let hash3 = hash2 * 1103515245u;

    if (pairing_counter >= gene_length && gene_length > 0u) {
        var split_reproduction = false;
        var first_gene_end: u32 = 0xFFFFFFFFu;
        var second_gene_start: u32 = 0xFFFFFFFFu;

        if (params.require_start_codon == 1u) {
            let first_start = genome_find_start_codon(agent_genome_packed, genome_offset, gene_length);
            if (first_start != 0xFFFFFFFFu) {
                var scan_pos = first_start + 3u;
                var found_stop = false;

                for (var i = 0u; i < MAX_BODY_PARTS && scan_pos + 2u < GENOME_LENGTH; i++) {
                    if (genome_is_stop_codon_at(agent_genome_packed, scan_pos, genome_offset, gene_length)) {
                        found_stop = true;
                        first_gene_end = scan_pos;
                        break;
                    }
                    scan_pos += 3u;
                }

                if (found_stop) {
                    scan_pos = first_gene_end + 3u;
                    let gene_end = genome_offset + gene_length;
                    for (var j = scan_pos; j + 2u < gene_end; j += 3u) {
                        let b0 = genome_get_base_ascii(agent_genome_packed, j, genome_offset, gene_length);
                        let b1 = genome_get_base_ascii(agent_genome_packed, j + 1u, genome_offset, gene_length);
                        let b2 = genome_get_base_ascii(agent_genome_packed, j + 2u, genome_offset, gene_length);
                        if (b0 == 65u && b1 == 85u && b2 == 71u) {
                            second_gene_start = j;
                            split_reproduction = true;
                            break;
                        }
                    }
                }
            }
        }

        let num_offspring = select(1u, 2u, split_reproduction);

        let inherited_energy = agent_energy_cur * 0.5;
        if (inherited_energy > 0.0) {
            let spawn_index = atomicAdd(&spawn_debug_counters[0], num_offspring);

            for (var offspring_idx = 0u; offspring_idx < num_offspring; offspring_idx++) {
                let current_spawn_index = spawn_index + offspring_idx;
                if (current_spawn_index >= MAX_SPAWN_REQUESTS) {
                    break;
                }

                var offspring_gene_offset = genome_offset;
                var offspring_gene_length = gene_length;

                if (split_reproduction) {
                    if (offspring_idx == 0u) {
                        offspring_gene_length = first_gene_end - genome_offset;
                    } else {
                        offspring_gene_offset = second_gene_start;
                        offspring_gene_length = (genome_offset + gene_length) - second_gene_start;
                    }
                }

                let offspring_hash = (hash3 ^ (current_spawn_index * 0x9e3779b9u) ^ (agent_id * 0x85ebca6bu) ^ (offspring_idx * 0x7f4a7c13u)) * 1664525u + 1013904223u;

                var offspring: Agent;

                offspring.rotation = hash_f32(offspring_hash) * 6.28318530718;

                {
                    let jitter_angle = hash_f32(offspring_hash ^ 0xBADC0FFEu) * 6.28318530718;
                    let jitter_dist = 5.0 + hash_f32(offspring_hash ^ 0x1B56C4E9u) * 10.0;
                    let jitter = vec2<f32>(cos(jitter_angle), sin(jitter_angle)) * jitter_dist;
                    offspring.position = clamp_position(agent_pos + jitter);
                }
                offspring.velocity = vec2<f32>(0.0);

                offspring.energy = 0.0;
                offspring.energy_capacity = 0.0;
                offspring.torque_debug = 0.0;
                offspring.morphology_origin = vec2<f32>(0.0);

                offspring.alive = 1u;
                offspring.body_count = 0u;
                offspring.pairing_counter = 0u;
                offspring.is_selected = 0u;
                offspring.generation = agent_generation + 1u;
                offspring.age = 0u;
                offspring.total_mass = 0.0;
                offspring.poison_resistant_count = 0u;

                offspring.gene_length = 0u;
                offspring.genome_offset = 0u;
                for (var w = 0u; w < GENOME_PACKED_WORDS; w++) {
                    offspring.genome_packed[w] = 0u;
                }

                var offspring_ascii: array<u32, GENOME_ASCII_WORDS>;
                if (params.asexual_reproduction == 1u) {
                    for (var w = 0u; w < GENOME_ASCII_WORDS; w++) {
                        let bi0 = w * 4u + 0u;
                        let bi1 = w * 4u + 1u;
                        let bi2 = w * 4u + 2u;
                        let bi3 = w * 4u + 3u;
                        let b0 = genome_get_base_ascii(agent_genome_packed, bi0, offspring_gene_offset, offspring_gene_length) & 0xFFu;
                        let b1 = genome_get_base_ascii(agent_genome_packed, bi1, offspring_gene_offset, offspring_gene_length) & 0xFFu;
                        let b2 = genome_get_base_ascii(agent_genome_packed, bi2, offspring_gene_offset, offspring_gene_length) & 0xFFu;
                        let b3 = genome_get_base_ascii(agent_genome_packed, bi3, offspring_gene_offset, offspring_gene_length) & 0xFFu;
                        offspring_ascii[w] = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
                    }
                } else {
                    for (var w = 0u; w < GENOME_ASCII_WORDS; w++) {
                        offspring_ascii[w] = genome_revcomp_ascii_word(agent_genome_packed, offspring_gene_offset, offspring_gene_length, w);
                    }
                }

                let mutation_multiplier = 1.0;
                let base_mut_rate = sanitize_f32(params.mutation_rate);
                var effective_mutation_rate = select(base_mut_rate * mutation_multiplier, 0.0, base_mut_rate == 0.0);

                {
                    var mp_count = 0u;
                    let bc = min(agent.body_count, MAX_BODY_PARTS);
                    for (var i = 0u; i < bc; i++) {
                        let base_type = get_base_part_type(agent.body[i].part_type);
                        if (base_type == 43u) {
                            mp_count += 1u;
                        }
                    }
                    if (mp_count > 0u) {
                        effective_mutation_rate *= pow(0.7, f32(mp_count));
                    }
                }
                effective_mutation_rate = clamp(sanitize_f32(effective_mutation_rate), 0.0, 1.0);

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
                    if (can_insert && insert_roll < (effective_mutation_rate * 4.0)) {
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
                            // Shift right by k starting from the end to avoid overwrite.
                            // Forward shifting would smear values and create artificial repeats.
                            var j: u32 = L;
                            loop {
                                if (j == pos) { break; }
                                j -= 1u;
                                seq[j + k] = seq[j];
                            }
                            for (var t = 0u; t < k; t++) {
                                let nb = get_random_rna_base(insert_seed ^ (t * 1664525u + 1013904223u));
                                seq[pos + t] = nb;
                            }
                            L += k;
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
                    if (has_active && delete_roll < (effective_mutation_rate * 4.0)) {
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
                                let mode = hash(delete_seed ^ 0x1B56C4E9u) % 3u;
                                var pos: u32 = 0u;
                                if (mode == 0u) { pos = 0u; }
                                else if (mode == 1u) { pos = L - k; }
                                else { pos = hash(delete_seed ^ 0x2C9F85A1u) % (L - k + 1u); }
                                var j = pos;
                                loop {
                                    if (j + k >= L) { break; }
                                    seq[j] = seq[j + k];
                                    j += 1u;
                                }
                                L -= k;
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

                // Optional X-mutation: mutate padding 'X's and incorporate if valid + adjacent
                {
                    let x_mut_rate = effective_mutation_rate * 0.5;
                    var new_active_start = active_start;
                    var new_active_end = active_end;

                    var left_pos = active_start;
                    while (left_pos > 0u) {
                        left_pos -= 1u;
                        let word = left_pos / 4u;
                        let byte_offset = left_pos % 4u;
                        let b = (offspring_ascii[word] >> (byte_offset * 8u)) & 0xFFu;
                        if (b != 88u) { break; }

                        let mut_seed = offspring_hash ^ (left_pos * 0x12345678u);
                        let mut_chance = hash_f32(mut_seed);
                        if (mut_chance < x_mut_rate) {
                            let new_base = get_random_rna_base(mut_seed * 1664525u);
                            let mask = ~(0xFFu << (byte_offset * 8u));
                            let current_word = offspring_ascii[word];
                            let updated_word = (current_word & mask) | (new_base << (byte_offset * 8u));
                            offspring_ascii[word] = updated_word;
                            mutated_count += 1u;
                            new_active_start = left_pos;
                        } else {
                            break;
                        }
                    }

                    var right_pos = active_end;
                    while (right_pos + 1u < GENOME_LENGTH) {
                        right_pos += 1u;
                        let word = right_pos / 4u;
                        let byte_offset = right_pos % 4u;
                        let b = (offspring_ascii[word] >> (byte_offset * 8u)) & 0xFFu;
                        if (b != 88u) { break; }

                        let mut_seed = offspring_hash ^ (right_pos * 0x87654321u);
                        let mut_chance = hash_f32(mut_seed);
                        if (mut_chance < x_mut_rate) {
                            let new_base = get_random_rna_base(mut_seed * 1664525u);
                            let mask = ~(0xFFu << (byte_offset * 8u));
                            let current_word = offspring_ascii[word];
                            let updated_word = (current_word & mask) | (new_base << (byte_offset * 8u));
                            offspring_ascii[word] = updated_word;
                            mutated_count += 1u;
                            new_active_end = right_pos;
                        } else {
                            break;
                        }
                    }

                    if (new_active_start != active_start || new_active_end != active_end) {
                        active_start = new_active_start;
                        active_end = new_active_end;
                    }
                }

                // Optional active-to-X mutation: low-prob point deletions, with split handling
                {
                    let x_delete_rate = effective_mutation_rate * 0.2;

                    if (active_end != 0xFFFFFFFFu) {
                        for (var bi = active_start; bi <= active_end; bi++) {
                            let delete_seed = offspring_hash ^ (bi * 0xFEDCBA98u);
                            let delete_chance = hash_f32(delete_seed);
                            if (delete_chance < x_delete_rate) {
                                let word = bi / 4u;
                                let byte_offset = bi % 4u;
                                let mask = ~(0xFFu << (byte_offset * 8u));
                                let current_word = offspring_ascii[word];
                                let updated_word = (current_word & mask) | (88u << (byte_offset * 8u));
                                offspring_ascii[word] = updated_word;
                                mutated_count += 1u;
                            }
                        }

                        var pieces: array<u32, 8>;
                        var piece_count = 0u;
                        var in_active = false;
                        var curr_start = 0u;

                        for (var bi = 0u; bi < GENOME_LENGTH; bi++) {
                            let word = bi / 4u;
                            let byte_offset = bi % 4u;
                            let b = (offspring_ascii[word] >> (byte_offset * 8u)) & 0xFFu;

                            if (b != 88u) {
                                if (!in_active) {
                                    in_active = true;
                                    curr_start = bi;
                                }
                            } else {
                                if (in_active) {
                                    in_active = false;
                                    let piece_len = bi - curr_start;
                                    if (piece_len >= MIN_GENE_LENGTH) {
                                        if (piece_count < 8u) {
                                            pieces[piece_count] = curr_start;
                                            pieces[piece_count + 1u] = bi - 1u;
                                            piece_count += 2u;
                                        }
                                    }
                                }
                            }
                        }
                        if (in_active) {
                            let piece_len = GENOME_LENGTH - curr_start;
                            if (piece_len >= MIN_GENE_LENGTH) {
                                if (piece_count < 8u) {
                                    pieces[piece_count] = curr_start;
                                    pieces[piece_count + 1u] = GENOME_LENGTH - 1u;
                                    piece_count += 2u;
                                }
                            }
                        }

                        if (piece_count > 0u) {
                            var max_len = 0u;
                            var best_piece_idx = 0u;
                            var best_has_start = false;

                            for (var p = 0u; p < piece_count; p += 2u) {
                                let p_start = pieces[p];
                                let p_end = pieces[p + 1u];
                                let p_len = p_end - p_start + 1u;

                                var has_start = false;
                                var scan_pos = p_start;
                                loop {
                                    if (scan_pos + 2u > p_end) { break; }
                                    let b0 = (offspring_ascii[scan_pos / 4u] >> ((scan_pos % 4u) * 8u)) & 0xFFu;
                                    let b1 = (offspring_ascii[(scan_pos + 1u) / 4u] >> (((scan_pos + 1u) % 4u) * 8u)) & 0xFFu;
                                    let b2 = (offspring_ascii[(scan_pos + 2u) / 4u] >> (((scan_pos + 2u) % 4u) * 8u)) & 0xFFu;
                                    if (b0 == 65u && b1 == 85u && b2 == 71u) {
                                        has_start = true;
                                        break;
                                    }
                                    scan_pos += 3u;
                                }

                                if (p_len > max_len || (p_len == max_len && has_start && !best_has_start)) {
                                    max_len = p_len;
                                    best_piece_idx = p;
                                    best_has_start = has_start;
                                }
                            }

                            active_start = pieces[best_piece_idx];
                            active_end = pieces[best_piece_idx + 1u];

                            for (var p = 0u; p < piece_count; p += 2u) {
                                if (p != best_piece_idx) {
                                    let d_start = pieces[p];
                                    let d_end = pieces[p + 1u];
                                    for (var bi = d_start; bi <= d_end; bi++) {
                                        let word = bi / 4u;
                                        let byte_offset = bi % 4u;
                                        let mask = ~(0xFFu << (byte_offset * 8u));
                                        let current_word = offspring_ascii[word];
                                        let updated_word = (current_word & mask) | (88u << (byte_offset * 8u));
                                        offspring_ascii[word] = updated_word;
                                    }
                                }
                            }
                        } else {
                            active_start = 0u;
                            active_end = 0xFFFFFFFFu;
                        }
                    }
                }

                if (active_end != 0xFFFFFFFFu) {
                    offspring.gene_length = active_end - active_start + 1u;
                    offspring.genome_offset = active_start;
                } else {
                    offspring.gene_length = 0u;
                    offspring.genome_offset = 0u;
                }

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

                let energy_per_offspring = inherited_energy / f32(num_offspring);
                offspring.energy = energy_per_offspring;

                for (var bi = 0u; bi < MAX_BODY_PARTS; bi++) {
                    offspring.body[bi].pos = vec2<f32>(0.0);
                    offspring.body[bi].data = 0.0;
                    offspring.body[bi].part_type = 0u;
                    offspring.body[bi].alpha_signal = 0.0;
                    offspring.body[bi].beta_signal = 0.0;
                    offspring.body[bi]._pad.x = bitcast<f32>(0u);
                    offspring.body[bi]._pad = vec2<f32>(0.0);
                }

                new_agents[current_spawn_index] = offspring;
            }

            agent_energy_cur -= inherited_energy;
        }
    }
}
