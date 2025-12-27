// Dedicated shader for compacting dead agents and merging new spawns
// This runs AFTER reproduction and process_agents have completed

// CRITICAL INSIGHT:
// The old compact/merge had a race condition - compact counted alive agents,
// then merge used atomicAdd to append spawns. If merge added at the same index,
// agents could overwrite each other.
//
// NEW STRATEGY:
// 1. Compact: Copy all alive agents to output buffer sequentially, track count
// 2. Merge: Append spawns directly after the compacted agents using atomic reservation
// This ensures no overlap between existing agents and spawns.

@compute @workgroup_size(64)
fn compact_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent_id = gid.x;

    // Scan a bit beyond agent_count to catch newborns from previous frame
    // but don't scan full max_agents to avoid picking up stale tail slots
    let scan_limit = min(params.agent_count + 2000u, params.max_agents);
    if (agent_id >= scan_limit) {
        return;
    }

    // Read from agents_in (the result of process_agents)
    let agent = agents_in[agent_id];

    // If agent is alive, claim a slot in the compacted output array
    if (agent.alive != 0u) {
        let idx = atomicAdd(&spawn_debug_counters[2], 1u);
        if (idx < params.max_agents) {
            agents_out[idx] = agent;
        }
    }
}

@compute @workgroup_size(64)
fn merge_agents_cooperative(@builtin(global_invocation_id) gid: vec3<u32>) {
    let spawn_id = gid.x;

    // Clamp spawn count to buffer size (2000) to avoid OOB reads
    let spawn_count = min(atomicLoad(&spawn_debug_counters[0]), 2000u);

    if (spawn_id >= spawn_count) {
        return;
    }

    // Read the new agent that was created by reproduction
    let new_agent = new_agents[spawn_id];

    // Append to end of compacted alive array using alive counter as running size
    // This is the CRITICAL FIX: we atomically reserve the next slot AFTER compaction
    let target_index = atomicAdd(&spawn_debug_counters[2], 1u);

    // Write the spawn to the reserved slot
    if (target_index < params.max_agents) {
        agents_out[target_index] = new_agent;
    }
}
