// Minimal bindings for compact_merge shader
// Only includes what's needed for compact and merge operations

const MAX_BODY_PARTS: u32 = 64u;
const GENOME_BYTES: u32 = 256u;
const GENOME_LENGTH: u32 = GENOME_BYTES;
const GENOME_ASCII_WORDS: u32 = GENOME_BYTES / 4u;
const GENOME_PACKED_WORDS: u32 = GENOME_BYTES / 16u;
const GENOME_BASES_PER_PACKED_WORD: u32 = 16u;

struct BodyPart {
    pos: vec2<f32>,
    data: f32,
    part_type: u32,
    alpha_signal: f32,
    beta_signal: f32,
    _pad: vec2<f32>,
}

struct Agent {
    position: vec2<f32>,
    velocity: vec2<f32>,
    rotation: f32,
    energy: f32,
    energy_capacity: f32,
    torque_debug: f32,
    morphology_origin: vec2<f32>,
    alive: u32,
    body_count: u32,
    pairing_counter: u32,
    is_selected: u32,
    generation: u32,
    age: u32,
    total_mass: f32,
    poison_resistant_count: u32,
    gene_length: u32,
    genome_offset: u32,
    genome_packed: array<u32, GENOME_PACKED_WORDS>,
    body: array<BodyPart, MAX_BODY_PARTS>,
}

struct SimParams {
    agent_count: u32,
    max_agents: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> agents_in: array<Agent>;
@group(0) @binding(1) var<storage, read_write> agents_out: array<Agent>;
@group(0) @binding(6) var<uniform> params: SimParams;
@group(0) @binding(9) var<storage, read_write> new_agents: array<Agent>;
@group(0) @binding(10) var<storage, read_write> spawn_debug_counters: array<atomic<u32>>;

// ============================================================================
// COMPACT AND MERGE LOGIC
// ============================================================================

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
