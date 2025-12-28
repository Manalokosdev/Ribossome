// Compact/Merge entry points used by Rust as a separate shader module.
//
// IMPORTANT:
// - Types (Agent, SimParams, etc) come from shared_types_only.wgsl.
// - Rust concatenates shared_types_only.wgsl before this file when building the module.

@group(0) @binding(0) var<storage, read> agents_in: array<Agent>;
@group(0) @binding(1) var<storage, read_write> agents_out: array<Agent>;
@group(0) @binding(6) var<uniform> params: SimParams;
@group(0) @binding(9) var<storage, read_write> new_agents: array<Agent>;
@group(0) @binding(10) var<storage, read_write> spawn_debug_counters: array<atomic<u32>, 3>;

@compute @workgroup_size(64)
fn compact_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent_id = gid.x;

    // Scan slightly beyond agent_count to avoid dropping newborns when CPU readback lags.
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

@compute @workgroup_size(64)
fn merge_agents_cooperative(@builtin(global_invocation_id) gid: vec3<u32>) {
    let spawn_id = gid.x;
    let spawn_count = min(atomicLoad(&spawn_debug_counters[0]), 2000u);
    if (spawn_id >= spawn_count) {
        return;
    }

    let target_index = atomicAdd(&spawn_debug_counters[2], 1u);
    if (target_index < params.max_agents) {
        agents_out[target_index] = new_agents[spawn_id];
    }
}
