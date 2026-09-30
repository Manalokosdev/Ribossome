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

var<workgroup> compact_scan: array<u32, 256>;

// STABLE compaction: dispatched as EXACTLY ONE workgroup. The old version
// used a global atomic cursor, so the order in which alive agents landed
// in the output was nondeterministic — the same agent could sit at a
// different index every frame, which broke everything that tracks agents
// by index (selection crosshair jumping between two agents, inspector
// following the wrong agent, species thumbnails sampling churn).
// This version walks the buffer in 256-wide chunks and computes each
// alive agent's destination with a workgroup prefix scan, preserving
// buffer order exactly: index i always holds the same agent as long as
// nothing before it died.
@compute @workgroup_size(256)
fn compact_agents(@builtin(local_invocation_id) lid_v: vec3<u32>) {
    let lid = lid_v.x;
    // Scan slightly beyond agent_count to avoid dropping newborns when CPU readback lags.
    let scan_limit = min(params.agent_count + 2000u, params.max_agents);
    let chunks = (scan_limit + 255u) / 256u;
    var base = 0u;
    for (var c = 0u; c < chunks; c++) {
        let i = c * 256u + lid;
        var flag = 0u;
        if (i < scan_limit) {
            if (agents_in[i].alive != 0u) {
                flag = 1u;
            }
        }
        compact_scan[lid] = flag;
        workgroupBarrier();
        // Hillis-Steele inclusive scan over the 256 flags.
        var offset = 1u;
        while (offset < 256u) {
            var v = compact_scan[lid];
            if (lid >= offset) {
                v += compact_scan[lid - offset];
            }
            workgroupBarrier();
            compact_scan[lid] = v;
            workgroupBarrier();
            offset = offset << 1u;
        }
        let excl = compact_scan[lid] - flag;
        let chunk_total = compact_scan[255];
        if (flag == 1u) {
            let dst = base + excl;
            if (dst < params.max_agents) {
                agents_out[dst] = agents_in[i];
            }
        }
        base += chunk_total;
        workgroupBarrier();
    }
    if (lid == 0u) {
        atomicStore(&spawn_debug_counters[2], base);
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
