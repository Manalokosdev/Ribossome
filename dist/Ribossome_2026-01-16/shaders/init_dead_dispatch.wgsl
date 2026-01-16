// Minimal shader to write indirect dispatch args for init-dead pass.
// Kept separate from the main simulation shader to avoid hitting wgpu's
// max_storage_buffers_per_shader_stage limit.

@group(0) @binding(0)
var<storage, read> spawn_debug_counters: array<atomic<u32>, 3>;

// Indirect dispatch args buffer: [x, y, z] = workgroup counts
@group(0) @binding(1)
var<storage, read_write> init_dead_dispatch_args: array<u32, 3>;

// 16-byte uniform (flat u32s) to avoid padding surprises.
struct InitDeadParams {
    max_agents: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(2)
var<uniform> init_dead_params: InitDeadParams;

@compute @workgroup_size(1)
fn write_init_dead_dispatch_args() {
    let alive_total: u32 = atomicLoad(&spawn_debug_counters[2]);
    let max_agents: u32 = init_dead_params.max_agents;

    // Saturating dead_count = max(0, max_agents - alive_total)
    let dead_count: u32 = select(max_agents - alive_total, 0u, alive_total > max_agents);

    let groups: u32 = (dead_count + 255u) / 256u;

    init_dead_dispatch_args[0] = groups;
    init_dead_dispatch_args[1] = 1u;
    init_dead_dispatch_args[2] = 1u;
}
