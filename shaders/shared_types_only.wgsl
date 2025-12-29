// Types-only WGSL module for compute-only pipelines (e.g. compact/merge).
//
// This intentionally avoids pulling in the full shared.wgsl, which contains
// inspector/render helpers that depend on additional draw_* functions.

// ============================================================================
// CONSTANTS (must match shared.wgsl)
// ============================================================================

const SIM_SIZE: u32 = 61440u;
const MAX_BODY_PARTS: u32 = 64u;
const GENOME_BYTES: u32 = 256u;
const GENOME_LENGTH: u32 = GENOME_BYTES;
const GENOME_ASCII_WORDS: u32 = GENOME_BYTES / 4u;
const GENOME_PACKED_WORDS: u32 = GENOME_BYTES / 16u;
const GENOME_BASES_PER_PACKED_WORD: u32 = 16u;
const MIN_GENE_LENGTH: u32 = 6u;

// ============================================================================
// STRUCTURES (must match shared.wgsl layout)
// ============================================================================

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

struct SpawnRequest {
    seed: u32,
    genome_seed: u32,
    flags: u32,
    _pad_seed: u32,
    position: vec2<f32>,
    energy: f32,
    rotation: f32,
    genome_override_len: u32,
    genome_override_offset: u32,
    genome_override_packed: array<u32, GENOME_PACKED_WORDS>,
    _pad_genome: array<u32, 2>,
}

// Keep the full prefix up through agent_count so field offsets match the main SimParams.
struct SimParams {
    dt: f32,
    frame_dt: f32,
    drag: f32,
    energy_cost: f32,
    amino_maintenance_cost: f32,
    spawn_probability: f32,
    death_probability: f32,
    grid_size: f32,
    camera_zoom: f32,
    camera_pan_x: f32,
    camera_pan_y: f32,
    prev_camera_pan_x: f32,
    prev_camera_pan_y: f32,
    follow_mode: u32,
    window_width: f32,
    window_height: f32,
    alpha_blur: f32,
    beta_blur: f32,
    gamma_diffuse: f32,
    gamma_blur: f32,
    gamma_shift: f32,
    alpha_slope_bias: f32,
    beta_slope_bias: f32,
    alpha_multiplier: f32,
    beta_multiplier: f32,
    dye_precipitation: f32,
    chemical_slope_scale_alpha: f32,
    chemical_slope_scale_beta: f32,
    mutation_rate: f32,
    food_power: f32,
    poison_power: f32,
    pairing_cost: f32,
    max_agents: u32,
    cpu_spawn_count: u32,
    agent_count: u32,
}
