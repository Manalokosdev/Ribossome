// Ribossome - GPU-Accelerated Artificial Life Simulator
// Copyright (c) 2025 Filipe da Veiga Ventura Alves
// Licensed under MIT License

// Shared WGSL module: constants, structs, bindings, amino/organ tables, utilities, genome helpers, translation logic

// ============================================================================
// CONSTANTS
// ============================================================================

const ENV_GRID_SIZE: u32 = 1024u;      // Environment grid resolution (alpha/beta/gamma)
const GRID_SIZE: u32 = ENV_GRID_SIZE;  // Alias for backward compatibility
const SPATIAL_GRID_SIZE: u32 = 512u;   // Spatial hash grid for agent collision detection
const SIM_SIZE: u32 = 15360u;          // Simulation world size (reduced to half)
const MAX_BODY_PARTS: u32 = 64u;
const GENOME_BYTES: u32 = 256u;
const GENOME_LENGTH: u32 = GENOME_BYTES; // Legacy alias used throughout shader
const GENOME_WORDS: u32 = GENOME_BYTES / 4u;
const PACKED_GENOME_WORDS: u32 = GENOME_BYTES / 16u;
const PACKED_BASES_PER_WORD: u32 = 16u;
const MIN_GENE_LENGTH: u32 = 6u;
const PROPELLERS_ENABLED: bool = true;
// Experimental: disable global agent orientation; body geometry defines facing.
const DISABLE_GLOBAL_ROTATION: bool = false;
// Smoothing factor for per-part signal-induced angle (0..1). Higher = faster response.
const ANGLE_SMOOTH_FACTOR: f32 = 0.2;
// Physics stabilization (no delta time): blend factors and clamps
const VELOCITY_BLEND: f32 = 0.6;      // 0..1, higher = quicker velocity changes
const ANGULAR_BLEND: f32 = 0.6;       // 0..1, higher = quicker rotation changes
const VEL_MAX: f32 = 24.0;             // Max linear speed per frame
const ANGVEL_MAX: f32 = 1.5;         // Max angular change (radians) per frame
// Signal-to-angle shaping (no dt): cap amplitude and per-frame change
const SIGNAL_GAIN: f32 =20;        // global scale for signal-driven angle (was 20.0)
// Separate gains for alpha vs beta to restore original triple-contribution tunability
const ANGLE_GAIN_ALPHA: f32 = 1.0;  // relative weighting for alpha term
const ANGLE_GAIN_BETA: f32 = 1.0;   // relative weighting for beta term
const MAX_SIGNAL_ANGLE: f32 = 2.4;    // hard cap on signal-induced angle (radians)
const MAX_SIGNAL_STEP: f32 = 0.8;    // max per-frame change due to signals (radians)
const PROP_TORQUE_COUPLING: f32 = 1; // 0=no spin from props, 1=full lever-arm torque
const INSPECTOR_WIDTH: u32 = 300u;    // Width of inspector panel on right side

// ============================================================================
// STRUCTURES (std430 aligned)
// ============================================================================

struct BodyPart {
    pos: vec2<f32>,           // relative position from agent center (8 bytes)
    size: f32,                // radius (4 bytes)
    part_type: u32,           // bits 0-7 = base type (amino acid or organ), bits 8-15 = organ parameter
    alpha_signal: f32,        // alpha signal propagating through body (4 bytes)
    beta_signal: f32,         // beta signal propagating through body (4 bytes)
    _pad: vec2<f32>,          // padding to 32 bytes (8 bytes)
                              // _pad.x = smoothed signal angle OR condenser charge OR clock signal OR vampire cooldown
                              // _pad.y = packed u16 prev_world_pos OR last drain amount (vampire mouths)
}

// ============================================================================
// BODY PART ENCODING HELPERS
// ============================================================================

fn pack_prev_pos(prev_pos: vec2<f32>) -> u32 {
    let scale = 65535.0 / f32(SIM_SIZE);
    let x16 = u32(clamp(prev_pos.x * scale, 0.0, 65535.0));
    let y16 = u32(clamp(prev_pos.y * scale, 0.0, 65535.0));
    return (x16 & 0xFFFFu) | ((y16 & 0xFFFFu) << 16u);
}

fn unpack_prev_pos(packed: u32) -> vec2<f32> {
    let scale = f32(SIM_SIZE) / 65535.0;
    let x16 = f32(packed & 0xFFFFu);
    let y16 = f32((packed >> 16u) & 0xFFFFu);
    return vec2<f32>(x16 * scale, y16 * scale);
}

fn get_base_part_type(part_type: u32) -> u32 {
    return part_type & 0xFFu;
}

fn get_organ_param(part_type: u32) -> u32 {
    return (part_type >> 8u) & 0xFFu;
}

fn encode_part_type(base_type: u32, organ_param: u32) -> u32 {
    return (base_type & 0xFFu) | ((organ_param & 0xFFu) << 8u);
}

fn organ_param_to_strength(param: u32) -> f32 {
    return f32(param) / 127.5;  // 0 -> 0.0, 127 -> ~1.0, 255 -> 2.0
}

struct Agent {
    position: vec2<f32>,      // world position
    velocity: vec2<f32>,      // current velocity
    rotation: f32,            // current rotation angle
    energy: f32,              // energy level
    energy_capacity: f32,     // maximum energy storage (sum of all mouths * 10)
    torque_debug: f32,        // accumulated torque this frame (for inspector)
    morphology_origin: vec2<f32>, // Where the chain origin is in local space after CoM centering
    alive: u32,               // 1 = alive, 0 = dead
    body_count: u32,          // number of body parts
    pairing_counter: u32,     // number of bases successfully paired (0..gene_length)
    is_selected: u32,         // 1 = selected for debug view, 0 = not selected
    generation: u32,          // lineage generation (0 = initial spawn)
    age: u32,                 // age in frames since spawn
    total_mass: f32,          // total mass computed after morphology
    poison_resistant_count: u32, // number of poison-resistant organs (type 29)
    gene_length: u32,        // number of non-X bases in genome (valid gene length)
    genome: array<u32, GENOME_WORDS>,   // GENOME_BYTES bytes genome (ASCII RNA bases)
    _pad_genome_align: array<u32, 5>, // padding to align body array to 16-byte boundary
    body: array<BodyPart, MAX_BODY_PARTS>, // body parts array
}

struct SpawnRequest {
    seed: u32,
    genome_seed: u32,
    flags: u32,
    _pad_seed: u32,
    position: vec2<f32>,
    energy: f32,
    rotation: f32,
    genome_override: array<u32, GENOME_WORDS>,
}

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
    gamma_blur: f32,
    alpha_slope_bias: f32,
    beta_slope_bias: f32,
    alpha_multiplier: f32,
    beta_multiplier: f32,
    chemical_slope_scale_alpha: f32,
    chemical_slope_scale_beta: f32,
    mutation_rate: f32,
    food_power: f32,
    poison_power: f32,
    pairing_cost: f32,
    max_agents: u32,
    cpu_spawn_count: u32,
    agent_count: u32,
    random_seed: u32,
    debug_mode: u32,
    visual_stride: u32,
    selected_agent_index: u32,  // Index of selected agent for debug visualization (u32::MAX if none)
    repulsion_strength: f32,
    agent_repulsion_strength: f32,
    gamma_strength: f32,
    prop_wash_strength: f32,
    gamma_vis_min: f32,
    gamma_vis_max: f32,
    draw_enabled: u32,
    gamma_debug: u32,
    gamma_hidden: u32,
    slope_debug: u32,
    alpha_show: u32,
    beta_show: u32,
    gamma_show: u32,
    slope_lighting: u32,
    slope_lighting_strength: f32,
    trail_diffusion: f32,
    trail_decay: f32,
    trail_opacity: f32,
    trail_show: u32,
    interior_isotropic: u32,   // When 1, override per-amino left/right multipliers with isotropic interior diffusion
    ignore_stop_codons: u32,   // When 1, translate entire genome to max body parts
    require_start_codon: u32,  // When 1, require AUG start codon before translation begins
    asexual_reproduction: u32, // When 1, offspring are direct mutated copies (asexual); when 0, reverse-complemented (sexual)
    // Visualization parameters
    background_color_r: f32,
    background_color_g: f32,
    background_color_b: f32,
    alpha_blend_mode: u32,
    beta_blend_mode: u32,
    gamma_blend_mode: u32,
    slope_blend_mode: u32,  // 0=none, 1=hard light, 2=soft light
    alpha_color_r: f32,
    alpha_color_g: f32,
    alpha_color_b: f32,
    beta_color_r: f32,
    beta_color_g: f32,
    beta_color_b: f32,
    gamma_color_r: f32,
    gamma_color_g: f32,
    gamma_color_b: f32,
    grid_interpolation: u32,  // 0=nearest, 1=bilinear, 2=bicubic
    alpha_gamma_adjust: f32,  // Gamma correction for alpha channel
    beta_gamma_adjust: f32,   // Gamma correction for beta channel
    gamma_gamma_adjust: f32,  // Gamma correction for gamma channel
    light_dir_x: f32,         // Light direction for slope-based lighting
    light_dir_y: f32,
    light_dir_z: f32,
    light_power: f32,         // Light intensity multiplier for 3D shading
    agent_blend_mode: u32,    // Agent visualization: 0=comp, 1=add, 2=subtract, 3=multiply
    agent_color_r: f32,
    agent_color_g: f32,
    agent_color_b: f32,
    agent_color_blend: f32,   // Blend factor: 0.0=amino color only, 1.0=agent color only
    epoch: u32,               // Current simulation epoch for time-based effects
    vector_force_power: f32,  // Global force multiplier (0.0 = off)
    vector_force_x: f32,      // Force direction X (-1.0 to 1.0)
    vector_force_y: f32,      // Force direction Y (-1.0 to 1.0)
    inspector_zoom: f32,      // Inspector preview zoom level (1.0 = default)
    agent_trail_decay: f32,   // Agent trail decay rate (0.0 = persistent, 1.0 = instant clear)
    _padding1: f32,
        fluid_wind_push_strength: f32,
}

struct EnvironmentInitParams {
    grid_resolution: u32,
    seed: u32,
    noise_octaves: u32,
    slope_octaves: u32,
    noise_scale: f32,
    noise_contrast: f32,
    slope_scale: f32,
    slope_contrast: f32,
    alpha_range: vec2<f32>,
    beta_range: vec2<f32>,
    gamma_height_range: vec2<f32>,
    trail_values: vec4<f32>,
    slope_pair: vec2<f32>,
    gen_params: vec4<u32>, // x=mode(0=all,1=a,2=b,3=g), y=type(0=flat,1=noise), z=value_bits, w=seed
    alpha_noise_scale: f32,
    beta_noise_scale: f32,
    gamma_noise_scale: f32,
    noise_power: f32,
}

// ============================================================================
// BINDINGS
// ============================================================================

@group(0) @binding(0)
var<storage, read_write> agents_in: array<Agent>;

@group(0) @binding(1)
var<storage, read_write> agents_out: array<Agent>;

@group(0) @binding(2)
var<storage, read_write> alpha_grid: array<f32>;  // 512x512 environment grid

@group(0) @binding(3)
var<storage, read_write> beta_grid: array<f32>;   // 512x512 environment grid

@group(0) @binding(4)
var<storage, read_write> visual_grid: array<vec4<f32>>; // RGBA render target

@group(0) @binding(5)
var<storage, read_write> agent_grid: array<vec4<f32>>; // Separate agent render buffer

@group(0) @binding(6)
var<uniform> params: SimParams;

@group(0) @binding(7)
var<storage, read_write> alive_counter: atomic<u32>;

@group(0) @binding(8)
var<storage, read_write> debug_counter: atomic<u32>;

@group(0) @binding(9)
var<storage, read_write> new_agents: array<Agent>;  // Buffer for spawned agents

@group(0) @binding(10)
var<storage, read_write> spawn_counter: atomic<u32>;  // Count of spawned agents this frame

@group(0) @binding(11)
var<storage, read> spawn_requests: array<SpawnRequest>;

@group(0) @binding(12)
var<storage, read_write> selected_agent_buffer: array<Agent>;  // Buffer to hold the selected agent for CPU readback

@group(0) @binding(13)
var<storage, read_write> gamma_grid: array<f32>; // Terrain height field + slope components

@group(0) @binding(14)
// NOTE: Use vec4 for std430-friendly 16-byte stride to match host buffer layout
var<storage, read_write> trail_grid: array<vec4<f32>>; // Agent color trail RGB + energy trail A (unclamped)

@group(0) @binding(15)
var<uniform> environment_init: EnvironmentInitParams;

@group(0) @binding(16)
var<storage, read> rain_map: array<vec2<f32>>; // x=alpha, y=beta

@group(0) @binding(17)
var<storage, read_write> agent_spatial_grid: array<atomic<u32>>; // Agent index per grid cell (atomic for vampire victim claiming)

// Spatial grid special markers
const SPATIAL_GRID_EMPTY: u32 = 0xFFFFFFFFu;     // No agent in this cell
const SPATIAL_GRID_CLAIMED: u32 = 0xFFFFFFFEu;   // Cell claimed by vampire (victim being drained)
const VAMPIRE_MOUTH_COOLDOWN: f32 = 60.0;         // Frames between drains (1 second at 60fps)

// ============================================================================
// AMINO ACID PROPERTIES
// ============================================================================

struct AminoAcidProperties {
    segment_length: f32,
    thickness: f32,
    base_angle: f32,
    mass: f32,
    alpha_sensitivity: f32,
    beta_sensitivity: f32,
    is_propeller: bool,
    thrust_force: f32,
    color: vec3<f32>,
    is_mouth: bool,
    energy_absorption_rate: f32,
    beta_absorption_rate: f32,
    beta_damage: f32,
    energy_storage: f32,
    energy_consumption: f32,
    is_alpha_sensor: bool,
    is_beta_sensor: bool,
    is_energy_sensor: bool,
    is_agent_alpha_sensor: bool,
    is_agent_beta_sensor: bool,
    is_trail_energy_sensor: bool,
    signal_decay: f32,
    alpha_left_mult: f32,
    alpha_right_mult: f32,
    beta_left_mult: f32,
    beta_right_mult: f32,
    is_displacer: bool,
    is_inhibitor: bool,
    is_condenser: bool,
    is_clock: bool,
    parameter1: f32,
}

// ============================================================================
// AMINO ACID & ORGAN PROPERTY LOOKUP TABLE (0â€“19 amino, 20â€“41 organs)
// ============================================================================

const AMINO_COUNT: u32 = 42u;

var<private> AMINO_DATA: array<array<vec4<f32>, 6>, AMINO_COUNT> = array<array<vec4<f32>, 6>, AMINO_COUNT>(
    // 0  A - Alanine
    array<vec4<f32>,6>( vec4<f32>(8.5, 2.5, 0.10, 0.015), vec4<f32>(-0.2, 0.2, 0.0, 0.001), vec4<f32>(0.3, 0.3, 0.3, 0.0), vec4<f32>(0.0, 0.3, -0.73, -0.23), vec4<f32>(0.2, 0.8, 0.2, 0.7), vec4<f32>(0.3, 0.0, 0.0, 0.0) ),
    // 1  C - Cysteine
    array<vec4<f32>,6>( vec4<f32>(10.0, 2.5, 0.13, 0.02), vec4<f32>(0.0, 0.349066, 0.0, 0.001), vec4<f32>(0.1, 0.1, 0.1, 0.0), vec4<f32>(0.0, 0.3, 0.42, 0.67), vec4<f32>(0.2, 0.5, 0.5, 0.5), vec4<f32>(0.5, 0.0, 0.0, 0.0) ),
    // 2  D - Aspartic acid
    array<vec4<f32>,6>( vec4<f32>(13.0, 3.0, 0.05, 0.018), vec4<f32>(-0.2, 0.3, 0.0, 0.001), vec4<f32>(0.1, 0.1, 0.1, 0.0), vec4<f32>(0.0, 0.3, -0.91, -0.91), vec4<f32>(0.2, -0.2, 1.2, -0.3), vec4<f32>(1.3, 0.0, 0.0, 0.0) ),
    // 3  E - Glutamic acid (Poison Resistance)
    array<vec4<f32>,6>( vec4<f32>(30.0, 10.0, -0.12, 10.0), vec4<f32>(0.1, 0.12, 0.0, 0.003), vec4<f32>(0.3, 0.4, 0.1, 0.0), vec4<f32>(0.0, 0.3, -0.58, -0.058), vec4<f32>(0.2, 0.6, 0.4, 0.55), vec4<f32>(0.45, 0.0, 0.0, 0.0) ),
    // 4  F - Phenylalanine
    array<vec4<f32>,6>( vec4<f32>(13.0, 4.5, 0.03, 0.01), vec4<f32>(0.2, -0.33, 0.0, 0.001), vec4<f32>(0.6, 0.3, 0.0, 0.0), vec4<f32>(0.0, 0.3, 0.17, 0.17), vec4<f32>(0.2, 1.4, -0.4, 1.3), vec4<f32>(-0.3, 0.0, 0.0, 0.0) ),
    // 5  G - Glycine
    array<vec4<f32>,6>( vec4<f32>(4.0, 3.0, -0.06, 0.02), vec4<f32>(0.7, 0.1, 0.0, 0.001), vec4<f32>(0.9, 0.9, 0.9, 0.0), vec4<f32>(0.0, 0.0, 0.88, 0.88), vec4<f32>(0.3, 0.5, 0.5, 0.5), vec4<f32>(0.5, 0.0, 0.0, 0.0) ),
    // 6  H - Histidine
    array<vec4<f32>,6>( vec4<f32>(9.0, 4.0, -0.07, 0.02), vec4<f32>(0.2, -0.61, 0.0, 0.001), vec4<f32>(0.3, 0.5, 0.8, 0.0), vec4<f32>(0.0, 0.3, -0.35, -0.35), vec4<f32>(0.2, 1.2, -0.2, -0.3), vec4<f32>(1.3, 0.0, 0.0, 0.0) ),
    // 7  I - Isoleucine
    array<vec4<f32>,6>( vec4<f32>(19.0, 5.5, 0.09, 0.02), vec4<f32>(-0.3, 0.69, 0.0, 0.001), vec4<f32>(0.38, 0.38, 0.38, 0.0), vec4<f32>(0.0, 0.3, 0.61, 0.61), vec4<f32>(0.2, 0.65, 0.35, 0.7), vec4<f32>(0.3, 0.0, 0.0, 0.0) ),
    // 8  K - Lysine
    array<vec4<f32>,6>( vec4<f32>(15.0, 3.5, 0.31, 0.03), vec4<f32>(0.6, -0.16, 0.0, 0.001), vec4<f32>(0.1, 0.1, 0.2, 0.0), vec4<f32>(0.0, 0.3, -0.12, -0.12), vec4<f32>(0.2, 0.7, 0.3, 0.65), vec4<f32>(0.35, 0.0, 0.0, 0.0) ),
    // 9  L - Leucine
    array<vec4<f32>,6>( vec4<f32>(13.0, 4.5, -0.24, 0.02), vec4<f32>(-0.3332, 0.1, 0.0, 0.001), vec4<f32>(0.6, 0.5, 0.2, 0.0), vec4<f32>(0.0, 0.3, 0.95, 0.95), vec4<f32>(0.2, 0.35, 0.65, 0.4), vec4<f32>(0.6, 0.0, 0.0, 0.0) ),
    // 10 M - Methionine (START)
    array<vec4<f32>,6>( vec4<f32>(8.5, 4.0, -0.52, 0.02), vec4<f32>(0.14, -0.64, 0.0, 0.001), vec4<f32>(0.3, 0.3, 0.2, 0.0), vec4<f32>(0.0, 0.3, -0.48, -0.48), vec4<f32>(0.2, 0.8, 0.2, -0.1), vec4<f32>(1.1, 0.0, 0.0, 0.0) ),
    // 11 N - Enabler
    array<vec4<f32>,6>( vec4<f32>(16.0, 6.0, 0.21, 0.15), vec4<f32>(0.2, 0.3, 0.0, 0.001), vec4<f32>(0.1, 0.1, 0.2, 0.0), vec4<f32>(0.0, 0.0, 0.24, 0.24), vec4<f32>(0.2, 0.5, 0.5, 0.5), vec4<f32>(0.5, 0.0, 0.0, 0.0) ),
    // 12 P - Proline (Propeller)
    array<vec4<f32>,6>( vec4<f32>(16.0, 8.0, -0.333, 0.05), vec4<f32>(0.5, -0.1, 2.5, 0.01), vec4<f32>(0.1, 0.0, 0.2, 0.0), vec4<f32>(0.0, 0.3, -0.77, -0.77), vec4<f32>(0.2, 1.5, -0.5, 1.4), vec4<f32>(-0.4, 0.0, 0.0, 0.0) ),
    // 13 Q - Glutamine
    array<vec4<f32>,6>( vec4<f32>(8.5, 3.0, -0.221, 0.02), vec4<f32>(0.24, -0.4, 0.0, 0.001), vec4<f32>(0.34, 0.34, 0.34, 0.0), vec4<f32>(0.0, 0.3, 0.53, 0.53), vec4<f32>(0.2, 1.0, 0.0, 0.75), vec4<f32>(0.25, 0.0, 0.0, 0.0) ),
    // 14 R - Arginine
    array<vec4<f32>,6>( vec4<f32>(18.5, 3.5, -0.27, 0.04), vec4<f32>(0.5, -0.15, 0.0, 0.001), vec4<f32>(0.29, 0.29, 0.29, 0.0), vec4<f32>(0.0, 0.3, -0.29, -0.29), vec4<f32>(0.2, -0.4, 1.4, -0.3), vec4<f32>(1.3, 0.0, 0.0, 0.0) ),
    // 15 S - Serine
    array<vec4<f32>,6>( vec4<f32>(10.5, 2.5, -0.349066, 0.02), vec4<f32>(-0.349066, 0.0, 0.0, 0.001), vec4<f32>(0.0, 0.2, 0.1, 0.0), vec4<f32>(0.0, 0.2, 0.71, 0.71), vec4<f32>(0.2, 0.5, 0.5, 0.5), vec4<f32>(0.5, 0.0, 0.0, 0.0) ),
    // 16 T - Threonine
    array<vec4<f32>,6>( vec4<f32>(10.5, 3.5, 0.10112, 0.02), vec4<f32>(0.1, -0.5, 0.0, 0.001), vec4<f32>(0.45, 0.15, 0.45, 0.0), vec4<f32>(0.0, 0.2, -0.66, -0.66), vec4<f32>(0.2, 0.6, 0.4, 0.7), vec4<f32>(0.3, 0.0, 0.0, 0.0) ),
    // 17 V - Valine
    array<vec4<f32>,6>( vec4<f32>(12.0, 8.0, 0.09, 0.04), vec4<f32>(-0.3, 0.73, 0.0, 0.001), vec4<f32>(0.2, 0.3, 0.4, 0.0), vec4<f32>(0.0, 0.3, 0.36, 0.36), vec4<f32>(0.2, 0.35, 0.65, 0.6), vec4<f32>(0.4, 0.0, 0.0, 0.0) ),
    // 18 W - Tryptophan
    array<vec4<f32>,6>( vec4<f32>(16.0, 4.0, 0.349066, 0.01), vec4<f32>(0.31, -0.1, 0.0, 0.001), vec4<f32>(0.3, 0.3, 0.2, 0.0), vec4<f32>(0.0, 0.4, -0.84, -0.84), vec4<f32>(0.2, 0.55, 0.45, 0.6), vec4<f32>(0.4, 0.0, 0.0, 0.0) ),
    // 19 Y - Tyrosine
    array<vec4<f32>,6>( vec4<f32>(11.5, 4.0, -0.523599, 0.03), vec4<f32>(-0.2, 0.52, 0.0, 0.001), vec4<f32>(0.5, 0.6, 0.4, 0.0), vec4<f32>(0.0, 0.3, 0.08, 0.08), vec4<f32>(0.2, 0.25, 0.75, 0.3), vec4<f32>(0.7, 0.0, 0.0, 0.0) ),

    // === ORGANS ===
    // 20 MOUTH
    array<vec4<f32>,6>( vec4<f32>(8.0, 3.5, 0.0872665, 0.05), vec4<f32>(0.6, -0.16, 0.0, 0.0025), vec4<f32>(1.0, 1.0, 0.0, 10.0), vec4<f32>(0.8, 0.8, -0.12, -0.12), vec4<f32>(0.2, 1.4, -0.4, 1.3), vec4<f32>(-0.3, 0.0, 0.0, 0.0) ),
    // 21 PROPELLER
    array<vec4<f32>,6>( vec4<f32>(16.0, 8.0, -0.523599, 0.05), vec4<f32>(0.5, -0.1, 2.5, 0.007), vec4<f32>(0.0, 0.0, 1.1, 0.0), vec4<f32>(0.0, 0.3, -0.77, -0.77), vec4<f32>(0.2, 0.45, 0.55, 0.5), vec4<f32>(0.5, 0.0, 0.0, 0.0) ),
    // 22 ALPHA SENSOR
    array<vec4<f32>,6>( vec4<f32>(10.5, 2.5, -0.2349066, 0.05), vec4<f32>(-0.1349066, -0.05, 0.0, 0.00005), vec4<f32>(0.0, 1.0, 0.0, 0.0), vec4<f32>(0.0, 0.2, 0.71, 0.71), vec4<f32>(0.1, 0.5, 0.5, 0.5), vec4<f32>(0.5, 0.0, 0.0, 0.0) ),
    // 23 BETA SENSOR
    array<vec4<f32>,6>( vec4<f32>(10.0, 2.5, 0.3523599, 0.05), vec4<f32>(0.0, 0.1349066, 0.0, 0.00005), vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.3, 0.42, 0.42), vec4<f32>(0.1, 0.5, 0.5, 0.5), vec4<f32>(0.5, 0.0, 0.0, 0.0) ),
    // 24 ENERGY SENSOR
    array<vec4<f32>,6>( vec4<f32>(10.5, 3.5, 0.4570796, 0.05), vec4<f32>(0.1, -0.15, 0.0, 0.00005), vec4<f32>(0.6, 0.0, 0.8, 0.0), vec4<f32>(0.0, 0.2, -0.66, -0.66), vec4<f32>(0.2, 0.9, 0.1, 0.85), vec4<f32>(0.15, 0.0, 0.0, 0.0) ),
    // 25 DISPLACER
    array<vec4<f32>,6>( vec4<f32>(12.0, 8.0, 0.08151, 0.02), vec4<f32>(-0.13, 0.173, 0.0, 0.007), vec4<f32>(0.0, 1.0, 1.0, 0.0), vec4<f32>(0.0, 0.3, 0.36, 0.36), vec4<f32>(0.2, -0.3, 1.3, 1.2), vec4<f32>(-0.2, 0.0, 0.0, 0.0) ),
    // 26 ENABLER
    array<vec4<f32>,6>( vec4<f32>(6.0, 6.0, 0.6785398, 0.05), vec4<f32>(0.2, 0.3, 0.0, 0.001), vec4<f32>(1.0, 1.0, 1.0, 0.0), vec4<f32>(0.0, 0.0, 0.24, 0.24), vec4<f32>(0.2, 0.5, 0.5, 0.5), vec4<f32>(0.5, 0.0, 0.0, 0.0) ),
    // 27 unused
    array<vec4<f32>,6>( vec4<f32>(8.0, 3.0, 0.0, 0.2), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.5, 0.5, 0.5, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.2, 0.5, 0.5, 0.5), vec4<f32>(0.5, 0.0, 0.0, 0.0) ),
    // 28 STORAGE
    array<vec4<f32>,6>( vec4<f32>(16.0, 22.0, 0.1349066, 1.3), vec4<f32>(0.31, -0.1, 0.0, 0.001), vec4<f32>(1.0, 0.5, 0.0, 100.0), vec4<f32>(0.0, 0.4, -0.84, -0.84), vec4<f32>(0.15, 0.55, 0.45, 0.6), vec4<f32>(0.4, 0.0, 0.0, 0.0) ),
    // 29 POISON RESISTANCE
    array<vec4<f32>,6>( vec4<f32>(16.0, 30.0, -0.47198, 10.0), vec4<f32>(0.1, 0.12, 0.0, 0.003), vec4<f32>(1.0, 0.4, 0.7, 0.0), vec4<f32>(0.0, 0.3, -0.58, -0.58), vec4<f32>(0.2, 0.6, 0.4, 0.55), vec4<f32>(0.45, 0.0, 0.0, 0.0) ),
    // 30 CHIRAL FLIPPER
    array<vec4<f32>,6>( vec4<f32>(13.0, 10.0, -0.174533, 0.02), vec4<f32>(-0.3332, 0.1, 0.0, 0.001), vec4<f32>(1.0, 0.0, 1.0, 0.0), vec4<f32>(0.0, 0.3, 0.95, 0.95), vec4<f32>(0.2, -0.3, 1.3, -0.2), vec4<f32>(1.2, 0.0, 0.0, 0.0) ),
    // 31 CLOCK
    array<vec4<f32>,6>( vec4<f32>(7.0, 7.0, 0.0, 0.03), vec4<f32>(0.0, 0.0, 0.0, 0.0001), vec4<f32>(1.0, 0.0, 1.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.05, 0.5, 0.5, 0.5), vec4<f32>(0.5, 0.0, 0.0, 0.0) ),
    // 32 SLOPE SENSOR
    array<vec4<f32>,6>( vec4<f32>(9.0, 3.0, 0.1745329, 0.05), vec4<f32>(0.0, 0.0, 0.0, 0.00005), vec4<f32>(0.0, 0.8, 0.8, 0.0), vec4<f32>(0.0, 0.1, 0.6, 0.6), vec4<f32>(0.1, 0.5, 0.5, 0.5), vec4<f32>(0.5, 0.0, 0.0, 0.0) ),
    // 33 VAMPIRE MOUTH
    array<vec4<f32>,6>( vec4<f32>(18.5, 8.0, 0.03, 0.015), vec4<f32>(0.2, -0.33, 0.0, 0.002), vec4<f32>(1.0, 0.0, 0.0, 20.0), vec4<f32>(0.0, 0.3, 0.17, 0.17), vec4<f32>(0.2, 1.4, -0.4, 1.3), vec4<f32>(-0.3, 0.0, 0.0, 0.0) ),
    // 34 AGENT ALPHA SENSOR
    array<vec4<f32>,6>( vec4<f32>(13.0, 10.0, -0.24, 0.025), vec4<f32>(-0.3332, 0.1, 0.0, 0.0015), vec4<f32>(0.2, 0.0, 0.2, 0.0), vec4<f32>(0.0, 0.3, 0.95, 0.95), vec4<f32>(0.2, 0.35, 0.65, 0.4), vec4<f32>(0.6, 0.0, 0.0, 0.0) ),
    // 35 AGENT BETA SENSOR
    array<vec4<f32>,6>( vec4<f32>(11.5, 9.0, -0.523599, 0.03), vec4<f32>(-0.2, 0.52, 0.0, 0.0015), vec4<f32>(0.26, 0.26, 0.26, 0.0), vec4<f32>(0.0, 0.3, 0.08, 0.08), vec4<f32>(0.2, 0.25, 0.75, 0.3), vec4<f32>(0.7, 0.0, 0.0, 0.0) ),
    // 36 unused
    array<vec4<f32>,6>( vec4<f32>(10.0, 5.0, 0.0, 0.02), vec4<f32>(0.0, 0.0, 0.0, 0.001), vec4<f32>(0.5, 0.5, 0.5, 0.0), vec4<f32>(0.0, 0.2, 0.0, 0.0), vec4<f32>(0.2, 0.5, 0.5, 0.5), vec4<f32>(0.0, 0.0, 0.0, 0.0) ),
    // 37 TRAIL ENERGY SENSOR
    array<vec4<f32>,6>( vec4<f32>(11.0, 3.0, 0.3, 0.04), vec4<f32>(0.05, -0.1, 0.0, 0.00005), vec4<f32>(0.8, 0.6, 0.2, 0.0), vec4<f32>(0.0, 0.25, 0.5, 0.5), vec4<f32>(0.15, 0.6, 0.4, 0.7), vec4<f32>(0.4, 0.0, 0.0, 0.0) ),
    // 38 ALPHA MAGNITUDE SENSOR
    array<vec4<f32>,6>( vec4<f32>(10.0, 2.5, -0.1, 0.045), vec4<f32>(-0.05, -0.08, 0.0, 0.00005), vec4<f32>(0.2, 0.9, 0.3, 0.0), vec4<f32>(0.0, 0.2, 0.65, 0.65), vec4<f32>(0.1, 0.5, 0.5, 0.5), vec4<f32>(0.5, 0.0, 0.0, 0.0) ),
    // 39 ALPHA MAGNITUDE SENSOR (variant)
    array<vec4<f32>,6>( vec4<f32>(10.5, 2.8, 0.15, 0.05), vec4<f32>(-0.1, -0.03, 0.0, 0.00005), vec4<f32>(0.3, 1.0, 0.4, 0.0), vec4<f32>(0.0, 0.2, 0.68, 0.68), vec4<f32>(0.1, 0.5, 0.5, 0.5), vec4<f32>(0.5, 0.0, 0.0, 0.0) ),
    // 40 BETA MAGNITUDE SENSOR
    array<vec4<f32>,6>( vec4<f32>(10.0, 2.5, 0.25, 0.05), vec4<f32>(0.05, 0.15, 0.0, 0.00005), vec4<f32>(0.9, 0.2, 0.3, 0.0), vec4<f32>(0.0, 0.3, 0.38, 0.38), vec4<f32>(0.1, 0.5, 0.5, 0.5), vec4<f32>(0.5, 0.0, 0.0, 0.0) ),
    // 41 BETA MAGNITUDE SENSOR (variant)
    array<vec4<f32>,6>( vec4<f32>(10.5, 2.8, 0.4, 0.05), vec4<f32>(0.0, 0.12, 0.0, 0.00005), vec4<f32>(1.0, 0.3, 0.4, 0.0), vec4<f32>(0.0, 0.3, 0.40, 0.40), vec4<f32>(0.1, 0.5, 0.5, 0.5), vec4<f32>(0.5, 0.0, 0.0, 0.0) )
);

var<private> AMINO_FLAGS: array<u32, AMINO_COUNT> = array<u32, AMINO_COUNT>(
    0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, (1u<<9), (1u<<0), 0u, 0u, 0u, 0u, 0u, 0u, 0u, // 0â€“19
    (1u<<1), (1u<<0), (1u<<2), (1u<<3), (1u<<4), (1u<<8), (1u<<9), 0u, 0u, 0u, 0u, (1u<<7), 0u, (1u<<1), (1u<<5), (1u<<6), 0u, (1u<<11), (1u<<2), (1u<<2), (1u<<3), (1u<<3) // 20â€“41
);

fn get_amino_acid_properties(amino_type: u32) -> AminoAcidProperties {
    let t = min(amino_type, AMINO_COUNT - 1u);
    let d = AMINO_DATA[t];
    let f = AMINO_FLAGS[t];

    var p: AminoAcidProperties;
    p.segment_length = d[0].x; p.thickness = d[0].y; p.base_angle = d[0].z; p.mass = d[0].w;
    p.alpha_sensitivity = d[1].x; p.beta_sensitivity = d[1].y; p.thrust_force = d[1].z; p.energy_consumption = d[1].w;
    p.color = d[2].xyz; p.energy_storage = d[2].w;
    p.energy_absorption_rate = d[3].x; p.beta_absorption_rate = d[3].y; p.beta_damage = d[3].z; p.parameter1 = d[3].w;
    p.signal_decay = d[4].x; p.alpha_left_mult = d[4].y; p.alpha_right_mult = d[4].z; p.beta_left_mult = d[4].w;
    p.beta_right_mult = d[5].x;

    p.is_propeller          = (f & (1u<<0))  != 0u;
    p.is_mouth              = (f & (1u<<1))  != 0u;
    p.is_alpha_sensor       = (f & (1u<<2))  != 0u;
    p.is_beta_sensor        = (f & (1u<<3))  != 0u;
    p.is_energy_sensor      = (f & (1u<<4))  != 0u;
    p.is_agent_alpha_sensor = (f & (1u<<5))  != 0u;
    p.is_agent_beta_sensor  = (f & (1u<<6))  != 0u;
    p.is_clock              = (f & (1u<<7))  != 0u;
    p.is_displacer          = (f & (1u<<8))  != 0u;
    p.is_inhibitor          = (f & (1u<<9))  != 0u; // enabler
    p.is_condenser          = (f & (1u<<10)) != 0u;
    p.is_trail_energy_sensor = (f & (1u<<11)) != 0u;

    return p;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

fn is_in_bounds(pos: vec2<f32>) -> bool {
    let ws = f32(SIM_SIZE);
    return pos.x >= 0.0 && pos.x <= ws && pos.y >= 0.0 && pos.y <= ws;
}

fn clamp_position(pos: vec2<f32>) -> vec2<f32> {
    let ws = f32(SIM_SIZE);
    return vec2<f32>(
        clamp(pos.x, 0.0, ws),
        clamp(pos.y, 0.0, ws)
    );
}

fn grid_index(pos: vec2<f32>) -> u32 {
    let clamped = clamp_position(pos);
    let scale = f32(SIM_SIZE) / f32(ENV_GRID_SIZE);
    var x: i32 = i32(clamped.x / scale);
    var y: i32 = i32(clamped.y / scale);
    x = clamp(x, 0, i32(ENV_GRID_SIZE) - 1);
    y = clamp(y, 0, i32(ENV_GRID_SIZE) - 1);
    return u32(y) * ENV_GRID_SIZE + u32(x);
}

struct PartName {
    chars: array<u32, 6>,
    len: u32,
};

fn get_part_name(part_type: u32) -> PartName {
    var name = PartName(array<u32, 6>(63u, 63u, 63u, 63u, 63u, 63u), 3u); // "???"

    switch (part_type) {
        // Amino acids (0-19) - single-letter codes
        case 0u: { name = PartName(array<u32,6>(65u,32u,32u,32u,32u,32u), 1u); }
        case 1u: { name = PartName(array<u32,6>(67u,32u,32u,32u,32u,32u), 1u); }
        case 2u: { name = PartName(array<u32,6>(68u,32u,32u,32u,32u,32u), 1u); }
        case 3u: { name = PartName(array<u32,6>(69u,32u,32u,32u,32u,32u), 1u); }
        case 4u: { name = PartName(array<u32,6>(70u,32u,32u,32u,32u,32u), 1u); }
        case 5u: { name = PartName(array<u32,6>(71u,32u,32u,32u,32u,32u), 1u); }
        case 6u: { name = PartName(array<u32,6>(72u,32u,32u,32u,32u,32u), 1u); }
        case 7u: { name = PartName(array<u32,6>(73u,32u,32u,32u,32u,32u), 1u); }
        case 8u: { name = PartName(array<u32,6>(75u,32u,32u,32u,32u,32u), 1u); }
        case 9u: { name = PartName(array<u32,6>(76u,32u,32u,32u,32u,32u), 1u); }
        case 10u:{ name = PartName(array<u32,6>(77u,32u,32u,32u,32u,32u), 1u); }
        case 11u:{ name = PartName(array<u32,6>(78u,32u,32u,32u,32u,32u), 1u); }
        case 12u:{ name = PartName(array<u32,6>(80u,32u,32u,32u,32u,32u), 1u); }
        case 13u:{ name = PartName(array<u32,6>(81u,32u,32u,32u,32u,32u), 1u); }
        case 14u:{ name = PartName(array<u32,6>(82u,32u,32u,32u,32u,32u), 1u); }
        case 15u:{ name = PartName(array<u32,6>(83u,32u,32u,32u,32u,32u), 1u); }
        case 16u:{ name = PartName(array<u32,6>(84u,32u,32u,32u,32u,32u), 1u); }
        case 17u:{ name = PartName(array<u32,6>(86u,32u,32u,32u,32u,32u), 1u); }
        case 18u:{ name = PartName(array<u32,6>(87u,32u,32u,32u,32u,32u), 1u); }
        case 19u:{ name = PartName(array<u32,6>(89u,32u,32u,32u,32u,32u), 1u); }

        // Organs (20-31) - clearer names
        case 20u:{ name = PartName(array<u32,6>(77u,79u,85u,84u,72u,32u), 5u); }
        case 21u:{ name = PartName(array<u32,6>(80u,82u,79u,80u,32u,32u), 4u); }
        case 22u:{ name = PartName(array<u32,6>(65u,76u,80u,72u,65u,32u), 5u); }
        case 23u:{ name = PartName(array<u32,6>(66u,69u,84u,65u,32u,32u), 4u); }
        case 24u:{ name = PartName(array<u32,6>(69u,78u,69u,82u,71u,89u), 6u); }
        case 25u:{ name = PartName(array<u32,6>(68u,73u,83u,80u,32u,32u), 4u); }
        case 26u:{ name = PartName(array<u32,6>(69u,78u,65u,66u,76u,32u), 5u); }
        case 28u:{ name = PartName(array<u32,6>(83u,84u,79u,82u,69u,32u), 5u); }
        case 29u:{ name = PartName(array<u32,6>(80u,79u,73u,83u,78u,32u), 5u); }
        case 30u:{ name = PartName(array<u32,6>(70u,76u,73u,80u,32u,32u), 4u); }
        case 31u:{ name = PartName(array<u32,6>(67u,76u,79u,67u,75u,32u), 5u); }
        case 32u:{ name = PartName(array<u32,6>(83u,76u,79u,80u,69u,32u), 5u); }
        case 33u:{ name = PartName(array<u32,6>(86u,77u,80u,82u,69u,32u), 5u); }
        case 34u:{ name = PartName(array<u32,6>(65u,71u,83u,78u,65u,32u), 5u); }
        case 35u:{ name = PartName(array<u32,6>(65u,71u,83u,78u,66u,32u), 5u); }
        case 36u:{ name = PartName(array<u32,6>(80u,65u,73u,82u,71u,32u), 5u); }
        case 38u:{ name = PartName(array<u32,6>(65u,77u,65u,71u,32u,32u), 4u); }
        case 39u:{ name = PartName(array<u32,6>(65u,77u,65u,71u,50u,32u), 5u); }
        case 40u:{ name = PartName(array<u32,6>(66u,77u,65u,71u,32u,32u), 4u); }
        case 41u:{ name = PartName(array<u32,6>(66u,77u,65u,71u,50u,32u), 5u); }
        default: { }
    }

    return name;
}

// ============================================================================
// GRID INTERPOLATION HELPERS
// ============================================================================

fn sample_grid_bilinear(pos: vec2<f32>, grid_type: u32) -> f32 {
    let clamped = clamp_position(pos);
    let scale = f32(SIM_SIZE) / f32(GRID_SIZE);
    let grid_x = (clamped.x / scale) - 0.5;
    let grid_y = (clamped.y / scale) - 0.5;

    let x0 = i32(floor(grid_x));
    let y0 = i32(floor(grid_y));
    let x1 = min(x0 + 1, i32(GRID_SIZE) - 1);
    let y1 = min(y0 + 1, i32(GRID_SIZE) - 1);

    let fx = fract(grid_x);
    let fy = fract(grid_y);

    let idx00 = u32(clamp(y0, 0, i32(GRID_SIZE) - 1)) * GRID_SIZE + u32(clamp(x0, 0, i32(GRID_SIZE) - 1));
    let idx10 = u32(clamp(y0, 0, i32(GRID_SIZE) - 1)) * GRID_SIZE + u32(clamp(x1, 0, i32(GRID_SIZE) - 1));
    let idx01 = u32(clamp(y1, 0, i32(GRID_SIZE) - 1)) * GRID_SIZE + u32(clamp(x0, 0, i32(GRID_SIZE) - 1));
    let idx11 = u32(clamp(y1, 0, i32(GRID_SIZE) - 1)) * GRID_SIZE + u32(clamp(x1, 0, i32(GRID_SIZE) - 1));

    var v00: f32;
    var v10: f32;
    var v01: f32;
    var v11: f32;

    if (grid_type == 0u) {
        v00 = alpha_grid[idx00]; v10 = alpha_grid[idx10]; v01 = alpha_grid[idx01]; v11 = alpha_grid[idx11];
    } else if (grid_type == 1u) {
        v00 = beta_grid[idx00]; v10 = beta_grid[idx10]; v01 = beta_grid[idx01]; v11 = beta_grid[idx11];
    } else {
        v00 = read_gamma_height(idx00); v10 = read_gamma_height(idx10); v01 = read_gamma_height(idx01); v11 = read_gamma_height(idx11);
    }

    let v0 = mix(v00, v10, fx);
    let v1 = mix(v01, v11, fx);
    return mix(v0, v1, fy);
}

fn cubic_hermite(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    return vec4<f32>(
        (-t3 + 3.0 * t2 - 3.0 * t + 1.0) / 6.0,
        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
        t3 / 6.0
    );
}

fn sample_grid_bicubic(pos: vec2<f32>, grid_type: u32) -> f32 {
    let clamped = clamp_position(pos);
    let scale = f32(SIM_SIZE) / f32(GRID_SIZE);
    let grid_x = (clamped.x / scale) - 0.5;
    let grid_y = (clamped.y / scale) - 0.5;

    let x_floor = floor(grid_x);
    let y_floor = floor(grid_y);
    let fx = grid_x - x_floor;
    let fy = grid_y - y_floor;

    let x = i32(x_floor);
    let y = i32(y_floor);

    var values: array<f32, 16>;
    for (var j = -1; j <= 2; j++) {
        for (var i = -1; i <= 2; i++) {
            let sx = clamp(x + i, 0, i32(GRID_SIZE) - 1);
            let sy = clamp(y + j, 0, i32(GRID_SIZE) - 1);
            let idx = u32(sy) * GRID_SIZE + u32(sx);

            if (grid_type == 0u) {
                values[(j + 1) * 4 + (i + 1)] = alpha_grid[idx];
            } else if (grid_type == 1u) {
                values[(j + 1) * 4 + (i + 1)] = beta_grid[idx];
            } else {
                values[(j + 1) * 4 + (i + 1)] = read_gamma_height(idx);
            }
        }
    }

    let wx = cubic_hermite(fx);
    let wy = cubic_hermite(fy);

    var result = 0.0;
    for (var j = 0; j < 4; j++) {
        var row_sum = 0.0;
        for (var i = 0; i < 4; i++) {
            row_sum += values[j * 4 + i] * wx[i];
        }
        result += row_sum * wy[j];
    }

    return clamp(result, 0.0, 1.0);
}

const GAMMA_LAYER_SIZE: u32 = GRID_SIZE * GRID_SIZE;
const GAMMA_SLOPE_X_OFFSET: u32 = GAMMA_LAYER_SIZE;
const GAMMA_SLOPE_Y_OFFSET: u32 = GAMMA_LAYER_SIZE * 2u;

fn clamp_gamma_coords(ix: i32, iy: i32) -> u32 {
    let clamped_x = clamp(ix, 0, i32(GRID_SIZE) - 1);
    let clamped_y = clamp(iy, 0, i32(GRID_SIZE) - 1);
    return u32(clamped_y) * GRID_SIZE + u32(clamped_x);
}

fn read_gamma_texel(ix: i32, iy: i32) -> f32 {
    let idx = clamp_gamma_coords(ix, iy);
    return gamma_grid[idx];
}

fn read_gamma_height(idx: u32) -> f32 { return gamma_grid[idx]; }
fn write_gamma_height(idx: u32, value: f32) { gamma_grid[idx] = clamp(value, 0.0, 1.0); }
fn read_gamma_slope(idx: u32) -> vec2<f32> { return vec2<f32>(gamma_grid[idx + GAMMA_SLOPE_X_OFFSET], gamma_grid[idx + GAMMA_SLOPE_Y_OFFSET]); }
fn write_gamma_slope(idx: u32, slope: vec2<f32>) { gamma_grid[idx + GAMMA_SLOPE_X_OFFSET] = slope.x; gamma_grid[idx + GAMMA_SLOPE_Y_OFFSET] = slope.y; }

fn read_combined_height(ix: i32, iy: i32) -> f32 {
    let idx = clamp_gamma_coords(ix, iy);
    var height = gamma_grid[idx];
    if (params.chemical_slope_scale_alpha != 0.0) { height += alpha_grid[idx] * params.chemical_slope_scale_alpha; }
    if (params.chemical_slope_scale_beta != 0.0) { height += beta_grid[idx] * params.chemical_slope_scale_beta; }
    return height;
}

fn hash(v: u32) -> u32 {
    var x = v;
    x = x ^ (x >> 16u); x = x * 0x7feb352du; x = x ^ (x >> 15u); x = x * 0x846ca68bu; x = x ^ (x >> 16u);
    return x;
}

fn hash_f32(v: u32) -> f32 { return f32(hash(v)) / 4294967295.0; }

// Sensor and sampling helpers
fn sample_stochastic_gaussian(center: vec2<f32>, base_radius: f32, seed: u32, grid_type: u32, debug_mode: bool, sensor_perpendicular: vec2<f32>, promoter_param1: f32, modifier_param1: f32) -> f32 {
    let sample_count = 14u;
    let combined_param = promoter_param1 + modifier_param1;
    let radius = base_radius * abs(combined_param);
    let signal_polarity = select(1.0, -1.0, combined_param < 0.0);

    var weighted_sum = 0.0;
    var weight_total = 0.0;

    for (var i = 0u; i < sample_count; i++) {
        let h1 = hash_f32(seed * 1000u + i * 17u);
        let h2 = hash_f32(seed * 1000u + i * 23u + 13u);

        let angle = h1 * 6.28318530718; // 2*PI
        let dist = sqrt(h2) * radius;

        let offset = vec2<f32>(cos(angle), sin(angle)) * dist;
        let sample_pos = center + offset;
        if (!is_in_bounds(sample_pos)) { continue; }
        let idx = grid_index(sample_pos);

        let sigma = radius * 0.15;
        let distance_weight = exp(-(dist * dist) / (2.0 * sigma * sigma));

        let direction = select(vec2<f32>(0.0), normalize(offset), dist > 1e-5);
        let directional_weight = dot(sensor_perpendicular, direction);
        let weight = distance_weight * directional_weight;

        var sample_value = 0.0;
        if (grid_type == 0u) { sample_value = alpha_grid[idx]; }
        else if (grid_type == 1u) { sample_value = beta_grid[idx]; }

        if (debug_mode) {
            let dot_val = directional_weight;
            let red_intensity = clamp(dot_val, 0.0, 1.0);
            let blue_intensity = clamp(-dot_val, 0.0, 1.0);
            let debug_color = vec4<f32>(red_intensity, 0.0, blue_intensity, 0.7);
            let sample_size = clamp(radius * 0.03, 2.0, 8.0);
            draw_filled_circle(sample_pos, sample_size, debug_color);
        }

        weighted_sum += sample_value * weight;
        weight_total += weight;
    }

    return weighted_sum * signal_polarity;
}

fn sample_magnitude_only(center: vec2<f32>, base_radius: f32, seed: u32, grid_type: u32, debug_mode: bool, promoter_param1: f32, modifier_param1: f32) -> f32 {
    let sample_count = 14u;
    let combined_param = promoter_param1 + modifier_param1;
    let radius = base_radius * abs(combined_param);
    let signal_polarity = select(1.0, -1.0, combined_param < 0.0);

    var weighted_sum = 0.0;
    var weight_total = 0.0;

    for (var i = 0u; i < sample_count; i++) {
        let h1 = hash_f32(seed * 1000u + i * 17u);
        let h2 = hash_f32(seed * 1000u + i * 23u + 13u);

        let angle = h1 * 6.28318530718;
        let dist = sqrt(h2) * radius;

        let offset = vec2<f32>(cos(angle), sin(angle)) * dist;
        let sample_pos = center + offset;
        if (!is_in_bounds(sample_pos)) { continue; }
        let idx = grid_index(sample_pos);

        let sigma = radius * 0.15;
        let weight = exp(-(dist * dist) / (2.0 * sigma * sigma));

        var sample_value = 0.0;
        if (grid_type == 0u) { sample_value = alpha_grid[idx]; }
        else if (grid_type == 1u) { sample_value = beta_grid[idx]; }

        if (debug_mode) {
            let debug_color = vec4<f32>(1.0, 0.7, 0.0, 0.7);
            let sample_size = clamp(radius * 0.03, 2.0, 8.0);
            draw_filled_circle(sample_pos, sample_size, debug_color);
        }

        weighted_sum += sample_value * weight;
        weight_total += weight;
    }

    let result = select(0.0, weighted_sum / weight_total, weight_total > 1e-6);
    return result * signal_polarity;
}

fn sample_neighbors_color(center: vec2<f32>, base_radius: f32, debug_mode: bool, sensor_perpendicular: vec2<f32>, agent_color: vec3<f32>, neighbor_ids: ptr<function, array<u32, 64>>, neighbor_count: u32) -> f32 {
    let radius = base_radius;
    var weighted_sum = 0.0;

    for (var n = 0u; n < neighbor_count; n++) {
        let other_agent = agents_in[(*neighbor_ids)[n]];
        let offset = other_agent.position - center;
        let dist = length(offset);
        if (dist <= radius && dist > 0.001) {
            let sigma = radius * 0.15;
            let distance_weight = exp(-(dist * dist) / (2.0 * sigma * sigma));
            let direction = normalize(offset);
            let directional_alignment = dot(sensor_perpendicular, direction);
            let weight = distance_weight * directional_alignment;

            let trail_idx = grid_index(other_agent.position);
            let trail_color = trail_grid[trail_idx].xyz;

            let trail_color_normalized = normalize(trail_color + vec3<f32>(1e-6));
            let agent_color_normalized = normalize(agent_color + vec3<f32>(1e-6));

            let color_diff_vec = trail_color_normalized - agent_color_normalized;
            let color_difference = length(color_diff_vec);

            if (debug_mode) {
                let dot_val = directional_alignment;
                let red_intensity = clamp(dot_val, 0.0, 1.0);
                let blue_intensity = clamp(-dot_val, 0.0, 1.0);
                let debug_color = vec4<f32>(red_intensity, 0.0, blue_intensity, 0.7);
                draw_filled_circle(other_agent.position, 5.0, debug_color);
            }

            weighted_sum += color_difference * weight;
        }
    }

    return weighted_sum;
}

fn sample_neighbors_energy(center: vec2<f32>, base_radius: f32, debug_mode: bool, sensor_perpendicular: vec2<f32>, neighbor_ids: ptr<function, array<u32, 64>>, neighbor_count: u32) -> f32 {
    let radius = base_radius;
    var weighted_sum = 0.0;

    for (var n = 0u; n < neighbor_count; n++) {
        let other_agent = agents_in[(*neighbor_ids)[n]];
        let offset = other_agent.position - center;
        let dist = length(offset);
        if (dist <= radius && dist > 0.001) {
            let sigma = radius * 0.15;
            let distance_weight = exp(-(dist * dist) / (2.0 * sigma * sigma));
            let direction = normalize(offset);
            let directional_alignment = dot(sensor_perpendicular, direction);
            let weight = distance_weight * directional_alignment;

            let trail_idx = grid_index(other_agent.position);
            let energy_value = trail_grid[trail_idx].w;

            if (debug_mode) {
                let dot_val = directional_alignment;
                let red_intensity = clamp(dot_val, 0.0, 1.0);
                let blue_intensity = clamp(-dot_val, 0.0, 1.0);
                let debug_color = vec4<f32>(red_intensity, 0.0, blue_intensity, 0.7);
                draw_filled_circle(other_agent.position, 5.0, debug_color);
            }

            weighted_sum += energy_value * weight;
        }
    }

    return weighted_sum;
}

fn noise2d(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash_f32(u32(i.x) + u32(i.y) * 57u);
    let b = hash_f32(u32(i.x + 1.0) + u32(i.y) * 57u);
    let c = hash_f32(u32(i.x) + u32(i.y + 1.0) * 57u);
    let d = hash_f32(u32(i.x + 1.0) + u32(i.y + 1.0) * 57u);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn noise3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let h000 = hash_f32(u32(i.x) * 73856093u ^ u32(i.y) * 19349663u ^ u32(i.z) * 83492791u);
    let h100 = hash_f32(u32(i.x + 1.0) * 73856093u ^ u32(i.y) * 19349663u ^ u32(i.z) * 83492791u);
    let h010 = hash_f32(u32(i.x) * 73856093u ^ u32(i.y + 1.0) * 19349663u ^ u32(i.z) * 83492791u);
    let h110 = hash_f32(u32(i.x + 1.0) * 73856093u ^ u32(i.y + 1.0) * 19349663u ^ u32(i.z) * 83492791u);
    let h001 = hash_f32(u32(i.x) * 73856093u ^ u32(i.y) * 19349663u ^ u32(i.z + 1.0) * 83492791u);
    let h101 = hash_f32(u32(i.x + 1.0) * 73856093u ^ u32(i.y) * 19349663u ^ u32(i.z + 1.0) * 83492791u);
    let h011 = hash_f32(u32(i.x) * 73856093u ^ u32(i.y + 1.0) * 19349663u ^ u32(i.z + 1.0) * 83492791u);
    let h111 = hash_f32(u32(i.x + 1.0) * 73856093u ^ u32(i.y + 1.0) * 19349663u ^ u32(i.z + 1.0) * 83492791u);

    let x00 = mix(h000, h100, u.x);
    let x10 = mix(h010, h110, u.x);
    let x01 = mix(h001, h101, u.x);
    let x11 = mix(h011, h111, u.x);
    let y0 = mix(x00, x10, u.y);
    let y1 = mix(x01, x11, u.y);
    return mix(y0, y1, u.z);
}

fn layered_noise(coord: vec2<f32>, seed: u32, octaves: u32, scale: f32, contrast: f32) -> f32 {
    let octave_count = max(octaves, 1u);
    var amplitude = 1.0;
    var frequency = 1.0;
    var sum = 0.0;
    var total = 0.0;
    var octave_seed = seed ^ 0x9E3779B1u;
    let safe_scale = max(scale, 0.0001);

    for (var i = 0u; i < octave_count; i = i + 1u) {
        let offset = vec2<f32>(hash_f32(octave_seed ^ 0xA511E9B5u) * 512.0, hash_f32(octave_seed ^ 0x63D3F6ABu) * 512.0);
        sum = sum + noise2d(coord * frequency * safe_scale + offset) * amplitude;
        total = total + amplitude;
        amplitude = amplitude * 0.5;
        frequency = frequency * 2.0;
        octave_seed = hash(octave_seed ^ i);
    }

    let normalized = sum / max(total, 0.0001);
    return clamp((normalized - 0.5) * contrast + 0.5, 0.0, 1.0);
}

fn layered_noise_3d(coord: vec3<f32>, seed: u32, octaves: u32, scale: f32, contrast: f32) -> f32 {
    let octave_count = max(octaves, 1u);
    var amplitude = 1.0;
    var frequency = 1.0;
    var sum = 0.0;
    var total = 0.0;
    var octave_seed = seed ^ 0x9E3779B1u;
    let safe_scale = max(scale, 0.0001);

    for (var i = 0u; i < octave_count; i = i + 1u) {
        let offset = vec3<f32>(
            hash_f32(octave_seed ^ 0xA511E9B5u) * 512.0,
            hash_f32(octave_seed ^ 0x63D3F6ABu) * 512.0,
            hash_f32(octave_seed ^ 0x7C159E3Du) * 512.0);
        sum = sum + noise3d(coord * frequency * safe_scale + offset) * amplitude;
        total = total + amplitude;
        amplitude = amplitude * 0.5;
        frequency = frequency * 2.0;
        octave_seed = hash(octave_seed ^ i);
    }

    let normalized = sum / max(total, 0.0001);
    return clamp((normalized - 0.5) * contrast + 0.5, 0.0, 1.0);
}

fn remap_unit(value: f32, range: vec2<f32>) -> f32 { return mix(range.x, range.y, value); }

fn rotate_vec2(v: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle); let s = sin(angle);
    return vec2<f32>(v.x * c - v.y * s, v.x * s + v.y * c);
}

fn apply_agent_rotation(v: vec2<f32>, angle: f32) -> vec2<f32> {
    if (DISABLE_GLOBAL_ROTATION) { return v; }
    return rotate_vec2(v, angle);
}

fn world_to_screen(world_pos: vec2<f32>) -> vec2<i32> {
    let safe_zoom = max(params.camera_zoom, 0.0001);
    let safe_width = max(params.window_width, 1.0);
    let safe_height = max(params.window_height, 1.0);
    let aspect_ratio = safe_width / safe_height;
    let view_width = params.grid_size / safe_zoom;
    let view_height = view_width / aspect_ratio;
    let cam_min_x = params.camera_pan_x - view_width * 0.5;
    let cam_min_y = params.camera_pan_y - view_height * 0.5;

    let norm_x = (world_pos.x - cam_min_x) / view_width;
    let norm_y = (world_pos.y - cam_min_y) / view_height;

    let screen_x = i32(norm_x * safe_width);
    let screen_y = i32(norm_y * safe_height);
    return vec2<i32>(screen_x, screen_y);
}

fn get_visible_position(world_pos: vec2<f32>) -> vec2<f32> {
    let ws = f32(SIM_SIZE);
    let safe_zoom = max(params.camera_zoom, 0.0001);
    let safe_width = max(params.window_width, 1.0);
    let safe_height = max(params.window_height, 1.0);
    let aspect_ratio = safe_width / safe_height;
    let view_width = params.grid_size / safe_zoom;
    let view_height = view_width / aspect_ratio;
    let cam_min_x = params.camera_pan_x - view_width * 0.5;
    let cam_max_x = params.camera_pan_x + view_width * 0.5;
    let cam_min_y = params.camera_pan_y - view_height * 0.5;
    let cam_max_y = params.camera_pan_y + view_height * 0.5;

    for (var wrap_x = -1; wrap_x <= 1; wrap_x++) {
        for (var wrap_y = -1; wrap_y <= 1; wrap_y++) {
            let test_pos = vec2<f32>(world_pos.x + f32(wrap_x) * ws, world_pos.y + f32(wrap_y) * ws);
            if (test_pos.x >= cam_min_x - 50.0 && test_pos.x <= cam_max_x + 50.0 && test_pos.y >= cam_min_y - 50.0 && test_pos.y <= cam_max_y + 50.0) {
                return test_pos;
            }
        }
    }
    return world_pos;
}

fn screen_to_grid_index(screen_pos: vec2<i32>) -> u32 {
    let x = u32(clamp(screen_pos.x, 0, i32(params.window_width) - 1));
    let y = u32(clamp(screen_pos.y, 0, i32(params.window_height) - 1));
    return y * params.visual_stride + x;
}

fn rna_complement(base: u32) -> u32 {
    if (base == 88u) { return 88u; }
    if (base == 65u) { return 85u; }
    else if (base == 85u) { return 65u; }
    else if (base == 67u) { return 71u; }
    else if (base == 71u) { return 67u; }
    else { return base; }
}

fn get_random_rna_base(seed: u32) -> u32 {
    let choice = hash(seed) % 4u;
    if (choice == 0u) { return 65u; }
    else if (choice == 1u) { return 85u; }
    else if (choice == 2u) { return 71u; }
    else { return 67u; }
}

fn codon_to_amino_index(b0: u32, b1: u32, b2: u32) -> u32 {
    if (b0 == 85u) {
        if (b1 == 85u) { if (b2 == 85u || b2 == 67u) { return 4u; } return 9u; }
        if (b1 == 67u) { return 15u; }
        if (b1 == 65u) { if (b2 == 85u || b2 == 67u) { return 19u; } return 19u; }
        if (b1 == 71u) { if (b2 == 85u || b2 == 67u) { return 1u; } if (b2 == 71u) { return 18u; } return 1u; }
    }
    if (b0 == 67u) {
        if (b1 == 85u) { return 9u; }
        if (b1 == 67u) { return 12u; }
        if (b1 == 65u) { if (b2 == 85u || b2 == 67u) { return 6u; } return 13u; }
        if (b1 == 71u) { return 14u; }
    }
    if (b0 == 65u) {
        if (b1 == 85u) { if (b2 == 71u) { return 10u; } return 7u; }
        if (b1 == 67u) { return 16u; }
        if (b1 == 65u) { if (b2 == 85u || b2 == 67u) { return 11u; } return 8u; }
        if (b1 == 71u) { if (b2 == 85u || b2 == 67u) { return 15u; } return 14u; }
    }
    if (b0 == 71u) {
        if (b1 == 85u) { return 17u; }
        if (b1 == 67u) { return 0u; }
        if (b1 == 65u) { if (b2 == 85u || b2 == 67u) { return 2u; } return 3u; }
        if (b1 == 71u) { return 5u; }
    }
    return 0u;
}

// Genome helpers and translation
fn genome_read_word(genome: array<u32, GENOME_WORDS>, index: u32) -> u32 {
    switch (index) {
        case 0u:  { return genome[0u]; }
        case 1u:  { return genome[1u]; }
        case 2u:  { return genome[2u]; }
        case 3u:  { return genome[3u]; }
        case 4u:  { return genome[4u]; }
        case 5u:  { return genome[5u]; }
        case 6u:  { return genome[6u]; }
        case 7u:  { return genome[7u]; }
        case 8u:  { return genome[8u]; }
        case 9u:  { return genome[9u]; }
        case 10u: { return genome[10u]; }
        case 11u: { return genome[11u]; }
        case 12u: { return genome[12u]; }
        case 13u: { return genome[13u]; }
        case 14u: { return genome[14u]; }
        case 15u: { return genome[15u]; }
        case 16u: { return genome[16u]; }
        case 17u: { return genome[17u]; }
        case 18u: { return genome[18u]; }
        case 19u: { return genome[19u]; }
        case 20u: { return genome[20u]; }
        case 21u: { return genome[21u]; }
        case 22u: { return genome[22u]; }
        case 23u: { return genome[23u]; }
        case 24u: { return genome[24u]; }
        case 25u: { return genome[25u]; }
        case 26u: { return genome[26u]; }
        case 27u: { return genome[27u]; }
        case 28u: { return genome[28u]; }
        case 29u: { return genome[29u]; }
        case 30u: { return genome[30u]; }
        case 31u: { return genome[31u]; }
        case 32u: { return genome[32u]; }
        case 33u: { return genome[33u]; }
        case 34u: { return genome[34u]; }
        case 35u: { return genome[35u]; }
        case 36u: { return genome[36u]; }
        case 37u: { return genome[37u]; }
        case 38u: { return genome[38u]; }
        case 39u: { return genome[39u]; }
        case 40u: { return genome[40u]; }
        case 41u: { return genome[41u]; }
        case 42u: { return genome[42u]; }
        case 43u: { return genome[43u]; }
        case 44u: { return genome[44u]; }
        case 45u: { return genome[45u]; }
        case 46u: { return genome[46u]; }
        case 47u: { return genome[47u]; }
        case 48u: { return genome[48u]; }
        case 49u: { return genome[49u]; }
        case 50u: { return genome[50u]; }
        case 51u: { return genome[51u]; }
        case 52u: { return genome[52u]; }
        case 53u: { return genome[53u]; }
        case 54u: { return genome[54u]; }
        case 55u: { return genome[55u]; }
        case 56u: { return genome[56u]; }
        case 57u: { return genome[57u]; }
        case 58u: { return genome[58u]; }
        case 59u: { return genome[59u]; }
        case 60u: { return genome[60u]; }
        case 61u: { return genome[61u]; }
        case 62u: { return genome[62u]; }
        case 63u: { return genome[63u]; }
        default: { return genome[63u]; }
    }
}

fn packed_read_word(packed: array<u32, PACKED_GENOME_WORDS>, index: u32) -> u32 {
    switch (index) {
        case 0u:  { return packed[0u]; }
        case 1u:  { return packed[1u]; }
        case 2u:  { return packed[2u]; }
        case 3u:  { return packed[3u]; }
        case 4u:  { return packed[4u]; }
        case 5u:  { return packed[5u]; }
        case 6u:  { return packed[6u]; }
        case 7u:  { return packed[7u]; }
        case 8u:  { return packed[8u]; }
        case 9u:  { return packed[9u]; }
        case 10u: { return packed[10u]; }
        case 11u: { return packed[11u]; }
        case 12u: { return packed[12u]; }
        case 13u: { return packed[13u]; }
        case 14u: { return packed[14u]; }
        case 15u: { return packed[15u]; }
        default: { return packed[15u]; }
    }
}

fn genome_get_base_ascii(genome: array<u32, GENOME_WORDS>, index: u32) -> u32 {
    if (index >= GENOME_LENGTH) { return 0u; }
    let w = index / 4u; let o = index % 4u; let word_val = genome_read_word(genome, w); return (word_val >> (o * 8u)) & 0xFFu;
}

fn genome_get_codon_ascii(genome: array<u32, GENOME_WORDS>, index: u32) -> vec3<u32> {
    return vec3<u32>(
        genome_get_base_ascii(genome, index),
        genome_get_base_ascii(genome, index + 1u),
        genome_get_base_ascii(genome, index + 2u)
    );
}

fn base_ascii_to_2bit(b: u32) -> u32 {
    if (b == 65u) { return 0u; }
    if (b == 85u) { return 1u; }
    if (b == 71u) { return 2u; }
    if (b == 67u) { return 3u; }
    return 0u;
}

fn base_2bit_to_ascii(v: u32) -> u32 {
    switch (v & 3u) {
        case 0u: { return 65u; }
        case 1u: { return 85u; }
        case 2u: { return 71u; }
        default: { return 67u; }
    }
}

fn genome_get_base_packed(packed: array<u32, PACKED_GENOME_WORDS>, index: u32) -> u32 {
    if (index >= GENOME_LENGTH) { return 0u; }
    let word_index = index / PACKED_BASES_PER_WORD;
    let bit_index = (index % PACKED_BASES_PER_WORD) * 2u;
    let word_val = packed_read_word(packed, word_index);
    let two_bits = (word_val >> bit_index) & 0x3u;
    return base_2bit_to_ascii(two_bits);
}

fn genome_pack_into(agent_in: Agent) -> array<u32, PACKED_GENOME_WORDS> {
    var out: array<u32, PACKED_GENOME_WORDS>;
    for (var i = 0u; i < GENOME_LENGTH; i++) {
        let b = genome_get_base_ascii(agent_in.genome, i);
        let v = base_ascii_to_2bit(b);
        let wi = i / PACKED_BASES_PER_WORD;
        let bi = (i % PACKED_BASES_PER_WORD) * 2u;
        let current_word = out[wi];
        out[wi] = current_word | (v << bi);
    }
    return out;
}

fn genome_find_start_codon(genome: array<u32, GENOME_WORDS>) -> u32 {
    for (var i = 0u; i < GENOME_LENGTH - 2u; i++) {
        let b0 = genome_get_base_ascii(genome, i);
        let b1 = genome_get_base_ascii(genome, i + 1u);
        let b2 = genome_get_base_ascii(genome, i + 2u);
        if (b0 == 65u && b1 == 85u && b2 == 71u) { return i; }
    }
    return 0xFFFFFFFFu;
}

fn genome_find_first_coding_triplet(genome: array<u32, GENOME_WORDS>) -> u32 {
    var i = 0u;
    loop {
        if (i + 2u >= GENOME_LENGTH) { break; }
        let codon = genome_get_codon_ascii(genome, i);
        if (codon.x == 88u || codon.y == 88u || codon.z == 88u) { i = i + 1u; continue; }
        return i;
    }
    return 0xFFFFFFFFu;
}

fn genome_is_stop_codon_at(genome: array<u32, GENOME_WORDS>, index: u32) -> bool {
    if (index + 2u >= GENOME_LENGTH) { return true; }
    let c = genome_get_codon_ascii(genome, index);
    if (c.x == 88u || c.y == 88u || c.z == 88u) { return true; }
    return (c.x == 85u && c.y == 65u && (c.z == 65u || c.z == 71u)) || (c.x == 85u && c.y == 71u && c.z == 65u);
}

struct TranslationStep {
    part_type: u32,
    bases_consumed: u32,
    is_stop: bool,
    is_valid: bool,
}

fn translate_codon_step(genome: array<u32, GENOME_WORDS>, pos_b: u32, ignore_stop_codons: bool) -> TranslationStep {
    var result: TranslationStep;
    result.part_type = 0u; result.bases_consumed = 3u; result.is_stop = false; result.is_valid = false;

    if (pos_b + 2u >= GENOME_LENGTH) { return result; }

    let is_stop_or_x = genome_is_stop_codon_at(genome, pos_b);
    if (is_stop_or_x) {
        let c = genome_get_codon_ascii(genome, pos_b);
        let has_x = (c.x == 88u || c.y == 88u || c.z == 88u);
        result.is_stop = !has_x;
        result.is_valid = ignore_stop_codons && !has_x;
        return result;
    }

    let codon = genome_get_codon_ascii(genome, pos_b);
    let amino_type = codon_to_amino_index(codon.x, codon.y, codon.z);

    let is_promoter = (amino_type == 9u || amino_type == 12u ||
                      amino_type == 8u || amino_type == 1u ||
                      amino_type == 17u || amino_type == 10u ||
                      amino_type == 6u || amino_type == 13u);

    var final_part_type = amino_type;
    var bases_consumed = 3u;

    if (is_promoter && pos_b + 5u < GENOME_LENGTH) {
        let b3 = genome_get_base_ascii(genome, pos_b + 3u);
        let b4 = genome_get_base_ascii(genome, pos_b + 4u);
        let b5 = genome_get_base_ascii(genome, pos_b + 5u);
        let second_codon_has_x = (b3 == 88u || b4 == 88u || b5 == 88u);

        if (!second_codon_has_x) {
            let codon2 = genome_get_codon_ascii(genome, pos_b + 3u);
            let modifier = codon_to_amino_index(codon2.x, codon2.y, codon2.z);
            var organ_base_type = 0u;

            if (amino_type == 17u || amino_type == 10u) {
                if (modifier == 7u) { organ_base_type = 38u; }
                else if (modifier == 8u) { organ_base_type = 39u; }
                else if (modifier == 9u) { organ_base_type = 34u; }
                else if (modifier == 16u) { organ_base_type = 40u; }
                else if (modifier == 17u) { organ_base_type = 41u; }
                else if (modifier == 19u) { organ_base_type = 35u; }
                else if (modifier < 10u) { organ_base_type = 22u; }
                else { organ_base_type = 23u; }
            }
            else if (amino_type == 9u || amino_type == 12u) {
                if (modifier < 10u) { organ_base_type = 21u; }
                else { organ_base_type = 25u; }
            }
            else if (amino_type == 8u || amino_type == 1u) {
                if (modifier < 4u) { organ_base_type = 20u; }
                else if (modifier < 7u) { organ_base_type = 33u; }
                else if (modifier < 10u) { organ_base_type = 26u; }
                else if (modifier < 14u) { organ_base_type = 32u; }
                else if (modifier < 16u) { organ_base_type = 31u; }
                else { organ_base_type = 31u; }
            }
            else if (amino_type == 6u || amino_type == 13u) {
                if (modifier < 7u) { organ_base_type = 28u; }
                else if (modifier < 9u) { organ_base_type = 36u; }
                else if (modifier == 9u) { organ_base_type = 37u; }
                else if (modifier < 14u) { organ_base_type = 29u; }
                else { organ_base_type = 30u; }
            }

            if (organ_base_type >= 20u) {
                var param_value = u32((f32(modifier) / 19.0) * 255.0);
                if (organ_base_type == 31u || organ_base_type == 32u || organ_base_type == 36u || organ_base_type == 37u) {
                    param_value = param_value & 127u;
                    let is_beta_promoter = (amino_type == 1u || amino_type == 13u);
                    if (is_beta_promoter) { param_value = param_value | 128u; }
                }

                final_part_type = encode_part_type(organ_base_type, param_value);
                bases_consumed = 6u;
            } else {
                bases_consumed = 6u; // consume both codons even if organ not formed
            }
        }
    }

    result.part_type = final_part_type;
    result.bases_consumed = bases_consumed;
    result.is_stop = false;
    result.is_valid = true;
    return result;
}

fn genome_revcomp_word(parent: array<u32, GENOME_WORDS>, wi: u32) -> u32 {
    let dst0 = wi * 4u + 0u;
    let dst1 = wi * 4u + 1u;
    let dst2 = wi * 4u + 2u;
    let dst3 = wi * 4u + 3u;
    let src0 = (GENOME_LENGTH - 1u) - dst0;
    let src1 = (GENOME_LENGTH - 1u) - dst1;
    let src2 = (GENOME_LENGTH - 1u) - dst2;
    let src3 = (GENOME_LENGTH - 1u) - dst3;
    let b0 = rna_complement(genome_get_base_ascii(parent, src0));
    let b1 = rna_complement(genome_get_base_ascii(parent, src1));
    let b2 = rna_complement(genome_get_base_ascii(parent, src2));
    let b3 = rna_complement(genome_get_base_ascii(parent, src3));
    return (b0 & 0xFFu) | ((b1 & 0xFFu) << 8u) | ((b2 & 0xFFu) << 16u) | ((b3 & 0xFFu) << 24u);
}
// ============================================================================
// RENDER.WGSL - ALL RENDERING FUNCTIONS AND KERNELS
// ============================================================================
// This file contains all rendering-related code:
// - Body part rendering functions (render_body_part, draw_selection_circle)
// - Drawing primitives (InspectorContext, draw_thick_line, draw_filled_circle, etc.)
// - Text rendering (vector font system)
// - Rendering kernels (clear_agent_grid, render_inspector, draw_inspector_agent, render_agents, composite_agents)
// ============================================================================

// ============================================================================
// PART RENDERING FUNCTION
// ============================================================================

// Inspector-specific render function (uses selected_agent_buffer instead of agents_out)
// Render a single body part with all its visual elements
fn render_body_part(
    part: BodyPart,
    part_index: u32,
    agent_id: u32,
    agent_position: vec2<f32>,
    agent_rotation: f32,
    agent_energy: f32,
    agent_color: vec3<f32>,
    body_count: u32,
    morphology_origin: vec2<f32>,
    amplification: f32,
    in_debug_mode: bool
) {
    render_body_part_ctx(part, part_index, agent_id, agent_position, agent_rotation, agent_energy, agent_color, body_count, morphology_origin, amplification, in_debug_mode, InspectorContext(vec2<f32>(-1.0), vec2<f32>(0.0), 1.0, vec2<f32>(0.0)));
}

fn render_body_part_ctx(
    part: BodyPart,
    part_index: u32,
    agent_id: u32,
    agent_position: vec2<f32>,
    agent_rotation: f32,
    agent_energy: f32,
    agent_color: vec3<f32>,
    body_count: u32,
    morphology_origin: vec2<f32>,
    amplification: f32,
    in_debug_mode: bool,
    ctx: InspectorContext
) {
    let base_type = get_base_part_type(part.part_type);
    let amino_props = get_amino_acid_properties(base_type);
    let rotated_pos = apply_agent_rotation(part.pos, agent_rotation);
    let world_pos = agent_position + rotated_pos;

    // Special agent_id value 0xFFFFFFFFu indicates we're rendering from selected_agent_buffer
    let use_selected_buffer = (agent_id == 0xFFFFFFFFu);

    // Determine segment start position
    var segment_start_world = agent_position + apply_agent_rotation(morphology_origin, agent_rotation);
    if (part_index > 0u) {
        var prev_part: BodyPart;
        if (use_selected_buffer) {
            prev_part = selected_agent_buffer[0].body[part_index - 1u];
        } else {
            prev_part = agents_out[agent_id].body[part_index - 1u];
        }
        let prev_rotated = apply_agent_rotation(prev_part.pos, agent_rotation);
        segment_start_world = agent_position + prev_rotated;
    }

    let is_first = part_index == 0u;
    let is_last = part_index == body_count - 1u;
    let is_single = body_count == 1u;

    // 1. STRUCTURAL RENDERING: Zigzag line with gradient shading
    if (!in_debug_mode && base_type < 20u) {
        // Draw zigzag structure for amino acids
        let base_color = mix(amino_props.color, agent_color, params.agent_color_blend);

        let seed = base_type * 12345u + 67890u;
        let segment_length = length(world_pos - segment_start_world);
        let organ_width = segment_length * 0.15;
        let line_width = organ_width;

        let point_count = 4u + (base_type % 3u);
        var prev_pos = segment_start_world;

        for (var i = 1u; i < point_count - 1u; i++) {
            let t = f32(i) / f32(point_count - 1u);
            let base_pos = mix(segment_start_world, world_pos, t);

            let offset_seed = seed + i * 9876u;
            let offset_angle = f32(offset_seed % 628u) / 100.0;
            let offset_dist = f32((offset_seed / 628u) % 100u) / 100.0 * organ_width * 3.5;
            let offset = vec2<f32>(cos(offset_angle) * offset_dist, sin(offset_angle) * offset_dist);
            let curr_pos = base_pos + offset;

            let dark_color = vec4<f32>(base_color * 0.5, 1.0);
            let light_color = vec4<f32>(base_color, 1.0);
            draw_thick_line_gradient_ctx(prev_pos, curr_pos, line_width, dark_color, light_color, ctx);
            prev_pos = curr_pos;
        }

        let dark_color = vec4<f32>(base_color * 0.5, 1.0);
        let light_color = vec4<f32>(base_color, 1.0);
        draw_thick_line_gradient_ctx(prev_pos, world_pos, line_width, dark_color, light_color, ctx);
    } else if (!in_debug_mode) {
        // Organs: same zigzag with gradient
        let base_color = mix(amino_props.color, agent_color, params.agent_color_blend);

        let seed = base_type * 12345u + 67890u;
        let segment_length = length(world_pos - segment_start_world);
        let organ_width = segment_length * 0.15;
        let line_width = part.size * 0.5;

        let point_count = 4u + (base_type % 3u);
        var prev_pos = segment_start_world;

        for (var i = 1u; i < point_count - 1u; i++) {
            let t = f32(i) / f32(point_count - 1u);
            let base_pos = mix(segment_start_world, world_pos, t);

            let offset_seed = seed + i * 9876u;
            let offset_angle = f32(offset_seed % 628u) / 100.0;
            let offset_dist = f32((offset_seed / 628u) % 100u) / 100.0 * organ_width * 3.5;
            let offset = vec2<f32>(cos(offset_angle) * offset_dist, sin(offset_angle) * offset_dist);
            let curr_pos = base_pos + offset;

            let dark_color = vec4<f32>(base_color * 0.5, 1.0);
            let light_color = vec4<f32>(base_color, 1.0);
            draw_thick_line_gradient_ctx(prev_pos, curr_pos, line_width, dark_color, light_color, ctx);
            prev_pos = curr_pos;
        }

        let dark_color = vec4<f32>(base_color * 0.5, 1.0);
        let light_color = vec4<f32>(base_color, 1.0);
        draw_thick_line_gradient_ctx(prev_pos, world_pos, line_width, dark_color, light_color, ctx);
    }

    // 2. DEBUG MODE RENDERING: Signal visualization
    if (in_debug_mode) {
        let a = part.alpha_signal;
        let b = part.beta_signal;
        let r = max(b, 0.0);
        let g = max(a, 0.0);
        let bl = max(max(-a, 0.0), max(-b, 0.0));
        let dbg_color = vec4<f32>(r, g, bl, 1.0);
        let thickness_dbg = max(part.size * 0.25, 0.5);
        draw_thick_line_ctx(segment_start_world, world_pos, thickness_dbg, dbg_color, ctx);
        if (!is_single && (is_first || is_last)) {
            draw_filled_circle_ctx(world_pos, thickness_dbg, dbg_color, ctx);
        }
        draw_filled_circle_ctx(world_pos, 1.5, dbg_color, ctx);
    }

    // 3. SPECIAL STRUCTURAL: Leucine (chirality flipper) - perpendicular bar
    if (base_type == 9u) {
        var segment_dir = vec2<f32>(0.0);
        if (part_index > 0u) {
            var prev: vec2<f32>;
            if (use_selected_buffer) {
                prev = selected_agent_buffer[0].body[part_index-1u].pos;
            } else {
                prev = agents_out[agent_id].body[part_index-1u].pos;
            }
            segment_dir = part.pos - prev;
        } else if (body_count > 1u) {
            var next: vec2<f32>;
            if (use_selected_buffer) {
                next = selected_agent_buffer[0].body[1u].pos;
            } else {
                next = agents_out[agent_id].body[1u].pos;
            }
            segment_dir = next - part.pos;
        } else {
            segment_dir = vec2<f32>(1.0, 0.0);
        }
        let seg_len = length(segment_dir);
        let axis_local = select(segment_dir / seg_len, vec2<f32>(1.0, 0.0), seg_len < 1e-4);
        let perp_local = vec2<f32>(-axis_local.y, axis_local.x);
        let perp_world = apply_agent_rotation(perp_local, agent_rotation);

        let half_length = part.size * 0.8;
        let p1 = world_pos - perp_world * half_length;
        let p2 = world_pos + perp_world * half_length;
        let perp_thickness = part.size * 0.3;
        let blended_color_leucine = mix(amino_props.color, agent_color, params.agent_color_blend);
        draw_thick_line_ctx(p1, p2, perp_thickness, vec4<f32>(blended_color_leucine, 1.0), ctx);
    }

    // 4. ORGAN: Condenser (charge storage/discharge)
    if (amino_props.is_condenser) {
        let signed_alpha_charge = part._pad.x;
        let signed_beta_charge = part._pad.y;
        let alpha_charge = clamp(abs(signed_alpha_charge), 0.0, 10.0);
        let beta_charge = clamp(abs(signed_beta_charge), 0.0, 10.0);
        let alpha_ratio = clamp(alpha_charge / 10.0, 0.0, 1.0);
        let beta_ratio = clamp(beta_charge / 10.0, 0.0, 1.0);
        let is_alpha_discharging = (signed_alpha_charge > 0.0);
        let is_beta_discharging = (signed_beta_charge > 0.0);

        let radius = max(part.size * 0.5, 3.0);

        var fill_color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        if (is_alpha_discharging || is_beta_discharging) {
            fill_color = vec4<f32>(1.0, 1.0, 1.0, 1.0); // White flash
        } else {
            let red_component = beta_ratio;
            let green_component = alpha_ratio;
            let charge_ratio = max(alpha_ratio, beta_ratio);
            let low_tint = vec3<f32>(red_component, green_component, 0.0) * 0.25;
            let base_tint = vec3<f32>(red_component, green_component, 0.0);
            let fill_rgb = mix(low_tint, base_tint, charge_ratio);
            fill_color = vec4<f32>(fill_rgb, 1.0);
        }

        // Fill circle
        let fill_segments = 32u;
        for (var s = 0u; s < fill_segments; s++) {
            let ang1 = f32(s) / f32(fill_segments) * 6.28318530718;
            let ang2 = f32(s + 1u) / f32(fill_segments) * 6.28318530718;
            let p1 = world_pos + vec2<f32>(cos(ang1) * radius, sin(ang1) * radius);
            let p2 = world_pos + vec2<f32>(cos(ang2) * radius, sin(ang2) * radius);
            draw_thick_line_ctx(world_pos, p1, radius * 0.5, fill_color, ctx);
            draw_thick_line_ctx(p1, p2, 1.0, fill_color, ctx);
        }

        // White outline
        let segments = 24u;
        var prev = world_pos + vec2<f32>(radius, 0.0);
        for (var s = 1u; s <= segments; s++) {
            let t = f32(s) / f32(segments);
            let ang = t * 6.28318530718;
            let p = world_pos + vec2<f32>(cos(ang) * radius, sin(ang) * radius);
            draw_thick_line_ctx(prev, p, 0.5, vec4<f32>(1.0, 1.0, 1.0, 1.0), ctx);
            prev = p;
        }
    }

    // 5. ORGAN: Enabler field visualization
    if (amino_props.is_inhibitor && params.camera_zoom > 5.0) {
        let radius = 20.0;
        let segments = 32u;
        let zoom = params.camera_zoom;
        let fade = clamp((zoom - 5.0) / 10.0, 0.0, 1.0);
        let alpha = 0.15 * fade;
        let color = vec4<f32>(0.2, 0.3, 0.2, alpha);
        var prev = world_pos + vec2<f32>(radius, 0.0);
        for (var s = 1u; s <= segments; s++) {
            let t = f32(s) / f32(segments);
            let ang = t * 6.28318530718;
            let p = world_pos + vec2<f32>(cos(ang)*radius, sin(ang)*radius);
            draw_thin_line_ctx(prev, p, color, ctx);
            prev = p;
        }
        let blended_color_enabler = mix(amino_props.color, agent_color, params.agent_color_blend);
        draw_filled_circle_ctx(world_pos, 2.0, vec4<f32>(blended_color_enabler, 0.95), ctx);
    }

    // 6. ORGAN: Propeller jet particles
    if (PROPELLERS_ENABLED && amino_props.is_propeller && agent_energy > 0.0 && params.camera_zoom > 2.0) {
        var segment_dir = vec2<f32>(0.0);
        if (part_index > 0u) {
            var prev: vec2<f32>;
            if (use_selected_buffer) {
                prev = selected_agent_buffer[0].body[part_index-1u].pos;
            } else {
                prev = agents_out[agent_id].body[part_index-1u].pos;
            }
            segment_dir = part.pos - prev;
        } else if (body_count > 1u) {
            var next: vec2<f32>;
            if (use_selected_buffer) {
                next = selected_agent_buffer[0].body[1u].pos;
            } else {
                next = agents_out[agent_id].body[1u].pos;
            }
            segment_dir = next - part.pos;
        } else {
            segment_dir = vec2<f32>(1.0, 0.0);
        }
        let seg_len = length(segment_dir);
        let axis_local = select(segment_dir / seg_len, vec2<f32>(1.0, 0.0), seg_len < 1e-4);
        let axis_world = apply_agent_rotation(axis_local, agent_rotation);
        let jet_dir = normalize(vec2<f32>(-axis_world.y, axis_world.x));
        let exhaust_dir = -jet_dir;
        let propeller_strength = part.size * 2.5 * amplification;
        let zoom_factor = clamp((params.camera_zoom - 2.0) / 8.0, 0.0, 1.0);
        let jet_length = propeller_strength * mix(0.6, 1.2, zoom_factor);
        let jet_seed = agent_id * 1000u + part_index * 17u;
        let particle_count = 1u + u32(round(amplification * 5.0)) + u32(round(zoom_factor * 3.0));
        draw_particle_jet_ctx(world_pos, exhaust_dir, jet_length, jet_seed, particle_count, ctx);
    }

    // 7. ORGAN: Mouth (feeding organ) - asterisk marker
    if (amino_props.is_mouth) {
        // Vampire mouths (F/G/H) get special big red 8-point asterisks
        let is_vampire = (base_type == 4u || base_type == 5u || base_type == 6u);
        let mouth_radius = select(max(part.size * 1.5, 4.0), max(part.size * 3.0, 8.0), is_vampire);
        let mouth_color = select(mix(amino_props.color, agent_color, params.agent_color_blend), vec3<f32>(1.0, 0.0, 0.0), is_vampire);
        draw_asterisk_8_ctx(world_pos, mouth_radius, vec4<f32>(mouth_color, 0.9), ctx);
    }

    // 8. ORGAN: Displacer (repulsion field) - diamond marker
    if (amino_props.is_displacer) {
        let blended_color_displacer = mix(amino_props.color, agent_color, params.agent_color_blend);
        let diamond_size = max(part.size * 1.2, 3.0);
        // Draw diamond as 4 lines forming a diamond shape
        let half_s = diamond_size;
        let top = world_pos + vec2<f32>(0.0, -half_s);
        let right = world_pos + vec2<f32>(half_s, 0.0);
        let bottom = world_pos + vec2<f32>(0.0, half_s);
        let left = world_pos + vec2<f32>(-half_s, 0.0);
        draw_thick_line_ctx(top, right, 1.0, vec4<f32>(blended_color_displacer, 0.9), ctx);
        draw_thick_line_ctx(right, bottom, 1.0, vec4<f32>(blended_color_displacer, 0.9), ctx);
        draw_thick_line_ctx(bottom, left, 1.0, vec4<f32>(blended_color_displacer, 0.9), ctx);
        draw_thick_line_ctx(left, top, 1.0, vec4<f32>(blended_color_displacer, 0.9), ctx);
    }

    // 9. ORGAN: Alpha/Beta Sensors - visual marker scaled by sensing radius
    if (amino_props.is_alpha_sensor || amino_props.is_beta_sensor) {
        // Extract organ parameters to calculate actual sensor radius
        let organ_param = get_organ_param(part.part_type);
        let modifier_index = u32((f32(organ_param) / 255.0) * 19.0);
        let promoter_props = get_amino_acid_properties(base_type);
        let modifier_props = get_amino_acid_properties(modifier_index);
        let combined_param = promoter_props.parameter1 + modifier_props.parameter1;

        // Calculate actual sensor radius (same formula as in sample_stochastic_gaussian)
        let base_radius = 100.0;
        let radius_variation = combined_param * 100.0;
        let sensor_radius = abs(base_radius + radius_variation);

        // Scale visual marker based on sensor radius (normalized to typical range)
        // Typical range: 0-300, so normalize and scale for visibility
        let visual_scale = clamp(sensor_radius / 200.0, 0.3, 4.0);
        let marker_size = part.size * 1.5 * visual_scale;

        // Choose color based on sensor type and signal polarity
        var sensor_color = vec3<f32>(0.0);
        if (amino_props.is_alpha_sensor) {
            // Green for positive polarity, cyan for negative polarity
            sensor_color = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 1.0), combined_param < 0.0);
        } else {
            // Red for positive polarity, magenta for negative polarity
            sensor_color = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 1.0), combined_param < 0.0);
        }
        let blended_sensor_color = mix(sensor_color, agent_color, params.agent_color_blend * 0.3);

        // Draw circle marker
        draw_filled_circle_ctx(world_pos, marker_size, vec4<f32>(blended_sensor_color, 0.8), ctx);

        // Draw outline circle to indicate sensing range at high zoom

        if (params.camera_zoom > 80.0) {
            let zoom_fade = clamp((params.camera_zoom - 8.0) / 12.0, 0.0, 1.0);
            let outline_alpha = 0.15 * zoom_fade;
            let outline_color = vec4<f32>(blended_sensor_color, outline_alpha);
            let segments = 32u;
            var prev_outline = world_pos + vec2<f32>(sensor_radius, 0.0);
            for (var s = 1u; s <= segments; s++) {
                let t = f32(s) / f32(segments);
                let ang = t * 6.28318530718;
                let p = world_pos + vec2<f32>(cos(ang) * sensor_radius, sin(ang) * sensor_radius);
                draw_thick_line_ctx(prev_outline, p, 0.3, outline_color, ctx);
                prev_outline = p;
            }
        }
    }

    // 10. ORGAN: Sine Wave Clock - large pulsating circle
    if (amino_props.is_clock) {
        // Get clock signal from _pad.x (stored during signal update pass)
        let clock_signal = part._pad.x; // Range: -1 to +1

        // Decode promoter type from part_type parameter (bit 7)
        let organ_param = get_organ_param(part.part_type);
        let is_C_promoter = ((organ_param & 128u) != 0u);

        // Dark green for K promoter (alpha), dark red for C promoter (beta)
        let clock_color = select(vec3<f32>(0.0, 0.5, 0.0), vec3<f32>(0.5, 0.0, 0.0), is_C_promoter);

        // Pulsate size based on signal output
        // Map sine output (-1 to +1) to size multiplier (0.7 to 1.3)
        let size_multiplier = 1.0 + clock_signal * 0.3;
        let pulsating_size = part.size * size_multiplier;

        // Draw large filled circle with full opacity
        draw_filled_circle_ctx(world_pos, pulsating_size, vec4<f32>(clock_color, 1.0), ctx);
    }
}

// Draw a selection circle around an agent
fn draw_selection_circle(center_pos: vec2<f32>, agent_id: u32, body_count: u32) {
    if (params.draw_enabled == 0u) { return; }
    // Calculate approximate radius based on body size from the agent's actual body
    var max_dist = 20.0; // minimum radius
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let part = agents_out[agent_id].body[i];
        let dist = length(part.pos) + part.size;
        max_dist = max(max_dist, dist);
    }

    let color = vec4<f32>(1.0, 1.0, 0.0, 1.0); // Yellow crosshair

    // Draw crosshair with fixed radius (4 long arms)
    let fixed_radius = 25.0;  // Fixed distance from center
    let arm_length = 30.0;    // Length of each arm

    // Top arm
    draw_line(center_pos + vec2<f32>(0.0, fixed_radius), center_pos + vec2<f32>(0.0, fixed_radius + arm_length), color);
    // Right arm
    draw_line(center_pos + vec2<f32>(fixed_radius, 0.0), center_pos + vec2<f32>(fixed_radius + arm_length, 0.0), color);
    // Bottom arm
    draw_line(center_pos + vec2<f32>(0.0, -fixed_radius), center_pos + vec2<f32>(0.0, -fixed_radius - arm_length), color);
    // Left arm
    draw_line(center_pos + vec2<f32>(-fixed_radius, 0.0), center_pos + vec2<f32>(-fixed_radius - arm_length, 0.0), color);
}

// ============================================================================
// HELPER FUNCTIONS FOR DRAWING
// ============================================================================

// Inspector rendering context (pass vec2(-1.0) for use_inspector_coords to disable)
struct InspectorContext {
    use_inspector_coords: vec2<f32>,  // if x >= 0, use inspector mode
    center: vec2<f32>,                // center of preview window
    scale: f32,                       // scale factor for inspector
    offset: vec2<f32>,                // offset to actual buffer position
}

// Helper function to draw a thick line in screen space
fn draw_thick_line(p0: vec2<f32>, p1: vec2<f32>, thickness: f32, color: vec4<f32>) {
    draw_thick_line_ctx(p0, p1, thickness, color, InspectorContext(vec2<f32>(-1.0), vec2<f32>(0.0), 1.0, vec2<f32>(0.0)));
}

fn draw_thick_line_ctx(p0: vec2<f32>, p1: vec2<f32>, thickness: f32, color: vec4<f32>, ctx: InspectorContext) {
    var screen_p0: vec2<i32>;
    var screen_p1: vec2<i32>;
    var screen_thickness: i32;

    if (ctx.use_inspector_coords.x >= 0.0) {
        // Inspector mode: direct coordinate mapping
        screen_p0 = vec2<i32>(i32(ctx.center.x + p0.x * ctx.scale), i32(ctx.center.y + p0.y * ctx.scale));
        screen_p1 = vec2<i32>(i32(ctx.center.x + p1.x * ctx.scale), i32(ctx.center.y + p1.y * ctx.scale));
        screen_thickness = clamp(i32(thickness * ctx.scale), 0, 50);  // Clamp to prevent overflow
    } else {
        // World mode: use world-to-screen conversion
        screen_p0 = world_to_screen(p0);
        screen_p1 = world_to_screen(p1);
        let world_to_screen_scale = params.window_width / (params.grid_size / params.camera_zoom);
        screen_thickness = clamp(i32(thickness * world_to_screen_scale), 0, 50);  // Clamp to prevent overflow
    }

    // Optimized capsule drawing: rectangle + endpoint circles
    let dx = f32(screen_p1.x - screen_p0.x);
    let dy = f32(screen_p1.y - screen_p0.y);
    let len = sqrt(dx * dx + dy * dy);

    if (len < 0.5) {
        // Degenerate case: just draw a circle
        draw_filled_circle_optimized(screen_p0, f32(screen_thickness), color, ctx);
        return;
    }

    // Normalized direction and perpendicular
    let dir_x = dx / len;
    let dir_y = dy / len;
    let perp_x = -dir_y;
    let perp_y = dir_x;

    // Calculate bounding box for the capsule
    let half_thick = f32(screen_thickness);
    let min_x = min(screen_p0.x, screen_p1.x) - screen_thickness;
    let max_x = max(screen_p0.x, screen_p1.x) + screen_thickness;
    let min_y = min(screen_p0.y, screen_p1.y) - screen_thickness;
    let max_y = max(screen_p0.y, screen_p1.y) + screen_thickness;

    // Iterate only over bounding box (much smaller than full screen)
    for (var py = min_y; py <= max_y; py++) {
        for (var px = min_x; px <= max_x; px++) {
            let pixel_x = f32(px);
            let pixel_y = f32(py);

            // Vector from p0 to pixel
            let to_pixel_x = pixel_x - f32(screen_p0.x);
            let to_pixel_y = pixel_y - f32(screen_p0.y);

            // Project onto line direction to get position along line (0 to len)
            let t = to_pixel_x * dir_x + to_pixel_y * dir_y;

            // Distance to capsule axis and gradient position
            var dist_sq: f32;
            var gradient_t: f32;

            if (t < 0.0) {
                // Before p0: distance to p0
                dist_sq = to_pixel_x * to_pixel_x + to_pixel_y * to_pixel_y;
                gradient_t = 0.0;
            } else if (t > len) {
                // After p1: distance to p1
                let to_p1_x = pixel_x - f32(screen_p1.x);
                let to_p1_y = pixel_y - f32(screen_p1.y);
                dist_sq = to_p1_x * to_p1_x + to_p1_y * to_p1_y;
                gradient_t = 1.0;
            } else {
                // Between p0 and p1: perpendicular distance to line
                let perp_dist = to_pixel_x * perp_x + to_pixel_y * perp_y;
                dist_sq = perp_dist * perp_dist;
                gradient_t = t / len;
            }

            // Check if pixel is within capsule radius
            if (dist_sq <= half_thick * half_thick) {
                // Cylindrical surface lighting with curved surface
                // Calculate the point on the cylinder axis closest to this pixel
                // Clamp t to [0, len] so normals always point perpendicular to cylinder axis
                let t_clamped = clamp(t, 0.0, len);
                let axis_point_x = f32(screen_p0.x) + t_clamped * dir_x;
                let axis_point_y = f32(screen_p0.y) + t_clamped * dir_y;

                // Calculate radial distance from axis
                let radial_offset_x = pixel_x - axis_point_x;
                let radial_offset_y = pixel_y - axis_point_y;
                let radial_dist_sq = radial_offset_x * radial_offset_x + radial_offset_y * radial_offset_y;
                let radial_factor_sq = radial_dist_sq / (half_thick * half_thick);  // normalized (0 to 1)

                // Calculate z-component for curved cylinder surface (like a sphere cross-section)
                let z_sq = 1.0 - radial_factor_sq;
                let z = sqrt(max(z_sq, 0.0));

                // Surface normal with curved profile
                let surface_normal = normalize(vec3<f32>(radial_offset_x / half_thick, radial_offset_y / half_thick, z));

                // Light direction from params
                let light_dir = normalize(vec3<f32>(params.light_dir_x, params.light_dir_y, params.light_dir_z));

                // Lambertian diffuse lighting
                let diffuse = max(dot(surface_normal, light_dir), 0.0);

                // Use material color as base, lighten lit areas (dodge-like)
                let highlight = diffuse * params.light_power;
                let lighting = 1.0 + highlight;  // 1.0 = base color, >1.0 = brightened

                let shaded_color = vec4<f32>(
                    color.rgb * lighting,
                    color.a
                );

                var screen_pos = vec2<i32>(px, py);
                var idx: u32;
                var in_bounds = false;

                if (ctx.use_inspector_coords.x >= 0.0) {
                    // Inspector mode: offset to actual buffer position and check inspector bounds
                    let buffer_pos = screen_pos + vec2<i32>(i32(ctx.offset.x), i32(ctx.offset.y));
                    // Allow drawing anywhere in the inspector area (300px wide, full height)
                    if (buffer_pos.x >= i32(ctx.offset.x) && buffer_pos.x < i32(ctx.offset.x) + i32(INSPECTOR_WIDTH) &&
                        buffer_pos.y >= 0 && buffer_pos.y < i32(params.window_height)) {
                        idx = u32(buffer_pos.y) * params.visual_stride + u32(buffer_pos.x);
                        in_bounds = true;
                    }
                } else {
                    // World mode: check screen bounds
                    // Exclude inspector area if inspector is active (selected_agent_index != u32::MAX)
                    let inspector_active = params.selected_agent_index != 0xFFFFFFFFu;
                    let max_x = select(i32(params.window_width),
                                       i32(params.window_width) - i32(INSPECTOR_WIDTH),
                                       inspector_active);
                    if (screen_pos.x >= 0 && screen_pos.x < max_x &&
                        screen_pos.y >= 0 && screen_pos.y < i32(params.window_height)) {
                        idx = screen_to_grid_index(screen_pos);
                        in_bounds = true;
                    }
                }

                if (in_bounds) {
                    agent_grid[idx] = shaded_color;
                }
            }
        }
    }
}

// Thin line version: draws 1-pixel wide lines using Bresenham-like algorithm (for technical overlays)
fn draw_thin_line_ctx(p0: vec2<f32>, p1: vec2<f32>, color: vec4<f32>, ctx: InspectorContext) {
    var screen_p0: vec2<i32>;
    var screen_p1: vec2<i32>;

    if (ctx.use_inspector_coords.x >= 0.0) {
        screen_p0 = vec2<i32>(i32(ctx.center.x + p0.x * ctx.scale), i32(ctx.center.y + p0.y * ctx.scale));
        screen_p1 = vec2<i32>(i32(ctx.center.x + p1.x * ctx.scale), i32(ctx.center.y + p1.y * ctx.scale));
    } else {
        screen_p0 = world_to_screen(p0);
        screen_p1 = world_to_screen(p1);
    }

    let dx = abs(screen_p1.x - screen_p0.x);
    let dy = abs(screen_p1.y - screen_p0.y);
    let sx = select(-1, 1, screen_p0.x < screen_p1.x);
    let sy = select(-1, 1, screen_p0.y < screen_p1.y);
    var err = dx - dy;

    var x = screen_p0.x;
    var y = screen_p0.y;

    // Bresenham's line algorithm
    for (var i = 0; i < 10000; i++) {
        var screen_pos = vec2<i32>(x, y);
        var idx: u32;
        var in_bounds = false;

        if (ctx.use_inspector_coords.x >= 0.0) {
            let buffer_pos = screen_pos + vec2<i32>(i32(ctx.offset.x), i32(ctx.offset.y));
            if (buffer_pos.x >= i32(ctx.offset.x) && buffer_pos.x < i32(ctx.offset.x) + i32(INSPECTOR_WIDTH) &&
                buffer_pos.y >= 0 && buffer_pos.y < i32(params.window_height)) {
                idx = u32(buffer_pos.y) * params.visual_stride + u32(buffer_pos.x);
                in_bounds = true;
            }
        } else {
            let inspector_active = params.selected_agent_index != 0xFFFFFFFFu;
            let max_x = select(i32(params.window_width),
                               i32(params.window_width) - i32(INSPECTOR_WIDTH),
                               inspector_active);
            if (screen_pos.x >= 0 && screen_pos.x < max_x &&
                screen_pos.y >= 0 && screen_pos.y < i32(params.window_height)) {
                idx = screen_to_grid_index(screen_pos);
                in_bounds = true;
            }
        }

        if (in_bounds) {
            agent_grid[idx] = color;
        }

        if (x == screen_p1.x && y == screen_p1.y) {
            break;
        }

        let e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x += sx;
        }
        if (e2 < dx) {
            err += dx;
            y += sy;
        }
    }
}

// Gradient version: interpolates between start_color and end_color based on position along line
fn draw_thick_line_gradient_ctx(p0: vec2<f32>, p1: vec2<f32>, thickness: f32,
                                 start_color: vec4<f32>, end_color: vec4<f32>, ctx: InspectorContext) {
    // Convert world coordinates to screen coordinates
    var screen_p0: vec2<i32>;
    var screen_p1: vec2<i32>;
    var screen_thickness: i32;

    if (ctx.use_inspector_coords.x >= 0.0) {
        // Inspector mode: direct coordinate mapping
        screen_p0 = vec2<i32>(i32(ctx.center.x + p0.x * ctx.scale), i32(ctx.center.y + p0.y * ctx.scale));
        screen_p1 = vec2<i32>(i32(ctx.center.x + p1.x * ctx.scale), i32(ctx.center.y + p1.y * ctx.scale));
        screen_thickness = clamp(i32(thickness * ctx.scale), 0, 50);
    } else {
        // World mode: use world-to-screen conversion
        screen_p0 = world_to_screen(p0);
        screen_p1 = world_to_screen(p1);
        let world_to_screen_scale = params.window_width / (params.grid_size / params.camera_zoom);
        screen_thickness = clamp(i32(thickness * world_to_screen_scale), 0, 50);
    }

    let dx = f32(screen_p1.x - screen_p0.x);
    let dy = f32(screen_p1.y - screen_p0.y);
    let len = sqrt(dx * dx + dy * dy);

    if (len < 0.5) {
        // Degenerate case: just draw a circle with average color
        let avg_color = mix(start_color, end_color, 0.5);
        draw_filled_circle_optimized(screen_p0, f32(screen_thickness), avg_color, ctx);
        return;
    }

    let dir_x = dx / len;
    let dir_y = dy / len;
    let perp_x = -dir_y;
    let perp_y = dir_x;
    let half_thick = f32(screen_thickness);

    // Calculate bounding box
    let bbox_min_x = min(screen_p0.x, screen_p1.x) - screen_thickness;
    let bbox_min_y = min(screen_p0.y, screen_p1.y) - screen_thickness;
    let bbox_max_x = max(screen_p0.x, screen_p1.x) + screen_thickness;
    let bbox_max_y = max(screen_p0.y, screen_p1.y) + screen_thickness;

    for (var py = bbox_min_y; py <= bbox_max_y; py++) {
        for (var px = bbox_min_x; px <= bbox_max_x; px++) {
            let pixel_x = f32(px);
            let pixel_y = f32(py);

            let to_pixel_x = pixel_x - f32(screen_p0.x);
            let to_pixel_y = pixel_y - f32(screen_p0.y);

            let t = to_pixel_x * dir_x + to_pixel_y * dir_y;

            var dist_sq: f32;
            var gradient_t: f32;

            if (t <= 0.0) {
                dist_sq = to_pixel_x * to_pixel_x + to_pixel_y * to_pixel_y;
                gradient_t = 0.0;
            } else if (t >= len) {
                let to_p1_x = pixel_x - f32(screen_p1.x);
                let to_p1_y = pixel_y - f32(screen_p1.y);
                dist_sq = to_p1_x * to_p1_x + to_p1_y * to_p1_y;
                gradient_t = 1.0;
            } else {
                let perp_dist = to_pixel_x * perp_x + to_pixel_y * perp_y;
                dist_sq = perp_dist * perp_dist;
                gradient_t = t / len;
            }

            if (dist_sq <= half_thick * half_thick) {
                // Interpolate color based on position along line
                let base_color = mix(start_color, end_color, gradient_t);

                // Cylindrical surface lighting with curved surface
                // Calculate the point on the cylinder axis closest to this pixel
                // Clamp t to [0, len] so normals always point perpendicular to cylinder axis
                let t_clamped = clamp(t, 0.0, len);
                let axis_point_x = f32(screen_p0.x) + t_clamped * dir_x;
                let axis_point_y = f32(screen_p0.y) + t_clamped * dir_y;

                // Calculate radial distance from axis
                let radial_offset_x = pixel_x - axis_point_x;
                let radial_offset_y = pixel_y - axis_point_y;
                let radial_dist_sq = radial_offset_x * radial_offset_x + radial_offset_y * radial_offset_y;
                let radial_factor_sq = radial_dist_sq / (half_thick * half_thick);  // normalized (0 to 1)

                // Calculate z-component for curved cylinder surface (like a sphere cross-section)
                let z_sq = 1.0 - radial_factor_sq;
                let z = sqrt(max(z_sq, 0.0));

                // Surface normal with curved profile
                let surface_normal = normalize(vec3<f32>(radial_offset_x / half_thick, radial_offset_y / half_thick, z));

                let light_dir = normalize(vec3<f32>(params.light_dir_x, params.light_dir_y, params.light_dir_z));
                let diffuse = max(dot(surface_normal, light_dir), 0.0);

                // Use material color as base, lighten lit areas (dodge-like)
                let highlight = diffuse * params.light_power;
                let lighting = 1.0 + highlight;  // 1.0 = base color, >1.0 = brightened

                let color = vec4<f32>(base_color.rgb * lighting, base_color.a);

                var screen_pos = vec2<i32>(px, py);
                var idx: u32;
                var in_bounds = false;

                if (ctx.use_inspector_coords.x >= 0.0) {
                    let buffer_pos = screen_pos + vec2<i32>(i32(ctx.offset.x), i32(ctx.offset.y));
                    if (buffer_pos.x >= i32(ctx.offset.x) && buffer_pos.x < i32(ctx.offset.x) + i32(INSPECTOR_WIDTH) &&
                        buffer_pos.y >= 0 && buffer_pos.y < i32(params.window_height)) {
                        idx = u32(buffer_pos.y) * params.visual_stride + u32(buffer_pos.x);
                        in_bounds = true;
                    }
                } else {
                    let inspector_active = params.selected_agent_index != 0xFFFFFFFFu;
                    let max_x = select(i32(params.window_width),
                                       i32(params.window_width) - i32(INSPECTOR_WIDTH),
                                       inspector_active);
                    if (screen_pos.x >= 0 && screen_pos.x < max_x &&
                        screen_pos.y >= 0 && screen_pos.y < i32(params.window_height)) {
                        idx = screen_to_grid_index(screen_pos);
                        in_bounds = true;
                    }
                }

                if (in_bounds) {
                    agent_grid[idx] = color;
                }
            }
        }
    }
}

// Helper for drawing optimized filled circles (used by optimized thick line)
fn draw_filled_circle_optimized(center: vec2<i32>, radius: f32, color: vec4<f32>, ctx: InspectorContext) {
    let radius_i = i32(ceil(radius));
    let radius_sq = radius * radius;

    for (var dy = -radius_i; dy <= radius_i; dy++) {
        for (var dx = -radius_i; dx <= radius_i; dx++) {
            let dist_sq = f32(dx * dx + dy * dy);
            if (dist_sq <= radius_sq) {
                // Radial gradient shading: lighter at center, darker at edges
                let dist_factor = sqrt(dist_sq) / radius;  // 0.0 at center, 1.0 at edge
                let shaded_color = vec4<f32>(
                    mix(color.rgb, color.rgb * 0.5, dist_factor),  // Interpolate brightness
                    color.a
                );

                var screen_pos = center + vec2<i32>(dx, dy);
                var idx: u32;
                var in_bounds = false;

                if (ctx.use_inspector_coords.x >= 0.0) {
                    // Inspector mode
                    let buffer_pos = screen_pos + vec2<i32>(i32(ctx.offset.x), i32(ctx.offset.y));
                    if (buffer_pos.x >= i32(ctx.offset.x) && buffer_pos.x < i32(ctx.offset.x + 280.0) &&
                        buffer_pos.y >= i32(ctx.offset.y) && buffer_pos.y < i32(ctx.offset.y + 280.0)) {
                        idx = u32(buffer_pos.y) * params.visual_stride + u32(buffer_pos.x);
                        in_bounds = true;
                    }
                } else {
                    // World mode
                    let max_x = select(i32(params.window_width), i32(params.window_width) - i32(INSPECTOR_WIDTH), params.selected_agent_index != 0xFFFFFFFFu);
                    // World mode
                    if (screen_pos.x >= 0 && screen_pos.x < max_x &&
                    // World mode
                        screen_pos.y >= 0 && screen_pos.y < i32(params.window_height)) {
                        idx = screen_to_grid_index(screen_pos);
                        in_bounds = true;
                    }
                }

                if (in_bounds) {
                    agent_grid[idx] = shaded_color;
                }
            }
        }
    }
}

// Helper function to draw a clean circle outline in screen space
fn draw_circle(center: vec2<f32>, radius: f32, color: vec4<f32>) {
    // Convert world position to screen coordinates
    let screen_center = world_to_screen(center);

    // Calculate screen-space radius (accounting for zoom)
    let world_to_screen_scale = params.window_width / (params.grid_size / params.camera_zoom);
    let screen_radius = radius * world_to_screen_scale;

    let radius_i = i32(ceil(screen_radius));
    let line_thickness = 1.0; // pixels

    for (var dy = -radius_i; dy <= radius_i; dy++) {
        for (var dx = -radius_i; dx <= radius_i; dx++) {
            let offset = vec2<f32>(f32(dx), f32(dy));
            let dist = length(offset);

            if (abs(dist - screen_radius) < line_thickness) {
                let screen_pos = screen_center + vec2<i32>(dx, dy);

                // Check if in visible window bounds
                // Exclude inspector area if inspector is active
                let max_x = select(i32(params.window_width), i32(params.window_width) - i32(INSPECTOR_WIDTH), params.selected_agent_index != 0xFFFFFFFFu);
                if (screen_pos.x >= 0 && screen_pos.x < max_x &&
                    screen_pos.y >= 0 && screen_pos.y < i32(params.window_height)) {

                    let idx = screen_to_grid_index(screen_pos);
                    agent_grid[idx] = color;
                }
            }
        }
    }
}

// Helper: draw a filled circle in screen space
fn draw_filled_circle(center: vec2<f32>, radius: f32, color: vec4<f32>) {
    draw_filled_circle_ctx(center, radius, color, InspectorContext(vec2<f32>(-1.0), vec2<f32>(0.0), 1.0, vec2<f32>(0.0)));
}

fn draw_filled_circle_ctx(center: vec2<f32>, radius: f32, color: vec4<f32>, ctx: InspectorContext) {
    var screen_center: vec2<i32>;
    var screen_radius: f32;

    if (ctx.use_inspector_coords.x >= 0.0) {
        // Inspector mode
        screen_center = vec2<i32>(i32(ctx.center.x + center.x * ctx.scale), i32(ctx.center.y + center.y * ctx.scale));
        screen_radius = clamp(radius * ctx.scale, 0.0, 50.0);  // Clamp to prevent overflow
    } else {
        // World mode
        screen_center = world_to_screen(center);
        let world_to_screen_scale = params.window_width / (params.grid_size / params.camera_zoom);
        screen_radius = clamp(radius * world_to_screen_scale, 0.0, 50.0);  // Clamp to prevent overflow
    }

    let radius_i = i32(ceil(screen_radius));

    for (var dy = -radius_i; dy <= radius_i; dy++) {
        for (var dx = -radius_i; dx <= radius_i; dx++) {
            let offset = vec2<f32>(f32(dx), f32(dy));
            let dist2 = dot(offset, offset);
            if (dist2 <= screen_radius * screen_radius) {
                // Spherical surface lighting (optimized: avoid extra sqrt)
                let radius_sq = screen_radius * screen_radius;
                let dist_factor_sq = dist2 / radius_sq;  // (dist/radius)Ã‚Â²

                // Calculate 3D surface normal for a sphere
                // In 2D view, we see a circle; assume sphere extends in z-direction
                let z_sq = 1.0 - dist_factor_sq;  // xÃ‚Â² + yÃ‚Â² + zÃ‚Â² = 1
                let z = sqrt(max(z_sq, 0.0));
                let surface_normal = normalize(vec3<f32>(offset.x / screen_radius, offset.y / screen_radius, z));

                // Light direction from params
                let light_dir = normalize(vec3<f32>(params.light_dir_x, params.light_dir_y, params.light_dir_z));

                // Lambertian diffuse lighting
                let diffuse = max(dot(surface_normal, light_dir), 0.0);

                // Use material color as base, lighten lit areas (dodge-like)
                let highlight = diffuse * params.light_power;
                let lighting = 1.0 + highlight;  // 1.0 = base color, >1.0 = brightened

                let shaded_color = vec4<f32>(
                    color.rgb * lighting,
                    color.a
                );

                var screen_pos = screen_center + vec2<i32>(dx, dy);
                var idx: u32;
                var in_bounds = false;

                if (ctx.use_inspector_coords.x >= 0.0) {
                    // Inspector mode
                    let buffer_pos = screen_pos + vec2<i32>(i32(ctx.offset.x), i32(ctx.offset.y));
                    if (buffer_pos.x >= i32(ctx.offset.x) && buffer_pos.x < i32(ctx.offset.x + 280.0) &&
                        buffer_pos.y >= i32(ctx.offset.y) && buffer_pos.y < i32(ctx.offset.y + 280.0)) {
                        idx = u32(buffer_pos.y) * params.visual_stride + u32(buffer_pos.x);
                        in_bounds = true;
                    }
                } else {
                    // World mode
                    let max_x = select(i32(params.window_width), i32(params.window_width) - i32(INSPECTOR_WIDTH), params.selected_agent_index != 0xFFFFFFFFu);
                    if (screen_pos.x >= 0 && screen_pos.x < max_x &&
                        screen_pos.y >= 0 && screen_pos.y < i32(params.window_height)) {
                        idx = screen_to_grid_index(screen_pos);
                        in_bounds = true;
                    }
                }

                if (in_bounds) {
                    // Alpha blending: blend new color with existing background
                    let bg_color = agent_grid[idx];
                    let src_alpha = shaded_color.a;
                    let inv_alpha = 1.0 - src_alpha;
                    let blended = vec4<f32>(
                        shaded_color.rgb * src_alpha + bg_color.rgb * inv_alpha,
                        max(shaded_color.a, bg_color.a)
                    );
                    agent_grid[idx] = blended;
                }
            }
        }
    }
}

// Helper: draw a 5-pointed star in screen space
fn draw_star(center: vec2<f32>, radius: f32, color: vec4<f32>) {
    // Convert world position to screen coordinates
    let screen_center = world_to_screen(center);

    // Calculate screen-space radius (accounting for zoom)
    let world_to_screen_scale = params.window_width / (params.grid_size / params.camera_zoom);
    let screen_radius = radius * world_to_screen_scale;

    // 5-pointed star with 10 points (5 outer, 5 inner)
    let num_points = 5u;
    let inner_radius = screen_radius * 0.38; // Inner points at ~38% of outer radius

    // Draw star as lines connecting the points
    for (var i = 0u; i < num_points; i++) {
        // Calculate angles (starting from top, going clockwise)
        let angle_outer = -1.57079632679 + f32(i) * 6.28318530718 / f32(num_points);
        let angle_inner = angle_outer + 3.14159265359 / f32(num_points);
        let angle_next_outer = -1.57079632679 + f32((i + 1u) % num_points) * 6.28318530718 / f32(num_points);

        // Calculate positions
        let outer_x = screen_center.x + i32(cos(angle_outer) * screen_radius);
        let outer_y = screen_center.y + i32(sin(angle_outer) * screen_radius);
        let inner_x = screen_center.x + i32(cos(angle_inner) * inner_radius);
        let inner_y = screen_center.y + i32(sin(angle_inner) * inner_radius);
        let next_outer_x = screen_center.x + i32(cos(angle_next_outer) * screen_radius);
        let next_outer_y = screen_center.y + i32(sin(angle_next_outer) * screen_radius);

        // Draw lines: outer -> inner -> next_outer
        draw_line_pixels(vec2<i32>(outer_x, outer_y), vec2<i32>(inner_x, inner_y), color);
        draw_line_pixels(vec2<i32>(inner_x, inner_y), vec2<i32>(next_outer_x, next_outer_y), color);
    }
}

// Helper: draw an 8-point asterisk for vampire mouths
fn draw_asterisk_8_ctx(center: vec2<f32>, radius: f32, color: vec4<f32>, ctx: InspectorContext) {
    let angle_step = 0.39269908; // PI / 8 = 22.5 degrees

    for (var i = 0u; i < 4u; i++) {
        let angle = f32(i) * angle_step * 2.0;
        let dx = cos(angle) * radius;
        let dy = sin(angle) * radius;
        let start = center - vec2<f32>(dx, dy);
        let end = center + vec2<f32>(dx, dy);
        draw_thick_line_ctx(start, end, 1.5, color, ctx);
    }
}

// Helper: draw an asterisk (*) with 4 crossing lines (vertical, horizontal, 2 diagonals)
fn draw_asterisk(center: vec2<f32>, radius: f32, color: vec4<f32>) {
    draw_asterisk_ctx(center, radius, color, InspectorContext(vec2<f32>(-1.0), vec2<f32>(0.0), 1.0, vec2<f32>(0.0)));
}

fn draw_asterisk_ctx(center: vec2<f32>, radius: f32, color: vec4<f32>, ctx: InspectorContext) {
    // Draw 4 lines: vertical, horizontal, and two diagonals
    let diag_offset = radius * 0.70710678; // radius / sqrt(2)

    // Vertical line
    let up = center + vec2<f32>(0.0, -radius);
    let down = center + vec2<f32>(0.0, radius);
    draw_thick_line_ctx(up, down, 1.0, color, ctx);

    // Horizontal line
    let left = center + vec2<f32>(-radius, 0.0);
    let right = center + vec2<f32>(radius, 0.0);
    draw_thick_line_ctx(left, right, 1.0, color, ctx);

    // Diagonal 1 (top-left to bottom-right)
    let tl = center + vec2<f32>(-diag_offset, -diag_offset);
    let br = center + vec2<f32>(diag_offset, diag_offset);
    draw_thick_line_ctx(tl, br, 1.0, color, ctx);

    // Diagonal 2 (top-right to bottom-left)
    let tr = center + vec2<f32>(diag_offset, -diag_offset);
    let bl = center + vec2<f32>(-diag_offset, diag_offset);
    draw_thick_line_ctx(tr, bl, 1.0, color, ctx);
}

// Helper: draw a cloud-like shape (fuzzy circle with some random bumps)
fn draw_cloud(center: vec2<f32>, radius: f32, color: vec4<f32>, seed: u32) {
    draw_cloud_ctx(center, radius, color, seed, InspectorContext(vec2<f32>(-1.0), vec2<f32>(0.0), 1.0, vec2<f32>(0.0)));
}

fn draw_cloud_ctx(center: vec2<f32>, radius: f32, color: vec4<f32>, seed: u32, ctx: InspectorContext) {
    // Optimized: single-pass cloud rendering instead of 9 separate circle draws
    var screen_center: vec2<i32>;
    var screen_radius: f32;

    if (ctx.use_inspector_coords.x >= 0.0) {
        screen_center = vec2<i32>(i32(ctx.center.x + center.x * ctx.scale), i32(ctx.center.y + center.y * ctx.scale));
        screen_radius = clamp(radius * ctx.scale, 0.0, 50.0);
    } else {
        screen_center = world_to_screen(center);
        let world_to_screen_scale = params.window_width / (params.grid_size / params.camera_zoom);
        screen_radius = clamp(radius * world_to_screen_scale, 0.0, 50.0);
    }

    // Pre-calculate all puff centers and radii
    let num_puffs = 8u;
    var puff_centers: array<vec2<f32>, 9>;
    var puff_radii: array<f32, 9>;

    // Central puff (larger)
    puff_centers[0] = vec2<f32>(f32(screen_center.x), f32(screen_center.y));
    puff_radii[0] = screen_radius * 0.7;

    // Surrounding puffs
    for (var i = 0u; i < num_puffs; i++) {
        let angle = f32(i) * 6.28318530718 / f32(num_puffs);
        let hash_val = hash_f32(seed * (i + 1u) * 2654435761u);
        let offset_dist = screen_radius * 0.4 * hash_val;
        puff_centers[i + 1u] = vec2<f32>(
            f32(screen_center.x) + cos(angle) * offset_dist,
            f32(screen_center.y) + sin(angle) * offset_dist
        );
        puff_radii[i + 1u] = screen_radius * (0.5 + 0.3 * hash_val);
    }

    // Find bounding box for all puffs
    var min_x = screen_center.x;
    var max_x = screen_center.x;
    var min_y = screen_center.y;
    var max_y = screen_center.y;

    for (var i = 0u; i < 9u; i++) {
        let r = i32(ceil(puff_radii[i]));
        min_x = min(min_x, i32(puff_centers[i].x) - r);
        max_x = max(max_x, i32(puff_centers[i].x) + r);
        min_y = min(min_y, i32(puff_centers[i].y) - r);
        max_y = max(max_y, i32(puff_centers[i].y) + r);
    }

    // Single pass over bounding box, check distance to all puffs
    for (var py = min_y; py <= max_y; py++) {
        for (var px = min_x; px <= max_x; px++) {
            let pixel_pos = vec2<f32>(f32(px), f32(py));
            var inside_any_puff = false;

            // Check if pixel is inside any of the 9 puffs
            for (var i = 0u; i < 9u; i++) {
                let dx = pixel_pos.x - puff_centers[i].x;
                let dy = pixel_pos.y - puff_centers[i].y;
                let dist_sq = dx * dx + dy * dy;
                if (dist_sq <= puff_radii[i] * puff_radii[i]) {
                    inside_any_puff = true;
                    break;
                }
            }

            if (inside_any_puff) {
                var screen_pos = vec2<i32>(px, py);
                var idx: u32;
                var in_bounds = false;

                if (ctx.use_inspector_coords.x >= 0.0) {
                    let buffer_pos = screen_pos + vec2<i32>(i32(ctx.offset.x), i32(ctx.offset.y));
                    if (buffer_pos.x >= i32(ctx.offset.x) && buffer_pos.x < i32(ctx.offset.x + 280.0) &&
                        buffer_pos.y >= i32(ctx.offset.y) && buffer_pos.y < i32(ctx.offset.y + 280.0)) {
                        idx = u32(buffer_pos.y) * params.visual_stride + u32(buffer_pos.x);
                        in_bounds = true;
                    }
                } else {
                    let max_x = select(i32(params.window_width), i32(params.window_width) - i32(INSPECTOR_WIDTH), params.selected_agent_index != 0xFFFFFFFFu);
                    if (screen_pos.x >= 0 && screen_pos.x < max_x &&
                        screen_pos.y >= 0 && screen_pos.y < i32(params.window_height)) {
                        idx = screen_to_grid_index(screen_pos);
                        in_bounds = true;
                    }
                }

                if (in_bounds) {
                    agent_grid[idx] = color;
                }
            }
        }
    }
}

// Helper: draw a particle jet (motion-blurred particles in a cone)
fn draw_particle_jet(origin: vec2<f32>, direction: vec2<f32>, length: f32, seed: u32, particle_count: u32) {
    draw_particle_jet_ctx(origin, direction, length, seed, particle_count, InspectorContext(vec2<f32>(-1.0), vec2<f32>(0.0), 1.0, vec2<f32>(0.0)));
}

fn draw_particle_jet_ctx(origin: vec2<f32>, direction: vec2<f32>, length: f32, seed: u32, particle_count: u32, ctx: InspectorContext) {
    // Draw motion-blurred particles spread in a cone
    let num_particles = clamp(particle_count, 2u, 10u);
    if (num_particles < 2u) { return; }
    let particle_color = vec4<f32>(0.2, 0.5, 1.0, 0.8); // Semi-transparent blue
    let cone_angle = 0.4; // Cone spread angle in radians (~23 degrees)

    for (var i = 0u; i < num_particles; i++) {
        // Generate two hash values for this particle
        let hash_val1 = hash_f32(seed * (i + 1u) * 2654435761u);
        let hash_val2 = hash_f32(seed * (i + 1u) * 1103515245u);

        // Distance along the jet (0 to 1)
        let denom = max(num_particles - 1u, 1u);
        let t = f32(i) / f32(denom);
        let distance = length * t * (0.7 + 0.6 * hash_val1);

        // Angular spread in cone (using hash to distribute evenly in cone)
        let angle_offset = (hash_val2 - 0.5) * cone_angle * (1.0 + t * 0.5); // Wider spread further out

        // Rotate direction by angle_offset
        let cos_angle = cos(angle_offset);
        let sin_angle = sin(angle_offset);
        let rotated_dir = vec2<f32>(
            direction.x * cos_angle - direction.y * sin_angle,
            direction.x * sin_angle + direction.y * cos_angle
        );

        // Calculate particle position
        let particle_pos = origin + rotated_dir * distance;

        // Motion blur: draw a short streak instead of a dot
        let streak_length = 1.6 * (1.0 - t * 0.35); // Longer streaks at base
        let streak_end = particle_pos + rotated_dir * streak_length;
        let streak_thickness = 0.45 * (1.0 - t * 0.6); // Thinner as they move away

        // Draw motion-blurred particle as a thick line
    let fade_color = vec4<f32>(particle_color.xyz, particle_color.w * (0.6 * (1.0 - t * 0.5)));
        draw_thick_line_ctx(particle_pos, streak_end, streak_thickness, fade_color, ctx);
    }
}
// ============================================================================
// COMPACT VECTOR FONT DATA (packed u32 format)
// Each u32 packs 4 coordinates as bytes: (p0.x, p0.y, p1.x, p1.y)
// Coordinates are scaled 0.0-1.0 Ã¢â€ â€™ 0-255 (decode with /255.0)
// Negative values (e.g., comma tail) are clamped to 0
// ============================================================================

var<private> FONT_SEGMENTS: array<u32, 160> = array<u32, 160>(
    0x00CC0033u,0xFFCC00CCu,0xFF33FFCCu,0x0033FF33u,0xFFCC0033u,0xFF800080u,0xCC4CFF80u,0xFFCCFF33u,
    0x80CCFFCCu,0x803380CCu,0x00338033u,0x00CC0033u,0xFFCCFF33u,0x00CCFFCCu,0x003300CCu,0x806680CCu,
    0x6633FF33u,0x66CC6633u,0xFFB200B2u,0xFF33FFCCu,0x8033FF33u,0x80CC8033u,0x00CC80CCu,0x003300CCu,
    0xFF33FFCCu,0x0033FF33u,0x00CC0033u,0x80CC00CCu,0x803380CCu,0xFFCCFF33u,0x0066FFCCu,0x00CC0033u,
    0xFFCC00CCu,0xFF33FFCCu,0x0033FF33u,0x80CC8033u,0x803380CCu,0xFF338033u,0xFFCCFF33u,0x00CCFFCCu,
    0x003300CCu,0xFF80001Au,0x00E6FF80u,0x66BF6640u,0xFF330033u,0xFFB2FF33u,0xBFCCFFB2u,0x80B2BFCCu,
    0x803380B2u,0x40CC80B2u,0x00B240CCu,0x003300B2u,0xFF33FFCCu,0x0033FF33u,0x00CC0033u,0xFF330033u,
    0xFF99FF33u,0xCCCCFF99u,0x33CCCCCCu,0x009933CCu,0x00330099u,0xFF33FFCCu,0x0033FF33u,0x80B28033u,
    0x00CC0033u,0xFF330033u,0xFFCCFF33u,0x80B28033u,0xFF33FFCCu,0x0033FF33u,0x00CC0033u,0x80CC00CCu,
    0x808080CCu,0xFF330033u,0xFFCC00CCu,0x80CC8033u,0xFF800080u,0x00B2004Cu,0xFFB2FF4Cu,0x3399FF99u,
    0x00663399u,0x33330066u,0xFF330033u,0x8033FFCCu,0x00CC8033u,0x0033FF33u,0x00CC0033u,0xFF1A001Au,
    0x8080FF1Au,0xFFE68080u,0x00E6FFE6u,0xFF330033u,0x00CCFF33u,0xFFCC00CCu,0x00CC0033u,0xFFCC00CCu,
    0xFF33FFCCu,0x0033FF33u,0xFF330033u,0xFFB2FF33u,0xBFCCFFB2u,0x80B2BFCCu,0x803380B2u,0x00CC0033u,
    0xFFCC00CCu,0xFF33FFCCu,0x0033FF33u,0x00E64C99u,0xFF330033u,0xFFB2FF33u,0xBFCCFFB2u,0x80B2BFCCu,
    0x803380B2u,0x00CC8080u,0xFF33FFCCu,0x8033FF33u,0x80CC8033u,0x00CC80CCu,0x003300CCu,0xFFCCFF33u,
    0x0080FF80u,0x3333FF33u,0x004C3333u,0x00B2004Cu,0x33CC00B2u,0xFFCC33CCu,0x0080FF1Au,0xFFE60080u,
    0x0033FF1Au,0x99800033u,0x00CC9980u,0xFFE600CCu,0xFFCC0033u,0x00CCFF33u,0x8080FF33u,0x8080FFCCu,
    0x00808080u,0xFFCCFF33u,0x0033FFCCu,0x00CC0033u,0x008C0073u,0x0D8C0D73u,0x00660080u,0x4C8C4C73u,
    0xB28CB273u,0x80CC8033u,0xCC803380u,0x80CC8033u,0x66CC6633u,0x99CC9933u,0xE64CCC33u,0x33CC1AB2u,
    0xFFCC0033u,0xB266FF99u,0x4C66B266u,0x00994C66u,0xB299FF66u,0x4C99B299u,0x00664C99u,0xFFCC0033u);

// Decode a packed segment into vec2 coordinates
fn unpack_segment(packed: u32) -> vec4<f32> {
    let p0x = f32((packed) & 0xFFu) / 255.0;
    let p0y = f32((packed >> 8u) & 0xFFu) / 255.0;
    let p1x = f32((packed >> 16u) & 0xFFu) / 255.0;
    let p1y = f32((packed >> 24u) & 0xFFu) / 255.0;
    return vec4<f32>(p0x, p0y, p1x, p1y);
}

// Compact font character data (offset + count per character)
struct FontChar {
    offset: u32,
    count: u32,
}

var<private> FONT_CHARS: array<FontChar, 47> = array<FontChar, 47>(
    // 0Ã¢â‚¬â€œ9: digits
    FontChar(0u,   5u),   // '0'
    FontChar(5u,   2u),   // '1'
    FontChar(7u,   5u),   // '2'
    FontChar(12u,  4u),   // '3'
    FontChar(16u,  3u),   // '4'
    FontChar(19u,  5u),   // '5'
    FontChar(24u,  5u),   // '6'
    FontChar(29u,  2u),   // '7'
    FontChar(31u,  5u),   // '8'
    FontChar(36u,  5u),   // '9'
    // 10Ã¢â‚¬â€œ35: AÃ¢â‚¬â€œZ
    FontChar(41u,  3u),   // 'A'
    FontChar(44u,  8u),   // 'B'
    FontChar(52u,  3u),   // 'C'
    FontChar(55u,  6u),   // 'D'
    FontChar(61u,  4u),   // 'E'
    FontChar(65u,  3u),   // 'F'
    FontChar(68u,  5u),   // 'G'
    FontChar(73u,  3u),   // 'H'
    FontChar(76u,  3u),   // 'I'
    FontChar(79u,  3u),   // 'J'
    FontChar(82u,  3u),   // 'K'
    FontChar(85u,  2u),   // 'L'
    FontChar(87u,  4u),   // 'M'
    FontChar(91u,  3u),   // 'N'
    FontChar(94u,  4u),   // 'O'
    FontChar(98u,  5u),   // 'P'
    FontChar(103u, 5u),   // 'Q'
    FontChar(108u, 6u),   // 'R'
    FontChar(114u, 5u),   // 'S'
    FontChar(119u, 2u),   // 'T'
    FontChar(121u, 5u),   // 'U'
    FontChar(126u, 2u),   // 'V'
    FontChar(128u, 4u),   // 'W'
    FontChar(132u, 2u),   // 'X'
    FontChar(134u, 3u),   // 'Y'
    FontChar(137u, 3u),   // 'Z'
    // 36Ã¢â‚¬â€œ46: symbols
    FontChar(140u, 0u),   // ' ' (space)
    FontChar(140u, 2u),   // '.'
    FontChar(142u, 1u),   // ','
    FontChar(143u, 2u),   // ':'
    FontChar(145u, 1u),   // '-'
    FontChar(146u, 2u),   // '+'
    FontChar(148u, 2u),   // '='
    FontChar(150u, 3u),   // '%'
    FontChar(153u, 3u),   // '('
    FontChar(156u, 3u),   // ')'
    FontChar(159u, 1u)    // '/'
);
fn char_index(c: u32) -> i32 {
    if (c >= 48u && c <= 57u) { return i32(c - 48u); }
    if (c >= 65u && c <= 90u) { return i32(c - 65u + 10u); }
    if (c == 32u) { return 36; }
    if (c == 46u) { return 37; }
    if (c == 44u) { return 38; }
    if (c == 58u) { return 39; }
    if (c == 45u) { return 40; }
    if (c == 43u) { return 41; }
    if (c == 61u) { return 42; }
    if (c == 37u) { return 43; }
    if (c == 40u) { return 44; }
    if (c == 41u) { return 45; }
    if (c == 47u) { return 46; }
    return -1;
}

// Get character width (relative to height=1.0)
fn get_char_width(c: u32) -> f32 {
    if (c == 32u) { return 0.5; } // space
    if (c == 46u || c == 44u || c == 58u) { return 0.3; } // punctuation
    if (c == 73u || c == 49u) { return 0.5; } // 'I' and '1'
    if (c == 77u || c == 87u) { return 1.2; } // 'M' and 'W'
    return 1.0; // default width
}

// Draw a single character at position with specified height
fn draw_char_vector(pos: vec2<f32>, c: u32, height: f32, color: vec4<f32>, ctx: InspectorContext) -> f32 {
    let idx = char_index(c);
    if (idx < 0) {
        return height * 0.4; // fallback spacing for unsupported chars
    }

    let ch = FONT_CHARS[u32(idx)];
    let base = ch.offset;
    let seg_count = ch.count;
    let char_width = get_char_width(c) * height;

    // Use ~1px lines (user request)
    let line_thickness = max(1.0, height * 0.1);

    for (var i = 0u; i < seg_count; i++) {
        let seg = unpack_segment(FONT_SEGMENTS[base + i]);
        // Flip Y so font baseline is at bottom and screen Y grows downward
        let p0 = pos + vec2<f32>(seg.x * char_width, (1.0 - seg.y) * height);
        let p1 = pos + vec2<f32>(seg.z * char_width, (1.0 - seg.w) * height);
        draw_thick_line_ctx(p0, p1, line_thickness, color, ctx);
    }

    return char_width + height * 0.2; // width plus a bit of spacing
}

// Distance from point to segment (in character-local space)
fn point_segment_distance(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let ab = b - a;
    let ab_len_sq = max(dot(ab, ab), 1e-6);
    let t = clamp(dot(p - a, ab) / ab_len_sq, 0.0, 1.0);
    let proj = a + ab * t;
    return length(p - proj);
}

// Vector font mask evaluated per pixel (avoids race/flicker from draw_thick_line writes)
fn char_vector_mask(local_px: vec2<u32>, c: u32, height: f32) -> bool {
    let idx = char_index(c);
    if (idx < 0) { return false; }

    let ch = FONT_CHARS[u32(idx)];
    let base = ch.offset;
    let seg_count = ch.count;
    let char_width = get_char_width(c) * height;
    let line_thickness = max(1.0, height * 0.1);
    let half_thick = line_thickness * 0.5;
    let p = vec2<f32>(f32(local_px.x) + 0.5, f32(local_px.y) + 0.5);

    for (var i = 0u; i < seg_count; i++) {
        let seg = unpack_segment(FONT_SEGMENTS[base + i]);
        let p0 = vec2<f32>(seg.x * char_width, (1.0 - seg.y) * height);
        let p1 = vec2<f32>(seg.z * char_width, (1.0 - seg.w) * height);
        let d = point_segment_distance(p, p0, p1);
        if (d <= half_thick) {
            return true;
        }
    }
    return false;
}

// Draw a string at position with specified height
fn draw_string_vector(pos: vec2<f32>, text: ptr<function, array<u32, 32>>, length: u32, height: f32, color: vec4<f32>, ctx: InspectorContext) -> f32 {
    var cursor_x = pos.x;

    for (var i = 0u; i < length && i < 32u; i++) {
        let char_code = (*text)[i];
        let width = draw_char_vector(vec2<f32>(cursor_x, pos.y), char_code, height, color, ctx);
        cursor_x += width;
    }

    return cursor_x - pos.x;
}

// Helper: Convert u32 number to string (max 10 digits)
fn u32_to_string(value: u32, out_str: ptr<function, array<u32, 32>>, start: u32) -> u32 {
    if (value == 0u) {
        (*out_str)[start] = 48u; // '0'
        return 1u;
    }

    var temp = value;
    var digit_count = 0u;
    var digits: array<u32, 10>;

    // Extract digits in reverse order
    while (temp > 0u && digit_count < 10u) {
        digits[digit_count] = (temp % 10u) + 48u; // Convert to ASCII
        temp = temp / 10u;
        digit_count++;
    }

    // Reverse into output string starting at `start`
    for (var i = 0u; i < digit_count; i++) {
        (*out_str)[start + i] = digits[digit_count - 1u - i];
    }

    return digit_count;
}

// Helper: Convert f32 to string (with 2 decimal places, max 16 chars)
fn f32_to_string(value: f32, out_str: ptr<function, array<u32, 32>>, start: u32) -> u32 {
    var pos = start;
    var val = value;

    // Handle negative
    if (val < 0.0) {
        (*out_str)[pos] = 45u; // '-'
        pos++;
        val = -val;
    }

    // Integer part
    let int_part = u32(floor(val));
    pos += u32_to_string(int_part, out_str, pos);

    // Decimal point
    (*out_str)[pos] = 46u; // '.'
    pos++;

    // Fractional part (2 decimal places)
    let frac = val - floor(val);
    let frac_scaled = u32(round(frac * 100.0));
    let tens = frac_scaled / 10u;
    let ones = frac_scaled % 10u;
    (*out_str)[pos] = tens + 48u;
    (*out_str)[pos + 1u] = ones + 48u;
    pos += 2u;

    return pos - start;
}

// ============================================================================
// AGENT RENDER BUFFER MANAGEMENT
// ============================================================================

@compute @workgroup_size(16, 16)
fn clear_agent_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    // Skip clearing the inspector area only when an agent is selected (to preserve inspector UI)
    if (params.selected_agent_index != 0xFFFFFFFFu && x >= width - INSPECTOR_WIDTH) {
        return;
    }

    let agent_idx = y * params.visual_stride + x;
    // Fade the agent trail based on decay rate (1.0 = instant clear, 0.0 = no clear)
    let current_color = agent_grid[agent_idx];
    agent_grid[agent_idx] = current_color * (1.0 - params.agent_trail_decay);
}

// Inspector bar layout configuration
struct BarLayout {
    bars_y_start: u32,
    bars_y_end: u32,
    genome_x_start: u32,
    genome_x_end: u32,
    amino_x_start: u32,
    amino_x_end: u32,
    label_x_start: u32,
    label_x_end: u32,
    alpha_x_start: u32,
    alpha_x_end: u32,
    beta_x_start: u32,
    beta_x_end: u32,
}

// Calculate inspector bar positions based on anchor position
// anchor_x, anchor_y: top-left corner of the bar area
// available_height: height available for bars
fn calculate_bar_layout(anchor_x: u32, anchor_y: u32, available_height: u32) -> BarLayout {
    let bar_width = 40u;       // Width of genome and amino bars (narrower to fit in 300px)
    let label_width = 80u;     // Width of legend area (fits 3-letter amino codes)
    let signal_width = 25u;    // Width of signal bars
    let gap_large = 3u;        // Gap between major sections
    let gap_small = 1u;        // Gap between related elements (keep signals tight to amino)

    var bar_layout: BarLayout;

    // Vertical extent
    bar_layout.bars_y_start = anchor_y;
    bar_layout.bars_y_end = anchor_y + available_height;

    // Horizontal layout (left to right): genome, amino, alpha, beta, labels
    bar_layout.genome_x_start = anchor_x;
    bar_layout.genome_x_end = bar_layout.genome_x_start + bar_width;

    bar_layout.amino_x_start = bar_layout.genome_x_end + gap_large;
    bar_layout.amino_x_end = bar_layout.amino_x_start + bar_width;

    bar_layout.alpha_x_start = bar_layout.amino_x_end + gap_small;
    bar_layout.alpha_x_end = bar_layout.alpha_x_start + signal_width;

    bar_layout.beta_x_start = bar_layout.alpha_x_end + gap_small;
    bar_layout.beta_x_end = bar_layout.beta_x_start + signal_width;

    bar_layout.label_x_start = bar_layout.beta_x_end + gap_large;
    bar_layout.label_x_end = bar_layout.label_x_start + label_width;

    return bar_layout;
}

// Render inspector panel background (called after clear, before agent drawing)
@compute @workgroup_size(16, 16)
fn render_inspector(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (params.draw_enabled == 0u) { return; }

    // Only draw if we have a selected agent
    if (params.selected_agent_index == 0xFFFFFFFFu) {
        return;
    }

    let x = gid.x;
    let y = gid.y;

    let safe_width = max(params.window_width, 1.0);
    let safe_height = max(params.window_height, 1.0);
    let window_width = u32(safe_width);
    let window_height = u32(safe_height);
    let inspector_height = window_height;

    // Only render in the inspector region (rightmost 300 pixels)
    if (x >= INSPECTOR_WIDTH || y >= inspector_height) {
        return;
    }

    // Flip Y coordinate so y=0 is at bottom, increases upward
    let flipped_y = window_height - 1u - y;

    // Map to actual buffer position (rightmost area)
    let buffer_x = window_width - INSPECTOR_WIDTH + x;

    // Transparent background (no background drawn)
    var color = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    // No border drawn

    // Agent preview window: y between 0 and 300
    let preview_y_start = 0u;
    let preview_y_end = 300u;
    let preview_x_start = 10u;
    let preview_x_end = 290u;

    // No preview window background drawn

    // Gene bars: y between 300 and 800
    let bar_anchor_x = 5u;
    let bar_anchor_y = 300u;  // Start at y=300 to avoid overlapping preview window
    let bar_available_height = 500u;  // Height from 300 to 800
    let bars = calculate_bar_layout(bar_anchor_x, bar_anchor_y, bar_available_height);

    let in_genome_bar = x >= bars.genome_x_start && x < bars.genome_x_end && flipped_y >= bars.bars_y_start && flipped_y < bars.bars_y_end;
    let in_amino_bar = x >= bars.amino_x_start && x < bars.amino_x_end && flipped_y >= bars.bars_y_start && flipped_y < bars.bars_y_end;
    let in_label_area = x >= bars.label_x_start && x < bars.label_x_end && flipped_y >= bars.bars_y_start && flipped_y < bars.bars_y_end;
    let in_alpha_bar = x >= bars.alpha_x_start && x < bars.alpha_x_end && flipped_y >= bars.bars_y_start && flipped_y < bars.bars_y_end;
    let in_beta_bar = x >= bars.beta_x_start && x < bars.beta_x_end && flipped_y >= bars.bars_y_start && flipped_y < bars.bars_y_end;

    // Draw coordinate grid on inspector panel (100px squares) - HIDDEN
    // let grid_spacing = 100u;
    // let is_grid_line = (x % grid_spacing == 0u || flipped_y % grid_spacing == 0u);
    // if (is_grid_line && x < 300u && flipped_y < window_height) {
    //     // Draw grid lines in bright cyan
    //     color = vec4<f32>(0.0, 1.0, 1.0, 1.0);
    // }

    // Draw coordinate numbers at grid intersections (2x scale for bigger text) - HIDDEN
    // let grid_x = (x / grid_spacing) * grid_spacing;
    // let grid_y = (flipped_y / grid_spacing) * grid_spacing;

    // Draw X coordinate (above the intersection point) - 2x scale
    // if (x >= grid_x + 2u && x < grid_x + 36u && flipped_y >= grid_y + 2u && flipped_y < grid_y + 16u) {
    //     let px = (x - grid_x - 2u) / 2u;  // Scale down for 2x rendering
    //     let py = (flipped_y - grid_y - 2u) / 2u;
    //     if (draw_number(grid_x, grid_x, grid_y, px, py)) {
    //         color = vec4<f32>(1.0, 1.0, 0.0, 1.0);  // Yellow text
    //     }
    // }

    // Draw Y coordinate (below the X coordinate) - 2x scale - HIDDEN
    // if (x >= grid_x + 2u && x < grid_x + 36u && flipped_y >= grid_y + 18u && flipped_y < grid_y + 32u) {
    //     let px = (x - grid_x - 2u) / 2u;  // Scale down for 2x rendering
    //     let py = (flipped_y - grid_y - 18u) / 2u;
    //     if (draw_number(grid_y, grid_x, grid_y, px, py)) {
    //         color = vec4<f32>(0.0, 1.0, 0.0, 1.0);  // Green text
    //     }
    // }

    // Draw full genome bar (all nucleotides from first non-X triplet) - VERTICAL
    if (in_genome_bar) {
        let genome = selected_agent_buffer[0].genome;
        let body_count = selected_agent_buffer[0].body_count;
        let available_height = bars.bars_y_end - bars.bars_y_start;
        let genome_pixel_y = flipped_y - bars.bars_y_start;  // Pixel position in bar (0 to available_height)
        let pixels_per_base = 2u;  // 2 pixels per base
        let base_index = genome_pixel_y / pixels_per_base;  // Which base are we displaying

        // Always start from first non-X triplet (gene start)
        let gene_start = genome_find_first_coding_triplet(genome);

        // Find translation start to know where active region begins
        var translation_start = 0xFFFFFFFFu;
        if (params.require_start_codon == 1u) {
            translation_start = genome_find_start_codon(genome);
        } else {
            translation_start = gene_start;
        }

        // Find stop codon position by simulating translation
        var stop_codon_end = 0xFFFFFFFFu;
        if (gene_start != 0xFFFFFFFFu && translation_start != 0xFFFFFFFFu) {
            var pos_b = translation_start;
            // Skip start codon (AUG) - consumed for initiation, not translated
            if (params.require_start_codon == 1u) {
                pos_b = translation_start + 3u;
            }
            var part_count = 0u;
            let offset_bases = translation_start - gene_start;
            var cumulative_bases = offset_bases;

            for (var i = 0u; i < MAX_BODY_PARTS; i++) {
                // Use centralized translation function
                let step = translate_codon_step(genome, pos_b, params.ignore_stop_codons == 1u);

                if (!step.is_valid) {
                    if (step.is_stop && part_count >= body_count) {
                        stop_codon_end = gene_start + cumulative_bases + 3u;
                    }
                    break;
                }

                if (part_count >= body_count) {
                    break;
                }

                pos_b += step.bases_consumed;
                cumulative_bases += step.bases_consumed;
                part_count += 1u;
            }
        }

        // Find first run of 3+ consecutive 'X's to mark end of gene
        var first_x_position = GENOME_LENGTH;
        if (gene_start != 0xFFFFFFFFu) {
            for (var scan_pos = gene_start; scan_pos < GENOME_LENGTH - 2u; scan_pos++) {
                let ascii0 = genome_get_base_ascii(genome, scan_pos);
                let ascii1 = genome_get_base_ascii(genome, scan_pos + 1u);
                let ascii2 = genome_get_base_ascii(genome, scan_pos + 2u);
                // Found 3 consecutive X's
                if (ascii0 == 88u && ascii1 == 88u && ascii2 == 88u) {
                    first_x_position = scan_pos;
                    break;
                }
            }
        }

        if (gene_start != 0xFFFFFFFFu && base_index < GENOME_LENGTH - gene_start) {
            let actual_base_index = gene_start + base_index;
            // Only draw if before first 'X'
            if (actual_base_index < first_x_position && actual_base_index < GENOME_LENGTH) {
                let base_ascii = genome_get_base_ascii(genome, actual_base_index);

                var base_color = vec3<f32>(0.5, 0.5, 0.5);
                if (base_ascii == 65u) {  // 'A'
                    base_color = vec3<f32>(0.0, 1.0, 0.0);
                } else if (base_ascii == 85u) {  // 'U'
                    base_color = vec3<f32>(0.0, 0.5, 1.0);
                } else if (base_ascii == 71u) {  // 'G'
                    base_color = vec3<f32>(1.0, 1.0, 0.0);
                } else if (base_ascii == 67u) {  // 'C'
                    base_color = vec3<f32>(1.0, 0.0, 0.0);
                }

                // Dim inactive parts (before translation start or after stop codon) by 75%
                if (translation_start != 0xFFFFFFFFu && actual_base_index < translation_start) {
                    base_color *= 0.25;
                } else if (stop_codon_end != 0xFFFFFFFFu && actual_base_index >= stop_codon_end) {
                    base_color *= 0.25;
                }

                color = vec4<f32>(base_color, 1.0);
            }
        }
    }

    // Draw amino acid/organ bar (only translated parts up to body_count) - VERTICAL WITH LABELS
    if (in_amino_bar || in_label_area) {
        let genome = selected_agent_buffer[0].genome;
        let body_count = selected_agent_buffer[0].body_count;
        let available_height = bars.bars_y_end - bars.bars_y_start;
        let genome_pixel_y = flipped_y - bars.bars_y_start;  // Pixel position in bar
        let pixels_per_base = 2u;  // 2 pixels per base
        let base_index_in_bar = genome_pixel_y / pixels_per_base;

        // Gene always starts at first non-X triplet
        let gene_start = genome_find_first_coding_triplet(genome);

        // Translation starts at AUG (if required) or gene start
        var translation_start = 0xFFFFFFFFu;
        if (params.require_start_codon == 1u) {
            translation_start = genome_find_start_codon(genome);
        } else {
            translation_start = gene_start;
        }

        if (gene_start != 0xFFFFFFFFu && translation_start != 0xFFFFFFFFu) {
            // Calculate offset from gene start to translation start (in bases)
            let offset_bases = translation_start - gene_start;

            // Walk through genome following translation logic in base space
            var cumulative_bases = offset_bases;  // start accounting from gene_start
            var cumulative_pixels = cumulative_bases * pixels_per_base;
            var pos_b = translation_start;
            // Skip start codon (AUG) - consumed for initiation, not translated
            if (params.require_start_codon == 1u) {
                pos_b = translation_start + 3u;
                cumulative_bases += 3u;
                cumulative_pixels += 3u * pixels_per_base;
            }
            var part_count = 0u;

            for (var i = 0u; i < MAX_BODY_PARTS; i++) {
                // Use centralized translation function
                let step = translate_codon_step(genome, pos_b, params.ignore_stop_codons == 1u);

                // Handle invalid translation or stop
                if (!step.is_valid) {
                    // Draw stop codon if needed
                    if (step.is_stop && part_count >= body_count && in_amino_bar) {
                        let span_start_pixels = cumulative_pixels;
                        let span_end_pixels = cumulative_pixels + 3u * pixels_per_base;
                        if (genome_pixel_y >= span_start_pixels && genome_pixel_y < span_end_pixels) {
                            color = vec4<f32>(0.0, 0.0, 0.0, 1.0);  // black
                        }
                    }
                    break;
                }

                if (part_count >= body_count) {
                    break;
                }

                // Get decoded part type and organ status
                let base_type = get_base_part_type(step.part_type);
                let is_organ = (base_type >= 20u);

                // Map genome_pixel_y (in pixels) into this part's span
                let span_start_pixels = cumulative_pixels;
                let span_height = step.bases_consumed * pixels_per_base;
                let span_end_pixels = span_start_pixels + span_height;
                let span_mid_pixels = (span_start_pixels + span_end_pixels) / 2u;

                if (genome_pixel_y >= span_start_pixels && genome_pixel_y < span_end_pixels) {
                    if (in_amino_bar) {
                        // Regular amino acid or organ color bar
                        let props = get_amino_acid_properties(base_type);
                        var base_color = props.color;

                        // For clock organs, oscillate color based on clock_signal in _pad.x
                        if (is_organ && base_type == 31u && part_count < body_count) {
                            // Read clock_signal from this part's _pad.x (range -1 to +1)
                            let clock_signal = selected_agent_buffer[0].body[part_count]._pad.y;
                            // Modulate brightness: 0.5 to 1.5 range based on signal
                            let brightness = 1.0 + clock_signal * 0.5;
                            base_color = base_color * brightness;
                        }

                        // For magnitude sensors (38-41), use brighter color tones to differentiate from directional sensors
                        let is_magnitude_sensor = base_type == 38u || base_type == 39u || base_type == 40u || base_type == 41u;
                        if (is_magnitude_sensor) {
                            // Brighten the color by 30%
                            base_color = base_color * 1.3;
                        }

                        // For organs that need amplification (propeller, displacer, mouth, vampire mouth, agent sensors), calculate and apply
                        let needs_amplification = props.is_propeller || props.is_displacer || props.is_mouth || base_type == 33u || base_type == 34u || base_type == 35u;
                        if (part_count < body_count && needs_amplification) {
                            let part_pos = selected_agent_buffer[0].body[part_count].pos;

                            // Calculate amplification from nearby enablers
                            var amp = 0.0;
                            for (var e = 0u; e < body_count; e++) {
                                let e_base_type = get_base_part_type(selected_agent_buffer[0].body[e].part_type);
                                let e_props = get_amino_acid_properties(e_base_type);
                                if (e_props.is_inhibitor) { // enabler flag
                                    let enabler_pos = selected_agent_buffer[0].body[e].pos;
                                    let d = length(part_pos - enabler_pos);
                                    if (d < 20.0) {
                                        amp += max(0.0, 1.0 - d / 20.0);
                                    }
                                }
                            }
                            let amplification = min(amp, 1.0);

                            // Multiply color by amplification (brighter = more amplification)
                            base_color = base_color * (1.0 + amplification);
                        }

                        color = vec4<f32>(base_color, 1.0);
                    } else if (in_label_area) {
                        // Per-pixel vector mask: stable, no race with draw_thick_line writes
                        let local_pixel_y = genome_pixel_y - span_start_pixels;

                        // Only consider rows within text height
                        let text_height = 8.0;   // Slightly smaller legend text
                        let text_rows = u32(ceil(text_height));
                        if (local_pixel_y < text_rows) {
                            let name = get_part_name(base_type);

                            let local_x = x - bars.label_x_start;
                            let text_start_x = 20u;
                            var cursor = text_start_x;
                            let char_spacing = 1u;
                            var hit = false;

                            // Unrolled to avoid dynamic array indexing
                            let c0 = name.chars[0];
                            if (name.len > 0u) {
                                let cw0 = u32(ceil(get_char_width(c0) * text_height));
                                if (local_x >= cursor && local_x < cursor + cw0) {
                                    let px = local_x - cursor;
                                    let py = local_pixel_y;
                                    if (char_vector_mask(vec2<u32>(px, py), c0, text_height)) { hit = true; }
                                }
                                cursor += cw0 + char_spacing;
                            }

                            let c1 = name.chars[1];
                            if (name.len > 1u) {
                                let cw1 = u32(ceil(get_char_width(c1) * text_height));
                                if (local_x >= cursor && local_x < cursor + cw1) {
                                    let px = local_x - cursor;
                                    let py = local_pixel_y;
                                    if (char_vector_mask(vec2<u32>(px, py), c1, text_height)) { hit = true; }
                                }
                                cursor += cw1 + char_spacing;
                            }

                            let c2 = name.chars[2];
                            if (name.len > 2u) {
                                let cw2 = u32(ceil(get_char_width(c2) * text_height));
                                if (local_x >= cursor && local_x < cursor + cw2) {
                                    let px = local_x - cursor;
                                    let py = local_pixel_y;
                                    if (char_vector_mask(vec2<u32>(px, py), c2, text_height)) { hit = true; }
                                }
                                cursor += cw2 + char_spacing;
                            }

                            let c3 = name.chars[3];
                            if (name.len > 3u) {
                                let cw3 = u32(ceil(get_char_width(c3) * text_height));
                                if (local_x >= cursor && local_x < cursor + cw3) {
                                    let px = local_x - cursor;
                                    let py = local_pixel_y;
                                    if (char_vector_mask(vec2<u32>(px, py), c3, text_height)) { hit = true; }
                                }
                                cursor += cw3 + char_spacing;
                            }

                            let c4 = name.chars[4];
                            if (name.len > 4u) {
                                let cw4 = u32(ceil(get_char_width(c4) * text_height));
                                if (local_x >= cursor && local_x < cursor + cw4) {
                                    let px = local_x - cursor;
                                    let py = local_pixel_y;
                                    if (char_vector_mask(vec2<u32>(px, py), c4, text_height)) { hit = true; }
                                }
                                cursor += cw4 + char_spacing;
                            }

                            let c5 = name.chars[5];
                            if (name.len > 5u) {
                                let cw5 = u32(ceil(get_char_width(c5) * text_height));
                                if (local_x >= cursor && local_x < cursor + cw5) {
                                    let px = local_x - cursor;
                                    let py = local_pixel_y;
                                    if (char_vector_mask(vec2<u32>(px, py), c5, text_height)) { hit = true; }
                                }
                                cursor += cw5 + char_spacing;
                            }

                            if (hit) {
                                color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
                            }
                        }
                    }
                }

                pos_b += step.bases_consumed;
                cumulative_bases += step.bases_consumed;
                cumulative_pixels += span_height;
                part_count += 1u;

                // Stop if we've rendered beyond the visible area
                if (cumulative_pixels > available_height) { break; }
            }
        }
    }

    // Draw signal bars (alpha and beta signals for each body part) - VERTICAL
    if (in_alpha_bar || in_beta_bar) {
        let genome = selected_agent_buffer[0].genome;
        let body_count = selected_agent_buffer[0].body_count;
        let available_height = bars.bars_y_end - bars.bars_y_start;
        let genome_pixel_y = flipped_y - bars.bars_y_start;  // Pixel position in bar
        let pixels_per_base = 2u;  // 2 pixels per base (must match amino bar)

        // Gene always starts at first non-X triplet
        let gene_start = genome_find_first_coding_triplet(genome);

        // Translation starts at AUG (if required) or gene start
        var translation_start = 0xFFFFFFFFu;
        if (params.require_start_codon == 1u) {
            translation_start = genome_find_start_codon(genome);
        } else {
            translation_start = gene_start;
        }

        if (gene_start != 0xFFFFFFFFu && translation_start != 0xFFFFFFFFu) {
            // Calculate offset from gene start to translation start (in bases)
            let offset_bases = translation_start - gene_start;

            // Walk through genome following translation logic in pixel space
            var cumulative_bases = offset_bases;
            var cumulative_pixels = cumulative_bases * pixels_per_base;
            var pos_b = translation_start;
            // Skip start codon (AUG) - consumed for initiation, not translated
            if (params.require_start_codon == 1u) {
                pos_b = translation_start + 3u;
                cumulative_bases += 3u;
                cumulative_pixels += 3u * pixels_per_base;
            }
            var part_count = 0u;

            for (var i = 0u; i < MAX_BODY_PARTS; i++) {
                // Use centralized translation function
                let step = translate_codon_step(genome, pos_b, params.ignore_stop_codons == 1u);

                if (!step.is_valid) {
                    break;
                }

                if (part_count >= body_count) {
                    break;
                }

                let base_type = get_base_part_type(step.part_type);

                // Map genome_pixel_y (in pixels) into this part's span
                let span_start_pixels = cumulative_pixels;
                let span_height = step.bases_consumed * pixels_per_base;
                let span_end_pixels = span_start_pixels + span_height;

                if (genome_pixel_y >= span_start_pixels && genome_pixel_y < span_end_pixels) {
                    // Get signals from actual body part
                    if (part_count < body_count) {
                        let part = selected_agent_buffer[0].body[part_count];
                        let a = part.alpha_signal;
                        let b = part.beta_signal;

                        // Debug mode color scheme: r=+beta, g=+alpha, blue=-alpha OR -beta
                        let r = max(b, 0.0);
                        let g = max(a, 0.0);
                        let bl = max(max(-a, 0.0), max(-b, 0.0));

                        color = vec4<f32>(r, g, bl, 1.0);
                    }
                }

                pos_b += step.bases_consumed;
                cumulative_bases += step.bases_consumed;
                cumulative_pixels += span_height;
                part_count += 1u;

                // Stop after drawing this part if it was before a stop codon
                if (step.is_stop) {
                    break;
                }

                // Stop if we've rendered beyond the visible area
                if (cumulative_pixels > available_height) { break; }
            }
        }
    }

    // Write to agent_grid using visual_stride
    let idx = y * params.visual_stride + buffer_x;
    agent_grid[idx] = color;
}

// Draw inspector agent (called after render_inspector, draws agent closeup in preview)
@compute @workgroup_size(16, 16)
fn draw_inspector_agent(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (params.draw_enabled == 0u) { return; }
    if (params.selected_agent_index == 0xFFFFFFFFu) { return; }

    // This shader doesn't need to do pixel-level work anymore
    // Just call render_body_part_ctx for each part with inspector context
    // Only run once (use thread 0,0)
    if (gid.x != 0u || gid.y != 0u) { return; }

    let body_count = min(selected_agent_buffer[0].body_count, MAX_BODY_PARTS);
    if (body_count == 0u) { return; }

    // Preview window setup (now at bottom with same coordinates as render_inspector)
    let preview_size = 280u;
    let preview_x_start = 10u;
    let safe_width = max(params.window_width, 1.0);
    let safe_height = max(params.window_height, 1.0);
    let window_width = u32(safe_width);
    let window_height = u32(safe_height);
    let preview_y_start = window_height - preview_size - 10u;  // 10px from bottom

    // Calculate auto-scale to fit agent
    var max_extent = 0.0;
    for (var i = 0u; i < MAX_BODY_PARTS; i++) {
        if (i < body_count) {
            let part = selected_agent_buffer[0].body[i];
            let dist = length(part.pos);
            max_extent = max(max_extent, dist + part.size);
        }
    }
    let available_space = f32(preview_size) * 0.45; // Half width, 90% of that
    let auto_scale = select(available_space / max_extent, 1.0, max_extent > 1.0);
    let scale_factor = auto_scale * params.inspector_zoom;  // Apply user zoom

    // Preview center
    let preview_center_x = f32(preview_x_start + preview_size / 2u);
    let preview_center_y = f32(preview_y_start + preview_size / 2u);

    // Calculate buffer offset (rightmost area)
    let buffer_offset_x = f32(window_width - INSPECTOR_WIDTH);

    // Create inspector context
    let ctx = InspectorContext(
        vec2<f32>(0.0, 0.0),  // use_inspector_coords (x >= 0 enables inspector mode)
        vec2<f32>(preview_center_x, preview_center_y),  // center of preview
        scale_factor,  // scale
        vec2<f32>(buffer_offset_x, 0.0)  // offset to actual buffer position
    );

    // Calculate agent color
    var color_sum = 0.0;
    for (var i = 0u; i < MAX_BODY_PARTS; i++) {
        if (i < body_count) {
            let base_type = get_base_part_type(selected_agent_buffer[0].body[i].part_type);
            let part_props = get_amino_acid_properties(base_type);
            color_sum += part_props.beta_damage;
        }
    }
    let agent_color = vec3<f32>(
        sin(color_sum * 3.0) * 0.5 + 0.5,
        sin(color_sum * 5.0) * 0.5 + 0.5,
        sin(color_sum * 7.0) * 0.5 + 0.5
    );

    // Get morphology origin (where chain starts in local space)
    let morphology_origin = selected_agent_buffer[0].morphology_origin;

    // Render all body parts using the unified render function
    // Note: agent is unrotated (rotation=0) in selected_agent_buffer
    let in_debug_mode = params.debug_mode != 0u;
    for (var i = 0u; i < MAX_BODY_PARTS; i++) {
        if (i >= body_count) { break; }

        let part = selected_agent_buffer[0].body[i];

        // Use special agent_id value 0xFFFFFFFFu to indicate selected_agent_buffer access
        render_body_part_ctx(
            part,
            i,
            0xFFFFFFFFu,  // special value to use selected_agent_buffer instead of agents_out
            vec2<f32>(0.0, 0.0),  // agent_position (will be offset by ctx)
            0.0,  // agent_rotation (already unrotated)
            selected_agent_buffer[0].energy,
            agent_color,
            body_count,
            morphology_origin,
            1.0,  // amplification
            in_debug_mode,
            ctx
        );
    }

    // ============================================================================
    // TEXT LABELS - Display agent information with scalable vector font
    // ============================================================================
    let text_color = vec4<f32>(1.0, 1.0, 1.0, 1.0);

    // Direct pixel inspector context (origin at inspector top-left).
    // Use scale of 0.33 to make lines thinner (1px * 0.33 G?? 0.33px which rounds to 1px)
    let text_ctx = InspectorContext(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
        0.33,
        vec2<f32>(buffer_offset_x, 0.0)
    );

    let text_height = 14.0;  // Will be scaled down by ctx.scale
    let char_width = 9.0;    // Approximate character width at this size (will be scaled)
    let line_height = 18.0;  // Spacing between lines (will be scaled)
    let max_width = 280.0;   // 300 - 20px padding

    // Simple top-down layout with explicit Y positions
    // Bars are at screen_height - 80, so text should start below that
    // Offset from bottom: bars at -80px, so text starts at -80px + 80px (bar height) = top
    // Actually place text starting from a safe position below bars
    let energy_y = f32(window_height) - 280.0 - 10.0 + 280.0 + 90.0;  // Below bars (bars ~80px tall)
    let organs_y = energy_y + 30.0;  // 30px below energy line

    // ENERGY label
    var energy_text: array<u32, 32>;
    energy_text[0] = 69u;  // 'E'
    energy_text[1] = 78u;  // 'N'
    energy_text[2] = 69u;  // 'E'
    energy_text[3] = 82u;  // 'R'
    energy_text[4] = 71u;  // 'G'
    energy_text[5] = 89u;  // 'Y'
    energy_text[6] = 58u;  // ':'
    energy_text[7] = 32u;  // ' '
    let energy_prefix_len = 8u;
    let energy_value_len = f32_to_string(selected_agent_buffer[0].energy, &energy_text, energy_prefix_len);
    let energy_total_len = energy_prefix_len + energy_value_len;
    let energy_pos = vec2<f32>(10.0, energy_y);
    draw_string_vector(energy_pos, &energy_text, energy_total_len, text_height, text_color, text_ctx);

    // Organ letters wrapped across multiple lines
    var current_y = organs_y;
    var current_x = 10.0;

    for (var i = 0u; i < body_count; i++) {
        let part = selected_agent_buffer[0].body[i];
        let base_type = get_base_part_type(part.part_type);

        // Get organ properties for color
        let part_props = get_amino_acid_properties(base_type);
        let letter_color = vec4<f32>(part_props.color, 1.0);

        // Convert organ type to letter
        var letter = 63u; // '?' default
        if (base_type < 20u) {
            // Amino acids A-T (0-19)
            letter = 65u + base_type;  // 'A' + offset
        } else {
            // Organs - assign letters
            switch (base_type) {
                case 20u: { letter = 77u; }  // 'M' = Mouth
                case 21u: { letter = 80u; }  // 'P' = Propeller
                case 22u: { letter = 65u; }  // 'A' = Alpha Sensor
                case 23u: { letter = 66u; }  // 'B' = Beta Sensor
                case 24u: { letter = 69u; }  // 'E' = Energy Sensor
                case 25u: { letter = 68u; }  // 'D' = Displacer
                case 26u: { letter = 78u; }  // 'N' = eNabler
                case 28u: { letter = 83u; }  // 'S' = Storage
                case 29u: { letter = 82u; }  // 'R' = poison Resistance
                case 30u: { letter = 67u; }  // 'C' = Chiral Flipper
                case 31u: { letter = 88u; }  // 'X' = Clock
                case 32u: { letter = 47u; }  // '/' = Slope Sensor
                case 33u: { letter = 86u; }  // 'V' = Vampire mouth
                case 34u: { letter = 60u; }  // '<' = Agent alpha sensor
                case 35u: { letter = 62u; }  // '>' = Agent beta sensor
                default: { letter = 63u; }   // '?' = unknown
            }
        }

        // Check if we need to wrap to next line
        if (current_x + char_width > max_width) {
            current_x = 10.0;
            current_y += line_height;  // Move down
        }

        // Draw the character with organ color
        let char_pos = vec2<f32>(current_x, current_y);
        current_x += draw_char_vector(char_pos, letter, text_height, letter_color, text_ctx);
    }
}

// ============================================================================
// AGENT RENDERING KERNEL
// ============================================================================

// Dedicated kernel to render all agents to agent_grid
// This runs after process_agents and before composite_agents
@compute @workgroup_size(256)
fn render_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (params.draw_enabled == 0u) { return; }

    let agent_id = gid.x;
    if (agent_id >= params.max_agents) {
        return;
    }

    // Only render alive agents
    if (agents_out[agent_id].alive == 0u) {
        return;
    }

    // Skip if no body parts
    let body_count = agents_out[agent_id].body_count;
    if (body_count == 0u) {
        return;
    }

    // Calculate camera bounds with aspect ratio
    let aspect_ratio = params.window_width / params.window_height;
    let camera_half_height = params.grid_size / (2.0 * params.camera_zoom);
    let camera_half_width = camera_half_height * aspect_ratio;
    let camera_center = vec2<f32>(params.camera_pan_x, params.camera_pan_y);
    let camera_min = camera_center - vec2<f32>(camera_half_width, camera_half_height);
    let camera_max = camera_center + vec2<f32>(camera_half_width, camera_half_height);

    // Frustum culling - check if agent is visible
    let margin = 20.0; // Maximum body extent
    let center = agents_out[agent_id].position;
    if (center.x + margin < camera_min.x || center.x - margin > camera_max.x ||
        center.y + margin < camera_min.y || center.y - margin > camera_max.y) {
        return; // Not visible, skip rendering
    }

    // Agent is visible - render it
    let in_debug_mode = params.debug_mode != 0u;

    // Calculate agent color
    var color_sum = 0.0;
    for (var i = 0u; i < MAX_BODY_PARTS; i++) {
        if (i < body_count) {
            let base_type = get_base_part_type(agents_out[agent_id].body[i].part_type);
            let part_props = get_amino_acid_properties(base_type);
            color_sum += part_props.beta_damage;
        }
    }
    let agent_color = vec3<f32>(
        sin(color_sum * 3.0) * 0.5 + 0.5,
        sin(color_sum * 5.0) * 0.5 + 0.5,
        sin(color_sum * 7.0) * 0.5 + 0.5
    );

    // Get the morphology origin
    let morphology_origin = agents_out[agent_id].morphology_origin;

    // Draw all body parts using unified rendering function
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let part = agents_out[agent_id].body[i];
        // Calculate jet amplification for this part
        var jet_amplification = 1.0;
        if (get_base_part_type(part.part_type) == 23u) { // Jet organ
            let alpha_signal = part.alpha_signal;
            let beta_signal = part.beta_signal;
            jet_amplification = sqrt(alpha_signal * alpha_signal + beta_signal * beta_signal) * 3.0;
        }

        render_body_part(
            part,
            i,
            agent_id,
            center,
            agents_out[agent_id].rotation,
            agents_out[agent_id].energy,
            agent_color,
            body_count,
            morphology_origin,
            jet_amplification,
            in_debug_mode
        );
    }

    if (in_debug_mode) {
        // Draw center cross marker only in debug mode
        let cross_size = 3.0;
        draw_thick_line(
            center + vec2<f32>(-cross_size, 0.0),
            center + vec2<f32>(cross_size, 0.0),
            0.5,
            vec4<f32>(1.0, 1.0, 1.0, 1.0),
        );
        draw_thick_line(
            center + vec2<f32>(0.0, -cross_size),
            center + vec2<f32>(0.0, cross_size),
            0.5,
            vec4<f32>(1.0, 1.0, 1.0, 1.0),
        );

        // Check if agent has vampire mouth (organ 33) and mark with red circle
        var has_vampire_mouth = false;
        for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
            let base_type = get_base_part_type(agents_out[agent_id].body[i].part_type);
            if (base_type == 33u) {
                has_vampire_mouth = true;
                break;
            }
        }

        if (has_vampire_mouth) {
            // Draw red circle around agent with vampire mouth
            let circle_radius = 15.0;
            let segments = 24u;
            var prev = center + vec2<f32>(circle_radius, 0.0);
            for (var s = 1u; s <= segments; s++) {
                let t = f32(s) / f32(segments);
                let ang = t * 6.28318530718;
                let p = center + vec2<f32>(cos(ang) * circle_radius, sin(ang) * circle_radius);
                draw_thick_line(prev, p, 2.0, vec4<f32>(1.0, 0.0, 0.0, 0.9));
                prev = p;
            }
        }
    }

    // Debug: count visible agents
    atomicAdd(&debug_counter, 1u);

    // Draw selection circle if this agent is selected
    if (agents_out[agent_id].is_selected == 1u) {
        draw_selection_circle(center, agent_id, body_count);
    }
}

// ============================================================================
// COMPOSITE.WGSL - COMPOSITING AND BLENDING
// ============================================================================
// This file contains the final compositing stage that blends the rendered
// agents (from agent_grid) onto the environment background (visual_grid).
//
// Supports multiple blend modes:
// - Mode 0: Normal (alpha blend)
// - Mode 1: Add (additive blending with color tint)
// - Mode 2: Subtract (subtractive blending with color tint)
// - Mode 3: Multiply (multiplicative blending with color tint)
// ============================================================================

@compute @workgroup_size(16, 16)
fn composite_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    let idx = y * params.visual_stride + x;
    let agent_pixel = agent_grid[idx];

    // Skip if no agent drawn here (transparent)
    if (agent_pixel.a == 0.0) {
        return;
    }

    let base_color = visual_grid[idx].rgb;
    let agent_color_param = vec3<f32>(
        clamp(params.agent_color_r, 0.0, 1.0),
        clamp(params.agent_color_g, 0.0, 1.0),
        clamp(params.agent_color_b, 0.0, 1.0)
    );

    var result_color = vec3<f32>(0.0);

    if (params.agent_blend_mode == 0u) {
        // Comp (normal) - alpha blend agent on top of base
        result_color = mix(base_color, agent_pixel.rgb, agent_pixel.a);
    } else if (params.agent_blend_mode == 1u) {
        // Add - add agent color tint to agent pixel, then composite
        let tinted_agent = clamp(agent_pixel.rgb + agent_color_param, vec3<f32>(0.0), vec3<f32>(1.0));
        result_color = clamp(base_color + tinted_agent * agent_pixel.a, vec3<f32>(0.0), vec3<f32>(1.0));
    } else if (params.agent_blend_mode == 2u) {
        // Subtract - subtract agent color tint from agent pixel, then composite
        let tinted_agent = clamp(agent_pixel.rgb - agent_color_param, vec3<f32>(0.0), vec3<f32>(1.0));
        result_color = mix(base_color, tinted_agent, agent_pixel.a);
    } else {
        // Multiply - multiply agent by color tint, then composite
        let tinted_agent = agent_pixel.rgb * agent_color_param;
        result_color = mix(base_color, tinted_agent, agent_pixel.a);
    }

    visual_grid[idx] = vec4<f32>(result_color, 1.0);
}
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

    // No neighbors to drain
    if (neighbor_count == 0u) {
        return;
    }

    // Process each vampire mouth organ (F=4u, G=5u, H=6u)
    var total_energy_gained = 0.0;

    // Check for enabler/disabler organs to control vampire mouth activity
    var enabler_sum = 0.0;
    var disabler_sum = 0.0;
    for (var i = 0u; i < MAX_BODY_PARTS; i++) {
        if (i >= body_count) { break; }
        let base_type = get_base_part_type(agents_in[agent_id].body[i].part_type);
        if (base_type == 26u) {  // Enabler
            enabler_sum += 1.0;
        } else if (base_type == 27u) {  // Disabler
            disabler_sum += 1.0;
        }
    }
    // Normal mode: positive enabler_sum enables, positive disabler_sum disables
    let global_mouth_activity = clamp(enabler_sum - disabler_sum, 0.0, 1.0);

    if (global_mouth_activity > 0.01) {
        for (var i = 0u; i < MAX_BODY_PARTS; i++) {
            if (i >= body_count) { break; }
            let part = agents_in[agent_id].body[i];
            let base_type = get_base_part_type(part.part_type);

            // Check if this is a vampire mouth organ (type 33)
            if (base_type == 33u) {
                // Decrement cooldown timer (stored in _pad.x)
                var current_cooldown = agents_in[agent_id].body[i]._pad.x;
                if (current_cooldown > 0.0) {
                    current_cooldown -= 1.0;
                    agents_in[agent_id].body[i]._pad.x = current_cooldown;
                }

                // Get mouth world position
                let part_pos = part.pos;
                let rotated_pos = apply_agent_rotation(part_pos, agents_in[agent_id].rotation);
                let mouth_world_pos = agents_in[agent_id].position + rotated_pos;

                // Find closest victim within drain range
                var closest_victim_id = 0xFFFFFFFFu;
                var closest_dist = 999999.0;
                let max_drain_distance = 100.0;

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
                        // INSTANT KILL: Vampire drains ALL energy from victim
                        let victim_energy = agents_in[closest_victim_id].energy;

                        // Distance-based falloff: full power at 0 distance, 0 power at max_drain_distance
                        let distance_factor = max(0.0, 1.0 - (closest_dist / max_drain_distance));

                        // Apply global mouth activity and distance falloff to determine if kill succeeds
                        let kill_effectiveness = distance_factor * global_mouth_activity;

                        // Minimum 10% absorption threshold - don't kill if effectiveness is too low
                        if (victim_energy > 0.0001 && kill_effectiveness >= 0.1) {
                            // Vampire absorbs ALL victim's energy (minimum 10% effectiveness ensures worthwhile kill)
                            let absorbed_energy = victim_energy * kill_effectiveness;

                            // Victim loses ALL energy (instant death)
                            agents_in[closest_victim_id].energy = 0.0;

                            total_energy_gained += absorbed_energy;

                            // Store absorbed amount in _pad.y for visualization
                            agents_in[agent_id].body[i]._pad.y = absorbed_energy;

                            // Set cooldown timer
                            agents_in[agent_id].body[i]._pad.x = VAMPIRE_MOUTH_COOLDOWN;
                        } else {
                            agents_in[agent_id].body[i]._pad.y = 0.0;
                        }

                        // Keep cell claimed (cleared next frame) - this prevents other vampires from attacking same victim
                    } else {
                        // Failed to claim - another vampire got here first
                        agents_in[agent_id].body[i]._pad.y = 0.0;
                    }
                } else {
                    agents_in[agent_id].body[i]._pad.y = 0.0;
                }
            }
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
        var vampiric_mouth_count = 0u;

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

            // Count vampiric mouth organs (type 33) for normal mouth reduction
            if (base_type == 33u) {
                vampiric_mouth_count += 1u;
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
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let part_pos = agents_out[agent_id].body[i].pos;
        let base_type = get_base_part_type(agents_out[agent_id].body[i].part_type);
        let amino_props = get_amino_acid_properties(base_type);

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

        // Sensors: stochastic gaussian sampling with 50% smoothing
        // Sample radius based on part size (larger radius for better field integration)
        let sensor_radius = 500.0;

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

            let sensed_value = sample_stochastic_gaussian(world_pos, sensor_radius, sensor_seed, 0u, params.debug_mode != 0u, perpendicular_world, promoter_param1, modifier_param1);
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

            let sensed_value = sample_stochastic_gaussian(world_pos, sensor_radius, sensor_seed, 1u, params.debug_mode != 0u, perpendicular_world, promoter_param1, modifier_param1);
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

            let sensed_value = sample_neighbors_energy(world_pos, sensor_radius, params.debug_mode != 0u, perpendicular_world, &neighbor_ids, neighbor_count);
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

            let sensed_value = sample_magnitude_only(world_pos, sensor_radius, sensor_seed, 0u, params.debug_mode != 0u, promoter_param1, modifier_param1);
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

            let sensed_value = sample_magnitude_only(world_pos, sensor_radius, sensor_seed, 1u, params.debug_mode != 0u, promoter_param1, modifier_param1);
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

            let sensed_value = sample_magnitude_only(world_pos, sensor_radius, sensor_seed, 0u, params.debug_mode != 0u, promoter_param1, modifier_param1);
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

            let sensed_value = sample_magnitude_only(world_pos, sensor_radius, sensor_seed, 1u, params.debug_mode != 0u, promoter_param1, modifier_param1);
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

            let sensed_value = sample_magnitude_only(world_pos, sensor_radius, sensor_seed, 0u, params.debug_mode != 0u, promoter_param1, modifier_param1);
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

            let sensed_value = sample_magnitude_only(world_pos, sensor_radius, sensor_seed, 1u, params.debug_mode != 0u, promoter_param1, modifier_param1);
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
            let sensed_value = sample_neighbors_color(sensor_world_pos, sensor_radius, params.debug_mode != 0u, perpendicular_world, agent_color, &neighbor_ids, neighbor_count);

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
            let sensed_value = sample_neighbors_color(sensor_world_pos, sensor_radius, params.debug_mode != 0u, perpendicular_world, agent_color, &neighbor_ids, neighbor_count);

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
                // Use _pad.y to accumulate sensed field over time
                var internal_value = agents_out[agent_id].body[i]._pad.y;
                internal_value += sensed_field;
                agents_out[agent_id].body[i]._pad.y = internal_value;

                let x = internal_value * param1;
                clock_signal = sin(x);
            }

            // Store clock signal in _pad.x for rendering
            agents_out[agent_id].body[i]._pad.x = clock_signal;

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

            // Use reduced mass for proper two-body physics: ÃŽÂ¼ = (m1 * m2) / (m1 + m2)
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

        // Slope force per amino acid
        let slope_gradient = read_gamma_slope(grid_index(world_pos));
        let slope_force = -slope_gradient * params.gamma_strength * part_mass;
        force += slope_force;
        torque += (r_com.x * slope_force.y - r_com.y * slope_force.x);

        // Cached amplification for this part (organs will use it, organs may ignore)
        let amplification = amplification_per_part[i];

        let part_weight = part_mass / total_mass;

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
        // Only mouths need to track previous position for speed-based absorption
        var mouth_speed = 0.0;
        var speed_absorption_multiplier = 1.0;

        if (props.is_mouth) {
            let packed_prev = bitcast<u32>(agents_out[agent_id].body[i]._pad.x);
            let prev_pos = unpack_prev_pos(packed_prev);
            let displacement_vec = world_pos - prev_pos;
            let displacement_sq = dot(displacement_vec, displacement_vec);
            // If displacement is very small OR prev_pos is near origin, treat as first frame
            let looks_like_first = (displacement_sq < 0.01) || (dot(prev_pos, prev_pos) < 1.0);
            mouth_speed = select(sqrt(displacement_sq), 0.0, looks_like_first);
            let normalized_mouth_speed = mouth_speed / VEL_MAX;
            speed_absorption_multiplier = exp(-8.0 * normalized_mouth_speed);

            // Update previous world position for next frame (only for mouths)
            agents_out[agent_id].body[i]._pad.x = bitcast<f32>(pack_prev_pos(world_pos));
        }

        // Debug output for first agent only
        if (agent_id == 0u && params.debug_mode != 0u && i == 0u) {
            agents_out[agent_id].body[63].pos.x = mouth_speed;
            agents_out[agent_id].body[63].pos.y = speed_absorption_multiplier;
        }

        // 1) Trail deposition: blend agent color + deposit energy trail
        let current_trail = trail_grid[idx].xyz;
        let blended = mix(current_trail, agent_color, trail_deposit_strength);

        // Deposit energy trail (unclamped) - scale by agent energy
        let current_energy_trail = trail_grid[idx].w;
        let energy_deposit = agent.energy * trail_deposit_strength * 0.1; // 10% of energy deposited
        let blended_energy = current_energy_trail + energy_deposit;

        trail_grid[idx] = vec4<f32>(clamp(blended, vec3<f32>(0.0), vec3<f32>(1.0)), blended_energy);

        // 2) Energy consumption: calculate costs per organ type
        // Calculate global vampire mouth activity for this agent (used for vampire mouth cost)
        var global_mouth_activity = 1.0; // Default: active
        if (base_type == 33u) { // Only calculate for vampire mouths
            var enabler_sum = 0.0;
            var disabler_sum = 0.0;
            for (var j = 0u; j < agents_out[agent_id].body_count; j++) {
                let check_type = get_base_part_type(agents_out[agent_id].body[j].part_type);
                if (check_type == 26u) { enabler_sum += 1.0; }      // Enabler
                else if (check_type == 27u) { disabler_sum += 1.0; } // Disabler
            }
            global_mouth_activity = clamp(enabler_sum - disabler_sum, 0.0, 1.0);
        }

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
            // Vampire mouths: high cost when active (global_mouth_activity from enablers/disablers)
            // Base cost always paid, but heavy penalty when actively draining
            let amp = amplification_per_part[i];
            let activity_cost = props.energy_consumption * global_mouth_activity * amp * 3.0; // 3x multiplier for vampire cost
            organ_extra = props.energy_consumption + activity_cost; // Base + activity penalty
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
        let current_count = atomicLoad(&alive_counter);
        if (current_count < params.max_agents) {
            let spawn_index = atomicAdd(&spawn_counter, 1u);
            if (spawn_index < 2000u) {
                // Generate hash for offspring randomization
                // CRITICAL: Include agent_id to ensure each parent's offspring gets unique mutations
                let offspring_hash = (hash3 ^ (spawn_index * 0x9e3779b9u) ^ (agent_id * 0x85ebca6bu)) * 1664525u + 1013904223u;

                // Create brand new offspring agent (don't copy parent)
                var offspring: Agent;

                // Random rotation
                offspring.rotation = hash_f32(offspring_hash) * 6.28318530718;

                // Spawn at same location as parent
                offspring.position = agent.position;
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

    // Get current values
    let current_alpha = alpha_grid[idx];
    let current_beta = beta_grid[idx];
    let current_gamma = gamma_grid[idx];
    let slope_here = read_gamma_slope(idx);

    var alpha_sum = 0.0;
    var beta_sum = 0.0;
    var gamma_sum = 0.0;

    for (var i = 0u; i < 9u; i++) {
        let dx = i32(i % 3u) - 1;
        let dy = i32(i / 3u) - 1;
        let nx_i = clamp(i32(x) + dx, 0, i32(GRID_SIZE) - 1);
        let ny_i = clamp(i32(y) + dy, 0, i32(GRID_SIZE) - 1);
        let nidx = u32(ny_i) * GRID_SIZE + u32(nx_i);
        let alpha_val = alpha_grid[nidx];
        let beta_val = beta_grid[nidx];
        let gamma_val = gamma_grid[nidx];

        alpha_sum += alpha_val;
        beta_sum += beta_val;
        gamma_sum += gamma_val;
    }

    let alpha_avg = alpha_sum / 9.0;
    let beta_avg = beta_sum / 9.0;
    let gamma_avg = gamma_sum / 9.0;

    // Apply blur factor (0 = no blur/keep current, 1 = full blur)
    let new_alpha = mix(current_alpha, alpha_avg, params.alpha_blur);
    let new_beta = mix(current_beta, beta_avg, params.beta_blur);
    let new_gamma = mix(current_gamma, gamma_avg, params.gamma_blur);

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

    let alpha_left = alpha_grid[left_idx];
    let alpha_right = alpha_grid[right_idx];
    let alpha_up = alpha_grid[up_idx];
    let alpha_down = alpha_grid[down_idx];
    let beta_left = beta_grid[left_idx];
    let beta_right = beta_grid[right_idx];
    let beta_up = beta_grid[up_idx];
    let beta_down = beta_grid[down_idx];

    let slope_left = read_gamma_slope(left_idx);
    let slope_right = read_gamma_slope(right_idx);
    let slope_up = read_gamma_slope(up_idx);
    let slope_down = read_gamma_slope(down_idx);

    let kernel_scale = 1.0 / 8.0;

    // Alpha fluxes (mass-conserving advection along slopes)
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

    // Alpha grid constrained to 0..1
    var final_alpha = clamp(new_alpha - alpha_flux * kernel_scale, 0.0, 1.0);

    // Beta fluxes reuse the same slope field with independent strength
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

    // Beta grid constrained to 0..1
    var final_beta = clamp(new_beta - beta_flux * kernel_scale, 0.0, 1.0);

    // Stochastic rain - randomly add food/poison droplets (saturated drops)
    // Use position and random seed to generate unique random values per cell
    let cell_seed = idx * 2654435761u + params.random_seed;
    let rain_chance = f32(hash(cell_seed)) / 4294967295.0;

    // Uniform alpha rain (food): remove spatial and beta-dependent gradients.
    // Each cell independently receives a saturated rain event with probability alpha_multiplier * 0.05.
    // (Scaling by 0.05 preserves prior expected value semantics.)
    let alpha_rain_factor = clamp(rain_map[idx].x, 0.0, 1.0);
    let alpha_probability_sat = params.alpha_multiplier * 0.05 * alpha_rain_factor;
    if (rain_chance < alpha_probability_sat) {
        final_alpha = 1.0;  // Saturated drop
    }

    // Uniform beta rain (poison): also no vertical gradient. Probability = beta_multiplier * 0.05.
    let beta_seed = cell_seed * 1103515245u;
    let beta_rain_chance = f32(hash(beta_seed)) / 4294967295.0;
    let beta_rain_factor = clamp(rain_map[idx].y, 0.0, 1.0);
    let beta_probability_sat = params.beta_multiplier * 0.05 * beta_rain_factor;
    if (beta_rain_chance < beta_probability_sat) {
        final_beta = 1.0;  // Saturated drop
    }

    // Apply the diffused values
    alpha_grid[idx] = final_alpha;
    beta_grid[idx] = final_beta;
    // Gamma also switched to 0..1
    gamma_grid[idx] = clamp(new_gamma, 0.0, 1.0);
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

// ============================================================================
// RGB TRAIL DIFFUSION
// ============================================================================

@compute @workgroup_size(16, 16)
fn diffuse_trails(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if (x >= GRID_SIZE || y >= GRID_SIZE) {
        return;
    }

    let idx = y * GRID_SIZE + x;

    // Get current trail value (RGB)
    let current_trail = trail_grid[idx].xyz;

    // Sample 3x3 neighborhood for diffusion
    var trail_sum = vec3<f32>(0.0);

    for (var i = 0u; i < 9u; i++) {
        let dx = i32(i % 3u) - 1;
        let dy = i32(i / 3u) - 1;
        let nx_i = clamp(i32(x) + dx, 0, i32(GRID_SIZE) - 1);
        let ny_i = clamp(i32(y) + dy, 0, i32(GRID_SIZE) - 1);
        let nidx = u32(ny_i) * GRID_SIZE + u32(nx_i);

    trail_sum += trail_grid[nidx].xyz;
    }

    let trail_avg = trail_sum / 9.0;

    // Diffusion: blend current with average (controlled by trail_diffusion parameter)
    let new_trail = mix(current_trail, trail_avg, params.trail_diffusion);

    // Decay: gradually fade trails over time (controlled by trail_decay parameter)
    let faded = clamp(new_trail * params.trail_decay, vec3<f32>(0.0), vec3<f32>(1.0));

    // Decay energy trail (alpha channel) separately
    let current_energy_trail = trail_grid[idx].w;
    let faded_energy = current_energy_trail * params.trail_decay;

    trail_grid[idx] = vec4<f32>(faded, faded_energy);
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Calculate pixel coordinates based on window size
    let safe_width = max(render_params.window_width, 1.0);
    let safe_height = max(render_params.window_height, 1.0);
    let window_width = u32(safe_width);
    let window_height = u32(safe_height);

    let pixel_x = u32(in.uv.x * f32(window_width));
    let pixel_y = u32(in.uv.y * f32(window_height));

    // Sample visual texture and composite with agent_grid (which includes inspector)
    let color = textureSample(visual_tex, visual_sampler, in.uv);

    // Check if there's an agent pixel to composite (or inspector if selected)
    let idx = pixel_y * render_params.visual_stride + pixel_x;
    let agent_pixel = agent_grid_render[idx];

    // Composite agent on top if it has alpha
    if (agent_pixel.a > 0.0) {
        return vec4<f32>(agent_pixel.rgb, 1.0);
    }

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
}

// ============================================================================
// SPAWN/DEATH MANAGEMENT SHADERS
// ============================================================================

// Merge spawned agents into main buffer
@compute @workgroup_size(64)
fn merge_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
    let spawn_id = gid.x;
    let spawn_count = atomicLoad(&spawn_counter);

    if (spawn_id >= spawn_count) {
        return;
    }

    // Append to end of compacted alive array using alive_counter as running size
    let target_index = atomicAdd(&alive_counter, 1u);
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

    let alive_now = atomicLoad(&alive_counter);
    if (alive_now >= params.max_agents) {
        return;
    }

    var spawn_index: u32 = 0u;
    loop {
        let current_spawn = atomicLoad(&spawn_counter);
        if (current_spawn >= 2000u) {
            return;
        }
        if (alive_now + current_spawn >= params.max_agents) {
            return;
        }
        let result = atomicCompareExchangeWeak(&spawn_counter, current_spawn, current_spawn + 1u);
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

    let alive_total = atomicLoad(&alive_counter);
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
        let idx = atomicAdd(&alive_counter, 1u);
        if (idx < params.max_agents) {
            agents_out[idx] = agent;
        }
    }
}

// Reset spawn counter for next frame
@compute @workgroup_size(1)
fn reset_spawn_counter(@builtin(global_invocation_id) gid: vec3<u32>) {
    atomicStore(&spawn_counter, 0u);
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

