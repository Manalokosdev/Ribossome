// Ribossome - GPU-Accelerated Artificial Life Simulator
// Copyright (c) 2025 Filipe da Veiga Ventura Alves
// Licensed under MIT License

// Shared WGSL module: constants, structs, bindings, amino/organ tables, utilities, genome helpers, translation logic

// ============================================================================
// CONSTANTS
// ============================================================================

const ENV_GRID_SIZE: u32 = 2048u;      // Environment grid resolution (alpha/beta/gamma)
const GRID_SIZE: u32 = ENV_GRID_SIZE;  // Alias for backward compatibility
const SPATIAL_GRID_SIZE: u32 = 1024u;   // Spatial hash grid for agent collision detection
const SIM_SIZE: u32 = 30720u;          // Simulation world size (doubled)
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
    data: f32,                // generic data slot (was size, now computed on-demand) (4 bytes)
    part_type: u32,           // bits 0-7 = base type (amino acid or organ), bits 8-15 = organ parameter
    alpha_signal: f32,        // alpha signal propagating through body (4 bytes)
    beta_signal: f32,         // beta signal propagating through body (4 bytes)
    _pad: vec2<f32>,          // padding to 32 bytes (8 bytes)
                              // _pad.x = smoothed signal angle OR condenser charge OR clock signal OR vampire cooldown
                              // _pad.y = packed u16 prev_world_pos OR last drain amount (vampire mouths)
}

// Helper function to compute visual size from part properties
fn get_part_visual_size(part_type: u32) -> f32 {
    let base_type = get_base_part_type(part_type);
    let props = get_amino_acid_properties(base_type);
    var size = props.thickness * 0.5;
    let is_sensor = props.is_alpha_sensor || props.is_beta_sensor || props.is_energy_sensor || props.is_agent_alpha_sensor || props.is_agent_beta_sensor || props.is_trail_energy_sensor;
    if (is_sensor) {
        size *= 2.0;
    }
    if (props.is_condenser) {
        size *= 0.5;
    }
    return size;
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
    ignore_stop_codons: u32,
    require_start_codon: u32,
    asexual_reproduction: u32,
    background_color_r: f32,
    background_color_g: f32,
    background_color_b: f32,
    alpha_blend_mode: u32,
    beta_blend_mode: u32,
    gamma_blend_mode: u32,
    slope_blend_mode: u32,
    alpha_color_r: f32,
    alpha_color_g: f32,
    alpha_color_b: f32,
    beta_color_r: f32,
    beta_color_g: f32,
    beta_color_b: f32,
    gamma_color_r: f32,
    gamma_color_g: f32,
    gamma_color_b: f32,
    grid_interpolation: u32,
    alpha_gamma_adjust: f32,
    beta_gamma_adjust: f32,
    gamma_gamma_adjust: f32,
    light_dir_x: f32,
    light_dir_y: f32,
    light_dir_z: f32,
    light_power: f32,
    agent_blend_mode: u32,
    agent_color_r: f32,
    agent_color_g: f32,
    agent_color_b: f32,
    agent_color_blend: f32,
    epoch: u32,
    vector_force_power: f32,
    vector_force_x: f32,
    vector_force_y: f32,
    inspector_zoom: f32,
    agent_trail_decay: f32,
    fluid_show: u32,
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

// Agent trail dye injection buffer (prepared each frame from trail_grid, then written by agents).
@group(0) @binding(7)
var<storage, read_write> trail_grid_inject: array<vec4<f32>>;

@group(0) @binding(8)
var<storage, read> fluid_velocity: array<vec2<f32>>; // 128x128 fluid velocity field for visualization (also used for dye when fluid_show enabled)

@group(0) @binding(9)
var<storage, read_write> new_agents: array<Agent>;  // Buffer for spawned agents

@group(0) @binding(10)
var<storage, read_write> spawn_debug_counters: array<atomic<u32>, 3>;  // [0]=spawn_counter, [1]=debug_counter, [2]=alive_counter

@group(0) @binding(11)
var<storage, read> spawn_requests: array<SpawnRequest>;

@group(0) @binding(12)
var<storage, read_write> selected_agent_buffer: array<Agent>;  // Buffer to hold the selected agent for CPU readback

@group(0) @binding(13)
var<storage, read_write> gamma_grid: array<f32>; // Terrain height field + slope components

@group(0) @binding(14)
// NOTE: Use vec4 for std430-friendly 16-byte stride to match host buffer layout
var<storage, read_write> trail_grid: array<vec4<f32>>; // Agent dye trail RGB + energy trail A (unclamped)

@group(0) @binding(15)
var<uniform> environment_init: EnvironmentInitParams;

@group(0) @binding(16)
var<storage, read_write> fluid_forces: array<vec2<f32>>; // Fluid forces buffer - agents write propeller forces directly with 100x boost

@group(0) @binding(17)
var<storage, read_write> agent_spatial_grid: array<atomic<u32>>; // Agent index per grid cell (atomic for vampire victim claiming)

// Spatial grid special markers
const SPATIAL_GRID_EMPTY: u32 = 0xFFFFFFFFu;     // No agent in this cell
const SPATIAL_GRID_CLAIMED: u32 = 0xFFFFFFFEu;   // Cell claimed by vampire (victim being drained)
const VAMPIRE_MOUTH_COOLDOWN: f32 = 60.0;         // Frames between drains (1 second at 60fps)
const VAMPIRE_NEWBORN_GRACE_FRAMES: u32 = 60u;    // Newborn agents ignore/are immune to vampire drain for 1 second

// Fluid constants
// NOTE: FLUID_GRID_SIZE is injected by Rust at shader compile time.
const FLUID_FORCE_SCALE: f32 = 5000.0;  // Multiplier for propeller forces injected into fluid (increased 10x for visibility)

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
    fluid_wind_coupling: f32,
}

// ============================================================================
// AMINO ACID & ORGAN PROPERTY LOOKUP TABLE (0–19 amino, 20–41 organs)
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
    0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, (1u<<9), (1u<<0), 0u, 0u, 0u, 0u, 0u, 0u, 0u, // 0–19
    (1u<<1), (1u<<0), (1u<<2), (1u<<3), (1u<<4), (1u<<8), (1u<<9), 0u, 0u, 0u, 0u, (1u<<7), 0u, (1u<<1), (1u<<5), (1u<<6), 0u, (1u<<11), (1u<<2), (1u<<2), (1u<<3), (1u<<3) // 20–41
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
    // Stored in AMINO_DATA[t][5].y (spare slot). If left as 0.0, default to 1.0.
    let raw_wind_coupling = d[5].y;
    p.fluid_wind_coupling = select(1.0, raw_wind_coupling, raw_wind_coupling != 0.0);

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

fn fluid_grid_index(pos: vec2<f32>) -> u32 {
    let clamped = clamp_position(pos);
    let scale = f32(SIM_SIZE) / f32(FLUID_GRID_SIZE);
    var x: i32 = i32(clamped.x / scale);
    var y: i32 = i32(clamped.y / scale);
    x = clamp(x, 0, i32(FLUID_GRID_SIZE) - 1);
    y = clamp(y, 0, i32(FLUID_GRID_SIZE) - 1);
    return u32(y) * FLUID_GRID_SIZE + u32(x);
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
fn sample_stochastic_gaussian(center: vec2<f32>, signal_gain: f32, seed: u32, grid_type: u32, debug_mode: bool, sensor_perpendicular: vec2<f32>, chirality_flip: f32, promoter_param1: f32, modifier_param1: f32) -> f32 {
    // Refactor: environment sensors now read from the env-resolution dye field.
    // Directional sensors sample only the immediate left/right env texels relative to the organ orientation.
    // The old "radius" parameter is repurposed as a signal gain multiplier.
    let combined_param = promoter_param1 + modifier_param1;
    // Sign from combined_param; magnitude from abs(combined_param).
    let sign_mult = select(1.0, -1.0, combined_param < 0.0);
    let gain = max(0.0, signal_gain) * abs(combined_param);

    // Keep seed in signature for compatibility (not used after refactor).
    let _seed = seed;

    // One env cell in world units.
    let cell = f32(SIM_SIZE) / f32(ENV_GRID_SIZE);
    // Chirality flips left/right by reversing the perpendicular direction.
    let chir = select(1.0, -1.0, chirality_flip < 0.0);
    let dir = select(vec2<f32>(1.0, 0.0), normalize(sensor_perpendicular), length(sensor_perpendicular) > 1e-5) * chir;

    let pos_left = center + dir * cell;
    let pos_right = center - dir * cell;

    var v_left = 0.0;
    var v_right = 0.0;
    if (is_in_bounds(pos_left)) {
        let idx = grid_index(pos_left);
        if (grid_type == 0u) { v_left = fluid_velocity[idx].y; }
        else if (grid_type == 1u) { v_left = fluid_velocity[idx].x; }
    }
    if (is_in_bounds(pos_right)) {
        let idx = grid_index(pos_right);
        if (grid_type == 0u) { v_right = fluid_velocity[idx].y; }
        else if (grid_type == 1u) { v_right = fluid_velocity[idx].x; }
    }

    if (debug_mode) {
        // Left sample = red, right sample = blue.
        draw_filled_circle(pos_left, 4.0, vec4<f32>(1.0, 0.0, 0.0, 0.7));
        draw_filled_circle(pos_right, 4.0, vec4<f32>(0.0, 0.0, 1.0, 0.7));
    }

    // Left/right with polarity-controlled sign.
    let diff = v_right - v_left;
    return diff * gain * sign_mult;
}

fn sample_magnitude_only(center: vec2<f32>, signal_gain: f32, seed: u32, grid_type: u32, debug_mode: bool, promoter_param1: f32, modifier_param1: f32) -> f32 {
    // Refactor: intensity sensors sample only the current env texel.
    // The old "radius" parameter is repurposed as a signal gain multiplier.
    let combined_param = promoter_param1 + modifier_param1;
    // Keep polarity explicit (requested): sign comes from (p+m), magnitude from abs(p+m).
    // Net multiplier = abs(p+m) * polarity == (p+m).
    let polarity = select(1.0, -1.0, combined_param < 0.0);
    let gain = max(0.0, signal_gain) * abs(combined_param);
    let _seed = seed;

    if (!is_in_bounds(center)) {
        return 0.0;
    }

    let idx = grid_index(center);
    var v = 0.0;
    if (grid_type == 0u) { v = fluid_velocity[idx].y; }
    else if (grid_type == 1u) { v = fluid_velocity[idx].x; }

    if (debug_mode) {
        draw_filled_circle(center, 4.0, vec4<f32>(1.0, 0.7, 0.0, 0.7));
    }

    return v * gain * polarity;
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
