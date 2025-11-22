// Ribossome - GPU-Accelerated Artificial Life Simulator
// Copyright (c) 2025 Filipe da Veiga Ventura Alves
// Licensed under MIT License

// Single unified shader for minimal GPU artificial life simulator
// Agents store genes, build bodies, sense environment grids, and draw themselves

// ============================================================================
// CONSTANTS
// ============================================================================

const GRID_SIZE: u32 = 2048u;          // Environment grid resolution (original)
const SIM_SIZE: u32 = 30720u;          // Simulation world size (original)
const MAX_BODY_PARTS: u32 = 64u;
const GENOME_BYTES: u32 = 128u;
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
const VELOCITY_BLEND: f32 = 0.4;      // 0..1, higher = quicker velocity changes
const ANGULAR_BLEND: f32 = 0.4;       // 0..1, higher = quicker rotation changes
const VEL_MAX: f32 = 12.0;             // Max linear speed per frame
const ANGVEL_MAX: f32 = 1.5;         // Max angular change (radians) per frame
// Signal-to-angle shaping (no dt): cap amplitude and per-frame change
const SIGNAL_GAIN: f32 =20;        // global scale for signal-driven angle (was 20.0)
// Separate gains for alpha vs beta to restore original triple-contribution tunability
const ANGLE_GAIN_ALPHA: f32 = 1.0;  // relative weighting for alpha term
const ANGLE_GAIN_BETA: f32 = 1.0;   // relative weighting for beta term
const MAX_SIGNAL_ANGLE: f32 = 2.4;    // hard cap on signal-induced angle (radians)
const MAX_SIGNAL_STEP: f32 = 0.8;    // max per-frame change due to signals (radians)
const PROP_TORQUE_COUPLING: f32 = 1; // 0=no spin from props, 1=full lever-arm torque

// ============================================================================
// STRUCTURES (std430 aligned)
// ============================================================================

struct BodyPart {
    pos: vec2<f32>,           // relative position from agent center
    size: f32,                // radius
    part_type: u32,           // encoded amino acid type
    alpha_signal: f32,        // alpha signal propagating through body
    beta_signal: f32,         // beta signal propagating through body
    _pad: vec2<f32>,
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
    rna_progress: u32,        // progress of RNA pairing (0..GENOME_WORDS)
    pairing_counter: u32,     // number of bases successfully paired (0..gene_length)
    is_selected: u32,         // 1 = selected for debug view, 0 = not selected
    generation: u32,          // lineage generation (0 = initial spawn)
    age: u32,                 // age in frames since spawn
    total_mass: f32,          // total mass computed after morphology
    genome: array<u32, GENOME_WORDS>,   // GENOME_BYTES bytes genome (ASCII RNA bases)
    genome_packed: array<u32, PACKED_GENOME_WORDS>, // GENOME_BYTES bases packed as 2 bits each (A/U/G/C -> 0/1/2/3)
    _pad_body_align: array<u32, 2>, // padding to align body to 16 bytes
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
    drag: f32,
    energy_cost: f32,
    amino_maintenance_cost: f32,
    spawn_probability: f32,
    death_probability: f32,
    grid_size: f32,
    camera_zoom: f32,
    camera_pan_x: f32,
    camera_pan_y: f32,
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
    trail_diffusion: f32,
    trail_decay: f32,
    trail_opacity: f32,
    trail_show: u32,
    // When 1, override per-amino left/right multipliers with isotropic interior diffusion
    interior_isotropic: u32,
    // When 1, ignore stop codons and translate entire genome to max body parts
    ignore_stop_codons: u32,
    // When 1, require AUG start codon before translation begins
    require_start_codon: u32,
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
}

// ============================================================================
// BINDINGS
// ============================================================================

@group(0) @binding(0)
var<storage, read> agents_in: array<Agent>;

@group(0) @binding(1)
var<storage, read_write> agents_out: array<Agent>;

@group(0) @binding(2)
var<storage, read_write> alpha_grid: array<f32>;  // 512x512 environment grid

@group(0) @binding(3)
var<storage, read_write> beta_grid: array<f32>;   // 512x512 environment grid

@group(0) @binding(4)
var<storage, read_write> visual_grid: array<vec4<f32>>; // RGBA render target

@group(0) @binding(5)
var<uniform> params: SimParams;

@group(0) @binding(6)
var<storage, read_write> alive_counter: atomic<u32>;

@group(0) @binding(7)
var<storage, read_write> debug_counter: atomic<u32>;

@group(0) @binding(8)
var<storage, read_write> new_agents: array<Agent>;  // Buffer for spawned agents

@group(0) @binding(9)
var<storage, read_write> spawn_counter: atomic<u32>;  // Count of spawned agents this frame

@group(0) @binding(10)
var<storage, read> spawn_requests: array<SpawnRequest>;

@group(0) @binding(11)
var<storage, read_write> selected_agent_buffer: array<Agent>;  // Buffer to hold the selected agent for CPU readback

@group(0) @binding(12)
var<storage, read_write> debug_parts_buffer: array<u32>; // [0]=count, [1..MAX_BODY_PARTS]=part types (u32 each)

@group(0) @binding(13)
var<storage, read_write> gamma_grid: array<f32>; // Terrain height field + slope components

@group(0) @binding(14)
// NOTE: Use vec4 for std430-friendly 16-byte stride to match host buffer layout
var<storage, read_write> trail_grid: array<vec4<f32>>; // Agent color trail RGBA (A unused)

@group(0) @binding(15)
var<uniform> environment_init: EnvironmentInitParams;

@group(0) @binding(16)
var<storage, read> alpha_rain_map: array<f32>;

@group(0) @binding(17)
var<storage, read> beta_rain_map: array<f32>;

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
    signal_decay: f32,
    alpha_left_mult: f32,
    alpha_right_mult: f32,
    beta_left_mult: f32,
    beta_right_mult: f32,
    is_displacer: bool,
    is_inhibitor: bool,
    is_condenser: bool,
}

// Returns per-amino-acid properties used to build and simulate body parts
fn get_amino_acid_properties(amino_type: u32) -> AminoAcidProperties {
    var props: AminoAcidProperties;
    // Default baseline to avoid undefined fields in cases that only override a subset
    props.segment_length = 8.0;
    props.thickness = 3.0;
    props.base_angle = 0.0;
    props.mass = 0.2;
    props.alpha_sensitivity = 0.0;
    props.beta_sensitivity = 0.0;
    props.is_propeller = false;
    props.thrust_force = 0.0;
    props.color = vec3<f32>(0.3, 0.3, 0.3);
    props.is_mouth = false;
    props.energy_absorption_rate = 0.0;
    props.beta_absorption_rate = 0.0;
    props.beta_damage = 0.0; // Will be set to random [-1, 1] per amino acid for color generation
    props.energy_storage = 0.0; // Default: no storage (only Mouth and Storage amino acids can store)
    props.energy_consumption = 0.0;
    props.is_alpha_sensor = false;
    props.is_beta_sensor = false;
    props.is_energy_sensor = false;
    props.signal_decay = 0.2;
    props.alpha_left_mult = 0.5;
    props.alpha_right_mult = 0.5;
    props.beta_left_mult = 0.5;
    props.beta_right_mult = 0.5;
    props.is_displacer = false;
    props.is_inhibitor = false;
    props.is_condenser = false;
    switch (amino_type) {
        case 0u: { // A - Alanine - Small, simple, common (real: CH3 side chain)
            props.segment_length = 8.5;
            props.thickness = 2.5;
            // Old CSV: Seed Angle = 20°
            props.base_angle = 0.349066; // 20 deg in radians
            props.alpha_sensitivity = -0.2;
            props.beta_sensitivity = 0.2;
            props.is_propeller = false;
            props.thrust_force = 0.0;
            props.color = vec3<f32>(0.3, 0.3, 0.3); // Dark grey
            props.is_mouth = false;
            props.energy_absorption_rate = 0.0;
            props.beta_absorption_rate = 0.3;
            props.beta_damage = -0.73; // Color value
            props.energy_storage = 0.0; // Regular amino - no storage
            props.energy_consumption = 0.001;
            props.is_alpha_sensor = false;
            props.is_beta_sensor = false;
            props.signal_decay = 0.2;
            props.alpha_left_mult = 0.8;
            props.alpha_right_mult = 0.2;
            props.beta_left_mult = 0.7;
            props.beta_right_mult = 0.3;
            props.mass = 0.015;
        }
    case 1u: { // C - Cysteine - BETA SENSOR - Small, polar (real: can form disulfide bonds)
            props.segment_length = 7.0;
            props.thickness = 2.5;
            // Old CSV: Seed Angle = 30°
            props.base_angle = 0.523599; // 30 deg in radians
            // CSV mapping: Beta sensor -> Alpha_sense 0°, Beta_sense +20°
            props.alpha_sensitivity = 0.0;
            props.beta_sensitivity = 0.349066; // 20 deg in radians
            props.is_propeller = false;
            props.thrust_force = 0.0;
            props.color = vec3<f32>(1.0, 0.0, 0.0); // Red (BETA SENSOR)
            props.is_mouth = false;
            props.energy_absorption_rate = 0.0;
            props.beta_absorption_rate = 0.3;
            props.beta_damage = 0.42; // Color value
            props.energy_storage = 0.0;
            // Reduced sensor energy consumption to near-zero to keep sensors viable
            props.energy_consumption = 0.0002;
            props.is_alpha_sensor = false;
            props.is_beta_sensor = true;
            props.signal_decay = 0.1;
            props.alpha_left_mult = 0.5;
            props.alpha_right_mult = 0.5;
            props.beta_left_mult = 0.5;
            props.beta_right_mult = 0.5;
            props.mass = 0.1;
        }
        case 2u: { // D - Aspartic acid - Small, charged (real: acidic, negatively charged)
            props.segment_length = 13.0;
            props.thickness = 3.0;
            // Old CSV: Seed Angle = 0°
            props.base_angle = 0.0;
            props.alpha_sensitivity = -0.2;
            props.beta_sensitivity = 0.3;
            props.is_propeller = false;
            props.thrust_force = 0.0;
            props.color = vec3<f32>(0.1, 0.1, 0.1); // Dark grey
            props.is_mouth = false;
            props.energy_absorption_rate = 0.0;
            props.beta_absorption_rate = 0.3;
            props.beta_damage = -0.91; // Color value
            props.energy_storage = 0.0;
            props.energy_consumption = 0.001;
            props.is_alpha_sensor = false;
            props.is_beta_sensor = false;
            props.signal_decay = 0.2;
            props.alpha_left_mult = -0.2;
            props.alpha_right_mult = 1.2;
            props.beta_left_mult = -0.3;
            props.beta_right_mult = 1.3;
            props.mass = 0.018;
        }
        case 3u: { // E - Glutamic acid - Medium, charged (real: acidic, longer than D)
            props.segment_length = 18.5;
            props.thickness = 3.0;
            // Old CSV: Seed Angle = 30°
            props.base_angle = 0.523599;
            props.alpha_sensitivity = 0.2;
            props.beta_sensitivity = -0.33;
            props.is_propeller = false;
            props.thrust_force = 0.0;
            props.color = vec3<f32>(0.2, 0.2, 0.2); // Dark grey
            props.is_mouth = false;
            props.energy_absorption_rate = 0.0;
            props.beta_absorption_rate = 0.3;
            props.beta_damage = 0.17; // Color value
            props.energy_storage = 0.0;
            props.energy_consumption = 0.001;
            props.is_alpha_sensor = false;
            props.is_beta_sensor = false;
            props.signal_decay = 0.2;
            props.alpha_left_mult = 1.4;
            props.alpha_right_mult = -0.4;
            props.beta_left_mult = 1.3;
            props.beta_right_mult = -0.3;
            props.mass = 0.01;
        }
        case 4u: { // F - Phenylalanine - POISON RESISTANT - Very heavy pink blob (cumulative: each F reduces poison damage by 10%)
            props.segment_length = 30.0;
            props.thickness = 30.0; // Very fat blob
            // Old CSV: Seed Angle = -60°
            props.base_angle = -1.047198;
            props.alpha_sensitivity = -0.5;
            props.beta_sensitivity = 0.12;
            props.is_propeller = false;
            props.thrust_force = 0.0;
            props.color = vec3<f32>(1.0, 0.4, 0.7); // Pink color
            props.is_mouth = false;
            props.energy_absorption_rate = 0.0;
            props.beta_absorption_rate = 0.3;
            props.beta_damage = -0.58; // Color value
            props.energy_storage = 0.0;
            props.energy_consumption = 0.001;
            props.is_alpha_sensor = false;
            props.is_beta_sensor = false;
            props.signal_decay = 0.2;
            props.alpha_left_mult = 0.6;
            props.alpha_right_mult = 0.4;
            props.beta_left_mult = 0.55;
            props.beta_right_mult = 0.45;
            props.mass = 10.0; // Very heavy - slows agent down significantly
        }
        case 5u: { // G - Glycine - BETA CONDENSER
            props.segment_length = 4.0;
            props.thickness = 0.75;
            // Old CSV: Seed Angle = -20°
            props.base_angle = -0.349066;
            props.alpha_sensitivity = 1.2;
            props.beta_sensitivity = 0.1;
            props.is_propeller = false;
            props.thrust_force = 0.0;
            props.color = vec3<f32>(0.4, 0.0, 0.0); // Dark red
            props.is_mouth = false;
            props.energy_absorption_rate = 0.0;
            props.beta_absorption_rate = 0.2;
            props.beta_damage = 0.88; // Color value
            props.energy_storage = 0.0;
            props.energy_consumption = 0.001;
            props.is_alpha_sensor = false;
            props.is_beta_sensor = false;
            props.signal_decay = 0.3;
            props.alpha_left_mult = 0.5;
            props.alpha_right_mult = 0.5;
            props.beta_left_mult = 0.5;
            props.beta_right_mult = 0.5;
            props.mass = 0.02;
            props.is_condenser = true;
        }
        case 6u: { // H - Histidine - Aromatic, charged (real: imidazole ring, pH-sensitive)
            props.segment_length = 9.0;
            props.thickness = 6.0;
            // Old CSV: Seed Angle = -10°
            props.base_angle = -0.174533;
            props.alpha_sensitivity = 0.2;
            props.beta_sensitivity = -0.61;
            props.is_propeller = false;
            props.thrust_force = 0.0;
            props.color = vec3<f32>(0.28, 0.28, 0.28); // Dark grey
            props.is_mouth = false;
            props.energy_absorption_rate = 0.0;
            props.beta_absorption_rate = 0.3;
            props.beta_damage = -0.35; // Color value
            props.energy_storage = 0.0;
            props.energy_consumption = 0.001;
            props.is_alpha_sensor = false;
            props.is_beta_sensor = false;
            props.signal_decay = 0.2;
            props.alpha_left_mult = 1.2;
            props.alpha_right_mult = -0.2;
            props.beta_left_mult = -0.3;
            props.beta_right_mult = 1.3;
            props.mass = 0.02;
        }
        case 7u: { // I - Isoleucine - Branched, hydrophobic (real: beta-branched aliphatic)
            props.segment_length = 19.0;
            props.thickness = 5.5;
            // Old CSV: Seed Angle = 30°
            props.base_angle = 0.523599;
            props.alpha_sensitivity = -0.3;
            props.beta_sensitivity = 0.9;
            props.is_propeller = false;
            props.thrust_force = 0.0;
            props.color = vec3<f32>(0.38, 0.38, 0.38); // Dark grey
            props.is_mouth = false;
            props.energy_absorption_rate = 0.0;
            props.beta_absorption_rate = 0.3;
            props.beta_damage = 0.61; // Color value
            props.energy_storage = 0.0;
            props.energy_consumption = 0.001;
            props.is_alpha_sensor = false;
            props.is_beta_sensor = false;
            props.signal_decay = 0.2;
            props.alpha_left_mult = 0.65;
            props.alpha_right_mult = 0.35;
            props.beta_left_mult = 0.7;
            props.beta_right_mult = 0.3;
            props.mass = 0.02;
        }
        case 8u: { // K - Lysine - MOUTH - Long, positively charged (real: long aliphatic + NH3+)
            props.segment_length = 1.0;
            props.thickness = 3.5;
            // Old CSV: Seed Angle = 5°
            props.base_angle = 0.0872665;
            props.alpha_sensitivity = 0.6;
            props.beta_sensitivity = -0.16;
            props.is_propeller = false;
            props.thrust_force = 0.0;
            props.color = vec3<f32>(1.0, 1.0, 0.0); // Yellow (MOUTH)
            props.is_mouth = true;
            props.energy_absorption_rate = 0.2;
            props.beta_absorption_rate = 0.2;
            props.beta_damage = -0.12; // Color value
            props.energy_storage = 10.0; // Reduced from 20.0
            props.energy_consumption = 0.001; // Reduced 10x (was 0.01)
            props.is_alpha_sensor = false;
            props.is_beta_sensor = false;
            props.signal_decay = 0.2;
            props.alpha_left_mult = 1.4;
            props.alpha_right_mult = -0.4;
            props.beta_left_mult = 1.3;
            props.beta_right_mult = -0.3;
            props.mass = 0.05;
        }
        case 9u: { // L - Leucine - CHIRAL FLIPPER - Flips angles of all following amino acids
            props.segment_length = 3.0; // Very short
            props.thickness = 10.0; // Very wide
            // Old CSV: Seed Angle = -10°
            props.base_angle = -0.174533;
            props.alpha_sensitivity = -0.3332;
            props.beta_sensitivity = 0.1;
            props.is_propeller = false;
            props.thrust_force = 0.0;
            props.color = vec3<f32>(1.0, 0.0, 1.0); // Cyan
            props.is_mouth = false;
            props.energy_absorption_rate = 0.0;
            props.beta_absorption_rate = 0.3;
            props.beta_damage = 0.95; // Color value
            props.energy_storage = 0.0;
            props.energy_consumption = 0.001;
            props.is_alpha_sensor = false;
            props.is_beta_sensor = false;
            props.signal_decay = 0.2;
            props.alpha_left_mult = -0.3;
            props.alpha_right_mult = 1.3;
            props.beta_left_mult = -0.2;
            props.beta_right_mult = 1.2;
            props.mass = 0.02;
        }
        case 10u: { // M - Methionine - START CODON - Medium, sulfur-containing (real: linear thioether)
            props.segment_length = 8.5;
            props.thickness = 4.0;
            // Old CSV: Seed Angle = -45°
            props.base_angle = -0.785398;
            props.alpha_sensitivity = 0.14;
            props.beta_sensitivity = -0.64;
            props.is_propeller = false;
            props.thrust_force = 0.0;
            props.color = vec3<f32>(0.8, 0.8, 0.2); // Pale yellow
            props.is_mouth = false;
            props.energy_absorption_rate = 0.0;
            props.beta_absorption_rate = 0.3;
            props.beta_damage = -0.48; // Color value
            props.energy_storage = 0.0;
            props.energy_consumption = 0.001;
            props.is_alpha_sensor = false;
            props.is_beta_sensor = false;
            props.signal_decay = 0.2;
            props.alpha_left_mult = 0.8;
            props.alpha_right_mult = 0.2;
            props.beta_left_mult = -0.1;
            props.beta_right_mult = 1.1;
            props.mass = 0.02;
        }
    case 11u: { // N - ENABLER (was INHIBITOR) - Increases power of nearby propellers, displacers, and mouths up to 40 units
            props.segment_length = 6.0;
            props.thickness = 6.0;
            // Old CSV: Seed Angle = 45°
            props.base_angle = 0.785398;
            props.alpha_sensitivity = 0.0;
            props.beta_sensitivity = 0.0;
            props.is_propeller = false;
            props.thrust_force = 0.0;
            props.color = vec3<f32>(1.0, 1.0, 1.0); // White
            props.is_mouth = false;
            props.energy_absorption_rate = 0.0;
            props.beta_absorption_rate = 0.0;
            props.beta_damage = 0.24; // Color value
            props.energy_storage = 0.0;
            props.energy_consumption = 0.001;
            props.is_alpha_sensor = false;
            props.is_beta_sensor = false;
            props.is_energy_sensor = false;
            props.signal_decay = 0.2;
            props.alpha_left_mult = 0.5;
            props.alpha_right_mult = 0.5;
            props.beta_left_mult = 0.5;
            props.beta_right_mult = 0.5;
            props.is_displacer = false;
            props.is_inhibitor = true; // reused flag; treated as enabler in simulation
            props.mass = 0.15;
        }
        case 12u: { // P - Proline - PROPELLER - Rigid cyclic (real: backbone constraint, helix breaker)
            props.segment_length = 16.0;
            props.thickness = 8.0;
            // Old CSV: Seed Angle = -30°
            props.base_angle = -0.523599;
            props.alpha_sensitivity = 0.0;
            props.beta_sensitivity = 0.0;
            props.is_propeller = true;
            props.thrust_force = 2.5; // Reduced by 4x (was 10.0)
            props.color = vec3<f32>(0.0, 0.0, 1); // Deep blue (PROPELLER)
            props.is_mouth = false;
            props.energy_absorption_rate = 0.0;
            props.beta_absorption_rate = 0.3;
            props.beta_damage = -0.77; // Color value
            props.energy_storage = 0.0;
            props.energy_consumption = 0.002; // Reduced 10x (was 0.05)
            props.is_alpha_sensor = false;
            props.is_beta_sensor = false;
            props.signal_decay = 0.2;
            props.alpha_left_mult = 1.5;
            props.alpha_right_mult = -0.5;
            props.beta_left_mult = 1.4;
            props.beta_right_mult = -0.4;
            props.mass = 0.05;
        }
        case 13u: { // Q - Glutamine - Medium, polar (real: amide side chain, similar to E)
            props.segment_length = 8.5;
            props.thickness = 3.0;
            // Old CSV: Seed Angle = -50°
            props.base_angle = -0.872665;
            props.alpha_sensitivity = 0.24;
            props.beta_sensitivity = -0.4;
            props.is_propeller = false;
            props.thrust_force = 0.0;
            props.color = vec3<f32>(0.34, 0.34, 0.34); // Dark grey
            props.is_mouth = false;
            props.energy_absorption_rate = 0.0;
            props.beta_absorption_rate = 0.3;
            props.beta_damage = 0.53; // Color value
            props.energy_storage = 0.0;
            props.energy_consumption = 0.001;
            props.is_alpha_sensor = false;
            props.is_beta_sensor = false;
            props.signal_decay = 0.2;
            props.alpha_left_mult = 1.0;
            props.alpha_right_mult = 0.0;
            props.beta_left_mult = 0.75;
            props.beta_right_mult = 0.25;
            props.mass = 0.02;
        }
        case 14u: { // R - Arginine - Very long, positively charged (real: longest, guanidinium group)
            props.segment_length = 18.5;
            props.thickness = 3.5;
            // Old CSV: Seed Angle = -10°
            props.base_angle = -0.174533;
            props.alpha_sensitivity = 0.5;
            props.beta_sensitivity = -0.15;
            props.is_propeller = false;
            props.thrust_force = 0.0;
            props.color = vec3<f32>(0.29, 0.29, 0.29); // Dark grey
            props.is_mouth = false;
            props.energy_absorption_rate = 0.0;
            props.beta_absorption_rate = 0.3;
            props.beta_damage = -0.29; // Color value
            props.energy_storage = 0.0;
            props.energy_consumption = 0.001;
            props.is_alpha_sensor = false;
            props.is_beta_sensor = false;
            props.signal_decay = 0.2;
            props.alpha_left_mult = -0.4;
            props.alpha_right_mult = 1.4;
            props.beta_left_mult = -0.3;
            props.beta_right_mult = 1.3;
            props.mass = 0.04;
        }
    case 15u: { // S - Serine - ALPHA SENSOR - Small, polar (real: hydroxyl group, similar to T)
            props.segment_length = 5.5;
            props.thickness = 2.5;
            // Old CSV: Seed Angle = -20°
            props.base_angle = -0.349066;
            // CSV mapping: Alpha sensor -> Alpha_sense -20°, Beta_sense 0°
            props.alpha_sensitivity = -0.349066; // -20 deg in radians
            props.beta_sensitivity = 0.0;
            props.is_propeller = false;
            props.thrust_force = 0.0;
            props.color = vec3<f32>(0.0, 1.0, 0.0); // Green (ALPHA SENSOR)
            props.is_mouth = false;
            props.energy_absorption_rate = 0.0;
            props.beta_absorption_rate = 0.2;
            props.beta_damage = 0.71; // Color value
            props.energy_storage = 0.0;
            // Reduced sensor energy consumption to near-zero
            props.energy_consumption = 0.0002;
            props.is_alpha_sensor = true;
            props.is_beta_sensor = false;
            props.signal_decay = 0.1;
            props.alpha_left_mult = 0.5;
            props.alpha_right_mult = 0.5;
            props.beta_left_mult = 0.5;
            props.beta_right_mult = 0.5;
            props.mass = 0.1;
        }
        case 16u: { // T - Threonine - ENERGY SENSOR - Small, polar (real: beta-branched hydroxyl)
            props.segment_length = 6.5;
            props.thickness = 3.5;
            // Old CSV: Seed Angle = 90°
            props.base_angle = 1.570796;
            props.alpha_sensitivity = 0.1;
            props.beta_sensitivity = -0.5;
            props.is_propeller = false;
            props.thrust_force = 0.0;
            props.color = vec3<f32>(0.6, 0.2, 0.8); // Purple (energy indicator)
            props.is_mouth = false;
            props.energy_absorption_rate = 0.0;
            props.beta_absorption_rate = 0.2;
            props.beta_damage = -0.66; // Color value
            props.energy_storage = 0.0;
            props.energy_consumption = 0.001;
            props.is_alpha_sensor = false;
            props.is_beta_sensor = false;
            props.is_energy_sensor = true;
            props.signal_decay = 0.2;
            props.alpha_left_mult = 0.9;
            props.alpha_right_mult = 0.1;
            props.beta_left_mult = 1.0;
            props.beta_right_mult = 0.0;
            props.mass = 0.1;
        }
        case 17u: { // V - Valine - DISPLACER - Cyan, short and fat, displaces environment
            props.segment_length = 6.0;
            props.thickness = 8.0;
            // Old CSV: Seed Angle = 0°
            props.base_angle = 0.0;
            props.alpha_sensitivity = -0.3;
            props.beta_sensitivity = 0.73;
            props.is_propeller = false;
            props.thrust_force = 0.0;
            props.color = vec3<f32>(0.0, 1.0, 1.0); // Cyan (DISPLACER)
            props.is_mouth = false;
            props.energy_absorption_rate = 0.0;
            props.beta_absorption_rate = 0.3;
            props.beta_damage = 0.36; // Color value
            props.energy_storage = 0.0;
            props.energy_consumption = 0.007; // Slightly higher energy cost
            props.is_alpha_sensor = false;
            props.is_beta_sensor = false;
            props.signal_decay = 0.2;
            props.alpha_left_mult = -0.3;
            props.alpha_right_mult = 1.3;
            props.beta_left_mult = 1.2;
            props.beta_right_mult = -0.2;
            props.mass = 0.15; // Heavier due to displacement mechanism
            props.is_displacer = true;
        }
        case 18u: { // W - Tryptophan - STORAGE - Largest, bulky aromatic (real: indole ring, massive)
            props.segment_length = 16.0;
            props.thickness = 22.0;             // Widest (was 10.0)
            // Old CSV: Seed Angle = 20°
            props.base_angle = 0.349066;
            props.alpha_sensitivity = 0.31;
            props.beta_sensitivity = -0.1;
            props.is_propeller = false;
            props.thrust_force = 0.0;
            props.color = vec3<f32>(1.0, 0.5, 0.0); // Orange (STORAGE)
            props.is_mouth = false;
            props.energy_absorption_rate = 0.0;
            props.beta_absorption_rate = 0.4;
            props.beta_damage = -0.84; // Color value
            props.energy_storage = 100.0;        // 5x mouth storage (was 12.0)
            props.energy_consumption = 0.001;   // Reduced 10x (was 0.005)
            props.is_alpha_sensor = false;
            props.is_beta_sensor = false;
            props.signal_decay = 0.15;
            props.alpha_left_mult = 0.55;
            props.alpha_right_mult = 0.45;
            props.beta_left_mult = 0.6;
            props.beta_right_mult = 0.4;
            props.mass = 1.3;
        }
        case 19u: { // Y - Tyrosine - ALPHA CONDENSER - Absorbs, stores, and discharges alpha signals
            props.segment_length = 11.5;
            props.thickness = 4.0;
            // Old CSV: Seed Angle = -30°
            props.base_angle = -0.523599;
            props.alpha_sensitivity = -0.2;
            props.beta_sensitivity = 0.52;
            props.is_propeller = false;
            props.thrust_force = 0.0;
            props.color = vec3<f32>(0.0, 0.4, 0.0); // Dark green for condenser
            props.is_mouth = false;
            props.energy_absorption_rate = 0.0;
            props.beta_absorption_rate = 0.3;
            props.beta_damage = 0.08; // Color value
            props.energy_storage = 0.0;
            props.energy_consumption = 0.001;
            props.is_alpha_sensor = false;
            props.is_beta_sensor = false;
            props.signal_decay = 0.2;
            props.alpha_left_mult = -0.5;
            props.alpha_right_mult = 1.5;
            props.beta_left_mult = -0.4;
            props.beta_right_mult = 1.4;
            props.mass = 0.04;
            props.is_condenser = true;
        }
        default: { // Fallback (should never happen)
            props.segment_length = 8.0;
            props.thickness = 3.0;
            props.base_angle = -0.7;
            props.alpha_sensitivity = 0.0;
            props.beta_sensitivity = 0.0;
            props.is_propeller = false;
            props.thrust_force = 0.0;
            props.color = vec3<f32>(0.5, 0.5, 0.5);
            props.is_mouth = false;
            props.energy_absorption_rate = 0.0;
            props.beta_absorption_rate = 0.0;
            props.beta_damage = 0.0;
            props.energy_storage = 0.0;
            props.energy_consumption = 0.0;
            props.is_alpha_sensor = false;
            props.is_beta_sensor = false;
            props.signal_decay = 0.0;
            props.alpha_left_mult = 0.5;
            props.alpha_right_mult = 0.5;
            props.beta_left_mult = 0.5;
            props.beta_right_mult = 0.5;
        }
    }
    
    

    return props;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Clamp position to world bounds (closed world)
fn clamp_position(pos: vec2<f32>) -> vec2<f32> {
    let ws = f32(SIM_SIZE);
    return vec2<f32>(
        clamp(pos.x, 0.0, ws),
        clamp(pos.y, 0.0, ws)
    );
}

fn grid_index(pos: vec2<f32>) -> u32 {
    // Clamp world pos into [0, SIM_SIZE]
    let clamped = clamp_position(pos);
    // Scale down from SIM_SIZE to GRID_SIZE for environment grids
    let scale = f32(SIM_SIZE) / f32(GRID_SIZE);
    var x: i32 = i32(clamped.x / scale);
    var y: i32 = i32(clamped.y / scale);
    x = clamp(x, 0, i32(GRID_SIZE) - 1);
    y = clamp(y, 0, i32(GRID_SIZE) - 1);
    return u32(y) * GRID_SIZE + u32(x);
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

fn read_gamma_height(idx: u32) -> f32 {
    return gamma_grid[idx];
}

fn write_gamma_height(idx: u32, value: f32) {
    // Gamma layer constrained to 0..1 range
    gamma_grid[idx] = clamp(value, 0.0, 1.0);
}

fn read_gamma_slope(idx: u32) -> vec2<f32> {
    let sx = gamma_grid[idx + GAMMA_SLOPE_X_OFFSET];
    let sy = gamma_grid[idx + GAMMA_SLOPE_Y_OFFSET];
    return vec2<f32>(sx, sy);
}

fn write_gamma_slope(idx: u32, slope: vec2<f32>) {
    gamma_grid[idx + GAMMA_SLOPE_X_OFFSET] = slope.x;
    gamma_grid[idx + GAMMA_SLOPE_Y_OFFSET] = slope.y;
}

fn read_combined_height(ix: i32, iy: i32) -> f32 {
    let idx = clamp_gamma_coords(ix, iy);
    var height = gamma_grid[idx];
    if (params.chemical_slope_scale_alpha != 0.0) {
        height += alpha_grid[idx] * params.chemical_slope_scale_alpha;
    }
    if (params.chemical_slope_scale_beta != 0.0) {
        height += beta_grid[idx] * params.chemical_slope_scale_beta;
    }
    return height;
}

fn hash(v: u32) -> u32 {
    var x = v;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

fn hash_f32(v: u32) -> f32 {
    return f32(hash(v)) / 4294967295.0;
}

// Perlin-like noise function
fn noise2d(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    
    // Smooth interpolation
    let u = f * f * (3.0 - 2.0 * f);
    
    // Hash corners
    let a = hash_f32(u32(i.x) + u32(i.y) * 57u);
    let b = hash_f32(u32(i.x + 1.0) + u32(i.y) * 57u);
    let c = hash_f32(u32(i.x) + u32(i.y + 1.0) * 57u);
    let d = hash_f32(u32(i.x + 1.0) + u32(i.y + 1.0) * 57u);
    
    // Bilinear interpolation
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn layered_noise(
    coord: vec2<f32>,
    seed: u32,
    octaves: u32,
    scale: f32,
    contrast: f32,
) -> f32 {
    let octave_count = max(octaves, 1u);
    var amplitude = 1.0;
    var frequency = 1.0;
    var sum = 0.0;
    var total = 0.0;
    var octave_seed = seed ^ 0x9E3779B1u;
    let safe_scale = max(scale, 0.0001);

    for (var i = 0u; i < octave_count; i = i + 1u) {
        let offset = vec2<f32>(
            hash_f32(octave_seed ^ 0xA511E9B5u) * 512.0,
            hash_f32(octave_seed ^ 0x63D3F6ABu) * 512.0,
        );
        sum = sum + noise2d(coord * frequency * safe_scale + offset) * amplitude;
        total = total + amplitude;
        amplitude = amplitude * 0.5;
        frequency = frequency * 2.0;
        octave_seed = hash(octave_seed ^ i);
    }

    let normalized = sum / max(total, 0.0001);
    return clamp((normalized - 0.5) * contrast + 0.5, 0.0, 1.0);
}

fn remap_unit(value: f32, range: vec2<f32>) -> f32 {
    return mix(range.x, range.y, value);
}

fn rotate_vec2(v: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec2<f32>(
        v.x * c - v.y * s,
        v.x * s + v.y * c
    );
}

// Wrapper that optionally bypasses global rotation controlled by DISABLE_GLOBAL_ROTATION
fn apply_agent_rotation(v: vec2<f32>, angle: f32) -> vec2<f32> {
    if (DISABLE_GLOBAL_ROTATION) { return v; }
    return rotate_vec2(v, angle);
}

// Convert world coordinates to screen pixel coordinates
fn world_to_screen(world_pos: vec2<f32>) -> vec2<i32> {
    // Calculate camera bounds with aspect ratio
    let safe_zoom = max(params.camera_zoom, 0.0001);
    let safe_width = max(params.window_width, 1.0);
    let safe_height = max(params.window_height, 1.0);
    let aspect_ratio = safe_width / safe_height;
    let view_width = params.grid_size / safe_zoom;
    let view_height = view_width / aspect_ratio;
    let cam_min_x = params.camera_pan_x - view_width * 0.5;
    let cam_min_y = params.camera_pan_y - view_height * 0.5;
    
    // Convert to normalized coordinates [0, 1]
    let norm_x = (world_pos.x - cam_min_x) / view_width;
    let norm_y = (world_pos.y - cam_min_y) / view_height;
    
    // Convert to screen pixels
    let screen_x = i32(norm_x * safe_width);
    let screen_y = i32(norm_y * safe_height);
    
    return vec2<i32>(screen_x, screen_y);
}

// Check if a world position is visible on screen (considering toroidal wrapping)
// Returns the wrapped position that should be drawn, or an off-screen position if not visible
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
    
    // Try the original position and all wrapped variants
    for (var wrap_x = -1; wrap_x <= 1; wrap_x++) {
        for (var wrap_y = -1; wrap_y <= 1; wrap_y++) {
            let test_pos = vec2<f32>(
                world_pos.x + f32(wrap_x) * ws,
                world_pos.y + f32(wrap_y) * ws
            );
            
            // Check if this wrapped position is in camera view
            if (test_pos.x >= cam_min_x - 50.0 && test_pos.x <= cam_max_x + 50.0 &&
                test_pos.y >= cam_min_y - 50.0 && test_pos.y <= cam_max_y + 50.0) {
                return test_pos;
            }
        }
    }
    
    // Not visible, return original position
    return world_pos;
}

// Convert screen pixel to visual grid index (window resolution)
fn screen_to_grid_index(screen_pos: vec2<i32>) -> u32 {
    // Direct mapping - visual grid now matches window size
    let x = u32(clamp(screen_pos.x, 0, i32(params.window_width) - 1));
    let y = u32(clamp(screen_pos.y, 0, i32(params.window_height) - 1));
    // Use padded visual stride for row indexing
    return y * params.visual_stride + x;
}

// RNA base-pair complement (A<->U, C<->G); padding 'X'(88) maps to itself
fn rna_complement(base: u32) -> u32 {
    // A(65) <-> U(85), C(67) <-> G(71)
    if (base == 88u) { return 88u; }     // padding stays padding
    if (base == 65u) { return 85u; }      // A -> U
    else if (base == 85u) { return 65u; } // U -> A
    else if (base == 67u) { return 71u; } // C -> G
    else if (base == 71u) { return 67u; } // G -> C
    else { return base; }                 // Unknown base, return as-is
}

// Get a random valid RNA base (A, U, G, or C)
fn get_random_rna_base(seed: u32) -> u32 {
    let choice = hash(seed) % 4u;
    if (choice == 0u) { return 65u; }      // A
    else if (choice == 1u) { return 85u; } // U
    else if (choice == 2u) { return 71u; } // G
    else { return 67u; }                   // C
}

// Map a codon (3 RNA bases) to amino acid index (0-19) using biological codon table
// YOUR amino acid indices (alphabetical): 
// 0=A(Ala), 1=C(Cys), 2=D(Asp), 3=E(Glu), 4=F(Phe), 5=G(Gly), 6=H(His), 7=I(Ile), 8=K(Lys-MOUTH), 9=L(Leu)
// 10=M(Met), 11=N(Asn), 12=P(Pro-PROPELLER), 13=Q(Gln), 14=R(Arg), 15=S(Ser-ALPHA), 16=T(Thr), 17=V(Val), 18=W(Trp-STORAGE), 19=Y(Tyr)
fn codon_to_amino_index(b0: u32, b1: u32, b2: u32) -> u32 {
    // U-starting codons
    if (b0 == 85u) {
        if (b1 == 85u) {
            // UUU, UUC -> Phe(4); UUA, UUG -> Leu(9)
            if (b2 == 85u || b2 == 67u) { return 4u; } // Phe
            return 9u; // Leu
        }
        if (b1 == 67u) { return 15u; } // UC* -> Ser(15)
        if (b1 == 65u) {
            // UAU, UAC -> Tyr(19); UAA, UAG are stop codons (handled elsewhere)
            if (b2 == 85u || b2 == 67u) { return 19u; } // Tyr
            return 19u; // Fallback Tyr for stop codons if reached
        }
        if (b1 == 71u) {
            // UGU, UGC -> Cys(1); UGA is stop; UGG -> Trp(18)
            if (b2 == 85u || b2 == 67u) { return 1u; } // Cys
            if (b2 == 71u) { return 18u; } // Trp
            return 1u; // Fallback Cys for UGA stop
        }
    }
    // C-starting codons
    if (b0 == 67u) {
        if (b1 == 85u) { return 9u; } // CU* -> Leu(9)
        if (b1 == 67u) { return 12u; } // CC* -> Pro(12)
        if (b1 == 65u) {
            // CAU, CAC -> His(6); CAA, CAG -> Gln(13)
            if (b2 == 85u || b2 == 67u) { return 6u; } // His
            return 13u; // Gln
        }
        if (b1 == 71u) { return 14u; } // CG* -> Arg(14)
    }
    // A-starting codons
    if (b0 == 65u) {
        if (b1 == 85u) {
            // AUU, AUC, AUA -> Ile(7); AUG -> Met(10)
            if (b2 == 71u) { return 10u; } // Met
            return 7u; // Ile
        }
        if (b1 == 67u) { return 16u; } // AC* -> Thr(16)
        if (b1 == 65u) {
            // AAU, AAC -> Asn(11); AAA, AAG -> Lys(8)
            if (b2 == 85u || b2 == 67u) { return 11u; } // Asn
            return 8u; // Lys
        }
        if (b1 == 71u) {
            // AGU, AGC -> Ser(15); AGA, AGG -> Arg(14)
            if (b2 == 85u || b2 == 67u) { return 15u; } // Ser
            return 14u; // Arg
        }
    }
    // G-starting codons
    if (b0 == 71u) {
        if (b1 == 85u) { return 17u; } // GU* -> Val(17)
        if (b1 == 67u) { return 0u; } // GC* -> Ala(0)
        if (b1 == 65u) {
            // GAU, GAC -> Asp(2); GAA, GAG -> Glu(3)
            if (b2 == 85u || b2 == 67u) { return 2u; } // Asp
            return 3u; // Glu
        }
        if (b1 == 71u) { return 5u; } // GG* -> Gly(5)
    }
    // Fallback (should not happen with valid RNA)
    return 0u; // Ala
}

// ============================================================================
// GENOME ACCESS HELPERS (ASCII/byte-based for transition)
// Centralize genome byte access so we can later switch to 2-bit packing.
// ============================================================================

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
        default:  { return genome[GENOME_WORDS - 1u]; }
    }
}

fn packed_read_word(packed: array<u32, PACKED_GENOME_WORDS>, index: u32) -> u32 {
    switch (index) {
        case 0u: { return packed[0u]; }
        case 1u: { return packed[1u]; }
        case 2u: { return packed[2u]; }
        case 3u: { return packed[3u]; }
        case 4u: { return packed[4u]; }
        case 5u: { return packed[5u]; }
        case 6u: { return packed[6u]; }
        case 7u: { return packed[7u]; }
        default: { return packed[PACKED_GENOME_WORDS - 1u]; }
    }
}

// Return single RNA base byte (A=65, U=85, G=71, C=67) at byte index [0..GENOME_LENGTH)
fn genome_get_base_ascii(genome: array<u32, GENOME_WORDS>, index: u32) -> u32 {
    if (index >= GENOME_LENGTH) { return 0u; }
    let w = index / 4u;
    let o = index % 4u;
    let word_val = genome_read_word(genome, w);
    return (word_val >> (o * 8u)) & 0xFFu;
}

// Return the 3-byte codon at byte index (no bounds wrap)
fn genome_get_codon_ascii(genome: array<u32, GENOME_WORDS>, index: u32) -> vec3<u32> {
    return vec3<u32>(
        genome_get_base_ascii(genome, index),
        genome_get_base_ascii(genome, index + 1u),
        genome_get_base_ascii(genome, index + 2u)
    );
}

// ============================================================================
// PACKED GENOME HELPERS (2-bit encoding)
// Encoding: A=0, U=1, G=2, C=3 packed 16 bases per u32 (LSB-first)
// ============================================================================

fn base_ascii_to_2bit(b: u32) -> u32 {
    // Map ASCII RNA bases to 2-bit values
    if (b == 65u) { return 0u; }  // A
    if (b == 85u) { return 1u; }  // U
    if (b == 71u) { return 2u; }  // G
    if (b == 67u) { return 3u; }  // C
    return 0u; // default A
}

fn base_2bit_to_ascii(v: u32) -> u32 {
    switch (v & 3u) {
        case 0u: { return 65u; } // A
        case 1u: { return 85u; } // U
        case 2u: { return 71u; } // G
        default: { return 67u; } // C
    }
}

// Get base from packed genome at [0..GENOME_LENGTH)
fn genome_get_base_packed(packed: array<u32, PACKED_GENOME_WORDS>, index: u32) -> u32 {
    if (index >= GENOME_LENGTH) { return 0u; }
    let word_index = index / PACKED_BASES_PER_WORD;         // 16 bases per word
    let bit_index = (index % PACKED_BASES_PER_WORD) * 2u;   // 2 bits per base
    let word_val = packed_read_word(packed, word_index);
    let two_bits = (word_val >> bit_index) & 0x3u;
    return base_2bit_to_ascii(two_bits);
}

// Pack the ASCII genome bytes into 2-bit representation
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

// Find the first start codon AUG (A=65, U=85, G=71)
// Searches every position in the genome, and once AUG is found,
// translation proceeds in 3-base strides from that position
// Returns 0xFFFFFFFF if no start codon exists
fn genome_find_start_codon(genome: array<u32, GENOME_WORDS>) -> u32 {
    // Search for AUG at every position (not just codon-aligned)
    for (var i = 0u; i < GENOME_LENGTH - 2u; i++) {
        let b0 = genome_get_base_ascii(genome, i);
        let b1 = genome_get_base_ascii(genome, i + 1u);
        let b2 = genome_get_base_ascii(genome, i + 2u);
        
        // Check for AUG start codon
        if (b0 == 65u && b1 == 85u && b2 == 71u) {
            return i;
        }
    }
    return 0xFFFFFFFFu;
}

// Find the first codon composed entirely of real bases (skips padding 'X')
// Returns 0xFFFFFFFF if no three-base codon exists
fn genome_find_first_coding_triplet(genome: array<u32, GENOME_WORDS>) -> u32 {
    var i = 0u;
    loop {
        if (i + 2u >= GENOME_LENGTH) { break; }
        let codon = genome_get_codon_ascii(genome, i);
        if (codon.x == 88u || codon.y == 88u || codon.z == 88u) {
            i = i + 1u;
            continue;
        }
        return i;
    }
    return 0xFFFFFFFFu;
}

// Check if the codon at index is a stop (UAA, UAG, UGA)
fn genome_is_stop_codon_at(genome: array<u32, GENOME_WORDS>, index: u32) -> bool {
    if (index + 2u >= GENOME_LENGTH) { return true; }
    let c = genome_get_codon_ascii(genome, index);
    // Padding 'X' (88) acts as a hard boundary -> stop
    if (c.x == 88u || c.y == 88u || c.z == 88u) { return true; }
    return (c.x == 85u && c.y == 65u && (c.z == 65u || c.z == 71u)) || (c.x == 85u && c.y == 71u && c.z == 65u);
}

// Build one 32-bit word of the reverse-complement genome for word index wi (0..GENOME_WORDS-1)
// Avoids dynamic indexing of local arrays by computing bytes directly
fn genome_revcomp_word(parent: array<u32, GENOME_WORDS>, wi: u32) -> u32 {
    // Output base indices [i0,i1,i2,i3] correspond to global base positions [wi*4+0..wi*4+3]
    // Reverse-complement maps: dst_idx -> src_idx = 63 - dst_idx
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
// UNIFIED AGENT KERNEL - Does everything in one pass
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
    // Genome scan only happens on first frame, but body rebuild happens every frame
    var body_count_val = agent.body_count;
    var start_byte = 0u;
    
    if (agent.body_count == 0u) {
        // First build: locate start position
        var start = 0xFFFFFFFFu;
        if (params.require_start_codon == 1u) {
            // Require AUG start codon
            start = genome_find_start_codon(agent.genome);
        } else {
            // Use first valid codon (skip padding)
            start = genome_find_first_coding_triplet(agent.genome);
        }
        
        if (start == 0xFFFFFFFFu) {
            // Non-viable genome (no start codon or no codons)
            agents_out[agent_id].alive = 0u;
            agents_out[agent_id].body_count = 0u;
            return;
        }
        start_byte = start;

    // Count codons starting from the first valid codon until stop codon (UAA, UAG, UGA) or limits
    var count = 0u;
    var pos_b = start_byte;
        for (var i = 0u; i < MAX_BODY_PARTS; i++) {
            if (pos_b + 2u >= GENOME_LENGTH) { break; }
            if (params.ignore_stop_codons == 0u && genome_is_stop_codon_at(agent.genome, pos_b)) { break; }
            count += 1u;
            pos_b += 3u;
        }
        
        // Natural selection: agents with bad genomes (no codons) die immediately
        count = clamp(count, 0u, MAX_BODY_PARTS);
        agents_out[agent_id].body_count = count;
        body_count_val = count;
        
        // Kill agents with no body parts
        if (count == 0u) {
            agents_out[agent_id].alive = 0u;
            return;
        }
    } else {
        // After first frame: find start position again for rebuilding
        var start = 0xFFFFFFFFu;
        if (params.require_start_codon == 1u) {
            start = genome_find_start_codon(agent.genome);
        } else {
            start = genome_find_first_coding_triplet(agent.genome);
        }
        if (start != 0xFFFFFFFFu) { start_byte = start; }
    }
    
    // ALWAYS rebuild body positions (enables dynamic shape changes from signals)
    if (body_count_val > 0u) {

        // Dynamic chain build - angles modulated by alpha/beta signals
    var current_pos = vec2<f32>(0.0);
    var current_angle = 0.0;
    var chirality_flip = 1.0; // Track cumulative chirality: 1.0 or -1.0
    
    // Accumulators for average angle calculation
    var sum_angle_mass = 0.0;
    var total_mass_angle = 0.0;

    // Build parts starting from the first valid codon
    var build_b = start_byte;
        var parts_built = 0u; // Track how many we actually build
        var total_capacity = 0.0; // Calculate energy capacity as we build
        for (var i = 0u; i < min(body_count_val, MAX_BODY_PARTS); i++) {
            if (build_b + 2u >= GENOME_LENGTH) { break; }
            let codon = genome_get_codon_ascii(agent.genome, build_b);
            let amino_type = codon_to_amino_index(codon.x, codon.y, codon.z);
            
            // Check if this amino acid is a chiral flipper (Leucine, 'L' = ASCII 76)
            let is_leucine = (codon.x == 76u || codon.y == 76u || codon.z == 76u);
            if (is_leucine) {
                chirality_flip = -chirality_flip; // Flip chirality for all following amino acids
            }
            
            let props = get_amino_acid_properties(amino_type);
            
            // Add this part's energy storage to total capacity
            total_capacity += props.energy_storage;
            
            // Read previous frame's signals for this part (preserved from physics phase)
            var alpha = 0.0;
            var beta = 0.0;
            if (body_count_val > 0u) {
                // Not first frame - read signals from previous frame
                alpha = agents_out[agent_id].body[i].alpha_signal;
                beta = agents_out[agent_id].body[i].beta_signal;
            }
            
            // Modulate angle based on signals and sensitivity
            // alpha_sensitivity controls how much alpha affects the angle
            // beta_sensitivity controls how much beta affects the angle
            // Raw signal effects with reduced gain
            let alpha_effect = alpha * props.alpha_sensitivity * SIGNAL_GAIN * ANGLE_GAIN_ALPHA;
            let beta_effect = beta * props.beta_sensitivity * SIGNAL_GAIN * ANGLE_GAIN_BETA;
            var target_signal_angle = alpha_effect + beta_effect;
            // Hard cap total contribution
            target_signal_angle = clamp(target_signal_angle, -MAX_SIGNAL_ANGLE, MAX_SIGNAL_ANGLE);

            // Progressive smoothing stored in _pad.x
            // No inertia: angle reacts instantly to signal
            var smoothed_signal = target_signal_angle;
            
            // Apply chirality flip to all angles (base_angle and signal modulation)
            current_angle += (props.base_angle + smoothed_signal) * chirality_flip;
            
            // Accumulate for average angle
            let m = max(props.mass, 0.01);
            sum_angle_mass += current_angle * m;
            total_mass_angle += m;

            current_pos.x += cos(current_angle) * props.segment_length;
            current_pos.y += sin(current_angle) * props.segment_length;
            agents_out[agent_id].body[i].pos = current_pos;
            var rendered_size = props.thickness * 0.5;
            let is_sensor = props.is_alpha_sensor || props.is_beta_sensor || props.is_energy_sensor;
            if (is_sensor) {
                rendered_size *= 2.0; // Sensors render larger for readability
            }
            if (props.is_condenser) {
                rendered_size *= 0.5; // Condensers render half-sized for contrast
            }
            agents_out[agent_id].body[i].size = rendered_size;
            agents_out[agent_id].body[i].part_type = amino_type;
            // Persist the smoothed angle contribution in _pad.x for next frame
            let keep_pad_y = agents_out[agent_id].body[i]._pad.y;
            agents_out[agent_id].body[i]._pad = vec2<f32>(smoothed_signal, keep_pad_y);
            build_b += 3u;
            parts_built += 1u;
        }
        
        // Update body_count to actual built count (may be less if we hit genome end)
        agents_out[agent_id].body_count = parts_built;
        body_count_val = parts_built;

        // Center of mass recentering (mass-weighted to reflect physical inertia)
        var com = vec2<f32>(0.0);
        let rec_n = body_count_val;
        if (rec_n > 0u) {
            var mass_sum = 0.0;
            for (var i = 0u; i < min(rec_n, MAX_BODY_PARTS); i++) {
                let part_type = agents_out[agent_id].body[i].part_type;
                let props = get_amino_acid_properties(part_type);
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

            // Removed angular re-orientation to avoid unstable whole-body rotations from small internal changes.
            // Agent's world rotation is driven only by physics (torque integration) ensuring temporal stability.
            
            // Calculate mass-weighted average angle of the body
            let avg_angle = sum_angle_mass / max(total_mass_angle, 0.0001);
            
            // Counteract internal rotation:
            // 1. Add average angle to global rotation (so the "visual average" stays put)
            if (!DISABLE_GLOBAL_ROTATION) {
                agents_out[agent_id].rotation += avg_angle;
            }
            
            // 2. Subtract average angle from all body parts (so they are centered around 0 locally)
            // We need to rotate the positions around the CoM (0,0) by -avg_angle
            let c_inv = cos(-avg_angle);
            let s_inv = sin(-avg_angle);
            
            for (var i = 0u; i < min(rec_n, MAX_BODY_PARTS); i++) {
                let p = agents_out[agent_id].body[i].pos;
                // Rotate p by -avg_angle
                agents_out[agent_id].body[i].pos = vec2<f32>(
                    p.x * c_inv - p.y * s_inv,
                    p.x * s_inv + p.y * c_inv
                );
            }
            
            // Also rotate the morphology origin
            let o = origin_local;
            agents_out[agent_id].morphology_origin = vec2<f32>(
                o.x * c_inv - o.y * s_inv,
                o.x * s_inv + o.y * c_inv
            );
        }
        
        // Energy capacity already calculated during build loop above
        agents_out[agent_id].energy_capacity = total_capacity;
    }

    let body_count = body_count_val; // Use computed value instead of reading from agent

    // ====== AMPLIFICATION CACHE (compute once) ======
    // For each part, sum contributions from all enablers (flag is_inhibitor reused) within 40 units.
    // Linear falloff: 1 at distance 0, 0 at >=40. Cap total at 1.0.
    var amplification_per_part: array<f32, MAX_BODY_PARTS>;
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let part_i = agents_out[agent_id].body[i];
        var amp = 0.0;
        for (var j = 0u; j < min(body_count, MAX_BODY_PARTS); j++) {
            let other = agents_out[agent_id].body[j];
            let other_props = get_amino_acid_properties(other.part_type);
            if (other_props.is_inhibitor) { // enabler role
                let d = length(part_i.pos - other.pos);
                if (d < 40.0) {
                    amp += max(0.0, 1.0 - d / 40.0);
                }
            }
        }
        amplification_per_part[i] = min(amp, 1.0);
    }
    
    // ====== SIGNAL PROPAGATION ======
    // Propagate alpha and beta signals through the amino acid chain
    // Based on reference: propagate from previous frame's signals, sensors ADD environment
    
    // Store old signals for propagation (from previous frame)
    var old_alpha: array<f32, MAX_BODY_PARTS>;
    var old_beta: array<f32, MAX_BODY_PARTS>;
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        old_alpha[i] = agents_out[agent_id].body[i].alpha_signal;
        old_beta[i] = agents_out[agent_id].body[i].beta_signal;
    }
    
    // Propagate signals through chain; either weighted (anisotropic) or isotropic depending on params
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let amino_props = get_amino_acid_properties(agents_out[agent_id].body[i].part_type);
        
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

        // Sensors: direct sampling with no accumulation or decay
        if (amino_props.is_alpha_sensor) {
            let rotated_pos = apply_agent_rotation(agents_out[agent_id].body[i].pos, agent.rotation);
            let world_pos = agent.position + rotated_pos;
            let sensed_value = alpha_grid[grid_index(world_pos)];
            new_alpha = sensed_value;
        }
        if (amino_props.is_beta_sensor) {
            let rotated_pos = apply_agent_rotation(agents_out[agent_id].body[i].pos, agent.rotation);
            let world_pos = agent.position + rotated_pos;
            let sensed_value = beta_grid[grid_index(world_pos)];
            new_beta = sensed_value;
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
        
        // CONDENSER: Absorbs signal into storage, then discharges at controlled rate
        // Uses _pad.y to store charge level with sign indicating mode:
        // NEGATIVE = charging mode (e.g., -5.0 = has 5.0 charge, still absorbing)
        // POSITIVE = discharging mode (e.g., +5.0 = has 5.0 charge, releasing)
        // Tyrosine (19u) = alpha condenser, Glycine (5u) = beta condenser
        if (amino_props.is_condenser) {
            let signed_charge = agents_out[agent_id].body[i]._pad.y; // Signed charge level
            let absorption_amount = 0.1; // Absorb 0.1 units per frame
            let max_charge = 10.0;
            let discharge_rate = 0.2; // Discharge 0.2 per frame
            
            let amino_type = agents_out[agent_id].body[i].part_type;
            let is_alpha_condenser = (amino_type == 19u); // Tyrosine
            let is_beta_condenser = (amino_type == 5u);   // Glycine
            
            if (signed_charge <= 0.0) {
                // CHARGING MODE (negative value)
                let charge = -signed_charge; // Get absolute charge value
                
                if (charge >= max_charge) {
                    // Reached full charge - switch to discharge mode (make positive)
                    agents_out[agent_id].body[i]._pad.y = max_charge;
                } else {
                    // Continue charging - absorb from signal
                    var absorbed = 0.0;
                    if (is_alpha_condenser && new_alpha > 0.0) {
                        absorbed = min(min(new_alpha, absorption_amount), max_charge - charge);
                        new_alpha -= absorbed;
                    } else if (is_beta_condenser && new_beta > 0.0) {
                        absorbed = min(min(new_beta, absorption_amount), max_charge - charge);
                        new_beta -= absorbed;
                    }
                    // Store as negative (charging mode)
                    agents_out[agent_id].body[i]._pad.y = -(charge + absorbed);
                }
            } else {
                // DISCHARGING MODE (positive value) or empty (0.0)
                let charge = signed_charge;
                
                if (charge <= 0.0) {
                    // Empty - restart charging with negative value
                    agents_out[agent_id].body[i]._pad.y = -1e-6;
                } else {
                    // Continue discharging
                    if (is_alpha_condenser) {
                        new_alpha += discharge_rate;
                    } else if (is_beta_condenser) {
                        new_beta += discharge_rate;
                    }
                    agents_out[agent_id].body[i]._pad.y = max(charge - discharge_rate, 0.0);
                }
            }
        }
        
        // Apply decay to non-sensor/non-condenser signals
        // Sensors are direct sources, condensers store independently
        if (!amino_props.is_alpha_sensor && !amino_props.is_condenser) { new_alpha *= 0.85; }
        if (!amino_props.is_beta_sensor && !amino_props.is_condenser) { new_beta *= 0.85; }
        
        // Clamp to -1.0 to 1.0 (allows inhibitory and excitatory signals)
        agents_out[agent_id].body[i].alpha_signal = clamp(new_alpha, -1.0, 1.0);
        agents_out[agent_id].body[i].beta_signal = clamp(new_beta, -1.0, 1.0);
    }
    
    // ====== PHYSICS CALCULATIONS ======
    // Calculate center of mass (CoM) and total mass AFTER morphology
    // Use segment midpoints for proper mass distribution
    var center_of_mass = vec2<f32>(0.0);
    var total_mass = 0.0;
    let morphology_origin = agents_out[agent_id].morphology_origin;
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let part = agents_out[agent_id].body[i];
        let part_props = get_amino_acid_properties(part.part_type);
        let mass = max(part_props.mass, 0.01);
        total_mass += mass;
        
        // Calculate segment midpoint for mass distribution
        var segment_start_chain = vec2<f32>(0.0);
        if (i > 0u) {
            segment_start_chain = agents_out[agent_id].body[i - 1u].pos;
        }
        let segment_midpoint_chain = (segment_start_chain + part.pos) * 0.5;
        let segment_midpoint = morphology_origin + segment_midpoint_chain;
        
        center_of_mass += segment_midpoint * mass;
    }
    total_mass = max(total_mass, 0.05); // Prevent division by zero
    center_of_mass /= total_mass;
    // Persist total_mass for inspector (will be overwritten each frame)
    agents_out[agent_id].total_mass = total_mass;
    
    // ====== AGENT COLOR CALCULATION ======
    // Calculate agent color from amino acid beta_damage values using sine waves
    // Sum beta_damage across all body parts
    var color_sum = 0.0;
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let part = agents_out[agent_id].body[i];
        let part_props = get_amino_acid_properties(part.part_type);
        color_sum += part_props.beta_damage;
    }
    // Apply sine waves with different multipliers to generate RGB channels
    let agent_color = vec3<f32>(
        sin(color_sum * 1.0) * 0.5 + 0.5,      // R: multiplier = 1.0
        sin(color_sum * 1.75) * 0.5 + 0.5,     // G: multiplier = 1.75
        sin(color_sum * 2.1215) * 0.5 + 0.5    // B: multiplier = 2.1215
    );
    
    let drag_coefficient = total_mass * 0.5;

    // Accumulate forces and torques (relative to CoM)
    var force = vec2<f32>(0.0);
    var torque = 0.0;
    
    // Now calculate forces using the updated morphology
    var chirality_flip_physics = 1.0; // Track cumulative chirality for propeller direction
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let part = agents_out[agent_id].body[i];
        
        // Get amino acid properties
        let amino_props = get_amino_acid_properties(part.part_type);
        
        // Check if this part is Leucine (index 9) and flip chirality
        if (part.part_type == 9u) {
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
        
        // Cached amplification for this part (organs will use it, others may ignore)
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

                            let gamma_transfer = min(center_gamma, transfer_amount);
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
                // Propeller strength scaled by amplification (enabler effect)
                let propeller_strength = amino_props.thrust_force * 3 * amplification; // 2x power
                let thrust_force = thrust_dir_world * propeller_strength;
                force += thrust_force;
                // Torque from lever arm r_com cross thrust (scaled down to reduce perpetual spinning)
                torque += (r_com.x * thrust_force.y - r_com.y * thrust_force.x) * (6.0 * PROP_TORQUE_COUPLING);
            }
        }

    // Displacer organ - throws environment to random neighboring pixel within radius 2
        // Only works if agent has enough energy to cover the displacer's consumption cost
        if (amino_props.is_displacer && agent.energy >= amino_props.energy_consumption) {
            let clamped_pos = clamp_position(world_pos);
            let grid_scale = f32(SIM_SIZE) / f32(GRID_SIZE);

            var gx = i32(clamped_pos.x / grid_scale);
            var gy = i32(clamped_pos.y / grid_scale);
            gx = clamp(gx, 0, i32(GRID_SIZE) - 1);
            gy = clamp(gy, 0, i32(GRID_SIZE) - 1);
            let center_idx = u32(gy) * GRID_SIZE + u32(gx);

            // Stronger than prop wash
            let disp_strength = max(params.prop_wash_strength * amplification * 3.0, 0.0);
            if (disp_strength > 0.0) {
                // Generate pseudo-random offset within radius 2 pixels using integer hashing to avoid directional bias
                let hashed_seed = hash(
                    u32(gx) * 73856093u ^
                    u32(gy) * 19349663u ^
                    (i + 1u) * 83492791u ^
                    params.random_seed
                );
                
                // Use integer modulo for perfectly uniform distribution in [-2, 2]
                // Previous round() method biased results towards -1, 0, 1
                let h1 = hash(hashed_seed);
                let h2 = hash(h1); // Chain hash for independence

                let offset_x = i32(h1 % 5u) - 2;
                let offset_y = i32(h2 % 5u) - 2;
                
                let target_gx = clamp(gx + offset_x, 0, i32(GRID_SIZE) - 1);
                let target_gy = clamp(gy + offset_y, 0, i32(GRID_SIZE) - 1);
                let target_idx = u32(target_gy) * GRID_SIZE + u32(target_gx);

                if (target_idx != center_idx) {
                    var center_gamma = read_gamma_height(center_idx);
                    var target_gamma = read_gamma_height(target_idx);
                    var center_alpha = alpha_grid[center_idx];
                    var target_alpha = alpha_grid[target_idx];
                    var center_beta = beta_grid[center_idx];
                    var target_beta = beta_grid[target_idx];

                    // Displacer transfer: move a proportional fraction of each layer present at the source.
                    // This makes the effect consistent across channels regardless of absolute quantities.
                    // Fraction scales with disp_strength and part mass share, capped for stability.
                    let transfer_fraction = clamp(disp_strength * 1 , 0.0, 0.5);

                    if (transfer_fraction > 0.0) {
                        // Capacities adjusted for 0..1 range
                        let alpha_capacity = max(0.0, 1.0 - target_alpha);
                        let beta_capacity = max(0.0, 1.0 - target_beta);
                        let gamma_capacity = max(0.0, 1.0 - target_gamma);

                        // Proportional transfer per channel
                        let gamma_transfer = min(center_gamma * transfer_fraction, gamma_capacity);
                        let alpha_transfer = min(center_alpha * transfer_fraction, alpha_capacity);
                        let beta_transfer = min(center_beta * transfer_fraction, beta_capacity);

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
    
    // Persist torque for inspector debugging
    agents_out[agent_id].torque_debug = torque;
    
    // Apply linear forces - overdamped regime (fluid dynamics at nanoscale)
    // In viscous fluids at low Reynolds number, velocity is directly proportional to force
    // No inertia: velocity = force / drag
    agent.velocity = force / drag_coefficient;
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
        let props = get_amino_acid_properties(part.part_type);
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

    // ====== TRAIL DEPOSITION ======
    // Deposit agent color to RGB trail at each body part position
    let trail_deposit_strength = 0.08; // Strength of trail deposition (0-1)
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let part = agents_out[agent_id].body[i];
        let rotated_pos = apply_agent_rotation(part.pos, agent.rotation);
        let world_pos = agent.position + rotated_pos;
        let idx = grid_index(world_pos);
        
    // Blend agent color with existing trail (not additive - replaces with weighted blend)
    let current_trail = trail_grid[idx].xyz;
    let blended = mix(current_trail, agent_color, trail_deposit_strength);
    trail_grid[idx] = vec4<f32>(clamp(blended, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    }

    // ====== FEEDING, ENERGY AND DEATH ======
    // Use the post-morphology capacity written into agents_out this frame
    let capacity = agents_out[agent_id].energy_capacity;
    // 1) Compute energy consumption from morphology (capacity already calculated in morphology phase)
    var energy_consumption = params.energy_cost; // base maintenance (can be 0)
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let part = agents_out[agent_id].body[i];
        let props = get_amino_acid_properties(part.part_type);
        let amp = select(0.0, amplification_per_part[i], (props.is_propeller || props.is_mouth || props.is_displacer));
        // Minimum baseline cost per amino acid (always paid)
        let baseline = params.amino_maintenance_cost;
        // Amplified organ extra cost (if organ); keep previous scaling
        let organ_extra = props.energy_consumption * amp * 1.5;
        energy_consumption += baseline + organ_extra;
    }

    // 2) Feeding: mouths pull from alpha grid and add to energy, consume beta and lose energy
    // Speed-dependent absorption: slower agents absorb more efficiently
    // Bite size is now independent of speed; keep fixed capture per frame
    
    // Count poison-resistant amino acids (F = Phenylalanine, amino index 4)
    // Each one reduces poison damage by 10%: 1 F -> 0.9x, 2 F -> 0.81x
    var poison_resistant_count = 0u;
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        if (agents_out[agent_id].body[i].part_type == 4u) { // Phenylalanine
            poison_resistant_count += 1u;
        }
    }
    // Each F reduces poison/radiation damage by 10%
    let poison_multiplier = pow(0.9, f32(poison_resistant_count));
    
    // Track total consumption for regurgitation
    var total_consumed_alpha = 0.0;
    var total_consumed_beta = 0.0;
    
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let part = agents_out[agent_id].body[i];
        let amino_props = get_amino_acid_properties(part.part_type);
        if (amino_props.is_mouth) {
            // Use cached amplification for mouth
            let amplification_mouth = amplification_per_part[i];
            
            let rotated_pos = apply_agent_rotation(part.pos, agent.rotation);
            let world_pos = agent.position + rotated_pos;
            let idx = grid_index(world_pos);
            
            // Consume alpha and beta based on per-amino absorption rates
            // and local availability, scaled by amplification and speed
            let available_alpha = alpha_grid[idx];
            let available_beta = beta_grid[idx];

            // Per-amino capture rates let us tune bite size vs. poison uptake
            let alpha_rate = max(amino_props.energy_absorption_rate, 0.0);
            let beta_rate  = max(amino_props.beta_absorption_rate, 0.0);

            // Total capture budget for this mouth this frame
            let rate_total = alpha_rate + beta_rate;
            if (rate_total > 0.0 && (available_alpha > 0.0 || available_beta > 0.0)) {
                let max_total = rate_total * amplification_mouth;

                // Weight consumption toward whichever is present and allowed by its rate
                let weighted_alpha = available_alpha * alpha_rate;
                let weighted_beta  = available_beta * beta_rate;
                let weighted_sum   = max(weighted_alpha + weighted_beta, 1e-6);
                let alpha_weight   = weighted_alpha / weighted_sum;
                let beta_weight    = 1.0 - alpha_weight;

                let consumed_alpha = min(available_alpha, max_total * alpha_weight);
                let consumed_beta  = min(available_beta,  max_total * beta_weight);

                // Apply alpha consumption
                if (consumed_alpha > 0.0) {
                    alpha_grid[idx] = clamp(available_alpha - consumed_alpha, 0.0, available_alpha);
                    agent.energy += consumed_alpha * params.food_power;
                    total_consumed_alpha += consumed_alpha;
                }

                // Apply beta consumption
                if (consumed_beta > 0.0) {
                    beta_grid[idx] = clamp(available_beta - consumed_beta, 0.0, available_beta);
                    agent.energy -= consumed_beta * params.poison_power * poison_multiplier;
                    total_consumed_beta += consumed_beta;
                }
            }
        }
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
        // Deposit remains: fixed decomposition amount (independent of pairing/energy)
        // Fixed total deposit = 1.0 (in 0..1 grid units), spread equally across parts
        if (body_count > 0u) {
            // Add both beta (toxin) and alpha (nutrient) to encourage sensor retention and recovery zones
            let total_beta_deposit = 1.0;
            let total_alpha_deposit = 0.3;
            let beta_per_part = total_beta_deposit / f32(body_count);
            let alpha_per_part = total_alpha_deposit / f32(body_count);
            for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
                let part = agents_out[agent_id].body[i];
                let rotated_pos = apply_agent_rotation(part.pos, agent.rotation);
                let world_pos = agent.position + rotated_pos;
                let idx = grid_index(world_pos);

                beta_grid[idx] = min(beta_grid[idx] + beta_per_part, 1.0);
                alpha_grid[idx] = min(alpha_grid[idx] + alpha_per_part, 1.0);
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
        // Poison resistance also protects against beta's pairing inhibition
        let effective_beta = beta_concentration * poison_multiplier;
        let radiation_factor = 1.0 / max(1.0 + effective_beta, 1.0);
        let seed = ((agent_id + 1u) * 747796405u) ^ (pairing_counter * 2891336453u) ^ (params.random_seed * 196613u) ^ pos_idx;
        let rnd = f32(hash(seed)) / 4294967295.0;
        let energy_for_pair = max(agent.energy, 0.0);
        
        // Probability to increment counter
        // Apply radiation_factor (beta acts as reproductive inhibitor)
        let pair_p = clamp(params.spawn_probability * (energy_for_pair + 1.0) * 0.1 * radiation_factor, 0.0, 1.0);
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
                let offspring_hash = (hash3 ^ (spawn_index * 0x9e3779b9u)) * 1664525u + 1013904223u;
                
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
                offspring.rna_progress = 0u;
                offspring.pairing_counter = 0u;
                offspring.is_selected = 0u;
                // Lineage and lifecycle
                offspring.generation = agents_out[agent_id].generation + 1u;
                offspring.age = 0u;
                offspring.total_mass = 0.0; // Will be computed after morphology build

                // Child genome: ALWAYS reverse complementary of parent (no generation condition)
                for (var w = 0u; w < GENOME_WORDS; w++) {
                    let rev_word = genome_revcomp_word(agents_out[agent_id].genome, w);
                    offspring.genome[w] = rev_word;
                }
                // Pack initial rev-comp genome
                {
                    let packed0 = genome_pack_into(offspring);
                    for (var i = 0u; i < PACKED_GENOME_WORDS; i++) {
                        let packed_word = packed_read_word(packed0, i);
                        offspring.genome_packed[i] = packed_word;
                    }
                }
                
                // Sample beta concentration at parent's location to calculate radiation-induced mutation rate
                let parent_idx = grid_index(agent.position);
                let beta_concentration = beta_grid[parent_idx];
                
                // Beta acts as mutagenic radiation - increases mutation rate with power-of-5 curve
                // This creates clear ecological zones: safe (beta 0-4), moderate (4-7), extreme (7-10)
                // At beta=0: 1x mutations, beta=5: ~2x, beta=7: ~6x, beta=10: ~11x
                // Beta grid is now in 0..1 range; normalize directly
                // Poison resistance also protects against beta's mutagenic effects
                let effective_beta_mutation = beta_concentration * poison_multiplier;
                let beta_normalized = clamp(effective_beta_mutation, 0.0, 1.0);
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
                    if (can_insert && insert_roll < (effective_mutation_rate * 0.2)) {
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
                    if (has_active && delete_roll < (effective_mutation_rate * 0.2)) {
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
                // Re-pack the genome after mutations so packed form stays consistent
                {
                    let packed1 = genome_pack_into(offspring);
                    for (var i = 0u; i < PACKED_GENOME_WORDS; i++) {
                        let packed_word = packed_read_word(packed1, i);
                        offspring.genome_packed[i] = packed_word;
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
                    offspring.body[bi]._pad = vec2<f32>(0.0);
                }

                new_agents[spawn_index] = offspring;
            }
        }
        // Reset pairing cycle after reproduction
        pairing_counter = 0u;
    }
    agents_out[agent_id].pairing_counter = pairing_counter;
    // Store the counter as progress for debug viewing
    agents_out[agent_id].rna_progress = pairing_counter;
    
    // ====== FRUSTUM CULLING & RENDERING ======
    // Skip rendering if no body parts (will die from energy loss naturally)
    if (agents_out[agent_id].body_count == 0u) {
        agents_out[agent_id].position = agent.position;
        agents_out[agent_id].velocity = agent.velocity;
        agents_out[agent_id].rotation = agent.rotation;
        agents_out[agent_id].energy = agent.energy;
        agents_out[agent_id].alive = agent.alive;
        // pairing_counter and rna_progress already written above
        return;
    }
    
    // Calculate camera bounds with aspect ratio
    let aspect_ratio = params.window_width / params.window_height;
    let camera_half_height = params.grid_size / (2.0 * params.camera_zoom);
    let camera_half_width = camera_half_height * aspect_ratio;
    let camera_center = vec2<f32>(params.camera_pan_x, params.camera_pan_y);
    let camera_min = camera_center - vec2<f32>(camera_half_width, camera_half_height);
    let camera_max = camera_center + vec2<f32>(camera_half_width, camera_half_height);
    
    // ====== RENDERING (only if draw is enabled) ======
    if (params.draw_enabled != 0u) {
        // Check if agent is visible and render it at its position only (no wrapping)
        let margin = 20.0; // Maximum body extent
        let center = agent.position;
        if (center.x + margin >= camera_min.x && center.x - margin <= camera_max.x &&
            center.y + margin >= camera_min.y && center.y - margin <= camera_max.y) {
            // Agent is visible - render it
            let in_debug_mode = params.debug_mode != 0u;

            // Get the morphology origin (where the chain starts in local space after CoM centering)
            let morphology_origin = agents_out[agent_id].morphology_origin;

            // Draw all body parts relative to this center position
            // For segments: draw from the START of each segment (previous endpoint or origin) to its END (current part pos)
            for (var i = 0u; i < min(agents_out[agent_id].body_count, MAX_BODY_PARTS); i++) {
                let part = agents_out[agent_id].body[i];
                let amino_props = get_amino_acid_properties(part.part_type);
                let rotated_pos = apply_agent_rotation(part.pos, agent.rotation);
                let world_pos = center + rotated_pos;

                // Determine segment start position
                var segment_start_world = center + apply_agent_rotation(morphology_origin, agent.rotation);
                if (i > 0u) {
                    // Multi-part: start from previous part's position
                    let prev_part = agents_out[agent_id].body[i - 1u];
                    let prev_rotated = apply_agent_rotation(prev_part.pos, agent.rotation);
                    segment_start_world = center + prev_rotated;
                }
                // For i == 0, segment_start_world uses stored morphology_origin (transformed chain origin)

                // Draw based on position in chain
                let is_first = i == 0u;
                let is_last = i == agents_out[agent_id].body_count - 1u;
                let is_single = agents_out[agent_id].body_count == 1u;
                
                // Unified rendering rule to avoid zero-length artifacts:
                // Draw a segment for EVERY part. For first part of a multi-part chain, segment starts at morphology_origin.
                // Overlay endpoint circles only for terminals (first/last) when body has >1 parts.
                if (!in_debug_mode) {
                    let thickness = part.size * 0.5;
                    draw_thick_line(segment_start_world, world_pos, thickness, vec4<f32>(amino_props.color, 1.0));
                    if (!is_single && (is_first || is_last)) {
                        draw_filled_circle(world_pos, thickness, vec4<f32>(amino_props.color, 1.0));
                    }
                }
                if (in_debug_mode) {
                    let a = agents_out[agent_id].body[i].alpha_signal;
                    let b = agents_out[agent_id].body[i].beta_signal;
                    let r = max(b, 0.0);
                    let g = max(a, 0.0);
                    let bl = max(max(-a, 0.0), max(-b, 0.0));
                    let dbg_color = vec4<f32>(r, g, bl, 1.0);
                    let thickness_dbg = max(part.size * 0.25, 0.5);
                    draw_thick_line(segment_start_world, world_pos, thickness_dbg, dbg_color);
                    if (!is_single && (is_first || is_last)) {
                        draw_filled_circle(world_pos, thickness_dbg, dbg_color);
                    }
                }
            
            // Special rendering for Leucine (chiral flipper) - draw as perpendicular segment
            if (part.part_type == 9u) {
                // Calculate segment direction
                var segment_dir = vec2<f32>(0.0);
                if (i > 0u) {
                    let prev = agents_out[agent_id].body[i-1u].pos;
                    segment_dir = part.pos - prev;
                } else if (agents_out[agent_id].body_count > 1u) {
                    let next = agents_out[agent_id].body[1u].pos;
                    segment_dir = next - part.pos;
                } else {
                    segment_dir = vec2<f32>(1.0, 0.0);
                }
                let seg_len = length(segment_dir);
                let axis_local = select(segment_dir / seg_len, vec2<f32>(1.0, 0.0), seg_len < 1e-4);
                // Perpendicular to segment axis
                let perp_local = vec2<f32>(-axis_local.y, axis_local.x);
                let perp_world = apply_agent_rotation(perp_local, agent.rotation);
                
                // Draw perpendicular line centered on the part position
                let half_length = part.size * 0.8; // Use the width (size) for perpendicular length
                let p1 = world_pos - perp_world * half_length;
                let p2 = world_pos + perp_world * half_length;
                let perp_thickness = part.size * 0.3; // Thinner than the normal segment
                draw_thick_line(p1, p2, perp_thickness, vec4<f32>(amino_props.color, 1.0));
            }
            
            // Special rendering for CONDENSER - filled circle with charge level
            // Tyrosine (alpha) = green fill, Glycine (beta) = red fill, white outline
            // FLASH: White when discharging (positive charge value)
            if (amino_props.is_condenser) {
                let signed_charge = part._pad.y; // Signed charge: negative=charging, positive=discharging
                let charge = abs(signed_charge); // Absolute charge level (0.0 to 10.0)
                let max_charge = 10.0;
                let is_discharging = (signed_charge > 0.0); // Positive = discharging
                let charge_ratio = clamp(charge / max_charge, 0.0, 1.0); // 0.0 to 1.0
                
                let min_radius = 6.0;
                let radius = max(part.size * 0.75, min_radius);
                let segments = 24u;
                
                let amino_type = part.part_type;
                let is_alpha_condenser = (amino_type == 19u); // Tyrosine - green
                let is_beta_condenser = (amino_type == 5u);   // Glycine - red
                
                // Fill color based on charge level: dark -> bright as it charges
                // FLASH WHITE when discharging. Both condensers share the exact same shading; only tint differs.
                var fill_color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
                if (is_discharging) {
                    // Flash white when discharging
                    fill_color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
                } else {
                    // Base tint per condenser type
                    var base_tint = vec3<f32>(0.0, 1.0, 0.0); // Alpha condenser = green
                    if (is_beta_condenser) {
                        base_tint = vec3<f32>(1.0, 0.0, 0.0); // Beta condenser = red
                    }
                    let low_tint = base_tint * 0.25; // Keep color identity even when uncharged
                    let fill_rgb = mix(low_tint, base_tint, charge_ratio);
                    fill_color = vec4<f32>(fill_rgb, 1.0);
                }
                
                // Draw filled circle by drawing many lines from center to edge
                let fill_segments = 32u;
                for (var s = 0u; s < fill_segments; s++) {
                    let ang1 = f32(s) / f32(fill_segments) * 6.28318530718;
                    let ang2 = f32(s + 1u) / f32(fill_segments) * 6.28318530718;
                    let p1 = world_pos + vec2<f32>(cos(ang1) * radius, sin(ang1) * radius);
                    let p2 = world_pos + vec2<f32>(cos(ang2) * radius, sin(ang2) * radius);
                    draw_thick_line(world_pos, p1, radius * 0.5, fill_color);
                    draw_thick_line(p1, p2, 1.0, fill_color);
                }
                
                // Draw white outline
                var prev = world_pos + vec2<f32>(radius, 0.0);
                for (var s = 1u; s <= segments; s++) {
                    let t = f32(s) / f32(segments);
                    let ang = t * 6.28318530718;
                    let p = world_pos + vec2<f32>(cos(ang) * radius, sin(ang) * radius);
                    draw_thick_line(prev, p, 0.5, vec4<f32>(1.0, 1.0, 1.0, 1.0));
                    prev = p;
                }
            }
            
            // Draw ultra-subtle circular outline for ENABLER amino acids (was inhibitor)
            // Fade rules: fully visible at zoom >= 15, starts fading below 15, fully transparent at zoom <= 5
            // Only visible in debug mode
            if (amino_props.is_inhibitor && params.camera_zoom > 5.0 && params.debug_mode > 0u) {
                let radius = 20.0;
                let segments = 32u;
                // Fade alpha linearly from zoom 5 -> 15: 0 at 5, 1 at 15, clamp outside
                let zoom = params.camera_zoom;
                let alpha_base = 0.02;
                let fade = clamp((zoom - 5.0) / 10.0, 0.0, 1.0);
                let alpha = alpha_base * fade;
                let color = vec4<f32>(0.15, 0.2, 0.15, alpha);
                var prev = world_pos + vec2<f32>(radius,0.0);
                for (var s = 1u; s <= segments; s++) {
                    let t = f32(s) / f32(segments);
                    let ang = t * 6.28318530718;
                    let p = world_pos + vec2<f32>(cos(ang)*radius, sin(ang)*radius);
                    draw_thick_line(prev, p, 0.25, color); // ultra-thin outline
                    prev = p;
                }
                // Small white center marker to indicate exact enabler position
                draw_filled_circle(world_pos, 2.0, vec4<f32>(1.0, 1.0, 1.0, 0.95));
            }

            if (PROPELLERS_ENABLED && amino_props.is_propeller && agent.energy > 0.0 && params.camera_zoom > 2.0) {
                // Use cached amplification for jet visuals
                let jet_amplification = amplification_per_part[i];
                // Match thrust visual direction to physics: use segment axis perpendicular
                var segment_dir = vec2<f32>(0.0);
                if (i > 0u) {
                    let prev = agents_out[agent_id].body[i-1u].pos;
                    segment_dir = part.pos - prev;
                } else if (agents_out[agent_id].body_count > 1u) {
                    let next = agents_out[agent_id].body[1u].pos;
                    segment_dir = next - part.pos;
                } else {
                    segment_dir = vec2<f32>(1.0, 0.0);
                }
                let seg_len = length(segment_dir);
                let axis_local = select(segment_dir / seg_len, vec2<f32>(1.0, 0.0), seg_len < 1e-4);
                let axis_world = apply_agent_rotation(axis_local, agent.rotation);
                let jet_dir = normalize(vec2<f32>(-axis_world.y, axis_world.x));
                // Exhaust particles point opposite to thrust direction
                let exhaust_dir = -jet_dir;
                let propeller_strength = part.size * 2.5 * jet_amplification; // Visual scale with amplification
                let zoom_factor = clamp((params.camera_zoom - 2.0) / 8.0, 0.0, 1.0);
                let jet_length = propeller_strength * mix(0.6, 1.2, zoom_factor);
                let jet_seed = agent_id * 1000u + i * 17u;
                let particle_count = 1u + u32(round(jet_amplification * 5.0)) + u32(round(zoom_factor * 3.0));
                draw_particle_jet(world_pos, exhaust_dir, jet_length, jet_seed, particle_count);
            }
            
            // Draw cloud-like sensors
            if (amino_props.is_alpha_sensor || amino_props.is_beta_sensor || amino_props.is_energy_sensor) {
                let sensor_radius = part.size * 2.0;
                let sensor_seed = agent_id * 500u + i * 13u;
                let sensor_color = vec4<f32>(amino_props.color * 0.6, 0.5); // Semi-transparent
                draw_cloud(world_pos, sensor_radius, sensor_color, sensor_seed);
            }

            if (in_debug_mode) {
                let a = agents_out[agent_id].body[i].alpha_signal;
                let b = agents_out[agent_id].body[i].beta_signal;
                let r = max(b, 0.0);
                let g = max(a, 0.0);
                let bl = max(max(-a, 0.0), max(-b, 0.0));
                let dbg_color = vec4<f32>(r, g, bl, 1.0);
                draw_filled_circle(world_pos, 1.5, dbg_color);
            }
            
            // Draw yellow asterisk (*) on mouth parts (larger for visibility)
            if (amino_props.is_mouth && !in_debug_mode) {
                draw_asterisk(world_pos, part.size * 2.5, vec4<f32>(1.0, 1.0, 0.0, 1.0));
            }
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
        }
        
            // Debug: count visible agents
            atomicAdd(&debug_counter, 1u);
            
            // Draw selection circle if this agent is selected
            if (agent.is_selected == 1u) {
                draw_selection_circle(agent.position, agent_id, body_count);
            }
        }
    } // End of draw_enabled check

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
// HELPER FUNCTIONS FOR DRAWING
// ============================================================================

// Helper function to draw a thick line in screen space
fn draw_thick_line(p0: vec2<f32>, p1: vec2<f32>, thickness: f32, color: vec4<f32>) {
    let screen_p0 = world_to_screen(p0);
    let screen_p1 = world_to_screen(p1);
    
    let dx = screen_p1.x - screen_p0.x;
    let dy = screen_p1.y - screen_p0.y;
    let steps = max(abs(dx), abs(dy));

    let world_to_screen_scale = params.window_width / (params.grid_size / params.camera_zoom);
    let screen_thickness = i32(thickness * world_to_screen_scale);
    
    for (var s = 0; s <= steps; s++) {
        let t = f32(s) / f32(max(steps, 1));
        let screen_x = i32(mix(f32(screen_p0.x), f32(screen_p1.x), t));
        let screen_y = i32(mix(f32(screen_p0.y), f32(screen_p1.y), t));
        
        // Draw thicker line by filling a small circle around each point
        for (var dy = -screen_thickness; dy <= screen_thickness; dy++) {
            for (var dx = -screen_thickness; dx <= screen_thickness; dx++) {
                if (dx * dx + dy * dy <= screen_thickness * screen_thickness) {
                    let screen_pos = vec2<i32>(screen_x + dx, screen_y + dy);
                    
                    // Check if in screen bounds
                    if (screen_pos.x >= 0 && screen_pos.x < i32(params.window_width) &&
                        screen_pos.y >= 0 && screen_pos.y < i32(params.window_height)) {
                        
                        let idx = screen_to_grid_index(screen_pos);
                        visual_grid[idx] = color;
                    }
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
            
            // Draw only the outline (distance close to radius)
            if (abs(dist - screen_radius) < line_thickness) {
                let screen_pos = screen_center + vec2<i32>(dx, dy);
                
                // Check if in screen bounds
                if (screen_pos.x >= 0 && screen_pos.x < i32(params.window_width) &&
                    screen_pos.y >= 0 && screen_pos.y < i32(params.window_height)) {
                    
                    let idx = screen_to_grid_index(screen_pos);
                    visual_grid[idx] = color;
                }
            }
        }
    }
}

// Helper: draw a filled circle in screen space
fn draw_filled_circle(center: vec2<f32>, radius: f32, color: vec4<f32>) {
    // Convert world position to screen coordinates
    let screen_center = world_to_screen(center);
    
    // Calculate screen-space radius (accounting for zoom)
    let world_to_screen_scale = params.window_width / (params.grid_size / params.camera_zoom);
    let screen_radius = radius * world_to_screen_scale;

    let radius_i = i32(ceil(screen_radius));
    
    for (var dy = -radius_i; dy <= radius_i; dy++) {
        for (var dx = -radius_i; dx <= radius_i; dx++) {
            let offset = vec2<f32>(f32(dx), f32(dy));
            let dist2 = dot(offset, offset);
            if (dist2 <= screen_radius * screen_radius) {
                let screen_pos = screen_center + vec2<i32>(dx, dy);
                // Check if in screen bounds
                if (screen_pos.x >= 0 && screen_pos.x < i32(params.window_width) &&
                    screen_pos.y >= 0 && screen_pos.y < i32(params.window_height)) {
                    let idx = screen_to_grid_index(screen_pos);
                    visual_grid[idx] = color;
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

// Helper: draw an asterisk (*) with 6 crossing lines (vertical, horizontal, 2 diagonals)
fn draw_asterisk(center: vec2<f32>, radius: f32, color: vec4<f32>) {
    let screen_center = world_to_screen(center);
    let world_to_screen_scale = params.window_width / (params.grid_size / params.camera_zoom);
    let r = radius * world_to_screen_scale;
    let rx = i32(r);
    let ry = i32(r);
    // Endpoints for lines
    let up    = vec2<i32>(screen_center.x, screen_center.y - ry);
    let down  = vec2<i32>(screen_center.x, screen_center.y + ry);
    let left  = vec2<i32>(screen_center.x - rx, screen_center.y);
    let right = vec2<i32>(screen_center.x + rx, screen_center.y);
    let diag = r * 0.70710678; // r / sqrt(2)
    let tl = vec2<i32>(screen_center.x - i32(diag), screen_center.y - i32(diag));
    let br = vec2<i32>(screen_center.x + i32(diag), screen_center.y + i32(diag));
    let tr = vec2<i32>(screen_center.x + i32(diag), screen_center.y - i32(diag));
    let bl = vec2<i32>(screen_center.x - i32(diag), screen_center.y + i32(diag));
    draw_line_pixels(up, down, color);
    draw_line_pixels(left, right, color);
    draw_line_pixels(tl, br, color);
    draw_line_pixels(tr, bl, color);
}

// Helper: draw a cloud-like shape (fuzzy circle with some random bumps)
fn draw_cloud(center: vec2<f32>, radius: f32, color: vec4<f32>, seed: u32) {
    // Draw multiple overlapping circles to create a fluffy cloud appearance
    let num_puffs = 8u;
    for (var i = 0u; i < num_puffs; i++) {
        let angle = f32(i) * 6.28318530718 / f32(num_puffs);
        let hash_val = hash_f32(seed * (i + 1u) * 2654435761u);
        let offset_dist = radius * 0.4 * hash_val;
        let puff_center = center + vec2<f32>(cos(angle) * offset_dist, sin(angle) * offset_dist);
        let puff_radius = radius * (0.5 + 0.3 * hash_val);
        draw_filled_circle(puff_center, puff_radius, color);
    }
    // Draw a larger central circle
    draw_filled_circle(center, radius * 0.7, color);
}

// Helper: draw a particle jet (motion-blurred particles in a cone)
fn draw_particle_jet(origin: vec2<f32>, direction: vec2<f32>, length: f32, seed: u32, particle_count: u32) {
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
        draw_thick_line(particle_pos, streak_end, streak_thickness, fade_color);
    }
}

// Helper: draw a line between two screen pixel positions
fn draw_line_pixels(p0: vec2<i32>, p1: vec2<i32>, color: vec4<f32>) {
    let dx = p1.x - p0.x;
    let dy = p1.y - p0.y;
    let steps = max(abs(dx), abs(dy));
    
    for (var s = 0; s <= steps; s++) {
        let t = f32(s) / f32(max(steps, 1));
        let screen_x = i32(mix(f32(p0.x), f32(p1.x), t));
        let screen_y = i32(mix(f32(p0.y), f32(p1.y), t));
        
        // Check if in screen bounds
        if (screen_x >= 0 && screen_x < i32(params.window_width) &&
            screen_y >= 0 && screen_y < i32(params.window_height)) {
            let idx = screen_to_grid_index(vec2<i32>(screen_x, screen_y));
            visual_grid[idx] = color;
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
        
        // Check if in screen bounds
        if (screen_pos.x >= 0 && screen_pos.x < i32(params.window_width) &&
            screen_pos.y >= 0 && screen_pos.y < i32(params.window_height)) {
            
            let idx = screen_to_grid_index(screen_pos);
            visual_grid[idx] = color;
        }
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
    
    let radius = max_dist + 5.0; // Add some padding
    let num_segments = 64u; // Circle segments
    let color = vec4<f32>(1.0, 1.0, 0.0, 1.0); // Yellow circle
    
    // Draw circle as line segments
    for (var i = 0u; i < num_segments; i++) {
        let angle1 = f32(i) / f32(num_segments) * 6.28318530718;
        let angle2 = f32(i + 1u) / f32(num_segments) * 6.28318530718;
        
        let p1 = center_pos + vec2<f32>(cos(angle1) * radius, sin(angle1) * radius);
        let p2 = center_pos + vec2<f32>(cos(angle2) * radius, sin(angle2) * radius);
        
        draw_thick_line(p1, p2, 2.0, color);
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
    
    // Stochastic rain - randomly add food/poison droplets instead of multipliers
    // Use position and random seed to generate unique random values per cell
    let cell_seed = idx * 2654435761u + params.random_seed;
    let rain_chance = f32(hash(cell_seed)) / 4294967295.0;
    // Uniform alpha rain (food): remove spatial and beta-dependent gradients.
    // Each cell independently receives a saturated rain event with probability alpha_multiplier * 0.05.
    // (Scaling by 0.05 preserves prior expected value semantics.)
    let alpha_rain_factor = clamp(alpha_rain_map[idx], 0.0, 1.0);
    let alpha_probability_sat = params.alpha_multiplier * 0.05 * alpha_rain_factor;
    if (rain_chance < alpha_probability_sat) {
        final_alpha = 1.0;
    }

    // Uniform beta rain (poison): also no vertical gradient. Probability = beta_multiplier * 0.05.
    let beta_seed = cell_seed * 1103515245u;
    let beta_rain_chance = f32(hash(beta_seed)) / 4294967295.0;
    let beta_rain_factor = clamp(beta_rain_map[idx], 0.0, 1.0);
    let beta_probability_sat = params.beta_multiplier * 0.05 * beta_rain_factor;
    if (beta_rain_chance < beta_probability_sat) {
        final_beta = 1.0;
    }
    
    // Apply the blurred + rain values
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
    
    // Diagonal neighbors (distance = sqrt(2) ≈ 1.414)
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
    let gradient = vec2<f32>(dx, dy) * inv_cell_size;
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
    trail_grid[idx] = vec4<f32>(faded, 1.0);
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
    
    // Sample environment grids with direct indexing
    let alpha = alpha_grid[grid_index(world_pos)];
    let beta = beta_grid[grid_index(world_pos)];
    let gamma = read_gamma_height(grid_index(world_pos));
    
    // Hide gamma if requested (treat gamma exactly like alpha/beta, no normalization)
    var gamma_display = gamma;
    if (params.gamma_hidden != 0u) {
        gamma_display = 0.0;
    } else {
        let vis_range = max(params.gamma_vis_max - params.gamma_vis_min, 0.0001);
        gamma_display = clamp((gamma - params.gamma_vis_min) / vis_range, 0.0, 1.0);
    }

    // Compute base color for environment visualization without early returns,
    // so we can always overlay the trail layer afterward.
    var base_color = vec3<f32>(0.0);

    // Slope visualization with optional lighting
    if (params.slope_debug != 0u) {
        let slope = read_gamma_slope(grid_index(world_pos));
        if (params.slope_lighting != 0u) {
            // Lighting mode: compute normal and shade with directional light
            let normal = normalize(vec3<f32>(-slope.x * 10.0, -slope.y * 10.0, 1.0));
            let light_dir = normalize(vec3<f32>(0.5, 0.5, 0.5));
            let diffuse = max(dot(normal, light_dir), 0.0);
            let brightness = (diffuse - 0.5) * 2.0;
            base_color = vec3<f32>(brightness, brightness, brightness);
        } else {
            // Raw slope mode: red=X, green=Y
            let red = slope.x * 100.0 + 0.5;
            let green = slope.y * 100.0 + 0.5;
            base_color = vec3<f32>(red, green, 0.0);
        }
    } else {
        // Count how many channels are visible
        var channel_count = 0u;
        if (params.alpha_show != 0u) { channel_count += 1u; }
        if (params.beta_show != 0u) { channel_count += 1u; }
        if (params.gamma_show != 0u) { channel_count += 1u; }

        if (channel_count == 0u) {
            // If no channels selected, show all by default (consistent scaling across channels)
            base_color = vec3<f32>(beta * 0.2, alpha * 0.2, gamma_display * 0.2);
        } else if (channel_count == 1u) {
            // Single channel mode: show as grayscale (use full 0..1 for all channels)
            if (params.alpha_show != 0u) {
                base_color = vec3<f32>(alpha, alpha, alpha);
            } else if (params.beta_show != 0u) {
                base_color = vec3<f32>(beta, beta, beta);
            } else { // gamma_show
                // Use raw gamma value for consistency with alpha/beta single channel display
                base_color = vec3<f32>(gamma, gamma, gamma);
            }
        } else {
            // Multi-channel mode: show as color mix
            var red = 0.0;
            var green = 0.0;
            var blue = 0.0;
            if (params.alpha_show != 0u) { green += alpha * 0.2; }
            if (params.beta_show != 0u) { red += beta * 0.2; }
            if (params.gamma_show != 0u) { blue += gamma_display * 0.2; }

            // Legacy gamma_debug mode for backwards compatibility
            if (params.gamma_debug != 0u) {
                base_color = vec3<f32>(gamma_display, gamma_display, gamma_display);
            } else {
                base_color = vec3<f32>(red, green, blue);
            }
        }
    }

    // Write base color before overlay
    visual_grid[visual_idx] = vec4<f32>(base_color, 1.0);
    
    // ====== RGB TRAIL OVERLAY ======
    // Sample trail grid and blend onto the visual output
    let trail_color = trail_grid[grid_index(world_pos)].xyz;
    
    // Trail-only mode: show just the trail on black background
    if (params.trail_show != 0u) {
        visual_grid[visual_idx] = vec4<f32>(trail_color * params.trail_opacity, 1.0);
    } else {
        // Normal mode: additive blending with opacity control (controlled by trail_opacity parameter)
        let base_color = visual_grid[visual_idx].rgb;
        let blended_color = base_color + trail_color * params.trail_opacity;
        visual_grid[visual_idx] = vec4<f32>(clamp(blended_color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
    }
}

// ============================================================================
// RENDER VERTEX/FRAGMENT SHADERS
// ============================================================================

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Direct 1:1 pixel mapping - no camera transform needed
    // The agents already drew in screen space with camera applied
    let color = textureSample(visual_tex, visual_sampler, in.uv);
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
    agent.rna_progress = 0u;
    agent.pairing_counter = 0u;
    agent.is_selected = 0u;
    agent.generation = 0u;
    agent.age = 0u;
    agent.total_mass = 0.0;

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
    // Initialize packed genome from ASCII genome
    {
        let packed = genome_pack_into(agent);
        for (var i = 0u; i < PACKED_GENOME_WORDS; i++) {
            let packed_word = packed_read_word(packed, i);
            agent.genome_packed[i] = packed_word;
        }
    }

    for (var i = 0u; i < MAX_BODY_PARTS; i++) {
        agent.body[i].pos = vec2<f32>(0.0);
        agent.body[i].size = 0.0;
        agent.body[i].part_type = 0u;
        agent.body[i].alpha_signal = 0.0;
        agent.body[i].beta_signal = 0.0;
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
    agent.rna_progress = 0u;
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
        let scale = environment_init.noise_scale;
        let contrast = environment_init.noise_contrast;
        let octaves = environment_init.noise_octaves;
        
        let coord = vec2<f32>(f32(x), f32(y)) / f32(GRID_SIZE);
        output_value = layered_noise(coord, seed, octaves, scale, contrast);
        output_value = clamp(output_value, 0.0, 1.0);
    }

    if (mode == 1u) {
        alpha_grid[idx] = output_value;
    } else if (mode == 2u) {
        beta_grid[idx] = output_value;
    } else if (mode == 3u) {
        gamma_grid[idx] = output_value;
    }
}
