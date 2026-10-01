// Vampire mouths drain energy from nearby living agents
// ============================================================================

// Helper: add a force into the low-res fluid force grid using bilinear splatting.
// This reduces axis-aligned artifacts vs. writing into a single nearest cell.
fn atomic_add_f32(idx: u32, v: f32) {
    // CAS loop: atomic add for f32 stored as u32 bits.
    // NOTE: naga currently disallows passing storage pointers into functions,
    // so we address the global buffer by index here.
    loop {
        let old_bits = atomicLoad(&fluid_forces[idx]);
        let old_val = bitcast<f32>(old_bits);
        let new_bits = bitcast<u32>(old_val + v);
        let res = atomicCompareExchangeWeak(&fluid_forces[idx], old_bits, new_bits);
        if (res.exchanged) {
            break;
        }
    }
}

fn add_fluid_force_splat(pos_world: vec2<f32>, f: vec2<f32>) {
    if (pos_world.x < 0.0 || pos_world.x >= f32(SIM_SIZE) || pos_world.y < 0.0 || pos_world.y >= f32(SIM_SIZE)) {
        return;
    }

    let ws = f32(SIM_SIZE);
    let grid_f = f32(FLUID_GRID_SIZE);

    // Continuous coords in fluid-cell space.
    let gx = (pos_world.x / ws) * grid_f;
    let gy = (pos_world.y / ws) * grid_f;

    // Match the fluid shader convention: cell centers at (x+0.5, y+0.5).
    let x = gx - 0.5;
    let y = gy - 0.5;

    let x0 = i32(floor(x));
    let y0 = i32(floor(y));
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = fract(x);
    let fy = fract(y);

    let ix0 = clamp(x0, 0, i32(FLUID_GRID_SIZE) - 1);
    let iy0 = clamp(y0, 0, i32(FLUID_GRID_SIZE) - 1);
    let ix1 = clamp(x1, 0, i32(FLUID_GRID_SIZE) - 1);
    let iy1 = clamp(y1, 0, i32(FLUID_GRID_SIZE) - 1);

    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    let idx00 = u32(iy0) * FLUID_GRID_SIZE + u32(ix0);
    let idx10 = u32(iy0) * FLUID_GRID_SIZE + u32(ix1);
    let idx01 = u32(iy1) * FLUID_GRID_SIZE + u32(ix0);
    let idx11 = u32(iy1) * FLUID_GRID_SIZE + u32(ix1);

    // Atomic accumulate into interleaved (x,y) entries per cell.
    let base00 = idx00 * 2u;
    let base10 = idx10 * 2u;
    let base01 = idx01 * 2u;
    let base11 = idx11 * 2u;

    atomic_add_f32(base00 + 0u, f.x * w00);
    atomic_add_f32(base00 + 1u, f.y * w00);
    atomic_add_f32(base10 + 0u, f.x * w10);
    atomic_add_f32(base10 + 1u, f.y * w10);
    atomic_add_f32(base01 + 0u, f.x * w01);
    atomic_add_f32(base01 + 1u, f.y * w01);
    atomic_add_f32(base11 + 0u, f.x * w11);
    atomic_add_f32(base11 + 1u, f.y * w11);
}

// Option 5B: force dipole (push–pull) to generate a wake/current with ~zero net momentum.
// `dir_world` need not be normalized.
fn add_fluid_force_dipole(pos_world: vec2<f32>, dir_world: vec2<f32>, f_mag: f32, separation_world: f32) {
    let dl = length(dir_world);
    if (dl < 1e-6 || separation_world <= 0.0 || f_mag == 0.0) {
        return;
    }
    let d = dir_world / dl;

    // Place push/pull around the body center.
    let p_push = pos_world - d * (0.5 * separation_world);
    let p_pull = pos_world + d * (0.5 * separation_world);
    let f = d * f_mag;

    // Equal and opposite forces -> dipole-like stirring.
    add_fluid_force_splat(p_push, f);
    add_fluid_force_splat(p_pull, -f);
}

fn is_bad_f32(x: f32) -> bool {
    // WGSL portability: detect NaN via (x != x). Detect Inf/overflow via a large threshold.
    return (x != x) || (abs(x) > 1e20);
}

fn sanitize_f32(x: f32) -> f32 {
    return select(x, 0.0, is_bad_f32(x));
}

const PI_F32: f32 = 3.141592653589793;
const TWO_PI_F32: f32 = 6.283185307179586;

fn wrap_angle_pi(x: f32) -> f32 {
    // Wrap to [-pi, pi).
    return x - TWO_PI_F32 * floor((x + PI_F32) / TWO_PI_F32);
}

// Experiment toggle:
// If false, propellers do NOT apply thrust/torque directly to agent movement,
// and instead only influence motion indirectly through the fluid.
const PROPELLERS_APPLY_DIRECT_FORCE: bool = true;

// Experiment toggle:
// Inject forces into the fluid based on body-part motion caused by morphology updates.
// This enables propulsion-by-undulation once fluids are enabled, even with propellers disabled.
const MORPHOLOGY_INJECT_FLUID_FORCE: bool = true;
// Strength of morphology-induced fluid coupling (keep small; morphology updates can move parts a lot per frame).
const MORPHOLOGY_FLUID_COUPLING: f32 = 1.0;
// Clamp to avoid huge impulses when morphology jumps between frames.
const MORPHOLOGY_MAX_WORLD_DELTA: f32 = 2.0;
const MORPHOLOGY_MAX_WORLD_VEL: f32 = 20.0;

// Virtual anisotropic drag swimming (gridless, low-Reynolds RFT-like).
// Uses ONLY internal morphology deformation (pre vs post morphology positions).
// Converts lateral undulation into forward thrust via drag anisotropy.
// MORPHOLOGY_SWIM_ENABLED is now a runtime parameter accessed via ms_enabled()
// When true, microswimming is handled by a dedicated compute pass (`microswim_agents`).
// This keeps `process_agents` easier to debug and prevents double-applying swim forces.
const MORPHOLOGY_SWIM_SEPARATE_PASS: bool = true;
// Overall strength multiplier (driven by UI prop-wash knobs; this scales the resulting thrust velocity).
const MORPHOLOGY_SWIM_COUPLING: f32 = 1.0;
// Parallel drag coefficient (low = slippery along body).
const MORPHOLOGY_SWIM_BASE_DRAG: f32 = 0.2;
// Perpendicular drag multiplier (high = strong side resistance). Typical flagella: ~3-6.
const MORPHOLOGY_SWIM_ANISOTROPY: f32 = 5.0;
// Clamp per-tick contribution to avoid instability.
const MORPHOLOGY_SWIM_MAX_FRAME_VEL: f32 = 2;

// Simple anisotropic fluid coupling (penalty method).
// Anisotropic drag allows undulation to generate net thrust by damping
// perpendicular motion more than parallel motion.
const FLUID_TWO_WAY_COUPLING_ENABLED: bool = false;

// Simple isotropic fluid coupling (penalty method).
// For option 5B we keep agent motion microswim-driven and only inject wakes into the fluid.
const FLUID_COUPLING_SIMPLE_ENABLED: bool = false;
// Anisotropic coupling: perpendicular drag > parallel drag along segment tangent.
// This allows undulation to generate net thrust.
const FLUID_COUPLING_ANISOTROPY: f32 = 1.0;


// Heading correction for morphology swimming:
// Keep the agent's rotation roughly aligned with its actual velocity direction.
// This prevents "crabbing" cases where an agent travels sideways with its body rotated ~90°.
const MORPHOLOGY_SWIM_HEADING_ALIGN_ENABLED: bool = true;
const MORPHOLOGY_SWIM_HEADING_ALIGN_STRENGTH: f32 = 0.5; // converts angle error -> angular velocity
const MORPHOLOGY_SWIM_HEADING_ALIGN_MAX_ANGVEL: f32 = 0.08;
const MORPHOLOGY_SWIM_HEADING_ALIGN_MIN_SPEED: f32 = 0.02; // per-tick speed
const MORPHOLOGY_SWIM_HEADING_ALIGN_FULL_SPEED: f32 = 0.25;

// Option 2: morphology determines a target orientation, physics follows.
// WARNING: For flagellar/undulatory swimmers (long sperm-like bodies), this feature can
// introduce slow cumulative spin drift because tiny floating-point noise in the best-fit
// rotation estimate accumulates over many frames. Keep it disabled by default.
const MORPHOLOGY_ORIENT_FOLLOW_ENABLED: bool = true;
const MORPHOLOGY_ORIENT_FOLLOW_STRENGTH: f32 = 0.25;
const MORPHOLOGY_ORIENT_MAX_FRAME_ANGVEL: f32 = 0.15;

// Vampire mouth and spike organ constants
// Reach is now variable per-part (10..80) derived from the next part's parameter1.
const DEFENSE_RADIUS_DEFAULT: f32 = 30.0;
const DEFENSE_RADIUS_MAX: f32 = 80.0;
const SPIKE_CHEM_DEPOSIT_RADIUS: f32 = 30.0; // Radius around mouth to deposit victim energy
// Vampire finisher: after a successful drain, vampire mouths may kill the victim.
// Probability scales with victim body length (body_count).
const VAMPIRE_KILL_PROB_PER_PART: f32 = 0.01;
const VAMPIRE_KILL_PROB_MAX: f32 = 0.75;
// Spike tuning:
// - Cooldown is stored in spike part body[i]._pad.x (no struct changes).
// - Open/close state is stored in body[i]._pad.y (smoothed, 0..1).
// - Spikes are meant to be passive defense: keep kill cost low.
const SPIKE_COOLDOWN_FRAMES: f32 = 2.0;
const SPIKE_KILL_COST_BASE: f32 = 0.02;
const SPIKE_KILL_COST_FRAC_OF_VICTIM: f32 = 0.00;

// Decomposition tuning:
// QUANTIZED uptake: per-frame probability that a part of mass 1.0 captures a
// monomer, PER monomer present in the cell (first-order, discrete). Scaled by
// part mass and agent stillness.
const UPTAKE_P_PER_QUANTUM: f32 = 0.01;

// Bulldozing: per-frame probability (per part of mass 1.0 at full speed) of
// shoving one gamma quantum from under the part to the cell behind it.
// Base digging probability per frame (per part of mass 1.0). Scaled at the
// use sites by contact pressure (a slow grinding body still digs — the old
// pure-speed scaling meant terrain GRIP suppressed digging almost
// entirely) and by the runtime Digging Rate slider
// (params._pad_dye_beta_color, repurposed slot).
const DIG_P_PER_FRAME: f32 = 0.3;

// Pairing capture reach: any body part can grab the complement monomer
// from a cell within this radius of itself (internal diffusion carries it
// to the replication site). Roughly one body-part diameter.
const PAIRING_REACH: f32 = 2.5;
// Hard cap on bases paired per epoch (bounds the per-agent scan cost).
const PAIRING_MAX_PER_EPOCH: u32 = 8u;

// Quantized terrain collision:
const GAMMA_SOLID_THRESHOLD: u32 = 3u;
// Terrain grip: how strongly embedded body parts anchor the agent,
// per quantum of gamma under the part.
const GAMMA_GRIP_SOLID: f32 = 1.0;   // part inside solid rock: near-pin
const GAMMA_GRIP_RUBBLE: f32 = 0.15; // part on loose rubble: slowdown
const GAMMA_EJECT_FORCE: f32 = 30.0;
const GAMMA_WALL_FORCE: f32 = 10.0;
const GAMMA_SOLID_BRAKE: f32 = 1.0;
const GAMMA_RUBBLE_DRAG: f32 = 0.3;

// Closed world: a STILLBIRTH (non-viable genome after mutation) must return
// everything it embodies — the strand its parent paid complements for, plus
// its inherited energy — otherwise matter silently vanishes with every
// failed mutation (this was a real, dominant mass leak).
fn deposit_stillbirth(agent_id: u32, packed: array<u32, GENOME_PACKED_WORDS>, offset: u32, len: u32, energy: f32, pos: vec2<f32>) {
    let comp = genome_composition(packed, offset, len);
    // Energy is harvested activation, not matter: it evaporates with the
    // stillbirth. Only the strand's monomers return (as SPENT).
    let _unused_energy = energy;
    let quanta = f32(len);
    let idx = grid_index(pos);
    let seed = (agent_id + 1u) * 2654435761u ^ (params.random_seed * 668265263u);
    for (var c = 0u; c < 4u; c++) {
        chem_deposit_prob(idx, c, comp[c] * quanta, seed ^ (c * 374761393u));
    }
}

// Displace one gamma quantum out of `from_idx`, mole-style: try the cell
// behind (relative to the push), then the two sides in hashed left/right
// order. The quantum lands in the first non-solid candidate; if all three
// are solid rock, nothing moves (you cannot compact rock into rock).
// Side-casting is what lets agents dig INTO deep terrain: head-on the
// "behind" cell is often solid too, but the flanks rarely are.
// Shoved quanta can HEAP: targets accept material up to this count (above
// the solid threshold), so pushed spoil accumulates into piles and berms
// instead of only spreading thin — agents can bank up walls, and the
// sandpile relaxation makes tall heaps slump into natural talus. Filipe's
// request: a not-completely-full neighbor should take the quantum.
const GAMMA_PILE_MAX: u32 = 6u;

fn gamma_shove(from_idx: u32, from_pos: vec2<f32>, back_dir: vec2<f32>, seed: u32) {
    let perp = vec2<f32>(-back_dir.y, back_dir.x);
    let flip = select(1.0, -1.0, (hash(seed) & 1u) == 1u);
    var cand: array<vec2<f32>, 3>;
    cand[0] = back_dir;
    cand[1] = perp * flip;
    cand[2] = perp * (-flip);
    for (var c = 0u; c < 3u; c++) {
        let tp = clamp_position(from_pos + cand[c] * 1.5);
        let ti = grid_index(tp);
        if (ti != from_idx && gamma_count(ti) < GAMMA_PILE_MAX) {
            let _gm = gamma_move_one(from_idx, ti);
            return;
        }
    }
}

// The velocity at which suspended bodies RIDE the fluid, in world units
// per tick — the same flow the dye visualizes. Agents at push strength 1
// follow the currents (and orbit the vortices) at full visual speed.
// Safety-capped below VEL_MAX so violent transients cannot launch agents
// beyond the physics envelope. NOTE: monomer QUANTA still hop at their
// capped law (max ~1 cell/epoch) — the soup pattern lags the flow in
// violent currents; that is a known approximation of the quanta transport,
// not of the agents.
// Empirical match between the fluid solver's velocity units and the speed
// the dye VISUALLY streams at per epoch (the solver has its own dt and
// substep cadence, so the analytic conversion chain is ambiguous). Filipe
// calibrated this by eye: with the factor at 1.0 the ride was ~6-7x the
// visual flow; 0.15 matches it.
// PARCEL-EXACT riding: a fluid parcel travels v * dt * cell_to_world world
// units per epoch, which is exactly this formula at factor 1.0. (The old
// 0.15/0.6 values were empirical compensations for the pre-rebuild
// rocket-inflated field.) Slider at 1 = agents move like the water itself.
const RIDE_VISUAL_MATCH: f32 = 1.0;

fn fluid_ride_velocity(world_pos: vec2<f32>) -> vec2<f32> {
    let v = sample_fluid_velocity_bilinear(world_pos);
    let cell_to_world = f32(SIM_SIZE) / f32(FLUID_GRID_SIZE);
    var vw = v * (cell_to_world * max(params.dt, 1e-3) * RIDE_VISUAL_MATCH);
    let m = length(vw);
    let ride_cap = VEL_MAX * 0.8;
    if (m > ride_cap) {
        vw *= ride_cap / m;
    }
    return vw;
}

// UV LIGHT WITH TERRAIN SHADOWS. Light enters at the top of the screen
// (HIGH y), attenuates with water depth (UV Attenuation slider), and is
// absorbed by gamma between the point and the surface: the column above
// is sampled at 16 points and each sample's quanta dim the light. Below
// the lit band the column scan is skipped (light is negligible anyway),
// which keeps the cost confined to the upper layer.
const UV_SHADOW_SAMPLES: u32 = 16u;
const UV_SHADOW_ABSORB: f32 = 0.35; // per gamma quantum per sample

fn uv_light_at_cell(x: u32, y: u32) -> f32 {
    let depth_frac = 1.0 - (f32(y) + 0.5) / f32(GRID_SIZE);
    let water = exp(-max(params.dye_alpha_color_g, 0.5) * depth_frac);
    if (water < 1e-2) {
        return water;
    }
    let span = f32(GRID_SIZE - 1u - min(y, GRID_SIZE - 1u));
    if (span < 1.0) {
        return water;
    }
    var q = 0.0;
    for (var k = 1u; k <= UV_SHADOW_SAMPLES; k++) {
        let sy = y + u32(span * f32(k) / f32(UV_SHADOW_SAMPLES));
        // Only SOLID rock casts shade; scattered rubble specks are
        // translucent (counting them darkened the whole water column).
        let gq = gamma_count(min(sy, GRID_SIZE - 1u) * GRID_SIZE + x);
        if (gq >= GAMMA_SOLID_THRESHOLD) { q += f32(gq); }
    }
    return water * exp(-UV_SHADOW_ABSORB * q);
}

fn uv_light_at_world(p: vec2<f32>) -> f32 {
    let cx = u32(clamp(p.x / f32(SIM_SIZE) * f32(GRID_SIZE), 0.0, f32(GRID_SIZE - 1u)));
    let cy = u32(clamp(p.y / f32(SIM_SIZE) * f32(GRID_SIZE), 0.0, f32(GRID_SIZE - 1u)));
    return uv_light_at_cell(cx, cy);
}

// Unified "ionization" chemical sensors: stochastic collision sampling radius
// (world units) and charge gained per monomer collision. Charge fades each
// frame at the variant's falloff rate.
const SENSOR_ION_RADIUS: f32 = 6.0;
const SENSOR_CHARGE_PER_HIT: f32 = 0.12;

@compute @workgroup_size(256)
fn drain_energy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent_id = gid.x;
    if (agent_id >= params.agent_count) {
        return;
    }

    // Spatial grid uses an epoch stamp so we don't have to clear it every frame.
    // Store stamp = (epoch + 1) so a zero-initialized buffer is always treated as empty.
    let current_stamp = params.epoch + 1u;

    // NOTE: This pass is intended to run after process_agents so that it can
    // operate on the fully materialized per-agent state in agents_out.
    let agent_ptr = &agents_out[agent_id];
    let agent_alive = (*agent_ptr).alive;
    let agent_energy = energy_from_u32(atomicLoad(&(*agent_ptr).energy));

    // Skip dead agents and pending-kill agents.
    // alive==2u is a kill signal (agent will decompose itself next tick in process_agents).
    if (agent_alive != 1u || agent_energy <= 0.0) {
        return;
    }

    // Newborn grace: don't attack or get attacked in the first few frames after spawning
    if ((*agent_ptr).age < VAMPIRE_NEWBORN_GRACE_FRAMES) {
        return;
    }

    // Collect vampire mouth organ indices (type 33).
    // This lets us iterate only vampire mouths later, instead of scanning every body part.
    let body_count = min((*agent_ptr).body_count, MAX_BODY_PARTS);
    var total_energy_gained = 0.0;
    var vampire_mouth_indices: array<u32, MAX_BODY_PARTS>;
    var vampire_mouth_count = 0u;
    for (var i = 0u; i < body_count; i++) {
        let base_type = get_base_part_type(agents_out[agent_id].body[i].part_type);
        if (base_type == 33u) {
            vampire_mouth_indices[vampire_mouth_count] = i;
            vampire_mouth_count += 1u;
        }
    }
    if (vampire_mouth_count == 0u) {
        return;
    }

    // Vampire reach and corresponding grid search radius.
    // NOTE: cap to 10 (previous behavior) to avoid pathological loop sizes.
    let scale = f32(SPATIAL_GRID_SIZE) / f32(SIM_SIZE);
    let cell_size = f32(SIM_SIZE) / f32(SPATIAL_GRID_SIZE);
    // Conservative grid search radius: max possible reach.
    let grid_radius = clamp(i32(ceil(DEFENSE_RADIUS_MAX / cell_size)), 1, 10);

    // Process each vampire mouth organ

    for (var mi = 0u; mi < vampire_mouth_count; mi++) {
        let i = vampire_mouth_indices[mi];
        let part = agents_out[agent_id].body[i];
        let mouth_local_pos = part.pos;

        // Open/close state stored in _pad.y (0..1). When closed, do nothing.
        let mouth_open = clamp(part._pad.y, 0.0, 1.0);
        if (mouth_open < 0.5) {
            // Keep visualization stable.
            continue;
        }

        // Variable reach from the NEXT part's parameter1 (or default when last).
        var reach = DEFENSE_RADIUS_DEFAULT;
        if (i + 1u < agents_out[agent_id].body_count) {
            let next_base = get_base_part_type(agents_out[agent_id].body[i + 1u].part_type);
            reach = param1_to_defense_radius(get_amino_acid_properties(next_base).parameter1);
        }

        // World position for this mouth. Keep this computed regardless of enable state so
        // we can always update the tracking data.
        let rotated_pos = apply_agent_rotation(mouth_local_pos, (*agent_ptr).rotation);
        let mouth_world_pos = (*agent_ptr).position + rotated_pos;

            // Cooldown timer (stored in _pad.x)
            var current_cooldown = agents_out[agent_id].body[i]._pad.x;
            if (current_cooldown > 0.0) {
                current_cooldown -= 1.0;
                agents_out[agent_id].body[i]._pad.x = current_cooldown;
            }

            // No longer disable vampire mouths when enablers are present
            var block = 0.0;
            block = min(block, 1.0);
            let quadratic_block = block * block;
            let mouth_activity = 1.0 - quadratic_block;

            // Only work if not fully suppressed
            if (mouth_activity > 0.0001) {
                // Mouth-centered spiral search over spatial grid.
                // We break early on the first in-range victim to save compute.
                var closest_victim_id = 0xFFFFFFFFu;
                let vamp_size = max(f32(body_count), 1.0);

                let mouth_grid_x = u32(clamp(mouth_world_pos.x * scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));
                let mouth_grid_y = u32(clamp(mouth_world_pos.y * scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));

                var found_victim = false;
                for (var radius: i32 = 0; radius <= grid_radius; radius++) {
                    if (found_victim) {
                        break;
                    }
                    if (radius == 0) {
                        // Single center cell
                        let check_x = i32(mouth_grid_x);
                        let check_y = i32(mouth_grid_y);
                        let check_cell = u32(check_y) * SPATIAL_GRID_SIZE + u32(check_x);
                        let stamp = atomicLoad(&agent_spatial_grid[spatial_epoch_index(check_cell)]);
                        if (stamp == current_stamp) {
                            let raw_neighbor_id = atomicLoad(&agent_spatial_grid[spatial_id_index(check_cell)]);
                            let neighbor_id = raw_neighbor_id & 0x7FFFFFFFu;
                            if (neighbor_id != SPATIAL_GRID_EMPTY && neighbor_id != SPATIAL_GRID_CLAIMED && neighbor_id != agent_id) {
                                let neighbor_alive = agents_out[neighbor_id].alive;
                                let neighbor_energy = energy_from_u32(atomicLoad(&agents_out[neighbor_id].energy));
                                let neighbor_age = agents_out[neighbor_id].age;
                                if (neighbor_alive == 1u && neighbor_energy > 0.0 && neighbor_age >= VAMPIRE_NEWBORN_GRACE_FRAMES) {
                                    let dist = length(mouth_world_pos - agents_out[neighbor_id].position);
                                    if (dist < reach) {
                                        // Size-based attack probability: larger vampires better at catching small prey.
                                        // Same size => 100%, vampire 2x bigger => 100%, victim 2x bigger => 50%.
                                        let victim_size = max(f32(min(agents_out[neighbor_id].body_count, MAX_BODY_PARTS)), 1.0);
                                        let attack_prob = clamp(vamp_size / victim_size, 0.0, 1.0);
                                        let attack_seed = (agent_id * 747796405u)
                                            ^ (neighbor_id * 2891336453u)
                                            ^ (params.epoch * 196613u)
                                            ^ (params.random_seed * 83492791u)
                                            ^ (i * 374761393u)
                                            ^ (check_cell * 668265263u);
                                        let attack_rnd = f32(hash(attack_seed)) / 4294967295.0;
                                        if (attack_rnd <= attack_prob) {
                                            closest_victim_id = neighbor_id;
                                            found_victim = true;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // Top and bottom edges
                        for (var dx: i32 = -radius; dx <= radius; dx++) {
                            if (found_victim) {
                                break;
                            }
                            // Top
                            var check_x = i32(mouth_grid_x) + dx;
                            var check_y = i32(mouth_grid_y) - radius;
                            if (check_x >= 0 && check_x < i32(SPATIAL_GRID_SIZE) && check_y >= 0 && check_y < i32(SPATIAL_GRID_SIZE)) {
                                let check_cell = u32(check_y) * SPATIAL_GRID_SIZE + u32(check_x);
                                let stamp = atomicLoad(&agent_spatial_grid[spatial_epoch_index(check_cell)]);
                                if (stamp == current_stamp) {
                                    let raw_neighbor_id = atomicLoad(&agent_spatial_grid[spatial_id_index(check_cell)]);
                                    let neighbor_id = raw_neighbor_id & 0x7FFFFFFFu;
                                    if (neighbor_id != SPATIAL_GRID_EMPTY && neighbor_id != SPATIAL_GRID_CLAIMED && neighbor_id != agent_id) {
                                        let neighbor_alive = agents_out[neighbor_id].alive;
                                        let neighbor_energy = energy_from_u32(atomicLoad(&agents_out[neighbor_id].energy));
                                        let neighbor_age = agents_out[neighbor_id].age;
                                        if (neighbor_alive == 1u && neighbor_energy > 0.0 && neighbor_age >= VAMPIRE_NEWBORN_GRACE_FRAMES) {
                                            let dist = length(mouth_world_pos - agents_out[neighbor_id].position);
                                            if (dist < reach) {
                                                let victim_size = max(f32(min(agents_out[neighbor_id].body_count, MAX_BODY_PARTS)), 1.0);
                                                let attack_prob = clamp(victim_size / vamp_size, 0.0, 1.0);
                                                let attack_seed = (agent_id * 747796405u)
                                                    ^ (neighbor_id * 2891336453u)
                                                    ^ (params.epoch * 196613u)
                                                    ^ (params.random_seed * 83492791u)
                                                    ^ (i * 374761393u)
                                                    ^ (check_cell * 668265263u);
                                                let attack_rnd = f32(hash(attack_seed)) / 4294967295.0;
                                                if (attack_rnd <= attack_prob) {
                                                    closest_victim_id = neighbor_id;
                                                    found_victim = true;
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // Bottom
                            if (!found_victim) {
                                check_x = i32(mouth_grid_x) + dx;
                                check_y = i32(mouth_grid_y) + radius;
                                if (check_x >= 0 && check_x < i32(SPATIAL_GRID_SIZE) && check_y >= 0 && check_y < i32(SPATIAL_GRID_SIZE)) {
                                    let check_cell = u32(check_y) * SPATIAL_GRID_SIZE + u32(check_x);
                                    let stamp = atomicLoad(&agent_spatial_grid[spatial_epoch_index(check_cell)]);
                                    if (stamp == current_stamp) {
                                        let raw_neighbor_id = atomicLoad(&agent_spatial_grid[spatial_id_index(check_cell)]);
                                        let neighbor_id = raw_neighbor_id & 0x7FFFFFFFu;
                                        if (neighbor_id != SPATIAL_GRID_EMPTY && neighbor_id != SPATIAL_GRID_CLAIMED && neighbor_id != agent_id) {
                                            let neighbor_alive = agents_out[neighbor_id].alive;
                                            let neighbor_energy = energy_from_u32(atomicLoad(&agents_out[neighbor_id].energy));
                                            let neighbor_age = agents_out[neighbor_id].age;
                                            if (neighbor_alive == 1u && neighbor_energy > 0.0 && neighbor_age >= VAMPIRE_NEWBORN_GRACE_FRAMES) {
                                                let dist = length(mouth_world_pos - agents_out[neighbor_id].position);
                                                if (dist < reach) {
                                                    let victim_size = max(f32(min(agents_out[neighbor_id].body_count, MAX_BODY_PARTS)), 1.0);
                                                    let attack_prob = clamp(vamp_size / victim_size, 0.0, 1.0);
                                                    let attack_seed = (agent_id * 747796405u)
                                                        ^ (neighbor_id * 2891336453u)
                                                        ^ (params.epoch * 196613u)
                                                        ^ (params.random_seed * 83492791u)
                                                        ^ (i * 374761393u)
                                                        ^ (check_cell * 668265263u);
                                                    let attack_rnd = f32(hash(attack_seed)) / 4294967295.0;
                                                    if (attack_rnd <= attack_prob) {
                                                        closest_victim_id = neighbor_id;
                                                        found_victim = true;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        // Left and right edges (excluding corners already covered)
                        for (var dy: i32 = -radius + 1; dy <= radius - 1; dy++) {
                            if (found_victim) {
                                break;
                            }
                            // Left
                            var check_x = i32(mouth_grid_x) - radius;
                            var check_y = i32(mouth_grid_y) + dy;
                            if (check_x >= 0 && check_x < i32(SPATIAL_GRID_SIZE) && check_y >= 0 && check_y < i32(SPATIAL_GRID_SIZE)) {
                                let check_cell = u32(check_y) * SPATIAL_GRID_SIZE + u32(check_x);
                                let stamp = atomicLoad(&agent_spatial_grid[spatial_epoch_index(check_cell)]);
                                if (stamp == current_stamp) {
                                    let raw_neighbor_id = atomicLoad(&agent_spatial_grid[spatial_id_index(check_cell)]);
                                    let neighbor_id = raw_neighbor_id & 0x7FFFFFFFu;
                                    if (neighbor_id != SPATIAL_GRID_EMPTY && neighbor_id != SPATIAL_GRID_CLAIMED && neighbor_id != agent_id) {
                                        let neighbor_alive = agents_out[neighbor_id].alive;
                                        let neighbor_energy = energy_from_u32(atomicLoad(&agents_out[neighbor_id].energy));
                                        let neighbor_age = agents_out[neighbor_id].age;
                                        if (neighbor_alive == 1u && neighbor_energy > 0.0 && neighbor_age >= VAMPIRE_NEWBORN_GRACE_FRAMES) {
                                            let dist = length(mouth_world_pos - agents_out[neighbor_id].position);
                                            if (dist < reach) {
                                                let victim_size = max(f32(min(agents_out[neighbor_id].body_count, MAX_BODY_PARTS)), 1.0);
                                                let attack_prob = clamp(vamp_size / victim_size, 0.0, 1.0);
                                                let attack_seed = (agent_id * 747796405u)
                                                    ^ (neighbor_id * 2891336453u)
                                                    ^ (params.epoch * 196613u)
                                                    ^ (params.random_seed * 83492791u)
                                                    ^ (i * 374761393u)
                                                    ^ (check_cell * 668265263u);
                                                let attack_rnd = f32(hash(attack_seed)) / 4294967295.0;
                                                if (attack_rnd <= attack_prob) {
                                                    closest_victim_id = neighbor_id;
                                                    found_victim = true;
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // Right
                            if (!found_victim) {
                                check_x = i32(mouth_grid_x) + radius;
                                check_y = i32(mouth_grid_y) + dy;
                                if (check_x >= 0 && check_x < i32(SPATIAL_GRID_SIZE) && check_y >= 0 && check_y < i32(SPATIAL_GRID_SIZE)) {
                                    let check_cell = u32(check_y) * SPATIAL_GRID_SIZE + u32(check_x);
                                    let stamp = atomicLoad(&agent_spatial_grid[spatial_epoch_index(check_cell)]);
                                    if (stamp == current_stamp) {
                                        let raw_neighbor_id = atomicLoad(&agent_spatial_grid[spatial_id_index(check_cell)]);
                                        let neighbor_id = raw_neighbor_id & 0x7FFFFFFFu;
                                        if (neighbor_id != SPATIAL_GRID_EMPTY && neighbor_id != SPATIAL_GRID_CLAIMED && neighbor_id != agent_id) {
                                            let neighbor_alive = agents_out[neighbor_id].alive;
                                            let neighbor_energy = energy_from_u32(atomicLoad(&agents_out[neighbor_id].energy));
                                            let neighbor_age = agents_out[neighbor_id].age;
                                            if (neighbor_alive == 1u && neighbor_energy > 0.0 && neighbor_age >= VAMPIRE_NEWBORN_GRACE_FRAMES) {
                                                let dist = length(mouth_world_pos - agents_out[neighbor_id].position);
                                                if (dist < reach) {
                                                    let victim_size = max(f32(min(agents_out[neighbor_id].body_count, MAX_BODY_PARTS)), 1.0);
                                                    let attack_prob = clamp(vamp_size / victim_size, 0.0, 1.0);
                                                    let attack_seed = (agent_id * 747796405u)
                                                        ^ (neighbor_id * 2891336453u)
                                                        ^ (params.epoch * 196613u)
                                                        ^ (params.random_seed * 83492791u)
                                                        ^ (i * 374761393u)
                                                        ^ (check_cell * 668265263u);
                                                    let attack_rnd = f32(hash(attack_seed)) / 4294967295.0;
                                                    if (attack_rnd <= attack_prob) {
                                                        closest_victim_id = neighbor_id;
                                                        found_victim = true;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // Drain from closest victim only
                if (closest_victim_id != 0xFFFFFFFFu) {
                    // Energy is updated atomically, so we don't need a per-victim lock.
                    // Allow multiple vampires to drain concurrently; the victim loss is saturating.
                    if (current_cooldown <= 0.0) {
                        let victim_energy = energy_from_u32(atomicLoad(&agents_out[closest_victim_id].energy));
                        if (victim_energy > 0.0001) {
                            // VAMPIRE DRAIN: Gradually drain energy from victim
                            // Distance-based falloff: full power at 0 distance, 0 power at max reach
                            let victim_pos = agents_out[closest_victim_id].position;
                            let victim_dist = length(mouth_world_pos - victim_pos);
                            let distance_factor = max(0.0, 1.0 - (victim_dist / max(reach, 1e-3)));

                            // Poison protection organs reduce vampire drain effectiveness by 50% each
                            let victim_poison_protection = agents_out[closest_victim_id].poison_resistant_count;
                            var drain_effectiveness = distance_factor;
                            if (victim_poison_protection > 0u) {
                                drain_effectiveness *= pow(0.5, f32(victim_poison_protection));
                            }

                            // Drain rate: scale by size ratio (vampire / victim) using body-part count proxy.
                            let vamp_size = max(f32(body_count), 1.0);
                            let victim_size = max(f32(min(agents_out[closest_victim_id].body_count, MAX_BODY_PARTS)), 1.0);
                            let size_ratio = vamp_size / victim_size;
                            let drain_rate = clamp(0.2 * size_ratio, 0.0, 0.95);
                            let drain_amount = victim_energy * drain_rate * drain_effectiveness;

                            // Minimum drain threshold
                            if (drain_amount >= 0.01) {
                                // Victim loses 2x the amount drained (harsh penalty), atomically.
                                // Compute vampire gain from the actual amount removed so we never mint energy via races.
                                let victim_energy_loss = drain_amount * 2.0;

                                // Saturating atomic subtract (CAS loop). Returns actual loss in f32.
                                let victim_loss_u = energy_to_u32(victim_energy_loss);
                                var actual_loss_u = 0u;
                                if (victim_loss_u != 0u) {
                                    loop {
                                        let old_u = atomicLoad(&agents_out[closest_victim_id].energy);
                                        if (old_u == 0u) {
                                            break;
                                        }
                                        let sub_u = min(old_u, victim_loss_u);
                                        let new_u = old_u - sub_u;
                                        let res = atomicCompareExchangeWeak(&agents_out[closest_victim_id].energy, old_u, new_u);
                                        if (res.exchanged) {
                                            actual_loss_u = sub_u;
                                            break;
                                        }
                                    }
                                }

                                let actual_loss = energy_from_u32(actual_loss_u);
                                let actual_gain = actual_loss * 0.5;
                                total_energy_gained += actual_gain;
                                // Energy is harvested activation, not matter:
                                // the destroyed half of the drain dissipates
                                // as heat (depositing it would mint mass).
                                // Set cooldown timer
                                agents_out[agent_id].body[i]._pad.x = VAMPIRE_MOUTH_COOLDOWN;
                            }
                        }
                    }
                }
            }

            // Store current position for next frame's movement calculation
            agents_out[agent_id].body[i].data = pack_position_to_f32(mouth_world_pos, f32(SIM_SIZE));
    }

    // Add gained energy to vampire
    if (total_energy_gained > 0.0) {
        let gain_u = energy_to_u32(total_energy_gained);
        if (gain_u != 0u) {
            _ = atomicAdd(&(*agent_ptr).energy, gain_u);
        }
    }
}

// ============================================================================
// SPIKE ORGAN KILL LOGIC - Deferred kill signal (victim decomposes next tick)
// ============================================================================

@compute @workgroup_size(256)
fn spike_kill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent_id = gid.x;
    if (agent_id >= params.agent_count) {
        return;
    }

    // Avoid copying full Agent (contains body arrays) into a local variable.
    let agent_alive = agents_out[agent_id].alive;
    let agent_energy = energy_from_u32(atomicLoad(&agents_out[agent_id].energy));
    let agent_age = agents_out[agent_id].age;
    let agent_position = agents_out[agent_id].position;
    let agent_rotation = agents_out[agent_id].rotation;
    let body_count = agents_out[agent_id].body_count;

    // Skip dead agents and pending-kill agents.
    if (agent_alive != 1u || agent_energy <= 0.0) {
        return;
    }

    // Newborn grace: don't attack or get attacked in the first few frames after spawning
    if (agent_age < VAMPIRE_NEWBORN_GRACE_FRAMES) {
        return;
    }

    // Spatial grid uses an epoch stamp so we don't have to clear it every frame.
    // Store stamp = (epoch + 1) so a zero-initialized buffer is always treated as empty.
    let current_stamp = params.epoch + 1u;

    let scale = f32(SPATIAL_GRID_SIZE) / f32(SIM_SIZE);
    let cell_size = f32(SIM_SIZE) / f32(SPATIAL_GRID_SIZE);
    let grid_radius = clamp(i32(ceil(DEFENSE_RADIUS_MAX / cell_size)), 1, 10);

    // Collect spike organ indices (type 46)
    var spike_indices: array<u32, MAX_BODY_PARTS>;
    var spike_count = 0u;

    for (var i = 0u; i < body_count; i++) {
        let base_type = get_base_part_type(agents_out[agent_id].body[i].part_type);
        if (base_type == 46u) {
            spike_indices[spike_count] = i;
            spike_count += 1u;
        }
    }

    // No spikes, nothing to do
    if (spike_count == 0u) {
        return;
    }

    // For each spike organ
    for (var si = 0u; si < spike_count; si++) {
        let i = spike_indices[si];

        // Open/close state stored in _pad.y (0..1). When closed, do nothing.
        let spike_open = clamp(agents_out[agent_id].body[i]._pad.y, 0.0, 1.0);
        if (spike_open < 0.5) {
            continue;
        }

        // Variable reach from the NEXT part's parameter1 (or default when last).
        var reach = DEFENSE_RADIUS_DEFAULT;
        if (i + 1u < body_count) {
            let next_base = get_base_part_type(agents_out[agent_id].body[i + 1u].part_type);
            reach = param1_to_defense_radius(get_amino_acid_properties(next_base).parameter1);
        }

        // Per-spike cooldown countdown (stored in body[i]._pad.x).
        let cd = agents_out[agent_id].body[i]._pad.x;
        if (cd > 0.0) {
            agents_out[agent_id].body[i]._pad.x = max(cd - 1.0, 0.0);
            continue;
        }

        let spike_rel_pos = agents_out[agent_id].body[i].pos;
        let spike_world_pos = agent_position + apply_agent_rotation(spike_rel_pos, agent_rotation);

        let spike_grid_x = u32(clamp(spike_world_pos.x * scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));
        let spike_grid_y = u32(clamp(spike_world_pos.y * scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));

        // Scan neighborhood cells and kill any nearby living victim.
        for (var dy: i32 = -grid_radius; dy <= grid_radius; dy++) {
            for (var dx: i32 = -grid_radius; dx <= grid_radius; dx++) {
                let check_x = i32(spike_grid_x) + dx;
                let check_y = i32(spike_grid_y) + dy;
                if (check_x < 0 || check_x >= i32(SPATIAL_GRID_SIZE) || check_y < 0 || check_y >= i32(SPATIAL_GRID_SIZE)) {
                    continue;
                }

                let check_cell = u32(check_y) * SPATIAL_GRID_SIZE + u32(check_x);
                let stamp = atomicLoad(&agent_spatial_grid[spatial_epoch_index(check_cell)]);
                if (stamp != current_stamp) {
                    continue;
                }

                let raw_neighbor_id = atomicLoad(&agent_spatial_grid[spatial_id_index(check_cell)]);
                let neighbor_id = raw_neighbor_id & 0x7FFFFFFFu;
                if (neighbor_id == SPATIAL_GRID_EMPTY || neighbor_id == SPATIAL_GRID_CLAIMED || neighbor_id == agent_id) {
                    continue;
                }

                let victim_alive = agents_out[neighbor_id].alive;
                let victim_energy = energy_from_u32(atomicLoad(&agents_out[neighbor_id].energy));
                let victim_age = agents_out[neighbor_id].age;
                if (victim_alive != 1u || victim_energy <= 0.0 || victim_age < VAMPIRE_NEWBORN_GRACE_FRAMES) {
                    continue;
                }

                let dist = length(spike_world_pos - agents_out[neighbor_id].position);
                if (dist < reach) {
                    // Size-based spike hit probability: small victims can slip past.
                    // Same size => 100%, half size => 50%, etc.
                    let spike_owner_size = max(f32(min(body_count, MAX_BODY_PARTS)), 1.0);
                    let victim_size = max(f32(min(agents_out[neighbor_id].body_count, MAX_BODY_PARTS)), 1.0);
                    let hit_prob = clamp(victim_size / spike_owner_size, 0.0, 1.0);
                    let hit_seed = (agent_id * 747796405u)
                        ^ (neighbor_id * 2891336453u)
                        ^ (params.epoch * 196613u)
                        ^ (params.random_seed * 83492791u)
                        ^ (i * 374761393u)
                        ^ (check_cell * 668265263u);
                    let hit_rnd = f32(hash(hit_seed)) / 4294967295.0;
                    if (hit_rnd > hit_prob) {
                        continue;
                    }

                    // No per-victim lock required here; redundant kill signals are harmless.

                    // Send kill signal: victim will decompose itself next tick in process_agents.
                    // alive==2u means "pending kill".
                    agents_out[neighbor_id].alive = 2u;
                    agents_out[neighbor_id].velocity = vec2<f32>(0.0);

                    // Pay an energy cost to prevent self-sustaining stationary spike blobs.
                    let cost = SPIKE_KILL_COST_BASE + SPIKE_KILL_COST_FRAC_OF_VICTIM * victim_energy;
                    let cost_u = energy_to_u32(cost);
                    if (cost_u != 0u) {
                        loop {
                            let old_u = atomicLoad(&agents_out[agent_id].energy);
                            if (old_u == 0u) {
                                break;
                            }
                            let sub_u = min(old_u, cost_u);
                            let new_u = old_u - sub_u;
                            let res = atomicCompareExchangeWeak(&agents_out[agent_id].energy, old_u, new_u);
                            if (res.exchanged) {
                                break;
                            }
                        }
                    }

                    // Set cooldown and limit to one kill per agent per epoch.
                    agents_out[agent_id].body[i]._pad.x = SPIKE_COOLDOWN_FRAMES;
                    return;
                }
            }
        }
    }
}

// ============================================================================

@compute @workgroup_size(256)
fn process_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent_id = gid.x;
    if (agent_id >= params.agent_count) {
        return;
    }

    // Spatial grid uses an epoch stamp so we don't have to clear it every frame.
    let current_stamp = params.epoch + 1u;

    // IMPORTANT: Avoid copying the full Agent payload into a local variable.
    // In practice this can pull in the entire body array and become a bandwidth hotspot.
    let agent_alive = agents_in[agent_id].alive;

    // Skip dead agents
    if (agent_alive == 0u) {
        // Avoid copying the full Agent payload for dead entries.
        agents_out[agent_id].alive = 0u;
        agents_out[agent_id].body_count = 0u;
        atomicStore(&agents_out[agent_id].energy, 0u);
        agents_out[agent_id].velocity = vec2<f32>(0.0);
        agents_out[agent_id].pairing_counter = 0u;
        agents_out[agent_id].is_selected = 0u;
        return;
    }

    // Pending-kill: decompose remaining energy using mouth-rule ratios, then die.
    // This is the deferred "kill signal" path (e.g. from spike kills) and avoids
    // double-counting / contention issues in the attacker kernel.
    if (agent_alive == 2u) {
        let body_count = min(agents_in[agent_id].body_count, MAX_BODY_PARTS);
        if (body_count > 0u) {
            // RNA-world decomposition (closed world): return exactly what the
            // agent embodies — the strand's mass (gene_length bases, split by
            // genome composition), the complements captured by the partial
            // pairing strand, and the remaining energy converted to matter.
            let comp = genome_composition(
                agents_in[agent_id].genome_packed,
                agents_in[agent_id].genome_offset,
                agents_in[agent_id].gene_length
            );
            let paired_counts = genome_paired_complement_counts(
                agents_in[agent_id].genome_packed,
                agents_in[agent_id].genome_offset,
                agents_in[agent_id].gene_length,
                agents_in[agent_id].pairing_counter
            );
            // Quantized: 1 monomer per genome base; paired complements are
            // already whole monomer counts; stored energy converts back at
            // (food_power x resistance multiplier) energy per monomer.
            // PRIMORDIAL SEEDS (generation 0): their strand was minted from
            // nothing at world start, so it vanishes on death instead of
            // depositing (keeps world total = what generation set). Their
            // paired complements and energy DID come from the environment
            // and are returned normally.
            let strand_quanta = select(f32(agents_in[agent_id].gene_length), 0.0, agents_in[agent_id].generation == 0u);
            // Energy = harvested activation, NOT matter: it evaporates on
            // death. Only strand + captured complements return (SPENT).
            let per_part = (comp * strand_quanta + paired_counts) * (1.0 / f32(body_count));
            let agent_pos = agents_in[agent_id].position;
            let agent_rot = agents_in[agent_id].rotation;
            for (var i = 0u; i < body_count; i++) {
                let part = agents_in[agent_id].body[i];
                let world_pos = agent_pos + apply_agent_rotation(part.pos, agent_rot);
                let idx = grid_index(world_pos);
                let dseed = (agent_id + 1u) * 2654435761u ^ (i * 668265263u) ^ params.random_seed;
                for (var dc = 0u; dc < 4u; dc++) {
                    chem_deposit_prob(idx, dc, per_part[dc], dseed ^ (dc * 374761393u));
                }
            }
        }

        // Kill now.
        agents_out[agent_id].alive = 0u;
        agents_out[agent_id].body_count = 0u;
        atomicStore(&agents_out[agent_id].energy, 0u);
        agents_out[agent_id].velocity = vec2<f32>(0.0);
        agents_out[agent_id].pairing_counter = 0u;
        agents_out[agent_id].is_selected = 0u;
        return;
    }

    // Pull only the scalar fields we use frequently into locals.
    let agent_position = agents_in[agent_id].position;
    let agent_velocity = agents_in[agent_id].velocity;
    let agent_rotation = agents_in[agent_id].rotation;
    let agent_energy = energy_from_u32(agents_in[agent_id].energy);
    let agent_energy_capacity = agents_in[agent_id].energy_capacity;
    let agent_torque_debug = agents_in[agent_id].torque_debug;
    let agent_morphology_origin = agents_in[agent_id].morphology_origin;
    let agent_body_count = agents_in[agent_id].body_count;
    let agent_pairing_counter = agents_in[agent_id].pairing_counter;
    let agent_is_selected = agents_in[agent_id].is_selected;
    let agent_generation = agents_in[agent_id].generation;
    let agent_age = agents_in[agent_id].age;
    let agent_total_mass = agents_in[agent_id].total_mass;
    let agent_poison_resistant_count = agents_in[agent_id].poison_resistant_count;
    let agent_gene_length = agents_in[agent_id].gene_length;
    let agent_genome_offset = agents_in[agent_id].genome_offset;
    let agent_genome_packed = agents_in[agent_id].genome_packed;

    // Working state for this invocation.
    var agent_pos = agent_position;
    var agent_vel = agent_velocity;
    var agent_rot = agent_rotation;
    var agent_energy_cur = agent_energy;
    var pairing_counter = agent_pairing_counter;

    // Per-frame energy cost for morphology (computed after body rebuild).
    var morphology_change_energy = 0.0;

    // Morphology-driven swim uses per-part drag inside the physics loop.

    // Initialize output scalars explicitly (avoid full Agent megacopy).
    agents_out[agent_id].position = agent_position;
    agents_out[agent_id].velocity = agent_velocity;
    agents_out[agent_id].rotation = agent_rotation;
    atomicStore(&agents_out[agent_id].energy, energy_to_u32(max(agent_energy, 0.0)));
    agents_out[agent_id].energy_capacity = agent_energy_capacity;
    agents_out[agent_id].torque_debug = agent_torque_debug;
    agents_out[agent_id].morphology_origin = agent_morphology_origin;
    agents_out[agent_id].alive = agent_alive;
    agents_out[agent_id].body_count = agent_body_count;
    agents_out[agent_id].pairing_counter = agent_pairing_counter;
    agents_out[agent_id].is_selected = agent_is_selected;
    agents_out[agent_id].generation = agent_generation;
    agents_out[agent_id].age = agent_age;
    agents_out[agent_id].total_mass = agent_total_mass;
    agents_out[agent_id].poison_resistant_count = agent_poison_resistant_count;
    agents_out[agent_id].gene_length = agent_gene_length;

    // Persist genome metadata across ping-pong buffers.
    agents_out[agent_id].genome_offset = agent_genome_offset;

    // Keep genome synchronized across ping-pong buffers.
    // NOTE: This is required for correctness because next frame's agents_in is this frame's
    // agents_out. If we don't write genome every frame, the buffer we write into will retain
    // stale/garbage genome data for existing agents.
    //
    // IMPORTANT: Naga/WGSL validation can reject dynamic indexing into certain array expressions.
    // Use constant indices for the fixed-size packed genome.
    agents_out[agent_id].genome_packed[0u] = agent_genome_packed[0u];
    agents_out[agent_id].genome_packed[1u] = agent_genome_packed[1u];
    agents_out[agent_id].genome_packed[2u] = agent_genome_packed[2u];
    agents_out[agent_id].genome_packed[3u] = agent_genome_packed[3u];
    agents_out[agent_id].genome_packed[4u] = agent_genome_packed[4u];
    agents_out[agent_id].genome_packed[5u] = agent_genome_packed[5u];
    agents_out[agent_id].genome_packed[6u] = agent_genome_packed[6u];
    agents_out[agent_id].genome_packed[7u] = agent_genome_packed[7u];
    agents_out[agent_id].genome_packed[8u] = agent_genome_packed[8u];
    agents_out[agent_id].genome_packed[9u] = agent_genome_packed[9u];
    agents_out[agent_id].genome_packed[10u] = agent_genome_packed[10u];
    agents_out[agent_id].genome_packed[11u] = agent_genome_packed[11u];
    agents_out[agent_id].genome_packed[12u] = agent_genome_packed[12u];
    agents_out[agent_id].genome_packed[13u] = agent_genome_packed[13u];
    agents_out[agent_id].genome_packed[14u] = agent_genome_packed[14u];
    agents_out[agent_id].genome_packed[15u] = agent_genome_packed[15u];

    // ====== MORPHOLOGY BUILD ======
    // Genome scan only happens on first frame (body_count == 0)
    // After that, we just use the cached part_type values in body[]
    var body_count_val = agent_body_count;
    var first_build = (agent_body_count == 0u);
    var start_byte = 0u;

    // Optimization: detect if agent has any signal-PRODUCING organs (sensors, clocks, emitters)
    // If not, signals will remain zero and morphology won't change, so skip rebuild and diffusion.
    var has_signal_producers = false;
    if (!first_build) {
        for (var i = 0u; i < min(body_count_val, MAX_BODY_PARTS); i++) {
            let base_type = get_base_part_type(agents_in[agent_id].body[i].part_type);
            let props = get_amino_acid_properties(base_type);
            // Check if this part produces signals:
            // - Sensors (alpha/beta env sensors, agent sensors, etc.)
            // - Clocks (standalone or signal-driven oscillators)
            // - Internal signal emitters
            // - Slope sensors (type 32)
            if (props.is_alpha_sensor || props.is_beta_sensor || props.is_clock || props.is_signal_emitter || base_type == 32u) {
                has_signal_producers = true;
                break;
            }
        }
    }

    // Cache previous-frame body geometry so we can estimate morphology-driven motion.
    // NOTE: agents_in.body[].pos is already in the *recentered + avg-angle-normalized* local frame
    // from last frame (what you see as stable orientation in the inspector preview).
    // We'll keep both a generic prev cache (used by other systems) and an explicit
    // prev_body_pos_recentered that we populate after the current frame's recentering.
    var prev_body_pos: array<vec2<f32>, MAX_BODY_PARTS>;
    var prev_body_pos_recentered: array<vec2<f32>, MAX_BODY_PARTS>;
    if (!first_build) {
        for (var i = 0u; i < min(body_count_val, MAX_BODY_PARTS); i++) {
            prev_body_pos[i] = agents_in[agent_id].body[i].pos;
        }
    }

    if (first_build) {
        // FIRST BUILD: Scan genome and populate body[].part_type
        var start = 0xFFFFFFFFu;
        if (params.require_start_codon == 1u) {
            start = genome_find_start_codon(agent_genome_packed, agent_genome_offset, agent_gene_length);
        } else {
            start = genome_find_first_coding_triplet(agent_genome_packed, agent_genome_offset, agent_gene_length);
        }

        if (start == 0xFFFFFFFFu) {
            // Non-viable genome (no start codon or no codons): stillbirth —
            // return the embodied matter before dying.
            deposit_stillbirth(agent_id, agent_genome_packed, agent_genome_offset, agent_gene_length, agent_energy, agent_position);
            agents_out[agent_id].alive = 0u;
            agents_out[agent_id].body_count = 0u;
            atomicStore(&agents_out[agent_id].energy, 0u);
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
            let step = translate_codon_step(agent_genome_packed, pos_b, agent_genome_offset, agent_gene_length, params.ignore_stop_codons == 1u);

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
            // Stillbirth (genome translated to zero parts): return matter.
            deposit_stillbirth(agent_id, agent_genome_packed, agent_genome_offset, agent_gene_length, agent_energy, agent_position);
            agents_out[agent_id].alive = 0u;
            atomicStore(&agents_out[agent_id].energy, 0u);
            return;
        }
    }

    // REBUILD body positions every frame (enables dynamic shape changes from signals)
    // Now we just read the cached part_type from body[] instead of re-scanning genome
    // OPTIMIZATION: Skip morphology rebuild if agent has no signal-producing organs

    // Initialize outside the if block so they're in scope for agent_color calculation
    var total_mass_morphology = 0.05; // Default minimum
    var color_sum_morphology = 0.0;

    if (body_count_val > 0u && (first_build || has_signal_producers)) {

        // Poison resistance affects only poison damage (handled later), not signal-controlled bending.

        // Dynamic chain build - angles modulated by alpha/beta signals
        var current_pos = vec2<f32>(0.0);
        var current_angle = 0.0;
        var chirality_flip = 1.0;
        var sum_angle_mass = 0.0;
        var total_mass_angle = 0.0;
        var com_sum = vec2<f32>(0.0);
        var mass_sum = 0.0;
        var total_capacity = 0.0;
        // Reset to 0 before accumulation
        total_mass_morphology = 0.0;
        color_sum_morphology = 0.0;
        var poison_resistant_count = 0u;

        // Loop through existing body parts and rebuild positions
        for (var i = 0u; i < min(body_count_val, MAX_BODY_PARTS); i++) {
            // Read cached part_type and persist it to output (ping-pong safety).
            // IMPORTANT: on the first build we just wrote part_type into agents_out above,
            // so we must read from agents_out here (agents_in is still empty).
            let final_part_type = select(
                agents_in[agent_id].body[i].part_type,
                agents_out[agent_id].body[i].part_type,
                first_build
            );
            agents_out[agent_id].body[i].part_type = final_part_type;
            let base_type = get_base_part_type(final_part_type);

            // Check for chiral flipper
            if (base_type == 30u) {
                chirality_flip = -chirality_flip;
            }

            // Count poison-resistant organs (type 29) for poison damage reduction.
            if (base_type == 29u) {
                poison_resistant_count += 1u;
            }

            let props = get_amino_acid_properties(base_type);
            total_capacity += props.energy_storage;

            // Read previous frame's signals from input (agents_out is not pre-copied).
            let alpha = agents_in[agent_id].body[i].alpha_signal;
            let beta = agents_in[agent_id].body[i].beta_signal;

            // Modulate angle based on signals
            let alpha_effect = alpha * props.alpha_sensitivity * SIGNAL_GAIN * ANGLE_GAIN_ALPHA;
            let beta_effect = beta * props.beta_sensitivity * SIGNAL_GAIN * ANGLE_GAIN_BETA;
            var target_signal_angle = alpha_effect + beta_effect;
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

            // Accumulate center-of-mass in the same loop to avoid a second pass.
            com_sum += current_pos * m;
            mass_sum += m;

            // Check organ type once for both data field and _pad handling
            let is_vampire_mouth = (base_type == 33u);

            // Preserve organ internal state across frames by default.
            // Non-organs and vampire mouths may overwrite _pad below.
            agents_out[agent_id].body[i]._pad = agents_in[agent_id].body[i]._pad;

            // Data field: for vampire mouths (type 33), store packed world position for movement tracking
            if (is_vampire_mouth) {
                // Store current world position (agent.position + rotated local offset)
                let rotated_pos = apply_agent_rotation(current_pos, agent_rotation);
                let mouth_world_pos = agent_position + rotated_pos;
                agents_out[agent_id].body[i].data = pack_position_to_f32(mouth_world_pos, f32(SIM_SIZE));
            } else {
                agents_out[agent_id].body[i].data = 0.0;
            }
        }

        // Energy capacity
        agents_out[agent_id].energy_capacity = total_capacity;

        // Store total mass (only changes when morphology rebuilds)
        agents_out[agent_id].total_mass = max(total_mass_morphology, 0.05);

        // Store poison resistance count (only changes when morphology rebuilds)
        agents_out[agent_id].poison_resistant_count = poison_resistant_count;

        // Center of mass recentering
        let rec_n = body_count_val;
        if (rec_n > 0u) {
            let com = com_sum / max(mass_sum, 0.0001);

            // Calculate mass-weighted average angle.
            // IMPORTANT: tiny floating-point noise here can accumulate over long runs into slow
            // rotation creep, even when the morphology is effectively rigid. Deadzone it.
            var avg_angle = sum_angle_mass / max(total_mass_angle, 0.0001);
            if (abs(avg_angle) < 1e-4) {
                avg_angle = 0.0;
            }

            // Counteract internal rotation
            if (!DISABLE_GLOBAL_ROTATION) {
                agents_out[agent_id].rotation += avg_angle;
            }

            // Rotate body parts by -avg_angle
            let c_inv = cos(-avg_angle);
            let s_inv = sin(-avg_angle);

            for (var i = 0u; i < min(rec_n, MAX_BODY_PARTS); i++) {
                let p0 = agents_out[agent_id].body[i].pos - com;
                agents_out[agent_id].body[i].pos = vec2<f32>(
                    p0.x * c_inv - p0.y * s_inv,
                    p0.x * s_inv + p0.y * c_inv
                );
            }

            // Body part positions are stored in a COM-centered local frame (COM at 0,0).
            // We also store the chain origin (the start point before the first segment)
            // in the same COM-centered frame for rendering/segment-0 geometry.
            let o = -com;
            agents_out[agent_id].morphology_origin = vec2<f32>(
                o.x * c_inv - o.y * s_inv,
                o.x * s_inv + o.y * c_inv
            );
        }
    } else if (!first_build) {
        // OPTIMIZATION: Morphology rebuild skipped - just copy body positions from previous frame
        for (var i = 0u; i < min(body_count_val, MAX_BODY_PARTS); i++) {
            agents_out[agent_id].body[i] = agents_in[agent_id].body[i];
        }
        // Preserve essential values
        total_mass_morphology = agents_in[agent_id].total_mass;
        color_sum_morphology = agents_in[agent_id].body[0].data; // Approximation: stored in first part's data field if needed
    }

    // Cache previous frame's *already-recentered* positions.
    // This matches the stable orientation frame used by the inspector preview.
    if (!first_build) {
        for (var i = 0u; i < min(body_count_val, MAX_BODY_PARTS); i++) {
            prev_body_pos_recentered[i] = agents_in[agent_id].body[i].pos;
        }
    } else {
        for (var i = 0u; i < MAX_BODY_PARTS; i++) {
            prev_body_pos_recentered[i] = agents_out[agent_id].body[i].pos;
        }
    }

    // Morphology change energy: sum over segments of (mass * |delta_angle|).
    // Uses recentered + avg-angle-normalized local positions for stable comparison.
    if (!first_build && body_count_val > 1u && params.morphology_change_cost != 0.0) {
        var angle_mass_sum = 0.0;
        for (var i = 1u; i < min(body_count_val, MAX_BODY_PARTS); i++) {
            let old_vec = prev_body_pos_recentered[i] - prev_body_pos_recentered[i - 1u];
            let new_vec = agents_out[agent_id].body[i].pos - agents_out[agent_id].body[i - 1u].pos;

            // Ignore degenerate segments (shouldn't happen, but keeps atan2 stable).
            if (length(old_vec) < 1e-6 || length(new_vec) < 1e-6) {
                continue;
            }

            let old_ang = atan2(old_vec.y, old_vec.x);
            let new_ang = atan2(new_vec.y, new_vec.x);
            let dtheta = abs(wrap_angle_pi(new_ang - old_ang));

            let base_type = get_base_part_type(agents_out[agent_id].body[i].part_type);
            let props = get_amino_acid_properties(base_type);
            let m = max(props.mass, 0.01);

            angle_mass_sum += dtheta * m;
        }
        morphology_change_energy = angle_mass_sum * max(params.morphology_change_cost, 0.0);
    }

    // NOTE: morphology recentering may update agents_out.rotation (avg_angle compensation).
    // We intentionally do NOT sync `agent_rot` here: `agent_rot` is the agent's physical rotation
    // integrated from torque, while the recentering rotation is a morphology gauge correction.

    // Morphology-follow orientation (Option 2): estimate rigid rotation of the body between frames.
    // WARNING: even with a rigid body, tiny floating-point noise can yield a non-zero best-fit
    // rotation which accumulates into slow spin over many frames. We therefore:
    // - gate on a stricter correlation magnitude, and
    // - deadzone very small angles to zero.
    var morph_orient_delta = 0.0;
    if (MORPHOLOGY_ORIENT_FOLLOW_ENABLED && !first_build && body_count_val > 1u) {
        var old_com = vec2<f32>(0.0);
        var new_com = vec2<f32>(0.0);
        var wsum = 0.0;

        for (var i = 0u; i < min(body_count_val, MAX_BODY_PARTS); i++) {
            let base_type = get_base_part_type(agents_out[agent_id].body[i].part_type);
            let props = get_amino_acid_properties(base_type);
            let w = max(props.mass, 0.01);
            old_com += prev_body_pos[i] * w;
            new_com += agents_out[agent_id].body[i].pos * w;
            wsum += w;
        }

        if (wsum > 1e-6) {
            old_com = old_com / wsum;
            new_com = new_com / wsum;

            var Sxx = 0.0;
            var Sxy = 0.0;

            for (var i = 0u; i < min(body_count_val, MAX_BODY_PARTS); i++) {
                let base_type = get_base_part_type(agents_out[agent_id].body[i].part_type);
                let props = get_amino_acid_properties(base_type);
                let w = max(props.mass, 0.01);

                let p = prev_body_pos[i] - old_com;
                let q = agents_out[agent_id].body[i].pos - new_com;

                Sxx += (p.x * q.x + p.y * q.y) * w;
                Sxy += (p.x * q.y - p.y * q.x) * w;
            }

            // If the body is nearly isotropic/compact, the best-fit rotation is ill-defined.
            // Gate it out to avoid noisy spinning.
            let corr2 = Sxx * Sxx + Sxy * Sxy;
            // Extremely strict threshold: reduces numerical drift for rigid bodies.
            // If you enable MORPHOLOGY_ORIENT_FOLLOW for compact tumbling shapes, this gating
            // should still allow strong, well-defined rotations through.
            if (corr2 > 1e-2) {
                morph_orient_delta = atan2(Sxy, Sxx);
                // Deadzone tiny angles: prevents cumulative spin from numerical noise.
                if (abs(morph_orient_delta) < 1e-3) {
                    morph_orient_delta = 0.0;
                }
            }
        }
    }

    // Inject morphology-driven motion into the fluid.
    // IMPORTANT: use *internal deformation only* (local deltas), not rigid-body rotation/translation.
    // Otherwise, agents can "propel" just by rotating (or via recentering gauge), even when not waving.
    // NOTE: When FLUID_TWO_WAY_COUPLING_ENABLED is on, we perform a more physically grounded two-way
    // coupling inside the physics pass (per-part slip + reaction force). Disable this older one-way
    // deformation->fluid injection to avoid double-counting.
    // Deformation-based fluid injection: body shape changes stir the fluid.
    if (!FLUID_TWO_WAY_COUPLING_ENABLED && MORPHOLOGY_INJECT_FLUID_FORCE && !first_build && params.fluid_wind_push_strength != 0.0) {
        let strength = params.prop_wash_strength_fluid * MORPHOLOGY_FLUID_COUPLING;
        if (abs(strength) > 0.0) {
            let dt_safe = max(params.dt, 1e-3);
            let new_rot = agents_out[agent_id].rotation;

            // Remove any uniform shift from recentering by subtracting the mass-weighted mean local delta.
            var mean_delta_local = vec2<f32>(0.0);
            var wsum = 0.0;
            for (var i = 0u; i < min(body_count_val, MAX_BODY_PARTS); i++) {
                let base_type = get_base_part_type(agents_out[agent_id].body[i].part_type);
                let props = get_amino_acid_properties(base_type);
                let w = max(props.mass, 0.01);
                mean_delta_local += (agents_out[agent_id].body[i].pos - prev_body_pos[i]) * w;
                wsum += w;
            }
            if (wsum > 1e-6) {
                mean_delta_local = mean_delta_local / wsum;
            }

            for (var i = 0u; i < min(body_count_val, MAX_BODY_PARTS); i++) {
                let old_local = prev_body_pos[i];
                let new_local = agents_out[agent_id].body[i].pos;

                // World position of the part (for splatting), using the current output rotation.
                let new_world = agent_position + apply_agent_rotation(new_local, new_rot);

                // Deformation-only displacement in world space (per tick).
                // Rotate by the *physical* agent rotation from the start of the tick so rigid-body
                // rotation changes do not appear as deformation.
                var delta_local = (new_local - old_local) - mean_delta_local;
                let delta_len = length(delta_local);
                if (delta_len < 1e-4) {
                    continue;
                }

                var delta_world = apply_agent_rotation(delta_local, agent_rotation);
                let dlen = length(delta_world);
                if (dlen > MORPHOLOGY_MAX_WORLD_DELTA) {
                    delta_world = delta_world * (MORPHOLOGY_MAX_WORLD_DELTA / max(dlen, 1e-6));
                }

                // Convert displacement to a velocity-like vector.
                var v = delta_world / dt_safe;
                let vlen = length(v);
                if (vlen > MORPHOLOGY_MAX_WORLD_VEL) {
                    v = v * (MORPHOLOGY_MAX_WORLD_VEL / max(vlen, 1e-6));
                }

                // Option 5B: inject as a dipole (push–pull) to create a wake without adding net momentum.
                // Use deformation velocity direction; magnitude scales with |v|.
                let vdir = v;
                let vmag = length(v);

                // Choose a dipole separation based on the local body segment length.
                var seg_dir = vec2<f32>(1.0, 0.0);
                if (i > 0u) {
                    seg_dir = agents_out[agent_id].body[i].pos - agents_out[agent_id].body[i - 1u].pos;
                } else if (i + 1u < min(body_count_val, MAX_BODY_PARTS)) {
                    seg_dir = agents_out[agent_id].body[i + 1u].pos - agents_out[agent_id].body[i].pos;
                }
                let sep = clamp(length(seg_dir), 1.0, 6.0);

                let fmag = vmag * FLUID_FORCE_SCALE * 0.1 * strength;
                add_fluid_force_dipole(new_world, vdir, fmag, sep);
            }
        }
    }

    // Morphology-driven swimming is handled as per-part drag inside the physics pass.
    let MORPHOLOGY_SWIM_ENABLED = ms_enabled();
    let PROPELLERS_ENABLED = propellers_enabled();

    // Calculate agent color from color_sum accumulated during morphology rebuild
    let agent_color = vec3<f32>(
        sin(color_sum_morphology * 3.0) * 0.5 + 0.5,      // R: multiplier = 3.0
        sin(color_sum_morphology * 5.25) * 0.5 + 0.5,     // G: multiplier = 5.25
        sin(color_sum_morphology * 7.364) * 0.5 + 0.5     // B: multiplier = 7.364
    );

    // Repurpose torque_debug as a packed 24-bit RGB cache for neighbor color sensing.
    // Packed as 0x00RRGGBB (8 bits per channel). Stored as a numeric f32 so it round-trips exactly.
    let rgb8 = vec3<u32>(
        u32(clamp(agent_color.x, 0.0, 1.0) * 255.0 + 0.5),
        u32(clamp(agent_color.y, 0.0, 1.0) * 255.0 + 0.5),
        u32(clamp(agent_color.z, 0.0, 1.0) * 255.0 + 0.5)
    );
    let packed_rgb = (rgb8.x & 255u) | ((rgb8.y & 255u) << 8u) | ((rgb8.z & 255u) << 16u);
    agents_out[agent_id].torque_debug = f32(packed_rgb);

    let body_count = body_count_val; // Use computed value instead of reading from agent

    // ====== UNIFIED SIGNAL PROCESSING LOOP ======
    // Optimized passes: enabler discovery, amplification calculation, signal storage, and propagation

    var amplification_per_part: array<f32, MAX_BODY_PARTS>;
    var propeller_thrust_magnitude: array<f32, MAX_BODY_PARTS>; // Store activity magnitude for cost calculation (propeller or displacer)
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
        old_alpha[i] = agents_in[agent_id].body[i].alpha_signal;
        old_beta[i] = agents_in[agent_id].body[i].beta_signal;

        // Collect enabler positions
        if (props.is_inhibitor) { // enabler role
            enabler_positions[enabler_count] = part_i.pos;
            enabler_count += 1u;
        }
    }

    // ====== COLLECT NEARBY AGENTS ONCE (for sensors and physics) ======
    let scale = f32(SPATIAL_GRID_SIZE) / f32(SIM_SIZE);
    let my_grid_x = u32(clamp(agent_position.x * scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));
    let my_grid_y = u32(clamp(agent_position.y * scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));

    var neighbor_count = 0u;
    var neighbor_ids: array<u32, 64>; // Store up to 64 nearby agents

    // When the local window contains more than 64 eligible neighbors, keeping the first 64
    // in a fixed dy/dx scan order creates a directional bias in repulsion (dense blobs).
    // Use reservoir sampling so the final set is an unbiased subset of all eligible neighbors.
    var neighbor_seen = 0u;

    // Spiral scan (square rings) around the agent's cell.
    // This is more symmetric than row-major dy/dx and reduces structured bias.
    for (var radius: i32 = 0; radius <= 10; radius++) {
        if (radius == 0) {
            // Center cell
            let check_x = i32(my_grid_x);
            let check_y = i32(my_grid_y);
            if (check_x >= 0 && check_x < i32(SPATIAL_GRID_SIZE) &&
                check_y >= 0 && check_y < i32(SPATIAL_GRID_SIZE)) {
                let check_cell = u32(check_y) * SPATIAL_GRID_SIZE + u32(check_x);
                let stamp = atomicLoad(&agent_spatial_grid[spatial_epoch_index(check_cell)]);
                if (stamp == current_stamp) {
                    let raw_neighbor_id = atomicLoad(&agent_spatial_grid[spatial_id_index(check_cell)]);
                    let neighbor_id = raw_neighbor_id & 0x7FFFFFFFu;
                    if (neighbor_id != SPATIAL_GRID_EMPTY && neighbor_id != SPATIAL_GRID_CLAIMED && neighbor_id != agent_id) {
                        let neighbor_alive = agents_in[neighbor_id].alive;
                        let neighbor_energy = agents_in[neighbor_id].energy;
                        if (neighbor_alive == 1u && neighbor_energy != 0u) {
                            neighbor_seen++;
                            if (neighbor_count < 64u) {
                                neighbor_ids[neighbor_count] = neighbor_id;
                                neighbor_count++;
                            } else {
                                let seed = (agent_id * 747796405u) ^ (params.epoch * 2891336453u) ^ (params.random_seed * 196613u) ^ check_cell;
                                let r = hash(seed) % neighbor_seen;
                                if (r < 64u) {
                                    neighbor_ids[r] = neighbor_id;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Top and bottom edges
            for (var dx: i32 = -radius; dx <= radius; dx++) {
                // Top
                var check_x = i32(my_grid_x) + dx;
                var check_y = i32(my_grid_y) - radius;
                if (check_x >= 0 && check_x < i32(SPATIAL_GRID_SIZE) &&
                    check_y >= 0 && check_y < i32(SPATIAL_GRID_SIZE)) {
                    let check_cell = u32(check_y) * SPATIAL_GRID_SIZE + u32(check_x);
                    let stamp = atomicLoad(&agent_spatial_grid[spatial_epoch_index(check_cell)]);
                    if (stamp == current_stamp) {
                        let raw_neighbor_id = atomicLoad(&agent_spatial_grid[spatial_id_index(check_cell)]);
                        let neighbor_id = raw_neighbor_id & 0x7FFFFFFFu;
                        if (neighbor_id != SPATIAL_GRID_EMPTY && neighbor_id != SPATIAL_GRID_CLAIMED && neighbor_id != agent_id) {
                            let neighbor_alive = agents_in[neighbor_id].alive;
                            let neighbor_energy = agents_in[neighbor_id].energy;
                            if (neighbor_alive == 1u && neighbor_energy != 0u) {
                                neighbor_seen++;
                                if (neighbor_count < 64u) {
                                    neighbor_ids[neighbor_count] = neighbor_id;
                                    neighbor_count++;
                                } else {
                                    let seed = (agent_id * 747796405u) ^ (params.epoch * 2891336453u) ^ (params.random_seed * 196613u) ^ check_cell;
                                    let r = hash(seed) % neighbor_seen;
                                    if (r < 64u) {
                                        neighbor_ids[r] = neighbor_id;
                                    }
                                }
                            }
                        }
                    }
                }

                // Bottom
                check_x = i32(my_grid_x) + dx;
                check_y = i32(my_grid_y) + radius;
                if (check_x >= 0 && check_x < i32(SPATIAL_GRID_SIZE) &&
                    check_y >= 0 && check_y < i32(SPATIAL_GRID_SIZE)) {
                    let check_cell = u32(check_y) * SPATIAL_GRID_SIZE + u32(check_x);
                    let stamp = atomicLoad(&agent_spatial_grid[spatial_epoch_index(check_cell)]);
                    if (stamp == current_stamp) {
                        let raw_neighbor_id = atomicLoad(&agent_spatial_grid[spatial_id_index(check_cell)]);
                        let neighbor_id = raw_neighbor_id & 0x7FFFFFFFu;
                        if (neighbor_id != SPATIAL_GRID_EMPTY && neighbor_id != SPATIAL_GRID_CLAIMED && neighbor_id != agent_id) {
                            let neighbor_alive = agents_in[neighbor_id].alive;
                            let neighbor_energy = agents_in[neighbor_id].energy;
                            if (neighbor_alive == 1u && neighbor_energy != 0u) {
                                neighbor_seen++;
                                if (neighbor_count < 64u) {
                                    neighbor_ids[neighbor_count] = neighbor_id;
                                    neighbor_count++;
                                } else {
                                    let seed = (agent_id * 747796405u) ^ (params.epoch * 2891336453u) ^ (params.random_seed * 196613u) ^ check_cell;
                                    let r = hash(seed) % neighbor_seen;
                                    if (r < 64u) {
                                        neighbor_ids[r] = neighbor_id;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Left and right edges (excluding corners already covered)
            for (var dy: i32 = -radius + 1; dy <= radius - 1; dy++) {
                // Left
                var check_x = i32(my_grid_x) - radius;
                var check_y = i32(my_grid_y) + dy;
                if (check_x >= 0 && check_x < i32(SPATIAL_GRID_SIZE) &&
                    check_y >= 0 && check_y < i32(SPATIAL_GRID_SIZE)) {
                    let check_cell = u32(check_y) * SPATIAL_GRID_SIZE + u32(check_x);
                    let stamp = atomicLoad(&agent_spatial_grid[spatial_epoch_index(check_cell)]);
                    if (stamp == current_stamp) {
                        let raw_neighbor_id = atomicLoad(&agent_spatial_grid[spatial_id_index(check_cell)]);
                        let neighbor_id = raw_neighbor_id & 0x7FFFFFFFu;
                        if (neighbor_id != SPATIAL_GRID_EMPTY && neighbor_id != SPATIAL_GRID_CLAIMED && neighbor_id != agent_id) {
                            let neighbor_alive = agents_in[neighbor_id].alive;
                            let neighbor_energy = agents_in[neighbor_id].energy;
                            if (neighbor_alive == 1u && neighbor_energy != 0u) {
                                neighbor_seen++;
                                if (neighbor_count < 64u) {
                                    neighbor_ids[neighbor_count] = neighbor_id;
                                    neighbor_count++;
                                } else {
                                    let seed = (agent_id * 747796405u) ^ (params.epoch * 2891336453u) ^ (params.random_seed * 196613u) ^ check_cell;
                                    let r = hash(seed) % neighbor_seen;
                                    if (r < 64u) {
                                        neighbor_ids[r] = neighbor_id;
                                    }
                                }
                            }
                        }
                    }
                }

                // Right
                check_x = i32(my_grid_x) + radius;
                check_y = i32(my_grid_y) + dy;
                if (check_x >= 0 && check_x < i32(SPATIAL_GRID_SIZE) &&
                    check_y >= 0 && check_y < i32(SPATIAL_GRID_SIZE)) {
                    let check_cell = u32(check_y) * SPATIAL_GRID_SIZE + u32(check_x);
                    let stamp = atomicLoad(&agent_spatial_grid[spatial_epoch_index(check_cell)]);
                    if (stamp == current_stamp) {
                        let raw_neighbor_id = atomicLoad(&agent_spatial_grid[spatial_id_index(check_cell)]);
                        let neighbor_id = raw_neighbor_id & 0x7FFFFFFFu;
                        if (neighbor_id != SPATIAL_GRID_EMPTY && neighbor_id != SPATIAL_GRID_CLAIMED && neighbor_id != agent_id) {
                            let neighbor_alive = agents_in[neighbor_id].alive;
                            let neighbor_energy = agents_in[neighbor_id].energy;
                            if (neighbor_alive == 1u && neighbor_energy != 0u) {
                                neighbor_seen++;
                                if (neighbor_count < 64u) {
                                    neighbor_ids[neighbor_count] = neighbor_id;
                                    neighbor_count++;
                                } else {
                                    let seed = (agent_id * 747796405u) ^ (params.epoch * 2891336453u) ^ (params.random_seed * 196613u) ^ check_cell;
                                    let r = hash(seed) % neighbor_seen;
                                    if (r < 64u) {
                                        neighbor_ids[r] = neighbor_id;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // OPTIMIZATION: Skip signal diffusion if agent has no signal producers
    if (first_build || has_signal_producers) {
        // Second pass: calculate amplification and propagate signals (merged for efficiency)
        // Track cumulative chirality so directional env sensors can swap left/right.
        var chirality_flip_signal = 1.0;
        for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let part_pos = agents_out[agent_id].body[i].pos;
        let base_type = get_base_part_type(agents_out[agent_id].body[i].part_type);
        let amino_props = get_amino_acid_properties(base_type);

        if (base_type == 30u) {
            chirality_flip_signal = -chirality_flip_signal;
        }

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

        // Sensors:
        // - Environment sensors now read from the (diffused) dye layer deterministically.
        //   The old "sensor radius" is now interpreted as a signal gain multiplier.
        //   Actual env-sensor strength is derived from promoter+modifier param1 inside the sampling helpers.
        // - Neighbor/trail sensors still use a true spatial search radius.
        let env_sensor_gain_mult = 1.0;
        let neighbor_search_radius = 500.0;

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
        let perpendicular_world = normalize(apply_agent_rotation(perpendicular_local, agent_rotation));

        // ==============================================================
        // UNIFIED CHEMICAL SENSORS — "ionization" model
        // Types 22/38/39 emit into the alpha chain; 23/40/41 into beta.
        // Stochastic collision sampling within a radius: monomers of the
        // sensor's TARGET type build up charge, which fades every frame
        // (bacteria-style TEMPORAL chemotaxis — the charge dynamics as
        // the agent moves carry the direction, not a spatial gradient).
        // Variant (modifier 0-19) decodes:
        //   target monomer = modifier % 4  (A/U/G/C)
        //   falloff tier   = (modifier/4) % 5  (fast -> slow memory)
        // Sensitivity magnitude + polarity come from promoter+modifier
        // parameter1, as before.
        // ==============================================================
        let is_chem_sensor_alpha = amino_props.is_alpha_sensor || base_type == 38u || base_type == 39u;
        let is_chem_sensor_beta = amino_props.is_beta_sensor || base_type == 40u || base_type == 41u;
        if (is_chem_sensor_alpha || is_chem_sensor_beta) {
            let rotated_pos = apply_agent_rotation(agents_out[agent_id].body[i].pos, agent_rotation);
            let world_pos = agent_position + rotated_pos;

            let organ_param = get_organ_param(agents_out[agent_id].body[i].part_type);
            let modifier_index = u32((f32(organ_param) / 255.0) * 19.0);
            let promoter_param1 = amino_props.parameter1;
            let modifier_props = get_amino_acid_properties(modifier_index);
            let modifier_param1 = modifier_props.parameter1;

            let target_ch = modifier_index % 4u;
            let falloff_tier = (modifier_index / 4u) % 5u;
            var ion_decay = 0.85;
            if (falloff_tier == 0u) { ion_decay = 0.50; }
            else if (falloff_tier == 1u) { ion_decay = 0.72; }
            else if (falloff_tier == 3u) { ion_decay = 0.93; }
            else if (falloff_tier == 4u) { ion_decay = 0.97; }

            let combined = promoter_param1 + modifier_param1;
            let polarity = select(1.0, -1.0, combined < 0.0);
            let gain = abs(combined);

            // Stochastic collisions within the sensing radius.
            var hits = 0.0;
            let base_seed = (agent_id * 1000u + i * 13u) ^ (params.random_seed * 747796405u);
            for (var s = 0u; s < 4u; s++) {
                let hs = hash(base_seed ^ (s * 2654435761u));
                let ang = hash_f32(hs) * 6.2831853;
                let rad = sqrt(hash_f32(hs ^ 0x9E3779B9u)) * SENSOR_ION_RADIUS;
                let sp = world_pos + vec2<f32>(cos(ang), sin(ang)) * rad;
                if (is_in_bounds(sp)) {
                    hits += f32(chem_act_count(grid_index(sp), target_ch)); // senses ACTIVATED (food) only
                }
            }

            // Ionization charge: accumulates with collisions, fades over time.
            var charge = max(agents_out[agent_id].body[i]._pad.x, 0.0);
            charge = charge * ion_decay + hits * SENSOR_CHARGE_PER_HIT;
            agents_out[agent_id].body[i]._pad.x = charge;

            let emitted = sqrt(clamp(charge * gain, 0.0, 1.0)) * polarity;
            if (is_chem_sensor_alpha) {
                new_alpha = new_alpha + emitted;
            } else {
                new_beta = new_beta + emitted;
            }
        }

        // TRAIL ENERGY SENSOR - Senses nearby agent energies from trail
        if (amino_props.is_trail_energy_sensor) {
            let rotated_pos = apply_agent_rotation(agents_out[agent_id].body[i].pos, agent_rotation);
            let world_pos = agent_position + rotated_pos;

            // Get sensor orientation (perpendicular to organ)
            let axis_local = normalize(agents_out[agent_id].body[i].pos);
            let perpendicular_local = normalize(vec2<f32>(-axis_local.y, axis_local.x));
            let perpendicular_world = normalize(apply_agent_rotation(perpendicular_local, agent_rotation));

            let sensed_value = sample_neighbors_energy(world_pos, neighbor_search_radius, params.debug_mode != 0u, perpendicular_world, &neighbor_ids, neighbor_count);
            // Normalize by a scaling factor (energy can be large, scale to -1..1 range)
            let normalized_value = tanh(sensed_value * 0.01); // tanh for soft clamping to -1..1
            // Split into alpha and beta based on sign (positive energy -> alpha, negative -> beta)
            new_alpha += max(normalized_value, 0.0);
            new_beta += max(-normalized_value, 0.0);
        }

        // (Magnitude sensor types 38-41 are folded into the unified
        //  ionization sensor above — same model, beta-family variants just
        //  emit into the beta chain.)

        // AGENT ALPHA SENSOR - Organ type 34 (V/M + L)
        // Senses nearby agent colors from trail
        if (base_type == 34u) {
            let rotated_pos = apply_agent_rotation(agents_out[agent_id].body[i].pos, agent_rotation);
            let sensor_world_pos = agent_position + rotated_pos;

            // Get sensor orientation (perpendicular to organ, rotated 90 degrees)
            let axis_local = normalize(agents_out[agent_id].body[i].pos);
            let perpendicular_local = normalize(vec2<f32>(-axis_local.y, axis_local.x));
            let perpendicular_world = normalize(apply_agent_rotation(perpendicular_local, agent_rotation));

            // Use agent_color calculated from color_sum_morphology
            let sensed_value = sample_neighbors_color(sensor_world_pos, neighbor_search_radius, params.debug_mode != 0u, perpendicular_world, agent_color, &neighbor_ids, neighbor_count);

            // Add agent color difference signal to alpha
            new_alpha += sensed_value;
        }

        // AGENT BETA SENSOR - Organ type 35 (V/M + Y)
        // Senses nearby agent colors from trail
        if (base_type == 35u) {
            let rotated_pos = apply_agent_rotation(agents_out[agent_id].body[i].pos, agent_rotation);
            let sensor_world_pos = agent_position + rotated_pos;

            // Get sensor orientation (perpendicular to organ, rotated 90 degrees)
            let axis_local = normalize(agents_out[agent_id].body[i].pos);
            let perpendicular_local = normalize(vec2<f32>(-axis_local.y, axis_local.x));
            let perpendicular_world = normalize(apply_agent_rotation(perpendicular_local, agent_rotation));

            // Use agent_color calculated from color_sum_morphology
            let sensed_value = sample_neighbors_color(sensor_world_pos, neighbor_search_radius, params.debug_mode != 0u, perpendicular_world, agent_color, &neighbor_ids, neighbor_count);

            // Add agent color difference signal to beta
            new_beta += sensed_value;
        }

        // PAIRING STATE SENSOR - Organ type 36 (H/Q + I/K)
        // Emits alpha or beta based on genome pairing completion percentage
        if (base_type == 36u) {
            // Get pairing percentage (0.0 to 1.0)
            let pairing_percentage = f32(agent_pairing_counter) / f32(GENOME_BYTES);

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
            let energy_t = clamp(agent_energy / 50.0, 0.0, 1.0);
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
                // Use _pad.x to accumulate sensed field over time (NOT used by prev_pos for organs)
                var internal_value = agents_out[agent_id].body[i]._pad.x;
                internal_value += sensed_field;
                agents_out[agent_id].body[i]._pad.x = internal_value;

                let x = internal_value * param1;
                clock_signal = sin(x);
            }

            // Store clock signal in _pad.y for rendering (avoids conflict with prev_pos)
            agents_out[agent_id].body[i]._pad.y = clock_signal;

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
            // GAMMA SENSOR (was slope sensor): the unified ionization model
            // applied to the quantized inert mass — stochastic collision
            // samples within the sensing radius accumulate charge from gamma
            // quanta; the charge fades at the variant's falloff rate and is
            // emitted into alpha (K promoter) or beta (C promoter). With
            // solid terrain as excluded volume, "smelling rock" is what a
            // digger or wall-follower needs.
            let world_pos = agent_pos + apply_agent_rotation(part_pos, agent_rot);

            let organ_param = get_organ_param(agents_out[agent_id].body[i].part_type);
            let is_C_promoter = ((organ_param & 128u) != 0u);
            let modifier_param1 = f32(organ_param & 127u) / 127.0;
            let g_gain = modifier_param1 + amino_props.parameter1;
            let g_polarity = select(1.0, -1.0, g_gain < 0.0);
            let g_mag = abs(g_gain);

            // Falloff tier from modifier bits 0-1 (fast -> slow memory).
            let g_tier = organ_param & 3u;
            var g_decay = 0.72;
            if (g_tier == 1u) { g_decay = 0.85; }
            else if (g_tier == 2u) { g_decay = 0.93; }
            else if (g_tier == 3u) { g_decay = 0.97; }

            var g_hits = 0.0;
            let gseed = (agent_id * 1000u + i * 29u) ^ (params.random_seed * 747796405u);
            for (var s = 0u; s < 4u; s++) {
                let hs = hash(gseed ^ (s * 2654435761u));
                let ang = hash_f32(hs) * 6.2831853;
                let rad = sqrt(hash_f32(hs ^ 0x9E3779B9u)) * SENSOR_ION_RADIUS;
                let sp = world_pos + vec2<f32>(cos(ang), sin(ang)) * rad;
                if (is_in_bounds(sp)) {
                    g_hits += f32(gamma_count(grid_index(sp)));
                }
            }

            var g_charge = max(agents_out[agent_id].body[i]._pad.x, 0.0);
            g_charge = g_charge * g_decay + g_hits * SENSOR_CHARGE_PER_HIT;
            agents_out[agent_id].body[i]._pad.x = g_charge;

            let g_emitted = sqrt(clamp(g_charge * g_mag, 0.0, 1.0)) * g_polarity;
            if (is_C_promoter) {
                new_beta = new_beta + g_emitted;
            } else {
                new_alpha = new_alpha + g_emitted;
            }
        }

        // INTERNAL SIGNAL EMITTERS
        // Emits a constant internal signal scaled by enabler proximity (amplification).
        // 25 (former Displacer A): +alpha for modifiers M/N, -alpha for P/Q/R
        // 27 (former Displacer B): +beta for modifier S, -beta for T/V
        if (amino_props.is_signal_emitter) {
            let amp_emit = amplification_per_part[i];
            if (amp_emit > 0.0 && agent_energy >= amino_props.energy_consumption) {
                let organ_param = get_organ_param(agents_out[agent_id].body[i].part_type);
                let modifier_index = u32((f32(organ_param) / 255.0) * 19.0);

                let emit_base = amp_emit * 0.6;

                if (base_type == 25u) {
                    let emit = select(emit_base, -emit_base, modifier_index >= 12u);
                    new_alpha += emit;
                } else if (base_type == 27u) {
                    let emit = select(emit_base, -emit_base, modifier_index >= 16u);
                    new_beta += emit;
                }

                // Track emitter activity so activity-based energy cost remains meaningful.
                // NOTE: emitters do not have meaningful thrust_force (often 0), so store a normalized
                // activity ratio directly to avoid 0/0 NaNs in the energy cost path.
                propeller_thrust_magnitude[i] = clamp(amp_emit, 0.0, 1.0);
            }
        }

        // Apply signal decay (requested): global decay for all parts (amino acids + organs).
        new_alpha *= 0.997;
        new_beta *= 0.997;

        // Smooth internal signal changes to prevent sudden oscillations (75% new, 25% old)
        let update_rate = 0.75;
        let smoothed_alpha = mix(old_alpha[i], new_alpha, update_rate);
        let smoothed_beta = mix(old_beta[i], new_beta, update_rate);

        // Clamp to -1.0 to 1.0 (allows inhibitory and excitatory signals)
        agents_out[agent_id].body[i].alpha_signal = clamp(smoothed_alpha, -1.0, 1.0);
        agents_out[agent_id].body[i].beta_signal = clamp(smoothed_beta, -1.0, 1.0);
    }
    } else {
        // OPTIMIZATION: Signal diffusion skipped - preserve signals from previous frame
        for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
            agents_out[agent_id].body[i].alpha_signal = agents_in[agent_id].body[i].alpha_signal;
            agents_out[agent_id].body[i].beta_signal = agents_in[agent_id].body[i].beta_signal;
        }
    }

    // ====== PHYSICS CALCULATIONS ======
    // Agent already centered at local (0,0) after morphology re-centering
    let center_of_mass = vec2<f32>(0.0);
    // total_mass was calculated during morphology, but Anchor (type 42) can temporarily
    // increase effective mass when active.
    var total_mass = agents_out[agent_id].total_mass;
    // Body positions are COM-centered; morphology_origin is the chain origin in the same frame.
    let morphology_origin = agents_out[agent_id].morphology_origin;

    // Anchor mass multiplier when active (stored in part._pad.y as 0..1 strength).
    let ANCHOR_MASS_MULT = 100.0;
    var anchor_mass_extra = 0.0;
    // Rough surface-area estimate from part geometry (length * thickness).
    var total_area = 0.0;
    // Extra drag boost from wide/open appendages (vamp mouths, spikes).
    // This intentionally affects locomotion by increasing effective drag, without changing
    // the base thrust model for normal microswimming.
    var vamp_spike_drag_boost = 0.0;
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let part_out = agents_out[agent_id].body[i];
        let base_type_out = get_base_part_type(part_out.part_type);

        total_area += get_part_area_estimate(part_out.part_type);

        // Open vampire mouths (33) and spikes (46) add extra effective area.
        // This area contribution fades out as the organ closes.
        if (base_type_out == 33u || base_type_out == 46u) {
            let open = clamp(part_out._pad.y, 0.0, 1.0);
            var r = DEFENSE_RADIUS_DEFAULT;
            if (i + 1u < body_count) {
                let next_base = get_base_part_type(agents_out[agent_id].body[i + 1u].part_type);
                r = param1_to_defense_radius(get_amino_acid_properties(next_base).parameter1);
            }
            // Empirical 2D proxy: opened fan area scales ~r^2.
            let open_area_proxy = open * (r * r);
            total_area += open_area_proxy * 0.02;
            // Strong additional drag for open mouths/spikes to prevent huge vamps from
            // "flying" faster than smaller agents.
            vamp_spike_drag_boost += open_area_proxy;
        }

        if (base_type_out == 42u) {
            let props = get_amino_acid_properties(base_type_out);
            let base_mass = max(props.mass, 0.01);
            let anchor_strength = clamp(part_out._pad.y, 0.0, 1.0);
            anchor_mass_extra += base_mass * (ANCHOR_MASS_MULT - 1.0) * anchor_strength;
        }
    }
    total_mass = max(total_mass + anchor_mass_extra, 0.05);
    agents_out[agent_id].total_mass = total_mass;

    total_area = max(total_area, 1e-3);
    let area_response = clamp(total_area / max(total_mass, 1e-6), 0.1, 10.0);

    // Anchor (type 42): if any anchor was active last frame, freeze the whole agent.
    // Read from agents_in to avoid same-frame ordering dependence.
    var anchor_active_prev = false;
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let part_in = agents_in[agent_id].body[i];
        let base_type_in = get_base_part_type(part_in.part_type);
        if (base_type_in == 42u && part_in._pad.y > 0.5) {
            anchor_active_prev = true;
            break;
        }
    }

    // Drag coefficient: base on mass, but scale by area response.
    // Open vampire mouths/spikes increase total_area, which increases drag and slows locomotion.
    // Apply a *much stronger* drag multiplier from open mouths/spikes, clamped for stability.
    let vamp_spike_drag_mult = clamp(1.0 + vamp_spike_drag_boost * 0.05, 1.0, 30.0);
    let drag_coefficient = total_mass * 0.5 * area_response * vamp_spike_drag_mult;
    let dt_safe = max(params.dt, 1e-3);

    // Strength for real-fluid two-way coupling (reuses existing UI knob).
    let fluid_two_way_strength = max(params.prop_wash_strength_fluid, 0.0) * MORPHOLOGY_FLUID_COUPLING;

    // Morphology swim strength (reuses the existing prop-wash UI knobs as a convenient control).
    // This is applied as per-part drag later (no direct angular-velocity injection).
    let morph_swim_strength = select(
        0.0,
        max(max(params.prop_wash_strength, 0.0), max(params.prop_wash_strength_fluid, 0.0)) * MORPHOLOGY_SWIM_COUPLING,
        (MORPHOLOGY_SWIM_ENABLED && !first_build)
    );

    // Accumulate forces and torques (relative to CoM)
    var force = vec2<f32>(0.0);
    var torque = 0.0;

    // Hard collision (minimum inter-agent separation).
    // Note: this uses the pre-collected neighbor list (max 64, one per spatial cell).
    // It's intended as a robust anti-clumping constraint even when repulsion is disabled.
    // Mass-dependent: larger (heavier) agents keep a larger minimum separation.
    let HARD_COLLISION_RADIUS_BASE: f32 = 10.0;
    let HARD_COLLISION_EPS: f32 = 1e-3;
    var hard_collision_vel = vec2<f32>(0.0);

    // (No global morphology-swim impulse here; handled per-part.)

    // Agent-to-agent repulsion (simplified: once per agent pair, using total masses)
    for (var n = 0u; n < neighbor_count; n++) {
        // IMPORTANT: read neighbors from agents_in (stable snapshot).
        // Reading from agents_out here creates a cross-invocation race (other agents are
        // writing their own agents_out entries concurrently), which can introduce artifacts
        // like directional drift/banding.
        let neighbor_id = neighbor_ids[n];
        // Avoid copying full neighbor Agent (includes body array).
        let neighbor_pos = agents_in[neighbor_id].position;
        let neighbor_mass = max(agents_in[neighbor_id].total_mass, 0.01);

        let delta = agent_pos - neighbor_pos;
        let dist = length(delta);

        // Mass-dependent separation scale based on the pair's average mass.
        // Using sqrt keeps growth sublinear (area/diameter proxy) and clamping avoids extremes.
        let pair_mass_avg = 0.5 * (total_mass + neighbor_mass);
        let mass_scale = clamp(sqrt(clamp(pair_mass_avg, 0.05, 50.0)), 0.6, 2.5);
        let hard_collision_radius = HARD_COLLISION_RADIUS_BASE * mass_scale;

        // Hard minimum distance: accumulate a per-tick positional correction as an
        // extra velocity term (applied after smoothing later).
        if (dist < hard_collision_radius) {
            let penetration = hard_collision_radius - dist;
            var direction = vec2<f32>(0.0);
            if (dist > HARD_COLLISION_EPS) {
                direction = delta / dist;
            } else {
                // Deterministic fallback direction for exact overlaps.
                let seed = (agent_id * 747796405u) ^ (neighbor_id * 2891336453u) ^ (params.epoch * 196613u) ^ (params.random_seed * 83492791u);
                let h = hash(seed);
                let angle = (f32(h & 4095u) / 4096.0) * 6.28318530718;
                direction = vec2<f32>(cos(angle), sin(angle));
            }
            // Split correction across both agents (approx; each agent applies its own).
            hard_collision_vel += direction * (penetration * 0.5);
        }

        // Distance-based repulsion force (inverse square law with cutoff)
        let max_repulsion_distance = 500.0 * mass_scale;

        if (dist < max_repulsion_distance && dist > 0.1) {
            // Inverse square repulsion: F = k / (d^2)
            let strength_scale = clamp(pair_mass_avg, 0.5, 3.0);
            let base_strength = params.agent_repulsion_strength * 100000.0 * strength_scale;
            // SOFTENED inverse square (Plummer, r0 = 10 = collision radius):
            // the bare 1/d^2 hit the 5000 cap at birth distances, and a
            // clump of newborns SUMMED several capped kicks into a leap.
            // Newborns also ramp repulsion in over their first 60 frames.
            let soft_d2 = dist * dist + 100.0;
            let grace_r = clamp(f32(agents_in[agent_id].age) / 60.0, 0.1, 1.0);
            let force_magnitude = grace_r * base_strength / soft_d2;

            // Clamp to prevent extreme forces at very small distances
            let clamped_force = min(abs(force_magnitude), 1000.0) * sign(force_magnitude);

            let direction = delta / dist; // Normalize
            force += direction * clamped_force;
        }
    }

    // Now calculate forces using the updated morphology (using pre-collected neighbors)

    // ===== VIRTUAL SWIM (DEFORMATION-ONLY, ANISOTROPIC DRAG) =====
    // Compute thrust from pre/post morphology local positions in the *stable normalized frame*.
    // Because both prev and current locals are already recentered and avg-angle-normalized,
    // the pure deformation signal is simply (new_local - old_local).
    if (!MORPHOLOGY_SWIM_SEPARATE_PASS && MORPHOLOGY_SWIM_ENABLED && morph_swim_strength > 0.0 && !first_build && body_count > 1u && !anchor_active_prev) {
        var thrust_world = vec2<f32>(0.0);
        var total_weight = 0.0;

        let c_par = MORPHOLOGY_SWIM_BASE_DRAG;
        let c_perp = MORPHOLOGY_SWIM_BASE_DRAG * MORPHOLOGY_SWIM_ANISOTROPY;

        // Iterate true segments between consecutive parts.
        // RFT (low-Re swimming): net drag/thrust scales with filament length (ds), not mass.
        // We therefore weight each segment contribution by its current geometric length.
        for (var j = 1u; j < min(body_count, MAX_BODY_PARTS); j++) {

            // Segment endpoints (local), previous and current.
            let seg_start_prev = prev_body_pos_recentered[j - 1u];
            let seg_end_prev = prev_body_pos_recentered[j];
            let seg_start_new = agents_out[agent_id].body[j - 1u].pos;
            let seg_end_new = agents_out[agent_id].body[j].pos;

            // Pure deformation displacement in the stable local frame.
            let delta_start_local = seg_start_new - seg_start_prev;
            let delta_end_local = seg_end_new - seg_end_prev;

            // Segment midpoint deformation velocity (world units per tick)
            let v_seg_local = (delta_start_local + delta_end_local) * 0.5;
            var v_seg_world = apply_agent_rotation(v_seg_local, agent_rot);

            // Clamp huge jumps (e.g. from abrupt morphology changes)
            let dlen = length(v_seg_world);
            if (dlen > MORPHOLOGY_MAX_WORLD_DELTA) {
                v_seg_world = v_seg_world * (MORPHOLOGY_MAX_WORLD_DELTA / max(dlen, 1e-6));
            }
            v_seg_world = v_seg_world / dt_safe;
            let vlen = length(v_seg_world);
            if (vlen > MORPHOLOGY_MAX_WORLD_VEL) {
                v_seg_world = v_seg_world * (MORPHOLOGY_MAX_WORLD_VEL / max(vlen, 1e-6));
            }

            // Segment tangent direction (defines parallel/perp axes)
            let seg_vec_new = seg_end_new - seg_start_new;
            let seg_vec_len = length(seg_vec_new);
            if (seg_vec_len < 1e-4) {
                continue;
            }
            let seg_len_world = seg_vec_len;
            let tangent_local = seg_vec_new / seg_vec_len;
            let tangent_world = normalize(apply_agent_rotation(tangent_local, agent_rot));

            // Decompose velocity into parallel and perpendicular components
            let v_parallel = dot(v_seg_world, tangent_world) * tangent_world;
            let v_perp = v_seg_world - v_parallel;

            // Anisotropic drag per unit length; multiply by segment length (ds) to get force.
            let drag_per_len = (-c_par * v_parallel) + (-c_perp * v_perp);

            // Reaction thrust on agent = -drag on "fluid".
            let segment_thrust = -drag_per_len * seg_len_world;

            thrust_world += segment_thrust;
            total_weight += seg_len_world;
        }

        if (total_weight > 1e-6) {
            // Average thrust per unit length (keeps behavior consistent across different body sizes).
            thrust_world = thrust_world / total_weight;

            // Convert to a desired velocity contribution and clamp.
            var final_thrust_vel = thrust_world * morph_swim_strength;
            let tl = length(final_thrust_vel);
            if (tl > MORPHOLOGY_SWIM_MAX_FRAME_VEL) {
                final_thrust_vel = final_thrust_vel * (MORPHOLOGY_SWIM_MAX_FRAME_VEL / max(tl, 1e-6));
            }

            // Overdamped regime: velocity = force / drag
            force += final_thrust_vel * drag_coefficient;
        }
    }

    // Water-frame drag accumulators: applied once after the loop in a
    // stable exponential form (see below).
    var wind_k = 0.0;
    var wind_target = vec2<f32>(0.0);

    var chirality_flip_physics = 1.0; // Track cumulative chirality for propeller direction
    // Accumulate rotational inertia and detect vampire/beta mouths in this same pass
    // to avoid extra full body scans.
    var moment_of_inertia = 0.0;
    var has_vampire_mouth = false;
    var has_beta_mouth = false;
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let part = agents_out[agent_id].body[i];

        // Get amino acid properties
        let base_type = get_base_part_type(part.part_type);
        let amino_props = get_amino_acid_properties(base_type);

        if (base_type == 33u) {
            has_vampire_mouth = true;
        }
        if (base_type == 44u) {
            has_beta_mouth = true;
        }

        // Check if this part is Leucine (index 9) and flip chirality
        if (base_type == 9u) {
            chirality_flip_physics = -chirality_flip_physics;
        }

        // Calculate segment midpoint for force application and torque
        var segment_start_chain = morphology_origin;
        if (i > 0u) {
            segment_start_chain = agents_out[agent_id].body[i - 1u].pos;
        }
        let segment_midpoint_chain = (segment_start_chain + part.pos) * 0.5;
        let segment_midpoint = segment_midpoint_chain;

        // Use midpoint for physics calculations
        let offset_from_com = segment_midpoint - center_of_mass;

        // Moment of inertia (rotation-invariant; can be computed in local space)
        let anchor_strength = select(0.0, clamp(part._pad.y, 0.0, 1.0), base_type == 42u);
        let anchor_active = anchor_strength > 0.5;
        let base_part_mass = max(amino_props.mass, 0.01);
        let part_mass = base_part_mass * (1.0 + anchor_strength * (ANCHOR_MASS_MULT - 1.0));
        moment_of_inertia += part_mass * dot(offset_from_com, offset_from_com);
        let r_com = apply_agent_rotation(offset_from_com, agent_rot);
        let rotated_midpoint = apply_agent_rotation(segment_midpoint, agent_rot);
        let world_pos = agent_pos + rotated_midpoint;

        let part_weight = part_mass / max(total_mass, 1e-6);
        let part_area_weight = get_part_area_estimate(part.part_type) / total_area;

        // (Virtual swim thrust is computed once per-agent above; avoid double-counting per-part.)

        // ==================================================================
        // QUANTIZED TERRAIN COLLISION (replaces slope-gradient forces)
        // Gamma cells holding >= GAMMA_SOLID_THRESHOLD quanta are SOLID:
        // embedded parts are ejected toward open space and braked; parts
        // beside walls feel repulsion; 1..THRESH-1 quanta are rubble
        // (viscous drag). Pressing into solid rock can dislodge quanta to
        // the rear (slow tunneling), so digging species remain possible.
        // Anchors (type 42) opt out, as they did for slope forces.
        // ==================================================================
        if (!anchor_active) {
            // Global vector force (was previously folded into the slope path).
            if (params.vector_force_power > 0.0) {
                let gravity_force = vec2<f32>(
                    params.vector_force_x * params.vector_force_power,
                    params.vector_force_y * params.vector_force_power
                ) * part_mass;
                force += gravity_force;
                torque += (r_com.x * gravity_force.y - r_com.y * gravity_force.x);
            }

            let t_idx = grid_index(world_pos);
            let g_here = gamma_count(t_idx);

            var solid_push = vec2<f32>(0.0);
            if (gamma_count(grid_index(world_pos + vec2<f32>( 1.0, 0.0))) >= GAMMA_SOLID_THRESHOLD) { solid_push.x -= 1.0; }
            if (gamma_count(grid_index(world_pos + vec2<f32>(-1.0, 0.0))) >= GAMMA_SOLID_THRESHOLD) { solid_push.x += 1.0; }
            if (gamma_count(grid_index(world_pos + vec2<f32>(0.0,  1.0))) >= GAMMA_SOLID_THRESHOLD) { solid_push.y -= 1.0; }
            if (gamma_count(grid_index(world_pos + vec2<f32>(0.0, -1.0))) >= GAMMA_SOLID_THRESHOLD) { solid_push.y += 1.0; }

            if (g_here >= GAMMA_SOLID_THRESHOLD) {
                // Embedded in solid: eject toward open space and brake hard.
                var eject = solid_push;
                if (dot(eject, eject) < 0.25) {
                    // Fully surrounded: pick a hashed escape direction.
                    let eh = hash((agent_id + 1u) * 31u ^ (i * 67u) ^ params.random_seed);
                    let ea = hash_f32(eh) * 6.2831853;
                    eject = vec2<f32>(cos(ea), sin(ea));
                }
                let ej = normalize(eject) * GAMMA_EJECT_FORCE * part_mass;
                force += ej;
                torque += (r_com.x * ej.y - r_com.y * ej.x);
                force -= agent_velocity * GAMMA_SOLID_BRAKE * part_mass;

                // Pressure digging: dislodge one quantum to the part's rear.
                let spd_p = length(agent_velocity);
                if (spd_p > 0.02) {
                    let dseed3 = (agent_id + 1u) * 151u ^ (i * 271u) ^ (params.random_seed * 631u);
                    if (hash_f32(dseed3) < DIG_P_PER_FRAME * max(part_mass, 0.1)) {
                        let behind_p = world_pos - (agent_velocity / spd_p) * 2.0;
                        if (is_in_bounds(behind_p)) {
                            let _gm = gamma_move_one(t_idx, grid_index(behind_p));
                        }
                    }
                }
            } else {
                if (g_here > 0u) {
                    // Rubble: viscous drag proportional to occupancy.
                    force -= agent_velocity * GAMMA_RUBBLE_DRAG * f32(g_here) * part_mass;
                }
                if (dot(solid_push, solid_push) > 0.25) {
                    // Beside a wall: soft repulsion keeps parts out of it.
                    let wp = normalize(solid_push) * GAMMA_WALL_FORCE * part_mass;
                    force += wp;
                    torque += (r_com.x * wp.y - r_com.y * wp.x);
                }
            }
        }

        // One-way fluid->agent wind push is replaced by FLUID_COUPLING_SIMPLE_ENABLED above.
        if (!FLUID_COUPLING_SIMPLE_ENABLED && !anchor_active && params.fluid_wind_push_strength != 0.0) {
            // Default body parts (amino acids + organs) to coupling=1.0 unless explicitly overridden.
            let wind_coupling = select(
                amino_props.fluid_wind_coupling,
                0.5,
                (amino_props.fluid_wind_coupling == 0.0)
            );

            if (wind_coupling != 0.0) {
                // Avoid clamping OOB into edge cells (prevents artificial edge wind).
                if (world_pos.x >= 0.0 && world_pos.x < f32(SIM_SIZE) && world_pos.y >= 0.0 && world_pos.y < f32(SIM_SIZE)) {
                    // Full visual-flow riding (see fluid_ride_velocity).
                    let v_frame = fluid_ride_velocity(world_pos);

                    // WATER FRAME: viscous drag toward the fluid's velocity.
                    // The torque keeps the explicit per-part form (angular
                    // velocity is clamped elsewhere); the LINEAR drag is only
                    // accumulated here and applied once after the loop in an
                    // unconditionally stable exponential form. The old path
                    // pushed v_rel*k through the overdamped velocity blend --
                    // explicit Euler, whose effective gain exceeded 2 for
                    // light (few-part) agents: their velocity flipped sign
                    // every epoch with slowly growing amplitude, rendering as
                    // agents blinking between two fixed positions.
                    // NOTE: no area_response here — it reached 10x for light
                    // agents and made them snap to the water velocity every
                    // frame, erasing microswim thrust. Light-vs-heavy
                    // differentiation comes from mass_smoothing at the
                    // application site. With slider=1 a light agent relaxes
                    // ~35%/epoch toward the current; a heavy one ~5%.
                    let k_part = wind_coupling * part_area_weight * params.fluid_wind_push_strength;
                    let v_rel = v_frame - agent_velocity;
                    let wind_force = v_rel * k_part * drag_coefficient;
                    torque += (r_com.x * wind_force.y - r_com.y * wind_force.x);
                    wind_k += k_part;
                    wind_target += v_frame * k_part;
                }
            }
        }

        // Cached amplification for this part (organs will use it, organs may ignore)
        let amplification = amplification_per_part[i];



    // Propeller force - check if this amino acid provides thrust
    // Propellers only work if agent has enough energy to cover their consumption cost
    if (PROPELLERS_ENABLED && amino_props.is_propeller && agent_energy_cur >= amino_props.energy_consumption) {
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
            let thrust_dir_world = apply_agent_rotation(thrust_local, agent_rot);

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
                    // Throw distance in cells for the direct (fluidless) prop wash.
                    // Increased range to make jets move material farther.
                    let distance = clamp(prop_strength * 4.0, 1.0, 10.0);
                    let target_world = clamped_pos + prop_dir * distance * grid_scale;
                    // IMPORTANT: keep mapping consistent with center cell selection.
                    // Using round() here creates a systematic half-cell (+0.5,+0.5) bias -> diagonal (45°) artifacts.
                    let target_gx = clamp(i32(target_world.x / grid_scale), 0, i32(GRID_SIZE) - 1);
                    let target_gy = clamp(i32(target_world.y / grid_scale), 0, i32(GRID_SIZE) - 1);
                    let target_idx = u32(target_gy) * GRID_SIZE + u32(target_gx);

                    if (target_idx != center_idx) {
                        // Quantized gamma wash: probabilistically shove one
                        // whole quantum (prob = transfer / quantum size).
                        let transfer_amount = prop_strength * 0.05 * part_weight;
                        if (transfer_amount > 0.0 && gamma_count(target_idx) < 40u) {
                            let p_move = clamp(transfer_amount / GAMMA_QUANTUM, 0.0, 1.0);
                            let wseed = (agent_id + 1u) * 193u ^ (i * 389u) ^ (params.random_seed * 769u);
                            if (hash_f32(wseed) < p_move) {
                                let _gm = gamma_move_one(center_idx, target_idx);
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
                if (PROPELLERS_APPLY_DIRECT_FORCE) {
                    force += thrust_force;
                    // Torque from lever arm (propeller organ position) cross thrust.
                    // Using the segment midpoint lever arm here can severely reduce torque
                    // compared to applying force at the organ itself.
                    let prop_offset_from_com = part.pos - center_of_mass;
                    let prop_r_com = apply_agent_rotation(prop_offset_from_com, agent_rot);
                    torque += (prop_r_com.x * thrust_force.y - prop_r_com.y * thrust_force.x) * (6.0 * PROP_TORQUE_COUPLING);
                }

                // INJECT PROPELLER FORCE INTO FLUID AS A DIPOLE (push–pull)
                // NOTE: Race condition possible with multiple agents, but the effect is additive so acceptable.
                let fmag = length(thrust_force) * FLUID_FORCE_SCALE * 0.1 * params.prop_wash_strength_fluid;
                let sep = clamp(seg_len, 1.0, 6.0);
                add_fluid_force_dipole(world_pos, thrust_dir_world, -fmag, sep);
            }
        }

        // MAGNET-LIKE ATTRACTION/REPULSION ORGANS (type 45)
        // These organs search for other agents with matching organs and apply forces like magnets:
        // - Opposite polarities (different modifiers) attract
        // - Same polarities (same modifier) repel
        if (base_type == 45u && agent_energy_cur >= amino_props.energy_consumption) {
            // Get this organ's polarity from modifier index
            let organ_param = get_organ_param(part.part_type);
            let my_modifier = u32(clamp(round((f32(organ_param) / 255.0) * 19.0), 0.0, 19.0));

            let MAGNET_SEARCH_RADIUS: f32 = 400.0;
            // Magnet force is 3x stronger than agent-to-agent repulsion (repulsion uses * 100000.0 above).
            let MAGNET_BASE_STRENGTH: f32 = params.agent_repulsion_strength * 300000.0;
            let MAGNET_CLOSEST_NEIGHBORS: u32 = 3u;

            // Find the 3 closest neighbors first
            var closest_distances: array<f32, 3> = array<f32, 3>(1e10, 1e10, 1e10);
            var closest_indices: array<u32, 3> = array<u32, 3>(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);

            for (var n = 0u; n < neighbor_count; n++) {
                let neighbor_id = neighbor_ids[n];
                let neighbor_pos = agents_in[neighbor_id].position;
                let dist_to_agent = length(neighbor_pos - agent_pos);

                if (dist_to_agent < MAGNET_SEARCH_RADIUS) {
                    // Insert into sorted closest list
                    if (dist_to_agent < closest_distances[0]) {
                        closest_distances[2] = closest_distances[1];
                        closest_indices[2] = closest_indices[1];
                        closest_distances[1] = closest_distances[0];
                        closest_indices[1] = closest_indices[0];
                        closest_distances[0] = dist_to_agent;
                        closest_indices[0] = neighbor_id;
                    } else if (dist_to_agent < closest_distances[1]) {
                        closest_distances[2] = closest_distances[1];
                        closest_indices[2] = closest_indices[1];
                        closest_distances[1] = dist_to_agent;
                        closest_indices[1] = neighbor_id;
                    } else if (dist_to_agent < closest_distances[2]) {
                        closest_distances[2] = dist_to_agent;
                        closest_indices[2] = neighbor_id;
                    }
                }
            }

            // Apply magnet forces only to the closest neighbors
            for (var c = 0u; c < MAGNET_CLOSEST_NEIGHBORS; c++) {
                if (closest_indices[c] == 0xFFFFFFFFu) {
                    continue;
                }

                let neighbor_id = closest_indices[c];
                let neighbor_pos = agents_in[neighbor_id].position;
                let neighbor_body_count = min(agents_in[neighbor_id].body_count, MAX_BODY_PARTS);
                let neighbor_rot = agents_in[neighbor_id].rotation;

                // Check each part of the neighbor for type 45 organs
                for (var j = 0u; j < neighbor_body_count; j++) {
                    let neighbor_part = agents_in[neighbor_id].body[j];
                    let neighbor_base_type = get_base_part_type(neighbor_part.part_type);

                    if (neighbor_base_type == 45u) {
                        // Found a matching organ type - check distance
                        let neighbor_part_offset = apply_agent_rotation(neighbor_part.pos, neighbor_rot);
                        let neighbor_part_world = neighbor_pos + neighbor_part_offset;
                        let delta_to_neighbor = neighbor_part_world - world_pos;
                        let dist_to_neighbor = length(delta_to_neighbor);

                        if (dist_to_neighbor > 0.1 && dist_to_neighbor < MAGNET_SEARCH_RADIUS) {
                            // Get neighbor organ's polarity
                            let neighbor_organ_param = get_organ_param(neighbor_part.part_type);
                            let neighbor_modifier = u32(clamp(round((f32(neighbor_organ_param) / 255.0) * 19.0), 0.0, 19.0));

                            // Calculate force based on polarity match
                            // Polarity is based on modifier parity:
                            // - odd repels odd; even repels even
                            // - odd attracts even (and vice-versa)
                            let my_polarity = my_modifier & 1u;
                            let neighbor_polarity = neighbor_modifier & 1u;
                            // Force sign: +1 = attract (toward), -1 = repel (away)
                            let polarity_sign = select(1.0, -1.0, my_polarity == neighbor_polarity);

                            let direction = delta_to_neighbor / dist_to_neighbor;

                            // "Stick" behavior for attraction:
                            // Pure inverse-square attraction tends to slingshot/oscillate when agents get close.
                            // Use a spring (rest distance) + damping (relative velocity along the line of action)
                            // so the force goes to ~0 at the target separation and motion is critically damped.
                            let MAGNET_REST_DISTANCE: f32 = 3.0;
                            let MAGNET_STICK_DEADZONE: f32 =2.0;
                            let MAGNET_SPRING_K: f32 = params.agent_repulsion_strength * 0.4;
                            let MAGNET_DAMP_C: f32 = params.agent_repulsion_strength * 2.0;

                            let rel_vel = agents_in[neighbor_id].velocity - agents_in[agent_id].velocity;
                            let dist_rate = dot(delta_to_neighbor, rel_vel) / dist_to_neighbor;

                            var force_magnitude = 0.0;
                            if (polarity_sign > 0.0) {
                                // Attraction: stable equilibrium at MAGNET_REST_DISTANCE.
                                // error>0 => pull together, error<0 => push apart.
                                var error = dist_to_neighbor - MAGNET_REST_DISTANCE;
                                if (abs(error) < MAGNET_STICK_DEADZONE) {
                                    error = 0.0;
                                }
                                force_magnitude = (MAGNET_SPRING_K * error) + (MAGNET_DAMP_C * dist_rate);
                            } else {
                                // Repulsion: keep inverse-square with softening to avoid singularity.
                                let soft = 25.0;
                                force_magnitude = -(MAGNET_BASE_STRENGTH / (dist_to_neighbor * dist_to_neighbor + soft));
                            }

                            let clamped_force_mag = clamp(force_magnitude, -15000.0, 15000.0);
                            let magnet_force = direction * clamped_force_mag;

                            // Energy sharing: when attractors are within 20 units, agents share energy (average their levels)
                            if (dist_to_neighbor < 20.0) {
                                let my_energy = agent_energy_cur;
                                let neighbor_energy = energy_from_u32(agents_in[neighbor_id].energy);

                                // Each agent transfers half the difference toward equilibrium
                                // Both agents do this calculation independently, converging to average
                                let energy_transfer = (neighbor_energy - my_energy) * 0.5f;
                                agent_energy_cur += energy_transfer;
                            }

                            // Apply force and torque
                            // Calculate lever arm from center of mass to organ position (not segment midpoint)
                            let organ_offset_from_com = part.pos - center_of_mass;
                            let organ_r_com = apply_agent_rotation(organ_offset_from_com, agent_rot);

                            force += magnet_force;
                            torque += (organ_r_com.x * magnet_force.y - organ_r_com.y * magnet_force.x)* (24.0 * PROP_TORQUE_COUPLING);

                            // Explicit alignment torque: rotate the agent so that the magnet's
                            // axis (COM -> organ) points toward the neighbor for attraction,
                            // and away for repulsion. This makes magnets visibly align even
                            // when the net force line-of-action passes near the COM.
                            let r_len = length(organ_r_com);
                            if (r_len > 1e-3) {
                                let axis = organ_r_com / r_len;
                                // Align axis toward target for attraction, away for repulsion.
                                let target_dir = direction * select(1.0, -1.0, polarity_sign > 0.0);
                                let c2 = axis.x * target_dir.y - axis.y * target_dir.x;
                                let d2 = axis.x * target_dir.x + axis.y * target_dir.y;
                                let angle_err = atan2(c2, d2);
                                let align_torque = clamp(angle_err * abs(clamped_force_mag) * r_len * 0.05, -20000.0, 20000.0);
                                torque += align_torque;
                            }
                        }
                    }
                }
            }
        }


        // Displacer slots are repurposed as internal signal emitters.
        // (No external chem sweep / fluid force injection here.)

    }

    // NOTE: torque_debug is repurposed as packed RGB; do not overwrite it with torque.

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

    // WATER-FRAME BASELINE. In the overdamped regime everything moves WITH
    // the medium; forces produce motion RELATIVE to it. The old structure
    // blended total velocity toward force/drag (i.e., toward zero for a
    // passive agent) while a separate drag term fought to pull it back to
    // the water velocity — the equilibrium sat at ~40% of an already-slow
    // drift and agents never visibly rode currents. Now the relative
    // velocity is what the overdamped blend decays, and the water velocity
    // (monomer transport law x push slider; 1 = ride exactly with the
    // food) is simply added on top — riding a current is the default
    // state, not a force that must win a fight.
    if (wind_k > 0.0) {
        // Slider headroom up to 2x so the ride can be trimmed against the
        // rebuilt (slower) velocity field without shader edits.
        let v_water = (wind_target / wind_k)
            * clamp(params.fluid_wind_push_strength, 0.0, 1.0);
        // Baseline = water motion + slow gravitational settling (marine
        // snow): the medium carries you sideways, gravity pulls you down,
        // and everything else is relative motion you must pay for.
        let v_base = v_water
            + vec2<f32>(0.0, -AGENT_SINK_RATE * max(params.dye_beta_color_g, 0.0) * clamp(total_mass, 0.1, 2.0));
        var relv = agent_vel - v_base;
        relv = mix(relv, new_velocity, mass_smoothing);
        agent_vel = v_base + relv;
    } else {
        agent_vel = mix(agent_vel, new_velocity, mass_smoothing);
    }

    // Apply hard-collision correction after smoothing so it can't be damped away
    // for heavy agents. Cap LOW: in dense clone clusters the penetrations of
    // several overlapping neighbors SUM, and the old 20-units/frame cap let
    // agents get catapulted across the screen for several consecutive
    // frames (visible as agents suddenly leaping). 5/frame still separates
    // a full overlap within a few frames, without ballistics.
    if (!anchor_active_prev) {
        // NEWBORN GRACE: fresh agents overlap their parent and siblings by
        // construction. Ramp the separation impulse in over the first ~60
        // frames so births ooze apart instead of catapulting.
        let grace = clamp(f32(agents_in[agent_id].age) / 60.0, 0.1, 1.0);
        let hc_cap = 5.0 * grace;
        let hc_len = length(hard_collision_vel);
        if (hc_len > hc_cap) {
            hard_collision_vel = hard_collision_vel * (hc_cap / max(hc_len, 1e-6));
        }
        agent_vel += hard_collision_vel;
    }

    if (anchor_active_prev) {
        agent_vel = vec2<f32>(0.0);
    }

    let v_len = length(agent_vel);
    if (v_len > VEL_MAX) {
        agent_vel = agent_vel * (VEL_MAX / v_len);
    }

    // Apply torque - overdamped angular motion (no angular inertia)
    // In viscous fluids, angular velocity is directly proportional to torque
    moment_of_inertia = max(moment_of_inertia, 0.01);

    // Overdamped rotation: angular_velocity = torque / rotational_drag
    // Add explicit "fluid" resistance to rotation that scales with body size.
    // In viscous flow, rotational drag increases strongly with object size; this helps
    // prevent unrealistic free-spinning of long bodies.
    var body_len = 0.0;
    for (var i = 1u; i < min(body_count, MAX_BODY_PARTS); i++) {
        body_len += length(agents_out[agent_id].body[i].pos - agents_out[agent_id].body[i - 1u].pos);
    }

    let rotational_drag_base = moment_of_inertia * 20.0;
    // Length-based rotational drag term (tuned to be mild but noticeable).
    let rotational_drag_len = drag_coefficient * body_len * body_len * 0.1;
    let rotational_drag = max(rotational_drag_base + rotational_drag_len, 0.01);
    var angular_velocity = torque / rotational_drag;

    // Option 2: morphology-follow orientation.
    // Convert the estimated morphology rotation between frames into a small, bounded angular velocity.
    if (MORPHOLOGY_ORIENT_FOLLOW_ENABLED) {
        let w_morph = clamp(morph_orient_delta * MORPHOLOGY_ORIENT_FOLLOW_STRENGTH, -MORPHOLOGY_ORIENT_MAX_FRAME_ANGVEL, MORPHOLOGY_ORIENT_MAX_FRAME_ANGVEL);
        angular_velocity += w_morph;
    }

    // Align heading to actual motion direction (helps when morphology-follow heading is ambiguous
    // or when propulsion produces sideways motion).
    if (!MORPHOLOGY_SWIM_SEPARATE_PASS && MORPHOLOGY_SWIM_ENABLED && MORPHOLOGY_SWIM_HEADING_ALIGN_ENABLED && morph_swim_strength > 0.0) {
        let spd = length(agent_vel);
        if (spd > MORPHOLOGY_SWIM_HEADING_ALIGN_MIN_SPEED) {
            let vel_dir = agent_vel / spd;
            let forward = vec2<f32>(cos(agent_rot), sin(agent_rot));
            let c = forward.x * vel_dir.y - forward.y * vel_dir.x;
            let d = forward.x * vel_dir.x + forward.y * vel_dir.y;
            let angle_err = atan2(c, d);
            let speed_scale = clamp(spd / MORPHOLOGY_SWIM_HEADING_ALIGN_FULL_SPEED, 0.0, 1.0);
            let w_align = clamp(angle_err * MORPHOLOGY_SWIM_HEADING_ALIGN_STRENGTH, -MORPHOLOGY_SWIM_HEADING_ALIGN_MAX_ANGVEL, MORPHOLOGY_SWIM_HEADING_ALIGN_MAX_ANGVEL);
            angular_velocity += w_align * speed_scale;
        }
    }

    // WEATHERVANE: instead of spinning with the curl, the body turns toward
    // the least-drag orientation — its axis aligns with the RELATIVE flow
    // (water velocity minus own velocity), like a rod streaming in a
    // current. The axis is a line, so we align to the nearest of the two
    // directions (error wrapped to [-pi/2, pi/2]).
    if (params.fluid_wind_push_strength != 0.0) {
        let com_world = agent_pos;
        if (com_world.x >= 0.0 && com_world.x < f32(SIM_SIZE) && com_world.y >= 0.0 && com_world.y < f32(SIM_SIZE)) {
            let v_w = fluid_ride_velocity(com_world);
            let v_rel_w = v_w - agent_vel;
            let rspd_w = length(v_rel_w);
            if (rspd_w > 0.02) {
                let fwd = vec2<f32>(cos(agent_rot), sin(agent_rot));
                let flow_dir = v_rel_w / rspd_w;
                var ang_err = atan2(fwd.x * flow_dir.y - fwd.y * flow_dir.x, dot(fwd, flow_dir));
                if (ang_err > 1.5707963) { ang_err -= 3.14159265; }
                if (ang_err < -1.5707963) { ang_err += 3.14159265; }
                let w_vane = clamp(
                    ang_err * WIND_VANE_STRENGTH * min(rspd_w, 1.0),
                    -0.05, 0.05
                ) * min(params.fluid_wind_push_strength, 2.0);
                angular_velocity += w_vane;
            }
        }
    }
    angular_velocity = angular_velocity * ANGULAR_BLEND;
    angular_velocity = clamp(angular_velocity, -ANGVEL_MAX, ANGVEL_MAX);

    if (anchor_active_prev) {
        angular_velocity = 0.0;
    }

    // GAMMA GRIP: body parts embedded in terrain hold the agent like an
    // anchor. Every quantum of gamma under a part adds friction — loose
    // rubble slows movement, a part inside solid rock nearly pins the
    // whole agent (slow creep only). This REPLACES the old positional
    // auto-ejection: buried agents stay buried until they grind free, the
    // terrain moves, or they die (leaving their matter as a buried vein).
    {
        var grip_q = 0.0;
        let bc_g = min(body_count_val, MAX_BODY_PARTS);
        for (var gi = 0u; gi < bc_g; gi++) {
            let wp_g = agent_pos + apply_agent_rotation(agents_out[agent_id].body[gi].pos, agent_rot);
            let cg = gamma_count(grid_index(wp_g));
            if (cg >= GAMMA_SOLID_THRESHOLD) {
                grip_q += f32(cg) * GAMMA_GRIP_SOLID;
            } else {
                grip_q += f32(cg) * GAMMA_GRIP_RUBBLE;
            }
        }
        if (grip_q > 0.0) {
            let hold = 1.0 / (1.0 + grip_q);
            agent_vel *= hold;
            angular_velocity *= hold;
        }
    }

    // Update rotation
    if (!DISABLE_GLOBAL_ROTATION) {
        agent_rot += angular_velocity;
    } else {
        agent_rot = 0.0; // keep zero for disabled global rotation experiment
    }

    // Update position with HARD terrain collision: solid rock (>= threshold
    // quanta) cannot be entered — forces alone can be overwhelmed by
    // microswim thrust, and fast agents would tunnel through 1-cell walls.
    // A blocked agent GRINDS against the wall: each blocked frame has a
    // chance (~ mass) to dislodge one quantum from the blocking cell to its
    // rear — tunneling by persistence.
    {
        // Ray-march the move in sub-cell steps: fast agents (up to VEL_MAX
        // units/frame) would otherwise jump clean over 1-cell walls. The
        // agent slides up to the last free position before the wall.
        let move_len = length(agent_vel);
        var hit = false;
        var hit_idx = 0u;
        var hit_pos = agent_pos;
        var dir_m = vec2<f32>(1.0, 0.0);
        if (move_len > 1e-5) {
            dir_m = agent_vel / move_len;
            let steps = min(u32(ceil(move_len / 0.9)) + 1u, 16u);
            let step_len = move_len / f32(steps);
            for (var s = 1u; s <= steps; s++) {
                let p = clamp_position(agent_pos + dir_m * (step_len * f32(s)));
                let ci = grid_index(p);
                if (gamma_count(ci) >= GAMMA_SOLID_THRESHOLD) {
                    hit = true;
                    hit_idx = ci;
                    hit_pos = p;
                    break;
                }
                agent_pos = p;
            }
        }
        if (hit) {
            // GRIND: pressing against the wall can dislodge one quantum
            // from the blocking cell — behind the agent if free, otherwise
            // cast to the side (lets diggers tunnel into deep rock instead
            // of stalling when the cell behind is also solid).
            let bseed0 = (agent_id + 1u) * 2246822519u ^ (params.random_seed * 3266489917u);
            if (hash_f32(bseed0) < DIG_P_PER_FRAME * max(total_mass, 0.1) * max(params._pad_dye_beta_color, 0.0)) {
                gamma_shove(hit_idx, hit_pos, -dir_m, bseed0 ^ 0x27D4EB2Fu);
            }
            // Wall normal from the local gamma gradient (points out of the
            // rock). Kill only the approach component of velocity (with a
            // 20% bounce) and keep the tangential part, so agents SLIDE
            // along walls instead of flip-flopping against their own
            // thrust every frame (visible as flickering).
            let gxm = f32(gamma_count(grid_index(clamp_position(hit_pos + vec2<f32>(-1.5, 0.0)))));
            let gxp = f32(gamma_count(grid_index(clamp_position(hit_pos + vec2<f32>( 1.5, 0.0)))));
            let gym = f32(gamma_count(grid_index(clamp_position(hit_pos + vec2<f32>(0.0, -1.5)))));
            let gyp = f32(gamma_count(grid_index(clamp_position(hit_pos + vec2<f32>(0.0,  1.5)))));
            var wall_n = vec2<f32>(gxm - gxp, gym - gyp);
            if (dot(wall_n, wall_n) < 1e-4) {
                wall_n = -dir_m;
            } else {
                wall_n = normalize(wall_n);
            }
            let vn = dot(agent_vel, wall_n);
            if (vn < 0.0) {
                agent_vel -= wall_n * vn * 1.2;
            }
        }
    }

    // (Positional part-extraction removed: buried parts now GRIP instead —
    // see the GAMMA GRIP block before the rotation update.)

    // ====== UNIFIED ORGAN ACTIVITY LOOP ======
    // Process trail deposition, energy consumption, and feeding in single pass

    // Use the post-morphology capacity written into agents_out this frame
    let capacity = agents_out[agent_id].energy_capacity;

    // poison_resistant_count stored in agent struct during morphology
    // Each poison-resistant organ reduces poison/radiation damage by 50%
    let poison_multiplier = pow(0.5, f32(agents_out[agent_id].poison_resistant_count));
    // Each poison-resistant organ also reduces food power by 25% (0.75^count)
    let food_power_multiplier = pow(0.75, f32(agents_out[agent_id].poison_resistant_count));

    // Initialize accumulators
    let trail_deposit_strength = 0.08; // Strength of trail deposition (0-1)
    var energy_consumption = params.energy_cost; // base maintenance (can be 0)
    energy_consumption += morphology_change_energy;
    var total_consumed_alpha = 0.0;
    var total_consumed_beta = 0.0;
    var consumed_frame = vec4<f32>(0.0);
    var local_alpha = 0.0;
    var local_beta = 0.0;

    // If an agent has any vampire mouth (type 33), deactivate normal mouths.
    // (Computed earlier during the physics pass.)

    // Single loop through all body parts
    for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
        let part = agents_out[agent_id].body[i];
        let base_type = get_base_part_type(part.part_type);
        let props = get_amino_acid_properties(base_type);
        let rotated_pos = apply_agent_rotation(part.pos, agent_rot);
        let world_pos = agent_pos + rotated_pos;
        let idx = grid_index(world_pos);

        // Local chem accumulation (pool sums) for sensors/telemetry.
        local_alpha += chem_pool_count_f(idx, 0u);
        local_beta += chem_pool_count_f(idx, 1u);

        // QUANTIZED uptake: each part attempts to capture ONE whole monomer
        // from the cell under it per frame. Attempt probability scales with
        // how many monomers are present (first-order, discrete events).
        {
            // FEEDING = harvesting ACTIVATION: only charged monomers are
            // food; the spent monomer stays in the cell (mass never leaves
            // the world through feeding — energy is not matter).
            let c0 = chem_act_count(idx, 0u);
            let c1 = chem_act_count(idx, 1u);
            let c2 = chem_act_count(idx, 2u);
            let c3 = chem_act_count(idx, 3u);
            let total_q = c0 + c1 + c2 + c3;
            let yield_per_q = max(params.food_power, 1e-6);
            if (total_q > 0u && agent_energy_cur + yield_per_q <= capacity) {
                // Stiller bodies absorb more (same foraging tradeoff mouths had).
                let speed_norm = clamp(length(agent_velocity) / VEL_MAX, 0.0, 1.0);
                let speed_mult = exp(-4.0 * speed_norm);
                let susp = suspension_factor(world_pos);
                // HUNGER GATE: uptake fades as the agent fills up. Without
                // this, uptake (which runs before pairing in the frame and
                // eats ANY channel) strip-mines the body's neighborhood for
                // energy and starves its own replication of complements.
                // Satiated agents leave the monomers alone; hungry newborns
                // feed fast.
                let hunger = clamp(1.0 - agent_energy_cur / max(capacity, 1e-3), 0.0, 1.0);
                let p_take = clamp(UPTAKE_P_PER_QUANTUM * max(props.mass, 0.1) * speed_mult * susp * f32(total_q) * food_power_multiplier * hunger, 0.0, 1.0);
                let useed = (agent_id + 1u) * 374761393u ^ (i * 668265263u) ^ (params.random_seed * 2246822519u);
                if (hash_f32(useed) < p_take) {
                    // Pick one present monomer, weighted by counts.
                    let pick = hash(useed ^ 0x68E31DA4u) % total_q;
                    var ch = 3u;
                    if (pick < c0) { ch = 0u; }
                    else if (pick < c0 + c1) { ch = 1u; }
                    else if (pick < c0 + c1 + c2) { ch = 2u; }
                    if (chem_spend_one(idx * 4u + ch)) {
                        agent_energy_cur += yield_per_q;
                        if (ch < 2u) { total_consumed_alpha += 1.0; } else { total_consumed_beta += 1.0; }
                        consumed_frame[ch] += 1.0;
                        // Visual: the dye tracks ACTIVATED monomers — the
                        // bite dims the color (the monomer greys out).
                        let after = chem_act_count(idx, ch);
                        var dv = fluid_dye[idx];
                        dv[ch] = dv[ch] * f32(after) / f32(after + 1u);
                        fluid_dye[idx] = dv;
                    }
                }
            }
        }

        // BULLDOZING: a moving body shoves inert gamma mass from under it to
        // the cell behind (relative to motion) — passive digging. Species can
        // evolve to exploit buried monomer veins this way; digging also
        // reshapes terrain, which redirects currents (niche construction).
        {
            let gq = gamma_count(idx);
            if (gq > 0u) {
                let spd_bd = length(agent_velocity);
                if (spd_bd > 0.01) {
                    // Contact pressure digs even at a crawl (0.3 base) —
                    // terrain grip slows bodies ON gamma, so a pure speed
                    // scaling starved digging of its own precondition.
                    let press = clamp(0.3 + spd_bd * 4.0 / VEL_MAX, 0.0, 1.0);
                    let p_dig = clamp(DIG_P_PER_FRAME * max(props.mass, 0.1) * press * max(params._pad_dye_beta_color, 0.0), 0.0, 1.0);
                    let bseed = (agent_id + 1u) * 97u ^ (i * 193u) ^ (params.random_seed * 389u);
                    if (hash_f32(bseed) < p_dig) {
                        // Mole-style: behind first, then side-cast.
                        gamma_shove(idx, world_pos, -(agent_velocity / spd_bd), bseed ^ 0xC2B2AE35u);
                    }
                }
            }
        }

        // Organ-part wash: during microswimming, body motion stirs/displaces the chemistry layers.
        // This is a direct, fluidless transfer similar to prop wash, driven by per-part motion
        // (rigid translation + rotation) so the whole organism can advect nearby chems.
        if (MORPHOLOGY_SWIM_ENABLED) {
            let wash_strength = max(params.prop_wash_strength, 0.0) * MORPHOLOGY_SWIM_COUPLING;
            if (wash_strength > 0.0) {
                // Per-tick world-space velocity of this part.
                let v_part = agent_vel + angular_velocity * vec2<f32>(-rotated_pos.y, rotated_pos.x);
                let spd = length(v_part);
                let norm_spd = spd / max(VEL_MAX, 1e-6);

                // Deadzone: avoid constant stirring when nearly stationary.
                if (norm_spd > 0.02) {
                    let dir = v_part / max(spd, 1e-6);
                    let clamped_pos = clamp_position(world_pos);
                    let grid_scale = f32(SIM_SIZE) / f32(GRID_SIZE);

                    let gx = clamp(i32(clamped_pos.x / grid_scale), 0, i32(GRID_SIZE) - 1);
                    let gy = clamp(i32(clamped_pos.y / grid_scale), 0, i32(GRID_SIZE) - 1);
                    let center_idx = u32(gy) * GRID_SIZE + u32(gx);

                    // Throw a small distance (in cells) proportional to motion.
                    let spd_cells = spd / max(grid_scale, 1e-6);
                    let distance = clamp(spd_cells * wash_strength * 0.75, 1.0, 6.0);
                    let target_world = clamped_pos + dir * distance * grid_scale;
                    let target_gx = clamp(i32(target_world.x / grid_scale), 0, i32(GRID_SIZE) - 1);
                    let target_gy = clamp(i32(target_world.y / grid_scale), 0, i32(GRID_SIZE) - 1);
                    let target_idx = u32(target_gy) * GRID_SIZE + u32(target_gx);

                    if (target_idx != center_idx) {
                        var center_gamma = read_gamma_height(center_idx);
                        var target_gamma = read_gamma_height(target_idx);
                        let part_mass = max(props.mass, 0.01);
                        let part_weight = part_mass / max(total_mass, 1e-6);

                        let transfer_amount = wash_strength * 0.02 * part_weight * clamp(norm_spd, 0.0, 1.0);
                        // Quantized gamma wash: probabilistically shove one
                        // whole quantum (prob = transfer / quantum size).
                        if (transfer_amount > 0.0 && gamma_count(target_idx) < 40u) {
                            let p_move = clamp(transfer_amount / GAMMA_QUANTUM, 0.0, 1.0);
                            let wseed = (agent_id + 1u) * 449u ^ (i * 577u) ^ (params.random_seed * 907u);
                            if (hash_f32(wseed) < p_move) {
                                let _gm = gamma_move_one(center_idx, target_idx);
                            }
                        }
                    }
                }
            }
        }

        // Anchor organ (type 42): signal-thresholded latch state.
        // - L-promoter anchors (LW/LY): use alpha signal (param bit 7 = 0)
        // - P-promoter anchors (PW/PY): use beta signal (param bit 7 = 1)
        if (base_type == 42u) {
            let organ_param = get_organ_param(part.part_type);
            let is_beta_anchor = (organ_param & 128u) != 0u;
            let signal = select(part.alpha_signal, part.beta_signal, is_beta_anchor);

            // Requested behavior: purely sign-based.
            // Active iff signal is positive; inactive if signal is 0 or negative.
            let now_active = signal > 0.0;

            // Requested behavior: anchors only "grab" where gamma exists.
            // Gamma in this sim is capped around 10.0 in transfer logic; normalize to 0..1.
            let gamma_here = max(read_gamma_height(idx), 0.0);
            let gamma_amount = clamp(gamma_here / 10.0, 0.0, 1.0);

            agents_out[agent_id].body[i]._pad.y = select(0.0, 1.0, now_active) * gamma_amount;
        }

        // Vampire mouths (33) and spikes (46):
        // - K-promoter variants: open/close based on enabler proximity (amplification_per_part)
        // - C-promoter variants: open/close and activation based on alpha signal
        // Open amount is CONTINUOUS (0..1), but behaviors still treat activity as on/off at a
        // threshold (e.g. open >= 0.5).
        if (base_type == 33u || base_type == 46u) {
            let organ_param = get_organ_param(part.part_type);
            let is_C_promoter = (organ_param & 128u) != 0u;

            let amp_open = clamp(amplification_per_part[i], 0.0, 1.0);
            let alpha_open = clamp(part.alpha_signal, 0.0, 1.0);
            let target_open = select(amp_open, alpha_open, is_C_promoter);

            let prev_open = clamp(agents_in[agent_id].body[i]._pad.y, 0.0, 1.0);
            let open = mix(prev_open, target_open, 0.2);
            agents_out[agent_id].body[i]._pad.y = open;
        }

        // Calculate actual mouth speed for mouth organs (distance moved since last frame)
        // Only non-vampire mouths need to track previous position for speed-based absorption
        // Vampire mouths use _pad.x for cooldown tracking instead
        var mouth_speed = 0.0;
        var speed_absorption_multiplier = 1.0;

        if (props.is_mouth && base_type != 33u) {
            let packed_prev = bitcast<u32>(agents_out[agent_id].body[i]._pad.x);
            let prev_pos = unpack_prev_pos(packed_prev);
            let displacement_vec = world_pos - prev_pos;
            let displacement_sq = dot(displacement_vec, displacement_vec);
            // If displacement is very small OR prev_pos is near origin, treat as first frame
            let looks_like_first = (displacement_sq < 0.01) || (dot(prev_pos, prev_pos) < 1.0);
            mouth_speed = select(sqrt(displacement_sq), 0.0, looks_like_first);
            let normalized_mouth_speed = mouth_speed / VEL_MAX;
            speed_absorption_multiplier = exp(-8.0 * normalized_mouth_speed);

            // Update previous world position for next frame (only for non-vampire mouths)
            agents_out[agent_id].body[i]._pad.x = bitcast<f32>(pack_prev_pos(world_pos));
        }

        // Debug output for first agent only
        if (agent_id == 0u && params.debug_mode != 0u && i == 0u) {
            agents_out[agent_id].body[63].pos.x = mouth_speed;
            agents_out[agent_id].body[63].pos.y = speed_absorption_multiplier;
        }

        // 1) Trail deposition: blend agent color + deposit energy trail
        // NOTE: write into trail_grid_inject; the displayed trail_grid is produced by fluid-pass advection.
        // Skip when trails are fully invisible.
        if (params.trail_opacity > 0.0) {
            let current_trail = trail_grid_inject[idx].xyz;
            let blended = mix(current_trail, agent_color, trail_deposit_strength);

            // Deposit energy trail (unclamped) - scale by agent energy
            let current_energy_trail = sanitize_f32(trail_grid_inject[idx].w);
            let agent_energy_nonneg = max(sanitize_f32(agent_energy_cur), 0.0);
            // Typical agent energy ~20; scale down contribution so trails don't saturate.
            let energy_deposit = agent_energy_nonneg * trail_deposit_strength * (0.1 / 20.0); // 20x weaker than before
            let blended_energy = sanitize_f32(current_energy_trail + energy_deposit);

            trail_grid_inject[idx] = vec4<f32>(clamp(blended, vec3<f32>(0.0), vec3<f32>(1.0)), blended_energy);
        }

        // 2) Energy consumption: calculate costs per organ type
        // DISABLED: Vampire mouths always active, no need to calculate enabler sum
        var global_mouth_activity = 1.0; // Always active
        // if (base_type == 33u) { // Only calculate for vampire mouths
        //     var enabler_sum = 0.0;
        //     var disabler_sum = 0.0;
        //     for (var j = 0u; j < agents_out[agent_id].body_count; j++) {
        //         let check_type = get_base_part_type(agents_out[agent_id].body[j].part_type);
        //         if (check_type == 26u) { enabler_sum += 1.0; }      // Enabler
        //         else if (check_type == 27u) { disabler_sum += 1.0; } // Disabler
        //     }
        //     global_mouth_activity = clamp(enabler_sum - disabler_sum, 0.0, 1.0);
        // }

        // Minimum baseline cost per amino acid (always paid)
        let baseline = params.amino_maintenance_cost;
        // Organ-specific energy costs
        var organ_extra = 0.0;
        if (props.is_mouth) {
            // Mouth energy cost (unified formula for all mouth types):
            // - Base cost (always paid)
            // - Activity cost scales with enabler amplification (mouth effectiveness)
            // Same pattern as propellers: base + (base × amplification)

            // 3) Feeding: mouths consume from alpha/beta grids
            // Get enabler amplification for this mouth
            let amplification = amplification_per_part[i];

            let activity_cost = props.energy_consumption * amplification;
            organ_extra = props.energy_consumption + activity_cost;

            // RNA-world: mouths no longer feed — uptake is whole-body (see the
            // per-part monomer uptake above). Mouth parts remain as organs
            // (costs, storage capacity) until contact chemistry repurposes them.
        } else if (PROPELLERS_ENABLED && props.is_propeller) {
            // Propellers: unified formula matching mouths
            // Base cost + activity cost scaled by amplification
            let amplification = amplification_per_part[i];
            let activity_cost = props.energy_consumption * amplification;
            organ_extra = props.energy_consumption + activity_cost;
        } else if (props.is_signal_emitter) {
            // Signal emitters: base cost + activity cost (linear with emission strength).
            // Use the stored normalized activity ratio (0..1) to avoid division by zero.
            let strength_ratio = clamp(propeller_thrust_magnitude[i], 0.0, 1.0);
            let activity_cost = props.energy_consumption * strength_ratio * 1.0;
            organ_extra = props.energy_consumption + activity_cost;
        } else {
            // Other organs use linear amplification scaling
            let amp = amplification_per_part[i];
            organ_extra = props.energy_consumption * amp * 1.5;
        }
        energy_consumption += baseline + organ_extra;
    }

    // ====== CATALYTIC TRIPEPTIDES (monomer-converting enzymes) ======
    // Three specific amino acids in consecutive body parts form a catalytic
    // site; READ ORDER sets the direction (X-Y-Z converts a->b, Z-Y-X
    // converts b->a). The site converts one monomer in the cell under the
    // middle residue (state preserved), paying energy. Conserving: one
    // quantum changes channel in place. Encoded by the genome, so enzymes
    // are inherited, mutate, and can be discovered or lost by evolution.
    {
        let bc_e = min(body_count, MAX_BODY_PARTS);
        if (bc_e >= 3u) {
            // var copies: naga 22 only allows dynamic indexing of variables.
            var motifs = ENZYME_MOTIFS;
            var reacts = ENZYME_REACT;
            for (var ei = 1u; ei + 1u < bc_e; ei++) {
                let t0 = get_base_part_type(agents_out[agent_id].body[ei - 1u].part_type);
                let t1 = get_base_part_type(agents_out[agent_id].body[ei].part_type);
                let t2 = get_base_part_type(agents_out[agent_id].body[ei + 1u].part_type);
                // from/to channels (A=0, U=1, G=2, C=3); 4u = no enzyme.
                var e_from = 4u;
                var e_to = 4u;
                for (var k = 0u; k < 4u; k++) {
                    let m = motifs[k];
                    if (t0 == m.x && t1 == m.y && t2 == m.z) { e_from = reacts[k].x; e_to = reacts[k].y; }
                    if (t0 == m.z && t1 == m.y && t2 == m.x) { e_from = reacts[k].y; e_to = reacts[k].x; }
                }
                if (e_from > 3u) { continue; }
                if (agent_energy_cur < ENZYME_COST + 1.0) { continue; }
                let eseed = (agent_id + 1u) * 2654435761u ^ (ei * 374761393u) ^ (params.random_seed * 668265263u);
                if (hash_f32(eseed) >= ENZYME_RATE) { continue; }
                let site = agent_pos + apply_agent_rotation(agents_out[agent_id].body[ei].pos, agent_rot);
                let cidx = grid_index(site);
                let slot_from = cidx * 4u + e_from;
                let vf = atomicLoad(&chem_grid[slot_from]);
                let af = vf & CHEM_STATE_MASK;
                let sf = vf >> 16u;
                if (af + sf == 0u) { continue; }
                let take_spent = hash_f32(eseed ^ 0x6C8E9CF5u) < f32(sf) / f32(af + sf);
                if (chem_take_state_one(slot_from, take_spent)) {
                    chem_add_state(cidx, e_to, 1u, take_spent);
                    agent_energy_cur -= ENZYME_COST;
                }
            }
        }
    }

    // Cap energy by storage capacity after feeding. Energy is harvested
    // ACTIVATION, not matter: overflow and maintenance simply dissipate as
    // heat (the old spill/excretion deposits would now MINT matter, since
    // feeding no longer removes monomers from the world).
    agent_energy_cur = clamp(agent_energy_cur, 0.0, max(capacity, 0.0));

    // 3) Maintenance: subtract consumption after feeding (dissipates).
    agent_energy_cur -= energy_consumption;

    // Defensive: avoid NaN propagation (can happen if any organ cost math goes unstable).
    // If energy becomes NaN, clamp it to 0 so the agent will simply starve rather than poisoning the sim state.
    if (agent_energy_cur != agent_energy_cur) {
        agent_energy_cur = 0.0;
    }

    // 4) Death check
    // Immediate death if energy reaches exactly 0 (prevents "zombie agents")
    var should_die = agent_energy_cur <= 0.0;

    // Otherwise, use probability-based death:
    // - Energy-based: death probability inversely proportional to energy (starvation)
    // Complexity is paid through energy (per-part maintenance + slower replication:
    // pairing_counter must reach gene_length), NOT through a death lottery — no
    // gene-length multiplier here.
    if (!should_die) {
        let death_seed = agent_id * 2654435761u + params.random_seed * 1103515245u;
        let death_rnd = f32(hash(death_seed)) / 4294967295.0;

        // Prevent division by zero and NaN: use max(energy, 0.01) as divisor
        // At energy=10: probability / 10 = very low death chance
        // At energy=1: probability / 1 = normal death chance
        // At energy=0.01: probability / 0.01 = 100x higher death chance (starvation)
        let energy_divisor = max(agent_energy_cur, 0.01);
        let energy_adjusted_death_prob = params.death_probability / energy_divisor;

        // Population pressure modifier (disabled for debugging):
        // When enabled, this multiplies death probability by 2^(floor(population_count / 20000)).
        let pop_mult = 1.0;
        // AQUARIUM TEMPERATURE: shallow lit water is HOT — higher death
        // rate in the rich fast lane at the top; the deep is cold, slow,
        // safe and poor. Same light curve as photo-activation.
        let light_a = uv_light_at_world(agent_pos);
        let temp_mult = mix(1.0, max(params.dye_alpha_color_b, 1.0), light_a);
        let final_death_prob = clamp(energy_adjusted_death_prob * pop_mult * temp_mult, 0.0, 1.0);

        should_die = death_rnd < final_death_prob;
    }

    if (should_die) {
        // RNA-world decomposition (closed world): return exactly what the agent
        // embodies — strand mass split by genome composition, the complements
        // captured by the partial pairing strand, and remaining energy as matter.
        if (body_count > 0u) {
            let comp = genome_composition(agent_genome_packed, agent_genome_offset, agent_gene_length);
            let paired_counts = genome_paired_complement_counts(
                agent_genome_packed, agent_genome_offset, agent_gene_length,
                agents_in[agent_id].pairing_counter
            );
            // Quantized: 1 monomer per genome base; paired complements are
            // whole monomer counts; stored energy converts back at yield.
            // PRIMORDIAL SEEDS (generation 0): minted strand vanishes on
            // death; paired complements + energy (environment-sourced)
            // deposit normally. See the pending-kill path for rationale.
            let strand_quanta = select(f32(agent_gene_length), 0.0, agents_out[agent_id].generation == 0u);
            // Energy = harvested activation, NOT matter: it evaporates on
            // death. Only strand + captured complements return (SPENT).
            let pc = f32(min(body_count, MAX_BODY_PARTS));
            let per_part = (comp * strand_quanta + paired_counts) * (1.0 / pc);
            for (var i = 0u; i < min(body_count, MAX_BODY_PARTS); i++) {
                let part = agents_out[agent_id].body[i];
                let rotated_pos = apply_agent_rotation(part.pos, agent_rot);
                let world_pos = agent_pos + rotated_pos;
                let idx = grid_index(world_pos);
                let dseed = (agent_id + 1u) * 2654435761u ^ (i * 668265263u) ^ params.random_seed;
                for (var dc = 0u; dc < 4u; dc++) {
                    chem_deposit_prob(idx, dc, per_part[dc], dseed ^ (dc * 374761393u));
                }
            }
        }

        // If this was the selected agent, transfer selection to a random nearby agent.
        // Also immediately invalidate the inspector buffer so it can't keep rendering a stale
        // "alive" snapshot when readbacks are throttled.
        if (agent_is_selected == 1u) {
            // NOTE: agents_out uses AgentAtomic (non-copyable due to atomic energy), so copy fields explicitly.
            selected_agent_buffer[0].position = agents_out[agent_id].position;
            selected_agent_buffer[0].velocity = agents_out[agent_id].velocity;
            selected_agent_buffer[0].rotation = 0.0;
            selected_agent_buffer[0].energy = atomicLoad(&agents_out[agent_id].energy);
            selected_agent_buffer[0].energy_capacity = agents_out[agent_id].energy_capacity;
            selected_agent_buffer[0].torque_debug = agents_out[agent_id].torque_debug;
            selected_agent_buffer[0].morphology_origin = agents_out[agent_id].morphology_origin;
            selected_agent_buffer[0].alive = 0u;
            selected_agent_buffer[0].body_count = agents_out[agent_id].body_count;
            selected_agent_buffer[0].pairing_counter = agents_out[agent_id].pairing_counter;
            selected_agent_buffer[0].is_selected = 0u;
            selected_agent_buffer[0].generation = agents_out[agent_id].generation;
            selected_agent_buffer[0].age = agents_out[agent_id].age;
            selected_agent_buffer[0].total_mass = agents_out[agent_id].total_mass;
            selected_agent_buffer[0].poison_resistant_count = agents_out[agent_id].poison_resistant_count;
            selected_agent_buffer[0].gene_length = agents_out[agent_id].gene_length;
            selected_agent_buffer[0].genome_offset = agents_out[agent_id].genome_offset;
            selected_agent_buffer[0].genome_packed = agents_out[agent_id].genome_packed;
            selected_agent_buffer[0].body = agents_out[agent_id].body;

            // NO selection transfer on death. The old behavior jumped the
            // selection to a random survivor, which in fast-turnover species
            // (short genomes, dense clone clouds) made the crosshair hop
            // between look-alike agents every few epochs — it read as one
            // agent flickering between distant positions. The CPU deselects
            // when it sees alive == 0 in the readback.
            agents_out[agent_id].is_selected = 0u;
        }

        // Avoid copying the full Agent payload on death.
        agents_out[agent_id].alive = 0u;
        agents_out[agent_id].body_count = 0u;
        atomicStore(&agents_out[agent_id].energy, 0u);
        agents_out[agent_id].velocity = vec2<f32>(0.0);
        agents_out[agent_id].pairing_counter = 0u;
        agents_out[agent_id].is_selected = 0u;
        return;
    }
    // Note: alive counting is handled in the compaction/merge passes

    // ====== RNA PAIRING ADVANCEMENT ======
    // Pairing counter probabilistically increments based on energy and chemical signals.
    // When it reaches gene_length, reproduction.wgsl will spawn offspring.

    // CRITICAL: Handle reproduction completion BEFORE pairing advancement.
    // Reproduction shader writes offspring when pairing_counter >= gene_length,
    // but it doesn't update the parent (different buffer). We must reset here.
    if (pairing_counter >= agent_gene_length && agent_gene_length > 0u) {
        // Reproduction happened - deduct 50% energy and reset counter
        agent_energy_cur *= 0.5;
        pairing_counter = 0u;
    }

    if (agent_gene_length > 0u && pairing_counter < agent_gene_length) {
        // Pairing requires energy capacity (storage) and sufficient energy (> 1.0)
        if (agent_energy_capacity > 0.0 && agent_energy_cur > 1.0) {
            // PAIRING RATE: params.spawn_probability is the expected number
            // of bases paired PER EPOCH, directly — no hidden energy/scale
            // factors. The integer part is guaranteed attempts, the
            // fractional part one extra bernoulli attempt. Each attempt
            // needs its complement within reach of some body part
            // (whole-body capture, internal diffusion to the site); the
            // first missing monomer ends the epoch's run.
            let rate = max(params.spawn_probability, 0.0);
            let rseed0 = ((agent_id + 1u) * 747796405u) ^ (params.random_seed * 196613u);
            var attempts = u32(rate);
            if (hash_f32(rseed0 ^ 0x1B873593u) < fract(rate)) { attempts += 1u; }
            attempts = min(attempts, PAIRING_MAX_PER_EPOCH);
            let bc_p = max(min(body_count, MAX_BODY_PARTS), 1u);
            let pairing_cost = max(params.pairing_cost, 0.0);

            for (var a = 0u; a < attempts; a++) {
                if (pairing_counter >= agent_gene_length) { break; }
                if (agent_energy_cur <= 1.0 || agent_energy_cur < pairing_cost) { break; }

                // Template-directed: base at position `pairing_counter`
                // captures its complementary free monomer.
                let next_base = genome_get_base_ascii(agent_genome_packed, agent_genome_offset + pairing_counter, agent_genome_offset, agent_gene_length);
                var comp_ch = 0u;
                if (next_base == 65u) { comp_ch = 1u; }      // A pairs a free U
                else if (next_base == 85u) { comp_ch = 0u; } // U pairs a free A
                else if (next_base == 71u) { comp_ch = 3u; } // G pairs a free C
                else if (next_base == 67u) { comp_ch = 2u; } // C pairs a free G

                let scan_seed = rseed0
                    ^ (pairing_counter * 2891336453u)
                    ^ (a * 374761393u);
                var cap_found = false;
                var cap_idx = 0u;
                var cap_pos = agent_pos;
                let start_i = hash(scan_seed) % bc_p;
                for (var k = 0u; k < bc_p; k++) {
                    let pi2 = (start_i + k) % bc_p;
                    let pp = agents_out[agent_id].body[pi2].pos;
                    // Small stochastic reach around the part.
                    let oh = hash(scan_seed ^ (pi2 * 668265263u));
                    let oa = hash_f32(oh) * 6.2831853;
                    let orr = hash_f32(oh ^ 0x9E3779B9u) * PAIRING_REACH;
                    let wp2 = clamp_position(
                        agent_pos + apply_agent_rotation(pp, agent_rot)
                            + vec2<f32>(cos(oa), sin(oa)) * orr
                    );
                    let gi2 = grid_index(wp2);
                    // Polymerization needs an ACTIVATED complement — its
                    // charge pays for the bond (spent monomers can't pair).
                    if (chem_act_count(gi2, comp_ch) > 0u) {
                        cap_found = true;
                        cap_idx = gi2;
                        cap_pos = wp2;
                        break;
                    }
                }
                if (!cap_found) { break; }

                // Suspension stub (currently 1.0) kept as a per-attempt gate.
                if (hash_f32(scan_seed ^ 0xB5297A4Du) >= suspension_factor(cap_pos)) { continue; }

                // Capture the complementary monomer (returned on death if
                // replication never completes).
                if (chem_take_one(cap_idx * 4u + comp_ch)) {
                    pairing_counter += 1u;
                    agent_energy_cur -= pairing_cost;
                    // Visual: bite the exact dye channel of the captured monomer.
                    let after = chem_act_count(cap_idx, comp_ch);
                    var dv = fluid_dye[cap_idx];
                    dv[comp_ch] = dv[comp_ch] * f32(after) / f32(after + 1u);
                    fluid_dye[cap_idx] = dv;
                } else {
                    break;
                }
            }
        }
    }

    agents_out[agent_id].pairing_counter = pairing_counter;

    // Always write selected agent to readback buffer for inspector (even when drawing disabled)
    if (agent_is_selected == 1u) {
        // Publish an unrotated copy for inspector preview.
        // NOTE: agents_out uses AgentAtomic (non-copyable due to atomic energy), so copy fields explicitly.
        selected_agent_buffer[0].position = agents_out[agent_id].position;
        selected_agent_buffer[0].velocity = agents_out[agent_id].velocity;
        selected_agent_buffer[0].rotation = 0.0;
        selected_agent_buffer[0].energy = atomicLoad(&agents_out[agent_id].energy);
        selected_agent_buffer[0].energy_capacity = agents_out[agent_id].energy_capacity;
        selected_agent_buffer[0].torque_debug = agents_out[agent_id].torque_debug;
        selected_agent_buffer[0].morphology_origin = agents_out[agent_id].morphology_origin;
        selected_agent_buffer[0].alive = agents_out[agent_id].alive;
        selected_agent_buffer[0].body_count = agents_out[agent_id].body_count;
        selected_agent_buffer[0].pairing_counter = agents_out[agent_id].pairing_counter;
        selected_agent_buffer[0].is_selected = agents_out[agent_id].is_selected;
        selected_agent_buffer[0].generation = agents_out[agent_id].generation;
        selected_agent_buffer[0].age = agents_out[agent_id].age;
        selected_agent_buffer[0].total_mass = agents_out[agent_id].total_mass;
        selected_agent_buffer[0].poison_resistant_count = agents_out[agent_id].poison_resistant_count;
        selected_agent_buffer[0].gene_length = agent_gene_length;
        selected_agent_buffer[0].genome_offset = agents_out[agent_id].genome_offset;
        selected_agent_buffer[0].genome_packed = agents_out[agent_id].genome_packed;
        selected_agent_buffer[0].body = agents_out[agent_id].body;
    }

    // Always write output state (simulation must continue even when not drawing)
    agents_out[agent_id].position = agent_pos;
    agents_out[agent_id].velocity = agent_vel;
    agents_out[agent_id].rotation = agent_rot;
    atomicStore(&agents_out[agent_id].energy, energy_to_u32(max(agent_energy_cur, 0.0)));
    agents_out[agent_id].gene_length = agent_gene_length;
    // Increment age for living agents
    if (agents_out[agent_id].alive == 1u) {
        agents_out[agent_id].age = agents_out[agent_id].age + 1u;
    }
    // Note: body[], genome[], body_count, generation already set correctly in agents_out
}

// ============================================================================
// ENVIRONMENT DIFFUSION & DECAY
// ============================================================================

// ============================================================================
// SANDPILE RELAXATION (quantized inert gamma mass)
// ============================================================================
// If a cell holds at least SANDPILE_DIFF more quanta than its lowest
// 4-neighbor, one quantum slides downhill with probability GAMMA_RELAX_P.
// Emergent angle of repose: slopes below the threshold are stable forever;
// disturbed piles produce landslides. Atomic take/add — exactly conserving.
// Runs twice per frame (the old stage1/stage2 dispatch slots).
// DIFF 3 => stable slopes up to 2 quanta/cell: spoil heaps and berms can
// actually gain height instead of the sandpile flattening every shoved
// quantum back to baseline within frames (with DIFF 2 the max stable
// slope was 1/cell, so a 6-high pile needed a ~36-quanta pyramid and
// transient bright pixels always relaxed away). Lower relax probability
// makes slides chunkier and terrain generally more rugged.
const SANDPILE_DIFF: u32 = 3u;
const GAMMA_RELAX_P: f32 = 0.12;
// Cohesion rate for isolated fines (see GAMMA COHESION in relax_gamma_pass).
// Geological timescale: at 0.01/pass x2 passes, a lone speck beside a rock
// merges in ~50 epochs; a field of scattered debris consolidates over
// thousands.
const GAMMA_COHESION_P: f32 = 0.01;
// Random-walk rate for fully isolated single quanta (all 8 neighbors
// empty) — the 0<->1 exchange that lets strays search for company.
const GAMMA_STRAY_WALK_P: f32 = 0.05;
// Loose debris rides currents at this fraction of the monomer advection
// law (inert mass is heavier than dissolved monomers).
const GAMMA_SEDIMENT_FACTOR: f32 = 0.5;
// Shedding rate: a 2-stack beside an empty cell drops its top quantum
// into the best-nestled vacancy (gravity, not bond-gated).
const GAMMA_SHED_P: f32 = 0.03;

fn relax_gamma_pass(gid: vec3<u32>, phase: u32) {
    let x = gid.x;
    let y = gid.y;
    if (x >= GRID_SIZE || y >= GRID_SIZE) {
        return;
    }
    let idx = y * GRID_SIZE + x;
    let n = gamma_count(idx);

    // LOOSE-GRAIN DYNAMICS (counts below the sandpile band): ONE unified
    // minimum-energy move rule. A grain ALWAYS moves toward the candidate
    // cell where it would have the MOST occupied neighbors (bond-count
    // maximization = surface-energy minimization: grains nestle into
    // corners and notches, growing compact blobs laterally — never
    // towers, since it also can never land on a cell higher than its own
    // level). Mobility falls 4x per current bond (contacts grip); the
    // fluid can still rip weakly-bonded grains downstream (sediment).
    // Ties are hash-broken to avoid scan-order directional drift.
    if (n >= 1u && n < SANDPILE_DIFF) {
        var bonds = 0u;
        var best_idx = idx;
        var best_score = -1i;
        var best_c = 0xFFFFu;
        var em_idx = idx;
        var em_score = -1i;
        var em_found = false;
        var tie_h = hash(idx ^ (params.epoch * 1597334677u) ^ (params.random_seed * 3812015801u) ^ (phase * 2654435761u));
        for (var dy = -1i; dy <= 1i; dy++) {
            for (var dx = -1i; dx <= 1i; dx++) {
                if (dx == 0i && dy == 0i) { continue; }
                let nx = i32(x) + dx;
                let ny = i32(y) + dy;
                if (nx < 0i || ny < 0i || nx >= i32(GRID_SIZE) || ny >= i32(GRID_SIZE)) {
                    continue;
                }
                let ni = u32(ny) * GRID_SIZE + u32(nx);
                let c = gamma_count(ni);
                if (c >= 1u) { bonds += 1u; }
                // Candidate destination: no higher than our own level.
                if (c <= n) {
                    // Score = occupied cells around the CANDIDATE, skipping
                    // our own cell (we vacate it on moving).
                    var score = 0i;
                    for (var sy = -1i; sy <= 1i; sy++) {
                        for (var sx = -1i; sx <= 1i; sx++) {
                            if (sx == 0i && sy == 0i) { continue; }
                            let qx = nx + sx;
                            let qy = ny + sy;
                            if (qx < 0i || qy < 0i || qx >= i32(GRID_SIZE) || qy >= i32(GRID_SIZE)) {
                                continue;
                            }
                            let qi = u32(qy) * GRID_SIZE + u32(qx);
                            if (qi == idx) { continue; }
                            if (gamma_count(qi) >= 1u) { score += 1i; }
                        }
                    }
                    tie_h = hash(tie_h ^ ni);
                    if (score > best_score
                        || (score == best_score && c < best_c)
                        || (score == best_score && c == best_c && (tie_h & 1u) == 1u)) {
                        best_score = score;
                        best_c = c;
                        best_idx = ni;
                    }
                    // Track the best EMPTY candidate separately (shedding
                    // target): the vacancy with the most material around it.
                    if (c == 0u && (score > em_score || (score == em_score && (tie_h & 2u) == 2u))) {
                        em_score = score;
                        em_idx = ni;
                        em_found = true;
                    }
                }
            }
        }
        if (bonds >= 4u) {
            return; // locked in
        }
        let mob = pow(0.25, f32(bonds));

        var vs = vec2<f32>(0.0);
        if (params._pad_rain0 == 1u) {
            let fxi = min((x * FLUID_GRID_SIZE) / GRID_SIZE, FLUID_GRID_SIZE - 1u);
            let fyi = min((y * FLUID_GRID_SIZE) / GRID_SIZE, FLUID_GRID_SIZE - 1u);
            vs = fluid_velocity[fyi * FLUID_GRID_SIZE + fxi];
        }
        let spd_s = length(vs);
        let p_sed = parcel_hop_p(vs) * GAMMA_SEDIMENT_FACTOR * mob;
        let hw = hash(idx ^ (params.epoch * 747796405u) ^ (phase * 196613u) ^ (params.random_seed * 2891336453u));
        let rw = f32(hw) / 4294967295.0;
        var dest = idx;
        var have_dest = false;
        // SHEDDING (the reverse rule): a 2-stack beside a VACANCY is a
        // mini-cliff — the top quantum steps down into the most-nestled
        // empty neighbor. Gravity-driven, so independent of bond grip.
        if (n >= 2u && em_found && hash_f32(hw ^ 0x632BE5ABu) < GAMMA_SHED_P) {
            dest = em_idx;
            have_dest = true;
        } else if (rw < p_sed) {
            // Downstream hop: axis chosen by |vx| vs |vy| weight
            // (mirrors transport_quanta). Never climbs above own level.
            var ddx = 0i;
            var ddy = 0i;
            let ax = abs(vs.x);
            let ay = abs(vs.y);
            if (hash_f32(hw ^ 0x85EBCA6Bu) < ax / max(ax + ay, 1e-6)) {
                ddx = select(-1i, 1i, vs.x > 0.0);
            } else {
                ddy = select(-1i, 1i, vs.y > 0.0);
            }
            let wx = i32(x) + ddx;
            let wy = i32(y) + ddy;
            if (wx >= 0i && wy >= 0i && wx < i32(GRID_SIZE) && wy < i32(GRID_SIZE)) {
                let wi = u32(wy) * GRID_SIZE + u32(wx);
                if (gamma_count(wi) <= n) {
                    dest = wi;
                    have_dest = true;
                }
            }
        } else if (rw < p_sed + GAMMA_STRAY_WALK_P * mob) {
            if (best_score > i32(bonds)) {
                // Energy-improving move: settle where we'd have MORE
                // neighbors than here.
                dest = best_idx;
                have_dest = true;
            } else if (bonds == 0u) {
                // Free debris at a local optimum of nothing: explore.
                var ddx = 0i;
                var ddy = 0i;
                let d = (hw >> 8u) % 8u;
                switch d {
                    case 0u: { ddx = 1i; }
                    case 1u: { ddx = -1i; }
                    case 2u: { ddy = 1i; }
                    case 3u: { ddy = -1i; }
                    case 4u: { ddx = 1i; ddy = 1i; }
                    case 5u: { ddx = -1i; ddy = 1i; }
                    case 6u: { ddx = 1i; ddy = -1i; }
                    default: { ddx = -1i; ddy = -1i; }
                }
                let wx = i32(x) + ddx;
                let wy = i32(y) + ddy;
                if (wx >= 0i && wy >= 0i && wx < i32(GRID_SIZE) && wy < i32(GRID_SIZE)) {
                    let wi = u32(wy) * GRID_SIZE + u32(wx);
                    if (gamma_count(wi) <= n) {
                        dest = wi;
                        have_dest = true;
                    }
                }
            }
        }
        if (have_dest && dest != idx) {
            let _gm = gamma_move_one(idx, dest);
        }
        return;
    }
    if (n < SANDPILE_DIFF) {
        return;
    }

    // Find the lowest in-bounds 4-neighbor.
    var best_idx = idx;
    var best_count = n;
    if (x > 0u) {
        let c = gamma_count(idx - 1u);
        if (c < best_count) { best_count = c; best_idx = idx - 1u; }
    }
    if (x + 1u < GRID_SIZE) {
        let c = gamma_count(idx + 1u);
        if (c < best_count) { best_count = c; best_idx = idx + 1u; }
    }
    if (y > 0u) {
        let c = gamma_count(idx - GRID_SIZE);
        if (c < best_count) { best_count = c; best_idx = idx - GRID_SIZE; }
    }
    if (y + 1u < GRID_SIZE) {
        let c = gamma_count(idx + GRID_SIZE);
        if (c < best_count) { best_count = c; best_idx = idx + GRID_SIZE; }
    }

    if (best_idx == idx || n - best_count < SANDPILE_DIFF) {
        return;
    }
    let h = hash(idx ^ (params.epoch * 668265263u) ^ (phase * 2654435761u) ^ (params.random_seed * 374761393u));
    if (f32(h) / 4294967295.0 < GAMMA_RELAX_P) {
        let _gm = gamma_move_one(idx, best_idx);
    }
}

@compute @workgroup_size(16, 16)
fn diffuse_grids_stage1(@builtin(global_invocation_id) gid: vec3<u32>) {
    relax_gamma_pass(gid, 0u);
}

@compute @workgroup_size(16, 16)
fn diffuse_grids_stage2(@builtin(global_invocation_id) gid: vec3<u32>) {
    relax_gamma_pass(gid, 1u);
}

// Dye-exclusive simulation: precipitation is disabled for now — all monomers
// are permanently suspended and bio-available everywhere. This stub is kept
// (and still called by uptake/pairing) so a settled/suspended model can be
// reintroduced later via params.suspend_full_speed / suspend_zero_speed.
fn suspension_factor(world_pos: vec2<f32>) -> f32 {
    let _unused = world_pos;
    return 1.0;
}

// ============================================================================
// QUANTIZED MONOMER TRANSPORT (Brownian diffusion + fluid advection in one)
// ============================================================================
// Each monomer independently attempts one hop per frame:
// - with probability p_adv (local fluid speed x Monomer Advection slider):
//   hop to the downstream neighbor (advection by the current);
// - else with probability DIFF_HOP_P: hop to a random 4-neighbor (diffusion).
// Atomic take/add makes every hop exactly mass-conserving; hops across the
// world boundary are cancelled. No staging buffer needed.
// Diffusion hop probability is AGITATION-DEPENDENT: near-still water only
// allows a small Brownian residue (DIFF_HOP_FLOOR of the base rate), while
// moving water mixes at the full rate. This is the anti-entropy mechanism:
// calm zones become persistent concentrated reservoirs instead of everything
// random-walking to uniform dilution over long runs.
const DIFF_HOP_P: f32 = 0.01;
const DIFF_HOP_FLOOR: f32 = 0.15;
const DIFF_AGITATION_SPEED: f32 = 0.5;
// Suspended monomers ALWAYS ride the current — fixed physical coupling, not
// a user setting (colloidal particles move with the water; there is nothing
// to tune, and no stale settings file can freeze the soup again).
// Cap on the drift speed (cells/epoch) — safety only; the parcel-exact
// law below rarely approaches it.
const MONOMER_ADV_CAP: f32 = 0.9;

// PARCEL-EXACT hop probability: expected displacement = |v| * dt *
// cell_to_world cells/epoch — the SAME distance a fluid parcel (and a
// riding agent) travels. The old law (|v| * 1.0, capped) moved quanta
// 3-13x FASTER than the flow and saturated at the cap: that nonlinearity
// EVACUATED persistent voids at plume acceleration zones and piled rings
// at deceleration fronts. A passive tracer advected consistently with an
// incompressible field stays uniform, as it must.
// NOTE the L1 norm (|vx|+|vy|), not Euclidean: the hop direction is split
// by |vx|/(|vx|+|vy|), so with an L1 total probability the per-axis flux
// is EXACTLY k*vx*n / k*vy*n — a conservative upwind scheme in
// expectation, which preserves uniform fields under divergence-free flow.
// With the L2 norm the flux carried a direction-dependent factor
// (|v|/(|vx|+|vy|) in [0.707, 1]) that broke discrete continuity wherever
// the flow direction ROTATES — plume heads and vortex rims — carving the
// persistent vacuum pockets Filipe kept (correctly!) refusing to accept.
// UNITS: hops are in ENV-GRID cells, so the conversion is env cells per
// fluid cell (GRID_SIZE / FLUID_GRID_SIZE) — NOT world units per fluid
// cell (SIM_SIZE is the much larger world size; using it overshot the hop
// rate ~30x and saturated the cap).
fn parcel_hop_p(v: vec2<f32>) -> f32 {
    let env_per_fluid = f32(GRID_SIZE) / f32(FLUID_GRID_SIZE);
    let l1 = abs(v.x) + abs(v.y);
    return clamp(l1 * env_per_fluid * max(params.dt, 1e-3), 0.0, MONOMER_ADV_CAP);
}
// Buried monomers (inside gamma mass) creep at this fraction of the base
// diffusion rate, further divided by local occupancy. They never advect.
// POROUS ROCK: buried monomers percolate at the full base diffusion rate,
// attenuated gently by occupancy, so nutrients seep in and out of gamma
// masses and agents can harvest the rims (was 0.25 and 1/(1+g): ~30x
// slower, gamma interiors were sealed vaults).
const BURIED_DIFF_FACTOR: f32 = 1.0;
// Downward settling hop probability per free monomer per epoch (light
// gravity: ~1 cell every 500 epochs, much slower than plume lift).
const MONOMER_SETTLE_P: f32 = 0.002;
// CATALYTIC TRIPEPTIDES: amino acid indices (0 A,1 C,2 D,3 E,4 F,5 G,6 H,
// 7 I,8 K,9 L,10 M,11 N,12 P,13 Q,14 R,15 S,16 T,17 V,18 W,19 Y).
// Read forward = REACT.x -> REACT.y; read backward = reverse reaction.
const ENZYME_MOTIFS = array<vec3<u32>, 4>(
    vec3<u32>(6u, 18u, 19u),  // H-W-Y : A <-> G (purines)
    vec3<u32>(1u, 10u, 4u),   // C-M-F : U <-> C (pyrimidines)
    vec3<u32>(13u, 18u, 11u), // Q-W-N : A <-> U
    vec3<u32>(19u, 10u, 12u)  // Y-M-P : G <-> C
);
const ENZYME_REACT = array<vec2<u32>, 4>(
    vec2<u32>(0u, 2u),
    vec2<u32>(1u, 3u),
    vec2<u32>(0u, 1u),
    vec2<u32>(2u, 3u)
);
const ENZYME_RATE: f32 = 0.1;  // conversion attempts per site per frame
const ENZYME_COST: f32 = 0.05; // energy per conversion
// MONOMER REACTION RULES (all conserving: they move or transform quanta,
// never create or destroy them).
const CHEM_SENSITIZE: f32 = 0.5;      // photo-activation boost per activated monomer in cell
const CHEM_SHIELD: f32 = 0.5;         // decay reduction per extra activated monomer in cell
// Abiotic interconversion RETIRED (unrealistic): base conversion is now
// enzymatic only (catalytic tripeptides).
const CHEM_INTERCONV_P: f32 = 0.0; // spent purine<->purine / pyrimidine<->pyrimidine per epoch
const CHEM_COHESION: f32 = 0.6;       // diffusion bias toward activated same-type neighbors
const GAMMA_POROSITY_K: f32 = 0.3;
// PORE CAPACITY per cell (total monomers, all channels and states):
// open water holds up to CHEM_CELL_CAP; each gamma quantum removes pore
// volume; solid rock (>= GAMMA_SOLID_THRESHOLD) holds none.
const CHEM_CELL_CAP: u32 = 48u;
const CHEM_SQUEEZE_P: f32 = 0.25;

fn chem_cell_total(cell: u32) -> u32 {
    var t = 0u;
    for (var c = 0u; c < 4u; c++) {
        let v = atomicLoad(&chem_grid[cell * 4u + c]);
        t += (v & CHEM_STATE_MASK) + (v >> 16u);
    }
    return t;
}

// THE ONLY WAY TERRAIN MOVES: a gamma quantum goes from `from` to `to`,
// and any monomers that no longer fit in `to` (its pore capacity just
// shrank) SWAP back into `from`, which just gained pore volume. Like a
// falling sand grain displacing water: terrain can never bury monomers,
// so no cell is ever pushed over capacity. Returns false if no gamma.
fn gamma_move_one(src_c: u32, dst_c: u32) -> bool {
    if (!gamma_take_one(src_c)) {
        return false;
    }
    gamma_add_one(dst_c);
    let cap = chem_capacity(gamma_count(dst_c));
    var guard = 0u;
    loop {
        if (guard >= 64u) { break; }
        guard += 1u;
        if (chem_cell_total(dst_c) <= cap) { break; }
        var moved = false;
        for (var c = 0u; c < 4u; c++) {
            let slot = dst_c * 4u + c;
            if (chem_take_state_one(slot, true)) {
                chem_add_state(src_c, c, 1u, true);
                moved = true;
                break;
            }
            if (chem_take_state_one(slot, false)) {
                chem_add_state(src_c, c, 1u, false);
                moved = true;
                break;
            }
        }
        if (!moved) { break; }
    }
    return true;
}

fn chem_capacity(g: u32) -> u32 {
    // ANY gamma (rubble or rock) holds no monomers: terrain is solid matter.
    if (g > 0u) { return 0u; }
    return CHEM_CELL_CAP;
}
const MAX_HOPS_PER_CELL_CH: u32 = 8u;
// Aquarium thermodynamics: UV activation at full light (per spent monomer
// per epoch), light attenuation with depth (top of world = high y = lit;
// bottom retains exp(-LIGHT_ATTEN) ~ 5%), and spontaneous activation decay.
// NOTE: the renderer maps SCREEN TOP to LOW world y — light enters at y=0.
const LIGHT_ACT_P: f32 = 0.006;
const LIGHT_ATTEN: f32 = 8.0;
const CHEM_DECAY_P: f32 = 0.0002;
// Hot shallow water: death probability multiplier at full light (the fast
// lane — rich but deadly; the deep is cold, slow, safe and poor).
const TEMP_DEATH_MULT_TOP: f32 = 2.5;
// MARINE SNOW: bodies are slightly denser than water and settle slowly
// (world units/epoch per unit mass, downward = -y). Staying in the lit
// zone costs swimming; passive drifters and carcass-feeders sink.
const AGENT_SINK_RATE: f32 = 0.05;

@compute @workgroup_size(16, 16)
fn transport_quanta(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= GRID_SIZE || y >= GRID_SIZE) {
        return;
    }
    let idx = y * GRID_SIZE + x;

    var v = vec2<f32>(0.0);
    if (params._pad_rain0 == 1u) {
        // Bilinear: nearest-cell sampling gave every 4x4 env block one
        // piecewise-constant velocity, creating artificial divergence
        // steps at block boundaries.
        // Takes a WORLD position: env cell -> world (SIM_SIZE != GRID_SIZE;
        // passing cell coords sampled only the world's corner and froze
        // every monomer).
        let cell_w = f32(SIM_SIZE) / f32(GRID_SIZE);
        v = sample_fluid_velocity_bilinear(vec2<f32>(f32(x) + 0.5, f32(y) + 0.5) * cell_w);
    }
    let speed_t = length(v);
    let p_adv = parcel_hop_p(v);
    let agitation = clamp(speed_t / DIFF_AGITATION_SPEED, 0.0, 1.0);
    let p_diff = DIFF_HOP_P * mix(DIFF_HOP_FLOOR, 1.0, agitation);

    // GAMMA INTERACTION (no ghosting): monomers INSIDE inert mass do not
    // ride currents — buried matter only creeps by slow diffusion, scaled
    // down by occupancy and independent of water agitation (the water does
    // not reach inside the rock).
    let g_src = gamma_count(idx);
    let src_total = chem_cell_total(idx);
    let src_over = g_src > 0u && src_total > chem_capacity(g_src);
    var p_adv_eff = p_adv;
    var p_diff_eff = p_diff;
    if (g_src > 0u) {
        p_adv_eff = 0.0;
        p_diff_eff = DIFF_HOP_P * BURIED_DIFF_FACTOR / (1.0 + GAMMA_POROSITY_K * f32(g_src));
        // SQUEEZE: a cell over its pore capacity (terrain slid, was shoved
        // or consolidated ON TOP of monomers) expels the excess fast, into
        // neighbors that still have room. Rock no longer hoards matter.
        if (src_over) {
            p_diff_eff = CHEM_SQUEEZE_P;
        }
    }

    // AQUARIUM LIGHT: UV enters at the TOP of the screen — empirically
    // HIGH world y (the present chain flips y once more than the
    // composite math suggests). Attenuates with depth.
    let light_t = uv_light_at_cell(x, y);

    for (var ch = 0u; ch < 4u; ch++) {
        let slot = idx * 4u + ch;
        let v_ch = atomicLoad(&chem_grid[slot]);
        let act_n = v_ch & CHEM_STATE_MASK;
        let spent_n = v_ch >> 16u;

        // PHOTO-ACTIVATION: spent monomers recharge under UV, scaled by
        // depth. DECAY: activation is slowly lost everywhere (hydrolysis).
        // Together with feeding (act->spent) this closes the energy cycle:
        // light pumps, life and decay drain.
        if (spent_n > 0u) {
            // PHOTOSENSITIZATION (autocatalysis with an external fuel):
            // activated monomers in the cell act as antennae and boost the
            // photo-activation of their spent neighbors. Energy still comes
            // only from light; the reaction just concentrates it.
            let sens = 1.0 + CHEM_SENSITIZE * f32(min(act_n, 8u));
            let exp_act = f32(min(spent_n, 8u)) * LIGHT_ACT_P * max(params.dye_alpha_color_r, 0.0) * light_t * sens;
            let wa = u32(floor(exp_act));
            var na = wa;
            if (hash_f32(slot ^ (params.random_seed * 1597334677u) ^ (params.epoch * 3812015801u)) < exp_act - f32(wa)) { na += 1u; }
            for (var ai = 0u; ai < min(na, 8u); ai++) {
                if (!chem_activate_one(slot)) { break; }
            }
        }
        if (act_n > 0u) {
            // SHIELDING: clustered activated monomers protect each other
            // from hydrolysis (decay slows with local activated density).
            let shield = 1.0 / (1.0 + CHEM_SHIELD * f32(min(act_n, 8u) - 1u));
            let exp_dec = f32(min(act_n, 8u)) * CHEM_DECAY_P * shield;
            if (hash_f32(slot ^ (params.random_seed * 2654435761u) ^ (params.epoch * 668265263u)) < exp_dec) {
                chem_spend_one(slot);
            }
        }

        // INTERCONVERSION: spent monomers slowly convert within their
        // chemical family — purines A<->G (ch 0<->2), pyrimidines U<->C
        // (ch 1<->3). Conserving (one quantum moves between channels of
        // the same cell); lets world composition drift and re-balance.
        // Acts on BOTH states (state preserved): a reloaded snapshot is fully
        // activated, and spent-only interconversion could not rebalance it.
        if (act_n + spent_n > 0u) {
            let exp_ic = f32(min(act_n + spent_n, 8u)) * CHEM_INTERCONV_P;
            if (hash_f32(slot ^ (params.random_seed * 3266489917u) ^ (params.epoch * 2246822519u)) < exp_ic) {
                let ic_spent = hash_f32(slot ^ 0x7F4A7C15u ^ params.epoch) < f32(spent_n) / f32(act_n + spent_n);
                if (chem_take_state_one(slot, ic_spent)) {
                    chem_add_state(idx, ch ^ 2u, 1u, ic_spent);
                }
            }
        }

        // COHESION weights: activated monomers bias their diffusion toward
        // neighbors already rich in activated monomers of the SAME type
        // (droplets / coacervates). Neighbor counts read once per channel.
        var coh = array<f32, 4>(1.0, 1.0, 1.0, 1.0);
        if (act_n > 0u && CHEM_COHESION > 0.0) {
            if (x + 1u < GRID_SIZE) { coh[0] = 1.0 + CHEM_COHESION * f32(min(chem_act_count(idx + 1u, ch), 8u)); }
            if (x > 0u) { coh[1] = 1.0 + CHEM_COHESION * f32(min(chem_act_count(idx - 1u, ch), 8u)); }
            if (y + 1u < GRID_SIZE) { coh[2] = 1.0 + CHEM_COHESION * f32(min(chem_act_count(idx + GRID_SIZE, ch), 8u)); }
            if (y > 0u) { coh[3] = 1.0 + CHEM_COHESION * f32(min(chem_act_count(idx - GRID_SIZE, ch), 8u)); }
        }
        let coh_sum = coh[0] + coh[1] + coh[2] + coh[3];

        let n = min(act_n + spent_n, MAX_HOPS_PER_CELL_CH);
        for (var k = 0u; k < n; k++) {
            let h = hash(slot ^ (k * 668265263u) ^ (params.random_seed * 2246822519u) ^ (params.epoch * 374761393u));
            let r = f32(h) / 4294967295.0;
            var dx = 0;
            var dy = 0;
            var is_adv = false;
            // State of the hopping monomer, chosen proportionally to the mix.
            let pick_spent = hash_f32(h ^ 0x51ED270Bu) < f32(spent_n) / f32(max(act_n + spent_n, 1u));
            if (r < p_adv_eff) {
                // Downstream hop: axis chosen by |vx| vs |vy| weight.
                is_adv = true;
                let ax = abs(v.x);
                let ay = abs(v.y);
                if (hash_f32(h ^ 0x85EBCA6Bu) < ax / max(ax + ay, 1e-6)) {
                    dx = select(-1, 1, v.x > 0.0);
                } else {
                    dy = select(-1, 1, v.y > 0.0);
                }
            } else if (r < p_adv_eff + p_diff_eff) {
                var d = hash(h ^ 0xC2B2AE35u) & 3u;
                if (!pick_spent && coh_sum > 4.0001) {
                    // Cohesive (weighted) direction for activated monomers.
                    var u = hash_f32(h ^ 0x9E3779B9u) * coh_sum;
                    d = 3u;
                    for (var cd = 0u; cd < 4u; cd++) {
                        if (u < coh[cd]) { d = cd; break; }
                        u -= coh[cd];
                    }
                }
                if (d == 0u) { dx = 1; } else if (d == 1u) { dx = -1; }
                else if (d == 2u) { dy = 1; } else { dy = -1; }
            } else if (g_src == 0u && r < p_adv_eff + p_diff_eff + MONOMER_SETTLE_P * max(params.dye_beta_color_r, 0.0)) {
                // LIGHT SEDIMENTATION: free monomers slowly settle toward
                // the bottom (-y = screen down), refilling the vents'
                // intake so plumes keep having material to lift. Settling
                // never enters rock: sediment RESTS ON TOP of boulders
                // (only lateral/random diffusion percolates inward).
                dy = -1;
                let below_y = i32(y) - 1;
                if (below_y >= 0 && gamma_count(u32(below_y) * GRID_SIZE + x) > 0u) {
                    continue;
                }
            } else {
                continue;
            }
            let nx = i32(x) + dx;
            let ny = i32(y) + dy;
            if (nx < 0 || ny < 0 || nx >= i32(GRID_SIZE) || ny >= i32(GRID_SIZE)) { continue; }
            let t_idx = u32(ny) * GRID_SIZE + u32(nx);
            // INFILTRATION: currents can NEVER push monomers into rock —
            // only diffusion enters gamma-occupied cells, attenuated by
            // occupancy, so nutrients seep slowly into buried veins and
            // stay there until eroded or dug out.
            let g_tgt = gamma_count(t_idx);
            // PORE CAPACITY: a cell can hold at most chem_capacity(gamma)
            // monomers (all channels). Full or solid targets refuse hops.
            let tgt_total = chem_cell_total(t_idx);
            if (tgt_total >= chem_capacity(g_tgt)) {
                // Overpressure percolation: an over-capacity cell may still
                // push into a LESS loaded neighbor, so excess deep inside a
                // massif migrates outward to the surface instead of sticking.
                // Random-walk percolation (<=, not <): in a uniform rubble
                // field strict inequality froze everything in place.
                if (!(src_over && tgt_total <= src_total)) { continue; }
            }
            if (g_tgt > 0u) {
                if (is_adv) { continue; }
                let gperm = 1.0 / (1.0 + GAMMA_POROSITY_K * f32(g_tgt));
                if (hash_f32(h ^ 0x27D4EB2Fu) >= gperm) { continue; }
            }
            // Hops preserve the monomer's state (act or spent).
            if (chem_take_state_one(slot, pick_spent)) {
                chem_add_state(t_idx, ch, 1u, pick_spent);
            }
        }
    }
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
    let gradient = vec2<f32>(dx, dy) * inv_cell_size;

    // IMPORTANT: gamma_slope is *terrain-only*.
    // Do NOT bake global vector forces (gravity/wind) into this buffer, because other
    // systems (e.g. fluid permeability/obstacles) interpret slope magnitude as terrain steepness.
    // If you want gravity to affect a particular sensor/force, add it at the usage site.
    write_gamma_slope(idx, gradient);
}

@compute @workgroup_size(16, 16)
fn diffuse_trails(@builtin(global_invocation_id) gid: vec3<u32>) {
    // If trails are fully invisible, skip all trail simulation work.
    // NOTE: Sensors no longer depend on trail_grid, so this is safe.
    if (params.trail_opacity <= 0.0) {
        return;
    }

    let x = gid.x;
    let y = gid.y;

    if (x >= GRID_SIZE || y >= GRID_SIZE) {
        return;
    }

    let idx = y * GRID_SIZE + x;

    // Prepare the injection buffer for dye-style trails:
    // - start from the currently displayed trail_grid
    // - apply optional mild diffusion
    // - apply decay
    // Agents then deposit into trail_grid_inject during process_agents.
    let current_raw = trail_grid[idx];
    let current = vec4<f32>(
        clamp(current_raw.xyz, vec3<f32>(0.0), vec3<f32>(1.0)),
        sanitize_f32(current_raw.w)
    );
    let strength = clamp(params.trail_diffusion, 0.0, 1.0);
    let decay = clamp(params.trail_decay, 0.0, 1.0);

    let xi = i32(x);
    let yi = i32(y);
    let x_l = u32(clamp(xi - 1, 0, i32(GRID_SIZE) - 1));
    let x_r = u32(clamp(xi + 1, 0, i32(GRID_SIZE) - 1));
    let y_u = u32(clamp(yi - 1, 0, i32(GRID_SIZE) - 1));
    let y_d = u32(clamp(yi + 1, 0, i32(GRID_SIZE) - 1));

    let l_raw = trail_grid[y * GRID_SIZE + x_l];
    let r_raw = trail_grid[y * GRID_SIZE + x_r];
    let u_raw = trail_grid[y_u * GRID_SIZE + x];
    let d_raw = trail_grid[y_d * GRID_SIZE + x];
    let l = vec4<f32>(clamp(l_raw.xyz, vec3<f32>(0.0), vec3<f32>(1.0)), sanitize_f32(l_raw.w));
    let r = vec4<f32>(clamp(r_raw.xyz, vec3<f32>(0.0), vec3<f32>(1.0)), sanitize_f32(r_raw.w));
    let u = vec4<f32>(clamp(u_raw.xyz, vec3<f32>(0.0), vec3<f32>(1.0)), sanitize_f32(u_raw.w));
    let d = vec4<f32>(clamp(d_raw.xyz, vec3<f32>(0.0), vec3<f32>(1.0)), sanitize_f32(d_raw.w));
    let neighbor_blur = (l + r + u + d) * 0.25;

    let mixed = mix(current, neighbor_blur, strength);
    let faded_rgb = clamp(mixed.xyz * decay, vec3<f32>(0.0), vec3<f32>(1.0));
    let faded_energy = sanitize_f32(mixed.w) * decay;
    trail_grid_inject[idx] = vec4<f32>(faded_rgb, faded_energy);
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
        gamma = max(sample_grid_bicubic(world_pos, 2u), 0.0);
    } else if (params.grid_interpolation == 1u) {
        // Bilinear (smooth)
        alpha = clamp(sample_grid_bilinear(world_pos, 0u), 0.0, 1.0);
        beta = clamp(sample_grid_bilinear(world_pos, 1u), 0.0, 1.0);
        gamma = max(sample_grid_bilinear(world_pos, 2u), 0.0);
    } else {
        // Nearest neighbor (pixelated)
        let cidx_n = grid_index(world_pos);
        alpha = clamp(chem_pool_count_f(cidx_n, 0u), 0.0, 1.0);
        beta = clamp(chem_pool_count_f(cidx_n, 1u), 0.0, 1.0);
        gamma = max(read_gamma_height(grid_index(world_pos)), 0.0);
    }

    // Within-pool monomer ratios (nearest cell) tint the pool colors so the
    // four monomers are visually distinguishable: A vs U and G vs C get the
    // base color vs. a channel-swizzled variant of it.
    let cc_idx = grid_index(world_pos);
    // Colors show ACTIVATED monomers; spent matter renders as grey below.
    let cc = vec4<f32>(
        f32(chem_act_count(cc_idx, 0u)),
        f32(chem_act_count(cc_idx, 1u)),
        f32(chem_act_count(cc_idx, 2u)),
        f32(chem_act_count(cc_idx, 3u))
    ) * QUANTA_VIS;
    let cc_spent = (f32(chem_spent_count(cc_idx, 0u))
        + f32(chem_spent_count(cc_idx, 1u))
        + f32(chem_spent_count(cc_idx, 2u))
        + f32(chem_spent_count(cc_idx, 3u))) * QUANTA_VIS;
    let pool_au = cc.x + cc.y;
    let pool_gc = cc.z + cc.w;
    let ratio_u = select(0.5, cc.y / max(pool_au, 1e-6), pool_au > 1e-6);
    let ratio_c = select(0.5, cc.w / max(pool_gc, 1e-6), pool_gc > 1e-6);

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
    } else if (params.nucleotide_debug_mask != 0u) {
        // Nucleotide debug view (keys 1-4): show raw per-monomer concentrations
        // as RGBW — A=red, U=green, G=blue, C=white. Multiple channels stack.
        // Keys 1-4: ACTIVATED monomers only, per-nucleotide palette
        // (A=red, U=yellow, G=green, C=blue).
        let m = params.nucleotide_debug_mask;
        var dbg = vec3<f32>(0.0);
        if ((m & 1u) != 0u) { dbg += vec3<f32>(1.0, 0.15, 0.1) * clamp(cc.x, 0.0, 1.0); }
        if ((m & 2u) != 0u) { dbg += vec3<f32>(1.0, 0.85, 0.1) * clamp(cc.y, 0.0, 1.0); }
        if ((m & 4u) != 0u) { dbg += vec3<f32>(0.15, 0.9, 0.25) * clamp(cc.z, 0.0, 1.0); }
        if ((m & 8u) != 0u) { dbg += vec3<f32>(0.2, 0.45, 1.0) * clamp(cc.w, 0.0, 1.0); }
        // Key 6: isolated gamma view — raw quantized terrain height as
        // grayscale (1 quantum = 0.25 grey step, 4+ = white). Combines
        // with keys 1-4 if both are toggled.
        if ((m & 16u) != 0u) { dbg += vec3<f32>(1.0, 1.0, 1.0) * clamp(gamma, 0.0, 1.0); }
        // Key 8: UV radiation map (violet), scaled by the UV strength slider.
        if ((m & 64u) != 0u) {
            let uv_l = uv_light_at_world(world_pos);
            dbg += vec3<f32>(0.7, 0.3, 1.0) * clamp(uv_l * max(params.dye_alpha_color_r, 0.0), 0.0, 1.0);
        }
        base_color = dbg;
    } else {
        // New visualization system: composite channels with blend modes

        // Deposit layer: crisp quantized monomer counts with the
        // per-nucleotide palette (matches the dye renderer): A=red,
        // U=yellow, G=green, C=blue, dimmed to 50% so agents stay
        // readable on top. alpha_show gates A/U, beta_show gates G/C.
        {
            // Two-tone energy palette (matches the dye compositor):
            // ACTIVATED = gold, SPENT = complementary cool blue. Mixed
            // per-nucleotide hues averaged to grey and hid the spent
            // state; per-channel identity lives on debug keys 1-4.
            var act_sum = 0.0;
            if (params.alpha_show != 0u) { act_sum += cc.x + cc.y; }
            if (params.beta_show != 0u) { act_sum += cc.z + cc.w; }
            // Hue by activated FRACTION (blue = spent, gold = charged),
            // brightness by total amount — additive gold+blue washed to
            // grey since cells hold both states.
            let total_amt = act_sum + cc_spent;
            let act_frac = smoothstep(0.25, 0.75, act_sum / max(total_amt, 1e-5));
            let hue = mix(vec3<f32>(0.05, 0.3, 0.85), vec3<f32>(1.0, 0.62, 0.05), act_frac);
            let inten = pow(clamp(total_amt, 0.0, 1.0), params.alpha_gamma_adjust);
            // Brightness slider (repurposed _pad_dye_alpha_color slot).
            // Opacity blend (matches the compositor).
            base_color = mix(base_color, hue, clamp(inten * clamp(params._pad_dye_alpha_color, 0.0, 1.0), 0.0, 1.0));
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
        let current_stamp = params.epoch + 1u;
        let scale = f32(SPATIAL_GRID_SIZE) / f32(SIM_SIZE);
        let grid_x = u32(clamp(world_x * scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));
        let grid_y = u32(clamp(world_y * scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));
        let grid_cell = grid_y * SPATIAL_GRID_SIZE + grid_x;

        // Skip stale cells from prior epochs.
        let stamp = atomicLoad(&agent_spatial_grid[spatial_epoch_index(grid_cell)]);
        if (stamp != current_stamp) {
            return;
        }
        let raw_agent_id = atomicLoad(&agent_spatial_grid[spatial_id_index(grid_cell)]);
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

    // ====== TRAIL OVERLAY ======
    // trail_grid stores RGB dye trail + energy trail in W (unclamped).
    let trail_cell = trail_grid[grid_index(world_pos)];
    let trail_color = clamp(trail_cell.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    let trail_energy = max(trail_cell.w, 0.0);

    // Trail visualization modes:
    // 0 = normal (add RGB trails onto base)
    // 1 = trails-only (RGB)
    // 2 = trails-only (energy)
    if (params.trail_show == 1u) {
        let trail_only = trail_color * clamp(params.trail_opacity, 0.0, 1.0);
        visual_grid[visual_idx] = vec4<f32>(trail_only, 1.0);
    } else if (params.trail_show == 2u) {
        // Energy is stored as an unclamped accumulator.
        // Do NOT clamp the opacity gain here (only clamp output via tone mapping).
        let w_raw = trail_cell.w;
        // WGSL portability: avoid isNan/isInf (not available on all validator versions).
        let w_is_nan = w_raw != w_raw;
        let w_is_inf = abs(w_raw) > 1.0e30;
        if (w_is_nan || w_is_inf) {
            // Bright magenta = bad/invalid energy values.
            visual_grid[visual_idx] = vec4<f32>(1.0, 0.0, 1.0, 1.0);
        } else {
            let exposure = max(params.trail_opacity, 0.0);
            let e = max(w_raw, 0.0);
            let x = sqrt(e) * exposure;
            // Exposure mapping: makes tiny values visible without hard clamping.
            let v = 1.0 - exp(-x);
            visual_grid[visual_idx] = vec4<f32>(vec3<f32>(v), 1.0);
        }
    } else {
        let blended_color = clamp(
            base_color + trail_color * clamp(params.trail_opacity, 0.0, 1.0),
            vec3<f32>(0.0),
            vec3<f32>(1.0)
        );
        visual_grid[visual_idx] = vec4<f32>(blended_color, 1.0);
    }
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct InspectorOverlayVertexOutput {
    @builtin(position) position: vec4<f32>,
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

@vertex
fn vs_inspector_overlay(@builtin(vertex_index) vid: u32) -> InspectorOverlayVertexOutput {
    // Quad covering only the inspector area (rightmost INSPECTOR_WIDTH pixels)
    let safe_width = max(params.window_width, 1.0);
    let inspector_w = f32(INSPECTOR_WIDTH);
    let x0 = 1.0 - 2.0 * (inspector_w / safe_width);

    var positions = array<vec2<f32>, 6>(
        vec2<f32>(x0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(x0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(x0, 1.0)
    );

    let pos = positions[vid];
    var out: InspectorOverlayVertexOutput;
    out.position = vec4<f32>(pos, 0.0, 1.0);
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

@group(0) @binding(16)
var<storage, read> fluid_velocity_render: array<vec2<f32>>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample visual texture (already has environment + agents + fluid dye composited)
    let color = textureSample(visual_tex, visual_sampler, in.uv);

    // Fluid visualization is now handled in composite.wgsl compute shader
    // which directly blends dye markers onto visual_grid before rendering

    return vec4<f32>(color.rgb, 1.0);
}

@fragment
fn fs_inspector_overlay(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    if (params.draw_enabled == 0u) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }
    if (params.selected_agent_index == 0xFFFFFFFFu) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }

    let safe_width = max(params.window_width, 1.0);
    let safe_height = max(params.window_height, 1.0);
    let window_width = u32(safe_width);
    let window_height = u32(safe_height);

    let x = u32(pos.x);
    let y = u32(pos.y);
    if (y >= window_height) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Map screen x into inspector-local x
    let start_x = select(window_width - INSPECTOR_WIDTH, 0u, window_width < INSPECTOR_WIDTH);
    if (x < start_x) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let local_x = x - start_x;

    return inspector_panel_pixel(local_x, y);
}

// ============================================================================
// SPECIES HASH (phenotype fingerprint for fast species counting)
// ============================================================================

// Species hash buffer lives in group 1 to stay within the 16 storage buffer limit on group 0.
@group(1) @binding(0)
var<storage, read_write> species_hash_buffer: array<u32>;

// Matter telemetry: free monomer totals per channel (A/U/G/C), reduced at the
// species-sampling cadence. Cleared CPU-side before each dispatch. Locked
// matter (strands, paired complements, energy) is summed on the CPU from the
// agents readback that species counting already performs.
@group(1) @binding(1)
// [0..3] free per channel, [4] free inside SOLID rock, [7] in rubble, [5] free in the
// bottom 10% of rows (low y = screen bottom), [6] activated free total.
var<storage, read_write> matter_totals: array<atomic<u32>, 8>;

const MATTER_REDUCE_THREADS: u32 = 4096u;

@compute @workgroup_size(64)
fn reduce_matter_chem(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= MATTER_REDUCE_THREADS) {
        return;
    }
    var s0 = 0u;
    var s1 = 0u;
    var s2 = 0u;
    var s3 = 0u;
    var s_rock = 0u;
    var s_bottom = 0u;
    var s_act = 0u;
    var s_rubble = 0u;
    let total_cells = GRID_SIZE * GRID_SIZE;
    var c = tid;
    loop {
        if (c >= total_cells) { break; }
        let base = c * 4u;
        let v0 = atomicLoad(&chem_grid[base]);
        let v1 = atomicLoad(&chem_grid[base + 1u]);
        let v2 = atomicLoad(&chem_grid[base + 2u]);
        let v3 = atomicLoad(&chem_grid[base + 3u]);
        let a_cell = (v0 & CHEM_STATE_MASK) + (v1 & CHEM_STATE_MASK) + (v2 & CHEM_STATE_MASK) + (v3 & CHEM_STATE_MASK);
        let t0 = (v0 & CHEM_STATE_MASK) + (v0 >> 16u);
        let t1 = (v1 & CHEM_STATE_MASK) + (v1 >> 16u);
        let t2 = (v2 & CHEM_STATE_MASK) + (v2 >> 16u);
        let t3 = (v3 & CHEM_STATE_MASK) + (v3 >> 16u);
        s0 += t0;
        s1 += t1;
        s2 += t2;
        s3 += t3;
        let t_cell = t0 + t1 + t2 + t3;
        let g_c = gamma_count(c);
        if (g_c >= GAMMA_SOLID_THRESHOLD) { s_rock += t_cell; } else if (g_c > 0u) { s_rubble += t_cell; }
        if (c / GRID_SIZE < GRID_SIZE / 10u) { s_bottom += t_cell; }
        s_act += a_cell;
        c += MATTER_REDUCE_THREADS;
    }
    atomicAdd(&matter_totals[0], s0);
    atomicAdd(&matter_totals[1], s1);
    atomicAdd(&matter_totals[2], s2);
    atomicAdd(&matter_totals[3], s3);
    atomicAdd(&matter_totals[4], s_rock);
    atomicAdd(&matter_totals[5], s_bottom);
    atomicAdd(&matter_totals[6], s_act);
    atomicAdd(&matter_totals[7], s_rubble);
}

@compute @workgroup_size(256)
fn compute_species_hash(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent_id = gid.x;
    if (agent_id >= params.max_agents) {
        return;
    }

    let alive = agents_out[agent_id].alive;
    let body_count = agents_out[agent_id].body_count;

    if (alive == 0u || body_count < 3u) {
        species_hash_buffer[agent_id] = 0u;
        return;
    }

    // Two-layer species hash with quantized tolerance:
    // 1) Bag-of-parts: counts of each part type (order-independent composition)
    // 2) Bigrams: counts of consecutive part-type pairs (captures local structure/order)
    // Both layers are quantized by bucket_size for distance tolerance.
    let bucket = max(params.species_bucket_size, 1u);
    let bc = min(body_count, MAX_BODY_PARTS);

    // Layer 1: part-type histogram
    var counts: array<u32, 48>;
    for (var i = 0u; i < 48u; i++) {
        counts[i] = 0u;
    }
    for (var i = 0u; i < bc; i++) {
        let pt = get_base_part_type(agents_out[agent_id].body[i].part_type);
        if (pt < 48u) {
            counts[pt] += 1u;
        }
    }

    // Layer 2: bigram histogram (consecutive part-type pairs)
    // Hash bigrams directly into h to avoid a huge 48x48 array.
    // Use a rolling hash: for each consecutive pair, mix the pair identity into h.
    let quantized_bc = ((body_count + bucket / 2u) / bucket) * bucket;

    var h = 2166136261u;
    h = h ^ quantized_bc;
    h = h * 16777619u;

    // Mix in quantized unigram counts (composition)
    for (var i = 0u; i < 48u; i++) {
        let q = ((counts[i] + bucket / 2u) / bucket) * bucket;
        if (q > 0u) {
            h = h ^ (i | (q << 8u));
            h = h * 16777619u;
        }
    }

    // Mix in bigrams (structure/order) — quantized by skipping every `bucket` parts
    // so small insertions don't shift all bigrams
    var prev_pt = get_base_part_type(agents_out[agent_id].body[0u].part_type);
    for (var i = 1u; i < bc; i++) {
        let cur_pt = get_base_part_type(agents_out[agent_id].body[i].part_type);
        // Quantize position: only include bigram at bucket boundaries
        // This makes the bigram set stable under small insertions/deletions
        if ((i % bucket) == 0u) {
            let bigram = prev_pt | (cur_pt << 8u);
            h = h ^ bigram;
            h = h * 16777619u;
        }
        prev_pt = cur_pt;
    }

    species_hash_buffer[agent_id] = select(h, 1u, h == 0u);
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

    // Sample rain map from texture (RG channels = alpha/beta rain multipliers)
    let rain_map_sample = textureLoad(rain_map_tex, vec2<u32>(x, y), 0);
    let alpha_rain_map = rain_map_sample.r;
    let beta_rain_map = rain_map_sample.g;

    // Use constant values for fast startup (can be overridden by loading terrain images)
    // Initialize with chemistry values and rain maps from texture
    // Primordial soup: equal amounts of all four monomers, quantized. The
    // universe's budget is (alpha_range + beta_range); each element gets a
    // quarter, converted to whole monomers with probabilistic rounding.
    // Rain map multipliers stay in rain_map_tex (read directly at use sites).
    let _unused_rain = vec2<f32>(alpha_rain_map, beta_rain_map);
    let total_init = environment_init.alpha_range.x + environment_init.beta_range.x;
    let qf = total_init * 0.25 * f32(CHEM_QUANTA_REF);
    let whole = u32(floor(qf));
    let qfrac = qf - f32(whole);
    for (var c4 = 0u; c4 < 4u; c4++) {
        var q = whole;
        if (hash_f32(idx * 747796405u ^ (c4 * 0x9E3779B9u)) < qfrac) { q += 1u; }
        atomicStore(&chem_grid[idx * 4u + c4], q);
    }
    atomicStore(&gamma_grid[idx], 0u);

    write_gamma_slope(idx, environment_init.slope_pair);

    trail_grid[idx] = environment_init.trail_values;
    trail_grid_inject[idx] = environment_init.trail_values;
}

// ============================================================================
// SPAWN/DEATH MANAGEMENT SHADERS
// ============================================================================

// Merge spawned agents into main buffer
@compute @workgroup_size(64)
fn merge_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
    let spawn_id = gid.x;
    // Clamp spawn count to buffer size (2000) to avoid OOB reads if more agents tried to spawn than fit
    let spawn_count = min(atomicLoad(&spawn_debug_counters[0]), 2000u);

    if (spawn_id >= spawn_count) {
        return;
    }

    // Append to end of compacted alive array using alive counter as running size
    let target_index = atomicAdd(&spawn_debug_counters[2], 1u);
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

    let alive_now = atomicLoad(&spawn_debug_counters[2]);
    if (alive_now >= params.max_agents) {
        return;
    }

    var spawn_index: u32 = 0u;
    loop {
        let current_spawn = atomicLoad(&spawn_debug_counters[0]);
        if (current_spawn >= 2000u) {
            return;
        }
        if (alive_now + current_spawn >= params.max_agents) {
            return;
        }
        let result = atomicCompareExchangeWeak(&spawn_debug_counters[0], current_spawn, current_spawn + 1u);
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

    let dst = &new_agents[spawn_index];
    (*dst).position = clamp_position(spawn_pos);
    (*dst).velocity = vec2<f32>(0.0);
    (*dst).rotation = rotation;
    atomicStore(&(*dst).energy, energy_to_u32(max(request.energy, 0.0)));
    (*dst).energy_capacity = 0.0; // Will be calculated after morphology builds
    (*dst).torque_debug = 0.0;
    (*dst).alive = 1u;
    (*dst).body_count = 0u;
    (*dst).pairing_counter = 0u;
    (*dst).is_selected = 0u;
    (*dst).generation = 0u;
    (*dst).age = 0u;
    (*dst).total_mass = 0.0;
    (*dst).poison_resistant_count = 0u;

    // If flags bit 0 set, use provided genome override (packed)
    if ((request.flags & 1u) != 0u) {
        (*dst).gene_length = request.genome_override_len;
        (*dst).genome_offset = request.genome_override_offset;
        // Manual unroll: some backends (via naga) require constant indexing into
        // fixed-size arrays inside storage-structs.
        (*dst).genome_packed[0u] = request.genome_override_packed[0u];
        (*dst).genome_packed[1u] = request.genome_override_packed[1u];
        (*dst).genome_packed[2u] = request.genome_override_packed[2u];
        (*dst).genome_packed[3u] = request.genome_override_packed[3u];
        (*dst).genome_packed[4u] = request.genome_override_packed[4u];
        (*dst).genome_packed[5u] = request.genome_override_packed[5u];
        (*dst).genome_packed[6u] = request.genome_override_packed[6u];
        (*dst).genome_packed[7u] = request.genome_override_packed[7u];
        (*dst).genome_packed[8u] = request.genome_override_packed[8u];
        (*dst).genome_packed[9u] = request.genome_override_packed[9u];
        (*dst).genome_packed[10u] = request.genome_override_packed[10u];
        (*dst).genome_packed[11u] = request.genome_override_packed[11u];
        (*dst).genome_packed[12u] = request.genome_override_packed[12u];
        (*dst).genome_packed[13u] = request.genome_override_packed[13u];
        (*dst).genome_packed[14u] = request.genome_override_packed[14u];
        (*dst).genome_packed[15u] = request.genome_override_packed[15u];
    } else {
    // Create centered variable-length genome with implicit 'X' padding on both sides.
    // Active region length in [MIN_GENE_LENGTH, GENOME_LENGTH].
        genome_seed = hash(genome_seed ^ base_seed);
        let gene_span = GENOME_LENGTH - MIN_GENE_LENGTH;
        let gene_len = MIN_GENE_LENGTH + (hash(genome_seed) % (gene_span + 1u));
        let left_pad = (GENOME_LENGTH - gene_len) / 2u;

        (*dst).gene_length = gene_len;
        (*dst).genome_offset = left_pad;
        for (var w = 0u; w < GENOME_PACKED_WORDS; w++) {
            (*dst).genome_packed[w] = 0u;
        }

        for (var k = 0u; k < gene_len; k++) {
            genome_seed = hash(genome_seed ^ (k * 1664525u + 1013904223u));
            let base_ascii = get_random_rna_base(genome_seed);
            let code = genome_base_ascii_to_2bit(base_ascii);
            let bi = left_pad + k;
            let wi = bi / GENOME_BASES_PER_PACKED_WORD;
            let bit_index = (bi % GENOME_BASES_PER_PACKED_WORD) * 2u;
            (*dst).genome_packed[wi] = (*dst).genome_packed[wi] | (code << bit_index);
        }
    }

    for (var i = 0u; i < MAX_BODY_PARTS; i++) {
        (*dst).body[i].pos = vec2<f32>(0.0);
        (*dst).body[i].data = 0.0;
        (*dst).body[i].part_type = 0u;
        (*dst).body[i].alpha_signal = 0.0;
        (*dst).body[i].beta_signal = 0.0;
        (*dst).body[i]._pad = vec2<f32>(0.0);
    }
}

@compute @workgroup_size(256)
fn initialize_dead_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
    let alive_total = atomicLoad(&spawn_debug_counters[2]);
    let max_agents = params.max_agents;
    if (alive_total >= max_agents) {
        return;
    }

    // This kernel is dispatched only for the dead tail; gid.x is relative to `alive_total`.
    let idx = alive_total + gid.x;
    if (idx >= max_agents) {
        return;
    }

    // IMPORTANT: Avoid copying the full Agent struct (genome + body arrays).
    // We only need to ensure a few fields are reset for dead slots.
    agents_out[idx].alive = 0u;
    agents_out[idx].body_count = 0u;
    atomicStore(&agents_out[idx].energy, 0u);
    agents_out[idx].velocity = vec2<f32>(0.0);
    agents_out[idx].pairing_counter = 0u;
}

// Compact living agents from input to output, producing a packed array at the front
@compute @workgroup_size(64)
fn compact_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
    let agent_id = gid.x;

    // IMPORTANT:
    // Compact must cover a bit beyond `params.agent_count` to avoid dropping newborns
    // when CPU-side `agent_count` readback lags by a frame.
    // However, scanning the full `params.max_agents` would also pick up stale/garbage tail
    // slots (e.g. after loading a snapshot with fewer agents), which can cause runaway
    // "ghost" agents and continuous reproduction.
    let scan_limit = min(params.agent_count + 2000u, params.max_agents);
    if (agent_id >= scan_limit) {
        return;
    }

    let agent = agents_in[agent_id];
    if (agent.alive != 0u) {
        let idx = atomicAdd(&spawn_debug_counters[2], 1u);
        if (idx < params.max_agents) {
            agents_out[idx].position = agent.position;
            agents_out[idx].velocity = agent.velocity;
            agents_out[idx].rotation = agent.rotation;
            atomicStore(&agents_out[idx].energy, agent.energy);
            agents_out[idx].energy_capacity = agent.energy_capacity;
            agents_out[idx].torque_debug = agent.torque_debug;
            agents_out[idx].morphology_origin = agent.morphology_origin;
            agents_out[idx].alive = agent.alive;
            agents_out[idx].body_count = agent.body_count;
            agents_out[idx].pairing_counter = agent.pairing_counter;
            agents_out[idx].is_selected = agent.is_selected;
            agents_out[idx].generation = agent.generation;
            agents_out[idx].age = agent.age;
            agents_out[idx].total_mass = agent.total_mass;
            agents_out[idx].poison_resistant_count = agent.poison_resistant_count;
            agents_out[idx].gene_length = agent.gene_length;
            agents_out[idx].genome_offset = agent.genome_offset;
            agents_out[idx].genome_packed = agent.genome_packed;
            agents_out[idx].body = agent.body;
        }
    }
}

// Reset spawn counter for next frame
@compute @workgroup_size(1)
fn reset_spawn_counter(@builtin(global_invocation_id) gid: vec3<u32>) {
    atomicStore(&spawn_debug_counters[0], 0u);
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
        // Amplitude scaling: when a level (> 0) is passed with noise, the
        // noise peaks at that level instead of 1.0 (scarce patchy worlds).
        if (value > 0.0) {
            output_value *= value;
        }
    }

    if (mode == 5u) {
        // RAIN — uniform ADDITIVE fill: every cell/channel gains `value`
        // expected quanta (probabilistic rounding). Used when Set World
        // Total must add matter: multiplicative scaling cannot create
        // quanta where there are none (0 x factor = 0), rain can.
        var dv5 = fluid_dye[idx];
        for (var mc = 0u; mc < 4u; mc++) {
            let slot = idx * 4u + mc;
            let whole5 = u32(floor(value));
            var nq5 = whole5;
            let fr5 = value - f32(whole5);
            if (hash_f32(idx * 747796405u ^ (mc * 0x9E3779B9u) ^ (seed * 668265263u)) < fr5) {
                nq5 += 1u;
            }
            // Terrain holds no monomers: rain only lands in open water.
            if (nq5 > 0u && gamma_count(idx) == 0u) {
                // Rain adds FRESH (activated) monomers.
                atomicAdd(&chem_grid[slot], nq5);
            }
            dv5[mc] = chem_act_f(idx, mc);
        }
        fluid_dye[idx] = dv5;
    } else if (mode == 4u) {
        // GOD MODE — scale the total monomer amount: activated and spent
        // counts are each multiplied by `value` with probabilistic rounding.
        let factor = max(output_value, 0.0);
        var dv4 = fluid_dye[idx];
        for (var mc = 0u; mc < 4u; mc++) {
            let slot = idx * 4u + mc;
            let cur = atomicLoad(&chem_grid[slot]);
            let cur_act = f32(cur & CHEM_STATE_MASK) * factor;
            let cur_spent = f32(cur >> 16u) * factor;
            let wa = u32(floor(cur_act));
            let ws = u32(floor(cur_spent));
            var na = wa;
            var ns = ws;
            if (hash_f32(idx * 747796405u ^ (mc * 0x9E3779B9u) ^ (seed * 668265263u)) < cur_act - f32(wa)) { na += 1u; }
            if (hash_f32(idx * 668265263u ^ (mc * 0x85EBCA6Bu) ^ (seed * 374761393u)) < cur_spent - f32(ws)) { ns += 1u; }
            atomicStore(&chem_grid[slot], na + (ns << 16u));
            dv4[mc] = f32(na) * QUANTA_VIS;
        }
        fluid_dye[idx] = dv4;
    } else if (mode == 1u) {
        // Terrain holds no monomers: generation writes 0 into gamma cells.
        write_chem_alpha(idx, select(output_value, 0.0, gamma_count(idx) > 0u));
        // Mirror the ACTUAL quantized ACTIVATED counts into the dye.
        var dv = fluid_dye[idx];
        dv.x = chem_act_f(idx, 0u);
        dv.y = chem_act_f(idx, 1u);
        fluid_dye[idx] = dv;
    } else if (mode == 2u) {
        write_chem_beta(idx, select(output_value, 0.0, gamma_count(idx) > 0u));
        var dv = fluid_dye[idx];
        dv.z = chem_act_f(idx, 2u);
        dv.w = chem_act_f(idx, 3u);
        fluid_dye[idx] = dv;
    } else if (mode == 3u) {
        write_gamma_height(idx, output_value);
    }
}

// ============================================================================
// MOTION BLUR (Applied BEFORE agents are rendered, blurs only background)
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
    let motion_scale = params.camera_zoom * time_scale * 0.25; // Halve blur length again
    let motion_vector = camera_motion * motion_scale;
    let motion_length = length(motion_vector);

    // Apply motion blur only if camera is moving significantly
    if (motion_length > 0.01) {
        let screen_pos = vec2<f32>(f32(x), f32(y));

        // Get current pixel color
        let base_color = visual_grid[visual_idx].xyz;

        // Take 8 samples in the flipped direction
        let sample_count = 8;
        var color_sum = base_color;

        // Simple hash for randomization
        let pixel_hash = hash(visual_idx * 73856093u + params.random_seed);

        for (var i = 1; i <= sample_count; i++) {
            // Sample distance along the blur vector (0.0 to 1.0 range)
            let sample_hash = hash(pixel_hash + u32(i) * 1664525u);
            let random_t = f32(sample_hash % 1000u) / 1000.0;

            // Sample along the motion vector (direction flipped vs. previous)
            let offset = motion_vector * random_t;
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

    let cell = y * SPATIAL_GRID_SIZE + x;
    atomicStore(&agent_spatial_grid[spatial_id_index(cell)], SPATIAL_GRID_EMPTY);
    atomicStore(&agent_spatial_grid[spatial_epoch_index(cell)], 0u);
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
    if (agent.alive == 0u || agent.energy == 0u) {
        return;
    }

    // Convert agent position (in SIM_SIZE space) to grid coordinates (SPATIAL_GRID_SIZE resolution)
    let scale = f32(SPATIAL_GRID_SIZE) / f32(SIM_SIZE);
    let grid_x = u32(clamp(agent.position.x * scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));
    let grid_y = u32(clamp(agent.position.y * scale, 0.0, f32(SPATIAL_GRID_SIZE - 1u)));
    let primary_cell = grid_y * SPATIAL_GRID_SIZE + grid_x;

    // Stamp is epoch+1 so 0 remains the "never written" value.
    let current_stamp = params.epoch + 1u;

    // IMPORTANT: the spatial grid stores stamp and id in two different atomics.
    // If we write the stamp first, readers can see `stamp == current_stamp` and
    // still read a stale id (from a previous epoch) before the id store lands.
    // To avoid that, we do a two-phase stamp:
    //   1) CAS stamp -> (current_stamp | IN_PROGRESS_BIT)
    //   2) store id
    //   3) store stamp -> current_stamp
    // Readers only accept `stamp == current_stamp`, so they ignore in-progress cells.
    let IN_PROGRESS_BIT: u32 = 0x80000000u;
    let STAMP_EPOCH_MASK: u32 = 0x7FFFFFFFu;

    // Try to claim the primary cell via epoch stamp.
    let primary_old_stamp_raw = atomicLoad(&agent_spatial_grid[spatial_epoch_index(primary_cell)]);
    var found = false;
    if ((primary_old_stamp_raw & STAMP_EPOCH_MASK) != current_stamp) {
        let primary_claim = atomicCompareExchangeWeak(
            &agent_spatial_grid[spatial_epoch_index(primary_cell)],
            primary_old_stamp_raw,
            current_stamp | IN_PROGRESS_BIT
        );
        if (primary_claim.exchanged) {
            atomicStore(&agent_spatial_grid[spatial_id_index(primary_cell)], agent_id);
            atomicStore(&agent_spatial_grid[spatial_epoch_index(primary_cell)], current_stamp);
            found = true;
        }
    }

    if (!found) {
        // Primary cell is occupied - search for nearest empty cell in a spiral pattern
        // This ensures all agents are findable even in crowded areas
        // found is already declared above.

        // Search in expanding square rings up to radius 5 (covers 11x11 area = 121 cells)
        for (var radius = 1u; radius <= 5u && !found; radius++) {
            // Top and bottom edges of the square
            for (var dx: i32 = -i32(radius); dx <= i32(radius) && !found; dx++) {
                // Top edge
                let check_x_top = i32(grid_x) + dx;
                let check_y_top = i32(grid_y) - i32(radius);
                if (check_x_top >= 0 && check_x_top < i32(SPATIAL_GRID_SIZE) &&
                    check_y_top >= 0 && check_y_top < i32(SPATIAL_GRID_SIZE)) {
                    let cell = u32(check_y_top) * SPATIAL_GRID_SIZE + u32(check_x_top);
                    let old_stamp_raw = atomicLoad(&agent_spatial_grid[spatial_epoch_index(cell)]);
                    if ((old_stamp_raw & STAMP_EPOCH_MASK) != current_stamp) {
                        let claim = atomicCompareExchangeWeak(
                            &agent_spatial_grid[spatial_epoch_index(cell)],
                            old_stamp_raw,
                            current_stamp | IN_PROGRESS_BIT
                        );
                        if (claim.exchanged) {
                            atomicStore(&agent_spatial_grid[spatial_id_index(cell)], agent_id);
                            atomicStore(&agent_spatial_grid[spatial_epoch_index(cell)], current_stamp);
                            found = true;
                        }
                    }
                }

                // Bottom edge (skip if radius == 0 to avoid duplicate)
                if (!found && radius > 0u) {
                    let check_x_bot = i32(grid_x) + dx;
                    let check_y_bot = i32(grid_y) + i32(radius);
                    if (check_x_bot >= 0 && check_x_bot < i32(SPATIAL_GRID_SIZE) &&
                        check_y_bot >= 0 && check_y_bot < i32(SPATIAL_GRID_SIZE)) {
                        let cell = u32(check_y_bot) * SPATIAL_GRID_SIZE + u32(check_x_bot);
                        let old_stamp_raw = atomicLoad(&agent_spatial_grid[spatial_epoch_index(cell)]);
                        if ((old_stamp_raw & STAMP_EPOCH_MASK) != current_stamp) {
                            let claim = atomicCompareExchangeWeak(
                                &agent_spatial_grid[spatial_epoch_index(cell)],
                                old_stamp_raw,
                                current_stamp | IN_PROGRESS_BIT
                            );
                            if (claim.exchanged) {
                                atomicStore(&agent_spatial_grid[spatial_id_index(cell)], agent_id);
                                atomicStore(&agent_spatial_grid[spatial_epoch_index(cell)], current_stamp);
                                found = true;
                            }
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
                    let cell = u32(check_y_left) * SPATIAL_GRID_SIZE + u32(check_x_left);
                    let old_stamp_raw = atomicLoad(&agent_spatial_grid[spatial_epoch_index(cell)]);
                    if ((old_stamp_raw & STAMP_EPOCH_MASK) != current_stamp) {
                        let claim = atomicCompareExchangeWeak(
                            &agent_spatial_grid[spatial_epoch_index(cell)],
                            old_stamp_raw,
                            current_stamp | IN_PROGRESS_BIT
                        );
                        if (claim.exchanged) {
                            atomicStore(&agent_spatial_grid[spatial_id_index(cell)], agent_id);
                            atomicStore(&agent_spatial_grid[spatial_epoch_index(cell)], current_stamp);
                            found = true;
                        }
                    }
                }

                // Right edge
                if (!found) {
                    let check_x_right = i32(grid_x) + i32(radius);
                    let check_y_right = i32(grid_y) + dy;
                    if (check_x_right >= 0 && check_x_right < i32(SPATIAL_GRID_SIZE) &&
                        check_y_right >= 0 && check_y_right < i32(SPATIAL_GRID_SIZE)) {
                        let cell = u32(check_y_right) * SPATIAL_GRID_SIZE + u32(check_x_right);
                        let old_stamp_raw = atomicLoad(&agent_spatial_grid[spatial_epoch_index(cell)]);
                        if ((old_stamp_raw & STAMP_EPOCH_MASK) != current_stamp) {
                            let claim = atomicCompareExchangeWeak(
                                &agent_spatial_grid[spatial_epoch_index(cell)],
                                old_stamp_raw,
                                current_stamp | IN_PROGRESS_BIT
                            );
                            if (claim.exchanged) {
                                atomicStore(&agent_spatial_grid[spatial_id_index(cell)], agent_id);
                                atomicStore(&agent_spatial_grid[spatial_epoch_index(cell)], current_stamp);
                                found = true;
                            }
                        }
                    }
                }
            }
        }

        // If still not found after searching 5 rings, agent won't be in spatial grid this frame
        // This is acceptable as it will retry next frame - prevents infinite loops
    }
}

