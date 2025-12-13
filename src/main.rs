// Ribossome - GPU-Accelerated Artificial Life Simulator
// Copyright (c) 2025 Filipe da Veiga Ventura Alves
// Licensed under MIT License

use bytemuck::{Pod, Zeroable};
use std::collections::VecDeque;
use egui_plot::{Line, Plot, PlotPoints};
use egui::{Color32, ColorImage, TextureHandle, TextureId, TextureOptions};
use egui_wgpu::ScreenDescriptor;
use image::imageops::FilterType;
use image::GrayImage;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use wgpu::util::DeviceExt;
use winit::{
    event::{ElementState, Event, KeyEvent, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

const SIM_SIZE: f32 = 30720.0; // World size (must match shader SIM_SIZE)
const GRID_DIM: usize = (SIM_SIZE / 15.0) as usize; // Environment grid resolution (alpha/beta/gamma) - derived from SIM_SIZE
const GRID_CELL_COUNT: usize = GRID_DIM * GRID_DIM;
const GRID_DIM_U32: u32 = GRID_DIM as u32;
const SPATIAL_GRID_DIM: usize = (SIM_SIZE / 30.0) as usize; // Spatial hash grid for agent collision detection - derived from SIM_SIZE
const SPATIAL_GRID_CELL_COUNT: usize = SPATIAL_GRID_DIM * SPATIAL_GRID_DIM;
const DIFFUSE_WG_SIZE_X: u32 = 16;
const DIFFUSE_WG_SIZE_Y: u32 = 16;
const CLEAR_WG_SIZE_X: u32 = 16;
const CLEAR_WG_SIZE_Y: u32 = 16;
const SLOPE_WG_SIZE_X: u32 = 16;
const SLOPE_WG_SIZE_Y: u32 = 16;
const TERRAIN_FORCE_SCALE: f32 = 250.0;
const GAMMA_CORRECTION_EXPONENT: f32 = 2.2;
const SETTINGS_FILE_NAME: &str = "simulation_settings.json";
const AUTO_SNAPSHOT_FILE_NAME: &str = "autosave_snapshot.png";
const AUTO_SNAPSHOT_INTERVAL: u64 = 10000; // Save every 10,000 epochs
const RAIN_THUMB_SIZE: usize = 128;

// Shared genome/body sizing (must stay in sync with shader constants)
const MAX_BODY_PARTS: usize = 64;
const GENOME_BYTES: usize = 256; // ASCII bases including padding
const GENOME_WORDS: usize = GENOME_BYTES / std::mem::size_of::<u32>();
const PACKED_GENOME_WORDS: usize = GENOME_BYTES / 16; // 16 bases per packed u32
const MIN_GENE_LENGTH: usize = 6;
const MAX_SPAWN_REQUESTS: usize = 2000;

// RGB colors per amino acid, kept in sync with shader get_amino_acid_properties()
const AMINO_COLORS: [[f32; 3]; 20] = [
    [0.3, 0.3, 0.3],    // A
    [1.0, 0.0, 0.0],    // C (beta sensor)
    [0.35, 0.35, 0.35], // D
    [0.4, 0.4, 0.4],    // E
    [1.0, 0.4, 0.7],    // F (poison resistant) - pink, very fat
    [0.4, 0.4, 0.4],    // G (structural - smallest amino acid)
    [0.28, 0.28, 0.28], // H
    [0.38, 0.38, 0.38], // I
    [1.0, 1.0, 0.0],    // K (mouth)
    [0.0, 1.0, 1.0],    // L (chiral flipper) - cyan
    [0.8, 0.8, 0.2],    // M
    [0.47, 0.63, 0.47], // N (enabler) - light green
    [0.0, 0.39, 1.0],   // P (propeller) - blue
    [0.34, 0.34, 0.34], // Q
    [0.29, 0.29, 0.29], // R
    [0.0, 1.0, 0.0],    // S (alpha sensor)
    [0.6, 0.2, 0.8],    // T (energy sensor) - purple
    [0.0, 1.0, 1.0],    // V (displacer) - cyan
    [1.0, 0.65, 0.0],   // W (storage) - orange
    [0.26, 0.26, 0.26], // Y (dual-channel condenser)
];

#[derive(Clone, Copy)]
struct AminoVisualFlags {
    is_mouth: bool,
    is_alpha_sensor: bool,
    is_beta_sensor: bool,
    is_energy_sensor: bool,
    is_inhibitor: bool,
    is_propeller: bool,
    is_condenser: bool,
    is_displacer: bool,
}

const DEFAULT_AMINO_FLAGS: AminoVisualFlags = AminoVisualFlags {
    is_mouth: false,
    is_alpha_sensor: false,
    is_beta_sensor: false,
    is_energy_sensor: false,
    is_inhibitor: false,
    is_propeller: false,
    is_condenser: false,
    is_displacer: false,
};

struct StartupProfiler {
    enabled: bool,
    start: std::time::Instant,
    last: std::time::Instant,
}

impl StartupProfiler {
    fn new() -> Self {
        let enabled = std::env::var("ALSIM_TRACE_STARTUP")
            .map(|value| value != "0")
            .unwrap_or(false);
        let now = std::time::Instant::now();
        Self {
            enabled,
            start: now,
            last: now,
        }
    }

    fn mark(&mut self, label: &str) {
        if !self.enabled {
            return;
        }

        let now = std::time::Instant::now();
        let delta_ms = now.duration_since(self.last).as_secs_f64() * 1000.0;
        let total_ms = now.duration_since(self.start).as_secs_f64() * 1000.0;
        println!(
            "[startup] {:>28}: +{:7.2} ms (total {:7.2} ms)",
            label, delta_ms, total_ms
        );
        self.last = now;
    }
}

const AMINO_FLAGS: [AminoVisualFlags; 20] = [
    AminoVisualFlags {
        is_mouth: false,
        is_alpha_sensor: false,
        is_beta_sensor: false,
        is_energy_sensor: false,
        is_inhibitor: false,
        is_propeller: false,
        is_condenser: false,
        is_displacer: false,
    }, // A
    AminoVisualFlags {
        is_mouth: false,
        is_alpha_sensor: false,
        is_beta_sensor: true,
        is_energy_sensor: false,
        is_inhibitor: false,
        is_propeller: false,
        is_condenser: false,
        is_displacer: false,
    }, // C (beta sensor)
    AminoVisualFlags {
        is_mouth: false,
        is_alpha_sensor: false,
        is_beta_sensor: false,
        is_energy_sensor: false,
        is_inhibitor: false,
        is_propeller: false,
        is_condenser: false,
        is_displacer: false,
    }, // D
    AminoVisualFlags {
        is_mouth: false,
        is_alpha_sensor: false,
        is_beta_sensor: false,
        is_energy_sensor: false,
        is_inhibitor: false,
        is_propeller: false,
        is_condenser: false,
        is_displacer: false,
    }, // E
    AminoVisualFlags {
        is_mouth: false,
        is_alpha_sensor: false,
        is_beta_sensor: false,
        is_energy_sensor: false,
        is_inhibitor: false,
        is_propeller: false,
        is_condenser: false,
        is_displacer: false,
    }, // F
    AminoVisualFlags {
        is_mouth: false,
        is_alpha_sensor: false,
        is_beta_sensor: false,
        is_energy_sensor: false,
        is_inhibitor: false,
        is_propeller: false,
        is_condenser: false,
        is_displacer: false,
    }, // G (structural - smallest amino acid)
    AminoVisualFlags {
        is_mouth: false,
        is_alpha_sensor: false,
        is_beta_sensor: false,
        is_energy_sensor: false,
        is_inhibitor: false,
        is_propeller: false,
        is_condenser: false,
        is_displacer: false,
    }, // H
    AminoVisualFlags {
        is_mouth: false,
        is_alpha_sensor: false,
        is_beta_sensor: false,
        is_energy_sensor: false,
        is_inhibitor: false,
        is_propeller: false,
        is_condenser: false,
        is_displacer: false,
    }, // I
    AminoVisualFlags {
        is_mouth: true,
        is_alpha_sensor: false,
        is_beta_sensor: false,
        is_energy_sensor: false,
        is_inhibitor: false,
        is_propeller: false,
        is_condenser: false,
        is_displacer: false,
    }, // K (mouth)
    AminoVisualFlags {
        is_mouth: false,
        is_alpha_sensor: false,
        is_beta_sensor: false,
        is_energy_sensor: false,
        is_inhibitor: false,
        is_propeller: false,
        is_condenser: false,
        is_displacer: false,
    }, // L
    AminoVisualFlags {
        is_mouth: false,
        is_alpha_sensor: false,
        is_beta_sensor: false,
        is_energy_sensor: false,
        is_inhibitor: false,
        is_propeller: false,
        is_condenser: false,
        is_displacer: false,
    }, // M
    AminoVisualFlags {
        is_mouth: false,
        is_alpha_sensor: false,
        is_beta_sensor: false,
        is_energy_sensor: false,
        is_inhibitor: true,
        is_propeller: false,
        is_condenser: false,
        is_displacer: false,
    }, // N - INHIBITOR (replaces Asparagine)
    AminoVisualFlags {
        is_mouth: false,
        is_alpha_sensor: false,
        is_beta_sensor: false,
        is_energy_sensor: false,
        is_inhibitor: false,
        is_propeller: true,
        is_condenser: false,
        is_displacer: false,
    }, // P (propeller)
    AminoVisualFlags {
        is_mouth: false,
        is_alpha_sensor: false,
        is_beta_sensor: false,
        is_energy_sensor: false,
        is_inhibitor: false,
        is_propeller: false,
        is_condenser: false,
        is_displacer: false,
    }, // Q
    AminoVisualFlags {
        is_mouth: false,
        is_alpha_sensor: false,
        is_beta_sensor: false,
        is_energy_sensor: false,
        is_inhibitor: false,
        is_propeller: false,
        is_condenser: false,
        is_displacer: false,
    }, // R
    AminoVisualFlags {
        is_mouth: false,
        is_alpha_sensor: true,
        is_beta_sensor: false,
        is_energy_sensor: false,
        is_inhibitor: false,
        is_propeller: false,
        is_condenser: false,
        is_displacer: false,
    }, // S (alpha sensor)
    AminoVisualFlags {
        is_mouth: false,
        is_alpha_sensor: false,
        is_beta_sensor: false,
        is_energy_sensor: true,
        is_inhibitor: false,
        is_propeller: false,
        is_condenser: false,
        is_displacer: false,
    }, // T (energy sensor)
    AminoVisualFlags {
        is_mouth: false,
        is_alpha_sensor: false,
        is_beta_sensor: false,
        is_energy_sensor: false,
        is_inhibitor: false,
        is_propeller: false,
        is_condenser: false,
        is_displacer: true,
    }, // V (displacer)
    AminoVisualFlags {
        is_mouth: false,
        is_alpha_sensor: false,
        is_beta_sensor: false,
        is_energy_sensor: false,
        is_inhibitor: false,
        is_propeller: false,
        is_condenser: false,
        is_displacer: false,
    }, // W (storage)
    AminoVisualFlags {
        is_mouth: false,
        is_alpha_sensor: false,
        is_beta_sensor: false,
        is_energy_sensor: false,
        is_inhibitor: false,
        is_propeller: false,
        is_condenser: true,
        is_displacer: false,
    }, // Y (alpha condenser)
];

const DEFAULT_PART_COLOR: [f32; 3] = [0.5, 0.5, 0.5];

#[inline]
fn rgb_component(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[inline]
fn rgb_to_color32(rgb: [f32; 3]) -> egui::Color32 {
    egui::Color32::from_rgb(
        rgb_component(rgb[0]),
        rgb_component(rgb[1]),
        rgb_component(rgb[2]),
    )
}

#[inline]
fn rgb_to_color32_with_alpha(rgb: [f32; 3], alpha: f32) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(
        rgb_component(rgb[0]),
        rgb_component(rgb[1]),
        rgb_component(rgb[2]),
        rgb_component(alpha),
    )
}

struct RainThumbnail {
    image: ColorImage,
    texture: Option<TextureHandle>,
    dirty: bool,
}

impl RainThumbnail {
    fn new(image: ColorImage) -> Self {
        Self {
            image,
            texture: None,
            dirty: true,
        }
    }

    fn ensure_texture(&mut self, ctx: &egui::Context, label: &str) -> TextureId {
        if self.dirty || self.texture.is_none() {
            let handle = ctx.load_texture(
                label.to_string(),
                self.image.clone(),
                TextureOptions::LINEAR,
            );
            self.texture = Some(handle);
            self.dirty = false;
        }
        self.texture.as_ref().unwrap().id()
    }
}

// Genetic code table - converts RNA codon to amino acid index (matches shader)
fn codon_to_amino_acid(b0: u8, b1: u8, b2: u8) -> usize {
    // U-starting codons
    if b0 == 85 {
        if b1 == 85 {
            // UUU, UUC -> Phe(4); UUA, UUG -> Leu(9)
            if b2 == 85 || b2 == 67 {
                return 4;
            } // Phe
            return 9; // Leu
        }
        if b1 == 67 {
            return 15;
        } // UC* -> Ser(15)
        if b1 == 65 {
            // UAU, UAC -> Tyr(19); UAA, UAG are stop codons
            if b2 == 85 || b2 == 67 {
                return 19;
            } // Tyr
            return 19; // Fallback Tyr for stop codons
        }
        if b1 == 71 {
            // UGU, UGC -> Cys(1); UGA is stop; UGG -> Trp(18)
            if b2 == 85 || b2 == 67 {
                return 1;
            } // Cys
            if b2 == 71 {
                return 18;
            } // Trp
            return 1; // Fallback Cys for UGA stop
        }
    }
    // C-starting codons
    if b0 == 67 {
        if b1 == 85 {
            return 9;
        } // CU* -> Leu(9)
        if b1 == 67 {
            return 12;
        } // CC* -> Pro(12)
        if b1 == 65 {
            // CAU, CAC -> His(6); CAA, CAG -> Gln(13)
            if b2 == 85 || b2 == 67 {
                return 6;
            } // His
            return 13; // Gln
        }
        if b1 == 71 {
            return 14;
        } // CG* -> Arg(14)
    }
    // A-starting codons
    if b0 == 65 {
        if b1 == 85 {
            // AUU, AUC, AUA -> Ile(7); AUG -> Met(10)
            if b2 == 71 {
                return 10;
            } // Met
            return 7; // Ile
        }
        if b1 == 67 {
            return 16;
        } // AC* -> Thr(16)
        if b1 == 65 {
            // AAU, AAC -> Asn(11); AAA, AAG -> Lys(8)
            if b2 == 85 || b2 == 67 {
                return 11;
            } // Asn
            return 8; // Lys
        }
        if b1 == 71 {
            // AGU, AGC -> Ser(15); AGA, AGG -> Arg(14)
            if b2 == 85 || b2 == 67 {
                return 15;
            } // Ser
            return 14; // Arg
        }
    }
    // G-starting codons
    if b0 == 71 {
        if b1 == 85 {
            return 17;
        } // GU* -> Val(17)
        if b1 == 67 {
            return 0;
        } // GC* -> Ala(0)
        if b1 == 65 {
            // GAU, GAC -> Asp(2); GAA, GAG -> Glu(3)
            if b2 == 85 || b2 == 67 {
                return 2;
            } // Asp
            return 3; // Glu
        }
        if b1 == 71 {
            return 5;
        } // GG* -> Gly(5)
    }
    // Fallback (should not happen with valid RNA)
    0 // Ala
}

fn paint_cloud(painter: &egui::Painter, center: egui::Pos2, radius: f32, color: egui::Color32, seed: u64) {
    // Draw multiple overlapping circles to create a fluffy cloud appearance - matching GPU shader
    // GPU draws 8 puffs + 1 central circle
    let num_puffs = 8;

    // Helper to generate a pseudo-random float from seed
    let hash_f32 = |s: u64| -> f32 {
        let h = s.wrapping_mul(2654435761u64);
        ((h ^ (h >> 32)) & 0xFFFFFFFF) as f32 / 4294967295.0
    };

    // Draw outer puffs
    for i in 0..num_puffs {
        let angle = (i as f32) * std::f32::consts::TAU / (num_puffs as f32);
        let hash_val = hash_f32(seed.wrapping_mul((i as u64 + 1).wrapping_mul(2654435761)));
        let offset_dist = radius * 0.4 * hash_val;
        let puff_center = egui::pos2(
            center.x + angle.cos() * offset_dist,
            center.y + angle.sin() * offset_dist,
        );
        let puff_radius = radius * (0.5 + 0.3 * hash_val);
        painter.circle_filled(puff_center, puff_radius, color);
    }
    // Draw larger central circle
    painter.circle_filled(center, radius * 0.7, color);
}

fn paint_asterisk(painter: &egui::Painter, center: egui::Pos2, radius: f32, color: egui::Color32) {
    // Draw 4 lines: vertical, horizontal, and two diagonals - matching GPU shader exactly
    // GPU uses fixed thickness of 1.0 world units, we'll use 1.0 screen pixels for consistency
    let line_thickness = 1.0;
    let d = radius * 0.70710678; // radius / sqrt(2)

    // Vertical line
    painter.line_segment(
        [
            egui::pos2(center.x, center.y - radius),
            egui::pos2(center.x, center.y + radius),
        ],
        egui::Stroke::new(line_thickness, color),
    );
    // Horizontal line
    painter.line_segment(
        [
            egui::pos2(center.x - radius, center.y),
            egui::pos2(center.x + radius, center.y),
        ],
        egui::Stroke::new(line_thickness, color),
    );
    // Diagonal 1 (top-left to bottom-right)
    painter.line_segment(
        [
            egui::pos2(center.x - d, center.y - d),
            egui::pos2(center.x + d, center.y + d),
        ],
        egui::Stroke::new(line_thickness, color),
    );
    // Diagonal 2 (top-right to bottom-left)
    painter.line_segment(
        [
            egui::pos2(center.x + d, center.y - d),
            egui::pos2(center.x - d, center.y + d),
        ],
        egui::Stroke::new(line_thickness, color),
    );
}

// ============================================================================
// GPU-FRIENDLY AGENT STRUCTURES (matching shader layout)
// ============================================================================

// Simple persisted representation for saving/loading a single agent's genome
// We persist the full GENOME_BYTES buffer including padding 'X' characters.
#[derive(Serialize, Deserialize)]
struct SavedAgent {
    genome_string: String, // Up to GENOME_BYTES characters of A/U/G/C/X (short strings are centered)
}

fn genome_to_string(genome: &[u32; GENOME_WORDS]) -> String {
    let mut result = String::with_capacity(GENOME_BYTES);
    for &word in genome {
        for i in 0..4 {
            let byte = ((word >> (i * 8)) & 0xFF) as u8;
            let ch = match byte {
                65 => 'A', // ASCII 'A'
                85 => 'U', // ASCII 'U'
                71 => 'G', // ASCII 'G'
                67 => 'C', // ASCII 'C'
                88 => 'X', // ASCII 'X' padding
                _ => 'X',  // Unknown -> treat as padding for safety
            };
            result.push(ch);
        }
    }
    result
}

fn string_to_genome(s: &str) -> anyhow::Result<[u32; GENOME_WORDS]> {
    let len = s.len();
    if len == 0 || len > GENOME_BYTES {
        anyhow::bail!(
            "Genome string must be between 1 and {} characters (including X padding), got {}",
            GENOME_BYTES,
            len
        );
    }

    let mut padded = vec!['X'; GENOME_BYTES];
    let left_pad = (GENOME_BYTES - len) / 2;
    for (i, ch) in s.chars().enumerate() {
        padded[left_pad + i] = match ch {
            'A' | 'U' | 'G' | 'C' | 'X' => ch,
            _ => 'X',
        };
    }

    let mut genome = [0u32; GENOME_WORDS];
    for word_idx in 0..GENOME_WORDS {
        let mut word = 0u32;
        for byte_idx in 0..4 {
            let char_idx = word_idx * 4 + byte_idx;
            let byte = match padded[char_idx] {
                'A' => 65u8,
                'U' => 85u8,
                'G' => 71u8,
                'C' => 67u8,
                'X' => 88u8,
                _ => 88u8,
            };
            word |= (byte as u32) << (byte_idx * 8);
        }
        genome[word_idx] = word;
    }

    Ok(genome)
}

fn save_agent_via_dialog(agent: &Agent) -> anyhow::Result<()> {
    let genome_str = genome_to_string(&agent.genome);
    let default_name = format!("agent_{}.json", &genome_str[0..8.min(genome_str.len())]);

    if let Some(path) = rfd::FileDialog::new()
        .add_filter("Agent", &["json"])
        .set_file_name(&default_name)
        .save_file()
    {
        let saved = SavedAgent {
            genome_string: genome_str,
        };
        let json = serde_json::to_string_pretty(&saved)?;
        fs::write(&path, json)?;
        println!("Saved agent genome to {}", path.display());
        Ok(())
    } else {
        anyhow::bail!("Save canceled")
    }
}

fn load_agent_via_dialog() -> anyhow::Result<[u32; GENOME_WORDS]> {
    if let Some(path) = rfd::FileDialog::new()
        .add_filter("Agent", &["json"])
        .pick_file()
    {
        let data = fs::read_to_string(&path)?;
        let saved: SavedAgent = serde_json::from_str(&data)?;
        let genome = string_to_genome(&saved.genome_string)?;
        println!(
            "Loaded agent genome from {} ({} bases)",
            path.display(),
            saved.genome_string.len()
        );
        Ok(genome)
    } else {
        anyhow::bail!("Load canceled")
    }
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BodyPart {
    pos: [f32; 2],            // 8 bytes
    size: f32,                // 4 bytes
    part_type: u32,           // 4 bytes
    alpha_signal: f32,        // 4 bytes
    beta_signal: f32,         // 4 bytes
    pad: [f32; 2],            // 8 bytes padding to reach 32 bytes (multiple of 16)
                              // pad[0] stores packed u16 prev_world_pos (bitcast)
                              // pad[1] stores signal/charge values
}

impl BodyPart {
    /// Extract base type from encoded part_type (0-19 for amino acids, 20+ for organs)
    fn base_type(&self) -> u32 {
        self.part_type & 0xFF
    }

    /// Extract organ parameter from encoded part_type (0-255)
    fn organ_param(&self) -> u32 {
        (self.part_type >> 8) & 0xFF
    }
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Zeroable)]
struct Agent {
    position: [f32; 2],                        // 8 bytes (0-7)
    velocity: [f32; 2],                        // 8 bytes (8-15)
    rotation: f32,                             // 4 bytes (16-19)
    energy: f32,                               // 4 bytes (20-23)
    energy_capacity: f32,                      // 4 bytes (24-27)
    torque_debug: f32,                         // 4 bytes (28-31) - accumulated torque (matches shader)
    morphology_origin: [f32; 2],               // 8 bytes (32-39) - chain origin after CoM centering
    alive: u32,                                // 4 bytes (40-43)
    body_count: u32,                           // 4 bytes (44-47)
    pairing_counter: u32,                      // 4 bytes (48-51) - number of bases paired
    is_selected: u32,                          // 4 bytes (52-55) - selected for debug view
    generation: u32,                           // 4 bytes (56-59) - lineage generation counter
    age: u32,                                  // 4 bytes (60-63) - age in frames
    total_mass: f32,                           // 4 bytes (64-67) - computed each frame after morphology
    poison_resistant_count: u32,               // 4 bytes (68-71) - number of poison-resistant organs
    gene_length: u32,                          // 4 bytes (72-75) - number of non-X bases in genome (valid gene length)
    genome: [u32; GENOME_WORDS],               // GENOME_BYTES bytes (ASCII bases) - 76 to 331
    _pad_genome_align: [u32; 5],               // 20 bytes (332-351) - padding to align body to 16-byte boundary
    body: [BodyPart; MAX_BODY_PARTS],          // starts at byte 352 (16-byte aligned)
}

// SAFETY: Agent is repr(C) with explicit padding, matching shader layout exactly
unsafe impl bytemuck::Pod for Agent {}

#[repr(C, align(16))]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SpawnRequest {
    seed: u32,
    genome_seed: u32,
    flags: u32,
    _pad_seed: u32,
    position: [f32; 2],
    energy: f32,
    rotation: f32,
    genome_override: [u32; GENOME_WORDS],
}
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
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
    prev_camera_pan_x: f32, // Previous frame camera position for motion blur
    prev_camera_pan_y: f32, // Previous frame camera position for motion blur
    follow_mode: u32,       // 1 if following an agent, 0 otherwise
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
    debug_mode: u32,           // 0 = off, 1 = per-segment debug overlay
    visual_stride: u32,        // pixels per row in visual_grid buffer (padded)
    selected_agent_index: u32, // Index of selected agent for debug visualization (u32::MAX if none)
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
    // When true (1), override per-amino left/right multipliers with isotropic interior diffusion
    interior_isotropic: u32,
    // When true (1), ignore stop codons and translate entire genome to max body parts
    ignore_stop_codons: u32,
    // When true (1), require AUG start codon before translation begins
    require_start_codon: u32,
    // When true (1), offspring are direct mutated copies (asexual); when false (0), offspring are reverse-complemented (sexual)
    asexual_reproduction: u32,
    // Visualization parameters
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
    grid_interpolation: u32,  // 0=nearest, 1=bilinear, 2=bicubic
    alpha_gamma_adjust: f32,  // Gamma correction for alpha channel
    beta_gamma_adjust: f32,   // Gamma correction for beta channel
    gamma_gamma_adjust: f32,  // Gamma correction for gamma channel
    light_dir_x: f32,         // Light direction for slope-based lighting
    light_dir_y: f32,
    light_dir_z: f32,
    light_power: f32,         // Light intensity multiplier for 3D shading
    agent_blend_mode: u32,    // Agent visualization blend mode
    agent_color_r: f32,
    agent_color_g: f32,
    agent_color_b: f32,
    agent_color_blend: f32,   // Blend factor: 0.0=amino only, 1.0=agent only
    epoch: u32,               // Current simulation epoch for time-based effects
    vector_force_power: f32,  // Global force multiplier (0.0 = off)
    vector_force_x: f32,      // Force direction X (-1.0 to 1.0)
    vector_force_y: f32,      // Force direction Y (-1.0 to 1.0)
    inspector_zoom: f32,      // Inspector preview zoom level (1.0 = default)
    agent_trail_decay: f32,   // Agent trail decay rate (0.0 = persistent, 1.0 = instant clear)
    fluid_show: u32,          // Show fluid simulation overlay
    _padding2: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct EnvironmentInitParams {
    grid_resolution: u32,
    seed: u32,
    noise_octaves: u32,
    slope_octaves: u32,
    noise_scale: f32,
    noise_contrast: f32,
    slope_scale: f32,
    slope_contrast: f32,
    alpha_range: [f32; 2],
    beta_range: [f32; 2],
    gamma_height_range: [f32; 2],
    _trail_alignment: [f32; 2],
    trail_values: [f32; 4],
    slope_pair: [f32; 2],
    _slope_alignment: [f32; 2],
    gen_params: [u32; 4], // [mode, type, value_bits, seed]
    alpha_noise_scale: f32,
    beta_noise_scale: f32,
    gamma_noise_scale: f32,
    noise_power: f32,
}

// Keep host layout in sync with the WGSL uniform buffer (std140, 128 bytes total).
const _: [(); 128] = [(); std::mem::size_of::<EnvironmentInitParams>()];

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct AutoDifficultyParam {
    enabled: bool,
    min_threshold: f32, // e.g. min population
    max_threshold: f32, // e.g. max population
    adjustment_percent: f32, // e.g. 10%
    cooldown_epochs: u64,
    last_adjustment_epoch: u64,
    difficulty_level: i32, // 0 is base, +1 is harder, -1 is easier
}

impl AutoDifficultyParam {
    // Calculate the effective value with difficulty multiplier applied
    fn apply_to(&self, base_value: f32, harder_increases: bool) -> f32 {
        if !self.enabled || self.difficulty_level == 0 {
            return base_value;
        }

        let factor = self.adjustment_percent / 100.0;
        let multiplier = if harder_increases {
            // Positive difficulty makes value higher
            1.0 + (self.difficulty_level as f32 * factor)
        } else {
            // Positive difficulty makes value lower
            1.0 - (self.difficulty_level as f32 * factor)
        };

        base_value * multiplier.max(0.01) // Prevent going to zero or negative
    }
}

impl Default for AutoDifficultyParam {
    fn default() -> Self {
        Self {
            enabled: false,
            min_threshold: 1000.0,
            max_threshold: 5000.0,
            adjustment_percent: 10.0,
            cooldown_epochs: 600, // 10 seconds at 60fps
            last_adjustment_epoch: 0,
            difficulty_level: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
struct DifficultySettings {
    food_power: AutoDifficultyParam,
    poison_power: AutoDifficultyParam,
    spawn_prob: AutoDifficultyParam,
    death_prob: AutoDifficultyParam,
    alpha_rain: AutoDifficultyParam,
    beta_rain: AutoDifficultyParam,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
struct SimulationSettings {
    camera_zoom: f32,
    spawn_probability: f32,
    death_probability: f32,
    mutation_rate: f32,
    auto_replenish: bool,
    diffusion_interval: u32,
    slope_interval: u32,
    alpha_blur: f32,
    beta_blur: f32,
    gamma_blur: f32,
    alpha_slope_bias: f32,
    beta_slope_bias: f32,
    alpha_multiplier: f32,
    beta_multiplier: f32,
    alpha_rain_map_path: Option<PathBuf>,
    beta_rain_map_path: Option<PathBuf>,
    chemical_slope_scale_alpha: f32,
    chemical_slope_scale_beta: f32,
    food_power: f32,
    poison_power: f32,
    amino_maintenance_cost: f32,
    pairing_cost: f32,
    prop_wash_strength: f32,
    repulsion_strength: f32,
    agent_repulsion_strength: f32,
    limit_fps: bool,
    limit_fps_25: bool,
    render_interval: u32,
    gamma_debug_visual: bool,
    slope_debug_visual: bool,
    rain_debug_visual: bool,  // Visualization mode for rain patterns
    fluid_show: bool,         // Show fluid simulation overlay
    vector_force_power: f32,  // Global force multiplier (0.0 = off)
    vector_force_x: f32,      // Force direction X (-1.0 to 1.0)
    vector_force_y: f32,      // Force direction Y (-1.0 to 1.0)
    gamma_hidden: bool,
    debug_per_segment: bool,
    gamma_vis_min: f32,
    gamma_vis_max: f32,
    alpha_show: bool,
    beta_show: bool,
    gamma_show: bool,
    slope_lighting: bool,
    slope_lighting_strength: f32,
    trail_diffusion: f32,
    trail_decay: f32,
    trail_opacity: f32,
    trail_show: bool,
    interior_isotropic: bool,
    ignore_stop_codons: bool,
    require_start_codon: bool,
    asexual_reproduction: bool,
    alpha_rain_variation: f32,
    beta_rain_variation: f32,
    alpha_rain_phase: f32,
    beta_rain_phase: f32,
    alpha_rain_freq: f32,
    beta_rain_freq: f32,
    difficulty: DifficultySettings,
    background_color: [f32; 3],
    alpha_blend_mode: u32,
    beta_blend_mode: u32,
    gamma_blend_mode: u32,
    slope_blend_mode: u32,
    alpha_color: [f32; 3],
    beta_color: [f32; 3],
    gamma_color: [f32; 3],
    grid_interpolation: u32,
    alpha_gamma_adjust: f32,
    beta_gamma_adjust: f32,
    gamma_gamma_adjust: f32,
    light_direction: [f32; 3],  // Light direction for slope-based lighting effects
    light_power: f32,            // Light intensity for 3D shading (0.0-2.0)
    agent_blend_mode: u32,  // Agent blend mode: 0=comp, 1=add, 2=subtract, 3=multiply
    agent_color: [f32; 3],  // Agent color tint
    agent_color_blend: f32,  // Blend factor between amino acid color (0.0) and agent color (1.0)
    alpha_noise_scale: f32,
    beta_noise_scale: f32,
    gamma_noise_scale: f32,
    noise_power: f32,
    agent_trail_decay: f32,  // Agent trail decay rate (0.0 = persistent, 1.0 = instant clear)
}

impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            camera_zoom: 1.0,
            spawn_probability: 0.01,
            death_probability: 0.001,
            mutation_rate: 0.005,
            auto_replenish: true,
            diffusion_interval: 1,
            slope_interval: 1,
            alpha_blur: 0.002,
            beta_blur: 0.0005,
            gamma_blur: 0.0005,
            alpha_slope_bias: -5.0,
            beta_slope_bias: 5.0,
            alpha_multiplier: 0.0001,
            beta_multiplier: 0.0,
            alpha_rain_map_path: None,
            beta_rain_map_path: None,
            chemical_slope_scale_alpha: 0.1,
            chemical_slope_scale_beta: 0.1,
            food_power: 3.0,
            poison_power: 1.0,
            amino_maintenance_cost: 0.001,
            pairing_cost: 0.1,
            prop_wash_strength: 1.0,
            repulsion_strength: 10.0,
            agent_repulsion_strength: 1.0,
            limit_fps: true,
            limit_fps_25: false,
            render_interval: 100, // Draw every 100 steps in fast mode
            gamma_debug_visual: false,
            slope_debug_visual: false,
            rain_debug_visual: false,
            fluid_show: false,  // Fluid simulation overlay disabled by default
            vector_force_power: 0.0,  // Disabled by default
            vector_force_x: 0.0,
            vector_force_y: -1.0,     // Downward gravity when enabled
            gamma_hidden: false,
            debug_per_segment: false,
            gamma_vis_min: 0.0,
            gamma_vis_max: 1.0,
            alpha_show: true,
            beta_show: true,
            gamma_show: true,
            slope_lighting: false,
            slope_lighting_strength: 1.0,
            trail_diffusion: 0.15,
            trail_decay: 0.995,
            trail_opacity: 0.5,
            trail_show: false,
            interior_isotropic: false, // Use asymmetric left/right multipliers from amino acids
            ignore_stop_codons: false, // Stop codons (UAA, UAG, UGA) terminate translation
            require_start_codon: true, // Translation starts at AUG (Methionine)
            asexual_reproduction: false, // Offspring are reverse-complemented (sexual reproduction)
            alpha_rain_variation: 0.0,
            beta_rain_variation: 0.0,
            alpha_rain_phase: 0.0,
            beta_rain_phase: 0.0,
            alpha_rain_freq: 1.0,
            beta_rain_freq: 1.0,
            difficulty: DifficultySettings::default(),
            background_color: [0.0, 0.0, 0.0],
            alpha_blend_mode: 0, // additive
            beta_blend_mode: 0,
            gamma_blend_mode: 0,
            slope_blend_mode: 0,  // No slope lighting by default
            alpha_color: [0.0, 1.0, 0.0], // green
            beta_color: [1.0, 0.0, 0.0],  // red
            gamma_color: [0.0, 0.0, 1.0], // blue
            grid_interpolation: 1, // bilinear by default
            alpha_gamma_adjust: 1.0,  // Linear (no adjustment)
            beta_gamma_adjust: 1.0,   // Linear (no adjustment)
            gamma_gamma_adjust: 1.0,  // Linear (no adjustment)
            light_direction: [0.5, 0.5, 0.5],  // Default diagonal light
            light_power: 1.0,                   // Default lighting strength
            agent_blend_mode: 0,  // Comp by default
            agent_color: [1.0, 1.0, 1.0],  // White
            agent_color_blend: 0.0,  // Default: show only amino acid colors (no agent color blend)
            alpha_noise_scale: 1.0,
            beta_noise_scale: 1.0,
            gamma_noise_scale: 1.0,
            noise_power: 1.0,
            agent_trail_decay: 1.0,  // Default to instant clear (original behavior)
        }
    }
}

// ============================================================================
// SNAPSHOT SAVE/LOAD STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AgentSnapshot {
    position: [f32; 2],
    rotation: f32,
    energy: f32,
    generation: u32,
    genome: Vec<u8>, // Store as bytes for compression
}

impl From<&Agent> for AgentSnapshot {
    fn from(agent: &Agent) -> Self {
        // Extract genome as bytes
        let mut genome_bytes = Vec::with_capacity(GENOME_BYTES);
        for word in &agent.genome {
            genome_bytes.extend_from_slice(&word.to_le_bytes());
        }

        Self {
            position: agent.position,
            rotation: agent.rotation,
            energy: agent.energy,
            generation: agent.generation,
            genome: genome_bytes,
        }
    }
}

impl AgentSnapshot {
    fn to_spawn_request(&self) -> SpawnRequest {
        let mut genome_override = [0u32; GENOME_WORDS];
        for (i, chunk) in self.genome.chunks_exact(4).enumerate() {
            if i < GENOME_WORDS {
                genome_override[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }

        SpawnRequest {
            seed: 0,
            genome_seed: 0,
            flags: 1, // Use genome override
            _pad_seed: 0,
            position: self.position,
            energy: self.energy,
            rotation: self.rotation,
            genome_override,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SimulationSnapshot {
    version: String,
    timestamp: String,
    epoch: u64,
    #[serde(default)]
    settings: Option<SimulationSettings>,
    agents: Vec<AgentSnapshot>,
}

impl SimulationSnapshot {
    fn new(epoch: u64, agents: &[Agent], settings: SimulationSettings) -> Self {
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();

        // agents should already be filtered for alive != 0 by caller
        // Randomly sample up to 5000 agents if there are more
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();

        let sampled_agents: Vec<&Agent> = if agents.len() <= 5000 {
            // Keep all agents
            agents.iter().collect()
        } else {
            // Sample 5000 random agents
            agents.iter().collect::<Vec<_>>().choose_multiple(&mut rng, 5000).copied().collect()
        };

        let agent_snapshots: Vec<AgentSnapshot> = sampled_agents
            .iter()
            .map(|a| AgentSnapshot::from(*a))
            .collect();

        println!("Snapshot created with {} agents (from {} living)", agent_snapshots.len(), agents.len());

        Self {
            version: "1.0".to_string(),
            timestamp,
            epoch,
            settings: Some(settings),
            agents: agent_snapshots,
        }
    }
}

impl SimulationSettings {
    fn default_path() -> PathBuf {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(SETTINGS_FILE_NAME)
    }

    fn load_from_disk(path: &Path) -> anyhow::Result<Self> {
        let data = fs::read_to_string(path)?;
        let settings = serde_json::from_str(&data)?;
        Ok(settings)
    }

    fn save_to_disk(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    fn sanitize(&mut self) {
        self.camera_zoom = self.camera_zoom.clamp(0.1, 2000.0);
        self.spawn_probability = self.spawn_probability.clamp(0.0, 1.0);
        self.death_probability = self.death_probability.clamp(0.0, 0.1);
        self.mutation_rate = self.mutation_rate.clamp(0.0, 0.1);
        // auto_replenish requires no sanitizing
        self.diffusion_interval = self.diffusion_interval.clamp(1, 64);
        self.slope_interval = self.slope_interval.clamp(1, 64);
        self.alpha_blur = self.alpha_blur.clamp(0.0, 0.1);
        self.beta_blur = self.beta_blur.clamp(0.0, 0.1);
        self.gamma_blur = self.gamma_blur.clamp(0.0, 0.1);
        self.alpha_slope_bias = self.alpha_slope_bias.clamp(-10.0, 10.0);
        self.beta_slope_bias = self.beta_slope_bias.clamp(-10.0, 10.0);
        self.alpha_multiplier = self.alpha_multiplier.clamp(0.0, 0.001);
        self.beta_multiplier = self.beta_multiplier.clamp(0.0, 0.001);
        self.chemical_slope_scale_alpha = self.chemical_slope_scale_alpha.clamp(0.0, 1.0);
        self.chemical_slope_scale_beta = self.chemical_slope_scale_beta.clamp(0.0, 1.0);
        self.food_power = self.food_power.clamp(0.0, 10.0);
        self.poison_power = self.poison_power.clamp(0.0, 10.0);
        self.amino_maintenance_cost = self.amino_maintenance_cost.clamp(0.0, 0.01);
        self.pairing_cost = self.pairing_cost.clamp(0.0, 1.0);
        self.prop_wash_strength = self.prop_wash_strength.clamp(0.0, 5.0);
        self.repulsion_strength = self.repulsion_strength.clamp(0.0, 100.0);
        self.agent_repulsion_strength = self.agent_repulsion_strength.clamp(0.0, 10.0);
        self.render_interval = self.render_interval.clamp(1, 10_000);
        self.gamma_vis_min = self.gamma_vis_min.clamp(0.0, 1.0);
        self.gamma_vis_max = self.gamma_vis_max.clamp(0.0, 1.0);
        if self.gamma_vis_min >= self.gamma_vis_max {
            self.gamma_vis_max = (self.gamma_vis_min + 0.001).clamp(0.0, 1.0);
            self.gamma_vis_min = (self.gamma_vis_max - 0.001).clamp(0.0, 1.0);
        }
        self.alpha_rain_variation = self.alpha_rain_variation.clamp(0.0, 1.0);
        self.beta_rain_variation = self.beta_rain_variation.clamp(0.0, 1.0);
        self.alpha_rain_phase = self.alpha_rain_phase.clamp(0.0, std::f32::consts::PI * 2.0);
        self.beta_rain_phase = self.beta_rain_phase.clamp(0.0, std::f32::consts::PI * 2.0);
        self.alpha_rain_freq = self.alpha_rain_freq.clamp(0.0, 100.0);
        self.beta_rain_freq = self.beta_rain_freq.clamp(0.0, 100.0);
    }
}

// ============================================================================
// GPU STATE
// ============================================================================

struct GpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,

    // Buffers
    agents_buffer_a: wgpu::Buffer,
    agents_buffer_b: wgpu::Buffer,
    alpha_grid: wgpu::Buffer,
    beta_grid: wgpu::Buffer,
    rain_map_buffer: wgpu::Buffer,
    agent_spatial_grid_buffer: wgpu::Buffer,
    gamma_grid: wgpu::Buffer,
    trail_grid: wgpu::Buffer,
    visual_grid_buffer: wgpu::Buffer,
    agent_grid_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    environment_init_params_buffer: wgpu::Buffer,
    alive_counter: wgpu::Buffer,
    spawn_debug_counters: wgpu::Buffer, // [spawn_counter, debug_counter]
    alive_readbacks: [Arc<wgpu::Buffer>; 2],
    alive_readback_pending: [Arc<Mutex<Option<Result<u32, ()>>>>; 2],
    alive_readback_inflight: [bool; 2],
    alive_readback_slot: usize,
    debug_readback: wgpu::Buffer,
    agents_readback: wgpu::Buffer, // Readback for agent inspection
    selected_agent_buffer: wgpu::Buffer, // GPU buffer for selected agent
    selected_agent_readback: wgpu::Buffer, // CPU readback for selected agent
    new_agents_buffer: wgpu::Buffer, // Buffer for spawned agents
    spawn_readback: wgpu::Buffer,  // Readback for spawn count
    spawn_requests_buffer: wgpu::Buffer, // CPU spawn requests seeds

    // Grid readback buffers for snapshot save
    alpha_grid_readback: wgpu::Buffer,
    beta_grid_readback: wgpu::Buffer,
    gamma_grid_readback: wgpu::Buffer,

    // Fluid simulation buffers (128x128 grid)
    fluid_velocity_a: wgpu::Buffer,
    fluid_velocity_b: wgpu::Buffer,
    fluid_pressure_a: wgpu::Buffer,
    fluid_pressure_b: wgpu::Buffer,
    fluid_divergence: wgpu::Buffer,
    fluid_forces: wgpu::Buffer,
    fluid_force_vectors: wgpu::Buffer,
    fluid_params_buffer: wgpu::Buffer,

    // Fluid simulation pipelines
    fluid_generate_forces_pipeline: wgpu::ComputePipeline,
    fluid_add_forces_pipeline: wgpu::ComputePipeline,
    fluid_advect_velocity_pipeline: wgpu::ComputePipeline,
    fluid_diffuse_velocity_pipeline: wgpu::ComputePipeline,
    fluid_enforce_boundaries_pipeline: wgpu::ComputePipeline,
    fluid_divergence_pipeline: wgpu::ComputePipeline,
    fluid_vorticity_confinement_pipeline: wgpu::ComputePipeline,
    fluid_clear_pressure_pipeline: wgpu::ComputePipeline,
    fluid_jacobi_pressure_pipeline: wgpu::ComputePipeline,
    fluid_subtract_gradient_pipeline: wgpu::ComputePipeline,
    fluid_copy_pipeline: wgpu::ComputePipeline,
    fluid_copy_velocity_to_forces_pipeline: wgpu::ComputePipeline,
    fluid_clear_forces_pipeline: wgpu::ComputePipeline,
    fluid_clear_force_vectors_pipeline: wgpu::ComputePipeline,
    fluid_clear_velocity_pipeline: wgpu::ComputePipeline,

    // Fluid bind groups
    fluid_bind_group_ab: wgpu::BindGroup,
    fluid_bind_group_ba: wgpu::BindGroup,

    // Fluid state
    fluid_time: f32,

    // Texture for visualization
    visual_texture: wgpu::Texture,
    visual_texture_view: wgpu::TextureView,
    sampler: wgpu::Sampler,

    // Pipelines
    process_pipeline: wgpu::ComputePipeline,
    diffuse_pipeline: wgpu::ComputePipeline,
    diffuse_trails_pipeline: wgpu::ComputePipeline,
    clear_visual_pipeline: wgpu::ComputePipeline,
    motion_blur_pipeline: wgpu::ComputePipeline,
    clear_agent_grid_pipeline: wgpu::ComputePipeline,
    render_agents_pipeline: wgpu::ComputePipeline, // Render all agents to agent_grid
    render_inspector_pipeline: wgpu::ComputePipeline,
    draw_inspector_agent_pipeline: wgpu::ComputePipeline,
    composite_agents_pipeline: wgpu::ComputePipeline,
    gamma_slope_pipeline: wgpu::ComputePipeline,
    merge_pipeline: wgpu::ComputePipeline, // Merge spawned agents
    compact_pipeline: wgpu::ComputePipeline, // Remove dead agents
    finalize_merge_pipeline: wgpu::ComputePipeline, // Reset spawn counter
    cpu_spawn_pipeline: wgpu::ComputePipeline, // Materialize CPU spawn requests on GPU
    initialize_dead_pipeline: wgpu::ComputePipeline, // Sanitize unused agent slots
    environment_init_pipeline: wgpu::ComputePipeline, // Fill alpha/beta/gamma/trails on GPU
    generate_map_pipeline: wgpu::ComputePipeline, // Generate specific map (flat/noise)
    clear_agent_spatial_grid_pipeline: wgpu::ComputePipeline, // Clear agent spatial grid
    populate_agent_spatial_grid_pipeline: wgpu::ComputePipeline, // Populate agent spatial grid
    drain_energy_pipeline: wgpu::ComputePipeline, // Vampire mouths drain energy from neighbors
    render_pipeline: wgpu::RenderPipeline,

    // Bind groups
    compute_bind_group_a: wgpu::BindGroup,
    compute_bind_group_b: wgpu::BindGroup,
    render_bind_group: wgpu::BindGroup,

    // State
    ping_pong: bool,
    agent_count: u32,
    alive_count: u32, // Number of living agents
    camera_zoom: f32,
    camera_pan: [f32; 2],
    prev_camera_pan: [f32; 2], // Previous frame camera position for motion blur

    // Agent management
    agents_cpu: Vec<Agent>,
    agent_buffer_capacity: usize,
    // CPU-triggered spawns queued for next frame merge
    cpu_spawn_queue: Vec<SpawnRequest>,
    spawn_request_count: u32,
    pending_spawn_upload: bool,
    spawn_probability: f32,
    death_probability: f32,
    auto_replenish: bool,
    // CPU-side copy of rain map data (interleaved alpha/beta)
    rain_map_data: Vec<f32>,
    difficulty: DifficultySettings,

    // FPS tracking
    frame_count: u32,
    last_fps_update: std::time::Instant,
    limit_fps: bool,
    limit_fps_25: bool,
    last_frame_time: std::time::Instant,

    // Simulation speed control
    render_interval: u32, // Draw every N steps in fast mode
    current_mode: u32,    // 0=VSync 60 FPS, 1=Full Speed, 2=Fast Draw, 3=Slow 25 FPS
    epoch: u64,           // Total simulation steps elapsed

    // Epoch tracking for speed display
    last_epoch_update: std::time::Instant,
    last_epoch_count: u64,

    // Inspector refresh throttling (update at ~25fps to save GPU time)
    inspector_frame_counter: u32,
    epochs_per_second: f32,

    // Population statistics tracking
    population_history: Vec<u32>, // Stores population at sample points
    alpha_rain_history: VecDeque<f32>,
    beta_rain_history: VecDeque<f32>,
    epoch_sample_interval: u64,   // Sample every N epochs (1000)
    last_sample_epoch: u64,       // Last epoch when we sampled
    max_history_points: usize,    // Maximum data points (5000)

    // Auto-snapshot tracking
    last_autosave_epoch: u64,     // Last epoch when auto-snapshot was saved

    // Mouse dragging state
    is_dragging: bool,
    last_mouse_pos: Option<[f32; 2]>,

    // Agent selection for debug panel
    selected_agent_index: Option<usize>,
    selected_agent_data: Option<Agent>,
    follow_selected_agent: bool, // New field
    camera_target: [f32; 2], // Target position for smooth camera following
    camera_velocity: [f32; 2], // Current camera velocity for smooth interpolation
    inspector_zoom: f32, // Zoom level for inspector preview (1.0 = default)
    agent_trail_decay: f32, // Agent trail decay rate (0.0 = persistent, 1.0 = instant clear)

    // RNG state for per-frame randomness
    rng_state: u64,

    // GUI state
    window: Arc<Window>,
    egui_renderer: egui_wgpu::Renderer,
    ui_tab: usize, // 0=Simulation, 1=Agents, 2=Environment
    ui_visible: bool, // Toggle control panel visibility with spacebar
    // Debug
    debug_per_segment: bool,
    is_paused: bool,
    // Snapshot save state
    snapshot_save_requested: bool,
    snapshot_load_requested: bool,
    // Visual buffer stride (pixels per row)
    visual_stride_pixels: u32,

    // Environment field controls
    alpha_blur: f32,
    beta_blur: f32,
    gamma_blur: f32,
    alpha_slope_bias: f32,
    beta_slope_bias: f32,
    alpha_multiplier: f32,
    beta_multiplier: f32,
    alpha_rain_map_path: Option<PathBuf>,
    beta_rain_map_path: Option<PathBuf>,
    alpha_rain_variation: f32,
    beta_rain_variation: f32,
    alpha_rain_phase: f32,
    beta_rain_phase: f32,
    alpha_rain_freq: f32,
    beta_rain_freq: f32,
    alpha_rain_thumbnail: Option<RainThumbnail>,
    beta_rain_thumbnail: Option<RainThumbnail>,
    chemical_slope_scale_alpha: f32,
    chemical_slope_scale_beta: f32,
    alpha_noise_scale: f32,
    beta_noise_scale: f32,
    gamma_noise_scale: f32,
    noise_power: f32,
    food_power: f32,
    poison_power: f32,
    amino_maintenance_cost: f32,
    pairing_cost: f32,
    diffusion_interval: u32,
    diffusion_counter: u32,
    slope_interval: u32,
    slope_counter: u32,

    // Evolution controls
    mutation_rate: f32,

    // Physics controls
    repulsion_strength: f32,
    agent_repulsion_strength: f32,
    gamma_debug_visual: bool,
    slope_debug_visual: bool,
    rain_debug_visual: bool,
    fluid_show: bool,  // Show fluid simulation overlay
    vector_force_power: f32,
    vector_force_x: f32,
    vector_force_y: f32,
    prop_wash_strength: f32,
    gamma_hidden: bool,
    gamma_vis_min: f32,
    gamma_vis_max: f32,
    alpha_show: bool,
    beta_show: bool,
    gamma_show: bool,
    slope_lighting: bool,
    slope_lighting_strength: f32,
    trail_diffusion: f32,
    trail_decay: f32,
    trail_opacity: f32,
    trail_show: bool,
    interior_isotropic: bool,
    ignore_stop_codons: bool,
    require_start_codon: bool,
    asexual_reproduction: bool,

    // Visualization controls
    background_color: [f32; 3],
    alpha_blend_mode: u32, // 0=additive, 1=multiply
    beta_blend_mode: u32,
    gamma_blend_mode: u32,
    slope_blend_mode: u32, // 0=none, 1=hard light, 2=soft light
    alpha_color: [f32; 3],
    beta_color: [f32; 3],
    gamma_color: [f32; 3],
    grid_interpolation: u32, // 0=nearest, 1=bilinear, 2=bicubic
    alpha_gamma_adjust: f32,
    beta_gamma_adjust: f32,
    gamma_gamma_adjust: f32,
    light_direction: [f32; 3],
    light_power: f32,
    agent_blend_mode: u32, // 0=comp, 1=add, 2=subtract, 3=multiply
    agent_color: [f32; 3],
    agent_color_blend: f32,

    settings_path: PathBuf,
    last_saved_settings: SimulationSettings,
    destroyed: bool,
}

impl GpuState {
    fn save_settings(&self, path: &Path) -> anyhow::Result<()> {
        let settings = SimulationSettings {
            camera_zoom: self.camera_zoom,
            spawn_probability: self.spawn_probability,
            death_probability: self.death_probability,
            mutation_rate: self.mutation_rate,
            auto_replenish: self.auto_replenish,
            diffusion_interval: self.diffusion_interval,
            slope_interval: self.slope_interval,
            alpha_blur: self.alpha_blur,
            beta_blur: self.beta_blur,
            gamma_blur: self.gamma_blur,
            alpha_slope_bias: self.alpha_slope_bias,
            beta_slope_bias: self.beta_slope_bias,
            alpha_multiplier: self.alpha_multiplier,
            beta_multiplier: self.beta_multiplier,
            alpha_rain_map_path: self.alpha_rain_map_path.clone(),
            beta_rain_map_path: self.beta_rain_map_path.clone(),
            chemical_slope_scale_alpha: self.chemical_slope_scale_alpha,
            chemical_slope_scale_beta: self.chemical_slope_scale_beta,
            alpha_noise_scale: self.alpha_noise_scale,
            beta_noise_scale: self.beta_noise_scale,
            gamma_noise_scale: self.gamma_noise_scale,
            noise_power: self.noise_power,
            food_power: self.food_power,
            poison_power: self.poison_power,
            amino_maintenance_cost: self.amino_maintenance_cost,
            pairing_cost: self.pairing_cost,
            prop_wash_strength: self.prop_wash_strength,
            repulsion_strength: self.repulsion_strength,
            agent_repulsion_strength: self.agent_repulsion_strength,
            limit_fps: self.limit_fps,
            limit_fps_25: self.limit_fps_25,
            render_interval: self.render_interval,
            gamma_debug_visual: self.gamma_debug_visual,
            slope_debug_visual: self.slope_debug_visual,
            rain_debug_visual: self.rain_debug_visual,
            fluid_show: self.fluid_show,
            vector_force_power: self.vector_force_power,
            vector_force_x: self.vector_force_x,
            vector_force_y: self.vector_force_y,
            gamma_hidden: self.gamma_hidden,
            debug_per_segment: self.debug_per_segment,
            gamma_vis_min: self.gamma_vis_min,
            gamma_vis_max: self.gamma_vis_max,
            alpha_show: self.alpha_show,
            beta_show: self.beta_show,
            gamma_show: self.gamma_show,
            slope_lighting: self.slope_lighting,
            slope_lighting_strength: self.slope_lighting_strength,
            trail_diffusion: self.trail_diffusion,
            trail_decay: self.trail_decay,
            trail_opacity: self.trail_opacity,
            trail_show: self.trail_show,
            interior_isotropic: self.interior_isotropic,
            ignore_stop_codons: self.ignore_stop_codons,
            require_start_codon: self.require_start_codon,
            asexual_reproduction: self.asexual_reproduction,
            alpha_rain_variation: self.alpha_rain_variation,
            beta_rain_variation: self.beta_rain_variation,
            alpha_rain_phase: self.alpha_rain_phase,
            beta_rain_phase: self.beta_rain_phase,
            alpha_rain_freq: self.alpha_rain_freq,
            beta_rain_freq: self.beta_rain_freq,
            difficulty: self.difficulty.clone(),
            background_color: self.background_color,
            alpha_blend_mode: self.alpha_blend_mode,
            beta_blend_mode: self.beta_blend_mode,
            gamma_blend_mode: self.gamma_blend_mode,
            slope_blend_mode: self.slope_blend_mode,
            alpha_color: self.alpha_color,
            beta_color: self.beta_color,
            gamma_color: self.gamma_color,
            grid_interpolation: self.grid_interpolation,
            alpha_gamma_adjust: self.alpha_gamma_adjust,
            beta_gamma_adjust: self.beta_gamma_adjust,
            gamma_gamma_adjust: self.gamma_gamma_adjust,
            light_direction: self.light_direction,
            light_power: self.light_power,
            agent_blend_mode: self.agent_blend_mode,
            agent_color: self.agent_color,
            agent_color_blend: self.agent_color_blend,
            agent_trail_decay: self.agent_trail_decay,
        };
        settings.save_to_disk(path)
    }

    fn load_settings(&mut self, path: &Path) -> anyhow::Result<()> {
        let settings = SimulationSettings::load_from_disk(path)?;
        self.camera_zoom = settings.camera_zoom;
        self.spawn_probability = settings.spawn_probability;
        self.death_probability = settings.death_probability;
        self.mutation_rate = settings.mutation_rate;
        self.auto_replenish = settings.auto_replenish;
        self.diffusion_interval = settings.diffusion_interval;
        self.slope_interval = settings.slope_interval;
        self.alpha_blur = settings.alpha_blur;
        self.beta_blur = settings.beta_blur;
        self.gamma_blur = settings.gamma_blur;
        self.alpha_slope_bias = settings.alpha_slope_bias;
        self.beta_slope_bias = settings.beta_slope_bias;
        self.alpha_multiplier = settings.alpha_multiplier;
        self.beta_multiplier = settings.beta_multiplier;
        self.alpha_rain_map_path = settings.alpha_rain_map_path;
        self.beta_rain_map_path = settings.beta_rain_map_path;
        self.chemical_slope_scale_alpha = settings.chemical_slope_scale_alpha;
        self.chemical_slope_scale_beta = settings.chemical_slope_scale_beta;
        self.alpha_noise_scale = settings.alpha_noise_scale;
        self.beta_noise_scale = settings.beta_noise_scale;
        self.gamma_noise_scale = settings.gamma_noise_scale;
        self.noise_power = settings.noise_power;
        self.food_power = settings.food_power;
        self.poison_power = settings.poison_power;
        self.amino_maintenance_cost = settings.amino_maintenance_cost;
        self.pairing_cost = settings.pairing_cost;
        self.prop_wash_strength = settings.prop_wash_strength;
        self.repulsion_strength = settings.repulsion_strength;
        self.agent_repulsion_strength = settings.agent_repulsion_strength;
        self.limit_fps = settings.limit_fps;
        self.limit_fps_25 = settings.limit_fps_25;
        self.render_interval = settings.render_interval;
        self.gamma_debug_visual = settings.gamma_debug_visual;
        self.slope_debug_visual = settings.slope_debug_visual;
        self.gamma_hidden = settings.gamma_hidden;
        self.debug_per_segment = settings.debug_per_segment;
        self.gamma_vis_min = settings.gamma_vis_min;
        self.gamma_vis_max = settings.gamma_vis_max;
        self.alpha_show = settings.alpha_show;
        self.beta_show = settings.beta_show;
        self.gamma_show = settings.gamma_show;
        self.slope_lighting = settings.slope_lighting;
        self.slope_lighting_strength = settings.slope_lighting_strength;
        self.trail_diffusion = settings.trail_diffusion;
        self.trail_decay = settings.trail_decay;
        self.trail_opacity = settings.trail_opacity;
        self.trail_show = settings.trail_show;
        self.interior_isotropic = settings.interior_isotropic;
        self.ignore_stop_codons = settings.ignore_stop_codons;
        self.require_start_codon = settings.require_start_codon;
        self.alpha_rain_variation = settings.alpha_rain_variation;
        self.beta_rain_variation = settings.beta_rain_variation;
        self.alpha_rain_phase = settings.alpha_rain_phase;
        self.beta_rain_phase = settings.beta_rain_phase;
        self.alpha_rain_freq = settings.alpha_rain_freq;
        self.beta_rain_freq = settings.beta_rain_freq;
        self.difficulty = settings.difficulty;
        self.background_color = settings.background_color;
        self.alpha_blend_mode = settings.alpha_blend_mode;
        self.beta_blend_mode = settings.beta_blend_mode;
        self.gamma_blend_mode = settings.gamma_blend_mode;
        self.slope_blend_mode = settings.slope_blend_mode;
        self.alpha_color = settings.alpha_color;
        self.beta_color = settings.beta_color;
        self.gamma_color = settings.gamma_color;
        self.grid_interpolation = settings.grid_interpolation;
        self.alpha_gamma_adjust = settings.alpha_gamma_adjust;
        self.beta_gamma_adjust = settings.beta_gamma_adjust;
        self.gamma_gamma_adjust = settings.gamma_gamma_adjust;
        self.light_direction = settings.light_direction;
        self.light_power = settings.light_power;
        self.agent_blend_mode = settings.agent_blend_mode;
        self.agent_color = settings.agent_color;
        self.agent_color_blend = settings.agent_color_blend;
        self.agent_trail_decay = settings.agent_trail_decay;

        if let Some(path) = &self.alpha_rain_map_path.clone() {
             let _ = self.load_alpha_rain_map(path);
        }
        if let Some(path) = &self.beta_rain_map_path.clone() {
             let _ = self.load_beta_rain_map(path);
        }

        Ok(())
    }

    fn generate_map(&mut self, mode: u32, gen_type: u32, value: f32, seed: u32) {
        let params = EnvironmentInitParams {
            grid_resolution: GRID_DIM_U32,
            seed,
            noise_octaves: 4,
            slope_octaves: 4,
            noise_scale: 5.0,
            noise_contrast: 1.0,
            slope_scale: 1.0,
            slope_contrast: 1.0,
            alpha_range: [0.0, 1.0],
            beta_range: [0.0, 1.0],
            gamma_height_range: [0.0, 1.0],
            _trail_alignment: [0.0; 2],
            trail_values: [0.0; 4],
            slope_pair: [0.0; 2],
            _slope_alignment: [0.0; 2],
            gen_params: [mode, gen_type, value.to_bits(), seed],
            alpha_noise_scale: self.alpha_noise_scale,
            beta_noise_scale: self.beta_noise_scale,
            gamma_noise_scale: self.gamma_noise_scale,
            noise_power: self.noise_power,
        };

        self.queue.write_buffer(
            &self.environment_init_params_buffer,
            0,
            bytemuck::bytes_of(&params),
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Generate Map Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Generate Map Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.generate_map_pipeline);
            if self.ping_pong {
                pass.set_bind_group(0, &self.compute_bind_group_b, &[]);
            } else {
                pass.set_bind_group(0, &self.compute_bind_group_a, &[]);
            }

            let workgroups = (GRID_DIM_U32 + 15) / 16;
            pass.dispatch_workgroups(workgroups, workgroups, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        // Create instance and adapter
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_storage_buffers_per_shader_stage: 16,
                        ..wgpu::Limits::default()
                    },
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .unwrap();

        Self::new_with_resources(window, instance, surface, adapter, device, queue).await
    }

    async fn new_with_resources(
        window: Arc<Window>,
        instance: wgpu::Instance,
        surface: wgpu::Surface<'static>,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Self {
        let size = window.inner_size();
        let mut profiler = StartupProfiler::new();
        profiler.mark("GpuState::new_with_resources begin");

        debug_assert_eq!(std::mem::size_of::<BodyPart>(), 32);
        debug_assert_eq!(std::mem::align_of::<BodyPart>(), 16);
        debug_assert_eq!(
            std::mem::size_of::<Agent>(),
            2400,
            "Agent layout mismatch for MAX_BODY_PARTS={}",
            MAX_BODY_PARTS
        );
        debug_assert_eq!(std::mem::align_of::<Agent>(), 16);
        // NOTE: SpawnRequest includes a GENOME_WORDS-word genome_override array (GENOME_BYTES bytes)
        // Layout breakdown (std430 / repr(C, align(16))):
        // seed/genome_seed/flags/_pad_seed = 16 bytes total
        // position ([f32;2]) = 8  -> offset 16..24
        // energy (4) + rotation (4) = 8 -> offset 24..32
        // genome_override ([u32; GENOME_WORDS]) = GENOME_BYTES -> offset 32..288 total
        // Total size = 288 bytes; alignment = 16 bytes.
        debug_assert_eq!(
            std::mem::size_of::<SpawnRequest>(),
            288,
            "SpawnRequest size mismatch; update buffer allocations/bindings if this fails"
        );
        debug_assert_eq!(std::mem::align_of::<SpawnRequest>(), 16);

        let window_clone = window.clone();
        profiler.mark("Resources received");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats[0];

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Immediate, // Start with max speed
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &surface_config);
        profiler.mark("Surface configured");

        // Create egui renderer before device gets moved
        let egui_renderer = egui_wgpu::Renderer::new(&device, surface_format, None, 1, false);
        profiler.mark("egui renderer");

        // Initialize agents with minimal data - GPU will generate genome and build body
        let max_agents = 50_000usize; // Increased with 2x world size (4x area): 3.4 KB/agent => ~170 MB per agent buffer
        let initial_agents = 0usize; // Start with 0, user spawns agents manually
        let agent_buffer_size = (max_agents * std::mem::size_of::<Agent>()) as u64;

        let mut agents = Vec::with_capacity(max_agents);
        agents.resize(initial_agents, Agent::zeroed());

        // Simple random seed for GPU
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Create GPU agent buffers without pre-filling from CPU memory; we'll clear them via GPU commands next
        let agents_buffer_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Agents Buffer A"),
            size: agent_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let agents_buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Agents Buffer B"),
            size: agent_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        profiler.mark("Agent buffers");

        let grid_size = GRID_CELL_COUNT;

        // Proper Perlin noise implementation (kept for potential later use)
        fn perlin_noise(x: f32, y: f32) -> f32 {
            // Permutation table for Perlin noise
            const PERM: [u8; 256] = [
                151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36,
                103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0,
                26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87,
                174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146,
                158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40,
                244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18,
                169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
                52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206,
                59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2,
                44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98,
                108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242,
                193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107,
                49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4,
                150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66,
                215, 61, 156, 180,
            ];

            let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
            let lerp = |a: f32, b: f32, t: f32| a + t * (b - a);

            let grad = |hash: u8, x: f32, y: f32| -> f32 {
                let h = hash & 3;
                match h {
                    0 => x + y,
                    1 => -x + y,
                    2 => x - y,
                    _ => -x - y,
                }
            };

            let xi = x.floor() as i32 & 255;
            let yi = y.floor() as i32 & 255;
            let xf = x - x.floor();
            let yf = y - y.floor();

            let u = fade(xf);
            let v = fade(yf);

            let aa = PERM[(PERM[xi as usize] as usize + yi as usize) & 255];
            let ab = PERM[(PERM[xi as usize] as usize + yi as usize + 1) & 255];
            let ba = PERM[(PERM[(xi as usize + 1) & 255] as usize + yi as usize) & 255];
            let bb = PERM[(PERM[(xi as usize + 1) & 255] as usize + yi as usize + 1) & 255];

            lerp(
                lerp(grad(aa, xf, yf), grad(ba, xf - 1.0, yf), u),
                lerp(grad(ab, xf, yf - 1.0), grad(bb, xf - 1.0, yf - 1.0), u),
                v,
            )
        }

        // Octaved Perlin noise for more natural patterns with contrast control
        fn fractal_noise(x: f32, y: f32, octaves: u32, scale: f32, contrast: f32) -> f32 {
            let mut value = 0.0;
            let mut amplitude = 1.0;
            let mut frequency = 1.0;
            let mut max_value = 0.0;

            for _ in 0..octaves {
                value += perlin_noise(x * frequency * scale, y * frequency * scale) * amplitude;
                max_value += amplitude;
                amplitude *= 0.5;
                frequency *= 2.0;
            }

            let normalized = (value / max_value + 1.0) * 0.5; // 0-1 range
                                                              // Apply contrast: shift to -0.5 to 0.5, multiply, shift back, clamp
            ((normalized - 0.5) * contrast + 0.5).clamp(0.0, 1.0)
        }

        // Skip expensive Perlin noise generation at startup for faster launch (GPU kernels will write defaults)
        // Alpha grid previously initialized to 0.5 everywhere; this value is now written via an initialization compute pass.

        let alpha_buffer_size = (grid_size * std::mem::size_of::<f32>()) as u64;
        let alpha_grid = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Alpha Grid"),
            size: alpha_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let beta_buffer_size = (grid_size * std::mem::size_of::<f32>()) as u64;
        let beta_grid = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Beta Grid"),
            size: beta_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Combined rain map buffer (vec2 per cell: x=alpha, y=beta)
        let rain_map_buffer_size = (grid_size * 2 * std::mem::size_of::<f32>()) as u64;
        let rain_map_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Rain Map"),
            size: rain_map_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Initialize with uniform rain (1.0 for both alpha and beta)
        let uniform_rain: Vec<f32> = (0..GRID_CELL_COUNT).flat_map(|_| [1.0f32, 1.0f32]).collect();
        queue.write_buffer(
            &rain_map_buffer,
            0,
            bytemuck::cast_slice(&uniform_rain),
        );
        profiler.mark("Rain map buffer");

        // Agent spatial grid for neighbor detection (u32 per cell at SPATIAL_GRID_DIM resolution)
        let agent_spatial_grid_size = (SPATIAL_GRID_CELL_COUNT * std::mem::size_of::<u32>()) as u64;
        let agent_spatial_grid_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Agent Spatial Grid"),
            size: agent_spatial_grid_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        profiler.mark("Agent spatial grid buffer");

        // Gamma grid packs height + 2 slope components per cell (3 floats)
        let gamma_buffer_size = (grid_size * 3 * std::mem::size_of::<f32>()) as u64;
        let gamma_grid = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gamma Grid"),
            size: gamma_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let trail_buffer_size = (grid_size * std::mem::size_of::<[f32; 4]>()) as u64;
        let trail_grid = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Trail Grid"),
            size: trail_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        profiler.mark("Alpha/Beta/Gamma/Trail buffers");

        let env_seed = (seed ^ (seed >> 32)) as u32;
        let environment_init = EnvironmentInitParams {
            grid_resolution: GRID_DIM_U32,
            seed: env_seed,
            noise_octaves: 5,
            slope_octaves: 3,
            noise_scale: 0.00035,
            noise_contrast: 1.35,
            slope_scale: 0.0015,
            slope_contrast: 1.0,
            alpha_range: [0.35, 0.85],
            beta_range: [0.0, 0.25],
            gamma_height_range: [-15.0, 20.0],
            _trail_alignment: [0.0; 2],
            trail_values: [0.0, 0.0, 0.0, 0.0],
            slope_pair: [0.0, 0.0],
            _slope_alignment: [0.0; 2],
            gen_params: [0, 0, 0, 0],
            alpha_noise_scale: 1.0,
            beta_noise_scale: 1.0,
            gamma_noise_scale: 1.0,
            noise_power: 1.0,
        };

        let environment_init_params_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Environment Init Params"),
                contents: bytemuck::bytes_of(&environment_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Create visual grid buffer with 256-byte aligned row stride
        let bytes_per_pixel: u32 = 16; // Rgba32Float
        let align: u32 = 256;
        let stride_bytes = ((surface_config.width * bytes_per_pixel + (align - 1)) / align) * align;
        let visual_stride_pixels = stride_bytes / bytes_per_pixel;
        let visual_grid_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Visual Grid"),
            size: (stride_bytes * surface_config.height) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        profiler.mark("Visual grid buffer");

        // Agent grid (same size as visual grid, leftmost 300px reserved for inspector)
        let agent_grid_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Agent Grid"),
            size: (stride_bytes * surface_config.height) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        profiler.mark("Agent grid buffer");

        let params = SimParams {
            dt: 0.016,
            frame_dt: 0.016,
            drag: 0.1,
            energy_cost: 0.0, // Disabled energy depletion for now
            amino_maintenance_cost: 0.001,
            spawn_probability: 0.01,
            death_probability: 0.001,
            grid_size: SIM_SIZE,
            camera_zoom: 1.0,
            camera_pan_x: SIM_SIZE / 2.0,
            camera_pan_y: SIM_SIZE / 2.0,
            prev_camera_pan_x: SIM_SIZE / 2.0, // Initialize to same as camera_pan
            prev_camera_pan_y: SIM_SIZE / 2.0, // Initialize to same as camera_pan
            follow_mode: 0,
            window_width: surface_config.width as f32,
            window_height: surface_config.height as f32,
            alpha_blur: 0.002,
            beta_blur: 0.0005,
            gamma_blur: 0.0005,
            alpha_slope_bias: -5.0,
            beta_slope_bias: 5.0,
            alpha_multiplier: 0.0001, // Rain probability: 0.01% per cell per frame
            beta_multiplier: 0.0,     // Poison rain disabled
            chemical_slope_scale_alpha: 0.1,
            chemical_slope_scale_beta: 0.1,
            mutation_rate: 0.005,
            food_power: 3.0,
            poison_power: 1.0,
            pairing_cost: 0.1,
            max_agents: max_agents as u32,
            cpu_spawn_count: 0,
            agent_count: initial_agents as u32,
            random_seed: seed as u32,
            debug_mode: 0,
            visual_stride: visual_stride_pixels,
            selected_agent_index: u32::MAX,
            repulsion_strength: 10.0,
            agent_repulsion_strength: 1.0,
            gamma_strength: 10.0 * TERRAIN_FORCE_SCALE,
            prop_wash_strength: 1.0,
            gamma_vis_min: 0.0,
            gamma_vis_max: 1.0,
            draw_enabled: 1,
            gamma_debug: 0,
            gamma_hidden: 0,
            slope_debug: 0,
            alpha_show: 1,
            beta_show: 1,
            gamma_show: 1,
            slope_lighting: 0,
            slope_lighting_strength: 1.0,
            trail_diffusion: 0.15,
            trail_decay: 0.995,
            trail_opacity: 0.5,
            trail_show: 0,
            // Initialize with default; will be overwritten after settings load and each frame
            interior_isotropic: 1,
            ignore_stop_codons: 0,
            require_start_codon: 1,
            asexual_reproduction: 0,
            background_color_r: 0.0,
            background_color_g: 0.0,
            background_color_b: 0.0,
            alpha_blend_mode: 0,
            beta_blend_mode: 0,
            gamma_blend_mode: 0,
            slope_blend_mode: 0,
            alpha_color_r: 0.0,
            alpha_color_g: 1.0,
            alpha_color_b: 0.0,
            beta_color_r: 1.0,
            beta_color_g: 0.0,
            beta_color_b: 0.0,
            gamma_color_r: 0.0,
            gamma_color_g: 0.0,
            gamma_color_b: 1.0,
            grid_interpolation: 1,  // Default to bilinear
            alpha_gamma_adjust: 1.0,
            beta_gamma_adjust: 1.0,
            gamma_gamma_adjust: 1.0,
            light_dir_x: 0.5,
            light_dir_y: 0.5,
            light_dir_z: 0.5,
            light_power: 1.0,
            agent_blend_mode: 0,
            agent_color_r: 1.0,
            agent_color_g: 1.0,
            agent_color_b: 1.0,
            agent_color_blend: 0.0,
            epoch: 0,
            vector_force_power: 0.0,
            vector_force_x: 0.0,
            vector_force_y: 0.0,
            inspector_zoom: 1.0,
            agent_trail_decay: 1.0,  // Default to instant clear (original behavior)
            fluid_show: 0,  // Fluid visualization disabled by default
            _padding2: 0.0,
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sim Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let alive_counter = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Alive Counter"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let spawn_debug_counters = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spawn/Debug Counters"),
            size: 8, // 2 x u32
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        queue.write_buffer(&alive_counter, 0, bytemuck::bytes_of(&0u32));
        queue.write_buffer(&spawn_debug_counters, 0, bytemuck::cast_slice(&[0u32, 0u32]));

        let alive_readbacks = [
            Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Alive Readback A"),
                size: 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Alive Readback B"),
                size: 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
        ];

        let alive_readback_pending: [Arc<Mutex<Option<Result<u32, ()>>>>; 2] =
            [Arc::new(Mutex::new(None)), Arc::new(Mutex::new(None))];

        let debug_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Debug Readback"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let agents_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Agents Readback"),
            size: (std::mem::size_of::<Agent>() * max_agents) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let selected_agent_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Selected Agent Buffer"),
            size: std::mem::size_of::<Agent>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let selected_agent_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Selected Agent Readback"),
            size: std::mem::size_of::<Agent>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Spawn/death buffers
        let new_agents_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("New Agents Buffer"),
            size: (std::mem::size_of::<Agent>() * MAX_SPAWN_REQUESTS) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let spawn_requests_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spawn Requests Buffer"),
            size: (std::mem::size_of::<SpawnRequest>() * MAX_SPAWN_REQUESTS) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let spawn_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spawn Readback"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        profiler.mark("Counters and readbacks");

        // Grid readback buffers for snapshot save
        let grid_size_bytes = (GRID_CELL_COUNT * std::mem::size_of::<f32>()) as u64;

        let alpha_grid_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Alpha Grid Readback"),
            size: grid_size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let beta_grid_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Beta Grid Readback"),
            size: grid_size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let gamma_grid_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gamma Grid Readback"),
            size: grid_size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        profiler.mark("Grid readback buffers");

        // Create visual texture at window resolution for crisp rendering
        let visual_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Visual Texture"),
            size: wgpu::Extent3d {
                width: surface_config.width,
                height: surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        profiler.mark("Visual texture");

        let visual_texture_view =
            visual_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        profiler.mark("Sampler");

        // Load main shader (concatenate shared + render + composite + simulation modules)
        let shader_source = format!(
            "{}\n{}\n{}\n{}",
            include_str!("../shaders/shared.wgsl"),
            include_str!("../shaders/render.wgsl"),
            include_str!("../shaders/composite.wgsl"),
            include_str!("../shaders/simulation.wgsl")
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        profiler.mark("Main shader compiled");

        // ============================================================================
        // FLUID SIMULATION BUFFERS (created early so they can be used in bind groups)
        // ============================================================================
        const FLUID_GRID_SIZE: u32 = 128;
        const FLUID_GRID_CELLS: usize = (FLUID_GRID_SIZE * FLUID_GRID_SIZE) as usize;

        let fluid_velocity_size = (FLUID_GRID_CELLS * std::mem::size_of::<[f32; 2]>()) as u64;
        let fluid_scalar_size = (FLUID_GRID_CELLS * std::mem::size_of::<f32>()) as u64;

        let fluid_velocity_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Velocity A"),
            size: fluid_velocity_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let fluid_velocity_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Velocity B"),
            size: fluid_velocity_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let fluid_forces = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Forces"),
            size: fluid_velocity_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Create force vectors buffer for per-frame propeller force injection
        let force_vectors_zeros = vec![[0.0f32, 0.0f32]; FLUID_GRID_CELLS];
        let fluid_force_vectors = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fluid Force Vectors"),
            contents: bytemuck::cast_slice(&force_vectors_zeros),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let fluid_pressure_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Pressure A"),
            size: fluid_scalar_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let fluid_pressure_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Pressure B"),
            size: fluid_scalar_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let fluid_divergence = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Divergence"),
            size: fluid_scalar_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        profiler.mark("Fluid buffers");

        // Create bind group layouts
        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false }, // Changed to allow drain_energy kernel to write
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 9,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 10,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 11,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 12,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 13,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 14,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 15,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 16,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 17,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        profiler.mark("Compute bind layout");

        // Create compute bind groups (ping-pong)
        let compute_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group A"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: agents_buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: agents_buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: alpha_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: beta_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: visual_grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: agent_grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: alive_counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: fluid_velocity_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: new_agents_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: spawn_debug_counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: spawn_requests_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: selected_agent_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: gamma_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: trail_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: environment_init_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: fluid_forces.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: agent_spatial_grid_buffer.as_entire_binding(),
                },
            ],
        });

        let compute_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group B"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: agents_buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: agents_buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: alpha_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: beta_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: visual_grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: agent_grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: alive_counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: fluid_velocity_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: new_agents_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: spawn_debug_counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: spawn_requests_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: selected_agent_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: gamma_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: trail_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: environment_init_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: fluid_forces.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: agent_spatial_grid_buffer.as_entire_binding(),
                },
            ],
        });
        profiler.mark("Compute bind groups");

        // Create compute pipelines
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });
        profiler.mark("Compute pipeline layout");

        let process_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Process Agents Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "process_agents",
            compilation_options: Default::default(),
            cache: None,
        });
        profiler.mark("process_agents pipeline");

        let diffuse_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Diffuse Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "diffuse_grids",
            compilation_options: Default::default(),
            cache: None,
        });
        profiler.mark("diffuse pipeline");

        let diffuse_trails_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Diffuse Trails Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "diffuse_trails",
                compilation_options: Default::default(),
                cache: None,
            });
        profiler.mark("diffuse trails pipeline");

        let gamma_slope_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Gamma Slope Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "compute_gamma_slope",
                compilation_options: Default::default(),
                cache: None,
            });
        profiler.mark("gamma slope pipeline");

        let clear_visual_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Clear Visual Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "clear_visual",
                compilation_options: Default::default(),
                cache: None,
            });
        profiler.mark("clear visual pipeline");

        let motion_blur_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Motion Blur Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "apply_motion_blur",
                compilation_options: Default::default(),
                cache: None,
            });
        profiler.mark("motion blur pipeline");

        let clear_agent_grid_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Clear Agent Grid Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "clear_agent_grid",
                compilation_options: Default::default(),
                cache: None,
            });
        profiler.mark("clear agent grid pipeline");

        let render_inspector_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Render Inspector Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "render_inspector",
                compilation_options: Default::default(),
                cache: None,
            });
        profiler.mark("render inspector pipeline");

        let draw_inspector_agent_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Draw Inspector Agent Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "draw_inspector_agent",
                compilation_options: Default::default(),
                cache: None,
            });
        profiler.mark("draw inspector agent pipeline");

        let render_agents_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Render Agents Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "render_agents",
                compilation_options: Default::default(),
                cache: None,
            });
        profiler.mark("render agents pipeline");

        let composite_agents_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Composite Agents Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "composite_agents",
                compilation_options: Default::default(),
                cache: None,
            });
        profiler.mark("composite agents pipeline");

        let merge_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Merge Agents Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "merge_agents",
            compilation_options: Default::default(),
            cache: None,
        });
        profiler.mark("merge pipeline");

        let compact_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compact Agents Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "compact_agents",
            compilation_options: Default::default(),
            cache: None,
        });
        profiler.mark("compact pipeline");

        let finalize_merge_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Reset Spawn Counter Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "reset_spawn_counter",
                compilation_options: Default::default(),
                cache: None,
            });
        profiler.mark("reset spawn counter pipeline");

        let cpu_spawn_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("CPU Spawn Requests Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "process_cpu_spawns",
            compilation_options: Default::default(),
            cache: None,
        });
        profiler.mark("cpu spawn pipeline");

        let initialize_dead_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Initialize Dead Agents Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "initialize_dead_agents",
                compilation_options: Default::default(),
                cache: None,
            });
        profiler.mark("initialize_dead pipeline");

        let environment_init_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Environment Init Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "initialize_environment",
                compilation_options: Default::default(),
                cache: None,
            });
        profiler.mark("environment init pipeline");

        let generate_map_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Generate Map Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "generate_map",
                compilation_options: Default::default(),
                cache: None,
            });
        profiler.mark("generate map pipeline");

        let clear_agent_spatial_grid_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Clear Agent Spatial Grid Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "clear_agent_spatial_grid",
                compilation_options: Default::default(),
                cache: None,
            });
        profiler.mark("clear agent spatial grid pipeline");

        let populate_agent_spatial_grid_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Populate Agent Spatial Grid Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "populate_agent_spatial_grid",
                compilation_options: Default::default(),
                cache: None,
            });
        profiler.mark("populate agent spatial grid pipeline");

        let drain_energy_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Drain Energy Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "drain_energy",
                compilation_options: Default::default(),
                cache: None,
            });
        profiler.mark("drain energy pipeline");

        // Create render bind group layout
        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Render Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 16,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&visual_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: agent_grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: fluid_velocity_a.as_entire_binding(),
                },
            ],
        });

        // Create render pipeline
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&render_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        profiler.mark("render pipeline");

        // ============================================================================
        // FLUID SIMULATION PIPELINES (buffers already created above, before bind groups)
        // ============================================================================

        // Fluid params buffer (moved from later - same params structure)
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct FluidParams {
            time: f32,
            dt: f32,
            decay: f32,
            grid_size: u32,
            mouse: [f32; 4],
            splat: [f32; 4],
        }

        let fluid_params = FluidParams {
            time: 0.0,
            dt: 0.016,
            decay: 0.9995,
            grid_size: FLUID_GRID_SIZE,
            mouse: [0.0; 4],
            splat: [0.0; 4],
        };

        let fluid_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fluid Params"),
            contents: bytemuck::cast_slice(&[fluid_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Load fluid shader
        let fluid_shader_source = std::fs::read_to_string("shaders/fluid.wgsl")
            .expect("Failed to load shaders/fluid.wgsl");
        let fluid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fluid Shader"),
            source: wgpu::ShaderSource::Wgsl(fluid_shader_source.into()),
        });

        // Fluid bind group layout
        let fluid_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fluid Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Fluid bind groups (ping-pong A->B and B->A)
        let fluid_bind_group_ab = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fluid Bind Group AB"),
            layout: &fluid_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: fluid_velocity_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: fluid_velocity_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: fluid_forces.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: fluid_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: fluid_pressure_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: fluid_pressure_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: fluid_divergence.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: fluid_force_vectors.as_entire_binding() },
            ],
        });

        let fluid_bind_group_ba = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fluid Bind Group BA"),
            layout: &fluid_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: fluid_velocity_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: fluid_velocity_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: fluid_forces.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: fluid_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: fluid_pressure_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: fluid_pressure_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: fluid_divergence.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: fluid_force_vectors.as_entire_binding() },
            ],
        });

        // Fluid pipeline layout
        let fluid_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fluid Pipeline Layout"),
            bind_group_layouts: &[&fluid_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create fluid compute pipelines
        let fluid_generate_forces_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Generate Forces"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "generate_test_forces",
            compilation_options: Default::default(),
            cache: None,
        });

        let fluid_add_forces_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Add Forces"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "add_forces",
            compilation_options: Default::default(),
            cache: None,
        });

        let fluid_advect_velocity_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Advect Velocity"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "advect_velocity",
            compilation_options: Default::default(),
            cache: None,
        });

        let fluid_diffuse_velocity_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Diffuse Velocity"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "diffuse_velocity",
            compilation_options: Default::default(),
            cache: None,
        });

        let fluid_enforce_boundaries_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Enforce Boundaries"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "enforce_boundaries",
            compilation_options: Default::default(),
            cache: None,
        });

        let fluid_divergence_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Divergence"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "compute_divergence",
            compilation_options: Default::default(),
            cache: None,
        });

        let fluid_vorticity_confinement_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Vorticity Confinement"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "vorticity_confinement",
            compilation_options: Default::default(),
            cache: None,
        });

        let fluid_clear_pressure_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Clear Pressure"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "clear_pressure",
            compilation_options: Default::default(),
            cache: None,
        });

        let fluid_jacobi_pressure_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Jacobi Pressure"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "jacobi_pressure",
            compilation_options: Default::default(),
            cache: None,
        });

        let fluid_subtract_gradient_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Subtract Gradient"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "subtract_gradient",
            compilation_options: Default::default(),
            cache: None,
        });

        let fluid_copy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Copy"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "copy_velocity",
            compilation_options: Default::default(),
            cache: None,
        });

        let fluid_copy_velocity_to_forces_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Copy Velocity To Forces"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "copy_velocity_to_forces",
            compilation_options: Default::default(),
            cache: None,
        });

        let fluid_clear_forces_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Clear Forces"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "clear_forces",
            compilation_options: Default::default(),
            cache: None,
        });

        let fluid_clear_force_vectors_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Clear Force Vectors"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "clear_force_vectors",
            compilation_options: Default::default(),
            cache: None,
        });

        let fluid_clear_velocity_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Clear Velocity"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "clear_velocity",
            compilation_options: Default::default(),
            cache: None,
        });

        profiler.mark("Fluid pipelines created");

        let mut init_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Startup Clear Encoder"),
        });

        for buffer in [
            &agents_buffer_a,
            &agents_buffer_b,
            &new_agents_buffer,
            &spawn_requests_buffer,
            &alive_counter,
            &spawn_debug_counters,
            &selected_agent_buffer,
            &visual_grid_buffer,
        ] {
            init_encoder.clear_buffer(buffer, 0, None);
        }

        let env_groups_x = (GRID_DIM_U32 + CLEAR_WG_SIZE_X - 1) / CLEAR_WG_SIZE_X;
        let env_groups_y = (GRID_DIM_U32 + CLEAR_WG_SIZE_Y - 1) / CLEAR_WG_SIZE_Y;
        {
            let mut pass = init_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Environment Init Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&environment_init_pipeline);
            pass.set_bind_group(0, &compute_bind_group_a, &[]);
            pass.dispatch_workgroups(env_groups_x, env_groups_y, 1);
        }

        let agent_clear_groups = ((max_agents as u32) + 255) / 256;
        {
            let mut pass = init_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Initialize Dead Agents A->B"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&initialize_dead_pipeline);
            pass.set_bind_group(0, &compute_bind_group_a, &[]);
            pass.dispatch_workgroups(agent_clear_groups, 1, 1);
        }
        {
            let mut pass = init_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Initialize Dead Agents B->A"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&initialize_dead_pipeline);
            pass.set_bind_group(0, &compute_bind_group_b, &[]);
            pass.dispatch_workgroups(agent_clear_groups, 1, 1);
        }

        // Clear fluid buffers
        let fluid_workgroups = (FLUID_GRID_SIZE + 15) / 16;
        for bg in [&fluid_bind_group_ab, &fluid_bind_group_ba] {
            let mut pass = init_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Clear Fluid Velocity"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&fluid_clear_velocity_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);
        }
        for bg in [&fluid_bind_group_ab, &fluid_bind_group_ba] {
            let mut pass = init_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Clear Fluid Pressure"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&fluid_clear_pressure_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);
        }

        queue.submit(std::iter::once(init_encoder.finish()));
        profiler.mark("Initial GPU passes submitted");

        let settings_path = SimulationSettings::default_path();
        let (mut settings, mut needs_save) =
            match SimulationSettings::load_from_disk(&settings_path) {
                Ok(value) => (value, false),
                Err(err) => {
                    eprintln!(
                        "Warning: failed to load simulation settings from {:?}: {err:?}",
                        &settings_path
                    );
                    (SimulationSettings::default(), true)
                }
            };

        // Reset difficulty levels to 0 on startup
        settings.difficulty.food_power.difficulty_level = 0;
        settings.difficulty.food_power.last_adjustment_epoch = 0;
        settings.difficulty.poison_power.difficulty_level = 0;
        settings.difficulty.poison_power.last_adjustment_epoch = 0;
        settings.difficulty.spawn_prob.difficulty_level = 0;
        settings.difficulty.spawn_prob.last_adjustment_epoch = 0;
        settings.difficulty.death_prob.difficulty_level = 0;
        settings.difficulty.death_prob.last_adjustment_epoch = 0;
        settings.difficulty.alpha_rain.difficulty_level = 0;
        settings.difficulty.alpha_rain.last_adjustment_epoch = 0;
        settings.difficulty.beta_rain.difficulty_level = 0;
        settings.difficulty.beta_rain.last_adjustment_epoch = 0;

        let original_settings = settings.clone();
        settings.sanitize();
        if settings != original_settings {
            needs_save = true;
        }
        profiler.mark("Settings loaded");

        let mut state = Self {
            device,
            queue,
            surface,
            surface_config,
            agents_buffer_a,
            agents_buffer_b,
            alpha_grid,
            beta_grid,
            rain_map_buffer,
            agent_spatial_grid_buffer,
            gamma_grid,
            trail_grid,
            visual_grid_buffer,
            agent_grid_buffer,
            params_buffer,
            environment_init_params_buffer,
            alive_counter,
            spawn_debug_counters,
            alive_readbacks,
            alive_readback_pending,
            alive_readback_inflight: [false; 2],
            alive_readback_slot: 0,
            debug_readback,
            agents_readback,
            selected_agent_buffer,
            selected_agent_readback,
            new_agents_buffer,
            spawn_readback,
            spawn_requests_buffer,
            alpha_grid_readback,
            beta_grid_readback,
            gamma_grid_readback,
            visual_texture,
            visual_texture_view,
            sampler,
            process_pipeline,
            diffuse_pipeline,
            diffuse_trails_pipeline,
            clear_visual_pipeline,
            motion_blur_pipeline,
            clear_agent_grid_pipeline,
            render_agents_pipeline,
            render_inspector_pipeline,
            draw_inspector_agent_pipeline,
            composite_agents_pipeline,
            gamma_slope_pipeline,
            merge_pipeline,
            compact_pipeline,
            finalize_merge_pipeline,
            cpu_spawn_pipeline,
            initialize_dead_pipeline,
            environment_init_pipeline,
            generate_map_pipeline,
            clear_agent_spatial_grid_pipeline,
            populate_agent_spatial_grid_pipeline,
            drain_energy_pipeline,
            render_pipeline,
            compute_bind_group_a,
            compute_bind_group_b,
            render_bind_group,
            ping_pong: false,
            agent_count: initial_agents as u32,
            alive_count: initial_agents as u32,
            camera_zoom: settings.camera_zoom,
            camera_pan: [SIM_SIZE / 2.0, SIM_SIZE / 2.0],
            prev_camera_pan: [SIM_SIZE / 2.0, SIM_SIZE / 2.0], // Initialize to same as camera_pan
            agents_cpu: agents,
            agent_buffer_capacity: max_agents,
            cpu_spawn_queue: Vec::new(),
            spawn_request_count: 0,
            pending_spawn_upload: false,
            spawn_probability: settings.spawn_probability,
            death_probability: settings.death_probability,
            auto_replenish: settings.auto_replenish,
            rain_map_data: vec![1.0f32; GRID_CELL_COUNT * 2], // Initialize with uniform rain
            difficulty: settings.difficulty.clone(),
            frame_count: 0,
            last_fps_update: std::time::Instant::now(),
            limit_fps: settings.limit_fps,
            limit_fps_25: settings.limit_fps_25,
            last_frame_time: std::time::Instant::now(),
            render_interval: settings.render_interval,
            current_mode: if settings.limit_fps {
                if settings.limit_fps_25 { 3 } else { 0 }
            } else {
                1
            },
            epoch: 0,
            last_epoch_update: std::time::Instant::now(),
            last_epoch_count: 0,
            inspector_frame_counter: 0,
            epochs_per_second: 0.0,
            population_history: Vec::new(),
            alpha_rain_history: VecDeque::new(),
            beta_rain_history: VecDeque::new(),
            epoch_sample_interval: 1000,
            last_sample_epoch: 0,
            max_history_points: 5000,
            last_autosave_epoch: 0,
            is_dragging: false,
            last_mouse_pos: None,
            selected_agent_index: None,
            selected_agent_data: None,
            follow_selected_agent: false, // Initialize to false
            camera_target: [SIM_SIZE / 2.0, SIM_SIZE / 2.0], // Initialize to center
            camera_velocity: [0.0, 0.0], // Initialize velocity to zero
            inspector_zoom: 1.0,
            agent_trail_decay: settings.agent_trail_decay,
            rng_state: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            window: window_clone,
            egui_renderer,
            ui_tab: 0, // Start on Simulation tab
            ui_visible: true, // Control panel visible by default
            debug_per_segment: settings.debug_per_segment,
            is_paused: false,
            snapshot_save_requested: false,
            snapshot_load_requested: false,
            visual_stride_pixels,
            alpha_blur: settings.alpha_blur,
            beta_blur: settings.beta_blur,
            gamma_blur: settings.gamma_blur,
            alpha_slope_bias: settings.alpha_slope_bias,
            beta_slope_bias: settings.beta_slope_bias,
            alpha_multiplier: settings.alpha_multiplier,
            beta_multiplier: settings.beta_multiplier,
            alpha_rain_variation: settings.alpha_rain_variation,
            beta_rain_variation: settings.beta_rain_variation,
            alpha_rain_phase: settings.alpha_rain_phase,
            beta_rain_phase: settings.beta_rain_phase,
            alpha_rain_freq: settings.alpha_rain_freq,
            beta_rain_freq: settings.beta_rain_freq,
            alpha_rain_map_path: None,
            beta_rain_map_path: None,
            alpha_rain_thumbnail: None,
            beta_rain_thumbnail: None,
            chemical_slope_scale_alpha: settings.chemical_slope_scale_alpha,
            chemical_slope_scale_beta: settings.chemical_slope_scale_beta,
            alpha_noise_scale: settings.alpha_noise_scale,
            beta_noise_scale: settings.beta_noise_scale,
            gamma_noise_scale: settings.gamma_noise_scale,
            noise_power: settings.noise_power,
            food_power: settings.food_power,
            poison_power: settings.poison_power,
            amino_maintenance_cost: settings.amino_maintenance_cost,
            pairing_cost: settings.pairing_cost,
            diffusion_interval: settings.diffusion_interval,
            diffusion_counter: 0,
            slope_interval: settings.slope_interval,
            slope_counter: 0,
            mutation_rate: settings.mutation_rate,
            repulsion_strength: settings.repulsion_strength,
            agent_repulsion_strength: settings.agent_repulsion_strength,
            gamma_debug_visual: settings.gamma_debug_visual,
            slope_debug_visual: settings.slope_debug_visual,
            rain_debug_visual: settings.rain_debug_visual,
            fluid_show: settings.fluid_show,
            fluid_velocity_a,
            fluid_velocity_b,
            fluid_pressure_a,
            fluid_pressure_b,
            fluid_divergence,
            fluid_forces,
            fluid_force_vectors,
            fluid_params_buffer,
            fluid_generate_forces_pipeline,
            fluid_add_forces_pipeline,
            fluid_advect_velocity_pipeline,
            fluid_diffuse_velocity_pipeline,
            fluid_enforce_boundaries_pipeline,
            fluid_divergence_pipeline,
            fluid_vorticity_confinement_pipeline,
            fluid_clear_pressure_pipeline,
            fluid_jacobi_pressure_pipeline,
            fluid_subtract_gradient_pipeline,
            fluid_copy_pipeline,
            fluid_copy_velocity_to_forces_pipeline,
            fluid_clear_forces_pipeline,
            fluid_clear_force_vectors_pipeline,
            fluid_clear_velocity_pipeline,
            fluid_bind_group_ab,
            fluid_bind_group_ba,
            fluid_time: 0.0,
            vector_force_power: settings.vector_force_power,
            vector_force_x: settings.vector_force_x,
            vector_force_y: settings.vector_force_y,
            prop_wash_strength: settings.prop_wash_strength,
            gamma_hidden: settings.gamma_hidden,
            gamma_vis_min: settings.gamma_vis_min,
            gamma_vis_max: settings.gamma_vis_max,
            alpha_show: settings.alpha_show,
            beta_show: settings.beta_show,
            gamma_show: settings.gamma_show,
            slope_lighting: settings.slope_lighting,
            slope_lighting_strength: settings.slope_lighting_strength,
            trail_diffusion: settings.trail_diffusion,
            trail_decay: settings.trail_decay,
            trail_opacity: settings.trail_opacity,
            trail_show: settings.trail_show,
            interior_isotropic: settings.interior_isotropic,
            ignore_stop_codons: settings.ignore_stop_codons,
            require_start_codon: settings.require_start_codon,
            asexual_reproduction: settings.asexual_reproduction,
            background_color: settings.background_color,
            alpha_blend_mode: settings.alpha_blend_mode,
            beta_blend_mode: settings.beta_blend_mode,
            gamma_blend_mode: settings.gamma_blend_mode,
            slope_blend_mode: settings.slope_blend_mode,
            alpha_color: settings.alpha_color,
            beta_color: settings.beta_color,
            gamma_color: settings.gamma_color,
            grid_interpolation: settings.grid_interpolation,
            alpha_gamma_adjust: settings.alpha_gamma_adjust,
            beta_gamma_adjust: settings.beta_gamma_adjust,
            gamma_gamma_adjust: settings.gamma_gamma_adjust,
            light_direction: settings.light_direction,
            light_power: settings.light_power,
            agent_blend_mode: settings.agent_blend_mode,
            agent_color: settings.agent_color,
            agent_color_blend: settings.agent_color_blend,
            settings_path: settings_path.clone(),
            last_saved_settings: settings.clone(),
            destroyed: false,
        };

        profiler.mark("GpuState constructed");

        state.update_present_mode();

        if let Some(path) = settings.alpha_rain_map_path.clone() {
            if let Err(err) = state.load_alpha_rain_map(&path) {
                eprintln!("Failed to load alpha rain map {}: {err:?}", path.display());
            }
        }
        if let Some(path) = settings.beta_rain_map_path.clone() {
            if let Err(err) = state.load_beta_rain_map(&path) {
                eprintln!("Failed to load beta rain map {}: {err:?}", path.display());
            }
        }

        if needs_save || !settings_path.exists() {
            if let Err(err) = settings.save_to_disk(&settings_path) {
                eprintln!(
                    "Warning: failed to write simulation settings to {:?}: {err:?}",
                    settings_path
                );
            }
        }

        state
    }

    // Queue N random agents to be spawned next frame via the GPU merge pass
    fn queue_random_spawns(&mut self, count: usize) {
        for _ in 0..count {
            // Advance RNG using same method as update()
            self.rng_state ^= self.rng_state << 13;
            self.rng_state ^= self.rng_state >> 7;
            self.rng_state ^= self.rng_state << 17;
            let seed = (self.rng_state >> 32) as u32;

            self.rng_state ^= self.rng_state << 13;
            self.rng_state ^= self.rng_state >> 7;
            self.rng_state ^= self.rng_state << 17;
            let genome_seed = (self.rng_state >> 32) as u32;

            let request = SpawnRequest {
                seed,
                genome_seed,
                flags: 0, // random genome generation path (ignore override)
                _pad_seed: 0,
                position: [0.0, 0.0], // Let GPU generate random position using seed
                energy: 10.0,
                rotation: 0.0, // Let GPU generate random rotation using seed
                genome_override: [0u32; GENOME_WORDS],
            };

            self.cpu_spawn_queue.push(request);
        }

        self.spawn_request_count = self.cpu_spawn_queue.len() as u32;
        self.pending_spawn_upload = true;
        self.window.request_redraw();
    }

    fn load_gamma_image<P: AsRef<Path>>(&mut self, path: P) -> anyhow::Result<()> {
        let path = path.as_ref();
        let image = image::open(path)?;
        let resized = image.resize_exact(GRID_DIM as u32, GRID_DIM as u32, FilterType::Lanczos3);
        let gray = resized.to_luma8();
        let width = gray.width() as usize;
        let height = gray.height() as usize;
        let raw = gray.as_raw();

        let mut gamma_values = Vec::with_capacity(GRID_CELL_COUNT);
        for row in (0..height).rev() {
            let row_offset = row * width;
            for col in 0..width {
                let pix = raw[row_offset + col] as f32 / 255.0;
                gamma_values.push(pix.clamp(0.0, 1.0));
            }
        }

        debug_assert_eq!(gamma_values.len(), GRID_CELL_COUNT);

        self.queue
            .write_buffer(&self.gamma_grid, 0, bytemuck::cast_slice(&gamma_values));
        self.slope_counter = 0;

        println!(
            "Loaded gamma terrain from {} ({}x{})",
            path.display(),
            width,
            height
        );

        Ok(())
    }

    fn read_rain_map(path: &Path) -> anyhow::Result<(Vec<f32>, ColorImage)> {
        let image = image::open(path)?;
        let resized = image.resize_exact(GRID_DIM as u32, GRID_DIM as u32, FilterType::Lanczos3);
        let gray = resized.to_luma8();
        let width = gray.width() as usize;
        let height = gray.height() as usize;
        let raw = gray.as_raw();

        let mut values = Vec::with_capacity(GRID_CELL_COUNT);
        for row in (0..height).rev() {
            let row_offset = row * width;
            for col in 0..width {
                let pix = raw[row_offset + col] as f32 / 255.0;
                values.push(pix.clamp(0.0, 1.0));
            }
        }

        debug_assert_eq!(values.len(), GRID_CELL_COUNT);
        let thumbnail = Self::build_rain_thumbnail(&gray);
        Ok((values, thumbnail))
    }

    fn build_rain_thumbnail(gray: &GrayImage) -> ColorImage {
        let thumbnail = image::imageops::resize(
            gray,
            RAIN_THUMB_SIZE as u32,
            RAIN_THUMB_SIZE as u32,
            FilterType::Lanczos3,
        );
        let width = thumbnail.width() as usize;
        let height = thumbnail.height() as usize;
        let mut pixels = Vec::with_capacity(width * height);
        let raw = thumbnail.as_raw();
        for row in 0..height {
            let row_offset = row * width;
            for col in 0..width {
                let intensity = raw[row_offset + col];
                pixels.push(Color32::from_gray(intensity));
            }
        }
        ColorImage {
            size: [width, height],
            pixels,
        }
    }

    fn alpha_rain_texture_id(&mut self, ctx: &egui::Context) -> Option<TextureId> {
        self.alpha_rain_thumbnail
            .as_mut()
            .map(|thumb| thumb.ensure_texture(ctx, "Alpha Rain Preview"))
    }

    fn beta_rain_texture_id(&mut self, ctx: &egui::Context) -> Option<TextureId> {
        self.beta_rain_thumbnail
            .as_mut()
            .map(|thumb| thumb.ensure_texture(ctx, "Beta Rain Preview"))
    }

    fn load_alpha_rain_map<P: AsRef<Path>>(&mut self, path: P) -> anyhow::Result<()> {
        let path = path.as_ref();
        let (alpha_values, thumbnail) = Self::read_rain_map(path)?;

        // Update alpha values in CPU-side data (even indices)
        for i in 0..GRID_CELL_COUNT {
            self.rain_map_data[i * 2] = alpha_values[i];
        }

        // Upload to GPU
        self.queue.write_buffer(
            &self.rain_map_buffer,
            0,
            bytemuck::cast_slice(&self.rain_map_data),
        );
        self.alpha_rain_map_path = Some(path.to_path_buf());
        self.alpha_rain_thumbnail = Some(RainThumbnail::new(thumbnail));
        println!("Loaded alpha rain map from {}", path.display());
        Ok(())
    }

    fn load_beta_rain_map<P: AsRef<Path>>(&mut self, path: P) -> anyhow::Result<()> {
        let path = path.as_ref();
        let (beta_values, thumbnail) = Self::read_rain_map(path)?;

        // Update beta values in CPU-side data (odd indices)
        for i in 0..GRID_CELL_COUNT {
            self.rain_map_data[i * 2 + 1] = beta_values[i];
        }

        // Upload to GPU
        self.queue.write_buffer(
            &self.rain_map_buffer,
            0,
            bytemuck::cast_slice(&self.rain_map_data),
        );
        self.beta_rain_map_path = Some(path.to_path_buf());
        self.beta_rain_thumbnail = Some(RainThumbnail::new(thumbnail));
        println!("Loaded beta rain map from {}", path.display());
        Ok(())
    }

    fn clear_alpha_rain_map(&mut self) {
        // Set alpha to uniform 1.0 in CPU-side data (even indices)
        for i in 0..GRID_CELL_COUNT {
            self.rain_map_data[i * 2] = 1.0;
        }

        // Upload to GPU
        self.queue.write_buffer(
            &self.rain_map_buffer,
            0,
            bytemuck::cast_slice(&self.rain_map_data),
        );
        self.alpha_rain_map_path = None;
        self.alpha_rain_thumbnail = None;
        println!("Cleared alpha rain map (uniform probability)");
    }

    fn clear_beta_rain_map(&mut self) {
        // Set beta to uniform 1.0 in CPU-side data (odd indices)
        for i in 0..GRID_CELL_COUNT {
            self.rain_map_data[i * 2 + 1] = 1.0;
        }

        // Upload to GPU
        self.queue.write_buffer(
            &self.rain_map_buffer,
            0,
            bytemuck::cast_slice(&self.rain_map_data),
        );
        self.beta_rain_map_path = None;
        self.beta_rain_thumbnail = None;
        println!("Cleared beta rain map (uniform probability)");
    }

    // Replenish population - spawns random agents when population is low
    fn replenish_population(&mut self) {
        // Only replenish if population drops below 100 agents AND there was a population before
        // (prevents spawning on startup with 0 agents)
        if self.alive_count >= 100 || self.alive_count == 0 {
            return;
        }

        // Spawn 100 completely random agents
        let spawn_count = 100;

        // Always spawn completely random agents (no cloning)
        {
            // Spawn completely random ones
            for _ in 0..spawn_count {
                // Advance RNG using same method as update() and queue_random_spawns()
                self.rng_state ^= self.rng_state << 13;
                self.rng_state ^= self.rng_state >> 7;
                self.rng_state ^= self.rng_state << 17;
                let seed = (self.rng_state >> 32) as u32;

                self.rng_state ^= self.rng_state << 13;
                self.rng_state ^= self.rng_state >> 7;
                self.rng_state ^= self.rng_state << 17;
                let genome_seed = (self.rng_state >> 32) as u32;

                let request = SpawnRequest {
                    seed,
                    genome_seed,
                    flags: 0u32, // random genome generation path
                    _pad_seed: 0,
                    position: [0.0, 0.0], // Let GPU generate random position using seed
                    energy: 10.0,
                    rotation: 0.0, // Let GPU generate random rotation using seed
                    genome_override: [0u32; GENOME_WORDS],
                };

                self.cpu_spawn_queue.push(request);
            }
        }

        self.spawn_request_count = self.cpu_spawn_queue.len() as u32;
        self.pending_spawn_upload = true;
        self.window.request_redraw();

        if !self.cpu_spawn_queue.is_empty() {
            self.spawn_request_count = self.cpu_spawn_queue.len() as u32;
            self.pending_spawn_upload = true;
            self.window.request_redraw();
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);
            // Recreate visual resources to match new size
            self.recreate_visual_resources();
        }
    }

    fn recreate_visual_resources(&mut self) {
        // Recreate visual texture
        self.visual_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Visual Texture"),
            size: wgpu::Extent3d {
                width: self.surface_config.width,
                height: self.surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.visual_texture_view = self
            .visual_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Recreate visual grid buffer with padded stride
        let bytes_per_pixel: u32 = 16;
        let align: u32 = 256;
        let stride_bytes =
            ((self.surface_config.width * bytes_per_pixel + (align - 1)) / align) * align;
        self.visual_stride_pixels = stride_bytes / bytes_per_pixel;
        self.visual_grid_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Visual Grid"),
            size: (stride_bytes * self.surface_config.height) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Agent grid (same size as visual grid, leftmost 300px reserved for inspector)
        self.agent_grid_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Agent Grid"),
            size: (stride_bytes * self.surface_config.height) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Recreate bind groups referencing updated resources
        // Compute bind groups
        let layout0 = self.process_pipeline.get_bind_group_layout(0);
        self.compute_bind_group_a = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group A"),
            layout: &layout0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.agents_buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.agents_buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.alpha_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.beta_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.visual_grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.agent_grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.alive_counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.fluid_velocity_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.new_agents_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: self.spawn_debug_counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.spawn_requests_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: self.selected_agent_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: self.gamma_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: self.trail_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: self.environment_init_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: self.rain_map_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: self.agent_spatial_grid_buffer.as_entire_binding(),
                },
            ],
        });
        self.compute_bind_group_b = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group B"),
            layout: &layout0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.agents_buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.agents_buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.alpha_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.beta_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.visual_grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.agent_grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.alive_counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.fluid_velocity_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.new_agents_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: self.spawn_debug_counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.spawn_requests_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: self.selected_agent_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: self.gamma_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: self.trail_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: self.environment_init_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: self.rain_map_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: self.agent_spatial_grid_buffer.as_entire_binding(),
                },
            ],
        });

        // Render bind group
        let rlayout0 = self.render_pipeline.get_bind_group_layout(0);
        self.render_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &rlayout0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.visual_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.agent_grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: self.fluid_velocity_a.as_entire_binding(),
                },
            ],
        });
    }

    fn update_present_mode(&mut self) {
        // Update present mode based on FPS limit setting
        self.surface_config.present_mode = if self.limit_fps {
            wgpu::PresentMode::Fifo // VSync (60 FPS)
        } else {
            wgpu::PresentMode::Immediate // Max speed
        };
        self.surface.configure(&self.device, &self.surface_config);
    }

    fn set_speed_mode(&mut self, mode: u32) {
        let sanitized = match mode {
            0 | 1 | 2 | 3 => mode,
            _ => self.current_mode,
        };
        self.current_mode = sanitized;
        match sanitized {
            0 => {
                self.limit_fps = true;
                self.limit_fps_25 = false;
            }
            1 => {
                self.limit_fps = false;
                self.limit_fps_25 = false;
            }
            2 => {
                self.limit_fps = false;
                self.limit_fps_25 = false;
            }
            3 => {
                self.limit_fps = true;
                self.limit_fps_25 = true;
            }
            _ => {}
        }
        self.update_present_mode();
    }

    fn frame_time_cap(&self) -> Option<std::time::Duration> {
        if !self.limit_fps {
            return None;
        }
        let micros = if self.limit_fps_25 { 40_000 } else { 16_667 };
        Some(std::time::Duration::from_micros(micros))
    }

    fn current_settings(&self) -> SimulationSettings {
        SimulationSettings {
            camera_zoom: self.camera_zoom,
            spawn_probability: self.spawn_probability,
            death_probability: self.death_probability,
            mutation_rate: self.mutation_rate,
            auto_replenish: self.auto_replenish,
            diffusion_interval: self.diffusion_interval,
            slope_interval: self.slope_interval,
            alpha_blur: self.alpha_blur,
            beta_blur: self.beta_blur,
            gamma_blur: self.gamma_blur,
            alpha_slope_bias: self.alpha_slope_bias,
            beta_slope_bias: self.beta_slope_bias,
            alpha_multiplier: self.alpha_multiplier,
            beta_multiplier: self.beta_multiplier,
            alpha_rain_map_path: self.alpha_rain_map_path.clone(),
            beta_rain_map_path: self.beta_rain_map_path.clone(),
            chemical_slope_scale_alpha: self.chemical_slope_scale_alpha,
            chemical_slope_scale_beta: self.chemical_slope_scale_beta,
            alpha_noise_scale: self.alpha_noise_scale,
            beta_noise_scale: self.beta_noise_scale,
            gamma_noise_scale: self.gamma_noise_scale,
            noise_power: self.noise_power,
            food_power: self.food_power,
            poison_power: self.poison_power,
            amino_maintenance_cost: self.amino_maintenance_cost,
            pairing_cost: self.pairing_cost,
            prop_wash_strength: self.prop_wash_strength,
            repulsion_strength: self.repulsion_strength,
            agent_repulsion_strength: self.agent_repulsion_strength,
            limit_fps: self.limit_fps,
            limit_fps_25: self.limit_fps_25,
            render_interval: self.render_interval,
            gamma_debug_visual: self.gamma_debug_visual,
            slope_debug_visual: self.slope_debug_visual,
            rain_debug_visual: self.rain_debug_visual,
            fluid_show: self.fluid_show,
            vector_force_power: self.vector_force_power,
            vector_force_x: self.vector_force_x,
            vector_force_y: self.vector_force_y,
            gamma_hidden: self.gamma_hidden,
            debug_per_segment: self.debug_per_segment,
            gamma_vis_min: self.gamma_vis_min,
            gamma_vis_max: self.gamma_vis_max,
            alpha_show: self.alpha_show,
            beta_show: self.beta_show,
            gamma_show: self.gamma_show,
            slope_lighting: self.slope_lighting,
            slope_lighting_strength: self.slope_lighting_strength,
            trail_diffusion: self.trail_diffusion,
            trail_decay: self.trail_decay,
            trail_opacity: self.trail_opacity,
            trail_show: self.trail_show,
            interior_isotropic: self.interior_isotropic,
            ignore_stop_codons: self.ignore_stop_codons,
            require_start_codon: self.require_start_codon,
            asexual_reproduction: self.asexual_reproduction,
            alpha_rain_variation: self.alpha_rain_variation,
            beta_rain_variation: self.beta_rain_variation,
            alpha_rain_phase: self.alpha_rain_phase,
            beta_rain_phase: self.beta_rain_phase,
            alpha_rain_freq: self.alpha_rain_freq,
            beta_rain_freq: self.beta_rain_freq,
            difficulty: self.difficulty.clone(),
            background_color: self.background_color,
            alpha_blend_mode: self.alpha_blend_mode,
            beta_blend_mode: self.beta_blend_mode,
            gamma_blend_mode: self.gamma_blend_mode,
            slope_blend_mode: self.slope_blend_mode,
            alpha_color: self.alpha_color,
            beta_color: self.beta_color,
            gamma_color: self.gamma_color,
            grid_interpolation: self.grid_interpolation,
            alpha_gamma_adjust: self.alpha_gamma_adjust,
            beta_gamma_adjust: self.beta_gamma_adjust,
            gamma_gamma_adjust: self.gamma_gamma_adjust,
            light_direction: self.light_direction,
            light_power: self.light_power,
            agent_blend_mode: self.agent_blend_mode,
            agent_color: self.agent_color,
            agent_color_blend: self.agent_color_blend,
            agent_trail_decay: self.agent_trail_decay,
        }
    }

    fn persist_settings_if_changed(&mut self) {
        let mut previous = self.last_saved_settings.clone();
        previous.sanitize();

        let mut current = self.current_settings();
        current.sanitize();

        if current.limit_fps != previous.limit_fps {
            self.update_present_mode();
        }

        if current != previous {
            if let Err(err) = current.save_to_disk(&self.settings_path) {
                eprintln!(
                    "Warning: failed to write simulation settings to {:?}: {err:?}",
                    &self.settings_path
                );
            } else {
                self.last_saved_settings = current;
            }
        }
    }

    fn process_completed_alive_readbacks(&mut self) {
        self.device.poll(wgpu::Maintain::Poll);
        for idx in 0..2 {
            let message = {
                let mut guard = self.alive_readback_pending[idx].lock().unwrap();
                guard.take()
            };
            if let Some(result) = message {
                if let Ok(new_count) = result {
                    self.agent_count = new_count;
                    self.alive_count = new_count;
                }
                self.alive_readback_inflight[idx] = false;
            }
        }
    }

    fn destroy_resources(&mut self) {
        if self.destroyed {
            return;
        }

        // Ensure GPU work completes before releasing resources to avoid device validation issues.
        self.device.poll(wgpu::Maintain::Wait);

        self.agents_buffer_a.destroy();
        self.agents_buffer_b.destroy();
        self.alpha_grid.destroy();
        self.beta_grid.destroy();
        self.rain_map_buffer.destroy();
        self.agent_spatial_grid_buffer.destroy();
        self.gamma_grid.destroy();
        self.trail_grid.destroy();
        self.visual_grid_buffer.destroy();
        self.params_buffer.destroy();
        self.environment_init_params_buffer.destroy();
        self.alive_counter.destroy();
        self.spawn_debug_counters.destroy();
        for buffer in &self.alive_readbacks {
            buffer.destroy();
        }
        self.debug_readback.destroy();
        self.agents_readback.destroy();
        self.selected_agent_buffer.destroy();
        self.selected_agent_readback.destroy();
        self.new_agents_buffer.destroy();
        self.spawn_readback.destroy();
        self.spawn_requests_buffer.destroy();
        self.visual_texture.destroy();

        self.cpu_spawn_queue.clear();
        self.pending_spawn_upload = false;
        self.spawn_request_count = 0;
        self.selected_agent_index = None;
        self.selected_agent_data = None;

        self.destroyed = true;
    }

    fn ensure_alive_slot_ready(&mut self) -> usize {
        let slot = self.alive_readback_slot;
        if self.alive_readback_inflight[slot] {
            self.device.poll(wgpu::Maintain::Wait);
            let message = {
                let mut guard = self.alive_readback_pending[slot].lock().unwrap();
                guard.take()
            };
            if let Some(result) = message {
                if let Ok(new_count) = result {
                    self.agent_count = new_count;
                    self.alive_count = new_count;
                }
            }
            self.alive_readback_inflight[slot] = false;
        }
        slot
    }

    fn copy_state_to_staging(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        slot: usize,
    ) -> Arc<wgpu::Buffer> {
        let alive_buffer = self.alive_readbacks[slot].clone();
        encoder.copy_buffer_to_buffer(&self.alive_counter, 0, alive_buffer.as_ref(), 0, 4);

        encoder.copy_buffer_to_buffer(
            &self.selected_agent_buffer,
            0,
            &self.selected_agent_readback,
            0,
            std::mem::size_of::<Agent>() as u64,
        );

        alive_buffer
    }

    fn kickoff_alive_readback(&mut self, slot: usize, alive_buffer: Arc<wgpu::Buffer>) {
        let buffer_for_map = alive_buffer.clone();
        let buffer_for_callback = buffer_for_map.clone();
        let pending = self.alive_readback_pending[slot].clone();
        self.alive_readback_inflight[slot] = true;
        buffer_for_map
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let message = match result {
                    Ok(()) => {
                        let slice = buffer_for_callback.slice(..);
                        let data = slice.get_mapped_range();
                        let new_count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                        drop(data);
                        buffer_for_callback.unmap();
                        Ok(new_count)
                    }
                    Err(_) => {
                        buffer_for_callback.unmap();
                        Err(())
                    }
                };

                if let Ok(mut guard) = pending.lock() {
                    *guard = Some(message);
                }
            });
        self.alive_readback_slot = (slot + 1) % 2;
    }

    fn perform_optional_readbacks(&mut self, do_readbacks: bool) {
        if !do_readbacks {
            return;
        }

        if self.selected_agent_index.is_some() {
            let selected_slice = self.selected_agent_readback.slice(..);
            selected_slice.map_async(wgpu::MapMode::Read, |_| {});
            self.device.poll(wgpu::Maintain::Wait);
            {
                let selected_data = selected_slice.get_mapped_range();
                if selected_data.len() >= std::mem::size_of::<Agent>() {
                    let agent_bytes = &selected_data[..std::mem::size_of::<Agent>()];
                    let agent: Agent = bytemuck::pod_read_unaligned(agent_bytes);
                    self.selected_agent_data = Some(agent);

                    // Update camera target if following this agent
                    if self.follow_selected_agent {
                        self.camera_target = agent.position;
                    }
                }
            }
            self.selected_agent_readback.unmap();
        }
    }

    fn process_spawn_requests_only(&mut self, cpu_spawn_count: u32, do_readbacks: bool) {
        self.process_completed_alive_readbacks();

        if cpu_spawn_count == 0 && !do_readbacks {
            return;
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Paused Spawn Encoder"),
            });

        if cpu_spawn_count > 0 {
            encoder.clear_buffer(&self.alive_counter, 0, None);
            encoder.clear_buffer(&self.spawn_debug_counters, 0, None);

            // Upload spawn requests for this batch so GPU has per-request seeds/data
            // Limit to the count actually being processed (capped at 2000 elsewhere)
            let upload_len = (cpu_spawn_count as usize)
                .min(self.cpu_spawn_queue.len())
                .min(2000);
            if upload_len > 0 {
                let slice = &self.cpu_spawn_queue[..upload_len];
                self.queue.write_buffer(
                    &self.spawn_requests_buffer,
                    0,
                    bytemuck::cast_slice(slice),
                );
                // After upload we no longer need to treat this as pending
                self.pending_spawn_upload = false;
            }

            let bg_swap = if self.ping_pong {
                &self.compute_bind_group_a
            } else {
                &self.compute_bind_group_b
            };

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Paused Spawn Pass"),
                    timestamp_writes: None,
                });

                cpass.set_pipeline(&self.compact_pipeline);
                cpass.set_bind_group(0, bg_swap, &[]);
                cpass.dispatch_workgroups((self.agent_count + 63) / 64, 1, 1);

                cpass.set_pipeline(&self.cpu_spawn_pipeline);
                cpass.set_bind_group(0, bg_swap, &[]);
                cpass.dispatch_workgroups((cpu_spawn_count + 63) / 64, 1, 1);

                cpass.set_pipeline(&self.merge_pipeline);
                cpass.set_bind_group(0, bg_swap, &[]);
                cpass.dispatch_workgroups((2000 + 63) / 64, 1, 1);

                let init_groups = ((self.agent_buffer_capacity as u32) + 255) / 256;
                cpass.set_pipeline(&self.initialize_dead_pipeline);
                cpass.set_bind_group(0, bg_swap, &[]);
                cpass.dispatch_workgroups(init_groups, 1, 1);

                cpass.set_pipeline(&self.finalize_merge_pipeline);
                cpass.set_bind_group(0, bg_swap, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }
        }

        let slot = self.ensure_alive_slot_ready();
        let alive_buffer = self.copy_state_to_staging(&mut encoder, slot);

        self.queue.submit(Some(encoder.finish()));

        self.kickoff_alive_readback(slot, alive_buffer);

        self.perform_optional_readbacks(do_readbacks);

        // Poll for the alive count readback to complete so agent_count is updated immediately
        self.device.poll(wgpu::Maintain::Wait);
        self.process_completed_alive_readbacks();

        // Clear the processed spawn requests from the queue
        if cpu_spawn_count > 0 {
            let drain_count = (cpu_spawn_count as usize).min(self.cpu_spawn_queue.len());
            self.cpu_spawn_queue.drain(0..drain_count);
            self.spawn_request_count = self.cpu_spawn_queue.len() as u32;
            if self.cpu_spawn_queue.is_empty() {
                self.pending_spawn_upload = false;
            }
        }
    }

    pub fn update(&mut self, should_draw: bool, frame_dt: f32) {
        // Smooth camera following with continuous integration
        if self.follow_selected_agent {
            // Store previous camera position for motion blur
            self.prev_camera_pan = self.camera_pan;

            // Frame-rate independent integration factor using exponential decay
            let clamped_dt = frame_dt.clamp(0.001, 0.1);
            let damping_rate = 8.0; // Much faster follow (~12% step at 60fps: 1 - exp(-8.0*0.016) ≈ 0.12)
            let integration_factor = 1.0 - (-damping_rate * clamped_dt).exp();

            // Smoothly integrate target position into camera position
            self.camera_pan[0] += (self.camera_target[0] - self.camera_pan[0]) * integration_factor;
            self.camera_pan[1] += (self.camera_target[1] - self.camera_pan[1]) * integration_factor;

            // Clamp to world bounds
            self.camera_pan[0] = self.camera_pan[0].clamp(-0.25 * SIM_SIZE, 1.25 * SIM_SIZE);
            self.camera_pan[1] = self.camera_pan[1].clamp(-0.25 * SIM_SIZE, 1.25 * SIM_SIZE);
        }

        // Auto Difficulty Logic
        if !self.is_paused {
            let pop = self.alive_count as f32;
            let current_epoch = self.epoch;

            // Helper macro to avoid repetition
            macro_rules! adjust_param {
                ($param_struct:expr, $is_harder_increase:expr) => {
                    if $param_struct.enabled {
                        if current_epoch >= $param_struct.last_adjustment_epoch + $param_struct.cooldown_epochs {
                            let mut adjusted = false;
                            if pop > $param_struct.max_threshold {
                                // Make Harder (Population too high)
                                $param_struct.difficulty_level += 1;
                                adjusted = true;
                            } else if pop < $param_struct.min_threshold {
                                // Make Easier (Population too low)
                                $param_struct.difficulty_level -= 1;
                                adjusted = true;
                            }

                            if adjusted {
                                $param_struct.last_adjustment_epoch = current_epoch;
                            }
                        }
                    }
                };
            }

            adjust_param!(self.difficulty.food_power, false); // Harder = Decrease
            adjust_param!(self.difficulty.poison_power, true); // Harder = Increase
            adjust_param!(self.difficulty.spawn_prob, false); // Harder = Decrease
            adjust_param!(self.difficulty.death_prob, true); // Harder = Increase
            adjust_param!(self.difficulty.alpha_rain, false); // Harder = Decrease
            adjust_param!(self.difficulty.beta_rain, true); // Harder = Increase
        }

        // Advance RNG & auto-replenish only when not paused AND there are living agents
        let has_living_agents = self.alive_count > 0;
        if !self.is_paused && has_living_agents {
            self.rng_state ^= self.rng_state << 13;
            self.rng_state ^= self.rng_state >> 7;
            self.rng_state ^= self.rng_state << 17;

            if self.auto_replenish {
                self.replenish_population();
            }
        }

        let mut gamma_vis_min = self.gamma_vis_min;
        let mut gamma_vis_max = self.gamma_vis_max;
        if !gamma_vis_min.is_finite()
            || !gamma_vis_max.is_finite()
            || gamma_vis_min >= gamma_vis_max
        {
            gamma_vis_min = 0.0;
            gamma_vis_max = 1.0;
            self.gamma_vis_min = gamma_vis_min;
            self.gamma_vis_max = gamma_vis_max;
        }

        let capacity_left = self
            .agent_buffer_capacity
            .saturating_sub(self.agent_count as usize);
        let mut cpu_spawn_count = self.cpu_spawn_queue.len().min(2000).min(capacity_left) as u32;
        if cpu_spawn_count == 0
            && self.pending_spawn_upload
            && !self.cpu_spawn_queue.is_empty()
            && self.spawn_request_count > 0
        {
            cpu_spawn_count = self.spawn_request_count.min(2000).min(capacity_left as u32);
        }

        if !self.cpu_spawn_queue.is_empty() && cpu_spawn_count == 0 {
            println!("WARNING: Cannot spawn {} agents - no capacity left! (agent_count: {}, capacity: {}, capacity_left: {})",
                self.cpu_spawn_queue.len(), self.agent_count, self.agent_buffer_capacity, capacity_left);
        }

        // When paused or no living agents, freeze simulation side-effects.
        // Approach: don't dispatch simulation kernels at all when paused or no agents alive.
        let should_run_simulation = !self.is_paused && has_living_agents;

        // Calculate rain values with variation
        let time = self.epoch as f32;

        let alpha_var = self.alpha_rain_variation;
        let beta_var = self.beta_rain_variation;

        let alpha_freq = self.alpha_rain_freq / 1000.0;
        let alpha_phase = self.alpha_rain_phase;
        let alpha_sin = (time * alpha_freq * 2.0 * std::f32::consts::PI + alpha_phase).sin();

        let beta_freq = self.beta_rain_freq / 1000.0;
        let beta_phase = self.beta_rain_phase;
        let beta_sin = (time * beta_freq * 2.0 * std::f32::consts::PI + beta_phase).sin();

        // Apply difficulty adjustments to base values, then apply rain variation
        let base_alpha = self.difficulty.alpha_rain.apply_to(self.alpha_multiplier, false);
        let base_beta = self.difficulty.beta_rain.apply_to(self.beta_multiplier, true);

        let current_alpha = base_alpha * (1.0 + alpha_sin * alpha_var).max(0.0);
        let current_beta = base_beta * (1.0 + beta_sin * beta_var).max(0.0);

        // Update history
        if self.alpha_rain_history.len() >= 500 {
            self.alpha_rain_history.pop_front();
            self.beta_rain_history.pop_front();
        }
        self.alpha_rain_history.push_back(current_alpha);
        self.beta_rain_history.push_back(current_beta);

        let effective_dt = if should_run_simulation { 0.016 } else { 0.0 };
        let effective_spawn_p = if should_run_simulation {
            self.difficulty.spawn_prob.apply_to(self.spawn_probability, false)
        } else {
            0.0
        };
        let effective_death_p = if should_run_simulation {
            self.difficulty.death_prob.apply_to(self.death_probability, true)
        } else {
            0.0
        };

        // Apply difficulty adjustments to parameters
        let effective_food_power = self.difficulty.food_power.apply_to(self.food_power, false);
        let effective_poison_power = self.difficulty.poison_power.apply_to(self.poison_power, true);

        let params = SimParams {
            dt: effective_dt,
            frame_dt,
            drag: 0.1,
            energy_cost: 0.0, // Disabled energy depletion for now
            amino_maintenance_cost: self.amino_maintenance_cost,
            spawn_probability: effective_spawn_p,
            death_probability: effective_death_p,
            grid_size: SIM_SIZE,
            camera_zoom: self.camera_zoom,
            camera_pan_x: self.camera_pan[0],
            camera_pan_y: self.camera_pan[1],
            prev_camera_pan_x: self.prev_camera_pan[0],
            prev_camera_pan_y: self.prev_camera_pan[1],
            follow_mode: if self.follow_selected_agent { 1 } else { 0 },
            window_width: self.surface_config.width as f32,
            window_height: self.surface_config.height as f32,
            alpha_blur: self.alpha_blur,
            beta_blur: self.beta_blur,
            gamma_blur: self.gamma_blur,
            alpha_slope_bias: self.alpha_slope_bias,
            beta_slope_bias: self.beta_slope_bias,
            alpha_multiplier: current_alpha,
            beta_multiplier: current_beta,
            chemical_slope_scale_alpha: self.chemical_slope_scale_alpha,
            chemical_slope_scale_beta: self.chemical_slope_scale_beta,
            mutation_rate: self.mutation_rate,
            food_power: effective_food_power,
            poison_power: effective_poison_power,
            pairing_cost: self.pairing_cost,
            max_agents: self.agent_buffer_capacity as u32,
            cpu_spawn_count,
            agent_count: self.agent_count,
            random_seed: (self.rng_state >> 32) as u32,
            debug_mode: if self.rain_debug_visual {
                2
            } else if self.debug_per_segment {
                1
            } else {
                0
            },
            visual_stride: self.visual_stride_pixels,
            selected_agent_index: self
                .selected_agent_index
                .map(|i| i as u32)
                .unwrap_or(u32::MAX),
            repulsion_strength: self.repulsion_strength,
            agent_repulsion_strength: self.agent_repulsion_strength,
            gamma_strength: self.repulsion_strength * TERRAIN_FORCE_SCALE,
            prop_wash_strength: self.prop_wash_strength,
            gamma_vis_min,
            gamma_vis_max,
            // Allow drawing even when paused so camera movement & inspection are visible.
            // Simulation logic itself is gated elsewhere by is_paused.
            draw_enabled: if should_draw { 1 } else { 0 },
            gamma_debug: if self.gamma_debug_visual { 1 } else { 0 },
            gamma_hidden: if self.gamma_hidden { 1 } else { 0 },
            slope_debug: if self.slope_debug_visual { 1 } else { 0 },
            alpha_show: if self.alpha_show { 1 } else { 0 },
            beta_show: if self.beta_show { 1 } else { 0 },
            gamma_show: if self.gamma_show { 1 } else { 0 },
            slope_lighting: if self.slope_lighting { 1 } else { 0 },
            slope_lighting_strength: self.slope_lighting_strength,
            trail_diffusion: self.trail_diffusion,
            trail_decay: self.trail_decay,
            trail_opacity: self.trail_opacity,
            trail_show: if self.trail_show { 1 } else { 0 },
            interior_isotropic: if self.interior_isotropic { 1 } else { 0 },
            ignore_stop_codons: if self.ignore_stop_codons { 1 } else { 0 },
            require_start_codon: if self.require_start_codon { 1 } else { 0 },
            asexual_reproduction: if self.asexual_reproduction { 1 } else { 0 },
            background_color_r: self.background_color[0],
            background_color_g: self.background_color[1],
            background_color_b: self.background_color[2],
            alpha_blend_mode: self.alpha_blend_mode,
            beta_blend_mode: self.beta_blend_mode,
            gamma_blend_mode: self.gamma_blend_mode,
            slope_blend_mode: self.slope_blend_mode,
            alpha_color_r: self.alpha_color[0],
            alpha_color_g: self.alpha_color[1],
            alpha_color_b: self.alpha_color[2],
            beta_color_r: self.beta_color[0],
            beta_color_g: self.beta_color[1],
            beta_color_b: self.beta_color[2],
            gamma_color_r: self.gamma_color[0],
            gamma_color_g: self.gamma_color[1],
            gamma_color_b: self.gamma_color[2],
            grid_interpolation: self.grid_interpolation,
            alpha_gamma_adjust: self.alpha_gamma_adjust,
            beta_gamma_adjust: self.beta_gamma_adjust,
            gamma_gamma_adjust: self.gamma_gamma_adjust,
            light_dir_x: self.light_direction[0],
            light_dir_y: self.light_direction[1],
            light_power: self.light_power,
            light_dir_z: self.light_direction[2],
            agent_blend_mode: self.agent_blend_mode,
            agent_color_r: self.agent_color[0],
            agent_color_g: self.agent_color[1],
            agent_color_b: self.agent_color[2],
            agent_color_blend: self.agent_color_blend,
            epoch: self.epoch as u32,
            vector_force_power: self.vector_force_power,
            vector_force_x: self.vector_force_x,
            vector_force_y: self.vector_force_y,
            inspector_zoom: self.inspector_zoom,
            agent_trail_decay: self.agent_trail_decay,
            fluid_show: if self.fluid_show { 1 } else { 0 },
            _padding2: 0.0,
        };
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        // Update fluid params with actual frame dt
        {
            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct FluidParams {
                time: f32,
                dt: f32,
                decay: f32,
                grid_size: u32,
                mouse: [f32; 4],
                splat: [f32; 4],
            }

            let fluid_params = FluidParams {
                time: self.epoch as f32,
                dt: 0.016,
                decay: 0.995,
                grid_size: 128,
                mouse: [0.0; 4],
                splat: [0.0; 4],
            };

            self.queue.write_buffer(&self.fluid_params_buffer, 0, bytemuck::bytes_of(&fluid_params));
        }

        // Handle CPU spawns - only when not paused (consistent with autospawn)
        if !self.is_paused && cpu_spawn_count > 0 {
            // Use the reliable paused spawn path for all manual spawns
            println!("Using reliable spawn path for {} requests", cpu_spawn_count);
            self.process_spawn_requests_only(cpu_spawn_count, true);
        }

        let diffusion_interval = self.diffusion_interval.max(1);
        if self.diffusion_counter >= diffusion_interval {
            self.diffusion_counter = 0;
        }
        let slope_interval = self.slope_interval.max(1);
        if self.slope_counter >= slope_interval {
            self.slope_counter = 0;
        }
        let run_diffusion = self.diffusion_counter == 0 && should_run_simulation;
        let run_slope = self.slope_counter == 0 && should_run_simulation;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Update Encoder"),
            });

        // Clear counters (spawn_counter is reset in-shader at end of previous frame)
        if should_run_simulation {
            encoder.clear_buffer(&self.alive_counter, 0, None);
            encoder.clear_buffer(&self.spawn_debug_counters, 0, None);
        }
        // Do NOT clear spawn_counter here; we may set it from CPU spawns below.

        // Select bind groups based on ping-pong orientation
        // bg_process: agents_in -> agents_out
        // bg_swap: (swapped) agents_in/agents_out reversed for compaction and merge
        let (bg_process, bg_swap) = if self.ping_pong {
            (&self.compute_bind_group_b, &self.compute_bind_group_a)
        } else {
            (&self.compute_bind_group_a, &self.compute_bind_group_b)
        };

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });

            if run_diffusion {
                cpass.set_pipeline(&self.diffuse_pipeline);
                cpass.set_bind_group(0, bg_process, &[]);
                let groups_x = (GRID_DIM_U32 + DIFFUSE_WG_SIZE_X - 1) / DIFFUSE_WG_SIZE_X;
                let groups_y = (GRID_DIM_U32 + DIFFUSE_WG_SIZE_Y - 1) / DIFFUSE_WG_SIZE_Y;
                cpass.dispatch_workgroups(groups_x, groups_y, 1);

                // Diffuse RGB trail grids
                cpass.set_pipeline(&self.diffuse_trails_pipeline);
                cpass.set_bind_group(0, bg_process, &[]);
                cpass.dispatch_workgroups(groups_x, groups_y, 1);
            }

            if run_slope {
                cpass.set_pipeline(&self.gamma_slope_pipeline);
                cpass.set_bind_group(0, bg_process, &[]);
                let groups_x = (GRID_DIM_U32 + SLOPE_WG_SIZE_X - 1) / SLOPE_WG_SIZE_X;
                let groups_y = (GRID_DIM_U32 + SLOPE_WG_SIZE_Y - 1) / SLOPE_WG_SIZE_Y;
                cpass.dispatch_workgroups(groups_x, groups_y, 1);
            }

            // Clear visual grid only when drawing this step
            if should_draw {
                cpass.set_pipeline(&self.clear_visual_pipeline);
                cpass.set_bind_group(0, bg_process, &[]);
                let width_workgroups =
                    (self.surface_config.width + CLEAR_WG_SIZE_X - 1) / CLEAR_WG_SIZE_X;
                let height_workgroups =
                    (self.surface_config.height + CLEAR_WG_SIZE_Y - 1) / CLEAR_WG_SIZE_Y;
                cpass.dispatch_workgroups(width_workgroups, height_workgroups, 1);

                // Apply motion blur if following an agent
                cpass.set_pipeline(&self.motion_blur_pipeline);
                cpass.set_bind_group(0, bg_process, &[]);
                cpass.dispatch_workgroups(width_workgroups, height_workgroups, 1);

                // Clear agent grid for agent rendering
                cpass.set_pipeline(&self.clear_agent_grid_pipeline);
                cpass.set_bind_group(0, bg_process, &[]);
                cpass.dispatch_workgroups(width_workgroups, height_workgroups, 1);

                // Render inspector panel at reduced rate (~25fps to save GPU time)
                // Update every 2-3 frames depending on main loop speed
                let should_update_inspector = self.inspector_frame_counter % 2 == 0;
                self.inspector_frame_counter = self.inspector_frame_counter.wrapping_add(1);

                if should_update_inspector {
                    // Render inspector panel background if an agent is selected
                    cpass.set_pipeline(&self.render_inspector_pipeline);
                    cpass.set_bind_group(0, bg_process, &[]);
                    // Inspector is 300 pixels wide, dispatch appropriate workgroups
                    let inspector_width_workgroups = (300 + 15) / 16; // 16x16 workgroup size
                    let inspector_height_workgroups = (self.surface_config.height + 15) / 16;
                    cpass.dispatch_workgroups(inspector_width_workgroups, inspector_height_workgroups, 1);

                    // Draw inspector agent closeup in preview window
                    cpass.set_pipeline(&self.draw_inspector_agent_pipeline);
                    cpass.set_bind_group(0, bg_process, &[]);
                    // Preview window is 280x280, dispatch appropriate workgroups
                    let preview_workgroups = (280 + 15) / 16;
                    cpass.dispatch_workgroups(preview_workgroups, preview_workgroups, 1);
                }
            }

            // Run simulation compute passes, but skip everything when paused or no living agents
            if should_run_simulation {
                // Clear agent spatial grid for neighbor detection - workgroup_size(16,16)
                let spatial_grid_workgroups = (SPATIAL_GRID_DIM as u32 + 15) / 16;
                cpass.set_pipeline(&self.clear_agent_spatial_grid_pipeline);
                cpass.set_bind_group(0, bg_process, &[]);
                cpass.dispatch_workgroups(spatial_grid_workgroups, spatial_grid_workgroups, 1);

                // Populate agent spatial grid with agent positions - workgroup_size(256)
                cpass.set_pipeline(&self.populate_agent_spatial_grid_pipeline);
                cpass.set_bind_group(0, bg_process, &[]);
                cpass.dispatch_workgroups((self.agent_count + 255) / 256, 1, 1);

                // Drain energy from neighbors (vampire mouths) - workgroup_size(256)
                // Must run BEFORE process_agents so agents see the drained energy
                cpass.set_pipeline(&self.drain_energy_pipeline);
                cpass.set_bind_group(0, bg_process, &[]);
                cpass.dispatch_workgroups((self.agent_count + 255) / 256, 1, 1);

                // Process all agents (sense, update, modify env, draw, spawn/death) - workgroup_size(256)
                // Agents will inject propeller forces directly into fluid forces buffer with 100x boost
                cpass.set_pipeline(&self.process_pipeline);
                cpass.set_bind_group(0, bg_process, &[]);
                cpass.dispatch_workgroups((self.agent_count + 255) / 256, 1, 1);

                // Run fluid solver - forces already written by agents
                // Stable Fluids order: add_forces → diffuse → advect → project
                {
                    const FLUID_GRID_SIZE: u32 = 128;
                    let fluid_workgroups = (FLUID_GRID_SIZE + 15) / 16;
                    let bg_ab = &self.fluid_bind_group_ab;
                    let bg_ba = &self.fluid_bind_group_ba;

                    // 1. Add forces to velocity (A -> B)
                    // Forces already written directly by agents with 100x boost
                    cpass.set_pipeline(&self.fluid_add_forces_pipeline);
                    cpass.set_bind_group(0, bg_ab, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    // 2. Diffuse velocity (B -> A)
                    cpass.set_pipeline(&self.fluid_diffuse_velocity_pipeline);
                    cpass.set_bind_group(0, bg_ba, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    // 3. Advect velocity (A -> B)
                    cpass.set_pipeline(&self.fluid_advect_velocity_pipeline);
                    cpass.set_bind_group(0, bg_ab, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    // 4. Vorticity confinement (B -> A)
                    cpass.set_pipeline(&self.fluid_vorticity_confinement_pipeline);
                    cpass.set_bind_group(0, bg_ba, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    // 5. Compute divergence (reads velocity_a)
                    cpass.set_pipeline(&self.fluid_divergence_pipeline);
                    cpass.set_bind_group(0, bg_ab, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    // 6. Clear pressure buffers
                    cpass.set_pipeline(&self.fluid_clear_pressure_pipeline);
                    cpass.set_bind_group(0, bg_ba, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    cpass.set_pipeline(&self.fluid_clear_pressure_pipeline);
                    cpass.set_bind_group(0, bg_ab, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    // 7. Jacobi iterations for pressure (31 iterations)
                    const JACOBI_ITERS: u32 = 31;
                    for i in 0..JACOBI_ITERS {
                        let bg = if (i & 1) == 0 { bg_ab } else { bg_ba };
                        cpass.set_pipeline(&self.fluid_jacobi_pressure_pipeline);
                        cpass.set_bind_group(0, bg, &[]);
                        cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);
                    }

                    // 8. Subtract pressure gradient (A->B)
                    cpass.set_pipeline(&self.fluid_subtract_gradient_pipeline);
                    cpass.set_bind_group(0, bg_ab, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    // 9. Enforce boundaries (B->A, final result in velocity_a)
                    cpass.set_pipeline(&self.fluid_enforce_boundaries_pipeline);
                    cpass.set_bind_group(0, bg_ba, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);
                }

                // Compact alive agents - workgroup_size(64)
                cpass.set_pipeline(&self.compact_pipeline);
                cpass.set_bind_group(0, bg_swap, &[]);
                cpass.dispatch_workgroups((self.agent_count + 63) / 64, 1, 1);

                // Process CPU spawn requests - workgroup_size(64)
                if cpu_spawn_count > 0 {
                    cpass.set_pipeline(&self.cpu_spawn_pipeline);
                    cpass.set_bind_group(0, bg_swap, &[]);
                    cpass.dispatch_workgroups((cpu_spawn_count + 63) / 64, 1, 1);
                }

                // Merge spawned agents - workgroup_size(64), max 2000 spawns
                cpass.set_pipeline(&self.merge_pipeline);
                cpass.set_bind_group(0, bg_swap, &[]);
                cpass.dispatch_workgroups((2000 + 63) / 64, 1, 1);

                // Sanitize unused agent slots - workgroup_size(256)
                let init_groups = ((self.agent_buffer_capacity as u32) + 255) / 256;
                cpass.set_pipeline(&self.initialize_dead_pipeline);
                cpass.set_bind_group(0, bg_swap, &[]);
                cpass.dispatch_workgroups(init_groups, 1, 1);

                // Reset spawn counter - workgroup_size(1)
                cpass.set_pipeline(&self.finalize_merge_pipeline);
                cpass.set_bind_group(0, bg_swap, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            } else if should_draw && self.agent_count > 0 {
                // When paused or no living agents: run process pipeline for rendering only (dt=0, probabilities=0)
                // but skip compaction, spawning, and buffer swapping
                {
                    const FLUID_GRID_SIZE: u32 = 128;
                    let fluid_workgroups = (FLUID_GRID_SIZE + 15) / 16;
                    cpass.set_pipeline(&self.fluid_clear_forces_pipeline);
                    cpass.set_bind_group(0, &self.fluid_bind_group_ab, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);
                }

                cpass.set_pipeline(&self.process_pipeline);
                cpass.set_bind_group(0, bg_process, &[]);
                cpass.dispatch_workgroups((self.agent_count + 255) / 256, 1, 1);

                // Keep the fluid field evolving even in render-only mode
                {
                    const FLUID_GRID_SIZE: u32 = 128;
                    let fluid_workgroups = (FLUID_GRID_SIZE + 15) / 16;
                    let bg_ab = &self.fluid_bind_group_ab;
                    let bg_ba = &self.fluid_bind_group_ba;

                    cpass.set_pipeline(&self.fluid_advect_velocity_pipeline);
                    cpass.set_bind_group(0, bg_ab, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    cpass.set_pipeline(&self.fluid_add_forces_pipeline);
                    cpass.set_bind_group(0, bg_ba, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    cpass.set_pipeline(&self.fluid_vorticity_confinement_pipeline);
                    cpass.set_bind_group(0, bg_ab, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    cpass.set_pipeline(&self.fluid_enforce_boundaries_pipeline);
                    cpass.set_bind_group(0, bg_ba, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    cpass.set_pipeline(&self.fluid_diffuse_velocity_pipeline);
                    cpass.set_bind_group(0, bg_ab, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    cpass.set_pipeline(&self.fluid_copy_pipeline);
                    cpass.set_bind_group(0, bg_ba, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    cpass.set_pipeline(&self.fluid_divergence_pipeline);
                    cpass.set_bind_group(0, bg_ab, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    cpass.set_pipeline(&self.fluid_clear_pressure_pipeline);
                    cpass.set_bind_group(0, bg_ba, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    cpass.set_pipeline(&self.fluid_clear_pressure_pipeline);
                    cpass.set_bind_group(0, bg_ab, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    const JACOBI_ITERS: u32 = 31;
                    for i in 0..JACOBI_ITERS {
                        let bg = if (i & 1) == 0 { bg_ab } else { bg_ba };
                        cpass.set_pipeline(&self.fluid_jacobi_pressure_pipeline);
                        cpass.set_bind_group(0, bg, &[]);
                        cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);
                    }

                    cpass.set_pipeline(&self.fluid_subtract_gradient_pipeline);
                    cpass.set_bind_group(0, bg_ab, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    cpass.set_pipeline(&self.fluid_enforce_boundaries_pipeline);
                    cpass.set_bind_group(0, bg_ba, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    cpass.set_pipeline(&self.fluid_copy_velocity_to_forces_pipeline);
                    cpass.set_bind_group(0, bg_ab, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);
                }
            }

            // Composite agents onto visual grid when drawing
            if should_draw {
                // Render all agents to agent_grid
                cpass.set_pipeline(&self.render_agents_pipeline);
                cpass.set_bind_group(0, bg_process, &[]);
                cpass.dispatch_workgroups((self.agent_count + 255) / 256, 1, 1);

                // Composite agent_grid onto visual_grid
                cpass.set_pipeline(&self.composite_agents_pipeline);
                cpass.set_bind_group(0, bg_process, &[]);
                let width_workgroups =
                    (self.surface_config.width + CLEAR_WG_SIZE_X - 1) / CLEAR_WG_SIZE_X;
                let height_workgroups =
                    (self.surface_config.height + CLEAR_WG_SIZE_Y - 1) / CLEAR_WG_SIZE_Y;
                cpass.dispatch_workgroups(width_workgroups, height_workgroups, 1);

                // Inspector panel is now drawn by the selected agent during simulation
            }
        }

        // Copy visual grid buffer to texture only when drawing this step
        if should_draw {
            let bytes_per_pixel: u32 = 16;
            let stride_bytes = self.visual_stride_pixels * bytes_per_pixel;
            encoder.copy_buffer_to_texture(
                wgpu::ImageCopyBuffer {
                    buffer: &self.visual_grid_buffer,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(stride_bytes),
                        rows_per_image: Some(self.surface_config.height),
                    },
                },
                wgpu::ImageCopyTexture {
                    texture: &self.visual_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: self.surface_config.width,
                    height: self.surface_config.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        self.process_completed_alive_readbacks();
        let slot = self.ensure_alive_slot_ready();
        let alive_buffer = self.copy_state_to_staging(&mut encoder, slot);

        self.queue.submit(Some(encoder.finish()));

        // Debug: check if spawn actually increased the count (readback may not be ready yet)
        if cpu_spawn_count > 0 && !self.is_paused {
            // Poll for readback completion
            self.device.poll(wgpu::Maintain::Poll);
            self.process_completed_alive_readbacks();
            println!("  -> After spawn: {} agents alive", self.alive_count);

            // Clear the processed spawn requests from the queue
            let drain_count = (cpu_spawn_count as usize).min(self.cpu_spawn_queue.len());
            self.cpu_spawn_queue.drain(0..drain_count);
            self.spawn_request_count = self.cpu_spawn_queue.len() as u32;
            if self.cpu_spawn_queue.is_empty() {
                self.pending_spawn_upload = false;
            }
        }

        // Only update counters when simulation is running
        if should_run_simulation {
            self.diffusion_counter = (self.diffusion_counter + 1) % diffusion_interval;
            self.slope_counter = (self.slope_counter + 1) % slope_interval;
        }

        self.kickoff_alive_readback(slot, alive_buffer);

        self.perform_optional_readbacks(should_draw);

        // Keep ping-pong orientation stable: after process (in->out),
        // compaction wrote results back to the original input buffer.
        // Therefore, we do NOT toggle ping-pong here.
    }

    fn render(
        &mut self,
        clipped_primitives: Vec<egui::ClippedPrimitive>,
        textures_delta: egui::TexturesDelta,
        screen_descriptor: ScreenDescriptor,
    ) -> Result<(), wgpu::SurfaceError> {
        // Update FPS counter
        self.frame_count += 1;
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last_fps_update).as_secs_f32();
        if elapsed >= 1.0 {
            let fps = self.frame_count as f32 / elapsed;
            self.window
                .set_title(&format!("Artificial Life Simulator - {:.1} FPS", fps));
            self.frame_count = 0;
            self.last_fps_update = now;
        }

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // Render simulation
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.render_bind_group, &[]);
            rpass.draw(0..6, 0..1);
        }

        // Submit simulation rendering
        self.queue.submit(Some(encoder.finish()));

        // --- Main simulation and rendering path (if not paused) ---

        // Render egui in a separate encoder
        for (id, image_delta) in &textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, image_delta);
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("egui Encoder"),
            });

        // Update buffers
        for (id, image_delta) in &textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, image_delta);
        }

        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &clipped_primitives,
            &screen_descriptor,
        );

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // SAFETY: The render pass lives long enough for this call.
            // The lifetime requirement is overly restrictive in egui-wgpu 0.29.
            let rpass_static: &mut wgpu::RenderPass<'static> =
                unsafe { std::mem::transmute(&mut rpass) };
            self.egui_renderer
                .render(rpass_static, &clipped_primitives, &screen_descriptor);
        }

        for id in &textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn render_ui_only(
        &mut self,
        clipped_primitives: Vec<egui::ClippedPrimitive>,
        textures_delta: egui::TexturesDelta,
        screen_descriptor: ScreenDescriptor,
    ) -> Result<(), wgpu::SurfaceError> {
        // Update FPS counter
        self.frame_count += 1;
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last_fps_update).as_secs_f32();
        if elapsed >= 1.0 {
            let fps = self.frame_count as f32 / elapsed;
            let speed = match self.current_mode {
                2 => " (Fast Mode)".to_string(),
                3 => " (25 FPS)".to_string(),
                _ => String::new(),
            };
            self.window.set_title(&format!(
                "Artificial Life Simulator{} - {:.1} FPS",
                speed, fps
            ));
            self.frame_count = 0;
            self.last_fps_update = now;
        }

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Render egui only (no simulation rendering)
        for (id, image_delta) in &textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, image_delta);
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("egui Encoder"),
            });

        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &clipped_primitives,
            &screen_descriptor,
        );

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Load existing frame instead of clearing
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // SAFETY: The render pass lives long enough for this call.
            let rpass_static: &mut wgpu::RenderPass<'static> =
                unsafe { std::mem::transmute(&mut rpass) };
            self.egui_renderer
                .render(rpass_static, &clipped_primitives, &screen_descriptor);
        }

        for id in &textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    fn read_counter(&self, buffer: &wgpu::Buffer) -> u32 {
        let mut result = [0u32; 1];
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |result| {
            result.unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        {
            let view = slice.get_mapped_range();
            result.copy_from_slice(bytemuck::cast_slice(&view));
        }
        buffer.unmap();
        result[0]
    }

    fn mutate_genome(&mut self, genome: &[u32; GENOME_WORDS]) -> [u32; GENOME_WORDS] {
        let mut new_genome = *genome;
        // Mutate 1-3 random bits using PCG (better distribution)
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005u64)
            .wrapping_add(1442695040888963407u64);
        let mutation_count = ((self.rng_state >> 32) % 3 + 1) as usize;

        for _ in 0..mutation_count {
            self.rng_state = self
                .rng_state
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            let byte_idx = ((self.rng_state >> 32) as usize) % GENOME_BYTES;
            let word = byte_idx / 4;
            let rand_bits = (self.rng_state >> 32) as u32;
            let bit = (rand_bits % 8) + ((byte_idx % 4) as u32 * 8);

            if word < GENOME_WORDS {
                new_genome[word] ^= 1 << bit;
            }
        }
        new_genome
    }

    fn select_agent_at_screen_pos(&mut self, screen_pos: [f32; 2]) {
        // Convert screen coordinates to world coordinates
        // Screen coords to normalized coords (0..1)
        let norm_x = screen_pos[0] / self.surface_config.width as f32;
        let norm_y = screen_pos[1] / self.surface_config.height as f32;

        // Account for aspect ratio when projecting into world space
        let aspect = if self.surface_config.width > 0 {
            self.surface_config.height as f32 / self.surface_config.width as f32
        } else {
            1.0
        };
        let half_view_x = SIM_SIZE / (2.0 * self.camera_zoom);
        let half_view_y = half_view_x * aspect;

        let mut world_x = self.camera_pan[0] + (norm_x - 0.5) * 2.0 * half_view_x;
        let mut world_y = self.camera_pan[1] - (norm_y - 0.5) * 2.0 * half_view_y; // Y inverted

        // Wrap to world bounds
        world_x = world_x.rem_euclid(SIM_SIZE);
        world_y = world_y.rem_euclid(SIM_SIZE);

        // Read back agents from GPU
        let current_buffer = if self.ping_pong {
            &self.agents_buffer_b
        } else {
            &self.agents_buffer_a
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Agent Selection Encoder"),
            });

        encoder.copy_buffer_to_buffer(
            current_buffer,
            0,
            &self.agents_readback,
            0,
            (std::mem::size_of::<Agent>() * self.agent_buffer_capacity) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        // Read back and find nearest agent
        let slice = self.agents_readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        {
            let data = slice.get_mapped_range();
            let agents: &[Agent] = bytemuck::cast_slice(&data);

            let mut nearest_idx: Option<usize> = None;
            let mut nearest_dist = f32::MAX;

            let wrap_delta = |delta: f32| {
                let mut d = delta;
                if d > SIM_SIZE * 0.5 {
                    d -= SIM_SIZE;
                } else if d < -SIM_SIZE * 0.5 {
                    d += SIM_SIZE;
                }
                d
            };

            const PICK_RADIUS: f32 = 400.0; // Ignore agents beyond this radius when selecting

            for (i, agent) in agents.iter().enumerate().take(self.agent_count as usize) {
                if agent.alive == 0 {
                    continue;
                }

                // Wrap-aware distance to account for toroidal world
                let dx = wrap_delta(agent.position[0] - world_x);
                let dy = wrap_delta(agent.position[1] - world_y);
                let dist = (dx * dx + dy * dy).sqrt();

                if dist < nearest_dist && dist <= PICK_RADIUS {
                    nearest_dist = dist;
                    nearest_idx = Some(i);
                }
            }

            // Copy agents to a mutable buffer
            let mut agents_vec: Vec<Agent> = agents.iter().copied().collect();

            // Clear all selection flags first, then set the nearest one
            let max_len = agents_vec.len();
            let take_len = (self.agent_count as usize).min(max_len);
            for agent in agents_vec.iter_mut().take(take_len) {
                agent.is_selected = 0;
            }

            // Set the selected flag on nearest agent
            if let Some(idx) = nearest_idx {
                agents_vec[idx].is_selected = 1;
                self.selected_agent_index = Some(idx);
                self.selected_agent_data = Some(agents_vec[idx]);
            } else {
                self.selected_agent_index = None;
                self.selected_agent_data = None;
            }

            drop(data);
            self.agents_readback.unmap();

            // Update GPU buffer with new selection flags
            let current_buffer = if self.ping_pong {
                &self.agents_buffer_b
            } else {
                &self.agents_buffer_a
            };
            self.queue.write_buffer(
                current_buffer,
                0,
                bytemuck::cast_slice(&agents_vec[..take_len]),
            );
        }
    }

    fn save_snapshot_to_file(&mut self, path: &Path) -> anyhow::Result<()> {
        // Copy grids and agents from GPU to readback buffers
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Snapshot Readback Encoder"),
        });

        let grid_size_bytes = (GRID_CELL_COUNT * std::mem::size_of::<f32>()) as u64;

        encoder.copy_buffer_to_buffer(&self.alpha_grid, 0, &self.alpha_grid_readback, 0, grid_size_bytes);
        encoder.copy_buffer_to_buffer(&self.beta_grid, 0, &self.beta_grid_readback, 0, grid_size_bytes);
        encoder.copy_buffer_to_buffer(&self.gamma_grid, 0, &self.gamma_grid_readback, 0, grid_size_bytes);

        // Copy agents from GPU (use current buffer based on ping-pong)
        let agents_size_bytes = (std::mem::size_of::<Agent>() * self.agent_buffer_capacity) as u64;
        let source_buffer = if self.ping_pong { &self.agents_buffer_b } else { &self.agents_buffer_a };
        encoder.copy_buffer_to_buffer(source_buffer, 0, &self.agents_readback, 0, agents_size_bytes);

        self.queue.submit(Some(encoder.finish()));

        // Map and read grids
        let alpha_slice = self.alpha_grid_readback.slice(..);
        let beta_slice = self.beta_grid_readback.slice(..);
        let gamma_slice = self.gamma_grid_readback.slice(..);
        let agents_slice = self.agents_readback.slice(..);

        alpha_slice.map_async(wgpu::MapMode::Read, |_| {});
        beta_slice.map_async(wgpu::MapMode::Read, |_| {});
        gamma_slice.map_async(wgpu::MapMode::Read, |_| {});
        agents_slice.map_async(wgpu::MapMode::Read, |_| {});

        self.device.poll(wgpu::Maintain::Wait);

        let (alpha_grid, beta_grid, gamma_grid, agents_gpu) = {
            let alpha_data = alpha_slice.get_mapped_range();
            let beta_data = beta_slice.get_mapped_range();
            let gamma_data = gamma_slice.get_mapped_range();
            let agents_data = agents_slice.get_mapped_range();

            let alpha: Vec<f32> = bytemuck::cast_slice(&alpha_data).to_vec();
            let beta: Vec<f32> = bytemuck::cast_slice(&beta_data).to_vec();
            let gamma: Vec<f32> = bytemuck::cast_slice(&gamma_data).to_vec();
            let agents: Vec<Agent> = bytemuck::cast_slice(&agents_data).to_vec();

            (alpha, beta, gamma, agents)
        };

        self.alpha_grid_readback.unmap();
        self.beta_grid_readback.unmap();
        self.gamma_grid_readback.unmap();
        self.agents_readback.unmap();

        // Filter only living agents before creating snapshot
        let living_agents: Vec<Agent> = agents_gpu
            .into_iter()
            .filter(|a| a.alive != 0)
            .collect();

        println!("Saving snapshot with {} living agents", living_agents.len());

        // Create snapshot from living agents with current settings
        let current_settings = self.current_settings();
        let snapshot = SimulationSnapshot::new(self.epoch, &living_agents, current_settings);

        // Save to PNG
        save_simulation_snapshot(path, &alpha_grid, &beta_grid, &gamma_grid, &snapshot)?;

        Ok(())
    }

    fn load_snapshot_from_file(&mut self, path: &Path) -> anyhow::Result<()> {
        // Reset simulation state before loading to prevent crashes on subsequent loads
        // Clear alive counter
        self.queue.write_buffer(&self.alive_counter, 0, bytemuck::bytes_of(&0u32));

        // Clear spawn counter
        self.queue.write_buffer(&self.spawn_debug_counters, 0, bytemuck::bytes_of(&0u32));

        // Zero out all agent buffers
        let zero_agents = vec![Agent::zeroed(); self.agent_buffer_capacity];
        self.queue.write_buffer(&self.agents_buffer_a, 0, bytemuck::cast_slice(&zero_agents));
        self.queue.write_buffer(&self.agents_buffer_b, 0, bytemuck::cast_slice(&zero_agents));

        // Clear CPU-side state
        self.agents_cpu.clear();
        self.cpu_spawn_queue.clear();
        self.agent_count = 0;
        self.alive_count = 0;
        self.spawn_request_count = 0;

        // Load snapshot from PNG file
        let (alpha_grid, beta_grid, gamma_grid, snapshot) = load_simulation_snapshot(path)?;

        // Apply loaded settings only if they exist in the snapshot (backwards compatibility)
        if let Some(settings) = &snapshot.settings {
            self.camera_zoom = settings.camera_zoom;
            self.spawn_probability = settings.spawn_probability;
            self.death_probability = settings.death_probability;
            self.mutation_rate = settings.mutation_rate;
            self.auto_replenish = settings.auto_replenish;
            self.diffusion_interval = settings.diffusion_interval;
            self.slope_interval = settings.slope_interval;
            self.alpha_blur = settings.alpha_blur;
            self.beta_blur = settings.beta_blur;
            self.gamma_blur = settings.gamma_blur;
            self.alpha_slope_bias = settings.alpha_slope_bias;
            self.beta_slope_bias = settings.beta_slope_bias;
            self.alpha_multiplier = settings.alpha_multiplier;
            self.beta_multiplier = settings.beta_multiplier;
            self.alpha_rain_map_path = settings.alpha_rain_map_path.clone();
            self.beta_rain_map_path = settings.beta_rain_map_path.clone();
            self.chemical_slope_scale_alpha = settings.chemical_slope_scale_alpha;
            self.chemical_slope_scale_beta = settings.chemical_slope_scale_beta;
            self.alpha_noise_scale = settings.alpha_noise_scale;
            self.beta_noise_scale = settings.beta_noise_scale;
            self.gamma_noise_scale = settings.gamma_noise_scale;
            self.noise_power = settings.noise_power;
            self.food_power = settings.food_power;
            self.poison_power = settings.poison_power;
            self.amino_maintenance_cost = settings.amino_maintenance_cost;
            self.pairing_cost = settings.pairing_cost;
            self.prop_wash_strength = settings.prop_wash_strength;
            self.repulsion_strength = settings.repulsion_strength;
            self.agent_repulsion_strength = settings.agent_repulsion_strength;
            self.limit_fps = settings.limit_fps;
            self.limit_fps_25 = settings.limit_fps_25;
            self.render_interval = settings.render_interval;
            self.gamma_debug_visual = settings.gamma_debug_visual;
            self.slope_debug_visual = settings.slope_debug_visual;
            self.gamma_hidden = settings.gamma_hidden;
            self.debug_per_segment = settings.debug_per_segment;
            self.gamma_vis_min = settings.gamma_vis_min;
            self.gamma_vis_max = settings.gamma_vis_max;
            self.alpha_show = settings.alpha_show;
            self.beta_show = settings.beta_show;
            self.gamma_show = settings.gamma_show;
            self.slope_lighting = settings.slope_lighting;
            self.slope_lighting_strength = settings.slope_lighting_strength;
            self.trail_diffusion = settings.trail_diffusion;
            self.trail_decay = settings.trail_decay;
            self.trail_opacity = settings.trail_opacity;
            self.trail_show = settings.trail_show;
            self.interior_isotropic = settings.interior_isotropic;
            self.ignore_stop_codons = settings.ignore_stop_codons;
            self.require_start_codon = settings.require_start_codon;
            self.alpha_rain_variation = settings.alpha_rain_variation;
            self.beta_rain_variation = settings.beta_rain_variation;
            self.alpha_rain_phase = settings.alpha_rain_phase;
            self.beta_rain_phase = settings.beta_rain_phase;
            self.alpha_rain_freq = settings.alpha_rain_freq;
            self.beta_rain_freq = settings.beta_rain_freq;
            self.difficulty = settings.difficulty.clone();
            self.background_color = settings.background_color;
            self.alpha_blend_mode = settings.alpha_blend_mode;
            self.beta_blend_mode = settings.beta_blend_mode;
            self.gamma_blend_mode = settings.gamma_blend_mode;
            self.slope_blend_mode = settings.slope_blend_mode;
            self.alpha_color = settings.alpha_color;
            self.beta_color = settings.beta_color;
            self.gamma_color = settings.gamma_color;
            self.grid_interpolation = settings.grid_interpolation;
            self.alpha_gamma_adjust = settings.alpha_gamma_adjust;
            self.beta_gamma_adjust = settings.beta_gamma_adjust;
            self.gamma_gamma_adjust = settings.gamma_gamma_adjust;
            self.light_direction = settings.light_direction;
            self.agent_blend_mode = settings.agent_blend_mode;
            self.agent_color = settings.agent_color;
            self.agent_color_blend = settings.agent_color_blend;

            if let Some(path) = &settings.alpha_rain_map_path {
                 let _ = self.load_alpha_rain_map(path);
            }
            if let Some(path) = &settings.beta_rain_map_path {
                 let _ = self.load_beta_rain_map(path);
            }
        } else {
            println!("ΓÜá Loaded snapshot without settings (old format) - using current settings");
        }

        // Upload grids to GPU
        self.queue.write_buffer(&self.alpha_grid, 0, bytemuck::cast_slice(&alpha_grid));
        self.queue.write_buffer(&self.beta_grid, 0, bytemuck::cast_slice(&beta_grid));
        self.queue.write_buffer(&self.gamma_grid, 0, bytemuck::cast_slice(&gamma_grid));

        // Queue agents from snapshot for spawning
        for agent_snap in snapshot.agents.iter() {
            let spawn_req = agent_snap.to_spawn_request();
            self.cpu_spawn_queue.push(spawn_req);
        }

        // Update spawn request count so they get processed
        self.spawn_request_count = self.cpu_spawn_queue.len() as u32;

        // Update epoch from snapshot
        self.epoch = snapshot.epoch;

        // Immediately save to autosave to ensure continuity
        let autosave_path = std::path::Path::new(AUTO_SNAPSHOT_FILE_NAME);
        if let Err(e) = self.save_snapshot_to_file(autosave_path) {
            eprintln!("ΓÜá Failed to update autosave after loading snapshot: {:?}", e);
        }

        println!("Γ£ô Loaded settings and queued {} agents from snapshot", self.cpu_spawn_queue.len());

        Ok(())
    }

    fn spawn_agent(&mut self, parent_agent: &Agent) {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005u64)
            .wrapping_add(1442695040888963407u64);

        let mut new_agent = Agent::zeroed();

        // Mutate genome
        new_agent.genome = self.mutate_genome(&parent_agent.genome);

        // Spawn near parent with random offset
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005u64)
            .wrapping_add(1442695040888963407u64);
        let offset_angle =
            ((self.rng_state >> 32) as f32 / u32::MAX as f32) * std::f32::consts::TAU;
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005u64)
            .wrapping_add(1442695040888963407u64);
        let offset_dist = 20.0 + ((self.rng_state >> 32) % 30) as f32;

        new_agent.position = [
            parent_agent.position[0] + offset_angle.cos() * offset_dist,
            parent_agent.position[1] + offset_angle.sin() * offset_dist,
        ];

        // Wrap to world bounds
        new_agent.position[0] = new_agent.position[0].rem_euclid(SIM_SIZE);
        new_agent.position[1] = new_agent.position[1].rem_euclid(SIM_SIZE);

        new_agent.velocity = [0.0, 0.0];
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005u64)
            .wrapping_add(1442695040888963407u64);
        new_agent.rotation =
            ((self.rng_state >> 32) as f32 / u32::MAX as f32) * std::f32::consts::TAU;
        new_agent.energy = 50.0;
        new_agent.alive = 1;
        new_agent.body_count = 0; // GPU will build body

        self.agents_cpu.push(new_agent);
        self.agent_count = self.agents_cpu.len() as u32;

        // Check if we need to reallocate buffers
        if self.agents_cpu.len() > self.agent_buffer_capacity {
            self.reallocate_agent_buffers();
        }
    }

    fn reallocate_agent_buffers(&mut self) {
        // Double capacity
        let new_capacity = self.agent_buffer_capacity * 2;
        println!(
            "Reallocating agent buffers: {} -> {}",
            self.agent_buffer_capacity, new_capacity
        );

        // Pad agents_cpu to new capacity
        self.agents_cpu.resize(new_capacity, Agent::zeroed());

        // Create new buffers
        let agents_buffer_a = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Agents Buffer A (Reallocated)"),
                contents: bytemuck::cast_slice(&self.agents_cpu),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        let agents_buffer_b = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Agents Buffer B (Reallocated)"),
                contents: bytemuck::cast_slice(&self.agents_cpu),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        // Replace old buffers
        self.agents_buffer_a = agents_buffer_a;
        self.agents_buffer_b = agents_buffer_b;
        self.agent_buffer_capacity = new_capacity;

        // Shrink agents_cpu back to actual count
        self.agents_cpu.truncate(self.agent_count as usize);

        // Recreate bind groups
        self.recreate_compute_bind_groups();
    }

    fn recreate_compute_bind_groups(&mut self) {
        // Get bind group layout from pipeline
        let bind_group_layout = self.process_pipeline.get_bind_group_layout(0);

        // Recreate bind group A
        self.compute_bind_group_a = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group A"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.agents_buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.agents_buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.alpha_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.beta_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.visual_grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.agent_grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.alive_counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.fluid_velocity_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.new_agents_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: self.spawn_debug_counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.spawn_requests_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: self.selected_agent_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: self.gamma_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: self.trail_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: self.environment_init_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: self.fluid_force_vectors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: self.agent_spatial_grid_buffer.as_entire_binding(),
                },
            ],
        });

        // Recreate bind group B (swapped)
        self.compute_bind_group_b = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group B"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.agents_buffer_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.agents_buffer_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.alpha_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.beta_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.visual_grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.agent_grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.alive_counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.fluid_velocity_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.new_agents_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: self.spawn_debug_counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.spawn_requests_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: self.selected_agent_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: self.gamma_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: self.trail_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: self.environment_init_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: self.fluid_force_vectors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: self.agent_spatial_grid_buffer.as_entire_binding(),
                },
            ],
        });
    }
}

impl Drop for GpuState {
    fn drop(&mut self) {
        // Don't call destroy_resources() - let GPU driver handle cleanup asynchronously
        // The slow device.poll(Wait) in destroy_resources takes 50+ seconds
        // Just let Rust drop all the buffer/texture handles naturally
    }
}

impl GpuState {
    fn fast_reset(&mut self) {
        // Reset all simulation state without recreating GPU resources

        // Reset agent count
        self.agent_count = 0;
        self.alive_count = 0;
        self.agents_cpu.clear();
        self.cpu_spawn_queue.clear();
        self.spawn_request_count = 0;
        self.pending_spawn_upload = false;

        // Reset epoch and timing
        self.epoch = 0;
        self.last_sample_epoch = 0;
        self.last_autosave_epoch = 0;
        self.last_epoch_count = 0;
        self.last_epoch_update = std::time::Instant::now();

        // Reset camera
        self.camera_zoom = 1.0;
        self.camera_pan = [SIM_SIZE / 2.0, SIM_SIZE / 2.0];
        self.prev_camera_pan = [SIM_SIZE / 2.0, SIM_SIZE / 2.0];
        self.camera_target = [SIM_SIZE / 2.0, SIM_SIZE / 2.0];
        self.camera_velocity = [0.0, 0.0];

        // Reset selection
        self.selected_agent_index = None;
        self.selected_agent_data = None;
        self.follow_selected_agent = false;

        // Reset statistics
        self.population_history.clear();
        self.alpha_rain_history.clear();
        self.beta_rain_history.clear();

        // Reset difficulty levels
        self.difficulty.food_power.difficulty_level = 0;
        self.difficulty.food_power.last_adjustment_epoch = 0;
        self.difficulty.poison_power.difficulty_level = 0;
        self.difficulty.poison_power.last_adjustment_epoch = 0;
        self.difficulty.spawn_prob.difficulty_level = 0;
        self.difficulty.spawn_prob.last_adjustment_epoch = 0;
        self.difficulty.death_prob.difficulty_level = 0;
        self.difficulty.death_prob.last_adjustment_epoch = 0;
        self.difficulty.alpha_rain.difficulty_level = 0;
        self.difficulty.alpha_rain.last_adjustment_epoch = 0;
        self.difficulty.beta_rain.difficulty_level = 0;
        self.difficulty.beta_rain.last_adjustment_epoch = 0;

        // Update RNG seed
        use std::time::{SystemTime, UNIX_EPOCH};
        self.rng_state = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Clear and reinitialize GPU buffers using compute shaders
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Reset Encoder"),
        });

        // Clear alive counter and spawn counter
        encoder.clear_buffer(&self.alive_counter, 0, None);
        encoder.clear_buffer(&self.spawn_debug_counters, 0, None);

        // Reinitialize environment grids (alpha, beta, gamma, trails) via GPU compute
        const CLEAR_WG_SIZE_X: u32 = 16;
        const CLEAR_WG_SIZE_Y: u32 = 16;
        let env_groups_x = (GRID_DIM_U32 + CLEAR_WG_SIZE_X - 1) / CLEAR_WG_SIZE_X;
        let env_groups_y = (GRID_DIM_U32 + CLEAR_WG_SIZE_Y - 1) / CLEAR_WG_SIZE_Y;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Environment Init Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.environment_init_pipeline);
            pass.set_bind_group(0, &self.compute_bind_group_a, &[]);
            pass.dispatch_workgroups(env_groups_x, env_groups_y, 1);
        }

        // Initialize all agent slots as dead
        let agent_clear_groups = ((self.agent_buffer_capacity as u32) + 255) / 256;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Initialize Dead Agents A->B"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.initialize_dead_pipeline);
            pass.set_bind_group(0, &self.compute_bind_group_a, &[]);
            pass.dispatch_workgroups(agent_clear_groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Initialize Dead Agents B->A"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.initialize_dead_pipeline);
            pass.set_bind_group(0, &self.compute_bind_group_b, &[]);
            pass.dispatch_workgroups(agent_clear_groups, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }
}

fn reset_simulation_state(
    state: &mut Option<GpuState>,
    window: &Arc<Window>,
    egui_state: &mut egui_winit::State,
) {
    if let Some(gpu_state) = state.as_mut() {
        // Fast reset: just clear buffers and reset state, keep GPU device
        gpu_state.fast_reset();
    } else {
        // First time initialization - create new state
        let mut new_state = pollster::block_on(GpuState::new(window.clone()));
        new_state.selected_agent_index = None;
        *state = Some(new_state);
    }

    // Recreate egui_winit state to clear all internal texture tracking
    let egui_ctx = egui::Context::default();
    *egui_state =
        egui_winit::State::new(egui_ctx, egui::ViewportId::ROOT, window, None, None, None);
}

// ============================================================================
// MAIN
// ============================================================================

fn render_splash_screen(
    window: &Window,
    instance: &wgpu::Instance,
    surface: &wgpu::Surface,
    adapter: &wgpu::Adapter,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    // Load splash image
    let splash_img = match image::open("maps/ribossome_splash_screen.png") {
        Ok(img) => img.to_rgba8(),
        Err(_) => {
            // If image not found, just return
            return;
        }
    };

    let splash_dimensions = splash_img.dimensions();
    let splash_size = wgpu::Extent3d {
        width: splash_dimensions.0,
        height: splash_dimensions.1,
        depth_or_array_layers: 1,
    };

    let splash_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Splash Texture"),
        size: splash_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &splash_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &splash_img,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * splash_dimensions.0),
            rows_per_image: Some(splash_dimensions.1),
        },
        splash_size,
    );

    let splash_view = splash_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let splash_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    // Create simple shader for rendering textured quad
    let splash_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Splash Shader"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
            r#"
            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) tex_coords: vec2<f32>,
            }

            @vertex
            fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
                var out: VertexOutput;
                // Generate a quad using two triangles (6 vertices)
                var pos = array<vec2<f32>, 6>(
                    vec2<f32>(-1.0, -1.0),  // Bottom-left
                    vec2<f32>(1.0, -1.0),   // Bottom-right
                    vec2<f32>(-1.0, 1.0),   // Top-left
                    vec2<f32>(-1.0, 1.0),   // Top-left
                    vec2<f32>(1.0, -1.0),   // Bottom-right
                    vec2<f32>(1.0, 1.0)     // Top-right
                );

                var uv = array<vec2<f32>, 6>(
                    vec2<f32>(0.0, 1.0),
                    vec2<f32>(1.0, 1.0),
                    vec2<f32>(0.0, 0.0),
                    vec2<f32>(0.0, 0.0),
                    vec2<f32>(1.0, 1.0),
                    vec2<f32>(1.0, 0.0)
                );

                out.position = vec4<f32>(pos[vertex_index], 0.0, 1.0);
                out.tex_coords = uv[vertex_index];
                return out;
            }

            @group(0) @binding(0) var splash_texture: texture_2d<f32>;
            @group(0) @binding(1) var splash_sampler: sampler;

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                return textureSample(splash_texture, splash_sampler, in.tex_coords);
            }
            "#,
        )),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Splash Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Splash Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&splash_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&splash_sampler),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Splash Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let swapchain_capabilities = surface.get_capabilities(adapter);
    let swapchain_format = swapchain_capabilities.formats[0];

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Splash Render Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &splash_shader,
            entry_point: "vs_main",
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &splash_shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: swapchain_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    // Configure surface
    let size = window.inner_size();
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: swapchain_format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: swapchain_capabilities.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(device, &config);

    // Render the splash screen
    let frame = surface
        .get_current_texture()
        .expect("Failed to acquire next swap chain texture");
    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Splash Render Encoder"),
    });

    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Splash Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_pipeline(&render_pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.draw(0..6, 0..1); // Draw 6 vertices (2 triangles = 1 quad)
    }

    queue.submit(Some(encoder.finish()));
    frame.present();
    device.poll(wgpu::Maintain::Wait);
}

// ============================================================================
// SNAPSHOT SAVE/LOAD FUNCTIONS
// ============================================================================

fn save_simulation_snapshot(
    path: &Path,
    alpha_grid: &[f32],
    beta_grid: &[f32],
    gamma_grid: &[f32],
    snapshot: &SimulationSnapshot,
) -> anyhow::Result<()> {
    use std::io::BufWriter;

    // 1. Create RGB image from grids
    let mut img_data = vec![0u8; GRID_CELL_COUNT * 3];
    for i in 0..GRID_CELL_COUNT {
        img_data[i * 3 + 0] = (beta_grid[i].clamp(0.0, 1.0) * 255.0) as u8;   // R = beta (poison)
        img_data[i * 3 + 1] = (alpha_grid[i].clamp(0.0, 1.0) * 255.0) as u8;  // G = alpha (food)
        img_data[i * 3 + 2] = (gamma_grid[i].clamp(0.0, 1.0) * 255.0) as u8;  // B = gamma (terrain)
    }

    // 2. Serialize and compress metadata
    let json = serde_json::to_string(snapshot)?;
    let compressed = zstd::encode_all(json.as_bytes(), 3)?;
    use base64::Engine;
    let encoded = base64::engine::general_purpose::STANDARD.encode(&compressed);

    // 3. Write PNG with custom text chunk
    let file = std::fs::File::create(path)?;
    let w = BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, GRID_DIM_U32, GRID_DIM_U32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);

    // Add metadata as uncompressed text chunk
    encoder.add_text_chunk("RibossomeSnapshot".to_string(), encoded)?;

    let mut writer = encoder.write_header()?;
    writer.write_image_data(&img_data)?;

    println!("Γ£ô Saved snapshot: {} agents, epoch {}", snapshot.agents.len(), snapshot.epoch);
    Ok(())
}

fn load_simulation_snapshot(
    path: &Path,
) -> anyhow::Result<(Vec<f32>, Vec<f32>, Vec<f32>, SimulationSnapshot)> {
    use std::io::BufReader;

    let file = std::fs::File::open(path)?;
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info()?;

    // Read image data
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;

    if info.width != GRID_DIM_U32 || info.height != GRID_DIM_U32 {
        anyhow::bail!("Invalid snapshot size: {}x{}, expected {}x{}",
            info.width, info.height, GRID_DIM_U32, GRID_DIM_U32);
    }

    // Extract grids from RGB channels
    let mut alpha_grid = vec![0.0f32; GRID_CELL_COUNT];
    let mut beta_grid = vec![0.0f32; GRID_CELL_COUNT];
    let mut gamma_grid = vec![0.0f32; GRID_CELL_COUNT];

    for i in 0..GRID_CELL_COUNT {
        beta_grid[i] = buf[i * 3 + 0] as f32 / 255.0;  // R = beta (poison)
        alpha_grid[i] = buf[i * 3 + 1] as f32 / 255.0; // G = alpha (food)
        gamma_grid[i] = buf[i * 3 + 2] as f32 / 255.0; // B = gamma (terrain)
    }

    // Extract metadata from text chunks
    let text_chunks = reader.info().uncompressed_latin1_text.clone();
    for chunk in text_chunks.iter() {
        if chunk.keyword == "RibossomeSnapshot" {
            use base64::Engine;
            let compressed = base64::engine::general_purpose::STANDARD.decode(&chunk.text)?;
            let json = zstd::decode_all(&compressed[..])?;
            let snapshot: SimulationSnapshot = serde_json::from_slice(&json)?;

            println!("Γ£ô Loaded snapshot: {} agents, epoch {}, saved {}",
                snapshot.agents.len(), snapshot.epoch, snapshot.timestamp);

            return Ok((alpha_grid, beta_grid, gamma_grid, snapshot));
        }
    }

    anyhow::bail!("No RibossomeSnapshot metadata found in PNG")
}

fn main() {
    use env_logger::Env;
    env_logger::Builder::from_env(Env::default().default_filter_or("error"))
        .init();

    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(
        event_loop
            .create_window(
                winit::window::WindowAttributes::default()
                    .with_title("Loading Ribossome...")
                    .with_inner_size(winit::dpi::LogicalSize::new(1100, 600)), // 800 + 300 for inspector
            )
            .unwrap(),
    );

    // Create WGPU resources once
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    });
    let surface = instance.create_surface(window.clone()).unwrap();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .unwrap();

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("GPU Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_storage_buffers_per_shader_stage: 16,
                ..wgpu::Limits::default()
            },
            memory_hints: Default::default(),
        },
        None,
    ))
    .unwrap();

    // Render splash screen
    render_splash_screen(&window, &instance, &surface, &adapter, &device, &queue);

    // Load GPU state in background using channel
    let (tx, rx) = std::sync::mpsc::channel();

    let window_clone = window.clone();
    std::thread::spawn(move || {
        let state = pollster::block_on(GpuState::new_with_resources(
            window_clone,
            instance,
            surface,
            adapter,
            device,
            queue,
        ));
        tx.send(state).unwrap();
    });

    // Animation state for loading screen
    let loading_start = std::time::Instant::now();
    let mut last_message_update = std::time::Instant::now();
    let mut current_message_index = 0;

    let loading_messages = [
        "Synthesizing nucleotides",
        "Assembling ribosomes",
        "Transcribing genetic code",
        "Folding proteins",
        "Calibrating amino acid properties",
        "Preparing primordial soup",
        "Initializing alpha field emitters",
        "Configuring beta signal receptors",
        "Establishing gamma terrain",
        "Compiling genetic instruction set",
        "Optimizing metabolic pathways",
        "Bootstrapping cellular automata",
        "Evolving sensor arrays",
        "Tuning mutation rates",
        "Priming energy gradients",
        "Randomizing initial conditions",
        "Preparing artificial life substrate",
        "Loading biochemical simulation",
    ];

    let mut state: Option<GpuState> = None;

    // Create egui context and winit state
    let mut egui_state = egui_winit::State::new(
        egui::Context::default(),
        egui::ViewportId::ROOT,
        &window,
        None,
        None,
        None,
    );

    let _ = event_loop.run(move |event, target| {
        // Animate loading messages in window title while waiting for GPU state
        if state.is_none() {
            let now = std::time::Instant::now();
            if now.duration_since(last_message_update).as_millis() > 1000 {
                current_message_index = (current_message_index + 1) % loading_messages.len();
                last_message_update = now;
                let elapsed = now.duration_since(loading_start).as_secs_f32();
                let dots = ".".repeat((elapsed * 2.0) as usize % 4);
                let message = format!("{}{}", loading_messages[current_message_index], dots);
                window.set_title(&format!("Ribossome - {}", &message));
                // Also print to console so messages are visible
                println!("Loading: {}", &message);
            }
        }

        // Check if loading finished
        if state.is_none() {
            if let Ok(mut loaded_state) = rx.try_recv() {
                // Try to auto-load previous session from autosave snapshot
                let autosave_path = std::path::Path::new(AUTO_SNAPSHOT_FILE_NAME);
                if autosave_path.exists() {
                    match loaded_state.load_snapshot_from_file(autosave_path) {
                        Ok(_) => {
                            println!("✓ Auto-loaded previous session from epoch {}", loaded_state.epoch);
                        }
                        Err(e) => {
                            eprintln!("❌ Failed to auto-load snapshot: {:?}", e);
                        }
                    }
                }

                state = Some(loaded_state);
                window.set_title("GPU Artificial Life Simulator");
                let _ = window.request_inner_size(winit::dpi::LogicalSize::new(1600, 800));
            }
        }

        match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => {
                // Let egui handle the event first
                let response = egui_state.on_window_event(&window, &event);

                // Only handle simulation controls if egui didn't consume the event
                if !response.consumed {
                    match event {
                        WindowEvent::CloseRequested => {
                            if let Some(mut existing) = state.take() {
                                existing.destroy_resources();
                            }
                            target.exit();
                        }
                        WindowEvent::Resized(physical_size) => {
                            if let Some(state) = state.as_mut() {
                                state.resize(physical_size);
                            }
                        }
                        WindowEvent::KeyboardInput {
                            event:
                                KeyEvent {
                                    physical_key,
                                    state: key_state,
                                    ..
                                },
                            ..
                        } => {
                            if let Some(state) = state.as_mut() {
                                if key_state == ElementState::Pressed {
                                    let mut camera_changed = false;
                                    match physical_key {
                                        PhysicalKey::Code(KeyCode::Equal)
                                        | PhysicalKey::Code(KeyCode::NumpadAdd) => {
                                            state.camera_zoom *= 1.1;
                                            camera_changed = true;
                                        }
                                        PhysicalKey::Code(KeyCode::Minus)
                                        | PhysicalKey::Code(KeyCode::NumpadSubtract) => {
                                            state.camera_zoom /= 1.1;
                                            state.camera_zoom = state.camera_zoom.max(0.1);
                                            camera_changed = true;
                                        }
                                        PhysicalKey::Code(KeyCode::KeyW) => {
                                            state.camera_pan[1] -= 200.0 / state.camera_zoom;
                                            state.camera_pan[1] = state.camera_pan[1].clamp(-0.25 * SIM_SIZE, 1.25 * SIM_SIZE);
                                            state.follow_selected_agent = false;
                                            camera_changed = true;
                                        }
                                        PhysicalKey::Code(KeyCode::KeyS) => {
                                            state.camera_pan[1] += 200.0 / state.camera_zoom;
                                            state.camera_pan[1] = state.camera_pan[1].clamp(-0.25 * SIM_SIZE, 1.25 * SIM_SIZE);
                                            state.follow_selected_agent = false;
                                            camera_changed = true;
                                        }
                                        PhysicalKey::Code(KeyCode::KeyA) => {
                                            state.camera_pan[0] -= 200.0 / state.camera_zoom;
                                            state.camera_pan[0] = state.camera_pan[0].clamp(-0.25 * SIM_SIZE, 1.25 * SIM_SIZE);
                                            state.follow_selected_agent = false;
                                            camera_changed = true;
                                        }
                                        PhysicalKey::Code(KeyCode::KeyD) => {
                                            state.camera_pan[0] += 200.0 / state.camera_zoom;
                                            state.camera_pan[0] = state.camera_pan[0].clamp(-0.25 * SIM_SIZE, 1.25 * SIM_SIZE);
                                            state.follow_selected_agent = false;
                                            camera_changed = true;
                                        }
                                        PhysicalKey::Code(KeyCode::KeyR) => {
                                            // Reset camera
                                            state.camera_zoom = 1.0;
                                            state.camera_pan = [SIM_SIZE / 2.0, SIM_SIZE / 2.0];
                                            camera_changed = true;
                                        }
                                        PhysicalKey::Code(KeyCode::Space) => {
                                            state.ui_visible = !state.ui_visible;
                                        }
                                        PhysicalKey::Code(KeyCode::KeyF) => {
                                            // Toggle follow mode
                                            if state.selected_agent_index.is_some() {
                                                state.follow_selected_agent = !state.follow_selected_agent;
                                                if state.follow_selected_agent {
                                                    // Initialize target to current camera position for smooth start
                                                    state.camera_target = state.camera_pan;
                                                    state.camera_velocity = [0.0, 0.0];
                                                }
                                            }
                                        }
                                        _ => {}
                                    }

                                    if camera_changed {
                                        window.request_redraw();
                                    }
                                }
                            }
                        }
                        WindowEvent::MouseInput {
                            state: button_state,
                            button,
                            ..
                        } => {
                            if let Some(state) = state.as_mut() {
                                if button == winit::event::MouseButton::Right {
                                    state.is_dragging = button_state == ElementState::Pressed;
                                    if !state.is_dragging {
                                        state.last_mouse_pos = None;
                                    }
                                } else if button == winit::event::MouseButton::Left
                                    && button_state == ElementState::Pressed
                                {
                                    // Left click - select agent for debug panel
                                    if let Some(mouse_pos) = state.last_mouse_pos {
                                        state.select_agent_at_screen_pos(mouse_pos);
                                    }
                                }
                            }
                        }
                        WindowEvent::CursorMoved { position, .. } => {
                            if let Some(state) = state.as_mut() {
                                let current_pos = [position.x as f32, position.y as f32];

                                if state.is_dragging {
                                    if let Some(last_pos) = state.last_mouse_pos {
                                        let delta_x = current_pos[0] - last_pos[0];
                                        let delta_y = current_pos[1] - last_pos[1];

                                        // Convert screen space delta to world space delta (Y inverted)
                                        let world_scale = (state.surface_config.width as f32
                                            / SIM_SIZE)
                                            * state.camera_zoom;
                                        state.camera_pan[0] -= delta_x / world_scale;
                                        state.camera_pan[1] += delta_y / world_scale; // Inverted Y

                                        // Clamp camera position to -0.25 to 1.25 of SIM_SIZE
                                        state.camera_pan[0] = state.camera_pan[0].clamp(-0.25 * SIM_SIZE, 1.25 * SIM_SIZE);
                                        state.camera_pan[1] = state.camera_pan[1].clamp(-0.25 * SIM_SIZE, 1.25 * SIM_SIZE);

                                        state.follow_selected_agent = false;
                                        window.request_redraw();
                                    }
                                }
                                // Always update mouse position (for both dragging and click selection)
                                state.last_mouse_pos = Some(current_pos);
                            }
                        }
                        WindowEvent::MouseWheel { delta, .. } => {
                            if let Some(state) = state.as_mut() {
                                let zoom_delta = match delta {
                                    MouseScrollDelta::LineDelta(_, y) => y * 0.1,
                                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                                };

                                // Check if mouse is over inspector area (rightmost 300px)
                                let mouse_over_inspector = if let Some(mouse_pos) = state.last_mouse_pos {
                                    mouse_pos[0] > (state.surface_config.width as f32 - 300.0)
                                } else {
                                    false
                                };

                                if mouse_over_inspector {
                                    // Zoom inspector
                                    state.inspector_zoom *= 1.0 + zoom_delta;
                                    state.inspector_zoom = state.inspector_zoom.clamp(0.1, 10.0);
                                } else {
                                    // Zoom camera
                                    state.camera_zoom *= 1.0 + zoom_delta;
                                    state.camera_zoom = state.camera_zoom.clamp(0.1, 2000.0);
                                }
                                window.request_redraw();
                            }
                        }
                        WindowEvent::DroppedFile(path) => {
                            // Handle drag-and-drop file loading
                            if let Some(ext) = path.extension() {
                                if ext == "png" {
                                    // Reset simulation state before loading
                                    reset_simulation_state(&mut state, &window, &mut egui_state);
                                    if let Some(gpu_state) = state.as_mut() {
                                        match gpu_state.load_snapshot_from_file(&path) {
                                            Ok(_) => println!("✓ Snapshot loaded from: {}", path.display()),
                                            Err(e) => eprintln!("❌ Failed to load snapshot: {}", e),
                                        }
                                    }
                                }
                            }
                        }
                        WindowEvent::RedrawRequested => {
                            let mut reset_requested = false;

                            // Render loading screen while state is being initialized
                            if state.is_none() {
                                window.request_redraw();
                                return; // Don't process further until state is ready
                            }

                            if let Some(state) = state.as_mut() {
                                // Frame rate limiting
                                if let Some(target_frame_time) = state.frame_time_cap() {
                                    let elapsed = state.last_frame_time.elapsed();
                                    if elapsed < target_frame_time {
                                        std::thread::sleep(target_frame_time - elapsed);
                                    }
                                }
                                let now = std::time::Instant::now();
                                let frame_dt = (now - state.last_frame_time).as_secs_f32().clamp(0.001, 0.1);
                                state.last_frame_time = now;

                                // Run one simulation step per frame
                                let should_draw = if state.is_paused {
                                    // Always draw when paused so camera movement is visible
                                    true
                                } else if state.current_mode == 2 {
                                    // Fast Draw mode: draw every N steps
                                    state.frame_count % state.render_interval == 0
                                } else {
                                    // VSync/Full Speed: always draw
                                    true
                                };
                                // Only run simulation when not paused AND there are living agents
                                let should_run_simulation = !state.is_paused && state.alive_count > 0;

                                state.update(should_draw, frame_dt);

                                // Only increment epoch and update stats when simulation is running
                                if should_run_simulation {
                                    state.epoch += 1;

                                    // Auto-snapshot every AUTO_SNAPSHOT_INTERVAL epochs
                                    if state.epoch > 0 && state.epoch % AUTO_SNAPSHOT_INTERVAL == 0 && state.epoch != state.last_autosave_epoch {
                                        state.last_autosave_epoch = state.epoch;
                                        let autosave_path = std::path::Path::new(AUTO_SNAPSHOT_FILE_NAME);
                                        if let Err(e) = state.save_snapshot_to_file(autosave_path) {
                                            eprintln!("ΓÜá Auto-snapshot failed at epoch {}: {:?}", state.epoch, e);
                                        } else {
                                            println!("Γ£ô Auto-snapshot saved at epoch {}", state.epoch);
                                        }
                                    }

                                    // Sample population for statistics graph
                                    if state.epoch - state.last_sample_epoch >= state.epoch_sample_interval {
                                        state.population_history.push(state.alive_count);
                                        // Keep only the last max_history_points
                                        if state.population_history.len() > state.max_history_points {
                                            state.population_history.remove(0);
                                        }
                                        state.last_sample_epoch = state.epoch;
                                    }

                                    // Update epochs per second counter
                                    let now = std::time::Instant::now();
                                    let elapsed = now.duration_since(state.last_epoch_update).as_secs_f32();
                                    if elapsed >= 0.5 {
                                        let epochs_elapsed = state.epoch - state.last_epoch_count;
                                        state.epochs_per_second = epochs_elapsed as f32 / elapsed;
                                        state.last_epoch_update = now;
                                        state.last_epoch_count = state.epoch;
                                    }
                                } else {
                                    // When paused or no agents, reset the epoch timing so it doesn't jump when resumed
                                    state.last_epoch_update = std::time::Instant::now();
                                    state.last_epoch_count = state.epoch;
                                }

                                // Build egui UI
                                let raw_input = egui_state.take_egui_input(&window);
                                let full_output = egui_state.egui_ctx().run(raw_input, |ctx| {
                                    // Left side panel for simulation controls (only show if ui_visible)
                                    if state.ui_visible {
                                    egui::SidePanel::left("simulation_controls")
                                        .default_width(350.0)
                                        .resizable(true)
                                        .frame(egui::Frame::none()
                                            .fill(egui::Color32::from_rgb(70, 70, 70))
                                            .inner_margin(egui::Margin::same(10.0)))
                                        .show(ctx, |ui| {
                                            // Action buttons at the top with lighter background
                                            egui::Frame::none()
                                                .fill(egui::Color32::from_rgb(85, 85, 85))
                                                .inner_margin(egui::Margin::same(8.0))
                                                .show(ui, |ui| {
                                                    ui.horizontal_wrapped(|ui| {
                                                        if ui.button("Spawn 20000 Agents").clicked() && !state.is_paused {
                                                            state.queue_random_spawns(20000);
                                                        }

                                                        if ui.button("Load Agent & Spawn 100...").clicked() && !state.is_paused {
                                                            match load_agent_via_dialog() {
                                                                Ok(genome) => {
                                                                    println!("Successfully loaded genome, spawning 100 clones...");

                                                                    for _ in 0..100 {
                                                                        state.rng_state = state.rng_state
                                                                            .wrapping_mul(6364136223846793005u64)
                                                                            .wrapping_add(1442695040888963407u64);
                                                                        let seed = (state.rng_state >> 32) as u32;

                                                                        state.rng_state = state.rng_state
                                                                            .wrapping_mul(6364136223846793005u64)
                                                                            .wrapping_add(1442695040888963407u64);
                                                                        let genome_seed = (state.rng_state >> 32) as u32;

                                                                        state.rng_state = state.rng_state
                                                                            .wrapping_mul(6364136223846793005u64)
                                                                            .wrapping_add(1442695040888963407u64);
                                                                        let rx = (state.rng_state >> 32) as f32 / u32::MAX as f32;

                                                                        state.rng_state = state.rng_state
                                                                            .wrapping_mul(6364136223846793005u64)
                                                                            .wrapping_add(1442695040888963407u64);
                                                                        let ry = (state.rng_state >> 32) as f32 / u32::MAX as f32;

                                                                        state.rng_state = state.rng_state
                                                                            .wrapping_mul(6364136223846793005u64)
                                                                            .wrapping_add(1442695040888963407u64);
                                                                        let rotation = ((state.rng_state >> 32) as f32 / u32::MAX as f32)
                                                                            * std::f32::consts::TAU;

                                                                        let request = SpawnRequest {
                                                                            seed,
                                                                            genome_seed,
                                                                            flags: 1,
                                                                            _pad_seed: 0,
                                                                            position: [rx * SIM_SIZE, ry * SIM_SIZE],
                                                                            energy: 10.0,
                                                                            rotation,
                                                                            genome_override: genome,
                                                                        };

                                                                        state.cpu_spawn_queue.push(request);
                                                                    }
                                                                    println!("Queued 100 clones for spawning.");
                                                                }
                                                                Err(err) => eprintln!("Load canceled or failed: {err:?}"),
                                                            }
                                                        }

                                                        if ui.button("Save Selected Agent").clicked() {
                                                            if let Some(agent) = state.selected_agent_data {
                                                                if let Err(err) = save_agent_via_dialog(&agent) {
                                                                    eprintln!("Save canceled or failed: {err:?}");
                                                                }
                                                            }
                                                        }
                                                    });
                                                });

                                            ui.add_space(4.0);

                                            // Top section (no tabs) - Always visible
                                            ui.horizontal(|ui| {
                                                if ui.button(if state.is_paused { "Resume" } else { "Pause" }).clicked() {
                                                    state.is_paused = !state.is_paused;
                                                }
                                                if ui.button("Reset Simulation").clicked() {
                                                    reset_requested = true;
                                                }
                                                let mut fps_cap_enabled = matches!(state.current_mode, 0 | 3);
                                                if ui.checkbox(&mut fps_cap_enabled, "Enable FPS Cap").changed() {
                                                    if fps_cap_enabled {
                                                        state.set_speed_mode(if state.current_mode == 3 { 3 } else { 0 });
                                                    } else {
                                                        state.set_speed_mode(1);
                                                    }
                                                }

                                                ui.separator();
                                                if ui.button("≡ƒÆ╛ Save Snapshot").clicked() {
                                                    state.snapshot_save_requested = true;
                                                }
                                                if ui.button("≡ƒôé Load Snapshot").clicked() {
                                                    state.snapshot_load_requested = true;
                                                }
                                            });

                                            ui.separator();
                                            ui.heading("Simulation Speed");
                                            let mut mode = state.current_mode;
                                            let old_mode = mode;
                                            ui.horizontal(|ui| {
                                                ui.selectable_value(&mut mode, 3, "Slow (25 FPS)");
                                                ui.selectable_value(&mut mode, 0, "VSync (60 FPS)");
                                                ui.selectable_value(&mut mode, 1, "Full Speed");
                                                ui.selectable_value(&mut mode, 2, "Fast Draw");
                                            });
                                            if mode != old_mode {
                                                state.set_speed_mode(mode);
                                            }
                                            if mode == 2 {
                                                ui.add(
                                                    egui::Slider::new(&mut state.render_interval, 1..=10000)
                                                        .text("Draw every N steps")
                                                        .logarithmic(true),
                                                );
                                            }
                                            ui.label(format!("Epoch: {}", state.epoch));
                                            ui.label(format!(
                                                "Epochs/sec: {:.1}",
                                                state.epochs_per_second
                                            ));

                                            ui.separator();
                                            ui.heading("Population Overview");
                                            ui.label(format!("Total Agents: {}", state.agent_count));
                                            ui.label(format!("Living Agents: {}", state.alive_count));
                                            ui.label(format!(
                                                "Capacity: {}",
                                                state.agent_buffer_capacity
                                            ));

                                            ui.separator();
                                            ui.collapsing("Population History", |ui| {
                                                ui.label(format!(
                                                    "Samples: {} (every {} epochs)",
                                                    state.population_history.len(),
                                                    state.epoch_sample_interval
                                                ));

                                                if !state.population_history.is_empty() {
                                                    use egui_plot::{Line, Plot, PlotPoints};

                                                    let points: PlotPoints = state
                                                        .population_history
                                                        .iter()
                                                        .enumerate()
                                                        .map(|(i, &pop)| [i as f64, pop as f64])
                                                        .collect();

                                                    let line = Line::new(points)
                                                        .color(egui::Color32::from_rgb(100, 200, 100))
                                                        .name("Population");

                                                    Plot::new("population_plot")
                                                        .height(150.0)
                                                        .show_axes(true)
                                                        .show_grid(true)
                                                        .allow_drag(true)
                                                        .allow_zoom([true, false])
                                                        .allow_scroll(false)
                                                        .show(ui, |plot_ui| {
                                                            plot_ui.line(line);
                                                        });
                                                }
                                            });

                                            ui.separator();
                                            // Tab selection for detailed controls with colored buttons
                                            ui.horizontal(|ui| {
                                                let tab_colors = [
                                                    ("Simulation", egui::Color32::from_rgb(50, 55, 60)),
                                                    ("Agents", egui::Color32::from_rgb(55, 50, 60)),
                                                    ("Environment", egui::Color32::from_rgb(50, 60, 55)),
                                                    ("Evolution", egui::Color32::from_rgb(60, 55, 50)),
                                                    ("Difficulty", egui::Color32::from_rgb(60, 50, 50)),
                                                    ("Visualization", egui::Color32::from_rgb(55, 55, 55)),
                                                ];

                                                for (idx, (name, color)) in tab_colors.iter().enumerate() {
                                                    let is_selected = state.ui_tab == idx;
                                                    let button = egui::Button::new(*name)
                                                        .fill(if is_selected { *color } else { egui::Color32::from_rgb(40, 40, 40) })
                                                        .stroke(egui::Stroke::new(1.0, if is_selected { egui::Color32::WHITE } else { *color }));

                                                    if ui.add(button).clicked() {
                                                        state.ui_tab = idx;
                                                    }
                                                }
                                            });
                                            ui.separator();

                                            // Tab content with colored backgrounds
                                            let tab_color = match state.ui_tab {
                                                0 => egui::Color32::from_rgb(50, 55, 60),  // Simulation - blue-gray
                                                1 => egui::Color32::from_rgb(55, 50, 60),  // Agents - purple-gray
                                                2 => egui::Color32::from_rgb(50, 60, 55),  // Environment - green-gray
                                                3 => egui::Color32::from_rgb(60, 55, 50),  // Evolution - orange-gray
                                                4 => egui::Color32::from_rgb(60, 50, 50),  // Difficulty - red-gray
                                                5 => egui::Color32::from_rgb(55, 55, 55),  // Visualization - neutral gray
                                                _ => egui::Color32::from_rgb(50, 50, 50),
                                            };

                                            // Fill the background of the remaining space
                                            let remaining_rect = ui.available_rect_before_wrap();
                                            ui.painter().rect_filled(remaining_rect, 0.0, tab_color);

                                            match state.ui_tab {
                                                0 => {
                                                    // Simulation tab
                                                    egui::ScrollArea::vertical().show(ui, |ui| {
                                                        ui.heading("Camera");
                                                        ui.add(
                                                            egui::Slider::new(&mut state.camera_zoom, 0.1..=2000.0)
                                                                .text("Zoom")
                                                                .logarithmic(true),
                                                        );
                                            if ui.button("Reset Camera (R)").clicked() {
                                                state.camera_zoom = 1.0;
                                                state.camera_pan = [SIM_SIZE / 2.0, SIM_SIZE / 2.0];
                                            }                                                        ui.separator();
                                                        ui.heading("Settings");
                                                        ui.horizontal(|ui| {
                                                            if ui.button("Save Settings").clicked() {
                                                                if let Some(path) = rfd::FileDialog::new()
                                                                    .set_file_name("simulation_settings.json")
                                                                    .add_filter("JSON", &["json"])
                                                                    .save_file()
                                                                {
                                                                    let settings = SimulationSettings {
                                                                        camera_zoom: state.camera_zoom,
                                                                        spawn_probability: state.spawn_probability,
                                                                        death_probability: state.death_probability,
                                                                        mutation_rate: state.mutation_rate,
                                                                        auto_replenish: state.auto_replenish,
                                                                        diffusion_interval: state.diffusion_interval,
                                                                        slope_interval: state.slope_interval,
                                                                        alpha_blur: state.alpha_blur,
                                                                        beta_blur: state.beta_blur,
                                                                        gamma_blur: state.gamma_blur,
                                                                        alpha_slope_bias: state.alpha_slope_bias,
                                                                        beta_slope_bias: state.beta_slope_bias,
                                                                        alpha_multiplier: state.alpha_multiplier,
                                                                        beta_multiplier: state.beta_multiplier,
                                                                        alpha_rain_map_path: state.alpha_rain_map_path.clone(),
                                                                        beta_rain_map_path: state.beta_rain_map_path.clone(),
                                                                        chemical_slope_scale_alpha: state.chemical_slope_scale_alpha,
                                                                        chemical_slope_scale_beta: state.chemical_slope_scale_beta,
                                                                        alpha_noise_scale: state.alpha_noise_scale,
                                                                        beta_noise_scale: state.beta_noise_scale,
                                                                        gamma_noise_scale: state.gamma_noise_scale,
                                                                        noise_power: state.noise_power,
                                                                        food_power: state.food_power,
                                                                        poison_power: state.poison_power,
                                                                        amino_maintenance_cost: state.amino_maintenance_cost,
                                                                        pairing_cost: state.pairing_cost,
                                                                        prop_wash_strength: state.prop_wash_strength,
                                                                        repulsion_strength: state.repulsion_strength,
                                                                        agent_repulsion_strength: state.agent_repulsion_strength,
                                                                        limit_fps: state.limit_fps,
                                                                        limit_fps_25: state.limit_fps_25,
                                                                        render_interval: state.render_interval,
                                                                        gamma_debug_visual: state.gamma_debug_visual,
                                                                        slope_debug_visual: state.slope_debug_visual,
                                                                        rain_debug_visual: state.rain_debug_visual,
                                                                        fluid_show: state.fluid_show,
                                                                        vector_force_power: state.vector_force_power,
                                                                        vector_force_x: state.vector_force_x,
                                                                        vector_force_y: state.vector_force_y,
                                                                        gamma_hidden: state.gamma_hidden,
                                                                        debug_per_segment: state.debug_per_segment,
                                                                        gamma_vis_min: state.gamma_vis_min,
                                                                        gamma_vis_max: state.gamma_vis_max,
                                                                        alpha_show: state.alpha_show,
                                                                        beta_show: state.beta_show,
                                                                        gamma_show: state.gamma_show,
                                                                        slope_lighting: state.slope_lighting,
                                                                        slope_lighting_strength: state.slope_lighting_strength,
                                                                        trail_diffusion: state.trail_diffusion,
                                                                        trail_decay: state.trail_decay,
                                                                        trail_opacity: state.trail_opacity,
                                                                        trail_show: state.trail_show,
                                                                        interior_isotropic: state.interior_isotropic,
                                                                        ignore_stop_codons: state.ignore_stop_codons,
                                                                        require_start_codon: state.require_start_codon,
                                                                        asexual_reproduction: state.asexual_reproduction,
                                                                        alpha_rain_variation: state.alpha_rain_variation,
                                                                        beta_rain_variation: state.beta_rain_variation,
                                                                        alpha_rain_phase: state.alpha_rain_phase,
                                                                        beta_rain_phase: state.beta_rain_phase,
                                                                        alpha_rain_freq: state.alpha_rain_freq,
                                                                        beta_rain_freq: state.beta_rain_freq,
                                                                        difficulty: state.difficulty.clone(),
                                                                        background_color: state.background_color,
                                                                        alpha_blend_mode: state.alpha_blend_mode,
                                                                        beta_blend_mode: state.beta_blend_mode,
                                                                        gamma_blend_mode: state.gamma_blend_mode,
                                                                        slope_blend_mode: state.slope_blend_mode,
                                                                        alpha_color: state.alpha_color,
                                                                        beta_color: state.beta_color,
                                                                        gamma_color: state.gamma_color,
                                                                        grid_interpolation: state.grid_interpolation,
                                                                        alpha_gamma_adjust: state.alpha_gamma_adjust,
                                                                        beta_gamma_adjust: state.beta_gamma_adjust,
                                                                        gamma_gamma_adjust: state.gamma_gamma_adjust,
                                                                        light_direction: state.light_direction,
                                                                        light_power: state.light_power,
                                                                        agent_blend_mode: state.agent_blend_mode,
                                                                        agent_color: state.agent_color,
                                                                        agent_color_blend: state.agent_color_blend,
                                                                        agent_trail_decay: state.agent_trail_decay,
                                                                    };
                                                                    if let Err(err) = settings.save_to_disk(&path) {
                                                                        eprintln!("Failed to save settings: {err:?}");
                                                                    }
                                                                }
                                                            }
                                                            if ui.button("Load Settings").clicked() {
                                                                if let Some(path) = rfd::FileDialog::new()
                                                                    .add_filter("JSON", &["json"])
                                                                    .pick_file()
                                                                {
                                                                    if let Ok(settings) = SimulationSettings::load_from_disk(&path) {
                                                                        state.camera_zoom = settings.camera_zoom;
                                                                        state.spawn_probability = settings.spawn_probability;
                                                                        state.death_probability = settings.death_probability;
                                                                        state.mutation_rate = settings.mutation_rate;
                                                                        state.auto_replenish = settings.auto_replenish;
                                                                        state.diffusion_interval = settings.diffusion_interval;
                                                                        state.slope_interval = settings.slope_interval;
                                                                        state.alpha_blur = settings.alpha_blur;
                                                                        state.beta_blur = settings.beta_blur;
                                                                        state.gamma_blur = settings.gamma_blur;
                                                                        state.alpha_slope_bias = settings.alpha_slope_bias;
                                                                        state.beta_slope_bias = settings.beta_slope_bias;
                                                                        state.alpha_multiplier = settings.alpha_multiplier;
                                                                        state.beta_multiplier = settings.beta_multiplier;
                                                                        state.alpha_rain_map_path = settings.alpha_rain_map_path.clone();
                                                                        state.beta_rain_map_path = settings.beta_rain_map_path.clone();
                                                                        state.chemical_slope_scale_alpha = settings.chemical_slope_scale_alpha;
                                                                        state.chemical_slope_scale_beta = settings.chemical_slope_scale_beta;
                                                                        state.food_power = settings.food_power;
                                                                        state.poison_power = settings.poison_power;
                                                                        state.amino_maintenance_cost = settings.amino_maintenance_cost;
                                                                        state.pairing_cost = settings.pairing_cost;
                                                                        state.prop_wash_strength = settings.prop_wash_strength;
                                                                        state.repulsion_strength = settings.repulsion_strength;
                                                                        state.limit_fps = settings.limit_fps;
                                                                        state.limit_fps_25 = settings.limit_fps_25;
                                                                        state.render_interval = settings.render_interval;
                                                                        state.gamma_debug_visual = settings.gamma_debug_visual;
                                                                        state.slope_debug_visual = settings.slope_debug_visual;
                                                                        state.gamma_hidden = settings.gamma_hidden;
                                                                        state.debug_per_segment = settings.debug_per_segment;
                                                                        state.gamma_vis_min = settings.gamma_vis_min;
                                                                        state.gamma_vis_max = settings.gamma_vis_max;
                                                                        state.alpha_show = settings.alpha_show;
                                                                        state.beta_show = settings.beta_show;
                                                                        state.gamma_show = settings.gamma_show;
                                                                        state.slope_lighting = settings.slope_lighting;
                                                                        state.slope_lighting_strength = settings.slope_lighting_strength;
                                                                        state.trail_diffusion = settings.trail_diffusion;
                                                                        state.trail_decay = settings.trail_decay;
                                                                        state.trail_opacity = settings.trail_opacity;
                                                                        state.trail_show = settings.trail_show;
                                                                        state.interior_isotropic = settings.interior_isotropic;
                                                                        state.ignore_stop_codons = settings.ignore_stop_codons;
                                                                        state.require_start_codon = settings.require_start_codon;
                                                                        state.alpha_rain_variation = settings.alpha_rain_variation;
                                                                        state.beta_rain_variation = settings.beta_rain_variation;
                                                                        state.alpha_rain_phase = settings.alpha_rain_phase;
                                                                        state.beta_rain_phase = settings.beta_rain_phase;
                                                                        state.alpha_rain_freq = settings.alpha_rain_freq;
                                                                        state.beta_rain_freq = settings.beta_rain_freq;
                                                                        state.difficulty = settings.difficulty;
                                                                        state.background_color = settings.background_color;
                                                                        state.alpha_blend_mode = settings.alpha_blend_mode;
                                                                        state.beta_blend_mode = settings.beta_blend_mode;
                                                                        state.gamma_blend_mode = settings.gamma_blend_mode;
                                                                        state.alpha_color = settings.alpha_color;
                                                                        state.beta_color = settings.beta_color;
                                                                        state.gamma_color = settings.gamma_color;
                                                                        state.grid_interpolation = settings.grid_interpolation;
                                                                        state.alpha_gamma_adjust = settings.alpha_gamma_adjust;
                                                                        state.beta_gamma_adjust = settings.beta_gamma_adjust;
                                                                        state.gamma_gamma_adjust = settings.gamma_gamma_adjust;

                                                                        if let Some(path) = &settings.alpha_rain_map_path {
                                                                            let _ = state.load_alpha_rain_map(path);
                                                                        }
                                                                        if let Some(path) = &settings.beta_rain_map_path {
                                                                            let _ = state.load_beta_rain_map(path);
                                                                        }
                                                                    } else {
                                                                        eprintln!("Failed to load settings from {}", path.display());
                                                                    }
                                                                }
                                                            }
                                                        });

                                                        ui.separator();
                                                        ui.heading("Simulation Speed");
                                                        let mut mode = state.current_mode;
                                                        let old_mode = mode;
                                                        ui.horizontal(|ui| {
                                                            ui.selectable_value(&mut mode, 3, "Slow (25 FPS)");
                                                            ui.selectable_value(&mut mode, 0, "VSync (60 FPS)");
                                                            ui.selectable_value(&mut mode, 1, "Full Speed");
                                                            ui.selectable_value(&mut mode, 2, "Fast Draw");
                                                        });
                                                        if mode != old_mode {
                                                            state.set_speed_mode(mode);
                                                        }
                                                        if mode == 2 {
                                                            ui.add(
                                                                egui::Slider::new(&mut state.render_interval, 1..=10000)
                                                                    .text("Draw every N steps")
                                                                    .logarithmic(true),
                                                            );
                                                        }
                                                        ui.label(format!("Epoch: {}", state.epoch));
                                                        ui.label(format!(
                                                            "Epochs/sec: {:.1}",
                                                            state.epochs_per_second
                                                        ));

                                                        ui.separator();
                                                        ui.heading("Population Overview");
                                                        ui.label(format!("Total Agents: {}", state.agent_count));
                                                        ui.label(format!("Living Agents: {}", state.alive_count));
                                                        ui.label(format!(
                                                            "Capacity: {}",
                                                            state.agent_buffer_capacity
                                                        ));

                                                        ui.separator();
                                                        ui.collapsing("Population History", |ui| {
                                                            ui.label(format!(
                                                                "Samples: {} (every {} epochs)",
                                                                state.population_history.len(),
                                                                state.epoch_sample_interval
                                                            ));

                                                            if !state.population_history.is_empty() {
                                                                use egui_plot::{Line, Plot, PlotPoints};

                                                                let points: PlotPoints = state
                                                                    .population_history
                                                                    .iter()
                                                                    .enumerate()
                                                                    .map(|(i, &pop)| [i as f64, pop as f64])
                                                                    .collect();

                                                                let line = Line::new(points)
                                                                    .color(egui::Color32::from_rgb(100, 200, 100))
                                                                    .name("Population");

                                                                Plot::new("population_plot")
                                                                    .height(150.0)
                                                                    .show_axes(true)
                                                                    .show_grid(true)
                                                                    .allow_drag(true)
                                                                    .allow_zoom([true, false])
                                                                    .allow_scroll(false)
                                                                    .show(ui, |plot_ui| {
                                                                        plot_ui.line(line);
                                                                    });
                                                            } else {
                                                                ui.label("No data yet (waiting for first sample)");
                                                            }
                                                        });

                                                        ui.separator();
                                                        ui.checkbox(
                                                            &mut state.debug_per_segment,
                                                            "Debug: Per-segment overlay",
                                                        );
                                                    });
                                                }
                                                1 => {
                                                    // Agents tab
                                                    egui::ScrollArea::vertical().show(ui, |ui| {
                                                        ui.heading("Population Controls");
                                                        ui.label(format!(
                                                            "Total Agents: {} ({} alive)",
                                                            state.agent_count,
                                                            state.alive_count
                                                        ));
                                                        ui.label(format!(
                                                            "Capacity: {}",
                                                            state.agent_buffer_capacity
                                                        ));
                                                        ui.horizontal(|ui| {
                                                            if ui
                                                                .button(if state.auto_replenish {
                                                                    "Auto Replenish: ON"
                                                                } else {
                                                                    "Auto Replenish: OFF"
                                                                })
                                                                .clicked()
                                                            {
                                                                state.auto_replenish = !state.auto_replenish;
                                                            }
                                                            if ui.button("Kill All").clicked() {
                                                                // Ensure no stale readbacks race in after we zero everything
                                                                state.device.poll(wgpu::Maintain::Wait);
                                                                state.process_completed_alive_readbacks();

                                                                // Kill all agents immediately
                                                                state.agent_count = 0;
                                                                state.alive_count = 0;
                                                                state.selected_agent_index = None;
                                                                state.selected_agent_data = None;
                                                                state.cpu_spawn_queue.clear();
                                                                state.spawn_request_count = 0;
                                                                state.pending_spawn_upload = false;
                                                                // Clear pending alive readbacks to prevent GPU overwriting our 0 count
                                                                for i in 0..2 {
                                                                    state.alive_readback_inflight[i] = false;
                                                                    if let Ok(mut guard) =
                                                                        state.alive_readback_pending[i].lock()
                                                                    {
                                                                        *guard = None;
                                                                    }
                                                                }

                                                                // Clear GPU buffers so killed agents cannot be resurrected
                                                                let mut encoder = state.device.create_command_encoder(
                                                                    &wgpu::CommandEncoderDescriptor {
                                                                        label: Some("KillAll Encoder"),
                                                                    },
                                                                );
                                                                encoder.clear_buffer(&state.agents_buffer_a, 0, None);
                                                                encoder.clear_buffer(&state.agents_buffer_b, 0, None);
                                                                encoder.clear_buffer(&state.new_agents_buffer, 0, None);
                                                                encoder.clear_buffer(&state.spawn_debug_counters, 0, None);
                                                                encoder.clear_buffer(&state.alive_counter, 0, None);
                                                                state.queue.submit(Some(encoder.finish()));
                                                                state.window.request_redraw();
                                                            }
                                                        });

                                                        ui.horizontal(|ui| {
                                                            if ui.button("Save Selected Agent...").clicked() {
                                                                if let Some(agent) = state.selected_agent_data {
                                                                    if let Err(err) = save_agent_via_dialog(&agent) {
                                                                        eprintln!(
                                                                            "Save canceled or failed: {err:?}"
                                                                        );
                                                                    }
                                                                }
                                                            }
                                                            if ui
                                                                .button("Load Agent & Spawn 100 Clones...")
                                                                .clicked()
                                                                && !state.is_paused
                                                            {
                                                                match load_agent_via_dialog() {
                                                                    Ok(genome) => {
                                                                        println!(
                                                                            "Successfully loaded genome, spawning 100 clones..."
                                                                        );

                                                                        for _ in 0..100 {
                                                                            state.rng_state = state
                                                                                .rng_state
                                                                                .wrapping_mul(6364136223846793005u64)
                                                                                .wrapping_add(1442695040888963407u64);
                                                                            let seed =
                                                                                (state.rng_state >> 32) as u32;

                                                                            state.rng_state = state
                                                                                .rng_state
                                                                                .wrapping_mul(6364136223846793005u64)
                                                                                .wrapping_add(1442695040888963407u64);
                                                                            let genome_seed =
                                                                                (state.rng_state >> 32) as u32;

                                                                            state.rng_state = state
                                                                                .rng_state
                                                                                .wrapping_mul(6364136223846793005u64)
                                                                                .wrapping_add(1442695040888963407u64);
                                                                            let rx = (state.rng_state >> 32) as f32
                                                                                / u32::MAX as f32;

                                                                            state.rng_state = state
                                                                                .rng_state
                                                                                .wrapping_mul(6364136223846793005u64)
                                                                                .wrapping_add(1442695040888963407u64);
                                                                            let ry = (state.rng_state >> 32) as f32
                                                                                / u32::MAX as f32;

                                                                            state.rng_state = state
                                                                                .rng_state
                                                                                .wrapping_mul(6364136223846793005u64)
                                                                                .wrapping_add(1442695040888963407u64);
                                                                            let rotation = ((state.rng_state >> 32) as f32
                                                                                / u32::MAX as f32)
                                                                                * std::f32::consts::TAU;

                                                                            let request = SpawnRequest {
                                                                                seed,
                                                                                genome_seed,
                                                                                flags: 1, // Bit 0 = use genome_override
                                                                                _pad_seed: 0,
                                                                                position: [
                                                                                    rx * SIM_SIZE,
                                                                                    ry * SIM_SIZE,
                                                                                ],
                                                                                energy: 10.0,
                                                                                rotation,
                                                                                genome_override: genome,
                                                                            };

                                                                            state.cpu_spawn_queue.push(request);
                                                                        }

                                                                        state.spawn_request_count =
                                                                            state.cpu_spawn_queue.len() as u32;
                                                                        state.pending_spawn_upload = true;
                                                                        state.window.request_redraw();
                                                                        println!("Enqueued 100 spawn requests");
                                                                    }
                                                                    Err(err) => {
                                                                        eprintln!("Failed to load agent: {err:?}");
                                                                    }
                                                                }
                                                            }
                                                        });

                                                        ui.separator();
                                                        ui.heading("Reproduction & Selection");
                                                        ui.add(
                                                            egui::Slider::new(
                                                                &mut state.spawn_probability,
                                                                0.0..=1.0,
                                                            )
                                                            .text("Pairing Probability"),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(
                                                                &mut state.death_probability,
                                                                0.0..=0.1,
                                                            )
                                                            .text("Death Probability"),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(
                                                                &mut state.mutation_rate,
                                                                0.0..=0.1,
                                                            )
                                                            .text("Mutation Rate (per base)")
                                                            .step_by(0.001),
                                                        );
                                                        ui.label(format!(
                                                            "Average mutations per offspring: {:.1}",
                                                            state.mutation_rate * 64.0
                                                        ));

                                                        ui.separator();
                                                        ui.heading("Genetics & Signals");
                                                        ui.checkbox(
                                                            &mut state.require_start_codon,
                                                            "Require AUG start codon",
                                                        )
                                                        .on_hover_text(
                                                            "When enabled, genomes must contain AUG (Methionine) to begin translation",
                                                        );
                                                        ui.checkbox(
                                                            &mut state.ignore_stop_codons,
                                                            "Ignore stop codons (experimental)",
                                                        )
                                                        .on_hover_text(
                                                            "When enabled, genomes translate to full 64 amino acids regardless of stop codons",
                                                        );
                                                        ui.checkbox(
                                                            &mut state.asexual_reproduction,
                                                            "Asexual reproduction (direct copy)",
                                                        )
                                                        .on_hover_text(
                                                            "When enabled, offspring are direct mutated copies of parent genome. When disabled, offspring are reverse-complemented (sexual reproduction with complementary pairing)",
                                                        );
                                                        ui.checkbox(
                                                            &mut state.interior_isotropic,
                                                            "Isotropic diffusion (simple averaging)",
                                                        )
                                                        .on_hover_text(
                                                            "When enabled, uses simple neighbor averaging. When disabled, uses asymmetric left/right multipliers from amino acid properties",
                                                        );

                                                        ui.separator();
                                                        if ui
                                                            .button("Spawn 5000 Random Agents")
                                                            .clicked()
                                                            && !state.is_paused
                                                        {
                                                            state.queue_random_spawns(5000);
                                                        }

                                                        ui.separator();
                                                        ui.heading("Energy & Costs");
                                                        ui.add(
                                                            egui::Slider::new(&mut state.food_power, 0.0..=10.0)
                                                                .text("Food Power"),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(
                                                                &mut state.amino_maintenance_cost,
                                                                0.0..=0.01,
                                                            )
                                                            .text("Amino Maintenance Cost")
                                                            .logarithmic(true)
                                                            .step_by(0.0001),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.pairing_cost, 0.0..=1.0)
                                                                .text("Pairing Cost per Increment")
                                                                .step_by(0.01),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.poison_power, 0.0..=10.0)
                                                                .text("Poison Power"),
                                                        );

                                                        ui.separator();
                                                        ui.heading("Physics");
                                                        ui.add(
                                                            egui::Slider::new(
                                                                &mut state.repulsion_strength,
                                                                0.0..=100.0,
                                                            )
                                                            .text("Obstacle / Terrain Force"),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(
                                                                &mut state.agent_repulsion_strength,
                                                                0.0..=10.0,
                                                            )
                                                            .text("Agent-Agent Repulsion"),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(
                                                                &mut state.agent_repulsion_strength,
                                                                0.0..=10.0,
                                                            )
                                                            .text("Agent-Agent Repulsion"),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.vector_force_power, 0.0..=100.0)
                                                                .text("Vector Force Power"),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.vector_force_x, -1.0..=1.0)
                                                                .text("Vector Force X"),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.vector_force_y, -1.0..=1.0)
                                                                .text("Vector Force Y"),
                                                        );
                                                        ui.label("Constant force applied to all agents (wind/gravity effect)");

                                                        ui.separator();
                                                        ui.heading("Trail Layer Controls");
                                                        ui.add(
                                                            egui::Slider::new(
                                                                &mut state.trail_diffusion,
                                                                0.0..=1.0,
                                                            )
                                                            .text("Trail Diffusion"),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(
                                                                &mut state.trail_decay,
                                                                0.9..=1.0,
                                                            )
                                                            .text("Trail Fade Rate"),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(
                                                                &mut state.trail_opacity,
                                                                0.0..=2.0,
                                                            )
                                                            .text("Trail Opacity"),
                                                        );
                                                        ui.checkbox(&mut state.trail_show, "Show Trail Only");
                                                    });
                                                }
                                                2 => {
                                                    // Environment tab
                                                    egui::ScrollArea::vertical().show(ui, |ui| {
                                                        ui.heading("Environment Scheduling");
                                                        ui.add(
                                                            egui::Slider::new(&mut state.diffusion_interval, 1..=64)
                                                                .text("Diffuse every N steps"),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.slope_interval, 1..=64)
                                                                .text("Rebuild slope every N steps"),
                                                        );

                                                        ui.separator();
                                                        ui.heading("Alpha Controls");
                                                        ui.horizontal(|ui| {
                                                            if ui.button("Set Flat 0.0").clicked() {
                                                                state.generate_map(1, 0, 0.0, 0);
                                                            }
                                                            if ui.button("Set Flat 0.5").clicked() {
                                                                state.generate_map(1, 0, 0.5, 0);
                                                            }
                                                            if ui.button("Generate Noise").clicked() {
                                                                let seed = rand::random::<u32>();
                                                                state.generate_map(1, 1, 0.0, seed);
                                                            }
                                                        });
                                                        ui.add(
                                                            egui::Slider::new(&mut state.alpha_noise_scale, 0.1..=10.0)
                                                                .logarithmic(true)
                                                                .text("Alpha Noise Scale"),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.noise_power, 0.1..=5.0)
                                                                .text("Noise Power (Contrast)"),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.alpha_blur, 0.0..=0.1)
                                                                .text("Distribution Blur"),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.alpha_slope_bias, -10.0..=10.0)
                                                                .text("Slope Bias"),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.alpha_multiplier, 0.0..=0.01)
                                                                .text("Rain Probability"),
                                                        );
                                                        ui.checkbox(&mut state.rain_debug_visual, "≡ƒÄ¿ Show Rain Pattern");
                                                        if state.rain_debug_visual {
                                                            ui.label("≡ƒƒó Green = Alpha (food) | ≡ƒö┤ Red = Beta (poison)");
                                                        }
                                                        ui.add(
                                                            egui::Slider::new(
                                                                &mut state.chemical_slope_scale_alpha,
                                                                0.0..=1.0,
                                                            )
                                                            .text("Alpha Slope Mix"),
                                                        );
                                                        ui.horizontal(|ui| {
                                                            if ui.button("Load Alpha Rain Map").clicked() {
                                                                if let Some(path) = rfd::FileDialog::new()
                                                                    .add_filter("Images", &["png", "jpg", "jpeg", "bmp"])
                                                                    .pick_file()
                                                                {
                                                                    if let Err(err) =
                                                                        state.load_alpha_rain_map(&path)
                                                                    {
                                                                        eprintln!(
                                                                            "Failed to load alpha rain map {}: {err:?}",
                                                                            path.display()
                                                                        );
                                                                    }
                                                                }
                                                            }
                                                            if ui.button("Clear").clicked() {
                                                                state.clear_alpha_rain_map();
                                                            }
                                                        });
                                                        ui.label(match state.alpha_rain_map_path.as_ref() {
                                                            Some(path) => {
                                                                let name = path
                                                                    .file_name()
                                                                    .and_then(|n| n.to_str())
                                                                    .map(|s| s.to_string())
                                                                    .unwrap_or_else(|| path.display().to_string());
                                                                format!("Alpha rain map: {}", name)
                                                            }
                                                            None => "Alpha rain map: uniform".to_string(),
                                                        });
                                                        if let Some(tex_id) = state.alpha_rain_texture_id(ui.ctx()) {
                                                            ui.add(
                                                                egui::Image::new((
                                                                    tex_id,
                                                                    egui::Vec2::splat(RAIN_THUMB_SIZE as f32),
                                                                ))
                                                                .bg_fill(egui::Color32::DARK_GRAY),
                                                            )
                                                            .on_hover_text(
                                                                "Grayscale preview of alpha rain probability (white = more rain)",
                                                            );
                                                        }

                                                        ui.separator();
                                                        ui.heading("Beta Controls");
                                                        ui.horizontal(|ui| {
                                                            if ui.button("Set Flat 0.0").clicked() {
                                                                state.generate_map(2, 0, 0.0, 0);
                                                            }
                                                            if ui.button("Set Flat 0.5").clicked() {
                                                                state.generate_map(2, 0, 0.5, 0);
                                                            }
                                                            if ui.button("Generate Noise").clicked() {
                                                                let seed = rand::random::<u32>();
                                                                state.generate_map(2, 1, 0.0, seed);
                                                            }
                                                        });
                                                        ui.add(
                                                            egui::Slider::new(&mut state.beta_noise_scale, 0.1..=10.0)
                                                                .logarithmic(true)
                                                                .text("Beta Noise Scale"),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.beta_blur, 0.0..=0.1)
                                                                .text("Distribution Blur"),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.beta_slope_bias, -10.0..=10.0)
                                                                .text("Slope Bias"),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.beta_multiplier, 0.0..=0.01)
                                                                .text("Rain Probability"),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(
                                                                &mut state.chemical_slope_scale_beta,
                                                                0.0..=1.0,
                                                            )
                                                            .text("Beta Slope Mix"),
                                                        );
                                                        ui.horizontal(|ui| {
                                                            if ui.button("Load Beta Rain Map").clicked() {
                                                                if let Some(path) = rfd::FileDialog::new()
                                                                    .add_filter("Images", &["png", "jpg", "jpeg", "bmp"])
                                                                    .pick_file()
                                                                {
                                                                    if let Err(err) = state.load_beta_rain_map(&path) {
                                                                        eprintln!(
                                                                            "Failed to load beta rain map {}: {err:?}",
                                                                            path.display()
                                                                        );
                                                                    }
                                                                }
                                                            }
                                                            if ui.button("Clear").clicked() {
                                                                state.clear_beta_rain_map();
                                                            }
                                                        });
                                                        ui.label(match state.beta_rain_map_path.as_ref() {
                                                            Some(path) => {
                                                                let name = path
                                                                    .file_name()
                                                                    .and_then(|n| n.to_str())
                                                                    .map(|s| s.to_string())
                                                                    .unwrap_or_else(|| path.display().to_string());
                                                                format!("Beta rain map: {}", name)
                                                            }
                                                            None => "Beta rain map: uniform".to_string(),
                                                        });
                                                        if let Some(tex_id) = state.beta_rain_texture_id(ui.ctx()) {
                                                            ui.add(
                                                                egui::Image::new((
                                                                    tex_id,
                                                                    egui::Vec2::splat(RAIN_THUMB_SIZE as f32),
                                                                ))
                                                                .bg_fill(egui::Color32::DARK_GRAY),
                                                            )
                                                            .on_hover_text(
                                                                "Grayscale preview of beta rain probability (white = more rain)",
                                                            );
                                                        }

                                                        ui.separator();
                                                        ui.heading("Gamma Controls");
                                                        ui.horizontal(|ui| {
                                                            if ui.button("Set Flat 0.0").clicked() {
                                                                state.generate_map(3, 0, 0.0, 0);
                                                            }
                                                            if ui.button("Set Flat 0.5").clicked() {
                                                                state.generate_map(3, 0, 0.5, 0);
                                                            }
                                                            if ui.button("Generate Noise").clicked() {
                                                                let seed = rand::random::<u32>();
                                                                state.generate_map(3, 1, 0.0, seed);
                                                            }
                                                        });
                                                        ui.add(
                                                            egui::Slider::new(&mut state.gamma_noise_scale, 0.1..=10.0)
                                                                .logarithmic(true)
                                                                .text("Gamma Noise Scale"),
                                                        );
                                                        if ui.button("Load Gamma Image").clicked() {
                                                            if let Some(path) = rfd::FileDialog::new()
                                                                .add_filter("Images", &["png", "jpg", "jpeg", "bmp"])
                                                                .pick_file()
                                                            {
                                                                if let Err(err) = state.load_gamma_image(&path) {
                                                                    eprintln!(
                                                                        "Failed to load gamma image {}: {err:?}",
                                                                        path.display()
                                                                    );
                                                                }
                                                            }
                                                        }
                                                        ui.checkbox(&mut state.gamma_hidden, "Hide Gamma in Composite");
                                                        ui.checkbox(&mut state.slope_debug_visual, "Show Raw Slopes");
                                                        ui.add(
                                                            egui::Slider::new(&mut state.gamma_blur, 0.0..=0.1)
                                                                .text("Gamma Blur"),
                                                        );
                                                        let min_changed = ui
                                                            .add(
                                                                egui::Slider::new(
                                                                    &mut state.gamma_vis_min,
                                                                    0.0..=1.0,
                                                                )
                                                                .text("Gamma Min"),
                                                            )
                                                            .changed();
                                                        let max_changed = ui
                                                            .add(
                                                                egui::Slider::new(
                                                                    &mut state.gamma_vis_max,
                                                                    0.0..=1.0,
                                                                )
                                                                .text("Gamma Max"),
                                                            )
                                                            .changed();
                                                        if state.gamma_vis_min >= state.gamma_vis_max {
                                                            state.gamma_vis_max =
                                                                (state.gamma_vis_min + 0.001).min(1.0);
                                                            state.gamma_vis_min =
                                                                (state.gamma_vis_max - 0.001).max(0.0);
                                                        } else if min_changed || max_changed {
                                                            state.gamma_vis_min =
                                                                state.gamma_vis_min.clamp(0.0, 1.0);
                                                            state.gamma_vis_max =
                                                                state.gamma_vis_max.clamp(0.0, 1.0);
                                                        }
                                                        ui.add(
                                                            egui::Slider::new(&mut state.prop_wash_strength, 0.0..=5.0)
                                                                .text("Prop Wash Strength"),
                                                        );
                                                        ui.label(
                                                            "Slope force scales with the Obstacle / Terrain Force slider.",
                                                        );
                                                    });
                                                }
                                                3 => {
                                                    // Evolution tab
                                                    egui::ScrollArea::vertical().show(ui, |ui| {
                                                        ui.heading("Rain Cycling");

                                                        let current_alpha = state.alpha_rain_history.back().copied().unwrap_or(state.alpha_multiplier);
                                                        let current_beta = state.beta_rain_history.back().copied().unwrap_or(state.beta_multiplier);

                                                        ui.separator();
                                                        ui.heading("Alpha Rain");
                                                        ui.label(egui::RichText::new(format!("Current Alpha Rain: {:.6}", current_alpha)).color(Color32::GREEN));
                                                        let mut alpha_var_percent = state.alpha_rain_variation * 100.0;
                                                        if ui.add(egui::Slider::new(&mut alpha_var_percent, 0.0..=100.0).text("Variation %")).changed() {
                                                            state.alpha_rain_variation = alpha_var_percent / 100.0;
                                                        }
                                                        ui.add(egui::Slider::new(&mut state.alpha_rain_phase, 0.0..=std::f32::consts::PI * 2.0).text("Phase"));
                                                        ui.add(egui::Slider::new(&mut state.alpha_rain_freq, 0.0..=10.0).text("Freq (cycles/1k)"));

                                                        ui.separator();
                                                        ui.heading("Beta Rain");
                                                        ui.label(egui::RichText::new(format!("Current Beta Rain: {:.6}", current_beta)).color(Color32::RED));
                                                        let mut beta_var_percent = state.beta_rain_variation * 100.0;
                                                        if ui.add(egui::Slider::new(&mut beta_var_percent, 0.0..=100.0).text("Variation %")).changed() {
                                                            state.beta_rain_variation = beta_var_percent / 100.0;
                                                        }
                                                        ui.add(egui::Slider::new(&mut state.beta_rain_phase, 0.0..=std::f32::consts::PI * 2.0).text("Phase"));
                                                        ui.add(egui::Slider::new(&mut state.beta_rain_freq, 0.0..=10.0).text("Freq (cycles/1k)"));

                                                        ui.separator();
                                                        ui.heading("Rain Projection (Future)");

                                                        // Apply difficulty to base values for accurate projection
                                                        let base_alpha = state.difficulty.alpha_rain.apply_to(state.alpha_multiplier, false);
                                                        let base_beta = state.difficulty.beta_rain.apply_to(state.beta_multiplier, true);

                                                        let time = state.epoch as f64;
                                                        let points = 500;
                                                        let alpha_points: PlotPoints = (0..points).map(|i| {
                                                            let t = time + i as f64;
                                                            let freq = state.alpha_rain_freq as f64 / 1000.0;
                                                            let phase = state.alpha_rain_phase as f64;
                                                            let sin_val = (t * freq * 2.0 * std::f64::consts::PI + phase).sin();
                                                            let val = base_alpha as f64 * (1.0 + sin_val * state.alpha_rain_variation as f64).max(0.0);
                                                            [i as f64, val]
                                                        }).collect();

                                                        let beta_points: PlotPoints = (0..points).map(|i| {
                                                            let t = time + i as f64;
                                                            let freq = state.beta_rain_freq as f64 / 1000.0;
                                                            let phase = state.beta_rain_phase as f64;
                                                            let sin_val = (t * freq * 2.0 * std::f64::consts::PI + phase).sin();
                                                            let val = base_beta as f64 * (1.0 + sin_val * state.beta_rain_variation as f64).max(0.0);
                                                            [i as f64, val]
                                                        }).collect();

                                                        Plot::new("rain_plot_future")
                                                            .view_aspect(2.0)
                                                            .auto_bounds([false, true].into())
                                                            .show(ui, |plot_ui| {
                                                                plot_ui.line(Line::new(alpha_points).name("Alpha").color(Color32::GREEN));
                                                                plot_ui.line(Line::new(beta_points).name("Beta").color(Color32::RED));
                                                            });
                                                    });
                                                }
                                                4 => {
                                                    // Difficulty tab
                                                    egui::ScrollArea::vertical().show(ui, |ui| {
                                                        ui.heading("Auto Difficulty Settings");
                                                        ui.label("Automatically adjust parameters based on population count.");
                                                        ui.label(format!("Current Population: {}", state.alive_count));

                                                        if ui.button("Reset All Difficulty Levels").clicked() {
                                                            state.difficulty.food_power.difficulty_level = 0;
                                                            state.difficulty.food_power.last_adjustment_epoch = 0;
                                                            state.difficulty.poison_power.difficulty_level = 0;
                                                            state.difficulty.poison_power.last_adjustment_epoch = 0;
                                                            state.difficulty.spawn_prob.difficulty_level = 0;
                                                            state.difficulty.spawn_prob.last_adjustment_epoch = 0;
                                                            state.difficulty.death_prob.difficulty_level = 0;
                                                            state.difficulty.death_prob.last_adjustment_epoch = 0;
                                                            state.difficulty.alpha_rain.difficulty_level = 0;
                                                            state.difficulty.alpha_rain.last_adjustment_epoch = 0;
                                                            state.difficulty.beta_rain.difficulty_level = 0;
                                                            state.difficulty.beta_rain.last_adjustment_epoch = 0;
                                                        }

                                                        let current_epoch = state.epoch;
                                                        let draw_param = |ui: &mut egui::Ui, param: &mut AutoDifficultyParam, name: &str, current_val: f32, current_epoch: u64| {
                                                            ui.separator();
                                                            ui.heading(name);
                                                            ui.horizontal(|ui| {
                                                                ui.checkbox(&mut param.enabled, "Enabled");
                                                                ui.label(egui::RichText::new(format!("Level: {}", param.difficulty_level)).strong());
                                                                ui.label(format!("Current Val: {:.5}", current_val));
                                                            });

                                                            if param.enabled {
                                                                ui.horizontal(|ui| {
                                                                    ui.label("Min Pop:");
                                                                    ui.add(egui::DragValue::new(&mut param.min_threshold).speed(10.0));
                                                                    ui.label("Max Pop:");
                                                                    ui.add(egui::DragValue::new(&mut param.max_threshold).speed(10.0));
                                                                });
                                                                ui.horizontal(|ui| {
                                                                    ui.label("Adjust %:");
                                                                    ui.add(egui::Slider::new(&mut param.adjustment_percent, 0.0..=100.0));
                                                                });
                                                                ui.horizontal(|ui| {
                                                                    ui.label("Cooldown (epochs):");
                                                                    ui.add(egui::DragValue::new(&mut param.cooldown_epochs));
                                                                });

                                                                let epochs_passed = current_epoch.saturating_sub(param.last_adjustment_epoch);
                                                                if epochs_passed < param.cooldown_epochs {
                                                                    let remaining = param.cooldown_epochs - epochs_passed;
                                                                    ui.label(format!("Cooldown: {} epochs", remaining));
                                                                    ui.add(egui::ProgressBar::new(epochs_passed as f32 / param.cooldown_epochs as f32));
                                                                }
                                                            }
                                                        };

                                                        // Calculate effective values with difficulty applied
                                                        let effective_food_power = state.difficulty.food_power.apply_to(state.food_power, false);
                                                        let effective_poison_power = state.difficulty.poison_power.apply_to(state.poison_power, true);
                                                        let effective_spawn_prob = state.difficulty.spawn_prob.apply_to(state.spawn_probability, false);
                                                        let effective_death_prob = state.difficulty.death_prob.apply_to(state.death_probability, true);
                                                        let effective_alpha_rain = state.difficulty.alpha_rain.apply_to(state.alpha_multiplier, false);
                                                        let effective_beta_rain = state.difficulty.beta_rain.apply_to(state.beta_multiplier, true);

                                                        draw_param(ui, &mut state.difficulty.food_power, "Food Power (Harder = Less)", effective_food_power, current_epoch);
                                                        draw_param(ui, &mut state.difficulty.poison_power, "Poison Power (Harder = More)", effective_poison_power, current_epoch);
                                                        draw_param(ui, &mut state.difficulty.spawn_prob, "Spawn Prob (Harder = Less)", effective_spawn_prob, current_epoch);
                                                        draw_param(ui, &mut state.difficulty.death_prob, "Death Prob (Harder = More)", effective_death_prob, current_epoch);
                                                        draw_param(ui, &mut state.difficulty.alpha_rain, "Alpha Rain (Harder = Less)", effective_alpha_rain, current_epoch);
                                                        draw_param(ui, &mut state.difficulty.beta_rain, "Beta Rain (Harder = More)", effective_beta_rain, current_epoch);
                                                    });
                                                }
                                                5 => {
                                                    // Visualization tab
                                                    egui::ScrollArea::vertical().show(ui, |ui| {
                                                        ui.heading("Visualization Settings");

                                                        ui.separator();
                                                        ui.label("Grid Interpolation");
                                                        ui.horizontal(|ui| {
                                                            ui.radio_value(&mut state.grid_interpolation, 0, "Nearest (Pixelated)");
                                                            ui.radio_value(&mut state.grid_interpolation, 1, "Bilinear (Smooth)");
                                                            ui.radio_value(&mut state.grid_interpolation, 2, "Bicubic (Smoothest)");
                                                        });

                                                        ui.separator();
                                                        ui.label("Background");
                                                        ui.horizontal(|ui| {
                                                            ui.label("Background Color:");
                                                            let mut bg_color = [
                                                                (state.background_color[0] * 255.0) as u8,
                                                                (state.background_color[1] * 255.0) as u8,
                                                                (state.background_color[2] * 255.0) as u8,
                                                            ];
                                                            if ui.color_edit_button_srgb(&mut bg_color).changed() {
                                                                state.background_color = [
                                                                    bg_color[0] as f32 / 255.0,
                                                                    bg_color[1] as f32 / 255.0,
                                                                    bg_color[2] as f32 / 255.0,
                                                                ];
                                                            }
                                                        });

                                                        ui.separator();
                                                        ui.heading("Slope Lighting");
                                                        ui.checkbox(&mut state.slope_lighting, "Enable Slope Lighting");
                                                        if state.slope_lighting {
                                                            ui.add(
                                                                egui::Slider::new(&mut state.slope_lighting_strength, 0.0..=5.0)
                                                                    .text("Lighting Strength")
                                                            );
                                                        }
                                                        ui.label("Light Effect:");
                                                        ui.horizontal(|ui| {
                                                            ui.radio_value(&mut state.slope_blend_mode, 0, "None");
                                                            ui.radio_value(&mut state.slope_blend_mode, 1, "Hard Light");
                                                            ui.radio_value(&mut state.slope_blend_mode, 2, "Soft Light");
                                                        });

                                                        ui.separator();
                                                        ui.heading("3D Shading Light");
                                                        ui.label("Light Direction:");
                                                        ui.add(
                                                            egui::Slider::new(&mut state.light_direction[0], -1.0..=1.0)
                                                                .text("Light X")
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.light_direction[1], -1.0..=1.0)
                                                                .text("Light Y")
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.light_direction[2], -1.0..=1.0)
                                                                .text("Light Z")
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.light_power, 0.0..=2.0)
                                                                .text("Light Intensity")
                                                        );

                                                        ui.separator();
                                                        ui.heading("Alpha Channel");
                                                        ui.checkbox(&mut state.alpha_show, "Show Alpha");
                                                        ui.label("Blend Mode:");
                                                        ui.horizontal(|ui| {
                                                            ui.radio_value(&mut state.alpha_blend_mode, 0, "Additive");
                                                            ui.radio_value(&mut state.alpha_blend_mode, 1, "Multiply");
                                                        });
                                                        ui.horizontal(|ui| {
                                                            ui.label("Color:");
                                                            let mut alpha_color = [
                                                                (state.alpha_color[0] * 255.0) as u8,
                                                                (state.alpha_color[1] * 255.0) as u8,
                                                                (state.alpha_color[2] * 255.0) as u8,
                                                            ];
                                                            if ui.color_edit_button_srgb(&mut alpha_color).changed() {
                                                                state.alpha_color = [
                                                                    alpha_color[0] as f32 / 255.0,
                                                                    alpha_color[1] as f32 / 255.0,
                                                                    alpha_color[2] as f32 / 255.0,
                                                                ];
                                                            }
                                                        });
                                                        ui.add(
                                                            egui::Slider::new(&mut state.alpha_gamma_adjust, 0.1..=5.0)
                                                                .text("Gamma Adjustment")
                                                                .logarithmic(true)
                                                        );

                                                        ui.separator();
                                                        ui.heading("Beta Channel");
                                                        ui.checkbox(&mut state.beta_show, "Show Beta");
                                                        ui.label("Blend Mode:");
                                                        ui.horizontal(|ui| {
                                                            ui.radio_value(&mut state.beta_blend_mode, 0, "Additive");
                                                            ui.radio_value(&mut state.beta_blend_mode, 1, "Multiply");
                                                        });
                                                        ui.horizontal(|ui| {
                                                            ui.label("Color:");
                                                            let mut beta_color = [
                                                                (state.beta_color[0] * 255.0) as u8,
                                                                (state.beta_color[1] * 255.0) as u8,
                                                                (state.beta_color[2] * 255.0) as u8,
                                                            ];
                                                            if ui.color_edit_button_srgb(&mut beta_color).changed() {
                                                                state.beta_color = [
                                                                    beta_color[0] as f32 / 255.0,
                                                                    beta_color[1] as f32 / 255.0,
                                                                    beta_color[2] as f32 / 255.0,
                                                                ];
                                                            }
                                                        });
                                                        ui.add(
                                                            egui::Slider::new(&mut state.beta_gamma_adjust, 0.1..=5.0)
                                                                .text("Gamma Adjustment")
                                                                .logarithmic(true)
                                                        );

                                                        ui.separator();
                                                        ui.heading("Gamma Channel");
                                                        ui.checkbox(&mut state.gamma_show, "Show Gamma");
                                                        ui.label("Blend Mode:");
                                                        ui.horizontal(|ui| {
                                                            ui.radio_value(&mut state.gamma_blend_mode, 0, "Additive");
                                                            ui.radio_value(&mut state.gamma_blend_mode, 1, "Multiply");
                                                        });
                                                        ui.horizontal(|ui| {
                                                            ui.label("Color:");
                                                            let mut gamma_color = [
                                                                (state.gamma_color[0] * 255.0) as u8,
                                                                (state.gamma_color[1] * 255.0) as u8,
                                                                (state.gamma_color[2] * 255.0) as u8,
                                                            ];
                                                            if ui.color_edit_button_srgb(&mut gamma_color).changed() {
                                                                state.gamma_color = [
                                                                    gamma_color[0] as f32 / 255.0,
                                                                    gamma_color[1] as f32 / 255.0,
                                                                    gamma_color[2] as f32 / 255.0,
                                                                ];
                                                            }
                                                        });
                                                        ui.add(
                                                            egui::Slider::new(&mut state.gamma_gamma_adjust, 0.1..=5.0)
                                                                .text("Gamma Adjustment")
                                                                .logarithmic(true)
                                                        );

                                                        ui.separator();
                                                        ui.heading("Agents");
                                                        ui.label("Blend Mode:");
                                                        ui.horizontal(|ui| {
                                                            ui.radio_value(&mut state.agent_blend_mode, 0, "Comp");
                                                            ui.radio_value(&mut state.agent_blend_mode, 1, "Add");
                                                        });
                                                        ui.horizontal(|ui| {
                                                            ui.radio_value(&mut state.agent_blend_mode, 2, "Subtract");
                                                            ui.radio_value(&mut state.agent_blend_mode, 3, "Multiply");
                                                        });
                                                        ui.horizontal(|ui| {
                                                            ui.label("Color Tint:");
                                                            let mut agent_color = [
                                                                (state.agent_color[0] * 255.0) as u8,
                                                                (state.agent_color[1] * 255.0) as u8,
                                                                (state.agent_color[2] * 255.0) as u8,
                                                            ];
                                                            if ui.color_edit_button_srgb(&mut agent_color).changed() {
                                                                state.agent_color = [
                                                                    agent_color[0] as f32 / 255.0,
                                                                    agent_color[1] as f32 / 255.0,
                                                                    agent_color[2] as f32 / 255.0,
                                                                ];
                                                            }
                                                        });
                                                        ui.add(
                                                            egui::Slider::new(&mut state.agent_color_blend, 0.0..=1.0)
                                                                .text("Color Blend")
                                                                .clamp_to_range(true)
                                                        ).on_hover_text("0.0 = amino acid colors only, 1.0 = agent color only");

                                                        ui.separator();
                                                        ui.label("Motion Blur / Trail:");
                                                        ui.add(
                                                            egui::Slider::new(&mut state.agent_trail_decay, 0.0..=1.0)
                                                                .text("Trail Decay")
                                                                .clamp_to_range(true)
                                                        ).on_hover_text("0.0 = persistent trail, 1.0 = instant clear (no trail)");

                                                        ui.separator();
                                                        ui.heading("Fluid Simulation");
                                                        ui.checkbox(&mut state.fluid_show, "Show Fluid Overlay")
                                                            .on_hover_text("Visualize propeller-driven fluid simulation (experimental)");
                                                    });
                                                }
                                                _ => {}
                                            }
                            });

                                   } // End ui_visible check

                                    // Inspector overlay - energy and pairing bars (always visible when agent selected)
                                    if let Some(agent) = &state.selected_agent_data {
                                        let screen_width = state.surface_config.width as f32;
                                        let screen_height = state.surface_config.height as f32;
                                        let inspector_x = screen_width - 300.0;

                                        // Position bars below the agent preview window
                                        // Agent preview is 280x280, positioned at (10, screen_height - 280 - 10)
                                        // Bars should go at (10, screen_height - 280 - 10 + 280 + 10) = (10, screen_height - 10)
                                        let bars_y = screen_height - 80.0; // 80px from bottom to leave room for bars

                                        egui::Area::new(egui::Id::new("inspector_bars"))
                                            .fixed_pos(egui::pos2(inspector_x + 10.0, bars_y))
                                            .show(ctx, |ui| {
                                                ui.set_width(280.0);

                                                // Energy bar
                                                ui.vertical(|ui| {
                                                    let energy_ratio = (agent.energy / agent.energy_capacity).clamp(0.0, 1.0);
                                                    let energy_color = if energy_ratio > 0.5 {
                                                        egui::Color32::from_rgb(0, 200, 0)
                                                    } else if energy_ratio > 0.25 {
                                                        egui::Color32::from_rgb(200, 200, 0)
                                                    } else {
                                                        egui::Color32::from_rgb(200, 0, 0)
                                                    };

                                                    ui.label(egui::RichText::new(
                                                        format!("Energy: {:.1}/{:.1}", agent.energy, agent.energy_capacity)
                                                    ).color(egui::Color32::WHITE));

                                                    let progress_bar = egui::ProgressBar::new(energy_ratio)
                                                        .fill(energy_color)
                                                        .animate(false);
                                                    ui.add(progress_bar);
                                                });

                                                ui.add_space(4.0);

                                                // Pairing state bar
                                                ui.vertical(|ui| {
                                                    let full_pairs = agent.gene_length; // Use actual gene length (non-X bases)
                                                    let pairing_ratio = if full_pairs > 0 {
                                                        (agent.pairing_counter as f32 / full_pairs as f32).clamp(0.0, 1.0)
                                                    } else {
                                                        0.0
                                                    };
                                                    let pairing_color = egui::Color32::from_rgb(100, 150, 255);

                                                    ui.label(egui::RichText::new(
                                                        format!("Pairing: {}/{}", agent.pairing_counter, full_pairs)
                                                    ).color(egui::Color32::WHITE));

                                                    let progress_bar = egui::ProgressBar::new(pairing_ratio)
                                                        .fill(pairing_color)
                                                        .animate(false);
                                                    ui.add(progress_bar);
                                                });
                                            });
                                    }
                                });

                                // Handle platform output
                                egui_state.handle_platform_output(&window, full_output.platform_output);
                                state.persist_settings_if_changed();

                                // Always render the simulation (cheap) and UI
                                // In fast mode, we're just running update() faster between renders

                                // Tessellate and prepare render data
                                let clipped_primitives = egui_state.egui_ctx()
                                    .tessellate(full_output.shapes, full_output.pixels_per_point);
                                let screen_descriptor = ScreenDescriptor {
                                    size_in_pixels: [
                                        state.surface_config.width,
                                        state.surface_config.height,
                                    ],
                                    pixels_per_point: window.scale_factor() as f32,
                                };

                                // Render with egui
                                match state.render(
                                    clipped_primitives,
                                    full_output.textures_delta,
                                    screen_descriptor,
                                ) {
                                    Ok(_) => {}
                                    Err(wgpu::SurfaceError::Lost) => state.resize(window.inner_size()),
                                    Err(wgpu::SurfaceError::Outdated) => {}
                                    Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
                                    Err(e) => eprintln!("{:?}", e),
                                }
                            }

                            if reset_requested {
                                reset_simulation_state(&mut state, &window, &mut egui_state);
                                window.request_redraw();
                            }

                            if let Some(gpu_state) = &mut state {
                                if gpu_state.snapshot_save_requested {
                                    gpu_state.snapshot_save_requested = false;

                                    // Open file dialog
                                    if let Some(path) = rfd::FileDialog::new()
                                        .add_filter("PNG Image", &["png"])
                                        .set_file_name(&format!("ribossome_epoch_{}.png", gpu_state.epoch))
                                        .save_file()
                                    {
                                        match gpu_state.save_snapshot_to_file(&path) {
                                            Ok(_) => println!("✓ Snapshot saved to: {}", path.display()),
                                            Err(e) => eprintln!("❌ Failed to save snapshot: {}", e),
                                        }
                                    }
                                }

                                if gpu_state.snapshot_load_requested {
                                    gpu_state.snapshot_load_requested = false;

                                    // Open file dialog
                                    if let Some(path) = rfd::FileDialog::new()
                                        .add_filter("PNG Image", &["png"])
                                        .pick_file()
                                    {
                                        // Reset simulation state before loading
                                        reset_simulation_state(&mut state, &window, &mut egui_state);
                                        if let Some(gpu_state) = state.as_mut() {
                                            match gpu_state.load_snapshot_from_file(&path) {
                                                Ok(_) => println!("✓ Snapshot loaded from: {}", path.display()),
                                                Err(e) => eprintln!("❌ Failed to load snapshot: {}", e),
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                } else {
                    // egui consumed the event, but we still need to handle some
                    match event {
                        WindowEvent::CloseRequested => {
                            if let Some(mut existing) = state.take() {
                                existing.destroy_resources();
                            }
                            target.exit();
                        }
                        WindowEvent::Resized(physical_size) => {
                            if let Some(state) = state.as_mut() {
                                state.resize(physical_size);
                            }
                        }
                        WindowEvent::DroppedFile(path) => {
                            // Handle drag-and-drop file loading
                            if let Some(ext) = path.extension() {
                                if ext == "png" {
                                    if let Some(gpu_state) = state.as_mut() {
                                        match gpu_state.load_snapshot_from_file(&path) {
                                            Ok(_) => println!("Γ£ô Snapshot loaded from dropped file: {}", path.display()),
                                            Err(e) => eprintln!("Γ£ù Failed to load dropped snapshot: {}", e),
                                        }
                                    }
                                }
                            }
                        }
                        WindowEvent::RedrawRequested => {
                            let mut reset_requested = false;

                            if let Some(state) = state.as_mut() {
                                // Frame rate limiting
                                if let Some(target_frame_time) = state.frame_time_cap() {
                                    let elapsed = state.last_frame_time.elapsed();
                                    if elapsed < target_frame_time {
                                        std::thread::sleep(target_frame_time - elapsed);
                                    }
                                }
                                let now = std::time::Instant::now();
                                let frame_dt = (now - state.last_frame_time).as_secs_f32().clamp(0.001, 0.1);
                                state.last_frame_time = now;

                                state.update(true, frame_dt); // Always do readbacks when not in fast mode

                                // Build egui UI
                                let raw_input = egui_state.take_egui_input(&window);
                                let full_output = egui_state.egui_ctx().run(raw_input, |ctx| {
                                    egui::Window::new("Simulation Controls")
                                        .default_pos([10.0, 10.0])
                                        .default_size([300.0, 400.0])
                                        .show(ctx, |ui| {
                                            ui.heading("Camera");
                                            ui.add(
                                                egui::Slider::new(&mut state.camera_zoom, 0.1..=2000.0)
                                                    .text("Zoom").logarithmic(true),
                                            );
                                            if ui.button("Reset Camera (R)").clicked() {
                                                state.camera_zoom = 1.0;
                                                state.camera_pan = [SIM_SIZE / 2.0, SIM_SIZE / 2.0];
                                            }

                                            ui.separator();
                                            ui.heading("Info");
                                            ui.label(format!("Agents: {}", state.agent_count));
                                            ui.label(format!("Living Agents: {}", state.alive_count));
                                            if ui.button("Reset Simulation").clicked() {
                                                reset_requested = true;
                                            }
                                            if ui.button("Spawn 5000 Random Agents").clicked() && !state.is_paused {
                                                state.queue_random_spawns(5000);
                                            }

                                            ui.separator();
                                            ui.heading("Snapshot");
                                            if ui.button("💾 Save Snapshot (PNG)").clicked() {
                                                state.snapshot_save_requested = true;
                                            }
                                            ui.label("Saves environment + up to 5000 agents (random sample)");

                                            ui.label(format!("World: {}x{}", SIM_SIZE as u32, SIM_SIZE as u32));
                                            ui.label("Grid: 2048x2048");
                                            ui.label(
                                                "Morphology responds to\nalpha field in environment",
                                            );

                                            ui.separator();
                                            ui.heading("Visualization");
                                                    let min_changed = ui
                                                        .add(
                                                            egui::Slider::new(
                                                                &mut state.gamma_vis_min,
                                                                0.0..=1.0,
                                                            )
                                                            .text("Gamma Min"),
                                                        )
                                                        .changed();
                                                    let max_changed = ui
                                                        .add(
                                                            egui::Slider::new(
                                                                &mut state.gamma_vis_max,
                                                                0.0..=1.0,
                                                            )
                                                            .text("Gamma Max"),
                                                        )
                                                        .changed();
                                                    if state.gamma_vis_min >= state.gamma_vis_max {
                                                        state.gamma_vis_max =
                                                            (state.gamma_vis_min + 0.001).min(1.0);
                                                        state.gamma_vis_min =
                                                            (state.gamma_vis_max - 0.001).max(0.0);
                                                    } else if min_changed || max_changed {
                                                        state.gamma_vis_min =
                                                            state.gamma_vis_min.clamp(0.0, 1.0);
                                                        state.gamma_vis_max =
                                                            state.gamma_vis_max.clamp(0.0, 1.0);
                                                    }                                            ui.separator();
                                            ui.collapsing("Evolution", |ui| {
                                                ui.label("Rain Cycling");
                                                ui.add(egui::Slider::new(&mut state.alpha_rain_variation, 0.0..=1.0).text("Alpha Var %"));
                                                ui.add(egui::Slider::new(&mut state.beta_rain_variation, 0.0..=1.0).text("Beta Var %"));

                                                ui.label("Alpha Cycle");
                                                ui.add(egui::Slider::new(&mut state.alpha_rain_phase, 0.0..=std::f32::consts::PI * 2.0).text("Phase"));
                                                ui.add(egui::Slider::new(&mut state.alpha_rain_freq, 0.0..=100.0).text("Freq (cycles/1k)"));

                                                ui.label("Beta Cycle");
                                                ui.add(egui::Slider::new(&mut state.beta_rain_phase, 0.0..=std::f32::consts::PI * 2.0).text("Phase"));
                                                ui.add(egui::Slider::new(&mut state.beta_rain_freq, 0.0..=100.0).text("Freq (cycles/1k)"));

                                                ui.label("Rain Projection (Future)");
                                                let time = state.epoch as f64;
                                                let points = 500;
                                                let alpha_points: PlotPoints = (0..points).map(|i| {
                                                    let t = time + i as f64;
                                                    let freq = state.alpha_rain_freq as f64 / 1000.0;
                                                    let phase = state.alpha_rain_phase as f64;
                                                    let sin_val = (t * freq * 2.0 * std::f64::consts::PI + phase).sin();
                                                    let val = state.alpha_multiplier as f64 * (1.0 + sin_val * state.alpha_rain_variation as f64).max(0.0);
                                                    [i as f64, val]
                                                }).collect();

                                                let beta_points: PlotPoints = (0..points).map(|i| {
                                                    let t = time + i as f64;
                                                    let freq = state.beta_rain_freq as f64 / 1000.0;
                                                    let phase = state.beta_rain_phase as f64;
                                                    let sin_val = (t * freq * 2.0 * std::f64::consts::PI + phase).sin();
                                                    let val = state.beta_multiplier as f64 * (1.0 + sin_val * state.beta_rain_variation as f64).max(0.0);
                                                    [i as f64, val]
                                                }).collect();

                                                Plot::new("rain_plot")
                                                    .view_aspect(2.0)
                                                    .show(ui, |plot_ui| {
                                                        plot_ui.line(Line::new(alpha_points).name("Alpha").color(Color32::GREEN));
                                                        plot_ui.line(Line::new(beta_points).name("Beta").color(Color32::RED));
                                                    });
                                            });

                                            ui.separator();
                                            ui.heading("Performance");
                                            let mut fps_cap_enabled = matches!(state.current_mode, 0 | 3);
                                            if ui.checkbox(&mut fps_cap_enabled, "Enable FPS Cap").changed() {
                                                if fps_cap_enabled {
                                                    state.set_speed_mode(0);
                                                } else {
                                                    state.set_speed_mode(1);
                                                }
                                            }
                                            if fps_cap_enabled {
                                                let mut slow_mode = state.current_mode == 3;
                                                if ui.checkbox(&mut slow_mode, "Slow (25 FPS)").changed() {
                                                    state.set_speed_mode(if slow_mode { 3 } else { 0 });
                                                }
                                            } else {
                                                ui.label("Running at maximum speed");
                                            }

                                            ui.separator();
                                            ui.label("Controls:");
                                            ui.label("* Mouse: Pan camera (right drag)");
                                            ui.label("* Wheel: Zoom");
                                            ui.label("* WASD: Pan camera");
                                            ui.label("* R: Reset camera");
                                        });
                                });

                                // Handle platform output
                                egui_state.handle_platform_output(&window, full_output.platform_output);
                                state.persist_settings_if_changed();

                                // Tessellate and prepare render data
                                let clipped_primitives = egui_state.egui_ctx()
                                    .tessellate(full_output.shapes, full_output.pixels_per_point);
                                let screen_descriptor = ScreenDescriptor {
                                    size_in_pixels: [
                                        state.surface_config.width,
                                        state.surface_config.height,
                                    ],
                                    pixels_per_point: window.scale_factor() as f32,
                                };

                                // Render with egui
                                match state.render(
                                    clipped_primitives,
                                    full_output.textures_delta,
                                    screen_descriptor,
                                ) {
                                    Ok(_) => {}
                                    Err(wgpu::SurfaceError::Lost) => state.resize(window.inner_size()),
                                    Err(wgpu::SurfaceError::Outdated) => {}
                                    Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
                                    Err(e) => eprintln!("{:?}", e),
                                }
                            }

                            if reset_requested {
                                reset_simulation_state(&mut state, &window, &mut egui_state);
                                window.request_redraw();
                            }
                        }
                        _ => {}
                    }
                }
            }
            Event::AboutToWait => {
                // Always request immediate redraw for max speed
                // Frame limiting happens in RedrawRequested handler
                window.request_redraw();
            }
            _ => {}
        }
    });
}
