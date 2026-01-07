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
use std::io::{BufWriter, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
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

mod naming;

fn pack_f32_uniform(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

// Default resolution values - can be overridden via simulation_settings.json
// Changes require restart to take effect as they affect shader compilation and buffer allocation
const DEFAULT_ENV_GRID_RESOLUTION: u32 = 2048;
const DEFAULT_FLUID_GRID_RESOLUTION: u32 = 512;
const DEFAULT_SPATIAL_GRID_RESOLUTION: u32 = 1024;

// World size calibrated for DEFAULT_ENV_GRID_RESOLUTION.
// For other env grid resolutions, we scale proportionally so that each env cell keeps the same
// world-space footprint.
const SIM_SIZE_AT_DEFAULT_ENV_RES: f32 = 61440.0;

fn sim_size_for_env_res(env_grid_res: u32) -> f32 {
    SIM_SIZE_AT_DEFAULT_ENV_RES * (env_grid_res as f32 / DEFAULT_ENV_GRID_RESOLUTION as f32)
}

// Agent buffer capacity calibrated for DEFAULT_ENV_GRID_RESOLUTION.
// Scale proportionally with resolution to save memory and improve performance at lower resolutions.
const MAX_AGENTS_AT_DEFAULT_ENV_RES: usize = 60_000;

fn max_agents_for_env_res(env_grid_res: u32) -> usize {
    let raw = (MAX_AGENTS_AT_DEFAULT_ENV_RES as f32 * (env_grid_res as f32 / DEFAULT_ENV_GRID_RESOLUTION as f32)) as usize;
    // Round up to nearest multiple of 64 for dispatch alignment
    (raw + 63) & !63
}

// DEPRECATED: These constants are kept for backward compatibility during transition.
// New code should load settings first and use those values.
// These will be removed once full settings-driven initialization is complete.
const FLUID_GRID_SIZE: u32 = DEFAULT_FLUID_GRID_RESOLUTION;
const GRID_DIM: usize = DEFAULT_ENV_GRID_RESOLUTION as usize;
const GRID_CELL_COUNT: usize = GRID_DIM * GRID_DIM;
const GRID_DIM_U32: u32 = GRID_DIM as u32;
const SPATIAL_GRID_DIM: usize = DEFAULT_SPATIAL_GRID_RESOLUTION as usize;
const SPATIAL_GRID_CELL_COUNT: usize = SPATIAL_GRID_DIM * SPATIAL_GRID_DIM;
const DIFFUSE_WG_SIZE_X: u32 = 16;
const DIFFUSE_WG_SIZE_Y: u32 = 16;
const CLEAR_WG_SIZE_X: u32 = 16;
const CLEAR_WG_SIZE_Y: u32 = 16;
const SLOPE_WG_SIZE_X: u32 = 16;
const SLOPE_WG_SIZE_Y: u32 = 16;
const TERRAIN_FORCE_SCALE: f32 = 250.0;
#[allow(dead_code)]
const GAMMA_CORRECTION_EXPONENT: f32 = 2.2;
const SETTINGS_FILE_NAME: &str = "simulation_settings.json";
const AUTO_SNAPSHOT_FILE_NAME: &str = "autosave_snapshot.png";
const AUTO_SNAPSHOT_INTERVAL: u64 = 10000; // Save every 10,000 epochs
const RAIN_THUMB_SIZE: usize = 128;

// Microswim params are provided to shaders as a flat f32 list, packed into vec4-aligned uniform storage.
// Keep this a multiple of 4 so WGSL can declare it as `array<vec4<f32>, N>`.
const MICROSWIM_PARAM_FLOATS: usize = 16;

// Fumaroles (fluid-only). Keep in sync with shaders/fluid.wgsl.
const MAX_FUMAROLES: usize = 64;
const FUMAROLE_STRIDE_F32: usize = 10;
const FUMAROLE_BUFFER_FLOATS: usize = 1 + MAX_FUMAROLES * FUMAROLE_STRIDE_F32;
const FUMAROLE_BUFFER_BYTES: usize = FUMAROLE_BUFFER_FLOATS * 4;

// Part base-angle override slots.
// NOTE: Shader defines part types 0..=46. We reserve 128 slots for future expansion.
const PART_TYPE_COUNT: usize = 47;
const PART_OVERRIDE_SLOTS: usize = 128;
const PART_OVERRIDE_VEC4S: usize = (PART_OVERRIDE_SLOTS + 3) / 4; // 32
const PART_OVERRIDES_CSV_PATH: &str = "config/part_base_angle_overrides.csv";

// Part (amino + organ) property table: 47 parts, each with 6x vec4<f32> entries.
// This mirrors the AMINO_DATA layout in shaders/shared.wgsl.
const PART_PROPS_VEC4S_PER_PART: usize = 6;
// NOTE: GPU buffer reserves enough vec4s for all part property overrides.
// 47 part types * 6 vec4s = 282 vec4s.
const PART_PROPS_OVERRIDE_VEC4S_USED: usize = PART_TYPE_COUNT * PART_PROPS_VEC4S_PER_PART; // 282
const PART_PROPS_OVERRIDE_VEC4S: usize = PART_PROPS_OVERRIDE_VEC4S_USED;
// bytemuck only implements Pod/Zeroable for certain array lengths; the full override table isn't one of them.
// We keep the exact same contiguous memory layout by splitting into two arrays.
const PART_PROPS_OVERRIDE_VEC4S_HEAD: usize = 256;
const PART_PROPS_OVERRIDE_VEC4S_TAIL: usize = PART_PROPS_OVERRIDE_VEC4S - PART_PROPS_OVERRIDE_VEC4S_HEAD; // 26
const PART_FLAGS_OVERRIDE_VEC4S: usize = (PART_TYPE_COUNT + 3) / 4; // 12
const PART_PROPERTIES_JSON_PATH: &str = "config/part_properties.json";

fn write_part_props_override_into_env_init(
    dst: &mut EnvironmentInitParams,
    src: &[[f32; 4]; PART_PROPS_OVERRIDE_VEC4S],
) {
    dst.part_props_override_head[..].copy_from_slice(&src[..PART_PROPS_OVERRIDE_VEC4S_HEAD]);
    dst.part_props_override_tail[..].copy_from_slice(&src[PART_PROPS_OVERRIDE_VEC4S_HEAD..]);
}

const PART_TYPE_NAMES: [&str; PART_TYPE_COUNT] = [
    // 0�19 amino acids
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
    // 20�45 organs / specials (keep in sync with shaders/shared.wgsl table comments)
    "MOUTH", "PROPELLER", "ALPHA_SENSOR", "BETA_SENSOR", "ENERGY_SENSOR", "ALPHA_EMITTER", "ENABLER", "BETA_EMITTER",
    "STORAGE", "POISON_RESIST", "CHIRAL_FLIPPER", "CLOCK", "SLOPE_SENSOR", "VAMPIRE_MOUTH", "AGENT_ALPHA_SENSOR",
    "AGENT_BETA_SENSOR", "UNUSED_36", "TRAIL_ENERGY_SENSOR", "ALPHA_MAG_SENSOR", "ALPHA_MAG_SENSOR_V2",
    "BETA_MAG_SENSOR", "BETA_MAG_SENSOR_V2", "ANCHOR", "MUTATION_PROTECTION", "BetaMouth", "ATTRACTOR_REPULSOR",
    "SPIKE",
];

const APP_NAME: &str = "Ribossome";
const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

fn ui_part_base_angle_overrides(ui: &mut egui::Ui, state: &mut GpuState) {
    ui.label("Overrides are in radians. NaN = use shader default.");
    ui.horizontal(|ui| {
        if ui.button("Load CSV").clicked() {
            match load_part_base_angle_overrides_csv(Path::new(PART_OVERRIDES_CSV_PATH)) {
                Ok(ov) => {
                    state.part_base_angle_overrides = ov;
                    state.part_base_angle_overrides_dirty = true;
                }
                Err(err) => eprintln!("Failed to load {}: {err:?}", PART_OVERRIDES_CSV_PATH),
            }
        }
        if ui.button("Save CSV").clicked() {
            if let Err(err) = save_part_base_angle_overrides_csv(
                Path::new(PART_OVERRIDES_CSV_PATH),
                &state.part_base_angle_overrides,
            ) {
                eprintln!("Failed to save {}: {err:?}", PART_OVERRIDES_CSV_PATH);
            }
        }
        if ui.button("Clear").clicked() {
            state.part_base_angle_overrides = [f32::NAN; PART_OVERRIDE_SLOTS];
            state.part_base_angle_overrides_dirty = true;
        }
    });

    egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
        for i in 0..PART_TYPE_COUNT {
            ui.horizontal(|ui| {
                ui.label(format!("{:02} {}", i, PART_TYPE_NAMES[i]));

                let mut enabled = !state.part_base_angle_overrides[i].is_nan();
                if ui.checkbox(&mut enabled, "override").changed() {
                    state.part_base_angle_overrides[i] = if enabled { 0.0 } else { f32::NAN };
                    state.part_base_angle_overrides_dirty = true;
                }

                if enabled {
                    let mut v = state.part_base_angle_overrides[i];
                    let changed = ui.add(egui::DragValue::new(&mut v).speed(0.01)).changed();
                    ui.label(format!("({:.1}�)", v.to_degrees()));
                    if changed {
                        state.part_base_angle_overrides[i] = v;
                        state.part_base_angle_overrides_dirty = true;
                    }
                } else {
                    ui.label("default");
                }
            });
        }
    });
}

fn ui_part_properties_editor_popup(ui: &egui::Ui, state: &mut GpuState) {
    if !state.show_part_properties_editor {
        return;
    }

    const VEC4_LABELS: [[&str; 4]; PART_PROPS_VEC4S_PER_PART] = [
        ["segment_length", "thickness", "base_angle", "mass"],
        ["alpha_sens", "beta_sens", "thrust_force", "energy_cons"],
        ["color_r", "color_g", "color_b", "energy_storage"],
        ["energy_absorb", "beta_absorb", "beta_damage", "parameter1"],
        ["signal_decay", "alpha_left", "alpha_right", "beta_left"],
        ["beta_right", "wind_coupling", "unused_5z", "unused_5w"],
    ];

    egui::Window::new("Part & Organ Properties")
        .open(&mut state.show_part_properties_editor)
        .resizable(true)
        .vscroll(true)
        .show(ui.ctx(), |ui| {
            ui.label("Edits apply immediately (no restart). Save to persist.");
            ui.label(format!("File: {}", PART_PROPERTIES_JSON_PATH));

            ui.horizontal(|ui| {
                if ui.button("Load JSON").clicked() {
                    match load_part_properties_json(Path::new(PART_PROPERTIES_JSON_PATH)) {
                        Ok((props, flags)) => {
                            // Start from shader defaults, then apply JSON values.
                            state.part_props_override = state.part_props_defaults;
                            for i in 0..PART_PROPS_OVERRIDE_VEC4S_USED {
                                for c in 0..4 {
                                    let v = props[i][c];
                                    if !v.is_nan() {
                                        state.part_props_override[i][c] = v;
                                    }
                                }
                            }

                            // Flags are intentionally ignored in the editor now.
                            let _ = flags;
                            state.part_flags_override = [f32::NAN; PART_TYPE_COUNT];
                            state.part_properties_dirty = true;
                        }
                        Err(err) => eprintln!("Failed to load {}: {err:?}", PART_PROPERTIES_JSON_PATH),
                    }
                }
                if ui.button("Save JSON").clicked() {
                    if let Err(err) = save_part_properties_json(
                        Path::new(PART_PROPERTIES_JSON_PATH),
                        &state.part_props_override,
                        &state.part_props_defaults,
                    ) {
                        eprintln!("Failed to save {}: {err:?}", PART_PROPERTIES_JSON_PATH);
                    }
                }
                if ui.button("Reset ALL to shader defaults").clicked() {
                    state.part_props_override = state.part_props_defaults;
                    state.part_flags_override = [f32::NAN; PART_TYPE_COUNT];
                    state.part_properties_dirty = true;
                }
            });

            ui.separator();

            egui::ScrollArea::vertical().show(ui, |ui| {
                for part_idx in 0..PART_TYPE_COUNT {
                    let name = PART_TYPE_NAMES[part_idx];
                    let header = format!("{:02} {}", part_idx, name);
                    ui.collapsing(header, |ui| {
                        let base = part_idx * PART_PROPS_VEC4S_PER_PART;
                        let vec4s_available = if base >= PART_PROPS_OVERRIDE_VEC4S {
                            0
                        } else {
                            (PART_PROPS_OVERRIDE_VEC4S - base).min(PART_PROPS_VEC4S_PER_PART)
                        };

                        if vec4s_available < PART_PROPS_VEC4S_PER_PART {
                            ui.label(format!(
                                "Note: only the first {} of {} vec4 blocks are editable (override buffer limit).",
                                vec4s_available, PART_PROPS_VEC4S_PER_PART
                            ));
                        }

                        ui.horizontal(|ui| {
                            if ui.button("Reset this part to defaults").clicked() {
                                for v in 0..vec4s_available {
                                    let k = base + v;
                                    state.part_props_override[k] = state.part_props_defaults[k];
                                }
                                state.part_properties_dirty = true;
                            }

                            ui.label("(numeric values only)");
                        });

                        // Property vec4 blocks
                        egui::Grid::new(format!("props_grid_{}", part_idx))
                            .num_columns(5)
                            .spacing([10.0, 6.0])
                            .striped(true)
                            .show(ui, |ui| {
                                ui.label("block");
                                ui.label("field");
                                ui.label("value");
                                ui.label("default");
                                ui.label("delta");
                                ui.end_row();

                                for v in 0..PART_PROPS_VEC4S_PER_PART {
                                    let k = base + v;
                                    let in_override_buffer = k < PART_PROPS_OVERRIDE_VEC4S;
                                    for c in 0..4 {
                                        let label = VEC4_LABELS[v][c];
                                        let def = state
                                            .part_props_defaults_full
                                            .get(k)
                                            .copied()
                                            .unwrap_or([0.0; 4])[c];

                                        let mut val = def;
                                        if in_override_buffer {
                                            val = state.part_props_override[k][c];
                                            if val.is_nan() {
                                                val = def;
                                                state.part_props_override[k][c] = val;
                                                state.part_properties_dirty = true;
                                            }
                                        }

                                        ui.label(format!("v{}", v));
                                        ui.label(label);

                                        let changed = ui
                                            .add_enabled(in_override_buffer, egui::DragValue::new(&mut val).speed(0.01))
                                            .changed();
                                        if changed && in_override_buffer {
                                            state.part_props_override[k][c] = val;
                                            state.part_properties_dirty = true;
                                        }

                                        ui.label(format!("{:.6}", def));
                                        ui.label(format!("{:+.6}", val - def));
                                        ui.end_row();
                                    }
                                }
                            });
                    });
                    ui.separator();
                }
            });
        });
}

fn pack_part_base_angle_overrides_vec4(overrides: &[f32; PART_OVERRIDE_SLOTS]) -> [[f32; 4]; PART_OVERRIDE_VEC4S] {
    let mut out = [[f32::NAN; 4]; PART_OVERRIDE_VEC4S];
    for i in 0..PART_OVERRIDE_SLOTS {
        out[i / 4][i & 3] = overrides[i];
    }
    out
}

fn load_part_base_angle_overrides_csv(path: &Path) -> anyhow::Result<[f32; PART_OVERRIDE_SLOTS]> {
    let text = fs::read_to_string(path)?;
    let mut out = [f32::NAN; PART_OVERRIDE_SLOTS];

    for (line_idx, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Allow a header line.
        if line_idx == 0 && line.to_ascii_lowercase().contains("index") {
            continue;
        }

        let mut parts = line.split(',').map(|s| s.trim());
        let idx_str = parts.next().unwrap_or_default();
        let col1 = parts.next().unwrap_or_default();
        let col2 = parts.next().unwrap_or_default();
        // Accept either:
        // - index,value
        // - index,name,value
        let val_str = if !col2.is_empty() { col2 } else { col1 };
        if idx_str.is_empty() {
            continue;
        }
        let idx: usize = idx_str.parse()?;
        if idx >= PART_OVERRIDE_SLOTS {
            continue;
        }
        if val_str.is_empty() || val_str.eq_ignore_ascii_case("nan") {
            out[idx] = f32::NAN;
            continue;
        }
        out[idx] = val_str.parse::<f32>()?;
    }

    Ok(out)
}

fn save_part_base_angle_overrides_csv(path: &Path, overrides: &[f32; PART_OVERRIDE_SLOTS]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut out = String::new();
    out.push_str("index,name,base_angle_rad\n");
    for i in 0..PART_OVERRIDE_SLOTS {
        let name = if i < PART_TYPE_COUNT { PART_TYPE_NAMES[i] } else { "" };
        let v = overrides[i];
        if v.is_nan() {
            out.push_str(&format!("{},{},\n", i, name));
        } else {
            out.push_str(&format!("{},{},{:.8}\n", i, name, v));
        }
    }
    fs::write(path, out)?;
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PartPropertiesJson {
    // Exactly PART_TYPE_COUNT entries.
    parts: Vec<PartPropertiesPartJson>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PartPropertiesPartJson {
    index: usize,
    name: String,
    // 6 vec4 blocks, component-wise optional. None = use shader default.
    vec4: [[Option<f32>; 4]; PART_PROPS_VEC4S_PER_PART],
    // Optional override of AMINO_FLAGS[t] (bitmask). None = use shader default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    flags: Option<u32>,
}

fn pack_part_flags_override_vec4(overrides: &[f32; PART_TYPE_COUNT]) -> [[f32; 4]; PART_FLAGS_OVERRIDE_VEC4S] {
    let mut out = [[f32::NAN; 4]; PART_FLAGS_OVERRIDE_VEC4S];
    for i in 0..PART_TYPE_COUNT {
        out[i / 4][i & 3] = overrides[i];
    }
    out
}

fn load_part_properties_json(
    path: &Path,
) -> anyhow::Result<(
    [[f32; 4]; PART_PROPS_OVERRIDE_VEC4S],
    [f32; PART_TYPE_COUNT],
)> {
    let text = fs::read_to_string(path)?;
    let parsed: PartPropertiesJson = serde_json::from_str(&text)?;

    let mut props = [[f32::NAN; 4]; PART_PROPS_OVERRIDE_VEC4S];
    let mut flags = [f32::NAN; PART_TYPE_COUNT];

    for part in parsed.parts {
        if part.index >= PART_TYPE_COUNT {
            continue;
        }
        for v in 0..PART_PROPS_VEC4S_PER_PART {
            let dst = part.index * PART_PROPS_VEC4S_PER_PART + v;
            if dst >= PART_PROPS_OVERRIDE_VEC4S {
                continue;
            }
            for c in 0..4 {
                props[dst][c] = part.vec4[v][c].unwrap_or(f32::NAN);
            }
        }
        flags[part.index] = part.flags.map(|f| f as f32).unwrap_or(f32::NAN);
    }

    Ok((props, flags))
}

fn save_part_properties_json(
    path: &Path,
    props_override: &[[f32; 4]; PART_PROPS_OVERRIDE_VEC4S],
    props_defaults: &[[f32; 4]; PART_PROPS_OVERRIDE_VEC4S],
) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let mut parts: Vec<PartPropertiesPartJson> = Vec::with_capacity(PART_TYPE_COUNT);
    for i in 0..PART_TYPE_COUNT {
        let mut vec4 = [[None; 4]; PART_PROPS_VEC4S_PER_PART];
        for v in 0..PART_PROPS_VEC4S_PER_PART {
            let src = i * PART_PROPS_VEC4S_PER_PART + v;
            if src >= PART_PROPS_OVERRIDE_VEC4S {
                // This part/vec4 block is not representable in the fixed-size override buffer.
                // Leave as None so the shader default is used.
                continue;
            }
            for c in 0..4 {
                // Always emit a complete numeric table: if somehow NaN is present,
                // fall back to the shader default.
                let x = props_override[src][c];
                let def = props_defaults[src][c];
                vec4[v][c] = Some(if x.is_nan() { def } else { x });
            }
        }

        parts.push(PartPropertiesPartJson {
            index: i,
            name: PART_TYPE_NAMES[i].to_string(),
            vec4,
            flags: None,
        });
    }

    let out = PartPropertiesJson { parts };
    let json = serde_json::to_string_pretty(&out)?;
    fs::write(path, json)?;
    Ok(())
}

fn parse_shared_wgsl_part_defaults(
    path: &Path,
) -> anyhow::Result<(Vec<[f32; 4]>, [u32; PART_TYPE_COUNT])> {
    let text = fs::read_to_string(path)?;

    // --- Parse AMINO_DATA vec4 list ---
    let amino_start = text
        .find("var<private> AMINO_DATA")
        .ok_or_else(|| anyhow::anyhow!("AMINO_DATA not found"))?;
    let amino_slice = &text[amino_start..];
    let amino_end = amino_slice
        .find("var<private> AMINO_FLAGS")
        .ok_or_else(|| anyhow::anyhow!("AMINO_FLAGS not found (after AMINO_DATA)"))?;
    let amino_block = &amino_slice[..amino_end];

    let mut vec4s: Vec<[f32; 4]> = Vec::with_capacity(PART_PROPS_OVERRIDE_VEC4S_USED);
    let mut scan = 0usize;
    let pat = "vec4<f32>(";
    while let Some(pos) = amino_block[scan..].find(pat) {
        let start = scan + pos + pat.len();
        let rest = &amino_block[start..];
        let close = rest
            .find(')')
            .ok_or_else(|| anyhow::anyhow!("Unclosed vec4<f32>(...) in AMINO_DATA"))?;
        let inside = &rest[..close];

        let parts: Vec<&str> = inside
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        if parts.len() != 4 {
            return Err(anyhow::anyhow!(
                "Expected 4 floats in vec4, got {}: {inside}",
                parts.len()
            ));
        }

        let mut vals = [0.0f32; 4];
        for k in 0..4 {
            vals[k] = parts[k].parse::<f32>()?;
        }
        vec4s.push(vals);
        scan = start + close + 1;
    }

    if vec4s.len() < PART_PROPS_OVERRIDE_VEC4S_USED {
        return Err(anyhow::anyhow!(
            "Expected at least {} vec4 entries in AMINO_DATA, found {}",
            PART_PROPS_OVERRIDE_VEC4S_USED,
            vec4s.len()
        ));
    }

    // --- Parse AMINO_FLAGS list ---
    let flags_start = text
        .find("var<private> AMINO_FLAGS")
        .ok_or_else(|| anyhow::anyhow!("AMINO_FLAGS not found"))?;
    let flags_slice = &text[flags_start..];
    let paren_open = flags_slice
        .find('(')
        .ok_or_else(|| anyhow::anyhow!("AMINO_FLAGS '(' not found"))?;

    // Find the matching closing ')' for the array constructor, accounting for nested parentheses
    // in expressions like (1u<<9), and ignoring // line comments.
    let mut depth: i32 = 0;
    let mut in_line_comment = false;
    let mut content_start: Option<usize> = None;
    let mut content_end: Option<usize> = None;
    let bytes = flags_slice.as_bytes();
    let mut i = paren_open;
    while i < bytes.len() {
        let b = bytes[i];
        if in_line_comment {
            if b == b'\n' {
                in_line_comment = false;
            }
            i += 1;
            continue;
        }
        if b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            in_line_comment = true;
            i += 2;
            continue;
        }

        if b == b'(' {
            depth += 1;
            if depth == 1 {
                content_start = Some(i + 1);
            }
        } else if b == b')' {
            depth -= 1;
            if depth == 0 {
                content_end = Some(i);
                break;
            }
        }
        i += 1;
    }
    let content_start = content_start.ok_or_else(|| anyhow::anyhow!("AMINO_FLAGS content start not found"))?;
    let content_end = content_end.ok_or_else(|| anyhow::anyhow!("AMINO_FLAGS matching ')' not found"))?;
    let flags_inside = &flags_slice[content_start..content_end];

    // Strip // comments before tokenizing.
    let mut flags_clean = String::new();
    for line in flags_inside.lines() {
        let line = match line.split_once("//") {
            Some((pre, _)) => pre,
            None => line,
        };
        flags_clean.push_str(line);
        flags_clean.push('\n');
    }

    fn eval_flag_expr(expr: &str) -> anyhow::Result<u32> {
        let e = expr.trim();
        if e.is_empty() {
            return Ok(0);
        }
        let mut acc: u32 = 0;
        for term in e.split('|') {
            let t = term.trim().trim_end_matches(',');
            if t.is_empty() || t == "0u" || t == "0" {
                continue;
            }
            if let Some(sh_pos) = t.find("<<") {
                let left = t[..sh_pos].trim().trim_start_matches('(').trim();
                let right = t[sh_pos + 2..]
                    .trim()
                    .trim_end_matches(')')
                    .trim_end_matches('u');
                let base = left.trim_end_matches('u');
                let base_v: u32 = base.parse()?;
                let shift: u32 = right.parse()?;
                acc |= base_v << shift;
            } else {
                let lit = t.trim_end_matches('u').trim_end_matches(')');
                acc |= lit.parse::<u32>()?;
            }
        }
        Ok(acc)
    }

    let mut flags_list: Vec<u32> = Vec::with_capacity(PART_TYPE_COUNT);
    for raw in flags_clean.split(',') {
        let item = raw.trim();
        if item.is_empty() {
            continue;
        }
        flags_list.push(eval_flag_expr(item)?);
        if flags_list.len() == PART_TYPE_COUNT {
            break;
        }
    }
    if flags_list.len() != PART_TYPE_COUNT {
        return Err(anyhow::anyhow!(
            "Expected {} AMINO_FLAGS entries, found {}",
            PART_TYPE_COUNT,
            flags_list.len()
        ));
    }

    let mut flags = [0u32; PART_TYPE_COUNT];
    for (idx, v) in flags_list.into_iter().enumerate() {
        flags[idx] = v;
    }

    Ok((vec4s, flags))
}

// Selected-agent CPU readback: use a small ring buffer so we can request ~60Hz updates
// without stalling on an in-flight map.
const SELECTED_AGENT_READBACK_SLOTS: usize = 4;
const SELECTED_AGENT_READBACK_INTERVAL_MS: u64 = 16; // ~60Hz

// Shared genome/body sizing (must stay in sync with shader constants)
const MAX_BODY_PARTS: usize = 64;
const GENOME_BYTES: usize = 256; // ASCII bases including padding
const GENOME_WORDS: usize = GENOME_BYTES / std::mem::size_of::<u32>();
const GENOME_PACKED_WORDS: usize = GENOME_BYTES / 16; // 16 bases per packed u32
const GENOME_BASES_PER_PACKED_WORD: usize = 16;
#[allow(dead_code)]
const MIN_GENE_LENGTH: usize = 6;
// NOTE: Keep this a multiple of 64 so we can dispatch a fixed workgroup count
// (MAX_SPAWN_REQUESTS/64) and rely on early returns in the kernels.
const MAX_SPAWN_REQUESTS: usize = 2048;

// Spawn/merge/compact WGSL kernels assume a hard cap of 2000 CPU spawns per batch.
// Keep this <= MAX_SPAWN_REQUESTS.
const MAX_CPU_SPAWNS_PER_BATCH: u32 = 2000;

const SNAPSHOT_VERSION: &str = "1.2";

// Snapshot files embed agent data as JSON inside a PNG.
// To keep file sizes and save times reasonable, we store a random sample when populations are huge.
const MAX_SNAPSHOT_AGENTS: usize = 10_000;

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
#[allow(dead_code)]
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

#[allow(dead_code)]
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

fn select_required_features(adapter_features: wgpu::Features) -> wgpu::Features {
    let desired = wgpu::Features::TIMESTAMP_QUERY
        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    adapter_features & desired
}

// Timestamp indices. We sample infrequently (default 1Hz) so we can afford a lot of markers.
// Keep these contiguous; the readback decoding assumes 0..GPU_TS_COUNT.
const TS_UPDATE_START: u32 = 0;

// --- "Compute Pass" breakdown (diffuse/trails/slope/clears/spatial/process/drain)
const TS_SIM_PASS_START: u32 = 1;
const TS_SIM_AFTER_DIFFUSE: u32 = 2;
const TS_SIM_AFTER_DIFFUSE_COMMIT: u32 = 3;
const TS_SIM_AFTER_TRAILS_PREP: u32 = 4;
const TS_SIM_AFTER_SLOPE: u32 = 5;
const TS_SIM_AFTER_CLEAR_VISUAL: u32 = 6;
const TS_SIM_AFTER_MOTION_BLUR: u32 = 7;
const TS_SIM_AFTER_CLEAR_AGENT_GRID: u32 = 8;
const TS_SIM_AFTER_CLEAR_SPATIAL: u32 = 9;
const TS_SIM_AFTER_POPULATE_SPATIAL: u32 = 10;
const TS_SIM_AFTER_CLEAR_FLUID_FORCE_VECTORS: u32 = 11;
const TS_SIM_AFTER_PROCESS_AGENTS: u32 = 12;
const TS_SIM_AFTER_DRAIN_ENERGY: u32 = 13;
const TS_UPDATE_AFTER_SIM: u32 = 14;

// --- "Fluid Compute Pass" breakdown
const TS_FLUID_PASS_START: u32 = 15;
const TS_FLUID_AFTER_FUMAROLE_FORCE: u32 = 16;
const TS_FLUID_AFTER_GENERATE_FORCES: u32 = 17;
const TS_FLUID_AFTER_ADD_FORCES: u32 = 18;
const TS_FLUID_AFTER_CLEAR_FORCES: u32 = 19;
const TS_FLUID_AFTER_DIFFUSE_VELOCITY: u32 = 20;
const TS_FLUID_AFTER_ADVECT_VELOCITY: u32 = 21;
const TS_FLUID_AFTER_VORTICITY: u32 = 22;
const TS_FLUID_AFTER_DIVERGENCE: u32 = 23;
const TS_FLUID_AFTER_CLEAR_PRESSURE: u32 = 24;
const TS_FLUID_AFTER_JACOBI: u32 = 25;
const TS_FLUID_AFTER_SUBTRACT_GRADIENT: u32 = 26;
const TS_FLUID_AFTER_BOUNDARIES: u32 = 27;
const TS_FLUID_AFTER_INJECT_DYE: u32 = 28;
const TS_FLUID_AFTER_ADVECT_DYE: u32 = 29;
const TS_FLUID_AFTER_FUMAROLE_DYE: u32 = 30;
const TS_FLUID_AFTER_ADVECT_TRAIL: u32 = 31;
const TS_UPDATE_AFTER_FLUID: u32 = 32;

// --- "Post-Fluid Compute Pass" breakdown
const TS_POST_PASS_START: u32 = 33;
const TS_POST_AFTER_COMPACT: u32 = 34;
const TS_POST_AFTER_CPU_SPAWN: u32 = 35;
const TS_POST_AFTER_MERGE: u32 = 36;
const TS_POST_AFTER_INIT_DEAD: u32 = 37;
const TS_POST_AFTER_FINALIZE_MERGE: u32 = 38;
const TS_POST_AFTER_PROCESS_PAUSED_RENDER_ONLY: u32 = 39;
const TS_POST_AFTER_RENDER_AGENTS: u32 = 40;
const TS_POST_AFTER_INSPECTOR_CLEAR: u32 = 41;
const TS_POST_AFTER_INSPECTOR_DRAW: u32 = 42;
const TS_UPDATE_AFTER_POST: u32 = 43;

// --- Composite pass + copy
const TS_COMPOSITE_PASS_START: u32 = 44;
const TS_UPDATE_AFTER_COMPOSITE: u32 = 45;
const TS_UPDATE_AFTER_COPY: u32 = 46;
const TS_UPDATE_END: u32 = 47;

// --- Render (main + overlay) + egui
const TS_RENDER_ENC_START: u32 = 48;
const TS_RENDER_MAIN_START: u32 = 49;
const TS_RENDER_MAIN_END: u32 = 50;
const TS_RENDER_OVERLAY_START: u32 = 51;
const TS_RENDER_OVERLAY_END: u32 = 52;
const TS_RENDER_ENC_END: u32 = 53;
const TS_EGUI_ENC_START: u32 = 54;
const TS_EGUI_PASS_START: u32 = 55;
const TS_EGUI_PASS_END: u32 = 56;
const TS_EGUI_ENC_END: u32 = 57;

// --- No-fluid fallback breakdown (measured inside the fluid compute pass)
// These are written only when `!fluid_enabled`, otherwise they will print as -1.
const TS_NOFLUID_AFTER_DYE_DIFFUSE: u32 = 58;
const TS_NOFLUID_AFTER_DYE_COPYBACK: u32 = 59;
const TS_NOFLUID_AFTER_TRAIL_COMMIT: u32 = 60;
// Microswim timing (separates agent processing vs microswim cost).
const TS_SIM_AFTER_MICROSWIM: u32 = 61;

const GPU_TS_COUNT: u32 = 62;

const FRAME_PROFILER_READBACK_SLOTS: usize = 2;

#[derive(Default)]
struct BenchSeries {
    values: Vec<f64>,
}

impl BenchSeries {
    fn push_opt(&mut self, v: Option<f64>) {
        if let Some(v) = v {
            if v.is_finite() {
                self.values.push(v);
            }
        }
    }

    fn summary(&mut self) -> Option<(usize, f64, f64, f64, f64, f64)> {
        if self.values.is_empty() {
            return None;
        }
        self.values.sort_by(|a, b| a.total_cmp(b));
        let n = self.values.len();
        let min = self.values[0];
        let max = self.values[n - 1];
        let mean = self.values.iter().sum::<f64>() / (n as f64);

        let q = |p: f64| -> f64 {
            let p = p.clamp(0.0, 1.0);
            let idx = ((n - 1) as f64 * p).round() as usize;
            self.values[idx.clamp(0, n - 1)]
        };
        let p50 = q(0.50);
        let p95 = q(0.95);

        Some((n, min, p50, p95, mean, max))
    }
}

struct BenchCollector {
    warmup_remaining: usize,
    samples_remaining: usize,
    printed: bool,

    sim_total_ms: BenchSeries,
    sim_diffuse_ms: BenchSeries,
    sim_process_ms: BenchSeries,
    sim_drain_ms: BenchSeries,
    cpu_update_ms: BenchSeries,
}

impl BenchCollector {
    fn new(warmup: usize, samples: usize) -> Self {
        Self {
            warmup_remaining: warmup,
            samples_remaining: samples.max(1),
            printed: false,
            sim_total_ms: BenchSeries::default(),
            sim_diffuse_ms: BenchSeries::default(),
            sim_process_ms: BenchSeries::default(),
            sim_drain_ms: BenchSeries::default(),
            cpu_update_ms: BenchSeries::default(),
        }
    }

    fn on_sample(
        &mut self,
        sim_total_ms: Option<f64>,
        sim_diffuse_ms: Option<f64>,
        sim_process_ms: Option<f64>,
        sim_drain_ms: Option<f64>,
        cpu_update_ms: f64,
    ) -> bool {
        if self.warmup_remaining > 0 {
            self.warmup_remaining -= 1;
            return false;
        }

        self.sim_total_ms.push_opt(sim_total_ms);
        self.sim_diffuse_ms.push_opt(sim_diffuse_ms);
        self.sim_process_ms.push_opt(sim_process_ms);
        self.sim_drain_ms.push_opt(sim_drain_ms);
        self.cpu_update_ms.push_opt(Some(cpu_update_ms));

        self.samples_remaining = self.samples_remaining.saturating_sub(1);
        self.samples_remaining == 0
    }

    fn print_summary_once(&mut self) {
        if self.printed {
            return;
        }
        self.printed = true;

        println!("\n[bench] Completed benchmark samples.");
        println!("[bench] All times are milliseconds; p50/p95 computed over captured samples.");

        let print_series = |name: &str, s: &mut BenchSeries| {
            if let Some((n, min, p50, p95, mean, max)) = s.summary() {
                println!(
                    "[bench] {:<12} n={:<4} min={:>7.3} p50={:>7.3} p95={:>7.3} mean={:>7.3} max={:>7.3}",
                    name, n, min, p50, p95, mean, max
                );
            } else {
                println!("[bench] {:<12} (no samples)", name);
            }
        };

        print_series("sim", &mut self.sim_total_ms);
        print_series("diffuse", &mut self.sim_diffuse_ms);
        print_series("process", &mut self.sim_process_ms);
        print_series("drain", &mut self.sim_drain_ms);
        print_series("cpuUpdate", &mut self.cpu_update_ms);

        println!("[bench] Exiting.");
    }
}

struct FrameProfiler {
    enabled: bool,
    gpu_supported: bool,
    gpu_active: bool,
    capture_this_frame: bool,
    sample_interval: std::time::Duration,
    next_sample_time: std::time::Instant,
    timestamp_period_ns: f64,
    query_set: Option<wgpu::QuerySet>,
    query_resolve: Option<wgpu::Buffer>,
    query_readback: Vec<wgpu::Buffer>,
    readback_busy: [bool; FRAME_PROFILER_READBACK_SLOTS],
    capture_index: u64,
    last_copied_slot: Option<usize>,
    pending_map: Option<(
        usize,
        std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    )>,
    wait_for_readback: bool,
    last_cpu_update_ms: f64,
    last_cpu_render_ms: f64,
    last_cpu_egui_ms: f64,
    dispatch_timing_enabled: bool,
    dispatch_timing_detail: bool,
    dispatch_timings: DispatchTimings,
    last_inspector_requested: bool,

    bench: Option<BenchCollector>,
    bench_verbose: bool,
    bench_exit_requested: bool,

    // Optional: auto-exit after printing N GPU timestamp captures.
    profile_max_captures: Option<u64>,
    profile_captures_done: u64,
}

#[derive(Clone, Copy, Default)]
struct DispatchTimings {
    sim_pre_ms: Option<f64>,
    sim_pre_diffuse_ms: Option<f64>,
    sim_pre_diffuse_commit_ms: Option<f64>,
    sim_pre_trails_ms: Option<f64>,
    sim_pre_slope_ms: Option<f64>,
    sim_pre_draw_ms: Option<f64>,
    sim_pre_repro_ms: Option<f64>,
    sim_pre_rain_ms: Option<f64>,
    sim_main_ms: Option<f64>,
    microswim_ms: Option<f64>,
    drain_ms: Option<f64>,
    fluid_ms: Option<f64>,
    post_ms: Option<f64>,
    composite_copy_ms: Option<f64>,
    update_tail_ms: Option<f64>,
    render_ms: Option<f64>,
    egui_ms: Option<f64>,
}

#[derive(Clone, Copy)]
enum DispatchSegment {
    SimPre,
    SimPreDiffuse,
    SimPreDiffuseCommit,
    SimPreTrails,
    SimPreSlope,
    SimPreDraw,
    SimPreRepro,
    SimPreRain,
    SimMain,
    Microswim,
    Drain,
    Fluid,
    Post,
    CompositeCopy,
    UpdateTail,
    Render,
    Egui,
}

impl FrameProfiler {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        // Bench mode is intentionally disabled.
        // (It has historically caused driver/device issues on some setups.)
        let bench_enabled = false;

        // Kept for backwards compatibility with existing env-var setups,
        // but bench is disabled so this is unused.
        let bench_gpu = false;

        let profile_enabled = std::env::var("ALSIM_PROFILE")
            .map(|value| value != "0")
            .unwrap_or(false);
        let enabled = profile_enabled || bench_enabled;

        // IMPORTANT: GPU timestamp queries are disabled by default because they are causing
        // device-loss crashes on some machines/drivers.
        // To force-enable (at your own risk): ALSIM_GPU_TIMESTAMPS=1
        let allow_gpu_timestamps = std::env::var("ALSIM_GPU_TIMESTAMPS")
            .map(|value| value != "0")
            .unwrap_or(false);

        // Default behavior is "blocking" (accurate per-sample printing but can stall the CPU).
        // Set ALSIM_PROFILE_WAIT=0 to avoid stalling; prints may be skipped until data is ready.
        // In bench mode we default to blocking unless explicitly overridden.
        let wait_for_readback = match std::env::var("ALSIM_PROFILE_WAIT") {
            Ok(value) => value != "0",
            Err(_) => bench_enabled,
        };

        let mut sample_interval = std::env::var("ALSIM_PROFILE_INTERVAL_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .map(std::time::Duration::from_millis)
            .unwrap_or_else(|| std::time::Duration::from_millis(1000));
        if bench_enabled {
            // Capture every frame; benchmark mode prints a summary at the end.
            sample_interval = std::time::Duration::from_millis(0);
        }

        let bench_verbose = false;

        let bench = None;

        let profile_max_captures = std::env::var("ALSIM_PROFILE_CAPTURES")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&n| n > 0);

        // CPU-side submit+wait timing for command-buffer segments.
        // This does NOT use timestamp queries, so it should be stable on drivers
        // that crash with TIMESTAMP_QUERY.
        // Enable with: ALSIM_PROFILE_DISPATCH_TIMING=1
        let dispatch_timing_enabled = std::env::var("ALSIM_PROFILE_DISPATCH_TIMING")
            .map(|value| value != "0")
            .unwrap_or(false);

        let dispatch_timing_detail = std::env::var("ALSIM_PROFILE_DISPATCH_TIMING_DETAIL")
            .map(|value| value != "0")
            .unwrap_or(false);

        // We use both:
        // - CommandEncoder::write_timestamp (requires TIMESTAMP_QUERY_INSIDE_ENCODERS)
        // - (Compute/Render)Pass::write_timestamp (requires TIMESTAMP_QUERY_INSIDE_PASSES)
        // Only enable GPU profiling when the device supports the full set *and* we allow it.
        let gpu_supported = allow_gpu_timestamps
            && device.features().contains(
                wgpu::Features::TIMESTAMP_QUERY
                    | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
                    | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
            );

        let gpu_active = enabled
            && gpu_supported
            && (profile_enabled || (bench_enabled && bench_gpu));

        if enabled {
            if gpu_active {
                eprintln!(
                    "[perf] Profiling enabled: GPU timestamps ACTIVE (ALSIM_GPU_TIMESTAMPS=1). \
This can cause device-loss on some drivers; unset Env:ALSIM_GPU_TIMESTAMPS to use CPU-only profiling."
                );
            } else if profile_enabled {
                if allow_gpu_timestamps && !gpu_supported {
                    eprintln!(
                        "[perf] Profiling enabled: CPU-only. Note: ALSIM_GPU_TIMESTAMPS was set, \
but required timestamp-query features are unavailable on this device."
                    );
                } else {
                    eprintln!("[perf] Profiling enabled: CPU-only.");
                }
            }
        }

        let timestamp_period_ns = queue.get_timestamp_period() as f64;
        let now = std::time::Instant::now();

        let (query_set, query_resolve, query_readback) = if gpu_active {
            let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("FrameProfiler QuerySet"),
                ty: wgpu::QueryType::Timestamp,
                count: GPU_TS_COUNT,
            });

            let bytes = (GPU_TS_COUNT as u64) * 8;
            let query_resolve = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("FrameProfiler QueryResolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let query_readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("FrameProfiler QueryReadback[0]"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let mut readbacks = Vec::with_capacity(FRAME_PROFILER_READBACK_SLOTS);
            readbacks.push(query_readback);
            for slot in 1..FRAME_PROFILER_READBACK_SLOTS {
                let label = format!("FrameProfiler QueryReadback[{slot}]");
                readbacks.push(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&label),
                    size: bytes,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }));
            }

            (Some(query_set), Some(query_resolve), readbacks)
        } else {
            (None, None, Vec::new())
        };

        Self {
            enabled,
            gpu_supported,
            gpu_active,
            capture_this_frame: false,
            sample_interval,
            next_sample_time: now + sample_interval,
            timestamp_period_ns,
            query_set,
            query_resolve,
            query_readback,
            readback_busy: [false; FRAME_PROFILER_READBACK_SLOTS],
            capture_index: 0,
            last_copied_slot: None,
            pending_map: None,
            wait_for_readback,
            last_cpu_update_ms: 0.0,
            last_cpu_render_ms: 0.0,
            last_cpu_egui_ms: 0.0,
            dispatch_timing_enabled,
            dispatch_timing_detail,
            dispatch_timings: DispatchTimings::default(),
            last_inspector_requested: false,

            bench,
            bench_verbose,
            bench_exit_requested: false,

            profile_max_captures,
            profile_captures_done: 0,
        }
    }

    fn should_time_dispatches(&self) -> bool {
        self.dispatch_timing_enabled && self.capture_this_frame
    }

    fn reset_dispatch_timings_if_needed(&mut self) {
        if self.should_time_dispatches() {
            self.dispatch_timings = DispatchTimings::default();
        }
    }

    fn record_dispatch_ms(&mut self, segment: DispatchSegment, ms: f64) {
        let ms = ms.max(0.0);
        match segment {
            DispatchSegment::SimPre => self.dispatch_timings.sim_pre_ms = Some(ms),
            DispatchSegment::SimPreDiffuse => self.dispatch_timings.sim_pre_diffuse_ms = Some(ms),
            DispatchSegment::SimPreDiffuseCommit => {
                self.dispatch_timings.sim_pre_diffuse_commit_ms = Some(ms)
            }
            DispatchSegment::SimPreTrails => self.dispatch_timings.sim_pre_trails_ms = Some(ms),
            DispatchSegment::SimPreSlope => self.dispatch_timings.sim_pre_slope_ms = Some(ms),
            DispatchSegment::SimPreDraw => self.dispatch_timings.sim_pre_draw_ms = Some(ms),
            DispatchSegment::SimPreRepro => self.dispatch_timings.sim_pre_repro_ms = Some(ms),
            DispatchSegment::SimPreRain => self.dispatch_timings.sim_pre_rain_ms = Some(ms),
            DispatchSegment::SimMain => self.dispatch_timings.sim_main_ms = Some(ms),
            DispatchSegment::Microswim => self.dispatch_timings.microswim_ms = Some(ms),
            DispatchSegment::Drain => self.dispatch_timings.drain_ms = Some(ms),
            DispatchSegment::Fluid => self.dispatch_timings.fluid_ms = Some(ms),
            DispatchSegment::Post => self.dispatch_timings.post_ms = Some(ms),
            DispatchSegment::CompositeCopy => self.dispatch_timings.composite_copy_ms = Some(ms),
            DispatchSegment::UpdateTail => self.dispatch_timings.update_tail_ms = Some(ms),
            DispatchSegment::Render => self.dispatch_timings.render_ms = Some(ms),
            DispatchSegment::Egui => self.dispatch_timings.egui_ms = Some(ms),
        }
    }

    fn sim_pre_effective_ms(&self) -> Option<f64> {
        if self.dispatch_timings.sim_pre_ms.is_some() {
            return self.dispatch_timings.sim_pre_ms;
        }
        let parts = [
            self.dispatch_timings.sim_pre_diffuse_ms,
            self.dispatch_timings.sim_pre_diffuse_commit_ms,
            self.dispatch_timings.sim_pre_rain_ms,
            self.dispatch_timings.sim_pre_trails_ms,
            self.dispatch_timings.sim_pre_slope_ms,
            self.dispatch_timings.sim_pre_draw_ms,
            self.dispatch_timings.sim_pre_repro_ms,
        ];
        if parts.iter().all(|p| p.is_none()) {
            return None;
        }
        Some(parts.iter().map(|p| p.unwrap_or(0.0)).sum())
    }

    fn submit_cmd_buffer(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, cmd: wgpu::CommandBuffer, segment: DispatchSegment) {
        if self.should_time_dispatches() {
            let start = std::time::Instant::now();
            queue.submit(Some(cmd));
            device.poll(wgpu::Maintain::Wait);
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            self.record_dispatch_ms(segment, ms);
        } else {
            queue.submit(Some(cmd));
        }
    }

    fn maybe_submit_encoder_segment(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        segment: DispatchSegment,
        next_label: &'static str,
    ) {
        if !self.should_time_dispatches() {
            return;
        }

        let new_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(next_label),
        });
        let old_encoder = std::mem::replace(encoder, new_encoder);
        self.submit_cmd_buffer(device, queue, old_encoder.finish(), segment);
    }

    fn bench_exit_requested(&self) -> bool {
        self.bench_exit_requested
    }

    fn set_inspector_requested(&mut self, requested: bool) {
        self.last_inspector_requested = requested;
    }

    fn begin_frame(&mut self) {
        if !self.enabled && !self.dispatch_timing_enabled {
            self.capture_this_frame = false;
            return;
        }
        self.capture_this_frame = std::time::Instant::now() >= self.next_sample_time;
        self.reset_dispatch_timings_if_needed();
    }

    fn write_ts_encoder(&self, encoder: &mut wgpu::CommandEncoder, index: u32) {
        if !self.capture_this_frame {
            return;
        }
        if let Some(query_set) = &self.query_set {
            encoder.write_timestamp(query_set, index);
        }
    }

    fn write_ts_compute_pass(&self, pass: &mut wgpu::ComputePass<'_>, index: u32) {
        if !self.capture_this_frame {
            return;
        }
        if let Some(query_set) = &self.query_set {
            pass.write_timestamp(query_set, index);
        }
    }

    fn write_ts_render_pass(&self, pass: &mut wgpu::RenderPass<'_>, index: u32) {
        if !self.capture_this_frame {
            return;
        }
        if let Some(query_set) = &self.query_set {
            pass.write_timestamp(query_set, index);
        }
    }

    fn resolve_and_copy(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if !self.capture_this_frame {
            return;
        }
        let (Some(query_set), Some(resolve)) = (&self.query_set, &self.query_resolve) else {
            return;
        };
        if self.query_readback.is_empty() {
            return;
        }

        let slot = (self.capture_index as usize) % FRAME_PROFILER_READBACK_SLOTS;
        if self.readback_busy[slot] {
            // If we're in non-blocking mode and readbacks are still busy, skip rather than stall.
            self.capture_this_frame = false;
            self.next_sample_time = std::time::Instant::now() + self.sample_interval;
            return;
        }

        let bytes = (GPU_TS_COUNT as u64) * 8;
        let readback = &self.query_readback[slot];
        encoder.resolve_query_set(query_set, 0..GPU_TS_COUNT, resolve, 0);
        encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, bytes);

        self.readback_busy[slot] = true;
        self.last_copied_slot = Some(slot);
        self.capture_index = self.capture_index.wrapping_add(1);
        self.capture_this_frame = false;
        self.next_sample_time = std::time::Instant::now() + self.sample_interval;
    }

    fn set_cpu_update_ms(&mut self, ms: f64) {
        self.last_cpu_update_ms = ms;
    }

    fn set_cpu_render_ms(&mut self, ms: f64) {
        self.last_cpu_render_ms = ms;
    }

    fn set_cpu_egui_ms(&mut self, ms: f64) {
        self.last_cpu_egui_ms = ms;
    }

    fn readback_and_print(&mut self, device: &wgpu::Device) {
        // Special case: dispatch timing can work independently of full profiler
        if !self.enabled && self.dispatch_timing_enabled {
            if !self.capture_this_frame {
                return;
            }
            let fmt = |v: Option<f64>| -> String {
                v.map(|ms| format!("{ms:7.2}")).unwrap_or_else(|| "   -   ".to_string())
            };
            println!(
                "[perf] submit_wait(ms): simPre={} sim={} micro={} drain={} fluid={} post={} comp={} tail={} render={} egui={}",
                fmt(self.sim_pre_effective_ms()),
                fmt(self.dispatch_timings.sim_main_ms),
                fmt(self.dispatch_timings.microswim_ms),
                fmt(self.dispatch_timings.drain_ms),
                fmt(self.dispatch_timings.fluid_ms),
                fmt(self.dispatch_timings.post_ms),
                fmt(self.dispatch_timings.composite_copy_ms),
                fmt(self.dispatch_timings.update_tail_ms),
                fmt(self.dispatch_timings.render_ms),
                fmt(self.dispatch_timings.egui_ms),
            );

            if self.dispatch_timing_detail {
                println!(
                    "[perf] simPre_parts(ms): diffuse={} commit={} rain={} trails={} slope={} draw={} repro={}  (sum shown as simPre)",
                    fmt(self.dispatch_timings.sim_pre_diffuse_ms),
                    fmt(self.dispatch_timings.sim_pre_diffuse_commit_ms),
                    fmt(self.dispatch_timings.sim_pre_rain_ms),
                    fmt(self.dispatch_timings.sim_pre_trails_ms),
                    fmt(self.dispatch_timings.sim_pre_slope_ms),
                    fmt(self.dispatch_timings.sim_pre_draw_ms),
                    fmt(self.dispatch_timings.sim_pre_repro_ms),
                );
            }
            self.capture_this_frame = false;
            self.next_sample_time = std::time::Instant::now() + self.sample_interval;
            return;
        }

        if !self.enabled {
            return;
        }

        // CPU-only profiling mode (safe; no timestamp queries).
        if !self.gpu_active {
            if !self.capture_this_frame {
                return;
            }
            println!(
                "[perf] cpu(ms): update={:7.2} render={:7.2} egui={:7.2}",
                self.last_cpu_update_ms,
                self.last_cpu_render_ms,
                self.last_cpu_egui_ms
            );

            if self.dispatch_timing_enabled {
                let fmt = |v: Option<f64>| -> String {
                    v.map(|ms| format!("{ms:7.2}")).unwrap_or_else(|| "   -   ".to_string())
                };
                println!(
                    "[perf] submit_wait(ms): simPre={} sim={} micro={} drain={} fluid={} post={} comp={} tail={} render={} egui={}",
                    fmt(self.sim_pre_effective_ms()),
                    fmt(self.dispatch_timings.sim_main_ms),
                    fmt(self.dispatch_timings.microswim_ms),
                    fmt(self.dispatch_timings.drain_ms),
                    fmt(self.dispatch_timings.fluid_ms),
                    fmt(self.dispatch_timings.post_ms),
                    fmt(self.dispatch_timings.composite_copy_ms),
                    fmt(self.dispatch_timings.update_tail_ms),
                    fmt(self.dispatch_timings.render_ms),
                    fmt(self.dispatch_timings.egui_ms),
                );

                if self.dispatch_timing_detail {
                    println!(
                        "[perf] simPre_parts(ms): diffuse={} commit={} rain={} trails={} slope={} draw={} repro={}  (sum shown as simPre)",
                        fmt(self.dispatch_timings.sim_pre_diffuse_ms),
                        fmt(self.dispatch_timings.sim_pre_diffuse_commit_ms),
                        fmt(self.dispatch_timings.sim_pre_rain_ms),
                        fmt(self.dispatch_timings.sim_pre_trails_ms),
                        fmt(self.dispatch_timings.sim_pre_slope_ms),
                        fmt(self.dispatch_timings.sim_pre_draw_ms),
                        fmt(self.dispatch_timings.sim_pre_repro_ms),
                    );
                }
            }
            self.capture_this_frame = false;
            self.next_sample_time = std::time::Instant::now() + self.sample_interval;
            return;
        }

        // Bench CPU-only path: if GPU profiling is disabled (either unsupported or opt-out),
        // still advance the bench counter and request exit when done.
        if let Some(bench) = &mut self.bench {
            if !self.gpu_active {
                let done = bench.on_sample(None, None, None, None, self.last_cpu_update_ms);
                if done {
                    bench.print_summary_once();
                    self.bench_exit_requested = true;
                }
                return;
            }
        }

        if !self.gpu_supported {
            return;
        }
        if self.query_readback.is_empty() {
            return;
        }

        // If there is a pending map, try to complete it first.
        if let Some((pending_slot, rx)) = self.pending_map.take() {
            if self.wait_for_readback {
                device.poll(wgpu::Maintain::Wait);
                let _ = rx.recv();
            } else {
                device.poll(wgpu::Maintain::Poll);
                if rx.try_recv().is_err() {
                    self.pending_map = Some((pending_slot, rx));
                    return;
                }
            }

            let pending_readback = &self.query_readback[pending_slot];
            let slice = pending_readback.slice(..);
            let stamps: Vec<u64> = {
                let view = slice.get_mapped_range();
                view.chunks_exact(8)
                    .take(GPU_TS_COUNT as usize)
                    .map(|chunk| {
                        let bytes: [u8; 8] = chunk.try_into().unwrap();
                        u64::from_le_bytes(bytes)
                    })
                    .collect()
            };
            pending_readback.unmap();
            self.readback_busy[pending_slot] = false;

            self.print_stamps(&stamps);
            return;
        }

        // Start mapping the most recently copied slot (or any busy slot if we missed it).
        let slot = if let Some(slot) = self.last_copied_slot {
            slot
        } else {
            return;
        };
        if !self.readback_busy[slot] {
            return;
        }

        let readback = &self.query_readback[slot];
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.pending_map = Some((slot, rx));

        // In blocking mode, complete it immediately so prints happen on the expected cadence.
        if self.wait_for_readback {
            device.poll(wgpu::Maintain::Wait);
            if let Some((pending_slot, rx)) = self.pending_map.take() {
                let _ = rx.recv();
                let pending_readback = &self.query_readback[pending_slot];
                let slice = pending_readback.slice(..);
                let stamps: Vec<u64> = {
                    let view = slice.get_mapped_range();
                    view.chunks_exact(8)
                        .take(GPU_TS_COUNT as usize)
                        .map(|chunk| {
                            let bytes: [u8; 8] = chunk.try_into().unwrap();
                            u64::from_le_bytes(bytes)
                        })
                        .collect()
                };
                pending_readback.unmap();
                self.readback_busy[pending_slot] = false;
                self.print_stamps(&stamps);
            }
        }
    }

    fn print_stamps(&mut self, stamps: &[u64]) {
        let dt_ms = |a: u32, b: u32| -> Option<f64> {
            let a = stamps.get(a as usize).copied().unwrap_or(0);
            let b = stamps.get(b as usize).copied().unwrap_or(0);
            if b <= a {
                return None;
            }
            Some(((b - a) as f64 * self.timestamp_period_ns) / 1_000_000.0)
        };

        let gpu_update_total = dt_ms(TS_UPDATE_START, TS_UPDATE_END);

        let gpu_sim = dt_ms(TS_SIM_PASS_START, TS_UPDATE_AFTER_SIM);
        let gpu_sim_diffuse = dt_ms(TS_SIM_PASS_START, TS_SIM_AFTER_DIFFUSE);
        let gpu_sim_diffuse_commit = dt_ms(TS_SIM_AFTER_DIFFUSE, TS_SIM_AFTER_DIFFUSE_COMMIT);
        let gpu_sim_trails = dt_ms(TS_SIM_AFTER_DIFFUSE_COMMIT, TS_SIM_AFTER_TRAILS_PREP);
        let gpu_sim_slope = dt_ms(TS_SIM_AFTER_TRAILS_PREP, TS_SIM_AFTER_SLOPE);
        let gpu_sim_clear_visual = dt_ms(TS_SIM_AFTER_SLOPE, TS_SIM_AFTER_CLEAR_VISUAL);
        let gpu_sim_motion_blur = dt_ms(TS_SIM_AFTER_CLEAR_VISUAL, TS_SIM_AFTER_MOTION_BLUR);
        let gpu_sim_clear_agent_grid = dt_ms(TS_SIM_AFTER_MOTION_BLUR, TS_SIM_AFTER_CLEAR_AGENT_GRID);
        // NOTE: Spatial grid clear was removed (epoch-stamped grid). This segment now includes
        // the reproduction compute work (and any other pre-sim work in Pass 1).
        let gpu_sim_repro_and_pre = dt_ms(TS_SIM_AFTER_CLEAR_AGENT_GRID, TS_SIM_AFTER_CLEAR_SPATIAL);
        let gpu_sim_populate_spatial = dt_ms(TS_SIM_AFTER_CLEAR_SPATIAL, TS_SIM_AFTER_POPULATE_SPATIAL);
        let gpu_sim_clear_force_vectors = dt_ms(TS_SIM_AFTER_POPULATE_SPATIAL, TS_SIM_AFTER_CLEAR_FLUID_FORCE_VECTORS);
        let gpu_sim_process = dt_ms(TS_SIM_AFTER_CLEAR_FLUID_FORCE_VECTORS, TS_SIM_AFTER_PROCESS_AGENTS);
        let gpu_sim_microswim = dt_ms(TS_SIM_AFTER_PROCESS_AGENTS, TS_SIM_AFTER_MICROSWIM);
        let gpu_sim_drain = dt_ms(TS_SIM_AFTER_MICROSWIM, TS_SIM_AFTER_DRAIN_ENERGY);

        let gpu_fluid = dt_ms(TS_FLUID_PASS_START, TS_UPDATE_AFTER_FLUID);
        let gpu_fluid_fumarole_force = dt_ms(TS_FLUID_PASS_START, TS_FLUID_AFTER_FUMAROLE_FORCE);
        let gpu_fluid_generate = dt_ms(TS_FLUID_AFTER_FUMAROLE_FORCE, TS_FLUID_AFTER_GENERATE_FORCES);
        let gpu_fluid_add = dt_ms(TS_FLUID_AFTER_GENERATE_FORCES, TS_FLUID_AFTER_ADD_FORCES);
        let gpu_fluid_clear = dt_ms(TS_FLUID_AFTER_ADD_FORCES, TS_FLUID_AFTER_CLEAR_FORCES);
        let gpu_fluid_diffuse = dt_ms(TS_FLUID_AFTER_CLEAR_FORCES, TS_FLUID_AFTER_DIFFUSE_VELOCITY);
        let gpu_fluid_advect = dt_ms(TS_FLUID_AFTER_DIFFUSE_VELOCITY, TS_FLUID_AFTER_ADVECT_VELOCITY);
        let gpu_fluid_vorticity = dt_ms(TS_FLUID_AFTER_ADVECT_VELOCITY, TS_FLUID_AFTER_VORTICITY);
        let gpu_fluid_div = dt_ms(TS_FLUID_AFTER_VORTICITY, TS_FLUID_AFTER_DIVERGENCE);
        let gpu_fluid_clear_pressure = dt_ms(TS_FLUID_AFTER_DIVERGENCE, TS_FLUID_AFTER_CLEAR_PRESSURE);
        let gpu_fluid_jacobi = dt_ms(TS_FLUID_AFTER_CLEAR_PRESSURE, TS_FLUID_AFTER_JACOBI);
        let gpu_fluid_grad = dt_ms(TS_FLUID_AFTER_JACOBI, TS_FLUID_AFTER_SUBTRACT_GRADIENT);
        let gpu_fluid_bounds = dt_ms(TS_FLUID_AFTER_SUBTRACT_GRADIENT, TS_FLUID_AFTER_BOUNDARIES);
        let gpu_fluid_inject_dye = dt_ms(TS_FLUID_AFTER_BOUNDARIES, TS_FLUID_AFTER_INJECT_DYE);
        let gpu_fluid_advect_dye = dt_ms(TS_FLUID_AFTER_INJECT_DYE, TS_FLUID_AFTER_ADVECT_DYE);
        let gpu_fluid_fumarole_dye = dt_ms(TS_FLUID_AFTER_ADVECT_DYE, TS_FLUID_AFTER_FUMAROLE_DYE);
        let gpu_fluid_advect_trail = dt_ms(TS_FLUID_AFTER_FUMAROLE_DYE, TS_FLUID_AFTER_ADVECT_TRAIL);

        // No-fluid fallback micro-breakdown (only meaningful when `!fluid_enabled`).
        let gpu_nofluid_dye_diffuse = dt_ms(TS_FLUID_PASS_START, TS_NOFLUID_AFTER_DYE_DIFFUSE);
        let gpu_nofluid_dye_copyback = dt_ms(TS_NOFLUID_AFTER_DYE_DIFFUSE, TS_NOFLUID_AFTER_DYE_COPYBACK);
        let gpu_nofluid_trail_commit = dt_ms(TS_NOFLUID_AFTER_DYE_COPYBACK, TS_NOFLUID_AFTER_TRAIL_COMMIT);

        let gpu_post = dt_ms(TS_POST_PASS_START, TS_UPDATE_AFTER_POST);
        let gpu_post_compact = dt_ms(TS_POST_PASS_START, TS_POST_AFTER_COMPACT);
        let gpu_post_cpu_spawn = dt_ms(TS_POST_AFTER_COMPACT, TS_POST_AFTER_CPU_SPAWN);
        let gpu_post_merge = dt_ms(TS_POST_AFTER_CPU_SPAWN, TS_POST_AFTER_MERGE);
        let gpu_post_init_dead = dt_ms(TS_POST_AFTER_MERGE, TS_POST_AFTER_INIT_DEAD);
        let gpu_post_finalize = dt_ms(TS_POST_AFTER_INIT_DEAD, TS_POST_AFTER_FINALIZE_MERGE);
        let gpu_post_paused_process = dt_ms(TS_POST_AFTER_FINALIZE_MERGE, TS_POST_AFTER_PROCESS_PAUSED_RENDER_ONLY);
        let gpu_post_render_agents = dt_ms(TS_POST_AFTER_PROCESS_PAUSED_RENDER_ONLY, TS_POST_AFTER_RENDER_AGENTS);
        let gpu_post_inspector_clear = dt_ms(TS_POST_AFTER_RENDER_AGENTS, TS_POST_AFTER_INSPECTOR_CLEAR);
        let gpu_post_inspector_draw = dt_ms(TS_POST_AFTER_INSPECTOR_CLEAR, TS_POST_AFTER_INSPECTOR_DRAW);

        let gpu_comp = dt_ms(TS_COMPOSITE_PASS_START, TS_UPDATE_AFTER_COMPOSITE);
        let gpu_copy = dt_ms(TS_UPDATE_AFTER_COMPOSITE, TS_UPDATE_AFTER_COPY);

        let gpu_render = dt_ms(TS_RENDER_ENC_START, TS_RENDER_ENC_END);
        let gpu_render_main = dt_ms(TS_RENDER_MAIN_START, TS_RENDER_MAIN_END);
        let gpu_render_overlay = dt_ms(TS_RENDER_OVERLAY_START, TS_RENDER_OVERLAY_END);

        let gpu_egui = dt_ms(TS_EGUI_ENC_START, TS_EGUI_ENC_END);
        let gpu_egui_pass = dt_ms(TS_EGUI_PASS_START, TS_EGUI_PASS_END);

        if let Some(bench) = &mut self.bench {
            let done = bench.on_sample(
                gpu_sim,
                gpu_sim_diffuse,
                gpu_sim_process,
                gpu_sim_drain,
                self.last_cpu_update_ms,
            );
            if done {
                bench.print_summary_once();
                self.bench_exit_requested = true;
            }

            if !self.bench_verbose {
                return;
            }
        }

        println!(
            "[perf] gpu(ms): update={:>7.2} sim={:>7.2} fluid={:>7.2} post={:>7.2} composite={:>7.2} copy={:>7.2} render={:>7.2} egui={:>7.2} | cpu(ms): update={:>7.2} render={:>7.2} egui={:>7.2}",
            gpu_update_total.unwrap_or(-1.0),
            gpu_sim.unwrap_or(-1.0),
            gpu_fluid.unwrap_or(-1.0),
            gpu_post.unwrap_or(-1.0),
            gpu_comp.unwrap_or(-1.0),
            gpu_copy.unwrap_or(-1.0),
            gpu_render.unwrap_or(-1.0),
            gpu_egui.unwrap_or(-1.0),
            self.last_cpu_update_ms,
            self.last_cpu_render_ms,
            self.last_cpu_egui_ms,
        );

        println!(
            "[perf] sim-kernels(ms): diffuse={:>6.2} commit={:>6.2} trails={:>6.2} slope={:>6.2} clearVis={:>6.2} blur={:>6.2} clrGrid={:>6.2} reproPre={:>6.2} popSpat={:>6.2} clrForce={:>6.2} process={:>6.2} microswim={:>6.2} drain={:>6.2}",
            gpu_sim_diffuse.unwrap_or(-1.0),
            gpu_sim_diffuse_commit.unwrap_or(-1.0),
            gpu_sim_trails.unwrap_or(-1.0),
            gpu_sim_slope.unwrap_or(-1.0),
            gpu_sim_clear_visual.unwrap_or(-1.0),
            gpu_sim_motion_blur.unwrap_or(-1.0),
            gpu_sim_clear_agent_grid.unwrap_or(-1.0),
            gpu_sim_repro_and_pre.unwrap_or(-1.0),
            gpu_sim_populate_spatial.unwrap_or(-1.0),
            gpu_sim_clear_force_vectors.unwrap_or(-1.0),
            gpu_sim_process.unwrap_or(-1.0),
            gpu_sim_microswim.unwrap_or(-1.0),
            gpu_sim_drain.unwrap_or(-1.0),
        );

        println!(
            "[perf] fluid-kernels(ms): fumaF={:>6.2} genF={:>6.2} addF={:>6.2} clrF={:>6.2} diffV={:>6.2} advV={:>6.2} vort={:>6.2} div={:>6.2} clrP={:>6.2} jacobi={:>6.2} grad={:>6.2} bound={:>6.2} injD={:>6.2} advD={:>6.2} fumaD={:>6.2} advT={:>6.2}",
            gpu_fluid_fumarole_force.unwrap_or(-1.0),
            gpu_fluid_generate.unwrap_or(-1.0),
            gpu_fluid_add.unwrap_or(-1.0),
            gpu_fluid_clear.unwrap_or(-1.0),
            gpu_fluid_diffuse.unwrap_or(-1.0),
            gpu_fluid_advect.unwrap_or(-1.0),
            gpu_fluid_vorticity.unwrap_or(-1.0),
            gpu_fluid_div.unwrap_or(-1.0),
            gpu_fluid_clear_pressure.unwrap_or(-1.0),
            gpu_fluid_jacobi.unwrap_or(-1.0),
            gpu_fluid_grad.unwrap_or(-1.0),
            gpu_fluid_bounds.unwrap_or(-1.0),
            gpu_fluid_inject_dye.unwrap_or(-1.0),
            gpu_fluid_advect_dye.unwrap_or(-1.0),
            gpu_fluid_fumarole_dye.unwrap_or(-1.0),
            gpu_fluid_advect_trail.unwrap_or(-1.0),
        );

        println!(
            "[perf] nofluid(ms): dyeDiffuse={:>6.2} dyeCopy={:>6.2} trailCommit={:>6.2}",
            gpu_nofluid_dye_diffuse.unwrap_or(-1.0),
            gpu_nofluid_dye_copyback.unwrap_or(-1.0),
            gpu_nofluid_trail_commit.unwrap_or(-1.0),
        );

        println!(
            "[perf] post-kernels(ms): compact={:>6.2} cpuSpawn={:>6.2} merge={:>6.2} initDead={:>6.2} finalize={:>6.2} pausedProc={:>6.2} renderAgents={:>6.2} inspClr={:>6.2} inspDraw={:>6.2} inspOn={}",
            gpu_post_compact.unwrap_or(-1.0),
            gpu_post_cpu_spawn.unwrap_or(-1.0),
            gpu_post_merge.unwrap_or(-1.0),
            gpu_post_init_dead.unwrap_or(-1.0),
            gpu_post_finalize.unwrap_or(-1.0),
            gpu_post_paused_process.unwrap_or(-1.0),
            gpu_post_render_agents.unwrap_or(-1.0),
            gpu_post_inspector_clear.unwrap_or(-1.0),
            gpu_post_inspector_draw.unwrap_or(-1.0),
            if self.last_inspector_requested { 1 } else { 0 },
        );

        println!(
            "[perf] render(ms): enc={:>6.2} main={:>6.2} overlay={:>6.2} | egui(ms): enc={:>6.2} pass={:>6.2}",
            gpu_render.unwrap_or(-1.0),
            gpu_render_main.unwrap_or(-1.0),
            gpu_render_overlay.unwrap_or(-1.0),
            gpu_egui.unwrap_or(-1.0),
            gpu_egui_pass.unwrap_or(-1.0),
        );

        // If requested, auto-exit after N captured samples.
        if let Some(max) = self.profile_max_captures {
            self.profile_captures_done = self.profile_captures_done.saturating_add(1);
            if self.profile_captures_done >= max {
                self.bench_exit_requested = true;
            }
        }
    }
}

#[allow(dead_code)]
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
        is_propeller: false,
        is_condenser: false,
        is_displacer: false,
    }, // P
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
#[allow(dead_code)]
fn rgb_to_color32_with_alpha(rgb: [f32; 3], alpha: f32) -> egui::Color32 {
    egui::Color32::from_rgba_unmultiplied(
        rgb_component(rgb[0]),
        rgb_component(rgb[1]),
        rgb_component(rgb[2]),
        rgb_component(alpha),
    )
}

#[inline]
fn genome_get_base_ascii(genome: &[u32; GENOME_WORDS], idx: usize) -> u8 {
    if idx >= GENOME_BYTES {
        return b'X';
    }
    let word = genome[idx / 4];
    let shift = (idx % 4) * 8;
    ((word >> shift) & 0xFF) as u8
}

#[inline]
fn genome_base_2bit_to_ascii(v: u32) -> u8 {
    match v & 3 {
        0 => b'A',
        1 => b'U',
        2 => b'G',
        _ => b'C',
    }
}

#[inline]
fn genome_base_ascii_to_2bit(b: u8) -> u32 {
    match b {
        b'A' => 0,
        b'U' => 1,
        b'G' => 2,
        b'C' => 3,
        _ => 0,
    }
}

#[inline]
fn genome_packed_get_base_ascii(
    packed: &[u32; GENOME_PACKED_WORDS],
    idx: usize,
    offset: u32,
    len: u32,
) -> u8 {
    if idx >= GENOME_BYTES {
        return b'X';
    }
    let idx_u32 = idx as u32;
    if idx_u32 < offset || idx_u32 >= offset.saturating_add(len) {
        return b'X';
    }
    let word_index = idx / GENOME_BASES_PER_PACKED_WORD;
    let bit_index = (idx % GENOME_BASES_PER_PACKED_WORD) * 2;
    let word_val = packed[word_index];
    let two_bits = (word_val >> bit_index) & 0x3;
    genome_base_2bit_to_ascii(two_bits)
}

fn genome_packed_to_ascii_words(
    packed: &[u32; GENOME_PACKED_WORDS],
    offset: u32,
    len: u32,
) -> [u32; GENOME_WORDS] {
    let mut out = [0u32; GENOME_WORDS];
    for idx in 0..GENOME_BYTES {
        let b = genome_packed_get_base_ascii(packed, idx, offset, len);
        let word = idx / 4;
        let shift = (idx % 4) * 8;
        out[word] |= (b as u32) << shift;
    }
    out
}

fn genome_pack_ascii_words(
    genome_ascii: &[u32; GENOME_WORDS],
) -> (u32, u32, [u32; GENOME_PACKED_WORDS]) {
    // Interpret the active region as the span [first_non_x, last_non_x] (inclusive).
    // NOTE: This assumes 'X' is used only for padding (no internal X holes).
    let mut first = GENOME_BYTES;
    let mut last = 0usize;
    for i in 0..GENOME_BYTES {
        let b = genome_get_base_ascii(genome_ascii, i);
        if b != b'X' {
            if first == GENOME_BYTES {
                first = i;
            }
            last = i;
        }
    }

    if first == GENOME_BYTES {
        return (0, 0, [0u32; GENOME_PACKED_WORDS]);
    }

    let offset = first as u32;
    let len = (last - first + 1) as u32;
    let mut packed = [0u32; GENOME_PACKED_WORDS];

    for idx in first..=last {
        let base = genome_get_base_ascii(genome_ascii, idx);
        if base == b'X' {
            continue;
        }
        let code = genome_base_ascii_to_2bit(base);
        let bi = idx as u32;
        let wi = (bi / GENOME_BASES_PER_PACKED_WORD as u32) as usize;
        let bit_index = (bi % GENOME_BASES_PER_PACKED_WORD as u32) * 2;
        packed[wi] |= code << bit_index;
    }

    (len, offset, packed)
}

#[inline]
fn genome_base_color(base: u8) -> egui::Color32 {
    match base {
        b'A' => egui::Color32::from_rgb(0, 255, 0),
        b'U' => egui::Color32::from_rgb(0, 128, 255),
        b'G' => egui::Color32::from_rgb(255, 166, 0),
        b'C' => egui::Color32::from_rgb(255, 0, 0),
        b'X' => egui::Color32::from_rgb(128, 128, 128),
        _ => egui::Color32::from_rgb(80, 80, 80),
    }
}

#[inline]
fn part_base_color_rgb(base_type: u32) -> [f32; 3] {
    // 0�19: amino acids; 20�42: organs (colors mirror shaders/shared.wgsl AMINO_DATA[*][2].xyz)
    if (base_type as usize) < AMINO_COLORS.len() {
        return AMINO_COLORS[base_type as usize];
    }

    match base_type {
        20 => [1.0, 1.0, 0.0],   // mouth
        21 => [0.0, 0.0, 1.0],   // propeller
        22 => [0.0, 1.0, 0.0],   // alpha sensor
        23 => [1.0, 0.0, 0.0],   // beta sensor
        24 => [0.6, 0.0, 0.8],   // energy sensor
        25 => [0.0, 0.39, 1.0],  // alpha emitter (repurposed displacer)
        26 => [1.0, 1.0, 1.0],   // enabler
        27 => [0.0, 0.39, 1.0],  // beta emitter (repurposed displacer)
        28 => [1.0, 0.5, 0.0],   // storage
        29 => [1.0, 0.4, 0.7],   // poison resistance
        30 => [1.0, 0.0, 1.0],   // chiral flipper
        31 => [1.0, 0.0, 1.0],   // clock
        32 => [0.0, 0.8, 0.8],   // slope sensor
        33 => [1.0, 0.0, 0.0],   // vampire mouth
        34 => [0.2, 0.0, 0.2],
        35 => [0.26, 0.26, 0.26],
        36 => [0.5, 0.5, 0.5],
        37 => [0.8, 0.6, 0.2],
        38 => [0.2, 0.9, 0.3],
        39 => [0.3, 1.0, 0.4],
        40 => [0.9, 0.2, 0.3],
        41 => [1.0, 0.3, 0.4],
        42 => [0.6, 0.0, 0.8],
        44 => [0.78, 0.55, 0.78], // BetaMouth (mauve)
        45 => [0.6, 0.7, 0.9], // ATTRACTOR_REPULSOR (matches shaders/shared.wgsl)
        46 => [0.2, 0.9, 0.2], // SPIKE (bright green - matches shaders/shared.wgsl)
        _ => DEFAULT_PART_COLOR,
    }
}

#[inline]
fn part_base_color32(base_type: u32) -> egui::Color32 {
    rgb_to_color32(part_base_color_rgb(base_type))
}

#[inline]
fn part_base_name(base_type: u32) -> &'static str {
    match base_type {
        0 => "A",
        1 => "C",
        2 => "D",
        3 => "E",
        4 => "F",
        5 => "G",
        6 => "H",
        7 => "I",
        8 => "K",
        9 => "L",
        10 => "M",
        11 => "N",
        12 => "P",
        13 => "Q",
        14 => "R",
        15 => "S",
        16 => "T",
        17 => "V",
        18 => "W",
        19 => "Y",
        20 => "Mouth",
        21 => "Propeller",
        22 => "Alpha Sensor",
        23 => "Beta Sensor",
        24 => "Energy Sensor",
        25 => "Alpha Emitter",
        26 => "Enabler",
        27 => "Beta Emitter",
        28 => "Storage",
        29 => "Poison Resistance",
        30 => "Chiral Flipper",
        31 => "Clock",
        32 => "Slope Sensor",
        33 => "Vampire Mouth",
        34 => "Agent Alpha Sensor",
        35 => "Agent Beta Sensor",
        36 => "Pairing Sensor",
        37 => "Trail Energy Sensor",
        38 => "Alpha Magnitude Sensor",
        39 => "Alpha Magnitude Sensor (var)",
        40 => "Beta Magnitude Sensor",
        41 => "Beta Magnitude Sensor (var)",
        42 => "Anchor",
        43 => "Mutation Protection",
        44 => "BetaMouth",
        45 => "ATTRAC",
        46 => "Spike",
        _ => "Organ",
    }
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

#[derive(Clone, Copy, Debug)]
struct TranslationStepCpu {
    bases_consumed: usize,
    is_stop: bool,
    is_valid: bool,
}

#[inline]
fn genome_find_first_coding_triplet_cpu(genome: &[u32; GENOME_WORDS]) -> Option<usize> {
    for i in 0..=(GENOME_BYTES.saturating_sub(3)) {
        let b0 = genome_get_base_ascii(genome, i);
        let b1 = genome_get_base_ascii(genome, i + 1);
        let b2 = genome_get_base_ascii(genome, i + 2);
        if b0 == b'X' || b1 == b'X' || b2 == b'X' {
            continue;
        }
        return Some(i);
    }
    None
}

#[inline]
fn genome_find_start_codon_cpu(genome: &[u32; GENOME_WORDS]) -> Option<usize> {
    for i in 0..=(GENOME_BYTES.saturating_sub(3)) {
        let b0 = genome_get_base_ascii(genome, i);
        let b1 = genome_get_base_ascii(genome, i + 1);
        let b2 = genome_get_base_ascii(genome, i + 2);
        if b0 == b'A' && b1 == b'U' && b2 == b'G' {
            return Some(i);
        }
    }
    None
}

#[inline]
fn genome_is_stop_codon_triplet(b0: u8, b1: u8, b2: u8) -> bool {
    // UAA, UAG, UGA
    (b0 == b'U' && b1 == b'A' && (b2 == b'A' || b2 == b'G')) || (b0 == b'U' && b1 == b'G' && b2 == b'A')
}

#[inline]
fn translate_codon_step_cpu(
    genome: &[u32; GENOME_WORDS],
    pos_b: usize,
    ignore_stop_codons: bool,
) -> TranslationStepCpu {
    if pos_b + 2 >= GENOME_BYTES {
        return TranslationStepCpu {
            bases_consumed: 3,
            is_stop: false,
            is_valid: false,
        };
    }

    let b0 = genome_get_base_ascii(genome, pos_b);
    let b1 = genome_get_base_ascii(genome, pos_b + 1);
    let b2 = genome_get_base_ascii(genome, pos_b + 2);
    let has_x = b0 == b'X' || b1 == b'X' || b2 == b'X';
    let is_stop = !has_x && genome_is_stop_codon_triplet(b0, b1, b2);

    if has_x || is_stop {
        // Mirrors WGSL: X always terminates; stop codons only translate when ignore_stop_codons=true.
        return TranslationStepCpu {
            bases_consumed: 3,
            is_stop,
            is_valid: ignore_stop_codons && !has_x,
        };
    }

    let amino_type = codon_to_amino_acid(b0, b1, b2);
    let is_promoter = matches!(amino_type, 9 | 12 | 8 | 1 | 17 | 10 | 6 | 13);

    let mut bases_consumed = 3usize;
    if is_promoter && pos_b + 5 < GENOME_BYTES {
        let b3 = genome_get_base_ascii(genome, pos_b + 3);
        let b4 = genome_get_base_ascii(genome, pos_b + 4);
        let b5 = genome_get_base_ascii(genome, pos_b + 5);
        let second_has_x = b3 == b'X' || b4 == b'X' || b5 == b'X';
        if !second_has_x {
            let second_is_stop = genome_is_stop_codon_triplet(b3, b4, b5);
            if second_is_stop {
                // Match WGSL: stop codon can't act as organ modifier.
                // Normal mode: consume only promoter (stop will terminate next step).
                // Debug ignore-stop mode: skip over stop codon (consume 6) but still no organ.
                bases_consumed = if ignore_stop_codons { 6 } else { 3 };
            } else {
            bases_consumed = 6;
            }
        }
    }

    TranslationStepCpu {
        bases_consumed,
        is_stop: false,
        is_valid: true,
    }
}

fn build_translation_map_for_inspector(
    genome: &[u32; GENOME_WORDS],
    body_count: usize,
    require_start_codon: bool,
    ignore_stop_codons: bool,
) -> Vec<bool> {
    let mut map = vec![false; GENOME_BYTES];
    if body_count == 0 {
        return map;
    }

    let gene_start = genome_find_first_coding_triplet_cpu(genome);
    let translation_start = if require_start_codon {
        genome_find_start_codon_cpu(genome)
    } else {
        gene_start
    };

    let Some(translation_start) = translation_start else {
        return map;
    };

    // Mark the start codon itself as part of the active translation region (even though it is not a body part).
    if require_start_codon && translation_start + 2 < GENOME_BYTES {
        map[translation_start] = true;
        map[translation_start + 1] = true;
        map[translation_start + 2] = true;
    }

    let mut pos_b = translation_start + if require_start_codon { 3 } else { 0 };
    let mut parts_emitted = 0usize;
    while parts_emitted < body_count {
        let step = translate_codon_step_cpu(genome, pos_b, ignore_stop_codons);
        if !step.is_valid {
            break;
        }
        let end = (pos_b + step.bases_consumed).min(GENOME_BYTES);
        for i in pos_b..end {
            map[i] = true;
        }
        pos_b = end;
        parts_emitted += 1;
    }

    // If the next codon is a stop codon and stop codons terminate translation, mark it as active too.
    let tail = translate_codon_step_cpu(genome, pos_b, ignore_stop_codons);
    if !tail.is_valid && tail.is_stop {
        if pos_b + 2 < GENOME_BYTES {
            map[pos_b] = true;
            map[pos_b + 1] = true;
            map[pos_b + 2] = true;
        }
    }

    map
}

fn compute_body_part_nucleotide_spans_for_inspector(
    genome: &[u32; GENOME_WORDS],
    body_count: usize,
    require_start_codon: bool,
    ignore_stop_codons: bool,
) -> Vec<f32> {
    if body_count == 0 {
        return Vec::new();
    }

    let gene_start = genome_find_first_coding_triplet_cpu(genome);
    let translation_start = if require_start_codon {
        genome_find_start_codon_cpu(genome)
    } else {
        gene_start
    };

    let Some(translation_start) = translation_start else {
        return Vec::new();
    };

    let mut spans = Vec::with_capacity(body_count);
    let mut pos_b = translation_start + if require_start_codon { 3 } else { 0 };
    for _ in 0..body_count {
        let step = translate_codon_step_cpu(genome, pos_b, ignore_stop_codons);
        if !step.is_valid {
            break;
        }
        spans.push(step.bases_consumed as f32);
        pos_b = (pos_b + step.bases_consumed).min(GENOME_BYTES);
    }
    spans
}

#[allow(dead_code)]
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

#[allow(dead_code)]
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
    let genome_ascii = genome_packed_to_ascii_words(&agent.genome_packed, agent.genome_offset, agent.gene_length);
    let genome_str = genome_to_string(&genome_ascii);
    let agent_name = naming::agent::generate_agent_name(agent);
    let genome_prefix = &genome_str[0..8.min(genome_str.len())];
    let default_name = format!("agent_{}_g{}_{}.json", agent_name, agent.generation, genome_prefix);

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
    #[allow(dead_code)]
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
    gene_length: u32,                          // number of active (non-padding) bases
    genome_offset: u32,                        // left padding so active bases are in [offset, offset+len)
    genome_packed: [u32; GENOME_PACKED_WORDS], // 2-bit packed genome (A/U/G/C)
    body: [BodyPart; MAX_BODY_PARTS],
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
    genome_override_len: u32,
    genome_override_offset: u32,
    genome_override_packed: [u32; GENOME_PACKED_WORDS],
    _pad_genome: [u32; 2],
}
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SimParams {
    dt: f32,
    frame_dt: f32,
    drag: f32,
    energy_cost: f32,
    amino_maintenance_cost: f32,
    morphology_change_cost: f32,
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
    gamma_diffuse: f32,
    gamma_blur: f32,
    gamma_shift: f32,
    alpha_slope_bias: f32,
    beta_slope_bias: f32,
    alpha_multiplier: f32,
    beta_multiplier: f32,
    _pad_rain0: u32,
    _pad_rain1: u32,
    // Targeted rain dispatch: number of rain drops to spawn this frame
    rain_drop_count: u32,
    alpha_rain_drop_count: u32,
    // Global scalar for rain/precipitation injection into chem grids.
    // 0 disables precipitation entirely.
    dye_precipitation: f32,
    chemical_slope_scale_alpha: f32,
    chemical_slope_scale_beta: f32,
    mutation_rate: f32,
    food_power: f32,
    poison_power: f32,
    pairing_cost: f32,
    max_agents: u32,
    cpu_spawn_count: u32,
    // NOTE: `agent_count` is used as the dispatch/scan bound in shaders.
    // It is typically set to buffer capacity for correctness (no skipped newborns).
    agent_count: u32,
    // Actual alive population estimate (may lag by a frame); used for population-pressure math.
    population_count: u32,
    random_seed: u32,
    debug_mode: u32,           // 0 = off, 1 = per-segment debug overlay
    visual_stride: u32,        // pixels per row in visual_grid buffer (padded)
    selected_agent_index: u32, // Index of selected agent for debug visualization (u32::MAX if none)
    repulsion_strength: f32,
    agent_repulsion_strength: f32,
    gamma_strength: f32,
    prop_wash_strength: f32,
    prop_wash_strength_fluid: f32,
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
    // Global multiplier for how much the fluid vector field pushes agents.
    // NOTE: This reuses the previous padding slot to avoid changing buffer layouts.
    fluid_wind_push_strength: f32,

    // Fluid-directed convolution strength (separate from classic diffusion blur).
    alpha_fluid_convolution: f32,
    beta_fluid_convolution: f32,
    // Fluid terrain influence parameters
    fluid_slope_force_scale: f32,   // How strongly terrain slope drives fluid flow (default: 100.0)
    fluid_obstacle_strength: f32,   // How strongly steep terrain blocks fluid (default: 200.0)

    // Fluid dye visualization colors (independent from environment alpha/beta colors).
    // Stored as two 16-byte blocks for stable uniform alignment.
    dye_alpha_color_r: f32,
    dye_alpha_color_g: f32,
    dye_alpha_color_b: f32,
    _pad_dye_alpha_color: f32,
    dye_beta_color_r: f32,
    dye_beta_color_g: f32,
    dye_beta_color_b: f32,
    _pad_dye_beta_color: f32,
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

    // Part base-angle overrides: packed vec4s with NaN sentinel (NaN = use shader default).
    // 128 slots reserved.
    part_angle_override: [[f32; 4]; PART_OVERRIDE_VEC4S],

    // Part property overrides: 47 parts × 6 vec4 blocks.
    // NaN sentinel per component means "use shader default".
    part_props_override_head: [[f32; 4]; PART_PROPS_OVERRIDE_VEC4S_HEAD],
    part_props_override_tail: [[f32; 4]; PART_PROPS_OVERRIDE_VEC4S_TAIL],

    // Optional override of AMINO_FLAGS bitmask.
    // Packed as vec4<f32> lanes with NaN sentinel = "use shader default".
    part_flags_override: [[f32; 4]; PART_FLAGS_OVERRIDE_VEC4S],
}

// Keep host layout in sync with the WGSL uniform buffer (std140).
// Keep this assertion updated if the uniform layout changes.
// (47 parts × 6 vec4) = 282 vec4 overrides.
const _: [(); 5344] = [(); std::mem::size_of::<EnvironmentInitParams>()];

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

// Default functions for backward compatibility with old settings files
fn default_alpha_fluid_convolution() -> f32 { 0.05 }
fn default_beta_fluid_convolution() -> f32 { 0.05 }
fn default_fluid_enabled() -> bool { true }
fn default_fluid_slope_force_scale() -> f32 { 100.0 }
fn default_fluid_obstacle_strength() -> f32 { 200.0 }
fn default_fluid_ooze_still_rate() -> f32 { 1.0 }
fn default_fluid_ooze_rate_beta_unset() -> f32 { -1.0 }
fn default_fluid_ooze_fade_rate_beta_unset() -> f32 { -1.0 }
fn default_fluid_ooze_rate_gamma_unset() -> f32 { -1.0 }
fn default_fluid_ooze_fade_rate_gamma_unset() -> f32 { -1.0 }
fn default_dye_diffusion() -> f32 { 0.01 }
fn default_dye_diffusion_no_fluid() -> f32 { 0.15 }
fn default_env_grid_resolution() -> u32 { DEFAULT_ENV_GRID_RESOLUTION }
fn default_fluid_grid_resolution() -> u32 { DEFAULT_FLUID_GRID_RESOLUTION }
fn default_spatial_grid_resolution() -> u32 { DEFAULT_SPATIAL_GRID_RESOLUTION }
fn default_fluid_dye_escape_rate() -> f32 { 0.0 }
fn default_fluid_dye_escape_rate_beta_unset() -> f32 { -1.0 }
fn default_prop_wash_strength_fluid_unset() -> f32 { -1.0 }
fn default_dye_alpha_color() -> [f32; 3] { [0.0, 1.0, 0.0] }
fn default_dye_beta_color() -> [f32; 3] { [1.0, 0.0, 0.0] }
fn default_dye_thinfilm_multiplier() -> f32 { 50.0 }
fn default_dye_precipitation() -> f32 { 1.0 }

fn default_microswim_enabled() -> bool { true }
fn default_propellers_enabled() -> bool { true }
fn default_morphology_change_cost() -> f32 { 0.0 }
fn default_microswim_coupling() -> f32 { 1.0 }
fn default_microswim_base_drag() -> f32 { 0.2 }
fn default_microswim_anisotropy() -> f32 { 5.0 }
fn default_microswim_max_frame_vel() -> f32 { 2.0 }
fn default_microswim_torque_strength() -> f32 { 0.1 }
fn default_microswim_min_seg_displacement() -> f32 { 0.005 }
fn default_microswim_min_total_deformation_sq() -> f32 { 0.0001 }
fn default_microswim_min_length_ratio() -> f32 { 0.8 }
fn default_microswim_max_length_ratio() -> f32 { 1.25 }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
struct FumaroleSettings {
    enabled: bool,
    // Position as fraction of the fluid grid (0..1).
    x_frac: f32,
    y_frac: f32,
    // Direction in degrees (0..360). Converted to unit vector for the shader.
    dir_degrees: f32,
    // Force strength injected into the fluid.
    strength: f32,
    // Radius in fluid-cell units.
    spread: f32,
    // Dye injection rate (per second, scaled by dt in shader).
    alpha_dye_rate: f32,
    #[serde(default)]
    beta_dye_rate: f32,
    // 0..1 jitter applied to direction/strength/rate.
    variation: f32,
}

impl Default for FumaroleSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            x_frac: 0.5,
            y_frac: 0.5,
            dir_degrees: 0.0,
            strength: 200.0,
            spread: 16.0,
            alpha_dye_rate: 1.0,
            beta_dye_rate: 0.0,
            variation: 0.0,
        }
    }
}

impl FumaroleSettings {
    fn sanitize(&mut self) {
        self.x_frac = self.x_frac.clamp(0.0, 1.0);
        self.y_frac = self.y_frac.clamp(0.0, 1.0);
        if !self.dir_degrees.is_finite() {
            self.dir_degrees = 0.0;
        }
        // Wrap into [0, 360).
        self.dir_degrees = self.dir_degrees.rem_euclid(360.0);
        self.strength = self.strength.clamp(0.0, 50_000.0);
        self.spread = self.spread.clamp(0.0, FLUID_GRID_SIZE as f32);
        // Backward-compat shim: older sessions may have used negative dye_rate to mean beta.
        // If alpha_dye_rate is negative, migrate magnitude into beta and clamp alpha to 0.
        if self.alpha_dye_rate < 0.0 {
            self.beta_dye_rate = self.beta_dye_rate.max(-self.alpha_dye_rate);
            self.alpha_dye_rate = 0.0;
        }
        self.alpha_dye_rate = self.alpha_dye_rate.clamp(0.0, 1000.0);
        self.beta_dye_rate = self.beta_dye_rate.clamp(0.0, 1000.0);
        self.variation = self.variation.clamp(0.0, 1.0);
    }
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
    gamma_diffuse: f32,
    #[serde(default = "default_alpha_fluid_convolution")]
    alpha_fluid_convolution: f32,
    #[serde(default = "default_beta_fluid_convolution")]
    beta_fluid_convolution: f32,
    #[serde(default = "default_fluid_slope_force_scale")]
    fluid_slope_force_scale: f32,
    #[serde(default = "default_fluid_obstacle_strength")]
    fluid_obstacle_strength: f32,
    gamma_blur: f32,
    gamma_shift: f32,
    alpha_slope_bias: f32,
    beta_slope_bias: f32,
    alpha_multiplier: f32,
    beta_multiplier: f32,
    #[serde(default = "default_dye_precipitation")]
    dye_precipitation: f32,
    alpha_rain_map_path: Option<PathBuf>,
    beta_rain_map_path: Option<PathBuf>,
    chemical_slope_scale_alpha: f32,
    chemical_slope_scale_beta: f32,
    food_power: f32,
    poison_power: f32,
    amino_maintenance_cost: f32,
    #[serde(default = "default_morphology_change_cost")]
    morphology_change_cost: f32,
    pairing_cost: f32,
    prop_wash_strength: f32,
    #[serde(default = "default_prop_wash_strength_fluid_unset")]
    prop_wash_strength_fluid: f32,

    // Propellers (organ-based propulsion) runtime toggle.
    // Kept independent from microswimming.
    #[serde(default = "default_propellers_enabled")]
    propellers_enabled: bool,

    // Microswimming (morphology-based propulsion) tuning.
    #[serde(default = "default_microswim_enabled")]
    microswim_enabled: bool,
    #[serde(default = "default_microswim_coupling")]
    microswim_coupling: f32,
    #[serde(default = "default_microswim_base_drag")]
    microswim_base_drag: f32,
    #[serde(default = "default_microswim_anisotropy")]
    microswim_anisotropy: f32,
    #[serde(default = "default_microswim_max_frame_vel")]
    microswim_max_frame_vel: f32,
    #[serde(default = "default_microswim_torque_strength")]
    microswim_torque_strength: f32,
    #[serde(default = "default_microswim_min_seg_displacement")]
    microswim_min_seg_displacement: f32,
    #[serde(default = "default_microswim_min_total_deformation_sq")]
    microswim_min_total_deformation_sq: f32,
    #[serde(default = "default_microswim_min_length_ratio")]
    microswim_min_length_ratio: f32,
    #[serde(default = "default_microswim_max_length_ratio")]
    microswim_max_length_ratio: f32,
    repulsion_strength: f32,
    agent_repulsion_strength: f32,
    limit_fps: bool,
    limit_fps_25: bool,
    render_interval: u32,
    gamma_debug_visual: bool,
    slope_debug_visual: bool,
    rain_debug_visual: bool,  // Visualization mode for rain patterns
    #[serde(default = "default_fluid_enabled")]
    fluid_enabled: bool,      // Enable/disable the fluid simulation + coupling
    fluid_show: bool,         // Show fluid simulation overlay
    fluid_dt: f32,            // Fluid solver dt (used by stable-fluids steps)
    fluid_decay: f32,         // Fluid decay/dissipation factor
    fluid_jacobi_iters: u32,  // Pressure solve iterations
    fluid_vorticity: f32,     // Vorticity confinement strength (0 disables)
    fluid_viscosity: f32,     // Viscosity (nu) for velocity diffusion (0 disables)
    // Lift/sedimentation controls for chem<->dye coupling (shared across alpha/beta/gamma).
    // NOTE: field names are legacy; see shaders/fluid.wgsl for the current mapping.
    fluid_ooze_rate: f32,      // lift_min_speed (dye-cells/sec)
    fluid_ooze_fade_rate: f32, // lift_multiplier (1/(dye-cells/sec))
    #[serde(default = "default_fluid_ooze_rate_beta_unset")]
    fluid_ooze_rate_beta: f32, // sedimentation_min_speed (dye-cells/sec)
    #[serde(default = "default_fluid_ooze_fade_rate_beta_unset")]
    fluid_ooze_fade_rate_beta: f32, // sedimentation_multiplier (1/(dye-cells/sec))
    #[serde(default = "default_fluid_ooze_rate_gamma_unset")]
    fluid_ooze_rate_gamma: f32, // (unused / legacy)
    #[serde(default = "default_fluid_ooze_fade_rate_gamma_unset")]
    fluid_ooze_fade_rate_gamma: f32, // (unused / legacy)
    #[serde(default = "default_fluid_ooze_still_rate")]
    fluid_ooze_still_rate: f32, // Chem?dye baseline ooze in still water (per-second fraction)
    #[serde(default = "default_fluid_dye_escape_rate")]
    fluid_dye_escape_rate: f32, // Dye decay (ALPHA): removes dye without depositing back into chem (1/sec)
    #[serde(default = "default_fluid_dye_escape_rate_beta_unset")]
    fluid_dye_escape_rate_beta: f32, // Dye decay (BETA): removes dye without depositing back into chem (1/sec)
    #[serde(default = "default_dye_diffusion")]
    dye_diffusion: f32,  // Dye diffusion strength when fluid is enabled (blend fraction per step)
    #[serde(default = "default_dye_diffusion_no_fluid")]
    dye_diffusion_no_fluid: f32,  // Dye diffusion strength when fluid is disabled (per epoch)
    fluid_wind_push_strength: f32, // Global multiplier for how much fluid pushes agents
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
    #[serde(default)]
    trail_show_energy: bool,
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
    #[serde(default = "default_dye_alpha_color")]
    dye_alpha_color: [f32; 3],
    #[serde(default = "default_dye_beta_color")]
    dye_beta_color: [f32; 3],

    // Fluid dye visualization mode controls.
    // When enabled, the dye is colored using a thin-film interference palette instead of a flat tint.
    #[serde(default)]
    dye_alpha_thinfilm: bool,
    #[serde(default = "default_dye_thinfilm_multiplier")]
    dye_alpha_thinfilm_mult: f32,
    #[serde(default)]
    dye_beta_thinfilm: bool,
    #[serde(default = "default_dye_thinfilm_multiplier")]
    dye_beta_thinfilm_mult: f32,

    // Fluid fumaroles (persisted in settings + snapshots). Sent to the fluid shader as a flat f32 buffer.
    #[serde(default)]
    fumaroles: Vec<FumaroleSettings>,
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

    // Resolution settings (requires restart to take effect)
    #[serde(default = "default_env_grid_resolution")]
    env_grid_resolution: u32,  // Environment grid (alpha/beta/gamma) resolution
    #[serde(default = "default_fluid_grid_resolution")]
    fluid_grid_resolution: u32,  // Fluid simulation grid resolution
    #[serde(default = "default_spatial_grid_resolution")]
    spatial_grid_resolution: u32,  // Spatial hash grid resolution
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
            alpha_blur: 0.05,
            beta_blur: 0.05,
            gamma_diffuse: 0.0,
            // If omitted in older settings files, serde defaults handle it
            alpha_fluid_convolution: 0.05,
            beta_fluid_convolution: 0.05,
            gamma_blur: 0.9995,
            gamma_shift: 0.0,
            alpha_slope_bias: -5.0,
            beta_slope_bias: 5.0,
            alpha_multiplier: 0.0001,
            beta_multiplier: 0.0,
            dye_precipitation: 1.0,
            alpha_rain_map_path: None,
            beta_rain_map_path: None,
            chemical_slope_scale_alpha: 0.1,
            chemical_slope_scale_beta: 0.1,
            food_power: 3.0,
            poison_power: 1.0,
            amino_maintenance_cost: 0.001,
            morphology_change_cost: default_morphology_change_cost(),
            pairing_cost: 0.1,
            prop_wash_strength: 1.0,
            prop_wash_strength_fluid: default_prop_wash_strength_fluid_unset(),

            propellers_enabled: default_propellers_enabled(),

            microswim_enabled: default_microswim_enabled(),
            microswim_coupling: default_microswim_coupling(),
            microswim_base_drag: default_microswim_base_drag(),
            microswim_anisotropy: default_microswim_anisotropy(),
            microswim_max_frame_vel: default_microswim_max_frame_vel(),
            microswim_torque_strength: default_microswim_torque_strength(),
            microswim_min_seg_displacement: default_microswim_min_seg_displacement(),
            microswim_min_total_deformation_sq: default_microswim_min_total_deformation_sq(),
            microswim_min_length_ratio: default_microswim_min_length_ratio(),
            microswim_max_length_ratio: default_microswim_max_length_ratio(),
            repulsion_strength: 10.0,
            agent_repulsion_strength: 1.0,
            limit_fps: true,
            limit_fps_25: false,
            render_interval: 100, // Draw every 100 steps in fast mode
            gamma_debug_visual: false,
            slope_debug_visual: false,
            rain_debug_visual: false,
            fluid_enabled: true,
            fluid_show: false,  // Fluid simulation overlay disabled by default
            fluid_dt: 0.016,
            fluid_decay: 0.995,
            fluid_jacobi_iters: 31,
            fluid_vorticity: 0.0,
            fluid_viscosity: 0.05,
            // Lift/sedimentation controls for chem<->dye coupling (shared across alpha/beta/gamma).
            // Lift: requires minimum speed, then increases with speed.
            fluid_ooze_rate: 0.2,      // lift_min_speed (dye-cells/sec)
            fluid_ooze_fade_rate: 2.5, // lift_multiplier (1/(dye-cells/sec))
            // Sedimentation: requires low speed, then increases as speed decreases.
            fluid_ooze_rate_beta: 0.2,      // sedimentation_min_speed (dye-cells/sec)
            fluid_ooze_fade_rate_beta: 2.5, // sedimentation_multiplier (1/(dye-cells/sec))
            // Gamma uses alpha defaults unless overridden.
            fluid_ooze_rate_gamma: default_fluid_ooze_rate_gamma_unset(),
            fluid_ooze_fade_rate_gamma: default_fluid_ooze_fade_rate_gamma_unset(),
            // Baseline chem?dye seepage in still water so sensors can read signals without flow.
            fluid_ooze_still_rate: default_fluid_ooze_still_rate(),
            // Dye escape: removes dye without precipitation (independent sink).
            fluid_dye_escape_rate: default_fluid_dye_escape_rate(),
            // Beta channel uses alpha defaults unless overridden.
            fluid_dye_escape_rate_beta: default_fluid_dye_escape_rate_beta_unset(),
            dye_diffusion: default_dye_diffusion(),
            dye_diffusion_no_fluid: default_dye_diffusion_no_fluid(),
            // Matches the previous hardcoded scale used in shaders/simulation.wgsl.
            fluid_wind_push_strength: 0.0005,
            fluid_slope_force_scale: 100.0,
            fluid_obstacle_strength: 200.0,
            vector_force_power: 0.0,  // Disabled by default
            vector_force_x: 0.0,
            vector_force_y: -1.0,     // Downward gravity when enabled
            gamma_hidden: false,
            debug_per_segment: false,
            gamma_vis_min: 0.0,
            gamma_vis_max: 50.0,
            alpha_show: true,
            beta_show: true,
            gamma_show: true,
            slope_lighting: false,
            slope_lighting_strength: 1.0,
            trail_diffusion: 0.15,
            trail_decay: 0.995,
            trail_opacity: 0.5,
            trail_show: false,
            trail_show_energy: false,
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
            dye_alpha_color: default_dye_alpha_color(),
            dye_beta_color: default_dye_beta_color(),
            dye_alpha_thinfilm: false,
            dye_alpha_thinfilm_mult: default_dye_thinfilm_multiplier(),
            dye_beta_thinfilm: false,
            dye_beta_thinfilm_mult: default_dye_thinfilm_multiplier(),
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

            fumaroles: Vec::new(),

            // Resolution settings
            env_grid_resolution: DEFAULT_ENV_GRID_RESOLUTION,
            fluid_grid_resolution: DEFAULT_FLUID_GRID_RESOLUTION,
            spatial_grid_resolution: DEFAULT_SPATIAL_GRID_RESOLUTION,
        }
    }
}

// ============================================================================
// SNAPSHOT SAVE/LOAD STRUCTURES
// ============================================================================

fn scrub_json_nulls(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            // Drop null keys to allow #[serde(default)] to fill values.
            let null_keys: Vec<String> = map
                .iter()
                .filter_map(|(k, v)| v.is_null().then(|| k.clone()))
                .collect();
            for k in null_keys {
                map.remove(&k);
            }

            for v in map.values_mut() {
                scrub_json_nulls(v);
            }
        }
        serde_json::Value::Array(arr) => {
            // Preserve array length: replace null elements with 0.0 (common for legacy float arrays).
            for v in arr.iter_mut() {
                if v.is_null() {
                    *v = serde_json::Value::Number(serde_json::Number::from_f64(0.0).unwrap());
                } else {
                    scrub_json_nulls(v);
                }
            }
        }
        _ => {}
    }
}

fn de_f32_null_default<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<f32>::deserialize(deserializer)?.unwrap_or(0.0))
}

fn de_f32_null_default_one<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<f32>::deserialize(deserializer)?.unwrap_or(1.0))
}

fn de_u32_null_default<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<u32>::deserialize(deserializer)?.unwrap_or(0))
}

fn de_vec2_f32_null_default<'de, D>(deserializer: D) -> Result<[f32; 2], D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt = Option::<[Option<f32>; 2]>::deserialize(deserializer)?;
    Ok(match opt {
        Some([x, y]) => [x.unwrap_or(0.0), y.unwrap_or(0.0)],
        None => [0.0, 0.0],
    })
}

fn de_settings_option_lossy<'de, D>(deserializer: D) -> Result<Option<SimulationSettings>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    let Some(mut v) = value else {
        return Ok(None);
    };
    scrub_json_nulls(&mut v);

    // If settings are malformed, treat them as absent (use current defaults).
    Ok(serde_json::from_value::<SimulationSettings>(v).ok())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AgentSnapshot {
    #[serde(default, deserialize_with = "de_vec2_f32_null_default")]
    position: [f32; 2],
    #[serde(default, deserialize_with = "de_f32_null_default")]
    rotation: f32,
    #[serde(default, deserialize_with = "de_f32_null_default_one")]
    energy: f32,
    #[serde(default, deserialize_with = "de_u32_null_default")]
    generation: u32,
    #[serde(default)]
    genome: Vec<u8>, // Store as bytes for compression
}

impl From<&Agent> for AgentSnapshot {
    fn from(agent: &Agent) -> Self {
        // Extract genome as bytes (legacy ASCII with X padding)
        let genome_ascii =
            genome_packed_to_ascii_words(&agent.genome_packed, agent.genome_offset, agent.gene_length);
        let mut genome_bytes = Vec::with_capacity(GENOME_BYTES);
        for word in &genome_ascii {
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

        let (genome_override_len, genome_override_offset, genome_override_packed) =
            genome_pack_ascii_words(&genome_override);

        SpawnRequest {
            seed: 0,
            genome_seed: 0,
            flags: 1, // Use genome override
            _pad_seed: 0,
            position: self.position,
            energy: self.energy,
            rotation: self.rotation,
            genome_override_len,
            genome_override_offset,
            genome_override_packed,
            _pad_genome: [0u32; 2],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SimulationSnapshot {
    #[serde(default)]
    version: String,
    #[serde(default)]
    timestamp: String,
    #[serde(default)]
    run_name: String,
    #[serde(default)]
    epoch: u64,
    #[serde(default, deserialize_with = "de_settings_option_lossy")]
    settings: Option<SimulationSettings>,
    #[serde(default)]
    agents: Vec<AgentSnapshot>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    rain_map_blob: Option<String>,
    // Resolution information for compatibility checking
    #[serde(default)]
    env_grid_resolution: u32,
    #[serde(default)]
    fluid_grid_resolution: u32,
    #[serde(default)]
    spatial_grid_resolution: u32,
}

impl SimulationSnapshot {
    fn new(
        epoch: u64,
        agents: &[Agent],
        settings: SimulationSettings,
        run_name: String,
        rain_map_blob: Option<String>,
        env_grid_res: u32,
        fluid_grid_res: u32,
        spatial_grid_res: u32,
    ) -> Self {
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();

        // agents should already be filtered for alive != 0 by caller
        // Randomly sample up to MAX_SNAPSHOT_AGENTS agents if there are more
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();

        let sampled_agents: Vec<&Agent> = if agents.len() <= MAX_SNAPSHOT_AGENTS {
            // Keep all agents
            agents.iter().collect()
        } else {
            // Sample MAX_SNAPSHOT_AGENTS random agents
            agents
                .iter()
                .collect::<Vec<_>>()
                .choose_multiple(&mut rng, MAX_SNAPSHOT_AGENTS)
                .copied()
                .collect()
        };

        let agent_snapshots: Vec<AgentSnapshot> = sampled_agents
            .iter()
            .map(|a| AgentSnapshot::from(*a))
            .collect();

        println!("Snapshot created with {} agents (from {} living)", agent_snapshots.len(), agents.len());

        Self {
            version: SNAPSHOT_VERSION.to_string(),
            timestamp,
            run_name,
            epoch,
            settings: Some(settings),
            agents: agent_snapshots,
            rain_map_blob,
            env_grid_resolution: env_grid_res,
            fluid_grid_resolution: fluid_grid_res,
            spatial_grid_resolution: spatial_grid_res,
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
        self.spawn_probability = self.spawn_probability.clamp(0.0, 5.0);
        self.death_probability = self.death_probability.clamp(0.0, 0.1);
        self.mutation_rate = self.mutation_rate.clamp(0.0, 0.1);
        // auto_replenish requires no sanitizing
        self.diffusion_interval = self.diffusion_interval.clamp(1, 64);
        self.slope_interval = self.slope_interval.clamp(1, 64);
        self.alpha_blur = self.alpha_blur.clamp(0.0, 1.0);
        self.beta_blur = self.beta_blur.clamp(0.0, 1.0);
        self.gamma_diffuse = self.gamma_diffuse.clamp(0.0, 1.0);
        // Fluid convolution has been removed; keep fields for backward-compat/uniform layout but force them off.
        self.alpha_fluid_convolution = 0.0;
        self.beta_fluid_convolution = 0.0;
        self.gamma_blur = self.gamma_blur.clamp(0.0, 1.0);
        self.gamma_shift = self.gamma_shift.clamp(0.0, 1.0);
        self.alpha_slope_bias = self.alpha_slope_bias.clamp(-10.0, 10.0);
        self.beta_slope_bias = self.beta_slope_bias.clamp(-10.0, 10.0);
        self.alpha_multiplier = self.alpha_multiplier.clamp(0.00001, 2.0);
        self.beta_multiplier = self.beta_multiplier.clamp(0.00001, 2.0);
        self.dye_precipitation = self.dye_precipitation.clamp(0.0, 1.0);
        self.chemical_slope_scale_alpha = self.chemical_slope_scale_alpha.clamp(0.0, 1.0);
        self.chemical_slope_scale_beta = self.chemical_slope_scale_beta.clamp(0.0, 1.0);
        self.food_power = self.food_power.clamp(0.0, 10.0);
        self.poison_power = self.poison_power.clamp(0.0, 10.0);
        self.amino_maintenance_cost = self.amino_maintenance_cost.clamp(0.0, 0.01);
        self.morphology_change_cost = self.morphology_change_cost.clamp(0.0, 10.0);
        self.pairing_cost = self.pairing_cost.clamp(0.0, 1.0);
        self.prop_wash_strength = self.prop_wash_strength.clamp(0.0, 5.0);
        // Back-compat: older settings files won't have the fluid prop-wash strength.
        // Use -1 sentinel default to mean "inherit direct wash".
        if self.prop_wash_strength_fluid < 0.0 {
            self.prop_wash_strength_fluid = self.prop_wash_strength;
        }
        self.prop_wash_strength_fluid = self.prop_wash_strength_fluid.clamp(-5.0, 5.0);

        // Microswimming
        self.microswim_coupling = self.microswim_coupling.clamp(0.0, 10.0);
        self.microswim_base_drag = self.microswim_base_drag.clamp(0.0, 5.0);
        self.microswim_anisotropy = self.microswim_anisotropy.clamp(0.0, 50.0);
        self.microswim_max_frame_vel = self.microswim_max_frame_vel.clamp(0.0, 20.0);
        self.microswim_torque_strength = self.microswim_torque_strength.clamp(0.0, 5.0);
        self.microswim_min_seg_displacement = self.microswim_min_seg_displacement.clamp(0.0, 0.5);
        self.microswim_min_total_deformation_sq = self.microswim_min_total_deformation_sq.clamp(0.0, 10.0);
        self.microswim_min_length_ratio = self.microswim_min_length_ratio.clamp(0.0, 5.0);
        self.microswim_max_length_ratio = self.microswim_max_length_ratio.clamp(0.0, 5.0);
        if self.microswim_min_length_ratio > self.microswim_max_length_ratio {
            std::mem::swap(
                &mut self.microswim_min_length_ratio,
                &mut self.microswim_max_length_ratio,
            );
        }
        self.repulsion_strength = self.repulsion_strength.clamp(0.0, 100.0);
        self.agent_repulsion_strength = self.agent_repulsion_strength.clamp(0.0, 10.0);
        self.render_interval = self.render_interval.clamp(1, 10_000);
        // Gamma is unbounded in simulation; visualization range must allow large values.
        self.gamma_vis_min = self.gamma_vis_min.clamp(0.0, 100_000.0);
        self.gamma_vis_max = self.gamma_vis_max.clamp(0.0, 100_000.0);
        if self.gamma_vis_min >= self.gamma_vis_max {
            self.gamma_vis_max = (self.gamma_vis_min + 0.001).clamp(0.0, 100_000.0);
            self.gamma_vis_min = (self.gamma_vis_max - 0.001).clamp(0.0, 100_000.0);
        }
        self.alpha_rain_variation = self.alpha_rain_variation.clamp(0.0, 1.0);
        self.beta_rain_variation = self.beta_rain_variation.clamp(0.0, 1.0);
        self.alpha_rain_phase = self.alpha_rain_phase.clamp(0.0, std::f32::consts::PI * 2.0);
        self.beta_rain_phase = self.beta_rain_phase.clamp(0.0, std::f32::consts::PI * 2.0);
        self.alpha_rain_freq = self.alpha_rain_freq.clamp(0.0, 100.0);
        self.beta_rain_freq = self.beta_rain_freq.clamp(0.0, 100.0);

        self.fluid_dt = self.fluid_dt.clamp(0.001, 0.05);
        self.fluid_decay = self.fluid_decay.clamp(0.5, 1.0);
        self.fluid_jacobi_iters = self.fluid_jacobi_iters.clamp(1, 128);
        self.fluid_vorticity = self.fluid_vorticity.clamp(0.0, 50.0);
        self.fluid_viscosity = self.fluid_viscosity.clamp(0.0, 5.0);
        self.dye_diffusion = self.dye_diffusion.clamp(0.0, 1.0);
        self.dye_diffusion_no_fluid = self.dye_diffusion_no_fluid.clamp(0.0, 1.0);

        // Validate and enforce resolution constraints
        self.env_grid_resolution = self.env_grid_resolution.clamp(256, 8192);
        self.fluid_grid_resolution = self.fluid_grid_resolution.clamp(64, 2048);
        self.spatial_grid_resolution = self.spatial_grid_resolution.clamp(64, 2048);

        // Enforce ratio: fluid and spatial should be env_res / 4 (or maintain user's ratio if reasonable)
        let ratio = self.env_grid_resolution / self.fluid_grid_resolution.max(1);
        if ratio < 2 || ratio > 16 {
            // Force 4:1 ratio if current ratio is unreasonable
            self.fluid_grid_resolution = self.env_grid_resolution / 4;
            self.spatial_grid_resolution = self.env_grid_resolution / 4;
        }

        // Back-compat: older settings files won't have beta thresholds.
        // Use -1 sentinel defaults to mean "inherit alpha".
        if self.fluid_ooze_rate_beta < 0.0 {
            self.fluid_ooze_rate_beta = self.fluid_ooze_rate;
        }
        if self.fluid_ooze_fade_rate_beta < 0.0 {
            self.fluid_ooze_fade_rate_beta = self.fluid_ooze_fade_rate;
        }
        if self.fluid_dye_escape_rate_beta < 0.0 {
            self.fluid_dye_escape_rate_beta = self.fluid_dye_escape_rate;
        }

        // Back-compat: older settings files won't have gamma thresholds.
        // Use -1 sentinel defaults to mean "inherit alpha".
        if self.fluid_ooze_rate_gamma < 0.0 {
            self.fluid_ooze_rate_gamma = self.fluid_ooze_rate;
        }
        if self.fluid_ooze_fade_rate_gamma < 0.0 {
            self.fluid_ooze_fade_rate_gamma = self.fluid_ooze_fade_rate;
        }

        self.fluid_ooze_rate = self.fluid_ooze_rate.clamp(0.0, 100.0);
        self.fluid_ooze_fade_rate = self.fluid_ooze_fade_rate.clamp(0.0, 50.0);
        self.fluid_ooze_rate_beta = self.fluid_ooze_rate_beta.clamp(0.0, 100.0);
        self.fluid_ooze_fade_rate_beta = self.fluid_ooze_fade_rate_beta.clamp(0.0, 50.0);
        self.fluid_ooze_rate_gamma = self.fluid_ooze_rate_gamma.clamp(0.0, 100.0);
        self.fluid_ooze_fade_rate_gamma = self.fluid_ooze_fade_rate_gamma.clamp(0.0, 50.0);
        self.fluid_ooze_still_rate = self.fluid_ooze_still_rate.clamp(0.0, 100.0);
        self.fluid_dye_escape_rate = self.fluid_dye_escape_rate.clamp(0.0, 50.0);
        self.fluid_dye_escape_rate_beta = self.fluid_dye_escape_rate_beta.clamp(0.0, 50.0);
        self.fluid_wind_push_strength = self.fluid_wind_push_strength.clamp(0.0, 2.0);
        self.fluid_slope_force_scale = self.fluid_slope_force_scale.clamp(0.0, 500.0);
        self.fluid_obstacle_strength = self.fluid_obstacle_strength.clamp(0.0, 1000.0);

        if self.fumaroles.len() > MAX_FUMAROLES {
            self.fumaroles.truncate(MAX_FUMAROLES);
        }
        for f in &mut self.fumaroles {
            f.sanitize();
        }

        for c in &mut self.dye_alpha_color {
            *c = c.clamp(0.0, 1.0);
        }
        for c in &mut self.dye_beta_color {
            *c = c.clamp(0.0, 1.0);
        }

        self.dye_alpha_thinfilm_mult = self.dye_alpha_thinfilm_mult.clamp(0.001, 10_000.0);
        self.dye_beta_thinfilm_mult = self.dye_beta_thinfilm_mult.clamp(0.001, 10_000.0);
    }
}

// ============================================================================
// GPU STATE
// ============================================================================

#[allow(dead_code)]
struct GpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,

    // Resolution settings (immutable after creation, used for buffer allocation and shader constants)
    env_grid_resolution: u32,
    fluid_grid_resolution: u32,
    spatial_grid_resolution: u32,
    env_grid_cell_count: usize,
    fluid_grid_cell_count: usize,
    spatial_grid_cell_count: usize,

    // World size in world-space units (scaled proportionally to env_grid_resolution).
    sim_size: f32,

    // Buffers
    agents_buffer_a: wgpu::Buffer,
    agents_buffer_b: wgpu::Buffer,
    chem_grid: wgpu::Buffer,
    rain_map_buffer: wgpu::Buffer,
    rain_map_texture: wgpu::Texture,
    rain_map_texture_view: wgpu::TextureView,
    agent_spatial_grid_buffer: wgpu::Buffer,
    gamma_grid: wgpu::Buffer,
    trail_grid: wgpu::Buffer,
    trail_grid_inject: wgpu::Buffer,
    trail_debug_readback: wgpu::Buffer,
    visual_grid_buffer: wgpu::Buffer,
    agent_grid_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    microswim_params_buffer: wgpu::Buffer,

    // CPU-side mirror of the uniform params buffer.
    // Some compute paths (e.g. snapshot-load spawning) can run before `update()` has populated
    // the params buffer for the current frame.
    sim_params_cpu: SimParams,
    microswim_params_cpu: [f32; MICROSWIM_PARAM_FLOATS],
    environment_init_cpu: EnvironmentInitParams,
    environment_init_params_buffer: wgpu::Buffer,
    spawn_debug_counters: wgpu::Buffer, // [spawn_counter, debug_counter, alive_counter]
    init_dead_dispatch_args: wgpu::Buffer, // Indirect dispatch args buffer for init-dead (x,y,z)
    init_dead_params_buffer: wgpu::Buffer, // 16-byte uniform: [max_agents, 0, 0, 0]
    init_dead_writer_bind_group_layout: wgpu::BindGroupLayout,
    init_dead_writer_bind_group: wgpu::BindGroup,
    reproduction_bind_group_a: wgpu::BindGroup,
    reproduction_bind_group_b: wgpu::BindGroup,
    reproduction_pipeline: wgpu::ComputePipeline,
    alive_readbacks: [Arc<wgpu::Buffer>; 2],
    alive_readback_pending: [Arc<Mutex<Option<Result<(u32, u32), ()>>>>; 2],
    alive_readback_inflight: [bool; 2],
    alive_readback_slot: usize,
    alive_readback_last_applied_epoch: u32,
    alive_readback_zero_streak: u8,
    debug_readback: wgpu::Buffer,
    agents_readback: wgpu::Buffer, // Readback for agent inspection
    selected_agent_buffer: wgpu::Buffer, // GPU buffer for selected agent
    selected_agent_readbacks: Vec<Arc<wgpu::Buffer>>, // CPU readbacks (ring) for selected agent
    selected_agent_readback_pending: Vec<Arc<Mutex<Option<Result<Agent, ()>>>>>,
    selected_agent_readback_inflight: Vec<bool>,
    selected_agent_readback_slot: usize,
    selected_agent_readback_last_request: std::time::Instant,
    new_agents_buffer: wgpu::Buffer, // Buffer for spawned agents
    spawn_readback: wgpu::Buffer,  // Readback for spawn count
    spawn_requests_buffer: wgpu::Buffer, // CPU spawn requests seeds

    // Grid readback buffers for snapshot save
    chem_grid_readback: wgpu::Buffer,
    gamma_grid_readback: wgpu::Buffer,

    // Fluid simulation buffers (128x128 grid)
    fluid_velocity_a: wgpu::Buffer,
    fluid_velocity_b: wgpu::Buffer,
    fluid_pressure_a: wgpu::Buffer,
    fluid_pressure_b: wgpu::Buffer,
    fluid_divergence: wgpu::Buffer,
    fluid_forces: wgpu::Buffer,
    fluid_force_vectors: wgpu::Buffer,
    fluid_dye_a: wgpu::Buffer,
    fluid_dye_b: wgpu::Buffer,
    fluid_params_buffer: wgpu::Buffer,
    fluid_fumaroles_buffer: wgpu::Buffer,

    // Part base-angle overrides (radians). NaN means: use shader default.
    part_base_angle_overrides: [f32; PART_OVERRIDE_SLOTS],
    part_base_angle_overrides_dirty: bool,

    // Part (amino + organ) property overrides, mirroring shaders/shared.wgsl AMINO_DATA layout.
    // NaN sentinel per component means: use shader default.
    part_props_override: [[f32; 4]; PART_PROPS_OVERRIDE_VEC4S],
    part_flags_override: [f32; PART_TYPE_COUNT], // NaN sentinel = use shader default; stored as numeric bitmask
    part_properties_dirty: bool,

    // Parsed shader defaults (for UI to show effective values).
    part_props_defaults: [[f32; 4]; PART_PROPS_OVERRIDE_VEC4S],
    // Full parsed shader defaults for UI display (may exceed fixed override buffer).
    part_props_defaults_full: Vec<[f32; 4]>,
    part_flags_defaults: [u32; PART_TYPE_COUNT],

    // UI
    show_part_properties_editor: bool,

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
    fluid_clear_forces_pipeline: wgpu::ComputePipeline,
    fluid_clear_force_vectors_pipeline: wgpu::ComputePipeline,
    fluid_fumarole_dye_pipeline: wgpu::ComputePipeline,
    fluid_fumarole_pipeline: wgpu::ComputePipeline,
    fluid_clear_velocity_pipeline: wgpu::ComputePipeline,
    fluid_inject_dye_pipeline: wgpu::ComputePipeline,
    fluid_advect_dye_pipeline: wgpu::ComputePipeline,
    fluid_advect_trail_pipeline: wgpu::ComputePipeline,
    fluid_clear_dye_pipeline: wgpu::ComputePipeline,

    // Non-fluid fallback: keep dye/trail layers usable as isotropic diffusion fields.
    fluid_diffuse_dye_no_fluid_pipeline: wgpu::ComputePipeline,
    fluid_copy_trail_no_fluid_pipeline: wgpu::ComputePipeline,

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
    microswim_pipeline: wgpu::ComputePipeline,
    clear_fluid_force_vectors_pipeline: wgpu::ComputePipeline,
    diffuse_pipeline: wgpu::ComputePipeline,
    diffuse_commit_pipeline: wgpu::ComputePipeline,
    diffuse_trails_pipeline: wgpu::ComputePipeline,
    rain_pipeline: wgpu::ComputePipeline,
    clear_visual_pipeline: wgpu::ComputePipeline,
    motion_blur_pipeline: wgpu::ComputePipeline,
    clear_agent_grid_pipeline: wgpu::ComputePipeline,
    clear_inspector_preview_pipeline: wgpu::ComputePipeline,
    render_agents_pipeline: wgpu::ComputePipeline, // Render all agents to agent_grid
    draw_inspector_agent_pipeline: wgpu::ComputePipeline,
    composite_agents_pipeline: wgpu::ComputePipeline,
    gamma_slope_pipeline: wgpu::ComputePipeline,
    merge_pipeline: wgpu::ComputePipeline, // Merge spawned agents
    compact_pipeline: wgpu::ComputePipeline, // Remove dead agents
    finalize_merge_pipeline: wgpu::ComputePipeline, // Reset spawn counter
    cpu_spawn_pipeline: wgpu::ComputePipeline, // Materialize CPU spawn requests on GPU
    write_init_dead_dispatch_args_pipeline: wgpu::ComputePipeline,
    initialize_dead_pipeline: wgpu::ComputePipeline, // Sanitize unused agent slots
    environment_init_pipeline: wgpu::ComputePipeline, // Fill alpha/beta/gamma/trails on GPU
    generate_map_pipeline: wgpu::ComputePipeline, // Generate specific map (flat/noise)
    clear_agent_spatial_grid_pipeline: wgpu::ComputePipeline, // Clear agent spatial grid
    populate_agent_spatial_grid_pipeline: wgpu::ComputePipeline, // Populate agent spatial grid
    drain_energy_pipeline: wgpu::ComputePipeline, // Vampire mouths drain energy from neighbors
    spike_kill_pipeline: wgpu::ComputePipeline, // Spike organs kill on contact
    render_pipeline: wgpu::RenderPipeline,
    inspector_overlay_pipeline: wgpu::RenderPipeline,

    // Bind groups
    compute_bind_group_a: wgpu::BindGroup,
    compute_bind_group_b: wgpu::BindGroup,
    composite_bind_group: wgpu::BindGroup,
    render_bind_group: wgpu::BindGroup,
    inspector_overlay_bind_group: wgpu::BindGroup,

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
    last_present_time: std::time::Instant,
    last_egui_update_time: std::time::Instant,
    cached_egui_primitives: Vec<egui::ClippedPrimitive>,

    frame_profiler: FrameProfiler,

    // Safe performance measurement mode (does not use timestamp queries).
    gpu_sync_profile: bool,
    gpu_sync_profile_interval_frames: u32,
    gpu_sync_profile_frame_counter: u32,
    gpu_sync_profile_accum_update_ms: f64,
    gpu_sync_profile_accum_render_ms: f64,

    // Perf A/B toggles (env-var controlled)
    perf_skip_trail_prep: bool,
    perf_skip_nofluid_dye: bool,
    perf_skip_nofluid_trail: bool,

    // Perf toggles (UI controlled)
    perf_skip_diffusion: bool,
    perf_skip_rain: bool,
    perf_skip_slope: bool,
    perf_skip_draw: bool,
    perf_skip_repro: bool,
    perf_force_gpu_sync: bool,

    // Simulation speed control
    render_interval: u32, // Draw every N steps in fast mode
    current_mode: u32,    // 0=VSync 60 FPS, 1=Full Speed, 2=Fast Draw, 3=Slow 25 FPS
    epoch: u64,           // Total simulation steps elapsed

    // Epoch tracking for speed display
    last_epoch_update: std::time::Instant,
    last_epoch_count: u64,

    // Inspector refresh throttling (update at ~25fps to save GPU time)
    inspector_frame_counter: u32,
    trail_energy_debug_next_epoch: u64,
    epochs_per_second: f32,

    // Population statistics tracking
    population_history: Vec<u32>, // Stores population at sample points
    population_plot_points: Vec<[f64; 2]>,
    alpha_rain_history: VecDeque<f32>,
    beta_rain_history: VecDeque<f32>,
    epoch_sample_interval: u64,   // Sample every N epochs (1000)
    last_sample_epoch: u64,       // Last epoch when we sampled
    max_history_points: usize,    // Maximum data points (5000)

    // Auto-snapshot tracking
    last_autosave_epoch: u64,     // Last epoch when auto-snapshot was saved

    // Debug: population scan for specific organ presence
    organ45_alive_with: u32,
    organ45_total_with: u32,
    organ45_last_scan: Option<std::time::Instant>,
    organ45_last_scan_error: Option<String>,

    // Run naming
    run_seed: u32,
    run_name: String,

    // Mouse dragging state
    is_dragging: bool,
    is_zoom_dragging: bool,
    zoom_drag_start_y: f32,
    zoom_drag_start_zoom: f32,
    last_mouse_pos: Option<[f32; 2]>,

    // Agent selection for debug panel
    selected_agent_index: Option<usize>,
    selected_agent_data: Option<Agent>,
    follow_selected_agent: bool, // New field
    camera_target: [f32; 2], // Target position for smooth camera following
    camera_velocity: [f32; 2], // Current camera velocity for smooth interpolation
    inspector_zoom: f32, // Zoom level for inspector preview (1.0 = default)
    agent_trail_decay: f32, // Agent trail decay rate (0.0 = persistent, 1.0 = instant clear)

    // Spawn mode state
    spawn_mode_active: bool, // Toggle for click-to-spawn mode
    spawn_template_genome: Option<[u32; GENOME_WORDS]>, // Loaded template genome for spawning

    // RNG state for per-frame randomness
    rng_state: u64,

    // GUI state
    window: Arc<Window>,
    egui_renderer: egui_wgpu::Renderer,
    ui_tab: usize, // Control panel tab index
    ui_visible: bool, // Toggle control panel visibility with spacebar
    selected_fumarole_index: usize,
    // Debug
    debug_per_segment: bool,
    is_paused: bool,
    // Snapshot save state
    snapshot_save_requested: bool,
    snapshot_load_requested: bool,
    screenshot_4k_requested: bool,
    screenshot_requested: bool,
    // Screen recording state
    recording: bool,
    recording_fps: u32,
    recording_width: u32,
    recording_height: u32,
    recording_format: RecordingFormat,  // MP4 or GIF
    recording_show_ui: bool,
    recording_bar_visible: bool,
    recording_center_norm: [f32; 2],
    recording_output_path: Option<PathBuf>,
    recording_error: Option<String>,
    recording_start_time: Option<std::time::Instant>,
    recording_last_frame_time: Option<std::time::Instant>,
    recording_pipe: Option<RecordingPipe>,
    recording_readbacks: Vec<RecordingReadbackSlot>,
    recording_readback_index: usize,
    // Resolution change state
    pending_resolution_change: Option<u32>, // If Some(res), reset with new resolution
    // Visual buffer stride (pixels per row)
    visual_stride_pixels: u32,

    // Environment field controls
    alpha_blur: f32,
    beta_blur: f32,
    gamma_diffuse: f32,
    alpha_fluid_convolution: f32,
    beta_fluid_convolution: f32,
    gamma_blur: f32,
    gamma_shift: f32,
    alpha_slope_bias: f32,
    beta_slope_bias: f32,
    alpha_multiplier: f32,
    beta_multiplier: f32,
    dye_precipitation: f32,
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
    morphology_change_cost: f32,
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
    fluid_enabled: bool,
    fluid_show: bool,  // Show fluid simulation overlay
    fluid_dt: f32,
    fluid_decay: f32,
    fluid_jacobi_iters: u32,
    fluid_vorticity: f32,
    fluid_viscosity: f32,
    fluid_ooze_rate: f32,
    fluid_ooze_fade_rate: f32,
    fluid_ooze_rate_beta: f32,
    fluid_ooze_fade_rate_beta: f32,
    fluid_ooze_rate_gamma: f32,
    fluid_ooze_fade_rate_gamma: f32,
    fluid_ooze_still_rate: f32,
    fluid_dye_escape_rate: f32,
    fluid_dye_escape_rate_beta: f32,
    dye_diffusion: f32,
    dye_diffusion_no_fluid: f32,
    fluid_wind_push_strength: f32,
    fluid_slope_force_scale: f32,
    fluid_obstacle_strength: f32,

    // Fluid fumaroles (runtime-editable list; persisted via SimulationSettings).
    fumaroles: Vec<FumaroleSettings>,

    // Part base-angle overrides are stored above with fixed 128-slot capacity.
    vector_force_power: f32,
    vector_force_x: f32,
    vector_force_y: f32,
    prop_wash_strength: f32,
    prop_wash_strength_fluid: f32,

    // Propellers (organ-based propulsion) runtime toggle.
    propellers_enabled: bool,

    // Microswimming (morphology-based propulsion) runtime controls.
    microswim_enabled: bool,
    microswim_coupling: f32,
    microswim_base_drag: f32,
    microswim_anisotropy: f32,
    microswim_max_frame_vel: f32,
    microswim_torque_strength: f32,
    microswim_min_seg_displacement: f32,
    microswim_min_total_deformation_sq: f32,
    microswim_min_length_ratio: f32,
    microswim_max_length_ratio: f32,

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
    trail_show_energy: bool,
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
    dye_alpha_color: [f32; 3],
    dye_beta_color: [f32; 3],
    dye_alpha_thinfilm: bool,
    dye_alpha_thinfilm_mult: f32,
    dye_beta_thinfilm: bool,
    dye_beta_thinfilm_mult: f32,
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

const FULL_SPEED_PRESENT_INTERVAL_MICROS: u64 = 16_667; // ~60 Hz
const EGUI_UPDATE_INTERVAL_MICROS: u64 = 33_333; // ~30 Hz

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecordingFormat {
    MP4,
    GIF,
}

struct RecordingPipe {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    output_path: PathBuf,
    fps: u32,
    out_width: u32,
    out_height: u32,
    in_pix_fmt: &'static str,
}

struct RecordingReadbackSlot {
    buffer: wgpu::Buffer,
    padded_bytes_per_row: u32,
    width: u32,
    height: u32,
    pending_copy: bool,
    rx: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    scratch: Vec<u8>,
}

#[allow(dead_code)]
impl GpuState {
    fn write_rain_map_texture(&self) {
        // Expand interleaved alpha/beta into RGBA32F (alpha->R, beta->G).
        // This is only called on map load/clear, so a temporary allocation is fine.
        let cell_count = self.env_grid_cell_count as usize;
        let mut rgba: Vec<f32> = Vec::with_capacity(cell_count * 4);
        for i in 0..cell_count {
            let a = self.rain_map_data[i * 2];
            let b = self.rain_map_data[i * 2 + 1];
            rgba.extend_from_slice(&[a, b, 0.0, 0.0]);
        }

        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.rain_map_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&rgba),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(self.env_grid_resolution * 16),
                rows_per_image: Some(self.env_grid_resolution),
            },
            wgpu::Extent3d {
                width: self.env_grid_resolution,
                height: self.env_grid_resolution,
                depth_or_array_layers: 1,
            },
        );
    }
    fn save_settings(&self, path: &Path) -> anyhow::Result<()> {
        let settings = self.current_settings();
        settings.save_to_disk(path)
    }

    fn apply_settings(&mut self, settings: &SimulationSettings) {
        let mut settings = settings.clone();
        settings.sanitize();

        self.camera_zoom = settings.camera_zoom;
        self.spawn_probability = settings.spawn_probability;
        self.death_probability = settings.death_probability;
        self.mutation_rate = settings.mutation_rate;
        self.auto_replenish = settings.auto_replenish;
        self.diffusion_interval = settings.diffusion_interval;
        self.slope_interval = settings.slope_interval;

        self.alpha_blur = settings.alpha_blur;
        self.beta_blur = settings.beta_blur;
        self.gamma_diffuse = settings.gamma_diffuse;
        self.alpha_fluid_convolution = settings.alpha_fluid_convolution;
        self.beta_fluid_convolution = settings.beta_fluid_convolution;
        self.fluid_slope_force_scale = settings.fluid_slope_force_scale;
        self.fluid_obstacle_strength = settings.fluid_obstacle_strength;
        self.gamma_blur = settings.gamma_blur;
        self.gamma_shift = settings.gamma_shift;
        self.alpha_slope_bias = settings.alpha_slope_bias;
        self.beta_slope_bias = settings.beta_slope_bias;
        self.alpha_multiplier = settings.alpha_multiplier;
        self.beta_multiplier = settings.beta_multiplier;
        self.dye_precipitation = settings.dye_precipitation;
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
        self.morphology_change_cost = settings.morphology_change_cost;
        self.pairing_cost = settings.pairing_cost;
        self.prop_wash_strength = settings.prop_wash_strength;
        self.prop_wash_strength_fluid = settings.prop_wash_strength_fluid;

        self.propellers_enabled = settings.propellers_enabled;

        self.microswim_enabled = settings.microswim_enabled;
        self.microswim_coupling = settings.microswim_coupling;
        self.microswim_base_drag = settings.microswim_base_drag;
        self.microswim_anisotropy = settings.microswim_anisotropy;
        self.microswim_max_frame_vel = settings.microswim_max_frame_vel;
        self.microswim_torque_strength = settings.microswim_torque_strength;
        self.microswim_min_seg_displacement = settings.microswim_min_seg_displacement;
        self.microswim_min_total_deformation_sq = settings.microswim_min_total_deformation_sq;
        self.microswim_min_length_ratio = settings.microswim_min_length_ratio;
        self.microswim_max_length_ratio = settings.microswim_max_length_ratio;
        self.repulsion_strength = settings.repulsion_strength;
        self.agent_repulsion_strength = settings.agent_repulsion_strength;

        self.limit_fps = settings.limit_fps;
        self.limit_fps_25 = settings.limit_fps_25;
        self.render_interval = settings.render_interval;

        self.gamma_debug_visual = settings.gamma_debug_visual;
        self.slope_debug_visual = settings.slope_debug_visual;
        self.rain_debug_visual = settings.rain_debug_visual;

        self.fluid_enabled = settings.fluid_enabled;
        self.fluid_show = settings.fluid_show;
        self.fluid_dt = settings.fluid_dt;
        self.fluid_decay = settings.fluid_decay;
        self.fluid_jacobi_iters = settings.fluid_jacobi_iters;
        self.fluid_vorticity = settings.fluid_vorticity;
        self.fluid_viscosity = settings.fluid_viscosity;
        self.fluid_ooze_rate = settings.fluid_ooze_rate;
        self.fluid_ooze_fade_rate = settings.fluid_ooze_fade_rate;
        self.fluid_ooze_rate_beta = settings.fluid_ooze_rate_beta;
        self.fluid_ooze_fade_rate_beta = settings.fluid_ooze_fade_rate_beta;
        self.fluid_ooze_rate_gamma = settings.fluid_ooze_rate_gamma;
        self.fluid_ooze_fade_rate_gamma = settings.fluid_ooze_fade_rate_gamma;
        self.fluid_ooze_still_rate = settings.fluid_ooze_still_rate;
        self.fluid_dye_escape_rate = settings.fluid_dye_escape_rate;
        self.fluid_dye_escape_rate_beta = settings.fluid_dye_escape_rate_beta;
        self.dye_diffusion = settings.dye_diffusion;
        self.dye_diffusion_no_fluid = settings.dye_diffusion_no_fluid;
        self.fluid_wind_push_strength = settings.fluid_wind_push_strength;

        self.fumaroles = settings.fumaroles;
        if !self.fumaroles.is_empty() {
            self.selected_fumarole_index = self
                .selected_fumarole_index
                .min(self.fumaroles.len().saturating_sub(1));
        } else {
            self.selected_fumarole_index = 0;
        }

        self.vector_force_power = settings.vector_force_power;
        self.vector_force_x = settings.vector_force_x;
        self.vector_force_y = settings.vector_force_y;

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
        self.trail_show_energy = settings.trail_show_energy;

        self.interior_isotropic = settings.interior_isotropic;
        self.ignore_stop_codons = settings.ignore_stop_codons;
        self.require_start_codon = settings.require_start_codon;
        self.asexual_reproduction = settings.asexual_reproduction;

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
        self.dye_alpha_color = settings.dye_alpha_color;
        self.dye_beta_color = settings.dye_beta_color;
        self.dye_alpha_thinfilm = settings.dye_alpha_thinfilm;
        self.dye_alpha_thinfilm_mult = settings.dye_alpha_thinfilm_mult;
        self.dye_beta_thinfilm = settings.dye_beta_thinfilm;
        self.dye_beta_thinfilm_mult = settings.dye_beta_thinfilm_mult;
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

        // Apply FPS cap changes immediately.
        self.update_present_mode();

        if let Some(path) = &self.alpha_rain_map_path.clone() {
            let _ = self.load_alpha_rain_map(path);
        }
        if let Some(path) = &self.beta_rain_map_path.clone() {
            let _ = self.load_beta_rain_map(path);
        }
    }

    fn load_settings(&mut self, path: &Path) -> anyhow::Result<()> {
        let settings = SimulationSettings::load_from_disk(path)?;
           self.apply_settings(&settings);

        Ok(())
    }

    fn generate_map(&mut self, mode: u32, gen_type: u32, value: f32, seed: u32) {
        let mut part_props_override_head = [[f32::NAN; 4]; PART_PROPS_OVERRIDE_VEC4S_HEAD];
        let mut part_props_override_tail = [[f32::NAN; 4]; PART_PROPS_OVERRIDE_VEC4S_TAIL];
        part_props_override_head[..].copy_from_slice(&self.part_props_override[..PART_PROPS_OVERRIDE_VEC4S_HEAD]);
        part_props_override_tail[..].copy_from_slice(&self.part_props_override[PART_PROPS_OVERRIDE_VEC4S_HEAD..]);

        let params = EnvironmentInitParams {
            grid_resolution: self.env_grid_resolution,
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
            part_angle_override: pack_part_base_angle_overrides_vec4(&self.part_base_angle_overrides),
            part_props_override_head,
            part_props_override_tail,
            part_flags_override: pack_part_flags_override_vec4(&self.part_flags_override),
        };

        self.environment_init_cpu = params;

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

            let workgroups = (self.env_grid_resolution + 15) / 16;
            pass.dispatch_workgroups(workgroups, workgroups, 1);
        }

        if self.gpu_sync_profile {
            let gpu_wait_start = std::time::Instant::now();
            self.queue.submit(Some(encoder.finish()));
            // Wait for all submitted GPU work to complete; this makes the wall time a rough
            // proxy for GPU cost of this update submission (stable even when timestamp queries crash).
            self.device.poll(wgpu::Maintain::Wait);

            let gpu_update_ms = gpu_wait_start.elapsed().as_secs_f64() * 1000.0;
            self.gpu_sync_profile_accum_update_ms += gpu_update_ms;
            self.gpu_sync_profile_frame_counter += 1;

            if self.gpu_sync_profile_frame_counter >= self.gpu_sync_profile_interval_frames {
                let n = self.gpu_sync_profile_frame_counter.max(1) as f64;
                let avg_update = self.gpu_sync_profile_accum_update_ms / n;
                println!(
                    "[gpu-sync] update(ms): avg={:.3} n={} skip(trail_prep={}, nofluid_dye={}, nofluid_trail={})",
                    avg_update,
                    self.gpu_sync_profile_frame_counter,
                    self.perf_skip_trail_prep,
                    self.perf_skip_nofluid_dye,
                    self.perf_skip_nofluid_trail,
                );
                self.gpu_sync_profile_frame_counter = 0;
                self.gpu_sync_profile_accum_update_ms = 0.0;
            }
        } else {
            self.queue.submit(Some(encoder.finish()));
        }
    }

    async fn new(window: Arc<Window>) -> Self {
        let _size = window.inner_size();

        // Create instance and adapter
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: if cfg!(target_os = "windows") {
                wgpu::Backends::VULKAN // Use Vulkan on Windows to support atomicCompareExchangeWeak
            } else {
                wgpu::Backends::PRIMARY
            },
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

        let required_features = select_required_features(adapter.features());
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("GPU Device"),
                    required_features,
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

        Self::new_with_resources(
            window,
            instance,
            surface,
            adapter,
            device,
            queue,
            DEFAULT_ENV_GRID_RESOLUTION,
            DEFAULT_FLUID_GRID_RESOLUTION,
            DEFAULT_SPATIAL_GRID_RESOLUTION,
        )
        .await
    }

    async fn new_from_settings(
        window: Arc<Window>,
        env_grid_res: u32,
        fluid_grid_res: u32,
        spatial_grid_res: u32,
    ) -> Self {
        let _size = window.inner_size();

        // Create instance and adapter
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: if cfg!(target_os = "windows") {
                wgpu::Backends::VULKAN // Use Vulkan on Windows to support atomicCompareExchangeWeak
            } else {
                wgpu::Backends::PRIMARY
            },
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

        let required_features = select_required_features(adapter.features());
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("GPU Device"),
                    required_features,
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

        Self::new_with_resources(
            window,
            instance,
            surface,
            adapter,
            device,
            queue,
            env_grid_res,
            fluid_grid_res,
            spatial_grid_res,
        )
        .await
    }

    async fn new_with_resources(
        window: Arc<Window>,
        _instance: wgpu::Instance,
        surface: wgpu::Surface<'static>,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        env_grid_res: u32,
        fluid_grid_res: u32,
        spatial_grid_res: u32,
    ) -> Self {
        let size = window.inner_size();
        let mut profiler = StartupProfiler::new();
        profiler.mark("GpuState::new_with_resources begin");

        // Calculate derived values from resolutions
        let env_grid_cell_count = (env_grid_res * env_grid_res) as usize;
        let fluid_grid_cell_count = (fluid_grid_res * fluid_grid_res) as usize;
        let spatial_grid_cell_count = (spatial_grid_res * spatial_grid_res) as usize;

        debug_assert_eq!(std::mem::size_of::<BodyPart>(), 32);
        debug_assert_eq!(std::mem::align_of::<BodyPart>(), 16);
        debug_assert_eq!(
            std::mem::size_of::<Agent>(),
            2192,
            "Agent layout mismatch for MAX_BODY_PARTS={}",
            MAX_BODY_PARTS
        );
        debug_assert_eq!(std::mem::align_of::<Agent>(), 16);
        // NOTE: SpawnRequest includes a packed genome override:
        // - genome_override_len (u32)
        // - genome_override_offset (u32)
        // - genome_override_packed ([u32; GENOME_PACKED_WORDS])
        // Layout breakdown (std430 / repr(C, align(16))):
        // seed/genome_seed/flags/_pad_seed = 16 bytes total
        // position ([f32;2]) = 8  -> offset 16..24
        // energy (4) + rotation (4) = 8 -> offset 24..32
        // genome_override_len (4) + genome_override_offset (4) -> offset 32..40
        // genome_override_packed (GENOME_PACKED_WORDS * 4 = 64) -> offset 40..104
        // _pad_genome ([u32;2] = 8) -> offset 104..112
        // Total size = 112 bytes; alignment = 16 bytes.
        debug_assert_eq!(
            std::mem::size_of::<SpawnRequest>(),
            112,
            "SpawnRequest size mismatch; update buffer allocations/bindings if this fails"
        );
        debug_assert_eq!(std::mem::align_of::<SpawnRequest>(), 16);

        let window_clone = window.clone();
        profiler.mark("Resources received");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats[0];

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
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

        let frame_profiler = FrameProfiler::new(&device, &queue);

        // Initialize agents with minimal data - GPU will generate genome and build body.
        // NOTE: With wgpu::Limits::default(), max_storage_buffer_binding_size is typically 128 MiB.
        // Agent is currently 2192 bytes (see debug_assert above), so per-agent storage buffer capacity is:
        // floor(128 MiB / 2192) = 61_230 agents. Keep some headroom.
        // IMPORTANT: keep max_agents a multiple of 64 so compute dispatch can use `max_agents/64`
        // without ceil-dispatch and still cover the full buffer.
        // Scale agent capacity proportionally with env resolution for better performance:
        // 2048→60,032 agents, 1024→30,016 agents, 512→15,008 agents
        let max_agents = max_agents_for_env_res(env_grid_res);
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

        let run_seed = seed as u32;

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

        let grid_size = env_grid_cell_count;

        // Proper Perlin noise implementation (kept for potential later use)
        #[allow(dead_code)]
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
        #[allow(dead_code)]
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

        // Packed environment chemistry grid (vec4 per cell): x=alpha, y=beta, z=alpha_rain_map, w=beta_rain_map
        let chem_buffer_size = (grid_size * std::mem::size_of::<[f32; 4]>()) as u64;
        let chem_grid = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Chem Grid"),
            size: chem_buffer_size,
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

        // Rain map texture used by shaders (avoids extra storage buffer binding).
        let rain_map_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Rain Map Texture"),
            size: wgpu::Extent3d {
                width: env_grid_res,
                height: env_grid_res,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let rain_map_texture_view = rain_map_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Initialize chem_grid with zeros for alpha/beta and uniform 1.0 for rain maps
        // vec4: [alpha=0.0, beta=0.0, alpha_rain_map=1.0, beta_rain_map=1.0]
        let initial_chem_data: Vec<f32> = (0..grid_size)
            .flat_map(|_| [0.0f32, 0.0f32, 1.0f32, 1.0f32])
            .collect();
        queue.write_buffer(
            &chem_grid,
            0,
            bytemuck::cast_slice(&initial_chem_data),
        );

        // Initialize rain_map_buffer with uniform rain (1.0 for both alpha and beta) - kept for compatibility
        let uniform_rain: Vec<f32> = (0..grid_size).flat_map(|_| [1.0f32, 1.0f32]).collect();
        queue.write_buffer(
            &rain_map_buffer,
            0,
            bytemuck::cast_slice(&uniform_rain),
        );

        // Seed the texture with uniform rain too (RGBA32F: r=alpha, g=beta) - kept for compatibility.
        let uniform_rain_rgba: Vec<f32> = (0..grid_size)
            .flat_map(|_| [1.0f32, 1.0f32, 0.0f32, 0.0f32])
            .collect();
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &rain_map_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&uniform_rain_rgba),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(env_grid_res * 16),
                rows_per_image: Some(env_grid_res),
            },
            wgpu::Extent3d {
                width: env_grid_res,
                height: env_grid_res,
                depth_or_array_layers: 1,
            },
        );
        profiler.mark("Rain map buffer");

        // Agent spatial grid for neighbor detection.
        // Layout: 2x u32 per cell: [id, epoch_stamp].
        let agent_spatial_grid_size = (spatial_grid_cell_count * 2 * std::mem::size_of::<u32>()) as u64;
        let agent_spatial_grid_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Agent Spatial Grid"),
            size: agent_spatial_grid_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        profiler.mark("Agent spatial grid buffer");

        // IMPORTANT: wgpu buffers are not guaranteed to be zero-initialized.
        // Clear the combined spatial buffer so no cell accidentally matches current_stamp.
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Init Spatial Grid Clear Encoder"),
            });
            encoder.clear_buffer(&agent_spatial_grid_buffer, 0, None);
            queue.submit(Some(encoder.finish()));
        }

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

        let trail_grid_inject = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Trail Grid Inject"),
            size: trail_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let trail_debug_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Trail Grid Debug Readback"),
            size: trail_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        profiler.mark("Alpha/Beta/Gamma/Trail buffers");

        let env_seed = (seed ^ (seed >> 32)) as u32;
        let environment_init = EnvironmentInitParams {
            grid_resolution: env_grid_res,
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
            part_angle_override: pack_part_base_angle_overrides_vec4(&[f32::NAN; PART_OVERRIDE_SLOTS]),
            part_props_override_head: [[f32::NAN; 4]; PART_PROPS_OVERRIDE_VEC4S_HEAD],
            part_props_override_tail: [[f32::NAN; 4]; PART_PROPS_OVERRIDE_VEC4S_TAIL],
            part_flags_override: [[f32::NAN; 4]; PART_FLAGS_OVERRIDE_VEC4S],
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

        let sim_size = sim_size_for_env_res(env_grid_res);

        let params = SimParams {
            dt: 0.016,
            frame_dt: 0.016,
            drag: 0.1,
            energy_cost: 0.0, // Disabled energy depletion for now
            amino_maintenance_cost: 0.001,
            morphology_change_cost: 0.0,
            spawn_probability: 0.01,
            death_probability: 0.001,
            grid_size: sim_size,
            camera_zoom: 1.0,
            camera_pan_x: sim_size / 2.0,
            camera_pan_y: sim_size / 2.0,
            prev_camera_pan_x: sim_size / 2.0, // Initialize to same as camera_pan
            prev_camera_pan_y: sim_size / 2.0, // Initialize to same as camera_pan
            follow_mode: 0,
            window_width: surface_config.width as f32,
            window_height: surface_config.height as f32,
            alpha_blur: 0.05,
            beta_blur: 0.05,
            gamma_diffuse: 0.0,
            gamma_blur: 0.9995,
            gamma_shift: 0.0,
            alpha_slope_bias: -5.0,
            beta_slope_bias: 5.0,
            alpha_multiplier: 0.0001, // Rain probability: 0.01% per cell per frame
            beta_multiplier: 0.0,     // Poison rain disabled
            // NOTE: Repurposed padding: used by shaders as a boolean flag.
            // 0 = fluid simulation disabled (dye layer is visual/sensing only)
            // 1 = fluid simulation enabled
            // This is just the initial value; the per-frame uniform upload overrides it.
            _pad_rain0: 1,
            _pad_rain1: 0,
            rain_drop_count: 0,
            alpha_rain_drop_count: 0,
            dye_precipitation: 1.0,
            chemical_slope_scale_alpha: 0.1,
            chemical_slope_scale_beta: 0.1,
            mutation_rate: 0.005,
            food_power: 3.0,
            poison_power: 1.0,
            pairing_cost: 0.1,
            max_agents: max_agents as u32,
            cpu_spawn_count: 0,
            agent_count: initial_agents as u32,
            population_count: initial_agents as u32,
            random_seed: seed as u32,
            debug_mode: 0,
            visual_stride: visual_stride_pixels,
            selected_agent_index: u32::MAX,
            repulsion_strength: 10.0,
            agent_repulsion_strength: 1.0,
            gamma_strength: 10.0 * TERRAIN_FORCE_SCALE,
            prop_wash_strength: 1.0,
            prop_wash_strength_fluid: 1.0,
            gamma_vis_min: 0.0,
            gamma_vis_max: 50.0,
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
            fluid_wind_push_strength: 0.0005,
            alpha_fluid_convolution: 0.05,
            beta_fluid_convolution: 0.05,
            fluid_slope_force_scale: 100.0,
            fluid_obstacle_strength: 200.0,
            dye_alpha_color_r: 0.0,
            dye_alpha_color_g: 1.0,
            dye_alpha_color_b: 0.0,
            _pad_dye_alpha_color: 0.0,
            dye_beta_color_r: 1.0,
            dye_beta_color_g: 0.0,
            dye_beta_color_b: 0.0,
            _pad_dye_beta_color: 0.0,
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sim Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Microswim params buffer must exist before settings load (bind groups reference it),
        // so initialize with defaults and overwrite after `SimulationSettings` is loaded.
        let microswim_params_f32_init: [f32; MICROSWIM_PARAM_FLOATS] = [
            // vec4 0
            if default_microswim_enabled() { 1.0 } else { 0.0 },
            default_microswim_coupling(),
            default_microswim_base_drag(),
            default_microswim_anisotropy(),
            // vec4 1
            default_microswim_max_frame_vel(),
            default_microswim_torque_strength(),
            default_microswim_min_seg_displacement(),
            default_microswim_min_total_deformation_sq(),
            // vec4 2
            default_microswim_min_length_ratio(),
            default_microswim_max_length_ratio(),
            0.0,
            0.0,
            // vec4 3 (reserved)
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        let microswim_params_bytes = pack_f32_uniform(&microswim_params_f32_init);
        let microswim_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Microswim Params"),
            contents: &microswim_params_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let spawn_debug_counters = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spawn/Debug Counters"),
            size: 12, // 3 x u32 ([0]=spawn, [1]=debug, [2]=alive)
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // [0]=spawn, [1]=debug, [2]=alive.
        queue.write_buffer(&spawn_debug_counters, 0, bytemuck::cast_slice(&[0u32, 0u32, 0u32]));

        // Indirect dispatch args for init-dead: [x, y, z]. Written by a tiny compute kernel.
        let init_dead_dispatch_args = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("InitDead Dispatch Args"),
            size: 12,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Default to a no-op dispatch (0 groups).
        queue.write_buffer(&init_dead_dispatch_args, 0, bytemuck::cast_slice(&[0u32, 1u32, 1u32]));

        // 16-byte uniform for the init-dead-dispatch writer shader: [max_agents, 0, 0, 0].
        let init_dead_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("InitDead Params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &init_dead_params_buffer,
            0,
            bytemuck::cast_slice(&[max_agents as u32, 0u32, 0u32, 0u32]),
        );

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

        let alive_readback_pending: [Arc<Mutex<Option<Result<(u32, u32), ()>>>>; 2] =
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

        let selected_agent_readbacks: Vec<Arc<wgpu::Buffer>> = (0..SELECTED_AGENT_READBACK_SLOTS)
            .map(|i| {
                Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Selected Agent Readback {i}")),
                    size: std::mem::size_of::<Agent>() as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }))
            })
            .collect();

        let selected_agent_readback_pending: Vec<Arc<Mutex<Option<Result<Agent, ()>>>>> =
            (0..SELECTED_AGENT_READBACK_SLOTS)
                .map(|_| Arc::new(Mutex::new(None)))
                .collect();

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
        let chem_grid_size_bytes = (env_grid_cell_count * std::mem::size_of::<[f32; 4]>()) as u64;

        let chem_grid_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Chem Grid Readback"),
            size: chem_grid_size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let gamma_grid_size_bytes = (env_grid_cell_count * std::mem::size_of::<f32>()) as u64;
        let gamma_grid_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gamma Grid Readback"),
            size: gamma_grid_size_bytes,
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
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
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

        // Load main shader (concatenate shared + render + simulation modules, composite is separate)
        // Inject compile-time constants from runtime resolution settings.
        let shader_source = format!(
            "const SIM_SIZE: u32 = {}u;\nconst ENV_GRID_SIZE: u32 = {}u;\nconst GRID_SIZE: u32 = {}u;\nconst SPATIAL_GRID_SIZE: u32 = {}u;\nconst FLUID_GRID_SIZE: u32 = {}u;\n{}\n{}\n{}\n{}\n{}\n{}",
            sim_size.round().max(1.0) as u32,
            env_grid_res,
            env_grid_res,
            spatial_grid_res,
            fluid_grid_res,
            include_str!("../shaders/shared.wgsl"),
            include_str!("../shaders/render.wgsl"),
            include_str!("../shaders/simulation.wgsl"),
            include_str!("../shaders/rain.wgsl"),
            include_str!("../shaders/microswim.wgsl"),
            include_str!("../shaders/reproduction.wgsl")
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        profiler.mark("Main shader compiled");

        // Load compact_merge shader (dedicated compact and merge operations)
        // IMPORTANT: It must share the exact Agent/SimParams/bindings from shared.wgsl,
        // otherwise compaction will misread `alive` and effectively "drop" prior batches.
        let compact_merge_shader_source = format!(
            "const FLUID_GRID_SIZE: u32 = {}u;\n{}\n{}",
            fluid_grid_res,
            include_str!("../shaders/shared_types_only.wgsl"),
            include_str!("../shaders/compact_merge_minimal.wgsl")
        );
        let compact_merge_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compact Merge Shader"),
            source: wgpu::ShaderSource::Wgsl(compact_merge_shader_source.into()),
        });
        profiler.mark("Compact/Merge shader compiled");

        // Load composite shader (standalone, minimal dependencies)
        let composite_shader_source = format!(
            "const FLUID_GRID_SIZE: u32 = {}u;\nconst GAMMA_GRID_DIM: u32 = {}u;\n{}",
            fluid_grid_res,
            env_grid_res,
            include_str!("../shaders/composite.wgsl")
        );
        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Composite Shader"),
            source: wgpu::ShaderSource::Wgsl(composite_shader_source.into()),
        });
        profiler.mark("Composite shader compiled");

        // Minimal shader: writes indirect dispatch args for init-dead.
        let init_dead_dispatch_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("InitDead Dispatch Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/init_dead_dispatch.wgsl").into()),
        });
        profiler.mark("initDead dispatch shader compiled");

        // ============================================================================
        // FLUID SIMULATION BUFFERS (created early so they can be used in bind groups)
        // ============================================================================
        let fluid_grid_cells: usize = (fluid_grid_res * fluid_grid_res) as usize;
        let env_grid_cells: usize = (env_grid_res * env_grid_res) as usize;

        let fluid_velocity_size = (fluid_grid_cells * std::mem::size_of::<[f32; 2]>()) as u64;
        let fluid_scalar_size = (fluid_grid_cells * std::mem::size_of::<f32>()) as u64;
        // Dye is stored at environment-grid resolution (GAMMA_GRID_DIM x GAMMA_GRID_DIM).
        // This replaces the previous fluid-resolution dye buffers without adding any new bindings.
        // Stored as vec4<f32> per cell to carry beta/alpha/gamma with 16-byte stride.
        let fluid_dye_size = (env_grid_cells * std::mem::size_of::<[f32; 4]>()) as u64;

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
        let force_vectors_zeros = vec![[0.0f32, 0.0f32]; fluid_grid_cells];
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

        let fluid_dye_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Dye A"),
            size: fluid_dye_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let fluid_dye_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fluid Dye B"),
            size: fluid_dye_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        profiler.mark("Fluid buffers");

        // Composite bind group layout (minimal bindings for compositing - created early before compute layout)
        let composite_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Composite Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let composite_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Composite Pipeline Layout"),
            bind_group_layouts: &[&composite_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create bind group layouts
        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Compute Bind Group Layout"),
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 18,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 19,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        profiler.mark("Compute bind layout");

        let init_dead_writer_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("InitDead Writer Bind Group Layout"),
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
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let init_dead_writer_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("InitDead Writer Bind Group"),
            layout: &init_dead_writer_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: spawn_debug_counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: init_dead_dispatch_args.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: init_dead_params_buffer.as_entire_binding(),
                },
            ],
        });
        profiler.mark("initDead writer bind group");

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
                    resource: chem_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: fluid_dye_a.as_entire_binding(),
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
                    resource: trail_grid_inject.as_entire_binding(),
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
                    resource: fluid_force_vectors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: agent_spatial_grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: wgpu::BindingResource::TextureView(&rain_map_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: microswim_params_buffer.as_entire_binding(),
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
                    resource: chem_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: fluid_dye_a.as_entire_binding(),
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
                    resource: trail_grid_inject.as_entire_binding(),
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
                    resource: fluid_force_vectors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: agent_spatial_grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: wgpu::BindingResource::TextureView(&rain_map_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: microswim_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create reproduction bind groups - binding 1 is INPUT buffer (read-write)
        // Reproduction reads/writes agents_in to update pairing_counter and energy,
        // then process_agents reads the updated values.
        // Binding 0 uses the OUTPUT buffer as a dummy since reproduction doesn't reference agents_in.
        let reproduction_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Reproduction Bind Group A"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: agents_buffer_b.as_entire_binding(),  // Dummy (reproduction doesn't use binding 0)
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: agents_buffer_a.as_entire_binding(),  // INPUT buffer (read-write)
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: chem_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: fluid_dye_a.as_entire_binding(),
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
                    resource: trail_grid_inject.as_entire_binding(),
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
                    resource: fluid_force_vectors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: agent_spatial_grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: wgpu::BindingResource::TextureView(&rain_map_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: microswim_params_buffer.as_entire_binding(),
                },
            ],
        });

        let reproduction_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Reproduction Bind Group B"),
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: agents_buffer_a.as_entire_binding(),  // Dummy (reproduction doesn't use binding 0)
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: agents_buffer_b.as_entire_binding(),  // INPUT buffer (read-write)
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: chem_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: fluid_dye_a.as_entire_binding(),
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
                    resource: trail_grid_inject.as_entire_binding(),
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
                    resource: fluid_force_vectors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: agent_spatial_grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: wgpu::BindingResource::TextureView(&rain_map_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: microswim_params_buffer.as_entire_binding(),
                },
            ],
        });

        profiler.mark("Compute bind groups");

        // Create composite bind group (uses separate bind group from compute)
        let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Composite Bind Group"),
            layout: &composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: visual_grid_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: agent_grid_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: fluid_dye_a.as_entire_binding() },
            ],
        });
        profiler.mark("Composite bind group");

        // Create compute pipelines
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });
        profiler.mark("Compute pipeline layout");

        let init_dead_writer_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("InitDead Writer Pipeline Layout"),
                bind_group_layouts: &[&init_dead_writer_bind_group_layout],
                push_constant_ranges: &[],
            });
        profiler.mark("initDead writer pipeline layout");

        let reproduction_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Reproduction Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "reproduce_agents",
            compilation_options: Default::default(),
            cache: None,
        });

        let process_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Process Agents Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "process_agents",
            compilation_options: Default::default(),
            cache: None,
        });
        profiler.mark("process_agents pipeline");

        let microswim_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Microswim Agents Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "microswim_agents",
            compilation_options: Default::default(),
            cache: None,
        });
        profiler.mark("microswim_agents pipeline");

        let diffuse_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Diffuse Pipeline (stage1)"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "diffuse_grids_stage1",
            compilation_options: Default::default(),
            cache: None,
        });
        profiler.mark("diffuse pipeline stage1");

        let diffuse_commit_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Diffuse Pipeline (stage2 commit)"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "diffuse_grids_stage2",
            compilation_options: Default::default(),
            cache: None,
        });
        profiler.mark("diffuse pipeline stage2");

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

        let rain_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Rain Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "apply_rain_drops",
                compilation_options: Default::default(),
                cache: None,
            });
        profiler.mark("rain pipeline");

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

        let clear_inspector_preview_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Clear Inspector Preview Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "clear_inspector_preview",
                compilation_options: Default::default(),
                cache: None,
            });
        profiler.mark("clear inspector preview pipeline");

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
                layout: Some(&composite_pipeline_layout),
                module: &composite_shader,
                entry_point: "composite_agents",
                compilation_options: Default::default(),
                cache: None,
            });
        profiler.mark("composite agents pipeline");

        let merge_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Merge Agents Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compact_merge_shader,
            entry_point: "merge_agents_cooperative",
            compilation_options: Default::default(),
            cache: None,
        });
        profiler.mark("merge pipeline");

        let compact_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compact Agents Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compact_merge_shader,
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

        let write_init_dead_dispatch_args_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Write InitDead Dispatch Args Pipeline"),
                layout: Some(&init_dead_writer_pipeline_layout),
                module: &init_dead_dispatch_shader,
                entry_point: "write_init_dead_dispatch_args",
                compilation_options: Default::default(),
                cache: None,
            });
        profiler.mark("write initDead dispatch args pipeline");

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

        let spike_kill_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Spike Kill Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &shader,
                entry_point: "spike_kill",
                compilation_options: Default::default(),
                cache: None,
            });
        profiler.mark("spike kill pipeline");

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

        // Inspector overlay render pipeline (fragment overlay for the inspector panel bars/labels)
        let inspector_overlay_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Inspector Overlay Bind Group Layout"),
                entries: &[
                    // `params` (shared.wgsl: @group(0) @binding(6))
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // `selected_agent_buffer` (shared.wgsl: @group(0) @binding(12))
                    wgpu::BindGroupLayoutEntry {
                        binding: 12,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            // WGSL declares `selected_agent_buffer` as read_write for the compute path.
                            // The fragment overlay only reads it, but the pipeline layout must still match.
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            },
        );

        let inspector_overlay_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Inspector Overlay Bind Group"),
            layout: &inspector_overlay_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: selected_agent_buffer.as_entire_binding(),
                },
            ],
        });

        let inspector_overlay_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Inspector Overlay Pipeline Layout"),
                bind_group_layouts: &[&inspector_overlay_bind_group_layout],
                push_constant_ranges: &[],
            });

        let inspector_overlay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Inspector Overlay Pipeline"),
            layout: Some(&inspector_overlay_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_inspector_overlay",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_inspector_overlay",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
        profiler.mark("inspector overlay pipeline");

        // ============================================================================
        // FLUID SIMULATION PIPELINES (buffers already created above, before bind groups)
        // ============================================================================

        // Fluid params buffer (flat f32 array, packed as vec4<f32> array in WGSL).
        // Indices must match shaders/fluid.wgsl FP_* constants.
        let fluid_params_f32: [f32; 32] = [
            0.0,                       // time
            0.016,                     // dt
            0.995,                     // decay
            fluid_grid_res as f32,     // grid_size (as f32)
            0.0, 0.0, 0.0, 0.0,        // mouse
            0.0, 0.0, 0.0, 0.0,        // splat
            100.0,                     // fluid_slope_force_scale
            200.0,                     // fluid_obstacle_strength
            0.0,                       // vector_force_x
            0.0,                       // vector_force_y
            0.0,                       // vector_force_power
            0.0,                       // chem_ooze_still_rate
            0.0,                       // sedimentation_min_speed
            0.0,                       // sedimentation_multiplier
            0.0,                       // dye_escape_rate_alpha
            0.0,                       // dye_escape_rate_beta
            1.0,                       // dye_deposit_scale (dye -> chem)
            0.0,                       // (unused / legacy)
            0.0,                       // (unused / legacy)
            0.01,                      // dye_diffusion (fluid on)
            0.15,                      // dye_diffusion_no_fluid (fluid off)
            100.0, 0.0, 0.0, 0.0, 0.0, // reserved (slope_steer_rate, _, _, _, _)
        ];
        let fluid_params_bytes = pack_f32_uniform(&fluid_params_f32);

        let fluid_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fluid Params"),
            contents: &fluid_params_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let fluid_fumaroles_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Fluid Fumaroles"),
            contents: &vec![0u8; FUMAROLE_BUFFER_BYTES],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Load fluid shader (standalone) and inject compile-time constants.
        let fluid_shader_source_raw = std::fs::read_to_string("shaders/fluid.wgsl")
            .expect("Failed to load shaders/fluid.wgsl");
        let fluid_shader_source = format!(
            "const FLUID_GRID_SIZE: u32 = {}u;\nconst GAMMA_GRID_DIM: u32 = {}u;\n{}",
            fluid_grid_res,
            env_grid_res,
            fluid_shader_source_raw
        );
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
                // Gamma grid (terrain). Made read-write so fluid can do gamma lift/deposition.
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
                // Packed environment chemistry grid (read-write):
                // - read for dye injection
                // - written by advect_dye() for fluid-driven erosion/deposition
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Fluid-only: packed fumarole list (see shaders/fluid.wgsl).
                wgpu::BindGroupLayoutEntry {
                    binding: 17,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                wgpu::BindGroupEntry { binding: 2, resource: gamma_grid.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: chem_grid.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: fluid_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: fluid_pressure_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: fluid_pressure_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: fluid_divergence.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: fluid_forces.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 16, resource: fluid_force_vectors.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: fluid_dye_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: fluid_dye_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: trail_grid.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 13, resource: trail_grid_inject.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 17, resource: fluid_fumaroles_buffer.as_entire_binding() },
            ],
        });

        let fluid_bind_group_ba = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fluid Bind Group BA"),
            layout: &fluid_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: fluid_velocity_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: fluid_velocity_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: gamma_grid.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: chem_grid.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: fluid_params_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: fluid_pressure_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: fluid_pressure_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: fluid_divergence.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: fluid_forces.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 16, resource: fluid_force_vectors.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: fluid_dye_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: fluid_dye_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: trail_grid.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 13, resource: trail_grid_inject.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 17, resource: fluid_fumaroles_buffer.as_entire_binding() },
            ],
        });

        // Fluid pipeline layout
        let fluid_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fluid Pipeline Layout"),
            bind_group_layouts: &[&fluid_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create fluid compute pipelines
        // Test force injection for debugging
        let fluid_generate_forces_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Inject Test Force"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "inject_test_force",
            compilation_options: Default::default(),
            cache: None,
        });

        let fluid_fumarole_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Inject Fumarole Force Vector"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "inject_fumarole_force_vector",
            compilation_options: Default::default(),
            cache: None,
        });

        let fluid_fumarole_dye_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Inject Fumarole Dye"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "inject_fumarole_dye",
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

        let fluid_clear_forces_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Clear Forces"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "clear_forces",
            compilation_options: Default::default(),
            cache: None,
        });

        // NOTE: This pipeline is unused but kept for struct compatibility
        let fluid_clear_force_vectors_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Clear Force Vectors (unused)"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "clear_forces",  // Use valid entry point since this is never dispatched
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

        let clear_fluid_force_vectors_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Clear Fluid Force Vectors"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "clear_fluid_force_vectors",
            compilation_options: Default::default(),
            cache: None,
        });

        let fluid_inject_dye_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Inject Dye"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "inject_dye",
            compilation_options: Default::default(),
            cache: None,
        });

        let fluid_advect_dye_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Advect Dye"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "advect_dye",
            compilation_options: Default::default(),
            cache: None,
        });

        let fluid_advect_trail_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Advect Trail"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "advect_trail",
            compilation_options: Default::default(),
            cache: None,
        });

        let fluid_diffuse_dye_no_fluid_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Diffuse Dye (No Fluid)"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "diffuse_dye_no_fluid",
            compilation_options: Default::default(),
            cache: None,
        });

        let fluid_copy_trail_no_fluid_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Copy Trail (No Fluid)"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "copy_trail_no_fluid",
            compilation_options: Default::default(),
            cache: None,
        });

        let fluid_clear_dye_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fluid Clear Dye"),
            layout: Some(&fluid_pipeline_layout),
            module: &fluid_shader,
            entry_point: "clear_dye",
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
            &spawn_debug_counters,
            &selected_agent_buffer,
            &visual_grid_buffer,
        ] {
            init_encoder.clear_buffer(buffer, 0, None);
        }

        let env_groups_x = (env_grid_res + CLEAR_WG_SIZE_X - 1) / CLEAR_WG_SIZE_X;
        let env_groups_y = (env_grid_res + CLEAR_WG_SIZE_Y - 1) / CLEAR_WG_SIZE_Y;
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
        let fluid_workgroups = (fluid_grid_res + 15) / 16;
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

        // Now that settings are loaded, push microswim params into the already-created buffer.
        let microswim_params_f32: [f32; MICROSWIM_PARAM_FLOATS] = [
            // vec4 0
            if settings.microswim_enabled { 1.0 } else { 0.0 },
            settings.microswim_coupling,
            settings.microswim_base_drag,
            settings.microswim_anisotropy,
            // vec4 1
            settings.microswim_max_frame_vel,
            settings.microswim_torque_strength,
            settings.microswim_min_seg_displacement,
            settings.microswim_min_total_deformation_sq,
            // vec4 2
            settings.microswim_min_length_ratio,
            settings.microswim_max_length_ratio,
            0.0,
            0.0,
            // vec4 3 (reserved)
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        let microswim_params_bytes = pack_f32_uniform(&microswim_params_f32);
        queue.write_buffer(&microswim_params_buffer, 0, &microswim_params_bytes);

        // Safe profiling/toggling for A/B cost measurements.
        // These do NOT use GPU timestamp queries (which can be unstable on some drivers).
        let gpu_sync_profile = std::env::var("ALSIM_GPU_SYNC_PROFILE")
            .map(|value| value != "0")
            .unwrap_or(false);
        let gpu_sync_profile_interval_frames = std::env::var("ALSIM_GPU_SYNC_PROFILE_INTERVAL_FRAMES")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(60)
            .max(1);
        let perf_skip_trail_prep = std::env::var("ALSIM_SKIP_TRAIL_PREP")
            .map(|value| value != "0")
            .unwrap_or(false);
        let perf_skip_nofluid_dye = std::env::var("ALSIM_SKIP_NOFLUID_DYE")
            .map(|value| value != "0")
            .unwrap_or(false);
        let perf_skip_nofluid_trail = std::env::var("ALSIM_SKIP_NOFLUID_TRAIL")
            .map(|value| value != "0")
            .unwrap_or(false);

        let mut state = Self {
            device,
            queue,
            surface,
            surface_config,

            // Resolution settings
            env_grid_resolution: env_grid_res,
            fluid_grid_resolution: fluid_grid_res,
            spatial_grid_resolution: spatial_grid_res,
            env_grid_cell_count,
            fluid_grid_cell_count,
            spatial_grid_cell_count,

            sim_size,

            agents_buffer_a,
            agents_buffer_b,
            chem_grid,
            rain_map_buffer,
            rain_map_texture,
            rain_map_texture_view,
            agent_spatial_grid_buffer,
            gamma_grid,
            trail_grid,
            trail_grid_inject,
            trail_debug_readback,
            visual_grid_buffer,
            agent_grid_buffer,
            params_buffer,
            microswim_params_buffer,
            environment_init_params_buffer,
            spawn_debug_counters,
            init_dead_dispatch_args,
            init_dead_params_buffer,
            init_dead_writer_bind_group_layout,
            init_dead_writer_bind_group,
            alive_readbacks,
            alive_readback_pending,
            alive_readback_inflight: [false; 2],
            alive_readback_slot: 0,
            alive_readback_last_applied_epoch: 0,
            alive_readback_zero_streak: 0,
            debug_readback,
            agents_readback,
            selected_agent_buffer,
            selected_agent_readbacks,
            selected_agent_readback_pending,
            selected_agent_readback_inflight: vec![false; SELECTED_AGENT_READBACK_SLOTS],
            selected_agent_readback_slot: 0,
            // Allow immediate first readback.
            selected_agent_readback_last_request: std::time::Instant::now()
                - std::time::Duration::from_secs(1),
            new_agents_buffer,
            spawn_readback,
            spawn_requests_buffer,
            chem_grid_readback,
            gamma_grid_readback,
            visual_texture,
            visual_texture_view,
            sampler,
            process_pipeline,
            microswim_pipeline,
            reproduction_bind_group_a,
            reproduction_bind_group_b,
            reproduction_pipeline,
            clear_fluid_force_vectors_pipeline,
            diffuse_pipeline,
            diffuse_commit_pipeline,
            diffuse_trails_pipeline,
            rain_pipeline,
            clear_visual_pipeline,
            motion_blur_pipeline,
            clear_agent_grid_pipeline,
            clear_inspector_preview_pipeline,
            render_agents_pipeline,
            draw_inspector_agent_pipeline,
            composite_agents_pipeline,
            gamma_slope_pipeline,
            merge_pipeline,
            compact_pipeline,
            finalize_merge_pipeline,
            cpu_spawn_pipeline,
            write_init_dead_dispatch_args_pipeline,
            initialize_dead_pipeline,
            environment_init_pipeline,
            generate_map_pipeline,
            clear_agent_spatial_grid_pipeline,
            populate_agent_spatial_grid_pipeline,
            drain_energy_pipeline,
            spike_kill_pipeline,
            render_pipeline,
            inspector_overlay_pipeline,
            compute_bind_group_a,
            compute_bind_group_b,
            composite_bind_group,
            render_bind_group,
            inspector_overlay_bind_group,
            ping_pong: false,
            agent_count: initial_agents as u32,
            alive_count: initial_agents as u32,
            camera_zoom: settings.camera_zoom,
            camera_pan: [sim_size / 2.0, sim_size / 2.0],
            prev_camera_pan: [sim_size / 2.0, sim_size / 2.0], // Initialize to same as camera_pan
            agents_cpu: agents,
            agent_buffer_capacity: max_agents,
            cpu_spawn_queue: Vec::new(),
            spawn_request_count: 0,
            pending_spawn_upload: false,
            spawn_probability: settings.spawn_probability,
            death_probability: settings.death_probability,
            auto_replenish: settings.auto_replenish,
            rain_map_data: vec![1.0f32; env_grid_cell_count * 2], // Initialize with uniform rain
            difficulty: settings.difficulty.clone(),
            frame_count: 0,
            last_fps_update: std::time::Instant::now(),
            limit_fps: settings.limit_fps,
            limit_fps_25: settings.limit_fps_25,
            last_frame_time: std::time::Instant::now(),
            // Allow immediate first present.
            last_present_time: std::time::Instant::now() - std::time::Duration::from_secs(1),
            // Allow immediate first egui update.
            last_egui_update_time: std::time::Instant::now() - std::time::Duration::from_secs(1),
            cached_egui_primitives: Vec::new(),
            frame_profiler,

            gpu_sync_profile,
            gpu_sync_profile_interval_frames,
            gpu_sync_profile_frame_counter: 0,
            gpu_sync_profile_accum_update_ms: 0.0,
            gpu_sync_profile_accum_render_ms: 0.0,

            perf_skip_trail_prep,
            perf_skip_nofluid_dye,
            perf_skip_nofluid_trail,

            perf_skip_diffusion: false,
            perf_skip_rain: false,
            perf_skip_slope: false,
            perf_skip_draw: false,
            perf_skip_repro: false,
            perf_force_gpu_sync: false,

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
            trail_energy_debug_next_epoch: 0,
            epochs_per_second: 0.0,
            population_history: Vec::new(),
            population_plot_points: Vec::new(),
            alpha_rain_history: VecDeque::new(),
            beta_rain_history: VecDeque::new(),
            epoch_sample_interval: 1000,
            last_sample_epoch: 0,
            max_history_points: 5000,
            last_autosave_epoch: 0,

            organ45_alive_with: 0,
            organ45_total_with: 0,
            organ45_last_scan: None,
            organ45_last_scan_error: None,

            run_seed,
            run_name: naming::sim::generate_sim_name(&settings, run_seed, max_agents as u32),

            is_dragging: false,
            is_zoom_dragging: false,
            zoom_drag_start_y: 0.0,
            zoom_drag_start_zoom: 1.0,
            last_mouse_pos: None,
            selected_agent_index: None,
            selected_agent_data: None,
            follow_selected_agent: false, // Initialize to false
            camera_target: [sim_size / 2.0, sim_size / 2.0], // Initialize to center
            camera_velocity: [0.0, 0.0], // Initialize velocity to zero
            inspector_zoom: 1.0,
            agent_trail_decay: settings.agent_trail_decay,
            spawn_mode_active: false,
            spawn_template_genome: None,
            rng_state: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            window: window_clone,
            egui_renderer,
            ui_tab: 0, // Start on Agents tab
            ui_visible: true, // Control panel visible by default
            selected_fumarole_index: 0,
            debug_per_segment: settings.debug_per_segment,
            is_paused: false,
            snapshot_save_requested: false,
            snapshot_load_requested: false,
            screenshot_4k_requested: false,
            screenshot_requested: false,
            recording: false,
            recording_fps: 30,
            recording_width: 720,
            recording_height: 720,
            recording_format: RecordingFormat::MP4,
            recording_show_ui: false,
            recording_bar_visible: false,
            recording_center_norm: [0.5, 0.5],
            recording_output_path: None,
            recording_error: None,
            recording_start_time: None,
            recording_last_frame_time: None,
            recording_pipe: None,
            recording_readbacks: Vec::new(),
            recording_readback_index: 0,
            pending_resolution_change: None,
            visual_stride_pixels,
            alpha_blur: settings.alpha_blur,
            beta_blur: settings.beta_blur,
            gamma_diffuse: settings.gamma_diffuse,
            alpha_fluid_convolution: settings.alpha_fluid_convolution,
            beta_fluid_convolution: settings.beta_fluid_convolution,
            gamma_blur: settings.gamma_blur,
            gamma_shift: settings.gamma_shift,
            alpha_slope_bias: settings.alpha_slope_bias,
            beta_slope_bias: settings.beta_slope_bias,
            alpha_multiplier: settings.alpha_multiplier,
            beta_multiplier: settings.beta_multiplier,
            dye_precipitation: settings.dye_precipitation,
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
            morphology_change_cost: settings.morphology_change_cost,
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
            fluid_enabled: settings.fluid_enabled,
            fluid_show: settings.fluid_show,
            fluid_dt: settings.fluid_dt,
            fluid_decay: settings.fluid_decay,
            fluid_jacobi_iters: settings.fluid_jacobi_iters,
            fluid_vorticity: settings.fluid_vorticity,
            fluid_viscosity: settings.fluid_viscosity,
            fluid_ooze_rate: settings.fluid_ooze_rate,
            fluid_ooze_fade_rate: settings.fluid_ooze_fade_rate,
            fluid_ooze_rate_beta: settings.fluid_ooze_rate_beta,
            fluid_ooze_fade_rate_beta: settings.fluid_ooze_fade_rate_beta,
            fluid_ooze_rate_gamma: settings.fluid_ooze_rate_gamma,
            fluid_ooze_fade_rate_gamma: settings.fluid_ooze_fade_rate_gamma,
            fluid_ooze_still_rate: settings.fluid_ooze_still_rate,
            fluid_dye_escape_rate: settings.fluid_dye_escape_rate,
            fluid_dye_escape_rate_beta: settings.fluid_dye_escape_rate_beta,
            dye_diffusion: settings.dye_diffusion,
            dye_diffusion_no_fluid: settings.dye_diffusion_no_fluid,
            fluid_wind_push_strength: settings.fluid_wind_push_strength,
            fluid_slope_force_scale: settings.fluid_slope_force_scale,
            fluid_obstacle_strength: settings.fluid_obstacle_strength,

            fumaroles: settings.fumaroles.clone(),
            fluid_velocity_a,
            fluid_velocity_b,
            fluid_pressure_a,
            fluid_pressure_b,
            fluid_divergence,
            fluid_forces,
            fluid_force_vectors,
            fluid_dye_a,
            fluid_dye_b,
            fluid_params_buffer,
            fluid_fumaroles_buffer,
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
            fluid_clear_forces_pipeline,
            fluid_clear_force_vectors_pipeline,
            fluid_fumarole_pipeline,
            fluid_fumarole_dye_pipeline,
            fluid_clear_velocity_pipeline,
            fluid_inject_dye_pipeline,
            fluid_advect_dye_pipeline,
            fluid_advect_trail_pipeline,
            fluid_clear_dye_pipeline,
            fluid_diffuse_dye_no_fluid_pipeline,
            fluid_copy_trail_no_fluid_pipeline,
            fluid_bind_group_ab,
            fluid_bind_group_ba,
            fluid_time: 0.0,
            vector_force_power: settings.vector_force_power,
            vector_force_x: settings.vector_force_x,
            vector_force_y: settings.vector_force_y,
            prop_wash_strength: settings.prop_wash_strength,
            prop_wash_strength_fluid: settings.prop_wash_strength_fluid,

            propellers_enabled: settings.propellers_enabled,

            microswim_enabled: settings.microswim_enabled,
            microswim_coupling: settings.microswim_coupling,
            microswim_base_drag: settings.microswim_base_drag,
            microswim_anisotropy: settings.microswim_anisotropy,
            microswim_max_frame_vel: settings.microswim_max_frame_vel,
            microswim_torque_strength: settings.microswim_torque_strength,
            microswim_min_seg_displacement: settings.microswim_min_seg_displacement,
            microswim_min_total_deformation_sq: settings.microswim_min_total_deformation_sq,
            microswim_min_length_ratio: settings.microswim_min_length_ratio,
            microswim_max_length_ratio: settings.microswim_max_length_ratio,
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
            trail_show_energy: settings.trail_show_energy,
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
            dye_alpha_color: settings.dye_alpha_color,
            dye_beta_color: settings.dye_beta_color,
            dye_alpha_thinfilm: settings.dye_alpha_thinfilm,
            dye_alpha_thinfilm_mult: settings.dye_alpha_thinfilm_mult,
            dye_beta_thinfilm: settings.dye_beta_thinfilm,
            dye_beta_thinfilm_mult: settings.dye_beta_thinfilm_mult,
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
            // NOTE: spawn-only paths (snapshot load / paused spawns) can run before the first
            // `update()` call. They use `sim_params_cpu` as the template for the GPU params
            // buffer; if it's zeroed, `grid_size` becomes 0 and world->grid mapping collapses.
            sim_params_cpu: params,
            microswim_params_cpu: microswim_params_f32,
            environment_init_cpu: environment_init,
            part_base_angle_overrides: [f32::NAN; PART_OVERRIDE_SLOTS],
            part_base_angle_overrides_dirty: false,
            part_props_override: [[f32::NAN; 4]; PART_PROPS_OVERRIDE_VEC4S],
            part_flags_override: [f32::NAN; PART_TYPE_COUNT],
            part_properties_dirty: false,
            part_props_defaults: [[0.0; 4]; PART_PROPS_OVERRIDE_VEC4S],
            part_props_defaults_full: Vec::new(),
            part_flags_defaults: [0u32; PART_TYPE_COUNT],
            show_part_properties_editor: false,
            destroyed: false,
        };

        // Initialize override table (attempt to load from CSV if present).
        state.part_base_angle_overrides = [f32::NAN; PART_OVERRIDE_SLOTS];
        if let Ok(overrides) = load_part_base_angle_overrides_csv(Path::new(PART_OVERRIDES_CSV_PATH)) {
            state.part_base_angle_overrides = overrides;
            state.part_base_angle_overrides_dirty = true;
        }

        // Apply the overrides immediately (so first frame matches CSV/state).
        if state.part_base_angle_overrides_dirty {
            state.environment_init_cpu.part_angle_override =
                pack_part_base_angle_overrides_vec4(&state.part_base_angle_overrides);
            state.queue.write_buffer(
                &state.environment_init_params_buffer,
                0,
                bytemuck::bytes_of(&state.environment_init_cpu),
            );
            state.part_base_angle_overrides_dirty = false;
        }

        // Parse shader defaults for the part property editor.
        match parse_shared_wgsl_part_defaults(Path::new("shaders/shared.wgsl")) {
            Ok((props_full, flags)) => {
                state.part_props_defaults_full = props_full;
                for i in 0..PART_PROPS_OVERRIDE_VEC4S {
                    state.part_props_defaults[i] = state
                        .part_props_defaults_full
                        .get(i)
                        .copied()
                        .unwrap_or([0.0; 4]);
                }
                state.part_flags_defaults = flags;

                // Prefer a fully-populated numeric table: start overrides from defaults.
                state.part_props_override = state.part_props_defaults;
                state.part_flags_override = [f32::NAN; PART_TYPE_COUNT];
                state.part_properties_dirty = true;
            }
            Err(err) => {
                eprintln!("Warning: failed to parse shaders/shared.wgsl defaults: {err:?}");
            }
        }

        // Load part properties (amino + organ) from JSON, if present.
        if let Ok((props, flags)) = load_part_properties_json(Path::new(PART_PROPERTIES_JSON_PATH)) {
            // Merge JSON on top of defaults.
            for i in 0..PART_PROPS_OVERRIDE_VEC4S_USED {
                for c in 0..4 {
                    let v = props[i][c];
                    if !v.is_nan() {
                        state.part_props_override[i][c] = v;
                    }
                }
            }

            // Flags are intentionally ignored in the editor now.
            let _ = flags;
            state.part_flags_override = [f32::NAN; PART_TYPE_COUNT];
            state.part_properties_dirty = true;
        }

        // Apply part-properties overrides immediately so simulation/render match from the first frame.
        if state.part_properties_dirty {
            write_part_props_override_into_env_init(&mut state.environment_init_cpu, &state.part_props_override);
            state.environment_init_cpu.part_flags_override =
                pack_part_flags_override_vec4(&state.part_flags_override);
            state.queue.write_buffer(
                &state.environment_init_params_buffer,
                0,
                bytemuck::bytes_of(&state.environment_init_cpu),
            );
            state.part_properties_dirty = false;
        }
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
                genome_override_len: 0,
                genome_override_offset: 0,
                genome_override_packed: [0u32; GENOME_PACKED_WORDS],
                _pad_genome: [0u32; 2],
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
        let resized = image.resize_exact(self.env_grid_resolution, self.env_grid_resolution, FilterType::Lanczos3);
        let gray = resized.to_luma8();
        let width = gray.width() as usize;
        let height = gray.height() as usize;
        let raw = gray.as_raw();

        let mut gamma_values = Vec::with_capacity(self.env_grid_cell_count);
        for row in (0..height).rev() {
            let row_offset = row * width;
            for col in 0..width {
                let pix = raw[row_offset + col] as f32 / 255.0;
                gamma_values.push(pix.clamp(0.0, 1.0));
            }
        }

        debug_assert_eq!(gamma_values.len(), self.env_grid_cell_count);

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

    fn read_rain_map(path: &Path, env_grid_res: u32) -> anyhow::Result<(Vec<f32>, ColorImage)> {
        let image = image::open(path)?;
        let resized = image.resize_exact(env_grid_res, env_grid_res, FilterType::Lanczos3);
        let gray = resized.to_luma8();
        let width = gray.width() as usize;
        let height = gray.height() as usize;
        let raw = gray.as_raw();

        let cell_count = (env_grid_res as usize) * (env_grid_res as usize);
        let mut values = Vec::with_capacity(cell_count);
        for row in (0..height).rev() {
            let row_offset = row * width;
            for col in 0..width {
                let pix = raw[row_offset + col] as f32 / 255.0;
                values.push(pix.clamp(0.0, 1.0));
            }
        }

        debug_assert_eq!(values.len(), cell_count);
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
        let (alpha_values, thumbnail) = Self::read_rain_map(path, self.env_grid_resolution)?;

        // Update alpha values in CPU-side data (even indices)
        for i in 0..self.env_grid_cell_count {
            self.rain_map_data[i * 2] = alpha_values[i];
        }

        // Upload to GPU
        self.queue.write_buffer(
            &self.rain_map_buffer,
            0,
            bytemuck::cast_slice(&self.rain_map_data),
        );
        self.write_rain_map_texture();
        self.alpha_rain_map_path = Some(path.to_path_buf());
        self.alpha_rain_thumbnail = Some(RainThumbnail::new(thumbnail));
        println!("Loaded alpha rain map from {}", path.display());
        Ok(())
    }

    fn load_beta_rain_map<P: AsRef<Path>>(&mut self, path: P) -> anyhow::Result<()> {
        let path = path.as_ref();
        let (beta_values, thumbnail) = Self::read_rain_map(path, self.env_grid_resolution)?;

        // Update beta values in CPU-side data (odd indices)
        for i in 0..self.env_grid_cell_count {
            self.rain_map_data[i * 2 + 1] = beta_values[i];
        }

        // Upload to GPU
        self.queue.write_buffer(
            &self.rain_map_buffer,
            0,
            bytemuck::cast_slice(&self.rain_map_data),
        );
        self.write_rain_map_texture();
        self.beta_rain_map_path = Some(path.to_path_buf());
        self.beta_rain_thumbnail = Some(RainThumbnail::new(thumbnail));
        println!("Loaded beta rain map from {}", path.display());
        Ok(())
    }

    fn clear_alpha_rain_map(&mut self) {
        // Set alpha to 0.0 (no rain) in CPU-side data (even indices)
        for i in 0..self.env_grid_cell_count {
            self.rain_map_data[i * 2] = 0.0;
        }

        // Upload to GPU
        self.queue.write_buffer(
            &self.rain_map_buffer,
            0,
            bytemuck::cast_slice(&self.rain_map_data),
        );
        self.write_rain_map_texture();
        self.alpha_rain_map_path = None;
        self.alpha_rain_thumbnail = None;
        println!("Cleared alpha rain map (no rain)");
    }

    fn clear_beta_rain_map(&mut self) {
        // Set beta to 0.0 (no rain) in CPU-side data (odd indices)
        for i in 0..self.env_grid_cell_count {
            self.rain_map_data[i * 2 + 1] = 0.0;
        }

        // Upload to GPU
        self.queue.write_buffer(
            &self.rain_map_buffer,
            0,
            bytemuck::cast_slice(&self.rain_map_data),
        );
        self.write_rain_map_texture();
        self.beta_rain_map_path = None;
        self.beta_rain_thumbnail = None;
        println!("Cleared beta rain map (no rain)");
    }

    // Replenish population - spawns random agents when population is low
    fn replenish_population(&mut self) {
        // Avoid stacking replenish batches while spawns are already queued.
        if self.pending_spawn_upload || !self.cpu_spawn_queue.is_empty() {
            return;
        }

        // Only replenish if population drops below 100 agents.
        if self.alive_count >= 100 {
            return;
        }

        // If the world is empty, seed a larger initial batch so the sim can start.
        let spawn_count = if self.alive_count == 0 { 2000 } else { 100 };

        if self.alive_count == 0 {
            println!("Auto-replenish: population is 0, seeding {} agents", spawn_count);
        }

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
                    genome_override_len: 0,
                    genome_override_offset: 0,
                    genome_override_packed: [0u32; GENOME_PACKED_WORDS],
                    _pad_genome: [0u32; 2],
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
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
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
                    resource: self.chem_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.fluid_dye_a.as_entire_binding(),
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
                    resource: self.trail_grid_inject.as_entire_binding(),
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
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: wgpu::BindingResource::TextureView(&self.rain_map_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: self.microswim_params_buffer.as_entire_binding(),
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
                    resource: self.chem_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.fluid_dye_a.as_entire_binding(),
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
                    resource: self.trail_grid_inject.as_entire_binding(),
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
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: wgpu::BindingResource::TextureView(&self.rain_map_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: self.microswim_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Composite bind group (must be recreated too, since it references the resized buffers)
        let clayout0 = self.composite_agents_pipeline.get_bind_group_layout(0);
        self.composite_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Composite Bind Group"),
            layout: &clayout0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.visual_grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.agent_grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.fluid_dye_a.as_entire_binding(),
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
            gamma_diffuse: self.gamma_diffuse,
            alpha_fluid_convolution: self.alpha_fluid_convolution,
            beta_fluid_convolution: self.beta_fluid_convolution,
            fluid_slope_force_scale: self.fluid_slope_force_scale,
            fluid_obstacle_strength: self.fluid_obstacle_strength,
            gamma_blur: self.gamma_blur,
            gamma_shift: self.gamma_shift,
            alpha_slope_bias: self.alpha_slope_bias,
            beta_slope_bias: self.beta_slope_bias,
            alpha_multiplier: self.alpha_multiplier,
            beta_multiplier: self.beta_multiplier,
            dye_precipitation: self.dye_precipitation,
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
            morphology_change_cost: self.morphology_change_cost,
            pairing_cost: self.pairing_cost,
            prop_wash_strength: self.prop_wash_strength,
            prop_wash_strength_fluid: self.prop_wash_strength_fluid,

            propellers_enabled: self.propellers_enabled,

            microswim_enabled: self.microswim_enabled,
            microswim_coupling: self.microswim_coupling,
            microswim_base_drag: self.microswim_base_drag,
            microswim_anisotropy: self.microswim_anisotropy,
            microswim_max_frame_vel: self.microswim_max_frame_vel,
            microswim_torque_strength: self.microswim_torque_strength,
            microswim_min_seg_displacement: self.microswim_min_seg_displacement,
            microswim_min_total_deformation_sq: self.microswim_min_total_deformation_sq,
            microswim_min_length_ratio: self.microswim_min_length_ratio,
            microswim_max_length_ratio: self.microswim_max_length_ratio,
            repulsion_strength: self.repulsion_strength,
            agent_repulsion_strength: self.agent_repulsion_strength,
            limit_fps: self.limit_fps,
            limit_fps_25: self.limit_fps_25,
            render_interval: self.render_interval,
            gamma_debug_visual: self.gamma_debug_visual,
            slope_debug_visual: self.slope_debug_visual,
            rain_debug_visual: self.rain_debug_visual,
            fluid_enabled: self.fluid_enabled,
            fluid_show: self.fluid_show,
            fluid_dt: self.fluid_dt,
            fluid_decay: self.fluid_decay,
            fluid_jacobi_iters: self.fluid_jacobi_iters,
            fluid_vorticity: self.fluid_vorticity,
            fluid_viscosity: self.fluid_viscosity,
            fluid_ooze_rate: self.fluid_ooze_rate,
            fluid_ooze_fade_rate: self.fluid_ooze_fade_rate,
            fluid_ooze_rate_beta: self.fluid_ooze_rate_beta,
            fluid_ooze_fade_rate_beta: self.fluid_ooze_fade_rate_beta,
            fluid_ooze_rate_gamma: self.fluid_ooze_rate_gamma,
            fluid_ooze_fade_rate_gamma: self.fluid_ooze_fade_rate_gamma,
            fluid_ooze_still_rate: self.fluid_ooze_still_rate,
            fluid_dye_escape_rate: self.fluid_dye_escape_rate,
            fluid_dye_escape_rate_beta: self.fluid_dye_escape_rate_beta,
            dye_diffusion: self.dye_diffusion,
            dye_diffusion_no_fluid: self.dye_diffusion_no_fluid,
            fluid_wind_push_strength: self.fluid_wind_push_strength,
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
            trail_show_energy: self.trail_show_energy,
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
            dye_alpha_color: self.dye_alpha_color,
            dye_beta_color: self.dye_beta_color,
            dye_alpha_thinfilm: self.dye_alpha_thinfilm,
            dye_alpha_thinfilm_mult: self.dye_alpha_thinfilm_mult,
            dye_beta_thinfilm: self.dye_beta_thinfilm,
            dye_beta_thinfilm_mult: self.dye_beta_thinfilm_mult,
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

            fumaroles: self.fumaroles.clone(),

            // Resolution settings are read-only at runtime (require restart to change)
            env_grid_resolution: GRID_DIM_U32,
            fluid_grid_resolution: FLUID_GRID_SIZE,
            spatial_grid_resolution: SPATIAL_GRID_DIM as u32,
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
                if let Ok((epoch, new_count)) = result {
                    self.apply_alive_count_readback(epoch, new_count);
                }
                self.alive_readback_inflight[idx] = false;
            }
        }

        // Selected agent readback is optional and should never stall the frame.
        // Drain any completed slot(s); last completion wins.
        for slot in 0..self.selected_agent_readbacks.len() {
            let message = {
                let mut guard = self.selected_agent_readback_pending[slot].lock().unwrap();
                guard.take()
            };
            if let Some(result) = message {
                if let Ok(agent) = result {
                    // IMPORTANT:
                    // Do NOT clear selection on alive==0.
                    // The GPU tries to transfer selection on death; keeping the inspector open
                    // avoids a close/freeze loop and lets the next readback show the new agent.
                    self.selected_agent_data = Some(agent);

                    // Update camera target if following this agent
                    if self.follow_selected_agent {
                        self.camera_target = agent.position;
                    }
                }
                self.selected_agent_readback_inflight[slot] = false;
            }
        }
    }

    fn destroy_resources(&mut self) {
        if self.destroyed {
            return;
        }

        // Ensure GPU work completes before releasing resources to avoid device validation issues.
        self.device.poll(wgpu::Maintain::Wait);

        // Drain any completed readbacks so their callbacks can unmap buffers before destruction.
        self.process_completed_alive_readbacks();

        self.agents_buffer_a.destroy();
        self.agents_buffer_b.destroy();
        self.chem_grid.destroy();
        self.rain_map_buffer.destroy();
        self.agent_spatial_grid_buffer.destroy();
        self.gamma_grid.destroy();
        self.trail_grid.destroy();
        self.trail_grid_inject.destroy();
        self.visual_grid_buffer.destroy();
        self.params_buffer.destroy();
        self.environment_init_params_buffer.destroy();
        self.spawn_debug_counters.destroy();
        self.init_dead_dispatch_args.destroy();
        self.init_dead_params_buffer.destroy();
        for buffer in &self.alive_readbacks {
            buffer.destroy();
        }
        self.debug_readback.destroy();
        self.agents_readback.destroy();
        self.selected_agent_buffer.destroy();
        for buffer in &self.selected_agent_readbacks {
            buffer.as_ref().destroy();
        }
        self.new_agents_buffer.destroy();
        self.spawn_readback.destroy();
        self.spawn_requests_buffer.destroy();
        self.visual_texture.destroy();

        self.cpu_spawn_queue.clear();
        self.pending_spawn_upload = false;
        self.spawn_request_count = 0;
        self.selected_agent_index = None;
        self.selected_agent_data = None;
        for inflight in &mut self.selected_agent_readback_inflight {
            *inflight = false;
        }
        for pending in &self.selected_agent_readback_pending {
            if let Ok(mut guard) = pending.lock() {
                *guard = None;
            }
        }

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
                if let Ok((epoch, new_count)) = result {
                    self.apply_alive_count_readback(epoch, new_count);
                }
            }
            self.alive_readback_inflight[slot] = false;
        }
        slot
    }

    fn apply_alive_count_readback(&mut self, epoch: u32, new_count: u32) {
        // Readbacks can complete out-of-order (we use multiple staging slots). Never let an
        // older completion overwrite a newer count.
        if epoch < self.alive_readback_last_applied_epoch {
            return;
        }

        // When paused, we still clear `spawn_debug_counters` to keep other counters tidy,
        // but we intentionally skip compaction/merge. That means the alive counter (slot [2])
        // is not rewritten and can read back stale/zero values even though agents still exist.
        // Ignore ALL readbacks while paused to prevent the UI showing 0 agents.
        // Exception: allow nonzero counts through so snapshot-load/spawns update immediately.
        if self.is_paused {
            if new_count == 0 {
                return;
            }
            // Accept nonzero updates even when paused (e.g., from snapshot load or manual spawn).
        }

        // Avoid UI/sim flicker: a single 0 readback can happen if we sample the alive counter
        // right after it was cleared (e.g. on a step where sim kernels were gated). Require two
        // consecutive 0s before accepting a transition from nonzero -> 0.
        if new_count == 0 && (self.alive_count > 0 || self.agent_count > 0) {
            self.alive_readback_zero_streak = self.alive_readback_zero_streak.saturating_add(1);
            if self.alive_readback_zero_streak < 2 {
                return;
            }
        } else {
            self.alive_readback_zero_streak = 0;
        }

        self.alive_readback_last_applied_epoch = epoch;
        self.agent_count = new_count;
        self.alive_count = new_count;
    }

    fn copy_state_to_staging(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        slot: usize,
    ) -> Arc<wgpu::Buffer> {
        let alive_buffer = self.alive_readbacks[slot].clone();
        // alive counter lives in spawn_debug_counters[2] (u32) at byte offset 8
        encoder.copy_buffer_to_buffer(&self.spawn_debug_counters, 8, alive_buffer.as_ref(), 0, 4);

        alive_buffer
    }

    fn kickoff_alive_readback(&mut self, slot: usize, alive_buffer: Arc<wgpu::Buffer>) {
        let buffer_for_map = alive_buffer.clone();
        let buffer_for_callback = buffer_for_map.clone();
        let pending = self.alive_readback_pending[slot].clone();
        let request_epoch = self.epoch as u32;
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
                        Ok((request_epoch, new_count))
                    }
                    Err(_) => {
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

        if self.selected_agent_index.is_none() {
            return;
        }

        // IMPORTANT: Never block the frame on a GPU->CPU readback.
        // We request an async map at ~60Hz and consume the result in
        // `process_completed_alive_readbacks()` (which polls with Maintain::Poll).
        let now = std::time::Instant::now();
        if now.duration_since(self.selected_agent_readback_last_request)
            < std::time::Duration::from_millis(SELECTED_AGENT_READBACK_INTERVAL_MS)
        {
            return;
        }

        // Find a free slot (not currently mapped/in-flight).
        let mut chosen: Option<usize> = None;
        let slots = self.selected_agent_readbacks.len();
        for offset in 0..slots {
            let idx = (self.selected_agent_readback_slot + offset) % slots;
            if !self.selected_agent_readback_inflight[idx] {
                chosen = Some(idx);
                break;
            }
        }
        let Some(slot) = chosen else {
            return;
        };

        self.selected_agent_readback_last_request = now;
        self.selected_agent_readback_inflight[slot] = true;
        self.selected_agent_readback_slot = (slot + 1) % self.selected_agent_readbacks.len();

        // Copy the latest selected agent into this slot right before mapping.
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Selected Agent Readback Encoder"),
                });
            encoder.copy_buffer_to_buffer(
                &self.selected_agent_buffer,
                0,
                self.selected_agent_readbacks[slot].as_ref(),
                0,
                std::mem::size_of::<Agent>() as u64,
            );
            self.queue.submit(Some(encoder.finish()));
        }

        let slice_for_map = self.selected_agent_readbacks[slot].as_ref().slice(..);
        let buffer_for_callback = self.selected_agent_readbacks[slot].clone();
        let pending = self.selected_agent_readback_pending[slot].clone();

        slice_for_map.map_async(wgpu::MapMode::Read, move |result| {
            let message = match result {
                Ok(()) => {
                    let slice = buffer_for_callback.slice(..);
                    let data = slice.get_mapped_range();

                    let size = std::mem::size_of::<Agent>();
                    let parsed = if data.len() >= size {
                        let agent_bytes = &data[..size];
                        Ok(bytemuck::pod_read_unaligned::<Agent>(agent_bytes))
                    } else {
                        Err(())
                    };

                    drop(data);
                    buffer_for_callback.unmap();
                    parsed
                }
                Err(_) => Err(()),
            };

            if let Ok(mut guard) = pending.lock() {
                *guard = Some(message);
            }
        });
    }

    fn process_spawn_requests_only(&mut self, cpu_spawn_count: u32, do_readbacks: bool) {
        self.process_completed_alive_readbacks();

        if cpu_spawn_count == 0 && !do_readbacks {
            return;
        }

        // The spawn/compact/merge WGSL path is designed around a 2000-per-batch limit.
        // Allowing larger counts (e.g. 2048) causes partial uploads and inconsistent counters.
        let cpu_spawn_count = cpu_spawn_count
            .min(MAX_CPU_SPAWNS_PER_BATCH)
            .min(MAX_SPAWN_REQUESTS as u32);

        // The spawn/merge/compact shaders use `params.cpu_spawn_count`, `params.agent_count`, and
        // `params.max_agents` for bounds checks. During snapshot load this path can run before
        // `update()` has written a fresh params buffer, which would result in 0 spawns.
        {
            let mut params = self.sim_params_cpu;
            // Ensure world/grid mapping is valid even before the first `update()`.
            params.grid_size = self.sim_size;
            params.max_agents = self.agent_buffer_capacity as u32;
            params.cpu_spawn_count = cpu_spawn_count;
            params.agent_count = self.agent_count;
            self.sim_params_cpu = params;
            self.queue
                .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Paused Spawn Encoder"),
            });

        // Always clear both spawn_debug_counters[0] and [2] so compact/merge never uses stale
        // offsets when this path is used (manual spawns / snapshot loads / paused spawns).
        encoder.clear_buffer(&self.spawn_debug_counters, 0, None);

        if cpu_spawn_count > 0 {
            encoder.clear_buffer(&self.spawn_debug_counters, 0, None);

            // Upload spawn requests for this batch so GPU has per-request seeds/data
            // Limit to the count actually being processed (capped at 2000 elsewhere)
            let upload_len = (cpu_spawn_count as usize)
                .min(self.cpu_spawn_queue.len())
                .min(MAX_CPU_SPAWNS_PER_BATCH as usize);
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

            // Spawn-only path can run while paused and skips the normal A->B process pass.
            // That means the "current" agent data lives in the ping-pong-selected input buffer
            // (A when ping_pong=false, B when ping_pong=true). Compact must read from that buffer,
            // otherwise each batch will compact an empty/stale buffer and overwrite prior batches.
            //
            // We also materialize results back into agents_buffer_a (the snapshot/save canonical).
            let (bg_compact_merge, wrote_to_a_directly) = if self.ping_pong {
                // Current is B, output is A.
                (&self.compute_bind_group_b, true)
            } else {
                // Current is A, output is B (will be copied back into A).
                (&self.compute_bind_group_a, false)
            };

            // Cheap tail-safety: clear the compaction destination buffer so any unwritten slots
            // remain fully zeroed (prevents stale/ghost agents from reappearing).
            let compact_dst = if wrote_to_a_directly {
                &self.agents_buffer_a
            } else {
                &self.agents_buffer_b
            };
            encoder.clear_buffer(compact_dst, 0, None);

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Paused Spawn Pass"),
                    timestamp_writes: None,
                });

                cpass.set_pipeline(&self.compact_pipeline);
                cpass.set_bind_group(0, bg_compact_merge, &[]);
                // Scan ENTIRE buffer to ensure no alive agents are missed
                cpass.dispatch_workgroups((self.agent_buffer_capacity as u32 + 63) / 64, 1, 1);

                cpass.set_pipeline(&self.cpu_spawn_pipeline);
                cpass.set_bind_group(0, bg_compact_merge, &[]);
                cpass.dispatch_workgroups((MAX_SPAWN_REQUESTS as u32) / 64, 1, 1);

                cpass.set_pipeline(&self.merge_pipeline);
                cpass.set_bind_group(0, bg_compact_merge, &[]);
                cpass.dispatch_workgroups((MAX_SPAWN_REQUESTS as u32) / 64, 1, 1);

                // NOTE: In this spawn-only path we already fully clear the compaction
                // destination buffer (`compact_dst`) before compact/merge. That guarantees
                // the tail is zeroed/dead, so we do not need to run initialize-dead here.
                // Skipping it also avoids any risk of clearing freshly-merged agents if
                // the alive counter read is not yet visible on some backends.

                cpass.set_pipeline(&self.finalize_merge_pipeline);
                cpass.set_bind_group(0, bg_compact_merge, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }

            // Ensure agents are materialized in buffer A for snapshot/save and for subsequent
            // snapshot-load batches.
            if !wrote_to_a_directly {
                let expected_alive = (self.agent_count + cpu_spawn_count)
                    .min(self.agent_buffer_capacity as u32);
                let bytes = (expected_alive as u64) * (std::mem::size_of::<Agent>() as u64);
                if bytes > 0 {
                    encoder.copy_buffer_to_buffer(
                        &self.agents_buffer_b,
                        0,
                        &self.agents_buffer_a,
                        0,
                        bytes,
                    );
                }
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

            // We guarantee agents are materialized in buffer A by the end of this path.
            // Make that buffer the "current" ping-pong input for subsequent simulation frames.
            self.ping_pong = false;
        }
    }

    pub fn update(&mut self, should_draw: bool, frame_dt: f32) {
        let cpu_update_start = std::time::Instant::now();

        // Make perf logs self-describing: inspector preview only runs on draw frames
        // and only when an agent is selected.
        self.frame_profiler
            .set_inspector_requested(should_draw && self.selected_agent_index.is_some());

        self.frame_profiler.begin_frame();

        // Push base-angle overrides into the existing EnvironmentInitParams uniform.
        // This avoids an extra binding while still allowing runtime edits.
        if self.part_base_angle_overrides_dirty {
            self.environment_init_cpu.part_angle_override =
                pack_part_base_angle_overrides_vec4(&self.part_base_angle_overrides);
            self.queue.write_buffer(
                &self.environment_init_params_buffer,
                0,
                bytemuck::bytes_of(&self.environment_init_cpu),
            );
            self.part_base_angle_overrides_dirty = false;
        }

        if self.part_properties_dirty {
            write_part_props_override_into_env_init(&mut self.environment_init_cpu, &self.part_props_override);
            self.environment_init_cpu.part_flags_override =
                pack_part_flags_override_vec4(&self.part_flags_override);
            self.queue.write_buffer(
                &self.environment_init_params_buffer,
                0,
                bytemuck::bytes_of(&self.environment_init_cpu),
            );
            self.part_properties_dirty = false;
        }

        // Smooth camera following with continuous integration
        if self.follow_selected_agent {
            // Store previous camera position for motion blur
            self.prev_camera_pan = self.camera_pan;

            // Frame-rate independent integration factor using exponential decay
            let clamped_dt = frame_dt.clamp(0.001, 0.1);
            let damping_rate = 8.0; // Much faster follow (~12% step at 60fps: 1 - exp(-8.0*0.016) � 0.12)
            let integration_factor = 1.0 - (-damping_rate * clamped_dt).exp();

            // Smoothly integrate target position into camera position
            self.camera_pan[0] += (self.camera_target[0] - self.camera_pan[0]) * integration_factor;
            self.camera_pan[1] += (self.camera_target[1] - self.camera_pan[1]) * integration_factor;

            // Clamp to world bounds
            let sim_size = self.sim_size;
            self.camera_pan[0] = self.camera_pan[0].clamp(-0.25 * sim_size, 1.25 * sim_size);
            self.camera_pan[1] = self.camera_pan[1].clamp(-0.25 * sim_size, 1.25 * sim_size);
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

        // Advance RNG only when not paused AND there are living agents.
        // Auto-replenish is handled separately so the world can recover from 0 agents.
        let has_living_agents = self.alive_count > 0;
        if !self.is_paused && has_living_agents {
            self.rng_state ^= self.rng_state << 13;
            self.rng_state ^= self.rng_state >> 7;
            self.rng_state ^= self.rng_state << 17;
        }

        if !self.is_paused && self.auto_replenish {
            self.replenish_population();
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
            self.difficulty
                .spawn_prob
                .apply_to(self.spawn_probability, false)
                .clamp(0.0, 5.0)
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
            morphology_change_cost: self.morphology_change_cost,
            spawn_probability: effective_spawn_p,
            death_probability: effective_death_p,
            grid_size: self.sim_size,
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
            gamma_diffuse: self.gamma_diffuse,
            gamma_blur: self.gamma_blur,
            gamma_shift: self.gamma_shift,
            alpha_slope_bias: self.alpha_slope_bias,
            beta_slope_bias: self.beta_slope_bias,
            alpha_multiplier: current_alpha,
            beta_multiplier: current_beta,
            // NOTE: Repurposed padding: used by shaders as a boolean flag.
            // 0 = fluid simulation disabled (dye layer is visual/sensing only)
            // 1 = fluid simulation enabled
            _pad_rain0: if self.fluid_enabled { 1 } else { 0 },
            _pad_rain1: 0,
            // Calculate expected rain drops for targeted dispatch
            // Expected drops = grid_cells * multiplier * 0.00005 * avg_rain_map
            // For simplicity, assume avg_rain_map ~= 1.0 (can refine later)
            rain_drop_count: {
                let grid_cells = (GRID_DIM * GRID_DIM) as f32;
                let alpha_drops = (grid_cells * current_alpha * 0.00005).ceil() as u32;
                let beta_drops = (grid_cells * current_beta * 0.00005).ceil() as u32;
                alpha_drops + beta_drops
            },
            alpha_rain_drop_count: {
                let grid_cells = (GRID_DIM * GRID_DIM) as f32;
                (grid_cells * current_alpha * 0.00005).ceil() as u32
            },
            dye_precipitation: self.dye_precipitation,
            chemical_slope_scale_alpha: self.chemical_slope_scale_alpha,
            chemical_slope_scale_beta: self.chemical_slope_scale_beta,
            mutation_rate: self.mutation_rate,
            food_power: effective_food_power,
            poison_power: effective_poison_power,
            pairing_cost: self.pairing_cost,
            max_agents: self.agent_buffer_capacity as u32,
            cpu_spawn_count,
            // IMPORTANT:
            // Do not rely on async CPU readback of alive_count/agent_count to bound
            // simulation dispatch. In Full Speed / Fast Draw modes, the readback can
            // lag behind newly spawned agents, causing them to be skipped for several
            // steps (which looks like hotspots / missing updates).
            //
            // All core kernels already early-return on `alive==0`, so it is safe to
            // dispatch over the full buffer capacity for correctness.
            agent_count: self.agent_buffer_capacity as u32,
            // Used for population-pressure scaling (death probability multiplier).
            // This should reflect actual alive population, not dispatch bound.
            population_count: self.agent_count,
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
            // NOTE: prop wash also drives direct chem/gamma displacement in the simulation shader,
            // so keep it active even when fluids are disabled.
            prop_wash_strength: self.prop_wash_strength,
            // Used as a general "wash/propulsion" strength knob in shaders.
            // Even when fluids are disabled, virtual-medium propulsion can still use this value;
            // actual fluid coupling remains disabled via fluid_wind_push_strength=0.
            prop_wash_strength_fluid: self.prop_wash_strength_fluid,
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
            trail_show: if self.trail_show {
                if self.trail_show_energy { 2 } else { 1 }
            } else {
                0
            },
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
            // Even when the full fluid solver is disabled, we can still use the dye/trail
            // layers as isotropic diffusion fields; keep the overlay toggle functional.
            fluid_show: if self.fluid_show { 1 } else { 0 },
            fluid_wind_push_strength: if self.fluid_enabled {
                self.fluid_wind_push_strength
            } else {
                0.0
            },
            alpha_fluid_convolution: if self.fluid_enabled {
                self.alpha_fluid_convolution
            } else {
                0.0
            },
            beta_fluid_convolution: if self.fluid_enabled {
                self.beta_fluid_convolution
            } else {
                0.0
            },
            fluid_slope_force_scale: self.fluid_slope_force_scale,
            fluid_obstacle_strength: self.fluid_obstacle_strength,
            dye_alpha_color_r: self.dye_alpha_color[0],
            dye_alpha_color_g: self.dye_alpha_color[1],
            dye_alpha_color_b: self.dye_alpha_color[2],
            _pad_dye_alpha_color: if self.dye_alpha_thinfilm {
                self.dye_alpha_thinfilm_mult
            } else {
                0.0
            },
            dye_beta_color_r: self.dye_beta_color[0],
            dye_beta_color_g: self.dye_beta_color[1],
            dye_beta_color_b: self.dye_beta_color[2],
            _pad_dye_beta_color: if self.dye_beta_thinfilm {
                self.dye_beta_thinfilm_mult
            } else {
                0.0
            },
        };

        // Keep CPU mirror in sync with the GPU uniform buffer.
        self.sim_params_cpu = params;
        self.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));

        // Update microswim params (flat float list; packed as vec4s in the shader).
        {
            let microswim_params_f32: [f32; MICROSWIM_PARAM_FLOATS] = [
                // vec4 0
                if self.microswim_enabled { 1.0 } else { 0.0 },
                self.microswim_coupling,
                self.microswim_base_drag,
                self.microswim_anisotropy,
                // vec4 1
                self.microswim_max_frame_vel,
                self.microswim_torque_strength,
                self.microswim_min_seg_displacement,
                self.microswim_min_total_deformation_sq,
                // vec4 2
                self.microswim_min_length_ratio,
                self.microswim_max_length_ratio,
                if self.propellers_enabled { 1.0 } else { 0.0 },
                0.0,
                // vec4 3 (reserved)
                0.0,
                0.0,
                0.0,
                0.0,
            ];
            self.microswim_params_cpu = microswim_params_f32;
            let microswim_params_bytes = pack_f32_uniform(&microswim_params_f32);
            self.queue.write_buffer(
                &self.microswim_params_buffer,
                0,
                &microswim_params_bytes,
            );
        }

        // Update fluid params (dt is the user-controlled fluid solver dt)
        {
            let fluid_params_f32: [f32; 32] = [
                self.epoch as f32,         // time
                self.fluid_dt,             // dt
                self.fluid_decay,          // decay
                FLUID_GRID_SIZE as f32,    // grid_size (as f32)
                0.0, 0.0, 0.0, 0.0,        // mouse
                // splat: x=lift_min_speed, y=lift_multiplier, z=vorticity confinement, w=viscosity
                self.fluid_ooze_rate,
                self.fluid_ooze_fade_rate,
                self.fluid_vorticity,
                self.fluid_viscosity,
                self.fluid_slope_force_scale,
                self.fluid_obstacle_strength,
                self.vector_force_x,
                self.vector_force_y,
                self.vector_force_power,
                self.fluid_ooze_still_rate,
                self.fluid_ooze_rate_beta,
                self.fluid_ooze_fade_rate_beta,
                self.fluid_dye_escape_rate,
                self.fluid_dye_escape_rate_beta,
                self.dye_precipitation,
                self.fluid_ooze_rate_gamma,
                self.fluid_ooze_fade_rate_gamma,
                self.dye_diffusion,
                self.dye_diffusion_no_fluid,
                0.0, 0.0, 0.0, 0.0, 0.0,  // padding
            ];
            let fluid_params_bytes = pack_f32_uniform(&fluid_params_f32);
            self.queue
                .write_buffer(&self.fluid_params_buffer, 0, &fluid_params_bytes);
        }

        // Update fluid fumaroles buffer (flat float list consumed only by shaders/fluid.wgsl).
        {
            let mut data: Vec<f32> = vec![0.0; FUMAROLE_BUFFER_FLOATS];
            let count = self.fumaroles.len().min(MAX_FUMAROLES);
            data[0] = count as f32;

            for (i, fum) in self.fumaroles.iter().take(MAX_FUMAROLES).enumerate() {
                let mut fum = fum.clone();
                fum.sanitize();

                let base = 1 + i * FUMAROLE_STRIDE_F32;
                let rad = fum.dir_degrees.to_radians();
                let dir_x = rad.cos();
                let dir_y = rad.sin();

                data[base + 0] = if fum.enabled { 1.0 } else { 0.0 };
                data[base + 1] = fum.x_frac;
                data[base + 2] = fum.y_frac;
                data[base + 3] = dir_x;
                data[base + 4] = dir_y;
                data[base + 5] = fum.strength;
                data[base + 6] = fum.spread;
                data[base + 7] = fum.alpha_dye_rate;
                data[base + 8] = fum.beta_dye_rate;
                data[base + 9] = fum.variation;
            }

            let bytes = pack_f32_uniform(&data);
            self.queue
                .write_buffer(&self.fluid_fumaroles_buffer, 0, &bytes);
        }

        // Handle CPU spawns - only when not paused (consistent with autospawn)
        if !self.is_paused && cpu_spawn_count > 0 {
            // Use the reliable paused spawn path for all manual spawns
            println!("Using reliable spawn path for {} requests", cpu_spawn_count);
            self.process_spawn_requests_only(cpu_spawn_count, true);

            // IMPORTANT: Avoid processing the same CPU spawn batch twice.
            // process_spawn_requests_only() already materializes + merges these requests.
            // If we keep cpu_spawn_count > 0, the normal post-fluid cpu_spawn_pipeline will
            // run again and generate duplicates/hotspots.
            cpu_spawn_count = 0;
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
        // Rebuild slope even when paused so UI changes (e.g. Alpha/Beta Slope Mix)
        // take effect immediately and chem-driven height mixing can't get "stuck".
        let run_slope = self.slope_counter == 0;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Update Encoder"),
            });

        let time_dispatches = self.frame_profiler.should_time_dispatches();
        let time_dispatches_detail = time_dispatches && self.frame_profiler.dispatch_timing_detail;

        self.frame_profiler.write_ts_encoder(&mut encoder, TS_UPDATE_START);

        // Clear counters (spawn_counter is reset in-shader at end of previous frame)
        // IMPORTANT: Always clear both spawn_debug_counters[0] and [2] to avoid stale offsets
        // in compact/merge when simulation is paused or should_run_simulation is false.
        encoder.clear_buffer(&self.spawn_debug_counters, 0, None);

        // Select bind groups based on ping-pong orientation
        // bg_process: agents_in -> agents_out
        // bg_swap: (swapped) agents_in/agents_out reversed for compaction and merge
        let (bg_process, bg_swap) = if self.ping_pong {
            (&self.compute_bind_group_b, &self.compute_bind_group_a)
        } else {
            (&self.compute_bind_group_a, &self.compute_bind_group_b)
        };

        // Pass 1: environment prep + optional reproduction.
        // We intentionally end this pass before running the main agent simulation so that any
        // reproduction writes to the agent buffer are guaranteed visible.
        if !time_dispatches_detail {
            // Normal (or coarse-timed) path: keep as a single pass.
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass"),
                    timestamp_writes: None,
                });

                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_SIM_PASS_START);

                if run_diffusion && !self.perf_skip_diffusion {
                    cpass.set_pipeline(&self.diffuse_pipeline);
                    cpass.set_bind_group(0, bg_process, &[]);
                    let groups_x = (GRID_DIM_U32 + DIFFUSE_WG_SIZE_X - 1) / DIFFUSE_WG_SIZE_X;
                    let groups_y = (GRID_DIM_U32 + DIFFUSE_WG_SIZE_Y - 1) / DIFFUSE_WG_SIZE_Y;
                    cpass.dispatch_workgroups(groups_x, groups_y, 1);

                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_DIFFUSE);

                    // Commit staged alpha/beta back into the environment grids.
                    cpass.set_pipeline(&self.diffuse_commit_pipeline);
                    cpass.set_bind_group(0, bg_process, &[]);
                    cpass.dispatch_workgroups(groups_x, groups_y, 1);

                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_DIFFUSE_COMMIT);

                    // Apply targeted rain drops AFTER diffusion commit
                    // This adds fresh saturated drops to specific cells after diffusion has spread existing values
                    if !self.perf_skip_rain {
                        let rain_drop_count = params.rain_drop_count;
                        if rain_drop_count > 0 {
                            cpass.set_pipeline(&self.rain_pipeline);
                            cpass.set_bind_group(0, bg_process, &[]);
                            let rain_workgroups = (rain_drop_count + 255) / 256;
                            cpass.dispatch_workgroups(rain_workgroups, 1, 1);
                        }
                    }
                }

                if !run_diffusion || self.perf_skip_diffusion {
                    // Keep timestamp progression stable even when skipping.
                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_DIFFUSE);
                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_DIFFUSE_COMMIT);
                }

                // Prepare trails every simulation frame (copy/decay + optional blur into trail_grid_inject).
                // The actual advection runs in the fluid pass.
                // Skip entirely when trails are fully invisible.
                if should_run_simulation && !self.perf_skip_trail_prep && self.trail_opacity > 0.0 {
                    let groups_x = (GRID_DIM_U32 + DIFFUSE_WG_SIZE_X - 1) / DIFFUSE_WG_SIZE_X;
                    let groups_y = (GRID_DIM_U32 + DIFFUSE_WG_SIZE_Y - 1) / DIFFUSE_WG_SIZE_Y;
                    cpass.set_pipeline(&self.diffuse_trails_pipeline);
                    cpass.set_bind_group(0, bg_process, &[]);
                    cpass.dispatch_workgroups(groups_x, groups_y, 1);
                }

                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_TRAILS_PREP);

                if run_slope && !self.perf_skip_slope {
                    cpass.set_pipeline(&self.gamma_slope_pipeline);
                    cpass.set_bind_group(0, bg_process, &[]);
                    let groups_x = (GRID_DIM_U32 + SLOPE_WG_SIZE_X - 1) / SLOPE_WG_SIZE_X;
                    let groups_y = (GRID_DIM_U32 + SLOPE_WG_SIZE_Y - 1) / SLOPE_WG_SIZE_Y;
                    cpass.dispatch_workgroups(groups_x, groups_y, 1);
                }

                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_SLOPE);

                // Clear visual grid only when drawing this step
                if should_draw && !self.perf_skip_draw {
                    cpass.set_pipeline(&self.clear_visual_pipeline);
                    cpass.set_bind_group(0, bg_process, &[]);
                    let width_workgroups =
                        (self.surface_config.width + CLEAR_WG_SIZE_X - 1) / CLEAR_WG_SIZE_X;
                    let height_workgroups =
                        (self.surface_config.height + CLEAR_WG_SIZE_Y - 1) / CLEAR_WG_SIZE_Y;
                    cpass.dispatch_workgroups(width_workgroups, height_workgroups, 1);

                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_CLEAR_VISUAL);

                    // Apply motion blur BEFORE agent rendering (blur only the background)
                    if self.follow_selected_agent {
                        cpass.set_pipeline(&self.motion_blur_pipeline);
                        cpass.set_bind_group(0, bg_process, &[]);
                        cpass.dispatch_workgroups(width_workgroups, height_workgroups, 1);
                    }

                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_MOTION_BLUR);

                    // Clear agent grid for agent rendering
                    cpass.set_pipeline(&self.clear_agent_grid_pipeline);
                    cpass.set_bind_group(0, bg_process, &[]);
                    cpass.dispatch_workgroups(width_workgroups, height_workgroups, 1);

                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_CLEAR_AGENT_GRID);
                }

                if !should_draw {
                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_CLEAR_VISUAL);
                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_MOTION_BLUR);
                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_CLEAR_AGENT_GRID);
                }

                // Run simulation compute passes, but skip everything when paused or no living agents
                if should_run_simulation && !self.perf_skip_repro {
                    // REPRODUCTION FIRST: Update pairing_counter and energy in agents_in
                    // Uses reproduction bind groups where binding 1 = agents_in (read-write)
                    let reproduction_bg = if self.ping_pong {
                        &self.reproduction_bind_group_b
                    } else {
                        &self.reproduction_bind_group_a
                    };
                    cpass.set_pipeline(&self.reproduction_pipeline);
                    cpass.set_bind_group(0, reproduction_bg, &[]);
                    cpass.dispatch_workgroups(
                        ((self.agent_count) + 255) / 256,
                        1,
                        1,
                    );

                    // Spatial grid clear disabled: we use an epoch-stamped spatial grid.
                    // populate_agent_spatial_grid writes the epoch stamp for touched cells;
                    // all readers ignore cells whose stamp != current epoch.

                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_CLEAR_SPATIAL);
                }

                if !should_run_simulation {
                    // Keep timestamp progression stable when paused.
                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_CLEAR_SPATIAL);
                }
            }

            // Optional per-segment submit+wait timing (CPU wall time).
            self.frame_profiler.maybe_submit_encoder_segment(
                &self.device,
                &self.queue,
                &mut encoder,
                DispatchSegment::SimPre,
                "Update Encoder (Timed Segment)",
            );
        } else {
            // Detailed timing path: split Pass 1 into sub-passes and submit+wait after each.
            // This is only active on sampled frames.

            // (a) Diffuse
            let groups_x = (GRID_DIM_U32 + DIFFUSE_WG_SIZE_X - 1) / DIFFUSE_WG_SIZE_X;
            let groups_y = (GRID_DIM_U32 + DIFFUSE_WG_SIZE_Y - 1) / DIFFUSE_WG_SIZE_Y;
            if run_diffusion && !self.perf_skip_diffusion {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass (SimPre Diffuse)"),
                    timestamp_writes: None,
                });

                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_SIM_PASS_START);

                cpass.set_pipeline(&self.diffuse_pipeline);
                cpass.set_bind_group(0, bg_process, &[]);
                cpass.dispatch_workgroups(groups_x, groups_y, 1);

                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_DIFFUSE);
            }
            // Only time diffusion when it's actually dispatched
            if run_diffusion && !self.perf_skip_diffusion {
                self.frame_profiler.maybe_submit_encoder_segment(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    DispatchSegment::SimPreDiffuse,
                    "Update Encoder (Timed Segment)",
                );
            }

            // (a2) Commit
            if run_diffusion && !self.perf_skip_diffusion {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass (SimPre Diffuse Commit)"),
                    timestamp_writes: None,
                });

                cpass.set_pipeline(&self.diffuse_commit_pipeline);
                cpass.set_bind_group(0, bg_process, &[]);
                cpass.dispatch_workgroups(groups_x, groups_y, 1);

                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_DIFFUSE_COMMIT);
            }
            // Only time commit when it's actually dispatched
            if run_diffusion && !self.perf_skip_diffusion {
                self.frame_profiler.maybe_submit_encoder_segment(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    DispatchSegment::SimPreDiffuseCommit,
                    "Update Encoder (Timed Segment)",
                );
            }

            // (a3) Rain - apply after diffusion commit (only when diffusion actually ran)
            if run_diffusion && !self.perf_skip_diffusion && !self.perf_skip_rain {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass (SimPre Rain)"),
                    timestamp_writes: None,
                });

                let rain_drop_count = params.rain_drop_count;
                if rain_drop_count > 0 {
                    cpass.set_pipeline(&self.rain_pipeline);
                    cpass.set_bind_group(0, bg_process, &[]);
                    let rain_workgroups = (rain_drop_count + 255) / 256;
                    cpass.dispatch_workgroups(rain_workgroups, 1, 1);
                }
            }
            // Only time rain when it's actually dispatched
            if run_diffusion && !self.perf_skip_diffusion && !self.perf_skip_rain {
                self.frame_profiler.maybe_submit_encoder_segment(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    DispatchSegment::SimPreRain,
                    "Update Encoder (Timed Segment)",
                );
            }

            // (b) Trails prep
            if should_run_simulation && !self.perf_skip_trail_prep && self.trail_opacity > 0.0 {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass (SimPre Trails)"),
                    timestamp_writes: None,
                });
                let groups_x = (GRID_DIM_U32 + DIFFUSE_WG_SIZE_X - 1) / DIFFUSE_WG_SIZE_X;
                let groups_y = (GRID_DIM_U32 + DIFFUSE_WG_SIZE_Y - 1) / DIFFUSE_WG_SIZE_Y;
                cpass.set_pipeline(&self.diffuse_trails_pipeline);
                cpass.set_bind_group(0, bg_process, &[]);
                cpass.dispatch_workgroups(groups_x, groups_y, 1);

                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_TRAILS_PREP);
            }
            // Only time trails when actually dispatched
            if should_run_simulation && !self.perf_skip_trail_prep && self.trail_opacity > 0.0 {
                self.frame_profiler.maybe_submit_encoder_segment(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    DispatchSegment::SimPreTrails,
                    "Update Encoder (Timed Segment)",
                );
            }

            // (c) Slope
            if run_slope && !self.perf_skip_slope {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass (SimPre Slope)"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.gamma_slope_pipeline);
                cpass.set_bind_group(0, bg_process, &[]);
                let groups_x = (GRID_DIM_U32 + SLOPE_WG_SIZE_X - 1) / SLOPE_WG_SIZE_X;
                let groups_y = (GRID_DIM_U32 + SLOPE_WG_SIZE_Y - 1) / SLOPE_WG_SIZE_Y;
                cpass.dispatch_workgroups(groups_x, groups_y, 1);
                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_SLOPE);
            }
            if run_slope && !self.perf_skip_slope {
                self.frame_profiler.maybe_submit_encoder_segment(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    DispatchSegment::SimPreSlope,
                    "Update Encoder (Timed Segment)",
                );
            }

            // (d) Draw prep
            if should_draw && !self.perf_skip_draw {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass (SimPre Draw Prep)"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.clear_visual_pipeline);
                cpass.set_bind_group(0, bg_process, &[]);
                let width_workgroups =
                    (self.surface_config.width + CLEAR_WG_SIZE_X - 1) / CLEAR_WG_SIZE_X;
                let height_workgroups =
                    (self.surface_config.height + CLEAR_WG_SIZE_Y - 1) / CLEAR_WG_SIZE_Y;
                cpass.dispatch_workgroups(width_workgroups, height_workgroups, 1);
                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_CLEAR_VISUAL);

                // Motion blur moved to Composite pass
                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_MOTION_BLUR);

                cpass.set_pipeline(&self.clear_agent_grid_pipeline);
                cpass.set_bind_group(0, bg_process, &[]);
                cpass.dispatch_workgroups(width_workgroups, height_workgroups, 1);
                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_CLEAR_AGENT_GRID);
            }
            if should_draw && !self.perf_skip_draw {
                self.frame_profiler.maybe_submit_encoder_segment(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    DispatchSegment::SimPreDraw,
                    "Update Encoder (Timed Segment)",
                );
            }

            // (e) Reproduction
            if should_run_simulation && !self.perf_skip_repro {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass (SimPre Reproduction)"),
                    timestamp_writes: None,
                });

                let reproduction_bg = if self.ping_pong {
                    &self.reproduction_bind_group_b
                } else {
                    &self.reproduction_bind_group_a
                };
                cpass.set_pipeline(&self.reproduction_pipeline);
                cpass.set_bind_group(0, reproduction_bg, &[]);
                cpass.dispatch_workgroups(
                    ((self.agent_count) + 255) / 256,
                    1,
                    1,
                );

                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_CLEAR_SPATIAL);
            }
            if should_run_simulation && !self.perf_skip_repro {
                self.frame_profiler.maybe_submit_encoder_segment(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    DispatchSegment::SimPreRepro,
                    "Update Encoder (Timed Segment)",
                );
            }
        }

        // Pass 2: main agent simulation.
        // This pass boundary is the synchronization point for reproduction -> simulation.
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass (After Reproduction)"),
                timestamp_writes: None,
            });

            if should_run_simulation {
                // Populate agent spatial grid with agent positions - workgroup_size(256)
                cpass.set_pipeline(&self.populate_agent_spatial_grid_pipeline);
                cpass.set_bind_group(0, bg_process, &[]);
                cpass.dispatch_workgroups(
                    ((self.agent_buffer_capacity as u32) + 255) / 256,
                    1,
                    1,
                );

                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_POPULATE_SPATIAL);

                // Clear fluid force vectors before agents write to it
                if self.fluid_enabled {
                    let fluid_workgroups = (self.fluid_grid_resolution + 15) / 16;
                    cpass.set_pipeline(&self.clear_fluid_force_vectors_pipeline);
                    cpass.set_bind_group(0, &self.fluid_bind_group_ab, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);
                }

                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_CLEAR_FLUID_FORCE_VECTORS);

                // Process all agents (sense, update, modify env, draw, spawn/death) - workgroup_size(256)
                // Agents will write propeller forces to fluid_force_vectors buffer with 100x boost
                cpass.set_pipeline(&self.process_pipeline);
                cpass.set_bind_group(0, bg_process, &[]);
                cpass.dispatch_workgroups(
                    ((self.agent_buffer_capacity as u32) + 255) / 256,
                    1,
                    1,
                );

                // Marker represents "after process_agents" only (microswim is timed separately).
                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_PROCESS_AGENTS);
            }

            if !should_run_simulation {
                // Keep timestamp progression stable when paused.
                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_POPULATE_SPATIAL);
                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_CLEAR_FLUID_FORCE_VECTORS);

                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_PROCESS_AGENTS);
            }
        }

        self.frame_profiler.maybe_submit_encoder_segment(
            &self.device,
            &self.queue,
            &mut encoder,
            DispatchSegment::SimMain,
            "Update Encoder (Timed Segment)",
        );

        // Start a new compute pass for microswimming to ensure a memory barrier for agents_out.
        // process_agents writes to agents_out, and microswim_agents reads/writes it.
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass (Microswim)"),
                timestamp_writes: None,
            });

            if should_run_simulation {
                // Skip dispatch entirely when disabled (saves a full-screen pass that otherwise
                // early-returns per-invocation in the shader).
                if self.microswim_enabled {
                    cpass.set_pipeline(&self.microswim_pipeline);
                    cpass.set_bind_group(0, bg_process, &[]);
                    cpass.dispatch_workgroups(
                        ((self.agent_buffer_capacity as u32) + 255) / 256,
                        1,
                        1,
                    );
                }
            }

            // Marker represents "after microswim".
            self.frame_profiler
                .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_MICROSWIM);
        }

        self.frame_profiler.maybe_submit_encoder_segment(
            &self.device,
            &self.queue,
            &mut encoder,
            DispatchSegment::Microswim,
            "Update Encoder (Timed Segment)",
        );

        // Start a new compute pass for drain_energy to ensure a memory barrier for agents_out.
        // process_agents writes to agents_out, and drain_energy reads from it.
        // In WebGPU, read-after-write in the same pass is not guaranteed to be visible.
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass (Drain Energy)"),
                timestamp_writes: None,
            });

            if should_run_simulation {
                // Drain energy from neighbors (vampire mouths) - workgroup_size(256)
                // Runs after process_agents so it can operate on agents_out without requiring
                // write access to agents_in (which would slow down all compute).
                cpass.set_pipeline(&self.drain_energy_pipeline);
                cpass.set_bind_group(0, bg_process, &[]);
                cpass.dispatch_workgroups(
                    ((self.agent_buffer_capacity as u32) + 255) / 256,
                    1,
                    1,
                );

                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_DRAIN_ENERGY);
            }

            if !should_run_simulation {
                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_SIM_AFTER_DRAIN_ENERGY);
            }
        }

        self.frame_profiler.maybe_submit_encoder_segment(
            &self.device,
            &self.queue,
            &mut encoder,
            DispatchSegment::Drain,
            "Update Encoder (Timed Segment)",
        );

        // Spike kill pass - runs right after drain energy
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass (Spike Kill)"),
                timestamp_writes: None,
            });

            if should_run_simulation {
                cpass.set_pipeline(&self.spike_kill_pipeline);
                cpass.set_bind_group(0, bg_process, &[]);
                cpass.dispatch_workgroups(
                    ((self.agent_buffer_capacity as u32) + 255) / 256,
                    1,
                    1,
                );
            }
        }

        self.frame_profiler
            .write_ts_encoder(&mut encoder, TS_UPDATE_AFTER_SIM);

        // End the first compute pass to ensure agent writes to fluid_force_vectors are visible
        // Start a new compute pass for fluid simulation
        {
            if self.fluid_enabled {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Fluid Compute Pass"),
                    timestamp_writes: None,
                });

                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_FLUID_PASS_START);

                if should_run_simulation {
                // Run fluid solver - forces already written by agents to force_vectors
                // Stable Fluids order: inject_test_force (combines) ? add_forces ? diffuse ? advect ? project
                {
                    let fluid_workgroups = (self.fluid_grid_resolution + 15) / 16;
                    let dye_workgroups = (self.env_grid_resolution + 15) / 16;
                    let bg_ab = &self.fluid_bind_group_ab;
                    let bg_ba = &self.fluid_bind_group_ba;

                    // Combine force_vectors (written in simulation pass) into fluid_forces.
                    // Inject a deterministic point-force "fumarole" for debugging/visual validation.
                    cpass.set_pipeline(&self.fluid_fumarole_pipeline);
                    cpass.set_bind_group(0, bg_ab, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_FLUID_AFTER_FUMAROLE_FORCE);

                    cpass.set_pipeline(&self.fluid_generate_forces_pipeline);
                    cpass.set_bind_group(0, bg_ab, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_FLUID_AFTER_GENERATE_FORCES);

                    // 1. Add forces to velocity (A -> B)
                    // Forces already written directly by agents with 100x boost + test force
                    cpass.set_pipeline(&self.fluid_add_forces_pipeline);
                    cpass.set_bind_group(0, bg_ab, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_FLUID_AFTER_ADD_FORCES);

                    // Clear forces after applying them (so agents start fresh next frame)
                    cpass.set_pipeline(&self.fluid_clear_forces_pipeline);
                    cpass.set_bind_group(0, bg_ba, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_FLUID_AFTER_CLEAR_FORCES);

                    // 2. Diffuse velocity (B -> A)
                    cpass.set_pipeline(&self.fluid_diffuse_velocity_pipeline);
                    cpass.set_bind_group(0, bg_ba, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_FLUID_AFTER_DIFFUSE_VELOCITY);

                    // 3. Advect velocity (A -> B)
                    cpass.set_pipeline(&self.fluid_advect_velocity_pipeline);
                    cpass.set_bind_group(0, bg_ab, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_FLUID_AFTER_ADVECT_VELOCITY);

                    // 4. Vorticity confinement (B -> A)
                    cpass.set_pipeline(&self.fluid_vorticity_confinement_pipeline);
                    cpass.set_bind_group(0, bg_ba, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_FLUID_AFTER_VORTICITY);

                    // 5. Compute divergence (reads velocity_a)
                    cpass.set_pipeline(&self.fluid_divergence_pipeline);
                    cpass.set_bind_group(0, bg_ab, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_FLUID_AFTER_DIVERGENCE);

                    // 6. Clear pressure buffers
                    cpass.set_pipeline(&self.fluid_clear_pressure_pipeline);
                    cpass.set_bind_group(0, bg_ba, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    cpass.set_pipeline(&self.fluid_clear_pressure_pipeline);
                    cpass.set_bind_group(0, bg_ab, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_FLUID_AFTER_CLEAR_PRESSURE);

                    // 7. Jacobi iterations for pressure
                    let jacobi_iters = self.fluid_jacobi_iters.clamp(1, 128);
                    for i in 0..jacobi_iters {
                        let bg = if (i & 1) == 0 { bg_ab } else { bg_ba };
                        cpass.set_pipeline(&self.fluid_jacobi_pressure_pipeline);
                        cpass.set_bind_group(0, bg, &[]);
                        cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);
                    }

                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_FLUID_AFTER_JACOBI);

                    // 8. Subtract pressure gradient (A->B)
                    cpass.set_pipeline(&self.fluid_subtract_gradient_pipeline);
                    cpass.set_bind_group(0, bg_ab, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_FLUID_AFTER_SUBTRACT_GRADIENT);

                    // 9. Enforce boundaries (B->A, final result in velocity_a)
                    cpass.set_pipeline(&self.fluid_enforce_boundaries_pipeline);
                    cpass.set_bind_group(0, bg_ba, &[]);
                    cpass.dispatch_workgroups(fluid_workgroups, fluid_workgroups, 1);

                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_FLUID_AFTER_BOUNDARIES);

                    // 10. Inject dye at propeller locations (A->B)
                    cpass.set_pipeline(&self.fluid_inject_dye_pipeline);
                    cpass.set_bind_group(0, bg_ab, &[]);
                    cpass.dispatch_workgroups(dye_workgroups, dye_workgroups, 1);

                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_FLUID_AFTER_INJECT_DYE);

                    // 11. Advect dye with velocity field (B->A, final result in dye_a)
                    cpass.set_pipeline(&self.fluid_advect_dye_pipeline);
                    cpass.set_bind_group(0, bg_ba, &[]);
                    cpass.dispatch_workgroups(dye_workgroups, dye_workgroups, 1);

                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_FLUID_AFTER_ADVECT_DYE);

                    // 11b. Inject fumarole alpha dye into the final dye buffer (dye_a).
                    // This keeps the point source visible even under strong local flow.
                    cpass.set_pipeline(&self.fluid_fumarole_dye_pipeline);
                    cpass.set_bind_group(0, bg_ba, &[]);
                    cpass.dispatch_workgroups(dye_workgroups, dye_workgroups, 1);

                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_FLUID_AFTER_FUMAROLE_DYE);

                    // 12. Advect agent trails with velocity field (trail_grid_inject -> trail_grid)
                    cpass.set_pipeline(&self.fluid_advect_trail_pipeline);
                    cpass.set_bind_group(0, bg_ab, &[]);
                    cpass.dispatch_workgroups(dye_workgroups, dye_workgroups, 1);

                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_FLUID_AFTER_ADVECT_TRAIL);
                }
            }
                // End fluid compute pass, start new pass for remaining simulation work
                drop(cpass);
            } else {
                // No-fluid fallback: keep dye + trail layers alive as simple diffusion fields.
                // This preserves signaling/visualization even when the full fluid solver is off.
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("No-Fluid Dye/Trails Pass"),
                    timestamp_writes: None,
                });

                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_FLUID_PASS_START);

                if should_run_simulation {
                    let dye_workgroups = (self.env_grid_resolution + 15) / 16;

                    // Diffuse dye once (A->B) then copy back (B->A) so the final stays in dye_a.
                    if !self.perf_skip_nofluid_dye {
                        cpass.set_pipeline(&self.fluid_diffuse_dye_no_fluid_pipeline);
                        cpass.set_bind_group(0, &self.fluid_bind_group_ab, &[]);
                        cpass.dispatch_workgroups(dye_workgroups, dye_workgroups, 1);

                        self.frame_profiler
                            .write_ts_compute_pass(&mut cpass, TS_NOFLUID_AFTER_DYE_DIFFUSE);

                        cpass.set_pipeline(&self.fluid_inject_dye_pipeline);
                        cpass.set_bind_group(0, &self.fluid_bind_group_ba, &[]);
                        cpass.dispatch_workgroups(dye_workgroups, dye_workgroups, 1);

                        self.frame_profiler
                            .write_ts_compute_pass(&mut cpass, TS_NOFLUID_AFTER_DYE_COPYBACK);
                    } else {
                        // Keep marker progression stable when skipping.
                        self.frame_profiler
                            .write_ts_compute_pass(&mut cpass, TS_NOFLUID_AFTER_DYE_DIFFUSE);
                        self.frame_profiler
                            .write_ts_compute_pass(&mut cpass, TS_NOFLUID_AFTER_DYE_COPYBACK);
                    }

                    // Commit prepared trails (trail_grid_inject -> trail_grid) without advection.
                    if !self.perf_skip_nofluid_trail {
                        cpass.set_pipeline(&self.fluid_copy_trail_no_fluid_pipeline);
                        cpass.set_bind_group(0, &self.fluid_bind_group_ab, &[]);
                        cpass.dispatch_workgroups(dye_workgroups, dye_workgroups, 1);

                        self.frame_profiler
                            .write_ts_compute_pass(&mut cpass, TS_NOFLUID_AFTER_TRAIL_COMMIT);
                    } else {
                        self.frame_profiler
                            .write_ts_compute_pass(&mut cpass, TS_NOFLUID_AFTER_TRAIL_COMMIT);
                    }
                }

                drop(cpass);
            }

            // Cheap tail-safety: clear the compaction destination buffer so any unwritten slots
            // remain fully zeroed (prevents stale/ghost agents from reappearing).
            if should_run_simulation {
                // When ping_pong=true, process runs B->A and compaction uses bg_swap A->B.
                // When ping_pong=false, process runs A->B and compaction uses bg_swap B->A.
                let compact_dst = if self.ping_pong {
                    &self.agents_buffer_b
                } else {
                    &self.agents_buffer_a
                };
                encoder.clear_buffer(compact_dst, 0, None);
            }

            self.frame_profiler
                .write_ts_encoder(&mut encoder, TS_UPDATE_AFTER_FLUID);

            self.frame_profiler.maybe_submit_encoder_segment(
                &self.device,
                &self.queue,
                &mut encoder,
                DispatchSegment::Fluid,
                "Update Encoder (Timed Segment)",
            );

            // Post-fluid Pass 1: compact/merge (and optional paused render-only process).
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Post-Fluid Compute Pass"),
                    timestamp_writes: None,
                });

                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_POST_PASS_START);

                if should_run_simulation {
                    // Compact alive agents - workgroup_size(64), scan ENTIRE buffer
                    cpass.set_pipeline(&self.compact_pipeline);
                    cpass.set_bind_group(0, bg_swap, &[]);

                    cpass.dispatch_workgroups((self.agent_buffer_capacity as u32 + 63) / 64, 1, 1);

                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_POST_AFTER_COMPACT);

                    // Process CPU spawn requests - workgroup_size(64)
                    if cpu_spawn_count > 0 {
                        cpass.set_pipeline(&self.cpu_spawn_pipeline);
                        cpass.set_bind_group(0, bg_swap, &[]);
                        cpass.dispatch_workgroups((MAX_SPAWN_REQUESTS as u32) / 64, 1, 1);
                    }

                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_POST_AFTER_CPU_SPAWN);

                    // Merge spawned agents - workgroup_size(64), max MAX_SPAWN_REQUESTS spawns
                    cpass.set_pipeline(&self.merge_pipeline);
                    cpass.set_bind_group(0, bg_swap, &[]);
                    cpass.dispatch_workgroups((MAX_SPAWN_REQUESTS as u32) / 64, 1, 1);

                    self.frame_profiler
                        .write_ts_compute_pass(&mut cpass, TS_POST_AFTER_MERGE);
                } else if should_draw && self.agent_count > 0 {
                    // When paused or no living agents: run process pipeline for rendering only (dt=0, probabilities=0)
                    // but skip compaction, spawning, and buffer swapping

                    cpass.set_pipeline(&self.process_pipeline);
                    cpass.set_bind_group(0, bg_process, &[]);
                    cpass.dispatch_workgroups((self.agent_count + 255) / 256, 1, 1);

                    self.frame_profiler.write_ts_compute_pass(
                        &mut cpass,
                        TS_POST_AFTER_PROCESS_PAUSED_RENDER_ONLY,
                    );
                }

                if should_run_simulation {
                    self.frame_profiler.write_ts_compute_pass(
                        &mut cpass,
                        TS_POST_AFTER_PROCESS_PAUSED_RENDER_ONLY,
                    );
                }
            }

            // Post-fluid Pass 1b: write init-dead indirect args.
            // This pass boundary ensures merge/compact counters are visible when we compute args.
            if should_run_simulation {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Post-Fluid Compute Pass (Init-Dead Args)"),
                    timestamp_writes: None,
                });

                // Compute indirect dispatch args for init-dead (based on alive_total).
                // NOTE: Must be in a separate compute pass from the indirect dispatch because
                // the args buffer is STORAGE here and then INDIRECT for dispatch.
                cpass.set_pipeline(&self.write_init_dead_dispatch_args_pipeline);
                cpass.set_bind_group(0, &self.init_dead_writer_bind_group, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }

            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Post-Fluid Compute Pass 2"),
                timestamp_writes: None,
            });

            if should_run_simulation {
                // Sanitize unused agent slots - workgroup_size(256), dispatched only for dead tail.
                cpass.set_pipeline(&self.initialize_dead_pipeline);
                cpass.set_bind_group(0, bg_swap, &[]);
                cpass.dispatch_workgroups_indirect(&self.init_dead_dispatch_args, 0);

                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_POST_AFTER_INIT_DEAD);

                // Reset spawn counter - workgroup_size(1)
                cpass.set_pipeline(&self.finalize_merge_pipeline);
                cpass.set_bind_group(0, bg_swap, &[]);
                cpass.dispatch_workgroups(1, 1, 1);

                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_POST_AFTER_FINALIZE_MERGE);
            }

            // Render all agents to agent_grid when drawing
            if should_draw {
                cpass.set_pipeline(&self.render_agents_pipeline);
                cpass.set_bind_group(0, bg_process, &[]);
                cpass.dispatch_workgroups((self.agent_count + 255) / 256, 1, 1);
            }

            self.frame_profiler
                .write_ts_compute_pass(&mut cpass, TS_POST_AFTER_RENDER_AGENTS);

            // Draw the selected agent into the inspector preview box (top of inspector).
            // Clear the inspector area first to ensure a clean background.
            if should_draw && self.selected_agent_index.is_some() {
                cpass.set_pipeline(&self.clear_inspector_preview_pipeline);
                cpass.set_bind_group(0, bg_process, &[]);
                let preview_groups_x = (300 + 15) / 16;
                let preview_groups_y = (300 + 15) / 16;
                cpass.dispatch_workgroups(preview_groups_x, preview_groups_y, 1);

                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_POST_AFTER_INSPECTOR_CLEAR);

                cpass.set_pipeline(&self.draw_inspector_agent_pipeline);
                cpass.set_bind_group(0, bg_process, &[]);
                // Tiled preview draw: shader uses workgroup_size(1,1) and clips each tile.
                // Keep these constants in sync with shaders/render.wgsl.
                let tile_size: u32 = 32;
                let tiles_x = (300 + tile_size - 1) / tile_size;
                let tiles_y = (300 + tile_size - 1) / tile_size;
                cpass.dispatch_workgroups(tiles_x, tiles_y, 1);

                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_POST_AFTER_INSPECTOR_DRAW);
            }

            if !(should_draw && self.selected_agent_index.is_some()) {
                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_POST_AFTER_INSPECTOR_CLEAR);
                self.frame_profiler
                    .write_ts_compute_pass(&mut cpass, TS_POST_AFTER_INSPECTOR_DRAW);
            }
        }

        self.frame_profiler
            .write_ts_encoder(&mut encoder, TS_UPDATE_AFTER_POST);

        self.frame_profiler.maybe_submit_encoder_segment(
            &self.device,
            &self.queue,
            &mut encoder,
            DispatchSegment::Post,
            "Update Encoder (Timed Segment)",
        );

        // End compute pass to ensure agent_grid writes are visible before composite reads
        // Start new compute pass for compositing
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Composite Compute Pass"),
                timestamp_writes: None,
            });

            self.frame_profiler
                .write_ts_compute_pass(&mut cpass, TS_COMPOSITE_PASS_START);

            // Composite agent_grid onto visual_grid when drawing
            if should_draw {
                cpass.set_pipeline(&self.composite_agents_pipeline);
                cpass.set_bind_group(0, &self.composite_bind_group, &[]);
                let width_workgroups =
                    (self.surface_config.width + CLEAR_WG_SIZE_X - 1) / CLEAR_WG_SIZE_X;
                let height_workgroups =
                    (self.surface_config.height + CLEAR_WG_SIZE_Y - 1) / CLEAR_WG_SIZE_Y;
                cpass.dispatch_workgroups(width_workgroups, height_workgroups, 1);
            }
        }

        self.frame_profiler
            .write_ts_encoder(&mut encoder, TS_UPDATE_AFTER_COMPOSITE);

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

            self.frame_profiler
                .write_ts_encoder(&mut encoder, TS_UPDATE_AFTER_COPY);
        }

        self.frame_profiler.maybe_submit_encoder_segment(
            &self.device,
            &self.queue,
            &mut encoder,
            DispatchSegment::CompositeCopy,
            "Update Encoder (Timed Segment)",
        );

        // Debug: when viewing energy trails, occasionally read back trail_grid.w stats.
        // This confirms whether the energy channel is actually accumulating (vs. visualization issues).
        let mut did_trail_energy_readback = false;
        if self.trail_show && self.trail_show_energy && self.epoch >= self.trail_energy_debug_next_epoch {
            let trail_buffer_size = (GRID_CELL_COUNT * std::mem::size_of::<[f32; 4]>()) as u64;
            encoder.copy_buffer_to_buffer(
                &self.trail_grid,
                0,
                &self.trail_debug_readback,
                0,
                trail_buffer_size,
            );
            did_trail_energy_readback = true;
            // Throttle to avoid stalling the GPU too often.
            self.trail_energy_debug_next_epoch = self.epoch.saturating_add(2000);
        }

        self.process_completed_alive_readbacks();
        let slot = self.ensure_alive_slot_ready();
        let alive_buffer = self.copy_state_to_staging(&mut encoder, slot);

        self.frame_profiler
            .write_ts_encoder(&mut encoder, TS_UPDATE_END);

        if !should_draw {
            // If we aren't going to render/egui this frame, resolve in the update encoder.
            self.frame_profiler.resolve_and_copy(&mut encoder);
        }

        self.frame_profiler.submit_cmd_buffer(
            &self.device,
            &self.queue,
            encoder.finish(),
            DispatchSegment::UpdateTail,
        );

        if !should_draw {
            self.frame_profiler
                .readback_and_print(&self.device);

            if self.frame_profiler.bench_exit_requested() {
                std::process::exit(0);
            }
        }

        if did_trail_energy_readback {
            let slice = self.trail_debug_readback.slice(..);
            slice.map_async(wgpu::MapMode::Read, |result| {
                result.unwrap();
            });
            self.device.poll(wgpu::Maintain::Wait);
            {
                let view = slice.get_mapped_range();
                let floats: &[f32] = bytemuck::cast_slice(&view);
                let mut max_w: f32 = 0.0;
                let mut sum_w: f64 = 0.0;
                let mut nonzero: u32 = 0;
                let mut nonfinite: u32 = 0;
                let mut nonfinite_x: u32 = 0;
                let mut nonfinite_y: u32 = 0;
                let mut nonfinite_z: u32 = 0;
                let cell_count = (floats.len() / 4).max(1);

                for i in (3..floats.len()).step_by(4) {
                    let w = floats[i];
                    if !w.is_finite() {
                        nonfinite += 1;
                        continue;
                    }
                    if w > max_w {
                        max_w = w;
                    }
                    if w > 0.0 {
                        nonzero += 1;
                    }
                    sum_w += w as f64;
                }

                for i in (0..floats.len()).step_by(4) {
                    let x = floats[i + 0];
                    let y = floats[i + 1];
                    let z = floats[i + 2];
                    if !x.is_finite() {
                        nonfinite_x += 1;
                    }
                    if !y.is_finite() {
                        nonfinite_y += 1;
                    }
                    if !z.is_finite() {
                        nonfinite_z += 1;
                    }
                }

                let mean_w = (sum_w / cell_count as f64) as f32;
                println!(
                    "Trail energy stats: max_w={:.3e} mean_w={:.3e} nonzero_cells={} nonfinite_w={} nonfinite_xyz=({}, {}, {})",
                    max_w, mean_w, nonzero, nonfinite, nonfinite_x, nonfinite_y, nonfinite_z
                );

                if floats.len() >= 16 {
                    let s0 = [floats[0], floats[1], floats[2], floats[3]];
                    let s1 = [floats[4], floats[5], floats[6], floats[7]];
                    let b0 = [s0[0].to_bits(), s0[1].to_bits(), s0[2].to_bits(), s0[3].to_bits()];
                    let b1 = [s1[0].to_bits(), s1[1].to_bits(), s1[2].to_bits(), s1[3].to_bits()];
                    println!(
                        "Trail sample[0]={:?} bits={:08x?}  sample[1]={:?} bits={:08x?}",
                        s0, b0, s1, b1
                    );
                }
            }
            self.trail_debug_readback.unmap();
        }

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

        // Diffusion is tied to simulation running; slope rebuild is allowed while paused.
        if should_run_simulation {
            self.diffusion_counter = (self.diffusion_counter + 1) % diffusion_interval;
        }
        self.slope_counter = (self.slope_counter + 1) % slope_interval;

        self.kickoff_alive_readback(slot, alive_buffer);

        self.perform_optional_readbacks(should_draw);

        // Keep ping-pong orientation stable: after process (in->out),
        // compaction wrote results back to the original input buffer.
        // Therefore, we do NOT toggle ping-pong here.

        let cpu_update_ms = cpu_update_start.elapsed().as_secs_f64() * 1000.0;
        self.frame_profiler.set_cpu_update_ms(cpu_update_ms);
    }

    fn render(
        &mut self,
        clipped_primitives: &[egui::ClippedPrimitive],
        textures_delta: egui::TexturesDelta,
        screen_descriptor: ScreenDescriptor,
    ) -> Result<(), wgpu::SurfaceError> {
        let cpu_render_start = std::time::Instant::now();

        let skip_egui = clipped_primitives.is_empty()
            && textures_delta.set.is_empty()
            && textures_delta.free.is_empty();

        // Update FPS counter
        self.frame_count += 1;
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last_fps_update).as_secs_f32();
        if elapsed >= 1.0 {
            let fps = self.frame_count as f32 / elapsed;
            self.window
                .set_title(&format!("{} v{} - {:.1} FPS", APP_NAME, APP_VERSION, fps));
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

        self.frame_profiler
            .write_ts_encoder(&mut encoder, TS_RENDER_ENC_START);

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

            self.frame_profiler
                .write_ts_render_pass(&mut rpass, TS_RENDER_MAIN_START);

            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.render_bind_group, &[]);
            rpass.draw(0..6, 0..1);

            self.frame_profiler
                .write_ts_render_pass(&mut rpass, TS_RENDER_MAIN_END);
        }

        // Render inspector overlay (fragment shader) on top of the main pass
        if self.sim_params_cpu.selected_agent_index != 0xFFFFFFFFu32 {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Inspector Overlay Render Pass"),
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

            self.frame_profiler
                .write_ts_render_pass(&mut rpass, TS_RENDER_OVERLAY_START);

            rpass.set_pipeline(&self.inspector_overlay_pipeline);
            rpass.set_bind_group(0, &self.inspector_overlay_bind_group, &[]);
            rpass.draw(0..6, 0..1);

            self.frame_profiler
                .write_ts_render_pass(&mut rpass, TS_RENDER_OVERLAY_END);
        }

        if self.sim_params_cpu.selected_agent_index == 0xFFFFFFFFu32 {
            // Keep markers stable when overlay pass is skipped.
            self.frame_profiler
                .write_ts_encoder(&mut encoder, TS_RENDER_OVERLAY_START);
            self.frame_profiler
                .write_ts_encoder(&mut encoder, TS_RENDER_OVERLAY_END);
        }

        self.frame_profiler
            .write_ts_encoder(&mut encoder, TS_RENDER_ENC_END);

        // If "Show UI" is disabled, capture BEFORE egui draws into the swapchain.
        // This keeps the on-screen UI visible while excluding it from the recording.
        if !skip_egui && self.recording && !self.recording_show_ui {
            self.recording_schedule_readback(&output.texture, &mut encoder, now);
        }

        if skip_egui {
            // Keep egui markers stable when egui is skipped.
            self.frame_profiler
                .write_ts_encoder(&mut encoder, TS_EGUI_ENC_START);
            self.frame_profiler
                .write_ts_encoder(&mut encoder, TS_EGUI_PASS_START);
            self.frame_profiler
                .write_ts_encoder(&mut encoder, TS_EGUI_PASS_END);
            self.frame_profiler
                .write_ts_encoder(&mut encoder, TS_EGUI_ENC_END);

            // Resolve/copy timestamps at the end of the frame work.
            self.frame_profiler.resolve_and_copy(&mut encoder);

            // Schedule recording readback from the swapchain (no egui this frame).
            if self.recording {
                self.recording_schedule_readback(&output.texture, &mut encoder, now);
            }

            // Submit simulation rendering and present.
            self.frame_profiler.submit_cmd_buffer(
                &self.device,
                &self.queue,
                encoder.finish(),
                DispatchSegment::Render,
            );

            if self.recording {
                self.recording_begin_pending_maps();
            }
            output.present();

            if self.recording {
                self.recording_drain_ready_frames();
                if !self.recording && self.recording_pipe.is_some() {
                    let _ = self.save_recording();
                }
            }

            if self.perf_force_gpu_sync {
                self.device.poll(wgpu::Maintain::Wait);
            }

            let cpu_render_ms = cpu_render_start.elapsed().as_secs_f64() * 1000.0;
            self.frame_profiler.set_cpu_render_ms(cpu_render_ms);
            self.frame_profiler.set_cpu_egui_ms(0.0);
            self.frame_profiler.readback_and_print(&self.device);

            if self.frame_profiler.bench_exit_requested() {
                std::process::exit(0);
            }
            return Ok(());
        }

        // Submit simulation rendering
        self.frame_profiler.submit_cmd_buffer(
            &self.device,
            &self.queue,
            encoder.finish(),
            DispatchSegment::Render,
        );

        let cpu_render_ms = cpu_render_start.elapsed().as_secs_f64() * 1000.0;
        self.frame_profiler.set_cpu_render_ms(cpu_render_ms);

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

        self.frame_profiler
            .write_ts_encoder(&mut encoder, TS_EGUI_ENC_START);

        let cpu_egui_start = std::time::Instant::now();

        // Update buffers
        for (id, image_delta) in &textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, image_delta);
        }

        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            clipped_primitives,
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

            self.frame_profiler
                .write_ts_render_pass(&mut rpass, TS_EGUI_PASS_START);

            // SAFETY: The render pass lives long enough for this call.
            // The lifetime requirement is overly restrictive in egui-wgpu 0.29.
            let rpass_static: &mut wgpu::RenderPass<'static> =
                unsafe { std::mem::transmute(&mut rpass) };
            self.egui_renderer
                .render(rpass_static, clipped_primitives, &screen_descriptor);

            self.frame_profiler
                .write_ts_render_pass(&mut rpass, TS_EGUI_PASS_END);
        }

        // Schedule recording readback after egui has been drawn into the swapchain.
        if self.recording && self.recording_show_ui {
            self.recording_schedule_readback(&output.texture, &mut encoder, now);
        }

        self.frame_profiler
            .write_ts_encoder(&mut encoder, TS_EGUI_ENC_END);
        self.frame_profiler.resolve_and_copy(&mut encoder);

        for id in &textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.frame_profiler.submit_cmd_buffer(
            &self.device,
            &self.queue,
            encoder.finish(),
            DispatchSegment::Egui,
        );

        if self.recording {
            self.recording_begin_pending_maps();
        }
        output.present();

        if self.perf_force_gpu_sync {
            self.device.poll(wgpu::Maintain::Wait);
        }

        if self.recording {
            self.recording_drain_ready_frames();
            if !self.recording && self.recording_pipe.is_some() {
                let _ = self.save_recording();
            }
        }

        let cpu_egui_ms = cpu_egui_start.elapsed().as_secs_f64() * 1000.0;
        self.frame_profiler.set_cpu_egui_ms(cpu_egui_ms);
        self.frame_profiler.readback_and_print(&self.device);

        if self.frame_profiler.bench_exit_requested() {
            std::process::exit(0);
        }

        if self.frame_profiler.bench_exit_requested() {
            std::process::exit(0);
        }

        Ok(())
    }

    fn render_ui_only(
        &mut self,
        clipped_primitives: Vec<egui::ClippedPrimitive>,
        textures_delta: egui::TexturesDelta,
        screen_descriptor: ScreenDescriptor,
    ) -> Result<(), wgpu::SurfaceError> {
        let cpu_egui_start = std::time::Instant::now();

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
                "{} v{}{} - {:.1} FPS",
                APP_NAME,
                APP_VERSION,
                speed,
                fps
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

        self.frame_profiler
            .write_ts_encoder(&mut encoder, TS_EGUI_ENC_START);

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

            self.frame_profiler
                .write_ts_render_pass(&mut rpass, TS_EGUI_PASS_START);

            // SAFETY: The render pass lives long enough for this call.
            let rpass_static: &mut wgpu::RenderPass<'static> =
                unsafe { std::mem::transmute(&mut rpass) };
            self.egui_renderer
                .render(rpass_static, &clipped_primitives, &screen_descriptor);

            self.frame_profiler
                .write_ts_render_pass(&mut rpass, TS_EGUI_PASS_END);
        }

        self.frame_profiler
            .write_ts_encoder(&mut encoder, TS_EGUI_ENC_END);
        self.frame_profiler.resolve_and_copy(&mut encoder);

        for id in &textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.frame_profiler.submit_cmd_buffer(
            &self.device,
            &self.queue,
            encoder.finish(),
            DispatchSegment::Egui,
        );
        output.present();

        if self.perf_force_gpu_sync {
            self.device.poll(wgpu::Maintain::Wait);
        }

        let cpu_egui_ms = cpu_egui_start.elapsed().as_secs_f64() * 1000.0;
        self.frame_profiler.set_cpu_render_ms(0.0);
        self.frame_profiler.set_cpu_egui_ms(cpu_egui_ms);
        self.frame_profiler.readback_and_print(&self.device);

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
        let sim_size = self.sim_size;
        let half_view_x = sim_size / (2.0 * self.camera_zoom);
        let half_view_y = half_view_x * aspect;

        let mut world_x = self.camera_pan[0] + (norm_x - 0.5) * 2.0 * half_view_x;
        let mut world_y = self.camera_pan[1] - (norm_y - 0.5) * 2.0 * half_view_y; // Y inverted

        // Wrap to world bounds
        world_x = world_x.rem_euclid(sim_size);
        world_y = world_y.rem_euclid(sim_size);

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
                if d > sim_size * 0.5 {
                    d -= sim_size;
                } else if d < -sim_size * 0.5 {
                    d += sim_size;
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

                // Reset readback state so the UI updates quickly.
                for inflight in &mut self.selected_agent_readback_inflight {
                    *inflight = false;
                }
                self.selected_agent_readback_last_request = std::time::Instant::now()
                    - std::time::Duration::from_secs(1);
                for pending in &self.selected_agent_readback_pending {
                    if let Ok(mut guard) = pending.lock() {
                        *guard = None;
                    }
                }
            } else {
                self.selected_agent_index = None;
                self.selected_agent_data = None;

                for inflight in &mut self.selected_agent_readback_inflight {
                    *inflight = false;
                }
                for pending in &self.selected_agent_readback_pending {
                    if let Ok(mut guard) = pending.lock() {
                        *guard = None;
                    }
                }
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

    fn scan_population_for_organ45(&mut self) {
        self.organ45_last_scan_error = None;

        let current_buffer = if self.ping_pong {
            &self.agents_buffer_b
        } else {
            &self.agents_buffer_a
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Organ45 Scan Encoder"),
            });

        encoder.copy_buffer_to_buffer(
            current_buffer,
            0,
            &self.agents_readback,
            0,
            (std::mem::size_of::<Agent>() * self.agent_buffer_capacity) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        let slice = self.agents_readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);

        let mut total_with = 0u32;
        let mut alive_with = 0u32;

        {
            let data = slice.get_mapped_range();
            let agents: &[Agent] = bytemuck::cast_slice(&data);

            let take_len = (self.agent_count as usize).min(agents.len());
            for agent in agents.iter().take(take_len) {
                let body_len = (agent.body_count as usize).min(MAX_BODY_PARTS);
                let mut has_organ45 = false;
                for part in agent.body.iter().take(body_len) {
                    if part.base_type() == 45 {
                        has_organ45 = true;
                        break;
                    }
                }
                if has_organ45 {
                    total_with += 1;
                    if agent.alive != 0 {
                        alive_with += 1;
                    }
                }
            }
        }

        self.agents_readback.unmap();
        self.organ45_total_with = total_with;
        self.organ45_alive_with = alive_with;
        self.organ45_last_scan = Some(std::time::Instant::now());
    }

    fn spawn_agent_at_cursor(&mut self, screen_pos: [f32; 2]) {
        // Convert screen coordinates to world coordinates (same logic as select_agent_at_screen_pos)
        let norm_x = screen_pos[0] / self.surface_config.width as f32;
        let norm_y = screen_pos[1] / self.surface_config.height as f32;

        let aspect = if self.surface_config.width > 0 {
            self.surface_config.height as f32 / self.surface_config.width as f32
        } else {
            1.0
        };
        let sim_size = self.sim_size;
        let half_view_x = sim_size / (2.0 * self.camera_zoom);
        let half_view_y = half_view_x * aspect;

        let mut world_x = self.camera_pan[0] + (norm_x - 0.5) * 2.0 * half_view_x;
        let mut world_y = self.camera_pan[1] - (norm_y - 0.5) * 2.0 * half_view_y; // Y inverted

        // Wrap to world bounds
        world_x = world_x.rem_euclid(sim_size);
        world_y = world_y.rem_euclid(sim_size);

        if let Some(template_genome) = self.spawn_template_genome {
            // Generate random seeds
            self.rng_state = self.rng_state
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            let seed = (self.rng_state >> 32) as u32;

            self.rng_state = self.rng_state
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            let genome_seed = (self.rng_state >> 32) as u32;

            self.rng_state = self.rng_state
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            let rotation = ((self.rng_state >> 32) as f32 / u32::MAX as f32) * std::f32::consts::TAU;

            let (genome_override_len, genome_override_offset, genome_override_packed) =
                genome_pack_ascii_words(&template_genome);

            let request = SpawnRequest {
                seed,
                genome_seed,
                flags: 1, // genome override
                _pad_seed: 0,
                position: [world_x, world_y],
                energy: 10.0,
                rotation,
                genome_override_len,
                genome_override_offset,
                genome_override_packed,
                _pad_genome: [0u32; 2],
            };

            self.cpu_spawn_queue.push(request);
            println!("? Spawned agent at ({:.1}, {:.1})", world_x, world_y);
        }
    }

    fn save_snapshot_to_file(&mut self, path: &Path) -> anyhow::Result<()> {
        // Copy grids and agents from GPU to readback buffers
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Snapshot Readback Encoder"),
        });

        let chem_grid_size_bytes =
            (self.env_grid_cell_count * std::mem::size_of::<[f32; 4]>()) as u64;
        let gamma_grid_size_bytes = (self.env_grid_cell_count * std::mem::size_of::<f32>()) as u64;

        encoder.copy_buffer_to_buffer(&self.chem_grid, 0, &self.chem_grid_readback, 0, chem_grid_size_bytes);
        encoder.copy_buffer_to_buffer(&self.gamma_grid, 0, &self.gamma_grid_readback, 0, gamma_grid_size_bytes);

        // Copy agents from GPU.
        // IMPORTANT: The authoritative, most-recent agent data lives in the ping-pong-selected
        // "current" buffer. Saving only from buffer A can capture stale/cleared data when the
        // latest results are in buffer B.
        let agents_size_bytes = (std::mem::size_of::<Agent>() * self.agent_buffer_capacity) as u64;
        let source_buffer = if self.ping_pong {
            &self.agents_buffer_b
        } else {
            &self.agents_buffer_a
        };
        encoder.copy_buffer_to_buffer(source_buffer, 0, &self.agents_readback, 0, agents_size_bytes);

        self.queue.submit(Some(encoder.finish()));

        // Map and read grids
        let chem_slice = self.chem_grid_readback.slice(..);
        let gamma_slice = self.gamma_grid_readback.slice(..);
        let agents_slice = self.agents_readback.slice(..);

        chem_slice.map_async(wgpu::MapMode::Read, |_| {});
        gamma_slice.map_async(wgpu::MapMode::Read, |_| {});
        agents_slice.map_async(wgpu::MapMode::Read, |_| {});

        self.device.poll(wgpu::Maintain::Wait);

        let (alpha_grid, beta_grid, gamma_grid, agents_gpu) = {
            let chem_data = chem_slice.get_mapped_range();
            let gamma_data = gamma_slice.get_mapped_range();
            let agents_data = agents_slice.get_mapped_range();

            let chem: Vec<[f32; 4]> = bytemuck::cast_slice(&chem_data).to_vec();
            let mut alpha: Vec<f32> = Vec::with_capacity(chem.len());
            let mut beta: Vec<f32> = Vec::with_capacity(chem.len());
            for v in &chem {
                alpha.push(v[0]);
                beta.push(v[1]);
                // v[2] and v[3] are rain_alpha and rain_beta, not saved in snapshot
            }
            let gamma: Vec<f32> = bytemuck::cast_slice(&gamma_data).to_vec();
            let agents: Vec<Agent> = bytemuck::cast_slice(&agents_data).to_vec();

            (alpha, beta, gamma, agents)
        };

        self.chem_grid_readback.unmap();
        self.gamma_grid_readback.unmap();
        self.agents_readback.unmap();

        // Filter only living agents before creating snapshot
        let living_agents: Vec<Agent> = agents_gpu
            .into_iter()
            .filter(|a| a.alive != 0)
            .collect();

        println!("Saving snapshot with {} living agents", living_agents.len());

        // Autosave robustness: never overwrite the autosave file with an empty snapshot.
        // A transient 0-living capture (e.g. wrong buffer sampled or counters temporarily stale)
        // can permanently break the next startup if it replaces the last good autosave.
        if living_agents.is_empty() {
            if path
                .file_name()
                .and_then(|s| s.to_str())
                .map(|name| name.eq_ignore_ascii_case(AUTO_SNAPSHOT_FILE_NAME))
                .unwrap_or(false)
            {
                println!(
                    "G�� Skipping autosave overwrite: captured 0 living agents (keeping previous autosave)"
                );
                return Ok(());
            }
        }

        // Create snapshot from living agents with current settings
        let current_settings = self.current_settings();
        let snapshot = SimulationSnapshot::new(
            self.epoch,
            &living_agents,
            current_settings,
            self.run_name.clone(),
            None,
            self.env_grid_resolution,
            self.fluid_grid_resolution,
            self.spatial_grid_resolution,
        );

        // Save to PNG
        save_simulation_snapshot(path, &alpha_grid, &beta_grid, &gamma_grid, &snapshot)?;

        Ok(())
    }

    fn capture_screenshot(&mut self) -> anyhow::Result<()> {
        // Capture the current visual_grid texture as-is (no tiling, just the current view)
        let texture_size = self.visual_texture.size();
        let width = texture_size.width;
        let height = texture_size.height;

        // Rgba32Float = 16 bytes per pixel
        let bytes_per_pixel = 16u32;
        // wgpu requires bytes_per_row to be a multiple of 256
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = ((unpadded_bytes_per_row + align - 1) / align) * align;

        let buffer_size = (padded_bytes_per_row * height) as u64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Screenshot Output Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Screenshot Encoder"),
        });

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.visual_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            texture_size,
        );

        self.queue.submit(Some(encoder.finish()));

        // Map the buffer and read the data
        let buffer_slice = output_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()??;

        let data = buffer_slice.get_mapped_range();

        // Convert Rgba32Float to RGBA8 by unpacking padded rows
        let mut rgba_data: Vec<u8> = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            let row_start = (y * padded_bytes_per_row) as usize;
            let row_data = &data[row_start..row_start + (width * bytes_per_pixel) as usize];

            // Rgba32Float: 4 f32s per pixel (16 bytes)
            for pixel_bytes in row_data.chunks_exact(16) {
                let r = f32::from_le_bytes([pixel_bytes[0], pixel_bytes[1], pixel_bytes[2], pixel_bytes[3]]);
                let g = f32::from_le_bytes([pixel_bytes[4], pixel_bytes[5], pixel_bytes[6], pixel_bytes[7]]);
                let b = f32::from_le_bytes([pixel_bytes[8], pixel_bytes[9], pixel_bytes[10], pixel_bytes[11]]);
                let a = f32::from_le_bytes([pixel_bytes[12], pixel_bytes[13], pixel_bytes[14], pixel_bytes[15]]);

                // Apply sRGB gamma correction (linear → sRGB) to match screen appearance
                let to_srgb = |linear: f32| -> u8 {
                    let linear = linear.clamp(0.0, 1.0);
                    let srgb = if linear <= 0.0031308 {
                        linear * 12.92
                    } else {
                        1.055 * linear.powf(1.0 / 2.4) - 0.055
                    };
                    (srgb * 255.0) as u8
                };

                rgba_data.push(to_srgb(r));
                rgba_data.push(to_srgb(g));
                rgba_data.push(to_srgb(b));
                rgba_data.push((a.clamp(0.0, 1.0) * 255.0) as u8); // Alpha stays linear
            }
        }

        drop(data);
        output_buffer.unmap();

        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Create screenshots directory if it doesn't exist
        std::fs::create_dir_all("screenshots")?;

        let filename = format!("screenshots/screenshot_{}x{}_{}_{}.jpg", width, height, self.run_name, timestamp);

        let img = image::RgbaImage::from_raw(width, height, rgba_data)
            .ok_or_else(|| anyhow::anyhow!("Failed to create image from buffer"))?;

        // Convert to RGB and save as JPEG with 90% quality
        let rgb_img = image::DynamicImage::ImageRgba8(img).to_rgb8();
        let mut output = std::fs::File::create(&filename)?;
        let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut output, 90);
        encoder.encode(
            rgb_img.as_raw(),
            rgb_img.width(),
            rgb_img.height(),
            image::ColorType::Rgb8,
        )?;

        println!("📸 Screenshot saved: {}", filename);
        Ok(())
    }

    fn save_recording(&mut self) -> anyhow::Result<()> {
        // Stop ffmpeg pipe (if any). The output file is finalized when stdin closes.
        self.recording = false;
        self.recording_start_time = None;
        self.recording_last_frame_time = None;

        self.recording_readbacks.clear();
        self.recording_readback_index = 0;

        if let Some(mut pipe) = self.recording_pipe.take() {
            // Drop stdin to signal EOF.
            drop(pipe.stdin);

            let output_path = pipe.output_path.clone();
            std::thread::spawn(move || {
                let _ = pipe.child.wait();
                println!("🎬 Recording saved: {}", output_path.display());
            });
        }

        Ok(())
    }

    fn start_recording(&mut self) -> anyhow::Result<()> {
        if self.recording_pipe.is_some() {
            return Ok(());
        }

        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        std::fs::create_dir_all("recordings")?;

        let requested_width = self.recording_width.max(16);
        let requested_height = self.recording_height.max(16);
        let fps = self.recording_fps.max(1);

        let surface_width = self.surface_config.width;
        let surface_height = self.surface_config.height;
        anyhow::ensure!(surface_width > 0 && surface_height > 0, "Cannot start recording with a zero-sized surface");
        let capture_width = requested_width.min(surface_width).max(1);
        let capture_height = requested_height.min(surface_height).max(1);

        let swap_is_bgra = matches!(
            self.surface_config.format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );
        let in_pix_fmt: &'static str = if swap_is_bgra { "bgra" } else { "rgba" };

        let (output_path, mut cmd) = match self.recording_format {
            RecordingFormat::MP4 => {
                let path = PathBuf::from(format!(
                    "recordings/recording_{}x{}_{}_{}.mp4",
                    capture_width, capture_height, self.run_name, timestamp
                ));
                let mut cmd = Command::new("ffmpeg");
                cmd.arg("-y")
                    .arg("-f")
                    .arg("rawvideo")
                    .arg("-pix_fmt")
                    .arg(in_pix_fmt)
                    .arg("-s")
                    .arg(format!("{}x{}", capture_width, capture_height))
                    .arg("-r")
                    .arg(format!("{}", fps))
                    .arg("-i")
                    .arg("-")
                    .arg("-an")
                    .arg("-c:v")
                    .arg("libx264")
                    .arg("-preset")
                    .arg("veryfast")
                    .arg("-crf")
                    .arg("18")
                    .arg("-pix_fmt")
                    .arg("yuv420p");
                (path, cmd)
            }
            RecordingFormat::GIF => {
                let path = PathBuf::from(format!(
                    "recordings/recording_{}x{}_{}_{}.gif",
                    capture_width, capture_height, self.run_name, timestamp
                ));
                let mut cmd = Command::new("ffmpeg");
                cmd.arg("-y")
                    .arg("-f")
                    .arg("rawvideo")
                    .arg("-pix_fmt")
                    .arg(in_pix_fmt)
                    .arg("-s")
                    .arg(format!("{}x{}", capture_width, capture_height))
                    .arg("-r")
                    .arg(format!("{}", fps))
                    .arg("-i")
                    .arg("-")
                    .arg("-vf")
                    .arg("split[s0][s1];[s0]palettegen=max_colors=256[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5")
                    .arg("-loop")
                    .arg("0");  // Loop forever
                (path, cmd)
            }
        };

        let mut child = cmd
            .arg(output_path.to_string_lossy().to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| {
                anyhow::anyhow!(
                    "Failed to start ffmpeg (is it installed and on PATH?): {e}. Install ffmpeg or add it to PATH to record videos."
                )
            })?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to open ffmpeg stdin"))?;

        self.recording_output_path = Some(output_path.clone());
        self.recording_error = None;
        self.recording_start_time = Some(std::time::Instant::now());
        self.recording_last_frame_time = None;
        self.recording_readbacks.clear();
        self.recording_readback_index = 0;
        self.recording_pipe = Some(RecordingPipe {
            child,
            stdin: BufWriter::new(stdin),
            output_path,
            fps,
            out_width: capture_width,
            out_height: capture_height,
            in_pix_fmt,
        });

        Ok(())
    }

    fn recording_schedule_readback(
        &mut self,
        output_texture: &wgpu::Texture,
        encoder: &mut wgpu::CommandEncoder,
        now: std::time::Instant,
    ) {
        if !self.recording {
            return;
        }
        let Some(pipe) = &self.recording_pipe else {
            return;
        };

        let frame_interval = std::time::Duration::from_secs_f64(1.0 / pipe.fps.max(1) as f64);
        if let Some(last) = self.recording_last_frame_time {
            if now.duration_since(last) < frame_interval {
                return;
            }
        }

        let width = self.surface_config.width;
        let height = self.surface_config.height;
        if width == 0 || height == 0 {
            return;
        }

        let capture_width = self.recording_width.min(width);
        let capture_height = self.recording_height.min(height);
        if capture_width == 0 || capture_height == 0 {
            return;
        }

        // Ensure readback ring matches the current capture dimensions.
        let needs_recreate = self
            .recording_readbacks
            .first()
            .map(|s| s.width != capture_width || s.height != capture_height)
            .unwrap_or(true);
        if needs_recreate {
            self.recording_readbacks.clear();
            self.recording_readback_index = 0;

            let bytes_per_pixel = 4u32;
            let unpadded_bytes_per_row = capture_width * bytes_per_pixel;
            let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let padded_bytes_per_row = ((unpadded_bytes_per_row + align - 1) / align) * align;
            let buffer_size = (padded_bytes_per_row * capture_height) as u64;

            for i in 0..2 {
                let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Recording Readback Buffer[{i}]")),
                    size: buffer_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                self.recording_readbacks.push(RecordingReadbackSlot {
                    buffer,
                    padded_bytes_per_row,
                    width: capture_width,
                    height: capture_height,
                    pending_copy: false,
                    rx: None,
                    scratch: Vec::new(),
                });
            }
        }

        if self.recording_readbacks.is_empty() {
            return;
        }

        let slot_index = self.recording_readback_index % self.recording_readbacks.len();
        let slot = &mut self.recording_readbacks[slot_index];
        if slot.rx.is_some() || slot.pending_copy {
            // Still busy; skip scheduling to avoid piling up.
            return;
        }

        let w_px = width as f32;
        let h_px = height as f32;
        let capture_width_px = capture_width as f32;
        let capture_height_px = capture_height as f32;

        let cx = (self.recording_center_norm[0] * w_px).clamp(0.0, w_px);
        let cy = (self.recording_center_norm[1] * h_px).clamp(0.0, h_px);

        let max_x = (w_px - capture_width_px).max(0.0);
        let max_y = (h_px - capture_height_px).max(0.0);

        let origin_x = (cx - capture_width_px * 0.5).clamp(0.0, max_x).round() as u32;
        let origin_y = (cy - capture_height_px * 0.5).clamp(0.0, max_y).round() as u32;

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: origin_x,
                    y: origin_y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &slot.buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(slot.padded_bytes_per_row),
                    rows_per_image: Some(capture_height),
                },
            },
            wgpu::Extent3d {
                width: capture_width,
                height: capture_height,
                depth_or_array_layers: 1,
            },
        );

        // IMPORTANT: Do NOT map here. Mapping before submission will trip wgpu validation
        // because the buffer is considered mapped while it is also used by the submitted copy.
        slot.pending_copy = true;

        self.recording_last_frame_time = Some(now);
        self.recording_readback_index = (slot_index + 1) % self.recording_readbacks.len();
    }

    fn recording_begin_pending_maps(&mut self) {
        if !self.recording {
            return;
        }
        if self.recording_pipe.is_none() {
            return;
        }

        for slot in &mut self.recording_readbacks {
            if slot.pending_copy && slot.rx.is_none() {
                let slice = slot.buffer.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |result| {
                    let _ = tx.send(result);
                });
                slot.rx = Some(rx);
                slot.pending_copy = false;
            }
        }
    }

    fn recording_drain_ready_frames(&mut self) {
        if !self.recording {
            return;
        }
        let Some(pipe) = &mut self.recording_pipe else {
            return;
        };

        self.device.poll(wgpu::Maintain::Poll);

        let swap_is_bgra = matches!(
            self.surface_config.format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        );
        let want_bgra = pipe.in_pix_fmt == "bgra";

        for slot in &mut self.recording_readbacks {
            let Some(rx) = slot.rx.take() else {
                continue;
            };

            match rx.try_recv() {
                Ok(Ok(())) => {
                    let slice = slot.buffer.slice(..);
                    let data = slice.get_mapped_range();

                    // Unpad rows and (optionally) swizzle channels.
                    let width = slot.width;
                    let height = slot.height;
                    let needed = (width * height * 4) as usize;
                    if slot.scratch.len() != needed {
                        slot.scratch.resize(needed, 0);
                    }
                    let row_bytes = (width * 4) as usize;

                    for y in 0..height as usize {
                        let src_start = y * slot.padded_bytes_per_row as usize;
                        let src_row = &data[src_start..src_start + row_bytes];
                        let dst_start = y * row_bytes;
                        let dst_row = &mut slot.scratch[dst_start..dst_start + row_bytes];

                        if swap_is_bgra != want_bgra {
                            for (src_px, dst_px) in src_row
                                .chunks_exact(4)
                                .zip(dst_row.chunks_exact_mut(4))
                            {
                                dst_px[0] = src_px[2];
                                dst_px[1] = src_px[1];
                                dst_px[2] = src_px[0];
                                dst_px[3] = src_px[3];
                            }
                        } else {
                            dst_row.copy_from_slice(src_row);
                        }
                    }

                    drop(data);
                    slot.buffer.unmap();

                    if let Err(e) = pipe.stdin.write_all(&slot.scratch) {
                        self.recording_error = Some(format!("Recording write failed: {e}"));
                        self.recording = false;
                        return;
                    }
                }
                Ok(Err(e)) => {
                    self.recording_error = Some(format!("Recording readback failed: {e:?}"));
                    self.recording = false;
                    return;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // Not ready yet; put receiver back.
                    slot.rx = Some(rx);
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.recording_error = Some("Recording readback channel disconnected".to_string());
                    self.recording = false;
                    return;
                }
            }
        }
    }

    fn capture_4k_screenshot(&mut self) -> anyhow::Result<()> {
        // wgpu enforces a max storage-buffer binding size (commonly 128 MiB).
        // A single 4096x4096 RGBA32F buffer is 256 MiB, so we render in tiles and stitch.
        const SCREENSHOT_SIZE: u32 = 4096;
        const TILE_SIZE: u32 = 2048;

        let tiles_per_axis = SCREENSHOT_SIZE / TILE_SIZE;
        anyhow::ensure!(
            SCREENSHOT_SIZE % TILE_SIZE == 0,
            "SCREENSHOT_SIZE must be divisible by TILE_SIZE"
        );

        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let mut stitched_rgba = vec![0u8; (SCREENSHOT_SIZE * SCREENSHOT_SIZE * 4) as usize];

        // Keep camera math in shader-space units.
        let grid_size = self.sim_params_cpu.grid_size;
        let tile_zoom = 2.0; // each tile covers half the world per axis

        // Match ping-pong orientation used by draw pipelines.
        let (agents_in, agents_out) = if self.ping_pong {
            (&self.agents_buffer_b, &self.agents_buffer_a)
        } else {
            (&self.agents_buffer_a, &self.agents_buffer_b)
        };

        for ty in 0..tiles_per_axis {
            for tx in 0..tiles_per_axis {
                // Tile camera center (world-space)
                let pan_x = grid_size * (0.25 + 0.5 * (tx as f32));
                let pan_y = grid_size * (0.25 + 0.5 * (ty as f32));

                // Tile visual grid is RGBA32F (16 bytes per pixel)
                let visual_bytes_per_pixel: u32 = 16;
                let visual_unpadded_bpr = TILE_SIZE * visual_bytes_per_pixel;
                let visual_padded_bpr = ((visual_unpadded_bpr + align - 1) / align) * align;
                let visual_stride_pixels = visual_padded_bpr / visual_bytes_per_pixel;

                let visual_tile_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Screenshot Tile Visual Buffer"),
                    size: (visual_padded_bpr * TILE_SIZE) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });

                let agent_tile_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Screenshot Tile Agent Buffer"),
                    size: (visual_padded_bpr * TILE_SIZE) as u64,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });

                let visual_tile_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Screenshot Tile Visual Texture"),
                    size: wgpu::Extent3d {
                        width: TILE_SIZE,
                        height: TILE_SIZE,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba32Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let visual_tile_view =
                    visual_tile_texture.create_view(&wgpu::TextureViewDescriptor::default());

                let screenshot_tile_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Screenshot Tile Output Texture"),
                    size: wgpu::Extent3d {
                        width: TILE_SIZE,
                        height: TILE_SIZE,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                let screenshot_tile_view =
                    screenshot_tile_texture.create_view(&wgpu::TextureViewDescriptor::default());

                // Params for this tile: force draw-enabled and disable follow mode.
                let tile_params_cpu = SimParams {
                    window_width: TILE_SIZE as f32,
                    window_height: TILE_SIZE as f32,
                    visual_stride: visual_stride_pixels,
                    camera_zoom: tile_zoom,
                    camera_pan_x: pan_x,
                    camera_pan_y: pan_y,
                    prev_camera_pan_x: pan_x,
                    prev_camera_pan_y: pan_y,
                    follow_mode: 0,
                    draw_enabled: 1,
                    ..self.sim_params_cpu
                };

                let tile_params_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Screenshot Tile Params Buffer"),
                            contents: bytemuck::bytes_of(&tile_params_cpu),
                            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        });

                // Bind groups
                let process_layout = self.process_pipeline.get_bind_group_layout(0);
                let tile_bg_process = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Screenshot Tile Process Bind Group"),
                    layout: &process_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: agents_in.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: agents_out.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.chem_grid.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.fluid_dye_a.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: visual_tile_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: agent_tile_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: tile_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: self.trail_grid_inject.as_entire_binding(),
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
                        wgpu::BindGroupEntry {
                            binding: 18,
                            resource: wgpu::BindingResource::TextureView(&self.rain_map_texture_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 19,
                            resource: self.microswim_params_buffer.as_entire_binding(),
                        },
                    ],
                });

                let tile_composite_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Screenshot Tile Composite Bind Group"),
                    layout: &self.composite_agents_pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: visual_tile_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: agent_tile_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: tile_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.fluid_dye_a.as_entire_binding(),
                        },
                    ],
                });

                let tile_render_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Screenshot Tile Render Bind Group"),
                    layout: &self.render_pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&visual_tile_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: tile_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: agent_tile_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 16,
                            resource: self.fluid_velocity_a.as_entire_binding(),
                        },
                    ],
                });

                // Encode passes
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Screenshot Tile Encoder"),
                        });

                let width_workgroups = (TILE_SIZE + CLEAR_WG_SIZE_X - 1) / CLEAR_WG_SIZE_X;
                let height_workgroups = (TILE_SIZE + CLEAR_WG_SIZE_Y - 1) / CLEAR_WG_SIZE_Y;

                // Draw prep + agents
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Screenshot Tile Draw Prep"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.clear_visual_pipeline);
                    cpass.set_bind_group(0, &tile_bg_process, &[]);
                    cpass.dispatch_workgroups(width_workgroups, height_workgroups, 1);

                    cpass.set_pipeline(&self.clear_agent_grid_pipeline);
                    cpass.set_bind_group(0, &tile_bg_process, &[]);
                    cpass.dispatch_workgroups(width_workgroups, height_workgroups, 1);

                    cpass.set_pipeline(&self.render_agents_pipeline);
                    cpass.set_bind_group(0, &tile_bg_process, &[]);
                    cpass.dispatch_workgroups((self.agent_count + 255) / 256, 1, 1);
                }

                // Composite
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Screenshot Tile Composite"),
                        timestamp_writes: None,
                    });
                    cpass.set_pipeline(&self.composite_agents_pipeline);
                    cpass.set_bind_group(0, &tile_composite_bg, &[]);
                    cpass.dispatch_workgroups(width_workgroups, height_workgroups, 1);
                }

                // Copy tile visual buffer -> tile visual texture
                let visual_stride_bytes = visual_stride_pixels * visual_bytes_per_pixel;
                encoder.copy_buffer_to_texture(
                    wgpu::ImageCopyBuffer {
                        buffer: &visual_tile_buffer,
                        layout: wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(visual_stride_bytes),
                            rows_per_image: Some(TILE_SIZE),
                        },
                    },
                    wgpu::ImageCopyTexture {
                        texture: &visual_tile_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: TILE_SIZE,
                        height: TILE_SIZE,
                        depth_or_array_layers: 1,
                    },
                );

                // Render tile to BGRA8 screenshot
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Screenshot Tile Render"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &screenshot_tile_view,
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
                    rpass.set_bind_group(0, &tile_render_bg, &[]);
                    rpass.draw(0..6, 0..1);
                }

                // Readback tile
                let out_bytes_per_pixel: u32 = 4;
                let out_unpadded_bpr = TILE_SIZE * out_bytes_per_pixel;
                let out_padded_bpr = ((out_unpadded_bpr + align - 1) / align) * align;
                let out_buffer_size = (out_padded_bpr * TILE_SIZE) as u64;

                let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Screenshot Tile Readback Buffer"),
                    size: out_buffer_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });

                encoder.copy_texture_to_buffer(
                    wgpu::ImageCopyTexture {
                        texture: &screenshot_tile_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::ImageCopyBuffer {
                        buffer: &out_buffer,
                        layout: wgpu::ImageDataLayout {
                            offset: 0,
                            bytes_per_row: Some(out_padded_bpr),
                            rows_per_image: Some(TILE_SIZE),
                        },
                    },
                    wgpu::Extent3d {
                        width: TILE_SIZE,
                        height: TILE_SIZE,
                        depth_or_array_layers: 1,
                    },
                );

                self.queue.submit(Some(encoder.finish()));

                let slice = out_buffer.slice(..);
                let (sender, receiver) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |res| {
                    sender.send(res).ok();
                });
                self.device.poll(wgpu::Maintain::Wait);
                receiver.recv()??;

                let data = slice.get_mapped_range();
                // Note: the GPU/shader coordinate system and image memory coordinate system
                // disagree on Y direction for our camera mapping. Flip the tile-row index so
                // the stitched PNG has correct top/bottom ordering.
                let dst_ty = (tiles_per_axis - 1) - ty;
                for y in 0..TILE_SIZE {
                    let src_offset = (y * out_padded_bpr) as usize;
                    let dst_y = dst_ty * TILE_SIZE + y;
                    let dst_x = tx * TILE_SIZE;
                    let dst_offset = ((dst_y * SCREENSHOT_SIZE + dst_x) * 4) as usize;

                    let src_row = &data[src_offset..src_offset + (out_unpadded_bpr as usize)];
                    let dst_row =
                        &mut stitched_rgba[dst_offset..dst_offset + (out_unpadded_bpr as usize)];
                    dst_row.copy_from_slice(src_row);

                    // BGRA -> RGBA
                    for px in dst_row.chunks_exact_mut(4) {
                        px.swap(0, 2);
                    }
                }

                drop(data);
                out_buffer.unmap();
            }
        }

        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Create screenshots directory if it doesn't exist
        std::fs::create_dir_all("screenshots")?;

        let filename = format!("screenshots/screenshot_4k_{}_{}.jpg", self.run_name, timestamp);

        let img = image::RgbaImage::from_raw(SCREENSHOT_SIZE, SCREENSHOT_SIZE, stitched_rgba)
            .ok_or_else(|| anyhow::anyhow!("Failed to create image from stitched buffer"))?;

        // Convert to RGB and save as JPEG with 90% quality
        let rgb_img = image::DynamicImage::ImageRgba8(img).to_rgb8();
        let mut output = std::fs::File::create(&filename)?;
        let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut output, 90);
        encoder.encode(
            rgb_img.as_raw(),
            rgb_img.width(),
            rgb_img.height(),
            image::ColorType::Rgb8,
        )?;

        println!("Screenshot saved: {}", filename);
        Ok(())
    }

    fn load_snapshot_from_file(&mut self, path: &Path) -> anyhow::Result<()> {
        // Snapshot load can race with in-flight async readbacks.
        // We must not submit commands that touch a buffer while it is mapped.
        // Don't call `unmap()` blindly (wgpu panics if the buffer isn't mapped);
        // instead, wait briefly and drain completions so the existing callbacks unmap.
        self.device.poll(wgpu::Maintain::Wait);
        self.process_completed_alive_readbacks();

        // Load snapshot from PNG file early so we can validate compatibility before touching GPU resources.
        let (alpha_grid, beta_grid, gamma_grid, snapshot) = load_simulation_snapshot(path)?;
        let loaded_run_name = snapshot.run_name.clone();

        // Snapshot compatibility gate.
        // - Empty version strings are treated as legacy "1.0".
        // - Current builds write SNAPSHOT_VERSION.
        match snapshot.version.as_str() {
            "" | "1.0" | "1.1" | "1.2" => {}
            other => {
                anyhow::bail!(
                    "Unsupported snapshot version '{other}'. This build supports snapshot versions 1.0, 1.1, and 1.2."
                );
            }
        }

        // Resolution compatibility / adaptation.
        // - New snapshots (v1.2+) include explicit resolution fields.
        // - Older snapshots may have zeros; infer resolution from grid length.
        // - If snapshot env resolution differs from current GPU env resolution, resample grids
        //   to prevent buffer/texture overruns.
        if alpha_grid.len() != beta_grid.len() || alpha_grid.len() != gamma_grid.len() {
            anyhow::bail!(
                "Snapshot grid length mismatch: alpha={}, beta={}, gamma={}",
                alpha_grid.len(),
                beta_grid.len(),
                gamma_grid.len()
            );
        }

        let snapshot_env_res = if snapshot.env_grid_resolution != 0 {
            snapshot.env_grid_resolution
        } else {
            let len = alpha_grid.len();
            let inferred = (len as f64).sqrt().round() as usize;
            if inferred * inferred == len {
                inferred as u32
            } else {
                0
            }
        };

        if snapshot_env_res != 0 && snapshot_env_res != self.env_grid_resolution {
            anyhow::bail!(
                "Snapshot env grid res {} differs from current {}. The app must reset to the snapshot resolution before loading.",
                snapshot_env_res,
                self.env_grid_resolution
            );
        }

        let (alpha_grid, beta_grid, gamma_grid) = (alpha_grid, beta_grid, gamma_grid);

        if alpha_grid.len() != self.env_grid_cell_count {
            anyhow::bail!(
                "Snapshot env grid length {} does not match current env grid cell count {} (env_res={}).",
                alpha_grid.len(),
                self.env_grid_cell_count,
                self.env_grid_resolution
            );
        }

        // Reset simulation state before loading to prevent crashes on subsequent loads
        // Clear alive counter
        // Clear [spawn, debug, alive] counters
        self.queue.write_buffer(&self.spawn_debug_counters, 0, bytemuck::cast_slice(&[0u32, 0u32, 0u32]));

        // Trails are not stored in the snapshot PNG. If we don't clear them here,
        // the GPU buffers can contain uninitialized data (often NaNs), which then
        // permanently poisons the energy trail channel.
        {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Clear trail buffers (snapshot load)"),
            });
            encoder.clear_buffer(&self.trail_grid, 0, None);
            encoder.clear_buffer(&self.trail_grid_inject, 0, None);
            self.queue.submit(Some(encoder.finish()));
        }

        // Extra safety: some backends/drivers can still leave stale data visible until the
        // first frame submits more work. Force a deterministic zero-fill via CPU upload.
        let trail_bytes = (self.env_grid_cell_count * std::mem::size_of::<[f32; 4]>()) as usize;
        let zeros = vec![0u8; trail_bytes];
        self.queue.write_buffer(&self.trail_grid, 0, &zeros);
        self.queue.write_buffer(&self.trail_grid_inject, 0, &zeros);

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

        self.pending_spawn_upload = false;

        // Reset readback state so epoch-0 readbacks are accepted.
        self.alive_readback_last_applied_epoch = 0;
        self.alive_readback_zero_streak = 0;
        for inflight in &mut self.alive_readback_inflight {
            *inflight = false;
        }
        for pending in &self.alive_readback_pending {
            if let Ok(mut guard) = pending.lock() {
                *guard = None;
            }
        }

        // Canonicalize buffer orientation during load: spawn path materializes into A.
        self.ping_pong = false;

        // Apply loaded settings only if they exist in the snapshot (backwards compatibility)
        if let Some(settings) = &snapshot.settings {
            self.apply_settings(settings);
        } else {
            println!("G�� Loaded snapshot without settings (old format) - using current settings");
        }

        // Resolution compatibility already validated above (before touching GPU resources).

        // Restore run name if present; otherwise generate one for this session.
        if !loaded_run_name.is_empty() {
            self.run_name = loaded_run_name;
        } else {
            self.run_name = naming::sim::generate_sim_name(
                &self.current_settings(),
                self.run_seed,
                self.agent_buffer_capacity as u32,
            );
        }

        // Upload grids to GPU (alpha/beta plus current rain maps stored in chem_grid.zw)
        let mut chem_grid: Vec<[f32; 4]> = Vec::with_capacity(alpha_grid.len());
        for i in 0..alpha_grid.len() {
            let rain_idx = i * 2;
            // Fallback to uniform rain if CPU cache is unexpectedly missing values.
            let rain_alpha = self.rain_map_data.get(rain_idx).copied().unwrap_or(1.0);
            let rain_beta = self
                .rain_map_data
                .get(rain_idx + 1)
                .copied()
                .unwrap_or(1.0);
            chem_grid.push([alpha_grid[i], beta_grid[i], rain_alpha, rain_beta]);
        }
        self.queue.write_buffer(&self.chem_grid, 0, bytemuck::cast_slice(&chem_grid));
        self.queue.write_buffer(&self.gamma_grid, 0, bytemuck::cast_slice(&gamma_grid));

        // Queue agents from snapshot for spawning
        for agent_snap in snapshot.agents.iter() {
            let spawn_req = agent_snap.to_spawn_request();
            self.cpu_spawn_queue.push(spawn_req);
        }

        // Update spawn request count so they get processed
        self.spawn_request_count = self.cpu_spawn_queue.len() as u32;

        println!(
            "Queued {} agents from snapshot for spawning",
            self.spawn_request_count
        );

        println!(
            "Spawning snapshot agents immediately (capacity: {}, currently: {})",
            self.agent_buffer_capacity,
            self.agent_count
        );

        // Spawn queued snapshot agents immediately (in batches), so the restored state is fully
        // materialized on the GPU before we write an autosave.
        while !self.cpu_spawn_queue.is_empty() {
            let capacity_left = self
                .agent_buffer_capacity
                .saturating_sub(self.agent_count as usize);
            if capacity_left == 0 {
                println!(
                    "WARNING: Snapshot contains more agents than current capacity (agent_count: {}, capacity: {}). Remaining queued: {}",
                    self.agent_count,
                    self.agent_buffer_capacity,
                    self.cpu_spawn_queue.len()
                );
                break;
            }

            let batch = (self.cpu_spawn_queue.len() as u32)
                .min(MAX_CPU_SPAWNS_PER_BATCH)
                .min(capacity_left as u32);

            println!(
                "  -> spawning batch of {} (queued remaining before: {})",
                batch,
                self.cpu_spawn_queue.len()
            );

            // Use the reliable spawn-only path (works even when paused).
            self.process_spawn_requests_only(batch, false);

            println!(
                "  -> after batch: agent_count={}, alive_count={}, queued remaining={}",
                self.agent_count,
                self.alive_count,
                self.cpu_spawn_queue.len()
            );
        }

        // Update epoch from snapshot
        self.epoch = snapshot.epoch;

        // DEFENSIVE: Ensure spawn queue and counters are completely clear after batch spawning.
        // This prevents phantom re-spawning on subsequent update() frames.
        self.cpu_spawn_queue.clear();
        self.spawn_request_count = 0;
        self.pending_spawn_upload = false;

        println!(
            "After spawn loop: queue.len()={}, spawn_request_count={}, pending_spawn_upload={}",
            self.cpu_spawn_queue.len(),
            self.spawn_request_count,
            self.pending_spawn_upload
        );

        // Clear spatial grid so we don't treat stale cells as valid for the new epoch.
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Load Snapshot Spatial Grid Clear Encoder"),
                });
            encoder.clear_buffer(&self.agent_spatial_grid_buffer, 0, None);
            self.queue.submit(Some(encoder.finish()));
        }

        println!(
            "G�� Loaded settings and restored snapshot agents (alive_count: {}, agent_count: {})",
            self.alive_count,
            self.agent_count
        );

        Ok(())
    }

    fn spawn_agent(&mut self, parent_agent: &Agent) {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005u64)
            .wrapping_add(1442695040888963407u64);

        let mut new_agent = Agent::zeroed();

        // Mutate genome (via legacy ASCII representation) then repack.
        let parent_ascii = genome_packed_to_ascii_words(
            &parent_agent.genome_packed,
            parent_agent.genome_offset,
            parent_agent.gene_length,
        );
        let mutated_ascii = self.mutate_genome(&parent_ascii);
        let (gene_length, genome_offset, genome_packed) = genome_pack_ascii_words(&mutated_ascii);
        new_agent.gene_length = gene_length;
        new_agent.genome_offset = genome_offset;
        new_agent.genome_packed = genome_packed;

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
        let sim_size = self.sim_size;
        new_agent.position[0] = new_agent.position[0].rem_euclid(sim_size);
        new_agent.position[1] = new_agent.position[1].rem_euclid(sim_size);

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

        // Keep the init-dead indirect-dispatch writer's max_agents in sync.
        self.queue.write_buffer(
            &self.init_dead_params_buffer,
            0,
            bytemuck::cast_slice(&[new_capacity as u32, 0u32, 0u32, 0u32]),
        );

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
                    resource: self.chem_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.fluid_dye_a.as_entire_binding(),
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
                    resource: self.trail_grid_inject.as_entire_binding(),
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
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: wgpu::BindingResource::TextureView(&self.rain_map_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: self.microswim_params_buffer.as_entire_binding(),
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
                    resource: self.chem_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.fluid_dye_a.as_entire_binding(),
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
                    resource: self.trail_grid_inject.as_entire_binding(),
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
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: wgpu::BindingResource::TextureView(&self.rain_map_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: self.microswim_params_buffer.as_entire_binding(),
                },
            ],
        });

        self.init_dead_writer_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("InitDead Writer Bind Group"),
            layout: &self.init_dead_writer_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.spawn_debug_counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.init_dead_dispatch_args.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.init_dead_params_buffer.as_entire_binding(),
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

        // IMPORTANT:
        // Reset can be triggered while async readbacks are still mapped/in-flight.
        // Wait for any in-flight maps to complete and drain their callbacks so buffers
        // get unmapped by the existing code paths before we submit new work.
        self.device.poll(wgpu::Maintain::Wait);
        self.process_completed_alive_readbacks();

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

        // Reset egui caching (force a rebuild on the next present)
        self.last_egui_update_time = std::time::Instant::now() - std::time::Duration::from_secs(1);
        self.cached_egui_primitives.clear();

        // Reset camera
        self.camera_zoom = 1.0;
        let sim_size = self.sim_size;
        self.camera_pan = [sim_size / 2.0, sim_size / 2.0];
        self.prev_camera_pan = [sim_size / 2.0, sim_size / 2.0];
        self.camera_target = [sim_size / 2.0, sim_size / 2.0];
        self.camera_velocity = [0.0, 0.0];

        // Reset selection
        self.selected_agent_index = None;
        self.selected_agent_data = None;
        self.follow_selected_agent = false;

        // Reset async readback state.
        // IMPORTANT: alive-count readbacks are epoch-tagged and can complete out-of-order.
        // If we reset `epoch` back to 0 but keep the last-applied epoch from the previous run,
        // then post-reset readbacks would be ignored as "out-of-order", making the sim appear
        // permanently empty (including spawns).
        self.alive_readback_last_applied_epoch = 0;
        self.alive_readback_zero_streak = 0;
        for inflight in &mut self.alive_readback_inflight {
            *inflight = false;
        }
        for pending in &self.alive_readback_pending {
            if let Ok(mut guard) = pending.lock() {
                *guard = None;
            }
        }

        self.alive_readback_slot = 0;

        for inflight in &mut self.selected_agent_readback_inflight {
            *inflight = false;
        }
        for pending in &self.selected_agent_readback_pending {
            if let Ok(mut guard) = pending.lock() {
                *guard = None;
            }
        }
        self.selected_agent_readback_last_request = std::time::Instant::now()
            - std::time::Duration::from_secs(1);

        // Canonicalize ping-pong orientation after reset so "current agents" lives in buffer A.
        self.ping_pong = false;

        // Reset statistics
        self.population_history.clear();
        self.population_plot_points.clear();
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

        self.run_seed = (self.rng_state >> 32) as u32;
        self.run_name = naming::sim::generate_sim_name(
            &self.current_settings(),
            self.run_seed,
            self.agent_buffer_capacity as u32,
        );

        // Clear and reinitialize GPU buffers using compute shaders
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Reset Encoder"),
        });

        // Clear spatial grid (prevents stale spatial cells after epoch resets).
        encoder.clear_buffer(&self.agent_spatial_grid_buffer, 0, None);

        // Clear [spawn, debug, alive] counters
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

        // Clear fluid simulation buffers (velocity/pressure/dye ping-pong + forces).
        // This prevents stale flow/dye from persisting across resets.
        let fluid_groups = (self.fluid_grid_resolution + 15) / 16;
        let dye_groups = (self.env_grid_resolution + 15) / 16;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Fluid Reset Pass"),
                timestamp_writes: None,
            });

            // Clear intermediate propeller force vectors (written by agents) and combined forces.
            pass.set_pipeline(&self.clear_fluid_force_vectors_pipeline);
            pass.set_bind_group(0, &self.fluid_bind_group_ab, &[]);
            pass.dispatch_workgroups(fluid_groups, fluid_groups, 1);

            pass.set_pipeline(&self.fluid_clear_forces_pipeline);
            pass.set_bind_group(0, &self.fluid_bind_group_ab, &[]);
            pass.dispatch_workgroups(fluid_groups, fluid_groups, 1);

            // Clear both velocity buffers (A and B).
            pass.set_pipeline(&self.fluid_clear_velocity_pipeline);
            pass.set_bind_group(0, &self.fluid_bind_group_ab, &[]);
            pass.dispatch_workgroups(fluid_groups, fluid_groups, 1);
            pass.set_bind_group(0, &self.fluid_bind_group_ba, &[]);
            pass.dispatch_workgroups(fluid_groups, fluid_groups, 1);

            // Clear both pressure buffers (A and B).
            pass.set_pipeline(&self.fluid_clear_pressure_pipeline);
            pass.set_bind_group(0, &self.fluid_bind_group_ab, &[]);
            pass.dispatch_workgroups(fluid_groups, fluid_groups, 1);
            pass.set_bind_group(0, &self.fluid_bind_group_ba, &[]);
            pass.dispatch_workgroups(fluid_groups, fluid_groups, 1);

            // Clear both dye buffers (A and B).
            pass.set_pipeline(&self.fluid_clear_dye_pipeline);
            pass.set_bind_group(0, &self.fluid_bind_group_ab, &[]);
            pass.dispatch_workgroups(dye_groups, dye_groups, 1);
            pass.set_bind_group(0, &self.fluid_bind_group_ba, &[]);
            pass.dispatch_workgroups(dye_groups, dye_groups, 1);
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
        // First time initialization - create new state with settings from file
        // Load resolution settings from simulation_settings.json
        let settings_path = std::path::Path::new(SETTINGS_FILE_NAME);
        let (env_res, fluid_res, spatial_res) = if let Ok(settings) = SimulationSettings::load_from_disk(settings_path) {
            (
                settings.env_grid_resolution,
                settings.fluid_grid_resolution,
                settings.spatial_grid_resolution,
            )
        } else {
            // Fallback to defaults if settings file doesn't exist
            (
                DEFAULT_ENV_GRID_RESOLUTION,
                DEFAULT_FLUID_GRID_RESOLUTION,
                DEFAULT_SPATIAL_GRID_RESOLUTION,
            )
        };

        let mut new_state = pollster::block_on(GpuState::new_from_settings(window.clone(), env_res, fluid_res, spatial_res));
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
    _instance: &wgpu::Instance,
    surface: &wgpu::Surface,
    adapter: &wgpu::Adapter,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    // Load splash image
    let splash_img = match image::open("maps/banner_v0.1.jpeg") {
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
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
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
    use anyhow::Context;

    if alpha_grid.len() != beta_grid.len() || alpha_grid.len() != gamma_grid.len() {
        anyhow::bail!(
            "Grid length mismatch: alpha={}, beta={}, gamma={}",
            alpha_grid.len(),
            beta_grid.len(),
            gamma_grid.len()
        );
    }

    let cell_count = alpha_grid.len();
    let env_res = if snapshot.env_grid_resolution != 0 {
        snapshot.env_grid_resolution
    } else {
        let inferred = (cell_count as f64).sqrt().round() as usize;
        if inferred * inferred == cell_count {
            inferred as u32
        } else {
            anyhow::bail!("Cannot infer snapshot resolution from cell_count={cell_count}");
        }
    };
    if (env_res as usize) * (env_res as usize) != cell_count {
        anyhow::bail!(
            "Snapshot env resolution {} does not match grid length {}",
            env_res,
            cell_count
        );
    }

    // 1. Create RGB image from grids
    let mut img_data = vec![0u8; cell_count * 3];
    for i in 0..cell_count {
        img_data[i * 3 + 0] = (beta_grid[i].clamp(0.0, 1.0) * 255.0) as u8;   // R = beta (poison)
        img_data[i * 3 + 1] = (alpha_grid[i].clamp(0.0, 1.0) * 255.0) as u8;  // G = alpha (food)
        img_data[i * 3 + 2] = (gamma_grid[i].clamp(0.0, 1.0) * 255.0) as u8;  // B = gamma (terrain)
    }

    // 2. Serialize and compress metadata
    let json = serde_json::to_string(snapshot)?;
    let compressed = zstd::encode_all(json.as_bytes(), 3)?;
    use base64::Engine;
    let encoded = base64::engine::general_purpose::STANDARD.encode(&compressed);

    // 3. Write PNG with custom text chunk (atomic write via temp + rename)
    let tmp_path = {
        let file_name = path
            .file_name()
            .context("Snapshot path has no file name")?
            .to_string_lossy();
        path.with_file_name(format!("{}.tmp", file_name))
    };

    {
        let file = std::fs::File::create(&tmp_path)?;
        let w = BufWriter::new(file);
        let mut encoder = png::Encoder::new(w, env_res, env_res);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);

        // Add metadata as uncompressed text chunk
        encoder.add_text_chunk("RibossomeSnapshot".to_string(), encoded)?;

        let mut writer = encoder.write_header()?;
        writer.write_image_data(&img_data)?;
        writer.finish()?;
    }

    // On Windows, rename over an existing file fails, so remove first.
    if path.exists() {
        let _ = std::fs::remove_file(path);
    }
    std::fs::rename(&tmp_path, path)?;

    if snapshot.run_name.is_empty() {
        println!("G�� Saved snapshot: {} agents, epoch {}", snapshot.agents.len(), snapshot.epoch);
    } else {
        println!(
            "G�� Saved snapshot: {} agents, epoch {}, run '{}'",
            snapshot.agents.len(),
            snapshot.epoch,
            snapshot.run_name
        );
    }
    Ok(())
}

fn load_simulation_snapshot(
    path: &Path,
) -> anyhow::Result<(Vec<f32>, Vec<f32>, Vec<f32>, SimulationSnapshot)> {
    use std::io::BufReader;
    use anyhow::Context;

    let file = std::fs::File::open(path)?;
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder.read_info()?;

    // Read image data
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf)?;

    if info.width == 0 || info.height == 0 || info.width != info.height {
        anyhow::bail!("Invalid snapshot size: {}x{} (expected square)", info.width, info.height);
    }

    let bytes_per_pixel = match (info.color_type, info.bit_depth) {
        (png::ColorType::Rgb, png::BitDepth::Eight) => 3usize,
        (png::ColorType::Rgba, png::BitDepth::Eight) => 4usize,
        other => {
            anyhow::bail!("Unsupported snapshot pixel format: {:?}", other);
        }
    };

    let cell_count = (info.width as usize) * (info.height as usize);
    let required = cell_count
        .checked_mul(bytes_per_pixel)
        .context("Snapshot image too large")?;
    if buf.len() < required {
        anyhow::bail!(
            "Snapshot buffer too small: have {}, need {}",
            buf.len(),
            required
        );
    }

    // Extract grids from RGB channels
    let mut alpha_grid = vec![0.0f32; cell_count];
    let mut beta_grid = vec![0.0f32; cell_count];
    let mut gamma_grid = vec![0.0f32; cell_count];

    for i in 0..cell_count {
        let base = i * bytes_per_pixel;
        beta_grid[i] = buf[base + 0] as f32 / 255.0; // R = beta (poison)
        alpha_grid[i] = buf[base + 1] as f32 / 255.0; // G = alpha (food)
        gamma_grid[i] = buf[base + 2] as f32 / 255.0; // B = gamma (terrain)
    }

    // Extract metadata from text chunks
    let text_chunks = reader.info().uncompressed_latin1_text.clone();
    for chunk in text_chunks.iter() {
        if chunk.keyword == "RibossomeSnapshot" {
            use base64::Engine;
            let compressed = base64::engine::general_purpose::STANDARD.decode(&chunk.text)?;
            let json = zstd::decode_all(&compressed[..])?;
            let mut value: serde_json::Value = serde_json::from_slice(&json)?;
            scrub_json_nulls(&mut value);
            let mut snapshot: SimulationSnapshot = serde_json::from_value(value)?;
            // Backfill resolution fields for legacy snapshots using the PNG dimensions.
            if snapshot.env_grid_resolution == 0 {
                snapshot.env_grid_resolution = info.width;
            }

            if snapshot.run_name.is_empty() {
                println!(
                    "G�� Loaded snapshot: {} agents, epoch {}, saved {}",
                    snapshot.agents.len(),
                    snapshot.epoch,
                    snapshot.timestamp
                );
            } else {
                println!(
                    "G�� Loaded snapshot: {} agents, epoch {}, saved {}, run '{}'",
                    snapshot.agents.len(),
                    snapshot.epoch,
                    snapshot.timestamp,
                    snapshot.run_name
                );
            }

            return Ok((alpha_grid, beta_grid, gamma_grid, snapshot));
        }
    }

    anyhow::bail!("No RibossomeSnapshot metadata found in PNG")
}


fn encode_rain_map_blob(rain_map: &[f32]) -> anyhow::Result<Option<String>> {
    const DEFAULT_EPS: f32 = 1e-5;
    if rain_map.iter().all(|&value| (value - 1.0).abs() <= DEFAULT_EPS) {
        return Ok(None);
    }

    let mut quantized = Vec::with_capacity(rain_map.len() * 2);
    for &value in rain_map {
        let clamped = value.clamp(0.0, 1.0);
        let scaled = (clamped * u16::MAX as f32).round().clamp(0.0, u16::MAX as f32) as u16;
        quantized.extend_from_slice(&scaled.to_le_bytes());
    }

    let compressed = zstd::encode_all(&quantized[..], 3)?;
    use base64::Engine;
    Ok(Some(base64::engine::general_purpose::STANDARD.encode(&compressed)))
}

fn decode_rain_map_blob(blob: &str) -> anyhow::Result<Vec<f32>> {
    use anyhow::bail;
    use base64::Engine;

    let decoded = base64::engine::general_purpose::STANDARD.decode(blob)?;
    let decompressed = zstd::decode_all(&decoded[..])?;
    let expected_len = GRID_CELL_COUNT * 2 * 2;
    if decompressed.len() != expected_len {
        bail!(
            "Rain map blob has {} bytes, expected {}",
            decompressed.len(),
            expected_len
        );
    }

    let mut result = vec![0.0f32; GRID_CELL_COUNT * 2];
    for (idx, chunk) in decompressed.chunks_exact(2).enumerate() {
        let quantized = u16::from_le_bytes([chunk[0], chunk[1]]);
        result[idx] = quantized as f32 / u16::MAX as f32;
    }
    Ok(result)
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
                    .with_title(format!("{} v{}", APP_NAME, APP_VERSION))
                    .with_inner_size(winit::dpi::LogicalSize::new(1100, 600)), // 800 + 300 for inspector
            )
            .unwrap(),
    );

    // Create WGPU resources once
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: if cfg!(target_os = "windows") {
            wgpu::Backends::VULKAN // Use Vulkan on Windows to support atomicCompareExchangeWeak
        } else {
            wgpu::Backends::PRIMARY
        },
        ..Default::default()
    });
    let surface = instance.create_surface(window.clone()).unwrap();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .unwrap();

    let required_features = select_required_features(adapter.features());
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("GPU Device"),
            required_features,
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

    // Load settings BEFORE creating GPU state so we can use resolution values.
    let settings_path = SimulationSettings::default_path();
    let loaded_settings = match SimulationSettings::load_from_disk(&settings_path) {
        Ok(mut s) => {
            s.sanitize();
            s
        }
        Err(err) => {
            eprintln!("Warning: failed to load settings from {:?}: {err:?}", &settings_path);
            let mut s = SimulationSettings::default();
            s.sanitize();
            s
        }
    };

    fn snapshot_target_resolutions(path: &std::path::Path) -> anyhow::Result<(u32, u32, u32)> {
        let (_a, _b, _g, snapshot) = load_simulation_snapshot(path)?;
        let env = snapshot.env_grid_resolution;
        let fluid = if snapshot.fluid_grid_resolution != 0 {
            snapshot.fluid_grid_resolution
        } else {
            env / 4
        };
        let spatial = if snapshot.spatial_grid_resolution != 0 {
            snapshot.spatial_grid_resolution
        } else {
            env / 4
        };
        Ok((env, fluid, spatial))
    }

    // If an autosave snapshot exists, prefer starting at its resolution.
    // This avoids a jarring “boot at default 2048 then immediately reset to 1024/512” flow.
    let (env_grid_res, fluid_grid_res, spatial_grid_res) = {
        let autosave_path = std::path::Path::new(AUTO_SNAPSHOT_FILE_NAME);
        if autosave_path.exists() {
            if let Ok((env_res, fluid_res, spatial_res)) = snapshot_target_resolutions(autosave_path) {
                println!(
                    "Gℹ️  Autosave detected at {}x{}; starting with autosave resolution.",
                    env_res, env_res
                );
                (env_res, fluid_res, spatial_res)
            } else {
                (
                    loaded_settings.env_grid_resolution,
                    loaded_settings.fluid_grid_resolution,
                    loaded_settings.spatial_grid_resolution,
                )
            }
        } else {
            (
                loaded_settings.env_grid_resolution,
                loaded_settings.fluid_grid_resolution,
                loaded_settings.spatial_grid_resolution,
            )
        }
    };

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
            env_grid_res,
            fluid_grid_res,
            spatial_grid_res,
        ));
        tx.send(state).unwrap();
    });

    // Animation state for loading screen
    let loading_start = std::time::Instant::now();
    let last_message_update = std::time::Instant::now();

    const LOADING_MESSAGES: &[&str] = &[
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

    struct App {
        state: Option<GpuState>,
        egui_state: egui_winit::State,
        window: Arc<winit::window::Window>,
        rx: std::sync::mpsc::Receiver<GpuState>,
        loading_start: std::time::Instant,
        last_message_update: std::time::Instant,
        current_message_index: usize,
        loading_messages: &'static [&'static str],
        skip_auto_load: bool, // Set to true when resolution changes to prevent loading old-resolution snapshot
        preserve_autosave_on_next_reset: bool,
        reset_in_progress: bool,
        pending_snapshot_load_path: Option<std::path::PathBuf>,
        pending_resolution_change_override: Option<(u32, u32, u32)>,
        force_reset_requested: bool,
    }

    impl winit::application::ApplicationHandler for App {
        fn resumed(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {}

        fn window_event(
            &mut self,
            event_loop: &winit::event_loop::ActiveEventLoop,
            _window_id: winit::window::WindowId,
            event: winit::event::WindowEvent,
        ) {
            self.handle_event(winit::event::Event::WindowEvent {
                window_id: self.window.id(),
                event,
            }, event_loop);
        }

        fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
            self.handle_event(winit::event::Event::AboutToWait, event_loop);
        }
    }

    impl App {
        fn handle_event(&mut self, event: winit::event::Event<()>, target: &winit::event_loop::ActiveEventLoop) {
            let window = &self.window;

        // Animate loading messages in window title while waiting for GPU state
        if self.state.is_none() {
            let now = std::time::Instant::now();
            if now.duration_since(self.last_message_update).as_millis() > 1000 {
                self.current_message_index = (self.current_message_index + 1) % self.loading_messages.len();
                self.last_message_update = now;
                let elapsed = now.duration_since(self.loading_start).as_secs_f32();
                let dots = ".".repeat((elapsed * 2.0) as usize % 4);
                let message = format!("{}{}", self.loading_messages[self.current_message_index], dots);
                window.set_title(&format!("{} v{} - {}", APP_NAME, APP_VERSION, &message));
                // Also print to console so messages are visible
                println!("Loading: {}", &message);
            }
        }

        // Check if loading finished
        if self.state.is_none() {
            if let Ok(mut loaded_state) = self.rx.try_recv() {
                // If we have a pending snapshot path (e.g. manual load requested a resolution reset),
                // load it immediately into this freshly-created GPU state.
                if let Some(path) = self.pending_snapshot_load_path.take() {
                    match loaded_state.load_snapshot_from_file(&path) {
                        Ok(_) => println!("? Snapshot loaded from: {}", path.display()),
                        Err(e) => eprintln!("? Failed to load snapshot after reset: {}", e),
                    }
                } else {
                    // Try to auto-load previous session from autosave snapshot.
                    // If autosave resolution differs, schedule a reset to the autosave resolution first.
                    // Don't call reset_simulation_state() - state is already fresh from creation.
                    let autosave_path = std::path::Path::new(AUTO_SNAPSHOT_FILE_NAME);
                    if autosave_path.exists() && !self.skip_auto_load {
                        match snapshot_target_resolutions(autosave_path) {
                            Ok((env_res, fluid_res, spatial_res)) => {
                                if env_res != loaded_state.env_grid_resolution
                                    || fluid_res != loaded_state.fluid_grid_resolution
                                    || spatial_res != loaded_state.spatial_grid_resolution
                                {
                                    println!(
                                        "G⚠️  Autosave resolution {}x{} differs from current {}x{}; resetting to match.",
                                        env_res,
                                        env_res,
                                        loaded_state.env_grid_resolution,
                                        loaded_state.env_grid_resolution
                                    );
                                    self.pending_snapshot_load_path = Some(autosave_path.to_path_buf());
                                    self.preserve_autosave_on_next_reset = true;
                                    self.pending_resolution_change_override = Some((env_res, fluid_res, spatial_res));
                                    self.force_reset_requested = true;
                                } else {
                                    match loaded_state.load_snapshot_from_file(autosave_path) {
                                        Ok(_) => println!(
                                            "? Auto-loaded previous session from epoch {}",
                                            loaded_state.epoch
                                        ),
                                        Err(e) => eprintln!("? Failed to auto-load snapshot: {:?}", e),
                                    }
                                }
                            }
                            Err(e) => eprintln!("? Failed to inspect autosave snapshot: {e:?}"),
                        }
                    } else if self.skip_auto_load {
                        println!("? Skipping auto-load due to resolution change");
                        self.skip_auto_load = false; // Reset flag
                    }
                }

                self.state = Some(loaded_state);
                window.set_title(&format!("{} v{}", APP_NAME, APP_VERSION));
                let _ = window.request_inner_size(winit::dpi::LogicalSize::new(1600, 800));
            }
        }

        match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => {
                // Let egui handle the event first
                let response = self.egui_state.on_window_event(&window, &event);

                // Only handle simulation controls if egui didn't consume the event
                if !response.consumed {
                    match event {
                        WindowEvent::CloseRequested => {
                            if let Some(mut existing) = self.state.take() {
                                existing.destroy_resources();
                            }
                            target.exit();
                        }
                        WindowEvent::Resized(physical_size) => {
                            if let Some(state) = self.state.as_mut() {
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
                            if let Some(state) = self.state.as_mut() {
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
                                            let sim_size = state.sim_size;
                                            state.camera_pan[1] = state.camera_pan[1].clamp(-0.25 * sim_size, 1.25 * sim_size);
                                            state.follow_selected_agent = false;
                                            camera_changed = true;
                                        }
                                        PhysicalKey::Code(KeyCode::KeyS) => {
                                            state.camera_pan[1] += 200.0 / state.camera_zoom;
                                            let sim_size = state.sim_size;
                                            state.camera_pan[1] = state.camera_pan[1].clamp(-0.25 * sim_size, 1.25 * sim_size);
                                            state.follow_selected_agent = false;
                                            camera_changed = true;
                                        }
                                        PhysicalKey::Code(KeyCode::KeyA) => {
                                            state.camera_pan[0] -= 200.0 / state.camera_zoom;
                                            let sim_size = state.sim_size;
                                            state.camera_pan[0] = state.camera_pan[0].clamp(-0.25 * sim_size, 1.25 * sim_size);
                                            state.follow_selected_agent = false;
                                            camera_changed = true;
                                        }
                                        PhysicalKey::Code(KeyCode::KeyD) => {
                                            state.camera_pan[0] += 200.0 / state.camera_zoom;
                                            let sim_size = state.sim_size;
                                            state.camera_pan[0] = state.camera_pan[0].clamp(-0.25 * sim_size, 1.25 * sim_size);
                                            state.follow_selected_agent = false;
                                            camera_changed = true;
                                        }
                                        PhysicalKey::Code(KeyCode::KeyR) => {
                                            // Reset camera
                                            state.camera_zoom = 1.0;
                                            let sim_size = state.sim_size;
                                            state.camera_pan = [sim_size / 2.0, sim_size / 2.0];
                                            camera_changed = true;
                                        }
                                        PhysicalKey::Code(KeyCode::Space) => {
                                            state.ui_visible = !state.ui_visible;
                                        }
                                        PhysicalKey::Code(KeyCode::F12) => {
                                            state.screenshot_4k_requested = true;
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
                            if let Some(state) = self.state.as_mut() {
                                if button == winit::event::MouseButton::Right {
                                    state.is_dragging = button_state == ElementState::Pressed;
                                    if !state.is_dragging {
                                        state.last_mouse_pos = None;
                                    }
                                } else if button == winit::event::MouseButton::Middle {
                                    state.is_zoom_dragging = button_state == ElementState::Pressed;
                                    if state.is_zoom_dragging {
                                        if let Some(mouse_pos) = state.last_mouse_pos {
                                            state.zoom_drag_start_y = mouse_pos[1];
                                            state.zoom_drag_start_zoom = state.camera_zoom;
                                        }
                                    }
                                } else if button == winit::event::MouseButton::Left
                                    && button_state == ElementState::Pressed
                                {
                                    if let Some(mouse_pos) = state.last_mouse_pos {
                                        if state.spawn_mode_active && state.spawn_template_genome.is_some() {
                                            // Spawn mode - create agent at cursor position
                                            state.spawn_agent_at_cursor(mouse_pos);
                                        } else {
                                            // Normal mode - select agent for debug panel
                                            state.select_agent_at_screen_pos(mouse_pos);
                                        }
                                    }
                                }
                            }
                        }
                        WindowEvent::CursorMoved { position, .. } => {
                            if let Some(state) = self.state.as_mut() {
                                let current_pos = [position.x as f32, position.y as f32];

                                if state.is_zoom_dragging {
                                    // Middle mouse drag for zoom
                                    let delta_y = state.zoom_drag_start_y - current_pos[1];
                                    // Use exponential scaling: 100 pixels = 2x zoom change
                                    let zoom_factor = (delta_y / 100.0).exp();
                                    state.camera_zoom = (state.zoom_drag_start_zoom * zoom_factor).clamp(0.1, 2000.0);
                                    window.request_redraw();
                                } else if state.is_dragging {
                                    if let Some(last_pos) = state.last_mouse_pos {
                                        let delta_x = current_pos[0] - last_pos[0];
                                        let delta_y = current_pos[1] - last_pos[1];

                                        // Convert screen space delta to world space delta (Y inverted)
                                        let world_scale = (state.surface_config.width as f32 / state.sim_size)
                                            * state.camera_zoom;
                                        state.camera_pan[0] -= delta_x / world_scale;
                                        state.camera_pan[1] += delta_y / world_scale; // Inverted Y

                                        // Clamp camera position to -0.25 to 1.25 of world size
                                        let sim_size = state.sim_size;
                                        state.camera_pan[0] = state.camera_pan[0].clamp(-0.25 * sim_size, 1.25 * sim_size);
                                        state.camera_pan[1] = state.camera_pan[1].clamp(-0.25 * sim_size, 1.25 * sim_size);

                                        state.follow_selected_agent = false;
                                        window.request_redraw();
                                    }
                                }
                                // Always update mouse position (for both dragging and click selection)
                                state.last_mouse_pos = Some(current_pos);
                            }
                        }
                        WindowEvent::MouseWheel { delta, .. } => {
                            if let Some(state) = self.state.as_mut() {
                                let zoom_delta = match delta {
                                    MouseScrollDelta::LineDelta(_, y) => y * 0.1,
                                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                                };

                                // Zoom inspector ONLY if mouse is over the preview box:
                                // rightmost 300px, and y in [0..300) from the top.
                                let mouse_over_inspector_preview = if let Some(mouse_pos) = state.last_mouse_pos {
                                    let w = state.surface_config.width as f32;
                                    mouse_pos[0] > (w - 300.0) && mouse_pos[1] >= 0.0 && mouse_pos[1] < 300.0
                                } else {
                                    false
                                };

                                if mouse_over_inspector_preview {
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
                                    if let Some(gpu_state) = self.state.as_ref() {
                                        match snapshot_target_resolutions(&path) {
                                            Ok((env_res, fluid_res, spatial_res)) => {
                                                if env_res != gpu_state.env_grid_resolution
                                                    || fluid_res != gpu_state.fluid_grid_resolution
                                                    || spatial_res != gpu_state.spatial_grid_resolution
                                                {
                                                    println!(
                                                        "🔄 Dropped snapshot resolution {}x{} differs from current {}x{}; resetting to match.",
                                                        env_res,
                                                        env_res,
                                                        gpu_state.env_grid_resolution,
                                                        gpu_state.env_grid_resolution
                                                    );
                                                    self.pending_snapshot_load_path = Some(path.clone());
                                                    self.pending_resolution_change_override =
                                                        Some((env_res, fluid_res, spatial_res));
                                                    self.force_reset_requested = true;
                                                    window.request_redraw();
                                                } else {
                                                    reset_simulation_state(
                                                        &mut self.state,
                                                        &window,
                                                        &mut self.egui_state,
                                                    );
                                                    if let Some(gpu_state) = self.state.as_mut() {
                                                        match gpu_state.load_snapshot_from_file(&path) {
                                                            Ok(_) => println!(
                                                                "? Snapshot loaded from: {}",
                                                                path.display()
                                                            ),
                                                            Err(e) => eprintln!(
                                                                "? Failed to load snapshot: {}",
                                                                e
                                                            ),
                                                        }
                                                    }
                                                }
                                            }
                                            Err(e) => eprintln!("? Failed to inspect dropped snapshot: {e:?}"),
                                        }
                                    }
                                }
                            }
                        }
                        WindowEvent::RedrawRequested => {
                            let mut reset_requested = self.force_reset_requested;
                            self.force_reset_requested = false;
                            let mut save_recording_requested = false;

                            // Render loading screen while state is being initialized
                            if self.state.is_none() {
                                window.request_redraw();
                                return; // Don't process further until state is ready
                            }

                            if let Some(state) = self.state.as_mut() {
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

                                let fast_draw_tick = state.render_interval != 0
                                    && (state.epoch % state.render_interval as u64 == 0);

                                // Avoid presenting hundreds of frames/sec in modes that can run uncapped.
                                // Full Speed: cap presents to ~60Hz.
                                // Fast Draw: cap presents to ~60Hz when UI is visible; otherwise only present on draw ticks.
                                let do_present = if state.is_paused {
                                    true
                                } else if state.current_mode == 1 {
                                    state.last_present_time.elapsed()
                                        >= std::time::Duration::from_micros(FULL_SPEED_PRESENT_INTERVAL_MICROS)
                                } else if state.current_mode == 2 {
                                    if state.ui_visible {
                                        state.last_present_time.elapsed()
                                            >= std::time::Duration::from_micros(FULL_SPEED_PRESENT_INTERVAL_MICROS)
                                    } else {
                                        fast_draw_tick
                                    }
                                } else {
                                    true
                                };

                                // Run one simulation step per frame
                                let should_draw = if state.is_paused {
                                    // Always draw when paused so camera movement is visible
                                    true
                                } else if state.current_mode == 2 {
                                    // Fast Draw mode: draw every N steps
                                    fast_draw_tick
                                } else if state.current_mode == 1 {
                                    // Full Speed: only update visualization when we're going to present.
                                    do_present
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

                                    // Sample population for statistics graph
                                    if state.epoch - state.last_sample_epoch >= state.epoch_sample_interval {
                                        state.population_history.push(state.alive_count);
                                        // Keep only the last max_history_points
                                        if state.population_history.len() > state.max_history_points {
                                            state.population_history.remove(0);
                                        }

                                        // Update cached plot points only when data changes (avoids per-frame allocs).
                                        // Downsample to cap egui_plot CPU cost at high FPS.
                                        const MAX_PLOT_POINTS: usize = 1024;
                                        let len = state.population_history.len();
                                        let stride = ((len + MAX_PLOT_POINTS - 1) / MAX_PLOT_POINTS).max(1);
                                        state.population_plot_points = state
                                            .population_history
                                            .iter()
                                            .enumerate()
                                            .step_by(stride)
                                            .map(|(i, &pop)| [i as f64, pop as f64])
                                            .collect();

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

                                if do_present {
                                    let screen_descriptor = ScreenDescriptor {
                                        size_in_pixels: [
                                            state.surface_config.width,
                                            state.surface_config.height,
                                        ],
                                        pixels_per_point: window.scale_factor() as f32,
                                    };

                                    // If the UI is hidden, skip egui entirely (saves CPU at high FPS).
                                    // The UI can be re-enabled via spacebar; next present will rebuild egui.
                                    if !state.ui_visible {
                                        state.last_present_time = now;
                                        match state.render(
                                            &[],
                                            egui::TexturesDelta::default(),
                                            screen_descriptor,
                                        ) {
                                            Ok(_) => {}
                                            Err(wgpu::SurfaceError::Lost) => state.resize(window.inner_size()),
                                            Err(wgpu::SurfaceError::Outdated) => {}
                                            Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
                                            Err(e) => eprintln!("{:?}", e),
                                        }
                                    } else {
                                        // Drain input every present, but only rebuild egui at ~30Hz (or immediately on interaction).
                                        let raw_input = self.egui_state.take_egui_input(&window);
                                        let input_active = !raw_input.events.is_empty();
                                        let do_egui_update = input_active
                                            || state.last_egui_update_time.elapsed()
                                                >= std::time::Duration::from_micros(EGUI_UPDATE_INTERVAL_MICROS);

                                        if do_egui_update {
                                        let full_output = self.egui_state.egui_ctx().run(raw_input, |ctx| {
                                        // Left side panel for simulation controls (only show if ui_visible)
                                        if state.ui_visible {
                                        egui::SidePanel::left("simulation_controls")
                                        .default_width(350.0)
                                        .resizable(true)
                                        .frame(egui::Frame::none()
                                            .fill(egui::Color32::from_rgb(70, 70, 70))
                                            .inner_margin(egui::Margin::same(10.0)))
                                        .show(ctx, |ui| {
                                            // Top section (no tabs) - Always visible (split into 3 rows to keep panel narrow)
                                            ui.vertical(|ui| {
                                                ui.horizontal(|ui| {
                                                    if ui.button(if state.is_paused { "Resume" } else { "Pause" }).clicked() {
                                                        state.is_paused = !state.is_paused;
                                                    }
                                                    ui.separator();
                                                    let mut mode = state.current_mode;
                                                    ui.selectable_value(&mut mode, 3, "Slow");
                                                    ui.selectable_value(&mut mode, 0, "VSync");
                                                    ui.selectable_value(&mut mode, 1, "Full Speed");
                                                    ui.selectable_value(&mut mode, 2, "Fast Draw");
                                                    if mode != state.current_mode {
                                                        state.set_speed_mode(mode);
                                                    }

                                                    ui.separator();
                                                    let reset_button = egui::Button::new("Reset Simulation")
                                                        .fill(egui::Color32::from_rgb(90, 25, 25))
                                                        .stroke(egui::Stroke::new(
                                                            1.0,
                                                            egui::Color32::from_rgb(180, 80, 80),
                                                        ));
                                                    if ui.add(reset_button).clicked() {
                                                        reset_requested = true;
                                                    }
                                                });
                                                if state.current_mode == 2 {
                                                    ui.horizontal(|ui| {
                                                        ui.label("Draw every");
                                                        ui.add(
                                                            egui::Slider::new(&mut state.render_interval, 1..=10000)
                                                                .logarithmic(true)
                                                                .suffix(" steps")
                                                        );
                                                    });
                                                }

                                                ui.separator();
                                                ui.heading("Agents");
                                                ui.horizontal_wrapped(|ui| {
                                                    if ui.button("Spawn 20000 Agents").clicked() && !state.is_paused {
                                                        state.queue_random_spawns(20000);
                                                    }

                                                    if ui.button("Load Agent & Spawn 100...").clicked() && !state.is_paused {
                                                        match load_agent_via_dialog() {
                                                            Ok(genome) => {
                                                                println!(
                                                                    "Successfully loaded genome, spawning 100 clones..."
                                                                );

                                                                let (
                                                                    genome_override_len,
                                                                    genome_override_offset,
                                                                    genome_override_packed,
                                                                ) = genome_pack_ascii_words(&genome);

                                                                for _ in 0..100 {
                                                                    state.rng_state = state
                                                                        .rng_state
                                                                        .wrapping_mul(6364136223846793005u64)
                                                                        .wrapping_add(1442695040888963407u64);
                                                                    let seed = (state.rng_state >> 32) as u32;

                                                                    state.rng_state = state
                                                                        .rng_state
                                                                        .wrapping_mul(6364136223846793005u64)
                                                                        .wrapping_add(1442695040888963407u64);
                                                                    let genome_seed = (state.rng_state >> 32) as u32;

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
                                                                    let rotation =
                                                                        ((state.rng_state >> 32) as f32
                                                                            / u32::MAX as f32)
                                                                            * std::f32::consts::TAU;

                                                                    let request = SpawnRequest {
                                                                        seed,
                                                                        genome_seed,
                                                                        flags: 1,
                                                                        _pad_seed: 0,
                                                                        position: [
                                                                            rx * state.sim_size,
                                                                            ry * state.sim_size,
                                                                        ],
                                                                        energy: 10.0,
                                                                        rotation,
                                                                        genome_override_len,
                                                                        genome_override_offset,
                                                                        genome_override_packed,
                                                                        _pad_genome: [0u32; 2],
                                                                    };

                                                                    state.cpu_spawn_queue.push(request);
                                                                }

                                                                state.spawn_request_count =
                                                                    state.cpu_spawn_queue.len() as u32;
                                                                state.pending_spawn_upload = true;
                                                                state.window.request_redraw();
                                                                println!("Queued 100 spawn requests");
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

                                                // Spawn mode controls
                                                ui.horizontal_wrapped(|ui| {
                                                    if ui.button("Load Template for Spawning...").clicked() {
                                                        match load_agent_via_dialog() {
                                                            Ok(genome) => {
                                                                state.spawn_template_genome = Some(genome);
                                                                println!("? Template loaded for spawn mode");
                                                            }
                                                            Err(err) => eprintln!("Load canceled or failed: {err:?}"),
                                                        }
                                                    }

                                                    let has_template = state.spawn_template_genome.is_some();
                                                    ui.add_enabled_ui(has_template, |ui| {
                                                        let spawn_mode_text = if state.spawn_mode_active {
                                                            "Spawn Mode ON"
                                                        } else {
                                                            "Enable Spawn Mode"
                                                        };

                                                        if ui.button(spawn_mode_text).clicked() {
                                                            state.spawn_mode_active = !state.spawn_mode_active;
                                                            if state.spawn_mode_active {
                                                                println!(
                                                                    "? Spawn mode enabled - click to spawn agents"
                                                                );
                                                            } else {
                                                                println!("? Spawn mode disabled");
                                                            }
                                                        }
                                                    });

                                                    if has_template && ui.button("Clear Template").clicked() {
                                                        state.spawn_template_genome = None;
                                                        state.spawn_mode_active = false;
                                                        println!("? Template cleared");
                                                    }
                                                });

                                                if state.spawn_mode_active {
                                                    ui.colored_label(
                                                        egui::Color32::from_rgb(100, 255, 100),
                                                        "Click anywhere to spawn agent",
                                                    );
                                                } else if state.spawn_template_genome.is_some() {
                                                    ui.colored_label(
                                                        egui::Color32::from_rgb(200, 200, 100),
                                                        "Template loaded - enable spawn mode to use",
                                                    );
                                                }

                                                ui.separator();
                                                ui.horizontal(|ui| {
                                                    if ui
                                                        .button("📸 Screenshot")
                                                        .on_hover_text("Capture current view as JPEG")
                                                        .clicked()
                                                    {
                                                        state.screenshot_requested = true;
                                                    }
                                                    if ui
                                                        .button("🖼️ 4K Screenshot")
                                                        .on_hover_text("Capture a full-world 4096×4096 PNG (tiled render)")
                                                        .clicked()
                                                    {
                                                        state.screenshot_4k_requested = true;
                                                    }
                                                });

                                                ui.horizontal(|ui| {
                                                    if ui
                                                        .button(if state.recording_bar_visible { "🎬 Hide Recording" } else { "🎬 Show Recording" })
                                                        .on_hover_text("Show/hide the floating recording controls")
                                                        .clicked()
                                                    {
                                                        state.recording_bar_visible = !state.recording_bar_visible;
                                                    }

                                                    if ui
                                                        .button("Overrides...")
                                                        .on_hover_text("Edit part & organ property overrides")
                                                        .clicked()
                                                    {
                                                        state.show_part_properties_editor = true;
                                                    }
                                                });
                                            });

                                            ui.separator();
                                            ui.heading("Settings");
                                            ui.horizontal(|ui| {
                                                ui.checkbox(&mut state.fluid_enabled, "Fluids");
                                                ui.checkbox(&mut state.microswim_enabled, "Microswimming");
                                                ui.checkbox(&mut state.propellers_enabled, "Propellers");
                                            });
                                            ui.horizontal(|ui| {
                                                if ui.button("Save Settings").clicked() {
                                                    if let Some(path) = rfd::FileDialog::new()
                                                        .set_file_name("simulation_settings.json")
                                                        .add_filter("JSON", &["json"])
                                                        .save_file()
                                                    {
                                                        let settings = state.current_settings();
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
                                                            state.apply_settings(&settings);
                                                        } else {
                                                            eprintln!("Failed to load settings from {}", path.display());
                                                        }
                                                    }
                                                }
                                            });

                                            ui.horizontal(|ui| {
                                                if ui.button("📷 Save Snapshot").clicked() {
                                                    state.snapshot_save_requested = true;
                                                }
                                                if ui.button("📂 Load Snapshot").clicked() {
                                                    state.snapshot_load_requested = true;
                                                }
                                            });

                                            ui.separator();
                                            ui.heading("Resolution");
                                            ui.horizontal(|ui| {
                                                ui.label(format!("Current: {}×{}", state.env_grid_resolution, state.env_grid_resolution));
                                                ui.separator();
                                                if ui.button("2048×2048").clicked() {
                                                    state.pending_resolution_change = Some(2048);
                                                }
                                                if ui.button("1024×1024").clicked() {
                                                    state.pending_resolution_change = Some(1024);
                                                }
                                                if ui.button("512×512").clicked() {
                                                    state.pending_resolution_change = Some(512);
                                                }
                                            });
                                            if state.pending_resolution_change.is_some() {
                                                ui.colored_label(
                                                    egui::Color32::from_rgb(255, 200, 100),
                                                    format!("⚡ Will reset to {}×{} resolution",
                                                        state.pending_resolution_change.unwrap(),
                                                        state.pending_resolution_change.unwrap()
                                                    )
                                                );
                                            }

                                            ui.separator();
                                            ui.collapsing("Population History", |ui| {
                                                ui.label(format!(
                                                    "Samples: {} (every {} epochs)",
                                                    state.population_history.len(),
                                                    state.epoch_sample_interval
                                                ));

                                                if !state.population_plot_points.is_empty() {
                                                    use egui_plot::{Line, Plot};

                                                    let line = Line::new(state.population_plot_points.clone())
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
                                            // Tab selection for detailed controls with colored buttons
                                            ui.horizontal(|ui| {
                                                let tab_colors = [
                                                    ("Agents", egui::Color32::from_rgb(55, 50, 60)),
                                                    ("Env", egui::Color32::from_rgb(50, 60, 55)),
                                                    ("EnvEvolution", egui::Color32::from_rgb(60, 55, 50)),
                                                    ("Microswimming", egui::Color32::from_rgb(50, 55, 60)),
                                                    ("Fluid", egui::Color32::from_rgb(50, 55, 60)),
                                                    ("Viz", egui::Color32::from_rgb(55, 55, 55)),
                                                ];

                                                // Guard against stale indices.
                                                if state.ui_tab >= tab_colors.len() {
                                                    state.ui_tab = 0;
                                                }

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
                                                0 => egui::Color32::from_rgb(55, 50, 60),  // Agents - purple-gray
                                                1 => egui::Color32::from_rgb(50, 60, 55),  // Env - green-gray
                                                2 => egui::Color32::from_rgb(60, 55, 50),  // EnvEvolution - orange-gray
                                                3 => egui::Color32::from_rgb(50, 55, 60),  // Microswimming
                                                4 => egui::Color32::from_rgb(50, 55, 60),  // Fluid - blue-gray
                                                5 => egui::Color32::from_rgb(55, 55, 55),  // Viz - neutral gray
                                                _ => egui::Color32::from_rgb(50, 50, 50),
                                            };

                                            // Fill the background of the remaining space
                                            let remaining_rect = ui.available_rect_before_wrap();
                                            ui.painter().rect_filled(remaining_rect, 0.0, tab_color);

                                            // Popup window (rendered regardless of active tab).
                                            ui_part_properties_editor_popup(ui, state);

                                            match state.ui_tab {
                                                0 => {
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
                                                            let denom = state.alive_count.max(1) as f32;
                                                            let pct = (state.organ45_alive_with as f32 / denom) * 100.0;
                                                            ui.label(format!(
                                                                "Attractor/Repulsor (45): {}/{} ({:.2}%)",
                                                                state.organ45_alive_with,
                                                                state.alive_count,
                                                                pct
                                                            ));
                                                            if ui.button("Scan").clicked() {
                                                                state.scan_population_for_organ45();
                                                            }
                                                        });
                                                        if let Some(t) = state.organ45_last_scan {
                                                            ui.label(format!(
                                                                "Last scan: {:.1}s ago",
                                                                t.elapsed().as_secs_f32()
                                                            ));
                                                        }
                                                        if let Some(err) = &state.organ45_last_scan_error {
                                                            ui.colored_label(
                                                                egui::Color32::from_rgb(255, 120, 120),
                                                                err,
                                                            );
                                                        }
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

                                                                        let (
                                                                            genome_override_len,
                                                                            genome_override_offset,
                                                                            genome_override_packed,
                                                                        ) = genome_pack_ascii_words(&genome);

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
                                                                                    rx * state.sim_size,
                                                                                    ry * state.sim_size,
                                                                                ],
                                                                                energy: 10.0,
                                                                                rotation,
                                                                                genome_override_len,
                                                                                genome_override_offset,
                                                                                genome_override_packed,
                                                                                _pad_genome: [0u32; 2],
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
                                                                0.0..=5.0,
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
                                                            egui::Slider::new(
                                                                &mut state.morphology_change_cost,
                                                                0.0..=1.0,
                                                            )
                                                            .text("Morphology Change Cost")
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
                                                    });
                                                }
                                                1 => {
                                                    // Env tab
                                                    egui::ScrollArea::vertical().show(ui, |ui| {
                                                        ui.heading("Environment Scheduling");
                                                        ui.add(
                                                            egui::Slider::new(&mut state.diffusion_interval, 1..=64)
                                                                .text("Update env from fluid every N steps"),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.slope_interval, 1..=64)
                                                                .text("Rebuild slope every N steps"),
                                                        );

                                                        ui.separator();
                                                        ui.heading("Classic Diffusion");
                                                        ui.add(
                                                            egui::Slider::new(&mut state.alpha_blur, 0.0..=1.0)
                                                                .text("Alpha Diffuse")
                                                                .custom_formatter(|n, _| format!("{:.3}", n)),
                                                        );
                                                        ui.label("(controls 3�3 blur strength per update)");
                                                        ui.add(
                                                            egui::Slider::new(&mut state.beta_blur, 0.0..=1.0)
                                                                .text("Beta Diffuse")
                                                                .custom_formatter(|n, _| format!("{:.3}", n)),
                                                        );
                                                        ui.label("(controls 3�3 blur strength per update)");

                                                        ui.add(
                                                            egui::Slider::new(&mut state.gamma_diffuse, 0.0..=1.0)
                                                                .text("Gamma Diffuse")
                                                                .custom_formatter(|n, _| format!("{:.3}", n)),
                                                        );
                                                        ui.label("(controls 3�3 blur strength per update)");

                                                        ui.add(
                                                            egui::Slider::new(&mut state.gamma_shift, 0.0..=1.0)
                                                                .text("Gamma Shift Strength"),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.gamma_blur, 0.9..=1.0)
                                                                .text("Env Persistence"),
                                                        );

                                                        ui.add(
                                                            egui::Slider::new(&mut state.fluid_ooze_rate, 0.0..=100.0)
                                                                .text("Lift Min Speed"),
                                                        );

                                                        ui.add(
                                                            egui::Slider::new(
                                                                &mut state.fluid_ooze_fade_rate,
                                                                0.0..=50.0,
                                                            )
                                                            .text("Lift Multiplier"),
                                                        );

                                                        ui.add(
                                                            egui::Slider::new(&mut state.fluid_ooze_rate_beta, 0.0..=100.0)
                                                                .text("Sedimentation Min Speed"),
                                                        );

                                                        ui.add(
                                                            egui::Slider::new(
                                                                &mut state.fluid_ooze_fade_rate_beta,
                                                                0.0..=50.0,
                                                            )
                                                            .text("Sedimentation Multiplier"),
                                                        );

                                                        ui.add(
                                                            egui::Slider::new(
                                                                &mut state.fluid_ooze_still_rate,
                                                                0.0..=100.0,
                                                            )
                                                            .text("Chem Ooze Rate (Still)"),
                                                        );

                                                        ui.add(
                                                            egui::Slider::new(
                                                                &mut state.fluid_dye_escape_rate,
                                                                0.0..=20.0,
                                                            )
                                                            .text("Dye Escape Rate (Alpha)")
                                                            .custom_formatter(|n, _| format!("{:.3}", n)),
                                                        );

                                                        ui.add(
                                                            egui::Slider::new(
                                                                &mut state.fluid_dye_escape_rate_beta,
                                                                0.0..=20.0,
                                                            )
                                                            .text("Dye Escape Rate (Beta)")
                                                            .custom_formatter(|n, _| format!("{:.3}", n)),
                                                        );

                                                        ui.separator();
                                                        ui.heading("Slope Mixing");
                                                        ui.add(
                                                            egui::Slider::new(
                                                                &mut state.chemical_slope_scale_alpha,
                                                                0.0..=1.0,
                                                            )
                                                            .text("Alpha Slope Mix"),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(
                                                                &mut state.chemical_slope_scale_beta,
                                                                0.0..=1.0,
                                                            )
                                                            .text("Beta Slope Mix"),
                                                        );

                                                        ui.separator();
                                                        ui.heading("Slope Bias");
                                                        ui.add(
                                                            egui::Slider::new(&mut state.alpha_slope_bias, -10.0..=10.0)
                                                            .text("Alpha Slope Bias")
                                                            .custom_formatter(|n, _| format!("{:.2}", n)),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.beta_slope_bias, -10.0..=10.0)
                                                            .text("Beta Slope Bias")
                                                            .custom_formatter(|n, _| format!("{:.2}", n)),
                                                        );
                                                        ui.label("(bias strength for slope-based shifting; sign flips direction)");

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
                                                            egui::Slider::new(&mut state.alpha_multiplier, 0.00001..=2.0)
                                                                .text("Rain Probability")
                                                                .logarithmic(true)
                                                                .custom_formatter(|n, _| format!("{:.6}", n)),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.dye_precipitation, 0.0..=1.0)
                                                                .text("Dye Precipitation"),
                                                        )
                                                        .on_hover_text("Scales rain/precipitation injection into the chem grids. 0 disables precipitation entirely.");
                                                        ui.checkbox(&mut state.rain_debug_visual, "=�Ŀ Show Rain Pattern");
                                                        if state.rain_debug_visual {
                                                            ui.label("=��� Green = Alpha (food) | =��� Red = Beta (poison)");
                                                        }
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
                                                            egui::Slider::new(&mut state.beta_multiplier, 0.00001..=2.0)
                                                                .text("Rain Probability")
                                                                .logarithmic(true)
                                                                .custom_formatter(|n, _| format!("{:.6}", n)),
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

                                                        ui.separator();
                                                        ui.label("Performance Toggles:");
                                                        ui.checkbox(&mut state.perf_skip_diffusion, "⚡ Skip Diffusion (Debug)");
                                                        ui.checkbox(&mut state.perf_skip_rain, "🌧️ Skip Rain Drops (Debug)");
                                                        ui.checkbox(&mut state.perf_skip_trail_prep, "🎨 Skip Trail Prep (Debug)");
                                                        ui.checkbox(&mut state.perf_skip_slope, "🏔️ Skip Slope Calc (Debug)");
                                                        ui.checkbox(&mut state.perf_skip_draw, "🖼️ Skip Visual Clear (Debug)");
                                                        ui.checkbox(&mut state.perf_skip_repro, "🧬 Skip Reproduction (Debug)");
                                                        ui.checkbox(&mut state.frame_profiler.dispatch_timing_enabled, "📊 Show Dispatch Timing");
                                                        if state.frame_profiler.dispatch_timing_enabled {
                                                            ui.checkbox(&mut state.frame_profiler.dispatch_timing_detail, "  └─ Detailed SimPre Breakdown");
                                                        }
                                                        ui.checkbox(
                                                            &mut state.perf_force_gpu_sync,
                                                            "⏱️ Force GPU Sync (Accurate FPS)",
                                                        );

                                                        let min_changed = ui
                                                            .add(
                                                                egui::Slider::new(
                                                                    &mut state.gamma_vis_min,
                                                                    0.0..=100_000.0,
                                                                )
                                                                .text("Gamma Min"),
                                                            )
                                                            .changed();
                                                        let max_changed = ui
                                                            .add(
                                                                egui::Slider::new(
                                                                    &mut state.gamma_vis_max,
                                                                    0.0..=100_000.0,
                                                                )
                                                                .text("Gamma Max"),
                                                            )
                                                            .changed();
                                                        if state.gamma_vis_min >= state.gamma_vis_max {
                                                            state.gamma_vis_max =
                                                                (state.gamma_vis_min + 0.001).min(100_000.0);
                                                            state.gamma_vis_min =
                                                                (state.gamma_vis_max - 0.001).max(0.0);
                                                        } else if min_changed || max_changed {
                                                            state.gamma_vis_min =
                                                                state.gamma_vis_min.clamp(0.0, 100_000.0);
                                                            state.gamma_vis_max =
                                                                state.gamma_vis_max.clamp(0.0, 100_000.0);
                                                        }
                                                        ui.label(
                                                            "Slope force scales with the Obstacle / Terrain Force slider.",
                                                        );
                                                    });
                                                }
                                                2 => {
                                                    // EnvEvolution tab
                                                    egui::ScrollArea::vertical().show(ui, |ui| {
                                                        ui.heading("Rain Cycling");

                                                        let current_alpha = state.alpha_rain_history.back().copied().unwrap_or(state.alpha_multiplier);
                                                        let current_beta = state.beta_rain_history.back().copied().unwrap_or(state.beta_multiplier);

                                                        // Calculate base values with difficulty (but without cycle modulation) for display
                                                        let base_alpha_with_difficulty = state.difficulty.alpha_rain.apply_to(state.alpha_multiplier, false);
                                                        let base_beta_with_difficulty = state.difficulty.beta_rain.apply_to(state.beta_multiplier, true);

                                                        ui.separator();
                                                        ui.heading("Alpha Rain");
                                                        ui.label(format!("Base value (saved): {:.6}", state.alpha_multiplier));
                                                        ui.label(format!("After difficulty: {:.6}", base_alpha_with_difficulty));
                                                        ui.label(egui::RichText::new(format!("Current (with cycle): {:.6}", current_alpha)).color(Color32::GREEN));
                                                        let mut alpha_var_percent = state.alpha_rain_variation * 100.0;
                                                        if ui.add(egui::Slider::new(&mut alpha_var_percent, 0.0..=100.0).text("Variation %")).changed() {
                                                            state.alpha_rain_variation = alpha_var_percent / 100.0;
                                                        }
                                                        ui.add(egui::Slider::new(&mut state.alpha_rain_phase, 0.0..=std::f32::consts::PI * 2.0).text("Phase"));
                                                        ui.add(egui::Slider::new(&mut state.alpha_rain_freq, 0.0..=10.0).text("Freq (cycles/1k)"));

                                                        ui.separator();
                                                        ui.heading("Beta Rain");
                                                        ui.label(format!("Base value (saved): {:.6}", state.beta_multiplier));
                                                        ui.label(format!("After difficulty: {:.6}", base_beta_with_difficulty));
                                                        ui.label(egui::RichText::new(format!("Current (with cycle): {:.6}", current_beta)).color(Color32::RED));
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
                                                        let points = 1000;
                                                        let alpha_points: PlotPoints = (0..points).map(|i| {
                                                            let t = time + i as f64 * 10.0;
                                                            let freq = state.alpha_rain_freq as f64 / 1000.0;
                                                            let phase = state.alpha_rain_phase as f64;
                                                            let sin_val = (t * freq * 2.0 * std::f64::consts::PI + phase).sin();
                                                            let val = base_alpha as f64 * (1.0 + sin_val * state.alpha_rain_variation as f64).max(0.0);
                                                            [t, val]
                                                        }).collect();

                                                        let beta_points: PlotPoints = (0..points).map(|i| {
                                                            let t = time + i as f64 * 10.0;
                                                            let freq = state.beta_rain_freq as f64 / 1000.0;
                                                            let phase = state.beta_rain_phase as f64;
                                                            let sin_val = (t * freq * 2.0 * std::f64::consts::PI + phase).sin();
                                                            let val = base_beta as f64 * (1.0 + sin_val * state.beta_rain_variation as f64).max(0.0);
                                                            [t, val]
                                                        }).collect();

                                                        Plot::new("rain_plot_future")
                                                            .view_aspect(2.0)
                                                            .auto_bounds([true, true].into())
                                                            .include_x(time)
                                                            .include_x(time + 10000.0)
                                                            .allow_drag(false)
                                                            .allow_zoom([false, false])
                                                            .allow_scroll(false)
                                                            .show(ui, |plot_ui| {
                                                                plot_ui.line(Line::new(alpha_points).name("Alpha").color(Color32::GREEN));
                                                                plot_ui.line(Line::new(beta_points).name("Beta").color(Color32::RED));
                                                            });

                                                        ui.separator();
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
                                                        let draw_param = |ui: &mut egui::Ui,
                                                                          param: &mut AutoDifficultyParam,
                                                                          name: &str,
                                                                          current_val: f32,
                                                                          current_epoch: u64| {
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

                                                                let epochs_passed =
                                                                    current_epoch.saturating_sub(param.last_adjustment_epoch);
                                                                if epochs_passed < param.cooldown_epochs {
                                                                    let remaining = param.cooldown_epochs - epochs_passed;
                                                                    ui.label(format!("Cooldown: {} epochs", remaining));
                                                                    ui.add(egui::ProgressBar::new(
                                                                        epochs_passed as f32 / param.cooldown_epochs as f32,
                                                                    ));
                                                                }
                                                            }
                                                        };

                                                        // Calculate effective values with difficulty applied
                                                        let effective_food_power =
                                                            state.difficulty.food_power.apply_to(state.food_power, false);
                                                        let effective_poison_power =
                                                            state.difficulty.poison_power.apply_to(state.poison_power, true);
                                                        let effective_spawn_prob = state
                                                            .difficulty
                                                            .spawn_prob
                                                            .apply_to(state.spawn_probability, false);
                                                        let effective_death_prob =
                                                            state.difficulty.death_prob.apply_to(state.death_probability, true);
                                                        let effective_alpha_rain = state
                                                            .difficulty
                                                            .alpha_rain
                                                            .apply_to(state.alpha_multiplier, false);
                                                        let effective_beta_rain =
                                                            state.difficulty.beta_rain.apply_to(state.beta_multiplier, true);

                                                        draw_param(
                                                            ui,
                                                            &mut state.difficulty.food_power,
                                                            "Food Power (Harder = Less)",
                                                            effective_food_power,
                                                            current_epoch,
                                                        );
                                                        draw_param(
                                                            ui,
                                                            &mut state.difficulty.poison_power,
                                                            "Poison Power (Harder = More)",
                                                            effective_poison_power,
                                                            current_epoch,
                                                        );
                                                        draw_param(
                                                            ui,
                                                            &mut state.difficulty.spawn_prob,
                                                            "Spawn Prob (Harder = Less)",
                                                            effective_spawn_prob,
                                                            current_epoch,
                                                        );
                                                        draw_param(
                                                            ui,
                                                            &mut state.difficulty.death_prob,
                                                            "Death Prob (Harder = More)",
                                                            effective_death_prob,
                                                            current_epoch,
                                                        );
                                                        draw_param(
                                                            ui,
                                                            &mut state.difficulty.alpha_rain,
                                                            "Alpha Rain (Harder = Less)",
                                                            effective_alpha_rain,
                                                            current_epoch,
                                                        );
                                                        draw_param(
                                                            ui,
                                                            &mut state.difficulty.beta_rain,
                                                            "Beta Rain (Harder = More)",
                                                            effective_beta_rain,
                                                            current_epoch,
                                                        );
                                                    });
                                                }
                                                5 => {
                                                    // Viz tab
                                                    egui::ScrollArea::vertical().show(ui, |ui| {
                                                        ui.heading("Viz Settings");

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
                                                                .clamping(egui::SliderClamping::Always)
                                                        ).on_hover_text("0.0 = amino acid colors only, 1.0 = agent color only");

                                                        ui.separator();
                                                        ui.label("Motion Blur / Trail:");
                                                        ui.add(
                                                            egui::Slider::new(&mut state.agent_trail_decay, 0.0..=1.0)
                                                                .text("Trail Decay")
                                                                .clamping(egui::SliderClamping::Always)
                                                        ).on_hover_text("0.0 = persistent trail, 1.0 = instant clear (no trail)");

                                                        ui.separator();
                                                        ui.heading("Trail Layer");
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
                                                                0.0..=1.0,
                                                            )
                                                            .text("Trail Fade Rate"),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(
                                                                &mut state.trail_opacity,
                                                                0.0..=200.0,
                                                            )
                                                            .text("Trail Opacity"),
                                                        );
                                                        ui.checkbox(
                                                            &mut state.trail_show,
                                                            "Show Trails Only (overrides alpha/beta/gamma; black background)",
                                                        );
                                                        ui.add_enabled(
                                                            state.trail_show,
                                                            egui::Checkbox::new(
                                                                &mut state.trail_show_energy,
                                                                "Show Energy Trail (instead of trail color)",
                                                            ),
                                                        );
                                                        if state.trail_show {
                                                            ui.label(
                                                                "Note: This mode intentionally hides the other layers. Turn it off to see alpha/beta/gamma (and the fluid overlay) again.",
                                                            );
                                                        }

                                                        ui.separator();
                                                        ui.heading("Fluid Simulation");
                                                        ui.checkbox(&mut state.fluid_show, "Show Fluid Overlay")
                                                            .on_hover_text("Visualize propeller-driven fluid simulation (experimental)");

                                                        ui.add_space(4.0);
                                                        ui.label("Fluid Dye Compositing");
                                                        ui.label("Note: dye uses independent colors, but reuses Alpha/Beta blend + gamma controls.");

                                                        ui.separator();
                                                        ui.label("Dye Alpha");
                                                        ui.horizontal(|ui| {
                                                            ui.radio_value(&mut state.alpha_blend_mode, 0, "Additive");
                                                            ui.radio_value(&mut state.alpha_blend_mode, 1, "Multiply");
                                                        });
                                                        ui.checkbox(&mut state.dye_alpha_thinfilm, "Thin Film Interference");
                                                        ui.add_enabled(
                                                            state.dye_alpha_thinfilm,
                                                            egui::Slider::new(
                                                                &mut state.dye_alpha_thinfilm_mult,
                                                                0.001..=10_000.0,
                                                            )
                                                            .text("Thin Film Multiplier")
                                                            .logarithmic(true),
                                                        );
                                                        ui.horizontal(|ui| {
                                                            ui.label("Color:");
                                                            ui.add_enabled_ui(!state.dye_alpha_thinfilm, |ui| {
                                                                let mut alpha_color = [
                                                                    (state.dye_alpha_color[0] * 255.0) as u8,
                                                                    (state.dye_alpha_color[1] * 255.0) as u8,
                                                                    (state.dye_alpha_color[2] * 255.0) as u8,
                                                                ];
                                                                if ui.color_edit_button_srgb(&mut alpha_color).changed() {
                                                                    state.dye_alpha_color = [
                                                                        alpha_color[0] as f32 / 255.0,
                                                                        alpha_color[1] as f32 / 255.0,
                                                                        alpha_color[2] as f32 / 255.0,
                                                                    ];
                                                                }
                                                            });
                                                        });
                                                        ui.add(
                                                            egui::Slider::new(&mut state.alpha_gamma_adjust, 0.1..=5.0)
                                                                .text("Gamma Adjustment")
                                                                .logarithmic(true),
                                                        );

                                                        ui.separator();
                                                        ui.label("Dye Beta");
                                                        ui.horizontal(|ui| {
                                                            ui.radio_value(&mut state.beta_blend_mode, 0, "Additive");
                                                            ui.radio_value(&mut state.beta_blend_mode, 1, "Multiply");
                                                        });
                                                        ui.checkbox(&mut state.dye_beta_thinfilm, "Thin Film Interference");
                                                        ui.add_enabled(
                                                            state.dye_beta_thinfilm,
                                                            egui::Slider::new(
                                                                &mut state.dye_beta_thinfilm_mult,
                                                                0.001..=10_000.0,
                                                            )
                                                            .text("Thin Film Multiplier")
                                                            .logarithmic(true),
                                                        );
                                                        ui.horizontal(|ui| {
                                                            ui.label("Color:");
                                                            ui.add_enabled_ui(!state.dye_beta_thinfilm, |ui| {
                                                                let mut beta_color = [
                                                                    (state.dye_beta_color[0] * 255.0) as u8,
                                                                    (state.dye_beta_color[1] * 255.0) as u8,
                                                                    (state.dye_beta_color[2] * 255.0) as u8,
                                                                ];
                                                                if ui.color_edit_button_srgb(&mut beta_color).changed() {
                                                                    state.dye_beta_color = [
                                                                        beta_color[0] as f32 / 255.0,
                                                                        beta_color[1] as f32 / 255.0,
                                                                        beta_color[2] as f32 / 255.0,
                                                                    ];
                                                                }
                                                            });
                                                        });
                                                        ui.add(
                                                            egui::Slider::new(&mut state.beta_gamma_adjust, 0.1..=5.0)
                                                                .text("Gamma Adjustment")
                                                                .logarithmic(true),
                                                        );
                                                    });
                                                }
                                                3 => {
                                                    // Microswimming tab
                                                    egui::ScrollArea::vertical().show(ui, |ui| {
                                                        ui.heading("Microswimming");

                                                        ui.checkbox(&mut state.microswim_enabled, "Enable microswimming");

                                                        ui.separator();
                                                        ui.heading("Strength");
                                                        ui.add(
                                                            egui::Slider::new(&mut state.microswim_coupling, 0.0..=10.0)
                                                                .text("Swim Strength")
                                                                .clamping(egui::SliderClamping::Always),
                                                        )
                                                        .on_hover_text(
                                                            "Overall multiplier for the microswim compute pass (independent of propellers).",
                                                        );

                                                        ui.separator();
                                                        ui.heading("Model");
                                                        ui.add(
                                                            egui::Slider::new(&mut state.microswim_base_drag, 0.0..=5.0)
                                                                .text("Base Drag")
                                                                .clamping(egui::SliderClamping::Always),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.microswim_anisotropy, 0.0..=50.0)
                                                                .text("Anisotropy")
                                                                .logarithmic(true)
                                                                .clamping(egui::SliderClamping::Always),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.microswim_max_frame_vel, 0.0..=20.0)
                                                                .text("Max Frame Velocity")
                                                                .logarithmic(true)
                                                                .clamping(egui::SliderClamping::Always),
                                                        );

                                                        ui.separator();
                                                        ui.heading("Turning");
                                                        ui.add(
                                                            egui::Slider::new(&mut state.microswim_torque_strength, 0.0..=5.0)
                                                                .text("Torque Strength")
                                                                .logarithmic(true)
                                                                .clamping(egui::SliderClamping::Always),
                                                        );

                                                        ui.separator();
                                                        ui.heading("Stability Filters");
                                                        ui.add(
                                                            egui::Slider::new(&mut state.microswim_min_seg_displacement, 0.0..=0.5)
                                                                .text("Min Segment Displacement")
                                                                .logarithmic(true)
                                                                .clamping(egui::SliderClamping::Always),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.microswim_min_total_deformation_sq, 0.0..=10.0)
                                                                .text("Min Total Deformation (sq)")
                                                                .logarithmic(true)
                                                                .clamping(egui::SliderClamping::Always),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.microswim_min_length_ratio, 0.0..=5.0)
                                                                .text("Min Length Ratio")
                                                                .clamping(egui::SliderClamping::Always),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.microswim_max_length_ratio, 0.0..=5.0)
                                                                .text("Max Length Ratio")
                                                                .clamping(egui::SliderClamping::Always),
                                                        );
                                                    });
                                                }
                                                4 => {
                                                    // Fluid tab
                                                    egui::ScrollArea::vertical().show(ui, |ui| {
                                                        ui.heading("Fluid Configuration");

                                                        ui.horizontal(|ui| {
                                                            ui.checkbox(&mut state.fluid_enabled, "Enable Fluids");
                                                        });

                                                        ui.separator();
                                                        ui.heading("Propeller Wash");
                                                        ui.add(
                                                            egui::Slider::new(&mut state.prop_wash_strength, 0.0..=5.0)
                                                                .text("Direct Wash Strength")
                                                                .clamping(egui::SliderClamping::Always),
                                                        )
                                                        .on_hover_text(
                                                            "Scales direct (non-fluid) propeller/displacer wash effects (e.g. chemical transport).",
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.prop_wash_strength_fluid, -5.0..=5.0)
                                                                .text("Fluid Wash Strength")
                                                                .clamping(egui::SliderClamping::Always),
                                                        )
                                                        .on_hover_text(
                                                            "Scales how strongly propellers/displacers inject forces into the fluid. Negative values reverse the direction.",
                                                        );

                                                        ui.separator();
                                                        ui.add(
                                                            egui::Slider::new(&mut state.fluid_dt, 0.001..=0.05)
                                                                .text("dt")
                                                                .clamping(egui::SliderClamping::Always),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.fluid_decay, 0.5..=1.0)
                                                                .text("Decay")
                                                                .clamping(egui::SliderClamping::Always),
                                                        );

                                                        ui.separator();
                                                        ui.horizontal(|ui| {
                                                            ui.label("Pressure Jacobi Iters");
                                                            ui.add(
                                                                egui::DragValue::new(&mut state.fluid_jacobi_iters)
                                                                    .range(1..=128)
                                                                    .speed(1.0),
                                                            );
                                                        });
                                                        ui.label("Higher iters = less compressible (more GPU cost)");

                                                        ui.separator();
                                                        ui.add(
                                                            egui::Slider::new(&mut state.fluid_vorticity, 0.0..=50.0)
                                                                .text("Vorticity")
                                                                .clamping(egui::SliderClamping::Always),
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.fluid_viscosity, 0.0..=5.0)
                                                                .text("Viscosity")
                                                                .clamping(egui::SliderClamping::Always),
                                                        );

                                                        ui.separator();
                                                        ui.heading("Dye Diffusion");
                                                        ui.add(
                                                            egui::Slider::new(&mut state.dye_diffusion, 0.0..=1.0)
                                                                .text("Dye Diffusion (Fluid On)")
                                                                .clamping(egui::SliderClamping::Always),
                                                        )
                                                        .on_hover_text(
                                                            "Dye diffusion strength when fluid simulation is enabled (blend fraction per step).",
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(&mut state.dye_diffusion_no_fluid, 0.0..=1.0)
                                                                .text("Dye Diffusion (Fluid Off)")
                                                                .clamping(egui::SliderClamping::Always),
                                                        )
                                                        .on_hover_text(
                                                            "Dye diffusion strength when fluid simulation is disabled (per epoch).",
                                                        );

                                                        ui.separator();
                                                        ui.heading("Agent Wind");
                                                        ui.add(
                                                            egui::Slider::new(
                                                                &mut state.fluid_wind_push_strength,
                                                                0.0..=2.0,
                                                            )
                                                            .text("Fluid Push Strength")
                                                            .clamping(egui::SliderClamping::Always),
                                                        )
                                                        .on_hover_text(
                                                            "Global multiplier for how strongly the fluid vector field pushes agents (in addition to per-amino coupling).",
                                                        );

                                                        ui.separator();
                                                        ui.heading("Terrain Influence");
                                                        ui.add(
                                                            egui::Slider::new(
                                                                &mut state.fluid_slope_force_scale,
                                                                0.0..=500.0,
                                                            )
                                                            .text("Slope Flow Force")
                                                            .clamping(egui::SliderClamping::Always),
                                                        )
                                                        .on_hover_text(
                                                            "How strongly terrain slope drives fluid downhill (0 = no slope flow, 100 = default, higher = stronger downhill flow).",
                                                        );
                                                        ui.add(
                                                            egui::Slider::new(
                                                                &mut state.fluid_obstacle_strength,
                                                                0.0..=1000.0,
                                                            )
                                                            .text("Obstacle Blocking")
                                                            .clamping(egui::SliderClamping::Always),
                                                        )
                                                        .on_hover_text(
                                                            "How strongly steep terrain blocks fluid flow (0 = no blocking, 200 = default, higher = steeper slopes act more like solid walls).",
                                                        );

                                                        ui.separator();
                                                        ui.heading("Fumaroles");
                                                        ui.horizontal(|ui| {
                                                            if ui.button("Add").clicked() {
                                                                if state.fumaroles.len() < MAX_FUMAROLES {
                                                                    state.fumaroles.push(FumaroleSettings::default());
                                                                    state.selected_fumarole_index = state.fumaroles.len() - 1;
                                                                }
                                                            }
                                                            let can_remove = !state.fumaroles.is_empty();
                                                            if ui.add_enabled(can_remove, egui::Button::new("Remove")).clicked() {
                                                                let idx = state
                                                                    .selected_fumarole_index
                                                                    .min(state.fumaroles.len().saturating_sub(1));
                                                                if idx < state.fumaroles.len() {
                                                                    state.fumaroles.remove(idx);
                                                                }
                                                                if state.fumaroles.is_empty() {
                                                                    state.selected_fumarole_index = 0;
                                                                } else {
                                                                    state.selected_fumarole_index = state
                                                                        .selected_fumarole_index
                                                                        .min(state.fumaroles.len() - 1);
                                                                }
                                                            }
                                                            ui.label(format!(
                                                                "{}/{}",
                                                                state.fumaroles.len(),
                                                                MAX_FUMAROLES
                                                            ));
                                                        });

                                                        if state.fumaroles.is_empty() {
                                                            ui.label("No fumaroles configured.");
                                                        } else {
                                                            ui.label("Select:");
                                                            egui::ScrollArea::vertical()
                                                                .max_height(120.0)
                                                                .show(ui, |ui| {
                                                                    for i in 0..state.fumaroles.len() {
                                                                        let on = state.fumaroles[i].enabled;
                                                                        let label = if on {
                                                                            format!("#{} (on)", i + 1)
                                                                        } else {
                                                                            format!("#{} (off)", i + 1)
                                                                        };
                                                                        if ui
                                                                            .selectable_label(state.selected_fumarole_index == i, label)
                                                                            .clicked()
                                                                        {
                                                                            state.selected_fumarole_index = i;
                                                                        }
                                                                    }
                                                                });

                                                            let idx = state
                                                                .selected_fumarole_index
                                                                .min(state.fumaroles.len().saturating_sub(1));
                                                            let fum = &mut state.fumaroles[idx];

                                                            ui.separator();
                                                            ui.checkbox(&mut fum.enabled, "Enabled");
                                                            ui.add(
                                                                egui::Slider::new(&mut fum.x_frac, 0.0..=1.0)
                                                                    .text("X (frac)")
                                                                    .clamping(egui::SliderClamping::Always),
                                                            );
                                                            ui.add(
                                                                egui::Slider::new(&mut fum.y_frac, 0.0..=1.0)
                                                                    .text("Y (frac)")
                                                                    .clamping(egui::SliderClamping::Always),
                                                            );
                                                            ui.add(
                                                                egui::Slider::new(&mut fum.dir_degrees, 0.0..=360.0)
                                                                    .text("Direction (deg)")
                                                                    .clamping(egui::SliderClamping::Always),
                                                            );
                                                            ui.add(
                                                                egui::Slider::new(&mut fum.strength, 0.0..=50_000.0)
                                                                    .text("Strength")
                                                                    .logarithmic(true)
                                                                    .clamping(egui::SliderClamping::Always),
                                                            );
                                                            ui.add(
                                                                egui::Slider::new(&mut fum.spread, 0.0..=FLUID_GRID_SIZE as f32)
                                                                    .text("Spread (cells)")
                                                                    .clamping(egui::SliderClamping::Always),
                                                            );
                                                            ui.add(
                                                                egui::Slider::new(&mut fum.alpha_dye_rate, 0.0..=1000.0)
                                                                    .text("Alpha Dye Rate")
                                                                    .logarithmic(true)
                                                                    .clamping(egui::SliderClamping::Always),
                                                            );
                                                            ui.add(
                                                                egui::Slider::new(&mut fum.beta_dye_rate, 0.0..=1000.0)
                                                                    .text("Beta Dye Rate")
                                                                    .logarithmic(true)
                                                                    .clamping(egui::SliderClamping::Always),
                                                            );
                                                            ui.add(
                                                                egui::Slider::new(&mut fum.variation, 0.0..=1.0)
                                                                    .text("Variation")
                                                                    .clamping(egui::SliderClamping::Always),
                                                            );
                                                        }
                                                    });
                                                }
                                                _ => {}
                                            }
                            });

                                    // One-line HUD over the simulation view.
                                    egui::Area::new(egui::Id::new("sim_hud_line"))
                                        .anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 4.0))
                                        .show(ctx, |ui| {
                                            ui.horizontal(|ui| {
                                                ui.colored_label(
                                                    egui::Color32::WHITE,
                                                    format!(
                                                        "{}  |  Epoch {}  |  {:.1} eps  |  Agents {}  |  Cap {}",
                                                        state.run_name,
                                                        state.epoch,
                                                        state.epochs_per_second,
                                                        state.alive_count,
                                                        state.agent_buffer_capacity
                                                    ),
                                                );
                                            });
                                        });

                                   } // End ui_visible check

                                    // Inspector overlay - energy and pairing bars (always visible when agent selected)
                                    if let Some(agent) = &state.selected_agent_data {
                                        let screen_width = state.surface_config.width as f32;
                                        let inspector_x = screen_width - 300.0;

                                        // Position bars below the agent preview window (preview is 300px tall at top-right).
                                        let preview_height = 300.0;
                                        let bar_spacing = 5.0;
                                        let bars_y = preview_height + bar_spacing;

                                        egui::Area::new(egui::Id::new("inspector_bars"))
                                            .fixed_pos(egui::pos2(inspector_x + 10.0, bars_y))
                                            .show(ctx, |ui| {
                                                ui.set_width(280.0);

                                                // Add dark grey background frame
                                                let frame = egui::Frame::none()
                                                    .fill(egui::Color32::from_rgb(25, 25, 25))
                                                    .inner_margin(egui::Margin::same(8.0));

                                                frame.show(ui, |ui| {

                                                ui.label(format!(
                                                    "{} (gen {}, {} parts, mass {:.2})",
                                                    naming::agent::generate_agent_name(agent),
                                                    agent.generation,
                                                    (agent.body_count as usize).min(MAX_BODY_PARTS),
                                                    agent.total_mass
                                                ));

                                                ui.add_space(6.0);

                                                // Energy bar
                                                ui.vertical(|ui| {
                                                    let energy_ratio = if agent.energy_capacity > 0.0 {
                                                        (agent.energy / agent.energy_capacity).clamp(0.0, 1.0)
                                                    } else {
                                                        0.0
                                                    };
                                                    let energy_color = if energy_ratio > 0.5 {
                                                        egui::Color32::from_rgb(0, 200, 0)
                                                    } else if energy_ratio > 0.25 {
                                                        egui::Color32::from_rgb(200, 200, 0)
                                                    } else {
                                                        egui::Color32::from_rgb(200, 0, 0)
                                                    };

                                                    let progress_bar = egui::ProgressBar::new(energy_ratio)
                                                        .fill(energy_color)
                                                        .animate(false)
                                                        .text(format!(
                                                            "Energy: {:.1}/{:.1}",
                                                            agent.energy, agent.energy_capacity
                                                        ));
                                                    ui.add(progress_bar);
                                                });

                                                ui.add_space(bar_spacing);

                                                // Pairing state bar
                                                ui.vertical(|ui| {
                                                    let full_pairs = agent.gene_length; // Use actual gene length (non-X bases)
                                                    let pairing_ratio = if full_pairs > 0 {
                                                        (agent.pairing_counter as f32 / full_pairs as f32).clamp(0.0, 1.0)
                                                    } else {
                                                        0.0
                                                    };
                                                    let pairing_color = egui::Color32::from_rgb(100, 150, 255);

                                                    let progress_bar = egui::ProgressBar::new(pairing_ratio)
                                                        .fill(pairing_color)
                                                        .animate(false)
                                                        .text(format!(
                                                            "Pairing: {}/{}",
                                                            agent.pairing_counter, full_pairs
                                                        ));
                                                    ui.add(progress_bar);
                                                });

                                                ui.add_space(8.0);

                                                // Genome (moved out of shader): 5x5 squares, 56 per line (280px wide)
                                                {
                                                    let bases_per_line = 56usize;
                                                    let cell = 5.0;
                                                    let lines = (GENOME_BYTES + bases_per_line - 1) / bases_per_line;
                                                    let size = egui::vec2(280.0, lines as f32 * cell);
                                                    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
                                                    let painter = ui.painter_at(rect);

                                                    // Match the simulation's translation rules (shared.wgsl::translate_codon_step):
                                                    // - Optional AUG requirement
                                                    // - Optional stop-codon ignore
                                                    // - Promoters consume 6 bases (promoter+modifier), even if the result is an amino
                                                    // - Translation does not "restart" after a stop
                                                    let genome_ascii = genome_packed_to_ascii_words(
                                                        &agent.genome_packed,
                                                        agent.genome_offset,
                                                        agent.gene_length,
                                                    );
                                                    let translation_map = build_translation_map_for_inspector(
                                                        &genome_ascii,
                                                        (agent.body_count as usize).min(MAX_BODY_PARTS),
                                                        state.require_start_codon,
                                                        state.ignore_stop_codons,
                                                    );

                                                    for idx in 0..GENOME_BYTES {
                                                        let row = idx / bases_per_line;
                                                        let col = idx % bases_per_line;
                                                        let base = genome_get_base_ascii(&genome_ascii, idx);
                                                        let mut color = genome_base_color(base);

                                                        // Darken untranslated regions
                                                        if !translation_map[idx] {
                                                            let (r, g, b, _) = color.to_tuple();
                                                            color = egui::Color32::from_rgb(
                                                                (r as f32 * 0.3) as u8,
                                                                (g as f32 * 0.3) as u8,
                                                                (b as f32 * 0.3) as u8,
                                                            );
                                                        }

                                                        let x0 = rect.left() + col as f32 * cell;
                                                        let y0 = rect.top() + row as f32 * cell;
                                                        let r = egui::Rect::from_min_size(
                                                            egui::pos2(x0, y0),
                                                            egui::vec2(cell, cell),
                                                        );
                                                        painter.rect_filled(r, 0.0, color);
                                                    }
                                                }

                                                ui.add_space(8.0);

                                                // Body part bars: type colors + combined per-part signals
                                                {
                                                    let body_count = (agent.body_count as usize).min(MAX_BODY_PARTS);
                                                    if body_count > 0 {
                                                        let bar_w = 280.0;
                                                        let bar_h = 8.0;

                                                        let genome_ascii = genome_packed_to_ascii_words(
                                                            &agent.genome_packed,
                                                            agent.genome_offset,
                                                            agent.gene_length,
                                                        );
                                                        let nucleotide_spans = compute_body_part_nucleotide_spans_for_inspector(
                                                            &genome_ascii,
                                                            body_count,
                                                            state.require_start_codon,
                                                            state.ignore_stop_codons,
                                                        );

                                                        // Calculate total width needed using the same translation stepping as the sim.
                                                        // This matters because promoters can consume 6 bases even when the emitted
                                                        // part is an amino acid (organ mapping failure fallback).
                                                        let mut total_nucleotides = 0.0;
                                                        for i in 0..body_count {
                                                            total_nucleotides += nucleotide_spans
                                                                .get(i)
                                                                .copied()
                                                                .unwrap_or_else(|| {
                                                                    let base_type = agent.body[i].base_type();
                                                                    if base_type < 20 { 3.0 } else { 6.0 }
                                                                });
                                                        }
                                                        let pixels_per_nucleotide = bar_w / total_nucleotides;

                                                        let draw_strip = |ui: &mut egui::Ui,
                                                                              color_at: &dyn Fn(usize) -> egui::Color32| {
                                                            let (rect, _) = ui.allocate_exact_size(
                                                                egui::vec2(bar_w, bar_h),
                                                                egui::Sense::hover(),
                                                            );
                                                            let painter = ui.painter_at(rect);
                                                            painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(30, 30, 30));

                                                            let mut x_pos = rect.left();
                                                            for i in 0..body_count {
                                                                let part = &agent.body[i];
                                                                let base_type = part.base_type();
                                                                let nucleotide_count = nucleotide_spans
                                                                    .get(i)
                                                                    .copied()
                                                                    .unwrap_or_else(|| if base_type < 20 { 3.0 } else { 6.0 });
                                                                let seg_w = nucleotide_count * pixels_per_nucleotide;

                                                                let r = egui::Rect::from_min_max(
                                                                    egui::pos2(x_pos, rect.top()),
                                                                    egui::pos2((x_pos + seg_w).min(rect.right()), rect.bottom()),
                                                                );
                                                                painter.rect_filled(r, 0.0, color_at(i));
                                                                x_pos += seg_w;
                                                            }
                                                        };

                                                        // Organ + amino layout (base color)
                                                        draw_strip(ui, &|i| {
                                                            let part = &agent.body[i];
                                                            let base_type = part.base_type();
                                                            if base_type < 20 {
                                                                // Amino acids: shades of bright grey
                                                                let shade = 160 + (base_type * 4) as u8;
                                                                egui::Color32::from_rgb(shade, shade, shade)
                                                            } else {
                                                                // Organs: original colors
                                                                part_base_color32(base_type)
                                                            }
                                                        });

                                                        ui.add_space(4.0);

                                                        // Combined signal strip (debug-style):
                                                        // - Beta: + = red, - = blue
                                                        // - Alpha: + = green, - = magenta
                                                        draw_strip(ui, &|i| {
                                                            let part = &agent.body[i];
                                                            let a = part.alpha_signal.clamp(-1.0, 1.0);
                                                            let b = part.beta_signal.clamp(-1.0, 1.0);

                                                            let alpha_pos = a.max(0.0);
                                                            let alpha_neg = (-a).max(0.0);
                                                            let beta_pos = b.max(0.0);
                                                            let beta_neg = (-b).max(0.0);

                                                            let r = (beta_pos + alpha_neg).min(1.0);
                                                            let g = alpha_pos.min(1.0);
                                                            let bl = (beta_neg + alpha_neg).min(1.0);

                                                            egui::Color32::from_rgb(
                                                                (r * 255.0) as u8,
                                                                (g * 255.0) as u8,
                                                                (bl * 255.0) as u8,
                                                            )
                                                        });

                                                        ui.add_space(8.0);
                                                        ui.label("Body (sequence):");

                                                        let mut any_parts = false;
                                                        ui.horizontal_wrapped(|ui| {
                                                            ui.spacing_mut().item_spacing.x = 8.0;
                                                            for i in 0..body_count {
                                                                let base = agent.body[i].base_type();
                                                                any_parts = true;
                                                                let color = if base < 20 {
                                                                    // Amino acids: shades of bright grey
                                                                    let shade = 160 + (base * 4) as u8;
                                                                    egui::Color32::from_rgb(shade, shade, shade)
                                                                } else {
                                                                    // Organs: original colors
                                                                    part_base_color32(base)
                                                                };
                                                                let name = part_base_name(base);
                                                                ui.colored_label(color, name);
                                                            }
                                                        });
                                                        if !any_parts {
                                                            ui.label("(none)");
                                                        }
                                                    }
                                                }
                                            });

                                        }); // Close the dark grey frame
                                    } // Close if let Some(agent)

                                        // Floating Recording Bar (optional)
                                        let mut recording_bar_open = state.recording_bar_visible;
                                        if state.recording_bar_visible || state.recording {
                                            egui::Window::new("🎬 Recording")
                                                .anchor(egui::Align2::CENTER_BOTTOM, [0.0, -10.0])
                                                .open(&mut recording_bar_open)
                                                .resizable(false)
                                                .collapsible(false)
                                                .show(ctx, |ui| {
                                                ui.horizontal(|ui| {
                                                    let btn_text = if state.recording { "⏹ Stop" } else { "🔴 Record" };
                                                    if ui.button(btn_text).clicked() {
                                                        if state.recording {
                                                            save_recording_requested = true;
                                                        } else {
                                                            match state.start_recording() {
                                                                Ok(()) => {
                                                                    state.recording = true;
                                                                }
                                                                Err(e) => {
                                                                    state.recording_error = Some(e.to_string());
                                                                    state.recording = false;
                                                                }
                                                            }
                                                        }
                                                    }

                                                    ui.separator();
                                                    ui.label("Size:");
                                                    ui.horizontal(|ui| {
                                                        ui.label("W:");
                                                        ui.add_enabled(
                                                            !state.recording,
                                                            egui::DragValue::new(&mut state.recording_width)
                                                                .speed(10)
                                                                .range(360..=1920),
                                                        );
                                                        ui.separator();
                                                        ui.label("H:");
                                                        ui.add_enabled(
                                                            !state.recording,
                                                            egui::DragValue::new(&mut state.recording_height)
                                                                .speed(10)
                                                                .range(360..=1920),
                                                        );
                                                    });

                                                    ui.separator();
                                                    ui.label("FPS:");
                                                    ui.add_enabled_ui(!state.recording, |ui| {
                                                        egui::ComboBox::from_id_salt("fps_combo")
                                                            .selected_text(format!("{}", state.recording_fps))
                                                            .show_ui(ui, |ui| {
                                                                ui.selectable_value(&mut state.recording_fps, 24, "24");
                                                                ui.selectable_value(&mut state.recording_fps, 30, "30");
                                                                ui.selectable_value(&mut state.recording_fps, 60, "60");
                                                            });
                                                    });

                                                    ui.separator();
                                                    ui.label("Format:");
                                                    ui.add_enabled_ui(!state.recording, |ui| {
                                                        egui::ComboBox::from_id_salt("format_combo")
                                                            .selected_text(match state.recording_format {
                                                                RecordingFormat::MP4 => "MP4",
                                                                RecordingFormat::GIF => "GIF",
                                                            })
                                                            .show_ui(ui, |ui| {
                                                                ui.selectable_value(&mut state.recording_format, RecordingFormat::MP4, "MP4");
                                                                ui.selectable_value(&mut state.recording_format, RecordingFormat::GIF, "GIF");
                                                            });
                                                    });

                                                    ui.separator();
                                                    ui.checkbox(&mut state.recording_show_ui, "Show UI");
                                                });

                                                ui.separator();
                                                ui.label("Frame center (px):");
                                                let w = state.surface_config.width as f32;
                                                let h = state.surface_config.height as f32;
                                                let capture_width = state.recording_width.min(state.surface_config.width) as f32;
                                                let capture_height = state.recording_height.min(state.surface_config.height) as f32;
                                                let min_cx = (capture_width * 0.5).min(w * 0.5);
                                                let max_cx = (w - capture_width * 0.5).max(min_cx);
                                                let min_cy = (capture_height * 0.5).min(h * 0.5);
                                                let max_cy = (h - capture_height * 0.5).max(min_cy);

                                                let mut cx = (state.recording_center_norm[0] * w).clamp(min_cx, max_cx);
                                                let mut cy = (state.recording_center_norm[1] * h).clamp(min_cy, max_cy);
                                                ui.horizontal(|ui| {
                                                    ui.label("X:");
                                                    ui.add(egui::DragValue::new(&mut cx).speed(1.0).range(min_cx..=max_cx));
                                                    ui.separator();
                                                    ui.label("Y:");
                                                    ui.add(egui::DragValue::new(&mut cy).speed(1.0).range(min_cy..=max_cy));
                                                    ui.separator();
                                                    if ui.button("Center").clicked() {
                                                        cx = w * 0.5;
                                                        cy = h * 0.5;
                                                    }
                                                });
                                                if w > 0.0 && h > 0.0 {
                                                    state.recording_center_norm = [
                                                        (cx / w).clamp(0.0, 1.0),
                                                        (cy / h).clamp(0.0, 1.0),
                                                    ];
                                                }

                                                if let Some(path) = &state.recording_output_path {
                                                    ui.label(format!("Output: {}", path.display()));
                                                } else {
                                                    ui.label("Output: (not started)");
                                                }

                                                if let Some(err) = &state.recording_error {
                                                    ui.colored_label(egui::Color32::RED, err);
                                                }
                                            });  // Close recording window
                                        }

                                        // If the user closed the recording bar via X while recording, stop recording.
                                        if state.recording && !recording_bar_open {
                                            save_recording_requested = true;
                                        }
                                        state.recording_bar_visible = recording_bar_open;

                                        // Capture frame overlay (grey preview, red when recording).
                                        if state.recording || state.recording_bar_visible {
                                            let ppp = ctx.pixels_per_point();
                                            let w_px = state.surface_config.width as f32;
                                            let h_px = state.surface_config.height as f32;
                                            let capture_width_px = state.recording_width.min(state.surface_config.width) as f32;
                                            let capture_height_px = state.recording_height.min(state.surface_config.height) as f32;

                                            let cx_px = (state.recording_center_norm[0] * w_px).clamp(0.0, w_px);
                                            let cy_px = (state.recording_center_norm[1] * h_px).clamp(0.0, h_px);

                                            let min_x = 0.0;
                                            let max_x = (w_px - capture_width_px).max(0.0);
                                            let min_y = 0.0;
                                            let max_y = (h_px - capture_height_px).max(0.0);

                                            let origin_x_px = (cx_px - capture_width_px * 0.5).clamp(min_x, max_x);
                                            let origin_y_px = (cy_px - capture_height_px * 0.5).clamp(min_y, max_y);

                                            let rect = egui::Rect::from_min_size(
                                                egui::pos2(origin_x_px / ppp, origin_y_px / ppp),
                                                egui::vec2(capture_width_px / ppp, capture_height_px / ppp),
                                            );
                                            let painter = ctx.layer_painter(egui::LayerId::new(
                                                egui::Order::Foreground,
                                                egui::Id::new("recording_frame"),
                                            ));

                                            // Grey when not recording, red when actively recording
                                            let frame_color = if state.recording {
                                                egui::Color32::RED
                                            } else {
                                                egui::Color32::GRAY
                                            };
                                            painter.rect_stroke(rect, 0.0, egui::Stroke::new(3.0, frame_color));

                                            // Show elapsed time when recording
                                            if state.recording {
                                                if let Some(start_time) = state.recording_start_time {
                                                    let elapsed = now.duration_since(start_time);
                                                    let total_secs = elapsed.as_secs();
                                                    let hours = total_secs / 3600;
                                                    let minutes = (total_secs % 3600) / 60;
                                                    let seconds = total_secs % 60;

                                                    let time_text = if hours > 0 {
                                                        format!("🔴 {:02}:{:02}:{:02}", hours, minutes, seconds)
                                                    } else {
                                                        format!("🔴 {:02}:{:02}", minutes, seconds)
                                                    };

                                                    // Position timer above the recording frame
                                                    let timer_pos = egui::pos2(
                                                        (origin_x_px + capture_width_px * 0.5) / ppp,
                                                        (origin_y_px - 30.0) / ppp,
                                                    );

                                                    painter.text(
                                                        timer_pos,
                                                        egui::Align2::CENTER_CENTER,
                                                        time_text,
                                                        egui::FontId::monospace(20.0),
                                                        egui::Color32::RED,
                                                    );
                                                }
                                            }
                                        }

                                        }); // Close egui .run() call

                                        // Handle platform output
                                        self.egui_state.handle_platform_output(&window, full_output.platform_output);
                                        state.persist_settings_if_changed();

                                        // Tessellate and cache render data
                                        state.cached_egui_primitives = self.egui_state
                                            .egui_ctx()
                                            .tessellate(full_output.shapes, full_output.pixels_per_point);
                                        state.last_egui_update_time = now;

                                        // Render with egui (fresh)
                                        state.last_present_time = now;
                                        let cached = std::mem::take(&mut state.cached_egui_primitives);
                                        let render_result = state.render(
                                            &cached,
                                            full_output.textures_delta,
                                            screen_descriptor,
                                        );
                                        state.cached_egui_primitives = cached;
                                        match render_result {
                                            Ok(_) => {}
                                            Err(wgpu::SurfaceError::Lost) => state.resize(window.inner_size()),
                                            Err(wgpu::SurfaceError::Outdated) => {}
                                            Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
                                            Err(e) => eprintln!("{:?}", e),
                                        }
                                        } else {
                                            // Render with cached egui (no rebuild / no texture updates)
                                            state.last_present_time = now;
                                            let cached = std::mem::take(&mut state.cached_egui_primitives);
                                            let render_result = state.render(
                                                &cached,
                                                egui::TexturesDelta::default(),
                                                screen_descriptor,
                                            );
                                            state.cached_egui_primitives = cached;
                                            match render_result {
                                                Ok(_) => {}
                                                Err(wgpu::SurfaceError::Lost) => state.resize(window.inner_size()),
                                                Err(wgpu::SurfaceError::Outdated) => {}
                                                Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
                                                Err(e) => eprintln!("{:?}", e),
                                            }
                                        }
                                    }
                                }
                            }

                            if reset_requested {
                                if self.reset_in_progress {
                                    // Prevent re-entrant resets (can cause stack overflows on some event backends).
                                    return;
                                }
                                self.reset_in_progress = true;

                                // Check if there's a pending resolution change
                                let requested = if let Some(triple) = self.pending_resolution_change_override.take() {
                                    Some(triple)
                                } else if let Some(gpu_state) = &self.state {
                                    gpu_state
                                        .pending_resolution_change
                                        .map(|env| (env, env / 4, env / 4))
                                } else {
                                    None
                                };

                                if let Some((env_res, fluid_res, spatial_res)) = requested {
                                    // Resolution change requested - update settings first
                                    println!(
                                        "🔄 Resetting simulation with new resolution: {}×{} (fluid={}, spatial={})",
                                        env_res,
                                        env_res,
                                        fluid_res,
                                        spatial_res
                                    );

                                    // Update simulation_settings.json
                                    let settings_path = std::path::Path::new(SETTINGS_FILE_NAME);
                                    if let Ok(mut settings) = SimulationSettings::load_from_disk(settings_path) {
                                        settings.env_grid_resolution = env_res;
                                        settings.fluid_grid_resolution = fluid_res;
                                        settings.spatial_grid_resolution = spatial_res;

                                        if let Ok(json) = serde_json::to_string_pretty(&settings) {
                                            if let Err(e) = fs::write(settings_path, json) {
                                                eprintln!("Failed to save updated settings: {:?}", e);
                                            } else {
                                                println!("✅ Settings updated: env={}, fluid={}, spatial={}",
                                                    env_res, fluid_res, spatial_res);
                                            }
                                        }
                                    }

                                    let preserve_autosave = self.preserve_autosave_on_next_reset;
                                    self.preserve_autosave_on_next_reset = false;

                                    // Delete autosave only when the user explicitly changes resolution.
                                    // If we're resetting to load a snapshot, keep autosave intact.
                                    if !preserve_autosave {
                                        let autosave_path = std::path::Path::new(AUTO_SNAPSHOT_FILE_NAME);
                                        if autosave_path.exists() {
                                            if let Err(e) = fs::remove_file(autosave_path) {
                                                eprintln!("⚠️  Failed to delete old autosave: {:?}", e);
                                            } else {
                                                println!("🗑️  Deleted old autosave (different resolution)");
                                            }
                                        }

                                        // Skip auto-load as additional safety (though file should be deleted)
                                        self.skip_auto_load = true;
                                    }

                                    // Drop old state to force full recreation
                                    self.state = None;

                                    // Recreate egui_winit state to clear all internal texture tracking
                                    let egui_ctx = egui::Context::default();
                                    self.egui_state = egui_winit::State::new(
                                        egui_ctx,
                                        egui::ViewportId::ROOT,
                                        window,
                                        None,
                                        None,
                                        None,
                                    );

                                    // Kick off GPU recreation in background, but create the Surface on the main thread.
                                    // On Windows, creating a Surface from a Window handle can fail on background threads.
                                    let env_res = env_res;
                                    let fluid_res = fluid_res;
                                    let spatial_res = spatial_res;

                                    let (tx, rx) = std::sync::mpsc::channel();
                                    self.rx = rx;
                                    self.loading_start = std::time::Instant::now();
                                    self.last_message_update = std::time::Instant::now();
                                    self.current_message_index = 0;

                                    let window_clone = window.clone();

                                    // Create instance + surface on main thread.
                                    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                                        backends: if cfg!(target_os = "windows") {
                                            wgpu::Backends::VULKAN
                                        } else {
                                            wgpu::Backends::PRIMARY
                                        },
                                        ..Default::default()
                                    });
                                    let surface = instance.create_surface(window_clone.clone()).unwrap();

                                    std::thread::spawn(move || {
                                        let adapter = pollster::block_on(instance.request_adapter(
                                            &wgpu::RequestAdapterOptions {
                                                power_preference: wgpu::PowerPreference::HighPerformance,
                                                compatible_surface: Some(&surface),
                                                force_fallback_adapter: false,
                                            },
                                        ))
                                        .unwrap();

                                        let required_features = select_required_features(adapter.features());
                                        let (device, queue) = pollster::block_on(adapter.request_device(
                                            &wgpu::DeviceDescriptor {
                                                label: Some("GPU Device"),
                                                required_features,
                                                required_limits: wgpu::Limits {
                                                    max_storage_buffers_per_shader_stage: 16,
                                                    ..wgpu::Limits::default()
                                                },
                                                memory_hints: Default::default(),
                                            },
                                            None,
                                        ))
                                        .unwrap();

                                        let state = pollster::block_on(GpuState::new_with_resources(
                                            window_clone,
                                            instance,
                                            surface,
                                            adapter,
                                            device,
                                            queue,
                                            env_res,
                                            fluid_res,
                                            spatial_res,
                                        ));
                                        let _ = tx.send(state);
                                    });
                                }

                                // Standard reset path (fast reset) when not recreating GPU
                                if self.state.is_some() {
                                    reset_simulation_state(&mut self.state, &window, &mut self.egui_state);
                                }
                                self.reset_in_progress = false;
                            }

                            let mut screenshot_requested = false;
                            let mut screenshot_4k_requested = false;

                            if let Some(gpu_state) = &mut self.state {
                                if gpu_state.screenshot_requested {
                                    gpu_state.screenshot_requested = false;
                                    screenshot_requested = true;
                                }
                                if gpu_state.screenshot_4k_requested {
                                    gpu_state.screenshot_4k_requested = false;
                                    screenshot_4k_requested = true;
                                }

                                // Auto-snapshot every AUTO_SNAPSHOT_INTERVAL epochs.
                                // NOTE: this runs after egui has updated state from sliders, so the
                                // autosave snapshot captures the latest control-panel values.
                                if !gpu_state.is_paused
                                    && gpu_state.alive_count > 0
                                    && gpu_state.epoch > 0
                                    && gpu_state.epoch % AUTO_SNAPSHOT_INTERVAL == 0
                                    && gpu_state.epoch != gpu_state.last_autosave_epoch
                                {
                                    gpu_state.last_autosave_epoch = gpu_state.epoch;
                                    let autosave_path = std::path::Path::new(AUTO_SNAPSHOT_FILE_NAME);
                                    if let Err(e) = gpu_state.save_snapshot_to_file(autosave_path) {
                                        eprintln!("G�� Auto-snapshot failed at epoch {}: {:?}", gpu_state.epoch, e);
                                    } else {
                                        println!("G�� Auto-snapshot saved at epoch {}", gpu_state.epoch);
                                    }
                                }

                                if gpu_state.snapshot_save_requested {
                                    gpu_state.snapshot_save_requested = false;

                                    // Open file dialog
                                    if let Some(path) = rfd::FileDialog::new()
                                        .add_filter("PNG Image", &["png"])
                                        .set_file_name(&format!("{}_epoch_{}.png", gpu_state.run_name, gpu_state.epoch))
                                        .save_file()
                                    {
                                        match gpu_state.save_snapshot_to_file(&path) {
                                            Ok(_) => println!("? Snapshot saved to: {}", path.display()),
                                            Err(e) => eprintln!("? Failed to save snapshot: {}", e),
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
                                        let current_env = gpu_state.env_grid_resolution;
                                        let current_fluid = gpu_state.fluid_grid_resolution;
                                        let current_spatial = gpu_state.spatial_grid_resolution;

                                        match snapshot_target_resolutions(&path) {
                                            Ok((env_res, fluid_res, spatial_res)) => {
                                                if env_res != current_env
                                                    || fluid_res != current_fluid
                                                    || spatial_res != current_spatial
                                                {
                                                    println!(
                                                        "🔄 Snapshot resolution {}x{} differs from current {}x{}; resetting to match.",
                                                        env_res,
                                                        env_res,
                                                        current_env,
                                                        current_env
                                                    );
                                                    self.pending_snapshot_load_path = Some(path);
                                                    self.preserve_autosave_on_next_reset = true;
                                                    self.pending_resolution_change_override =
                                                        Some((env_res, fluid_res, spatial_res));
                                                    self.force_reset_requested = true;
                                                    window.request_redraw();
                                                } else {
                                                    // Resolution matches; load directly.
                                                    reset_simulation_state(
                                                        &mut self.state,
                                                        &window,
                                                        &mut self.egui_state,
                                                    );
                                                    if let Some(gpu_state) = self.state.as_mut() {
                                                        match gpu_state.load_snapshot_from_file(&path) {
                                                            Ok(_) => println!(
                                                                "? Snapshot loaded from: {}",
                                                                path.display()
                                                            ),
                                                            Err(e) => eprintln!(
                                                                "? Failed to load snapshot: {}",
                                                                e
                                                            ),
                                                        }
                                                    }
                                                }
                                            }
                                            Err(e) => eprintln!("? Failed to inspect snapshot: {e:?}"),
                                        }
                                    }
                                }
                            }

                            if screenshot_requested {
                                if let Some(gpu_state) = self.state.as_mut() {
                                    println!("Capturing screenshot...");
                                    if let Err(e) = gpu_state.capture_screenshot() {
                                        eprintln!("Screenshot failed: {e:?}");
                                    }
                                }
                            }

                            if screenshot_4k_requested {
                                if let Some(gpu_state) = self.state.as_mut() {
                                    println!("Capturing 4K screenshot...");
                                    if let Err(e) = gpu_state.capture_4k_screenshot() {
                                        eprintln!("4K screenshot failed: {e:?}");
                                    }
                                }
                            }

                            if save_recording_requested {
                                if let Some(gpu_state) = self.state.as_mut() {
                                    if let Err(e) = gpu_state.save_recording() {
                                        eprintln!("Recording save failed: {e:?}");
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
                            if let Some(mut existing) = self.state.take() {
                                existing.destroy_resources();
                            }
                            target.exit();
                        }
                        WindowEvent::Resized(physical_size) => {
                            if let Some(state) = self.state.as_mut() {
                                state.resize(physical_size);
                            }
                        }
                        WindowEvent::DroppedFile(path) => {
                            // Handle drag-and-drop file loading
                            if let Some(ext) = path.extension() {
                                if ext == "png" {
                                    if let Some(gpu_state) = self.state.as_ref() {
                                        match snapshot_target_resolutions(&path) {
                                            Ok((env_res, fluid_res, spatial_res)) => {
                                                if env_res != gpu_state.env_grid_resolution
                                                    || fluid_res != gpu_state.fluid_grid_resolution
                                                    || spatial_res != gpu_state.spatial_grid_resolution
                                                {
                                                    println!(
                                                        "🔄 Dropped snapshot resolution {}x{} differs from current {}x{}; resetting to match.",
                                                        env_res,
                                                        env_res,
                                                        gpu_state.env_grid_resolution,
                                                        gpu_state.env_grid_resolution
                                                    );
                                                    self.pending_snapshot_load_path = Some(path.clone());
                                                    self.pending_resolution_change_override =
                                                        Some((env_res, fluid_res, spatial_res));
                                                    self.force_reset_requested = true;
                                                    window.request_redraw();
                                                } else {
                                                    reset_simulation_state(
                                                        &mut self.state,
                                                        &window,
                                                        &mut self.egui_state,
                                                    );
                                                    if let Some(gpu_state) = self.state.as_mut() {
                                                        match gpu_state.load_snapshot_from_file(&path) {
                                                            Ok(_) => println!(
                                                                "G�� Snapshot loaded from dropped file: {}",
                                                                path.display()
                                                            ),
                                                            Err(e) => eprintln!(
                                                                "G�� Failed to load dropped snapshot: {}",
                                                                e
                                                            ),
                                                        }
                                                    }
                                                }
                                            }
                                            Err(e) => eprintln!("? Failed to inspect dropped snapshot: {e:?}"),
                                        }
                                    }
                                }
                            }
                        }
                        WindowEvent::RedrawRequested => {
                            let mut reset_requested = self.force_reset_requested;
                            self.force_reset_requested = false;

                            if let Some(state) = self.state.as_mut() {
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

                                let fast_draw_tick = state.render_interval != 0
                                    && (state.epoch % state.render_interval as u64 == 0);

                                // In Full Speed mode, avoid presenting hundreds of frames/sec.
                                // In Fast Draw mode, avoid presenting every sim step.
                                let do_present = if state.is_paused {
                                    true
                                } else if state.current_mode == 1 {
                                    state.last_present_time.elapsed()
                                        >= std::time::Duration::from_micros(FULL_SPEED_PRESENT_INTERVAL_MICROS)
                                } else if state.current_mode == 2 {
                                    if state.ui_visible {
                                        state.last_present_time.elapsed()
                                            >= std::time::Duration::from_micros(FULL_SPEED_PRESENT_INTERVAL_MICROS)
                                    } else {
                                        fast_draw_tick
                                    }
                                } else {
                                    true
                                };

                                state.update(do_present, frame_dt);

                                if !do_present {
                                    return;
                                }

                                let screen_descriptor = ScreenDescriptor {
                                    size_in_pixels: [
                                        state.surface_config.width,
                                        state.surface_config.height,
                                    ],
                                    pixels_per_point: window.scale_factor() as f32,
                                };

                                if !state.ui_visible {
                                    state.last_present_time = now;
                                    match state.render(&[], egui::TexturesDelta::default(), screen_descriptor) {
                                        Ok(_) => {}
                                        Err(wgpu::SurfaceError::Lost) => state.resize(window.inner_size()),
                                        Err(wgpu::SurfaceError::Outdated) => {}
                                        Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
                                        Err(e) => eprintln!("{:?}", e),
                                    }
                                    return;
                                }

                                // Drain input every present, but only rebuild egui at ~30Hz (or immediately on interaction).
                                let raw_input = self.egui_state.take_egui_input(&window);
                                let input_active = !raw_input.events.is_empty();
                                let do_egui_update = input_active
                                    || state.last_egui_update_time.elapsed()
                                        >= std::time::Duration::from_micros(EGUI_UPDATE_INTERVAL_MICROS);

                                if do_egui_update {
                                let full_output = self.egui_state.egui_ctx().run(raw_input, |ctx| {
                                    egui::Window::new("Simulation Controls")
                                        .default_pos([10.0, 10.0])
                                        .default_size([300.0, 400.0])
                                        .show(ctx, |ui| {
                                            ui.heading("Info");
                                            ui.label(format!("Agents: {}", state.agent_count));
                                            ui.label(format!("Living Agents: {}", state.alive_count));
                                            let reset_button = egui::Button::new("Reset Simulation")
                                                .fill(egui::Color32::from_rgb(90, 25, 25))
                                                .stroke(egui::Stroke::new(
                                                    1.0,
                                                    egui::Color32::from_rgb(180, 80, 80),
                                                ));
                                            if ui.add(reset_button).clicked() {
                                                reset_requested = true;
                                            }
                                            if ui.button("Spawn 5000 Random Agents").clicked() && !state.is_paused {
                                                state.queue_random_spawns(5000);
                                            }

                                            ui.separator();
                                            ui.heading("Snapshot");
                                            if ui.button("?? Save Snapshot (PNG)").clicked() {
                                                state.snapshot_save_requested = true;
                                            }
                                            ui.label("Saves environment + up to 5000 agents (random sample)");

                                            ui.label(format!("World: {}x{}", state.sim_size as u32, state.sim_size as u32));
                                            ui.label(format!("Grid: {}x{}", state.env_grid_resolution, state.env_grid_resolution));
                                            ui.label(
                                                "Morphology responds to\nalpha field in environment",
                                            );

                                            ui.separator();
                                            ui.heading("Visualization");
                                                    let min_changed = ui
                                                        .add(
                                                            egui::Slider::new(
                                                                &mut state.gamma_vis_min,
                                                                0.0..=100_000.0,
                                                            )
                                                            .text("Gamma Min"),
                                                        )
                                                        .changed();
                                                    let max_changed = ui
                                                        .add(
                                                            egui::Slider::new(
                                                                &mut state.gamma_vis_max,
                                                                0.0..=100_000.0,
                                                            )
                                                            .text("Gamma Max"),
                                                        )
                                                        .changed();
                                                    if state.gamma_vis_min >= state.gamma_vis_max {
                                                        state.gamma_vis_max =
                                                            (state.gamma_vis_min + 0.001).min(100_000.0);
                                                        state.gamma_vis_min =
                                                            (state.gamma_vis_max - 0.001).max(0.0);
                                                    } else if min_changed || max_changed {
                                                        state.gamma_vis_min =
                                                            state.gamma_vis_min.clamp(0.0, 100_000.0);
                                                        state.gamma_vis_max =
                                                            state.gamma_vis_max.clamp(0.0, 100_000.0);
                                                    }

                                            ui.collapsing("Compositing", |ui| {
                                                ui.checkbox(&mut state.fluid_show, "Show Fluid Overlay")
                                                    .on_hover_text("Fluid dye overlay compositing uses Alpha/Beta colors + blend modes");

                                                ui.separator();
                                                ui.label("Alpha layer");
                                                ui.checkbox(&mut state.alpha_show, "Show Alpha");
                                                ui.horizontal(|ui| {
                                                    ui.radio_value(&mut state.alpha_blend_mode, 0, "Add");
                                                    ui.radio_value(&mut state.alpha_blend_mode, 1, "Multiply");
                                                });
                                                ui.horizontal(|ui| {
                                                    ui.label("Color");
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
                                                        .text("Gamma")
                                                        .logarithmic(true),
                                                );

                                                ui.separator();
                                                ui.label("Beta layer");
                                                ui.checkbox(&mut state.beta_show, "Show Beta");
                                                ui.horizontal(|ui| {
                                                    ui.radio_value(&mut state.beta_blend_mode, 0, "Add");
                                                    ui.radio_value(&mut state.beta_blend_mode, 1, "Multiply");
                                                });
                                                ui.horizontal(|ui| {
                                                    ui.label("Color");
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
                                                        .text("Gamma")
                                                        .logarithmic(true),
                                                );

                                                ui.separator();
                                                ui.label("Gamma layer");
                                                ui.checkbox(&mut state.gamma_show, "Show Gamma");
                                                ui.horizontal(|ui| {
                                                    ui.radio_value(&mut state.gamma_blend_mode, 0, "Add");
                                                    ui.radio_value(&mut state.gamma_blend_mode, 1, "Multiply");
                                                });
                                                ui.horizontal(|ui| {
                                                    ui.label("Color");
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
                                                        .text("Gamma")
                                                        .logarithmic(true),
                                                );
                                            });

                                            ui.separator();
                                            ui.collapsing("EnvEvolution", |ui| {
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
                                                    .allow_drag(false)
                                                    .allow_zoom([false, false])
                                                    .allow_scroll(false)
                                                    .show(ui, |plot_ui| {
                                                        plot_ui.line(Line::new(alpha_points).name("Alpha").color(Color32::GREEN));
                                                        plot_ui.line(Line::new(beta_points).name("Beta").color(Color32::RED));
                                                    });
                                            });

                                            ui.separator();
                                            ui.collapsing("Part Base-Angle Overrides", |ui| {
                                                ui_part_base_angle_overrides(ui, state);
                                            });

                                            ui.separator();
                                            ui.label("Controls:");
                                            ui.label("* Mouse: Pan camera (right drag)");
                                            ui.label("* Wheel: Zoom");
                                            ui.label("* WASD: Pan camera");
                                            ui.label("* R: Reset camera");
                                        });
                                });

                                // Handle platform output
                                self.egui_state.handle_platform_output(&window, full_output.platform_output);
                                state.persist_settings_if_changed();

                                // Tessellate and cache render data
                                state.cached_egui_primitives = self.egui_state
                                    .egui_ctx()
                                    .tessellate(full_output.shapes, full_output.pixels_per_point);
                                state.last_egui_update_time = now;

                                // Render with egui (fresh)
                                state.last_present_time = now;
                                let cached = std::mem::take(&mut state.cached_egui_primitives);
                                let render_result = state.render(
                                    &cached,
                                    full_output.textures_delta,
                                    screen_descriptor,
                                );
                                state.cached_egui_primitives = cached;
                                match render_result {
                                    Ok(_) => {}
                                    Err(wgpu::SurfaceError::Lost) => state.resize(window.inner_size()),
                                    Err(wgpu::SurfaceError::Outdated) => {}
                                    Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
                                    Err(e) => eprintln!("{:?}", e),
                                }
                                } else {
                                    // Render with cached egui (no rebuild / no texture updates)
                                    state.last_present_time = now;
                                    let cached = std::mem::take(&mut state.cached_egui_primitives);
                                    let render_result = state.render(
                                        &cached,
                                        egui::TexturesDelta::default(),
                                        screen_descriptor,
                                    );
                                    state.cached_egui_primitives = cached;
                                    match render_result {
                                        Ok(_) => {}
                                        Err(wgpu::SurfaceError::Lost) => state.resize(window.inner_size()),
                                        Err(wgpu::SurfaceError::Outdated) => {}
                                        Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
                                        Err(e) => eprintln!("{:?}", e),
                                    }
                                }
                            }

                            if reset_requested {
                                if self.reset_in_progress {
                                    return;
                                }
                                self.reset_in_progress = true;
                                reset_simulation_state(&mut self.state, &window, &mut self.egui_state);
                                self.reset_in_progress = false;
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
    }
    }  // Close impl App

    let mut app = App {
        state: None,
        egui_state: egui_winit::State::new(
            egui::Context::default(),
            egui::ViewportId::ROOT,
            &window,
            None,
            None,
            None,
        ),
        window: window.clone(),
        rx,
        loading_start,
        last_message_update,
        current_message_index: 0,
        loading_messages: LOADING_MESSAGES,
        skip_auto_load: false,
        preserve_autosave_on_next_reset: false,
        reset_in_progress: false,
        pending_snapshot_load_path: None,
        pending_resolution_change_override: None,
        force_reset_requested: false,
    };

    let _ = event_loop.run_app(&mut app);
}
