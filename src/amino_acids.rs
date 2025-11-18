// Ribossome - GPU-Accelerated Artificial Life Simulator
// Copyright (c) 2025 Filipe da Veiga Ventura Alves
// Licensed under MIT License

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[repr(C, align(4))]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AminoAcidProperties {
    pub segment_length: f32,        // offset 0
    pub thickness: f32,              // offset 4
    pub base_angle: f32,             // offset 8
    pub alpha_sensitivity: f32,      // offset 12
    pub beta_sensitivity: f32,       // offset 16
    pub is_propeller: u32,           // offset 20
    pub thrust_force: f32,           // offset 24
    pub _pad0: f32,                  // offset 28
    pub color: [f32; 3],             // offset 32-43 (WGSL aligns vec3 to 16 bytes!)
    pub _pad_color: f32,             // offset 44 (explicit padding to align to next field)
    pub is_mouth: u32,               // offset 48
    pub energy_absorption_rate: f32,
    pub beta_absorption_rate: f32,
    pub beta_damage: f32,
    pub energy_storage: f32,
    pub energy_consumption: f32,
    pub is_alpha_sensor: u32,
    pub is_beta_sensor: u32,
    pub signal_decay: f32,
    pub alpha_left_mult: f32,
    pub alpha_right_mult: f32,
    pub beta_left_mult: f32,
    pub beta_right_mult: f32,
    pub _padding: [f32; 2],  // Padding
    pub _pad_end: f32,       // Final pad to bring struct size to 112 bytes (uniform array stride)
}

#[derive(Debug, Deserialize, Serialize)]
struct AminoAcidJson {
    code: String,
    name: String,
    segment_length: f32,
    thickness: f32,
    base_angle: f32,
    alpha_sensitivity: f32,
    beta_sensitivity: f32,
    is_propeller: bool,
    thrust_force: f32,
    color: [f32; 3],
    is_mouth: bool,
    energy_absorption_rate: f32,
    beta_absorption_rate: f32,
    beta_damage: f32,
    energy_storage: f32,
    energy_consumption: f32,
    is_alpha_sensor: bool,
    is_beta_sensor: bool,
    signal_decay: f32,
    alpha_left_mult: f32,
    alpha_right_mult: f32,
    beta_left_mult: f32,
    beta_right_mult: f32,
}

#[derive(Debug, Deserialize, Serialize)]
struct AminoAcidsConfig {
    amino_acids: Vec<AminoAcidJson>,
}

impl From<&AminoAcidJson> for AminoAcidProperties {
    fn from(json: &AminoAcidJson) -> Self {
        Self {
            segment_length: json.segment_length,
            thickness: json.thickness,
            base_angle: json.base_angle,
            alpha_sensitivity: json.alpha_sensitivity,
            beta_sensitivity: json.beta_sensitivity,
            is_propeller: if json.is_propeller { 1 } else { 0 },
            thrust_force: json.thrust_force,
            _pad0: 0.0,
            color: json.color,
            _pad_color: 0.0,
            is_mouth: if json.is_mouth { 1 } else { 0 },
            energy_absorption_rate: json.energy_absorption_rate,
            beta_absorption_rate: json.beta_absorption_rate,
            beta_damage: json.beta_damage,
            energy_storage: json.energy_storage,
            energy_consumption: json.energy_consumption,
            is_alpha_sensor: if json.is_alpha_sensor { 1 } else { 0 },
            is_beta_sensor: if json.is_beta_sensor { 1 } else { 0 },
            signal_decay: json.signal_decay,
            alpha_left_mult: json.alpha_left_mult,
            alpha_right_mult: json.alpha_right_mult,
            beta_left_mult: json.beta_left_mult,
            beta_right_mult: json.beta_right_mult,
            _padding: [0.0, 0.0],
            _pad_end: 0.0,
        }
    }
}

pub fn load_amino_acids<P: AsRef<Path>>(path: P) -> Result<[AminoAcidProperties; 20], Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let config: AminoAcidsConfig = serde_json::from_str(&content)?;
    
    if config.amino_acids.len() != 20 {
        return Err(format!("Expected 20 amino acids, found {}", config.amino_acids.len()).into());
    }
    
    let mut result = [AminoAcidProperties {
        segment_length: 0.0,
        thickness: 0.0,
        base_angle: 0.0,
        alpha_sensitivity: 0.0,
        beta_sensitivity: 0.0,
        is_propeller: 0,
        thrust_force: 0.0,
        _pad0: 0.0,
        color: [0.0; 3],
        _pad_color: 0.0,
        is_mouth: 0,
        energy_absorption_rate: 0.0,
        beta_absorption_rate: 0.0,
        beta_damage: 0.0,
        energy_storage: 0.0,
        energy_consumption: 0.0,
        is_alpha_sensor: 0,
        is_beta_sensor: 0,
        signal_decay: 0.0,
        alpha_left_mult: 0.0,
        alpha_right_mult: 0.0,
        beta_left_mult: 0.0,
        beta_right_mult: 0.0,
        _padding: [0.0, 0.0],
        _pad_end: 0.0,
    }; 20];
    
    for (i, amino) in config.amino_acids.iter().enumerate() {
        result[i] = amino.into();
    }
    
    Ok(result)
}

pub fn get_default_amino_acids() -> [AminoAcidProperties; 20] {
    // Current hardcoded values as fallback
    [
        // 0: A - Alanine
        AminoAcidProperties {
            segment_length: 6.5, thickness: 2.5, base_angle: 0.0,
            alpha_sensitivity: -0.5, beta_sensitivity: 0.5,
            is_propeller: 0, thrust_force: 0.0, _pad0: 0.0,
            color: [0.3, 0.3, 0.3], _pad_color: 0.0, is_mouth: 0,
            energy_absorption_rate: 0.0, beta_absorption_rate: 0.3, beta_damage: 0.5,
            energy_storage: 0.0, energy_consumption: 0.001,
            is_alpha_sensor: 0, is_beta_sensor: 0, signal_decay: 0.2,
            alpha_left_mult: 0.8, alpha_right_mult: 0.2,
            beta_left_mult: 0.7, beta_right_mult: 0.3, _padding: [0.0, 0.0], _pad_end: 0.0,
        },
        // 1: C - Cysteine (Beta sensor)
        AminoAcidProperties {
            segment_length: 6.0, thickness: 2.5, base_angle: 0.0,
            alpha_sensitivity: 0.9, beta_sensitivity: -0.7,
            is_propeller: 0, thrust_force: 0.0, _pad0: 0.0,
            color: [1.0, 0.0, 0.0], _pad_color: 0.0, is_mouth: 0,
            energy_absorption_rate: 0.0, beta_absorption_rate: 0.3, beta_damage: 0.5,
            energy_storage: 0.0, energy_consumption: 0.001,
            is_alpha_sensor: 0, is_beta_sensor: 1, signal_decay: 0.1,
            alpha_left_mult: 0.5, alpha_right_mult: 0.5,
            beta_left_mult: 0.5, beta_right_mult: 0.5, _padding: [0.0, 0.0], _pad_end: 0.0,
        },
        // 2: D - Aspartic acid
        AminoAcidProperties {
            segment_length: 7.0, thickness: 3.0, base_angle: 0.0,
            alpha_sensitivity: -0.4, beta_sensitivity: 0.4,
            is_propeller: 0, thrust_force: 0.0, _pad0: 0.0,
            color: [0.35, 0.35, 0.35], _pad_color: 0.0, is_mouth: 0,
            energy_absorption_rate: 0.0, beta_absorption_rate: 0.3, beta_damage: 0.5,
            energy_storage: 0.0, energy_consumption: 0.001,
            is_alpha_sensor: 0, is_beta_sensor: 0, signal_decay: 0.2,
            alpha_left_mult: -0.2, alpha_right_mult: 1.2,
            beta_left_mult: -0.3, beta_right_mult: 1.3, _padding: [0.0, 0.0], _pad_end: 0.0,
        },
        // 3: E - Glutamic acid
        AminoAcidProperties {
            segment_length: 8.5, thickness: 3.0, base_angle: 0.0,
            alpha_sensitivity: 0.5, beta_sensitivity: -0.5,
            is_propeller: 0, thrust_force: 0.0, _pad0: 0.0,
            color: [0.4, 0.4, 0.4], _pad_color: 0.0, is_mouth: 0,
            energy_absorption_rate: 0.0, beta_absorption_rate: 0.3, beta_damage: 0.5,
            energy_storage: 0.0, energy_consumption: 0.001,
            is_alpha_sensor: 0, is_beta_sensor: 0, signal_decay: 0.2,
            alpha_left_mult: 1.4, alpha_right_mult: -0.4,
            beta_left_mult: 1.3, beta_right_mult: -0.3, _padding: [0.0, 0.0], _pad_end: 0.0,
        },
        // 4: F - Phenylalanine
        AminoAcidProperties {
            segment_length: 11.0, thickness: 8.0, base_angle: 0.0,
            alpha_sensitivity: -0.2, beta_sensitivity: 0.2,
            is_propeller: 0, thrust_force: 0.0, _pad0: 0.0,
            color: [0.25, 0.25, 0.25], _pad_color: 0.0, is_mouth: 0,
            energy_absorption_rate: 0.0, beta_absorption_rate: 0.3, beta_damage: 0.5,
            energy_storage: 0.0, energy_consumption: 0.001,
            is_alpha_sensor: 0, is_beta_sensor: 0, signal_decay: 0.2,
            alpha_left_mult: 0.6, alpha_right_mult: 0.4,
            beta_left_mult: 0.55, beta_right_mult: 0.45, _padding: [0.0, 0.0], _pad_end: 0.0,
        },
        // 5: G - Glycine
        AminoAcidProperties {
            segment_length: 4.0, thickness: 0.75, base_angle: 0.0,
            alpha_sensitivity: 0.9, beta_sensitivity: 0.9,
            is_propeller: 0, thrust_force: 0.0, _pad0: 0.0,
            color: [0.32, 0.32, 0.32], _pad_color: 0.0, is_mouth: 0,
            energy_absorption_rate: 0.0, beta_absorption_rate: 0.2, beta_damage: 0.3,
            energy_storage: 0.0, energy_consumption: 0.001,
            is_alpha_sensor: 0, is_beta_sensor: 0, signal_decay: 0.3,
            alpha_left_mult: 0.5, alpha_right_mult: 0.5,
            beta_left_mult: 0.5, beta_right_mult: 0.5, _padding: [0.0, 0.0], _pad_end: 0.0,
        },
        // 6: H - Histidine
        AminoAcidProperties {
            segment_length: 9.0, thickness: 6.0, base_angle: -1.0,
            alpha_sensitivity: 0.3, beta_sensitivity: -0.3,
            is_propeller: 0, thrust_force: 0.0, _pad0: 0.0,
            color: [0.28, 0.28, 0.28], _pad_color: 0.0, is_mouth: 0,
            energy_absorption_rate: 0.0, beta_absorption_rate: 0.3, beta_damage: 0.5,
            energy_storage: 0.0, energy_consumption: 0.001,
            is_alpha_sensor: 0, is_beta_sensor: 0, signal_decay: 0.2,
            alpha_left_mult: 1.2, alpha_right_mult: -0.2,
            beta_left_mult: -0.3, beta_right_mult: 1.3, _padding: [0.0, 0.0], _pad_end: 0.0,
        },
        // 7: I - Isoleucine
        AminoAcidProperties {
            segment_length: 9.0, thickness: 5.5, base_angle: 0.65,
            alpha_sensitivity: -0.3, beta_sensitivity: 0.3,
            is_propeller: 0, thrust_force: 0.0, _pad0: 0.0,
            color: [0.38, 0.38, 0.38], _pad_color: 0.0, is_mouth: 0,
            energy_absorption_rate: 0.0, beta_absorption_rate: 0.3, beta_damage: 0.5,
            energy_storage: 0.0, energy_consumption: 0.001,
            is_alpha_sensor: 0, is_beta_sensor: 0, signal_decay: 0.2,
            alpha_left_mult: 0.65, alpha_right_mult: 0.35,
            beta_left_mult: 0.7, beta_right_mult: 0.3, _padding: [0.0, 0.0], _pad_end: 0.0,
        },
        // 8: K - Lysine (MOUTH)
        AminoAcidProperties {
            segment_length: 10.0, thickness: 3.5, base_angle: -0.5,
            alpha_sensitivity: 0.6, beta_sensitivity: -0.6,
            is_propeller: 0, thrust_force: 0.0, _pad0: 0.0,
            color: [1.0, 1.0, 0.0], _pad_color: 0.0, is_mouth: 1,
            energy_absorption_rate: 0.2, beta_absorption_rate: 0.2, beta_damage: 1.0,
            energy_storage: 10.0, energy_consumption: 0.002,
            is_alpha_sensor: 0, is_beta_sensor: 0, signal_decay: 0.2,
            alpha_left_mult: 1.4, alpha_right_mult: -0.4,
            beta_left_mult: 1.3, beta_right_mult: -0.3, _padding: [0.0, 0.0], _pad_end: 0.0,
        },
        // 9: L - Leucine
        AminoAcidProperties {
            segment_length: 8.5, thickness: 5.0, base_angle: 1.2,
            alpha_sensitivity: -0.3, beta_sensitivity: 0.3,
            is_propeller: 0, thrust_force: 0.0, _pad0: 0.0,
            color: [0.36, 0.36, 0.36], _pad_color: 0.0, is_mouth: 0,
            energy_absorption_rate: 0.0, beta_absorption_rate: 0.3, beta_damage: 0.5,
            energy_storage: 0.0, energy_consumption: 0.001,
            is_alpha_sensor: 0, is_beta_sensor: 0, signal_decay: 0.2,
            alpha_left_mult: -0.3, alpha_right_mult: 1.3,
            beta_left_mult: -0.2, beta_right_mult: 1.2, _padding: [0.0, 0.0], _pad_end: 0.0,
        },
        // 10: M - Methionine (START)
        AminoAcidProperties {
            segment_length: 8.5, thickness: 4.0, base_angle: -0.85,
            alpha_sensitivity: 0.4, beta_sensitivity: -0.4,
            is_propeller: 0, thrust_force: 0.0, _pad0: 0.0,
            color: [0.8, 0.8, 0.2], _pad_color: 0.0, is_mouth: 0,
            energy_absorption_rate: 0.0, beta_absorption_rate: 0.3, beta_damage: 0.5,
            energy_storage: 0.0, energy_consumption: 0.001,
            is_alpha_sensor: 0, is_beta_sensor: 0, signal_decay: 0.2,
            alpha_left_mult: 0.8, alpha_right_mult: 0.2,
            beta_left_mult: -0.1, beta_right_mult: 1.1, _padding: [0.0, 0.0], _pad_end: 0.0,
        },
        // 11: N - Asparagine
        AminoAcidProperties {
            segment_length: 7.0, thickness: 3.0, base_angle: 0.3,
            alpha_sensitivity: 0.5, beta_sensitivity: -0.5,
            is_propeller: 0, thrust_force: 0.0, _pad0: 0.0,
            color: [0.27, 0.27, 0.27], _pad_color: 0.0, is_mouth: 0,
            energy_absorption_rate: 0.0, beta_absorption_rate: 0.3, beta_damage: 0.5,
            energy_storage: 0.0, energy_consumption: 0.001,
            is_alpha_sensor: 0, is_beta_sensor: 0, signal_decay: 0.2,
            alpha_left_mult: 0.7, alpha_right_mult: 0.3,
            beta_left_mult: 0.65, beta_right_mult: 0.35, _padding: [0.0, 0.0], _pad_end: 0.0,
        },
        // 12: P - Proline (PROPELLER)
        AminoAcidProperties {
            segment_length: 6.0, thickness: 4.0, base_angle: -1.5,
            alpha_sensitivity: -1.0, beta_sensitivity: 1.0,
            is_propeller: 1, thrust_force: 5.0, _pad0: 0.0,
            color: [0.0, 0.0, 0.5], _pad_color: 0.0, is_mouth: 0,
            energy_absorption_rate: 0.0, beta_absorption_rate: 0.3, beta_damage: 0.5,
            energy_storage: 0.0, energy_consumption: 0.004,
            is_alpha_sensor: 0, is_beta_sensor: 0, signal_decay: 0.2,
            alpha_left_mult: 1.5, alpha_right_mult: -0.5,
            beta_left_mult: 1.4, beta_right_mult: -0.4, _padding: [0.0, 0.0], _pad_end: 0.0,
        },
        // 13: Q - Glutamine
        AminoAcidProperties {
            segment_length: 8.5, thickness: 3.0, base_angle: 1.0,
            alpha_sensitivity: 0.4, beta_sensitivity: -0.4,
            is_propeller: 0, thrust_force: 0.0, _pad0: 0.0,
            color: [0.34, 0.34, 0.34], _pad_color: 0.0, is_mouth: 0,
            energy_absorption_rate: 0.0, beta_absorption_rate: 0.3, beta_damage: 0.5,
            energy_storage: 0.0, energy_consumption: 0.001,
            is_alpha_sensor: 0, is_beta_sensor: 0, signal_decay: 0.2,
            alpha_left_mult: 1.0, alpha_right_mult: 0.0,
            beta_left_mult: 0.75, beta_right_mult: 0.25, _padding: [0.0, 0.0], _pad_end: 0.0,
        },
        // 14: R - Arginine
        AminoAcidProperties {
            segment_length: 11.5, thickness: 3.5, base_angle: -0.6,
            alpha_sensitivity: 0.5, beta_sensitivity: -0.5,
            is_propeller: 0, thrust_force: 0.0, _pad0: 0.0,
            color: [0.29, 0.29, 0.29], _pad_color: 0.0, is_mouth: 0,
            energy_absorption_rate: 0.0, beta_absorption_rate: 0.3, beta_damage: 0.5,
            energy_storage: 0.0, energy_consumption: 0.001,
            is_alpha_sensor: 0, is_beta_sensor: 0, signal_decay: 0.2,
            alpha_left_mult: -0.4, alpha_right_mult: 1.4,
            beta_left_mult: -0.3, beta_right_mult: 1.3, _padding: [0.0, 0.0], _pad_end: 0.0,
        },
        // 15: S - Serine (ALPHA SENSOR)
        AminoAcidProperties {
            segment_length: 5.5, thickness: 2.5, base_angle: 0.15,
            alpha_sensitivity: -0.2, beta_sensitivity: 0.3,
            is_propeller: 0, thrust_force: 0.0, _pad0: 0.0,
            color: [0.0, 1.0, 0.0], _pad_color: 0.0, is_mouth: 0,
            energy_absorption_rate: 0.0, beta_absorption_rate: 0.2, beta_damage: 0.4,
            energy_storage: 0.0, energy_consumption: 0.001,
            is_alpha_sensor: 1, is_beta_sensor: 0, signal_decay: 0.1,
            alpha_left_mult: 0.5, alpha_right_mult: 0.5,
            beta_left_mult: 0.5, beta_right_mult: 0.5, _padding: [0.0, 0.0], _pad_end: 0.0,
        },
        // 16: T - Threonine
        AminoAcidProperties {
            segment_length: 6.5, thickness: 3.5, base_angle: -0.8,
            alpha_sensitivity: 0.5, beta_sensitivity: -0.5,
            is_propeller: 0, thrust_force: 0.0, _pad0: 0.0,
            color: [0.31, 0.31, 0.31], _pad_color: 0.0, is_mouth: 0,
            energy_absorption_rate: 0.0, beta_absorption_rate: 0.2, beta_damage: 0.4,
            energy_storage: 0.0, energy_consumption: 0.001,
            is_alpha_sensor: 0, is_beta_sensor: 0, signal_decay: 0.2,
            alpha_left_mult: 0.9, alpha_right_mult: 0.1,
            beta_left_mult: 1.0, beta_right_mult: 0.0, _padding: [0.0, 0.0], _pad_end: 0.0,
        },
        // 17: V - Valine
        AminoAcidProperties {
            segment_length: 7.5, thickness: 5.0, base_angle: 1.4,
            alpha_sensitivity: -0.3, beta_sensitivity: 0.3,
            is_propeller: 0, thrust_force: 0.0, _pad0: 0.0,
            color: [0.37, 0.37, 0.37], _pad_color: 0.0, is_mouth: 0,
            energy_absorption_rate: 0.0, beta_absorption_rate: 0.3, beta_damage: 0.5,
            energy_storage: 0.0, energy_consumption: 0.001,
            is_alpha_sensor: 0, is_beta_sensor: 0, signal_decay: 0.2,
            alpha_left_mult: -0.3, alpha_right_mult: 1.3,
            beta_left_mult: 1.2, beta_right_mult: -0.2, _padding: [0.0, 0.0], _pad_end: 0.0,
        },
        // 18: W - Tryptophan (STORAGE)
        AminoAcidProperties {
            segment_length: 16.0, thickness: 12.0, base_angle: -0.35,
            alpha_sensitivity: 0.1, beta_sensitivity: -0.1,
            is_propeller: 0, thrust_force: 0.0, _pad0: 0.0,
            color: [1.0, 0.5, 0.0], _pad_color: 0.0, is_mouth: 0,
            energy_absorption_rate: 0.0, beta_absorption_rate: 0.4, beta_damage: 0.6,
            energy_storage: 10.0, energy_consumption: 0.001,
            is_alpha_sensor: 0, is_beta_sensor: 0, signal_decay: 0.15,
            alpha_left_mult: 0.55, alpha_right_mult: 0.45,
            beta_left_mult: 0.6, beta_right_mult: 0.4, _padding: [0.0, 0.0], _pad_end: 0.0,
        },
        // 19: Y - Tyrosine
        AminoAcidProperties {
            segment_length: 11.5, thickness: 4.0, base_angle: 0.9,
            alpha_sensitivity: -0.2, beta_sensitivity: 0.2,
            is_propeller: 0, thrust_force: 0.0, _pad0: 0.0,
            color: [0.26, 0.26, 0.26], _pad_color: 0.0, is_mouth: 0,
            energy_absorption_rate: 0.0, beta_absorption_rate: 0.3, beta_damage: 0.5,
            energy_storage: 0.0, energy_consumption: 0.001,
            is_alpha_sensor: 0, is_beta_sensor: 0, signal_decay: 0.2,
            alpha_left_mult: -0.5, alpha_right_mult: 1.5,
            beta_left_mult: -0.4, beta_right_mult: 1.4, _padding: [0.0, 0.0], _pad_end: 0.0,
        },
    ]
}
