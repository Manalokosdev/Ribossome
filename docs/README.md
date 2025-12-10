# Ribossome

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Ribossome** is a GPU-accelerated artificial life simulator written in Rust using WebGPU (wgpu). It evolves complex creatures with genetic genomes, morphology, sensors, propellers, mouths (including vampire predators!), and rich ecological interactions—all in a single massive compute shader.

Watch predators emerge, ecosystems balance, and bizarre life forms evolve in real-time at 1000+ FPS on modern GPUs.

## Features

- Fully GPU-driven simulation (agents, physics, environment diffusion, rendering)
- Genetic translation with codons → amino acids → body parts and organs
- Evolving predators with vampire mouths and trail-following
- Dynamic environment with food/poison rain, terrain, prop wash, and chemical slopes
- Rich visualization options (lighting, trails, interpolation modes)
- Snapshot save/load (PNG with embedded metadata)
- Auto-difficulty, rain cycling, and extensive tuning UI

## Screenshots / Video

*(Add your own screenshots or a GIF/video link here, e.g., predators evolving!)*

## Building & Running

Requirements:
- Rust toolchain (stable)
- A GPU with WebGPU support (modern NVIDIA/AMD/Intel, or Vulkan/Metal/DX12)

```bash
git clone https://github.com/Manalokosdev/Ribossome.git
cd ribossome
cargo run --release
```

The first run may take 10–60 seconds due to shader compilation (normal for large WGSL shaders). Subsequent runs are instant.

## Controls

- **WASD**: Pan camera
- **Mouse wheel**: Zoom
- **Right-drag**: Pan
- **Left-click**: Select agent for inspector
- **Space**: Toggle UI
- **F**: Follow selected agent
- **R**: Reset camera

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! See CONTRIBUTING.md for guidelines.

## Acknowledgments

Built with Rust, wgpu, egui, and a lot of passion for artificial life.

---

Copyright © 2025 Filipe da Veiga Ventura Alves
