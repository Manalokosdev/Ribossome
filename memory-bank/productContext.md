# Product Context

Describe the product.

## Overview

Provide a high-level overview of the project.

## Core Features

- Feature 1
- Feature 2

## Technical Stack

- Tech 1
- Tech 2

## Project Description

Artificial life simulation where agents evolve through genetic algorithms, with WebGPU acceleration for performance. Features include real-time evolution, spawn controls, pause functionality, and genome saving/loading.



## Architecture

WebGPU compute shaders for agent simulation with morphology building, spawn processing, and paused state handling. Rust application with wgpu backend for GPU compute, spawn request queue system with genome override capability.



## Technologies

- Rust
- WebGPU compute shaders (WGSL)
- GLSL-like shader language for GPU compute
- 64-base RNA genome representation (AUGC nucleotides)
- Real-time GPU-accelerated evolution simulation



## Libraries and Dependencies

- wgpu (WebGPU Rust bindings)
- winit (window management)
- egui (immediate mode GUI)
- serde (serialization)
- rfd (file dialogs)
- anyhow (error handling)

