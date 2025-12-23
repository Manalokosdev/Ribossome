# Project Structure (ALsimulatorv3 / Ribossome)

This document is a quick map of what each major file/folder is for, and which items are **runtime inputs**, **source code**, or **generated/debug artifacts**.

## Runtime entrypoints

- `src/main.rs`
  - The application entrypoint (UI, simulation loop, GPU setup, snapshot save/load, settings load/save).

## Core source code

- `src/`
  - Rust code for the app.
  - Notable files:
    - `src/main.rs`: main application.
    - `src/amino_acids.rs`: amino acid tables/config decoding.

- `shaders/`
  - Modular WGSL shader files.
  - Typical roles:
    - `shaders/shared.wgsl`: common structs/constants/helpers used by multiple shader stages.
    - `shaders/simulation.wgsl`: main simulation compute.
    - `shaders/fluid.wgsl`: fluid / dye advection pipelines.
    - `shaders/render.wgsl`: rendering pipeline.
    - `shaders/composite.wgsl`, `shaders/*_vis.wgsl`: visualization passes.

## Runtime configuration (edited via UI / loaded at startup)

- `simulation_settings.json`
  - Primary settings file loaded/saved by the UI.
  - Path is referenced in code as `SETTINGS_FILE_NAME` in `src/main.rs`.

- `config/part_properties.json`
  - Part property table used by the simulation (can be treated as an “overrides” source).
  - Path is referenced in code as `PART_PROPERTIES_JSON_PATH` in `src/main.rs`.

- `config/part_base_angle_overrides.csv`
  - The **Overrides tab** base-angle override file.
  - Path is referenced in code as `PART_OVERRIDES_CSV_PATH` in `src/main.rs`.
  - Semantics: values are radians; `NaN` means “use shader default”.

- `config/*.csv`, `config/*.json`
  - Static data tables used by the sim (amino acids, organ slots/table, etc.).
  - Special note: `config/ORGAN_TABLE.csv` is the single source of truth for organ mapping.

- `config/presets/`
  - Optional curated presets / test inputs (not required by runtime unless you load them manually).
  - Example: `config/presets/vampire_tester.json`

## Snapshots (stateful outputs)

- `autosave_snapshot.png`
  - Auto-saved snapshot (PNG with embedded metadata).
  - Path is referenced in code as `AUTO_SNAPSHOT_FILE_NAME` in `src/main.rs`.

- `maps/`
  - Saved images / snapshots / map assets used by the app (e.g. splash image).

## Developer notes / audits

- `docs/`
  - Main documentation (README, contributing, roadmaps).

- `ORGAN_RENDERING_AUDIT.md`, `REFACTORING_AUDIT.md`
  - Internal audit notes.

- `AGENT_FILE_FORMAT.md`
  - Notes/spec for agent/snapshot file structure.

## Generated / debug artifacts (safe to ignore in git)

- `target/`
  - Rust build output (generated).

- `archive/`
  - Scratch files, backups, experimental snapshots.
  - Recommended: keep *important* reference files; ignore ad-hoc backup folders like `archive/revert_backup_*`.

- `combined_shaders.wgsl`, `shader_source_dump.wgsl`
  - Large concatenated/dumped shader files used for comparison/debugging.
  - Not used by runtime code; treated as generated artifacts and ignored by git.

- `archive/notes/new_functions.txt`, `archive/notes/old_functions.txt`
  - Temporary notes during refactors.

## Scripts

- `scripts/`
  - Helper scripts for building/running and data generation.

## Suggested conventions

- Treat these as **tracked inputs**:
  - `config/part_base_angle_overrides.csv`
  - `config/part_properties.json`
  - `simulation_settings.json`
  - (Optionally) one canonical `autosave_snapshot.png` if you want a “default state” in-repo.

- Treat these as **untracked artifacts**:
  - `archive/revert_backup_*`
  - Anything under `target/`
  - One-off shader dumps if they’re regenerated frequently.
