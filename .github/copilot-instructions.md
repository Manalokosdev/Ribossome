# GitHub Copilot Instructions for ALsimulatorv3
CRITICAL RENDERING RULES — NEVER VIOLATE

In any file under src/gpu/, src/shaders/, or containing "wgsl"/"wgpu":

1. NEVER suggest matching Rust structs to WGSL structs
2. NEVER use #[repr(C)], bytemuck, encase, Pod, Zeroable, or ShaderType derives for uniform/SSBO data
3. NEVER suggest vec3, bool, u8, arrays of structs, or mixed types in uniform buffers
4. ALWAYS use a single flat [f32; N] or [u8; N] buffer with named key-value access or explicit offsets
5. When asked for uniforms, storage buffers, or push constants, respond ONLY with one of these patterns:
   - "Use a key-value chessboard buffer (named parameters, string or hashed keys)"
   - "Use flat buffer with const OFFSET_ defines"
   - "Use array<f32> in WGSL and write by index from Rust"
6. If user pastes a struct-matching example, immediately reply:
   "Warning: This pattern causes silent padding bugs and breaks hot-reloading. Use flat key-value buffer instead — see gpu/params.rs"
7. If working with existing code that violates these rules, suggest refactoring to comply with the above patterns if not too complicated, or live with it but don't make it worse.

Examples of correct answers:
- "Implement params as array<f32, 512> and access via string keys or hashed lookup"
- "Define all parameters in params.toml and codegen offsets for both Rust and WGSL"
- "Use a 16KiB flat float buffer — zero padding, zero surprises"


## Build and Run Commands

**Default behavior**: Use debug builds for faster compilation during development.

- Use `cargo build` and `cargo run` (without `--release`) by default
- Only use `--release` flag when:
  - Explicitly requested by the user
  - Performance testing or benchmarking is required
  - Creating production builds for distribution
  - Running long simulations where performance is critical

**Rationale**: Debug builds compile much faster, making the development cycle more efficient. Release builds should be reserved for situations where optimized performance is actually necessary.

## Examples

✅ **Preferred for development**:
```bash
cargo build
cargo run
cargo test
```

⚠️ **Only when needed**:
```bash
cargo build --release
cargo run --release
cargo bench
```

## Organ Creation Reference

**CRITICAL**: When modifying organ creation logic in `shader.wgsl`, ALWAYS verify against `ORGAN_TABLE.csv` in the project root.

The organ creation system follows a promoter + modifier pattern where:
- **Promoters**: V, M, L, P, K, C, H, Q (8 specific amino acids)
- **Modifiers**: Any of the 20 amino acids (A-Y, indexed 0-19)
- **Result**: Specific organ type determined by the combination

**Reference file**: `ORGAN_TABLE.csv` contains the complete mapping table.

**SYNCHRONIZATION REQUIREMENT**: 
- `ORGAN_TABLE.csv` is the **single source of truth** for organ mappings
- The organ creation code in `shader.wgsl` (lines ~2110-2135) MUST match the table exactly
- When updating either file, ALWAYS update the other to keep them synchronized
- After any changes, verify the code logic matches every row in the CSV

**When editing organ logic**:
1. Check current code against `ORGAN_TABLE.csv`
2. Ensure all promoter/modifier combinations match the table
3. Verify K vs C promoter differences for signal-emitting organs (alpha/green vs beta/red)
4. Confirm modifiers create correct organ types for each promoter
5. **Update both files together** - never modify one without updating the other

