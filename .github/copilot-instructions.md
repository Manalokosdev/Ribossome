# GitHub Copilot Instructions for ALsimulatorv3

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

