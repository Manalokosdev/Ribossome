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
