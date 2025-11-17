# Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-11-09 | Adopt debug runs (cargo run) instead of release runs for this session. | User requested not to use --release; debug builds allow faster iteration and easier debugging while performance optimization is not currently the focus. |
| 2025-11-13 | Restore original triple angle contribution in morphology build (seed + alpha*gain + beta*gain) with separate alpha/beta gains. | Matches the old Python model’s per-part angle composition while retaining current smoothing/clamping; adds ANGLE_GAIN_ALPHA/BETA constants to tune alpha vs beta influence independently. Will reference the legacy Python program for future parity tweaks (chirality, per-amino multipliers). |
