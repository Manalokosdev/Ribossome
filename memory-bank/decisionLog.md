# Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-11-09 | Adopt debug runs (cargo run) instead of release runs for this session. | User requested not to use --release; debug builds allow faster iteration and easier debugging while performance optimization is not currently the focus. |
| 2025-11-13 | Restore original triple angle contribution in morphology build (seed + alpha*gain + beta*gain) with separate alpha/beta gains. | Matches the old Python model’s per-part angle composition while retaining current smoothing/clamping; adds ANGLE_GAIN_ALPHA/BETA constants to tune alpha vs beta influence independently. Will reference the legacy Python program for future parity tweaks (chirality, per-amino multipliers). |
| 2025-12-18 | Fluid obstacles no longer damp velocity via permeability; use neighbor-normal reflection (energy-preserving) plus elastic domain wall bounce. Slope coupling switched from direct slope acceleration to direction-steering based on (s_dir - v_dir) scaled by (1 - dot). | User wanted no energy loss at obstacles and to experiment with a slope–flow interaction that compares normalized flow and slope directions; reflection preserves kinetic energy better than permeability damping, and steering matches the requested approach while allowing easy strength tuning via existing scale parameter. |
