# Progress (Updated: 2025-11-11)

## Done

- Fixed spawn button functionality
- Ensured pause consistency
- Implemented consistent RNG advancement in queue_random_spawns using xorshift
- Fixed continuous spawning by clearing spawn queue after processing
- Fixed panic when draining spawn queue beyond available elements
- Changed spawn system to use GPU-based position randomization
- Fixed RNG inconsistency in replenish_population function

## Doing



## Next

- Test both manual and automatic spawns to verify unique agent generation
