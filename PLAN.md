# Local Outcome Prediction Plan

## Goal

Move outcomes that are naturally local to rooms, doors, frontiers, or room parts
away from purely pooled/global prediction. Use the most local valid information
available at each generation state:

- Unplaced entities are predicted from the post-pooling global embedding.
- Placed entities are predicted from local node embeddings when the relevant
  local node exists.
- Outcomes that are already known are wired directly into the prediction output
  so generation reward uses the known value and training gradients are masked
  for that entry.

## Current Step

Add known-finalized overrides for directed save/refill proximity utilities.
The model already predicts four directed proximity utilities for save/refill
reachability quality; this step adds deterministic overrides for values that
are known from the current graph state.

## Known Directed Distance Outcomes

A directed save/refill distance can be finalized before episode end when future
steps cannot improve it.

For room part `p`:

- `p -> nearest save` is finalized when the current finite `p -> save` distance
  is less than or equal to the current `p -> nearest frontier` distance.
- `nearest save -> p` is finalized when the current finite `save -> p` distance
  is less than or equal to the current `nearest frontier -> p` distance.
- The same rules apply for refill distances.

Unreachable outcomes can also be finalized:

- `p -> save` is finalized as unreachable when there is no current path from
  `p` to any save and no current path from `p` to any frontier.
- `save -> p` is finalized as unreachable when there is no current path from
  any save to `p` and no current path from any frontier to `p`.
- The same rules apply for refill distances.

Known values should be substituted in model forward as proximity utilities,
using the same numeric scale as the target. Finite finalized distances use
`scale / (d + scale)`. Finalized unreachable distances use `0`. This makes
generation and training consume the same model output, while cutting gradients
through finalized entries.

The model should predict expected proximity utility, not expected distance
conditioned on reachability. This keeps unreachable states well-defined without
turning save/refill reachability into a separate validity objective.

The existing save/refill reward and loss weights continue to apply to both
directions for each category.

## Implementation Sequence

1. Add direction-specific finalized masks and known values.
   - Compute finalized-known state in Rust.
   - Use active save/refill room-part lists and frontier distance information.
   - Preserve required fields across Python/Rust bindings.

2. Split current room-part distance features.
   - Replace combined save/refill feature encodings with directed feature
     encodings.
   - Split frontier distance features into `room_part -> frontier` and
     `frontier -> room_part`.
   - Keep the existing global-feature path initially; do not introduce
     room-part nodes in this step.

3. Update Python data plumbing.
   - Extend `EndOutcomes`, feature dataclasses, buffer allocation, and transfer
     code to carry finalized-known fields.
   - Update training batch construction and generated outcome concatenation.
   - Use named dataclass construction throughout.

4. Update model forward override.
   - In forward, substitute finalized known utilities with `torch.where`.
   - Use `0` for finalized unreachable utilities.
   - Ensure substituted entries cut gradients to learned predictions.

5. Test and validate.
   - Add Rust tests for finalized masks.
   - Add Python smoke tests for shape compatibility and model forward.
   - Run `cargo test`, `maturin develop`, and Python compile/smoke checks in
     the `map-gen` conda environment.

## Later Room-Part Node Architecture

Add a second sparse node type for placed room parts.

Node types:

- Frontier nodes use the current larger frontier embedding.
- Placed room-part nodes use a smaller room-part embedding.

Message passing:

- Frontier nodes exchange messages with neighboring frontier nodes.
- Room-part nodes exchange messages with neighboring frontier nodes.
- Frontier nodes receive messages from neighboring room-part nodes.
- Room-part nodes do not exchange messages directly with other room-part nodes.

Prediction routing:

- Placed room-part outcomes are predicted from room-part node embeddings.
- Unplaced room-part outcomes are predicted from the post-pooling global
  embedding and routed to the corresponding output indices.
- Finalized known outcomes override learned predictions.

Room-part node features:

- distance from nearest save
- distance to nearest save
- distance from nearest refill
- distance to nearest refill
- distance from furthest room part
- distance to furthest room part

Room-part/frontier pair features:

- distance from room part to frontier
- distance from frontier to room part

Frontier/frontier pair features:

- graph distance from source frontier to destination frontier
- graph distance from destination frontier to source frontier

The existing combined `room_part_frontier_distance` feature should be removed
when the pair-feature route is introduced.

## Open Design Checks

- Decide the sparse room-part row identity scheme before implementing
  room-part nodes.
- Check the cost of frontier/frontier graph-distance pair features before
  enabling dense all-pairs features; prefer sparse edges if dense pairs are too
  expensive.
