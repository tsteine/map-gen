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

## Current First Step

Split the current room-part save/refill distance outcomes into directed
proximity-utility components.

The current save/refill distance outcome is a combined round-trip value:

- nearest save/refill to room part
- room part to nearest save/refill

Replace this with separate directed utility outcomes:

- proximity utility from nearest save to room part
- proximity utility from room part to nearest save
- proximity utility from nearest refill to room part
- proximity utility from room part to nearest refill

The utility for a finite directed distance `d` is:

```text
u(d) = scale / (d + scale)
```

Unreachable is treated as infinite distance, so its utility is the limiting
value:

```text
u(unreachable) = 0
```

`scale` should be a required config value. Larger values make long finite
distances retain more reward; smaller values concentrate the reward near short
distances.

The training and generation config can keep one weight for save distance and
one weight for refill distance. Each weight applies to both directions.

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

## First Implementation Sequence

1. Rename and split Rust outcome generation.
   - Replace combined save/refill room-part outcome vectors with directed
     `to` and `from` utility vectors.
   - Keep masks direction-specific.
   - Add direction-specific finalized masks and known values.
   - Add a required proximity-utility scale config value and use it to convert
     distances to target utilities.
   - Preserve strict required fields across Python/Rust bindings.

2. Split current room-part distance features.
   - Replace combined save/refill feature encodings with directed feature
     encodings.
   - Split frontier distance features into `room_part -> frontier` and
     `frontier -> room_part`.
   - Keep the existing global-feature path initially; do not introduce
     room-part nodes in this step.

3. Update Python data plumbing.
   - Update `EndOutcomes`, feature dataclasses, buffer allocation, and transfer
     code to carry directed fields.
   - Update training batch construction, generated outcome concatenation, and
     metrics.
   - Use named dataclass construction throughout.

4. Update model heads and forward override.
   - Replace each combined save/refill output head with directed heads.
   - Predict directed proximity utilities rather than raw distances.
   - In forward, substitute finalized known utilities with `torch.where`.
   - Use `0` for finalized unreachable utilities.

5. Update loss and generation reward.
   - Apply the existing save/refill weights to both directed components.
   - Change generation reward from negative distance penalty to positive
     proximity utility reward.
   - Train with MSE against directed proximity utility targets.
   - Keep logging the existing `save_distance` and `refill_distance` aggregates
     as average round-trip distances conditioned on reachability, so new runs
     remain comparable with previous experiments.
   - Add finalized-unreachable frequency and average proximity utility metrics
     so the new objective is visible.

6. Test and validate.
   - Add Rust tests for directed distances and finalized masks.
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

- Decide whether directed utility predictions should be represented as four
  separate fields or grouped named objects in Python and PyO3 result classes.
- Decide the sparse room-part row identity scheme before implementing
  room-part nodes.
- Check the cost of frontier/frontier graph-distance pair features before
  enabling dense all-pairs features; prefer sparse edges if dense pairs are too
  expensive.
