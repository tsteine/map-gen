# Local Outcome Architecture Plan

## Goal

Put local outcomes on a sounder modeling footing while keeping generation cost
predictable.

The model should use local representations for outcomes whose truth depends on
specific room parts, missing connections, and nearby frontiers. This local
information must enter the shared frontier representation used for proposal
scores, not only the final output heads, because generation fully scores only a
small number of proposed candidates.

## Guiding Design

- Improve proposal-visible local representations before hard-wiring local
  rewards into model outputs. Prior experiments with deterministic
  finalized-known save/refill overrides regressed generation quality, likely by
  making short-term reward fulfillment too attractive before the model could
  represent long-term consequences accurately.
- Add graph nodes only for unresolved placed room parts. A placed room part is
  included as a local node when at least one attached local outcome can still be
  affected by future placements.
- Keep unplaced room-part outcomes routed through the global state.
- Use bounded, fixed-width neighbor tensors for the first implementation. Rust
  may generate dynamic candidate edges, but Python should receive top-k padded
  tensors so message-passing cost remains predictable.
- Let frontier and room-part nodes exchange messages in both directions at each
  layer. The final frontier node state remains the proposal state.

## Current State: Directed Save/Refill/Frontier Features

The global room-part distance features now expose directional information
instead of round-trip/compressed distances.

Per room part, the global features include:

- distance from room part to nearest save
- distance from nearest save to room part
- distance from room part to nearest refill
- distance from nearest refill to room part
- distance from room part to nearest frontier
- distance from nearest frontier to room part
- furthest destination distance
- furthest source distance

These features use the compact distance encoding convention:

- `0` for unreachable or absent
- `distance + 1` for finite distances, saturated as needed

The model still uses the current global-feature route for these signals. No
deterministic finalized-known utilities are substituted into outputs.

## Step 1: Unresolved Room-Part Nodes

Add a second sparse node type for placed room parts whose local outcomes are not
fully determined.

A room part should get a node when any of these are true:

- at least one directed save/refill outcome for that part is not finalized
- the part is an endpoint of an unresolved missing-connect outcome
- a future frontier can still affect a local outcome attached to the part

Room-part node identity:

- Each node stores the global room-part index.
- Python receives row-to-snapshot and row-to-room-part tensors, analogous to
  frontier row metadata.
- Outputs route from room-part rows back to global room-part output indices.

Room-part node features:

- room/part identity embedding
- active/placed state
- directed distance from room part to nearest save
- directed distance from nearest save to room part
- directed distance from room part to nearest refill
- directed distance from nearest refill to room part
- directed distance from room part to nearest frontier
- directed distance from nearest frontier to room part
- directed furthest-part distances
- unresolved objective flags

## Deferred: Finalized-Known Directed Overrides

Finalized-known directed save/refill overrides remain a plausible later
optimization, but should be deferred until proposal-visible local structure is
stronger.

For room part `p`, the candidate rules are:

- `p -> nearest save` is finalized when the current finite `p -> save` distance
  is less than or equal to the current `p -> nearest frontier` distance.
- `nearest save -> p` is finalized when the current finite `save -> p` distance
  is less than or equal to the current `nearest frontier -> p` distance.
- The same rules apply for refill distances.
- Unreachable outcomes are finalized when neither the objective nor any frontier
  is reachable in the relevant direction.

If reintroduced, known values should use the same numeric scale as the target:

- finite finalized distances: `scale / (d + scale)`
- finalized unreachable distances: `0`

Before enabling these overrides for generation, validate that the proposal
representation can model long-term tradeoffs well enough that deterministic
short-term reward improvements do not dominate candidate selection.

## Step 2: Bounded Part-Frontier Message Passing

Add bidirectional sparse edges between unresolved room-part nodes and relevant
frontier nodes.

Use separate top-k bounds for each direction:

- `part <- frontier`: nearest or most relevant frontiers for each unresolved
  room part, including both graph directions where applicable.
- `frontier <- part`: most relevant unresolved room parts for each frontier,
  ranked by local objective pressure or potential improvement.

Rust should build candidate edge lists from graph-distance caches, rank them,
and pack the selected top-k edges into fixed-width tensors with `-1` padding.
Python should consume those tensors with the same gather-and-mask style as the
current frontier-neighbor message passing.

Edge features should include:

- directed graph distances for the edge
- same-component/reachability flags where useful
- local objective flags: save, refill, missing-connect endpoint
- improvement margin against the current finalized-known threshold where
  applicable

At each message-passing layer:

- frontier nodes receive frontier-neighbor messages
- frontier nodes receive room-part messages
- room-part nodes receive frontier messages
- both node types update from their current state, incoming messages, and global
  state

The final frontier node state remains `proposal_state`, so proposal scoring
automatically benefits from unresolved local-outcome information.

Track truncation diagnostics:

- unresolved room-part node count
- candidate part-frontier edge count before top-k
- fraction of part rows and frontier rows hitting each cap
- average and max selected fan-in/fan-out

If truncation is frequent and appears quality-limiting, raise caps or consider a
COO/segment-reduce representation for only the affected edge direction.

## Step 3: Local Outcome Heads

Route local outcomes through local node/query representations.

Save/refill utilities:

- For placed room parts with unresolved nodes, predict directed save/refill
  utilities from the room-part node state.
- For finalized entries, initially keep learned predictions unless a later
  experiment re-enables deterministic finalized-known overrides.
- For unplaced room parts, keep using the global pooled state.

Missing-connect outcomes:

- Treat each missing connection as a directed local query:
  `source_part -> destination_part`.
- Predict missing-connect validity and distance from endpoint states plus
  directed pair features and global state.
- If endpoints have room-part nodes, use those node states.
- If an endpoint is placed but omitted because all attached outcomes are
  finalized, use deterministic known values where available or a compact
  non-message-passed endpoint embedding.
- If the room is unplaced, route through the global state as today.

Door/frontier proposal outcomes:

- Keep proposal scoring on final frontier node states.
- Do not add a separate proposal-only integration path unless diagnostics show
  the final frontier state is not carrying local information effectively.

## Tests And Validation

Rust tests:

- unresolved room-part node selection
- part-frontier top-k edge packing and cap/truncation diagnostics
- missing-connect endpoint/query metadata

Python tests:

- dataclass and PyO3 shape compatibility
- model forward with zero room-part nodes
- model forward with room-part nodes and part-frontier edges
- output routing for placed, unplaced, finalized, and missing-connect outcomes

Validation commands:

- `cargo test`
- `conda run -n map-gen maturin develop`
- Python compile/smoke checks in the `map-gen` conda environment

## Open Design Checks

- Choose initial top-k caps for `part <- frontier` and `frontier <- part`.
- Define the exact ranking score for `frontier <- part` edges so proposal states
  receive the most useful unresolved local pressure.
- Decide whether missing-connect validity and missing-connect distance should
  share one directed query representation or use separate heads from the same
  query state.
- Revisit frontier-neighbor count after local nodes are added; extra
  frontier-frontier neighbors may become more useful late in training but should
  be evaluated separately from the room-part-node change.
- Decide when, if ever, to re-enable finalized-known deterministic overrides
  after proposal-visible local architecture is in place.
