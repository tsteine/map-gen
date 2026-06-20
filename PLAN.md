# Frontier Query Output Heads

## Context

Currently local outputs (such as missing-connects, save/refill distances) are
predicted based on global state. An earlier experiment attempted to introduce
room-part nodes and message passing between room-parts nodes and frontiers to
construct local predictions, but was slow and did not achieve good quality.

## Goal

In light of the failed room-part-node/local-output experiment, implement a new
approach with output heads that query the existing frontier states. The aim is
to give local outcomes access to the relevant local frontier evidence without
adding room-part graph nodes or extra cross-type message passing.

This should preserve the useful part of the previous direction:

- local predictions for outcomes that are naturally about specific room parts or
  room connections;
- better data efficiency by reusing frontier representations tied to the
  possible future actions;
- proposal relevance, by explicitly exposing query participation information to
  proposal scoring.

But it should avoid the main costs and risks of the room-part-node approach:

- no large population of extra graph nodes;
- no frontier state perturbation from room-part messages;
- no expensive bidirectional part/frontier message passing;
- no need to tune room-part embedding widths.

## Core Idea

For an output whose truth depends on future connections through frontiers,
construct one or more frontier sets and aggregate the final frontier states over
those sets.

For a missing-connect outcome `r -> s`:

- `F = {frontier | r can currently reach frontier}`
- `G = {frontier | frontier can currently reach s}`
- the connection can become valid if future placements create some path from a
  frontier in `F` to a frontier in `G`.

The query head receives aggregate embeddings for `F` and `G`, plus small
scalar/count features, and predicts a residual or replacement for the
corresponding output.

The current code already computes the key missing-connect masks as
`frontier_connection_reachability`: bit 1 means `r -> frontier`, and bit 2 means
`frontier -> s`.

## Design Preference

Use query heads as zero-initialized residuals on top of the existing global
heads, not as immediate hard replacements.

For example:

```text
connection_logit = global_connection_logit + query_connection_delta
```

with the final query projection initialized to zero. This keeps initial behavior
equivalent to the current model and lets training turn on the new path only when
it helps.

This is safer than replacing the existing head because the existing global head
already learns useful broad context, and the query head may initially be
under-calibrated.

Also expose query information to the proposal state. Query heads can improve
full-scored logits, but proposal scoring only sees the frontier proposal state.
If query information is absent from that state, proposal distillation can only
learn an incomplete approximation. The proposal-query summary should be added as
its own step after the first output-query experiment is working.

## Step 1: Missing-Connect Frontier Query Head

Start with missing-connect validity. The existing dense
`frontier_connection_reachability` feature proves the needed reachability sets
are already available, but the implementation should use sparse CSR-style query
rows so memory and fan-in are controlled.

Implementation shape:

- Keep the existing global `connection_output`.
- Add a `MissingConnectFrontierQueryHead`.
- Input:
  - final frontier states `X`: `[frontier_row, embedding_width]`;
  - `row_snapshot_idx`: `[frontier_row]`;
  - missing-connect query rows, one row per unresolved connection output;
  - bounded source frontier indices for each query row;
  - bounded target frontier indices for each query row;
  - source/target distance buckets or scalar distance features;
  - full source/target counts and cap-hit indicators;
  - global pooled state, if useful.
- For each missing-connect query row:
  - `F = {frontier | r can currently reach frontier}`;
  - `G = {frontier | frontier can currently reach s}`;
  - if an `r -> s` path already exists through placed rooms, the validity
    outcome is already determined and no validity query row should be emitted;
  - rank `F` by shortest graph distance `r -> frontier`;
  - rank `G` by shortest graph distance `frontier -> s`;
  - keep bounded nearest frontiers on each side;
  - aggregate `X` over bounded `F` into `source_pool`;
  - aggregate `X` over bounded `G` into `target_pool`;
  - include count/empty features for both sets;
  - feed `[source_pool, target_pool, count_features, optional_global_state]`
    into a small MLP;
  - scatter-add the zero-initialized output as a delta to the corresponding
    `connection_invalid` logit.

The existing dense `frontier_connection_reachability` feature can remain as a
baseline or temporary implementation aid, but the planned query representation
should be CSR-style rather than dense `[frontier, connection]`.

Initial aggregation should be cheap:

- mean pool;
- max pool;
- log/count features.

Do not introduce pairwise `F x G` computation initially.

Expected benefit:

- Missing-connect validity gets direct access to the frontier states that can
  satisfy the connection.

Risks:

- Independent pooling of `F` and `G` may be too weak for the true existential
  pair condition.
- Nearest-distance ranking may miss a farther frontier that is topologically
  more useful.
- Counts and empty-set indicators are important; otherwise the MLP may confuse
  "no relevant frontier" with a zero-looking aggregate.

Fallback/upgrade:

- If pooled sets help but are too weak, add a small pair-aware term later.
  Possible forms:
  - top-k frontiers from each set by learned relevance, then evaluate `k x k`
    pairs;
  - mixed nearest/diverse frontier selection;
  - low-rank/bilinear compatibility summaries;
  - attention from one set into the other with bounded k.

## Step 2: Proposal Query Summary

After the missing-connect output query head is working, expose the same query
information to proposal scoring.

Construct per-frontier summaries of query participation:

- source-side participation: missing-connect queries where the frontier is in
  `F`;
- target-side participation: missing-connect queries where the frontier is in
  `G`;
- optional aggregate query context: projected connection/query embeddings or
  global query pools;
- count/empty features, normalized to avoid scale shifts when many queries are
  active.

Initial integration should be late and proposal-specific:

```text
proposal_state = frontier_state + zero_init_projection(query_summary)
```

or:

```text
proposal_state = ProposalStateMLP([frontier_state, query_summary])
```

with the new path initialized so the initial proposal behavior is unchanged.

Why late first:

- it isolates proposal-ranking effects from shared output heads;
- it avoids perturbing frontier message passing;
- it is easier to ablate and less likely to recreate the room-part-node
  regression.

Later, if the late summary helps but is too weak, try early query conditioning:

- add query summary features to the initial frontier state;
- allow normal frontier-frontier message passing to propagate and enrich them;
- keep this behind a separate toggle because it changes the shared frontier
  representation.

## Step 3: Missing-Connect Distance Query

After validity is stable, use the same query inputs for
`missing_connect_distance`.

This should remain a residual on the global distance head:

```text
distance = global_distance + query_distance_delta
```

Use the existing distance loss mask. Keep this separate from Step 1 because some
configs set `missing_connect_distance_weight` and
`reward_missing_connect_distance` to zero, so validity is the more important
first target.

When an `r -> s` path already exists, the distance target is determined by the
shortest existing path length `d_a`. A query is only needed if a future path
through frontiers could be shorter.

For the missing-connect sets:

- `F = {frontier | r can currently reach frontier}`;
- `G = {frontier | frontier can currently reach s}`;
- `d_F = min distance(r -> frontier)` over `F`;
- `d_G = min distance(frontier -> s)` over `G`.

If:

```text
d_F + d_G >= d_a
```

then no frontier-mediated path can improve the already-existing path, so the
distance outcome is determined and no distance query row should be emitted.

The same bound can prune individual frontier entries before CSR truncation:

- prune `f` from `F` when `distance(r -> f) + d_G >= d_a`;
- prune `g` from `G` when `d_F + distance(g -> s) >= d_a`.

This keeps distance queries focused on frontiers that can actually improve the
known shortest path. If no existing `r -> s` path exists, treat `d_a` as
infinite and keep the reachable frontier sets before normal CSR ranking and
truncation.

## Step 4: Early Query Conditioning Experiment

Only try this after the late proposal-query summary is understood.

Instead of adding query information only to the final proposal state, inject
query summaries into the frontier state before frontier message passing:

```text
initial_frontier_state = initial_frontier_state + zero_init_projection(query_summary)
```

This lets the network propagate query pressure through frontier-frontier message
passing before both proposal scoring and output prediction.

Potential upside:

- proposal state and output heads share a query-aware representation;
- nearby frontiers can exchange information about which missing-connect
  constraints they may satisfy;
- the model may learn richer interactions than late local summaries allow.

Risks:

- changes all downstream frontier representations;
- may perturb door, balance, and other output heads;
- can create broad "constraint pressure" signals that hurt calibration.

Use a separate toggle and compare against late-only integration.

## Step 5: Save/Refill Frontier Query Heads

Apply the same idea to save/refill utilities, but do not assume the current
feature set is sufficient.

For an active room part `p` and save/refill objective set `D`, the directional
targets are:

- `from_room`: path from `p` to some objective part in `D`;
- `to_room`: path from some objective part in `D` to `p`.

When the path is not already determined through placed rooms, the useful
frontier sets can be limited by the current best placed-objective distance.

For `from_room`, suppose the shortest already-known path from `p` to an
already-placed save/refill is `d_a`. For a frontier `f`, let `d_f` be the
shortest path length from `p` to `f`. A newly placed objective beyond `f` cannot
improve the target unless:

```text
d_f <= d_a - 1
```

The `-1` accounts for the fact that anything beyond the frontier must be at
least one step past the frontier. If `d_f > d_a - 1`, then even the best
possible future objective beyond `f` cannot beat the already-placed objective.

The reverse-direction outcome uses the analogous condition:

```text
distance(f -> p) <= d_a_reverse - 1
```

where `d_a_reverse` is the shortest already-known path from an already-placed
objective to `p`.

So the relevant sets are not simply all reachable frontiers. They are bounded
frontier sets:

- `from_room`: frontiers reachable from `p` with `distance(p -> frontier) <= d_a - 1`;
- `to_room`: frontiers that can reach `p` with `distance(frontier -> p) <= d_a_reverse - 1`.

If no already-placed objective is reachable in that direction, treat `d_a` as
infinite and allow all reachable frontiers in that direction.

If the relevant frontier set is empty, the outcome is already determined by the
currently placed graph and no query row is needed. In that case the model should
leave the existing global/determined prediction path alone rather than emitting
a local query residual.

This makes save/refill queries more local and cheaper than a naive room-part to
all-frontiers query, while still focusing exactly on frontiers that could change
the output target.

The current dense room-part distance features provide nearest-distance
summaries, and the room-part frontier distance cache already tracks directed
distances to frontier parts. The query feature should expose only the thresholded
frontier sets needed for pooling. Add this feature only if the missing-connect
query experiment is promising.

Preferred feature design:

- Avoid persistent room-part nodes.
- Add sparse query metadata for unresolved save/refill outputs rather than dense
  `frontier x room_part` tensors if possible.
- Emit query rows only for save/refill outcomes with a non-empty relevant
  frontier set.
- For each query row, store:
  - snapshot index;
  - room-part index;
  - output kind: save/refill and from/to;
  - current best placed-objective distance `d_a` for the relevant direction;
  - bounded frontier indices satisfying the relevance threshold;
  - distances or distance buckets for those frontier indices;
  - count/empty indicators for the full unbounded sets.

Start with bounded CSR-style query-frontier edges so memory is controlled and
predictable.

Output integration:

- Keep the existing global save/refill utility heads.
- Add zero-initialized residual deltas for queried active/unresolved rows only.
- Scatter the residuals into the dense `[snapshot, room_part]` utility outputs.
- Add matching proposal-query summaries for frontiers that participate in
  save/refill query sets.

## Step 6: Diagnostics And Ablations

Add toggles before enabling multiple heads at once:

- `missing_connect_frontier_query_outputs`;
- `missing_connect_proposal_query_summary`;
- `early_frontier_query_conditioning`;
- `missing_connect_distance_frontier_query_outputs`;
- `save_refill_frontier_query_outputs`.

Useful metrics:

- number of missing-connect query rows;
- average `F` count and `G` count;
- fraction of empty `F` or empty `G`;
- source/target frontier cap-hit rates;
- per-frontier source/target query participation counts;
- query residual magnitude by output type;
- proposal query residual magnitude;
- loss split for queried vs non-queried outputs;
- proposal loss and candidate diagnostics when query heads are enabled.

Important ablations:

- global heads only;
- missing-connect validity query only, without proposal query summary;
- missing-connect validity query plus late proposal query summary;
- missing-connect validity query plus early frontier query conditioning;
- missing-connect validity plus distance query;
- save/refill query only;
- all query heads.

## Implementation Notes

- Keep output tensor shapes unchanged.
- Keep config/checkpoint changes explicit and required.
- Prefer zero-initialized residuals over hard replacement.
- Keep the first implementation focused on missing-connect validity because the
  feature masks already exist.
- Avoid dense pairwise frontier computations unless the cheap pooled query is
  shown to be insufficient.
- Avoid reintroducing room-part graph nodes or room-part message passing.
- Keep late proposal summaries and early frontier conditioning as separate
  toggles.
