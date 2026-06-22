# Frontier Query Output Heads

## Context

Local outputs such as missing-connect validity and save/refill distances should
be predicted from the frontier states that can actually affect those outcomes.
An earlier room-part-node experiment attempted to add local room-part graph
nodes and message passing, but was slow and did not achieve good quality.

The current query-head approach keeps the useful part of that direction without
adding room-part graph nodes:

- local predictions can use frontier evidence tied to the specific unresolved
  outcome;
- query structure can condition the shared frontier representation before normal
  frontier-frontier message passing;
- proposal scoring and other output heads can see query pressure through the
  frontier states rather than through output heads alone.

## Current State

Steps 1 and 2 are implemented for missing-connect validity.

The implementation now has sparse, bounded missing-connect query features:

- `missing_connect_query_snapshot_idx`;
- `missing_connect_query_connection_idx`;
- `missing_connect_query_source_frontier`;
- `missing_connect_query_target_frontier`;
- `missing_connect_query_source_distance`;
- `missing_connect_query_target_distance`;
- `missing_connect_query_source_count`;
- `missing_connect_query_target_count`;
- `missing_connect_query_source_cap_hit`;
- `missing_connect_query_target_cap_hit`.

For a missing-connect outcome `r -> s`, the query rows use:

- `F = {frontier | r can currently reach frontier}`;
- `G = {frontier | frontier can currently reach s}`;
- bounded nearest frontiers on each side, ranked by directed graph distance;
- full source/target counts and cap-hit indicators.

Already-reachable `r -> s` connections do not emit missing-connect query rows.
The dense `frontier_connection_reachability` feature still exists separately,
but the missing-connect query path uses the sparse bounded tensors above.

The direct missing-connect validity query head is implemented as
`MissingConnectFrontierQueryHead`. It pools final frontier states over the
bounded source and target frontier sets, combines those pools with scalar
distance/count features and global state, and scatters query logits back to the
matching `connection_invalid` positions.

Queried missing-connect validity outputs use the query prediction as a
replacement for the global connection prediction:

```text
connection_invalid =
    where(query_connection_mask, query_connection_invalid, connection_invalid)
```

This replacement behavior is intentional. Query outputs are not residual terms
on top of global predictions. For queried rows, the query path owns the local
prediction; the global head remains responsible for rows without query output.

The early whole-query frontier summary is implemented as
`MissingConnectFrontierQuerySummary`. It pools initial frontier states over the
same source and target query sets, builds a whole-query embedding, scatters
source-side and target-side messages back to participating frontiers, and adds a
zero-initialized projected summary into the initial frontier state before
frontier-frontier message passing.

The current configs expose the implemented missing-connect switches as:

- `features.missing_connect_query`;
- `features.missing_connect_query_summary`;
- `generation.missing_connect_query_frontier_count`;
- `model.missing_connect_hidden_width`;
- `model.missing_connect_query_summary_hidden_width`.

## Design Preference

Query heads should replace global predictions for the queried output rows they
cover. They should not be added as residual deltas onto global predictions.

This is especially important for missing-connect validity: the outcome should
depend on whether some frontier in `F` can eventually connect to some frontier
in `G`, not on the identity of the room part where the missing-connect
originated. Replacement keeps that symmetry clearer.

For future query heads, keep the same rule unless there is a specific reason an
output is genuinely decomposed into a global term plus a local correction. If a
query row is emitted, the query prediction should be the prediction for that
row. If no query row is emitted because the outcome is already determined or not
eligible, leave the existing non-query prediction path responsible for that row.

New conditioning paths should continue to be initialized conservatively. The
projection that injects a query summary into the initial frontier state should
remain zero-initialized so enabling the path does not immediately perturb
behavior before training learns useful weights.

## Step 3: Missing-Connect Distance Query

Next, use the existing missing-connect query inputs for
`missing_connect_distance`.

The distance query should follow the same replacement semantics as validity:
for emitted distance query rows, the query distance prediction replaces the
global/non-query prediction for those rows. It should not be modeled as:

```text
distance = global_distance + query_distance_delta
```

Keep this separate from validity because some configs set
`missing_connect_distance_weight` and `reward_missing_connect_distance` to zero,
so distance training signal may be absent depending on the run.

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

The same bound can prune individual frontier entries before truncation:

- prune `f` from `F` when `distance(r -> f) + d_G >= d_a`;
- prune `g` from `G` when `d_F + distance(g -> s) >= d_a`.

If no existing `r -> s` path exists, treat `d_a` as infinite and keep reachable
frontier sets before normal ranking and truncation.

Implementation notes:

- reuse the current missing-connect query feature layout where possible;
- add distance-specific query rows only if validity rows are not sufficient for
  the distance eligibility rules;
- keep output tensor shapes unchanged;
- scatter query distance predictions into `missing_connect_distance` for queried
  rows only;
- preserve the existing distance loss mask.

Risks:

- pooled source/target sets may be too weak for the true best frontier-pair
  distance condition;
- nearest-distance truncation may discard a farther but more useful frontier;
- configs with zero distance reward/loss will not provide useful signal for
  this head.

Possible upgrades:

- top-k learned frontier relevance on each side, followed by bounded `k x k`
  pair scoring;
- mixed nearest/diverse frontier selection;
- low-rank or bilinear source/target compatibility summaries;
- bounded attention from one side into the other.

## Step 4: Save/Refill Query Extensions

After the missing-connect query summary is understood experimentally, apply the
same sparse query pattern to save/refill outcomes.

For a room part `p` with already-known shortest path length `d_a` to an
already-placed save/refill, a frontier at distance `d_f` from `p` is relevant
only if:

```text
d_f <= d_a - 1
```

Otherwise, a save/refill placed beyond that frontier cannot improve the known
outcome. If the relevant frontier set is empty, the outcome is already
determined and no query should be generated.

Use sparse bounded frontier sets and the same two-stage pattern:

1. build per-outcome query embeddings from the relevant frontier set and scalar
   distance/count features;
2. scatter whole-query embeddings back to participating frontiers for early
   conditioning;
3. add direct output heads for the corresponding save/refill utility or
   distance predictions once the conditioning path is stable.

For an active room part `p` and save/refill objective set `D`, the directional
targets are:

- `from_room`: path from `p` to some objective part in `D`;
- `to_room`: path from some objective part in `D` to `p`.

The relevant sets are:

- `from_room`: frontiers reachable from `p` with
  `distance(p -> frontier) <= d_a - 1`;
- `to_room`: frontiers that can reach `p` with
  `distance(frontier -> p) <= d_a_reverse - 1`.

If no already-placed objective is reachable in that direction, treat `d_a` as
infinite and allow all reachable frontiers in that direction.

Preferred feature design:

- avoid persistent room-part nodes;
- add sparse query metadata for unresolved save/refill outputs rather than
  dense `frontier x room_part` tensors if possible;
- emit query rows only for save/refill outcomes with a non-empty relevant
  frontier set;
- for each query row, store snapshot index, room-part index, output kind,
  current best placed-objective distance, bounded frontier indices, frontier
  distances or distance buckets, full unbounded count, and cap-hit indicator.

Output integration:

- keep existing global save/refill heads for non-queried rows;
- for queried rows, use direct query predictions as replacements, not residuals;
- scatter direct-query outputs into the dense `[snapshot, room_part]` utility or
  distance outputs;
- add matching whole-query early summaries for frontiers that participate in
  save/refill query sets.

Risks:

- early conditioning changes all downstream frontier representations;
- broad constraint-pressure signals may perturb door, balance, and other output
  heads;
- save/refill queries may be weaker than missing-connect queries if relevant
  frontier sets are large or noisy.

## Step 5: Diagnostics And Ablations

The implemented missing-connect query and query-summary switches already allow
basic ablations. Add new toggles before enabling additional query heads at once:

- `missing_connect_distance_query`;
- `save_refill_query_summary`;
- `save_refill_query_outputs`;
- optional global query-summary injection, if a global-only head needs it.

Useful metrics:

- number of missing-connect query rows;
- average `F` count and `G` count;
- fraction of empty `F` or empty `G`;
- source/target frontier cap-hit rates;
- per-frontier source/target query participation counts;
- query-summary update norm before and after the zero-initialized projection;
- direct-query output magnitude by output type;
- loss split for queried vs non-queried outputs;
- proposal loss and candidate diagnostics when query summaries are enabled.

Important ablations:

- global heads only;
- missing-connect validity query only, without early query summary;
- missing-connect validity query plus early whole-query frontier summary;
- early whole-query frontier summary without direct missing-connect output
  replacement;
- early whole-query frontier summary with and without global query summary;
- missing-connect validity plus distance query;
- missing-connect summary with and without connection-output identity embedding;
- save/refill query only;
- all query heads.

## Implementation Notes

- Keep output tensor shapes unchanged.
- Keep config/checkpoint changes explicit and required.
- Use replacement for queried output rows; do not add query outputs as residuals
  onto global predictions.
- Keep zero-initialized new conditioning projections so initial behavior is
  unchanged.
- Avoid dense pairwise frontier computations unless the cheap pooled query is
  shown to be insufficient.
- Avoid reintroducing room-part graph nodes or room-part message passing.
- Prefer shared early query conditioning over proposal-only special cases.
