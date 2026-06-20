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
- proposal relevance and broader outcome consistency, by exposing query
  structure to the shared frontier representation rather than only to output
  heads.

But it should avoid the main costs and risks of the room-part-node approach:

- no large population of extra graph nodes;
- no frontier state perturbation from room-part messages;
- no expensive bidirectional part/frontier message passing;
- no need to tune room-part embedding widths.

## Core Idea

For an output whose truth depends on future connections through frontiers,
construct one or more frontier sets and aggregate frontier states over those
sets.

For a missing-connect outcome `r -> s`:

- `F = {frontier | r can currently reach frontier}`
- `G = {frontier | frontier can currently reach s}`
- the connection can become valid if future placements create some path from a
  frontier in `F` to a frontier in `G`.

The direct query head receives aggregate embeddings for `F` and `G`, plus small
scalar/count features, and predicts the corresponding output.

The next step is to build a query summary that mirrors the same whole-query
construction using the initial frontier states, then scatter the resulting query
embeddings back to the participating frontiers before frontier-frontier message
passing. This gives proposal scoring and other output heads access to the
structure of active missing-connect constraints, not only the final
missing-connect output head.

The current code already computes the key missing-connect masks as
`frontier_connection_reachability`: bit 1 means `r -> frontier`, and bit 2 means
`frontier -> s`.

## Design Preference

Missing-connect validity should be predicted by the query path for query rows
rather than added as a residual on top of the global connection head. The
missing-connect outcome is intended to depend on whether some frontier in `F`
will connect to some frontier in `G`, not on the identity of the room part where
the missing-connect originated. Using the query prediction as the replacement
keeps that symmetry clearer.

New conditioning paths should still be initialized conservatively. In
particular, the projection that injects a query summary into the initial
frontier state should be zero-initialized:

```text
initial_frontier_state =
    initial_frontier_state + zero_init_projection(query_summary)
```

This keeps initial behavior unchanged while allowing training to decide how
much query structure should affect message passing, proposal scoring, and other
output heads.

## Step 1: Missing-Connect Frontier Query Head

Start with missing-connect validity. The existing dense
`frontier_connection_reachability` feature proves the needed reachability sets
are already available, but the implementation should use sparse CSR-style query
rows so memory and fan-in are controlled.

Implementation shape:

- Keep the existing global `connection_output` for connection outputs that do
  not have query rows.
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
  - scatter the query prediction to the corresponding `connection_invalid`
    logit and use it in place of the global prediction for that queried output.

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

## Step 2: Early Whole-Query Summary

After the missing-connect output query head is working, expose the structure of
active missing-connect queries to the shared frontier representation before
frontier-frontier message passing.

The summary should mirror the direct query construction, but use the initial
frontier state instead of the final frontier state:

```text
source_pool_q = pool(initial_frontier_state over F_q)
target_pool_q = pool(initial_frontier_state over G_q)

query_embedding_q = QuerySummaryMLP([
    source_pool_q,
    target_pool_q,
    source_count/count flags/distances,
    target_count/count flags/distances,
])
```

Then scatter the full query embedding back to the frontiers that participate in
that query:

```text
for f in F_q:
    source_message(f, q) = SourceSideMLP([
        query_embedding_q,
        distance(r -> f),
    ])

for g in G_q:
    target_message(g, q) = TargetSideMLP([
        query_embedding_q,
        distance(g -> s),
    ])
```

Aggregate these messages per packed frontier row:

```text
source_summary_f = mean source_message(f, q) over source-side query edges
target_summary_f = mean target_message(f, q) over target-side query edges

query_summary_f = [
    source_summary_f,
    target_summary_f,
    log_source_participation_count,
    log_target_participation_count,
    source_any,
    target_any,
]
```

Finally inject the summary into the initial frontier state:

```text
initial_frontier_state =
    initial_frontier_state + zero_init_projection(query_summary_f)
```

This is different from marginal participation statistics. A source frontier in
`F_q` receives a message derived from both `F_q` and `G_q`, so the summary
preserves the structure of the full query instead of only saying that the
frontier participated on one side of some query.

Implementation notes:

- reuse the existing bounded sparse query tensors:
  - `source_frontier`: `[query_count, k]`;
  - `target_frontier`: `[query_count, k]`;
  - `source_distance`: `[query_count, k]`;
  - `target_distance`: `[query_count, k]`;
- flatten valid source and target entries into query/frontier edges;
- convert local frontier indices to packed frontier rows using
  `row_start_by_snapshot + local_frontier`;
- use `index_add_`/`scatter_add_` plus counts to compute per-frontier means;
- keep source-side and target-side transforms separate at first;
- do not feed query logits back into the summary, to avoid a circular dependency
  between query-conditioned frontier states and query predictions;
- do not introduce pairwise `F x G` computation initially.

Expected benefit:

- proposal scoring sees query structure through the final frontier state;
- door validity and other heads can respond to query pressure through the shared
  frontier representation and global frontier pooling;
- the final missing-connect query head can use frontier states that have already
  propagated query context through normal message passing.

Optional secondary path:

- add a zero-initialized per-snapshot global query summary built from
  `query_embedding_q` into `global_state`;
- this may help global-only heads, but it should not replace the frontier-local
  scatter-back path because global aggregation loses which frontiers participate
  in each query.

Possible ablations:

- no missing-connect identity embedding vs a small learned connection-output
  embedding in `query_embedding_q`;
- mean-only scatter aggregation vs mean plus max;
- separate source/target side MLPs vs shared MLP with side embedding;
- early query summary enabled without the direct output query head, to test
  whether shared conditioning alone carries useful signal.

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

## Step 4: Save/Refill Query Extensions

After the missing-connect query summary is understood, apply the same pattern to
save/refill outcomes.

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
2. scatter those whole-query embeddings back to the participating frontiers for
   early conditioning;
3. add direct output heads for the corresponding save/refill utility or
   distance predictions once the conditioning path is stable.

For an active room part `p` and save/refill objective set `D`, the directional
targets are:

- `from_room`: path from `p` to some objective part in `D`;
- `to_room`: path from some objective part in `D` to `p`.

The relevant sets are:

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
query summary experiment is promising.

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
- For direct output heads, start with zero-initialized residual deltas for
  queried active/unresolved rows only, unless experiments suggest replacement is
  theoretically cleaner for a specific output.
- Scatter the direct-query outputs into the dense `[snapshot, room_part]`
  utility outputs.
- Add matching whole-query early summaries for frontiers that participate in
  save/refill query sets.

Risks:

- early conditioning changes all downstream frontier representations;
- broad "constraint pressure" signals may perturb door, balance, and other
  output heads;
- save/refill queries may be weaker than missing-connect queries if relevant
  frontier sets are large or noisy.

## Step 5: Diagnostics And Ablations

Add toggles before enabling multiple heads at once:

- `missing_connect_frontier_query_outputs`;
- `missing_connect_frontier_query_summary`;
- `missing_connect_query_summary_global_state`;
- `missing_connect_distance_frontier_query_outputs`;
- `save_refill_frontier_query_summary`;
- `save_refill_frontier_query_outputs`.

Useful metrics:

- number of missing-connect query rows;
- average `F` count and `G` count;
- fraction of empty `F` or empty `G`;
- source/target frontier cap-hit rates;
- per-frontier source/target query participation counts;
- query-summary update norm before and after the zero-initialized projection;
- direct-query output magnitude by output type;
- loss split for queried vs non-queried outputs;
- proposal loss and candidate diagnostics when query heads are enabled.

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
- Use replacement for queried missing-connect validity outputs; keep residuals
  available for outputs where the global head remains theoretically useful.
- Zero-initialize new conditioning projections so initial behavior is unchanged.
- Keep the first implementation focused on missing-connect validity because the
  feature masks already exist.
- Avoid dense pairwise frontier computations unless the cheap pooled query is
  shown to be insufficient.
- Avoid reintroducing room-part graph nodes or room-part message passing.
- Prefer shared early query conditioning over proposal-only special cases.
