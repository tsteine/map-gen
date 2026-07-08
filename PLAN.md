# Move Area Assignment Into Generation

## Goal

Area assignment currently happens after generation in
`python/area_assignment.py`. Move the primary six-area assignment into the
generation action so every placed room has an area immediately. The generation
action becomes:

- `room_idx`
- `room_x`
- `room_y`
- `room_area`

Subarea and subsubarea assignment remain serving-time post-processing.

This is intentionally backward-incompatible for configs, checkpoints, and
episode data. Missing config/checkpoint fields should fail clearly rather than
being defaulted.

Generated area IDs have fixed semantic meanings from the beginning:

- `0`: Crateria
- `1`: Brinstar
- `2`: Norfair
- `3`: Wrecked Ship
- `4`: Maridia
- `5`: Tourian

Future optional constraints and rewards will refer to these specific IDs, and
area size targets use this same order.

## Design Principles

- Treat area assignment as part of the candidate action space, not as a late
  repair step.
- Keep the proposal model's frontier-local action space aligned with Rust's
  proposal candidate mask and shortlist unpacking.
- Make area constraints explicit outcomes/rewards so the model can learn them,
  while still hard-masking area choices that are immediately impossible.
- Preserve candidate diversity by preventing one placement from filling the full
  scoring pool with all six area variants before other placements are tried.
- Keep room/area state in Rust as the source of truth during generation; Python
  should consume tensors exported by Rust rather than recomputing generation
  state.
- Leave extension points for future room-to-area preferences and requirements.
  Future rules should support both hard candidate masks for strict requirements
  and reward shaping for soft preferences.

## Phase 1: Data Model And Rust Environment State

Add a required area field to the core action representation.

- Extend `common::Action` with `area: u8` or a dedicated `AreaIdx` alias.
- Add per-room area storage to `Environment`, parallel to `room_x`, `room_y`,
  and `room_used`.
- Update `step`, `step_known`, lookahead apply/restore, feature planning, and
  action history storage to carry `room_area`.
- Add `room_area` to `get_actions` and candidate buffer outputs.
- Use an explicit dummy area value for dummy candidates. The dummy candidate
  should remain identifiable by dummy `room_idx`; Python selection should not
  depend on the dummy area value.
- Update Rust tests that construct `Action` with named fields.

Implementation notes:

- Keep area count centralized as `AREA_COUNT = 6` in Rust and Python, with the
  fixed semantic ID order listed in the goal section.
- Area values should be validated in Rust on step input; reject values outside
  `0..AREA_COUNT`.
- Existing `room_idx >= room_count` dummy handling should continue to short
  circuit geometry and feature application.

## Phase 2: Python Tensor Plumbing

Thread `room_area` through the Python wrappers and training data classes.

- Extend `env.Actions` with `room_area`, including `select`, `to`, and `slice`.
- Extend `EpisodeData`, `CandidateSlot`, `EnvironmentGroup.step`, `step_known`,
  `get_actions`, and candidate extraction to pass area tensors.
- Extend `ProposalData` with flattened `proposal_action_idx`, where
  `proposal_action_idx = door_variant_idx * AREA_COUNT + room_area`. Keep
  `proposal_door_variant_idx` and `proposal_room_area` only where useful as
  diagnostics or candidate metadata.
- Update training batch construction in `python/learn.py` so next-action tensors
  include the selected area.
- Update serialization/checkpoint/export paths that persist actions and proposal
  data.
- Update serving code to read final areas from generated episode actions instead
  of calling the six-area assignment search for the primary area assignment.

Tests:

- Add a small Rust/Python binding smoke test that steps known actions with areas
  and verifies `get_actions()` returns the same area sequence.
- Add a training data shape test to catch missing `room_area` fields early.

## Phase 3: Proposal Action Space Expansion

Represent proposal candidates as the Cartesian product of placement door variant
and area.

- Change `FrontierModel.proposal_output` width from `num_door_variants` to
  `num_door_variants * AREA_COUNT`.
- Define helper functions for flattening/unflattening:
  `proposal_action_idx = door_variant_idx * AREA_COUNT + room_area`.
- Use flattened `proposal_action_idx` for model/loss indexing, proposal masks,
  shortlist sampling, and cached proposal scores. Use helper functions to
  recover `door_variant_idx` and `room_area` for diagnostics.
- Update `ProposalCandidateMask` so the packed mask covers flattened proposal
  actions. A placement that is geometrically valid may still have only a subset
  of its six areas enabled.
- Update `sample_proposal_shortlist`, `row_scores_for_mask`,
  `compute_cached_proposal_scores`, `proposal_scores_for_frontier`, and
  `proposal_batch_loss` to use flattened proposal action indices.
- Ensure cached proposal scores remain valid when the next selected frontier is
  the same row but the mask differs by area validity.

Tests:

- Unit-test flatten/unflatten helpers.
- Unit-test proposal mask unpacking where only selected areas for one door
  variant are valid.
- Unit-test proposal loss indexing with two candidates sharing a door variant
  but using different areas.

## Phase 4: Immediate Area Validity Masking

Mask area choices that are impossible at candidate proposal time.

Initial hard mask:

- A candidate area is invalid if adding the candidate room to that area would
  make that area's bounding box exceed configured maximum dimensions.
- A candidate is invalid if it assigns a second map room to an area that already
  has one map room.
- A candidate is invalid if it creates a Toilet crossing where the Toilet room
  and crossed room are assigned to different areas. This should be checked both
  when placing the Toilet room and when placing a room that crosses an already
  placed Toilet room.

Required Rust state:

- Per-area `min_x`, `max_x`, `min_y`, `max_y`, and whether the area is used.
- Per-area map room counts.
- Per-room geometry bounds already exist conceptually in Python; move or expose
  equivalent geometry-bound calculations in Rust.
- Lookahead calculation for candidate bounding box validity without mutating the
  environment.
- Lookahead calculation for Toilet crossing area compatibility.

Config changes:

- Introduce generation config fields for area bounding box limits. Use required
  fields, not defaults.
- Decide whether existing serving fields `area_bounding_box_width` and
  `area_bounding_box_height` move into generation config, remain serving-only
  for subarea post-processing, or are duplicated with clearer names.

Important distinction:

- Bounding box max size is a hard candidate mask.
- Minimum/maximum occupied tile area is an outcome/reward and hard final
  validity criterion only. Do not use `min_area_size` or `max_area_size` for
  proposal-time pruning.

Tests:

- Rust unit test for area bounding box state after `step`.
- Rust unit test for proposal mask where a placement is valid in one area but
  invalid in another due to bounding box width/height.
- Rust unit test that a second map room in the same area is rejected.
- Rust unit test that Toilet/crossed-room area mismatch is rejected regardless
  of which room is placed second.

## Phase 5: Candidate Shortlist Diversity

Postpone duplicate placements with different area choices while processing the
shortlist.

Behavior:

- While scanning the sampled shortlist, identify each candidate by placement:
  `(frontier_idx, door_variant_idx)`. This postponement key is intentionally
  before concrete room representative selection.
- If a clean candidate with the same `(frontier_idx, door_variant_idx)` has
  already been evaluated in the current shortlist pass, push later area variants
  onto a postponed queue.
- If a rejected candidate has the same `(frontier_idx, door_variant_idx)` as an
  earlier evaluated candidate, skip it instead of adding it to the postponed
  queue.
- Continue scanning non-duplicate placements until the shortlist ends or
  `recommended_candidates` clean candidates are collected.
- If the end of the shortlist is reached and the clean pool is still short,
  process postponed candidates.
- Only after the normal and postponed queues fail to produce clean candidates
  should the fallback phase consider rejected candidates.

Implementation notes:

- Count postponed evaluations separately in profiling if useful; at minimum keep
  existing evaluated/rejected/clean counts meaningful.
- The postponed queue should preserve shortlist order.
- A postponed candidate that resolves to no action should be ignored, like the
  current invalid proposal entries.
- Concrete room selection from `(frontier_idx, door_variant_idx)` should still
  happen at the end, when the candidate is actually evaluated. If multiple
  concrete rooms are associated with the same key, they should not fill the full
  scoring pool ahead of other shortlist keys; they are considered only through
  normal postponed processing after other shortlist candidates are exhausted.

Tests:

- Rust unit test that a shortlist with one placement's six area variants and a
  second placement considers the second placement before later area variants of
  the first.
- Rust unit test that postponed candidates are considered before rejected
  fallback candidates.

## Phase 6: Area Outcomes

Add generated outcomes for area quality and validity.

New outcomes:

- `area_connected_components`: shape `[batch, time, AREA_COUNT]`, count of
  connected components per area in the room graph. Zero means the area is
  unused; one means used and connected; values above one are disconnected.
- `area_crossings`: shape `[batch, time]`, count of matched doors whose rooms
  are assigned different areas.
- `area_size`: shape `[batch, time, AREA_COUNT]`, occupied tile count per area.
- `area_map_station_count`: shape `[batch, time, AREA_COUNT]`, count of map
  rooms assigned to each area. A valid completed map has exactly one map room in
  every area.

Rust calculations:

- Derive a room-level undirected graph from Rust's existing `door_matches`
  state. Rust already maintains door matches and the directed room-part/SCC
  graph during generation, but it does not currently maintain the serving-time
  room-node graph built by `python/area_assignment.py`.
- For the derived room graph, nodes are placed rooms and edges are symmetric
  door matches.
- Count area crossings from matched door pairs; count each match once.
- Maintain or compute occupied tile counts per area from room tile counts.
- Maintain or compute map room counts per area from room metadata and
  `room_area`.
- Export these outcomes as final episode outcomes in `EndOutcomes`; do not add
  them as direct `StepOutcomes` reward targets.
- When some area outcome information is already knowable before the end of the
  episode, expose that information through lookahead-outcome features computed
  from post-candidate state rather than by making the outcomes step-level reward
  targets.

Implementation note:

- Start with an on-demand derivation from `door_matches` for correctness and
  simplicity. Add an incremental room-area connectivity structure only if
  profiling shows the on-demand scan is too expensive.

Python data model:

- Extend `EndOutcomes` with the new tensors.
- Extend `to` and `slice` methods for final outcome tensors.
- Add consistency checks that outcome widths are exactly `AREA_COUNT`.

Feature design note:

- Review which area quantities are knowable from post-candidate state and should
  be exposed as lookahead features. The planned feature list already includes
  current area sizes, current area bounding boxes, current area connected
  component counts, current map station counts, current area crossing count, and
  frontier-node area. If other known post-candidate quantities would materially
  help predict final area outcomes, add them explicitly to the feature list
  before implementation.

## Phase 7: Model Features, Heads And Losses

Add input features, prediction heads, and training losses for area outcomes.

Input features:

- Global feature: current occupied tile count for each area, width `AREA_COUNT`.
- Global feature: current area bounding boxes, width `AREA_COUNT * 4` for
  `min_x`, `max_x`, `min_y`, and `max_y`.
- Global feature: current connected component count for each area, width
  `AREA_COUNT`.
- Global feature: current map station count for each area, width `AREA_COUNT`.
- Global feature: current area crossing count, scalar.
- Frontier-node feature: area of the already-placed room at the frontier.

Implementation notes:

- Area bounding box features need an explicit representation for unused areas,
  because their min/max coordinates are otherwise undefined. Prefer adding an
  area-used mask or a clearly documented sentinel representation instead of
  relying on implicit zero/default behavior.
- These features should be controlled by required feature config fields,
  matching the existing feature-gating pattern.

Model predictions:

- `area_used`: log-odds that each area has at least one assigned room, width
  `AREA_COUNT`.
- `area_excess_components`: expected excess connected components for each area,
  width `AREA_COUNT`, where the training target is
  `max(area_connected_components - 1, 0)`.
- `area_crossings`: scalar regression/count prediction.
- `area_size`: bucketed classifier with three buckets per area: below
  `min_area_size`, in range, and above `max_area_size`, for total width
  `AREA_COUNT * 3`.
- `area_map_station_count`: bucketed classifier with three buckets per area:
  `0`, `1`, and `2+`, for total width `AREA_COUNT * 3`.
- Later: room-to-area requirement/preference heads can be added without changing
  the action representation. The framework should allow individual rules to opt
  into either hard candidate masking or reward-only shaping.

Training config:

- Add required train weights: `area_used_weight`,
  `area_excess_components_weight`, `area_crossing_weight`,
  `area_size_weight`, and `area_map_station_weight`.
- Add required area size config: `min_area_size` `max_area_size`.
- Defer generation reward fields (`reward_area_connected`,
  `reward_area_used`, `reward_area_crossing`, `reward_area_size_valid`) to
  Phase 8, where prediction outputs are incorporated into candidate reward.
- Defer `target_area_size` and `reward_area_size` to a later follow-up. When
  added, `target_area_size` should be ordered as
  `[Crateria, Brinstar, Norfair, Wrecked Ship, Maridia, Tourian]`.

Loss/reward handling:

- Train `area_used` with binary cross-entropy using
  `area_connected_components > 0` as the target.
- Train `area_excess_components` with MSE against
  `max(area_connected_components - 1, 0)`.
- Train area crossings as non-negative counts.
- Train `area_map_station_count` with cross-entropy over three buckets per area:
  `0`, `1`, and `2+`.
- Train `area_size` with cross-entropy over three buckets per area: below
  `min_area_size`, in range, and above `max_area_size`.
- These 3-bucket heads use softmax classification losses, not separate BCE
  heads.
- Consider normalization for count/size targets so loss scales are stable.
- Update checkpoint metadata expectations so missing new heads fail clearly.

Tests:

- Unit-test feature extraction shapes and values for the new global area
  features, including map station counts and area crossing count.
- Unit-test the frontier-node area feature on a frontier attached to a known
  placed room.
- Unit-test model output shapes from `get_predictions`.
- Unit-test loss masking and repeated-outcome construction for area tensors.
- Add a tiny training batch smoke test after fixtures are updated.

## Phase 8: Reward Integration

Incorporate area predictions into `compute_expected_reward`.

Reward terms:

- Area connected:
  `-reward_area_connected * sum(predicted_area_excess_components)`
- Area used: `+reward_area_used * sum(sigmoid(area_used_logits))`
- Area crossings: `-reward_area_crossing * predicted_area_crossings`
- Area map stations: positive reward terms based on the predicted probability of
  the `1` bucket for each area's `area_map_station_count` classifier. The `0`
  and `2+` buckets are distinct training targets because they represent
  different failure modes, even though reward only uses the `1` bucket
  probability. Candidate generation also hard-rejects known second map stations
  in the same area.
- Area size validity: positive reward terms based on the predicted probability
  of the in-range bucket for each area's `area_size` classifier.
- For both 3-bucket classifiers, compute reward probabilities by applying
  softmax to the bucket logits and selecting the middle bucket.
- Deferred area target size: later add
  `-reward_area_size * sum((predicted_area_size - target_area_size)^2)`.

## Phase 9: Semi-Finalized Area Lookahead

Add lookahead validity rules for areas that have no remaining frontiers.

- Add a "semi-finalized area" concept. An area is semi-finalized when it has at
  least one placed room but no frontier whose source room is assigned to that
  area.
- Once an area is semi-finalized, assigning a later room to that area would make
  the area disconnected, or increase its connected component count. Reject clean
  candidates that assign rooms to semi-finalized areas.
- If a candidate causes an area to become semi-finalized before that area has a
  map station, then the area cannot later gain a map station without becoming
  disconnected. Reject clean candidates that create this state.
- These rules are "semi" final because rejected fallback candidates can still be
  applied when no clean candidate exists, which may allow the area to exit this
  state.
- This phase can be implemented after the initial area-action generation path is
  working, because it refines candidate validity rather than defining the core
  action representation.

Tests:

- Rust unit test that clean candidates assigning rooms to a semi-finalized area
  are rejected.
- Rust unit test that clean candidates creating a semi-finalized area with no
  map station are rejected.
- Rust unit test that rejected fallback application can still change
  semi-finalized state when no clean candidate exists.

## Phase 10: Serving-Time Changes

Remove primary area assignment from serving post-processing.

- Replace `assign_room_areas(...)` as the source of `final_area_list` with
  generated `episode_data.actions.room_area`.
- Keep subarea/subsubarea generation in serving. Refactor
  `python/area_assignment.py` so reusable helpers for adjacency, crossings, and
  subarea splitting are separate from the old six-area search.
- Remove serving-time enforcement of the primary map-station and Toilet
  same-area constraints once generation enforces them and exports the
  corresponding map-station outcome.
- Ensure `small_map` pruning receives generated area values unchanged.

Tests:

- Update `python/test_area_assignment.py` to focus on remaining helper behavior.
- Update serving request tests to expect generated areas instead of sampled
  post-processing areas.

## Phase 11: Config And Checkpoint Migration

Make the break intentional and easy to diagnose.

- Update all configs with required area fields.
- Update `GENERATION_VARIABLE_FLOAT_FIELDS` with new reward fields that should
  be scheduleable/model-visible.
- Update model export/loading metadata to include new output widths and area
  action representation.
- Remove any compatibility fallback that silently fabricates `room_area`.
- Add clear validation errors for:
  - wrong `target_area_size` length once target-size rewards are added
  - invalid min/max area sizes
  - proposal output width not divisible by `AREA_COUNT`
  - action tensors missing `room_area`

## Phase 12: Validation And Performance

Run correctness and performance checks after the full path is wired.

- Rust unit tests: `cargo test`.
- Python binding rebuild: `maturin develop` in the `map-gen` conda environment.
- Python tests in the `map-gen` conda environment.
- Generation smoke test with a small config.
- Serving smoke test using `scripts/sample_generate_request.py` or equivalent.
- Compare profile counters before/after:
  - proposal mask time
  - candidate resolve time
  - candidate evaluation count
  - postponed candidate count if added
  - GPU proposal scoring time

Performance risks:

- Proposal output width grows 6x.
- Proposal masks grow 6x.
- Shortlist sampling may need a larger `shortlist_candidates` to preserve the
  same placement diversity if many area variants are valid.
- Per-candidate area outcome computation can be expensive if it repeatedly
  derives the room-level graph from `door_matches` and recomputes connected
  components. Maintain area connected components incrementally during generation
  instead, with lookahead rollback support.

## Open Questions

- Should `area_connected_components > 0` replace `area_used` conceptually as
  the source of truth for whether an area is used, or should `area_used` remain
  as separate explicit state for bbox logic and readability?
