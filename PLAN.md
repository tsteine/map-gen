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

Postpone area variants beyond the configured per-placement limit while
processing the shortlist.

Behavior:

- While scanning the sampled shortlist, identify each candidate by placement:
  `(frontier_idx, door_variant_idx)`. This postponement key is intentionally
  before concrete room representative selection.
- Track the number of clean candidates accepted for each
  `(frontier_idx, door_variant_idx)` in the current shortlist pass.
- If the count for a placement reaches
  `generation.max_candidate_areas_per_placement`, push later area variants onto
  a postponed queue without evaluating them in the initial pass.
- If a rejected candidate has the same `(frontier_idx, door_variant_idx)` as an
  earlier evaluated candidate, do not count it against the per-placement limit.
- Continue scanning placements that are still under the per-placement limit until
  the shortlist ends or `recommended_candidates` clean candidates are collected.
- If the end of the shortlist is reached and the clean pool is still short,
  process postponed candidates.
- Only after the normal and postponed queues fail to produce clean candidates
  should the fallback phase consider rejected candidates.

Implementation notes:

- Count postponed evaluations separately in profiling if useful; at minimum keep
  existing evaluated/rejected/clean counts meaningful.
- Add required config field `generation.max_candidate_areas_per_placement` and
  validate it is in `[1, AREA_COUNT]`. Use `2` in checked-in configs initially.
- The postponed queue should preserve shortlist order.
- A postponed candidate that is invalid should be ignored, like the
  current invalid proposal entries.
- Concrete room selection from `(frontier_idx, door_variant_idx)` should still
  happen at the end, when the candidate is actually evaluated. If multiple
  concrete rooms are associated with the same key, they should not fill the full
  scoring pool ahead of other shortlist keys; they are considered only through
  normal postponed processing after other shortlist candidates are exhausted.

Tests:

- Rust unit test that `max_candidate_areas_per_placement = 1` preserves the
  original duplicate-postponement behavior.
- Rust unit test that `max_candidate_areas_per_placement = 2` accepts two clean
  area variants from the same placement before postponing the third.
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

- `area_connected_component_bucket_logits`: bucketed classifier for each area's
  connected component count. Configure the inclusive bucket upper bounds with
  `train.area_connected_component_bucket_upper_bounds`; the initial value is
  `[0, 1, 2, 3]`, producing buckets `0`, `1`, `2`, `3`, and `4+`. The first
  two buckets must remain exactly `0` and `1` so the valid outcome is always
  bucket index `1`.
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

- Add required train weights: `area_connected_component_weight`,
  `area_crossing_weight`, `area_size_weight`, and `area_map_station_weight`.
- Add required connected-component bucket config:
  `area_connected_component_bucket_upper_bounds`.
- Add required area size config: `min_area_size` `max_area_size`.
- Defer generation reward fields (`reward_area_connected`,
  `reward_area_connected_excess`, `reward_area_crossing`,
  `reward_area_size_valid`) to Phase 8, where prediction outputs are
  incorporated into candidate reward.
- Defer `target_area_size` and `reward_area_size` to a later follow-up. When
  added, `target_area_size` should be ordered as
  `[Crateria, Brinstar, Norfair, Wrecked Ship, Maridia, Tourian]`.

Loss/reward handling:

- Train `area_connected_component_bucket_logits` with cross-entropy over the
  configured component-count buckets.
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

- Area connected validity:
  `reward_area_connected * sum(log P(area_connected_components == 1))`
- Area connected excess shaping:
  `-reward_area_connected_excess * sum(E[max(area_connected_components - 1, 0)])`,
  with the expectation derived from the connected-component bucket
  probabilities and each bucket's representative lower-bound component count.
- Area crossings: `-reward_area_crossing * predicted_area_crossings`
- Area map stations:
  `reward_area_map_station * sum(log P(area_map_station_count == 1))`.
- Area size validity:
  `reward_area_size_valid * sum(log P(min_area_size <= area_size <= max_area_size))`.
- Use `log_softmax` for connected validity, map-station validity, and area-size
  validity so these rewards behave like other validity log-probability terms.
- Deferred area target size: later add
  `-reward_area_size * sum((predicted_area_size - target_area_size)^2)`.

## Phase 9: Finalized Area Lookahead

Expose exact area-size and map-station buckets when their final values become
known, and reject candidates that newly make either outcome invalid.

- An area is finalized once it has at least one placed room and no usable
  frontier whose source room belongs to that area. A frontier with no remaining
  candidates is exhausted and does not keep the area open.
- A finalized area exposes its exact size bucket (below range, valid range, or
  above range) and map-station bucket (zero, one, or two-or-more).
- Size becomes known as above-range immediately after exceeding
  `max_area_size`, even before finalization. Map-station count likewise becomes
  known as two-or-more immediately after exceeding one station.
- Unused areas remain unknown until generation finishes. At finish, all area
  buckets become exact.
- Reject candidates that transition either area bucket from unknown to an
  invalid value. Keep those candidates in the existing fallback pool, and do
  not repeatedly reject later candidates for an invalid outcome already
  committed through fallback.
- Allow second-map-station proposals to reach lookahead instead of filtering
  them during proposal resolution, so they can be rejected normally and expose
  the known two-or-more feature value.
- Add exact categorical area buckets to lookahead model inputs. Do not override
  area prediction logits or area reward calculations with the known values.

Tests:

- Rust unit tests for finalized, oversized, and two-or-more-station bucket
  outcomes, including exhausted frontier candidate lists.
- Rust unit tests that newly invalid candidates are rejected but remain usable
  as fallback, and committed invalid outcomes do not trigger repeated rejection.
- Python tests for outcome buffer plumbing and categorical lookahead encoding.

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
