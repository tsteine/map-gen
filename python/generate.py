from __future__ import annotations

from env import Actions, EnvironmentGroup, GenerateConfig, Outcomes, SparseStateFeatures, StateFeatures
from model import Predictions
from profile_stats import ProfileStats
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import torch


KNOWN_INVALID_REWARD = -100.0


class Prefetcher:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)

    def close(self):
        self.executor.shutdown()

    def map(self, items, prepare, profiler=None, wait_name=None):
        items = iter(items)
        try:
            future = self.executor.submit(prepare, next(items))
        except StopIteration:
            return
        for item in items:
            if profiler is None or wait_name is None:
                result = future.result()
            else:
                with profiler.timer(wait_name):
                    result = future.result()
            future = self.executor.submit(prepare, item)
            yield result
        if profiler is None or wait_name is None:
            yield future.result()
        else:
            with profiler.timer(wait_name):
                yield future.result()


def rand_choice(p):
    cumul_p = torch.cumsum(p, dim=1)
    rnd = torch.rand([p.shape[0], 1], device=p.device)
    choice = torch.clamp(torch.searchsorted(cumul_p, rnd), max=p.shape[1] - 1).view(-1)
    return choice


def outcome_reward(model_logprobs: torch.Tensor, known_invalid: torch.Tensor) -> torch.Tensor:
    if known_invalid.ndim == model_logprobs.ndim - 1:
        known_invalid = known_invalid.unsqueeze(1)
    known_valid_reward = torch.zeros_like(model_logprobs)
    known_invalid_reward = torch.full_like(model_logprobs, KNOWN_INVALID_REWARD)
    known_reward = torch.where(known_invalid == 0, known_valid_reward, known_invalid_reward)
    return torch.where(known_invalid < 0, model_logprobs, known_reward)


# preds.door_invalid: [batch_size, max_candidates, num_outputs]
# preds.connection_invalid: [batch_size, max_candidates, num_outputs]
def compute_expected_reward(preds, outcomes, config: GenerateConfig):
    door_logprobs = torch.nn.functional.logsigmoid(-preds.door_invalid)
    connection_logprobs = torch.nn.functional.logsigmoid(-preds.connection_invalid)
    door_logprobs = outcome_reward(door_logprobs, outcomes.door_invalid)
    connection_logprobs = outcome_reward(connection_logprobs, outcomes.connection_invalid)
    return torch.sum(door_logprobs, dim=2) + torch.sum(connection_logprobs, dim=2)


def select_outcomes(outcomes: Outcomes, index: torch.Tensor) -> Outcomes:
    gather_index = index.view(-1, 1, 1)
    return Outcomes(
        door_invalid=torch.gather(
            outcomes.door_invalid, 1, gather_index.expand(-1, 1, outcomes.door_invalid.shape[2])
        ).squeeze(1),
        connection_invalid=torch.gather(
            outcomes.connection_invalid,
            1,
            gather_index.expand(-1, 1, outcomes.connection_invalid.shape[2]),
        ).squeeze(1),
    )


def merge_verified_outcomes(
    known_outcomes: Outcomes | None,
    current_outcomes: Outcomes,
    stage: str,
) -> Outcomes:
    if known_outcomes is None:
        return current_outcomes

    def merge_known(known: torch.Tensor, current: torch.Tensor, outcome_name: str):
        inconsistent = (known >= 0) & (current >= 0) & (known != current)
        if torch.any(inconsistent):
            first_idx = torch.nonzero(inconsistent, as_tuple=False)[0].tolist()
            invalid_to_valid = torch.sum((known == 1) & (current == 0)).item()
            valid_to_invalid = torch.sum((known == 0) & (current == 1)).item()
            raise RuntimeError(
                f"{outcome_name} outcome changed after becoming known at {stage}: "
                f"first index {first_idx}, invalid->valid {invalid_to_valid}, "
                f"valid->invalid {valid_to_invalid}"
            )
        return torch.where(known >= 0, known, current)

    return Outcomes(
        door_invalid=merge_known(
            known_outcomes.door_invalid, current_outcomes.door_invalid, "door"
        ),
        connection_invalid=merge_known(
            known_outcomes.connection_invalid,
            current_outcomes.connection_invalid,
            "connection",
        ),
    )


def extract_candidate_features(
    env: EnvironmentGroup,
    chunk: Actions,
    env_start: int,
    env_end: int,
    profiler: ProfileStats,
    sparse_frontiers: bool = False,
    feature_slot: PinnedSparseStateFeatureSlot | None = None,
):
    env_chunk = Actions(
        chunk.room_idx[env_start:env_end],
        chunk.room_x[env_start:env_end],
        chunk.room_y[env_start:env_end],
    )
    with profiler.timer("gen.cpu_extract"):
        if sparse_frontiers and feature_slot is not None:
            frontier_count, sparse_row_count, worker_sparse_row_counts = (
                env.env.get_sparse_state_feature_requirements_after_candidates(
                    env_chunk.room_idx.contiguous().cpu().numpy(),
                    env_chunk.room_x.contiguous().cpu().numpy(),
                    env_chunk.room_y.contiguous().cpu().numpy(),
                    env_start,
                )
            )
            feature_slot.ensure(
                (env_end - env_start) * env_chunk.room_idx.shape[1],
                sparse_row_count,
                profiler,
            )
            env.env.get_sparse_state_features_after_candidates_into(
                env_chunk.room_idx.contiguous().cpu().numpy(),
                env_chunk.room_x.contiguous().cpu().numpy(),
                env_chunk.room_y.contiguous().cpu().numpy(),
                env_start,
                frontier_count,
                sparse_row_count,
                worker_sparse_row_counts,
                feature_slot.inventory.numpy(),
                feature_slot.room_x.numpy(),
                feature_slot.room_y.numpy(),
                feature_slot.room_placed.numpy(),
                feature_slot.frontier.numpy(),
                feature_slot.frontier_occupancy.numpy(),
                feature_slot.frontier_neighbor.numpy(),
                feature_slot.frontier_neighbor_pair.numpy(),
                feature_slot.connection_reachability.numpy(),
                feature_slot.frontier_connection_reachability.numpy(),
                feature_slot.dense_row_idx.numpy(),
            )
            features = feature_slot.features(
                env_end - env_start,
                env_chunk.room_idx.shape[1],
                sparse_row_count,
                frontier_count,
            ).flatten_candidates()
        elif sparse_frontiers:
            features = env.get_sparse_state_features_after_candidates(
                env_chunk, torch.device("cpu"), env_start
            ).flatten_candidates()
        else:
            features = env.get_state_features_after_candidates(
                env_chunk, torch.device("cpu"), env_start
            ).flatten_candidates()
    if profiler.enabled:
        (
            worker_seconds,
            pack_seconds,
            profile_calls,
            snapshot_apply_cpu_seconds,
            restore_cpu_seconds,
            assemble_cpu_seconds,
            assemble_setup_cpu_seconds,
            assemble_frontier_cpu_seconds,
            assemble_neighbor_cpu_seconds,
            assemble_pair_cpu_seconds,
            assemble_pair_flags_cpu_seconds,
            assemble_output_cpu_seconds,
        ) = env.take_state_feature_profile()
        if profile_calls != 1:
            raise RuntimeError(
                f"unexpected state feature profile call count: {profile_calls}"
            )
        profiler.add("gen.cpu_extract_worker", worker_seconds)
        profiler.add("gen.cpu_extract_pack", pack_seconds)
        profiler.add("gen.cpu_extract_snapshot_apply_sum", snapshot_apply_cpu_seconds)
        profiler.add("gen.cpu_extract_restore_sum", restore_cpu_seconds)
        profiler.add("gen.cpu_extract_assemble_sum", assemble_cpu_seconds)
        profiler.add("gen.cpu_extract_assemble_setup_sum", assemble_setup_cpu_seconds)
        profiler.add("gen.cpu_extract_assemble_frontier_sum", assemble_frontier_cpu_seconds)
        profiler.add("gen.cpu_extract_assemble_neighbor_sum", assemble_neighbor_cpu_seconds)
        profiler.add("gen.cpu_extract_assemble_pair_sum", assemble_pair_cpu_seconds)
        profiler.add("gen.cpu_extract_assemble_pair_flags_sum", assemble_pair_flags_cpu_seconds)
        profiler.add("gen.cpu_extract_assemble_output_sum", assemble_output_cpu_seconds)
    return env_start, env_end, features


def transfer_state_features(
    features: StateFeatures | SparseStateFeatures,
    device: torch.device,
    profiler: ProfileStats,
    transfer_stream: torch.cuda.Stream | None = None,
) -> StateFeatures:
    if isinstance(features, SparseStateFeatures):
        if transfer_stream is None or device.type != "cuda":
            return transfer_state_features_sync(features, device, profiler)
        current_stream = torch.cuda.current_stream(device)
        with torch.cuda.device(device), torch.cuda.stream(transfer_stream):
            result = transfer_state_features_sync(features, device, profiler, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(transfer_stream)
        current_stream.wait_event(ready)
        return result
    return features.to(device)


def transfer_state_features_sync(
    features: SparseStateFeatures,
    device: torch.device,
    profiler: ProfileStats,
    non_blocking: bool = False,
) -> StateFeatures:
    dense_shape = (features.inventory.shape[0], features.frontier_count)
    with profiler.timer("gen.cpu_transfer_sparse_index"):
        dense_row_idx = features.dense_row_idx.to(device, non_blocking=non_blocking)

    def densify(value: torch.Tensor, fill_value: int = 0):
        with profiler.timer("gen.cpu_frontier_dense_init_submit"):
            dense_value = torch.full(
                (*dense_shape, *value.shape[1:]),
                fill_value,
                dtype=value.dtype,
                device=device,
            )
        with profiler.timer("gen.cpu_transfer_sparse_frontier"):
            sparse_value = value.to(device, non_blocking=non_blocking)
        with profiler.timer("gen.cpu_frontier_scatter_submit"):
            dense_value.flatten(0, 1).view(torch.uint8).index_copy_(
                0, dense_row_idx, sparse_value.view(torch.uint8)
            )
        return dense_value

    with profiler.timer("gen.cpu_transfer_fixed"):
        inventory = features.inventory.to(device, non_blocking=non_blocking)
        room_x = features.room_x.to(device, non_blocking=non_blocking)
        room_y = features.room_y.to(device, non_blocking=non_blocking)
        room_placed = features.room_placed.to(device, non_blocking=non_blocking)
        connection_reachability = features.connection_reachability.to(
            device, non_blocking=non_blocking
        )
    return StateFeatures(
        inventory,
        room_x,
        room_y,
        room_placed,
        densify(features.frontier),
        densify(features.frontier_occupancy),
        densify(features.frontier_neighbor, -1),
        densify(features.frontier_neighbor_pair),
        connection_reachability,
        densify(features.frontier_connection_reachability),
    )


class PinnedSparseStateFeatureSlot:
    def __init__(self, env: EnvironmentGroup, pin_memory: bool):
        state_features = env.engine.state_features
        inventory_count, max_frontier_count, room_count = env.engine.get_state_feature_sizes()
        _, connection_count = env.engine.get_output_sizes()
        self.inventory_width = inventory_count * int(state_features.get("inventory", False))
        self.room_width = room_count * int(state_features.get("room_position", False))
        self.frontier_count_capacity = max_frontier_count
        self.frontier_occupancy_width = (
            (env.frontier_window_size * env.frontier_window_size + 7) // 8
        ) * int(state_features.get("frontier_occupancy", False))
        self.frontier_neighbor_width = (
            env.frontier_neighbor_count * int(state_features.get("frontier_neighbor", False))
        )
        self.frontier_neighbor_pair_width = (
            env.frontier_neighbor_count
            * int(state_features.get("frontier_neighbor_flags", False))
        )
        self.connection_reachability_width = (
            connection_count * int(state_features.get("connection_reachability", False))
        )
        self.frontier_connection_reachability_width = (
            connection_count
            * int(state_features.get("frontier_connection_reachability", False))
        )
        self.pin_memory = pin_memory
        self.snapshot_capacity = 0
        self.sparse_row_capacity = 0
        self.inventory = None
        self.room_x = None
        self.room_y = None
        self.room_placed = None
        self.frontier = None
        self.frontier_occupancy = None
        self.frontier_neighbor = None
        self.frontier_neighbor_pair = None
        self.connection_reachability = None
        self.frontier_connection_reachability = None
        self.dense_row_idx = None

    def _empty(self, shape, dtype):
        return torch.empty(shape, dtype=dtype, pin_memory=self.pin_memory)

    def ensure(self, snapshot_count: int, sparse_row_count: int, profiler: ProfileStats):
        if (
            self.snapshot_capacity >= snapshot_count
            and self.sparse_row_capacity >= sparse_row_count
        ):
            return
        with profiler.timer("gen.cpu_pinned_feature_alloc"):
            self.snapshot_capacity = max(self.snapshot_capacity, snapshot_count)
            self.sparse_row_capacity = max(self.sparse_row_capacity, sparse_row_count)
            self.inventory = self._empty(
                (self.snapshot_capacity, self.inventory_width), torch.uint8
            )
            self.room_x = self._empty((self.snapshot_capacity, self.room_width), torch.int8)
            self.room_y = self._empty((self.snapshot_capacity, self.room_width), torch.int8)
            self.room_placed = self._empty(
                (self.snapshot_capacity, self.room_width), torch.uint8
            )
            self.frontier = self._empty(
                (self.sparse_row_capacity, 5), torch.int16
            )
            self.frontier_occupancy = self._empty(
                (self.sparse_row_capacity, self.frontier_occupancy_width), torch.uint8
            )
            self.frontier_neighbor = self._empty(
                (self.sparse_row_capacity, self.frontier_neighbor_width), torch.int16
            )
            self.frontier_neighbor_pair = self._empty(
                (self.sparse_row_capacity, self.frontier_neighbor_pair_width), torch.uint8
            )
            self.connection_reachability = self._empty(
                (self.snapshot_capacity, self.connection_reachability_width), torch.uint8
            )
            self.frontier_connection_reachability = self._empty(
                (
                    self.sparse_row_capacity,
                    self.frontier_connection_reachability_width,
                ),
                torch.uint8,
            )
            self.dense_row_idx = self._empty((self.sparse_row_capacity,), torch.int64)

    def features(
        self,
        environment_count: int,
        candidate_count: int,
        sparse_row_count: int,
        frontier_count: int,
    ) -> SparseStateFeatures:
        snapshot_count = environment_count * candidate_count
        return SparseStateFeatures(
            self.inventory[:snapshot_count].view(
                environment_count, candidate_count, self.inventory_width
            ),
            self.room_x[:snapshot_count].view(environment_count, candidate_count, self.room_width),
            self.room_y[:snapshot_count].view(environment_count, candidate_count, self.room_width),
            self.room_placed[:snapshot_count].view(
                environment_count, candidate_count, self.room_width
            ),
            self.frontier[:sparse_row_count],
            self.frontier_occupancy[:sparse_row_count],
            self.frontier_neighbor[:sparse_row_count],
            self.frontier_neighbor_pair[:sparse_row_count],
            self.connection_reachability[:snapshot_count].view(
                environment_count, candidate_count, self.connection_reachability_width
            ),
            self.frontier_connection_reachability[:sparse_row_count],
            self.dense_row_idx[:sparse_row_count],
            frontier_count,
        )

@dataclass
class StateFeatureGenerationCohort:
    env: EnvironmentGroup
    config: GenerateConfig
    known_outcomes: Outcomes | None = None
    step: int = 0
    candidates: Actions | None = None
    outcomes: Outcomes | None = None
    candidate_start: int = 0
    env_start: int = 0
    env_rewards: list[torch.Tensor] | None = None
    candidate_rewards: list[torch.Tensor] | None = None
    feature_slot: PinnedSparseStateFeatureSlot | None = None


def generate_state_feature_cohorts(
    envs: list[EnvironmentGroup],
    model,
    configs: list[GenerateConfig],
    device: torch.device,
    verify_outcome_consistency: bool = False,
    profiler: ProfileStats | None = None,
):
    profiler = profiler or ProfileStats(False)
    transfer_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    if not envs or len(envs) != len(configs):
        raise ValueError("generation cohorts require one config per environment group")
    num_rooms = len(envs[0].engine.rooms)
    cohorts = [
        StateFeatureGenerationCohort(env, config)
        for env, config in zip(envs, configs)
    ]
    executor = ThreadPoolExecutor(max_workers=len(cohorts))
    ready = deque()

    def finish_step(cohort):
        expected_reward = torch.cat(cohort.candidate_rewards, dim=1)
        dummy_candidate = cohort.candidates.room_idx == num_rooms
        expected_reward = torch.where(
            dummy_candidate,
            torch.full_like(expected_reward, float("-inf")),
            expected_reward,
        )
        with profiler.cuda_timer("gen.gpu_select", device):
            probs = torch.softmax(
                expected_reward / torch.unsqueeze(cohort.config.temperature, 1), dim=1
            )
            action_index = rand_choice(probs)
            selected_actions = cohort.candidates.select(action_index)
        if verify_outcome_consistency and cohort.config.lookahead_outcomes:
            cohort.known_outcomes = merge_verified_outcomes(
                cohort.known_outcomes,
                select_outcomes(cohort.outcomes, action_index),
                f"lookahead step {cohort.step}",
            )
        with profiler.timer("gen.cpu_step"):
            cohort.env.step(selected_actions)
        if verify_outcome_consistency:
            cohort.known_outcomes = merge_verified_outcomes(
                cohort.known_outcomes,
                cohort.env.get_outcomes(device),
                f"step {cohort.step}",
            )
        cohort.step += 1

    def submit_extract(cohort):
        if device.type == "cuda" and cohort.feature_slot is None:
            cohort.feature_slot = PinnedSparseStateFeatureSlot(cohort.env, pin_memory=True)
        candidate_end = cohort.candidate_start + cohort.config.state_candidate_chunk
        env_end = min(
            cohort.env_start + cohort.config.state_environment_chunk,
            cohort.env.num_envs,
        )
        chunk = Actions(
            cohort.candidates.room_idx[:, cohort.candidate_start:candidate_end],
            cohort.candidates.room_x[:, cohort.candidate_start:candidate_end],
            cohort.candidates.room_y[:, cohort.candidate_start:candidate_end],
        )
        ready.append((
            cohort,
            chunk,
            executor.submit(
                extract_candidate_features,
                cohort.env,
                chunk,
                cohort.env_start,
                env_end,
                profiler,
                device.type == "cuda",
                cohort.feature_slot,
            ),
        ))

    def start_step(cohort):
        while cohort.step < cohort.config.episode_length:
            if cohort.config.lookahead_outcomes:
                with profiler.timer("gen.cpu_candidates"):
                    cohort.candidates, cohort.outcomes = cohort.env.get_candidates_with_outcomes(
                        cohort.config.max_candidates, device
                    )
            else:
                with profiler.timer("gen.cpu_candidates"):
                    cohort.candidates = cohort.env.get_candidates(cohort.config.max_candidates, device)
                    cohort.outcomes = cohort.env.get_outcomes(device)
            if cohort.candidates.room_idx.shape[1] != 1:
                cohort.candidate_start = 0
                cohort.env_start = 0
                cohort.env_rewards = []
                cohort.candidate_rewards = []
                submit_extract(cohort)
                return
            action_index = torch.zeros(
                cohort.candidates.room_idx.shape[0], dtype=torch.int64, device=device
            )
            if verify_outcome_consistency and cohort.config.lookahead_outcomes:
                cohort.known_outcomes = merge_verified_outcomes(
                    cohort.known_outcomes,
                    select_outcomes(cohort.outcomes, action_index),
                    f"lookahead step {cohort.step}",
                )
            with profiler.timer("gen.cpu_step"):
                cohort.env.step(cohort.candidates.select(action_index))
            if verify_outcome_consistency:
                cohort.known_outcomes = merge_verified_outcomes(
                    cohort.known_outcomes,
                    cohort.env.get_outcomes(device),
                    f"step {cohort.step}",
                )
            cohort.step += 1

    try:
        with torch.no_grad():
            for cohort in cohorts:
                cohort.env.clear()
                start_step(cohort)
            while ready:
                cohort, chunk, future = ready.popleft()
                with profiler.timer("gen.cpu_extract_wait"):
                    env_start, env_end, features = future.result()
                with profiler.timer("gen.cpu_transfer_submit"):
                    env_features = transfer_state_features(
                        features, device, profiler, transfer_stream
                    )
                candidate_count = chunk.room_idx.shape[1]
                chunk_outcomes = Outcomes(
                    cohort.outcomes.door_invalid[env_start:env_end, cohort.candidate_start:cohort.candidate_start + candidate_count]
                    if cohort.outcomes.door_invalid.ndim == 3 else cohort.outcomes.door_invalid[env_start:env_end],
                    cohort.outcomes.connection_invalid[env_start:env_end, cohort.candidate_start:cohort.candidate_start + candidate_count]
                    if cohort.outcomes.connection_invalid.ndim == 3 else cohort.outcomes.connection_invalid[env_start:env_end],
                )
                with profiler.cuda_timer("gen.gpu_model_reward", device):
                    with torch.amp.autocast(
                        "cuda",
                        dtype=torch.bfloat16,
                        enabled=device.type == "cuda" and cohort.config.state_autocast,
                    ):
                        chunk_preds = model(env_features)
                    cohort.env_rewards.append(compute_expected_reward(
                        Predictions(
                            chunk_preds.door_invalid.view(env_end - env_start, candidate_count, -1),
                            chunk_preds.connection_invalid.view(env_end - env_start, candidate_count, -1),
                        ),
                        chunk_outcomes,
                        cohort.config,
                    ))
                cohort.env_start = env_end
                if cohort.env_start < cohort.env.num_envs:
                    submit_extract(cohort)
                    continue
                cohort.candidate_rewards.append(torch.cat(cohort.env_rewards, dim=0))
                cohort.candidate_start += candidate_count
                if cohort.candidate_start < cohort.candidates.room_idx.shape[1]:
                    cohort.env_start = 0
                    cohort.env_rewards = []
                    submit_extract(cohort)
                    continue
                finish_step(cohort)
                start_step(cohort)
    finally:
        executor.shutdown()

    results = []
    for cohort in cohorts:
        with profiler.timer("gen.cpu_finish"):
            cohort.env.finish()
            actions = cohort.env.get_actions(device)
            outcomes = cohort.env.get_outcomes(device)
        if verify_outcome_consistency:
            merge_verified_outcomes(cohort.known_outcomes, outcomes, "finish")
        results.append((actions, outcomes))
    return (
        Actions(*(torch.cat([getattr(actions, name) for actions, _ in results]) for name in vars(results[0][0]))),
        Outcomes(*(torch.cat([getattr(outcomes, name) for _, outcomes in results]) for name in vars(results[0][1]))),
    )


def generate_cohorts(
    envs: list[EnvironmentGroup],
    model,
    configs: list[GenerateConfig],
    device: torch.device,
    verify_outcome_consistency: bool = False,
    profiler: ProfileStats | None = None,
):
    if not envs or len(envs) != len(configs):
        raise ValueError("generation cohorts require one config per environment group")
    if len(envs) == 1:
        return generate(
            envs[0],
            model,
            configs[0],
            device,
            verify_outcome_consistency=verify_outcome_consistency,
            profiler=profiler,
        )
    if not getattr(model, "uses_state_features", False):
        raise ValueError("multiple generation cohorts require a state-feature model")
    return generate_state_feature_cohorts(
        envs,
        model,
        configs,
        device,
        verify_outcome_consistency=verify_outcome_consistency,
        profiler=profiler,
    )


def generate(
    env: EnvironmentGroup,
    model,
    config: GenerateConfig,
    device: torch.device,
    verify_outcome_consistency: bool = False,
    profiler: ProfileStats | None = None,
):
    num_envs = env.num_envs
    engine = env.engine
    num_rooms = len(engine.rooms)

    uses_state_features = getattr(model, "uses_state_features", False)
    kv_cache = None if uses_state_features else model.get_initial_kv_cache(num_envs, device)
    env.clear()
    known_outcomes = None
    profiler = profiler or ProfileStats(False)
    transfer_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    prefetcher = Prefetcher()
    feature_slots = (
        [PinnedSparseStateFeatureSlot(env, pin_memory=True) for _ in range(2)]
        if uses_state_features and device.type == "cuda"
        else None
    )
    try:
        with torch.no_grad():
            for step in range(config.episode_length):
                if config.lookahead_outcomes:
                    # Get candidate actions and their post-step known outcomes from environment.
                    with profiler.timer("gen.cpu_candidates"):
                        candidates, outcomes = env.get_candidates_with_outcomes(config.max_candidates, device)
                else:
                    # Use current known outcomes for all candidates.
                    with profiler.timer("gen.cpu_candidates"):
                        candidates = env.get_candidates(config.max_candidates, device)
                        outcomes = env.get_outcomes(device)

                if candidates.room_idx.shape[1] == 1:
                    # Only one candidate, so select it directly (e.g. on the first step)
                    if not uses_state_features:
                        _, kv_cache_candidates = model.generate(candidates, kv_cache, config)
                    action_index = torch.zeros(candidates.room_idx.shape[0], dtype=torch.int64, device=device)
                    selected_actions = candidates.select(action_index)
                else:
                    if uses_state_features:
                        candidate_rewards = []
                        for start in range(0, candidates.room_idx.shape[1], config.state_candidate_chunk):
                            end = start + config.state_candidate_chunk
                            chunk = Actions(
                                candidates.room_idx[:, start:end],
                                candidates.room_x[:, start:end],
                                candidates.room_y[:, start:end],
                            )
                            env_rewards = []
                            env_starts = range(0, num_envs, config.state_environment_chunk)

                            def prepare_features(env_start):
                                env_end = min(env_start + config.state_environment_chunk, num_envs)
                                feature_slot = None
                                if feature_slots is not None:
                                    slot_idx = (
                                        env_start // config.state_environment_chunk
                                    ) % len(feature_slots)
                                    feature_slot = feature_slots[slot_idx]
                                return extract_candidate_features(
                                    env,
                                    chunk,
                                    env_start,
                                    env_end,
                                    profiler,
                                    device.type == "cuda",
                                    feature_slot,
                                )

                            for env_start, env_end, features in prefetcher.map(
                                env_starts, prepare_features, profiler, "gen.cpu_extract_wait"
                            ):
                                with profiler.timer("gen.cpu_transfer_submit"):
                                    env_features = transfer_state_features(
                                        features, device, profiler, transfer_stream
                                    )
                                candidate_count = chunk.room_idx.shape[1]
                                chunk_outcomes = Outcomes(
                                    outcomes.door_invalid[env_start:env_end, start:end]
                                    if outcomes.door_invalid.ndim == 3 else outcomes.door_invalid[env_start:env_end],
                                    outcomes.connection_invalid[env_start:env_end, start:end]
                                    if outcomes.connection_invalid.ndim == 3 else outcomes.connection_invalid[env_start:env_end],
                                )
                                with profiler.cuda_timer("gen.gpu_model_reward", device):
                                    with torch.amp.autocast(
                                        "cuda",
                                        dtype=torch.bfloat16,
                                        enabled=device.type == "cuda" and config.state_autocast,
                                    ):
                                        chunk_preds = model(env_features)
                                    env_rewards.append(compute_expected_reward(
                                        Predictions(
                                            chunk_preds.door_invalid.view(env_end - env_start, candidate_count, -1),
                                            chunk_preds.connection_invalid.view(env_end - env_start, candidate_count, -1),
                                        ),
                                        chunk_outcomes,
                                        config,
                                    ))
                            candidate_rewards.append(torch.cat(env_rewards, dim=0))
                        expected_reward = torch.cat(candidate_rewards, dim=1)
                        kv_cache_candidates = None
                    else:
                        # Model inference to get predictions and updated key-value cache for next step
                        preds, kv_cache_candidates = model.generate(candidates, kv_cache, config)
                        expected_reward = compute_expected_reward(preds, outcomes, config)
                    # Compute expected reward and sample to select an action (per environment)
                    dummy_candidate = candidates.room_idx == num_rooms
                    expected_reward = torch.where(
                        dummy_candidate,
                        torch.full_like(expected_reward, float('-inf')),
                        expected_reward,
                    )
                    with profiler.cuda_timer("gen.gpu_select", device):
                        probs = torch.softmax(expected_reward / torch.unsqueeze(config.temperature, 1), dim=1)
                        action_index = rand_choice(probs)
                        selected_actions = candidates.select(action_index)

                if verify_outcome_consistency and config.lookahead_outcomes:
                    known_outcomes = merge_verified_outcomes(
                        known_outcomes,
                        select_outcomes(outcomes, action_index),
                        f"lookahead step {step}",
                    )

                # Apply the selected action to the environment
                with profiler.timer("gen.cpu_step"):
                    env.step(selected_actions)

                if verify_outcome_consistency:
                    known_outcomes = merge_verified_outcomes(
                        known_outcomes,
                        env.get_outcomes(device),
                        f"step {step}",
                    )

                # Finalize the kv cache update based on the selected action
                if not uses_state_features and kv_cache_candidates is not None:
                    kv_cache = model.get_updated_kv_cache(kv_cache, kv_cache_candidates, action_index)
    finally:
        prefetcher.close()
        
    with profiler.timer("gen.cpu_finish"):
        env.finish()
        actions = env.get_actions(device)
        outcomes = env.get_outcomes(device)
    if verify_outcome_consistency:
        merge_verified_outcomes(known_outcomes, outcomes, "finish")
    return actions, outcomes
