from __future__ import annotations

from env import (
    Actions,
    DoorMatchCounts,
    Engine,
    EnvironmentGroup,
    GenerateConfig,
    Outcomes,
    SparseStateFeatures,
    StateFeatures,
)
from model import Predictions
from profile_stats import ProfileStats
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import logging
import threading
import torch

from train_config import Config


KNOWN_INVALID_REWARD = -100.0


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

    return Outcomes(
        door_invalid=merge_known_outcome(
            known_outcomes.door_invalid, current_outcomes.door_invalid, "door", stage
        ),
        connection_invalid=merge_known_outcome(
            known_outcomes.connection_invalid,
            current_outcomes.connection_invalid,
            "connection",
            stage,
        ),
    )


def merge_known_outcome(
    known: torch.Tensor,
    current: torch.Tensor,
    outcome_name: str,
    stage: str,
) -> torch.Tensor:
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


def extract_candidate_features(
    env: EnvironmentGroup,
    candidates: Actions,
    profiler: ProfileStats,
    sparse_frontiers: bool = False,
    feature_slot: PinnedSparseStateFeatureSlot | None = None,
):
    with profiler.timer("gen.cpu_extract"):
        if sparse_frontiers and feature_slot is not None:
            frontier_count, sparse_row_count, worker_sparse_row_counts = (
                env.env.get_sparse_state_feature_requirements_after_candidates(
                    candidates.room_idx.contiguous().cpu().numpy(),
                    candidates.room_x.contiguous().cpu().numpy(),
                    candidates.room_y.contiguous().cpu().numpy(),
                    0,
                )
            )
            feature_slot.ensure(
                candidates.room_idx.numel(),
                sparse_row_count,
                profiler,
            )
            env.env.get_sparse_state_features_after_candidates_into(
                candidates.room_idx.contiguous().cpu().numpy(),
                candidates.room_x.contiguous().cpu().numpy(),
                candidates.room_y.contiguous().cpu().numpy(),
                0,
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
                candidates.room_idx.shape[0],
                candidates.room_idx.shape[1],
                sparse_row_count,
                frontier_count,
            ).flatten_candidates()
        elif sparse_frontiers:
            features = env.get_sparse_state_features_after_candidates(
                candidates, torch.device("cpu"), 0
            ).flatten_candidates()
        else:
            features = env.get_state_features_after_candidates(
                candidates, torch.device("cpu"), 0
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
    return features


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
        densify_sparse_feature(
            features.frontier, 0, dense_shape, dense_row_idx, device, profiler, non_blocking
        ),
        densify_sparse_feature(
            features.frontier_occupancy, 0, dense_shape, dense_row_idx, device, profiler, non_blocking
        ),
        densify_sparse_feature(
            features.frontier_neighbor, -1, dense_shape, dense_row_idx, device, profiler, non_blocking
        ),
        densify_sparse_feature(
            features.frontier_neighbor_pair, 0, dense_shape, dense_row_idx, device, profiler, non_blocking
        ),
        connection_reachability,
        densify_sparse_feature(
            features.frontier_connection_reachability,
            0,
            dense_shape,
            dense_row_idx,
            device,
            profiler,
            non_blocking,
        ),
    )


def densify_sparse_feature(
    value: torch.Tensor,
    fill_value: int,
    dense_shape: tuple[int, int],
    dense_row_idx: torch.Tensor,
    device: torch.device,
    profiler: ProfileStats,
    non_blocking: bool,
) -> torch.Tensor:
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


class PinnedSparseStateFeatureSlot:
    def __init__(self, env: EnvironmentGroup, pin_memory: bool):
        state_features = env.engine.state_features
        inventory_count, max_frontier_count, room_count = env.engine.get_state_feature_sizes()
        _, connection_count = env.engine.get_output_sizes()
        self.inventory_width = inventory_count * int(state_features.inventory)
        self.room_width = room_count * int(state_features.room_position)
        self.frontier_occupancy_width = (
            (env.frontier_window_size * env.frontier_window_size + 7) // 8
        ) * int(state_features.frontier_occupancy)
        self.frontier_neighbor_width = (
            env.frontier_neighbor_count * int(state_features.frontier_neighbor)
        )
        self.frontier_neighbor_pair_width = (
            env.frontier_neighbor_count
            * int(state_features.frontier_neighbor_flags)
        )
        self.connection_reachability_width = (
            connection_count * int(state_features.connection_reachability)
        )
        self.frontier_connection_reachability_width = (
            connection_count
            * int(state_features.frontier_connection_reachability)
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
                (self.sparse_row_capacity, 5), torch.int8
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
class GenerationCohort:
    env: EnvironmentGroup
    config: GenerateConfig
    known_outcomes: Outcomes | None
    step: int
    feature_slot: PinnedSparseStateFeatureSlot | None


@dataclass
class PendingGenerationStep:
    cohort: GenerationCohort
    candidates: Actions
    outcomes: Outcomes
    future: Future[StateFeatures | SparseStateFeatures]


def create_generation_environment_groups(
    config: Config,
    engine: Engine,
    generation_devices: list[torch.device],
) -> list[list[EnvironmentGroup]]:
    num_generation_cohorts = (
        config.generation.num_devices * config.generation.state_pipeline_cohorts
    )
    generation_cohort_environments = config.generation.num_environments // num_generation_cohorts
    generation_cohort_threads = (
        None
        if config.generation.num_threads is None
        else config.generation.num_threads // config.generation.state_pipeline_cohorts
    )
    logging.info(
        "Using %s state pipeline cohort(s) per generation device with %s environment(s) and %s Rust worker(s) per cohort.",
        config.generation.state_pipeline_cohorts,
        generation_cohort_environments,
        generation_cohort_threads if generation_cohort_threads is not None else "automatic",
    )
    return [
        [
            engine.create_environment_group(
                config.map_size,
                generation_cohort_environments,
                seed=device_index * config.generation.state_pipeline_cohorts + cohort_index,
                frontier_neighbor_algorithm=config.generation.frontier_neighbor_algorithm,
                frontier_neighbor_count=config.generation.frontier_neighbor_count,
                frontier_window_size=config.generation.frontier_window_size,
                num_threads=generation_cohort_threads,
            )
            for cohort_index in range(config.generation.state_pipeline_cohorts)
        ]
        for device_index in range(len(generation_devices))
    ]


def get_generation_candidates(
    cohort: GenerationCohort,
    device: torch.device,
    profiler: ProfileStats,
) -> tuple[Actions, Outcomes]:
    with profiler.timer("gen.cpu_candidates"):
        if cohort.config.lookahead_outcomes:
            return cohort.env.get_candidates_with_outcomes(
                cohort.config.max_candidates, device
            )
        candidates = cohort.env.get_candidates(cohort.config.max_candidates, device)
        return candidates, cohort.env.get_outcomes(device)


def select_candidate_actions(
    cohort: GenerationCohort,
    model,
    candidates: Actions,
    outcomes: Outcomes,
    features: StateFeatures | SparseStateFeatures,
    device: torch.device,
    gpu_lock: threading.Lock,
    transfer_stream: torch.cuda.Stream | None,
    profiler: ProfileStats,
    num_rooms: int,
) -> tuple[torch.Tensor, Actions]:
    environment_count, candidate_count = candidates.room_idx.shape
    with gpu_lock:
        with profiler.timer("gen.cpu_transfer_submit"):
            env_features = transfer_state_features(features, device, profiler, transfer_stream)
        with profiler.cuda_timer("gen.gpu_model_reward", device):
            with torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and cohort.config.state_autocast,
            ):
                preds = model(env_features)
            expected_reward = compute_expected_reward(
                Predictions(
                    preds.door_invalid.view(environment_count, candidate_count, -1),
                    preds.connection_invalid.view(environment_count, candidate_count, -1),
                ),
                outcomes,
                cohort.config,
            )
            expected_reward = torch.where(
                candidates.room_idx == num_rooms,
                torch.full_like(expected_reward, float("-inf")),
                expected_reward,
            )
        with profiler.cuda_timer("gen.gpu_select", device):
            probs = torch.softmax(
                expected_reward / torch.unsqueeze(cohort.config.temperature, 1),
                dim=1,
            )
            action_index = rand_choice(probs)
            selected_actions = candidates.select(action_index)
    return action_index, selected_actions


def verify_and_step(
    cohort: GenerationCohort,
    selected_actions: Actions,
    action_index: torch.Tensor,
    outcomes: Outcomes,
    step: int,
    device: torch.device,
    verify_outcome_consistency: bool,
    profiler: ProfileStats,
) -> None:
    if verify_outcome_consistency and cohort.config.lookahead_outcomes:
        cohort.known_outcomes = merge_verified_outcomes(
            cohort.known_outcomes,
            select_outcomes(outcomes, action_index),
            f"lookahead step {step}",
        )
    with profiler.timer("gen.cpu_step"):
        cohort.env.step(selected_actions)
    if verify_outcome_consistency:
        cohort.known_outcomes = merge_verified_outcomes(
            cohort.known_outcomes,
            cohort.env.get_outcomes(device),
            f"step {step}",
        )


def submit_generation_step(
    cohort: GenerationCohort,
    candidates: Actions,
    outcomes: Outcomes,
    sparse_frontiers: bool,
    profiler: ProfileStats,
    executor: ThreadPoolExecutor,
    pending: deque[PendingGenerationStep],
) -> None:
    pending.append(
        PendingGenerationStep(
            cohort,
            candidates,
            outcomes,
            executor.submit(
                extract_candidate_features,
                cohort.env,
                candidates,
                profiler,
                sparse_frontiers,
                cohort.feature_slot,
            ),
        )
    )


def process_single_candidate_step(
    cohort: GenerationCohort,
    device: torch.device,
    verify_outcome_consistency: bool,
    profiler: ProfileStats,
    candidates: Actions,
    outcomes: Outcomes,
) -> None:
    action_index = torch.zeros(
        candidates.room_idx.shape[0],
        dtype=torch.int64,
        device=device,
    )
    selected_actions = candidates.select(action_index)

    verify_and_step(
        cohort,
        selected_actions,
        action_index,
        outcomes,
        cohort.step,
        device,
        verify_outcome_consistency,
        profiler,
    )
    cohort.step += 1


def process_scored_candidate_step(
    cohort: GenerationCohort,
    candidates: Actions,
    outcomes: Outcomes,
    features: StateFeatures | SparseStateFeatures,
    model,
    device: torch.device,
    gpu_lock: threading.Lock,
    transfer_stream: torch.cuda.Stream | None,
    verify_outcome_consistency: bool,
    profiler: ProfileStats,
    num_rooms: int,
) -> None:
    action_index, selected_actions = select_candidate_actions(
        cohort,
        model,
        candidates,
        outcomes,
        features,
        device,
        gpu_lock,
        transfer_stream,
        profiler,
        num_rooms,
    )
    verify_and_step(
        cohort,
        selected_actions,
        action_index,
        outcomes,
        cohort.step,
        device,
        verify_outcome_consistency,
        profiler,
    )
    cohort.step += 1


def start_generation_step(
    cohort: GenerationCohort,
    device: torch.device,
    sparse_frontiers: bool,
    verify_outcome_consistency: bool,
    profiler: ProfileStats,
    executor: ThreadPoolExecutor,
    pending: deque[PendingGenerationStep],
) -> None:
    while cohort.step < cohort.config.episode_length:
        candidates, outcomes = get_generation_candidates(cohort, device, profiler)
        if candidates.room_idx.shape[1] != 1:
            submit_generation_step(
                cohort,
                candidates,
                outcomes,
                sparse_frontiers,
                profiler,
                executor,
                pending,
            )
            return
        process_single_candidate_step(
            cohort,
            device,
            verify_outcome_consistency,
            profiler,
            candidates,
            outcomes,
        )


def finish_generation_cohort(
    cohort: GenerationCohort,
    device: torch.device,
    verify_outcome_consistency: bool,
    profiler: ProfileStats,
) -> tuple[Actions, Outcomes, DoorMatchCounts]:
    with profiler.timer("gen.cpu_finish"):
        cohort.env.finish()
        actions = cohort.env.get_actions(device)
        outcomes = cohort.env.get_outcomes(device)
        door_match_counts = cohort.env.get_door_match_counts(device)
    if verify_outcome_consistency:
        merge_verified_outcomes(cohort.known_outcomes, outcomes, "finish")
    return actions, outcomes, door_match_counts


def merge_generation_results(
    results: list[tuple[Actions, Outcomes, DoorMatchCounts]],
) -> tuple[Actions, Outcomes, DoorMatchCounts]:
    return (
        Actions(
            *(
                torch.cat([getattr(actions, name) for actions, _, _ in results])
                for name in vars(results[0][0])
            )
        ),
        Outcomes(
            *(
                torch.cat([getattr(outcomes, name) for _, outcomes, _ in results])
                for name in vars(results[0][1])
            )
        ),
        DoorMatchCounts(
            *(
                torch.sum(
                    torch.stack([getattr(counts, name) for _, _, counts in results]),
                    dim=0,
                )
                for name in vars(results[0][2])
            )
        ),
    )


def generate_cohorts(
    envs: list[EnvironmentGroup],
    model,
    configs: list[GenerateConfig],
    device: torch.device,
    verify_outcome_consistency: bool = False,
    profiler: ProfileStats | None = None,
) -> tuple[Actions, Outcomes, DoorMatchCounts]:
    if not envs or len(envs) != len(configs):
        raise ValueError("generation cohorts require one config per environment group")
    profiler = profiler or ProfileStats(False)
    transfer_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    gpu_lock = threading.Lock()
    num_rooms = len(envs[0].engine.rooms)
    sparse_frontiers = device.type == "cuda"
    cohorts = [
        GenerationCohort(
            env,
            config,
            None,
            0,
            PinnedSparseStateFeatureSlot(env, pin_memory=True)
            if device.type == "cuda"
            else None,
        )
        for env, config in zip(envs, configs)
    ]
    with ThreadPoolExecutor(max_workers=len(cohorts)) as executor:
        pending: deque[PendingGenerationStep] = deque()
        with torch.no_grad():
            for cohort in cohorts:
                cohort.env.clear()
                start_generation_step(
                    cohort,
                    device,
                    sparse_frontiers,
                    verify_outcome_consistency,
                    profiler,
                    executor,
                    pending,
                )
            while pending:
                step = pending.popleft()
                with profiler.timer("gen.cpu_extract_wait"):
                    features = step.future.result()
                process_scored_candidate_step(
                    step.cohort,
                    step.candidates,
                    step.outcomes,
                    features,
                    model,
                    device,
                    gpu_lock,
                    transfer_stream,
                    verify_outcome_consistency,
                    profiler,
                    num_rooms,
                )
                if step.cohort.step < step.cohort.config.episode_length:
                    start_generation_step(
                        step.cohort,
                        device,
                        sparse_frontiers,
                        verify_outcome_consistency,
                        profiler,
                        executor,
                        pending,
                    )
        results = [
            finish_generation_cohort(
                cohort,
                device,
                verify_outcome_consistency,
                profiler,
            )
            for cohort in cohorts
        ]
    return merge_generation_results(results)
