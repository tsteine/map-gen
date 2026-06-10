from __future__ import annotations

from env import (
    Actions,
    DoorMatchCounts,
    Engine,
    EnvironmentGroup,
    EpisodeData,
    EpisodeOutcomes,
    GenerateConfig,
    PreliminaryOutcomes,
    ProposalData,
    SparseFeatureRequirements,
    SparseFeatures,
    Features,
)
from model import Predictions
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import logging
import math
import threading
import time
import torch

from train_config import Config

type ProfileReport = list[tuple[str, int, int]]


# We make use of a somewhat complicated way of pipelining the generation process,
# to keep the GPU busy while CPU extraction is ongoing. This pays off heavily
# because we are using a relatively small model on the GPU.
#
# Generation runs several environment groups (typically two) per generation device.
# The coordinator thread reads candidates/outcomes, submits only CPU feature
# extraction to a small executor, then consumes completed extractions in FIFO
# order. For each completed group step it transfers and scores candidates on the
# generation device, steps that group's environment, and immediately starts the
# next step for that group. This keeps expensive CPU extraction overlapped
# across groups while avoiding CUDA work from multiple Python worker threads.

def rand_choice(p):
    cumul_p = torch.cumsum(p, dim=1)
    rnd = torch.rand([p.shape[0], 1], device=p.device)
    choice = torch.clamp(torch.searchsorted(cumul_p, rnd), max=p.shape[1] - 1).view(-1)
    return choice


class GenerationProfiler:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.counts: dict[str, int] = {}
        self.nanos: dict[str, int] = {}

    def add(self, name: str, start: int) -> None:
        if not self.enabled:
            return
        self.counts[name] = self.counts.get(name, 0) + 1
        self.nanos[name] = self.nanos.get(name, 0) + time.perf_counter_ns() - start

    def report(self) -> ProfileReport:
        return [
            (name, self.counts[name], self.nanos[name])
            for name in sorted(self.counts)
        ]


def profile_start(enabled: bool) -> int:
    return time.perf_counter_ns() if enabled else 0


def sync_profile_device(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()


@dataclass
class CachedProposalScores:
    tensor: torch.Tensor
    ready_event: torch.cuda.Event | None

    def resolve(self) -> torch.Tensor:
        if self.ready_event is not None:
            self.ready_event.synchronize()
            self.ready_event = None
        return self.tensor


def cache_proposal_scores_tensor(
    scores: torch.Tensor,
    copy_stream: torch.cuda.Stream | None,
) -> CachedProposalScores:
    scores = scores.to(torch.float32)
    if scores.device.type != "cuda":
        return CachedProposalScores(scores.to(torch.device("cpu")), None)
    if copy_stream is None:
        raise ValueError("CUDA proposal score caching requires a copy stream")
    cpu_scores = torch.empty(
        scores.shape,
        dtype=scores.dtype,
        device=torch.device("cpu"),
        pin_memory=True,
    )
    copy_stream.wait_stream(torch.cuda.current_stream(scores.device))
    with torch.cuda.device(scores.device), torch.cuda.stream(copy_stream):
        cpu_scores.copy_(scores, non_blocking=True)
        ready_event = torch.cuda.Event()
        ready_event.record(copy_stream)
    return CachedProposalScores(cpu_scores, ready_event)


def outcome_reward(model_logprobs: torch.Tensor, known_invalid: torch.Tensor) -> torch.Tensor:
    if known_invalid.ndim == model_logprobs.ndim - 1:
        known_invalid = known_invalid.unsqueeze(1)
    known_reward = torch.zeros_like(model_logprobs)
    return torch.where(known_invalid < 0, model_logprobs, known_reward)


def balance_reward(
    balance_score: torch.Tensor,
    door_invalid: torch.Tensor,
    known_invalid: torch.Tensor,
) -> torch.Tensor:
    if known_invalid.ndim == balance_score.ndim - 1:
        known_invalid = known_invalid.unsqueeze(1)
    match_probability = torch.sigmoid(-door_invalid)
    known_match_probability = torch.where(
        known_invalid == 0,
        torch.ones_like(match_probability),
        torch.zeros_like(match_probability),
    )
    match_probability = torch.where(
        known_invalid < 0,
        match_probability,
        known_match_probability,
    )
    model_reward = -balance_score * match_probability
    known_reward = torch.zeros_like(model_reward)
    return torch.where(known_invalid == 0, known_reward, model_reward)


# preds.door_invalid: [batch_size, max_candidates, num_outputs]
# preds.connection_invalid: [batch_size, max_candidates, num_outputs]
def compute_expected_reward(
    preds,
    outcomes,
    config: GenerateConfig,
):
    door_logprobs = torch.nn.functional.logsigmoid(-preds.door_invalid)
    connection_logprobs = torch.nn.functional.logsigmoid(-preds.connection_invalid)
    door_logprobs = outcome_reward(door_logprobs, outcomes.door_invalid)
    connection_logprobs = outcome_reward(connection_logprobs, outcomes.connection_invalid)
    balance_scores = balance_reward(
        preds.balance_score,
        preds.door_invalid,
        outcomes.door_invalid,
    )
    return (
        config.reward_door * torch.sum(door_logprobs, dim=2)
        + config.reward_connection * torch.sum(connection_logprobs, dim=2)
        + config.reward_balance * torch.sum(balance_scores, dim=2)
        - config.reward_frontier * preds.avg_frontiers.to(torch.float32)
    )


def extract_candidate_features(
    env: EnvironmentGroup,
    candidates: Actions,
    log_temperature: torch.Tensor,
    include_temperature: bool,
    log_recommended_candidates: torch.Tensor,
    include_recommended_candidates: bool,
    log_exploration_candidates: torch.Tensor,
    include_exploration_candidates: bool,
    lookahead_outcomes: PreliminaryOutcomes,
    include_lookahead_outcomes: bool,
    sparse_feature_requirements: SparseFeatureRequirements,
    sparse_frontiers: bool = False,
    feature_slot: PinnedSparseFeatureSlot | None = None,
):
    if sparse_frontiers and feature_slot is not None:
        frontier_count = sparse_feature_requirements.frontier_count
        sparse_row_count = sparse_feature_requirements.sparse_row_count
        worker_sparse_row_counts = sparse_feature_requirements.worker_sparse_row_counts
        feature_slot.ensure(
            candidates.room_idx.numel(),
            sparse_row_count,
        )
        env.env.pack_sparse_features_after_candidates_into(
            candidates.room_idx.shape[0],
            candidates.room_idx.shape[1],
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
        return feature_slot.features(
            candidates.room_idx.shape[0],
            candidates.room_idx.shape[1],
            log_temperature,
            include_temperature,
            log_recommended_candidates,
            include_recommended_candidates,
            log_exploration_candidates,
            include_exploration_candidates,
            lookahead_outcomes,
            include_lookahead_outcomes,
            sparse_row_count,
            frontier_count,
        ).flatten_candidates()
    if sparse_frontiers:
        return env.get_sparse_features_after_candidates(
            candidates,
            torch.device("cpu"),
            log_temperature,
            include_temperature,
            log_recommended_candidates,
            include_recommended_candidates,
            log_exploration_candidates,
            include_exploration_candidates,
            lookahead_outcomes,
            include_lookahead_outcomes,
            0,
        ).flatten_candidates()
    return env.get_features_after_candidates(
        candidates,
        torch.device("cpu"),
        log_temperature,
        include_temperature,
        log_recommended_candidates,
        include_recommended_candidates,
        log_exploration_candidates,
        include_exploration_candidates,
        lookahead_outcomes,
        include_lookahead_outcomes,
        0,
    ).flatten_candidates()


def transfer_features(
    features: Features | SparseFeatures,
    device: torch.device,
    transfer_stream: torch.cuda.Stream | None = None,
) -> Features:
    if isinstance(features, SparseFeatures):
        if transfer_stream is None or device.type != "cuda":
            return transfer_features_sync(features, device)
        current_stream = torch.cuda.current_stream(device)
        with torch.cuda.device(device), torch.cuda.stream(transfer_stream):
            result = transfer_features_sync(features, device, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(transfer_stream)
        current_stream.wait_event(ready)
        return result
    return features.to(device)


def transfer_features_sync(
    features: SparseFeatures,
    device: torch.device,
    non_blocking: bool = False,
) -> Features:
    dense_shape = (features.inventory.shape[0], features.frontier_count)
    dense_row_idx = features.dense_row_idx.to(device, non_blocking=non_blocking)
    inventory = features.inventory.to(device, non_blocking=non_blocking)
    room_x = features.room_x.to(device, non_blocking=non_blocking)
    room_y = features.room_y.to(device, non_blocking=non_blocking)
    room_placed = features.room_placed.to(device, non_blocking=non_blocking)
    log_temperature = features.log_temperature.to(device, non_blocking=non_blocking)
    log_recommended_candidates = features.log_recommended_candidates.to(
        device, non_blocking=non_blocking
    )
    log_exploration_candidates = features.log_exploration_candidates.to(
        device, non_blocking=non_blocking
    )
    lookahead_door_invalid = features.lookahead_door_invalid.to(
        device, non_blocking=non_blocking
    )
    lookahead_door_match = features.lookahead_door_match.to(
        device, non_blocking=non_blocking
    )
    lookahead_connection_invalid = features.lookahead_connection_invalid.to(
        device, non_blocking=non_blocking
    )
    connection_reachability = features.connection_reachability.to(
        device, non_blocking=non_blocking
    )
    return Features(
        inventory,
        room_x,
        room_y,
        room_placed,
        log_temperature,
        log_recommended_candidates,
        log_exploration_candidates,
        lookahead_door_invalid,
        lookahead_door_match,
        lookahead_connection_invalid,
        densify_sparse_feature(
            features.frontier, 0, dense_shape, dense_row_idx, device, non_blocking
        ),
        densify_sparse_feature(
            features.frontier_occupancy, 0, dense_shape, dense_row_idx, device, non_blocking
        ),
        densify_sparse_feature(
            features.frontier_neighbor, -1, dense_shape, dense_row_idx, device, non_blocking
        ),
        densify_sparse_feature(
            features.frontier_neighbor_pair, 0, dense_shape, dense_row_idx, device, non_blocking
        ),
        connection_reachability,
        densify_sparse_feature(
            features.frontier_connection_reachability,
            0,
            dense_shape,
            dense_row_idx,
            device,
            non_blocking,
        ),
    )


def densify_sparse_feature(
    value: torch.Tensor,
    fill_value: int,
    dense_shape: tuple[int, int],
    dense_row_idx: torch.Tensor,
    device: torch.device,
    non_blocking: bool,
) -> torch.Tensor:
    dense_value = torch.full(
        (*dense_shape, *value.shape[1:]),
        fill_value,
        dtype=value.dtype,
        device=device,
    )
    sparse_value = value.to(device, non_blocking=non_blocking)
    dense_value.flatten(0, 1).view(torch.uint8).index_copy_(
        0, dense_row_idx, sparse_value.view(torch.uint8)
    )
    return dense_value


# When a GPU is available, we use pinned memory for model input tensors,
# to allow for asynchronous CPU-to-GPU transfers.
class PinnedSparseFeatureSlot:
    def __init__(self, env: EnvironmentGroup, pin_memory: bool):
        features = env.engine.features
        inventory_count, max_frontier_count, room_count = env.engine.get_feature_sizes()
        _, connection_count = env.engine.get_output_sizes()
        self.inventory_width = inventory_count * int(features.inventory)
        self.room_width = room_count * int(features.room_position)
        self.frontier_occupancy_width = (
            (env.frontier_window_size * env.frontier_window_size + 7) // 8
        ) * int(features.frontier_occupancy)
        self.frontier_neighbor_width = (
            env.frontier_neighbor_count * int(features.frontier_neighbor)
        )
        self.frontier_neighbor_pair_width = (
            env.frontier_neighbor_count
            * int(features.frontier_neighbor_flags)
        )
        self.connection_reachability_width = (
            connection_count * int(features.connection_reachability)
        )
        self.frontier_connection_reachability_width = (
            connection_count
            * int(features.frontier_connection_reachability)
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

    def ensure(self, snapshot_count: int, sparse_row_count: int):
        if (
            self.snapshot_capacity >= snapshot_count
            and self.sparse_row_capacity >= sparse_row_count
        ):
            return
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
        log_temperature: torch.Tensor,
        include_temperature: bool,
        log_recommended_candidates: torch.Tensor,
        include_recommended_candidates: bool,
        log_exploration_candidates: torch.Tensor,
        include_exploration_candidates: bool,
        lookahead_outcomes: PreliminaryOutcomes,
        include_lookahead_outcomes: bool,
        sparse_row_count: int,
        frontier_count: int,
    ) -> SparseFeatures:
        snapshot_count = environment_count * candidate_count
        if not include_temperature:
            log_temperature = log_temperature.new_empty(
                [environment_count, candidate_count, 0]
            )
        if not include_recommended_candidates:
            log_recommended_candidates = log_recommended_candidates.new_empty(
                [environment_count, candidate_count, 0]
            )
        if not include_exploration_candidates:
            log_exploration_candidates = log_exploration_candidates.new_empty(
                [environment_count, candidate_count, 0]
            )
        lookahead_door_invalid = lookahead_outcomes.door_invalid
        lookahead_door_match = lookahead_outcomes.door_match
        lookahead_connection_invalid = lookahead_outcomes.connection_invalid
        if not include_lookahead_outcomes:
            lookahead_door_invalid = lookahead_door_invalid.new_empty(
                [environment_count, candidate_count, 0]
            )
            lookahead_door_match = lookahead_door_match.new_empty(
                [environment_count, candidate_count, 0]
            )
            lookahead_connection_invalid = lookahead_connection_invalid.new_empty(
                [environment_count, candidate_count, 0]
            )
        return SparseFeatures(
            self.inventory[:snapshot_count].view(
                environment_count, candidate_count, self.inventory_width
            ),
            self.room_x[:snapshot_count].view(environment_count, candidate_count, self.room_width),
            self.room_y[:snapshot_count].view(environment_count, candidate_count, self.room_width),
            self.room_placed[:snapshot_count].view(
                environment_count, candidate_count, self.room_width
            ),
            log_temperature,
            log_recommended_candidates,
            log_exploration_candidates,
            lookahead_door_invalid,
            lookahead_door_match,
            lookahead_connection_invalid,
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
class GenerationGroup:
    env: EnvironmentGroup
    config: GenerateConfig
    step: int
    feature_slot: PinnedSparseFeatureSlot | None
    previous_lookahead_outcomes: PreliminaryOutcomes | None
    previous_proposal_scores: CachedProposalScores | None


@dataclass
class CandidateBatch:
    candidates: Actions
    proposal_frontier_idx: torch.Tensor
    proposal_door_variant_idx: torch.Tensor
    reward_outcomes: PreliminaryOutcomes
    post_candidate_outcomes: PreliminaryOutcomes
    sparse_feature_requirements: SparseFeatureRequirements

    def to(self, device: torch.device) -> "CandidateBatch":
        return CandidateBatch(
            self.candidates.to(device),
            self.proposal_frontier_idx.to(device),
            self.proposal_door_variant_idx.to(device),
            self.reward_outcomes.to(device),
            self.post_candidate_outcomes.to(device),
            self.sparse_feature_requirements,
        )


@dataclass
class PreparedGenerationStep:
    candidate_batch: CandidateBatch
    features: Features | SparseFeatures | None


@dataclass
class PendingCandidateStep:
    group: GenerationGroup
    future: Future[PreparedGenerationStep]


@dataclass
class PendingProposalStep:
    group: GenerationGroup
    future: Future[Features | None]


def create_generation_environment_groups(
    config: Config,
    engine: Engine,
    generation_devices: list[torch.device],
) -> list[list[EnvironmentGroup]]:
    num_generation_groups = (
        config.generation.num_devices * config.generation.pipeline_groups
    )
    generation_group_environments = config.generation.num_environments // num_generation_groups
    generation_group_threads = (
        None
        if config.generation.num_threads is None
        else config.generation.num_threads // config.generation.pipeline_groups
    )
    logging.info(
        "Using %s pipeline group(s) per generation device with %s environment(s) and %s Rust worker(s) per group.",
        config.generation.pipeline_groups,
        generation_group_environments,
        generation_group_threads if generation_group_threads is not None else "automatic",
    )
    return [
        [
            engine.create_environment_group(
                config.map_size,
                generation_group_environments,
                seed=device_index * config.generation.pipeline_groups + group_index,
                frontier_neighbor_algorithm=config.generation.frontier_neighbor_algorithm,
                frontier_neighbor_count=config.generation.frontier_neighbor_count,
                frontier_window_size=config.generation.frontier_window_size,
                num_threads=generation_group_threads,
            )
            for group_index in range(config.generation.pipeline_groups)
        ]
        for device_index in range(len(generation_devices))
    ]


def get_generation_candidate_batch(
    group: GenerationGroup,
    proposal_scores: CachedProposalScores | None,
    device: torch.device,
) -> CandidateBatch:
    resolved_proposal_scores = None if proposal_scores is None else proposal_scores.resolve()
    (
        candidates,
        proposal_frontier_idx,
        proposal_door_variant_idx,
        reward_outcomes,
        post_candidate_outcomes,
        sparse_feature_requirements,
    ) = group.env.get_candidates_with_outcomes(
        group.config.recommended_candidates,
        group.config.exploration_candidates,
        group.config.proposal_temperature,
        resolved_proposal_scores,
        device,
    )
    return CandidateBatch(
        candidates,
        proposal_frontier_idx,
        proposal_door_variant_idx,
        reward_outcomes,
        post_candidate_outcomes,
        sparse_feature_requirements,
    )


def candidate_log_inputs(
    config: GenerateConfig,
    candidate_shape: torch.Size,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_log_temperature = config.temperature.to(torch.device("cpu")).log().unsqueeze(1)
    candidate_log_temperature = candidate_log_temperature.expand(candidate_shape).contiguous()
    candidate_log_recommended_candidates = torch.full(
        candidate_shape,
        math.log(config.recommended_candidates + 1),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    candidate_log_exploration_candidates = torch.full(
        candidate_shape,
        math.log(config.exploration_candidates + 1),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    return (
        candidate_log_temperature,
        candidate_log_recommended_candidates,
        candidate_log_exploration_candidates,
    )


def state_log_inputs(
    config: GenerateConfig,
    environment_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_temperature = config.temperature.to(torch.device("cpu")).log()
    log_recommended_candidates = torch.full(
        [environment_count],
        math.log(config.recommended_candidates + 1),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    log_exploration_candidates = torch.full(
        [environment_count],
        math.log(config.exploration_candidates + 1),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    return log_temperature, log_recommended_candidates, log_exploration_candidates


def select_outcomes(outcomes: PreliminaryOutcomes, index: torch.Tensor) -> PreliminaryOutcomes:
    def gather(values: torch.Tensor) -> torch.Tensor:
        gather_index = index.view(-1, 1, 1).expand(-1, 1, values.shape[2])
        return torch.gather(values, 1, gather_index).squeeze(1)

    return PreliminaryOutcomes(
        gather(outcomes.door_invalid),
        gather(outcomes.connection_invalid),
        gather(outcomes.door_match),
    )


def prepare_proposal_features(group: GenerationGroup) -> Features | None:
    if (
        group.step == 0
        or group.config.recommended_candidates == 0
        or group.previous_lookahead_outcomes is None
    ):
        return None
    environment_count = group.config.temperature.shape[0]
    (
        log_temperature,
        log_recommended_candidates,
        log_exploration_candidates,
    ) = state_log_inputs(group.config, environment_count)
    return group.env.get_features(
        torch.device("cpu"),
        log_temperature,
        group.env.engine.features.temperature,
        log_recommended_candidates,
        group.env.engine.features.recommended_candidates,
        log_exploration_candidates,
        group.env.engine.features.exploration_candidates,
        group.previous_lookahead_outcomes,
        group.env.engine.features.lookahead_outcomes,
    )


def prepare_candidate_features(
    env: EnvironmentGroup,
    config: GenerateConfig,
    candidate_batch: CandidateBatch,
    sparse_frontiers: bool,
    feature_slot: PinnedSparseFeatureSlot | None,
) -> PreparedGenerationStep:
    candidates = candidate_batch.candidates
    if candidates.room_idx.shape[1] == 1:
        return PreparedGenerationStep(candidate_batch, None)
    (
        candidate_log_temperature,
        candidate_log_recommended_candidates,
        candidate_log_exploration_candidates,
    ) = candidate_log_inputs(
        config,
        candidates.room_idx.shape,
    )
    return PreparedGenerationStep(
        candidate_batch,
        extract_candidate_features(
            env,
            candidates,
            candidate_log_temperature,
            env.engine.features.temperature,
            candidate_log_recommended_candidates,
            env.engine.features.recommended_candidates,
            candidate_log_exploration_candidates,
            env.engine.features.exploration_candidates,
            candidate_batch.post_candidate_outcomes,
            env.engine.features.lookahead_outcomes,
            candidate_batch.sparse_feature_requirements,
            sparse_frontiers,
            feature_slot,
        ),
    )


def prepare_lookahead_generation_step(
    group: GenerationGroup,
    proposal_scores: CachedProposalScores | None,
    sparse_frontiers: bool,
) -> PreparedGenerationStep:
    candidate_batch = get_generation_candidate_batch(
        group,
        proposal_scores,
        torch.device("cpu"),
    )
    return prepare_candidate_features(
        group.env,
        group.config,
        candidate_batch,
        sparse_frontiers,
        group.feature_slot,
    )


def select_candidate_actions(
    group: GenerationGroup,
    model,
    candidates: Actions,
    outcomes: PreliminaryOutcomes,
    features: Features | SparseFeatures,
    device: torch.device,
    gpu_lock: threading.Lock,
    transfer_stream: torch.cuda.Stream | None,
    proposal_copy_stream: torch.cuda.Stream | None,
    num_rooms: int,
    profiler: GenerationProfiler,
) -> tuple[torch.Tensor, Actions, torch.Tensor, CachedProposalScores | None]:
    environment_count, candidate_count = candidates.room_idx.shape
    profile = profiler.enabled
    with gpu_lock:
        sync_profile_device(device, profile)
        profile_time = profile_start(profile)
        env_features = transfer_features(features, device, transfer_stream)
        sync_profile_device(device, profile)
        profiler.add("python.score.transfer_features", profile_time)

        profile_time = profile_start(profile)
        with torch.amp.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and group.config.autocast,
        ):
            include_proposal = group.config.recommended_candidates > 0
            preds = model(
                env_features,
                include_proposal=include_proposal,
                return_proposal_state=False,
            )
        sync_profile_device(device, profile)
        profiler.add("python.score.model_forward", profile_time)

        profile_time = profile_start(profile)
        expected_reward = compute_expected_reward(
            Predictions(
                door_invalid=preds.door_invalid.view(environment_count, candidate_count, -1),
                connection_invalid=preds.connection_invalid.view(
                    environment_count,
                    candidate_count,
                    -1,
                ),
                balance_score=preds.balance_score.view(environment_count, candidate_count, -1),
                avg_frontiers=preds.avg_frontiers.view(environment_count, candidate_count),
                proposal_score=preds.proposal_score,
                proposal_state=preds.proposal_state,
            ),
            outcomes,
            group.config,
        )
        sync_profile_device(device, profile)
        profiler.add("python.score.reward", profile_time)

        profile_time = profile_start(profile)
        # Replace dummy candidates to have -inf reward, so they are never selected unless there are no other candidates.
        expected_reward = torch.where(
            candidates.room_idx == num_rooms,
            torch.full_like(expected_reward, float("-inf")),
            expected_reward,
        )
        candidate_logits = expected_reward / torch.unsqueeze(group.config.temperature, 1)
        probs = torch.softmax(candidate_logits, dim=1)
        action_index = rand_choice(probs)
        selected_actions = candidates.select(action_index)
        sync_profile_device(device, profile)
        profiler.add("python.score.sample", profile_time)

        profile_time = profile_start(profile)
        selected_proposal_scores = None
        if include_proposal:
            proposal_score = preds.proposal_score.view(
                environment_count,
                candidate_count,
                preds.proposal_score.shape[1],
                preds.proposal_score.shape[2],
            )
            selected_proposal_scores = proposal_score[
                torch.arange(environment_count, device=device),
                action_index,
            ]
            selected_proposal_scores = cache_proposal_scores_tensor(
                selected_proposal_scores,
                proposal_copy_stream,
            )
        profiler.add("python.score.cache_proposal", profile_time)
    return action_index, selected_actions, candidate_logits, selected_proposal_scores


def compute_proposal_scores(
    group: GenerationGroup,
    model,
    features: Features,
    device: torch.device,
    gpu_lock: threading.Lock,
    transfer_stream: torch.cuda.Stream | None,
    proposal_copy_stream: torch.cuda.Stream | None,
) -> CachedProposalScores:
    with gpu_lock:
        env_features = transfer_features(features, device, transfer_stream)
        with torch.amp.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and group.config.autocast,
        ):
            preds = model(env_features, include_proposal=True)
        return cache_proposal_scores_tensor(preds.proposal_score, proposal_copy_stream)


def verify_and_step(
    group: GenerationGroup,
    selected_actions: Actions,
    device: torch.device,
    verify_outcome_consistency: bool,
) -> None:
    group.env.step(selected_actions)
    if verify_outcome_consistency:
        group.env.get_outcomes(device, verify_consistency=True)


def start_generation_step(
    group: GenerationGroup,
    sparse_frontiers: bool,
    executor: ThreadPoolExecutor,
    pending_proposals: deque[PendingProposalStep],
    pending_candidates: deque[PendingCandidateStep],
) -> None:
    if group.step >= group.config.episode_length:
        return
    if group.step == 0 or group.config.recommended_candidates == 0:
        pending_candidates.append(
            PendingCandidateStep(
                group,
                executor.submit(
                    prepare_lookahead_generation_step,
                    group,
                    None,
                    sparse_frontiers,
                ),
            )
        )
        return
    if group.previous_proposal_scores is not None:
        start_candidate_step(
            group,
            group.previous_proposal_scores,
            sparse_frontiers,
            executor,
            pending_candidates,
        )
        return
    pending_proposals.append(
        PendingProposalStep(
            group,
            executor.submit(
                prepare_proposal_features,
                group,
            )
        )
    )


def start_candidate_step(
    group: GenerationGroup,
    proposal_scores: CachedProposalScores | None,
    sparse_frontiers: bool,
    executor: ThreadPoolExecutor,
    pending_candidates: deque[PendingCandidateStep],
) -> None:
    pending_candidates.append(
        PendingCandidateStep(
            group,
            executor.submit(
                prepare_lookahead_generation_step,
                group,
                proposal_scores,
                sparse_frontiers,
            ),
        )
    )


def merge_generation_results(
    results: list[tuple[EpisodeData, EpisodeOutcomes, DoorMatchCounts, ProposalData]],
) -> tuple[EpisodeData, EpisodeOutcomes, DoorMatchCounts, ProposalData]:
    return (
        EpisodeData(
            actions=Actions(
                *(
                    torch.cat([getattr(episode_data.actions, name) for episode_data, _, _, _ in results])
                    for name in vars(results[0][0].actions)
                )
            ),
            temperature=torch.cat([episode_data.temperature for episode_data, _, _, _ in results]),
            recommended_candidates=torch.cat([
                episode_data.recommended_candidates for episode_data, _, _, _ in results
            ]),
            exploration_candidates=torch.cat([
                episode_data.exploration_candidates for episode_data, _, _, _ in results
            ]),
        ),
        EpisodeOutcomes(
            validity=PreliminaryOutcomes(
                *(
                    torch.cat([
                        getattr(episode_outcomes.validity, name)
                        for _, episode_outcomes, _, _ in results
                    ])
                    for name in vars(results[0][1].validity)
                )
            ),
            avg_frontiers=torch.cat([
                episode_outcomes.avg_frontiers
                for _, episode_outcomes, _, _ in results
            ]),
        ),
        DoorMatchCounts(
            *(
                torch.sum(
                    torch.stack([getattr(counts, name) for _, _, counts, _ in results]),
                    dim=0,
                )
                for name in vars(results[0][2])
            )
        ),
        ProposalData(
            *(
                torch.cat([getattr(proposal, name) for _, _, _, proposal in results])
                for name in vars(results[0][3])
            )
        ),
    )


def run_generation_groups(
    envs: list[EnvironmentGroup],
    model,
    configs: list[GenerateConfig],
    device: torch.device,
    verify_outcome_consistency: bool = False,
    profile: bool = False,
) -> tuple[EpisodeData, EpisodeOutcomes, DoorMatchCounts, ProposalData, ProfileReport]:
    if not envs or len(envs) != len(configs):
        raise ValueError("generation groups require one config per environment group")
    profiler = GenerationProfiler(profile)
    transfer_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    proposal_copy_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    gpu_lock = threading.Lock()
    num_rooms = len(envs[0].engine.rooms)
    sparse_frontiers = device.type == "cuda"
    groups = [
        GenerationGroup(
            env,
            config,
            0,
            PinnedSparseFeatureSlot(env, pin_memory=True)
            if device.type == "cuda"
            else None,
            None,
            None,
        )
        for env, config in zip(envs, configs)
    ]
    group_index_by_id = {id(group): idx for idx, group in enumerate(groups)}
    with ThreadPoolExecutor(max_workers=len(groups)) as executor:
        pending_proposals: deque[PendingProposalStep] = deque()
        pending_candidates: deque[PendingCandidateStep] = deque()
        group_proposal_frontier_idx = [[] for _ in groups]
        group_proposal_door_variant_idx = [[] for _ in groups]
        group_selected_candidate = [[] for _ in groups]
        group_proposal_target_logits = [[] for _ in groups]
        with torch.no_grad():
            for group in groups:
                group.env.clear()
                group.previous_lookahead_outcomes = None
                group.previous_proposal_scores = None
                start_generation_step(
                    group,
                    sparse_frontiers,
                    executor,
                    pending_proposals,
                    pending_candidates,
                )
            while pending_proposals or pending_candidates:
                if not pending_candidates:
                    while pending_proposals:
                        proposal_step = pending_proposals.popleft()
                        profile_time = profile_start(profile)
                        proposal_features = proposal_step.future.result()
                        profiler.add("python.wait_proposal_features", profile_time)
                        proposal_scores = (
                            None
                            if proposal_features is None
                            else compute_proposal_scores(
                                proposal_step.group,
                                model,
                                proposal_features,
                                device,
                                gpu_lock,
                                transfer_stream,
                                proposal_copy_stream,
                            )
                        )
                        start_candidate_step(
                            proposal_step.group,
                            proposal_scores,
                            sparse_frontiers,
                            executor,
                            pending_candidates,
                        )
                    continue

                step = pending_candidates.popleft()
                profile_time = profile_start(profile)
                prepared_step = step.future.result()
                profiler.add("python.wait_candidate_features", profile_time)
                profile_time = profile_start(profile)
                candidate_batch = prepared_step.candidate_batch.to(device)
                profiler.add("python.transfer_candidate_batch", profile_time)
                candidates = candidate_batch.candidates
                if prepared_step.features is None:
                    action_index = torch.zeros(
                        candidates.room_idx.shape[0],
                        dtype=torch.int64,
                        device=device,
                    )
                    selected_actions = candidates.select(action_index)
                    candidate_logits = torch.zeros(
                        candidates.room_idx.shape,
                        dtype=torch.float32,
                        device=device,
                    )
                    selected_proposal_scores = None
                else:
                    (
                        action_index,
                        selected_actions,
                        candidate_logits,
                        selected_proposal_scores,
                    ) = select_candidate_actions(
                        step.group,
                        model,
                        candidates,
                        candidate_batch.reward_outcomes,
                        prepared_step.features,
                        device,
                        gpu_lock,
                        transfer_stream,
                        proposal_copy_stream,
                        num_rooms,
                        profiler,
                    )
                group_index = group_index_by_id[id(step.group)]
                profile_time = profile_start(profile)
                max_candidates = step.group.config.max_candidates
                frontier_idx = torch.full(
                    [candidates.room_idx.shape[0], max_candidates],
                    -1,
                    dtype=candidate_batch.proposal_frontier_idx.dtype,
                    device=device,
                )
                door_variant_idx = torch.full(
                    [candidates.room_idx.shape[0], max_candidates],
                    -1,
                    dtype=candidate_batch.proposal_door_variant_idx.dtype,
                    device=device,
                )
                target_logits = torch.full(
                    [candidates.room_idx.shape[0], max_candidates],
                    float("-inf"),
                    dtype=torch.float32,
                    device=device,
                )
                frontier_idx[:, :candidate_batch.proposal_frontier_idx.shape[1]] = (
                    candidate_batch.proposal_frontier_idx
                )
                door_variant_idx[:, :candidate_batch.proposal_door_variant_idx.shape[1]] = (
                    candidate_batch.proposal_door_variant_idx
                )
                target_logits[:, :candidate_logits.shape[1]] = candidate_logits.to(torch.float32)
                group_proposal_frontier_idx[group_index].append(frontier_idx)
                group_proposal_door_variant_idx[group_index].append(door_variant_idx)
                group_selected_candidate[group_index].append(action_index)
                group_proposal_target_logits[group_index].append(target_logits)
                profiler.add("python.record_proposal_data", profile_time)
                profile_time = profile_start(profile)
                step.group.previous_lookahead_outcomes = select_outcomes(
                    candidate_batch.post_candidate_outcomes,
                    action_index,
                ).to(torch.device("cpu"))
                step.group.previous_proposal_scores = selected_proposal_scores
                profiler.add("python.cache_next_proposal", profile_time)
                profile_time = profile_start(profile)
                verify_and_step(
                    step.group,
                    selected_actions,
                    device,
                    verify_outcome_consistency,
                )
                profiler.add("python.step_environment", profile_time)
                step.group.step += 1
                if step.group.step < step.group.config.episode_length:
                    start_generation_step(
                        step.group,
                        sparse_frontiers,
                        executor,
                        pending_proposals,
                        pending_candidates,
                    )
        results = []
        for group_index, group in enumerate(groups):
            profile_time = profile_start(profile)
            group.env.finish()
            actions = group.env.get_actions(device)
            episode_outcomes = group.env.get_outcomes(
                device, verify_consistency=verify_outcome_consistency
            )
            door_match_counts = group.env.get_door_match_counts(device)
            results.append((
                EpisodeData(
                    actions,
                    group.config.temperature,
                    torch.full_like(
                        group.config.temperature,
                        group.config.recommended_candidates,
                        dtype=torch.float32,
                    ),
                    torch.full_like(
                        group.config.temperature,
                        group.config.exploration_candidates,
                        dtype=torch.float32,
                    ),
                ),
                episode_outcomes,
                door_match_counts,
                ProposalData(
                    torch.stack(group_proposal_frontier_idx[group_index], dim=1),
                    torch.stack(group_proposal_door_variant_idx[group_index], dim=1),
                    torch.stack(group_selected_candidate[group_index], dim=1),
                    torch.stack(group_proposal_target_logits[group_index], dim=1),
                ),
            ))
            profiler.add("python.finish_group", profile_time)
    (
        episode_data,
        outcomes,
        door_match_counts,
        proposal_data,
    ) = merge_generation_results(results)
    return episode_data, outcomes, door_match_counts, proposal_data, profiler.report()
