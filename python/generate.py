from __future__ import annotations

from env import (
    Actions,
    CandidateStats,
    DoorMatchCounts,
    Engine,
    EnvironmentGroup,
    EpisodeData,
    EpisodeOutcomes,
    GenerateConfig,
    PreliminaryOutcomes,
    ProposalData,
    ProposalCandidateMask,
    CandidateFeatureRequirements,
    SparseFeatures,
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
type GenerationStats = dict[str, float]


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
    lookahead_outcomes: PreliminaryOutcomes,
    include_lookahead_outcomes: bool,
    feature_requirements: CandidateFeatureRequirements,
    feature_slot: SparseFeatureSlot,
):
    sparse_row_count = feature_requirements.sparse_row_count
    worker_sparse_row_counts = feature_requirements.worker_sparse_row_counts
    feature_slot.ensure(
        candidates.room_idx.numel(),
        sparse_row_count,
    )
    env.env.pack_sparse_features_after_candidates_into(
        candidates.room_idx.shape[0],
        candidates.room_idx.shape[1],
        0,
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
        feature_slot.row_snapshot_idx.numpy(),
        feature_slot.row_frontier_idx.numpy(),
    )
    return feature_slot.features(
        candidates.room_idx.shape[0],
        candidates.room_idx.shape[1],
        log_temperature,
        include_temperature,
        log_recommended_candidates,
        include_recommended_candidates,
        lookahead_outcomes,
        include_lookahead_outcomes,
        sparse_row_count,
    ).flatten_candidates()


def transfer_features(
    features: SparseFeatures,
    device: torch.device,
    transfer_stream: torch.cuda.Stream | None = None,
) -> SparseFeatures:
    if transfer_stream is None or device.type != "cuda":
        return features.to(device)
    current_stream = torch.cuda.current_stream(device)
    with torch.cuda.device(device), torch.cuda.stream(transfer_stream):
        result = features.to(device, non_blocking=True)
        ready = torch.cuda.Event()
        ready.record(transfer_stream)
    current_stream.wait_event(ready)
    return result


# When a GPU is available, we use pinned memory for model input tensors,
# to allow for asynchronous CPU-to-GPU transfers.
class SparseFeatureSlot:
    def __init__(self, env: EnvironmentGroup, pin_memory: bool):
        features = env.engine.features
        inventory_count, _, room_count = env.engine.get_feature_sizes()
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
        self.row_snapshot_idx = None
        self.row_frontier_idx = None

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
        self.row_snapshot_idx = self._empty((self.sparse_row_capacity,), torch.int64)
        self.row_frontier_idx = self._empty((self.sparse_row_capacity,), torch.int16)

    def features(
        self,
        environment_count: int,
        candidate_count: int,
        log_temperature: torch.Tensor,
        include_temperature: bool,
        log_recommended_candidates: torch.Tensor,
        include_recommended_candidates: bool,
        lookahead_outcomes: PreliminaryOutcomes,
        include_lookahead_outcomes: bool,
        sparse_row_count: int,
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
            self.row_snapshot_idx[:sparse_row_count],
            self.row_frontier_idx[:sparse_row_count],
        )


@dataclass
class GenerationGroup:
    env: EnvironmentGroup
    config: GenerateConfig
    step: int
    feature_slot: SparseFeatureSlot
    previous_lookahead_outcomes: PreliminaryOutcomes | None
    previous_proposal_scores: SparseProposalCache | None


@dataclass
class SparseProposalCache:
    state: torch.Tensor
    row_start_idx: torch.Tensor
    row_count: torch.Tensor


@dataclass
class CandidateBatch:
    candidates: Actions
    proposal_frontier_idx: torch.Tensor
    proposal_door_variant_idx: torch.Tensor
    reward_outcomes: PreliminaryOutcomes
    post_candidate_outcomes: PreliminaryOutcomes
    feature_requirements: CandidateFeatureRequirements
    stats: CandidateStats

    def to(self, device: torch.device) -> "CandidateBatch":
        return CandidateBatch(
            self.candidates.to(device),
            self.proposal_frontier_idx.to(device),
            self.proposal_door_variant_idx.to(device),
            self.reward_outcomes.to(device),
            self.post_candidate_outcomes.to(device),
            self.feature_requirements,
            self.stats.to(device),
        )


@dataclass
class PreparedGenerationStep:
    candidate_batch: CandidateBatch
    features: SparseFeatures | None


@dataclass
class PendingCandidateStep:
    group: GenerationGroup
    future: Future[PreparedGenerationStep]
    shortlist_limited: torch.Tensor


@dataclass
class PendingProposalStep:
    group: GenerationGroup
    future: Future[ProposalInputs]


@dataclass
class ProposalInputs:
    features: SparseFeatures | None
    mask: ProposalCandidateMask


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


def get_initial_candidate_batch(group: GenerationGroup, device: torch.device) -> CandidateBatch:
    (
        candidates,
        proposal_frontier_idx,
        proposal_door_variant_idx,
        reward_outcomes,
        post_candidate_outcomes,
        feature_requirements,
        stats,
    ) = group.env.get_candidates_with_outcomes(
        group.config.recommended_candidates,
        group.config.proposal_temperature,
        None,
        device,
    )
    return CandidateBatch(
        candidates,
        proposal_frontier_idx,
        proposal_door_variant_idx,
        reward_outcomes,
        post_candidate_outcomes,
        feature_requirements,
        stats,
    )


def get_shortlist_candidate_batch(
    group: GenerationGroup,
    sampled_frontier_idx: torch.Tensor,
    sampled_door_variant_idx: torch.Tensor,
    device: torch.device,
) -> CandidateBatch:
    (
        candidates,
        proposal_frontier_idx,
        proposal_door_variant_idx,
        reward_outcomes,
        post_candidate_outcomes,
        feature_requirements,
        stats,
    ) = group.env.get_candidates_from_proposals(
        sampled_frontier_idx,
        sampled_door_variant_idx,
        group.config.recommended_candidates,
        device,
    )
    return CandidateBatch(
        candidates,
        proposal_frontier_idx,
        proposal_door_variant_idx,
        reward_outcomes,
        post_candidate_outcomes,
        feature_requirements,
        stats,
    )


def unpack_proposal_mask(mask: ProposalCandidateMask, device: torch.device) -> torch.Tensor:
    packed = mask.mask.to(device)
    shifts = torch.arange(8, device=device, dtype=packed.dtype)
    bits = ((packed.unsqueeze(-1) >> shifts) & 1).to(torch.bool).flatten(1)
    return bits[:, : mask.door_variant_count]


def row_scores_for_mask(
    row_scores: torch.Tensor,
    row_snapshot_idx: torch.Tensor,
    row_frontier_idx: torch.Tensor,
    proposal_mask: ProposalCandidateMask,
    device: torch.device,
) -> torch.Tensor:
    proposal_frontier_idx = proposal_mask.proposal_frontier_idx.to(device)
    door_variant_count = row_scores.shape[1]
    result = torch.full(
        (proposal_frontier_idx.shape[0], door_variant_count),
        float("-inf"),
        dtype=row_scores.dtype,
        device=device,
    )
    if row_scores.shape[0] == 0:
        return result
    row_snapshot_idx = row_snapshot_idx.to(device)
    row_frontier_idx = row_frontier_idx.to(device)
    row_valid = (
        (row_snapshot_idx >= 0)
        & (row_snapshot_idx < proposal_frontier_idx.shape[0])
        & (row_frontier_idx == proposal_frontier_idx[row_snapshot_idx])
    )
    if torch.any(row_valid):
        result[row_snapshot_idx[row_valid]] = row_scores[row_valid]
    return result


def sample_proposal_shortlist(
    proposal_scores: torch.Tensor,
    proposal_mask: ProposalCandidateMask,
    config: GenerateConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    proposal_door_variant_count = proposal_scores.shape[1]
    environment_count = proposal_mask.proposal_frontier_idx.shape[0]
    if proposal_door_variant_count == 0:
        empty_sampled = torch.full(
            (environment_count, config.shortlist_candidates),
            -1,
            dtype=torch.int16,
            device=device,
        )
        return empty_sampled, empty_sampled
    frontier_idx = proposal_mask.proposal_frontier_idx.to(device)
    valid_frontier = frontier_idx >= 0
    valid = unpack_proposal_mask(proposal_mask, device)[:, :proposal_door_variant_count]
    valid = valid & valid_frontier.unsqueeze(1)
    sample_keys = proposal_scores.to(dtype=torch.float32, copy=True)
    sample_keys.div_(config.proposal_temperature.to(device).view(-1, 1).clamp_min(1e-6))
    sample_keys.masked_fill_(~valid, float("-inf"))
    shortlist_candidates = min(config.shortlist_candidates, sample_keys.shape[1])
    gumbel = torch.empty_like(sample_keys).exponential_().log_().neg_()
    sample_keys.add_(gumbel)
    sampled_flat = torch.topk(
        sample_keys,
        shortlist_candidates,
        dim=1,
        sorted=True,
    ).indices
    sampled_is_valid = valid.gather(1, sampled_flat)
    sampled_door_variant_idx = sampled_flat
    sampled_door_variant_idx = torch.where(
        sampled_is_valid,
        sampled_door_variant_idx,
        torch.full_like(sampled_door_variant_idx, -1),
    )
    if shortlist_candidates < config.shortlist_candidates:
        padding = torch.full(
            (environment_count, config.shortlist_candidates - shortlist_candidates),
            -1,
            dtype=sampled_flat.dtype,
            device=device,
        )
        sampled_door_variant_idx = torch.cat([sampled_door_variant_idx, padding], dim=1)
    sampled_frontier_idx = frontier_idx.unsqueeze(1).expand(-1, config.shortlist_candidates)
    sampled_frontier_idx = torch.where(
        sampled_door_variant_idx >= 0,
        sampled_frontier_idx,
        torch.full_like(sampled_frontier_idx, -1),
    )
    return (
        sampled_frontier_idx.to(torch.int16),
        sampled_door_variant_idx.to(torch.int16),
    )


def candidate_log_inputs(
    config: GenerateConfig,
    candidate_shape: torch.Size,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidate_log_temperature = config.temperature.to(torch.device("cpu")).log().unsqueeze(1)
    candidate_log_temperature = candidate_log_temperature.expand(candidate_shape).contiguous()
    candidate_log_recommended_candidates = torch.full(
        candidate_shape,
        math.log(config.recommended_candidates + 1),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    return (
        candidate_log_temperature,
        candidate_log_recommended_candidates,
    )


def state_log_inputs(
    config: GenerateConfig,
    environment_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_temperature = config.temperature.to(torch.device("cpu")).log()
    log_recommended_candidates = torch.full(
        [environment_count],
        math.log(config.recommended_candidates + 1),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    return log_temperature, log_recommended_candidates


def select_outcomes(outcomes: PreliminaryOutcomes, index: torch.Tensor) -> PreliminaryOutcomes:
    def gather(values: torch.Tensor) -> torch.Tensor:
        gather_index = index.view(-1, 1, 1).expand(-1, 1, values.shape[2])
        return torch.gather(values, 1, gather_index).squeeze(1)

    return PreliminaryOutcomes(
        gather(outcomes.door_invalid),
        gather(outcomes.connection_invalid),
        gather(outcomes.door_match),
    )


def prepare_proposal_inputs(group: GenerationGroup) -> ProposalInputs:
    proposal_mask = group.env.get_proposal_candidate_mask(
        torch.device("cpu"),
    )
    if group.previous_proposal_scores is not None:
        return ProposalInputs(None, proposal_mask)
    if group.previous_lookahead_outcomes is None:
        raise ValueError("proposal features require previous lookahead outcomes")
    environment_count = group.config.temperature.shape[0]
    (
        log_temperature,
        log_recommended_candidates,
    ) = state_log_inputs(group.config, environment_count)
    return ProposalInputs(
        group.env.get_sparse_features(
            torch.device("cpu"),
            log_temperature,
            group.env.engine.features.temperature,
            log_recommended_candidates,
            group.env.engine.features.recommended_candidates,
            group.previous_lookahead_outcomes,
            group.env.engine.features.lookahead_outcomes,
        ),
        proposal_mask,
    )


def prepare_candidate_features(
    env: EnvironmentGroup,
    config: GenerateConfig,
    candidate_batch: CandidateBatch,
    feature_slot: SparseFeatureSlot,
) -> PreparedGenerationStep:
    candidates = candidate_batch.candidates
    if candidates.room_idx.shape[1] == 1:
        return PreparedGenerationStep(candidate_batch, None)
    (
        candidate_log_temperature,
        candidate_log_recommended_candidates,
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
            candidate_batch.post_candidate_outcomes,
            env.engine.features.lookahead_outcomes,
            candidate_batch.feature_requirements,
            feature_slot,
        ),
    )


def prepare_initial_generation_step(
    group: GenerationGroup,
) -> PreparedGenerationStep:
    candidate_batch = get_initial_candidate_batch(group, torch.device("cpu"))
    return prepare_candidate_features(
        group.env,
        group.config,
        candidate_batch,
        group.feature_slot,
    )


def prepare_shortlist_generation_step(
    group: GenerationGroup,
    sampled_frontier_idx: torch.Tensor,
    sampled_door_variant_idx: torch.Tensor,
) -> PreparedGenerationStep:
    candidate_batch = get_shortlist_candidate_batch(
        group,
        sampled_frontier_idx,
        sampled_door_variant_idx,
        torch.device("cpu"),
    )
    return prepare_candidate_features(
        group.env,
        group.config,
        candidate_batch,
        group.feature_slot,
    )


def select_candidate_actions(
    group: GenerationGroup,
    model,
    candidates: Actions,
    outcomes: PreliminaryOutcomes,
    features: SparseFeatures,
    device: torch.device,
    gpu_lock: threading.Lock,
    transfer_stream: torch.cuda.Stream | None,
    num_rooms: int,
    profiler: GenerationProfiler,
) -> tuple[torch.Tensor, Actions, torch.Tensor, torch.Tensor | None]:
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
                include_proposal=False,
                return_proposal_state=include_proposal,
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
                proposal_row_snapshot_idx=preds.proposal_row_snapshot_idx,
                proposal_row_frontier_idx=preds.proposal_row_frontier_idx,
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
        valid_row = torch.any(torch.isfinite(candidate_logits), dim=1)
        safe_candidate_logits = torch.where(
            valid_row.unsqueeze(1),
            candidate_logits,
            torch.where(
                torch.arange(candidate_count, device=device).unsqueeze(0) == 0,
                torch.zeros_like(candidate_logits),
                torch.full_like(candidate_logits, float("-inf")),
            ),
        )
        probs = torch.softmax(safe_candidate_logits, dim=1)
        action_index = rand_choice(probs)
        selected_actions = candidates.select(action_index)
        sync_profile_device(device, profile)
        profiler.add("python.score.sample", profile_time)

        profile_time = profile_start(profile)
        selected_proposal_scores = None
        if include_proposal:
            proposal_row_snapshot_idx = preds.proposal_row_snapshot_idx
            row_environment_idx = torch.div(
                proposal_row_snapshot_idx,
                candidate_count,
                rounding_mode="floor",
            )
            row_candidate_idx = proposal_row_snapshot_idx - (
                row_environment_idx * candidate_count
            )
            row_selected = row_candidate_idx == action_index[row_environment_idx]
            selected_state = preds.proposal_state[row_selected]
            row_count_by_environment = torch.bincount(
                row_environment_idx[row_selected],
                minlength=environment_count,
            )
            row_start_idx = row_count_by_environment.cumsum(0) - row_count_by_environment
            selected_proposal_scores = SparseProposalCache(
                selected_state,
                row_start_idx,
                row_count_by_environment,
            )
            sync_profile_device(device, profile)
        profiler.add("python.score.cache_proposal", profile_time)
    return action_index, selected_actions, candidate_logits, selected_proposal_scores


def compute_proposal_scores(
    group: GenerationGroup,
    model,
    features: SparseFeatures,
    proposal_mask: ProposalCandidateMask,
    device: torch.device,
    gpu_lock: threading.Lock,
    transfer_stream: torch.cuda.Stream | None,
) -> torch.Tensor:
    with gpu_lock:
        env_features = transfer_features(features, device, transfer_stream)
        with torch.amp.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and group.config.autocast,
        ):
            preds = model(env_features, include_proposal=True)
        return row_scores_for_mask(
            preds.proposal_score,
            preds.proposal_row_snapshot_idx,
            preds.proposal_row_frontier_idx,
            proposal_mask,
            device,
        )


def compute_cached_proposal_scores(
    group: GenerationGroup,
    model,
    cache: SparseProposalCache,
    proposal_mask: ProposalCandidateMask,
    device: torch.device,
) -> torch.Tensor:
    proposal_frontier_idx = proposal_mask.proposal_frontier_idx.to(device)
    door_variant_count = model.proposal_output.out_features
    if cache.state.shape[0] == 0:
        return torch.full(
            (proposal_frontier_idx.shape[0], door_variant_count),
            float("-inf"),
            dtype=cache.state.dtype,
            device=device,
        )
    row_start_idx = cache.row_start_idx.to(device)
    row_count = cache.row_count.to(device)
    row_valid = (
        (proposal_frontier_idx >= 0)
        & (proposal_frontier_idx < row_count)
    )
    safe_frontier_idx = torch.minimum(
        proposal_frontier_idx.clamp_min(0).to(torch.int64),
        row_count.clamp_min(1) - 1,
    )
    row_idx = (row_start_idx + safe_frontier_idx).clamp_max(cache.state.shape[0] - 1)
    with torch.amp.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and group.config.autocast,
    ):
        scores = model.proposal_output(cache.state[row_idx])
    return torch.where(
        row_valid.unsqueeze(1),
        scores,
        torch.full_like(scores, float("-inf")),
    )


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
    executor: ThreadPoolExecutor,
    pending_proposals: deque[PendingProposalStep],
    pending_candidates: deque[PendingCandidateStep],
) -> None:
    if group.step >= group.config.episode_length:
        return
    if group.step == 0:
        pending_candidates.append(
            PendingCandidateStep(
                group,
                executor.submit(
                    prepare_initial_generation_step,
                    group,
                ),
                torch.zeros(
                    [group.config.temperature.shape[0]],
                    dtype=torch.bool,
                    device=torch.device("cpu"),
                ),
            )
        )
        return
    pending_proposals.append(
        PendingProposalStep(
            group,
            executor.submit(
                prepare_proposal_inputs,
                group,
            )
        )
    )


def start_candidate_step(
    group: GenerationGroup,
    sampled_frontier_idx: torch.Tensor,
    sampled_door_variant_idx: torch.Tensor,
    shortlist_limited: torch.Tensor,
    executor: ThreadPoolExecutor,
    pending_candidates: deque[PendingCandidateStep],
) -> None:
    pending_candidates.append(
        PendingCandidateStep(
            group,
            executor.submit(
                prepare_shortlist_generation_step,
                group,
                sampled_frontier_idx,
                sampled_door_variant_idx,
            ),
            shortlist_limited,
        )
    )


def pop_ready_proposal_step(
    pending_proposals: deque[PendingProposalStep],
) -> PendingProposalStep | None:
    for index, proposal_step in enumerate(pending_proposals):
        if proposal_step.future.done():
            del pending_proposals[index]
            return proposal_step
    return None


def process_proposal_step(
    proposal_step: PendingProposalStep,
    model,
    device: torch.device,
    gpu_lock: threading.Lock,
    transfer_stream: torch.cuda.Stream | None,
    executor: ThreadPoolExecutor,
    pending_candidates: deque[PendingCandidateStep],
    profiler: GenerationProfiler,
    stat_totals: GenerationStats,
    profile: bool,
) -> None:
    profile_time = profile_start(profile)
    proposal_inputs = proposal_step.future.result()
    profiler.add("python.wait_proposal_features", profile_time)
    before_shortlist_time = profile_start(profile)
    valid_counts = proposal_inputs.mask.valid_counts
    row_valid_counts = valid_counts
    shortlist_limited = (
        row_valid_counts > proposal_step.group.config.shortlist_candidates
    )
    stat_totals["proposal_mask_rows"] += float(row_valid_counts.numel())
    stat_totals["proposal_valid_cells"] += float(row_valid_counts.sum().item())
    stat_totals["proposal_full_set_rows"] += float(
        (row_valid_counts <= proposal_step.group.config.shortlist_candidates)
        .sum()
        .item()
    )
    if proposal_step.group.previous_proposal_scores is not None:
        profile_time = profile_start(profile)
        proposal_scores = compute_cached_proposal_scores(
            proposal_step.group,
            model,
            proposal_step.group.previous_proposal_scores,
            proposal_inputs.mask,
            device,
        )
        sync_profile_device(device, profile)
        profiler.add("python.proposal.compute_cached_scores", profile_time)
    else:
        if proposal_inputs.features is None:
            raise ValueError("proposal scores require proposal features")
        profile_time = profile_start(profile)
        proposal_scores = compute_proposal_scores(
            proposal_step.group,
            model,
            proposal_inputs.features,
            proposal_inputs.mask,
            device,
            gpu_lock,
            transfer_stream,
        )
        sync_profile_device(device, profile)
        profiler.add("python.proposal.compute_fresh_scores", profile_time)
    profiler.add("python.proposal.total_before_shortlist", before_shortlist_time)
    profile_time = profile_start(profile)
    (
        sampled_frontier_idx,
        sampled_door_variant_idx,
    ) = sample_proposal_shortlist(
        proposal_scores,
        proposal_inputs.mask,
        proposal_step.group.config,
        device,
    )
    sync_profile_device(device, profile)
    profiler.add("python.proposal.sample_shortlist", profile_time)
    start_candidate_step(
        proposal_step.group,
        sampled_frontier_idx.to(torch.device("cpu")),
        sampled_door_variant_idx.to(torch.device("cpu")),
        shortlist_limited.to(torch.device("cpu")),
        executor,
        pending_candidates,
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
) -> tuple[EpisodeData, EpisodeOutcomes, DoorMatchCounts, ProposalData, GenerationStats, ProfileReport]:
    if not envs or len(envs) != len(configs):
        raise ValueError("generation groups require one config per environment group")
    profiler = GenerationProfiler(profile)
    transfer_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    gpu_lock = threading.Lock()
    num_rooms = len(envs[0].engine.rooms)
    groups = [
        GenerationGroup(
            env,
            config,
            0,
            SparseFeatureSlot(env, pin_memory=device.type == "cuda"),
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
        stat_totals = {
            "proposal_mask_rows": 0.0,
            "proposal_valid_cells": 0.0,
            "proposal_full_set_rows": 0.0,
            "proposal_clean_candidates": 0.0,
            "proposal_evaluated_candidates": 0.0,
            "proposal_rejected_candidates": 0.0,
            "proposal_exhausted_rows": 0.0,
        }
        with torch.no_grad():
            for group in groups:
                group.env.clear()
                group.previous_lookahead_outcomes = None
                group.previous_proposal_scores = None
                start_generation_step(
                    group,
                    executor,
                    pending_proposals,
                    pending_candidates,
                )
            while pending_proposals or pending_candidates:
                proposal_step = pop_ready_proposal_step(pending_proposals)
                if proposal_step is not None:
                    profile_time = profile_start(profile)
                    process_proposal_step(
                        proposal_step,
                        model,
                        device,
                        gpu_lock,
                        transfer_stream,
                        executor,
                        pending_candidates,
                        profiler,
                        stat_totals,
                        profile,
                    )
                    profiler.add(
                        "python.pipeline.ready_proposal_while_candidate_pending",
                        profile_time,
                    )
                    continue
                if not pending_candidates:
                    proposal_drain_time = profile_start(profile)
                    while pending_proposals:
                        process_proposal_step(
                            pending_proposals.popleft(),
                            model,
                            device,
                            gpu_lock,
                            transfer_stream,
                            executor,
                            pending_candidates,
                            profiler,
                            stat_totals,
                            profile,
                        )
                    profiler.add("python.pipeline.no_pending_candidates", proposal_drain_time)
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
                        num_rooms,
                        profiler,
                    )
                if step.group.step > 0:
                    stats = candidate_batch.stats
                    stat_totals["proposal_clean_candidates"] += float(
                        stats.clean_counts.sum().item()
                    )
                    stat_totals["proposal_evaluated_candidates"] += float(
                        stats.evaluated_counts.sum().item()
                    )
                    stat_totals["proposal_rejected_candidates"] += float(
                        stats.rejected_counts.sum().item()
                    )
                    stat_totals["proposal_exhausted_rows"] += float(
                        (
                            (
                                stats.clean_counts
                                < step.group.config.recommended_candidates
                            )
                            & step.shortlist_limited.to(device)
                        ).sum().item()
                    )
                group_index = group_index_by_id[id(step.group)]
                profile_time = profile_start(profile)
                max_candidates = step.group.config.max_candidates
                candidate_frontier_idx = candidate_batch.proposal_frontier_idx
                frontier_idx = (
                    candidate_frontier_idx[:, 0]
                    if candidate_frontier_idx.shape[1] > 0
                    else torch.full(
                        [candidates.room_idx.shape[0]],
                        -1,
                        dtype=candidate_frontier_idx.dtype,
                        device=device,
                    )
                )
                if candidate_batch.proposal_door_variant_idx.shape[1] == max_candidates:
                    door_variant_idx = candidate_batch.proposal_door_variant_idx
                else:
                    door_variant_idx = torch.full(
                        [candidates.room_idx.shape[0], max_candidates],
                        -1,
                        dtype=candidate_batch.proposal_door_variant_idx.dtype,
                        device=device,
                    )
                    door_variant_idx[:, :candidate_batch.proposal_door_variant_idx.shape[1]] = (
                        candidate_batch.proposal_door_variant_idx
                    )
                if candidate_logits.shape[1] == max_candidates:
                    target_logits = candidate_logits.to(torch.float32)
                else:
                    target_logits = torch.full(
                        [candidates.room_idx.shape[0], max_candidates],
                        float("-inf"),
                        dtype=torch.float32,
                        device=device,
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
    proposal_rows = max(stat_totals["proposal_mask_rows"], 1.0)
    evaluated = max(stat_totals["proposal_evaluated_candidates"], 1.0)
    generation_stats = {
        "proposal_valid_cells": stat_totals["proposal_valid_cells"] / proposal_rows,
        "proposal_full_set_rate": stat_totals["proposal_full_set_rows"] / proposal_rows,
        "proposal_clean_candidates": stat_totals["proposal_clean_candidates"] / proposal_rows,
        "proposal_rejection_rate": stat_totals["proposal_rejected_candidates"] / evaluated,
        "proposal_exhaustion_rate": stat_totals["proposal_exhausted_rows"] / proposal_rows,
    }
    return episode_data, outcomes, door_match_counts, proposal_data, generation_stats, profiler.report()
