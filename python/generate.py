from __future__ import annotations

from env import (
    Actions,
    CandidateStats,
    CandidateSlot,
    DoorMatchCounts,
    Engine,
    EndOutcomes,
    EnvironmentGroup,
    EpisodeData,
    EpisodeOutcomes,
    GenerateConfig,
    StepOutcomes,
    ProposalData,
    ProposalCandidateMask,
    FeatureRequirements,
    FeatureSlot,
    Features,
    extract_candidate_features,
)
from loss import compute_step_balance_score_target_logits
from model import BalancePredictions, Predictions
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Empty, Full, Queue
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
# Generation runs several environment groups per generation device. Each group
# has a CPU producer that owns environment mutation and feature extraction. A
# transfer coordinator stages completed candidate batches onto a CUDA transfer
# stream, and a GPU scorer consumes those staged batches. The scorer returns
# selected actions through a per-group queue so the owning CPU producer can step
# its environment and continue.


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
        self.lock = threading.Lock()

    def add(self, name: str, start: int) -> None:
        if not self.enabled:
            return
        elapsed = time.perf_counter_ns() - start
        with self.lock:
            self.counts[name] = self.counts.get(name, 0) + 1
            self.nanos[name] = self.nanos.get(name, 0) + elapsed

    def report(self) -> ProfileReport:
        with self.lock:
            return [(name, self.counts[name], self.nanos[name]) for name in sorted(self.counts)]


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


def toilet_balance_reward(
    toilet_balance_score: torch.Tensor,
    toilet_invalid: torch.Tensor,
    known_invalid: torch.Tensor,
) -> torch.Tensor:
    if known_invalid.ndim == toilet_balance_score.ndim - 1:
        known_invalid = known_invalid.unsqueeze(1)
    valid_probability = torch.sigmoid(-toilet_invalid)
    known_valid_probability = torch.where(
        known_invalid == 0,
        torch.ones_like(valid_probability),
        torch.zeros_like(valid_probability),
    )
    valid_probability = torch.where(
        known_invalid < 0,
        valid_probability,
        known_valid_probability,
    )
    return -toilet_balance_score * valid_probability


def total_proximity_utility(utility: torch.Tensor) -> torch.Tensor:
    utility = utility.to(torch.float32)
    return torch.sum(utility, dim=2)


# preds.door_invalid: [batch_size, max_candidates, num_outputs]
# preds.connection_invalid: [batch_size, max_candidates, num_outputs]
# preds.toilet_invalid: [batch_size, max_candidates]
# preds.phantoon_invalid: [batch_size, max_candidates]
def compute_expected_reward(
    preds,
    outcomes,
    config: GenerateConfig,
):
    def batch_weight(value: float | torch.Tensor) -> float | torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(preds.door_invalid.device).view(-1, 1)
        return value

    door_logprobs = torch.nn.functional.logsigmoid(-preds.door_invalid)
    connection_logprobs = torch.nn.functional.logsigmoid(-preds.connection_invalid)
    toilet_logprobs = torch.nn.functional.logsigmoid(-preds.toilet_invalid)
    phantoon_logprobs = torch.nn.functional.logsigmoid(-preds.phantoon_invalid)
    door_logprobs = outcome_reward(door_logprobs, outcomes.door_invalid)
    connection_logprobs = outcome_reward(connection_logprobs, outcomes.connection_invalid)
    toilet_logprobs = outcome_reward(toilet_logprobs, outcomes.toilet_invalid)
    phantoon_logprobs = outcome_reward(phantoon_logprobs, outcomes.phantoon_invalid)
    balance_scores = balance_reward(
        preds.balance_score,
        preds.door_invalid,
        outcomes.door_invalid,
    )
    toilet_balance_scores = toilet_balance_reward(
        preds.toilet_balance_score,
        preds.toilet_invalid,
        outcomes.toilet_invalid,
    )
    return (
        batch_weight(config.reward_door) * torch.sum(door_logprobs, dim=2)
        + batch_weight(config.reward_connection) * torch.sum(connection_logprobs, dim=2)
        + batch_weight(config.reward_toilet) * toilet_logprobs
        + batch_weight(config.reward_phantoon) * phantoon_logprobs
        + batch_weight(config.reward_balance) * torch.sum(balance_scores, dim=2)
        + batch_weight(config.reward_toilet_balance) * toilet_balance_scores
        - batch_weight(config.reward_frontier) * preds.avg_frontiers.to(torch.float32)
        - batch_weight(config.reward_graph_diameter) * preds.graph_diameter.to(torch.float32)
        + batch_weight(config.reward_save_distance)
        * (
            total_proximity_utility(preds.save_to_room_utility)
            + total_proximity_utility(preds.save_from_room_utility)
        )
        + batch_weight(config.reward_refill_distance)
        * (
            total_proximity_utility(preds.refill_to_room_utility)
            + total_proximity_utility(preds.refill_from_room_utility)
        )
        + (
            batch_weight(config.reward_missing_connect_utility)
            * total_proximity_utility(preds.missing_connect_utility)
        )
    )


def transfer_features(
    features: Features,
    device: torch.device,
    transfer_stream: torch.cuda.Stream | None = None,
) -> Features:
    if transfer_stream is None or device.type != "cuda":
        result = features.to(device)
        result.mark_dynamic()
        return result
    current_stream = torch.cuda.current_stream(device)
    with torch.cuda.device(device), torch.cuda.stream(transfer_stream):
        result = features.to(device, non_blocking=True)
        ready = torch.cuda.Event()
        ready.record(transfer_stream)
    current_stream.wait_event(ready)
    result.mark_dynamic()
    return result


@dataclass
class ProposalCache:
    state: torch.Tensor
    row_start_idx: torch.Tensor
    row_count: torch.Tensor
    action_index: torch.Tensor
    candidate_count: int


@dataclass
class CandidateBatch:
    candidates: Actions
    proposal_frontier_idx: torch.Tensor
    proposal_door_variant_idx: torch.Tensor
    reward_outcomes: StepOutcomes
    post_candidate_outcomes: StepOutcomes
    feature_requirements: FeatureRequirements
    stats: CandidateStats

    def to(self, device: torch.device, non_blocking: bool = False) -> "CandidateBatch":
        return CandidateBatch(
            candidates=self.candidates.to(device, non_blocking=non_blocking),
            proposal_frontier_idx=self.proposal_frontier_idx.to(device, non_blocking=non_blocking),
            proposal_door_variant_idx=self.proposal_door_variant_idx.to(
                device, non_blocking=non_blocking
            ),
            reward_outcomes=self.reward_outcomes.to(device, non_blocking=non_blocking),
            post_candidate_outcomes=self.post_candidate_outcomes.to(
                device, non_blocking=non_blocking
            ),
            feature_requirements=self.feature_requirements,
            stats=self.stats.to(device, non_blocking=non_blocking),
        )


@dataclass
class CandidateScoreSuccess:
    action_index: torch.Tensor
    selected_actions: Actions
    selected_outcomes: StepOutcomes
    selected_proposal_scores: ProposalCache | None
    proposal_frontier_idx: torch.Tensor
    proposal_door_variant_idx: torch.Tensor
    selected_candidate: torch.Tensor
    target_logits: torch.Tensor


@dataclass
class PipelineFailure:
    error: BaseException


type CandidateScoreResult = CandidateScoreSuccess | PipelineFailure


@dataclass
class GenerationGroup:
    env: EnvironmentGroup
    config: GenerateConfig
    step: int
    feature_slot: FeatureSlot
    candidate_slot: CandidateSlot
    balance_preds: BalancePredictions
    previous_lookahead_outcomes: StepOutcomes | None
    previous_proposal_scores: ProposalCache | None
    score_result_queue: Queue[CandidateScoreResult]


@dataclass
class PreparedGenerationStep:
    candidate_batch: CandidateBatch
    features: Features | None


@dataclass
class CandidateScoreRequest:
    group: GenerationGroup
    group_index: int
    prepared_step: PreparedGenerationStep
    shortlist_limited: torch.Tensor


@dataclass
class StagedCandidateScoreRequest:
    request: CandidateScoreRequest
    candidate_batch: CandidateBatch
    features: Features | None
    ready_event: torch.cuda.Event | None

@dataclass
class StopPipeline:
    pass


type CpuReadyMessage = CandidateScoreRequest | StopPipeline
type GpuReadyMessage = StagedCandidateScoreRequest | StopPipeline


@dataclass
class GroupPipelineOutput:
    proposal_frontier_idx: list[torch.Tensor]
    proposal_door_variant_idx: list[torch.Tensor]
    selected_candidate: list[torch.Tensor]
    target_logits: list[torch.Tensor]


@dataclass
class PipelineSharedState:
    profiler: GenerationProfiler
    stat_totals: GenerationStats
    stat_lock: threading.Lock
    gpu_lock: threading.Lock
    cancellation_event: threading.Event
    groups: list[GenerationGroup]


@dataclass
class PipelineThreadFailure:
    error: BaseException


@dataclass
class ProposalInputs:
    features: Features | None
    mask: ProposalCandidateMask


def create_generation_environment_groups(
    config: Config,
    engine: Engine,
    generation_devices: list[torch.device],
) -> list[list[EnvironmentGroup]]:
    num_generation_groups = config.generation.num_devices * config.generation.pipeline_groups
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
                config.generation.candidate_spatial_cell_size,
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


def get_shortlist_candidate_batch(
    group: GenerationGroup,
    sampled_frontier_idx: torch.Tensor,
    sampled_door_variant_idx: torch.Tensor,
) -> CandidateBatch:
    (
        candidates,
        proposal_frontier_idx,
        proposal_door_variant_idx,
        reward_outcomes,
        post_candidate_outcomes,
        feature_requirements,
        stats,
    ) = group.env.extract_candidates_from_proposals(
        group.candidate_slot,
        sampled_frontier_idx,
        sampled_door_variant_idx,
        group.config.recommended_candidates,
    )
    return CandidateBatch(
        candidates=candidates,
        proposal_frontier_idx=proposal_frontier_idx,
        proposal_door_variant_idx=proposal_door_variant_idx,
        reward_outcomes=reward_outcomes,
        post_candidate_outcomes=post_candidate_outcomes,
        feature_requirements=feature_requirements,
        stats=stats,
    )


def unpack_proposal_mask(mask: ProposalCandidateMask, device: torch.device) -> torch.Tensor:
    packed = mask.mask.to(device)
    shifts = torch.arange(8, device=device, dtype=packed.dtype)
    bits = ((packed.unsqueeze(-1) >> shifts) & 1).to(torch.bool).flatten(1)
    return bits[:, : mask.door_variant_count]


def row_scores_for_mask(
    proposal_output: torch.nn.Module,
    proposal_state: torch.Tensor,
    row_snapshot_idx: torch.Tensor,
    row_frontier_idx: torch.Tensor,
    proposal_mask: ProposalCandidateMask,
    device: torch.device,
) -> torch.Tensor:
    proposal_frontier_idx = proposal_mask.proposal_frontier_idx.to(device)
    result = torch.full(
        (proposal_frontier_idx.shape[0], proposal_output.out_features),
        float("-inf"),
        dtype=proposal_output.output_dtype,
        device=device,
    )
    if proposal_state.shape[0] == 0:
        return result
    row_snapshot_idx = row_snapshot_idx.to(device)
    row_frontier_idx = row_frontier_idx.to(device)
    row_valid = (
        (row_snapshot_idx >= 0)
        & (row_snapshot_idx < proposal_frontier_idx.shape[0])
        & (row_frontier_idx == proposal_frontier_idx[row_snapshot_idx])
    )
    if torch.any(row_valid):
        result[row_snapshot_idx[row_valid]] = proposal_output(
            proposal_state[row_valid].to(proposal_output.output_dtype)
        )
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_log_temperature = config.temperature.to(torch.device("cpu")).log().unsqueeze(1)
    candidate_log_temperature = candidate_log_temperature.expand(candidate_shape).contiguous()
    candidate_log_recommended_candidates = torch.full(
        candidate_shape,
        math.log(config.recommended_candidates + 1),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    candidate_generation_variable_floats = (
        config.generation_variable_floats.to(torch.device("cpu"))
        .unsqueeze(1)
        .expand(*candidate_shape, config.generation_variable_floats.shape[1])
        .contiguous()
    )
    return (
        candidate_log_temperature,
        candidate_log_recommended_candidates,
        candidate_generation_variable_floats,
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
    generation_variable_floats = config.generation_variable_floats.to(torch.device("cpu"))
    return log_temperature, log_recommended_candidates, generation_variable_floats


def select_outcomes(outcomes: StepOutcomes, index: torch.Tensor) -> StepOutcomes:
    def gather(values: torch.Tensor) -> torch.Tensor:
        gather_index = index.view(-1, 1, 1).expand(-1, 1, values.shape[2])
        return torch.gather(values, 1, gather_index).squeeze(1)

    def gather_scalar(values: torch.Tensor) -> torch.Tensor:
        return torch.gather(values, 1, index.view(-1, 1)).squeeze(1)

    return StepOutcomes(
        door_invalid=gather(outcomes.door_invalid),
        connection_invalid=gather(outcomes.connection_invalid),
        toilet_invalid=gather_scalar(outcomes.toilet_invalid),
        phantoon_invalid=gather_scalar(outcomes.phantoon_invalid),
        door_match=gather(outcomes.door_match),
    )


def prepare_proposal_inputs(group: GenerationGroup) -> ProposalInputs:
    proposal_mask = group.env.get_proposal_candidate_mask(
        torch.device("cpu"),
    )
    if group.previous_proposal_scores is not None:
        return ProposalInputs(features=None, mask=proposal_mask)
    if group.previous_lookahead_outcomes is None:
        raise ValueError("proposal features require previous lookahead outcomes")
    environment_count = group.config.temperature.shape[0]
    (
        log_temperature,
        log_recommended_candidates,
        generation_variable_floats,
    ) = state_log_inputs(group.config, environment_count)
    return ProposalInputs(
        features=group.env.extract_features(
            group.feature_slot,
            log_temperature,
            group.env.engine.features.temperature,
            log_recommended_candidates,
            group.env.engine.features.recommended_candidates,
            generation_variable_floats,
            group.env.engine.features.generation_variable_floats,
            group.previous_lookahead_outcomes,
            group.env.engine.features.lookahead_outcomes,
        ),
        mask=proposal_mask,
    )


def prepare_candidate_features(
    env: EnvironmentGroup,
    config: GenerateConfig,
    candidate_batch: CandidateBatch,
    feature_slot: FeatureSlot,
) -> PreparedGenerationStep:
    candidates = candidate_batch.candidates
    if candidates.room_idx.shape[1] == 1:
        return PreparedGenerationStep(candidate_batch=candidate_batch, features=None)
    (
        candidate_log_temperature,
        candidate_log_recommended_candidates,
        candidate_generation_variable_floats,
    ) = candidate_log_inputs(
        config,
        candidates.room_idx.shape,
    )
    return PreparedGenerationStep(
        candidate_batch=candidate_batch,
        features=extract_candidate_features(
            env,
            candidates,
            candidate_log_temperature,
            env.engine.features.temperature,
            candidate_log_recommended_candidates,
            env.engine.features.recommended_candidates,
            candidate_generation_variable_floats,
            env.engine.features.generation_variable_floats,
            candidate_batch.post_candidate_outcomes,
            env.engine.features.lookahead_outcomes,
            candidate_batch.feature_requirements,
            feature_slot,
        ),
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
    outcomes: StepOutcomes,
    post_candidate_door_match: torch.Tensor,
    features: Features,
    device: torch.device,
    num_rooms: int,
    profiler: GenerationProfiler,
) -> tuple[torch.Tensor, Actions, torch.Tensor, torch.Tensor | None]:
    environment_count, candidate_count = candidates.room_idx.shape
    profile = profiler.enabled
    sync_profile_device(device, profile)
    profile_time = profile_start(profile)
    with torch.amp.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and group.config.autocast,
    ):
        return_proposal_state = group.config.recommended_candidates > 0
        preds = model(
            features,
            return_proposal_state=return_proposal_state,
        )
    sync_profile_device(device, profile)
    profiler.add("python.score.model_forward", profile_time)

    profile_time = profile_start(profile)
    balance_score = preds.balance_score.view(environment_count, candidate_count, -1)
    actual_balance_score, actual_balance_score_mask = (
        compute_step_balance_score_target_logits(
            group.balance_preds,
            post_candidate_door_match,
        )
    )
    balance_score = torch.where(
        actual_balance_score_mask,
        actual_balance_score,
        balance_score,
    )
    expected_reward = compute_expected_reward(
        Predictions(
            door_invalid=preds.door_invalid.view(environment_count, candidate_count, -1),
            connection_invalid=preds.connection_invalid.view(
                environment_count,
                candidate_count,
                -1,
            ),
            toilet_invalid=preds.toilet_invalid.view(environment_count, candidate_count),
            phantoon_invalid=preds.phantoon_invalid.view(environment_count, candidate_count),
            balance_score=balance_score,
            toilet_balance_score=preds.toilet_balance_score.view(
                environment_count,
                candidate_count,
            ),
            avg_frontiers=preds.avg_frontiers.view(environment_count, candidate_count),
            graph_diameter=preds.graph_diameter.view(environment_count, candidate_count),
            save_to_room_utility=preds.save_to_room_utility.view(
                environment_count,
                candidate_count,
                -1,
            ),
            save_from_room_utility=preds.save_from_room_utility.view(
                environment_count,
                candidate_count,
                -1,
            ),
            refill_to_room_utility=preds.refill_to_room_utility.view(
                environment_count,
                candidate_count,
                -1,
            ),
            refill_from_room_utility=preds.refill_from_room_utility.view(
                environment_count,
                candidate_count,
                -1,
            ),
            missing_connect_utility=preds.missing_connect_utility.view(
                environment_count,
                candidate_count,
                -1,
            ),
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
    dummy_candidate = candidates.room_idx == num_rooms
    candidate_logits = expected_reward / torch.unsqueeze(group.config.temperature, 1)
    candidate_logits = torch.where(
        dummy_candidate,
        torch.full_like(candidate_logits, float("-inf")),
        candidate_logits,
    )
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
    if return_proposal_state:
        proposal_row_snapshot_idx = preds.proposal_row_snapshot_idx
        row_count_by_snapshot = torch.bincount(
            proposal_row_snapshot_idx,
            minlength=environment_count * candidate_count,
        )
        row_start_idx = row_count_by_snapshot.cumsum(0) - row_count_by_snapshot
        selected_proposal_scores = ProposalCache(
            state=preds.proposal_state,
            row_start_idx=row_start_idx,
            row_count=row_count_by_snapshot,
            action_index=action_index,
            candidate_count=candidate_count,
        )
        sync_profile_device(device, profile)
    profiler.add("python.score.cache_proposal", profile_time)
    return action_index, selected_actions, candidate_logits, selected_proposal_scores


def compute_proposal_scores(
    group: GenerationGroup,
    model,
    features: Features,
    proposal_mask: ProposalCandidateMask,
    device: torch.device,
    transfer_stream: torch.cuda.Stream | None,
) -> torch.Tensor:
    env_features = transfer_features(features, device, transfer_stream)
    with torch.amp.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and group.config.autocast,
    ):
        preds = model(
            env_features,
            return_proposal_state=True,
        )
    return row_scores_for_mask(
        model.proposal_output,
        preds.proposal_state,
        preds.proposal_row_snapshot_idx,
        preds.proposal_row_frontier_idx,
        proposal_mask,
        device,
    )


def compute_cached_proposal_scores(
    group: GenerationGroup,
    model,
    cache: ProposalCache,
    proposal_mask: ProposalCandidateMask,
    device: torch.device,
) -> torch.Tensor:
    proposal_frontier_idx = proposal_mask.proposal_frontier_idx.to(
        device=device,
        dtype=torch.int64,
    )
    door_variant_count = model.proposal_output.out_features
    if cache.state.shape[0] == 0:
        return cache.state.new_zeros(
            (proposal_frontier_idx.shape[0], door_variant_count),
        )
    row_start_idx = cache.row_start_idx.to(device)
    row_count = cache.row_count.to(device)
    action_index = cache.action_index.to(device)
    selected_snapshot_idx = (
        torch.arange(proposal_frontier_idx.shape[0], device=device) * cache.candidate_count
        + action_index
    )
    row_start_idx = row_start_idx[selected_snapshot_idx]
    row_count = row_count[selected_snapshot_idx]
    safe_frontier_idx = torch.minimum(
        proposal_frontier_idx.clamp_min(0),
        row_count.clamp_min(1) - 1,
    )
    row_idx = (row_start_idx + safe_frontier_idx).clamp_max(cache.state.shape[0] - 1)
    with torch.amp.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and group.config.autocast,
    ):
        return model.proposal_output(cache.state[row_idx])


def verify_and_step(
    group: GenerationGroup,
    selected_actions: Actions,
    device: torch.device,
    verify_outcome_consistency: bool,
) -> None:
    group.env.step(selected_actions)
    if verify_outcome_consistency:
        group.env.get_outcomes(device, verify_consistency=True)


def put_queue_until_done(
    queue: Queue,
    item,
    cancellation_event: threading.Event,
) -> bool:
    while not cancellation_event.is_set():
        try:
            queue.put(item, timeout=0.05)
            return True
        except Full:
            continue
    return False


def get_queue_until_done(
    queue: Queue,
    cancellation_event: threading.Event,
):
    while not cancellation_event.is_set():
        try:
            return queue.get(timeout=0.05)
        except Empty:
            continue
    return PipelineFailure(RuntimeError("generation pipeline cancelled"))


def broadcast_pipeline_failure(
    groups: list[GenerationGroup],
    error: BaseException,
) -> None:
    failure = PipelineFailure(error)
    for group in groups:
        try:
            group.score_result_queue.put_nowait(failure)
        except Full:
            pass


def stage_candidate_score_request(
    request: CandidateScoreRequest,
    device: torch.device,
    transfer_stream: torch.cuda.Stream | None,
    profiler: GenerationProfiler,
) -> StagedCandidateScoreRequest:
    profile = profiler.enabled
    profile_time = profile_start(profile)
    if transfer_stream is None or device.type != "cuda":
        candidate_batch = request.prepared_step.candidate_batch.to(device)
        features = (
            None
            if request.prepared_step.features is None
            else request.prepared_step.features.to(device)
        )
        if features is not None:
            features.mark_dynamic()
        profiler.add("python.transfer.stage_candidate", profile_time)
        return StagedCandidateScoreRequest(
            request=request,
            candidate_batch=candidate_batch,
            features=features,
            ready_event=None,
        )
    with torch.cuda.device(device), torch.cuda.stream(transfer_stream):
        candidate_batch = request.prepared_step.candidate_batch.to(
            device,
            non_blocking=True,
        )
        features = (
            None
            if request.prepared_step.features is None
            else request.prepared_step.features.to(device, non_blocking=True)
        )
        if features is not None:
            features.mark_dynamic()
        ready_event = torch.cuda.Event()
        ready_event.record(transfer_stream)
    profiler.add("python.transfer.stage_candidate", profile_time)
    return StagedCandidateScoreRequest(
        request=request,
        candidate_batch=candidate_batch,
        features=features,
        ready_event=ready_event,
    )


def wait_for_staged_candidate(
    staged: StagedCandidateScoreRequest,
    device: torch.device,
) -> None:
    if staged.ready_event is None or device.type != "cuda":
        return
    torch.cuda.current_stream(device).wait_event(staged.ready_event)


def score_staged_candidate_request(
    staged: StagedCandidateScoreRequest,
    model,
    device: torch.device,
    num_rooms: int,
    profiler: GenerationProfiler,
) -> CandidateScoreSuccess:
    wait_for_staged_candidate(staged, device)
    group = staged.request.group
    candidate_batch = staged.candidate_batch
    candidates = candidate_batch.candidates
    if staged.features is None:
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
            group,
            model,
            candidates,
            candidate_batch.reward_outcomes,
            candidate_batch.post_candidate_outcomes.door_match,
            staged.features,
            device,
            num_rooms,
            profiler,
        )
    profile = profiler.enabled
    profile_time = profile_start(profile)
    max_candidates = group.config.recommended_candidates
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
        door_variant_idx[:, : candidate_batch.proposal_door_variant_idx.shape[1]] = (
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
        target_logits[:, : candidate_logits.shape[1]] = candidate_logits.to(torch.float32)
    selected_outcomes = select_outcomes(
        candidate_batch.post_candidate_outcomes,
        action_index,
    )
    sync_profile_device(device, profile)
    result = CandidateScoreSuccess(
        action_index=action_index.to(device="cpu", copy=True),
        selected_actions=selected_actions.to(torch.device("cpu")),
        selected_outcomes=selected_outcomes.to(torch.device("cpu")),
        selected_proposal_scores=selected_proposal_scores,
        proposal_frontier_idx=frontier_idx.to(device="cpu", copy=True),
        proposal_door_variant_idx=door_variant_idx.to(device="cpu", copy=True),
        selected_candidate=action_index.to(device="cpu", copy=True),
        target_logits=target_logits.to(device="cpu", copy=True),
    )
    profiler.add("python.record_proposal_data", profile_time)
    return result


def run_transfer_coordinator(
    cpu_ready_queue: Queue[CpuReadyMessage],
    gpu_ready_queue: Queue[GpuReadyMessage],
    device: torch.device,
    transfer_stream: torch.cuda.Stream | None,
    shared: PipelineSharedState,
) -> None:
    try:
        while not shared.cancellation_event.is_set():
            message = get_queue_until_done(cpu_ready_queue, shared.cancellation_event)
            if isinstance(message, PipelineFailure):
                return
            if isinstance(message, StopPipeline):
                put_queue_until_done(gpu_ready_queue, message, shared.cancellation_event)
                return
            staged = stage_candidate_score_request(
                message,
                device,
                transfer_stream,
                shared.profiler,
            )
            if not put_queue_until_done(gpu_ready_queue, staged, shared.cancellation_event):
                return
    except BaseException as error:
        shared.cancellation_event.set()
        broadcast_pipeline_failure(shared.groups, error)
        raise


def run_gpu_scorer(
    gpu_ready_queue: Queue[GpuReadyMessage],
    model,
    device: torch.device,
    num_rooms: int,
    shared: PipelineSharedState,
) -> None:
    try:
        if device.type == "cuda":
            torch.cuda.set_device(device)
        with torch.no_grad():
            while not shared.cancellation_event.is_set():
                message = get_queue_until_done(gpu_ready_queue, shared.cancellation_event)
                if isinstance(message, PipelineFailure):
                    return
                if isinstance(message, StopPipeline):
                    return
                profile_time = profile_start(shared.profiler.enabled)
                with shared.gpu_lock:
                    result = score_staged_candidate_request(
                        message,
                        model,
                        device,
                        num_rooms,
                        shared.profiler,
                    )
                shared.profiler.add("python.score.total_candidate", profile_time)
                put_queue_until_done(
                    message.request.group.score_result_queue,
                    result,
                    shared.cancellation_event,
                )
    except BaseException as error:
        shared.cancellation_event.set()
        broadcast_pipeline_failure(shared.groups, error)
        raise


def run_direct_gpu_scorer(
    cpu_ready_queue: Queue[CpuReadyMessage],
    model,
    device: torch.device,
    num_rooms: int,
    shared: PipelineSharedState,
) -> None:
    try:
        if device.type == "cuda":
            torch.cuda.set_device(device)
        with torch.no_grad():
            while not shared.cancellation_event.is_set():
                message = get_queue_until_done(cpu_ready_queue, shared.cancellation_event)
                if isinstance(message, PipelineFailure):
                    return
                if isinstance(message, StopPipeline):
                    return
                with shared.gpu_lock:
                    staged = stage_candidate_score_request(
                        message,
                        device,
                        None,
                        shared.profiler,
                    )
                    result = score_staged_candidate_request(
                        staged,
                        model,
                        device,
                        num_rooms,
                        shared.profiler,
                    )
                put_queue_until_done(
                    message.group.score_result_queue,
                    result,
                    shared.cancellation_event,
                )
    except BaseException as error:
        shared.cancellation_event.set()
        broadcast_pipeline_failure(shared.groups, error)
        raise


def add_stat_totals(
    shared: PipelineSharedState,
    updates: GenerationStats,
) -> None:
    with shared.stat_lock:
        for key, value in updates.items():
            shared.stat_totals[key] += value


def compute_group_proposal_shortlist(
    group: GenerationGroup,
    model,
    device: torch.device,
    shared: PipelineSharedState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    profile = shared.profiler.enabled
    profile_time = profile_start(profile)
    proposal_inputs = prepare_proposal_inputs(group)
    shared.profiler.add("python.wait_proposal_features", profile_time)
    before_shortlist_time = profile_start(profile)
    valid_counts = proposal_inputs.mask.valid_counts
    row_valid_counts = valid_counts
    add_stat_totals(
        shared,
        {
            "proposal_mask_rows": float(row_valid_counts.numel()),
            "proposal_valid_cells": float(row_valid_counts.sum().item()),
            "proposal_full_set_rows": float(
                (row_valid_counts <= group.config.shortlist_candidates).sum().item()
            ),
            "proposal_clean_candidates": 0.0,
            "proposal_evaluated_candidates": 0.0,
            "proposal_rejected_candidates": 0.0,
            "proposal_exhausted_rows": 0.0,
        },
    )
    shortlist_limited = row_valid_counts > group.config.shortlist_candidates
    with shared.gpu_lock:
        if group.previous_proposal_scores is not None:
            profile_time = profile_start(profile)
            proposal_scores = compute_cached_proposal_scores(
                group,
                model,
                group.previous_proposal_scores,
                proposal_inputs.mask,
                device,
            )
            sync_profile_device(device, profile)
            shared.profiler.add("python.proposal.compute_cached_scores", profile_time)
        else:
            if proposal_inputs.features is None:
                raise ValueError("proposal scores require proposal features")
            profile_time = profile_start(profile)
            proposal_scores = compute_proposal_scores(
                group,
                model,
                proposal_inputs.features,
                proposal_inputs.mask,
                device,
                None,
            )
            sync_profile_device(device, profile)
            shared.profiler.add("python.proposal.compute_fresh_scores", profile_time)
        shared.profiler.add("python.proposal.total_before_shortlist", before_shortlist_time)
        profile_time = profile_start(profile)
        (
            sampled_frontier_idx,
            sampled_door_variant_idx,
        ) = sample_proposal_shortlist(
            proposal_scores,
            proposal_inputs.mask,
            group.config,
            device,
        )
        sync_profile_device(device, profile)
        shared.profiler.add("python.proposal.sample_shortlist", profile_time)
    return (
        sampled_frontier_idx.to(torch.device("cpu")),
        sampled_door_variant_idx.to(torch.device("cpu")),
        shortlist_limited.to(torch.device("cpu")),
    )


def record_candidate_stats(
    request: CandidateScoreRequest,
    shared: PipelineSharedState,
) -> None:
    stats = request.prepared_step.candidate_batch.stats
    shortlist_limited = request.shortlist_limited
    clean_candidates = float(stats.clean_counts.sum().item())
    evaluated_candidates = float(stats.evaluated_counts.sum().item())
    rejected_candidates = float(stats.rejected_counts.sum().item())
    exhausted_rows = float(
        (
            (stats.clean_counts < request.group.config.recommended_candidates)
            & shortlist_limited
        )
        .sum()
        .item()
    )
    add_stat_totals(
        shared,
        {
            "proposal_mask_rows": 0.0,
            "proposal_valid_cells": 0.0,
            "proposal_full_set_rows": 0.0,
            "proposal_clean_candidates": clean_candidates,
            "proposal_evaluated_candidates": evaluated_candidates,
            "proposal_rejected_candidates": rejected_candidates,
            "proposal_exhausted_rows": exhausted_rows,
        },
    )


def run_group_producer(
    group: GenerationGroup,
    group_index: int,
    model,
    device: torch.device,
    cpu_ready_queue: Queue[CpuReadyMessage],
    output: GroupPipelineOutput,
    shared: PipelineSharedState,
    verify_outcome_consistency: bool,
) -> None:
    try:
        group.env.clear()
        group.env.step_initial()
        group.step = 1
        group.previous_lookahead_outcomes = bootstrap_lookahead_outcomes(
            group.env.get_outcomes(
                torch.device("cpu"),
                verify_consistency=False,
            ).step_outcomes
        )
        group.previous_proposal_scores = None
        while group.step < group.config.episode_length and not shared.cancellation_event.is_set():
            (
                sampled_frontier_idx,
                sampled_door_variant_idx,
                shortlist_limited,
            ) = compute_group_proposal_shortlist(
                group,
                model,
                device,
                shared,
            )
            profile_time = profile_start(shared.profiler.enabled)
            prepared_step = prepare_shortlist_generation_step(
                group,
                sampled_frontier_idx,
                sampled_door_variant_idx,
            )
            shared.profiler.add("python.wait_candidate_features", profile_time)
            request = CandidateScoreRequest(
                group=group,
                group_index=group_index,
                prepared_step=prepared_step,
                shortlist_limited=shortlist_limited,
            )
            profile_time = profile_start(shared.profiler.enabled)
            if not put_queue_until_done(cpu_ready_queue, request, shared.cancellation_event):
                return
            result = get_queue_until_done(
                group.score_result_queue,
                shared.cancellation_event,
            )
            shared.profiler.add("python.pipeline.wait_score_result", profile_time)
            if isinstance(result, PipelineFailure):
                raise result.error
            record_candidate_stats(request, shared)
            output.proposal_frontier_idx.append(result.proposal_frontier_idx)
            output.proposal_door_variant_idx.append(result.proposal_door_variant_idx)
            output.selected_candidate.append(result.selected_candidate)
            output.target_logits.append(result.target_logits)
            profile_time = profile_start(shared.profiler.enabled)
            group.previous_lookahead_outcomes = result.selected_outcomes
            group.previous_proposal_scores = result.selected_proposal_scores
            shared.profiler.add("python.cache_next_proposal", profile_time)
            profile_time = profile_start(shared.profiler.enabled)
            verify_and_step(
                group,
                result.selected_actions,
                torch.device("cpu"),
                verify_outcome_consistency,
            )
            shared.profiler.add("python.step_environment", profile_time)
            group.step += 1
    except BaseException as error:
        shared.cancellation_event.set()
        broadcast_pipeline_failure(shared.groups, error)
        raise


def merge_generation_results(
    results: list[tuple[EpisodeData, EpisodeOutcomes, DoorMatchCounts, ProposalData]],
) -> tuple[EpisodeData, EpisodeOutcomes, DoorMatchCounts, ProposalData]:
    return (
        EpisodeData(
            actions=Actions(
                room_idx=torch.cat(
                    [episode_data.actions.room_idx for episode_data, _, _, _ in results]
                ),
                room_x=torch.cat(
                    [episode_data.actions.room_x for episode_data, _, _, _ in results]
                ),
                room_y=torch.cat(
                    [episode_data.actions.room_y for episode_data, _, _, _ in results]
                ),
            ),
            temperature=torch.cat([episode_data.temperature for episode_data, _, _, _ in results]),
            recommended_candidates=torch.cat(
                [episode_data.recommended_candidates for episode_data, _, _, _ in results]
            ),
            generation_variable_floats=torch.cat(
                [episode_data.generation_variable_floats for episode_data, _, _, _ in results]
            ),
        ),
        EpisodeOutcomes(
            step_outcomes=StepOutcomes(
                door_invalid=torch.cat(
                    [
                        episode_outcomes.step_outcomes.door_invalid
                        for _, episode_outcomes, _, _ in results
                    ]
                ),
                connection_invalid=torch.cat(
                    [
                        episode_outcomes.step_outcomes.connection_invalid
                        for _, episode_outcomes, _, _ in results
                    ]
                ),
                toilet_invalid=torch.cat(
                    [
                        episode_outcomes.step_outcomes.toilet_invalid
                        for _, episode_outcomes, _, _ in results
                    ]
                ),
                phantoon_invalid=torch.cat(
                    [
                        episode_outcomes.step_outcomes.phantoon_invalid
                        for _, episode_outcomes, _, _ in results
                    ]
                ),
                door_match=torch.cat(
                    [
                        episode_outcomes.step_outcomes.door_match
                        for _, episode_outcomes, _, _ in results
                    ]
                ),
            ),
            end_outcomes=EndOutcomes(
                toilet_crossed_room_idx=torch.cat(
                    [
                        episode_outcomes.end_outcomes.toilet_crossed_room_idx
                        for _, episode_outcomes, _, _ in results
                    ]
                ),
                avg_frontiers=torch.cat(
                    [
                        episode_outcomes.end_outcomes.avg_frontiers
                        for _, episode_outcomes, _, _ in results
                    ]
                ),
                graph_diameter=torch.cat(
                    [
                        episode_outcomes.end_outcomes.graph_diameter
                        for _, episode_outcomes, _, _ in results
                    ]
                ),
                active_room_part_mask=torch.cat(
                    [
                        episode_outcomes.end_outcomes.active_room_part_mask
                        for _, episode_outcomes, _, _ in results
                    ]
                ),
                save_distance=torch.cat(
                    [
                        episode_outcomes.end_outcomes.save_distance
                        for _, episode_outcomes, _, _ in results
                    ]
                ),
                save_distance_mask=torch.cat(
                    [
                        episode_outcomes.end_outcomes.save_distance_mask
                        for _, episode_outcomes, _, _ in results
                    ]
                ),
                save_to_room_distance=torch.cat(
                    [
                        episode_outcomes.end_outcomes.save_to_room_distance
                        for _, episode_outcomes, _, _ in results
                    ]
                ),
                save_to_room_distance_mask=torch.cat(
                    [
                        episode_outcomes.end_outcomes.save_to_room_distance_mask
                        for _, episode_outcomes, _, _ in results
                    ]
                ),
                save_from_room_distance=torch.cat(
                    [
                        episode_outcomes.end_outcomes.save_from_room_distance
                        for _, episode_outcomes, _, _ in results
                    ]
                ),
                save_from_room_distance_mask=torch.cat(
                    [
                        episode_outcomes.end_outcomes.save_from_room_distance_mask
                        for _, episode_outcomes, _, _ in results
                    ]
                ),
                refill_distance=torch.cat(
                    [
                        episode_outcomes.end_outcomes.refill_distance
                        for _, episode_outcomes, _, _ in results
                    ]
                ),
                refill_distance_mask=torch.cat(
                    [
                        episode_outcomes.end_outcomes.refill_distance_mask
                        for _, episode_outcomes, _, _ in results
                    ]
                ),
                refill_to_room_distance=torch.cat(
                    [
                        episode_outcomes.end_outcomes.refill_to_room_distance
                        for _, episode_outcomes, _, _ in results
                    ]
                ),
                refill_to_room_distance_mask=torch.cat(
                    [
                        episode_outcomes.end_outcomes.refill_to_room_distance_mask
                        for _, episode_outcomes, _, _ in results
                    ]
                ),
                refill_from_room_distance=torch.cat(
                    [
                        episode_outcomes.end_outcomes.refill_from_room_distance
                        for _, episode_outcomes, _, _ in results
                    ]
                ),
                refill_from_room_distance_mask=torch.cat(
                    [
                        episode_outcomes.end_outcomes.refill_from_room_distance_mask
                        for _, episode_outcomes, _, _ in results
                    ]
                ),
                missing_connect_distance=torch.cat(
                    [
                        episode_outcomes.end_outcomes.missing_connect_distance
                        for _, episode_outcomes, _, _ in results
                    ]
                ),
                missing_connect_distance_mask=torch.cat(
                    [
                        episode_outcomes.end_outcomes.missing_connect_distance_mask
                        for _, episode_outcomes, _, _ in results
                    ]
                ),
            ),
        ),
        DoorMatchCounts(
            horizontal=torch.sum(
                torch.stack([counts.horizontal for _, _, counts, _ in results]),
                dim=0,
            ),
            vertical=torch.sum(
                torch.stack([counts.vertical for _, _, counts, _ in results]),
                dim=0,
            ),
        ),
        ProposalData(
            frontier_idx=torch.cat([proposal.frontier_idx for _, _, _, proposal in results]),
            door_variant_idx=torch.cat(
                [proposal.door_variant_idx for _, _, _, proposal in results]
            ),
            selected_candidate=torch.cat(
                [proposal.selected_candidate for _, _, _, proposal in results]
            ),
            target_logits=torch.cat([proposal.target_logits for _, _, _, proposal in results]),
        ),
    )


def empty_proposal_data(
    environment_count: int,
    max_candidates: int,
    device: torch.device,
) -> ProposalData:
    return ProposalData(
        frontier_idx=torch.empty((environment_count, 0), dtype=torch.int16, device=device),
        door_variant_idx=torch.empty(
            (environment_count, 0, max_candidates), dtype=torch.int16, device=device
        ),
        selected_candidate=torch.empty((environment_count, 0), dtype=torch.int64, device=device),
        target_logits=torch.empty(
            (environment_count, 0, max_candidates), dtype=torch.float32, device=device
        ),
    )


def bootstrap_lookahead_outcomes(outcomes: StepOutcomes) -> StepOutcomes:
    return StepOutcomes(
        door_invalid=outcomes.door_invalid,
        connection_invalid=outcomes.connection_invalid,
        toilet_invalid=outcomes.toilet_invalid,
        phantoon_invalid=outcomes.phantoon_invalid,
        door_match=torch.full_like(outcomes.door_invalid, -1, dtype=torch.int16),
    )


def run_generation_groups(
    envs: list[EnvironmentGroup],
    model,
    balance_model,
    configs: list[GenerateConfig],
    device: torch.device,
    verify_outcome_consistency: bool = False,
    profile: bool = False,
) -> tuple[
    EpisodeData, EpisodeOutcomes, DoorMatchCounts, ProposalData, GenerationStats, ProfileReport
]:
    if not envs or len(envs) != len(configs):
        raise ValueError("generation groups require one config per environment group")
    profiler = GenerationProfiler(profile)
    transfer_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    num_rooms = len(envs[0].engine.rooms)
    gpu_prefetch_batches = configs[0].gpu_prefetch_batches
    if any(config.gpu_prefetch_batches != gpu_prefetch_batches for config in configs):
        raise ValueError("generation groups require matching gpu_prefetch_batches")
    groups = [
        GenerationGroup(
            env=env,
            config=config,
            step=0,
            feature_slot=FeatureSlot(env, pin_memory=device.type == "cuda"),
            candidate_slot=CandidateSlot(env, pin_memory=device.type == "cuda"),
            balance_preds=balance_model(config.generation_variable_floats),
            previous_lookahead_outcomes=None,
            previous_proposal_scores=None,
            score_result_queue=Queue(maxsize=1),
        )
        for env, config in zip(envs, configs)
    ]
    group_outputs = [
        GroupPipelineOutput(
            proposal_frontier_idx=[],
            proposal_door_variant_idx=[],
            selected_candidate=[],
            target_logits=[],
        )
        for _ in groups
    ]
    stat_totals = {
        "proposal_mask_rows": 0.0,
        "proposal_valid_cells": 0.0,
        "proposal_full_set_rows": 0.0,
        "proposal_clean_candidates": 0.0,
        "proposal_evaluated_candidates": 0.0,
        "proposal_rejected_candidates": 0.0,
        "proposal_exhausted_rows": 0.0,
    }
    cancellation_event = threading.Event()
    shared = PipelineSharedState(
        profiler=profiler,
        stat_totals=stat_totals,
        stat_lock=threading.Lock(),
        gpu_lock=threading.Lock(),
        cancellation_event=cancellation_event,
        groups=groups,
    )
    cpu_ready_queue: Queue[CpuReadyMessage] = Queue(maxsize=len(groups))
    worker_count = len(groups) + 1 + int(gpu_prefetch_batches > 0)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        if gpu_prefetch_batches > 0:
            gpu_ready_queue: Queue[GpuReadyMessage] = Queue(maxsize=gpu_prefetch_batches)
            transfer_future = executor.submit(
                run_transfer_coordinator,
                cpu_ready_queue,
                gpu_ready_queue,
                device,
                transfer_stream,
                shared,
            )
            scorer_future = executor.submit(
                run_gpu_scorer,
                gpu_ready_queue,
                model,
                device,
                num_rooms,
                shared,
            )
        else:
            transfer_future = None
            scorer_future = executor.submit(
                run_direct_gpu_scorer,
                cpu_ready_queue,
                model,
                device,
                num_rooms,
                shared,
            )
        producer_futures = [
            executor.submit(
                run_group_producer,
                group,
                group_index,
                model,
                device,
                cpu_ready_queue,
                group_outputs[group_index],
                shared,
                verify_outcome_consistency,
            )
            for group_index, group in enumerate(groups)
        ]
        producer_error = None
        for future in producer_futures:
            try:
                future.result()
            except BaseException as error:
                producer_error = error
                cancellation_event.set()
                break
        if producer_error is None:
            put_queue_until_done(cpu_ready_queue, StopPipeline(), cancellation_event)
        else:
            broadcast_pipeline_failure(groups, producer_error)
        if transfer_future is not None:
            transfer_future.result()
        scorer_future.result()
        if producer_error is not None:
            raise producer_error
        results = []
        for group_index, group in enumerate(groups):
            profile_time = profile_start(profile)
            group.env.finish()
            actions = group.env.get_actions(device)
            episode_outcomes = group.env.get_outcomes(
                device, verify_consistency=verify_outcome_consistency
            )
            door_match_counts = group.env.get_door_match_counts(device)
            results.append(
                (
                    EpisodeData(
                        actions=actions,
                        temperature=group.config.temperature,
                        recommended_candidates=torch.full_like(
                            group.config.temperature,
                            group.config.recommended_candidates,
                            dtype=torch.float32,
                        ),
                        generation_variable_floats=group.config.generation_variable_floats,
                    ),
                    episode_outcomes,
                    door_match_counts,
                    (
                        ProposalData(
                            frontier_idx=torch.stack(
                                group_outputs[group_index].proposal_frontier_idx, dim=1
                            ),
                            door_variant_idx=torch.stack(
                                group_outputs[group_index].proposal_door_variant_idx,
                                dim=1,
                            ),
                            selected_candidate=torch.stack(
                                group_outputs[group_index].selected_candidate, dim=1
                            ),
                            target_logits=torch.stack(
                                group_outputs[group_index].target_logits, dim=1
                            ),
                        )
                        if group_outputs[group_index].proposal_frontier_idx
                        else empty_proposal_data(
                            group.config.temperature.shape[0],
                            group.config.recommended_candidates,
                            device,
                        )
                    ),
                )
            )
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
    return (
        episode_data,
        outcomes,
        door_match_counts,
        proposal_data,
        generation_stats,
        profiler.report(),
    )
