from __future__ import annotations

from env import (
    Actions,
    DoorMatchCounts,
    Engine,
    EnvironmentGroup,
    EpisodeData,
    GenerateConfig,
    Outcomes,
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
import torch

from train_config import Config


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
    balance_prob = torch.sigmoid(balance_score)
    return -balance_prob * match_probability


# preds.door_invalid: [batch_size, max_candidates, num_outputs]
# preds.connection_invalid: [batch_size, max_candidates, num_outputs]
def compute_expected_reward(preds, outcomes, config: GenerateConfig):
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
    )


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
    log_temperature: torch.Tensor,
    include_temperature: bool,
    log_action_candidates: torch.Tensor,
    include_action_candidates: bool,
    sparse_frontiers: bool = False,
    feature_slot: PinnedSparseFeatureSlot | None = None,
):
    if sparse_frontiers and feature_slot is not None:
        frontier_count, sparse_row_count, worker_sparse_row_counts = (
            env.env.get_sparse_feature_requirements_after_candidates(
                candidates.room_idx.contiguous().cpu().numpy(),
                candidates.room_x.contiguous().cpu().numpy(),
                candidates.room_y.contiguous().cpu().numpy(),
                0,
            )
        )
        feature_slot.ensure(
            candidates.room_idx.numel(),
            sparse_row_count,
        )
        env.env.get_sparse_features_after_candidates_into(
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
        return feature_slot.features(
            candidates.room_idx.shape[0],
            candidates.room_idx.shape[1],
            log_temperature,
            include_temperature,
            log_action_candidates,
            include_action_candidates,
            sparse_row_count,
            frontier_count,
        ).flatten_candidates()
    if sparse_frontiers:
        return env.get_sparse_features_after_candidates(
            candidates,
            torch.device("cpu"),
            log_temperature,
            include_temperature,
            log_action_candidates,
            include_action_candidates,
            0,
        ).flatten_candidates()
    return env.get_features_after_candidates(
        candidates,
        torch.device("cpu"),
        log_temperature,
        include_temperature,
        log_action_candidates,
        include_action_candidates,
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
    log_action_candidates = features.log_action_candidates.to(
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
        log_action_candidates,
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
        log_action_candidates: torch.Tensor,
        include_action_candidates: bool,
        sparse_row_count: int,
        frontier_count: int,
    ) -> SparseFeatures:
        snapshot_count = environment_count * candidate_count
        if not include_temperature:
            log_temperature = log_temperature.new_empty(
                [environment_count, candidate_count, 0]
            )
        if not include_action_candidates:
            log_action_candidates = log_action_candidates.new_empty(
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
            log_action_candidates,
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
    known_outcomes: Outcomes | None
    step: int
    feature_slot: PinnedSparseFeatureSlot | None


@dataclass
class PendingGenerationStep:
    group: GenerationGroup
    candidates: Actions
    outcomes: Outcomes
    future: Future[Features | SparseFeatures]


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


def get_generation_candidates(
    group: GenerationGroup,
    device: torch.device,
) -> tuple[Actions, Outcomes]:
    if group.config.lookahead_outcomes:
        return group.env.get_candidates_with_outcomes(
            group.config.max_candidates, device
        )
    candidates = group.env.get_candidates(group.config.max_candidates, device)
    return candidates, group.env.get_outcomes(device)


def select_candidate_actions(
    group: GenerationGroup,
    model,
    candidates: Actions,
    outcomes: Outcomes,
    features: Features | SparseFeatures,
    device: torch.device,
    gpu_lock: threading.Lock,
    transfer_stream: torch.cuda.Stream | None,
    num_rooms: int,
) -> tuple[torch.Tensor, Actions]:
    environment_count, candidate_count = candidates.room_idx.shape
    with gpu_lock:
        env_features = transfer_features(features, device, transfer_stream)
        with torch.amp.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and group.config.autocast,
        ):
            preds = model(env_features)
        expected_reward = compute_expected_reward(
            Predictions(
                preds.door_invalid.view(environment_count, candidate_count, -1),
                preds.connection_invalid.view(environment_count, candidate_count, -1),
                preds.balance_score.view(environment_count, candidate_count, -1),
            ),
            outcomes,
            group.config,
        )
        # Replace dummy candidates to have -inf reward, so they are never selected unless there are no other candidates.
        expected_reward = torch.where(
            candidates.room_idx == num_rooms,
            torch.full_like(expected_reward, float("-inf")),
            expected_reward,
        )
        probs = torch.softmax(
            expected_reward / torch.unsqueeze(group.config.temperature, 1),
            dim=1,
        )
        action_index = rand_choice(probs)
        selected_actions = candidates.select(action_index)
    return action_index, selected_actions


def verify_and_step(
    group: GenerationGroup,
    selected_actions: Actions,
    action_index: torch.Tensor,
    outcomes: Outcomes,
    step: int,
    device: torch.device,
    verify_outcome_consistency: bool,
) -> None:
    if verify_outcome_consistency and group.config.lookahead_outcomes:
        group.known_outcomes = merge_verified_outcomes(
            group.known_outcomes,
            select_outcomes(outcomes, action_index),
            f"lookahead step {step}",
        )
    group.env.step(selected_actions)
    if verify_outcome_consistency:
        group.known_outcomes = merge_verified_outcomes(
            group.known_outcomes,
            group.env.get_outcomes(device),
            f"step {step}",
        )


def start_generation_step(
    group: GenerationGroup,
    device: torch.device,
    sparse_frontiers: bool,
    verify_outcome_consistency: bool,
    executor: ThreadPoolExecutor,
    pending: deque[PendingGenerationStep],
) -> None:
    while group.step < group.config.episode_length:
        candidates, outcomes = get_generation_candidates(group, device)
        if candidates.room_idx.shape[1] != 1:
            candidate_log_temperature = torch.log(group.config.temperature).to(
                torch.device("cpu")
            ).unsqueeze(1).expand(candidates.room_idx.shape).contiguous()
            candidate_log_action_candidates = torch.full(
                candidates.room_idx.shape,
                math.log(group.config.max_candidates),
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
            pending.append(
                PendingGenerationStep(
                    group,
                    candidates,
                    outcomes,
                    executor.submit(
                        extract_candidate_features,
                        group.env,
                        candidates,
                        candidate_log_temperature,
                        group.env.engine.features.temperature,
                        candidate_log_action_candidates,
                        group.env.engine.features.action_candidates,
                        sparse_frontiers,
                        group.feature_slot,
                    ),
                )
            )
            return
        action_index = torch.zeros(
            candidates.room_idx.shape[0],
            dtype=torch.int64,
            device=device,
        )
        verify_and_step(
            group,
            candidates.select(action_index),
            action_index,
            outcomes,
            group.step,
            device,
            verify_outcome_consistency,
        )
        group.step += 1


def merge_generation_results(
    results: list[tuple[EpisodeData, Outcomes, DoorMatchCounts]],
) -> tuple[EpisodeData, Outcomes, DoorMatchCounts]:
    return (
        EpisodeData(
            actions=Actions(
                *(
                    torch.cat([getattr(episode_data.actions, name) for episode_data, _, _ in results])
                    for name in vars(results[0][0].actions)
                )
            ),
            temperature=torch.cat([episode_data.temperature for episode_data, _, _ in results]),
            action_candidates=torch.cat([
                episode_data.action_candidates for episode_data, _, _ in results
            ]),
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


def run_generation_groups(
    envs: list[EnvironmentGroup],
    model,
    configs: list[GenerateConfig],
    device: torch.device,
    verify_outcome_consistency: bool = False,
) -> tuple[EpisodeData, Outcomes, DoorMatchCounts]:
    if not envs or len(envs) != len(configs):
        raise ValueError("generation groups require one config per environment group")
    transfer_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    gpu_lock = threading.Lock()
    num_rooms = len(envs[0].engine.rooms)
    sparse_frontiers = device.type == "cuda"
    groups = [
        GenerationGroup(
            env,
            config,
            None,
            0,
            PinnedSparseFeatureSlot(env, pin_memory=True)
            if device.type == "cuda"
            else None,
        )
        for env, config in zip(envs, configs)
    ]
    with ThreadPoolExecutor(max_workers=len(groups)) as executor:
        pending: deque[PendingGenerationStep] = deque()
        with torch.no_grad():
            for group in groups:
                group.env.clear()
                start_generation_step(
                    group,
                    device,
                    sparse_frontiers,
                    verify_outcome_consistency,
                    executor,
                    pending,
                )
            while pending:
                step = pending.popleft()
                features = step.future.result()
                action_index, selected_actions = select_candidate_actions(
                    step.group,
                    model,
                    step.candidates,
                    step.outcomes,
                    features,
                    device,
                    gpu_lock,
                    transfer_stream,
                    num_rooms,
                )
                verify_and_step(
                    step.group,
                    selected_actions,
                    action_index,
                    step.outcomes,
                    step.group.step,
                    device,
                    verify_outcome_consistency,
                )
                step.group.step += 1
                if step.group.step < step.group.config.episode_length:
                    start_generation_step(
                        step.group,
                        device,
                        sparse_frontiers,
                        verify_outcome_consistency,
                        executor,
                        pending,
                    )
        results = []
        for group in groups:
            group.env.finish()
            actions = group.env.get_actions(device)
            outcomes = group.env.get_outcomes(device)
            door_match_counts = group.env.get_door_match_counts(device)
            if verify_outcome_consistency:
                merge_verified_outcomes(group.known_outcomes, outcomes, "finish")
            results.append((
                EpisodeData(
                    actions,
                    group.config.temperature,
                    torch.full_like(
                        group.config.temperature,
                        group.config.max_candidates,
                        dtype=torch.float32,
                    ),
                ),
                outcomes,
                door_match_counts,
            ))
    return merge_generation_results(results)
