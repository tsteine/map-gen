from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields, is_dataclass
import math
import logging
from typing import Literal

import torch

from env import (
    Actions,
    DoorMatches,
    EpisodeData,
    EpisodeOutcomes,
    Features,
    FeatureSlot,
    StepOutcomes,
    ProposalData,
    GeneratedFeatureData,
    extract_candidate_features,
    slice_features,
)
from experience import ExperienceStorage
from loss import (
    LossConfig,
    compute_balance_loss,
    compute_balance_score_target_logits,
    compute_toilet_balance_score_target_logits,
    compute_loss_breakdown,
)
from train_config import Config, episodes_per_round


def distance_proximity_utility(
    distance: torch.Tensor,
    distance_mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    reachable = distance_mask.to(torch.bool)
    scale_tensor = distance.new_tensor(scale, dtype=torch.float32)
    utility = scale_tensor / (distance.to(torch.float32) + scale_tensor)
    return torch.where(reachable, utility, torch.zeros_like(utility))


@dataclass
class TrainBatchTask:
    kind: Literal["fresh", "replay"]
    start: int | None
    env_index: int


@dataclass
class FeatureTrainBatch:
    features: Features
    proposal_frontier_idx: torch.Tensor | None
    proposal_action_idx: torch.Tensor | None
    proposal_invalid: torch.Tensor | None
    proposal_target_logits: torch.Tensor | None


@dataclass
class PreparedTrainBatch:
    kind: Literal["fresh", "replay"]
    episode_data: EpisodeData
    outcomes: EpisodeOutcomes
    door_matches: DoorMatches
    feature_batches: list[FeatureTrainBatch]
    feature_mismatches: list["FeatureMismatch"]
    feature_compared_tensors: int
    feature_compared_values: int


@dataclass
class FeatureMismatch:
    path: str
    step: int
    generated_shape: tuple[int, ...]
    replayed_shape: tuple[int, ...]
    generated_dtype: torch.dtype
    replayed_dtype: torch.dtype
    compared_values: int
    mismatched_values: int
    example: str


@dataclass
class MainLossBreakdown:
    total: float
    door: float
    connection: float
    toilet: float
    phantoon: float
    balance: float
    toilet_balance: float
    avg_frontiers: float
    graph_diameter: float
    save_distance: float
    refill_distance: float
    missing_connect_utility: float
    area_crossings: float
    area_size: float
    area_map_station: float
    proposal: float
    door_contribution: float
    connection_contribution: float
    toilet_contribution: float
    phantoon_contribution: float
    balance_contribution: float
    toilet_balance_contribution: float
    avg_frontiers_contribution: float
    graph_diameter_contribution: float
    save_distance_contribution: float
    refill_distance_contribution: float
    missing_connect_utility_contribution: float
    area_crossings_contribution: float
    area_size_contribution: float
    area_map_station_contribution: float
    proposal_contribution: float


@dataclass
class CandidateDiagnostics:
    target_entropy: torch.Tensor
    uniform_kl: torch.Tensor
    selected_probability: torch.Tensor


@dataclass
class TrainRoundContext:
    config: Config
    step_config: Config
    device: torch.device
    train_batch_envs: list
    main_model: torch.nn.Module
    balance_model: torch.nn.Module
    main_optimizer: torch.optim.Optimizer
    balance_optimizer: torch.optim.Optimizer
    loss_config: LossConfig
    experience: ExperienceStorage
    train_batch_prefetcher: object
    update_ema_model: Callable[[float], None]
    num_rooms: int
    episode_length: int
    feature_mismatches: list[FeatureMismatch]
    feature_compared_tensors: int
    feature_compared_values: int


def empty_main_loss_breakdown() -> MainLossBreakdown:
    return MainLossBreakdown(
        total=0.0,
        door=0.0,
        connection=0.0,
        toilet=0.0,
        phantoon=0.0,
        balance=0.0,
        toilet_balance=0.0,
        avg_frontiers=0.0,
        graph_diameter=0.0,
        save_distance=0.0,
        refill_distance=0.0,
        missing_connect_utility=0.0,
        area_crossings=0.0,
        area_size=0.0,
        area_map_station=0.0,
        proposal=0.0,
        door_contribution=0.0,
        connection_contribution=0.0,
        toilet_contribution=0.0,
        phantoon_contribution=0.0,
        balance_contribution=0.0,
        toilet_balance_contribution=0.0,
        avg_frontiers_contribution=0.0,
        graph_diameter_contribution=0.0,
        save_distance_contribution=0.0,
        refill_distance_contribution=0.0,
        missing_connect_utility_contribution=0.0,
        area_crossings_contribution=0.0,
        area_size_contribution=0.0,
        area_map_station_contribution=0.0,
        proposal_contribution=0.0,
    )


def accumulate_main_loss(target: MainLossBreakdown, source: MainLossBreakdown) -> None:
    target.total += source.total
    target.door += source.door
    target.connection += source.connection
    target.toilet += source.toilet
    target.phantoon += source.phantoon
    target.balance += source.balance
    target.toilet_balance += source.toilet_balance
    target.avg_frontiers += source.avg_frontiers
    target.graph_diameter += source.graph_diameter
    target.save_distance += source.save_distance
    target.refill_distance += source.refill_distance
    target.missing_connect_utility += source.missing_connect_utility
    target.area_crossings += source.area_crossings
    target.area_size += source.area_size
    target.area_map_station += source.area_map_station
    target.proposal += source.proposal
    target.door_contribution += source.door_contribution
    target.connection_contribution += source.connection_contribution
    target.toilet_contribution += source.toilet_contribution
    target.phantoon_contribution += source.phantoon_contribution
    target.balance_contribution += source.balance_contribution
    target.toilet_balance_contribution += source.toilet_balance_contribution
    target.avg_frontiers_contribution += source.avg_frontiers_contribution
    target.graph_diameter_contribution += source.graph_diameter_contribution
    target.save_distance_contribution += source.save_distance_contribution
    target.refill_distance_contribution += source.refill_distance_contribution
    target.missing_connect_utility_contribution += source.missing_connect_utility_contribution
    target.area_crossings_contribution += source.area_crossings_contribution
    target.area_size_contribution += source.area_size_contribution
    target.area_map_station_contribution += source.area_map_station_contribution
    target.proposal_contribution += source.proposal_contribution


def average_main_loss(total_loss: MainLossBreakdown, count: int) -> MainLossBreakdown:
    return MainLossBreakdown(
        total=total_loss.total / count,
        door=total_loss.door / count,
        connection=total_loss.connection / count,
        toilet=total_loss.toilet / count,
        phantoon=total_loss.phantoon / count,
        balance=total_loss.balance / count,
        toilet_balance=total_loss.toilet_balance / count,
        avg_frontiers=total_loss.avg_frontiers / count,
        graph_diameter=total_loss.graph_diameter / count,
        save_distance=total_loss.save_distance / count,
        refill_distance=total_loss.refill_distance / count,
        missing_connect_utility=total_loss.missing_connect_utility / count,
        area_crossings=total_loss.area_crossings / count,
        area_size=total_loss.area_size / count,
        area_map_station=total_loss.area_map_station / count,
        proposal=total_loss.proposal / count,
        door_contribution=total_loss.door_contribution / count,
        connection_contribution=total_loss.connection_contribution / count,
        toilet_contribution=total_loss.toilet_contribution / count,
        phantoon_contribution=total_loss.phantoon_contribution / count,
        balance_contribution=total_loss.balance_contribution / count,
        toilet_balance_contribution=total_loss.toilet_balance_contribution / count,
        avg_frontiers_contribution=total_loss.avg_frontiers_contribution / count,
        graph_diameter_contribution=total_loss.graph_diameter_contribution / count,
        save_distance_contribution=total_loss.save_distance_contribution / count,
        refill_distance_contribution=total_loss.refill_distance_contribution / count,
        missing_connect_utility_contribution=(
            total_loss.missing_connect_utility_contribution / count
        ),
        area_crossings_contribution=total_loss.area_crossings_contribution / count,
        area_size_contribution=total_loss.area_size_contribution / count,
        area_map_station_contribution=total_loss.area_map_station_contribution / count,
        proposal_contribution=total_loss.proposal_contribution / count,
    )


def compute_candidate_diagnostics(proposal_data: ProposalData) -> CandidateDiagnostics:
    target_logits = proposal_data.target_logits.to(torch.float32)
    present = (
        (proposal_data.frontier_idx >= 0)
        & (proposal_data.action_idx >= 0)
        & torch.isfinite(target_logits)
    )
    resolved = present & ~proposal_data.invalid
    candidate_count = target_logits.shape[-1]
    flat_logits = target_logits.reshape(-1, candidate_count)
    flat_present = present.reshape(-1, candidate_count)
    flat_resolved = resolved.reshape(-1, candidate_count)
    row_valid = torch.any(flat_resolved, dim=1)
    if not torch.any(row_valid):
        zero = torch.sum(target_logits) * 0.0
        return CandidateDiagnostics(
            target_entropy=zero,
            uniform_kl=zero,
            selected_probability=zero,
        )

    row_logits = torch.where(
        flat_present[row_valid],
        flat_logits[row_valid],
        torch.full_like(flat_logits[row_valid], float("-inf")),
    )
    row_mask = flat_present[row_valid]
    target_log_probs = torch.nn.functional.log_softmax(row_logits, dim=1)
    safe_target_log_probs = torch.where(
        row_mask,
        target_log_probs,
        torch.zeros_like(target_log_probs),
    )
    target_probs = torch.where(
        row_mask,
        torch.exp(target_log_probs),
        torch.zeros_like(target_log_probs),
    )
    entropy_per_row = torch.sum(-target_probs * safe_target_log_probs, dim=1)
    target_entropy = torch.mean(entropy_per_row)
    valid_counts = torch.sum(row_mask, dim=1).to(torch.float32)
    uniform_kl = torch.mean(torch.log(valid_counts) - entropy_per_row)

    selected_candidate = proposal_data.selected_candidate.reshape(-1)[row_valid].to(torch.int64)
    selected_in_range = (selected_candidate >= 0) & (selected_candidate < candidate_count)
    safe_selected_candidate = selected_candidate.clamp_min(0).clamp_max(candidate_count - 1)
    selected_valid = selected_in_range & torch.gather(
        row_mask,
        1,
        safe_selected_candidate.unsqueeze(1),
    ).squeeze(1)
    if torch.any(selected_valid):
        selected_probability = torch.mean(
            torch.gather(
                target_probs[selected_valid],
                1,
                selected_candidate[selected_valid].unsqueeze(1),
            ).squeeze(1)
        )
    else:
        selected_probability = torch.sum(target_logits) * 0.0
    return CandidateDiagnostics(
        target_entropy=target_entropy,
        uniform_kl=uniform_kl,
        selected_probability=selected_probability,
    )


def iter_train_batch_tasks(config: Config, experience: ExperienceStorage) -> list[TrainBatchTask]:
    tasks = []
    task_idx = 0
    round_episodes = episodes_per_round(config)
    fresh_batches = int(
        math.ceil(round_episodes * config.train.fresh_pass_factor / config.train.batch_size)
    )
    for batch_idx in range(fresh_batches):
        start = (batch_idx * config.train.batch_size) % round_episodes
        tasks.append(
            TrainBatchTask(
                kind="fresh",
                start=start,
                env_index=task_idx % config.train.pipeline_groups,
            )
        )
        task_idx += 1
    if experience.num_files > 0:
        replay_batches = int(
            math.ceil(round_episodes * config.train.replay_pass_factor / config.train.batch_size)
        )
        for _ in range(replay_batches):
            tasks.append(
                TrainBatchTask(
                    kind="replay",
                    start=None,
                    env_index=task_idx % config.train.pipeline_groups,
                )
            )
            task_idx += 1
    return tasks


def verify_feature_tensor(
    generated: torch.Tensor,
    replayed: torch.Tensor,
    path: str,
    step: int,
) -> FeatureMismatch | None:
    if generated.shape != replayed.shape or generated.dtype != replayed.dtype:
        return FeatureMismatch(
            path=path,
            step=step,
            generated_shape=tuple(generated.shape),
            replayed_shape=tuple(replayed.shape),
            generated_dtype=generated.dtype,
            replayed_dtype=replayed.dtype,
            compared_values=0,
            mismatched_values=0,
            example="shape or dtype mismatch",
        )
    different = generated != replayed
    mismatch_count = int(torch.count_nonzero(different).item())
    if mismatch_count == 0:
        return None
    first = torch.nonzero(different, as_tuple=False)[0]
    index = tuple(first.tolist())
    return FeatureMismatch(
        path=path,
        step=step,
        generated_shape=tuple(generated.shape),
        replayed_shape=tuple(replayed.shape),
        generated_dtype=generated.dtype,
        replayed_dtype=replayed.dtype,
        compared_values=generated.numel(),
        mismatched_values=mismatch_count,
        example=(
            f"index={index} generated={generated[index].item()!r} "
            f"replayed={replayed[index].item()!r}"
        ),
    )


def collect_feature_mismatches(
    generated, replayed, step: int, path: str = "features"
) -> tuple[list[FeatureMismatch], int, int]:
    if type(generated) is not type(replayed) or not is_dataclass(generated):
        raise TypeError(f"cannot compare feature values at {path}")
    mismatches = []
    compared_tensors = 0
    compared_values = 0
    for field in fields(generated):
        generated_value = getattr(generated, field.name)
        replayed_value = getattr(replayed, field.name)
        field_path = f"{path}.{field.name}"
        if isinstance(generated_value, torch.Tensor):
            compared_tensors += 1
            if generated_value.shape == replayed_value.shape:
                compared_values += generated_value.numel()
            mismatch = verify_feature_tensor(generated_value, replayed_value, field_path, step)
            if mismatch is not None:
                mismatches.append(mismatch)
        else:
            child_mismatches, child_tensors, child_values = collect_feature_mismatches(
                generated_value, replayed_value, step, field_path
            )
            mismatches.extend(child_mismatches)
            compared_tensors += child_tensors
            compared_values += child_values
    return mismatches, compared_tensors, compared_values


def collect_selected_action_feature_mismatches(
    features: Features,
    actions: Actions,
    step: int,
) -> tuple[list[FeatureMismatch], int, int]:
    room_idx = actions.room_idx[:, step].to(device="cpu", dtype=torch.int64)
    num_rooms = features.global_features.room_placed.shape[1]
    valid = (room_idx >= 0) & (room_idx < num_rooms)
    if not torch.any(valid):
        return [], 0, 0
    environment_idx = torch.arange(room_idx.shape[0], dtype=torch.int64)[valid]
    selected_room_idx = room_idx[valid]
    generated_values = (
        features.global_features.room_placed[environment_idx, selected_room_idx],
        features.global_features.room_x[environment_idx, selected_room_idx],
        features.global_features.room_y[environment_idx, selected_room_idx],
    )
    expected_values = (
        torch.ones_like(generated_values[0]),
        actions.room_x[:, step].to(torch.device("cpu"))[valid],
        actions.room_y[:, step].to(torch.device("cpu"))[valid],
    )
    names = ("room_placed", "room_x", "room_y")
    mismatches = []
    compared_values = 0
    for name, generated, expected in zip(names, generated_values, expected_values):
        compared_values += generated.numel()
        mismatch = verify_feature_tensor(
            generated,
            expected,
            f"capture.selected_action.{name}",
            step,
        )
        if mismatch is not None:
            mismatches.append(mismatch)
    return mismatches, len(names), compared_values


def collect_action_prefix_feature_mismatches(
    features: Features,
    actions: Actions,
    step: int,
) -> tuple[list[FeatureMismatch], int, int]:
    current_room_idx = actions.room_idx[:, step].to(device="cpu", dtype=torch.int64)
    current_real = (
        current_room_idx >= 0
    ) & (current_room_idx < features.global_features.room_placed.shape[1])
    if not torch.any(current_real):
        return [], 0, 0
    generated_values = (
        features.global_features.room_placed[current_real],
        features.global_features.room_x[current_real],
        features.global_features.room_y[current_real],
    )
    expected_placed = torch.zeros_like(generated_values[0])
    expected_x = torch.zeros_like(generated_values[1])
    expected_y = torch.zeros_like(generated_values[2])
    num_environments, num_rooms = expected_placed.shape
    environment_idx = torch.arange(num_environments, dtype=torch.int64)
    real_action_room_idx = actions.room_idx[current_real].to(device="cpu", dtype=torch.int64)
    real_action_room_x = actions.room_x[current_real].to(device="cpu")
    real_action_room_y = actions.room_y[current_real].to(device="cpu")
    for prefix_step in range(step + 1):
        room_idx = real_action_room_idx[:, prefix_step]
        valid = (room_idx >= 0) & (room_idx < num_rooms)
        valid_environment_idx = environment_idx[valid]
        valid_room_idx = room_idx[valid]
        expected_placed[valid_environment_idx, valid_room_idx] = 1
        expected_x[valid_environment_idx, valid_room_idx] = real_action_room_x[valid, prefix_step]
        expected_y[valid_environment_idx, valid_room_idx] = real_action_room_y[valid, prefix_step]
    expected_x = torch.where(expected_placed != 0, expected_x, generated_values[1])
    expected_y = torch.where(expected_placed != 0, expected_y, generated_values[2])
    expected_values = (expected_placed, expected_x, expected_y)
    names = ("room_placed", "room_x", "room_y")
    mismatches = []
    compared_values = 0
    for name, generated, expected in zip(names, generated_values, expected_values):
        compared_values += generated.numel()
        mismatch = verify_feature_tensor(
            generated,
            expected,
            f"capture.action_prefix.{name}",
            step,
        )
        if mismatch is not None:
            mismatches.append(mismatch)
    return mismatches, len(names), compared_values


def collect_generated_feature_continuity_mismatches(
    previous_features: Features,
    current_features: Features,
    actions: Actions,
    step: int,
) -> tuple[list[FeatureMismatch], int, int]:
    current_room_idx = actions.room_idx[:, step].to(device="cpu", dtype=torch.int64)
    current_real = (
        current_room_idx >= 0
    ) & (current_room_idx < current_features.global_features.room_placed.shape[1])
    if not torch.any(current_real):
        return [], 0, 0
    previous_values = (
        previous_features.global_features.room_placed[current_real],
        previous_features.global_features.room_x[current_real],
        previous_features.global_features.room_y[current_real],
    )
    current_values = (
        current_features.global_features.room_placed[current_real],
        current_features.global_features.room_x[current_real],
        current_features.global_features.room_y[current_real],
    )
    expected_values = tuple(value.clone() for value in previous_values)
    room_idx = current_room_idx[current_real]
    num_environments, num_rooms = expected_values[0].shape
    valid = (room_idx >= 0) & (room_idx < num_rooms)
    environment_idx = torch.arange(num_environments, dtype=torch.int64)[valid]
    valid_room_idx = room_idx[valid]
    expected_values[0][environment_idx, valid_room_idx] = 1
    expected_values[1][environment_idx, valid_room_idx] = actions.room_x[current_real][
        valid, step
    ].to(
        torch.device("cpu")
    )
    expected_values[2][environment_idx, valid_room_idx] = actions.room_y[current_real][
        valid, step
    ].to(
        torch.device("cpu")
    )
    expected_values = (
        expected_values[0],
        torch.where(expected_values[0] != 0, expected_values[1], current_values[1]),
        torch.where(expected_values[0] != 0, expected_values[2], current_values[2]),
    )
    names = ("room_placed", "room_x", "room_y")
    mismatches = []
    compared_values = 0
    for name, current, expected in zip(names, current_values, expected_values):
        compared_values += current.numel()
        mismatch = verify_feature_tensor(
            current,
            expected,
            f"capture.continuity.{name}",
            step,
        )
        if mismatch is not None:
            mismatches.append(mismatch)
    return mismatches, len(names), compared_values


def prepare_feature_batches(
    config: Config,
    train_episode_data: EpisodeData,
    proposal_data: ProposalData | None,
    env,
    episode_length: int,
    pin_memory: bool,
    generated_feature_batches: list[Features | None] | None,
) -> tuple[list[FeatureTrainBatch], list[FeatureMismatch], int, int]:
    offset = torch.randint(0, config.train.sample_period, [1]).item()
    train_actions = train_episode_data.actions
    train_actions_cpu = train_actions.to(torch.device("cpu"))
    log_temperature = torch.log(train_episode_data.temperature).to(torch.device("cpu"))
    log_recommended_candidates = torch.log(train_episode_data.recommended_candidates + 1).to(
        torch.device("cpu")
    )
    generation_variable_floats = train_episode_data.generation_variable_floats.to(
        torch.device("cpu")
    )
    env.clear()
    feature_batches = []
    feature_mismatches = []
    feature_compared_tensors = 0
    feature_compared_values = 0
    for step in range(episode_length):
        next_actions = Actions(
            room_idx=train_actions_cpu.room_idx[:, step],
            room_x=train_actions_cpu.room_x[:, step],
            room_y=train_actions_cpu.room_y[:, step],
            room_area=train_actions_cpu.room_area[:, step],
        )
        sample_step = step % config.train.sample_period == offset
        if config.features.lookahead_outcomes:
            env.step(next_actions)
        else:
            env.step_known(next_actions)
        if sample_step:
            if config.features.lookahead_outcomes:
                next_lookahead_outcomes = env.get_current_feature_outcomes(
                    torch.device("cpu"),
                    0,
                    train_actions.room_idx.shape[0],
                )
            else:
                next_lookahead_outcomes = None
            proposal_frontier_idx = None
            proposal_action_idx = None
            proposal_invalid = None
            proposal_target_logits = None
            if proposal_data is not None and step + 1 < episode_length:
                proposal_frontier_idx = proposal_data.frontier_idx[:, step]
                proposal_action_idx = proposal_data.action_idx[:, step]
                proposal_invalid = proposal_data.invalid[:, step]
                proposal_target_logits = proposal_data.target_logits[:, step]
            feature_slot = FeatureSlot(env, pin_memory=pin_memory)
            if generated_feature_batches is not None and next_lookahead_outcomes is not None:
                replay_feature_requirements = env.get_replay_action_feature_requirements(
                    next_actions,
                    0,
                    train_actions.room_idx.shape[0],
                )
                replay_features = extract_candidate_features(
                    env,
                    Actions(
                        room_idx=next_actions.room_idx.unsqueeze(1),
                        room_x=next_actions.room_x.unsqueeze(1),
                        room_y=next_actions.room_y.unsqueeze(1),
                        room_area=next_actions.room_area.unsqueeze(1),
                    ),
                    log_temperature.unsqueeze(1),
                    config.features.temperature,
                    log_recommended_candidates.unsqueeze(1),
                    config.features.recommended_candidates,
                    generation_variable_floats.unsqueeze(1),
                    config.features.generation_variable_floats,
                    StepOutcomes(
                        door_invalid=next_lookahead_outcomes.door_invalid.unsqueeze(1),
                        connection_invalid=(
                            next_lookahead_outcomes.connection_invalid.unsqueeze(1)
                        ),
                        toilet_invalid=next_lookahead_outcomes.toilet_invalid.unsqueeze(1),
                        phantoon_invalid=next_lookahead_outcomes.phantoon_invalid.unsqueeze(1),
                        area_size_bucket=next_lookahead_outcomes.area_size_bucket.unsqueeze(1),
                        area_map_station_count_bucket=(
                            next_lookahead_outcomes.area_map_station_count_bucket.unsqueeze(1)
                        ),
                        door_match=next_lookahead_outcomes.door_match.unsqueeze(1),
                    ),
                    config.features.lookahead_outcomes,
                    replay_feature_requirements,
                    feature_slot,
                )
            else:
                replay_features = env.extract_features(
                        feature_slot,
                        log_temperature,
                        config.features.temperature,
                        log_recommended_candidates,
                        config.features.recommended_candidates,
                        generation_variable_floats,
                        config.features.generation_variable_floats,
                        next_lookahead_outcomes,
                        config.features.lookahead_outcomes,
                        0,
                        train_actions.room_idx.shape[0],
                    )
            if generated_feature_batches is not None:
                generated_features = generated_feature_batches[step]
                if generated_features is not None:
                    capture_mismatches, capture_tensors, capture_values = (
                        collect_selected_action_feature_mismatches(
                            generated_features,
                            train_actions_cpu,
                            step,
                        )
                    )
                    prefix_mismatches, prefix_tensors, prefix_values = (
                        collect_action_prefix_feature_mismatches(
                            generated_features,
                            train_actions_cpu,
                            step,
                        )
                    )
                    continuity_mismatches = []
                    continuity_tensors = 0
                    continuity_values = 0
                    if step > 0:
                        previous_generated_features = generated_feature_batches[step - 1]
                        if previous_generated_features is not None:
                            (
                                continuity_mismatches,
                                continuity_tensors,
                                continuity_values,
                            ) = collect_generated_feature_continuity_mismatches(
                                previous_generated_features,
                                generated_features,
                                train_actions_cpu,
                                step,
                            )
                    mismatches, compared_tensors, compared_values = collect_feature_mismatches(
                        generated_features, replay_features, step
                    )
                    feature_mismatches.extend(capture_mismatches)
                    feature_mismatches.extend(prefix_mismatches)
                    feature_mismatches.extend(continuity_mismatches)
                    feature_mismatches.extend(mismatches)
                    feature_compared_tensors += (
                        capture_tensors
                        + prefix_tensors
                        + continuity_tensors
                        + compared_tensors
                    )
                    feature_compared_values += (
                        capture_values
                        + prefix_values
                        + continuity_values
                        + compared_values
                    )
            feature_batches.append(
                FeatureTrainBatch(
                    features=replay_features,
                    proposal_frontier_idx=proposal_frontier_idx,
                    proposal_action_idx=proposal_action_idx,
                    proposal_invalid=proposal_invalid,
                    proposal_target_logits=proposal_target_logits,
                )
            )
    return (
        feature_batches,
        feature_mismatches,
        feature_compared_tensors,
        feature_compared_values,
    )


def prepare_feature_batch(
    config: Config,
    device: torch.device,
    kind: Literal["fresh", "replay"],
    train_episode_data: EpisodeData,
    train_outcomes: EpisodeOutcomes,
    proposal_data: ProposalData | None,
    env,
    episode_length: int,
    generated_feature_batches: list[Features | None] | None,
) -> PreparedTrainBatch:
    feature_batches, feature_mismatches, compared_tensors, compared_values = (
        prepare_feature_batches(
            config,
            train_episode_data,
            proposal_data,
            env,
            episode_length,
            device.type == "cuda",
            generated_feature_batches,
        )
    )
    door_matches = env.get_door_matches(device)
    return PreparedTrainBatch(
        kind=kind,
        episode_data=train_episode_data,
        outcomes=train_outcomes,
        door_matches=door_matches,
        feature_batches=feature_batches,
        feature_mismatches=feature_mismatches,
        feature_compared_tensors=compared_tensors,
        feature_compared_values=compared_values,
    )


def prepare_train_batch_task(
    context: TrainRoundContext,
    task: TrainBatchTask,
    fresh_episode_data: EpisodeData,
    fresh_outcomes: EpisodeOutcomes,
    fresh_proposal_data: ProposalData,
    generated_feature_data: GeneratedFeatureData,
) -> PreparedTrainBatch:
    env = context.train_batch_envs[task.env_index]
    if task.kind == "fresh":
        if task.start is None:
            raise ValueError("fresh train batch task requires a start index")
        end = task.start + context.config.train.batch_size
        train_episode_data = fresh_episode_data.slice(task.start, end)
        train_outcomes = fresh_outcomes.slice(task.start, end)
        train_proposal_data = fresh_proposal_data.slice(task.start, end)
        return prepare_feature_batch(
            context.config,
            context.device,
            task.kind,
            train_episode_data,
            train_outcomes,
            train_proposal_data,
            env,
            context.episode_length,
            (
                [
                    None if features is None else slice_features(features, task.start, end)
                    for features in generated_feature_data.feature_batches
                ]
                if generated_feature_data.feature_batches
                else None
            ),
        )

    replay_episode_data = context.experience.sample(
        context.config.train.batch_size,
        context.config.train.episodes_per_file,
        context.config.train.hist_c,
    )
    feature_batches, feature_mismatches, compared_tensors, compared_values = (
        prepare_feature_batches(
            context.config,
            replay_episode_data,
            None,
            env,
            context.episode_length,
            context.device.type == "cuda",
            None,
        )
    )
    replay_door_matches = env.get_door_matches(context.device)
    env.finish()
    replay_episode_data = replay_episode_data.to(context.device)
    replay_outcomes = env.get_outcomes(context.device, verify_consistency=False)
    return PreparedTrainBatch(
        kind=task.kind,
        episode_data=replay_episode_data,
        outcomes=replay_outcomes,
        door_matches=replay_door_matches,
        feature_batches=feature_batches,
        feature_mismatches=feature_mismatches,
        feature_compared_tensors=compared_tensors,
        feature_compared_values=compared_values,
    )


def train_balance_batch_backward(
    balance_model: torch.nn.Module,
    prepared_batch: PreparedTrainBatch,
    loss_scale: float,
) -> torch.Tensor:
    preds = balance_model(prepared_batch.episode_data.generation_variable_floats)
    balance_loss = compute_balance_loss(
        preds,
        prepared_batch.door_matches,
        prepared_batch.outcomes.end_outcomes.toilet_crossed_room_idx,
    )
    (balance_loss * loss_scale).backward()
    return balance_loss


def proposal_batch_loss(
    candidate_score: torch.Tensor,
    target_logits: torch.Tensor,
    invalid: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    target_logits = target_logits.to(device, dtype=torch.float32)
    invalid = invalid.to(device=device, dtype=torch.bool)
    present = torch.isfinite(candidate_score) & torch.isfinite(target_logits)
    valid = present & ~invalid
    row_valid = torch.any(valid, dim=1)
    if not torch.any(row_valid):
        return candidate_score.new_zeros(())
    invalid_logit = torch.finfo(candidate_score.dtype).min
    candidate_score = torch.where(
        present,
        candidate_score,
        torch.full_like(candidate_score, invalid_logit),
    ).to(torch.float32)
    target_logits = torch.where(
        present,
        target_logits,
        torch.full_like(target_logits, invalid_logit),
    )
    row_candidate_logits = candidate_score[row_valid]
    row_target_logits = target_logits[row_valid]
    row_mask = valid[row_valid]
    proposal_log_probs = torch.nn.functional.log_softmax(
        row_candidate_logits,
        dim=1,
    )
    target_log_probs = torch.nn.functional.log_softmax(
        row_target_logits,
        dim=1,
    )
    safe_target_log_probs = torch.where(
        row_mask,
        target_log_probs,
        torch.zeros_like(target_log_probs),
    )
    safe_proposal_log_probs = torch.where(
        row_mask,
        proposal_log_probs,
        torch.zeros_like(proposal_log_probs),
    )
    target_probs = torch.where(
        row_mask,
        torch.exp(target_log_probs),
        torch.zeros_like(target_log_probs),
    )
    kl_terms = target_probs * (safe_target_log_probs - safe_proposal_log_probs)
    proposal_loss = (
        torch.sum(torch.where(row_mask, kl_terms, torch.zeros_like(kl_terms))) / row_mask.shape[0]
    )
    return proposal_loss


def proposal_scores_for_candidates(
    proposal_output: torch.nn.Module,
    proposal_state: torch.Tensor,
    row_snapshot_idx: torch.Tensor,
    row_frontier_idx: torch.Tensor,
    frontier_idx: torch.Tensor,
    action_idx: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    frontier_idx = frontier_idx.to(device=device, dtype=torch.int64)
    action_idx = action_idx.to(device=device, dtype=torch.int64)
    result = torch.full(
        frontier_idx.shape,
        float("-inf"),
        dtype=proposal_output.output_dtype,
        device=device,
    )
    if proposal_state.shape[0] == 0:
        return result
    row_snapshot_idx = row_snapshot_idx.to(device=device, dtype=torch.int64)
    row_frontier_idx = row_frontier_idx.to(device=device, dtype=torch.int64)
    snapshot_count = frontier_idx.shape[0]
    row_counts = torch.bincount(row_snapshot_idx, minlength=snapshot_count)
    row_starts = row_counts.cumsum(0) - row_counts
    batch_idx = torch.arange(snapshot_count, device=device, dtype=torch.int64).unsqueeze(1)
    candidate_valid = (
        (frontier_idx >= 0)
        & (frontier_idx < row_counts.unsqueeze(1))
        & (action_idx >= 0)
        & (action_idx < proposal_output.out_features)
    )
    safe_frontier_idx = torch.minimum(
        frontier_idx.clamp_min(0),
        (row_counts.unsqueeze(1) - 1).clamp_min(0),
    )
    candidate_row_idx = row_starts.unsqueeze(1) + safe_frontier_idx
    safe_candidate_row_idx = candidate_row_idx.clamp_max(proposal_state.shape[0] - 1)
    candidate_valid = (
        candidate_valid
        & (row_snapshot_idx[safe_candidate_row_idx] == batch_idx)
        & (row_frontier_idx[safe_candidate_row_idx] == frontier_idx)
    )
    if torch.any(candidate_valid):
        selected_row_idx = safe_candidate_row_idx[candidate_valid]
        unique_row_idx, inverse_idx = torch.unique(selected_row_idx, return_inverse=True)
        row_scores = proposal_output(
            proposal_state[unique_row_idx].to(proposal_output.output_dtype)
        )
        result[candidate_valid] = row_scores[inverse_idx, action_idx[candidate_valid]]
    return result


def train_feature_batch_backward(
    context: TrainRoundContext,
    prepared_batch: PreparedTrainBatch,
    loss_scale: float,
) -> MainLossBreakdown:
    if len(prepared_batch.feature_batches) == 0:
        raise RuntimeError("feature training batch has no sampled prefixes")

    step_outcomes = prepared_batch.outcomes.step_outcomes
    end_outcomes = prepared_batch.outcomes.end_outcomes
    repeated_outcomes = StepOutcomes(
        door_invalid=step_outcomes.door_invalid.unsqueeze(1),
        connection_invalid=step_outcomes.connection_invalid.unsqueeze(1),
        toilet_invalid=step_outcomes.toilet_invalid.unsqueeze(1),
        phantoon_invalid=step_outcomes.phantoon_invalid.unsqueeze(1),
        area_size_bucket=step_outcomes.area_size_bucket.unsqueeze(1),
        area_map_station_count_bucket=step_outcomes.area_map_station_count_bucket.unsqueeze(1),
        door_match=step_outcomes.door_match.unsqueeze(1),
    )
    with torch.no_grad():
        balance_preds = context.balance_model(
            prepared_batch.episode_data.generation_variable_floats
        )
        balance_score_target_logits, balance_score_mask = compute_balance_score_target_logits(
            balance_preds,
            prepared_batch.door_matches,
        )
        toilet_balance_score_target_logits, toilet_balance_score_mask = (
            compute_toilet_balance_score_target_logits(
                balance_preds,
                end_outcomes.toilet_crossed_room_idx,
            )
        )
    repeated_balance_score_target_logits = balance_score_target_logits.unsqueeze(1)
    repeated_toilet_balance_score_target_logits = toilet_balance_score_target_logits.unsqueeze(1)
    repeated_toilet_balance_score_mask = toilet_balance_score_mask.unsqueeze(1)
    batch_size = prepared_batch.episode_data.actions.room_idx.shape[0]
    avg_frontiers_target = end_outcomes.avg_frontiers.to(context.device).unsqueeze(1)
    avg_frontiers_mask = torch.ones(
        [batch_size, 1],
        dtype=torch.bool,
        device=context.device,
    )
    graph_diameter_target = end_outcomes.graph_diameter.to(context.device).unsqueeze(1)
    graph_diameter_mask = torch.ones(
        [batch_size, 1],
        dtype=torch.bool,
        device=context.device,
    )
    active_room_part_mask = end_outcomes.active_room_part_mask.to(
        device=context.device,
        dtype=torch.bool,
    ).unsqueeze(1)
    save_to_room_utility_target = distance_proximity_utility(
        end_outcomes.save_to_room_distance.to(context.device),
        end_outcomes.save_to_room_distance_mask.to(context.device),
        context.loss_config.distance_proximity_scale,
    ).unsqueeze(1)
    save_from_room_utility_target = distance_proximity_utility(
        end_outcomes.save_from_room_distance.to(context.device),
        end_outcomes.save_from_room_distance_mask.to(context.device),
        context.loss_config.distance_proximity_scale,
    ).unsqueeze(1)
    refill_to_room_utility_target = distance_proximity_utility(
        end_outcomes.refill_to_room_distance.to(context.device),
        end_outcomes.refill_to_room_distance_mask.to(context.device),
        context.loss_config.distance_proximity_scale,
    ).unsqueeze(1)
    refill_from_room_utility_target = distance_proximity_utility(
        end_outcomes.refill_from_room_distance.to(context.device),
        end_outcomes.refill_from_room_distance_mask.to(context.device),
        context.loss_config.distance_proximity_scale,
    ).unsqueeze(1)
    missing_connect_utility_target = distance_proximity_utility(
        end_outcomes.missing_connect_distance.to(context.device),
        end_outcomes.missing_connect_distance_mask.to(context.device),
        context.loss_config.distance_proximity_scale,
    ).unsqueeze(1)
    missing_connect_utility_mask = torch.ones_like(
        missing_connect_utility_target,
        dtype=torch.bool,
    )
    area_crossings_target = end_outcomes.area_crossings.to(
        device=context.device,
        dtype=torch.float32,
    ).unsqueeze(1)
    area_size_values = end_outcomes.area_size.to(context.device)
    area_size_target = torch.where(
        area_size_values < context.config.generation.min_area_size,
        torch.zeros_like(area_size_values),
        torch.where(
            area_size_values <= context.config.generation.max_area_size,
            torch.ones_like(area_size_values),
            torch.full_like(area_size_values, 2),
        ),
    ).unsqueeze(1)
    area_map_station_values = end_outcomes.area_map_station_count.to(context.device)
    area_map_station_target = torch.clamp(area_map_station_values, max=2).unsqueeze(1)
    area_mask = torch.ones_like(area_size_target, dtype=torch.bool)
    area_crossings_mask = torch.ones_like(area_crossings_target, dtype=torch.bool)
    mask = torch.ones(
        [batch_size, 1, 1],
        dtype=torch.bool,
        device=context.device,
    )
    total_loss = empty_main_loss_breakdown()
    prefix_weight = 1.0 / len(prepared_batch.feature_batches)

    for feature_batch in prepared_batch.feature_batches:
        features = feature_batch.features.to(context.device)
        features.mark_dynamic()
        return_proposal_state = (
            prepared_batch.kind == "fresh"
            and feature_batch.proposal_frontier_idx is not None
            and feature_batch.proposal_action_idx is not None
            and feature_batch.proposal_invalid is not None
            and feature_batch.proposal_target_logits is not None
        )
        with torch.amp.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=context.device.type == "cuda" and context.config.model.autocast,
        ):
            preds = context.main_model(
                features,
                return_proposal_state=return_proposal_state,
            )
        prefix_balance_score_mask = balance_score_mask
        if features.global_features.lookahead_door_match.shape[-1] > 0:
            prefix_balance_score_mask = balance_score_mask & (
                features.global_features.lookahead_door_match < 0
            )
        prefix_loss = compute_loss_breakdown(
            preds,
            repeated_outcomes,
            mask,
            repeated_balance_score_target_logits,
            prefix_balance_score_mask.unsqueeze(1),
            repeated_toilet_balance_score_target_logits,
            repeated_toilet_balance_score_mask,
            avg_frontiers_target,
            avg_frontiers_mask,
            graph_diameter_target,
            graph_diameter_mask,
            save_to_room_utility_target,
            save_from_room_utility_target,
            active_room_part_mask,
            refill_to_room_utility_target,
            refill_from_room_utility_target,
            active_room_part_mask,
            missing_connect_utility_target,
            missing_connect_utility_mask,
            area_crossings_target,
            area_size_target,
            area_map_station_target,
            area_mask,
            area_crossings_mask,
            context.loss_config,
        )
        backward_loss = prefix_loss.total * prefix_weight
        total_loss.total += prefix_loss.total.item() * prefix_weight
        total_loss.door += prefix_loss.door.item() * prefix_weight
        total_loss.connection += prefix_loss.connection.item() * prefix_weight
        total_loss.toilet += prefix_loss.toilet.item() * prefix_weight
        total_loss.phantoon += prefix_loss.phantoon.item() * prefix_weight
        total_loss.balance += prefix_loss.balance.item() * prefix_weight
        total_loss.toilet_balance += prefix_loss.toilet_balance.item() * prefix_weight
        total_loss.avg_frontiers += prefix_loss.avg_frontiers.item() * prefix_weight
        total_loss.graph_diameter += prefix_loss.graph_diameter.item() * prefix_weight
        total_loss.save_distance += prefix_loss.save_distance.item() * prefix_weight
        total_loss.refill_distance += prefix_loss.refill_distance.item() * prefix_weight
        total_loss.missing_connect_utility += (
            prefix_loss.missing_connect_utility.item() * prefix_weight
        )
        total_loss.area_crossings += prefix_loss.area_crossings.item() * prefix_weight
        total_loss.area_size += prefix_loss.area_size.item() * prefix_weight
        total_loss.area_map_station += prefix_loss.area_map_station.item() * prefix_weight
        total_loss.door_contribution += prefix_loss.door_contribution.item() * prefix_weight
        total_loss.connection_contribution += (
            prefix_loss.connection_contribution.item() * prefix_weight
        )
        total_loss.toilet_contribution += prefix_loss.toilet_contribution.item() * prefix_weight
        total_loss.phantoon_contribution += (
            prefix_loss.phantoon_contribution.item() * prefix_weight
        )
        total_loss.balance_contribution += prefix_loss.balance_contribution.item() * prefix_weight
        total_loss.toilet_balance_contribution += (
            prefix_loss.toilet_balance_contribution.item() * prefix_weight
        )
        total_loss.avg_frontiers_contribution += (
            prefix_loss.avg_frontiers_contribution.item() * prefix_weight
        )
        total_loss.graph_diameter_contribution += (
            prefix_loss.graph_diameter_contribution.item() * prefix_weight
        )
        total_loss.save_distance_contribution += (
            prefix_loss.save_distance_contribution.item() * prefix_weight
        )
        total_loss.refill_distance_contribution += (
            prefix_loss.refill_distance_contribution.item() * prefix_weight
        )
        total_loss.missing_connect_utility_contribution += (
            prefix_loss.missing_connect_utility_contribution.item() * prefix_weight
        )
        total_loss.area_crossings_contribution += (
            prefix_loss.area_crossings_contribution.item() * prefix_weight
        )
        total_loss.area_size_contribution += (
            prefix_loss.area_size_contribution.item() * prefix_weight
        )
        total_loss.area_map_station_contribution += (
            prefix_loss.area_map_station_contribution.item() * prefix_weight
        )
        if return_proposal_state:
            proposal_score = proposal_scores_for_candidates(
                context.main_model.proposal_output,
                preds.proposal_state,
                preds.proposal_row_snapshot_idx,
                preds.proposal_row_frontier_idx,
                feature_batch.proposal_frontier_idx,
                feature_batch.proposal_action_idx,
                context.device,
            )
            batch_proposal_loss = proposal_batch_loss(
                proposal_score,
                feature_batch.proposal_target_logits,
                feature_batch.proposal_invalid,
                context.device,
            )
            weighted_proposal_loss = (
                context.config.train.proposal_weight * batch_proposal_loss * prefix_weight
            )
            backward_loss = backward_loss + weighted_proposal_loss
            total_loss.total += weighted_proposal_loss.item()
            total_loss.proposal += batch_proposal_loss.item() * prefix_weight
            total_loss.proposal_contribution += weighted_proposal_loss.item()
        (backward_loss * loss_scale).backward()
    return total_loss


def train_batch_backward(
    context: TrainRoundContext,
    prepared_batch: PreparedTrainBatch,
    loss_scale: float,
) -> tuple[MainLossBreakdown, float]:
    loss = train_feature_batch_backward(context, prepared_batch, loss_scale)
    balance_loss = train_balance_batch_backward(
        context.balance_model,
        prepared_batch,
        loss_scale,
    )

    if not math.isfinite(loss.total):
        raise RuntimeError(f"non-finite loss before backward: {loss}")
    if not torch.isfinite(balance_loss):
        raise RuntimeError(f"non-finite balance loss before backward: {balance_loss.item()}")

    return loss, balance_loss.item()


def train_optimizer_step(context: TrainRoundContext) -> None:
    grad_norm = torch.nn.utils.clip_grad_norm_(context.main_model.parameters(), max_norm=1.0)
    if not torch.isfinite(grad_norm):
        raise RuntimeError(f"non-finite gradient norm: {grad_norm.item()}")
    balance_grad_norm = torch.nn.utils.clip_grad_norm_(
        context.balance_model.parameters(),
        max_norm=1.0,
    )
    if not torch.isfinite(balance_grad_norm):
        raise RuntimeError(f"non-finite balance gradient norm: {balance_grad_norm.item()}")
    context.main_optimizer.step()
    context.balance_optimizer.step()
    context.update_ema_model(context.step_config.train.ema_decay)


def train_prepared_batch_group(
    context: TrainRoundContext,
    prepared_batches: Iterable[PreparedTrainBatch],
    batch_count: int,
) -> tuple[MainLossBreakdown, float, int]:
    if batch_count <= 0:
        raise ValueError("batch_count must be greater than zero")
    context.main_model.zero_grad()
    context.balance_model.zero_grad()
    loss_scale = 1.0 / batch_count
    group_loss = empty_main_loss_breakdown()
    group_balance_loss = 0.0
    processed_count = 0
    for prepared_batch in prepared_batches:
        processed_count += 1
        context.feature_mismatches.extend(prepared_batch.feature_mismatches)
        context.feature_compared_tensors += prepared_batch.feature_compared_tensors
        context.feature_compared_values += prepared_batch.feature_compared_values
        batch_loss, batch_balance_loss = train_batch_backward(
            context,
            prepared_batch,
            loss_scale,
        )
        accumulate_main_loss(group_loss, batch_loss)
        group_balance_loss += batch_balance_loss
    if processed_count != batch_count:
        raise RuntimeError(f"expected {batch_count} prepared batch(es), got {processed_count}")
    train_optimizer_step(context)
    return group_loss, group_balance_loss, processed_count


def add_completed_batch_group(
    context: TrainRoundContext,
    total_loss: MainLossBreakdown,
    total_balance_loss: float,
    train_batch_count: int,
    prepared_batch_group: Iterable[PreparedTrainBatch],
    group_size: int,
) -> tuple[float, int]:
    group_loss, group_balance_loss, group_count = train_prepared_batch_group(
        context,
        prepared_batch_group,
        group_size,
    )
    accumulate_main_loss(total_loss, group_loss)
    return total_balance_loss + group_balance_loss, train_batch_count + group_count


def pop_random_prepared_batch(buffer: list[PreparedTrainBatch]) -> PreparedTrainBatch:
    index = torch.randint(len(buffer), [1]).item()
    return buffer.pop(index)


def iter_shuffled_prepared_batches(
    prepared_batches,
    buffer_size: int,
):
    if buffer_size <= 0:
        raise ValueError("shuffle buffer size must be greater than zero")
    buffer = []
    for prepared_batch in prepared_batches:
        buffer.append(prepared_batch)
        if len(buffer) >= buffer_size:
            yield pop_random_prepared_batch(buffer)
    while buffer:
        yield pop_random_prepared_batch(buffer)


def train_round(
    context: TrainRoundContext,
    episode_data: EpisodeData,
    episode_outcomes: EpisodeOutcomes,
    proposal_data: ProposalData,
    generated_feature_data: GeneratedFeatureData,
) -> tuple[MainLossBreakdown, float]:
    set_optimizer_lrs(context.main_optimizer, context.step_config.optimizer)
    set_optimizer_lrs(context.balance_optimizer, context.step_config.balance_optimizer)

    total_loss = empty_main_loss_breakdown()
    total_balance_loss = 0.0
    train_batch_count = 0

    train_batch_tasks = iter_train_batch_tasks(context.config, context.experience)
    prepared_batches = iter(
        context.train_batch_prefetcher.map(
            train_batch_tasks,
            lambda task: prepare_train_batch_task(
                context,
                task,
                episode_data,
                episode_outcomes,
                proposal_data,
                generated_feature_data,
            ),
        )
    )
    shuffled_prepared_batches = iter(
        iter_shuffled_prepared_batches(
            prepared_batches,
            context.config.train.shuffle_buffer_batches,
        )
    )
    remaining_batches = len(train_batch_tasks)
    while remaining_batches > 0:
        group_size = min(
            context.config.train.gradient_accumulation_steps,
            remaining_batches,
        )
        total_balance_loss, train_batch_count = add_completed_batch_group(
            context,
            total_loss,
            total_balance_loss,
            train_batch_count,
            (next(shuffled_prepared_batches) for _ in range(group_size)),
            group_size,
        )
        remaining_batches -= group_size

    log_feature_mismatch_summary(context)

    if train_batch_count == 0:
        return empty_main_loss_breakdown(), 0.0
    return (
        average_main_loss(total_loss, train_batch_count),
        total_balance_loss / train_batch_count,
    )


def log_feature_mismatch_summary(context: TrainRoundContext) -> None:
    if context.feature_compared_tensors == 0:
        return
    mismatched_tensors = len(context.feature_mismatches)
    mismatched_values = sum(
        mismatch.mismatched_values for mismatch in context.feature_mismatches
    )
    logging.warning(
        "Generation feature verification: %s/%s tensors and %s/%s values mismatched.",
        mismatched_tensors,
        context.feature_compared_tensors,
        mismatched_values,
        context.feature_compared_values,
    )
    by_path: dict[str, list[FeatureMismatch]] = {}
    for mismatch in context.feature_mismatches:
        by_path.setdefault(mismatch.path, []).append(mismatch)
    ranked_paths = sorted(
        by_path.items(),
        key=lambda item: (
            sum(value.mismatched_values for value in item[1]),
            len(item[1]),
        ),
        reverse=True,
    )
    for path, mismatches in ranked_paths:
        path_mismatched_values = sum(value.mismatched_values for value in mismatches)
        path_compared_values = sum(value.compared_values for value in mismatches)
        first = mismatches[0]
        logging.warning(
            "  %s: %s mismatched tensor(s), %s/%s values; step=%s %s",
            path,
            len(mismatches),
            path_mismatched_values,
            path_compared_values,
            first.step,
            first.example,
        )


def set_optimizer_lrs(optimizer, config) -> None:
    if hasattr(optimizer, "set_lrs"):
        optimizer.set_lrs(config)
    else:
        optimizer.param_groups[0]["lr"] = config.lr
