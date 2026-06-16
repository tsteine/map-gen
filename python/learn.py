from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Literal

import torch

from env import (
    Actions,
    DoorMatches,
    EpisodeData,
    EpisodeOutcomes,
    SparseFeatures,
    SparseFeatureSlot,
    PreliminaryOutcomes,
    ProposalData,
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


@dataclass
class TrainBatchTask:
    kind: Literal["fresh", "replay"]
    start: int | None
    env_index: int


@dataclass
class FeatureTrainBatch:
    features: SparseFeatures
    proposal_frontier_idx: torch.Tensor | None
    proposal_door_variant_idx: torch.Tensor | None
    proposal_selected_candidate: torch.Tensor | None
    proposal_target_logits: torch.Tensor | None


@dataclass
class PreparedTrainBatch:
    kind: Literal["fresh", "replay"]
    episode_data: EpisodeData
    outcomes: PreliminaryOutcomes
    toilet_crossed_room_idx: torch.Tensor
    avg_frontiers: torch.Tensor
    graph_diameter: torch.Tensor
    save_distance: torch.Tensor
    save_distance_mask: torch.Tensor
    refill_distance: torch.Tensor
    refill_distance_mask: torch.Tensor
    missing_connect_distance: torch.Tensor
    missing_connect_distance_mask: torch.Tensor
    door_matches: DoorMatches
    prefix_count: int
    feature_batches: list[FeatureTrainBatch]


@dataclass
class MainLossBreakdown:
    total: float
    door: float
    connection: float
    toilet: float
    balance: float
    toilet_balance: float
    avg_frontiers: float
    graph_diameter: float
    save_distance: float
    refill_distance: float
    missing_connect_distance: float
    proposal: float
    door_contribution: float
    connection_contribution: float
    toilet_contribution: float
    balance_contribution: float
    toilet_balance_contribution: float
    avg_frontiers_contribution: float
    graph_diameter_contribution: float
    save_distance_contribution: float
    refill_distance_contribution: float
    missing_connect_distance_contribution: float
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


def empty_main_loss_breakdown() -> MainLossBreakdown:
    return MainLossBreakdown(
        total=0.0,
        door=0.0,
        connection=0.0,
        toilet=0.0,
        balance=0.0,
        toilet_balance=0.0,
        avg_frontiers=0.0,
        graph_diameter=0.0,
        save_distance=0.0,
        refill_distance=0.0,
        missing_connect_distance=0.0,
        proposal=0.0,
        door_contribution=0.0,
        connection_contribution=0.0,
        toilet_contribution=0.0,
        balance_contribution=0.0,
        toilet_balance_contribution=0.0,
        avg_frontiers_contribution=0.0,
        graph_diameter_contribution=0.0,
        save_distance_contribution=0.0,
        refill_distance_contribution=0.0,
        missing_connect_distance_contribution=0.0,
        proposal_contribution=0.0,
    )


def accumulate_main_loss(target: MainLossBreakdown, source: MainLossBreakdown) -> None:
    target.total += source.total
    target.door += source.door
    target.connection += source.connection
    target.toilet += source.toilet
    target.balance += source.balance
    target.toilet_balance += source.toilet_balance
    target.avg_frontiers += source.avg_frontiers
    target.graph_diameter += source.graph_diameter
    target.save_distance += source.save_distance
    target.refill_distance += source.refill_distance
    target.missing_connect_distance += source.missing_connect_distance
    target.proposal += source.proposal
    target.door_contribution += source.door_contribution
    target.connection_contribution += source.connection_contribution
    target.toilet_contribution += source.toilet_contribution
    target.balance_contribution += source.balance_contribution
    target.toilet_balance_contribution += source.toilet_balance_contribution
    target.avg_frontiers_contribution += source.avg_frontiers_contribution
    target.graph_diameter_contribution += source.graph_diameter_contribution
    target.save_distance_contribution += source.save_distance_contribution
    target.refill_distance_contribution += source.refill_distance_contribution
    target.missing_connect_distance_contribution += (
        source.missing_connect_distance_contribution
    )
    target.proposal_contribution += source.proposal_contribution


def average_main_loss(total_loss: MainLossBreakdown, count: int) -> MainLossBreakdown:
    return MainLossBreakdown(
        total=total_loss.total / count,
        door=total_loss.door / count,
        connection=total_loss.connection / count,
        toilet=total_loss.toilet / count,
        balance=total_loss.balance / count,
        toilet_balance=total_loss.toilet_balance / count,
        avg_frontiers=total_loss.avg_frontiers / count,
        graph_diameter=total_loss.graph_diameter / count,
        save_distance=total_loss.save_distance / count,
        refill_distance=total_loss.refill_distance / count,
        missing_connect_distance=total_loss.missing_connect_distance / count,
        proposal=total_loss.proposal / count,
        door_contribution=total_loss.door_contribution / count,
        connection_contribution=total_loss.connection_contribution / count,
        toilet_contribution=total_loss.toilet_contribution / count,
        balance_contribution=total_loss.balance_contribution / count,
        toilet_balance_contribution=total_loss.toilet_balance_contribution / count,
        avg_frontiers_contribution=total_loss.avg_frontiers_contribution / count,
        graph_diameter_contribution=total_loss.graph_diameter_contribution / count,
        save_distance_contribution=total_loss.save_distance_contribution / count,
        refill_distance_contribution=total_loss.refill_distance_contribution / count,
        missing_connect_distance_contribution=(
            total_loss.missing_connect_distance_contribution / count
        ),
        proposal_contribution=total_loss.proposal_contribution / count,
    )


def compute_candidate_diagnostics(proposal_data: ProposalData) -> CandidateDiagnostics:
    target_logits = proposal_data.target_logits.to(torch.float32)
    frontier_valid = proposal_data.frontier_idx >= 0
    while frontier_valid.ndim < target_logits.ndim:
        frontier_valid = frontier_valid.unsqueeze(-1)
    valid = (
        frontier_valid
        & (proposal_data.door_variant_idx >= 0)
        & torch.isfinite(target_logits)
    )
    candidate_count = target_logits.shape[-1]
    flat_logits = target_logits.reshape(-1, candidate_count)
    flat_valid = valid.reshape(-1, candidate_count)
    row_valid = torch.any(flat_valid, dim=1)
    if not torch.any(row_valid):
        zero = torch.sum(target_logits) * 0.0
        return CandidateDiagnostics(zero, zero, zero)

    row_logits = torch.where(
        flat_valid[row_valid],
        flat_logits[row_valid],
        torch.full_like(flat_logits[row_valid], float("-inf")),
    )
    row_mask = flat_valid[row_valid]
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
    selected_in_range = (
        (selected_candidate >= 0)
        & (selected_candidate < candidate_count)
    )
    safe_selected_candidate = selected_candidate.clamp_min(0).clamp_max(candidate_count - 1)
    selected_valid = selected_in_range & torch.gather(
        row_mask,
        1,
        safe_selected_candidate.unsqueeze(1),
    ).squeeze(1)
    if torch.any(selected_valid):
        selected_probability = torch.mean(torch.gather(
            target_probs[selected_valid],
            1,
            selected_candidate[selected_valid].unsqueeze(1),
        ).squeeze(1))
    else:
        selected_probability = torch.sum(target_logits) * 0.0
    return CandidateDiagnostics(target_entropy, uniform_kl, selected_probability)


def select_batch(
    episode_data: EpisodeData,
    outcomes: PreliminaryOutcomes,
    toilet_crossed_room_idx: torch.Tensor,
    start: int,
    batch_size: int,
) -> tuple[EpisodeData, PreliminaryOutcomes, torch.Tensor]:
    end = start + batch_size
    return (
        episode_data.slice(start, end),
        PreliminaryOutcomes(
            door_invalid=outcomes.door_invalid[start:end],
            connection_invalid=outcomes.connection_invalid[start:end],
            toilet_invalid=outcomes.toilet_invalid[start:end],
            door_match=outcomes.door_match[start:end],
        ),
        toilet_crossed_room_idx[start:end],
    )


def iter_train_batch_tasks(config: Config, experience: ExperienceStorage) -> list[TrainBatchTask]:
    tasks = []
    task_idx = 0
    round_episodes = episodes_per_round(config)
    fresh_batches = int(
        math.ceil(
            round_episodes
            * config.train.fresh_pass_factor
            / config.train.batch_size
        )
    )
    for batch_idx in range(fresh_batches):
        start = (batch_idx * config.train.batch_size) % round_episodes
        tasks.append(TrainBatchTask("fresh", start, task_idx % config.train.pipeline_groups))
        task_idx += 1
    if experience.num_files > 0:
        replay_batches = int(
            math.ceil(
                round_episodes
                * config.train.replay_pass_factor
                / config.train.batch_size
            )
        )
        for _ in range(replay_batches):
            tasks.append(TrainBatchTask("replay", None, task_idx % config.train.pipeline_groups))
            task_idx += 1
    return tasks


def prepare_feature_batches(
    config: Config,
    train_episode_data: EpisodeData,
    proposal_data: ProposalData | None,
    env,
    num_rooms: int,
    episode_length: int,
    pin_memory: bool,
) -> tuple[int, list[FeatureTrainBatch]]:
    offset = torch.randint(0, config.train.sample_period, [1]).item()
    train_actions = train_episode_data.actions
    train_actions_cpu = train_actions.to(torch.device("cpu"))
    log_temperature = torch.log(train_episode_data.temperature).to(torch.device("cpu"))
    log_recommended_candidates = torch.log(train_episode_data.recommended_candidates + 1).to(
        torch.device("cpu")
    )
    env.clear()
    feature_batches = []
    for step in range(episode_length):
        next_actions = Actions(
            train_actions_cpu.room_idx[:, step],
            train_actions_cpu.room_x[:, step],
            train_actions_cpu.room_y[:, step],
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
                dummy_action = next_actions.room_idx >= num_rooms
                next_lookahead_outcomes = PreliminaryOutcomes(
                    torch.where(
                        dummy_action[:, None],
                        torch.full_like(next_lookahead_outcomes.door_invalid, -1),
                        next_lookahead_outcomes.door_invalid,
                    ),
                    torch.where(
                        dummy_action[:, None],
                        torch.full_like(next_lookahead_outcomes.connection_invalid, -1),
                        next_lookahead_outcomes.connection_invalid,
                    ),
                    torch.where(
                        dummy_action,
                        torch.full_like(next_lookahead_outcomes.toilet_invalid, -1),
                        next_lookahead_outcomes.toilet_invalid,
                    ),
                    torch.where(
                        dummy_action[:, None],
                        torch.full_like(next_lookahead_outcomes.door_match, -1),
                        next_lookahead_outcomes.door_match,
                    ),
                )
            else:
                door_count, connection_count = env.engine.get_output_sizes()
                environment_count = train_actions.room_idx.shape[0]
                next_lookahead_outcomes = PreliminaryOutcomes(
                    torch.empty([environment_count, door_count], dtype=torch.int8),
                    torch.empty([environment_count, connection_count], dtype=torch.int8),
                    torch.empty([environment_count], dtype=torch.int8),
                    torch.empty([environment_count, door_count], dtype=torch.int16),
                )
            proposal_frontier_idx = None
            proposal_door_variant_idx = None
            proposal_selected_candidate = None
            proposal_target_logits = None
            if proposal_data is not None and step + 1 < episode_length:
                proposal_frontier_idx = proposal_data.frontier_idx[:, step]
                proposal_door_variant_idx = proposal_data.door_variant_idx[:, step]
                proposal_selected_candidate = proposal_data.selected_candidate[:, step]
                proposal_target_logits = proposal_data.target_logits[:, step]
            feature_slot = SparseFeatureSlot(env, pin_memory=pin_memory)
            feature_batches.append(
                FeatureTrainBatch(
                    env.extract_sparse_features(
                        feature_slot,
                        log_temperature,
                        config.features.temperature,
                        log_recommended_candidates,
                        config.features.recommended_candidates,
                        PreliminaryOutcomes(
                            next_lookahead_outcomes.door_invalid,
                            next_lookahead_outcomes.connection_invalid,
                            next_lookahead_outcomes.toilet_invalid,
                            next_lookahead_outcomes.door_match,
                        ),
                        config.features.lookahead_outcomes,
                        0,
                        train_actions.room_idx.shape[0],
                    ),
                    proposal_frontier_idx,
                    proposal_door_variant_idx,
                    proposal_selected_candidate,
                    proposal_target_logits,
                )
            )
    return len(feature_batches), feature_batches


def prepare_feature_batch(
    config: Config,
    device: torch.device,
    kind: Literal["fresh", "replay"],
    train_episode_data: EpisodeData,
    train_outcomes: PreliminaryOutcomes,
    toilet_crossed_room_idx: torch.Tensor,
    avg_frontiers: torch.Tensor,
    graph_diameter: torch.Tensor,
    save_distance: torch.Tensor,
    save_distance_mask: torch.Tensor,
    refill_distance: torch.Tensor,
    refill_distance_mask: torch.Tensor,
    missing_connect_distance: torch.Tensor,
    missing_connect_distance_mask: torch.Tensor,
    proposal_data: ProposalData | None,
    env,
    num_rooms: int,
    episode_length: int,
) -> PreparedTrainBatch:
    prefix_count, feature_batches = prepare_feature_batches(
        config,
        train_episode_data,
        proposal_data,
        env,
        num_rooms,
        episode_length,
        device.type == "cuda",
    )
    door_matches = env.get_door_matches(device)
    return PreparedTrainBatch(
        kind,
        train_episode_data,
        train_outcomes,
        toilet_crossed_room_idx,
        avg_frontiers,
        graph_diameter,
        save_distance,
        save_distance_mask,
        refill_distance,
        refill_distance_mask,
        missing_connect_distance,
        missing_connect_distance_mask,
        door_matches,
        prefix_count=prefix_count,
        feature_batches=feature_batches,
    )


def prepare_train_batch_task(
    context: TrainRoundContext,
    task: TrainBatchTask,
    fresh_episode_data: EpisodeData,
    fresh_outcomes: EpisodeOutcomes,
    fresh_proposal_data: ProposalData,
) -> PreparedTrainBatch:
    env = context.train_batch_envs[task.env_index]
    if task.kind == "fresh":
        if task.start is None:
            raise ValueError("fresh train batch task requires a start index")
        train_episode_data, train_outcomes, toilet_crossed_room_idx = select_batch(
            fresh_episode_data,
            fresh_outcomes.validity,
            fresh_outcomes.toilet_crossed_room_idx,
            task.start,
            context.config.train.batch_size,
        )
        avg_frontiers = fresh_outcomes.avg_frontiers[
            task.start:task.start + context.config.train.batch_size
        ]
        graph_diameter = fresh_outcomes.graph_diameter[
            task.start:task.start + context.config.train.batch_size
        ]
        save_distance = fresh_outcomes.save_distance[
            task.start:task.start + context.config.train.batch_size
        ]
        save_distance_mask = fresh_outcomes.save_distance_mask[
            task.start:task.start + context.config.train.batch_size
        ]
        refill_distance = fresh_outcomes.refill_distance[
            task.start:task.start + context.config.train.batch_size
        ]
        refill_distance_mask = fresh_outcomes.refill_distance_mask[
            task.start:task.start + context.config.train.batch_size
        ]
        missing_connect_distance = fresh_outcomes.missing_connect_distance[
            task.start:task.start + context.config.train.batch_size
        ]
        missing_connect_distance_mask = fresh_outcomes.missing_connect_distance_mask[
            task.start:task.start + context.config.train.batch_size
        ]
        train_proposal_data = fresh_proposal_data.slice(
            task.start,
            task.start + context.config.train.batch_size,
        )
        return prepare_feature_batch(
            context.config,
            context.device,
            task.kind,
            train_episode_data,
            train_outcomes,
            toilet_crossed_room_idx,
            avg_frontiers,
            graph_diameter,
            save_distance,
            save_distance_mask,
            refill_distance,
            refill_distance_mask,
            missing_connect_distance,
            missing_connect_distance_mask,
            train_proposal_data,
            env,
            context.num_rooms,
            context.episode_length,
        )

    replay_episode_data = context.experience.sample(
        context.config.train.batch_size,
        context.config.train.episodes_per_file,
        context.config.train.hist_c,
    )
    prefix_count, feature_batches = prepare_feature_batches(
        context.config,
        replay_episode_data,
        None,
        env,
        context.num_rooms,
        context.episode_length,
        context.device.type == "cuda",
    )
    replay_door_matches = env.get_door_matches(context.device)
    env.finish()
    replay_episode_data = replay_episode_data.to(context.device)
    replay_outcomes = env.get_outcomes(context.device, verify_consistency=False)
    return PreparedTrainBatch(
        task.kind,
        replay_episode_data,
        replay_outcomes.validity,
        replay_outcomes.toilet_crossed_room_idx,
        replay_outcomes.avg_frontiers,
        replay_outcomes.graph_diameter,
        replay_outcomes.save_distance,
        replay_outcomes.save_distance_mask,
        replay_outcomes.refill_distance,
        replay_outcomes.refill_distance_mask,
        replay_outcomes.missing_connect_distance,
        replay_outcomes.missing_connect_distance_mask,
        replay_door_matches,
        prefix_count=prefix_count,
        feature_batches=feature_batches,
    )


def train_balance_batch_backward(
    balance_model: torch.nn.Module,
    prepared_batch: PreparedTrainBatch,
    loss_scale: float,
) -> torch.Tensor:
    log_temperature = torch.log(prepared_batch.episode_data.temperature)
    preds = balance_model(log_temperature)
    balance_loss = compute_balance_loss(
        preds,
        prepared_batch.door_matches,
        prepared_batch.toilet_crossed_room_idx,
    )
    (balance_loss * loss_scale).backward()
    return balance_loss


def proposal_batch_loss(
    proposal_score: torch.Tensor,
    frontier_idx: torch.Tensor,
    door_variant_idx: torch.Tensor,
    target_logits: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    frontier_idx = frontier_idx.to(device, dtype=torch.int64)
    door_variant_idx = door_variant_idx.to(device, dtype=torch.int64)
    target_logits = target_logits.to(device, dtype=torch.float32)
    frontier_valid = frontier_idx >= 0
    if frontier_valid.ndim == 1:
        frontier_valid = frontier_valid.unsqueeze(1)
    valid = (
        frontier_valid
        & (door_variant_idx >= 0)
        & (door_variant_idx < proposal_score.shape[1])
        & torch.isfinite(target_logits)
    )
    row_valid = torch.any(valid, dim=1)
    if not torch.any(row_valid):
        return torch.sum(proposal_score) * 0.0
    safe_door_variant_idx = door_variant_idx.clamp_min(0)
    batch_idx = torch.arange(
        door_variant_idx.shape[0],
        dtype=torch.int64,
        device=device,
    ).unsqueeze(1)
    candidate_logits = proposal_score[
        batch_idx,
        safe_door_variant_idx,
    ]
    candidate_logits = torch.where(
        valid,
        candidate_logits,
        torch.full_like(candidate_logits, float("-inf")),
    ).to(torch.float32)
    target_logits = torch.where(
        valid,
        target_logits,
        torch.full_like(target_logits, float("-inf")),
    )
    row_candidate_logits = candidate_logits[row_valid]
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
        torch.sum(torch.where(row_mask, kl_terms, torch.zeros_like(kl_terms)))
        / row_mask.shape[0]
    )
    return proposal_loss


def proposal_scores_for_frontier(
    proposal_score: torch.Tensor,
    row_snapshot_idx: torch.Tensor,
    row_frontier_idx: torch.Tensor,
    frontier_idx: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    frontier_idx = frontier_idx.to(device)
    result = torch.full(
        (frontier_idx.shape[0], proposal_score.shape[1]),
        float("-inf"),
        dtype=proposal_score.dtype,
        device=device,
    )
    if proposal_score.shape[0] == 0:
        return result
    row_snapshot_idx = row_snapshot_idx.to(device)
    row_frontier_idx = row_frontier_idx.to(device)
    row_valid = (
        (row_snapshot_idx >= 0)
        & (row_snapshot_idx < frontier_idx.shape[0])
        & (row_frontier_idx == frontier_idx[row_snapshot_idx])
    )
    if torch.any(row_valid):
        result[row_snapshot_idx[row_valid]] = proposal_score[row_valid]
    return result


def train_feature_batch_backward(
    context: TrainRoundContext,
    prepared_batch: PreparedTrainBatch,
    loss_scale: float,
) -> MainLossBreakdown:
    if prepared_batch.prefix_count == 0:
        raise RuntimeError("feature training batch has no sampled prefixes")

    train_outcomes = prepared_batch.outcomes
    repeated_outcomes = PreliminaryOutcomes(
        door_invalid=train_outcomes.door_invalid.unsqueeze(1),
        connection_invalid=train_outcomes.connection_invalid.unsqueeze(1),
        toilet_invalid=train_outcomes.toilet_invalid.unsqueeze(1),
        door_match=train_outcomes.door_match.unsqueeze(1),
    )
    with torch.no_grad():
        balance_preds = context.balance_model(torch.log(prepared_batch.episode_data.temperature))
        balance_score_target_logits, balance_score_mask = compute_balance_score_target_logits(
            balance_preds,
            prepared_batch.door_matches,
        )
        toilet_balance_score_target_logits, toilet_balance_score_mask = (
            compute_toilet_balance_score_target_logits(
                balance_preds,
                prepared_batch.toilet_crossed_room_idx,
            )
        )
    repeated_balance_score_target_logits = balance_score_target_logits.unsqueeze(1)
    repeated_balance_score_mask = balance_score_mask.unsqueeze(1)
    repeated_toilet_balance_score_target_logits = toilet_balance_score_target_logits.unsqueeze(1)
    repeated_toilet_balance_score_mask = toilet_balance_score_mask.unsqueeze(1)
    batch_size = prepared_batch.episode_data.actions.room_idx.shape[0]
    avg_frontiers_target = prepared_batch.avg_frontiers.to(context.device).unsqueeze(1)
    avg_frontiers_mask = torch.ones(
        [batch_size, 1],
        dtype=torch.bool,
        device=context.device,
    )
    graph_diameter_target = prepared_batch.graph_diameter.to(context.device).unsqueeze(1)
    graph_diameter_mask = torch.ones(
        [batch_size, 1],
        dtype=torch.bool,
        device=context.device,
    )
    save_distance_target = prepared_batch.save_distance.to(context.device).unsqueeze(1)
    save_distance_mask = prepared_batch.save_distance_mask.to(
        device=context.device,
        dtype=torch.bool,
    ).unsqueeze(1)
    refill_distance_target = prepared_batch.refill_distance.to(context.device).unsqueeze(1)
    refill_distance_mask = prepared_batch.refill_distance_mask.to(
        device=context.device,
        dtype=torch.bool,
    ).unsqueeze(1)
    missing_connect_distance_target = prepared_batch.missing_connect_distance.to(
        context.device
    ).unsqueeze(1)
    missing_connect_distance_mask = prepared_batch.missing_connect_distance_mask.to(
        device=context.device,
        dtype=torch.bool,
    ).unsqueeze(1)
    mask = torch.ones(
        [batch_size, 1, 1],
        dtype=torch.bool,
        device=context.device,
    )
    total_loss = empty_main_loss_breakdown()
    prefix_weight = 1.0 / prepared_batch.prefix_count

    for feature_batch in prepared_batch.feature_batches:
        features = feature_batch.features.to(context.device)
        include_proposal = (
            prepared_batch.kind == "fresh"
            and feature_batch.proposal_frontier_idx is not None
            and feature_batch.proposal_door_variant_idx is not None
            and feature_batch.proposal_selected_candidate is not None
            and feature_batch.proposal_target_logits is not None
        )
        with torch.amp.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=context.device.type == "cuda" and context.config.model.autocast,
        ):
            preds = context.main_model(features, include_proposal=include_proposal)
        prefix_loss = compute_loss_breakdown(
            preds,
            repeated_outcomes,
            mask,
            repeated_balance_score_target_logits,
            repeated_balance_score_mask,
            repeated_toilet_balance_score_target_logits,
            repeated_toilet_balance_score_mask,
            avg_frontiers_target,
            avg_frontiers_mask,
            graph_diameter_target,
            graph_diameter_mask,
            save_distance_target,
            save_distance_mask,
            refill_distance_target,
            refill_distance_mask,
            missing_connect_distance_target,
            missing_connect_distance_mask,
            context.loss_config,
        )
        backward_loss = prefix_loss.total * prefix_weight
        total_loss.total += prefix_loss.total.item() * prefix_weight
        total_loss.door += prefix_loss.door.item() * prefix_weight
        total_loss.connection += prefix_loss.connection.item() * prefix_weight
        total_loss.toilet += prefix_loss.toilet.item() * prefix_weight
        total_loss.balance += prefix_loss.balance.item() * prefix_weight
        total_loss.toilet_balance += prefix_loss.toilet_balance.item() * prefix_weight
        total_loss.avg_frontiers += prefix_loss.avg_frontiers.item() * prefix_weight
        total_loss.graph_diameter += prefix_loss.graph_diameter.item() * prefix_weight
        total_loss.save_distance += prefix_loss.save_distance.item() * prefix_weight
        total_loss.refill_distance += prefix_loss.refill_distance.item() * prefix_weight
        total_loss.missing_connect_distance += (
            prefix_loss.missing_connect_distance.item() * prefix_weight
        )
        total_loss.door_contribution += prefix_loss.door_contribution.item() * prefix_weight
        total_loss.connection_contribution += (
            prefix_loss.connection_contribution.item() * prefix_weight
        )
        total_loss.toilet_contribution += (
            prefix_loss.toilet_contribution.item() * prefix_weight
        )
        total_loss.balance_contribution += (
            prefix_loss.balance_contribution.item() * prefix_weight
        )
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
        total_loss.missing_connect_distance_contribution += (
            prefix_loss.missing_connect_distance_contribution.item() * prefix_weight
        )
        if include_proposal:
            proposal_score = proposal_scores_for_frontier(
                preds.proposal_score,
                preds.proposal_row_snapshot_idx,
                preds.proposal_row_frontier_idx,
                feature_batch.proposal_frontier_idx,
                context.device,
            )
            batch_proposal_loss = proposal_batch_loss(
                proposal_score,
                feature_batch.proposal_frontier_idx,
                feature_batch.proposal_door_variant_idx,
                feature_batch.proposal_target_logits,
                context.device,
            )
            weighted_proposal_loss = (
                context.config.train.proposal_weight
                * batch_proposal_loss
                * prefix_weight
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
        raise RuntimeError(f"non-finite loss before backward: {loss.total}")
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
    prepared_batches: list[PreparedTrainBatch],
) -> tuple[MainLossBreakdown, float, int]:
    context.main_model.zero_grad()
    context.balance_model.zero_grad()
    loss_scale = 1.0 / len(prepared_batches)
    group_loss = empty_main_loss_breakdown()
    group_balance_loss = 0.0
    for prepared_batch in prepared_batches:
        batch_loss, batch_balance_loss = train_batch_backward(
            context,
            prepared_batch,
            loss_scale,
        )
        accumulate_main_loss(group_loss, batch_loss)
        group_balance_loss += batch_balance_loss
    train_optimizer_step(context)
    return group_loss, group_balance_loss, len(prepared_batches)


def add_completed_batch_group(
    context: TrainRoundContext,
    total_loss: MainLossBreakdown,
    total_balance_loss: float,
    train_batch_count: int,
    prepared_batch_group: list[PreparedTrainBatch],
) -> tuple[float, int]:
    group_loss, group_balance_loss, group_count = train_prepared_batch_group(
        context,
        prepared_batch_group,
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
) -> tuple[MainLossBreakdown, float]:
    set_optimizer_lrs(context.main_optimizer, context.step_config.optimizer)
    set_optimizer_lrs(context.balance_optimizer, context.step_config.balance_optimizer)

    total_loss = empty_main_loss_breakdown()
    total_balance_loss = 0.0
    train_batch_count = 0
    prepared_batch_group = []

    prepared_batches = context.train_batch_prefetcher.map(
        iter_train_batch_tasks(context.config, context.experience),
        lambda task: prepare_train_batch_task(
            context,
            task,
            episode_data,
            episode_outcomes,
            proposal_data,
        ),
    )
    for prepared_batch in iter_shuffled_prepared_batches(
        prepared_batches,
        context.config.train.shuffle_buffer_batches,
    ):
        prepared_batch_group.append(prepared_batch)
        if len(prepared_batch_group) == context.config.train.gradient_accumulation_steps:
            total_balance_loss, train_batch_count = add_completed_batch_group(
                context,
                total_loss,
                total_balance_loss,
                train_batch_count,
                prepared_batch_group,
            )
            prepared_batch_group = []
    if prepared_batch_group:
        total_balance_loss, train_batch_count = add_completed_batch_group(
            context,
            total_loss,
            total_balance_loss,
            train_batch_count,
            prepared_batch_group,
        )

    if train_batch_count == 0:
        return empty_main_loss_breakdown(), 0.0
    return (
        average_main_loss(total_loss, train_batch_count),
        total_balance_loss / train_batch_count,
    )


def set_optimizer_lrs(optimizer, config) -> None:
    if hasattr(optimizer, "set_lrs"):
        optimizer.set_lrs(config)
    else:
        optimizer.param_groups[0]["lr"] = config.lr
