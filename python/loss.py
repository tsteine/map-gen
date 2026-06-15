from dataclasses import dataclass

import torch

from env import DoorMatches, PreliminaryOutcomes
from model import BalancePredictions, Predictions

BALANCE_TARGET_LOG_ODDS_LIMIT = 20.0


@dataclass
class LossConfig:
    door_weight: float
    connection_weight: float
    toilet_weight: float
    balance_weight: float
    toilet_balance_weight: float
    avg_frontiers_weight: float
    graph_diameter_weight: float
    save_distance_weight: float
    refill_distance_weight: float


@dataclass
class LossBreakdown:
    total: torch.Tensor
    door: torch.Tensor
    connection: torch.Tensor
    toilet: torch.Tensor
    balance: torch.Tensor
    toilet_balance: torch.Tensor
    avg_frontiers: torch.Tensor
    graph_diameter: torch.Tensor
    save_distance: torch.Tensor
    refill_distance: torch.Tensor
    door_contribution: torch.Tensor
    connection_contribution: torch.Tensor
    toilet_contribution: torch.Tensor
    balance_contribution: torch.Tensor
    toilet_balance_contribution: torch.Tensor
    avg_frontiers_contribution: torch.Tensor
    graph_diameter_contribution: torch.Tensor
    save_distance_contribution: torch.Tensor
    refill_distance_contribution: torch.Tensor


def masked_binary_cross_entropy_loss(preds: torch.Tensor, outcomes: torch.Tensor, mask: torch.Tensor, weight: float) -> torch.Tensor:
    mask = (mask & (outcomes >= 0)).to(preds.dtype)
    binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        preds, outcomes.to(preds.dtype), reduction='none')
    return weight * torch.sum(binary_loss * mask), weight * torch.sum(mask)


def masked_bernoulli_kl_loss(
    logits: torch.Tensor,
    target_logits: torch.Tensor,
    mask: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    logits = logits.to(torch.float32)
    mask = mask.to(logits.dtype)
    target_logits = target_logits.detach().to(logits.dtype)
    target_prob = torch.sigmoid(target_logits)
    prediction_cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        target_prob,
        reduction="none",
    )
    target_entropy = -(
        target_prob * torch.nn.functional.logsigmoid(target_logits)
        + (1.0 - target_prob) * torch.nn.functional.logsigmoid(-target_logits)
    )
    return (
        weight * torch.sum((prediction_cross_entropy - target_entropy) * mask),
        weight * torch.sum(mask),
    )


def masked_mse_loss(
    preds: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = mask.to(torch.float32)
    error = preds.to(torch.float32) - target.to(torch.float32)
    return weight * torch.sum(error.square() * mask), weight * torch.sum(mask)


def compute_loss_breakdown(
    preds: Predictions,
    outcomes: PreliminaryOutcomes,
    mask: torch.Tensor,
    balance_score_target_logits: torch.Tensor,
    balance_score_mask: torch.Tensor,
    toilet_balance_score_target_logits: torch.Tensor,
    toilet_balance_score_mask: torch.Tensor,
    avg_frontiers_target: torch.Tensor,
    avg_frontiers_mask: torch.Tensor,
    graph_diameter_target: torch.Tensor,
    graph_diameter_mask: torch.Tensor,
    save_distance_target: torch.Tensor,
    save_distance_mask: torch.Tensor,
    refill_distance_target: torch.Tensor,
    refill_distance_mask: torch.Tensor,
    config: LossConfig,
) -> LossBreakdown:
    door_loss, door_wt = masked_binary_cross_entropy_loss(
        preds.door_invalid, outcomes.door_invalid, mask, config.door_weight)
    conn_loss, conn_wt = masked_binary_cross_entropy_loss(
        preds.connection_invalid, outcomes.connection_invalid, mask, config.connection_weight)
    toilet_loss, toilet_wt = masked_binary_cross_entropy_loss(
        preds.toilet_invalid, outcomes.toilet_invalid, mask.squeeze(-1), config.toilet_weight)
    balance_loss, balance_wt = masked_bernoulli_kl_loss(
        preds.balance_score,
        balance_score_target_logits,
        mask & balance_score_mask,
        config.balance_weight,
    )
    toilet_balance_loss, toilet_balance_wt = masked_bernoulli_kl_loss(
        preds.toilet_balance_score,
        toilet_balance_score_target_logits,
        mask.squeeze(-1) & toilet_balance_score_mask,
        config.toilet_balance_weight,
    )
    avg_frontiers_mask = avg_frontiers_mask.to(torch.float32)
    avg_frontiers_error = (
        preds.avg_frontiers.to(torch.float32) - avg_frontiers_target.to(torch.float32)
    )
    avg_frontiers_loss = config.avg_frontiers_weight * torch.sum(
        avg_frontiers_error.square() * avg_frontiers_mask
    )
    avg_frontiers_wt = config.avg_frontiers_weight * torch.sum(avg_frontiers_mask)
    graph_diameter_loss, graph_diameter_wt = masked_mse_loss(
        preds.graph_diameter,
        graph_diameter_target,
        graph_diameter_mask,
        config.graph_diameter_weight,
    )
    save_distance_loss, save_distance_wt = masked_mse_loss(
        preds.save_distance,
        save_distance_target,
        save_distance_mask,
        config.save_distance_weight,
    )
    refill_distance_loss, refill_distance_wt = masked_mse_loss(
        preds.refill_distance,
        refill_distance_target,
        refill_distance_mask,
        config.refill_distance_weight,
    )
    total_weight = (
        door_wt
        + conn_wt
        + toilet_wt
        + balance_wt
        + toilet_balance_wt
        + avg_frontiers_wt
        + graph_diameter_wt
        + save_distance_wt
        + refill_distance_wt
        + 1e-15
    )
    door_contribution = door_loss / total_weight
    connection_contribution = conn_loss / total_weight
    toilet_contribution = toilet_loss / total_weight
    balance_contribution = balance_loss / total_weight
    toilet_balance_contribution = toilet_balance_loss / total_weight
    avg_frontiers_contribution = avg_frontiers_loss / total_weight
    graph_diameter_contribution = graph_diameter_loss / total_weight
    save_distance_contribution = save_distance_loss / total_weight
    refill_distance_contribution = refill_distance_loss / total_weight
    mean_loss = (
        door_contribution
        + connection_contribution
        + toilet_contribution
        + balance_contribution
        + toilet_balance_contribution
        + avg_frontiers_contribution
        + graph_diameter_contribution
        + save_distance_contribution
        + refill_distance_contribution
    )
    return LossBreakdown(
        total=mean_loss,
        door=door_loss / (door_wt + 1e-15),
        connection=conn_loss / (conn_wt + 1e-15),
        toilet=toilet_loss / (toilet_wt + 1e-15),
        balance=balance_loss / (balance_wt + 1e-15),
        toilet_balance=toilet_balance_loss / (toilet_balance_wt + 1e-15),
        avg_frontiers=avg_frontiers_loss / (avg_frontiers_wt + 1e-15),
        graph_diameter=graph_diameter_loss / (graph_diameter_wt + 1e-15),
        save_distance=save_distance_loss / (save_distance_wt + 1e-15),
        refill_distance=refill_distance_loss / (refill_distance_wt + 1e-15),
        door_contribution=door_contribution,
        connection_contribution=connection_contribution,
        toilet_contribution=toilet_contribution,
        balance_contribution=balance_contribution,
        toilet_balance_contribution=toilet_balance_contribution,
        avg_frontiers_contribution=avg_frontiers_contribution,
        graph_diameter_contribution=graph_diameter_contribution,
        save_distance_contribution=save_distance_contribution,
        refill_distance_contribution=refill_distance_contribution,
    )


def direction_balance_loss(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = targets >= 0
    if not torch.any(mask):
        return torch.sum(logits) * 0.0, logits.new_tensor(0.0)
    return (
        torch.nn.functional.cross_entropy(logits[mask], targets[mask], reduction="sum"),
        torch.sum(mask).to(logits.dtype),
    )


def compute_balance_loss(
    preds: BalancePredictions,
    door_matches: DoorMatches,
    toilet_crossed_room_idx: torch.Tensor,
) -> torch.Tensor:
    left_loss, left_weight = direction_balance_loss(preds.left, door_matches.left)
    right_loss, right_weight = direction_balance_loss(preds.right, door_matches.right)
    up_loss, up_weight = direction_balance_loss(preds.up, door_matches.up)
    down_loss, down_weight = direction_balance_loss(preds.down, door_matches.down)
    toilet_loss, toilet_weight = direction_balance_loss(
        preds.toilet_crossed_room,
        toilet_crossed_room_idx,
    )
    total_loss = left_loss + right_loss + up_loss + down_loss + toilet_loss
    total_weight = left_weight + right_weight + up_weight + down_weight + toilet_weight
    return total_loss / (total_weight + 1e-15)


def compute_balance_door_match_ss(preds: BalancePredictions) -> torch.Tensor:
    return (
        torch.sum(torch.softmax(preds.left, dim=-1).square())
        + torch.sum(torch.softmax(preds.right, dim=-1).square())
        + torch.sum(torch.softmax(preds.up, dim=-1).square())
        + torch.sum(torch.softmax(preds.down, dim=-1).square())
    )


def compute_balance_toilet_crossed_room_ss(preds: BalancePredictions) -> torch.Tensor:
    return torch.sum(torch.softmax(preds.toilet_crossed_room, dim=-1).square())


def direction_balance_score_target_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = targets >= 0
    if logits.shape[-1] == 0:
        return logits.new_empty(targets.shape, dtype=torch.float32), mask
    safe_targets = torch.clamp(targets, min=0).to(torch.int64)
    target_logits = torch.gather(
        direction_balance_score_logit_table(logits),
        -1,
        safe_targets.unsqueeze(-1),
    ).squeeze(-1)
    return target_logits.detach(), mask


def direction_balance_score_logit_table(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.to(torch.float32)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    non_target_log_probs = torch.log(-torch.expm1(log_probs))
    return torch.clamp(
        log_probs - non_target_log_probs,
        min=-BALANCE_TARGET_LOG_ODDS_LIMIT,
        max=BALANCE_TARGET_LOG_ODDS_LIMIT,
    )


def compute_balance_score_target_logits(
    preds: BalancePredictions,
    door_matches: DoorMatches,
) -> tuple[torch.Tensor, torch.Tensor]:
    left_values, left_mask = direction_balance_score_target_logits(preds.left, door_matches.left)
    right_values, right_mask = direction_balance_score_target_logits(preds.right, door_matches.right)
    up_values, up_mask = direction_balance_score_target_logits(preds.up, door_matches.up)
    down_values, down_mask = direction_balance_score_target_logits(preds.down, door_matches.down)
    return (
        torch.cat([left_values, right_values, up_values, down_values], dim=-1),
        torch.cat([left_mask, right_mask, up_mask, down_mask], dim=-1),
    )


def compute_toilet_balance_score_target_logits(
    preds: BalancePredictions,
    toilet_crossed_room_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return direction_balance_score_target_logits(
        preds.toilet_crossed_room,
        toilet_crossed_room_idx,
    )
