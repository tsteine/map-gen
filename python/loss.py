from dataclasses import dataclass

import torch

from env import DoorMatches, Outcomes
from model import BalancePredictions, Predictions


@dataclass
class LossConfig:
    door_weight: float
    connection_weight: float


def masked_binary_cross_entropy_loss(preds: torch.Tensor, outcomes: torch.Tensor, mask: torch.Tensor, weight: float) -> torch.Tensor:
    mask = (mask & (outcomes >= 0)).to(preds.dtype)
    binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        preds, outcomes.to(preds.dtype), reduction='none')
    return weight * torch.sum(binary_loss * mask), weight * torch.sum(mask)


def compute_loss(preds: Predictions, outcomes: Outcomes, mask: torch.Tensor, config: LossConfig) -> torch.Tensor:
    door_loss, door_wt = masked_binary_cross_entropy_loss(
        preds.door_invalid, outcomes.door_invalid, mask, config.door_weight)
    conn_loss, conn_wt = masked_binary_cross_entropy_loss(
        preds.connection_invalid, outcomes.connection_invalid, mask, config.connection_weight)
    mean_loss = (door_loss + conn_loss) / (door_wt + conn_wt + 1e-15)
    return mean_loss


def direction_balance_loss(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = targets >= 0
    if not torch.any(mask):
        return torch.sum(logits) * 0.0, logits.new_tensor(0.0)
    return (
        torch.nn.functional.cross_entropy(logits[mask], targets[mask], reduction="sum"),
        torch.sum(mask).to(logits.dtype),
    )


def compute_balance_loss(preds: BalancePredictions, door_matches: DoorMatches) -> torch.Tensor:
    left_loss, left_weight = direction_balance_loss(preds.left, door_matches.left)
    right_loss, right_weight = direction_balance_loss(preds.right, door_matches.right)
    up_loss, up_weight = direction_balance_loss(preds.up, door_matches.up)
    down_loss, down_weight = direction_balance_loss(preds.down, door_matches.down)
    total_loss = left_loss + right_loss + up_loss + down_loss
    total_weight = left_weight + right_weight + up_weight + down_weight
    return total_loss / (total_weight + 1e-15)


def compute_balance_door_match_ss(preds: BalancePredictions) -> torch.Tensor:
    return (
        torch.sum(torch.softmax(preds.left, dim=-1).square())
        + torch.sum(torch.softmax(preds.right, dim=-1).square())
        + torch.sum(torch.softmax(preds.up, dim=-1).square())
        + torch.sum(torch.softmax(preds.down, dim=-1).square())
    )
