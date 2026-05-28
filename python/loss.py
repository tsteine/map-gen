import torch

from model import Predictions
from env import Outcomes
from dataclasses import dataclass


@dataclass
class LossConfig:
    door_weight: float
    connection_weight: float


def masked_binary_cross_entropy_loss(preds: torch.Tensor, outcomes: torch.Tensor) -> torch.Tensor:
    mask = (outcomes < 0).to(preds.dtype)
    binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, outcomes, reduction='none')
    return torch.mean(binary_loss * mask)


def compute_loss(self, preds: Predictions, outcomes: Outcomes, config: LossConfig):
    door_loss = masked_binary_cross_entropy_loss(preds.door_invalid, outcomes.door_invalid)
    connection_loss = masked_binary_cross_entropy_loss(preds.connection_invalid, outcomes.connection_invalid)
    total_loss = config.door_weight * door_loss + config.connection_weight * connection_loss
    return total_loss