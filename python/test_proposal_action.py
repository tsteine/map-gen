import torch

from env import (
    AREA_COUNT,
    proposal_action_door_variant_idx,
    proposal_action_idx,
    proposal_action_room_area,
)
from learn import proposal_batch_loss


def test_proposal_action_helpers_flatten_area_variants() -> None:
    door_variant_idx = torch.tensor([0, 0, 1, 2], dtype=torch.int16)
    room_area = torch.tensor([0, 5, 3, 4], dtype=torch.int16)
    action_idx = proposal_action_idx(door_variant_idx, room_area)

    assert AREA_COUNT == 6
    assert action_idx.tolist() == [0, 5, 9, 16]
    assert proposal_action_door_variant_idx(action_idx).tolist() == [0, 0, 1, 2]
    assert proposal_action_room_area(action_idx).tolist() == [0, 5, 3, 4]


def test_proposal_loss_indexes_flattened_area_actions() -> None:
    device = torch.device("cpu")
    frontier_idx = torch.tensor([0], dtype=torch.int16)
    action_idx = torch.tensor([[6, 7]], dtype=torch.int16)
    target_logits = torch.tensor([[0.0, 10.0]], dtype=torch.float32)

    aligned_score = torch.zeros((1, AREA_COUNT * 2), dtype=torch.float32)
    aligned_score[0, 6] = 0.0
    aligned_score[0, 7] = 10.0
    reversed_score = torch.zeros((1, AREA_COUNT * 2), dtype=torch.float32)
    reversed_score[0, 6] = 10.0
    reversed_score[0, 7] = 0.0

    aligned_loss = proposal_batch_loss(
        aligned_score,
        frontier_idx,
        action_idx,
        target_logits,
        device,
    )
    reversed_loss = proposal_batch_loss(
        reversed_score,
        frontier_idx,
        action_idx,
        target_logits,
        device,
    )

    assert torch.isfinite(aligned_loss)
    assert torch.isfinite(reversed_loss)
    assert aligned_loss < reversed_loss


def main() -> None:
    test_proposal_action_helpers_flatten_area_variants()
    test_proposal_loss_indexes_flattened_area_actions()


if __name__ == "__main__":
    main()
