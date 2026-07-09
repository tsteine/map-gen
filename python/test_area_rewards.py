import torch

from env import GenerateConfig, StepOutcomes
from generate import compute_expected_reward
from model import Predictions
from train_config import GENERATION_VARIABLE_FLOAT_FIELDS


def zero_generate_config(**area_rewards) -> GenerateConfig:
    values = {
        "reward_area_connected": 0.0,
        "reward_area_connected_excess": 0.0,
        "reward_area_crossing": 0.0,
        "reward_area_size_valid": 0.0,
        "reward_area_map_station": 0.0,
    }
    values.update(area_rewards)
    generation_variable_floats = torch.zeros([1, len(GENERATION_VARIABLE_FLOAT_FIELDS)])
    return GenerateConfig(
        episode_length=1,
        recommended_candidates=2,
        shortlist_candidates=2,
        max_candidate_areas_per_placement=2,
        gpu_prefetch_batches=0,
        temperature=torch.ones([1]),
        frontier_temperature=torch.ones([1]),
        proposal_temperature=torch.ones([1]),
        reward_door=0.0,
        reward_connection=0.0,
        reward_toilet=0.0,
        reward_phantoon=0.0,
        reward_balance=0.0,
        reward_toilet_balance=0.0,
        reward_frontier=0.0,
        reward_graph_diameter=0.0,
        reward_save_distance=0.0,
        reward_refill_distance=0.0,
        reward_missing_connect_utility=0.0,
        reward_area_connected=values["reward_area_connected"],
        reward_area_connected_excess=values["reward_area_connected_excess"],
        reward_area_crossing=values["reward_area_crossing"],
        reward_area_size_valid=values["reward_area_size_valid"],
        reward_area_map_station=values["reward_area_map_station"],
        area_connected_component_bucket_excess=torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0]),
        generation_variable_floats=generation_variable_floats,
        log_temperature_model=torch.zeros([1]),
        log_recommended_candidates_model=torch.zeros([1]),
        generation_variable_floats_model=generation_variable_floats,
        candidate_log_temperature_model=torch.zeros([1, 2]),
        candidate_log_recommended_candidates_model=torch.zeros([1, 2]),
        candidate_generation_variable_floats_model=torch.zeros(
            [1, 2, len(GENERATION_VARIABLE_FLOAT_FIELDS)]
        ),
        distance_proximity_scale=1.0,
        autocast=False,
    )


def area_predictions() -> Predictions:
    batch = 1
    candidate = 2
    door = 3
    connection = 4
    room_part = 5
    area = 6
    return Predictions(
        door_invalid=torch.zeros([batch, candidate, door]),
        connection_invalid=torch.zeros([batch, candidate, connection]),
        toilet_invalid=torch.zeros([batch, candidate]),
        phantoon_invalid=torch.zeros([batch, candidate]),
        balance_score=torch.zeros([batch, candidate, door]),
        toilet_balance_score=torch.zeros([batch, candidate]),
        avg_frontiers=torch.zeros([batch, candidate]),
        graph_diameter=torch.zeros([batch, candidate]),
        save_to_room_utility=torch.zeros([batch, candidate, room_part]),
        save_from_room_utility=torch.zeros([batch, candidate, room_part]),
        refill_to_room_utility=torch.zeros([batch, candidate, room_part]),
        refill_from_room_utility=torch.zeros([batch, candidate, room_part]),
        missing_connect_utility=torch.zeros([batch, candidate, connection]),
        area_connected_component_bucket_logits=torch.tensor(
            [
                [
                    [[0.0, 2.0, 0.0, 0.0, 0.0]] * area,
                    [[0.0, 0.0, 0.0, 0.0, 2.0]] * area,
                ]
            ]
        ),
        area_crossings=torch.tensor([[2.0, 0.0]]),
        area_size=torch.zeros([batch, candidate, area, 3]),
        area_map_station_count=torch.zeros([batch, candidate, area, 3]),
        proposal_score=torch.empty([0]),
        frontier_value_score=torch.empty([0]),
        proposal_state=torch.empty([0]),
        proposal_row_snapshot_idx=torch.empty([0], dtype=torch.int64),
        proposal_row_frontier_idx=torch.empty([0], dtype=torch.int64),
    )


def unknown_outcomes() -> StepOutcomes:
    return StepOutcomes(
        door_invalid=torch.full([1, 2, 3], -1.0),
        connection_invalid=torch.full([1, 2, 4], -1.0),
        toilet_invalid=torch.full([1, 2], -1.0),
        phantoon_invalid=torch.full([1, 2], -1.0),
        door_match=torch.full([1, 2, 3], -1.0),
    )


def test_zero_area_rewards_leave_reward_unchanged() -> None:
    reward = compute_expected_reward(
        area_predictions(),
        unknown_outcomes(),
        zero_generate_config(),
    )
    assert torch.equal(reward, torch.zeros([1, 2]))


def test_area_rewards_use_valid_bucket_logprobs_and_excess_penalty() -> None:
    reward = compute_expected_reward(
        area_predictions(),
        unknown_outcomes(),
        zero_generate_config(
            reward_area_connected=2.0,
            reward_area_connected_excess=3.0,
            reward_area_crossing=5.0,
            reward_area_size_valid=7.0,
            reward_area_map_station=11.0,
        ),
    )
    connected_log_probability = torch.log_softmax(
        area_predictions().area_connected_component_bucket_logits,
        dim=-1,
    )[..., 1]
    connected_probability = torch.softmax(
        area_predictions().area_connected_component_bucket_logits,
        dim=-1,
    )
    connected_excess = torch.sum(
        connected_probability * torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0]),
        dim=-1,
    )
    middle_bucket_log_probability = torch.log_softmax(torch.zeros([3]), dim=0)[1]
    expected = (
        2.0 * torch.sum(connected_log_probability, dim=2)
        - 3.0 * torch.sum(connected_excess, dim=2)
        - 5.0 * torch.tensor([[2.0, 0.0]])
        + 7.0 * 6 * middle_bucket_log_probability
        + 11.0 * 6 * middle_bucket_log_probability
    )
    assert torch.allclose(reward, expected, atol=1e-6)


def main() -> None:
    test_zero_area_rewards_leave_reward_unchanged()
    test_area_rewards_use_valid_bucket_logprobs_and_excess_penalty()


if __name__ == "__main__":
    main()
