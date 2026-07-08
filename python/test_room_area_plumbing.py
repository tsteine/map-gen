import tempfile
from pathlib import Path

import torch

from env import Actions, AREA_COUNT, DUMMY_AREA, Engine, EpisodeData
from experience import ExperienceStorage
from train_config import FeatureConfig


def disabled_features() -> FeatureConfig:
    return FeatureConfig(
        inventory=False,
        temperature=False,
        recommended_candidates=False,
        generation_variable_floats=False,
        lookahead_outcomes=0,
        room_position=False,
        global_room_position=0,
        room_part_furthest_distance=0,
        room_part_save_distance=0,
        room_part_refill_distance=0,
        room_part_frontier_distance=0,
        frontier_mask=False,
        frontier_position=0,
        frontier_orientation=0,
        frontier_kind=0,
        frontier_door_variant=0,
        frontier_occupancy=False,
        frontier_neighbor=False,
        frontier_neighbor_position_embedding=0,
        frontier_neighbor_flags=False,
        connection_reachability=0,
        frontier_connection_reachability=False,
        missing_connect_query=False,
        save_utility_query=False,
        refill_utility_query=False,
        toilet_crossed_room=0,
        known_distance=0,
    )


def one_tile_room(name: str, direction: str) -> dict:
    return {
        "name": name,
        "map": [[1]],
        "doors": [[{"id": 0, "direction": direction, "x": 0, "y": 0, "kind": 0}]],
        "connections": [],
        "missing_connections": [],
        "toilet_crossing_x": [],
    }


def test_environment_group_round_trips_room_area() -> None:
    engine = Engine(
        [
            one_tile_room("Right", "right"),
            one_tile_room("Left", "left"),
        ],
        disabled_features(),
    )
    env = engine.create_environment_group(
        map_size=(4, 4),
        num_envs=1,
        candidate_spatial_cell_size=4,
        seed=0,
        num_threads=1,
    )
    device = torch.device("cpu")

    first = Actions(
        room_idx=torch.tensor([0], dtype=torch.uint8),
        room_x=torch.tensor([0], dtype=torch.int8),
        room_y=torch.tensor([0], dtype=torch.int8),
        room_area=torch.tensor([2], dtype=torch.uint8),
    )
    second = Actions(
        room_idx=torch.tensor([1], dtype=torch.uint8),
        room_x=torch.tensor([1], dtype=torch.int8),
        room_y=torch.tensor([0], dtype=torch.int8),
        room_area=torch.tensor([4], dtype=torch.uint8),
    )
    finish = Actions(
        room_idx=torch.tensor([2], dtype=torch.uint8),
        room_x=torch.tensor([0], dtype=torch.int8),
        room_y=torch.tensor([0], dtype=torch.int8),
        room_area=torch.tensor([DUMMY_AREA], dtype=torch.uint8),
    )

    env.step_known(first)
    env.step_known(second)
    env.step(finish)
    actions = env.get_actions(device)

    assert AREA_COUNT == 6
    assert actions.room_idx.tolist() == [[0, 1, 2]]
    assert actions.room_x.tolist() == [[0, 1, 0]]
    assert actions.room_y.tolist() == [[0, 0, 0]]
    assert actions.room_area.tolist() == [[2, 4, DUMMY_AREA]]


def test_experience_storage_round_trips_room_area() -> None:
    episode_data = EpisodeData(
        actions=Actions(
            room_idx=torch.tensor([[0, 1], [1, 2]], dtype=torch.uint8),
            room_x=torch.tensor([[0, 1], [1, 2]], dtype=torch.int8),
            room_y=torch.tensor([[2, 3], [3, 4]], dtype=torch.int8),
            room_area=torch.tensor([[0, 5], [3, DUMMY_AREA]], dtype=torch.uint8),
        ),
        temperature=torch.tensor([1.0, 2.0]),
        recommended_candidates=torch.tensor([8.0, 8.0]),
        generation_variable_floats=torch.empty((2, 0)),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        storage = ExperienceStorage(
            num_rooms=2,
            data_path=Path(temp_dir),
            episodes_per_file=2,
        )
        storage.store(episode_data)
        loaded = storage.read_files([0], episodes_per_file=2)

    loaded_order = torch.argsort(loaded.actions.room_idx[:, 0])
    episode_order = torch.argsort(episode_data.actions.room_idx[:, 0])
    assert torch.equal(
        loaded.actions.room_idx[loaded_order],
        episode_data.actions.room_idx[episode_order],
    )
    assert torch.equal(
        loaded.actions.room_x[loaded_order],
        episode_data.actions.room_x[episode_order],
    )
    assert torch.equal(
        loaded.actions.room_y[loaded_order],
        episode_data.actions.room_y[episode_order],
    )
    assert torch.equal(
        loaded.actions.room_area[loaded_order],
        episode_data.actions.room_area[episode_order],
    )


def main() -> None:
    test_environment_group_round_trips_room_area()
    test_experience_storage_round_trips_room_area()


if __name__ == "__main__":
    main()
