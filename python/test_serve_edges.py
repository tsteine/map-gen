import torch

from env import DoorMatches
from serve import (
    build_door_lookups,
    response_edges,
    response_toilet_crossing_room_placement_idx,
)


def room(name: str, direction: str) -> dict:
    return {
        "name": name,
        "map": [[1]],
        "doors": [[{"direction": direction, "x": 0, "y": 0, "kind": 0}]],
        "connections": [],
        "missing_connections": [],
        "toilet_crossing_x": [],
    }


def main() -> None:
    rooms = [
        room("Right Door", "right"),
        room("Left Door", "left"),
    ]
    door_lookups = build_door_lookups(rooms)
    door_matches = DoorMatches(
        left=torch.tensor([[0]], dtype=torch.int16),
        right=torch.tensor([[0]], dtype=torch.int16),
        up=torch.empty((1, 0), dtype=torch.int16),
        down=torch.empty((1, 0), dtype=torch.int16),
    )
    edges = response_edges([[0, 1]], door_matches, door_lookups)

    assert edges == {
        "from_room_placement_idx": [[0]],
        "from_door_idx": [[0]],
        "to_room_placement_idx": [[1]],
        "to_door_idx": [[0]],
    }

    toilet_crossing_room_placement_idx = response_toilet_crossing_room_placement_idx(
        room_idx=[
            [4, 7, 2],
            [5, 6, 8],
            [1, 3, 9],
        ],
        toilet_crossed_room_idx=[7, -1, 4],
    )
    assert toilet_crossing_room_placement_idx == [1, -1, -1]


if __name__ == "__main__":
    main()
