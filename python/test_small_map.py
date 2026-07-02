import torch

from env import DoorMatches
from small_map import (
    SmallMapConfig,
    build_door_data,
    build_room_part_data,
    prune_small_maps,
)


def room(name: str, door_groups: list[list[dict]], connections: list[list[int]]) -> dict:
    return {
        "name": name,
        "map": [[1]],
        "doors": door_groups,
        "connections": connections,
        "missing_connections": [],
        "toilet_crossing_x": [],
    }


def door(direction: str, door_id: int, kind: int = 0) -> dict:
    return {"id": door_id, "direction": direction, "x": 0, "y": 0, "kind": kind}


def chain_door_matches(left_matches: list[int], right_matches: list[int]) -> DoorMatches:
    return DoorMatches(
        left=torch.tensor([left_matches], dtype=torch.int64),
        right=torch.tensor([right_matches], dtype=torch.int64),
        up=torch.empty((1, 0), dtype=torch.int64),
        down=torch.empty((1, 0), dtype=torch.int64),
    )


def prune(
    rooms: list[dict],
    room_idx: list[int],
    area: list[int],
    door_matches: DoorMatches,
    config: SmallMapConfig,
):
    return prune_small_maps(
        room_idx=[room_idx],
        room_x=[list(range(len(room_idx)))],
        room_y=[[value + 10 for value in range(len(room_idx))]],
        area=[area],
        subarea=[[value + 20 for value in range(len(room_idx))]],
        subsubarea=[[value + 30 for value in range(len(room_idx))]],
        toilet_crossed_room_idx=[room_idx[0]],
        door_matches=door_matches,
        door_data=build_door_data(rooms),
        room_part_data=build_room_part_data(rooms),
        config=config,
    )


def test_prunes_room_columns_and_keeps_source_index() -> None:
    rooms = [
        room("A", [[door("right", 10)]], []),
        room("B", [[door("left", 20), door("right", 21)]], []),
        room("C", [[door("left", 30)]], []),
    ]
    result = prune(
        rooms=rooms,
        room_idx=[0, 1, 2],
        area=[0, 1, 2],
        door_matches=chain_door_matches(left_matches=[0, 1], right_matches=[0, 1]),
        config=SmallMapConfig(min_rooms=2, max_rooms=2, target_rooms=2),
    )
    assert result.source_map_idx == [0]
    assert result.room_idx == [[0, 1]]
    assert result.room_x == [[0, 1]]
    assert result.room_y == [[10, 11]]
    assert result.area == [[0, 1]]
    assert result.subarea == [[20, 21]]
    assert result.subsubarea == [[30, 31]]
    assert result.toilet_crossed_room_idx == [0]


def test_non_kind_zero_crossing_discards_map() -> None:
    rooms = [
        room("A", [[door("right", 10, kind=1)]], []),
        room("B", [[door("left", 20, kind=1), door("right", 21, kind=1)]], []),
        room("C", [[door("left", 30, kind=1)]], []),
    ]
    result = prune(
        rooms=rooms,
        room_idx=[0, 1, 2],
        area=[0, 1, 2],
        door_matches=chain_door_matches(left_matches=[0, 1], right_matches=[0, 1]),
        config=SmallMapConfig(min_rooms=2, max_rooms=2, target_rooms=2),
    )
    assert result.room_idx == []


def test_selects_room_count_closest_to_target() -> None:
    rooms = [
        room("A", [[door("right", 10)]], []),
        room("B", [[door("left", 20), door("right", 21)]], []),
        room("C", [[door("left", 30), door("right", 31)]], []),
        room("D", [[door("left", 40)]], []),
    ]
    result = prune(
        rooms=rooms,
        room_idx=[0, 1, 2, 3],
        area=[0, 1, 2, 3],
        door_matches=chain_door_matches(left_matches=[0, 1, 2], right_matches=[0, 1, 2]),
        config=SmallMapConfig(min_rooms=2, max_rooms=4, target_rooms=3),
    )
    assert result.room_idx == [[0, 1, 2]]


def test_room_part_graph_uses_directed_connections() -> None:
    rooms = [
        room("A", [[door("right", 10)]], []),
        room("B", [[door("left", 20)], [door("right", 21)]], [[0, 1]]),
        room("C", [[door("left", 30)]], []),
    ]
    one_way_result = prune(
        rooms=rooms,
        room_idx=[0, 1, 2],
        area=[0, 1, 2],
        door_matches=chain_door_matches(left_matches=[0, 1], right_matches=[0, 1]),
        config=SmallMapConfig(min_rooms=3, max_rooms=3, target_rooms=3),
    )
    assert one_way_result.room_idx == []

    rooms[1]["connections"] = [[0, 1], [1, 0]]
    two_way_result = prune(
        rooms=rooms,
        room_idx=[0, 1, 2],
        area=[0, 1, 2],
        door_matches=chain_door_matches(left_matches=[0, 1], right_matches=[0, 1]),
        config=SmallMapConfig(min_rooms=3, max_rooms=3, target_rooms=3),
    )
    assert two_way_result.room_idx == [[0, 1, 2]]


def main() -> None:
    test_prunes_room_columns_and_keeps_source_index()
    test_non_kind_zero_crossing_discards_map()
    test_selects_room_count_closest_to_target()
    test_room_part_graph_uses_directed_connections()


if __name__ == "__main__":
    main()
