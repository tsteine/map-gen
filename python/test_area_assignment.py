import torch

from area_assignment import (
    AREA_COUNT,
    apply_map_station_swaps,
    build_map_station_data,
    build_toilet_data,
    map_station_area_valid_mask,
    map_station_area_valid_mask_compiled,
    split_assignment_by_balanced_centers,
    toilet_area_valid_mask,
    toilet_area_valid_mask_compiled,
)


def room(
    name: str,
    direction: str,
    map_station: bool,
    special_type: str | None = None,
    save: bool = False,
    refill: bool = False,
) -> dict:
    room_data = {
        "name": name,
        "map": [[1]],
        "doors": [[{"id": 0, "direction": direction, "x": 0, "y": 0, "kind": 0}]],
        "connections": [],
        "missing_connections": [],
        "toilet_crossing_x": [],
    }
    if map_station:
        room_data["map_station"] = True
    if special_type is not None:
        room_data["special_type"] = special_type
    if save:
        room_data["save"] = True
    if refill:
        room_data["refill"] = True
    return room_data


def nonslot_room(name: str) -> dict:
    return {
        "name": name,
        "map": [[1, 1]],
        "doors": [[{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]],
        "connections": [],
        "missing_connections": [],
        "toilet_crossing_x": [],
    }


def assert_one_map_station_per_area(
    room_idx: torch.Tensor,
    area: torch.Tensor,
    map_station: torch.Tensor,
) -> None:
    placed_map_station = map_station[room_idx.to(torch.int64)]
    for area_id in range(AREA_COUNT):
        area_map_station_count = torch.sum(placed_map_station[area == area_id].to(torch.int64))
        assert int(area_map_station_count.item()) == 1


def main() -> None:
    rooms = [
        room("Phantoon Map", "right", True, special_type="phantoon_map"),
        room("Left Map 1", "left", True),
        room("Left Map 2", "left", True),
        room("Left Map 3", "left", True),
        room("Right Map 1", "right", True),
        room("Right Map 2", "right", True),
        room("Left Slot 1", "left", False),
        room("Right Slot 1", "right", False),
        room("Left Slot 2", "left", False),
        room("Left Slot 3", "left", False),
        nonslot_room("No Slot"),
        room("Save Left Slot", "left", False, save=True),
    ]
    device = torch.device("cpu")
    map_station_data = build_map_station_data(rooms, device)

    line_adjacency = torch.eye(6, device=device, dtype=torch.bool)[None, :, :]
    line_positions = torch.arange(5, device=device)
    line_adjacency[:, line_positions, line_positions + 1] = True
    line_adjacency[:, line_positions + 1, line_positions] = True
    parent_assignment = torch.tensor([[0, 0, 0, 0, 1, 1]], device=device)
    child_assignment = split_assignment_by_balanced_centers(
        adjacency=line_adjacency,
        parent_assignments=(parent_assignment,),
        parent_counts=(2,),
    )
    assert child_assignment.tolist() == [[0, 0, 1, 1, 0, 1]]
    grandchild_assignment = split_assignment_by_balanced_centers(
        adjacency=line_adjacency,
        parent_assignments=(parent_assignment, child_assignment),
        parent_counts=(2, 2),
    )
    assert grandchild_assignment.tolist() == [[0, 1, 0, 1, 0, 0]]

    shortcut_adjacency = torch.eye(5, device=device, dtype=torch.bool)[None, :, :]
    shortcut_edges = torch.tensor(
        [
            [0, 1],
            [1, 3],
            [2, 3],
            [0, 4],
            [2, 4],
        ],
        device=device,
    )
    shortcut_adjacency[:, shortcut_edges[:, 0], shortcut_edges[:, 1]] = True
    shortcut_adjacency[:, shortcut_edges[:, 1], shortcut_edges[:, 0]] = True
    shortcut_parent_assignment = torch.tensor([[0, 0, 0, 0, 1]], device=device)
    shortcut_child_assignment = split_assignment_by_balanced_centers(
        adjacency=shortcut_adjacency,
        parent_assignments=(shortcut_parent_assignment,),
        parent_counts=(2,),
    )
    assert shortcut_child_assignment.tolist() == [[0, 0, 1, 1, 0]]

    room_idx = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], device=device)
    valid_area = torch.tensor([[0, 1, 2, 5, 3, 5, 2, 4]], device=device)
    valid_mask = map_station_area_valid_mask(
        valid_area[:, None, :],
        room_idx,
        map_station_data,
    )
    assert bool(valid_mask.item())
    compiled_valid_mask = map_station_area_valid_mask_compiled(
        valid_area[:, None, :],
        room_idx,
        map_station_data,
    )
    assert bool(compiled_valid_mask.item())
    swapped_room_idx = apply_map_station_swaps(room_idx, valid_area, map_station_data)
    assert int(swapped_room_idx[0, 0].item()) == 0
    assert_one_map_station_per_area(
        swapped_room_idx[0],
        valid_area[0],
        map_station_data.map_station,
    )
    area_two_map_positions = set()
    for seed in range(100):
        torch.manual_seed(seed)
        sampled_room_idx = apply_map_station_swaps(room_idx, valid_area, map_station_data)
        assert_one_map_station_per_area(
            sampled_room_idx[0],
            valid_area[0],
            map_station_data.map_station,
        )
        area_two_positions = torch.where(valid_area[0] == 2)[0]
        area_two_map_mask = map_station_data.map_station[
            sampled_room_idx[0, area_two_positions].to(torch.int64)
        ]
        map_position = area_two_positions[area_two_map_mask][0]
        area_two_map_positions.add(int(map_position.item()))
    assert area_two_map_positions == {2, 6}

    overlap_area = torch.tensor([[0, 4, 1, 5, 4, 3, 2, 4]], device=device)
    overlap_valid_mask = map_station_area_valid_mask(
        overlap_area[:, None, :],
        room_idx,
        map_station_data,
    )
    assert bool(overlap_valid_mask.item())
    overlap_swapped_room_idx = apply_map_station_swaps(room_idx, overlap_area, map_station_data)
    assert_one_map_station_per_area(
        overlap_swapped_room_idx[0],
        overlap_area[0],
        map_station_data.map_station,
    )

    toilet_rooms = [
        nonslot_room("Crossed Room"),
        room("Toilet", "right", False, special_type="toilet"),
    ]
    toilet_data = build_toilet_data(toilet_rooms, device)
    toilet_room_idx = torch.tensor([[0, 1]], device=device)
    toilet_crossed_room_idx = torch.tensor([0], device=device)
    same_area = torch.tensor([[[2, 2]]], device=device)
    different_area = torch.tensor([[[2, 3]]], device=device)
    assert bool(
        toilet_area_valid_mask(
            same_area,
            toilet_room_idx,
            toilet_crossed_room_idx,
            toilet_data,
        ).item()
    )
    assert not bool(
        toilet_area_valid_mask(
            different_area,
            toilet_room_idx,
            toilet_crossed_room_idx,
            toilet_data,
        ).item()
    )
    assert bool(
        toilet_area_valid_mask_compiled(
            same_area,
            toilet_room_idx,
            toilet_crossed_room_idx,
            toilet_data,
        ).item()
    )
    assert not bool(
        toilet_area_valid_mask_compiled(
            different_area,
            toilet_room_idx,
            toilet_crossed_room_idx,
            toilet_data,
        ).item()
    )

    no_slot_room_idx = torch.tensor([[0, 1, 2, 3, 4, 5, 7, 10]], device=device)
    no_slot_area = torch.tensor([[0, 1, 2, 3, 4, 4, 4, 5]], device=device)
    no_slot_valid_mask = map_station_area_valid_mask(
        no_slot_area[:, None, :],
        no_slot_room_idx,
        map_station_data,
    )
    assert not bool(no_slot_valid_mask.item())

    too_many_left_only_area = torch.tensor([[0, 1, 2, 3, 5, 5, 2, 3, 4, 5]], device=device)
    too_many_left_only_room_idx = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 8, 9, 7]], device=device)
    too_many_left_only_valid_mask = map_station_area_valid_mask(
        too_many_left_only_area[:, None, :],
        too_many_left_only_room_idx,
        map_station_data,
    )
    assert not bool(too_many_left_only_valid_mask.item())

    save_slot_room_idx = torch.tensor([[0, 1, 2, 3, 4, 5, 11]], device=device)
    save_slot_area = torch.tensor([[0, 1, 2, 3, 4, 4, 5]], device=device)
    save_slot_valid_mask = map_station_area_valid_mask(
        save_slot_area[:, None, :],
        save_slot_room_idx,
        map_station_data,
    )
    assert not bool(save_slot_valid_mask.item())


if __name__ == "__main__":
    main()
