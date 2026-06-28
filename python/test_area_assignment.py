import torch

from area_assignment import (
    AREA_COUNT,
    apply_map_station_swaps,
    build_map_station_data,
    map_station_area_valid_mask,
    map_station_area_valid_mask_compiled,
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
        "doors": [[{"direction": direction, "x": 0, "y": 0, "kind": 0}]],
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
        "doors": [[{"direction": "left", "x": 0, "y": 0, "kind": 0}]],
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
