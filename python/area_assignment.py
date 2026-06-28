from dataclasses import dataclass

import torch

from generate import GenerationProfiler, profile_start, sync_profile_device


AREA_COUNT = 6
DIRECTIONS = ("left", "right", "up", "down")


@dataclass(frozen=True)
class DoorRoomLookup:
    left: torch.Tensor
    right: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    room_count: int


@dataclass(frozen=True)
class RoomGeometry:
    min_x: torch.Tensor
    max_x: torch.Tensor
    min_y: torch.Tensor
    max_y: torch.Tensor


@dataclass(frozen=True)
class MapStationData:
    map_station: torch.Tensor
    phantoon_map: torch.Tensor
    left_slot: torch.Tensor
    right_slot: torch.Tensor
    movable_left_map: torch.Tensor
    movable_right_map: torch.Tensor


@dataclass(frozen=True)
class ToiletData:
    toilet: torch.Tensor
    room_count: int


@dataclass(frozen=True)
class AreaAssignment:
    room_idx: torch.Tensor
    area: torch.Tensor
    valid_mask: torch.Tensor
    crossing_count: torch.Tensor


def add_area_profile(
    profiler: GenerationProfiler,
    device: torch.device,
    name: str,
    start: int,
) -> None:
    sync_profile_device(device, profiler.enabled)
    profiler.add(name, start)


def build_door_room_lookup(rooms: list[dict], device: torch.device) -> DoorRoomLookup:
    door_rooms = {direction: [] for direction in DIRECTIONS}
    for room_idx, room in enumerate(rooms):
        for door_group in room["doors"]:
            for door in door_group:
                door_rooms[door["direction"]].append(room_idx)
    return DoorRoomLookup(
        left=torch.tensor(door_rooms["left"], device=device, dtype=torch.int64),
        right=torch.tensor(door_rooms["right"], device=device, dtype=torch.int64),
        up=torch.tensor(door_rooms["up"], device=device, dtype=torch.int64),
        down=torch.tensor(door_rooms["down"], device=device, dtype=torch.int64),
        room_count=len(rooms),
    )


def occupied_cells(room: dict) -> list[tuple[int, int]]:
    return [
        (x, y)
        for y, row in enumerate(room["map"])
        for x, value in enumerate(row)
        if value
    ]


def build_room_geometry(rooms: list[dict], device: torch.device) -> RoomGeometry:
    min_x = []
    max_x = []
    min_y = []
    max_y = []
    for room in rooms:
        cells = occupied_cells(room)
        xs = [x for x, _ in cells] or [0]
        ys = [y for _, y in cells] or [0]
        min_x.append(min(xs))
        max_x.append(max(xs) + 1)
        min_y.append(min(ys))
        max_y.append(max(ys) + 1)
    return RoomGeometry(
        min_x=torch.tensor(min_x, device=device, dtype=torch.int64),
        max_x=torch.tensor(max_x, device=device, dtype=torch.int64),
        min_y=torch.tensor(min_y, device=device, dtype=torch.int64),
        max_y=torch.tensor(max_y, device=device, dtype=torch.int64),
    )


def flat_doors(room: dict) -> list[dict]:
    return [door for door_group in room["doors"] for door in door_group]


def is_single_tile_single_door_room(room: dict, direction: str) -> bool:
    doors = flat_doors(room)
    return (
        room["map"] == [[1]]
        and len(doors) == 1
        and doors[0]["direction"] == direction
        and doors[0]["x"] == 0
        and doors[0]["y"] == 0
        and doors[0]["kind"] == 0
    )


def build_map_station_data(rooms: list[dict], device: torch.device) -> MapStationData:
    map_station = []
    phantoon_map = []
    left_slot = []
    right_slot = []
    movable_left_map = []
    movable_right_map = []
    for room in rooms:
        is_map_station = bool(room.get("map_station", False))
        is_phantoon_map = room.get("special_type") == "phantoon_map"
        can_host_station = not room.get("save", False) and not room.get("refill", False)
        is_left_slot = can_host_station and is_single_tile_single_door_room(room, "left")
        is_right_slot = can_host_station and is_single_tile_single_door_room(room, "right")
        map_station.append(is_map_station)
        phantoon_map.append(is_phantoon_map)
        left_slot.append(is_left_slot and not is_phantoon_map)
        right_slot.append(is_right_slot and not is_phantoon_map)
        movable_left_map.append(is_map_station and not is_phantoon_map and is_left_slot)
        movable_right_map.append(is_map_station and not is_phantoon_map and is_right_slot)
    return MapStationData(
        map_station=torch.tensor(map_station, device=device, dtype=torch.bool),
        phantoon_map=torch.tensor(phantoon_map, device=device, dtype=torch.bool),
        left_slot=torch.tensor(left_slot, device=device, dtype=torch.bool),
        right_slot=torch.tensor(right_slot, device=device, dtype=torch.bool),
        movable_left_map=torch.tensor(movable_left_map, device=device, dtype=torch.bool),
        movable_right_map=torch.tensor(movable_right_map, device=device, dtype=torch.bool),
    )


def build_toilet_data(rooms: list[dict], device: torch.device) -> ToiletData:
    return ToiletData(
        toilet=torch.tensor(
            [room.get("special_type") == "toilet" for room in rooms],
            device=device,
            dtype=torch.bool,
        ),
        room_count=len(rooms),
    )


def add_matched_door_edges(
    adjacency: torch.Tensor,
    matches: torch.Tensor,
    source_room_by_door: torch.Tensor,
    target_room_by_door: torch.Tensor,
    room_position_by_room: torch.Tensor,
) -> None:
    valid_match = matches >= 0
    environment_idx, source_door_idx = torch.where(valid_match)
    target_door_idx = matches[environment_idx, source_door_idx].to(torch.int64)
    source_room = source_room_by_door[source_door_idx]
    target_room = target_room_by_door[target_door_idx]
    source_position = room_position_by_room[environment_idx, source_room]
    target_position = room_position_by_room[environment_idx, target_room]
    placed_edge = (source_position >= 0) & (target_position >= 0)
    environment_idx = environment_idx[placed_edge]
    source_position = source_position[placed_edge]
    target_position = target_position[placed_edge]
    adjacency[environment_idx, source_position, target_position] = True
    adjacency[environment_idx, target_position, source_position] = True


def build_room_position_by_room(room_idx: torch.Tensor, total_room_count: int) -> torch.Tensor:
    environment_count, room_count = room_idx.shape
    room_position_by_room = torch.full(
        (environment_count, total_room_count),
        -1,
        device=room_idx.device,
        dtype=torch.int64,
    )
    environment_idx = torch.arange(environment_count, device=room_idx.device)[:, None].expand_as(
        room_idx
    )
    room_position = torch.arange(room_count, device=room_idx.device)[None, :].expand_as(room_idx)
    room_position_by_room[environment_idx, room_idx.to(torch.int64)] = room_position
    return room_position_by_room


def build_room_adjacency(
    room_idx: torch.Tensor,
    door_matches,
    door_room_lookup: DoorRoomLookup,
) -> torch.Tensor:
    environment_count, room_count = room_idx.shape
    room_position_by_room = build_room_position_by_room(room_idx, door_room_lookup.room_count)
    adjacency = torch.zeros(
        (environment_count, room_count, room_count),
        device=room_idx.device,
        dtype=torch.bool,
    )
    diagonal = torch.arange(room_count, device=room_idx.device)
    adjacency[:, diagonal, diagonal] = True
    add_matched_door_edges(
        adjacency,
        door_matches.left,
        door_room_lookup.left,
        door_room_lookup.right,
        room_position_by_room,
    )
    add_matched_door_edges(
        adjacency,
        door_matches.right,
        door_room_lookup.right,
        door_room_lookup.left,
        room_position_by_room,
    )
    add_matched_door_edges(
        adjacency,
        door_matches.up,
        door_room_lookup.up,
        door_room_lookup.down,
        room_position_by_room,
    )
    add_matched_door_edges(
        adjacency,
        door_matches.down,
        door_room_lookup.down,
        door_room_lookup.up,
        room_position_by_room,
    )
    return adjacency


def compute_room_distances(adjacency: torch.Tensor) -> torch.Tensor:
    room_count = adjacency.shape[1]
    unreachable_distance = room_count + 1
    distances = torch.full(
        adjacency.shape,
        unreachable_distance,
        device=adjacency.device,
        dtype=torch.int16,
    )
    off_diagonal = ~torch.eye(room_count, device=adjacency.device, dtype=torch.bool)
    distances = torch.where(adjacency & off_diagonal, torch.ones_like(distances), distances)
    distances = torch.where(adjacency & ~off_diagonal, torch.zeros_like(distances), distances)
    for midpoint in range(room_count):
        distances = torch.minimum(
            distances,
            distances[:, :, midpoint, None] + distances[:, None, midpoint, :],
        )
    return distances


def sample_area_centers(
    room_idx: torch.Tensor, attempt_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    environment_count, room_count = room_idx.shape
    if room_count < AREA_COUNT:
        return (
            torch.zeros(
                (environment_count, attempt_count, AREA_COUNT),
                device=room_idx.device,
                dtype=torch.int64,
            ),
            torch.zeros((environment_count,), device=room_idx.device, dtype=torch.bool),
        )
    random_scores = torch.rand(
        (environment_count, attempt_count, room_count),
        device=room_idx.device,
    )
    centers = torch.topk(random_scores, k=AREA_COUNT, dim=2).indices
    valid_mask = torch.ones((environment_count,), device=room_idx.device, dtype=torch.bool)
    return centers, valid_mask


def assign_rooms_to_centers(distances: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    environment_count, room_count, _ = distances.shape
    attempt_count = centers.shape[1]
    center_distances = torch.gather(
        distances[:, :, None, :].expand(environment_count, room_count, attempt_count, room_count),
        dim=3,
        index=centers[:, None, :, :].expand(
            environment_count,
            room_count,
            attempt_count,
            AREA_COUNT,
        ),
    )
    return torch.argmin(center_distances, dim=3).permute(0, 2, 1)


def placed_room_bounds(
    room_idx: torch.Tensor,
    room_x: torch.Tensor,
    room_y: torch.Tensor,
    room_geometry: RoomGeometry,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    room_idx = room_idx.to(torch.int64)
    return (
        room_x.to(torch.int64) + room_geometry.min_x[room_idx],
        room_x.to(torch.int64) + room_geometry.max_x[room_idx],
        room_y.to(torch.int64) + room_geometry.min_y[room_idx],
        room_y.to(torch.int64) + room_geometry.max_y[room_idx],
    )


def area_assignment_valid_mask(
    area_by_attempt: torch.Tensor,
    room_idx: torch.Tensor,
    room_x: torch.Tensor,
    room_y: torch.Tensor,
    room_geometry: RoomGeometry,
    max_width: int,
    max_height: int,
    min_rooms: int,
    max_rooms: int,
) -> torch.Tensor:
    environment_count, attempt_count, room_count = area_by_attempt.shape
    min_x, max_x, min_y, max_y = placed_room_bounds(room_idx, room_x, room_y, room_geometry)
    area_ids = torch.arange(AREA_COUNT, device=room_idx.device)
    area_mask = area_by_attempt[:, :, None, :] == area_ids[None, None, :, None]
    area_room_count = torch.sum(area_mask.to(torch.int64), dim=3)
    high = torch.iinfo(torch.int64).max
    low = torch.iinfo(torch.int64).min
    area_min_x = torch.where(area_mask, min_x[:, None, None, :], high).amin(dim=3)
    area_max_x = torch.where(area_mask, max_x[:, None, None, :], low).amax(dim=3)
    area_min_y = torch.where(area_mask, min_y[:, None, None, :], high).amin(dim=3)
    area_max_y = torch.where(area_mask, max_y[:, None, None, :], low).amax(dim=3)
    area_has_room = area_mask.any(dim=3)
    area_valid = (
        area_has_room
        & (area_room_count >= min_rooms)
        & (area_room_count <= max_rooms)
        & (area_max_x - area_min_x <= max_width)
        & (area_max_y - area_min_y <= max_height)
    )
    return area_valid.all(dim=2).reshape(environment_count, attempt_count)


def map_station_area_valid_mask(
    area_by_attempt: torch.Tensor,
    room_idx: torch.Tensor,
    map_station_data: MapStationData,
) -> torch.Tensor:
    environment_count, attempt_count, _room_count = area_by_attempt.shape
    area_ids = torch.arange(AREA_COUNT, device=room_idx.device)
    area_mask = area_by_attempt[:, :, None, :] == area_ids[None, None, :, None]
    placed_map_station = map_station_data.map_station[room_idx.to(torch.int64)]
    placed_phantoon_map = map_station_data.phantoon_map[room_idx.to(torch.int64)]
    placed_left_slot = map_station_data.left_slot[room_idx.to(torch.int64)]
    placed_right_slot = map_station_data.right_slot[room_idx.to(torch.int64)]
    placed_movable_left_map = map_station_data.movable_left_map[room_idx.to(torch.int64)]
    placed_movable_right_map = map_station_data.movable_right_map[room_idx.to(torch.int64)]

    area_has_phantoon_map = torch.any(area_mask & placed_phantoon_map[:, None, None, :], dim=3)
    area_has_left_slot = torch.any(area_mask & placed_left_slot[:, None, None, :], dim=3)
    area_has_right_slot = torch.any(area_mask & placed_right_slot[:, None, None, :], dim=3)

    unsatisfied_area = ~area_has_phantoon_map
    left_only_area = unsatisfied_area & area_has_left_slot & ~area_has_right_slot
    right_only_area = unsatisfied_area & area_has_right_slot & ~area_has_left_slot
    both_area = unsatisfied_area & area_has_left_slot & area_has_right_slot
    no_slot_area = unsatisfied_area & ~area_has_left_slot & ~area_has_right_slot

    placed_map_station_count = torch.sum(placed_map_station.to(torch.int64), dim=1)
    placed_phantoon_map_count = torch.sum(placed_phantoon_map.to(torch.int64), dim=1)
    movable_left_map_count = torch.sum(placed_movable_left_map.to(torch.int64), dim=1)
    movable_right_map_count = torch.sum(placed_movable_right_map.to(torch.int64), dim=1)

    left_only_count = torch.sum(left_only_area.to(torch.int64), dim=2)
    right_only_count = torch.sum(right_only_area.to(torch.int64), dim=2)
    both_count = torch.sum(both_area.to(torch.int64), dim=2)
    remaining_left_map_count = movable_left_map_count[:, None] - left_only_count
    remaining_right_map_count = movable_right_map_count[:, None] - right_only_count

    return (
        (placed_map_station_count == AREA_COUNT)[:, None]
        & (placed_phantoon_map_count == 1)[:, None]
        & (torch.sum(area_has_phantoon_map.to(torch.int64), dim=2) == 1)
        & ~torch.any(no_slot_area, dim=2)
        & (remaining_left_map_count >= 0)
        & (remaining_right_map_count >= 0)
        & (both_count <= remaining_left_map_count + remaining_right_map_count)
    ).reshape(environment_count, attempt_count)


def toilet_area_valid_mask(
    area_by_attempt: torch.Tensor,
    room_idx: torch.Tensor,
    toilet_crossed_room_idx: torch.Tensor,
    toilet_data: ToiletData,
) -> torch.Tensor:
    environment_count, attempt_count, _room_count = area_by_attempt.shape
    placed_toilet = toilet_data.toilet[room_idx.to(torch.int64)]
    has_toilet = torch.any(placed_toilet, dim=1)
    toilet_position = torch.argmax(placed_toilet.to(torch.int64), dim=1)

    room_position_by_room = build_room_position_by_room(room_idx, toilet_data.room_count)
    crossed_room_idx = toilet_crossed_room_idx.to(torch.int64)
    crossed_lookup_idx = torch.clamp(crossed_room_idx, min=0)
    crossed_position = room_position_by_room[
        torch.arange(environment_count, device=room_idx.device),
        crossed_lookup_idx,
    ]
    has_crossed_room = crossed_room_idx >= 0
    crossed_room_is_placed = crossed_position >= 0

    environment_idx = torch.arange(environment_count, device=room_idx.device)
    toilet_area = area_by_attempt[
        environment_idx[:, None],
        torch.arange(attempt_count, device=room_idx.device)[None, :],
        toilet_position[:, None],
    ]
    crossed_area = area_by_attempt[
        environment_idx[:, None],
        torch.arange(attempt_count, device=room_idx.device)[None, :],
        torch.clamp(crossed_position, min=0)[:, None],
    ]
    constraint_applies = has_toilet & has_crossed_room & crossed_room_is_placed
    return (~constraint_applies[:, None]) | (toilet_area == crossed_area)


def area_crossing_counts(area_by_attempt: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    upper_adjacency = torch.triu(adjacency, diagonal=1)
    same_area = area_by_attempt[:, :, :, None] == area_by_attempt[:, :, None, :]
    crossing_edge = upper_adjacency[:, None, :, :] & ~same_area
    return torch.sum(crossing_edge.to(torch.int64), dim=(2, 3))


def map_station_target_direction(
    area: torch.Tensor,
    room_idx: torch.Tensor,
    map_station_data: MapStationData,
) -> torch.Tensor:
    area_ids = torch.arange(AREA_COUNT, device=room_idx.device)
    area_mask = area[:, None, :] == area_ids[None, :, None]
    placed_phantoon_map = map_station_data.phantoon_map[room_idx.to(torch.int64)]
    placed_left_slot = map_station_data.left_slot[room_idx.to(torch.int64)]
    placed_right_slot = map_station_data.right_slot[room_idx.to(torch.int64)]
    placed_movable_left_map = map_station_data.movable_left_map[room_idx.to(torch.int64)]

    area_has_phantoon_map = torch.any(area_mask & placed_phantoon_map[:, None, :], dim=2)
    area_has_left_slot = torch.any(area_mask & placed_left_slot[:, None, :], dim=2)
    area_has_right_slot = torch.any(area_mask & placed_right_slot[:, None, :], dim=2)

    unsatisfied_area = ~area_has_phantoon_map
    left_only_area = unsatisfied_area & area_has_left_slot & ~area_has_right_slot
    right_only_area = unsatisfied_area & area_has_right_slot & ~area_has_left_slot
    both_area = unsatisfied_area & area_has_left_slot & area_has_right_slot
    left_only_count = torch.sum(left_only_area.to(torch.int64), dim=1)
    movable_left_map_count = torch.sum(placed_movable_left_map.to(torch.int64), dim=1)
    remaining_left_map_count = movable_left_map_count - left_only_count
    both_rank = torch.cumsum(both_area.to(torch.int64), dim=1) - 1
    both_uses_left = both_area & (both_rank < remaining_left_map_count[:, None])

    left_target = left_only_area | both_uses_left
    right_target = right_only_area | (both_area & ~both_uses_left)
    zero_target = torch.zeros_like(left_target, dtype=torch.int64)
    return torch.where(
        left_target,
        torch.ones_like(zero_target),
        torch.where(right_target, torch.full_like(zero_target, 2), zero_target),
    )


def random_position(mask: torch.Tensor) -> torch.Tensor:
    random_scores = torch.rand(mask.shape, device=mask.device)
    ranked_scores = torch.where(mask, random_scores, torch.full_like(random_scores, 2.0))
    return torch.argmin(ranked_scores, dim=1)


def ranked_positions(mask: torch.Tensor, count: int) -> torch.Tensor:
    room_count = mask.shape[1]
    high = torch.full(mask.shape, room_count, device=mask.device, dtype=torch.int64)
    room_position = torch.arange(room_count, device=mask.device, dtype=torch.int64)[None, :]
    ranked = torch.where(mask, room_position, high)
    return torch.topk(ranked, k=count, dim=1, largest=False).values


def apply_map_station_swaps(
    room_idx: torch.Tensor,
    area: torch.Tensor,
    map_station_data: MapStationData,
) -> torch.Tensor:
    if room_idx.shape[0] == 0:
        return room_idx

    swapped_room_idx = room_idx.clone()
    room_idx_i64 = room_idx.to(torch.int64)
    target_direction = map_station_target_direction(area, room_idx, map_station_data)
    environment_idx = torch.arange(room_idx.shape[0], device=room_idx.device)

    for direction, slot_by_room, movable_map_by_room in (
        (1, map_station_data.left_slot, map_station_data.movable_left_map),
        (2, map_station_data.right_slot, map_station_data.movable_right_map),
    ):
        direction_target_area = target_direction == direction
        max_target_count = int(direction_target_area.to(torch.int64).sum(dim=1).max().item())
        if max_target_count == 0:
            continue

        placed_slot = slot_by_room[room_idx_i64]
        placed_movable_map = movable_map_by_room[room_idx_i64]
        direction_target_count = torch.sum(direction_target_area.to(torch.int64), dim=1)
        target_positions = []
        for area_id in range(AREA_COUNT):
            area_target = direction_target_area[:, area_id]
            target_mask = area_target[:, None] & placed_slot & (area == area_id)
            target_positions.append(random_position(target_mask))
        target_positions = torch.stack(target_positions, dim=1)
        active_target_areas = ranked_positions(direction_target_area, max_target_count).clamp(
            max=AREA_COUNT - 1
        )
        target_positions = torch.gather(target_positions, dim=1, index=active_target_areas)
        target_rank = torch.arange(max_target_count, device=room_idx.device)[None, :]
        active = target_rank < direction_target_count[:, None]
        target_is_self = (
            torch.gather(placed_movable_map, dim=1, index=target_positions) & active
        )
        self_source_mask = torch.zeros_like(placed_movable_map)
        active_environment_idx = environment_idx[:, None].expand_as(active)[active]
        self_source_mask[active_environment_idx, target_positions[active]] = target_is_self[active]
        remaining_source_mask = placed_movable_map & ~self_source_mask
        max_remaining_source_count = int(
            (active & ~target_is_self).to(torch.int64).sum(dim=1).max().item()
        )
        if max_remaining_source_count > 0:
            remaining_source_positions = ranked_positions(
                remaining_source_mask,
                max_remaining_source_count,
            ).clamp(max=room_idx.shape[1] - 1)
            remaining_source_rank = torch.cumsum((active & ~target_is_self).to(torch.int64), dim=1)
            remaining_source_rank = (remaining_source_rank - 1).clamp(min=0)
            remaining_source_rank = remaining_source_rank.clamp(max=max_remaining_source_count - 1)
            nonself_source_positions = torch.gather(
                remaining_source_positions,
                dim=1,
                index=remaining_source_rank,
            )
        else:
            nonself_source_positions = target_positions
        source_positions = torch.where(target_is_self, target_positions, nonself_source_positions)

        active_source_positions = source_positions[active]
        active_target_positions = target_positions[active]
        source_values = room_idx[active_environment_idx, active_source_positions]
        target_values = room_idx[active_environment_idx, active_target_positions]
        swapped_room_idx[active_environment_idx, active_source_positions] = target_values
        swapped_room_idx[active_environment_idx, active_target_positions] = source_values

    return swapped_room_idx


def select_best_valid_assignment(
    room_idx: torch.Tensor,
    area_by_attempt: torch.Tensor,
    valid_attempt_mask: torch.Tensor,
    crossing_counts: torch.Tensor,
) -> AreaAssignment:
    environment_count, _, _room_count = area_by_attempt.shape
    valid_map_mask = valid_attempt_mask.any(dim=1)
    high = torch.iinfo(crossing_counts.dtype).max
    ranked_crossings = torch.where(valid_attempt_mask, crossing_counts, high)
    best_attempt = torch.argmin(ranked_crossings, dim=1)
    selected_area = area_by_attempt[
        torch.arange(environment_count, device=area_by_attempt.device),
        best_attempt,
    ]
    selected_crossing_count = crossing_counts[
        torch.arange(environment_count, device=area_by_attempt.device),
        best_attempt,
    ]
    return AreaAssignment(
        room_idx=room_idx[valid_map_mask],
        area=selected_area[valid_map_mask],
        valid_mask=valid_map_mask,
        crossing_count=selected_crossing_count[valid_map_mask],
    )


@torch.compile
def build_room_adjacency_compiled(
    room_idx: torch.Tensor,
    door_matches,
    door_room_lookup: DoorRoomLookup,
) -> torch.Tensor:
    return build_room_adjacency(room_idx, door_matches, door_room_lookup)


@torch.compile
def compute_room_distances_compiled(adjacency: torch.Tensor) -> torch.Tensor:
    return compute_room_distances(adjacency)


@torch.compile
def assign_rooms_to_centers_compiled(
    distances: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    return assign_rooms_to_centers(distances, centers)


@torch.compile
def area_assignment_valid_mask_compiled(
    area_by_attempt: torch.Tensor,
    room_idx: torch.Tensor,
    room_x: torch.Tensor,
    room_y: torch.Tensor,
    room_geometry: RoomGeometry,
    max_width: int,
    max_height: int,
    min_rooms: int,
    max_rooms: int,
) -> torch.Tensor:
    return area_assignment_valid_mask(
        area_by_attempt=area_by_attempt,
        room_idx=room_idx,
        room_x=room_x,
        room_y=room_y,
        room_geometry=room_geometry,
        max_width=max_width,
        max_height=max_height,
        min_rooms=min_rooms,
        max_rooms=max_rooms,
    )


@torch.compile
def map_station_area_valid_mask_compiled(
    area_by_attempt: torch.Tensor,
    room_idx: torch.Tensor,
    map_station_data: MapStationData,
) -> torch.Tensor:
    return map_station_area_valid_mask(area_by_attempt, room_idx, map_station_data)


@torch.compile
def toilet_area_valid_mask_compiled(
    area_by_attempt: torch.Tensor,
    room_idx: torch.Tensor,
    toilet_crossed_room_idx: torch.Tensor,
    toilet_data: ToiletData,
) -> torch.Tensor:
    return toilet_area_valid_mask(
        area_by_attempt,
        room_idx,
        toilet_crossed_room_idx,
        toilet_data,
    )


@torch.compile
def area_crossing_counts_compiled(
    area_by_attempt: torch.Tensor,
    adjacency: torch.Tensor,
) -> torch.Tensor:
    return area_crossing_counts(area_by_attempt, adjacency)


@torch.compile
def select_best_valid_assignment_compiled(
    room_idx: torch.Tensor,
    area_by_attempt: torch.Tensor,
    valid_attempt_mask: torch.Tensor,
    center_valid_mask: torch.Tensor,
    crossing_counts: torch.Tensor,
) -> AreaAssignment:
    return select_best_valid_assignment(
        room_idx,
        area_by_attempt,
        valid_attempt_mask & center_valid_mask[:, None],
        crossing_counts,
    )


def assign_room_areas_from_centers(
    room_idx: torch.Tensor,
    room_x: torch.Tensor,
    room_y: torch.Tensor,
    toilet_crossed_room_idx: torch.Tensor,
    door_matches,
    door_room_lookup: DoorRoomLookup,
    room_geometry: RoomGeometry,
    map_station_data: MapStationData,
    toilet_data: ToiletData,
    centers: torch.Tensor,
    center_valid_mask: torch.Tensor,
    max_width: int,
    max_height: int,
    min_rooms: int,
    max_rooms: int,
    profiler: GenerationProfiler,
) -> AreaAssignment:
    device = room_idx.device

    profile_time = profile_start(profiler.enabled)
    adjacency = build_room_adjacency_compiled(room_idx, door_matches, door_room_lookup)
    add_area_profile(profiler, device, "python.area_assignment.build_room_adjacency", profile_time)

    profile_time = profile_start(profiler.enabled)
    distances = compute_room_distances_compiled(adjacency)
    add_area_profile(profiler, device, "python.area_assignment.compute_room_distances", profile_time)

    profile_time = profile_start(profiler.enabled)
    area_by_attempt = assign_rooms_to_centers_compiled(distances, centers)
    add_area_profile(profiler, device, "python.area_assignment.assign_rooms_to_centers", profile_time)

    profile_time = profile_start(profiler.enabled)
    valid_attempt_mask = area_assignment_valid_mask_compiled(
        area_by_attempt,
        room_idx,
        room_x,
        room_y,
        room_geometry,
        max_width,
        max_height,
        min_rooms,
        max_rooms,
    )
    add_area_profile(
        profiler,
        device,
        "python.area_assignment.bounding_box_valid_mask",
        profile_time,
    )

    profile_time = profile_start(profiler.enabled)
    valid_attempt_mask = valid_attempt_mask & map_station_area_valid_mask_compiled(
        area_by_attempt,
        room_idx,
        map_station_data,
    )
    add_area_profile(
        profiler,
        device,
        "python.area_assignment.map_station_valid_mask",
        profile_time,
    )

    profile_time = profile_start(profiler.enabled)
    valid_attempt_mask = valid_attempt_mask & toilet_area_valid_mask_compiled(
        area_by_attempt,
        room_idx,
        toilet_crossed_room_idx,
        toilet_data,
    )
    add_area_profile(
        profiler,
        device,
        "python.area_assignment.toilet_area_valid_mask",
        profile_time,
    )

    profile_time = profile_start(profiler.enabled)
    crossing_counts = area_crossing_counts_compiled(area_by_attempt, adjacency)
    add_area_profile(
        profiler,
        device,
        "python.area_assignment.area_crossing_counts",
        profile_time,
    )

    profile_time = profile_start(profiler.enabled)
    assignment = select_best_valid_assignment_compiled(
        room_idx,
        area_by_attempt,
        valid_attempt_mask,
        center_valid_mask,
        crossing_counts,
    )
    add_area_profile(
        profiler,
        device,
        "python.area_assignment.select_best_valid_assignment",
        profile_time,
    )

    profile_time = profile_start(profiler.enabled)
    swapped_room_idx = apply_map_station_swaps(
        assignment.room_idx,
        assignment.area,
        map_station_data,
    )
    add_area_profile(
        profiler,
        device,
        "python.area_assignment.apply_map_station_swaps",
        profile_time,
    )
    return AreaAssignment(
        room_idx=swapped_room_idx,
        area=assignment.area,
        valid_mask=assignment.valid_mask,
        crossing_count=assignment.crossing_count,
    )


def assign_room_areas(
    room_idx: torch.Tensor,
    room_x: torch.Tensor,
    room_y: torch.Tensor,
    toilet_crossed_room_idx: torch.Tensor,
    door_matches,
    door_room_lookup: DoorRoomLookup,
    room_geometry: RoomGeometry,
    map_station_data: MapStationData,
    toilet_data: ToiletData,
    attempt_count: int,
    max_width: int,
    max_height: int,
    min_rooms: int,
    max_rooms: int,
    profiler: GenerationProfiler,
) -> AreaAssignment:
    if room_idx.shape[0] == 0:
        return AreaAssignment(
            room_idx=torch.empty(room_idx.shape, device=room_idx.device, dtype=room_idx.dtype),
            area=torch.empty(room_idx.shape, device=room_idx.device, dtype=torch.int64),
            valid_mask=torch.empty((0,), device=room_idx.device, dtype=torch.bool),
            crossing_count=torch.empty((0,), device=room_idx.device, dtype=torch.int64),
        )

    profile_time = profile_start(profiler.enabled)
    centers, center_valid_mask = sample_area_centers(room_idx, attempt_count)
    add_area_profile(
        profiler,
        room_idx.device,
        "python.area_assignment.sample_area_centers",
        profile_time,
    )

    return assign_room_areas_from_centers(
        room_idx,
        room_x,
        room_y,
        toilet_crossed_room_idx,
        door_matches,
        door_room_lookup,
        room_geometry,
        map_station_data,
        toilet_data,
        centers,
        center_valid_mask,
        max_width,
        max_height,
        min_rooms,
        max_rooms,
        profiler,
    )
