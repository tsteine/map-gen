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
class AreaAssignment:
    area: torch.Tensor
    valid_mask: torch.Tensor


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


def select_first_valid_assignment(
    area_by_attempt: torch.Tensor,
    valid_attempt_mask: torch.Tensor,
) -> AreaAssignment:
    environment_count, _, room_count = area_by_attempt.shape
    valid_map_mask = valid_attempt_mask.any(dim=1)
    first_valid_attempt = torch.argmax(valid_attempt_mask.to(torch.int64), dim=1)
    selected_area = area_by_attempt[
        torch.arange(environment_count, device=area_by_attempt.device),
        first_valid_attempt,
    ]
    return AreaAssignment(area=selected_area[valid_map_mask], valid_mask=valid_map_mask)


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
def select_first_valid_assignment_compiled(
    area_by_attempt: torch.Tensor,
    valid_attempt_mask: torch.Tensor,
    center_valid_mask: torch.Tensor,
) -> AreaAssignment:
    return select_first_valid_assignment(
        area_by_attempt,
        valid_attempt_mask & center_valid_mask[:, None],
    )


def assign_room_areas_from_centers(
    room_idx: torch.Tensor,
    room_x: torch.Tensor,
    room_y: torch.Tensor,
    door_matches,
    door_room_lookup: DoorRoomLookup,
    room_geometry: RoomGeometry,
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
    assignment = select_first_valid_assignment_compiled(
        area_by_attempt,
        valid_attempt_mask,
        center_valid_mask,
    )
    add_area_profile(
        profiler,
        device,
        "python.area_assignment.select_first_valid_assignment",
        profile_time,
    )
    return assignment


def assign_room_areas(
    room_idx: torch.Tensor,
    room_x: torch.Tensor,
    room_y: torch.Tensor,
    door_matches,
    door_room_lookup: DoorRoomLookup,
    room_geometry: RoomGeometry,
    attempt_count: int,
    max_width: int,
    max_height: int,
    min_rooms: int,
    max_rooms: int,
    profiler: GenerationProfiler,
) -> AreaAssignment:
    if room_idx.shape[0] == 0:
        return AreaAssignment(
            area=torch.empty(room_idx.shape, device=room_idx.device, dtype=torch.int64),
            valid_mask=torch.empty((0,), device=room_idx.device, dtype=torch.bool),
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
        door_matches,
        door_room_lookup,
        room_geometry,
        centers,
        center_valid_mask,
        max_width,
        max_height,
        min_rooms,
        max_rooms,
        profiler,
    )
