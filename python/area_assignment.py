from dataclasses import dataclass

import torch


AREA_COUNT = 6
DIRECTIONS = ("left", "right", "up", "down")


@dataclass(frozen=True)
class DoorRoomLookup:
    left: torch.Tensor
    right: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor


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
    )


def add_matched_door_edges(
    adjacency: torch.Tensor,
    matches: torch.Tensor,
    source_room_by_door: torch.Tensor,
    target_room_by_door: torch.Tensor,
    room_position_by_room: torch.Tensor,
) -> None:
    valid_match = matches >= 0
    if not bool(torch.any(valid_match).item()):
        return
    environment_idx, source_door_idx = torch.where(valid_match)
    target_door_idx = matches[environment_idx, source_door_idx].to(torch.int64)
    source_room = source_room_by_door[source_door_idx]
    target_room = target_room_by_door[target_door_idx]
    source_position = room_position_by_room[environment_idx, source_room]
    target_position = room_position_by_room[environment_idx, target_room]
    placed_edge = (source_position >= 0) & (target_position >= 0)
    if not bool(torch.any(placed_edge).item()):
        return
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


def max_lookup_room_count(door_room_lookup: DoorRoomLookup) -> int:
    max_room = -1
    for room_by_door in (
        door_room_lookup.left,
        door_room_lookup.right,
        door_room_lookup.up,
        door_room_lookup.down,
    ):
        if room_by_door.numel() > 0:
            max_room = max(max_room, int(torch.max(room_by_door).item()))
    return max_room + 1


def build_room_adjacency(
    room_idx: torch.Tensor,
    door_matches,
    door_room_lookup: DoorRoomLookup,
) -> torch.Tensor:
    environment_count, room_count = room_idx.shape
    total_room_count = max(room_count, max_lookup_room_count(door_room_lookup))
    if room_idx.numel() > 0:
        total_room_count = max(total_room_count, int(torch.max(room_idx).item()) + 1)
    room_position_by_room = build_room_position_by_room(room_idx, total_room_count)
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


def sample_area_centers(room_idx: torch.Tensor) -> torch.Tensor:
    centers = []
    for row in room_idx:
        unique_rooms = torch.unique(row)
        if unique_rooms.numel() < AREA_COUNT:
            raise ValueError(
                f"area assignment requires at least {AREA_COUNT} distinct rooms, "
                f"got {unique_rooms.numel()}"
            )
        selected_rooms = unique_rooms[
            torch.randperm(unique_rooms.numel(), device=row.device)[:AREA_COUNT]
        ]
        center_positions = torch.argmax(
            (row[None, :] == selected_rooms[:, None]).to(torch.int64),
            dim=1,
        )
        centers.append(center_positions)
    return torch.stack(centers)


def assign_room_areas(
    room_idx: torch.Tensor,
    door_matches,
    door_room_lookup: DoorRoomLookup,
) -> torch.Tensor:
    if room_idx.shape[0] == 0:
        return torch.empty(room_idx.shape, device=room_idx.device, dtype=torch.int64)
    adjacency = build_room_adjacency(room_idx, door_matches, door_room_lookup)
    distances = compute_room_distances(adjacency)
    environment_count, room_count = room_idx.shape
    centers = sample_area_centers(room_idx)
    center_distances = torch.gather(
        distances,
        dim=2,
        index=centers[:, None, :].expand(environment_count, room_count, AREA_COUNT),
    )
    return torch.argmin(center_distances, dim=2)
