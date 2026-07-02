from __future__ import annotations

from dataclasses import dataclass

from env import DoorMatches


AREA_COUNT = 6
DIRECTIONS = ("left", "right", "up", "down")
OPPOSITE_DIRECTIONS = {
    "left": "right",
    "right": "left",
    "up": "down",
    "down": "up",
}


@dataclass(frozen=True)
class SmallMapConfig:
    min_rooms: int
    max_rooms: int
    target_rooms: int


@dataclass(frozen=True)
class DirectionDoorData:
    room_idx: list[int]
    part_idx: list[int]
    door_id: list[int]
    kind: list[int]


@dataclass(frozen=True)
class DoorData:
    left: DirectionDoorData
    right: DirectionDoorData
    up: DirectionDoorData
    down: DirectionDoorData


@dataclass(frozen=True)
class RoomPartData:
    part_count: list[int]
    connections: list[list[tuple[int, int]]]


@dataclass(frozen=True)
class SmallMapResult:
    source_map_idx: list[int]
    room_idx: list[list[int]]
    room_x: list[list[int]]
    room_y: list[list[int]]
    area: list[list[int]]
    subarea: list[list[int]]
    subsubarea: list[list[int]]
    toilet_crossed_room_idx: list[int]


def flat_doors(room: dict) -> list[dict]:
    return [door for door_group in room["doors"] for door in door_group]


def tensor_to_list(tensor) -> list:
    return tensor.detach().cpu().tolist()


def build_door_data(rooms: list[dict]) -> DoorData:
    door_rooms = {direction: [] for direction in DIRECTIONS}
    door_parts = {direction: [] for direction in DIRECTIONS}
    door_ids = {direction: [] for direction in DIRECTIONS}
    door_kinds = {direction: [] for direction in DIRECTIONS}
    for room_idx, room in enumerate(rooms):
        for part_idx, door_group in enumerate(room["doors"]):
            for door in door_group:
                direction = door["direction"]
                door_rooms[direction].append(room_idx)
                door_parts[direction].append(part_idx)
                door_ids[direction].append(int(door["id"]))
                door_kinds[direction].append(int(door["kind"]))
    return DoorData(
        left=DirectionDoorData(
            room_idx=door_rooms["left"],
            part_idx=door_parts["left"],
            door_id=door_ids["left"],
            kind=door_kinds["left"],
        ),
        right=DirectionDoorData(
            room_idx=door_rooms["right"],
            part_idx=door_parts["right"],
            door_id=door_ids["right"],
            kind=door_kinds["right"],
        ),
        up=DirectionDoorData(
            room_idx=door_rooms["up"],
            part_idx=door_parts["up"],
            door_id=door_ids["up"],
            kind=door_kinds["up"],
        ),
        down=DirectionDoorData(
            room_idx=door_rooms["down"],
            part_idx=door_parts["down"],
            door_id=door_ids["down"],
            kind=door_kinds["down"],
        ),
    )


def build_room_part_data(rooms: list[dict]) -> RoomPartData:
    return RoomPartData(
        part_count=[len(room["doors"]) for room in rooms],
        connections=[
            [(int(from_part), int(to_part)) for from_part, to_part in room["connections"]]
            for room in rooms
        ],
    )


def placement_positions_by_room(room_idx: list[int]) -> dict[int, int]:
    return {room: position for position, room in enumerate(room_idx)}


def reachable_from(start: int, graph: list[list[int]]) -> set[int]:
    seen = {start}
    stack = [start]
    while stack:
        source = stack.pop()
        for target in graph[source]:
            if target not in seen:
                seen.add(target)
                stack.append(target)
    return seen


def is_strongly_connected(adjacency: list[list[int]]) -> bool:
    node_count = len(adjacency)
    if node_count <= 1:
        return True

    reverse_adjacency = [[] for _ in range(node_count)]
    for source, targets in enumerate(adjacency):
        for target in targets:
            reverse_adjacency[target].append(source)

    all_nodes = set(range(node_count))
    return (
        reachable_from(0, adjacency) == all_nodes
        and reachable_from(0, reverse_adjacency) == all_nodes
    )


def subset_valid(
    area_mask: int,
    map_room_idx: list[int],
    map_area: list[int],
    direction_matches: dict[str, list[int]],
    direction_data: dict[str, DirectionDoorData],
    room_part_data: RoomPartData,
    config: SmallMapConfig,
) -> bool:
    included = [(area_mask & (1 << area)) != 0 for area in map_area]
    included_count = sum(1 for is_included in included if is_included)
    if included_count < config.min_rooms or included_count > config.max_rooms:
        return False

    room_part_offset_by_position = {}
    included_part_count = 0
    for old_position, is_included in enumerate(included):
        if is_included:
            room_idx = map_room_idx[old_position]
            room_part_offset_by_position[old_position] = included_part_count
            included_part_count += room_part_data.part_count[room_idx]

    room_position = placement_positions_by_room(map_room_idx)
    adjacency = [[] for _ in range(included_part_count)]
    for room_position_idx, is_included in enumerate(included):
        if not is_included:
            continue
        room_idx = map_room_idx[room_position_idx]
        room_part_offset = room_part_offset_by_position[room_position_idx]
        for from_part, to_part in room_part_data.connections[room_idx]:
            adjacency[room_part_offset + from_part].append(room_part_offset + to_part)

    for direction in DIRECTIONS:
        source_data = direction_data[direction]
        target_data = direction_data[OPPOSITE_DIRECTIONS[direction]]
        for source_direction_door_idx, target_direction_door_idx in enumerate(
            direction_matches[direction]
        ):
            if target_direction_door_idx < 0:
                continue
            source_room_position = room_position.get(
                source_data.room_idx[source_direction_door_idx]
            )
            target_room_position = room_position.get(
                target_data.room_idx[target_direction_door_idx]
            )
            if source_room_position is None or target_room_position is None:
                continue

            source_included = included[source_room_position]
            target_included = included[target_room_position]
            if source_included and target_included:
                source_part_position = (
                    room_part_offset_by_position[source_room_position]
                    + source_data.part_idx[source_direction_door_idx]
                )
                target_part_position = (
                    room_part_offset_by_position[target_room_position]
                    + target_data.part_idx[target_direction_door_idx]
                )
                adjacency[source_part_position].append(target_part_position)
                adjacency[target_part_position].append(source_part_position)
                continue
            if source_included == target_included:
                continue
            if (
                source_data.kind[source_direction_door_idx] != 0
                or target_data.kind[target_direction_door_idx] != 0
            ):
                return False
    return is_strongly_connected(adjacency)


def select_area_mask(
    map_room_idx: list[int],
    map_area: list[int],
    direction_matches: dict[str, list[int]],
    direction_data: dict[str, DirectionDoorData],
    room_part_data: RoomPartData,
    config: SmallMapConfig,
) -> int | None:
    best_mask = None
    best_distance = None
    for area_mask in range(1 << AREA_COUNT):
        if not subset_valid(
            area_mask,
            map_room_idx,
            map_area,
            direction_matches,
            direction_data,
            room_part_data,
            config,
        ):
            continue
        included_count = sum(1 for area in map_area if (area_mask & (1 << area)) != 0)
        distance = abs(included_count - config.target_rooms)
        if best_distance is None or distance < best_distance:
            best_mask = area_mask
            best_distance = distance
    return best_mask


def prune_values(values: list[int], keep: list[bool]) -> list[int]:
    return [value for value, keep_value in zip(values, keep) if keep_value]


def prune_small_maps(
    room_idx: list[list[int]],
    room_x: list[list[int]],
    room_y: list[list[int]],
    area: list[list[int]],
    subarea: list[list[int]],
    subsubarea: list[list[int]],
    toilet_crossed_room_idx: list[int],
    door_matches: DoorMatches,
    door_data: DoorData,
    room_part_data: RoomPartData,
    config: SmallMapConfig,
) -> SmallMapResult:
    direction_matches_by_map = {
        "left": tensor_to_list(door_matches.left),
        "right": tensor_to_list(door_matches.right),
        "up": tensor_to_list(door_matches.up),
        "down": tensor_to_list(door_matches.down),
    }
    direction_data = {
        "left": door_data.left,
        "right": door_data.right,
        "up": door_data.up,
        "down": door_data.down,
    }
    result = SmallMapResult(
        source_map_idx=[],
        room_idx=[],
        room_x=[],
        room_y=[],
        area=[],
        subarea=[],
        subsubarea=[],
        toilet_crossed_room_idx=[],
    )
    for map_idx, map_room_idx in enumerate(room_idx):
        direction_matches = {
            direction: direction_matches_by_map[direction][map_idx] for direction in DIRECTIONS
        }
        area_mask = select_area_mask(
            map_room_idx,
            area[map_idx],
            direction_matches,
            direction_data,
            room_part_data,
            config,
        )
        if area_mask is None:
            continue

        keep = [(area_mask & (1 << room_area)) != 0 for room_area in area[map_idx]]
        result.source_map_idx.append(map_idx)
        result.room_idx.append(prune_values(map_room_idx, keep))
        result.room_x.append(prune_values(room_x[map_idx], keep))
        result.room_y.append(prune_values(room_y[map_idx], keep))
        result.area.append(prune_values(area[map_idx], keep))
        result.subarea.append(prune_values(subarea[map_idx], keep))
        result.subsubarea.append(prune_values(subsubarea[map_idx], keep))
        result.toilet_crossed_room_idx.append(toilet_crossed_room_idx[map_idx])
    return result
