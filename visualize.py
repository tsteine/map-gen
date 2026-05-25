# Vibe-coded throwaway visualization code for debugging and demos.
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple


Action = Tuple[int, int, int]


def _as_list(values: Any) -> List[Any]:
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def _select_environment(values: Any, environment_index: int) -> List[Any]:
    values = _as_list(values)
    if values and isinstance(values[0], (list, tuple)):
        return list(values[environment_index])
    return values


def _normalize_actions(actions: Any, environment_index: int) -> List[Action]:
    """Convert engine.get_actions() or a sequence of placements into triples."""
    if isinstance(actions, tuple) and len(actions) == 3:
        room_idx, room_x, room_y = actions
        room_idx = _select_environment(room_idx, environment_index)
        room_x = _select_environment(room_x, environment_index)
        room_y = _select_environment(room_y, environment_index)
        return [(int(r), int(x), int(y)) for r, x, y in zip(room_idx, room_x, room_y)]

    normalized = []
    for action in actions:
        if isinstance(action, dict):
            normalized.append(
                (
                    int(action["room_idx"]),
                    int(action.get("x", action.get("room_x"))),
                    int(action.get("y", action.get("room_y"))),
                )
            )
        else:
            room_idx, x, y = action
            normalized.append((int(room_idx), int(x), int(y)))
    return normalized


def _occupied_cells(room: dict) -> set[Tuple[int, int]]:
    return {
        (x, y)
        for y, row in enumerate(room["map"])
        for x, value in enumerate(row)
        if value
    }


def _door_sides(room: dict) -> set[Tuple[int, int, str]]:
    return {
        (int(door["x"]), int(door["y"]), door["direction"])
        for door_group in room.get("doors", [])
        for door in door_group
    }


def _draw_wall_with_optional_gap(
    ax: Any,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    gap: bool,
    **style: Any,
) -> None:
    if not gap:
        ax.plot([x1, x2], [y1, y2], **style)
        return

    gap_fraction = 0.62
    trim = (1.0 - gap_fraction) / 2.0
    dx = x2 - x1
    dy = y2 - y1
    ax.plot([x1, x1 + dx * trim], [y1, y1 + dy * trim], **style)
    ax.plot([x2 - dx * trim, x2], [y2 - dy * trim, y2], **style)


def display_map(
    rooms: Sequence[dict],
    actions: Any,
    *,
    environment_index: int = 0,
    ax: Optional[Any] = None,
    show_names: bool = True,
) -> Any:
    """Display room placements returned by ``engine.get_actions()``.

    ``rooms`` is the parsed room JSON. ``actions`` may be the raw
    ``engine.get_actions()`` tuple of ``(room_idx, x, y)`` arrays, or any
    iterable of ``(room_idx, x, y)`` placements.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise ImportError("display_map requires matplotlib") from exc

    placements = [
        action
        for action in _normalize_actions(actions, environment_index)
        if 0 <= action[0] < len(rooms)
    ]

    if ax is None:
        _, ax = plt.subplots()

    if not placements:
        ax.set_aspect("equal")
        ax.set_axis_off()
        return ax

    min_x = min(x + cx for room_idx, x, _ in placements for cx, _ in _occupied_cells(rooms[room_idx]))
    max_x = max(x + cx + 1 for room_idx, x, _ in placements for cx, _ in _occupied_cells(rooms[room_idx]))
    min_y = min(y + cy for room_idx, _, y in placements for _, cy in _occupied_cells(rooms[room_idx]))
    max_y = max(y + cy + 1 for room_idx, _, y in placements for _, cy in _occupied_cells(rooms[room_idx]))

    colors = [
        "#8ecae6",
        "#ffb703",
        "#90be6d",
        "#f28482",
        "#b8a1ff",
        "#f4a261",
        "#7bdff2",
        "#cdb4db",
    ]

    for placement_index, (room_idx, room_x, room_y) in enumerate(placements):
        room = rooms[room_idx]
        cells = _occupied_cells(room)
        doors = _door_sides(room)
        fill = colors[placement_index % len(colors)]

        for cell_x, cell_y in cells:
            x = room_x + cell_x
            y = room_y + cell_y
            ax.add_patch(
                Rectangle((x, y), 1, 1, facecolor=fill, edgecolor="none", alpha=0.45)
            )

            sides = {
                "left": ((x, y), (x, y + 1), (cell_x - 1, cell_y)),
                "right": ((x + 1, y), (x + 1, y + 1), (cell_x + 1, cell_y)),
                "up": ((x, y), (x + 1, y), (cell_x, cell_y - 1)),
                "down": ((x, y + 1), (x + 1, y + 1), (cell_x, cell_y + 1)),
            }
            for direction, (start, end, neighbor) in sides.items():
                if neighbor in cells:
                    continue
                _draw_wall_with_optional_gap(
                    ax,
                    start[0],
                    start[1],
                    end[0],
                    end[1],
                    (cell_x, cell_y, direction) in doors,
                    color="black",
                    linewidth=2.0,
                    solid_capstyle="butt",
                )

        if show_names:
            xs = [room_x + x for x, _ in cells]
            ys = [room_y + y for _, y in cells]
            label = room.get("name", str(room.get("room_id", room_idx)))
            ax.text(
                (min(xs) + max(xs) + 1) / 2,
                (min(ys) + max(ys) + 1) / 2,
                label,
                ha="center",
                va="center",
                fontsize=8,
                clip_on=True,
            )

    ax.set_xlim(min_x - 1, max_x + 1)
    ax.set_ylim(max_y + 1, min_y - 1)
    ax.set_aspect("equal")
    ax.set_xticks(range(min_x - 1, max_x + 2))
    ax.set_yticks(range(min_y - 1, max_y + 2))
    ax.grid(color="#dddddd", linewidth=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax
