from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


Action = Tuple[int, int, int]
_ROOM_GEOMETRY_CACHE: Dict[int, List[dict]] = {}
_COLORS = [
    "#8ecae6",
    "#ffb703",
    "#90be6d",
    "#f28482",
    "#b8a1ff",
    "#f4a261",
    "#7bdff2",
    "#cdb4db",
]


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


def _occupied_cells(room: dict) -> Set[Tuple[int, int]]:
    return {
        (x, y)
        for y, row in enumerate(room["map"])
        for x, value in enumerate(row)
        if value
    }


def _door_sides(room: dict) -> Set[Tuple[int, int, str]]:
    return {
        (int(door["x"]), int(door["y"]), door["direction"])
        for door_group in room.get("doors", [])
        for door in door_group
    }


def _room_geometries(rooms: Sequence[dict]) -> List[dict]:
    cache_key = id(rooms)
    cached = _ROOM_GEOMETRY_CACHE.get(cache_key)
    if cached is not None and len(cached) == len(rooms):
        return cached

    geometries = []
    for room_idx, room in enumerate(rooms):
        cells = _occupied_cells(room)
        doors = _door_sides(room)
        xs = [x for x, _ in cells] or [0]
        ys = [y for _, y in cells] or [0]
        geometries.append(
            {
                "cells": cells,
                "doors": doors,
                "label": room.get("name", str(room.get("room_id", room_idx))),
                "min_x": min(xs),
                "max_x": max(xs) + 1,
                "min_y": min(ys),
                "max_y": max(ys) + 1,
            }
        )

    _ROOM_GEOMETRY_CACHE[cache_key] = geometries
    return geometries


def _add_wall_segments(
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    gap: bool,
) -> None:
    if not gap:
        segments.append(((x1, y1), (x2, y2)))
        return

    gap_fraction = 0.62
    trim = (1.0 - gap_fraction) / 2.0
    dx = x2 - x1
    dy = y2 - y1
    segments.append(((x1, y1), (x1 + dx * trim, y1 + dy * trim)))
    segments.append(((x2 - dx * trim, y2 - dy * trim), (x2, y2)))


def _placement_elements(
    geometry: dict,
    room_x: int,
    room_y: int,
    fill: str,
) -> Tuple[List[Tuple[Tuple[int, int], ...]], List[str], List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    room_polygons = []
    room_colors = []
    wall_segments = []
    cells = geometry["cells"]
    doors = geometry["doors"]

    for cell_x, cell_y in cells:
        x = room_x + cell_x
        y = room_y + cell_y
        room_polygons.append(((x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)))
        room_colors.append(fill)

        sides = {
            "left": ((x, y), (x, y + 1), (cell_x - 1, cell_y)),
            "right": ((x + 1, y), (x + 1, y + 1), (cell_x + 1, cell_y)),
            "up": ((x, y), (x + 1, y), (cell_x, cell_y - 1)),
            "down": ((x, y + 1), (x + 1, y + 1), (cell_x, cell_y + 1)),
        }
        for direction, (start, end, neighbor) in sides.items():
            if neighbor in cells:
                continue
            _add_wall_segments(
                wall_segments,
                start[0],
                start[1],
                end[0],
                end[1],
                (cell_x, cell_y, direction) in doors,
            )

    return room_polygons, room_colors, wall_segments


def _room_center(geometry: dict, room_x: int, room_y: int) -> Tuple[float, float]:
    cells = geometry["cells"]
    xs = [room_x + x for x, _ in cells]
    ys = [room_y + y for _, y in cells]
    return (min(xs) + max(xs) + 1) / 2, (min(ys) + max(ys) + 1) / 2


class MapVisualizer:
    """Incrementally update a map plot as rooms are placed."""

    def __init__(
        self,
        rooms: Sequence[dict],
        *,
        ax: Optional[Any] = None,
        map_size: Optional[Tuple[int, int]] = None,
        interactive: bool = False,
        show_names: bool = False,
        show_count: bool = True,
    ) -> None:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.collections import LineCollection, PolyCollection
        except ImportError as exc:
            raise ImportError("MapVisualizer requires matplotlib") from exc

        self.LineCollection = LineCollection
        self.PolyCollection = PolyCollection
        self.plt = plt
        self.interactive = interactive
        if interactive:
            plt.ion()
        self.rooms = rooms
        self.geometries = _room_geometries(rooms)
        self.ax = ax if ax is not None else plt.subplots()[1]
        self.fig = self.ax.figure
        self.map_size = map_size
        self.show_names = show_names
        self.show_count = show_count
        self.placement_count = 0
        self.room_polygons: List[Tuple[Tuple[int, int], ...]] = []
        self.room_colors: List[str] = []
        self.wall_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        self.label_artists = []
        self.dirty = False
        self.room_collection = PolyCollection(
            [],
            edgecolors="none",
            alpha=0.45,
        )
        self.wall_collection = LineCollection(
            [],
            colors="black",
            linewidths=2.0,
            capstyle="butt",
        )
        self.ax.add_collection(self.room_collection)
        self.ax.add_collection(self.wall_collection)
        self.count_text = None
        if show_count:
            self.count_text = self.ax.text(
                0.01,
                0.99,
                "Rooms: 0",
                transform=self.ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
            )

        self.min_x: Optional[int] = None
        self.max_x: Optional[int] = None
        self.min_y: Optional[int] = None
        self.max_y: Optional[int] = None
        self._configure_axes()
        if interactive:
            plt.show(block=False)

    def add_placements(self, actions: Any, *, environment_index: int = 0) -> None:
        for action in _normalize_actions(actions, environment_index):
            self.add_placement(action)

    def add_engine_actions(self, actions: Any, *, environment_index: int = 0) -> None:
        self.add_placements(actions, environment_index=environment_index)
        self.sync_artists()

    def add_selected_candidate(
        self,
        room_idx: Any,
        room_x: Any,
        room_y: Any,
        *,
        environment_index: int = 0,
    ) -> Action:
        action = (
            int(room_idx[environment_index]),
            int(room_x[environment_index]),
            int(room_y[environment_index]),
        )
        self.add_placement(action)
        return action

    def add_placement(self, action: Action) -> None:
        room_idx, room_x, room_y = action
        if not 0 <= room_idx < len(self.rooms):
            return

        geometry = self.geometries[room_idx]
        fill = _COLORS[self.placement_count % len(_COLORS)]
        polygons, colors, segments = _placement_elements(geometry, room_x, room_y, fill)
        self.room_polygons.extend(polygons)
        self.room_colors.extend(colors)
        self.wall_segments.extend(segments)
        self.placement_count += 1
        self.dirty = True

        if self.show_names:
            cells = geometry["cells"]
            xs = [room_x + x for x, _ in cells]
            ys = [room_y + y for _, y in cells]
            self.label_artists.append(
                self.ax.text(
                    (min(xs) + max(xs) + 1) / 2,
                    (min(ys) + max(ys) + 1) / 2,
                    geometry["label"],
                    ha="center",
                    va="center",
                    fontsize=8,
                    clip_on=True,
                )
            )

        self._update_bounds(room_idx, room_x, room_y)
        if self.count_text is not None:
            self.count_text.set_text(f"Rooms: {self.placement_count}")

    def sync_artists(self) -> None:
        if not self.dirty:
            return
        self.room_collection.set_verts(self.room_polygons)
        self.room_collection.set_facecolors(self.room_colors)
        self.wall_collection.set_segments(self.wall_segments)
        self.dirty = False

    def update(self, pause: float = 0.0) -> None:
        self.sync_artists()
        self.fig.canvas.draw_idle()
        if pause > 0:
            self.plt.pause(pause)

    def show(self, *, block: bool = True) -> None:
        if self.interactive:
            self.plt.ioff()
        self.plt.show(block=block)

    def _update_bounds(self, room_idx: int, room_x: int, room_y: int) -> None:
        if self.map_size is not None:
            return

        geometry = self.geometries[room_idx]
        min_x = room_x + geometry["min_x"]
        max_x = room_x + geometry["max_x"]
        min_y = room_y + geometry["min_y"]
        max_y = room_y + geometry["max_y"]
        self.min_x = min_x if self.min_x is None else min(self.min_x, min_x)
        self.max_x = max_x if self.max_x is None else max(self.max_x, max_x)
        self.min_y = min_y if self.min_y is None else min(self.min_y, min_y)
        self.max_y = max_y if self.max_y is None else max(self.max_y, max_y)
        self.ax.set_xlim(self.min_x - 1, self.max_x + 1)
        self.ax.set_ylim(self.max_y + 1, self.min_y - 1)

    def _configure_axes(self) -> None:
        self.ax.set_aspect("equal")
        self.ax.grid(color="#dddddd", linewidth=0.6)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        if self.map_size is not None:
            width, height = self.map_size
            self.ax.set_xlim(-1, width + 1)
            self.ax.set_ylim(height + 1, -1)
            self.ax.set_xticks(range(0, width + 1, 4))
            self.ax.set_yticks(range(0, height + 1, 4))


def display_map(
    rooms: Sequence[dict],
    actions: Any,
    *,
    environment_index: int = 0,
    ax: Optional[Any] = None,
    show_names: bool = True,
    show_count: bool = True,
) -> Any:
    """Display room placements returned by ``engine.get_actions()``.

    ``rooms`` is the parsed room JSON. ``actions`` may be the raw
    ``engine.get_actions()`` tuple of ``(room_idx, x, y)`` arrays, or any
    iterable of ``(room_idx, x, y)`` placements.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection, PolyCollection
    except ImportError as exc:
        raise ImportError("display_map requires matplotlib") from exc

    geometries = _room_geometries(rooms)
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

    min_x = min(x + geometries[room_idx]["min_x"] for room_idx, x, _ in placements)
    max_x = max(x + geometries[room_idx]["max_x"] for room_idx, x, _ in placements)
    min_y = min(y + geometries[room_idx]["min_y"] for room_idx, _, y in placements)
    max_y = max(y + geometries[room_idx]["max_y"] for room_idx, _, y in placements)

    room_polygons = []
    room_colors = []
    wall_segments = []
    label_specs = []

    for placement_index, (room_idx, room_x, room_y) in enumerate(placements):
        geometry = geometries[room_idx]
        cells = geometry["cells"]
        polygons, colors, segments = _placement_elements(
            geometry,
            room_x,
            room_y,
            _COLORS[placement_index % len(_COLORS)],
        )
        room_polygons.extend(polygons)
        room_colors.extend(colors)
        wall_segments.extend(segments)

        if show_names:
            xs = [room_x + x for x, _ in cells]
            ys = [room_y + y for _, y in cells]
            label_specs.append(
                (
                    (min(xs) + max(xs) + 1) / 2,
                    (min(ys) + max(ys) + 1) / 2,
                    geometry["label"],
                )
            )

    ax.add_collection(
        PolyCollection(
            room_polygons,
            facecolors=room_colors,
            edgecolors="none",
            alpha=0.45,
        )
    )
    ax.add_collection(
        LineCollection(
            wall_segments,
            colors="black",
            linewidths=2.0,
            capstyle="butt",
        )
    )

    for label_x, label_y, label in label_specs:
        ax.text(
            label_x,
            label_y,
            label,
            ha="center",
            va="center",
            fontsize=8,
            clip_on=True,
        )

    if show_count:
        ax.text(
            0.01,
            0.99,
            f"Rooms: {len(placements)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
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


def save_episode_frames(
    rooms: Sequence[dict],
    actions: Any,
    output_dir: Path,
    map_size: Tuple[int, int],
    environment_index: int,
) -> List[Path]:
    """Save one PNG after each placement in a generated episode."""
    from PIL import Image, ImageDraw

    output_dir.mkdir(parents=True, exist_ok=True)
    scale = 12
    margin = 36
    width, height = map_size
    canvas_size = (width * scale + 2 * margin, height * scale + 2 * margin)
    geometries = _room_geometries(rooms)
    room_polygons: List[Tuple[Tuple[int, int], ...]] = []
    room_colors: List[str] = []
    wall_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    saved_paths = []
    placements = _normalize_actions(actions, environment_index)
    save_label_positions: List[Tuple[float, float]] = []
    for step_idx, action in enumerate(placements):
        room_idx, room_x, room_y = action
        if not 0 <= room_idx < len(rooms):
            continue

        geometry = geometries[room_idx]
        polygons, colors, segments = _placement_elements(
            geometry,
            room_x,
            room_y,
            _COLORS[step_idx % len(_COLORS)],
        )
        room_polygons.extend(polygons)
        room_colors.extend(colors)
        wall_segments.extend(segments)
        if rooms[room_idx].get("save", False):
            save_label_positions.append(_room_center(geometry, room_x, room_y))

        image = Image.new("RGB", canvas_size, "white")
        draw = ImageDraw.Draw(image)
        for grid_x in range(width + 1):
            pixel_x = margin + grid_x * scale
            draw.line([(pixel_x, margin), (pixel_x, margin + height * scale)], fill="#dddddd")
        for grid_y in range(height + 1):
            pixel_y = margin + grid_y * scale
            draw.line([(margin, pixel_y), (margin + width * scale, pixel_y)], fill="#dddddd")

        for polygon, color in zip(room_polygons, room_colors):
            draw.polygon(
                [(margin + x * scale, margin + y * scale) for x, y in polygon],
                fill=color,
            )
        for (x1, y1), (x2, y2) in wall_segments:
            draw.line(
                [
                    (margin + x1 * scale, margin + y1 * scale),
                    (margin + x2 * scale, margin + y2 * scale),
                ],
                fill="black",
                width=2,
            )
        for label_x, label_y in save_label_positions:
            pixel_x = margin + label_x * scale
            pixel_y = margin + label_y * scale
            text_bbox = draw.textbbox((0, 0), "S")
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.text(
                (pixel_x - text_width / 2, pixel_y - text_height / 2),
                "S",
                fill="black",
            )
        frame_path = output_dir / f"step_{step_idx + 1:03d}.png"
        image.save(frame_path)
        saved_paths.append(frame_path)

    return saved_paths
