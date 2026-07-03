from typing import Any

import torch

from env import Engine
from model import BalanceModel
from train_config import Config


def without_prefix(
    tensors: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    full_prefix = f"{prefix}."
    compiled_prefix = f"{full_prefix}_orig_mod."
    return {
        (
            name[len(compiled_prefix) :]
            if name.startswith(compiled_prefix)
            else name[len(full_prefix) :]
        ): value
        for name, value in tensors.items()
        if name.startswith(full_prefix)
    }


def frontier_model_kwargs(
    config: Config,
    rooms: list[dict],
    engine: Engine,
) -> dict[str, Any]:
    return {
        "num_rooms": len(rooms),
        "output_metadata": engine.get_output_metadata(),
        "map_x": config.map_size[0],
        "map_y": config.map_size[1],
        "embedding_width": config.model.embedding_width,
        "global_embedding_width": config.model.global_embedding_width,
        "hidden_width": config.model.hidden_width,
        "proposal_hidden_width": config.model.proposal_hidden_width,
        "missing_connect_query_hidden_width": config.model.missing_connect_query_hidden_width,
        "missing_connect_query_frontier_width": config.model.missing_connect_query_frontier_width,
        "missing_connect_query_distance_width": config.model.missing_connect_query_distance_width,
        "utility_query_hidden_width": config.model.utility_query_hidden_width,
        "utility_query_frontier_width": config.model.utility_query_frontier_width,
        "known_save_refill_utility_override": config.model.known_save_refill_utility_override,
        "distance_proximity_scale": config.distance_proximity_scale,
        "num_layers": config.model.num_layers,
        "door_counts": (
            count_room_doors_by_direction(rooms, "left"),
            count_room_doors_by_direction(rooms, "right"),
            count_room_doors_by_direction(rooms, "up"),
            count_room_doors_by_direction(rooms, "down"),
        ),
        "frontier_window_size": config.generation.frontier_window_size,
        "features": config.features,
    }


def count_room_doors_by_direction(rooms: list[dict], direction: str) -> int:
    return sum(
        1
        for room in rooms
        for door_group in room["doors"]
        for door in door_group
        if door["direction"] == direction
    )


def create_balance_model(
    config: Config, rooms: list[dict], device: torch.device
) -> torch.nn.Module:
    return BalanceModel(
        left_count=count_room_doors_by_direction(rooms, "left"),
        right_count=count_room_doors_by_direction(rooms, "right"),
        up_count=count_room_doors_by_direction(rooms, "up"),
        down_count=count_room_doors_by_direction(rooms, "down"),
        num_rooms=len(rooms),
        hidden_width=config.balance_model.hidden_width,
        num_layers=config.balance_model.num_layers,
    ).to(device)
