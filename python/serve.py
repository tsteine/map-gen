from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

import torch
from flask import Flask, Response, jsonify, request
from pydantic import BaseModel, ConfigDict, ValidationError
from safetensors import safe_open
from werkzeug.exceptions import BadRequest

from area_assignment import (
    DoorRoomLookup,
    MapStationData,
    RoomGeometry,
    ToiletData,
    assign_room_areas,
    build_door_room_lookup,
    build_map_station_data,
    build_room_geometry,
    build_toilet_data,
)
from env import DoorMatches, Engine, EpisodeData, EpisodeOutcomes, GenerateConfig
from generate import GenerationProfiler, profile_start, run_generation_groups, sync_profile_device
from model import FrontierModel
from small_map import (
    DoorData,
    RequiredRoomData,
    RoomPartData,
    SmallMapConfig,
    build_door_data,
    build_required_room_data,
    build_room_part_data,
    prune_small_maps,
)
from model_loading import create_balance_model, frontier_model_kwargs, without_prefix
from train_config import Config, GENERATION_VARIABLE_FLOAT_FIELDS, validate_config


MODEL_EXPORT_FORMAT = "map-gen-model-export-v1"
TRAINING_CHECKPOINT_FORMAT = "map-gen-training-session-checkpoint-v3"
MODEL_INPUT_FORMATS = (MODEL_EXPORT_FORMAT, TRAINING_CHECKPOINT_FORMAT)
MODEL_PREFIXES = ("ema_model", "balance_model")

DIRECTIONS = ("left", "right", "up", "down")
OPPOSITE_DIRECTIONS = {
    "left": "right",
    "right": "left",
    "up": "down",
    "down": "up",
}

app = Flask(__name__)
app.json.sort_keys = False  # Disables alphabetical key sorting

class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServingConfig(StrictBaseModel):
    host: str
    port: int
    device: str
    compile_model: bool
    cuda_memory_fraction: float
    model_dtype: str
    autocast: bool
    verify_outcome_consistency: bool
    gpu_prefetch_batches: int
    num_warmup_requests: int
    prefetch_queue_max_size: int
    prefetch_max_queues: int
    prefetch_delay_seconds: float
    area_assignment_attempts: int
    area_bounding_box_width: int
    area_bounding_box_height: int
    area_min_rooms: int
    area_max_rooms: int
    room_set: Path
    num_environments: int
    pipeline_groups: int
    num_threads: int


class GenerateRequest(StrictBaseModel):
    episode_length: int
    recommended_candidates: int
    shortlist_candidates: int
    temperature: float
    proposal_temperature: float
    reward_door: float
    reward_connection: float
    reward_toilet: float
    reward_phantoon: float
    reward_balance: float
    reward_toilet_balance: float
    reward_frontier: float
    reward_graph_diameter: float
    reward_save_distance: float
    reward_refill_distance: float
    reward_missing_connect_utility: float
    area_assignment_base_order: Literal["random", "depth", "size"]
    small_map: bool
    min_rooms: int | None = None
    max_rooms: int | None = None
    target_rooms: int | None = None


@dataclass
class ModelExport:
    training_config: Config
    tensors: dict[str, torch.Tensor]


class DoorLookup(NamedTuple):
    room_idx: list[int]
    door_id: list[int]


@dataclass
class DoorLookups:
    left: DoorLookup
    right: DoorLookup
    up: DoorLookup
    down: DoorLookup


@dataclass
class GenerationRunResult:
    episode_data: EpisodeData
    outcomes: EpisodeOutcomes
    door_matches: DoorMatches
    profile_report: list[tuple[str, int, int]]


@dataclass
class PrefetchQueueState:
    request: GenerateRequest
    responses: deque[str]
    refill_debt: int


@dataclass
class PrefetchState:
    queues: OrderedDict[str, PrefetchQueueState]
    condition: threading.Condition
    refill_scheduled: bool
    due_time: float
    schedule_version: int
    foreground_waiting: int


@dataclass
class ServingState:
    serving_config: ServingConfig
    training_config: Config
    rooms: list[dict]
    device: torch.device
    envs: list
    model: torch.nn.Module
    balance_model: torch.nn.Module
    door_room_lookup: DoorRoomLookup
    room_geometry: RoomGeometry
    map_station_data: MapStationData
    toilet_data: ToiletData
    door_lookups: DoorLookups
    door_data: DoorData
    room_part_data: RoomPartData
    required_room_data: RequiredRoomData
    profile: bool
    lock: threading.Lock
    prefetch: PrefetchState | None


SERVING_STATE: ServingState | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve map generation requests from an exported model safetensors file.",
    )
    parser.add_argument(
        "serving_config",
        type=Path,
        help="Serving config JSON file.",
    )
    parser.add_argument(
        "model_export",
        type=Path,
        help="Model export or training checkpoint safetensors file.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Collect and return Python/Rust generation profile timings.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Use deterministic environment group seeds starting from this value.",
    )
    return parser.parse_args()


def load_serving_config(path: Path) -> ServingConfig:
    return ServingConfig.model_validate_json(path.read_text())


def validate_model_input_metadata(path: Path, metadata: dict[str, str] | None) -> dict[str, str]:
    if metadata is None:
        raise ValueError(f"model input metadata missing in {path}")
    for field in ("format", "config"):
        if field not in metadata:
            raise ValueError(f"model input metadata field {field!r} missing in {path}")
    if metadata["format"] not in MODEL_INPUT_FORMATS:
        raise ValueError(f"unsupported model input format in {path}")
    return metadata


def load_model_input(path: Path) -> ModelExport:
    with safe_open(path, framework="pt", device="cpu") as model_input:
        metadata = validate_model_input_metadata(path, model_input.metadata())
        tensors = {name: model_input.get_tensor(name) for name in model_input.keys()}
    missing_prefixes = [
        prefix
        for prefix in MODEL_PREFIXES
        if not any(name.startswith(f"{prefix}.") for name in tensors)
    ]
    if missing_prefixes:
        raise ValueError(f"model input missing tensor group(s): {', '.join(missing_prefixes)}")
    training_config = Config.model_validate_json(metadata["config"])
    validate_config(training_config)
    return ModelExport(training_config=training_config, tensors=tensors)


def create_environment_groups(
    serving_config: ServingConfig,
    training_config: Config,
    engine: Engine,
    seed: int | None,
) -> list:
    validate_serving_config(serving_config)
    generation_config = training_config.generation
    group_environments = serving_config.num_environments // serving_config.pipeline_groups
    group_threads = serving_config.num_threads // serving_config.pipeline_groups
    return [
        engine.create_environment_group(
            training_config.map_size,
            group_environments,
            generation_config.candidate_spatial_cell_size,
            seed=None if seed is None else seed + group_index,
            frontier_neighbor_algorithm=generation_config.frontier_neighbor_algorithm,
            frontier_neighbor_count=generation_config.frontier_neighbor_count,
            frontier_window_size=generation_config.frontier_window_size,
            num_threads=group_threads,
        )
        for group_index in range(serving_config.pipeline_groups)
    ]


def validate_serving_config(serving_config: ServingConfig) -> None:
    if serving_config.num_environments <= 0:
        raise ValueError("num_environments must be greater than zero")
    if serving_config.pipeline_groups <= 0:
        raise ValueError("pipeline_groups must be greater than zero")
    if serving_config.num_environments % serving_config.pipeline_groups != 0:
        raise ValueError(
            "num_environments must be divisible by pipeline_groups"
        )
    if serving_config.num_threads <= 0:
        raise ValueError("num_threads must be greater than zero")
    if serving_config.num_threads // serving_config.pipeline_groups <= 0:
        raise ValueError("num_threads must be at least pipeline_groups")
    if serving_config.gpu_prefetch_batches < 0:
        raise ValueError("gpu_prefetch_batches must be greater than or equal to zero")
    if serving_config.num_warmup_requests < 0:
        raise ValueError("num_warmup_requests must be greater than or equal to zero")
    if serving_config.prefetch_queue_max_size < 0:
        raise ValueError("prefetch_queue_max_size must be greater than or equal to zero")
    if serving_config.prefetch_max_queues < 0:
        raise ValueError("prefetch_max_queues must be greater than or equal to zero")
    if serving_config.prefetch_delay_seconds < 0:
        raise ValueError("prefetch_delay_seconds must be greater than or equal to zero")
    if serving_config.area_assignment_attempts <= 0:
        raise ValueError("area_assignment_attempts must be greater than zero")
    if serving_config.area_bounding_box_width <= 0:
        raise ValueError("area_bounding_box_width must be greater than zero")
    if serving_config.area_bounding_box_height <= 0:
        raise ValueError("area_bounding_box_height must be greater than zero")
    if serving_config.area_min_rooms <= 0:
        raise ValueError("area_min_rooms must be greater than zero")
    if serving_config.area_max_rooms < serving_config.area_min_rooms:
        raise ValueError("area_max_rooms must be at least area_min_rooms")


def serving_model_dtype(serving_config: ServingConfig) -> torch.dtype:
    if serving_config.model_dtype == "float32":
        return torch.float32
    if serving_config.model_dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError('model_dtype must be "float32" or "bfloat16"')


def create_prefetch_state() -> PrefetchState:
    lock = threading.Lock()
    return PrefetchState(
        queues=OrderedDict(),
        condition=threading.Condition(lock),
        refill_scheduled=False,
        due_time=0.0,
        schedule_version=0,
        foreground_waiting=0,
    )


def prefetch_enabled(serving_config: ServingConfig) -> bool:
    return (
        serving_config.prefetch_queue_max_size > 0
        and serving_config.prefetch_max_queues > 0
    )


def create_serving_state(
    serving_config: ServingConfig,
    model_export: ModelExport,
    seed: int | None,
    profile: bool,
) -> ServingState:
    rooms = json.loads(serving_config.room_set.read_text())
    device = torch.device(serving_config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.memory.set_per_process_memory_fraction(
            serving_config.cuda_memory_fraction,
            device,
        )
        torch.set_float32_matmul_precision("high")
    model_dtype = serving_model_dtype(serving_config)
    engine = Engine(rooms, model_export.training_config.features)
    model = FrontierModel(**frontier_model_kwargs(model_export.training_config, rooms, engine)).to(
        device
    )
    model.load_state_dict(without_prefix(model_export.tensors, "ema_model"))
    model.to(dtype=model_dtype)
    model.requires_grad_(False)
    model.eval()
    balance_model = create_balance_model(model_export.training_config, rooms, device)
    balance_model.load_state_dict(without_prefix(model_export.tensors, "balance_model"))
    balance_model.to(dtype=model_dtype)
    balance_model.requires_grad_(False)
    balance_model.eval()
    if serving_config.compile_model:
        model = torch.compile(model)
        balance_model = torch.compile(balance_model)
    envs = create_environment_groups(serving_config, model_export.training_config, engine, seed)
    door_room_lookup = build_door_room_lookup(rooms, device)
    room_geometry = build_room_geometry(rooms, device)
    map_station_data = build_map_station_data(rooms, device)
    toilet_data = build_toilet_data(rooms, device)
    door_lookups = build_door_lookups(rooms)
    door_data = build_door_data(rooms)
    room_part_data = build_room_part_data(rooms)
    required_room_data = build_required_room_data(rooms)
    prefetch = create_prefetch_state() if prefetch_enabled(serving_config) else None
    state = ServingState(
        serving_config=serving_config,
        training_config=model_export.training_config,
        rooms=rooms,
        device=device,
        envs=envs,
        model=model,
        balance_model=balance_model,
        door_room_lookup=door_room_lookup,
        room_geometry=room_geometry,
        map_station_data=map_station_data,
        toilet_data=toilet_data,
        door_lookups=door_lookups,
        door_data=door_data,
        room_part_data=room_part_data,
        required_room_data=required_room_data,
        profile=profile,
        lock=threading.Lock(),
        prefetch=prefetch,
    )
    if prefetch is not None:
        threading.Thread(
            target=run_prefetch_worker,
            args=(state,),
            daemon=True,
        ).start()
    return state


def validate_generate_request(generate_request: GenerateRequest, rooms: list[dict]) -> None:
    if generate_request.episode_length <= 0:
        raise ValueError("episode_length must be greater than zero")
    if generate_request.episode_length > len(rooms):
        raise ValueError("episode_length must not exceed the room count")
    if generate_request.recommended_candidates <= 0:
        raise ValueError("recommended_candidates must be greater than zero")
    if generate_request.shortlist_candidates < generate_request.recommended_candidates:
        raise ValueError("shortlist_candidates must be at least recommended_candidates")
    if generate_request.temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if generate_request.proposal_temperature <= 0:
        raise ValueError("proposal_temperature must be greater than zero")
    if generate_request.small_map:
        missing_fields = [
            field
            for field in ("min_rooms", "max_rooms", "target_rooms")
            if getattr(generate_request, field) is None
        ]
        if missing_fields:
            raise ValueError(
                f"small_map requires {', '.join(missing_fields)}"
            )
        assert generate_request.min_rooms is not None
        assert generate_request.max_rooms is not None
        assert generate_request.target_rooms is not None
        if generate_request.min_rooms <= 0:
            raise ValueError("min_rooms must be greater than zero")
        if generate_request.max_rooms < generate_request.min_rooms:
            raise ValueError("max_rooms must be at least min_rooms")
        if generate_request.target_rooms <= 0:
            raise ValueError("target_rooms must be greater than zero")


def create_generate_configs(
    generate_request: GenerateRequest,
    state: ServingState,
    envs: list,
    device: torch.device,
) -> list[GenerateConfig]:
    generation_variable_float_values = {
        "temperature": generate_request.temperature,
        "proposal_temperature": generate_request.proposal_temperature,
        "reward_door": generate_request.reward_door,
        "reward_connection": generate_request.reward_connection,
        "reward_toilet": generate_request.reward_toilet,
        "reward_phantoon": generate_request.reward_phantoon,
        "reward_balance": generate_request.reward_balance,
        "reward_toilet_balance": generate_request.reward_toilet_balance,
        "reward_frontier": generate_request.reward_frontier,
        "reward_graph_diameter": generate_request.reward_graph_diameter,
        "reward_save_distance": generate_request.reward_save_distance,
        "reward_refill_distance": generate_request.reward_refill_distance,
        "reward_missing_connect_utility": generate_request.reward_missing_connect_utility,
    }
    return [
        GenerateConfig(
            episode_length=generate_request.episode_length,
            recommended_candidates=generate_request.recommended_candidates,
            shortlist_candidates=generate_request.shortlist_candidates,
            gpu_prefetch_batches=state.serving_config.gpu_prefetch_batches,
            temperature=torch.full(
                [env.num_envs],
                generate_request.temperature,
                dtype=torch.float32,
                device=device,
            ),
            proposal_temperature=torch.full(
                [env.num_envs],
                generate_request.proposal_temperature,
                dtype=torch.float32,
                device=device,
            ),
            reward_door=generate_request.reward_door,
            reward_connection=generate_request.reward_connection,
            reward_toilet=generate_request.reward_toilet,
            reward_phantoon=generate_request.reward_phantoon,
            reward_balance=generate_request.reward_balance,
            reward_toilet_balance=generate_request.reward_toilet_balance,
            reward_frontier=generate_request.reward_frontier,
            reward_graph_diameter=generate_request.reward_graph_diameter,
            reward_save_distance=generate_request.reward_save_distance,
            reward_refill_distance=generate_request.reward_refill_distance,
            reward_missing_connect_utility=generate_request.reward_missing_connect_utility,
            generation_variable_floats=torch.tensor(
                [
                    [
                        generation_variable_float_values[name]
                        for name in GENERATION_VARIABLE_FLOAT_FIELDS
                    ]
                ],
                dtype=torch.float32,
                device=device,
            ).expand(env.num_envs, len(GENERATION_VARIABLE_FLOAT_FIELDS)),
            distance_proximity_scale=state.training_config.distance_proximity_scale,
            autocast=state.serving_config.autocast,
        )
        for env in envs
    ]


def tensor_to_list(tensor: torch.Tensor) -> list:
    return tensor.detach().cpu().tolist()


def flat_doors(room: dict) -> list[dict]:
    return [door for door_group in room["doors"] for door in door_group]


def build_door_lookups(rooms: list[dict]) -> DoorLookups:
    door_rooms = {direction: [] for direction in DIRECTIONS}
    door_ids = {direction: [] for direction in DIRECTIONS}
    for room_idx, room in enumerate(rooms):
        for door in flat_doors(room):
            direction = door["direction"]
            door_rooms[direction].append(room_idx)
            door_ids[direction].append(int(door["id"]))
    return DoorLookups(
        left=DoorLookup(room_idx=door_rooms["left"], door_id=door_ids["left"]),
        right=DoorLookup(room_idx=door_rooms["right"], door_id=door_ids["right"]),
        up=DoorLookup(room_idx=door_rooms["up"], door_id=door_ids["up"]),
        down=DoorLookup(room_idx=door_rooms["down"], door_id=door_ids["down"]),
    )


def tensor_has_invalid_outcome(tensor: torch.Tensor) -> torch.Tensor:
    invalid = tensor > 0
    if invalid.ndim == 1:
        return invalid
    return torch.any(invalid, dim=tuple(range(1, invalid.ndim)))


def valid_map_mask(outcomes) -> torch.Tensor:
    step_outcomes = outcomes.step_outcomes
    invalid = (
        tensor_has_invalid_outcome(step_outcomes.door_invalid)
        | tensor_has_invalid_outcome(step_outcomes.connection_invalid)
        | tensor_has_invalid_outcome(step_outcomes.toilet_invalid)
        | tensor_has_invalid_outcome(step_outcomes.phantoon_invalid)
    )
    return ~invalid


def collect_door_matches(envs: list, device: torch.device) -> DoorMatches:
    group_door_matches = [env.get_door_matches(device) for env in envs]
    return DoorMatches(
        left=torch.cat([door_matches.left for door_matches in group_door_matches]),
        right=torch.cat([door_matches.right for door_matches in group_door_matches]),
        up=torch.cat([door_matches.up for door_matches in group_door_matches]),
        down=torch.cat([door_matches.down for door_matches in group_door_matches]),
    )


def filter_door_matches(door_matches: DoorMatches, mask: torch.Tensor) -> DoorMatches:
    return DoorMatches(
        left=door_matches.left[mask],
        right=door_matches.right[mask],
        up=door_matches.up[mask],
        down=door_matches.down[mask],
    )


def select_door_matches(door_matches: DoorMatches, index: torch.Tensor) -> DoorMatches:
    return DoorMatches(
        left=door_matches.left[index],
        right=door_matches.right[index],
        up=door_matches.up[index],
        down=door_matches.down[index],
    )


def placement_positions_by_room(room_idx: list[int]) -> dict[int, int]:
    return {room: position for position, room in enumerate(room_idx)}


def append_edge(
    edges: dict[str, list[int]],
    seen_edges: set[tuple[int, int, int, int]],
    from_endpoint: tuple[int, int],
    to_endpoint: tuple[int, int],
) -> None:
    if to_endpoint < from_endpoint:
        from_endpoint, to_endpoint = to_endpoint, from_endpoint
    edge = (*from_endpoint, *to_endpoint)
    if edge in seen_edges:
        return
    seen_edges.add(edge)
    edges["from_room_placement_idx"].append(from_endpoint[0])
    edges["from_door_id"].append(from_endpoint[1])
    edges["to_room_placement_idx"].append(to_endpoint[0])
    edges["to_door_id"].append(to_endpoint[1])


def collect_response_direction_edges(
    edges: dict[str, list[int]],
    seen_edges: set[tuple[int, int, int, int]],
    source_matches: list[int],
    source_lookup: DoorLookup,
    target_lookup: DoorLookup,
    room_position: dict[int, int],
) -> None:
    for source_direction_door_idx, target_direction_door_idx in enumerate(source_matches):
        if target_direction_door_idx < 0:
            continue
        source_room_idx = source_lookup.room_idx[source_direction_door_idx]
        target_room_idx = target_lookup.room_idx[target_direction_door_idx]
        source_room_position = room_position.get(source_room_idx)
        target_room_position = room_position.get(target_room_idx)
        if source_room_position is None or target_room_position is None:
            continue
        append_edge(
            edges,
            seen_edges,
            (source_room_position, source_lookup.door_id[source_direction_door_idx]),
            (target_room_position, target_lookup.door_id[target_direction_door_idx]),
        )


def response_edges(
    room_idx: list[list[int]],
    door_matches: DoorMatches,
    door_lookups: DoorLookups,
) -> dict[str, list[list[int]]]:
    edge_lists = {
        "from_room_placement_idx": [],
        "from_door_id": [],
        "to_room_placement_idx": [],
        "to_door_id": [],
    }
    direction_matches = {
        "left": tensor_to_list(door_matches.left),
        "right": tensor_to_list(door_matches.right),
        "up": tensor_to_list(door_matches.up),
        "down": tensor_to_list(door_matches.down),
    }
    direction_lookups = {
        "left": door_lookups.left,
        "right": door_lookups.right,
        "up": door_lookups.up,
        "down": door_lookups.down,
    }
    for map_idx, map_room_idx in enumerate(room_idx):
        map_edges = {key: [] for key in edge_lists}
        seen_edges = set()
        room_position = placement_positions_by_room(map_room_idx)
        for direction in DIRECTIONS:
            collect_response_direction_edges(
                map_edges,
                seen_edges,
                direction_matches[direction][map_idx],
                direction_lookups[direction],
                direction_lookups[OPPOSITE_DIRECTIONS[direction]],
                room_position,
            )
        for key, values in map_edges.items():
            edge_lists[key].append(values)
    return edge_lists


def response_toilet_crossing_room_placement_idx(
    room_idx: list[list[int]],
    toilet_crossed_room_idx: list[int],
) -> list[int]:
    return [
        placement_positions_by_room(map_room_idx).get(crossed_room_idx, -1)
        for map_room_idx, crossed_room_idx in zip(room_idx, toilet_crossed_room_idx)
    ]

def response_room_ids(room_idx: list[list[int]], rooms: list[dict]) -> list[list[int]]:
    return [[int(rooms[room]["room_id"]) for room in map_room_idx] for map_room_idx in room_idx]


def initialize_serving_state(state: ServingState) -> None:
    global SERVING_STATE
    SERVING_STATE = state


def warmup_generate_request() -> GenerateRequest:
    return GenerateRequest(
        episode_length=253,
        recommended_candidates=4,
        shortlist_candidates=16,
        temperature=0.03,
        proposal_temperature=0.3,
        reward_door=1.0,
        reward_connection=1.0,
        reward_toilet=1.0,
        reward_phantoon=1.0,
        reward_balance=0.1,
        reward_toilet_balance=0.1,
        reward_frontier=0.0,
        reward_graph_diameter=0.1,
        reward_save_distance=0.1,
        reward_refill_distance=0.1,
        reward_missing_connect_utility=0.5,
        area_assignment_base_order="random",
        small_map=False,
    )


def run_warmup_requests(state: ServingState) -> None:
    for request_idx in range(state.serving_config.num_warmup_requests):
        logging.info(
            "Running warmup request %s/%s",
            request_idx + 1,
            state.serving_config.num_warmup_requests,
        )
        _ = generate_response_data_uncached_validated(state, warmup_generate_request())


def serving_state() -> ServingState:
    if SERVING_STATE is None:
        raise RuntimeError("serving state has not been initialized")
    return SERVING_STATE


def add_serving_profile(
    profiler: GenerationProfiler,
    device: torch.device,
    name: str,
    start: int,
) -> None:
    sync_profile_device(device, profiler.enabled)
    profiler.add(name, start)


def prefetch_request_key(generate_request: GenerateRequest) -> str:
    return generate_request.model_dump_json()


def serialize_generate_response(response: dict) -> str:
    return f"{app.json.dumps(response)}\n"


def touch_prefetch_queue(
    state: ServingState,
    key: str,
    generate_request: GenerateRequest,
) -> PrefetchQueueState:
    if state.prefetch is None:
        raise RuntimeError("prefetch is not enabled")
    prefetch = state.prefetch
    queue_state = prefetch.queues.get(key)
    if queue_state is None:
        queue_state = PrefetchQueueState(
            request=generate_request,
            responses=deque(),
            refill_debt=0,
        )
        prefetch.queues[key] = queue_state
        while len(prefetch.queues) > state.serving_config.prefetch_max_queues:
            prefetch.queues.popitem(last=False)
    else:
        prefetch.queues.move_to_end(key)
    return queue_state


def schedule_prefetch_refill(
    state: ServingState,
    key: str,
    generate_request: GenerateRequest,
) -> None:
    if state.prefetch is None:
        return
    prefetch = state.prefetch
    with prefetch.condition:
        queue_state = touch_prefetch_queue(state, key, generate_request)
        queue_state.refill_debt += 2
        prefetch.refill_scheduled = True
        prefetch.due_time = time.monotonic() + state.serving_config.prefetch_delay_seconds
        prefetch.schedule_version += 1
        prefetch.condition.notify()


def pop_prefetch_response(
    state: ServingState,
    key: str,
    generate_request: GenerateRequest,
) -> str | None:
    if state.prefetch is None:
        return None
    prefetch = state.prefetch
    with prefetch.condition:
        queue_state = touch_prefetch_queue(state, key, generate_request)
        if not queue_state.responses:
            return None
        return queue_state.responses.popleft()


def foreground_generation_waiting(state: ServingState) -> bool:
    if state.prefetch is None:
        return False
    return state.prefetch.foreground_waiting > 0


def acquire_foreground_generation_lock(state: ServingState) -> None:
    if state.prefetch is not None:
        with state.prefetch.condition:
            state.prefetch.foreground_waiting += 1
            state.prefetch.condition.notify()
    state.lock.acquire()
    if state.prefetch is not None:
        with state.prefetch.condition:
            state.prefetch.foreground_waiting -= 1
            state.prefetch.condition.notify()


def run_generation_with_generation_lock_held(
    state: ServingState,
    configs: list[GenerateConfig],
    serving_profiler: GenerationProfiler,
) -> GenerationRunResult:
    profile_time = profile_start(state.profile)
    (
        episode_data,
        outcomes,
        _door_match_counts,
        _proposal_data,
        _generation_stats,
        profile_report,
    ) = run_generation_groups(
        state.envs,
        state.model,
        state.balance_model,
        configs,
        state.device,
        verify_outcome_consistency=state.serving_config.verify_outcome_consistency,
        profile=state.profile,
    )
    add_serving_profile(
        serving_profiler,
        state.device,
        "python.serve.run_generation_groups",
        profile_time,
    )

    profile_time = profile_start(state.profile)
    door_matches = collect_door_matches(state.envs, state.device)
    add_serving_profile(
        serving_profiler,
        state.device,
        "python.serve.collect_door_matches",
        profile_time,
    )
    return GenerationRunResult(
        episode_data=episode_data,
        outcomes=outcomes,
        door_matches=door_matches,
        profile_report=profile_report,
    )


def run_foreground_generation(
    state: ServingState,
    configs: list[GenerateConfig],
    serving_profiler: GenerationProfiler,
) -> GenerationRunResult:
    acquire_foreground_generation_lock(state)
    try:
        with torch.inference_mode():
            return run_generation_with_generation_lock_held(state, configs, serving_profiler)
    finally:
        state.lock.release()


def build_generate_response_data(
    state: ServingState,
    generate_request: GenerateRequest,
    serving_profiler: GenerationProfiler,
    request_start: int,
    generation_result: GenerationRunResult,
) -> dict:
    episode_data = generation_result.episode_data
    outcomes = generation_result.outcomes
    door_matches = generation_result.door_matches
    profile_report = generation_result.profile_report

    profile_time = profile_start(state.profile)
    valid_mask = valid_map_mask(outcomes)
    valid_room_idx = episode_data.actions.room_idx[valid_mask]
    valid_room_x = episode_data.actions.room_x[valid_mask]
    valid_room_y = episode_data.actions.room_y[valid_mask]
    valid_toilet_crossed_room_idx = outcomes.end_outcomes.toilet_crossed_room_idx[valid_mask]
    valid_door_matches = filter_door_matches(door_matches, valid_mask)
    add_serving_profile(
        serving_profiler,
        state.device,
        "python.serve.filter_valid_maps",
        profile_time,
    )

    profile_time = profile_start(state.profile)
    area_assignment = assign_room_areas(
        valid_room_idx,
        valid_room_x,
        valid_room_y,
        valid_toilet_crossed_room_idx,
        valid_door_matches,
        state.door_room_lookup,
        state.room_geometry,
        state.map_station_data,
        state.toilet_data,
        state.serving_config.area_assignment_attempts,
        state.serving_config.area_bounding_box_width,
        state.serving_config.area_bounding_box_height,
        state.serving_config.area_min_rooms,
        state.serving_config.area_max_rooms,
        generate_request.area_assignment_base_order,
        serving_profiler,
    )
    add_serving_profile(
        serving_profiler,
        state.device,
        "python.serve.assign_room_areas",
        profile_time,
    )

    profile_time = profile_start(state.profile)
    area_valid_mask = area_assignment.valid_mask
    final_room_idx = area_assignment.room_idx
    final_room_x = valid_room_x[area_valid_mask]
    final_room_y = valid_room_y[area_valid_mask]
    final_toilet_crossed_room_idx = valid_toilet_crossed_room_idx[area_valid_mask]
    final_door_matches = area_assignment.door_matches
    final_room_idx_list = tensor_to_list(final_room_idx)
    final_room_x_list = tensor_to_list(final_room_x)
    final_room_y_list = tensor_to_list(final_room_y)
    final_area_list = tensor_to_list(area_assignment.area)
    final_subarea_list = tensor_to_list(area_assignment.subarea)
    final_subsubarea_list = tensor_to_list(area_assignment.subsubarea)
    final_toilet_crossed_room_idx_list = tensor_to_list(final_toilet_crossed_room_idx)
    if generate_request.small_map:
        assert generate_request.min_rooms is not None
        assert generate_request.max_rooms is not None
        assert generate_request.target_rooms is not None
        small_map_result = prune_small_maps(
            room_idx=final_room_idx_list,
            room_x=final_room_x_list,
            room_y=final_room_y_list,
            area=final_area_list,
            subarea=final_subarea_list,
            subsubarea=final_subsubarea_list,
            toilet_crossed_room_idx=final_toilet_crossed_room_idx_list,
            door_matches=final_door_matches,
            door_data=state.door_data,
            room_part_data=state.room_part_data,
            required_room_data=state.required_room_data,
            config=SmallMapConfig(
                min_rooms=generate_request.min_rooms,
                max_rooms=generate_request.max_rooms,
                target_rooms=generate_request.target_rooms,
            ),
        )
        final_room_idx_list = small_map_result.room_idx
        final_room_x_list = small_map_result.room_x
        final_room_y_list = small_map_result.room_y
        final_area_list = small_map_result.area
        final_subarea_list = small_map_result.subarea
        final_subsubarea_list = small_map_result.subsubarea
        final_toilet_crossed_room_idx_list = small_map_result.toilet_crossed_room_idx
        small_map_index = torch.tensor(
            small_map_result.source_map_idx,
            device=state.device,
            dtype=torch.int64,
        )
        final_door_matches = select_door_matches(final_door_matches, small_map_index)
    final_room_id_list = response_room_ids(final_room_idx_list, state.rooms)
    num_generated = int(episode_data.actions.room_idx.shape[0])
    num_pre_valid = int(torch.sum(valid_mask).item())
    num_valid = len(final_room_idx_list)
    add_serving_profile(
        serving_profiler,
        state.device,
        "python.serve.prepare_response_tensors",
        profile_time,
    )

    profile_time = profile_start(state.profile)
    response = {
        "stats": {
            "num_generated": num_generated,
            "num_pre_valid": num_pre_valid,
            "num_valid": num_valid,
        },
        "rooms": {
            "id": final_room_id_list,
            "x": final_room_x_list,
            "y": final_room_y_list,
            "area": final_area_list,
            "subarea": final_subarea_list,
            "subsubarea": final_subsubarea_list,
        },
        "edges": response_edges(final_room_idx_list, final_door_matches, state.door_lookups),
        "toilet_crossing_room_placement_idx": response_toilet_crossing_room_placement_idx(
            final_room_idx_list,
            final_toilet_crossed_room_idx_list,
        ),
    }
    add_serving_profile(
        serving_profiler,
        state.device,
        "python.serve.build_response_json",
        profile_time,
    )
    add_serving_profile(serving_profiler, state.device, "python.serve.total", request_start)
    if state.profile:
        response["profile"] = profile_report + serving_profiler.report()
    logging.info("Response stats: %s", response["stats"])
    return response


def generate_response_data_uncached_validated(
    state: ServingState,
    generate_request: GenerateRequest,
) -> dict:
    serving_profiler = GenerationProfiler(state.profile)
    request_start = profile_start(state.profile)

    profile_time = profile_start(state.profile)
    configs = create_generate_configs(generate_request, state, state.envs, state.device)
    add_serving_profile(
        serving_profiler,
        state.device,
        "python.serve.create_generate_configs",
        profile_time,
    )

    generation_result = run_foreground_generation(state, configs, serving_profiler)
    return build_generate_response_data(
        state,
        generate_request,
        serving_profiler,
        request_start,
        generation_result,
    )


def generate_prefetch_response_with_generation_lock_held(
    state: ServingState,
    generate_request: GenerateRequest,
) -> dict:
    serving_profiler = GenerationProfiler(state.profile)
    request_start = profile_start(state.profile)

    profile_time = profile_start(state.profile)
    configs = create_generate_configs(generate_request, state, state.envs, state.device)
    add_serving_profile(
        serving_profiler,
        state.device,
        "python.serve.create_generate_configs",
        profile_time,
    )

    with torch.inference_mode():
        generation_result = run_generation_with_generation_lock_held(
            state,
            configs,
            serving_profiler,
        )
    return build_generate_response_data(
        state,
        generate_request,
        serving_profiler,
        request_start,
        generation_result,
    )


def reschedule_prefetch_refill_locked(state: ServingState) -> None:
    if state.prefetch is None:
        return
    state.prefetch.refill_scheduled = True
    state.prefetch.due_time = (
        time.monotonic() + state.serving_config.prefetch_delay_seconds
    )
    state.prefetch.schedule_version += 1
    state.prefetch.condition.notify()


def next_prefetch_refill_key(state: ServingState) -> str | None:
    if state.prefetch is None:
        return None
    for key, queue_state in state.prefetch.queues.items():
        if queue_state.refill_debt <= 0:
            continue
        if len(queue_state.responses) >= state.serving_config.prefetch_queue_max_size:
            queue_state.refill_debt = 0
            continue
        return key
    return None


def append_prefetch_response(
    state: ServingState,
    key: str,
    response: dict,
) -> bool:
    if state.prefetch is None:
        return False
    prefetch = state.prefetch
    with prefetch.condition:
        queue_state = prefetch.queues.get(key)
        if queue_state is None:
            return False
        if len(queue_state.responses) >= state.serving_config.prefetch_queue_max_size:
            queue_state.refill_debt = 0
            return False
        queue_state.responses.append(serialize_generate_response(response))
        queue_state.refill_debt -= 1
        if len(queue_state.responses) >= state.serving_config.prefetch_queue_max_size:
            queue_state.refill_debt = 0
        return True


def run_prefetch_refill_pass(state: ServingState, schedule_version: int) -> None:
    if state.prefetch is None:
        return
    prefetch = state.prefetch
    while True:
        with prefetch.condition:
            if prefetch.schedule_version != schedule_version:
                return
            if foreground_generation_waiting(state):
                reschedule_prefetch_refill_locked(state)
                return
            key = next_prefetch_refill_key(state)
            if key is None:
                prefetch.refill_scheduled = False
                return
            generate_request = prefetch.queues[key].request
        if not state.lock.acquire(blocking=False):
            with prefetch.condition:
                if prefetch.schedule_version == schedule_version:
                    reschedule_prefetch_refill_locked(state)
            return
        try:
            try:
                response = generate_prefetch_response_with_generation_lock_held(
                    state,
                    generate_request,
                )
            except BaseException:
                logging.exception("Prefetch refill failed")
                with prefetch.condition:
                    queue_state = prefetch.queues.get(key)
                    if queue_state is not None:
                        queue_state.refill_debt = 0
                return
        finally:
            state.lock.release()
        append_prefetch_response(state, key, response)


def run_prefetch_worker(state: ServingState) -> None:
    if state.prefetch is None:
        return
    prefetch = state.prefetch
    while True:
        with prefetch.condition:
            while not prefetch.refill_scheduled:
                prefetch.condition.wait()
            delay_seconds = prefetch.due_time - time.monotonic()
            if delay_seconds > 0:
                prefetch.condition.wait(delay_seconds)
                continue
            schedule_version = prefetch.schedule_version
        run_prefetch_refill_pass(state, schedule_version)


@app.post("/generate")
def generate_response():
    state = serving_state()
    body = request.get_json(silent=False)
    logging.info("Request body: %s", body)
    generate_request = GenerateRequest.model_validate(body)
    response_data = generate_response_data(state, generate_request)
    if isinstance(response_data, str):
        return Response(response_data, mimetype="application/json")
    return jsonify(response_data)


def generate_response_data(
    state: ServingState,
    generate_request: GenerateRequest,
) -> dict | str:
    serving_profiler = GenerationProfiler(state.profile)
    profile_time = profile_start(state.profile)
    validate_generate_request(generate_request, state.rooms)
    add_serving_profile(
        serving_profiler,
        state.device,
        "python.serve.validate_request",
        profile_time,
    )
    if state.prefetch is None:
        return generate_response_data_uncached_validated(state, generate_request)
    key = prefetch_request_key(generate_request)
    response = pop_prefetch_response(state, key, generate_request)
    if response is not None:
        schedule_prefetch_refill(state, key, generate_request)
        return response
    response = generate_response_data_uncached_validated(state, generate_request)
    schedule_prefetch_refill(state, key, generate_request)
    return response


@app.get("/health")
def health_response():
    return jsonify({"status": "ok"})


@app.errorhandler(ValueError)
def value_error_response(error: ValueError):
    return jsonify({"error": str(error)}), 400


@app.errorhandler(ValidationError)
def validation_error_response(error: ValidationError):
    return jsonify({"error": error.errors()}), 400


@app.errorhandler(BadRequest)
def bad_request_response(error: BadRequest):
    return jsonify({"error": error.description}), 400


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        level=logging.INFO)
    args = parse_args()
    serving_config = load_serving_config(args.serving_config)
    model_export = load_model_input(args.model_export)
    state = create_serving_state(
        serving_config,
        model_export,
        args.seed,
        args.profile,
    )
    initialize_serving_state(state)
    run_warmup_requests(state)
    app.run(
        host=serving_config.host,
        port=serving_config.port,
        threaded=False,
    )


if __name__ == "__main__":
    main()
