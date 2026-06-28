from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
from flask import Flask, jsonify, request
from pydantic import BaseModel, ConfigDict, ValidationError
from safetensors import safe_open
from werkzeug.exceptions import BadRequest

from area_assignment import (
    DoorRoomLookup,
    RoomGeometry,
    assign_room_areas,
    build_door_room_lookup,
    build_room_geometry,
)
from env import DoorMatches, Engine, GenerateConfig
from generate import GenerationProfiler, profile_start, run_generation_groups, sync_profile_device
from model import FrontierModel
from train import create_balance_model, frontier_model_kwargs, without_prefix
from train_config import Config, validate_config


MODEL_EXPORT_FORMAT = "map-gen-model-export-v1"
TRAINING_CHECKPOINT_FORMAT = "map-gen-training-session-checkpoint-v3"
MODEL_INPUT_FORMATS = (MODEL_EXPORT_FORMAT, TRAINING_CHECKPOINT_FORMAT)
MODEL_PREFIXES = ("ema_model", "balance_model")
app = Flask(__name__)


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServingConfig(StrictBaseModel):
    host: str
    port: int
    device: str
    compile_model: bool
    autocast: bool
    verify_outcome_consistency: bool
    gpu_prefetch_batches: int
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


@dataclass
class ModelExport:
    training_config: Config
    tensors: dict[str, torch.Tensor]


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
    profile: bool
    lock: threading.Lock


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
    generation_config = training_config.generation
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
    if serving_config.gpu_prefetch_batches < 0:
        raise ValueError("gpu_prefetch_batches must be greater than or equal to zero")
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
    group_environments = serving_config.num_environments // serving_config.pipeline_groups
    group_threads = serving_config.num_threads // serving_config.pipeline_groups
    if group_threads <= 0:
        raise ValueError("num_threads must be at least pipeline_groups")
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
        torch.set_float32_matmul_precision("high")
    engine = Engine(rooms, model_export.training_config.features)
    model = FrontierModel(**frontier_model_kwargs(model_export.training_config, rooms, engine)).to(
        device
    )
    model.load_state_dict(without_prefix(model_export.tensors, "ema_model"))
    model.requires_grad_(False)
    model.eval()
    balance_model = create_balance_model(model_export.training_config, rooms, device)
    balance_model.load_state_dict(without_prefix(model_export.tensors, "balance_model"))
    balance_model.requires_grad_(False)
    balance_model.eval()
    if serving_config.compile_model:
        model = torch.compile(model)
        balance_model = torch.compile(balance_model)
    envs = create_environment_groups(serving_config, model_export.training_config, engine, seed)
    door_room_lookup = build_door_room_lookup(rooms, device)
    room_geometry = build_room_geometry(rooms, device)
    return ServingState(
        serving_config=serving_config,
        training_config=model_export.training_config,
        rooms=rooms,
        device=device,
        envs=envs,
        model=model,
        balance_model=balance_model,
        door_room_lookup=door_room_lookup,
        room_geometry=room_geometry,
        profile=profile,
        lock=threading.Lock(),
    )


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


def create_generate_configs(
    generate_request: GenerateRequest,
    state: ServingState,
    envs: list,
    device: torch.device,
) -> list[GenerateConfig]:
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
            distance_proximity_scale=state.training_config.distance_proximity_scale,
            autocast=state.serving_config.autocast,
        )
        for env in envs
    ]


def tensor_to_list(tensor: torch.Tensor) -> list:
    return tensor.detach().cpu().tolist()


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


def initialize_serving_state(state: ServingState) -> None:
    global SERVING_STATE
    SERVING_STATE = state


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


@app.post("/generate")
def generate_response():
    state = serving_state()
    serving_profiler = GenerationProfiler(state.profile)
    request_start = profile_start(state.profile)

    profile_time = profile_start(state.profile)
    body = request.get_json(silent=False)
    generate_request = GenerateRequest.model_validate(body)
    validate_generate_request(generate_request, state.rooms)
    add_serving_profile(
        serving_profiler,
        state.device,
        "python.serve.parse_validate_request",
        profile_time,
    )

    profile_time = profile_start(state.profile)
    configs = create_generate_configs(generate_request, state, state.envs, state.device)
    add_serving_profile(
        serving_profiler,
        state.device,
        "python.serve.create_generate_configs",
        profile_time,
    )

    with state.lock, torch.inference_mode():
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

    profile_time = profile_start(state.profile)
    valid_mask = valid_map_mask(outcomes)
    valid_room_idx = episode_data.actions.room_idx[valid_mask]
    valid_room_x = episode_data.actions.room_x[valid_mask]
    valid_room_y = episode_data.actions.room_y[valid_mask]
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
        valid_door_matches,
        state.door_room_lookup,
        state.room_geometry,
        state.serving_config.area_assignment_attempts,
        state.serving_config.area_bounding_box_width,
        state.serving_config.area_bounding_box_height,
        state.serving_config.area_min_rooms,
        state.serving_config.area_max_rooms,
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
    final_room_idx = valid_room_idx[area_valid_mask]
    final_room_x = valid_room_x[area_valid_mask]
    final_room_y = valid_room_y[area_valid_mask]
    num_generated = int(episode_data.actions.room_idx.shape[0])
    num_pre_valid = int(torch.sum(valid_mask).item())
    num_valid = int(torch.sum(area_valid_mask).item())
    add_serving_profile(
        serving_profiler,
        state.device,
        "python.serve.prepare_response_tensors",
        profile_time,
    )

    profile_time = profile_start(state.profile)
    response = {
        "num_generated": num_generated,
        "num_pre_valid": num_pre_valid,
        "num_valid": num_valid,
        "actions": {
            "room_idx": tensor_to_list(final_room_idx),
            "room_x": tensor_to_list(final_room_x),
            "room_y": tensor_to_list(final_room_y),
        },
        "area": tensor_to_list(area_assignment.area),
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
    return jsonify(response)


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
    app.run(
        host=serving_config.host,
        port=serving_config.port,
        threaded=False,
    )


if __name__ == "__main__":
    main()
