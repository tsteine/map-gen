import argparse
from collections import deque
import copy
import json
import logging
import math
import multiprocessing
import os
import signal
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import safetensors.torch
import torch
import map_gen
from aim import Run
from aim.sdk.errors import MissingRunError
from safetensors import safe_open

from env import (
    Actions,
    DoorMatchCounts,
    Engine,
    EndOutcomes,
    EpisodeData,
    EpisodeOutcomes,
    GenerateConfig,
    StepOutcomes,
    ProposalData,
)
from experience import ExperienceStorage
from generate import GenerationStats, run_generation_groups
from learn import (
    CandidateDiagnostics,
    MainLossBreakdown,
    TrainRoundContext,
    compute_candidate_diagnostics,
    distance_proximity_utility,
    train_round as run_train_round,
)
from loss import (
    LossConfig,
    compute_balance_toilet_crossed_room_ss,
    compute_balance_door_match_ss,
)
from model import BalanceModel, FrontierModel
from optimizers import Muon
from train_config import (
    AdamOptimizerConfig,
    AdamParamsConfig,
    Config,
    MuonOptimizerConfig,
    OptimizerConfig,
    VariableFloat,
    VariableRange,
    VariableSchedule,
    episodes_per_round,
    instantiate_scheduleable_config,
    validate_config,
)
from visualize import save_episode_frames


@dataclass
class Args:
    config: Path
    verify_outcome_consistency: bool
    device: str
    load_checkpoint: Path | None
    profile: bool
    ignore_scores: bool


type RustProfileReport = list[tuple[str, int, int]]

IGNORE_SCORES_TEMPERATURE = 1.0e9
TRAINING_CHECKPOINT_FORMAT = "map-gen-training-session-checkpoint-v3"


def compute_door_match_count_ss(counts: torch.Tensor, dim: int) -> torch.Tensor:
    totals = torch.sum(counts, dim=dim, keepdim=True)
    if torch.any(totals <= 1):
        return counts.new_full((), torch.nan)
    return torch.sum(counts * (counts - 1) / (totals * (totals - 1)))


class Prefetcher:
    def __init__(self, max_workers=1):
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than zero")
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def close(self):
        self.executor.shutdown()

    def map(self, items, prepare):
        items = iter(items)
        pending = deque()

        for _ in range(self.max_workers):
            if not submit_prefetch_item(self.executor, pending, items, prepare):
                break

        while pending:
            future = pending.popleft()
            result = future.result()
            submit_prefetch_item(self.executor, pending, items, prepare)
            yield result


def submit_prefetch_item(executor: ThreadPoolExecutor, pending: deque, items, prepare) -> bool:
    try:
        pending.append(executor.submit(prepare, next(items)))
        return True
    except StopIteration:
        return False


def as_checkpoint_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().contiguous()


def prefixed_state_dict(prefix: str, module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}.{name}": as_checkpoint_tensor(value)
        for name, value in unwrap_compiled_module(module).state_dict().items()
    }


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


def optimizer_checkpoint_state(
    optimizer: torch.optim.Optimizer,
    prefix: str,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    state_dict = optimizer.state_dict()
    tensors = {}
    scalar_state = {}
    for param_id, param_state in state_dict["state"].items():
        param_scalar_state = {}
        for state_name, value in param_state.items():
            if torch.is_tensor(value):
                tensors[f"{prefix}.state.{param_id}.{state_name}"] = as_checkpoint_tensor(value)
            else:
                param_scalar_state[state_name] = value
        if param_scalar_state:
            scalar_state[str(param_id)] = param_scalar_state
    return tensors, state_dict["param_groups"], scalar_state


def load_optimizer_checkpoint_state(
    optimizer: torch.optim.Optimizer,
    tensors: dict[str, torch.Tensor],
    param_groups: list[dict[str, Any]],
    scalar_state: dict[str, dict[str, Any]],
    prefix: str,
) -> None:
    state: dict[int, dict[str, Any]] = {}
    state_prefix = f"{prefix}.state."
    for key, value in tensors.items():
        if not key.startswith(state_prefix):
            continue
        suffix = key[len(state_prefix) :]
        param_id_text, state_name = suffix.split(".", 1)
        state.setdefault(int(param_id_text), {})[state_name] = value
    for param_id_text, param_scalar_state in scalar_state.items():
        state.setdefault(int(param_id_text), {}).update(param_scalar_state)
    optimizer.load_state_dict(
        {
            "state": state,
            "param_groups": param_groups,
        }
    )


class MainOptimizerBundle:
    def __init__(self, adam_optimizer: torch.optim.Optimizer, muon_optimizer: Muon):
        self.adam_optimizer = adam_optimizer
        self.muon_optimizer = muon_optimizer

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.adam_optimizer.param_groups + self.muon_optimizer.param_groups

    def step(self) -> None:
        self.adam_optimizer.step()
        self.muon_optimizer.step()

    def set_lrs(self, config: OptimizerConfig) -> None:
        if not isinstance(config, MuonOptimizerConfig):
            raise TypeError("Muon optimizer bundle requires a Muon optimizer config")
        self.adam_optimizer.param_groups[0]["lr"] = config.adam.lr
        self.muon_optimizer.param_groups[0]["lr"] = config.muon.lr

    def named_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        return {
            "adam": self.adam_optimizer,
            "muon": self.muon_optimizer,
        }


def named_checkpoint_optimizers(optimizer: Any) -> dict[str, torch.optim.Optimizer]:
    if isinstance(optimizer, MainOptimizerBundle):
        return optimizer.named_optimizers()
    return {"adam": optimizer}


def save_named_optimizer_checkpoint_state(
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str],
    optimizer: Any,
    prefix: str,
) -> None:
    names = []
    for name, named_optimizer in named_checkpoint_optimizers(optimizer).items():
        names.append(name)
        part_prefix = f"{prefix}.{name}"
        part_tensors, param_groups, scalar_state = optimizer_checkpoint_state(
            named_optimizer,
            part_prefix,
        )
        tensors.update(part_tensors)
        metadata[f"{prefix}_{name}_param_groups"] = json.dumps(param_groups)
        metadata[f"{prefix}_{name}_scalar_state"] = json.dumps(scalar_state)
    metadata[f"{prefix}_names"] = json.dumps(names)


def load_named_optimizer_checkpoint_state(
    optimizer: Any,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str],
    prefix: str,
) -> None:
    optimizers = named_checkpoint_optimizers(optimizer)
    names = json.loads(metadata[f"{prefix}_names"])
    if set(names) != set(optimizers):
        raise ValueError(
            f"checkpoint has {prefix} optimizer part(s) {names}, but config created {list(optimizers)}"
        )
    for name in names:
        load_optimizer_checkpoint_state(
            optimizers[name],
            tensors,
            json.loads(metadata[f"{prefix}_{name}_param_groups"]),
            json.loads(metadata[f"{prefix}_{name}_scalar_state"]),
            f"{prefix}.{name}",
        )


def validate_checkpoint_metadata(path: Path, metadata: dict[str, str] | None) -> dict[str, str]:
    if metadata is None:
        raise ValueError(f"checkpoint metadata missing in {path}")
    if metadata["format"] != TRAINING_CHECKPOINT_FORMAT:
        logging.warning(f"unsupported checkpoint format in {path}")
    for field in (
        "aim_run_hash",
        "num_episodes",
        "experience_num_files",
    ):
        if field not in metadata:
            raise ValueError(f"checkpoint metadata field {field!r} missing in {path}")
    return metadata


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


def create_adam_optimizer(
    parameters,
    config: AdamOptimizerConfig | AdamParamsConfig,
    initial_config: AdamOptimizerConfig | AdamParamsConfig,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        parameters,
        lr=initial_config.lr,
        betas=(config.beta1, config.beta2),
    )


def unwrap_compiled_module(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def split_muon_parameters(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    model = unwrap_compiled_module(model)
    muon_params = []
    muon_param_ids = set()
    for module in model.modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        weight = module.weight
        if not weight.requires_grad or id(weight) in muon_param_ids:
            continue
        muon_params.append(weight)
        muon_param_ids.add(id(weight))

    adam_params = [
        param
        for param in model.parameters()
        if param.requires_grad and id(param) not in muon_param_ids
    ]
    trainable_param_ids = {id(param) for param in model.parameters() if param.requires_grad}
    assigned_param_ids = {id(param) for param in adam_params} | muon_param_ids
    if assigned_param_ids != trainable_param_ids:
        raise ValueError(
            "Muon optimizer parameter split omitted or duplicated trainable parameters"
        )
    if not muon_params:
        raise ValueError("Muon optimizer requires at least one Linear weight parameter")
    return adam_params, muon_params


def create_main_optimizer(
    model: torch.nn.Module,
    config: OptimizerConfig,
    initial_config: OptimizerConfig,
) -> Any:
    if isinstance(config, MuonOptimizerConfig):
        if not isinstance(initial_config, MuonOptimizerConfig):
            raise TypeError("initial optimizer config must have the same type as optimizer config")
        adam_params, muon_params = split_muon_parameters(model)
        logging.info(
            "Using Muon for %s Linear weight parameter tensor(s) and Adam for %s other parameter tensor(s).",
            len(muon_params),
            len(adam_params),
        )
        return MainOptimizerBundle(
            create_adam_optimizer(adam_params, config.adam, initial_config.adam),
            Muon(
                muon_params,
                lr=initial_config.muon.lr,
                momentum=config.muon.momentum,
                nesterov=config.muon.nesterov,
                backend=config.muon.backend,
                backend_steps=config.muon.backend_steps,
            ),
        )
    return create_adam_optimizer(
        model.parameters(),
        config,
        initial_config,
    )


def optimizer_metric_values(config: OptimizerConfig) -> dict[str, float]:
    if isinstance(config, MuonOptimizerConfig):
        return {
            "adam_lr": config.adam.lr,
            "muon_lr": config.muon.lr,
        }
    return {"lr": config.lr}


def variable_float_metric_value(value: VariableFloat, path: str) -> float:
    if isinstance(value, VariableSchedule):
        if (value.linear is None) == (value.log is None):
            raise ValueError(f"{path} must have exactly one value: 'linear' or 'log'")
        range_value = value.linear if value.linear is not None else value.log
        if not isinstance(range_value, VariableRange):
            raise ValueError(f"{path} must be instantiated before metric logging")
        min_value = float(range_value.min)
        max_value = float(range_value.max)
        if value.linear is not None:
            return 0.5 * (min_value + max_value)
        if min_value <= 0.0:
            raise ValueError(f"{path}.min must be greater than zero for a log range")
        return math.exp(0.5 * (math.log(min_value) + math.log(max_value)))
    return float(value)


def topk_or_zeros(values: torch.Tensor, k0: int) -> torch.Tensor:
    if values.numel() == 0:
        return torch.zeros([k0], dtype=torch.float32)
    k = min(k0, values.numel())
    top = torch.topk(values.flatten(), k=k).values.to(torch.float32)
    if k == k0:
        return top
    return torch.cat([top, top.new_zeros([k0 - k])])


def toilet_crossed_room_distribution(
    toilet_crossed_room_idx: torch.Tensor,
    num_rooms: int,
) -> torch.Tensor:
    valid = toilet_crossed_room_idx >= 0
    counts = torch.bincount(
        toilet_crossed_room_idx[valid].to(torch.int64),
        minlength=num_rooms,
    ).to(torch.float32)
    total = torch.sum(counts)
    return counts / total.clamp_min(1.0)


def create_generation_environment_groups_for_device(
    config: Config,
    engine: Engine,
    device_index: int,
):
    num_generation_groups = config.generation.num_devices * config.generation.pipeline_groups
    generation_group_environments = config.generation.num_environments // num_generation_groups
    generation_group_threads = (
        None
        if config.generation.num_threads is None
        else config.generation.num_threads // config.generation.pipeline_groups
    )
    return [
        engine.create_environment_group(
            config.map_size,
            generation_group_environments,
            config.generation.candidate_spatial_cell_size,
            seed=device_index * config.generation.pipeline_groups + group_index,
            frontier_neighbor_algorithm=config.generation.frontier_neighbor_algorithm,
            frontier_neighbor_count=config.generation.frontier_neighbor_count,
            frontier_window_size=config.generation.frontier_window_size,
            num_threads=generation_group_threads,
        )
        for group_index in range(config.generation.pipeline_groups)
    ]


def create_generate_config(
    config: Config,
    episode_length: int,
    num_envs: int,
    device: torch.device,
    ignore_scores: bool,
) -> GenerateConfig:
    def variable_float_tensor(value: VariableFloat, path: str) -> torch.Tensor:
        if isinstance(value, VariableSchedule):
            if (value.linear is None) == (value.log is None):
                raise ValueError(f"{path} must have exactly one value: 'linear' or 'log'")
            range_value = value.linear if value.linear is not None else value.log
            if not isinstance(range_value, VariableRange):
                raise ValueError(f"{path} must be instantiated before creating a generate config")
            min_value = float(range_value.min)
            max_value = float(range_value.max)
            if min_value > max_value:
                raise ValueError(f"{path}.min must be less than or equal to {path}.max")
            sample = torch.rand([num_envs], dtype=torch.float32, device=device)
            if value.linear is not None:
                return min_value + sample * (max_value - min_value)
            if min_value <= 0.0:
                raise ValueError(f"{path}.min must be greater than zero for a log range")
            log_min = math.log(min_value)
            log_max = math.log(max_value)
            return torch.exp(log_min + sample * (log_max - log_min))
        return torch.full([num_envs], float(value), dtype=torch.float32, device=device)

    temperature = (
        torch.full([num_envs], IGNORE_SCORES_TEMPERATURE, dtype=torch.float32, device=device)
        if ignore_scores
        else variable_float_tensor(config.generation.temperature, "generation.temperature")
    )
    proposal_temperature = (
        torch.full([num_envs], IGNORE_SCORES_TEMPERATURE, dtype=torch.float32, device=device)
        if ignore_scores
        else variable_float_tensor(
            config.generation.proposal_temperature,
            "generation.proposal_temperature",
        )
    )
    return GenerateConfig(
        episode_length=episode_length,
        recommended_candidates=config.generation.recommended_candidates,
        shortlist_candidates=config.generation.shortlist_candidates,
        gpu_prefetch_batches=config.generation.gpu_prefetch_batches,
        temperature=temperature,
        proposal_temperature=proposal_temperature,
        reward_door=variable_float_tensor(config.generation.reward_door, "generation.reward_door"),
        reward_connection=variable_float_tensor(
            config.generation.reward_connection,
            "generation.reward_connection",
        ),
        reward_toilet=variable_float_tensor(
            config.generation.reward_toilet,
            "generation.reward_toilet",
        ),
        reward_phantoon=variable_float_tensor(
            config.generation.reward_phantoon,
            "generation.reward_phantoon",
        ),
        reward_balance=variable_float_tensor(
            config.generation.reward_balance,
            "generation.reward_balance",
        ),
        reward_toilet_balance=variable_float_tensor(
            config.generation.reward_toilet_balance,
            "generation.reward_toilet_balance",
        ),
        reward_frontier=variable_float_tensor(
            config.generation.reward_frontier,
            "generation.reward_frontier",
        ),
        reward_graph_diameter=variable_float_tensor(
            config.generation.reward_graph_diameter,
            "generation.reward_graph_diameter",
        ),
        reward_save_distance=variable_float_tensor(
            config.generation.reward_save_distance,
            "generation.reward_save_distance",
        ),
        reward_refill_distance=variable_float_tensor(
            config.generation.reward_refill_distance,
            "generation.reward_refill_distance",
        ),
        reward_missing_connect_utility=variable_float_tensor(
            config.generation.reward_missing_connect_utility,
            "generation.reward_missing_connect_utility",
        ),
        distance_proximity_scale=config.distance_proximity_scale,
        autocast=config.model.generation_autocast,
    )


@dataclass
class GenerationProcessState:
    config: Config
    episode_length: int
    device: torch.device
    envs: list
    model: torch.nn.Module
    balance_model: torch.nn.Module
    profile: bool
    ignore_scores: bool
    cleared_cuda_cache_after_first_task: bool


GENERATION_PROCESS_STATE: GenerationProcessState | None = None


def initialize_generation_process(
    config_json: str,
    rooms_json: str,
    device_text: str,
    device_index: int,
    profile: bool,
    ignore_scores: bool,
) -> None:
    global GENERATION_PROCESS_STATE
    config = Config.model_validate_json(config_json)
    rooms = json.loads(rooms_json)
    device = torch.device(device_text)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("high")
    engine = Engine(rooms, config.features)
    envs = create_generation_environment_groups_for_device(
        config,
        engine,
        device_index,
    )
    model = FrontierModel(**frontier_model_kwargs(config, rooms, engine)).to(device)
    model.requires_grad_(False)
    model.eval()
    if config.model.compile:
        model = torch.compile(model)
    balance_model = create_balance_model(config, rooms, device)
    balance_model.requires_grad_(False)
    balance_model.eval()
    map_gen.set_profile_enabled(profile)
    GENERATION_PROCESS_STATE = GenerationProcessState(
        config=config,
        episode_length=len(rooms),
        device=device,
        envs=envs,
        model=model,
        balance_model=balance_model,
        profile=profile,
        ignore_scores=ignore_scores,
        cleared_cuda_cache_after_first_task=False,
    )


def run_generation_process_task(
    model_state: dict[str, torch.Tensor],
    balance_model_state: dict[str, torch.Tensor],
    generation_config_json: str,
    verify_outcome_consistency: bool,
) -> tuple[
    EpisodeData, EpisodeOutcomes, DoorMatchCounts, ProposalData, GenerationStats, RustProfileReport
]:
    if GENERATION_PROCESS_STATE is None:
        raise RuntimeError("generation process was not initialized")
    state = GENERATION_PROCESS_STATE
    if state.profile:
        map_gen.reset_profile()
    unwrap_compiled_module(state.model).load_state_dict(model_state)
    unwrap_compiled_module(state.balance_model).load_state_dict(balance_model_state)
    generation_config = Config.model_validate_json(generation_config_json)
    gen_configs = [
        create_generate_config(
            generation_config,
            state.episode_length,
            env.num_envs,
            state.device,
            state.ignore_scores,
        )
        for env in state.envs
    ]
    (
        episode_data,
        outcomes,
        door_match_counts,
        proposal_data,
        generation_stats,
        python_profile_report,
    ) = run_generation_groups(
        state.envs,
        state.model,
        state.balance_model,
        gen_configs,
        state.device,
        verify_outcome_consistency=verify_outcome_consistency,
        profile=state.profile,
    )
    profile_report = map_gen.profile_report() + python_profile_report if state.profile else []
    episode_data_cpu = episode_data.to(torch.device("cpu"))
    outcomes_cpu = outcomes.to(torch.device("cpu"))
    door_match_counts_cpu = door_match_counts.to(torch.device("cpu"))
    proposal_data_cpu = proposal_data.to(torch.device("cpu"))
    del episode_data, outcomes, door_match_counts, proposal_data
    if state.device.type == "cuda" and not state.cleared_cuda_cache_after_first_task:
        torch.cuda.empty_cache()
        state.cleared_cuda_cache_after_first_task = True
    return (
        episode_data_cpu,
        outcomes_cpu,
        door_match_counts_cpu,
        proposal_data_cpu,
        generation_stats,
        profile_report,
    )


def merge_profile_reports(reports: list[RustProfileReport]) -> RustProfileReport:
    merged = {}
    for report in reports:
        for name, count, nanos in report:
            merged_count, merged_nanos = merged.get(name, (0, 0))
            merged[name] = (merged_count + count, merged_nanos + nanos)
    return [(name, count, nanos) for name, (count, nanos) in merged.items()]


@dataclass
class TrainingSession:
    args: Args
    config: Config
    run_path: str
    rooms: list[dict]
    device: torch.device
    generation_devices: list[torch.device]
    engine: Engine
    train_batch_envs: list
    main_model: torch.nn.Module
    ema_model: torch.nn.Module
    balance_model: torch.nn.Module
    main_optimizer: Any
    balance_optimizer: torch.optim.Optimizer
    loss_config: LossConfig
    experience: ExperienceStorage
    train_batch_prefetcher: Prefetcher
    generation_executors: list[ProcessPoolExecutor]
    aim_run: Run = field(init=False)
    num_episodes: int = 0
    stop_requested: bool = False

    @property
    def num_rooms(self) -> int:
        return len(self.rooms)

    @property
    def episode_length(self) -> int:
        return len(self.rooms)

    @property
    def episodes_per_round(self) -> int:
        return episodes_per_round(self.config)

    @property
    def train_pipeline_groups(self) -> int:
        return self.config.train.pipeline_groups

    def room_door_labels_by_direction(self, direction: str) -> list[str]:
        labels = []
        for room in self.rooms:
            room_name = room["name"]
            for door_group in room["doors"]:
                for door in door_group:
                    if door["direction"] == direction:
                        labels.append(f"{room_name} {direction}({door['x']}, {door['y']})")
        return labels

    def format_horizontal_topk_door_connections(
        self,
        proportions: torch.Tensor,
        k: int,
    ) -> str:
        left_door_labels = self.room_door_labels_by_direction("left")
        right_door_labels = self.room_door_labels_by_direction("right")
        if proportions.shape != (len(left_door_labels), len(right_door_labels)):
            raise ValueError(
                "horizontal door match proportions shape does not match left/right door counts: "
                f"{tuple(proportions.shape)} vs ({len(left_door_labels)}, {len(right_door_labels)})"
            )

        topk = torch.topk(proportions.flatten(), k=k)
        column_count = proportions.shape[1]
        pairs = []
        for rank, (flat_idx, value) in enumerate(
            zip(topk.indices.tolist(), topk.values.tolist()),
            start=1,
        ):
            left_idx = flat_idx // column_count
            right_idx = flat_idx % column_count
            pairs.append(
                f"top{rank}: {left_door_labels[left_idx]} -> {right_door_labels[right_idx]} ({value:.4f})"
            )
        return "; ".join(pairs)

    def request_stop(self) -> None:
        self.stop_requested = True
        logging.info("Stop signal received; training will stop after the current round finishes.")

    def checkpoint_path(self, completed_round: int) -> Path:
        return Path(self.run_path) / "checkpoints" / f"round_{completed_round}.safetensors"

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {}
        tensors.update(prefixed_state_dict("main_model", self.main_model))
        tensors.update(prefixed_state_dict("ema_model", self.ema_model))
        tensors.update(prefixed_state_dict("balance_model", self.balance_model))
        metadata = {
            "format": TRAINING_CHECKPOINT_FORMAT,
            "config": self.config.model_dump_json(),
            "aim_run_hash": self.aim_run.hash,
            "num_episodes": str(self.num_episodes),
            "experience_num_files": str(self.experience.num_files),
        }
        save_named_optimizer_checkpoint_state(
            tensors,
            metadata,
            self.main_optimizer,
            "optimizer",
        )
        save_named_optimizer_checkpoint_state(
            tensors,
            metadata,
            self.balance_optimizer,
            "balance_optimizer",
        )
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        safetensors.torch.save_file(tensors, temp_path, metadata=metadata)
        os.replace(temp_path, path)
        logging.info("Saved checkpoint: %s", path)

    def load_checkpoint(self, path: Path) -> dict[str, str]:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            metadata = validate_checkpoint_metadata(path, checkpoint.metadata())
            tensors = {name: checkpoint.get_tensor(name) for name in checkpoint.keys()}

        unwrap_compiled_module(self.main_model).load_state_dict(
            without_prefix(tensors, "main_model")
        )
        unwrap_compiled_module(self.ema_model).load_state_dict(
            without_prefix(tensors, "ema_model")
        )
        unwrap_compiled_module(self.balance_model).load_state_dict(
            without_prefix(tensors, "balance_model")
        )
        load_named_optimizer_checkpoint_state(
            self.main_optimizer,
            tensors,
            metadata,
            "optimizer",
        )
        load_named_optimizer_checkpoint_state(
            self.balance_optimizer,
            tensors,
            metadata,
            "balance_optimizer",
        )
        self.num_episodes = int(metadata["num_episodes"])
        if self.num_episodes % self.episodes_per_round != 0:
            raise ValueError(
                f"checkpoint num_episodes={self.num_episodes} is not divisible by episodes_per_round="
                f"{self.episodes_per_round}"
            )
        self.experience.num_files = int(metadata["experience_num_files"])
        logging.info(
            "Loaded checkpoint %s at %s episode(s) with %s replay file(s).",
            path,
            self.num_episodes,
            self.experience.num_files,
        )
        return metadata

    def update_ema_model(self, ema_decay: float) -> None:
        with torch.no_grad():
            for ema_param, main_param in zip(
                self.ema_model.parameters(), self.main_model.parameters()
            ):
                ema_param.lerp_(main_param, 1.0 - ema_decay)

    def generate_round(
        self,
    ) -> tuple[
        EpisodeData,
        EpisodeOutcomes,
        DoorMatchCounts,
        ProposalData,
        GenerationStats,
        RustProfileReport,
    ]:
        episode_data_iterations = []
        outcome_iterations = []
        door_match_count_iterations = []
        proposal_data_iterations = []
        generation_stats_iterations = []
        profile_reports = []
        model_state = {
            name: as_checkpoint_tensor(value)
            for name, value in unwrap_compiled_module(self.ema_model).state_dict().items()
        }
        balance_model_state = {
            name: as_checkpoint_tensor(value)
            for name, value in unwrap_compiled_module(self.balance_model).state_dict().items()
        }
        generation_config = instantiate_scheduleable_config(
            self.config,
            self.num_episodes,
        )
        for iteration in range(self.config.generation.num_iterations):
            futures = [
                executor.submit(
                    run_generation_process_task,
                    model_state,
                    balance_model_state,
                    generation_config.model_dump_json(),
                    self.args.verify_outcome_consistency,
                )
                for executor in self.generation_executors
            ]
            shard_results = [future.result() for future in futures]

            for (
                iteration_episode_data,
                iteration_outcomes,
                iteration_door_match_counts,
                iteration_proposal_data,
                iteration_generation_stats,
                iteration_profile_report,
            ) in shard_results:
                episode_data_iterations.append(iteration_episode_data.to(self.device))
                outcome_iterations.append(iteration_outcomes.to(self.device))
                door_match_count_iterations.append(iteration_door_match_counts)
                proposal_data_iterations.append(iteration_proposal_data.to(self.device))
                generation_stats_iterations.append(iteration_generation_stats)
                profile_reports.append(iteration_profile_report)

        generation_stats = {
            name: sum(stats[name] for stats in generation_stats_iterations)
            / len(generation_stats_iterations)
            for name in generation_stats_iterations[0]
        }

        return (
            EpisodeData(
                actions=Actions(
                    room_idx=torch.cat(
                        [episode_data.actions.room_idx for episode_data in episode_data_iterations]
                    ),
                    room_x=torch.cat(
                        [episode_data.actions.room_x for episode_data in episode_data_iterations]
                    ),
                    room_y=torch.cat(
                        [episode_data.actions.room_y for episode_data in episode_data_iterations]
                    ),
                ),
                temperature=torch.cat(
                    [episode_data.temperature for episode_data in episode_data_iterations]
                ),
                recommended_candidates=torch.cat(
                    [
                        episode_data.recommended_candidates
                        for episode_data in episode_data_iterations
                    ]
                ),
            ),
            EpisodeOutcomes(
                step_outcomes=StepOutcomes(
                    door_invalid=torch.cat(
                        [outcomes.step_outcomes.door_invalid for outcomes in outcome_iterations]
                    ),
                    connection_invalid=torch.cat(
                        [
                            outcomes.step_outcomes.connection_invalid
                            for outcomes in outcome_iterations
                        ]
                    ),
                    toilet_invalid=torch.cat(
                        [outcomes.step_outcomes.toilet_invalid for outcomes in outcome_iterations]
                    ),
                    phantoon_invalid=torch.cat(
                        [
                            outcomes.step_outcomes.phantoon_invalid
                            for outcomes in outcome_iterations
                        ]
                    ),
                    door_match=torch.cat(
                        [outcomes.step_outcomes.door_match for outcomes in outcome_iterations]
                    ),
                ),
                end_outcomes=EndOutcomes(
                    toilet_crossed_room_idx=torch.cat(
                        [
                            outcomes.end_outcomes.toilet_crossed_room_idx
                            for outcomes in outcome_iterations
                        ]
                    ),
                    avg_frontiers=torch.cat(
                        [outcomes.end_outcomes.avg_frontiers for outcomes in outcome_iterations]
                    ),
                    graph_diameter=torch.cat(
                        [outcomes.end_outcomes.graph_diameter for outcomes in outcome_iterations]
                    ),
                    active_room_part_mask=torch.cat(
                        [
                            outcomes.end_outcomes.active_room_part_mask
                            for outcomes in outcome_iterations
                        ]
                    ),
                    save_distance=torch.cat(
                        [outcomes.end_outcomes.save_distance for outcomes in outcome_iterations]
                    ),
                    save_distance_mask=torch.cat(
                        [
                            outcomes.end_outcomes.save_distance_mask
                            for outcomes in outcome_iterations
                        ]
                    ),
                    save_to_room_distance=torch.cat(
                        [
                            outcomes.end_outcomes.save_to_room_distance
                            for outcomes in outcome_iterations
                        ]
                    ),
                    save_to_room_distance_mask=torch.cat(
                        [
                            outcomes.end_outcomes.save_to_room_distance_mask
                            for outcomes in outcome_iterations
                        ]
                    ),
                    save_from_room_distance=torch.cat(
                        [
                            outcomes.end_outcomes.save_from_room_distance
                            for outcomes in outcome_iterations
                        ]
                    ),
                    save_from_room_distance_mask=torch.cat(
                        [
                            outcomes.end_outcomes.save_from_room_distance_mask
                            for outcomes in outcome_iterations
                        ]
                    ),
                    refill_distance=torch.cat(
                        [outcomes.end_outcomes.refill_distance for outcomes in outcome_iterations]
                    ),
                    refill_distance_mask=torch.cat(
                        [
                            outcomes.end_outcomes.refill_distance_mask
                            for outcomes in outcome_iterations
                        ]
                    ),
                    refill_to_room_distance=torch.cat(
                        [
                            outcomes.end_outcomes.refill_to_room_distance
                            for outcomes in outcome_iterations
                        ]
                    ),
                    refill_to_room_distance_mask=torch.cat(
                        [
                            outcomes.end_outcomes.refill_to_room_distance_mask
                            for outcomes in outcome_iterations
                        ]
                    ),
                    refill_from_room_distance=torch.cat(
                        [
                            outcomes.end_outcomes.refill_from_room_distance
                            for outcomes in outcome_iterations
                        ]
                    ),
                    refill_from_room_distance_mask=torch.cat(
                        [
                            outcomes.end_outcomes.refill_from_room_distance_mask
                            for outcomes in outcome_iterations
                        ]
                    ),
                    missing_connect_distance=torch.cat(
                        [
                            outcomes.end_outcomes.missing_connect_distance
                            for outcomes in outcome_iterations
                        ]
                    ),
                    missing_connect_distance_mask=torch.cat(
                        [
                            outcomes.end_outcomes.missing_connect_distance_mask
                            for outcomes in outcome_iterations
                        ]
                    ),
                ),
            ),
            DoorMatchCounts(
                horizontal=torch.sum(
                    torch.stack([counts.horizontal for counts in door_match_count_iterations]),
                    dim=0,
                ),
                vertical=torch.sum(
                    torch.stack([counts.vertical for counts in door_match_count_iterations]),
                    dim=0,
                ),
            ),
            ProposalData(
                frontier_idx=torch.cat(
                    [proposal_data.frontier_idx for proposal_data in proposal_data_iterations]
                ),
                door_variant_idx=torch.cat(
                    [proposal_data.door_variant_idx for proposal_data in proposal_data_iterations]
                ),
                selected_candidate=torch.cat(
                    [
                        proposal_data.selected_candidate
                        for proposal_data in proposal_data_iterations
                    ]
                ),
                target_logits=torch.cat(
                    [proposal_data.target_logits for proposal_data in proposal_data_iterations]
                ),
            ),
            generation_stats,
            merge_profile_reports(profile_reports),
        )

    def train_round(
        self,
        episode_data: EpisodeData,
        episode_outcomes: EpisodeOutcomes,
        proposal_data: ProposalData,
        step_config: Config,
    ) -> tuple[MainLossBreakdown, float]:
        return run_train_round(
            TrainRoundContext(
                config=self.config,
                step_config=step_config,
                device=self.device,
                train_batch_envs=self.train_batch_envs,
                main_model=self.main_model,
                balance_model=self.balance_model,
                main_optimizer=self.main_optimizer,
                balance_optimizer=self.balance_optimizer,
                loss_config=self.loss_config,
                experience=self.experience,
                train_batch_prefetcher=self.train_batch_prefetcher,
                update_ema_model=self.update_ema_model,
                num_rooms=self.num_rooms,
                episode_length=self.episode_length,
            ),
            episode_data,
            episode_outcomes,
            proposal_data,
        )

    def visualize_round(self, episode_data: EpisodeData, round_idx: int) -> None:
        output_root = Path(self.run_path) / "visualizations" / f"round_{round_idx:04d}"
        actions = (
            episode_data.actions.room_idx.cpu(),
            episode_data.actions.room_x.cpu(),
            episode_data.actions.room_y.cpu(),
        )
        episode_count = min(self.config.visualize, episode_data.actions.room_idx.shape[0])
        total_frames = 0
        for episode_idx in range(episode_count):
            saved_paths = save_episode_frames(
                self.rooms,
                actions,
                output_root / f"episode_{episode_idx:04d}",
                self.config.map_size,
                episode_idx,
            )
            total_frames += len(saved_paths)

    def log_outcomes(
        self,
        episode_outcomes: EpisodeOutcomes,
        door_match_counts: DoorMatchCounts,
        loss: MainLossBreakdown,
        candidate_diagnostics: CandidateDiagnostics,
        generation_stats: GenerationStats,
        balance_loss: float,
        round_idx: int,
        step_config: Config,
    ) -> None:
        outcomes = episode_outcomes.step_outcomes
        door_invalid = torch.sum(outcomes.door_invalid != 0, dim=1)
        avg_door = torch.mean(door_invalid.to(torch.float32))
        min_door = torch.min(door_invalid)

        conn_invalid = torch.sum(outcomes.connection_invalid != 0, dim=1)
        avg_conn = torch.mean(conn_invalid.to(torch.float32))
        min_conn = torch.min(conn_invalid)

        toilet_invalid = (outcomes.toilet_invalid != 0).to(torch.int64)
        avg_toilet = torch.mean(toilet_invalid.to(torch.float32))

        phantoon_invalid = (outcomes.phantoon_invalid != 0).to(torch.int64)
        avg_phantoon = torch.mean(phantoon_invalid.to(torch.float32))

        total_invalid = door_invalid + conn_invalid + toilet_invalid + phantoon_invalid
        avg_invalid = torch.mean(total_invalid.to(torch.float32))
        min_invalid = torch.min(total_invalid)
        end_outcomes = episode_outcomes.end_outcomes
        avg_frontiers = torch.mean(end_outcomes.avg_frontiers.to(torch.float32))
        graph_diameter = torch.mean(end_outcomes.graph_diameter.to(torch.float32))
        save_distance_mask = end_outcomes.save_distance_mask.to(torch.float32)
        save_distance_mask_count = torch.sum(save_distance_mask)
        save_distance = torch.sum(
            end_outcomes.save_distance.to(torch.float32) * save_distance_mask
        ) / (save_distance_mask_count + 1e-15)
        save_distance_mask_fraction = torch.mean(save_distance_mask)
        refill_distance_mask = end_outcomes.refill_distance_mask.to(torch.float32)
        refill_distance_mask_count = torch.sum(refill_distance_mask)
        refill_distance = torch.sum(
            end_outcomes.refill_distance.to(torch.float32) * refill_distance_mask
        ) / (refill_distance_mask_count + 1e-15)
        refill_distance_mask_fraction = torch.mean(refill_distance_mask)
        active_room_part_mask = end_outcomes.active_room_part_mask.to(torch.float32)
        active_room_part_count = torch.sum(active_room_part_mask)
        save_to_room_distance_mask = end_outcomes.save_to_room_distance_mask.to(torch.float32)
        save_from_room_distance_mask = end_outcomes.save_from_room_distance_mask.to(torch.float32)
        refill_to_room_distance_mask = end_outcomes.refill_to_room_distance_mask.to(torch.float32)
        refill_from_room_distance_mask = end_outcomes.refill_from_room_distance_mask.to(
            torch.float32
        )
        save_to_room_utility = distance_proximity_utility(
            end_outcomes.save_to_room_distance,
            end_outcomes.save_to_room_distance_mask,
            step_config.distance_proximity_scale,
        )
        save_from_room_utility = distance_proximity_utility(
            end_outcomes.save_from_room_distance,
            end_outcomes.save_from_room_distance_mask,
            step_config.distance_proximity_scale,
        )
        refill_to_room_utility = distance_proximity_utility(
            end_outcomes.refill_to_room_distance,
            end_outcomes.refill_to_room_distance_mask,
            step_config.distance_proximity_scale,
        )
        refill_from_room_utility = distance_proximity_utility(
            end_outcomes.refill_from_room_distance,
            end_outcomes.refill_from_room_distance_mask,
            step_config.distance_proximity_scale,
        )
        save_proximity_utility = torch.sum(
            (save_to_room_utility + save_from_room_utility) * active_room_part_mask
        ) / (2.0 * active_room_part_count + 1e-15)
        refill_proximity_utility = torch.sum(
            (refill_to_room_utility + refill_from_room_utility) * active_room_part_mask
        ) / (2.0 * active_room_part_count + 1e-15)
        save_unreachable_fraction = torch.sum(
            (
                2.0 * active_room_part_mask
                - save_to_room_distance_mask
                - save_from_room_distance_mask
            )
        ) / (2.0 * active_room_part_count + 1e-15)
        refill_unreachable_fraction = torch.sum(
            (
                2.0 * active_room_part_mask
                - refill_to_room_distance_mask
                - refill_from_room_distance_mask
            )
        ) / (2.0 * active_room_part_count + 1e-15)
        missing_connect_distance_mask = end_outcomes.missing_connect_distance_mask.to(
            torch.float32
        )
        missing_connect_distance_mask_count = torch.sum(missing_connect_distance_mask)
        missing_connect_distance = torch.sum(
            end_outcomes.missing_connect_distance.to(torch.float32) * missing_connect_distance_mask
        ) / (missing_connect_distance_mask_count + 1e-15)
        missing_connect_distance_mask_fraction = torch.mean(missing_connect_distance_mask)
        missing_connect_utility_values = distance_proximity_utility(
            end_outcomes.missing_connect_distance,
            end_outcomes.missing_connect_distance_mask,
            step_config.distance_proximity_scale,
        )
        if missing_connect_utility_values.shape[1] == 0:
            missing_connect_utility = torch.sum(missing_connect_utility_values)
        else:
            missing_connect_utility = torch.mean(missing_connect_utility_values)

        success = total_invalid == 0
        success_rate = torch.mean(success.to(torch.float32))
        success_door = torch.mean((door_invalid == 0).to(torch.float32))
        success_conn = torch.mean((conn_invalid == 0).to(torch.float32))
        success_toilet = torch.mean((toilet_invalid == 0).to(torch.float32))
        success_phantoon = torch.mean((phantoon_invalid == 0).to(torch.float32))

        horizontal_door_match_counts = door_match_counts.horizontal[:-1, :-1].to(torch.float64)
        vertical_door_match_counts = door_match_counts.vertical[:-1, :-1].to(torch.float64)
        left_door_match_p = horizontal_door_match_counts / torch.sum(
            horizontal_door_match_counts, dim=1, keepdim=True
        )
        up_door_match_p = vertical_door_match_counts / torch.sum(
            vertical_door_match_counts, dim=1, keepdim=True
        )
        left_topk = torch.topk(left_door_match_p.flatten(), k=3).values
        up_topk = torch.topk(up_door_match_p.flatten(), k=3).values
        door_match_ss = (
            compute_door_match_count_ss(horizontal_door_match_counts, dim=1)
            + compute_door_match_count_ss(horizontal_door_match_counts, dim=0)
            + compute_door_match_count_ss(vertical_door_match_counts, dim=1)
            + compute_door_match_count_ss(vertical_door_match_counts, dim=0)
        )
        toilet_crossed_room_p = toilet_crossed_room_distribution(
            end_outcomes.toilet_crossed_room_idx,
            self.num_rooms,
        )
        toilet_crossed_room_topk = topk_or_zeros(toilet_crossed_room_p, 4)
        toilet_crossed_room_ss = torch.sum(toilet_crossed_room_p.square())
        with torch.no_grad():
            generate_config = create_generate_config(
                step_config,
                self.episode_length,
                1,
                self.device,
                False,
            )
            balance_preds = self.balance_model(torch.log(generate_config.temperature))
            balance_door_match_ss = compute_balance_door_match_ss(balance_preds)
            balance_left_topk = topk_or_zeros(
                torch.softmax(balance_preds.left, dim=-1).flatten(),
                3,
            )
            balance_up_topk = topk_or_zeros(
                torch.softmax(balance_preds.up, dim=-1).flatten(),
                3,
            )
            balance_toilet_crossed_room_p = torch.softmax(
                balance_preds.toilet_crossed_room,
                dim=-1,
            ).squeeze(0)
            balance_toilet_crossed_room_topk = topk_or_zeros(balance_toilet_crossed_room_p, 4)
            balance_toilet_crossed_room_ss = compute_balance_toilet_crossed_room_ss(balance_preds)
        loss_denominator = loss.total + 1e-15
        door_loss_pct = 100.0 * loss.door_contribution / loss_denominator
        connection_loss_pct = 100.0 * loss.connection_contribution / loss_denominator
        toilet_loss_pct = 100.0 * loss.toilet_contribution / loss_denominator
        phantoon_loss_pct = 100.0 * loss.phantoon_contribution / loss_denominator
        main_balance_loss_pct = 100.0 * loss.balance_contribution / loss_denominator
        main_toilet_balance_loss_pct = 100.0 * loss.toilet_balance_contribution / loss_denominator
        avg_frontiers_loss_pct = 100.0 * loss.avg_frontiers_contribution / loss_denominator
        graph_diameter_loss_pct = 100.0 * loss.graph_diameter_contribution / loss_denominator
        save_distance_loss_pct = 100.0 * loss.save_distance_contribution / loss_denominator
        refill_distance_loss_pct = 100.0 * loss.refill_distance_contribution / loss_denominator
        missing_connect_utility_loss_pct = (
            100.0 * loss.missing_connect_utility_contribution / loss_denominator
        )
        proposal_loss_pct = 100.0 * loss.proposal_contribution / loss_denominator

        metrics = {
            "loss": loss.total,
            "door_loss": loss.door,
            "door_loss_pct": door_loss_pct,
            "connection_loss": loss.connection,
            "connection_loss_pct": connection_loss_pct,
            "toilet_loss": loss.toilet,
            "toilet_loss_pct": toilet_loss_pct,
            "phantoon_loss": loss.phantoon,
            "phantoon_loss_pct": phantoon_loss_pct,
            "main_balance_loss": loss.balance,
            "main_balance_loss_pct": main_balance_loss_pct,
            "main_toilet_balance_loss": loss.toilet_balance,
            "main_toilet_balance_loss_pct": main_toilet_balance_loss_pct,
            "avg_frontiers_loss": loss.avg_frontiers,
            "avg_frontiers_loss_pct": avg_frontiers_loss_pct,
            "graph_diameter_loss": loss.graph_diameter,
            "graph_diameter_loss_pct": graph_diameter_loss_pct,
            "save_distance_loss": loss.save_distance,
            "save_distance_loss_pct": save_distance_loss_pct,
            "refill_distance_loss": loss.refill_distance,
            "refill_distance_loss_pct": refill_distance_loss_pct,
            "missing_connect_utility_loss": loss.missing_connect_utility,
            "missing_connect_utility_loss_pct": missing_connect_utility_loss_pct,
            "proposal_loss": loss.proposal,
            "proposal_loss_pct": proposal_loss_pct,
            "candidate_target_entropy": candidate_diagnostics.target_entropy,
            "candidate_uniform_kl": candidate_diagnostics.uniform_kl,
            "candidate_selected_probability": candidate_diagnostics.selected_probability,
            "balance_loss": balance_loss,
            "success_rate": success_rate,
            "success_door": success_door,
            "success_conn": success_conn,
            "success_toilet": success_toilet,
            "success_phantoon": success_phantoon,
            "avg_invalid": avg_invalid,
            "avg_frontiers": avg_frontiers,
            "graph_diameter": graph_diameter,
            "save_distance": save_distance,
            "save_distance_mask_fraction": save_distance_mask_fraction,
            "save_proximity_utility": save_proximity_utility,
            "save_unreachable_fraction": save_unreachable_fraction,
            "refill_distance": refill_distance,
            "refill_distance_mask_fraction": refill_distance_mask_fraction,
            "refill_proximity_utility": refill_proximity_utility,
            "refill_unreachable_fraction": refill_unreachable_fraction,
            "missing_connect_distance": missing_connect_distance,
            "missing_connect_distance_mask_fraction": missing_connect_distance_mask_fraction,
            "missing_connect_utility": missing_connect_utility,
            "avg_door": avg_door,
            "avg_conn": avg_conn,
            "avg_toilet": avg_toilet,
            "avg_phantoon": avg_phantoon,
            "min_invalid": min_invalid,
            "min_door": min_door,
            "min_conn": min_conn,
            "num_episodes": self.num_episodes,
            **optimizer_metric_values(step_config.optimizer),
            "temperature": variable_float_metric_value(
                step_config.generation.temperature,
                "generation.temperature",
            ),
            "recommended_candidates": step_config.generation.recommended_candidates,
            "shortlist_candidates": step_config.generation.shortlist_candidates,
            "proposal_temperature": variable_float_metric_value(
                step_config.generation.proposal_temperature,
                "generation.proposal_temperature",
            ),
            "reward_door": variable_float_metric_value(
                step_config.generation.reward_door,
                "generation.reward_door",
            ),
            "reward_connection": variable_float_metric_value(
                step_config.generation.reward_connection,
                "generation.reward_connection",
            ),
            "reward_toilet": variable_float_metric_value(
                step_config.generation.reward_toilet,
                "generation.reward_toilet",
            ),
            "reward_phantoon": variable_float_metric_value(
                step_config.generation.reward_phantoon,
                "generation.reward_phantoon",
            ),
            "reward_balance": variable_float_metric_value(
                step_config.generation.reward_balance,
                "generation.reward_balance",
            ),
            "reward_toilet_balance": variable_float_metric_value(
                step_config.generation.reward_toilet_balance,
                "generation.reward_toilet_balance",
            ),
            "reward_frontier": variable_float_metric_value(
                step_config.generation.reward_frontier,
                "generation.reward_frontier",
            ),
            "reward_graph_diameter": variable_float_metric_value(
                step_config.generation.reward_graph_diameter,
                "generation.reward_graph_diameter",
            ),
            "reward_save_distance": variable_float_metric_value(
                step_config.generation.reward_save_distance,
                "generation.reward_save_distance",
            ),
            "reward_refill_distance": variable_float_metric_value(
                step_config.generation.reward_refill_distance,
                "generation.reward_refill_distance",
            ),
            "reward_missing_connect_utility": (
                variable_float_metric_value(
                    step_config.generation.reward_missing_connect_utility,
                    "generation.reward_missing_connect_utility",
                )
            ),
            "distance_proximity_scale": step_config.distance_proximity_scale,
            "ema_decay": step_config.train.ema_decay,
            "toilet_weight": step_config.train.toilet_weight,
            "phantoon_weight": step_config.train.phantoon_weight,
            "toilet_balance_weight": step_config.train.toilet_balance_weight,
            "avg_frontiers_weight": step_config.train.avg_frontiers_weight,
            "graph_diameter_weight": step_config.train.graph_diameter_weight,
            "save_distance_weight": step_config.train.save_distance_weight,
            "refill_distance_weight": step_config.train.refill_distance_weight,
            "missing_connect_utility_weight": step_config.train.missing_connect_utility_weight,
            "door_match_left_top1": left_topk[0],
            "door_match_left_top2": left_topk[1],
            "door_match_left_top3": left_topk[2],
            "door_match_up_top1": up_topk[0],
            "door_match_up_top2": up_topk[1],
            "door_match_up_top3": up_topk[2],
            "door_match_ss": door_match_ss,
            "balance_door_match_left_top1": balance_left_topk[0],
            "balance_door_match_left_top2": balance_left_topk[1],
            "balance_door_match_left_top3": balance_left_topk[2],
            "balance_door_match_up_top1": balance_up_topk[0],
            "balance_door_match_up_top2": balance_up_topk[1],
            "balance_door_match_up_top3": balance_up_topk[2],
            "balance_door_match_ss": balance_door_match_ss,
            "toilet_crossed_room_top1": toilet_crossed_room_topk[0],
            "toilet_crossed_room_top2": toilet_crossed_room_topk[1],
            "toilet_crossed_room_top3": toilet_crossed_room_topk[2],
            "toilet_crossed_room_top4": toilet_crossed_room_topk[3],
            "toilet_crossed_room_ss": toilet_crossed_room_ss,
            "balance_toilet_crossed_room_top1": balance_toilet_crossed_room_topk[0],
            "balance_toilet_crossed_room_top2": balance_toilet_crossed_room_topk[1],
            "balance_toilet_crossed_room_top3": balance_toilet_crossed_room_topk[2],
            "balance_toilet_crossed_room_top4": balance_toilet_crossed_room_topk[3],
            "balance_toilet_crossed_room_ss": balance_toilet_crossed_room_ss,
            **generation_stats,
        }
        for name, value in metrics.items():
            self.aim_run.track(value, name=name, step=round_idx)

        def scalar(value):
            return value.item() if isinstance(value, torch.Tensor) else value

        schedule_progress = min(self.num_episodes / self.config.knot_episodes[-1], 1.0)
        logging.info(
            "round %s, loss %.4f (d %.1f%%, c %.1f%%, t %.1f%%, ph %.1f%%, "
            "b %.1f%%, tb %.1f%%, d %.1f%%, "
            "s %.1f%%, r %.1f%%, p %.1f%%), "
            "succ %.4f, total %.2f (min %s), door %.2f (min %s), "
            "conn %.2f (min %s), tube %.2f, diam %.2f, ss %.3f, "
            "p %.4f, "
            "frac %.4f",
            round_idx,
            loss.total,
            door_loss_pct,
            connection_loss_pct,
            toilet_loss_pct,
            phantoon_loss_pct,
            main_balance_loss_pct,
            main_toilet_balance_loss_pct,
            graph_diameter_loss_pct,
            save_distance_loss_pct,
            refill_distance_loss_pct,
            proposal_loss_pct,
            scalar(success_rate),
            scalar(avg_invalid),
            scalar(min_invalid),
            scalar(avg_door),
            scalar(min_door),
            scalar(avg_conn),
            scalar(min_conn),
            scalar(avg_toilet),
            scalar(graph_diameter),
            scalar(door_match_ss),
            scalar(candidate_diagnostics.selected_probability),
            schedule_progress,
        )
        # logging.info(
        #     "round %s horizontal_topk door connections: %s",
        #     round_idx,
        #     self.format_horizontal_topk_door_connections(
        #         horizontal_door_match_proportions,
        #         k=3,
        #     ),
        # )

    def log_profile_report(self, report: RustProfileReport, round_idx: int) -> None:
        rows = [(name, count, nanos) for name, count, nanos in report if count > 0 or nanos > 0]
        if not rows:
            logging.info("round %s Rust profile: no samples recorded", round_idx)
            return

        for section_name, prefix in [
            ("Python generation spans", "python."),
            ("worker commands", "worker."),
            ("environment step spans", "env.step."),
            ("environment proposal spans", "env.proposal."),
            ("environment lookahead spans", "env.lookahead."),
            ("environment feature spans", "env.features."),
            ("feature pack spans", "pack.features."),
            ("environment counters", "env.counter."),
        ]:
            section_rows = [row for row in rows if row[0].startswith(prefix)]
            if not section_rows:
                continue

            if prefix == "env.counter.":
                logging.info("round %s Rust profile: %s", round_idx, section_name)
                for name, count, _ in sorted(section_rows, key=lambda row: row[1], reverse=True):
                    logging.info("  %-55s %10s count", name, count)
                continue

            total_nanos = sum(nanos for _, _, nanos in section_rows)
            logging.info("round %s Rust profile: %s", round_idx, section_name)
            for name, count, nanos in sorted(section_rows, key=lambda row: row[2], reverse=True):
                total_ms = nanos / 1_000_000.0
                avg_us = nanos / count / 1_000.0 if count > 0 else 0.0
                pct = nanos / total_nanos * 100.0 if total_nanos > 0 else 0.0
                logging.info(
                    "  %-55s %10.3f ms %6.2f%% %8s calls %10.3f us/call",
                    name,
                    total_ms,
                    pct,
                    count,
                    avg_us,
                )

    def run(self) -> None:
        try:
            total_episodes = self.config.knot_episodes[-1]
            start_round = self.num_episodes // self.episodes_per_round
            for round_idx in range(start_round, total_episodes // self.episodes_per_round):
                if self.args.profile:
                    map_gen.reset_profile()
                (
                    episode_data,
                    episode_outcomes,
                    door_match_counts,
                    proposal_data,
                    generation_stats,
                    generation_profile,
                ) = self.generate_round()
                if self.config.visualize > 0:
                    self.visualize_round(episode_data, round_idx)
                candidate_diagnostics = compute_candidate_diagnostics(proposal_data)
                self.num_episodes += self.episodes_per_round
                step_config = instantiate_scheduleable_config(self.config, self.num_episodes)
                avg_loss, avg_balance_loss = self.train_round(
                    episode_data,
                    episode_outcomes,
                    proposal_data,
                    step_config,
                )
                episode_outcomes = episode_outcomes.to(torch.device("cpu"))

                self.experience.store(episode_data)

                self.log_outcomes(
                    episode_outcomes,
                    door_match_counts,
                    avg_loss,
                    candidate_diagnostics,
                    generation_stats,
                    avg_balance_loss,
                    round_idx,
                    step_config,
                )
                if self.args.profile:
                    parent_profile = map_gen.profile_report()
                    self.log_profile_report(
                        merge_profile_reports([parent_profile, generation_profile]),
                        round_idx,
                    )
                completed_round = round_idx + 1
                if completed_round % self.config.checkpoint_period == 0:
                    self.save_checkpoint(self.checkpoint_path(completed_round))

                if self.stop_requested:
                    logging.info("Stopping training after completing round %s.", round_idx)
                    break
        finally:
            self.train_batch_prefetcher.close()
            for generation_executor in self.generation_executors:
                generation_executor.shutdown()
            self.aim_run.close()


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--verify-outcome-consistency",
        action="store_true",
        help="fail if a known per-step outcome later changes",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "device selection: auto, cpu, cuda, or a comma-separated CUDA device list "
            "(default: auto; training uses the first selected device)"
        ),
    )
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        help="resume mutable training state from a safetensors checkpoint",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="log per-round Rust engine command timing",
    )
    parser.add_argument(
        "--ignore-scores",
        action="store_true",
        help="set generation temperature and proposal_temperature to a large finite value",
    )
    namespace = parser.parse_args()
    return Args(
        config=namespace.config,
        verify_outcome_consistency=namespace.verify_outcome_consistency,
        device=namespace.device,
        load_checkpoint=namespace.load_checkpoint,
        profile=namespace.profile,
        ignore_scores=namespace.ignore_scores,
    )


def select_devices(args: Args, config: Config) -> tuple[torch.device, list[torch.device]]:
    if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
        device = torch.device("cpu")
        generation_devices = [device]
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(f"--device {args.device} requested, but CUDA is not available")
        if args.device in ("auto", "cuda"):
            generation_devices = [
                torch.device(f"cuda:{index}") for index in range(config.generation.num_devices)
            ]
        else:
            try:
                generation_devices = [torch.device(value) for value in args.device.split(",")]
            except RuntimeError as error:
                raise ValueError(f"invalid --device value: {args.device}") from error
            if (
                not generation_devices
                or any(
                    generation_device.type != "cuda" for generation_device in generation_devices
                )
                or any(generation_device.index is None for generation_device in generation_devices)
            ):
                raise ValueError(
                    "--device must be auto, cpu, cuda, or a comma-separated list such as cuda:0,cuda:1"
                )
            if len(set(generation_devices)) != len(generation_devices):
                raise ValueError("--device CUDA list must not contain duplicates")
        device = generation_devices[0]
        torch.set_float32_matmul_precision("high")

    if device.type != "cuda" and config.generation.num_devices != 1:
        raise RuntimeError("generation.num_devices must be 1 when CUDA is not in use")
    if len(generation_devices) != config.generation.num_devices:
        raise RuntimeError(
            f"generation.num_devices={config.generation.num_devices}, but --device selected "
            f"{len(generation_devices)} device(s)"
        )
    invalid_cuda_devices = [
        str(generation_device)
        for generation_device in generation_devices
        if generation_device.type == "cuda"
        and generation_device.index >= torch.cuda.device_count()
    ]
    if invalid_cuda_devices:
        raise RuntimeError(
            f"CUDA device(s) not available: {', '.join(invalid_cuda_devices)}; "
            f"found {torch.cuda.device_count()} CUDA device(s)"
        )
    if device.type == "cuda" and (config.model.autocast or config.model.generation_autocast):
        unsupported_bf16_devices = []
        for generation_device in generation_devices:
            with torch.cuda.device(generation_device):
                if not torch.cuda.is_bf16_supported():
                    unsupported_bf16_devices.append(str(generation_device))
        if unsupported_bf16_devices:
            raise RuntimeError(
                "CUDA bfloat16 autocast requested, but these GPUs do not support bfloat16: "
                f"{', '.join(unsupported_bf16_devices)}. Use --device cpu for float32 CPU "
                "execution or set model.autocast=false and model.generation_autocast=false "
                "for float32 CUDA execution."
            )
    return device, generation_devices


def setup_logging(config: Config, args: Args) -> str:
    start_time = datetime.now()
    if args.load_checkpoint is not None:
        if args.load_checkpoint.parent.name != "checkpoints":
            raise ValueError(
                "--load-checkpoint must point to a file in a run's checkpoints directory"
            )
        run_path = f"{args.load_checkpoint.parent.parent}/"
    else:
        run_path = f"runs/{start_time.isoformat()}-{config.experiment_name}/"
    os.makedirs(run_path, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f"{run_path}/train-{start_time.isoformat()}.log"),
            logging.StreamHandler(),
        ],
    )

    logging.info("Config:\n%s", config.model_dump_json(indent=2))
    if args.verify_outcome_consistency:
        logging.info("Outcome consistency verification enabled.")
    if args.load_checkpoint is not None:
        logging.info("Loading checkpoint from %s", args.load_checkpoint)
    if args.profile:
        logging.info("Rust engine profiling enabled.")
    if args.ignore_scores:
        logging.info(
            "Generation scores ignored with sampling temperatures set to %s.",
            IGNORE_SCORES_TEMPERATURE,
        )
    return run_path


def create_train_batch_environment_groups(config: Config, engine: Engine):
    train_group_threads = (
        None
        if config.generation.num_threads is None
        else config.generation.num_threads // config.train.pipeline_groups
    )
    logging.info(
        "Using %s training pipeline group(s) with %s Rust worker(s) per group.",
        config.train.pipeline_groups,
        train_group_threads if train_group_threads is not None else "automatic",
    )
    return [
        engine.create_environment_group(
            config.map_size,
            config.train.batch_size,
            config.generation.candidate_spatial_cell_size,
            frontier_neighbor_algorithm=config.generation.frontier_neighbor_algorithm,
            frontier_neighbor_count=config.generation.frontier_neighbor_count,
            frontier_window_size=config.generation.frontier_window_size,
            num_threads=train_group_threads,
        )
        for _ in range(config.train.pipeline_groups)
    ]


def create_models(
    config: Config, rooms: list[dict], engine: Engine, device: torch.device, generation_devices
):
    main_model = FrontierModel(**frontier_model_kwargs(config, rooms, engine)).to(device)
    num_params = sum(p.numel() for p in main_model.parameters())
    logging.info(f"Main model parameters: {num_params}")
    logging.info(f"Main model: {main_model}")

    ema_model = copy.deepcopy(main_model).to(device)
    ema_model.requires_grad_(False)
    ema_model.eval()
    balance_model = create_balance_model(config, rooms, device)
    balance_num_params = sum(p.numel() for p in balance_model.parameters())
    logging.info(f"Balance model parameters: {balance_num_params}")
    if config.model.compile:
        main_model = torch.compile(main_model)
        ema_model = torch.compile(ema_model)

    return main_model, ema_model, balance_model


def create_generation_process_executors(
    config: Config,
    rooms: list[dict],
    generation_devices: list[torch.device],
    profile: bool,
    ignore_scores: bool,
) -> list[ProcessPoolExecutor]:
    logging.info(
        "Using %s generation process(es), one per generation device.",
        len(generation_devices),
    )
    context = multiprocessing.get_context("spawn")
    config_json = config.model_dump_json()
    rooms_json = json.dumps(rooms)
    return [
        ProcessPoolExecutor(
            max_workers=1,
            mp_context=context,
            initializer=initialize_generation_process,
            initargs=(
                config_json,
                rooms_json,
                str(generation_device),
                device_index,
                profile,
                ignore_scores,
            ),
        )
        for device_index, generation_device in enumerate(generation_devices)
    ]


def open_or_create_aim_run(run_hash: str, experiment_name: str) -> Run:
    try:
        return Run(
            run_hash,
            experiment=experiment_name,
            system_tracking_interval=None,
        )
    except MissingRunError:
        logging.warning(
            "Checkpoint references missing Aim run %s; creating a new Aim run.",
            run_hash,
        )
        return Run(experiment=experiment_name, system_tracking_interval=None)


def build_session(args: Args) -> TrainingSession:
    config = Config.model_validate_json(args.config.read_text())
    validate_config(config)
    map_gen.set_profile_enabled(args.profile)
    round_episode_count = episodes_per_round(config)
    run_path = setup_logging(config, args)
    rooms = json.loads(config.room_set.read_text())
    device, generation_devices = select_devices(args, config)

    train_precision = (
        "bfloat16 autocast" if device.type == "cuda" and config.model.autocast else "float32"
    )
    generation_precision = (
        "bfloat16 autocast"
        if device.type == "cuda" and config.model.generation_autocast
        else "float32"
    )
    logging.info(
        "Using device %s with %s training and %s generation across %s device(s): %s.",
        device,
        train_precision,
        generation_precision,
        len(generation_devices),
        ", ".join(str(generation_device) for generation_device in generation_devices),
    )

    engine = Engine(rooms, config.features)
    train_batch_envs = create_train_batch_environment_groups(config, engine)
    main_model, ema_model, balance_model = create_models(
        config,
        rooms,
        engine,
        device,
        generation_devices,
    )
    generation_executors = create_generation_process_executors(
        config,
        rooms,
        generation_devices,
        args.profile,
        args.ignore_scores,
    )
    initial_config = instantiate_scheduleable_config(config, 0)
    main_optimizer = create_main_optimizer(
        main_model,
        config.optimizer,
        initial_config.optimizer,
    )
    balance_optimizer = create_adam_optimizer(
        balance_model.parameters(),
        config.balance_optimizer,
        initial_config.balance_optimizer,
    )
    session = TrainingSession(
        args=args,
        config=config,
        run_path=run_path,
        rooms=rooms,
        device=device,
        generation_devices=generation_devices,
        engine=engine,
        train_batch_envs=train_batch_envs,
        main_model=main_model,
        ema_model=ema_model,
        balance_model=balance_model,
        main_optimizer=main_optimizer,
        balance_optimizer=balance_optimizer,
        loss_config=LossConfig(
            door_weight=config.train.door_weight,
            connection_weight=config.train.connection_weight,
            toilet_weight=config.train.toilet_weight,
            phantoon_weight=config.train.phantoon_weight,
            balance_weight=config.train.balance_weight,
            toilet_balance_weight=config.train.toilet_balance_weight,
            avg_frontiers_weight=config.train.avg_frontiers_weight,
            graph_diameter_weight=config.train.graph_diameter_weight,
            save_distance_weight=config.train.save_distance_weight,
            refill_distance_weight=config.train.refill_distance_weight,
            missing_connect_utility_weight=config.train.missing_connect_utility_weight,
            distance_proximity_scale=config.distance_proximity_scale,
        ),
        experience=ExperienceStorage(
            len(rooms),
            f"{run_path}/experience",
            round_episode_count,
        ),
        train_batch_prefetcher=Prefetcher(max_workers=config.train.pipeline_groups),
        generation_executors=generation_executors,
    )
    if args.load_checkpoint is not None:
        checkpoint_metadata = session.load_checkpoint(args.load_checkpoint)
        aim_run = open_or_create_aim_run(
            checkpoint_metadata["aim_run_hash"],
            config.experiment_name,
        )
    else:
        aim_run = Run(experiment=config.experiment_name, system_tracking_interval=None)
    aim_run["config"] = json.loads(config.model_dump_json())
    session.aim_run = aim_run
    return session


def main() -> None:
    args = parse_args()
    session = build_session(args)
    signal.signal(signal.SIGINT, lambda _signum, _frame: session.request_stop())
    signal.signal(signal.SIGTERM, lambda _signum, _frame: session.request_stop())
    session.run()


if __name__ == "__main__":
    main()
