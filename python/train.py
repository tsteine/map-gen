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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import safetensors.torch
import torch
import map_gen
from aim import Run
from safetensors import safe_open

from env import (
    Actions,
    DoorMatchCounts,
    DoorMatches,
    Engine,
    EpisodeData,
    GenerateConfig,
    Outcomes,
    Features,
    ProposalData,
)
from experience import ExperienceStorage
from generate import run_generation_groups
from loss import (
    LossConfig,
    compute_balance_door_match_ss,
    compute_balance_loss,
    compute_balance_score_target_logits,
    compute_loss_breakdown,
)
from model import BalanceModel, FrontierModel
from train_config import Config, episodes_per_round, instantiate_scheduleable_config, validate_config


@dataclass
class Args:
    config: Path
    verify_outcome_consistency: bool
    device: str
    load_checkpoint: Path | None
    profile: bool


type RustProfileReport = list[tuple[str, int, int]]


@dataclass
class TrainBatchTask:
    kind: Literal["fresh", "replay"]
    start: int | None
    env_index: int


@dataclass
class PreparedTrainBatch:
    kind: Literal["fresh", "replay"]
    episode_data: EpisodeData
    outcomes: Outcomes
    door_matches: DoorMatches
    prefix_count: int
    feature_batches: list["FeatureTrainBatch"]


@dataclass
class FeatureTrainBatch:
    features: Features
    proposal_frontier_idx: torch.Tensor | None
    proposal_door_variant_idx: torch.Tensor | None
    proposal_selected_candidate: torch.Tensor | None
    proposal_target_logits: torch.Tensor | None


@dataclass
class MainLossBreakdown:
    total: float
    door: float
    connection: float
    balance: float
    proposal: float
    door_contribution: float
    connection_contribution: float
    balance_contribution: float
    proposal_contribution: float


@dataclass
class CandidateDiagnostics:
    target_entropy: torch.Tensor
    uniform_kl: torch.Tensor
    selected_probability: torch.Tensor


def compute_door_match_count_ss(counts: torch.Tensor, dim: int) -> torch.Tensor:
    totals = torch.sum(counts, dim=dim, keepdim=True)
    if torch.any(totals <= 1):
        raise RuntimeError("door_match_ss requires at least two samples per row/column")
    return torch.sum(counts * (counts - 1) / (totals * (totals - 1)))


def compute_candidate_diagnostics(proposal_data: ProposalData) -> CandidateDiagnostics:
    target_logits = proposal_data.target_logits.to(torch.float32)
    valid = (
        (proposal_data.frontier_idx >= 0)
        & (proposal_data.door_variant_idx >= 0)
        & torch.isfinite(target_logits)
    )
    candidate_count = target_logits.shape[-1]
    flat_logits = target_logits.reshape(-1, candidate_count)
    flat_valid = valid.reshape(-1, candidate_count)
    row_valid = torch.any(flat_valid, dim=1)
    if not torch.any(row_valid):
        zero = torch.sum(target_logits) * 0.0
        return CandidateDiagnostics(zero, zero, zero)

    row_logits = torch.where(
        flat_valid[row_valid],
        flat_logits[row_valid],
        torch.full_like(flat_logits[row_valid], float("-inf")),
    )
    row_mask = flat_valid[row_valid]
    target_log_probs = torch.nn.functional.log_softmax(row_logits, dim=1)
    safe_target_log_probs = torch.where(
        row_mask,
        target_log_probs,
        torch.zeros_like(target_log_probs),
    )
    target_probs = torch.where(
        row_mask,
        torch.exp(target_log_probs),
        torch.zeros_like(target_log_probs),
    )
    entropy_per_row = torch.sum(-target_probs * safe_target_log_probs, dim=1)
    target_entropy = torch.mean(entropy_per_row)
    valid_counts = torch.sum(row_mask, dim=1).to(torch.float32)
    uniform_kl = torch.mean(torch.log(valid_counts) - entropy_per_row)

    selected_candidate = proposal_data.selected_candidate.reshape(-1)[row_valid].to(torch.int64)
    selected_in_range = (
        (selected_candidate >= 0)
        & (selected_candidate < candidate_count)
    )
    safe_selected_candidate = selected_candidate.clamp_min(0).clamp_max(candidate_count - 1)
    selected_valid = selected_in_range & torch.gather(
        row_mask,
        1,
        safe_selected_candidate.unsqueeze(1),
    ).squeeze(1)
    if torch.any(selected_valid):
        selected_probability = torch.mean(torch.gather(
            target_probs[selected_valid],
            1,
            selected_candidate[selected_valid].unsqueeze(1),
        ).squeeze(1))
    else:
        selected_probability = torch.sum(target_logits) * 0.0
    return CandidateDiagnostics(target_entropy, uniform_kl, selected_probability)


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
        for name, value in module.state_dict().items()
    }


def without_prefix(
    tensors: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    full_prefix = f"{prefix}."
    return {
        name[len(full_prefix):]: value
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
        suffix = key[len(state_prefix):]
        param_id_text, state_name = suffix.split(".", 1)
        state.setdefault(int(param_id_text), {})[state_name] = value
    for param_id_text, param_scalar_state in scalar_state.items():
        state.setdefault(int(param_id_text), {}).update(param_scalar_state)
    optimizer.load_state_dict({
        "state": state,
        "param_groups": param_groups,
    })


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
        "door_match_embedding_width": config.model.door_match_embedding_width,
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


def create_balance_model(config: Config, rooms: list[dict], device: torch.device) -> torch.nn.Module:
    return BalanceModel(
        left_count=count_room_doors_by_direction(rooms, "left"),
        right_count=count_room_doors_by_direction(rooms, "right"),
        up_count=count_room_doors_by_direction(rooms, "up"),
        down_count=count_room_doors_by_direction(rooms, "down"),
        hidden_width=config.balance_model.hidden_width,
        num_layers=config.balance_model.num_layers,
    ).to(device)


def create_generation_environment_groups_for_device(
    config: Config,
    engine: Engine,
    device_index: int,
):
    num_generation_groups = (
        config.generation.num_devices * config.generation.pipeline_groups
    )
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
) -> GenerateConfig:
    return GenerateConfig(
        episode_length=episode_length,
        recommended_candidates=config.generation.recommended_candidates,
        exploration_candidates=config.generation.exploration_candidates,
        temperature=torch.full(
            [num_envs],
            config.generation.temperature,
            dtype=torch.float32,
            device=device,
        ),
        proposal_temperature=torch.full(
            [num_envs],
            config.generation.proposal_temperature,
            dtype=torch.float32,
            device=device,
        ),
        reward_door=config.generation.reward_door,
        reward_connection=config.generation.reward_connection,
        reward_balance=config.generation.reward_balance,
        autocast=config.model.generation_autocast,
    )


@dataclass
class GenerationProcessState:
    config: Config
    episode_length: int
    device: torch.device
    envs: list
    model: torch.nn.Module
    profile: bool


GENERATION_PROCESS_STATE: GenerationProcessState | None = None


def initialize_generation_process(
    config_json: str,
    rooms_json: str,
    device_text: str,
    device_index: int,
    profile: bool,
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
        model = torch.compile(model, dynamic=True)
    map_gen.set_profile_enabled(profile)
    GENERATION_PROCESS_STATE = GenerationProcessState(
        config,
        len(rooms),
        device,
        envs,
        model,
        profile,
    )


def run_generation_process_task(
    model_state: dict[str, torch.Tensor],
    generation_config_json: str,
    verify_outcome_consistency: bool,
) -> tuple[EpisodeData, Outcomes, DoorMatchCounts, ProposalData, RustProfileReport]:
    if GENERATION_PROCESS_STATE is None:
        raise RuntimeError("generation process was not initialized")
    state = GENERATION_PROCESS_STATE
    if state.profile:
        map_gen.reset_profile()
    state.model.load_state_dict(model_state)
    generation_config = Config.model_validate_json(generation_config_json)
    gen_configs = [
        create_generate_config(
            generation_config,
            state.episode_length,
            env.num_envs,
            state.device,
        )
        for env in state.envs
    ]
    (
        episode_data,
        outcomes,
        door_match_counts,
        proposal_data,
        python_profile_report,
    ) = run_generation_groups(
        state.envs,
        state.model,
        gen_configs,
        state.device,
        verify_outcome_consistency=verify_outcome_consistency,
        profile=state.profile,
    )
    profile_report = (
        map_gen.profile_report() + python_profile_report
        if state.profile
        else []
    )
    return (
        episode_data.to(torch.device("cpu")),
        outcomes.to(torch.device("cpu")),
        door_match_counts.to(torch.device("cpu")),
        proposal_data.to(torch.device("cpu")),
        profile_report,
    )


def merge_profile_reports(reports: list[RustProfileReport]) -> RustProfileReport:
    merged = {}
    for report in reports:
        for name, count, nanos in report:
            merged_count, merged_nanos = merged.get(name, (0, 0))
            merged[name] = (merged_count + count, merged_nanos + nanos)
    return [
        (name, count, nanos)
        for name, (count, nanos) in merged.items()
    ]


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
    main_optimizer: torch.optim.Optimizer
    balance_optimizer: torch.optim.Optimizer
    aim_run: Run
    loss_config: LossConfig
    experience: ExperienceStorage
    train_batch_prefetcher: Prefetcher
    generation_executors: list[ProcessPoolExecutor]
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
                f"top{rank}: {left_door_labels[left_idx]} -> {right_door_labels[right_idx]} "
                f"({value:.4f})"
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
        optimizer_tensors, optimizer_param_groups, optimizer_scalar_state = optimizer_checkpoint_state(
            self.main_optimizer,
            "optimizer",
        )
        balance_optimizer_tensors, balance_optimizer_param_groups, balance_optimizer_scalar_state = optimizer_checkpoint_state(
            self.balance_optimizer,
            "balance_optimizer",
        )
        tensors.update(optimizer_tensors)
        tensors.update(balance_optimizer_tensors)
        metadata = {
            "format": "map-gen-training-session-checkpoint-v1",
            "num_episodes": str(self.num_episodes),
            "experience_num_files": str(self.experience.num_files),
            "optimizer_param_groups": json.dumps(optimizer_param_groups),
            "optimizer_scalar_state": json.dumps(optimizer_scalar_state),
            "balance_optimizer_param_groups": json.dumps(balance_optimizer_param_groups),
            "balance_optimizer_scalar_state": json.dumps(balance_optimizer_scalar_state),
        }
        temp_path = path.with_suffix(f"{path.suffix}.tmp")
        safetensors.torch.save_file(tensors, temp_path, metadata=metadata)
        os.replace(temp_path, path)
        logging.info("Saved checkpoint: %s", path)

    def load_checkpoint(self, path: Path) -> None:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
            tensors = {name: checkpoint.get_tensor(name) for name in checkpoint.keys()}
        if metadata.get("format") != "map-gen-training-session-checkpoint-v1":
            raise ValueError(f"unsupported checkpoint format in {path}")

        self.main_model.load_state_dict(without_prefix(tensors, "main_model"))
        self.ema_model.load_state_dict(without_prefix(tensors, "ema_model"))
        self.balance_model.load_state_dict(without_prefix(tensors, "balance_model"))
        load_optimizer_checkpoint_state(
            self.main_optimizer,
            tensors,
            json.loads(metadata["optimizer_param_groups"]),
            json.loads(metadata["optimizer_scalar_state"]),
            "optimizer",
        )
        load_optimizer_checkpoint_state(
            self.balance_optimizer,
            tensors,
            json.loads(metadata["balance_optimizer_param_groups"]),
            json.loads(metadata["balance_optimizer_scalar_state"]),
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

    def update_ema_model(self) -> None:
        with torch.no_grad():
            for ema_param, main_param in zip(self.ema_model.parameters(), self.main_model.parameters()):
                ema_param.lerp_(main_param, 1.0 - self.config.train.ema_decay)

    def select_batch(
        self,
        episode_data: EpisodeData,
        outcomes: Outcomes,
        start: int,
    ) -> tuple[EpisodeData, Outcomes]:
        end = start + self.config.train.batch_size
        return (
            episode_data.slice(start, end),
            Outcomes(
                door_invalid=outcomes.door_invalid[start:end],
                connection_invalid=outcomes.connection_invalid[start:end],
                door_match=outcomes.door_match[start:end],
            ),
        )

    def iter_fresh_batch_starts(self) -> range:
        num_batches = int(
            math.ceil(
                self.episodes_per_round
                * self.config.train.fresh_pass_factor
                / self.config.train.batch_size
            )
        )
        return range(num_batches)

    def iter_train_batch_tasks(self) -> list[TrainBatchTask]:
        tasks = []
        task_idx = 0
        for batch_idx in self.iter_fresh_batch_starts():
            start = (batch_idx * self.config.train.batch_size) % self.episodes_per_round
            tasks.append(TrainBatchTask("fresh", start, task_idx % self.train_pipeline_groups))
            task_idx += 1
        if self.experience.num_files > 0:
            replay_batches = int(
                math.ceil(
                    self.episodes_per_round
                    * self.config.train.replay_pass_factor
                    / self.config.train.batch_size
                )
            )
            for _ in range(replay_batches):
                tasks.append(TrainBatchTask("replay", None, task_idx % self.train_pipeline_groups))
                task_idx += 1
        return tasks

    def prepare_feature_batches(
        self,
        train_episode_data: EpisodeData,
        proposal_data: ProposalData | None,
        env,
    ) -> tuple[int, list[FeatureTrainBatch]]:
        offset = torch.randint(0, self.config.train.sample_period, [1]).item()
        train_actions = train_episode_data.actions
        train_actions_cpu = train_actions.to(torch.device("cpu"))
        log_temperature = torch.log(train_episode_data.temperature).to(torch.device("cpu"))
        log_recommended_candidates = torch.log(train_episode_data.recommended_candidates + 1).to(
            torch.device("cpu")
        )
        log_exploration_candidates = torch.log(train_episode_data.exploration_candidates + 1).to(
            torch.device("cpu")
        )
        env.clear()
        feature_batches = []
        for step in range(self.episode_length):
            next_actions = Actions(
                train_actions_cpu.room_idx[:, step],
                train_actions_cpu.room_x[:, step],
                train_actions_cpu.room_y[:, step],
            )
            sample_step = step % self.config.train.sample_period == offset
            if sample_step:
                next_lookahead_outcomes = env.get_outcomes_after_candidates(
                    Actions(
                        next_actions.room_idx.unsqueeze(1),
                        next_actions.room_x.unsqueeze(1),
                        next_actions.room_y.unsqueeze(1),
                    ),
                    torch.device("cpu"),
                    0,
                )
                dummy_action = next_actions.room_idx >= self.num_rooms
                next_lookahead_outcomes = Outcomes(
                    torch.where(
                        dummy_action[:, None, None],
                        torch.full_like(next_lookahead_outcomes.door_invalid, -1),
                        next_lookahead_outcomes.door_invalid,
                    ),
                    torch.where(
                        dummy_action[:, None, None],
                        torch.full_like(next_lookahead_outcomes.connection_invalid, -1),
                        next_lookahead_outcomes.connection_invalid,
                    ),
                    torch.where(
                        dummy_action[:, None, None],
                        torch.full_like(next_lookahead_outcomes.door_match, -1),
                        next_lookahead_outcomes.door_match,
                    ),
                )
            if self.config.features.lookahead_outcomes:
                env.step(next_actions)
            else:
                env.step_known(next_actions)
            if sample_step:
                proposal_frontier_idx = None
                proposal_door_variant_idx = None
                proposal_selected_candidate = None
                proposal_target_logits = None
                if proposal_data is not None and step + 1 < self.episode_length:
                    proposal_frontier_idx = proposal_data.frontier_idx[:, step + 1]
                    proposal_door_variant_idx = proposal_data.door_variant_idx[:, step + 1]
                    proposal_selected_candidate = proposal_data.selected_candidate[:, step + 1]
                    proposal_target_logits = proposal_data.target_logits[:, step + 1]
                feature_batches.append(
                    FeatureTrainBatch(
                        env.get_features(
                            torch.device("cpu"),
                            log_temperature,
                            self.config.features.temperature,
                            log_recommended_candidates,
                            self.config.features.recommended_candidates,
                            log_exploration_candidates,
                            self.config.features.exploration_candidates,
                            Outcomes(
                                next_lookahead_outcomes.door_invalid.squeeze(1),
                                next_lookahead_outcomes.connection_invalid.squeeze(1),
                                next_lookahead_outcomes.door_match.squeeze(1),
                            ),
                            self.config.features.lookahead_outcomes,
                            0,
                            train_actions.room_idx.shape[0],
                        ),
                        proposal_frontier_idx,
                        proposal_door_variant_idx,
                        proposal_selected_candidate,
                        proposal_target_logits,
                    )
                )
        return len(feature_batches), feature_batches

    def prepare_feature_batch(
        self,
        kind: Literal["fresh", "replay"],
        train_episode_data: EpisodeData,
        train_outcomes: Outcomes,
        proposal_data: ProposalData | None,
        env,
    ) -> PreparedTrainBatch:
        prefix_count, feature_batches = self.prepare_feature_batches(
            train_episode_data,
            proposal_data,
            env,
        )
        door_matches = env.get_door_matches(self.device)
        return PreparedTrainBatch(
            kind,
            train_episode_data,
            train_outcomes,
            door_matches,
            prefix_count=prefix_count,
            feature_batches=feature_batches,
        )

    def prepare_train_batch_task(
        self,
        task: TrainBatchTask,
        fresh_episode_data: EpisodeData,
        fresh_outcomes: Outcomes,
        fresh_proposal_data: ProposalData,
    ) -> PreparedTrainBatch:
        env = self.train_batch_envs[task.env_index]
        if task.kind == "fresh":
            assert task.start is not None
            train_episode_data, train_outcomes = self.select_batch(
                fresh_episode_data,
                fresh_outcomes,
                task.start,
            )
            train_proposal_data = fresh_proposal_data.slice(
                task.start,
                task.start + self.config.train.batch_size,
            )
            return self.prepare_feature_batch(
                task.kind,
                train_episode_data,
                train_outcomes,
                train_proposal_data,
                env,
            )

        replay_episode_data = self.experience.sample(
            self.config.train.batch_size,
            self.config.train.episodes_per_file,
            self.config.train.hist_c,
        )
        prefix_count, feature_batches = self.prepare_feature_batches(
            replay_episode_data,
            None,
            env,
        )
        replay_door_matches = env.get_door_matches(self.device)
        env.finish()
        replay_episode_data = replay_episode_data.to(self.device)
        replay_outcomes = env.get_outcomes(self.device, verify_consistency=False)
        return PreparedTrainBatch(
            task.kind,
            replay_episode_data,
            replay_outcomes,
            replay_door_matches,
            prefix_count=prefix_count,
            feature_batches=feature_batches,
        )

    def train_batch_backward(
        self,
        prepared_batch: PreparedTrainBatch,
        loss_scale: float,
    ) -> tuple[MainLossBreakdown, float]:
        loss = self.train_feature_batch_backward(prepared_batch, loss_scale)
        balance_loss = self.train_balance_batch_backward(prepared_batch, loss_scale)

        if not math.isfinite(loss.total):
            raise RuntimeError(f"non-finite loss before backward: {loss.total}")
        if not torch.isfinite(balance_loss):
            raise RuntimeError(f"non-finite balance loss before backward: {balance_loss.item()}")

        return loss, balance_loss.item()

    def train_balance_batch_backward(
        self,
        prepared_batch: PreparedTrainBatch,
        loss_scale: float,
    ) -> torch.Tensor:
        log_temperature = torch.log(prepared_batch.episode_data.temperature)
        preds = self.balance_model(log_temperature)
        balance_loss = compute_balance_loss(preds, prepared_batch.door_matches)
        (balance_loss * loss_scale).backward()
        return balance_loss

    def proposal_batch_loss(
        self,
        proposal_score: torch.Tensor,
        frontier_idx: torch.Tensor,
        door_variant_idx: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> torch.Tensor:
        frontier_idx = frontier_idx.to(self.device, dtype=torch.int64)
        door_variant_idx = door_variant_idx.to(self.device, dtype=torch.int64)
        target_logits = target_logits.to(self.device, dtype=torch.float32)
        valid = (frontier_idx >= 0) & (door_variant_idx >= 0) & torch.isfinite(target_logits)
        safe_frontier_idx = frontier_idx.clamp_min(0)
        safe_door_variant_idx = door_variant_idx.clamp_min(0)
        batch_idx = torch.arange(
            frontier_idx.shape[0],
            dtype=torch.int64,
            device=self.device,
        ).unsqueeze(1)
        candidate_logits = proposal_score[
            batch_idx,
            safe_frontier_idx,
            safe_door_variant_idx,
        ]
        candidate_logits = torch.where(
            valid,
            candidate_logits,
            torch.full_like(candidate_logits, float("-inf")),
        ).to(torch.float32)
        target_logits = torch.where(
            valid,
            target_logits,
            torch.full_like(target_logits, float("-inf")),
        )
        row_valid = torch.any(valid, dim=1)
        if not torch.any(row_valid):
            return torch.sum(proposal_score) * 0.0
        row_candidate_logits = candidate_logits[row_valid]
        row_target_logits = target_logits[row_valid]
        row_mask = valid[row_valid]
        proposal_log_probs = torch.nn.functional.log_softmax(
            row_candidate_logits,
            dim=1,
        )
        target_log_probs = torch.nn.functional.log_softmax(
            row_target_logits,
            dim=1,
        )
        safe_target_log_probs = torch.where(
            row_mask,
            target_log_probs,
            torch.zeros_like(target_log_probs),
        )
        safe_proposal_log_probs = torch.where(
            row_mask,
            proposal_log_probs,
            torch.zeros_like(proposal_log_probs),
        )
        target_probs = torch.where(
            row_mask,
            torch.exp(target_log_probs),
            torch.zeros_like(target_log_probs),
        )
        kl_terms = target_probs * (safe_target_log_probs - safe_proposal_log_probs)
        proposal_loss = (
            torch.sum(torch.where(row_mask, kl_terms, torch.zeros_like(kl_terms)))
            / row_mask.shape[0]
        )
        return proposal_loss

    def train_feature_batch_backward(
        self,
        prepared_batch: PreparedTrainBatch,
        loss_scale: float,
    ) -> MainLossBreakdown:
        if prepared_batch.prefix_count == 0:
            raise RuntimeError("feature training batch has no sampled prefixes")

        train_outcomes = prepared_batch.outcomes
        repeated_outcomes = Outcomes(
            door_invalid=train_outcomes.door_invalid.unsqueeze(1),
            connection_invalid=train_outcomes.connection_invalid.unsqueeze(1),
            door_match=train_outcomes.door_match.unsqueeze(1),
        )
        with torch.no_grad():
            balance_preds = self.balance_model(torch.log(prepared_batch.episode_data.temperature))
            balance_score_target_logits, balance_score_mask = compute_balance_score_target_logits(
                balance_preds,
                prepared_batch.door_matches,
            )
        repeated_balance_score_target_logits = balance_score_target_logits.unsqueeze(1)
        repeated_balance_score_mask = balance_score_mask.unsqueeze(1)
        mask = torch.ones(
            [prepared_batch.episode_data.actions.room_idx.shape[0], 1, 1],
            dtype=torch.bool,
            device=self.device,
        )
        total_loss = MainLossBreakdown(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )
        prefix_weight = 1.0 / prepared_batch.prefix_count

        for feature_batch in prepared_batch.feature_batches:
            features = feature_batch.features.to(self.device)
            include_proposal = (
                prepared_batch.kind == "fresh"
                and feature_batch.proposal_frontier_idx is not None
                and feature_batch.proposal_door_variant_idx is not None
                and feature_batch.proposal_selected_candidate is not None
                and feature_batch.proposal_target_logits is not None
            )
            with torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda" and self.config.model.autocast,
            ):
                preds = self.main_model(features, include_proposal=include_proposal)
            prefix_loss = compute_loss_breakdown(
                preds,
                repeated_outcomes,
                mask,
                repeated_balance_score_target_logits,
                repeated_balance_score_mask,
                self.loss_config,
            )
            backward_loss = prefix_loss.total * prefix_weight
            total_loss.total += prefix_loss.total.item() * prefix_weight
            total_loss.door += prefix_loss.door.item() * prefix_weight
            total_loss.connection += prefix_loss.connection.item() * prefix_weight
            total_loss.balance += prefix_loss.balance.item() * prefix_weight
            total_loss.door_contribution += prefix_loss.door_contribution.item() * prefix_weight
            total_loss.connection_contribution += (
                prefix_loss.connection_contribution.item() * prefix_weight
            )
            total_loss.balance_contribution += (
                prefix_loss.balance_contribution.item() * prefix_weight
            )
            if include_proposal:
                batch_proposal_loss = self.proposal_batch_loss(
                    preds.proposal_score,
                    feature_batch.proposal_frontier_idx,
                    feature_batch.proposal_door_variant_idx,
                    feature_batch.proposal_target_logits,
                )
                weighted_proposal_loss = (
                    self.config.train.proposal_weight
                    * batch_proposal_loss
                    * prefix_weight
                )
                backward_loss = backward_loss + weighted_proposal_loss
                total_loss.total += weighted_proposal_loss.item()
                total_loss.proposal += batch_proposal_loss.item() * prefix_weight
                total_loss.proposal_contribution += weighted_proposal_loss.item()
            (backward_loss * loss_scale).backward()
        return total_loss

    def train_optimizer_step(self) -> None:
        grad_norm = torch.nn.utils.clip_grad_norm_(self.main_model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient norm: {grad_norm.item()}")
        balance_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.balance_model.parameters(),
            max_norm=1.0,
        )
        if not torch.isfinite(balance_grad_norm):
            raise RuntimeError(f"non-finite balance gradient norm: {balance_grad_norm.item()}")
        self.main_optimizer.step()
        self.balance_optimizer.step()
        self.update_ema_model()

    def generate_round(
        self,
    ) -> tuple[EpisodeData, Outcomes, DoorMatchCounts, ProposalData, RustProfileReport]:
        episode_data_iterations = []
        outcome_iterations = []
        door_match_count_iterations = []
        proposal_data_iterations = []
        profile_reports = []
        model_state = {
            name: as_checkpoint_tensor(value)
            for name, value in self.ema_model.state_dict().items()
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
                iteration_profile_report,
            ) in shard_results:
                episode_data_iterations.append(iteration_episode_data.to(self.device))
                outcome_iterations.append(iteration_outcomes.to(self.device))
                door_match_count_iterations.append(iteration_door_match_counts.to(self.device))
                proposal_data_iterations.append(iteration_proposal_data.to(self.device))
                profile_reports.append(iteration_profile_report)

        return (
            EpisodeData(
                actions=Actions(
                    room_idx=torch.cat([
                        episode_data.actions.room_idx for episode_data in episode_data_iterations
                    ]),
                    room_x=torch.cat([
                        episode_data.actions.room_x for episode_data in episode_data_iterations
                    ]),
                    room_y=torch.cat([
                        episode_data.actions.room_y for episode_data in episode_data_iterations
                    ]),
                ),
                temperature=torch.cat([
                    episode_data.temperature for episode_data in episode_data_iterations
                ]),
                recommended_candidates=torch.cat([
                    episode_data.recommended_candidates
                    for episode_data in episode_data_iterations
                ]),
                exploration_candidates=torch.cat([
                    episode_data.exploration_candidates
                    for episode_data in episode_data_iterations
                ]),
            ),
            Outcomes(
                door_invalid=torch.cat([outcomes.door_invalid for outcomes in outcome_iterations]),
                connection_invalid=torch.cat(
                    [outcomes.connection_invalid for outcomes in outcome_iterations]
                ),
                door_match=torch.cat([outcomes.door_match for outcomes in outcome_iterations]),
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
                frontier_idx=torch.cat([
                    proposal_data.frontier_idx
                    for proposal_data in proposal_data_iterations
                ]),
                door_variant_idx=torch.cat([
                    proposal_data.door_variant_idx
                    for proposal_data in proposal_data_iterations
                ]),
                selected_candidate=torch.cat([
                    proposal_data.selected_candidate
                    for proposal_data in proposal_data_iterations
                ]),
                target_logits=torch.cat([
                    proposal_data.target_logits
                    for proposal_data in proposal_data_iterations
                ]),
            ),
            merge_profile_reports(profile_reports),
        )

    def train_round(
        self,
        episode_data: EpisodeData,
        gen_outcomes: Outcomes,
        proposal_data: ProposalData,
        step_config: Config,
    ) -> tuple[MainLossBreakdown, float]:
        self.main_optimizer.param_groups[0]["lr"] = step_config.optimizer.lr
        self.balance_optimizer.param_groups[0]["lr"] = step_config.balance_optimizer.lr

        total_loss = MainLossBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        total_balance_loss = 0.0
        train_batch_count = 0

        def prepare_train_task(task: TrainBatchTask) -> PreparedTrainBatch:
            return self.prepare_train_batch_task(
                task,
                episode_data,
                gen_outcomes,
                proposal_data,
            )

        def train_prepared_batch_group(prepared_batches: list[PreparedTrainBatch]) -> tuple[MainLossBreakdown, float, int]:
            self.main_model.zero_grad()
            self.balance_model.zero_grad()
            loss_scale = 1.0 / len(prepared_batches)
            group_loss = MainLossBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            group_balance_loss = 0.0
            for prepared_batch in prepared_batches:
                batch_loss, batch_balance_loss = self.train_batch_backward(
                    prepared_batch,
                    loss_scale,
                )
                group_loss.total += batch_loss.total
                group_loss.door += batch_loss.door
                group_loss.connection += batch_loss.connection
                group_loss.balance += batch_loss.balance
                group_loss.proposal += batch_loss.proposal
                group_loss.door_contribution += batch_loss.door_contribution
                group_loss.connection_contribution += batch_loss.connection_contribution
                group_loss.balance_contribution += batch_loss.balance_contribution
                group_loss.proposal_contribution += batch_loss.proposal_contribution
                group_balance_loss += batch_balance_loss
            self.train_optimizer_step()
            return group_loss, group_balance_loss, len(prepared_batches)

        prepared_batch_group = []
        for prepared_batch in self.train_batch_prefetcher.map(
            self.iter_train_batch_tasks(),
            prepare_train_task,
        ):
            prepared_batch_group.append(prepared_batch)
            if len(prepared_batch_group) == self.config.train.gradient_accumulation_steps:
                group_loss, group_balance_loss, group_count = train_prepared_batch_group(prepared_batch_group)
                total_loss.total += group_loss.total
                total_loss.door += group_loss.door
                total_loss.connection += group_loss.connection
                total_loss.balance += group_loss.balance
                total_loss.proposal += group_loss.proposal
                total_loss.door_contribution += group_loss.door_contribution
                total_loss.connection_contribution += group_loss.connection_contribution
                total_loss.balance_contribution += group_loss.balance_contribution
                total_loss.proposal_contribution += group_loss.proposal_contribution
                total_balance_loss += group_balance_loss
                train_batch_count += group_count
                prepared_batch_group = []
        if prepared_batch_group:
            group_loss, group_balance_loss, group_count = train_prepared_batch_group(prepared_batch_group)
            total_loss.total += group_loss.total
            total_loss.door += group_loss.door
            total_loss.connection += group_loss.connection
            total_loss.balance += group_loss.balance
            total_loss.proposal += group_loss.proposal
            total_loss.door_contribution += group_loss.door_contribution
            total_loss.connection_contribution += group_loss.connection_contribution
            total_loss.balance_contribution += group_loss.balance_contribution
            total_loss.proposal_contribution += group_loss.proposal_contribution
            total_balance_loss += group_balance_loss
            train_batch_count += group_count

        if train_batch_count == 0:
            return MainLossBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0
        return (
            MainLossBreakdown(
                total_loss.total / train_batch_count,
                total_loss.door / train_batch_count,
                total_loss.connection / train_batch_count,
                total_loss.balance / train_batch_count,
                total_loss.proposal / train_batch_count,
                total_loss.door_contribution / train_batch_count,
                total_loss.connection_contribution / train_batch_count,
                total_loss.balance_contribution / train_batch_count,
                total_loss.proposal_contribution / train_batch_count,
            ),
            total_balance_loss / train_batch_count,
        )

    def log_outcomes(
        self,
        outcomes: Outcomes,
        door_match_counts: DoorMatchCounts,
        loss: MainLossBreakdown,
        candidate_diagnostics: CandidateDiagnostics,
        balance_loss: float,
        round_idx: int,
        step_config: Config,
    ) -> None:
        door_invalid = torch.sum(outcomes.door_invalid != 0, dim=1)
        avg_door = torch.mean(door_invalid.to(torch.float32))
        min_door = torch.min(door_invalid)

        conn_invalid = torch.sum(outcomes.connection_invalid != 0, dim=1)
        avg_conn = torch.mean(conn_invalid.to(torch.float32))
        min_conn = torch.min(conn_invalid)

        total_invalid = door_invalid + conn_invalid
        avg_invalid = torch.mean(total_invalid.to(torch.float32))
        min_invalid = torch.min(total_invalid)

        success = total_invalid == 0
        success_rate = torch.mean(success.to(torch.float32))
        success_door = torch.mean((door_invalid == 0).to(torch.float32))
        success_conn = torch.mean((conn_invalid == 0).to(torch.float32))

        horizontal_door_match_counts = door_match_counts.horizontal[:-1, :-1].to(torch.float64)
        vertical_door_match_counts = door_match_counts.vertical[:-1, :-1].to(torch.float64)
        left_door_match_p = (
            horizontal_door_match_counts / torch.sum(horizontal_door_match_counts, dim=1, keepdim=True)
        )
        up_door_match_p = (
            vertical_door_match_counts / torch.sum(vertical_door_match_counts, dim=1, keepdim=True)
        )
        left_topk = torch.topk(left_door_match_p.flatten(), k=3).values
        up_topk = torch.topk(up_door_match_p.flatten(), k=3).values
        door_match_ss = (
            compute_door_match_count_ss(horizontal_door_match_counts, dim=1)
            + compute_door_match_count_ss(horizontal_door_match_counts, dim=0)
            + compute_door_match_count_ss(vertical_door_match_counts, dim=1)
            + compute_door_match_count_ss(vertical_door_match_counts, dim=0)
        )
        with torch.no_grad():
            generate_config = create_generate_config(
                step_config,
                self.episode_length,
                1,
                self.device,
            )
            balance_preds = self.balance_model(torch.log(generate_config.temperature))
            balance_door_match_ss = compute_balance_door_match_ss(balance_preds)
        loss_denominator = loss.total + 1e-15
        door_loss_pct = 100.0 * loss.door_contribution / loss_denominator
        connection_loss_pct = 100.0 * loss.connection_contribution / loss_denominator
        main_balance_loss_pct = 100.0 * loss.balance_contribution / loss_denominator
        proposal_loss_pct = 100.0 * loss.proposal_contribution / loss_denominator

        metrics = {
            "loss": loss.total,
            "door_loss": loss.door,
            "door_loss_pct": door_loss_pct,
            "connection_loss": loss.connection,
            "connection_loss_pct": connection_loss_pct,
            "main_balance_loss": loss.balance,
            "main_balance_loss_pct": main_balance_loss_pct,
            "proposal_loss": loss.proposal,
            "proposal_loss_pct": proposal_loss_pct,
            "candidate_target_entropy": candidate_diagnostics.target_entropy,
            "candidate_uniform_kl": candidate_diagnostics.uniform_kl,
            "candidate_selected_probability": candidate_diagnostics.selected_probability,
            "balance_loss": balance_loss,
            "success_rate": success_rate,
            "success_door": success_door,
            "success_conn": success_conn,
            "avg_invalid": avg_invalid,
            "avg_door": avg_door,
            "avg_conn": avg_conn,
            "min_invalid": min_invalid,
            "min_door": min_door,
            "min_conn": min_conn,
            "num_episodes": self.num_episodes,
            "lr": step_config.optimizer.lr,
            "temperature": step_config.generation.temperature,
            "recommended_candidates": step_config.generation.recommended_candidates,
            "exploration_candidates": step_config.generation.exploration_candidates,
            "proposal_temperature": step_config.generation.proposal_temperature,
            "reward_door": step_config.generation.reward_door,
            "reward_connection": step_config.generation.reward_connection,
            "reward_balance": step_config.generation.reward_balance,
            "door_match_left_top1": left_topk[0],
            "door_match_left_top2": left_topk[1],
            "door_match_left_top3": left_topk[2],
            "door_match_up_top1": up_topk[0],
            "door_match_up_top2": up_topk[1],
            "door_match_up_top3": up_topk[2],
            "door_match_ss": door_match_ss,
            "balance_door_match_ss": balance_door_match_ss,
        }
        for name, value in metrics.items():
            self.aim_run.track(value, name=name, step=round_idx)

        def scalar(value):
            return value.item() if isinstance(value, torch.Tensor) else value

        schedule_progress = min(self.num_episodes / self.config.knot_episodes[-1], 1.0)
        logging.info(
            "round %s, loss %.4f (door %.4f %.1f%%, conn %.4f %.1f%%, "
            "main_bal %.4f %.1f%%, prop %.4f %.1f%%), "
            "succ %.4f, total %.2f (min %s), door %.2f (min %s), "
            "conn %.2f (min %s), ss %.4f, ent %.4f, u_kl %.4f, "
            "p %.4f, "
            "cand %d, frac %.4f",
            round_idx,
            loss.total,
            loss.door,
            door_loss_pct,
            loss.connection,
            connection_loss_pct,
            loss.balance,
            main_balance_loss_pct,
            loss.proposal,
            proposal_loss_pct,
            scalar(success_rate),
            scalar(avg_invalid),
            scalar(min_invalid),
            scalar(avg_door),
            scalar(min_door),
            scalar(avg_conn),
            scalar(min_conn),
            scalar(door_match_ss),
            scalar(candidate_diagnostics.target_entropy),
            scalar(candidate_diagnostics.uniform_kl),
            scalar(candidate_diagnostics.selected_probability),
            step_config.generation.recommended_candidates + step_config.generation.exploration_candidates,
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
        rows = [
            (name, count, nanos)
            for name, count, nanos in report
            if count > 0 or nanos > 0
        ]
        if not rows:
            logging.info("round %s Rust profile: no samples recorded", round_idx)
            return

        for section_name, prefix in [
            ("Python generation spans", "python."),
            ("worker commands", "worker."),
            ("environment step spans", "env."),
        ]:
            section_rows = [row for row in rows if row[0].startswith(prefix)]
            if not section_rows:
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
                    gen_outcomes,
                    door_match_counts,
                    proposal_data,
                    generation_profile,
                ) = self.generate_round()
                candidate_diagnostics = compute_candidate_diagnostics(proposal_data)
                self.num_episodes += self.episodes_per_round
                step_config = instantiate_scheduleable_config(self.config, self.num_episodes)
                avg_loss, avg_balance_loss = self.train_round(
                    episode_data,
                    gen_outcomes,
                    proposal_data,
                    step_config,
                )

                self.experience.store(episode_data)

                self.log_outcomes(
                    gen_outcomes,
                    door_match_counts,
                    avg_loss,
                    candidate_diagnostics,
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
    namespace = parser.parse_args()
    return Args(
        config=namespace.config,
        verify_outcome_consistency=namespace.verify_outcome_consistency,
        device=namespace.device,
        load_checkpoint=namespace.load_checkpoint,
        profile=namespace.profile,
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
                or any(generation_device.type != "cuda" for generation_device in generation_devices)
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
            raise ValueError("--load-checkpoint must point to a file in a run's checkpoints directory")
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
            frontier_neighbor_algorithm=config.generation.frontier_neighbor_algorithm,
            frontier_neighbor_count=config.generation.frontier_neighbor_count,
            frontier_window_size=config.generation.frontier_window_size,
            num_threads=train_group_threads,
        )
        for _ in range(config.train.pipeline_groups)
    ]


def create_models(config: Config, rooms: list[dict], engine: Engine, device: torch.device, generation_devices):
    main_model = FrontierModel(**frontier_model_kwargs(config, rooms, engine)).to(device)
    num_params = sum(p.numel() for p in main_model.parameters())
    logging.info(f"Main model parameters: {num_params}")    

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
            initargs=(config_json, rooms_json, str(generation_device), device_index, profile),
        )
        for device_index, generation_device in enumerate(generation_devices)
    ]


def build_session(args: Args) -> TrainingSession:
    config = Config.model_validate_json(args.config.read_text())
    validate_config(config)
    map_gen.set_profile_enabled(args.profile)
    round_episode_count = episodes_per_round(config)
    run_path = setup_logging(config, args)
    rooms = json.loads(config.room_set.read_text())
    device, generation_devices = select_devices(args, config)

    train_precision = "bfloat16 autocast" if device.type == "cuda" and config.model.autocast else "float32"
    generation_precision = (
        "bfloat16 autocast" if device.type == "cuda" and config.model.generation_autocast else "float32"
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
    )
    initial_config = instantiate_scheduleable_config(config, 0)
    main_optimizer = torch.optim.Adam(
        main_model.parameters(),
        lr=initial_config.optimizer.lr,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
    )
    balance_optimizer = torch.optim.Adam(
        balance_model.parameters(),
        lr=initial_config.balance_optimizer.lr,
        betas=(config.balance_optimizer.beta1, config.balance_optimizer.beta2),
    )
    aim_run = Run(experiment=config.experiment_name, system_tracking_interval=None)
    aim_run["config"] = json.loads(config.model_dump_json())

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
        aim_run=aim_run,
        loss_config=LossConfig(
            door_weight=config.train.door_weight,
            connection_weight=config.train.connection_weight,
            balance_weight=config.train.balance_weight,
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
        session.load_checkpoint(args.load_checkpoint)
    return session


def main() -> None:
    args = parse_args()
    session = build_session(args)
    signal.signal(signal.SIGINT, lambda _signum, _frame: session.request_stop())
    signal.signal(signal.SIGTERM, lambda _signum, _frame: session.request_stop())
    session.run()


if __name__ == "__main__":
    main()
