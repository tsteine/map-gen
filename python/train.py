import argparse
from collections import deque
import copy
import json
import logging
import math
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import safetensors.torch
import torch
from aim import Run
from safetensors import safe_open

from env import Actions, DoorMatchCounts, Engine, GenerateConfig, Outcomes, StateFeatures
from experience import ExperienceStorage
from generate import generate_cohorts
from loss import LossConfig, compute_loss
from model import FrontierStateModel
from profile_stats import ProfileStats
from train_config import Config, episodes_per_round, instantiate_scheduleable_config, validate_config


@dataclass
class Args:
    config: Path
    verify_outcome_consistency: bool
    profile: bool
    device: str
    load_checkpoint: Path | None


@dataclass
class TrainBatchTask:
    kind: Literal["fresh", "replay"]
    start: int | None
    env_index: int


@dataclass
class PreparedTrainBatch:
    kind: Literal["fresh", "replay"]
    actions: Actions
    outcomes: Outcomes
    prefix_count: int
    state_feature_batches: list[StateFeatures]


class Prefetcher:
    def __init__(self, max_workers=1):
        if max_workers <= 0:
            raise ValueError("max_workers must be greater than zero")
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def close(self):
        self.executor.shutdown()

    def map(self, items, prepare, profiler=None, wait_name=None):
        items = iter(items)
        pending = deque()

        for _ in range(self.max_workers):
            if not submit_prefetch_item(self.executor, pending, items, prepare):
                break

        while pending:
            future = pending.popleft()
            if profiler is None or wait_name is None:
                result = future.result()
            else:
                with profiler.timer(wait_name):
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
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    state_dict = optimizer.state_dict()
    tensors = {}
    scalar_state = {}
    for param_id, param_state in state_dict["state"].items():
        param_scalar_state = {}
        for state_name, value in param_state.items():
            if torch.is_tensor(value):
                tensors[f"optimizer.state.{param_id}.{state_name}"] = as_checkpoint_tensor(value)
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
) -> None:
    state: dict[int, dict[str, Any]] = {}
    prefix = "optimizer.state."
    for key, value in tensors.items():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix):]
        param_id_text, state_name = suffix.split(".", 1)
        state.setdefault(int(param_id_text), {})[state_name] = value
    for param_id_text, param_scalar_state in scalar_state.items():
        state.setdefault(int(param_id_text), {}).update(param_scalar_state)
    optimizer.load_state_dict({
        "state": state,
        "param_groups": param_groups,
    })


@dataclass
class TrainingSession:
    args: Args
    config: Config
    profiler: ProfileStats
    run_path: str
    rooms: list[dict]
    device: torch.device
    generation_devices: list[torch.device]
    engine: Engine
    gen_envs: list[list]
    train_batch_envs: list
    main_model: torch.nn.Module
    ema_model: torch.nn.Module
    generation_models: list[torch.nn.Module]
    main_optimizer: torch.optim.Optimizer
    aim_run: Run
    loss_config: LossConfig
    experience: ExperienceStorage
    train_batch_prefetcher: Prefetcher
    generation_executor: ThreadPoolExecutor
    generation_models_warmed_up: bool
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
    def train_state_pipeline_cohorts(self) -> int:
        return self.config.train.state_pipeline_cohorts

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
        optimizer_tensors, optimizer_param_groups, optimizer_scalar_state = optimizer_checkpoint_state(
            self.main_optimizer
        )
        tensors.update(optimizer_tensors)
        metadata = {
            "format": "map-gen-training-session-checkpoint-v1",
            "num_episodes": str(self.num_episodes),
            "experience_num_files": str(self.experience.num_files),
            "optimizer_param_groups": json.dumps(optimizer_param_groups),
            "optimizer_scalar_state": json.dumps(optimizer_scalar_state),
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
        load_optimizer_checkpoint_state(
            self.main_optimizer,
            tensors,
            json.loads(metadata["optimizer_param_groups"]),
            json.loads(metadata["optimizer_scalar_state"]),
        )
        self.num_episodes = int(metadata["num_episodes"])
        if self.num_episodes % self.episodes_per_round != 0:
            raise ValueError(
                f"checkpoint num_episodes={self.num_episodes} is not divisible by episodes_per_round="
                f"{self.episodes_per_round}"
            )
        self.experience.num_files = int(metadata["experience_num_files"])
        self.sync_generation_models()
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

    def sync_generation_models(self) -> None:
        with torch.no_grad():
            for generation_model in self.generation_models[1:]:
                for generation_param, ema_param in zip(
                    generation_model.parameters(), self.ema_model.parameters()
                ):
                    generation_param.copy_(ema_param)

    def get_gen_config(
        self,
        step_config: Config,
        num_environments: int,
        generation_device: torch.device,
    ) -> GenerateConfig:
        return GenerateConfig(
            episode_length=self.episode_length,
            max_candidates=step_config.generation.action_candidates,
            temperature=torch.full(
                [num_environments],
                step_config.generation.temperature,
                dtype=torch.float32,
                device=generation_device,
            ),
            lookahead_outcomes=step_config.generation.lookahead_outcomes,
            state_autocast=step_config.model.generation_autocast,
        )

    def select_batch(self, actions: Actions, outcomes: Outcomes, start: int) -> tuple[Actions, Outcomes]:
        end = start + self.config.train.batch_size
        return (
            Actions(
                room_idx=actions.room_idx[start:end],
                room_x=actions.room_x[start:end],
                room_y=actions.room_y[start:end],
            ),
            Outcomes(
                door_invalid=outcomes.door_invalid[start:end],
                connection_invalid=outcomes.connection_invalid[start:end],
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
            tasks.append(TrainBatchTask("fresh", start, task_idx % self.train_state_pipeline_cohorts))
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
                tasks.append(TrainBatchTask("replay", None, task_idx % self.train_state_pipeline_cohorts))
                task_idx += 1
        return tasks

    def prepare_state_feature_batches(self, train_actions: Actions, env) -> tuple[int, list[StateFeatures]]:
        with self.profiler.timer("train.cpu_setup"):
            offset = torch.randint(0, self.config.train.sample_period, [1]).item()
            train_actions_cpu = train_actions.to(torch.device("cpu"))
            env.clear()
            state_feature_batches = []
        with self.profiler.timer("train.cpu_prefix_prepare"):
            for step in range(self.episode_length):
                env.step(Actions(
                    train_actions_cpu.room_idx[:, step],
                    train_actions_cpu.room_x[:, step],
                    train_actions_cpu.room_y[:, step],
                ))
                if step % self.config.train.sample_period == offset:
                    state_feature_batches.append(
                        env.get_state_features(
                            torch.device("cpu"),
                            0,
                            train_actions.room_idx.shape[0],
                        )
                    )
        return len(state_feature_batches), state_feature_batches

    def prepare_state_feature_batch(
        self,
        kind: Literal["fresh", "replay"],
        train_actions: Actions,
        train_outcomes: Outcomes,
        env,
    ) -> PreparedTrainBatch:
        prefix_count, state_feature_batches = self.prepare_state_feature_batches(
            train_actions,
            env,
        )
        return PreparedTrainBatch(
            kind,
            train_actions,
            train_outcomes,
            prefix_count=prefix_count,
            state_feature_batches=state_feature_batches,
        )

    def prepare_train_batch_task(
        self,
        task: TrainBatchTask,
        fresh_actions: Actions,
        fresh_outcomes: Outcomes,
    ) -> PreparedTrainBatch:
        env = self.train_batch_envs[task.env_index]
        if task.kind == "fresh":
            assert task.start is not None
            train_actions, train_outcomes = self.select_batch(fresh_actions, fresh_outcomes, task.start)
            return self.prepare_state_feature_batch(task.kind, train_actions, train_outcomes, env)

        with self.profiler.timer("round.replay_prepare"):
            replay_actions = self.experience.sample(
                self.config.train.batch_size,
                self.config.train.episodes_per_file,
                self.config.train.hist_c,
            )
            env.replay(replay_actions)
            replay_actions = replay_actions.to(self.device)
            replay_outcomes = env.get_outcomes(self.device)
        return self.prepare_state_feature_batch(task.kind, replay_actions, replay_outcomes, env)

    def train_batch_backward(
        self,
        prepared_batch: PreparedTrainBatch,
        loss_scale: float,
    ) -> float:
        loss = self.train_state_feature_batch_backward(prepared_batch, loss_scale)

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss before backward: {loss.item()}")

        return loss.item()

    def train_state_feature_batch_backward(
        self,
        prepared_batch: PreparedTrainBatch,
        loss_scale: float,
    ) -> torch.Tensor:
        if prepared_batch.prefix_count == 0:
            raise RuntimeError("state-feature training batch has no sampled prefixes")

        train_outcomes = prepared_batch.outcomes
        repeated_outcomes = Outcomes(
            door_invalid=train_outcomes.door_invalid.unsqueeze(1),
            connection_invalid=train_outcomes.connection_invalid.unsqueeze(1),
        )
        mask = torch.ones(
            [prepared_batch.actions.room_idx.shape[0], 1, 1],
            dtype=torch.bool,
            device=self.device,
        )
        total_loss = 0.0
        prefix_weight = 1.0 / prepared_batch.prefix_count

        for state_features in prepared_batch.state_feature_batches:
            with self.profiler.timer("train.cpu_transfer_submit"):
                state_features = state_features.to(self.device)
            with self.profiler.cuda_timer("train.gpu_forward_backward", self.device):
                with torch.amp.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda" and self.config.model.autocast,
                ):
                    preds = self.main_model(state_features)
                prefix_loss = compute_loss(preds, repeated_outcomes, mask, self.loss_config)
                (prefix_loss * prefix_weight * loss_scale).backward()
            total_loss += prefix_loss.item() * prefix_weight
        return torch.tensor(total_loss, device=self.device)

    def train_optimizer_step(self) -> None:
        with self.profiler.cuda_timer("train.gpu_optimizer", self.device):
            grad_norm = torch.nn.utils.clip_grad_norm_(self.main_model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"non-finite gradient norm: {grad_norm.item()}")
            self.main_optimizer.step()
            self.update_ema_model()

    def generate_round(self) -> tuple[Actions, Outcomes, DoorMatchCounts]:
        action_iterations = []
        outcome_iterations = []
        door_match_count_iterations = []
        with self.profiler.timer("round.generate"):
            self.sync_generation_models()
            for iteration in range(self.config.generation.num_iterations):
                generation_config = instantiate_scheduleable_config(
                    self.config,
                    self.num_episodes + iteration * self.config.generation.num_environments,
                )
                shard_args = []
                for device_envs, generation_model, generation_device in zip(
                    self.gen_envs,
                    self.generation_models,
                    self.generation_devices,
                ):
                    gen_configs = [
                        self.get_gen_config(generation_config, gen_env.num_envs, generation_device)
                        for gen_env in device_envs
                    ]
                    shard_args.append((device_envs, generation_model, gen_configs, generation_device))

                if self.generation_models_warmed_up:
                    shard_results = [
                        self.generation_executor.submit(
                            generate_cohorts,
                            *args,
                            verify_outcome_consistency=self.args.verify_outcome_consistency,
                            profiler=self.profiler,
                        )
                        for args in shard_args
                    ]
                    shard_results = [future.result() for future in shard_results]
                else:
                    logging.info(
                        "Warming up compiled generation models serially before concurrent generation."
                    )
                    shard_results = [
                        generate_cohorts(
                            *args,
                            verify_outcome_consistency=self.args.verify_outcome_consistency,
                            profiler=self.profiler,
                        )
                        for args in shard_args
                    ]
                    self.generation_models_warmed_up = True

                for (
                    iteration_actions,
                    iteration_outcomes,
                    iteration_door_match_counts,
                ) in shard_results:
                    action_iterations.append(iteration_actions.to(self.device))
                    outcome_iterations.append(iteration_outcomes.to(self.device))
                    door_match_count_iterations.append(iteration_door_match_counts.to(self.device))

        return (
            Actions(
                room_idx=torch.cat([actions.room_idx for actions in action_iterations]),
                room_x=torch.cat([actions.room_x for actions in action_iterations]),
                room_y=torch.cat([actions.room_y for actions in action_iterations]),
            ),
            Outcomes(
                door_invalid=torch.cat([outcomes.door_invalid for outcomes in outcome_iterations]),
                connection_invalid=torch.cat(
                    [outcomes.connection_invalid for outcomes in outcome_iterations]
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
        )

    def train_round(self, actions: Actions, gen_outcomes: Outcomes, step_config: Config) -> float:
        self.main_optimizer.param_groups[0]["lr"] = step_config.optimizer.lr

        total_loss = 0.0
        train_batch_count = 0

        def prepare_train_task(task: TrainBatchTask) -> PreparedTrainBatch:
            return self.prepare_train_batch_task(task, actions, gen_outcomes)

        def train_prepared_batch_group(prepared_batches: list[PreparedTrainBatch]) -> tuple[float, int]:
            self.main_model.zero_grad()
            loss_scale = 1.0 / len(prepared_batches)
            group_loss = 0.0
            for prepared_batch in prepared_batches:
                timer_name = (
                    "round.train_fresh"
                    if prepared_batch.kind == "fresh"
                    else "round.train_replay"
                )
                with self.profiler.timer(timer_name):
                    group_loss += self.train_batch_backward(
                        prepared_batch,
                        loss_scale,
                    )
            self.train_optimizer_step()
            return group_loss, len(prepared_batches)

        prepared_batch_group = []
        for prepared_batch in self.train_batch_prefetcher.map(
            self.iter_train_batch_tasks(),
            prepare_train_task,
            self.profiler,
            "round.train_batch_wait",
        ):
            prepared_batch_group.append(prepared_batch)
            if len(prepared_batch_group) == self.config.train.gradient_accumulation_steps:
                group_loss, group_count = train_prepared_batch_group(prepared_batch_group)
                total_loss += group_loss
                train_batch_count += group_count
                prepared_batch_group = []
        if prepared_batch_group:
            group_loss, group_count = train_prepared_batch_group(prepared_batch_group)
            total_loss += group_loss
            train_batch_count += group_count

        return total_loss / train_batch_count if train_batch_count > 0 else 0.0

    def log_outcomes(
        self,
        outcomes: Outcomes,
        door_match_counts: DoorMatchCounts,
        loss: float,
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
        horizontal_door_match_proportions = (
            horizontal_door_match_counts / torch.sum(horizontal_door_match_counts, dim=1, keepdim=True)
        )
        vertical_door_match_proportions = (
            vertical_door_match_counts / torch.sum(vertical_door_match_counts, dim=1, keepdim=True)
        )
        horizontal_topk = torch.topk(horizontal_door_match_proportions.flatten(), k=3).values
        vertical_topk = torch.topk(vertical_door_match_proportions.flatten(), k=3).values
        door_match_sum_squares = torch.sum(horizontal_door_match_proportions.square()) + torch.sum(
            vertical_door_match_proportions.square()
        )

        metrics = {
            "loss": loss,
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
            "door_match_horizontal_top1": horizontal_topk[0],
            "door_match_horizontal_top2": horizontal_topk[1],
            "door_match_horizontal_top3": horizontal_topk[2],
            "door_match_vertical_top1": vertical_topk[0],
            "door_match_vertical_top2": vertical_topk[1],
            "door_match_vertical_top3": vertical_topk[2],
            "door_match_sum_squares": door_match_sum_squares,
        }
        for name, value in metrics.items():
            self.aim_run.track(value, name=name, step=round_idx)

        def scalar(value):
            return value.item() if isinstance(value, torch.Tensor) else value

        schedule_progress = min(self.num_episodes / self.config.knot_episodes[-1], 1.0)
        logging.info(
            "round %s, loss %.4f, succ %.4f, total %.2f (min %s), door %.2f (min %s), "
            "conn %.2f (min %s), door_match_ss %.4f, schedule_progress %.4f",
            round_idx,
            loss,
            scalar(success_rate),
            scalar(avg_invalid),
            scalar(min_invalid),
            scalar(avg_door),
            scalar(min_door),
            scalar(avg_conn),
            scalar(min_conn),
            scalar(door_match_sum_squares),
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

    def run(self) -> None:
        try:
            total_episodes = self.config.knot_episodes[-1]
            start_round = self.num_episodes // self.episodes_per_round
            for round_idx in range(start_round, total_episodes // self.episodes_per_round):
                self.profiler.reset()
                round_start = time.perf_counter()

                actions, gen_outcomes, door_match_counts = self.generate_round()
                self.num_episodes += self.episodes_per_round
                step_config = instantiate_scheduleable_config(self.config, self.num_episodes)
                avg_loss = self.train_round(actions, gen_outcomes, step_config)

                with self.profiler.timer("round.store"):
                    self.experience.store(actions)

                self.log_outcomes(
                    gen_outcomes,
                    door_match_counts,
                    avg_loss,
                    round_idx,
                    step_config,
                )
                self.profiler.add("round.total", time.perf_counter() - round_start)
                if self.profiler.enabled:
                    for name, value in self.profiler.metrics().items():
                        self.aim_run.track(value, name=name, step=round_idx)
                    logging.info("profile round %s:\n%s", round_idx, self.profiler.format())

                completed_round = round_idx + 1
                if completed_round % self.config.checkpoint_period == 0:
                    self.save_checkpoint(self.checkpoint_path(completed_round))

                if self.stop_requested:
                    logging.info("Stopping training after completing round %s.", round_idx)
                    break
        finally:
            self.train_batch_prefetcher.close()
            self.generation_executor.shutdown()
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
        "--profile",
        action="store_true",
        help="log synchronized per-round timing breakdowns (changes CUDA throughput)",
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
    namespace = parser.parse_args()
    return Args(
        config=namespace.config,
        verify_outcome_consistency=namespace.verify_outcome_consistency,
        profile=namespace.profile,
        device=namespace.device,
        load_checkpoint=namespace.load_checkpoint,
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


def setup_logging(config: Config, args: Args) -> tuple[ProfileStats, str]:
    profiler = ProfileStats(args.profile)
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
    if profiler.enabled:
        logging.info("Profiling enabled. CUDA timings synchronize the device and change throughput.")
    if args.load_checkpoint is not None:
        logging.info("Loading checkpoint from %s", args.load_checkpoint)
    return profiler, run_path


def create_environment_groups(config: Config, engine: Engine, generation_devices: list[torch.device]):
    num_generation_cohorts = (
        config.generation.num_devices * config.generation.state_pipeline_cohorts
    )
    generation_cohort_environments = config.generation.num_environments // num_generation_cohorts
    generation_cohort_threads = (
        None
        if config.generation.num_threads is None
        else config.generation.num_threads // config.generation.state_pipeline_cohorts
    )
    train_state_cohort_threads = (
        None
        if config.generation.num_threads is None
        else config.generation.num_threads // config.train.state_pipeline_cohorts
    )
    logging.info(
        "Using %s state pipeline cohort(s) per generation device with %s environment(s) and %s Rust worker(s) per cohort.",
        config.generation.state_pipeline_cohorts,
        generation_cohort_environments,
        generation_cohort_threads if generation_cohort_threads is not None else "automatic",
    )
    logging.info(
        "Using %s training state pipeline cohort(s) with %s Rust worker(s) per cohort.",
        config.train.state_pipeline_cohorts,
        train_state_cohort_threads if train_state_cohort_threads is not None else "automatic",
    )
    gen_envs = [
        [
            engine.create_environment_group(
                config.map_size,
                generation_cohort_environments,
                seed=device_index * config.generation.state_pipeline_cohorts + cohort_index,
                frontier_neighbor_algorithm=config.generation.frontier_neighbor_algorithm,
                frontier_neighbor_count=config.generation.frontier_neighbor_count,
                frontier_window_size=config.generation.frontier_window_size,
                num_threads=generation_cohort_threads,
            )
            for cohort_index in range(config.generation.state_pipeline_cohorts)
        ]
        for device_index in range(len(generation_devices))
    ]
    train_batch_envs = [
        engine.create_environment_group(
            config.map_size,
            config.train.batch_size,
            frontier_neighbor_algorithm=config.generation.frontier_neighbor_algorithm,
            frontier_neighbor_count=config.generation.frontier_neighbor_count,
            frontier_window_size=config.generation.frontier_window_size,
            num_threads=train_state_cohort_threads,
        )
        for _ in range(config.train.state_pipeline_cohorts)
    ]
    return gen_envs, train_batch_envs


def create_models(config: Config, rooms: list[dict], engine: Engine, device: torch.device, generation_devices):
    output_metadata = engine.get_output_metadata()
    model_kwargs = {
        "num_rooms": len(rooms),
        "output_metadata": output_metadata,
        "map_x": config.map_size[0],
        "map_y": config.map_size[1],
        "embedding_width": config.model.embedding_width,
        "hidden_width": config.model.hidden_width,
        "num_layers": config.model.num_layers,
        "frontier_window_size": config.generation.frontier_window_size,
        "state_features": config.state_features,
    }

    main_model = FrontierStateModel(**model_kwargs).to(device)

    ema_model = copy.deepcopy(main_model).to(device)
    ema_model.requires_grad_(False)
    ema_model.eval()
    generation_models = [
        ema_model,
        *[
            copy.deepcopy(ema_model).to(generation_device)
            for generation_device in generation_devices[1:]
        ],
    ]
    if config.model.compile:
        main_model = torch.compile(main_model)
        generation_models = [torch.compile(model) for model in generation_models]
        ema_model = generation_models[0]

    return main_model, ema_model, generation_models


def build_session(args: Args) -> TrainingSession:
    config = Config.model_validate_json(args.config.read_text())
    validate_config(config)
    round_episode_count = episodes_per_round(config)
    profiler, run_path = setup_logging(config, args)
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

    engine = Engine(rooms, config.state_features)
    gen_envs, train_batch_envs = create_environment_groups(config, engine, generation_devices)
    main_model, ema_model, generation_models = create_models(
        config,
        rooms,
        engine,
        device,
        generation_devices,
    )
    initial_config = instantiate_scheduleable_config(config, 0)
    main_optimizer = torch.optim.Adam(
        main_model.parameters(),
        lr=initial_config.optimizer.lr,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
    )
    aim_run = Run(experiment=config.experiment_name, system_tracking_interval=None)
    aim_run["config"] = json.loads(config.model_dump_json())

    session = TrainingSession(
        args=args,
        config=config,
        profiler=profiler,
        run_path=run_path,
        rooms=rooms,
        device=device,
        generation_devices=generation_devices,
        engine=engine,
        gen_envs=gen_envs,
        train_batch_envs=train_batch_envs,
        main_model=main_model,
        ema_model=ema_model,
        generation_models=generation_models,
        main_optimizer=main_optimizer,
        aim_run=aim_run,
        loss_config=LossConfig(
            door_weight=config.train.door_weight,
            connection_weight=config.train.connection_weight,
        ),
        experience=ExperienceStorage(
            len(rooms),
            f"{run_path}/experience",
            round_episode_count,
        ),
        train_batch_prefetcher=Prefetcher(max_workers=config.train.state_pipeline_cohorts),
        generation_executor=ThreadPoolExecutor(max_workers=len(generation_devices)),
        generation_models_warmed_up=not (
            config.model.compile
            and len(generation_devices) > 1
        ),
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
