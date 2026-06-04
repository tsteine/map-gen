import argparse
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
from typing import Literal

import torch
from aim import Run

from env import Actions, Engine, GenerateConfig, Outcomes, StateFeatures
from experience import ExperienceStorage
from generate import Prefetcher, generate_cohorts
from loss import LossConfig, compute_loss
from model import CausalTransformerModel, FrontierStateModel
from profile_stats import ProfileStats
from train_config import Config, episodes_per_round, instantiate_scheduleable_config, validate_config


@dataclass
class Args:
    config: Path
    verify_outcome_consistency: bool
    profile: bool
    device: str


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
    prefix_count: int | None
    state_feature_batches: list[StateFeatures] | None = None


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

    def request_stop(self) -> None:
        self.stop_requested = True
        logging.info("Stop signal received; training will stop after the current round finishes.")

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
            state_candidate_chunk=step_config.generation.state_candidate_chunk,
            state_environment_chunk=step_config.generation.state_environment_chunk,
            state_autocast=step_config.model.generation_autocast,
            training_autocast=step_config.model.autocast,
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
            if getattr(self.main_model, "uses_state_features", False):
                return self.prepare_state_feature_batch(task.kind, train_actions, train_outcomes, env)
            return PreparedTrainBatch(task.kind, train_actions, train_outcomes, None, None)

        with self.profiler.timer("round.replay_prepare"):
            replay_actions = self.experience.sample(
                self.config.train.batch_size,
                self.config.train.episodes_per_file,
                self.config.train.hist_c,
            )
            env.replay(replay_actions)
            replay_actions = replay_actions.to(self.device)
            replay_outcomes = env.get_outcomes(self.device)
        if getattr(self.main_model, "uses_state_features", False):
            return self.prepare_state_feature_batch(task.kind, replay_actions, replay_outcomes, env)
        return PreparedTrainBatch(task.kind, replay_actions, replay_outcomes, None, None)

    def train_batch_backward(
        self,
        prepared_batch: PreparedTrainBatch,
        gen_config: GenerateConfig,
        loss_scale: float,
    ) -> float:
        train_actions = prepared_batch.actions
        train_outcomes = prepared_batch.outcomes
        if getattr(self.main_model, "uses_state_features", False):
            loss = self.train_state_feature_batch_backward(prepared_batch, loss_scale)
        else:
            with self.profiler.cuda_timer("train.gpu_forward", self.device):
                preds = self.main_model(train_actions, gen_config)
                repeated_outcomes = Outcomes(
                    door_invalid=train_outcomes.door_invalid.unsqueeze(1).repeat(1, self.episode_length, 1),
                    connection_invalid=train_outcomes.connection_invalid.unsqueeze(1).repeat(1, self.episode_length, 1),
                )
                mask = (train_actions.room_idx < self.num_rooms).unsqueeze(2)
                loss = compute_loss(preds, repeated_outcomes, mask, self.loss_config)

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss before backward: {loss.item()}")

        if not getattr(self.main_model, "uses_state_features", False):
            with self.profiler.cuda_timer("train.gpu_backward", self.device):
                (loss * loss_scale).backward()
        return loss.item()

    def train_state_feature_batch_backward(
        self,
        prepared_batch: PreparedTrainBatch,
        loss_scale: float,
    ) -> torch.Tensor:
        if prepared_batch.state_feature_batches is None or prepared_batch.prefix_count is None:
            raise RuntimeError("state-feature training batch was not prepared")
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

    def generate_round(self) -> tuple[Actions, Outcomes]:
        action_iterations = []
        outcome_iterations = []
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

                for iteration_actions, iteration_outcomes in shard_results:
                    action_iterations.append(iteration_actions.to(self.device))
                    outcome_iterations.append(iteration_outcomes.to(self.device))

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
        )

    def train_round(self, actions: Actions, gen_outcomes: Outcomes, step_config: Config) -> float:
        train_gen_config = self.get_gen_config(step_config, self.config.train.batch_size, self.device)
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
                        train_gen_config,
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
        }
        for name, value in metrics.items():
            self.aim_run.track(value, name=name, step=round_idx)

        def scalar(value):
            return value.item() if isinstance(value, torch.Tensor) else value

        schedule_progress = min(self.num_episodes / self.config.knot_episodes[-1], 1.0)
        logging.info(
            "round %s, loss %.4f, succ %.4f, total %.2f (min %s), door %.2f (min %s), "
            "conn %.2f (min %s), schedule_progress %.4f",
            round_idx,
            loss,
            scalar(success_rate),
            scalar(avg_invalid),
            scalar(min_invalid),
            scalar(avg_door),
            scalar(min_door),
            scalar(avg_conn),
            scalar(min_conn),
            schedule_progress,
        )

    def run(self) -> None:
        try:
            total_episodes = self.config.knot_episodes[-1]
            for round_idx in range(total_episodes // self.episodes_per_round):
                self.profiler.reset()
                round_start = time.perf_counter()

                actions, gen_outcomes = self.generate_round()
                self.num_episodes += self.episodes_per_round
                step_config = instantiate_scheduleable_config(self.config, self.num_episodes)
                avg_loss = self.train_round(actions, gen_outcomes, step_config)

                with self.profiler.timer("round.store"):
                    self.experience.store(actions)

                self.log_outcomes(gen_outcomes, avg_loss, round_idx, step_config)
                self.profiler.add("round.total", time.perf_counter() - round_start)
                if self.profiler.enabled:
                    for name, value in self.profiler.metrics().items():
                        self.aim_run.track(value, name=name, step=round_idx)
                    logging.info("profile round %s:\n%s", round_idx, self.profiler.format())

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
    namespace = parser.parse_args()
    return Args(
        config=namespace.config,
        verify_outcome_consistency=namespace.verify_outcome_consistency,
        profile=namespace.profile,
        device=namespace.device,
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
    model_class = {
        "causal_transformer": CausalTransformerModel,
        "frontier_state": FrontierStateModel,
    }.get(config.model.type)
    if model_class is None:
        raise ValueError(f"unknown model.type: {config.model.type}")

    main_model = model_class(
        num_rooms=len(rooms),
        output_metadata=output_metadata,
        map_x=config.map_size[0],
        map_y=config.map_size[1],
        embedding_width=config.model.embedding_width,
        key_width=config.model.key_width,
        value_width=config.model.value_width,
        attn_heads=config.model.attn_heads,
        head_groups=config.model.head_groups,
        hidden_width=config.model.hidden_width,
        num_layers=config.model.num_layers,
        frontier_neighbor_count=config.generation.frontier_neighbor_count,
        frontier_window_size=config.generation.frontier_window_size,
        state_features=config.state_features.model_dump(),
    ).to(device)

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
    if config.model.type == "frontier_state" and config.model.compile:
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

    engine = Engine(rooms, config.state_features.model_dump())
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

    return TrainingSession(
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
            config.model.type == "frontier_state"
            and config.model.compile
            and len(generation_devices) > 1
        ),
    )


def main() -> None:
    args = parse_args()
    session = build_session(args)
    signal.signal(signal.SIGINT, lambda _signum, _frame: session.request_stop())
    signal.signal(signal.SIGTERM, lambda _signum, _frame: session.request_stop())
    session.run()


if __name__ == "__main__":
    main()
