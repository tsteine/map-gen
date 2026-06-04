import json
import math
import copy
import torch
import logging
import signal
import time
from datetime import datetime
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal
import numpy as np 
import os

from aim import Run
from pydantic import BaseModel
from pathlib import Path

from env import Actions, Engine, GenerateConfig, Outcomes, StateFeatures
from model import CausalTransformerModel, FrontierStateModel
from loss import LossConfig, compute_loss
from generate import Prefetcher, generate_cohorts
from experience import ExperienceStorage
from profile_stats import ProfileStats


class Schedule(BaseModel):
    linear: list[float] | None = None
    log: list[float] | None = None
    

type ScheduleableFloat = float | Schedule

class ModelConfig(BaseModel):
    type: str
    compile: bool
    autocast: bool
    generation_autocast: bool
    embedding_width: int 
    key_width: int
    value_width: int
    attn_heads: int
    head_groups: int
    hidden_width: int 
    num_layers: int


class OptimizerConfig(BaseModel):
    lr: ScheduleableFloat
    beta1: float
    beta2: float
    

class GenerationConfig(BaseModel):
    num_environments: int  # number of maps to generate in parallel
    num_iterations: int  # number of sequential parallel batches to generate per training round
    num_devices: int  # number of GPUs to use for generation; training remains on the first device
    state_pipeline_cohorts: int  # independently stepped CPU cohorts scheduled on each GPU
    action_candidates: int  # number of candidates to score for each room placement step
    lookahead_outcomes: bool  # use post-candidate known outcomes when scoring candidates (greater CPU usage, better accuracy)
    temperature: ScheduleableFloat  # temperature (higher = candidates selected more randomly)
    state_candidate_chunk: int
    state_environment_chunk: int
    frontier_neighbor_algorithm: Literal["delaunay", "nearest", "nearest-exclusive"]
    frontier_neighbor_count: int
    frontier_window_size: int
    num_threads: int | None


class StateFeatureConfig(BaseModel):
    inventory: bool
    room_position: bool
    frontier_mask: bool
    frontier_position: bool
    frontier_orientation: bool
    frontier_kind: bool
    frontier_occupancy: bool
    frontier_neighbor: bool
    frontier_neighbor_position: bool
    frontier_neighbor_flags: bool
    connection_reachability: bool
    frontier_connection_reachability: bool


class TrainConfig(BaseModel):
    batch_size: int  # number of episodes per training batch
    fresh_pass_factor: float  # number of passes over just-generated episodes
    replay_pass_factor: float  # average number of passes over past episodes
    sample_period: int  # number of steps between training within an episode (e.g. 8 = train on every 8th step)
    episodes_per_file: int  # number of episodes to read from each file (higher values = lower disk I/O, lower values = more diverse sampling)
    hist_c: float  # extent to which replay sampling biases towards recent episodes (0.0 = no bias)
    door_weight: float  # amount of weight assigned to door outcomes in the loss function
    connection_weight: float  # amount of weight assigned to connection outcomes in the loss function
    ema_decay: float  # decay factor for exponential moving average model used during generation
    state_batch_chunk: int
    state_gpu_chunk: int  # maximum prepared state rows per GPU training batch
    state_pipeline_cohorts: int
    gradient_accumulation_steps: int


class Config(BaseModel):
    experiment_name: str  # string/identifier for the experiment (doesn't have to be unique)
    room_set: Path  # path to JSON file defining the set of rooms to use for map generation
    map_size: tuple[int, int]  # dimensions of the map grid within which rooms are placed
    knot_episodes: list[int]  # episodes defining knots for linear splines, controlling learning rate, etc.
    model: ModelConfig
    optimizer: OptimizerConfig
    generation: GenerationConfig
    state_features: StateFeatureConfig
    train: TrainConfig


def instantiate_scheduleable_config(config: Config, num_episodes: int) -> Config:
    knot_episodes = config.knot_episodes

    def instantiate_model(model: BaseModel, path: str) -> BaseModel:
        updates = {}
        for field_name, field_info in model.__class__.model_fields.items():
            value = getattr(model, field_name)
            field_path = f"{path}.{field_name}"
            if field_info.annotation is ScheduleableFloat:
                updates[field_name] = instantiate_float(value, field_path)
            elif isinstance(value, BaseModel):
                updates[field_name] = instantiate_model(value, field_path)
        return model.model_copy(update=updates)

    def instantiate_float(value: ScheduleableFloat, path: str) -> float:
        if isinstance(value, Schedule):
            if value.linear is None and value.log is None:
                raise ValueError(f"{path} must have exactly one schedule value: 'linear' or 'log'")
            x = value.linear if value.linear is not None else value.log
            if len(x) != len(knot_episodes):
                raise ValueError(
                    f"{path} has {len(x)} schedule value(s), but knot_episodes has "
                    f"{len(knot_episodes)} knot(s)"
                )
            if value.linear is not None:
                return float(np.interp(num_episodes, knot_episodes, x))
            elif value.log is not None:
                return float(np.exp(np.interp(num_episodes, knot_episodes, np.log(x))))
        return float(value)

    return instantiate_model(config, "config")


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
args = parser.parse_args()
config = Config.parse_file(args.config)

verify_outcome_consistency = args.verify_outcome_consistency
profiler = ProfileStats(args.profile)
if not config.knot_episodes:
    raise ValueError("knot_episodes must contain at least one episode count")
total_episodes = config.knot_episodes[-1]
if total_episodes <= 0:
    raise ValueError("last knot_episodes value must be greater than zero")
if config.generation.num_iterations <= 0:
    raise ValueError("generation.num_iterations must be greater than zero")
if config.generation.num_devices <= 0:
    raise ValueError("generation.num_devices must be greater than zero")
if config.generation.state_pipeline_cohorts <= 0:
    raise ValueError("generation.state_pipeline_cohorts must be greater than zero")
if config.generation.num_devices > config.generation.num_environments:
    raise ValueError("generation.num_devices must not exceed generation.num_environments")
num_generation_cohorts = config.generation.num_devices * config.generation.state_pipeline_cohorts
if config.generation.num_environments % num_generation_cohorts != 0:
    raise ValueError(
        "generation.num_environments must be divisible by "
        "generation.num_devices * generation.state_pipeline_cohorts"
    )
if config.generation.state_candidate_chunk <= 0:
    raise ValueError("generation.state_candidate_chunk must be greater than zero")
if config.generation.state_environment_chunk <= 0:
    raise ValueError("generation.state_environment_chunk must be greater than zero")
if config.generation.frontier_neighbor_count <= 0:
    raise ValueError("generation.frontier_neighbor_count must be greater than zero")
if config.generation.frontier_window_size <= 0:
    raise ValueError("generation.frontier_window_size must be greater than zero")
if config.generation.num_threads is not None and config.generation.num_threads <= 0:
    raise ValueError("generation.num_threads must be greater than zero")
if (
    config.generation.num_threads is not None
    and config.generation.num_threads % config.generation.state_pipeline_cohorts != 0
):
    raise ValueError("generation.num_threads must be divisible by generation.state_pipeline_cohorts")
if config.model.type != "frontier_state" and config.generation.state_pipeline_cohorts != 1:
    raise ValueError("generation.state_pipeline_cohorts must be 1 unless model.type is frontier_state")
if config.train.sample_period <= 0:
    raise ValueError("train.sample_period must be greater than zero")
if config.train.state_batch_chunk <= 0:
    raise ValueError("train.state_batch_chunk must be greater than zero")
if config.train.state_gpu_chunk <= 0:
    raise ValueError("train.state_gpu_chunk must be greater than zero")
if config.train.state_pipeline_cohorts <= 0:
    raise ValueError("train.state_pipeline_cohorts must be greater than zero")
if config.train.gradient_accumulation_steps <= 0:
    raise ValueError("train.gradient_accumulation_steps must be greater than zero")
train_state_pipeline_cohorts = config.train.state_pipeline_cohorts
if (
    config.generation.num_threads is not None
    and config.generation.num_threads % train_state_pipeline_cohorts != 0
):
    raise ValueError("generation.num_threads must be divisible by train.state_pipeline_cohorts")
if (
    config.state_features.frontier_position
    or config.state_features.frontier_orientation
    or config.state_features.frontier_kind
    or config.state_features.frontier_occupancy
    or config.state_features.frontier_neighbor
    or config.state_features.frontier_connection_reachability
) and not config.state_features.frontier_mask:
    raise ValueError("frontier state features require state_features.frontier_mask")
if (
    config.state_features.frontier_neighbor_position
    or config.state_features.frontier_neighbor_flags
) and not config.state_features.frontier_neighbor:
    raise ValueError("frontier neighbor pair features require state_features.frontier_neighbor")


start_time = datetime.now()
run_path = f"runs/{start_time.isoformat()}-{config.experiment_name}/"
os.makedirs(run_path, exist_ok=True)
logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(f"{run_path}/train-{start_time.isoformat()}.log"),
                              logging.StreamHandler()])

logging.info("Config:\n{}".format(config.model_dump_json(indent=2)))
if verify_outcome_consistency:
    logging.info("Outcome consistency verification enabled.")
if profiler.enabled:
    logging.info("Profiling enabled. CUDA timings synchronize the device and change throughput.")

episodes_per_round = config.generation.num_iterations * config.generation.num_environments
if config.train.fresh_pass_factor != 0.0 and episodes_per_round % config.train.batch_size != 0:
    raise ValueError(
        "train.batch_size must evenly divide the number of episodes generated per round when "
        "train.fresh_pass_factor is non-zero"
    )

stop_requested = False


def handle_stop(signum, frame):
    global stop_requested
    stop_requested = True
    logging.info("Stop signal received; training will stop after the current round finishes.")


signal.signal(signal.SIGINT, handle_stop)
signal.signal(signal.SIGTERM, handle_stop)

rooms_str = open(config.room_set, "r").read()
rooms = json.loads(rooms_str)
num_rooms = len(rooms)
episode_length = num_rooms
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
    torch.set_float32_matmul_precision('high')
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
generation_cohort_environments = config.generation.num_environments // num_generation_cohorts
generation_cohort_threads = (
    None
    if config.generation.num_threads is None
    else config.generation.num_threads // config.generation.state_pipeline_cohorts
)
train_state_cohort_threads = (
    None
    if config.generation.num_threads is None
    else config.generation.num_threads // train_state_pipeline_cohorts
)
logging.info(
    "Using %s state pipeline cohort(s) per generation device with %s environment(s) and %s Rust worker(s) per cohort.",
    config.generation.state_pipeline_cohorts,
    generation_cohort_environments,
    generation_cohort_threads if generation_cohort_threads is not None else "automatic",
)
logging.info(
    "Using %s training state pipeline cohort(s) with %s Rust worker(s) per cohort.",
    train_state_pipeline_cohorts,
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
    for _ in range(train_state_pipeline_cohorts)
]
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


def update_ema_model():
    with torch.no_grad():
        for ema_param, main_param in zip(ema_model.parameters(), main_model.parameters()):
            ema_param.lerp_(main_param, 1.0 - config.train.ema_decay)


def sync_generation_models():
    with torch.no_grad():
        for generation_model in generation_models[1:]:
            for generation_param, ema_param in zip(
                generation_model.parameters(), ema_model.parameters()
            ):
                generation_param.copy_(ema_param)


# @dataclass
# class TrainSession:
#     model: torch.nn.Module
#     optimizer: torch.optim.Optimizer
#     num_episodes: int = 0

#     def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, num_episodes: int = 0):
#         self.model = model
#         self.optimizer = optimizer
#         self.num_episodes = num_episodes

# Log experiment using Aim
aim_run = Run(experiment=config.experiment_name, system_tracking_interval=None)
aim_run["config"] = json.loads(config.model_dump_json())

loss_config = LossConfig(
    door_weight=config.train.door_weight,
    connection_weight=config.train.connection_weight,
)


experience_path = f"{run_path}/experience"
experience = ExperienceStorage(num_rooms, experience_path, episodes_per_round)

num_episodes = 0
initial_config = instantiate_scheduleable_config(config, num_episodes)

main_optimizer = torch.optim.Adam(
    main_model.parameters(),
    lr=initial_config.optimizer.lr,
    betas=(config.optimizer.beta1, config.optimizer.beta2))

train_batch_prefetcher = Prefetcher(max_workers=train_state_pipeline_cohorts)
generation_executor = ThreadPoolExecutor(max_workers=len(generation_devices))
generation_models_warmed_up = not (
    config.model.type == "frontier_state"
    and config.model.compile
    and len(generation_devices) > 1
)


@dataclass
class TrainBatchTask:
    kind: Literal["fresh", "replay"]
    start: int | None
    env_index: int


@dataclass
class PreparedStateFeatureBatch:
    row_count: int
    ranges: list[tuple[int, int]]
    features: StateFeatures


@dataclass
class PreparedTrainBatch:
    kind: Literal["fresh", "replay"]
    actions: Actions
    outcomes: Outcomes
    prefix_count: int | None
    state_feature_batches: list[PreparedStateFeatureBatch] | None = None


def select_batch(actions, outcomes, start, batch_size):
    end = start + batch_size
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


def iter_fresh_batch_starts(num_items, pass_factor, batch_size):
    num_batches = int(math.ceil(num_items * pass_factor / batch_size))
    for batch_idx in range(num_batches):
        yield (batch_idx * batch_size) % num_items


def num_replay_batches(num_items, pass_factor, batch_size):
    return int(math.ceil(num_items * pass_factor / batch_size))


def iter_train_batch_tasks(num_items, fresh_pass_factor, replay_pass_factor, batch_size, has_replay):
    task_idx = 0
    for start in iter_fresh_batch_starts(num_items, fresh_pass_factor, batch_size):
        yield TrainBatchTask("fresh", start, task_idx % train_state_pipeline_cohorts)
        task_idx += 1
    if has_replay:
        for _ in range(num_replay_batches(num_items, replay_pass_factor, batch_size)):
            yield TrainBatchTask("replay", None, task_idx % train_state_pipeline_cohorts)
            task_idx += 1


def prepare_state_feature_training_chunks(train_actions, env):
    with profiler.timer("train.cpu_setup"):
        offset = torch.randint(0, config.train.sample_period, [1]).item()
        train_actions_cpu = train_actions.to(torch.device("cpu"))
        env.clear()
        feature_chunks = []
    with profiler.timer("train.cpu_prefix_prepare"):
        for step in range(0, episode_length):
            env.step(Actions(
                train_actions_cpu.room_idx[:, step],
                train_actions_cpu.room_x[:, step],
                train_actions_cpu.room_y[:, step],
            ))
            if step % config.train.sample_period == offset:
                for start in range(0, train_actions.room_idx.shape[0], config.train.state_batch_chunk):
                    end = min(start + config.train.state_batch_chunk, train_actions.room_idx.shape[0])
                feature_chunks.append((
                    start,
                    end,
                    env.get_state_features(
                        torch.device("cpu"),
                        start,
                        end - start,
                    ),
                ))
    return len(feature_chunks), feature_chunks


def prepare_state_feature_batch(kind, train_actions, train_outcomes, env):
    prefix_count, feature_chunks = prepare_state_feature_training_chunks(
        train_actions, env
    )
    state_feature_batches = [
        PreparedStateFeatureBatch(
            row_count=sum(end - start for start, end, _ in gpu_chunks),
            ranges=[
                (start, end)
                for start, end, _ in gpu_chunks
            ],
            features=cat_state_features([
                chunk_features
                for _, _, chunk_features in gpu_chunks
            ]),
        )
        for gpu_chunks in iter_state_feature_gpu_batches(feature_chunks)
    ]
    return PreparedTrainBatch(
        kind,
        train_actions,
        train_outcomes,
        prefix_count=prefix_count,
        state_feature_batches=state_feature_batches,
    )


def prepare_train_batch_task(task, fresh_actions, fresh_outcomes):
    env = train_batch_envs[task.env_index]
    if task.kind == "fresh":
        assert task.start is not None
        train_actions, train_outcomes = select_batch(
            fresh_actions, fresh_outcomes, task.start, config.train.batch_size
        )
        if getattr(main_model, "uses_state_features", False):
            return prepare_state_feature_batch(task.kind, train_actions, train_outcomes, env)
        return PreparedTrainBatch(task.kind, train_actions, train_outcomes, None, None)

    with profiler.timer("round.replay_prepare"):
        replay_actions = experience.sample(
            config.train.batch_size,
            config.train.episodes_per_file,
            config.train.hist_c,
        )
        env.replay(replay_actions)
        replay_actions = replay_actions.to(device)
        replay_outcomes = env.get_outcomes(device)
    if getattr(main_model, "uses_state_features", False):
        return prepare_state_feature_batch(task.kind, replay_actions, replay_outcomes, env)
    return PreparedTrainBatch(task.kind, replay_actions, replay_outcomes, None, None)


def cat_state_features(features: list[StateFeatures]) -> StateFeatures:
    def cat_feature(name):
        values = [getattr(feature, name) for feature in features]
        if all(value.shape[1:] == values[0].shape[1:] for value in values):
            return torch.cat(values, dim=0)

        max_shape = [
            max(value.shape[dim] for value in values)
            for dim in range(len(values[0].shape))
        ]
        fill_value = -1 if name == "frontier_neighbor" else 0
        padded_values = []
        for value in values:
            padded = value.new_full(
                (value.shape[0], *max_shape[1:]),
                fill_value,
            )
            slices = tuple(slice(0, size) for size in value.shape)
            padded[slices] = value
            padded_values.append(padded)
        return torch.cat(padded_values, dim=0)

    return StateFeatures(*(
        cat_feature(name)
        for name in vars(features[0])
    ))


def iter_state_feature_gpu_batches(feature_chunks):
    max_rows = config.train.state_gpu_chunk
    current_chunks = []
    current_rows = 0
    for chunk in feature_chunks:
        start, end, _ = chunk
        rows = end - start
        if current_chunks and current_rows + rows > max_rows:
            yield current_chunks
            current_chunks = []
            current_rows = 0
        current_chunks.append(chunk)
        current_rows += rows
    if current_chunks:
        yield current_chunks


def train_batch_backward(prepared_batch, gen_config, loss_scale):
    train_actions = prepared_batch.actions
    train_outcomes = prepared_batch.outcomes
    if getattr(main_model, "uses_state_features", False):
        if prepared_batch.state_feature_batches is None or prepared_batch.prefix_count is None:
            raise RuntimeError("state-feature training batch was not prepared")
        repeated_outcomes = Outcomes(
            door_invalid=train_outcomes.door_invalid.unsqueeze(1),
            connection_invalid=train_outcomes.connection_invalid.unsqueeze(1),
        )
        mask = torch.ones([train_actions.room_idx.shape[0], 1, 1], dtype=torch.bool, device=device)
        total_loss = 0.0

        for gpu_batch in prepared_batch.state_feature_batches:
            chunk_weight = (
                gpu_batch.row_count
                / train_actions.room_idx.shape[0]
                / prepared_batch.prefix_count
            )
            chunk_outcomes = Outcomes(
                door_invalid=torch.cat([
                    repeated_outcomes.door_invalid[start:end]
                    for start, end in gpu_batch.ranges
                ]),
                connection_invalid=torch.cat([
                    repeated_outcomes.connection_invalid[start:end]
                    for start, end in gpu_batch.ranges
                ]),
            )
            chunk_mask = torch.cat([
                mask[start:end]
                for start, end in gpu_batch.ranges
            ])
            with profiler.timer("train.cpu_transfer_submit"):
                chunk_features = gpu_batch.features.to(device)
            with profiler.cuda_timer("train.gpu_forward_backward", device):
                with torch.amp.autocast(
                    "cuda",
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda" and config.model.autocast,
                ):
                    chunk_preds = main_model(chunk_features)
                chunk_loss = compute_loss(
                    chunk_preds,
                    chunk_outcomes,
                    chunk_mask,
                    loss_config,
                )
                (chunk_loss * chunk_weight * loss_scale).backward()
            total_loss += chunk_loss.item() * chunk_weight
        loss = torch.tensor(total_loss, device=device)
    else:
        with profiler.cuda_timer("train.gpu_forward", device):
            preds = main_model(train_actions, gen_config)
            repeated_outcomes = Outcomes(
                door_invalid=train_outcomes.door_invalid.unsqueeze(1).repeat(1, episode_length, 1),
                connection_invalid=train_outcomes.connection_invalid.unsqueeze(1).repeat(1, episode_length, 1),
            )
            mask = (train_actions.room_idx < num_rooms).unsqueeze(2)  # exclude dummy actions
            loss = compute_loss(preds, repeated_outcomes, mask, loss_config)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss before backward: {loss.item()}")

    if not getattr(main_model, "uses_state_features", False):
        with profiler.cuda_timer("train.gpu_backward", device):
            (loss * loss_scale).backward()
    return loss.item()


def train_optimizer_step():
    with profiler.cuda_timer("train.gpu_optimizer", device):
        grad_norm = torch.nn.utils.clip_grad_norm_(main_model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient norm: {grad_norm.item()}")
        main_optimizer.step()
        update_ema_model()

try:
    def log_outcomes(outcomes, loss, round, step_config, num_episodes):
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
            "num_episodes": num_episodes,
            "lr": step_config.optimizer.lr,
            "temperature": step_config.generation.temperature,
        }
    
        for name, value in metrics.items():
            aim_run.track(value, name=name, step=round)

        schedule_progress = min(num_episodes / config.knot_episodes[-1], 1.0)
        logging.info(f"round {round}, loss {loss:.4f}, succ {success_rate:.4f}, total {avg_invalid:.2f} (min {min_invalid}), door {avg_door:.2f} (min {min_door}), conn {avg_conn:.2f} (min {min_conn}), schedule_progress {schedule_progress:.4f}")

    def get_gen_config(step_config, num_environments, generation_device):
        return GenerateConfig(
            episode_length=len(rooms),
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

        
    for round in range(total_episodes // episodes_per_round):
        profiler.reset()
        round_start = time.perf_counter()

        # Generate new maps:
        action_iterations = []
        outcome_iterations = []
        with profiler.timer("round.generate"):
            sync_generation_models()
            for iteration in range(config.generation.num_iterations):
                generation_config = instantiate_scheduleable_config(
                    config,
                    num_episodes + iteration * config.generation.num_environments,
                )
                shard_args = []
                for device_envs, generation_model, generation_device in zip(
                    gen_envs, generation_models, generation_devices
                ):
                    gen_configs = [
                        get_gen_config(generation_config, gen_env.num_envs, generation_device)
                        for gen_env in device_envs
                    ]
                    shard_args.append((
                        device_envs,
                        generation_model,
                        gen_configs,
                        generation_device,
                    ))
                if generation_models_warmed_up:
                    shard_results = [
                        generation_executor.submit(
                            generate_cohorts,
                            *args,
                            verify_outcome_consistency=verify_outcome_consistency,
                            profiler=profiler,
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
                            verify_outcome_consistency=verify_outcome_consistency,
                            profiler=profiler,
                        )
                        for args in shard_args
                    ]
                    generation_models_warmed_up = True
                for iteration_actions, iteration_outcomes in shard_results:
                    action_iterations.append(iteration_actions.to(device))
                    outcome_iterations.append(iteration_outcomes.to(device))
        actions = Actions(
            room_idx=torch.cat([actions.room_idx for actions in action_iterations]),
            room_x=torch.cat([actions.room_x for actions in action_iterations]),
            room_y=torch.cat([actions.room_y for actions in action_iterations]),
        )
        gen_outcomes = Outcomes(
            door_invalid=torch.cat(
                [outcomes.door_invalid for outcomes in outcome_iterations]
            ),
            connection_invalid=torch.cat(
                [outcomes.connection_invalid for outcomes in outcome_iterations]
            )
        )
        num_episodes += episodes_per_round
        step_config = instantiate_scheduleable_config(config, num_episodes)
        train_gen_config = get_gen_config(step_config, config.train.batch_size, device)

        # Train the model on the episodes generated in this round.
        main_optimizer.param_groups[0]['lr'] = step_config.optimizer.lr

        total_loss = 0.0
        train_batch_count = 0
        train_tasks = iter_train_batch_tasks(
            episodes_per_round,
            config.train.fresh_pass_factor,
            config.train.replay_pass_factor,
            config.train.batch_size,
            experience.num_files > 0,
        )

        def prepare_train_task(task):
            return prepare_train_batch_task(task, actions, gen_outcomes)

        def train_prepared_batch_group(prepared_batches):
            main_model.zero_grad()
            loss_scale = 1.0 / len(prepared_batches)
            group_loss = 0.0
            for prepared_batch in prepared_batches:
                timer_name = (
                    "round.train_fresh"
                    if prepared_batch.kind == "fresh"
                    else "round.train_replay"
                )
                with profiler.timer(timer_name):
                    group_loss += train_batch_backward(
                        prepared_batch,
                        train_gen_config,
                        loss_scale,
                    )
            train_optimizer_step()
            return group_loss, len(prepared_batches)

        prepared_batch_group = []
        for prepared_batch in train_batch_prefetcher.map(
            train_tasks, prepare_train_task, profiler, "round.train_batch_wait"
        ):
            prepared_batch_group.append(prepared_batch)
            if len(prepared_batch_group) == config.train.gradient_accumulation_steps:
                group_loss, group_count = train_prepared_batch_group(prepared_batch_group)
                total_loss += group_loss
                train_batch_count += group_count
                prepared_batch_group = []
        if prepared_batch_group:
            group_loss, group_count = train_prepared_batch_group(prepared_batch_group)
            total_loss += group_loss
            train_batch_count += group_count

        # Store this round for future replay after direct fresh training is complete.
        with profiler.timer("round.store"):
            experience.store(actions)

        avg_loss = total_loss / train_batch_count if train_batch_count > 0 else 0.0
        log_outcomes(gen_outcomes, avg_loss, round, step_config, num_episodes)
        profiler.add("round.total", time.perf_counter() - round_start)
        if profiler.enabled:
            for name, value in profiler.metrics().items():
                aim_run.track(value, name=name, step=round)
            logging.info("profile round %s:\n%s", round, profiler.format())

        if stop_requested:
            logging.info("Stopping training after completing round %s.", round)
            break
finally:
    train_batch_prefetcher.close()
    generation_executor.shutdown()
    aim_run.close()
