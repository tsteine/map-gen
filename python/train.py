import json
import math
import copy
import torch
import logging
import signal
import time
from datetime import datetime
import argparse

import os

from aim import Run
from pydantic import BaseModel
from pathlib import Path

from env import Actions, Engine, GenerateConfig, Outcomes
from model import CausalTransformerModel, FrontierStateModel
from loss import LossConfig, compute_loss
from generate import Prefetcher, generate
from experience import ExperienceStorage
from profile_stats import ProfileStats

class ModelConfig(BaseModel):
    type: str = "causal_transformer"
    compile: bool = True
    autocast: bool = True
    generation_autocast: bool = False
    embedding_width: int 
    key_width: int
    value_width: int
    attn_heads: int
    head_groups: int
    hidden_width: int 
    num_layers: int


class OptimizerConfig(BaseModel):
    lr: float
    beta1: float
    beta2: float
    

class GenerationConfig(BaseModel):
    num_environments: int  # number of maps to generate in parallel
    action_candidates: int  # number of candidates to score for each room placement step
    lookahead_outcomes: bool  # use post-candidate known outcomes when scoring candidates (greater CPU usage, better accuracy)
    temperature0: float  # initial temperature (higher = candidates selected more randomly)
    temperature1: float  # final temperature
    state_candidate_chunk: int = 1
    state_environment_chunk: int = 8
    frontier_neighbor_count: int = 4
    frontier_window_size: int = 16
    num_threads: int | None = None

    
class TrainConfig(BaseModel):
    batch_size: int  # number of episodes per training batch
    fresh_pass_factor: float  # number of passes over just-generated episodes
    replay_pass_factor: float  # average number of passes over past episodes
    episodes_per_file: int  # number of episodes to read from each file (higher values = lower disk I/O, lower values = more diverse sampling)
    hist_c: float  # extent to which replay sampling biases towards recent episodes (0.0 = no bias)
    door_weight: float  # amount of weight assigned to door outcomes in the loss function
    connection_weight: float  # amount of weight assigned to connection outcomes in the loss function
    ema_decay: float  # decay factor for exponential moving average model used during generation
    state_prefix_samples: int = 1
    state_batch_chunk: int = 8


class Config(BaseModel):
    experiment_name: str  # string/identifier for the experiment (doesn't have to be unique)
    room_set: Path  # path to JSON file defining the set of rooms to use for map generation
    annealing_episodes: int  # number of episodes over which to ramp the temperature from temperature0 to temperature1
    total_episodes: int  # total number of episodes to generate
    map_size: tuple[int, int]  # dimensions of the map grid within which rooms are placed
    model: ModelConfig
    optimizer: OptimizerConfig
    generation: GenerationConfig
    train: TrainConfig


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
    choices=("auto", "cpu", "cuda"),
    default="auto",
    help="training device (default: auto; uses CUDA when available)",
)
args = parser.parse_args()
config = Config.parse_file(args.config)

verify_outcome_consistency = args.verify_outcome_consistency
profiler = ProfileStats(args.profile)
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
if config.train.state_prefix_samples <= 0:
    raise ValueError("train.state_prefix_samples must be greater than zero")
if config.train.state_batch_chunk <= 0:
    raise ValueError("train.state_batch_chunk must be greater than zero")


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

if config.train.fresh_pass_factor != 0.0 and config.generation.num_environments % config.train.batch_size != 0:
    raise ValueError(
        "train.batch_size must evenly divide generation.num_environments when "
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
elif not torch.cuda.is_available():
    raise RuntimeError("--device cuda requested, but CUDA is not available")
else:
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision('high')
    if (config.model.autocast or config.model.generation_autocast) and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "CUDA bfloat16 autocast requested, but this GPU does not support bfloat16. "
            "Use --device cpu for float32 CPU execution or set model.autocast=false "
            "and model.generation_autocast=false for float32 CUDA execution."
        )

train_precision = "bfloat16 autocast" if device.type == "cuda" and config.model.autocast else "float32"
generation_precision = (
    "bfloat16 autocast" if device.type == "cuda" and config.model.generation_autocast else "float32"
)
logging.info(
    "Using device %s with %s training and %s generation.",
    device,
    train_precision,
    generation_precision,
)

engine = Engine(rooms)
gen_env = engine.create_environment_group(
    config.map_size,
    config.generation.num_environments,
    seed=0,
    frontier_neighbor_count=config.generation.frontier_neighbor_count,
    frontier_window_size=config.generation.frontier_window_size,
    num_threads=config.generation.num_threads,
)
train_env = engine.create_environment_group(
    config.map_size,
    config.train.batch_size,
    frontier_neighbor_count=config.generation.frontier_neighbor_count,
    frontier_window_size=config.generation.frontier_window_size,
    num_threads=config.generation.num_threads,
)
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
).to(device)

ema_model = copy.deepcopy(main_model).to(device)
ema_model.requires_grad_(False)
ema_model.eval()
if config.model.type == "frontier_state" and config.model.compile:
    main_model = torch.compile(main_model)
    ema_model = torch.compile(ema_model)


def update_ema_model():
    with torch.no_grad():
        for ema_param, main_param in zip(ema_model.parameters(), main_model.parameters()):
            ema_param.lerp_(main_param, 1.0 - config.train.ema_decay)


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

def log_outcomes(outcomes, loss, round, frac, num_episodes):
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
        "frac": frac,
        "num_episodes": num_episodes,
    }

    for name, value in metrics.items():
        aim_run.track(value, name=name, step=round)
    
    logging.info(f"round {round}, loss {loss:.4f}, succ {success_rate:.4f}, total {avg_invalid:.2f} (min {min_invalid}), door {avg_door:.2f} (min {min_door}), conn {avg_conn:.2f} (min {min_conn}), frac {frac:.4f}")


def get_gen_config(frac):
    temperature0 = config.generation.temperature0
    temperature1 = config.generation.temperature1
    temperature = temperature0 * (temperature1 / temperature0) ** frac
    return GenerateConfig(
        episode_length=len(rooms),
        max_candidates=config.generation.action_candidates,
        temperature=torch.full([config.generation.num_environments],
            temperature, dtype=torch.float32, device=device),
        lookahead_outcomes=config.generation.lookahead_outcomes,
        state_candidate_chunk=config.generation.state_candidate_chunk,
        state_environment_chunk=config.generation.state_environment_chunk,
        state_autocast=config.model.generation_autocast,
        training_autocast=config.model.autocast,
    )


episodes_per_round = config.generation.num_environments
experience_path = f"{run_path}/experience"
experience = ExperienceStorage(num_rooms, experience_path, episodes_per_round)

main_optimizer = torch.optim.Adam(
    main_model.parameters(),
    lr=config.optimizer.lr,
    betas=(config.optimizer.beta1, config.optimizer.beta2))

train_prefetcher = Prefetcher()

num_episodes = 0


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
    

def train_batch(train_actions, train_outcomes, gen_config):
    main_model.zero_grad()
    if getattr(main_model, "uses_state_features", False):
        repeated_outcomes = Outcomes(
            door_invalid=train_outcomes.door_invalid.unsqueeze(1),
            connection_invalid=train_outcomes.connection_invalid.unsqueeze(1),
        )
        mask = torch.ones([train_actions.room_idx.shape[0], 1, 1], dtype=torch.bool, device=device)
        prefix_lengths = torch.randint(
            1, episode_length + 1, [config.train.state_prefix_samples]).sort().values.tolist()
        with profiler.timer("train.cpu_setup"):
            train_actions_cpu = train_actions.to(torch.device("cpu"))
            train_env.clear()
        current_prefix_length = 0
        total_loss = 0.0

        def prepare_prefix(prefix_length):
            nonlocal current_prefix_length
            with profiler.timer("train.cpu_prefix_prepare"):
                for action_idx in range(current_prefix_length, prefix_length):
                    train_env.step(Actions(
                        train_actions_cpu.room_idx[:, action_idx],
                        train_actions_cpu.room_x[:, action_idx],
                        train_actions_cpu.room_y[:, action_idx],
                    ))
                current_prefix_length = prefix_length
                return [
                    (
                        start,
                        min(start + config.train.state_batch_chunk, train_actions.room_idx.shape[0]),
                        train_env.get_state_features(
                            torch.device("cpu"),
                            start,
                            min(config.train.state_batch_chunk, train_actions.room_idx.shape[0] - start),
                        ),
                    )
                    for start in range(0, train_actions.room_idx.shape[0], config.train.state_batch_chunk)
                ]

        for feature_chunks in train_prefetcher.map(
            prefix_lengths, prepare_prefix, profiler, "train.cpu_prefix_wait"
        ):
            for start, end, chunk_features in feature_chunks:
                chunk_weight = (end - start) / train_actions.room_idx.shape[0] / len(prefix_lengths)
                with profiler.timer("train.cpu_transfer_submit"):
                    chunk_features = chunk_features.to(device)
                with profiler.cuda_timer("train.gpu_forward_backward", device):
                    with torch.amp.autocast(
                        "cuda",
                        dtype=torch.bfloat16,
                        enabled=device.type == "cuda" and config.model.autocast,
                    ):
                        chunk_preds = main_model(chunk_features)
                    chunk_loss = compute_loss(
                        chunk_preds,
                        Outcomes(
                            repeated_outcomes.door_invalid[start:end],
                            repeated_outcomes.connection_invalid[start:end],
                        ),
                        mask[start:end],
                        loss_config,
                    )
                    (chunk_loss * chunk_weight).backward()
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
            loss.backward()
    with profiler.cuda_timer("train.gpu_optimizer", device):
        grad_norm = torch.nn.utils.clip_grad_norm_(main_model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient norm: {grad_norm.item()}")
        main_optimizer.step()
        update_ema_model()

    return loss.item()

try:
    for round in range(config.total_episodes // episodes_per_round):
        profiler.reset()
        round_start = time.perf_counter()
        frac = min(num_episodes / config.annealing_episodes, 1.0)

        # Generate new maps:
        gen_config = get_gen_config(frac)
        with profiler.timer("round.generate"):
            actions, gen_outcomes = generate(
                gen_env,
                ema_model,
                gen_config,
                device,
                verify_outcome_consistency=verify_outcome_consistency,
                profiler=profiler,
            )
        num_episodes += config.generation.num_environments

        # Train the model on the episodes generated in this round.
        total_loss = 0.0
        train_batch_count = 0
        for start in iter_fresh_batch_starts(episodes_per_round, config.train.fresh_pass_factor, config.train.batch_size):
            train_actions, train_outcomes = select_batch(actions, gen_outcomes, start, config.train.batch_size)
            with profiler.timer("round.train_fresh"):
                total_loss += train_batch(train_actions, train_outcomes, gen_config)
            train_batch_count += 1

        # Train on replay batches sampled from previous rounds' stored experience.
        if experience.num_files > 0:
            for _ in range(num_replay_batches(episodes_per_round, config.train.replay_pass_factor, config.train.batch_size)):
                with profiler.timer("round.replay_prepare"):
                    replay_actions = experience.sample(
                        config.train.batch_size,
                        config.train.episodes_per_file,
                        config.train.hist_c,
                    )
                    train_env.replay(replay_actions)
                    replay_actions = replay_actions.to(device)
                    replay_outcomes = train_env.get_outcomes(device)
                with profiler.timer("round.train_replay"):
                    total_loss += train_batch(replay_actions, replay_outcomes, gen_config)
                train_batch_count += 1

        # Store this round for future replay after direct fresh training is complete.
        with profiler.timer("round.store"):
            experience.store(actions)

        avg_loss = total_loss / train_batch_count if train_batch_count > 0 else 0.0
        log_outcomes(gen_outcomes, avg_loss, round, frac, num_episodes)
        profiler.add("round.total", time.perf_counter() - round_start)
        if profiler.enabled:
            for name, value in profiler.metrics().items():
                aim_run.track(value, name=name, step=round)
            logging.info("profile round %s: %s", round, profiler.format())

        if stop_requested:
            logging.info("Stopping training after completing round %s.", round)
            break
finally:
    train_prefetcher.close()
    aim_run.close()
