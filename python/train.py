import json
import math
import copy
import torch
import logging
import signal
from datetime import datetime
import argparse

import os

from aim import Run
from pydantic import BaseModel
from pathlib import Path

from env import Actions, Engine, GenerateConfig, Outcomes
from model import CausalTransformerModel
from loss import LossConfig, compute_loss
from generate import generate
from experience import ExperienceStorage

class ModelConfig(BaseModel):
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
    temperature0: float  # initial temperature (higher = candidates selected more randomly)
    temperature1: float  # final temperature

    
class TrainConfig(BaseModel):
    batch_size: int  # number of episodes per training batch
    fresh_pass_factor: float  # number of passes over just-generated episodes
    replay_pass_factor: float  # average number of passes over past episodes
    episodes_per_file: int  # number of episodes to read from each file (higher values = lower disk I/O, lower values = more diverse sampling)
    hist_c: float  # extent to which replay sampling biases towards recent episodes (0.0 = no bias)
    door_weight: float  # amount of weight assigned to door outcomes in the loss function
    connection_weight: float  # amount of weight assigned to connection outcomes in the loss function
    ema_decay: float  # decay factor for exponential moving average model used during generation


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
args = parser.parse_args()
config = Config.parse_file(args.config)


start_time = datetime.now()
run_path = f"runs/{start_time.isoformat()}-{config.experiment_name}/"
os.makedirs(run_path, exist_ok=True)
logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(f"{run_path}/train-{start_time.isoformat()}.log"),
                              logging.StreamHandler()])

logging.info("Config:\n{}".format(config.model_dump_json(indent=2)))

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
device = torch.device("cuda:0")

engine = Engine(rooms)
gen_env = engine.create_environment_group(config.map_size, config.generation.num_environments, seed=0)
train_env = engine.create_environment_group(config.map_size, config.train.batch_size)
output_sizes = engine.get_output_sizes()

main_model = CausalTransformerModel(
    num_rooms=len(rooms),
    map_x=config.map_size[0],
    map_y=config.map_size[1],
    output_sizes=output_sizes,
    embedding_width=config.model.embedding_width,
    key_width=config.model.key_width,
    value_width=config.model.value_width,
    attn_heads=config.model.attn_heads,
    head_groups=config.model.head_groups,
    hidden_width=config.model.hidden_width,
    num_layers=config.model.num_layers,
).to(device)

ema_model = copy.deepcopy(main_model).to(device)
ema_model.requires_grad_(False)
ema_model.eval()


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

def log_outcomes(outcomes, loss, round, frac):
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
    )


episodes_per_round = config.generation.num_environments
experience_path = f"{run_path}/experience"
experience = ExperienceStorage(num_rooms, experience_path, episodes_per_round)

main_optimizer = torch.optim.Adam(
    main_model.parameters(),
    lr=config.optimizer.lr,
    betas=(config.optimizer.beta1, config.optimizer.beta2))

scaler = torch.amp.GradScaler('cuda')

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
    preds = main_model(train_actions, gen_config)

    repeated_outcomes = Outcomes(
        door_invalid=train_outcomes.door_invalid.unsqueeze(1).repeat(1, episode_length, 1),
        connection_invalid=train_outcomes.connection_invalid.unsqueeze(1).repeat(1, episode_length, 1),
    )
    mask = (train_actions.room_idx < num_rooms).unsqueeze(2)  # exclude dummy actions
    loss = compute_loss(preds, repeated_outcomes, mask, loss_config)

    scaler.scale(loss).backward()
    scaler.step(main_optimizer)
    scaler.update()
    update_ema_model()

    return loss.item()

try:
    for round in range(config.total_episodes // episodes_per_round):
        frac = min(num_episodes / config.annealing_episodes, 1.0)

        # Generate new maps:
        gen_config = get_gen_config(frac)
        actions, gen_outcomes = generate(gen_env, ema_model, gen_config, device)
        num_episodes += config.generation.num_environments

        # Train the model on the episodes generated in this round.
        total_loss = 0.0
        train_batch_count = 0
        for start in iter_fresh_batch_starts(episodes_per_round, config.train.fresh_pass_factor, config.train.batch_size):
            train_actions, train_outcomes = select_batch(actions, gen_outcomes, start, config.train.batch_size)
            total_loss += train_batch(train_actions, train_outcomes, gen_config)
            train_batch_count += 1

        # Train on replay batches sampled from previous rounds' stored experience.
        if experience.num_files > 0:
            for _ in range(num_replay_batches(episodes_per_round, config.train.replay_pass_factor, config.train.batch_size)):
                replay_actions = experience.sample(
                    config.train.batch_size,
                    config.train.episodes_per_file,
                    config.train.hist_c,
                )
                train_env.replay(replay_actions)
                replay_actions = replay_actions.to(device)
                replay_outcomes = train_env.get_outcomes(device)
                total_loss += train_batch(replay_actions, replay_outcomes, gen_config)
                train_batch_count += 1

        # Store this round for future replay after direct fresh training is complete.
        experience.store(actions)

        avg_loss = total_loss / train_batch_count if train_batch_count > 0 else 0.0
        log_outcomes(gen_outcomes, avg_loss, round, frac)

        if stop_requested:
            logging.info("Stopping training after completing round %s.", round)
            break
finally:
    aim_run.close()
