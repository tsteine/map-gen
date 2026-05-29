import json
import math
import torch
import logging
from datetime import datetime
import argparse

import os

from aim import Run
from pydantic import BaseModel
from pathlib import Path

from env import Engine, GenerateConfig, Outcomes
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
    batch_size: int  # number of episodes to sample per training batch
    pass_factor: float  # average number of total times a given episode is sampled
    episodes_per_file: int  # number of episodes to read from each file (higher values = lower disk I/O, lower values = more diverse sampling)
    hist_c: float  # extent to which sampling biases towards recent episodes (0.0 = no bias)
    door_weight: float  # amount of weight assigned to door outcomes in the loss function
    connection_weight: float  # amount of weight assigned to connection outcomes in the loss function


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

rooms_str = open(config.room_set, "r").read()
rooms = json.loads(rooms_str)
num_rooms = len(rooms)
episode_length = num_rooms
device = torch.device("cuda:0")

engine = Engine(rooms)
env = engine.create_environment_group(config.map_size, config.generation.num_environments, seed=0)
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
run = Run(experiment=config.experiment_name)
run["config"] = json.loads(config.model_dump_json())

loss_config = LossConfig(
    door_weight=config.train.door_weight,
    connection_weight=config.train.connection_weight,
)

def log_outcomes(outcomes, loss, round, frac):
    door_invalid = torch.sum(outcomes.door_invalid != 0, dim=1)
    avg_door = torch.mean(door_invalid.to(torch.float32))
    min_door = torch.min(door_invalid)
    run.track(avg_door, name="avg_door", step=round)
    run.track(min_door, name="min_door", step=round)
    
    connection_invalid = torch.sum(outcomes.connection_invalid != 0, dim=1)
    avg_connection = torch.mean(connection_invalid.to(torch.float32))
    min_connection = torch.min(connection_invalid)
    run.track(avg_connection, name="avg_connection", step=round)
    run.track(min_connection, name="min_connection", step=round)
    
    total_invalid = door_invalid + connection_invalid
    avg_invalid = torch.mean(total_invalid.to(torch.float32))
    min_invalid = torch.min(total_invalid)
    run.track(avg_invalid, name="avg_invalid", step=round)
    run.track(min_invalid, name="min_invalid", step=round)
    
    logging.info(f"round {round}, loss {loss:.4f}, total {avg_invalid:.2f} (min {min_invalid}), door {avg_door:.2f} (min {min_door}), conn {avg_connection:.2f} (min {min_connection}), frac {frac:.4f}")


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


experience_path = f"{run_path}/experience"
experience = ExperienceStorage(num_rooms, experience_path) 

main_optimizer = torch.optim.Adam(
    main_model.parameters(),
    lr=config.optimizer.lr,
    betas=(config.optimizer.beta1, config.optimizer.beta2))

scaler = torch.cuda.amp.GradScaler()

episodes_per_round = config.generation.num_environments
num_batches = int(math.ceil(episodes_per_round * config.train.pass_factor / config.train.batch_size))
num_episodes = 0

for round in range(config.total_episodes):
    frac = min(num_episodes / config.annealing_episodes, 1.0)

    # Generate new maps:
    gen_config = get_gen_config(frac)
    actions, outcomes = generate(env, main_model, gen_config, device)
    num_episodes += config.generation.num_environments

    # Store them as experience (to disk):
    experience.store(actions)    

    # Train the model on samples of past experience
    for _ in range(num_batches):
        actions = experience.sample(config.train.batch_size, config.train.episodes_per_file, config.train.hist_c)
        
        
        main_model.zero_grad()
        preds = main_model(actions, gen_config)

        repeated_outcomes = Outcomes(
            door_invalid=outcomes.door_invalid.unsqueeze(1).repeat(1, episode_length, 1),
            connection_invalid=outcomes.connection_invalid.unsqueeze(1).repeat(1, episode_length, 1),
        )
        mask = (actions.room_idx < num_rooms).unsqueeze(2)  # exclude dummy actions
        loss = compute_loss(preds, repeated_outcomes, mask, loss_config)
    
        scaler.scale(loss).backward()
        scaler.step(main_optimizer)
        scaler.update()
    
    log_outcomes(outcomes, loss, round, frac)
