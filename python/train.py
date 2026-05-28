from doctest import OutputChecker

import map_gen
import json
import time
import numpy as np
import torch
import logging
from datetime import datetime
import os

from aim import Run

from model import CausalTransformerModel
from generate import generate, GenerationConfig

start_time = datetime.now()
os.makedirs("logs", exist_ok=True)
logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(f"logs/train-{start_time.isoformat()}.log"),
                              logging.StreamHandler()])


# rooms_str = open("room_definitions/crateria.json", "r").read()
rooms_str = open("room_definitions/zebes.json", "r").read()
rooms = json.loads(rooms_str)
num_environments = 4
num_rounds = 10
max_candidates = 32
map_size = (72, 72)
temperature = 1.0
device = torch.device("cpu")

engine = map_gen.Engine(rooms_str)
env = engine.create_environment_group(map_size, num_environments, seed=6)
output_sizes = engine.get_output_sizes()

embedding_width = 256
key_width = 64
value_width = 64
attn_heads = 16
head_groups = 4
hidden_width = 512
num_layers = 4

main_model = CausalTransformerModel(
    num_rooms=len(rooms),
    map_x=map_size[0],
    map_y=map_size[1],
    output_sizes=output_sizes,
    embedding_width=embedding_width,
    key_width=key_width,
    value_width=value_width,
    attn_heads=attn_heads,
    head_groups=head_groups,
    hidden_width=hidden_width,
    num_layers=num_layers,
)

# Log experiment using Aim
run = Run(experiment="initial testing")
run["model"] = {
    "num_rooms": len(rooms),
    "map_x": map_size[0],
    "map_y": map_size[1],
    "output_sizes": output_sizes,
    "embedding_width": embedding_width,
    "key_width": key_width,
    "value_width": value_width,
    "attn_heads": attn_heads,
    "head_groups": head_groups,
    "hidden_width": hidden_width,
    "num_layers": num_layers,
}



config = GenerationConfig(
    episode_length=len(rooms),
    max_candidates=max_candidates,
    temperature=torch.full([num_environments], temperature, dtype=torch.float32),
)

def log_outcomes(outcomes, round):
    door_invalid, connection_invalid = outcomes
    
    door_invalid = np.count_nonzero(door_invalid, axis=1)
    avg_door = np.mean(door_invalid)
    min_door = np.min(door_invalid)
    run.track(avg_door, name="avg_door", step=round)
    run.track(min_door, name="min_door", step=round)
    
    connection_invalid = np.count_nonzero(connection_invalid, axis=1)
    avg_connection = np.mean(connection_invalid)
    min_connection = np.min(connection_invalid)
    run.track(avg_connection, name="avg_connection", step=round)
    run.track(min_connection, name="min_connection", step=round)
    
    total_invalid = door_invalid + connection_invalid
    avg_invalid = np.mean(total_invalid)
    min_invalid = np.min(total_invalid)
    run.track(avg_invalid, name="avg_invalid", step=round)
    run.track(min_invalid, name="min_invalid", step=round)
    
    logging.info(f"total: {avg_invalid:.2f} (min: {min_invalid}), door: {avg_door:.2f} (min: {min_door}), conn: {avg_connection:.2f} (min: {min_connection})")
    

start = time.perf_counter()
for round in range(num_rounds):
    actions, outcomes = generate(env, main_model, config, device)
    log_outcomes(outcomes, round)

end = time.perf_counter()
print(f"Elapsed time: {(end - start):.3f} seconds, {(end - start)/(num_rounds*num_environments):.5f} seconds per episode")
