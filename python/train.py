import json
import time
import numpy as np
import torch
import logging
from datetime import datetime
import os

from aim import Run

from env import Engine, GenerationConfig, Outcomes
from model import CausalTransformerModel
from loss import LossConfig, compute_loss
from generate import generate

start_time = datetime.now()
os.makedirs("logs", exist_ok=True)
logging.basicConfig(format='%(asctime)s %(message)s',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(f"logs/train-{start_time.isoformat()}.log"),
                              logging.StreamHandler()])


rooms_str = open("room_definitions/crateria.json", "r").read()
# rooms_str = open("room_definitions/zebes.json", "r").read()
rooms = json.loads(rooms_str)
episode_length = len(rooms)
num_environments = 512
num_rounds = 10000
max_candidates = 16
map_size = (72, 72)
temperature = 0.03
device = torch.device("cuda:0")

engine = Engine(rooms)
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
).to(device)

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


gen_config = GenerationConfig(
    episode_length=len(rooms),
    max_candidates=max_candidates,
    temperature=torch.full([num_environments], temperature, dtype=torch.float32, device=device),
)

loss_config = LossConfig(
    door_weight=1.0,
    connection_weight=1.0,
)

def log_outcomes(outcomes, loss, round):
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
    
    logging.info(f"loss: {loss:.4f}, total: {avg_invalid:.2f} (min: {min_invalid}), door: {avg_door:.2f} (min: {min_door}), conn: {avg_connection:.2f} (min: {min_connection})")
    

main_optimizer = torch.optim.Adam(main_model.parameters(), lr=0.0001)


start = time.perf_counter()
for round in range(num_rounds):
    actions, outcomes = generate(env, main_model, gen_config, device)

    main_optimizer.zero_grad()
    preds = main_model(actions, gen_config)
    repeated_outcomes = Outcomes(
        door_invalid=outcomes.door_invalid.unsqueeze(1).repeat(1, episode_length, 1),
        connection_invalid=outcomes.connection_invalid.unsqueeze(1).repeat(1, episode_length, 1),
    )
    loss = compute_loss(preds, repeated_outcomes, loss_config)
    loss.backward()
    main_optimizer.step()

    log_outcomes(outcomes, loss, round)

end = time.perf_counter()
print(f"Elapsed time: {(end - start):.3f} seconds, {(end - start)/(num_rounds*num_environments):.5f} seconds per episode")
