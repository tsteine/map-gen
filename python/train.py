import map_gen
import json
import time
import numpy as np
import torch

from model import CausalTransformerModel

# rooms_str = open("room_definitions/crateria.json", "r").read()
rooms_str = open("room_definitions/zebes.json", "r").read()
rooms = json.loads(rooms_str)
num_environments = 4096
num_rounds = 1000
max_candidates = 32
map_size = (72, 72)

engine = map_gen.Engine(rooms_str)
env = engine.create_environment_group(map_size, num_environments, seed=6)

num_doors, num_connects = engine.get_output_sizes()
num_outputs = num_doors + num_connects

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
    num_outputs=num_outputs,
    embedding_width=embedding_width,
    key_width=key_width,
    value_width=value_width,
    attn_heads=attn_heads,
    head_groups=head_groups,
    hidden_width=hidden_width,
    num_layers=num_layers,
)




# visualizer = MapVisualizer(
#     rooms,
#     map_size=map_size,
#     interactive=True,
#     show_names=False,
# )

start = time.perf_counter()
for _ in range(num_rounds):
    # round_start = time.perf_counter()
    env.clear()
    env.initial_step()
    
    
    
    # visualizer.add_engine_actions(env.get_actions())
    for step in range(len(rooms) - 1):
        cand_room_idx, cand_x, cand_y = env.get_candidates(
            max_candidates=max_candidates
        )

        # print("step {}: candidates: {}".format(step, np.count_nonzero(cand_room_idx != 253, axis=1)))
        selected_cand_room_idx = np.ascontiguousarray(cand_room_idx[:, 0])
        selected_cand_x = np.ascontiguousarray(cand_x[:, 0])
        selected_cand_y = np.ascontiguousarray(cand_y[:, 0])
        env.step(selected_cand_room_idx, selected_cand_x, selected_cand_y)
    
    env.finish()
    
    
        # print(outcomes)
        # visualizer.add_selected_candidate(
        #     selected_cand_room_idx,
        #     selected_cand_x,
        #     selected_cand_y,
        # )
        # visualizer.update(pause=0.1)

    # door_valid, connection_valid = env.get_outcomes()
    # assert np.all(door_valid >= 0)
    # assert np.all(connection_valid >= 0)
    # total_invalid_door = np.count_nonzero(door_valid, axis=1)
    # total_invalid_connection = np.count_nonzero(connection_valid, axis=1)
    # total_invalid = total_invalid_door + total_invalid_connection
    # print(f"Total invalid outcomes per environment: {sorted(list(total_invalid))}")
    # print(f"Total invalid doors per environment: {sorted(list(total_invalid_door))}")
    # print(f"Total invalid connections per environment: {sorted(list(total_invalid_connection))}")

    # room_idx, x, y = env.get_actions()
    # dummy_cnt = np.count_nonzero(room_idx == len(rooms))
    # round_end = time.perf_counter()
    # print(f"Elapsed time: {round_end - round_start:.4f} seconds, placed {dummy_cnt} dummy rooms")
    # visualizer.show()

end = time.perf_counter()
print(f"Elapsed time: {(end - start)/(num_rounds*num_environments):.5f} seconds per episode")
