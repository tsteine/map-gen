import map_gen
import json
import time
import numpy as np

from visualize import MapVisualizer

# rooms_str = open("room_geometry/crateria.json", "r").read()
rooms_str = open("room_geometry/zebes.json", "r").read()
rooms = json.loads(rooms_str)
# num_environments = 4096
num_environments = 1
max_candidates = 32
num_rounds = 128
map_size = (32, 32)
# map_size = (72, 72)

engine = map_gen.Engine(rooms_str)
env = engine.create_environment_group(map_size, num_environments, seed=6)

# visualizer = MapVisualizer(
#     rooms,
#     map_size=map_size,
#     interactive=True,
#     show_names=False,
# )

start = time.perf_counter()
for _ in range(num_rounds):
    round_start = time.perf_counter()
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
        
        # outcomes = env.get_outcomes()
        # print(outcomes)
        # visualizer.add_selected_candidate(
        #     selected_cand_room_idx,
        #     selected_cand_x,
        #     selected_cand_y,
        # )
        # visualizer.update(pause=0.1)

    room_idx, x, y = env.get_actions()
    dummy_cnt = np.count_nonzero(room_idx == len(rooms))
    round_end = time.perf_counter()
    print(f"Elapsed time: {round_end - round_start:.4f} seconds, placed {dummy_cnt} dummy rooms")
    # visualizer.show()

end = time.perf_counter()
print(f"Elapsed time: {(end - start)/(num_rounds*num_environments):.5f} seconds per episode")
