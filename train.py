import map_gen
import json
import time
import numpy as np

from visualize import MapVisualizer

rooms = open("room_geometry.json", "r").read()
# rooms = open("test_geometry.json", "r").read()
room_data = json.loads(rooms)
num_environments = 1
max_candidates = 128
map_size = (72, 72)

start = time.perf_counter()
engine = map_gen.Engine(rooms, map_size, num_environments, seed=2)
end = time.perf_counter()
print(f"Elapsed time: {end - start:.2f} seconds to create the engine")

# visualizer = MapVisualizer(
#     room_data,
#     map_size=map_size,
#     interactive=True,
#     show_names=False,
# )


for _ in range(1000):
    start = time.perf_counter()
    engine.clear()
    engine.initial_step()
    # visualizer.add_engine_actions(engine.get_actions())
    for _ in range(252):
        cand_room_idx, cand_x, cand_y = engine.get_candidates(
            max_candidates=max_candidates,
            start=0,
            end=num_environments)
        selected_cand_room_idx = np.ascontiguousarray(cand_room_idx[:, 0])
        selected_cand_x = np.ascontiguousarray(cand_x[:, 0])
        selected_cand_y = np.ascontiguousarray(cand_y[:, 0])
        engine.step(selected_cand_room_idx, selected_cand_x, selected_cand_y, start=0)
        # visualizer.add_selected_candidate(
        #     selected_cand_room_idx,
        #     selected_cand_x,
        #     selected_cand_y,
        # )
        # visualizer.update(pause=0.1)

    room_idx, x, y = engine.get_actions()
    dummy_cnt = np.count_nonzero(room_idx == 253)
    end = time.perf_counter()
    # print(f"Elapsed time: {end - start:.3f} seconds, placed {dummy_cnt} dummy rooms")
    # visualizer.show()
