import map_gen
import json
import matplotlib.pyplot as plt

from visualize import display_map

rooms = open("room_geometry.json", "r").read()
# rooms = open("test_geometry.json", "r").read()
room_data = json.loads(rooms)
num_environments = 1
map_size = (72, 72)
engine = map_gen.Engine(rooms, map_size, num_environments, seed=2)

for _ in range(260):
    cand_room_idx, cand_x, cand_y = engine.get_candidates(max_candidates=8, start=0, end=1)
    selected_cand_room_idx = cand_room_idx[:, 0]
    selected_cand_x = cand_x[:, 0]
    selected_cand_y = cand_y[:, 0]
    engine.step(selected_cand_room_idx, selected_cand_x, selected_cand_y, start=0)

    action_room_idx, action_x, action_y = engine.get_actions()
    display_map(room_data, (action_room_idx[0, :], action_x[0, :], action_y[0, :]))
    plt.show()

action_room_idx, action_x, action_y = engine.get_actions()
for room_idx in action_room_idx[0, :]:
    if room_idx < len(room_data):
        print(room_data[room_idx]["name"])
