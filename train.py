import map_gen
import json
import matplotlib.pyplot as plt

from visualize import MapVisualizer

rooms = open("room_geometry.json", "r").read()
# rooms = open("test_geometry.json", "r").read()
room_data = json.loads(rooms)
num_environments = 1
map_size = (72, 72)
engine = map_gen.Engine(rooms, map_size, num_environments, seed=2)

plt.ion()
fig, ax = plt.subplots()
visualizer = MapVisualizer(room_data, ax=ax, map_size=map_size, show_names=False)
plt.show(block=False)

action_room_idx, action_x, action_y = engine.get_actions()
placements = list(
    zip(
        action_room_idx[0, :].tolist(),
        action_x[0, :].tolist(),
        action_y[0, :].tolist(),
    )
)
visualizer.add_placements(placements)
visualizer.sync_artists()

for _ in range(260):
    cand_room_idx, cand_x, cand_y = engine.get_candidates(max_candidates=8, start=0, end=1)
    selected_cand_room_idx = cand_room_idx[:, 0]
    selected_cand_x = cand_x[:, 0]
    selected_cand_y = cand_y[:, 0]
    engine.step(selected_cand_room_idx, selected_cand_x, selected_cand_y, start=0)
    placement = (
        int(selected_cand_room_idx[0]),
        int(selected_cand_x[0]),
        int(selected_cand_y[0]),
    )
    placements.append(placement)
    visualizer.add_placement(placement)

    visualizer.sync_artists()
    fig.canvas.draw_idle()
    plt.pause(0.1)

plt.ioff()
plt.show()
