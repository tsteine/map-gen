import map_gen
import json

from visualize import MapVisualizer

rooms = open("room_geometry.json", "r").read()
# rooms = open("test_geometry.json", "r").read()
room_data = json.loads(rooms)
num_environments = 1
map_size = (72, 72)
engine = map_gen.Engine(rooms, map_size, num_environments, seed=2)

visualizer = MapVisualizer(
    room_data,
    map_size=map_size,
    interactive=True,
    show_names=False,
)
visualizer.add_engine_actions(engine.get_actions())

for _ in range(260):
    cand_room_idx, cand_x, cand_y = engine.get_candidates(max_candidates=8, start=0, end=1)
    selected_cand_room_idx = cand_room_idx[:, 0]
    selected_cand_x = cand_x[:, 0]
    selected_cand_y = cand_y[:, 0]
    engine.step(selected_cand_room_idx, selected_cand_x, selected_cand_y, start=0)
    visualizer.add_selected_candidate(
        selected_cand_room_idx,
        selected_cand_x,
        selected_cand_y,
    )
    visualizer.update(pause=0.1)

visualizer.show()
