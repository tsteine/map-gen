import map_gen


# rooms = open("room_geometry.json", "r").read()
rooms = open("test_geometry.json", "r").read()
num_environments = 1
map_size = (20, 20)
engine = map_gen.Engine(rooms, map_size, num_environments, seed=2)

for _ in range(5):
    cand_room_idx, cand_x, cand_y = engine.get_candidates(max_candidates=8, start=0, end=1)
    selected_cand_room_idx = cand_room_idx[:, 0]
    selected_cand_x = cand_x[:, 0]
    selected_cand_y = cand_y[:, 0]
    engine.step(selected_cand_room_idx, selected_cand_x, selected_cand_y, start=0)

print(engine.get_actions())

