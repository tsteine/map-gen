import map_gen


# rooms = open("room_geometry.json", "r").read()
rooms = open("test_geometry.json", "r").read()
num_environments = 1
map_size = (20, 20)
engine = map_gen.Engine(rooms, map_size, num_environments, seed=0)
print(engine.actions())