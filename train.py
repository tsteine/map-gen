import map_gen


rooms = open("room_geometry.json", "r").read()
engine = map_gen.Engine(rooms, map_size=(72, 72), batch_size=4)
