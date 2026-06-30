import json
import re
from pathlib import Path


ROOM_DEFINITION_DIR = Path("room_definitions")
OLD_GEOMETRY_PATH = Path("/home/kerby/MapRandomizer/room_geometry.json")
SM_JSON_REGION_DIR = Path("/home/kerby/sm-json-data/region")

DOOR_RE = re.compile(r"\{([^{}\n]*\"direction\"[^{}\n]*\"kind\"[^{}\n]*)\}")


def load_json(path):
    return json.loads(path.read_text())


def build_nodes_by_room_and_address():
    out = {}
    for path in SM_JSON_REGION_DIR.rglob("*.json"):
        room = load_json(path)
        for node in room.get("nodes", []):
            node_address = node.get("nodeAddress")
            if node_address is None:
                continue
            address = int(node_address, 16)
            value = (node["id"], room["id"], room["name"], path)
            out.setdefault((room["id"], address), []).append((node, value))
    return out


def build_old_door_index():
    old_rooms = load_json(OLD_GEOMETRY_PATH)
    by_room = {}
    duplicate_keys = []
    none_exit_ptr = []
    for room in old_rooms:
        doors = {}
        for door in room["doors"]:
            key = (door["direction"], door["x"], door["y"])
            if key in doors:
                duplicate_keys.append((room["room_id"], room["name"], key))
            doors[key] = door
            if door.get("exit_ptr") is None:
                none_exit_ptr.append((room["room_id"], room["name"], door))
        by_room[room["room_id"]] = (room["name"], doors)
    if duplicate_keys:
        raise RuntimeError(f"duplicate old door keys: {duplicate_keys[:10]}")
    return by_room, none_exit_ptr


def parse_door_object(raw):
    wrapped = "{" + raw + "}"
    return json.loads(wrapped)


def node_marks_door_tile(node, x, y):
    mask = node.get("mapTileMask")
    if mask is None or y >= len(mask) or x >= len(mask[y]):
        return False
    return mask[y][x] == 2


def apply_ids_to_file(path, old_doors_by_room, nodes_by_room_and_address, dry_run):
    rooms = load_json(path)
    room_by_id = {room["room_id"]: room for room in rooms}
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    room_iter = iter(rooms)
    current_room = None
    current_room_name = None
    changed = 0
    skipped_none_exit_ptr = []
    skipped_unresolved = []
    failures = []

    for i, line in enumerate(lines):
        if "\"room_id\"" in line:
            current_room = next(room_iter)
            current_room_name = current_room["name"]

        if "\"direction\"" not in line:
            continue

        def replace(match):
            nonlocal changed
            door = parse_door_object(match.group(1))
            if "id" in door:
                return match.group(0)

            old_room_name, old_doors = old_doors_by_room[current_room["room_id"]]
            key = (door["direction"], door["x"], door["y"])
            old_door = old_doors.get(key)
            if old_door is None:
                failures.append(
                    f"{path}:{i + 1}: no old door for room {current_room['room_id']} "
                    f"{current_room_name!r} key {key}"
                )
                return match.group(0)
            if old_room_name != current_room_name:
                failures.append(
                    f"{path}:{i + 1}: room name mismatch for room {current_room['room_id']}: "
                    f"{current_room_name!r} vs {old_room_name!r}"
                )
                return match.group(0)
            exit_ptr = old_door.get("exit_ptr")
            if exit_ptr is None:
                skipped_none_exit_ptr.append((current_room["room_id"], current_room_name, key))
                return match.group(0)
            nodes = nodes_by_room_and_address.get((current_room["room_id"], exit_ptr), [])
            candidates = [
                value
                for node, value in nodes
                if node.get("doorOrientation") == door["direction"]
                and node_marks_door_tile(node, door["x"], door["y"])
            ]
            if not candidates:
                fallback_candidates = [
                    value
                    for values in nodes_by_room_and_address.values()
                    for node, value in values
                    if value[1] == current_room["room_id"]
                    and node.get("doorOrientation") == door["direction"]
                    and node_marks_door_tile(node, door["x"], door["y"])
                ]
                if len(fallback_candidates) == 1:
                    candidates = fallback_candidates
                else:
                    skipped_unresolved.append(
                        (current_room["room_id"], current_room_name, key, exit_ptr)
                    )
                    return match.group(0)
            if len(candidates) > 1:
                failures.append(
                    f"{path}:{i + 1}: multiple nodes for exit_ptr {exit_ptr} "
                    f"in room {current_room['room_id']} {current_room_name!r} key {key}: "
                    f"{candidates}"
                )
                return match.group(0)
            node_id, sm_room_id, sm_room_name, sm_path = candidates[0]
            changed += 1
            return "{\"id\": " + str(node_id) + ", " + match.group(1) + "}"

        lines[i] = DOOR_RE.sub(replace, line)

    if failures:
        raise RuntimeError("\n".join(failures))

    new_text = "".join(lines)
    if not dry_run and new_text != text:
        path.write_text(new_text)
    return changed, skipped_none_exit_ptr, skipped_unresolved


def main():
    dry_run = "--write" not in __import__("sys").argv
    nodes_by_room_and_address = build_nodes_by_room_and_address()
    old_doors_by_room, none_exit_ptr = build_old_door_index()
    total = 0
    skipped = []
    unresolved = []
    for path in sorted(ROOM_DEFINITION_DIR.glob("*.json")):
        changed, file_skipped, file_unresolved = apply_ids_to_file(
            path, old_doors_by_room, nodes_by_room_and_address, dry_run
        )
        total += changed
        skipped.extend(file_skipped)
        unresolved.extend(file_unresolved)
        print(f"{path}: {'would add' if dry_run else 'added'} {changed} ids")

    print(f"total {'would add' if dry_run else 'added'}: {total}")
    print(f"old geometry exit_ptr None count: {len(none_exit_ptr)}")
    if skipped:
        print("current room doors skipped because exit_ptr is None:")
        for item in skipped:
            print(item)
    if unresolved:
        print("current room doors skipped because no current-room nodeAddress match exists:")
        for item in unresolved:
            print(item)


if __name__ == "__main__":
    main()
