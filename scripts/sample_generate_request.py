#!/usr/bin/env python3
import json
import urllib.error
import urllib.request


URL = "http://127.0.0.1:5000/generate"
AREA_COUNT = 6


def area_room_counts(area_assignments: list[list[int]]) -> list[list[int]]:
    return [
        [sum(1 for area in map_areas if area == area_id) for area_id in range(AREA_COUNT)]
        for map_areas in area_assignments
    ]


def main() -> int:
    payload = {
        "episode_length": 253,
        "recommended_candidates": 4,
        "shortlist_candidates": 16,
        "temperature": 0.03,
        "proposal_temperature": 0.1,
        "reward_door": 1.0,
        "reward_connection": 1.0,
        "reward_toilet": 1.0,
        "reward_phantoon": 1.0,
        "reward_balance": 0.05,
        "reward_toilet_balance": 0.05,
        "reward_frontier": 0.0,
        "reward_graph_diameter": 0.1,
        "reward_save_distance": 0.1,
        "reward_refill_distance": 0.1,
        "reward_missing_connect_utility": 0.5,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        print(error.read().decode("utf-8"))
        return 1
    # print(response_body)
    response_data = json.loads(response_body)
    # print(json.dumps(area_room_counts(response_data["area"])))
    print(json.dumps(response_data["area_crossings"]))
    print(response_data["avg_area_crossings"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
