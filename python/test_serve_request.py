from pydantic import ValidationError

from serve import GenerateRequest, validate_generate_request


def base_payload() -> dict:
    return {
        "episode_length": 3,
        "recommended_candidates": 1,
        "shortlist_candidates": 1,
        "temperature": 1.0,
        "proposal_temperature": 1.0,
        "reward_door": 1.0,
        "reward_connection": 1.0,
        "reward_toilet": 1.0,
        "reward_phantoon": 1.0,
        "reward_balance": 1.0,
        "reward_toilet_balance": 1.0,
        "reward_frontier": 1.0,
        "reward_graph_diameter": 1.0,
        "reward_save_distance": 1.0,
        "reward_refill_distance": 1.0,
        "reward_missing_connect_utility": 1.0,
    }


def assert_invalid_value(payload: dict, expected_message: str) -> None:
    request = GenerateRequest.model_validate(payload)
    try:
        validate_generate_request(request, rooms=[{}, {}, {}])
    except ValueError as error:
        assert str(error) == expected_message
    else:
        raise AssertionError("expected ValueError")


def main() -> None:
    full_map_payload = base_payload() | {"small_map": False}
    full_map_request = GenerateRequest.model_validate(full_map_payload)
    validate_generate_request(full_map_request, rooms=[{}, {}, {}])

    try:
        GenerateRequest.model_validate(base_payload())
    except ValidationError:
        pass
    else:
        raise AssertionError("small_map should be required")

    assert_invalid_value(
        base_payload() | {"small_map": True},
        "small_map requires min_rooms, max_rooms, target_rooms",
    )
    assert_invalid_value(
        base_payload()
        | {
            "small_map": True,
            "min_rooms": 0,
            "max_rooms": 2,
            "target_rooms": 2,
        },
        "min_rooms must be greater than zero",
    )
    assert_invalid_value(
        base_payload()
        | {
            "small_map": True,
            "min_rooms": 3,
            "max_rooms": 2,
            "target_rooms": 2,
        },
        "max_rooms must be at least min_rooms",
    )
    assert_invalid_value(
        base_payload()
        | {
            "small_map": True,
            "min_rooms": 1,
            "max_rooms": 2,
            "target_rooms": 0,
        },
        "target_rooms must be greater than zero",
    )

    small_map_request = GenerateRequest.model_validate(
        base_payload()
        | {
            "small_map": True,
            "min_rooms": 1,
            "max_rooms": 2,
            "target_rooms": 2,
        }
    )
    validate_generate_request(small_map_request, rooms=[{}, {}, {}])


if __name__ == "__main__":
    main()
