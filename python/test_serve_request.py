from types import SimpleNamespace

from pydantic import ValidationError

import serve
from serve import (
    GenerateRequest,
    ServingConfig,
    run_warmup_requests,
    validate_generate_request,
    validate_serving_config,
    warmup_generate_request,
)


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
        "area_assignment_base_order": "random",
    }


def base_serving_config_payload() -> dict:
    return {
        "host": "127.0.0.1",
        "port": 5000,
        "device": "cpu",
        "compile_model": False,
        "cuda_memory_fraction": 1.0,
        "model_dtype": "float32",
        "autocast": False,
        "verify_outcome_consistency": False,
        "gpu_prefetch_batches": 1,
        "num_warmup_requests": 0,
        "area_assignment_attempts": 1,
        "area_bounding_box_width": 1,
        "area_bounding_box_height": 1,
        "area_min_rooms": 1,
        "area_max_rooms": 1,
        "room_set": "room_definitions/zebes.json",
        "num_environments": 1,
        "pipeline_groups": 1,
        "num_threads": 1,
    }


def assert_invalid_value(payload: dict, expected_message: str) -> None:
    request = GenerateRequest.model_validate(payload)
    try:
        validate_generate_request(request, rooms=[{}, {}, {}])
    except ValueError as error:
        assert str(error) == expected_message
    else:
        raise AssertionError("expected ValueError")


def assert_invalid_serving_config(payload: dict, expected_message: str) -> None:
    serving_config = ServingConfig.model_validate(payload)
    try:
        validate_serving_config(serving_config)
    except ValueError as error:
        assert str(error) == expected_message
    else:
        raise AssertionError("expected ValueError")


def assert_warmup_requests_run() -> None:
    original_generate_response_data = serve.generate_response_data
    calls = []
    state = SimpleNamespace(serving_config=SimpleNamespace(num_warmup_requests=3))

    def generate_response_data(_state, generate_request):
        calls.append(generate_request)
        return {"stats": {"num_valid": 0}}

    try:
        serve.generate_response_data = generate_response_data
        run_warmup_requests(state)
    finally:
        serve.generate_response_data = original_generate_response_data

    assert len(calls) == 3
    assert all(isinstance(call, GenerateRequest) for call in calls)


def assert_warmup_request_failure_propagates() -> None:
    original_generate_response_data = serve.generate_response_data
    state = SimpleNamespace(serving_config=SimpleNamespace(num_warmup_requests=1))

    def generate_response_data(_state, _generate_request):
        raise RuntimeError("warmup failed")

    try:
        serve.generate_response_data = generate_response_data
        try:
            run_warmup_requests(state)
        except RuntimeError as error:
            assert str(error) == "warmup failed"
        else:
            raise AssertionError("expected RuntimeError")
    finally:
        serve.generate_response_data = original_generate_response_data


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

    missing_area_order_payload = base_payload() | {"small_map": False}
    del missing_area_order_payload["area_assignment_base_order"]
    try:
        GenerateRequest.model_validate(missing_area_order_payload)
    except ValidationError:
        pass
    else:
        raise AssertionError("area_assignment_base_order should be required")

    try:
        GenerateRequest.model_validate(
            base_payload()
            | {
                "area_assignment_base_order": "invalid",
                "small_map": False,
            }
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("area_assignment_base_order should reject invalid values")

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

    try:
        ServingConfig.model_validate(
            {
                key: value
                for key, value in base_serving_config_payload().items()
                if key != "num_warmup_requests"
            }
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("num_warmup_requests should be required")

    zero_warmup_config = ServingConfig.model_validate(base_serving_config_payload())
    validate_serving_config(zero_warmup_config)
    assert_invalid_serving_config(
        base_serving_config_payload() | {"num_warmup_requests": -1},
        "num_warmup_requests must be greater than or equal to zero",
    )

    warmup_request = warmup_generate_request()
    assert warmup_request.episode_length == 253
    assert warmup_request.recommended_candidates == 4
    assert warmup_request.shortlist_candidates == 16
    assert warmup_request.temperature == 0.03
    assert warmup_request.proposal_temperature == 0.3
    assert warmup_request.reward_balance == 0.1
    assert warmup_request.reward_toilet_balance == 0.1
    assert warmup_request.reward_missing_connect_utility == 0.5
    assert warmup_request.area_assignment_base_order == "random"
    assert not warmup_request.small_map
    validate_generate_request(warmup_request, rooms=[{} for _ in range(253)])
    assert_warmup_requests_run()
    assert_warmup_request_failure_propagates()


if __name__ == "__main__":
    main()
