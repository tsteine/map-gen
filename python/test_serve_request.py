import threading
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
        "max_candidate_areas_per_placement": 2,
        "temperature": 1.0,
        "frontier_temperature": 1.0,
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
        "reward_area_connected": 1.0,
        "reward_area_connected_excess": 1.0,
        "reward_area_crossing": 1.0,
        "reward_area_size_valid": 1.0,
        "reward_area_map_station": 1.0,
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
        "prefetch_queue_max_size": 2,
        "prefetch_max_queues": 2,
        "prefetch_delay_seconds": 1.0,
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
    original_generate_response_data_uncached_validated = (
        serve.generate_response_data_uncached_validated
    )
    calls = []
    state = SimpleNamespace(serving_config=SimpleNamespace(num_warmup_requests=3))

    def generate_response_data(_state, generate_request):
        raise AssertionError("warmup should not use prefetch-aware generation")

    def generate_response_data_uncached_validated(_state, generate_request):
        calls.append(generate_request)
        return {"stats": {"num_valid": 0}}

    try:
        serve.generate_response_data = generate_response_data
        serve.generate_response_data_uncached_validated = (
            generate_response_data_uncached_validated
        )
        run_warmup_requests(state)
    finally:
        serve.generate_response_data = original_generate_response_data
        serve.generate_response_data_uncached_validated = (
            original_generate_response_data_uncached_validated
        )

    assert len(calls) == 3
    assert all(isinstance(call, GenerateRequest) for call in calls)


def assert_warmup_request_failure_propagates() -> None:
    original_generate_response_data_uncached_validated = (
        serve.generate_response_data_uncached_validated
    )
    state = SimpleNamespace(serving_config=SimpleNamespace(num_warmup_requests=1))

    def generate_response_data_uncached_validated(_state, _generate_request):
        raise RuntimeError("warmup failed")

    try:
        serve.generate_response_data_uncached_validated = (
            generate_response_data_uncached_validated
        )
        try:
            run_warmup_requests(state)
        except RuntimeError as error:
            assert str(error) == "warmup failed"
        else:
            raise AssertionError("expected RuntimeError")
    finally:
        serve.generate_response_data_uncached_validated = (
            original_generate_response_data_uncached_validated
        )


def prefetch_test_state() -> SimpleNamespace:
    return SimpleNamespace(
        serving_config=ServingConfig.model_validate(base_serving_config_payload()),
        rooms=[{}, {}, {}],
        device=None,
        profile=False,
        lock=threading.Lock(),
        prefetch=serve.create_prefetch_state(),
    )


def assert_prefetch_disabled_uses_uncached_generation() -> None:
    original_generate = serve.generate_response_data_uncached_validated
    calls = []
    state = prefetch_test_state()
    state.prefetch = None
    request = GenerateRequest.model_validate(base_payload() | {"small_map": False})

    def generate_response_data_uncached_validated(_state, generate_request):
        calls.append(generate_request)
        return {"response": len(calls)}

    try:
        serve.generate_response_data_uncached_validated = generate_response_data_uncached_validated
        assert serve.generate_response_data(state, request) == {"response": 1}
    finally:
        serve.generate_response_data_uncached_validated = original_generate

    assert calls == [request]


def assert_prefetch_hit_and_miss_behavior() -> None:
    original_generate = serve.generate_response_data_uncached_validated
    calls = []
    state = prefetch_test_state()
    request = GenerateRequest.model_validate(base_payload() | {"small_map": False})
    key = serve.prefetch_request_key(request)

    def generate_response_data_uncached_validated(_state, generate_request):
        calls.append(generate_request)
        return {"response": len(calls)}

    try:
        serve.generate_response_data_uncached_validated = generate_response_data_uncached_validated
        assert serve.generate_response_data(state, request) == {"response": 1}
        with state.prefetch.condition:
            queue_state = state.prefetch.queues[key]
            assert queue_state.refill_debt == 2
            queue_state.responses.append(serve.serialize_generate_response({"response": "cached"}))
        cached_response = serve.generate_response_data(state, request)
        assert isinstance(cached_response, str)
        assert serve.app.json.loads(cached_response) == {"response": "cached"}
        with state.prefetch.condition:
            assert state.prefetch.queues[key].refill_debt == 4
    finally:
        serve.generate_response_data_uncached_validated = original_generate

    assert calls == [request]


def assert_generate_route_returns_serialized_prefetch_response() -> None:
    original_generate = serve.generate_response_data
    original_state = serve.SERVING_STATE
    state = prefetch_test_state()

    def generate_response_data(_state, _generate_request):
        return serve.serialize_generate_response({"response": "cached"})

    try:
        serve.SERVING_STATE = state
        serve.generate_response_data = generate_response_data
        response = serve.app.test_client().post(
            "/generate",
            json=base_payload() | {"small_map": False},
        )
    finally:
        serve.generate_response_data = original_generate
        serve.SERVING_STATE = original_state

    assert response.status_code == 200
    assert response.mimetype == "application/json"
    assert response.get_json() == {"response": "cached"}


def assert_prefetch_refill_drains_until_full() -> None:
    original_generate = serve.generate_prefetch_response_with_generation_lock_held
    state = prefetch_test_state()
    request = GenerateRequest.model_validate(base_payload() | {"small_map": False})
    key = serve.prefetch_request_key(request)
    calls = []

    def generate_prefetch_response_with_generation_lock_held(_state, generate_request):
        calls.append(generate_request)
        return {"response": len(calls)}

    try:
        serve.generate_prefetch_response_with_generation_lock_held = (
            generate_prefetch_response_with_generation_lock_held
        )
        with state.prefetch.condition:
            queue_state = serve.touch_prefetch_queue(state, key, request)
            queue_state.refill_debt = 5
            state.prefetch.refill_scheduled = True
            state.prefetch.schedule_version = 1
        serve.run_prefetch_refill_pass(state, 1)
    finally:
        serve.generate_prefetch_response_with_generation_lock_held = original_generate

    with state.prefetch.condition:
        queue_state = state.prefetch.queues[key]
        assert [
            serve.app.json.loads(response)
            for response in queue_state.responses
        ] == [{"response": 1}, {"response": 2}]
        assert queue_state.refill_debt == 0
        assert not state.prefetch.refill_scheduled
    assert calls == [request, request]


def assert_prefetch_refill_skips_full_queue() -> None:
    original_generate = serve.generate_prefetch_response_with_generation_lock_held
    state = prefetch_test_state()
    request = GenerateRequest.model_validate(base_payload() | {"small_map": False})
    key = serve.prefetch_request_key(request)
    calls = []

    def generate_prefetch_response_with_generation_lock_held(_state, generate_request):
        calls.append(generate_request)
        return {"response": len(calls)}

    try:
        serve.generate_prefetch_response_with_generation_lock_held = (
            generate_prefetch_response_with_generation_lock_held
        )
        with state.prefetch.condition:
            queue_state = serve.touch_prefetch_queue(state, key, request)
            queue_state.responses.extend(
                [
                    serve.serialize_generate_response({"response": 1}),
                    serve.serialize_generate_response({"response": 2}),
                ]
            )
            queue_state.refill_debt = 2
            state.prefetch.refill_scheduled = True
            state.prefetch.schedule_version = 1
        serve.run_prefetch_refill_pass(state, 1)
    finally:
        serve.generate_prefetch_response_with_generation_lock_held = original_generate

    with state.prefetch.condition:
        assert state.prefetch.queues[key].refill_debt == 0
    assert calls == []


def assert_prefetch_lru_eviction() -> None:
    state = prefetch_test_state()
    request_one = GenerateRequest.model_validate(base_payload() | {"small_map": False})
    request_two = GenerateRequest.model_validate(
        base_payload() | {"small_map": False, "temperature": 2.0}
    )
    request_three = GenerateRequest.model_validate(
        base_payload() | {"small_map": False, "temperature": 3.0}
    )
    key_one = serve.prefetch_request_key(request_one)
    key_two = serve.prefetch_request_key(request_two)
    key_three = serve.prefetch_request_key(request_three)

    serve.schedule_prefetch_refill(state, key_one, request_one)
    serve.schedule_prefetch_refill(state, key_two, request_two)
    with state.prefetch.condition:
        serve.touch_prefetch_queue(state, key_one, request_one)
    serve.schedule_prefetch_refill(state, key_three, request_three)

    with state.prefetch.condition:
        assert list(state.prefetch.queues) == [key_one, key_three]
        assert key_two not in state.prefetch.queues


def assert_prefetch_refill_yields_to_foreground() -> None:
    original_generate = serve.generate_prefetch_response_with_generation_lock_held
    state = prefetch_test_state()
    request = GenerateRequest.model_validate(base_payload() | {"small_map": False})
    key = serve.prefetch_request_key(request)
    calls = []

    def generate_prefetch_response_with_generation_lock_held(_state, generate_request):
        calls.append(generate_request)
        return {"response": len(calls)}

    try:
        serve.generate_prefetch_response_with_generation_lock_held = (
            generate_prefetch_response_with_generation_lock_held
        )
        with state.prefetch.condition:
            queue_state = serve.touch_prefetch_queue(state, key, request)
            queue_state.refill_debt = 2
            state.prefetch.foreground_waiting = 1
            state.prefetch.refill_scheduled = True
            state.prefetch.schedule_version = 1
        serve.run_prefetch_refill_pass(state, 1)
    finally:
        serve.generate_prefetch_response_with_generation_lock_held = original_generate

    with state.prefetch.condition:
        assert state.prefetch.queues[key].refill_debt == 2
        assert state.prefetch.refill_scheduled
        assert state.prefetch.schedule_version == 2
    assert calls == []


def main() -> None:
    full_map_payload = base_payload() | {"small_map": False}
    full_map_request = GenerateRequest.model_validate(full_map_payload)
    validate_generate_request(full_map_request, rooms=[{}, {}, {}])
    assert_invalid_value(
        base_payload() | {"small_map": False, "max_candidate_areas_per_placement": 0},
        "max_candidate_areas_per_placement must be greater than zero",
    )
    assert_invalid_value(
        base_payload() | {"small_map": False, "max_candidate_areas_per_placement": 7},
        "max_candidate_areas_per_placement must be at most AREA_COUNT",
    )

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
    for field in (
        "prefetch_queue_max_size",
        "prefetch_max_queues",
        "prefetch_delay_seconds",
    ):
        try:
            ServingConfig.model_validate(
                {
                    key: value
                    for key, value in base_serving_config_payload().items()
                    if key != field
                }
            )
        except ValidationError:
            pass
        else:
            raise AssertionError(f"{field} should be required")

    zero_warmup_config = ServingConfig.model_validate(base_serving_config_payload())
    validate_serving_config(zero_warmup_config)
    assert_invalid_serving_config(
        base_serving_config_payload() | {"num_warmup_requests": -1},
        "num_warmup_requests must be greater than or equal to zero",
    )
    assert_invalid_serving_config(
        base_serving_config_payload() | {"prefetch_queue_max_size": -1},
        "prefetch_queue_max_size must be greater than or equal to zero",
    )
    assert_invalid_serving_config(
        base_serving_config_payload() | {"prefetch_max_queues": -1},
        "prefetch_max_queues must be greater than or equal to zero",
    )
    assert_invalid_serving_config(
        base_serving_config_payload() | {"prefetch_delay_seconds": -1.0},
        "prefetch_delay_seconds must be greater than or equal to zero",
    )

    warmup_request = warmup_generate_request()
    assert warmup_request.episode_length == 253
    assert warmup_request.recommended_candidates == 4
    assert warmup_request.shortlist_candidates == 16
    assert warmup_request.max_candidate_areas_per_placement == 2
    assert warmup_request.temperature == 0.03
    assert warmup_request.frontier_temperature == 0.3
    assert warmup_request.proposal_temperature == 0.3
    assert warmup_request.reward_balance == 0.1
    assert warmup_request.reward_toilet_balance == 0.1
    assert warmup_request.reward_missing_connect_utility == 0.5
    assert warmup_request.area_assignment_base_order == "random"
    assert not warmup_request.small_map
    validate_generate_request(warmup_request, rooms=[{} for _ in range(253)])
    assert_warmup_requests_run()
    assert_warmup_request_failure_propagates()
    assert_prefetch_disabled_uses_uncached_generation()
    assert_prefetch_hit_and_miss_behavior()
    assert_generate_route_returns_serialized_prefetch_response()
    assert_prefetch_refill_drains_until_full()
    assert_prefetch_refill_skips_full_queue()
    assert_prefetch_lru_eviction()
    assert_prefetch_refill_yields_to_foreground()


if __name__ == "__main__":
    main()
