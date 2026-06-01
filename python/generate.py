from env import Actions, EnvironmentGroup, GenerateConfig, Outcomes
from model import Predictions
from profile_stats import ProfileStats
from concurrent.futures import ThreadPoolExecutor
import torch


KNOWN_INVALID_REWARD = -100.0


class Prefetcher:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)

    def close(self):
        self.executor.shutdown()

    def map(self, items, prepare, profiler=None, wait_name=None):
        items = iter(items)
        try:
            future = self.executor.submit(prepare, next(items))
        except StopIteration:
            return
        for item in items:
            if profiler is None or wait_name is None:
                result = future.result()
            else:
                with profiler.timer(wait_name):
                    result = future.result()
            future = self.executor.submit(prepare, item)
            yield result
        if profiler is None or wait_name is None:
            yield future.result()
        else:
            with profiler.timer(wait_name):
                yield future.result()


def rand_choice(p):
    cumul_p = torch.cumsum(p, dim=1)
    rnd = torch.rand([p.shape[0], 1], device=p.device)
    choice = torch.clamp(torch.searchsorted(cumul_p, rnd), max=p.shape[1] - 1).view(-1)
    return choice


def outcome_reward(model_logprobs: torch.Tensor, known_invalid: torch.Tensor) -> torch.Tensor:
    if known_invalid.ndim == model_logprobs.ndim - 1:
        known_invalid = known_invalid.unsqueeze(1)
    known_valid_reward = torch.zeros_like(model_logprobs)
    known_invalid_reward = torch.full_like(model_logprobs, KNOWN_INVALID_REWARD)
    known_reward = torch.where(known_invalid == 0, known_valid_reward, known_invalid_reward)
    return torch.where(known_invalid < 0, model_logprobs, known_reward)


# preds.door_invalid: [batch_size, max_candidates, num_outputs]
# preds.connection_invalid: [batch_size, max_candidates, num_outputs]
def compute_expected_reward(preds, outcomes, config: GenerateConfig):
    door_logprobs = torch.nn.functional.logsigmoid(-preds.door_invalid)
    connection_logprobs = torch.nn.functional.logsigmoid(-preds.connection_invalid)
    door_logprobs = outcome_reward(door_logprobs, outcomes.door_invalid)
    connection_logprobs = outcome_reward(connection_logprobs, outcomes.connection_invalid)
    return torch.sum(door_logprobs, dim=2) + torch.sum(connection_logprobs, dim=2)


def select_outcomes(outcomes: Outcomes, index: torch.Tensor) -> Outcomes:
    gather_index = index.view(-1, 1, 1)
    return Outcomes(
        door_invalid=torch.gather(
            outcomes.door_invalid, 1, gather_index.expand(-1, 1, outcomes.door_invalid.shape[2])
        ).squeeze(1),
        connection_invalid=torch.gather(
            outcomes.connection_invalid,
            1,
            gather_index.expand(-1, 1, outcomes.connection_invalid.shape[2]),
        ).squeeze(1),
    )


def merge_verified_outcomes(
    known_outcomes: Outcomes | None,
    current_outcomes: Outcomes,
    stage: str,
) -> Outcomes:
    if known_outcomes is None:
        return current_outcomes

    def merge_known(known: torch.Tensor, current: torch.Tensor, outcome_name: str):
        inconsistent = (known >= 0) & (current >= 0) & (known != current)
        if torch.any(inconsistent):
            first_idx = torch.nonzero(inconsistent, as_tuple=False)[0].tolist()
            invalid_to_valid = torch.sum((known == 1) & (current == 0)).item()
            valid_to_invalid = torch.sum((known == 0) & (current == 1)).item()
            raise RuntimeError(
                f"{outcome_name} outcome changed after becoming known at {stage}: "
                f"first index {first_idx}, invalid->valid {invalid_to_valid}, "
                f"valid->invalid {valid_to_invalid}"
            )
        return torch.where(known >= 0, known, current)

    return Outcomes(
        door_invalid=merge_known(
            known_outcomes.door_invalid, current_outcomes.door_invalid, "door"
        ),
        connection_invalid=merge_known(
            known_outcomes.connection_invalid,
            current_outcomes.connection_invalid,
            "connection",
        ),
    )


def generate(
    env: EnvironmentGroup,
    model,
    config: GenerateConfig,
    device: torch.device,
    verify_outcome_consistency: bool = False,
    profiler: ProfileStats | None = None,
):
    num_envs = env.num_envs
    engine = env.engine
    num_rooms = len(engine.rooms)

    uses_state_features = getattr(model, "uses_state_features", False)
    kv_cache = None if uses_state_features else model.get_initial_kv_cache(num_envs, device)
    env.clear()
    known_outcomes = None
    profiler = profiler or ProfileStats(False)

    prefetcher = Prefetcher()
    try:
        with torch.no_grad():
            for step in range(config.episode_length):
                if config.lookahead_outcomes:
                    # Get candidate actions and their post-step known outcomes from environment.
                    with profiler.timer("gen.cpu_candidates"):
                        candidates, outcomes = env.get_candidates_with_outcomes(config.max_candidates, device)
                else:
                    # Use current known outcomes for all candidates.
                    with profiler.timer("gen.cpu_candidates"):
                        candidates = env.get_candidates(config.max_candidates, device)
                        outcomes = env.get_outcomes(device)

                if candidates.room_idx.shape[1] == 1:
                    # Only one candidate, so select it directly (e.g. on the first step)
                    if not uses_state_features:
                        _, kv_cache_candidates = model.generate(candidates, kv_cache, config)
                    action_index = torch.zeros(candidates.room_idx.shape[0], dtype=torch.int64, device=device)
                    selected_actions = candidates.select(action_index)
                else:
                    if uses_state_features:
                        candidate_rewards = []
                        for start in range(0, candidates.room_idx.shape[1], config.state_candidate_chunk):
                            end = start + config.state_candidate_chunk
                            chunk = Actions(
                                candidates.room_idx[:, start:end],
                                candidates.room_x[:, start:end],
                                candidates.room_y[:, start:end],
                            )
                            env_rewards = []
                            env_starts = range(0, num_envs, config.state_environment_chunk)

                            def prepare_features(env_start):
                                env_end = min(env_start + config.state_environment_chunk, num_envs)
                                env_chunk = Actions(
                                    chunk.room_idx[env_start:env_end],
                                    chunk.room_x[env_start:env_end],
                                    chunk.room_y[env_start:env_end],
                                )
                                with profiler.timer("gen.cpu_extract"):
                                    features = env.get_state_features_after_candidates(
                                        env_chunk, torch.device("cpu"), env_start
                                    ).flatten_candidates()
                                if profiler.enabled:
                                    (
                                        worker_seconds,
                                        pack_seconds,
                                        profile_calls,
                                        occupancy_prefix_cpu_seconds,
                                        clone_cpu_seconds,
                                        step_cpu_seconds,
                                        assemble_cpu_seconds,
                                        assemble_setup_cpu_seconds,
                                        assemble_frontier_cpu_seconds,
                                        assemble_neighbor_cpu_seconds,
                                        assemble_pair_cpu_seconds,
                                        assemble_pair_flags_cpu_seconds,
                                        assemble_pair_obstruction_cpu_seconds,
                                        assemble_pair_obstruction_base_cpu_seconds,
                                        assemble_pair_obstruction_candidate_cpu_seconds,
                                        assemble_output_cpu_seconds,
                                    ) = env.take_state_feature_profile()
                                    if profile_calls != 1:
                                        raise RuntimeError(
                                            "unexpected state feature profile call count: "
                                            f"{profile_calls}"
                                        )
                                    profiler.add("gen.cpu_extract_worker", worker_seconds)
                                    profiler.add("gen.cpu_extract_pack", pack_seconds)
                                    profiler.add("gen.cpu_extract_occupancy_prefix_sum", occupancy_prefix_cpu_seconds)
                                    profiler.add("gen.cpu_extract_clone_sum", clone_cpu_seconds)
                                    profiler.add("gen.cpu_extract_step_sum", step_cpu_seconds)
                                    profiler.add("gen.cpu_extract_assemble_sum", assemble_cpu_seconds)
                                    profiler.add("gen.cpu_extract_assemble_setup_sum", assemble_setup_cpu_seconds)
                                    profiler.add("gen.cpu_extract_assemble_frontier_sum", assemble_frontier_cpu_seconds)
                                    profiler.add("gen.cpu_extract_assemble_neighbor_sum", assemble_neighbor_cpu_seconds)
                                    profiler.add("gen.cpu_extract_assemble_pair_sum", assemble_pair_cpu_seconds)
                                    profiler.add("gen.cpu_extract_assemble_pair_flags_sum", assemble_pair_flags_cpu_seconds)
                                    profiler.add("gen.cpu_extract_assemble_pair_obstruction_sum", assemble_pair_obstruction_cpu_seconds)
                                    profiler.add("gen.cpu_extract_assemble_pair_obstruction_base_sum", assemble_pair_obstruction_base_cpu_seconds)
                                    profiler.add("gen.cpu_extract_assemble_pair_obstruction_candidate_sum", assemble_pair_obstruction_candidate_cpu_seconds)
                                    profiler.add("gen.cpu_extract_assemble_output_sum", assemble_output_cpu_seconds)
                                return env_start, env_end, features

                            for env_start, env_end, features in prefetcher.map(
                                env_starts, prepare_features, profiler, "gen.cpu_extract_wait"
                            ):
                                with profiler.timer("gen.cpu_transfer_submit"):
                                    env_features = features.to(device)
                                candidate_count = chunk.room_idx.shape[1]
                                chunk_outcomes = Outcomes(
                                    outcomes.door_invalid[env_start:env_end, start:end]
                                    if outcomes.door_invalid.ndim == 3 else outcomes.door_invalid[env_start:env_end],
                                    outcomes.connection_invalid[env_start:env_end, start:end]
                                    if outcomes.connection_invalid.ndim == 3 else outcomes.connection_invalid[env_start:env_end],
                                )
                                with profiler.cuda_timer("gen.gpu_model_reward", device):
                                    with torch.amp.autocast(
                                        "cuda",
                                        enabled=device.type == "cuda" and config.state_autocast,
                                    ):
                                        chunk_preds = model(env_features)
                                    env_rewards.append(compute_expected_reward(
                                        Predictions(
                                            chunk_preds.door_invalid.view(env_end - env_start, candidate_count, -1),
                                            chunk_preds.connection_invalid.view(env_end - env_start, candidate_count, -1),
                                        ),
                                        chunk_outcomes,
                                        config,
                                    ))
                            candidate_rewards.append(torch.cat(env_rewards, dim=0))
                        expected_reward = torch.cat(candidate_rewards, dim=1)
                        kv_cache_candidates = None
                    else:
                        # Model inference to get predictions and updated key-value cache for next step
                        preds, kv_cache_candidates = model.generate(candidates, kv_cache, config)
                        expected_reward = compute_expected_reward(preds, outcomes, config)
                    # Compute expected reward and sample to select an action (per environment)
                    dummy_candidate = candidates.room_idx == num_rooms
                    expected_reward = torch.where(
                        dummy_candidate,
                        torch.full_like(expected_reward, float('-inf')),
                        expected_reward,
                    )
                    with profiler.cuda_timer("gen.gpu_select", device):
                        probs = torch.softmax(expected_reward / torch.unsqueeze(config.temperature, 1), dim=1)
                        action_index = rand_choice(probs)
                        selected_actions = candidates.select(action_index)

                if verify_outcome_consistency and config.lookahead_outcomes:
                    known_outcomes = merge_verified_outcomes(
                        known_outcomes,
                        select_outcomes(outcomes, action_index),
                        f"lookahead step {step}",
                    )

                # Apply the selected action to the environment
                with profiler.timer("gen.cpu_step"):
                    env.step(selected_actions)

                if verify_outcome_consistency:
                    known_outcomes = merge_verified_outcomes(
                        known_outcomes,
                        env.get_outcomes(device),
                        f"step {step}",
                    )

                # Finalize the kv cache update based on the selected action
                if not uses_state_features and kv_cache_candidates is not None:
                    kv_cache = model.get_updated_kv_cache(kv_cache, kv_cache_candidates, action_index)
    finally:
        prefetcher.close()
        
    with profiler.timer("gen.cpu_finish"):
        env.finish()
        actions = env.get_actions(device)
        outcomes = env.get_outcomes(device)
    if verify_outcome_consistency:
        merge_verified_outcomes(known_outcomes, outcomes, "finish")
    return actions, outcomes
