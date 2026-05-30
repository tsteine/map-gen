from env import EnvironmentGroup, GenerateConfig
import torch


KNOWN_INVALID_REWARD = -100.0


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


def generate(env: EnvironmentGroup, model, config: GenerateConfig, device: torch.device):
    num_envs = env.num_envs
    engine = env.engine
    num_rooms = len(engine.rooms)

    kv_cache = model.get_initial_kv_cache(num_envs, device)
    env.clear()

    with torch.no_grad():
        for _ in range(config.episode_length):
            # Get candidate actions and their post-step known outcomes from environment.
            candidates, outcomes = env.get_candidates_with_outcomes(config.max_candidates, device)
            
            # Model inference to get predictions and updated key-value cache for next step
            preds, kv_cache_candidates = model.generate(candidates, kv_cache, config)
    
            if candidates.room_idx.shape[1] == 1:
                # Only one candidate, so select it directly (e.g. on the first step)
                action_index = torch.zeros(candidates.room_idx.shape[0], dtype=torch.int64, device=device)
                selected_actions = candidates.select(action_index)
            else:
                # Compute expected reward and sample to select an action (per environment)
                expected_reward = compute_expected_reward(preds, outcomes, config)
                dummy_candidate = candidates.room_idx == num_rooms
                has_real_candidate = torch.any(~dummy_candidate, dim=1, keepdim=True)
                expected_reward = torch.where(
                    dummy_candidate & has_real_candidate,
                    torch.full_like(expected_reward, float('-inf')),
                    expected_reward,
                )
                probs = torch.softmax(expected_reward / torch.unsqueeze(config.temperature, 1), dim=1)
                action_index = rand_choice(probs)
                selected_actions = candidates.select(action_index)
            
            # Apply the selected action to the environment
            env.step(selected_actions)
            
            # Finalize the kv cache update based on the selected action
            kv_cache = model.get_updated_kv_cache(kv_cache, kv_cache_candidates, action_index)
        
    env.finish()
    actions = env.get_actions(device)
    outcomes = env.get_outcomes(device)
    return actions, outcomes
