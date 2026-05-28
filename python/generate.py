from dataclasses import dataclass
from map_gen import EnvironmentGroup
import torch



@dataclass
class GenerationConfig:
    episode_length: int
    max_candidates: int
    temperature: torch.Tensor


def rand_choice(p):
    cumul_p = torch.cumsum(p, dim=1)
    rnd = torch.rand([p.shape[0], 1], device=p.device)
    choice = torch.clamp(torch.searchsorted(cumul_p, rnd), max=p.shape[1] - 1).view(-1)
    return choice

# preds.door_invalid: [batch_size, max_candidates, num_outputs]
# preds.connection_invalid: [batch_size, max_candidates, num_outputs]
def compute_expected_reward(preds, config: GenerationConfig):
    door_logprobs = torch.logaddexp(preds.door_invalid, torch.zeros_like(preds.door_invalid))
    connection_logprobs = torch.logaddexp(preds.connection_invalid, torch.zeros_like(preds.connection_invalid))
    total_logprobs = torch.sum(door_logprobs, dim=2) + torch.sum(connection_logprobs, dim=2)
    return total_logprobs


def generate(env: EnvironmentGroup, model, config: GenerationConfig, device):
    num_envs = env.num_environments()
    engine = env.get_engine()
    num_rooms = engine.num_rooms()

    kv_cache = model.get_initial_kv_cache(num_envs, device)
    env.clear()
    env.initial_step()
    
    for _ in range(config.episode_length - 1):
        # Get candidate actions from environment, and load them to device (e.g. GPU)
        cand_room_idx, cand_x, cand_y = env.get_candidates(config.max_candidates)
        cand_room_idx = torch.from_numpy(cand_room_idx).to(device)
        cand_x = torch.from_numpy(cand_x).to(device)
        cand_y = torch.from_numpy(cand_y).to(device)
        
        # Model inference to get predictions and updated key-value cache for next step
        preds, kv_cache_candidates = model.generate(cand_room_idx, cand_x, cand_y, kv_cache, config)

        # Compute expected reward and sample to select an action (per environment)
        expected_reward = compute_expected_reward(preds, config)
        expected_reward = torch.where(cand_room_idx == num_rooms, # dummy action should only be selected if no other choice
                                      torch.full_like(expected_reward, float('-inf')),
                                      expected_reward)
        probs = torch.softmax(expected_reward / torch.unsqueeze(config.temperature, 1), dim=1)
        action_index = rand_choice(probs)
        selected_cand_room_idx = torch.gather(cand_room_idx, 1, action_index.unsqueeze(1)).squeeze(1)
        selected_cand_x = torch.gather(cand_x, 1, action_index.unsqueeze(1)).squeeze(1)
        selected_cand_y = torch.gather(cand_y, 1, action_index.unsqueeze(1)).squeeze(1)
        
        # Apply the selected action to the environment
        env.step(selected_cand_room_idx.cpu().numpy(), selected_cand_x.cpu().numpy(), selected_cand_y.cpu().numpy())
        
        # Finalize the kv cache update based on the selected action
        kv_cache = model.get_updated_kv_cache(kv_cache, kv_cache_candidates, action_index)
        
    env.finish()
    actions = env.get_actions()
    outcomes = env.get_outcomes()
    return actions, outcomes
