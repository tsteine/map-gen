# Python wrappers around the Rust map generation engine, includes (zero-copy) conversions
# between numpy and torch tensors.
from dataclasses import dataclass
from typing import Optional

import torch
import json

import map_gen

@dataclass
class GenerateConfig:
    episode_length: int
    max_candidates: int
    temperature: torch.Tensor
    lookahead_outcomes: bool = True
    state_candidate_chunk: int = 1
    state_environment_chunk: int = 8
    state_autocast: bool = True


# Each tensor here is uint8 with shape
#    [batch, time]  during training,
#    [batch, candidate]  during generation
@dataclass
class Actions:
    room_idx: torch.Tensor
    room_x: torch.Tensor
    room_y: torch.Tensor

    def select(self, index: torch.Tensor) -> "Actions":
        selected_room_idx = torch.gather(self.room_idx, 1, index.unsqueeze(1)).squeeze(1)
        selected_room_x = torch.gather(self.room_x, 1, index.unsqueeze(1)).squeeze(1)
        selected_room_y = torch.gather(self.room_y, 1, index.unsqueeze(1)).squeeze(1)
        return Actions(selected_room_idx, selected_room_x, selected_room_y)

    def to(self, device: torch.device) -> "Actions":
        return Actions(self.room_idx.to(device), self.room_x.to(device), self.room_y.to(device))


# Each tensor here is int8 with shape
#    [batch, time, output]  during training,
#    [batch, candidate, output]  during generation
@dataclass
class Outcomes:
    # -1 = unknown, 0 = valid (door is connected), 1 = invalid (door is not connected)
    door_invalid: torch.Tensor
    # -1 = unknown, 0 = valid (connection has return path), 1 = invalid (connection does not have return path)
    connection_invalid: torch.Tensor


@dataclass
class StateFeatures:
    inventory: torch.Tensor
    room_x: torch.Tensor
    room_y: torch.Tensor
    room_placed: torch.Tensor
    frontier: torch.Tensor
    frontier_neighbor: torch.Tensor
    frontier_neighbor_pair: torch.Tensor
    frontier_obstruction: torch.Tensor

    def to(self, device: torch.device) -> "StateFeatures":
        return StateFeatures(*(value.to(device) for value in vars(self).values()))

    def flatten_candidates(self) -> "StateFeatures":
        return StateFeatures(*(value.flatten(0, 1) for value in vars(self).values()))

    def slice(self, start: int, end: int) -> "StateFeatures":
        return StateFeatures(*(value[start:end] for value in vars(self).values()))

    def compact_frontiers(self) -> "StateFeatures":
        frontier_count = int(torch.sum(self.frontier[:, :, 0] != 0, dim=1).max().item())
        frontier_bucket = next(
            (bucket for bucket in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 24, 32, 40) if bucket >= frontier_count),
            self.frontier.shape[1],
        )
        frontier_bucket = min(frontier_bucket, self.frontier.shape[1])
        return StateFeatures(
            inventory=self.inventory,
            room_x=self.room_x,
            room_y=self.room_y,
            room_placed=self.room_placed,
            frontier=self.frontier[:, :frontier_bucket],
            frontier_neighbor=self.frontier_neighbor[:, :frontier_bucket],
            frontier_neighbor_pair=self.frontier_neighbor_pair[:, :frontier_bucket],
            frontier_obstruction=self.frontier_obstruction[:, :frontier_bucket],
        )


@dataclass
class OutputMetadata:
    door: list[tuple[int, int]]
    connection: list[tuple[int, int]]
    num_door_variants: int
    num_connection_variants: int
    room_connection_variant_idx: list[int]
    num_room_connection_variants: int

    def get_output_sizes(self) -> tuple[int, int]:
        return len(self.door), len(self.connection)


class Engine:
    engine: map_gen.Engine
    rooms: list[dict]

    def __init__(self, rooms: list[dict]):
        self.engine = map_gen.Engine(json.dumps(rooms))
        self.rooms = rooms

    def create_environment_group(self, map_size: tuple[int, int], num_envs: int, seed: Optional[int] = None) -> "EnvironmentGroup":
        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, ()).item())
        env = self.engine.create_environment_group(map_size, num_envs, seed)
        return EnvironmentGroup(self, env, map_size, num_envs)

    def get_output_sizes(self) -> tuple[int, int]:
        return self.engine.get_output_sizes()

    def get_output_metadata(self) -> OutputMetadata:
        (
            door,
            connection,
            num_door_variants,
            num_connection_variants,
            room_connection_variant_idx,
            num_room_connection_variants,
        ) = (
            self.engine.get_output_metadata()
        )
        return OutputMetadata(
            door=door,
            connection=connection,
            num_door_variants=num_door_variants,
            num_connection_variants=num_connection_variants,
            room_connection_variant_idx=room_connection_variant_idx,
            num_room_connection_variants=num_room_connection_variants,
        )

    def get_state_feature_sizes(self) -> tuple[int, int, int]:
        return self.engine.get_state_feature_sizes()


class EnvironmentGroup:
    engine: Engine
    env: map_gen.EnvironmentGroup
    map_size: tuple[int, int]
    num_envs: int

    def __init__(self, engine: Engine, env: map_gen.EnvironmentGroup, map_size: tuple[int, int], num_envs: int):
        self.engine = engine
        self.env = env
        self.map_size = map_size
        self.num_envs = num_envs

    def clear(self):
        self.env.clear()

    def get_initial_action(self, device: torch.device) -> Actions:
        room_idx, room_x, room_y = self.env.get_initial_action()
        return Actions(
            room_idx=torch.from_numpy(room_idx).to(device),
            room_x=torch.from_numpy(room_x).to(device),
            room_y=torch.from_numpy(room_y),
        )

    def step(self, actions: Actions):
        self.env.step(
            actions.room_idx.cpu().numpy(),
            actions.room_x.cpu().numpy(),
            actions.room_y.cpu().numpy(),
        )

    def replay(self, actions: Actions):
        self.env.replay(
            actions.room_idx.contiguous().cpu().numpy(),
            actions.room_x.contiguous().cpu().numpy(),
            actions.room_y.contiguous().cpu().numpy(),
        )
        
    def get_actions(self, device: torch.device) -> Actions:
        room_idx, room_x, room_y = self.env.get_actions()
        return Actions(
            room_idx=torch.from_numpy(room_idx).to(device),
            room_x=torch.from_numpy(room_x).to(device),
            room_y=torch.from_numpy(room_y).to(device),
        )

    def get_candidates(self, max_candidates: int, device: torch.device) -> Actions:
        room_idx, room_x, room_y = self.env.get_candidates(max_candidates)
        return Actions(
            room_idx=torch.from_numpy(room_idx).to(device),
            room_x=torch.from_numpy(room_x).to(device),
            room_y=torch.from_numpy(room_y).to(device),
        )

    def get_candidates_with_outcomes(
        self, max_candidates: int, device: torch.device
    ) -> tuple[Actions, Outcomes]:
        room_idx, room_x, room_y, door_invalid, connection_invalid = (
            self.env.get_candidates_with_outcomes(max_candidates)
        )
        return (
            Actions(
                room_idx=torch.from_numpy(room_idx).to(device),
                room_x=torch.from_numpy(room_x).to(device),
                room_y=torch.from_numpy(room_y).to(device),
            ),
            Outcomes(
                door_invalid=torch.from_numpy(door_invalid).to(device),
                connection_invalid=torch.from_numpy(connection_invalid).to(device),
            ),
        )

    def get_outcomes(self, device: torch.device) -> Outcomes:
        door_invalid, connection_invalid = self.env.get_outcomes()
        return Outcomes(
            door_invalid=torch.from_numpy(door_invalid).to(device),
            connection_invalid=torch.from_numpy(connection_invalid).to(device),
        )

    @staticmethod
    def _state_features(values, device: torch.device) -> StateFeatures:
        return StateFeatures(*(torch.from_numpy(value).to(device) for value in values))

    def get_state_features(self, device: torch.device) -> StateFeatures:
        return self._state_features(self.env.get_state_features(), device)

    def get_state_features_after_candidates(
        self, actions: Actions, device: torch.device, environment_start: int = 0
    ) -> StateFeatures:
        values = self.env.get_state_features_after_candidates(
            actions.room_idx.contiguous().cpu().numpy(),
            actions.room_x.contiguous().cpu().numpy(),
            actions.room_y.contiguous().cpu().numpy(),
            environment_start,
        )
        return self._state_features(values, device)

    def finish(self):
        self.env.finish()
