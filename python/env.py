from __future__ import annotations

# Python wrappers around the Rust map generation engine, includes (zero-copy) conversions
# between numpy and torch tensors.
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

import torch
import json

import map_gen

if TYPE_CHECKING:
    from train_config import StateFeatureConfig

@dataclass
class GenerateConfig:
    episode_length: int
    max_candidates: int
    temperature: torch.Tensor
    lookahead_outcomes: bool
    state_feature_batch_size: int
    state_autocast: bool


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

    def to(self, device: torch.device) -> "Outcomes":
        return Outcomes(
            self.door_invalid.to(device),
            self.connection_invalid.to(device),
        )


@dataclass
class DoorMatchCounts:
    horizontal: torch.Tensor
    vertical: torch.Tensor

    def to(self, device: torch.device) -> "DoorMatchCounts":
        return DoorMatchCounts(
            self.horizontal.to(device),
            self.vertical.to(device),
        )


@dataclass
class StateFeatures:
    inventory: torch.Tensor
    room_x: torch.Tensor
    room_y: torch.Tensor
    room_placed: torch.Tensor
    frontier: torch.Tensor
    frontier_occupancy: torch.Tensor
    frontier_neighbor: torch.Tensor
    frontier_neighbor_pair: torch.Tensor
    connection_reachability: torch.Tensor
    frontier_connection_reachability: torch.Tensor

    def to(self, device: torch.device) -> "StateFeatures":
        return StateFeatures(*(value.to(device) for value in vars(self).values()))

    def flatten_candidates(self) -> "StateFeatures":
        return StateFeatures(*(value.flatten(0, 1) for value in vars(self).values()))

    def slice(self, start: int, end: int) -> "StateFeatures":
        return StateFeatures(*(value[start:end] for value in vars(self).values()))


@dataclass
class SparseStateFeatures:
    inventory: torch.Tensor
    room_x: torch.Tensor
    room_y: torch.Tensor
    room_placed: torch.Tensor
    frontier: torch.Tensor
    frontier_occupancy: torch.Tensor
    frontier_neighbor: torch.Tensor
    frontier_neighbor_pair: torch.Tensor
    connection_reachability: torch.Tensor
    frontier_connection_reachability: torch.Tensor
    dense_row_idx: torch.Tensor
    frontier_count: int

    def flatten_candidates(self) -> "SparseStateFeatures":
        return SparseStateFeatures(
            self.inventory.flatten(0, 1),
            self.room_x.flatten(0, 1),
            self.room_y.flatten(0, 1),
            self.room_placed.flatten(0, 1),
            self.frontier,
            self.frontier_occupancy,
            self.frontier_neighbor,
            self.frontier_neighbor_pair,
            self.connection_reachability.flatten(0, 1),
            self.frontier_connection_reachability,
            self.dense_row_idx,
            self.frontier_count,
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

    def __init__(self, rooms: list[dict], state_features: StateFeatureConfig):
        self.state_features = state_features
        self.engine = map_gen.Engine(json.dumps(rooms), state_features.model_dump_json())
        self.rooms = rooms

    def create_environment_group(
        self,
        map_size: tuple[int, int],
        num_envs: int,
        seed: Optional[int] = None,
        frontier_neighbor_count: int = 4,
        frontier_window_size: int = 16,
        num_threads: Optional[int] = None,
        frontier_neighbor_algorithm: Literal["delaunay", "nearest", "nearest-exclusive"] = "delaunay",
    ) -> "EnvironmentGroup":
        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, ()).item())
        env = self.engine.create_environment_group(
            map_size, num_envs, seed, frontier_neighbor_count, frontier_window_size, num_threads,
            frontier_neighbor_algorithm
        )
        return EnvironmentGroup(
            self, env, map_size, num_envs, frontier_neighbor_count, frontier_window_size
        )

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

    def __init__(
        self,
        engine: Engine,
        env: map_gen.EnvironmentGroup,
        map_size: tuple[int, int],
        num_envs: int,
        frontier_neighbor_count: int,
        frontier_window_size: int,
    ):
        self.engine = engine
        self.env = env
        self.map_size = map_size
        self.num_envs = num_envs
        self.frontier_neighbor_count = frontier_neighbor_count
        self.frontier_window_size = frontier_window_size

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
            actions.room_idx.contiguous().cpu().numpy(),
            actions.room_x.contiguous().cpu().numpy(),
            actions.room_y.contiguous().cpu().numpy(),
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

    def get_door_match_counts(self, device: torch.device) -> DoorMatchCounts:
        horizontal, vertical = self.env.get_door_match_counts()
        return DoorMatchCounts(
            horizontal=torch.from_numpy(horizontal).to(device=device, dtype=torch.int64),
            vertical=torch.from_numpy(vertical).to(device=device, dtype=torch.int64),
        )

    @staticmethod
    def _state_features(values, device: torch.device) -> StateFeatures:
        return StateFeatures(*(torch.from_numpy(value).to(device) for value in values))

    def get_state_features(
        self, device: torch.device, environment_start: int = 0, environment_count: Optional[int] = None
    ) -> StateFeatures:
        return self._state_features(
            self.env.get_state_features(environment_start, environment_count), device
        )

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

    def get_sparse_state_features_after_candidates(
        self, actions: Actions, device: torch.device, environment_start: int = 0
    ) -> SparseStateFeatures:
        values, frontier_count = self.env.get_sparse_state_features_after_candidates(
            actions.room_idx.contiguous().cpu().numpy(),
            actions.room_x.contiguous().cpu().numpy(),
            actions.room_y.contiguous().cpu().numpy(),
            environment_start,
        )
        return SparseStateFeatures(
            *(torch.from_numpy(value).to(device) for value in values),
            frontier_count,
        )

    def take_state_feature_profile(self) -> list[float]:
        return self.env.take_state_feature_profile()

    def finish(self):
        self.env.finish()
