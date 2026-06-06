from __future__ import annotations

# Python wrappers around the Rust map generation engine, includes (zero-copy) conversions
# between numpy and torch tensors.
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

import torch
import json

import map_gen

if TYPE_CHECKING:
    from train_config import FeatureConfig

@dataclass
class GenerateConfig:
    episode_length: int
    max_candidates: int
    temperature: torch.Tensor
    lookahead_outcomes: bool
    autocast: bool


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

    def slice(self, start: int, end: int) -> "Actions":
        return Actions(
            self.room_idx[start:end],
            self.room_x[start:end],
            self.room_y[start:end],
        )


@dataclass
class EpisodeData:
    actions: Actions
    temperature: torch.Tensor
    action_candidates: torch.Tensor

    def to(self, device: torch.device) -> "EpisodeData":
        return EpisodeData(
            self.actions.to(device),
            self.temperature.to(device),
            self.action_candidates.to(device),
        )

    def slice(self, start: int, end: int) -> "EpisodeData":
        return EpisodeData(
            self.actions.slice(start, end),
            self.temperature[start:end],
            self.action_candidates[start:end],
        )


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
class Features:
    inventory: torch.Tensor
    room_x: torch.Tensor
    room_y: torch.Tensor
    room_placed: torch.Tensor
    log_temperature: torch.Tensor
    log_action_candidates: torch.Tensor
    frontier: torch.Tensor
    frontier_occupancy: torch.Tensor
    frontier_neighbor: torch.Tensor
    frontier_neighbor_pair: torch.Tensor
    connection_reachability: torch.Tensor
    frontier_connection_reachability: torch.Tensor

    def to(self, device: torch.device) -> "Features":
        return Features(*(value.to(device) for value in vars(self).values()))

    def flatten_candidates(self) -> "Features":
        return Features(*(value.flatten(0, 1) for value in vars(self).values()))

    def slice(self, start: int, end: int) -> "Features":
        return Features(*(value[start:end] for value in vars(self).values()))


@dataclass
class SparseFeatures:
    inventory: torch.Tensor
    room_x: torch.Tensor
    room_y: torch.Tensor
    room_placed: torch.Tensor
    log_temperature: torch.Tensor
    log_action_candidates: torch.Tensor
    frontier: torch.Tensor
    frontier_occupancy: torch.Tensor
    frontier_neighbor: torch.Tensor
    frontier_neighbor_pair: torch.Tensor
    connection_reachability: torch.Tensor
    frontier_connection_reachability: torch.Tensor
    dense_row_idx: torch.Tensor
    frontier_count: int

    def flatten_candidates(self) -> "SparseFeatures":
        return SparseFeatures(
            self.inventory.flatten(0, 1),
            self.room_x.flatten(0, 1),
            self.room_y.flatten(0, 1),
            self.room_placed.flatten(0, 1),
            self.log_temperature.flatten(0, 1),
            self.log_action_candidates.flatten(0, 1),
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

    def __init__(self, rooms: list[dict], features: FeatureConfig):
        self.features = features
        self.engine = map_gen.Engine(json.dumps(rooms), features.model_dump_json())
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

    def get_feature_sizes(self) -> tuple[int, int, int]:
        return self.engine.get_feature_sizes()


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

    def step_known(self, actions: Actions):
        self.env.step_known(
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
    def _features(
        values,
        device: torch.device,
        log_temperature: torch.Tensor,
        include_temperature: bool,
        log_action_candidates: torch.Tensor,
        include_action_candidates: bool,
    ) -> Features:
        tensors = [torch.from_numpy(value).to(device) for value in values]
        log_temperature = log_temperature.to(device)
        if not include_temperature:
            log_temperature = log_temperature.new_empty([*log_temperature.shape, 0])
        log_action_candidates = log_action_candidates.to(device)
        if not include_action_candidates:
            log_action_candidates = log_action_candidates.new_empty([
                *log_action_candidates.shape,
                0,
            ])
        return Features(
            *tensors[:4],
            log_temperature,
            log_action_candidates,
            *tensors[4:],
        )

    def get_features(
        self,
        device: torch.device,
        log_temperature: torch.Tensor,
        include_temperature: bool,
        log_action_candidates: torch.Tensor,
        include_action_candidates: bool,
        environment_start: int = 0,
        environment_count: Optional[int] = None,
    ) -> Features:
        return self._features(
            self.env.get_features(environment_start, environment_count),
            device,
            log_temperature,
            include_temperature,
            log_action_candidates,
            include_action_candidates,
        )

    def get_features_after_candidates(
        self,
        actions: Actions,
        device: torch.device,
        log_temperature: torch.Tensor,
        include_temperature: bool,
        log_action_candidates: torch.Tensor,
        include_action_candidates: bool,
        environment_start: int = 0,
    ) -> Features:
        values = self.env.get_features_after_candidates(
            actions.room_idx.contiguous().cpu().numpy(),
            actions.room_x.contiguous().cpu().numpy(),
            actions.room_y.contiguous().cpu().numpy(),
            environment_start,
        )
        return self._features(
            values,
            device,
            log_temperature,
            include_temperature,
            log_action_candidates,
            include_action_candidates,
        )

    def get_sparse_features_after_candidates(
        self,
        actions: Actions,
        device: torch.device,
        log_temperature: torch.Tensor,
        include_temperature: bool,
        log_action_candidates: torch.Tensor,
        include_action_candidates: bool,
        environment_start: int = 0,
    ) -> SparseFeatures:
        values, frontier_count = self.env.get_sparse_features_after_candidates(
            actions.room_idx.contiguous().cpu().numpy(),
            actions.room_x.contiguous().cpu().numpy(),
            actions.room_y.contiguous().cpu().numpy(),
            environment_start,
        )
        return SparseFeatures(
            *(torch.from_numpy(value).to(device) for value in values[:4]),
            log_temperature.to(device) if include_temperature else log_temperature.new_empty([
                log_temperature.shape[0],
                log_temperature.shape[1],
                0,
            ]).to(device),
            log_action_candidates.to(device) if include_action_candidates else log_action_candidates.new_empty([
                log_action_candidates.shape[0],
                log_action_candidates.shape[1],
                0,
            ]).to(device),
            *(torch.from_numpy(value).to(device) for value in values[4:]),
            frontier_count,
        )

    def finish(self):
        self.env.finish()
