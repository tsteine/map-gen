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
    recommended_candidates: int
    shortlist_candidates: int
    temperature: torch.Tensor
    proposal_temperature: torch.Tensor
    reward_door: float
    reward_connection: float
    reward_toilet: float
    reward_balance: float
    reward_toilet_balance: float
    reward_frontier: float
    reward_graph_diameter: float
    reward_save_distance: float
    reward_refill_distance: float
    reward_missing_connect_distance: float
    autocast: bool

    @property
    def max_candidates(self) -> int:
        return self.recommended_candidates


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

    def to(self, device: torch.device, non_blocking: bool = False) -> "Actions":
        return Actions(
            self.room_idx.to(device, non_blocking=non_blocking),
            self.room_x.to(device, non_blocking=non_blocking),
            self.room_y.to(device, non_blocking=non_blocking),
        )

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
    recommended_candidates: torch.Tensor

    def to(self, device: torch.device) -> "EpisodeData":
        return EpisodeData(
            self.actions.to(device),
            self.temperature.to(device),
            self.recommended_candidates.to(device),
        )

    def slice(self, start: int, end: int) -> "EpisodeData":
        return EpisodeData(
            self.actions.slice(start, end),
            self.temperature[start:end],
            self.recommended_candidates[start:end],
        )


@dataclass
class ProposalData:
    frontier_idx: torch.Tensor
    door_variant_idx: torch.Tensor
    selected_candidate: torch.Tensor
    target_logits: torch.Tensor

    def to(self, device: torch.device) -> "ProposalData":
        return ProposalData(
            self.frontier_idx.to(device),
            self.door_variant_idx.to(device),
            self.selected_candidate.to(device),
            self.target_logits.to(device),
        )

    def slice(self, start: int, end: int) -> "ProposalData":
        return ProposalData(
            self.frontier_idx[start:end],
            self.door_variant_idx[start:end],
            self.selected_candidate[start:end],
            self.target_logits[start:end],
        )


# Each tensor here is int8 with shape
#    [batch, time, output]  during training,
#    [batch, candidate, output]  during generation
@dataclass
class PreliminaryOutcomes:
    # -1 = unknown, 0 = valid (door is connected), 1 = invalid (door is not connected)
    door_invalid: torch.Tensor
    # -1 = unknown, 0 = valid (connection has return path), 1 = invalid (connection does not have return path)
    connection_invalid: torch.Tensor
    # -1 = unknown, 0 = valid (the Toilet crosses exactly one room), 1 = invalid
    toilet_invalid: torch.Tensor
    # -1 = unknown; for a valid door this is its matched partner's index within
    # the opposite direction; for an invalid door this is the opposite direction
    # door count sentinel.
    door_match: torch.Tensor

    def to(self, device: torch.device, non_blocking: bool = False) -> "PreliminaryOutcomes":
        return PreliminaryOutcomes(
            self.door_invalid.to(device, non_blocking=non_blocking),
            self.connection_invalid.to(device, non_blocking=non_blocking),
            self.toilet_invalid.to(device, non_blocking=non_blocking),
            self.door_match.to(device, non_blocking=non_blocking),
        )


@dataclass
class EpisodeOutcomes:
    validity: PreliminaryOutcomes
    toilet_crossed_room_idx: torch.Tensor
    avg_frontiers: torch.Tensor
    graph_diameter: torch.Tensor
    save_distance: torch.Tensor
    save_distance_mask: torch.Tensor
    refill_distance: torch.Tensor
    refill_distance_mask: torch.Tensor
    missing_connect_distance: torch.Tensor
    missing_connect_distance_mask: torch.Tensor

    def to(self, device: torch.device) -> "EpisodeOutcomes":
        return EpisodeOutcomes(
            self.validity.to(device),
            self.toilet_crossed_room_idx.to(device),
            self.avg_frontiers.to(device),
            self.graph_diameter.to(device),
            self.save_distance.to(device),
            self.save_distance_mask.to(device),
            self.refill_distance.to(device),
            self.refill_distance_mask.to(device),
            self.missing_connect_distance.to(device),
            self.missing_connect_distance_mask.to(device),
        )


@dataclass
class SparseFeatureRequirements:
    sparse_row_count: int
    worker_sparse_row_counts: list[int]


@dataclass
class ProposalCandidateMask:
    proposal_frontier_idx: torch.Tensor
    mask: torch.Tensor
    valid_counts: torch.Tensor
    door_variant_count: int

    def to(self, device: torch.device) -> "ProposalCandidateMask":
        return ProposalCandidateMask(
            self.proposal_frontier_idx.to(device),
            self.mask.to(device),
            self.valid_counts.to(device),
            self.door_variant_count,
        )


@dataclass
class CandidateStats:
    clean_counts: torch.Tensor
    evaluated_counts: torch.Tensor
    rejected_counts: torch.Tensor

    def to(self, device: torch.device, non_blocking: bool = False) -> "CandidateStats":
        return CandidateStats(
            self.clean_counts.to(device, non_blocking=non_blocking),
            self.evaluated_counts.to(device, non_blocking=non_blocking),
            self.rejected_counts.to(device, non_blocking=non_blocking),
        )


class CandidateSlot:
    def __init__(self, env: "EnvironmentGroup", pin_memory: bool):
        door_count, connection_count = env.engine.get_output_sizes()
        self.environment_capacity = 0
        self.candidate_capacity = 0
        self.door_count = door_count
        self.connection_count = connection_count
        self.pin_memory = pin_memory
        self.room_idx = None
        self.room_x = None
        self.room_y = None
        self.proposal_frontier_idx = None
        self.proposal_door_variant_idx = None
        self.pre_door_invalid = None
        self.pre_connection_invalid = None
        self.pre_toilet_invalid = None
        self.door_invalid = None
        self.connection_invalid = None
        self.toilet_invalid = None
        self.door_match = None
        self.clean_counts = None
        self.evaluated_counts = None
        self.rejected_counts = None

    def _empty(self, shape, dtype):
        return torch.empty(shape, dtype=dtype, pin_memory=self.pin_memory)

    def ensure(self, environment_count: int, candidate_count: int):
        if (
            self.room_idx is not None
            and self.environment_capacity >= environment_count
            and self.candidate_capacity >= candidate_count
        ):
            return
        self.environment_capacity = max(self.environment_capacity, environment_count)
        self.candidate_capacity = max(self.candidate_capacity, candidate_count)
        candidate_shape = (self.environment_capacity, self.candidate_capacity)
        self.room_idx = self._empty(candidate_shape, torch.uint8)
        self.room_x = self._empty(candidate_shape, torch.int8)
        self.room_y = self._empty(candidate_shape, torch.int8)
        self.proposal_frontier_idx = self._empty(candidate_shape, torch.int16)
        self.proposal_door_variant_idx = self._empty(candidate_shape, torch.int16)
        self.pre_door_invalid = self._empty(
            (self.environment_capacity, self.door_count),
            torch.int8,
        )
        self.pre_connection_invalid = self._empty(
            (self.environment_capacity, self.connection_count),
            torch.int8,
        )
        self.pre_toilet_invalid = self._empty((self.environment_capacity,), torch.int8)
        self.door_invalid = self._empty(
            (*candidate_shape, self.door_count),
            torch.int8,
        )
        self.connection_invalid = self._empty(
            (*candidate_shape, self.connection_count),
            torch.int8,
        )
        self.toilet_invalid = self._empty(candidate_shape, torch.int8)
        self.door_match = self._empty((*candidate_shape, self.door_count), torch.int16)
        self.clean_counts = self._empty((self.environment_capacity,), torch.int64)
        self.evaluated_counts = self._empty((self.environment_capacity,), torch.int64)
        self.rejected_counts = self._empty((self.environment_capacity,), torch.int64)

    def actions(self, environment_count: int, candidate_count: int) -> Actions:
        return Actions(
            self.room_idx[:environment_count, :candidate_count],
            self.room_x[:environment_count, :candidate_count],
            self.room_y[:environment_count, :candidate_count],
        )

    def proposal_frontiers(
        self,
        environment_count: int,
        candidate_count: int,
    ) -> torch.Tensor:
        return self.proposal_frontier_idx[:environment_count, :candidate_count]

    def proposal_door_variants(
        self,
        environment_count: int,
        candidate_count: int,
    ) -> torch.Tensor:
        return self.proposal_door_variant_idx[:environment_count, :candidate_count]

    def reward_outcomes(self, environment_count: int) -> PreliminaryOutcomes:
        return PreliminaryOutcomes(
            self.pre_door_invalid[:environment_count],
            self.pre_connection_invalid[:environment_count],
            self.pre_toilet_invalid[:environment_count],
            self.door_match.new_empty((environment_count, 0)),
        )

    def post_candidate_outcomes(
        self,
        environment_count: int,
        candidate_count: int,
    ) -> PreliminaryOutcomes:
        return PreliminaryOutcomes(
            self.door_invalid[:environment_count, :candidate_count],
            self.connection_invalid[:environment_count, :candidate_count],
            self.toilet_invalid[:environment_count, :candidate_count],
            self.door_match[:environment_count, :candidate_count],
        )

    def stats(self, environment_count: int) -> CandidateStats:
        return CandidateStats(
            self.clean_counts[:environment_count],
            self.evaluated_counts[:environment_count],
            self.rejected_counts[:environment_count],
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
class DoorMatches:
    left: torch.Tensor
    right: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor

    def to(self, device: torch.device) -> "DoorMatches":
        return DoorMatches(*(value.to(device) for value in vars(self).values()))

    def slice(self, start: int, end: int) -> "DoorMatches":
        return DoorMatches(*(value[start:end] for value in vars(self).values()))


@dataclass
class Features:
    inventory: torch.Tensor
    room_x: torch.Tensor
    room_y: torch.Tensor
    room_placed: torch.Tensor
    room_part_furthest_destination: torch.Tensor
    room_part_furthest_source: torch.Tensor
    room_part_save_distance: torch.Tensor
    room_part_refill_distance: torch.Tensor
    room_part_frontier_distance: torch.Tensor
    log_temperature: torch.Tensor
    log_recommended_candidates: torch.Tensor
    lookahead_door_invalid: torch.Tensor
    lookahead_door_match: torch.Tensor
    lookahead_connection_invalid: torch.Tensor
    lookahead_toilet_invalid: torch.Tensor
    frontier: torch.Tensor
    frontier_occupancy: torch.Tensor
    frontier_neighbor: torch.Tensor
    frontier_neighbor_pair: torch.Tensor
    connection_reachability: torch.Tensor
    frontier_connection_reachability: torch.Tensor
    toilet_crossed_room_idx: torch.Tensor

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
    room_part_furthest_destination: torch.Tensor
    room_part_furthest_source: torch.Tensor
    room_part_save_distance: torch.Tensor
    room_part_refill_distance: torch.Tensor
    room_part_frontier_distance: torch.Tensor
    log_temperature: torch.Tensor
    log_recommended_candidates: torch.Tensor
    lookahead_door_invalid: torch.Tensor
    lookahead_door_match: torch.Tensor
    lookahead_connection_invalid: torch.Tensor
    lookahead_toilet_invalid: torch.Tensor
    frontier: torch.Tensor
    frontier_occupancy: torch.Tensor
    frontier_neighbor: torch.Tensor
    frontier_neighbor_pair: torch.Tensor
    connection_reachability: torch.Tensor
    frontier_connection_reachability: torch.Tensor
    toilet_crossed_room_idx: torch.Tensor
    row_snapshot_idx: torch.Tensor
    row_frontier_idx: torch.Tensor

    def to(self, device: torch.device, non_blocking: bool = False) -> "SparseFeatures":
        return SparseFeatures(
            *(value.to(device, non_blocking=non_blocking) for value in vars(self).values())
        )

    def flatten_candidates(self) -> "SparseFeatures":
        return SparseFeatures(
            self.inventory.flatten(0, 1),
            self.room_x.flatten(0, 1),
            self.room_y.flatten(0, 1),
            self.room_placed.flatten(0, 1),
            self.room_part_furthest_destination.flatten(0, 1),
            self.room_part_furthest_source.flatten(0, 1),
            self.room_part_save_distance.flatten(0, 1),
            self.room_part_refill_distance.flatten(0, 1),
            self.room_part_frontier_distance.flatten(0, 1),
            self.log_temperature.flatten(0, 1),
            self.log_recommended_candidates.flatten(0, 1),
            self.lookahead_door_invalid.flatten(0, 1),
            self.lookahead_door_match.flatten(0, 1),
            self.lookahead_connection_invalid.flatten(0, 1),
            self.lookahead_toilet_invalid.flatten(0, 1),
            self.frontier,
            self.frontier_occupancy,
            self.frontier_neighbor,
            self.frontier_neighbor_pair,
            self.connection_reachability.flatten(0, 1),
            self.frontier_connection_reachability,
            self.toilet_crossed_room_idx.flatten(0, 1),
            self.row_snapshot_idx,
            self.row_frontier_idx,
        )


@dataclass
class OutputMetadata:
    door: list[tuple[int, int]]
    connection: list[tuple[int, int]]
    num_door_variants: int
    num_connection_variants: int
    room_connection_variant_idx: list[int]
    num_room_connection_variants: int
    num_room_parts: int

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
            num_room_parts,
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
            num_room_parts=num_room_parts,
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

    def step(self, actions: Actions):
        self.env.step(
            actions.room_idx.contiguous().cpu().numpy(),
            actions.room_x.contiguous().cpu().numpy(),
            actions.room_y.contiguous().cpu().numpy(),
        )

    def step_initial(self):
        self.env.step_initial()

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

    def get_proposal_candidate_mask(
        self,
        device: torch.device,
    ) -> ProposalCandidateMask:
        result = self.env.get_proposal_candidate_mask()
        return ProposalCandidateMask(
            torch.from_numpy(result.proposal_frontier_idx).to(device),
            torch.from_numpy(result.mask).to(device),
            torch.from_numpy(result.valid_counts).to(device=device, dtype=torch.int64),
            result.door_variant_count,
        )

    def extract_candidates_from_proposals(
        self,
        candidate_slot: CandidateSlot,
        sampled_frontier_idx: torch.Tensor,
        sampled_door_variant_idx: torch.Tensor,
        recommended_candidates: int,
    ) -> tuple[
        Actions,
        torch.Tensor,
        torch.Tensor,
        PreliminaryOutcomes,
        PreliminaryOutcomes,
        SparseFeatureRequirements,
        CandidateStats,
    ]:
        candidate_count = recommended_candidates
        candidate_slot.ensure(self.num_envs, candidate_count)
        result = self.env.pack_candidates_from_proposals_into(
            sampled_frontier_idx.contiguous().cpu().numpy(),
            sampled_door_variant_idx.contiguous().cpu().numpy(),
            recommended_candidates,
            candidate_slot.room_idx[:self.num_envs, :candidate_count].numpy(),
            candidate_slot.room_x[:self.num_envs, :candidate_count].numpy(),
            candidate_slot.room_y[:self.num_envs, :candidate_count].numpy(),
            candidate_slot.proposal_frontier_idx[:self.num_envs, :candidate_count].numpy(),
            candidate_slot.proposal_door_variant_idx[:self.num_envs, :candidate_count].numpy(),
            candidate_slot.pre_door_invalid[:self.num_envs].numpy(),
            candidate_slot.pre_connection_invalid[:self.num_envs].numpy(),
            candidate_slot.pre_toilet_invalid[:self.num_envs].numpy(),
            candidate_slot.door_invalid[:self.num_envs, :candidate_count].numpy(),
            candidate_slot.connection_invalid[:self.num_envs, :candidate_count].numpy(),
            candidate_slot.toilet_invalid[:self.num_envs, :candidate_count].numpy(),
            candidate_slot.door_match[:self.num_envs, :candidate_count].numpy(),
            candidate_slot.clean_counts[:self.num_envs].numpy(),
            candidate_slot.evaluated_counts[:self.num_envs].numpy(),
            candidate_slot.rejected_counts[:self.num_envs].numpy(),
        )
        return self._candidate_slot_result(candidate_slot, candidate_count, result)

    def _candidate_slot_result(
        self,
        candidate_slot: CandidateSlot,
        candidate_count: int,
        feature_requirements,
    ) -> tuple[
        Actions,
        torch.Tensor,
        torch.Tensor,
        PreliminaryOutcomes,
        PreliminaryOutcomes,
        SparseFeatureRequirements,
        CandidateStats,
    ]:
        return (
            candidate_slot.actions(self.num_envs, candidate_count),
            candidate_slot.proposal_frontiers(self.num_envs, candidate_count),
            candidate_slot.proposal_door_variants(self.num_envs, candidate_count),
            candidate_slot.reward_outcomes(self.num_envs),
            candidate_slot.post_candidate_outcomes(self.num_envs, candidate_count),
            SparseFeatureRequirements(
                feature_requirements.sparse_row_count,
                feature_requirements.worker_sparse_row_counts,
            ),
            candidate_slot.stats(self.num_envs),
        )

    def get_outcomes(self, device: torch.device, verify_consistency: bool) -> EpisodeOutcomes:
        result = self.env.get_outcomes(verify_consistency)
        return EpisodeOutcomes(
            validity=PreliminaryOutcomes(
                door_invalid=torch.from_numpy(result.door_valid).to(device),
                connection_invalid=torch.from_numpy(result.connections_valid).to(device),
                toilet_invalid=torch.from_numpy(result.toilet_valid).to(device),
                door_match=torch.empty(
                    [result.door_valid.shape[0], 0],
                    dtype=torch.int16,
                    device=device,
                ),
            ),
            toilet_crossed_room_idx=torch.from_numpy(result.toilet_crossed_room_idx).to(
                device=device,
                dtype=torch.int64,
            ),
            avg_frontiers=torch.from_numpy(result.avg_frontiers).to(device),
            graph_diameter=torch.from_numpy(result.graph_diameter).to(device),
            save_distance=torch.from_numpy(result.save_distance).to(device),
            save_distance_mask=torch.from_numpy(result.save_distance_mask).to(device),
            refill_distance=torch.from_numpy(result.refill_distance).to(device),
            refill_distance_mask=torch.from_numpy(result.refill_distance_mask).to(device),
            missing_connect_distance=torch.from_numpy(
                result.missing_connect_distance
            ).to(device),
            missing_connect_distance_mask=torch.from_numpy(
                result.missing_connect_distance_mask
            ).to(device),
        )

    def get_current_feature_outcomes(
        self,
        device: torch.device,
        environment_start: int,
        environment_count: int,
    ) -> PreliminaryOutcomes:
        door_invalid, connection_invalid, toilet_invalid, door_match = (
            self.env.get_current_feature_outcomes(
                environment_start,
                environment_count,
            )
        )
        return PreliminaryOutcomes(
            door_invalid=torch.from_numpy(door_invalid).to(device),
            connection_invalid=torch.from_numpy(connection_invalid).to(device),
            toilet_invalid=torch.from_numpy(toilet_invalid).to(device),
            door_match=torch.from_numpy(door_match).to(device),
        )

    def get_door_match_counts(self, device: torch.device) -> DoorMatchCounts:
        horizontal, vertical = self.env.get_door_match_counts()
        return DoorMatchCounts(
            horizontal=torch.from_numpy(horizontal).to(device=device, dtype=torch.int64),
            vertical=torch.from_numpy(vertical).to(device=device, dtype=torch.int64),
        )

    def get_door_matches(self, device: torch.device) -> DoorMatches:
        left, right, up, down = self.env.get_door_matches()
        return DoorMatches(
            left=torch.from_numpy(left).to(device=device, dtype=torch.int64),
            right=torch.from_numpy(right).to(device=device, dtype=torch.int64),
            up=torch.from_numpy(up).to(device=device, dtype=torch.int64),
            down=torch.from_numpy(down).to(device=device, dtype=torch.int64),
        )

    def get_sparse_feature_requirements(
        self,
        environment_start: int = 0,
        environment_count: Optional[int] = None,
    ) -> SparseFeatureRequirements:
        result = self.env.get_sparse_feature_requirements(
            environment_start,
            environment_count,
        )
        return SparseFeatureRequirements(
            result.sparse_row_count,
            result.worker_sparse_row_counts,
        )

    def extract_sparse_features(
        self,
        feature_slot: "SparseFeatureSlot",
        log_temperature: torch.Tensor,
        include_temperature: bool,
        log_recommended_candidates: torch.Tensor,
        include_recommended_candidates: bool,
        lookahead_outcomes: PreliminaryOutcomes,
        include_lookahead_outcomes: bool,
        environment_start: int = 0,
        environment_count: Optional[int] = None,
    ) -> SparseFeatures:
        if environment_count is None:
            environment_count = self.num_envs - environment_start
        feature_requirements = self.get_sparse_feature_requirements(
            environment_start,
            environment_count,
        )
        feature_slot.ensure(environment_count, feature_requirements.sparse_row_count)
        self.env.pack_sparse_features_into(
            environment_count,
            1,
            environment_start,
            feature_requirements.sparse_row_count,
            feature_requirements.worker_sparse_row_counts,
            feature_slot.inventory.numpy(),
            feature_slot.room_x.numpy(),
            feature_slot.room_y.numpy(),
            feature_slot.room_placed.numpy(),
            feature_slot.room_part_furthest_destination.numpy(),
            feature_slot.room_part_furthest_source.numpy(),
            feature_slot.room_part_save_distance.numpy(),
            feature_slot.room_part_refill_distance.numpy(),
            feature_slot.room_part_frontier_distance.numpy(),
            feature_slot.frontier.numpy(),
            feature_slot.frontier_occupancy.numpy(),
            feature_slot.frontier_neighbor.numpy(),
            feature_slot.frontier_neighbor_pair.numpy(),
            feature_slot.connection_reachability.numpy(),
            feature_slot.frontier_connection_reachability.numpy(),
            feature_slot.toilet_crossed_room_idx.numpy(),
            feature_slot.row_snapshot_idx.numpy(),
            feature_slot.row_frontier_idx.numpy(),
        )
        return feature_slot.state_features(
            environment_count,
            log_temperature,
            include_temperature,
            log_recommended_candidates,
            include_recommended_candidates,
            lookahead_outcomes,
            include_lookahead_outcomes,
            feature_requirements.sparse_row_count,
        )

    def finish(self):
        self.env.finish()


# When a GPU is available, we use pinned memory for model input tensors,
# to allow for asynchronous CPU-to-GPU transfers.
class SparseFeatureSlot:
    def __init__(self, env: EnvironmentGroup, pin_memory: bool):
        features = env.engine.features
        inventory_count, _, room_count = env.engine.get_feature_sizes()
        room_part_count = env.engine.get_output_metadata().num_room_parts
        _, connection_count = env.engine.get_output_sizes()
        self.inventory_width = inventory_count * int(features.inventory)
        self.room_width = room_count * int(features.room_position)
        self.room_part_width = room_part_count * int(features.room_part_furthest_distance)
        self.room_part_save_distance_width = (
            room_part_count * int(features.room_part_save_distance)
        )
        self.room_part_refill_distance_width = (
            room_part_count * int(features.room_part_refill_distance)
        )
        self.room_part_frontier_distance_width = (
            room_part_count * int(features.room_part_frontier_distance)
        )
        self.frontier_occupancy_width = (
            (env.frontier_window_size * env.frontier_window_size + 7) // 8
        ) * int(features.frontier_occupancy)
        self.frontier_neighbor_width = (
            env.frontier_neighbor_count * int(features.frontier_neighbor)
        )
        self.frontier_neighbor_pair_width = (
            env.frontier_neighbor_count * int(features.frontier_neighbor_flags)
        )
        self.connection_reachability_width = (
            connection_count * int(features.connection_reachability)
        )
        self.frontier_connection_reachability_width = (
            connection_count * int(features.frontier_connection_reachability)
        )
        self.toilet_crossed_room_width = int(features.toilet_crossed_room)
        self.pin_memory = pin_memory
        self.snapshot_capacity = 0
        self.sparse_row_capacity = 0
        self.inventory = None
        self.room_x = None
        self.room_y = None
        self.room_placed = None
        self.room_part_furthest_destination = None
        self.room_part_furthest_source = None
        self.room_part_save_distance = None
        self.room_part_refill_distance = None
        self.room_part_frontier_distance = None
        self.frontier = None
        self.frontier_occupancy = None
        self.frontier_neighbor = None
        self.frontier_neighbor_pair = None
        self.connection_reachability = None
        self.frontier_connection_reachability = None
        self.toilet_crossed_room_idx = None
        self.row_snapshot_idx = None
        self.row_frontier_idx = None

    def _empty(self, shape, dtype):
        return torch.empty(shape, dtype=dtype, pin_memory=self.pin_memory)

    def ensure(self, snapshot_count: int, sparse_row_count: int):
        if (
            self.inventory is not None and
            self.snapshot_capacity >= snapshot_count
            and self.sparse_row_capacity >= sparse_row_count
        ):
            return
        self.snapshot_capacity = max(self.snapshot_capacity, snapshot_count)
        self.sparse_row_capacity = max(self.sparse_row_capacity, sparse_row_count)
        self.inventory = self._empty(
            (self.snapshot_capacity, self.inventory_width), torch.uint8
        )
        self.room_x = self._empty((self.snapshot_capacity, self.room_width), torch.int8)
        self.room_y = self._empty((self.snapshot_capacity, self.room_width), torch.int8)
        self.room_placed = self._empty(
            (self.snapshot_capacity, self.room_width), torch.uint8
        )
        self.room_part_furthest_destination = self._empty(
            (self.snapshot_capacity, self.room_part_width), torch.uint8
        )
        self.room_part_furthest_source = self._empty(
            (self.snapshot_capacity, self.room_part_width), torch.uint8
        )
        self.room_part_save_distance = self._empty(
            (self.snapshot_capacity, self.room_part_save_distance_width), torch.uint8
        )
        self.room_part_refill_distance = self._empty(
            (self.snapshot_capacity, self.room_part_refill_distance_width), torch.uint8
        )
        self.room_part_frontier_distance = self._empty(
            (self.snapshot_capacity, self.room_part_frontier_distance_width), torch.uint8
        )
        self.frontier = self._empty((self.sparse_row_capacity, 5), torch.int8)
        self.frontier_occupancy = self._empty(
            (self.sparse_row_capacity, self.frontier_occupancy_width), torch.uint8
        )
        self.frontier_neighbor = self._empty(
            (self.sparse_row_capacity, self.frontier_neighbor_width), torch.int16
        )
        self.frontier_neighbor_pair = self._empty(
            (self.sparse_row_capacity, self.frontier_neighbor_pair_width), torch.uint8
        )
        self.connection_reachability = self._empty(
            (self.snapshot_capacity, self.connection_reachability_width), torch.uint8
        )
        self.frontier_connection_reachability = self._empty(
            (self.sparse_row_capacity, self.frontier_connection_reachability_width),
            torch.uint8,
        )
        self.toilet_crossed_room_idx = self._empty(
            (self.snapshot_capacity, self.toilet_crossed_room_width),
            torch.int16,
        )
        self.row_snapshot_idx = self._empty((self.sparse_row_capacity,), torch.int64)
        self.row_frontier_idx = self._empty((self.sparse_row_capacity,), torch.int16)

    def state_features(
        self,
        environment_count: int,
        log_temperature: torch.Tensor,
        include_temperature: bool,
        log_recommended_candidates: torch.Tensor,
        include_recommended_candidates: bool,
        lookahead_outcomes: PreliminaryOutcomes,
        include_lookahead_outcomes: bool,
        sparse_row_count: int,
    ) -> SparseFeatures:
        if not include_temperature:
            log_temperature = log_temperature.new_empty([*log_temperature.shape, 0])
        if not include_recommended_candidates:
            log_recommended_candidates = log_recommended_candidates.new_empty([
                *log_recommended_candidates.shape,
                0,
            ])
        lookahead_door_invalid = lookahead_outcomes.door_invalid
        lookahead_door_match = lookahead_outcomes.door_match
        lookahead_connection_invalid = lookahead_outcomes.connection_invalid
        lookahead_toilet_invalid = lookahead_outcomes.toilet_invalid
        if not include_lookahead_outcomes:
            lookahead_door_invalid = lookahead_door_invalid.new_empty([
                *lookahead_door_invalid.shape[:-1],
                0,
            ])
            lookahead_door_match = lookahead_door_match.new_empty([
                *lookahead_door_match.shape[:-1],
                0,
            ])
            lookahead_connection_invalid = lookahead_connection_invalid.new_empty([
                *lookahead_connection_invalid.shape[:-1],
                0,
            ])
            lookahead_toilet_invalid = lookahead_toilet_invalid.new_empty([
                *lookahead_toilet_invalid.shape,
                0,
            ])
        return SparseFeatures(
            self.inventory[:environment_count],
            self.room_x[:environment_count],
            self.room_y[:environment_count],
            self.room_placed[:environment_count],
            self.room_part_furthest_destination[:environment_count],
            self.room_part_furthest_source[:environment_count],
            self.room_part_save_distance[:environment_count],
            self.room_part_refill_distance[:environment_count],
            self.room_part_frontier_distance[:environment_count],
            log_temperature,
            log_recommended_candidates,
            lookahead_door_invalid,
            lookahead_door_match,
            lookahead_connection_invalid,
            lookahead_toilet_invalid,
            self.frontier[:sparse_row_count],
            self.frontier_occupancy[:sparse_row_count],
            self.frontier_neighbor[:sparse_row_count],
            self.frontier_neighbor_pair[:sparse_row_count],
            self.connection_reachability[:environment_count],
            self.frontier_connection_reachability[:sparse_row_count],
            self.toilet_crossed_room_idx[:environment_count],
            self.row_snapshot_idx[:sparse_row_count],
            self.row_frontier_idx[:sparse_row_count],
        )

    def features(
        self,
        environment_count: int,
        candidate_count: int,
        log_temperature: torch.Tensor,
        include_temperature: bool,
        log_recommended_candidates: torch.Tensor,
        include_recommended_candidates: bool,
        lookahead_outcomes: PreliminaryOutcomes,
        include_lookahead_outcomes: bool,
        sparse_row_count: int,
    ) -> SparseFeatures:
        snapshot_count = environment_count * candidate_count
        if not include_temperature:
            log_temperature = log_temperature.new_empty(
                [environment_count, candidate_count, 0]
            )
        if not include_recommended_candidates:
            log_recommended_candidates = log_recommended_candidates.new_empty(
                [environment_count, candidate_count, 0]
            )
        lookahead_door_invalid = lookahead_outcomes.door_invalid
        lookahead_door_match = lookahead_outcomes.door_match
        lookahead_connection_invalid = lookahead_outcomes.connection_invalid
        lookahead_toilet_invalid = lookahead_outcomes.toilet_invalid
        if not include_lookahead_outcomes:
            lookahead_door_invalid = lookahead_door_invalid.new_empty(
                [environment_count, candidate_count, 0]
            )
            lookahead_door_match = lookahead_door_match.new_empty(
                [environment_count, candidate_count, 0]
            )
            lookahead_connection_invalid = lookahead_connection_invalid.new_empty(
                [environment_count, candidate_count, 0]
            )
            lookahead_toilet_invalid = lookahead_toilet_invalid.new_empty(
                [environment_count, candidate_count, 0]
            )
        return SparseFeatures(
            self.inventory[:snapshot_count].view(
                environment_count, candidate_count, self.inventory_width
            ),
            self.room_x[:snapshot_count].view(environment_count, candidate_count, self.room_width),
            self.room_y[:snapshot_count].view(environment_count, candidate_count, self.room_width),
            self.room_placed[:snapshot_count].view(
                environment_count, candidate_count, self.room_width
            ),
            self.room_part_furthest_destination[:snapshot_count].view(
                environment_count, candidate_count, self.room_part_width
            ),
            self.room_part_furthest_source[:snapshot_count].view(
                environment_count, candidate_count, self.room_part_width
            ),
            self.room_part_save_distance[:snapshot_count].view(
                environment_count, candidate_count, self.room_part_save_distance_width
            ),
            self.room_part_refill_distance[:snapshot_count].view(
                environment_count, candidate_count, self.room_part_refill_distance_width
            ),
            self.room_part_frontier_distance[:snapshot_count].view(
                environment_count, candidate_count, self.room_part_frontier_distance_width
            ),
            log_temperature,
            log_recommended_candidates,
            lookahead_door_invalid,
            lookahead_door_match,
            lookahead_connection_invalid,
            lookahead_toilet_invalid,
            self.frontier[:sparse_row_count],
            self.frontier_occupancy[:sparse_row_count],
            self.frontier_neighbor[:sparse_row_count],
            self.frontier_neighbor_pair[:sparse_row_count],
            self.connection_reachability[:snapshot_count].view(
                environment_count, candidate_count, self.connection_reachability_width
            ),
            self.frontier_connection_reachability[:sparse_row_count],
            self.toilet_crossed_room_idx[:snapshot_count].view(
                environment_count, candidate_count, self.toilet_crossed_room_width
            ),
            self.row_snapshot_idx[:sparse_row_count],
            self.row_frontier_idx[:sparse_row_count],
        )


def extract_candidate_features(
    env: EnvironmentGroup,
    candidates: Actions,
    log_temperature: torch.Tensor,
    include_temperature: bool,
    log_recommended_candidates: torch.Tensor,
    include_recommended_candidates: bool,
    lookahead_outcomes: PreliminaryOutcomes,
    include_lookahead_outcomes: bool,
    feature_requirements: SparseFeatureRequirements,
    feature_slot: SparseFeatureSlot,
) -> SparseFeatures:
    sparse_row_count = feature_requirements.sparse_row_count
    worker_sparse_row_counts = feature_requirements.worker_sparse_row_counts
    feature_slot.ensure(candidates.room_idx.numel(), sparse_row_count)
    env.env.pack_sparse_features_into(
        candidates.room_idx.shape[0],
        candidates.room_idx.shape[1],
        0,
        sparse_row_count,
        worker_sparse_row_counts,
        feature_slot.inventory.numpy(),
        feature_slot.room_x.numpy(),
        feature_slot.room_y.numpy(),
        feature_slot.room_placed.numpy(),
        feature_slot.room_part_furthest_destination.numpy(),
        feature_slot.room_part_furthest_source.numpy(),
        feature_slot.room_part_save_distance.numpy(),
        feature_slot.room_part_refill_distance.numpy(),
        feature_slot.room_part_frontier_distance.numpy(),
        feature_slot.frontier.numpy(),
        feature_slot.frontier_occupancy.numpy(),
        feature_slot.frontier_neighbor.numpy(),
        feature_slot.frontier_neighbor_pair.numpy(),
        feature_slot.connection_reachability.numpy(),
        feature_slot.frontier_connection_reachability.numpy(),
        feature_slot.toilet_crossed_room_idx.numpy(),
        feature_slot.row_snapshot_idx.numpy(),
        feature_slot.row_frontier_idx.numpy(),
    )
    return feature_slot.features(
        candidates.room_idx.shape[0],
        candidates.room_idx.shape[1],
        log_temperature,
        include_temperature,
        log_recommended_candidates,
        include_recommended_candidates,
        lookahead_outcomes,
        include_lookahead_outcomes,
        sparse_row_count,
    ).flatten_candidates()
