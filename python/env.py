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

    def to(self, device: torch.device) -> "PreliminaryOutcomes":
        return PreliminaryOutcomes(
            self.door_invalid.to(device),
            self.connection_invalid.to(device),
            self.toilet_invalid.to(device),
            self.door_match.to(device),
        )


@dataclass
class EpisodeOutcomes:
    validity: PreliminaryOutcomes
    toilet_crossed_room_idx: torch.Tensor
    avg_frontiers: torch.Tensor
    graph_diameter: torch.Tensor
    save_distance: torch.Tensor
    save_distance_mask: torch.Tensor

    def to(self, device: torch.device) -> "EpisodeOutcomes":
        return EpisodeOutcomes(
            self.validity.to(device),
            self.toilet_crossed_room_idx.to(device),
            self.avg_frontiers.to(device),
            self.graph_diameter.to(device),
            self.save_distance.to(device),
            self.save_distance_mask.to(device),
        )


@dataclass
class CandidateFeatureRequirements:
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

    def to(self, device: torch.device) -> "CandidateStats":
        return CandidateStats(
            self.clean_counts.to(device),
            self.evaluated_counts.to(device),
            self.rejected_counts.to(device),
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


FEATURE_RESULT_FIELDS = (
    "inventory",
    "room_x",
    "room_y",
    "room_placed",
    "room_part_furthest_destination",
    "room_part_furthest_source",
    "room_part_save_distance",
    "room_part_frontier_distance",
    "frontier",
    "frontier_occupancy",
    "frontier_neighbor",
    "frontier_neighbor_pair",
    "connection_reachability",
    "frontier_connection_reachability",
    "toilet_crossed_room_idx",
)

SPARSE_FEATURE_RESULT_FIELDS = (
    *FEATURE_RESULT_FIELDS,
    "row_snapshot_idx",
    "row_frontier_idx",
)


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

    def get_candidates_with_outcomes(
        self,
        recommended_candidates: int,
        proposal_temperature: torch.Tensor,
        proposal_scores: torch.Tensor | None,
        device: torch.device,
    ) -> tuple[
        Actions,
        torch.Tensor,
        torch.Tensor,
        PreliminaryOutcomes,
        PreliminaryOutcomes,
        CandidateFeatureRequirements,
        CandidateStats,
    ]:
        result = self.env.get_candidates_with_outcomes(
            recommended_candidates,
            0,
            proposal_temperature.contiguous().cpu().numpy(),
            None if proposal_scores is None else proposal_scores.contiguous().cpu().numpy(),
        )
        return self._candidate_result(result, device)

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

    def get_candidates_from_proposals(
        self,
        sampled_frontier_idx: torch.Tensor,
        sampled_door_variant_idx: torch.Tensor,
        recommended_candidates: int,
        device: torch.device,
    ) -> tuple[
        Actions,
        torch.Tensor,
        torch.Tensor,
        PreliminaryOutcomes,
        PreliminaryOutcomes,
        CandidateFeatureRequirements,
        CandidateStats,
    ]:
        result = self.env.get_candidates_from_proposals(
            sampled_frontier_idx.contiguous().cpu().numpy(),
            sampled_door_variant_idx.contiguous().cpu().numpy(),
            recommended_candidates,
        )
        return self._candidate_result(result, device)

    @staticmethod
    def _candidate_result(
        result,
        device: torch.device,
    ) -> tuple[
        Actions,
        torch.Tensor,
        torch.Tensor,
        PreliminaryOutcomes,
        PreliminaryOutcomes,
        CandidateFeatureRequirements,
        CandidateStats,
    ]:
        return (
            Actions(
                room_idx=torch.from_numpy(result.room_idx).to(device),
                room_x=torch.from_numpy(result.room_x).to(device),
                room_y=torch.from_numpy(result.room_y).to(device),
            ),
            torch.from_numpy(result.proposal_frontier_idx).to(device),
            torch.from_numpy(result.proposal_door_variant_idx).to(device),
            PreliminaryOutcomes(
                door_invalid=torch.from_numpy(result.pre_door_valid).to(device),
                connection_invalid=torch.from_numpy(result.pre_connections_valid).to(device),
                toilet_invalid=torch.from_numpy(result.pre_toilet_valid).to(device),
                door_match=torch.empty(
                    [result.pre_door_valid.shape[0], 0],
                    dtype=torch.int16,
                    device=device,
                ),
            ),
            PreliminaryOutcomes(
                door_invalid=torch.from_numpy(result.door_valid).to(device),
                connection_invalid=torch.from_numpy(result.connections_valid).to(device),
                toilet_invalid=torch.from_numpy(result.toilet_valid).to(device),
                door_match=torch.from_numpy(result.door_match).to(device),
            ),
            CandidateFeatureRequirements(
                result.sparse_row_count,
                result.worker_sparse_row_counts,
            ),
            CandidateStats(
                clean_counts=torch.from_numpy(result.clean_counts).to(
                    device=device,
                    dtype=torch.int64,
                ),
                evaluated_counts=torch.from_numpy(result.evaluated_counts).to(
                    device=device,
                    dtype=torch.int64,
                ),
                rejected_counts=torch.from_numpy(result.rejected_counts).to(
                    device=device,
                    dtype=torch.int64,
                ),
            ),
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
        )

    def get_outcomes_after_candidates(
        self,
        actions: Actions,
        device: torch.device,
        environment_start: int = 0,
    ) -> PreliminaryOutcomes:
        door_invalid, connection_invalid, toilet_invalid, door_match = (
            self.env.get_outcomes_after_candidates(
                actions.room_idx.contiguous().cpu().numpy(),
                actions.room_x.contiguous().cpu().numpy(),
                actions.room_y.contiguous().cpu().numpy(),
                environment_start,
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

    @staticmethod
    def _result_tensors(result, fields: tuple[str, ...], device: torch.device):
        return [torch.from_numpy(getattr(result, field)).to(device) for field in fields]

    @staticmethod
    def _features(
        values,
        device: torch.device,
        log_temperature: torch.Tensor,
        include_temperature: bool,
        log_recommended_candidates: torch.Tensor,
        include_recommended_candidates: bool,
        lookahead_outcomes: PreliminaryOutcomes,
        include_lookahead_outcomes: bool,
    ) -> Features:
        tensors = EnvironmentGroup._result_tensors(
            result=values,
            fields=FEATURE_RESULT_FIELDS,
            device=device,
        )
        log_temperature = log_temperature.to(device)
        if not include_temperature:
            log_temperature = log_temperature.new_empty([*log_temperature.shape, 0])
        log_recommended_candidates = log_recommended_candidates.to(device)
        if not include_recommended_candidates:
            log_recommended_candidates = log_recommended_candidates.new_empty([
                *log_recommended_candidates.shape,
                0,
            ])
        lookahead_door_invalid = lookahead_outcomes.door_invalid.to(device)
        lookahead_door_match = lookahead_outcomes.door_match.to(device)
        lookahead_connection_invalid = lookahead_outcomes.connection_invalid.to(device)
        lookahead_toilet_invalid = lookahead_outcomes.toilet_invalid.to(device)
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
        return Features(
            *tensors[:8],
            log_temperature,
            log_recommended_candidates,
            lookahead_door_invalid,
            lookahead_door_match,
            lookahead_connection_invalid,
            lookahead_toilet_invalid,
            *tensors[8:],
        )

    def get_features(
        self,
        device: torch.device,
        log_temperature: torch.Tensor,
        include_temperature: bool,
        log_recommended_candidates: torch.Tensor,
        include_recommended_candidates: bool,
        lookahead_outcomes: PreliminaryOutcomes,
        include_lookahead_outcomes: bool,
        environment_start: int = 0,
        environment_count: Optional[int] = None,
    ) -> Features:
        return self._features(
            self.env.get_features(environment_start, environment_count),
            device,
            log_temperature,
            include_temperature,
            log_recommended_candidates,
            include_recommended_candidates,
            lookahead_outcomes,
            include_lookahead_outcomes,
        )

    def get_sparse_features(
        self,
        device: torch.device,
        log_temperature: torch.Tensor,
        include_temperature: bool,
        log_recommended_candidates: torch.Tensor,
        include_recommended_candidates: bool,
        lookahead_outcomes: PreliminaryOutcomes,
        include_lookahead_outcomes: bool,
        environment_start: int = 0,
        environment_count: Optional[int] = None,
    ) -> SparseFeatures:
        values = self.env.get_sparse_features(environment_start, environment_count)
        tensors = self._result_tensors(values, SPARSE_FEATURE_RESULT_FIELDS, device)
        log_temperature = log_temperature.to(device)
        if not include_temperature:
            log_temperature = log_temperature.new_empty([*log_temperature.shape, 0])
        log_recommended_candidates = log_recommended_candidates.to(device)
        if not include_recommended_candidates:
            log_recommended_candidates = log_recommended_candidates.new_empty([
                *log_recommended_candidates.shape,
                0,
            ])
        lookahead_door_invalid = lookahead_outcomes.door_invalid.to(device)
        lookahead_door_match = lookahead_outcomes.door_match.to(device)
        lookahead_connection_invalid = lookahead_outcomes.connection_invalid.to(device)
        lookahead_toilet_invalid = lookahead_outcomes.toilet_invalid.to(device)
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
            *tensors[:8],
            log_temperature,
            log_recommended_candidates,
            lookahead_door_invalid,
            lookahead_door_match,
            lookahead_connection_invalid,
            lookahead_toilet_invalid,
            *tensors[8:],
        )

    def finish(self):
        self.env.finish()
