from __future__ import annotations

# Python wrappers around the Rust map generation engine, includes (zero-copy) conversions
# between numpy and torch tensors.
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

import torch
import json

import map_gen

if TYPE_CHECKING:
    from train_config import EngineFeatureConfig, FeatureConfig

AREA_COUNT = 6
DUMMY_AREA = AREA_COUNT


def proposal_action_idx(door_variant_idx: torch.Tensor, room_area: torch.Tensor) -> torch.Tensor:
    return door_variant_idx * AREA_COUNT + room_area


def proposal_action_door_variant_idx(action_idx: torch.Tensor) -> torch.Tensor:
    return torch.div(action_idx, AREA_COUNT, rounding_mode="floor")


def proposal_action_room_area(action_idx: torch.Tensor) -> torch.Tensor:
    return action_idx % AREA_COUNT


def area_connected_component_bucket_excess(
    upper_bounds: list[int],
    device: torch.device,
) -> torch.Tensor:
    representatives = [*upper_bounds, upper_bounds[-1] + 1]
    return torch.tensor(
        [max(components - 1, 0) for components in representatives],
        dtype=torch.float32,
        device=device,
    )


@dataclass
class GenerateConfig:
    episode_length: int
    recommended_candidates: int
    shortlist_candidates: int
    max_candidate_areas_per_placement: int
    gpu_prefetch_batches: int
    temperature: torch.Tensor
    frontier_temperature: torch.Tensor
    proposal_temperature: torch.Tensor
    reward_door: float | torch.Tensor
    reward_connection: float | torch.Tensor
    reward_toilet: float | torch.Tensor
    reward_phantoon: float | torch.Tensor
    reward_balance: float | torch.Tensor
    reward_toilet_balance: float | torch.Tensor
    reward_frontier: float | torch.Tensor
    reward_graph_diameter: float | torch.Tensor
    reward_save_distance: float | torch.Tensor
    reward_refill_distance: float | torch.Tensor
    reward_missing_connect_utility: float | torch.Tensor
    reward_area_connected: float | torch.Tensor
    reward_area_connected_excess: float | torch.Tensor
    reward_area_crossing: float | torch.Tensor
    reward_area_size_valid: float | torch.Tensor
    reward_area_map_station: float | torch.Tensor
    area_connected_component_bucket_excess: torch.Tensor
    generation_variable_floats: torch.Tensor
    log_temperature_model: torch.Tensor
    log_recommended_candidates_model: torch.Tensor
    generation_variable_floats_model: torch.Tensor
    candidate_log_temperature_model: torch.Tensor
    candidate_log_recommended_candidates_model: torch.Tensor
    candidate_generation_variable_floats_model: torch.Tensor
    distance_proximity_scale: float
    autocast: bool


# Each tensor here is uint8 with shape
#    [batch, time]  during training,
#    [batch, candidate]  during generation
@dataclass
class Actions:
    room_idx: torch.Tensor
    room_x: torch.Tensor
    room_y: torch.Tensor
    room_area: torch.Tensor

    def select(self, index: torch.Tensor) -> "Actions":
        return Actions(
            room_idx=torch.gather(self.room_idx, 1, index.unsqueeze(1)).squeeze(1),
            room_x=torch.gather(self.room_x, 1, index.unsqueeze(1)).squeeze(1),
            room_y=torch.gather(self.room_y, 1, index.unsqueeze(1)).squeeze(1),
            room_area=torch.gather(self.room_area, 1, index.unsqueeze(1)).squeeze(1),
        )

    def to(self, device: torch.device, non_blocking: bool = False) -> "Actions":
        return Actions(
            room_idx=self.room_idx.to(device, non_blocking=non_blocking),
            room_x=self.room_x.to(device, non_blocking=non_blocking),
            room_y=self.room_y.to(device, non_blocking=non_blocking),
            room_area=self.room_area.to(device, non_blocking=non_blocking),
        )

    def slice(self, start: int, end: int) -> "Actions":
        return Actions(
            room_idx=self.room_idx[start:end],
            room_x=self.room_x[start:end],
            room_y=self.room_y[start:end],
            room_area=self.room_area[start:end],
        )


@dataclass
class EpisodeData:
    actions: Actions
    temperature: torch.Tensor
    recommended_candidates: torch.Tensor
    generation_variable_floats: torch.Tensor

    def to(self, device: torch.device) -> "EpisodeData":
        return EpisodeData(
            actions=self.actions.to(device),
            temperature=self.temperature.to(device),
            recommended_candidates=self.recommended_candidates.to(device),
            generation_variable_floats=self.generation_variable_floats.to(device),
        )

    def slice(self, start: int, end: int) -> "EpisodeData":
        return EpisodeData(
            actions=self.actions.slice(start, end),
            temperature=self.temperature[start:end],
            recommended_candidates=self.recommended_candidates[start:end],
            generation_variable_floats=self.generation_variable_floats[start:end],
        )


@dataclass
class ProposalData:
    frontier_idx: torch.Tensor
    action_idx: torch.Tensor
    selected_candidate: torch.Tensor
    target_logits: torch.Tensor
    frontier_value_target: torch.Tensor

    def to(self, device: torch.device) -> "ProposalData":
        return ProposalData(
            frontier_idx=self.frontier_idx.to(device),
            action_idx=self.action_idx.to(device),
            selected_candidate=self.selected_candidate.to(device),
            target_logits=self.target_logits.to(device),
            frontier_value_target=self.frontier_value_target.to(device),
        )

    def slice(self, start: int, end: int) -> "ProposalData":
        return ProposalData(
            frontier_idx=self.frontier_idx[start:end],
            action_idx=self.action_idx[start:end],
            selected_candidate=self.selected_candidate[start:end],
            target_logits=self.target_logits[start:end],
            frontier_value_target=self.frontier_value_target[start:end],
        )


# Each tensor here is int8 with shape
#    [batch, time, output]  during training,
#    [batch, candidate, output]  during generation
@dataclass
class StepOutcomes:
    # -1 = unknown, 0 = valid (door is connected), 1 = invalid (door is not connected)
    door_invalid: torch.Tensor
    # -1 = unknown, 0 = valid (connection has return path), 1 = invalid (connection does not have return path)
    connection_invalid: torch.Tensor
    # -1 = unknown, 0 = valid (the Toilet crosses exactly one room), 1 = invalid
    toilet_invalid: torch.Tensor
    # -1 = unknown, 0 = valid (Phantoon rooms connect to the same room), 1 = invalid
    phantoon_invalid: torch.Tensor
    # -1 = unknown; for a valid door this is its matched partner's index within
    # the opposite direction; for an invalid door this is the opposite direction
    # door count sentinel.
    door_match: torch.Tensor

    def to(self, device: torch.device, non_blocking: bool = False) -> "StepOutcomes":
        return StepOutcomes(
            door_invalid=self.door_invalid.to(device, non_blocking=non_blocking),
            connection_invalid=self.connection_invalid.to(device, non_blocking=non_blocking),
            toilet_invalid=self.toilet_invalid.to(device, non_blocking=non_blocking),
            phantoon_invalid=self.phantoon_invalid.to(device, non_blocking=non_blocking),
            door_match=self.door_match.to(device, non_blocking=non_blocking),
        )

    def slice(self, start: int, end: int) -> "StepOutcomes":
        return StepOutcomes(
            door_invalid=self.door_invalid[start:end],
            connection_invalid=self.connection_invalid[start:end],
            toilet_invalid=self.toilet_invalid[start:end],
            phantoon_invalid=self.phantoon_invalid[start:end],
            door_match=self.door_match[start:end],
        )


@dataclass
class EndOutcomes:
    toilet_crossed_room_idx: torch.Tensor
    avg_frontiers: torch.Tensor
    graph_diameter: torch.Tensor
    active_room_part_mask: torch.Tensor
    save_distance: torch.Tensor
    save_distance_mask: torch.Tensor
    save_to_room_distance: torch.Tensor
    save_to_room_distance_mask: torch.Tensor
    save_from_room_distance: torch.Tensor
    save_from_room_distance_mask: torch.Tensor
    refill_distance: torch.Tensor
    refill_distance_mask: torch.Tensor
    refill_to_room_distance: torch.Tensor
    refill_to_room_distance_mask: torch.Tensor
    refill_from_room_distance: torch.Tensor
    refill_from_room_distance_mask: torch.Tensor
    missing_connect_distance: torch.Tensor
    missing_connect_distance_mask: torch.Tensor
    area_connected_components: torch.Tensor
    area_crossings: torch.Tensor
    area_size: torch.Tensor
    area_map_station_count: torch.Tensor

    def to(self, device: torch.device) -> "EndOutcomes":
        return EndOutcomes(
            toilet_crossed_room_idx=self.toilet_crossed_room_idx.to(device),
            avg_frontiers=self.avg_frontiers.to(device),
            graph_diameter=self.graph_diameter.to(device),
            active_room_part_mask=self.active_room_part_mask.to(device),
            save_distance=self.save_distance.to(device),
            save_distance_mask=self.save_distance_mask.to(device),
            save_to_room_distance=self.save_to_room_distance.to(device),
            save_to_room_distance_mask=self.save_to_room_distance_mask.to(device),
            save_from_room_distance=self.save_from_room_distance.to(device),
            save_from_room_distance_mask=self.save_from_room_distance_mask.to(device),
            refill_distance=self.refill_distance.to(device),
            refill_distance_mask=self.refill_distance_mask.to(device),
            refill_to_room_distance=self.refill_to_room_distance.to(device),
            refill_to_room_distance_mask=self.refill_to_room_distance_mask.to(device),
            refill_from_room_distance=self.refill_from_room_distance.to(device),
            refill_from_room_distance_mask=self.refill_from_room_distance_mask.to(device),
            missing_connect_distance=self.missing_connect_distance.to(device),
            missing_connect_distance_mask=self.missing_connect_distance_mask.to(device),
            area_connected_components=self.area_connected_components.to(device),
            area_crossings=self.area_crossings.to(device),
            area_size=self.area_size.to(device),
            area_map_station_count=self.area_map_station_count.to(device),
        )

    def slice(self, start: int, end: int) -> "EndOutcomes":
        return EndOutcomes(
            toilet_crossed_room_idx=self.toilet_crossed_room_idx[start:end],
            avg_frontiers=self.avg_frontiers[start:end],
            graph_diameter=self.graph_diameter[start:end],
            active_room_part_mask=self.active_room_part_mask[start:end],
            save_distance=self.save_distance[start:end],
            save_distance_mask=self.save_distance_mask[start:end],
            save_to_room_distance=self.save_to_room_distance[start:end],
            save_to_room_distance_mask=self.save_to_room_distance_mask[start:end],
            save_from_room_distance=self.save_from_room_distance[start:end],
            save_from_room_distance_mask=self.save_from_room_distance_mask[start:end],
            refill_distance=self.refill_distance[start:end],
            refill_distance_mask=self.refill_distance_mask[start:end],
            refill_to_room_distance=self.refill_to_room_distance[start:end],
            refill_to_room_distance_mask=self.refill_to_room_distance_mask[start:end],
            refill_from_room_distance=self.refill_from_room_distance[start:end],
            refill_from_room_distance_mask=self.refill_from_room_distance_mask[start:end],
            missing_connect_distance=self.missing_connect_distance[start:end],
            missing_connect_distance_mask=self.missing_connect_distance_mask[start:end],
            area_connected_components=self.area_connected_components[start:end],
            area_crossings=self.area_crossings[start:end],
            area_size=self.area_size[start:end],
            area_map_station_count=self.area_map_station_count[start:end],
        )


@dataclass
class AreaOutcomeState:
    area_connected_components: torch.Tensor
    area_crossings: torch.Tensor
    area_size: torch.Tensor
    area_map_station_count: torch.Tensor


@dataclass
class EpisodeOutcomes:
    step_outcomes: StepOutcomes
    end_outcomes: EndOutcomes

    def to(self, device: torch.device) -> "EpisodeOutcomes":
        return EpisodeOutcomes(
            step_outcomes=self.step_outcomes.to(device),
            end_outcomes=self.end_outcomes.to(device),
        )

    def slice(self, start: int, end: int) -> "EpisodeOutcomes":
        return EpisodeOutcomes(
            step_outcomes=self.step_outcomes.slice(start, end),
            end_outcomes=self.end_outcomes.slice(start, end),
        )


@dataclass
class FeatureRequirements:
    frontier_row_count: int
    worker_frontier_row_counts: list[int]
    missing_connect_query_row_count: int
    worker_missing_connect_query_row_counts: list[int]
    save_refill_utility_query_row_count: int
    worker_save_refill_utility_query_row_counts: list[int]


@dataclass
class ProposalCandidateMask:
    proposal_frontier_idx: torch.Tensor
    mask: torch.Tensor
    valid_counts: torch.Tensor
    proposal_action_count: int

    def to(self, device: torch.device) -> "ProposalCandidateMask":
        return ProposalCandidateMask(
            proposal_frontier_idx=self.proposal_frontier_idx.to(device),
            mask=self.mask.to(device),
            valid_counts=self.valid_counts.to(device),
            proposal_action_count=self.proposal_action_count,
        )


@dataclass
class CandidateStats:
    clean_counts: torch.Tensor
    evaluated_counts: torch.Tensor
    rejected_counts: torch.Tensor

    def to(self, device: torch.device, non_blocking: bool = False) -> "CandidateStats":
        return CandidateStats(
            clean_counts=self.clean_counts.to(device, non_blocking=non_blocking),
            evaluated_counts=self.evaluated_counts.to(device, non_blocking=non_blocking),
            rejected_counts=self.rejected_counts.to(device, non_blocking=non_blocking),
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
        self.room_area = None
        self.proposal_frontier_idx = None
        self.proposal_action_idx = None
        self.pre_door_invalid = None
        self.pre_connection_invalid = None
        self.pre_toilet_invalid = None
        self.pre_phantoon_invalid = None
        self.door_invalid = None
        self.connection_invalid = None
        self.toilet_invalid = None
        self.phantoon_invalid = None
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
        self.room_area = self._empty(candidate_shape, torch.uint8)
        self.proposal_frontier_idx = self._empty(candidate_shape, torch.int16)
        self.proposal_action_idx = self._empty(candidate_shape, torch.int16)
        self.pre_door_invalid = self._empty(
            (self.environment_capacity, self.door_count),
            torch.int8,
        )
        self.pre_connection_invalid = self._empty(
            (self.environment_capacity, self.connection_count),
            torch.int8,
        )
        self.pre_toilet_invalid = self._empty((self.environment_capacity,), torch.int8)
        self.pre_phantoon_invalid = self._empty((self.environment_capacity,), torch.int8)
        self.door_invalid = self._empty(
            (*candidate_shape, self.door_count),
            torch.int8,
        )
        self.connection_invalid = self._empty(
            (*candidate_shape, self.connection_count),
            torch.int8,
        )
        self.toilet_invalid = self._empty(candidate_shape, torch.int8)
        self.phantoon_invalid = self._empty(candidate_shape, torch.int8)
        self.door_match = self._empty((*candidate_shape, self.door_count), torch.int16)
        self.clean_counts = self._empty((self.environment_capacity,), torch.int64)
        self.evaluated_counts = self._empty((self.environment_capacity,), torch.int64)
        self.rejected_counts = self._empty((self.environment_capacity,), torch.int64)

    def actions(self, environment_count: int, candidate_count: int) -> Actions:
        return Actions(
            room_idx=self.room_idx[:environment_count, :candidate_count],
            room_x=self.room_x[:environment_count, :candidate_count],
            room_y=self.room_y[:environment_count, :candidate_count],
            room_area=self.room_area[:environment_count, :candidate_count],
        )

    def proposal_frontiers(
        self,
        environment_count: int,
        candidate_count: int,
    ) -> torch.Tensor:
        return self.proposal_frontier_idx[:environment_count, :candidate_count]

    def proposal_actions(
        self,
        environment_count: int,
        candidate_count: int,
    ) -> torch.Tensor:
        return self.proposal_action_idx[:environment_count, :candidate_count]

    def reward_outcomes(self, environment_count: int) -> StepOutcomes:
        return StepOutcomes(
            door_invalid=self.pre_door_invalid[:environment_count],
            connection_invalid=self.pre_connection_invalid[:environment_count],
            toilet_invalid=self.pre_toilet_invalid[:environment_count],
            phantoon_invalid=self.pre_phantoon_invalid[:environment_count],
            door_match=self.door_match.new_empty((environment_count, 0)),
        )

    def post_candidate_outcomes(
        self,
        environment_count: int,
        candidate_count: int,
    ) -> StepOutcomes:
        return StepOutcomes(
            door_invalid=self.door_invalid[:environment_count, :candidate_count],
            connection_invalid=self.connection_invalid[:environment_count, :candidate_count],
            toilet_invalid=self.toilet_invalid[:environment_count, :candidate_count],
            phantoon_invalid=self.phantoon_invalid[:environment_count, :candidate_count],
            door_match=self.door_match[:environment_count, :candidate_count],
        )

    def stats(self, environment_count: int) -> CandidateStats:
        return CandidateStats(
            clean_counts=self.clean_counts[:environment_count],
            evaluated_counts=self.evaluated_counts[:environment_count],
            rejected_counts=self.rejected_counts[:environment_count],
        )


@dataclass
class DoorMatchCounts:
    horizontal: torch.Tensor
    vertical: torch.Tensor

    def to(self, device: torch.device) -> "DoorMatchCounts":
        return DoorMatchCounts(
            horizontal=self.horizontal.to(device),
            vertical=self.vertical.to(device),
        )


@dataclass
class DoorMatches:
    left: torch.Tensor
    right: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor

    def to(self, device: torch.device) -> "DoorMatches":
        return DoorMatches(
            left=self.left.to(device),
            right=self.right.to(device),
            up=self.up.to(device),
            down=self.down.to(device),
        )

    def slice(self, start: int, end: int) -> "DoorMatches":
        return DoorMatches(
            left=self.left[start:end],
            right=self.right[start:end],
            up=self.up[start:end],
            down=self.down[start:end],
        )


@dataclass
class GlobalFeatures:
    inventory: torch.Tensor
    room_x: torch.Tensor
    room_y: torch.Tensor
    room_placed: torch.Tensor
    room_part_furthest_destination: torch.Tensor
    room_part_furthest_source: torch.Tensor
    room_part_save_from_room_distance: torch.Tensor
    room_part_save_to_room_distance: torch.Tensor
    room_part_refill_from_room_distance: torch.Tensor
    room_part_refill_to_room_distance: torch.Tensor
    room_part_frontier_from_room_distance: torch.Tensor
    room_part_frontier_to_room_distance: torch.Tensor
    known_save_from_room_distance: torch.Tensor
    known_save_to_room_distance: torch.Tensor
    known_refill_from_room_distance: torch.Tensor
    known_refill_to_room_distance: torch.Tensor
    area_used: torch.Tensor
    area_min_x: torch.Tensor
    area_max_x: torch.Tensor
    area_min_y: torch.Tensor
    area_max_y: torch.Tensor
    area_connected_components: torch.Tensor
    area_crossings: torch.Tensor
    area_size: torch.Tensor
    area_map_station_count: torch.Tensor
    log_temperature: torch.Tensor
    log_recommended_candidates: torch.Tensor
    generation_variable_floats: torch.Tensor
    lookahead_door_invalid: torch.Tensor
    lookahead_door_match: torch.Tensor
    lookahead_connection_invalid: torch.Tensor
    lookahead_toilet_invalid: torch.Tensor
    lookahead_phantoon_invalid: torch.Tensor
    connection_reachability: torch.Tensor
    toilet_crossed_room_idx: torch.Tensor

    def to(self, device: torch.device, non_blocking: bool = False) -> "GlobalFeatures":
        return GlobalFeatures(
            inventory=self.inventory.to(device, non_blocking=non_blocking),
            room_x=self.room_x.to(device, non_blocking=non_blocking),
            room_y=self.room_y.to(device, non_blocking=non_blocking),
            room_placed=self.room_placed.to(device, non_blocking=non_blocking),
            room_part_furthest_destination=self.room_part_furthest_destination.to(
                device, non_blocking=non_blocking
            ),
            room_part_furthest_source=self.room_part_furthest_source.to(
                device, non_blocking=non_blocking
            ),
            room_part_save_from_room_distance=self.room_part_save_from_room_distance.to(
                device, non_blocking=non_blocking
            ),
            room_part_save_to_room_distance=self.room_part_save_to_room_distance.to(
                device, non_blocking=non_blocking
            ),
            room_part_refill_from_room_distance=self.room_part_refill_from_room_distance.to(
                device, non_blocking=non_blocking
            ),
            room_part_refill_to_room_distance=self.room_part_refill_to_room_distance.to(
                device, non_blocking=non_blocking
            ),
            room_part_frontier_from_room_distance=self.room_part_frontier_from_room_distance.to(
                device, non_blocking=non_blocking
            ),
            room_part_frontier_to_room_distance=self.room_part_frontier_to_room_distance.to(
                device, non_blocking=non_blocking
            ),
            known_save_from_room_distance=self.known_save_from_room_distance.to(
                device, non_blocking=non_blocking
            ),
            known_save_to_room_distance=self.known_save_to_room_distance.to(
                device, non_blocking=non_blocking
            ),
            known_refill_from_room_distance=self.known_refill_from_room_distance.to(
                device, non_blocking=non_blocking
            ),
            known_refill_to_room_distance=self.known_refill_to_room_distance.to(
                device, non_blocking=non_blocking
            ),
            area_used=self.area_used.to(device, non_blocking=non_blocking),
            area_min_x=self.area_min_x.to(device, non_blocking=non_blocking),
            area_max_x=self.area_max_x.to(device, non_blocking=non_blocking),
            area_min_y=self.area_min_y.to(device, non_blocking=non_blocking),
            area_max_y=self.area_max_y.to(device, non_blocking=non_blocking),
            area_connected_components=self.area_connected_components.to(
                device, non_blocking=non_blocking
            ),
            area_crossings=self.area_crossings.to(device, non_blocking=non_blocking),
            area_size=self.area_size.to(device, non_blocking=non_blocking),
            area_map_station_count=self.area_map_station_count.to(
                device, non_blocking=non_blocking
            ),
            log_temperature=self.log_temperature.to(device, non_blocking=non_blocking),
            log_recommended_candidates=self.log_recommended_candidates.to(
                device, non_blocking=non_blocking
            ),
            generation_variable_floats=self.generation_variable_floats.to(
                device, non_blocking=non_blocking
            ),
            lookahead_door_invalid=self.lookahead_door_invalid.to(
                device, non_blocking=non_blocking
            ),
            lookahead_door_match=self.lookahead_door_match.to(device, non_blocking=non_blocking),
            lookahead_connection_invalid=self.lookahead_connection_invalid.to(
                device, non_blocking=non_blocking
            ),
            lookahead_toilet_invalid=self.lookahead_toilet_invalid.to(
                device, non_blocking=non_blocking
            ),
            lookahead_phantoon_invalid=self.lookahead_phantoon_invalid.to(
                device, non_blocking=non_blocking
            ),
            connection_reachability=self.connection_reachability.to(
                device, non_blocking=non_blocking
            ),
            toilet_crossed_room_idx=self.toilet_crossed_room_idx.to(
                device, non_blocking=non_blocking
            ),
        )

    def flatten_candidates(self) -> "GlobalFeatures":
        return GlobalFeatures(
            inventory=self.inventory.flatten(0, 1),
            room_x=self.room_x.flatten(0, 1),
            room_y=self.room_y.flatten(0, 1),
            room_placed=self.room_placed.flatten(0, 1),
            room_part_furthest_destination=self.room_part_furthest_destination.flatten(0, 1),
            room_part_furthest_source=self.room_part_furthest_source.flatten(0, 1),
            room_part_save_from_room_distance=self.room_part_save_from_room_distance.flatten(0, 1),
            room_part_save_to_room_distance=self.room_part_save_to_room_distance.flatten(0, 1),
            room_part_refill_from_room_distance=self.room_part_refill_from_room_distance.flatten(
                0, 1
            ),
            room_part_refill_to_room_distance=self.room_part_refill_to_room_distance.flatten(0, 1),
            room_part_frontier_from_room_distance=self.room_part_frontier_from_room_distance.flatten(
                0, 1
            ),
            room_part_frontier_to_room_distance=self.room_part_frontier_to_room_distance.flatten(
                0, 1
            ),
            known_save_from_room_distance=self.known_save_from_room_distance.flatten(0, 1),
            known_save_to_room_distance=self.known_save_to_room_distance.flatten(0, 1),
            known_refill_from_room_distance=self.known_refill_from_room_distance.flatten(0, 1),
            known_refill_to_room_distance=self.known_refill_to_room_distance.flatten(0, 1),
            area_used=self.area_used.flatten(0, 1),
            area_min_x=self.area_min_x.flatten(0, 1),
            area_max_x=self.area_max_x.flatten(0, 1),
            area_min_y=self.area_min_y.flatten(0, 1),
            area_max_y=self.area_max_y.flatten(0, 1),
            area_connected_components=self.area_connected_components.flatten(0, 1),
            area_crossings=self.area_crossings.flatten(0, 1),
            area_size=self.area_size.flatten(0, 1),
            area_map_station_count=self.area_map_station_count.flatten(0, 1),
            log_temperature=self.log_temperature.flatten(0, 1),
            log_recommended_candidates=self.log_recommended_candidates.flatten(0, 1),
            generation_variable_floats=self.generation_variable_floats.flatten(0, 1),
            lookahead_door_invalid=self.lookahead_door_invalid.flatten(0, 1),
            lookahead_door_match=self.lookahead_door_match.flatten(0, 1),
            lookahead_connection_invalid=self.lookahead_connection_invalid.flatten(0, 1),
            lookahead_toilet_invalid=self.lookahead_toilet_invalid.flatten(0, 1),
            lookahead_phantoon_invalid=self.lookahead_phantoon_invalid.flatten(0, 1),
            connection_reachability=self.connection_reachability.flatten(0, 1),
            toilet_crossed_room_idx=self.toilet_crossed_room_idx.flatten(0, 1),
        )


@dataclass
class FrontierFeatures:
    frontier: torch.Tensor
    frontier_door_variant: torch.Tensor
    frontier_area: torch.Tensor
    frontier_occupancy: torch.Tensor
    frontier_neighbor: torch.Tensor
    frontier_neighbor_pair: torch.Tensor
    frontier_connection_reachability: torch.Tensor
    row_snapshot_idx: torch.Tensor
    row_frontier_idx: torch.Tensor
    row_door_output_idx: torch.Tensor

    def to(self, device: torch.device, non_blocking: bool = False) -> "FrontierFeatures":
        return FrontierFeatures(
            frontier=self.frontier.to(device, non_blocking=non_blocking),
            frontier_door_variant=self.frontier_door_variant.to(device, non_blocking=non_blocking),
            frontier_area=self.frontier_area.to(device, non_blocking=non_blocking),
            frontier_occupancy=self.frontier_occupancy.to(device, non_blocking=non_blocking),
            frontier_neighbor=self.frontier_neighbor.to(device, non_blocking=non_blocking),
            frontier_neighbor_pair=self.frontier_neighbor_pair.to(
                device, non_blocking=non_blocking
            ),
            frontier_connection_reachability=self.frontier_connection_reachability.to(
                device, non_blocking=non_blocking
            ),
            row_snapshot_idx=self.row_snapshot_idx.to(device, non_blocking=non_blocking),
            row_frontier_idx=self.row_frontier_idx.to(device, non_blocking=non_blocking),
            row_door_output_idx=self.row_door_output_idx.to(device, non_blocking=non_blocking),
        )

    def mark_dynamic(self) -> None:
        torch._dynamo.maybe_mark_dynamic(self.frontier, 0)
        torch._dynamo.maybe_mark_dynamic(self.frontier_door_variant, 0)
        torch._dynamo.maybe_mark_dynamic(self.frontier_area, 0)
        torch._dynamo.maybe_mark_dynamic(self.frontier_occupancy, 0)
        torch._dynamo.maybe_mark_dynamic(self.frontier_neighbor, 0)
        torch._dynamo.maybe_mark_dynamic(self.frontier_neighbor_pair, 0)
        torch._dynamo.maybe_mark_dynamic(self.frontier_connection_reachability, 0)
        torch._dynamo.maybe_mark_dynamic(self.row_snapshot_idx, 0)
        torch._dynamo.maybe_mark_dynamic(self.row_frontier_idx, 0)
        torch._dynamo.maybe_mark_dynamic(self.row_door_output_idx, 0)


@dataclass
class MissingConnectQueryFeatures:
    query_snapshot_idx: torch.Tensor
    query_connection_idx: torch.Tensor
    source_frontier: torch.Tensor
    target_frontier: torch.Tensor
    source_distance: torch.Tensor
    target_distance: torch.Tensor
    current_distance: torch.Tensor

    def to(
        self,
        device: torch.device,
        non_blocking: bool = False,
    ) -> "MissingConnectQueryFeatures":
        return MissingConnectQueryFeatures(
            query_snapshot_idx=self.query_snapshot_idx.to(device, non_blocking=non_blocking),
            query_connection_idx=self.query_connection_idx.to(device, non_blocking=non_blocking),
            source_frontier=self.source_frontier.to(device, non_blocking=non_blocking),
            target_frontier=self.target_frontier.to(device, non_blocking=non_blocking),
            source_distance=self.source_distance.to(device, non_blocking=non_blocking),
            target_distance=self.target_distance.to(device, non_blocking=non_blocking),
            current_distance=self.current_distance.to(device, non_blocking=non_blocking),
        )

    def mark_dynamic(self) -> None:
        torch._dynamo.maybe_mark_dynamic(self.query_snapshot_idx, 0)
        torch._dynamo.maybe_mark_dynamic(self.query_connection_idx, 0)
        torch._dynamo.maybe_mark_dynamic(self.source_frontier, 0)
        torch._dynamo.maybe_mark_dynamic(self.target_frontier, 0)
        torch._dynamo.maybe_mark_dynamic(self.source_distance, 0)
        torch._dynamo.maybe_mark_dynamic(self.target_distance, 0)
        torch._dynamo.maybe_mark_dynamic(self.current_distance, 0)


@dataclass
class SaveRefillUtilityQueryFeatures:
    query_snapshot_idx: torch.Tensor
    query_room_part_idx: torch.Tensor
    target_mask: torch.Tensor
    frontier: torch.Tensor
    frontier_distance: torch.Tensor
    save_to_current_distance: torch.Tensor
    save_from_current_distance: torch.Tensor
    refill_to_current_distance: torch.Tensor
    refill_from_current_distance: torch.Tensor

    def to(
        self,
        device: torch.device,
        non_blocking: bool = False,
    ) -> "SaveRefillUtilityQueryFeatures":
        return SaveRefillUtilityQueryFeatures(
            query_snapshot_idx=self.query_snapshot_idx.to(device, non_blocking=non_blocking),
            query_room_part_idx=self.query_room_part_idx.to(device, non_blocking=non_blocking),
            target_mask=self.target_mask.to(device, non_blocking=non_blocking),
            frontier=self.frontier.to(device, non_blocking=non_blocking),
            frontier_distance=self.frontier_distance.to(device, non_blocking=non_blocking),
            save_to_current_distance=self.save_to_current_distance.to(
                device, non_blocking=non_blocking
            ),
            save_from_current_distance=self.save_from_current_distance.to(
                device, non_blocking=non_blocking
            ),
            refill_to_current_distance=self.refill_to_current_distance.to(
                device, non_blocking=non_blocking
            ),
            refill_from_current_distance=self.refill_from_current_distance.to(
                device, non_blocking=non_blocking
            ),
        )

    def mark_dynamic(self) -> None:
        torch._dynamo.maybe_mark_dynamic(self.query_snapshot_idx, 0)
        torch._dynamo.maybe_mark_dynamic(self.query_room_part_idx, 0)
        torch._dynamo.maybe_mark_dynamic(self.target_mask, 0)
        torch._dynamo.maybe_mark_dynamic(self.frontier, 0)
        torch._dynamo.maybe_mark_dynamic(self.frontier_distance, 0)
        torch._dynamo.maybe_mark_dynamic(self.save_to_current_distance, 0)
        torch._dynamo.maybe_mark_dynamic(self.save_from_current_distance, 0)
        torch._dynamo.maybe_mark_dynamic(self.refill_to_current_distance, 0)
        torch._dynamo.maybe_mark_dynamic(self.refill_from_current_distance, 0)


@dataclass
class Features:
    global_features: GlobalFeatures
    frontier_features: FrontierFeatures
    missing_connect_query_features: MissingConnectQueryFeatures
    save_refill_utility_query_features: SaveRefillUtilityQueryFeatures

    def to(self, device: torch.device, non_blocking: bool = False) -> "Features":
        return Features(
            global_features=self.global_features.to(device, non_blocking=non_blocking),
            frontier_features=self.frontier_features.to(device, non_blocking=non_blocking),
            missing_connect_query_features=self.missing_connect_query_features.to(
                device,
                non_blocking=non_blocking,
            ),
            save_refill_utility_query_features=self.save_refill_utility_query_features.to(
                device,
                non_blocking=non_blocking,
            ),
        )

    def mark_dynamic(self) -> None:
        self.frontier_features.mark_dynamic()
        self.missing_connect_query_features.mark_dynamic()
        self.save_refill_utility_query_features.mark_dynamic()

    def flatten_candidates(self) -> "Features":
        return Features(
            global_features=self.global_features.flatten_candidates(),
            frontier_features=self.frontier_features,
            missing_connect_query_features=self.missing_connect_query_features,
            save_refill_utility_query_features=self.save_refill_utility_query_features,
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
    features: EngineFeatureConfig

    def __init__(self, rooms: list[dict], features: FeatureConfig):
        self.features = features.engine_config()
        self.engine = map_gen.Engine(json.dumps(rooms), self.features.model_dump_json())
        self.rooms = rooms

    def create_environment_group(
        self,
        map_size: tuple[int, int],
        num_envs: int,
        candidate_spatial_cell_size: int,
        area_bounding_box_width: int,
        area_bounding_box_height: int,
        seed: Optional[int] = None,
        frontier_neighbor_count: int = 4,
        frontier_window_size: int = 16,
        num_threads: Optional[int] = None,
        frontier_neighbor_algorithm: Literal[
            "delaunay", "nearest", "nearest-exclusive"
        ] = "delaunay",
    ) -> "EnvironmentGroup":
        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, ()).item())
        env = self.engine.create_environment_group(
            map_size,
            num_envs,
            seed,
            frontier_neighbor_count,
            frontier_window_size,
            candidate_spatial_cell_size,
            area_bounding_box_width,
            area_bounding_box_height,
            num_threads,
            frontier_neighbor_algorithm,
        )
        return EnvironmentGroup(
            self,
            env,
            map_size,
            num_envs,
            frontier_neighbor_count,
            frontier_window_size,
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
        ) = self.engine.get_output_metadata()
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
            actions.room_area.contiguous().cpu().numpy(),
        )

    def step_initial(self):
        self.env.step_initial()

    def step_known(self, actions: Actions):
        self.env.step_known(
            actions.room_idx.contiguous().cpu().numpy(),
            actions.room_x.contiguous().cpu().numpy(),
            actions.room_y.contiguous().cpu().numpy(),
            actions.room_area.contiguous().cpu().numpy(),
        )

    def get_actions(self, device: torch.device) -> Actions:
        room_idx, room_x, room_y, room_area = self.env.get_actions()
        return Actions(
            room_idx=torch.from_numpy(room_idx).to(device),
            room_x=torch.from_numpy(room_x).to(device),
            room_y=torch.from_numpy(room_y).to(device),
            room_area=torch.from_numpy(room_area).to(device),
        )

    def get_proposal_candidate_mask(
        self,
        sampled_frontier_idx: torch.Tensor,
        device: torch.device,
    ) -> ProposalCandidateMask:
        result = self.env.get_proposal_candidate_mask(
            sampled_frontier_idx.contiguous().cpu().numpy(),
        )
        return ProposalCandidateMask(
            proposal_frontier_idx=torch.from_numpy(result.proposal_frontier_idx).to(device),
            mask=torch.from_numpy(result.mask).to(device),
            valid_counts=torch.from_numpy(result.valid_counts).to(
                device=device, dtype=torch.int64
            ),
            proposal_action_count=result.proposal_action_count,
        )

    def extract_candidates_from_proposals(
        self,
        candidate_slot: CandidateSlot,
        sampled_frontier_idx: torch.Tensor,
        sampled_proposal_action_idx: torch.Tensor,
        recommended_candidates: int,
        max_candidate_areas_per_placement: int,
    ) -> tuple[
        Actions,
        torch.Tensor,
        torch.Tensor,
        StepOutcomes,
        StepOutcomes,
        FeatureRequirements,
        CandidateStats,
    ]:
        candidate_count = recommended_candidates
        candidate_slot.ensure(self.num_envs, candidate_count)
        result = self.env.pack_candidates_from_proposals_into(
            map_gen.ProposalCandidateBuffers(
                {
                    "sampled_frontier_idx": sampled_frontier_idx.contiguous().cpu().numpy(),
                    "sampled_proposal_action_idx": sampled_proposal_action_idx.contiguous()
                    .cpu()
                    .numpy(),
                    "recommended_candidates": recommended_candidates,
                    "max_candidate_areas_per_placement": max_candidate_areas_per_placement,
                    "room_idx": candidate_slot.room_idx[: self.num_envs, :candidate_count].numpy(),
                    "room_x": candidate_slot.room_x[: self.num_envs, :candidate_count].numpy(),
                    "room_y": candidate_slot.room_y[: self.num_envs, :candidate_count].numpy(),
                    "room_area": candidate_slot.room_area[
                        : self.num_envs, :candidate_count
                    ].numpy(),
                    "proposal_frontier_idx": candidate_slot.proposal_frontier_idx[
                        : self.num_envs, :candidate_count
                    ].numpy(),
                    "proposal_action_idx": candidate_slot.proposal_action_idx[
                        : self.num_envs, :candidate_count
                    ].numpy(),
                    "pre_door_valid": candidate_slot.pre_door_invalid[: self.num_envs].numpy(),
                    "pre_connections_valid": candidate_slot.pre_connection_invalid[
                        : self.num_envs
                    ].numpy(),
                    "pre_toilet_valid": candidate_slot.pre_toilet_invalid[: self.num_envs].numpy(),
                    "pre_phantoon_valid": candidate_slot.pre_phantoon_invalid[
                        : self.num_envs
                    ].numpy(),
                    "door_valid": candidate_slot.door_invalid[
                        : self.num_envs, :candidate_count
                    ].numpy(),
                    "connections_valid": candidate_slot.connection_invalid[
                        : self.num_envs, :candidate_count
                    ].numpy(),
                    "toilet_valid": candidate_slot.toilet_invalid[
                        : self.num_envs, :candidate_count
                    ].numpy(),
                    "phantoon_valid": candidate_slot.phantoon_invalid[
                        : self.num_envs, :candidate_count
                    ].numpy(),
                    "door_match": candidate_slot.door_match[
                        : self.num_envs, :candidate_count
                    ].numpy(),
                    "clean_counts": candidate_slot.clean_counts[: self.num_envs].numpy(),
                    "evaluated_counts": candidate_slot.evaluated_counts[: self.num_envs].numpy(),
                    "rejected_counts": candidate_slot.rejected_counts[: self.num_envs].numpy(),
                }
            )
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
        StepOutcomes,
        StepOutcomes,
        FeatureRequirements,
        CandidateStats,
    ]:
        return (
            candidate_slot.actions(self.num_envs, candidate_count),
            candidate_slot.proposal_frontiers(self.num_envs, candidate_count),
            candidate_slot.proposal_actions(self.num_envs, candidate_count),
            candidate_slot.reward_outcomes(self.num_envs),
            candidate_slot.post_candidate_outcomes(self.num_envs, candidate_count),
            FeatureRequirements(
                frontier_row_count=feature_requirements.frontier_row_count,
                worker_frontier_row_counts=feature_requirements.worker_frontier_row_counts,
                missing_connect_query_row_count=(
                    feature_requirements.missing_connect_query_row_count
                ),
                worker_missing_connect_query_row_counts=(
                    feature_requirements.worker_missing_connect_query_row_counts
                ),
                save_refill_utility_query_row_count=(
                    feature_requirements.save_refill_utility_query_row_count
                ),
                worker_save_refill_utility_query_row_counts=(
                    feature_requirements.worker_save_refill_utility_query_row_counts
                ),
            ),
            candidate_slot.stats(self.num_envs),
        )

    def get_outcomes(self, device: torch.device, verify_consistency: bool) -> EpisodeOutcomes:
        result = self.env.get_outcomes(verify_consistency)
        return EpisodeOutcomes(
            step_outcomes=StepOutcomes(
                door_invalid=torch.from_numpy(result.step_outcomes.door_valid).to(device),
                connection_invalid=torch.from_numpy(result.step_outcomes.connections_valid).to(
                    device
                ),
                toilet_invalid=torch.from_numpy(result.step_outcomes.toilet_valid).to(device),
                phantoon_invalid=torch.from_numpy(result.step_outcomes.phantoon_valid).to(device),
                door_match=torch.empty(
                    [result.step_outcomes.door_valid.shape[0], 0],
                    dtype=torch.int16,
                    device=device,
                ),
            ),
            end_outcomes=EndOutcomes(
                toilet_crossed_room_idx=torch.from_numpy(
                    result.end_outcomes.toilet_crossed_room_idx
                ).to(
                    device=device,
                    dtype=torch.int64,
                ),
                avg_frontiers=torch.from_numpy(result.end_outcomes.avg_frontiers).to(device),
                graph_diameter=torch.from_numpy(result.end_outcomes.graph_diameter).to(device),
                active_room_part_mask=torch.from_numpy(
                    result.end_outcomes.active_room_part_mask
                ).to(device),
                save_distance=torch.from_numpy(result.end_outcomes.save_distance).to(device),
                save_distance_mask=torch.from_numpy(result.end_outcomes.save_distance_mask).to(
                    device
                ),
                save_to_room_distance=torch.from_numpy(
                    result.end_outcomes.save_to_room_distance
                ).to(device),
                save_to_room_distance_mask=torch.from_numpy(
                    result.end_outcomes.save_to_room_distance_mask
                ).to(device),
                save_from_room_distance=torch.from_numpy(
                    result.end_outcomes.save_from_room_distance
                ).to(device),
                save_from_room_distance_mask=torch.from_numpy(
                    result.end_outcomes.save_from_room_distance_mask
                ).to(device),
                refill_distance=torch.from_numpy(result.end_outcomes.refill_distance).to(device),
                refill_distance_mask=torch.from_numpy(result.end_outcomes.refill_distance_mask).to(
                    device
                ),
                refill_to_room_distance=torch.from_numpy(
                    result.end_outcomes.refill_to_room_distance
                ).to(device),
                refill_to_room_distance_mask=torch.from_numpy(
                    result.end_outcomes.refill_to_room_distance_mask
                ).to(device),
                refill_from_room_distance=torch.from_numpy(
                    result.end_outcomes.refill_from_room_distance
                ).to(device),
                refill_from_room_distance_mask=torch.from_numpy(
                    result.end_outcomes.refill_from_room_distance_mask
                ).to(device),
                missing_connect_distance=torch.from_numpy(
                    result.end_outcomes.missing_connect_distance
                ).to(device),
                missing_connect_distance_mask=torch.from_numpy(
                    result.end_outcomes.missing_connect_distance_mask
                ).to(device),
                area_connected_components=torch.from_numpy(
                    result.end_outcomes.area_connected_components
                ).to(device=device, dtype=torch.int64),
                area_crossings=torch.from_numpy(result.end_outcomes.area_crossings).to(
                    device=device,
                    dtype=torch.int64,
                ),
                area_size=torch.from_numpy(result.end_outcomes.area_size).to(
                    device=device,
                    dtype=torch.int64,
                ),
                area_map_station_count=torch.from_numpy(
                    result.end_outcomes.area_map_station_count
                ).to(device=device, dtype=torch.int64),
            ),
        )

    def get_area_outcome_state(self, device: torch.device) -> AreaOutcomeState:
        result = self.env.get_area_outcome_state()
        return AreaOutcomeState(
            area_connected_components=torch.from_numpy(
                result.area_connected_components
            ).to(device=device, dtype=torch.int64),
            area_crossings=torch.from_numpy(result.area_crossings).to(
                device=device,
                dtype=torch.int64,
            ),
            area_size=torch.from_numpy(result.area_size).to(device=device, dtype=torch.int64),
            area_map_station_count=torch.from_numpy(result.area_map_station_count).to(
                device=device,
                dtype=torch.int64,
            ),
        )

    def get_current_feature_outcomes(
        self,
        device: torch.device,
        environment_start: int,
        environment_count: int,
    ) -> StepOutcomes:
        door_invalid, connection_invalid, toilet_invalid, phantoon_invalid, door_match = (
            self.env.get_current_feature_outcomes(
                environment_start,
                environment_count,
            )
        )
        return StepOutcomes(
            door_invalid=torch.from_numpy(door_invalid).to(device),
            connection_invalid=torch.from_numpy(connection_invalid).to(device),
            toilet_invalid=torch.from_numpy(toilet_invalid).to(device),
            phantoon_invalid=torch.from_numpy(phantoon_invalid).to(device),
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

    def get_feature_requirements(
        self,
        environment_start: int = 0,
        environment_count: Optional[int] = None,
    ) -> FeatureRequirements:
        result = self.env.get_feature_requirements(
            environment_start,
            environment_count,
        )
        return FeatureRequirements(
            frontier_row_count=result.frontier_row_count,
            worker_frontier_row_counts=result.worker_frontier_row_counts,
            missing_connect_query_row_count=result.missing_connect_query_row_count,
            worker_missing_connect_query_row_counts=(
                result.worker_missing_connect_query_row_counts
            ),
            save_refill_utility_query_row_count=(result.save_refill_utility_query_row_count),
            worker_save_refill_utility_query_row_counts=(
                result.worker_save_refill_utility_query_row_counts
            ),
        )

    def extract_features(
        self,
        feature_slot: "FeatureSlot",
        log_temperature: torch.Tensor,
        include_temperature: bool,
        log_recommended_candidates: torch.Tensor,
        include_recommended_candidates: bool,
        generation_variable_floats: torch.Tensor,
        include_generation_variable_floats: bool,
        lookahead_outcomes: StepOutcomes,
        include_lookahead_outcomes: bool,
        environment_start: int = 0,
        environment_count: Optional[int] = None,
    ) -> Features:
        if environment_count is None:
            environment_count = self.num_envs - environment_start
        feature_requirements = self.get_feature_requirements(
            environment_start,
            environment_count,
        )
        feature_slot.ensure(
            environment_count,
            feature_requirements.frontier_row_count,
            feature_requirements.missing_connect_query_row_count,
            feature_requirements.save_refill_utility_query_row_count,
        )
        self.env.pack_features_into(
            map_gen.FeatureBuffers(
                {
                    "environment_count": environment_count,
                    "candidate_count": 1,
                    "environment_start": environment_start,
                    "frontier_row_count": feature_requirements.frontier_row_count,
                    "worker_frontier_row_counts": feature_requirements.worker_frontier_row_counts,
                    "missing_connect_query_row_count": (
                        feature_requirements.missing_connect_query_row_count
                    ),
                    "worker_missing_connect_query_row_counts": (
                        feature_requirements.worker_missing_connect_query_row_counts
                    ),
                    "save_refill_utility_query_row_count": (
                        feature_requirements.save_refill_utility_query_row_count
                    ),
                    "worker_save_refill_utility_query_row_counts": (
                        feature_requirements.worker_save_refill_utility_query_row_counts
                    ),
                    "inventory": feature_slot.inventory.numpy(),
                    "room_x": feature_slot.room_x.numpy(),
                    "room_y": feature_slot.room_y.numpy(),
                    "room_placed": feature_slot.room_placed.numpy(),
                    "room_part_furthest_destination": feature_slot.room_part_furthest_destination.numpy(),
                    "room_part_furthest_source": feature_slot.room_part_furthest_source.numpy(),
                    "room_part_save_from_room_distance": (
                        feature_slot.room_part_save_from_room_distance.numpy()
                    ),
                    "room_part_save_to_room_distance": (
                        feature_slot.room_part_save_to_room_distance.numpy()
                    ),
                    "room_part_refill_from_room_distance": (
                        feature_slot.room_part_refill_from_room_distance.numpy()
                    ),
                    "room_part_refill_to_room_distance": (
                        feature_slot.room_part_refill_to_room_distance.numpy()
                    ),
                    "room_part_frontier_from_room_distance": (
                        feature_slot.room_part_frontier_from_room_distance.numpy()
                    ),
                    "room_part_frontier_to_room_distance": (
                        feature_slot.room_part_frontier_to_room_distance.numpy()
                    ),
                    "known_save_from_room_distance": (
                        feature_slot.known_save_from_room_distance.numpy()
                    ),
                    "known_save_to_room_distance": (
                        feature_slot.known_save_to_room_distance.numpy()
                    ),
                    "known_refill_from_room_distance": (
                        feature_slot.known_refill_from_room_distance.numpy()
                    ),
                    "known_refill_to_room_distance": (
                        feature_slot.known_refill_to_room_distance.numpy()
                    ),
                    "area_used": feature_slot.area_used.numpy(),
                    "area_min_x": feature_slot.area_min_x.numpy(),
                    "area_max_x": feature_slot.area_max_x.numpy(),
                    "area_min_y": feature_slot.area_min_y.numpy(),
                    "area_max_y": feature_slot.area_max_y.numpy(),
                    "area_connected_components": (
                        feature_slot.area_connected_components.numpy()
                    ),
                    "area_crossings": feature_slot.area_crossings.numpy(),
                    "area_size": feature_slot.area_size.numpy(),
                    "area_map_station_count": feature_slot.area_map_station_count.numpy(),
                    "frontier": feature_slot.frontier.numpy(),
                    "frontier_door_variant": feature_slot.frontier_door_variant.numpy(),
                    "frontier_area": feature_slot.frontier_area.numpy(),
                    "frontier_occupancy": feature_slot.frontier_occupancy.numpy(),
                    "frontier_neighbor": feature_slot.frontier_neighbor.numpy(),
                    "frontier_neighbor_pair": feature_slot.frontier_neighbor_pair.numpy(),
                    "connection_reachability": feature_slot.connection_reachability.numpy(),
                    "frontier_connection_reachability": feature_slot.frontier_connection_reachability.numpy(),
                    "missing_connect_query_snapshot_idx": (
                        feature_slot.missing_connect_query_snapshot_idx.numpy()
                    ),
                    "missing_connect_query_connection_idx": (
                        feature_slot.missing_connect_query_connection_idx.numpy()
                    ),
                    "missing_connect_query_source_frontier": (
                        feature_slot.missing_connect_query_source_frontier.numpy()
                    ),
                    "missing_connect_query_target_frontier": (
                        feature_slot.missing_connect_query_target_frontier.numpy()
                    ),
                    "missing_connect_query_source_distance": (
                        feature_slot.missing_connect_query_source_distance.numpy()
                    ),
                    "missing_connect_query_target_distance": (
                        feature_slot.missing_connect_query_target_distance.numpy()
                    ),
                    "missing_connect_query_current_distance": (
                        feature_slot.missing_connect_query_current_distance.numpy()
                    ),
                    "save_refill_utility_query_snapshot_idx": (
                        feature_slot.save_refill_utility_query_snapshot_idx.numpy()
                    ),
                    "save_refill_utility_query_room_part_idx": (
                        feature_slot.save_refill_utility_query_room_part_idx.numpy()
                    ),
                    "save_refill_utility_query_target_mask": (
                        feature_slot.save_refill_utility_query_target_mask.numpy()
                    ),
                    "save_refill_utility_query_frontier": (
                        feature_slot.save_refill_utility_query_frontier.numpy()
                    ),
                    "save_refill_utility_query_frontier_distance": (
                        feature_slot.save_refill_utility_query_frontier_distance.numpy()
                    ),
                    "save_refill_utility_query_save_to_current_distance": (
                        feature_slot.save_refill_utility_query_save_to_current_distance.numpy()
                    ),
                    "save_refill_utility_query_save_from_current_distance": (
                        feature_slot.save_refill_utility_query_save_from_current_distance.numpy()
                    ),
                    "save_refill_utility_query_refill_to_current_distance": (
                        feature_slot.save_refill_utility_query_refill_to_current_distance.numpy()
                    ),
                    "save_refill_utility_query_refill_from_current_distance": (
                        feature_slot.save_refill_utility_query_refill_from_current_distance.numpy()
                    ),
                    "toilet_crossed_room_idx": feature_slot.toilet_crossed_room_idx.numpy(),
                    "row_snapshot_idx": feature_slot.row_snapshot_idx.numpy(),
                    "row_frontier_idx": feature_slot.row_frontier_idx.numpy(),
                    "row_door_output_idx": feature_slot.row_door_output_idx.numpy(),
                }
            )
        )
        return feature_slot.state_features(
            environment_count,
            log_temperature,
            include_temperature,
            log_recommended_candidates,
            include_recommended_candidates,
            generation_variable_floats,
            include_generation_variable_floats,
            lookahead_outcomes,
            include_lookahead_outcomes,
            feature_requirements.frontier_row_count,
            feature_requirements.missing_connect_query_row_count,
            feature_requirements.save_refill_utility_query_row_count,
        )

    def finish(self):
        self.env.finish()


# When a GPU is available, we use pinned memory for model input tensors,
# to allow for asynchronous CPU-to-GPU transfers.
class FeatureSlot:
    def __init__(self, env: EnvironmentGroup, pin_memory: bool):
        features = env.engine.features
        inventory_count, _, room_count = env.engine.get_feature_sizes()
        _, connection_count = env.engine.get_output_sizes()
        room_part_count = env.engine.get_output_metadata().num_room_parts
        self.inventory_width = inventory_count * int(features.inventory)
        self.room_width = room_count * int(features.room_position)
        self.room_part_width = room_part_count * int(features.room_part_furthest_distance)
        self.room_part_save_distance_width = room_part_count * int(
            features.room_part_save_distance
        )
        self.room_part_refill_distance_width = room_part_count * int(
            features.room_part_refill_distance
        )
        self.room_part_frontier_distance_width = room_part_count * int(
            features.room_part_frontier_distance
        )
        self.known_distance_width = room_part_count
        self.area_width = AREA_COUNT * int(features.area_state)
        self.area_crossings_width = int(features.area_state)
        self.frontier_occupancy_width = (
            (env.frontier_window_size * env.frontier_window_size + 7) // 8
        ) * int(features.frontier_occupancy)
        self.frontier_neighbor_width = env.frontier_neighbor_count * int(
            features.frontier_neighbor
        )
        self.frontier_neighbor_pair_width = env.frontier_neighbor_count * int(
            features.frontier_neighbor_flags
        )
        self.connection_reachability_width = connection_count * int(
            features.connection_reachability
        )
        self.frontier_connection_reachability_width = connection_count * int(
            features.frontier_connection_reachability
        )
        self.missing_connect_query_frontier_width = int(features.missing_connect_query)
        self.toilet_crossed_room_width = int(features.toilet_crossed_room)
        self.pin_memory = pin_memory
        self.snapshot_capacity = 0
        self.frontier_row_capacity = 0
        self.missing_connect_query_row_capacity = 0
        self.save_refill_utility_query_row_capacity = 0
        self.inventory = None
        self.room_x = None
        self.room_y = None
        self.room_placed = None
        self.room_part_furthest_destination = None
        self.room_part_furthest_source = None
        self.room_part_save_from_room_distance = None
        self.room_part_save_to_room_distance = None
        self.room_part_refill_from_room_distance = None
        self.room_part_refill_to_room_distance = None
        self.room_part_frontier_from_room_distance = None
        self.room_part_frontier_to_room_distance = None
        self.known_save_from_room_distance = None
        self.known_save_to_room_distance = None
        self.known_refill_from_room_distance = None
        self.known_refill_to_room_distance = None
        self.area_used = None
        self.area_min_x = None
        self.area_max_x = None
        self.area_min_y = None
        self.area_max_y = None
        self.area_connected_components = None
        self.area_crossings = None
        self.area_size = None
        self.area_map_station_count = None
        self.frontier = None
        self.frontier_door_variant = None
        self.frontier_area = None
        self.frontier_occupancy = None
        self.frontier_neighbor = None
        self.frontier_neighbor_pair = None
        self.connection_reachability = None
        self.frontier_connection_reachability = None
        self.missing_connect_query_snapshot_idx = None
        self.missing_connect_query_connection_idx = None
        self.missing_connect_query_source_frontier = None
        self.missing_connect_query_target_frontier = None
        self.missing_connect_query_source_distance = None
        self.missing_connect_query_target_distance = None
        self.missing_connect_query_current_distance = None
        self.save_refill_utility_query_snapshot_idx = None
        self.save_refill_utility_query_room_part_idx = None
        self.save_refill_utility_query_target_mask = None
        self.save_refill_utility_query_frontier = None
        self.save_refill_utility_query_frontier_distance = None
        self.save_refill_utility_query_save_to_current_distance = None
        self.save_refill_utility_query_save_from_current_distance = None
        self.save_refill_utility_query_refill_to_current_distance = None
        self.save_refill_utility_query_refill_from_current_distance = None
        self.toilet_crossed_room_idx = None
        self.row_snapshot_idx = None
        self.row_frontier_idx = None
        self.row_door_output_idx = None

    def _empty(self, shape, dtype):
        return torch.empty(shape, dtype=dtype, pin_memory=self.pin_memory)

    def ensure(
        self,
        snapshot_count: int,
        frontier_row_count: int,
        missing_connect_query_row_count: int,
        save_refill_utility_query_row_count: int,
    ):
        if (
            self.inventory is not None
            and self.snapshot_capacity >= snapshot_count
            and self.frontier_row_capacity >= frontier_row_count
            and self.missing_connect_query_row_capacity >= missing_connect_query_row_count
            and self.save_refill_utility_query_row_capacity >= save_refill_utility_query_row_count
        ):
            return
        self.snapshot_capacity = max(self.snapshot_capacity, snapshot_count)
        self.frontier_row_capacity = max(self.frontier_row_capacity, frontier_row_count)
        self.missing_connect_query_row_capacity = max(
            self.missing_connect_query_row_capacity,
            missing_connect_query_row_count,
        )
        self.save_refill_utility_query_row_capacity = max(
            self.save_refill_utility_query_row_capacity,
            save_refill_utility_query_row_count,
        )
        self.inventory = self._empty((self.snapshot_capacity, self.inventory_width), torch.uint8)
        self.room_x = self._empty((self.snapshot_capacity, self.room_width), torch.int8)
        self.room_y = self._empty((self.snapshot_capacity, self.room_width), torch.int8)
        self.room_placed = self._empty((self.snapshot_capacity, self.room_width), torch.uint8)
        self.room_part_furthest_destination = self._empty(
            (self.snapshot_capacity, self.room_part_width), torch.uint8
        )
        self.room_part_furthest_source = self._empty(
            (self.snapshot_capacity, self.room_part_width), torch.uint8
        )
        self.room_part_save_from_room_distance = self._empty(
            (self.snapshot_capacity, self.room_part_save_distance_width), torch.uint8
        )
        self.room_part_save_to_room_distance = self._empty(
            (self.snapshot_capacity, self.room_part_save_distance_width), torch.uint8
        )
        self.room_part_refill_from_room_distance = self._empty(
            (self.snapshot_capacity, self.room_part_refill_distance_width), torch.uint8
        )
        self.room_part_refill_to_room_distance = self._empty(
            (self.snapshot_capacity, self.room_part_refill_distance_width), torch.uint8
        )
        self.room_part_frontier_from_room_distance = self._empty(
            (self.snapshot_capacity, self.room_part_frontier_distance_width), torch.uint8
        )
        self.room_part_frontier_to_room_distance = self._empty(
            (self.snapshot_capacity, self.room_part_frontier_distance_width), torch.uint8
        )
        self.known_save_from_room_distance = self._empty(
            (self.snapshot_capacity, self.known_distance_width), torch.uint8
        )
        self.known_save_to_room_distance = self._empty(
            (self.snapshot_capacity, self.known_distance_width), torch.uint8
        )
        self.known_refill_from_room_distance = self._empty(
            (self.snapshot_capacity, self.known_distance_width), torch.uint8
        )
        self.known_refill_to_room_distance = self._empty(
            (self.snapshot_capacity, self.known_distance_width), torch.uint8
        )
        self.area_used = self._empty((self.snapshot_capacity, self.area_width), torch.uint8)
        self.area_min_x = self._empty((self.snapshot_capacity, self.area_width), torch.int8)
        self.area_max_x = self._empty((self.snapshot_capacity, self.area_width), torch.int8)
        self.area_min_y = self._empty((self.snapshot_capacity, self.area_width), torch.int8)
        self.area_max_y = self._empty((self.snapshot_capacity, self.area_width), torch.int8)
        self.area_connected_components = self._empty(
            (self.snapshot_capacity, self.area_width),
            torch.uint8,
        )
        self.area_crossings = self._empty(
            (self.snapshot_capacity, self.area_crossings_width),
            torch.uint16,
        )
        self.area_size = self._empty((self.snapshot_capacity, self.area_width), torch.uint16)
        self.area_map_station_count = self._empty(
            (self.snapshot_capacity, self.area_width),
            torch.uint8,
        )
        self.frontier = self._empty((self.frontier_row_capacity, 5), torch.int8)
        self.frontier_door_variant = self._empty((self.frontier_row_capacity,), torch.int16)
        self.frontier_area = self._empty((self.frontier_row_capacity,), torch.uint8)
        self.frontier_occupancy = self._empty(
            (self.frontier_row_capacity, self.frontier_occupancy_width), torch.uint8
        )
        self.frontier_neighbor = self._empty(
            (self.frontier_row_capacity, self.frontier_neighbor_width), torch.int16
        )
        self.frontier_neighbor_pair = self._empty(
            (self.frontier_row_capacity, self.frontier_neighbor_pair_width), torch.uint8
        )
        self.connection_reachability = self._empty(
            (self.snapshot_capacity, self.connection_reachability_width), torch.uint8
        )
        self.frontier_connection_reachability = self._empty(
            (self.frontier_row_capacity, self.frontier_connection_reachability_width),
            torch.uint8,
        )
        self.missing_connect_query_snapshot_idx = self._empty(
            (self.missing_connect_query_row_capacity,),
            torch.int64,
        )
        self.missing_connect_query_connection_idx = self._empty(
            (self.missing_connect_query_row_capacity,),
            torch.int64,
        )
        self.missing_connect_query_source_frontier = self._empty(
            (
                self.missing_connect_query_row_capacity,
                self.missing_connect_query_frontier_width,
            ),
            torch.int16,
        )
        self.missing_connect_query_target_frontier = self._empty(
            (
                self.missing_connect_query_row_capacity,
                self.missing_connect_query_frontier_width,
            ),
            torch.int16,
        )
        self.missing_connect_query_source_distance = self._empty(
            (
                self.missing_connect_query_row_capacity,
                self.missing_connect_query_frontier_width,
            ),
            torch.uint8,
        )
        self.missing_connect_query_target_distance = self._empty(
            (
                self.missing_connect_query_row_capacity,
                self.missing_connect_query_frontier_width,
            ),
            torch.uint8,
        )
        self.missing_connect_query_current_distance = self._empty(
            (self.missing_connect_query_row_capacity,),
            torch.uint8,
        )
        self.save_refill_utility_query_snapshot_idx = self._empty(
            (self.save_refill_utility_query_row_capacity,),
            torch.int64,
        )
        self.save_refill_utility_query_room_part_idx = self._empty(
            (self.save_refill_utility_query_row_capacity,),
            torch.int64,
        )
        self.save_refill_utility_query_target_mask = self._empty(
            (self.save_refill_utility_query_row_capacity,),
            torch.uint8,
        )
        self.save_refill_utility_query_frontier = self._empty(
            (self.save_refill_utility_query_row_capacity,),
            torch.int16,
        )
        self.save_refill_utility_query_frontier_distance = self._empty(
            (self.save_refill_utility_query_row_capacity,),
            torch.uint8,
        )
        self.save_refill_utility_query_save_to_current_distance = self._empty(
            (self.save_refill_utility_query_row_capacity,),
            torch.uint8,
        )
        self.save_refill_utility_query_save_from_current_distance = self._empty(
            (self.save_refill_utility_query_row_capacity,),
            torch.uint8,
        )
        self.save_refill_utility_query_refill_to_current_distance = self._empty(
            (self.save_refill_utility_query_row_capacity,),
            torch.uint8,
        )
        self.save_refill_utility_query_refill_from_current_distance = self._empty(
            (self.save_refill_utility_query_row_capacity,),
            torch.uint8,
        )
        self.toilet_crossed_room_idx = self._empty(
            (self.snapshot_capacity, self.toilet_crossed_room_width),
            torch.int16,
        )
        self.row_snapshot_idx = self._empty((self.frontier_row_capacity,), torch.int64)
        self.row_frontier_idx = self._empty((self.frontier_row_capacity,), torch.int16)
        self.row_door_output_idx = self._empty((self.frontier_row_capacity,), torch.int16)

    def state_features(
        self,
        environment_count: int,
        log_temperature: torch.Tensor,
        include_temperature: bool,
        log_recommended_candidates: torch.Tensor,
        include_recommended_candidates: bool,
        generation_variable_floats: torch.Tensor,
        include_generation_variable_floats: bool,
        lookahead_outcomes: StepOutcomes,
        include_lookahead_outcomes: bool,
        frontier_row_count: int,
        missing_connect_query_row_count: int,
        save_refill_utility_query_row_count: int,
    ) -> Features:
        if not include_temperature:
            log_temperature = log_temperature.new_empty([*log_temperature.shape, 0])
        if not include_recommended_candidates:
            log_recommended_candidates = log_recommended_candidates.new_empty(
                [
                    *log_recommended_candidates.shape,
                    0,
                ]
            )
        if not include_generation_variable_floats:
            generation_variable_floats = generation_variable_floats.new_empty(
                [*generation_variable_floats.shape[:-1], 0]
            )
        lookahead_door_invalid = lookahead_outcomes.door_invalid
        lookahead_door_match = lookahead_outcomes.door_match
        lookahead_connection_invalid = lookahead_outcomes.connection_invalid
        lookahead_toilet_invalid = lookahead_outcomes.toilet_invalid
        lookahead_phantoon_invalid = lookahead_outcomes.phantoon_invalid
        if not include_lookahead_outcomes:
            lookahead_door_invalid = lookahead_door_invalid.new_empty(
                [
                    *lookahead_door_invalid.shape[:-1],
                    0,
                ]
            )
            lookahead_door_match = lookahead_door_match.new_empty(
                [
                    *lookahead_door_match.shape[:-1],
                    0,
                ]
            )
            lookahead_connection_invalid = lookahead_connection_invalid.new_empty(
                [
                    *lookahead_connection_invalid.shape[:-1],
                    0,
                ]
            )
            lookahead_toilet_invalid = lookahead_toilet_invalid.new_empty(
                [
                    *lookahead_toilet_invalid.shape,
                    0,
                ]
            )
            lookahead_phantoon_invalid = lookahead_phantoon_invalid.new_empty(
                [
                    *lookahead_phantoon_invalid.shape,
                    0,
                ]
            )
        return Features(
            global_features=GlobalFeatures(
                inventory=self.inventory[:environment_count],
                room_x=self.room_x[:environment_count],
                room_y=self.room_y[:environment_count],
                room_placed=self.room_placed[:environment_count],
                room_part_furthest_destination=self.room_part_furthest_destination[
                    :environment_count
                ],
                room_part_furthest_source=self.room_part_furthest_source[:environment_count],
                room_part_save_from_room_distance=self.room_part_save_from_room_distance[
                    :environment_count
                ],
                room_part_save_to_room_distance=self.room_part_save_to_room_distance[
                    :environment_count
                ],
                room_part_refill_from_room_distance=self.room_part_refill_from_room_distance[
                    :environment_count
                ],
                room_part_refill_to_room_distance=self.room_part_refill_to_room_distance[
                    :environment_count
                ],
                room_part_frontier_from_room_distance=self.room_part_frontier_from_room_distance[
                    :environment_count
                ],
                room_part_frontier_to_room_distance=self.room_part_frontier_to_room_distance[
                    :environment_count
                ],
                known_save_from_room_distance=self.known_save_from_room_distance[
                    :environment_count
                ],
                known_save_to_room_distance=self.known_save_to_room_distance[:environment_count],
                known_refill_from_room_distance=self.known_refill_from_room_distance[
                    :environment_count
                ],
                known_refill_to_room_distance=self.known_refill_to_room_distance[
                    :environment_count
                ],
                area_used=self.area_used[:environment_count],
                area_min_x=self.area_min_x[:environment_count],
                area_max_x=self.area_max_x[:environment_count],
                area_min_y=self.area_min_y[:environment_count],
                area_max_y=self.area_max_y[:environment_count],
                area_connected_components=self.area_connected_components[:environment_count],
                area_crossings=self.area_crossings[:environment_count],
                area_size=self.area_size[:environment_count],
                area_map_station_count=self.area_map_station_count[:environment_count],
                log_temperature=log_temperature,
                log_recommended_candidates=log_recommended_candidates,
                generation_variable_floats=generation_variable_floats,
                lookahead_door_invalid=lookahead_door_invalid,
                lookahead_door_match=lookahead_door_match,
                lookahead_connection_invalid=lookahead_connection_invalid,
                lookahead_toilet_invalid=lookahead_toilet_invalid,
                lookahead_phantoon_invalid=lookahead_phantoon_invalid,
                connection_reachability=self.connection_reachability[:environment_count],
                toilet_crossed_room_idx=self.toilet_crossed_room_idx[:environment_count],
            ),
            frontier_features=FrontierFeatures(
                frontier=self.frontier[:frontier_row_count],
                frontier_door_variant=self.frontier_door_variant[:frontier_row_count],
                frontier_area=self.frontier_area[:frontier_row_count],
                frontier_occupancy=self.frontier_occupancy[:frontier_row_count],
                frontier_neighbor=self.frontier_neighbor[:frontier_row_count],
                frontier_neighbor_pair=self.frontier_neighbor_pair[:frontier_row_count],
                frontier_connection_reachability=self.frontier_connection_reachability[
                    :frontier_row_count
                ],
                row_snapshot_idx=self.row_snapshot_idx[:frontier_row_count],
                row_frontier_idx=self.row_frontier_idx[:frontier_row_count],
                row_door_output_idx=self.row_door_output_idx[:frontier_row_count],
            ),
            missing_connect_query_features=MissingConnectQueryFeatures(
                query_snapshot_idx=self.missing_connect_query_snapshot_idx[
                    :missing_connect_query_row_count
                ],
                query_connection_idx=self.missing_connect_query_connection_idx[
                    :missing_connect_query_row_count
                ],
                source_frontier=self.missing_connect_query_source_frontier[
                    :missing_connect_query_row_count
                ],
                target_frontier=self.missing_connect_query_target_frontier[
                    :missing_connect_query_row_count
                ],
                source_distance=self.missing_connect_query_source_distance[
                    :missing_connect_query_row_count
                ],
                target_distance=self.missing_connect_query_target_distance[
                    :missing_connect_query_row_count
                ],
                current_distance=self.missing_connect_query_current_distance[
                    :missing_connect_query_row_count
                ],
            ),
            save_refill_utility_query_features=SaveRefillUtilityQueryFeatures(
                query_snapshot_idx=self.save_refill_utility_query_snapshot_idx[
                    :save_refill_utility_query_row_count
                ],
                query_room_part_idx=self.save_refill_utility_query_room_part_idx[
                    :save_refill_utility_query_row_count
                ],
                target_mask=self.save_refill_utility_query_target_mask[
                    :save_refill_utility_query_row_count
                ],
                frontier=self.save_refill_utility_query_frontier[
                    :save_refill_utility_query_row_count
                ],
                frontier_distance=self.save_refill_utility_query_frontier_distance[
                    :save_refill_utility_query_row_count
                ],
                save_to_current_distance=self.save_refill_utility_query_save_to_current_distance[
                    :save_refill_utility_query_row_count
                ],
                save_from_current_distance=self.save_refill_utility_query_save_from_current_distance[
                    :save_refill_utility_query_row_count
                ],
                refill_to_current_distance=self.save_refill_utility_query_refill_to_current_distance[
                    :save_refill_utility_query_row_count
                ],
                refill_from_current_distance=self.save_refill_utility_query_refill_from_current_distance[
                    :save_refill_utility_query_row_count
                ],
            ),
        )

    def features(
        self,
        environment_count: int,
        candidate_count: int,
        log_temperature: torch.Tensor,
        include_temperature: bool,
        log_recommended_candidates: torch.Tensor,
        include_recommended_candidates: bool,
        generation_variable_floats: torch.Tensor,
        include_generation_variable_floats: bool,
        lookahead_outcomes: StepOutcomes,
        include_lookahead_outcomes: bool,
        frontier_row_count: int,
        missing_connect_query_row_count: int,
        save_refill_utility_query_row_count: int,
    ) -> Features:
        snapshot_count = environment_count * candidate_count
        if not include_temperature:
            log_temperature = log_temperature.new_empty([environment_count, candidate_count, 0])
        if not include_recommended_candidates:
            log_recommended_candidates = log_recommended_candidates.new_empty(
                [environment_count, candidate_count, 0]
            )
        if not include_generation_variable_floats:
            generation_variable_floats = generation_variable_floats.new_empty(
                [environment_count, candidate_count, 0]
            )
        lookahead_door_invalid = lookahead_outcomes.door_invalid
        lookahead_door_match = lookahead_outcomes.door_match
        lookahead_connection_invalid = lookahead_outcomes.connection_invalid
        lookahead_toilet_invalid = lookahead_outcomes.toilet_invalid
        lookahead_phantoon_invalid = lookahead_outcomes.phantoon_invalid
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
            lookahead_phantoon_invalid = lookahead_phantoon_invalid.new_empty(
                [environment_count, candidate_count, 0]
            )
        return Features(
            global_features=GlobalFeatures(
                inventory=self.inventory[:snapshot_count].view(
                    environment_count, candidate_count, self.inventory_width
                ),
                room_x=self.room_x[:snapshot_count].view(
                    environment_count, candidate_count, self.room_width
                ),
                room_y=self.room_y[:snapshot_count].view(
                    environment_count, candidate_count, self.room_width
                ),
                room_placed=self.room_placed[:snapshot_count].view(
                    environment_count, candidate_count, self.room_width
                ),
                room_part_furthest_destination=self.room_part_furthest_destination[
                    :snapshot_count
                ].view(environment_count, candidate_count, self.room_part_width),
                room_part_furthest_source=self.room_part_furthest_source[:snapshot_count].view(
                    environment_count, candidate_count, self.room_part_width
                ),
                room_part_save_from_room_distance=self.room_part_save_from_room_distance[
                    :snapshot_count
                ].view(environment_count, candidate_count, self.room_part_save_distance_width),
                room_part_save_to_room_distance=self.room_part_save_to_room_distance[
                    :snapshot_count
                ].view(environment_count, candidate_count, self.room_part_save_distance_width),
                room_part_refill_from_room_distance=self.room_part_refill_from_room_distance[
                    :snapshot_count
                ].view(environment_count, candidate_count, self.room_part_refill_distance_width),
                room_part_refill_to_room_distance=self.room_part_refill_to_room_distance[
                    :snapshot_count
                ].view(environment_count, candidate_count, self.room_part_refill_distance_width),
                room_part_frontier_from_room_distance=self.room_part_frontier_from_room_distance[
                    :snapshot_count
                ].view(environment_count, candidate_count, self.room_part_frontier_distance_width),
                room_part_frontier_to_room_distance=self.room_part_frontier_to_room_distance[
                    :snapshot_count
                ].view(environment_count, candidate_count, self.room_part_frontier_distance_width),
                known_save_from_room_distance=self.known_save_from_room_distance[
                    :snapshot_count
                ].view(environment_count, candidate_count, self.known_distance_width),
                known_save_to_room_distance=self.known_save_to_room_distance[:snapshot_count].view(
                    environment_count, candidate_count, self.known_distance_width
                ),
                known_refill_from_room_distance=self.known_refill_from_room_distance[
                    :snapshot_count
                ].view(environment_count, candidate_count, self.known_distance_width),
                known_refill_to_room_distance=self.known_refill_to_room_distance[
                    :snapshot_count
                ].view(environment_count, candidate_count, self.known_distance_width),
                area_used=self.area_used[:snapshot_count].view(
                    environment_count, candidate_count, self.area_width
                ),
                area_min_x=self.area_min_x[:snapshot_count].view(
                    environment_count, candidate_count, self.area_width
                ),
                area_max_x=self.area_max_x[:snapshot_count].view(
                    environment_count, candidate_count, self.area_width
                ),
                area_min_y=self.area_min_y[:snapshot_count].view(
                    environment_count, candidate_count, self.area_width
                ),
                area_max_y=self.area_max_y[:snapshot_count].view(
                    environment_count, candidate_count, self.area_width
                ),
                area_connected_components=self.area_connected_components[:snapshot_count].view(
                    environment_count, candidate_count, self.area_width
                ),
                area_crossings=self.area_crossings[:snapshot_count].view(
                    environment_count, candidate_count, self.area_crossings_width
                ),
                area_size=self.area_size[:snapshot_count].view(
                    environment_count, candidate_count, self.area_width
                ),
                area_map_station_count=self.area_map_station_count[:snapshot_count].view(
                    environment_count, candidate_count, self.area_width
                ),
                log_temperature=log_temperature,
                log_recommended_candidates=log_recommended_candidates,
                generation_variable_floats=generation_variable_floats,
                lookahead_door_invalid=lookahead_door_invalid,
                lookahead_door_match=lookahead_door_match,
                lookahead_connection_invalid=lookahead_connection_invalid,
                lookahead_toilet_invalid=lookahead_toilet_invalid,
                lookahead_phantoon_invalid=lookahead_phantoon_invalid,
                connection_reachability=self.connection_reachability[:snapshot_count].view(
                    environment_count, candidate_count, self.connection_reachability_width
                ),
                toilet_crossed_room_idx=self.toilet_crossed_room_idx[:snapshot_count].view(
                    environment_count, candidate_count, self.toilet_crossed_room_width
                ),
            ),
            frontier_features=FrontierFeatures(
                frontier=self.frontier[:frontier_row_count],
                frontier_door_variant=self.frontier_door_variant[:frontier_row_count],
                frontier_area=self.frontier_area[:frontier_row_count],
                frontier_occupancy=self.frontier_occupancy[:frontier_row_count],
                frontier_neighbor=self.frontier_neighbor[:frontier_row_count],
                frontier_neighbor_pair=self.frontier_neighbor_pair[:frontier_row_count],
                frontier_connection_reachability=self.frontier_connection_reachability[
                    :frontier_row_count
                ],
                row_snapshot_idx=self.row_snapshot_idx[:frontier_row_count],
                row_frontier_idx=self.row_frontier_idx[:frontier_row_count],
                row_door_output_idx=self.row_door_output_idx[:frontier_row_count],
            ),
            missing_connect_query_features=MissingConnectQueryFeatures(
                query_snapshot_idx=self.missing_connect_query_snapshot_idx[
                    :missing_connect_query_row_count
                ],
                query_connection_idx=self.missing_connect_query_connection_idx[
                    :missing_connect_query_row_count
                ],
                source_frontier=self.missing_connect_query_source_frontier[
                    :missing_connect_query_row_count
                ],
                target_frontier=self.missing_connect_query_target_frontier[
                    :missing_connect_query_row_count
                ],
                source_distance=self.missing_connect_query_source_distance[
                    :missing_connect_query_row_count
                ],
                target_distance=self.missing_connect_query_target_distance[
                    :missing_connect_query_row_count
                ],
                current_distance=self.missing_connect_query_current_distance[
                    :missing_connect_query_row_count
                ],
            ),
            save_refill_utility_query_features=SaveRefillUtilityQueryFeatures(
                query_snapshot_idx=self.save_refill_utility_query_snapshot_idx[
                    :save_refill_utility_query_row_count
                ],
                query_room_part_idx=self.save_refill_utility_query_room_part_idx[
                    :save_refill_utility_query_row_count
                ],
                target_mask=self.save_refill_utility_query_target_mask[
                    :save_refill_utility_query_row_count
                ],
                frontier=self.save_refill_utility_query_frontier[
                    :save_refill_utility_query_row_count
                ],
                frontier_distance=self.save_refill_utility_query_frontier_distance[
                    :save_refill_utility_query_row_count
                ],
                save_to_current_distance=self.save_refill_utility_query_save_to_current_distance[
                    :save_refill_utility_query_row_count
                ],
                save_from_current_distance=self.save_refill_utility_query_save_from_current_distance[
                    :save_refill_utility_query_row_count
                ],
                refill_to_current_distance=self.save_refill_utility_query_refill_to_current_distance[
                    :save_refill_utility_query_row_count
                ],
                refill_from_current_distance=self.save_refill_utility_query_refill_from_current_distance[
                    :save_refill_utility_query_row_count
                ],
            ),
        )


def extract_candidate_features(
    env: EnvironmentGroup,
    candidates: Actions,
    log_temperature: torch.Tensor,
    include_temperature: bool,
    log_recommended_candidates: torch.Tensor,
    include_recommended_candidates: bool,
    generation_variable_floats: torch.Tensor,
    include_generation_variable_floats: bool,
    lookahead_outcomes: StepOutcomes,
    include_lookahead_outcomes: bool,
    feature_requirements: FeatureRequirements,
    feature_slot: FeatureSlot,
) -> Features:
    frontier_row_count = feature_requirements.frontier_row_count
    worker_frontier_row_counts = feature_requirements.worker_frontier_row_counts
    missing_connect_query_row_count = feature_requirements.missing_connect_query_row_count
    worker_missing_connect_query_row_counts = (
        feature_requirements.worker_missing_connect_query_row_counts
    )
    save_refill_utility_query_row_count = feature_requirements.save_refill_utility_query_row_count
    worker_save_refill_utility_query_row_counts = (
        feature_requirements.worker_save_refill_utility_query_row_counts
    )
    feature_slot.ensure(
        candidates.room_idx.numel(),
        frontier_row_count,
        missing_connect_query_row_count,
        save_refill_utility_query_row_count,
    )
    env.env.pack_features_into(
        map_gen.FeatureBuffers(
            {
                "environment_count": candidates.room_idx.shape[0],
                "candidate_count": candidates.room_idx.shape[1],
                "environment_start": 0,
                "frontier_row_count": frontier_row_count,
                "worker_frontier_row_counts": worker_frontier_row_counts,
                "missing_connect_query_row_count": missing_connect_query_row_count,
                "worker_missing_connect_query_row_counts": (
                    worker_missing_connect_query_row_counts
                ),
                "save_refill_utility_query_row_count": save_refill_utility_query_row_count,
                "worker_save_refill_utility_query_row_counts": (
                    worker_save_refill_utility_query_row_counts
                ),
                "inventory": feature_slot.inventory.numpy(),
                "room_x": feature_slot.room_x.numpy(),
                "room_y": feature_slot.room_y.numpy(),
                "room_placed": feature_slot.room_placed.numpy(),
                "room_part_furthest_destination": feature_slot.room_part_furthest_destination.numpy(),
                "room_part_furthest_source": feature_slot.room_part_furthest_source.numpy(),
                "room_part_save_from_room_distance": (
                    feature_slot.room_part_save_from_room_distance.numpy()
                ),
                "room_part_save_to_room_distance": (
                    feature_slot.room_part_save_to_room_distance.numpy()
                ),
                "room_part_refill_from_room_distance": (
                    feature_slot.room_part_refill_from_room_distance.numpy()
                ),
                "room_part_refill_to_room_distance": (
                    feature_slot.room_part_refill_to_room_distance.numpy()
                ),
                "room_part_frontier_from_room_distance": (
                    feature_slot.room_part_frontier_from_room_distance.numpy()
                ),
                "room_part_frontier_to_room_distance": (
                    feature_slot.room_part_frontier_to_room_distance.numpy()
                ),
                "known_save_from_room_distance": (
                    feature_slot.known_save_from_room_distance.numpy()
                ),
                "known_save_to_room_distance": (feature_slot.known_save_to_room_distance.numpy()),
                "known_refill_from_room_distance": (
                    feature_slot.known_refill_from_room_distance.numpy()
                ),
                "known_refill_to_room_distance": (
                    feature_slot.known_refill_to_room_distance.numpy()
                ),
                "area_used": feature_slot.area_used.numpy(),
                "area_min_x": feature_slot.area_min_x.numpy(),
                "area_max_x": feature_slot.area_max_x.numpy(),
                "area_min_y": feature_slot.area_min_y.numpy(),
                "area_max_y": feature_slot.area_max_y.numpy(),
                "area_connected_components": (
                    feature_slot.area_connected_components.numpy()
                ),
                "area_crossings": feature_slot.area_crossings.numpy(),
                "area_size": feature_slot.area_size.numpy(),
                "area_map_station_count": feature_slot.area_map_station_count.numpy(),
                "frontier": feature_slot.frontier.numpy(),
                "frontier_door_variant": feature_slot.frontier_door_variant.numpy(),
                "frontier_area": feature_slot.frontier_area.numpy(),
                "frontier_occupancy": feature_slot.frontier_occupancy.numpy(),
                "frontier_neighbor": feature_slot.frontier_neighbor.numpy(),
                "frontier_neighbor_pair": feature_slot.frontier_neighbor_pair.numpy(),
                "connection_reachability": feature_slot.connection_reachability.numpy(),
                "frontier_connection_reachability": feature_slot.frontier_connection_reachability.numpy(),
                "missing_connect_query_snapshot_idx": (
                    feature_slot.missing_connect_query_snapshot_idx.numpy()
                ),
                "missing_connect_query_connection_idx": (
                    feature_slot.missing_connect_query_connection_idx.numpy()
                ),
                "missing_connect_query_source_frontier": (
                    feature_slot.missing_connect_query_source_frontier.numpy()
                ),
                "missing_connect_query_target_frontier": (
                    feature_slot.missing_connect_query_target_frontier.numpy()
                ),
                "missing_connect_query_source_distance": (
                    feature_slot.missing_connect_query_source_distance.numpy()
                ),
                "missing_connect_query_target_distance": (
                    feature_slot.missing_connect_query_target_distance.numpy()
                ),
                "missing_connect_query_current_distance": (
                    feature_slot.missing_connect_query_current_distance.numpy()
                ),
                "save_refill_utility_query_snapshot_idx": (
                    feature_slot.save_refill_utility_query_snapshot_idx.numpy()
                ),
                "save_refill_utility_query_room_part_idx": (
                    feature_slot.save_refill_utility_query_room_part_idx.numpy()
                ),
                "save_refill_utility_query_target_mask": (
                    feature_slot.save_refill_utility_query_target_mask.numpy()
                ),
                "save_refill_utility_query_frontier": (
                    feature_slot.save_refill_utility_query_frontier.numpy()
                ),
                "save_refill_utility_query_frontier_distance": (
                    feature_slot.save_refill_utility_query_frontier_distance.numpy()
                ),
                "save_refill_utility_query_save_to_current_distance": (
                    feature_slot.save_refill_utility_query_save_to_current_distance.numpy()
                ),
                "save_refill_utility_query_save_from_current_distance": (
                    feature_slot.save_refill_utility_query_save_from_current_distance.numpy()
                ),
                "save_refill_utility_query_refill_to_current_distance": (
                    feature_slot.save_refill_utility_query_refill_to_current_distance.numpy()
                ),
                "save_refill_utility_query_refill_from_current_distance": (
                    feature_slot.save_refill_utility_query_refill_from_current_distance.numpy()
                ),
                "toilet_crossed_room_idx": feature_slot.toilet_crossed_room_idx.numpy(),
                "row_snapshot_idx": feature_slot.row_snapshot_idx.numpy(),
                "row_frontier_idx": feature_slot.row_frontier_idx.numpy(),
                "row_door_output_idx": feature_slot.row_door_output_idx.numpy(),
            }
        )
    )
    return feature_slot.features(
        candidates.room_idx.shape[0],
        candidates.room_idx.shape[1],
        log_temperature,
        include_temperature,
        log_recommended_candidates,
        include_recommended_candidates,
        generation_variable_floats,
        include_generation_variable_floats,
        lookahead_outcomes,
        include_lookahead_outcomes,
        frontier_row_count,
        missing_connect_query_row_count,
        save_refill_utility_query_row_count,
    ).flatten_candidates()
