from __future__ import annotations

import torch
import math
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

from env import OutputMetadata, Features
from features import (
    COORD_OFFSET,
    FRONTIER_NODE_FEATURES,
    FRONTIER_PAIR_FEATURES,
    GLOBAL_FEATURES,
    NUM_COORD_VALUES,
    FeatureContext,
)

if TYPE_CHECKING:
    from train_config import FeatureConfig

DETERMINISTIC_INVALID_LOGIT = 20.0
ProfileCallback = Callable[[str, Callable[[], object]], object]


def no_profile(_name: str, fn: Callable[[], object]) -> object:
    return fn()


# These tensors are all f32 with shape
#    [batch, time, output]  during training,
#    [batch, candidate, output]  during generation
@dataclass
class Predictions:
    # log-odds of invalid door (unconnected):
    door_invalid: torch.Tensor
    # log-odds of invalid connection (lack of return path):
    connection_invalid: torch.Tensor
    # log-odds of invalid Toilet crossing count:
    toilet_invalid: torch.Tensor
    # Predicted balance-model log-odds for the matched target door:
    balance_score: torch.Tensor
    # Predicted balance-model log-odds for the room crossed by the Toilet:
    toilet_balance_score: torch.Tensor
    # Predicted average live frontier count across the full episode:
    avg_frontiers: torch.Tensor
    # Predicted graph diameter across placed room parts:
    graph_diameter: torch.Tensor
    # Predicted save-to-room proximity utility for each global room part:
    save_to_room_utility: torch.Tensor
    # Predicted room-to-save proximity utility for each global room part:
    save_from_room_utility: torch.Tensor
    # Predicted refill-to-room proximity utility for each global room part:
    refill_to_room_utility: torch.Tensor
    # Predicted room-to-refill proximity utility for each global room part:
    refill_from_room_utility: torch.Tensor
    # Predicted utility for each required missing connection:
    missing_connect_utility: torch.Tensor
    # Frontier-local proposal logits for door variants:
    proposal_score: torch.Tensor
    # Optional frontier-local state before global pooling:
    proposal_state: torch.Tensor
    proposal_row_snapshot_idx: torch.Tensor
    proposal_row_frontier_idx: torch.Tensor


@dataclass
class BalancePredictions:
    left: torch.Tensor
    right: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor
    toilet_crossed_room: torch.Tensor


def get_predictions(raw_preds, output_sizes):
    preds = []
    col = 0
    for size in output_sizes:
        preds.append(raw_preds[:, :, col : (col + size)])
        col += size

    return Predictions(
        door_invalid=preds[0],
        connection_invalid=preds[1],
        toilet_invalid=preds[2].squeeze(-1),
        balance_score=preds[3],
        toilet_balance_score=preds[4].squeeze(-1),
        avg_frontiers=raw_preds.new_empty([raw_preds.shape[0], raw_preds.shape[1]]),
        graph_diameter=raw_preds.new_empty([raw_preds.shape[0], raw_preds.shape[1]]),
        save_to_room_utility=raw_preds.new_empty([raw_preds.shape[0], raw_preds.shape[1], 0]),
        save_from_room_utility=raw_preds.new_empty([raw_preds.shape[0], raw_preds.shape[1], 0]),
        refill_to_room_utility=raw_preds.new_empty([raw_preds.shape[0], raw_preds.shape[1], 0]),
        refill_from_room_utility=raw_preds.new_empty([raw_preds.shape[0], raw_preds.shape[1], 0]),
        missing_connect_utility=raw_preds.new_empty([raw_preds.shape[0], raw_preds.shape[1], 0]),
        proposal_score=raw_preds.new_empty([raw_preds.shape[0], raw_preds.shape[1], 0]),
        proposal_state=raw_preds.new_empty([raw_preds.shape[0], raw_preds.shape[1], 0]),
        proposal_row_snapshot_idx=raw_preds.new_empty([0], dtype=torch.int64),
        proposal_row_frontier_idx=raw_preds.new_empty([0], dtype=torch.int16),
    )


def apply_known_invalid_logits(
    invalid_logits: torch.Tensor,
    known_invalid: torch.Tensor,
    outcome_name: str,
) -> torch.Tensor:
    if known_invalid.shape[-1] == 0:
        return invalid_logits
    torch._assert(
        known_invalid.shape[-1] == invalid_logits.shape[-1],
        f"known {outcome_name} outcomes must match {outcome_name} prediction width",
    )
    while known_invalid.ndim < invalid_logits.ndim:
        known_invalid = known_invalid.unsqueeze(1)
    deterministic_logits = torch.where(
        known_invalid == 0,
        -DETERMINISTIC_INVALID_LOGIT,
        DETERMINISTIC_INVALID_LOGIT,
    ).to(invalid_logits.dtype)
    return torch.where(known_invalid >= 0, deterministic_logits, invalid_logits)


def apply_known_distance_utility(
    utility: torch.Tensor,
    known_distance: torch.Tensor,
    distance_proximity_scale: float,
    outcome_name: str,
) -> torch.Tensor:
    if known_distance.shape[-1] == 0:
        return utility
    torch._assert(
        known_distance.shape[-1] == utility.shape[-1],
        f"known {outcome_name} distances must match {outcome_name} prediction width",
    )
    while known_distance.ndim < utility.ndim:
        known_distance = known_distance.unsqueeze(1)
    scale = utility.new_tensor(distance_proximity_scale)
    finite_distance = (known_distance.to(utility.dtype) - 2).clamp_min(0)
    known_utility = torch.where(
        known_distance == 1,
        torch.zeros_like(utility),
        scale / (finite_distance + scale),
    )
    return torch.where(known_distance > 0, known_utility, utility)


def apply_frontier_door_invalid_logits(
    door_invalid: torch.Tensor,
    frontier_door_invalid: torch.Tensor,
    row_snapshot_idx: torch.Tensor,
    row_door_output_idx: torch.Tensor,
) -> torch.Tensor:
    if door_invalid.shape[-1] == 0 or frontier_door_invalid.shape[0] == 0:
        return door_invalid
    snapshot_count = door_invalid.shape[0]
    door_output_count = door_invalid.shape[-1]
    frontier_door_invalid = frontier_door_invalid.squeeze(-1).to(door_invalid.dtype)
    row_snapshot_idx = row_snapshot_idx.to(device=door_invalid.device, dtype=torch.int64)
    row_door_output_idx = row_door_output_idx.to(device=door_invalid.device, dtype=torch.int64)
    valid_rows = (
        (row_snapshot_idx >= 0)
        & (row_snapshot_idx < snapshot_count)
        & (row_door_output_idx >= 0)
        & (row_door_output_idx < door_output_count)
    )
    safe_row_snapshot_idx = row_snapshot_idx.clamp(0, snapshot_count - 1)
    safe_row_door_output_idx = row_door_output_idx.clamp(0, door_output_count - 1)
    row_lookup_idx = safe_row_snapshot_idx * door_output_count + safe_row_door_output_idx
    door_invalid_flat = door_invalid.flatten().clone()
    scatter_values = torch.where(
        valid_rows,
        frontier_door_invalid,
        door_invalid_flat.detach().gather(0, row_lookup_idx),
    )
    door_invalid_flat.scatter_(0, row_lookup_idx, scatter_values)
    return door_invalid_flat.view_as(door_invalid)


def normalize(x: torch.Tensor):
    return torch.nn.functional.rms_norm(x, (x.size(-1),))


def activation_dtype(device: torch.device, parameter_dtype: torch.dtype) -> torch.dtype:
    if device.type == "cuda" and torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    return parameter_dtype


class FactorizedOutcomeHead(torch.nn.Module):
    def __init__(self, output_metadata, num_geometry_outcomes, embedding_width):
        super().__init__()
        self.embedding_width = embedding_width
        self.num_outputs = len(output_metadata)
        metadata = torch.tensor(output_metadata, dtype=torch.int64).reshape(self.num_outputs, 2)
        self.register_buffer("room_idx", metadata[:, 0])
        self.register_buffer("geometry_outcome_idx", metadata[:, 1])
        self.geometry_outcome_embedding = torch.nn.Parameter(
            torch.randn([num_geometry_outcomes, embedding_width]) / math.sqrt(embedding_width)
        )
        self.state = torch.nn.Linear(embedding_width, embedding_width, bias=False)
        self.logit_scale = torch.nn.Parameter(
            torch.tensor(math.log(math.sqrt(embedding_width) / 2))
        )

    def forward(self, X, room_x, room_y, room_placed, pos_embedding_x, pos_embedding_y):
        if self.num_outputs == 0:
            return X.new_empty([X.shape[0], X.shape[1], 0], dtype=torch.float32)
        state = self.state(X)
        # Keep normalization, base logits, and final logits out of reduced
        # precision. These scores directly drive both the loss and candidate
        # selection.
        with torch.amp.autocast(X.device.type, enabled=False):
            state = torch.nn.functional.normalize(state.to(torch.float32), dim=-1)
            geometry_outcome_embedding = torch.nn.functional.normalize(
                self.geometry_outcome_embedding.to(torch.float32), dim=-1
            )
            pos_embedding_x = torch.nn.functional.normalize(
                pos_embedding_x.to(torch.float32), dim=-1
            )
            pos_embedding_y = torch.nn.functional.normalize(
                pos_embedding_y.to(torch.float32), dim=-1
            )
            base_query = geometry_outcome_embedding[self.geometry_outcome_idx]
            base_logits = torch.matmul(state, base_query.transpose(0, 1))
            x_logits = torch.matmul(state, pos_embedding_x.transpose(0, 1))
            y_logits = torch.matmul(state, pos_embedding_y.transpose(0, 1))
            room_logits = torch.gather(x_logits, -1, room_x) + torch.gather(y_logits, -1, room_y)
            room_logits = torch.where(room_placed, room_logits, 0.0)
            position_logits = room_logits[..., self.room_idx]
            return (base_logits + position_logits) * torch.exp(
                torch.clamp(self.logit_scale.to(torch.float32), max=math.log(100.0))
            )


class ProposalOutput(torch.nn.Module):
    def __init__(
        self,
        input_width: int,
        hidden_width: int,
        output_width: int,
    ):
        super().__init__()
        if hidden_width <= 0:
            raise ValueError("proposal_hidden_width must be greater than zero")
        self.out_features = output_width
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_width, hidden_width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_width, output_width, bias=False),
        )
        self.layers[-1].weight.data.zero_()

    @property
    def output_dtype(self) -> torch.dtype:
        return self.layers[-1].weight.dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MissingConnectQueryHead(torch.nn.Module):
    scalar_width = 4
    max_distance_index = 510
    unreachable_distance = 255

    def __init__(
        self,
        embedding_width: int,
        global_embedding_width: int,
        frontier_width: int,
        distance_width: int,
        hidden_width: int,
    ):
        super().__init__()
        if frontier_width <= 0:
            raise ValueError("missing_connect_query_frontier_width must be greater than zero")
        if distance_width <= 0:
            raise ValueError("missing_connect_query_distance_width must be greater than zero")
        if hidden_width <= 0:
            raise ValueError("missing_connect_query_hidden_width must be greater than zero")
        self.frontier_projection = torch.nn.Linear(embedding_width, frontier_width, bias=False)
        self.total_distance_embedding = torch.nn.Embedding(
            self.max_distance_index + 1,
            distance_width,
        )
        self.margin_embedding = torch.nn.Embedding(self.max_distance_index + 1, distance_width)
        self.pair_layers = torch.nn.Sequential(
            torch.nn.Linear(
                frontier_width * 2 + distance_width * 2,
                hidden_width,
                bias=False,
            ),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_width, hidden_width, bias=False),
        )
        self.output_layers = torch.nn.Sequential(
            torch.nn.Linear(
                hidden_width * 2 + global_embedding_width + self.scalar_width,
                hidden_width,
                bias=False,
            ),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_width, 2, bias=False),
        )
        self.output_layers[-1].weight.data.zero_()

    def _gather(
        self,
        frontier_state: torch.Tensor,
        frontier: torch.Tensor,
        query_snapshot_idx: torch.Tensor,
        row_count_by_snapshot: torch.Tensor,
        row_start_by_snapshot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_frontier = frontier.to(torch.int64)
        safe_local_frontier = local_frontier.clamp_min(0)
        row_count = row_count_by_snapshot[query_snapshot_idx].unsqueeze(1)
        valid = (local_frontier >= 0) & (safe_local_frontier < row_count)
        packed_frontier = (
            row_start_by_snapshot[query_snapshot_idx].unsqueeze(1) + safe_local_frontier
        )
        packed_frontier = packed_frontier.clamp_max(max(frontier_state.shape[0] - 1, 0))
        return self.frontier_projection(frontier_state[packed_frontier]), valid

    def forward(
        self,
        frontier_state: torch.Tensor,
        global_state: torch.Tensor,
        row_count_by_snapshot: torch.Tensor,
        row_start_by_snapshot: torch.Tensor,
        query,
        connection_output_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_count = query.query_connection_idx.shape[0]
        snapshot_count = global_state.shape[0]
        if query_count == 0 or frontier_state.shape[0] == 0:
            return (
                frontier_state.new_zeros([snapshot_count, 1, connection_output_count]),
                torch.zeros(
                    [snapshot_count, 1, connection_output_count],
                    dtype=torch.bool,
                    device=frontier_state.device,
                ),
            )
        query_snapshot_idx = query.query_snapshot_idx.to(torch.int64)
        source_state, source_valid = self._gather(
            frontier_state,
            query.source_frontier,
            query_snapshot_idx,
            row_count_by_snapshot,
            row_start_by_snapshot,
        )
        target_state, target_valid = self._gather(
            frontier_state,
            query.target_frontier,
            query_snapshot_idx,
            row_count_by_snapshot,
            row_start_by_snapshot,
        )
        pair_valid = source_valid & target_valid
        total_distance = query.source_distance.to(torch.int16) + query.target_distance.to(
            torch.int16
        )
        total_distance_idx = total_distance.clamp(0, self.max_distance_index).to(torch.int64)
        current_distance = query.current_distance.to(torch.int16).view(query_count, 1)
        finite_current = current_distance != self.unreachable_distance
        finite_margin_idx = (current_distance - total_distance + 255).clamp(
            0,
            self.max_distance_index - 1,
        )
        margin_idx = torch.where(
            finite_current,
            finite_margin_idx,
            torch.full_like(finite_margin_idx, self.max_distance_index),
        ).to(torch.int64)
        total_distance_embedding = self.total_distance_embedding(total_distance_idx).to(
            frontier_state.dtype
        )
        margin_embedding = self.margin_embedding(margin_idx).to(frontier_state.dtype)
        pair_embedding = self.pair_layers(
            torch.cat(
                [
                    source_state,
                    target_state,
                    total_distance_embedding,
                    margin_embedding,
                ],
                dim=-1,
            )
        )
        pair_mask = pair_valid.unsqueeze(-1)
        valid_pair_count = pair_mask.sum(dim=1).clamp_min(1)
        pair_mean = (pair_embedding * pair_mask).sum(dim=1) / valid_pair_count
        pair_max = pair_embedding.masked_fill(~pair_mask, -torch.inf).amax(dim=1)
        pair_max = torch.where(torch.isfinite(pair_max), pair_max, 0)
        source_count = query.source_count.to(frontier_state.dtype)
        target_count = query.target_count.to(frontier_state.dtype)
        scalar = torch.cat(
            [
                torch.log1p(source_count).unsqueeze(1) / math.log(65536.0),
                torch.log1p(target_count).unsqueeze(1) / math.log(65536.0),
                source_valid.any(dim=1).to(frontier_state.dtype).unsqueeze(1),
                target_valid.any(dim=1).to(frontier_state.dtype).unsqueeze(1),
            ],
            dim=1,
        )
        query_output = self.output_layers(
            torch.cat(
                [
                    pair_mean,
                    pair_max,
                    global_state[query_snapshot_idx],
                    scalar,
                ],
                dim=1,
            )
        )
        flat_idx = query_snapshot_idx * connection_output_count + query.query_connection_idx.to(
            torch.int64
        )
        output = frontier_state.new_zeros([snapshot_count * connection_output_count])
        utility_output = output.clone()
        output.index_copy_(0, flat_idx, query_output[:, 0])
        utility_output.index_copy_(0, flat_idx, query_output[:, 1])
        mask = torch.zeros_like(output, dtype=torch.bool)
        mask[flat_idx] = True
        classification_mask = torch.zeros_like(output, dtype=torch.bool)
        classification_mask[flat_idx] = query.current_distance == self.unreachable_distance
        output_shape = (snapshot_count, 1, connection_output_count)
        return (
            output.view(output_shape),
            utility_output.view(output_shape),
            classification_mask.view(output_shape),
            mask.view(output_shape),
        )


class SaveRefillUtilityQueryHead(torch.nn.Module):
    scalar_width = 2
    kind_count = 4
    max_distance_index = 510
    unreachable_distance = 255

    def __init__(
        self,
        embedding_width: int,
        hidden_width: int,
        frontier_width: int,
        kind_width: int,
    ):
        super().__init__()
        if hidden_width <= 0:
            raise ValueError("utility_query_hidden_width must be greater than zero")
        if frontier_width <= 0:
            raise ValueError("utility_query_frontier_width must be greater than zero")
        if kind_width <= 0:
            raise ValueError("utility_query_kind_width must be greater than zero")
        self.frontier_projection = torch.nn.Linear(embedding_width, frontier_width, bias=False)
        self.distance_embedding = torch.nn.Embedding(self.max_distance_index + 1, 1)
        self.margin_embedding = torch.nn.Embedding(self.max_distance_index + 1, 1)
        self.kind_embedding = torch.nn.Embedding(self.kind_count, kind_width)
        self.output_layers = torch.nn.Sequential(
            torch.nn.Linear(
                frontier_width + 2 + kind_width + self.scalar_width,
                hidden_width,
                bias=False,
            ),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_width, 1, bias=False),
        )
        self.output_layers[-1].weight.data.zero_()

    def _gather(
        self,
        frontier_state: torch.Tensor,
        frontier: torch.Tensor,
        query_snapshot_idx: torch.Tensor,
        row_count_by_snapshot: torch.Tensor,
        row_start_by_snapshot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_frontier = frontier.to(torch.int64)
        safe_local_frontier = local_frontier.clamp_min(0)
        row_count = row_count_by_snapshot[query_snapshot_idx]
        valid = (local_frontier >= 0) & (safe_local_frontier < row_count)
        packed_frontier = row_start_by_snapshot[query_snapshot_idx] + safe_local_frontier
        packed_frontier = packed_frontier.clamp_max(max(frontier_state.shape[0] - 1, 0))
        return frontier_state[packed_frontier], valid

    def forward(
        self,
        frontier_state: torch.Tensor,
        row_count_by_snapshot: torch.Tensor,
        row_start_by_snapshot: torch.Tensor,
        query,
        room_part_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_count = query.query_room_part_idx.shape[0]
        snapshot_count = row_count_by_snapshot.shape[0]
        if query_count == 0 or frontier_state.shape[0] == 0:
            return (
                frontier_state.new_zeros([self.kind_count, snapshot_count, 1, room_part_count]),
                torch.zeros(
                    [self.kind_count, snapshot_count, 1, room_part_count],
                    dtype=torch.bool,
                    device=frontier_state.device,
                ),
            )
        query_snapshot_idx = query.query_snapshot_idx.to(torch.int64)
        projected_frontier_state = self.frontier_projection(frontier_state)
        frontier, valid = self._gather(
            projected_frontier_state,
            query.frontier,
            query_snapshot_idx,
            row_count_by_snapshot,
            row_start_by_snapshot,
        )
        distance = query.frontier_distance.to(torch.int16)
        distance_idx = distance.clamp(0, self.max_distance_index).to(torch.int64)
        current_distance = query.current_distance.to(torch.int16)
        finite_current = current_distance != self.unreachable_distance
        finite_margin_idx = (current_distance - distance + 255).clamp(
            0,
            self.max_distance_index - 1,
        )
        margin_idx = torch.where(
            finite_current,
            finite_margin_idx,
            torch.full_like(finite_margin_idx, self.max_distance_index),
        ).to(torch.int64)
        safe_frontier = torch.where(valid.unsqueeze(1), frontier, torch.zeros_like(frontier))
        current_distance_value = torch.where(
            finite_current,
            torch.log1p(query.current_distance.to(frontier_state.dtype)) / math.log(256.0),
            query.current_distance.new_zeros([query_count], dtype=frontier_state.dtype),
        )
        scalar = torch.cat(
            [
                finite_current.to(frontier_state.dtype).unsqueeze(1),
                current_distance_value.unsqueeze(1),
            ],
            dim=1,
        )
        query_kind = query.query_kind.to(torch.int64).clamp(0, self.kind_count - 1)
        query_utility = self.output_layers(
            torch.cat(
                [
                    safe_frontier,
                    self.distance_embedding(distance_idx).to(frontier_state.dtype),
                    self.margin_embedding(margin_idx).to(frontier_state.dtype),
                    self.kind_embedding(query_kind).to(frontier_state.dtype),
                    scalar,
                ],
                dim=1,
            )
        ).squeeze(1)
        safe_room_part_idx = query.query_room_part_idx.to(torch.int64).clamp(
            0,
            max(room_part_count - 1, 0),
        )
        flat_idx = (
            query_kind * (snapshot_count * room_part_count)
            + query_snapshot_idx * room_part_count
            + safe_room_part_idx
        )
        output = frontier_state.new_zeros([self.kind_count * snapshot_count * room_part_count])
        output.index_copy_(0, flat_idx, query_utility)
        output_mask = torch.zeros_like(output, dtype=torch.bool)
        output_mask[flat_idx] = True
        return output.view(self.kind_count, snapshot_count, 1, room_part_count), output_mask.view(
            self.kind_count,
            snapshot_count,
            1,
            room_part_count,
        )


class FrontierModel(torch.nn.Module):
    def __init__(
        self,
        num_rooms,
        output_metadata: OutputMetadata,
        map_x,
        map_y,
        embedding_width,
        global_embedding_width,
        hidden_width,
        proposal_hidden_width,
        missing_connect_query_hidden_width,
        missing_connect_query_frontier_width,
        missing_connect_query_distance_width,
        utility_query_hidden_width,
        utility_query_frontier_width,
        utility_query_kind_width,
        known_save_refill_utility_override,
        distance_proximity_scale,
        num_layers,
        door_counts,
        frontier_window_size,
        features: FeatureConfig,
    ):
        super().__init__()
        self.features = features
        self.num_rooms = num_rooms
        self.map_x = map_x
        self.map_y = map_y
        self.embedding_width = embedding_width
        self.global_embedding_width = global_embedding_width
        if distance_proximity_scale <= 0:
            raise ValueError("distance_proximity_scale must be greater than zero")
        self.distance_proximity_scale = distance_proximity_scale
        self.known_save_refill_utility_override = known_save_refill_utility_override
        self.num_room_parts = output_metadata.num_room_parts
        if global_embedding_width <= 0:
            raise ValueError("global_embedding_width must be greater than zero")
        self.left_count, self.right_count, self.up_count, self.down_count = door_counts
        if self.features.missing_connect_query and missing_connect_query_hidden_width <= 0:
            raise ValueError("missing_connect_query_hidden_width must be greater than zero")
        if self.features.missing_connect_query and missing_connect_query_frontier_width <= 0:
            raise ValueError("missing_connect_query_frontier_width must be greater than zero")
        if self.features.missing_connect_query and missing_connect_query_distance_width <= 0:
            raise ValueError("missing_connect_query_distance_width must be greater than zero")
        if (
            self.features.save_utility_query or self.features.refill_utility_query
        ) and utility_query_hidden_width <= 0:
            raise ValueError("utility_query_hidden_width must be greater than zero")
        door_output_size, connection_output_size = output_metadata.get_output_sizes()
        self.output_sizes = (
            door_output_size,
            connection_output_size,
            1,
            door_output_size,
            1,
        )
        if sum(door_counts) != door_output_size:
            raise ValueError("door_counts must sum to the door output size")
        self.num_connection_outputs = len(output_metadata.connection)
        if self.features.global_room_position and not self.features.room_position:
            raise ValueError("features.global_room_position requires features.room_position")
        feature_context = FeatureContext(
            features=features,
            output_metadata=output_metadata,
            num_rooms=num_rooms,
            num_room_parts=self.num_room_parts,
            num_connection_outputs=self.num_connection_outputs,
            door_counts=(self.left_count, self.right_count, self.up_count, self.down_count),
            frontier_window_area=frontier_window_size**2,
        )
        frontier_node_feature_classes = [
            feature_class
            for feature_class in FRONTIER_NODE_FEATURES
            if feature_class.is_enabled(features)
        ]
        frontier_node_width = sum(
            feature_class.tensor_width(feature_context)
            for feature_class in frontier_node_feature_classes
        )
        self.frontier_node_features = torch.nn.ModuleList(
            [
                feature_class.build(feature_context)
                for feature_class in frontier_node_feature_classes
            ]
        )
        self.frontier_node_mlp = (
            torch.nn.Linear(frontier_node_width, embedding_width, bias=False)
            if frontier_node_width > 0
            else None
        )
        frontier_pair_feature_classes = [
            feature_class
            for feature_class in FRONTIER_PAIR_FEATURES
            if feature_class.is_enabled(features)
        ]
        pair_width = sum(
            feature_class.tensor_width(feature_context)
            for feature_class in frontier_pair_feature_classes
        )
        self.frontier_pair_features = torch.nn.ModuleList(
            [
                feature_class.build(feature_context)
                for feature_class in frontier_pair_feature_classes
            ]
        )
        use_neighbors = self.features.frontier_neighbor
        self.source_message_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(embedding_width, hidden_width, bias=False)
                for _ in range(num_layers if use_neighbors else 0)
            ]
        )
        self.pair_message_layers = (
            torch.nn.ModuleList(
                [
                    torch.nn.Linear(pair_width, hidden_width, bias=False)
                    for _ in range(num_layers if use_neighbors else 0)
                ]
            )
            if pair_width > 0
            else [None] * (num_layers if use_neighbors else 0)
        )
        self.message_output_layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_width, embedding_width, bias=False),
                )
                for _ in range(num_layers if use_neighbors else 0)
            ]
        )
        self.update_layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(
                        embedding_width * 2 + global_embedding_width,
                        hidden_width,
                        bias=False,
                    ),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_width, embedding_width, bias=False),
                )
                for _ in range(num_layers if use_neighbors else 0)
            ]
        )
        global_feature_classes = [
            feature_class
            for feature_class in GLOBAL_FEATURES
            if feature_class.is_enabled(features)
        ]
        self.global_features = torch.nn.ModuleList(
            [feature_class.build(feature_context) for feature_class in global_feature_classes]
        )
        global_width = sum(
            feature_class.tensor_width(feature_context) for feature_class in global_feature_classes
        )
        self.global_mlp = (
            torch.nn.Linear(global_width, global_embedding_width, bias=False)
            if global_width > 0
            else None
        )
        pooled_width = global_embedding_width + 2 * embedding_width * int(
            self.features.frontier_mask
        )
        self.pooled_mlp = torch.nn.Sequential(
            torch.nn.Linear(pooled_width, hidden_width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_width, embedding_width, bias=False),
        )
        self.pos_embedding_x = torch.nn.Parameter(
            torch.randn([NUM_COORD_VALUES, embedding_width]) / math.sqrt(embedding_width)
        )
        self.pos_embedding_y = torch.nn.Parameter(
            torch.randn([NUM_COORD_VALUES, embedding_width]) / math.sqrt(embedding_width)
        )
        door_output_metadata = torch.tensor(output_metadata.door, dtype=torch.int64).reshape(
            door_output_size,
            2,
        )
        self.register_buffer("door_variant_outcome_idx", door_output_metadata[:, 1])
        self.door_output = torch.nn.Linear(embedding_width, output_metadata.num_door_variants)
        self.frontier_door_invalid_output = torch.nn.Linear(embedding_width, 1)
        self.connection_output = FactorizedOutcomeHead(
            output_metadata.connection, output_metadata.num_connection_variants, embedding_width
        )
        self.missing_connect_query_output = (
            MissingConnectQueryHead(
                embedding_width,
                global_embedding_width,
                missing_connect_query_frontier_width,
                missing_connect_query_distance_width,
                missing_connect_query_hidden_width,
            )
            if self.features.missing_connect_query
            else None
        )
        self.toilet_output = torch.nn.Linear(embedding_width, 1)
        self.balance_score_output = FactorizedOutcomeHead(
            output_metadata.door, output_metadata.num_door_variants, embedding_width
        )
        self.toilet_balance_score_output = torch.nn.Linear(embedding_width, 1)
        self.avg_frontiers_output = torch.nn.Linear(embedding_width, 1)
        self.graph_diameter_output = torch.nn.Linear(embedding_width, 1)
        self.save_to_room_utility_output = torch.nn.Linear(embedding_width, self.num_room_parts)
        self.save_from_room_utility_output = torch.nn.Linear(embedding_width, self.num_room_parts)
        self.refill_to_room_utility_output = torch.nn.Linear(embedding_width, self.num_room_parts)
        self.refill_from_room_utility_output = torch.nn.Linear(
            embedding_width,
            self.num_room_parts,
        )
        self.save_refill_utility_query_output = (
            SaveRefillUtilityQueryHead(
                embedding_width,
                utility_query_hidden_width,
                utility_query_frontier_width,
                utility_query_kind_width,
            )
            if self.features.save_utility_query or self.features.refill_utility_query
            else None
        )
        self.missing_connect_utility_output = torch.nn.Linear(
            embedding_width,
            self.num_connection_outputs,
        )
        self.proposal_output = ProposalOutput(
            embedding_width,
            proposal_hidden_width,
            output_metadata.num_door_variants,
        )

    def _position_embedding(self, x, y, embedding_x, embedding_y, dtype, offset=0):
        x = x.to(torch.int64) + offset
        y = y.to(torch.int64) + offset
        return embedding_x[x].to(dtype) + embedding_y[y].to(dtype)

    def _pair_features(self, features, neighbor, dtype):
        values = []
        for pair_feature in self.frontier_pair_features:
            values.append(pair_feature(features, neighbor, dtype))
        return torch.cat(values, dim=-1) if values else None

    def _activation_dtype(self, device: torch.device) -> torch.dtype:
        return activation_dtype(device, next(self.parameters()).dtype)

    def forward(
        self,
        features: Features,
        return_proposal_state: bool,
        profile: ProfileCallback,
    ):
        # Shapes below use: s=snapshot, r=frontier row, k=neighbors, e=embedding width,
        # h=message hidden width.
        # node: [r, 5]
        node = features.frontier_features.frontier
        row_snapshot_idx = features.frontier_features.row_snapshot_idx.to(torch.int64)
        snapshot_count = features.global_features.inventory.shape[0]
        row_count = node.shape[0]
        dtype = self._activation_dtype(node.device)
        # X: [r, e]
        X = node.new_zeros([row_count, self.embedding_width], dtype=dtype)
        if self.frontier_node_mlp is not None:
            node_inputs = []
            for frontier_node_feature in self.frontier_node_features:
                node_inputs.append(frontier_node_feature(features, dtype))
            X = X + self.frontier_node_mlp(torch.cat(node_inputs, dim=-1))
        if row_count == 0:
            row_count_by_snapshot = torch.zeros(
                [snapshot_count],
                dtype=torch.int64,
                device=row_snapshot_idx.device,
            )
        else:
            row_count_by_snapshot = torch.bincount(
                row_snapshot_idx,
                minlength=snapshot_count,
            )
        row_start_by_snapshot = row_count_by_snapshot.cumsum(0) - row_count_by_snapshot
        if self.global_mlp is not None:
            global_inputs = []
            for global_feature in self.global_features:
                global_inputs.append(global_feature(features, X.dtype))
            global_state = self.global_mlp(torch.cat(global_inputs, dim=-1))
        else:
            global_state = X.new_zeros([snapshot_count, self.global_embedding_width])
        if row_count == 0:
            mean_pool = max_pool = X.new_zeros([snapshot_count, self.embedding_width])
        else:
            # pair: [r, k, pair_width], neighbor: [r, k], pair_mask: [r, k, 1]
            frontier_neighbor = features.frontier_features.frontier_neighbor
            local_neighbor = frontier_neighbor.clamp_min(0).to(torch.int64)
            row_neighbor_count = row_count_by_snapshot[row_snapshot_idx].unsqueeze(1)
            neighbor_valid = (frontier_neighbor >= 0) & (local_neighbor < row_neighbor_count)
            neighbor = row_start_by_snapshot[row_snapshot_idx].unsqueeze(1) + local_neighbor
            pair_mask = neighbor_valid.unsqueeze(-1)
            pair = self._pair_features(features, neighbor, dtype)
            single_neighbor = neighbor.shape[1] == 1
            if single_neighbor:
                neighbor = neighbor[:, 0]
                pair_mask = pair_mask[:, 0]
                if pair is not None:
                    pair = pair[:, 0]
            else:
                pair_count = pair_mask.sum(1).clamp_min(1)
            global_rows = global_state[row_snapshot_idx]
            for source_layer, pair_layer, output_layer, update_layer in zip(
                self.source_message_layers,
                self.pair_message_layers,
                self.message_output_layers,
                self.update_layers,
            ):
                # source: [r, h]
                source = source_layer(X)
                # Gather each frontier's neighbors: source: [r, k, h]
                source = source[neighbor]
                messages = source if pair_layer is None else source + pair_layer(pair)
                messages = output_layer(messages) * pair_mask
                if not single_neighbor:
                    # messages: [r, k, e]
                    messages = messages.sum(1) / pair_count
                # messages [r, e]
                X = X + update_layer(torch.cat([X, messages, global_rows], dim=-1))
            if X.device.type == "cuda":
                mean_pool = X.new_zeros([snapshot_count, self.embedding_width])
                mean_pool.index_add_(0, row_snapshot_idx, X)
                count = row_count_by_snapshot.to(X.dtype).unsqueeze(1).clamp_min(1)
                mean_pool = mean_pool / count
                max_pool = X.new_full([snapshot_count, self.embedding_width], -torch.inf)
                max_pool.scatter_reduce_(
                    0,
                    row_snapshot_idx.unsqueeze(1).expand(-1, self.embedding_width),
                    X,
                    reduce="amax",
                    include_self=True,
                )
            else:
                count = row_count_by_snapshot.to(X.dtype).unsqueeze(1).clamp_min(1)
                mean_pool = (
                    torch.segment_reduce(
                        X,
                        "sum",
                        lengths=row_count_by_snapshot,
                        axis=0,
                    )
                    / count
                )
                max_pool = torch.segment_reduce(
                    X,
                    "max",
                    lengths=row_count_by_snapshot,
                    axis=0,
                )
            max_pool = torch.where(torch.isfinite(max_pool), max_pool, 0)
        frontier_door_invalid = self.frontier_door_invalid_output(X)
        proposal_state = X if return_proposal_state else X.new_empty([row_count, 0])
        frontier_state = X
        # mean_pool, max_pool, pooled_state: [s, e]
        pooled_inputs = [global_state]
        if self.features.frontier_mask:
            pooled_inputs.extend([mean_pool, max_pool])
        pooled_state = self.pooled_mlp(torch.cat(pooled_inputs, dim=-1))
        if self.features.room_position:
            room_x = (features.global_features.room_x.to(torch.int64) + COORD_OFFSET).unsqueeze(1)
            room_y = (features.global_features.room_y.to(torch.int64) + COORD_OFFSET).unsqueeze(1)
            room_placed = features.global_features.room_placed.to(torch.bool).unsqueeze(1)
        else:
            room_x = torch.full(
                [snapshot_count, 1, self.num_rooms],
                COORD_OFFSET,
                dtype=torch.int64,
                device=X.device,
            )
            room_y = room_x
            room_placed = torch.zeros(
                [snapshot_count, 1, self.num_rooms], dtype=torch.bool, device=X.device
            )
        # X: [s, 1, e]
        X = pooled_state.unsqueeze(1)
        door_variant = self.door_output(X)
        door = door_variant[..., self.door_variant_outcome_idx]
        connection = self.connection_output(
            X, room_x, room_y, room_placed, self.pos_embedding_x, self.pos_embedding_y
        )
        toilet = self.toilet_output(X)
        balance_score = self.balance_score_output(
            X,
            room_x,
            room_y,
            room_placed,
            self.pos_embedding_x,
            self.pos_embedding_y,
        )
        toilet_balance_score = self.toilet_balance_score_output(X)
        avg_frontiers = self.avg_frontiers_output(X).squeeze(-1).to(torch.float32)
        graph_diameter = self.graph_diameter_output(X).squeeze(-1).to(torch.float32)
        (
            save_to_room_utility,
            save_from_room_utility,
            refill_to_room_utility,
            refill_from_room_utility,
        ) = profile(
            "python.model.save_refill_linear_heads",
            lambda: (
                self.save_to_room_utility_output(X).to(torch.float32),
                self.save_from_room_utility_output(X).to(torch.float32),
                self.refill_to_room_utility_output(X).to(torch.float32),
                self.refill_from_room_utility_output(X).to(torch.float32),
            ),
        )
        missing_connect_utility = self.missing_connect_utility_output(X).to(torch.float32)
        preds = get_predictions(
            torch.cat([door, connection, toilet, balance_score, toilet_balance_score], dim=-1),
            self.output_sizes,
        )
        door_invalid = apply_frontier_door_invalid_logits(
            preds.door_invalid,
            frontier_door_invalid,
            row_snapshot_idx,
            features.frontier_features.row_door_output_idx,
        )
        door_invalid = apply_known_invalid_logits(
            door_invalid,
            features.global_features.lookahead_door_invalid,
            "door",
        )
        connection_invalid = preds.connection_invalid
        if self.missing_connect_query_output is not None:
            (
                query_connection_invalid,
                query_missing_connect_utility,
                query_connection_mask,
                query_utility_mask,
            ) = self.missing_connect_query_output(
                frontier_state,
                global_state,
                row_count_by_snapshot,
                row_start_by_snapshot,
                features.missing_connect_query_features,
                self.num_connection_outputs,
            )
            connection_invalid = torch.where(
                query_connection_mask & self.features.missing_connect_query,
                query_connection_invalid,
                connection_invalid,
            )
            query_missing_connect_utility = query_missing_connect_utility.to(torch.float32)
            missing_connect_utility = torch.where(
                query_utility_mask,
                query_missing_connect_utility,
                missing_connect_utility,
            )
        connection_invalid = apply_known_invalid_logits(
            connection_invalid,
            features.global_features.lookahead_connection_invalid,
            "connection",
        )
        if self.save_refill_utility_query_output is not None:
            query_save_refill_utility, query_save_refill_mask = profile(
                "python.model.save_refill_query_head",
                lambda: self.save_refill_utility_query_output(
                    frontier_state,
                    row_count_by_snapshot,
                    row_start_by_snapshot,
                    features.save_refill_utility_query_features,
                    self.num_room_parts,
                ),
            )
            query_save_refill_utility = query_save_refill_utility.to(torch.float32)
            (
                save_to_room_utility,
                save_from_room_utility,
                refill_to_room_utility,
                refill_from_room_utility,
            ) = profile(
                "python.model.save_refill_query_replace",
                lambda: (
                    torch.where(
                        query_save_refill_mask[0],
                        query_save_refill_utility[0],
                        save_to_room_utility,
                    ),
                    torch.where(
                        query_save_refill_mask[1],
                        query_save_refill_utility[1],
                        save_from_room_utility,
                    ),
                    torch.where(
                        query_save_refill_mask[2],
                        query_save_refill_utility[2],
                        refill_to_room_utility,
                    ),
                    torch.where(
                        query_save_refill_mask[3],
                        query_save_refill_utility[3],
                        refill_from_room_utility,
                    ),
                ),
            )
        if self.known_save_refill_utility_override:
            save_to_room_utility = apply_known_distance_utility(
                save_to_room_utility,
                features.global_features.known_save_to_room_distance,
                self.distance_proximity_scale,
                "save-to-room utility",
            )
            save_from_room_utility = apply_known_distance_utility(
                save_from_room_utility,
                features.global_features.known_save_from_room_distance,
                self.distance_proximity_scale,
                "save-from-room utility",
            )
            refill_to_room_utility = apply_known_distance_utility(
                refill_to_room_utility,
                features.global_features.known_refill_to_room_distance,
                self.distance_proximity_scale,
                "refill-to-room utility",
            )
            refill_from_room_utility = apply_known_distance_utility(
                refill_from_room_utility,
                features.global_features.known_refill_from_room_distance,
                self.distance_proximity_scale,
                "refill-from-room utility",
            )
        return Predictions(
            door_invalid=door_invalid,
            connection_invalid=connection_invalid,
            toilet_invalid=preds.toilet_invalid,
            balance_score=preds.balance_score,
            toilet_balance_score=preds.toilet_balance_score,
            avg_frontiers=avg_frontiers,
            graph_diameter=graph_diameter,
            save_to_room_utility=save_to_room_utility,
            save_from_room_utility=save_from_room_utility,
            refill_to_room_utility=refill_to_room_utility,
            refill_from_room_utility=refill_from_room_utility,
            missing_connect_utility=missing_connect_utility,
            proposal_score=X.new_empty([row_count, 0]),
            proposal_state=proposal_state,
            proposal_row_snapshot_idx=(
                row_snapshot_idx if return_proposal_state else row_snapshot_idx.new_empty([0])
            ),
            proposal_row_frontier_idx=(
                features.frontier_features.row_frontier_idx
                if return_proposal_state
                else features.frontier_features.row_frontier_idx.new_empty([0])
            ),
        )


class BalanceModel(torch.nn.Module):
    def __init__(
        self,
        left_count: int,
        right_count: int,
        up_count: int,
        down_count: int,
        num_rooms: int,
        hidden_width: int,
        num_layers: int,
    ):
        super().__init__()
        if hidden_width <= 0:
            raise ValueError("balance model hidden_width must be greater than zero")
        if num_layers <= 0:
            raise ValueError("balance model num_layers must be greater than zero")
        self.left_count = left_count
        self.right_count = right_count
        self.up_count = up_count
        self.down_count = down_count
        self.num_rooms = num_rooms
        self.output_width = (
            left_count * right_count
            + right_count * left_count
            + up_count * down_count
            + down_count * up_count
            + num_rooms
        )

        layers: list[torch.nn.Module] = []
        input_width = 1
        for _ in range(num_layers):
            layers.extend(
                [
                    torch.nn.Linear(input_width, hidden_width),
                    torch.nn.GELU(),
                ]
            )
            input_width = hidden_width
        output_layer = torch.nn.Linear(input_width, self.output_width)
        output_layer.weight.data.zero_()
        layers.append(output_layer)
        self.net = torch.nn.Sequential(*layers)

    def forward(self, log_temperature: torch.Tensor) -> BalancePredictions:
        parameter_dtype = next(self.parameters()).dtype
        raw = self.net(
            log_temperature.to(
                activation_dtype(log_temperature.device, parameter_dtype)
            ).unsqueeze(-1)
        ).to(torch.float32)
        offset = 0
        left_size = self.left_count * self.right_count
        right_size = self.right_count * self.left_count
        up_size = self.up_count * self.down_count
        down_size = self.down_count * self.up_count
        left = raw[:, offset : offset + left_size].reshape(
            log_temperature.shape[0], self.left_count, self.right_count
        )
        offset += left_size
        right = raw[:, offset : offset + right_size].reshape(
            log_temperature.shape[0], self.right_count, self.left_count
        )
        offset += right_size
        up = raw[:, offset : offset + up_size].reshape(
            log_temperature.shape[0], self.up_count, self.down_count
        )
        offset += up_size
        down = raw[:, offset : offset + down_size].reshape(
            log_temperature.shape[0], self.down_count, self.up_count
        )
        offset += down_size
        toilet_crossed_room = raw[:, offset : offset + self.num_rooms]
        return BalancePredictions(
            left=left,
            right=right,
            up=up,
            down=down,
            toilet_crossed_room=toilet_crossed_room,
        )
