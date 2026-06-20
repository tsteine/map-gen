from __future__ import annotations

import torch
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from env import OutputMetadata, Features

if TYPE_CHECKING:
    from train_config import FeatureConfig

NUM_COORD_VALUES = 256
COORD_OFFSET = 128
DETERMINISTIC_INVALID_LOGIT = 20.0


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
    # Predicted distance for each required missing connection:
    missing_connect_distance: torch.Tensor
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
        missing_connect_distance=raw_preds.new_empty([raw_preds.shape[0], raw_preds.shape[1], 0]),
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


class MissingConnectFrontierQueryHead(torch.nn.Module):
    scalar_width = 10

    def __init__(
        self,
        embedding_width: int,
        global_embedding_width: int,
        hidden_width: int,
    ):
        super().__init__()
        if hidden_width <= 0:
            raise ValueError("missing_connect_hidden_width must be greater than zero")
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(embedding_width * 4 + global_embedding_width + self.scalar_width, hidden_width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_width, 1, bias=False),
        )
        self.layers[-1].weight.data.zero_()

    def _pool(
        self,
        frontier_state: torch.Tensor,
        frontier: torch.Tensor,
        distance: torch.Tensor,
        query_snapshot_idx: torch.Tensor,
        row_count_by_snapshot: torch.Tensor,
        row_start_by_snapshot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        local_frontier = frontier.to(torch.int64)
        safe_local_frontier = local_frontier.clamp_min(0)
        row_count = row_count_by_snapshot[query_snapshot_idx].unsqueeze(1)
        valid = (local_frontier >= 0) & (safe_local_frontier < row_count)
        packed_frontier = row_start_by_snapshot[query_snapshot_idx].unsqueeze(1) + safe_local_frontier
        packed_frontier = packed_frontier.clamp_max(max(frontier_state.shape[0] - 1, 0))
        gathered = frontier_state[packed_frontier]
        mask = valid.unsqueeze(-1)
        masked = gathered * mask
        count = mask.sum(dim=1).clamp_min(1)
        mean = masked.sum(dim=1) / count
        max_values = gathered.masked_fill(~mask, -torch.inf).amax(dim=1)
        max_values = torch.where(torch.isfinite(max_values), max_values, 0)
        distance_value = torch.log1p(distance.to(frontier_state.dtype)) / math.log(256.0)
        distance_value = distance_value * valid.to(frontier_state.dtype)
        scalar_count = valid.sum(dim=1).clamp_min(1).to(frontier_state.dtype)
        mean_distance = distance_value.sum(dim=1, keepdim=True) / scalar_count.unsqueeze(1)
        min_distance = distance_value.masked_fill(~valid, float("inf")).amin(dim=1, keepdim=True)
        min_distance = torch.where(torch.isfinite(min_distance), min_distance, 0)
        return mean, max_values, mean_distance, min_distance

    def forward(
        self,
        frontier_state: torch.Tensor,
        global_state: torch.Tensor,
        row_count_by_snapshot: torch.Tensor,
        row_start_by_snapshot: torch.Tensor,
        features: Features,
        connection_output_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = features.missing_connect_query_features
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
        source_mean, source_max, source_mean_distance, source_min_distance = self._pool(
            frontier_state,
            query.source_frontier,
            query.source_distance,
            query_snapshot_idx,
            row_count_by_snapshot,
            row_start_by_snapshot,
        )
        target_mean, target_max, target_mean_distance, target_min_distance = self._pool(
            frontier_state,
            query.target_frontier,
            query.target_distance,
            query_snapshot_idx,
            row_count_by_snapshot,
            row_start_by_snapshot,
        )
        source_count = query.source_count.to(frontier_state.dtype)
        target_count = query.target_count.to(frontier_state.dtype)
        scalar = torch.cat(
            [
                torch.log1p(source_count).unsqueeze(1) / math.log(65536.0),
                torch.log1p(target_count).unsqueeze(1) / math.log(65536.0),
                query.source_cap_hit.to(frontier_state.dtype).unsqueeze(1),
                query.target_cap_hit.to(frontier_state.dtype).unsqueeze(1),
                source_mean_distance,
                target_mean_distance,
                source_min_distance,
                target_min_distance,
                (source_count > 0).to(frontier_state.dtype).unsqueeze(1),
                (target_count > 0).to(frontier_state.dtype).unsqueeze(1),
            ],
            dim=1,
        )
        query_logits = self.layers(
            torch.cat(
                [
                    source_mean,
                    source_max,
                    target_mean,
                    target_max,
                    global_state[query_snapshot_idx],
                    scalar,
                ],
                dim=1,
            )
        ).squeeze(1)
        flat_idx = query_snapshot_idx * connection_output_count + query.query_connection_idx.to(
            torch.int64
        )
        output = frontier_state.new_zeros([snapshot_count * connection_output_count])
        output.index_copy_(0, flat_idx, query_logits)
        mask = torch.zeros_like(output, dtype=torch.bool)
        mask[flat_idx] = True
        return output.view(snapshot_count, 1, connection_output_count), mask.view(
            snapshot_count,
            1,
            connection_output_count,
        )


class MissingConnectFrontierQuerySummary(torch.nn.Module):
    scalar_width = 10
    participation_width = 4

    def __init__(
        self,
        embedding_width: int,
        hidden_width: int,
    ):
        super().__init__()
        if hidden_width <= 0:
            raise ValueError(
                "missing_connect_query_summary_hidden_width must be greater than zero"
            )
        self.hidden_width = hidden_width
        self.query_layers = torch.nn.Sequential(
            torch.nn.Linear(embedding_width * 4 + self.scalar_width, hidden_width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_width, hidden_width, bias=False),
        )
        self.source_message_layers = torch.nn.Sequential(
            torch.nn.Linear(hidden_width + 1, hidden_width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_width, hidden_width, bias=False),
        )
        self.target_message_layers = torch.nn.Sequential(
            torch.nn.Linear(hidden_width + 1, hidden_width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_width, hidden_width, bias=False),
        )
        self.output_layers = torch.nn.Sequential(
            torch.nn.Linear(hidden_width * 2 + self.participation_width, hidden_width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_width, embedding_width, bias=False),
        )
        self.output_layers[-1].weight.data.zero_()

    def _pool(
        self,
        frontier_state: torch.Tensor,
        frontier: torch.Tensor,
        distance: torch.Tensor,
        query_snapshot_idx: torch.Tensor,
        row_count_by_snapshot: torch.Tensor,
        row_start_by_snapshot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        local_frontier = frontier.to(torch.int64)
        safe_local_frontier = local_frontier.clamp_min(0)
        row_count = row_count_by_snapshot[query_snapshot_idx].unsqueeze(1)
        valid = (local_frontier >= 0) & (safe_local_frontier < row_count)
        packed_frontier = row_start_by_snapshot[query_snapshot_idx].unsqueeze(1) + safe_local_frontier
        packed_frontier = packed_frontier.clamp_max(max(frontier_state.shape[0] - 1, 0))
        gathered = frontier_state[packed_frontier]
        mask = valid.unsqueeze(-1)
        masked = gathered * mask
        count = mask.sum(dim=1).clamp_min(1)
        mean = masked.sum(dim=1) / count
        max_values = gathered.masked_fill(~mask, -torch.inf).amax(dim=1)
        max_values = torch.where(torch.isfinite(max_values), max_values, 0)
        distance_value = torch.log1p(distance.to(frontier_state.dtype)) / math.log(256.0)
        distance_value = distance_value * valid.to(frontier_state.dtype)
        scalar_count = valid.sum(dim=1).clamp_min(1).to(frontier_state.dtype)
        mean_distance = distance_value.sum(dim=1, keepdim=True) / scalar_count.unsqueeze(1)
        min_distance = distance_value.masked_fill(~valid, float("inf")).amin(dim=1, keepdim=True)
        min_distance = torch.where(torch.isfinite(min_distance), min_distance, 0)
        return mean, max_values, mean_distance, min_distance

    def _scatter_side(
        self,
        query_embedding: torch.Tensor,
        frontier: torch.Tensor,
        distance: torch.Tensor,
        query_snapshot_idx: torch.Tensor,
        row_count_by_snapshot: torch.Tensor,
        row_start_by_snapshot: torch.Tensor,
        frontier_count: int,
        message_layers: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_frontier = frontier.to(torch.int64)
        safe_local_frontier = local_frontier.clamp_min(0)
        row_count = row_count_by_snapshot[query_snapshot_idx].unsqueeze(1)
        valid = (local_frontier >= 0) & (safe_local_frontier < row_count)
        packed_frontier = row_start_by_snapshot[query_snapshot_idx].unsqueeze(1) + safe_local_frontier
        packed_frontier = packed_frontier.clamp_max(max(frontier_count - 1, 0))
        query_idx = torch.arange(
            query_embedding.shape[0],
            dtype=torch.int64,
            device=query_embedding.device,
        ).unsqueeze(1)
        query_idx = query_idx.expand_as(local_frontier).flatten()
        distance_value = torch.log1p(distance.to(query_embedding.dtype)) / math.log(256.0)
        messages = message_layers(
            torch.cat(
                [
                    query_embedding[query_idx],
                    distance_value.flatten().unsqueeze(1),
                ],
                dim=1,
            )
        )
        valid_value = valid.flatten().to(query_embedding.dtype).unsqueeze(1)
        messages = messages * valid_value
        packed_frontier = packed_frontier.flatten()
        summary = query_embedding.new_zeros([frontier_count, self.hidden_width])
        summary.index_add_(0, packed_frontier, messages)
        count = query_embedding.new_zeros([summary.shape[0], 1])
        count.index_add_(0, packed_frontier, valid_value)
        summary = summary / count.clamp_min(1)
        return summary, count

    def forward(
        self,
        frontier_state: torch.Tensor,
        row_count_by_snapshot: torch.Tensor,
        row_start_by_snapshot: torch.Tensor,
        features: Features,
    ) -> torch.Tensor:
        query = features.missing_connect_query_features
        query_count = query.query_connection_idx.shape[0]
        frontier_count = frontier_state.shape[0]
        if query_count == 0 or frontier_count == 0:
            return torch.zeros_like(frontier_state)
        query_snapshot_idx = query.query_snapshot_idx.to(torch.int64)
        source_mean, source_max, source_mean_distance, source_min_distance = self._pool(
            frontier_state,
            query.source_frontier,
            query.source_distance,
            query_snapshot_idx,
            row_count_by_snapshot,
            row_start_by_snapshot,
        )
        target_mean, target_max, target_mean_distance, target_min_distance = self._pool(
            frontier_state,
            query.target_frontier,
            query.target_distance,
            query_snapshot_idx,
            row_count_by_snapshot,
            row_start_by_snapshot,
        )
        source_count = query.source_count.to(frontier_state.dtype)
        target_count = query.target_count.to(frontier_state.dtype)
        scalar = torch.cat(
            [
                torch.log1p(source_count).unsqueeze(1) / math.log(65536.0),
                torch.log1p(target_count).unsqueeze(1) / math.log(65536.0),
                query.source_cap_hit.to(frontier_state.dtype).unsqueeze(1),
                query.target_cap_hit.to(frontier_state.dtype).unsqueeze(1),
                source_mean_distance,
                target_mean_distance,
                source_min_distance,
                target_min_distance,
                (source_count > 0).to(frontier_state.dtype).unsqueeze(1),
                (target_count > 0).to(frontier_state.dtype).unsqueeze(1),
            ],
            dim=1,
        )
        query_embedding = self.query_layers(
            torch.cat(
                [
                    source_mean,
                    source_max,
                    target_mean,
                    target_max,
                    scalar,
                ],
                dim=1,
            )
        )
        source_summary, source_participation_count = self._scatter_side(
            query_embedding,
            query.source_frontier,
            query.source_distance,
            query_snapshot_idx,
            row_count_by_snapshot,
            row_start_by_snapshot,
            frontier_count,
            self.source_message_layers,
        )
        target_summary, target_participation_count = self._scatter_side(
            query_embedding,
            query.target_frontier,
            query.target_distance,
            query_snapshot_idx,
            row_count_by_snapshot,
            row_start_by_snapshot,
            frontier_count,
            self.target_message_layers,
        )
        participation = torch.cat(
            [
                torch.log1p(source_participation_count) / math.log(65536.0),
                torch.log1p(target_participation_count) / math.log(65536.0),
                (source_participation_count > 0).to(frontier_state.dtype),
                (target_participation_count > 0).to(frontier_state.dtype),
            ],
            dim=1,
        )
        return self.output_layers(torch.cat([source_summary, target_summary, participation], dim=1))


class FrontierModel(torch.nn.Module):
    def __init__(
        self,
        num_rooms,
        output_metadata: OutputMetadata,
        map_x,
        map_y,
        embedding_width,
        global_embedding_width,
        global_room_position_embedding_width,
        hidden_width,
        proposal_hidden_width,
        missing_connect_hidden_width,
        missing_connect_query_summary_hidden_width,
        door_match_embedding_width,
        toilet_crossed_room_embedding_width,
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
        self.global_room_position_embedding_width = global_room_position_embedding_width
        self.num_room_parts = output_metadata.num_room_parts
        if global_embedding_width <= 0:
            raise ValueError("global_embedding_width must be greater than zero")
        if global_room_position_embedding_width <= 0:
            raise ValueError("global_room_position_embedding_width must be greater than zero")
        self.left_count, self.right_count, self.up_count, self.down_count = door_counts
        if self.features.lookahead_outcomes and door_match_embedding_width <= 0:
            raise ValueError("door_match_embedding_width must be greater than zero")
        if self.features.toilet_crossed_room and toilet_crossed_room_embedding_width <= 0:
            raise ValueError("toilet_crossed_room_embedding_width must be greater than zero")
        if self.features.missing_connect_query and missing_connect_hidden_width <= 0:
            raise ValueError("missing_connect_hidden_width must be greater than zero")
        if (
            self.features.missing_connect_query_summary
            and missing_connect_query_summary_hidden_width <= 0
        ):
            raise ValueError(
                "missing_connect_query_summary_hidden_width must be greater than zero"
            )
        if self.features.missing_connect_query_summary and not self.features.missing_connect_query:
            raise ValueError("missing_connect_query_summary requires missing_connect_query")
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
        self.include_inventory = self.features.inventory
        if self.features.global_room_position and not self.features.room_position:
            raise ValueError("features.global_room_position requires features.room_position")
        self.register_buffer(
            "room_connection_variant_idx",
            torch.tensor(output_metadata.room_connection_variant_idx, dtype=torch.int64),
        )
        # self.inventory_embedding = torch.nn.Parameter(
        #     torch.randn([output_metadata.num_room_connection_variants, embedding_width]) / math.sqrt(embedding_width)
        # ) if self.features.inventory else None
        self.orientation_embedding = (
            torch.nn.Embedding(2, embedding_width) if self.features.frontier_orientation else None
        )
        self.kind_embedding = (
            torch.nn.Embedding(256, embedding_width) if self.features.frontier_kind else None
        )
        self.frontier_door_variant_embedding = (
            torch.nn.Embedding(output_metadata.num_door_variants, embedding_width)
            if self.features.frontier_door_variant
            else None
        )
        node_numeric_width = (
            frontier_window_size**2 * self.features.frontier_occupancy
            + 2 * self.num_connection_outputs * self.features.frontier_connection_reachability
        )
        self.node_numeric = (
            torch.nn.Linear(node_numeric_width, embedding_width, bias=False)
            if node_numeric_width > 0
            else None
        )
        self.frontier_window_area = frontier_window_size**2
        self.register_buffer(
            "frontier_occupancy_bits",
            1 << torch.arange(8, dtype=torch.uint8),
            persistent=False,
        )
        pair_width = 3 * self.features.frontier_neighbor_flags
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
        global_width = (
            output_metadata.num_room_connection_variants * self.features.inventory
            + embedding_width
            * (self.features.connection_reachability and self.num_connection_outputs > 0)
            + int(self.features.temperature)
            + int(self.features.recommended_candidates)
            + global_room_position_embedding_width * int(self.features.global_room_position)
            + 2 * self.num_room_parts * int(self.features.room_part_furthest_distance)
            + 2 * self.num_room_parts * int(self.features.room_part_save_distance)
            + 2 * self.num_room_parts * int(self.features.room_part_refill_distance)
            + 2 * self.num_room_parts * int(self.features.room_part_frontier_distance)
            + 4 * self.num_room_parts
            + (door_match_embedding_width + 2 * connection_output_size + 2)
            * int(self.features.lookahead_outcomes)
            + (toilet_crossed_room_embedding_width * int(self.features.toilet_crossed_room))
        )
        self.global_mlp = (
            torch.nn.Linear(global_width, global_embedding_width, bias=False)
            if global_width > 0
            else None
        )
        pooled_width = (
            output_metadata.num_room_connection_variants * self.features.inventory
            + embedding_width
            * (
                2 * self.features.frontier_mask
                + (self.features.connection_reachability and self.num_connection_outputs > 0)
            )
            + int(self.features.temperature)
            + int(self.features.recommended_candidates)
            + global_room_position_embedding_width * int(self.features.global_room_position)
            + (door_match_embedding_width + 2 * connection_output_size + 2)
            * int(self.features.lookahead_outcomes)
            + (toilet_crossed_room_embedding_width * int(self.features.toilet_crossed_room))
            + 4 * self.num_room_parts
        )
        self.pooled_mlp = (
            torch.nn.Sequential(
                torch.nn.Linear(pooled_width, hidden_width, bias=False),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_width, embedding_width, bias=False),
            )
            if pooled_width > 0
            else None
        )
        self.door_match_embedding_width = door_match_embedding_width
        self.left_door_match_embedding = self._door_match_embedding(
            self.left_count,
            self.right_count,
            door_match_embedding_width,
        )
        self.right_door_match_embedding = self._door_match_embedding(
            self.right_count,
            self.left_count,
            door_match_embedding_width,
        )
        self.up_door_match_embedding = self._door_match_embedding(
            self.up_count,
            self.down_count,
            door_match_embedding_width,
        )
        self.down_door_match_embedding = self._door_match_embedding(
            self.down_count,
            self.up_count,
            door_match_embedding_width,
        )
        self.connection_reachability_embedding = (
            torch.nn.Linear(self.num_connection_outputs, embedding_width, bias=False)
            if self.features.connection_reachability and self.num_connection_outputs > 0
            else None
        )
        self.toilet_crossed_room_embedding = (
            torch.nn.Embedding(num_rooms + 1, toilet_crossed_room_embedding_width)
            if self.features.toilet_crossed_room
            else None
        )
        self.room_part_furthest_distance_embedding = (
            torch.nn.Embedding(256, 1) if self.features.room_part_furthest_distance else None
        )
        self.room_part_save_distance_embedding = (
            torch.nn.Embedding(256, 1) if self.features.room_part_save_distance else None
        )
        self.room_part_refill_distance_embedding = (
            torch.nn.Embedding(256, 1) if self.features.room_part_refill_distance else None
        )
        self.room_part_frontier_distance_embedding = (
            torch.nn.Embedding(256, 1) if self.features.room_part_frontier_distance else None
        )
        self.known_distance_embedding = torch.nn.Embedding(256, 1)
        self.frontier_pos_embedding_x = (
            torch.nn.Parameter(
                torch.randn([NUM_COORD_VALUES, embedding_width]) / math.sqrt(embedding_width)
            )
            if self.features.frontier_position
            else None
        )
        self.frontier_pos_embedding_y = (
            torch.nn.Parameter(
                torch.randn([NUM_COORD_VALUES, embedding_width]) / math.sqrt(embedding_width)
            )
            if self.features.frontier_position
            else None
        )
        self.frontier_relative_pos_embedding_x = (
            torch.nn.Parameter(
                torch.randn([NUM_COORD_VALUES, hidden_width]) / math.sqrt(hidden_width)
            )
            if self.features.frontier_neighbor_position_embedding
            else None
        )
        self.frontier_relative_pos_embedding_y = (
            torch.nn.Parameter(
                torch.randn([NUM_COORD_VALUES, hidden_width]) / math.sqrt(hidden_width)
            )
            if self.features.frontier_neighbor_position_embedding
            else None
        )
        self.global_room_pos_embedding_x = (
            torch.nn.Parameter(
                torch.randn(
                    [
                        output_metadata.num_room_connection_variants,
                        NUM_COORD_VALUES,
                        global_room_position_embedding_width,
                    ]
                )
                / math.sqrt(global_room_position_embedding_width)
            )
            if self.features.global_room_position
            else None
        )
        self.global_room_pos_embedding_y = (
            torch.nn.Parameter(
                torch.randn(
                    [
                        output_metadata.num_room_connection_variants,
                        NUM_COORD_VALUES,
                        global_room_position_embedding_width,
                    ]
                )
                / math.sqrt(global_room_position_embedding_width)
            )
            if self.features.global_room_position
            else None
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
            MissingConnectFrontierQueryHead(
                embedding_width,
                global_embedding_width,
                missing_connect_hidden_width,
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
        self.missing_connect_distance_output = torch.nn.Linear(
            embedding_width,
            self.num_connection_outputs,
        )
        self.proposal_output = ProposalOutput(
            embedding_width,
            proposal_hidden_width,
            output_metadata.num_door_variants,
        )
        self.missing_connect_query_summary = (
            MissingConnectFrontierQuerySummary(
                embedding_width,
                missing_connect_query_summary_hidden_width,
            )
            if self.features.missing_connect_query_summary
            else None
        )

    def _door_match_embedding(
        self,
        source_count: int,
        partner_count: int,
        width: int,
    ) -> torch.nn.Parameter | None:
        if not self.features.lookahead_outcomes:
            return None
        return torch.nn.Parameter(
            torch.randn([source_count, partner_count + 1, width]) / math.sqrt(width)
        )

    def _position_embedding(self, x, y, embedding_x, embedding_y, dtype, offset=0):
        x = x.to(torch.int64) + offset
        y = y.to(torch.int64) + offset
        return embedding_x[x].to(dtype) + embedding_y[y].to(dtype)

    def _global_room_position_features(self, features: Features, dtype):
        if not self.features.global_room_position:
            return None
        room_x = features.global_features.room_x.to(torch.int64) + COORD_OFFSET
        room_y = features.global_features.room_y.to(torch.int64) + COORD_OFFSET
        room_connection_variant_idx = (
            self.room_connection_variant_idx.to(room_x.device).unsqueeze(0).expand_as(room_x)
        )
        room_position = (
            self.global_room_pos_embedding_x[room_connection_variant_idx, room_x]
            + self.global_room_pos_embedding_y[room_connection_variant_idx, room_y]
        ).to(dtype)
        placed = features.global_features.room_placed.to(dtype).unsqueeze(-1)
        placed_count = placed.sum(dim=1).clamp_min(1)
        return (room_position * placed).sum(dim=1) / torch.sqrt(placed_count)

    def _pair_features(self, features, dtype):
        values = []
        if self.features.frontier_neighbor_flags:
            flags = features.frontier_features.frontier_neighbor_pair
            values.append(
                torch.stack(
                    [
                        (flags & 1 != 0).to(dtype),
                        (flags & 2 != 0).to(dtype),
                        (flags & 4 != 0).to(dtype),
                    ],
                    dim=-1,
                )
            )
        return torch.cat(values, dim=-1) if values else None

    def _activation_dtype(self, device: torch.device) -> torch.dtype:
        return activation_dtype(device, next(self.parameters()).dtype)

    def _direction_door_match_features(
        self,
        matches: torch.Tensor,
        embedding: torch.nn.Parameter | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if embedding is None or matches.shape[-1] == 0:
            return matches.new_zeros(
                [matches.shape[0], self.door_match_embedding_width],
                dtype=dtype,
            )
        known = matches >= 0
        safe_matches = matches.clamp(min=0).to(torch.int64)
        source_idx = torch.arange(
            embedding.shape[0],
            dtype=torch.int64,
            device=matches.device,
        ).unsqueeze(0)
        values = embedding.to(dtype)[source_idx, safe_matches]
        return torch.sum(values * known.unsqueeze(-1), dim=1)

    def _lookahead_outcome_features(
        self,
        features: Features,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        left, right, up, down = torch.split(
            features.global_features.lookahead_door_match,
            [self.left_count, self.right_count, self.up_count, self.down_count],
            dim=-1,
        )
        door_match_features = (
            self._direction_door_match_features(left, self.left_door_match_embedding, dtype)
            + self._direction_door_match_features(right, self.right_door_match_embedding, dtype)
            + self._direction_door_match_features(up, self.up_door_match_embedding, dtype)
            + self._direction_door_match_features(down, self.down_door_match_embedding, dtype)
        )
        connection_features = torch.stack(
            [
                (features.global_features.lookahead_connection_invalid == 0).to(dtype),
                (features.global_features.lookahead_connection_invalid == 1).to(dtype),
            ],
            dim=-1,
        ).flatten(1)
        toilet_features = torch.stack(
            [
                (features.global_features.lookahead_toilet_invalid == 0).to(dtype),
                (features.global_features.lookahead_toilet_invalid == 1).to(dtype),
            ],
            dim=-1,
        ).flatten(1)
        return torch.cat([door_match_features, connection_features, toilet_features], dim=-1)

    def _toilet_crossed_room_features(
        self,
        features: Features,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.toilet_crossed_room_embedding is None:
            return None
        crossed_room = features.global_features.toilet_crossed_room_idx.to(torch.int64)
        torch._assert(
            torch.all((crossed_room >= -1) & (crossed_room < self.num_rooms)),
            "toilet_crossed_room_idx must be -1 or a valid room index",
        )
        return self.toilet_crossed_room_embedding(crossed_room + 1).squeeze(-2).to(dtype)

    def _room_part_furthest_distance_features(
        self,
        features: Features,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.room_part_furthest_distance_embedding is None:
            return None
        distances = torch.cat(
            [
                features.global_features.room_part_furthest_destination,
                features.global_features.room_part_furthest_source,
            ],
            dim=-1,
        ).to(torch.int64)
        return self.room_part_furthest_distance_embedding(distances).flatten(1).to(dtype)

    def _room_part_save_distance_features(
        self,
        features: Features,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.room_part_save_distance_embedding is None:
            return None
        distances = torch.cat(
            [
                features.global_features.room_part_save_from_room_distance,
                features.global_features.room_part_save_to_room_distance,
            ],
            dim=-1,
        ).to(torch.int64)
        return self.room_part_save_distance_embedding(distances).flatten(1).to(dtype)

    def _room_part_refill_distance_features(
        self,
        features: Features,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.room_part_refill_distance_embedding is None:
            return None
        distances = torch.cat(
            [
                features.global_features.room_part_refill_from_room_distance,
                features.global_features.room_part_refill_to_room_distance,
            ],
            dim=-1,
        ).to(torch.int64)
        return self.room_part_refill_distance_embedding(distances).flatten(1).to(dtype)

    def _room_part_frontier_distance_features(
        self,
        features: Features,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.room_part_frontier_distance_embedding is None:
            return None
        distances = torch.cat(
            [
                features.global_features.room_part_frontier_from_room_distance,
                features.global_features.room_part_frontier_to_room_distance,
            ],
            dim=-1,
        ).to(torch.int64)
        return self.room_part_frontier_distance_embedding(distances).flatten(1).to(dtype)

    def _known_distance_features(
        self,
        features: Features,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        distances = torch.cat(
            [
                features.global_features.known_save_from_room_distance,
                features.global_features.known_save_to_room_distance,
                features.global_features.known_refill_from_room_distance,
                features.global_features.known_refill_to_room_distance,
            ],
            dim=-1,
        ).to(torch.int64)
        return self.known_distance_embedding(distances).flatten(1).to(dtype)

    def _relative_position_features(self, features, neighbor):
        if self.frontier_relative_pos_embedding_x is None:
            return None
        node = features.frontier_features.frontier
        raw_x = node[:, 1].to(torch.int64)
        raw_y = node[:, 2].to(torch.int64)
        raw_x0, raw_x1 = raw_x.unsqueeze(1), raw_x[neighbor]
        raw_y0, raw_y1 = raw_y.unsqueeze(1), raw_y[neighbor]
        return self._position_embedding(
            raw_x1 - raw_x0,
            raw_y1 - raw_y0,
            self.frontier_relative_pos_embedding_x,
            self.frontier_relative_pos_embedding_y,
            self._activation_dtype(features.frontier_features.frontier.device),
            COORD_OFFSET,
        )

    def forward(
        self,
        features: Features,
        return_proposal_state: bool,
    ):
        # Shapes below use: s=snapshot, r=frontier row, k=neighbors, e=embedding width,
        # h=message hidden width.
        # node: [r, 5]
        node = features.frontier_features.frontier
        row_snapshot_idx = features.frontier_features.row_snapshot_idx.to(torch.int64)
        snapshot_count = features.global_features.inventory.shape[0]
        row_count = node.shape[0]
        # numeric: [r, numeric_width]
        numeric = []
        dtype = self._activation_dtype(node.device)
        if self.features.frontier_occupancy:
            numeric.append(
                features.frontier_features.frontier_occupancy.unsqueeze(-1)
                .bitwise_and(self.frontier_occupancy_bits)
                .ne(0)
                .flatten(-2)[..., : self.frontier_window_area]
                .to(dtype)
            )
        if self.features.frontier_connection_reachability:
            flags = features.frontier_features.frontier_connection_reachability
            numeric.append(
                torch.stack(
                    [
                        (flags & 1 != 0).to(dtype),
                        (flags & 2 != 0).to(dtype),
                    ],
                    dim=-1,
                ).flatten(-2)
            )
        # X: [r, e]
        X = node.new_zeros([row_count, self.embedding_width], dtype=dtype)
        if self.node_numeric is not None:
            X = X + self.node_numeric(torch.cat(numeric, dim=-1))
        if self.frontier_pos_embedding_x is not None:
            X = X + self._position_embedding(
                node[:, 1],
                node[:, 2],
                self.frontier_pos_embedding_x,
                self.frontier_pos_embedding_y,
                dtype,
            )
        if self.orientation_embedding is not None:
            X = X + self.orientation_embedding(node[:, 3].to(torch.int64)).to(dtype)
        if self.kind_embedding is not None:
            X = X + self.kind_embedding(node[:, 4].to(torch.int64)).to(dtype)
        if self.frontier_door_variant_embedding is not None:
            X = X + self.frontier_door_variant_embedding(
                features.frontier_features.frontier_door_variant.to(torch.int64)
            ).to(dtype)
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
        if self.missing_connect_query_summary is not None:
            X = X + self.missing_connect_query_summary(
                X,
                row_count_by_snapshot,
                row_start_by_snapshot,
                features,
            )
        # if self.inventory_embedding is not None:
        #     X = X + torch.matmul(
        #         features.global_features.inventory.to(torch.float32), self.inventory_embedding
        #     ).unsqueeze(1)
        # if self.connection_reachability_embedding is not None:
        #     X = X + self.connection_reachability_embedding(
        #         features.global_features.connection_reachability.to(torch.float32)
        #     ).unsqueeze(1)
        inventory_features = features.global_features.inventory.to(X.dtype) if self.include_inventory else None
        connection_features = (
            self.connection_reachability_embedding(features.global_features.connection_reachability.to(X.dtype))
            if self.connection_reachability_embedding is not None
            else None
        )
        temperature_features = (
            features.global_features.log_temperature.to(X.dtype).unsqueeze(-1)
            if self.features.temperature
            else None
        )
        recommended_candidate_features = (
            features.global_features.log_recommended_candidates.to(X.dtype).unsqueeze(-1)
            if self.features.recommended_candidates
            else None
        )
        lookahead_features = (
            self._lookahead_outcome_features(features, X.dtype)
            if self.features.lookahead_outcomes
            else None
        )
        toilet_crossed_room_features = self._toilet_crossed_room_features(features, X.dtype)
        global_room_position_features = self._global_room_position_features(features, X.dtype)
        room_part_furthest_distance_features = self._room_part_furthest_distance_features(
            features,
            X.dtype,
        )
        room_part_save_distance_features = self._room_part_save_distance_features(
            features,
            X.dtype,
        )
        room_part_refill_distance_features = self._room_part_refill_distance_features(
            features,
            X.dtype,
        )
        room_part_frontier_distance_features = self._room_part_frontier_distance_features(
            features,
            X.dtype,
        )
        known_distance_features = self._known_distance_features(features, X.dtype)
        global_inputs = []
        if inventory_features is not None:
            global_inputs.append(inventory_features)
        if connection_features is not None:
            global_inputs.append(connection_features)
        if temperature_features is not None:
            global_inputs.append(temperature_features)
        if recommended_candidate_features is not None:
            global_inputs.append(recommended_candidate_features)
        if lookahead_features is not None:
            global_inputs.append(lookahead_features)
        if toilet_crossed_room_features is not None:
            global_inputs.append(toilet_crossed_room_features)
        if global_room_position_features is not None:
            global_inputs.append(global_room_position_features)
        if room_part_furthest_distance_features is not None:
            global_inputs.append(room_part_furthest_distance_features)
        if room_part_save_distance_features is not None:
            global_inputs.append(room_part_save_distance_features)
        if room_part_refill_distance_features is not None:
            global_inputs.append(room_part_refill_distance_features)
        if room_part_frontier_distance_features is not None:
            global_inputs.append(room_part_frontier_distance_features)
        global_inputs.append(known_distance_features)
        global_state = (
            self.global_mlp(torch.cat(global_inputs, dim=-1))
            if self.global_mlp is not None
            else X.new_zeros([snapshot_count, self.global_embedding_width])
        )
        if row_count == 0:
            mean_pool = max_pool = X.new_zeros([snapshot_count, self.embedding_width])
        else:
            # pair: [r, k, pair_width], neighbor: [r, k], pair_mask: [r, k, 1]
            pair = self._pair_features(features, dtype)
            frontier_neighbor = features.frontier_features.frontier_neighbor
            local_neighbor = frontier_neighbor.clamp_min(0).to(torch.int64)
            row_neighbor_count = row_count_by_snapshot[row_snapshot_idx].unsqueeze(1)
            neighbor_valid = (frontier_neighbor >= 0) & (local_neighbor < row_neighbor_count)
            neighbor = row_start_by_snapshot[row_snapshot_idx].unsqueeze(1) + local_neighbor
            pair_mask = neighbor_valid.unsqueeze(-1)
            relative_position = self._relative_position_features(features, neighbor)
            single_neighbor = neighbor.shape[1] == 1
            if single_neighbor:
                neighbor = neighbor[:, 0]
                pair_mask = pair_mask[:, 0]
                if pair is not None:
                    pair = pair[:, 0]
                if relative_position is not None:
                    relative_position = relative_position[:, 0]
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
                if relative_position is not None:
                    messages = messages + relative_position
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
        pooled_inputs = []
        if inventory_features is not None:
            # pooled_inputs.append(torch.matmul(features.global_features.inventory.to(torch.float32), self.inventory_embedding))
            pooled_inputs.append(inventory_features)
        if self.features.frontier_mask:
            pooled_inputs.extend([mean_pool, max_pool])
        if connection_features is not None:
            pooled_inputs.append(connection_features)
        if temperature_features is not None:
            pooled_inputs.append(temperature_features)
        if recommended_candidate_features is not None:
            pooled_inputs.append(recommended_candidate_features)
        if lookahead_features is not None:
            pooled_inputs.append(lookahead_features)
        if toilet_crossed_room_features is not None:
            pooled_inputs.append(toilet_crossed_room_features)
        if global_room_position_features is not None:
            pooled_inputs.append(global_room_position_features)
        pooled_inputs.append(known_distance_features)
        pooled_state = (
            self.pooled_mlp(torch.cat(pooled_inputs, dim=-1))
            if self.pooled_mlp is not None
            else X.new_zeros([snapshot_count, self.embedding_width])
        )
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
        save_to_room_utility = torch.sigmoid(
            self.save_to_room_utility_output(X).to(torch.float32)
        )
        save_from_room_utility = torch.sigmoid(
            self.save_from_room_utility_output(X).to(torch.float32)
        )
        refill_to_room_utility = torch.sigmoid(
            self.refill_to_room_utility_output(X).to(torch.float32)
        )
        refill_from_room_utility = torch.sigmoid(
            self.refill_from_room_utility_output(X).to(torch.float32)
        )
        missing_connect_distance = self.missing_connect_distance_output(X).to(torch.float32)
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
            query_connection_invalid, query_connection_mask = self.missing_connect_query_output(
                frontier_state,
                global_state,
                row_count_by_snapshot,
                row_start_by_snapshot,
                features,
                self.num_connection_outputs,
            )
            connection_invalid = torch.where(
                query_connection_mask,
                query_connection_invalid,
                connection_invalid,
            )
        connection_invalid = apply_known_invalid_logits(
            connection_invalid,
            features.global_features.lookahead_connection_invalid,
            "connection",
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
            missing_connect_distance=missing_connect_distance,
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
