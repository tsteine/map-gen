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

# These tensors are all f32 with shape
#    [batch, time, output]  during training,
#    [batch, candidate, output]  during generation
@dataclass
class Predictions:
    # log-odds of invalid door (unconnected):
    door_invalid: torch.Tensor
    # log-odds of invalid connection (lack of return path):
    connection_invalid: torch.Tensor


@dataclass
class BalancePredictions:
    left: torch.Tensor
    right: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor


def get_predictions(raw_preds, output_sizes):
    preds = []
    col = 0
    for size in output_sizes:
        preds.append(raw_preds[:, :, col:(col + size)])
        col += size

    return Predictions(
        door_invalid=preds[0],
        connection_invalid=preds[1],
    )


def normalize(x: torch.Tensor):
    return torch.nn.functional.rms_norm(x, (x.size(-1),))


class FactorizedOutcomeHead(torch.nn.Module):
    def __init__(self, output_metadata, num_geometry_outcomes, embedding_width):
        super().__init__()
        self.embedding_width = embedding_width
        self.num_outputs = len(output_metadata)
        metadata = torch.tensor(output_metadata, dtype=torch.int64).reshape(self.num_outputs, 2)
        self.register_buffer("room_idx", metadata[:, 0])
        self.register_buffer("geometry_outcome_idx", metadata[:, 1])
        self.geometry_outcome_embedding = torch.nn.Parameter(
            torch.randn([num_geometry_outcomes, embedding_width]) / math.sqrt(embedding_width))
        self.state = torch.nn.Linear(embedding_width, embedding_width, bias=False)
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(math.sqrt(embedding_width) / 2)))

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
                self.geometry_outcome_embedding.to(torch.float32), dim=-1)
            pos_embedding_x = torch.nn.functional.normalize(pos_embedding_x.to(torch.float32), dim=-1)
            pos_embedding_y = torch.nn.functional.normalize(pos_embedding_y.to(torch.float32), dim=-1)
            base_query = geometry_outcome_embedding[self.geometry_outcome_idx]
            base_logits = torch.matmul(state, base_query.transpose(0, 1))
        x_logits = torch.matmul(state, pos_embedding_x.transpose(0, 1))
        y_logits = torch.matmul(state, pos_embedding_y.transpose(0, 1))
        room_logits = torch.gather(x_logits, -1, room_x) + torch.gather(y_logits, -1, room_y)
        room_logits = torch.where(room_placed, room_logits, 0.0)
        position_logits = room_logits[..., self.room_idx]
        return (base_logits + position_logits) * torch.exp(
            torch.clamp(self.logit_scale, max=math.log(100.0))
        )


class FrontierModel(torch.nn.Module):
    def __init__(
        self,
        num_rooms,
        output_metadata: OutputMetadata,
        map_x,
        map_y,
        embedding_width,
        hidden_width,
        num_layers,
        frontier_window_size,
        features: FeatureConfig,
    ):
        super().__init__()
        self.features = features
        self.num_rooms = num_rooms
        self.map_x = map_x
        self.map_y = map_y
        self.embedding_width = embedding_width
        self.output_sizes = output_metadata.get_output_sizes()
        self.num_connection_outputs = len(output_metadata.connection)
        self.include_inventory = self.features.inventory
        # self.inventory_embedding = torch.nn.Parameter(
        #     torch.randn([output_metadata.num_room_connection_variants, embedding_width]) / math.sqrt(embedding_width)
        # ) if self.features.inventory else None
        self.orientation_embedding = (
            torch.nn.Embedding(2, embedding_width)
            if self.features.frontier_orientation else None
        )
        self.kind_embedding = (
            torch.nn.Embedding(256, embedding_width)
            if self.features.frontier_kind else None
        )
        node_numeric_width = (
            frontier_window_size**2 * self.features.frontier_occupancy
            + 2 * self.num_connection_outputs * self.features.frontier_connection_reachability
        )
        self.node_numeric = (
            torch.nn.Linear(node_numeric_width, embedding_width, bias=False)
            if node_numeric_width > 0 else None
        )
        self.frontier_window_area = frontier_window_size**2
        self.register_buffer(
            "frontier_occupancy_bits",
            1 << torch.arange(8, dtype=torch.uint8),
            persistent=False,
        )
        pair_width = 3 * self.features.frontier_neighbor_flags
        use_neighbors = self.features.frontier_neighbor
        self.source_message_layers = torch.nn.ModuleList([
            torch.nn.Linear(embedding_width, hidden_width, bias=False)
            for _ in range(num_layers if use_neighbors else 0)
        ])
        self.pair_message_layers = (
            torch.nn.ModuleList([
                torch.nn.Linear(pair_width, hidden_width, bias=False)
                for _ in range(num_layers if use_neighbors else 0)
            ])
            if pair_width > 0 else [None] * (num_layers if use_neighbors else 0)
        )
        self.message_output_layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.GELU(),
                torch.nn.Linear(hidden_width, embedding_width, bias=False),
            )
            for _ in range(num_layers if use_neighbors else 0)
        ])
        self.update_layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(embedding_width * 2, hidden_width, bias=False),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_width, embedding_width, bias=False),
            ) for _ in range(num_layers if use_neighbors else 0)
        ])
        global_width = (
            output_metadata.num_room_connection_variants * self.features.inventory
            + embedding_width * (
                2 * self.features.frontier_mask
                + (
                    self.features.connection_reachability
                    and self.num_connection_outputs > 0
                )
            )
            + int(self.features.temperature)
            + int(self.features.action_candidates)
        )
        self.global_mlp = torch.nn.Sequential(
            torch.nn.Linear(global_width, hidden_width, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_width, embedding_width, bias=False),
        ) if global_width > 0 else None
        self.connection_reachability_embedding = (
            torch.nn.Linear(self.num_connection_outputs, embedding_width, bias=False)
            if self.features.connection_reachability
            and self.num_connection_outputs > 0 else None
        )
        self.frontier_pos_embedding_x = (
            torch.nn.Parameter(
                torch.randn([NUM_COORD_VALUES, embedding_width]) / math.sqrt(embedding_width))
            if self.features.frontier_position else None
        )
        self.frontier_pos_embedding_y = (
            torch.nn.Parameter(
                torch.randn([NUM_COORD_VALUES, embedding_width]) / math.sqrt(embedding_width))
            if self.features.frontier_position else None
        )
        self.frontier_relative_pos_embedding_x = (
            torch.nn.Parameter(
                torch.randn([NUM_COORD_VALUES, hidden_width]) / math.sqrt(hidden_width))
            if self.features.frontier_neighbor_position_embedding else None
        )
        self.frontier_relative_pos_embedding_y = (
            torch.nn.Parameter(
                torch.randn([NUM_COORD_VALUES, hidden_width]) / math.sqrt(hidden_width))
            if self.features.frontier_neighbor_position_embedding else None
        )
        self.pos_embedding_x = torch.nn.Parameter(
            torch.randn([NUM_COORD_VALUES, embedding_width]) / math.sqrt(embedding_width))
        self.pos_embedding_y = torch.nn.Parameter(
            torch.randn([NUM_COORD_VALUES, embedding_width]) / math.sqrt(embedding_width))
        self.door_output = FactorizedOutcomeHead(
            output_metadata.door, output_metadata.num_door_variants, embedding_width)
        self.connection_output = FactorizedOutcomeHead(
            output_metadata.connection, output_metadata.num_connection_variants, embedding_width)

    def _position_embedding(self, x, y, embedding_x, embedding_y, offset=0):
        x = x.to(torch.int64) + offset
        y = y.to(torch.int64) + offset
        return embedding_x[x] + embedding_y[y]

    def _pair_features(self, features, dtype):
        values = []
        if self.features.frontier_neighbor_flags:
            flags = features.frontier_neighbor_pair
            values.append(torch.stack([
                (flags & 1 != 0).to(dtype),
                (flags & 2 != 0).to(dtype),
                (flags & 4 != 0).to(dtype),
            ], dim=-1))
        return torch.cat(values, dim=-1) if values else None

    def _feature_dtype(self, device):
        device_type = device.type
        if torch.is_autocast_enabled(device_type):
            return torch.get_autocast_dtype(device_type)
        return next(self.parameters()).dtype

    def _relative_position_features(self, features):
        if self.frontier_relative_pos_embedding_x is None:
            return None
        node = features.frontier
        neighbor = features.frontier_neighbor.clamp_min(0).to(torch.int64)

        def gather_neighbor(values):
            return torch.gather(
                values.unsqueeze(2).expand(-1, -1, neighbor.shape[2]), 1, neighbor
            )

        raw_x = node[:, :, 1].to(torch.int64)
        raw_y = node[:, :, 2].to(torch.int64)
        raw_x0, raw_x1 = raw_x.unsqueeze(2), gather_neighbor(raw_x)
        raw_y0, raw_y1 = raw_y.unsqueeze(2), gather_neighbor(raw_y)
        return self._position_embedding(
            raw_x1 - raw_x0,
            raw_y1 - raw_y0,
            self.frontier_relative_pos_embedding_x,
            self.frontier_relative_pos_embedding_y,
            COORD_OFFSET,
        )

    def forward(self, features: Features):
        # Shapes below use: b=batch, f=frontiers, k=neighbors, e=embedding width,
        # h=message hidden width.
        # node: [b, f, 5]
        node = features.frontier
        node_mask = node[:, :, 0] != 0
        # numeric: [b, f, numeric_width]
        numeric = []
        dtype = self._feature_dtype(node.device)
        if self.features.frontier_occupancy:
            numeric.append(
                features.frontier_occupancy.unsqueeze(-1)
                .bitwise_and(self.frontier_occupancy_bits)
                .ne(0)
                .flatten(-2)[..., :self.frontier_window_area]
                .to(dtype)
            )
        if self.features.frontier_connection_reachability:
            flags = features.frontier_connection_reachability
            numeric.append(torch.stack([
                (flags & 1 != 0).to(dtype),
                (flags & 2 != 0).to(dtype),
            ], dim=-1).flatten(-2))
        # X: [b, f, e]
        X = node.new_zeros([node.shape[0], node.shape[1], self.embedding_width], dtype=dtype)
        if self.node_numeric is not None:
            numeric = [value.unsqueeze(-1) if value.ndim == 2 else value for value in numeric]
            X = X + self.node_numeric(torch.cat(numeric, dim=-1))
        if self.frontier_pos_embedding_x is not None:
            X = X + self._position_embedding(
                node[:, :, 1],
                node[:, :, 2],
                self.frontier_pos_embedding_x,
                self.frontier_pos_embedding_y,
            )
        if self.orientation_embedding is not None:
            X = X + self.orientation_embedding(node[:, :, 3].to(torch.int64))
        if self.kind_embedding is not None:
            X = X + self.kind_embedding(node[:, :, 4].to(torch.int64))
        # if self.inventory_embedding is not None:
        #     X = X + torch.matmul(
        #         features.inventory.to(torch.float32), self.inventory_embedding
        #     ).unsqueeze(1)
        # if self.connection_reachability_embedding is not None:
        #     X = X + self.connection_reachability_embedding(
        #         features.connection_reachability.to(torch.float32)
        #     ).unsqueeze(1)
        X = X * node_mask.unsqueeze(-1)
        if node.shape[1] == 0:
            mean_pool = max_pool = X.new_zeros([X.shape[0], X.shape[2]])
        else:
            # pair: [b, f, k, pair_width], neighbor: [b, f, k], pair_mask: [b, f, k, 1]
            pair = self._pair_features(features, dtype)
            relative_position = self._relative_position_features(features)
            neighbor = features.frontier_neighbor.clamp_min(0).to(torch.int64)
            pair_mask = (features.frontier_neighbor >= 0).unsqueeze(-1)
            for source_layer, pair_layer, output_layer, update_layer in zip(
                self.source_message_layers,
                self.pair_message_layers,
                self.message_output_layers,
                self.update_layers,
            ):
                # source: [b, f, h]
                source = source_layer(X)
                # Gather each frontier's neighbors: source: [b, f, k, h]
                source = torch.gather(
                    source,
                    1,
                    neighbor.flatten(1).unsqueeze(-1).expand(-1, -1, source.shape[-1]),
                ).view(*neighbor.shape, source.shape[-1])
                messages = source if pair_layer is None else source + pair_layer(pair)
                if relative_position is not None:
                    messages = messages + relative_position
                # messages: [b, f, k, h]
                messages = output_layer(messages) * pair_mask
                # messages: [b, f, k, e]
                messages = messages.sum(2) / pair_mask.sum(2).clamp_min(1)
                # messages [b, f, e]
                X = X + update_layer(torch.cat([X, messages], dim=-1))
                X = X * node_mask.unsqueeze(-1)
            count = node_mask.sum(1, keepdim=True).clamp_min(1)
            mean_pool = X.sum(1) / count
            max_pool = torch.where(node_mask.unsqueeze(-1), X, -torch.inf).max(1).values
            max_pool = torch.where(torch.isfinite(max_pool), max_pool, 0)
        # mean_pool, max_pool, global_state: [b, e]
        global_inputs = []
        if self.include_inventory:
            # global_inputs.append(torch.matmul(features.inventory.to(torch.float32), self.inventory_embedding))
            global_inputs.append(features.inventory.to(X.dtype))
        if self.features.frontier_mask:
            global_inputs.extend([mean_pool, max_pool])
        if self.connection_reachability_embedding is not None:
            global_inputs.append(self.connection_reachability_embedding(
                features.connection_reachability.to(X.dtype)
            ))
        if self.features.temperature:
            global_inputs.append(features.log_temperature.to(X.dtype).unsqueeze(-1))
        if self.features.action_candidates:
            global_inputs.append(features.log_action_candidates.to(X.dtype).unsqueeze(-1))
        global_state = (
            self.global_mlp(torch.cat(global_inputs, dim=-1))
            if self.global_mlp is not None
            else X.new_zeros([X.shape[0], self.embedding_width])
        )
        if self.features.room_position:
            room_x = (features.room_x.to(torch.int64) + COORD_OFFSET).unsqueeze(1)
            room_y = (features.room_y.to(torch.int64) + COORD_OFFSET).unsqueeze(1)
            room_placed = features.room_placed.to(torch.bool).unsqueeze(1)
        else:
            room_x = torch.full([X.shape[0], 1, self.num_rooms], COORD_OFFSET, dtype=torch.int64, device=X.device)
            room_y = room_x
            room_placed = torch.zeros([X.shape[0], 1, self.num_rooms], dtype=torch.bool, device=X.device)
        # X: [b, 1, e]
        X = global_state.unsqueeze(1)
        door = self.door_output(X, room_x, room_y, room_placed, self.pos_embedding_x, self.pos_embedding_y)
        connection = self.connection_output(X, room_x, room_y, room_placed, self.pos_embedding_x, self.pos_embedding_y)
        return get_predictions(torch.cat([door, connection], dim=-1), self.output_sizes)


class BalanceModel(torch.nn.Module):
    def __init__(
        self,
        left_count: int,
        right_count: int,
        up_count: int,
        down_count: int,
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
        self.output_width = (
            left_count * right_count
            + right_count * left_count
            + up_count * down_count
            + down_count * up_count
        )

        layers: list[torch.nn.Module] = []
        input_width = 1
        for _ in range(num_layers):
            layers.extend([
                torch.nn.Linear(input_width, hidden_width),
                torch.nn.GELU(),
            ])
            input_width = hidden_width
        layers.append(torch.nn.Linear(input_width, self.output_width))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, log_temperature: torch.Tensor) -> BalancePredictions:
        raw = self.net(log_temperature.to(next(self.parameters()).dtype).unsqueeze(-1))
        offset = 0
        left_size = self.left_count * self.right_count
        right_size = self.right_count * self.left_count
        up_size = self.up_count * self.down_count
        down_size = self.down_count * self.up_count
        left = raw[:, offset:offset + left_size].reshape(
            log_temperature.shape[0], self.left_count, self.right_count
        )
        offset += left_size
        right = raw[:, offset:offset + right_size].reshape(
            log_temperature.shape[0], self.right_count, self.left_count
        )
        offset += right_size
        up = raw[:, offset:offset + up_size].reshape(
            log_temperature.shape[0], self.up_count, self.down_count
        )
        offset += up_size
        down = raw[:, offset:offset + down_size].reshape(
            log_temperature.shape[0], self.down_count, self.up_count
        )
        return BalancePredictions(left, right, up, down)
