from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from env import AREA_COUNT, Features, OutputMetadata
from train_config import GENERATION_VARIABLE_FLOAT_FIELDS

if TYPE_CHECKING:
    from train_config import FeatureConfig

NUM_COORD_VALUES = 256
COORD_OFFSET = 128


@dataclass(frozen=True)
class FeatureContext:
    features: FeatureConfig
    output_metadata: OutputMetadata
    num_rooms: int
    num_room_parts: int
    num_connection_outputs: int
    door_counts: tuple[int, int, int, int]
    frontier_window_area: int


class GlobalFeature(torch.nn.Module, ABC):
    @classmethod
    @abstractmethod
    def is_enabled(cls, config: FeatureConfig) -> bool:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def tensor_width(cls, context: FeatureContext) -> int:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build(cls, context: FeatureContext) -> GlobalFeature:
        raise NotImplementedError

    @abstractmethod
    def forward(self, features: Features, dtype: torch.dtype) -> torch.Tensor:
        raise NotImplementedError


class FrontierNodeFeature(torch.nn.Module, ABC):
    @classmethod
    @abstractmethod
    def is_enabled(cls, config: FeatureConfig) -> bool:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def tensor_width(cls, context: FeatureContext) -> int:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build(cls, context: FeatureContext) -> FrontierNodeFeature:
        raise NotImplementedError

    @abstractmethod
    def forward(self, features: Features, dtype: torch.dtype) -> torch.Tensor:
        raise NotImplementedError


class FrontierPairFeature(torch.nn.Module, ABC):
    @classmethod
    @abstractmethod
    def is_enabled(cls, config: FeatureConfig) -> bool:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def tensor_width(cls, context: FeatureContext) -> int:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build(cls, context: FeatureContext) -> FrontierPairFeature:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        features: Features,
        neighbor: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        raise NotImplementedError


class InventoryFeature(GlobalFeature):
    @classmethod
    def is_enabled(cls, config: FeatureConfig) -> bool:
        return config.inventory

    @classmethod
    def tensor_width(cls, context: FeatureContext) -> int:
        return context.output_metadata.num_room_connection_variants

    @classmethod
    def build(cls, context: FeatureContext) -> InventoryFeature:
        return cls()

    def forward(self, features: Features, dtype: torch.dtype) -> torch.Tensor:
        return features.global_features.inventory.to(dtype)


class TemperatureFeature(GlobalFeature):
    @classmethod
    def is_enabled(cls, config: FeatureConfig) -> bool:
        return config.temperature

    @classmethod
    def tensor_width(cls, context: FeatureContext) -> int:
        return 1

    @classmethod
    def build(cls, context: FeatureContext) -> TemperatureFeature:
        return cls()

    def forward(self, features: Features, dtype: torch.dtype) -> torch.Tensor:
        return features.global_features.log_temperature.to(dtype).unsqueeze(-1)


class RecommendedCandidatesFeature(GlobalFeature):
    @classmethod
    def is_enabled(cls, config: FeatureConfig) -> bool:
        return config.recommended_candidates

    @classmethod
    def tensor_width(cls, context: FeatureContext) -> int:
        return 1

    @classmethod
    def build(cls, context: FeatureContext) -> RecommendedCandidatesFeature:
        return cls()

    def forward(self, features: Features, dtype: torch.dtype) -> torch.Tensor:
        return features.global_features.log_recommended_candidates.to(dtype).unsqueeze(-1)


class GenerationVariableFloatsFeature(GlobalFeature):
    @classmethod
    def is_enabled(cls, config: FeatureConfig) -> bool:
        return config.generation_variable_floats

    @classmethod
    def tensor_width(cls, context: FeatureContext) -> int:
        return len(GENERATION_VARIABLE_FLOAT_FIELDS)

    @classmethod
    def build(cls, context: FeatureContext) -> GenerationVariableFloatsFeature:
        return cls()

    def forward(self, features: Features, dtype: torch.dtype) -> torch.Tensor:
        return features.global_features.generation_variable_floats.to(dtype)


class LookaheadFeature(GlobalFeature):
    def __init__(
        self,
        left_count: int,
        right_count: int,
        up_count: int,
        down_count: int,
        door_match_width: int,
    ):
        super().__init__()
        self.left_count = left_count
        self.right_count = right_count
        self.up_count = up_count
        self.down_count = down_count
        self.door_match_width = door_match_width
        self.left_embedding = self._door_match_embedding(
            left_count,
            right_count,
            door_match_width,
        )
        self.right_embedding = self._door_match_embedding(
            right_count,
            left_count,
            door_match_width,
        )
        self.up_embedding = self._door_match_embedding(
            up_count,
            down_count,
            door_match_width,
        )
        self.down_embedding = self._door_match_embedding(
            down_count,
            up_count,
            door_match_width,
        )

    @classmethod
    def is_enabled(cls, config: FeatureConfig) -> bool:
        return config.lookahead_outcomes > 0

    @classmethod
    def tensor_width(cls, context: FeatureContext) -> int:
        return context.features.lookahead_outcomes + 2 * context.num_connection_outputs + 4

    @classmethod
    def build(cls, context: FeatureContext) -> LookaheadFeature:
        left_count, right_count, up_count, down_count = context.door_counts
        return cls(
            left_count,
            right_count,
            up_count,
            down_count,
            context.features.lookahead_outcomes,
        )

    def forward(self, features: Features, dtype: torch.dtype) -> torch.Tensor:
        left, right, up, down = torch.split(
            features.global_features.lookahead_door_match,
            [self.left_count, self.right_count, self.up_count, self.down_count],
            dim=-1,
        )
        door_match_features = (
            self._direction_features(left, self.left_embedding, dtype)
            + self._direction_features(right, self.right_embedding, dtype)
            + self._direction_features(up, self.up_embedding, dtype)
            + self._direction_features(down, self.down_embedding, dtype)
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
        phantoon_features = torch.stack(
            [
                (features.global_features.lookahead_phantoon_invalid == 0).to(dtype),
                (features.global_features.lookahead_phantoon_invalid == 1).to(dtype),
            ],
            dim=-1,
        ).flatten(1)
        return torch.cat(
            [door_match_features, connection_features, toilet_features, phantoon_features],
            dim=-1,
        )

    @staticmethod
    def _door_match_embedding(
        source_count: int,
        partner_count: int,
        width: int,
    ) -> torch.nn.Parameter:
        return torch.nn.Parameter(
            torch.randn([source_count, partner_count + 1, width]) / math.sqrt(width)
        )

    def _direction_features(
        self,
        matches: torch.Tensor,
        embedding: torch.nn.Parameter,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if matches.shape[-1] == 0:
            return matches.new_zeros([matches.shape[0], self.door_match_width], dtype=dtype)
        known = matches >= 0
        safe_matches = matches.clamp(min=0).to(torch.int64)
        source_idx = torch.arange(
            embedding.shape[0],
            dtype=torch.int64,
            device=matches.device,
        ).unsqueeze(0)
        values = embedding.to(dtype)[source_idx, safe_matches]
        return torch.sum(values * known.unsqueeze(-1), dim=1)


class ConnectionReachabilityFeature(GlobalFeature):
    def __init__(self, num_connection_outputs: int, width: int):
        super().__init__()
        self.embedding = torch.nn.Linear(num_connection_outputs, width, bias=False)

    @classmethod
    def is_enabled(cls, config: FeatureConfig) -> bool:
        return config.connection_reachability > 0

    @classmethod
    def tensor_width(cls, context: FeatureContext) -> int:
        return context.features.connection_reachability

    @classmethod
    def build(cls, context: FeatureContext) -> ConnectionReachabilityFeature:
        return cls(
            context.num_connection_outputs,
            context.features.connection_reachability,
        )

    def forward(self, features: Features, dtype: torch.dtype) -> torch.Tensor:
        return self.embedding(features.global_features.connection_reachability.to(dtype))


class ToiletCrossedRoomFeature(GlobalFeature):
    def __init__(self, num_rooms: int, width: int):
        super().__init__()
        self.num_rooms = num_rooms
        self.embedding = torch.nn.Embedding(num_rooms + 1, width)

    @classmethod
    def is_enabled(cls, config: FeatureConfig) -> bool:
        return config.toilet_crossed_room > 0

    @classmethod
    def tensor_width(cls, context: FeatureContext) -> int:
        return context.features.toilet_crossed_room

    @classmethod
    def build(cls, context: FeatureContext) -> ToiletCrossedRoomFeature:
        return cls(context.num_rooms, context.features.toilet_crossed_room)

    def forward(self, features: Features, dtype: torch.dtype) -> torch.Tensor:
        crossed_room = features.global_features.toilet_crossed_room_idx.to(torch.int64)
        return self.embedding(crossed_room + 1).squeeze(-2).to(dtype)


class GlobalRoomPositionFeature(GlobalFeature):
    def __init__(
        self,
        room_connection_variant_idx: list[int],
        num_room_connection_variants: int,
        width: int,
    ):
        super().__init__()
        self.register_buffer(
            "room_connection_variant_idx",
            torch.tensor(room_connection_variant_idx, dtype=torch.int64),
        )
        self.embedding_x = torch.nn.Parameter(
            torch.randn([num_room_connection_variants, NUM_COORD_VALUES, width])
            / math.sqrt(width)
        )
        self.embedding_y = torch.nn.Parameter(
            torch.randn([num_room_connection_variants, NUM_COORD_VALUES, width])
            / math.sqrt(width)
        )

    @classmethod
    def is_enabled(cls, config: FeatureConfig) -> bool:
        return config.global_room_position > 0

    @classmethod
    def tensor_width(cls, context: FeatureContext) -> int:
        return context.features.global_room_position

    @classmethod
    def build(cls, context: FeatureContext) -> GlobalRoomPositionFeature:
        return cls(
            context.output_metadata.room_connection_variant_idx,
            context.output_metadata.num_room_connection_variants,
            context.features.global_room_position,
        )

    def forward(self, features: Features, dtype: torch.dtype) -> torch.Tensor:
        room_x = features.global_features.room_x.to(torch.int64) + COORD_OFFSET
        room_y = features.global_features.room_y.to(torch.int64) + COORD_OFFSET
        room_connection_variant_idx = (
            self.room_connection_variant_idx.to(room_x.device).unsqueeze(0).expand_as(room_x)
        )
        room_position = (
            self.embedding_x[room_connection_variant_idx, room_x]
            * self.embedding_y[room_connection_variant_idx, room_y]
        ).to(dtype)
        placed = features.global_features.room_placed.to(dtype).unsqueeze(-1)
        placed_count = placed.sum(dim=1).clamp_min(1)
        return (room_position * placed).sum(dim=1) / torch.sqrt(placed_count)


class DistanceSlotFeature(GlobalFeature):
    def __init__(self, slot_count: int, width: int):
        super().__init__()
        self.width = width
        self.embedding = torch.nn.Embedding(NUM_COORD_VALUES, width)
        self.slot_count = slot_count

    @classmethod
    def _tensor_width(cls, slot_count: int, width: int) -> int:
        return slot_count * width

    def _embed_distances(self, distances: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        distances = distances.to(torch.int64)
        return self.embedding(distances).flatten(1).to(dtype)


class RoomPartFurthestDistanceFeature(DistanceSlotFeature):
    @classmethod
    def is_enabled(cls, config: FeatureConfig) -> bool:
        return config.room_part_furthest_distance > 0

    @classmethod
    def tensor_width(cls, context: FeatureContext) -> int:
        return cls._tensor_width(
            2 * context.num_room_parts,
            context.features.room_part_furthest_distance,
        )

    @classmethod
    def build(cls, context: FeatureContext) -> RoomPartFurthestDistanceFeature:
        return cls(
            2 * context.num_room_parts,
            context.features.room_part_furthest_distance,
        )

    def forward(self, features: Features, dtype: torch.dtype) -> torch.Tensor:
        return self._embed_distances(
            torch.cat(
                [
                    features.global_features.room_part_furthest_destination,
                    features.global_features.room_part_furthest_source,
                ],
                dim=-1,
            ),
            dtype,
        )


class RoomPartSaveDistanceFeature(DistanceSlotFeature):
    @classmethod
    def is_enabled(cls, config: FeatureConfig) -> bool:
        return config.room_part_save_distance > 0

    @classmethod
    def tensor_width(cls, context: FeatureContext) -> int:
        return cls._tensor_width(
            2 * context.num_room_parts,
            context.features.room_part_save_distance,
        )

    @classmethod
    def build(cls, context: FeatureContext) -> RoomPartSaveDistanceFeature:
        return cls(2 * context.num_room_parts, context.features.room_part_save_distance)

    def forward(self, features: Features, dtype: torch.dtype) -> torch.Tensor:
        return self._embed_distances(
            torch.cat(
                [
                    features.global_features.room_part_save_from_room_distance,
                    features.global_features.room_part_save_to_room_distance,
                ],
                dim=-1,
            ),
            dtype,
        )


class RoomPartRefillDistanceFeature(DistanceSlotFeature):
    @classmethod
    def is_enabled(cls, config: FeatureConfig) -> bool:
        return config.room_part_refill_distance > 0

    @classmethod
    def tensor_width(cls, context: FeatureContext) -> int:
        return cls._tensor_width(
            2 * context.num_room_parts,
            context.features.room_part_refill_distance,
        )

    @classmethod
    def build(cls, context: FeatureContext) -> RoomPartRefillDistanceFeature:
        return cls(2 * context.num_room_parts, context.features.room_part_refill_distance)

    def forward(self, features: Features, dtype: torch.dtype) -> torch.Tensor:
        return self._embed_distances(
            torch.cat(
                [
                    features.global_features.room_part_refill_from_room_distance,
                    features.global_features.room_part_refill_to_room_distance,
                ],
                dim=-1,
            ),
            dtype,
        )


class RoomPartFrontierDistanceFeature(DistanceSlotFeature):
    @classmethod
    def is_enabled(cls, config: FeatureConfig) -> bool:
        return config.room_part_frontier_distance > 0

    @classmethod
    def tensor_width(cls, context: FeatureContext) -> int:
        return cls._tensor_width(
            2 * context.num_room_parts,
            context.features.room_part_frontier_distance,
        )

    @classmethod
    def build(cls, context: FeatureContext) -> RoomPartFrontierDistanceFeature:
        return cls(
            2 * context.num_room_parts,
            context.features.room_part_frontier_distance,
        )

    def forward(self, features: Features, dtype: torch.dtype) -> torch.Tensor:
        return self._embed_distances(
            torch.cat(
                [
                    features.global_features.room_part_frontier_from_room_distance,
                    features.global_features.room_part_frontier_to_room_distance,
                ],
                dim=-1,
            ),
            dtype,
        )


class KnownDistanceFeature(DistanceSlotFeature):
    @classmethod
    def is_enabled(cls, config: FeatureConfig) -> bool:
        return config.known_distance > 0

    @classmethod
    def tensor_width(cls, context: FeatureContext) -> int:
        return cls._tensor_width(4 * context.num_room_parts, context.features.known_distance)

    @classmethod
    def build(cls, context: FeatureContext) -> KnownDistanceFeature:
        return cls(4 * context.num_room_parts, context.features.known_distance)

    def forward(self, features: Features, dtype: torch.dtype) -> torch.Tensor:
        return self._embed_distances(
            torch.cat(
                [
                    features.global_features.known_save_from_room_distance,
                    features.global_features.known_save_to_room_distance,
                    features.global_features.known_refill_from_room_distance,
                    features.global_features.known_refill_to_room_distance,
                ],
                dim=-1,
            ),
            dtype,
        )


class AreaStateFeature(GlobalFeature):
    @classmethod
    def is_enabled(cls, config: FeatureConfig) -> bool:
        return config.area_state

    @classmethod
    def tensor_width(cls, context: FeatureContext) -> int:
        return AREA_COUNT * 8 + 1

    @classmethod
    def build(cls, context: FeatureContext) -> AreaStateFeature:
        return cls()

    def forward(self, features: Features, dtype: torch.dtype) -> torch.Tensor:
        area = features.global_features
        values = [
            area.area_used.to(dtype),
            area.area_min_x.to(dtype),
            area.area_max_x.to(dtype),
            area.area_min_y.to(dtype),
            area.area_max_y.to(dtype),
            area.area_connected_components.to(dtype),
            area.area_size.to(dtype),
            area.area_map_station_count.to(dtype),
            area.area_crossings.to(dtype),
        ]
        return torch.cat(values, dim=-1)


GLOBAL_FEATURES: list[type[GlobalFeature]] = [
    InventoryFeature,
    TemperatureFeature,
    RecommendedCandidatesFeature,
    GenerationVariableFloatsFeature,
    LookaheadFeature,
    ConnectionReachabilityFeature,
    ToiletCrossedRoomFeature,
    GlobalRoomPositionFeature,
    RoomPartFurthestDistanceFeature,
    RoomPartSaveDistanceFeature,
    RoomPartRefillDistanceFeature,
    RoomPartFrontierDistanceFeature,
    KnownDistanceFeature,
    AreaStateFeature,
]


class FrontierNodeNumericFeature(FrontierNodeFeature):
    def __init__(
        self,
        frontier_window_area: int,
        num_connection_outputs: int,
        include_occupancy: bool,
        include_connection_reachability: bool,
    ):
        super().__init__()
        self.frontier_window_area = frontier_window_area
        self.num_connection_outputs = num_connection_outputs
        self.include_occupancy = include_occupancy
        self.include_connection_reachability = include_connection_reachability
        self.register_buffer(
            "frontier_occupancy_bits",
            1 << torch.arange(8, dtype=torch.uint8),
            persistent=False,
        )

    @classmethod
    def is_enabled(cls, config: FeatureConfig) -> bool:
        return config.frontier_occupancy or config.frontier_connection_reachability

    @classmethod
    def tensor_width(cls, context: FeatureContext) -> int:
        occupancy_width = context.frontier_window_area * int(
            context.features.frontier_occupancy
        )
        reachability_width = (
            2
            * context.num_connection_outputs
            * int(context.features.frontier_connection_reachability)
        )
        return occupancy_width + reachability_width

    @classmethod
    def build(cls, context: FeatureContext) -> FrontierNodeNumericFeature:
        return cls(
            context.frontier_window_area,
            context.num_connection_outputs,
            context.features.frontier_occupancy,
            context.features.frontier_connection_reachability,
        )

    def forward(self, features: Features, dtype: torch.dtype) -> torch.Tensor:
        values = []
        if self.include_occupancy:
            values.append(
                features.frontier_features.frontier_occupancy.unsqueeze(-1)
                .bitwise_and(self.frontier_occupancy_bits)
                .ne(0)
                .flatten(-2)[..., : self.frontier_window_area]
                .to(dtype)
            )
        if self.include_connection_reachability:
            flags = features.frontier_features.frontier_connection_reachability
            values.append(
                torch.stack(
                    [
                        (flags & 1 != 0).to(dtype),
                        (flags & 2 != 0).to(dtype),
                    ],
                    dim=-1,
                ).flatten(-2)
            )
        return torch.cat(values, dim=-1)


class FrontierPositionFeature(FrontierNodeFeature):
    def __init__(self, width: int):
        super().__init__()
        self.embedding_x = torch.nn.Parameter(
            torch.randn([NUM_COORD_VALUES, width]) / math.sqrt(width)
        )
        self.embedding_y = torch.nn.Parameter(
            torch.randn([NUM_COORD_VALUES, width]) / math.sqrt(width)
        )

    @classmethod
    def is_enabled(cls, config: FeatureConfig) -> bool:
        return config.frontier_position > 0

    @classmethod
    def tensor_width(cls, context: FeatureContext) -> int:
        return context.features.frontier_position

    @classmethod
    def build(cls, context: FeatureContext) -> FrontierPositionFeature:
        return cls(context.features.frontier_position)

    def forward(self, features: Features, dtype: torch.dtype) -> torch.Tensor:
        frontier = features.frontier_features.frontier
        x = frontier[:, 1].to(torch.int64)
        y = frontier[:, 2].to(torch.int64)
        return (self.embedding_x[x] * self.embedding_y[y]).to(dtype)


class FrontierOrientationFeature(FrontierNodeFeature):
    def __init__(self, width: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(2, width)

    @classmethod
    def is_enabled(cls, config: FeatureConfig) -> bool:
        return config.frontier_orientation > 0

    @classmethod
    def tensor_width(cls, context: FeatureContext) -> int:
        return context.features.frontier_orientation

    @classmethod
    def build(cls, context: FeatureContext) -> FrontierOrientationFeature:
        return cls(context.features.frontier_orientation)

    def forward(self, features: Features, dtype: torch.dtype) -> torch.Tensor:
        frontier = features.frontier_features.frontier
        return self.embedding(frontier[:, 3].to(torch.int64)).to(dtype)


class FrontierKindFeature(FrontierNodeFeature):
    def __init__(self, width: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(256, width)

    @classmethod
    def is_enabled(cls, config: FeatureConfig) -> bool:
        return config.frontier_kind > 0

    @classmethod
    def tensor_width(cls, context: FeatureContext) -> int:
        return context.features.frontier_kind

    @classmethod
    def build(cls, context: FeatureContext) -> FrontierKindFeature:
        return cls(context.features.frontier_kind)

    def forward(self, features: Features, dtype: torch.dtype) -> torch.Tensor:
        frontier = features.frontier_features.frontier
        return self.embedding(frontier[:, 4].to(torch.int64)).to(dtype)


class FrontierDoorVariantFeature(FrontierNodeFeature):
    def __init__(self, num_door_variants: int, width: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_door_variants, width)

    @classmethod
    def is_enabled(cls, config: FeatureConfig) -> bool:
        return config.frontier_door_variant > 0

    @classmethod
    def tensor_width(cls, context: FeatureContext) -> int:
        return context.features.frontier_door_variant

    @classmethod
    def build(cls, context: FeatureContext) -> FrontierDoorVariantFeature:
        return cls(
            context.output_metadata.num_door_variants,
            context.features.frontier_door_variant,
        )

    def forward(self, features: Features, dtype: torch.dtype) -> torch.Tensor:
        return self.embedding(
            features.frontier_features.frontier_door_variant.to(torch.int64)
        ).to(dtype)


class FrontierAreaFeature(FrontierNodeFeature):
    def __init__(self, width: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(AREA_COUNT, width)

    @classmethod
    def is_enabled(cls, config: FeatureConfig) -> bool:
        return config.frontier_area > 0

    @classmethod
    def tensor_width(cls, context: FeatureContext) -> int:
        return context.features.frontier_area

    @classmethod
    def build(cls, context: FeatureContext) -> FrontierAreaFeature:
        return cls(context.features.frontier_area)

    def forward(self, features: Features, dtype: torch.dtype) -> torch.Tensor:
        return self.embedding(features.frontier_features.frontier_area.to(torch.int64)).to(dtype)


class FrontierNeighborFlagsFeature(FrontierPairFeature):
    @classmethod
    def is_enabled(cls, config: FeatureConfig) -> bool:
        return config.frontier_neighbor_flags

    @classmethod
    def tensor_width(cls, context: FeatureContext) -> int:
        return 3

    @classmethod
    def build(cls, context: FeatureContext) -> FrontierNeighborFlagsFeature:
        return cls()

    def forward(
        self,
        features: Features,
        neighbor: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        flags = features.frontier_features.frontier_neighbor_pair
        return torch.stack(
            [
                (flags & 1 != 0).to(dtype),
                (flags & 2 != 0).to(dtype),
                (flags & 4 != 0).to(dtype),
            ],
            dim=-1,
        )


class FrontierRelativePositionFeature(FrontierPairFeature):
    def __init__(self, width: int):
        super().__init__()
        self.embedding_x = torch.nn.Parameter(
            torch.randn([NUM_COORD_VALUES, width]) / math.sqrt(width)
        )
        self.embedding_y = torch.nn.Parameter(
            torch.randn([NUM_COORD_VALUES, width]) / math.sqrt(width)
        )

    @classmethod
    def is_enabled(cls, config: FeatureConfig) -> bool:
        return config.frontier_neighbor_position_embedding > 0

    @classmethod
    def tensor_width(cls, context: FeatureContext) -> int:
        return context.features.frontier_neighbor_position_embedding

    @classmethod
    def build(cls, context: FeatureContext) -> FrontierRelativePositionFeature:
        return cls(context.features.frontier_neighbor_position_embedding)

    def forward(
        self,
        features: Features,
        neighbor: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        node = features.frontier_features.frontier
        raw_x = node[:, 1].to(torch.int64)
        raw_y = node[:, 2].to(torch.int64)
        raw_x0, raw_x1 = raw_x.unsqueeze(1), raw_x[neighbor]
        raw_y0, raw_y1 = raw_y.unsqueeze(1), raw_y[neighbor]
        x = raw_x1 - raw_x0 + COORD_OFFSET
        y = raw_y1 - raw_y0 + COORD_OFFSET
        return self.embedding_x[x].to(dtype) + self.embedding_y[y].to(dtype)


FRONTIER_NODE_FEATURES: list[type[FrontierNodeFeature]] = [
    FrontierNodeNumericFeature,
    FrontierPositionFeature,
    FrontierOrientationFeature,
    FrontierKindFeature,
    FrontierDoorVariantFeature,
    FrontierAreaFeature,
]

FRONTIER_PAIR_FEATURES: list[type[FrontierPairFeature]] = [
    FrontierNeighborFlagsFeature,
    FrontierRelativePositionFeature,
]
