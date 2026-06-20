from pathlib import Path
import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Schedule(StrictBaseModel):
    linear: list[float] | None = None
    log: list[float] | None = None


type ScheduleableFloat = float | Schedule
type ScheduleableInt = int | Schedule


class ModelConfig(StrictBaseModel):
    compile: bool
    autocast: bool
    generation_autocast: bool
    embedding_width: int
    global_embedding_width: int
    global_room_position_embedding_width: int
    hidden_width: int
    proposal_hidden_width: int
    missing_connect_hidden_width: int
    door_match_embedding_width: int
    toilet_crossed_room_embedding_width: int
    num_layers: int


class AdamOptimizerConfig(StrictBaseModel):
    type: Literal["adam"]
    lr: ScheduleableFloat
    beta1: float
    beta2: float


class AdamParamsConfig(StrictBaseModel):
    lr: ScheduleableFloat
    beta1: float
    beta2: float


class MuonParamsConfig(StrictBaseModel):
    lr: ScheduleableFloat
    momentum: float
    nesterov: bool
    backend: Literal["newtonschulz5"]
    backend_steps: int


class MuonOptimizerConfig(StrictBaseModel):
    type: Literal["muon"]
    adam: AdamParamsConfig
    muon: MuonParamsConfig


type OptimizerConfig = AdamOptimizerConfig | MuonOptimizerConfig


class BalanceModelConfig(StrictBaseModel):
    hidden_width: int
    num_layers: int


class GenerationConfig(StrictBaseModel):
    num_environments: int
    num_iterations: int
    num_devices: int
    pipeline_groups: int
    recommended_candidates: ScheduleableInt
    shortlist_candidates: ScheduleableInt
    temperature: ScheduleableFloat
    proposal_temperature: ScheduleableFloat
    reward_door: ScheduleableFloat
    reward_connection: ScheduleableFloat
    reward_toilet: ScheduleableFloat
    reward_balance: ScheduleableFloat
    reward_toilet_balance: ScheduleableFloat
    reward_frontier: ScheduleableFloat
    reward_graph_diameter: ScheduleableFloat
    reward_save_distance: ScheduleableFloat
    reward_refill_distance: ScheduleableFloat
    reward_missing_connect_distance: ScheduleableFloat
    frontier_neighbor_algorithm: Literal["delaunay", "nearest", "nearest-exclusive"]
    frontier_neighbor_count: int
    frontier_window_size: int
    missing_connect_query_frontier_count: int
    candidate_spatial_cell_size: int
    num_threads: int | None


class FeatureConfig(StrictBaseModel):
    inventory: bool
    temperature: bool
    recommended_candidates: bool
    lookahead_outcomes: bool
    room_position: bool
    global_room_position: bool
    room_part_furthest_distance: bool
    room_part_save_distance: bool
    room_part_refill_distance: bool
    room_part_frontier_distance: bool
    frontier_mask: bool
    frontier_position: bool
    frontier_orientation: bool
    frontier_kind: bool
    frontier_occupancy: bool
    frontier_neighbor: bool
    frontier_neighbor_position_embedding: bool
    frontier_neighbor_flags: bool
    connection_reachability: bool
    frontier_connection_reachability: bool
    missing_connect_query: bool
    toilet_crossed_room: bool


class TrainConfig(StrictBaseModel):
    batch_size: int
    fresh_pass_factor: float
    replay_pass_factor: float
    sample_period: int
    episodes_per_file: int
    hist_c: float
    door_weight: float
    connection_weight: float
    toilet_weight: float
    balance_weight: float
    toilet_balance_weight: float
    avg_frontiers_weight: float
    graph_diameter_weight: float
    save_distance_weight: float
    refill_distance_weight: float
    missing_connect_distance_weight: float
    proposal_weight: float
    ema_decay: ScheduleableFloat
    pipeline_groups: int
    gradient_accumulation_steps: int
    shuffle_buffer_batches: int


class Config(StrictBaseModel):
    experiment_name: str
    room_set: Path
    map_size: tuple[int, int]
    knot_episodes: list[int]
    checkpoint_period: int
    visualize: int
    distance_proximity_scale: float
    model: ModelConfig
    optimizer: OptimizerConfig
    balance_model: BalanceModelConfig
    balance_optimizer: AdamOptimizerConfig
    generation: GenerationConfig
    features: FeatureConfig
    train: TrainConfig


def instantiate_scheduleable_config(config: Config, num_episodes: int) -> Config:
    knot_episodes = config.knot_episodes

    def instantiate_model(model: BaseModel, path: str) -> BaseModel:
        updates = {}
        for field_name, field_info in model.__class__.model_fields.items():
            value = getattr(model, field_name)
            field_path = f"{path}.{field_name}"
            if field_info.annotation is ScheduleableFloat:
                updates[field_name] = instantiate_float(value, field_path)
            elif field_info.annotation is ScheduleableInt:
                updates[field_name] = instantiate_int(value, field_path)
            elif isinstance(value, BaseModel):
                updates[field_name] = instantiate_model(value, field_path)
        return model.model_copy(update=updates)

    def instantiate_int(value: ScheduleableInt, path: str) -> int:
        result = round(instantiate_float(value, path))
        if path in {
            "config.generation.recommended_candidates",
            "config.generation.shortlist_candidates",
        }:
            if result < 0:
                raise ValueError(f"{path} must round to an integer greater than or equal to zero")
        elif result <= 0:
            raise ValueError(f"{path} must round to an integer greater than zero")
        return result

    def instantiate_float(value: ScheduleableFloat, path: str) -> float:
        if isinstance(value, Schedule):
            if (value.linear is None) == (value.log is None):
                raise ValueError(f"{path} must have exactly one schedule value: 'linear' or 'log'")
            x = value.linear if value.linear is not None else value.log
            if len(x) != len(knot_episodes):
                raise ValueError(
                    f"{path} has {len(x)} schedule value(s), but knot_episodes has {len(knot_episodes)} knot(s)"
                )
            if value.linear is not None:
                return float(np.interp(num_episodes, knot_episodes, x))
            return float(np.exp(np.interp(num_episodes, knot_episodes, np.log(x))))
        return float(value)

    return instantiate_model(config, "config")


def validate_config(config: Config) -> None:
    if not config.knot_episodes:
        raise ValueError("knot_episodes must contain at least one episode count")
    if config.knot_episodes[-1] <= 0:
        raise ValueError("last knot_episodes value must be greater than zero")
    if config.checkpoint_period <= 0:
        raise ValueError("checkpoint_period must be greater than zero")
    if config.visualize < 0:
        raise ValueError("visualize must be greater than or equal to zero")
    if config.distance_proximity_scale <= 0:
        raise ValueError("distance_proximity_scale must be greater than zero")
    if config.model.global_embedding_width <= 0:
        raise ValueError("model.global_embedding_width must be greater than zero")
    if config.model.global_room_position_embedding_width <= 0:
        raise ValueError("model.global_room_position_embedding_width must be greater than zero")
    if config.model.proposal_hidden_width <= 0:
        raise ValueError("model.proposal_hidden_width must be greater than zero")
    if config.model.missing_connect_hidden_width <= 0:
        raise ValueError("model.missing_connect_hidden_width must be greater than zero")
    if (
        config.features.toilet_crossed_room
        and config.model.toilet_crossed_room_embedding_width <= 0
    ):
        raise ValueError("model.toilet_crossed_room_embedding_width must be greater than zero")
    validate_optimizer_config(config.optimizer, "optimizer")
    validate_optimizer_config(config.balance_optimizer, "balance_optimizer")
    if config.generation.num_iterations <= 0:
        raise ValueError("generation.num_iterations must be greater than zero")
    if config.generation.num_devices <= 0:
        raise ValueError("generation.num_devices must be greater than zero")
    if config.generation.pipeline_groups <= 0:
        raise ValueError("generation.pipeline_groups must be greater than zero")
    if (
        isinstance(config.generation.recommended_candidates, int)
        and config.generation.recommended_candidates <= 0
    ):
        raise ValueError("generation.recommended_candidates must be greater than zero")
    if (
        isinstance(config.generation.shortlist_candidates, int)
        and config.generation.shortlist_candidates < 0
    ):
        raise ValueError("generation.shortlist_candidates must be greater than or equal to zero")
    if (
        isinstance(config.generation.recommended_candidates, int)
        and isinstance(config.generation.shortlist_candidates, int)
        and config.generation.shortlist_candidates < config.generation.recommended_candidates
    ):
        raise ValueError("generation.shortlist_candidates must be at least recommended_candidates")
    if config.generation.num_devices > config.generation.num_environments:
        raise ValueError("generation.num_devices must not exceed generation.num_environments")
    num_generation_groups = config.generation.num_devices * config.generation.pipeline_groups
    if config.generation.num_environments % num_generation_groups != 0:
        raise ValueError(
            "generation.num_environments must be divisible by generation.num_devices * generation.pipeline_groups"
        )
    if config.generation.frontier_neighbor_count < 0:
        raise ValueError(
            "generation.frontier_neighbor_count must be greater than or equal to zero"
        )
    if config.generation.frontier_window_size < 0:
        raise ValueError("generation.frontier_window_size must be greater than or equal to zero")
    if config.generation.missing_connect_query_frontier_count <= 0:
        raise ValueError(
            "generation.missing_connect_query_frontier_count must be greater than zero"
        )
    if config.generation.candidate_spatial_cell_size <= 0:
        raise ValueError("generation.candidate_spatial_cell_size must be greater than zero")
    validate_nonnegative_scheduleable_float(
        config.generation.reward_toilet_balance,
        "generation.reward_toilet_balance",
    )
    validate_nonnegative_scheduleable_float(
        config.generation.reward_graph_diameter,
        "generation.reward_graph_diameter",
    )
    validate_nonnegative_scheduleable_float(
        config.generation.reward_save_distance,
        "generation.reward_save_distance",
    )
    validate_nonnegative_scheduleable_float(
        config.generation.reward_refill_distance,
        "generation.reward_refill_distance",
    )
    validate_nonnegative_scheduleable_float(
        config.generation.reward_missing_connect_distance,
        "generation.reward_missing_connect_distance",
    )
    if config.generation.num_threads is not None and config.generation.num_threads <= 0:
        raise ValueError("generation.num_threads must be greater than zero")
    if (
        config.generation.num_threads is not None
        and config.generation.num_threads % config.generation.pipeline_groups != 0
    ):
        raise ValueError("generation.num_threads must be divisible by generation.pipeline_groups")
    if config.train.sample_period <= 0:
        raise ValueError("train.sample_period must be greater than zero")
    if config.train.pipeline_groups <= 0:
        raise ValueError("train.pipeline_groups must be greater than zero")
    if config.train.gradient_accumulation_steps <= 0:
        raise ValueError("train.gradient_accumulation_steps must be greater than zero")
    if config.train.shuffle_buffer_batches <= 0:
        raise ValueError("train.shuffle_buffer_batches must be greater than zero")
    if config.train.proposal_weight < 0:
        raise ValueError("train.proposal_weight must be greater than or equal to zero")
    if config.train.toilet_weight < 0:
        raise ValueError("train.toilet_weight must be greater than or equal to zero")
    if config.train.toilet_balance_weight < 0:
        raise ValueError("train.toilet_balance_weight must be greater than or equal to zero")
    if config.train.avg_frontiers_weight < 0:
        raise ValueError("train.avg_frontiers_weight must be greater than or equal to zero")
    if config.train.graph_diameter_weight < 0:
        raise ValueError("train.graph_diameter_weight must be greater than or equal to zero")
    if config.train.save_distance_weight < 0:
        raise ValueError("train.save_distance_weight must be greater than or equal to zero")
    if config.train.refill_distance_weight < 0:
        raise ValueError("train.refill_distance_weight must be greater than or equal to zero")
    if config.train.missing_connect_distance_weight < 0:
        raise ValueError(
            "train.missing_connect_distance_weight must be greater than or equal to zero"
        )
    validate_ema_decay_config(config.train.ema_decay, "train.ema_decay", config.knot_episodes)
    if (
        config.generation.num_threads is not None
        and config.generation.num_threads % config.train.pipeline_groups != 0
    ):
        raise ValueError("generation.num_threads must be divisible by train.pipeline_groups")
    if (
        config.features.frontier_position
        or config.features.frontier_orientation
        or config.features.frontier_kind
        or config.features.frontier_occupancy
        or config.features.frontier_neighbor
        or config.features.frontier_connection_reachability
        or config.features.missing_connect_query
    ) and not config.features.frontier_mask:
        raise ValueError("frontier query and frontier features require features.frontier_mask")
    if (
        config.features.inventory or config.features.connection_reachability
    ) and not config.features.frontier_mask:
        raise ValueError("start-of-network features require features.frontier_mask")
    if (
        config.features.frontier_neighbor_position_embedding
        or config.features.frontier_neighbor_flags
    ) and not config.features.frontier_neighbor:
        raise ValueError("frontier neighbor pair features require features.frontier_neighbor")
    if config.features.global_room_position and not config.features.room_position:
        raise ValueError("features.global_room_position requires features.room_position")


def episodes_per_round(config: Config) -> int:
    value = config.generation.num_iterations * config.generation.num_environments
    if config.train.fresh_pass_factor != 0.0 and value % config.train.batch_size != 0:
        raise ValueError(
            "train.batch_size must evenly divide the number of episodes generated per round when "
            "train.fresh_pass_factor is non-zero"
        )
    return value


def validate_optimizer_config(config: OptimizerConfig, path: str) -> None:
    if isinstance(config, AdamOptimizerConfig):
        validate_adam_params(config, path)
        return
    validate_adam_params(config.adam, f"{path}.adam")
    validate_muon_params(config.muon, f"{path}.muon")


def validate_adam_params(config: AdamOptimizerConfig | AdamParamsConfig, path: str) -> None:
    validate_beta(config.beta1, f"{path}.beta1")
    validate_beta(config.beta2, f"{path}.beta2")


def validate_muon_params(config: MuonParamsConfig, path: str) -> None:
    if config.momentum < 0.0 or config.momentum >= 1.0:
        raise ValueError(
            f"{path}.momentum must be greater than or equal to zero and less than one"
        )
    if config.backend_steps <= 0:
        raise ValueError(f"{path}.backend_steps must be greater than zero")


def validate_beta(value: float, path: str) -> None:
    if value < 0.0 or value >= 1.0:
        raise ValueError(f"{path} must be greater than or equal to zero and less than one")


def validate_nonnegative_scheduleable_float(value: ScheduleableFloat, path: str) -> None:
    if isinstance(value, Schedule):
        values = value.linear if value.linear is not None else value.log
        if values is None:
            return
        for index, item in enumerate(values):
            if item < 0:
                raise ValueError(f"{path}[{index}] must be greater than or equal to zero")
        return
    if value < 0:
        raise ValueError(f"{path} must be greater than or equal to zero")


def validate_ema_decay(value: float, path: str) -> None:
    if not math.isfinite(value) or value < 0.0 or value >= 1.0:
        raise ValueError(
            f"{path} must be finite, greater than or equal to zero, and less than one"
        )


def validate_ema_decay_config(
    value: ScheduleableFloat, path: str, knot_episodes: list[int]
) -> None:
    if isinstance(value, Schedule):
        if (value.linear is None) == (value.log is None):
            raise ValueError(f"{path} must have exactly one schedule value: 'linear' or 'log'")
        values = value.linear if value.linear is not None else value.log
        if len(values) != len(knot_episodes):
            raise ValueError(
                f"{path} has {len(values)} schedule value(s), but knot_episodes has {len(knot_episodes)} knot(s)"
            )
        for index, knot_value in enumerate(values):
            validate_ema_decay(knot_value, f"{path}[{index}]")
            if value.log is not None and knot_value <= 0.0:
                raise ValueError(f"{path}[{index}] must be greater than zero for a log schedule")
        return
    validate_ema_decay(value, path)
