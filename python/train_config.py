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


type VariableEndpoint = float | list[float]


class VariableRange(StrictBaseModel):
    min: VariableEndpoint
    max: VariableEndpoint


class VariableSchedule(StrictBaseModel):
    linear: list[float] | VariableRange | None = None
    log: list[float] | VariableRange | None = None


type ScheduleableFloat = float | Schedule
type ScheduleableInt = int | Schedule
type VariableFloat = float | VariableSchedule


class ModelConfig(StrictBaseModel):
    compile: bool
    autocast: bool
    generation_autocast: bool
    embedding_width: int
    global_embedding_width: int
    hidden_width: int
    proposal_hidden_width: int
    missing_connect_query_hidden_width: int
    missing_connect_query_frontier_width: int
    missing_connect_query_distance_width: int
    utility_query_hidden_width: int
    utility_query_frontier_width: int
    known_save_refill_utility_override: bool
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
    gpu_prefetch_batches: int
    recommended_candidates: ScheduleableInt
    shortlist_candidates: ScheduleableInt
    temperature: VariableFloat
    proposal_temperature: VariableFloat
    reward_door: VariableFloat
    reward_connection: VariableFloat
    reward_toilet: VariableFloat
    reward_phantoon: VariableFloat
    reward_balance: VariableFloat
    reward_toilet_balance: VariableFloat
    reward_frontier: VariableFloat
    reward_graph_diameter: VariableFloat
    reward_save_distance: VariableFloat
    reward_refill_distance: VariableFloat
    reward_missing_connect_utility: VariableFloat
    frontier_neighbor_algorithm: Literal["delaunay", "nearest", "nearest-exclusive"]
    frontier_neighbor_count: int
    frontier_window_size: int
    candidate_spatial_cell_size: int
    num_threads: int | None


class FeatureConfig(StrictBaseModel):
    inventory: bool
    temperature: bool
    recommended_candidates: bool
    lookahead_outcomes: int
    room_position: bool
    global_room_position: int
    room_part_furthest_distance: int
    room_part_save_distance: int
    room_part_refill_distance: int
    room_part_frontier_distance: int
    frontier_mask: bool
    frontier_position: int
    frontier_orientation: int
    frontier_kind: int
    frontier_door_variant: int
    frontier_occupancy: bool
    frontier_neighbor: bool
    frontier_neighbor_position_embedding: int
    frontier_neighbor_flags: bool
    connection_reachability: int
    frontier_connection_reachability: bool
    missing_connect_query: bool
    save_utility_query: bool
    refill_utility_query: bool
    toilet_crossed_room: int
    known_distance: int

    def engine_config(self) -> "EngineFeatureConfig":
        return EngineFeatureConfig(
            inventory=self.inventory,
            temperature=self.temperature,
            recommended_candidates=self.recommended_candidates,
            lookahead_outcomes=self.lookahead_outcomes > 0,
            room_position=self.room_position,
            global_room_position=self.global_room_position > 0,
            room_part_furthest_distance=self.room_part_furthest_distance > 0,
            room_part_save_distance=self.room_part_save_distance > 0,
            room_part_refill_distance=self.room_part_refill_distance > 0,
            room_part_frontier_distance=self.room_part_frontier_distance > 0,
            frontier_mask=self.frontier_mask,
            frontier_position=self.frontier_position > 0,
            frontier_orientation=self.frontier_orientation > 0,
            frontier_kind=self.frontier_kind > 0,
            frontier_door_variant=self.frontier_door_variant > 0,
            frontier_occupancy=self.frontier_occupancy,
            frontier_neighbor=self.frontier_neighbor,
            frontier_neighbor_position_embedding=self.frontier_neighbor_position_embedding > 0,
            frontier_neighbor_flags=self.frontier_neighbor_flags,
            connection_reachability=self.connection_reachability > 0,
            frontier_connection_reachability=self.frontier_connection_reachability,
            missing_connect_query=self.missing_connect_query,
            save_utility_query=self.save_utility_query,
            refill_utility_query=self.refill_utility_query,
            toilet_crossed_room=self.toilet_crossed_room > 0,
        )


class EngineFeatureConfig(StrictBaseModel):
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
    frontier_door_variant: bool
    frontier_occupancy: bool
    frontier_neighbor: bool
    frontier_neighbor_position_embedding: bool
    frontier_neighbor_flags: bool
    connection_reachability: bool
    frontier_connection_reachability: bool
    missing_connect_query: bool
    save_utility_query: bool
    refill_utility_query: bool
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
    phantoon_weight: float
    balance_weight: float
    toilet_balance_weight: float
    avg_frontiers_weight: float
    graph_diameter_weight: float
    save_distance_weight: float
    refill_distance_weight: float
    missing_connect_utility_weight: float
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
            elif field_info.annotation is VariableFloat:
                updates[field_name] = instantiate_variable_float(value, field_path)
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

    def instantiate_endpoint(value: VariableEndpoint, path: str, log_scale: bool) -> float:
        if isinstance(value, list):
            if len(value) != len(knot_episodes):
                raise ValueError(
                    f"{path} has {len(value)} schedule value(s), but knot_episodes has {len(knot_episodes)} knot(s)"
                )
            if log_scale:
                for index, item in enumerate(value):
                    if item <= 0.0:
                        raise ValueError(
                            f"{path}[{index}] must be greater than zero for a log schedule"
                        )
                return float(np.exp(np.interp(num_episodes, knot_episodes, np.log(value))))
            return float(np.interp(num_episodes, knot_episodes, value))
        if log_scale and value <= 0.0:
            raise ValueError(f"{path} must be greater than zero for a log schedule")
        return float(value)

    def instantiate_variable_float(value: VariableFloat, path: str) -> VariableFloat:
        if isinstance(value, VariableSchedule):
            if (value.linear is None) == (value.log is None):
                raise ValueError(f"{path} must have exactly one schedule value: 'linear' or 'log'")
            schedule_value = value.linear if value.linear is not None else value.log
            log_scale = value.log is not None
            if isinstance(schedule_value, VariableRange):
                min_value = instantiate_endpoint(
                    schedule_value.min,
                    f"{path}.min",
                    log_scale,
                )
                max_value = instantiate_endpoint(
                    schedule_value.max,
                    f"{path}.max",
                    log_scale,
                )
                if min_value > max_value:
                    raise ValueError(f"{path}.min must be less than or equal to {path}.max")
                range_value = VariableRange(min=min_value, max=max_value)
                if value.linear is not None:
                    return VariableSchedule(linear=range_value)
                return VariableSchedule(log=range_value)
            if len(schedule_value) != len(knot_episodes):
                raise ValueError(
                    f"{path} has {len(schedule_value)} schedule value(s), but knot_episodes has {len(knot_episodes)} knot(s)"
                )
            if value.linear is not None:
                return float(np.interp(num_episodes, knot_episodes, schedule_value))
            for index, item in enumerate(schedule_value):
                if item <= 0.0:
                    raise ValueError(
                        f"{path}[{index}] must be greater than zero for a log schedule"
                    )
            return float(np.exp(np.interp(num_episodes, knot_episodes, np.log(schedule_value))))
        return float(value)

    return instantiate_model(config, "config")


def validate_config(config: Config) -> None:
    def validate_feature_width(name: str, value: int) -> None:
        if value < 0:
            raise ValueError(f"features.{name} must be greater than or equal to zero")

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
    if config.model.proposal_hidden_width <= 0:
        raise ValueError("model.proposal_hidden_width must be greater than zero")
    if config.model.missing_connect_query_hidden_width <= 0:
        raise ValueError("model.missing_connect_query_hidden_width must be greater than zero")
    if config.model.missing_connect_query_frontier_width <= 0:
        raise ValueError("model.missing_connect_query_frontier_width must be greater than zero")
    if config.model.missing_connect_query_distance_width <= 0:
        raise ValueError("model.missing_connect_query_distance_width must be greater than zero")
    if config.model.utility_query_hidden_width <= 0:
        raise ValueError("model.utility_query_hidden_width must be greater than zero")
    if config.model.utility_query_frontier_width <= 0:
        raise ValueError("model.utility_query_frontier_width must be greater than zero")
    validate_feature_width("lookahead_outcomes", config.features.lookahead_outcomes)
    validate_feature_width("global_room_position", config.features.global_room_position)
    validate_feature_width(
        "room_part_furthest_distance",
        config.features.room_part_furthest_distance,
    )
    validate_feature_width("room_part_save_distance", config.features.room_part_save_distance)
    validate_feature_width("room_part_refill_distance", config.features.room_part_refill_distance)
    validate_feature_width(
        "room_part_frontier_distance",
        config.features.room_part_frontier_distance,
    )
    validate_feature_width("connection_reachability", config.features.connection_reachability)
    validate_feature_width("frontier_position", config.features.frontier_position)
    validate_feature_width("frontier_orientation", config.features.frontier_orientation)
    validate_feature_width("frontier_kind", config.features.frontier_kind)
    validate_feature_width("frontier_door_variant", config.features.frontier_door_variant)
    validate_feature_width(
        "frontier_neighbor_position_embedding",
        config.features.frontier_neighbor_position_embedding,
    )
    validate_feature_width("toilet_crossed_room", config.features.toilet_crossed_room)
    validate_feature_width("known_distance", config.features.known_distance)
    validate_optimizer_config(config.optimizer, "optimizer")
    validate_optimizer_config(config.balance_optimizer, "balance_optimizer")
    if config.generation.num_iterations <= 0:
        raise ValueError("generation.num_iterations must be greater than zero")
    if config.generation.num_devices <= 0:
        raise ValueError("generation.num_devices must be greater than zero")
    if config.generation.pipeline_groups <= 0:
        raise ValueError("generation.pipeline_groups must be greater than zero")
    if config.generation.gpu_prefetch_batches < 0:
        raise ValueError("generation.gpu_prefetch_batches must be greater than or equal to zero")
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
    if config.generation.candidate_spatial_cell_size <= 0:
        raise ValueError("generation.candidate_spatial_cell_size must be greater than zero")
    validate_positive_variable_float(
        config.generation.temperature,
        "generation.temperature",
    )
    validate_positive_variable_float(
        config.generation.proposal_temperature,
        "generation.proposal_temperature",
    )
    validate_nonnegative_variable_float(
        config.generation.reward_phantoon,
        "generation.reward_phantoon",
    )
    validate_nonnegative_variable_float(
        config.generation.reward_toilet_balance,
        "generation.reward_toilet_balance",
    )
    validate_nonnegative_variable_float(
        config.generation.reward_graph_diameter,
        "generation.reward_graph_diameter",
    )
    validate_nonnegative_variable_float(
        config.generation.reward_save_distance,
        "generation.reward_save_distance",
    )
    validate_nonnegative_variable_float(
        config.generation.reward_refill_distance,
        "generation.reward_refill_distance",
    )
    validate_nonnegative_variable_float(
        config.generation.reward_missing_connect_utility,
        "generation.reward_missing_connect_utility",
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
    if config.train.phantoon_weight < 0:
        raise ValueError("train.phantoon_weight must be greater than or equal to zero")
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
    if config.train.missing_connect_utility_weight < 0:
        raise ValueError(
            "train.missing_connect_utility_weight must be greater than or equal to zero"
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
        or config.features.frontier_door_variant
        or config.features.frontier_occupancy
        or config.features.frontier_neighbor
        or config.features.frontier_connection_reachability
        or config.features.missing_connect_query
        or config.features.save_utility_query
        or config.features.refill_utility_query
    ) and not config.features.frontier_mask:
        raise ValueError("frontier query and frontier features require features.frontier_mask")
    if (
        config.features.inventory or config.features.connection_reachability
    ) and not config.features.frontier_mask:
        raise ValueError("start-of-network features require features.frontier_mask")
    if (
        config.features.frontier_neighbor_position_embedding > 0
        or config.features.frontier_neighbor_flags
    ) and not config.features.frontier_neighbor:
        raise ValueError("frontier neighbor pair features require features.frontier_neighbor")
    if config.features.global_room_position > 0 and not config.features.room_position:
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


def validate_nonnegative_variable_float(value: VariableFloat, path: str) -> None:
    def validate_endpoint(endpoint: VariableEndpoint, endpoint_path: str) -> None:
        if isinstance(endpoint, list):
            for index, item in enumerate(endpoint):
                if item < 0:
                    raise ValueError(
                        f"{endpoint_path}[{index}] must be greater than or equal to zero"
                    )
            return
        if endpoint < 0:
            raise ValueError(f"{endpoint_path} must be greater than or equal to zero")

    if isinstance(value, VariableSchedule):
        values = value.linear if value.linear is not None else value.log
        if values is None:
            return
        if isinstance(values, VariableRange):
            validate_endpoint(values.min, f"{path}.min")
            validate_endpoint(values.max, f"{path}.max")
            return
        for index, item in enumerate(values):
            if item < 0:
                raise ValueError(f"{path}[{index}] must be greater than or equal to zero")
        return
    if value < 0:
        raise ValueError(f"{path} must be greater than or equal to zero")


def validate_positive_variable_float(value: VariableFloat, path: str) -> None:
    def validate_endpoint(endpoint: VariableEndpoint, endpoint_path: str) -> None:
        if isinstance(endpoint, list):
            for index, item in enumerate(endpoint):
                if item <= 0.0:
                    raise ValueError(f"{endpoint_path}[{index}] must be greater than zero")
            return
        if endpoint <= 0.0:
            raise ValueError(f"{endpoint_path} must be greater than zero")

    if isinstance(value, VariableSchedule):
        values = value.linear if value.linear is not None else value.log
        if values is None:
            return
        if isinstance(values, VariableRange):
            validate_endpoint(values.min, f"{path}.min")
            validate_endpoint(values.max, f"{path}.max")
            return
        for index, item in enumerate(values):
            if item <= 0.0:
                raise ValueError(f"{path}[{index}] must be greater than zero")
        return
    if value <= 0.0:
        raise ValueError(f"{path} must be greater than zero")


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
