from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Schedule(StrictBaseModel):
    linear: list[float] | None = None
    log: list[float] | None = None


type ScheduleableFloat = float | Schedule


class ModelConfig(StrictBaseModel):
    type: str
    compile: bool
    autocast: bool
    generation_autocast: bool
    embedding_width: int
    key_width: int
    value_width: int
    attn_heads: int
    head_groups: int
    hidden_width: int
    num_layers: int


class OptimizerConfig(StrictBaseModel):
    lr: ScheduleableFloat
    beta1: float
    beta2: float


class GenerationConfig(StrictBaseModel):
    num_environments: int
    num_iterations: int
    num_devices: int
    state_pipeline_cohorts: int
    action_candidates: int
    lookahead_outcomes: bool
    temperature: ScheduleableFloat
    state_candidate_chunk: int
    state_environment_chunk: int
    frontier_neighbor_algorithm: Literal["delaunay", "nearest", "nearest-exclusive"]
    frontier_neighbor_count: int
    frontier_window_size: int
    num_threads: int | None


class StateFeatureConfig(StrictBaseModel):
    inventory: bool
    room_position: bool
    frontier_mask: bool
    frontier_position: bool
    frontier_orientation: bool
    frontier_kind: bool
    frontier_occupancy: bool
    frontier_neighbor: bool
    frontier_neighbor_position: bool
    frontier_neighbor_flags: bool
    connection_reachability: bool
    frontier_connection_reachability: bool


class TrainConfig(StrictBaseModel):
    batch_size: int
    fresh_pass_factor: float
    replay_pass_factor: float
    sample_period: int
    episodes_per_file: int
    hist_c: float
    door_weight: float
    connection_weight: float
    ema_decay: float
    state_pipeline_cohorts: int
    gradient_accumulation_steps: int


class Config(StrictBaseModel):
    experiment_name: str
    room_set: Path
    map_size: tuple[int, int]
    knot_episodes: list[int]
    model: ModelConfig
    optimizer: OptimizerConfig
    generation: GenerationConfig
    state_features: StateFeatureConfig
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
            elif isinstance(value, BaseModel):
                updates[field_name] = instantiate_model(value, field_path)
        return model.model_copy(update=updates)

    def instantiate_float(value: ScheduleableFloat, path: str) -> float:
        if isinstance(value, Schedule):
            if (value.linear is None) == (value.log is None):
                raise ValueError(f"{path} must have exactly one schedule value: 'linear' or 'log'")
            x = value.linear if value.linear is not None else value.log
            if len(x) != len(knot_episodes):
                raise ValueError(
                    f"{path} has {len(x)} schedule value(s), but knot_episodes has "
                    f"{len(knot_episodes)} knot(s)"
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
    if config.generation.num_iterations <= 0:
        raise ValueError("generation.num_iterations must be greater than zero")
    if config.generation.num_devices <= 0:
        raise ValueError("generation.num_devices must be greater than zero")
    if config.generation.state_pipeline_cohorts <= 0:
        raise ValueError("generation.state_pipeline_cohorts must be greater than zero")
    if config.generation.num_devices > config.generation.num_environments:
        raise ValueError("generation.num_devices must not exceed generation.num_environments")
    num_generation_cohorts = (
        config.generation.num_devices * config.generation.state_pipeline_cohorts
    )
    if config.generation.num_environments % num_generation_cohorts != 0:
        raise ValueError(
            "generation.num_environments must be divisible by "
            "generation.num_devices * generation.state_pipeline_cohorts"
        )
    if config.generation.state_candidate_chunk <= 0:
        raise ValueError("generation.state_candidate_chunk must be greater than zero")
    if config.generation.state_environment_chunk <= 0:
        raise ValueError("generation.state_environment_chunk must be greater than zero")
    if config.generation.frontier_neighbor_count < 0:
        raise ValueError("generation.frontier_neighbor_count must be greater than or equal to zero")
    if config.generation.frontier_window_size <= 0:
        raise ValueError("generation.frontier_window_size must be greater than zero")
    if config.generation.num_threads is not None and config.generation.num_threads <= 0:
        raise ValueError("generation.num_threads must be greater than zero")
    if (
        config.generation.num_threads is not None
        and config.generation.num_threads % config.generation.state_pipeline_cohorts != 0
    ):
        raise ValueError("generation.num_threads must be divisible by generation.state_pipeline_cohorts")
    if config.model.type != "frontier_state" and config.generation.state_pipeline_cohorts != 1:
        raise ValueError("generation.state_pipeline_cohorts must be 1 unless model.type is frontier_state")
    if config.train.sample_period <= 0:
        raise ValueError("train.sample_period must be greater than zero")
    if config.train.state_pipeline_cohorts <= 0:
        raise ValueError("train.state_pipeline_cohorts must be greater than zero")
    if config.train.gradient_accumulation_steps <= 0:
        raise ValueError("train.gradient_accumulation_steps must be greater than zero")
    if (
        config.generation.num_threads is not None
        and config.generation.num_threads % config.train.state_pipeline_cohorts != 0
    ):
        raise ValueError("generation.num_threads must be divisible by train.state_pipeline_cohorts")
    if (
        config.state_features.frontier_position
        or config.state_features.frontier_orientation
        or config.state_features.frontier_kind
        or config.state_features.frontier_occupancy
        or config.state_features.frontier_neighbor
        or config.state_features.frontier_connection_reachability
    ) and not config.state_features.frontier_mask:
        raise ValueError("frontier state features require state_features.frontier_mask")
    if (
        config.state_features.frontier_neighbor_position
        or config.state_features.frontier_neighbor_flags
    ) and not config.state_features.frontier_neighbor:
        raise ValueError("frontier neighbor pair features require state_features.frontier_neighbor")


def episodes_per_round(config: Config) -> int:
    value = config.generation.num_iterations * config.generation.num_environments
    if config.train.fresh_pass_factor != 0.0 and value % config.train.batch_size != 0:
        raise ValueError(
            "train.batch_size must evenly divide the number of episodes generated per round when "
            "train.fresh_pass_factor is non-zero"
        )
    return value
