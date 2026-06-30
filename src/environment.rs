use bitvec::vec::BitVec;
use delaunator::{EMPTY, Point, next_halfedge, triangulate};
use hashbrown::HashMap;
use rand::SeedableRng;
use rand::prelude::*;
use serde::Deserialize;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::{Duration, Instant};

use crate::common::{
    Action, ActionIdx, CommonData, ConnectionVariantIdx, Coord, DirDoorIdx, Direction, DoorKind,
    DoorLocation, DoorValidOutcome, DoorVariantIdx, FrontierIdx, GeometryData, GeometryIdx,
    GraphDistance, NUM_DIRS, PartIdx, RoomIdx, RoomPartIdx, SpatialCellIdx,
    get_behind_door_position,
};
use crate::engine::{ProfileMetric, profile_enabled, record_profile_count, record_profile_metric};
use crate::scc_dag::SccDag;

const NO_COMPONENT: usize = usize::MAX;
const UNREACHABLE_DISTANCE: GraphDistance = GraphDistance::MAX;
const KNOWN_DISTANCE_UNKNOWN: u8 = 0;
const KNOWN_DISTANCE_UNREACHABLE: u8 = 1;
pub const FEATURE_FRONTIER_WIDTH: usize = 5;

fn encode_known_finalized_distance(
    current_distance: GraphDistance,
    frontier_distance: GraphDistance,
) -> u8 {
    if current_distance != UNREACHABLE_DISTANCE && current_distance <= frontier_distance {
        current_distance.min(253) + 2
    } else if current_distance == UNREACHABLE_DISTANCE && frontier_distance == UNREACHABLE_DISTANCE
    {
        KNOWN_DISTANCE_UNREACHABLE
    } else {
        KNOWN_DISTANCE_UNKNOWN
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StepMode {
    CommitFull,
    CommitKnown,
    Lookahead,
    FeatureOnly,
}

impl StepMode {
    fn records_action(self) -> bool {
        match self {
            StepMode::CommitFull => true,
            StepMode::CommitKnown => true,
            StepMode::Lookahead => true,
            StepMode::FeatureOnly => false,
        }
    }

    fn updates_geometry_inventory(self) -> bool {
        match self {
            StepMode::CommitFull => true,
            StepMode::CommitKnown => true,
            StepMode::Lookahead => true,
            StepMode::FeatureOnly => false,
        }
    }

    fn updates_connection_variant_inventory(self) -> bool {
        match self {
            StepMode::CommitFull => true,
            StepMode::CommitKnown => true,
            StepMode::Lookahead => true,
            StepMode::FeatureOnly => true,
        }
    }

    fn updates_occupancy(self) -> bool {
        match self {
            StepMode::CommitFull => true,
            StepMode::CommitKnown => true,
            StepMode::Lookahead => false,
            StepMode::FeatureOnly => false,
        }
    }

    fn updates_door_matches(self) -> bool {
        match self {
            StepMode::CommitFull => true,
            StepMode::CommitKnown => true,
            StepMode::Lookahead => true,
            StepMode::FeatureOnly => false,
        }
    }

    fn builds_frontier_candidates(self) -> bool {
        match self {
            StepMode::CommitFull => true,
            StepMode::CommitKnown => false,
            StepMode::Lookahead => true,
            StepMode::FeatureOnly => false,
        }
    }

    fn stores_full_candidate_lists(self) -> bool {
        match self {
            StepMode::CommitFull => true,
            StepMode::CommitKnown => false,
            StepMode::Lookahead => false,
            StepMode::FeatureOnly => false,
        }
    }

    fn filters_existing_frontier_candidates(self) -> bool {
        match self {
            StepMode::CommitFull => true,
            StepMode::CommitKnown => false,
            StepMode::Lookahead => true,
            StepMode::FeatureOnly => false,
        }
    }
}

fn profile_start() -> Option<Instant> {
    profile_enabled().then(Instant::now)
}

fn profile_end(metric: ProfileMetric, start: Option<Instant>) {
    if let Some(start) = start {
        record_profile_metric(metric, start.elapsed());
    }
}

fn graph_distance_sum(distances: &[GraphDistance]) -> Option<GraphDistance> {
    let mut total: GraphDistance = 0;
    for &distance in distances {
        if distance == UNREACHABLE_DISTANCE {
            return None;
        }
        total = total.checked_add(distance)?;
        if total == UNREACHABLE_DISTANCE {
            return None;
        }
    }
    Some(total)
}

fn check_outcome_transition_consistency(
    before: &[DoorValidOutcome],
    after: &[DoorValidOutcome],
    outcome_name: &str,
    stage: &str,
) -> Result<(), String> {
    debug_assert_eq!(before.len(), after.len());
    for (idx, (&before, &after)) in before.iter().zip(after).enumerate() {
        if before != DoorValidOutcome::Unknown && before != after {
            return Err(format!(
                "{outcome_name} outcome changed after becoming known at {stage}: \
                 index {idx}, before {before:?}, after {after:?}"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
fn introduces_invalid_outcome(before: &StepOutcomes, after: &StepOutcomes) -> bool {
    before
        .door_valid
        .iter()
        .zip(&after.door_valid)
        .any(|(&before, &after)| {
            before == DoorValidOutcome::Unknown && after == DoorValidOutcome::Invalid
        })
        || before
            .connections_valid
            .iter()
            .zip(&after.connections_valid)
            .any(|(&before, &after)| {
                before == DoorValidOutcome::Unknown && after == DoorValidOutcome::Invalid
            })
        || (before.toilet_valid == DoorValidOutcome::Unknown
            && after.toilet_valid == DoorValidOutcome::Invalid)
        || (before.phantoon_valid == DoorValidOutcome::Unknown
            && after.phantoon_valid == DoorValidOutcome::Invalid)
}

enum CandidateOutcome {
    Clean(StepOutcomes, Vec<i16>, FeaturePlan),
    Rejected,
}

#[derive(Clone, Copy, Debug)]
struct FrontierEdge {
    endpoints: [usize; 2],
    length_squared: i32,
    active: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrontierNeighborAlgorithm {
    Delaunay,
    Nearest,
    NearestExclusive,
}

// Frontier: location of an unconnected door on the map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frontier {
    direction: Direction,
    dir_door_idx: DirDoorIdx,
    door_output_idx: i16,
    door_variant_idx: DoorVariantIdx,
    room_part_idx: RoomPartIdx,
    component: usize,
    kind: DoorKind,
    candidates: Vec<GeometryAction>, // possible geometry placements to connect to this frontier
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct GeometryAction {
    geometry_idx: GeometryIdx,
    x: Coord,
    y: Coord,
    door_direction: Direction,
    door_x: Coord,
    door_y: Coord,
    door_kind: DoorKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CandidateAction {
    pub action: Action,
    pub frontier_idx: FrontierIdx,
    pub door_variant_idx: DoorVariantIdx,
}

#[derive(Clone)]
pub struct StepOutcomes {
    // For each door, whether it is connected to another door.
    pub door_valid: Vec<DoorValidOutcome>,
    // For each connection, whether its destination can reach its source.
    pub connections_valid: Vec<DoorValidOutcome>,
    // Whether the Toilet crosses exactly one room.
    pub toilet_valid: DoorValidOutcome,
    // Whether Phantoon's Room and Wrecked Ship Map Room connect to the same room.
    pub phantoon_valid: DoorValidOutcome,
    // Concrete room crossed by the Toilet when exactly one non-Toilet room crosses it.
    pub toilet_crossed_room_idx: i16,
}

#[derive(Clone)]
pub struct FeatureOutcomes {
    pub step_outcomes: StepOutcomes,
    pub door_match: Vec<i16>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureConfig {
    pub inventory: bool,
    pub temperature: bool,
    pub recommended_candidates: bool,
    // This is attached by Python from outcome tensors; Rust only needs to accept
    // the strict feature config field.
    #[allow(dead_code)]
    pub lookahead_outcomes: bool,
    pub room_position: bool,
    pub global_room_position: bool,
    pub room_part_furthest_distance: bool,
    pub room_part_save_distance: bool,
    pub room_part_refill_distance: bool,
    pub room_part_frontier_distance: bool,
    pub frontier_mask: bool,
    pub frontier_position: bool,
    pub frontier_orientation: bool,
    pub frontier_kind: bool,
    pub frontier_door_variant: bool,
    pub frontier_occupancy: bool,
    pub frontier_neighbor: bool,
    pub frontier_neighbor_position_embedding: bool,
    pub frontier_neighbor_flags: bool,
    pub connection_reachability: bool,
    pub frontier_connection_reachability: bool,
    pub missing_connect_query: bool,
    pub save_utility_query: bool,
    pub refill_utility_query: bool,
    pub toilet_crossed_room: bool,
}

impl FeatureConfig {
    pub fn is_empty(&self) -> bool {
        !self.inventory
            && !self.temperature
            && !self.recommended_candidates
            && !self.room_position
            && !self.global_room_position
            && !self.room_part_furthest_distance
            && !self.room_part_save_distance
            && !self.room_part_refill_distance
            && !self.room_part_frontier_distance
            && !self.connection_reachability
            && !self.toilet_crossed_room
            && !self.missing_connect_query
            && !self.save_utility_query
            && !self.refill_utility_query
            && !self.has_frontier_features()
    }

    pub fn has_frontier_features(&self) -> bool {
        self.frontier_mask
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if (self.frontier_position
            || self.frontier_orientation
            || self.frontier_kind
            || self.frontier_door_variant
            || self.frontier_occupancy
            || self.frontier_neighbor
            || self.frontier_connection_reachability
            || self.missing_connect_query
            || self.save_utility_query
            || self.refill_utility_query)
            && !self.frontier_mask
        {
            return Err("frontier query and frontier features require frontier_mask");
        }
        if (self.frontier_neighbor_position_embedding || self.frontier_neighbor_flags)
            && !self.frontier_neighbor
        {
            return Err("frontier neighbor pair features require frontier_neighbor");
        }
        if self.global_room_position && !self.room_position {
            return Err("global_room_position requires room_position");
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn all() -> Self {
        Self {
            inventory: true,
            temperature: true,
            recommended_candidates: true,
            lookahead_outcomes: true,
            room_position: true,
            global_room_position: true,
            room_part_furthest_distance: true,
            room_part_save_distance: true,
            room_part_refill_distance: true,
            room_part_frontier_distance: true,
            frontier_mask: true,
            frontier_position: true,
            frontier_orientation: true,
            frontier_kind: true,
            frontier_door_variant: true,
            frontier_occupancy: true,
            frontier_neighbor: true,
            frontier_neighbor_position_embedding: true,
            frontier_neighbor_flags: true,
            connection_reachability: true,
            frontier_connection_reachability: true,
            missing_connect_query: true,
            save_utility_query: true,
            refill_utility_query: true,
            toilet_crossed_room: true,
        }
    }

    #[cfg(test)]
    pub fn all_disabled() -> Self {
        Self {
            inventory: false,
            temperature: false,
            recommended_candidates: false,
            lookahead_outcomes: false,
            room_position: false,
            global_room_position: false,
            room_part_furthest_distance: false,
            room_part_save_distance: false,
            room_part_refill_distance: false,
            room_part_frontier_distance: false,
            frontier_mask: false,
            frontier_position: false,
            frontier_orientation: false,
            frontier_kind: false,
            frontier_door_variant: false,
            frontier_occupancy: false,
            frontier_neighbor: false,
            frontier_neighbor_position_embedding: false,
            frontier_neighbor_flags: false,
            connection_reachability: false,
            frontier_connection_reachability: false,
            missing_connect_query: false,
            save_utility_query: false,
            refill_utility_query: false,
            toilet_crossed_room: false,
        }
    }
}

#[cfg(test)]
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Features {
    pub inventory: Vec<u8>,
    pub room_x: Vec<Coord>,
    pub room_y: Vec<Coord>,
    pub room_placed: Vec<u8>,
    pub room_part_furthest_destination: Vec<u8>,
    pub room_part_furthest_source: Vec<u8>,
    pub room_part_save_from_room_distance: Vec<u8>,
    pub room_part_save_to_room_distance: Vec<u8>,
    pub room_part_refill_from_room_distance: Vec<u8>,
    pub room_part_refill_to_room_distance: Vec<u8>,
    pub room_part_frontier_from_room_distance: Vec<u8>,
    pub room_part_frontier_to_room_distance: Vec<u8>,
    pub known_save_from_room_distance: Vec<u8>,
    pub known_save_to_room_distance: Vec<u8>,
    pub known_refill_from_room_distance: Vec<u8>,
    pub known_refill_to_room_distance: Vec<u8>,
    // mask, x, y, vertical, kind
    pub frontier: Vec<i8>,
    // Door variant index for each frontier row, keyed by the unmatched door variant.
    pub frontier_door_variant: Vec<DoorVariantIdx>,
    // Global door output index for each frontier row, or -1 when unavailable.
    pub row_door_output_idx: Vec<i16>,
    // Occupied tiles in a square window centered on each frontier, packed row-major.
    pub frontier_occupancy: Vec<u8>,
    // Indices into frontier. Semantics depend on FrontierNeighborAlgorithm.
    // -1 marks padding.
    pub frontier_neighbor: Vec<i16>,
    // Bit flags: same SCC, source reaches destination, destination reaches source.
    pub frontier_neighbor_pair: Vec<u8>,
    // Whether each required closure edge already has an interior path.
    pub connection_reachability: Vec<u8>,
    // Bit flags per frontier and required closure edge: source reaches frontier,
    // frontier reaches destination.
    pub frontier_connection_reachability: Vec<u8>,
    // Sparse missing-connect frontier query rows. Frontier indices are snapshot-local.
    pub missing_connect_query_connection_idx: Vec<i64>,
    pub missing_connect_query_source_frontier: Vec<i16>,
    pub missing_connect_query_target_frontier: Vec<i16>,
    pub missing_connect_query_source_distance: Vec<u8>,
    pub missing_connect_query_target_distance: Vec<u8>,
    pub missing_connect_query_current_distance: Vec<u8>,
    // Sparse utility query rows for save/refill distances. Frontier indices are
    // snapshot-local; -1 marks padding.
    pub save_refill_utility_query_room_part_idx: Vec<i64>,
    pub save_refill_utility_query_target_mask: Vec<u8>,
    pub save_refill_utility_query_frontier: Vec<i16>,
    pub save_refill_utility_query_frontier_distance: Vec<u8>,
    pub save_refill_utility_query_save_to_current_distance: Vec<u8>,
    pub save_refill_utility_query_save_from_current_distance: Vec<u8>,
    pub save_refill_utility_query_refill_to_current_distance: Vec<u8>,
    pub save_refill_utility_query_refill_from_current_distance: Vec<u8>,
    // Concrete room crossed by the Toilet when exactly one non-Toilet room crosses it.
    pub toilet_crossed_room_idx: Vec<i16>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeaturePlanKind {
    Current,
    Candidate(Action),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeatureExtraOccupied {
    pub geometry_idx: GeometryIdx,
    pub x: Coord,
    pub y: Coord,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeatureFrontierPlanRow {
    pub location: DoorLocation,
    pub door_variant_idx: DoorVariantIdx,
    pub row_door_output_idx: i16,
    pub component: usize,
    pub kind: DoorKind,
    pub room_part_idx: RoomPartIdx,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FeaturePlan {
    pub environment_idx: usize,
    pub kind: FeaturePlanKind,
    pub extra_occupied: Option<FeatureExtraOccupied>,
    pub scc_dag: SccDag,
    pub room_part_furthest_destination: Vec<u8>,
    pub room_part_furthest_source: Vec<u8>,
    pub room_part_save_from_room_distance: Vec<u8>,
    pub room_part_save_to_room_distance: Vec<u8>,
    pub room_part_refill_from_room_distance: Vec<u8>,
    pub room_part_refill_to_room_distance: Vec<u8>,
    pub room_part_frontier_from_room_distance: Vec<u8>,
    pub room_part_frontier_to_room_distance: Vec<u8>,
    pub known_save_from_room_distance: Vec<u8>,
    pub known_save_to_room_distance: Vec<u8>,
    pub known_refill_from_room_distance: Vec<u8>,
    pub known_refill_to_room_distance: Vec<u8>,
    pub frontiers: Vec<FeatureFrontierPlanRow>,
    pub connection_reachability: Vec<u8>,
    pub frontier_connection_reachability: Vec<u8>,
    pub missing_connect_query_connection_idx: Vec<i64>,
    pub missing_connect_query_source_frontier: Vec<i16>,
    pub missing_connect_query_target_frontier: Vec<i16>,
    pub missing_connect_query_source_distance: Vec<u8>,
    pub missing_connect_query_target_distance: Vec<u8>,
    pub missing_connect_query_current_distance: Vec<u8>,
    pub save_refill_utility_query_room_part_idx: Vec<i64>,
    pub save_refill_utility_query_target_mask: Vec<u8>,
    pub save_refill_utility_query_frontier: Vec<i16>,
    pub save_refill_utility_query_frontier_distance: Vec<u8>,
    pub save_refill_utility_query_save_to_current_distance: Vec<u8>,
    pub save_refill_utility_query_save_from_current_distance: Vec<u8>,
    pub save_refill_utility_query_refill_to_current_distance: Vec<u8>,
    pub save_refill_utility_query_refill_from_current_distance: Vec<u8>,
    pub toilet_crossed_room_idx: Vec<i16>,
}

impl Default for FeaturePlanKind {
    fn default() -> Self {
        Self::Current
    }
}

impl FeaturePlan {
    pub fn frontier_row_count(&self) -> usize {
        self.frontiers.len()
    }

    pub fn missing_connect_query_row_count(&self) -> usize {
        self.missing_connect_query_connection_idx.len()
    }

    pub fn save_refill_utility_query_row_count(&self) -> usize {
        self.save_refill_utility_query_room_part_idx.len()
    }

    fn push_save_refill_utility_query_row(
        &mut self,
        room_part: RoomPartIdx,
        row: SaveRefillUtilityQueryRow,
    ) {
        self.save_refill_utility_query_room_part_idx
            .push(i64::from(room_part));
        self.save_refill_utility_query_target_mask
            .push(row.target_mask);
        self.save_refill_utility_query_frontier.push(row.frontier);
        self.save_refill_utility_query_frontier_distance
            .push(row.frontier_distance);
        self.save_refill_utility_query_save_to_current_distance
            .push(row.save_to_current_distance);
        self.save_refill_utility_query_save_from_current_distance
            .push(row.save_from_current_distance);
        self.save_refill_utility_query_refill_to_current_distance
            .push(row.refill_to_current_distance);
        self.save_refill_utility_query_refill_from_current_distance
            .push(row.refill_from_current_distance);
    }

    fn clear_all(&mut self) {
        self.environment_idx = 0;
        self.kind = FeaturePlanKind::Current;
        self.extra_occupied = None;
        self.scc_dag.clear();
        self.room_part_furthest_destination.clear();
        self.room_part_furthest_source.clear();
        self.room_part_save_from_room_distance.clear();
        self.room_part_save_to_room_distance.clear();
        self.room_part_refill_from_room_distance.clear();
        self.room_part_refill_to_room_distance.clear();
        self.room_part_frontier_from_room_distance.clear();
        self.room_part_frontier_to_room_distance.clear();
        self.known_save_from_room_distance.clear();
        self.known_save_to_room_distance.clear();
        self.known_refill_from_room_distance.clear();
        self.known_refill_to_room_distance.clear();
        self.frontiers.clear();
        self.connection_reachability.clear();
        self.frontier_connection_reachability.clear();
        self.missing_connect_query_connection_idx.clear();
        self.missing_connect_query_source_frontier.clear();
        self.missing_connect_query_target_frontier.clear();
        self.missing_connect_query_source_distance.clear();
        self.missing_connect_query_target_distance.clear();
        self.missing_connect_query_current_distance.clear();
        self.save_refill_utility_query_room_part_idx.clear();
        self.save_refill_utility_query_target_mask.clear();
        self.save_refill_utility_query_frontier.clear();
        self.save_refill_utility_query_frontier_distance.clear();
        self.save_refill_utility_query_save_to_current_distance
            .clear();
        self.save_refill_utility_query_save_from_current_distance
            .clear();
        self.save_refill_utility_query_refill_to_current_distance
            .clear();
        self.save_refill_utility_query_refill_from_current_distance
            .clear();
        self.toilet_crossed_room_idx.clear();
    }
}

#[cfg(test)]
impl Features {
    #[cfg(test)]
    fn clear_all(&mut self) {
        self.inventory.clear();
        self.room_x.clear();
        self.room_y.clear();
        self.room_placed.clear();
        self.room_part_furthest_destination.clear();
        self.room_part_furthest_source.clear();
        self.room_part_save_from_room_distance.clear();
        self.room_part_save_to_room_distance.clear();
        self.room_part_refill_from_room_distance.clear();
        self.room_part_refill_to_room_distance.clear();
        self.room_part_frontier_from_room_distance.clear();
        self.room_part_frontier_to_room_distance.clear();
        self.known_save_from_room_distance.clear();
        self.known_save_to_room_distance.clear();
        self.known_refill_from_room_distance.clear();
        self.known_refill_to_room_distance.clear();
        self.frontier.clear();
        self.frontier_door_variant.clear();
        self.row_door_output_idx.clear();
        self.frontier_occupancy.clear();
        self.frontier_neighbor.clear();
        self.frontier_neighbor_pair.clear();
        self.connection_reachability.clear();
        self.frontier_connection_reachability.clear();
        self.missing_connect_query_connection_idx.clear();
        self.missing_connect_query_source_frontier.clear();
        self.missing_connect_query_target_frontier.clear();
        self.missing_connect_query_source_distance.clear();
        self.missing_connect_query_target_distance.clear();
        self.missing_connect_query_current_distance.clear();
        self.save_refill_utility_query_room_part_idx.clear();
        self.save_refill_utility_query_target_mask.clear();
        self.save_refill_utility_query_frontier.clear();
        self.save_refill_utility_query_frontier_distance.clear();
        self.save_refill_utility_query_save_to_current_distance
            .clear();
        self.save_refill_utility_query_save_from_current_distance
            .clear();
        self.save_refill_utility_query_refill_to_current_distance
            .clear();
        self.save_refill_utility_query_refill_from_current_distance
            .clear();
        self.toilet_crossed_room_idx.clear();
    }
}

#[derive(Default)]
pub struct FeatureScratch {
    #[cfg(test)]
    feature_pool: Vec<Features>,
    plan_pool: Vec<FeaturePlan>,
    frontier_locations: Vec<DoorLocation>,
    nearest_neighbor_indices: Vec<usize>,
    nearest_neighbor_keys: Vec<(Coord, usize, usize)>,
    delaunay_midpoints: Vec<(i16, i16)>,
    delaunay_points: Vec<Point>,
    delaunay_edges: Vec<FrontierEdge>,
    delaunay_incident_edges: Vec<Vec<usize>>,
    delaunay_degrees: Vec<usize>,
    delaunay_output_counts: Vec<usize>,
}

impl FeatureScratch {
    #[cfg(test)]
    fn take_features(&mut self) -> Features {
        let mut features = self.feature_pool.pop().unwrap_or_default();
        features.clear_all();
        features
    }

    fn take_plan(&mut self) -> FeaturePlan {
        let mut plan = self.plan_pool.pop().unwrap_or_default();
        plan.clear_all();
        plan
    }

    pub fn recycle_plan(&mut self, mut plan: FeaturePlan) {
        plan.clear_all();
        self.plan_pool.push(plan);
    }

    pub fn recycle_plan_vec(&mut self, plans: &mut Vec<FeaturePlan>) {
        for plan in plans.drain(..) {
            self.recycle_plan(plan);
        }
    }

    pub fn frontier_locations(&mut self) -> &mut Vec<DoorLocation> {
        &mut self.frontier_locations
    }
}

fn save_refill_utility_distance_can_improve(
    distance: GraphDistance,
    current_distance: GraphDistance,
) -> bool {
    current_distance == UNREACHABLE_DISTANCE
        || u16::from(distance) + 1 < u16::from(current_distance)
}

fn encode_room_part_distance_feature(distance: GraphDistance) -> u8 {
    if distance == UNREACHABLE_DISTANCE {
        0
    } else {
        distance.saturating_add(1)
    }
}

#[derive(Clone, Copy)]
struct SaveRefillUtilityQueryRow {
    target_mask: u8,
    frontier: i16,
    frontier_distance: GraphDistance,
    save_to_current_distance: GraphDistance,
    save_from_current_distance: GraphDistance,
    refill_to_current_distance: GraphDistance,
    refill_from_current_distance: GraphDistance,
}

impl SaveRefillUtilityQueryRow {
    fn new(frontier: i16, frontier_distance: GraphDistance) -> Self {
        Self {
            target_mask: 0,
            frontier,
            frontier_distance,
            save_to_current_distance: UNREACHABLE_DISTANCE,
            save_from_current_distance: UNREACHABLE_DISTANCE,
            refill_to_current_distance: UNREACHABLE_DISTANCE,
            refill_from_current_distance: UNREACHABLE_DISTANCE,
        }
    }

    fn add_target(&mut self, target_bit: u8, current_distance: GraphDistance) {
        self.target_mask |= 1u8 << target_bit;
        match target_bit {
            0 => self.save_to_current_distance = current_distance,
            1 => self.save_from_current_distance = current_distance,
            2 => self.refill_to_current_distance = current_distance,
            3 => self.refill_from_current_distance = current_distance,
            _ => unreachable!(),
        }
    }

    fn same_context(&self, other: &Self) -> bool {
        self.frontier == other.frontier && self.frontier_distance == other.frontier_distance
    }

    fn merge(&mut self, other: Self) {
        self.target_mask |= other.target_mask;
        if other.target_mask & 1 != 0 {
            self.save_to_current_distance = other.save_to_current_distance;
        }
        if other.target_mask & 2 != 0 {
            self.save_from_current_distance = other.save_from_current_distance;
        }
        if other.target_mask & 4 != 0 {
            self.refill_to_current_distance = other.refill_to_current_distance;
        }
        if other.target_mask & 8 != 0 {
            self.refill_from_current_distance = other.refill_from_current_distance;
        }
    }
}

fn frontier_midpoint(location: DoorLocation) -> (i16, i16) {
    if location.vertical() {
        (i16::from(location.x()) * 2 + 1, i16::from(location.y()) * 2)
    } else {
        (i16::from(location.x()) * 2, i16::from(location.y()) * 2 + 1)
    }
}

fn write_frontier_delaunay_neighbors(
    locations: &[DoorLocation],
    max_degree: usize,
    output: &mut [i16],
    scratch: &mut FeatureScratch,
) {
    debug_assert_eq!(output.len(), locations.len() * max_degree);
    output.fill(-1);

    scratch.delaunay_midpoints.clear();
    scratch
        .delaunay_midpoints
        .extend(locations.iter().copied().map(frontier_midpoint));
    scratch.delaunay_points.clear();
    scratch
        .delaunay_points
        .extend(scratch.delaunay_midpoints.iter().map(|&(x, y)| Point {
            x: f64::from(x),
            y: f64::from(y),
        }));
    scratch.delaunay_edges.clear();
    scratch
        .delaunay_incident_edges
        .resize_with(locations.len(), Vec::new);
    scratch.delaunay_incident_edges.truncate(locations.len());
    for incident_edges in &mut scratch.delaunay_incident_edges {
        incident_edges.clear();
    }
    scratch.delaunay_degrees.clear();
    scratch.delaunay_degrees.resize(locations.len(), 0);

    let midpoints = &scratch.delaunay_midpoints;
    let points = &scratch.delaunay_points;
    let edges = &mut scratch.delaunay_edges;
    let incident_edges = &mut scratch.delaunay_incident_edges;
    let degrees = &mut scratch.delaunay_degrees;
    let mut add_edge = |a: usize, b: usize| {
        debug_assert_ne!(a, b);
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        let dx = i32::from(midpoints[a].0) - i32::from(midpoints[b].0);
        let dy = i32::from(midpoints[a].1) - i32::from(midpoints[b].1);
        let edge_idx = edges.len();
        edges.push(FrontierEdge {
            endpoints: [a, b],
            length_squared: dx * dx + dy * dy,
            active: true,
        });
        incident_edges[a].push(edge_idx);
        incident_edges[b].push(edge_idx);
        degrees[a] += 1;
        degrees[b] += 1;
    };

    match locations.len() {
        0 | 1 => {}
        2 => add_edge(0, 1),
        _ => {
            let triangulation = triangulate(&points);
            if triangulation.is_empty() {
                for pair in triangulation.hull.windows(2) {
                    add_edge(pair[0], pair[1]);
                }
            } else {
                for (halfedge_idx, &twin) in triangulation.halfedges.iter().enumerate() {
                    if twin == EMPTY || halfedge_idx < twin {
                        add_edge(
                            triangulation.triangles[halfedge_idx],
                            triangulation.triangles[next_halfedge(halfedge_idx)],
                        );
                    }
                }
            }
        }
    }

    prune_frontier_edges(edges, incident_edges, degrees, max_degree);

    scratch.delaunay_output_counts.clear();
    scratch.delaunay_output_counts.resize(locations.len(), 0);
    let output_counts = &mut scratch.delaunay_output_counts;
    for edge in edges.iter().copied().filter(|edge| edge.active) {
        let [a, b] = edge.endpoints;
        let a_offset = a * max_degree + output_counts[a];
        output[a_offset] = b as i16;
        output_counts[a] += 1;
        let b_offset = b * max_degree + output_counts[b];
        output[b_offset] = a as i16;
        output_counts[b] += 1;
    }
    for row_idx in 0..locations.len() {
        let row_start = row_idx * max_degree;
        let row_end = row_start + output_counts[row_idx];
        output[row_start..row_end].sort_unstable();
        debug_assert!(output_counts[row_idx] <= max_degree);
    }
}

#[cfg(test)]
fn frontier_delaunay_neighbors(locations: &[DoorLocation], max_degree: usize) -> Vec<Vec<usize>> {
    let mut scratch = FeatureScratch::default();
    let mut output = vec![-1; locations.len() * max_degree];
    write_frontier_delaunay_neighbors(locations, max_degree, &mut output, &mut scratch);
    flat_frontier_neighbors_to_rows(&output, max_degree)
}

#[cfg(test)]
fn frontier_nearest_neighbors(
    locations: &[DoorLocation],
    neighbor_count: usize,
    include_self: bool,
) -> Vec<Vec<usize>> {
    let mut scratch = FeatureScratch::default();
    let mut output = vec![-1; locations.len() * neighbor_count];
    write_frontier_nearest_neighbors(
        locations,
        neighbor_count,
        include_self,
        &mut output,
        &mut scratch,
    );
    flat_frontier_neighbors_to_rows(&output, neighbor_count)
}

#[cfg(test)]
fn flat_frontier_neighbors_to_rows(output: &[i16], neighbor_count: usize) -> Vec<Vec<usize>> {
    output
        .chunks(neighbor_count)
        .map(|row| {
            row.iter()
                .copied()
                .take_while(|&idx| idx >= 0)
                .map(|idx| idx as usize)
                .collect()
        })
        .collect()
}

fn write_frontier_nearest_neighbors(
    locations: &[DoorLocation],
    neighbor_count: usize,
    include_self: bool,
    output: &mut [i16],
    scratch: &mut FeatureScratch,
) {
    debug_assert_eq!(output.len(), locations.len() * neighbor_count);
    output.fill(-1);
    scratch.nearest_neighbor_indices.clear();
    scratch
        .nearest_neighbor_indices
        .resize(neighbor_count, usize::MAX);
    scratch.nearest_neighbor_keys.clear();
    scratch
        .nearest_neighbor_keys
        .resize(neighbor_count, (Coord::MAX, usize::MAX, usize::MAX));
    let neighbors = &mut scratch.nearest_neighbor_indices;
    let neighbor_keys = &mut scratch.nearest_neighbor_keys;
    for (src_idx, src) in locations.iter().enumerate() {
        neighbors.fill(usize::MAX);
        neighbor_keys.fill((Coord::MAX, usize::MAX, usize::MAX));
        let mut count = 0;
        for dst_idx in 0..locations.len() {
            if !include_self && dst_idx == src_idx {
                continue;
            }
            let dst_key = {
                let dst = locations[dst_idx];
                (
                    (src.x() - dst.x()).abs() + (src.y() - dst.y()).abs(),
                    usize::from(dst_idx != src_idx),
                    dst_idx,
                )
            };
            let insert_idx = (0..count)
                .position(|idx| neighbor_keys[idx] > dst_key)
                .unwrap_or(count);
            if insert_idx >= neighbor_count {
                continue;
            }
            count = (count + 1).min(neighbor_count);
            for idx in (insert_idx + 1..count).rev() {
                neighbors[idx] = neighbors[idx - 1];
                neighbor_keys[idx] = neighbor_keys[idx - 1];
            }
            neighbors[insert_idx] = dst_idx;
            neighbor_keys[insert_idx] = dst_key;
        }
        let row_start = src_idx * neighbor_count;
        for (idx, &neighbor) in neighbors[..count].iter().enumerate() {
            output[row_start + idx] = neighbor as i16;
        }
    }
}

fn write_single_frontier_nearest_neighbor(
    locations: &[DoorLocation],
    include_self: bool,
    output: &mut [i16],
) {
    debug_assert_eq!(locations.len(), output.len());
    for (src_idx, src) in locations.iter().enumerate() {
        let mut best_key = (Coord::MAX, usize::MAX, usize::MAX);
        let mut best_idx = -1;
        for (dst_idx, dst) in locations.iter().enumerate() {
            if !include_self && dst_idx == src_idx {
                continue;
            }
            let key = (
                (src.x() - dst.x()).abs() + (src.y() - dst.y()).abs(),
                usize::from(dst_idx != src_idx),
                dst_idx,
            );
            if key < best_key {
                best_key = key;
                best_idx = dst_idx as i16;
            }
        }
        output[src_idx] = best_idx;
    }
}

pub(crate) fn write_frontier_neighbors(
    locations: &[DoorLocation],
    algorithm: FrontierNeighborAlgorithm,
    neighbor_count: usize,
    output: &mut [i16],
    scratch: &mut FeatureScratch,
) {
    match algorithm {
        FrontierNeighborAlgorithm::Nearest if neighbor_count == 1 => {
            write_single_frontier_nearest_neighbor(locations, true, output);
        }
        FrontierNeighborAlgorithm::NearestExclusive if neighbor_count == 1 => {
            write_single_frontier_nearest_neighbor(locations, false, output);
        }
        FrontierNeighborAlgorithm::Delaunay => {
            write_frontier_delaunay_neighbors(locations, neighbor_count, output, scratch);
        }
        FrontierNeighborAlgorithm::Nearest => {
            write_frontier_nearest_neighbors(locations, neighbor_count, true, output, scratch);
        }
        FrontierNeighborAlgorithm::NearestExclusive => {
            write_frontier_nearest_neighbors(locations, neighbor_count, false, output, scratch);
        }
    }
}

fn prune_frontier_edges(
    edges: &mut [FrontierEdge],
    incident_edges: &[Vec<usize>],
    degrees: &mut [usize],
    max_degree: usize,
) {
    let mut excess_vertices = BinaryHeap::new();
    for (vertex, &degree) in degrees.iter().enumerate() {
        if degree > max_degree {
            excess_vertices.push((degree, Reverse(vertex)));
        }
    }
    while let Some((queued_degree, Reverse(vertex))) = excess_vertices.pop() {
        if degrees[vertex] != queued_degree || degrees[vertex] <= max_degree {
            continue;
        }
        let edge_idx = incident_edges[vertex]
            .iter()
            .copied()
            .filter(|&edge_idx| edges[edge_idx].active)
            .max_by_key(|&edge_idx| {
                let edge = edges[edge_idx];
                let neighbor = if edge.endpoints[0] == vertex {
                    edge.endpoints[1]
                } else {
                    edge.endpoints[0]
                };
                (degrees[neighbor], edge.length_squared, Reverse(neighbor))
            })
            .unwrap();
        let edge = &mut edges[edge_idx];
        edge.active = false;
        for &endpoint in &edge.endpoints {
            degrees[endpoint] -= 1;
            if degrees[endpoint] > max_degree {
                excess_vertices.push((degrees[endpoint], Reverse(endpoint)));
            }
        }
    }
}

pub struct Environment {
    rng: rand::rngs::StdRng, // for randomly choosing the initial room placement
    map_size: (Coord, Coord),
    actions: Vec<Action>, // history of room placements so far
    finished: bool,
    frontier: HashMap<DoorLocation, Frontier>, // info about each unconnected door on the map
    // Grouped by door direction: for each door, the index of the matching door on the other side (or DirDoorIdx::MAX if none):
    door_matches: [Vec<DirDoorIdx>; NUM_DIRS],
    room_used: BitVec,                           // whether each room has been used
    room_x: Vec<Coord>, // x position of each room (only valid for used rooms)
    room_y: Vec<Coord>, // y position of each room (only valid for used rooms)
    geometry_unused_count: Vec<usize>, // number of unused room representatives for each geometry
    connection_variant_unused_count: Vec<usize>, // number of unused room representatives for each connection variant
    room_part_component: Vec<usize>,             // maps placed room door groups to SCC components
    scc_dag: SccDag, // DAG of strongly connected components (condensation graph)
    active_room_parts: Vec<RoomPartIdx>,
    active_save_room_parts: Vec<RoomPartIdx>,
    active_refill_room_parts: Vec<RoomPartIdx>,
    graph_distance: Vec<GraphDistance>,
    room_part_furthest_distance_cache: RoomPartFurthestDistanceCache,
    room_part_save_distance_cache: RoomPartSaveDistanceCache,
    room_part_refill_distance_cache: RoomPartSaveDistanceCache,
    room_part_frontier_distance_cache: RoomPartFrontierDistanceCache,
    occupancy: Vec<u8>,
    placed_room_index: PlacedRoomIndex,
    intersection_query_actions: Vec<ActionIdx>,
    intersection_query_seen: Vec<u32>,
    intersection_query_generation: u32,
    known_outcomes: Option<StepOutcomes>,
    frontier_count_sum: u64,
    frontier_count_steps: u32,
}

struct FeatureSnapshot {
    finished: bool,
    frontier: HashMap<DoorLocation, Frontier>,
    connection_variant_idx: Option<ConnectionVariantIdx>,
    connection_variant_unused_count: usize,
    room_part_component: Vec<usize>,
    scc_dag: SccDag,
    active_room_parts_len: usize,
    active_save_room_parts_len: usize,
    active_refill_room_parts_len: usize,
    graph_distance_snapshot: GraphDistanceSnapshot,
    room_part_frontier_distance_cache: RoomPartFrontierDistanceCache,
}

struct LookaheadSnapshot {
    action_len: usize,
    finished: bool,
    frontier: HashMap<DoorLocation, Frontier>,
    room_idx: Option<RoomIdx>,
    room_used: bool,
    room_x: Coord,
    room_y: Coord,
    geometry_idx: Option<GeometryIdx>,
    geometry_unused_count: usize,
    connection_variant_idx: Option<ConnectionVariantIdx>,
    connection_variant_unused_count: usize,
    door_matches: Vec<(usize, usize, DirDoorIdx)>,
    room_part_component: Vec<usize>,
    scc_dag: SccDag,
    active_room_parts_len: usize,
    active_save_room_parts_len: usize,
    active_refill_room_parts_len: usize,
    graph_distance_snapshot: GraphDistanceSnapshot,
    room_part_frontier_distance_cache: RoomPartFrontierDistanceCache,
    placed_room_index_len: usize,
}

enum GraphDistanceSnapshot {
    None,
    NewRoom {
        room_idx: RoomIdx,
        furthest_distance_cache: RoomPartFurthestDistanceCache,
        save_distance_cache: RoomPartSaveDistanceCache,
        refill_distance_cache: RoomPartSaveDistanceCache,
    },
    Full {
        graph_distance: Vec<GraphDistance>,
        furthest_distance_cache: RoomPartFurthestDistanceCache,
        save_distance_cache: RoomPartSaveDistanceCache,
        refill_distance_cache: RoomPartSaveDistanceCache,
    },
}

struct PlacedRoomIndex {
    cell_size: usize,
    width: usize,
    height: usize,
    cells: Vec<Vec<ActionIdx>>,
    insertions: Vec<SpatialCellIdx>,
}

impl PlacedRoomIndex {
    fn new(map_size: (Coord, Coord), cell_size: usize) -> Self {
        debug_assert!(cell_size > 0);
        let width = (map_size.0 as usize).div_ceil(cell_size);
        let height = (map_size.1 as usize).div_ceil(cell_size);
        let cell_count = width * height;
        assert!(
            cell_count <= SpatialCellIdx::MAX as usize + 1,
            "spatial cell count must fit in SpatialCellIdx"
        );
        Self {
            cell_size,
            width,
            height,
            cells: vec![Vec::new(); cell_count],
            insertions: Vec::new(),
        }
    }

    fn clear(&mut self) {
        for cell in &mut self.cells {
            cell.clear();
        }
        self.insertions.clear();
    }

    fn insertion_len(&self) -> usize {
        self.insertions.len()
    }

    fn truncate_insertions(&mut self, len: usize) {
        while self.insertions.len() > len {
            let cell_idx = self
                .insertions
                .pop()
                .expect("insertion log length already checked");
            self.cells[cell_idx as usize]
                .pop()
                .expect("insertion log must match indexed cell contents");
        }
    }

    fn insert_action(&mut self, action_idx: ActionIdx, action: Action, geometry: &GeometryData) {
        let Some((x0, x1, y0, y1)) = self.cell_range(geometry, action.x, action.y) else {
            return;
        };
        for y in y0..=y1 {
            for x in x0..=x1 {
                let cell_idx = y * self.width + x;
                self.cells[cell_idx].push(action_idx);
                self.insertions.push(
                    SpatialCellIdx::try_from(cell_idx)
                        .expect("spatial cell index must fit in SpatialCellIdx"),
                );
            }
        }
    }

    fn query_geometry(
        &self,
        geometry: &GeometryData,
        x: Coord,
        y: Coord,
        output: &mut Vec<ActionIdx>,
    ) {
        output.clear();
        let Some((x0, x1, y0, y1)) = self.cell_range(geometry, x, y) else {
            return;
        };
        for cell_y in y0..=y1 {
            for cell_x in x0..=x1 {
                let cell_idx = cell_y * self.width + cell_x;
                output.extend(self.cells[cell_idx].iter().copied());
            }
        }
    }

    fn cell_range(
        &self,
        geometry: &GeometryData,
        x: Coord,
        y: Coord,
    ) -> Option<(usize, usize, usize, usize)> {
        if self.width == 0 || self.height == 0 {
            return None;
        }
        let min_x = (x as isize + geometry.min_x as isize).max(0) as usize;
        let max_x = (x as isize + geometry.max_x as isize).max(0) as usize;
        let min_y = (y as isize + geometry.min_y as isize).max(0) as usize;
        let max_y = (y as isize + geometry.max_y as isize).max(0) as usize;
        let x0 = (min_x / self.cell_size).min(self.width - 1);
        let x1 = (max_x / self.cell_size).min(self.width - 1);
        let y0 = (min_y / self.cell_size).min(self.height - 1);
        let y1 = (max_y / self.cell_size).min(self.height - 1);
        Some((x0, x1, y0, y1))
    }
}

#[derive(Clone)]
struct RoomPartFurthestDistanceCache {
    furthest_destination: Vec<GraphDistance>,
    furthest_source: Vec<GraphDistance>,
}

impl RoomPartFurthestDistanceCache {
    fn new(graph_size: usize) -> Self {
        Self {
            furthest_destination: vec![UNREACHABLE_DISTANCE; graph_size],
            furthest_source: vec![UNREACHABLE_DISTANCE; graph_size],
        }
    }

    fn clear(&mut self) {
        self.furthest_destination.fill(UNREACHABLE_DISTANCE);
        self.furthest_source.fill(UNREACHABLE_DISTANCE);
    }

    fn set_distance(
        &mut self,
        graph_distance: &[GraphDistance],
        graph_size: usize,
        from_part: usize,
        to_part: usize,
        old_distance: GraphDistance,
        new_distance: GraphDistance,
    ) {
        if old_distance == new_distance {
            return;
        }
        if new_distance != UNREACHABLE_DISTANCE
            && (self.furthest_destination[from_part] == UNREACHABLE_DISTANCE
                || new_distance > self.furthest_destination[from_part])
        {
            self.furthest_destination[from_part] = new_distance;
        } else if old_distance == self.furthest_destination[from_part]
            && new_distance < old_distance
        {
            self.furthest_destination[from_part] =
                Self::furthest_destination_for_source(graph_distance, graph_size, from_part);
        }

        if new_distance != UNREACHABLE_DISTANCE
            && (self.furthest_source[to_part] == UNREACHABLE_DISTANCE
                || new_distance > self.furthest_source[to_part])
        {
            self.furthest_source[to_part] = new_distance;
        } else if old_distance == self.furthest_source[to_part] && new_distance < old_distance {
            self.furthest_source[to_part] =
                Self::furthest_source_for_destination(graph_distance, graph_size, to_part);
        }
    }

    fn add_distance(&mut self, from_part: usize, to_part: usize, distance: GraphDistance) {
        if distance == UNREACHABLE_DISTANCE {
            return;
        }
        if self.furthest_destination[from_part] == UNREACHABLE_DISTANCE
            || distance > self.furthest_destination[from_part]
        {
            self.furthest_destination[from_part] = distance;
        }
        if self.furthest_source[to_part] == UNREACHABLE_DISTANCE
            || distance > self.furthest_source[to_part]
        {
            self.furthest_source[to_part] = distance;
        }
    }

    fn furthest_destination_for_source(
        graph_distance: &[GraphDistance],
        graph_size: usize,
        from_part: usize,
    ) -> GraphDistance {
        graph_distance[from_part * graph_size..(from_part + 1) * graph_size]
            .iter()
            .copied()
            .filter(|&distance| distance != UNREACHABLE_DISTANCE)
            .max()
            .unwrap_or(UNREACHABLE_DISTANCE)
    }

    fn furthest_source_for_destination(
        graph_distance: &[GraphDistance],
        graph_size: usize,
        to_part: usize,
    ) -> GraphDistance {
        (0..graph_size)
            .map(|from_part| graph_distance[from_part * graph_size + to_part])
            .filter(|&distance| distance != UNREACHABLE_DISTANCE)
            .max()
            .unwrap_or(UNREACHABLE_DISTANCE)
    }
}

#[derive(Clone)]
struct RoomPartSaveDistanceCache {
    save_room_part: Vec<bool>,
    nearest_save_destination: Vec<GraphDistance>,
    nearest_save_source: Vec<GraphDistance>,
}

impl RoomPartSaveDistanceCache {
    fn new(graph_size: usize) -> Self {
        Self {
            save_room_part: vec![false; graph_size],
            nearest_save_destination: vec![UNREACHABLE_DISTANCE; graph_size],
            nearest_save_source: vec![UNREACHABLE_DISTANCE; graph_size],
        }
    }

    fn clear(&mut self) {
        self.save_room_part.fill(false);
        self.nearest_save_destination.fill(UNREACHABLE_DISTANCE);
        self.nearest_save_source.fill(UNREACHABLE_DISTANCE);
    }

    fn add_save_part(
        &mut self,
        graph_distance: &[GraphDistance],
        graph_size: usize,
        active_room_parts: &[RoomPartIdx],
        save_part: usize,
    ) {
        self.save_room_part[save_part] = true;
        for &part in active_room_parts {
            let part = part as usize;
            let to_save = graph_distance[part * graph_size + save_part];
            if to_save < self.nearest_save_destination[part] {
                self.nearest_save_destination[part] = to_save;
            }
            let from_save = graph_distance[save_part * graph_size + part];
            if from_save < self.nearest_save_source[part] {
                self.nearest_save_source[part] = from_save;
            }
        }
    }

    fn set_distance(
        &mut self,
        graph_distance: &[GraphDistance],
        graph_size: usize,
        from_part: usize,
        to_part: usize,
        old_distance: GraphDistance,
        new_distance: GraphDistance,
    ) {
        if old_distance == new_distance {
            return;
        }
        if self.save_room_part[to_part] && new_distance < self.nearest_save_destination[from_part] {
            self.nearest_save_destination[from_part] = new_distance;
        } else if self.save_room_part[to_part]
            && old_distance == self.nearest_save_destination[from_part]
            && new_distance > old_distance
        {
            self.nearest_save_destination[from_part] =
                self.nearest_save_destination_for_part(graph_distance, graph_size, from_part);
        }

        if self.save_room_part[from_part] && new_distance < self.nearest_save_source[to_part] {
            self.nearest_save_source[to_part] = new_distance;
        } else if self.save_room_part[from_part]
            && old_distance == self.nearest_save_source[to_part]
            && new_distance > old_distance
        {
            self.nearest_save_source[to_part] =
                self.nearest_save_source_for_part(graph_distance, graph_size, to_part);
        }
    }

    fn initialize_single_attachment_room_part(
        &mut self,
        part: usize,
        attached_part: usize,
        to_attached_part: GraphDistance,
        from_attached_part: GraphDistance,
    ) {
        self.nearest_save_destination[part] = graph_distance_sum(&[
            to_attached_part,
            self.nearest_save_destination[attached_part],
        ])
        .unwrap_or(UNREACHABLE_DISTANCE);
        self.nearest_save_source[part] =
            graph_distance_sum(&[self.nearest_save_source[attached_part], from_attached_part])
                .unwrap_or(UNREACHABLE_DISTANCE);
    }

    fn nearest_save_destination_for_part(
        &self,
        graph_distance: &[GraphDistance],
        graph_size: usize,
        part: usize,
    ) -> GraphDistance {
        self.save_room_part
            .iter()
            .enumerate()
            .filter(|&(_, &is_save)| is_save)
            .map(|(save_part, _)| graph_distance[part * graph_size + save_part])
            .min()
            .unwrap_or(UNREACHABLE_DISTANCE)
    }

    fn nearest_save_source_for_part(
        &self,
        graph_distance: &[GraphDistance],
        graph_size: usize,
        part: usize,
    ) -> GraphDistance {
        self.save_room_part
            .iter()
            .enumerate()
            .filter(|&(_, &is_save)| is_save)
            .map(|(save_part, _)| graph_distance[save_part * graph_size + part])
            .min()
            .unwrap_or(UNREACHABLE_DISTANCE)
    }
}

#[derive(Clone)]
struct RoomPartFrontierDistanceCache {
    frontier_room_part_count: Vec<u16>,
    nearest_frontier_destination: Vec<GraphDistance>,
    nearest_frontier_destination_part: Vec<RoomPartIdx>,
    nearest_frontier_source: Vec<GraphDistance>,
    nearest_frontier_source_part: Vec<RoomPartIdx>,
}

impl RoomPartFrontierDistanceCache {
    fn new(graph_size: usize) -> Self {
        Self {
            frontier_room_part_count: vec![0; graph_size],
            nearest_frontier_destination: vec![UNREACHABLE_DISTANCE; graph_size],
            nearest_frontier_destination_part: vec![RoomPartIdx::MAX; graph_size],
            nearest_frontier_source: vec![UNREACHABLE_DISTANCE; graph_size],
            nearest_frontier_source_part: vec![RoomPartIdx::MAX; graph_size],
        }
    }

    fn clear(&mut self) {
        self.frontier_room_part_count.fill(0);
        self.nearest_frontier_destination.fill(UNREACHABLE_DISTANCE);
        self.nearest_frontier_destination_part
            .fill(RoomPartIdx::MAX);
        self.nearest_frontier_source.fill(UNREACHABLE_DISTANCE);
        self.nearest_frontier_source_part.fill(RoomPartIdx::MAX);
    }

    fn add_frontier_part(
        &mut self,
        graph_distance: &[GraphDistance],
        graph_size: usize,
        active_room_parts: &[RoomPartIdx],
        frontier_part: usize,
    ) {
        self.frontier_room_part_count[frontier_part] += 1;
        if self.frontier_room_part_count[frontier_part] > 1 {
            return;
        }
        for &part in active_room_parts {
            let part = part as usize;
            let to_frontier = graph_distance[part * graph_size + frontier_part];
            if to_frontier < self.nearest_frontier_destination[part] {
                self.nearest_frontier_destination[part] = to_frontier;
                self.nearest_frontier_destination_part[part] = frontier_part as RoomPartIdx;
            }
            let from_frontier = graph_distance[frontier_part * graph_size + part];
            if from_frontier < self.nearest_frontier_source[part] {
                self.nearest_frontier_source[part] = from_frontier;
                self.nearest_frontier_source_part[part] = frontier_part as RoomPartIdx;
            }
        }
    }

    fn remove_frontier_part(
        &mut self,
        graph_distance: &[GraphDistance],
        graph_size: usize,
        active_room_parts: &[RoomPartIdx],
        frontier_part: usize,
    ) {
        debug_assert!(self.frontier_room_part_count[frontier_part] > 0);
        self.frontier_room_part_count[frontier_part] -= 1;
        if self.frontier_room_part_count[frontier_part] > 0 {
            return;
        }
        for &part in active_room_parts {
            let part = part as usize;
            if self.nearest_frontier_destination_part[part] == frontier_part as RoomPartIdx {
                let (distance, nearest_part) =
                    self.nearest_frontier_destination_for_part(graph_distance, graph_size, part);
                self.nearest_frontier_destination[part] = distance;
                self.nearest_frontier_destination_part[part] = nearest_part;
            }
            if self.nearest_frontier_source_part[part] == frontier_part as RoomPartIdx {
                let (distance, nearest_part) =
                    self.nearest_frontier_source_for_part(graph_distance, graph_size, part);
                self.nearest_frontier_source[part] = distance;
                self.nearest_frontier_source_part[part] = nearest_part;
            }
        }
    }

    fn set_distance(
        &mut self,
        graph_distance: &[GraphDistance],
        graph_size: usize,
        from_part: usize,
        to_part: usize,
        old_distance: GraphDistance,
        new_distance: GraphDistance,
    ) {
        if old_distance == new_distance {
            return;
        }
        if self.frontier_room_part_count[to_part] > 0
            && new_distance < self.nearest_frontier_destination[from_part]
        {
            self.nearest_frontier_destination[from_part] = new_distance;
            self.nearest_frontier_destination_part[from_part] = to_part as RoomPartIdx;
        } else if self.frontier_room_part_count[to_part] > 0
            && old_distance == self.nearest_frontier_destination[from_part]
            && new_distance > old_distance
            && self.nearest_frontier_destination_part[from_part] == to_part as RoomPartIdx
        {
            let (distance, nearest_part) =
                self.nearest_frontier_destination_for_part(graph_distance, graph_size, from_part);
            self.nearest_frontier_destination[from_part] = distance;
            self.nearest_frontier_destination_part[from_part] = nearest_part;
        }

        if self.frontier_room_part_count[from_part] > 0
            && new_distance < self.nearest_frontier_source[to_part]
        {
            self.nearest_frontier_source[to_part] = new_distance;
            self.nearest_frontier_source_part[to_part] = from_part as RoomPartIdx;
        } else if self.frontier_room_part_count[from_part] > 0
            && old_distance == self.nearest_frontier_source[to_part]
            && new_distance > old_distance
            && self.nearest_frontier_source_part[to_part] == from_part as RoomPartIdx
        {
            let (distance, nearest_part) =
                self.nearest_frontier_source_for_part(graph_distance, graph_size, to_part);
            self.nearest_frontier_source[to_part] = distance;
            self.nearest_frontier_source_part[to_part] = nearest_part;
        }
    }

    fn initialize_single_attachment_room_part(
        &mut self,
        part: usize,
        attached_part: usize,
        to_attached_part: GraphDistance,
        from_attached_part: GraphDistance,
    ) {
        self.nearest_frontier_destination[part] = graph_distance_sum(&[
            to_attached_part,
            self.nearest_frontier_destination[attached_part],
        ])
        .unwrap_or(UNREACHABLE_DISTANCE);
        self.nearest_frontier_destination_part[part] =
            self.nearest_frontier_destination_part[attached_part];
        self.nearest_frontier_source[part] = graph_distance_sum(&[
            self.nearest_frontier_source[attached_part],
            from_attached_part,
        ])
        .unwrap_or(UNREACHABLE_DISTANCE);
        self.nearest_frontier_source_part[part] = self.nearest_frontier_source_part[attached_part];
    }

    fn nearest_frontier_destination_for_part(
        &self,
        graph_distance: &[GraphDistance],
        graph_size: usize,
        part: usize,
    ) -> (GraphDistance, RoomPartIdx) {
        self.frontier_room_part_count
            .iter()
            .enumerate()
            .filter(|&(_, &count)| count > 0)
            .map(|(frontier_part, _)| {
                (
                    graph_distance[part * graph_size + frontier_part],
                    frontier_part as RoomPartIdx,
                )
            })
            .min_by_key(|&(distance, frontier_part)| (distance, frontier_part))
            .unwrap_or((UNREACHABLE_DISTANCE, RoomPartIdx::MAX))
    }

    fn nearest_frontier_source_for_part(
        &self,
        graph_distance: &[GraphDistance],
        graph_size: usize,
        part: usize,
    ) -> (GraphDistance, RoomPartIdx) {
        self.frontier_room_part_count
            .iter()
            .enumerate()
            .filter(|&(_, &count)| count > 0)
            .map(|(frontier_part, _)| {
                (
                    graph_distance[frontier_part * graph_size + part],
                    frontier_part as RoomPartIdx,
                )
            })
            .min_by_key(|&(distance, frontier_part)| (distance, frontier_part))
            .unwrap_or((UNREACHABLE_DISTANCE, RoomPartIdx::MAX))
    }
}

impl Environment {
    pub fn new(
        common: &CommonData,
        map_size: (Coord, Coord),
        candidate_spatial_cell_size: usize,
        seed: u64,
    ) -> Self {
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            map_size,
            actions: vec![],
            finished: false,
            frontier: HashMap::new(),
            door_matches: std::array::from_fn(|i| {
                vec![DirDoorIdx::MAX; common.room_dir_door[i].len()]
            }),
            room_used: BitVec::repeat(false, common.room.len()),
            room_x: vec![0; common.room.len()],
            room_y: vec![0; common.room.len()],
            geometry_unused_count: common
                .geometry_rooms
                .iter()
                .map(|rooms| rooms.len())
                .collect(),
            connection_variant_unused_count: common
                .connection_variant_rooms
                .iter()
                .map(|rooms| rooms.len())
                .collect(),
            room_part_component: vec![NO_COMPONENT; common.room_part.len()],
            scc_dag: SccDag::default(),
            active_room_parts: Vec::new(),
            active_save_room_parts: Vec::new(),
            active_refill_room_parts: Vec::new(),
            graph_distance: vec![
                UNREACHABLE_DISTANCE;
                common.room_part.len() * common.room_part.len()
            ],
            room_part_furthest_distance_cache: RoomPartFurthestDistanceCache::new(
                common.room_part.len(),
            ),
            room_part_save_distance_cache: RoomPartSaveDistanceCache::new(common.room_part.len()),
            room_part_refill_distance_cache: RoomPartSaveDistanceCache::new(common.room_part.len()),
            room_part_frontier_distance_cache: RoomPartFrontierDistanceCache::new(
                common.room_part.len(),
            ),
            occupancy: vec![0; map_size.0 as usize * map_size.1 as usize],
            placed_room_index: PlacedRoomIndex::new(map_size, candidate_spatial_cell_size),
            intersection_query_actions: Vec::new(),
            intersection_query_seen: vec![0; common.room.len()],
            intersection_query_generation: 0,
            known_outcomes: None,
            frontier_count_sum: 0,
            frontier_count_steps: 0,
        }
    }

    pub fn clear(&mut self, common: &CommonData) {
        self.actions.clear();
        self.finished = false;
        self.frontier.clear();
        self.door_matches
            .iter_mut()
            .for_each(|matches| matches.fill(DirDoorIdx::MAX));
        self.room_used.fill(false);
        self.geometry_unused_count.clear();
        self.geometry_unused_count
            .extend(common.geometry_rooms.iter().map(|rooms| rooms.len()));
        self.connection_variant_unused_count.clear();
        self.connection_variant_unused_count.extend(
            common
                .connection_variant_rooms
                .iter()
                .map(|rooms| rooms.len()),
        );
        self.room_part_component.fill(NO_COMPONENT);
        self.scc_dag.clear();
        self.active_room_parts.clear();
        self.active_save_room_parts.clear();
        self.active_refill_room_parts.clear();
        self.graph_distance.fill(UNREACHABLE_DISTANCE);
        self.room_part_furthest_distance_cache.clear();
        self.room_part_save_distance_cache.clear();
        self.room_part_refill_distance_cache.clear();
        self.room_part_frontier_distance_cache.clear();
        self.occupancy.fill(0);
        self.placed_room_index.clear();
        self.intersection_query_actions.clear();
        self.intersection_query_seen.fill(0);
        self.intersection_query_generation = 0;
        self.known_outcomes = None;
        self.frontier_count_sum = 0;
        self.frontier_count_steps = 0;
    }

    pub fn get_initial_action(&mut self, common: &CommonData) -> Action {
        // Select a room and position uniformly at random.
        let room_idx = self.rng.random_range(0..common.room.len() as RoomIdx);
        let geometry_idx = common.room[room_idx as usize].geometry_idx;
        let geometry = &common.geometry[geometry_idx as usize];
        let min_x = -geometry.min_x;
        let max_x = self.map_size.0 - 1 - geometry.max_x;
        let min_y = -geometry.min_y;
        let max_y = self.map_size.1 - 1 - geometry.max_y;
        let x = self.rng.random_range(min_x..=max_x);
        let y = self.rng.random_range(min_y..=max_y);
        Action { room_idx, x, y }
    }

    fn sorted_frontiers(&self) -> Vec<(&DoorLocation, &Frontier)> {
        let mut sorted_frontiers = self.frontier.iter().collect::<Vec<_>>();
        sorted_frontiers.sort_unstable_by_key(|(location, _)| **location);
        sorted_frontiers
    }

    fn sorted_frontier_locations(&self) -> Vec<DoorLocation> {
        self.sorted_frontiers()
            .iter()
            .map(|(location, _)| **location)
            .collect()
    }

    pub fn proposal_candidate_mask(
        &self,
        common: &CommonData,
        proposal_door_variant_count: usize,
        frontier_idx: &mut FrontierIdx,
        output: &mut [u8],
        valid_counts: &mut usize,
    ) {
        let mask_byte_count = proposal_door_variant_count.div_ceil(8);
        debug_assert_eq!(output.len(), mask_byte_count);
        *frontier_idx = -1;
        *valid_counts = 0;
        output.fill(0);
        if self.actions.is_empty() {
            return;
        }
        let sorted_frontiers = self.sorted_frontiers();
        let mut selected_frontiers = sorted_frontiers
            .iter()
            .enumerate()
            .filter(|(_, (_, frontier))| !frontier.candidates.is_empty())
            .collect::<Vec<_>>();
        selected_frontiers.sort_unstable_by_key(|(frontier_idx, (_, frontier))| {
            (frontier.candidates.len(), *frontier_idx)
        });
        if let Some((selected_frontier_idx, (_, frontier))) = selected_frontiers.into_iter().next()
        {
            *frontier_idx = selected_frontier_idx as FrontierIdx;
            let mut valid_count = 0;
            for candidate in &frontier.candidates {
                for &connection_variant_idx in
                    common.geometry_connection_variants[candidate.geometry_idx as usize].iter()
                {
                    if self.connection_variant_unused_count[connection_variant_idx as usize] == 0 {
                        continue;
                    }
                    let door_variant_idx = common.door_variant_idx(
                        connection_variant_idx,
                        candidate.door_direction,
                        candidate.door_x,
                        candidate.door_y,
                        candidate.door_kind,
                    );
                    let door_variant_idx = door_variant_idx as usize;
                    assert!(door_variant_idx < proposal_door_variant_count);
                    let byte = &mut output[door_variant_idx / 8];
                    let mask = 1 << (door_variant_idx % 8);
                    if *byte & mask == 0 {
                        *byte |= mask;
                        valid_count += 1;
                    }
                }
            }
            debug_assert!(valid_count > 0);
            *valid_counts = valid_count;
        }
    }

    fn action_for_proposal_candidate(
        &mut self,
        common: &CommonData,
        sorted_frontier_locations: &[DoorLocation],
        frontier_idx: FrontierIdx,
        door_variant_idx: DoorVariantIdx,
    ) -> Option<Action> {
        if frontier_idx < 0 || door_variant_idx < 0 {
            return None;
        }
        let frontier_idx = frontier_idx as usize;
        let door_variant_idx = door_variant_idx as usize;
        let frontier = self
            .frontier
            .get(sorted_frontier_locations.get(frontier_idx)?)?;
        let mut matching_count = 0usize;
        let mut selected = None;
        for &candidate in &frontier.candidates {
            for &connection_variant_idx in
                common.geometry_connection_variants[candidate.geometry_idx as usize].iter()
            {
                if self.connection_variant_unused_count[connection_variant_idx as usize] == 0 {
                    continue;
                }
                if common.door_variant_idx(
                    connection_variant_idx,
                    candidate.door_direction,
                    candidate.door_x,
                    candidate.door_y,
                    candidate.door_kind,
                ) as usize
                    != door_variant_idx
                {
                    continue;
                }
                for &room_idx in
                    common.connection_variant_rooms[connection_variant_idx as usize].iter()
                {
                    if self.room_used[room_idx as usize] {
                        continue;
                    }
                    matching_count += 1;
                    let action = Action {
                        room_idx,
                        x: candidate.x,
                        y: candidate.y,
                    };
                    if self.rng.random_range(0..matching_count) == 0 {
                        selected = Some(action);
                    }
                }
            }
        }
        selected
    }

    pub fn step(&mut self, action: Action, common: &CommonData) {
        self.record_frontier_count();
        self.step_impl(action, common, StepMode::CommitFull);
    }

    pub fn step_known(&mut self, action: Action, common: &CommonData) {
        self.record_frontier_count();
        self.step_impl(action, common, StepMode::CommitKnown);
    }

    fn record_frontier_count(&mut self) {
        self.frontier_count_sum += self.frontier.len() as u64;
        self.frontier_count_steps += 1;
    }

    pub fn avg_frontiers(&self) -> Result<f32, String> {
        if self.frontier_count_steps == 0 {
            return Err("avg_frontiers requires at least one recorded step".to_string());
        }
        Ok(self.frontier_count_sum as f32 / self.frontier_count_steps as f32)
    }

    pub fn graph_diameter(&self) -> GraphDistance {
        self.graph_distance
            .iter()
            .copied()
            .filter(|&distance| distance != UNREACHABLE_DISTANCE)
            .max()
            .unwrap_or(0)
    }

    pub fn active_room_part_mask(&self, common: &CommonData) -> Vec<u8> {
        let mut mask = vec![0; common.room_part.len()];
        for &room_part in &self.active_room_parts {
            mask[room_part as usize] = 1;
        }
        mask
    }

    fn room_distances(
        &self,
        common: &CommonData,
        destination_parts: &[RoomPartIdx],
    ) -> (Vec<f32>, Vec<u8>) {
        let graph_size = common.room_part.len();
        let mut values = vec![0.0; graph_size];
        let mut mask = vec![0; graph_size];

        if destination_parts.is_empty() {
            return (values, mask);
        }

        for &room_part in &self.active_room_parts {
            let part = room_part as usize;
            let mut from_destination = UNREACHABLE_DISTANCE;
            let mut to_destination = UNREACHABLE_DISTANCE;
            for &destination_part in destination_parts {
                let destination_part = destination_part as usize;
                from_destination =
                    from_destination.min(self.graph_distance[destination_part * graph_size + part]);
                to_destination =
                    to_destination.min(self.graph_distance[part * graph_size + destination_part]);
            }
            if from_destination != UNREACHABLE_DISTANCE && to_destination != UNREACHABLE_DISTANCE {
                values[part] = f32::from(from_destination) + f32::from(to_destination);
                mask[part] = 1;
            }
        }

        (values, mask)
    }

    pub fn save_distances(&self, common: &CommonData) -> (Vec<f32>, Vec<u8>) {
        self.room_distances(common, &self.active_save_room_parts)
    }

    pub fn refill_distances(&self, common: &CommonData) -> (Vec<f32>, Vec<u8>) {
        self.room_distances(common, &self.active_refill_room_parts)
    }

    fn directed_room_distances(
        &self,
        common: &CommonData,
        destination_parts: &[RoomPartIdx],
    ) -> (Vec<f32>, Vec<u8>, Vec<f32>, Vec<u8>) {
        let graph_size = common.room_part.len();
        let mut to_room = vec![0.0; graph_size];
        let mut to_room_mask = vec![0; graph_size];
        let mut from_room = vec![0.0; graph_size];
        let mut from_room_mask = vec![0; graph_size];

        if destination_parts.is_empty() {
            return (to_room, to_room_mask, from_room, from_room_mask);
        }

        for &room_part in &self.active_room_parts {
            let part = room_part as usize;
            let mut nearest_to_room = UNREACHABLE_DISTANCE;
            let mut nearest_from_room = UNREACHABLE_DISTANCE;
            for &destination_part in destination_parts {
                let destination_part = destination_part as usize;
                nearest_to_room =
                    nearest_to_room.min(self.graph_distance[destination_part * graph_size + part]);
                nearest_from_room = nearest_from_room
                    .min(self.graph_distance[part * graph_size + destination_part]);
            }
            if nearest_to_room != UNREACHABLE_DISTANCE {
                to_room[part] = f32::from(nearest_to_room);
                to_room_mask[part] = 1;
            }
            if nearest_from_room != UNREACHABLE_DISTANCE {
                from_room[part] = f32::from(nearest_from_room);
                from_room_mask[part] = 1;
            }
        }

        (to_room, to_room_mask, from_room, from_room_mask)
    }

    pub fn directed_save_distances(
        &self,
        common: &CommonData,
    ) -> (Vec<f32>, Vec<u8>, Vec<f32>, Vec<u8>) {
        self.directed_room_distances(common, &self.active_save_room_parts)
    }

    pub fn directed_refill_distances(
        &self,
        common: &CommonData,
    ) -> (Vec<f32>, Vec<u8>, Vec<f32>, Vec<u8>) {
        self.directed_room_distances(common, &self.active_refill_room_parts)
    }

    pub fn missing_connect_distances(&self, common: &CommonData) -> (Vec<f32>, Vec<u8>) {
        let graph_size = common.room_part.len();
        let mut values = vec![0.0; common.room_connection.len()];
        let mut mask = vec![0; common.room_connection.len()];
        for (connection_idx, connection) in common.room_connection.iter().enumerate() {
            let source_part =
                Self::room_part_idx(common, connection.room_idx, connection.from_part);
            let destination_part =
                Self::room_part_idx(common, connection.room_idx, connection.to_part);
            let source_part = usize::from(source_part);
            let destination_part = usize::from(destination_part);
            let distance = self.graph_distance[source_part * graph_size + destination_part];
            if distance != UNREACHABLE_DISTANCE {
                values[connection_idx] = f32::from(distance);
                mask[connection_idx] = 1;
            }
        }
        (values, mask)
    }

    #[cfg(test)]
    fn room_part_furthest_distance_features(&self, common: &CommonData) -> (Vec<u8>, Vec<u8>) {
        let mut destination = Vec::new();
        let mut source = Vec::new();
        self.room_part_furthest_distance_features_into(common, &mut destination, &mut source);
        (destination, source)
    }

    fn room_part_furthest_distance_features_into(
        &self,
        common: &CommonData,
        destination: &mut Vec<u8>,
        source: &mut Vec<u8>,
    ) {
        debug_assert_eq!(
            self.room_part_furthest_distance_cache
                .furthest_destination
                .len(),
            common.room_part.len()
        );
        destination.extend(
            self.room_part_furthest_distance_cache
                .furthest_destination
                .iter()
                .copied()
                .map(encode_room_part_distance_feature),
        );
        source.extend(
            self.room_part_furthest_distance_cache
                .furthest_source
                .iter()
                .copied()
                .map(encode_room_part_distance_feature),
        );
    }

    fn encode_room_part_directed_distance_features_into(
        destination: &[GraphDistance],
        source: &[GraphDistance],
        output_destination: &mut Vec<u8>,
        output_source: &mut Vec<u8>,
    ) {
        output_destination.extend(
            destination
                .iter()
                .copied()
                .map(encode_room_part_distance_feature),
        );
        output_source.extend(
            source
                .iter()
                .copied()
                .map(encode_room_part_distance_feature),
        );
    }

    #[cfg(test)]
    fn room_part_save_distance_features(&self, common: &CommonData) -> (Vec<u8>, Vec<u8>) {
        let mut destination = Vec::new();
        let mut source = Vec::new();
        self.room_part_save_distance_features_into(common, &mut destination, &mut source);
        (destination, source)
    }

    fn room_part_save_distance_features_into(
        &self,
        common: &CommonData,
        destination: &mut Vec<u8>,
        source: &mut Vec<u8>,
    ) {
        debug_assert_eq!(
            self.room_part_save_distance_cache
                .nearest_save_destination
                .len(),
            common.room_part.len()
        );
        Self::encode_room_part_directed_distance_features_into(
            &self.room_part_save_distance_cache.nearest_save_destination,
            &self.room_part_save_distance_cache.nearest_save_source,
            destination,
            source,
        );
    }

    fn room_part_refill_distance_features_into(
        &self,
        common: &CommonData,
        destination: &mut Vec<u8>,
        source: &mut Vec<u8>,
    ) {
        debug_assert_eq!(
            self.room_part_refill_distance_cache
                .nearest_save_destination
                .len(),
            common.room_part.len()
        );
        Self::encode_room_part_directed_distance_features_into(
            &self
                .room_part_refill_distance_cache
                .nearest_save_destination,
            &self.room_part_refill_distance_cache.nearest_save_source,
            destination,
            source,
        );
    }

    #[cfg(test)]
    fn room_part_frontier_distance_features(&self, common: &CommonData) -> (Vec<u8>, Vec<u8>) {
        let mut destination = Vec::new();
        let mut source = Vec::new();
        self.room_part_frontier_distance_features_into(common, &mut destination, &mut source);
        (destination, source)
    }

    fn room_part_frontier_distance_features_into(
        &self,
        common: &CommonData,
        destination: &mut Vec<u8>,
        source: &mut Vec<u8>,
    ) {
        debug_assert_eq!(
            self.room_part_frontier_distance_cache
                .nearest_frontier_destination
                .len(),
            common.room_part.len()
        );
        Self::encode_room_part_directed_distance_features_into(
            &self
                .room_part_frontier_distance_cache
                .nearest_frontier_destination,
            &self
                .room_part_frontier_distance_cache
                .nearest_frontier_source,
            destination,
            source,
        );
    }

    fn known_save_refill_distance_features_into(
        &self,
        common: &CommonData,
        save_distance_cache: &RoomPartSaveDistanceCache,
        from_room: &mut Vec<u8>,
        to_room: &mut Vec<u8>,
    ) {
        debug_assert_eq!(
            save_distance_cache.nearest_save_destination.len(),
            common.room_part.len()
        );
        from_room.resize(common.room_part.len(), KNOWN_DISTANCE_UNKNOWN);
        to_room.resize(common.room_part.len(), KNOWN_DISTANCE_UNKNOWN);
        for &part in &self.active_room_parts {
            let part = part as usize;
            from_room[part] = encode_known_finalized_distance(
                save_distance_cache.nearest_save_destination[part],
                self.room_part_frontier_distance_cache
                    .nearest_frontier_destination[part],
            );
            to_room[part] = encode_known_finalized_distance(
                save_distance_cache.nearest_save_source[part],
                self.room_part_frontier_distance_cache
                    .nearest_frontier_source[part],
            );
        }
    }

    #[cfg(test)]
    fn slow_room_part_furthest_distance_features(&self, common: &CommonData) -> (Vec<u8>, Vec<u8>) {
        fn encode_distance(distance: GraphDistance) -> u8 {
            if distance == UNREACHABLE_DISTANCE {
                0
            } else {
                distance + 1
            }
        }

        let graph_size = common.room_part.len();
        let mut furthest_destination = vec![UNREACHABLE_DISTANCE; graph_size];
        let mut furthest_source = vec![UNREACHABLE_DISTANCE; graph_size];
        for source in 0..graph_size {
            for destination in 0..graph_size {
                let distance = self.graph_distance[source * graph_size + destination];
                if distance == UNREACHABLE_DISTANCE {
                    continue;
                }
                if furthest_destination[source] == UNREACHABLE_DISTANCE
                    || distance > furthest_destination[source]
                {
                    furthest_destination[source] = distance;
                }
                if furthest_source[destination] == UNREACHABLE_DISTANCE
                    || distance > furthest_source[destination]
                {
                    furthest_source[destination] = distance;
                }
            }
        }
        (
            furthest_destination
                .into_iter()
                .map(encode_distance)
                .collect(),
            furthest_source.into_iter().map(encode_distance).collect(),
        )
    }

    #[cfg(test)]
    fn slow_room_part_save_distance_features(&self, common: &CommonData) -> (Vec<u8>, Vec<u8>) {
        fn encode_distance(distance: GraphDistance) -> u8 {
            if distance == UNREACHABLE_DISTANCE {
                0
            } else {
                distance.saturating_add(1)
            }
        }

        let graph_size = common.room_part.len();
        let save_parts = common
            .save_room_part
            .iter()
            .copied()
            .filter(|room_part| self.active_room_parts.contains(room_part))
            .map(usize::from)
            .collect::<Vec<_>>();
        let mut destination = Vec::with_capacity(graph_size);
        let mut source = Vec::with_capacity(graph_size);
        for part in 0..graph_size {
            let nearest_save_destination = save_parts
                .iter()
                .map(|&save_part| self.graph_distance[part * graph_size + save_part])
                .min()
                .unwrap_or(UNREACHABLE_DISTANCE);
            let nearest_save_source = save_parts
                .iter()
                .map(|&save_part| self.graph_distance[save_part * graph_size + part])
                .min()
                .unwrap_or(UNREACHABLE_DISTANCE);
            destination.push(encode_distance(nearest_save_destination));
            source.push(encode_distance(nearest_save_source));
        }
        (destination, source)
    }

    #[cfg(test)]
    fn slow_room_part_frontier_distance_features(&self, common: &CommonData) -> (Vec<u8>, Vec<u8>) {
        fn encode_distance(distance: GraphDistance) -> u8 {
            if distance == UNREACHABLE_DISTANCE {
                0
            } else {
                distance.saturating_add(1)
            }
        }

        let graph_size = common.room_part.len();
        let frontier_parts = self
            .frontier
            .values()
            .map(|frontier| frontier.room_part_idx as usize)
            .collect::<Vec<_>>();
        let mut destination = Vec::with_capacity(graph_size);
        let mut source = Vec::with_capacity(graph_size);
        for part in 0..graph_size {
            let nearest_frontier_destination = frontier_parts
                .iter()
                .map(|&frontier_part| self.graph_distance[part * graph_size + frontier_part])
                .min()
                .unwrap_or(UNREACHABLE_DISTANCE);
            let nearest_frontier_source = frontier_parts
                .iter()
                .map(|&frontier_part| self.graph_distance[frontier_part * graph_size + part])
                .min()
                .unwrap_or(UNREACHABLE_DISTANCE);
            destination.push(encode_distance(nearest_frontier_destination));
            source.push(encode_distance(nearest_frontier_source));
        }
        (destination, source)
    }

    #[cfg(test)]
    fn assert_room_part_furthest_distance_cache_matches_slow(&self, common: &CommonData) {
        assert_eq!(
            self.room_part_furthest_distance_features(common),
            self.slow_room_part_furthest_distance_features(common)
        );
    }

    #[cfg(test)]
    fn assert_room_part_save_distance_cache_matches_slow(&self, common: &CommonData) {
        assert_eq!(
            self.room_part_save_distance_features(common),
            self.slow_room_part_save_distance_features(common)
        );
    }

    #[cfg(test)]
    fn assert_room_part_frontier_distance_cache_matches_slow(&self, common: &CommonData) {
        assert_eq!(
            self.room_part_frontier_distance_features(common),
            self.slow_room_part_frontier_distance_features(common)
        );
    }

    fn step_for_lookahead(&mut self, action: Action, common: &CommonData) {
        self.step_impl(action, common, StepMode::Lookahead);
    }

    fn step_for_features(&mut self, action: Action, common: &CommonData) {
        self.step_impl(action, common, StepMode::FeatureOnly);
    }

    fn step_impl(&mut self, action: Action, common: &CommonData, mode: StepMode) {
        let mut indexed_action_idx = None;
        if mode.records_action() {
            let profile = profile_start();
            indexed_action_idx = Some(
                ActionIdx::try_from(self.actions.len())
                    .expect("placed action index must fit in ActionIdx"),
            );
            self.actions.push(action);
            profile_end(ProfileMetric::EnvStepPushAction, profile);
        }
        if self.finished {
            return;
        }
        if action.room_idx >= common.room.len() as RoomIdx {
            // Dummy/invalid action: do nothing more.
            self.finished = true;
            return;
        }
        let room = &common.room[action.room_idx as usize];
        let action_geometry_idx = room.geometry_idx;
        let connection_variant_idx = room.connection_variant_idx;
        assert!(!self.room_used[action.room_idx as usize]);
        if let Some(action_idx) = indexed_action_idx {
            self.placed_room_index.insert_action(
                action_idx,
                action,
                &common.geometry[action_geometry_idx as usize],
            );
        }
        let profile = profile_start();
        self.room_used.set(action.room_idx as usize, true);
        self.room_x[action.room_idx as usize] = action.x;
        self.room_y[action.room_idx as usize] = action.y;
        if mode.updates_geometry_inventory() {
            self.geometry_unused_count[action_geometry_idx as usize] -= 1;
        }
        if mode.updates_connection_variant_inventory() {
            self.connection_variant_unused_count[connection_variant_idx as usize] -= 1;
        }
        profile_end(ProfileMetric::EnvStepMarkRoomUsed, profile);

        let profile = profile_start();
        self.add_room_components_and_edges(action, common);
        profile_end(ProfileMetric::EnvStepComponentsEdges, profile);

        let profile = profile_start();
        if mode.updates_occupancy() {
            for &(dx, dy) in &common.geometry[action_geometry_idx as usize].occupied_tiles {
                let x = action.x + dx;
                let y = action.y + dy;
                if x >= 0 && y >= 0 && x < self.map_size.0 && y < self.map_size.1 {
                    self.occupancy[y as usize * self.map_size.0 as usize + x as usize] = 1;
                }
            }
        }
        profile_end(ProfileMetric::EnvStepOccupancy, profile);

        // Remove the frontiers that the new room connects to (if any),
        // and update the frontier with the new unconnected doors of the new room.
        for door in room.doors.iter() {
            let door_loc = DoorLocation::new(door, action.x, action.y);
            if let Some(frontier) = self.frontier.remove(&door_loc) {
                let profile = profile_start();
                // This frontier is now connected, so remove it and mark the doors as connected:
                let i1 = door.dir_door_idx;
                let i2 = frontier.dir_door_idx;
                if mode.updates_door_matches() {
                    self.door_matches[door.direction as usize][i1 as usize] = i2;
                    self.door_matches[door.direction.opposite() as usize][i2 as usize] = i1;
                }
                let p1 = common.room_dir_door[door.direction as usize][i1 as usize].room_part_idx;
                let p2 = common.room_dir_door[door.direction.opposite() as usize][i2 as usize]
                    .room_part_idx;
                self.room_part_frontier_distance_cache.remove_frontier_part(
                    &self.graph_distance,
                    common.room_part.len(),
                    &self.active_room_parts,
                    frontier.room_part_idx as usize,
                );
                self.add_component_edge(
                    self.room_part_component[p1 as usize],
                    self.room_part_component[p2 as usize],
                );
                self.add_component_edge(
                    self.room_part_component[p2 as usize],
                    self.room_part_component[p1 as usize],
                );
                profile_end(ProfileMetric::EnvStepMatchExistingFrontiers, profile);
            } else {
                // This door is not connected to any existing frontier, so it becomes a new frontier.
                // Check all doors with the given orientation, to list which ones could connect here.
                let mut candidates = vec![];
                if mode.builds_frontier_candidates() {
                    let profile = profile_start();
                    let (x1, y1) = get_behind_door_position(
                        door.direction,
                        action.x + door.x,
                        action.y + door.y,
                    );
                    'door: for opp_door in
                        common.geometry_dir_door[door.direction.opposite() as usize].iter()
                    {
                        if self.geometry_unused_count[opp_door.geometry_idx as usize] == 0 {
                            // A geometry with no unused room representatives cannot be used again.
                            continue;
                        }
                        let room_x = x1 - opp_door.x;
                        let room_y = y1 - opp_door.y;
                        let geometry = &common.geometry[opp_door.geometry_idx as usize];
                        if room_x < -geometry.min_x
                            || room_x > self.map_size.0 - 1 - geometry.max_x
                            || room_y < -geometry.min_y
                            || room_y > self.map_size.1 - 1 - geometry.max_y
                        {
                            // The room cannot be placed at this position due to map boundaries.
                            continue;
                        }

                        if self.candidate_intersects_placed_room(
                            common,
                            opp_door.geometry_idx,
                            room_x,
                            room_y,
                        ) {
                            continue 'door;
                        }

                        // The geometry had no intersections with existing rooms, so it is a valid candidate at this frontier.
                        let candidate = GeometryAction {
                            geometry_idx: opp_door.geometry_idx,
                            x: room_x,
                            y: room_y,
                            door_direction: door.direction.opposite(),
                            door_x: opp_door.x,
                            door_y: opp_door.y,
                            door_kind: opp_door.kind,
                        };
                        candidates.push(candidate);
                        if !mode.stores_full_candidate_lists() {
                            break 'door;
                        }
                    }
                    profile_end(ProfileMetric::EnvStepBuildNewFrontierCandidates, profile);
                }
                let frontier_part = common.room_dir_door[door.direction as usize]
                    [door.dir_door_idx as usize]
                    .room_part_idx;
                let door_output_idx = common.room_dir_door[..door.direction as usize]
                    .iter()
                    .map(Vec::len)
                    .sum::<usize>()
                    + door.dir_door_idx as usize;
                let door_kind =
                    common.room_dir_door[door.direction as usize][door.dir_door_idx as usize].kind;
                let door_variant_idx = common.door_variant_idx(
                    common.room[action.room_idx as usize].connection_variant_idx,
                    door.direction,
                    door.x,
                    door.y,
                    door_kind,
                );
                let frontier = Frontier {
                    direction: door.direction,
                    dir_door_idx: door.dir_door_idx,
                    door_output_idx: door_output_idx as i16,
                    door_variant_idx,
                    room_part_idx: frontier_part,
                    component: self.room_part_component(common, action.room_idx, door.part_idx),
                    kind: door_kind,
                    candidates,
                };
                self.room_part_frontier_distance_cache.add_frontier_part(
                    &self.graph_distance,
                    common.room_part.len(),
                    &self.active_room_parts,
                    frontier_part as usize,
                );
                self.frontier.insert(door_loc, frontier);
            }
        }

        // Filter existing frontiers to remove geometries blocked by the new room or with no unused representatives.
        if mode.filters_existing_frontier_candidates() {
            let profile = profile_start();
            let geometry_unused_count = &self.geometry_unused_count;
            for frontier in self.frontier.values_mut() {
                let keep_candidate = |cand: &GeometryAction| {
                    geometry_unused_count[cand.geometry_idx as usize] > 0
                        && !common.has_geometry_intersection(
                            action_geometry_idx,
                            action.x,
                            action.y,
                            cand.geometry_idx,
                            cand.x,
                            cand.y,
                        )
                };
                if mode.stores_full_candidate_lists() {
                    frontier.candidates.retain(keep_candidate);
                } else if let Some(candidate_idx) =
                    frontier.candidates.iter().position(keep_candidate)
                {
                    let candidate = frontier.candidates[candidate_idx];
                    frontier.candidates.clear();
                    frontier.candidates.push(candidate);
                } else {
                    frontier.candidates.clear();
                }
            }
            profile_end(ProfileMetric::EnvStepFilterExistingFrontiers, profile);
        }
    }

    fn candidate_intersects_placed_room(
        &mut self,
        common: &CommonData,
        candidate_geometry_idx: GeometryIdx,
        candidate_x: Coord,
        candidate_y: Coord,
    ) -> bool {
        self.intersection_query_generation = self.intersection_query_generation.wrapping_add(1);
        if self.intersection_query_generation == 0 {
            self.intersection_query_seen.fill(0);
            self.intersection_query_generation = 1;
        }
        let generation = self.intersection_query_generation;
        self.placed_room_index.query_geometry(
            &common.geometry[candidate_geometry_idx as usize],
            candidate_x,
            candidate_y,
            &mut self.intersection_query_actions,
        );
        for &action_idx in &self.intersection_query_actions {
            let action_idx = action_idx as usize;
            if self.intersection_query_seen[action_idx] == generation {
                continue;
            }
            self.intersection_query_seen[action_idx] = generation;
            let action = self.actions[action_idx];
            let placed_geometry_idx = common.room[action.room_idx as usize].geometry_idx;
            if common.has_geometry_intersection(
                placed_geometry_idx,
                action.x,
                action.y,
                candidate_geometry_idx,
                candidate_x,
                candidate_y,
            ) {
                return true;
            }
        }
        false
    }

    pub fn finish(&mut self) {
        self.finished = true;
    }

    fn add_room_components_and_edges(&mut self, action: Action, common: &CommonData) {
        let room_idx = action.room_idx;
        let room = &common.room[room_idx as usize];
        let mut attached_room_parts = vec![Vec::new(); room.door_group_count];
        let mut external_distance_edges = Vec::new();
        for door in &room.doors {
            let door_loc = DoorLocation::new(door, action.x, action.y);
            if let Some(frontier) = self.frontier.get(&door_loc) {
                let attached_room_part = common.room_dir_door[door.direction.opposite() as usize]
                    [frontier.dir_door_idx as usize]
                    .room_part_idx;
                attached_room_parts[door.part_idx as usize].push(attached_room_part);
                external_distance_edges.push((
                    Self::room_part_idx(common, room_idx, door.part_idx),
                    attached_room_part,
                ));
            }
        }
        self.add_room_part_distances(common, room_idx, &external_distance_edges);

        for (part_idx, attached_parts) in attached_room_parts.iter().enumerate() {
            let room_part_idx = (room.door_group_offset + part_idx) as RoomPartIdx;
            if attached_parts.is_empty() {
                self.room_part_component[room_part_idx as usize] = self.scc_dag.add_component();
                continue;
            }

            let first_attached_part = attached_parts[0];
            for &attached_part in &attached_parts[1..] {
                let from = self.room_part_component[first_attached_part as usize];
                let to = self.room_part_component[attached_part as usize];
                self.add_component_edge(from, to);
                let from = self.room_part_component[attached_part as usize];
                let to = self.room_part_component[first_attached_part as usize];
                self.add_component_edge(from, to);
            }
            self.room_part_component[room_part_idx as usize] =
                self.room_part_component[first_attached_part as usize];
        }
        for door in &room.doors {
            debug_assert_eq!(
                common.room_dir_door[door.direction as usize][door.dir_door_idx as usize]
                    .room_part_idx,
                Self::room_part_idx(common, room_idx, door.part_idx)
            );
        }
        for &(from_part, to_part) in &room.connections {
            let from = self.room_part_component(common, room_idx, from_part);
            let to = self.room_part_component(common, room_idx, to_part);
            self.add_component_edge(from, to);
        }
    }

    fn add_room_part_distances(
        &mut self,
        common: &CommonData,
        room_idx: RoomIdx,
        external_edges: &[(RoomPartIdx, RoomPartIdx)],
    ) {
        let room = &common.room[room_idx as usize];
        let graph_size = common.room_part.len();
        let old_active_room_parts_len = self.active_room_parts.len();
        for local_part in 0..room.door_group_count {
            self.active_room_parts
                .push((room.door_group_offset + local_part) as RoomPartIdx);
        }
        let new_room_part_start = room.door_group_offset as RoomPartIdx;
        let new_room_part_end = new_room_part_start + room.door_group_count as RoomPartIdx;
        if let [(room_part, attached_part)] = external_edges {
            self.add_single_attachment_room_distances(
                common,
                room_idx,
                *room_part,
                *attached_part,
                old_active_room_parts_len,
            );
        } else {
            self.add_room_local_distances(common, room_idx);
            for &(room_part, attached_part) in external_edges {
                self.add_graph_distance_edge(common, room_part, attached_part, 1);
                self.add_graph_distance_edge(common, attached_part, room_part, 1);
            }
        }
        for &room_part in &common.save_room_part {
            if room_part < new_room_part_start || room_part >= new_room_part_end {
                continue;
            }
            self.active_save_room_parts.push(room_part);
            self.room_part_save_distance_cache.add_save_part(
                &self.graph_distance,
                graph_size,
                &self.active_room_parts,
                room_part as usize,
            );
        }
        for &room_part in &common.refill_room_part {
            if room_part < new_room_part_start || room_part >= new_room_part_end {
                continue;
            }
            self.active_refill_room_parts.push(room_part);
            self.room_part_refill_distance_cache.add_save_part(
                &self.graph_distance,
                graph_size,
                &self.active_room_parts,
                room_part as usize,
            );
        }
    }

    fn add_room_local_distances(&mut self, common: &CommonData, room_idx: RoomIdx) {
        let room = &common.room[room_idx as usize];
        let graph_size = common.room_part.len();
        for from_part in 0..room.door_group_count {
            let from_room_part = room.door_group_offset + from_part;
            for to_part in 0..room.door_group_count {
                let to_room_part = room.door_group_offset + to_part;
                self.set_graph_distance(
                    graph_size,
                    from_room_part,
                    to_room_part,
                    room.part_distances[from_part * room.door_group_count + to_part],
                );
            }
        }
    }

    fn add_single_attachment_room_distances(
        &mut self,
        common: &CommonData,
        room_idx: RoomIdx,
        room_part: RoomPartIdx,
        attached_part: RoomPartIdx,
        old_active_room_parts_len: usize,
    ) {
        let room = &common.room[room_idx as usize];
        let graph_size = common.room_part.len();
        let room_start = room.door_group_offset;
        let local_attachment = room_part as usize - room_start;
        let attached_part = attached_part as usize;

        let write_distance = |graph_distance: &mut [GraphDistance],
                              furthest_cache: &mut RoomPartFurthestDistanceCache,
                              from_part: usize,
                              to_part: usize,
                              distance: GraphDistance| {
            let idx = from_part * graph_size + to_part;
            if distance < graph_distance[idx] {
                graph_distance[idx] = distance;
                furthest_cache.add_distance(from_part, to_part, distance);
            }
        };

        for local_from in 0..room.door_group_count {
            let from_part = room_start + local_from;
            for local_to in 0..room.door_group_count {
                let to_part = room_start + local_to;
                write_distance(
                    &mut self.graph_distance,
                    &mut self.room_part_furthest_distance_cache,
                    from_part,
                    to_part,
                    room.part_distances[local_from * room.door_group_count + local_to],
                );
            }
        }

        for local_from in 0..room.door_group_count {
            let from_part = room_start + local_from;
            let to_attachment =
                room.part_distances[local_from * room.door_group_count + local_attachment];
            if to_attachment != UNREACHABLE_DISTANCE {
                for to_part_idx in 0..old_active_room_parts_len {
                    let to_part = self.active_room_parts[to_part_idx] as usize;
                    let old_distance = self.graph_distance[attached_part * graph_size + to_part];
                    if let Some(distance) = graph_distance_sum(&[to_attachment, 1, old_distance]) {
                        write_distance(
                            &mut self.graph_distance,
                            &mut self.room_part_furthest_distance_cache,
                            from_part,
                            to_part,
                            distance,
                        );
                    }
                }
            }

            let from_attachment =
                room.part_distances[local_attachment * room.door_group_count + local_from];
            if from_attachment != UNREACHABLE_DISTANCE {
                for from_old_part_idx in 0..old_active_room_parts_len {
                    let from_old_part = self.active_room_parts[from_old_part_idx] as usize;
                    let old_distance =
                        self.graph_distance[from_old_part * graph_size + attached_part];
                    if let Some(distance) = graph_distance_sum(&[old_distance, 1, from_attachment])
                    {
                        write_distance(
                            &mut self.graph_distance,
                            &mut self.room_part_furthest_distance_cache,
                            from_old_part,
                            from_part,
                            distance,
                        );
                    }
                }
            }
        }

        for local_from in 0..room.door_group_count {
            let from_part = room_start + local_from;
            let to_attachment =
                room.part_distances[local_from * room.door_group_count + local_attachment];
            if to_attachment == UNREACHABLE_DISTANCE {
                continue;
            }
            for local_to in 0..room.door_group_count {
                let to_part = room_start + local_to;
                let from_attachment =
                    room.part_distances[local_attachment * room.door_group_count + local_to];
                if let Some(distance) = graph_distance_sum(&[to_attachment, 2, from_attachment]) {
                    write_distance(
                        &mut self.graph_distance,
                        &mut self.room_part_furthest_distance_cache,
                        from_part,
                        to_part,
                        distance,
                    );
                }
            }
        }

        for local_part in 0..room.door_group_count {
            let part = room_start + local_part;
            let to_attached_part = self.graph_distance[part * graph_size + attached_part];
            let from_attached_part = self.graph_distance[attached_part * graph_size + part];
            self.room_part_save_distance_cache
                .initialize_single_attachment_room_part(
                    part,
                    attached_part,
                    to_attached_part,
                    from_attached_part,
                );
            self.room_part_refill_distance_cache
                .initialize_single_attachment_room_part(
                    part,
                    attached_part,
                    to_attached_part,
                    from_attached_part,
                );
            self.room_part_frontier_distance_cache
                .initialize_single_attachment_room_part(
                    part,
                    attached_part,
                    to_attached_part,
                    from_attached_part,
                );
        }
    }

    fn add_graph_distance_edge(
        &mut self,
        common: &CommonData,
        from_part: RoomPartIdx,
        to_part: RoomPartIdx,
        cost: GraphDistance,
    ) {
        let graph_size = common.room_part.len();
        let from_part = from_part as usize;
        let to_part = to_part as usize;
        let edge_idx = from_part * graph_size + to_part;
        if cost < self.graph_distance[edge_idx] {
            self.set_graph_distance(graph_size, from_part, to_part, cost);
        }

        for source_idx in 0..self.active_room_parts.len() {
            let source = self.active_room_parts[source_idx] as usize;
            let source_distance = self.graph_distance[source * graph_size + from_part];
            if source_distance == UNREACHABLE_DISTANCE {
                continue;
            }
            let Some(prefix_distance) = graph_distance_sum(&[source_distance, cost]) else {
                continue;
            };
            for destination_idx in 0..self.active_room_parts.len() {
                let destination = self.active_room_parts[destination_idx] as usize;
                let destination_distance = self.graph_distance[to_part * graph_size + destination];
                let Some(distance) = graph_distance_sum(&[prefix_distance, destination_distance])
                else {
                    continue;
                };
                self.set_graph_distance_min(graph_size, source, destination, distance);
            }
        }
    }

    fn set_graph_distance_min(
        &mut self,
        graph_size: usize,
        from_part: usize,
        to_part: usize,
        distance: GraphDistance,
    ) {
        let idx = from_part * graph_size + to_part;
        if distance < self.graph_distance[idx] {
            self.set_graph_distance(graph_size, from_part, to_part, distance);
        }
    }

    fn set_graph_distance(
        &mut self,
        graph_size: usize,
        from_part: usize,
        to_part: usize,
        distance: GraphDistance,
    ) {
        let idx = from_part * graph_size + to_part;
        let old_distance = self.graph_distance[idx];
        if old_distance == distance {
            return;
        }
        self.graph_distance[idx] = distance;
        self.room_part_furthest_distance_cache.set_distance(
            &self.graph_distance,
            graph_size,
            from_part,
            to_part,
            old_distance,
            distance,
        );
        self.room_part_save_distance_cache.set_distance(
            &self.graph_distance,
            graph_size,
            from_part,
            to_part,
            old_distance,
            distance,
        );
        self.room_part_refill_distance_cache.set_distance(
            &self.graph_distance,
            graph_size,
            from_part,
            to_part,
            old_distance,
            distance,
        );
        self.room_part_frontier_distance_cache.set_distance(
            &self.graph_distance,
            graph_size,
            from_part,
            to_part,
            old_distance,
            distance,
        );
    }

    fn graph_distance_snapshot_for_candidate(
        &self,
        common: &CommonData,
        candidate: Action,
    ) -> GraphDistanceSnapshot {
        if self.finished || candidate.room_idx >= common.room.len() as RoomIdx {
            return GraphDistanceSnapshot::None;
        }
        let room = &common.room[candidate.room_idx as usize];
        let external_edge_count = room
            .doors
            .iter()
            .filter(|door| {
                self.frontier
                    .contains_key(&DoorLocation::new(door, candidate.x, candidate.y))
            })
            .count();
        if external_edge_count >= 2 {
            GraphDistanceSnapshot::Full {
                graph_distance: self.graph_distance.clone(),
                furthest_distance_cache: self.room_part_furthest_distance_cache.clone(),
                save_distance_cache: self.room_part_save_distance_cache.clone(),
                refill_distance_cache: self.room_part_refill_distance_cache.clone(),
            }
        } else {
            GraphDistanceSnapshot::NewRoom {
                room_idx: candidate.room_idx,
                furthest_distance_cache: self.room_part_furthest_distance_cache.clone(),
                save_distance_cache: self.room_part_save_distance_cache.clone(),
                refill_distance_cache: self.room_part_refill_distance_cache.clone(),
            }
        }
    }

    fn restore_graph_distance_snapshot(
        &mut self,
        common: &CommonData,
        snapshot: GraphDistanceSnapshot,
    ) {
        match snapshot {
            GraphDistanceSnapshot::None => {}
            GraphDistanceSnapshot::NewRoom {
                room_idx,
                furthest_distance_cache,
                save_distance_cache,
                refill_distance_cache,
            } => {
                self.clear_room_graph_distances(common, room_idx);
                self.room_part_furthest_distance_cache = furthest_distance_cache;
                self.room_part_save_distance_cache = save_distance_cache;
                self.room_part_refill_distance_cache = refill_distance_cache;
            }
            GraphDistanceSnapshot::Full {
                graph_distance,
                furthest_distance_cache,
                save_distance_cache,
                refill_distance_cache,
            } => {
                self.graph_distance = graph_distance;
                self.room_part_furthest_distance_cache = furthest_distance_cache;
                self.room_part_save_distance_cache = save_distance_cache;
                self.room_part_refill_distance_cache = refill_distance_cache;
            }
        }
    }

    fn clear_room_graph_distances(&mut self, common: &CommonData, room_idx: RoomIdx) {
        let room = &common.room[room_idx as usize];
        let graph_size = common.room_part.len();
        for local_part in 0..room.door_group_count {
            let room_part = room.door_group_offset + local_part;
            for &active_part in &self.active_room_parts {
                let active_part = active_part as usize;
                self.graph_distance[room_part * graph_size + active_part] = UNREACHABLE_DISTANCE;
                self.graph_distance[active_part * graph_size + room_part] = UNREACHABLE_DISTANCE;
            }
            for other_local_part in 0..room.door_group_count {
                let other_part = room.door_group_offset + other_local_part;
                self.graph_distance[room_part * graph_size + other_part] = UNREACHABLE_DISTANCE;
                self.graph_distance[other_part * graph_size + room_part] = UNREACHABLE_DISTANCE;
            }
        }
    }

    #[cfg(test)]
    fn graph_distance(
        &self,
        common: &CommonData,
        from_part: RoomPartIdx,
        to_part: RoomPartIdx,
    ) -> GraphDistance {
        let graph_size = common.room_part.len();
        self.graph_distance[from_part as usize * graph_size + to_part as usize]
    }

    fn add_component_edge(&mut self, from_component: usize, to_component: usize) {
        debug_assert_ne!(from_component, NO_COMPONENT);
        debug_assert_ne!(to_component, NO_COMPONENT);
        if let Some(component_merge) = self.scc_dag.add_edge(from_component, to_component) {
            for component in &mut self.room_part_component {
                if *component != NO_COMPONENT {
                    *component = component_merge.component_remap[*component];
                }
            }
            for frontier in self.frontier.values_mut() {
                frontier.component = component_merge.component_remap[frontier.component];
            }
        }
    }

    fn room_part_idx(common: &CommonData, room_idx: RoomIdx, part_idx: PartIdx) -> RoomPartIdx {
        (common.room[room_idx as usize].door_group_offset + part_idx as usize) as RoomPartIdx
    }

    fn room_part_component(
        &self,
        common: &CommonData,
        room_idx: RoomIdx,
        part_idx: PartIdx,
    ) -> usize {
        let component =
            self.room_part_component[Self::room_part_idx(common, room_idx, part_idx) as usize];
        debug_assert_ne!(component, NO_COMPONENT);
        component
    }

    fn scc_dag_with_merged_frontiers(&self) -> (SccDag, Vec<usize>) {
        let mut scc_dag = self.scc_dag.clone();
        let mut component_remap = (0..self.scc_dag.component_count).collect::<Vec<_>>();
        let mut frontier_components = self
            .frontier
            .values()
            .filter(|frontier| !frontier.candidates.is_empty())
            .map(|frontier| frontier.component)
            .collect::<Vec<_>>();
        frontier_components.sort_unstable();
        frontier_components.dedup();

        if frontier_components.len() >= 2 {
            component_remap = scc_dag
                .merge_components(&frontier_components)
                .component_remap;
        }

        (scc_dag, component_remap)
    }

    pub fn get_proposal_candidates_with_outcomes(
        &mut self,
        common: &CommonData,
        sampled_frontier_idx: &[FrontierIdx],
        sampled_door_variant_idx: &[DoorVariantIdx],
        recommended_candidates: usize,
        config: &FeatureConfig,
        frontier_neighbor_algorithm: FrontierNeighborAlgorithm,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
        scratch: &mut FeatureScratch,
    ) -> Result<
        (
            StepOutcomes,
            Vec<Action>,
            Vec<FrontierIdx>,
            Vec<DoorVariantIdx>,
            Vec<StepOutcomes>,
            Vec<Vec<i16>>,
            Vec<FeaturePlan>,
            usize,
            usize,
        ),
        String,
    > {
        debug_assert_eq!(sampled_frontier_idx.len(), sampled_door_variant_idx.len());
        record_profile_count(ProfileMetric::EnvCounterProposalCalls, 1);
        record_profile_count(
            ProfileMetric::EnvCounterProposalShortlistCandidates,
            sampled_frontier_idx.len() as u64,
        );
        let profile = profile_start();
        let pre_candidate_outcomes = self.outcomes(common);
        profile_end(ProfileMetric::EnvProposalPreOutcomes, profile);

        let profile = profile_start();
        let sorted_frontier_locations = self.sorted_frontier_locations();
        profile_end(ProfileMetric::EnvProposalSortFrontiers, profile);

        let mut clean = Vec::with_capacity(recommended_candidates);
        let mut rejected = Vec::new();
        let mut evaluated_count = 0;
        let mut rejected_count = 0;
        for (&frontier_idx, &door_variant_idx) in
            sampled_frontier_idx.iter().zip(sampled_door_variant_idx)
        {
            if clean.len() == recommended_candidates {
                break;
            }
            let profile = profile_start();
            let Some(action) = self.action_for_proposal_candidate(
                common,
                &sorted_frontier_locations,
                frontier_idx,
                door_variant_idx,
            ) else {
                profile_end(ProfileMetric::EnvProposalResolveAction, profile);
                continue;
            };
            profile_end(ProfileMetric::EnvProposalResolveAction, profile);
            evaluated_count += 1;
            match self.evaluate_candidate_outcome(
                common,
                &pre_candidate_outcomes,
                action,
                config,
                frontier_neighbor_algorithm,
                frontier_neighbor_count,
                frontier_window_size,
                scratch,
            )? {
                CandidateOutcome::Rejected => {
                    rejected_count += 1;
                    rejected.push(CandidateAction {
                        action,
                        frontier_idx,
                        door_variant_idx,
                    });
                }
                CandidateOutcome::Clean(post_candidate_outcomes, door_match, features) => {
                    clean.push((
                        CandidateAction {
                            action,
                            frontier_idx,
                            door_variant_idx,
                        },
                        post_candidate_outcomes,
                        door_match,
                        features,
                    ));
                }
            }
        }

        let clean_count = clean.len();
        let candidates_with_outcomes = if clean.is_empty() && !rejected.is_empty() {
            let profile = profile_start();
            let fallback = rejected
                .into_iter()
                .take(recommended_candidates)
                .map(|candidate| {
                    let (post_candidate_outcomes, door_match, features) = self
                        .outcomes_and_features_after_candidate(
                            common,
                            candidate.action,
                            config,
                            frontier_neighbor_algorithm,
                            frontier_neighbor_count,
                            frontier_window_size,
                            scratch,
                        );
                    (candidate, post_candidate_outcomes, door_match, features)
                })
                .collect::<Vec<_>>();
            profile_end(ProfileMetric::EnvProposalFallbackRecompute, profile);
            record_profile_count(
                ProfileMetric::EnvCounterProposalFallbackCandidates,
                fallback.len() as u64,
            );
            fallback
        } else {
            clean
        };
        let output_candidate_count = candidates_with_outcomes.len();
        record_profile_count(
            ProfileMetric::EnvCounterProposalEvaluatedCandidates,
            evaluated_count as u64,
        );
        record_profile_count(
            ProfileMetric::EnvCounterProposalRejectedCandidates,
            rejected_count as u64,
        );
        record_profile_count(
            ProfileMetric::EnvCounterProposalCleanCandidates,
            clean_count as u64,
        );
        record_profile_count(
            ProfileMetric::EnvCounterProposalOutputCandidates,
            output_candidate_count as u64,
        );

        let mut candidates = Vec::with_capacity(candidates_with_outcomes.len());
        let mut proposal_frontier_idx = Vec::with_capacity(candidates_with_outcomes.len());
        let mut proposal_door_variant_idx = Vec::with_capacity(candidates_with_outcomes.len());
        let mut post_candidate_outcomes = Vec::with_capacity(candidates_with_outcomes.len());
        let mut door_matches = Vec::with_capacity(candidates_with_outcomes.len());
        let mut features = Vec::with_capacity(candidates_with_outcomes.len());
        for (candidate, outcomes, door_match, candidate_features) in candidates_with_outcomes {
            candidates.push(candidate.action);
            proposal_frontier_idx.push(candidate.frontier_idx);
            proposal_door_variant_idx.push(candidate.door_variant_idx);
            post_candidate_outcomes.push(outcomes);
            door_matches.push(door_match);
            features.push(candidate_features);
        }
        Ok((
            pre_candidate_outcomes,
            candidates,
            proposal_frontier_idx,
            proposal_door_variant_idx,
            post_candidate_outcomes,
            door_matches,
            features,
            evaluated_count,
            rejected_count,
        ))
    }

    fn evaluate_candidate_outcome(
        &mut self,
        common: &CommonData,
        pre_candidate_outcomes: &StepOutcomes,
        candidate: Action,
        config: &FeatureConfig,
        frontier_neighbor_algorithm: FrontierNeighborAlgorithm,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
        scratch: &mut FeatureScratch,
    ) -> Result<CandidateOutcome, String> {
        let profile = profile_start();
        let snapshot = self.apply_lookahead_candidate(candidate, common);
        profile_end(ProfileMetric::EnvProposalApplyLookahead, profile);

        let profile = profile_start();
        let mut door_valid = Vec::with_capacity(pre_candidate_outcomes.door_valid.len());
        let mut outcome_idx = 0;
        for dir in 0..NUM_DIRS {
            for i in 0..common.room_dir_door[dir].len() {
                let before = pre_candidate_outcomes.door_valid[outcome_idx];
                if before == DoorValidOutcome::Unknown {
                    let after = self.door_outcome(common, dir, i);
                    if after == DoorValidOutcome::Invalid {
                        profile_end(ProfileMetric::EnvProposalDoorOutcomes, profile);
                        let profile = profile_start();
                        self.restore_lookahead_candidate(common, snapshot);
                        profile_end(ProfileMetric::EnvProposalRestore, profile);
                        return Ok(CandidateOutcome::Rejected);
                    }
                    door_valid.push(after);
                } else {
                    door_valid.push(before);
                }
                outcome_idx += 1;
            }
        }
        profile_end(ProfileMetric::EnvProposalDoorOutcomes, profile);

        let profile = profile_start();
        let frontier_reachability = if self.finished {
            None
        } else {
            Some(self.scc_dag_with_merged_frontiers())
        };
        let mut connections_valid =
            Vec::with_capacity(pre_candidate_outcomes.connections_valid.len());
        for connection_idx in 0..common.room_connection.len() {
            let before = pre_candidate_outcomes.connections_valid[connection_idx];
            if before == DoorValidOutcome::Unknown {
                let after =
                    self.connection_outcome(common, connection_idx, frontier_reachability.as_ref());
                if after == DoorValidOutcome::Invalid {
                    profile_end(ProfileMetric::EnvProposalConnectionOutcomes, profile);
                    let profile = profile_start();
                    self.restore_lookahead_candidate(common, snapshot);
                    profile_end(ProfileMetric::EnvProposalRestore, profile);
                    return Ok(CandidateOutcome::Rejected);
                }
                connections_valid.push(after);
            } else {
                connections_valid.push(before);
            }
        }
        profile_end(ProfileMetric::EnvProposalConnectionOutcomes, profile);

        let toilet_valid = if pre_candidate_outcomes.toilet_valid == DoorValidOutcome::Unknown {
            let after = self.toilet_outcome(common);
            if after == DoorValidOutcome::Invalid {
                let profile = profile_start();
                self.restore_lookahead_candidate(common, snapshot);
                profile_end(ProfileMetric::EnvProposalRestore, profile);
                return Ok(CandidateOutcome::Rejected);
            }
            after
        } else {
            pre_candidate_outcomes.toilet_valid
        };

        let phantoon_valid = if pre_candidate_outcomes.phantoon_valid == DoorValidOutcome::Unknown {
            let after = self.phantoon_outcome(common);
            if after == DoorValidOutcome::Invalid {
                let profile = profile_start();
                self.restore_lookahead_candidate(common, snapshot);
                profile_end(ProfileMetric::EnvProposalRestore, profile);
                return Ok(CandidateOutcome::Rejected);
            }
            after
        } else {
            pre_candidate_outcomes.phantoon_valid
        };

        let profile = profile_start();
        let features = self.feature_plan_for_applied_candidate(
            common,
            candidate,
            config,
            frontier_neighbor_algorithm,
            frontier_neighbor_count,
            frontier_window_size,
            scratch,
        );
        profile_end(ProfileMetric::EnvProposalFeatures, profile);
        let step_outcomes = StepOutcomes {
            door_valid: door_valid.clone(),
            connections_valid: connections_valid.clone(),
            toilet_valid,
            phantoon_valid,
            toilet_crossed_room_idx: self.toilet_crossed_room_idx(common),
        };
        let profile = profile_start();
        let door_match = self.door_match_feature(common, &step_outcomes);
        profile_end(ProfileMetric::EnvProposalDoorMatch, profile);
        let profile = profile_start();
        self.restore_lookahead_candidate(common, snapshot);
        profile_end(ProfileMetric::EnvProposalRestore, profile);
        Ok(CandidateOutcome::Clean(step_outcomes, door_match, features))
    }

    fn outcomes_and_features_after_candidate(
        &mut self,
        common: &CommonData,
        candidate: Action,
        config: &FeatureConfig,
        frontier_neighbor_algorithm: FrontierNeighborAlgorithm,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
        scratch: &mut FeatureScratch,
    ) -> (StepOutcomes, Vec<i16>, FeaturePlan) {
        let profile = profile_start();
        let snapshot = self.apply_lookahead_candidate(candidate, common);
        profile_end(ProfileMetric::EnvProposalApplyLookahead, profile);
        let feature_outcomes = self.feature_outcomes(common);
        let profile = profile_start();
        let features = self.feature_plan_for_applied_candidate(
            common,
            candidate,
            config,
            frontier_neighbor_algorithm,
            frontier_neighbor_count,
            frontier_window_size,
            scratch,
        );
        profile_end(ProfileMetric::EnvProposalFeatures, profile);
        let profile = profile_start();
        self.restore_lookahead_candidate(common, snapshot);
        profile_end(ProfileMetric::EnvProposalRestore, profile);
        (
            feature_outcomes.step_outcomes,
            feature_outcomes.door_match,
            features,
        )
    }

    #[cfg(test)]
    fn outcomes_after_candidate(
        &mut self,
        common: &CommonData,
        candidate: Action,
    ) -> FeatureOutcomes {
        let snapshot = self.apply_lookahead_candidate(candidate, common);
        let feature_outcomes = self.feature_outcomes(common);
        self.restore_lookahead_candidate(common, snapshot);
        feature_outcomes
    }

    pub fn feature_outcomes(&self, common: &CommonData) -> FeatureOutcomes {
        let step_outcomes = self.outcomes(common);
        let profile = profile_start();
        let door_match = self.door_match_feature(common, &step_outcomes);
        profile_end(ProfileMetric::EnvProposalDoorMatch, profile);
        FeatureOutcomes {
            step_outcomes,
            door_match,
        }
    }

    fn door_match_feature(&self, common: &CommonData, outcomes: &StepOutcomes) -> Vec<i16> {
        let mut result = Vec::with_capacity(outcomes.door_valid.len());
        let mut outcome_idx = 0;
        for dir in 0..NUM_DIRS {
            let opposite_dir = match dir {
                0 => Direction::Right as usize,
                1 => Direction::Left as usize,
                2 => Direction::Down as usize,
                3 => Direction::Up as usize,
                _ => unreachable!(),
            };
            let invalid_sentinel = common.room_dir_door[opposite_dir].len() as i16;
            for door_idx in 0..common.room_dir_door[dir].len() {
                result.push(match outcomes.door_valid[outcome_idx] {
                    DoorValidOutcome::Unknown => -1,
                    DoorValidOutcome::Valid => self.door_matches[dir][door_idx] as i16,
                    DoorValidOutcome::Invalid => invalid_sentinel,
                });
                outcome_idx += 1;
            }
        }
        result
    }

    fn feature_plan_for_applied_candidate(
        &self,
        common: &CommonData,
        candidate: Action,
        config: &FeatureConfig,
        frontier_neighbor_algorithm: FrontierNeighborAlgorithm,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
        scratch: &mut FeatureScratch,
    ) -> FeaturePlan {
        if config.is_empty() {
            return scratch.take_plan();
        }
        let extra_occupied = if candidate.room_idx < common.room.len() as RoomIdx {
            let geometry_idx = common.room[candidate.room_idx as usize].geometry_idx;
            Some(FeatureExtraOccupied {
                geometry_idx,
                x: candidate.x,
                y: candidate.y,
            })
        } else {
            None
        };
        self.feature_plan_with_occupancy(
            common,
            config,
            extra_occupied,
            FeaturePlanKind::Candidate(candidate),
            frontier_neighbor_algorithm,
            frontier_neighbor_count,
            frontier_window_size,
            scratch,
        )
    }

    fn apply_lookahead_candidate(
        &mut self,
        candidate: Action,
        common: &CommonData,
    ) -> LookaheadSnapshot {
        let profile = profile_start();
        let room_idx =
            (candidate.room_idx < common.room.len() as RoomIdx).then_some(candidate.room_idx);
        let geometry_idx = room_idx.map(|room_idx| common.room[room_idx as usize].geometry_idx);
        let connection_variant_idx =
            room_idx.map(|room_idx| common.room[room_idx as usize].connection_variant_idx);
        let mut door_matches = Vec::new();
        let graph_distance_snapshot = self.graph_distance_snapshot_for_candidate(common, candidate);
        if !self.finished {
            if let Some(room_idx) = room_idx {
                for door in &common.room[room_idx as usize].doors {
                    let door_loc = DoorLocation::new(door, candidate.x, candidate.y);
                    if let Some(frontier) = self.frontier.get(&door_loc) {
                        let dir = door.direction as usize;
                        let opposite_dir = door.direction.opposite() as usize;
                        door_matches.push((
                            dir,
                            door.dir_door_idx as usize,
                            self.door_matches[dir][door.dir_door_idx as usize],
                        ));
                        door_matches.push((
                            opposite_dir,
                            frontier.dir_door_idx as usize,
                            self.door_matches[opposite_dir][frontier.dir_door_idx as usize],
                        ));
                    }
                }
            }
        }
        let snapshot = LookaheadSnapshot {
            action_len: self.actions.len(),
            finished: self.finished,
            frontier: self.frontier.clone(),
            room_idx,
            room_used: room_idx.is_some_and(|room_idx| self.room_used[room_idx as usize]),
            room_x: room_idx.map_or(0, |room_idx| self.room_x[room_idx as usize]),
            room_y: room_idx.map_or(0, |room_idx| self.room_y[room_idx as usize]),
            geometry_idx,
            geometry_unused_count: geometry_idx
                .map_or(0, |idx| self.geometry_unused_count[idx as usize]),
            connection_variant_idx,
            connection_variant_unused_count: connection_variant_idx
                .map_or(0, |idx| self.connection_variant_unused_count[idx as usize]),
            door_matches,
            room_part_component: self.room_part_component.clone(),
            scc_dag: self.scc_dag.clone(),
            active_room_parts_len: self.active_room_parts.len(),
            active_save_room_parts_len: self.active_save_room_parts.len(),
            active_refill_room_parts_len: self.active_refill_room_parts.len(),
            graph_distance_snapshot,
            room_part_frontier_distance_cache: self.room_part_frontier_distance_cache.clone(),
            placed_room_index_len: self.placed_room_index.insertion_len(),
        };
        profile_end(ProfileMetric::EnvLookaheadSnapshot, profile);
        let profile = profile_start();
        self.step_for_lookahead(candidate, common);
        profile_end(ProfileMetric::EnvLookaheadStep, profile);
        snapshot
    }

    fn restore_lookahead_candidate(&mut self, common: &CommonData, snapshot: LookaheadSnapshot) {
        self.actions.truncate(snapshot.action_len);
        self.finished = snapshot.finished;
        self.frontier = snapshot.frontier;
        if let Some(room_idx) = snapshot.room_idx {
            self.room_used.set(room_idx as usize, snapshot.room_used);
            self.room_x[room_idx as usize] = snapshot.room_x;
            self.room_y[room_idx as usize] = snapshot.room_y;
        }
        if let Some(geometry_idx) = snapshot.geometry_idx {
            self.geometry_unused_count[geometry_idx as usize] = snapshot.geometry_unused_count;
        }
        if let Some(connection_variant_idx) = snapshot.connection_variant_idx {
            self.connection_variant_unused_count[connection_variant_idx as usize] =
                snapshot.connection_variant_unused_count;
        }
        for (dir, idx, value) in snapshot.door_matches {
            self.door_matches[dir][idx] = value;
        }
        self.room_part_component = snapshot.room_part_component;
        self.scc_dag = snapshot.scc_dag;
        self.active_room_parts
            .truncate(snapshot.active_room_parts_len);
        self.active_save_room_parts
            .truncate(snapshot.active_save_room_parts_len);
        self.active_refill_room_parts
            .truncate(snapshot.active_refill_room_parts_len);
        self.restore_graph_distance_snapshot(common, snapshot.graph_distance_snapshot);
        self.room_part_frontier_distance_cache = snapshot.room_part_frontier_distance_cache;
        self.placed_room_index
            .truncate_insertions(snapshot.placed_room_index_len);
    }

    fn apply_feature_candidate(
        &mut self,
        candidate: Action,
        common: &CommonData,
    ) -> FeatureSnapshot {
        let graph_distance_snapshot = self.graph_distance_snapshot_for_candidate(common, candidate);
        let frontier = std::mem::take(&mut self.frontier);
        self.frontier = frontier
            .iter()
            .map(|(&location, frontier)| {
                (
                    location,
                    Frontier {
                        direction: frontier.direction,
                        dir_door_idx: frontier.dir_door_idx,
                        door_output_idx: frontier.door_output_idx,
                        door_variant_idx: frontier.door_variant_idx,
                        room_part_idx: frontier.room_part_idx,
                        component: frontier.component,
                        kind: frontier.kind,
                        candidates: vec![],
                    },
                )
            })
            .collect();
        let room_idx =
            (candidate.room_idx < common.room.len() as RoomIdx).then_some(candidate.room_idx);
        let connection_variant_idx =
            room_idx.map(|room_idx| common.room[room_idx as usize].connection_variant_idx);
        let snapshot = FeatureSnapshot {
            finished: self.finished,
            frontier,
            connection_variant_idx,
            connection_variant_unused_count: connection_variant_idx
                .map_or(0, |idx| self.connection_variant_unused_count[idx as usize]),
            room_part_component: self.room_part_component.clone(),
            scc_dag: self.scc_dag.clone(),
            active_room_parts_len: self.active_room_parts.len(),
            active_save_room_parts_len: self.active_save_room_parts.len(),
            active_refill_room_parts_len: self.active_refill_room_parts.len(),
            graph_distance_snapshot,
            room_part_frontier_distance_cache: self.room_part_frontier_distance_cache.clone(),
        };
        self.step_for_features(candidate, common);
        snapshot
    }

    fn restore_feature_candidate(
        &mut self,
        common: &CommonData,
        candidate: Action,
        snapshot: FeatureSnapshot,
    ) {
        self.finished = snapshot.finished;
        self.frontier = snapshot.frontier;
        if candidate.room_idx < self.room_used.len() as RoomIdx {
            // Coordinates for unused rooms are intentionally left unspecified.
            self.room_used.set(candidate.room_idx as usize, false);
        }
        if let Some(connection_variant_idx) = snapshot.connection_variant_idx {
            self.connection_variant_unused_count[connection_variant_idx as usize] =
                snapshot.connection_variant_unused_count;
        }
        self.room_part_component = snapshot.room_part_component;
        self.scc_dag = snapshot.scc_dag;
        self.active_room_parts
            .truncate(snapshot.active_room_parts_len);
        self.active_save_room_parts
            .truncate(snapshot.active_save_room_parts_len);
        self.active_refill_room_parts
            .truncate(snapshot.active_refill_room_parts_len);
        self.restore_graph_distance_snapshot(common, snapshot.graph_distance_snapshot);
        self.room_part_frontier_distance_cache = snapshot.room_part_frontier_distance_cache;
    }

    #[cfg(test)]
    pub fn feature_frontier_count_after_candidate(
        &self,
        candidate: Action,
        common: &CommonData,
    ) -> usize {
        if self.finished || candidate.room_idx >= common.room.len() as RoomIdx {
            return self.frontier.len();
        }
        let mut frontier_count = self.frontier.len();
        let mut toggled_locations = hashbrown::HashSet::new();
        for door in &common.room[candidate.room_idx as usize].doors {
            let location = DoorLocation::new(door, candidate.x, candidate.y);
            let contains =
                self.frontier.contains_key(&location) ^ toggled_locations.contains(&location);
            if contains {
                frontier_count -= 1;
            } else {
                frontier_count += 1;
            }
            if !toggled_locations.insert(location) {
                toggled_locations.remove(&location);
            }
        }
        frontier_count
    }

    pub fn max_frontiers(common: &CommonData) -> usize {
        common
            .room
            .iter()
            .map(|room| room.doors.len().saturating_sub(2))
            .sum::<usize>()
            + 2
    }

    #[cfg(test)]
    pub fn features(
        &self,
        common: &CommonData,
        config: &FeatureConfig,
        frontier_neighbor_algorithm: FrontierNeighborAlgorithm,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
    ) -> Features {
        let mut scratch = FeatureScratch::default();
        self.features_with_scratch(
            common,
            config,
            frontier_neighbor_algorithm,
            frontier_neighbor_count,
            frontier_window_size,
            &mut scratch,
        )
    }

    #[cfg(test)]
    pub fn features_with_scratch(
        &self,
        common: &CommonData,
        config: &FeatureConfig,
        frontier_neighbor_algorithm: FrontierNeighborAlgorithm,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
        scratch: &mut FeatureScratch,
    ) -> Features {
        self.features_with_occupancy(
            common,
            config,
            &self.occupancy,
            None,
            frontier_neighbor_algorithm,
            frontier_neighbor_count,
            frontier_window_size,
            scratch,
        )
    }

    pub fn feature_plan_with_scratch(
        &self,
        common: &CommonData,
        config: &FeatureConfig,
        frontier_neighbor_algorithm: FrontierNeighborAlgorithm,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
        scratch: &mut FeatureScratch,
    ) -> FeaturePlan {
        self.feature_plan_with_occupancy(
            common,
            config,
            None,
            FeaturePlanKind::Current,
            frontier_neighbor_algorithm,
            frontier_neighbor_count,
            frontier_window_size,
            scratch,
        )
    }

    fn feature_plan_with_occupancy(
        &self,
        common: &CommonData,
        config: &FeatureConfig,
        extra_occupied: Option<FeatureExtraOccupied>,
        kind: FeaturePlanKind,
        _frontier_neighbor_algorithm: FrontierNeighborAlgorithm,
        _frontier_neighbor_count: usize,
        _frontier_window_size: usize,
        scratch: &mut FeatureScratch,
    ) -> FeaturePlan {
        assert!(self.frontier.len() <= Self::max_frontiers(common));
        let mut plan = scratch.take_plan();
        plan.kind = kind;
        plan.extra_occupied = extra_occupied;
        plan.scc_dag = self.scc_dag.clone();
        let profile = profile_start();
        let frontier_count = if config.has_frontier_features() {
            self.frontier.len()
        } else {
            0
        };
        record_profile_count(ProfileMetric::EnvCounterFeatureCalls, 1);
        record_profile_count(
            ProfileMetric::EnvCounterFeatureFrontiers,
            frontier_count as u64,
        );
        if config.room_part_furthest_distance {
            self.room_part_furthest_distance_features_into(
                common,
                &mut plan.room_part_furthest_destination,
                &mut plan.room_part_furthest_source,
            );
        }
        if config.room_part_save_distance {
            self.room_part_save_distance_features_into(
                common,
                &mut plan.room_part_save_from_room_distance,
                &mut plan.room_part_save_to_room_distance,
            );
        }
        if config.room_part_refill_distance {
            self.room_part_refill_distance_features_into(
                common,
                &mut plan.room_part_refill_from_room_distance,
                &mut plan.room_part_refill_to_room_distance,
            );
        }
        if config.room_part_frontier_distance {
            self.room_part_frontier_distance_features_into(
                common,
                &mut plan.room_part_frontier_from_room_distance,
                &mut plan.room_part_frontier_to_room_distance,
            );
        }
        self.known_save_refill_distance_features_into(
            common,
            &self.room_part_save_distance_cache,
            &mut plan.known_save_from_room_distance,
            &mut plan.known_save_to_room_distance,
        );
        self.known_save_refill_distance_features_into(
            common,
            &self.room_part_refill_distance_cache,
            &mut plan.known_refill_from_room_distance,
            &mut plan.known_refill_to_room_distance,
        );
        if config.connection_reachability {
            plan.connection_reachability
                .resize(common.room_connection.len(), 0);
        }
        if config.frontier_connection_reachability {
            plan.frontier_connection_reachability
                .resize(frontier_count * common.room_connection.len(), 0);
        }
        profile_end(ProfileMetric::EnvFeaturesSetup, profile);

        let profile = profile_start();
        let mut sorted_frontiers = if config.has_frontier_features() {
            self.frontier.iter().collect::<Vec<_>>()
        } else {
            vec![]
        };
        sorted_frontiers.sort_unstable_by_key(|(location, _)| **location);
        profile_end(ProfileMetric::EnvFeaturesSortFrontiers, profile);

        let graph_size = common.room_part.len();
        let mut first_frontier_row_by_part = vec![-1; graph_size];
        for (idx, (location, data)) in sorted_frontiers.iter().enumerate() {
            let frontier_part = data.room_part_idx as usize;
            if first_frontier_row_by_part[frontier_part] < 0 {
                first_frontier_row_by_part[frontier_part] = idx as i16;
            }
            plan.frontiers.push(FeatureFrontierPlanRow {
                location: **location,
                door_variant_idx: data.door_variant_idx,
                row_door_output_idx: data.door_output_idx,
                component: data.component,
                kind: data.kind,
                room_part_idx: data.room_part_idx,
            });
        }

        let profile = profile_start();
        let detailed_connection_profile = profile_enabled();
        let mut base_profile_duration = Duration::ZERO;
        let mut frontier_profile_duration = Duration::ZERO;
        let mut missing_connect_profile_duration = Duration::ZERO;
        let mut used_connection_count = 0usize;
        for (connection_idx, connection) in common.room_connection.iter().enumerate() {
            if !self.room_used[connection.room_idx as usize] {
                continue;
            }
            used_connection_count += 1;
            let from_component =
                self.room_part_component(common, connection.room_idx, connection.from_part);
            let to_component =
                self.room_part_component(common, connection.room_idx, connection.to_part);
            let detail_start = detailed_connection_profile.then(Instant::now);
            let already_reachable = self.scc_dag.can_reach(from_component, to_component);
            if config.connection_reachability && already_reachable {
                plan.connection_reachability[connection_idx] = 1;
            }
            if let Some(start) = detail_start {
                base_profile_duration += start.elapsed();
            }
            if config.frontier_connection_reachability {
                let detail_start = detailed_connection_profile.then(Instant::now);
                for (frontier_idx, frontier) in plan.frontiers.iter().enumerate() {
                    let mut flags = 0;
                    if self.scc_dag.can_reach(from_component, frontier.component) {
                        flags |= 1;
                    }
                    if self.scc_dag.can_reach(frontier.component, to_component) {
                        flags |= 2;
                    }
                    plan.frontier_connection_reachability
                        [frontier_idx * common.room_connection.len() + connection_idx] = flags;
                }
                if let Some(start) = detail_start {
                    frontier_profile_duration += start.elapsed();
                }
            }
            if config.missing_connect_query {
                let detail_start = detailed_connection_profile.then(Instant::now);
                let from_part =
                    Self::room_part_idx(common, connection.room_idx, connection.from_part) as usize;
                let to_part =
                    Self::room_part_idx(common, connection.room_idx, connection.to_part) as usize;
                let source_distance = self
                    .room_part_frontier_distance_cache
                    .nearest_frontier_destination[from_part];
                let source_frontier_part = self
                    .room_part_frontier_distance_cache
                    .nearest_frontier_destination_part[from_part];
                let source_frontier = if source_frontier_part == RoomPartIdx::MAX {
                    -1
                } else {
                    first_frontier_row_by_part[source_frontier_part as usize]
                };
                let target_distance = self
                    .room_part_frontier_distance_cache
                    .nearest_frontier_source[to_part];
                let target_frontier_part = self
                    .room_part_frontier_distance_cache
                    .nearest_frontier_source_part[to_part];
                let target_frontier = if target_frontier_part == RoomPartIdx::MAX {
                    -1
                } else {
                    first_frontier_row_by_part[target_frontier_part as usize]
                };
                let current_distance = self.graph_distance[from_part * graph_size + to_part];
                let pair_total_distance = if source_frontier >= 0 && target_frontier >= 0 {
                    u16::from(source_distance) + u16::from(target_distance)
                } else {
                    0
                };
                let pair_can_improve = source_frontier >= 0
                    && target_frontier >= 0
                    && (current_distance == UNREACHABLE_DISTANCE
                        || pair_total_distance + 2 < u16::from(current_distance));
                let emit_missing_connect_query = source_frontier >= 0
                    && target_frontier >= 0
                    && (!already_reachable || pair_can_improve);
                if emit_missing_connect_query {
                    plan.missing_connect_query_connection_idx
                        .push(connection_idx as i64);
                    plan.missing_connect_query_current_distance
                        .push(current_distance);
                    plan.missing_connect_query_source_frontier
                        .push(source_frontier);
                    plan.missing_connect_query_target_frontier
                        .push(target_frontier);
                    plan.missing_connect_query_source_distance
                        .push(source_distance);
                    plan.missing_connect_query_target_distance
                        .push(target_distance);
                }
                if let Some(start) = detail_start {
                    missing_connect_profile_duration += start.elapsed();
                }
            }
        }
        profile_end(ProfileMetric::EnvFeaturesConnectionReachability, profile);
        record_profile_metric(
            ProfileMetric::EnvFeaturesConnectionReachabilityBase,
            base_profile_duration,
        );
        record_profile_metric(
            ProfileMetric::EnvFeaturesConnectionReachabilityFrontiers,
            frontier_profile_duration,
        );
        record_profile_metric(
            ProfileMetric::EnvFeaturesMissingConnectQueries,
            missing_connect_profile_duration,
        );
        record_profile_count(
            ProfileMetric::EnvCounterFeatureUsedConnections,
            used_connection_count as u64,
        );
        record_profile_count(
            ProfileMetric::EnvCounterFeatureConnectionFrontierPairs,
            (used_connection_count * frontier_count) as u64,
        );
        record_profile_count(
            ProfileMetric::EnvCounterFeatureMissingConnectQueryRows,
            plan.missing_connect_query_connection_idx.len() as u64,
        );

        let profile = profile_start();
        if config.save_utility_query || config.refill_utility_query {
            let make_save_refill_row =
                |frontier_distance: GraphDistance, frontier_part: RoomPartIdx| {
                    if frontier_distance == UNREACHABLE_DISTANCE {
                        return None;
                    }
                    let frontier_idx = first_frontier_row_by_part
                        .get(frontier_part as usize)
                        .copied()
                        .unwrap_or(-1);
                    (frontier_idx >= 0)
                        .then(|| SaveRefillUtilityQueryRow::new(frontier_idx, frontier_distance))
                };
            for &room_part in &self.active_room_parts {
                let part = room_part as usize;
                let save_current_to_room = config
                    .save_utility_query
                    .then_some(self.room_part_save_distance_cache.nearest_save_source[part]);
                let save_current_from_room = config
                    .save_utility_query
                    .then_some(self.room_part_save_distance_cache.nearest_save_destination[part]);
                let refill_current_to_room = config
                    .refill_utility_query
                    .then_some(self.room_part_refill_distance_cache.nearest_save_source[part]);
                let refill_current_from_room = config.refill_utility_query.then_some(
                    self.room_part_refill_distance_cache
                        .nearest_save_destination[part],
                );
                let to_room_distance = self
                    .room_part_frontier_distance_cache
                    .nearest_frontier_source[part];
                let to_room_frontier_part = self
                    .room_part_frontier_distance_cache
                    .nearest_frontier_source_part[part];
                let from_room_distance = self
                    .room_part_frontier_distance_cache
                    .nearest_frontier_destination[part];
                let from_room_frontier_part = self
                    .room_part_frontier_distance_cache
                    .nearest_frontier_destination_part[part];
                let mut to_row = make_save_refill_row(to_room_distance, to_room_frontier_part);
                let mut from_row =
                    make_save_refill_row(from_room_distance, from_room_frontier_part);
                if config.save_utility_query {
                    let current_distance = save_current_to_room.unwrap();
                    if let Some(row) = &mut to_row {
                        if save_refill_utility_distance_can_improve(
                            row.frontier_distance,
                            current_distance,
                        ) {
                            row.add_target(0, current_distance);
                        }
                    }
                    let current_distance = save_current_from_room.unwrap();
                    if let Some(row) = &mut from_row {
                        if save_refill_utility_distance_can_improve(
                            row.frontier_distance,
                            current_distance,
                        ) {
                            row.add_target(1, current_distance);
                        }
                    }
                }
                if config.refill_utility_query {
                    let current_distance = refill_current_to_room.unwrap();
                    if let Some(row) = &mut to_row {
                        if save_refill_utility_distance_can_improve(
                            row.frontier_distance,
                            current_distance,
                        ) {
                            row.add_target(2, current_distance);
                        }
                    }
                    let current_distance = refill_current_from_room.unwrap();
                    if let Some(row) = &mut from_row {
                        if save_refill_utility_distance_can_improve(
                            row.frontier_distance,
                            current_distance,
                        ) {
                            row.add_target(3, current_distance);
                        }
                    }
                }
                if let Some(mut to_row) = to_row.filter(|row| row.target_mask != 0) {
                    if let Some(from_row) = from_row.filter(|row| row.target_mask != 0) {
                        if to_row.same_context(&from_row) {
                            to_row.merge(from_row);
                        } else {
                            plan.push_save_refill_utility_query_row(room_part, from_row);
                        }
                    }
                    plan.push_save_refill_utility_query_row(room_part, to_row);
                } else if let Some(from_row) = from_row.filter(|row| row.target_mask != 0) {
                    plan.push_save_refill_utility_query_row(room_part, from_row);
                }
            }
        }
        profile_end(ProfileMetric::EnvFeaturesSaveRefillUtilityQuery, profile);
        record_profile_count(
            ProfileMetric::EnvCounterFeatureSaveRefillUtilityRows,
            plan.save_refill_utility_query_room_part_idx.len() as u64,
        );
        if profile_enabled() {
            let mut save_to_masks = 0u64;
            let mut save_from_masks = 0u64;
            let mut refill_to_masks = 0u64;
            let mut refill_from_masks = 0u64;
            for &target_mask in &plan.save_refill_utility_query_target_mask {
                save_to_masks += u64::from(target_mask & 1 != 0);
                save_from_masks += u64::from(target_mask & 2 != 0);
                refill_to_masks += u64::from(target_mask & 4 != 0);
                refill_from_masks += u64::from(target_mask & 8 != 0);
            }
            record_profile_count(
                ProfileMetric::EnvCounterFeatureSaveRefillUtilitySaveToMasks,
                save_to_masks,
            );
            record_profile_count(
                ProfileMetric::EnvCounterFeatureSaveRefillUtilitySaveFromMasks,
                save_from_masks,
            );
            record_profile_count(
                ProfileMetric::EnvCounterFeatureSaveRefillUtilityRefillToMasks,
                refill_to_masks,
            );
            record_profile_count(
                ProfileMetric::EnvCounterFeatureSaveRefillUtilityRefillFromMasks,
                refill_from_masks,
            );
        }

        if config.toilet_crossed_room {
            plan.toilet_crossed_room_idx
                .push(self.toilet_crossed_room_idx(common));
        }
        plan
    }

    #[cfg(test)]
    fn features_with_occupancy(
        &self,
        common: &CommonData,
        config: &FeatureConfig,
        occupancy: &[u8],
        extra_occupied: Option<(&GeometryData, Coord, Coord)>,
        frontier_neighbor_algorithm: FrontierNeighborAlgorithm,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
        scratch: &mut FeatureScratch,
    ) -> Features {
        assert!(self.frontier.len() <= Self::max_frontiers(common));
        let mut output = scratch.take_features();
        let profile = profile_start();
        let frontier_count = if config.has_frontier_features() {
            self.frontier.len()
        } else {
            0
        };
        record_profile_count(ProfileMetric::EnvCounterFeatureCalls, 1);
        record_profile_count(
            ProfileMetric::EnvCounterFeatureFrontiers,
            frontier_count as u64,
        );
        let mut inventory = std::mem::take(&mut output.inventory);
        if config.inventory {
            inventory.extend(
                self.connection_variant_unused_count
                    .iter()
                    .map(|&count| count as u8),
            );
        }
        let mut room_placed = std::mem::take(&mut output.room_placed);
        if config.room_position {
            room_placed.extend(self.room_used.iter().map(|bit| u8::from(*bit)));
        }
        let mut room_part_furthest_destination =
            std::mem::take(&mut output.room_part_furthest_destination);
        let mut room_part_furthest_source = std::mem::take(&mut output.room_part_furthest_source);
        if config.room_part_furthest_distance {
            self.room_part_furthest_distance_features_into(
                common,
                &mut room_part_furthest_destination,
                &mut room_part_furthest_source,
            );
        }
        let mut room_part_save_from_room_distance =
            std::mem::take(&mut output.room_part_save_from_room_distance);
        let mut room_part_save_to_room_distance =
            std::mem::take(&mut output.room_part_save_to_room_distance);
        if config.room_part_save_distance {
            self.room_part_save_distance_features_into(
                common,
                &mut room_part_save_from_room_distance,
                &mut room_part_save_to_room_distance,
            );
        }
        let mut room_part_refill_from_room_distance =
            std::mem::take(&mut output.room_part_refill_from_room_distance);
        let mut room_part_refill_to_room_distance =
            std::mem::take(&mut output.room_part_refill_to_room_distance);
        if config.room_part_refill_distance {
            self.room_part_refill_distance_features_into(
                common,
                &mut room_part_refill_from_room_distance,
                &mut room_part_refill_to_room_distance,
            );
        }
        let mut room_part_frontier_from_room_distance =
            std::mem::take(&mut output.room_part_frontier_from_room_distance);
        let mut room_part_frontier_to_room_distance =
            std::mem::take(&mut output.room_part_frontier_to_room_distance);
        if config.room_part_frontier_distance {
            self.room_part_frontier_distance_features_into(
                common,
                &mut room_part_frontier_from_room_distance,
                &mut room_part_frontier_to_room_distance,
            );
        }
        let mut known_save_from_room_distance =
            std::mem::take(&mut output.known_save_from_room_distance);
        let mut known_save_to_room_distance =
            std::mem::take(&mut output.known_save_to_room_distance);
        self.known_save_refill_distance_features_into(
            common,
            &self.room_part_save_distance_cache,
            &mut known_save_from_room_distance,
            &mut known_save_to_room_distance,
        );
        let mut known_refill_from_room_distance =
            std::mem::take(&mut output.known_refill_from_room_distance);
        let mut known_refill_to_room_distance =
            std::mem::take(&mut output.known_refill_to_room_distance);
        self.known_save_refill_distance_features_into(
            common,
            &self.room_part_refill_distance_cache,
            &mut known_refill_from_room_distance,
            &mut known_refill_to_room_distance,
        );
        let mut frontier = std::mem::take(&mut output.frontier);
        frontier.resize(frontier_count * FEATURE_FRONTIER_WIDTH, 0);
        let mut frontier_door_variant = std::mem::take(&mut output.frontier_door_variant);
        if config.frontier_door_variant {
            frontier_door_variant.resize(frontier_count, 0);
        }
        let frontier_window_area = frontier_window_size * frontier_window_size;
        let packed_frontier_window_size = frontier_window_area.div_ceil(8);
        let mut frontier_occupancy = std::mem::take(&mut output.frontier_occupancy);
        if config.frontier_occupancy {
            frontier_occupancy.resize(frontier_count * packed_frontier_window_size, 0);
        }
        let mut frontier_neighbor = std::mem::take(&mut output.frontier_neighbor);
        if config.frontier_neighbor {
            frontier_neighbor.resize(frontier_count * frontier_neighbor_count, -1);
        }
        let mut frontier_neighbor_pair = std::mem::take(&mut output.frontier_neighbor_pair);
        if config.frontier_neighbor_flags {
            frontier_neighbor_pair.resize(frontier_count * frontier_neighbor_count, 0);
        }
        let mut connection_reachability = std::mem::take(&mut output.connection_reachability);
        if config.connection_reachability {
            connection_reachability.resize(common.room_connection.len(), 0);
        }
        let mut frontier_connection_reachability =
            std::mem::take(&mut output.frontier_connection_reachability);
        if config.frontier_connection_reachability {
            frontier_connection_reachability
                .resize(frontier_count * common.room_connection.len(), 0);
        }
        let mut missing_connect_query_connection_idx =
            std::mem::take(&mut output.missing_connect_query_connection_idx);
        let mut missing_connect_query_source_frontier =
            std::mem::take(&mut output.missing_connect_query_source_frontier);
        let mut missing_connect_query_target_frontier =
            std::mem::take(&mut output.missing_connect_query_target_frontier);
        let mut missing_connect_query_source_distance =
            std::mem::take(&mut output.missing_connect_query_source_distance);
        let mut missing_connect_query_target_distance =
            std::mem::take(&mut output.missing_connect_query_target_distance);
        let mut missing_connect_query_current_distance =
            std::mem::take(&mut output.missing_connect_query_current_distance);
        let mut save_refill_utility_query_room_part_idx =
            std::mem::take(&mut output.save_refill_utility_query_room_part_idx);
        let mut save_refill_utility_query_target_mask =
            std::mem::take(&mut output.save_refill_utility_query_target_mask);
        let mut save_refill_utility_query_frontier =
            std::mem::take(&mut output.save_refill_utility_query_frontier);
        let mut save_refill_utility_query_frontier_distance =
            std::mem::take(&mut output.save_refill_utility_query_frontier_distance);
        let mut save_refill_utility_query_save_to_current_distance =
            std::mem::take(&mut output.save_refill_utility_query_save_to_current_distance);
        let mut save_refill_utility_query_save_from_current_distance =
            std::mem::take(&mut output.save_refill_utility_query_save_from_current_distance);
        let mut save_refill_utility_query_refill_to_current_distance =
            std::mem::take(&mut output.save_refill_utility_query_refill_to_current_distance);
        let mut save_refill_utility_query_refill_from_current_distance =
            std::mem::take(&mut output.save_refill_utility_query_refill_from_current_distance);
        profile_end(ProfileMetric::EnvFeaturesSetup, profile);

        let profile = profile_start();
        let mut sorted_frontiers = if config.has_frontier_features() {
            self.frontier.iter().collect::<Vec<_>>()
        } else {
            vec![]
        };
        sorted_frontiers.sort_unstable_by_key(|(location, _)| **location);
        profile_end(ProfileMetric::EnvFeaturesSortFrontiers, profile);

        let mut row_door_output_idx = std::mem::take(&mut output.row_door_output_idx);
        if config.has_frontier_features() {
            row_door_output_idx.extend(
                sorted_frontiers
                    .iter()
                    .map(|(_, frontier)| frontier.door_output_idx),
            );
        }
        let graph_size = common.room_part.len();
        let mut first_frontier_row_by_part = vec![-1; graph_size];

        let map_width = self.map_size.0 as usize;
        for (idx, (location, data)) in sorted_frontiers.iter().enumerate() {
            let frontier_part = data.room_part_idx as usize;
            if first_frontier_row_by_part[frontier_part] < 0 {
                first_frontier_row_by_part[frontier_part] = idx as i16;
            }
            let row = idx * FEATURE_FRONTIER_WIDTH;
            frontier[row] = i8::from(config.frontier_mask);
            if config.frontier_position || config.frontier_neighbor_position_embedding {
                frontier[row + 1] = location.x();
                frontier[row + 2] = location.y();
            }
            if config.frontier_orientation {
                frontier[row + 3] = i8::from(location.vertical());
            }
            if config.frontier_kind {
                frontier[row + 4] = data.kind;
            }
            if config.frontier_door_variant {
                frontier_door_variant[idx] = data.door_variant_idx;
            }
            if !config.frontier_occupancy {
                continue;
            }
            let window_start_x = location.x() as isize - frontier_window_size as isize / 2;
            let window_start_y = location.y() as isize - frontier_window_size as isize / 2;
            let window_start = idx * packed_frontier_window_size;
            let map_height = self.map_size.1 as usize;
            let src_x_start = window_start_x.max(0) as usize;
            let src_x_end = (window_start_x + frontier_window_size as isize)
                .min(map_width as isize)
                .max(0) as usize;
            let src_y_start = window_start_y.max(0) as usize;
            let src_y_end = (window_start_y + frontier_window_size as isize)
                .min(map_height as isize)
                .max(0) as usize;
            if src_x_start < src_x_end && src_y_start < src_y_end {
                let dst_x_start = (src_x_start as isize - window_start_x) as usize;
                let copy_width = src_x_end - src_x_start;
                for src_y in src_y_start..src_y_end {
                    let dst_y = (src_y as isize - window_start_y) as usize;
                    let src_start = src_y * map_width + src_x_start;
                    if dst_x_start == 0
                        && copy_width == frontier_window_size
                        && frontier_window_size.is_multiple_of(8)
                    {
                        let dst_start = window_start + dst_y * frontier_window_size / 8;
                        let dst_end = dst_start + frontier_window_size / 8;
                        for (dst, src) in frontier_occupancy[dst_start..dst_end]
                            .iter_mut()
                            .zip(occupancy[src_start..src_start + copy_width].chunks_exact(8))
                        {
                            *dst = src
                                .iter()
                                .enumerate()
                                .fold(0, |byte, (bit_idx, &occupied)| byte | (occupied << bit_idx));
                        }
                        continue;
                    }
                    for src_x in src_x_start..src_x_end {
                        if occupancy[src_y * map_width + src_x] == 0 {
                            continue;
                        }
                        let dst_x = dst_x_start + src_x - src_x_start;
                        let bit_idx = dst_y * frontier_window_size + dst_x;
                        frontier_occupancy[window_start + bit_idx / 8] |= 1 << (bit_idx % 8);
                    }
                }
            }
            if let Some((geometry, offset_x, offset_y)) = extra_occupied {
                if offset_x + geometry.max_x < window_start_x as Coord
                    || offset_x + geometry.min_x
                        >= (window_start_x + frontier_window_size as isize) as Coord
                    || offset_y + geometry.max_y < window_start_y as Coord
                    || offset_y + geometry.min_y
                        >= (window_start_y + frontier_window_size as isize) as Coord
                {
                    continue;
                }
                for &(dx, dy) in &geometry.occupied_tiles {
                    let window_x = offset_x as isize + dx as isize - window_start_x;
                    let window_y = offset_y as isize + dy as isize - window_start_y;
                    if window_x >= 0
                        && window_x < frontier_window_size as isize
                        && window_y >= 0
                        && window_y < frontier_window_size as isize
                    {
                        let bit_idx = window_y as usize * frontier_window_size + window_x as usize;
                        frontier_occupancy[window_start + bit_idx / 8] |= 1 << (bit_idx % 8);
                    }
                }
            }
        }

        let profile = profile_start();
        let detailed_connection_profile = profile_enabled();
        let mut base_profile_duration = Duration::ZERO;
        let mut frontier_profile_duration = Duration::ZERO;
        let mut missing_connect_profile_duration = Duration::ZERO;
        let mut used_connection_count = 0usize;
        for (connection_idx, connection) in common.room_connection.iter().enumerate() {
            if !self.room_used[connection.room_idx as usize] {
                continue;
            }
            used_connection_count += 1;
            let from_component =
                self.room_part_component(common, connection.room_idx, connection.from_part);
            let to_component =
                self.room_part_component(common, connection.room_idx, connection.to_part);
            let detail_start = detailed_connection_profile.then(Instant::now);
            let already_reachable = self.scc_dag.can_reach(from_component, to_component);
            if config.connection_reachability && already_reachable {
                connection_reachability[connection_idx] = 1;
            }
            if let Some(start) = detail_start {
                base_profile_duration += start.elapsed();
            }
            if config.frontier_connection_reachability {
                let detail_start = detailed_connection_profile.then(Instant::now);
                for (frontier_idx, (_, frontier)) in sorted_frontiers.iter().enumerate() {
                    let mut flags = 0;
                    if self.scc_dag.can_reach(from_component, frontier.component) {
                        flags |= 1;
                    }
                    if self.scc_dag.can_reach(frontier.component, to_component) {
                        flags |= 2;
                    }
                    frontier_connection_reachability
                        [frontier_idx * common.room_connection.len() + connection_idx] = flags;
                }
                if let Some(start) = detail_start {
                    frontier_profile_duration += start.elapsed();
                }
            }
            if config.missing_connect_query {
                let detail_start = detailed_connection_profile.then(Instant::now);
                let from_part =
                    Self::room_part_idx(common, connection.room_idx, connection.from_part) as usize;
                let to_part =
                    Self::room_part_idx(common, connection.room_idx, connection.to_part) as usize;
                let source_distance = self
                    .room_part_frontier_distance_cache
                    .nearest_frontier_destination[from_part];
                let source_frontier_part = self
                    .room_part_frontier_distance_cache
                    .nearest_frontier_destination_part[from_part];
                let source_frontier = if source_frontier_part == RoomPartIdx::MAX {
                    -1
                } else {
                    first_frontier_row_by_part[source_frontier_part as usize]
                };
                let target_distance = self
                    .room_part_frontier_distance_cache
                    .nearest_frontier_source[to_part];
                let target_frontier_part = self
                    .room_part_frontier_distance_cache
                    .nearest_frontier_source_part[to_part];
                let target_frontier = if target_frontier_part == RoomPartIdx::MAX {
                    -1
                } else {
                    first_frontier_row_by_part[target_frontier_part as usize]
                };
                let current_distance = self.graph_distance[from_part * graph_size + to_part];
                let pair_total_distance = if source_frontier >= 0 && target_frontier >= 0 {
                    u16::from(source_distance) + u16::from(target_distance)
                } else {
                    0
                };
                let pair_can_improve = source_frontier >= 0
                    && target_frontier >= 0
                    && (current_distance == UNREACHABLE_DISTANCE
                        || pair_total_distance + 2 < u16::from(current_distance));
                let emit_missing_connect_query = source_frontier >= 0
                    && target_frontier >= 0
                    && (!already_reachable || pair_can_improve);
                if emit_missing_connect_query {
                    missing_connect_query_connection_idx.push(connection_idx as i64);
                    missing_connect_query_current_distance.push(current_distance);
                    missing_connect_query_source_frontier.push(source_frontier);
                    missing_connect_query_target_frontier.push(target_frontier);
                    missing_connect_query_source_distance.push(if source_frontier >= 0 {
                        source_distance
                    } else {
                        0
                    });
                    missing_connect_query_target_distance.push(if target_frontier >= 0 {
                        target_distance
                    } else {
                        0
                    });
                }
                if let Some(start) = detail_start {
                    missing_connect_profile_duration += start.elapsed();
                }
            }
        }
        profile_end(ProfileMetric::EnvFeaturesConnectionReachability, profile);
        record_profile_metric(
            ProfileMetric::EnvFeaturesConnectionReachabilityBase,
            base_profile_duration,
        );
        record_profile_metric(
            ProfileMetric::EnvFeaturesConnectionReachabilityFrontiers,
            frontier_profile_duration,
        );
        record_profile_metric(
            ProfileMetric::EnvFeaturesMissingConnectQueries,
            missing_connect_profile_duration,
        );
        record_profile_count(
            ProfileMetric::EnvCounterFeatureUsedConnections,
            used_connection_count as u64,
        );
        record_profile_count(
            ProfileMetric::EnvCounterFeatureConnectionFrontierPairs,
            (used_connection_count * frontier_count) as u64,
        );
        record_profile_count(
            ProfileMetric::EnvCounterFeatureMissingConnectQueryRows,
            missing_connect_query_connection_idx.len() as u64,
        );
        let profile = profile_start();
        if config.save_utility_query || config.refill_utility_query {
            let make_save_refill_row =
                |frontier_distance: GraphDistance, frontier_part: RoomPartIdx| {
                    if frontier_distance == UNREACHABLE_DISTANCE {
                        return None;
                    }
                    let frontier_idx = first_frontier_row_by_part
                        .get(frontier_part as usize)
                        .copied()
                        .unwrap_or(-1);
                    (frontier_idx >= 0)
                        .then(|| SaveRefillUtilityQueryRow::new(frontier_idx, frontier_distance))
                };
            let mut push_save_refill_query =
                |room_part: RoomPartIdx, row: SaveRefillUtilityQueryRow| {
                    save_refill_utility_query_room_part_idx.push(i64::from(room_part));
                    save_refill_utility_query_target_mask.push(row.target_mask);
                    save_refill_utility_query_frontier.push(row.frontier);
                    save_refill_utility_query_frontier_distance.push(row.frontier_distance);
                    save_refill_utility_query_save_to_current_distance
                        .push(row.save_to_current_distance);
                    save_refill_utility_query_save_from_current_distance
                        .push(row.save_from_current_distance);
                    save_refill_utility_query_refill_to_current_distance
                        .push(row.refill_to_current_distance);
                    save_refill_utility_query_refill_from_current_distance
                        .push(row.refill_from_current_distance);
                };
            for &room_part in &self.active_room_parts {
                let part = room_part as usize;
                let save_current_to_room = config
                    .save_utility_query
                    .then_some(self.room_part_save_distance_cache.nearest_save_source[part]);
                let save_current_from_room = config
                    .save_utility_query
                    .then_some(self.room_part_save_distance_cache.nearest_save_destination[part]);
                let refill_current_to_room = config
                    .refill_utility_query
                    .then_some(self.room_part_refill_distance_cache.nearest_save_source[part]);
                let refill_current_from_room = config.refill_utility_query.then_some(
                    self.room_part_refill_distance_cache
                        .nearest_save_destination[part],
                );
                let to_room_distance = self
                    .room_part_frontier_distance_cache
                    .nearest_frontier_source[part];
                let to_room_frontier_part = self
                    .room_part_frontier_distance_cache
                    .nearest_frontier_source_part[part];
                let from_room_distance = self
                    .room_part_frontier_distance_cache
                    .nearest_frontier_destination[part];
                let from_room_frontier_part = self
                    .room_part_frontier_distance_cache
                    .nearest_frontier_destination_part[part];
                let mut to_row = make_save_refill_row(to_room_distance, to_room_frontier_part);
                let mut from_row =
                    make_save_refill_row(from_room_distance, from_room_frontier_part);
                if config.save_utility_query {
                    let current_distance = save_current_to_room.unwrap();
                    if let Some(row) = &mut to_row {
                        if save_refill_utility_distance_can_improve(
                            row.frontier_distance,
                            current_distance,
                        ) {
                            row.add_target(0, current_distance);
                        }
                    }
                    let current_distance = save_current_from_room.unwrap();
                    if let Some(row) = &mut from_row {
                        if save_refill_utility_distance_can_improve(
                            row.frontier_distance,
                            current_distance,
                        ) {
                            row.add_target(1, current_distance);
                        }
                    }
                }
                if config.refill_utility_query {
                    let current_distance = refill_current_to_room.unwrap();
                    if let Some(row) = &mut to_row {
                        if save_refill_utility_distance_can_improve(
                            row.frontier_distance,
                            current_distance,
                        ) {
                            row.add_target(2, current_distance);
                        }
                    }
                    let current_distance = refill_current_from_room.unwrap();
                    if let Some(row) = &mut from_row {
                        if save_refill_utility_distance_can_improve(
                            row.frontier_distance,
                            current_distance,
                        ) {
                            row.add_target(3, current_distance);
                        }
                    }
                }
                if let Some(mut to_row) = to_row.filter(|row| row.target_mask != 0) {
                    if let Some(from_row) = from_row.filter(|row| row.target_mask != 0) {
                        if to_row.same_context(&from_row) {
                            to_row.merge(from_row);
                        } else {
                            push_save_refill_query(room_part, from_row);
                        }
                    }
                    push_save_refill_query(room_part, to_row);
                } else if let Some(from_row) = from_row.filter(|row| row.target_mask != 0) {
                    push_save_refill_query(room_part, from_row);
                }
            }
        }
        profile_end(ProfileMetric::EnvFeaturesSaveRefillUtilityQuery, profile);
        record_profile_count(
            ProfileMetric::EnvCounterFeatureSaveRefillUtilityRows,
            save_refill_utility_query_room_part_idx.len() as u64,
        );
        if profile_enabled() {
            let mut save_to_masks = 0u64;
            let mut save_from_masks = 0u64;
            let mut refill_to_masks = 0u64;
            let mut refill_from_masks = 0u64;
            for &target_mask in &save_refill_utility_query_target_mask {
                save_to_masks += u64::from(target_mask & 1 != 0);
                save_from_masks += u64::from(target_mask & 2 != 0);
                refill_to_masks += u64::from(target_mask & 4 != 0);
                refill_from_masks += u64::from(target_mask & 8 != 0);
            }
            record_profile_count(
                ProfileMetric::EnvCounterFeatureSaveRefillUtilitySaveToMasks,
                save_to_masks,
            );
            record_profile_count(
                ProfileMetric::EnvCounterFeatureSaveRefillUtilitySaveFromMasks,
                save_from_masks,
            );
            record_profile_count(
                ProfileMetric::EnvCounterFeatureSaveRefillUtilityRefillToMasks,
                refill_to_masks,
            );
            record_profile_count(
                ProfileMetric::EnvCounterFeatureSaveRefillUtilityRefillFromMasks,
                refill_from_masks,
            );
        }

        if config.frontier_neighbor {
            let profile = profile_start();
            let mut locations = std::mem::take(&mut scratch.frontier_locations);
            locations.clear();
            locations.extend(sorted_frontiers.iter().map(|(location, _)| **location));
            match frontier_neighbor_algorithm {
                FrontierNeighborAlgorithm::Nearest if frontier_neighbor_count == 1 => {
                    write_single_frontier_nearest_neighbor(
                        &locations,
                        true,
                        &mut frontier_neighbor,
                    );
                }
                FrontierNeighborAlgorithm::NearestExclusive if frontier_neighbor_count == 1 => {
                    write_single_frontier_nearest_neighbor(
                        &locations,
                        false,
                        &mut frontier_neighbor,
                    );
                }
                _ => match frontier_neighbor_algorithm {
                    FrontierNeighborAlgorithm::Delaunay => write_frontier_delaunay_neighbors(
                        &locations,
                        frontier_neighbor_count,
                        &mut frontier_neighbor,
                        scratch,
                    ),
                    FrontierNeighborAlgorithm::Nearest => write_frontier_nearest_neighbors(
                        &locations,
                        frontier_neighbor_count,
                        true,
                        &mut frontier_neighbor,
                        scratch,
                    ),
                    FrontierNeighborAlgorithm::NearestExclusive => {
                        write_frontier_nearest_neighbors(
                            &locations,
                            frontier_neighbor_count,
                            false,
                            &mut frontier_neighbor,
                            scratch,
                        )
                    }
                },
            }
            scratch.frontier_locations = locations;
            profile_end(ProfileMetric::EnvFeaturesFrontierNeighbor, profile);
        }
        if config.frontier_neighbor_flags {
            let profile = profile_start();
            for (src_idx, (_, src)) in sorted_frontiers.iter().enumerate() {
                for neighbor_idx in 0..frontier_neighbor_count {
                    let dst_idx =
                        frontier_neighbor[src_idx * frontier_neighbor_count + neighbor_idx];
                    if dst_idx < 0 {
                        break;
                    }
                    let dst_idx = dst_idx as usize;
                    let (_, dst) = sorted_frontiers[dst_idx];
                    let mut flags = 0;
                    if src.component == dst.component {
                        flags |= 1;
                    }
                    if self.scc_dag.can_reach(src.component, dst.component) {
                        flags |= 2;
                    }
                    if self.scc_dag.can_reach(dst.component, src.component) {
                        flags |= 4;
                    }
                    let pair_idx = src_idx * frontier_neighbor_count + neighbor_idx;
                    frontier_neighbor_pair[pair_idx] = flags;
                }
            }
            profile_end(ProfileMetric::EnvFeaturesFrontierNeighborFlags, profile);
        }
        let profile = profile_start();
        let mut room_x = std::mem::take(&mut output.room_x);
        let mut room_y = std::mem::take(&mut output.room_y);
        if config.room_position {
            room_x.extend_from_slice(&self.room_x);
            room_y.extend_from_slice(&self.room_y);
        }
        profile_end(ProfileMetric::EnvFeaturesRoomPositionClone, profile);

        let mut toilet_crossed_room_idx = std::mem::take(&mut output.toilet_crossed_room_idx);
        if config.toilet_crossed_room {
            toilet_crossed_room_idx.push(self.toilet_crossed_room_idx(common));
        }

        let profile = profile_start();
        let result = Features {
            inventory,
            room_x,
            room_y,
            room_placed,
            room_part_furthest_destination,
            room_part_furthest_source,
            room_part_save_from_room_distance,
            room_part_save_to_room_distance,
            room_part_refill_from_room_distance,
            room_part_refill_to_room_distance,
            room_part_frontier_from_room_distance,
            room_part_frontier_to_room_distance,
            known_save_from_room_distance,
            known_save_to_room_distance,
            known_refill_from_room_distance,
            known_refill_to_room_distance,
            frontier,
            frontier_door_variant,
            row_door_output_idx,
            frontier_occupancy,
            frontier_neighbor,
            frontier_neighbor_pair,
            connection_reachability,
            frontier_connection_reachability,
            missing_connect_query_connection_idx,
            missing_connect_query_source_frontier,
            missing_connect_query_target_frontier,
            missing_connect_query_source_distance,
            missing_connect_query_target_distance,
            missing_connect_query_current_distance,
            save_refill_utility_query_room_part_idx,
            save_refill_utility_query_target_mask,
            save_refill_utility_query_frontier,
            save_refill_utility_query_frontier_distance,
            save_refill_utility_query_save_to_current_distance,
            save_refill_utility_query_save_from_current_distance,
            save_refill_utility_query_refill_to_current_distance,
            save_refill_utility_query_refill_from_current_distance,
            toilet_crossed_room_idx,
        };
        profile_end(ProfileMetric::EnvFeaturesOutput, profile);
        result
    }

    #[cfg(test)]
    pub fn features_after_candidate(
        &mut self,
        common: &CommonData,
        candidate: Action,
        config: &FeatureConfig,
        frontier_neighbor_algorithm: FrontierNeighborAlgorithm,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
    ) -> Features {
        let mut scratch = FeatureScratch::default();
        self.features_after_candidate_with_scratch(
            common,
            candidate,
            config,
            frontier_neighbor_algorithm,
            frontier_neighbor_count,
            frontier_window_size,
            &mut scratch,
        )
    }

    #[cfg(test)]
    pub fn features_after_candidate_with_scratch(
        &mut self,
        common: &CommonData,
        candidate: Action,
        config: &FeatureConfig,
        frontier_neighbor_algorithm: FrontierNeighborAlgorithm,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
        scratch: &mut FeatureScratch,
    ) -> Features {
        if config.is_empty() {
            return scratch.take_features();
        }
        let extra_occupied =
            if config.frontier_occupancy && candidate.room_idx < common.room.len() as RoomIdx {
                let geometry_idx = common.room[candidate.room_idx as usize].geometry_idx;
                Some((
                    &common.geometry[geometry_idx as usize],
                    candidate.x,
                    candidate.y,
                ))
            } else {
                None
            };
        let profile = profile_start();
        let snapshot = self.apply_feature_candidate(candidate, common);
        profile_end(ProfileMetric::EnvFeaturesApplyCandidate, profile);
        let features = self.features_with_occupancy(
            common,
            config,
            &self.occupancy,
            extra_occupied,
            frontier_neighbor_algorithm,
            frontier_neighbor_count,
            frontier_window_size,
            scratch,
        );
        let profile = profile_start();
        self.restore_feature_candidate(common, candidate, snapshot);
        profile_end(ProfileMetric::EnvFeaturesApplyCandidate, profile);
        features
    }

    pub fn feature_plan_after_candidate_with_scratch(
        &mut self,
        common: &CommonData,
        candidate: Action,
        config: &FeatureConfig,
        frontier_neighbor_algorithm: FrontierNeighborAlgorithm,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
        scratch: &mut FeatureScratch,
    ) -> FeaturePlan {
        if config.is_empty() {
            return scratch.take_plan();
        }
        let extra_occupied = if candidate.room_idx < common.room.len() as RoomIdx {
            let geometry_idx = common.room[candidate.room_idx as usize].geometry_idx;
            Some(FeatureExtraOccupied {
                geometry_idx,
                x: candidate.x,
                y: candidate.y,
            })
        } else {
            None
        };
        let profile = profile_start();
        let snapshot = self.apply_feature_candidate(candidate, common);
        profile_end(ProfileMetric::EnvFeaturesApplyCandidate, profile);
        let plan = self.feature_plan_with_occupancy(
            common,
            config,
            extra_occupied,
            FeaturePlanKind::Candidate(candidate),
            frontier_neighbor_algorithm,
            frontier_neighbor_count,
            frontier_window_size,
            scratch,
        );
        let profile = profile_start();
        self.restore_feature_candidate(common, candidate, snapshot);
        profile_end(ProfileMetric::EnvFeaturesApplyCandidate, profile);
        plan
    }

    pub fn actions(&self) -> &[Action] {
        &self.actions
    }

    pub fn map_size(&self) -> (Coord, Coord) {
        self.map_size
    }

    pub fn occupancy(&self) -> &[u8] {
        &self.occupancy
    }

    pub fn connection_variant_unused_count(&self) -> &[usize] {
        &self.connection_variant_unused_count
    }

    pub fn room_x(&self) -> &[Coord] {
        &self.room_x
    }

    pub fn room_y(&self) -> &[Coord] {
        &self.room_y
    }

    pub fn room_used_at(&self, room_idx: RoomIdx) -> bool {
        self.room_used[room_idx as usize]
    }

    pub fn add_door_match_counts(
        &self,
        common: &CommonData,
        horizontal_counts: &mut [u64],
        vertical_counts: &mut [u64],
    ) {
        add_orientation_match_counts(
            &self.door_matches,
            common,
            Direction::Left,
            Direction::Right,
            horizontal_counts,
        );
        add_orientation_match_counts(
            &self.door_matches,
            common,
            Direction::Up,
            Direction::Down,
            vertical_counts,
        );
    }

    pub fn write_door_matches(
        &self,
        left: &mut [i16],
        right: &mut [i16],
        up: &mut [i16],
        down: &mut [i16],
    ) {
        write_direction_door_matches(&self.door_matches[Direction::Left as usize], left);
        write_direction_door_matches(&self.door_matches[Direction::Right as usize], right);
        write_direction_door_matches(&self.door_matches[Direction::Up as usize], up);
        write_direction_door_matches(&self.door_matches[Direction::Down as usize], down);
    }

    pub fn outcomes(&self, common: &CommonData) -> StepOutcomes {
        let mut door_valid = vec![];
        for dir in 0..NUM_DIRS {
            for i in 0..common.room_dir_door[dir].len() {
                door_valid.push(self.door_outcome(common, dir, i));
            }
        }

        let frontier_reachability = if self.finished {
            None
        } else {
            Some(self.scc_dag_with_merged_frontiers())
        };
        let mut connections_valid = Vec::with_capacity(common.room_connection.len());
        for connection_idx in 0..common.room_connection.len() {
            connections_valid.push(self.connection_outcome(
                common,
                connection_idx,
                frontier_reachability.as_ref(),
            ));
        }

        StepOutcomes {
            door_valid,
            connections_valid,
            toilet_valid: self.toilet_outcome(common),
            phantoon_valid: self.phantoon_outcome(common),
            toilet_crossed_room_idx: self.toilet_crossed_room_idx(common),
        }
    }

    fn toilet_outcome(&self, common: &CommonData) -> DoorValidOutcome {
        let Some(toilet_room_idx) = common.toilet_room_idx() else {
            return DoorValidOutcome::Valid;
        };
        if !self.room_used[toilet_room_idx as usize] {
            return if self.finished {
                DoorValidOutcome::Invalid
            } else {
                DoorValidOutcome::Unknown
            };
        }

        let toilet_x = self.room_x[toilet_room_idx as usize];
        let toilet_y = self.room_y[toilet_room_idx as usize];
        let mut crossing_count = 0;
        for action in &self.actions {
            if action.room_idx == toilet_room_idx {
                continue;
            }
            if room_crosses_toilet(common, *action, toilet_x, toilet_y) {
                crossing_count += 1;
                if crossing_count > 1 {
                    return DoorValidOutcome::Invalid;
                }
            }
        }

        if self.finished {
            if crossing_count == 1 {
                DoorValidOutcome::Valid
            } else {
                DoorValidOutcome::Invalid
            }
        } else {
            DoorValidOutcome::Unknown
        }
    }

    fn toilet_crossed_room_idx(&self, common: &CommonData) -> i16 {
        let Some(toilet_room_idx) = common.toilet_room_idx() else {
            return -1;
        };
        if !self.room_used[toilet_room_idx as usize] {
            return -1;
        }

        let toilet_x = self.room_x[toilet_room_idx as usize];
        let toilet_y = self.room_y[toilet_room_idx as usize];
        let mut crossed_room_idx = -1;
        for action in &self.actions {
            if action.room_idx == toilet_room_idx {
                continue;
            }
            if room_crosses_toilet(common, *action, toilet_x, toilet_y) {
                if crossed_room_idx >= 0 {
                    return -1;
                }
                crossed_room_idx = action.room_idx as i16;
            }
        }
        crossed_room_idx
    }

    fn phantoon_outcome(&self, common: &CommonData) -> DoorValidOutcome {
        let (Some(boss_room_idx), Some(map_room_idx)) = (
            common.phantoon_boss_room_idx(),
            common.phantoon_map_room_idx(),
        ) else {
            return DoorValidOutcome::Valid;
        };
        let boss_used = self.room_used[boss_room_idx as usize];
        let map_used = self.room_used[map_room_idx as usize];
        let boss_neighbor = boss_used
            .then(|| self.matched_neighbor_room_idx(common, common.phantoon_boss_door()))
            .flatten();
        let map_neighbor = map_used
            .then(|| self.matched_neighbor_room_idx(common, common.phantoon_map_door()))
            .flatten();
        match (boss_neighbor, map_neighbor) {
            (Some(boss_neighbor), Some(map_neighbor)) => {
                if boss_neighbor == map_neighbor {
                    DoorValidOutcome::Valid
                } else {
                    DoorValidOutcome::Invalid
                }
            }
            (Some(boss_neighbor), None) => {
                if self.room_can_match_neighbor_frontier(
                    common,
                    map_room_idx,
                    common.phantoon_map_door(),
                    boss_neighbor,
                ) {
                    DoorValidOutcome::Unknown
                } else {
                    DoorValidOutcome::Invalid
                }
            }
            (None, Some(map_neighbor)) => {
                if self.room_can_match_neighbor_frontier(
                    common,
                    boss_room_idx,
                    common.phantoon_boss_door(),
                    map_neighbor,
                ) {
                    DoorValidOutcome::Unknown
                } else {
                    DoorValidOutcome::Invalid
                }
            }
            _ => {
                if self.finished {
                    DoorValidOutcome::Invalid
                } else {
                    DoorValidOutcome::Unknown
                }
            }
        }
    }

    fn matched_neighbor_room_idx(
        &self,
        common: &CommonData,
        door: Option<(Direction, DirDoorIdx)>,
    ) -> Option<RoomIdx> {
        let (direction, dir_door_idx) = door?;
        let matched_door_idx = self.door_matches[direction as usize][dir_door_idx as usize];
        if matched_door_idx == DirDoorIdx::MAX {
            return None;
        }
        Some(
            common.room_dir_door[direction.opposite() as usize][matched_door_idx as usize].room_idx,
        )
    }

    fn room_can_match_neighbor_frontier(
        &self,
        common: &CommonData,
        room_idx: RoomIdx,
        door: Option<(Direction, DirDoorIdx)>,
        neighbor_room_idx: RoomIdx,
    ) -> bool {
        if self.room_used[room_idx as usize] {
            return false;
        }
        let (direction, _dir_door_idx) = match door {
            Some(door) => door,
            None => return false,
        };
        self.frontier.values().any(|frontier| {
            let frontier_room_idx = common.room_part[frontier.room_part_idx as usize].0;
            frontier_room_idx == neighbor_room_idx && frontier.direction == direction.opposite()
        })
    }

    fn door_outcome(&self, common: &CommonData, dir: usize, i: usize) -> DoorValidOutcome {
        if self.door_matches[dir][i] != DirDoorIdx::MAX {
            return DoorValidOutcome::Valid;
        }
        if self.finished {
            return DoorValidOutcome::Invalid;
        }

        let room_dir_door = &common.room_dir_door[dir][i];
        let room_idx = room_dir_door.room_idx;
        if !self.room_used[room_idx as usize] {
            return DoorValidOutcome::Unknown;
        }
        match self.frontier.get(&DoorLocation::from_room_dir_door(
            room_dir_door,
            self.room_x[room_idx as usize],
            self.room_y[room_idx as usize],
        )) {
            None => DoorValidOutcome::Invalid,
            Some(frontier) if frontier.candidates.is_empty() => DoorValidOutcome::Invalid,
            Some(_) => DoorValidOutcome::Unknown,
        }
    }

    fn connection_outcome(
        &self,
        common: &CommonData,
        connection_idx: usize,
        frontier_reachability: Option<&(SccDag, Vec<usize>)>,
    ) -> DoorValidOutcome {
        let connection = &common.room_connection[connection_idx];
        if self.room_used[connection.room_idx as usize] {
            let from_component =
                self.room_part_component(common, connection.room_idx, connection.from_part);
            let to_component =
                self.room_part_component(common, connection.room_idx, connection.to_part);
            if self.scc_dag.can_reach(from_component, to_component) {
                DoorValidOutcome::Valid
            } else if let Some((frontier_merged_scc_dag, frontier_merged_component_remap)) =
                frontier_reachability
            {
                if frontier_merged_scc_dag.can_reach(
                    frontier_merged_component_remap[from_component],
                    frontier_merged_component_remap[to_component],
                ) {
                    DoorValidOutcome::Unknown
                } else {
                    DoorValidOutcome::Invalid
                }
            } else {
                DoorValidOutcome::Invalid
            }
        } else if self.finished {
            DoorValidOutcome::Invalid
        } else {
            DoorValidOutcome::Unknown
        }
    }

    pub fn verified_outcomes(
        &mut self,
        common: &CommonData,
        stage: &str,
    ) -> Result<StepOutcomes, String> {
        let outcomes = self.outcomes(common);
        if let Some(known_outcomes) = &self.known_outcomes {
            check_outcome_transition_consistency(
                &known_outcomes.door_valid,
                &outcomes.door_valid,
                "door",
                stage,
            )?;
            check_outcome_transition_consistency(
                &known_outcomes.connections_valid,
                &outcomes.connections_valid,
                "connection",
                stage,
            )?;
            check_outcome_transition_consistency(
                &[known_outcomes.toilet_valid],
                &[outcomes.toilet_valid],
                "toilet",
                stage,
            )?;
            check_outcome_transition_consistency(
                &[known_outcomes.phantoon_valid],
                &[outcomes.phantoon_valid],
                "phantoon",
                stage,
            )?;
        }
        self.known_outcomes = Some(merge_known_outcomes(
            self.known_outcomes.as_ref(),
            &outcomes,
        ));
        Ok(outcomes)
    }
}

fn merge_known_outcomes(known: Option<&StepOutcomes>, current: &StepOutcomes) -> StepOutcomes {
    let Some(known) = known else {
        return current.clone();
    };
    StepOutcomes {
        door_valid: merge_known_outcome_values(&known.door_valid, &current.door_valid),
        connections_valid: merge_known_outcome_values(
            &known.connections_valid,
            &current.connections_valid,
        ),
        toilet_valid: merge_known_outcome_value(known.toilet_valid, current.toilet_valid),
        phantoon_valid: merge_known_outcome_value(known.phantoon_valid, current.phantoon_valid),
        toilet_crossed_room_idx: current.toilet_crossed_room_idx,
    }
}

fn room_crosses_toilet(
    common: &CommonData,
    action: Action,
    toilet_x: Coord,
    toilet_y: Coord,
) -> bool {
    if action.room_idx >= common.room.len() as RoomIdx {
        return false;
    }
    let geometry_idx = common.room[action.room_idx as usize].geometry_idx;
    let geometry = &common.geometry[geometry_idx as usize];
    let crossing_x = toilet_x - action.x;
    if crossing_x < 0 || crossing_x >= geometry.map[0].len() as Coord {
        return false;
    }
    for open_y in 2..=7 {
        let room_y = toilet_y + open_y - action.y;
        if room_y >= 0
            && room_y < geometry.map.len() as Coord
            && geometry.map[room_y as usize][crossing_x as usize] != 0
        {
            return true;
        }
    }
    false
}

fn merge_known_outcome_value(
    known: DoorValidOutcome,
    current: DoorValidOutcome,
) -> DoorValidOutcome {
    if known == DoorValidOutcome::Unknown {
        current
    } else {
        known
    }
}

fn merge_known_outcome_values(
    known: &[DoorValidOutcome],
    current: &[DoorValidOutcome],
) -> Vec<DoorValidOutcome> {
    debug_assert_eq!(known.len(), current.len());
    known
        .iter()
        .zip(current)
        .map(|(&known, &current)| {
            if known == DoorValidOutcome::Unknown {
                current
            } else {
                known
            }
        })
        .collect()
}

fn add_orientation_match_counts(
    door_matches: &[Vec<DirDoorIdx>; NUM_DIRS],
    common: &CommonData,
    row_direction: Direction,
    column_direction: Direction,
    counts: &mut [u64],
) {
    let row_count = common.room_dir_door[row_direction as usize].len();
    let column_count = common.room_dir_door[column_direction as usize].len();
    debug_assert_eq!(counts.len(), (row_count + 1) * (column_count + 1));

    let unmatched_column = column_count;
    for (row_idx, &column_idx) in door_matches[row_direction as usize].iter().enumerate() {
        let column_idx = if column_idx == DirDoorIdx::MAX {
            unmatched_column
        } else {
            column_idx as usize
        };
        counts[row_idx * (column_count + 1) + column_idx] += 1;
    }

    let unmatched_row = row_count;
    for (column_idx, &row_idx) in door_matches[column_direction as usize].iter().enumerate() {
        if row_idx == DirDoorIdx::MAX {
            counts[unmatched_row * (column_count + 1) + column_idx] += 1;
        }
    }
}

fn write_direction_door_matches(matches: &[DirDoorIdx], output: &mut [i16]) {
    debug_assert_eq!(matches.len(), output.len());
    for (dst, &door_idx) in output.iter_mut().zip(matches) {
        *dst = if door_idx == DirDoorIdx::MAX {
            -1
        } else {
            i16::from(door_idx)
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{Direction, Room};

    fn door_location(x: Coord, y: Coord, vertical: bool) -> DoorLocation {
        DoorLocation::from_parts(
            if vertical {
                Direction::Up
            } else {
                Direction::Left
            },
            x,
            y,
            0,
            0,
        )
    }

    fn assert_symmetric_neighbors(neighbors: &[Vec<usize>], max_degree: usize) {
        for (src, row) in neighbors.iter().enumerate() {
            assert!(row.len() <= max_degree);
            assert!(!row.contains(&src));
            assert!(row.windows(2).all(|pair| pair[0] < pair[1]));
            for &dst in row {
                assert!(neighbors[dst].contains(&src));
            }
        }
    }

    fn feature_outcome_test_common() -> CommonData {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        CommonData::new(rooms).unwrap()
    }

    fn feature_outcome_test_env(common: &CommonData) -> Environment {
        let mut env = Environment::new(common, (4, 4), 8, 0);
        env.step(
            Action {
                room_idx: 0,
                x: 0,
                y: 0,
            },
            common,
        );
        env
    }

    fn spatial_index_test_common() -> CommonData {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [[]],
                "connections": [],
                "missing_connections": []
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [[]],
                "connections": [],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        CommonData::new(rooms).unwrap()
    }

    #[test]
    fn spatial_index_shortlists_without_over_rejecting_coarse_cell_matches() {
        let common = spatial_index_test_common();
        let mut env = Environment::new(&common, (8, 8), 8, 0);
        env.step(
            Action {
                room_idx: 0,
                x: 0,
                y: 0,
            },
            &common,
        );

        assert!(!env.candidate_intersects_placed_room(&common, 0, 1, 0));
        assert!(env.candidate_intersects_placed_room(&common, 0, 0, 0));

        env.clear(&common);
        assert!(!env.candidate_intersects_placed_room(&common, 0, 0, 0));
    }

    #[test]
    fn outcomes_after_candidate_restores_spatial_index() {
        let common = spatial_index_test_common();
        let mut env = Environment::new(&common, (8, 8), 8, 0);
        env.step(
            Action {
                room_idx: 0,
                x: 0,
                y: 0,
            },
            &common,
        );
        let insertion_len = env.placed_room_index.insertion_len();

        env.outcomes_after_candidate(
            &common,
            Action {
                room_idx: 1,
                x: 2,
                y: 0,
            },
        );

        assert_eq!(env.placed_room_index.insertion_len(), insertion_len);
        assert!(!env.candidate_intersects_placed_room(&common, 0, 2, 0));
    }

    fn assert_feature_outcomes_eq(left: &FeatureOutcomes, right: &FeatureOutcomes) {
        assert_eq!(
            left.step_outcomes.door_valid,
            right.step_outcomes.door_valid
        );
        assert_eq!(
            left.step_outcomes.connections_valid,
            right.step_outcomes.connections_valid
        );
        assert_eq!(
            left.step_outcomes.toilet_valid,
            right.step_outcomes.toilet_valid
        );
        assert_eq!(
            left.step_outcomes.toilet_crossed_room_idx,
            right.step_outcomes.toilet_crossed_room_idx
        );
        assert_eq!(left.door_match, right.door_match);
    }

    #[test]
    fn proposal_candidate_mask_marks_placeable_frontier_cells() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1], [1, 0]],
                "missing_connections": []
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1], [1, 0]],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 8, 0);
        env.step(
            Action {
                room_idx: 0,
                x: 2,
                y: 2,
            },
            &common,
        );

        let frontier_count = Environment::max_frontiers(&common);
        let door_variant_count = common.num_door_output_variants;
        let mut mask = vec![0; door_variant_count.div_ceil(8)];
        let mut frontier_idx = -1;
        let mut valid_counts = 0;
        env.proposal_candidate_mask(
            &common,
            door_variant_count,
            &mut frontier_idx,
            &mut mask,
            &mut valid_counts,
        );

        assert!(frontier_count > 0);
        assert!(frontier_idx >= 0);
        assert!(valid_counts > 0);
        assert!(mask.iter().any(|&byte| byte != 0));
    }

    #[test]
    fn proposal_candidate_mask_selects_frontier_with_fewest_candidates() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1], [1, 0]],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 8, 0);
        env.actions.push(Action {
            room_idx: 0,
            x: 2,
            y: 2,
        });
        let candidate = GeometryAction {
            geometry_idx: 0,
            x: 1,
            y: 2,
            door_direction: Direction::Right,
            door_x: 0,
            door_y: 0,
            door_kind: 0,
        };
        env.frontier.insert(
            door_location(0, 0, false),
            Frontier {
                direction: Direction::Right,
                dir_door_idx: 0,
                door_output_idx: -1,
                door_variant_idx: 0,
                room_part_idx: 0,
                component: 0,
                kind: 0,
                candidates: vec![candidate, candidate],
            },
        );
        env.frontier.insert(
            door_location(1, 0, false),
            Frontier {
                direction: Direction::Right,
                dir_door_idx: 0,
                door_output_idx: -1,
                door_variant_idx: 0,
                room_part_idx: 0,
                component: 0,
                kind: 0,
                candidates: vec![candidate],
            },
        );

        let door_variant_count = common.num_door_output_variants;
        let mask_byte_count = door_variant_count.div_ceil(8);
        let mut mask = vec![0; mask_byte_count];
        let mut frontier_idx = -1;
        let mut valid_counts = 0;
        env.proposal_candidate_mask(
            &common,
            door_variant_count,
            &mut frontier_idx,
            &mut mask,
            &mut valid_counts,
        );

        assert_eq!(frontier_idx, 1);
        assert!(valid_counts > 0);
        assert!(mask.iter().any(|&byte| byte != 0));
    }

    #[test]
    fn proposal_shortlist_resolves_sampled_cell_and_pads_to_quota() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1], [1, 0]],
                "missing_connections": []
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1], [1, 0]],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 8, 0);
        env.step(
            Action {
                room_idx: 0,
                x: 2,
                y: 2,
            },
            &common,
        );

        let frontier_count = Environment::max_frontiers(&common);
        let door_variant_count = common.num_door_output_variants;
        let mut mask = vec![0; door_variant_count.div_ceil(8)];
        let mut frontier_idx = -1;
        let mut valid_counts = 0;
        env.proposal_candidate_mask(
            &common,
            door_variant_count,
            &mut frontier_idx,
            &mut mask,
            &mut valid_counts,
        );
        assert!(frontier_count > 0);
        assert!(frontier_idx >= 0);
        let door_variant_idx = (0..door_variant_count)
            .find(|&idx| mask[idx / 8] & (1 << (idx % 8)) != 0)
            .expect("test setup should have a valid proposal candidate");
        let sampled_frontier_idx = [frontier_idx, -1];
        let sampled_door_variant_idx = [door_variant_idx as DoorVariantIdx, -1];
        let mut scratch = FeatureScratch::default();

        let (
            _pre_outcomes,
            candidates,
            candidate_frontier_idx,
            candidate_door_variant_idx,
            post_outcomes,
            door_matches,
            features,
            evaluated_count,
            rejected_count,
        ) = env
            .get_proposal_candidates_with_outcomes(
                &common,
                &sampled_frontier_idx,
                &sampled_door_variant_idx,
                2,
                &FeatureConfig::all_disabled(),
                FrontierNeighborAlgorithm::Nearest,
                1,
                4,
                &mut scratch,
            )
            .unwrap();

        assert_eq!(evaluated_count, 1);
        assert_eq!(rejected_count, 1);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidate_frontier_idx.len(), candidates.len());
        assert_eq!(candidate_door_variant_idx.len(), candidates.len());
        assert_eq!(post_outcomes.len(), candidates.len());
        assert_eq!(door_matches.len(), candidates.len());
        assert_eq!(features.len(), candidates.len());
    }

    #[test]
    fn outcome_consistency_rejects_known_to_unknown_and_known_changes() {
        use DoorValidOutcome::{Invalid, Unknown, Valid};

        assert!(check_outcome_transition_consistency(&[Unknown], &[Valid], "door", "test").is_ok());
        assert!(check_outcome_transition_consistency(&[Valid], &[Valid], "door", "test").is_ok());
        assert!(
            check_outcome_transition_consistency(&[Invalid], &[Invalid], "door", "test").is_ok()
        );
        assert!(
            check_outcome_transition_consistency(&[Valid], &[Unknown], "door", "test").is_err()
        );
        assert!(
            check_outcome_transition_consistency(&[Invalid], &[Unknown], "door", "test").is_err()
        );
        assert!(
            check_outcome_transition_consistency(&[Valid], &[Invalid], "door", "test").is_err()
        );
        assert!(
            check_outcome_transition_consistency(&[Invalid], &[Valid], "door", "test").is_err()
        );
    }

    #[test]
    fn introduces_invalid_outcome_only_detects_unknown_to_invalid() {
        use DoorValidOutcome::{Invalid, Unknown, Valid};

        assert!(introduces_invalid_outcome(
            &StepOutcomes {
                door_valid: vec![Unknown],
                connections_valid: vec![Valid],
                toilet_valid: Valid,
                phantoon_valid: Valid,
                toilet_crossed_room_idx: -1,
            },
            &StepOutcomes {
                door_valid: vec![Invalid],
                connections_valid: vec![Valid],
                toilet_valid: Valid,
                phantoon_valid: Valid,
                toilet_crossed_room_idx: -1,
            },
        ));
        assert!(!introduces_invalid_outcome(
            &StepOutcomes {
                door_valid: vec![Invalid],
                connections_valid: vec![Unknown],
                toilet_valid: Unknown,
                phantoon_valid: Unknown,
                toilet_crossed_room_idx: -1,
            },
            &StepOutcomes {
                door_valid: vec![Invalid],
                connections_valid: vec![Valid],
                toilet_valid: Unknown,
                phantoon_valid: Unknown,
                toilet_crossed_room_idx: -1,
            },
        ));
    }

    fn sparse_graph(
        edge_data: &[(usize, usize, i32)],
        vertex_count: usize,
    ) -> (Vec<FrontierEdge>, Vec<Vec<usize>>, Vec<usize>) {
        let mut edges = vec![];
        let mut incident_edges = vec![vec![]; vertex_count];
        let mut degrees = vec![0; vertex_count];
        for &(a, b, length_squared) in edge_data {
            let edge_idx = edges.len();
            edges.push(FrontierEdge {
                endpoints: [a, b],
                length_squared,
                active: true,
            });
            incident_edges[a].push(edge_idx);
            incident_edges[b].push(edge_idx);
            degrees[a] += 1;
            degrees[b] += 1;
        }
        (edges, incident_edges, degrees)
    }

    #[test]
    fn frontier_midpoints_distinguish_door_orientations() {
        assert_eq!(frontier_midpoint(door_location(2, 3, false)), (4, 7));
        assert_eq!(frontier_midpoint(door_location(2, 3, true)), (5, 6));
    }

    #[test]
    fn delaunay_neighbors_handle_tiny_collinear_and_cocircular_inputs() {
        assert_eq!(
            frontier_delaunay_neighbors(&[], 4),
            Vec::<Vec<usize>>::new()
        );
        assert_eq!(
            frontier_delaunay_neighbors(&[door_location(0, 0, false)], 4),
            vec![Vec::<usize>::new()]
        );
        assert_eq!(
            frontier_delaunay_neighbors(
                &[door_location(0, 0, false), door_location(1, 0, false)],
                4
            ),
            vec![vec![1], vec![0]]
        );

        let collinear = frontier_delaunay_neighbors(
            &[
                door_location(0, 0, false),
                door_location(1, 0, false),
                door_location(2, 0, false),
            ],
            4,
        );
        assert_eq!(collinear, vec![vec![1], vec![0, 2], vec![1]]);

        let cocircular = frontier_delaunay_neighbors(
            &[
                door_location(0, 0, false),
                door_location(1, 0, true),
                door_location(1, 1, false),
                door_location(0, 1, true),
            ],
            2,
        );
        assert_symmetric_neighbors(&cocircular, 2);
    }

    #[test]
    fn nearest_neighbors_keep_self_then_manhattan_nearest_frontiers() {
        let neighbors = frontier_nearest_neighbors(
            &[
                door_location(0, 0, false),
                door_location(2, 0, false),
                door_location(0, 1, false),
                door_location(1, 0, false),
            ],
            3,
            true,
        );
        assert_eq!(neighbors[0], vec![0, 2, 3]);
        assert_eq!(neighbors[1], vec![1, 3, 0]);

        let neighbors = frontier_nearest_neighbors(
            &[
                door_location(0, 0, false),
                door_location(2, 0, false),
                door_location(0, 1, false),
                door_location(1, 0, false),
            ],
            3,
            false,
        );
        assert_eq!(neighbors[0], vec![2, 3, 1]);
        assert_eq!(neighbors[1], vec![3, 0, 2]);

        let locations = [
            door_location(0, 0, false),
            door_location(2, 0, false),
            door_location(0, 1, false),
            door_location(1, 0, false),
        ];
        let mut single = vec![-1; locations.len()];
        write_single_frontier_nearest_neighbor(&locations, false, &mut single);
        assert_eq!(single, vec![2, 3, 0, 0]);

        write_single_frontier_nearest_neighbor(&locations[..1], false, &mut single[..1]);
        assert_eq!(single[0], -1);
    }

    #[test]
    fn pruning_prefers_high_degree_neighbors_then_long_edges() {
        let (mut edges, incident_edges, mut degrees) = sparse_graph(
            &[(0, 1, 100), (0, 2, 10), (0, 3, 1), (3, 4, 1), (3, 5, 1)],
            6,
        );
        prune_frontier_edges(&mut edges, &incident_edges, &mut degrees, 2);
        assert!(!edges[2].active);

        let (mut edges, incident_edges, mut degrees) =
            sparse_graph(&[(0, 1, 1), (0, 2, 100), (0, 3, 4)], 4);
        prune_frontier_edges(&mut edges, &incident_edges, &mut degrees, 2);
        assert!(!edges[1].active);

        let (mut edges, incident_edges, mut degrees) =
            sparse_graph(&[(0, 1, 1), (0, 2, 1), (0, 3, 1)], 4);
        prune_frontier_edges(&mut edges, &incident_edges, &mut degrees, 2);
        assert!(!edges[0].active);
    }

    #[test]
    fn environment_tracks_room_connections_physical_edges_and_clear() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "down", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (4, 4), 8, 0);

        env.step(
            Action {
                room_idx: 0,
                x: 0,
                y: 0,
            },
            &common,
        );
        let room0_part0 = env.room_part_component(&common, 0, 0);
        let room0_part1 = env.room_part_component(&common, 0, 1);
        assert!(env.scc_dag.can_reach(room0_part0, room0_part1));

        env.step(
            Action {
                room_idx: 1,
                x: 1,
                y: 0,
            },
            &common,
        );
        let room0_part0 = env.room_part_component(&common, 0, 0);
        let room0_part1 = env.room_part_component(&common, 0, 1);
        let room1_part0 = env.room_part_component(&common, 1, 0);
        assert_eq!(room0_part0, room1_part0);
        assert_eq!(env.scc_dag.component_count, 2);
        assert!(env.scc_dag.can_reach(room1_part0, room0_part1));

        env.clear(&common);
        assert_eq!(env.scc_dag.component_count, 0);
        assert!(env.active_room_parts.is_empty());
        assert!(
            env.room_part_component
                .iter()
                .all(|&component| component == NO_COMPONENT)
        );
        assert!(
            env.graph_distance
                .iter()
                .all(|&distance| distance == GraphDistance::MAX)
        );
    }

    #[test]
    fn active_save_refill_room_parts_track_placement_and_restore() {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"
            [
                {
                    "save": true,
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [
                        [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}]
                    ],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "refill": true,
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [
                        [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]
                    ],
                    "connections": [],
                    "missing_connections": []
                }
            ]
            "#,
        )
        .unwrap();
        let common = CommonData::new(rooms).unwrap();
        assert_eq!(common.save_room_part, vec![0]);
        assert_eq!(common.refill_room_part, vec![1]);
        let mut env = Environment::new(&common, (4, 4), 8, 0);

        env.step(
            Action {
                room_idx: 0,
                x: 0,
                y: 0,
            },
            &common,
        );
        assert_eq!(env.active_save_room_parts, vec![0]);
        assert!(env.active_refill_room_parts.is_empty());

        let candidate = Action {
            room_idx: 1,
            x: 1,
            y: 0,
        };
        let config = FeatureConfig::all();
        let expected_active_save_room_parts = env.active_save_room_parts.clone();
        let expected_active_refill_room_parts = env.active_refill_room_parts.clone();
        env.features_after_candidate(
            &common,
            candidate,
            &config,
            FrontierNeighborAlgorithm::Delaunay,
            4,
            4,
        );
        assert_eq!(env.active_save_room_parts, expected_active_save_room_parts);
        assert_eq!(
            env.active_refill_room_parts,
            expected_active_refill_room_parts
        );

        env.step(candidate, &common);
        assert_eq!(env.active_save_room_parts, vec![0]);
        assert_eq!(env.active_refill_room_parts, vec![1]);

        env.clear(&common);
        assert!(env.active_save_room_parts.is_empty());
        assert!(env.active_refill_room_parts.is_empty());
    }

    #[test]
    fn save_refill_utility_queries_consolidate_shared_frontier_context() {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"
            [{
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            }]
            "#,
        )
        .unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (4, 4), 8, 0);
        env.step(
            Action {
                room_idx: 0,
                x: 0,
                y: 0,
            },
            &common,
        );
        let config = FeatureConfig {
            frontier_mask: true,
            save_utility_query: true,
            refill_utility_query: true,
            ..FeatureConfig::all_disabled()
        };
        let features = env.features(&common, &config, FrontierNeighborAlgorithm::Delaunay, 4, 4);

        assert_eq!(features.save_refill_utility_query_room_part_idx, vec![0]);
        assert_eq!(features.save_refill_utility_query_target_mask, vec![0b1111]);
        assert_eq!(
            features.save_refill_utility_query_save_to_current_distance,
            vec![UNREACHABLE_DISTANCE]
        );
        assert_eq!(
            features.save_refill_utility_query_save_from_current_distance,
            vec![UNREACHABLE_DISTANCE]
        );
        assert_eq!(
            features.save_refill_utility_query_refill_to_current_distance,
            vec![UNREACHABLE_DISTANCE]
        );
        assert_eq!(
            features.save_refill_utility_query_refill_from_current_distance,
            vec![UNREACHABLE_DISTANCE]
        );
        assert_eq!(features.save_refill_utility_query_frontier, vec![0]);
    }

    #[test]
    fn save_refill_utility_queries_split_different_frontier_contexts() {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"
            [{
                "map": [[1, 0, 1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "left", "x": 2, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": [[0, 1], [1, 0]]
            }]
            "#,
        )
        .unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (6, 4), 8, 0);
        env.step(
            Action {
                room_idx: 0,
                x: 1,
                y: 1,
            },
            &common,
        );
        env.room_part_frontier_distance_cache
            .nearest_frontier_source[0] = 1;
        env.room_part_frontier_distance_cache
            .nearest_frontier_source_part[0] = 0;
        env.room_part_frontier_distance_cache
            .nearest_frontier_destination[0] = 1;
        env.room_part_frontier_distance_cache
            .nearest_frontier_destination_part[0] = 1;
        let config = FeatureConfig {
            frontier_mask: true,
            save_utility_query: true,
            refill_utility_query: true,
            ..FeatureConfig::all_disabled()
        };
        let features = env.features(&common, &config, FrontierNeighborAlgorithm::Delaunay, 4, 4);
        let rows_for_part_zero = features
            .save_refill_utility_query_room_part_idx
            .iter()
            .zip(&features.save_refill_utility_query_target_mask)
            .filter(|&(&room_part, _)| room_part == 0)
            .map(|(_, &target_mask)| target_mask)
            .collect::<Vec<_>>();

        assert_eq!(rows_for_part_zero.len(), 2);
        assert!(rows_for_part_zero.contains(&0b0101));
        assert!(rows_for_part_zero.contains(&0b1010));
    }

    #[test]
    fn environment_tracks_global_graph_distances() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "down", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 8, 0);

        env.step(
            Action {
                room_idx: 0,
                x: 0,
                y: 0,
            },
            &common,
        );
        let room0_part0 = Environment::room_part_idx(&common, 0, 0);
        let room0_part1 = Environment::room_part_idx(&common, 0, 1);
        assert_eq!(env.graph_distance(&common, room0_part0, room0_part1), 0);
        assert_eq!(
            env.graph_distance(&common, room0_part1, room0_part0),
            GraphDistance::MAX
        );

        env.step(
            Action {
                room_idx: 1,
                x: 1,
                y: 0,
            },
            &common,
        );
        let room1_part0 = Environment::room_part_idx(&common, 1, 0);
        let room1_part1 = Environment::room_part_idx(&common, 1, 1);
        assert_eq!(env.graph_distance(&common, room0_part0, room1_part0), 1);
        assert_eq!(env.graph_distance(&common, room1_part0, room0_part0), 1);
        assert_eq!(env.graph_distance(&common, room0_part0, room1_part1), 1);
        assert_eq!(
            env.graph_distance(&common, room1_part1, room0_part0),
            GraphDistance::MAX
        );

        env.step(
            Action {
                room_idx: 2,
                x: 2,
                y: 0,
            },
            &common,
        );
        let room2_part0 = Environment::room_part_idx(&common, 2, 0);
        assert_eq!(env.graph_distance(&common, room0_part0, room2_part0), 2);
        assert_eq!(
            env.graph_distance(&common, room2_part0, room0_part0),
            GraphDistance::MAX
        );
        assert_eq!(
            env.graph_distance(&common, room0_part1, room2_part0),
            GraphDistance::MAX
        );
    }

    #[test]
    fn single_attachment_distance_fast_path_matches_full_relaxation() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            },
            {
                "save": true,
                "refill": true,
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut fast_env = Environment::new(&common, (4, 4), 8, 0);
        let mut full_env = Environment::new(&common, (4, 4), 8, 0);
        let first_action = Action {
            room_idx: 0,
            x: 0,
            y: 0,
        };
        fast_env.step(first_action, &common);
        full_env.step(first_action, &common);

        fast_env.step(
            Action {
                room_idx: 1,
                x: 1,
                y: 0,
            },
            &common,
        );
        let room_part = Environment::room_part_idx(&common, 1, 0);
        let attached_part = Environment::room_part_idx(&common, 0, 0);
        full_env.add_room_part_distances(&common, 1, &[]);
        full_env.add_graph_distance_edge(&common, room_part, attached_part, 1);
        full_env.add_graph_distance_edge(&common, attached_part, room_part, 1);

        assert_eq!(fast_env.graph_distance, full_env.graph_distance);
        fast_env.assert_room_part_furthest_distance_cache_matches_slow(&common);
        fast_env.assert_room_part_save_distance_cache_matches_slow(&common);
        fast_env.assert_room_part_frontier_distance_cache_matches_slow(&common);
    }

    #[test]
    fn graph_distance_relaxation_updates_existing_parts_through_new_room() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 8, 0);

        env.step(
            Action {
                room_idx: 0,
                x: 0,
                y: 0,
            },
            &common,
        );
        env.step(
            Action {
                room_idx: 1,
                x: 2,
                y: 0,
            },
            &common,
        );
        let left_part = Environment::room_part_idx(&common, 0, 0);
        let right_part = Environment::room_part_idx(&common, 1, 0);
        assert_eq!(
            env.graph_distance(&common, left_part, right_part),
            GraphDistance::MAX
        );

        env.step(
            Action {
                room_idx: 2,
                x: 1,
                y: 0,
            },
            &common,
        );

        assert_eq!(env.graph_distance(&common, left_part, right_part), 2);
        assert_eq!(
            env.graph_distance(&common, right_part, left_part),
            GraphDistance::MAX
        );
    }

    #[test]
    fn graph_diameter_ignores_unreachable_pairs_and_zero_cost_room_edges() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "down", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 8, 0);

        assert_eq!(env.graph_diameter(), 0);
        env.step(
            Action {
                room_idx: 0,
                x: 0,
                y: 0,
            },
            &common,
        );
        assert_eq!(env.graph_diameter(), 0);
        env.step(
            Action {
                room_idx: 1,
                x: 1,
                y: 0,
            },
            &common,
        );
        assert_eq!(env.graph_diameter(), 1);
        env.step(
            Action {
                room_idx: 2,
                x: 2,
                y: 0,
            },
            &common,
        );
        assert_eq!(env.graph_diameter(), 2);
    }

    #[test]
    fn save_distances_sum_nearest_directed_distances_to_save_parts() {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"
            [
                {
                    "save": true,
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [
                        [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}]
                    ],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [
                        [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}],
                        [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}]
                    ],
                    "connections": [[0, 1], [1, 0]],
                    "missing_connections": []
                },
                {
                    "save": true,
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [
                        [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]
                    ],
                    "connections": [],
                    "missing_connections": []
                }
            ]
            "#,
        )
        .unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 8, 0);
        env.step_known(
            Action {
                room_idx: 0,
                x: 0,
                y: 0,
            },
            &common,
        );
        env.step_known(
            Action {
                room_idx: 1,
                x: 1,
                y: 0,
            },
            &common,
        );
        env.step_known(
            Action {
                room_idx: 2,
                x: 2,
                y: 0,
            },
            &common,
        );

        let (distance, mask) = env.save_distances(&common);

        assert_eq!(distance, vec![0.0, 2.0, 2.0, 0.0]);
        assert_eq!(mask, vec![1, 1, 1, 1]);
    }

    #[test]
    fn directed_save_distances_report_each_direction() {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"
            [
                {
                    "save": true,
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [
                        [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}]
                    ],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [
                        [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}],
                        [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}]
                    ],
                    "connections": [[0, 1]],
                    "missing_connections": [[1, 0]]
                },
                {
                    "save": true,
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [
                        [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]
                    ],
                    "connections": [],
                    "missing_connections": []
                }
            ]
            "#,
        )
        .unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 8, 0);
        env.step_known(
            Action {
                room_idx: 0,
                x: 0,
                y: 0,
            },
            &common,
        );
        env.step_known(
            Action {
                room_idx: 1,
                x: 1,
                y: 0,
            },
            &common,
        );
        env.step_known(
            Action {
                room_idx: 2,
                x: 2,
                y: 0,
            },
            &common,
        );

        let (to_room, to_room_mask, from_room, from_room_mask) =
            env.directed_save_distances(&common);

        assert_eq!(to_room, vec![0.0, 1.0, 1.0, 0.0]);
        assert_eq!(to_room_mask, vec![1, 1, 1, 1]);
        assert_eq!(from_room, vec![0.0, 1.0, 1.0, 0.0]);
        assert_eq!(from_room_mask, vec![1, 1, 1, 1]);
    }

    #[test]
    fn save_distances_mask_parts_without_reachable_save() {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"
            [
                {
                    "save": true,
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[]],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[]],
                    "connections": [],
                    "missing_connections": []
                }
            ]
            "#,
        )
        .unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 8, 0);
        env.step_known(
            Action {
                room_idx: 0,
                x: 0,
                y: 0,
            },
            &common,
        );
        env.step_known(
            Action {
                room_idx: 1,
                x: 2,
                y: 0,
            },
            &common,
        );

        let (distance, mask) = env.save_distances(&common);

        assert_eq!(distance, vec![0.0, 0.0]);
        assert_eq!(mask, vec![1, 0]);
    }

    #[test]
    fn room_part_furthest_distance_features_encode_furthest_finite_distances() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "down", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": [[0, 1], [1, 2], [2, 0]]
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 8, 0);
        let graph_size = common.room_part.len();
        for (from_part, to_part, distance) in
            [(0, 0, 0), (0, 1, 2), (1, 0, 1), (1, 1, 0), (1, 2, 3)]
        {
            env.set_graph_distance(graph_size, from_part, to_part, distance);
        }

        let (furthest_destination, furthest_source) =
            env.room_part_furthest_distance_features(&common);

        assert_eq!(furthest_destination, vec![3, 4, 0]);
        assert_eq!(furthest_source, vec![2, 3, 4]);
        env.assert_room_part_furthest_distance_cache_matches_slow(&common);
    }

    #[test]
    fn room_part_furthest_distance_cache_handles_decreased_max_distance() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "down", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": [[0, 1], [1, 2], [2, 0]]
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 8, 0);
        let graph_size = common.room_part.len();
        for (from_part, to_part, distance) in
            [(0, 0, 0), (0, 1, 4), (0, 2, 2), (1, 2, 4), (2, 2, 0)]
        {
            env.set_graph_distance(graph_size, from_part, to_part, distance);
        }
        env.set_graph_distance(graph_size, 0, 1, 1);
        env.set_graph_distance(graph_size, 1, 2, 3);

        let (furthest_destination, furthest_source) =
            env.room_part_furthest_distance_features(&common);

        assert_eq!(furthest_destination, vec![3, 4, 1]);
        assert_eq!(furthest_source, vec![1, 2, 4]);
        env.assert_room_part_furthest_distance_cache_matches_slow(&common);
    }

    #[test]
    fn room_part_save_distance_features_encode_directed_save_distances() {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"
            [
                {
                    "save": true,
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[]],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[]],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[]],
                    "connections": [],
                    "missing_connections": []
                }
            ]
            "#,
        )
        .unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 8, 0);
        let graph_size = common.room_part.len();
        env.active_room_parts = vec![0, 1, 2];
        for part in 0..graph_size {
            env.set_graph_distance(graph_size, part, part, 0);
        }
        env.set_graph_distance(graph_size, 1, 0, 2);
        env.set_graph_distance(graph_size, 0, 1, 4);
        env.set_graph_distance(graph_size, 2, 0, 5);
        let active_room_parts = env.active_room_parts.clone();
        env.room_part_save_distance_cache.add_save_part(
            &env.graph_distance,
            graph_size,
            &active_room_parts,
            0,
        );

        assert_eq!(
            env.room_part_save_distance_features(&common),
            (vec![1, 3, 6], vec![1, 5, 0])
        );
        env.assert_room_part_save_distance_cache_matches_slow(&common);
    }

    #[test]
    fn room_part_save_distance_cache_handles_distance_decreases_and_saturation() {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"
            [
                {
                    "save": true,
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[]],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "save": true,
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[]],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[]],
                    "connections": [],
                    "missing_connections": []
                }
            ]
            "#,
        )
        .unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 8, 0);
        let graph_size = common.room_part.len();
        env.active_room_parts = vec![0, 1, 2];
        for part in 0..graph_size {
            env.set_graph_distance(graph_size, part, part, 0);
        }
        env.set_graph_distance(graph_size, 2, 0, 250);
        env.set_graph_distance(graph_size, 0, 2, 10);
        env.set_graph_distance(graph_size, 2, 1, 20);
        env.set_graph_distance(graph_size, 1, 2, 30);
        let active_room_parts = env.active_room_parts.clone();
        env.room_part_save_distance_cache.add_save_part(
            &env.graph_distance,
            graph_size,
            &active_room_parts,
            0,
        );
        env.room_part_save_distance_cache.add_save_part(
            &env.graph_distance,
            graph_size,
            &active_room_parts,
            1,
        );

        assert_eq!(
            env.room_part_save_distance_features(&common),
            (vec![1, 1, 21], vec![1, 1, 11])
        );

        env.set_graph_distance(graph_size, 1, 2, 5);
        assert_eq!(
            env.room_part_save_distance_features(&common),
            (vec![1, 1, 21], vec![1, 1, 6])
        );

        env.set_graph_distance(graph_size, 2, 1, 250);
        env.set_graph_distance(graph_size, 1, 2, 250);
        assert_eq!(
            env.room_part_save_distance_features(&common),
            (vec![1, 1, 251], vec![1, 1, 11])
        );
        env.assert_room_part_save_distance_cache_matches_slow(&common);
    }

    #[test]
    fn known_finalized_distance_encoding_handles_all_states() {
        assert_eq!(encode_known_finalized_distance(7, 7), 9);
        assert_eq!(encode_known_finalized_distance(7, 8), 9);
        assert_eq!(
            encode_known_finalized_distance(8, 7),
            KNOWN_DISTANCE_UNKNOWN
        );
        assert_eq!(
            encode_known_finalized_distance(UNREACHABLE_DISTANCE, UNREACHABLE_DISTANCE),
            KNOWN_DISTANCE_UNREACHABLE
        );
        assert_eq!(
            encode_known_finalized_distance(UNREACHABLE_DISTANCE, 9),
            KNOWN_DISTANCE_UNKNOWN
        );
        assert_eq!(
            encode_known_finalized_distance(254, UNREACHABLE_DISTANCE),
            255
        );
    }

    #[test]
    fn features_include_known_finalized_reachable_directed_distances() {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"
            [
                {
                    "save": true,
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[]],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[]],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[]],
                    "connections": [],
                    "missing_connections": []
                }
            ]
            "#,
        )
        .unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 8, 0);
        let graph_size = common.room_part.len();
        env.active_room_parts = vec![0, 1];
        for part in 0..graph_size {
            env.set_graph_distance(graph_size, part, part, 0);
        }
        env.set_graph_distance(graph_size, 1, 0, 4);
        env.set_graph_distance(graph_size, 0, 1, 3);
        env.set_graph_distance(graph_size, 1, 2, 5);
        env.set_graph_distance(graph_size, 2, 1, 2);
        let active_room_parts = env.active_room_parts.clone();
        env.room_part_save_distance_cache.add_save_part(
            &env.graph_distance,
            graph_size,
            &active_room_parts,
            0,
        );
        env.room_part_frontier_distance_cache.add_frontier_part(
            &env.graph_distance,
            graph_size,
            &active_room_parts,
            2,
        );

        let features = env.features(
            &common,
            &FeatureConfig::all_disabled(),
            FrontierNeighborAlgorithm::Nearest,
            1,
            4,
        );

        assert_eq!(features.known_save_from_room_distance, vec![2, 6, 0]);
        assert_eq!(features.known_save_to_room_distance, vec![2, 0, 0]);
        assert_eq!(features.known_refill_from_room_distance, vec![1, 0, 0]);
        assert_eq!(features.known_refill_to_room_distance, vec![1, 0, 0]);
    }

    #[test]
    fn features_include_known_finalized_unreachable_directed_distances() {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"
            [
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[]],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[]],
                    "connections": [],
                    "missing_connections": []
                }
            ]
            "#,
        )
        .unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 8, 0);
        env.active_room_parts = vec![0];

        let features = env.features(
            &common,
            &FeatureConfig::all_disabled(),
            FrontierNeighborAlgorithm::Nearest,
            1,
            4,
        );

        assert_eq!(features.known_save_from_room_distance, vec![1, 0]);
        assert_eq!(features.known_save_to_room_distance, vec![1, 0]);
        assert_eq!(features.known_refill_from_room_distance, vec![1, 0]);
        assert_eq!(features.known_refill_to_room_distance, vec![1, 0]);
    }

    #[test]
    fn room_part_frontier_distance_features_encode_directed_frontier_distances() {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"
            [
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[]],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[]],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[]],
                    "connections": [],
                    "missing_connections": []
                }
            ]
            "#,
        )
        .unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 8, 0);
        let graph_size = common.room_part.len();
        env.active_room_parts = vec![0, 1, 2];
        for part in 0..graph_size {
            env.set_graph_distance(graph_size, part, part, 0);
        }
        env.set_graph_distance(graph_size, 1, 0, 2);
        env.set_graph_distance(graph_size, 0, 1, 4);
        env.set_graph_distance(graph_size, 2, 0, 5);
        env.frontier.insert(
            door_location(0, 0, false),
            Frontier {
                direction: Direction::Right,
                dir_door_idx: 0,
                door_output_idx: -1,
                door_variant_idx: 0,
                room_part_idx: 0,
                component: 0,
                kind: 0,
                candidates: vec![],
            },
        );
        let active_room_parts = env.active_room_parts.clone();
        env.room_part_frontier_distance_cache.add_frontier_part(
            &env.graph_distance,
            graph_size,
            &active_room_parts,
            0,
        );

        assert_eq!(
            env.room_part_frontier_distance_features(&common),
            (vec![1, 3, 6], vec![1, 5, 0])
        );
        env.assert_room_part_frontier_distance_cache_matches_slow(&common);
    }

    #[test]
    fn room_part_frontier_distance_cache_handles_removal_counts_and_saturation() {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"
            [
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[]],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[]],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[]],
                    "connections": [],
                    "missing_connections": []
                }
            ]
            "#,
        )
        .unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 8, 0);
        let graph_size = common.room_part.len();
        env.active_room_parts = vec![0, 1, 2];
        for part in 0..graph_size {
            env.set_graph_distance(graph_size, part, part, 0);
        }
        env.set_graph_distance(graph_size, 2, 0, 250);
        env.set_graph_distance(graph_size, 0, 2, 10);
        env.set_graph_distance(graph_size, 2, 1, 20);
        env.set_graph_distance(graph_size, 1, 2, 30);
        let active_room_parts = env.active_room_parts.clone();

        for (idx, frontier_part) in [(0, 0), (1, 1), (2, 1)] {
            env.frontier.insert(
                door_location(idx, 0, false),
                Frontier {
                    direction: Direction::Right,
                    dir_door_idx: 0,
                    door_output_idx: -1,
                    door_variant_idx: 0,
                    room_part_idx: frontier_part,
                    component: 0,
                    kind: 0,
                    candidates: vec![],
                },
            );
            env.room_part_frontier_distance_cache.add_frontier_part(
                &env.graph_distance,
                graph_size,
                &active_room_parts,
                frontier_part as usize,
            );
        }

        assert_eq!(
            env.room_part_frontier_distance_features(&common),
            (vec![1, 1, 21], vec![1, 1, 11])
        );

        env.set_graph_distance(graph_size, 1, 2, 5);
        assert_eq!(
            env.room_part_frontier_distance_features(&common),
            (vec![1, 1, 21], vec![1, 1, 6])
        );

        env.frontier.remove(&door_location(1, 0, false));
        env.room_part_frontier_distance_cache.remove_frontier_part(
            &env.graph_distance,
            graph_size,
            &active_room_parts,
            1,
        );
        assert_eq!(
            env.room_part_frontier_distance_features(&common),
            (vec![1, 1, 21], vec![1, 1, 6])
        );

        env.frontier.remove(&door_location(2, 0, false));
        env.room_part_frontier_distance_cache.remove_frontier_part(
            &env.graph_distance,
            graph_size,
            &active_room_parts,
            1,
        );
        assert_eq!(
            env.room_part_frontier_distance_features(&common),
            (vec![1, 0, 251], vec![1, 0, 11])
        );
        env.assert_room_part_frontier_distance_cache_matches_slow(&common);
    }

    #[test]
    fn room_part_frontier_distance_cache_tracks_nearest_identity_for_ties() {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"
            [
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[]],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[]],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[]],
                    "connections": [],
                    "missing_connections": []
                }
            ]
            "#,
        )
        .unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 8, 0);
        let graph_size = common.room_part.len();
        env.active_room_parts = vec![0, 1, 2];
        for part in 0..graph_size {
            env.set_graph_distance(graph_size, part, part, 0);
        }
        env.set_graph_distance(graph_size, 2, 0, 5);
        env.set_graph_distance(graph_size, 2, 1, 5);
        env.set_graph_distance(graph_size, 0, 2, 5);
        env.set_graph_distance(graph_size, 1, 2, 5);
        let active_room_parts = env.active_room_parts.clone();

        for frontier_part in [0, 1] {
            env.room_part_frontier_distance_cache.add_frontier_part(
                &env.graph_distance,
                graph_size,
                &active_room_parts,
                frontier_part,
            );
        }

        assert_eq!(
            env.room_part_frontier_distance_cache
                .nearest_frontier_destination_part[2],
            0
        );
        assert_eq!(
            env.room_part_frontier_distance_cache
                .nearest_frontier_source_part[2],
            0
        );
        assert_eq!(
            env.room_part_frontier_distance_features(&common),
            (vec![1, 1, 6], vec![1, 1, 6])
        );

        env.room_part_frontier_distance_cache.remove_frontier_part(
            &env.graph_distance,
            graph_size,
            &active_room_parts,
            1,
        );
        assert_eq!(
            env.room_part_frontier_distance_cache
                .nearest_frontier_destination_part[2],
            0
        );
        assert_eq!(
            env.room_part_frontier_distance_features(&common),
            (vec![1, 0, 6], vec![1, 0, 6])
        );
        env.room_part_frontier_distance_cache.add_frontier_part(
            &env.graph_distance,
            graph_size,
            &active_room_parts,
            1,
        );
        env.room_part_frontier_distance_cache.remove_frontier_part(
            &env.graph_distance,
            graph_size,
            &active_room_parts,
            0,
        );
        assert_eq!(
            env.room_part_frontier_distance_cache
                .nearest_frontier_destination_part[2],
            1
        );
        assert_eq!(
            env.room_part_frontier_distance_cache
                .nearest_frontier_source_part[2],
            1
        );
        assert_eq!(
            env.room_part_frontier_distance_features(&common),
            (vec![0, 1, 6], vec![0, 1, 6])
        );

        env.room_part_frontier_distance_cache.clear();
        assert!(
            env.room_part_frontier_distance_cache
                .nearest_frontier_destination_part
                .iter()
                .all(|&part| part == RoomPartIdx::MAX)
        );
        assert!(
            env.room_part_frontier_distance_cache
                .nearest_frontier_source_part
                .iter()
                .all(|&part| part == RoomPartIdx::MAX)
        );
    }

    #[test]
    fn door_match_counts_include_unmatched_row_and_column() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 4), 8, 0);

        env.step(
            Action {
                room_idx: 0,
                x: 0,
                y: 0,
            },
            &common,
        );
        env.step(
            Action {
                room_idx: 1,
                x: 1,
                y: 0,
            },
            &common,
        );
        env.step(
            Action {
                room_idx: 2,
                x: 0,
                y: 2,
            },
            &common,
        );
        env.step(
            Action {
                room_idx: 3,
                x: 3,
                y: 2,
            },
            &common,
        );

        let mut horizontal_counts = vec![0; 9];
        let mut vertical_counts = vec![0; 1];
        env.add_door_match_counts(&common, &mut horizontal_counts, &mut vertical_counts);

        assert_eq!(horizontal_counts, vec![1, 0, 0, 0, 0, 1, 0, 1, 0]);
        assert_eq!(vertical_counts, vec![0]);

        let mut left = vec![-1; 2];
        let mut right = vec![-1; 2];
        let mut up = vec![];
        let mut down = vec![];
        env.write_door_matches(&mut left, &mut right, &mut up, &mut down);

        assert_eq!(left, vec![0, -1]);
        assert_eq!(right, vec![0, -1]);
        assert_eq!(up, Vec::<i16>::new());
        assert_eq!(down, Vec::<i16>::new());
    }

    #[test]
    fn strongly_connected_room_has_no_missing_connection_outcomes() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "down", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1], [1, 0]],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (4, 4), 8, 0);

        env.step(
            Action {
                room_idx: 0,
                x: 0,
                y: 0,
            },
            &common,
        );

        let outcomes = env.outcomes(&common);
        assert!(outcomes.connections_valid.is_empty());
    }

    #[test]
    fn finish_marks_unresolved_connection_outcomes_invalid() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "down", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (4, 4), 8, 0);

        env.step(
            Action {
                room_idx: 0,
                x: 0,
                y: 0,
            },
            &common,
        );

        let outcomes = env.outcomes(&common);
        assert_eq!(outcomes.connections_valid.len(), 1);
        assert!(matches!(
            outcomes.connections_valid.as_slice(),
            [DoorValidOutcome::Invalid]
        ));

        env.finish();
        let outcomes = env.outcomes(&common);
        assert!(matches!(
            outcomes.connections_valid.as_slice(),
            [DoorValidOutcome::Invalid]
        ));
        let features = env.features(
            &common,
            &FeatureConfig {
                missing_connect_query: true,
                ..FeatureConfig::all_disabled()
            },
            FrontierNeighborAlgorithm::Delaunay,
            1,
            4,
        );
        assert!(features.missing_connect_query_connection_idx.is_empty());
    }

    #[test]
    fn missing_connect_distances_mask_unreachable_connections() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "down", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": [[0, 1], [1, 2], [2, 0]]
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (4, 4), 8, 0);
        let part0 = usize::from(Environment::room_part_idx(&common, 0, 0));
        let part1 = usize::from(Environment::room_part_idx(&common, 0, 1));
        let graph_size = common.room_part.len();
        env.graph_distance[part0 * graph_size + part1] = 7;

        let (distance, mask) = env.missing_connect_distances(&common);

        assert_eq!(distance, vec![7.0, 0.0, 0.0]);
        assert_eq!(mask, vec![1, 0, 0]);
    }

    #[test]
    fn connection_outcome_is_invalid_when_frontier_merge_cannot_make_path() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "down", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (4, 4), 8, 0);

        env.step(
            Action {
                room_idx: 0,
                x: 0,
                y: 0,
            },
            &common,
        );
        assert!(matches!(
            env.outcomes(&common).connections_valid.as_slice(),
            [DoorValidOutcome::Invalid]
        ));

        env.step(
            Action {
                room_idx: 1,
                x: 1,
                y: 0,
            },
            &common,
        );

        let outcomes = env.outcomes(&common);
        assert!(matches!(
            outcomes.connections_valid.as_slice(),
            [DoorValidOutcome::Invalid]
        ));
    }

    #[test]
    fn empty_frontiers_do_not_make_connection_outcomes_unknown() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "down", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "up", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (4, 1), 8, 0);

        env.step(
            Action {
                room_idx: 0,
                x: 0,
                y: 0,
            },
            &common,
        );

        let outcomes = env.outcomes(&common);
        assert!(matches!(
            outcomes.connections_valid.as_slice(),
            [DoorValidOutcome::Invalid]
        ));
    }

    #[test]
    fn known_steps_rebuild_episode_and_finish_environment() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "down", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (4, 4), 8, 0);

        env.step(
            Action {
                room_idx: 1,
                x: 2,
                y: 2,
            },
            &common,
        );

        let actions = [
            Action {
                room_idx: 0,
                x: 0,
                y: 0,
            },
            Action {
                room_idx: 1,
                x: 1,
                y: 0,
            },
        ];
        env.clear(&common);
        for action in actions {
            env.step_known(action, &common);
        }
        env.finish();

        assert_eq!(env.actions(), actions);
        assert!(env.room_used[0]);
        assert!(env.room_used[1]);
        assert_eq!(env.room_x[1], 1);
        assert_eq!(env.room_y[1], 0);

        let outcomes = env.outcomes(&common);
        assert!(matches!(
            outcomes.connections_valid.as_slice(),
            [DoorValidOutcome::Invalid]
        ));
    }

    #[test]
    fn features_after_candidate_match_direct_step() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "down", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": [[0, 1], [1, 2], [2, 0]]
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        assert_eq!(Environment::max_frontiers(&common), 3);
        let mut env = Environment::new(&common, (4, 4), 8, 0);
        env.step(
            Action {
                room_idx: 0,
                x: 0,
                y: 0,
            },
            &common,
        );
        let candidate = Action {
            room_idx: 1,
            x: 1,
            y: 0,
        };
        let config = FeatureConfig::all();
        let expected_actions = env.actions.clone();
        let expected_frontier = env.frontier.clone();
        let expected_room_used = env.room_used.clone();
        let expected_connection_variant_unused_count = env.connection_variant_unused_count.clone();
        let expected_room_part_component = env.room_part_component.clone();
        let expected_scc_dag = env.scc_dag.clone();
        let expected_active_room_parts = env.active_room_parts.clone();
        let expected_graph_distance = env.graph_distance.clone();
        let expected_frontier_count =
            env.feature_frontier_count_after_candidate(candidate, &common);
        let simulated = env.features_after_candidate(
            &common,
            candidate,
            &config,
            FrontierNeighborAlgorithm::Delaunay,
            4,
            4,
        );
        assert_eq!(
            simulated.frontier.len() / FEATURE_FRONTIER_WIDTH,
            expected_frontier_count
        );
        assert_eq!(env.actions, expected_actions);
        assert_eq!(env.frontier, expected_frontier);
        assert_eq!(env.room_used, expected_room_used);
        assert_eq!(env.room_x[candidate.room_idx as usize], candidate.x);
        assert_eq!(env.room_y[candidate.room_idx as usize], candidate.y);
        assert_eq!(
            env.connection_variant_unused_count,
            expected_connection_variant_unused_count
        );
        assert_eq!(env.room_part_component, expected_room_part_component);
        assert_eq!(env.scc_dag, expected_scc_dag);
        assert_eq!(env.active_room_parts, expected_active_room_parts);
        assert_eq!(env.graph_distance, expected_graph_distance);
        env.assert_room_part_furthest_distance_cache_matches_slow(&common);
        env.step(candidate, &common);
        assert_eq!(
            simulated,
            env.features(&common, &config, FrontierNeighborAlgorithm::Delaunay, 4, 4)
        );
        assert_eq!(env.occupancy[0], 1);
        assert_eq!(env.occupancy[1], 1);
        assert_eq!(simulated.frontier_occupancy, vec![0, 12, 192, 0]);

        let dummy_candidate = Action {
            room_idx: common.room.len() as RoomIdx,
            x: 0,
            y: 0,
        };
        let simulated = env.features_after_candidate(
            &common,
            dummy_candidate,
            &config,
            FrontierNeighborAlgorithm::Delaunay,
            4,
            4,
        );
        env.step(dummy_candidate, &common);
        assert_eq!(
            simulated,
            env.features(&common, &config, FrontierNeighborAlgorithm::Delaunay, 4, 4)
        );
    }

    #[test]
    fn door_frontier_indices_track_unmatched_placed_doors() {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"
            [
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}]],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]],
                    "connections": [],
                    "missing_connections": []
                }
            ]
            "#,
        )
        .unwrap();
        let common = CommonData::new(rooms).unwrap();
        let config = FeatureConfig::all();
        let mut env = Environment::new(&common, (4, 4), 8, 0);

        env.step(
            Action {
                room_idx: 0,
                x: 0,
                y: 0,
            },
            &common,
        );
        let features = env.features(&common, &config, FrontierNeighborAlgorithm::Delaunay, 4, 4);
        assert_eq!(features.frontier.len() / FEATURE_FRONTIER_WIDTH, 1);
        let placed_door_output_idx = common
            .door_output
            .iter()
            .position(|output| output.room_idx == 0)
            .unwrap() as i16;
        let placed_door_variant_idx = common.door_variant_idx(
            common.room[0].connection_variant_idx,
            Direction::Right,
            0,
            0,
            0,
        );
        assert_eq!(features.row_door_output_idx, vec![placed_door_output_idx]);
        assert_eq!(
            features.frontier_door_variant,
            vec![placed_door_variant_idx]
        );

        env.step(
            Action {
                room_idx: 1,
                x: 1,
                y: 0,
            },
            &common,
        );
        let features = env.features(&common, &config, FrontierNeighborAlgorithm::Delaunay, 4, 4);
        assert!(features.row_door_output_idx.is_empty());
    }

    #[test]
    fn outcomes_after_candidate_restores_graph_distances() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (4, 4), 8, 0);
        env.step(
            Action {
                room_idx: 0,
                x: 0,
                y: 0,
            },
            &common,
        );
        let expected_graph_distance = env.graph_distance.clone();
        let expected_active_room_parts = env.active_room_parts.clone();

        env.outcomes_after_candidate(
            &common,
            Action {
                room_idx: 1,
                x: 1,
                y: 0,
            },
        );

        assert_eq!(env.active_room_parts, expected_active_room_parts);
        assert_eq!(env.graph_distance, expected_graph_distance);
        env.assert_room_part_furthest_distance_cache_matches_slow(&common);
    }

    #[test]
    fn feature_outcomes_match_after_candidate_and_committed_step() {
        let common = feature_outcome_test_common();
        let actions = [
            Action {
                room_idx: 1,
                x: 1,
                y: 0,
            },
            Action {
                room_idx: common.room.len() as RoomIdx,
                x: 0,
                y: 0,
            },
        ];

        for action in actions {
            let mut lookahead_env = feature_outcome_test_env(&common);
            let expected = lookahead_env.outcomes_after_candidate(&common, action);

            let mut stepped_env = feature_outcome_test_env(&common);
            stepped_env.step(action, &common);
            let actual = stepped_env.feature_outcomes(&common);

            assert_feature_outcomes_eq(&expected, &actual);
        }
    }

    #[test]
    fn features_do_not_depend_on_frontier_candidate_lists() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "down", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": [[0, 1], [1, 2], [2, 0]]
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let config = FeatureConfig::all();
        let actions = [
            Action {
                room_idx: 0,
                x: 0,
                y: 0,
            },
            Action {
                room_idx: 1,
                x: 1,
                y: 0,
            },
        ];
        let mut full_env = Environment::new(&common, (4, 4), 8, 0);
        let mut known_env = Environment::new(&common, (4, 4), 8, 0);

        for action in actions {
            full_env.step(action, &common);
            known_env.step_known(action, &common);
            assert_eq!(
                full_env.features(&common, &config, FrontierNeighborAlgorithm::Delaunay, 4, 4),
                known_env.features(&common, &config, FrontierNeighborAlgorithm::Delaunay, 4, 4)
            );
        }
    }

    #[test]
    fn finished_outcomes_do_not_depend_on_frontier_candidate_lists() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "down", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": [[0, 1], [1, 2], [2, 0]]
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let actions = [
            Action {
                room_idx: 0,
                x: 0,
                y: 0,
            },
            Action {
                room_idx: 1,
                x: 1,
                y: 0,
            },
        ];
        let mut full_env = Environment::new(&common, (4, 4), 8, 0);
        let mut replay_env = Environment::new(&common, (4, 4), 8, 0);

        for action in actions {
            full_env.step(action, &common);
        }
        full_env.finish();
        replay_env.clear(&common);
        for action in actions {
            replay_env.step_known(action, &common);
        }
        replay_env.finish();

        let full_outcomes = full_env.outcomes(&common);
        let replay_outcomes = replay_env.outcomes(&common);
        assert!(full_outcomes.door_valid == replay_outcomes.door_valid);
        assert!(full_outcomes.connections_valid == replay_outcomes.connections_valid);
    }

    #[test]
    fn feature_config_validates_dependencies() {
        assert!(
            FeatureConfig {
                frontier_position: true,
                ..FeatureConfig::all_disabled()
            }
            .validate()
            .is_err()
        );
        assert!(
            FeatureConfig {
                frontier_neighbor_flags: true,
                ..FeatureConfig::all_disabled()
            }
            .validate()
            .is_err()
        );
        assert!(
            FeatureConfig {
                frontier_neighbor_position_embedding: true,
                ..FeatureConfig::all_disabled()
            }
            .validate()
            .is_err()
        );
        assert!(
            FeatureConfig {
                global_room_position: true,
                ..FeatureConfig::all_disabled()
            }
            .validate()
            .is_err()
        );
        assert!(FeatureConfig::all_disabled().validate().is_ok());
        assert!(FeatureConfig::all().validate().is_ok());
        let err = serde_json::from_str::<FeatureConfig>(
            r#"{
                "inventory": false,
                "room_position": false,
                "frontier_mask": false,
                "frontier_position": false,
                "frontier_orientation": false,
                "frontier_kind": false,
                "frontier_occupancy": false,
                "frontier_neighbor": false,
                "frontier_neighbor_position_embedding": false,
                "frontier_neighbor_flags": false,
                "connection_reachability": false
            }"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("missing field"));
    }

    #[test]
    fn features_include_missing_connection_reachability() {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"
            [{
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"id": 0, "direction": "down", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            }]
            "#,
        )
        .unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (4, 4), 8, 0);
        env.step(
            Action {
                room_idx: 0,
                x: 0,
                y: 0,
            },
            &common,
        );
        let config = FeatureConfig {
            frontier_mask: true,
            connection_reachability: true,
            frontier_connection_reachability: true,
            missing_connect_query: true,
            ..FeatureConfig::all_disabled()
        };
        let features = env.features(&common, &config, FrontierNeighborAlgorithm::Delaunay, 1, 4);
        assert_eq!(features.connection_reachability, vec![0]);
        assert_eq!(features.frontier_connection_reachability, vec![1, 2]);
        assert_eq!(features.missing_connect_query_connection_idx, vec![0]);
        assert_eq!(features.missing_connect_query_source_frontier, vec![0]);
        assert_eq!(features.missing_connect_query_target_frontier, vec![1]);
        assert_eq!(features.missing_connect_query_source_distance, vec![0]);
        assert_eq!(features.missing_connect_query_target_distance, vec![0]);
        assert_eq!(
            features.missing_connect_query_current_distance,
            vec![UNREACHABLE_DISTANCE]
        );
    }

    #[test]
    fn disabled_features_skip_candidate_simulation() {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"[{"map": [[1]], "toilet_crossing_x": [], "doors": [], "connections": [], "missing_connections": []}]"#,
        )
        .unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (4, 4), 8, 0);
        let features = env.features_after_candidate(
            &common,
            Action {
                room_idx: 0,
                x: 0,
                y: 0,
            },
            &FeatureConfig::all_disabled(),
            FrontierNeighborAlgorithm::Delaunay,
            4,
            4,
        );
        assert_eq!(features, Features::default());
    }

    fn toilet_outcome_test_common() -> CommonData {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"
            [
                {
                    "map": [[1], [1]],
                    "toilet_crossing_x": [],
                    "doors": [],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1], [1]],
                    "toilet_crossing_x": [],
                    "doors": [],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1], [1], [0], [0], [0], [0], [0], [0], [1], [1]],
                    "toilet_crossing_x": [],
                    "special_type": "toilet",
                    "doors": [],
                    "connections": [],
                    "missing_connections": []
                }
            ]
            "#,
        )
        .unwrap();
        CommonData::new(rooms).unwrap()
    }

    #[test]
    fn toilet_outcome_is_valid_without_toilet_room() {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"[{"map": [[1]], "toilet_crossing_x": [], "doors": [], "connections": [], "missing_connections": []}]"#,
        )
        .unwrap();
        let common = CommonData::new(rooms).unwrap();
        let env = Environment::new(&common, (8, 12), 8, 0);

        assert_eq!(env.outcomes(&common).toilet_valid, DoorValidOutcome::Valid);
        assert_eq!(env.outcomes(&common).toilet_crossed_room_idx, -1);
    }

    #[test]
    fn toilet_outcome_requires_exactly_one_crossing_at_finish() {
        let common = toilet_outcome_test_common();
        let mut env = Environment::new(&common, (8, 12), 8, 0);
        env.step_known(
            Action {
                room_idx: 2,
                x: 0,
                y: 0,
            },
            &common,
        );
        assert_eq!(
            env.outcomes(&common).toilet_valid,
            DoorValidOutcome::Unknown
        );
        env.finish();
        assert_eq!(
            env.outcomes(&common).toilet_valid,
            DoorValidOutcome::Invalid
        );

        let mut env = Environment::new(&common, (8, 12), 8, 0);
        env.step_known(
            Action {
                room_idx: 2,
                x: 0,
                y: 0,
            },
            &common,
        );
        env.step_known(
            Action {
                room_idx: 0,
                x: 0,
                y: 2,
            },
            &common,
        );
        assert_eq!(
            env.outcomes(&common).toilet_valid,
            DoorValidOutcome::Unknown
        );
        env.finish();
        let outcomes = env.outcomes(&common);
        assert_eq!(outcomes.toilet_valid, DoorValidOutcome::Valid);
        assert_eq!(outcomes.toilet_crossed_room_idx, 0);
    }

    #[test]
    fn toilet_outcome_is_invalid_after_second_crossing() {
        let common = toilet_outcome_test_common();
        let mut env = Environment::new(&common, (8, 12), 8, 0);
        env.step_known(
            Action {
                room_idx: 2,
                x: 0,
                y: 0,
            },
            &common,
        );
        env.step_known(
            Action {
                room_idx: 0,
                x: 0,
                y: 2,
            },
            &common,
        );

        let outcomes = env
            .outcomes_after_candidate(
                &common,
                Action {
                    room_idx: 1,
                    x: 0,
                    y: 4,
                },
            )
            .step_outcomes;
        assert_eq!(outcomes.toilet_valid, DoorValidOutcome::Invalid);

        env.step_known(
            Action {
                room_idx: 1,
                x: 0,
                y: 4,
            },
            &common,
        );
        assert_eq!(
            env.outcomes(&common).toilet_valid,
            DoorValidOutcome::Invalid
        );
        assert_eq!(env.outcomes(&common).toilet_crossed_room_idx, -1);
    }

    #[test]
    fn toilet_outcome_ignores_dummy_candidate() {
        let common = toilet_outcome_test_common();
        let mut env = Environment::new(&common, (8, 12), 8, 0);
        env.step_known(
            Action {
                room_idx: 2,
                x: 0,
                y: 0,
            },
            &common,
        );

        let outcomes = env
            .outcomes_after_candidate(
                &common,
                Action {
                    room_idx: common.room.len() as RoomIdx,
                    x: 0,
                    y: 0,
                },
            )
            .step_outcomes;
        assert_eq!(outcomes.toilet_valid, DoorValidOutcome::Invalid);
        assert_eq!(outcomes.toilet_crossed_room_idx, -1);
    }

    fn phantoon_outcome_test_common() -> CommonData {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"
            [
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[
                        {"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0},
                        {"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}
                    ]],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[
                        {"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0},
                        {"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}
                    ]],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "special_type": "phantoon_boss",
                    "doors": [[{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}]],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "special_type": "phantoon_map",
                    "doors": [[{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]],
                    "connections": [],
                    "missing_connections": []
                }
            ]
            "#,
        )
        .unwrap();
        CommonData::new(rooms).unwrap()
    }

    fn phantoon_outcome_dead_end_test_common() -> CommonData {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"
            [
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "special_type": "phantoon_boss",
                    "doors": [[{"id": 0, "direction": "right", "x": 0, "y": 0, "kind": 0}]],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "special_type": "phantoon_map",
                    "doors": [[{"id": 0, "direction": "left", "x": 0, "y": 0, "kind": 0}]],
                    "connections": [],
                    "missing_connections": []
                }
            ]
            "#,
        )
        .unwrap();
        CommonData::new(rooms).unwrap()
    }

    #[test]
    fn phantoon_outcome_is_valid_without_special_rooms() {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"[{"map": [[1]], "toilet_crossing_x": [], "doors": [], "connections": [], "missing_connections": []}]"#,
        )
        .unwrap();
        let common = CommonData::new(rooms).unwrap();
        let env = Environment::new(&common, (4, 4), 8, 0);

        assert_eq!(
            env.outcomes(&common).phantoon_valid,
            DoorValidOutcome::Valid
        );
    }

    #[test]
    fn phantoon_outcome_requires_placed_rooms_at_finish() {
        let common = phantoon_outcome_test_common();
        let mut env = Environment::new(&common, (8, 4), 8, 0);

        assert_eq!(
            env.outcomes(&common).phantoon_valid,
            DoorValidOutcome::Unknown
        );
        env.finish();
        assert_eq!(
            env.outcomes(&common).phantoon_valid,
            DoorValidOutcome::Invalid
        );
    }

    #[test]
    fn phantoon_outcome_accepts_same_neighbor_room() {
        let common = phantoon_outcome_test_common();
        let mut env = Environment::new(&common, (8, 4), 8, 0);
        env.step_known(
            Action {
                room_idx: 0,
                x: 2,
                y: 1,
            },
            &common,
        );
        env.step_known(
            Action {
                room_idx: 2,
                x: 1,
                y: 1,
            },
            &common,
        );
        assert_eq!(
            env.outcomes(&common).phantoon_valid,
            DoorValidOutcome::Unknown
        );
        env.step_known(
            Action {
                room_idx: 3,
                x: 3,
                y: 1,
            },
            &common,
        );

        assert_eq!(
            env.outcomes(&common).phantoon_valid,
            DoorValidOutcome::Valid
        );
    }

    #[test]
    fn phantoon_outcome_rejects_different_neighbor_rooms() {
        let common = phantoon_outcome_test_common();
        let mut env = Environment::new(&common, (8, 4), 8, 0);
        env.step_known(
            Action {
                room_idx: 0,
                x: 2,
                y: 1,
            },
            &common,
        );
        env.step_known(
            Action {
                room_idx: 1,
                x: 4,
                y: 1,
            },
            &common,
        );
        env.step_known(
            Action {
                room_idx: 2,
                x: 1,
                y: 1,
            },
            &common,
        );
        env.step_known(
            Action {
                room_idx: 3,
                x: 5,
                y: 1,
            },
            &common,
        );

        assert_eq!(
            env.outcomes(&common).phantoon_valid,
            DoorValidOutcome::Invalid
        );
    }

    #[test]
    fn phantoon_outcome_rejects_partial_match_without_neighbor_frontier() {
        let common = phantoon_outcome_dead_end_test_common();
        let mut env = Environment::new(&common, (8, 4), 8, 0);
        env.step_known(
            Action {
                room_idx: 0,
                x: 2,
                y: 1,
            },
            &common,
        );
        env.step_known(
            Action {
                room_idx: 1,
                x: 1,
                y: 1,
            },
            &common,
        );

        assert_eq!(
            env.outcomes(&common).phantoon_valid,
            DoorValidOutcome::Invalid
        );
    }
}
