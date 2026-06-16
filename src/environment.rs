use bitvec::vec::BitVec;
use delaunator::{EMPTY, Point, next_halfedge, triangulate};
use hashbrown::HashMap;
use rand::SeedableRng;
use rand::prelude::*;
use serde::Deserialize;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::Instant;

use crate::common::{
    Action, CommonData, ConnectionVariantIdx, Coord, DirDoorIdx, Direction, DoorKind, DoorLocation,
    DoorValidOutcome, DoorVariantIdx, FrontierIdx, GeometryData, GeometryIdx, GraphDistance,
    NUM_DIRS, PartIdx, RoomIdx, RoomPartIdx, get_behind_door_position,
};
use crate::engine::{ProfileMetric, profile_enabled, record_profile_metric};
use crate::scc_dag::SccDag;

const NO_COMPONENT: usize = usize::MAX;
const UNREACHABLE_DISTANCE: GraphDistance = GraphDistance::MAX;
pub const FEATURE_FRONTIER_WIDTH: usize = 5;

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
fn introduces_invalid_outcome(before: &PreliminaryOutcomes, after: &PreliminaryOutcomes) -> bool {
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
}

enum CandidateOutcome {
    Clean(PreliminaryOutcomes, Vec<i16>, Features),
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
    dir_door_idx: DirDoorIdx,
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
pub struct PreliminaryOutcomes {
    // For each door, whether it is connected to another door.
    pub door_valid: Vec<DoorValidOutcome>,
    // For each connection, whether its destination can reach its source.
    pub connections_valid: Vec<DoorValidOutcome>,
    // Whether the Toilet crosses exactly one room.
    pub toilet_valid: DoorValidOutcome,
    // Concrete room crossed by the Toilet when exactly one non-Toilet room crosses it.
    pub toilet_crossed_room_idx: i16,
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
    pub frontier_occupancy: bool,
    pub frontier_neighbor: bool,
    pub frontier_neighbor_position_embedding: bool,
    pub frontier_neighbor_flags: bool,
    pub connection_reachability: bool,
    pub frontier_connection_reachability: bool,
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
            && !self.has_frontier_features()
    }

    pub fn has_frontier_features(&self) -> bool {
        self.frontier_mask
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if (self.frontier_position
            || self.frontier_orientation
            || self.frontier_kind
            || self.frontier_occupancy
            || self.frontier_neighbor
            || self.frontier_connection_reachability)
            && !self.frontier_mask
        {
            return Err("frontier features require frontier_mask");
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
            frontier_occupancy: true,
            frontier_neighbor: true,
            frontier_neighbor_position_embedding: true,
            frontier_neighbor_flags: true,
            connection_reachability: true,
            frontier_connection_reachability: true,
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
            frontier_occupancy: false,
            frontier_neighbor: false,
            frontier_neighbor_position_embedding: false,
            frontier_neighbor_flags: false,
            connection_reachability: false,
            frontier_connection_reachability: false,
            toilet_crossed_room: false,
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Features {
    pub inventory: Vec<u8>,
    pub room_x: Vec<Coord>,
    pub room_y: Vec<Coord>,
    pub room_placed: Vec<u8>,
    pub room_part_furthest_destination: Vec<u8>,
    pub room_part_furthest_source: Vec<u8>,
    pub room_part_save_distance: Vec<u8>,
    pub room_part_refill_distance: Vec<u8>,
    pub room_part_frontier_distance: Vec<u8>,
    // mask, x, y, vertical, kind
    pub frontier: Vec<i8>,
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
    // Concrete room crossed by the Toilet when exactly one non-Toilet room crosses it.
    pub toilet_crossed_room_idx: Vec<i16>,
}

fn frontier_midpoint(location: DoorLocation) -> (i16, i16) {
    if location.vertical() {
        (i16::from(location.x()) * 2 + 1, i16::from(location.y()) * 2)
    } else {
        (i16::from(location.x()) * 2, i16::from(location.y()) * 2 + 1)
    }
}

fn frontier_delaunay_neighbors(locations: &[DoorLocation], max_degree: usize) -> Vec<Vec<usize>> {
    let midpoints = locations
        .iter()
        .copied()
        .map(frontier_midpoint)
        .collect::<Vec<_>>();
    let points = midpoints
        .iter()
        .map(|&(x, y)| Point {
            x: f64::from(x),
            y: f64::from(y),
        })
        .collect::<Vec<_>>();
    let mut edges = vec![];
    let mut incident_edges = vec![vec![]; locations.len()];
    let mut degrees = vec![0; locations.len()];
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

    prune_frontier_edges(&mut edges, &incident_edges, &mut degrees, max_degree);

    let mut neighbors = vec![vec![]; locations.len()];
    for edge in edges.into_iter().filter(|edge| edge.active) {
        let [a, b] = edge.endpoints;
        neighbors[a].push(b);
        neighbors[b].push(a);
    }
    for row in &mut neighbors {
        row.sort_unstable();
        debug_assert!(row.len() <= max_degree);
    }
    neighbors
}

fn frontier_nearest_neighbors(
    locations: &[DoorLocation],
    neighbor_count: usize,
    include_self: bool,
) -> Vec<Vec<usize>> {
    let mut rows = Vec::with_capacity(locations.len());
    let mut neighbors = vec![usize::MAX; neighbor_count];
    let mut neighbor_keys = vec![(Coord::MAX, usize::MAX, usize::MAX); neighbor_count];
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
        rows.push(neighbors[..count].to_vec());
    }
    rows
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
    graph_distance: Vec<GraphDistance>,
    room_part_furthest_distance_cache: RoomPartFurthestDistanceCache,
    room_part_save_distance_cache: RoomPartSaveDistanceCache,
    room_part_refill_distance_cache: RoomPartSaveDistanceCache,
    room_part_frontier_distance_cache: RoomPartFrontierDistanceCache,
    occupancy: Vec<u8>,
    known_outcomes: Option<PreliminaryOutcomes>,
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
    graph_distance_snapshot: GraphDistanceSnapshot,
    room_part_frontier_distance_cache: RoomPartFrontierDistanceCache,
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
        save_part: usize,
    ) {
        self.save_room_part[save_part] = true;
        for part in 0..graph_size {
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
    nearest_frontier_source: Vec<GraphDistance>,
}

impl RoomPartFrontierDistanceCache {
    fn new(graph_size: usize) -> Self {
        Self {
            frontier_room_part_count: vec![0; graph_size],
            nearest_frontier_destination: vec![UNREACHABLE_DISTANCE; graph_size],
            nearest_frontier_source: vec![UNREACHABLE_DISTANCE; graph_size],
        }
    }

    fn clear(&mut self) {
        self.frontier_room_part_count.fill(0);
        self.nearest_frontier_destination.fill(UNREACHABLE_DISTANCE);
        self.nearest_frontier_source.fill(UNREACHABLE_DISTANCE);
    }

    fn add_frontier_part(
        &mut self,
        graph_distance: &[GraphDistance],
        graph_size: usize,
        frontier_part: usize,
    ) {
        self.frontier_room_part_count[frontier_part] += 1;
        if self.frontier_room_part_count[frontier_part] > 1 {
            return;
        }
        for part in 0..graph_size {
            let to_frontier = graph_distance[part * graph_size + frontier_part];
            if to_frontier < self.nearest_frontier_destination[part] {
                self.nearest_frontier_destination[part] = to_frontier;
            }
            let from_frontier = graph_distance[frontier_part * graph_size + part];
            if from_frontier < self.nearest_frontier_source[part] {
                self.nearest_frontier_source[part] = from_frontier;
            }
        }
    }

    fn remove_frontier_part(
        &mut self,
        graph_distance: &[GraphDistance],
        graph_size: usize,
        frontier_part: usize,
    ) {
        debug_assert!(self.frontier_room_part_count[frontier_part] > 0);
        self.frontier_room_part_count[frontier_part] -= 1;
        if self.frontier_room_part_count[frontier_part] > 0 {
            return;
        }
        for part in 0..graph_size {
            if self.nearest_frontier_destination[part]
                == graph_distance[part * graph_size + frontier_part]
            {
                self.nearest_frontier_destination[part] =
                    self.nearest_frontier_destination_for_part(graph_distance, graph_size, part);
            }
            if self.nearest_frontier_source[part]
                == graph_distance[frontier_part * graph_size + part]
            {
                self.nearest_frontier_source[part] =
                    self.nearest_frontier_source_for_part(graph_distance, graph_size, part);
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
        } else if self.frontier_room_part_count[to_part] > 0
            && old_distance == self.nearest_frontier_destination[from_part]
            && new_distance > old_distance
        {
            self.nearest_frontier_destination[from_part] =
                self.nearest_frontier_destination_for_part(graph_distance, graph_size, from_part);
        }

        if self.frontier_room_part_count[from_part] > 0
            && new_distance < self.nearest_frontier_source[to_part]
        {
            self.nearest_frontier_source[to_part] = new_distance;
        } else if self.frontier_room_part_count[from_part] > 0
            && old_distance == self.nearest_frontier_source[to_part]
            && new_distance > old_distance
        {
            self.nearest_frontier_source[to_part] =
                self.nearest_frontier_source_for_part(graph_distance, graph_size, to_part);
        }
    }

    fn nearest_frontier_destination_for_part(
        &self,
        graph_distance: &[GraphDistance],
        graph_size: usize,
        part: usize,
    ) -> GraphDistance {
        self.frontier_room_part_count
            .iter()
            .enumerate()
            .filter(|&(_, &count)| count > 0)
            .map(|(frontier_part, _)| graph_distance[part * graph_size + frontier_part])
            .min()
            .unwrap_or(UNREACHABLE_DISTANCE)
    }

    fn nearest_frontier_source_for_part(
        &self,
        graph_distance: &[GraphDistance],
        graph_size: usize,
        part: usize,
    ) -> GraphDistance {
        self.frontier_room_part_count
            .iter()
            .enumerate()
            .filter(|&(_, &count)| count > 0)
            .map(|(frontier_part, _)| graph_distance[frontier_part * graph_size + part])
            .min()
            .unwrap_or(UNREACHABLE_DISTANCE)
    }
}

impl Environment {
    pub fn new(common: &CommonData, map_size: (Coord, Coord), seed: u64) -> Self {
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
        self.graph_distance.fill(UNREACHABLE_DISTANCE);
        self.room_part_furthest_distance_cache.clear();
        self.room_part_save_distance_cache.clear();
        self.room_part_refill_distance_cache.clear();
        self.room_part_frontier_distance_cache.clear();
        self.occupancy.fill(0);
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

    fn choose_unused_room_in_connection_variant(
        &mut self,
        common: &CommonData,
        connection_variant_idx: ConnectionVariantIdx,
    ) -> Option<RoomIdx> {
        let remaining = self.connection_variant_unused_count[connection_variant_idx as usize];
        if remaining == 0 {
            return None;
        }
        let mut target = self.rng.random_range(0..remaining);
        for &room_idx in common.connection_variant_rooms[connection_variant_idx as usize].iter() {
            if self.room_used[room_idx as usize] {
                continue;
            }
            if target == 0 {
                return Some(room_idx);
            }
            target -= 1;
        }
        None
    }

    fn push_candidate_representatives(
        &mut self,
        common: &CommonData,
        candidate: GeometryAction,
        frontier_idx: FrontierIdx,
        actions: &mut Vec<CandidateAction>,
    ) {
        for &connection_variant_idx in
            common.geometry_connection_variants[candidate.geometry_idx as usize].iter()
        {
            if self.connection_variant_unused_count[connection_variant_idx as usize] == 0 {
                continue;
            }
            if let Some(room_idx) =
                self.choose_unused_room_in_connection_variant(common, connection_variant_idx)
            {
                actions.push(CandidateAction {
                    action: Action {
                        room_idx,
                        x: candidate.x,
                        y: candidate.y,
                    },
                    frontier_idx,
                    door_variant_idx: common.door_variant_idx(
                        connection_variant_idx,
                        candidate.door_direction,
                        candidate.door_x,
                        candidate.door_y,
                        candidate.door_kind,
                    ),
                });
            }
        }
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
        &self,
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
                    return Some(Action {
                        room_idx,
                        x: candidate.x,
                        y: candidate.y,
                    });
                }
            }
        }
        None
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

    fn room_distances(
        &self,
        common: &CommonData,
        is_destination_room: impl Fn(&crate::common::RoomData) -> bool,
    ) -> (Vec<f32>, Vec<u8>) {
        let graph_size = common.room_part.len();
        let mut values = vec![0.0; graph_size];
        let mut mask = vec![0; graph_size];
        let destination_parts: Vec<_> = self
            .active_room_parts
            .iter()
            .copied()
            .filter(|&room_part| {
                let (room_idx, _) = common.room_part[room_part as usize];
                is_destination_room(&common.room[room_idx as usize])
            })
            .map(usize::from)
            .collect();

        if destination_parts.is_empty() {
            return (values, mask);
        }

        for &room_part in &self.active_room_parts {
            let part = room_part as usize;
            let nearest_from_destination = destination_parts
                .iter()
                .map(|&destination_part| self.graph_distance[destination_part * graph_size + part])
                .filter(|&distance| distance != UNREACHABLE_DISTANCE)
                .min();
            let nearest_to_destination = destination_parts
                .iter()
                .map(|&destination_part| self.graph_distance[part * graph_size + destination_part])
                .filter(|&distance| distance != UNREACHABLE_DISTANCE)
                .min();
            if let (Some(from_destination), Some(to_destination)) =
                (nearest_from_destination, nearest_to_destination)
            {
                values[part] = f32::from(from_destination) + f32::from(to_destination);
                mask[part] = 1;
            }
        }

        (values, mask)
    }

    pub fn save_distances(&self, common: &CommonData) -> (Vec<f32>, Vec<u8>) {
        self.room_distances(common, |room| room.save)
    }

    pub fn refill_distances(&self, common: &CommonData) -> (Vec<f32>, Vec<u8>) {
        self.room_distances(common, |room| room.refill)
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

    fn room_part_furthest_distance_features(&self, common: &CommonData) -> (Vec<u8>, Vec<u8>) {
        fn encode_distance(distance: GraphDistance) -> u8 {
            if distance == UNREACHABLE_DISTANCE {
                0
            } else {
                distance + 1
            }
        }

        debug_assert_eq!(
            self.room_part_furthest_distance_cache
                .furthest_destination
                .len(),
            common.room_part.len()
        );
        (
            self.room_part_furthest_distance_cache
                .furthest_destination
                .iter()
                .copied()
                .map(encode_distance)
                .collect(),
            self.room_part_furthest_distance_cache
                .furthest_source
                .iter()
                .copied()
                .map(encode_distance)
                .collect(),
        )
    }

    fn room_part_save_distance_features(&self, common: &CommonData) -> Vec<u8> {
        debug_assert_eq!(
            self.room_part_save_distance_cache
                .nearest_save_destination
                .len(),
            common.room_part.len()
        );
        self.room_part_save_distance_cache
            .nearest_save_destination
            .iter()
            .zip(&self.room_part_save_distance_cache.nearest_save_source)
            .map(|(&to_save, &from_save)| {
                if to_save == UNREACHABLE_DISTANCE || from_save == UNREACHABLE_DISTANCE {
                    0
                } else {
                    to_save.saturating_add(from_save).saturating_add(1)
                }
            })
            .collect()
    }

    fn room_part_refill_distance_features(&self, common: &CommonData) -> Vec<u8> {
        debug_assert_eq!(
            self.room_part_refill_distance_cache
                .nearest_save_destination
                .len(),
            common.room_part.len()
        );
        self.room_part_refill_distance_cache
            .nearest_save_destination
            .iter()
            .zip(&self.room_part_refill_distance_cache.nearest_save_source)
            .map(|(&to_refill, &from_refill)| {
                if to_refill == UNREACHABLE_DISTANCE || from_refill == UNREACHABLE_DISTANCE {
                    0
                } else {
                    to_refill.saturating_add(from_refill).saturating_add(1)
                }
            })
            .collect()
    }

    fn room_part_frontier_distance_features(&self, common: &CommonData) -> Vec<u8> {
        debug_assert_eq!(
            self.room_part_frontier_distance_cache
                .nearest_frontier_destination
                .len(),
            common.room_part.len()
        );
        self.room_part_frontier_distance_cache
            .nearest_frontier_destination
            .iter()
            .zip(
                &self
                    .room_part_frontier_distance_cache
                    .nearest_frontier_source,
            )
            .map(|(&to_frontier, &from_frontier)| {
                if to_frontier == UNREACHABLE_DISTANCE || from_frontier == UNREACHABLE_DISTANCE {
                    0
                } else {
                    to_frontier.saturating_add(from_frontier).saturating_add(1)
                }
            })
            .collect()
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
    fn slow_room_part_save_distance_features(&self, common: &CommonData) -> Vec<u8> {
        let graph_size = common.room_part.len();
        let save_parts = self
            .active_room_parts
            .iter()
            .copied()
            .filter(|&room_part| {
                let (room_idx, _) = common.room_part[room_part as usize];
                common.room[room_idx as usize].save
            })
            .map(usize::from)
            .collect::<Vec<_>>();
        (0..graph_size)
            .map(|part| {
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
                if nearest_save_destination == UNREACHABLE_DISTANCE
                    || nearest_save_source == UNREACHABLE_DISTANCE
                {
                    0
                } else {
                    nearest_save_destination
                        .saturating_add(nearest_save_source)
                        .saturating_add(1)
                }
            })
            .collect()
    }

    #[cfg(test)]
    fn slow_room_part_frontier_distance_features(&self, common: &CommonData) -> Vec<u8> {
        let graph_size = common.room_part.len();
        let frontier_parts = self
            .frontier
            .values()
            .map(|frontier| frontier.room_part_idx as usize)
            .collect::<Vec<_>>();
        (0..graph_size)
            .map(|part| {
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
                if nearest_frontier_destination == UNREACHABLE_DISTANCE
                    || nearest_frontier_source == UNREACHABLE_DISTANCE
                {
                    0
                } else {
                    nearest_frontier_destination
                        .saturating_add(nearest_frontier_source)
                        .saturating_add(1)
                }
            })
            .collect()
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
        if mode.records_action() {
            let profile = profile_start();
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

                        for a in &self.actions {
                            let placed_geometry_idx = common.room[a.room_idx as usize].geometry_idx;
                            if common.has_geometry_intersection(
                                placed_geometry_idx,
                                a.x,
                                a.y,
                                opp_door.geometry_idx,
                                room_x,
                                room_y,
                            ) {
                                continue 'door;
                            }
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
                let frontier = Frontier {
                    dir_door_idx: door.dir_door_idx,
                    room_part_idx: frontier_part,
                    component: self.room_part_component(common, action.room_idx, door.part_idx),
                    kind: common.room_dir_door[door.direction as usize][door.dir_door_idx as usize]
                        .kind,
                    candidates,
                };
                self.room_part_frontier_distance_cache.add_frontier_part(
                    &self.graph_distance,
                    common.room_part.len(),
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
        for from_part in 0..room.door_group_count {
            let from_room_part = room.door_group_offset + from_part;
            self.active_room_parts.push(from_room_part as RoomPartIdx);
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
        if let [(room_part, attached_part)] = external_edges {
            self.add_single_attachment_room_distances(
                common,
                room_idx,
                *room_part,
                *attached_part,
                old_active_room_parts_len,
            );
        } else {
            for &(room_part, attached_part) in external_edges {
                self.add_graph_distance_edge(common, room_part, attached_part, 1);
                self.add_graph_distance_edge(common, attached_part, room_part, 1);
            }
        }
        if room.save {
            for local_part in 0..room.door_group_count {
                self.room_part_save_distance_cache.add_save_part(
                    &self.graph_distance,
                    graph_size,
                    room.door_group_offset + local_part,
                );
            }
        }
        if room.refill {
            for local_part in 0..room.door_group_count {
                self.room_part_refill_distance_cache.add_save_part(
                    &self.graph_distance,
                    graph_size,
                    room.door_group_offset + local_part,
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

        for local_from in 0..room.door_group_count {
            let from_part = room_start + local_from;
            let to_attachment =
                room.part_distances[local_from * room.door_group_count + local_attachment];
            if to_attachment != UNREACHABLE_DISTANCE {
                for to_part_idx in 0..old_active_room_parts_len {
                    let to_part = self.active_room_parts[to_part_idx] as usize;
                    let old_distance = self.graph_distance[attached_part * graph_size + to_part];
                    if let Some(distance) = graph_distance_sum(&[to_attachment, 1, old_distance]) {
                        self.set_graph_distance_min(graph_size, from_part, to_part, distance);
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
                        self.set_graph_distance_min(graph_size, from_old_part, from_part, distance);
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
                    self.set_graph_distance_min(graph_size, from_part, to_part, distance);
                }
            }
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

    fn get_all_candidates(&mut self, common: &CommonData) -> Vec<CandidateAction> {
        if self.actions.is_empty() {
            return vec![CandidateAction {
                action: self.get_initial_action(common),
                frontier_idx: -1,
                door_variant_idx: -1,
            }];
        }
        let mut sorted_frontiers = self.frontier.iter().collect::<Vec<_>>();
        sorted_frontiers.sort_unstable_by_key(|(location, _)| **location);
        let frontier_candidates = sorted_frontiers
            .iter()
            .enumerate()
            .map(|(frontier_idx, (_, frontier))| {
                (frontier_idx as FrontierIdx, frontier.candidates.clone())
            })
            .collect::<Vec<_>>();
        let mut candidates = Vec::new();
        for (frontier_idx, candidate_geometries) in frontier_candidates {
            for candidate in candidate_geometries {
                self.push_candidate_representatives(
                    common,
                    candidate,
                    frontier_idx,
                    &mut candidates,
                );
            }
        }
        candidates
    }

    fn candidate_proposal_score(
        candidate: CandidateAction,
        proposal_scores: &[f32],
        proposal_frontier_count: usize,
        proposal_door_variant_count: usize,
    ) -> f32 {
        if candidate.frontier_idx < 0 || candidate.door_variant_idx < 0 {
            return f32::NEG_INFINITY;
        }
        let frontier_idx = candidate.frontier_idx as usize;
        let door_variant_idx = candidate.door_variant_idx as usize;
        if frontier_idx >= proposal_frontier_count
            || door_variant_idx >= proposal_door_variant_count
        {
            return f32::NEG_INFINITY;
        }
        proposal_scores[frontier_idx * proposal_door_variant_count + door_variant_idx]
    }

    fn proposal_sample_weights(
        candidates: &[CandidateAction],
        proposal_scores: &[f32],
        proposal_frontier_count: usize,
        proposal_door_variant_count: usize,
        proposal_temperature: f32,
    ) -> Vec<f32> {
        let temperature = proposal_temperature.max(1e-6);
        let logits = candidates
            .iter()
            .map(|&candidate| {
                Self::candidate_proposal_score(
                    candidate,
                    proposal_scores,
                    proposal_frontier_count,
                    proposal_door_variant_count,
                ) / temperature
            })
            .collect::<Vec<_>>();
        let max_logit = logits
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(f32::NEG_INFINITY, f32::max);
        if !max_logit.is_finite() {
            return vec![1.0; candidates.len()];
        }
        logits
            .iter()
            .map(|&logit| {
                if logit.is_finite() {
                    (logit - max_logit).exp()
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>()
    }

    fn sample_weighted_remaining(&mut self, weights: &[f32], consumed: &[bool]) -> Option<usize> {
        let total_weight = weights
            .iter()
            .zip(consumed)
            .filter_map(|(&weight, &consumed)| (!consumed && weight > 0.0).then_some(weight))
            .sum::<f32>();
        if total_weight <= 0.0 || !total_weight.is_finite() {
            let remaining = consumed
                .iter()
                .enumerate()
                .filter_map(|(idx, &consumed)| (!consumed).then_some(idx))
                .collect::<Vec<_>>();
            return remaining.choose(&mut self.rng).copied();
        }
        let mut target = self.rng.random_range(0.0..total_weight);
        for (idx, (&weight, &consumed)) in weights.iter().zip(consumed).enumerate() {
            if consumed || weight <= 0.0 {
                continue;
            }
            if target < weight {
                return Some(idx);
            }
            target -= weight;
        }
        weights
            .iter()
            .zip(consumed)
            .rposition(|(&weight, &consumed)| !consumed && weight > 0.0)
    }

    pub fn get_filtered_candidates_with_outcomes(
        &mut self,
        common: &CommonData,
        recommended_candidates: usize,
        exploration_candidates: usize,
        proposal_temperature: f32,
        proposal_scores: Option<&[f32]>,
        proposal_frontier_count: usize,
        proposal_door_variant_count: usize,
        config: &FeatureConfig,
        frontier_neighbor_algorithm: FrontierNeighborAlgorithm,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
    ) -> Result<
        (
            PreliminaryOutcomes,
            Vec<Action>,
            Vec<FrontierIdx>,
            Vec<DoorVariantIdx>,
            Vec<PreliminaryOutcomes>,
            Vec<Vec<i16>>,
            Vec<Features>,
        ),
        String,
    > {
        let pre_candidate_outcomes = self.outcomes(common);
        let candidates = self.get_all_candidates(common);
        let max_candidates = recommended_candidates + exploration_candidates;
        let mut clean = Vec::with_capacity(max_candidates.min(candidates.len()));
        let mut rejected = Vec::new();
        let mut consumed = vec![false; candidates.len()];

        let proposal_weights = if recommended_candidates > 0 {
            let proposal_scores = proposal_scores.ok_or_else(|| {
                "proposal scores are required when recommended_candidates is greater than zero"
                    .to_string()
            })?;
            Self::proposal_sample_weights(
                &candidates,
                proposal_scores,
                proposal_frontier_count,
                proposal_door_variant_count,
                proposal_temperature,
            )
        } else {
            Vec::new()
        };
        let mut recommended_clean = 0;
        while recommended_clean < recommended_candidates {
            let Some(candidate_idx) = self.sample_weighted_remaining(&proposal_weights, &consumed)
            else {
                break;
            };
            consumed[candidate_idx] = true;
            let candidate = candidates[candidate_idx];
            match self.evaluate_candidate_outcome(
                common,
                &pre_candidate_outcomes,
                candidate.action,
                config,
                frontier_neighbor_algorithm,
                frontier_neighbor_count,
                frontier_window_size,
            )? {
                CandidateOutcome::Rejected => rejected.push(candidate),
                CandidateOutcome::Clean(post_candidate_outcomes, door_match, features) => {
                    clean.push((candidate, post_candidate_outcomes, door_match, features));
                    recommended_clean += 1;
                }
            }
        }

        let mut exploration_order = consumed
            .iter()
            .enumerate()
            .filter_map(|(idx, &consumed)| (!consumed).then_some(idx))
            .collect::<Vec<_>>();
        exploration_order.shuffle(&mut self.rng);
        let mut exploration_clean = 0;
        for candidate_idx in exploration_order {
            if exploration_clean == exploration_candidates {
                break;
            }
            let candidate = candidates[candidate_idx];
            match self.evaluate_candidate_outcome(
                common,
                &pre_candidate_outcomes,
                candidate.action,
                config,
                frontier_neighbor_algorithm,
                frontier_neighbor_count,
                frontier_window_size,
            )? {
                CandidateOutcome::Rejected => rejected.push(candidate),
                CandidateOutcome::Clean(post_candidate_outcomes, door_match, features) => {
                    clean.push((candidate, post_candidate_outcomes, door_match, features));
                    exploration_clean += 1;
                }
            }
        }

        let mut candidates_with_outcomes = if clean.is_empty() && !rejected.is_empty() {
            rejected
                .into_iter()
                .take(max_candidates)
                .map(|candidate| {
                    let (post_candidate_outcomes, door_match, features) = self
                        .outcomes_and_features_after_candidate(
                            common,
                            candidate.action,
                            config,
                            frontier_neighbor_algorithm,
                            frontier_neighbor_count,
                            frontier_window_size,
                        );
                    (candidate, post_candidate_outcomes, door_match, features)
                })
                .collect()
        } else {
            clean
        };
        candidates_with_outcomes.truncate(max_candidates);
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
        ))
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
    ) -> Result<
        (
            PreliminaryOutcomes,
            Vec<Action>,
            Vec<FrontierIdx>,
            Vec<DoorVariantIdx>,
            Vec<PreliminaryOutcomes>,
            Vec<Vec<i16>>,
            Vec<Features>,
            usize,
            usize,
        ),
        String,
    > {
        debug_assert_eq!(sampled_frontier_idx.len(), sampled_door_variant_idx.len());
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
                        );
                    (candidate, post_candidate_outcomes, door_match, features)
                })
                .collect::<Vec<_>>();
            profile_end(ProfileMetric::EnvProposalFallbackRecompute, profile);
            fallback
        } else {
            clean
        };

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
        pre_candidate_outcomes: &PreliminaryOutcomes,
        candidate: Action,
        config: &FeatureConfig,
        frontier_neighbor_algorithm: FrontierNeighborAlgorithm,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
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

        let profile = profile_start();
        let features = self.features_for_applied_candidate(
            common,
            candidate,
            config,
            frontier_neighbor_algorithm,
            frontier_neighbor_count,
            frontier_window_size,
        );
        profile_end(ProfileMetric::EnvProposalFeatures, profile);
        let outcomes = PreliminaryOutcomes {
            door_valid: door_valid.clone(),
            connections_valid: connections_valid.clone(),
            toilet_valid,
            toilet_crossed_room_idx: self.toilet_crossed_room_idx(common),
        };
        let profile = profile_start();
        let door_match = self.door_match_feature(common, &outcomes);
        profile_end(ProfileMetric::EnvProposalDoorMatch, profile);
        let profile = profile_start();
        self.restore_lookahead_candidate(common, snapshot);
        profile_end(ProfileMetric::EnvProposalRestore, profile);
        Ok(CandidateOutcome::Clean(outcomes, door_match, features))
    }

    fn outcomes_and_features_after_candidate(
        &mut self,
        common: &CommonData,
        candidate: Action,
        config: &FeatureConfig,
        frontier_neighbor_algorithm: FrontierNeighborAlgorithm,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
    ) -> (PreliminaryOutcomes, Vec<i16>, Features) {
        let profile = profile_start();
        let snapshot = self.apply_lookahead_candidate(candidate, common);
        profile_end(ProfileMetric::EnvProposalApplyLookahead, profile);
        let outcomes = self.outcomes(common);
        let profile = profile_start();
        let door_match = self.door_match_feature(common, &outcomes);
        profile_end(ProfileMetric::EnvProposalDoorMatch, profile);
        let profile = profile_start();
        let features = self.features_for_applied_candidate(
            common,
            candidate,
            config,
            frontier_neighbor_algorithm,
            frontier_neighbor_count,
            frontier_window_size,
        );
        profile_end(ProfileMetric::EnvProposalFeatures, profile);
        let profile = profile_start();
        self.restore_lookahead_candidate(common, snapshot);
        profile_end(ProfileMetric::EnvProposalRestore, profile);
        (outcomes, door_match, features)
    }

    pub fn outcomes_after_candidate(
        &mut self,
        common: &CommonData,
        candidate: Action,
    ) -> (PreliminaryOutcomes, Vec<i16>) {
        let snapshot = self.apply_lookahead_candidate(candidate, common);
        let outcomes = self.outcomes(common);
        let door_match = self.door_match_feature(common, &outcomes);
        self.restore_lookahead_candidate(common, snapshot);
        (outcomes, door_match)
    }

    fn door_match_feature(&self, common: &CommonData, outcomes: &PreliminaryOutcomes) -> Vec<i16> {
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

    fn features_for_applied_candidate(
        &self,
        common: &CommonData,
        candidate: Action,
        config: &FeatureConfig,
        frontier_neighbor_algorithm: FrontierNeighborAlgorithm,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
    ) -> Features {
        if config.is_empty() {
            return Features::default();
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
        self.features_with_occupancy(
            common,
            config,
            &self.occupancy,
            extra_occupied,
            frontier_neighbor_algorithm,
            frontier_neighbor_count,
            frontier_window_size,
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
            graph_distance_snapshot,
            room_part_frontier_distance_cache: self.room_part_frontier_distance_cache.clone(),
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
        self.restore_graph_distance_snapshot(common, snapshot.graph_distance_snapshot);
        self.room_part_frontier_distance_cache = snapshot.room_part_frontier_distance_cache;
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
                        dir_door_idx: frontier.dir_door_idx,
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

    pub fn features(
        &self,
        common: &CommonData,
        config: &FeatureConfig,
        frontier_neighbor_algorithm: FrontierNeighborAlgorithm,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
    ) -> Features {
        self.features_with_occupancy(
            common,
            config,
            &self.occupancy,
            None,
            frontier_neighbor_algorithm,
            frontier_neighbor_count,
            frontier_window_size,
        )
    }

    fn features_with_occupancy(
        &self,
        common: &CommonData,
        config: &FeatureConfig,
        occupancy: &[u8],
        extra_occupied: Option<(&GeometryData, Coord, Coord)>,
        frontier_neighbor_algorithm: FrontierNeighborAlgorithm,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
    ) -> Features {
        assert!(self.frontier.len() <= Self::max_frontiers(common));
        let profile = profile_start();
        let frontier_count = if config.has_frontier_features() {
            self.frontier.len()
        } else {
            0
        };
        let inventory = if config.inventory {
            self.connection_variant_unused_count
                .iter()
                .map(|&count| count as u8)
                .collect::<Vec<_>>()
        } else {
            vec![]
        };
        let room_placed = if config.room_position {
            self.room_used
                .iter()
                .map(|bit| u8::from(*bit))
                .collect::<Vec<_>>()
        } else {
            vec![]
        };
        let (room_part_furthest_destination, room_part_furthest_source) =
            if config.room_part_furthest_distance {
                self.room_part_furthest_distance_features(common)
            } else {
                (vec![], vec![])
            };
        let room_part_save_distance = if config.room_part_save_distance {
            self.room_part_save_distance_features(common)
        } else {
            vec![]
        };
        let room_part_refill_distance = if config.room_part_refill_distance {
            self.room_part_refill_distance_features(common)
        } else {
            vec![]
        };
        let room_part_frontier_distance = if config.room_part_frontier_distance {
            self.room_part_frontier_distance_features(common)
        } else {
            vec![]
        };
        let mut frontier = vec![0; frontier_count * FEATURE_FRONTIER_WIDTH];
        let frontier_window_area = frontier_window_size * frontier_window_size;
        let packed_frontier_window_size = frontier_window_area.div_ceil(8);
        let mut frontier_occupancy = if config.frontier_occupancy {
            vec![0; frontier_count * packed_frontier_window_size]
        } else {
            vec![]
        };
        let mut frontier_neighbor = if config.frontier_neighbor {
            vec![-1; frontier_count * frontier_neighbor_count]
        } else {
            vec![]
        };
        let mut frontier_neighbor_pair = if config.frontier_neighbor_flags {
            vec![0; frontier_count * frontier_neighbor_count]
        } else {
            vec![]
        };
        let mut connection_reachability = if config.connection_reachability {
            vec![0; common.room_connection.len()]
        } else {
            vec![]
        };
        let mut frontier_connection_reachability = if config.frontier_connection_reachability {
            vec![0; frontier_count * common.room_connection.len()]
        } else {
            vec![]
        };
        profile_end(ProfileMetric::EnvFeaturesSetup, profile);

        let profile = profile_start();
        let mut sorted_frontiers = if config.has_frontier_features() {
            self.frontier.iter().collect::<Vec<_>>()
        } else {
            vec![]
        };
        sorted_frontiers.sort_unstable_by_key(|(location, _)| **location);
        profile_end(ProfileMetric::EnvFeaturesSortFrontiers, profile);

        let map_width = self.map_size.0 as usize;
        for (idx, (location, data)) in sorted_frontiers.iter().enumerate() {
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
        for (connection_idx, connection) in common.room_connection.iter().enumerate() {
            if !self.room_used[connection.room_idx as usize] {
                continue;
            }
            let from_component =
                self.room_part_component(common, connection.room_idx, connection.from_part);
            let to_component =
                self.room_part_component(common, connection.room_idx, connection.to_part);
            if config.connection_reachability
                && self.scc_dag.can_reach(from_component, to_component)
            {
                connection_reachability[connection_idx] = 1;
            }
            if config.frontier_connection_reachability {
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
            }
        }
        profile_end(ProfileMetric::EnvFeaturesConnectionReachability, profile);

        if config.frontier_neighbor {
            let profile = profile_start();
            let locations = sorted_frontiers
                .iter()
                .map(|(location, _)| **location)
                .collect::<Vec<_>>();
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
                _ => {
                    let neighbors = match frontier_neighbor_algorithm {
                        FrontierNeighborAlgorithm::Delaunay => {
                            frontier_delaunay_neighbors(&locations, frontier_neighbor_count)
                        }
                        FrontierNeighborAlgorithm::Nearest => {
                            frontier_nearest_neighbors(&locations, frontier_neighbor_count, true)
                        }
                        FrontierNeighborAlgorithm::NearestExclusive => {
                            frontier_nearest_neighbors(&locations, frontier_neighbor_count, false)
                        }
                    };
                    for (src_idx, neighbors) in neighbors.iter().enumerate() {
                        for (neighbor_idx, &dst_idx) in neighbors.iter().enumerate() {
                            frontier_neighbor[src_idx * frontier_neighbor_count + neighbor_idx] =
                                dst_idx as i16;
                        }
                    }
                }
            }
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
        let room_x = if config.room_position {
            self.room_x.clone()
        } else {
            vec![]
        };
        let room_y = if config.room_position {
            self.room_y.clone()
        } else {
            vec![]
        };
        profile_end(ProfileMetric::EnvFeaturesRoomPositionClone, profile);

        let toilet_crossed_room_idx = if config.toilet_crossed_room {
            vec![self.toilet_crossed_room_idx(common)]
        } else {
            vec![]
        };

        let profile = profile_start();
        let result = Features {
            inventory,
            room_x,
            room_y,
            room_placed,
            room_part_furthest_destination,
            room_part_furthest_source,
            room_part_save_distance,
            room_part_refill_distance,
            room_part_frontier_distance,
            frontier,
            frontier_occupancy,
            frontier_neighbor,
            frontier_neighbor_pair,
            connection_reachability,
            frontier_connection_reachability,
            toilet_crossed_room_idx,
        };
        profile_end(ProfileMetric::EnvFeaturesOutput, profile);
        result
    }

    pub fn features_after_candidate(
        &mut self,
        common: &CommonData,
        candidate: Action,
        config: &FeatureConfig,
        frontier_neighbor_algorithm: FrontierNeighborAlgorithm,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
    ) -> Features {
        if config.is_empty() {
            return Features::default();
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
        );
        let profile = profile_start();
        self.restore_feature_candidate(common, candidate, snapshot);
        profile_end(ProfileMetric::EnvFeaturesApplyCandidate, profile);
        features
    }

    pub fn actions(&self) -> &[Action] {
        &self.actions
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

    pub fn outcomes(&self, common: &CommonData) -> PreliminaryOutcomes {
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

        PreliminaryOutcomes {
            door_valid,
            connections_valid,
            toilet_valid: self.toilet_outcome(common),
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
    ) -> Result<PreliminaryOutcomes, String> {
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
        }
        self.known_outcomes = Some(merge_known_outcomes(
            self.known_outcomes.as_ref(),
            &outcomes,
        ));
        Ok(outcomes)
    }
}

fn merge_known_outcomes(
    known: Option<&PreliminaryOutcomes>,
    current: &PreliminaryOutcomes,
) -> PreliminaryOutcomes {
    let Some(known) = known else {
        return current.clone();
    };
    PreliminaryOutcomes {
        door_valid: merge_known_outcome_values(&known.door_valid, &current.door_valid),
        connections_valid: merge_known_outcome_values(
            &known.connections_valid,
            &current.connections_valid,
        ),
        toilet_valid: merge_known_outcome_value(known.toilet_valid, current.toilet_valid),
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

    #[test]
    fn proposal_sample_order_prefers_high_scored_candidate() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (4, 4), 0);
        let candidates = vec![
            CandidateAction {
                action: Action {
                    room_idx: 0,
                    x: 0,
                    y: 0,
                },
                frontier_idx: 0,
                door_variant_idx: 0,
            },
            CandidateAction {
                action: Action {
                    room_idx: 0,
                    x: 1,
                    y: 0,
                },
                frontier_idx: 1,
                door_variant_idx: 2,
            },
            CandidateAction {
                action: Action {
                    room_idx: 0,
                    x: 2,
                    y: 0,
                },
                frontier_idx: 1,
                door_variant_idx: 1,
            },
        ];
        let proposal_scores = vec![0.0, 0.0, 0.0, 0.0, -10.0, 10.0];

        assert_eq!(
            Environment::candidate_proposal_score(candidates[1], &proposal_scores, 2, 3),
            10.0
        );
        let weights =
            Environment::proposal_sample_weights(&candidates, &proposal_scores, 2, 3, 0.01);
        assert_eq!(
            env.sample_weighted_remaining(&weights, &[false, false, false]),
            Some(1)
        );
    }

    #[test]
    fn proposal_candidate_mask_marks_placeable_frontier_cells() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1], [1, 0]],
                "missing_connections": []
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1], [1, 0]],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 0);
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
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1], [1, 0]],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 0);
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
                dir_door_idx: 0,
                room_part_idx: 0,
                component: 0,
                kind: 0,
                candidates: vec![candidate, candidate],
            },
        );
        env.frontier.insert(
            door_location(1, 0, false),
            Frontier {
                dir_door_idx: 0,
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
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1], [1, 0]],
                "missing_connections": []
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1], [1, 0]],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 0);
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
                1,
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
            &PreliminaryOutcomes {
                door_valid: vec![Unknown],
                connections_valid: vec![Valid],
                toilet_valid: Valid,
                toilet_crossed_room_idx: -1,
            },
            &PreliminaryOutcomes {
                door_valid: vec![Invalid],
                connections_valid: vec![Valid],
                toilet_valid: Valid,
                toilet_crossed_room_idx: -1,
            },
        ));
        assert!(!introduces_invalid_outcome(
            &PreliminaryOutcomes {
                door_valid: vec![Invalid],
                connections_valid: vec![Unknown],
                toilet_valid: Unknown,
                toilet_crossed_room_idx: -1,
            },
            &PreliminaryOutcomes {
                door_valid: vec![Invalid],
                connections_valid: vec![Valid],
                toilet_valid: Unknown,
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
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "down", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (4, 4), 0);

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
    fn environment_tracks_global_graph_distances() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "down", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 0);

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
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut fast_env = Environment::new(&common, (4, 4), 0);
        let mut full_env = Environment::new(&common, (4, 4), 0);
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
    }

    #[test]
    fn graph_distance_relaxation_updates_existing_parts_through_new_room() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 0);

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
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "down", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 0);

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
                        [{"direction": "right", "x": 0, "y": 0, "kind": 0}]
                    ],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [
                        [{"direction": "left", "x": 0, "y": 0, "kind": 0}],
                        [{"direction": "right", "x": 0, "y": 0, "kind": 0}]
                    ],
                    "connections": [[0, 1], [1, 0]],
                    "missing_connections": []
                },
                {
                    "save": true,
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [
                        [{"direction": "left", "x": 0, "y": 0, "kind": 0}]
                    ],
                    "connections": [],
                    "missing_connections": []
                }
            ]
            "#,
        )
        .unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 0);
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
        let mut env = Environment::new(&common, (5, 5), 0);
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
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "down", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": [[0, 1], [1, 2], [2, 0]]
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 0);
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
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "down", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": [[0, 1], [1, 2], [2, 0]]
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 5), 0);
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
    fn room_part_save_distance_features_encode_nearest_round_trip_save_distance() {
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
        let mut env = Environment::new(&common, (5, 5), 0);
        let graph_size = common.room_part.len();
        env.active_room_parts = vec![0, 1, 2];
        for part in 0..graph_size {
            env.set_graph_distance(graph_size, part, part, 0);
        }
        env.set_graph_distance(graph_size, 1, 0, 2);
        env.set_graph_distance(graph_size, 0, 1, 4);
        env.set_graph_distance(graph_size, 2, 0, 5);
        env.room_part_save_distance_cache
            .add_save_part(&env.graph_distance, graph_size, 0);

        assert_eq!(env.room_part_save_distance_features(&common), vec![1, 7, 0]);
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
        let mut env = Environment::new(&common, (5, 5), 0);
        let graph_size = common.room_part.len();
        env.active_room_parts = vec![0, 1, 2];
        for part in 0..graph_size {
            env.set_graph_distance(graph_size, part, part, 0);
        }
        env.set_graph_distance(graph_size, 2, 0, 250);
        env.set_graph_distance(graph_size, 0, 2, 10);
        env.set_graph_distance(graph_size, 2, 1, 20);
        env.set_graph_distance(graph_size, 1, 2, 30);
        env.room_part_save_distance_cache
            .add_save_part(&env.graph_distance, graph_size, 0);
        env.room_part_save_distance_cache
            .add_save_part(&env.graph_distance, graph_size, 1);

        assert_eq!(
            env.room_part_save_distance_features(&common),
            vec![1, 1, 31]
        );

        env.set_graph_distance(graph_size, 1, 2, 5);
        assert_eq!(
            env.room_part_save_distance_features(&common),
            vec![1, 1, 26]
        );

        env.set_graph_distance(graph_size, 2, 1, 250);
        env.set_graph_distance(graph_size, 1, 2, 250);
        assert_eq!(
            env.room_part_save_distance_features(&common),
            vec![1, 1, 255]
        );
        env.assert_room_part_save_distance_cache_matches_slow(&common);
    }

    #[test]
    fn room_part_frontier_distance_features_encode_nearest_round_trip_frontier_distance() {
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
        let mut env = Environment::new(&common, (5, 5), 0);
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
                dir_door_idx: 0,
                room_part_idx: 0,
                component: 0,
                kind: 0,
                candidates: vec![],
            },
        );
        env.room_part_frontier_distance_cache
            .add_frontier_part(&env.graph_distance, graph_size, 0);

        assert_eq!(
            env.room_part_frontier_distance_features(&common),
            vec![1, 7, 0]
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
        let mut env = Environment::new(&common, (5, 5), 0);
        let graph_size = common.room_part.len();
        env.active_room_parts = vec![0, 1, 2];
        for part in 0..graph_size {
            env.set_graph_distance(graph_size, part, part, 0);
        }
        env.set_graph_distance(graph_size, 2, 0, 250);
        env.set_graph_distance(graph_size, 0, 2, 10);
        env.set_graph_distance(graph_size, 2, 1, 20);
        env.set_graph_distance(graph_size, 1, 2, 30);

        for (idx, frontier_part) in [(0, 0), (1, 1), (2, 1)] {
            env.frontier.insert(
                door_location(idx, 0, false),
                Frontier {
                    dir_door_idx: 0,
                    room_part_idx: frontier_part,
                    component: 0,
                    kind: 0,
                    candidates: vec![],
                },
            );
            env.room_part_frontier_distance_cache.add_frontier_part(
                &env.graph_distance,
                graph_size,
                frontier_part as usize,
            );
        }

        assert_eq!(
            env.room_part_frontier_distance_features(&common),
            vec![1, 1, 31]
        );

        env.set_graph_distance(graph_size, 1, 2, 5);
        assert_eq!(
            env.room_part_frontier_distance_features(&common),
            vec![1, 1, 26]
        );

        env.frontier.remove(&door_location(1, 0, false));
        env.room_part_frontier_distance_cache.remove_frontier_part(
            &env.graph_distance,
            graph_size,
            1,
        );
        assert_eq!(
            env.room_part_frontier_distance_features(&common),
            vec![1, 1, 26]
        );

        env.frontier.remove(&door_location(2, 0, false));
        env.room_part_frontier_distance_cache.remove_frontier_part(
            &env.graph_distance,
            graph_size,
            1,
        );
        assert_eq!(
            env.room_part_frontier_distance_features(&common),
            vec![1, 0, 255]
        );
        env.assert_room_part_frontier_distance_cache_matches_slow(&common);
    }

    #[test]
    fn door_match_counts_include_unmatched_row_and_column() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (5, 4), 0);

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
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "down", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1], [1, 0]],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (4, 4), 0);

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
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "down", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (4, 4), 0);

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
    }

    #[test]
    fn missing_connect_distances_mask_unreachable_connections() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "down", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": [[0, 1], [1, 2], [2, 0]]
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (4, 4), 0);
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
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "down", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (4, 4), 0);

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
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "down", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "up", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (4, 1), 0);

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
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "down", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (4, 4), 0);

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
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "down", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": [[0, 1], [1, 2], [2, 0]]
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        assert_eq!(Environment::max_frontiers(&common), 3);
        let mut env = Environment::new(&common, (4, 4), 0);
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
    fn outcomes_after_candidate_restores_graph_distances() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            }
        ]
        "#;
        let rooms: Vec<Room> = serde_json::from_str(rooms_json).unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (4, 4), 0);
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
    fn features_do_not_depend_on_frontier_candidate_lists() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "down", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": [[0, 1], [1, 2], [2, 0]]
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}]
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
        let mut full_env = Environment::new(&common, (4, 4), 0);
        let mut known_env = Environment::new(&common, (4, 4), 0);

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
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "down", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": [[0, 1], [1, 2], [2, 0]]
            },
            {
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}]
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
        let mut full_env = Environment::new(&common, (4, 4), 0);
        let mut replay_env = Environment::new(&common, (4, 4), 0);

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
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "down", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            }]
            "#,
        )
        .unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (4, 4), 0);
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
            ..FeatureConfig::all_disabled()
        };
        let features = env.features(&common, &config, FrontierNeighborAlgorithm::Delaunay, 1, 1);
        assert_eq!(features.connection_reachability, vec![0]);
        assert_eq!(features.frontier_connection_reachability, vec![1, 2]);
    }

    #[test]
    fn disabled_features_skip_candidate_simulation() {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"[{"map": [[1]], "toilet_crossing_x": [], "doors": [], "connections": [], "missing_connections": []}]"#,
        )
        .unwrap();
        let common = CommonData::new(rooms).unwrap();
        let mut env = Environment::new(&common, (4, 4), 0);
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
        let env = Environment::new(&common, (8, 12), 0);

        assert_eq!(env.outcomes(&common).toilet_valid, DoorValidOutcome::Valid);
        assert_eq!(env.outcomes(&common).toilet_crossed_room_idx, -1);
    }

    #[test]
    fn toilet_outcome_requires_exactly_one_crossing_at_finish() {
        let common = toilet_outcome_test_common();
        let mut env = Environment::new(&common, (8, 12), 0);
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

        let mut env = Environment::new(&common, (8, 12), 0);
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
        let mut env = Environment::new(&common, (8, 12), 0);
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

        let (outcomes, _) = env.outcomes_after_candidate(
            &common,
            Action {
                room_idx: 1,
                x: 0,
                y: 4,
            },
        );
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
        let mut env = Environment::new(&common, (8, 12), 0);
        env.step_known(
            Action {
                room_idx: 2,
                x: 0,
                y: 0,
            },
            &common,
        );

        let (outcomes, _) = env.outcomes_after_candidate(
            &common,
            Action {
                room_idx: common.room.len() as RoomIdx,
                x: 0,
                y: 0,
            },
        );
        assert_eq!(outcomes.toilet_valid, DoorValidOutcome::Invalid);
        assert_eq!(outcomes.toilet_crossed_room_idx, -1);
    }
}
