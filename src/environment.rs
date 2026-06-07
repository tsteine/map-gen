use bitvec::vec::BitVec;
use delaunator::{EMPTY, Point, next_halfedge, triangulate};
use hashbrown::{HashMap, HashSet};
use rand::SeedableRng;
use rand::prelude::*;
use serde::Deserialize;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::Instant;

use crate::common::{
    Action, CommonData, ConnectionVariantIdx, Coord, DirDoorIdx, Direction, DoorKind, DoorLocation,
    DoorValidOutcome, GeometryData, GeometryIdx, NUM_DIRS, PartIdx, RoomIdx, RoomPartIdx,
    get_behind_door_position,
};
use crate::engine::{profile_enabled, record_profile_metric};
use crate::scc_dag::SccDag;

const NO_COMPONENT: usize = usize::MAX;
pub const FEATURE_FRONTIER_WIDTH: usize = 5;
const PROFILE_STEP_PUSH_ACTION: usize = 13;
const PROFILE_STEP_MARK_ROOM_USED: usize = 14;
const PROFILE_STEP_COMPONENTS_EDGES: usize = 15;
const PROFILE_STEP_OCCUPANCY: usize = 16;
const PROFILE_STEP_MATCH_EXISTING_FRONTIERS: usize = 17;
const PROFILE_STEP_BUILD_NEW_FRONTIER_CANDIDATES: usize = 18;
const PROFILE_STEP_FILTER_EXISTING_FRONTIERS: usize = 19;

#[derive(Clone, Copy, PartialEq, Eq)]
enum CandidateUpdate {
    Build,
    Skip,
}

fn profile_start() -> Option<Instant> {
    profile_enabled().then(Instant::now)
}

fn profile_end(metric_idx: usize, start: Option<Instant>) {
    if let Some(start) = start {
        record_profile_metric(metric_idx, start.elapsed());
    }
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
    component: usize,
    kind: DoorKind,
    candidates: Vec<GeometryAction>, // possible geometry placements to connect to this frontier
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct GeometryAction {
    geometry_idx: GeometryIdx,
    x: Coord,
    y: Coord,
}

pub struct Outcomes {
    // For each door, whether it is connected to another door.
    pub door_valid: Vec<DoorValidOutcome>,
    // For each connection, whether its destination can reach its source.
    pub connections_valid: Vec<DoorValidOutcome>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureConfig {
    pub inventory: bool,
    pub temperature: bool,
    pub action_candidates: bool,
    pub room_position: bool,
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
}

impl FeatureConfig {
    pub fn is_empty(&self) -> bool {
        !self.inventory
            && !self.temperature
            && !self.action_candidates
            && !self.room_position
            && !self.connection_reachability
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
        Ok(())
    }

    #[cfg(test)]
    pub fn all() -> Self {
        Self {
            inventory: true,
            temperature: true,
            action_candidates: true,
            room_position: true,
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
        }
    }

    #[cfg(test)]
    pub fn all_disabled() -> Self {
        Self {
            inventory: false,
            temperature: false,
            action_candidates: false,
            room_position: false,
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
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Features {
    pub inventory: Vec<u8>,
    pub room_x: Vec<Coord>,
    pub room_y: Vec<Coord>,
    pub room_placed: Vec<u8>,
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
    occupancy: Vec<u8>,
}

struct FeatureSnapshot {
    finished: bool,
    frontier: HashMap<DoorLocation, Frontier>,
    connection_variant_idx: Option<ConnectionVariantIdx>,
    connection_variant_unused_count: usize,
    room_part_component: Vec<usize>,
    scc_dag: SccDag,
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
            occupancy: vec![0; map_size.0 as usize * map_size.1 as usize],
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
        self.occupancy.fill(0);
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
        actions: &mut Vec<Action>,
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
                actions.push(Action {
                    room_idx,
                    x: candidate.x,
                    y: candidate.y,
                });
            }
        }
    }

    pub fn step(&mut self, action: Action, common: &CommonData) {
        self.step_impl(action, common, CandidateUpdate::Build);
    }

    pub fn step_known(&mut self, action: Action, common: &CommonData) {
        self.step_impl(action, common, CandidateUpdate::Skip);
    }

    fn step_for_features(&mut self, action: Action, common: &CommonData) {
        if self.finished {
            return;
        }
        if action.room_idx >= common.room.len() as RoomIdx {
            // Dummy/invalid action: do nothing more.
            self.finished = true;
            return;
        }
        let room = &common.room[action.room_idx as usize];
        let connection_variant_idx = room.connection_variant_idx;
        assert!(!self.room_used[action.room_idx as usize]);
        self.room_used.set(action.room_idx as usize, true);
        self.room_x[action.room_idx as usize] = action.x;
        self.room_y[action.room_idx as usize] = action.y;
        self.connection_variant_unused_count[connection_variant_idx as usize] -= 1;
        self.add_room_components_and_edges(action, common);

        // Feature extraction only needs frontier metadata. Future attachment candidates are
        // maintained by committed steps when an action is selected.
        for door in &room.doors {
            let door_loc = DoorLocation::new(door, action.x, action.y);
            if let Some(frontier) = self.frontier.remove(&door_loc) {
                let i1 = door.dir_door_idx;
                let i2 = frontier.dir_door_idx;
                let p1 = common.room_dir_door[door.direction as usize][i1 as usize].room_part_idx;
                let p2 = common.room_dir_door[door.direction.opposite() as usize][i2 as usize]
                    .room_part_idx;
                self.add_component_edge(
                    self.room_part_component[p1 as usize],
                    self.room_part_component[p2 as usize],
                );
                self.add_component_edge(
                    self.room_part_component[p2 as usize],
                    self.room_part_component[p1 as usize],
                );
            } else {
                self.frontier.insert(
                    door_loc,
                    Frontier {
                        dir_door_idx: door.dir_door_idx,
                        component: self.room_part_component(common, action.room_idx, door.part_idx),
                        kind: common.room_dir_door[door.direction as usize]
                            [door.dir_door_idx as usize]
                            .kind,
                        candidates: vec![],
                    },
                );
            }
        }
    }

    fn step_impl(
        &mut self,
        action: Action,
        common: &CommonData,
        candidate_update: CandidateUpdate,
    ) {
        let profile = profile_start();
        self.actions.push(action);
        profile_end(PROFILE_STEP_PUSH_ACTION, profile);
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
        self.geometry_unused_count[action_geometry_idx as usize] -= 1;
        self.connection_variant_unused_count[connection_variant_idx as usize] -= 1;
        profile_end(PROFILE_STEP_MARK_ROOM_USED, profile);

        let profile = profile_start();
        self.add_room_components_and_edges(action, common);
        profile_end(PROFILE_STEP_COMPONENTS_EDGES, profile);

        let profile = profile_start();
        for &(dx, dy) in &common.geometry[action_geometry_idx as usize].occupied_tiles {
            let x = action.x + dx;
            let y = action.y + dy;
            if x >= 0 && y >= 0 && x < self.map_size.0 && y < self.map_size.1 {
                self.occupancy[y as usize * self.map_size.0 as usize + x as usize] = 1;
            }
        }
        profile_end(PROFILE_STEP_OCCUPANCY, profile);

        // Remove the frontiers that the new room connects to (if any),
        // and update the frontier with the new unconnected doors of the new room.
        for door in room.doors.iter() {
            let door_loc = DoorLocation::new(door, action.x, action.y);
            if let Some(frontier) = self.frontier.remove(&door_loc) {
                let profile = profile_start();
                // This frontier is now connected, so remove it and mark the doors as connected:
                let i1 = door.dir_door_idx;
                let i2 = frontier.dir_door_idx;
                self.door_matches[door.direction as usize][i1 as usize] = i2;
                self.door_matches[door.direction.opposite() as usize][i2 as usize] = i1;
                let p1 = common.room_dir_door[door.direction as usize][i1 as usize].room_part_idx;
                let p2 = common.room_dir_door[door.direction.opposite() as usize][i2 as usize]
                    .room_part_idx;
                self.add_component_edge(
                    self.room_part_component[p1 as usize],
                    self.room_part_component[p2 as usize],
                );
                self.add_component_edge(
                    self.room_part_component[p2 as usize],
                    self.room_part_component[p1 as usize],
                );
                profile_end(PROFILE_STEP_MATCH_EXISTING_FRONTIERS, profile);
            } else {
                // This door is not connected to any existing frontier, so it becomes a new frontier.
                // Check all doors with the given orientation, to list which ones could connect here.
                let mut candidates = vec![];
                if candidate_update == CandidateUpdate::Build {
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
                        };
                        candidates.push(candidate);
                    }
                    profile_end(PROFILE_STEP_BUILD_NEW_FRONTIER_CANDIDATES, profile);
                }
                let frontier = Frontier {
                    dir_door_idx: door.dir_door_idx,
                    component: self.room_part_component(common, action.room_idx, door.part_idx),
                    kind: common.room_dir_door[door.direction as usize][door.dir_door_idx as usize]
                        .kind,
                    candidates,
                };
                self.frontier.insert(door_loc, frontier);
            }
        }

        // Filter existing frontiers to remove geometries blocked by the new room or with no unused representatives.
        if candidate_update == CandidateUpdate::Build {
            let profile = profile_start();
            let geometry_unused_count = &self.geometry_unused_count;
            for frontier in self.frontier.values_mut() {
                frontier.candidates.retain(|cand| {
                    geometry_unused_count[cand.geometry_idx as usize] > 0
                        && !common.has_geometry_intersection(
                            action_geometry_idx,
                            action.x,
                            action.y,
                            cand.geometry_idx,
                            cand.x,
                            cand.y,
                        )
                });
            }
            profile_end(PROFILE_STEP_FILTER_EXISTING_FRONTIERS, profile);
        }
    }

    pub fn finish(&mut self) {
        self.finished = true;
    }

    fn add_room_components_and_edges(&mut self, action: Action, common: &CommonData) {
        let room_idx = action.room_idx;
        let room = &common.room[room_idx as usize];
        let mut attached_room_parts = vec![Vec::new(); room.door_group_count];
        for door in &room.doors {
            let door_loc = DoorLocation::new(door, action.x, action.y);
            if let Some(frontier) = self.frontier.get(&door_loc) {
                let attached_room_part = common.room_dir_door[door.direction.opposite() as usize]
                    [frontier.dir_door_idx as usize]
                    .room_part_idx;
                attached_room_parts[door.part_idx as usize].push(attached_room_part);
            }
        }

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

    pub fn get_candidates(&mut self, common: &CommonData, max_candidates: usize) -> Vec<Action> {
        if self.actions.is_empty() {
            return vec![self.get_initial_action(common)];
        }
        let smallest_frontier_size = self
            .frontier
            .values()
            .map(|frontier| frontier.candidates.len())
            .filter(|&x| x > 0)
            .min()
            .unwrap_or(1);
        let candidate_geometries = {
            self.frontier
                .iter()
                .filter(|(_, frontier)| frontier.candidates.len() == smallest_frontier_size)
                .min_by_key(|(door_loc, _)| *door_loc)
                .map(|(_, frontier)| frontier.candidates.clone())
                .unwrap_or_default()
        };
        let mut candidates = Vec::with_capacity(candidate_geometries.len());
        for candidate in candidate_geometries {
            self.push_candidate_representatives(common, candidate, &mut candidates);
        }
        candidates.shuffle(&mut self.rng);
        candidates.truncate(max_candidates);
        candidates
    }

    pub fn get_candidates_with_outcomes(
        &mut self,
        common: &CommonData,
        max_candidates: usize,
    ) -> (Vec<Action>, Vec<Outcomes>) {
        let candidates = self.get_candidates(common, max_candidates);
        let outcomes = candidates
            .iter()
            .map(|&candidate| self.outcomes_after_candidate(common, candidate))
            .collect();
        (candidates, outcomes)
    }

    pub fn outcomes_after_candidate(&self, common: &CommonData, candidate: Action) -> Outcomes {
        let mut env = self.clone_for_lookahead();
        env.step(candidate, common);
        env.outcomes(common)
    }

    fn clone_for_lookahead(&self) -> Self {
        Self {
            // Lookahead only calls step() and outcomes(); candidate RNG state must not advance.
            rng: rand::rngs::StdRng::seed_from_u64(0),
            map_size: self.map_size,
            actions: self.actions.clone(),
            finished: self.finished,
            frontier: self.frontier.clone(),
            door_matches: self.door_matches.clone(),
            room_used: self.room_used.clone(),
            room_x: self.room_x.clone(),
            room_y: self.room_y.clone(),
            geometry_unused_count: self.geometry_unused_count.clone(),
            connection_variant_unused_count: self.connection_variant_unused_count.clone(),
            room_part_component: self.room_part_component.clone(),
            scc_dag: self.scc_dag.clone(),
            occupancy: self.occupancy.clone(),
        }
    }

    fn apply_feature_candidate(
        &mut self,
        candidate: Action,
        common: &CommonData,
    ) -> FeatureSnapshot {
        let frontier = std::mem::take(&mut self.frontier);
        self.frontier = frontier
            .iter()
            .map(|(&location, frontier)| {
                (
                    location,
                    Frontier {
                        dir_door_idx: frontier.dir_door_idx,
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
        };
        self.step_for_features(candidate, common);
        snapshot
    }

    fn restore_feature_candidate(&mut self, candidate: Action, snapshot: FeatureSnapshot) {
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
    }

    pub fn feature_frontier_count_after_candidate(
        &self,
        candidate: Action,
        common: &CommonData,
    ) -> usize {
        if self.finished || candidate.room_idx >= common.room.len() as RoomIdx {
            return self.frontier.len();
        }
        let mut frontier_count = self.frontier.len();
        let mut toggled_locations = HashSet::new();
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
        let mut sorted_frontiers = if config.has_frontier_features() {
            self.frontier.iter().collect::<Vec<_>>()
        } else {
            vec![]
        };
        sorted_frontiers.sort_unstable_by_key(|(location, _)| **location);
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
        if config.frontier_neighbor {
            let locations = sorted_frontiers
                .iter()
                .map(|(location, _)| **location)
                .collect::<Vec<_>>();
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
        if config.frontier_neighbor_flags {
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
        }
        Features {
            inventory,
            room_x: if config.room_position {
                self.room_x.clone()
            } else {
                vec![]
            },
            room_y: if config.room_position {
                self.room_y.clone()
            } else {
                vec![]
            },
            room_placed,
            frontier,
            frontier_occupancy,
            frontier_neighbor,
            frontier_neighbor_pair,
            connection_reachability,
            frontier_connection_reachability,
        }
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
        let snapshot = self.apply_feature_candidate(candidate, common);
        let features = self.features_with_occupancy(
            common,
            config,
            &self.occupancy,
            extra_occupied,
            frontier_neighbor_algorithm,
            frontier_neighbor_count,
            frontier_window_size,
        );
        self.restore_feature_candidate(candidate, snapshot);
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

    pub fn outcomes(&self, common: &CommonData) -> Outcomes {
        let mut door_valid = vec![];
        for dir in 0..NUM_DIRS {
            let matches = &self.door_matches[dir];
            for (i, &m) in matches.iter().enumerate() {
                let outcome = if m != DirDoorIdx::MAX {
                    DoorValidOutcome::Valid
                } else if self.finished {
                    // The episode is ended, so any unmatched door is invalid.
                    DoorValidOutcome::Invalid
                } else {
                    // The door is not yet matched. It is invalid if there is no candidate that could connect to it,
                    // otherwise it is unknown.
                    let room_dir_door = &common.room_dir_door[dir][i];
                    let room_idx = room_dir_door.room_idx;
                    if self.room_used[room_idx as usize] {
                        match self.frontier.get(&DoorLocation::from_room_dir_door(
                            room_dir_door,
                            self.room_x[room_idx as usize],
                            self.room_y[room_idx as usize],
                        )) {
                            None => DoorValidOutcome::Invalid, // No frontier means this door is blocked by the new room.
                            Some(frontier) if frontier.candidates.is_empty() => {
                                DoorValidOutcome::Invalid
                            }
                            Some(_) => DoorValidOutcome::Unknown,
                        }
                    } else {
                        DoorValidOutcome::Unknown
                    }
                };
                door_valid.push(outcome);
            }
        }

        let frontier_reachability = if self.finished {
            None
        } else {
            Some(self.scc_dag_with_merged_frontiers())
        };
        let mut connections_valid = Vec::with_capacity(common.room_connection.len());
        for connection in &common.room_connection {
            let outcome = if self.room_used[connection.room_idx as usize] {
                let from_component =
                    self.room_part_component(common, connection.room_idx, connection.from_part);
                let to_component =
                    self.room_part_component(common, connection.room_idx, connection.to_part);
                if self.scc_dag.can_reach(from_component, to_component) {
                    DoorValidOutcome::Valid
                } else if let Some((frontier_merged_scc_dag, frontier_merged_component_remap)) =
                    &frontier_reachability
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
            };
            connections_valid.push(outcome);
        }

        Outcomes {
            door_valid,
            connections_valid,
        }
    }
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
                "doors": [
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "down", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            },
            {
                "map": [[1]],
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
        assert!(
            env.room_part_component
                .iter()
                .all(|&component| component == NO_COMPONENT)
        );
    }

    #[test]
    fn door_match_counts_include_unmatched_row_and_column() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "doors": [
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            },
            {
                "map": [[1]],
                "doors": [
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            },
            {
                "map": [[1]],
                "doors": [
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            },
            {
                "map": [[1]],
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
    fn connection_outcome_is_invalid_when_frontier_merge_cannot_make_path() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "doors": [
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "down", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            },
            {
                "map": [[1]],
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
                "doors": [
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "down", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            },
            {
                "map": [[1]],
                "doors": [
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [],
                "missing_connections": []
            },
            {
                "map": [[1]],
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
                "doors": [
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "down", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1]],
                "missing_connections": [[1, 0]]
            },
            {
                "map": [[1]],
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
    fn features_do_not_depend_on_frontier_candidate_lists() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
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
            r#"[{"map": [[1]], "doors": [], "connections": [], "missing_connections": []}]"#,
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
}
