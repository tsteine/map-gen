use bitvec::vec::BitVec;
use hashbrown::HashMap;
use rand::SeedableRng;
use rand::prelude::*;

use crate::common::{
    Action, CommonData, ConnectionVariantIdx, Coord, DirDoorIdx, DoorLocation, DoorValidOutcome,
    GeometryIdx, NUM_DIRS, PartIdx, RoomIdx, RoomPartIdx, get_behind_door_position,
};
use crate::scc_dag::SccDag;

const NO_COMPONENT: usize = usize::MAX;

// Frontier: location of an unconnected door on the map.
#[derive(Debug)]
pub struct Frontier {
    dir_door_idx: DirDoorIdx,
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
    }

    pub fn initial_step(&mut self, common: &CommonData) {
        let action = self.get_initial_action(common);
        self.step(action, common);
    }

    fn get_initial_action(&mut self, common: &CommonData) -> Action {
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
        self.actions.push(action);
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
        self.room_used.set(action.room_idx as usize, true);
        self.room_x[action.room_idx as usize] = action.x;
        self.room_y[action.room_idx as usize] = action.y;
        self.geometry_unused_count[action_geometry_idx as usize] -= 1;
        self.connection_variant_unused_count[connection_variant_idx as usize] -= 1;
        self.add_room_components_and_edges(action, common);

        // Remove the frontiers that the new room connects to (if any),
        // and update the frontier with the new unconnected doors of the new room.
        for door in room.doors.iter() {
            let door_loc = DoorLocation::new(door, action.x, action.y);
            if let Some(frontier) = self.frontier.remove(&door_loc) {
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
            } else {
                // This door is not connected to any existing frontier, so it becomes a new frontier.
                // Check all doors with the given orientation, to list which ones could connect here.
                let (x1, y1) =
                    get_behind_door_position(door.direction, action.x + door.x, action.y + door.y);
                let mut candidates = vec![];
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
                let frontier = Frontier {
                    dir_door_idx: door.dir_door_idx,
                    candidates,
                };
                self.frontier.insert(door_loc, frontier);
            }
        }

        // Filter existing frontiers to remove geometries blocked by the new room or with no unused representatives.
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

    pub fn get_candidates(&mut self, common: &CommonData, max_candidates: usize) -> Vec<Action> {
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

    pub fn actions(&self) -> &[Action] {
        &self.actions
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

        let mut connections_valid = Vec::with_capacity(common.room_connection.len());
        for connection in &common.room_connection {
            let mut outcome = if self.room_used[connection.room_idx as usize] {
                let from_component =
                    self.room_part_component(common, connection.room_idx, connection.from_part);
                let to_component =
                    self.room_part_component(common, connection.room_idx, connection.to_part);
                if self.scc_dag.can_reach(to_component, from_component) {
                    DoorValidOutcome::Valid
                } else {
                    DoorValidOutcome::Unknown
                }
            } else {
                DoorValidOutcome::Unknown
            };
            if self.finished && outcome != DoorValidOutcome::Valid {
                outcome = DoorValidOutcome::Invalid;
            }
            connections_valid.push(outcome);
        }

        Outcomes {
            door_valid,
            connections_valid,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Room;

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
                "connections": [[0, 1]]
            },
            {
                "map": [[1]],
                "doors": [
                    [{"direction": "left", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": []
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
    fn connection_outcome_is_valid_when_destination_reaches_source() {
        let rooms_json = r#"
        [
            {
                "map": [[1]],
                "doors": [
                    [{"direction": "right", "x": 0, "y": 0, "kind": 0}],
                    [{"direction": "down", "x": 0, "y": 0, "kind": 0}]
                ],
                "connections": [[0, 1], [1, 0]]
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
        assert_eq!(outcomes.connections_valid.len(), 2);
        assert!(matches!(
            outcomes.connections_valid.as_slice(),
            [DoorValidOutcome::Valid, DoorValidOutcome::Valid]
        ));
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
                "connections": [[0, 1]]
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
            [DoorValidOutcome::Unknown]
        ));

        env.finish();
        let outcomes = env.outcomes(&common);
        assert!(matches!(
            outcomes.connections_valid.as_slice(),
            [DoorValidOutcome::Invalid]
        ));
    }
}
