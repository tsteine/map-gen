use bitvec::vec::BitVec;
use hashbrown::HashMap;
use rand::SeedableRng;
use rand::prelude::*;

use crate::common::{
    Action, CommonData, ConnectionVariantIdx, Coord, DirDoorIdx, DoorLocation, DoorValidOutcome,
    GeometryIdx, NUM_DIRS, RoomIdx, get_behind_door_position,
};

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
}

pub struct Environment {
    rng: rand::rngs::StdRng, // for randomly choosing the initial room placement
    map_size: (Coord, Coord),
    actions: Vec<Action>, // history of room placements so far
    frontier: HashMap<DoorLocation, Frontier>, // info about each unconnected door on the map
    // Grouped by door direction: for each door, the index of the matching door on the other side (or DirDoorIdx::MAX if none):
    door_matches: [Vec<DirDoorIdx>; NUM_DIRS],
    room_used: BitVec,                           // whether each room has been used
    room_x: Vec<Coord>, // x position of each room (only valid for used rooms)
    room_y: Vec<Coord>, // y position of each room (only valid for used rooms)
    geometry_unused_count: Vec<usize>, // number of unused room representatives for each geometry
    connection_variant_unused_count: Vec<usize>, // number of unused room representatives for each connection variant
}

impl Environment {
    pub fn new(common: &CommonData, map_size: (Coord, Coord), seed: u64) -> Self {
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            map_size,
            actions: vec![],
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
        }
    }

    pub fn clear(&mut self, common: &CommonData) {
        self.actions.clear();
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
        if action.room_idx >= common.room.len() as RoomIdx {
            // Dummy/invalid action: do nothing more.
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

    pub fn get_candidates(&mut self, common: &CommonData, max_candidates: usize) -> Vec<Action> {
        let smallest_frontier_size = self
            .frontier
            .values()
            .map(|frontier| frontier.candidates.len())
            .filter(|&x| x > 0)
            .min()
            .unwrap_or(1);
        let candidate_geometries = {
            let eligible_frontiers: Vec<&Frontier> = self
                .frontier
                .values()
                .filter(|frontier| frontier.candidates.len() == smallest_frontier_size)
                .collect();
            if eligible_frontiers.is_empty() {
                vec![]
            } else {
                let frontier = eligible_frontiers
                    .choose(&mut self.rng)
                    .expect("eligible_frontiers is not empty");
                frontier.candidates.clone()
            }
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
                } else if self.actions.len() == common.room.len() {
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
        Outcomes { door_valid }
    }
}
