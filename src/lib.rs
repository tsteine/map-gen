use anyhow::Result;
use bitvec::vec::BitVec;
use hashbrown::HashMap;
use pyo3::prelude::*;
use rand::prelude::*;
use rand::{RngExt, SeedableRng};
use serde::Deserialize;

mod engine;
use engine::{Engine, EnvironmentGroup};

type RoomIdx = u8;
type GeometryIdx = u8;
type ConnectionVariantIdx = u8;
type Coord = i8;
type PartIdx = u8;
type DoorKind = u8;
type DirDoorIdx = u8; // index of a door among all doors with the given direction, across all rooms

const NUM_DIRS: usize = 4; // left, right, up, down

#[derive(Clone, Deserialize)]
struct Room {
    map: Vec<Vec<u8>>,
    doors: Vec<Vec<Door>>,
    connections: Vec<(PartIdx, PartIdx)>,
}

#[derive(Clone, Debug, Deserialize)]
struct Door {
    direction: Direction,
    x: Coord,
    y: Coord,
    kind: DoorKind,
}

#[derive(Copy, Clone, Debug, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
#[repr(u8)]
enum Direction {
    Left = 0,
    Right = 1,
    Up = 2,
    Down = 3,
}

impl Direction {
    fn opposite(&self) -> Self {
        match self {
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
            Direction::Up => Direction::Down,
            Direction::Down => Direction::Up,
        }
    }
}

// Action: a placement of a room. The top-left corner is placed at (x, y) on the map.
#[derive(Copy, Clone, Debug)]
pub struct Action {
    room_idx: RoomIdx,
    x: Coord,
    y: Coord,
}

// Frontier: location of an unconnected door on the map.
#[derive(Debug)]
pub struct Frontier {
    dir_door_idx: DirDoorIdx,
    candidates: Vec<GeometryAction>, // possible geometry placements to connect to this frontier
}

// Get the coordinates of the tile behind a door:
fn get_behind_door_position(direction: Direction, x: Coord, y: Coord) -> (Coord, Coord) {
    match direction {
        Direction::Left => (x - 1, y),
        Direction::Right => (x + 1, y),
        Direction::Up => (x, y - 1),
        Direction::Down => (x, y + 1),
    }
}

struct RoomDoorData {
    x: Coord,
    y: Coord,
    direction: Direction,
    dir_door_idx: DirDoorIdx,
}

struct RoomData {
    geometry_idx: GeometryIdx,
    connection_variant_idx: ConnectionVariantIdx,
    doors: Vec<RoomDoorData>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct GeometryDoorData {
    x: Coord,
    y: Coord,
    direction: Direction,
    kind: DoorKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct GeometryKey {
    map: Vec<Vec<u8>>,
    doors: Vec<GeometryDoorData>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ConnectionsKey {
    connections: Vec<(PartIdx, PartIdx)>,
}

struct GeometryData {
    map: Vec<Vec<u8>>,
    doors: Vec<GeometryDoorData>,
    // Minimum and maximum x and y coordinates at which the room can be placed without going out of bounds.
    min_x: Coord,
    max_x: Coord,
    min_y: Coord,
    max_y: Coord,
}

struct GeometryDirDoorData {
    geometry_idx: GeometryIdx,
    x: Coord,
    y: Coord,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct GeometryAction {
    geometry_idx: GeometryIdx,
    x: Coord,
    y: Coord,
}

struct CommonData {
    room: Vec<RoomData>,
    geometry: Vec<GeometryData>,
    geometry_rooms: Vec<Vec<RoomIdx>>,
    geometry_connection_variants: Vec<Vec<ConnectionVariantIdx>>,
    connection_variant_rooms: Vec<Vec<RoomIdx>>,
    // set of pairs of geometry placements that would cause an intersection
    intersection_idx: Vec<u32>, // maps a pair of geometry ids to the index of their intersection bits in the intersection_bitvec
    intersection_bitvec: BitVec,
    // for each direction, a list of all doors in that direction across all unique geometries
    geometry_dir_door: [Vec<GeometryDirDoorData>; NUM_DIRS],
    // for each direction, number of room doors in that direction across all rooms
    num_room_dir_doors: [usize; NUM_DIRS],
}

impl GeometryKey {
    fn from_room(room: &Room) -> Self {
        let map = room.map.clone();
        let mut doors: Vec<_> = room
            .doors
            .iter()
            .flatten()
            .map(|door| GeometryDoorData {
                x: door.x,
                y: door.y,
                direction: door.direction,
                kind: door.kind,
            })
            .collect();
        doors.sort_by_key(|door| (door.direction as u8, door.x, door.y, door.kind));
        Self { map, doors }
    }
}

impl ConnectionsKey {
    fn from_room(room: &Room) -> Self {
        let mut connections = room.connections.clone();
        connections.sort_unstable();
        Self { connections }
    }
}

impl GeometryData {
    fn new(key: &GeometryKey) -> Result<Self> {
        let mut min_x = Coord::MAX;
        let mut max_x = Coord::MIN;
        let mut min_y = Coord::MAX;
        let mut max_y = Coord::MIN;
        let room_width = key.map[0].len() as Coord;
        let room_height = key.map.len() as Coord;
        for y in 0..room_height {
            for x in 0..room_width {
                if key.map[y as usize][x as usize] != 0 {
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    min_y = min_y.min(y);
                    max_y = max_y.max(y);
                }
            }
        }
        for door in key.doors.iter() {
            let (door_x, door_y) = get_behind_door_position(door.direction, door.x, door.y);
            min_x = min_x.min(door_x);
            max_x = max_x.max(door_x);
            min_y = min_y.min(door_y);
            max_y = max_y.max(door_y);
        }
        Ok(Self {
            map: key.map.clone(),
            doors: key.doors.clone(),
            min_x,
            max_x,
            min_y,
            max_y,
        })
    }
}

impl CommonData {
    fn new(rooms: Vec<Room>) -> Result<Self> {
        let mut room_data = vec![];
        let mut geometry_data = vec![];
        let mut geometry_rooms = vec![];
        let mut geometry_connection_variants = vec![];
        let mut connection_variant_rooms = vec![];
        let mut geometry_by_key = HashMap::new();
        let mut connection_variant_by_key = HashMap::new();
        let mut geometry_dir_door: [Vec<GeometryDirDoorData>; NUM_DIRS] =
            std::array::from_fn(|_| vec![]);
        let mut num_room_dir_doors = [0; NUM_DIRS];

        for (room_idx, room) in rooms.iter().enumerate() {
            let mut door_data = vec![];
            for door in room.doors.iter().flatten() {
                let dir_idx = door.direction as usize;
                let dir_door_idx = num_room_dir_doors[dir_idx] as DirDoorIdx;
                num_room_dir_doors[dir_idx] += 1;
                door_data.push(RoomDoorData {
                    x: door.x,
                    y: door.y,
                    direction: door.direction,
                    dir_door_idx,
                });
            }

            let geometry_key = GeometryKey::from_room(room);
            let geometry_idx = if let Some(&geometry_idx) = geometry_by_key.get(&geometry_key) {
                geometry_idx
            } else {
                let geometry_idx = geometry_data.len() as GeometryIdx;
                let geometry = GeometryData::new(&geometry_key)?;
                for door in geometry.doors.iter() {
                    geometry_dir_door[door.direction as usize].push(GeometryDirDoorData {
                        geometry_idx,
                        x: door.x,
                        y: door.y,
                    });
                }
                geometry_data.push(geometry);
                geometry_rooms.push(vec![]);
                geometry_connection_variants.push(vec![]);
                geometry_by_key.insert(geometry_key, geometry_idx);
                geometry_idx
            };

            let connections_key = ConnectionsKey::from_room(room);
            let connection_variant_idx = if let Some(&connection_variant_idx) =
                connection_variant_by_key.get(&(geometry_idx, connections_key.clone()))
            {
                connection_variant_idx
            } else {
                let connection_variant_idx = connection_variant_rooms.len() as ConnectionVariantIdx;
                connection_variant_rooms.push(vec![]);
                geometry_connection_variants[geometry_idx as usize].push(connection_variant_idx);
                connection_variant_by_key
                    .insert((geometry_idx, connections_key), connection_variant_idx);
                connection_variant_idx
            };

            geometry_rooms[geometry_idx as usize].push(room_idx as RoomIdx);
            connection_variant_rooms[connection_variant_idx as usize].push(room_idx as RoomIdx);
            room_data.push(RoomData {
                geometry_idx,
                connection_variant_idx,
                doors: door_data,
            });
        }

        let mut common = Self {
            room: room_data,
            geometry: geometry_data,
            geometry_rooms,
            geometry_connection_variants,
            connection_variant_rooms,
            intersection_idx: vec![],
            intersection_bitvec: BitVec::new(),
            geometry_dir_door,
            num_room_dir_doors,
        };
        common.build_intersection_set();
        println!(
            "Finished building intersection set with {} bits across {} geometries",
            common.intersection_bitvec.len(),
            common.geometry.len()
        );
        Ok(common)
    }

    fn build_intersection_set(&mut self) {
        self.intersection_idx
            .resize(self.geometry.len() * self.geometry.len(), 0);
        for geometry_idx1 in 0..self.geometry.len() {
            let geometry1 = &self.geometry[geometry_idx1];
            for geometry_idx2 in geometry_idx1..self.geometry.len() {
                let geometry2 = &self.geometry[geometry_idx2];
                let x0 = -geometry2.max_x + geometry1.min_x;
                let x1 = geometry1.max_x - geometry2.min_x;
                let y0 = -geometry2.max_y + geometry1.min_y;
                let y1 = geometry1.max_y - geometry2.min_y;
                let bit_idx = self.intersection_bitvec.len();
                self.intersection_idx[geometry_idx1 * self.geometry.len() + geometry_idx2] =
                    bit_idx as u32;
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        let b = self.slow_has_geometry_intersection(
                            geometry_idx1 as GeometryIdx,
                            0,
                            0,
                            geometry_idx2 as GeometryIdx,
                            x,
                            y,
                        );
                        self.intersection_bitvec.push(b);
                    }
                }
            }
        }
    }

    // Fast method using the pre-computed intersection_set:
    fn has_geometry_intersection(
        &self,
        mut geometry_id1: GeometryIdx,
        mut x1: Coord,
        mut y1: Coord,
        mut geometry_id2: GeometryIdx,
        mut x2: Coord,
        mut y2: Coord,
    ) -> bool {
        if geometry_id1 > geometry_id2 {
            std::mem::swap(&mut geometry_id1, &mut geometry_id2);
            std::mem::swap(&mut x1, &mut x2);
            std::mem::swap(&mut y1, &mut y2);
        }
        let geometry1 = &self.geometry[geometry_id1 as usize];
        let geometry2 = &self.geometry[geometry_id2 as usize];
        let x = x2 - x1;
        let y = y2 - y1;
        let x0 = -geometry2.max_x + geometry1.min_x;
        let x1 = geometry1.max_x - geometry2.min_x;
        let y0 = -geometry2.max_y + geometry1.min_y;
        let y1 = geometry1.max_y - geometry2.min_y;
        if x < x0 || x > x1 || y < y0 || y > y1 {
            // Bounding boxes do not intersect, so the geometries cannot intersect.
            return false;
        }
        let w = x1 - x0 + 1;
        let i = self.intersection_idx
            [geometry_id1 as usize * self.geometry.len() + geometry_id2 as usize];
        let bit_idx = i as usize + (y - y0) as usize * w as usize + (x - x0) as usize;
        self.intersection_bitvec[bit_idx]
    }

    // Check if placing geometry1 at (x1, y1) and geometry2 at (x2, y2) would cause an intersection.
    // This includes overlapping tiles or blocked or mismatched doors.
    // Slow method for computing the intersection_set, used during start-up.
    fn slow_has_geometry_intersection(
        &self,
        geometry_id1: GeometryIdx,
        x1: Coord,
        y1: Coord,
        geometry_id2: GeometryIdx,
        x2: Coord,
        y2: Coord,
    ) -> bool {
        let geometry1 = &self.geometry[geometry_id1 as usize];
        let geometry2 = &self.geometry[geometry_id2 as usize];
        for (dy, row) in geometry1.map.iter().enumerate() {
            for (dx, &tile) in row.iter().enumerate() {
                if tile != 0 {
                    let other_x = x1 - x2 + dx as Coord;
                    let other_y = y1 - y2 + dy as Coord;
                    if other_y >= 0
                        && other_x >= 0
                        && other_y < geometry2.map.len() as Coord
                        && other_x < geometry2.map[0].len() as Coord
                        && geometry2.map[other_y as usize][other_x as usize] != 0
                    {
                        return true; // Intersection detected
                    }
                }
            }
        }

        'outer: for door1 in geometry1.doors.iter() {
            let loc1 = DoorLocation::from_parts(door1.direction, door1.x, door1.y, x1, y1);
            let (door_x1, door_y1) =
                get_behind_door_position(door1.direction, x1 + door1.x, y1 + door1.y);
            let other_x = door_x1 - x2;
            let other_y = door_y1 - y2;
            if other_y >= 0
                && other_x >= 0
                && other_y < geometry2.map.len() as Coord
                && other_x < geometry2.map[0].len() as Coord
                && geometry2.map[other_y as usize][other_x as usize] != 0
            {
                for door2 in geometry2.doors.iter() {
                    let loc2 = DoorLocation::from_parts(door2.direction, door2.x, door2.y, x2, y2);
                    if loc1 == loc2
                        && door1.direction == door2.direction.opposite()
                        && door1.kind == door2.kind
                    {
                        continue 'outer; // Doors match, check next door1
                    }
                }
                return true; // Mismatched door
            }
        }

        'outer: for door2 in geometry2.doors.iter() {
            let loc2 = DoorLocation::from_parts(door2.direction, door2.x, door2.y, x2, y2);
            let (door_x2, door_y2) =
                get_behind_door_position(door2.direction, x2 + door2.x, y2 + door2.y);
            let other_x = door_x2 - x1;
            let other_y = door_y2 - y1;
            if other_y >= 0
                && other_x >= 0
                && other_y < geometry1.map.len() as Coord
                && other_x < geometry1.map[0].len() as Coord
                && geometry1.map[other_y as usize][other_x as usize] != 0
            {
                for door1 in geometry1.doors.iter() {
                    let loc1 = DoorLocation::from_parts(door1.direction, door1.x, door1.y, x1, y1);
                    if loc1 == loc2
                        && door1.direction == door2.direction.opposite()
                        && door1.kind == door2.kind
                    {
                        continue 'outer; // Doors match, check next door2
                    }
                }
                return true; // Mismatched door
            }
        }

        false // No intersection
    }
}

// DoorLocation: used as the key in the frontier hashmap to identify unconnected doors on the map.
// These are designed to match between the two sides of a door. A right-facing door gives the same
// DoorLocation as a left-facing door on the other side, and similarly for up/down doors.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct DoorLocation {
    x: Coord,
    y: Coord,
    vertical: bool,
}

impl DoorLocation {
    fn from_parts(
        direction: Direction,
        door_x: Coord,
        door_y: Coord,
        x0: Coord,
        y0: Coord,
    ) -> Self {
        let (x, y) = match direction {
            Direction::Left => (x0 + door_x, y0 + door_y),
            Direction::Right => (x0 + door_x + 1, y0 + door_y),
            Direction::Up => (x0 + door_x, y0 + door_y),
            Direction::Down => (x0 + door_x, y0 + door_y + 1),
        };
        let vertical = matches!(direction, Direction::Up | Direction::Down);
        Self { x, y, vertical }
    }

    // Get the DoorLocation for a door given the room placement, where (x0, y0) is the
    // location of the room's top-left corner on the map.
    fn new(door: &RoomDoorData, x0: Coord, y0: Coord) -> Self {
        Self::from_parts(door.direction, door.x, door.y, x0, y0)
    }
}

pub struct Environment {
    rng: rand::rngs::StdRng, // for randomly choosing the initial room placement
    map_size: (Coord, Coord),
    actions: Vec<Action>, // history of room placements so far
    frontier: HashMap<DoorLocation, Frontier>, // info about each unconnected door on the map
    // Grouped by door direction: for each door, the index of the matching door on the other side (or DirDoorIdx::MAX if none):
    door_matches: [Vec<DirDoorIdx>; NUM_DIRS],
    room_used: BitVec,                           // whether each room has been used
    geometry_unused_count: Vec<usize>, // number of unused room representatives for each geometry
    connection_variant_unused_count: Vec<usize>, // number of unused room representatives for each connection variant
}

impl Environment {
    fn new(common: &CommonData, map_size: (Coord, Coord), seed: u64) -> Self {
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            map_size,
            actions: vec![],
            frontier: HashMap::new(),
            door_matches: std::array::from_fn(|i| {
                vec![DirDoorIdx::MAX; common.num_room_dir_doors[i]]
            }),
            room_used: BitVec::repeat(false, common.room.len()),
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

    fn clear(&mut self, common: &CommonData) {
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

    fn initial_step(&mut self, common: &CommonData) {
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

    fn step(&mut self, action: Action, common: &CommonData) {
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

    fn get_candidates(&mut self, common: &CommonData, max_candidates: usize) -> Vec<Action> {
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
}

#[pymodule]
fn map_gen(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add_class::<EnvironmentGroup>()?;
    Ok(())
}
