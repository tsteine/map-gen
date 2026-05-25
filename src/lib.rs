use anyhow::{Result, bail};
use bitvec::vec::BitVec;
use hashbrown::HashMap;
use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use rand::{RngExt, SeedableRng};
use serde::Deserialize;

type RoomIdx = u8;
type Coord = i8;
type PartIdx = u8;
type DoorKind = u8;
type DoorIdx = u8; // index of a door among all doors in the same room
type DirDoorIdx = u8; // index of a door among all doors with the given direction, across all rooms

const NUM_DIRS: usize = 4; // left, right, up, down

#[derive(Clone, Deserialize)]
struct Room {
    room_id: i64,
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

#[derive(Copy, Clone, Debug, Deserialize, PartialEq, Eq)]
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
    direction: Direction,
    candidates: Vec<Action>, // possible actions to connect to this frontier
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
    kind: DoorKind, // TODO: probably remove this later
    dir_door_idx: DirDoorIdx,
}

struct RoomData {
    map: Vec<Vec<u8>>, // TODO: probably remove this later
    doors: Vec<RoomDoorData>,
    // Minimum and maximum x and y coordinates at which the room can be placed without going out of bounds.
    min_x: Coord,
    max_x: Coord,
    min_y: Coord,
    max_y: Coord,
}

struct DirDoorData {
    room_idx: RoomIdx,
    door_idx: DoorIdx,
    x: Coord,
    y: Coord,
}

struct CommonData {
    room: Vec<RoomData>,
    // set of pairs of room placements that would cause an intersection
    // intersection_set: HashSet<(RoomIdx, RoomIdx, Coord, Coord)>,
    intersection_idx: Vec<u32>, // maps a pair of room ids to the index of their intersection bits in the intersection_bitvec
    intersection_bitvec: BitVec,
    // for each direction, a list of all doors in that direction across all rooms
    dir_door: [Vec<DirDoorData>; NUM_DIRS],
}

impl CommonData {
    fn new(rooms: Vec<Room>) -> Result<Self> {
        let mut room_data = vec![];
        let mut dir_door: [Vec<DirDoorData>; NUM_DIRS] = std::array::from_fn(|_| vec![]);
        for (room_idx, room) in rooms.iter().enumerate() {
            let mut door_data = vec![];
            for (door_idx, door) in room.doors.iter().flatten().enumerate() {
                let dir_door_idx = dir_door[door.direction as usize].len() as DirDoorIdx;
                dir_door[door.direction as usize].push(DirDoorData {
                    room_idx: room_idx as RoomIdx,
                    door_idx: door_idx as DoorIdx,
                    x: door.x,
                    y: door.y,
                });
                door_data.push(RoomDoorData {
                    x: door.x,
                    y: door.y,
                    direction: door.direction,
                    kind: door.kind,
                    dir_door_idx,
                });
            }

            let mut min_x = Coord::MAX;
            let mut max_x = Coord::MIN;
            let mut min_y = Coord::MAX;
            let mut max_y = Coord::MIN;
            let room_width = room.map[0].len() as Coord;
            let room_height = room.map.len() as Coord;
            for y in 0..room_height {
                for x in 0..room_width {
                    if room.map[y as usize][x as usize] != 0 {
                        min_x = min_x.min(x as Coord);
                        max_x = max_x.max(x as Coord);
                        min_y = min_y.min(y as Coord);
                        max_y = max_y.max(y as Coord);
                    }
                }
            }
            for door in room.doors.iter().flatten() {
                let (door_x, door_y) = get_behind_door_position(door.direction, door.x, door.y);
                min_x = min_x.min(door_x);
                max_x = max_x.max(door_x);
                min_y = min_y.min(door_y);
                max_y = max_y.max(door_y);
            }
            if min_x > max_x || min_y > max_y {
                bail!(
                    "Room id {} (index {}) cannot fit within the map boundaries",
                    room.room_id,
                    room_idx
                );
            }

            room_data.push(RoomData {
                map: room.map.clone(),
                doors: door_data,
                min_x,
                max_x,
                min_y,
                max_y,
            });
        }

        let mut common = Self {
            room: room_data,
            dir_door,
            intersection_idx: vec![],
            intersection_bitvec: BitVec::new(),
        };
        common.build_intersection_set();
        println!(
            "Finished building intersection set with {} bits",
            common.intersection_bitvec.len()
        );
        Ok(common)
    }

    fn build_intersection_set(&mut self) {
        self.intersection_idx
            .resize(self.room.len() * self.room.len(), 0);
        for room_idx1 in 0..self.room.len() {
            let room1 = &self.room[room_idx1];
            for room_idx2 in room_idx1..self.room.len() {
                let room2 = &self.room[room_idx2];
                let x0 = -room2.max_x + room1.min_x;
                let x1 = room1.max_x - room2.min_x;
                let y0 = -room2.max_y + room1.min_y;
                let y1 = room1.max_y - room2.min_y;
                let bit_idx = self.intersection_bitvec.len();
                self.intersection_idx[room_idx1 * self.room.len() + room_idx2] = bit_idx as u32;
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        let b = self.slow_has_intersection(
                            room_idx1 as RoomIdx,
                            0,
                            0,
                            room_idx2 as RoomIdx,
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
    fn has_intersection(
        &self,
        mut room_id1: RoomIdx,
        mut x1: Coord,
        mut y1: Coord,
        mut room_id2: RoomIdx,
        mut x2: Coord,
        mut y2: Coord,
    ) -> bool {
        if room_id1 > room_id2 {
            std::mem::swap(&mut room_id1, &mut room_id2);
            std::mem::swap(&mut x1, &mut x2);
            std::mem::swap(&mut y1, &mut y2);
        }
        let room1 = &self.room[room_id1 as usize];
        let room2 = &self.room[room_id2 as usize];
        let x = x2 - x1;
        let y = y2 - y1;
        let x0 = -room2.max_x + room1.min_x;
        let x1 = room1.max_x - room2.min_x;
        let y0 = -room2.max_y + room1.min_y;
        let y1 = room1.max_y - room2.min_y;
        if x < x0 || x > x1 || y < y0 || y > y1 {
            // Bounding boxes do not intersect, so the rooms cannot intersect.
            return false;
        }
        let w = x1 - x0 + 1;
        let i = self.intersection_idx[room_id1 as usize * self.room.len() + room_id2 as usize];
        let bit_idx = i as usize + (y - y0) as usize * w as usize + (x - x0) as usize;
        self.intersection_bitvec[bit_idx]
    }

    // Check if placing room1 at (x1, y1) and room2 at (x2, y2) would cause an intersection.
    // This includes overlapping tiles or blocked or mismatched doors.
    // Slow method for computing the intersection_set, used during start-up.
    fn slow_has_intersection(
        &self,
        room_id1: RoomIdx,
        x1: Coord,
        y1: Coord,
        room_id2: RoomIdx,
        x2: Coord,
        y2: Coord,
    ) -> bool {
        let room1 = &self.room[room_id1 as usize];
        let room2 = &self.room[room_id2 as usize];
        for (dy, row) in room1.map.iter().enumerate() {
            for (dx, &tile) in row.iter().enumerate() {
                if tile != 0 {
                    let other_x = x1 - x2 + dx as Coord;
                    let other_y = y1 - y2 + dy as Coord;
                    if other_y >= 0
                        && other_x >= 0
                        && other_y < room2.map.len() as Coord
                        && other_x < room2.map[0].len() as Coord
                        && room2.map[other_y as usize][other_x as usize] != 0
                    {
                        return true; // Intersection detected
                    }
                }
            }
        }

        'outer: for door1 in room1.doors.iter() {
            let loc1 = DoorLocation::new(door1, x1, y1);
            let (door_x1, door_y1) =
                get_behind_door_position(door1.direction, x1 + door1.x, y1 + door1.y);
            let other_x = door_x1 - x2;
            let other_y = door_y1 - y2;
            if other_y >= 0
                && other_x >= 0
                && other_y < room2.map.len() as Coord
                && other_x < room2.map[0].len() as Coord
                && room2.map[other_y as usize][other_x as usize] != 0
            {
                for door2 in room2.doors.iter() {
                    let loc2 = DoorLocation::new(door2, x2, y2);
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

        'outer: for door2 in room2.doors.iter() {
            let loc2 = DoorLocation::new(door2, x2, y2);
            let (door_x2, door_y2) =
                get_behind_door_position(door2.direction, x2 + door2.x, y2 + door2.y);
            let other_x = door_x2 - x1;
            let other_y = door_y2 - y1;
            if other_y >= 0
                && other_x >= 0
                && other_y < room1.map.len() as Coord
                && other_x < room1.map[0].len() as Coord
                && room1.map[other_y as usize][other_x as usize] != 0
            {
                for door1 in room1.doors.iter() {
                    let loc1 = DoorLocation::new(door1, x1, y1);
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
    // Get the DoorLocation for a door given the room placement, where (x0, y0) is the
    // location of the room's top-left corner on the map.
    fn new(door: &RoomDoorData, x0: Coord, y0: Coord) -> Self {
        let (x, y) = match door.direction {
            Direction::Left => (x0 + door.x, y0 + door.y),
            Direction::Right => (x0 + door.x + 1, y0 + door.y),
            Direction::Up => (x0 + door.x, y0 + door.y),
            Direction::Down => (x0 + door.x, y0 + door.y + 1),
        };
        let vertical = matches!(door.direction, Direction::Up | Direction::Down);
        Self { x, y, vertical }
    }
}

pub struct Environment {
    rng: rand::rngs::StdRng, // for randomly choosing the initial room placement
    map_size: (Coord, Coord),
    actions: Vec<Action>, // history of room placements so far
    frontier: HashMap<DoorLocation, Frontier>, // info about each unconnected door on the map
    // Grouped by door direction: for each door, the index of the matching door on the other side (or DoorIdx::MAX if none):
    door_matches: [Vec<DirDoorIdx>; NUM_DIRS],
    room_used: BitVec, // whether each room has been used
}

impl Environment {
    fn new(rooms: &[Room], common: &CommonData, map_size: (Coord, Coord), seed: u64) -> Self {
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            map_size,
            actions: vec![],
            frontier: HashMap::new(),
            door_matches: std::array::from_fn(|i| vec![DoorIdx::MAX; common.dir_door[i].len()]),
            room_used: BitVec::repeat(false, rooms.len()),
        }
    }

    fn clear(&mut self) {
        self.actions.clear();
        self.frontier.clear();
        self.door_matches
            .iter_mut()
            .for_each(|matches| matches.fill(DoorIdx::MAX));
        self.room_used.fill(false);
    }

    fn initial_step(&mut self, common: &CommonData) {
        let action = self.get_initial_action(common);
        self.step(action, common);
    }

    fn get_initial_action(&mut self, common: &CommonData) -> Action {
        // Select a room and position uniformly at random.
        let room_idx = self.rng.random_range(0..common.room.len() as RoomIdx);
        let min_x = -common.room[room_idx as usize].min_x;
        let max_x = self.map_size.0 - 1 - common.room[room_idx as usize].max_x;
        let min_y = -common.room[room_idx as usize].min_y;
        let max_y = self.map_size.1 - 1 - common.room[room_idx as usize].max_y;
        let x = self.rng.random_range(min_x..=max_x);
        let y = self.rng.random_range(min_y..=max_y);
        Action { room_idx, x, y }
    }

    fn step(&mut self, action: Action, common: &CommonData) {
        self.actions.push(action);
        if action.room_idx >= common.room.len() as RoomIdx {
            // Dummy/invalid action: do nothing more.
            return;
        }
        self.room_used.set(action.room_idx as usize, true);
        let room = &common.room[action.room_idx as usize];

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
                'door: for opp_door in common.dir_door[door.direction.opposite() as usize].iter() {
                    if self.room_used[opp_door.room_idx as usize] {
                        // A room that is already used cannot be used again.
                        continue;
                    }
                    let room_x = x1 - opp_door.x;
                    let room_y = y1 - opp_door.y;
                    let room = &common.room[opp_door.room_idx as usize];
                    if room_x < -room.min_x
                        || room_x > self.map_size.0 - 1 - room.max_x
                        || room_y < -room.min_y
                        || room_y > self.map_size.1 - 1 - room.max_y
                    {
                        // The room cannot be placed at this position due to map boundaries.
                        continue;
                    }

                    for a in &self.actions {
                        if common.has_intersection(
                            a.room_idx,
                            a.x,
                            a.y,
                            opp_door.room_idx,
                            room_x,
                            room_y,
                        ) {
                            continue 'door;
                        }
                    }

                    // The room had no intersections with existing rooms, so it is a valid candidate at this frontier.
                    candidates.push(Action {
                        room_idx: opp_door.room_idx,
                        x: room_x,
                        y: room_y,
                    });
                }
                let frontier = Frontier {
                    dir_door_idx: door.dir_door_idx,
                    direction: door.direction,
                    candidates,
                };
                self.frontier.insert(door_loc, frontier);
            }
        }

        // Filter existing frontiers to remove those blocked by the new room or identical to it
        for frontier in self.frontier.values_mut() {
            frontier.candidates.retain(|cand| {
                !common.has_intersection(
                    action.room_idx,
                    action.x,
                    action.y,
                    cand.room_idx,
                    cand.x,
                    cand.y,
                ) && action.room_idx != cand.room_idx
            });
        }
    }
}

#[pyclass]
pub struct Engine {
    common_data: CommonData, // pre-computed data that can be shared across environments
    environments: Vec<Environment>, // list of parallel environments for batch processing
}

#[pymethods]
impl Engine {
    #[new]
    fn new(
        rooms_json: &str,
        map_size: (Coord, Coord),
        num_environments: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let rooms: Vec<Room> = serde_json::from_str(rooms_json)
            .map_err(|err| PyValueError::new_err(format!("failed to parse rooms JSON: {err}")))?;
        let common_data = CommonData::new(rooms.clone())?;
        let mut environments = Vec::with_capacity(num_environments);
        for i in 0..num_environments {
            environments.push(Environment::new(
                &rooms,
                &common_data,
                map_size,
                seed ^ i as u64,
            ));
        }
        Ok(Self {
            common_data,
            environments,
        })
    }

    fn clear(&mut self) {
        for env in &mut self.environments {
            env.clear();
        }
    }

    fn initial_step(&mut self) {
        for env in &mut self.environments {
            env.initial_step(&self.common_data);
        }
    }

    #[allow(clippy::type_complexity)]
    fn get_actions<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray2<RoomIdx>>,
        Bound<'py, PyArray2<Coord>>,
        Bound<'py, PyArray2<Coord>>,
    )> {
        let mut room_idx = Vec::with_capacity(self.environments.len());
        let mut room_x = Vec::with_capacity(self.environments.len());
        let mut room_y = Vec::with_capacity(self.environments.len());
        for env in &self.environments {
            room_idx.push(env.actions.iter().map(|action| action.room_idx).collect());
            room_x.push(env.actions.iter().map(|action| action.x).collect());
            room_y.push(env.actions.iter().map(|action| action.y).collect());
        }
        Ok((
            PyArray2::from_vec2(py, &room_idx)
                .map_err(|_| PyValueError::new_err("environment action histories are ragged"))?,
            PyArray2::from_vec2(py, &room_x)
                .map_err(|_| PyValueError::new_err("environment action histories are ragged"))?,
            PyArray2::from_vec2(py, &room_y)
                .map_err(|_| PyValueError::new_err("environment action histories are ragged"))?,
        ))
    }

    fn step<'py>(
        &mut self,
        room_idx: PyReadonlyArray1<'py, RoomIdx>,
        room_x: PyReadonlyArray1<'py, Coord>,
        room_y: PyReadonlyArray1<'py, Coord>,
        start: usize,
    ) -> PyResult<()> {
        let room_idx = room_idx
            .as_slice()
            .map_err(|_| PyValueError::new_err("room_idx must be a contiguous 1D numpy array"))?;
        let room_x = room_x
            .as_slice()
            .map_err(|_| PyValueError::new_err("room_x must be a contiguous 1D numpy array"))?;
        let room_y = room_y
            .as_slice()
            .map_err(|_| PyValueError::new_err("room_y must be a contiguous 1D numpy array"))?;

        if room_idx.len() != room_x.len() || room_idx.len() != room_y.len() {
            return Err(PyValueError::new_err(format!(
                "room_idx, room_x, and room_y must have the same length; got {}, {}, and {}",
                room_idx.len(),
                room_x.len(),
                room_y.len()
            )));
        }

        let end = start + room_idx.len();
        if end > self.environments.len() {
            return Err(PyValueError::new_err(format!(
                "action arrays with length {} starting at {} exceed num_environments {}",
                room_idx.len(),
                start,
                self.environments.len(),
            )));
        }

        for (((env, &room_idx), &x), &y) in self.environments[start..end]
            .iter_mut()
            .zip(room_idx.iter())
            .zip(room_x.iter())
            .zip(room_y.iter())
        {
            let action = Action { room_idx, x, y };
            env.step(action, &self.common_data);
        }
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn get_candidates<'py>(
        &mut self,
        py: Python<'py>,
        max_candidates: usize,
        start: usize,
        end: usize,
    ) -> PyResult<(
        Bound<'py, PyArray2<RoomIdx>>,
        Bound<'py, PyArray2<Coord>>,
        Bound<'py, PyArray2<Coord>>,
    )> {
        if start > end || end > self.environments.len() {
            return Err(PyValueError::new_err(format!(
                "environment range [{}, {}) is invalid for num_environments {}",
                start,
                end,
                self.environments.len()
            )));
        }

        let num_environments = end - start;
        let mut room_idx = Vec::with_capacity(num_environments);
        let mut room_x = Vec::with_capacity(num_environments);
        let mut room_y = Vec::with_capacity(num_environments);

        for env in self.environments[start..end].iter_mut() {
            let smallest_frontier_size = env
                .frontier
                .values()
                .map(|frontier| frontier.candidates.len())
                .filter(|&x| x > 0)
                .min()
                .unwrap_or(1);
            let eligible_frontiers: Vec<&Frontier> = env
                .frontier
                .values()
                .filter(|frontier| frontier.candidates.len() == smallest_frontier_size)
                .collect();
            let mut candidates = if eligible_frontiers.is_empty() {
                vec![]
            } else {
                let frontier = eligible_frontiers
                    .choose(&mut env.rng)
                    .expect("eligible_frontiers is not empty");
                let mut candidates = frontier.candidates.clone();
                candidates.shuffle(&mut env.rng);
                candidates.truncate(max_candidates);
                candidates
            };
            let dummy_candidate = Action {
                room_idx: self.common_data.room.len() as RoomIdx, // an invalid room index to indicate no-op
                x: 0,
                y: 0,
            };
            candidates.resize(max_candidates, dummy_candidate);

            room_idx.push(
                candidates
                    .iter()
                    .map(|candidate| candidate.room_idx)
                    .collect(),
            );
            room_x.push(candidates.iter().map(|candidate| candidate.x).collect());
            room_y.push(candidates.iter().map(|candidate| candidate.y).collect());
        }

        Ok((
            PyArray2::from_vec2(py, &room_idx)?,
            PyArray2::from_vec2(py, &room_x)?,
            PyArray2::from_vec2(py, &room_y)?,
        ))
    }
}

#[pymodule]
fn map_gen(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    Ok(())
}
