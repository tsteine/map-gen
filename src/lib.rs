use anyhow::{Context, Result, bail};
use bitvec::vec::BitVec;
use hashbrown::HashMap;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::RngExt;
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
#[derive(Copy, Clone)]
pub struct Action {
    room_idx: RoomIdx,
    x: Coord,
    y: Coord,
}

// Frontier: location of an unconnected door on the map.
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

struct IntersectionChecker {
    rooms: Vec<Room>,
    map_size: (Coord, Coord),
    // For each room, the min/max x/y coordinate where room can be placed without going out of bounds:
    min_x_cand: Vec<Coord>,
    max_x_cand: Vec<Coord>,
    min_y_cand: Vec<Coord>,
    max_y_cand: Vec<Coord>,
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
    // for each direction, a list of all doors in that direction across all rooms
    dir_door: [Vec<DirDoorData>; NUM_DIRS],
}

impl CommonData {
    // Check if placing room1 at (x1, y1) and room2 at (x2, y2) would cause an intersection.
    // This includes overlapping tiles or blocked or mismatched doors.
    fn has_intersection(
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
                    let (door_x2, door_y2) =
                        get_behind_door_position(door2.direction, x2 + door2.x, y2 + door2.y);
                    if door_x1 == door_x2
                        && door_y1 == door_y2
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
                    let (door_x1, door_y1) =
                        get_behind_door_position(door1.direction, x1 + door1.x, y1 + door1.y);
                    if door_x1 == door_x2
                        && door_y1 == door_y2
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
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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
    actions: Vec<Action>,    // history of room placements so far
    frontier: HashMap<DoorLocation, Frontier>, // info about each unconnected door on the map
    // Grouped by door direction: for each door, the index of the matching door on the other side (or DoorIdx::MAX if none):
    door_matches: [Vec<DirDoorIdx>; NUM_DIRS],
    room_used: BitVec, // whether each room has been used
}

impl Environment {
    fn new(rooms: &[Room], common: &CommonData) -> Self {
        let mut env = Self {
            rng: rand::make_rng(),
            actions: vec![],
            frontier: HashMap::new(),
            door_matches: std::array::from_fn(|i| vec![DoorIdx::MAX; common.dir_door[i].len()]),
            room_used: BitVec::repeat(false, rooms.len()),
        };
        let action = env.get_initial_action(common);
        env.step(action, common);
        env
    }

    fn get_initial_action(&mut self, common: &CommonData) -> Action {
        // Select a room and position uniformly at random.
        let room_idx = self.rng.random_range(0..common.room.len() as RoomIdx);
        let min_x = common.room[room_idx as usize].min_x;
        let max_x = common.room[room_idx as usize].max_x;
        let min_y = common.room[room_idx as usize].min_y;
        let max_y = common.room[room_idx as usize].max_y;
        let x = self.rng.random_range(min_x..=max_x);
        let y = self.rng.random_range(min_y..=max_y);
        Action { room_idx, x, y }
    }

    fn step(&mut self, action: Action, common: &CommonData) {
        self.actions.push(action);
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
                    if room_x < room.min_x
                        || room_x > room.max_x
                        || room_y < room.min_y
                        || room_y > room.max_y
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

        // Filter existing frontiers to remove those blocked by the new room
        for frontier in self.frontier.values_mut() {
            frontier.candidates.retain(|cand| {
                !common.has_intersection(
                    action.room_idx,
                    action.x,
                    action.y,
                    cand.room_idx,
                    cand.x,
                    cand.y,
                )
            });
        }
    }
}

impl CommonData {
    fn new(rooms: Vec<Room>, map_size: (Coord, Coord)) -> Result<Self> {
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
                min_x: -min_x,
                max_x: map_size.0 - 1 - max_x,
                min_y: -min_y,
                max_y: map_size.1 - 1 - max_y,
            });
        }
        Ok(Self {
            room: room_data,
            dir_door,
        })
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
    fn new(rooms_json: &str, map_size: (Coord, Coord), batch_size: usize) -> PyResult<Self> {
        let rooms: Vec<Room> = serde_json::from_str(rooms_json)
            .map_err(|err| PyValueError::new_err(format!("failed to parse rooms JSON: {err}")))?;
        let common_data = CommonData::new(rooms.clone(), map_size)?;
        let mut environments = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            environments.push(Environment::new(&rooms, &common_data));
        }
        Ok(Self {
            common_data,
            environments,
        })
    }

    fn step(&mut self, actions: Vec<Action>) {
        for (env, action) in self.environments.iter_mut().zip(actions.into_iter()) {
            env.step(action, &self.common_data);
        }
    }

    // fn get_candidates(&self, start: usize, end: usize) -> Vec<()> {
    //     let mut candidates = vec![];
    //     // for (i, frontier) in self.frontier.iter().enumerate() {
    //     //     for room_id in start as RoomIdx..end as RoomIdx {
    //     //         if !self.room_used[room_id as usize] {
    //     //             for x in -10..=10 {
    //     //                 for y in -10..=10 {
    //     //                     if !IntersectionChecker::new(&self.rooms, (100, 100)).has_intersection(
    //     //                         frontier.candidates[0].room,
    //     //                         frontier.candidates[0].x,
    //     //                         frontier.candidates[0].y,
    //     //                         room_id,
    //     //                         x,
    //     //                         y,
    //     //                     ) {
    //     //                         candidates[i].push(Action {
    //     //                             room_idx: room_id,
    //     //                             x,
    //     //                             y,
    //     //                         });
    //     //                     }
    //     //                 }
    //     //             }
    //     //         }
    //     //     }
    //     // }
    //     candidates
    // }
}

#[pymodule]
fn map_gen(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    Ok(())
}
