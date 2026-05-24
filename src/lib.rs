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
fn get_behind_door_position(door: &Door, x: Coord, y: Coord) -> (Coord, Coord) {
    match door.direction {
        Direction::Left => (x + door.x - 1, y + door.y),
        Direction::Right => (x + door.x + 1, y + door.y),
        Direction::Up => (x + door.x, y + door.y - 1),
        Direction::Down => (x + door.x, y + door.y + 1),
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

impl IntersectionChecker {
    fn new(rooms: &[Room], map_size: (Coord, Coord)) -> Result<Self> {
        let mut min_x_cand = vec![];
        let mut max_x_cand = vec![];
        let mut min_y_cand = vec![];
        let mut max_y_cand = vec![];

        for room in rooms {
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
                let (door_x, door_y) = get_behind_door_position(door, 0, 0);
                min_x = min_x.min(door_x);
                max_x = max_x.max(door_x);
                min_y = min_y.min(door_y);
                max_y = max_y.max(door_y);
            }
            min_x_cand.push(-min_x);
            max_x_cand.push(map_size.0 - 1 - max_x);
            min_y_cand.push(-min_y);
            max_y_cand.push(map_size.1 - 1 - max_y);
        }

        for i in 0..rooms.len() {
            if min_x_cand[i] > max_x_cand[i] || min_y_cand[i] > max_y_cand[i] {
                bail!(
                    "Room id {} (index {}) cannot fit within the map boundaries",
                    rooms[i].room_id,
                    i
                );
            }
        }
        Ok(Self {
            rooms: rooms.to_vec(),
            map_size,
            min_x_cand,
            max_x_cand,
            min_y_cand,
            max_y_cand,
        })
    }

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
        let room1 = &self.rooms[room_id1 as usize];
        let room2 = &self.rooms[room_id2 as usize];
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

        'outer: for door1 in room1.doors.iter().flatten() {
            let (door_x1, door_y1) = get_behind_door_position(door1, x1, y1);
            let other_x = door_x1 - x2;
            let other_y = door_y1 - y2;
            if other_y >= 0
                && other_x >= 0
                && other_y < room2.map.len() as Coord
                && other_x < room2.map[0].len() as Coord
                && room2.map[other_y as usize][other_x as usize] != 0
            {
                for door2 in room2.doors.iter().flatten() {
                    let (door_x2, door_y2) = get_behind_door_position(door2, x2, y2);
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

        'outer: for door2 in room2.doors.iter().flatten() {
            let (door_x2, door_y2) = get_behind_door_position(door2, x2, y2);
            let other_x = door_x2 - x1;
            let other_y = door_y2 - y1;
            if other_y >= 0
                && other_x >= 0
                && other_y < room1.map.len() as Coord
                && other_x < room1.map[0].len() as Coord
                && room1.map[other_y as usize][other_x as usize] != 0
            {
                for door1 in room1.doors.iter().flatten() {
                    let (door_x1, door_y1) = get_behind_door_position(door1, x1, y1);
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

struct RoomDoorData {
    x: Coord,
    y: Coord,
    direction: Direction,
    dir_door_idx: DirDoorIdx,
}

struct RoomData {
    doors: Vec<RoomDoorData>,
}

struct DirDoorData {
    room_idx: RoomIdx,
    door_idx: DoorIdx,
}

struct CommonData {
    room_data: Vec<RoomData>,
    // for each direction, a list of all doors in that direction across all rooms
    dir_door: [Vec<DirDoorData>; NUM_DIRS],
    intersection_checker: IntersectionChecker,
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

struct DoorMatches {
    left: Vec<DirDoorIdx>, // for each left door, the index of the matching right door on the other side (or DoorIdx::MAX if none)
    right: Vec<DirDoorIdx>, // for each right door, the index of the matching left door on the other side (or DoorIdx::MAX if none)
    up: Vec<DirDoorIdx>, // for each up door, the index of the matching down door on the other side (or DoorIdx::MAX if none)
    down: Vec<DirDoorIdx>, // for each down door, the index of the matching up door on the other side (or DoorIdx::MAX if none)
}

pub struct Environment {
    rng: rand::rngs::StdRng, // for random choice of initial room placement
    actions: Vec<Action>,    // history of room placements so far
    frontier: HashMap<DoorLocation, Frontier>, // info about each unconnected door on the map
    // grouped by door direction, for each door, the index of the matching door on the other side (or DoorIdx::MAX if none)
    door_matches: [Vec<DirDoorIdx>; NUM_DIRS],
    room_used: BitVec, // whether each room has been used
}

impl Environment {
    fn new(rooms: &[Room], common: &CommonData) -> Self {
        let mut env = Self {
            rng: rand::make_rng(),
            actions: vec![],
            frontier: HashMap::new(),
            door_matches: [
                vec![DoorIdx::MAX; common.dir_door[0].len()],
                vec![DoorIdx::MAX; common.dir_door[1].len()],
                vec![DoorIdx::MAX; common.dir_door[2].len()],
                vec![DoorIdx::MAX; common.dir_door[3].len()],
            ],
            room_used: BitVec::repeat(false, rooms.len()),
        };
        let action = env.get_initial_action(common);
        env.step(action, common);
        env
    }

    fn get_initial_action(&mut self, common: &CommonData) -> Action {
        // Select a room and position uniformly at random.
        let room_idx = self.rng.random_range(0..common.rooms.len() as RoomIdx);
        let x = self.rng.random_range(
            common.intersection_checker.min_x_cand[room_idx as usize]
                ..=common.intersection_checker.max_x_cand[room_idx as usize],
        );
        let y = self.rng.random_range(
            common.intersection_checker.min_y_cand[room_idx as usize]
                ..=common.intersection_checker.max_y_cand[room_idx as usize],
        );
        Action { room_idx, x, y }
    }

    fn step(&mut self, action: Action, common: &CommonData) {
        self.actions.push(action);
        self.room_used.set(action.room_idx as usize, true);
        let room = &common.room_data[action.room_idx as usize];

        // Remove the frontiers that the new room connects to (if any),
        // and update the frontier with the new unconnected doors of the new room.
        for (door_idx, door) in room.doors.iter().enumerate() {
            let door_loc = DoorLocation::new(door, action.x, action.y);
            if let Some(frontier) = self.frontier.remove(&door_loc) {
                // This frontier is now connected, so remove it and mark the doors as connected:
                let door_idx = door.dir_door_idx;
                let frontier_idx = frontier.dir_door_idx;
                match door.direction {
                    Direction::Right => {
                        assert!(frontier.direction == Direction::Left);
                        self.door_matches.right[door.dir_door_idx as usize] = frontier.dir_door_idx;
                        self.door_matches.left[frontier.dir_door_idx as usize] = door.dir_door_idx;
                    }
                    Direction::Left => {
                        assert!(frontier.direction == Direction::Right);
                        self.door_matches.left[door.dir_door_idx as usize] = frontier.dir_door_idx;
                        self.door_matches.right[frontier.dir_door_idx as usize] = door.dir_door_idx;
                    }
                    Direction::Down => {
                        assert!(frontier.direction == Direction::Up);
                        self.door_matches.down[door.dir_door_idx as usize] = frontier.dir_door_idx;
                        self.door_matches.up[frontier.dir_door_idx as usize] = door.dir_door_idx;
                    }
                    Direction::Up => {
                        assert!(frontier.direction == Direction::Down);
                        self.door_matches.up[door.dir_door_idx as usize] = frontier.dir_door_idx;
                        self.door_matches.down[frontier.dir_door_idx as usize] = door.dir_door_idx;
                    }
                }
            } else {
                // This door is not connected to any existing frontier, so it becomes a new frontier.
                let mut candidates = vec![];

                let mut frontier = Frontier {
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
                !common.intersection_checker.has_intersection(
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
        let intersection_checker = IntersectionChecker::new(&rooms, map_size)?;

        let mut room_data = vec![];
        let mut left_door = vec![];
        let mut right_door = vec![];
        let mut up_door = vec![];
        let mut down_door = vec![];
        for (room_idx, room) in rooms.iter().enumerate() {
            let mut door_data = vec![];
            for (door_idx, door) in room.doors.iter().flatten().enumerate() {
                let dir = match door.direction {
                    Direction::Left => {
                        left_door.push(DirDoorData {
                            room_idx: room_idx as RoomIdx,
                            door_idx: door_idx as DoorIdx,
                        });
                    }
                    Direction::Right => {
                        right_door.push(DirDoorData {
                            room_idx: room_idx as RoomIdx,
                            door_idx: door_idx as DoorIdx,
                        });
                    }
                    Direction::Up => {
                        up_door.push(DirDoorData {
                            room_idx: room_idx as RoomIdx,
                            door_idx: door_idx as DoorIdx,
                        });
                    }
                    Direction::Down => {
                        down_door.push(DirDoorData {
                            room_idx: room_idx as RoomIdx,
                            door_idx: door_idx as DoorIdx,
                        });
                    }
                };
                let dir_idx = next_dir_idx[dir];
                next_dir_idx[dir] = next_dir_idx[dir].checked_add(1).with_context(|| {
                    format!(
                        "Too many doors in direction {:?}, exceeds DirDoorIdx capacity",
                        door.direction
                    )
                })?;
                door_data.push(RoomDoorData {
                    x: door.x,
                    y: door.y,
                    direction: door.direction,
                    dir_door_idx: dir_idx,
                });
            }
            room_data.push(RoomData { doors: door_data });
        }
        Ok(Self {
            room_data,
            total_left_doors: next_dir_idx[0],
            total_right_doors: next_dir_idx[1],
            total_up_doors: next_dir_idx[2],
            total_down_doors: next_dir_idx[3],
            intersection_checker,
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

    fn get_candidates(&self, start: usize, end: usize) -> Vec<()> {
        let mut candidates = vec![];
        // for (i, frontier) in self.frontier.iter().enumerate() {
        //     for room_id in start as RoomIdx..end as RoomIdx {
        //         if !self.room_used[room_id as usize] {
        //             for x in -10..=10 {
        //                 for y in -10..=10 {
        //                     if !IntersectionChecker::new(&self.rooms, (100, 100)).has_intersection(
        //                         frontier.candidates[0].room,
        //                         frontier.candidates[0].x,
        //                         frontier.candidates[0].y,
        //                         room_id,
        //                         x,
        //                         y,
        //                     ) {
        //                         candidates[i].push(Action {
        //                             room_idx: room_id,
        //                             x,
        //                             y,
        //                         });
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }
        candidates
    }
}

#[pymodule]
fn map_gen(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    Ok(())
}
