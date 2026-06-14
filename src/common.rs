use anyhow::{Result, bail};
use bitvec::vec::BitVec;
use hashbrown::HashMap;
use serde::Deserialize;

pub type RoomIdx = u8; // index into provided room geometry JSON array
pub type GeometryIdx = u8; // flat index of unique room geometries (map + door layout)
pub type ConnectionVariantIdx = u8; // flat index of unique room types (map + door layout + connections)
pub type FrontierIdx = i16; // index into the sorted frontier feature rows
pub type DoorVariantIdx = i16; // index into door variants keyed by room type and local door identity
pub type Coord = i8; // x or y position on the map
pub type PartIdx = u8; // index of part within a room
pub type RoomPartIdx = u16; // flat index of part across all rooms
pub type GraphDistance = u8; // shortest-path cost between graph parts; u8::MAX means unreachable
pub type DoorKind = i8; // distinguishes different types of "doors", e.g. regular, elevator, and sand.
pub type DirDoorIdx = u8; // index of a door among all doors with the given direction, across all rooms

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i8)]
pub enum DoorValidOutcome {
    Unknown = -1,
    Valid = 0,
    Invalid = 1,
}

pub const NUM_DIRS: usize = 4; // left, right, up, down

#[derive(Clone, Deserialize)]
pub struct Room {
    map: Vec<Vec<u8>>,
    toilet_crossing_x: Vec<Coord>,
    special_type: Option<SpecialType>,
    doors: Vec<Vec<Door>>,
    connections: Vec<(PartIdx, PartIdx)>,
    missing_connections: Vec<(PartIdx, PartIdx)>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum SpecialType {
    Toilet,
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
pub enum Direction {
    Left = 0,
    Right = 1,
    Up = 2,
    Down = 3,
}

impl Direction {
    pub fn opposite(&self) -> Self {
        match self {
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
            Direction::Up => Direction::Down,
            Direction::Down => Direction::Up,
        }
    }
}

// Action: a placement of a room. The top-left corner is placed at (x, y) on the map.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Action {
    pub room_idx: RoomIdx,
    pub x: Coord,
    pub y: Coord,
}

// Get the coordinates of the tile behind a door:
pub fn get_behind_door_position(direction: Direction, x: Coord, y: Coord) -> (Coord, Coord) {
    match direction {
        Direction::Left => (x - 1, y),
        Direction::Right => (x + 1, y),
        Direction::Up => (x, y - 1),
        Direction::Down => (x, y + 1),
    }
}

// DoorLocation: used as the key in the frontier hashmap to identify unconnected doors on the map.
// These are designed to match between the two sides of a door. A right-facing door gives the same
// DoorLocation as a left-facing door on the other side, and similarly for up/down doors.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct DoorLocation {
    x: Coord,
    y: Coord,
    vertical: bool,
}

impl DoorLocation {
    pub fn x(&self) -> Coord {
        self.x
    }

    pub fn y(&self) -> Coord {
        self.y
    }

    pub fn vertical(&self) -> bool {
        self.vertical
    }

    pub fn from_parts(
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
    pub fn new(door: &RoomDoorData, x0: Coord, y0: Coord) -> Self {
        Self::from_parts(door.direction, door.x, door.y, x0, y0)
    }

    pub fn from_room_dir_door(door: &RoomDirDoorData, x0: Coord, y0: Coord) -> Self {
        Self::from_parts(door.direction, door.x, door.y, x0, y0)
    }
}

pub struct RoomDoorData {
    pub x: Coord,
    pub y: Coord,
    pub direction: Direction,
    pub dir_door_idx: DirDoorIdx,
    pub part_idx: PartIdx,
}

pub struct RoomDirDoorData {
    pub room_idx: RoomIdx,
    pub room_part_idx: RoomPartIdx,
    pub x: Coord,
    pub y: Coord,
    pub direction: Direction,
    pub kind: DoorKind,
}

pub struct RoomConnectionData {
    pub room_idx: RoomIdx,
    pub from_part: PartIdx,
    pub to_part: PartIdx,
}

pub struct OutputData {
    pub room_idx: RoomIdx,
    pub variant_outcome_idx: usize,
}

pub struct RoomData {
    pub geometry_idx: GeometryIdx,
    pub connection_variant_idx: ConnectionVariantIdx,
    pub doors: Vec<RoomDoorData>,
    pub door_group_offset: usize,
    pub door_group_count: usize,
    pub connections: Vec<(PartIdx, PartIdx)>,
    pub part_distances: Vec<GraphDistance>,
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
    toilet_crossing_x: Vec<Coord>,
    is_toilet: bool,
    doors: Vec<GeometryDoorData>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ConnectionsKey {
    connections: Vec<(PartIdx, PartIdx)>,
    missing_connections: Vec<(PartIdx, PartIdx)>,
}

pub struct GeometryData {
    pub map: Vec<Vec<u8>>,
    toilet_crossing_x: Vec<Coord>,
    is_toilet: bool,
    pub occupied_tiles: Vec<(Coord, Coord)>,
    doors: Vec<GeometryDoorData>,
    pub min_x: Coord,
    pub max_x: Coord,
    pub min_y: Coord,
    pub max_y: Coord,
}

pub struct GeometryDirDoorData {
    pub geometry_idx: GeometryIdx,
    pub x: Coord,
    pub y: Coord,
    pub kind: DoorKind,
}

pub struct CommonData {
    pub room: Vec<RoomData>,
    toilet_room_idx: Option<RoomIdx>,
    pub geometry: Vec<GeometryData>,
    pub geometry_rooms: Vec<Vec<RoomIdx>>,
    pub geometry_connection_variants: Vec<Vec<ConnectionVariantIdx>>,
    pub connection_variant_rooms: Vec<Vec<RoomIdx>>,
    // set of pairs of geometry placements that would cause an intersection
    intersection_idx: Vec<u32>, // maps a pair of geometry ids to the index of their intersection bits in the intersection_bitvec
    intersection_bitvec: BitVec,
    // for each direction, a list of all doors in that direction across all unique geometries
    pub geometry_dir_door: [Vec<GeometryDirDoorData>; NUM_DIRS],
    // for each direction, number of room doors in that direction across all rooms
    pub room_dir_door: [Vec<RoomDirDoorData>; NUM_DIRS],
    pub room_part: Vec<(RoomIdx, PartIdx)>,
    pub room_connection: Vec<RoomConnectionData>,
    pub door_output: Vec<OutputData>,
    pub connection_output: Vec<OutputData>,
    pub num_door_output_variants: usize,
    pub num_connection_output_variants: usize,
    door_variant_idx_by_key:
        HashMap<(ConnectionVariantIdx, Direction, Coord, Coord, DoorKind), DoorVariantIdx>,
}

impl GeometryKey {
    fn from_room(room: &Room) -> Self {
        let map = room.map.clone();
        let mut toilet_crossing_x = room.toilet_crossing_x.clone();
        toilet_crossing_x.sort_unstable();
        let is_toilet = room.special_type == Some(SpecialType::Toilet);
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
        Self {
            map,
            toilet_crossing_x,
            is_toilet,
            doors,
        }
    }
}

impl ConnectionsKey {
    fn from_room(room: &Room) -> Self {
        let mut connections = room.connections.clone();
        connections.sort_unstable();
        let mut missing_connections = room.missing_connections.clone();
        missing_connections.sort_unstable();
        Self {
            connections,
            missing_connections,
        }
    }
}

fn validate_missing_connections(room_idx: usize, room: &Room) -> Result<()> {
    let part_count = room.doors.len();
    let mut reachable = vec![vec![false; part_count]; part_count];
    for (part_idx, row) in reachable.iter_mut().enumerate() {
        row[part_idx] = true;
    }
    for &(from_part, to_part) in &room.connections {
        reachable[from_part as usize][to_part as usize] = true;
    }
    for via in 0..part_count {
        for from in 0..part_count {
            for to in 0..part_count {
                reachable[from][to] |= reachable[from][via] && reachable[via][to];
            }
        }
    }
    let mut component = vec![usize::MAX; part_count];
    let mut component_count = 0;
    for from in 0..part_count {
        if component[from] != usize::MAX {
            continue;
        }
        for to in 0..part_count {
            if reachable[from][to] && reachable[to][from] {
                component[to] = component_count;
            }
        }
        component_count += 1;
    }
    let minimum_count = if component_count <= 1 {
        0
    } else {
        let mut has_predecessor = vec![false; component_count];
        let mut has_successor = vec![false; component_count];
        for &(from_part, to_part) in &room.connections {
            let from = component[from_part as usize];
            let to = component[to_part as usize];
            if from != to {
                has_successor[from] = true;
                has_predecessor[to] = true;
            }
        }
        has_predecessor
            .iter()
            .filter(|&&value| !value)
            .count()
            .max(has_successor.iter().filter(|&&value| !value).count())
    };
    if room.missing_connections.len() != minimum_count {
        bail!(
            "room {room_idx} has {} missing connections, expected the minimum {minimum_count}",
            room.missing_connections.len()
        );
    }
    let mut closure = room.connections.clone();
    for &connection in &room.missing_connections {
        if closure.contains(&connection) {
            bail!("room {room_idx} has duplicate connection {connection:?}");
        }
        closure.push(connection);
    }
    let mut reachable = vec![vec![false; part_count]; part_count];
    for (part_idx, row) in reachable.iter_mut().enumerate() {
        row[part_idx] = true;
    }
    for &(from_part, to_part) in &closure {
        reachable[from_part as usize][to_part as usize] = true;
    }
    for via in 0..part_count {
        for from in 0..part_count {
            for to in 0..part_count {
                reachable[from][to] |= reachable[from][via] && reachable[via][to];
            }
        }
    }
    if reachable.iter().flatten().any(|&value| !value) {
        bail!("room {room_idx} connections and missing_connections are not strongly connected");
    }
    Ok(())
}

fn room_part_distances(room: &Room) -> Vec<GraphDistance> {
    let part_count = room.doors.len();
    let mut distances = vec![GraphDistance::MAX; part_count * part_count];
    for part_idx in 0..part_count {
        distances[part_idx * part_count + part_idx] = 0;
    }
    for &(from_part, to_part) in &room.connections {
        distances[from_part as usize * part_count + to_part as usize] = 0;
    }
    for via in 0..part_count {
        for from in 0..part_count {
            if distances[from * part_count + via] == GraphDistance::MAX {
                continue;
            }
            for to in 0..part_count {
                if distances[via * part_count + to] == GraphDistance::MAX {
                    continue;
                }
                distances[from * part_count + to] = 0;
            }
        }
    }
    distances
}

impl GeometryData {
    fn new(key: &GeometryKey) -> Result<Self> {
        let mut min_x = Coord::MAX;
        let mut max_x = Coord::MIN;
        let mut min_y = Coord::MAX;
        let mut max_y = Coord::MIN;
        let mut occupied_tiles = vec![];
        let room_width = key.map[0].len() as Coord;
        let room_height = key.map.len() as Coord;
        for y in 0..room_height {
            for x in 0..room_width {
                if key.map[y as usize][x as usize] != 0 {
                    occupied_tiles.push((x, y));
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
            toilet_crossing_x: key.toilet_crossing_x.clone(),
            is_toilet: key.is_toilet,
            occupied_tiles,
            doors: key.doors.clone(),
            min_x,
            max_x,
            min_y,
            max_y,
        })
    }
}

fn has_disallowed_toilet_crossing(
    toilet_x: Coord,
    toilet_y: Coord,
    crossed: &GeometryData,
    crossed_x: Coord,
    crossed_y: Coord,
) -> bool {
    let crossing_x = toilet_x - crossed_x;
    let crosses_room = (2..=7).any(|toilet_open_y| {
        let crossed_tile_y = toilet_y + toilet_open_y - crossed_y;
        crossed_tile_y >= 0
            && crossing_x >= 0
            && crossed_tile_y < crossed.map.len() as Coord
            && crossing_x < crossed.map[0].len() as Coord
            && crossed.map[crossed_tile_y as usize][crossing_x as usize] != 0
    });
    crosses_room && !crossed.toilet_crossing_x.contains(&crossing_x)
}

impl CommonData {
    pub fn toilet_room_idx(&self) -> Option<RoomIdx> {
        self.toilet_room_idx
    }

    pub fn new(rooms: Vec<Room>) -> Result<Self> {
        if rooms.len() > RoomIdx::MAX as usize {
            bail!(
                "room set has {} rooms, exceeding the maximum {} supported by RoomIdx plus one dummy action",
                rooms.len(),
                RoomIdx::MAX
            );
        }

        let mut room_data = vec![];
        let mut geometry_data = vec![];
        let mut geometry_rooms = vec![];
        let mut geometry_connection_variants = vec![];
        let mut connection_variant_rooms = vec![];
        let mut door_group_count = 0;
        let mut room_part = vec![];
        let mut room_connection = vec![];
        let mut geometry_by_key = HashMap::new();
        let mut connection_variant_by_key = HashMap::new();
        let mut toilet_room_idx = None;
        let mut geometry_dir_door: [Vec<GeometryDirDoorData>; NUM_DIRS] =
            std::array::from_fn(|_| vec![]);
        let mut room_dir_door: [Vec<RoomDirDoorData>; NUM_DIRS] = std::array::from_fn(|_| vec![]);

        for (room_idx, room) in rooms.iter().enumerate() {
            if room.special_type == Some(SpecialType::Toilet) {
                if let Some(first_toilet_room_idx) = toilet_room_idx {
                    bail!(
                        "rooms {first_toilet_room_idx} and {room_idx} both have special_type toilet"
                    );
                }
                toilet_room_idx = Some(room_idx as RoomIdx);
            }
            if room.doors.len() > PartIdx::MAX as usize {
                bail!(
                    "room {room_idx} has {} door groups, exceeding the maximum {}",
                    room.doors.len(),
                    PartIdx::MAX
                );
            }
            if door_group_count + room.doors.len() > RoomPartIdx::MAX as usize {
                bail!(
                    "rooms have {} total door groups, exceeding the maximum {}",
                    door_group_count + room.doors.len(),
                    RoomPartIdx::MAX
                );
            }
            for &(from_part, to_part) in room.connections.iter().chain(&room.missing_connections) {
                if from_part as usize >= room.doors.len() || to_part as usize >= room.doors.len() {
                    bail!(
                        "room {room_idx} has connection ({from_part}, {to_part}) outside its {} door groups",
                        room.doors.len()
                    );
                }
            }
            for (connection_idx, connection) in room.connections.iter().enumerate() {
                if room.connections[..connection_idx].contains(connection) {
                    bail!("room {room_idx} has duplicate connection {connection:?}");
                }
            }
            for (connection_idx, connection) in room.missing_connections.iter().enumerate() {
                if room.missing_connections[..connection_idx].contains(connection) {
                    bail!("room {room_idx} has duplicate missing connection {connection:?}");
                }
            }
            for (x_idx, &x) in room.toilet_crossing_x.iter().enumerate() {
                if x < 0 || x as usize >= room.map[0].len() {
                    bail!(
                        "room {room_idx} has toilet_crossing_x value {x} outside its width {}",
                        room.map[0].len()
                    );
                }
                if room.toilet_crossing_x[..x_idx].contains(&x) {
                    bail!("room {room_idx} has duplicate toilet_crossing_x value {x}");
                }
            }
            validate_missing_connections(room_idx, room)?;

            let mut door_data = vec![];
            for (part_idx, door_group) in room.doors.iter().enumerate() {
                let room_part_idx = (door_group_count + part_idx) as RoomPartIdx;
                for door in door_group {
                    let dir_idx = door.direction as usize;
                    if room_dir_door[dir_idx].len() >= DirDoorIdx::MAX as usize {
                        bail!(
                            "room set has too many {:?} doors, exceeding the maximum {} usable door indices before the sentinel",
                            door.direction,
                            DirDoorIdx::MAX
                        );
                    }
                    let dir_door_idx = room_dir_door[dir_idx].len() as DirDoorIdx;
                    room_dir_door[dir_idx].push(RoomDirDoorData {
                        room_idx: room_idx as RoomIdx,
                        room_part_idx,
                        x: door.x,
                        y: door.y,
                        direction: door.direction,
                        kind: door.kind,
                    });
                    door_data.push(RoomDoorData {
                        x: door.x,
                        y: door.y,
                        direction: door.direction,
                        dir_door_idx,
                        part_idx: part_idx as PartIdx,
                    });
                }
            }

            let geometry_key = GeometryKey::from_room(room);
            let geometry_idx = if let Some(&geometry_idx) = geometry_by_key.get(&geometry_key) {
                geometry_idx
            } else {
                if geometry_data.len() > GeometryIdx::MAX as usize {
                    bail!(
                        "room set has too many unique geometries, exceeding the maximum {}",
                        GeometryIdx::MAX as usize + 1
                    );
                }
                let geometry_idx = geometry_data.len() as GeometryIdx;
                let geometry = GeometryData::new(&geometry_key)?;
                for door in geometry.doors.iter() {
                    geometry_dir_door[door.direction as usize].push(GeometryDirDoorData {
                        geometry_idx,
                        x: door.x,
                        y: door.y,
                        kind: door.kind,
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
                if connection_variant_rooms.len() > ConnectionVariantIdx::MAX as usize {
                    bail!(
                        "room set has too many connection variants, exceeding the maximum {}",
                        ConnectionVariantIdx::MAX as usize + 1
                    );
                }
                let connection_variant_idx = connection_variant_rooms.len() as ConnectionVariantIdx;
                connection_variant_rooms.push(vec![]);
                geometry_connection_variants[geometry_idx as usize].push(connection_variant_idx);
                connection_variant_by_key
                    .insert((geometry_idx, connections_key), connection_variant_idx);
                connection_variant_idx
            };

            geometry_rooms[geometry_idx as usize].push(room_idx as RoomIdx);
            connection_variant_rooms[connection_variant_idx as usize].push(room_idx as RoomIdx);
            for part_idx in 0..room.doors.len() {
                room_part.push((room_idx as RoomIdx, part_idx as PartIdx));
            }
            for &(from_part, to_part) in &room.missing_connections {
                room_connection.push(RoomConnectionData {
                    room_idx: room_idx as RoomIdx,
                    from_part,
                    to_part,
                });
            }
            room_data.push(RoomData {
                geometry_idx,
                connection_variant_idx,
                doors: door_data,
                door_group_offset: door_group_count,
                door_group_count: room.doors.len(),
                connections: room.connections.clone(),
                part_distances: room_part_distances(room),
            });
            door_group_count += room.doors.len();
        }

        let mut door_output = vec![];
        let mut door_output_variant_by_key = HashMap::new();
        for door in room_dir_door.iter().flatten() {
            let connection_variant_idx = room_data[door.room_idx as usize].connection_variant_idx;
            let next_variant_idx = door_output_variant_by_key.len();
            let variant_outcome_idx = *door_output_variant_by_key
                .entry((
                    connection_variant_idx,
                    door.direction,
                    door.x,
                    door.y,
                    door.kind,
                ))
                .or_insert(next_variant_idx);
            door_output.push(OutputData {
                room_idx: door.room_idx,
                variant_outcome_idx,
            });
        }
        if door_output_variant_by_key.len() > DoorVariantIdx::MAX as usize {
            bail!(
                "room set has too many door output variants, exceeding the maximum {}",
                DoorVariantIdx::MAX
            );
        }
        let door_variant_idx_by_key = door_output_variant_by_key
            .iter()
            .map(|(&key, &idx)| (key, idx as DoorVariantIdx))
            .collect();

        let mut connection_output = vec![];
        let mut connection_output_variant_by_key = HashMap::new();
        for connection in &room_connection {
            let connection_variant_idx =
                room_data[connection.room_idx as usize].connection_variant_idx;
            let next_variant_idx = connection_output_variant_by_key.len();
            let variant_outcome_idx = *connection_output_variant_by_key
                .entry((
                    connection_variant_idx,
                    connection.from_part,
                    connection.to_part,
                ))
                .or_insert(next_variant_idx);
            connection_output.push(OutputData {
                room_idx: connection.room_idx,
                variant_outcome_idx,
            });
        }

        let mut common = Self {
            room: room_data,
            toilet_room_idx,
            geometry: geometry_data,
            geometry_rooms,
            geometry_connection_variants,
            connection_variant_rooms,
            intersection_idx: vec![],
            intersection_bitvec: BitVec::new(),
            geometry_dir_door,
            room_dir_door,
            room_part,
            room_connection,
            door_output,
            connection_output,
            num_door_output_variants: door_output_variant_by_key.len(),
            num_connection_output_variants: connection_output_variant_by_key.len(),
            door_variant_idx_by_key,
        };
        common.build_intersection_set();
        println!(
            "Finished building intersection set with {} bits across {} geometries",
            common.intersection_bitvec.len(),
            common.geometry.len()
        );
        Ok(common)
    }

    pub fn door_variant_idx(
        &self,
        connection_variant_idx: ConnectionVariantIdx,
        direction: Direction,
        x: Coord,
        y: Coord,
        kind: DoorKind,
    ) -> DoorVariantIdx {
        *self
            .door_variant_idx_by_key
            .get(&(connection_variant_idx, direction, x, y, kind))
            .expect("candidate door must have a door output variant")
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
    pub fn has_geometry_intersection(
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
        if geometry1.is_toilet && has_disallowed_toilet_crossing(x1, y1, geometry2, x2, y2) {
            return true;
        }
        if geometry2.is_toilet && has_disallowed_toilet_crossing(x2, y2, geometry1, x1, y1) {
            return true;
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_metadata_shares_only_matching_connection_variants() {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"
            [
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[
                        {"direction": "left", "x": 0, "y": 0, "kind": 0},
                        {"direction": "right", "x": 0, "y": 0, "kind": 0}
                    ]],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[
                        {"direction": "left", "x": 0, "y": 0, "kind": 0},
                        {"direction": "right", "x": 0, "y": 0, "kind": 0}
                    ]],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "doors": [[
                        {"direction": "left", "x": 0, "y": 0, "kind": 0},
                        {"direction": "right", "x": 0, "y": 0, "kind": 0}
                    ]],
                    "connections": [[0, 0]],
                    "missing_connections": []
                }
            ]
            "#,
        )
        .unwrap();

        let common = CommonData::new(rooms).unwrap();
        let door_output: Vec<_> = common
            .door_output
            .iter()
            .map(|output| (output.room_idx, output.variant_outcome_idx))
            .collect();
        assert_eq!(
            door_output,
            vec![(0, 0), (1, 0), (2, 1), (0, 2), (1, 2), (2, 3)]
        );
        assert_eq!(common.num_door_output_variants, 4);

        let connection_output: Vec<_> = common
            .connection_output
            .iter()
            .map(|output| (output.room_idx, output.variant_outcome_idx))
            .collect();
        assert!(connection_output.is_empty());
        assert_eq!(common.num_connection_output_variants, 0);

        let room_connection_variant_idx: Vec<_> = common
            .room
            .iter()
            .map(|room| room.connection_variant_idx)
            .collect();
        assert_eq!(room_connection_variant_idx, vec![0, 0, 1]);
        assert_eq!(common.connection_variant_rooms.len(), 2);
    }

    #[test]
    fn missing_connections_must_be_a_minimum_strong_completion() {
        let parse = |missing_connections| {
            serde_json::from_str::<Vec<Room>>(&format!(
                r#"[{{"map": [[1]], "toilet_crossing_x": [], "doors": [[], []], "connections": [], "missing_connections": {missing_connections}}}]"#
            ))
            .unwrap()
        };
        assert!(CommonData::new(parse("[[0, 1], [1, 0]]")).is_ok());
        assert!(CommonData::new(parse("[]")).is_err());
        assert!(CommonData::new(parse("[[0, 1], [1, 0], [0, 1]]")).is_err());
        assert!(CommonData::new(parse("[[0, 1], [0, 1]]")).is_err());
    }

    #[test]
    fn room_part_distances_use_directed_zero_cost_connections() {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"
            [{
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [[], [], []],
                "connections": [[0, 1], [1, 2]],
                "missing_connections": [[2, 0]]
            }]
            "#,
        )
        .unwrap();
        let common = CommonData::new(rooms).unwrap();

        assert_eq!(
            common.room[0].part_distances,
            vec![
                0,
                0,
                0,
                GraphDistance::MAX,
                0,
                0,
                GraphDistance::MAX,
                GraphDistance::MAX,
                0,
            ]
        );
    }

    #[test]
    fn room_part_distances_exclude_missing_connections() {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"
            [{
                "map": [[1]],
                "toilet_crossing_x": [],
                "doors": [[], []],
                "connections": [],
                "missing_connections": [[0, 1], [1, 0]]
            }]
            "#,
        )
        .unwrap();
        let common = CommonData::new(rooms).unwrap();

        assert_eq!(
            common.room[0].part_distances,
            vec![0, GraphDistance::MAX, GraphDistance::MAX, 0]
        );
    }

    #[test]
    fn toilet_crossing_x_restricts_toilet_intersections() {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"
            [
                {
                    "map": [[1, 1, 1]],
                    "toilet_crossing_x": [1],
                    "doors": [],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1], [1], [0], [0], [0], [0], [0], [0], [1], [1]],
                    "toilet_crossing_x": [],
                    "special_type": "toilet",
                    "doors": [[
                        {"direction": "down", "x": 0, "y": 9, "kind": 0},
                        {"direction": "up", "x": 0, "y": 0, "kind": 0}
                    ]],
                    "connections": [],
                    "missing_connections": []
                }
            ]
            "#,
        )
        .unwrap();
        let common = CommonData::new(rooms).unwrap();
        let room_geometry_idx = common.room[0].geometry_idx;
        let toilet_geometry_idx = common.room[1].geometry_idx;
        assert_eq!(common.toilet_room_idx(), Some(1));

        assert!(!common.has_geometry_intersection(
            room_geometry_idx,
            0,
            0,
            toilet_geometry_idx,
            1,
            -2
        ));
        assert!(common.has_geometry_intersection(
            room_geometry_idx,
            0,
            0,
            toilet_geometry_idx,
            0,
            -2
        ));
    }

    #[test]
    fn common_data_rejects_multiple_toilet_rooms() {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"
            [
                {
                    "map": [[1]],
                    "toilet_crossing_x": [],
                    "special_type": "toilet",
                    "doors": [],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1]],
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

        assert!(CommonData::new(rooms).is_err());
    }

    #[test]
    fn toilet_cannot_cross_main_hall_rightmost_column() {
        let rooms: Vec<Room> = serde_json::from_str(
            r#"
            [
                {
                    "map": [
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1]
                    ],
                    "toilet_crossing_x": [],
                    "doors": [[
                        {"direction": "left", "x": 0, "y": 2, "kind": 0},
                        {"direction": "right", "x": 7, "y": 2, "kind": 0},
                        {"direction": "up", "x": 4, "y": 0, "kind": 1}
                    ]],
                    "connections": [],
                    "missing_connections": []
                },
                {
                    "map": [[1], [1], [0], [0], [0], [0], [0], [0], [1], [1]],
                    "toilet_crossing_x": [],
                    "special_type": "toilet",
                    "doors": [[
                        {"direction": "down", "x": 0, "y": 9, "kind": 0},
                        {"direction": "up", "x": 0, "y": 0, "kind": 0}
                    ]],
                    "connections": [],
                    "missing_connections": []
                }
            ]
            "#,
        )
        .unwrap();
        let common = CommonData::new(rooms).unwrap();
        let main_hall_geometry_idx = common.room[0].geometry_idx;
        let toilet_geometry_idx = common.room[1].geometry_idx;

        assert!(common.has_geometry_intersection(
            main_hall_geometry_idx,
            0,
            0,
            toilet_geometry_idx,
            7,
            -5
        ));
    }
}
