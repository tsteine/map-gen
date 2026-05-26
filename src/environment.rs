use bitvec::vec::BitVec;
use hashbrown::HashMap;
use rand::SeedableRng;
use rand::prelude::*;

use crate::common::{
    Action, CommonData, ConnectionVariantIdx, Coord, DirDoorIdx, DoorLocation, DoorValidOutcome,
    GeometryIdx, NUM_DIRS, PartIdx, RoomIdx, get_behind_door_position,
};

const NO_COMPONENT: usize = usize::MAX;
const NO_ROOM_PART: usize = usize::MAX;

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

#[derive(Debug, Default)]
struct SccDag {
    component_count: usize,
    reachability: Vec<bool>,
}

#[derive(Debug, PartialEq, Eq)]
struct ComponentMerge {
    merged_components: Vec<usize>,
    component_remap: Vec<usize>,
}

impl SccDag {
    fn clear(&mut self) {
        self.component_count = 0;
        self.reachability.clear();
    }

    fn add_component(&mut self) -> usize {
        let old_count = self.component_count;
        let new_count = old_count + 1;
        let mut reachability = vec![false; new_count * new_count];
        for row in 0..old_count {
            for col in 0..old_count {
                reachability[row * new_count + col] = self.reachability[row * old_count + col];
            }
        }
        reachability[old_count * new_count + old_count] = true;
        self.component_count = new_count;
        self.reachability = reachability;
        old_count
    }

    fn add_edge(&mut self, from_component: usize, to_component: usize) -> Option<ComponentMerge> {
        if from_component == to_component {
            return None;
        }
        if !self.can_reach(to_component, from_component) {
            self.connect_components(from_component, to_component);
            return None;
        }

        let mut merge_components = vec![];
        for component in 0..self.component_count {
            if self.can_reach(to_component, component) && self.can_reach(component, from_component)
            {
                merge_components.push(component);
            }
        }
        Some(self.merge_components(&merge_components))
    }

    fn merge_components(&mut self, merge_components: &[usize]) -> ComponentMerge {
        debug_assert!(!merge_components.is_empty());
        let target = merge_components.iter().copied().min().unwrap();
        let mut in_merge = vec![false; self.component_count];
        for &component in merge_components {
            in_merge[component] = true;
        }

        let mut predecessors = vec![];
        let mut successors = vec![];
        for (component, is_merged) in in_merge.iter().copied().enumerate() {
            if !is_merged
                && merge_components
                    .iter()
                    .any(|&merged| self.can_reach(component, merged))
            {
                predecessors.push(component);
            }
            if !is_merged
                && merge_components
                    .iter()
                    .any(|&merged| self.can_reach(merged, component))
            {
                successors.push(component);
            }
        }

        let old_count = self.component_count;
        let mut component_remap = vec![NO_COMPONENT; old_count];
        let mut new_count = 0;
        for (component, is_merged) in in_merge.iter().copied().enumerate() {
            if component == target || !is_merged {
                component_remap[component] = new_count;
                new_count += 1;
            } else {
                component_remap[component] = target;
            }
        }
        // let target = component_remap[target];  // The target component is already mapped to itself, since it was chosen to be minimal.

        let old_reachability =
            std::mem::replace(&mut self.reachability, vec![false; new_count * new_count]);
        self.component_count = new_count;

        for from in 0..old_count {
            if in_merge[from] {
                continue;
            }
            for to in 0..old_count {
                if in_merge[to] {
                    continue;
                }
                if old_reachability[from * old_count + to] {
                    self.set_reachable(component_remap[from], component_remap[to]);
                }
            }
        }
        self.set_reachable(target, target);
        for predecessor in predecessors {
            let predecessor = component_remap[predecessor];
            self.set_reachable(predecessor, target);
            for &successor in &successors {
                self.set_reachable(predecessor, component_remap[successor]);
            }
        }
        for successor in successors {
            let successor = component_remap[successor];
            self.set_reachable(target, successor);
        }

        ComponentMerge {
            merged_components: merge_components.to_vec(),
            component_remap,
        }
    }

    fn connect_components(&mut self, from_component: usize, to_component: usize) {
        let mut predecessors = vec![];
        let mut successors = vec![];
        for component in 0..self.component_count {
            if self.can_reach(component, from_component) {
                predecessors.push(component);
            }
            if self.can_reach(to_component, component) {
                successors.push(component);
            }
        }
        for predecessor in predecessors {
            for &successor in &successors {
                self.set_reachable(predecessor, successor);
            }
        }
    }

    fn can_reach(&self, from_component: usize, to_component: usize) -> bool {
        self.reachability[from_component * self.component_count + to_component]
    }

    fn set_reachable(&mut self, from_component: usize, to_component: usize) {
        self.reachability[from_component * self.component_count + to_component] = true;
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
    room_x: Vec<Coord>, // x position of each room (only valid for used rooms)
    room_y: Vec<Coord>, // y position of each room (only valid for used rooms)
    geometry_unused_count: Vec<usize>, // number of unused room representatives for each geometry
    connection_variant_unused_count: Vec<usize>, // number of unused room representatives for each connection variant
    room_part_component: Vec<usize>,             // maps placed room door groups to SCC components
    door_group_part_by_dir_door: [Vec<usize>; NUM_DIRS],
    scc_dag: SccDag,
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
            room_part_component: vec![
                NO_COMPONENT;
                common
                    .room
                    .iter()
                    .map(|room| room.door_group_offset + room.door_group_count)
                    .max()
                    .unwrap_or(0)
            ],
            door_group_part_by_dir_door: std::array::from_fn(|i| {
                vec![NO_ROOM_PART; common.room_dir_door[i].len()]
            }),
            scc_dag: SccDag::default(),
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
        self.room_part_component.fill(NO_COMPONENT);
        self.door_group_part_by_dir_door
            .iter_mut()
            .for_each(|parts| parts.fill(NO_ROOM_PART));
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
        self.add_room_components_and_edges(action.room_idx, common);

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
                let p1 = self.door_group_part_by_dir_door[door.direction as usize][i1 as usize];
                let p2 = self.door_group_part_by_dir_door[door.direction.opposite() as usize]
                    [i2 as usize];
                debug_assert_ne!(p1, NO_ROOM_PART);
                debug_assert_ne!(p2, NO_ROOM_PART);
                self.add_component_edge(self.room_part_component[p1], self.room_part_component[p2]);
                self.add_component_edge(self.room_part_component[p2], self.room_part_component[p1]);
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

    fn add_room_components_and_edges(&mut self, room_idx: RoomIdx, common: &CommonData) {
        let room = &common.room[room_idx as usize];
        for part_idx in 0..room.door_group_count {
            let room_part_idx = room.door_group_offset + part_idx;
            self.room_part_component[room_part_idx] = self.scc_dag.add_component();
        }
        for door in &room.doors {
            let room_part_idx = Self::room_part_idx(common, room_idx, door.part_idx);
            self.door_group_part_by_dir_door[door.direction as usize][door.dir_door_idx as usize] =
                room_part_idx;
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

    fn room_part_idx(common: &CommonData, room_idx: RoomIdx, part_idx: PartIdx) -> usize {
        common.room[room_idx as usize].door_group_offset + part_idx as usize
    }

    fn room_part_component(
        &self,
        common: &CommonData,
        room_idx: RoomIdx,
        part_idx: PartIdx,
    ) -> usize {
        let component = self.room_part_component[Self::room_part_idx(common, room_idx, part_idx)];
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Room;

    fn sorted_merge(mut merge_components: Vec<usize>) -> Vec<usize> {
        merge_components.sort_unstable();
        merge_components
    }

    #[test]
    fn scc_dag_keeps_acyclic_edges_between_separate_components() {
        let mut scc = SccDag::default();
        let a = scc.add_component();
        let b = scc.add_component();
        let c = scc.add_component();

        assert_eq!(scc.add_edge(a, b), None);
        assert_eq!(scc.add_edge(b, c), None);

        assert!(scc.can_reach(a, b));
        assert!(scc.can_reach(b, c));
        assert!(scc.can_reach(a, c));
        assert!(!scc.can_reach(c, a));
    }

    #[test]
    fn scc_dag_merges_two_components_for_bidirectional_edges() {
        let mut scc = SccDag::default();
        let a = scc.add_component();
        let b = scc.add_component();

        assert_eq!(scc.add_edge(a, b), None);
        let merge = scc.add_edge(b, a).unwrap();

        assert_eq!(sorted_merge(merge.merged_components), vec![a, b]);
        assert_eq!(merge.component_remap, vec![0, 0]);
        assert_eq!(scc.component_count, 1);
        assert!(scc.can_reach(0, 0));
    }

    #[test]
    fn scc_dag_merges_cycle_path_and_preserves_external_edges() {
        let mut scc = SccDag::default();
        let a = scc.add_component();
        let b = scc.add_component();
        let c = scc.add_component();
        let d = scc.add_component();

        assert_eq!(scc.add_edge(a, b), None);
        assert_eq!(scc.add_edge(b, c), None);
        assert_eq!(scc.add_edge(c, d), None);
        let merge = scc.add_edge(c, a).unwrap();

        assert_eq!(sorted_merge(merge.merged_components), vec![a, b, c]);
        let merged = merge.component_remap[a];
        let d = merge.component_remap[d];
        assert_eq!(scc.component_count, 2);
        assert!(scc.can_reach(merged, d));
        assert!(!scc.can_reach(d, merged));
    }

    #[test]
    fn scc_dag_duplicate_edges_are_idempotent() {
        let mut scc = SccDag::default();
        let a = scc.add_component();
        let b = scc.add_component();

        assert_eq!(scc.add_edge(a, b), None);
        let reachability = scc.reachability.clone();
        assert_eq!(scc.add_edge(a, b), None);

        assert_eq!(scc.reachability, reachability);
    }

    #[test]
    fn scc_dag_preserves_external_reachability_on_merge() {
        let mut scc = SccDag::default();
        let source = scc.add_component();
        let a = scc.add_component();
        let b = scc.add_component();
        let sink = scc.add_component();

        assert_eq!(scc.add_edge(source, a), None);
        assert_eq!(scc.add_edge(a, b), None);
        assert_eq!(scc.add_edge(b, sink), None);
        let merge = scc.add_edge(b, a).unwrap();

        assert_eq!(sorted_merge(merge.merged_components), vec![a, b]);
        let source = merge.component_remap[source];
        let merged = merge.component_remap[a];
        let sink = merge.component_remap[sink];
        assert_eq!(scc.component_count, 3);
        assert!(scc.can_reach(source, merged));
        assert!(scc.can_reach(merged, sink));
        assert!(scc.can_reach(source, sink));
        assert!(!scc.can_reach(sink, merged));
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
        assert!(env.scc_dag.can_reach(room1_part0, room0_part1));

        env.clear(&common);
        assert_eq!(env.scc_dag.component_count, 0);
        assert!(
            env.room_part_component
                .iter()
                .all(|&component| component == NO_COMPONENT)
        );
    }
}
