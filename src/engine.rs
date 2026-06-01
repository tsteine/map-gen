/// The `engine` module exposes the map generation environment to Python through the Engine and
/// EnvironmentGroup classes. It handles the creation and management of worker threads that run
/// environment simulations in parallel.
use crate::common::{Action, CommonData, Coord, DoorValidOutcome, Room, RoomIdx};
use crate::environment::{Environment, StateFeatureProfile, StateFeatures};
use crossbeam_channel as channel;
use numpy::{
    Element, IntoPyArray, PyArray2, PyArray3, PyArray4, PyArray5, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArray2,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::cmp::{max, min};
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Instant;

fn pyarray2_from_flat_vec<'py, T: Element>(
    py: Python<'py>,
    data: Vec<T>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<T>>> {
    data.into_pyarray(py).reshape([rows, cols])
}

fn pyarray3_from_flat_vec<'py, T: Element>(
    py: Python<'py>,
    data: Vec<T>,
    dim0: usize,
    dim1: usize,
    dim2: usize,
) -> PyResult<Bound<'py, PyArray3<T>>> {
    data.into_pyarray(py).reshape([dim0, dim1, dim2])
}

fn pyarray4_from_flat_vec<'py, T: Element>(
    py: Python<'py>,
    data: Vec<T>,
    dim0: usize,
    dim1: usize,
    dim2: usize,
    dim3: usize,
) -> PyResult<Bound<'py, PyArray4<T>>> {
    data.into_pyarray(py).reshape([dim0, dim1, dim2, dim3])
}

fn pyarray5_from_flat_vec<'py, T: Element>(
    py: Python<'py>,
    data: Vec<T>,
    dim0: usize,
    dim1: usize,
    dim2: usize,
    dim3: usize,
    dim4: usize,
) -> PyResult<Bound<'py, PyArray5<T>>> {
    data.into_pyarray(py)
        .reshape([dim0, dim1, dim2, dim3, dim4])
}

// We use shards to share slices of memory between the main thread and worker threads. This allows us
// to avoid copying data back and forth through channels. The raw pointers are necessary because the
// worker threads are long-lived while the shared memory is short-lived, making it difficult to
// convince the Rust borrow checker that the references are valid. We ensure safety by using the raw
// pointers only within the scope of a single command; in every case, the main thread waits for a
// "done" response from the worker thread before relinquishing ownership of the shared memory.
//
// An alternative approach would be to use scoped threads to allow safely borrowing the slices
// directly, but that would less performant because it would break the affinity of worker threads
// to their environment shards.
#[derive(Clone, Copy)]
struct InputShard<T> {
    ptr: *const T,
    len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: Sync> Send for InputShard<T> {}

impl<T> InputShard<T> {
    fn from_slice(slice: &[T]) -> Self {
        Self {
            ptr: slice.as_ptr(),
            len: slice.len(),
            _marker: PhantomData,
        }
    }

    unsafe fn into_slice<'a>(self) -> &'a [T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

#[derive(Clone, Copy)]
struct OutputShard<T> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for OutputShard<T> {}

impl<T> OutputShard<T> {
    fn from_slice(slice: &mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            _marker: PhantomData,
        }
    }

    unsafe fn into_mut_slice<'a>(self) -> &'a mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

enum WorkerCommand {
    Clear,
    Finish,
    Step {
        room_idx: InputShard<RoomIdx>,
        room_x: InputShard<Coord>,
        room_y: InputShard<Coord>,
    },
    Replay {
        action_count: usize,
        room_idx: InputShard<RoomIdx>,
        room_x: InputShard<Coord>,
        room_y: InputShard<Coord>,
    },
    GetCandidates {
        max_candidates: usize,
        room_idx: OutputShard<RoomIdx>,
        room_x: OutputShard<Coord>,
        room_y: OutputShard<Coord>,
    },
    GetCandidatesWithOutcomes {
        max_candidates: usize,
        room_idx: OutputShard<RoomIdx>,
        room_x: OutputShard<Coord>,
        room_y: OutputShard<Coord>,
        door_outcome_count: usize,
        connection_outcome_count: usize,
        door_valid: OutputShard<i8>,
        connections_valid: OutputShard<i8>,
    },
    GetActions {
        action_count: usize,
        room_idx: OutputShard<RoomIdx>,
        room_x: OutputShard<Coord>,
        room_y: OutputShard<Coord>,
    },
    GetOutcomes {
        door_outcome_count: usize,
        connection_outcome_count: usize,
        door_valid: OutputShard<i8>,
        connections_valid: OutputShard<i8>,
    },
    GetStateFeatures {
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
        environment_start: usize,
        environment_count: usize,
    },
    GetStateFeaturesAfterCandidates {
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
        environment_start: usize,
        environment_count: usize,
        candidate_count: usize,
        room_idx: InputShard<RoomIdx>,
        room_x: InputShard<Coord>,
        room_y: InputShard<Coord>,
    },
    PackStateFeatures {
        outputs: StateFeatureOutputShards,
    },
    Shutdown,
}

// State feature preparation reports only metadata needed to allocate output buffers. Bulk data is
// written through shared memory, and other commands return "done" when they finish.
enum WorkerResponse {
    Done,
    StateFeatureProfile(StateFeatureProfile, usize),
}

struct WorkerHandle {
    start: usize,
    len: usize,
    command_tx: channel::Sender<WorkerCommand>,
    response_rx: channel::Receiver<WorkerResponse>,
    join_handle: Option<JoinHandle<()>>,
}

impl WorkerHandle {
    fn end(&self) -> usize {
        self.start + self.len
    }

    fn send(&self, command: WorkerCommand) -> PyResult<()> {
        self.command_tx
            .send(command)
            .map_err(|_| PyRuntimeError::new_err("engine worker thread stopped unexpectedly"))
    }

    fn recv(&self) -> PyResult<WorkerResponse> {
        self.response_rx
            .recv()
            .map_err(|_| PyRuntimeError::new_err("engine worker thread stopped unexpectedly"))
    }

    fn recv_done(&self) -> PyResult<()> {
        match self.recv()? {
            WorkerResponse::Done => Ok(()),
            WorkerResponse::StateFeatureProfile(_, _) => Err(PyRuntimeError::new_err(
                "engine worker thread returned unexpected state feature profile",
            )),
        }
    }

    fn shutdown(&mut self) {
        let _ = self.command_tx.send(WorkerCommand::Shutdown);
        if let Some(join_handle) = self.join_handle.take() {
            let _ = join_handle.join();
        }
    }
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn worker_loop(
    mut environments: Vec<Environment>,
    common_data: Arc<CommonData>,
    command_rx: channel::Receiver<WorkerCommand>,
    response_tx: channel::Sender<WorkerResponse>,
) {
    let mut pending_state_features = Vec::new();
    while let Ok(command) = command_rx.recv() {
        let response = match command {
            WorkerCommand::Clear => {
                for env in &mut environments {
                    env.clear(&common_data);
                }
                WorkerResponse::Done
            }
            WorkerCommand::Finish => {
                for env in &mut environments {
                    env.finish();
                }
                WorkerResponse::Done
            }
            WorkerCommand::Step {
                room_idx,
                room_x,
                room_y,
            } => {
                // SAFETY: The main thread guarantees that for the duration of this command,
                // the input slices remain valid and that no other thread mutates them.
                let room_idx = unsafe { room_idx.into_slice() };
                let room_x = unsafe { room_x.into_slice() };
                let room_y = unsafe { room_y.into_slice() };
                debug_assert_eq!(room_idx.len(), environments.len());
                debug_assert_eq!(room_x.len(), environments.len());
                debug_assert_eq!(room_y.len(), environments.len());

                for (env_idx, env) in environments.iter_mut().enumerate() {
                    env.step(
                        Action {
                            room_idx: room_idx[env_idx],
                            x: room_x[env_idx],
                            y: room_y[env_idx],
                        },
                        &common_data,
                    );
                }
                WorkerResponse::Done
            }
            WorkerCommand::Replay {
                action_count,
                room_idx,
                room_x,
                room_y,
            } => {
                // SAFETY: The main thread guarantees that for the duration of this command,
                // the input slices remain valid and that no other thread mutates them.
                let room_idx = unsafe { room_idx.into_slice() };
                let room_x = unsafe { room_x.into_slice() };
                let room_y = unsafe { room_y.into_slice() };
                debug_assert_eq!(room_idx.len(), environments.len() * action_count);
                debug_assert_eq!(room_x.len(), environments.len() * action_count);
                debug_assert_eq!(room_y.len(), environments.len() * action_count);

                for (env_idx, env) in environments.iter_mut().enumerate() {
                    let row_start = env_idx * action_count;
                    let actions = (0..action_count)
                        .map(|action_idx| {
                            let idx = row_start + action_idx;
                            Action {
                                room_idx: room_idx[idx],
                                x: room_x[idx],
                                y: room_y[idx],
                            }
                        })
                        .collect::<Vec<_>>();
                    env.replay(&actions, &common_data);
                }
                WorkerResponse::Done
            }
            WorkerCommand::GetCandidates {
                max_candidates,
                room_idx,
                room_x,
                room_y,
            } => {
                // SAFETY: The main thread guarantees that for the duration of this command,
                // the output slices remain valid and that no other thread accesses them.
                let room_idx = unsafe { room_idx.into_mut_slice() };
                let room_x = unsafe { room_x.into_mut_slice() };
                let room_y = unsafe { room_y.into_mut_slice() };
                debug_assert_eq!(room_idx.len(), environments.len() * max_candidates);
                debug_assert_eq!(room_x.len(), environments.len() * max_candidates);
                debug_assert_eq!(room_y.len(), environments.len() * max_candidates);

                for (env_idx, env) in environments.iter_mut().enumerate() {
                    let candidates = env.get_candidates(&common_data, max_candidates);
                    let row_start = env_idx * max_candidates;
                    for (candidate_idx, candidate) in candidates.iter().enumerate() {
                        let idx = row_start + candidate_idx;
                        room_idx[idx] = candidate.room_idx;
                        room_x[idx] = candidate.x;
                        room_y[idx] = candidate.y;
                    }
                }
                WorkerResponse::Done
            }
            WorkerCommand::GetCandidatesWithOutcomes {
                max_candidates,
                room_idx,
                room_x,
                room_y,
                door_outcome_count,
                connection_outcome_count,
                door_valid,
                connections_valid,
            } => {
                // SAFETY: The main thread guarantees that for the duration of this command,
                // the output slices remain valid and that no other thread accesses them.
                let room_idx = unsafe { room_idx.into_mut_slice() };
                let room_x = unsafe { room_x.into_mut_slice() };
                let room_y = unsafe { room_y.into_mut_slice() };
                let door_valid = unsafe { door_valid.into_mut_slice() };
                let connections_valid = unsafe { connections_valid.into_mut_slice() };
                debug_assert_eq!(room_idx.len(), environments.len() * max_candidates);
                debug_assert_eq!(room_x.len(), environments.len() * max_candidates);
                debug_assert_eq!(room_y.len(), environments.len() * max_candidates);
                debug_assert_eq!(
                    door_valid.len(),
                    environments.len() * max_candidates * door_outcome_count
                );
                debug_assert_eq!(
                    connections_valid.len(),
                    environments.len() * max_candidates * connection_outcome_count
                );

                for (env_idx, env) in environments.iter_mut().enumerate() {
                    let (candidates, outcomes) =
                        env.get_candidates_with_outcomes(&common_data, max_candidates);
                    let row_start = env_idx * max_candidates;
                    let dummy_candidate = Action {
                        room_idx: common_data.room.len() as RoomIdx,
                        x: 0,
                        y: 0,
                    };
                    let dummy_outcome = if candidates.len() < max_candidates {
                        Some(env.outcomes_after_candidate(&common_data, dummy_candidate))
                    } else {
                        None
                    };
                    for candidate_idx in 0..max_candidates {
                        let idx = row_start + candidate_idx;
                        if let Some(candidate) = candidates.get(candidate_idx) {
                            room_idx[idx] = candidate.room_idx;
                            room_x[idx] = candidate.x;
                            room_y[idx] = candidate.y;
                        }

                        let outcome = outcomes
                            .get(candidate_idx)
                            .or(dummy_outcome.as_ref())
                            .expect("dummy outcome must exist for padded candidates");
                        let door_start = idx * door_outcome_count;
                        let door_end = door_start + door_outcome_count;
                        let connection_start = idx * connection_outcome_count;
                        let connection_end = connection_start + connection_outcome_count;
                        for (dst, &outcome) in door_valid[door_start..door_end]
                            .iter_mut()
                            .zip(&outcome.door_valid)
                        {
                            *dst = outcome as i8;
                        }
                        for (dst, &outcome) in connections_valid[connection_start..connection_end]
                            .iter_mut()
                            .zip(&outcome.connections_valid)
                        {
                            *dst = outcome as i8;
                        }
                    }
                }
                WorkerResponse::Done
            }
            WorkerCommand::GetActions {
                action_count,
                room_idx,
                room_x,
                room_y,
            } => {
                // SAFETY: The main thread guarantees that for the duration of this command,
                // the output slices remain valid and that no other thread accesses them.
                let room_idx = unsafe { room_idx.into_mut_slice() };
                let room_x = unsafe { room_x.into_mut_slice() };
                let room_y = unsafe { room_y.into_mut_slice() };
                debug_assert_eq!(room_idx.len(), environments.len() * action_count);
                debug_assert_eq!(room_x.len(), environments.len() * action_count);
                debug_assert_eq!(room_y.len(), environments.len() * action_count);

                for (env_idx, env) in environments.iter().enumerate() {
                    debug_assert_eq!(env.actions().len(), action_count);
                    let row_start = env_idx * action_count;
                    for (action_idx, action) in env.actions().iter().enumerate() {
                        let idx = row_start + action_idx;
                        room_idx[idx] = action.room_idx;
                        room_x[idx] = action.x;
                        room_y[idx] = action.y;
                    }
                }
                WorkerResponse::Done
            }
            WorkerCommand::GetOutcomes {
                door_outcome_count,
                connection_outcome_count,
                door_valid,
                connections_valid,
            } => {
                // SAFETY: The main thread guarantees that for the duration of this command,
                // the output slices remain valid and that no other thread accesses them.
                let door_valid = unsafe { door_valid.into_mut_slice() };
                let connections_valid = unsafe { connections_valid.into_mut_slice() };
                debug_assert_eq!(door_valid.len(), environments.len() * door_outcome_count);
                debug_assert_eq!(
                    connections_valid.len(),
                    environments.len() * connection_outcome_count
                );

                for (env_idx, env) in environments.iter().enumerate() {
                    let outcomes = env.outcomes(&common_data);
                    debug_assert_eq!(outcomes.door_valid.len(), door_outcome_count);
                    debug_assert_eq!(outcomes.connections_valid.len(), connection_outcome_count);
                    let door_row_start = env_idx * door_outcome_count;
                    for (outcome_idx, outcome) in outcomes.door_valid.iter().enumerate() {
                        door_valid[door_row_start + outcome_idx] = match outcome {
                            DoorValidOutcome::Unknown => -1,
                            DoorValidOutcome::Valid => 0,
                            DoorValidOutcome::Invalid => 1,
                        };
                    }
                    let connection_row_start = env_idx * connection_outcome_count;
                    for (outcome_idx, outcome) in outcomes.connections_valid.iter().enumerate() {
                        connections_valid[connection_row_start + outcome_idx] = match outcome {
                            DoorValidOutcome::Unknown => -1,
                            DoorValidOutcome::Valid => 0,
                            DoorValidOutcome::Invalid => 1,
                        };
                    }
                }
                WorkerResponse::Done
            }
            WorkerCommand::GetStateFeatures {
                frontier_neighbor_count,
                frontier_window_size,
                environment_start,
                environment_count,
            } => {
                pending_state_features = environments
                    .iter()
                    .skip(environment_start)
                    .take(environment_count)
                    .map(|env| {
                        env.state_features(
                            &common_data,
                            frontier_neighbor_count,
                            frontier_window_size,
                        )
                    })
                    .collect();
                let frontier_count = max_state_feature_frontier_count(&pending_state_features);
                WorkerResponse::StateFeatureProfile(StateFeatureProfile::default(), frontier_count)
            }
            WorkerCommand::GetStateFeaturesAfterCandidates {
                frontier_neighbor_count,
                frontier_window_size,
                environment_start,
                environment_count,
                candidate_count,
                room_idx,
                room_x,
                room_y,
            } => {
                let room_idx = unsafe { room_idx.into_slice() };
                let room_x = unsafe { room_x.into_slice() };
                let room_y = unsafe { room_y.into_slice() };
                pending_state_features.clear();
                pending_state_features.reserve(environment_count * candidate_count);
                let mut profile = StateFeatureProfile::default();
                for (env_idx, env) in environments
                    .iter()
                    .skip(environment_start)
                    .take(environment_count)
                    .enumerate()
                {
                    let start = Instant::now();
                    let occupancy_prefix = env.occupancy_prefix();
                    profile.occupancy_prefix_ns += start.elapsed().as_nanos() as u64;
                    for candidate_idx in 0..candidate_count {
                        let idx = env_idx * candidate_count + candidate_idx;
                        let (features, candidate_profile) = env
                            .state_features_after_candidate_with_occupancy(
                                &common_data,
                                Action {
                                    room_idx: room_idx[idx],
                                    x: room_x[idx],
                                    y: room_y[idx],
                                },
                                &occupancy_prefix,
                                frontier_neighbor_count,
                                frontier_window_size,
                            );
                        pending_state_features.push(features);
                        profile.add(&candidate_profile);
                    }
                }
                let frontier_count = max_state_feature_frontier_count(&pending_state_features);
                WorkerResponse::StateFeatureProfile(profile, frontier_count)
            }
            WorkerCommand::PackStateFeatures { outputs } => {
                let mut outputs = unsafe { outputs.into_slices() };
                for (idx, features) in pending_state_features.drain(..).enumerate() {
                    outputs.write_features(idx, &features);
                }
                WorkerResponse::Done
            }
            WorkerCommand::Shutdown => break,
        };

        if response_tx.send(response).is_err() {
            break;
        }
    }
}

fn spawn_worker(
    worker_idx: usize,
    start: usize,
    environments: Vec<Environment>,
    common_data: Arc<CommonData>,
) -> PyResult<WorkerHandle> {
    let len = environments.len();
    let (command_tx, command_rx) = channel::bounded(1);
    let (response_tx, response_rx) = channel::bounded(1);
    let join_handle = thread::Builder::new()
        .name(format!("map-gen-worker-{worker_idx}"))
        .spawn(move || worker_loop(environments, common_data, command_rx, response_tx))
        .map_err(|err| PyRuntimeError::new_err(format!("failed to spawn worker thread: {err}")))?;

    Ok(WorkerHandle {
        start,
        len,
        command_tx,
        response_rx,
        join_handle: Some(join_handle),
    })
}

fn requested_num_threads(num_threads: Option<usize>) -> PyResult<usize> {
    match num_threads {
        Some(0) => Err(PyValueError::new_err("num_threads must be greater than 0")),
        Some(num_threads) => Ok(num_threads),
        None => Ok(thread::available_parallelism()
            .map(|num_threads| num_threads.get())
            .unwrap_or(1)),
    }
}

fn set_first_error(first_error: &mut Option<PyErr>, err: PyErr) {
    if first_error.is_none() {
        *first_error = Some(err);
    }
}

fn wait_for_done_responses(
    workers: &[WorkerHandle],
    sent_workers: Vec<usize>,
    mut first_error: Option<PyErr>,
) -> PyResult<()> {
    for worker_idx in sent_workers {
        if let Err(err) = workers[worker_idx].recv_done() {
            set_first_error(&mut first_error, err);
        }
    }

    if let Some(err) = first_error {
        Err(err)
    } else {
        Ok(())
    }
}

fn collect_state_feature_profiles(
    workers: &[WorkerHandle],
    sent_workers: Vec<usize>,
    mut first_error: Option<PyErr>,
) -> PyResult<(StateFeatureProfile, usize)> {
    let mut profile = StateFeatureProfile::default();
    let mut frontier_count = 0;
    for worker_idx in sent_workers {
        match workers[worker_idx].recv() {
            Ok(WorkerResponse::StateFeatureProfile(worker_profile, worker_frontier_count)) => {
                profile.add(&worker_profile);
                frontier_count = max(frontier_count, worker_frontier_count);
            }
            Ok(WorkerResponse::Done) => set_first_error(
                &mut first_error,
                PyRuntimeError::new_err("engine worker thread returned no state feature profile"),
            ),
            Err(err) => set_first_error(&mut first_error, err),
        }
    }

    if let Some(err) = first_error {
        Err(err)
    } else {
        Ok((profile, frontier_count))
    }
}

fn max_state_feature_frontier_count(features: &[StateFeatures]) -> usize {
    features
        .iter()
        .map(|features| features.frontier.len() / 7)
        .max()
        .unwrap_or(0)
}

#[pyclass(module = "map_gen")]
pub struct Engine {
    common_data: Arc<CommonData>, // pre-computed data that can be shared across environments
}

#[pyclass(module = "map_gen")]
pub struct EnvironmentGroup {
    common_data: Arc<CommonData>,
    workers: Vec<WorkerHandle>, // fixed worker-owned environment shards
    num_environments: usize,
    frontier_neighbor_count: usize,
    frontier_window_size: usize,
    action_count: usize,
    state_feature_worker_ns: AtomicU64,
    state_feature_pack_ns: AtomicU64,
    state_feature_profile_calls: AtomicUsize,
    state_feature_occupancy_prefix_ns: AtomicU64,
    state_feature_clone_ns: AtomicU64,
    state_feature_step_ns: AtomicU64,
    state_feature_assemble_ns: AtomicU64,
    state_feature_assemble_setup_ns: AtomicU64,
    state_feature_assemble_frontier_ns: AtomicU64,
    state_feature_assemble_neighbor_ns: AtomicU64,
    state_feature_assemble_pair_ns: AtomicU64,
    state_feature_assemble_pair_flags_ns: AtomicU64,
    state_feature_assemble_pair_obstruction_ns: AtomicU64,
    state_feature_assemble_pair_obstruction_base_ns: AtomicU64,
    state_feature_assemble_pair_obstruction_candidate_ns: AtomicU64,
    state_feature_assemble_output_ns: AtomicU64,
}

fn output_sizes(common_data: &CommonData) -> (usize, usize) {
    let door_outcome_count = common_data
        .room_dir_door
        .iter()
        .map(|doors| doors.len())
        .sum();
    let connection_outcome_count = common_data.room_connection.len();
    (door_outcome_count, connection_outcome_count)
}

struct StateFeatureBuffers {
    inventory: Vec<u8>,
    room_x: Vec<Coord>,
    room_y: Vec<Coord>,
    room_placed: Vec<u8>,
    frontier: Vec<i16>,
    frontier_occupancy: Vec<u8>,
    frontier_neighbor: Vec<i16>,
    frontier_neighbor_pair: Vec<u8>,
    frontier_obstruction: Vec<u16>,
}

struct StateFeatureOutputShards {
    inventory: OutputShard<u8>,
    room_x: OutputShard<Coord>,
    room_y: OutputShard<Coord>,
    room_placed: OutputShard<u8>,
    frontier: OutputShard<i16>,
    frontier_occupancy: OutputShard<u8>,
    frontier_neighbor: OutputShard<i16>,
    frontier_neighbor_pair: OutputShard<u8>,
    frontier_obstruction: OutputShard<u16>,
    inventory_count: usize,
    room_count: usize,
    frontier_count: usize,
    frontier_neighbor_count: usize,
    frontier_window_size: usize,
}

struct StateFeatureOutputSlices<'a> {
    inventory: &'a mut [u8],
    room_x: &'a mut [Coord],
    room_y: &'a mut [Coord],
    room_placed: &'a mut [u8],
    frontier: &'a mut [i16],
    frontier_occupancy: &'a mut [u8],
    frontier_neighbor: &'a mut [i16],
    frontier_neighbor_pair: &'a mut [u8],
    frontier_obstruction: &'a mut [u16],
    inventory_count: usize,
    room_count: usize,
    frontier_count: usize,
    frontier_neighbor_count: usize,
    frontier_window_size: usize,
}

impl StateFeatureOutputShards {
    unsafe fn into_slices<'a>(self) -> StateFeatureOutputSlices<'a> {
        StateFeatureOutputSlices {
            inventory: unsafe { self.inventory.into_mut_slice() },
            room_x: unsafe { self.room_x.into_mut_slice() },
            room_y: unsafe { self.room_y.into_mut_slice() },
            room_placed: unsafe { self.room_placed.into_mut_slice() },
            frontier: unsafe { self.frontier.into_mut_slice() },
            frontier_occupancy: unsafe { self.frontier_occupancy.into_mut_slice() },
            frontier_neighbor: unsafe { self.frontier_neighbor.into_mut_slice() },
            frontier_neighbor_pair: unsafe { self.frontier_neighbor_pair.into_mut_slice() },
            frontier_obstruction: unsafe { self.frontier_obstruction.into_mut_slice() },
            inventory_count: self.inventory_count,
            room_count: self.room_count,
            frontier_count: self.frontier_count,
            frontier_neighbor_count: self.frontier_neighbor_count,
            frontier_window_size: self.frontier_window_size,
        }
    }
}

impl StateFeatureOutputSlices<'_> {
    fn write_features(&mut self, idx: usize, features: &StateFeatures) {
        fn copy_row<T: Copy>(dst: &mut [T], row: &[T], idx: usize, stride: usize) {
            dst[idx * stride..idx * stride + row.len()].copy_from_slice(row);
        }

        fn pack_occupancy_windows(
            dst: &mut [u8],
            src: &[u8],
            idx: usize,
            frontier_count: usize,
            frontier_window_size: usize,
        ) {
            let window_area = frontier_window_size * frontier_window_size;
            let packed_window_size = window_area.div_ceil(8);
            let dst = &mut dst[idx * frontier_count * packed_window_size
                ..(idx + 1) * frontier_count * packed_window_size];
            for (src_window, dst_window) in src
                .chunks_exact(window_area)
                .zip(dst.chunks_exact_mut(packed_window_size))
            {
                for (dst_byte, src_bits) in dst_window.iter_mut().zip(src_window.chunks(8)) {
                    *dst_byte = src_bits
                        .iter()
                        .enumerate()
                        .fold(0, |byte, (bit_idx, &bit)| byte | (bit << bit_idx));
                }
            }
        }

        copy_row(
            &mut self.inventory,
            &features.inventory,
            idx,
            self.inventory_count,
        );
        copy_row(&mut self.room_x, &features.room_x, idx, self.room_count);
        copy_row(&mut self.room_y, &features.room_y, idx, self.room_count);
        copy_row(
            &mut self.room_placed,
            &features.room_placed,
            idx,
            self.room_count,
        );
        copy_row(
            &mut self.frontier,
            &features.frontier,
            idx,
            self.frontier_count * 7,
        );
        pack_occupancy_windows(
            &mut self.frontier_occupancy,
            &features.frontier_occupancy,
            idx,
            self.frontier_count,
            self.frontier_window_size,
        );
        copy_row(
            &mut self.frontier_neighbor,
            &features.frontier_neighbor,
            idx,
            self.frontier_count * self.frontier_neighbor_count,
        );
        copy_row(
            &mut self.frontier_neighbor_pair,
            &features.frontier_neighbor_pair,
            idx,
            self.frontier_count * self.frontier_neighbor_count,
        );
        copy_row(
            &mut self.frontier_obstruction,
            &features.frontier_obstruction,
            idx,
            self.frontier_count * self.frontier_neighbor_count * 3,
        );
    }
}

impl StateFeatureBuffers {
    fn new(
        common_data: &CommonData,
        snapshot_count: usize,
        frontier_count: usize,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
    ) -> Self {
        Self {
            inventory: vec![0; snapshot_count * common_data.connection_variant_rooms.len()],
            room_x: vec![0; snapshot_count * common_data.room.len()],
            room_y: vec![0; snapshot_count * common_data.room.len()],
            room_placed: vec![0; snapshot_count * common_data.room.len()],
            frontier: vec![0; snapshot_count * frontier_count * 7],
            frontier_occupancy: vec![
                0;
                snapshot_count
                    * frontier_count
                    * (frontier_window_size * frontier_window_size).div_ceil(8)
            ],
            frontier_neighbor: vec![-1; snapshot_count * frontier_count * frontier_neighbor_count],
            frontier_neighbor_pair: vec![
                0;
                snapshot_count * frontier_count * frontier_neighbor_count
            ],
            frontier_obstruction: vec![
                0;
                snapshot_count * frontier_count * frontier_neighbor_count * 3
            ],
        }
    }

    fn output_shard(
        &mut self,
        snapshot_start: usize,
        snapshot_count: usize,
        inventory_count: usize,
        room_count: usize,
        frontier_count: usize,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
    ) -> StateFeatureOutputShards {
        fn output_shard<T>(values: &mut [T], start: usize, len: usize) -> OutputShard<T> {
            OutputShard::from_slice(&mut values[start..start + len])
        }

        let inventory_start = snapshot_start * inventory_count;
        let room_start = snapshot_start * room_count;
        let frontier_start = snapshot_start * frontier_count;
        let packed_window_size = (frontier_window_size * frontier_window_size).div_ceil(8);
        StateFeatureOutputShards {
            inventory: output_shard(
                &mut self.inventory,
                inventory_start,
                snapshot_count * inventory_count,
            ),
            room_x: output_shard(&mut self.room_x, room_start, snapshot_count * room_count),
            room_y: output_shard(&mut self.room_y, room_start, snapshot_count * room_count),
            room_placed: output_shard(
                &mut self.room_placed,
                room_start,
                snapshot_count * room_count,
            ),
            frontier: output_shard(
                &mut self.frontier,
                frontier_start * 7,
                snapshot_count * frontier_count * 7,
            ),
            frontier_occupancy: output_shard(
                &mut self.frontier_occupancy,
                frontier_start * packed_window_size,
                snapshot_count * frontier_count * packed_window_size,
            ),
            frontier_neighbor: output_shard(
                &mut self.frontier_neighbor,
                frontier_start * frontier_neighbor_count,
                snapshot_count * frontier_count * frontier_neighbor_count,
            ),
            frontier_neighbor_pair: output_shard(
                &mut self.frontier_neighbor_pair,
                frontier_start * frontier_neighbor_count,
                snapshot_count * frontier_count * frontier_neighbor_count,
            ),
            frontier_obstruction: output_shard(
                &mut self.frontier_obstruction,
                frontier_start * frontier_neighbor_count * 3,
                snapshot_count * frontier_count * frontier_neighbor_count * 3,
            ),
            inventory_count,
            room_count,
            frontier_count,
            frontier_neighbor_count,
            frontier_window_size,
        }
    }
}

impl Drop for EnvironmentGroup {
    fn drop(&mut self) {
        for worker in &mut self.workers {
            worker.shutdown();
        }
    }
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (rooms_json))]
    fn new(rooms_json: &str) -> PyResult<Self> {
        let rooms: Vec<Room> = serde_json::from_str(rooms_json)
            .map_err(|err| PyValueError::new_err(format!("failed to parse rooms JSON: {err}")))?;
        let common_data = Arc::new(CommonData::new(rooms)?);

        Ok(Self { common_data })
    }

    fn num_rooms(&self) -> usize {
        self.common_data.room.len()
    }

    #[pyo3(signature = (map_size, num_environments, seed, frontier_neighbor_count, frontier_window_size, num_threads=None))]
    fn create_environment_group(
        &self,
        map_size: (Coord, Coord),
        num_environments: usize,
        seed: u64,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
        num_threads: Option<usize>,
    ) -> PyResult<EnvironmentGroup> {
        EnvironmentGroup::new(
            Arc::clone(&self.common_data),
            map_size,
            num_environments,
            seed,
            frontier_neighbor_count,
            frontier_window_size,
            num_threads,
        )
    }

    fn get_output_sizes(&self) -> (usize, usize) {
        output_sizes(&self.common_data)
    }

    fn get_state_feature_sizes(&self) -> (usize, usize, usize) {
        (
            self.common_data.connection_variant_rooms.len(),
            Environment::max_frontiers(&self.common_data),
            self.common_data.room.len(),
        )
    }

    fn get_output_metadata(
        &self,
    ) -> (
        Vec<(usize, usize)>,
        Vec<(usize, usize)>,
        usize,
        usize,
        Vec<usize>,
        usize,
    ) {
        let door_output = self
            .common_data
            .door_output
            .iter()
            .map(|output| (output.room_idx as usize, output.variant_outcome_idx))
            .collect();
        let connection_output = self
            .common_data
            .connection_output
            .iter()
            .map(|output| (output.room_idx as usize, output.variant_outcome_idx))
            .collect();
        (
            door_output,
            connection_output,
            self.common_data.num_door_output_variants,
            self.common_data.num_connection_output_variants,
            self.common_data
                .room
                .iter()
                .map(|room| room.connection_variant_idx as usize)
                .collect(),
            self.common_data.connection_variant_rooms.len(),
        )
    }
}

impl EnvironmentGroup {
    fn new(
        common_data: Arc<CommonData>,
        map_size: (Coord, Coord),
        num_environments: usize,
        seed: u64,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
        if frontier_neighbor_count == 0 {
            return Err(PyValueError::new_err(
                "frontier_neighbor_count must be greater than zero",
            ));
        }
        if frontier_window_size == 0 {
            return Err(PyValueError::new_err(
                "frontier_window_size must be greater than zero",
            ));
        }
        let requested_threads = requested_num_threads(num_threads)?;
        let worker_count = min(requested_threads, max(num_environments, 1));

        let base_shard_len = num_environments / worker_count;
        let remainder = num_environments % worker_count;
        let mut workers = Vec::with_capacity(worker_count);
        let mut start = 0;
        for worker_idx in 0..worker_count {
            let shard_len = base_shard_len + usize::from(worker_idx < remainder);
            let end = start + shard_len;
            let mut environments = Vec::with_capacity(shard_len);
            for env_idx in start..end {
                environments.push(Environment::new(
                    &common_data,
                    map_size,
                    seed ^ env_idx as u64,
                ));
            }
            workers.push(spawn_worker(
                worker_idx,
                start,
                environments,
                Arc::clone(&common_data),
            )?);
            start = end;
        }

        Ok(Self {
            common_data,
            workers,
            num_environments,
            frontier_neighbor_count,
            frontier_window_size,
            action_count: 0,
            state_feature_worker_ns: AtomicU64::new(0),
            state_feature_pack_ns: AtomicU64::new(0),
            state_feature_profile_calls: AtomicUsize::new(0),
            state_feature_occupancy_prefix_ns: AtomicU64::new(0),
            state_feature_clone_ns: AtomicU64::new(0),
            state_feature_step_ns: AtomicU64::new(0),
            state_feature_assemble_ns: AtomicU64::new(0),
            state_feature_assemble_setup_ns: AtomicU64::new(0),
            state_feature_assemble_frontier_ns: AtomicU64::new(0),
            state_feature_assemble_neighbor_ns: AtomicU64::new(0),
            state_feature_assemble_pair_ns: AtomicU64::new(0),
            state_feature_assemble_pair_flags_ns: AtomicU64::new(0),
            state_feature_assemble_pair_obstruction_ns: AtomicU64::new(0),
            state_feature_assemble_pair_obstruction_base_ns: AtomicU64::new(0),
            state_feature_assemble_pair_obstruction_candidate_ns: AtomicU64::new(0),
            state_feature_assemble_output_ns: AtomicU64::new(0),
        })
    }
}

#[pymethods]
impl EnvironmentGroup {
    fn clear(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                if let Err(err) = worker.send(WorkerCommand::Clear) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }

            wait_for_done_responses(&self.workers, sent_workers, first_error)
        })?;

        self.action_count = 0;
        Ok(())
    }

    fn finish(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                if let Err(err) = worker.send(WorkerCommand::Finish) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }

            wait_for_done_responses(&self.workers, sent_workers, first_error)
        })
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
        let action_count = self.action_count;
        let output_len = self.num_environments * action_count;
        let mut room_idx = vec![0; output_len];
        let mut room_x = vec![0; output_len];
        let mut room_y = vec![0; output_len];

        py.detach(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                let output_start = worker.start * action_count;
                let output_end = output_start + worker.len * action_count;

                if let Err(err) = worker.send(WorkerCommand::GetActions {
                    action_count,
                    room_idx: OutputShard::from_slice(&mut room_idx[output_start..output_end]),
                    room_x: OutputShard::from_slice(&mut room_x[output_start..output_end]),
                    room_y: OutputShard::from_slice(&mut room_y[output_start..output_end]),
                }) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }

            wait_for_done_responses(&self.workers, sent_workers, first_error)
        })?;

        Ok((
            pyarray2_from_flat_vec(py, room_idx, self.num_environments, action_count)?,
            pyarray2_from_flat_vec(py, room_x, self.num_environments, action_count)?,
            pyarray2_from_flat_vec(py, room_y, self.num_environments, action_count)?,
        ))
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        room_idx: PyReadonlyArray1<'py, RoomIdx>,
        room_x: PyReadonlyArray1<'py, Coord>,
        room_y: PyReadonlyArray1<'py, Coord>,
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

        if room_idx.len() != self.num_environments {
            return Err(PyValueError::new_err(format!(
                "action arrays must have length num_environments {}; got {}",
                self.num_environments,
                room_idx.len(),
            )));
        }

        py.detach(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                let action_start = worker.start;
                let action_end = worker.end();
                if let Err(err) = worker.send(WorkerCommand::Step {
                    room_idx: InputShard::from_slice(&room_idx[action_start..action_end]),
                    room_x: InputShard::from_slice(&room_x[action_start..action_end]),
                    room_y: InputShard::from_slice(&room_y[action_start..action_end]),
                }) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }

            wait_for_done_responses(&self.workers, sent_workers, first_error)
        })?;

        self.action_count += 1;
        Ok(())
    }

    fn replay<'py>(
        &mut self,
        py: Python<'py>,
        room_idx: PyReadonlyArray2<'py, RoomIdx>,
        room_x: PyReadonlyArray2<'py, Coord>,
        room_y: PyReadonlyArray2<'py, Coord>,
    ) -> PyResult<()> {
        let room_idx_shape = room_idx.as_array().shape().to_vec();
        let room_x_shape = room_x.as_array().shape().to_vec();
        let room_y_shape = room_y.as_array().shape().to_vec();
        if room_idx_shape != room_x_shape || room_idx_shape != room_y_shape {
            return Err(PyValueError::new_err(format!(
                "room_idx, room_x, and room_y must have the same shape; got {:?}, {:?}, and {:?}",
                room_idx_shape, room_x_shape, room_y_shape
            )));
        }

        if room_idx_shape[0] != self.num_environments {
            return Err(PyValueError::new_err(format!(
                "action arrays must have first dimension num_environments {}; got {}",
                self.num_environments, room_idx_shape[0],
            )));
        }

        let action_count = room_idx_shape[1];
        let room_idx = room_idx
            .as_slice()
            .map_err(|_| PyValueError::new_err("room_idx must be a contiguous 2D numpy array"))?;
        let room_x = room_x
            .as_slice()
            .map_err(|_| PyValueError::new_err("room_x must be a contiguous 2D numpy array"))?;
        let room_y = room_y
            .as_slice()
            .map_err(|_| PyValueError::new_err("room_y must be a contiguous 2D numpy array"))?;

        py.detach(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                let action_start = worker.start * action_count;
                let action_end = worker.end() * action_count;
                if let Err(err) = worker.send(WorkerCommand::Replay {
                    action_count,
                    room_idx: InputShard::from_slice(&room_idx[action_start..action_end]),
                    room_x: InputShard::from_slice(&room_x[action_start..action_end]),
                    room_y: InputShard::from_slice(&room_y[action_start..action_end]),
                }) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }

            wait_for_done_responses(&self.workers, sent_workers, first_error)
        })?;

        self.action_count = action_count;
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn get_candidates<'py>(
        &mut self,
        py: Python<'py>,
        mut max_candidates: usize,
    ) -> PyResult<(
        Bound<'py, PyArray2<RoomIdx>>,
        Bound<'py, PyArray2<Coord>>,
        Bound<'py, PyArray2<Coord>>,
    )> {
        if self.action_count == 0 {
            max_candidates = 1;
        }
        let output_len = self.num_environments * max_candidates;
        let dummy_candidate = Action {
            room_idx: self.common_data.room.len() as RoomIdx, // an invalid room index to indicate no-op
            x: 0,
            y: 0,
        };

        let mut room_idx = vec![dummy_candidate.room_idx; output_len];
        let mut room_x = vec![dummy_candidate.x; output_len];
        let mut room_y = vec![dummy_candidate.y; output_len];

        py.detach(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                let output_start = worker.start * max_candidates;
                let output_end = output_start + worker.len * max_candidates;

                if let Err(err) = worker.send(WorkerCommand::GetCandidates {
                    max_candidates,
                    room_idx: OutputShard::from_slice(&mut room_idx[output_start..output_end]),
                    room_x: OutputShard::from_slice(&mut room_x[output_start..output_end]),
                    room_y: OutputShard::from_slice(&mut room_y[output_start..output_end]),
                }) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }

            wait_for_done_responses(&self.workers, sent_workers, first_error)
        })?;

        Ok((
            pyarray2_from_flat_vec(py, room_idx, self.num_environments, max_candidates)?,
            pyarray2_from_flat_vec(py, room_x, self.num_environments, max_candidates)?,
            pyarray2_from_flat_vec(py, room_y, self.num_environments, max_candidates)?,
        ))
    }

    #[allow(clippy::type_complexity)]
    fn get_candidates_with_outcomes<'py>(
        &mut self,
        py: Python<'py>,
        mut max_candidates: usize,
    ) -> PyResult<(
        Bound<'py, PyArray2<RoomIdx>>,
        Bound<'py, PyArray2<Coord>>,
        Bound<'py, PyArray2<Coord>>,
        Bound<'py, PyArray3<i8>>,
        Bound<'py, PyArray3<i8>>,
    )> {
        if self.action_count == 0 {
            max_candidates = 1;
        }
        let (door_outcome_count, connection_outcome_count) = output_sizes(&self.common_data);
        let output_len = self.num_environments * max_candidates;
        let door_output_len = output_len * door_outcome_count;
        let connection_output_len = output_len * connection_outcome_count;
        let dummy_candidate = Action {
            room_idx: self.common_data.room.len() as RoomIdx, // an invalid room index to indicate no-op
            x: 0,
            y: 0,
        };

        let mut room_idx = vec![dummy_candidate.room_idx; output_len];
        let mut room_x = vec![dummy_candidate.x; output_len];
        let mut room_y = vec![dummy_candidate.y; output_len];
        let mut door_valid = vec![DoorValidOutcome::Unknown as i8; door_output_len];
        let mut connections_valid = vec![DoorValidOutcome::Unknown as i8; connection_output_len];

        py.detach(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                let output_start = worker.start * max_candidates;
                let output_end = output_start + worker.len * max_candidates;
                let door_output_start = output_start * door_outcome_count;
                let door_output_end = output_end * door_outcome_count;
                let connection_output_start = output_start * connection_outcome_count;
                let connection_output_end = output_end * connection_outcome_count;

                if let Err(err) = worker.send(WorkerCommand::GetCandidatesWithOutcomes {
                    max_candidates,
                    room_idx: OutputShard::from_slice(&mut room_idx[output_start..output_end]),
                    room_x: OutputShard::from_slice(&mut room_x[output_start..output_end]),
                    room_y: OutputShard::from_slice(&mut room_y[output_start..output_end]),
                    door_outcome_count,
                    connection_outcome_count,
                    door_valid: OutputShard::from_slice(
                        &mut door_valid[door_output_start..door_output_end],
                    ),
                    connections_valid: OutputShard::from_slice(
                        &mut connections_valid[connection_output_start..connection_output_end],
                    ),
                }) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }

            wait_for_done_responses(&self.workers, sent_workers, first_error)
        })?;

        Ok((
            pyarray2_from_flat_vec(py, room_idx, self.num_environments, max_candidates)?,
            pyarray2_from_flat_vec(py, room_x, self.num_environments, max_candidates)?,
            pyarray2_from_flat_vec(py, room_y, self.num_environments, max_candidates)?,
            pyarray3_from_flat_vec(
                py,
                door_valid,
                self.num_environments,
                max_candidates,
                door_outcome_count,
            )?,
            pyarray3_from_flat_vec(
                py,
                connections_valid,
                self.num_environments,
                max_candidates,
                connection_outcome_count,
            )?,
        ))
    }

    fn get_outcomes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray2<i8>>, Bound<'py, PyArray2<i8>>)> {
        let (door_outcome_count, connection_outcome_count) = output_sizes(&self.common_data);
        let door_output_len = self.num_environments * door_outcome_count;
        let connection_output_len = self.num_environments * connection_outcome_count;
        let mut door_valid = vec![DoorValidOutcome::Unknown as i8; door_output_len];
        let mut connections_valid = vec![DoorValidOutcome::Unknown as i8; connection_output_len];

        py.detach(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                let door_output_start = worker.start * door_outcome_count;
                let door_output_end = door_output_start + worker.len * door_outcome_count;
                let connection_output_start = worker.start * connection_outcome_count;
                let connection_output_end =
                    connection_output_start + worker.len * connection_outcome_count;

                if let Err(err) = worker.send(WorkerCommand::GetOutcomes {
                    door_outcome_count,
                    connection_outcome_count,
                    door_valid: OutputShard::from_slice(
                        &mut door_valid[door_output_start..door_output_end],
                    ),
                    connections_valid: OutputShard::from_slice(
                        &mut connections_valid[connection_output_start..connection_output_end],
                    ),
                }) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }

            wait_for_done_responses(&self.workers, sent_workers, first_error)
        })?;

        Ok((
            pyarray2_from_flat_vec(py, door_valid, self.num_environments, door_outcome_count)?,
            pyarray2_from_flat_vec(
                py,
                connections_valid,
                self.num_environments,
                connection_outcome_count,
            )?,
        ))
    }

    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (environment_start=0, environment_count=None))]
    fn get_state_features<'py>(
        &self,
        py: Python<'py>,
        environment_start: usize,
        environment_count: Option<usize>,
    ) -> PyResult<(
        Bound<'py, PyArray2<u8>>,
        Bound<'py, PyArray2<Coord>>,
        Bound<'py, PyArray2<Coord>>,
        Bound<'py, PyArray2<u8>>,
        Bound<'py, PyArray3<i16>>,
        Bound<'py, PyArray3<u8>>,
        Bound<'py, PyArray3<i16>>,
        Bound<'py, PyArray3<u8>>,
        Bound<'py, PyArray4<u16>>,
    )> {
        let inventory_count = self.common_data.connection_variant_rooms.len();
        let room_count = self.common_data.room.len();
        let Some(remaining_environments) = self.num_environments.checked_sub(environment_start)
        else {
            return Err(PyValueError::new_err(
                "state feature range must fit within the environment group",
            ));
        };
        let environment_count = environment_count.unwrap_or(remaining_environments);
        if environment_count > remaining_environments {
            return Err(PyValueError::new_err(
                "state feature range must fit within the environment group",
            ));
        }
        let (_, frontier_count) = py.detach(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                let start = max(environment_start, worker.start);
                let end = min(environment_start + environment_count, worker.end());
                if start >= end {
                    continue;
                }
                if let Err(err) = worker.send(WorkerCommand::GetStateFeatures {
                    frontier_neighbor_count: self.frontier_neighbor_count,
                    frontier_window_size: self.frontier_window_size,
                    environment_start: start - worker.start,
                    environment_count: end - start,
                }) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }
            collect_state_feature_profiles(&self.workers, sent_workers, first_error)
        })?;
        let mut buffers = StateFeatureBuffers::new(
            &self.common_data,
            environment_count,
            frontier_count,
            self.frontier_neighbor_count,
            self.frontier_window_size,
        );
        py.detach(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                let start = max(environment_start, worker.start);
                let end = min(environment_start + environment_count, worker.end());
                if start >= end {
                    continue;
                }
                if let Err(err) = worker.send(WorkerCommand::PackStateFeatures {
                    outputs: buffers.output_shard(
                        start - environment_start,
                        end - start,
                        inventory_count,
                        room_count,
                        frontier_count,
                        self.frontier_neighbor_count,
                        self.frontier_window_size,
                    ),
                }) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }
            wait_for_done_responses(&self.workers, sent_workers, first_error)
        })?;
        Ok((
            pyarray2_from_flat_vec(py, buffers.inventory, environment_count, inventory_count)?,
            pyarray2_from_flat_vec(py, buffers.room_x, environment_count, room_count)?,
            pyarray2_from_flat_vec(py, buffers.room_y, environment_count, room_count)?,
            pyarray2_from_flat_vec(py, buffers.room_placed, environment_count, room_count)?,
            pyarray3_from_flat_vec(py, buffers.frontier, environment_count, frontier_count, 7)?,
            pyarray3_from_flat_vec(
                py,
                buffers.frontier_occupancy,
                environment_count,
                frontier_count,
                (self.frontier_window_size * self.frontier_window_size).div_ceil(8),
            )?,
            pyarray3_from_flat_vec(
                py,
                buffers.frontier_neighbor,
                environment_count,
                frontier_count,
                self.frontier_neighbor_count,
            )?,
            pyarray3_from_flat_vec(
                py,
                buffers.frontier_neighbor_pair,
                environment_count,
                frontier_count,
                self.frontier_neighbor_count,
            )?,
            pyarray4_from_flat_vec(
                py,
                buffers.frontier_obstruction,
                environment_count,
                frontier_count,
                self.frontier_neighbor_count,
                3,
            )?,
        ))
    }

    #[allow(clippy::type_complexity)]
    fn get_state_features_after_candidates<'py>(
        &self,
        py: Python<'py>,
        room_idx: PyReadonlyArray2<'py, RoomIdx>,
        room_x: PyReadonlyArray2<'py, Coord>,
        room_y: PyReadonlyArray2<'py, Coord>,
        environment_start: usize,
    ) -> PyResult<(
        Bound<'py, PyArray3<u8>>,
        Bound<'py, PyArray3<Coord>>,
        Bound<'py, PyArray3<Coord>>,
        Bound<'py, PyArray3<u8>>,
        Bound<'py, PyArray4<i16>>,
        Bound<'py, PyArray4<u8>>,
        Bound<'py, PyArray4<i16>>,
        Bound<'py, PyArray4<u8>>,
        Bound<'py, PyArray5<u16>>,
    )> {
        let shape = room_idx.as_array().shape().to_vec();
        if room_x.as_array().shape() != shape
            || room_y.as_array().shape() != shape
            || environment_start + shape[0] > self.num_environments
        {
            return Err(PyValueError::new_err(
                "candidate action arrays must fit within the environment group",
            ));
        }
        let candidate_count = shape[1];
        let room_idx = room_idx
            .as_slice()
            .map_err(|_| PyValueError::new_err("room_idx must be contiguous"))?;
        let room_x = room_x
            .as_slice()
            .map_err(|_| PyValueError::new_err("room_x must be contiguous"))?;
        let room_y = room_y
            .as_slice()
            .map_err(|_| PyValueError::new_err("room_y must be contiguous"))?;
        let inventory_count = self.common_data.connection_variant_rooms.len();
        let room_count = self.common_data.room.len();
        let environment_count = shape[0];
        let snapshot_count = environment_count * candidate_count;
        let worker_start = Instant::now();
        let (worker_profile, frontier_count) = py.detach(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                let start = max(environment_start, worker.start);
                let end = min(environment_start + environment_count, worker.end());
                if start >= end {
                    continue;
                }
                let environment_count = end - start;
                let input_start = (start - environment_start) * candidate_count;
                let len = environment_count * candidate_count;
                if let Err(err) = worker.send(WorkerCommand::GetStateFeaturesAfterCandidates {
                    frontier_neighbor_count: self.frontier_neighbor_count,
                    frontier_window_size: self.frontier_window_size,
                    environment_start: start - worker.start,
                    environment_count,
                    candidate_count,
                    room_idx: InputShard::from_slice(&room_idx[input_start..input_start + len]),
                    room_x: InputShard::from_slice(&room_x[input_start..input_start + len]),
                    room_y: InputShard::from_slice(&room_y[input_start..input_start + len]),
                }) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }
            collect_state_feature_profiles(&self.workers, sent_workers, first_error)
        })?;
        self.state_feature_worker_ns
            .fetch_add(worker_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let mut buffers = StateFeatureBuffers::new(
            &self.common_data,
            snapshot_count,
            frontier_count,
            self.frontier_neighbor_count,
            self.frontier_window_size,
        );
        let pack_start = Instant::now();
        py.detach(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                let start = max(environment_start, worker.start);
                let end = min(environment_start + environment_count, worker.end());
                if start >= end {
                    continue;
                }
                let snapshot_start = (start - environment_start) * candidate_count;
                let snapshot_count = (end - start) * candidate_count;
                if let Err(err) = worker.send(WorkerCommand::PackStateFeatures {
                    outputs: buffers.output_shard(
                        snapshot_start,
                        snapshot_count,
                        inventory_count,
                        room_count,
                        frontier_count,
                        self.frontier_neighbor_count,
                        self.frontier_window_size,
                    ),
                }) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }
            wait_for_done_responses(&self.workers, sent_workers, first_error)
        })?;
        self.state_feature_pack_ns
            .fetch_add(pack_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.state_feature_occupancy_prefix_ns
            .fetch_add(worker_profile.occupancy_prefix_ns, Ordering::Relaxed);
        self.state_feature_clone_ns
            .fetch_add(worker_profile.clone_ns, Ordering::Relaxed);
        self.state_feature_step_ns
            .fetch_add(worker_profile.step_ns, Ordering::Relaxed);
        self.state_feature_assemble_ns
            .fetch_add(worker_profile.assemble_ns, Ordering::Relaxed);
        self.state_feature_assemble_setup_ns
            .fetch_add(worker_profile.assemble_setup_ns, Ordering::Relaxed);
        self.state_feature_assemble_frontier_ns
            .fetch_add(worker_profile.assemble_frontier_ns, Ordering::Relaxed);
        self.state_feature_assemble_neighbor_ns
            .fetch_add(worker_profile.assemble_neighbor_ns, Ordering::Relaxed);
        self.state_feature_assemble_pair_ns
            .fetch_add(worker_profile.assemble_pair_ns, Ordering::Relaxed);
        self.state_feature_assemble_pair_flags_ns
            .fetch_add(worker_profile.assemble_pair_flags_ns, Ordering::Relaxed);
        self.state_feature_assemble_pair_obstruction_ns.fetch_add(
            worker_profile.assemble_pair_obstruction_ns,
            Ordering::Relaxed,
        );
        self.state_feature_assemble_pair_obstruction_base_ns
            .fetch_add(
                worker_profile.assemble_pair_obstruction_base_ns,
                Ordering::Relaxed,
            );
        self.state_feature_assemble_pair_obstruction_candidate_ns
            .fetch_add(
                worker_profile.assemble_pair_obstruction_candidate_ns,
                Ordering::Relaxed,
            );
        self.state_feature_assemble_output_ns
            .fetch_add(worker_profile.assemble_output_ns, Ordering::Relaxed);
        self.state_feature_profile_calls
            .fetch_add(1, Ordering::Relaxed);
        Ok((
            pyarray3_from_flat_vec(
                py,
                buffers.inventory,
                environment_count,
                candidate_count,
                inventory_count,
            )?,
            pyarray3_from_flat_vec(
                py,
                buffers.room_x,
                environment_count,
                candidate_count,
                room_count,
            )?,
            pyarray3_from_flat_vec(
                py,
                buffers.room_y,
                environment_count,
                candidate_count,
                room_count,
            )?,
            pyarray3_from_flat_vec(
                py,
                buffers.room_placed,
                environment_count,
                candidate_count,
                room_count,
            )?,
            pyarray4_from_flat_vec(
                py,
                buffers.frontier,
                environment_count,
                candidate_count,
                frontier_count,
                7,
            )?,
            pyarray4_from_flat_vec(
                py,
                buffers.frontier_occupancy,
                environment_count,
                candidate_count,
                frontier_count,
                (self.frontier_window_size * self.frontier_window_size).div_ceil(8),
            )?,
            pyarray4_from_flat_vec(
                py,
                buffers.frontier_neighbor,
                environment_count,
                candidate_count,
                frontier_count,
                self.frontier_neighbor_count,
            )?,
            pyarray4_from_flat_vec(
                py,
                buffers.frontier_neighbor_pair,
                environment_count,
                candidate_count,
                frontier_count,
                self.frontier_neighbor_count,
            )?,
            pyarray5_from_flat_vec(
                py,
                buffers.frontier_obstruction,
                environment_count,
                candidate_count,
                frontier_count,
                self.frontier_neighbor_count,
                3,
            )?,
        ))
    }

    fn take_state_feature_profile(&self) -> Vec<f64> {
        vec![
            self.state_feature_worker_ns.swap(0, Ordering::Relaxed) as f64 / 1_000_000_000.0,
            self.state_feature_pack_ns.swap(0, Ordering::Relaxed) as f64 / 1_000_000_000.0,
            self.state_feature_profile_calls.swap(0, Ordering::Relaxed) as f64,
            self.state_feature_occupancy_prefix_ns
                .swap(0, Ordering::Relaxed) as f64
                / 1_000_000_000.0,
            self.state_feature_clone_ns.swap(0, Ordering::Relaxed) as f64 / 1_000_000_000.0,
            self.state_feature_step_ns.swap(0, Ordering::Relaxed) as f64 / 1_000_000_000.0,
            self.state_feature_assemble_ns.swap(0, Ordering::Relaxed) as f64 / 1_000_000_000.0,
            self.state_feature_assemble_setup_ns
                .swap(0, Ordering::Relaxed) as f64
                / 1_000_000_000.0,
            self.state_feature_assemble_frontier_ns
                .swap(0, Ordering::Relaxed) as f64
                / 1_000_000_000.0,
            self.state_feature_assemble_neighbor_ns
                .swap(0, Ordering::Relaxed) as f64
                / 1_000_000_000.0,
            self.state_feature_assemble_pair_ns
                .swap(0, Ordering::Relaxed) as f64
                / 1_000_000_000.0,
            self.state_feature_assemble_pair_flags_ns
                .swap(0, Ordering::Relaxed) as f64
                / 1_000_000_000.0,
            self.state_feature_assemble_pair_obstruction_ns
                .swap(0, Ordering::Relaxed) as f64
                / 1_000_000_000.0,
            self.state_feature_assemble_pair_obstruction_base_ns
                .swap(0, Ordering::Relaxed) as f64
                / 1_000_000_000.0,
            self.state_feature_assemble_pair_obstruction_candidate_ns
                .swap(0, Ordering::Relaxed) as f64
                / 1_000_000_000.0,
            self.state_feature_assemble_output_ns
                .swap(0, Ordering::Relaxed) as f64
                / 1_000_000_000.0,
        ]
    }
}
