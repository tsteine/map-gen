/// The `engine` module exposes the map generation environment to Python through the Engine and
/// EnvironmentGroup classes. It handles the creation and management of worker threads that run
/// environment simulations in parallel.
use crate::common::{
    Action, CommonData, Coord, Direction, DoorValidOutcome, DoorVariantIdx, FrontierIdx, Room,
    RoomIdx,
};
use crate::environment::{
    Environment, FEATURE_FRONTIER_WIDTH, FeatureConfig, Features, FrontierNeighborAlgorithm,
    Outcomes,
};
use crossbeam_channel as channel;
use numpy::{
    Element, IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods, PyReadonlyArray1,
    PyReadonlyArray2, PyReadonlyArray3, PyReadwriteArray1, PyReadwriteArray2,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::cmp::{max, min};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const PROFILE_METRIC_COUNT: usize = 21;
const PROFILE_METRIC_NAMES: [&str; PROFILE_METRIC_COUNT] = [
    "worker.clear",
    "worker.finish",
    "worker.step",
    "worker.get_candidates_with_outcomes",
    "worker.get_actions",
    "worker.get_outcomes",
    "worker.get_door_match_counts",
    "worker.get_door_matches",
    "worker.get_features",
    "worker.get_features_after_candidates",
    "worker.get_sparse_features_after_candidates",
    "worker.get_feature_frontier_count_after_candidates",
    "worker.pack_features",
    "env.step.push_action",
    "env.step.mark_room_used",
    "env.step.components_edges",
    "env.step.occupancy",
    "env.step.match_existing_frontiers",
    "env.step.build_new_frontier_candidates",
    "env.step.filter_existing_frontiers",
    "worker.step_known",
];

static PROFILE_ENABLED: AtomicBool = AtomicBool::new(false);
static PROFILE_COUNTS: [AtomicU64; PROFILE_METRIC_COUNT] =
    [const { AtomicU64::new(0) }; PROFILE_METRIC_COUNT];
static PROFILE_NANOS: [AtomicU64; PROFILE_METRIC_COUNT] =
    [const { AtomicU64::new(0) }; PROFILE_METRIC_COUNT];

pub(crate) fn profile_enabled() -> bool {
    PROFILE_ENABLED.load(Ordering::Relaxed)
}

pub(crate) fn record_profile_metric(metric_idx: usize, duration: Duration) {
    if PROFILE_ENABLED.load(Ordering::Relaxed) {
        PROFILE_COUNTS[metric_idx].fetch_add(1, Ordering::Relaxed);
        PROFILE_NANOS[metric_idx].fetch_add(
            duration.as_nanos().min(u128::from(u64::MAX)) as u64,
            Ordering::Relaxed,
        );
    }
}

pub fn set_profile_enabled(enabled: bool) {
    PROFILE_ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn reset_profile() {
    for metric_idx in 0..PROFILE_METRIC_COUNT {
        PROFILE_COUNTS[metric_idx].store(0, Ordering::Relaxed);
        PROFILE_NANOS[metric_idx].store(0, Ordering::Relaxed);
    }
}

pub fn profile_report() -> Vec<(String, u64, u64)> {
    PROFILE_METRIC_NAMES
        .iter()
        .enumerate()
        .map(|(metric_idx, name)| {
            (
                (*name).to_string(),
                PROFILE_COUNTS[metric_idx].load(Ordering::Relaxed),
                PROFILE_NANOS[metric_idx].load(Ordering::Relaxed),
            )
        })
        .collect()
}

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
    fn empty() -> Self {
        Self {
            ptr: NonNull::<T>::dangling().as_ptr(),
            len: 0,
            _marker: PhantomData,
        }
    }

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
    StepKnown {
        room_idx: InputShard<RoomIdx>,
        room_x: InputShard<Coord>,
        room_y: InputShard<Coord>,
    },
    GetCandidatesWithOutcomes {
        frontier_neighbor_algorithm: FrontierNeighborAlgorithm,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
        recommended_candidates: usize,
        exploration_candidates: usize,
        proposal_temperature: InputShard<f32>,
        proposal_scores: Option<InputShard<f32>>,
        proposal_frontier_count: usize,
        proposal_door_variant_count: usize,
        room_idx: OutputShard<RoomIdx>,
        room_x: OutputShard<Coord>,
        room_y: OutputShard<Coord>,
        frontier_count: OutputShard<u16>,
        proposal_frontier_idx: OutputShard<FrontierIdx>,
        proposal_door_variant_idx: OutputShard<DoorVariantIdx>,
        door_outcome_count: usize,
        connection_outcome_count: usize,
        pre_door_valid: OutputShard<i8>,
        pre_connections_valid: OutputShard<i8>,
        door_valid: OutputShard<i8>,
        connections_valid: OutputShard<i8>,
        door_match: OutputShard<i16>,
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
        verify_consistency: bool,
        door_valid: OutputShard<i8>,
        connections_valid: OutputShard<i8>,
    },
    GetOutcomesAfterCandidates {
        environment_start: usize,
        environment_count: usize,
        candidate_count: usize,
        room_idx: InputShard<RoomIdx>,
        room_x: InputShard<Coord>,
        room_y: InputShard<Coord>,
        door_outcome_count: usize,
        connection_outcome_count: usize,
        door_valid: OutputShard<i8>,
        connections_valid: OutputShard<i8>,
        door_match: OutputShard<i16>,
    },
    GetDoorMatchCounts {
        horizontal_counts: OutputShard<u64>,
        vertical_counts: OutputShard<u64>,
    },
    GetDoorMatches {
        left_count: usize,
        right_count: usize,
        up_count: usize,
        down_count: usize,
        left: OutputShard<i16>,
        right: OutputShard<i16>,
        up: OutputShard<i16>,
        down: OutputShard<i16>,
    },
    GetFeatures {
        frontier_neighbor_algorithm: FrontierNeighborAlgorithm,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
        environment_start: usize,
        environment_count: usize,
    },
    GetFeaturesAfterCandidates {
        frontier_neighbor_algorithm: FrontierNeighborAlgorithm,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
        environment_start: usize,
        environment_count: usize,
        candidate_count: usize,
        room_idx: InputShard<RoomIdx>,
        room_x: InputShard<Coord>,
        room_y: InputShard<Coord>,
        outputs: FeatureOutputShards,
    },
    GetSparseFeaturesAfterCandidates {
        frontier_neighbor_algorithm: FrontierNeighborAlgorithm,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
        environment_start: usize,
        environment_count: usize,
        candidate_count: usize,
        room_idx: InputShard<RoomIdx>,
        room_x: InputShard<Coord>,
        room_y: InputShard<Coord>,
        outputs: SparseFeatureOutputShards,
    },
    GetFeatureFrontierCountAfterCandidates {
        environment_start: usize,
        environment_count: usize,
        candidate_count: usize,
        room_idx: InputShard<RoomIdx>,
        room_x: InputShard<Coord>,
        room_y: InputShard<Coord>,
    },
    PackFeatures {
        outputs: FeatureOutputShards,
    },
    PackSparseFeatures {
        outputs: SparseFeatureOutputShards,
    },
    Shutdown,
}

impl WorkerCommand {
    fn profile_metric_idx(&self) -> Option<usize> {
        match self {
            WorkerCommand::Clear => Some(0),
            WorkerCommand::Finish => Some(1),
            WorkerCommand::Step { .. } => Some(2),
            WorkerCommand::GetCandidatesWithOutcomes { .. } => Some(3),
            WorkerCommand::GetActions { .. } => Some(4),
            WorkerCommand::GetOutcomes { .. } => Some(5),
            WorkerCommand::GetOutcomesAfterCandidates { .. } => Some(5),
            WorkerCommand::GetDoorMatchCounts { .. } => Some(6),
            WorkerCommand::GetDoorMatches { .. } => Some(7),
            WorkerCommand::GetFeatures { .. } => Some(8),
            WorkerCommand::GetFeaturesAfterCandidates { .. } => Some(9),
            WorkerCommand::GetSparseFeaturesAfterCandidates { .. } => Some(10),
            WorkerCommand::GetFeatureFrontierCountAfterCandidates { .. } => Some(11),
            WorkerCommand::PackFeatures { .. } => Some(12),
            WorkerCommand::PackSparseFeatures { .. } => Some(12),
            WorkerCommand::StepKnown { .. } => Some(20),
            WorkerCommand::Shutdown => None,
        }
    }
}

// Feature preparation reports only metadata needed to allocate output buffers. Bulk data is
// written through shared memory, and other commands return "done" when they finish.
enum WorkerResponse {
    Done,
    Error(String),
    FeatureInfo(usize, usize),
}

struct WorkerHandle {
    start: usize,
    len: usize,
    command_tx: channel::Sender<WorkerCommand>,
    response_rx: channel::Receiver<WorkerResponse>,
    join_handle: Option<JoinHandle<()>>,
}

#[derive(Clone, Copy)]
enum StepCommandKind {
    Step,
    StepKnown,
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
            WorkerResponse::Error(err) => Err(PyRuntimeError::new_err(err)),
            WorkerResponse::FeatureInfo(_, _) => Err(PyRuntimeError::new_err(
                "engine worker thread returned unexpected feature info",
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
    features: FeatureConfig,
    command_rx: channel::Receiver<WorkerCommand>,
    response_tx: channel::Sender<WorkerResponse>,
) {
    let mut pending_features = Vec::new();
    while let Ok(command) = command_rx.recv() {
        let profile_metric_idx = command.profile_metric_idx();
        let profile_start = if PROFILE_ENABLED.load(Ordering::Relaxed) {
            Some(Instant::now())
        } else {
            None
        };
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
            WorkerCommand::StepKnown {
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
                    env.step_known(
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
            WorkerCommand::GetCandidatesWithOutcomes {
                frontier_neighbor_algorithm,
                frontier_neighbor_count,
                frontier_window_size,
                recommended_candidates,
                exploration_candidates,
                proposal_temperature,
                proposal_scores,
                proposal_frontier_count,
                proposal_door_variant_count,
                room_idx,
                room_x,
                room_y,
                frontier_count,
                proposal_frontier_idx,
                proposal_door_variant_idx,
                door_outcome_count,
                connection_outcome_count,
                pre_door_valid,
                pre_connections_valid,
                door_valid,
                connections_valid,
                door_match,
            } => {
                // SAFETY: The main thread guarantees that for the duration of this command,
                // the output slices remain valid and that no other thread accesses them.
                let room_idx = unsafe { room_idx.into_mut_slice() };
                let room_x = unsafe { room_x.into_mut_slice() };
                let room_y = unsafe { room_y.into_mut_slice() };
                let current_frontier_count = unsafe { frontier_count.into_mut_slice() };
                let proposal_temperature = unsafe { proposal_temperature.into_slice() };
                let proposal_scores = proposal_scores.map(|scores| unsafe { scores.into_slice() });
                let proposal_frontier_idx = unsafe { proposal_frontier_idx.into_mut_slice() };
                let proposal_door_variant_idx =
                    unsafe { proposal_door_variant_idx.into_mut_slice() };
                let pre_door_valid = unsafe { pre_door_valid.into_mut_slice() };
                let pre_connections_valid = unsafe { pre_connections_valid.into_mut_slice() };
                let door_valid = unsafe { door_valid.into_mut_slice() };
                let connections_valid = unsafe { connections_valid.into_mut_slice() };
                let door_match = unsafe { door_match.into_mut_slice() };
                let max_candidates = recommended_candidates + exploration_candidates;
                debug_assert_eq!(room_idx.len(), environments.len() * max_candidates);
                debug_assert_eq!(room_x.len(), environments.len() * max_candidates);
                debug_assert_eq!(room_y.len(), environments.len() * max_candidates);
                debug_assert_eq!(current_frontier_count.len(), environments.len());
                debug_assert_eq!(proposal_temperature.len(), environments.len());
                debug_assert_eq!(
                    proposal_frontier_idx.len(),
                    environments.len() * max_candidates
                );
                debug_assert_eq!(
                    proposal_door_variant_idx.len(),
                    environments.len() * max_candidates
                );
                debug_assert_eq!(
                    pre_door_valid.len(),
                    environments.len() * door_outcome_count
                );
                debug_assert_eq!(
                    pre_connections_valid.len(),
                    environments.len() * connection_outcome_count
                );
                debug_assert_eq!(
                    door_valid.len(),
                    environments.len() * max_candidates * door_outcome_count
                );
                debug_assert_eq!(
                    connections_valid.len(),
                    environments.len() * max_candidates * connection_outcome_count
                );
                debug_assert_eq!(
                    door_match.len(),
                    environments.len() * max_candidates * door_outcome_count
                );

                let mut consistency_error = None;
                pending_features.clear();
                for (env_idx, env) in environments.iter_mut().enumerate() {
                    current_frontier_count[env_idx] = u16::try_from(env.frontier_count())
                        .expect("frontier count must fit in u16");
                    let proposal_score_start =
                        env_idx * proposal_frontier_count * proposal_door_variant_count;
                    let proposal_score_end = proposal_score_start
                        + proposal_frontier_count * proposal_door_variant_count;
                    let env_proposal_scores = proposal_scores
                        .as_ref()
                        .map(|scores| &scores[proposal_score_start..proposal_score_end]);
                    let (
                        pre_candidate_outcomes,
                        candidates,
                        candidate_frontier_idx,
                        candidate_door_variant_idx,
                        outcomes,
                        door_matches,
                        mut candidate_features,
                    ) = match env.get_filtered_candidates_with_outcomes(
                        &common_data,
                        recommended_candidates,
                        exploration_candidates,
                        proposal_temperature[env_idx],
                        env_proposal_scores,
                        proposal_frontier_count,
                        proposal_door_variant_count,
                        &features,
                        frontier_neighbor_algorithm,
                        frontier_neighbor_count,
                        frontier_window_size,
                    ) {
                        Ok(result) => result,
                        Err(err) => {
                            consistency_error = Some(err);
                            break;
                        }
                    };
                    let pre_door_start = env_idx * door_outcome_count;
                    for (outcome_idx, outcome) in
                        pre_candidate_outcomes.door_valid.iter().enumerate()
                    {
                        pre_door_valid[pre_door_start + outcome_idx] = *outcome as i8;
                    }
                    let pre_connection_start = env_idx * connection_outcome_count;
                    for (outcome_idx, outcome) in
                        pre_candidate_outcomes.connections_valid.iter().enumerate()
                    {
                        pre_connections_valid[pre_connection_start + outcome_idx] = *outcome as i8;
                    }
                    let row_start = env_idx * max_candidates;
                    let dummy_outcome = if candidates.len() < max_candidates {
                        Some(Outcomes {
                            door_valid: vec![DoorValidOutcome::Unknown; door_outcome_count],
                            connections_valid: vec![
                                DoorValidOutcome::Unknown;
                                connection_outcome_count
                            ],
                        })
                    } else {
                        None
                    };
                    let dummy_door_match = if candidates.len() < max_candidates {
                        Some(vec![-1; door_outcome_count])
                    } else {
                        None
                    };
                    let dummy_candidate = Action {
                        room_idx: common_data.room.len() as RoomIdx,
                        x: 0,
                        y: 0,
                    };
                    for candidate_idx in 0..max_candidates {
                        let idx = row_start + candidate_idx;
                        if let Some(candidate) = candidates.get(candidate_idx) {
                            room_idx[idx] = candidate.room_idx;
                            room_x[idx] = candidate.x;
                            room_y[idx] = candidate.y;
                        }
                        if let Some(&frontier_idx) = candidate_frontier_idx.get(candidate_idx) {
                            proposal_frontier_idx[idx] = frontier_idx;
                        }
                        if let Some(&door_variant_idx) =
                            candidate_door_variant_idx.get(candidate_idx)
                        {
                            proposal_door_variant_idx[idx] = door_variant_idx;
                        }

                        let outcome = outcomes
                            .get(candidate_idx)
                            .or(dummy_outcome.as_ref())
                            .expect("dummy outcome must exist for padded candidates");
                        let match_values = door_matches
                            .get(candidate_idx)
                            .or(dummy_door_match.as_ref())
                            .expect("dummy door match must exist for padded candidates");
                        if candidate_idx >= candidate_features.len() {
                            candidate_features.push(env.features_after_candidate(
                                &common_data,
                                dummy_candidate,
                                &features,
                                frontier_neighbor_algorithm,
                                frontier_neighbor_count,
                                frontier_window_size,
                            ));
                        }
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
                        for (dst, &value) in door_match[door_start..door_end]
                            .iter_mut()
                            .zip(match_values)
                        {
                            *dst = value;
                        }
                    }
                    pending_features.append(&mut candidate_features);
                }
                match consistency_error {
                    Some(err) => WorkerResponse::Error(err),
                    None => {
                        let frontier_count = max_feature_frontier_count(&pending_features);
                        WorkerResponse::FeatureInfo(
                            frontier_count,
                            pending_features
                                .iter()
                                .map(|features| features.frontier.len() / FEATURE_FRONTIER_WIDTH)
                                .sum(),
                        )
                    }
                }
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

                for (env_idx, env) in environments.iter_mut().enumerate() {
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
                verify_consistency,
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

                let mut consistency_error = None;
                for (env_idx, env) in environments.iter_mut().enumerate() {
                    let outcomes = if verify_consistency {
                        match env.verified_outcomes(&common_data, "get_outcomes") {
                            Ok(outcomes) => outcomes,
                            Err(err) => {
                                consistency_error = Some(err);
                                break;
                            }
                        }
                    } else {
                        env.outcomes(&common_data)
                    };
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
                match consistency_error {
                    Some(err) => WorkerResponse::Error(err),
                    None => WorkerResponse::Done,
                }
            }
            WorkerCommand::GetOutcomesAfterCandidates {
                environment_start,
                environment_count,
                candidate_count,
                room_idx,
                room_x,
                room_y,
                door_outcome_count,
                connection_outcome_count,
                door_valid,
                connections_valid,
                door_match,
            } => {
                // SAFETY: The main thread guarantees that for the duration of this command,
                // the input and output slices remain valid and that no other thread mutates
                // them.
                let room_idx = unsafe { room_idx.into_slice() };
                let room_x = unsafe { room_x.into_slice() };
                let room_y = unsafe { room_y.into_slice() };
                let door_valid = unsafe { door_valid.into_mut_slice() };
                let connections_valid = unsafe { connections_valid.into_mut_slice() };
                let door_match = unsafe { door_match.into_mut_slice() };
                debug_assert_eq!(room_idx.len(), environment_count * candidate_count);
                debug_assert_eq!(room_x.len(), environment_count * candidate_count);
                debug_assert_eq!(room_y.len(), environment_count * candidate_count);
                debug_assert_eq!(
                    door_valid.len(),
                    environment_count * candidate_count * door_outcome_count
                );
                debug_assert_eq!(
                    connections_valid.len(),
                    environment_count * candidate_count * connection_outcome_count
                );
                debug_assert_eq!(
                    door_match.len(),
                    environment_count * candidate_count * door_outcome_count
                );

                for (env_idx, env) in environments
                    .iter_mut()
                    .skip(environment_start)
                    .take(environment_count)
                    .enumerate()
                {
                    for candidate_idx in 0..candidate_count {
                        let input_idx = env_idx * candidate_count + candidate_idx;
                        let (outcomes, match_values) = env.outcomes_after_candidate(
                            &common_data,
                            Action {
                                room_idx: room_idx[input_idx],
                                x: room_x[input_idx],
                                y: room_y[input_idx],
                            },
                        );
                        debug_assert_eq!(outcomes.door_valid.len(), door_outcome_count);
                        debug_assert_eq!(
                            outcomes.connections_valid.len(),
                            connection_outcome_count
                        );
                        debug_assert_eq!(match_values.len(), door_outcome_count);
                        let door_start = input_idx * door_outcome_count;
                        for (outcome_idx, outcome) in outcomes.door_valid.iter().enumerate() {
                            door_valid[door_start + outcome_idx] = match outcome {
                                DoorValidOutcome::Unknown => -1,
                                DoorValidOutcome::Valid => 0,
                                DoorValidOutcome::Invalid => 1,
                            };
                        }
                        for (outcome_idx, &value) in match_values.iter().enumerate() {
                            door_match[door_start + outcome_idx] = value;
                        }
                        let connection_start = input_idx * connection_outcome_count;
                        for (outcome_idx, outcome) in outcomes.connections_valid.iter().enumerate()
                        {
                            connections_valid[connection_start + outcome_idx] = match outcome {
                                DoorValidOutcome::Unknown => -1,
                                DoorValidOutcome::Valid => 0,
                                DoorValidOutcome::Invalid => 1,
                            };
                        }
                    }
                }
                WorkerResponse::Done
            }
            WorkerCommand::GetDoorMatchCounts {
                horizontal_counts,
                vertical_counts,
            } => {
                // SAFETY: The main thread guarantees that for the duration of this command,
                // the output slices remain valid and that no other thread accesses them.
                let horizontal_counts = unsafe { horizontal_counts.into_mut_slice() };
                let vertical_counts = unsafe { vertical_counts.into_mut_slice() };

                for env in &environments {
                    env.add_door_match_counts(&common_data, horizontal_counts, vertical_counts);
                }
                WorkerResponse::Done
            }
            WorkerCommand::GetDoorMatches {
                left_count,
                right_count,
                up_count,
                down_count,
                left,
                right,
                up,
                down,
            } => {
                // SAFETY: The main thread guarantees that for the duration of this command,
                // the output slices remain valid and that no other thread accesses them.
                let left = unsafe { left.into_mut_slice() };
                let right = unsafe { right.into_mut_slice() };
                let up = unsafe { up.into_mut_slice() };
                let down = unsafe { down.into_mut_slice() };
                debug_assert_eq!(left.len(), environments.len() * left_count);
                debug_assert_eq!(right.len(), environments.len() * right_count);
                debug_assert_eq!(up.len(), environments.len() * up_count);
                debug_assert_eq!(down.len(), environments.len() * down_count);

                for (env_idx, env) in environments.iter().enumerate() {
                    let left_start = env_idx * left_count;
                    let right_start = env_idx * right_count;
                    let up_start = env_idx * up_count;
                    let down_start = env_idx * down_count;
                    env.write_door_matches(
                        &mut left[left_start..left_start + left_count],
                        &mut right[right_start..right_start + right_count],
                        &mut up[up_start..up_start + up_count],
                        &mut down[down_start..down_start + down_count],
                    );
                }
                WorkerResponse::Done
            }
            WorkerCommand::GetFeatures {
                frontier_neighbor_algorithm,
                frontier_neighbor_count,
                frontier_window_size,
                environment_start,
                environment_count,
            } => {
                pending_features = environments
                    .iter()
                    .skip(environment_start)
                    .take(environment_count)
                    .map(|env| {
                        env.features(
                            &common_data,
                            &features,
                            frontier_neighbor_algorithm,
                            frontier_neighbor_count,
                            frontier_window_size,
                        )
                    })
                    .collect();
                let frontier_count = max_feature_frontier_count(&pending_features);
                WorkerResponse::FeatureInfo(
                    frontier_count,
                    pending_features
                        .iter()
                        .map(|features| features.frontier.len() / FEATURE_FRONTIER_WIDTH)
                        .sum(),
                )
            }
            WorkerCommand::GetFeaturesAfterCandidates {
                frontier_neighbor_algorithm,
                frontier_neighbor_count,
                frontier_window_size,
                environment_start,
                environment_count,
                candidate_count,
                room_idx,
                room_x,
                room_y,
                outputs,
            } => {
                let room_idx = unsafe { room_idx.into_slice() };
                let room_x = unsafe { room_x.into_slice() };
                let room_y = unsafe { room_y.into_slice() };
                let mut outputs = unsafe { outputs.into_slices() };
                for (env_idx, env) in environments
                    .iter_mut()
                    .skip(environment_start)
                    .take(environment_count)
                    .enumerate()
                {
                    for candidate_idx in 0..candidate_count {
                        let idx = env_idx * candidate_count + candidate_idx;
                        let features = env.features_after_candidate(
                            &common_data,
                            Action {
                                room_idx: room_idx[idx],
                                x: room_x[idx],
                                y: room_y[idx],
                            },
                            &features,
                            frontier_neighbor_algorithm,
                            frontier_neighbor_count,
                            frontier_window_size,
                        );
                        outputs.write_features(idx, &features);
                    }
                }
                WorkerResponse::FeatureInfo(0, 0)
            }
            WorkerCommand::GetSparseFeaturesAfterCandidates {
                frontier_neighbor_algorithm,
                frontier_neighbor_count,
                frontier_window_size,
                environment_start,
                environment_count,
                candidate_count,
                room_idx,
                room_x,
                room_y,
                outputs,
            } => {
                let room_idx = unsafe { room_idx.into_slice() };
                let room_x = unsafe { room_x.into_slice() };
                let room_y = unsafe { room_y.into_slice() };
                let mut outputs = unsafe { outputs.into_slices() };
                for (env_idx, env) in environments
                    .iter_mut()
                    .skip(environment_start)
                    .take(environment_count)
                    .enumerate()
                {
                    for candidate_idx in 0..candidate_count {
                        let idx = env_idx * candidate_count + candidate_idx;
                        let features = env.features_after_candidate(
                            &common_data,
                            Action {
                                room_idx: room_idx[idx],
                                x: room_x[idx],
                                y: room_y[idx],
                            },
                            &features,
                            frontier_neighbor_algorithm,
                            frontier_neighbor_count,
                            frontier_window_size,
                        );
                        outputs.write_features(idx, &features);
                    }
                }
                WorkerResponse::FeatureInfo(0, outputs.sparse_row_count)
            }
            WorkerCommand::GetFeatureFrontierCountAfterCandidates {
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
                let mut frontier_count = 0;
                let mut sparse_row_count = 0;
                for (env_idx, env) in environments
                    .iter()
                    .skip(environment_start)
                    .take(environment_count)
                    .enumerate()
                {
                    for candidate_idx in 0..candidate_count {
                        let idx = env_idx * candidate_count + candidate_idx;
                        let candidate_frontier_count = env.feature_frontier_count_after_candidate(
                            Action {
                                room_idx: room_idx[idx],
                                x: room_x[idx],
                                y: room_y[idx],
                            },
                            &common_data,
                        );
                        frontier_count = max(frontier_count, candidate_frontier_count);
                        sparse_row_count += candidate_frontier_count;
                    }
                }
                WorkerResponse::FeatureInfo(frontier_count, sparse_row_count)
            }
            WorkerCommand::PackFeatures { outputs } => {
                let mut outputs = unsafe { outputs.into_slices() };
                for (idx, features) in pending_features.drain(..).enumerate() {
                    outputs.write_features(idx, &features);
                }
                WorkerResponse::Done
            }
            WorkerCommand::PackSparseFeatures { outputs } => {
                let mut outputs = unsafe { outputs.into_slices() };
                for (idx, features) in pending_features.drain(..).enumerate() {
                    outputs.write_features(idx, &features);
                }
                WorkerResponse::FeatureInfo(0, outputs.sparse_row_count)
            }
            WorkerCommand::Shutdown => break,
        };
        if let (Some(metric_idx), Some(start)) = (profile_metric_idx, profile_start) {
            record_profile_metric(metric_idx, start.elapsed());
        }

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
    features: FeatureConfig,
) -> PyResult<WorkerHandle> {
    let len = environments.len();
    let (command_tx, command_rx) = channel::bounded(1);
    let (response_tx, response_rx) = channel::bounded(1);
    let join_handle = thread::Builder::new()
        .name(format!("map-gen-worker-{worker_idx}"))
        .spawn(move || worker_loop(environments, common_data, features, command_rx, response_tx))
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

fn collect_feature_info(
    workers: &[WorkerHandle],
    sent_workers: Vec<usize>,
    mut first_error: Option<PyErr>,
) -> PyResult<(usize, usize, Vec<usize>)> {
    let mut frontier_count = 0;
    let mut sparse_row_count = 0;
    let mut worker_sparse_row_counts = vec![0; workers.len()];
    for worker_idx in sent_workers {
        match workers[worker_idx].recv() {
            Ok(WorkerResponse::FeatureInfo(worker_frontier_count, worker_sparse_row_count)) => {
                frontier_count = max(frontier_count, worker_frontier_count);
                sparse_row_count += worker_sparse_row_count;
                worker_sparse_row_counts[worker_idx] = worker_sparse_row_count;
            }
            Ok(WorkerResponse::Done) => set_first_error(
                &mut first_error,
                PyRuntimeError::new_err("engine worker thread returned no feature info"),
            ),
            Ok(WorkerResponse::Error(err)) => {
                set_first_error(&mut first_error, PyRuntimeError::new_err(err))
            }
            Err(err) => set_first_error(&mut first_error, err),
        }
    }

    if let Some(err) = first_error {
        Err(err)
    } else {
        Ok((frontier_count, sparse_row_count, worker_sparse_row_counts))
    }
}

fn check_dim(name: &str, actual: usize, expected: usize) -> PyResult<()> {
    if actual != expected {
        Err(PyValueError::new_err(format!(
            "{name} has wrong width: expected {expected}, got {actual}"
        )))
    } else {
        Ok(())
    }
}

fn max_feature_frontier_count(features: &[Features]) -> usize {
    features
        .iter()
        .map(|features| features.frontier.len() / FEATURE_FRONTIER_WIDTH)
        .max()
        .unwrap_or(0)
}

#[pyclass(module = "map_gen")]
pub struct Engine {
    common_data: Arc<CommonData>, // pre-computed data that can be shared across environments
    features: FeatureConfig,
}

#[pyclass(module = "map_gen")]
pub struct EnvironmentGroup {
    common_data: Arc<CommonData>,
    features: FeatureConfig,
    workers: Vec<WorkerHandle>, // fixed worker-owned environment shards
    num_environments: usize,
    frontier_neighbor_algorithm: FrontierNeighborAlgorithm,
    frontier_neighbor_count: usize,
    frontier_window_size: usize,
    action_count: usize,
}

#[pyclass(module = "map_gen")]
pub struct CandidatesWithOutcomes {
    room_idx: Py<PyArray2<RoomIdx>>,
    room_x: Py<PyArray2<Coord>>,
    room_y: Py<PyArray2<Coord>>,
    frontier_count: Py<PyArray1<u16>>,
    proposal_frontier_idx: Py<PyArray2<FrontierIdx>>,
    proposal_door_variant_idx: Py<PyArray2<DoorVariantIdx>>,
    pre_door_valid: Py<PyArray2<i8>>,
    pre_connections_valid: Py<PyArray2<i8>>,
    door_valid: Py<PyArray3<i8>>,
    connections_valid: Py<PyArray3<i8>>,
    door_match: Py<PyArray3<i16>>,
    #[pyo3(get)]
    feature_frontier_count: usize,
    #[pyo3(get)]
    sparse_row_count: usize,
    #[pyo3(get)]
    worker_sparse_row_counts: Vec<usize>,
}

#[pymethods]
impl CandidatesWithOutcomes {
    #[getter]
    fn room_idx(&self, py: Python<'_>) -> Py<PyArray2<RoomIdx>> {
        self.room_idx.clone_ref(py)
    }

    #[getter]
    fn room_x(&self, py: Python<'_>) -> Py<PyArray2<Coord>> {
        self.room_x.clone_ref(py)
    }

    #[getter]
    fn room_y(&self, py: Python<'_>) -> Py<PyArray2<Coord>> {
        self.room_y.clone_ref(py)
    }

    #[getter]
    fn frontier_count(&self, py: Python<'_>) -> Py<PyArray1<u16>> {
        self.frontier_count.clone_ref(py)
    }

    #[getter]
    fn proposal_frontier_idx(&self, py: Python<'_>) -> Py<PyArray2<FrontierIdx>> {
        self.proposal_frontier_idx.clone_ref(py)
    }

    #[getter]
    fn proposal_door_variant_idx(&self, py: Python<'_>) -> Py<PyArray2<DoorVariantIdx>> {
        self.proposal_door_variant_idx.clone_ref(py)
    }

    #[getter]
    fn pre_door_valid(&self, py: Python<'_>) -> Py<PyArray2<i8>> {
        self.pre_door_valid.clone_ref(py)
    }

    #[getter]
    fn pre_connections_valid(&self, py: Python<'_>) -> Py<PyArray2<i8>> {
        self.pre_connections_valid.clone_ref(py)
    }

    #[getter]
    fn door_valid(&self, py: Python<'_>) -> Py<PyArray3<i8>> {
        self.door_valid.clone_ref(py)
    }

    #[getter]
    fn connections_valid(&self, py: Python<'_>) -> Py<PyArray3<i8>> {
        self.connections_valid.clone_ref(py)
    }

    #[getter]
    fn door_match(&self, py: Python<'_>) -> Py<PyArray3<i16>> {
        self.door_match.clone_ref(py)
    }
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

struct FeatureBuffers {
    inventory: Vec<u8>,
    room_x: Vec<Coord>,
    room_y: Vec<Coord>,
    room_placed: Vec<u8>,
    frontier: Vec<i8>,
    frontier_occupancy: Vec<u8>,
    frontier_neighbor: Vec<i16>,
    frontier_neighbor_pair: Vec<u8>,
    connection_reachability: Vec<u8>,
    frontier_connection_reachability: Vec<u8>,
}

struct FeatureOutputShards {
    inventory: OutputShard<u8>,
    room_x: OutputShard<Coord>,
    room_y: OutputShard<Coord>,
    room_placed: OutputShard<u8>,
    frontier: OutputShard<i8>,
    frontier_occupancy: OutputShard<u8>,
    frontier_neighbor: OutputShard<i16>,
    frontier_neighbor_pair: OutputShard<u8>,
    connection_reachability: OutputShard<u8>,
    frontier_connection_reachability: OutputShard<u8>,
    inventory_count: usize,
    room_count: usize,
    connection_count: usize,
    frontier_count: usize,
    frontier_neighbor_count: usize,
    frontier_window_size: usize,
}

struct FeatureOutputSlices<'a> {
    inventory: &'a mut [u8],
    room_x: &'a mut [Coord],
    room_y: &'a mut [Coord],
    room_placed: &'a mut [u8],
    frontier: &'a mut [i8],
    frontier_occupancy: &'a mut [u8],
    frontier_neighbor: &'a mut [i16],
    frontier_neighbor_pair: &'a mut [u8],
    connection_reachability: &'a mut [u8],
    frontier_connection_reachability: &'a mut [u8],
    inventory_count: usize,
    room_count: usize,
    connection_count: usize,
    frontier_count: usize,
    frontier_neighbor_count: usize,
    frontier_window_size: usize,
}

impl FeatureOutputShards {
    unsafe fn into_slices<'a>(self) -> FeatureOutputSlices<'a> {
        FeatureOutputSlices {
            inventory: unsafe { self.inventory.into_mut_slice() },
            room_x: unsafe { self.room_x.into_mut_slice() },
            room_y: unsafe { self.room_y.into_mut_slice() },
            room_placed: unsafe { self.room_placed.into_mut_slice() },
            frontier: unsafe { self.frontier.into_mut_slice() },
            frontier_occupancy: unsafe { self.frontier_occupancy.into_mut_slice() },
            frontier_neighbor: unsafe { self.frontier_neighbor.into_mut_slice() },
            frontier_neighbor_pair: unsafe { self.frontier_neighbor_pair.into_mut_slice() },
            connection_reachability: unsafe { self.connection_reachability.into_mut_slice() },
            frontier_connection_reachability: unsafe {
                self.frontier_connection_reachability.into_mut_slice()
            },
            inventory_count: self.inventory_count,
            room_count: self.room_count,
            connection_count: self.connection_count,
            frontier_count: self.frontier_count,
            frontier_neighbor_count: self.frontier_neighbor_count,
            frontier_window_size: self.frontier_window_size,
        }
    }
}

impl FeatureOutputSlices<'_> {
    fn write_fixed_features(&mut self, idx: usize, features: &Features) {
        fn copy_row<T: Copy>(dst: &mut [T], row: &[T], idx: usize, stride: usize) {
            if row.is_empty() {
                return;
            }
            dst[idx * stride..idx * stride + row.len()].copy_from_slice(row);
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
            &mut self.connection_reachability,
            &features.connection_reachability,
            idx,
            self.connection_count,
        );
    }

    fn write_features(&mut self, idx: usize, features: &Features) {
        fn copy_row<T: Copy>(dst: &mut [T], row: &[T], idx: usize, stride: usize) {
            if row.is_empty() {
                return;
            }
            dst[idx * stride..idx * stride + row.len()].copy_from_slice(row);
        }

        self.write_fixed_features(idx, features);
        copy_row(
            &mut self.frontier,
            &features.frontier,
            idx,
            self.frontier_count * FEATURE_FRONTIER_WIDTH,
        );
        copy_row(
            &mut self.frontier_occupancy,
            &features.frontier_occupancy,
            idx,
            self.frontier_count
                * (self.frontier_window_size * self.frontier_window_size).div_ceil(8),
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
            &mut self.frontier_connection_reachability,
            &features.frontier_connection_reachability,
            idx,
            self.frontier_count * self.connection_count,
        );
    }

    fn write_frontier_row(&mut self, dst_idx: usize, features: &Features, src_idx: usize) {
        fn copy_row<T: Copy>(
            dst: &mut [T],
            src: &[T],
            dst_idx: usize,
            src_idx: usize,
            row_width: usize,
        ) {
            if src.is_empty() {
                return;
            }
            dst[dst_idx * row_width..(dst_idx + 1) * row_width]
                .copy_from_slice(&src[src_idx * row_width..(src_idx + 1) * row_width]);
        }

        copy_row(
            &mut self.frontier,
            &features.frontier,
            dst_idx,
            src_idx,
            FEATURE_FRONTIER_WIDTH,
        );
        copy_row(
            &mut self.frontier_occupancy,
            &features.frontier_occupancy,
            dst_idx,
            src_idx,
            self.frontier_window_size.pow(2).div_ceil(8),
        );
        copy_row(
            &mut self.frontier_neighbor,
            &features.frontier_neighbor,
            dst_idx,
            src_idx,
            self.frontier_neighbor_count,
        );
        copy_row(
            &mut self.frontier_neighbor_pair,
            &features.frontier_neighbor_pair,
            dst_idx,
            src_idx,
            self.frontier_neighbor_count,
        );
        copy_row(
            &mut self.frontier_connection_reachability,
            &features.frontier_connection_reachability,
            dst_idx,
            src_idx,
            self.connection_count,
        );
    }
}

struct SparseFeatureBuffers {
    fixed: FeatureBuffers,
    sparse: FeatureBuffers,
    dense_row_idx: Vec<i64>,
}

struct SparseFeatureOutputShards {
    fixed: FeatureOutputShards,
    sparse: FeatureOutputShards,
    dense_row_idx: OutputShard<i64>,
    snapshot_start: usize,
    dense_frontier_count: usize,
}

struct SparseFeatureOutputSlices<'a> {
    fixed: FeatureOutputSlices<'a>,
    sparse: FeatureOutputSlices<'a>,
    dense_row_idx: &'a mut [i64],
    snapshot_start: usize,
    dense_frontier_count: usize,
    sparse_row_count: usize,
}

impl SparseFeatureOutputShards {
    unsafe fn into_slices<'a>(self) -> SparseFeatureOutputSlices<'a> {
        SparseFeatureOutputSlices {
            fixed: unsafe { self.fixed.into_slices() },
            sparse: unsafe { self.sparse.into_slices() },
            dense_row_idx: unsafe { self.dense_row_idx.into_mut_slice() },
            snapshot_start: self.snapshot_start,
            dense_frontier_count: self.dense_frontier_count,
            sparse_row_count: 0,
        }
    }
}

impl SparseFeatureOutputSlices<'_> {
    fn write_features(&mut self, snapshot_idx: usize, features: &Features) {
        self.fixed.write_fixed_features(snapshot_idx, features);
        let frontier_count = features.frontier.len() / FEATURE_FRONTIER_WIDTH;
        for frontier_idx in 0..frontier_count {
            let sparse_row_idx = self.sparse_row_count;
            self.dense_row_idx[sparse_row_idx] = ((self.snapshot_start + snapshot_idx)
                * self.dense_frontier_count
                + frontier_idx) as i64;
            self.sparse
                .write_frontier_row(sparse_row_idx, features, frontier_idx);
            self.sparse_row_count += 1;
        }
    }
}

impl FeatureBuffers {
    fn new(
        common_data: &CommonData,
        features: &FeatureConfig,
        snapshot_count: usize,
        frontier_count: usize,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
    ) -> Self {
        Self {
            inventory: vec![
                0;
                snapshot_count
                    * common_data.connection_variant_rooms.len()
                    * usize::from(features.inventory)
            ],
            room_x: vec![
                0;
                snapshot_count
                    * common_data.room.len()
                    * usize::from(features.room_position)
            ],
            room_y: vec![
                0;
                snapshot_count
                    * common_data.room.len()
                    * usize::from(features.room_position)
            ],
            room_placed: vec![
                0;
                snapshot_count
                    * common_data.room.len()
                    * usize::from(features.room_position)
            ],
            frontier: vec![0; snapshot_count * frontier_count * FEATURE_FRONTIER_WIDTH],
            frontier_occupancy: vec![
                0;
                snapshot_count
                    * frontier_count
                    * (frontier_window_size * frontier_window_size).div_ceil(8)
                    * usize::from(features.frontier_occupancy)
            ],
            frontier_neighbor: vec![
                -1;
                snapshot_count
                    * frontier_count
                    * frontier_neighbor_count
                    * usize::from(features.frontier_neighbor)
            ],
            frontier_neighbor_pair: vec![
                0;
                snapshot_count
                    * frontier_count
                    * frontier_neighbor_count
                    * usize::from(features.frontier_neighbor_flags)
            ],
            connection_reachability: vec![
                0;
                snapshot_count
                    * common_data.room_connection.len()
                    * usize::from(features.connection_reachability)
            ],
            frontier_connection_reachability: vec![
                0;
                snapshot_count
                    * frontier_count
                    * common_data.room_connection.len()
                    * usize::from(
                        features.frontier_connection_reachability
                    )
            ],
        }
    }

    fn output_shard(
        &mut self,
        snapshot_start: usize,
        snapshot_count: usize,
        inventory_count: usize,
        room_count: usize,
        connection_count: usize,
        frontier_count: usize,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
        features: &FeatureConfig,
    ) -> FeatureOutputShards {
        fn output_shard<T>(values: &mut [T], start: usize, len: usize) -> OutputShard<T> {
            OutputShard::from_slice(&mut values[start..start + len])
        }

        let inventory_count = inventory_count * usize::from(features.inventory);
        let room_count = room_count * usize::from(features.room_position);
        let inventory_start = snapshot_start * inventory_count;
        let room_start = snapshot_start * room_count;
        let frontier_start = snapshot_start * frontier_count;
        let packed_window_size = (frontier_window_size * frontier_window_size).div_ceil(8)
            * usize::from(features.frontier_occupancy);
        let output_neighbor_count =
            frontier_neighbor_count * usize::from(features.frontier_neighbor);
        let pair_neighbor_count =
            frontier_neighbor_count * usize::from(features.frontier_neighbor_flags);
        let direct_connection_count =
            connection_count * usize::from(features.connection_reachability);
        let frontier_connection_count =
            connection_count * usize::from(features.frontier_connection_reachability);
        let connection_start = snapshot_start * direct_connection_count;
        FeatureOutputShards {
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
                frontier_start * FEATURE_FRONTIER_WIDTH,
                snapshot_count * frontier_count * FEATURE_FRONTIER_WIDTH,
            ),
            frontier_occupancy: output_shard(
                &mut self.frontier_occupancy,
                frontier_start * packed_window_size,
                snapshot_count * frontier_count * packed_window_size,
            ),
            frontier_neighbor: output_shard(
                &mut self.frontier_neighbor,
                frontier_start * output_neighbor_count,
                snapshot_count * frontier_count * output_neighbor_count,
            ),
            frontier_neighbor_pair: output_shard(
                &mut self.frontier_neighbor_pair,
                frontier_start * pair_neighbor_count,
                snapshot_count * frontier_count * pair_neighbor_count,
            ),
            connection_reachability: output_shard(
                &mut self.connection_reachability,
                connection_start,
                snapshot_count * direct_connection_count,
            ),
            frontier_connection_reachability: output_shard(
                &mut self.frontier_connection_reachability,
                frontier_start * frontier_connection_count,
                snapshot_count * frontier_count * frontier_connection_count,
            ),
            inventory_count,
            room_count,
            connection_count: direct_connection_count.max(frontier_connection_count),
            frontier_count,
            frontier_neighbor_count,
            frontier_window_size,
        }
    }
}

impl SparseFeatureBuffers {
    fn new(
        common_data: &CommonData,
        features: &FeatureConfig,
        snapshot_count: usize,
        sparse_row_count: usize,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
    ) -> Self {
        let sparse_features = FeatureConfig {
            inventory: false,
            temperature: false,
            recommended_candidates: false,
            exploration_candidates: false,
            room_position: false,
            connection_reachability: false,
            ..*features
        };
        Self {
            fixed: FeatureBuffers::new(
                common_data,
                features,
                snapshot_count,
                0,
                frontier_neighbor_count,
                frontier_window_size,
            ),
            sparse: FeatureBuffers::new(
                common_data,
                &sparse_features,
                sparse_row_count,
                1,
                frontier_neighbor_count,
                frontier_window_size,
            ),
            dense_row_idx: vec![0; sparse_row_count],
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn output_shard(
        &mut self,
        snapshot_start: usize,
        snapshot_count: usize,
        sparse_row_start: usize,
        sparse_row_count: usize,
        inventory_count: usize,
        room_count: usize,
        connection_count: usize,
        dense_frontier_count: usize,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
        features: &FeatureConfig,
    ) -> SparseFeatureOutputShards {
        let sparse_features = FeatureConfig {
            inventory: false,
            temperature: false,
            recommended_candidates: false,
            exploration_candidates: false,
            room_position: false,
            connection_reachability: false,
            ..*features
        };
        SparseFeatureOutputShards {
            fixed: self.fixed.output_shard(
                snapshot_start,
                snapshot_count,
                inventory_count,
                room_count,
                connection_count,
                0,
                frontier_neighbor_count,
                frontier_window_size,
                features,
            ),
            sparse: self.sparse.output_shard(
                sparse_row_start,
                sparse_row_count,
                inventory_count,
                room_count,
                connection_count,
                1,
                frontier_neighbor_count,
                frontier_window_size,
                &sparse_features,
            ),
            dense_row_idx: OutputShard::from_slice(
                &mut self.dense_row_idx[sparse_row_start..sparse_row_start + sparse_row_count],
            ),
            snapshot_start,
            dense_frontier_count,
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
    #[pyo3(signature = (rooms_json, features_json))]
    fn new(rooms_json: &str, features_json: &str) -> PyResult<Self> {
        let rooms: Vec<Room> = serde_json::from_str(rooms_json)
            .map_err(|err| PyValueError::new_err(format!("failed to parse rooms JSON: {err}")))?;
        let features: FeatureConfig = serde_json::from_str(features_json).map_err(|err| {
            PyValueError::new_err(format!("failed to parse features JSON: {err}"))
        })?;
        features.validate().map_err(PyValueError::new_err)?;
        let common_data = Arc::new(CommonData::new(rooms)?);

        Ok(Self {
            common_data,
            features,
        })
    }

    fn num_rooms(&self) -> usize {
        self.common_data.room.len()
    }

    #[pyo3(signature = (map_size, num_environments, seed, frontier_neighbor_count, frontier_window_size, num_threads=None, frontier_neighbor_algorithm="delaunay"))]
    fn create_environment_group(
        &self,
        map_size: (Coord, Coord),
        num_environments: usize,
        seed: u64,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
        num_threads: Option<usize>,
        frontier_neighbor_algorithm: &str,
    ) -> PyResult<EnvironmentGroup> {
        let frontier_neighbor_algorithm = match frontier_neighbor_algorithm {
            "delaunay" => FrontierNeighborAlgorithm::Delaunay,
            "nearest" => FrontierNeighborAlgorithm::Nearest,
            "nearest-exclusive" => FrontierNeighborAlgorithm::NearestExclusive,
            _ => {
                return Err(PyValueError::new_err(
                    "frontier_neighbor_algorithm must be \"delaunay\", \"nearest\", or \"nearest-exclusive\"",
                ));
            }
        };
        EnvironmentGroup::new(
            Arc::clone(&self.common_data),
            self.features,
            map_size,
            num_environments,
            seed,
            frontier_neighbor_algorithm,
            frontier_neighbor_count,
            frontier_window_size,
            num_threads,
        )
    }

    fn get_output_sizes(&self) -> (usize, usize) {
        output_sizes(&self.common_data)
    }

    fn get_feature_sizes(&self) -> (usize, usize, usize) {
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
        features: FeatureConfig,
        map_size: (Coord, Coord),
        num_environments: usize,
        seed: u64,
        frontier_neighbor_algorithm: FrontierNeighborAlgorithm,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
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
                features,
            )?);
            start = end;
        }

        Ok(Self {
            common_data,
            features,
            workers,
            num_environments,
            frontier_neighbor_algorithm,
            frontier_neighbor_count,
            frontier_window_size,
            action_count: 0,
        })
    }

    fn step_with_kind<'py>(
        &mut self,
        py: Python<'py>,
        room_idx: PyReadonlyArray1<'py, RoomIdx>,
        room_x: PyReadonlyArray1<'py, Coord>,
        room_y: PyReadonlyArray1<'py, Coord>,
        kind: StepCommandKind,
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
                let command = match kind {
                    StepCommandKind::Step => WorkerCommand::Step {
                        room_idx: InputShard::from_slice(&room_idx[action_start..action_end]),
                        room_x: InputShard::from_slice(&room_x[action_start..action_end]),
                        room_y: InputShard::from_slice(&room_y[action_start..action_end]),
                    },
                    StepCommandKind::StepKnown => WorkerCommand::StepKnown {
                        room_idx: InputShard::from_slice(&room_idx[action_start..action_end]),
                        room_x: InputShard::from_slice(&room_x[action_start..action_end]),
                        room_y: InputShard::from_slice(&room_y[action_start..action_end]),
                    },
                };
                if let Err(err) = worker.send(command) {
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
        self.step_with_kind(py, room_idx, room_x, room_y, StepCommandKind::Step)
    }

    fn step_known<'py>(
        &mut self,
        py: Python<'py>,
        room_idx: PyReadonlyArray1<'py, RoomIdx>,
        room_x: PyReadonlyArray1<'py, Coord>,
        room_y: PyReadonlyArray1<'py, Coord>,
    ) -> PyResult<()> {
        self.step_with_kind(py, room_idx, room_x, room_y, StepCommandKind::StepKnown)
    }

    fn get_candidates_with_outcomes<'py>(
        &mut self,
        py: Python<'py>,
        mut recommended_candidates: usize,
        mut exploration_candidates: usize,
        proposal_temperature: PyReadonlyArray1<'py, f32>,
        proposal_scores: Option<PyReadonlyArray3<'py, f32>>,
    ) -> PyResult<CandidatesWithOutcomes> {
        if self.action_count == 0 {
            recommended_candidates = 0;
            exploration_candidates = 1;
        }
        let max_candidates = recommended_candidates + exploration_candidates;
        let proposal_temperature = proposal_temperature
            .as_slice()
            .map_err(|_| PyValueError::new_err("proposal_temperature must be contiguous"))?;
        if proposal_temperature.len() != self.num_environments {
            return Err(PyValueError::new_err(
                "proposal_temperature must have one value per environment",
            ));
        }
        let proposal_score_shape = proposal_scores
            .as_ref()
            .map(|scores| scores.as_array().shape().to_vec());
        let (proposal_frontier_count, proposal_door_variant_count) =
            if let Some(shape) = proposal_score_shape {
                if shape.len() != 3 || shape[0] != self.num_environments {
                    return Err(PyValueError::new_err(
                        "proposal_scores must have shape [environment, frontier, door_variant]",
                    ));
                }
                (shape[1], shape[2])
            } else {
                (0, 0)
            };
        if self.action_count > 0 && recommended_candidates > 0 && proposal_scores.is_none() {
            return Err(PyValueError::new_err(
                "proposal_scores are required when recommended_candidates is greater than zero",
            ));
        }
        let proposal_scores = proposal_scores
            .as_ref()
            .map(|scores| {
                scores
                    .as_slice()
                    .map_err(|_| PyValueError::new_err("proposal_scores must be contiguous"))
            })
            .transpose()?;
        let (door_outcome_count, connection_outcome_count) = output_sizes(&self.common_data);
        let output_len = self.num_environments * max_candidates;
        let pre_door_output_len = self.num_environments * door_outcome_count;
        let pre_connection_output_len = self.num_environments * connection_outcome_count;
        let door_output_len = output_len * door_outcome_count;
        let connection_output_len = output_len * connection_outcome_count;
        let door_match_output_len = output_len * door_outcome_count;
        let dummy_candidate = Action {
            room_idx: self.common_data.room.len() as RoomIdx, // an invalid room index to indicate no-op
            x: 0,
            y: 0,
        };

        let mut room_idx = vec![dummy_candidate.room_idx; output_len];
        let mut room_x = vec![dummy_candidate.x; output_len];
        let mut room_y = vec![dummy_candidate.y; output_len];
        let mut current_frontier_count = vec![0; self.num_environments];
        let mut proposal_frontier_idx = vec![-1; output_len];
        let mut proposal_door_variant_idx = vec![-1; output_len];
        let mut pre_door_valid = vec![DoorValidOutcome::Unknown as i8; pre_door_output_len];
        let mut pre_connections_valid =
            vec![DoorValidOutcome::Unknown as i8; pre_connection_output_len];
        let mut door_valid = vec![DoorValidOutcome::Unknown as i8; door_output_len];
        let mut connections_valid = vec![DoorValidOutcome::Unknown as i8; connection_output_len];
        let mut door_match = vec![-1; door_match_output_len];

        let (feature_frontier_count, sparse_row_count, worker_sparse_row_counts) =
            py.detach(|| {
                let mut sent_workers = Vec::with_capacity(self.workers.len());
                let mut first_error = None;
                for (worker_idx, worker) in self.workers.iter().enumerate() {
                    let output_start = worker.start * max_candidates;
                    let output_end = output_start + worker.len * max_candidates;
                    let frontier_count_start = worker.start;
                    let frontier_count_end = worker.end();
                    let pre_door_output_start = worker.start * door_outcome_count;
                    let pre_door_output_end =
                        pre_door_output_start + worker.len * door_outcome_count;
                    let pre_connection_output_start = worker.start * connection_outcome_count;
                    let pre_connection_output_end =
                        pre_connection_output_start + worker.len * connection_outcome_count;
                    let door_output_start = output_start * door_outcome_count;
                    let door_output_end = output_end * door_outcome_count;
                    let connection_output_start = output_start * connection_outcome_count;
                    let connection_output_end = output_end * connection_outcome_count;
                    let door_match_output_start = output_start * door_outcome_count;
                    let door_match_output_end = output_end * door_outcome_count;
                    let proposal_score_start =
                        worker.start * proposal_frontier_count * proposal_door_variant_count;
                    let proposal_score_end =
                        worker.end() * proposal_frontier_count * proposal_door_variant_count;

                    if let Err(err) = worker.send(WorkerCommand::GetCandidatesWithOutcomes {
                        recommended_candidates,
                        exploration_candidates,
                        proposal_temperature: InputShard::from_slice(
                            &proposal_temperature[worker.start..worker.end()],
                        ),
                        proposal_scores: proposal_scores.map(|scores| {
                            InputShard::from_slice(
                                &scores[proposal_score_start..proposal_score_end],
                            )
                        }),
                        proposal_frontier_count,
                        proposal_door_variant_count,
                        room_idx: OutputShard::from_slice(&mut room_idx[output_start..output_end]),
                        room_x: OutputShard::from_slice(&mut room_x[output_start..output_end]),
                        room_y: OutputShard::from_slice(&mut room_y[output_start..output_end]),
                        frontier_count: OutputShard::from_slice(
                            &mut current_frontier_count[frontier_count_start..frontier_count_end],
                        ),
                        proposal_frontier_idx: OutputShard::from_slice(
                            &mut proposal_frontier_idx[output_start..output_end],
                        ),
                        proposal_door_variant_idx: OutputShard::from_slice(
                            &mut proposal_door_variant_idx[output_start..output_end],
                        ),
                        frontier_neighbor_algorithm: self.frontier_neighbor_algorithm,
                        frontier_neighbor_count: self.frontier_neighbor_count,
                        frontier_window_size: self.frontier_window_size,
                        door_outcome_count,
                        connection_outcome_count,
                        pre_door_valid: OutputShard::from_slice(
                            &mut pre_door_valid[pre_door_output_start..pre_door_output_end],
                        ),
                        pre_connections_valid: OutputShard::from_slice(
                            &mut pre_connections_valid
                                [pre_connection_output_start..pre_connection_output_end],
                        ),
                        door_valid: OutputShard::from_slice(
                            &mut door_valid[door_output_start..door_output_end],
                        ),
                        connections_valid: OutputShard::from_slice(
                            &mut connections_valid[connection_output_start..connection_output_end],
                        ),
                        door_match: OutputShard::from_slice(
                            &mut door_match[door_match_output_start..door_match_output_end],
                        ),
                    }) {
                        set_first_error(&mut first_error, err);
                        break;
                    }
                    sent_workers.push(worker_idx);
                }

                collect_feature_info(&self.workers, sent_workers, first_error)
            })?;
        let feature_frontier_count =
            feature_frontier_count * usize::from(self.features.has_frontier_features());
        let sparse_row_count =
            sparse_row_count * usize::from(self.features.has_frontier_features());
        let worker_sparse_row_counts = worker_sparse_row_counts
            .into_iter()
            .map(|count| count * usize::from(self.features.has_frontier_features()))
            .collect::<Vec<_>>();

        Ok(CandidatesWithOutcomes {
            room_idx: pyarray2_from_flat_vec(py, room_idx, self.num_environments, max_candidates)?
                .unbind(),
            room_x: pyarray2_from_flat_vec(py, room_x, self.num_environments, max_candidates)?
                .unbind(),
            room_y: pyarray2_from_flat_vec(py, room_y, self.num_environments, max_candidates)?
                .unbind(),
            frontier_count: current_frontier_count.into_pyarray(py).unbind(),
            proposal_frontier_idx: pyarray2_from_flat_vec(
                py,
                proposal_frontier_idx,
                self.num_environments,
                max_candidates,
            )?
            .unbind(),
            proposal_door_variant_idx: pyarray2_from_flat_vec(
                py,
                proposal_door_variant_idx,
                self.num_environments,
                max_candidates,
            )?
            .unbind(),
            pre_door_valid: pyarray2_from_flat_vec(
                py,
                pre_door_valid,
                self.num_environments,
                door_outcome_count,
            )?
            .unbind(),
            pre_connections_valid: pyarray2_from_flat_vec(
                py,
                pre_connections_valid,
                self.num_environments,
                connection_outcome_count,
            )?
            .unbind(),
            door_valid: pyarray3_from_flat_vec(
                py,
                door_valid,
                self.num_environments,
                max_candidates,
                door_outcome_count,
            )?
            .unbind(),
            connections_valid: pyarray3_from_flat_vec(
                py,
                connections_valid,
                self.num_environments,
                max_candidates,
                connection_outcome_count,
            )?
            .unbind(),
            door_match: pyarray3_from_flat_vec(
                py,
                door_match,
                self.num_environments,
                max_candidates,
                door_outcome_count,
            )?
            .unbind(),
            feature_frontier_count,
            sparse_row_count,
            worker_sparse_row_counts,
        })
    }

    fn get_outcomes<'py>(
        &mut self,
        py: Python<'py>,
        verify_consistency: bool,
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
                    verify_consistency,
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

    fn get_outcomes_after_candidates<'py>(
        &mut self,
        py: Python<'py>,
        room_idx: PyReadonlyArray2<'py, RoomIdx>,
        room_x: PyReadonlyArray2<'py, Coord>,
        room_y: PyReadonlyArray2<'py, Coord>,
        environment_start: usize,
    ) -> PyResult<(
        Bound<'py, PyArray3<i8>>,
        Bound<'py, PyArray3<i8>>,
        Bound<'py, PyArray3<i16>>,
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
        let environment_count = shape[0];
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
        let (door_outcome_count, connection_outcome_count) = output_sizes(&self.common_data);
        let door_output_len = environment_count * candidate_count * door_outcome_count;
        let connection_output_len = environment_count * candidate_count * connection_outcome_count;
        let door_match_output_len = environment_count * candidate_count * door_outcome_count;
        let mut door_valid = vec![DoorValidOutcome::Unknown as i8; door_output_len];
        let mut connections_valid = vec![DoorValidOutcome::Unknown as i8; connection_output_len];
        let mut door_match = vec![-1; door_match_output_len];

        py.detach(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                let start = max(environment_start, worker.start);
                let end = min(environment_start + environment_count, worker.end());
                if start >= end {
                    continue;
                }
                let input_start = (start - environment_start) * candidate_count;
                let environment_count = end - start;
                let input_len = environment_count * candidate_count;
                let door_output_start = input_start * door_outcome_count;
                let door_output_end = door_output_start + input_len * door_outcome_count;
                let connection_output_start = input_start * connection_outcome_count;
                let connection_output_end =
                    connection_output_start + input_len * connection_outcome_count;
                let door_match_output_start = input_start * door_outcome_count;
                let door_match_output_end =
                    door_match_output_start + input_len * door_outcome_count;
                if let Err(err) = worker.send(WorkerCommand::GetOutcomesAfterCandidates {
                    environment_start: start - worker.start,
                    environment_count,
                    candidate_count,
                    room_idx: InputShard::from_slice(
                        &room_idx[input_start..input_start + input_len],
                    ),
                    room_x: InputShard::from_slice(&room_x[input_start..input_start + input_len]),
                    room_y: InputShard::from_slice(&room_y[input_start..input_start + input_len]),
                    door_outcome_count,
                    connection_outcome_count,
                    door_valid: OutputShard::from_slice(
                        &mut door_valid[door_output_start..door_output_end],
                    ),
                    connections_valid: OutputShard::from_slice(
                        &mut connections_valid[connection_output_start..connection_output_end],
                    ),
                    door_match: OutputShard::from_slice(
                        &mut door_match[door_match_output_start..door_match_output_end],
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
            pyarray3_from_flat_vec(
                py,
                door_valid,
                environment_count,
                candidate_count,
                door_outcome_count,
            )?,
            pyarray3_from_flat_vec(
                py,
                connections_valid,
                environment_count,
                candidate_count,
                connection_outcome_count,
            )?,
            pyarray3_from_flat_vec(
                py,
                door_match,
                environment_count,
                candidate_count,
                door_outcome_count,
            )?,
        ))
    }

    fn get_door_match_counts<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray2<u64>>, Bound<'py, PyArray2<u64>>)> {
        let left_count = self.common_data.room_dir_door[Direction::Left as usize].len();
        let right_count = self.common_data.room_dir_door[Direction::Right as usize].len();
        let up_count = self.common_data.room_dir_door[Direction::Up as usize].len();
        let down_count = self.common_data.room_dir_door[Direction::Down as usize].len();

        let horizontal_len = (left_count + 1) * (right_count + 1);
        let vertical_len = (up_count + 1) * (down_count + 1);
        let worker_count = self.workers.len();
        let mut worker_horizontal_counts = vec![0; worker_count * horizontal_len];
        let mut worker_vertical_counts = vec![0; worker_count * vertical_len];

        py.detach(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                let horizontal_start = worker_idx * horizontal_len;
                let horizontal_end = horizontal_start + horizontal_len;
                let vertical_start = worker_idx * vertical_len;
                let vertical_end = vertical_start + vertical_len;
                if let Err(err) = worker.send(WorkerCommand::GetDoorMatchCounts {
                    horizontal_counts: OutputShard::from_slice(
                        &mut worker_horizontal_counts[horizontal_start..horizontal_end],
                    ),
                    vertical_counts: OutputShard::from_slice(
                        &mut worker_vertical_counts[vertical_start..vertical_end],
                    ),
                }) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }

            wait_for_done_responses(&self.workers, sent_workers, first_error)
        })?;

        let mut horizontal_counts = vec![0; horizontal_len];
        for worker_counts in worker_horizontal_counts.chunks_exact(horizontal_len) {
            for (dst, &count) in horizontal_counts.iter_mut().zip(worker_counts) {
                *dst += count;
            }
        }
        let mut vertical_counts = vec![0; vertical_len];
        for worker_counts in worker_vertical_counts.chunks_exact(vertical_len) {
            for (dst, &count) in vertical_counts.iter_mut().zip(worker_counts) {
                *dst += count;
            }
        }

        Ok((
            pyarray2_from_flat_vec(py, horizontal_counts, left_count + 1, right_count + 1)?,
            pyarray2_from_flat_vec(py, vertical_counts, up_count + 1, down_count + 1)?,
        ))
    }

    #[allow(clippy::type_complexity)]
    fn get_door_matches<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray2<i16>>,
        Bound<'py, PyArray2<i16>>,
        Bound<'py, PyArray2<i16>>,
        Bound<'py, PyArray2<i16>>,
    )> {
        let left_count = self.common_data.room_dir_door[Direction::Left as usize].len();
        let right_count = self.common_data.room_dir_door[Direction::Right as usize].len();
        let up_count = self.common_data.room_dir_door[Direction::Up as usize].len();
        let down_count = self.common_data.room_dir_door[Direction::Down as usize].len();

        let mut left = vec![-1; self.num_environments * left_count];
        let mut right = vec![-1; self.num_environments * right_count];
        let mut up = vec![-1; self.num_environments * up_count];
        let mut down = vec![-1; self.num_environments * down_count];

        py.detach(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                let left_start = worker.start * left_count;
                let left_end = left_start + worker.len * left_count;
                let right_start = worker.start * right_count;
                let right_end = right_start + worker.len * right_count;
                let up_start = worker.start * up_count;
                let up_end = up_start + worker.len * up_count;
                let down_start = worker.start * down_count;
                let down_end = down_start + worker.len * down_count;

                if let Err(err) = worker.send(WorkerCommand::GetDoorMatches {
                    left_count,
                    right_count,
                    up_count,
                    down_count,
                    left: OutputShard::from_slice(&mut left[left_start..left_end]),
                    right: OutputShard::from_slice(&mut right[right_start..right_end]),
                    up: OutputShard::from_slice(&mut up[up_start..up_end]),
                    down: OutputShard::from_slice(&mut down[down_start..down_end]),
                }) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }

            wait_for_done_responses(&self.workers, sent_workers, first_error)
        })?;

        Ok((
            pyarray2_from_flat_vec(py, left, self.num_environments, left_count)?,
            pyarray2_from_flat_vec(py, right, self.num_environments, right_count)?,
            pyarray2_from_flat_vec(py, up, self.num_environments, up_count)?,
            pyarray2_from_flat_vec(py, down, self.num_environments, down_count)?,
        ))
    }

    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (environment_start=0, environment_count=None))]
    fn get_features<'py>(
        &self,
        py: Python<'py>,
        environment_start: usize,
        environment_count: Option<usize>,
    ) -> PyResult<(
        Bound<'py, PyArray2<u8>>,
        Bound<'py, PyArray2<Coord>>,
        Bound<'py, PyArray2<Coord>>,
        Bound<'py, PyArray2<u8>>,
        Bound<'py, PyArray3<i8>>,
        Bound<'py, PyArray3<u8>>,
        Bound<'py, PyArray3<i16>>,
        Bound<'py, PyArray3<u8>>,
        Bound<'py, PyArray2<u8>>,
        Bound<'py, PyArray3<u8>>,
    )> {
        let inventory_count = self.common_data.connection_variant_rooms.len();
        let room_count = self.common_data.room.len();
        let connection_count = self.common_data.room_connection.len();
        let Some(remaining_environments) = self.num_environments.checked_sub(environment_start)
        else {
            return Err(PyValueError::new_err(
                "feature range must fit within the environment group",
            ));
        };
        let environment_count = environment_count.unwrap_or(remaining_environments);
        if environment_count > remaining_environments {
            return Err(PyValueError::new_err(
                "feature range must fit within the environment group",
            ));
        }
        let (frontier_count, _, _) = py.detach(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                let start = max(environment_start, worker.start);
                let end = min(environment_start + environment_count, worker.end());
                if start >= end {
                    continue;
                }
                if let Err(err) = worker.send(WorkerCommand::GetFeatures {
                    frontier_neighbor_algorithm: self.frontier_neighbor_algorithm,
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
            collect_feature_info(&self.workers, sent_workers, first_error)
        })?;
        let mut buffers = FeatureBuffers::new(
            &self.common_data,
            &self.features,
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
                if let Err(err) = worker.send(WorkerCommand::PackFeatures {
                    outputs: buffers.output_shard(
                        start - environment_start,
                        end - start,
                        inventory_count,
                        room_count,
                        connection_count,
                        frontier_count,
                        self.frontier_neighbor_count,
                        self.frontier_window_size,
                        &self.features,
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
            pyarray2_from_flat_vec(
                py,
                buffers.inventory,
                environment_count,
                inventory_count * usize::from(self.features.inventory),
            )?,
            pyarray2_from_flat_vec(
                py,
                buffers.room_x,
                environment_count,
                room_count * usize::from(self.features.room_position),
            )?,
            pyarray2_from_flat_vec(
                py,
                buffers.room_y,
                environment_count,
                room_count * usize::from(self.features.room_position),
            )?,
            pyarray2_from_flat_vec(
                py,
                buffers.room_placed,
                environment_count,
                room_count * usize::from(self.features.room_position),
            )?,
            pyarray3_from_flat_vec(
                py,
                buffers.frontier,
                environment_count,
                frontier_count,
                FEATURE_FRONTIER_WIDTH,
            )?,
            pyarray3_from_flat_vec(
                py,
                buffers.frontier_occupancy,
                environment_count,
                frontier_count,
                (self.frontier_window_size * self.frontier_window_size).div_ceil(8)
                    * usize::from(self.features.frontier_occupancy),
            )?,
            pyarray3_from_flat_vec(
                py,
                buffers.frontier_neighbor,
                environment_count,
                frontier_count,
                self.frontier_neighbor_count * usize::from(self.features.frontier_neighbor),
            )?,
            pyarray3_from_flat_vec(
                py,
                buffers.frontier_neighbor_pair,
                environment_count,
                frontier_count,
                self.frontier_neighbor_count * usize::from(self.features.frontier_neighbor_flags),
            )?,
            pyarray2_from_flat_vec(
                py,
                buffers.connection_reachability,
                environment_count,
                connection_count * usize::from(self.features.connection_reachability),
            )?,
            pyarray3_from_flat_vec(
                py,
                buffers.frontier_connection_reachability,
                environment_count,
                frontier_count,
                connection_count * usize::from(self.features.frontier_connection_reachability),
            )?,
        ))
    }

    #[allow(clippy::type_complexity)]
    fn get_features_after_candidates<'py>(
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
        Bound<'py, PyArray4<i8>>,
        Bound<'py, PyArray4<u8>>,
        Bound<'py, PyArray4<i16>>,
        Bound<'py, PyArray4<u8>>,
        Bound<'py, PyArray3<u8>>,
        Bound<'py, PyArray4<u8>>,
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
        let connection_count = self.common_data.room_connection.len();
        let environment_count = shape[0];
        let snapshot_count = environment_count * candidate_count;
        let (frontier_count, _, _) = py.detach(|| {
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
                if let Err(err) =
                    worker.send(WorkerCommand::GetFeatureFrontierCountAfterCandidates {
                        environment_start: start - worker.start,
                        environment_count,
                        candidate_count,
                        room_idx: InputShard::from_slice(&room_idx[input_start..input_start + len]),
                        room_x: InputShard::from_slice(&room_x[input_start..input_start + len]),
                        room_y: InputShard::from_slice(&room_y[input_start..input_start + len]),
                    })
                {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }
            collect_feature_info(&self.workers, sent_workers, first_error)
        })?;
        let frontier_count = frontier_count * usize::from(self.features.has_frontier_features());
        let mut buffers = FeatureBuffers::new(
            &self.common_data,
            &self.features,
            snapshot_count,
            frontier_count,
            self.frontier_neighbor_count,
            self.frontier_window_size,
        );
        let _ = py.detach(|| {
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
                let input_start = snapshot_start;
                let len = snapshot_count;
                if let Err(err) = worker.send(WorkerCommand::GetFeaturesAfterCandidates {
                    frontier_neighbor_algorithm: self.frontier_neighbor_algorithm,
                    frontier_neighbor_count: self.frontier_neighbor_count,
                    frontier_window_size: self.frontier_window_size,
                    environment_start: start - worker.start,
                    environment_count: end - start,
                    candidate_count,
                    room_idx: InputShard::from_slice(&room_idx[input_start..input_start + len]),
                    room_x: InputShard::from_slice(&room_x[input_start..input_start + len]),
                    room_y: InputShard::from_slice(&room_y[input_start..input_start + len]),
                    outputs: buffers.output_shard(
                        snapshot_start,
                        snapshot_count,
                        inventory_count,
                        room_count,
                        connection_count,
                        frontier_count,
                        self.frontier_neighbor_count,
                        self.frontier_window_size,
                        &self.features,
                    ),
                }) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }
            collect_feature_info(&self.workers, sent_workers, first_error)
        })?;
        Ok((
            pyarray3_from_flat_vec(
                py,
                buffers.inventory,
                environment_count,
                candidate_count,
                inventory_count * usize::from(self.features.inventory),
            )?,
            pyarray3_from_flat_vec(
                py,
                buffers.room_x,
                environment_count,
                candidate_count,
                room_count * usize::from(self.features.room_position),
            )?,
            pyarray3_from_flat_vec(
                py,
                buffers.room_y,
                environment_count,
                candidate_count,
                room_count * usize::from(self.features.room_position),
            )?,
            pyarray3_from_flat_vec(
                py,
                buffers.room_placed,
                environment_count,
                candidate_count,
                room_count * usize::from(self.features.room_position),
            )?,
            pyarray4_from_flat_vec(
                py,
                buffers.frontier,
                environment_count,
                candidate_count,
                frontier_count,
                FEATURE_FRONTIER_WIDTH,
            )?,
            pyarray4_from_flat_vec(
                py,
                buffers.frontier_occupancy,
                environment_count,
                candidate_count,
                frontier_count,
                (self.frontier_window_size * self.frontier_window_size).div_ceil(8)
                    * usize::from(self.features.frontier_occupancy),
            )?,
            pyarray4_from_flat_vec(
                py,
                buffers.frontier_neighbor,
                environment_count,
                candidate_count,
                frontier_count,
                self.frontier_neighbor_count * usize::from(self.features.frontier_neighbor),
            )?,
            pyarray4_from_flat_vec(
                py,
                buffers.frontier_neighbor_pair,
                environment_count,
                candidate_count,
                frontier_count,
                self.frontier_neighbor_count * usize::from(self.features.frontier_neighbor_flags),
            )?,
            pyarray3_from_flat_vec(
                py,
                buffers.connection_reachability,
                environment_count,
                candidate_count,
                connection_count * usize::from(self.features.connection_reachability),
            )?,
            pyarray4_from_flat_vec(
                py,
                buffers.frontier_connection_reachability,
                environment_count,
                candidate_count,
                frontier_count,
                connection_count * usize::from(self.features.frontier_connection_reachability),
            )?,
        ))
    }

    #[allow(clippy::type_complexity)]
    fn get_sparse_features_after_candidates<'py>(
        &self,
        py: Python<'py>,
        room_idx: PyReadonlyArray2<'py, RoomIdx>,
        room_x: PyReadonlyArray2<'py, Coord>,
        room_y: PyReadonlyArray2<'py, Coord>,
        environment_start: usize,
    ) -> PyResult<(
        (
            Bound<'py, PyArray3<u8>>,
            Bound<'py, PyArray3<Coord>>,
            Bound<'py, PyArray3<Coord>>,
            Bound<'py, PyArray3<u8>>,
            Bound<'py, PyArray2<i8>>,
            Bound<'py, PyArray2<u8>>,
            Bound<'py, PyArray2<i16>>,
            Bound<'py, PyArray2<u8>>,
            Bound<'py, PyArray3<u8>>,
            Bound<'py, PyArray2<u8>>,
            Bound<'py, PyArray1<i64>>,
        ),
        usize,
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
        let connection_count = self.common_data.room_connection.len();
        let environment_count = shape[0];
        let snapshot_count = environment_count * candidate_count;
        let (frontier_count, sparse_row_count, worker_sparse_row_counts) = py.detach(|| {
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
                if let Err(err) =
                    worker.send(WorkerCommand::GetFeatureFrontierCountAfterCandidates {
                        environment_start: start - worker.start,
                        environment_count,
                        candidate_count,
                        room_idx: InputShard::from_slice(&room_idx[input_start..input_start + len]),
                        room_x: InputShard::from_slice(&room_x[input_start..input_start + len]),
                        room_y: InputShard::from_slice(&room_y[input_start..input_start + len]),
                    })
                {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }
            collect_feature_info(&self.workers, sent_workers, first_error)
        })?;
        let frontier_count = frontier_count * usize::from(self.features.has_frontier_features());
        let sparse_row_count =
            sparse_row_count * usize::from(self.features.has_frontier_features());
        let mut buffers = SparseFeatureBuffers::new(
            &self.common_data,
            &self.features,
            snapshot_count,
            sparse_row_count,
            self.frontier_neighbor_count,
            self.frontier_window_size,
        );
        let (_, actual_sparse_row_count, _) = py.detach(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            let mut sparse_row_start = 0;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                let start = max(environment_start, worker.start);
                let end = min(environment_start + environment_count, worker.end());
                if start >= end {
                    continue;
                }
                let snapshot_start = (start - environment_start) * candidate_count;
                let snapshot_count = (end - start) * candidate_count;
                let len = snapshot_count;
                let worker_sparse_row_count = worker_sparse_row_counts[worker_idx]
                    * usize::from(self.features.has_frontier_features());
                if let Err(err) = worker.send(WorkerCommand::GetSparseFeaturesAfterCandidates {
                    frontier_neighbor_algorithm: self.frontier_neighbor_algorithm,
                    frontier_neighbor_count: self.frontier_neighbor_count,
                    frontier_window_size: self.frontier_window_size,
                    environment_start: start - worker.start,
                    environment_count: end - start,
                    candidate_count,
                    room_idx: InputShard::from_slice(
                        &room_idx[snapshot_start..snapshot_start + len],
                    ),
                    room_x: InputShard::from_slice(&room_x[snapshot_start..snapshot_start + len]),
                    room_y: InputShard::from_slice(&room_y[snapshot_start..snapshot_start + len]),
                    outputs: buffers.output_shard(
                        snapshot_start,
                        snapshot_count,
                        sparse_row_start,
                        worker_sparse_row_count,
                        inventory_count,
                        room_count,
                        connection_count,
                        frontier_count,
                        self.frontier_neighbor_count,
                        self.frontier_window_size,
                        &self.features,
                    ),
                }) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
                sparse_row_start += worker_sparse_row_count;
            }
            collect_feature_info(&self.workers, sent_workers, first_error)
        })?;
        if actual_sparse_row_count != sparse_row_count {
            return Err(PyRuntimeError::new_err(format!(
                "sparse feature row count changed between passes: expected {sparse_row_count}, got {actual_sparse_row_count}"
            )));
        }
        Ok((
            (
                pyarray3_from_flat_vec(
                    py,
                    buffers.fixed.inventory,
                    environment_count,
                    candidate_count,
                    inventory_count * usize::from(self.features.inventory),
                )?,
                pyarray3_from_flat_vec(
                    py,
                    buffers.fixed.room_x,
                    environment_count,
                    candidate_count,
                    room_count * usize::from(self.features.room_position),
                )?,
                pyarray3_from_flat_vec(
                    py,
                    buffers.fixed.room_y,
                    environment_count,
                    candidate_count,
                    room_count * usize::from(self.features.room_position),
                )?,
                pyarray3_from_flat_vec(
                    py,
                    buffers.fixed.room_placed,
                    environment_count,
                    candidate_count,
                    room_count * usize::from(self.features.room_position),
                )?,
                pyarray2_from_flat_vec(
                    py,
                    buffers.sparse.frontier,
                    sparse_row_count,
                    FEATURE_FRONTIER_WIDTH,
                )?,
                pyarray2_from_flat_vec(
                    py,
                    buffers.sparse.frontier_occupancy,
                    sparse_row_count,
                    (self.frontier_window_size * self.frontier_window_size).div_ceil(8)
                        * usize::from(self.features.frontier_occupancy),
                )?,
                pyarray2_from_flat_vec(
                    py,
                    buffers.sparse.frontier_neighbor,
                    sparse_row_count,
                    self.frontier_neighbor_count * usize::from(self.features.frontier_neighbor),
                )?,
                pyarray2_from_flat_vec(
                    py,
                    buffers.sparse.frontier_neighbor_pair,
                    sparse_row_count,
                    self.frontier_neighbor_count
                        * usize::from(self.features.frontier_neighbor_flags),
                )?,
                pyarray3_from_flat_vec(
                    py,
                    buffers.fixed.connection_reachability,
                    environment_count,
                    candidate_count,
                    connection_count * usize::from(self.features.connection_reachability),
                )?,
                pyarray2_from_flat_vec(
                    py,
                    buffers.sparse.frontier_connection_reachability,
                    sparse_row_count,
                    connection_count * usize::from(self.features.frontier_connection_reachability),
                )?,
                buffers.dense_row_idx.into_pyarray(py),
            ),
            frontier_count,
        ))
    }

    fn get_sparse_feature_requirements_after_candidates<'py>(
        &self,
        py: Python<'py>,
        room_idx: PyReadonlyArray2<'py, RoomIdx>,
        room_x: PyReadonlyArray2<'py, Coord>,
        room_y: PyReadonlyArray2<'py, Coord>,
        environment_start: usize,
    ) -> PyResult<(usize, usize, Vec<usize>)> {
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
        let environment_count = shape[0];
        let (frontier_count, sparse_row_count, worker_sparse_row_counts) = py.detach(|| {
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
                if let Err(err) =
                    worker.send(WorkerCommand::GetFeatureFrontierCountAfterCandidates {
                        environment_start: start - worker.start,
                        environment_count,
                        candidate_count,
                        room_idx: InputShard::from_slice(&room_idx[input_start..input_start + len]),
                        room_x: InputShard::from_slice(&room_x[input_start..input_start + len]),
                        room_y: InputShard::from_slice(&room_y[input_start..input_start + len]),
                    })
                {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }
            collect_feature_info(&self.workers, sent_workers, first_error)
        })?;
        Ok((
            frontier_count * usize::from(self.features.has_frontier_features()),
            sparse_row_count * usize::from(self.features.has_frontier_features()),
            worker_sparse_row_counts
                .into_iter()
                .map(|count| count * usize::from(self.features.has_frontier_features()))
                .collect(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn pack_sparse_features_after_candidates_into<'py>(
        &self,
        py: Python<'py>,
        environment_count: usize,
        candidate_count: usize,
        environment_start: usize,
        frontier_count: usize,
        sparse_row_count: usize,
        worker_sparse_row_counts: Vec<usize>,
        mut inventory: PyReadwriteArray2<'py, u8>,
        mut out_room_x: PyReadwriteArray2<'py, Coord>,
        mut out_room_y: PyReadwriteArray2<'py, Coord>,
        mut room_placed: PyReadwriteArray2<'py, u8>,
        mut frontier: PyReadwriteArray2<'py, i8>,
        mut frontier_occupancy: PyReadwriteArray2<'py, u8>,
        mut frontier_neighbor: PyReadwriteArray2<'py, i16>,
        mut frontier_neighbor_pair: PyReadwriteArray2<'py, u8>,
        mut connection_reachability: PyReadwriteArray2<'py, u8>,
        mut frontier_connection_reachability: PyReadwriteArray2<'py, u8>,
        mut dense_row_idx: PyReadwriteArray1<'py, i64>,
    ) -> PyResult<()> {
        if environment_start + environment_count > self.num_environments {
            return Err(PyValueError::new_err(
                "candidate dimensions must fit within the environment group",
            ));
        }
        let snapshot_count = environment_count * candidate_count;

        let inventory_count = self.common_data.connection_variant_rooms.len();
        let room_count = self.common_data.room.len();
        let connection_count = self.common_data.room_connection.len();
        let inventory_width = inventory_count * usize::from(self.features.inventory);
        let room_width = room_count * usize::from(self.features.room_position);
        let frontier_occupancy_width = (self.frontier_window_size * self.frontier_window_size)
            .div_ceil(8)
            * usize::from(self.features.frontier_occupancy);
        let frontier_neighbor_width =
            self.frontier_neighbor_count * usize::from(self.features.frontier_neighbor);
        let frontier_neighbor_pair_width =
            self.frontier_neighbor_count * usize::from(self.features.frontier_neighbor_flags);
        let connection_reachability_width =
            connection_count * usize::from(self.features.connection_reachability);
        let frontier_connection_width =
            connection_count * usize::from(self.features.frontier_connection_reachability);

        let inventory_shape = inventory.as_array().shape().to_vec();
        let room_x_shape = out_room_x.as_array().shape().to_vec();
        let room_y_shape = out_room_y.as_array().shape().to_vec();
        let room_placed_shape = room_placed.as_array().shape().to_vec();
        let frontier_shape = frontier.as_array().shape().to_vec();
        let frontier_occupancy_shape = frontier_occupancy.as_array().shape().to_vec();
        let frontier_neighbor_shape = frontier_neighbor.as_array().shape().to_vec();
        let frontier_neighbor_pair_shape = frontier_neighbor_pair.as_array().shape().to_vec();
        let connection_reachability_shape = connection_reachability.as_array().shape().to_vec();
        let frontier_connection_reachability_shape =
            frontier_connection_reachability.as_array().shape().to_vec();
        let dense_row_idx_shape = dense_row_idx.as_array().shape().to_vec();
        if inventory_shape[0] < snapshot_count
            || room_x_shape[0] < snapshot_count
            || room_y_shape[0] < snapshot_count
            || room_placed_shape[0] < snapshot_count
            || connection_reachability_shape[0] < snapshot_count
            || frontier_shape[0] < sparse_row_count
            || frontier_occupancy_shape[0] < sparse_row_count
            || frontier_neighbor_shape[0] < sparse_row_count
            || frontier_neighbor_pair_shape[0] < sparse_row_count
            || frontier_connection_reachability_shape[0] < sparse_row_count
            || dense_row_idx_shape[0] < sparse_row_count
        {
            return Err(PyValueError::new_err(
                "sparse feature output buffer is too small",
            ));
        }
        check_dim("inventory", inventory_shape[1], inventory_width)?;
        check_dim("room_x", room_x_shape[1], room_width)?;
        check_dim("room_y", room_y_shape[1], room_width)?;
        check_dim("room_placed", room_placed_shape[1], room_width)?;
        check_dim("frontier", frontier_shape[1], FEATURE_FRONTIER_WIDTH)?;
        check_dim(
            "frontier_occupancy",
            frontier_occupancy_shape[1],
            frontier_occupancy_width,
        )?;
        check_dim(
            "frontier_neighbor",
            frontier_neighbor_shape[1],
            frontier_neighbor_width,
        )?;
        check_dim(
            "frontier_neighbor_pair",
            frontier_neighbor_pair_shape[1],
            frontier_neighbor_pair_width,
        )?;
        check_dim(
            "connection_reachability",
            connection_reachability_shape[1],
            connection_reachability_width,
        )?;
        check_dim(
            "frontier_connection_reachability",
            frontier_connection_reachability_shape[1],
            frontier_connection_width,
        )?;

        let inventory = inventory
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("inventory must be contiguous"))?;
        let out_room_x = out_room_x
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("room_x must be contiguous"))?;
        let out_room_y = out_room_y
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("room_y must be contiguous"))?;
        let room_placed = room_placed
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("room_placed must be contiguous"))?;
        let frontier = frontier
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("frontier must be contiguous"))?;
        let frontier_occupancy = frontier_occupancy
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("frontier_occupancy must be contiguous"))?;
        let frontier_neighbor = frontier_neighbor
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("frontier_neighbor must be contiguous"))?;
        let frontier_neighbor_pair = frontier_neighbor_pair
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("frontier_neighbor_pair must be contiguous"))?;
        let connection_reachability = connection_reachability
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("connection_reachability must be contiguous"))?;
        let frontier_connection_reachability = frontier_connection_reachability
            .as_slice_mut()
            .map_err(|_| {
                PyValueError::new_err("frontier_connection_reachability must be contiguous")
            })?;
        let dense_row_idx = dense_row_idx
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("dense_row_idx must be contiguous"))?;

        if worker_sparse_row_counts.len() != self.workers.len() {
            return Err(PyValueError::new_err(
                "worker sparse row count length does not match worker count",
            ));
        }

        let (_, actual_sparse_row_count, _) = py.detach(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            let mut sparse_row_start = 0;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                let start = max(environment_start, worker.start);
                let end = min(environment_start + environment_count, worker.end());
                if start >= end {
                    continue;
                }
                let snapshot_start = (start - environment_start) * candidate_count;
                let snapshot_count = (end - start) * candidate_count;
                let worker_sparse_row_count = worker_sparse_row_counts[worker_idx];
                let outputs = SparseFeatureOutputShards {
                    fixed: FeatureOutputShards {
                        inventory: OutputShard::from_slice(
                            &mut inventory[snapshot_start * inventory_width
                                ..(snapshot_start + snapshot_count) * inventory_width],
                        ),
                        room_x: OutputShard::from_slice(
                            &mut out_room_x[snapshot_start * room_width
                                ..(snapshot_start + snapshot_count) * room_width],
                        ),
                        room_y: OutputShard::from_slice(
                            &mut out_room_y[snapshot_start * room_width
                                ..(snapshot_start + snapshot_count) * room_width],
                        ),
                        room_placed: OutputShard::from_slice(
                            &mut room_placed[snapshot_start * room_width
                                ..(snapshot_start + snapshot_count) * room_width],
                        ),
                        frontier: OutputShard::empty(),
                        frontier_occupancy: OutputShard::empty(),
                        frontier_neighbor: OutputShard::empty(),
                        frontier_neighbor_pair: OutputShard::empty(),
                        connection_reachability: OutputShard::from_slice(
                            &mut connection_reachability[snapshot_start
                                * connection_reachability_width
                                ..(snapshot_start + snapshot_count)
                                    * connection_reachability_width],
                        ),
                        frontier_connection_reachability: OutputShard::empty(),
                        inventory_count: inventory_width,
                        room_count: room_width,
                        connection_count: connection_reachability_width,
                        frontier_count: 0,
                        frontier_neighbor_count: self.frontier_neighbor_count,
                        frontier_window_size: self.frontier_window_size,
                    },
                    sparse: FeatureOutputShards {
                        inventory: OutputShard::empty(),
                        room_x: OutputShard::empty(),
                        room_y: OutputShard::empty(),
                        room_placed: OutputShard::empty(),
                        frontier: OutputShard::from_slice(
                            &mut frontier[sparse_row_start * FEATURE_FRONTIER_WIDTH
                                ..(sparse_row_start + worker_sparse_row_count)
                                    * FEATURE_FRONTIER_WIDTH],
                        ),
                        frontier_occupancy: OutputShard::from_slice(
                            &mut frontier_occupancy[sparse_row_start * frontier_occupancy_width
                                ..(sparse_row_start + worker_sparse_row_count)
                                    * frontier_occupancy_width],
                        ),
                        frontier_neighbor: OutputShard::from_slice(
                            &mut frontier_neighbor[sparse_row_start * frontier_neighbor_width
                                ..(sparse_row_start + worker_sparse_row_count)
                                    * frontier_neighbor_width],
                        ),
                        frontier_neighbor_pair: OutputShard::from_slice(
                            &mut frontier_neighbor_pair[sparse_row_start
                                * frontier_neighbor_pair_width
                                ..(sparse_row_start + worker_sparse_row_count)
                                    * frontier_neighbor_pair_width],
                        ),
                        connection_reachability: OutputShard::empty(),
                        frontier_connection_reachability: OutputShard::from_slice(
                            &mut frontier_connection_reachability[sparse_row_start
                                * frontier_connection_width
                                ..(sparse_row_start + worker_sparse_row_count)
                                    * frontier_connection_width],
                        ),
                        inventory_count: 0,
                        room_count: 0,
                        connection_count: frontier_connection_width,
                        frontier_count: 1,
                        frontier_neighbor_count: self.frontier_neighbor_count,
                        frontier_window_size: self.frontier_window_size,
                    },
                    dense_row_idx: OutputShard::from_slice(
                        &mut dense_row_idx
                            [sparse_row_start..sparse_row_start + worker_sparse_row_count],
                    ),
                    snapshot_start,
                    dense_frontier_count: frontier_count,
                };
                if let Err(err) = worker.send(WorkerCommand::PackSparseFeatures { outputs }) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
                sparse_row_start += worker_sparse_row_count;
            }
            collect_feature_info(&self.workers, sent_workers, first_error)
        })?;
        if actual_sparse_row_count != sparse_row_count {
            return Err(PyRuntimeError::new_err(format!(
                "sparse feature row count changed between passes: expected {sparse_row_count}, got {actual_sparse_row_count}"
            )));
        }
        Ok(())
    }
}
