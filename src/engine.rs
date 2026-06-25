/// The `engine` module exposes the map generation environment to Python through the Engine and
/// EnvironmentGroup classes. It handles the creation and management of worker threads that run
/// environment simulations in parallel.
use crate::common::{
    Action, CommonData, Coord, Direction, DoorValidOutcome, DoorVariantIdx, FrontierIdx, Room,
    RoomIdx,
};
use crate::environment::{
    Environment, FEATURE_FRONTIER_WIDTH, FeatureConfig, FeatureScratch, Features,
    FrontierNeighborAlgorithm, StepOutcomes as EnvironmentStepOutcomes,
};
use crossbeam_channel as channel;
use numpy::{Element, IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::cmp::{max, min};
use std::marker::PhantomData;
#[cfg(test)]
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const MISSING_CONNECT_QUERY_FRONTIER_COUNT: usize = 1;

macro_rules! required_py_field {
    ($fields:expr, $name:literal) => {
        $fields
            .get_item($name)?
            .ok_or_else(|| PyValueError::new_err(format!("missing required field {}", $name)))?
            .extract()?
    };
}

macro_rules! profile_metrics {
    ($($variant:ident => $name:literal,)+) => {
        const PROFILE_METRIC_COUNT: usize = <[()]>::len(&[$(profile_metrics!(@unit $variant)),+]);

        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[repr(usize)]
        pub(crate) enum ProfileMetric {
            $($variant,)+
        }

        impl ProfileMetric {
            const ALL: [Self; PROFILE_METRIC_COUNT] = [$(Self::$variant,)+];

            fn idx(self) -> usize {
                self as usize
            }

            fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => $name,)+
                }
            }
        }
    };
    (@unit $variant:ident) => {
        ()
    };
}

profile_metrics! {
    WorkerClear => "worker.clear",
    WorkerFinish => "worker.finish",
    WorkerStepInitial => "worker.step_initial",
    WorkerStep => "worker.step",
    WorkerGetActions => "worker.get_actions",
    WorkerGetOutcomes => "worker.get_outcomes",
    WorkerGetDoorMatchCounts => "worker.get_door_match_counts",
    WorkerGetDoorMatches => "worker.get_door_matches",
    WorkerGetFeatures => "worker.get_features",
    WorkerPackFeatures => "worker.pack_features",
    EnvStepPushAction => "env.step.push_action",
    EnvStepMarkRoomUsed => "env.step.mark_room_used",
    EnvStepComponentsEdges => "env.step.components_edges",
    EnvStepOccupancy => "env.step.occupancy",
    EnvStepMatchExistingFrontiers => "env.step.match_existing_frontiers",
    EnvStepBuildNewFrontierCandidates => "env.step.build_new_frontier_candidates",
    EnvStepFilterExistingFrontiers => "env.step.filter_existing_frontiers",
    WorkerStepKnown => "worker.step_known",
    WorkerGetProposalCandidateMask => "worker.get_proposal_candidate_mask",
    WorkerGetCandidatesFromProposals => "worker.get_candidates_from_proposals",
    EnvProposalPreOutcomes => "env.proposal.pre_outcomes",
    EnvProposalSortFrontiers => "env.proposal.sort_frontiers",
    EnvProposalResolveAction => "env.proposal.resolve_action",
    EnvProposalApplyLookahead => "env.proposal.apply_lookahead",
    EnvProposalDoorOutcomes => "env.proposal.door_outcomes",
    EnvProposalConnectionOutcomes => "env.proposal.connection_outcomes",
    EnvProposalFeatures => "env.proposal.features",
    EnvProposalDoorMatch => "env.proposal.door_match",
    EnvProposalRestore => "env.proposal.restore",
    EnvProposalFallbackRecompute => "env.proposal.fallback_recompute",
    EnvLookaheadSnapshot => "env.lookahead.snapshot",
    EnvLookaheadStep => "env.lookahead.step",
    EnvFeaturesSetup => "env.features.setup",
    EnvFeaturesSortFrontiers => "env.features.sort_frontiers",
    EnvFeaturesConnectionReachability => "env.features.connection_reachability",
    EnvFeaturesSaveRefillUtilityQuery => "env.features.save_refill_utility_query",
    EnvFeaturesFrontierNeighbor => "env.features.frontier_neighbor",
    EnvFeaturesFrontierNeighborFlags => "env.features.frontier_neighbor_flags",
    EnvFeaturesRoomPositionClone => "env.features.room_position_clone",
    EnvFeaturesOutput => "env.features.output",
    EnvFeaturesApplyCandidate => "env.features.apply_candidate",
    EnvFeaturesConnectionReachabilityBase => "env.features.connection_reachability.base",
    EnvFeaturesConnectionReachabilityFrontiers => "env.features.connection_reachability.frontiers",
    EnvFeaturesMissingConnectQueries => "env.features.connection_reachability.missing_connect_queries",
    EnvCounterProposalCalls => "env.counter.proposal.calls",
    EnvCounterProposalShortlistCandidates => "env.counter.proposal.shortlist_candidates",
    EnvCounterProposalEvaluatedCandidates => "env.counter.proposal.evaluated_candidates",
    EnvCounterProposalCleanCandidates => "env.counter.proposal.clean_candidates",
    EnvCounterProposalRejectedCandidates => "env.counter.proposal.rejected_candidates",
    EnvCounterProposalFallbackCandidates => "env.counter.proposal.fallback_candidates",
    EnvCounterProposalOutputCandidates => "env.counter.proposal.output_candidates",
    EnvCounterFeatureCalls => "env.counter.features.calls",
    EnvCounterFeatureFrontiers => "env.counter.features.frontiers",
    EnvCounterFeatureUsedConnections => "env.counter.features.used_connections",
    EnvCounterFeatureConnectionFrontierPairs => "env.counter.features.connection_frontier_pairs",
    EnvCounterFeatureMissingConnectQueryRows => "env.counter.features.missing_connect_query_rows",
    EnvCounterFeatureSaveRefillUtilityRows => "env.counter.features.save_refill_utility_rows",
    EnvCounterFeatureSaveRefillUtilitySaveToMasks => "env.counter.features.save_refill_utility_save_to_masks",
    EnvCounterFeatureSaveRefillUtilitySaveFromMasks => "env.counter.features.save_refill_utility_save_from_masks",
    EnvCounterFeatureSaveRefillUtilityRefillToMasks => "env.counter.features.save_refill_utility_refill_to_masks",
    EnvCounterFeatureSaveRefillUtilityRefillFromMasks => "env.counter.features.save_refill_utility_refill_from_masks",
}

static PROFILE_ENABLED: AtomicBool = AtomicBool::new(false);
static PROFILE_COUNTS: [AtomicU64; PROFILE_METRIC_COUNT] =
    [const { AtomicU64::new(0) }; PROFILE_METRIC_COUNT];
static PROFILE_NANOS: [AtomicU64; PROFILE_METRIC_COUNT] =
    [const { AtomicU64::new(0) }; PROFILE_METRIC_COUNT];

pub(crate) fn profile_enabled() -> bool {
    PROFILE_ENABLED.load(Ordering::Relaxed)
}

pub(crate) fn record_profile_metric(metric: ProfileMetric, duration: Duration) {
    if PROFILE_ENABLED.load(Ordering::Relaxed) {
        let metric_idx = metric.idx();
        PROFILE_COUNTS[metric_idx].fetch_add(1, Ordering::Relaxed);
        PROFILE_NANOS[metric_idx].fetch_add(
            duration.as_nanos().min(u128::from(u64::MAX)) as u64,
            Ordering::Relaxed,
        );
    }
}

pub(crate) fn record_profile_count(metric: ProfileMetric, count: u64) {
    if PROFILE_ENABLED.load(Ordering::Relaxed) {
        PROFILE_COUNTS[metric.idx()].fetch_add(count, Ordering::Relaxed);
    }
}

pub fn set_profile_enabled(enabled: bool) {
    PROFILE_ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn reset_profile() {
    for metric in ProfileMetric::ALL {
        let metric_idx = metric.idx();
        PROFILE_COUNTS[metric_idx].store(0, Ordering::Relaxed);
        PROFILE_NANOS[metric_idx].store(0, Ordering::Relaxed);
    }
}

pub fn profile_report() -> Vec<(String, u64, u64)> {
    ProfileMetric::ALL
        .iter()
        .map(|&metric| {
            let metric_idx = metric.idx();
            (
                metric.name().to_string(),
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
    #[cfg(test)]
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
    StepInitial,
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
    GetProposalCandidateMask {
        proposal_door_variant_count: usize,
        proposal_mask_byte_count: usize,
        proposal_frontier_idx: OutputShard<FrontierIdx>,
        mask: OutputShard<u8>,
        valid_counts: OutputShard<usize>,
    },
    GetCandidatesFromProposals {
        frontier_neighbor_algorithm: FrontierNeighborAlgorithm,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
        recommended_candidates: usize,
        shortlist_candidates: usize,
        sampled_frontier_idx: InputShard<FrontierIdx>,
        sampled_door_variant_idx: InputShard<DoorVariantIdx>,
        room_idx: OutputShard<RoomIdx>,
        room_x: OutputShard<Coord>,
        room_y: OutputShard<Coord>,
        proposal_frontier_idx: OutputShard<FrontierIdx>,
        proposal_door_variant_idx: OutputShard<DoorVariantIdx>,
        door_outcome_count: usize,
        connection_outcome_count: usize,
        pre_door_valid: OutputShard<i8>,
        pre_connections_valid: OutputShard<i8>,
        pre_toilet_valid: OutputShard<i8>,
        door_valid: OutputShard<i8>,
        connections_valid: OutputShard<i8>,
        toilet_valid: OutputShard<i8>,
        door_match: OutputShard<i16>,
        clean_counts: OutputShard<usize>,
        evaluated_counts: OutputShard<usize>,
        rejected_counts: OutputShard<usize>,
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
        toilet_valid: OutputShard<i8>,
        toilet_crossed_room_idx: OutputShard<i16>,
        avg_frontiers: OutputShard<f32>,
        graph_diameter: OutputShard<f32>,
        active_room_part_mask: OutputShard<u8>,
        save_distance: OutputShard<f32>,
        save_distance_mask: OutputShard<u8>,
        save_to_room_distance: OutputShard<f32>,
        save_to_room_distance_mask: OutputShard<u8>,
        save_from_room_distance: OutputShard<f32>,
        save_from_room_distance_mask: OutputShard<u8>,
        refill_distance: OutputShard<f32>,
        refill_distance_mask: OutputShard<u8>,
        refill_to_room_distance: OutputShard<f32>,
        refill_to_room_distance_mask: OutputShard<u8>,
        refill_from_room_distance: OutputShard<f32>,
        refill_from_room_distance_mask: OutputShard<u8>,
        missing_connect_distance: OutputShard<f32>,
        missing_connect_distance_mask: OutputShard<u8>,
    },
    GetCurrentFeatureOutcomes {
        environment_start: usize,
        environment_count: usize,
        door_outcome_count: usize,
        connection_outcome_count: usize,
        door_valid: OutputShard<i8>,
        connections_valid: OutputShard<i8>,
        toilet_valid: OutputShard<i8>,
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
    PackFeatures {
        outputs: FeatureOutputShards,
        expected_snapshot_count: usize,
    },
    Shutdown,
}

impl WorkerCommand {
    fn profile_metric(&self) -> Option<ProfileMetric> {
        match self {
            WorkerCommand::Clear => Some(ProfileMetric::WorkerClear),
            WorkerCommand::Finish => Some(ProfileMetric::WorkerFinish),
            WorkerCommand::StepInitial => Some(ProfileMetric::WorkerStepInitial),
            WorkerCommand::Step { .. } => Some(ProfileMetric::WorkerStep),
            WorkerCommand::GetActions { .. } => Some(ProfileMetric::WorkerGetActions),
            WorkerCommand::GetOutcomes { .. } => Some(ProfileMetric::WorkerGetOutcomes),
            WorkerCommand::GetCurrentFeatureOutcomes { .. } => {
                Some(ProfileMetric::WorkerGetOutcomes)
            }
            WorkerCommand::GetDoorMatchCounts { .. } => {
                Some(ProfileMetric::WorkerGetDoorMatchCounts)
            }
            WorkerCommand::GetDoorMatches { .. } => Some(ProfileMetric::WorkerGetDoorMatches),
            WorkerCommand::GetFeatures { .. } => Some(ProfileMetric::WorkerGetFeatures),
            WorkerCommand::PackFeatures { .. } => Some(ProfileMetric::WorkerPackFeatures),
            WorkerCommand::StepKnown { .. } => Some(ProfileMetric::WorkerStepKnown),
            WorkerCommand::GetProposalCandidateMask { .. } => {
                Some(ProfileMetric::WorkerGetProposalCandidateMask)
            }
            WorkerCommand::GetCandidatesFromProposals { .. } => {
                Some(ProfileMetric::WorkerGetCandidatesFromProposals)
            }
            WorkerCommand::Shutdown => None,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct FeatureInfo {
    frontier_row_count: usize,
    missing_connect_query_row_count: usize,
    save_refill_utility_query_row_count: usize,
}

// Feature preparation reports only metadata needed to allocate output buffers. Bulk data is
// written through shared memory, and other commands return "done" when they finish.
enum WorkerResponse {
    Done,
    Error(String),
    FeatureInfo(FeatureInfo),
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
            WorkerResponse::FeatureInfo(_) => Err(PyRuntimeError::new_err(
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
    let mut feature_scratch = FeatureScratch::default();
    while let Ok(command) = command_rx.recv() {
        let profile_metric = command.profile_metric();
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
            WorkerCommand::StepInitial => {
                for env in &mut environments {
                    let action = env.get_initial_action(&common_data);
                    env.step(action, &common_data);
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
            WorkerCommand::GetProposalCandidateMask {
                proposal_door_variant_count,
                proposal_mask_byte_count,
                proposal_frontier_idx,
                mask,
                valid_counts,
            } => {
                let proposal_frontier_idx = unsafe { proposal_frontier_idx.into_mut_slice() };
                let mask = unsafe { mask.into_mut_slice() };
                let valid_counts = unsafe { valid_counts.into_mut_slice() };
                debug_assert_eq!(proposal_frontier_idx.len(), environments.len());
                debug_assert_eq!(mask.len(), environments.len() * proposal_mask_byte_count);
                debug_assert_eq!(valid_counts.len(), environments.len());

                for (env_idx, env) in environments.iter().enumerate() {
                    let mask_start = env_idx * proposal_mask_byte_count;
                    let mask_end = mask_start + proposal_mask_byte_count;
                    env.proposal_candidate_mask(
                        &common_data,
                        proposal_door_variant_count,
                        &mut proposal_frontier_idx[env_idx],
                        &mut mask[mask_start..mask_end],
                        &mut valid_counts[env_idx],
                    );
                }
                WorkerResponse::Done
            }
            WorkerCommand::GetCandidatesFromProposals {
                frontier_neighbor_algorithm,
                frontier_neighbor_count,
                frontier_window_size,
                recommended_candidates,
                shortlist_candidates,
                sampled_frontier_idx,
                sampled_door_variant_idx,
                room_idx,
                room_x,
                room_y,
                proposal_frontier_idx,
                proposal_door_variant_idx,
                door_outcome_count,
                connection_outcome_count,
                pre_door_valid,
                pre_connections_valid,
                pre_toilet_valid,
                door_valid,
                connections_valid,
                toilet_valid,
                door_match,
                clean_counts,
                evaluated_counts,
                rejected_counts,
            } => {
                let sampled_frontier_idx = unsafe { sampled_frontier_idx.into_slice() };
                let sampled_door_variant_idx = unsafe { sampled_door_variant_idx.into_slice() };
                let room_idx = unsafe { room_idx.into_mut_slice() };
                let room_x = unsafe { room_x.into_mut_slice() };
                let room_y = unsafe { room_y.into_mut_slice() };
                let proposal_frontier_idx = unsafe { proposal_frontier_idx.into_mut_slice() };
                let proposal_door_variant_idx =
                    unsafe { proposal_door_variant_idx.into_mut_slice() };
                let pre_door_valid = unsafe { pre_door_valid.into_mut_slice() };
                let pre_connections_valid = unsafe { pre_connections_valid.into_mut_slice() };
                let pre_toilet_valid = unsafe { pre_toilet_valid.into_mut_slice() };
                let door_valid = unsafe { door_valid.into_mut_slice() };
                let connections_valid = unsafe { connections_valid.into_mut_slice() };
                let toilet_valid = unsafe { toilet_valid.into_mut_slice() };
                let door_match = unsafe { door_match.into_mut_slice() };
                let clean_counts = unsafe { clean_counts.into_mut_slice() };
                let evaluated_counts = unsafe { evaluated_counts.into_mut_slice() };
                let rejected_counts = unsafe { rejected_counts.into_mut_slice() };

                debug_assert_eq!(
                    sampled_frontier_idx.len(),
                    environments.len() * shortlist_candidates
                );
                debug_assert_eq!(
                    sampled_door_variant_idx.len(),
                    environments.len() * shortlist_candidates
                );
                debug_assert_eq!(room_idx.len(), environments.len() * recommended_candidates);
                debug_assert_eq!(room_x.len(), environments.len() * recommended_candidates);
                debug_assert_eq!(room_y.len(), environments.len() * recommended_candidates);
                debug_assert_eq!(
                    proposal_frontier_idx.len(),
                    environments.len() * recommended_candidates
                );
                debug_assert_eq!(
                    proposal_door_variant_idx.len(),
                    environments.len() * recommended_candidates
                );
                debug_assert_eq!(
                    pre_door_valid.len(),
                    environments.len() * door_outcome_count
                );
                debug_assert_eq!(
                    pre_connections_valid.len(),
                    environments.len() * connection_outcome_count
                );
                debug_assert_eq!(pre_toilet_valid.len(), environments.len());
                debug_assert_eq!(
                    door_valid.len(),
                    environments.len() * recommended_candidates * door_outcome_count
                );
                debug_assert_eq!(
                    connections_valid.len(),
                    environments.len() * recommended_candidates * connection_outcome_count
                );
                debug_assert_eq!(
                    toilet_valid.len(),
                    environments.len() * recommended_candidates
                );
                debug_assert_eq!(
                    door_match.len(),
                    environments.len() * recommended_candidates * door_outcome_count
                );
                let mut consistency_error = None;
                feature_scratch.recycle_feature_vec(&mut pending_features);
                for (env_idx, env) in environments.iter_mut().enumerate() {
                    let shortlist_start = env_idx * shortlist_candidates;
                    let shortlist_end = shortlist_start + shortlist_candidates;
                    let (
                        pre_candidate_outcomes,
                        candidates,
                        candidate_frontier_idx,
                        candidate_door_variant_idx,
                        outcomes,
                        door_matches,
                        mut candidate_features,
                        evaluated_count,
                        rejected_count,
                    ) = match env.get_proposal_candidates_with_outcomes(
                        &common_data,
                        &sampled_frontier_idx[shortlist_start..shortlist_end],
                        &sampled_door_variant_idx[shortlist_start..shortlist_end],
                        recommended_candidates,
                        &features,
                        frontier_neighbor_algorithm,
                        frontier_neighbor_count,
                        frontier_window_size,
                        &mut feature_scratch,
                    ) {
                        Ok(result) => result,
                        Err(err) => {
                            consistency_error = Some(err);
                            break;
                        }
                    };
                    clean_counts[env_idx] = candidates.len();
                    evaluated_counts[env_idx] = evaluated_count;
                    rejected_counts[env_idx] = rejected_count;
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
                    pre_toilet_valid[env_idx] = outcome_to_i8(pre_candidate_outcomes.toilet_valid);
                    let row_start = env_idx * recommended_candidates;
                    let dummy_outcome = if candidates.len() < recommended_candidates {
                        Some(EnvironmentStepOutcomes {
                            door_valid: vec![DoorValidOutcome::Unknown; door_outcome_count],
                            connections_valid: vec![
                                DoorValidOutcome::Unknown;
                                connection_outcome_count
                            ],
                            toilet_valid: DoorValidOutcome::Unknown,
                            toilet_crossed_room_idx: -1,
                        })
                    } else {
                        None
                    };
                    let dummy_door_match = if candidates.len() < recommended_candidates {
                        Some(vec![-1; door_outcome_count])
                    } else {
                        None
                    };
                    let dummy_candidate = Action {
                        room_idx: common_data.room.len() as RoomIdx,
                        x: 0,
                        y: 0,
                    };
                    for candidate_idx in 0..recommended_candidates {
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
                            candidate_features.push(env.features_after_candidate_with_scratch(
                                &common_data,
                                dummy_candidate,
                                &features,
                                frontier_neighbor_algorithm,
                                frontier_neighbor_count,
                                frontier_window_size,
                                &mut feature_scratch,
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
                        toilet_valid[idx] = outcome_to_i8(outcome.toilet_valid);
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
                    None => WorkerResponse::FeatureInfo(feature_info(&pending_features)),
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
                toilet_valid,
                toilet_crossed_room_idx,
                avg_frontiers,
                graph_diameter,
                active_room_part_mask,
                save_distance,
                save_distance_mask,
                save_to_room_distance,
                save_to_room_distance_mask,
                save_from_room_distance,
                save_from_room_distance_mask,
                refill_distance,
                refill_distance_mask,
                refill_to_room_distance,
                refill_to_room_distance_mask,
                refill_from_room_distance,
                refill_from_room_distance_mask,
                missing_connect_distance,
                missing_connect_distance_mask,
            } => {
                // SAFETY: The main thread guarantees that for the duration of this command,
                // the output slices remain valid and that no other thread accesses them.
                let door_valid = unsafe { door_valid.into_mut_slice() };
                let connections_valid = unsafe { connections_valid.into_mut_slice() };
                let toilet_valid = unsafe { toilet_valid.into_mut_slice() };
                let toilet_crossed_room_idx = unsafe { toilet_crossed_room_idx.into_mut_slice() };
                let avg_frontiers = unsafe { avg_frontiers.into_mut_slice() };
                let graph_diameter = unsafe { graph_diameter.into_mut_slice() };
                let active_room_part_mask = unsafe { active_room_part_mask.into_mut_slice() };
                let save_distance = unsafe { save_distance.into_mut_slice() };
                let save_distance_mask = unsafe { save_distance_mask.into_mut_slice() };
                let save_to_room_distance = unsafe { save_to_room_distance.into_mut_slice() };
                let save_to_room_distance_mask =
                    unsafe { save_to_room_distance_mask.into_mut_slice() };
                let save_from_room_distance = unsafe { save_from_room_distance.into_mut_slice() };
                let save_from_room_distance_mask =
                    unsafe { save_from_room_distance_mask.into_mut_slice() };
                let refill_distance = unsafe { refill_distance.into_mut_slice() };
                let refill_distance_mask = unsafe { refill_distance_mask.into_mut_slice() };
                let refill_to_room_distance = unsafe { refill_to_room_distance.into_mut_slice() };
                let refill_to_room_distance_mask =
                    unsafe { refill_to_room_distance_mask.into_mut_slice() };
                let refill_from_room_distance =
                    unsafe { refill_from_room_distance.into_mut_slice() };
                let refill_from_room_distance_mask =
                    unsafe { refill_from_room_distance_mask.into_mut_slice() };
                let missing_connect_distance = unsafe { missing_connect_distance.into_mut_slice() };
                let missing_connect_distance_mask =
                    unsafe { missing_connect_distance_mask.into_mut_slice() };
                debug_assert_eq!(door_valid.len(), environments.len() * door_outcome_count);
                debug_assert_eq!(
                    connections_valid.len(),
                    environments.len() * connection_outcome_count
                );
                debug_assert_eq!(toilet_valid.len(), environments.len());
                debug_assert_eq!(toilet_crossed_room_idx.len(), environments.len());
                debug_assert_eq!(avg_frontiers.len(), environments.len());
                debug_assert_eq!(graph_diameter.len(), environments.len());
                debug_assert_eq!(
                    active_room_part_mask.len(),
                    environments.len() * common_data.room_part.len()
                );
                debug_assert_eq!(
                    save_distance.len(),
                    environments.len() * common_data.room_part.len()
                );
                debug_assert_eq!(
                    save_distance_mask.len(),
                    environments.len() * common_data.room_part.len()
                );
                debug_assert_eq!(
                    save_to_room_distance.len(),
                    environments.len() * common_data.room_part.len()
                );
                debug_assert_eq!(
                    save_to_room_distance_mask.len(),
                    environments.len() * common_data.room_part.len()
                );
                debug_assert_eq!(
                    save_from_room_distance.len(),
                    environments.len() * common_data.room_part.len()
                );
                debug_assert_eq!(
                    save_from_room_distance_mask.len(),
                    environments.len() * common_data.room_part.len()
                );
                debug_assert_eq!(
                    refill_distance.len(),
                    environments.len() * common_data.room_part.len()
                );
                debug_assert_eq!(
                    refill_distance_mask.len(),
                    environments.len() * common_data.room_part.len()
                );
                debug_assert_eq!(
                    refill_to_room_distance.len(),
                    environments.len() * common_data.room_part.len()
                );
                debug_assert_eq!(
                    refill_to_room_distance_mask.len(),
                    environments.len() * common_data.room_part.len()
                );
                debug_assert_eq!(
                    refill_from_room_distance.len(),
                    environments.len() * common_data.room_part.len()
                );
                debug_assert_eq!(
                    refill_from_room_distance_mask.len(),
                    environments.len() * common_data.room_part.len()
                );
                debug_assert_eq!(
                    missing_connect_distance.len(),
                    environments.len() * connection_outcome_count
                );
                debug_assert_eq!(
                    missing_connect_distance_mask.len(),
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
                    let avg_frontier_count = match env.avg_frontiers() {
                        Ok(value) => value,
                        Err(err) => {
                            consistency_error = Some(err);
                            break;
                        }
                    };
                    debug_assert_eq!(outcomes.door_valid.len(), door_outcome_count);
                    debug_assert_eq!(outcomes.connections_valid.len(), connection_outcome_count);
                    avg_frontiers[env_idx] = avg_frontier_count;
                    graph_diameter[env_idx] = f32::from(env.graph_diameter());
                    let env_active_room_part_mask = env.active_room_part_mask(&common_data);
                    let (env_save_distance, env_save_distance_mask) =
                        env.save_distances(&common_data);
                    let (
                        env_save_to_room_distance,
                        env_save_to_room_distance_mask,
                        env_save_from_room_distance,
                        env_save_from_room_distance_mask,
                    ) = env.directed_save_distances(&common_data);
                    let save_distance_start = env_idx * common_data.room_part.len();
                    let save_distance_end = save_distance_start + common_data.room_part.len();
                    active_room_part_mask[save_distance_start..save_distance_end]
                        .copy_from_slice(&env_active_room_part_mask);
                    save_distance[save_distance_start..save_distance_end]
                        .copy_from_slice(&env_save_distance);
                    save_distance_mask[save_distance_start..save_distance_end]
                        .copy_from_slice(&env_save_distance_mask);
                    save_to_room_distance[save_distance_start..save_distance_end]
                        .copy_from_slice(&env_save_to_room_distance);
                    save_to_room_distance_mask[save_distance_start..save_distance_end]
                        .copy_from_slice(&env_save_to_room_distance_mask);
                    save_from_room_distance[save_distance_start..save_distance_end]
                        .copy_from_slice(&env_save_from_room_distance);
                    save_from_room_distance_mask[save_distance_start..save_distance_end]
                        .copy_from_slice(&env_save_from_room_distance_mask);
                    let (env_refill_distance, env_refill_distance_mask) =
                        env.refill_distances(&common_data);
                    let (
                        env_refill_to_room_distance,
                        env_refill_to_room_distance_mask,
                        env_refill_from_room_distance,
                        env_refill_from_room_distance_mask,
                    ) = env.directed_refill_distances(&common_data);
                    refill_distance[save_distance_start..save_distance_end]
                        .copy_from_slice(&env_refill_distance);
                    refill_distance_mask[save_distance_start..save_distance_end]
                        .copy_from_slice(&env_refill_distance_mask);
                    refill_to_room_distance[save_distance_start..save_distance_end]
                        .copy_from_slice(&env_refill_to_room_distance);
                    refill_to_room_distance_mask[save_distance_start..save_distance_end]
                        .copy_from_slice(&env_refill_to_room_distance_mask);
                    refill_from_room_distance[save_distance_start..save_distance_end]
                        .copy_from_slice(&env_refill_from_room_distance);
                    refill_from_room_distance_mask[save_distance_start..save_distance_end]
                        .copy_from_slice(&env_refill_from_room_distance_mask);
                    let (env_missing_connect_distance, env_missing_connect_distance_mask) =
                        env.missing_connect_distances(&common_data);
                    let connection_row_start = env_idx * connection_outcome_count;
                    let connection_row_end = connection_row_start + connection_outcome_count;
                    missing_connect_distance[connection_row_start..connection_row_end]
                        .copy_from_slice(&env_missing_connect_distance);
                    missing_connect_distance_mask[connection_row_start..connection_row_end]
                        .copy_from_slice(&env_missing_connect_distance_mask);
                    let door_row_start = env_idx * door_outcome_count;
                    for (outcome_idx, outcome) in outcomes.door_valid.iter().enumerate() {
                        door_valid[door_row_start + outcome_idx] = match outcome {
                            DoorValidOutcome::Unknown => -1,
                            DoorValidOutcome::Valid => 0,
                            DoorValidOutcome::Invalid => 1,
                        };
                    }
                    for (outcome_idx, outcome) in outcomes.connections_valid.iter().enumerate() {
                        connections_valid[connection_row_start + outcome_idx] = match outcome {
                            DoorValidOutcome::Unknown => -1,
                            DoorValidOutcome::Valid => 0,
                            DoorValidOutcome::Invalid => 1,
                        };
                    }
                    toilet_valid[env_idx] = outcome_to_i8(outcomes.toilet_valid);
                    toilet_crossed_room_idx[env_idx] = outcomes.toilet_crossed_room_idx;
                }
                match consistency_error {
                    Some(err) => WorkerResponse::Error(err),
                    None => WorkerResponse::Done,
                }
            }
            WorkerCommand::GetCurrentFeatureOutcomes {
                environment_start,
                environment_count,
                door_outcome_count,
                connection_outcome_count,
                door_valid,
                connections_valid,
                toilet_valid,
                door_match,
            } => {
                // SAFETY: The main thread guarantees that for the duration of this command,
                // the output slices remain valid and that no other thread mutates them.
                let door_valid = unsafe { door_valid.into_mut_slice() };
                let connections_valid = unsafe { connections_valid.into_mut_slice() };
                let toilet_valid = unsafe { toilet_valid.into_mut_slice() };
                let door_match = unsafe { door_match.into_mut_slice() };
                debug_assert_eq!(door_valid.len(), environment_count * door_outcome_count);
                debug_assert_eq!(
                    connections_valid.len(),
                    environment_count * connection_outcome_count
                );
                debug_assert_eq!(toilet_valid.len(), environment_count);
                debug_assert_eq!(door_match.len(), environment_count * door_outcome_count);

                for (env_idx, env) in environments
                    .iter()
                    .skip(environment_start)
                    .take(environment_count)
                    .enumerate()
                {
                    let feature_outcomes = env.feature_outcomes(&common_data);
                    let outcomes = feature_outcomes.step_outcomes;
                    debug_assert_eq!(outcomes.door_valid.len(), door_outcome_count);
                    debug_assert_eq!(outcomes.connections_valid.len(), connection_outcome_count);
                    debug_assert_eq!(feature_outcomes.door_match.len(), door_outcome_count);
                    let door_start = env_idx * door_outcome_count;
                    for (outcome_idx, outcome) in outcomes.door_valid.iter().enumerate() {
                        door_valid[door_start + outcome_idx] = match outcome {
                            DoorValidOutcome::Unknown => -1,
                            DoorValidOutcome::Valid => 0,
                            DoorValidOutcome::Invalid => 1,
                        };
                    }
                    for (outcome_idx, &value) in feature_outcomes.door_match.iter().enumerate() {
                        door_match[door_start + outcome_idx] = value;
                    }
                    let connection_start = env_idx * connection_outcome_count;
                    for (outcome_idx, outcome) in outcomes.connections_valid.iter().enumerate() {
                        connections_valid[connection_start + outcome_idx] = match outcome {
                            DoorValidOutcome::Unknown => -1,
                            DoorValidOutcome::Valid => 0,
                            DoorValidOutcome::Invalid => 1,
                        };
                    }
                    toilet_valid[env_idx] = outcome_to_i8(outcomes.toilet_valid);
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
                feature_scratch.recycle_feature_vec(&mut pending_features);
                for env in environments
                    .iter()
                    .skip(environment_start)
                    .take(environment_count)
                {
                    pending_features.push(env.features_with_scratch(
                        &common_data,
                        &features,
                        frontier_neighbor_algorithm,
                        frontier_neighbor_count,
                        frontier_window_size,
                        &mut feature_scratch,
                    ));
                }
                WorkerResponse::FeatureInfo(feature_info(&pending_features))
            }
            WorkerCommand::PackFeatures {
                outputs,
                expected_snapshot_count,
            } => {
                if pending_features.len() != expected_snapshot_count {
                    let actual = pending_features.len();
                    feature_scratch.recycle_feature_vec(&mut pending_features);
                    WorkerResponse::Error(format!(
                        "pending feature count mismatch: expected {expected_snapshot_count}, got {actual}"
                    ))
                } else {
                    let mut outputs = unsafe { outputs.into_slices() };
                    for (idx, features) in pending_features.drain(..).enumerate() {
                        outputs.write_features(idx, &features);
                        feature_scratch.recycle_features(features);
                    }
                    WorkerResponse::FeatureInfo(FeatureInfo {
                        frontier_row_count: outputs.frontier_row_count,
                        missing_connect_query_row_count: outputs.missing_connect_query_row_count,
                        save_refill_utility_query_row_count: outputs
                            .save_refill_utility_query_row_count,
                    })
                }
            }
            WorkerCommand::Shutdown => break,
        };
        if let (Some(metric), Some(start)) = (profile_metric, profile_start) {
            record_profile_metric(metric, start.elapsed());
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
) -> PyResult<(FeatureInfo, Vec<FeatureInfo>)> {
    let mut total = FeatureInfo::default();
    let mut worker_frontier_row_counts = vec![0; workers.len()];
    let mut worker_query_row_counts = vec![0; workers.len()];
    let mut worker_save_refill_utility_query_row_counts = vec![0; workers.len()];
    for worker_idx in sent_workers {
        match workers[worker_idx].recv() {
            Ok(WorkerResponse::FeatureInfo(worker_feature_info)) => {
                total.frontier_row_count += worker_feature_info.frontier_row_count;
                total.missing_connect_query_row_count +=
                    worker_feature_info.missing_connect_query_row_count;
                total.save_refill_utility_query_row_count +=
                    worker_feature_info.save_refill_utility_query_row_count;
                worker_frontier_row_counts[worker_idx] = worker_feature_info.frontier_row_count;
                worker_query_row_counts[worker_idx] =
                    worker_feature_info.missing_connect_query_row_count;
                worker_save_refill_utility_query_row_counts[worker_idx] =
                    worker_feature_info.save_refill_utility_query_row_count;
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
        let worker_info = worker_frontier_row_counts
            .into_iter()
            .zip(worker_query_row_counts)
            .zip(worker_save_refill_utility_query_row_counts)
            .map(
                |(
                    (frontier_row_count, missing_connect_query_row_count),
                    save_refill_utility_query_row_count,
                )| FeatureInfo {
                    frontier_row_count,
                    missing_connect_query_row_count,
                    save_refill_utility_query_row_count,
                },
            )
            .collect();
        Ok((total, worker_info))
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

fn check_shape(name: &str, actual: &[usize], expected: &[usize]) -> PyResult<()> {
    if actual != expected {
        Err(PyValueError::new_err(format!(
            "{name} has wrong shape: expected {expected:?}, got {actual:?}"
        )))
    } else {
        Ok(())
    }
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
pub struct ProposalCandidateMask {
    proposal_frontier_idx: Py<PyArray1<FrontierIdx>>,
    mask: Py<PyArray2<u8>>,
    valid_counts: Py<PyArray1<usize>>,
    #[pyo3(get)]
    door_variant_count: usize,
}

#[pyclass(module = "map_gen")]
pub struct StepOutcomes {
    door_valid: Py<PyArray2<i8>>,
    connections_valid: Py<PyArray2<i8>>,
    toilet_valid: Py<PyArray1<i8>>,
}

#[pyclass(module = "map_gen")]
pub struct EndOutcomes {
    toilet_crossed_room_idx: Py<PyArray1<i16>>,
    avg_frontiers: Py<PyArray1<f32>>,
    graph_diameter: Py<PyArray1<f32>>,
    active_room_part_mask: Py<PyArray2<u8>>,
    save_distance: Py<PyArray2<f32>>,
    save_distance_mask: Py<PyArray2<u8>>,
    save_to_room_distance: Py<PyArray2<f32>>,
    save_to_room_distance_mask: Py<PyArray2<u8>>,
    save_from_room_distance: Py<PyArray2<f32>>,
    save_from_room_distance_mask: Py<PyArray2<u8>>,
    refill_distance: Py<PyArray2<f32>>,
    refill_distance_mask: Py<PyArray2<u8>>,
    refill_to_room_distance: Py<PyArray2<f32>>,
    refill_to_room_distance_mask: Py<PyArray2<u8>>,
    refill_from_room_distance: Py<PyArray2<f32>>,
    refill_from_room_distance_mask: Py<PyArray2<u8>>,
    missing_connect_distance: Py<PyArray2<f32>>,
    missing_connect_distance_mask: Py<PyArray2<u8>>,
}

#[pyclass(module = "map_gen")]
pub struct EpisodeOutcomes {
    step_outcomes: StepOutcomes,
    end_outcomes: EndOutcomes,
}

#[pyclass(module = "map_gen")]
pub struct FeatureRequirements {
    #[pyo3(get)]
    frontier_row_count: usize,
    #[pyo3(get)]
    worker_frontier_row_counts: Vec<usize>,
    #[pyo3(get)]
    missing_connect_query_row_count: usize,
    #[pyo3(get)]
    worker_missing_connect_query_row_counts: Vec<usize>,
    #[pyo3(get)]
    save_refill_utility_query_row_count: usize,
    #[pyo3(get)]
    worker_save_refill_utility_query_row_counts: Vec<usize>,
}

#[pyclass(module = "map_gen")]
pub struct ProposalCandidateBuffers {
    sampled_frontier_idx: Py<PyArray2<FrontierIdx>>,
    sampled_door_variant_idx: Py<PyArray2<DoorVariantIdx>>,
    #[pyo3(get)]
    recommended_candidates: usize,
    room_idx: Py<PyArray2<RoomIdx>>,
    room_x: Py<PyArray2<Coord>>,
    room_y: Py<PyArray2<Coord>>,
    proposal_frontier_idx: Py<PyArray2<FrontierIdx>>,
    proposal_door_variant_idx: Py<PyArray2<DoorVariantIdx>>,
    pre_door_valid: Py<PyArray2<i8>>,
    pre_connections_valid: Py<PyArray2<i8>>,
    pre_toilet_valid: Py<PyArray1<i8>>,
    door_valid: Py<PyArray3<i8>>,
    connections_valid: Py<PyArray3<i8>>,
    toilet_valid: Py<PyArray2<i8>>,
    door_match: Py<PyArray3<i16>>,
    clean_counts: Py<PyArray1<i64>>,
    evaluated_counts: Py<PyArray1<i64>>,
    rejected_counts: Py<PyArray1<i64>>,
}

#[pyclass(module = "map_gen")]
pub struct FeatureBuffers {
    #[pyo3(get)]
    environment_count: usize,
    #[pyo3(get)]
    candidate_count: usize,
    #[pyo3(get)]
    environment_start: usize,
    #[pyo3(get)]
    frontier_row_count: usize,
    #[pyo3(get)]
    worker_frontier_row_counts: Vec<usize>,
    #[pyo3(get)]
    missing_connect_query_row_count: usize,
    #[pyo3(get)]
    worker_missing_connect_query_row_counts: Vec<usize>,
    #[pyo3(get)]
    save_refill_utility_query_row_count: usize,
    #[pyo3(get)]
    worker_save_refill_utility_query_row_counts: Vec<usize>,
    inventory: Py<PyArray2<u8>>,
    out_room_x: Py<PyArray2<Coord>>,
    out_room_y: Py<PyArray2<Coord>>,
    room_placed: Py<PyArray2<u8>>,
    room_part_furthest_destination: Py<PyArray2<u8>>,
    room_part_furthest_source: Py<PyArray2<u8>>,
    room_part_save_from_room_distance: Py<PyArray2<u8>>,
    room_part_save_to_room_distance: Py<PyArray2<u8>>,
    room_part_refill_from_room_distance: Py<PyArray2<u8>>,
    room_part_refill_to_room_distance: Py<PyArray2<u8>>,
    room_part_frontier_from_room_distance: Py<PyArray2<u8>>,
    room_part_frontier_to_room_distance: Py<PyArray2<u8>>,
    known_save_from_room_distance: Py<PyArray2<u8>>,
    known_save_to_room_distance: Py<PyArray2<u8>>,
    known_refill_from_room_distance: Py<PyArray2<u8>>,
    known_refill_to_room_distance: Py<PyArray2<u8>>,
    frontier: Py<PyArray2<i8>>,
    frontier_door_variant: Py<PyArray1<DoorVariantIdx>>,
    frontier_occupancy: Py<PyArray2<u8>>,
    frontier_neighbor: Py<PyArray2<i16>>,
    frontier_neighbor_pair: Py<PyArray2<u8>>,
    connection_reachability: Py<PyArray2<u8>>,
    frontier_connection_reachability: Py<PyArray2<u8>>,
    missing_connect_query_snapshot_idx: Py<PyArray1<i64>>,
    missing_connect_query_connection_idx: Py<PyArray1<i64>>,
    missing_connect_query_source_frontier: Py<PyArray2<i16>>,
    missing_connect_query_target_frontier: Py<PyArray2<i16>>,
    missing_connect_query_source_distance: Py<PyArray2<u8>>,
    missing_connect_query_target_distance: Py<PyArray2<u8>>,
    missing_connect_query_current_distance: Py<PyArray1<u8>>,
    save_refill_utility_query_snapshot_idx: Py<PyArray1<i64>>,
    save_refill_utility_query_room_part_idx: Py<PyArray1<i64>>,
    save_refill_utility_query_target_mask: Py<PyArray1<u8>>,
    save_refill_utility_query_frontier: Py<PyArray1<i16>>,
    save_refill_utility_query_frontier_distance: Py<PyArray1<u8>>,
    save_refill_utility_query_save_to_current_distance: Py<PyArray1<u8>>,
    save_refill_utility_query_save_from_current_distance: Py<PyArray1<u8>>,
    save_refill_utility_query_refill_to_current_distance: Py<PyArray1<u8>>,
    save_refill_utility_query_refill_from_current_distance: Py<PyArray1<u8>>,
    toilet_crossed_room_idx: Py<PyArray2<i16>>,
    row_snapshot_idx: Py<PyArray1<i64>>,
    row_frontier_idx: Py<PyArray1<FrontierIdx>>,
    row_door_output_idx: Py<PyArray1<i16>>,
}

#[pymethods]
impl ProposalCandidateBuffers {
    #[new]
    fn new(fields: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self {
            sampled_frontier_idx: required_py_field!(fields, "sampled_frontier_idx"),
            sampled_door_variant_idx: required_py_field!(fields, "sampled_door_variant_idx"),
            recommended_candidates: required_py_field!(fields, "recommended_candidates"),
            room_idx: required_py_field!(fields, "room_idx"),
            room_x: required_py_field!(fields, "room_x"),
            room_y: required_py_field!(fields, "room_y"),
            proposal_frontier_idx: required_py_field!(fields, "proposal_frontier_idx"),
            proposal_door_variant_idx: required_py_field!(fields, "proposal_door_variant_idx"),
            pre_door_valid: required_py_field!(fields, "pre_door_valid"),
            pre_connections_valid: required_py_field!(fields, "pre_connections_valid"),
            pre_toilet_valid: required_py_field!(fields, "pre_toilet_valid"),
            door_valid: required_py_field!(fields, "door_valid"),
            connections_valid: required_py_field!(fields, "connections_valid"),
            toilet_valid: required_py_field!(fields, "toilet_valid"),
            door_match: required_py_field!(fields, "door_match"),
            clean_counts: required_py_field!(fields, "clean_counts"),
            evaluated_counts: required_py_field!(fields, "evaluated_counts"),
            rejected_counts: required_py_field!(fields, "rejected_counts"),
        })
    }
}

#[pymethods]
impl FeatureBuffers {
    #[new]
    fn new(fields: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self {
            environment_count: required_py_field!(fields, "environment_count"),
            candidate_count: required_py_field!(fields, "candidate_count"),
            environment_start: required_py_field!(fields, "environment_start"),
            frontier_row_count: required_py_field!(fields, "frontier_row_count"),
            worker_frontier_row_counts: required_py_field!(fields, "worker_frontier_row_counts"),
            missing_connect_query_row_count: required_py_field!(
                fields,
                "missing_connect_query_row_count"
            ),
            worker_missing_connect_query_row_counts: required_py_field!(
                fields,
                "worker_missing_connect_query_row_counts"
            ),
            save_refill_utility_query_row_count: required_py_field!(
                fields,
                "save_refill_utility_query_row_count"
            ),
            worker_save_refill_utility_query_row_counts: required_py_field!(
                fields,
                "worker_save_refill_utility_query_row_counts"
            ),
            inventory: required_py_field!(fields, "inventory"),
            out_room_x: required_py_field!(fields, "room_x"),
            out_room_y: required_py_field!(fields, "room_y"),
            room_placed: required_py_field!(fields, "room_placed"),
            room_part_furthest_destination: required_py_field!(
                fields,
                "room_part_furthest_destination"
            ),
            room_part_furthest_source: required_py_field!(fields, "room_part_furthest_source"),
            room_part_save_from_room_distance: required_py_field!(
                fields,
                "room_part_save_from_room_distance"
            ),
            room_part_save_to_room_distance: required_py_field!(
                fields,
                "room_part_save_to_room_distance"
            ),
            room_part_refill_from_room_distance: required_py_field!(
                fields,
                "room_part_refill_from_room_distance"
            ),
            room_part_refill_to_room_distance: required_py_field!(
                fields,
                "room_part_refill_to_room_distance"
            ),
            room_part_frontier_from_room_distance: required_py_field!(
                fields,
                "room_part_frontier_from_room_distance"
            ),
            room_part_frontier_to_room_distance: required_py_field!(
                fields,
                "room_part_frontier_to_room_distance"
            ),
            known_save_from_room_distance: required_py_field!(
                fields,
                "known_save_from_room_distance"
            ),
            known_save_to_room_distance: required_py_field!(fields, "known_save_to_room_distance"),
            known_refill_from_room_distance: required_py_field!(
                fields,
                "known_refill_from_room_distance"
            ),
            known_refill_to_room_distance: required_py_field!(
                fields,
                "known_refill_to_room_distance"
            ),
            frontier: required_py_field!(fields, "frontier"),
            frontier_door_variant: required_py_field!(fields, "frontier_door_variant"),
            frontier_occupancy: required_py_field!(fields, "frontier_occupancy"),
            frontier_neighbor: required_py_field!(fields, "frontier_neighbor"),
            frontier_neighbor_pair: required_py_field!(fields, "frontier_neighbor_pair"),
            connection_reachability: required_py_field!(fields, "connection_reachability"),
            frontier_connection_reachability: required_py_field!(
                fields,
                "frontier_connection_reachability"
            ),
            missing_connect_query_snapshot_idx: required_py_field!(
                fields,
                "missing_connect_query_snapshot_idx"
            ),
            missing_connect_query_connection_idx: required_py_field!(
                fields,
                "missing_connect_query_connection_idx"
            ),
            missing_connect_query_source_frontier: required_py_field!(
                fields,
                "missing_connect_query_source_frontier"
            ),
            missing_connect_query_target_frontier: required_py_field!(
                fields,
                "missing_connect_query_target_frontier"
            ),
            missing_connect_query_source_distance: required_py_field!(
                fields,
                "missing_connect_query_source_distance"
            ),
            missing_connect_query_target_distance: required_py_field!(
                fields,
                "missing_connect_query_target_distance"
            ),
            missing_connect_query_current_distance: required_py_field!(
                fields,
                "missing_connect_query_current_distance"
            ),
            save_refill_utility_query_snapshot_idx: required_py_field!(
                fields,
                "save_refill_utility_query_snapshot_idx"
            ),
            save_refill_utility_query_room_part_idx: required_py_field!(
                fields,
                "save_refill_utility_query_room_part_idx"
            ),
            save_refill_utility_query_target_mask: required_py_field!(
                fields,
                "save_refill_utility_query_target_mask"
            ),
            save_refill_utility_query_frontier: required_py_field!(
                fields,
                "save_refill_utility_query_frontier"
            ),
            save_refill_utility_query_frontier_distance: required_py_field!(
                fields,
                "save_refill_utility_query_frontier_distance"
            ),
            save_refill_utility_query_save_to_current_distance: required_py_field!(
                fields,
                "save_refill_utility_query_save_to_current_distance"
            ),
            save_refill_utility_query_save_from_current_distance: required_py_field!(
                fields,
                "save_refill_utility_query_save_from_current_distance"
            ),
            save_refill_utility_query_refill_to_current_distance: required_py_field!(
                fields,
                "save_refill_utility_query_refill_to_current_distance"
            ),
            save_refill_utility_query_refill_from_current_distance: required_py_field!(
                fields,
                "save_refill_utility_query_refill_from_current_distance"
            ),
            toilet_crossed_room_idx: required_py_field!(fields, "toilet_crossed_room_idx"),
            row_snapshot_idx: required_py_field!(fields, "row_snapshot_idx"),
            row_frontier_idx: required_py_field!(fields, "row_frontier_idx"),
            row_door_output_idx: required_py_field!(fields, "row_door_output_idx"),
        })
    }
}

#[pymethods]
impl StepOutcomes {
    #[getter]
    fn door_valid(&self, py: Python<'_>) -> Py<PyArray2<i8>> {
        self.door_valid.clone_ref(py)
    }

    #[getter]
    fn connections_valid(&self, py: Python<'_>) -> Py<PyArray2<i8>> {
        self.connections_valid.clone_ref(py)
    }

    #[getter]
    fn toilet_valid(&self, py: Python<'_>) -> Py<PyArray1<i8>> {
        self.toilet_valid.clone_ref(py)
    }
}

#[pymethods]
impl EndOutcomes {
    #[getter]
    fn toilet_crossed_room_idx(&self, py: Python<'_>) -> Py<PyArray1<i16>> {
        self.toilet_crossed_room_idx.clone_ref(py)
    }

    #[getter]
    fn avg_frontiers(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        self.avg_frontiers.clone_ref(py)
    }

    #[getter]
    fn graph_diameter(&self, py: Python<'_>) -> Py<PyArray1<f32>> {
        self.graph_diameter.clone_ref(py)
    }

    #[getter]
    fn active_room_part_mask(&self, py: Python<'_>) -> Py<PyArray2<u8>> {
        self.active_room_part_mask.clone_ref(py)
    }

    #[getter]
    fn save_distance(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.save_distance.clone_ref(py)
    }

    #[getter]
    fn save_distance_mask(&self, py: Python<'_>) -> Py<PyArray2<u8>> {
        self.save_distance_mask.clone_ref(py)
    }

    #[getter]
    fn save_to_room_distance(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.save_to_room_distance.clone_ref(py)
    }

    #[getter]
    fn save_to_room_distance_mask(&self, py: Python<'_>) -> Py<PyArray2<u8>> {
        self.save_to_room_distance_mask.clone_ref(py)
    }

    #[getter]
    fn save_from_room_distance(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.save_from_room_distance.clone_ref(py)
    }

    #[getter]
    fn save_from_room_distance_mask(&self, py: Python<'_>) -> Py<PyArray2<u8>> {
        self.save_from_room_distance_mask.clone_ref(py)
    }

    #[getter]
    fn refill_distance(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.refill_distance.clone_ref(py)
    }

    #[getter]
    fn refill_distance_mask(&self, py: Python<'_>) -> Py<PyArray2<u8>> {
        self.refill_distance_mask.clone_ref(py)
    }

    #[getter]
    fn refill_to_room_distance(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.refill_to_room_distance.clone_ref(py)
    }

    #[getter]
    fn refill_to_room_distance_mask(&self, py: Python<'_>) -> Py<PyArray2<u8>> {
        self.refill_to_room_distance_mask.clone_ref(py)
    }

    #[getter]
    fn refill_from_room_distance(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.refill_from_room_distance.clone_ref(py)
    }

    #[getter]
    fn refill_from_room_distance_mask(&self, py: Python<'_>) -> Py<PyArray2<u8>> {
        self.refill_from_room_distance_mask.clone_ref(py)
    }

    #[getter]
    fn missing_connect_distance(&self, py: Python<'_>) -> Py<PyArray2<f32>> {
        self.missing_connect_distance.clone_ref(py)
    }

    #[getter]
    fn missing_connect_distance_mask(&self, py: Python<'_>) -> Py<PyArray2<u8>> {
        self.missing_connect_distance_mask.clone_ref(py)
    }
}

#[pymethods]
impl EpisodeOutcomes {
    #[getter]
    fn step_outcomes(&self, py: Python<'_>) -> StepOutcomes {
        StepOutcomes {
            door_valid: self.step_outcomes.door_valid.clone_ref(py),
            connections_valid: self.step_outcomes.connections_valid.clone_ref(py),
            toilet_valid: self.step_outcomes.toilet_valid.clone_ref(py),
        }
    }

    #[getter]
    fn end_outcomes(&self, py: Python<'_>) -> EndOutcomes {
        EndOutcomes {
            toilet_crossed_room_idx: self.end_outcomes.toilet_crossed_room_idx.clone_ref(py),
            avg_frontiers: self.end_outcomes.avg_frontiers.clone_ref(py),
            graph_diameter: self.end_outcomes.graph_diameter.clone_ref(py),
            active_room_part_mask: self.end_outcomes.active_room_part_mask.clone_ref(py),
            save_distance: self.end_outcomes.save_distance.clone_ref(py),
            save_distance_mask: self.end_outcomes.save_distance_mask.clone_ref(py),
            save_to_room_distance: self.end_outcomes.save_to_room_distance.clone_ref(py),
            save_to_room_distance_mask: self.end_outcomes.save_to_room_distance_mask.clone_ref(py),
            save_from_room_distance: self.end_outcomes.save_from_room_distance.clone_ref(py),
            save_from_room_distance_mask: self
                .end_outcomes
                .save_from_room_distance_mask
                .clone_ref(py),
            refill_distance: self.end_outcomes.refill_distance.clone_ref(py),
            refill_distance_mask: self.end_outcomes.refill_distance_mask.clone_ref(py),
            refill_to_room_distance: self.end_outcomes.refill_to_room_distance.clone_ref(py),
            refill_to_room_distance_mask: self
                .end_outcomes
                .refill_to_room_distance_mask
                .clone_ref(py),
            refill_from_room_distance: self.end_outcomes.refill_from_room_distance.clone_ref(py),
            refill_from_room_distance_mask: self
                .end_outcomes
                .refill_from_room_distance_mask
                .clone_ref(py),
            missing_connect_distance: self.end_outcomes.missing_connect_distance.clone_ref(py),
            missing_connect_distance_mask: self
                .end_outcomes
                .missing_connect_distance_mask
                .clone_ref(py),
        }
    }
}

#[pymethods]
impl ProposalCandidateMask {
    #[getter]
    fn proposal_frontier_idx(&self, py: Python<'_>) -> Py<PyArray1<FrontierIdx>> {
        self.proposal_frontier_idx.clone_ref(py)
    }

    #[getter]
    fn mask(&self, py: Python<'_>) -> Py<PyArray2<u8>> {
        self.mask.clone_ref(py)
    }

    #[getter]
    fn valid_counts(&self, py: Python<'_>) -> Py<PyArray1<usize>> {
        self.valid_counts.clone_ref(py)
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

fn outcome_to_i8(outcome: DoorValidOutcome) -> i8 {
    match outcome {
        DoorValidOutcome::Unknown => -1,
        DoorValidOutcome::Valid => 0,
        DoorValidOutcome::Invalid => 1,
    }
}

fn feature_info(features: &[Features]) -> FeatureInfo {
    FeatureInfo {
        frontier_row_count: features
            .iter()
            .map(|features| features.frontier.len() / FEATURE_FRONTIER_WIDTH)
            .sum(),
        missing_connect_query_row_count: features
            .iter()
            .map(|features| features.missing_connect_query_connection_idx.len())
            .sum(),
        save_refill_utility_query_row_count: features
            .iter()
            .map(|features| features.save_refill_utility_query_room_part_idx.len())
            .sum(),
    }
}

struct GlobalFeatureOutputShards {
    inventory: OutputShard<u8>,
    room_x: OutputShard<Coord>,
    room_y: OutputShard<Coord>,
    room_placed: OutputShard<u8>,
    room_part_furthest_destination: OutputShard<u8>,
    room_part_furthest_source: OutputShard<u8>,
    room_part_save_from_room_distance: OutputShard<u8>,
    room_part_save_to_room_distance: OutputShard<u8>,
    room_part_refill_from_room_distance: OutputShard<u8>,
    room_part_refill_to_room_distance: OutputShard<u8>,
    room_part_frontier_from_room_distance: OutputShard<u8>,
    room_part_frontier_to_room_distance: OutputShard<u8>,
    known_save_from_room_distance: OutputShard<u8>,
    known_save_to_room_distance: OutputShard<u8>,
    known_refill_from_room_distance: OutputShard<u8>,
    known_refill_to_room_distance: OutputShard<u8>,
    connection_reachability: OutputShard<u8>,
    toilet_crossed_room_idx: OutputShard<i16>,
    inventory_count: usize,
    room_count: usize,
    room_part_furthest_count: usize,
    room_part_save_distance_count: usize,
    room_part_refill_distance_count: usize,
    room_part_frontier_distance_count: usize,
    known_distance_count: usize,
    connection_count: usize,
    toilet_crossed_room_count: usize,
}

struct FrontierFeatureOutputShards {
    frontier: OutputShard<i8>,
    frontier_door_variant: OutputShard<DoorVariantIdx>,
    frontier_occupancy: OutputShard<u8>,
    frontier_neighbor: OutputShard<i16>,
    frontier_neighbor_pair: OutputShard<u8>,
    frontier_connection_reachability: OutputShard<u8>,
    frontier_neighbor_count: usize,
    connection_count: usize,
    frontier_window_size: usize,
}

struct GlobalFeatureOutputSlices<'a> {
    inventory: &'a mut [u8],
    room_x: &'a mut [Coord],
    room_y: &'a mut [Coord],
    room_placed: &'a mut [u8],
    room_part_furthest_destination: &'a mut [u8],
    room_part_furthest_source: &'a mut [u8],
    room_part_save_from_room_distance: &'a mut [u8],
    room_part_save_to_room_distance: &'a mut [u8],
    room_part_refill_from_room_distance: &'a mut [u8],
    room_part_refill_to_room_distance: &'a mut [u8],
    room_part_frontier_from_room_distance: &'a mut [u8],
    room_part_frontier_to_room_distance: &'a mut [u8],
    known_save_from_room_distance: &'a mut [u8],
    known_save_to_room_distance: &'a mut [u8],
    known_refill_from_room_distance: &'a mut [u8],
    known_refill_to_room_distance: &'a mut [u8],
    connection_reachability: &'a mut [u8],
    toilet_crossed_room_idx: &'a mut [i16],
    inventory_count: usize,
    room_count: usize,
    room_part_furthest_count: usize,
    room_part_save_distance_count: usize,
    room_part_refill_distance_count: usize,
    room_part_frontier_distance_count: usize,
    known_distance_count: usize,
    connection_count: usize,
    toilet_crossed_room_count: usize,
}

struct FrontierFeatureOutputSlices<'a> {
    frontier: &'a mut [i8],
    frontier_door_variant: &'a mut [DoorVariantIdx],
    frontier_occupancy: &'a mut [u8],
    frontier_neighbor: &'a mut [i16],
    frontier_neighbor_pair: &'a mut [u8],
    frontier_connection_reachability: &'a mut [u8],
    frontier_neighbor_count: usize,
    connection_count: usize,
    frontier_window_size: usize,
}

impl GlobalFeatureOutputShards {
    unsafe fn into_slices<'a>(self) -> GlobalFeatureOutputSlices<'a> {
        GlobalFeatureOutputSlices {
            inventory: unsafe { self.inventory.into_mut_slice() },
            room_x: unsafe { self.room_x.into_mut_slice() },
            room_y: unsafe { self.room_y.into_mut_slice() },
            room_placed: unsafe { self.room_placed.into_mut_slice() },
            room_part_furthest_destination: unsafe {
                self.room_part_furthest_destination.into_mut_slice()
            },
            room_part_furthest_source: unsafe { self.room_part_furthest_source.into_mut_slice() },
            room_part_save_from_room_distance: unsafe {
                self.room_part_save_from_room_distance.into_mut_slice()
            },
            room_part_save_to_room_distance: unsafe {
                self.room_part_save_to_room_distance.into_mut_slice()
            },
            room_part_refill_from_room_distance: unsafe {
                self.room_part_refill_from_room_distance.into_mut_slice()
            },
            room_part_refill_to_room_distance: unsafe {
                self.room_part_refill_to_room_distance.into_mut_slice()
            },
            room_part_frontier_from_room_distance: unsafe {
                self.room_part_frontier_from_room_distance.into_mut_slice()
            },
            room_part_frontier_to_room_distance: unsafe {
                self.room_part_frontier_to_room_distance.into_mut_slice()
            },
            known_save_from_room_distance: unsafe {
                self.known_save_from_room_distance.into_mut_slice()
            },
            known_save_to_room_distance: unsafe {
                self.known_save_to_room_distance.into_mut_slice()
            },
            known_refill_from_room_distance: unsafe {
                self.known_refill_from_room_distance.into_mut_slice()
            },
            known_refill_to_room_distance: unsafe {
                self.known_refill_to_room_distance.into_mut_slice()
            },
            connection_reachability: unsafe { self.connection_reachability.into_mut_slice() },
            toilet_crossed_room_idx: unsafe { self.toilet_crossed_room_idx.into_mut_slice() },
            inventory_count: self.inventory_count,
            room_count: self.room_count,
            room_part_furthest_count: self.room_part_furthest_count,
            room_part_save_distance_count: self.room_part_save_distance_count,
            room_part_refill_distance_count: self.room_part_refill_distance_count,
            room_part_frontier_distance_count: self.room_part_frontier_distance_count,
            known_distance_count: self.known_distance_count,
            connection_count: self.connection_count,
            toilet_crossed_room_count: self.toilet_crossed_room_count,
        }
    }
}

impl FrontierFeatureOutputShards {
    unsafe fn into_slices<'a>(self) -> FrontierFeatureOutputSlices<'a> {
        FrontierFeatureOutputSlices {
            frontier: unsafe { self.frontier.into_mut_slice() },
            frontier_door_variant: unsafe { self.frontier_door_variant.into_mut_slice() },
            frontier_occupancy: unsafe { self.frontier_occupancy.into_mut_slice() },
            frontier_neighbor: unsafe { self.frontier_neighbor.into_mut_slice() },
            frontier_neighbor_pair: unsafe { self.frontier_neighbor_pair.into_mut_slice() },
            frontier_connection_reachability: unsafe {
                self.frontier_connection_reachability.into_mut_slice()
            },
            frontier_neighbor_count: self.frontier_neighbor_count,
            connection_count: self.connection_count,
            frontier_window_size: self.frontier_window_size,
        }
    }
}

impl GlobalFeatureOutputSlices<'_> {
    fn write_features(&mut self, idx: usize, features: &Features) {
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
            &mut self.room_part_furthest_destination,
            &features.room_part_furthest_destination,
            idx,
            self.room_part_furthest_count,
        );
        copy_row(
            &mut self.room_part_furthest_source,
            &features.room_part_furthest_source,
            idx,
            self.room_part_furthest_count,
        );
        copy_row(
            &mut self.room_part_save_from_room_distance,
            &features.room_part_save_from_room_distance,
            idx,
            self.room_part_save_distance_count,
        );
        copy_row(
            &mut self.room_part_save_to_room_distance,
            &features.room_part_save_to_room_distance,
            idx,
            self.room_part_save_distance_count,
        );
        copy_row(
            &mut self.room_part_refill_from_room_distance,
            &features.room_part_refill_from_room_distance,
            idx,
            self.room_part_refill_distance_count,
        );
        copy_row(
            &mut self.room_part_refill_to_room_distance,
            &features.room_part_refill_to_room_distance,
            idx,
            self.room_part_refill_distance_count,
        );
        copy_row(
            &mut self.room_part_frontier_from_room_distance,
            &features.room_part_frontier_from_room_distance,
            idx,
            self.room_part_frontier_distance_count,
        );
        copy_row(
            &mut self.room_part_frontier_to_room_distance,
            &features.room_part_frontier_to_room_distance,
            idx,
            self.room_part_frontier_distance_count,
        );
        copy_row(
            &mut self.known_save_from_room_distance,
            &features.known_save_from_room_distance,
            idx,
            self.known_distance_count,
        );
        copy_row(
            &mut self.known_save_to_room_distance,
            &features.known_save_to_room_distance,
            idx,
            self.known_distance_count,
        );
        copy_row(
            &mut self.known_refill_from_room_distance,
            &features.known_refill_from_room_distance,
            idx,
            self.known_distance_count,
        );
        copy_row(
            &mut self.known_refill_to_room_distance,
            &features.known_refill_to_room_distance,
            idx,
            self.known_distance_count,
        );
        copy_row(
            &mut self.connection_reachability,
            &features.connection_reachability,
            idx,
            self.connection_count,
        );
        copy_row(
            &mut self.toilet_crossed_room_idx,
            &features.toilet_crossed_room_idx,
            idx,
            self.toilet_crossed_room_count,
        );
    }
}

impl FrontierFeatureOutputSlices<'_> {
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
        if !features.frontier_door_variant.is_empty() {
            self.frontier_door_variant[dst_idx] = features.frontier_door_variant[src_idx];
        }
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

struct FeatureOutputShards {
    global: GlobalFeatureOutputShards,
    frontier_rows: FrontierFeatureOutputShards,
    row_snapshot_idx: OutputShard<i64>,
    row_frontier_idx: OutputShard<FrontierIdx>,
    row_door_output_idx: OutputShard<i16>,
    missing_connect_query_snapshot_idx: OutputShard<i64>,
    missing_connect_query_connection_idx: OutputShard<i64>,
    missing_connect_query_source_frontier: OutputShard<i16>,
    missing_connect_query_target_frontier: OutputShard<i16>,
    missing_connect_query_source_distance: OutputShard<u8>,
    missing_connect_query_target_distance: OutputShard<u8>,
    missing_connect_query_current_distance: OutputShard<u8>,
    save_refill_utility_query_snapshot_idx: OutputShard<i64>,
    save_refill_utility_query_room_part_idx: OutputShard<i64>,
    save_refill_utility_query_target_mask: OutputShard<u8>,
    save_refill_utility_query_frontier: OutputShard<i16>,
    save_refill_utility_query_frontier_distance: OutputShard<u8>,
    save_refill_utility_query_save_to_current_distance: OutputShard<u8>,
    save_refill_utility_query_save_from_current_distance: OutputShard<u8>,
    save_refill_utility_query_refill_to_current_distance: OutputShard<u8>,
    save_refill_utility_query_refill_from_current_distance: OutputShard<u8>,
    snapshot_start: usize,
}

struct FeatureOutputSlices<'a> {
    global: GlobalFeatureOutputSlices<'a>,
    frontier_rows: FrontierFeatureOutputSlices<'a>,
    row_snapshot_idx: &'a mut [i64],
    row_frontier_idx: &'a mut [FrontierIdx],
    row_door_output_idx: &'a mut [i16],
    missing_connect_query_snapshot_idx: &'a mut [i64],
    missing_connect_query_connection_idx: &'a mut [i64],
    missing_connect_query_source_frontier: &'a mut [i16],
    missing_connect_query_target_frontier: &'a mut [i16],
    missing_connect_query_source_distance: &'a mut [u8],
    missing_connect_query_target_distance: &'a mut [u8],
    missing_connect_query_current_distance: &'a mut [u8],
    save_refill_utility_query_snapshot_idx: &'a mut [i64],
    save_refill_utility_query_room_part_idx: &'a mut [i64],
    save_refill_utility_query_target_mask: &'a mut [u8],
    save_refill_utility_query_frontier: &'a mut [i16],
    save_refill_utility_query_frontier_distance: &'a mut [u8],
    save_refill_utility_query_save_to_current_distance: &'a mut [u8],
    save_refill_utility_query_save_from_current_distance: &'a mut [u8],
    save_refill_utility_query_refill_to_current_distance: &'a mut [u8],
    save_refill_utility_query_refill_from_current_distance: &'a mut [u8],
    snapshot_start: usize,
    frontier_row_count: usize,
    missing_connect_query_row_count: usize,
    save_refill_utility_query_row_count: usize,
}

impl FeatureOutputShards {
    unsafe fn into_slices<'a>(self) -> FeatureOutputSlices<'a> {
        FeatureOutputSlices {
            global: unsafe { self.global.into_slices() },
            frontier_rows: unsafe { self.frontier_rows.into_slices() },
            row_snapshot_idx: unsafe { self.row_snapshot_idx.into_mut_slice() },
            row_frontier_idx: unsafe { self.row_frontier_idx.into_mut_slice() },
            row_door_output_idx: unsafe { self.row_door_output_idx.into_mut_slice() },
            missing_connect_query_snapshot_idx: unsafe {
                self.missing_connect_query_snapshot_idx.into_mut_slice()
            },
            missing_connect_query_connection_idx: unsafe {
                self.missing_connect_query_connection_idx.into_mut_slice()
            },
            missing_connect_query_source_frontier: unsafe {
                self.missing_connect_query_source_frontier.into_mut_slice()
            },
            missing_connect_query_target_frontier: unsafe {
                self.missing_connect_query_target_frontier.into_mut_slice()
            },
            missing_connect_query_source_distance: unsafe {
                self.missing_connect_query_source_distance.into_mut_slice()
            },
            missing_connect_query_target_distance: unsafe {
                self.missing_connect_query_target_distance.into_mut_slice()
            },
            missing_connect_query_current_distance: unsafe {
                self.missing_connect_query_current_distance.into_mut_slice()
            },
            save_refill_utility_query_snapshot_idx: unsafe {
                self.save_refill_utility_query_snapshot_idx.into_mut_slice()
            },
            save_refill_utility_query_room_part_idx: unsafe {
                self.save_refill_utility_query_room_part_idx
                    .into_mut_slice()
            },
            save_refill_utility_query_target_mask: unsafe {
                self.save_refill_utility_query_target_mask.into_mut_slice()
            },
            save_refill_utility_query_frontier: unsafe {
                self.save_refill_utility_query_frontier.into_mut_slice()
            },
            save_refill_utility_query_frontier_distance: unsafe {
                self.save_refill_utility_query_frontier_distance
                    .into_mut_slice()
            },
            save_refill_utility_query_save_to_current_distance: unsafe {
                self.save_refill_utility_query_save_to_current_distance
                    .into_mut_slice()
            },
            save_refill_utility_query_save_from_current_distance: unsafe {
                self.save_refill_utility_query_save_from_current_distance
                    .into_mut_slice()
            },
            save_refill_utility_query_refill_to_current_distance: unsafe {
                self.save_refill_utility_query_refill_to_current_distance
                    .into_mut_slice()
            },
            save_refill_utility_query_refill_from_current_distance: unsafe {
                self.save_refill_utility_query_refill_from_current_distance
                    .into_mut_slice()
            },
            snapshot_start: self.snapshot_start,
            frontier_row_count: 0,
            missing_connect_query_row_count: 0,
            save_refill_utility_query_row_count: 0,
        }
    }
}

impl FeatureOutputSlices<'_> {
    fn write_missing_connect_query_rows(
        query_row_count: &mut usize,
        snapshot_start: usize,
        snapshot_idx: usize,
        frontier_count: usize,
        query_snapshot_idx: &mut [i64],
        query_connection_idx: &mut [i64],
        query_source_frontier: &mut [i16],
        query_target_frontier: &mut [i16],
        query_source_distance: &mut [u8],
        query_target_distance: &mut [u8],
        query_current_distance: &mut [u8],
        feature_connection_idx: &[i64],
        feature_source_frontier: &[i16],
        feature_target_frontier: &[i16],
        feature_source_distance: &[u8],
        feature_target_distance: &[u8],
        feature_current_distance: &[u8],
    ) {
        for query_idx in 0..feature_connection_idx.len() {
            let dst_idx = *query_row_count;
            query_snapshot_idx[dst_idx] = (snapshot_start + snapshot_idx) as i64;
            query_connection_idx[dst_idx] = feature_connection_idx[query_idx];
            query_current_distance[dst_idx] = feature_current_distance[query_idx];
            let query_start = query_idx * frontier_count;
            let query_end = query_start + frontier_count;
            let dst_start = dst_idx * frontier_count;
            let dst_end = dst_start + frontier_count;
            query_source_frontier[dst_start..dst_end]
                .copy_from_slice(&feature_source_frontier[query_start..query_end]);
            query_target_frontier[dst_start..dst_end]
                .copy_from_slice(&feature_target_frontier[query_start..query_end]);
            query_source_distance[dst_start..dst_end]
                .copy_from_slice(&feature_source_distance[query_start..query_end]);
            query_target_distance[dst_start..dst_end]
                .copy_from_slice(&feature_target_distance[query_start..query_end]);
            *query_row_count += 1;
        }
    }

    fn write_save_refill_utility_query_rows(
        query_row_count: &mut usize,
        snapshot_start: usize,
        snapshot_idx: usize,
        query_snapshot_idx: &mut [i64],
        query_room_part_idx: &mut [i64],
        query_target_mask: &mut [u8],
        query_frontier: &mut [i16],
        query_frontier_distance: &mut [u8],
        query_save_to_current_distance: &mut [u8],
        query_save_from_current_distance: &mut [u8],
        query_refill_to_current_distance: &mut [u8],
        query_refill_from_current_distance: &mut [u8],
        feature_room_part_idx: &[i64],
        feature_target_mask: &[u8],
        feature_frontier: &[i16],
        feature_frontier_distance: &[u8],
        feature_save_to_current_distance: &[u8],
        feature_save_from_current_distance: &[u8],
        feature_refill_to_current_distance: &[u8],
        feature_refill_from_current_distance: &[u8],
    ) {
        for query_idx in 0..feature_room_part_idx.len() {
            let dst_idx = *query_row_count;
            query_snapshot_idx[dst_idx] = (snapshot_start + snapshot_idx) as i64;
            query_room_part_idx[dst_idx] = feature_room_part_idx[query_idx];
            query_target_mask[dst_idx] = feature_target_mask[query_idx];
            query_frontier[dst_idx] = feature_frontier[query_idx];
            query_frontier_distance[dst_idx] = feature_frontier_distance[query_idx];
            query_save_to_current_distance[dst_idx] = feature_save_to_current_distance[query_idx];
            query_save_from_current_distance[dst_idx] =
                feature_save_from_current_distance[query_idx];
            query_refill_to_current_distance[dst_idx] =
                feature_refill_to_current_distance[query_idx];
            query_refill_from_current_distance[dst_idx] =
                feature_refill_from_current_distance[query_idx];
            *query_row_count += 1;
        }
    }

    fn write_features(&mut self, snapshot_idx: usize, features: &Features) {
        self.global.write_features(snapshot_idx, features);
        let frontier_count = features.frontier.len() / FEATURE_FRONTIER_WIDTH;
        for frontier_idx in 0..frontier_count {
            let frontier_row_idx = self.frontier_row_count;
            self.row_snapshot_idx[frontier_row_idx] = (self.snapshot_start + snapshot_idx) as i64;
            self.row_frontier_idx[frontier_row_idx] = frontier_idx as FrontierIdx;
            self.row_door_output_idx[frontier_row_idx] = features
                .row_door_output_idx
                .get(frontier_idx)
                .copied()
                .unwrap_or(-1);
            self.frontier_rows
                .write_frontier_row(frontier_row_idx, features, frontier_idx);
            self.frontier_row_count += 1;
        }
        Self::write_missing_connect_query_rows(
            &mut self.missing_connect_query_row_count,
            self.snapshot_start,
            snapshot_idx,
            MISSING_CONNECT_QUERY_FRONTIER_COUNT,
            self.missing_connect_query_snapshot_idx,
            self.missing_connect_query_connection_idx,
            self.missing_connect_query_source_frontier,
            self.missing_connect_query_target_frontier,
            self.missing_connect_query_source_distance,
            self.missing_connect_query_target_distance,
            self.missing_connect_query_current_distance,
            &features.missing_connect_query_connection_idx,
            &features.missing_connect_query_source_frontier,
            &features.missing_connect_query_target_frontier,
            &features.missing_connect_query_source_distance,
            &features.missing_connect_query_target_distance,
            &features.missing_connect_query_current_distance,
        );
        Self::write_save_refill_utility_query_rows(
            &mut self.save_refill_utility_query_row_count,
            self.snapshot_start,
            snapshot_idx,
            self.save_refill_utility_query_snapshot_idx,
            self.save_refill_utility_query_room_part_idx,
            self.save_refill_utility_query_target_mask,
            self.save_refill_utility_query_frontier,
            self.save_refill_utility_query_frontier_distance,
            self.save_refill_utility_query_save_to_current_distance,
            self.save_refill_utility_query_save_from_current_distance,
            self.save_refill_utility_query_refill_to_current_distance,
            self.save_refill_utility_query_refill_from_current_distance,
            &features.save_refill_utility_query_room_part_idx,
            &features.save_refill_utility_query_target_mask,
            &features.save_refill_utility_query_frontier,
            &features.save_refill_utility_query_frontier_distance,
            &features.save_refill_utility_query_save_to_current_distance,
            &features.save_refill_utility_query_save_from_current_distance,
            &features.save_refill_utility_query_refill_to_current_distance,
            &features.save_refill_utility_query_refill_from_current_distance,
        );
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
        let _toilet_room_idx = common_data.toilet_room_idx();

        Ok(Self {
            common_data,
            features,
        })
    }

    #[pyo3(signature = (map_size, num_environments, seed, frontier_neighbor_count, frontier_window_size, candidate_spatial_cell_size, num_threads=None, frontier_neighbor_algorithm="delaunay"))]
    fn create_environment_group(
        &self,
        map_size: (Coord, Coord),
        num_environments: usize,
        seed: u64,
        frontier_neighbor_count: usize,
        frontier_window_size: usize,
        candidate_spatial_cell_size: usize,
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
            candidate_spatial_cell_size,
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
            self.common_data.room_part.len(),
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
        candidate_spatial_cell_size: usize,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
        if candidate_spatial_cell_size == 0 {
            return Err(PyValueError::new_err(
                "candidate_spatial_cell_size must be greater than 0",
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
                    candidate_spatial_cell_size,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_global_feature_output_shards() -> GlobalFeatureOutputShards {
        GlobalFeatureOutputShards {
            inventory: OutputShard::empty(),
            room_x: OutputShard::empty(),
            room_y: OutputShard::empty(),
            room_placed: OutputShard::empty(),
            room_part_furthest_destination: OutputShard::empty(),
            room_part_furthest_source: OutputShard::empty(),
            room_part_save_from_room_distance: OutputShard::empty(),
            room_part_save_to_room_distance: OutputShard::empty(),
            room_part_refill_from_room_distance: OutputShard::empty(),
            room_part_refill_to_room_distance: OutputShard::empty(),
            room_part_frontier_from_room_distance: OutputShard::empty(),
            room_part_frontier_to_room_distance: OutputShard::empty(),
            known_save_from_room_distance: OutputShard::empty(),
            known_save_to_room_distance: OutputShard::empty(),
            known_refill_from_room_distance: OutputShard::empty(),
            known_refill_to_room_distance: OutputShard::empty(),
            connection_reachability: OutputShard::empty(),
            toilet_crossed_room_idx: OutputShard::empty(),
            inventory_count: 0,
            room_count: 0,
            room_part_furthest_count: 0,
            room_part_save_distance_count: 0,
            room_part_refill_distance_count: 0,
            room_part_frontier_distance_count: 0,
            known_distance_count: 0,
            connection_count: 0,
            toilet_crossed_room_count: 0,
        }
    }

    fn empty_frontier_feature_output_shards() -> FrontierFeatureOutputShards {
        FrontierFeatureOutputShards {
            frontier: OutputShard::empty(),
            frontier_door_variant: OutputShard::empty(),
            frontier_occupancy: OutputShard::empty(),
            frontier_neighbor: OutputShard::empty(),
            frontier_neighbor_pair: OutputShard::empty(),
            frontier_connection_reachability: OutputShard::empty(),
            frontier_neighbor_count: 1,
            connection_count: 0,
            frontier_window_size: 1,
        }
    }

    #[test]
    fn feature_writer_keeps_frontier_neighbors_snapshot_local() {
        let features = Features {
            frontier: vec![1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            frontier_neighbor: vec![1, 0],
            ..Default::default()
        };
        let mut frontier = vec![0; 4 * FEATURE_FRONTIER_WIDTH];
        let mut frontier_neighbor = vec![-1; 4];
        let mut row_snapshot_idx = vec![-1; 4];
        let mut row_frontier_idx = vec![-1; 4];
        let mut row_door_output_idx = vec![-1; 4];
        let mut missing_connect_query_snapshot_idx = Vec::new();
        let mut missing_connect_query_connection_idx = Vec::new();
        let mut missing_connect_query_source_frontier = Vec::new();
        let mut missing_connect_query_target_frontier = Vec::new();
        let mut missing_connect_query_source_distance = Vec::new();
        let mut missing_connect_query_target_distance = Vec::new();
        let mut missing_connect_query_current_distance = Vec::new();
        let mut save_refill_utility_query_snapshot_idx = Vec::new();
        let mut save_refill_utility_query_room_part_idx = Vec::new();
        let mut save_refill_utility_query_target_mask = Vec::new();
        let mut save_refill_utility_query_frontier = Vec::new();
        let mut save_refill_utility_query_frontier_distance = Vec::new();
        let mut save_refill_utility_query_save_to_current_distance = Vec::new();
        let mut save_refill_utility_query_save_from_current_distance = Vec::new();
        let mut save_refill_utility_query_refill_to_current_distance = Vec::new();
        let mut save_refill_utility_query_refill_from_current_distance = Vec::new();

        let outputs = FeatureOutputShards {
            global: empty_global_feature_output_shards(),
            frontier_rows: FrontierFeatureOutputShards {
                frontier: OutputShard::from_slice(&mut frontier),
                frontier_neighbor: OutputShard::from_slice(&mut frontier_neighbor),
                ..empty_frontier_feature_output_shards()
            },
            row_snapshot_idx: OutputShard::from_slice(&mut row_snapshot_idx),
            row_frontier_idx: OutputShard::from_slice(&mut row_frontier_idx),
            row_door_output_idx: OutputShard::from_slice(&mut row_door_output_idx),
            missing_connect_query_snapshot_idx: OutputShard::from_slice(
                &mut missing_connect_query_snapshot_idx,
            ),
            missing_connect_query_connection_idx: OutputShard::from_slice(
                &mut missing_connect_query_connection_idx,
            ),
            missing_connect_query_source_frontier: OutputShard::from_slice(
                &mut missing_connect_query_source_frontier,
            ),
            missing_connect_query_target_frontier: OutputShard::from_slice(
                &mut missing_connect_query_target_frontier,
            ),
            missing_connect_query_source_distance: OutputShard::from_slice(
                &mut missing_connect_query_source_distance,
            ),
            missing_connect_query_target_distance: OutputShard::from_slice(
                &mut missing_connect_query_target_distance,
            ),
            missing_connect_query_current_distance: OutputShard::from_slice(
                &mut missing_connect_query_current_distance,
            ),
            save_refill_utility_query_snapshot_idx: OutputShard::from_slice(
                &mut save_refill_utility_query_snapshot_idx,
            ),
            save_refill_utility_query_room_part_idx: OutputShard::from_slice(
                &mut save_refill_utility_query_room_part_idx,
            ),
            save_refill_utility_query_target_mask: OutputShard::from_slice(
                &mut save_refill_utility_query_target_mask,
            ),
            save_refill_utility_query_frontier: OutputShard::from_slice(
                &mut save_refill_utility_query_frontier,
            ),
            save_refill_utility_query_frontier_distance: OutputShard::from_slice(
                &mut save_refill_utility_query_frontier_distance,
            ),
            save_refill_utility_query_save_to_current_distance: OutputShard::from_slice(
                &mut save_refill_utility_query_save_to_current_distance,
            ),
            save_refill_utility_query_save_from_current_distance: OutputShard::from_slice(
                &mut save_refill_utility_query_save_from_current_distance,
            ),
            save_refill_utility_query_refill_to_current_distance: OutputShard::from_slice(
                &mut save_refill_utility_query_refill_to_current_distance,
            ),
            save_refill_utility_query_refill_from_current_distance: OutputShard::from_slice(
                &mut save_refill_utility_query_refill_from_current_distance,
            ),
            snapshot_start: 10,
        };
        let mut outputs = unsafe { outputs.into_slices() };

        outputs.write_features(0, &features);
        outputs.write_features(1, &features);

        assert_eq!(frontier_neighbor, vec![1, 0, 1, 0]);
        assert_eq!(row_snapshot_idx, vec![10, 10, 11, 11]);
        assert_eq!(row_frontier_idx, vec![0, 1, 0, 1]);
        assert_eq!(row_door_output_idx, vec![-1, -1, -1, -1]);
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

    fn step_initial(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.action_count != 0 {
            return Err(PyValueError::new_err(
                "step_initial is only valid before any actions have been applied",
            ));
        }
        py.detach(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                if let Err(err) = worker.send(WorkerCommand::StepInitial) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }

            wait_for_done_responses(&self.workers, sent_workers, first_error)
        })?;

        self.action_count = 1;
        Ok(())
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

    fn get_proposal_candidate_mask<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<ProposalCandidateMask> {
        let door_variant_count = self.common_data.num_door_output_variants;
        let mask_byte_count = door_variant_count.div_ceil(8);
        let mut proposal_frontier_idx = vec![-1; self.num_environments];
        let mut mask = vec![0; self.num_environments * mask_byte_count];
        let mut valid_counts = vec![0; self.num_environments];

        py.detach(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                let mask_start = worker.start * mask_byte_count;
                let mask_end = worker.end() * mask_byte_count;
                if let Err(err) = worker.send(WorkerCommand::GetProposalCandidateMask {
                    proposal_door_variant_count: door_variant_count,
                    proposal_mask_byte_count: mask_byte_count,
                    proposal_frontier_idx: OutputShard::from_slice(
                        &mut proposal_frontier_idx[worker.start..worker.end()],
                    ),
                    mask: OutputShard::from_slice(&mut mask[mask_start..mask_end]),
                    valid_counts: OutputShard::from_slice(
                        &mut valid_counts[worker.start..worker.end()],
                    ),
                }) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }

            wait_for_done_responses(&self.workers, sent_workers, first_error)
        })?;

        Ok(ProposalCandidateMask {
            proposal_frontier_idx: proposal_frontier_idx.into_pyarray(py).unbind(),
            mask: pyarray2_from_flat_vec(py, mask, self.num_environments, mask_byte_count)?
                .unbind(),
            valid_counts: valid_counts.into_pyarray(py).unbind(),
            door_variant_count,
        })
    }

    fn pack_candidates_from_proposals_into<'py>(
        &mut self,
        py: Python<'py>,
        buffers: PyRef<'py, ProposalCandidateBuffers>,
    ) -> PyResult<FeatureRequirements> {
        if self.action_count == 0 {
            return Err(PyValueError::new_err(
                "pack_candidates_from_proposals_into requires step_initial to be called first",
            ));
        }
        let sampled_frontier_idx = buffers.sampled_frontier_idx.bind(py).readonly();
        let sampled_door_variant_idx = buffers.sampled_door_variant_idx.bind(py).readonly();
        let recommended_candidates = buffers.recommended_candidates;
        let mut room_idx = buffers.room_idx.bind(py).readwrite();
        let mut room_x = buffers.room_x.bind(py).readwrite();
        let mut room_y = buffers.room_y.bind(py).readwrite();
        let mut proposal_frontier_idx = buffers.proposal_frontier_idx.bind(py).readwrite();
        let mut proposal_door_variant_idx = buffers.proposal_door_variant_idx.bind(py).readwrite();
        let mut pre_door_valid = buffers.pre_door_valid.bind(py).readwrite();
        let mut pre_connections_valid = buffers.pre_connections_valid.bind(py).readwrite();
        let mut pre_toilet_valid = buffers.pre_toilet_valid.bind(py).readwrite();
        let mut door_valid = buffers.door_valid.bind(py).readwrite();
        let mut connections_valid = buffers.connections_valid.bind(py).readwrite();
        let mut toilet_valid = buffers.toilet_valid.bind(py).readwrite();
        let mut door_match = buffers.door_match.bind(py).readwrite();
        let mut clean_counts = buffers.clean_counts.bind(py).readwrite();
        let mut evaluated_counts = buffers.evaluated_counts.bind(py).readwrite();
        let mut rejected_counts = buffers.rejected_counts.bind(py).readwrite();
        let sampled_shape = sampled_frontier_idx.as_array().shape().to_vec();
        if sampled_shape.len() != 2
            || sampled_door_variant_idx.as_array().shape() != sampled_shape
            || sampled_shape[0] != self.num_environments
        {
            return Err(PyValueError::new_err(
                "sampled proposal arrays must have shape [environment, shortlist_candidate]",
            ));
        }
        let shortlist_candidates = sampled_shape[1];
        let sampled_frontier_idx = sampled_frontier_idx
            .as_slice()
            .map_err(|_| PyValueError::new_err("sampled_frontier_idx must be contiguous"))?;
        let sampled_door_variant_idx = sampled_door_variant_idx
            .as_slice()
            .map_err(|_| PyValueError::new_err("sampled_door_variant_idx must be contiguous"))?;
        let (door_outcome_count, connection_outcome_count) = output_sizes(&self.common_data);
        let dummy_candidate = Action {
            room_idx: self.common_data.room.len() as RoomIdx,
            x: 0,
            y: 0,
        };

        check_shape(
            "room_idx",
            room_idx.as_array().shape(),
            &[self.num_environments, recommended_candidates],
        )?;
        check_shape(
            "room_x",
            room_x.as_array().shape(),
            &[self.num_environments, recommended_candidates],
        )?;
        check_shape(
            "room_y",
            room_y.as_array().shape(),
            &[self.num_environments, recommended_candidates],
        )?;
        check_shape(
            "proposal_frontier_idx",
            proposal_frontier_idx.as_array().shape(),
            &[self.num_environments, recommended_candidates],
        )?;
        check_shape(
            "proposal_door_variant_idx",
            proposal_door_variant_idx.as_array().shape(),
            &[self.num_environments, recommended_candidates],
        )?;
        check_shape(
            "pre_door_valid",
            pre_door_valid.as_array().shape(),
            &[self.num_environments, door_outcome_count],
        )?;
        check_shape(
            "pre_connections_valid",
            pre_connections_valid.as_array().shape(),
            &[self.num_environments, connection_outcome_count],
        )?;
        check_shape(
            "pre_toilet_valid",
            pre_toilet_valid.as_array().shape(),
            &[self.num_environments],
        )?;
        check_shape(
            "door_valid",
            door_valid.as_array().shape(),
            &[
                self.num_environments,
                recommended_candidates,
                door_outcome_count,
            ],
        )?;
        check_shape(
            "connections_valid",
            connections_valid.as_array().shape(),
            &[
                self.num_environments,
                recommended_candidates,
                connection_outcome_count,
            ],
        )?;
        check_shape(
            "toilet_valid",
            toilet_valid.as_array().shape(),
            &[self.num_environments, recommended_candidates],
        )?;
        check_shape(
            "door_match",
            door_match.as_array().shape(),
            &[
                self.num_environments,
                recommended_candidates,
                door_outcome_count,
            ],
        )?;
        check_shape(
            "clean_counts",
            clean_counts.as_array().shape(),
            &[self.num_environments],
        )?;
        check_shape(
            "evaluated_counts",
            evaluated_counts.as_array().shape(),
            &[self.num_environments],
        )?;
        check_shape(
            "rejected_counts",
            rejected_counts.as_array().shape(),
            &[self.num_environments],
        )?;

        let room_idx = room_idx
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("room_idx must be contiguous"))?;
        let room_x = room_x
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("room_x must be contiguous"))?;
        let room_y = room_y
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("room_y must be contiguous"))?;
        let proposal_frontier_idx = proposal_frontier_idx
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("proposal_frontier_idx must be contiguous"))?;
        let proposal_door_variant_idx = proposal_door_variant_idx
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("proposal_door_variant_idx must be contiguous"))?;
        let pre_door_valid = pre_door_valid
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("pre_door_valid must be contiguous"))?;
        let pre_connections_valid = pre_connections_valid
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("pre_connections_valid must be contiguous"))?;
        let pre_toilet_valid = pre_toilet_valid
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("pre_toilet_valid must be contiguous"))?;
        let door_valid = door_valid
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("door_valid must be contiguous"))?;
        let connections_valid = connections_valid
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("connections_valid must be contiguous"))?;
        let toilet_valid = toilet_valid
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("toilet_valid must be contiguous"))?;
        let door_match = door_match
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("door_match must be contiguous"))?;
        let stats_clean_counts = clean_counts
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("clean_counts must be contiguous"))?;
        let stats_evaluated_counts = evaluated_counts
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("evaluated_counts must be contiguous"))?;
        let stats_rejected_counts = rejected_counts
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("rejected_counts must be contiguous"))?;

        room_idx.fill(dummy_candidate.room_idx);
        room_x.fill(dummy_candidate.x);
        room_y.fill(dummy_candidate.y);
        proposal_frontier_idx.fill(-1);
        proposal_door_variant_idx.fill(-1);
        pre_door_valid.fill(DoorValidOutcome::Unknown as i8);
        pre_connections_valid.fill(DoorValidOutcome::Unknown as i8);
        pre_toilet_valid.fill(DoorValidOutcome::Unknown as i8);
        door_valid.fill(DoorValidOutcome::Unknown as i8);
        connections_valid.fill(DoorValidOutcome::Unknown as i8);
        toilet_valid.fill(DoorValidOutcome::Unknown as i8);
        door_match.fill(-1);
        let mut worker_clean_counts = vec![0; self.num_environments];
        let mut worker_evaluated_counts = vec![0; self.num_environments];
        let mut worker_rejected_counts = vec![0; self.num_environments];

        let (feature_info, worker_feature_info) = py.detach(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                let output_start = worker.start * recommended_candidates;
                let output_end = worker.end() * recommended_candidates;
                let shortlist_start = worker.start * shortlist_candidates;
                let shortlist_end = worker.end() * shortlist_candidates;
                let pre_door_output_start = worker.start * door_outcome_count;
                let pre_door_output_end = pre_door_output_start + worker.len * door_outcome_count;
                let pre_connection_output_start = worker.start * connection_outcome_count;
                let pre_connection_output_end =
                    pre_connection_output_start + worker.len * connection_outcome_count;
                let door_output_start = output_start * door_outcome_count;
                let door_output_end = output_end * door_outcome_count;
                let connection_output_start = output_start * connection_outcome_count;
                let connection_output_end = output_end * connection_outcome_count;
                let door_match_output_start = output_start * door_outcome_count;
                let door_match_output_end =
                    door_match_output_start + (output_end - output_start) * door_outcome_count;
                if let Err(err) = worker.send(WorkerCommand::GetCandidatesFromProposals {
                    recommended_candidates,
                    shortlist_candidates,
                    sampled_frontier_idx: InputShard::from_slice(
                        &sampled_frontier_idx[shortlist_start..shortlist_end],
                    ),
                    sampled_door_variant_idx: InputShard::from_slice(
                        &sampled_door_variant_idx[shortlist_start..shortlist_end],
                    ),
                    room_idx: OutputShard::from_slice(&mut room_idx[output_start..output_end]),
                    room_x: OutputShard::from_slice(&mut room_x[output_start..output_end]),
                    room_y: OutputShard::from_slice(&mut room_y[output_start..output_end]),
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
                    pre_toilet_valid: OutputShard::from_slice(
                        &mut pre_toilet_valid[worker.start..worker.end()],
                    ),
                    door_valid: OutputShard::from_slice(
                        &mut door_valid[door_output_start..door_output_end],
                    ),
                    connections_valid: OutputShard::from_slice(
                        &mut connections_valid[connection_output_start..connection_output_end],
                    ),
                    toilet_valid: OutputShard::from_slice(
                        &mut toilet_valid[output_start..output_end],
                    ),
                    door_match: OutputShard::from_slice(
                        &mut door_match[door_match_output_start..door_match_output_end],
                    ),
                    clean_counts: OutputShard::from_slice(
                        &mut worker_clean_counts[worker.start..worker.end()],
                    ),
                    evaluated_counts: OutputShard::from_slice(
                        &mut worker_evaluated_counts[worker.start..worker.end()],
                    ),
                    rejected_counts: OutputShard::from_slice(
                        &mut worker_rejected_counts[worker.start..worker.end()],
                    ),
                }) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }

            collect_feature_info(&self.workers, sent_workers, first_error)
        })?;
        let frontier_row_count =
            feature_info.frontier_row_count * usize::from(self.features.has_frontier_features());
        let missing_connect_query_row_count = feature_info.missing_connect_query_row_count
            * usize::from(self.features.missing_connect_query);
        let save_refill_utility_query_enabled =
            self.features.save_utility_query || self.features.refill_utility_query;
        let save_refill_utility_query_row_count = feature_info.save_refill_utility_query_row_count
            * usize::from(save_refill_utility_query_enabled);
        let worker_frontier_row_counts = worker_feature_info
            .iter()
            .map(|info| {
                info.frontier_row_count * usize::from(self.features.has_frontier_features())
            })
            .collect::<Vec<_>>();
        let worker_missing_connect_query_row_counts = worker_feature_info
            .iter()
            .map(|info| {
                info.missing_connect_query_row_count
                    * usize::from(self.features.missing_connect_query)
            })
            .collect::<Vec<_>>();
        let worker_save_refill_utility_query_row_counts = worker_feature_info
            .into_iter()
            .map(|info| {
                info.save_refill_utility_query_row_count
                    * usize::from(save_refill_utility_query_enabled)
            })
            .collect::<Vec<_>>();

        for (out, count) in stats_clean_counts
            .iter_mut()
            .zip(worker_clean_counts.into_iter())
        {
            *out = count as i64;
        }
        for (out, count) in stats_evaluated_counts
            .iter_mut()
            .zip(worker_evaluated_counts.into_iter())
        {
            *out = count as i64;
        }
        for (out, count) in stats_rejected_counts
            .iter_mut()
            .zip(worker_rejected_counts.into_iter())
        {
            *out = count as i64;
        }

        Ok(FeatureRequirements {
            frontier_row_count,
            worker_frontier_row_counts,
            missing_connect_query_row_count,
            worker_missing_connect_query_row_counts,
            save_refill_utility_query_row_count,
            worker_save_refill_utility_query_row_counts,
        })
    }

    fn get_outcomes<'py>(
        &mut self,
        py: Python<'py>,
        verify_consistency: bool,
    ) -> PyResult<EpisodeOutcomes> {
        let (door_outcome_count, connection_outcome_count) = output_sizes(&self.common_data);
        let door_output_len = self.num_environments * door_outcome_count;
        let connection_output_len = self.num_environments * connection_outcome_count;
        let mut door_valid = vec![DoorValidOutcome::Unknown as i8; door_output_len];
        let mut connections_valid = vec![DoorValidOutcome::Unknown as i8; connection_output_len];
        let mut toilet_valid = vec![DoorValidOutcome::Unknown as i8; self.num_environments];
        let mut toilet_crossed_room_idx = vec![-1i16; self.num_environments];
        let mut avg_frontiers = vec![0.0; self.num_environments];
        let mut graph_diameter = vec![0.0; self.num_environments];
        let room_part_count = self.common_data.room_part.len();
        let mut active_room_part_mask = vec![0; self.num_environments * room_part_count];
        let mut save_distance = vec![0.0; self.num_environments * room_part_count];
        let mut save_distance_mask = vec![0; self.num_environments * room_part_count];
        let mut save_to_room_distance = vec![0.0; self.num_environments * room_part_count];
        let mut save_to_room_distance_mask = vec![0; self.num_environments * room_part_count];
        let mut save_from_room_distance = vec![0.0; self.num_environments * room_part_count];
        let mut save_from_room_distance_mask = vec![0; self.num_environments * room_part_count];
        let mut refill_distance = vec![0.0; self.num_environments * room_part_count];
        let mut refill_distance_mask = vec![0; self.num_environments * room_part_count];
        let mut refill_to_room_distance = vec![0.0; self.num_environments * room_part_count];
        let mut refill_to_room_distance_mask = vec![0; self.num_environments * room_part_count];
        let mut refill_from_room_distance = vec![0.0; self.num_environments * room_part_count];
        let mut refill_from_room_distance_mask = vec![0; self.num_environments * room_part_count];
        let mut missing_connect_distance =
            vec![0.0; self.num_environments * connection_outcome_count];
        let mut missing_connect_distance_mask =
            vec![0; self.num_environments * connection_outcome_count];

        py.detach(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                let door_output_start = worker.start * door_outcome_count;
                let door_output_end = door_output_start + worker.len * door_outcome_count;
                let connection_output_start = worker.start * connection_outcome_count;
                let connection_output_end =
                    connection_output_start + worker.len * connection_outcome_count;
                let avg_frontiers_start = worker.start;
                let avg_frontiers_end = worker.end();
                let graph_diameter_start = worker.start;
                let graph_diameter_end = worker.end();
                let save_distance_start = worker.start * room_part_count;
                let save_distance_end = worker.end() * room_part_count;

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
                    toilet_valid: OutputShard::from_slice(
                        &mut toilet_valid[worker.start..worker.end()],
                    ),
                    toilet_crossed_room_idx: OutputShard::from_slice(
                        &mut toilet_crossed_room_idx[worker.start..worker.end()],
                    ),
                    avg_frontiers: OutputShard::from_slice(
                        &mut avg_frontiers[avg_frontiers_start..avg_frontiers_end],
                    ),
                    graph_diameter: OutputShard::from_slice(
                        &mut graph_diameter[graph_diameter_start..graph_diameter_end],
                    ),
                    active_room_part_mask: OutputShard::from_slice(
                        &mut active_room_part_mask[save_distance_start..save_distance_end],
                    ),
                    save_distance: OutputShard::from_slice(
                        &mut save_distance[save_distance_start..save_distance_end],
                    ),
                    save_distance_mask: OutputShard::from_slice(
                        &mut save_distance_mask[save_distance_start..save_distance_end],
                    ),
                    save_to_room_distance: OutputShard::from_slice(
                        &mut save_to_room_distance[save_distance_start..save_distance_end],
                    ),
                    save_to_room_distance_mask: OutputShard::from_slice(
                        &mut save_to_room_distance_mask[save_distance_start..save_distance_end],
                    ),
                    save_from_room_distance: OutputShard::from_slice(
                        &mut save_from_room_distance[save_distance_start..save_distance_end],
                    ),
                    save_from_room_distance_mask: OutputShard::from_slice(
                        &mut save_from_room_distance_mask[save_distance_start..save_distance_end],
                    ),
                    refill_distance: OutputShard::from_slice(
                        &mut refill_distance[save_distance_start..save_distance_end],
                    ),
                    refill_distance_mask: OutputShard::from_slice(
                        &mut refill_distance_mask[save_distance_start..save_distance_end],
                    ),
                    refill_to_room_distance: OutputShard::from_slice(
                        &mut refill_to_room_distance[save_distance_start..save_distance_end],
                    ),
                    refill_to_room_distance_mask: OutputShard::from_slice(
                        &mut refill_to_room_distance_mask[save_distance_start..save_distance_end],
                    ),
                    refill_from_room_distance: OutputShard::from_slice(
                        &mut refill_from_room_distance[save_distance_start..save_distance_end],
                    ),
                    refill_from_room_distance_mask: OutputShard::from_slice(
                        &mut refill_from_room_distance_mask[save_distance_start..save_distance_end],
                    ),
                    missing_connect_distance: OutputShard::from_slice(
                        &mut missing_connect_distance
                            [connection_output_start..connection_output_end],
                    ),
                    missing_connect_distance_mask: OutputShard::from_slice(
                        &mut missing_connect_distance_mask
                            [connection_output_start..connection_output_end],
                    ),
                }) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }

            wait_for_done_responses(&self.workers, sent_workers, first_error)
        })?;

        Ok(EpisodeOutcomes {
            step_outcomes: StepOutcomes {
                door_valid: pyarray2_from_flat_vec(
                    py,
                    door_valid,
                    self.num_environments,
                    door_outcome_count,
                )?
                .unbind(),
                connections_valid: pyarray2_from_flat_vec(
                    py,
                    connections_valid,
                    self.num_environments,
                    connection_outcome_count,
                )?
                .unbind(),
                toilet_valid: toilet_valid.into_pyarray(py).unbind(),
            },
            end_outcomes: EndOutcomes {
                toilet_crossed_room_idx: toilet_crossed_room_idx.into_pyarray(py).unbind(),
                avg_frontiers: avg_frontiers.into_pyarray(py).unbind(),
                graph_diameter: graph_diameter.into_pyarray(py).unbind(),
                active_room_part_mask: pyarray2_from_flat_vec(
                    py,
                    active_room_part_mask,
                    self.num_environments,
                    room_part_count,
                )?
                .unbind(),
                save_distance: pyarray2_from_flat_vec(
                    py,
                    save_distance,
                    self.num_environments,
                    room_part_count,
                )?
                .unbind(),
                save_distance_mask: pyarray2_from_flat_vec(
                    py,
                    save_distance_mask,
                    self.num_environments,
                    room_part_count,
                )?
                .unbind(),
                save_to_room_distance: pyarray2_from_flat_vec(
                    py,
                    save_to_room_distance,
                    self.num_environments,
                    room_part_count,
                )?
                .unbind(),
                save_to_room_distance_mask: pyarray2_from_flat_vec(
                    py,
                    save_to_room_distance_mask,
                    self.num_environments,
                    room_part_count,
                )?
                .unbind(),
                save_from_room_distance: pyarray2_from_flat_vec(
                    py,
                    save_from_room_distance,
                    self.num_environments,
                    room_part_count,
                )?
                .unbind(),
                save_from_room_distance_mask: pyarray2_from_flat_vec(
                    py,
                    save_from_room_distance_mask,
                    self.num_environments,
                    room_part_count,
                )?
                .unbind(),
                refill_distance: pyarray2_from_flat_vec(
                    py,
                    refill_distance,
                    self.num_environments,
                    room_part_count,
                )?
                .unbind(),
                refill_distance_mask: pyarray2_from_flat_vec(
                    py,
                    refill_distance_mask,
                    self.num_environments,
                    room_part_count,
                )?
                .unbind(),
                refill_to_room_distance: pyarray2_from_flat_vec(
                    py,
                    refill_to_room_distance,
                    self.num_environments,
                    room_part_count,
                )?
                .unbind(),
                refill_to_room_distance_mask: pyarray2_from_flat_vec(
                    py,
                    refill_to_room_distance_mask,
                    self.num_environments,
                    room_part_count,
                )?
                .unbind(),
                refill_from_room_distance: pyarray2_from_flat_vec(
                    py,
                    refill_from_room_distance,
                    self.num_environments,
                    room_part_count,
                )?
                .unbind(),
                refill_from_room_distance_mask: pyarray2_from_flat_vec(
                    py,
                    refill_from_room_distance_mask,
                    self.num_environments,
                    room_part_count,
                )?
                .unbind(),
                missing_connect_distance: pyarray2_from_flat_vec(
                    py,
                    missing_connect_distance,
                    self.num_environments,
                    connection_outcome_count,
                )?
                .unbind(),
                missing_connect_distance_mask: pyarray2_from_flat_vec(
                    py,
                    missing_connect_distance_mask,
                    self.num_environments,
                    connection_outcome_count,
                )?
                .unbind(),
            },
        })
    }

    fn get_current_feature_outcomes<'py>(
        &mut self,
        py: Python<'py>,
        environment_start: usize,
        environment_count: usize,
    ) -> PyResult<(
        Bound<'py, PyArray2<i8>>,
        Bound<'py, PyArray2<i8>>,
        Bound<'py, PyArray1<i8>>,
        Bound<'py, PyArray2<i16>>,
    )> {
        if environment_start + environment_count > self.num_environments {
            return Err(PyValueError::new_err(
                "requested environments must fit within the environment group",
            ));
        }
        let (door_outcome_count, connection_outcome_count) = output_sizes(&self.common_data);
        let door_output_len = environment_count * door_outcome_count;
        let connection_output_len = environment_count * connection_outcome_count;
        let mut door_valid = vec![DoorValidOutcome::Unknown as i8; door_output_len];
        let mut connections_valid = vec![DoorValidOutcome::Unknown as i8; connection_output_len];
        let mut toilet_valid = vec![DoorValidOutcome::Unknown as i8; environment_count];
        let mut door_match = vec![-1; door_output_len];

        py.detach(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                let start = max(environment_start, worker.start);
                let end = min(environment_start + environment_count, worker.end());
                if start >= end {
                    continue;
                }
                let input_start = start - environment_start;
                let environment_count = end - start;
                let door_output_start = input_start * door_outcome_count;
                let door_output_end = door_output_start + environment_count * door_outcome_count;
                let connection_output_start = input_start * connection_outcome_count;
                let connection_output_end =
                    connection_output_start + environment_count * connection_outcome_count;
                let door_match_output_start = input_start * door_outcome_count;
                let door_match_output_end =
                    door_match_output_start + environment_count * door_outcome_count;
                if let Err(err) = worker.send(WorkerCommand::GetCurrentFeatureOutcomes {
                    environment_start: start - worker.start,
                    environment_count,
                    door_outcome_count,
                    connection_outcome_count,
                    door_valid: OutputShard::from_slice(
                        &mut door_valid[door_output_start..door_output_end],
                    ),
                    connections_valid: OutputShard::from_slice(
                        &mut connections_valid[connection_output_start..connection_output_end],
                    ),
                    toilet_valid: OutputShard::from_slice(
                        &mut toilet_valid[input_start..input_start + environment_count],
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
            pyarray2_from_flat_vec(py, door_valid, environment_count, door_outcome_count)?,
            pyarray2_from_flat_vec(
                py,
                connections_valid,
                environment_count,
                connection_outcome_count,
            )?,
            toilet_valid.into_pyarray(py),
            pyarray2_from_flat_vec(py, door_match, environment_count, door_outcome_count)?,
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

    #[pyo3(signature = (environment_start=0, environment_count=None))]
    fn get_feature_requirements<'py>(
        &self,
        py: Python<'py>,
        environment_start: usize,
        environment_count: Option<usize>,
    ) -> PyResult<FeatureRequirements> {
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
        let (feature_info, worker_feature_info) = py.detach(|| {
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
        let frontier_row_count =
            feature_info.frontier_row_count * usize::from(self.features.has_frontier_features());
        let missing_connect_query_row_count = feature_info.missing_connect_query_row_count
            * usize::from(self.features.missing_connect_query);
        let save_refill_utility_query_enabled =
            self.features.save_utility_query || self.features.refill_utility_query;
        let save_refill_utility_query_row_count = feature_info.save_refill_utility_query_row_count
            * usize::from(save_refill_utility_query_enabled);
        let worker_frontier_row_counts = worker_feature_info
            .iter()
            .map(|info| {
                info.frontier_row_count * usize::from(self.features.has_frontier_features())
            })
            .collect::<Vec<_>>();
        let worker_missing_connect_query_row_counts = worker_feature_info
            .iter()
            .map(|info| {
                info.missing_connect_query_row_count
                    * usize::from(self.features.missing_connect_query)
            })
            .collect::<Vec<_>>();
        let worker_save_refill_utility_query_row_counts = worker_feature_info
            .into_iter()
            .map(|info| {
                info.save_refill_utility_query_row_count
                    * usize::from(save_refill_utility_query_enabled)
            })
            .collect::<Vec<_>>();
        Ok(FeatureRequirements {
            frontier_row_count,
            worker_frontier_row_counts,
            missing_connect_query_row_count,
            worker_missing_connect_query_row_counts,
            save_refill_utility_query_row_count,
            worker_save_refill_utility_query_row_counts,
        })
    }

    fn pack_features_into<'py>(
        &self,
        py: Python<'py>,
        buffers: PyRef<'py, FeatureBuffers>,
    ) -> PyResult<()> {
        let environment_count = buffers.environment_count;
        let candidate_count = buffers.candidate_count;
        let environment_start = buffers.environment_start;
        let frontier_row_count = buffers.frontier_row_count;
        let worker_frontier_row_counts = &buffers.worker_frontier_row_counts;
        let missing_connect_query_row_count = buffers.missing_connect_query_row_count;
        let worker_missing_connect_query_row_counts =
            &buffers.worker_missing_connect_query_row_counts;
        let save_refill_utility_query_row_count = buffers.save_refill_utility_query_row_count;
        let worker_save_refill_utility_query_row_counts =
            &buffers.worker_save_refill_utility_query_row_counts;
        let mut inventory = buffers.inventory.bind(py).readwrite();
        let mut out_room_x = buffers.out_room_x.bind(py).readwrite();
        let mut out_room_y = buffers.out_room_y.bind(py).readwrite();
        let mut room_placed = buffers.room_placed.bind(py).readwrite();
        let mut room_part_furthest_destination =
            buffers.room_part_furthest_destination.bind(py).readwrite();
        let mut room_part_furthest_source = buffers.room_part_furthest_source.bind(py).readwrite();
        let mut room_part_save_from_room_distance = buffers
            .room_part_save_from_room_distance
            .bind(py)
            .readwrite();
        let mut room_part_save_to_room_distance =
            buffers.room_part_save_to_room_distance.bind(py).readwrite();
        let mut room_part_refill_from_room_distance = buffers
            .room_part_refill_from_room_distance
            .bind(py)
            .readwrite();
        let mut room_part_refill_to_room_distance = buffers
            .room_part_refill_to_room_distance
            .bind(py)
            .readwrite();
        let mut room_part_frontier_from_room_distance = buffers
            .room_part_frontier_from_room_distance
            .bind(py)
            .readwrite();
        let mut room_part_frontier_to_room_distance = buffers
            .room_part_frontier_to_room_distance
            .bind(py)
            .readwrite();
        let mut known_save_from_room_distance =
            buffers.known_save_from_room_distance.bind(py).readwrite();
        let mut known_save_to_room_distance =
            buffers.known_save_to_room_distance.bind(py).readwrite();
        let mut known_refill_from_room_distance =
            buffers.known_refill_from_room_distance.bind(py).readwrite();
        let mut known_refill_to_room_distance =
            buffers.known_refill_to_room_distance.bind(py).readwrite();
        let mut frontier = buffers.frontier.bind(py).readwrite();
        let mut frontier_door_variant = buffers.frontier_door_variant.bind(py).readwrite();
        let mut frontier_occupancy = buffers.frontier_occupancy.bind(py).readwrite();
        let mut frontier_neighbor = buffers.frontier_neighbor.bind(py).readwrite();
        let mut frontier_neighbor_pair = buffers.frontier_neighbor_pair.bind(py).readwrite();
        let mut connection_reachability = buffers.connection_reachability.bind(py).readwrite();
        let mut frontier_connection_reachability = buffers
            .frontier_connection_reachability
            .bind(py)
            .readwrite();
        let mut missing_connect_query_snapshot_idx = buffers
            .missing_connect_query_snapshot_idx
            .bind(py)
            .readwrite();
        let mut missing_connect_query_connection_idx = buffers
            .missing_connect_query_connection_idx
            .bind(py)
            .readwrite();
        let mut missing_connect_query_source_frontier = buffers
            .missing_connect_query_source_frontier
            .bind(py)
            .readwrite();
        let mut missing_connect_query_target_frontier = buffers
            .missing_connect_query_target_frontier
            .bind(py)
            .readwrite();
        let mut missing_connect_query_source_distance = buffers
            .missing_connect_query_source_distance
            .bind(py)
            .readwrite();
        let mut missing_connect_query_target_distance = buffers
            .missing_connect_query_target_distance
            .bind(py)
            .readwrite();
        let mut missing_connect_query_current_distance = buffers
            .missing_connect_query_current_distance
            .bind(py)
            .readwrite();
        let mut save_refill_utility_query_snapshot_idx = buffers
            .save_refill_utility_query_snapshot_idx
            .bind(py)
            .readwrite();
        let mut save_refill_utility_query_room_part_idx = buffers
            .save_refill_utility_query_room_part_idx
            .bind(py)
            .readwrite();
        let mut save_refill_utility_query_target_mask = buffers
            .save_refill_utility_query_target_mask
            .bind(py)
            .readwrite();
        let mut save_refill_utility_query_frontier = buffers
            .save_refill_utility_query_frontier
            .bind(py)
            .readwrite();
        let mut save_refill_utility_query_frontier_distance = buffers
            .save_refill_utility_query_frontier_distance
            .bind(py)
            .readwrite();
        let mut save_refill_utility_query_save_to_current_distance = buffers
            .save_refill_utility_query_save_to_current_distance
            .bind(py)
            .readwrite();
        let mut save_refill_utility_query_save_from_current_distance = buffers
            .save_refill_utility_query_save_from_current_distance
            .bind(py)
            .readwrite();
        let mut save_refill_utility_query_refill_to_current_distance = buffers
            .save_refill_utility_query_refill_to_current_distance
            .bind(py)
            .readwrite();
        let mut save_refill_utility_query_refill_from_current_distance = buffers
            .save_refill_utility_query_refill_from_current_distance
            .bind(py)
            .readwrite();
        let mut toilet_crossed_room_idx = buffers.toilet_crossed_room_idx.bind(py).readwrite();
        let mut row_snapshot_idx = buffers.row_snapshot_idx.bind(py).readwrite();
        let mut row_frontier_idx = buffers.row_frontier_idx.bind(py).readwrite();
        let mut row_door_output_idx = buffers.row_door_output_idx.bind(py).readwrite();
        if environment_start + environment_count > self.num_environments {
            return Err(PyValueError::new_err(
                "candidate dimensions must fit within the environment group",
            ));
        }
        let snapshot_count = environment_count * candidate_count;

        let inventory_count = self.common_data.connection_variant_rooms.len();
        let room_count = self.common_data.room.len();
        let room_part_count = self.common_data.room_part.len();
        let connection_count = self.common_data.room_connection.len();
        let inventory_width = inventory_count * usize::from(self.features.inventory);
        let room_width = room_count * usize::from(self.features.room_position);
        let room_part_furthest_width =
            room_part_count * usize::from(self.features.room_part_furthest_distance);
        let room_part_save_distance_width =
            room_part_count * usize::from(self.features.room_part_save_distance);
        let room_part_refill_distance_width =
            room_part_count * usize::from(self.features.room_part_refill_distance);
        let room_part_frontier_distance_width =
            room_part_count * usize::from(self.features.room_part_frontier_distance);
        let known_distance_width = room_part_count;
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
        let missing_connect_query_frontier_width =
            MISSING_CONNECT_QUERY_FRONTIER_COUNT * usize::from(self.features.missing_connect_query);
        let toilet_crossed_room_width = usize::from(self.features.toilet_crossed_room);

        let inventory_shape = inventory.as_array().shape().to_vec();
        let room_x_shape = out_room_x.as_array().shape().to_vec();
        let room_y_shape = out_room_y.as_array().shape().to_vec();
        let room_placed_shape = room_placed.as_array().shape().to_vec();
        let room_part_furthest_destination_shape =
            room_part_furthest_destination.as_array().shape().to_vec();
        let room_part_furthest_source_shape = room_part_furthest_source.as_array().shape().to_vec();
        let room_part_save_from_room_distance_shape = room_part_save_from_room_distance
            .as_array()
            .shape()
            .to_vec();
        let room_part_save_to_room_distance_shape =
            room_part_save_to_room_distance.as_array().shape().to_vec();
        let room_part_refill_from_room_distance_shape = room_part_refill_from_room_distance
            .as_array()
            .shape()
            .to_vec();
        let room_part_refill_to_room_distance_shape = room_part_refill_to_room_distance
            .as_array()
            .shape()
            .to_vec();
        let room_part_frontier_from_room_distance_shape = room_part_frontier_from_room_distance
            .as_array()
            .shape()
            .to_vec();
        let room_part_frontier_to_room_distance_shape = room_part_frontier_to_room_distance
            .as_array()
            .shape()
            .to_vec();
        let known_save_from_room_distance_shape =
            known_save_from_room_distance.as_array().shape().to_vec();
        let known_save_to_room_distance_shape =
            known_save_to_room_distance.as_array().shape().to_vec();
        let known_refill_from_room_distance_shape =
            known_refill_from_room_distance.as_array().shape().to_vec();
        let known_refill_to_room_distance_shape =
            known_refill_to_room_distance.as_array().shape().to_vec();
        let frontier_shape = frontier.as_array().shape().to_vec();
        let frontier_door_variant_shape = frontier_door_variant.as_array().shape().to_vec();
        let frontier_occupancy_shape = frontier_occupancy.as_array().shape().to_vec();
        let frontier_neighbor_shape = frontier_neighbor.as_array().shape().to_vec();
        let frontier_neighbor_pair_shape = frontier_neighbor_pair.as_array().shape().to_vec();
        let connection_reachability_shape = connection_reachability.as_array().shape().to_vec();
        let frontier_connection_reachability_shape =
            frontier_connection_reachability.as_array().shape().to_vec();
        let missing_connect_query_snapshot_idx_shape = missing_connect_query_snapshot_idx
            .as_array()
            .shape()
            .to_vec();
        let missing_connect_query_connection_idx_shape = missing_connect_query_connection_idx
            .as_array()
            .shape()
            .to_vec();
        let missing_connect_query_source_frontier_shape = missing_connect_query_source_frontier
            .as_array()
            .shape()
            .to_vec();
        let missing_connect_query_target_frontier_shape = missing_connect_query_target_frontier
            .as_array()
            .shape()
            .to_vec();
        let missing_connect_query_source_distance_shape = missing_connect_query_source_distance
            .as_array()
            .shape()
            .to_vec();
        let missing_connect_query_target_distance_shape = missing_connect_query_target_distance
            .as_array()
            .shape()
            .to_vec();
        let missing_connect_query_current_distance_shape = missing_connect_query_current_distance
            .as_array()
            .shape()
            .to_vec();
        let save_refill_utility_query_snapshot_idx_shape = save_refill_utility_query_snapshot_idx
            .as_array()
            .shape()
            .to_vec();
        let save_refill_utility_query_room_part_idx_shape = save_refill_utility_query_room_part_idx
            .as_array()
            .shape()
            .to_vec();
        let save_refill_utility_query_target_mask_shape = save_refill_utility_query_target_mask
            .as_array()
            .shape()
            .to_vec();
        let save_refill_utility_query_frontier_shape = save_refill_utility_query_frontier
            .as_array()
            .shape()
            .to_vec();
        let save_refill_utility_query_frontier_distance_shape =
            save_refill_utility_query_frontier_distance
                .as_array()
                .shape()
                .to_vec();
        let save_refill_utility_query_save_to_current_distance_shape =
            save_refill_utility_query_save_to_current_distance
                .as_array()
                .shape()
                .to_vec();
        let save_refill_utility_query_save_from_current_distance_shape =
            save_refill_utility_query_save_from_current_distance
                .as_array()
                .shape()
                .to_vec();
        let save_refill_utility_query_refill_to_current_distance_shape =
            save_refill_utility_query_refill_to_current_distance
                .as_array()
                .shape()
                .to_vec();
        let save_refill_utility_query_refill_from_current_distance_shape =
            save_refill_utility_query_refill_from_current_distance
                .as_array()
                .shape()
                .to_vec();
        let toilet_crossed_room_shape = toilet_crossed_room_idx.as_array().shape().to_vec();
        let row_snapshot_idx_shape = row_snapshot_idx.as_array().shape().to_vec();
        let row_frontier_idx_shape = row_frontier_idx.as_array().shape().to_vec();
        let row_door_output_idx_shape = row_door_output_idx.as_array().shape().to_vec();
        if inventory_shape[0] < snapshot_count
            || room_x_shape[0] < snapshot_count
            || room_y_shape[0] < snapshot_count
            || room_placed_shape[0] < snapshot_count
            || room_part_furthest_destination_shape[0] < snapshot_count
            || room_part_furthest_source_shape[0] < snapshot_count
            || room_part_save_from_room_distance_shape[0] < snapshot_count
            || room_part_save_to_room_distance_shape[0] < snapshot_count
            || room_part_refill_from_room_distance_shape[0] < snapshot_count
            || room_part_refill_to_room_distance_shape[0] < snapshot_count
            || room_part_frontier_from_room_distance_shape[0] < snapshot_count
            || room_part_frontier_to_room_distance_shape[0] < snapshot_count
            || known_save_from_room_distance_shape[0] < snapshot_count
            || known_save_to_room_distance_shape[0] < snapshot_count
            || known_refill_from_room_distance_shape[0] < snapshot_count
            || known_refill_to_room_distance_shape[0] < snapshot_count
            || connection_reachability_shape[0] < snapshot_count
            || toilet_crossed_room_shape[0] < snapshot_count
            || frontier_shape[0] < frontier_row_count
            || frontier_door_variant_shape[0] < frontier_row_count
            || frontier_occupancy_shape[0] < frontier_row_count
            || frontier_neighbor_shape[0] < frontier_row_count
            || frontier_neighbor_pair_shape[0] < frontier_row_count
            || frontier_connection_reachability_shape[0] < frontier_row_count
            || missing_connect_query_snapshot_idx_shape[0] < missing_connect_query_row_count
            || missing_connect_query_connection_idx_shape[0] < missing_connect_query_row_count
            || missing_connect_query_source_frontier_shape[0] < missing_connect_query_row_count
            || missing_connect_query_target_frontier_shape[0] < missing_connect_query_row_count
            || missing_connect_query_source_distance_shape[0] < missing_connect_query_row_count
            || missing_connect_query_target_distance_shape[0] < missing_connect_query_row_count
            || missing_connect_query_current_distance_shape[0] < missing_connect_query_row_count
            || save_refill_utility_query_snapshot_idx_shape[0] < save_refill_utility_query_row_count
            || save_refill_utility_query_room_part_idx_shape[0]
                < save_refill_utility_query_row_count
            || save_refill_utility_query_target_mask_shape[0] < save_refill_utility_query_row_count
            || save_refill_utility_query_frontier_shape[0] < save_refill_utility_query_row_count
            || save_refill_utility_query_frontier_distance_shape[0]
                < save_refill_utility_query_row_count
            || save_refill_utility_query_save_to_current_distance_shape[0]
                < save_refill_utility_query_row_count
            || save_refill_utility_query_save_from_current_distance_shape[0]
                < save_refill_utility_query_row_count
            || save_refill_utility_query_refill_to_current_distance_shape[0]
                < save_refill_utility_query_row_count
            || save_refill_utility_query_refill_from_current_distance_shape[0]
                < save_refill_utility_query_row_count
            || row_snapshot_idx_shape[0] < frontier_row_count
            || row_frontier_idx_shape[0] < frontier_row_count
            || row_door_output_idx_shape[0] < frontier_row_count
        {
            return Err(PyValueError::new_err(
                "frontier feature output buffer is too small",
            ));
        }
        check_dim("inventory", inventory_shape[1], inventory_width)?;
        check_dim("room_x", room_x_shape[1], room_width)?;
        check_dim("room_y", room_y_shape[1], room_width)?;
        check_dim("room_placed", room_placed_shape[1], room_width)?;
        check_dim(
            "room_part_furthest_destination",
            room_part_furthest_destination_shape[1],
            room_part_furthest_width,
        )?;
        check_dim(
            "room_part_furthest_source",
            room_part_furthest_source_shape[1],
            room_part_furthest_width,
        )?;
        check_dim(
            "room_part_save_from_room_distance",
            room_part_save_from_room_distance_shape[1],
            room_part_save_distance_width,
        )?;
        check_dim(
            "room_part_save_to_room_distance",
            room_part_save_to_room_distance_shape[1],
            room_part_save_distance_width,
        )?;
        check_dim(
            "room_part_refill_from_room_distance",
            room_part_refill_from_room_distance_shape[1],
            room_part_refill_distance_width,
        )?;
        check_dim(
            "room_part_refill_to_room_distance",
            room_part_refill_to_room_distance_shape[1],
            room_part_refill_distance_width,
        )?;
        check_dim(
            "room_part_frontier_from_room_distance",
            room_part_frontier_from_room_distance_shape[1],
            room_part_frontier_distance_width,
        )?;
        check_dim(
            "room_part_frontier_to_room_distance",
            room_part_frontier_to_room_distance_shape[1],
            room_part_frontier_distance_width,
        )?;
        check_dim(
            "known_save_from_room_distance",
            known_save_from_room_distance_shape[1],
            known_distance_width,
        )?;
        check_dim(
            "known_save_to_room_distance",
            known_save_to_room_distance_shape[1],
            known_distance_width,
        )?;
        check_dim(
            "known_refill_from_room_distance",
            known_refill_from_room_distance_shape[1],
            known_distance_width,
        )?;
        check_dim(
            "known_refill_to_room_distance",
            known_refill_to_room_distance_shape[1],
            known_distance_width,
        )?;
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
        check_dim(
            "missing_connect_query_source_frontier",
            missing_connect_query_source_frontier_shape[1],
            missing_connect_query_frontier_width,
        )?;
        check_dim(
            "missing_connect_query_target_frontier",
            missing_connect_query_target_frontier_shape[1],
            missing_connect_query_frontier_width,
        )?;
        check_dim(
            "missing_connect_query_source_distance",
            missing_connect_query_source_distance_shape[1],
            missing_connect_query_frontier_width,
        )?;
        check_dim(
            "missing_connect_query_target_distance",
            missing_connect_query_target_distance_shape[1],
            missing_connect_query_frontier_width,
        )?;
        check_dim(
            "toilet_crossed_room_idx",
            toilet_crossed_room_shape[1],
            toilet_crossed_room_width,
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
        let room_part_furthest_destination =
            room_part_furthest_destination.as_slice_mut().map_err(|_| {
                PyValueError::new_err("room_part_furthest_destination must be contiguous")
            })?;
        let room_part_furthest_source = room_part_furthest_source
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("room_part_furthest_source must be contiguous"))?;
        let room_part_save_from_room_distance = room_part_save_from_room_distance
            .as_slice_mut()
            .map_err(|_| {
                PyValueError::new_err("room_part_save_from_room_distance must be contiguous")
            })?;
        let room_part_save_to_room_distance = room_part_save_to_room_distance
            .as_slice_mut()
            .map_err(|_| {
                PyValueError::new_err("room_part_save_to_room_distance must be contiguous")
            })?;
        let room_part_refill_from_room_distance = room_part_refill_from_room_distance
            .as_slice_mut()
            .map_err(|_| {
                PyValueError::new_err("room_part_refill_from_room_distance must be contiguous")
            })?;
        let room_part_refill_to_room_distance = room_part_refill_to_room_distance
            .as_slice_mut()
            .map_err(|_| {
                PyValueError::new_err("room_part_refill_to_room_distance must be contiguous")
            })?;
        let room_part_frontier_from_room_distance = room_part_frontier_from_room_distance
            .as_slice_mut()
            .map_err(|_| {
                PyValueError::new_err("room_part_frontier_from_room_distance must be contiguous")
            })?;
        let room_part_frontier_to_room_distance = room_part_frontier_to_room_distance
            .as_slice_mut()
            .map_err(|_| {
                PyValueError::new_err("room_part_frontier_to_room_distance must be contiguous")
            })?;
        let known_save_from_room_distance =
            known_save_from_room_distance.as_slice_mut().map_err(|_| {
                PyValueError::new_err("known_save_from_room_distance must be contiguous")
            })?;
        let known_save_to_room_distance = known_save_to_room_distance
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("known_save_to_room_distance must be contiguous"))?;
        let known_refill_from_room_distance = known_refill_from_room_distance
            .as_slice_mut()
            .map_err(|_| {
                PyValueError::new_err("known_refill_from_room_distance must be contiguous")
            })?;
        let known_refill_to_room_distance =
            known_refill_to_room_distance.as_slice_mut().map_err(|_| {
                PyValueError::new_err("known_refill_to_room_distance must be contiguous")
            })?;
        let frontier = frontier
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("frontier must be contiguous"))?;
        let frontier_door_variant = frontier_door_variant
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("frontier_door_variant must be contiguous"))?;
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
        let missing_connect_query_snapshot_idx = missing_connect_query_snapshot_idx
            .as_slice_mut()
            .map_err(|_| {
                PyValueError::new_err("missing_connect_query_snapshot_idx must be contiguous")
            })?;
        let missing_connect_query_connection_idx = missing_connect_query_connection_idx
            .as_slice_mut()
            .map_err(|_| {
                PyValueError::new_err("missing_connect_query_connection_idx must be contiguous")
            })?;
        let missing_connect_query_source_frontier = missing_connect_query_source_frontier
            .as_slice_mut()
            .map_err(|_| {
                PyValueError::new_err("missing_connect_query_source_frontier must be contiguous")
            })?;
        let missing_connect_query_target_frontier = missing_connect_query_target_frontier
            .as_slice_mut()
            .map_err(|_| {
                PyValueError::new_err("missing_connect_query_target_frontier must be contiguous")
            })?;
        let missing_connect_query_source_distance = missing_connect_query_source_distance
            .as_slice_mut()
            .map_err(|_| {
                PyValueError::new_err("missing_connect_query_source_distance must be contiguous")
            })?;
        let missing_connect_query_target_distance = missing_connect_query_target_distance
            .as_slice_mut()
            .map_err(|_| {
                PyValueError::new_err("missing_connect_query_target_distance must be contiguous")
            })?;
        let missing_connect_query_current_distance = missing_connect_query_current_distance
            .as_slice_mut()
            .map_err(|_| {
                PyValueError::new_err("missing_connect_query_current_distance must be contiguous")
            })?;
        let save_refill_utility_query_snapshot_idx = save_refill_utility_query_snapshot_idx
            .as_slice_mut()
            .map_err(|_| {
                PyValueError::new_err("save_refill_utility_query_snapshot_idx must be contiguous")
            })?;
        let save_refill_utility_query_room_part_idx = save_refill_utility_query_room_part_idx
            .as_slice_mut()
            .map_err(|_| {
                PyValueError::new_err("save_refill_utility_query_room_part_idx must be contiguous")
            })?;
        let save_refill_utility_query_target_mask = save_refill_utility_query_target_mask
            .as_slice_mut()
            .map_err(|_| {
                PyValueError::new_err("save_refill_utility_query_target_mask must be contiguous")
            })?;
        let save_refill_utility_query_frontier = save_refill_utility_query_frontier
            .as_slice_mut()
            .map_err(|_| {
                PyValueError::new_err("save_refill_utility_query_frontier must be contiguous")
            })?;
        let save_refill_utility_query_frontier_distance =
            save_refill_utility_query_frontier_distance
                .as_slice_mut()
                .map_err(|_| {
                    PyValueError::new_err(
                        "save_refill_utility_query_frontier_distance must be contiguous",
                    )
                })?;
        let save_refill_utility_query_save_to_current_distance =
            save_refill_utility_query_save_to_current_distance
                .as_slice_mut()
                .map_err(|_| {
                    PyValueError::new_err(
                        "save_refill_utility_query_save_to_current_distance must be contiguous",
                    )
                })?;
        let save_refill_utility_query_save_from_current_distance =
            save_refill_utility_query_save_from_current_distance
                .as_slice_mut()
                .map_err(|_| {
                    PyValueError::new_err(
                        "save_refill_utility_query_save_from_current_distance must be contiguous",
                    )
                })?;
        let save_refill_utility_query_refill_to_current_distance =
            save_refill_utility_query_refill_to_current_distance
                .as_slice_mut()
                .map_err(|_| {
                    PyValueError::new_err(
                        "save_refill_utility_query_refill_to_current_distance must be contiguous",
                    )
                })?;
        let save_refill_utility_query_refill_from_current_distance =
            save_refill_utility_query_refill_from_current_distance
                .as_slice_mut()
                .map_err(|_| {
                    PyValueError::new_err(
                        "save_refill_utility_query_refill_from_current_distance must be contiguous",
                    )
                })?;
        let toilet_crossed_room_idx = toilet_crossed_room_idx
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("toilet_crossed_room_idx must be contiguous"))?;
        let row_snapshot_idx = row_snapshot_idx
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("row_snapshot_idx must be contiguous"))?;
        let row_frontier_idx = row_frontier_idx
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("row_frontier_idx must be contiguous"))?;
        let row_door_output_idx = row_door_output_idx
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("row_door_output_idx must be contiguous"))?;

        if worker_frontier_row_counts.len() != self.workers.len() {
            return Err(PyValueError::new_err(
                "worker frontier row count length does not match worker count",
            ));
        }
        if worker_missing_connect_query_row_counts.len() != self.workers.len() {
            return Err(PyValueError::new_err(
                "worker missing connect query row count length does not match worker count",
            ));
        }
        if worker_save_refill_utility_query_row_counts.len() != self.workers.len() {
            return Err(PyValueError::new_err(
                "worker save/refill utility query row count length does not match worker count",
            ));
        }

        let (actual_frontier_row_count, _) = py.detach(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            let mut frontier_row_start = 0;
            let mut missing_connect_query_row_start = 0;
            let mut save_refill_utility_query_row_start = 0;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                let start = max(environment_start, worker.start);
                let end = min(environment_start + environment_count, worker.end());
                if start >= end {
                    continue;
                }
                let snapshot_start = (start - environment_start) * candidate_count;
                let snapshot_count = (end - start) * candidate_count;
                let worker_frontier_row_count = worker_frontier_row_counts[worker_idx];
                let worker_missing_connect_query_row_count =
                    worker_missing_connect_query_row_counts[worker_idx];
                let worker_save_refill_utility_query_row_count =
                    worker_save_refill_utility_query_row_counts[worker_idx];
                let outputs = FeatureOutputShards {
                    global: GlobalFeatureOutputShards {
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
                        room_part_furthest_destination: OutputShard::from_slice(
                            &mut room_part_furthest_destination[snapshot_start
                                * room_part_furthest_width
                                ..(snapshot_start + snapshot_count) * room_part_furthest_width],
                        ),
                        room_part_furthest_source: OutputShard::from_slice(
                            &mut room_part_furthest_source[snapshot_start * room_part_furthest_width
                                ..(snapshot_start + snapshot_count) * room_part_furthest_width],
                        ),
                        room_part_save_from_room_distance: OutputShard::from_slice(
                            &mut room_part_save_from_room_distance[snapshot_start
                                * room_part_save_distance_width
                                ..(snapshot_start + snapshot_count)
                                    * room_part_save_distance_width],
                        ),
                        room_part_save_to_room_distance: OutputShard::from_slice(
                            &mut room_part_save_to_room_distance[snapshot_start
                                * room_part_save_distance_width
                                ..(snapshot_start + snapshot_count)
                                    * room_part_save_distance_width],
                        ),
                        room_part_refill_from_room_distance: OutputShard::from_slice(
                            &mut room_part_refill_from_room_distance[snapshot_start
                                * room_part_refill_distance_width
                                ..(snapshot_start + snapshot_count)
                                    * room_part_refill_distance_width],
                        ),
                        room_part_refill_to_room_distance: OutputShard::from_slice(
                            &mut room_part_refill_to_room_distance[snapshot_start
                                * room_part_refill_distance_width
                                ..(snapshot_start + snapshot_count)
                                    * room_part_refill_distance_width],
                        ),
                        room_part_frontier_from_room_distance: OutputShard::from_slice(
                            &mut room_part_frontier_from_room_distance[snapshot_start
                                * room_part_frontier_distance_width
                                ..(snapshot_start + snapshot_count)
                                    * room_part_frontier_distance_width],
                        ),
                        room_part_frontier_to_room_distance: OutputShard::from_slice(
                            &mut room_part_frontier_to_room_distance[snapshot_start
                                * room_part_frontier_distance_width
                                ..(snapshot_start + snapshot_count)
                                    * room_part_frontier_distance_width],
                        ),
                        known_save_from_room_distance: OutputShard::from_slice(
                            &mut known_save_from_room_distance[snapshot_start * known_distance_width
                                ..(snapshot_start + snapshot_count) * known_distance_width],
                        ),
                        known_save_to_room_distance: OutputShard::from_slice(
                            &mut known_save_to_room_distance[snapshot_start * known_distance_width
                                ..(snapshot_start + snapshot_count) * known_distance_width],
                        ),
                        known_refill_from_room_distance: OutputShard::from_slice(
                            &mut known_refill_from_room_distance[snapshot_start
                                * known_distance_width
                                ..(snapshot_start + snapshot_count) * known_distance_width],
                        ),
                        known_refill_to_room_distance: OutputShard::from_slice(
                            &mut known_refill_to_room_distance[snapshot_start * known_distance_width
                                ..(snapshot_start + snapshot_count) * known_distance_width],
                        ),
                        connection_reachability: OutputShard::from_slice(
                            &mut connection_reachability[snapshot_start
                                * connection_reachability_width
                                ..(snapshot_start + snapshot_count)
                                    * connection_reachability_width],
                        ),
                        toilet_crossed_room_idx: OutputShard::from_slice(
                            &mut toilet_crossed_room_idx[snapshot_start * toilet_crossed_room_width
                                ..(snapshot_start + snapshot_count) * toilet_crossed_room_width],
                        ),
                        inventory_count: inventory_width,
                        room_count: room_width,
                        room_part_furthest_count: room_part_furthest_width,
                        room_part_save_distance_count: room_part_save_distance_width,
                        room_part_refill_distance_count: room_part_refill_distance_width,
                        room_part_frontier_distance_count: room_part_frontier_distance_width,
                        known_distance_count: known_distance_width,
                        connection_count: connection_reachability_width,
                        toilet_crossed_room_count: toilet_crossed_room_width,
                    },
                    frontier_rows: FrontierFeatureOutputShards {
                        frontier: OutputShard::from_slice(
                            &mut frontier[frontier_row_start * FEATURE_FRONTIER_WIDTH
                                ..(frontier_row_start + worker_frontier_row_count)
                                    * FEATURE_FRONTIER_WIDTH],
                        ),
                        frontier_door_variant: OutputShard::from_slice(
                            &mut frontier_door_variant[frontier_row_start
                                ..frontier_row_start + worker_frontier_row_count],
                        ),
                        frontier_occupancy: OutputShard::from_slice(
                            &mut frontier_occupancy[frontier_row_start * frontier_occupancy_width
                                ..(frontier_row_start + worker_frontier_row_count)
                                    * frontier_occupancy_width],
                        ),
                        frontier_neighbor: OutputShard::from_slice(
                            &mut frontier_neighbor[frontier_row_start * frontier_neighbor_width
                                ..(frontier_row_start + worker_frontier_row_count)
                                    * frontier_neighbor_width],
                        ),
                        frontier_neighbor_pair: OutputShard::from_slice(
                            &mut frontier_neighbor_pair[frontier_row_start
                                * frontier_neighbor_pair_width
                                ..(frontier_row_start + worker_frontier_row_count)
                                    * frontier_neighbor_pair_width],
                        ),
                        frontier_connection_reachability: OutputShard::from_slice(
                            &mut frontier_connection_reachability[frontier_row_start
                                * frontier_connection_width
                                ..(frontier_row_start + worker_frontier_row_count)
                                    * frontier_connection_width],
                        ),
                        frontier_neighbor_count: self.frontier_neighbor_count,
                        connection_count: frontier_connection_width,
                        frontier_window_size: self.frontier_window_size,
                    },
                    row_snapshot_idx: OutputShard::from_slice(
                        &mut row_snapshot_idx
                            [frontier_row_start..frontier_row_start + worker_frontier_row_count],
                    ),
                    row_frontier_idx: OutputShard::from_slice(
                        &mut row_frontier_idx
                            [frontier_row_start..frontier_row_start + worker_frontier_row_count],
                    ),
                    row_door_output_idx: OutputShard::from_slice(
                        &mut row_door_output_idx
                            [frontier_row_start..frontier_row_start + worker_frontier_row_count],
                    ),
                    missing_connect_query_snapshot_idx: OutputShard::from_slice(
                        &mut missing_connect_query_snapshot_idx[missing_connect_query_row_start
                            ..missing_connect_query_row_start
                                + worker_missing_connect_query_row_count],
                    ),
                    missing_connect_query_connection_idx: OutputShard::from_slice(
                        &mut missing_connect_query_connection_idx[missing_connect_query_row_start
                            ..missing_connect_query_row_start
                                + worker_missing_connect_query_row_count],
                    ),
                    missing_connect_query_source_frontier: OutputShard::from_slice(
                        &mut missing_connect_query_source_frontier[missing_connect_query_row_start
                            * missing_connect_query_frontier_width
                            ..(missing_connect_query_row_start
                                + worker_missing_connect_query_row_count)
                                * missing_connect_query_frontier_width],
                    ),
                    missing_connect_query_target_frontier: OutputShard::from_slice(
                        &mut missing_connect_query_target_frontier[missing_connect_query_row_start
                            * missing_connect_query_frontier_width
                            ..(missing_connect_query_row_start
                                + worker_missing_connect_query_row_count)
                                * missing_connect_query_frontier_width],
                    ),
                    missing_connect_query_source_distance: OutputShard::from_slice(
                        &mut missing_connect_query_source_distance[missing_connect_query_row_start
                            * missing_connect_query_frontier_width
                            ..(missing_connect_query_row_start
                                + worker_missing_connect_query_row_count)
                                * missing_connect_query_frontier_width],
                    ),
                    missing_connect_query_target_distance: OutputShard::from_slice(
                        &mut missing_connect_query_target_distance[missing_connect_query_row_start
                            * missing_connect_query_frontier_width
                            ..(missing_connect_query_row_start
                                + worker_missing_connect_query_row_count)
                                * missing_connect_query_frontier_width],
                    ),
                    missing_connect_query_current_distance: OutputShard::from_slice(
                        &mut missing_connect_query_current_distance[missing_connect_query_row_start
                            ..missing_connect_query_row_start
                                + worker_missing_connect_query_row_count],
                    ),
                    save_refill_utility_query_snapshot_idx: OutputShard::from_slice(
                        &mut save_refill_utility_query_snapshot_idx
                            [save_refill_utility_query_row_start
                                ..save_refill_utility_query_row_start
                                    + worker_save_refill_utility_query_row_count],
                    ),
                    save_refill_utility_query_room_part_idx: OutputShard::from_slice(
                        &mut save_refill_utility_query_room_part_idx
                            [save_refill_utility_query_row_start
                                ..save_refill_utility_query_row_start
                                    + worker_save_refill_utility_query_row_count],
                    ),
                    save_refill_utility_query_target_mask: OutputShard::from_slice(
                        &mut save_refill_utility_query_target_mask
                            [save_refill_utility_query_row_start
                                ..save_refill_utility_query_row_start
                                    + worker_save_refill_utility_query_row_count],
                    ),
                    save_refill_utility_query_frontier: OutputShard::from_slice(
                        &mut save_refill_utility_query_frontier[save_refill_utility_query_row_start
                            ..save_refill_utility_query_row_start
                                + worker_save_refill_utility_query_row_count],
                    ),
                    save_refill_utility_query_frontier_distance: OutputShard::from_slice(
                        &mut save_refill_utility_query_frontier_distance
                            [save_refill_utility_query_row_start
                                ..save_refill_utility_query_row_start
                                    + worker_save_refill_utility_query_row_count],
                    ),
                    save_refill_utility_query_save_to_current_distance: OutputShard::from_slice(
                        &mut save_refill_utility_query_save_to_current_distance
                            [save_refill_utility_query_row_start
                                ..save_refill_utility_query_row_start
                                    + worker_save_refill_utility_query_row_count],
                    ),
                    save_refill_utility_query_save_from_current_distance: OutputShard::from_slice(
                        &mut save_refill_utility_query_save_from_current_distance
                            [save_refill_utility_query_row_start
                                ..save_refill_utility_query_row_start
                                    + worker_save_refill_utility_query_row_count],
                    ),
                    save_refill_utility_query_refill_to_current_distance: OutputShard::from_slice(
                        &mut save_refill_utility_query_refill_to_current_distance
                            [save_refill_utility_query_row_start
                                ..save_refill_utility_query_row_start
                                    + worker_save_refill_utility_query_row_count],
                    ),
                    save_refill_utility_query_refill_from_current_distance: OutputShard::from_slice(
                        &mut save_refill_utility_query_refill_from_current_distance
                            [save_refill_utility_query_row_start
                                ..save_refill_utility_query_row_start
                                    + worker_save_refill_utility_query_row_count],
                    ),
                    snapshot_start,
                };
                if let Err(err) = worker.send(WorkerCommand::PackFeatures {
                    outputs,
                    expected_snapshot_count: snapshot_count,
                }) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
                frontier_row_start += worker_frontier_row_count;
                missing_connect_query_row_start += worker_missing_connect_query_row_count;
                save_refill_utility_query_row_start += worker_save_refill_utility_query_row_count;
            }
            collect_feature_info(&self.workers, sent_workers, first_error)
        })?;
        if actual_frontier_row_count.frontier_row_count != frontier_row_count {
            return Err(PyRuntimeError::new_err(format!(
                "frontier feature row count changed between passes: expected {frontier_row_count}, got {}",
                actual_frontier_row_count.frontier_row_count
            )));
        }
        if actual_frontier_row_count.missing_connect_query_row_count
            != missing_connect_query_row_count
        {
            return Err(PyRuntimeError::new_err(format!(
                "missing connect query row count changed between passes: expected {missing_connect_query_row_count}, got {}",
                actual_frontier_row_count.missing_connect_query_row_count
            )));
        }
        if actual_frontier_row_count.save_refill_utility_query_row_count
            != save_refill_utility_query_row_count
        {
            return Err(PyRuntimeError::new_err(format!(
                "save/refill utility query row count changed between passes: expected {save_refill_utility_query_row_count}, got {}",
                actual_frontier_row_count.save_refill_utility_query_row_count
            )));
        }
        Ok(())
    }
}
