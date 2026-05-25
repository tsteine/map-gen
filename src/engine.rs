use crate::{Action, CommonData, Coord, Environment, Room, RoomIdx};
use crossbeam_channel as channel;
use numpy::{Element, IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::cmp::{max, min};
use std::marker::PhantomData;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

fn pyarray2_from_flat_vec<'py, T: Element>(
    py: Python<'py>,
    data: Vec<T>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<T>>> {
    data.into_pyarray(py).reshape([rows, cols])
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

    unsafe fn as_mut_slice<'a>(self) -> &'a mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

type ActionRows = Vec<(Vec<RoomIdx>, Vec<Coord>, Vec<Coord>)>;

enum WorkerCommand {
    Clear,
    InitialStep,
    Step {
        local_start: usize,
        actions: Vec<Action>,
    },
    GetCandidates {
        local_start: usize,
        local_len: usize,
        max_candidates: usize,
        room_idx: OutputShard<RoomIdx>,
        room_x: OutputShard<Coord>,
        room_y: OutputShard<Coord>,
    },
    GetActions,
    Shutdown,
}

enum WorkerResponse {
    Done,
    Actions(ActionRows),
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
            WorkerResponse::Actions(_) => Err(PyRuntimeError::new_err(
                "engine worker returned an unexpected response",
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
    while let Ok(command) = command_rx.recv() {
        let response = match command {
            WorkerCommand::Clear => {
                for env in &mut environments {
                    env.clear(&common_data);
                }
                WorkerResponse::Done
            }
            WorkerCommand::InitialStep => {
                for env in &mut environments {
                    env.initial_step(&common_data);
                }
                WorkerResponse::Done
            }
            WorkerCommand::Step {
                local_start,
                actions,
            } => {
                let local_end = local_start + actions.len();
                debug_assert!(local_end <= environments.len());
                for (env, action) in environments[local_start..local_end].iter_mut().zip(actions) {
                    env.step(action, &common_data);
                }
                WorkerResponse::Done
            }
            WorkerCommand::GetCandidates {
                local_start,
                local_len,
                max_candidates,
                room_idx,
                room_x,
                room_y,
            } => {
                let local_end = local_start + local_len;
                debug_assert!(local_end <= environments.len());
                let room_idx = unsafe { room_idx.as_mut_slice() };
                let room_x = unsafe { room_x.as_mut_slice() };
                let room_y = unsafe { room_y.as_mut_slice() };
                debug_assert_eq!(room_idx.len(), local_len * max_candidates);
                debug_assert_eq!(room_x.len(), local_len * max_candidates);
                debug_assert_eq!(room_y.len(), local_len * max_candidates);

                for (env_idx, env) in environments[local_start..local_end].iter_mut().enumerate() {
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
            WorkerCommand::GetActions => {
                let mut rows = Vec::with_capacity(environments.len());
                for env in &environments {
                    rows.push((
                        env.actions.iter().map(|action| action.room_idx).collect(),
                        env.actions.iter().map(|action| action.x).collect(),
                        env.actions.iter().map(|action| action.y).collect(),
                    ));
                }
                WorkerResponse::Actions(rows)
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

fn checked_range_end(start: usize, len: usize) -> PyResult<usize> {
    start.checked_add(len).ok_or_else(|| {
        PyValueError::new_err(format!(
            "range start {start} with length {len} overflows usize"
        ))
    })
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

#[pyclass]
pub struct Engine {
    common_data: Arc<CommonData>, // pre-computed data that can be shared across environments
    workers: Vec<WorkerHandle>,   // fixed worker-owned environment shards
    num_environments: usize,
}

impl Drop for Engine {
    fn drop(&mut self) {
        for worker in &mut self.workers {
            worker.shutdown();
        }
    }
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (rooms_json, map_size, num_environments, seed, num_threads=None))]
    fn new(
        rooms_json: &str,
        map_size: (Coord, Coord),
        num_environments: usize,
        seed: u64,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
        let requested_threads = requested_num_threads(num_threads)?;
        let worker_count = min(requested_threads, max(num_environments, 1));
        let rooms: Vec<Room> = serde_json::from_str(rooms_json)
            .map_err(|err| PyValueError::new_err(format!("failed to parse rooms JSON: {err}")))?;
        let common_data = Arc::new(CommonData::new(rooms)?);

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
        })
    }

    fn clear(&mut self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
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
        })
    }

    fn initial_step(&mut self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                if let Err(err) = worker.send(WorkerCommand::InitialStep) {
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
        let rows = py.allow_threads(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                if let Err(err) = worker.send(WorkerCommand::GetActions) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }

            let mut rows = Vec::with_capacity(self.num_environments);
            for worker_idx in sent_workers {
                match self.workers[worker_idx].recv() {
                    Ok(WorkerResponse::Actions(worker_rows)) => rows.extend(worker_rows),
                    Ok(WorkerResponse::Done) => {
                        set_first_error(
                            &mut first_error,
                            PyRuntimeError::new_err(
                                "engine worker returned an unexpected response",
                            ),
                        );
                    }
                    Err(err) => set_first_error(&mut first_error, err),
                }
            }

            if let Some(err) = first_error {
                Err(err)
            } else {
                Ok(rows)
            }
        })?;

        let mut room_idx = Vec::with_capacity(rows.len());
        let mut room_x = Vec::with_capacity(rows.len());
        let mut room_y = Vec::with_capacity(rows.len());
        for (idx_row, x_row, y_row) in rows {
            room_idx.push(idx_row);
            room_x.push(x_row);
            room_y.push(y_row);
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
        py: Python<'py>,
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

        let end = checked_range_end(start, room_idx.len())?;
        if end > self.num_environments {
            return Err(PyValueError::new_err(format!(
                "action arrays with length {} starting at {} exceed num_environments {}",
                room_idx.len(),
                start,
                self.num_environments,
            )));
        }

        let actions: Vec<_> = room_idx
            .iter()
            .zip(room_x.iter())
            .zip(room_y.iter())
            .map(|((&room_idx, &x), &y)| Action { room_idx, x, y })
            .collect();

        py.allow_threads(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                let overlap_start = max(start, worker.start);
                let overlap_end = min(end, worker.end());
                if overlap_start >= overlap_end {
                    continue;
                }
                let action_start = overlap_start - start;
                let action_end = overlap_end - start;
                if let Err(err) = worker.send(WorkerCommand::Step {
                    local_start: overlap_start - worker.start,
                    actions: actions[action_start..action_end].to_vec(),
                }) {
                    set_first_error(&mut first_error, err);
                    break;
                }
                sent_workers.push(worker_idx);
            }

            wait_for_done_responses(&self.workers, sent_workers, first_error)
        })
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
        if start > end || end > self.num_environments {
            return Err(PyValueError::new_err(format!(
                "environment range [{}, {}) is invalid for num_environments {}",
                start, end, self.num_environments
            )));
        }

        let num_environments = end - start;
        let output_len = num_environments
            .checked_mul(max_candidates)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "candidate output shape ({num_environments}, {max_candidates}) is too large"
                ))
            })?;
        let dummy_candidate = Action {
            room_idx: self.common_data.room.len() as RoomIdx, // an invalid room index to indicate no-op
            x: 0,
            y: 0,
        };

        let mut room_idx = vec![dummy_candidate.room_idx; output_len];
        let mut room_x = vec![dummy_candidate.x; output_len];
        let mut room_y = vec![dummy_candidate.y; output_len];

        py.allow_threads(|| {
            let mut sent_workers = Vec::with_capacity(self.workers.len());
            let mut first_error = None;
            for (worker_idx, worker) in self.workers.iter().enumerate() {
                let overlap_start = max(start, worker.start);
                let overlap_end = min(end, worker.end());
                if overlap_start >= overlap_end {
                    continue;
                }

                let output_row_start = overlap_start - start;
                let local_len = overlap_end - overlap_start;
                let output_start = output_row_start * max_candidates;
                let output_end = output_start + local_len * max_candidates;

                if let Err(err) = worker.send(WorkerCommand::GetCandidates {
                    local_start: overlap_start - worker.start,
                    local_len,
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
            pyarray2_from_flat_vec(py, room_idx, num_environments, max_candidates)?,
            pyarray2_from_flat_vec(py, room_x, num_environments, max_candidates)?,
            pyarray2_from_flat_vec(py, room_y, num_environments, max_candidates)?,
        ))
    }
}
