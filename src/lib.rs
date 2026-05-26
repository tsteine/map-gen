use pyo3::prelude::*;

mod common;
mod engine;
mod environment;
mod scc_dag;

use engine::{Engine, EnvironmentGroup};

// The Python module definition. This is the entry point for the Python bindings.
// It exposes the Engine and EnvironmentGroup classes to Python.
#[pymodule]
fn map_gen(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add_class::<EnvironmentGroup>()?;
    Ok(())
}
