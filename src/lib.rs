use pyo3::prelude::*;

mod common;
mod engine;
mod environment;
mod scc_dag;

use engine::{
    AreaOutcomeBuffers, EndOutcomes, Engine, EnvironmentGroup, EpisodeOutcomes, FeatureBuffers,
    ProposalCandidateBuffers, StepOutcomes,
};

#[pyfunction]
fn set_profile_enabled(enabled: bool) {
    engine::set_profile_enabled(enabled);
}

#[pyfunction]
fn reset_profile() {
    engine::reset_profile();
}

#[pyfunction]
fn profile_report() -> Vec<(String, u64, u64)> {
    engine::profile_report()
}

#[pymodule]
fn map_gen(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add_class::<EnvironmentGroup>()?;
    m.add_class::<StepOutcomes>()?;
    m.add_class::<EndOutcomes>()?;
    m.add_class::<AreaOutcomeBuffers>()?;
    m.add_class::<EpisodeOutcomes>()?;
    m.add_class::<ProposalCandidateBuffers>()?;
    m.add_class::<FeatureBuffers>()?;
    m.add_function(wrap_pyfunction!(set_profile_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(reset_profile, m)?)?;
    m.add_function(wrap_pyfunction!(profile_report, m)?)?;
    Ok(())
}
