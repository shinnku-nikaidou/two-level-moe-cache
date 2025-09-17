//! Core library module for two-level MoE cache
//!
//! This library provides high-performance Rust implementations of caching algorithms
//! for Mixture-of-Experts models, with Python bindings via PyO3.

use pyo3::prelude::*;

// Module declarations
pub mod cache;
pub mod types;
pub mod utils;

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "rust_core")]
pub fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // New two-tier watermark cache types
    m.add_class::<types::memory::MemoryTier>()?;
    m.add_class::<types::expert::ExpertKey>()?;
    m.add_class::<types::config::WatermarkConfig>()?;
    m.add_class::<types::expert::ExpertParamType>()?;
    m.add_class::<cache::manager::TwoTireWmExpertCacheManager>()?;

    Ok(())
}
