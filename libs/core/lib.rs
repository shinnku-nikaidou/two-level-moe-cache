//! Core library module for two-level MoE cache
//!
//! This library provides high-performance Rust implementations of caching algorithms
//! for Mixture-of-Experts models, with Python bindings via PyO3.

use pyo3::prelude::*;

// Module declarations
pub mod utils;
pub mod python_types;
pub mod watermark_cache;

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "rust_core")]
pub fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // New two-tier watermark cache types
    m.add_class::<python_types::MemoryTier>()?;
    m.add_class::<python_types::ExpertKey>()?;
    m.add_class::<python_types::ExpertRef>()?;
    m.add_class::<python_types::WatermarkConfig>()?;
    m.add_class::<python_types::ExpertParamType>()?;
    m.add_class::<watermark_cache::TwoTireWmExpertCacheManager>()?;

    Ok(())
}
