//! Core library module for two-level MoE cache
//!
//! This library provides high-performance Rust implementations of caching algorithms
//! for Mixture-of-Experts models, with Python bindings via PyO3.

use pyo3::prelude::*;
use std::sync::Once;
use tracing_subscriber::{EnvFilter, FmtSubscriber};

// Module declarations
pub mod cache;
pub mod types;
pub mod utils;

static TRACING_INIT: Once = Once::new();

/// Initialize tracing subscriber for logging
///
/// This should be called once when the Python module is imported.
/// Supports environment variable RUST_LOG for log level control.
///
/// Examples:
/// - RUST_LOG=debug python script.py  # Debug level
/// - RUST_LOG=info python script.py   # Info level
/// - RUST_LOG=trace python script.py  # Trace level
#[pyfunction]
fn init_logging() -> PyResult<()> {
    TRACING_INIT.call_once(|| {
        let subscriber = FmtSubscriber::builder()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .with_target(true)
            .with_thread_ids(false)
            .with_file(true)
            .with_line_number(true)
            .finish();

        tracing::subscriber::set_global_default(subscriber)
            .expect("Failed to set tracing subscriber");
    });

    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "rust_core")]
pub fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize logging when module is imported
    init_logging()?;

    // Export logging initialization function for manual control
    m.add_function(wrap_pyfunction!(init_logging, m)?)?;

    // New two-tier watermark cache types
    m.add_class::<types::memory::RustMemoryTier>()?;
    m.add_class::<types::expert::RustExpertKey>()?;
    m.add_class::<types::expert::RustExpertParamType>()?;
    m.add_class::<types::status::RustExpertStatus>()?;
    m.add_class::<cache::manager::RustTwoTierWmExpertCacheManager>()?;

    Ok(())
}
