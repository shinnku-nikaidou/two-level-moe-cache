//! Core library module for two-level MoE cache
//! 
//! This library provides high-performance Rust implementations of caching algorithms
//! for Mixture-of-Experts models, with Python bindings via PyO3.

use pyo3::prelude::*;

// Module declarations
pub mod python_types;
pub mod watermark_cache;

/// Add two numbers together (legacy function)
#[pyfunction]
fn add(a: i64, b: i64) -> i64 {
    a + b
}

/// Get the version of the core library
#[pyfunction]
fn get_version() -> String {
    "0.1.0".to_string()
}

/// Core data structures and functionality (legacy)
pub struct CoreCache {
    capacity: usize,
}

impl CoreCache {
    pub fn new(capacity: usize) -> Self {
        Self { capacity }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Python wrapper for CoreCache (legacy)
#[pyclass]
struct PyCoreCache {
    inner: CoreCache,
}

#[pymethods]
impl PyCoreCache {
    #[new]
    fn new(capacity: usize) -> Self {
        Self {
            inner: CoreCache::new(capacity),
        }
    }

    fn capacity(&self) -> usize {
        self.inner.capacity()
    }
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "rust_core")]
pub fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Legacy functions
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    m.add_class::<PyCoreCache>()?;
    
    // New two-tier watermark cache types
    m.add_class::<python_types::MemoryTier>()?;
    m.add_class::<python_types::ExpertKey>()?;
    m.add_class::<python_types::ExpertRef>()?;
    m.add_class::<python_types::WatermarkConfig>()?;
    m.add_class::<python_types::ExpertParamType>()?;
    m.add_class::<watermark_cache::TwoTireWmExpertCacheManager>()?;
    
    Ok(())
}
