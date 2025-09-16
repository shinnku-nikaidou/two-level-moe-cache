// Core library module for two-level MoE cache

use pyo3::prelude::*;

/// Add two numbers together
#[pyfunction]
fn add(a: i64, b: i64) -> i64 {
    a + b
}

/// Get the version of the core library
#[pyfunction]
fn get_version() -> String {
    "0.1.0".to_string()
}

/// Core data structures and functionality
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

/// Python wrapper for CoreCache
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
pub fn core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    m.add_class::<PyCoreCache>()?;
    Ok(())
}
