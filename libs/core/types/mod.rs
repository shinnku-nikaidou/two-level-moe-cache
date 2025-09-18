//! Python type conversions for PyO3 bindings
//!
//! This module provides conversions between Python types used in the domain layer
//! and Rust types used in the core implementation.

use pyo3::prelude::*;

pub mod expert;
pub mod memory;
pub mod model;
pub mod status;

// Type conversion utilities for Python-Rust interoperability

/// Extension trait to add convenient error conversion methods
pub trait PyResultExt<T> {
    /// Convert a Result to a PyResult with enhanced error context
    fn py_context(self, context: &str) -> PyResult<T>;
}

impl<T, E: std::fmt::Display> PyResultExt<T> for Result<T, E> {
    fn py_context(self, context: &str) -> PyResult<T> {
        self.map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}: {}", context, e))
        })
    }
}

/// Error handling utilities for type conversions
pub trait ConversionResult<T> {
    /// Convert a Result to a PyResult with enhanced error context
    fn with_context(self, context: &str) -> PyResult<T>;
}

impl<T, E> ConversionResult<T> for Result<T, E>
where
    E: std::fmt::Display,
{
    fn with_context(self, context: &str) -> PyResult<T> {
        self.map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}: {}", context, e))
        })
    }
}

/// Utility function to validate numeric ranges for Python parameters
pub fn validate_range<T>(value: T, min: T, max: T, name: &str) -> PyResult<T>
where
    T: PartialOrd + std::fmt::Display + Copy,
{
    if value < min || value > max {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{} must be in range [{}, {}], got {}",
            name, min, max, value
        )));
    }
    Ok(value)
}

/// Utility function to validate positive numeric values
pub fn validate_positive<T>(value: T, name: &str) -> PyResult<T>
where
    T: PartialOrd + std::fmt::Display + Copy + From<i32>,
{
    if value <= T::from(0) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "{} must be positive, got {}",
            name, value
        )));
    }
    Ok(value)
}
