//! Type conversion utilities for Python-Rust interoperability
//!
//! This module provides utility functions for converting between Python and Rust
//! types, along with error handling helpers.

use pyo3::prelude::*;

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
