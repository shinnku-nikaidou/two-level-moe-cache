//! Configuration types for the watermark algorithm
//!
//! This module defines configuration structures and validation logic
//! for the two-level watermark caching system.

use pyo3::prelude::*;

/// Configuration for watermark algorithm
#[pyclass]
#[derive(Debug, Clone)]
pub struct WatermarkConfig {
    #[pyo3(get, set)]
    pub vram_capacity: usize,
    #[pyo3(get, set)]
    pub ram_capacity: usize,
    #[pyo3(get, set)]
    pub vram_learning_rate: f64, // η_G
    #[pyo3(get, set)]
    pub ram_learning_rate: f64, // η_R
    #[pyo3(get, set)]
    pub fusion_eta: f64, // η for EWMA-ScoutGate fusion
    #[pyo3(get, set)]
    pub reuse_decay_gamma: f64, // γ for reuse distance decay
    #[pyo3(get, set)]
    pub hysteresis_factor: f64, // Factor for admit/evict threshold separation
}

#[pymethods]
impl WatermarkConfig {
    #[new]
    pub fn new(
        vram_capacity: usize,
        ram_capacity: usize,
        vram_learning_rate: Option<f64>,
        ram_learning_rate: Option<f64>,
        fusion_eta: Option<f64>,
        reuse_decay_gamma: Option<f64>,
        hysteresis_factor: Option<f64>,
    ) -> Self {
        Self {
            vram_capacity,
            ram_capacity,
            vram_learning_rate: vram_learning_rate.unwrap_or(0.01),
            ram_learning_rate: ram_learning_rate.unwrap_or(0.01),
            fusion_eta: fusion_eta.unwrap_or(0.5),
            reuse_decay_gamma: reuse_decay_gamma.unwrap_or(0.1),
            hysteresis_factor: hysteresis_factor.unwrap_or(1.1),
        }
    }

    pub fn validate(&self) -> PyResult<()> {
        if self.vram_capacity == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "vram_capacity must be > 0",
            ));
        }
        if self.ram_capacity == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "ram_capacity must be > 0",
            ));
        }
        if self.vram_learning_rate <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "vram_learning_rate must be > 0",
            ));
        }
        if self.ram_learning_rate <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "ram_learning_rate must be > 0",
            ));
        }
        if !(0.0..=1.0).contains(&self.fusion_eta) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "fusion_eta must be in [0, 1]",
            ));
        }
        if self.reuse_decay_gamma <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "reuse_decay_gamma must be > 0",
            ));
        }
        if self.hysteresis_factor < 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "hysteresis_factor must be >= 1.0",
            ));
        }
        Ok(())
    }
}
