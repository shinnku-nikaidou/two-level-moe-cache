//! Python interface methods for the cache manager
//!
//! This module contains all PyO3 methods and Python-facing interfaces
//! for the TwoTireWmExpertCacheManager.

use pyo3::prelude::*;

use super::manager::TwoTireWmExpertCacheManager;
use crate::types::{ExpertKey, ModelType};

#[pymethods]
impl TwoTireWmExpertCacheManager {
    #[new]
    pub fn py_new(
        model_type: ModelType,
        total_layers: usize,
        vram_capacity: usize,
        ram_capacity: usize,
    ) -> PyResult<Self> {
        Self::new(model_type, total_layers, vram_capacity, ram_capacity)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    /// Get expert by key - now delegates to actual watermark algorithm
    pub fn get(&mut self, _expert_key: ExpertKey) -> PyResult<()> {
        // For now, we don't have EWMA/ScoutGate predictions to work with
        // This method would need to be redesigned once we integrate those components
        // Current implementation is a placeholder that compiles
        Ok(())
    }

    /// Update with new layer activations - placeholder for integration with EWMA/ScoutGate
    pub fn update_activations(&mut self, _activated_experts: Vec<usize>) -> PyResult<()> {
        // This would update EWMA predictors and ScoutGate with activation data
        // For now it's a placeholder
        Ok(())
    }

    /// Advance to next time step
    pub fn step_forward(&mut self) -> PyResult<()> {
        self.current_time += 1;
        self.current_layer = (self.current_layer + 1) % self.total_layers;
        
        // Step the watermark algorithm
        self.watermark_algorithm.step();
        
        Ok(())
    }

    /// Get current time
    pub fn current_time(&self) -> u64 {
        self.current_time
    }

    /// Get current layer
    pub fn current_layer(&self) -> usize {
        self.current_layer
    }

    /// Get total layers
    pub fn total_layers(&self) -> usize {
        self.total_layers
    }

    /// Get watermark values for debugging
    pub fn get_watermarks(&self) -> (f64, f64) {
        self.watermark_algorithm.get_watermarks()
    }

    /// Get memory usage for debugging
    pub fn get_memory_usage(&self) -> (usize, usize) {
        self.watermark_algorithm.get_memory_usage()
    }
}
