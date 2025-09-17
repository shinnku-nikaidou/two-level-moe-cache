//! Python interface methods for the cache manager
//!
//! This module contains all PyO3 methods and Python-facing interfaces
//! for the TwoTireWmExpertCacheManager.

use super::manager::TwoTireWmExpertCacheManager;
use crate::types::{expert::ExpertKey, model::ModelType, status::ExpertStatus};
use policy::watermark::MemoryTier;
use pyo3::prelude::*;

#[pymethods]
impl TwoTireWmExpertCacheManager {
    #[new]
    pub fn py_new(
        model_type: ModelType,
        vram_capacity: usize,
        ram_capacity: usize,
    ) -> PyResult<Self> {
        Self::new(model_type, vram_capacity, ram_capacity)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    /// Get expert by key - now delegates to actual watermark algorithm
    pub fn get(&mut self, _expert_key: ExpertKey) -> PyResult<()> {
        // For now, we don't have EWMA/ScoutGate predictions to work with
        // This method would need to be redesigned once we integrate those components
        // Current implementation is a placeholder that compiles
        panic!("No need to implement get() here;");
    }

    /// Update with new layer activations - placeholder for integration with EWMA/ScoutGate
    pub fn update_activations(&mut self, activated_experts: Vec<usize>) -> PyResult<()> {
        // This would update EWMA predictors and ScoutGate with activation data
        // For now it's a placeholder
        Ok(())
    }

    /// Advance to next time step
    pub fn step_forward(&mut self) -> PyResult<()> {
        self.current_time += 1;
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
        todo!()
    }

    /// Get total layers
    pub fn total_layers(&self) -> usize {
        todo!()
    }

    /// Get watermark values for debugging
    pub fn get_watermarks(&self) -> (f64, f64) {
        self.watermark_algorithm.get_watermarks()
    }

    /// Get memory usage for debugging
    pub fn get_memory_usage(&self) -> (usize, usize) {
        self.watermark_algorithm.get_memory_usage()
    }

    /// Get simplified status of all tracked experts
    pub fn experts_status(&self) -> Vec<ExpertStatus> {
        self.watermark_algorithm
            .expert_states()
            .iter()
            .map(|(_, expert_state)| {
                let tier_u8 = match expert_state.current_tier {
                    MemoryTier::VRAM => 0,
                    MemoryTier::RAM => 1,
                    MemoryTier::Disk => 2,
                };
                
                // Convert policy::ExpertKey to core::types::expert::ExpertKey
                let core_expert_key = ExpertKey::new(
                    expert_state.expert_key.layer_id,
                    expert_state.expert_key.expert_id,
                    expert_state.expert_key.param_type.into(),
                );
                
                ExpertStatus::new(core_expert_key, tier_u8)
            })
            .collect()
    }
}
