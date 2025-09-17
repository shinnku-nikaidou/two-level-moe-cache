//! Python interface methods for the cache manager
//!
//! This module contains all PyO3 methods and Python-facing interfaces
//! for the TwoTireWmExpertCacheManager.

use super::manager::RustTwoTireWmExpertCacheManager;
use crate::types::{expert::RustExpertKey, model::RustModelType, status::RustExpertStatus};
use policy::watermark::algorithm::MemoryTier;
use pyo3::prelude::*;

#[pymethods]
impl RustTwoTireWmExpertCacheManager {
    #[new]
    pub fn py_new(
        model_type: RustModelType,
        vram_capacity: usize,
        ram_capacity: usize,
    ) -> PyResult<Self> {
        Self::new(model_type, vram_capacity, ram_capacity)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    /// Get expert by key - now delegates to actual watermark algorithm
    pub fn get(&mut self, _expert_key: RustExpertKey) -> PyResult<()> {
        // For now, we don't have EWMA/ScoutGate predictions to work with
        // This method would need to be redesigned once we integrate those components
        // Current implementation is a placeholder that compiles
        panic!("No need to implement get() here;");
    }

    /// Update with new layer activations - placeholder for integration with EWMA/ScoutGate
    pub fn update_activations(&mut self, _activated_experts: Vec<usize>) -> PyResult<()> {
        // This would update EWMA predictors and ScoutGate with activation data
        // For now it's a placeholder
        Ok(())
    }

    /// Advance to next time step
    pub fn step_forward(&mut self) -> PyResult<()> {
        self.timer.step();
        Ok(())
    }

    /// Get simplified status of all tracked experts
    pub fn experts_status(&self) -> Vec<RustExpertStatus> {
        self.watermark_algorithm
            .expert_states()
            .iter()
            .flat_map(|(expert, memory_tier)| {
                let tier_u8 = match memory_tier {
                    MemoryTier::Vram => 0,
                    MemoryTier::Ram => 1,
                    MemoryTier::Disk => 2,
                };

                // Generate RustExpertKey for all 4 parameter types per expert
                vec![
                    crate::types::expert::RustExpertParamType::MLP1_WEIGHT,
                    crate::types::expert::RustExpertParamType::MLP1_BIAS,
                    crate::types::expert::RustExpertParamType::MLP2_WEIGHT,
                    crate::types::expert::RustExpertParamType::MLP2_BIAS,
                ]
                .into_iter()
                .map(move |param_type| {
                    let core_expert_key =
                        RustExpertKey::new(expert.layer_id, expert.expert_id, param_type);
                    RustExpertStatus::new(core_expert_key, tier_u8)
                })
            })
            .collect()
    }
}
