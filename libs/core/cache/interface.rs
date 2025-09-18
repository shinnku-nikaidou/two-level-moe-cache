//! Python interface methods for the cache manager
//!
//! This module contains all PyO3 methods and Python-facing interfaces
//! for the TwoTireWmExpertCacheManager.

use super::manager::RustTwoTireWmExpertCacheManager;
use crate::types::{expert::RustExpertKey, model::RustModelType, status::RustExpertStatus};
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

    /// Update with new layer activations - executes complete data flow pipeline
    pub fn update_activations(&mut self, activated_experts: Vec<usize>) -> PyResult<()> {
        // Step 1: Update EWMA predictor with current layer activations
        self.ewma_predictor
            .update_layer_activations(&activated_experts)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "EWMA update error: {}",
                    e
                ))
            })?;

        // Step 2: Update ScoutGate with placeholder token
        // TODO: In a real implementation, this would receive the actual token from the inference context
        let placeholder_token = 0u32; // Placeholder - should come from actual token stream
        self.scoutgate_predictor
            .update_token_context(placeholder_token)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "ScoutGate update error: {}",
                    e
                ))
            })?;

        // Step 3: Get predictions from both components
        let ewma_predictions = self.ewma_predictor.get_all_probabilities();
        let scoutgate_predictions = self.scoutgate_predictor.get_probabilities();

        // Step 4: Fuse probabilities with forward-causal weights
        let fused_predictions = self
            .probability_fuser
            .fuse(ewma_predictions, scoutgate_predictions)
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Probability fusion error: {}",
                    e
                ))
            })?;

        // Step 5: Make cache decisions using watermark algorithm
        // Note: Currently watermark_algorithm.make_cache_decisions() returns ExpertState,
        // but we don't need to expose it through this interface since Python side
        // will call experts_status() separately to get the current cache state.
        let _cache_decisions = self
            .watermark_algorithm
            .make_cache_decisions(&fused_predictions);

        Ok(())
    }

    /// Advance to next time step - simple timer advancement
    pub fn step_forward(&mut self) -> PyResult<()> {
        // Simply advance the timer to next step
        let mut timer = self.timer.write().unwrap();
        timer.step();
        Ok(())
    }

    /// Get simplified status of all tracked experts
    pub fn experts_status(&self) -> Vec<RustExpertStatus> {
        todo!()
        // self.watermark_algorithm
        //     .expert_states()
        //     .iter()
        //     .flat_map(|(expert, memory_tier)| {
        //         let tier_u8 = match memory_tier {
        //             MemoryTier::Vram => 0,
        //             MemoryTier::Ram => 1,
        //             MemoryTier::Disk => 2,
        //         };

        //         // Generate RustExpertKey for all 4 parameter types per expert
        //         vec![
        //             crate::types::expert::RustExpertParamType::MLP1_WEIGHT,
        //             crate::types::expert::RustExpertParamType::MLP1_BIAS,
        //             crate::types::expert::RustExpertParamType::MLP2_WEIGHT,
        //             crate::types::expert::RustExpertParamType::MLP2_BIAS,
        //         ]
        //         .into_iter()
        //         .map(move |param_type| {
        //             let core_expert_key =
        //                 RustExpertKey::new(expert.layer_id, expert.expert_id, param_type);
        //             RustExpertStatus::new(core_expert_key, tier_u8)
        //         })
        //     })
        //     .collect()
    }
}
