//! Python interface methods for the cache manager
//!
//! This module contains all PyO3 methods and Python-facing interfaces
//! for the TwoTireWmExpertCacheManager.

use super::manager::RustTwoTireWmExpertCacheManager;
use crate::types::{expert::RustExpertKey, model::RustModelType, status::RustExpertStatus};
use pyo3::prelude::*;

/// Extension trait to add convenient error conversion methods
trait PyResultExt<T> {
    fn py_context(self, context: &str) -> PyResult<T>;
}

impl<T, E: std::fmt::Display> PyResultExt<T> for Result<T, E> {
    fn py_context(self, context: &str) -> PyResult<T> {
        self.map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}: {}", context, e))
        })
    }
}

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
            .py_context("EWMA update error")?;

        // Step 2: Get predictions from both components
        let ewma_predictions = self.ewma_predictor.get_probabilities();
        let scoutgate_predictions = self.scoutgate_predictor.get_probabilities();

        // Step 3: Fuse probabilities with forward-causal weights
        let fused_predictions = self
            .probability_fuser
            .fuse(ewma_predictions, scoutgate_predictions)
            .py_context("Probability fusion error")?;

        // Step 4: Make cache decisions using watermark algorithm
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
        {
            let mut timer = self.timer.write().unwrap();
            timer.step();
        }

        // TODO: In a real implementation, this would receive the actual token from the inference context
        let placeholder_token = 0u32; // Placeholder - should come from actual token stream
        self.scoutgate_predictor
            .update_token_context(placeholder_token)
            .py_context("ScoutGate update error")?;

        Ok(())
    }

    /// Get current expert status across all memory tiers
    ///
    /// This method provides a complete snapshot of expert-parameter placement
    /// decisions based on current algorithm state. It recomputes the entire
    /// prediction and decision pipeline to get the most up-to-date status.
    ///
    /// # Returns  
    /// Vector of `RustExpertStatus` objects representing current tier assignments.
    /// Each expert generates 4 status entries (one per MLP parameter type).
    ///
    /// # Performance
    /// - Time complexity: O(L × E) for prediction + decision pipeline
    /// - Recomputes predictions and cache decisions each call
    /// - Memory tier encoding: 0=VRAM, 1=RAM, 2=DISK
    ///
    /// # Notes
    /// - Uses current predictor state (EWMA values, ScoutGate context)
    /// - All 4 parameter types per expert have same tier assignment
    pub fn experts_status(&self) -> Vec<RustExpertStatus> {
        let fused_predictions = self
            .probability_fuser
            .fuse(
                self.ewma_predictor.get_probabilities(),
                self.scoutgate_predictor.get_probabilities(),
            )
            .unwrap();

        let expert_state = self
            .watermark_algorithm
            .make_cache_decisions(&fused_predictions);

        expert_state
            .inner
            .iter()
            .enumerate()
            .flat_map(|(layer_idx, layer_experts)| {
                // Process all experts in current layer
                layer_experts
                    .iter()
                    .enumerate()
                    .flat_map(move |(expert_idx, &memory_tier)| {
                        // Convert MemoryTier enum to u8 for Python interface
                        let tier_u8 = match memory_tier {
                            policy::watermark::algorithm::MemoryTier::Vram => 0,
                            policy::watermark::algorithm::MemoryTier::Ram => 1,
                            policy::watermark::algorithm::MemoryTier::Disk => 2,
                        };

                        // Generate status for all 4 parameter types per expert
                        [
                            crate::types::expert::RustExpertParamType::MLP1_WEIGHT,
                            crate::types::expert::RustExpertParamType::MLP1_BIAS,
                            crate::types::expert::RustExpertParamType::MLP2_WEIGHT,
                            crate::types::expert::RustExpertParamType::MLP2_BIAS,
                        ]
                        .into_iter()
                        .map(move |param_type| {
                            let expert_key = RustExpertKey::new(layer_idx, expert_idx, param_type);
                            RustExpertStatus::new(expert_key, tier_u8)
                        })
                    })
            })
            .collect()
    }
}
