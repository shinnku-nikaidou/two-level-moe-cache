//! Python interface methods for the cache manager
//!
//! This module contains all PyO3 methods and Python-facing interfaces
//! for the TwoTireWmExpertCacheManager.

use super::manager::RustTwoTireWmExpertCacheManager;
use crate::types::{
    expert::{RustExpertKey, RustExpertParamType},
    model::RustModelType,
    status::RustExpertStatus,
};
use policy::watermark::algorithm::MemoryTier;
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

        self.scoutgate_predictor
            .update_predictions()
            .py_context("ScoutGate update error")?;

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

        // Pre-allocate result vector with exact capacity
        // Each layer × experts_per_layer × 4_param_types = total entries
        let total_layers = expert_state.inner.len();
        let experts_per_layer = if total_layers > 0 {
            expert_state.inner[0].len()
        } else {
            panic!("No layers in expert state")
        };
        let total_entries = total_layers * experts_per_layer * 4;
        let mut result = Vec::with_capacity(total_entries);

        // Parameter types for each expert (all 4 MLP parameters)
        let param_types = [
            RustExpertParamType::MLP1_WEIGHT,
            RustExpertParamType::MLP1_BIAS,
            RustExpertParamType::MLP2_WEIGHT,
            RustExpertParamType::MLP2_BIAS,
        ];

        // Simple nested loops for clear logic
        for (layer_idx, layer_experts) in expert_state.inner.iter().enumerate() {
            for (expert_idx, &memory_tier) in layer_experts.iter().enumerate() {
                // Convert MemoryTier enum to u8 for Python interface
                let tier_u8 = match memory_tier {
                    MemoryTier::Vram => 0,
                    MemoryTier::Ram => 1,
                    MemoryTier::Disk => 2,
                };

                // Generate status for all 4 parameter types per expert
                for param_type in param_types {
                    let expert_key = RustExpertKey::new(layer_idx, expert_idx, param_type);
                    result.push(RustExpertStatus::new(expert_key, tier_u8));
                }
            }
        }

        result
    }
}
