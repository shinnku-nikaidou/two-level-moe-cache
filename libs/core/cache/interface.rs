//! Python interface methods for the cache manager
//!
//! This module contains all PyO3 methods and Python-facing interfaces
//! for the TwoTierWmExpertCacheManager.

use super::manager::RustTwoTierWmExpertCacheManager;
use crate::types::{
    PyResultExt,
    expert::{RustExpertKey, RustExpertParamType},
    model::RustModelType,
    status::RustExpertStatus,
};
use policy::watermark::algorithm::MemoryTier;
use pyo3::prelude::*;
use tracing::{debug, info};

#[pymethods]
impl RustTwoTierWmExpertCacheManager {
    #[new]
    pub fn py_new(
        model_type: RustModelType,
        vram_capacity: f64,
        ram_capacity: f64,
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
        // Cache the currently activated experts for forced VRAM placement
        self.current_activated_experts = activated_experts.clone();

        // Step 1: Update EWMA predictor with current layer activations
        self.ewma
            .update_layer_activations(&activated_experts)
            .py_context("EWMA update error")?;

        self.scoutgate
            .update_predictions()
            .py_context("ScoutGate update error")?;

        // Step 2: Fuse predictions and update watermarks
        let fused_predictions = self
            .fuser
            .fuse(
                self.ewma.get_probabilities(),
                self.scoutgate.get_probabilities(),
            )
            .py_context("Probability fusion error")?;

        // Step 3: Update watermark thresholds based on new predictions
        self.watermark.update_watermarks(&fused_predictions);

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
        self.scoutgate
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
    /// - IMPORTANT: Forcibly promotes currently activated experts to VRAM to ensure inference correctness
    pub fn experts_status(&self) -> Vec<RustExpertStatus> {
        let fused_predictions = self
            .fuser
            .fuse(
                self.ewma.get_probabilities(),
                self.scoutgate.get_probabilities(),
            )
            .unwrap();

        debug!("Fused predictions: {:?}", fused_predictions);

        let mut expert_state = self.watermark.make_cache_decisions(&fused_predictions);

        // Get current layer from timer
        let current_layer = {
            let timer = self.timer.read().unwrap();
            timer.current_layer()
        };

        info!(
            "current_activated_experts: {:?}",
            self.current_activated_experts
        );

        // Force currently activated experts to VRAM tier to ensure inference correctness
        // This overrides watermark algorithm decisions for experts that are actively needed
        for &expert_id in &self.current_activated_experts {
            expert_state.inner[current_layer][expert_id] = MemoryTier::Vram;
        }

        debug!(
            "Expert state after forced VRAM placement: {:?}",
            expert_state
        );

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
                for param_type in RustExpertParamType::ALL_PARAM_TYPES {
                    let expert_key = RustExpertKey::new(layer_idx, expert_idx, param_type);
                    result.push(RustExpertStatus::new(expert_key, tier_u8));
                }
            }
        }

        result
    }
}
