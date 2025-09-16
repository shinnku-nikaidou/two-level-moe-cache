//! Python interface methods for the cache manager
//!
//! This module contains all PyO3 methods and Python-facing interfaces
//! for the TwoTireWmExpertCacheManager.

use pyo3::prelude::*;

use policy::{ExpertKey as PolicyExpertKey, ExpertParamType};

use super::manager::TwoTireWmExpertCacheManager;
use super::mock;
use crate::types::{ExpertKey, ExpertRef, MemoryTier, ModelType};

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
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
    }

    /// Get expert by key - delegates to policy layer (mock implementation)
    pub fn get(&mut self, expert_key: ExpertKey) -> PyResult<ExpertRef> {
        // Convert Python ExpertKey to Policy ExpertKey
        let policy_key = PolicyExpertKey::new(
            expert_key.expert_id,
            expert_key.layer_idx,
            ExpertParamType::MLP1Weight,
        );

        // Delegate to mock implementation
        let (should_cache, _ewma_prob) =
            mock::get_cache_decision(&mut self.mock_ewma_probs, &policy_key);

        // Return expert reference with tier information
        let mut expert_ref = ExpertRef::new(expert_key);
        expert_ref.set_tier(if should_cache {
            Some(MemoryTier::VRAM)
        } else {
            Some(MemoryTier::DISK)
        });
        expert_ref.set_size(1024); // Default size

        Ok(expert_ref)
    }

    /// Update with new layer activations - delegates to policy layer (mock implementation)
    pub fn update_activations(&mut self, activated_experts: Vec<usize>) -> PyResult<()> {
        mock::update_activations(
            &mut self.mock_ewma_probs,
            self.current_layer,
            &activated_experts,
        );
        Ok(())
    }

    /// Advance to next time step
    pub fn step_forward(&mut self) -> PyResult<()> {
        self.current_time += 1;
        self.current_layer = (self.current_layer + 1) % self.total_layers;
        Ok(())
    }
}
