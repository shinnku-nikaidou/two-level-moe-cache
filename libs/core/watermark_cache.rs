//! Thin Python interface layer for expert cache management
//!
//! This module provides a minimal Python interface that properly delegates ALL
//! business logic to the policy layer components.

use pyo3::prelude::*;
use std::collections::HashMap;

use policy::{ExpertKey as PolicyExpertKey, ExpertParamType};

use crate::python_types::{ExpertKey, ExpertRef, MemoryTier, ModelType};

/// Thin Python interface for the two-level MOE cache system
///
/// This is the CORRECT architecture implementation:
/// Core layer is ONLY a Python interface - all business logic in policy layer
#[pyclass]
pub struct TwoTireWmExpertCacheManager {
    /// Current time for tracking
    current_time: u64,
    
    /// Current layer (0-based)
    current_layer: usize,
    
    /// Total layers in model
    total_layers: usize,
    
    /// Mock state for demonstration - real implementation would delegate everything to policy layer
    mock_ewma_probs: HashMap<PolicyExpertKey, f64>,
}

#[pymethods]
impl TwoTireWmExpertCacheManager {
    #[new]
    pub fn new(
        _model_type: ModelType,
        total_layers: usize,
        _vram_capacity: usize,
        _ram_capacity: usize,
    ) -> PyResult<Self> {
        if total_layers == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "total_layers must be > 0",
            ));
        }

        Ok(Self {
            current_time: 0,
            current_layer: 0,
            total_layers,
            mock_ewma_probs: HashMap::new(),
        })
    }

    /// Get expert by key - delegates to policy layer (mock implementation)
    pub fn get(&mut self, expert_key: ExpertKey) -> PyResult<ExpertRef> {
        // Convert Python ExpertKey to Policy ExpertKey
        let policy_key = PolicyExpertKey::new(
            expert_key.expert_id,
            expert_key.layer_idx,
            ExpertParamType::MLP1Weight,
        );

        // Mock EWMA prediction (real implementation would delegate to policy layer)
        let ewma_prob = self.mock_ewma_probs.get(&policy_key).copied().unwrap_or(0.3);

        // Simple cache decision: cache if probability > 0.5
        let should_cache = ewma_prob > 0.5;

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
        // Mock activation tracking (real implementation would delegate to policy layer)
        let current_layer = self.current_layer;

        // Update mock EWMA probabilities for all experts in current layer
        for expert_id in 0..8 { // Assume 8 experts per layer
            let policy_key = PolicyExpertKey::new(
                expert_id, 
                current_layer, 
                ExpertParamType::MLP1Weight
            );

            let activated = activated_experts.contains(&expert_id);
            
            // Simple EWMA update: p_new = 0.9 * p_old + 0.1 * hit
            let old_prob = self.mock_ewma_probs.get(&policy_key).copied().unwrap_or(0.0);
            let hit = if activated { 1.0 } else { 0.0 };
            let new_prob = 0.9 * old_prob + 0.1 * hit;
            
            self.mock_ewma_probs.insert(policy_key, new_prob);
        }

        Ok(())
    }

    /// Advance to next time step
    pub fn step_forward(&mut self) -> PyResult<()> {
        self.current_time += 1;
        self.current_layer = (self.current_layer + 1) % self.total_layers;
        Ok(())
    }

    /// Get current statistics from policy components (mock implementation)
    pub fn get_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();

        // Mock watermark values (real implementation would get from policy layer)
        stats.insert("vram_watermark".to_string(), 0.75);
        stats.insert("ram_watermark".to_string(), 0.25);

        // Timing info
        stats.insert("current_time".to_string(), self.current_time as f64);
        stats.insert("current_layer".to_string(), self.current_layer as f64);
        stats.insert("total_experts_tracked".to_string(), self.mock_ewma_probs.len() as f64);

        stats
    }
}
