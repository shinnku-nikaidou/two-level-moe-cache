//! Core manager structure for the two-level watermark cache
//!
//! This module defines the main TwoTireWmExpertCacheManager struct and its
//! core data structures, serving as a thin Python interface layer.

use policy::ExpertKey as PolicyExpertKey;
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::types::ModelType;

/// Thin Python interface for the two-level MOE cache system
///
/// This is the CORRECT architecture implementation:
/// Core layer is ONLY a Python interface - all business logic in policy layer
#[pyclass]
pub struct TwoTireWmExpertCacheManager {
    /// Current time for tracking
    pub(crate) current_time: u64,

    /// Current layer (0-based)
    pub(crate) current_layer: usize,

    /// Total layers in model
    pub(crate) total_layers: usize,

    /// Mock state for demonstration - real implementation would delegate everything to policy layer
    pub(crate) mock_ewma_probs: HashMap<PolicyExpertKey, f64>,
}

impl TwoTireWmExpertCacheManager {
    /// Create a new cache manager instance
    pub fn new(
        _model_type: ModelType,
        total_layers: usize,
        _vram_capacity: usize,
        _ram_capacity: usize,
    ) -> Result<Self, String> {
        if total_layers == 0 {
            return Err("total_layers must be > 0".to_string());
        }

        Ok(Self {
            current_time: 0,
            current_layer: 0,
            total_layers,
            mock_ewma_probs: HashMap::new(),
        })
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
}
