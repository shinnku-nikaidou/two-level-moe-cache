//! Core manager structure for the two-level watermark cache
//!
//! This module defines the main TwoTireWmExpertCacheManager struct and its
//! core data structures, serving as a thin Python interface layer.

use policy::{
    fusion::{FusionConfig, ProbabilityFusion},
    watermark::WatermarkAlgorithm,
};
use pyo3::prelude::*;

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

    /// Probability fusion component
    pub(crate) fusion: ProbabilityFusion,

    /// Watermark algorithm for cache decisions
    pub(crate) watermark_algorithm: WatermarkAlgorithm,
}

impl TwoTireWmExpertCacheManager {
    /// Create a new cache manager instance
    pub fn new(
        _model_type: ModelType,
        total_layers: usize,
        vram_capacity: usize,
        ram_capacity: usize,
    ) -> Result<Self, String> {
        if total_layers == 0 {
            return Err("total_layers must be > 0".to_string());
        }

        // Create probability fusion
        let fusion_config = FusionConfig::for_gptoss20b();
        let fusion = ProbabilityFusion::new(fusion_config)
            .map_err(|e| format!("Failed to create probability fusion: {}", e))?;

        // Create watermark algorithm
        let watermark_algorithm = WatermarkAlgorithm::for_gptoss20b(vram_capacity, ram_capacity)
            .map_err(|e| format!("Failed to create watermark algorithm: {}", e))?;

        Ok(Self {
            current_time: 0,
            current_layer: 0,
            total_layers,
            fusion,
            watermark_algorithm,
        })
    }
}
