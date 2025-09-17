//! Core manager structure for the two-level watermark cache
//!
//! This module defines the main TwoTireWmExpertCacheManager struct and its
//! core data structures, serving as a thin Python interface layer.

use crate::types::model::ModelType;
use policy::{
    ExpertKey,
    fusion::{FusionConfig, ProbabilityFusion},
    watermark::WatermarkAlgorithm,
};
use pyo3::prelude::*;

/// Thin Python interface for the two-level MOE cache system
///
/// This is the CORRECT architecture implementation:
/// Core layer is ONLY a Python interface - all business logic in policy layer
#[pyclass]
pub struct RustTwoTireWmExpertCacheManager {
    /// Current time for tracking
    pub(crate) current_time: u64,

    /// Probability fusion component
    pub(crate) fusion: ProbabilityFusion,

    /// All experts in the model
    pub(crate) all_experts: Vec<ExpertKey>,

    /// Watermark algorithm for cache decisions
    pub(crate) watermark_algorithm: WatermarkAlgorithm,
}

impl RustTwoTireWmExpertCacheManager {
    /// Create a new cache manager instance
    pub fn new(
        _model_type: ModelType,
        vram_capacity: usize,
        ram_capacity: usize,
    ) -> Result<Self, String> {
        // Create probability fusion
        let fusion_config = FusionConfig::for_gptoss20b();
        let fusion = ProbabilityFusion::new(fusion_config)
            .map_err(|e| format!("Failed to create probability fusion: {}", e))?;

        // Create watermark algorithm
        let watermark_algorithm = WatermarkAlgorithm::for_gptoss20b(vram_capacity, ram_capacity)
            .map_err(|e| format!("Failed to create watermark algorithm: {}", e))?;

        Ok(Self {
            current_time: 0,
            fusion,
            watermark_algorithm,
            all_experts: Vec::new(), // Initialize with empty vector for now
        })
    }
}
