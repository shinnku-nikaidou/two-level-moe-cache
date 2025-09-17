//! Core manager structure for the two-level watermark cache
//!
//! This module defines the main TwoTireWmExpertCacheManager struct and its
//! core data structures, serving as a thin Python interface layer.

use crate::types::model::RustModelType;
use policy::{
    ExpertKey,
    constants::models::ModelConfig,
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

    /// All experts' keys in the model
    pub(crate) all_experts_key: Vec<ExpertKey>,

    /// Watermark algorithm for cache decisions
    pub(crate) watermark_algorithm: WatermarkAlgorithm,
}

impl RustTwoTireWmExpertCacheManager {
    /// Create a new cache manager instance
    ///
    /// This function automatically derives all expert keys from the provided model type
    /// by mapping it to the corresponding model configuration and generating all possible
    /// expert parameter combinations (layers × experts × parameter types).
    ///
    /// # Arguments
    ///
    /// * `model_type` - The type of MoE model to manage (e.g., GptOss20B, GptOss120B)
    /// * `vram_capacity` - VRAM cache capacity in number of expert parameters
    /// * `ram_capacity` - RAM cache capacity in number of expert parameters
    ///
    /// # Returns
    ///
    /// Returns `Ok(Self)` if successful, or `Err(String)` if:
    /// - Model type is not supported
    /// - Expert key generation fails
    /// - Component initialization fails
    ///
    /// # Expert Key Generation
    ///
    /// For each supported model, generates keys for:
    /// - All layers (0-based indexing)
    /// - All experts per layer (0-based indexing)  
    /// - All parameter types (MLP1_WEIGHT, MLP1_BIAS, MLP2_WEIGHT, MLP2_BIAS)
    pub fn new(
        model_type: RustModelType,
        vram_capacity: usize,
        ram_capacity: usize,
    ) -> Result<Self, String> {
        // Get model configuration from model_type using From trait
        let config = model_type.into();

        // Generate all expert keys for this model
        let all_experts_key = ExpertKey::all_experts(&config);

        // Validate the generated expert keys count
        let expected_count = config.total_layers * config.experts_per_layer * 4; // 4 param types per expert
        if all_experts_key.len() != expected_count {
            return Err(format!(
                "Expert key count mismatch for model type : expected {}, got {}",
                expected_count,
                all_experts_key.len()
            ));
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
            fusion,
            watermark_algorithm,
            all_experts_key,
        })
    }
}
