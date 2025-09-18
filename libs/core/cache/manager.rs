//! Core manager structure for the two-level watermark cache
//!
//! This module defines the main TwoTierWmExpertCacheManager struct and its
//! core data structures, serving as a thin Python interface layer.

use crate::types::model::RustModelType;
use policy::{
    constants::{ModelConfig, ModelType},
    ewma::predictor::EwmaPredictor,
    fusion::ProbabilityFusion,
    scoutgate::predictor::ScoutGatePredictor,
    timer::Timer,
    watermark::algorithm::WatermarkAlgorithm,
};
use pyo3::prelude::*;
use std::sync::{Arc, RwLock};
/// Thin Python interface for the two-level MOE cache system
///
/// This is the CORRECT architecture implementation:
/// Core layer is ONLY a Python interface - all business logic in policy layer
#[pyclass]
pub struct RustTwoTierWmExpertCacheManager {
    /// Timer for time step management
    pub(crate) timer: Arc<RwLock<Timer>>,

    /// EWMA predictor for expert activation probability estimation
    pub(crate) ewma_predictor: EwmaPredictor,

    /// ScoutGate predictor for semantic-based expert activation prediction
    pub(crate) scoutgate_predictor: ScoutGatePredictor,

    /// Probability fusion component
    pub(crate) probability_fuser: ProbabilityFusion,

    /// Watermark algorithm for cache decisions
    pub(crate) watermark_algorithm: WatermarkAlgorithm,

    /// Cache of currently activated experts for the current layer
    /// Used to force these experts to VRAM tier during experts_status() calls
    pub(crate) current_activated_experts: Vec<usize>,
}

impl RustTwoTierWmExpertCacheManager {
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
        let config: ModelConfig = model_type.clone().into();
        let model_type: ModelType = model_type.into();

        // Create timer from model configuration
        let timer = Arc::new(RwLock::new(Timer::from_config(&config)));

        // Create EWMA predictor with shared timer reference
        let ewma_predictor = EwmaPredictor::from_model(timer.clone(), model_type);

        // Create ScoutGate predictor with shared timer reference
        let scoutgate_predictor = ScoutGatePredictor::from_model(timer.clone(), model_type);

        // Create probability fusion with shared timer reference
        let probability_fuser = ProbabilityFusion::from_model(model_type, timer.clone());

        // Create watermark algorithm
        let watermark_algorithm =
            WatermarkAlgorithm::from_model(model_type, vram_capacity, ram_capacity);

        Ok(Self {
            timer,
            ewma_predictor,
            scoutgate_predictor,
            probability_fuser,
            watermark_algorithm,
            current_activated_experts: Vec::new(),
        })
    }
}
