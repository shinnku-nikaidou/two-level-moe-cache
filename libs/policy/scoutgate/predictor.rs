//! ScoutGate predictor implementation
//!
//! This module implements the complete ScoutGate predictor that provides semantic-based
//! expert activation probability predictions. The implementation includes all components:
//! - Token embedding processing with sliding window context
//! - Layer-specific embeddings
//! - Context processing pipeline
//! - Expert embedding management
//! - Two-tower scoring architecture
//! - Integration with Timer and ExpertProbability systems

use burn::tensor::Tensor;
use burn_ndarray::{NdArray, NdArrayDevice};

use crate::ExpertProbability;
use crate::constants::{ModelConfig, ModelType};
use crate::timer::Timer;
use std::sync::{Arc, RwLock};

use super::config::ScoutGateConfig;
use super::context_processing::ContextProcessor;
use super::error::ScoutGateError;
use super::expert_embedding::ExpertEmbeddingStore;
use super::layer_embedding::LayerEmbeddingManager;
use super::token_embedding::TokenEmbeddingProcessor;
use super::two_tower::TwoTowerScorer;

type Backend = NdArray<f32>;
type Device = NdArrayDevice;

/// ScoutGate predictor for semantic-based expert activation probability prediction
///
/// Complete implementation with all ScoutGate components:
/// - Token embedding processing with sliding window context
/// - Layer-specific embeddings for routing pattern differentiation  
/// - Context processing pipeline with concatenation and normalization
/// - Expert embedding store with precomputed projection matrices
/// - Two-tower architecture for efficient expert scoring
/// - Integration with Timer and ExpertProbability systems
pub struct ScoutGatePredictor {
    /// Shared timer for time step management
    timer: Arc<RwLock<Timer>>,

    /// ScoutGate configuration parameters
    config: ScoutGateConfig,

    /// Model configuration (layers, experts per layer, etc.)
    model_config: ModelConfig,

    /// Device for tensor operations
    device: Device,

    /// Current prediction values for all expert-layer pairs
    predictions: ExpertProbability,

    /// Token embedding processor
    token_processor: TokenEmbeddingProcessor,

    /// Layer embedding manager
    layer_manager: LayerEmbeddingManager,

    /// Context processing pipeline
    context_processor: ContextProcessor,

    /// Expert embedding store
    expert_store: ExpertEmbeddingStore,

    /// Two-tower scorer
    two_tower_scorer: TwoTowerScorer,
}

impl ScoutGatePredictor {
    /// Create a new ScoutGate predictor with shared timer
    pub fn new(
        timer: Arc<RwLock<Timer>>,
        model_config: ModelConfig,
        config: ScoutGateConfig,
    ) -> Result<Self, ScoutGateError> {
        // Validate configuration
        config.validate()?;

        let device = Device::default();

        // Create predictions matrix
        let predictions =
            ExpertProbability::new(model_config.total_layers, model_config.experts_per_layer);

        // Initialize all components
        let token_processor = TokenEmbeddingProcessor::new(
            4096, // d_emb - placeholder, should be from model config
            config.projection_dim,
            config.context_window_size,
            device,
        )?;

        let layer_manager =
            LayerEmbeddingManager::new(model_config.clone(), config.layer_embedding_dim, device)?;

        let context_processor = ContextProcessor::new(
            config.projection_dim,
            config.layer_embedding_dim,
            config.context_window_size,
            config.hidden_dim,
            true, // use compression
            device,
        )?;

        let expert_store = ExpertEmbeddingStore::new(
            model_config.clone(),
            config.expert_embedding_dim,
            config.low_rank_dim,
            device,
        )?;

        let two_tower_scorer = TwoTowerScorer::new(config.hidden_dim, config.low_rank_dim, device)?;

        Ok(ScoutGatePredictor {
            timer,
            config,
            model_config,
            device,
            predictions,
            token_processor,
            layer_manager,
            context_processor,
            expert_store,
            two_tower_scorer,
        })
    }

    /// Update predictions using complete ScoutGate pipeline
    pub fn update_predictions(&mut self) -> Result<(), ScoutGateError> {
        // Run complete prediction pipeline for all layers
        self.predict_all_layers()?;
        Ok(())
    }

    /// Create ScoutGate predictor from model type
    pub fn from_model(
        timer: Arc<RwLock<Timer>>,
        model_type: ModelType,
    ) -> Result<Self, ScoutGateError> {
        let model_config: ModelConfig = model_type.into();
        Self::new(timer, model_config, ScoutGateConfig::default())
    }

    /// Update token context with new token
    ///
    /// This method maintains a sliding window of the most recent m tokens for ScoutGate prediction.
    /// When a new token is added:
    /// 1. Add the token to the token processor's context window
    /// 2. Maintain the window size by removing oldest tokens if needed
    /// 3. Trigger re-computation of predictions using the complete pipeline
    ///
    /// # Arguments
    /// * `new_token` - The new token ID to add to the context
    ///
    /// # Returns
    /// * `Ok(())` if successful
    /// * `Err(ScoutGateError)` if there's an error
    pub fn update_token_context(&mut self, new_token: u32) -> Result<(), ScoutGateError> {
        // Add token to the token embedding processor
        self.token_processor.add_token(new_token)?;

        // Trigger prediction update with new context
        self.update_predictions()?;

        Ok(())
    }

    /// Predict expert activations for all layers
    ///
    /// This is the main inference method that runs the complete ScoutGate pipeline:
    /// 1. Process token context through embedding and projection layers
    /// 2. Get layer embeddings for all layers
    /// 3. Run context processing pipeline  
    /// 4. Score experts using two-tower architecture
    /// 5. Update predictions matrix
    pub fn predict_all_layers(&mut self) -> Result<(), ScoutGateError> {
        // Step 1: Process token context (using the method without arguments)
        let token_context = self.token_processor.process_context()?;

        // Step 2: Process all layers simultaneously
        for layer_id in 0..self.model_config.total_layers {
            // Get layer embedding (returns Tensor<Backend, 1>)
            let layer_embedding = self.layer_manager.get_layer_embedding(layer_id)?;

            // Convert 1D layer embedding to 2D for context processing
            let layer_embedding_2d = layer_embedding.unsqueeze::<2>();

            // Process context (concatenate token and layer embeddings)
            let processed_context = self
                .context_processor
                .process_context(token_context.clone(), layer_embedding_2d)?;

            // Get expert embeddings matrix for this layer
            let expert_matrix = self.expert_store.get_precomputed_matrix(layer_id)?;

            // Score experts using two-tower architecture
            let scores = self.two_tower_scorer.score_experts(
                processed_context,
                expert_matrix.clone(),
                layer_id,
            )?;

            // Extract scores and update predictions matrix
            let scores_data = scores.to_data();
            let scores_vec: Vec<f32> = scores_data.to_vec().unwrap();

            // Update predictions for this layer (using the correct method name)
            for (expert_id, &score) in scores_vec.iter().enumerate() {
                if expert_id < self.model_config.experts_per_layer {
                    self.predictions.set(layer_id, expert_id, score as f64);
                }
            }
        }

        Ok(())
    }

    /// Predict expert activations for a specific layer
    pub fn predict_layer(&mut self, layer_id: usize) -> Result<Vec<f64>, ScoutGateError> {
        // Validate layer ID
        if layer_id >= self.model_config.total_layers {
            return Err(ScoutGateError::LayerEmbeddingError {
                message: format!(
                    "Layer ID {} out of bounds (max: {})",
                    layer_id,
                    self.model_config.total_layers - 1
                ),
            });
        }

        // Step 1: Process token context
        let token_context = self.token_processor.process_context()?;

        // Step 2: Get layer embedding
        let layer_embedding = self.layer_manager.get_layer_embedding(layer_id)?;
        let layer_embedding_2d = layer_embedding.unsqueeze::<2>();

        // Step 3: Process context
        let processed_context = self
            .context_processor
            .process_context(token_context, layer_embedding_2d)?;

        // Step 4: Get expert embeddings matrix
        let expert_matrix = self.expert_store.get_precomputed_matrix(layer_id)?;

        // Step 5: Score experts
        let scores = self.two_tower_scorer.score_experts(
            processed_context,
            expert_matrix.clone(),
            layer_id,
        )?;

        // Step 6: Extract and return scores
        let scores_data = scores.to_data();
        let scores_vec: Vec<f32> = scores_data.to_vec().unwrap();
        let result: Vec<f64> = scores_vec.iter().map(|&x| x as f64).collect();

        Ok(result)
    }

    /// Get activation probability predictions for all layers and experts
    ///
    /// **PLACEHOLDER**: Returns stored predictions
    ///
    /// This corresponds to the full ScoutGate output across all layers.
    /// In full implementation, this would be the main prediction method.
    pub fn get_probabilities(&self) -> &ExpertProbability {
        &self.predictions
    }
}
