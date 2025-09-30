//! Two-tower scoring architecture for ScoutGate
//!
//! This module implements the two-tower architecture for expert scoring:
//! - Context tower: W_h: d_h -> d'
//! - Expert tower: W_e: d_e -> d' (handled in expert_embedding.rs)
//! - Dot-product similarity computation
//! - Sigmoid activation for probability outputs
//! - Vectorized batch inference optimization

use burn_ndarray::{NdArray, NdArrayDevice};
use burn::nn::{LinearConfig};
use burn::tensor::{Tensor, Shape, activation};

use crate::scoutgate::error::ScoutGateError;

type Backend = NdArray<f32>;
type Device = NdArrayDevice;

/// Two-tower scorer for ScoutGate expert activation prediction
///
/// Implements the two-tower architecture with context and expert towers
/// projecting to a shared low-rank space for efficient similarity computation.
pub struct TwoTowerScorer {
    /// Device for tensor operations
    device: Device,
    
    /// Context input dimension (d_h)
    d_context: usize,
    
    /// Low-rank shared dimension (d' = 64)
    d_prime: usize,
    
    /// Context tower: W_h [d_h, d']
    context_tower: burn::nn::Linear<Backend>,
    
    /// Bias terms per layer (optional): layer_id -> [num_experts]
    layer_biases: std::collections::HashMap<usize, Tensor<Backend, 1>>,
}

impl TwoTowerScorer {
    /// Create a new two-tower scorer
    pub fn new(
        d_context: usize,
        d_prime: usize,
        device: Device,
    ) -> Result<Self, ScoutGateError> {
        // Initialize context tower
        let context_tower_config = LinearConfig::new(d_context, d_prime);
        let context_tower = context_tower_config.init(&device);
        
        Ok(Self {
            device,
            d_context,
            d_prime,
            context_tower,
            layer_biases: std::collections::HashMap::new(),
        })
    }
    
    /// Project context through context tower
    ///
    /// Input: context [batch_size, d_h]  
    /// Output: projected context [batch_size, d']
    pub fn project_context(&self, context: Tensor<Backend, 2>) -> Result<Tensor<Backend, 2>, ScoutGateError> {
        // Project context through the context tower
        let projected = self.context_tower.forward(context);
        Ok(projected)
    }
    
    /// Compute similarity scores between projected context and expert embeddings
    ///
    /// Input: projected_context [batch_size, d'], expert_matrix [num_experts, d']
    /// Output: similarity scores [batch_size, num_experts]
    pub fn compute_similarity_scores(
        &self,
        projected_context: Tensor<Backend, 2>,
        expert_matrix: Tensor<Backend, 2>,
    ) -> Result<Tensor<Backend, 2>, ScoutGateError> {
        // Compute dot product: context @ expert_matrix.T
        // projected_context: [batch_size, d']
        // expert_matrix: [num_experts, d'] -> transposed to [d', num_experts]
        let transposed_expert = expert_matrix.transpose();
        let scores = projected_context.matmul(transposed_expert);
        Ok(scores)
    }
    
    /// Apply sigmoid activation to get probabilities
    pub fn apply_sigmoid(&self, scores: Tensor<Backend, 2>) -> Result<Tensor<Backend, 2>, ScoutGateError> {
        // Apply sigmoid activation
        let activated_scores = burn::tensor::activation::sigmoid(scores);
        Ok(activated_scores)
    }
    
    /// Complete scoring pipeline for a layer
    ///
    /// Input: context [batch_size, d_h], expert_matrix [num_experts, d']
    /// Output: probabilities [batch_size, num_experts]
    pub fn score_experts(
        &self,
        context: Tensor<Backend, 2>,
        expert_matrix: Tensor<Backend, 2>,
        layer_id: usize,
    ) -> Result<Tensor<Backend, 2>, ScoutGateError> {
        // Project context through context tower
        let projected_context = self.project_context(context)?;
        
        // Compute similarity scores with expert embeddings
        let raw_scores = self.compute_similarity_scores(projected_context, expert_matrix)?;
        
        // Add layer-specific bias if available
        let biased_scores = if let Some(bias) = self.layer_biases.get(&layer_id) {
            let expanded_bias = bias.clone().unsqueeze::<2>().expand([raw_scores.shape().dims[0], bias.shape().dims[0]]);
            raw_scores + expanded_bias
        } else {
            raw_scores
        };
        
        // Apply sigmoid activation to get probabilities
        let probabilities = self.apply_sigmoid(biased_scores)?;
        
        Ok(probabilities)
    }
    
    /// Batch scoring for multiple layers
    ///
    /// For efficiency when predicting all layers simultaneously
    pub fn score_experts_batch(
        &self,
        context: Tensor<Backend, 2>,
        expert_matrices: &[Tensor<Backend, 2>],
        layer_ids: &[usize],
    ) -> Result<Vec<Tensor<Backend, 2>>, ScoutGateError> {
        let mut results = Vec::new();
        
        for (expert_matrix, &layer_id) in expert_matrices.iter().zip(layer_ids.iter()) {
            let scores = self.score_experts(context.clone(), expert_matrix.clone(), layer_id)?;
            results.push(scores);
        }
        
        Ok(results)
    }
    
    /// Initialize bias vector for a layer
    pub fn initialize_layer_bias(
        &mut self,
        layer_id: usize,
        num_experts: usize,
    ) -> Result<(), ScoutGateError> {
        // Initialize zero bias vector for layer
        let bias = Tensor::<Backend, 1>::zeros([num_experts], &self.device);
        self.layer_biases.insert(layer_id, bias);
        Ok(())
    }
    
    /// Update bias terms for a layer (for training)
    pub fn update_layer_bias(
        &mut self,
        layer_id: usize,
        new_bias: Tensor<Backend, 1>,
    ) -> Result<(), ScoutGateError> {
        // Update bias vector for specific layer
        self.layer_biases.insert(layer_id, new_bias);
        Ok(())
    }
    
    /// Get bias for layer (for inspection)
    pub fn get_layer_bias(&self, layer_id: usize) -> Option<&Tensor<Backend, 1>> {
        self.layer_biases.get(&layer_id)
    }

    // Direct similarity computation for testing and validation
    fn compute_raw_similarity(
        &self,
        context: &Tensor<Backend, 2>,
        expert_matrix: &Tensor<Backend, 2>,
    ) -> Result<Tensor<Backend, 2>, ScoutGateError> {
        // Project context first
        let projected = self.project_context(context.clone())?;
        
        // Then compute similarity  
        let scores = self.compute_similarity_scores(projected, expert_matrix.clone())?;
        
        Ok(scores)
    }
}