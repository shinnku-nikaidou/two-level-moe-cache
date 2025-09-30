//! Expert embedding storage and precomputed matrix management for ScoutGate
//!
//! This module manages expert embeddings and precomputed projection matrices:
//! - Expert embeddings V_ℓ for each layer ℓ
//! - Precomputed matrices M_ℓ = V_ℓ W_e^T for efficient scoring
//! - Expert tower weights W_e for projection operations
//! - Lazy computation and caching for performance optimization

use std::collections::HashMap;
use burn_ndarray::{NdArray, NdArrayDevice};
use burn::tensor::Tensor;

use crate::constants::ModelConfig;
use crate::scoutgate::error::ScoutGateError;

type Backend = NdArray<f32>;
type Device = NdArrayDevice;

/// Expert embedding store for ScoutGate
///
/// Manages expert embeddings and precomputed projection matrices
/// for each expert-layer pair
pub struct ExpertEmbeddingStore {
    /// Device for tensor operations
    device: Device,
    
    /// Model configuration
    model_config: ModelConfig,
    
    /// Expert embedding dimension (d_expert = 512)
    d_expert: usize,
    
    /// Low-rank projection dimension (d' = 64)
    d_prime: usize,
    
    /// Expert embeddings per layer: layer_id -> [num_experts, d_expert]
    expert_embeddings: HashMap<usize, Tensor<Backend, 2>>,
    
    /// Precomputed matrices per layer: layer_id -> M_ℓ = V_ℓ W_e^T
    /// Shape: [num_experts, d_prime]
    precomputed_matrices: HashMap<usize, Tensor<Backend, 2>>,
    
    /// Expert tower weights: W_e [d_expert, d_prime]
    expert_tower_weights: Tensor<Backend, 2>,
}

impl ExpertEmbeddingStore {
    /// Create a new expert embedding store
    pub fn new(
        model_config: ModelConfig,
        d_expert: usize,
        d_prime: usize,
        device: Device,
    ) -> Result<Self, ScoutGateError> {
        // Initialize expert tower weights randomly
        let expert_tower_weights = Tensor::random(
            [d_expert, d_prime], 
            burn::tensor::Distribution::Normal(0.0, 1.0), 
            &device
        );
        
        // Initialize expert embeddings for all layers
        let mut expert_embeddings = HashMap::new();
        for layer_id in 0..model_config.total_layers {
            let layer_expert_embeddings = Tensor::random(
                [model_config.experts_per_layer, d_expert],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device
            );
            expert_embeddings.insert(layer_id, layer_expert_embeddings);
        }
        
        let precomputed_matrices = HashMap::new();

        Ok(Self {
            device,
            model_config,
            d_expert,
            d_prime,
            expert_embeddings,
            precomputed_matrices,
            expert_tower_weights,
        })
    }

    /// Get expert embeddings for a specific layer
    pub fn get_layer_expert_embeddings(&self, layer_id: usize) -> Result<&Tensor<Backend, 2>, ScoutGateError> {
        // Validate layer ID
        if layer_id >= self.model_config.total_layers {
            return Err(ScoutGateError::ExpertEmbeddingError {
                message: format!("Layer ID {} exceeds total layers {}", layer_id, self.model_config.total_layers)
            });
        }
        
        // Get expert embeddings for this layer
        if let Some(embeddings) = self.expert_embeddings.get(&layer_id) {
            Ok(embeddings)
        } else {
            Err(ScoutGateError::ExpertEmbeddingError {
                message: format!("Expert embeddings not found for layer {}", layer_id)
            })
        }
    }

    /// Get or compute precomputed projection matrix for a layer
    pub fn get_precomputed_matrix(&mut self, layer_id: usize) -> Result<&Tensor<Backend, 2>, ScoutGateError> {
        // Check if matrix already exists
        if self.precomputed_matrices.contains_key(&layer_id) {
            return Ok(self.precomputed_matrices.get(&layer_id).unwrap());
        }
        
        // Validate layer ID
        if layer_id >= self.model_config.total_layers {
            return Err(ScoutGateError::ExpertEmbeddingError {
                message: format!("Layer ID {} exceeds total layers {}", layer_id, self.model_config.total_layers)
            });
        }
        
        // Get expert embeddings for this layer
        let expert_embeddings = self.expert_embeddings.get(&layer_id)
            .ok_or_else(|| ScoutGateError::ExpertEmbeddingError {
                message: format!("Expert embeddings not found for layer {}", layer_id)
            })?;
        
        // Compute M_ℓ = V_ℓ W_e^T
        // expert_embeddings: [num_experts, d_expert]  
        // expert_tower_weights: [d_expert, d_prime]
        // Result: [num_experts, d_prime]
        let precomputed_matrix = expert_embeddings.clone().matmul(self.expert_tower_weights.clone());
        
        // Store and return reference
        self.precomputed_matrices.insert(layer_id, precomputed_matrix);
        Ok(self.precomputed_matrices.get(&layer_id).unwrap())
    }

    /// Update expert embeddings for a layer (for training)
    pub fn update_layer_expert_embeddings(
        &mut self,
        layer_id: usize,
        new_embeddings: Tensor<Backend, 2>,
    ) -> Result<(), ScoutGateError> {
        // Validate layer ID
        if layer_id >= self.model_config.total_layers {
            return Err(ScoutGateError::ExpertEmbeddingError {
                message: format!("Layer ID {} exceeds total layers {}", layer_id, self.model_config.total_layers)
            });
        }
        
        // Update expert embeddings
        self.expert_embeddings.insert(layer_id, new_embeddings);
        
        // Invalidate precomputed matrix for this layer
        self.precomputed_matrices.remove(&layer_id);
        
        Ok(())
    }

    /// Update expert tower weights (for training)
    pub fn update_expert_tower_weights(
        &mut self,
        new_weights: Tensor<Backend, 2>,
    ) -> Result<(), ScoutGateError> {
        // Update expert tower weights
        self.expert_tower_weights = new_weights;
        
        // Invalidate all precomputed matrices since W_e changed
        self.precomputed_matrices.clear();
        
        Ok(())
    }

    /// Get embedding for specific expert
    pub fn get_expert_embedding(
        &self,
        layer_id: usize,
        expert_id: usize,
    ) -> Result<Tensor<Backend, 1>, ScoutGateError> {
        // Validate IDs
        self.validate_expert_id(layer_id, expert_id)?;
        
        // Get layer embeddings
        let layer_embeddings = self.get_layer_expert_embeddings(layer_id)?;
        
        // Extract specific expert embedding [d_expert]
        let expert_embedding = layer_embeddings.clone().slice([expert_id..expert_id+1, 0..self.d_expert]);
        let expert_embedding_1d = expert_embedding.squeeze::<1>(0);
        
        Ok(expert_embedding_1d)
    }

    /// Get number of experts for a layer
    pub fn get_layer_expert_count(&self, layer_id: usize) -> Result<usize, ScoutGateError> {
        // Validate layer ID
        if layer_id >= self.model_config.total_layers {
            return Err(ScoutGateError::ExpertEmbeddingError {
                message: format!("Layer ID {} exceeds total layers {}", layer_id, self.model_config.total_layers)
            });
        }
        
        Ok(self.model_config.experts_per_layer)
    }

    /// Validate expert ID bounds
    pub fn validate_expert_id(&self, layer_id: usize, expert_id: usize) -> Result<(), ScoutGateError> {
        // Validate layer ID
        if layer_id >= self.model_config.total_layers {
            return Err(ScoutGateError::ExpertEmbeddingError {
                message: format!("Layer ID {} exceeds total layers {}", layer_id, self.model_config.total_layers)
            });
        }
        
        // Validate expert ID
        if expert_id >= self.model_config.experts_per_layer {
            return Err(ScoutGateError::ExpertEmbeddingError {
                message: format!("Expert ID {} exceeds experts per layer {}", expert_id, self.model_config.experts_per_layer)
            });
        }
        
        Ok(())
    }

    /// Get expert tower weights reference
    pub fn get_expert_tower_weights(&self) -> &Tensor<Backend, 2> {
        &self.expert_tower_weights
    }

    /// Get dimensions
    pub fn get_expert_dimension(&self) -> usize {
        self.d_expert
    }

    pub fn get_projection_dimension(&self) -> usize {
        self.d_prime
    }
}