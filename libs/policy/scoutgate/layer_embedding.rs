//! Layer embedding management for ScoutGate
//!
//! This module manages layer-specific embeddings that capture routing patterns
//! for different Transformer layers. Each layer has its own embedding vector
//! z_ℓ ∈ R^{d_ℓ} to distinguish layer-specific expert activation patterns.

use burn_ndarray::{NdArray, NdArrayDevice};
use burn::tensor::{Tensor, Shape};

use crate::constants::ModelConfig;
use crate::scoutgate::error::ScoutGateError;

type Backend = NdArray<f32>;
type Device = NdArrayDevice;

/// Layer embedding manager for ScoutGate
///
/// Manages layer-specific embeddings z_ℓ for each transformer layer,
/// supporting different layer configurations across models.
#[derive(Debug)]
pub struct LayerEmbeddingManager {
    /// Device for tensor operations
    device: Device,
    
    /// Model configuration
    model_config: ModelConfig,
    
    /// Embedding dimension for each layer
    d_layer: usize,
    
    /// Layer embeddings: [total_layers, d_layer]
    layer_embeddings: Tensor<Backend, 2>,
}

impl LayerEmbeddingManager {
    /// Create a new layer embedding manager
    pub fn new(
        model_config: ModelConfig,
        d_layer: usize,
        device: Device,
    ) -> Result<Self, ScoutGateError> {
        // Initialize layer embeddings with random values
        let layer_embeddings = Tensor::random(
            Shape::new([model_config.total_layers, d_layer]),
            burn::tensor::Distribution::Default,
            &device,
        );

        Ok(Self {
            device,
            model_config,
            d_layer,
            layer_embeddings,
        })
    }

    /// Get layer embedding for specific layer
    pub fn get_layer_embedding(&self, layer_id: usize) -> Result<Tensor<Backend, 1>, ScoutGateError> {
        // 验证layer_id
        self.validate_layer_id(layer_id)?;
        
        // 从tensor中提取对应layer的embedding
        // layer_embeddings: [total_layers, d_layer]
        // 我们需要取第layer_id行
        let embedding = self.layer_embeddings.clone().slice([layer_id..layer_id+1, 0..self.d_layer]);
        
        // 将2D的[1, d_layer]压缩为1D的[d_layer]
        let embedding_1d = embedding.squeeze::<1>(0);
        
        Ok(embedding_1d)
    }

    /// Get batch of layer embeddings
    pub fn get_layer_embeddings_batch(&self, layer_ids: &[usize]) -> Result<Tensor<Backend, 2>, ScoutGateError> {
        // 验证所有layer_ids
        for &layer_id in layer_ids {
            self.validate_layer_id(layer_id)?;
        }
        
        // 收集所有embeddings  
        let mut embeddings = Vec::new();
        for &layer_id in layer_ids {
            let embedding = self.get_layer_embedding(layer_id)?;
            embeddings.push(embedding);
        }
        
        // 将embeddings堆叠成batch tensor [batch_size, d_layer]
        let batch_tensor = Tensor::stack(embeddings, 0);
        Ok(batch_tensor)
    }

    /// Update layer embedding for specific layer
    pub fn update_layer_embedding(
        &mut self,
        layer_id: usize,
        new_embedding: Tensor<Backend, 1>,
    ) -> Result<(), ScoutGateError> {
        // 验证layer_id
        self.validate_layer_id(layer_id)?;
        
        // 验证embedding维度
        let shape = new_embedding.shape();
        if shape.dims[0] != self.d_layer {
            return Err(ScoutGateError::LayerEmbeddingError {
                message: format!("Expected embedding dimension {}, got {}", self.d_layer, shape.dims[0])
            });
        }
        
        // 更新对应位置的embedding
        // 将1D embedding扩展为2D [1, d_layer]
        let expanded = new_embedding.unsqueeze::<2>();
        
        // 更新layer_embeddings中的对应行
        let mut updated_embeddings = self.layer_embeddings.clone();
        updated_embeddings = updated_embeddings.slice_assign([layer_id..layer_id+1, 0..self.d_layer], expanded);
        
        self.layer_embeddings = updated_embeddings;
        Ok(())
    }

    /// Validate layer ID bounds
    pub fn validate_layer_id(&self, layer_id: usize) -> Result<(), ScoutGateError> {
        if layer_id >= self.model_config.total_layers {
            return Err(ScoutGateError::LayerEmbeddingError {
                message: format!("Layer ID {} exceeds total layers {}", layer_id, self.model_config.total_layers)
            });
        }
        Ok(())
    }

    /// Get total number of layers
    pub fn get_total_layers(&self) -> usize {
        self.model_config.total_layers
    }

    /// Get layer embedding dimension
    pub fn get_layer_dimension(&self) -> usize {
        self.d_layer
    }
}