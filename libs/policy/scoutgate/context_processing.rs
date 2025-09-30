//! Context processing pipeline for ScoutGate
//!
//! This module implements context processing operations:
//! - Concatenation: [z_{t-m+1} || ... || z_t || z_ℓ]
//! - Optional linear compression to hidden dimension d_h
//! - Layer normalization for stable processing

use burn_ndarray::{NdArray, NdArrayDevice};
use burn::nn::{LinearConfig, LayerNormConfig};
use burn::tensor::{Tensor, Shape};

use crate::scoutgate::error::ScoutGateError;

type Backend = NdArray<f32>;
type Device = NdArrayDevice;

/// Context processor for ScoutGate context preparation
///
/// Handles context concatenation, optional compression,
/// and normalization to prepare context for two-tower architecture.
pub struct ContextProcessor {
    /// Device for tensor operations
    device: Device,
    
    /// Token projection dimension (d_proj = 128)
    d_proj: usize,
    
    /// Layer embedding dimension (d_ℓ)
    d_layer: usize,
    
    /// Context window size (m = 8)
    context_window_size: usize,
    
    /// Hidden dimension (d_h = 256)
    d_hidden: usize,
    
    /// Optional context compression layer: (m * d_proj + d_ℓ) -> d_h
    context_compression: Option<burn::nn::Linear<Backend>>,
    
    /// Layer normalization for context
    context_norm: burn::nn::LayerNorm<Backend>,
}

impl ContextProcessor {
    /// Create a new context processor
    pub fn new(
        d_proj: usize,
        d_layer: usize,
        context_window_size: usize,
        d_hidden: usize,
        use_compression: bool,
        device: Device,
    ) -> Result<Self, ScoutGateError> {
        let input_dim = context_window_size * d_proj + d_layer;
        
        // Optional compression layer
        let context_compression = if use_compression {
            let compression_config = LinearConfig::new(input_dim, d_hidden);
            Some(compression_config.init(&device))
        } else {
            None
        };
        
        // Context normalization (on final dimension)
        let norm_dim = if use_compression { d_hidden } else { input_dim };
        let context_norm_config = LayerNormConfig::new(norm_dim);
        let context_norm = context_norm_config.init(&device);
        
        Ok(Self {
            device,
            d_proj,
            d_layer,
            context_window_size,
            d_hidden,
            context_compression,
            context_norm,
        })
    }

    /// Concatenate token embeddings with layer embedding
    ///
    /// Combines processed token embeddings [seq_len, d_proj] with layer embedding [d_layer]
    /// to create unified context representation [seq_len, d_proj + d_layer]
    pub fn concatenate_context(
        &self,
        token_embeddings: Tensor<Backend, 2>,
        layer_embedding: Tensor<Backend, 2>,
    ) -> Result<Tensor<Backend, 2>, ScoutGateError> {
        // Validate input dimensions
        let token_shape = token_embeddings.shape();
        let layer_shape = layer_embedding.shape();
        
        // token_embeddings: [seq_len, d_proj]
        // layer_embedding: [1, d_layer] or [seq_len, d_layer]
        let seq_len = token_shape.dims[0];
        
        // If layer_embedding is [1, d_layer], expand to [seq_len, d_layer]
        let expanded_layer_embedding = if layer_shape.dims[0] == 1 {
            // Expand layer embedding to match token sequence length
            layer_embedding.repeat(&[seq_len, layer_shape.dims[1]])
        } else if layer_shape.dims[0] == seq_len {
            layer_embedding
        } else {
            return Err(ScoutGateError::ContextProcessingError {
                message: format!(
                    "Layer embedding batch size {} doesn't match token sequence length {}",
                    layer_shape.dims[0], seq_len
                )
            });
        };
        
        // Concatenate on the last dimension
        let context = Tensor::cat(vec![token_embeddings, expanded_layer_embedding], 1);
        
        Ok(context)
    }

    /// Apply linear compression to context (optional)
    ///
    /// If compression is enabled, applies linear transformation to reduce dimensionality
    pub fn compress_context(
        &self,
        context: Tensor<Backend, 2>,
    ) -> Result<Tensor<Backend, 2>, ScoutGateError> {
        if let Some(ref compression_layer) = self.context_compression {
            let compressed = compression_layer.forward(context);
            Ok(compressed)
        } else {
            // No compression, return as is
            Ok(context)
        }
    }

    /// Apply layer normalization to context
    pub fn normalize_context(
        &self,
        context: Tensor<Backend, 2>,
    ) -> Result<Tensor<Backend, 2>, ScoutGateError> {
        let normalized = self.context_norm.forward(context);
        Ok(normalized)
    }

    /// Complete context processing pipeline
    ///
    /// Processes token embeddings and layer embedding through the full pipeline:
    /// 1. Concatenate token and layer embeddings
    /// 2. Apply optional compression
    /// 3. Apply layer normalization
    pub fn process_context(
        &self,
        token_embeddings: Tensor<Backend, 2>,
        layer_embedding: Tensor<Backend, 2>,
    ) -> Result<Tensor<Backend, 2>, ScoutGateError> {
        // Step 1: Concatenate
        let context = self.concatenate_context(token_embeddings, layer_embedding)?;
        
        // Step 2: Optional compression
        let compressed = self.compress_context(context)?;
        
        // Step 3: Normalization
        let normalized = self.normalize_context(compressed)?;
        
        Ok(normalized)
    }

    /// Batch process multiple contexts
    ///
    /// Process multiple token-layer embedding pairs efficiently
    pub fn process_context_batch(
        &self,
        token_embeddings: &Tensor<Backend, 2>,
        layer_embedding: &Tensor<Backend, 2>,
    ) -> Result<Tensor<Backend, 2>, ScoutGateError> {
        // For batch processing, delegate to single context processing
        // In a more optimized implementation, this could be vectorized
        self.process_context(token_embeddings.clone(), layer_embedding.clone())
    }

    /// Get expected input dimension for context
    pub fn get_input_dimension(&self) -> usize {
        self.context_window_size * self.d_proj + self.d_layer
    }

    /// Get output dimension after processing
    pub fn get_output_dimension(&self) -> usize {
        if self.context_compression.is_some() {
            self.d_hidden
        } else {
            self.get_input_dimension()
        }
    }
}