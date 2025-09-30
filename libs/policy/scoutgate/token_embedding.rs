//! Token embedding processing module for ScoutGate
//!
//! This module handles token embedding operations including:
//! - Token embedding retrieval from main model (placeholder interface)
//! - Projection from d_emb to d_proj=128
//! - Layer normalization
//! - Sliding window context management (recent m=8 tokens)

use burn_ndarray::{NdArray, NdArrayDevice};
use burn::nn::{LinearConfig, LayerNormConfig};
use burn::tensor::{Tensor, Shape};

use crate::scoutgate::error::ScoutGateError;

type Backend = NdArray<f32>;
type Device = NdArrayDevice;

/// Token embedding processor for ScoutGate context preparation
///
/// Handles the token preprocessing pipeline:
/// 1. Token embedding retrieval (placeholder for main model integration)
/// 2. Projection: d_emb -> d_proj (128)
/// 3. Layer normalization
/// 4. Sliding window context management
pub struct TokenEmbeddingProcessor {
    /// Device for tensor operations
    device: Device,
    
    /// Main model embedding dimension (e.g., 4096 for GPT-OSS models)
    d_emb: usize,
    
    /// Projection dimension (d_proj = 128 from hyperparameters)
    d_proj: usize,
    
    /// Context window size (m = 8 tokens)
    context_window_size: usize,
    
    /// Token context history (sliding window)
    token_context: Vec<u32>,
    
    /// Projection layer: d_emb -> d_proj
    projection_layer: burn::nn::Linear<Backend>,
    
    /// Layer normalization for projected tokens
    layer_norm: burn::nn::LayerNorm<Backend>,
}

impl TokenEmbeddingProcessor {
    /// Create a new token embedding processor
    pub fn new(
        d_emb: usize,
        d_proj: usize,
        context_window_size: usize,
        device: Device,
    ) -> Result<Self, ScoutGateError> {
        // Initialize projection layer
        let projection_config = LinearConfig::new(d_emb, d_proj);
        let projection_layer = projection_config.init(&device);
        
        // Initialize layer normalization
        let layer_norm_config = LayerNormConfig::new(d_proj);
        let layer_norm = layer_norm_config.init(&device);
        
        Ok(Self {
            device,
            d_emb,
            d_proj,
            context_window_size,
            token_context: Vec::new(),
            projection_layer,
            layer_norm,
        })
    }
    
    /// Add new token to sliding window context
    pub fn add_token(&mut self, token_id: u32) -> Result<(), ScoutGateError> {
        self.token_context.push(token_id);
        
        // Maintain sliding window size
        if self.token_context.len() > self.context_window_size {
            self.token_context.remove(0);
        }
        
        Ok(())
    }
    
    /// Get raw token embeddings from main model (placeholder interface)
    ///
    /// In production, this would interface with the main LLM's embedding layer
    pub fn get_raw_embeddings(&self, _token_ids: &[u32]) -> Result<Tensor<Backend, 2>, ScoutGateError> {
        // 这里应该从主模型的embedding层获取embeddings
        // 目前使用随机初始化作为占位符，实际应该调用主模型的embedding layer
        let batch_size = _token_ids.len();
        let embeddings = Tensor::random([batch_size, self.d_emb], burn::tensor::Distribution::Normal(0.0, 1.0), &self.device);
        
        Ok(embeddings)
    }
    
    /// Project token embeddings: d_emb -> d_proj
    pub fn project_embeddings(&self, embeddings: Tensor<Backend, 2>) -> Result<Tensor<Backend, 2>, ScoutGateError> {
        // 应用线性投影层
        let projected = self.projection_layer.forward(embeddings);
        Ok(projected)
    }
    
    /// Apply Layer Normalization to embeddings
    pub fn normalize_embeddings(&self, embeddings: Tensor<Backend, 2>) -> Result<Tensor<Backend, 2>, ScoutGateError> {
        // 应用层归一化
        let normalized = self.layer_norm.forward(embeddings);
        Ok(normalized)
    }
    
    /// Process current token context window
    ///
    /// Returns projected and normalized embeddings for all tokens in context
    pub fn process_context(&self) -> Result<Tensor<Backend, 2>, ScoutGateError> {
        // 如果context为空，返回错误
        if self.token_context.is_empty() {
            return Err(ScoutGateError::TokenEmbeddingError { message: "Context window is empty".to_string() });
        }
        
        // 获取原始embeddings
        let raw_embeddings = self.get_raw_embeddings(&self.token_context)?;
        
        // 应用投影
        let projected = self.project_embeddings(raw_embeddings)?;
        
        // 应用层归一化
        let normalized = self.normalize_embeddings(projected)?;
        
        Ok(normalized)
    }
    
    /// Get current context window size
    pub fn current_context_size(&self) -> usize {
        self.token_context.len()
    }
    
    /// Get current token IDs in context window
    pub fn get_context_tokens(&self) -> &[u32] {
        &self.token_context
    }
}