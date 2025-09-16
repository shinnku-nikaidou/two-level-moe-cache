//! Configuration module for ScoutGate predictor
//!
//! This module defines configuration parameters for the ScoutGate semantic prediction
//! algorithm as described in the documentation. For now, this provides a basic
//! configuration structure that can be extended when the full implementation is added.

/// Configuration for ScoutGate predictor parameters
///
/// Based on the ScoutGate algorithm specification in the documentation:
/// - Uses recent m tokens for context (default m=8)
/// - Projects embeddings to lower dimension (default d_proj=128)
/// - Uses two-tower architecture for expert scoring
#[derive(Debug, Clone)]
pub struct ScoutGateConfig {
    /// Number of recent tokens to use for context (m in documentation)
    pub context_window_size: usize,

    /// Projection dimension for token embeddings (d_proj in documentation)
    pub projection_dim: usize,

    /// Layer embedding dimension (d_ℓ in documentation)
    pub layer_embedding_dim: usize,

    /// Expert embedding dimension for two-tower architecture (d_e in documentation)
    pub expert_embedding_dim: usize,

    /// Hidden dimension for context processing (d_h in documentation)
    pub hidden_dim: usize,

    /// Low-rank dimension for two-tower mappings (d' in documentation)
    pub low_rank_dim: usize,

    /// Layer normalization epsilon for numerical stability
    pub layer_norm_eps: f64,
}

impl ScoutGateConfig {
    /// Create a new ScoutGate configuration
    pub fn new(
        context_window_size: usize,
        projection_dim: usize,
        layer_embedding_dim: usize,
        expert_embedding_dim: usize,
        hidden_dim: usize,
        low_rank_dim: usize,
        layer_norm_eps: f64,
    ) -> Self {
        Self {
            context_window_size,
            projection_dim,
            layer_embedding_dim,
            expert_embedding_dim,
            hidden_dim,
            low_rank_dim,
            layer_norm_eps,
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), crate::scoutgate::error::ScoutGateError> {
        use crate::scoutgate::error::ScoutGateError;

        if self.context_window_size == 0 {
            return Err(ScoutGateError::ConfigurationError {
                message: "context_window_size must be > 0".to_string(),
            });
        }

        if self.projection_dim == 0 {
            return Err(ScoutGateError::ConfigurationError {
                message: "projection_dim must be > 0".to_string(),
            });
        }

        if self.layer_embedding_dim == 0 {
            return Err(ScoutGateError::ConfigurationError {
                message: "layer_embedding_dim must be > 0".to_string(),
            });
        }

        if self.expert_embedding_dim == 0 {
            return Err(ScoutGateError::ConfigurationError {
                message: "expert_embedding_dim must be > 0".to_string(),
            });
        }

        if self.hidden_dim == 0 {
            return Err(ScoutGateError::ConfigurationError {
                message: "hidden_dim must be > 0".to_string(),
            });
        }

        if self.low_rank_dim == 0 {
            return Err(ScoutGateError::ConfigurationError {
                message: "low_rank_dim must be > 0".to_string(),
            });
        }

        if self.layer_norm_eps <= 0.0 {
            return Err(ScoutGateError::ConfigurationError {
                message: "layer_norm_eps must be > 0.0".to_string(),
            });
        }

        Ok(())
    }
}

impl Default for ScoutGateConfig {
    /// Default configuration based on documentation specifications
    fn default() -> Self {
        Self {
            context_window_size: 8,   // m = 8 tokens
            projection_dim: 128,      // d_proj = 128
            layer_embedding_dim: 64,  // d_ℓ reasonable default
            expert_embedding_dim: 64, // d_e reasonable default
            hidden_dim: 256,          // d_h reasonable default
            low_rank_dim: 64,         // d' reasonable default
            layer_norm_eps: 1e-5,     // Standard epsilon
        }
    }
}
