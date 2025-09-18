//! ScoutGate hyperparameters
//!
//! This module contains the default hyperparameter values for ScoutGate predictor
//! as specified in the documentation.

/// Number of recent tokens to use for context (m in documentation)
pub const DEFAULT_CONTEXT_WINDOW_SIZE: usize = 8;

/// Projection dimension for token embeddings (d_proj in documentation)
pub const DEFAULT_PROJECTION_DIM: usize = 128;

/// Layer embedding dimension (d_ℓ in documentation)
pub const DEFAULT_LAYER_EMBEDDING_DIM: usize = 64;

/// Expert embedding dimension for two-tower architecture (d_e in documentation)
pub const DEFAULT_EXPERT_EMBEDDING_DIM: usize = 64;

/// Hidden dimension for context processing (d_h in documentation)
pub const DEFAULT_HIDDEN_DIM: usize = 256;

/// Low-rank dimension for two-tower mappings (d' in documentation)
pub const DEFAULT_LOW_RANK_DIM: usize = 64;

/// Layer normalization epsilon for numerical stability
pub const DEFAULT_LAYER_NORM_EPS: f64 = 1e-5;
