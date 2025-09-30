//! ScoutGate predictor module
//!
//! This module implements the ScoutGate semantic prediction algorithm as described
//! in the documentation. ScoutGate provides activation probability predictions for
//! MoE experts using recent token context and learned embeddings.
//!
//! **Current Status: PLACEHOLDER IMPLEMENTATION**
//! The current implementation is a placeholder that returns 1.0 for any expert key.
//!
//! The full implementation will include:
//! - Token embedding and projection layers
//! - Layer embeddings for different transformer layers  
//! - Two-tower architecture for expert scoring
//! - Context processing with sliding window of recent tokens
//! - Sigmoid activation for probability outputs
//! - Integration with the main model's token embeddings
//!
//! ## Architecture Overview
//!
//! Based on the documentation, ScoutGate should implement:
//!
//! 1. **Fetch Token**: Take recent m tokens (default m=8)
//! 2. **Embedding**: Use main model's token embedding  
//! 3. **Projection**: Project embeddings to lower dimension (d_proj=128)
//! 4. **Concatenate**: Combine with layer embeddings
//! 5. **Two-tower scoring**: Score experts using low-rank mappings
//! 6. **Output**: Sigmoid probabilities for all experts in all layers
//!
//! ## Usage
//!
//! ```rust
//! use policy::scoutgate::{ScoutGatePredictor, ScoutGateConfig};
//! use policy::constants::ModelType;
//! use policy::timer::Timer;
//! use std::sync::{Arc, RwLock};
//!
//! // Create timer and predictor
//! let timer = Arc::new(RwLock::new(Timer::new()));
//! let mut predictor = ScoutGatePredictor::from_model(timer, ModelType::SmallMoE);
//!
//! // Update with new token
//! predictor.update_token_context(123).expect("Failed to update token context");
//!
//! // Get predictions for all layers
//! let probabilities = predictor.get_probabilities();
//! ```

pub mod config;
pub mod error;
pub mod hyperparamater;
pub mod predictor;

// ScoutGate component modules
pub mod token_embedding;
pub mod layer_embedding;
pub mod context_processing;
pub mod expert_embedding;
pub mod two_tower;

// Re-export main types for easier access
pub use config::ScoutGateConfig;
pub use error::ScoutGateError;
pub use predictor::ScoutGatePredictor;
