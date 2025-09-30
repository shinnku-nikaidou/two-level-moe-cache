//! Error handling for ScoutGate predictor
//!
//! This module defines error types specific to ScoutGate operations,
//! following the same pattern as other predictor modules.

use std::fmt;

/// Error types for ScoutGate predictor operations
#[derive(Debug, Clone, PartialEq)]
pub enum ScoutGateError {
    /// Configuration validation errors
    ConfigurationError { message: String },

    /// Model-related errors (e.g., invalid layer/expert indices)
    ModelError { message: String },

    /// Token processing errors
    TokenProcessingError { message: String },

    /// Timer integration errors
    TimerError { message: String },

    /// Embedding or neural network computation errors  
    ComputationError { message: String },

    /// Resource allocation errors (memory, GPU, etc.)
    ResourceError { message: String },

    /// Token embedding processing errors
    TokenEmbeddingError { message: String },

    /// Layer embedding management errors
    LayerEmbeddingError { message: String },

    /// Context processing pipeline errors
    ContextProcessingError { message: String },

    /// Expert embedding store errors
    ExpertEmbeddingError { message: String },

    /// Two-tower scoring errors
    TwoTowerError { message: String },

    /// Tensor shape or dimension mismatch errors
    DimensionError { expected: String, actual: String },

    /// Index out of bounds errors
    IndexError { index: usize, max: usize },
}

impl fmt::Display for ScoutGateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScoutGateError::ConfigurationError { message } => {
                write!(f, "ScoutGate configuration error: {}", message)
            }
            ScoutGateError::ModelError { message } => {
                write!(f, "ScoutGate model error: {}", message)
            }
            ScoutGateError::TokenProcessingError { message } => {
                write!(f, "ScoutGate token processing error: {}", message)
            }
            ScoutGateError::TimerError { message } => {
                write!(f, "ScoutGate timer error: {}", message)
            }
            ScoutGateError::ComputationError { message } => {
                write!(f, "ScoutGate computation error: {}", message)
            }
            ScoutGateError::ResourceError { message } => {
                write!(f, "ScoutGate resource error: {}", message)
            }
            ScoutGateError::TokenEmbeddingError { message } => {
                write!(f, "ScoutGate token embedding error: {}", message)
            }
            ScoutGateError::LayerEmbeddingError { message } => {
                write!(f, "ScoutGate layer embedding error: {}", message)
            }
            ScoutGateError::ContextProcessingError { message } => {
                write!(f, "ScoutGate context processing error: {}", message)
            }
            ScoutGateError::ExpertEmbeddingError { message } => {
                write!(f, "ScoutGate expert embedding error: {}", message)
            }
            ScoutGateError::TwoTowerError { message } => {
                write!(f, "ScoutGate two-tower error: {}", message)
            }
            ScoutGateError::DimensionError { expected, actual } => {
                write!(
                    f,
                    "ScoutGate dimension error: expected {}, got {}",
                    expected, actual
                )
            }
            ScoutGateError::IndexError { index, max } => {
                write!(
                    f,
                    "ScoutGate index error: index {} out of bounds (max {})",
                    index, max
                )
            }
        }
    }
}

impl std::error::Error for ScoutGateError {}
