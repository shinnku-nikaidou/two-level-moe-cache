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
        }
    }
}

impl std::error::Error for ScoutGateError {}

/// Convert timer errors to ScoutGate errors
impl From<crate::timer::TimerError> for ScoutGateError {
    fn from(error: crate::timer::TimerError) -> Self {
        ScoutGateError::TimerError {
            message: error.to_string(),
        }
    }
}
