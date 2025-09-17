//! Error types for probability fusion operations
//!
//! This module defines error types that can occur during probability fusion,
//! including runtime fusion errors.

/// Error types for probability fusion operations
#[derive(Debug, Clone, PartialEq)]
pub enum FusionError {
    /// Invalid current layer (must be < total_layers)
    InvalidCurrentLayer {
        current_layer: usize,
        total_layers: usize,
    },

    /// Mismatched prediction maps (different expert keys)
    MismatchedPredictions {
        ewma_keys: usize,
        scoutgate_keys: usize,
    },

    /// Invalid probability value (must be in [0,1])
    InvalidProbability {
        expert_key: String,
        probability: f64,
    },
}

impl std::fmt::Display for FusionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FusionError::InvalidCurrentLayer {
                current_layer,
                total_layers,
            } => {
                write!(
                    f,
                    "Invalid current layer {} (must be < {})",
                    current_layer, total_layers
                )
            }
            FusionError::MismatchedPredictions {
                ewma_keys,
                scoutgate_keys,
            } => {
                write!(
                    f,
                    "Mismatched prediction maps: EWMA has {} keys, ScoutGate has {} keys",
                    ewma_keys, scoutgate_keys
                )
            }
            FusionError::InvalidProbability {
                expert_key,
                probability,
            } => {
                write!(
                    f,
                    "Invalid probability {} for expert {} (must be in [0,1])",
                    probability, expert_key
                )
            }
        }
    }
}

impl std::error::Error for FusionError {}
