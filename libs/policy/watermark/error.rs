//! Error types for watermark algorithm operations

/// Error types for watermark algorithm operations
#[derive(Debug, Clone, PartialEq)]
pub enum WatermarkError {
    /// Invalid expert key
    InvalidExpertKey(String),

    /// Invalid prediction probability (must be in [0,1])
    InvalidProbability {
        expert_key: String,
        probability: f64,
    },

    /// Capacity constraint violation
    CapacityViolation {
        tier: String,
        required: f64,    // Changed to f64 for MB units
        available: f64,   // Changed to f64 for MB units
    },

    /// Expert not found in tracking state
    ExpertNotFound(String),
}

impl std::fmt::Display for WatermarkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WatermarkError::InvalidExpertKey(key) => {
                write!(f, "Invalid expert key: {}", key)
            }
            WatermarkError::InvalidProbability {
                expert_key,
                probability,
            } => {
                write!(
                    f,
                    "Invalid probability {} for expert {} (must be in [0,1])",
                    probability, expert_key
                )
            }
            WatermarkError::CapacityViolation {
                tier,
                required,
                available,
            } => {
                write!(
                    f,
                    "Capacity violation in {}: required {:.2} MB, available {:.2} MB",
                    tier, required, available
                )
            }
            WatermarkError::ExpertNotFound(key) => {
                write!(f, "Expert not found in tracking state: {}", key)
            }
        }
    }
}

impl std::error::Error for WatermarkError {}
