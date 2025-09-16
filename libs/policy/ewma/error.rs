use crate::ExpertKey;
use crate::timer::TimerError;

/// Error types for EWMA operations
#[derive(Debug, Clone, PartialEq)]
pub enum EwmaError {
    InvalidAlpha(f64),
    InvalidInitValue(f64),
    TimerError(TimerError),
    ExpertNotFound(ExpertKey),
}

impl std::fmt::Display for EwmaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EwmaError::InvalidAlpha(alpha) => {
                write!(f, "Invalid alpha {} (must be in (0,1])", alpha)
            }
            EwmaError::InvalidInitValue(val) => {
                write!(f, "Invalid initialization value {} (must be in [0,1])", val)
            }
            EwmaError::TimerError(e) => write!(f, "Timer error: {}", e),
            EwmaError::ExpertNotFound(key) => {
                write!(
                    f,
                    "Expert not found: layer={}, expert={}",
                    key.layer_id, key.expert_id
                )
            }
        }
    }
}

impl std::error::Error for EwmaError {}

impl From<TimerError> for EwmaError {
    fn from(error: TimerError) -> Self {
        EwmaError::TimerError(error)
    }
}
