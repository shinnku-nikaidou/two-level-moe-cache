/// Configuration for EWMA predictor
#[derive(Debug, Clone)]
pub struct EwmaConfig {
    /// EWMA smoothing parameter α ∈ (0,1]
    pub alpha: f64,
}

impl Default for EwmaConfig {
    fn default() -> Self {
        Self {
            alpha: crate::constants::ALPHA, // Use global default
        }
    }
}

impl EwmaConfig {
    /// Create a new EWMA configuration
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    /// Create configuration with custom alpha
    pub fn with_alpha(alpha: f64) -> Self {
        Self { alpha }
    }

    /// Validate the configuration parameters
    pub fn validate(&self) -> Result<(), super::error::EwmaError> {
        if self.alpha <= 0.0 || self.alpha > 1.0 {
            return Err(super::error::EwmaError::InvalidAlpha(self.alpha));
        }

        Ok(())
    }
}
