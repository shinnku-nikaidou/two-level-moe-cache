/// Configuration for EWMA predictor
#[derive(Debug, Clone)]
pub struct EwmaConfig {
    /// EWMA smoothing parameter α ∈ (0,1]
    pub alpha: f64,

    /// Initialization value p₀ for new expert-layer pairs
    pub initialization_value: f64,
}

impl Default for EwmaConfig {
    fn default() -> Self {
        Self {
            alpha: crate::constants::ALPHA, // Use global default
            initialization_value: 0.1,      // Small positive initialization
        }
    }
}

impl EwmaConfig {
    /// Create a new EWMA configuration
    pub fn new(alpha: f64, initialization_value: f64) -> Self {
        Self {
            alpha,
            initialization_value,
        }
    }

    /// Create configuration with custom alpha, default initialization value
    pub fn with_alpha(alpha: f64) -> Self {
        Self {
            alpha,
            initialization_value: 0.1,
        }
    }

    /// Create configuration with custom initialization value, default alpha
    pub fn with_init_value(initialization_value: f64) -> Self {
        Self {
            alpha: crate::constants::ALPHA,
            initialization_value,
        }
    }

    /// Validate the configuration parameters
    pub fn validate(&self) -> Result<(), super::error::EwmaError> {
        if self.alpha <= 0.0 || self.alpha > 1.0 {
            return Err(super::error::EwmaError::InvalidAlpha(self.alpha));
        }

        if self.initialization_value < 0.0 || self.initialization_value > 1.0 {
            return Err(super::error::EwmaError::InvalidInitValue(
                self.initialization_value,
            ));
        }

        Ok(())
    }
}
