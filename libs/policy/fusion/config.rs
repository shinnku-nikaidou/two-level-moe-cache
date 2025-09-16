//! Configuration for probability fusion algorithm
//!
//! This module provides configuration structures for the probability fusion
//! component that blends EWMA and ScoutGate predictions with forward-causal weights.

/// Configuration for probability fusion algorithm
#[derive(Debug, Clone, PartialEq)]
pub struct FusionConfig {
    /// η parameter for EWMA-ScoutGate blending (0.0 = pure EWMA, 1.0 = pure ScoutGate)
    pub eta: f64,
    
    /// γ parameter for reuse distance decay in forward-causal weights
    pub gamma: f64,
    
    /// Total number of layers in the model (for reuse distance calculation)
    pub total_layers: usize,
}

impl FusionConfig {
    /// Create a new fusion configuration
    pub fn new(eta: f64, gamma: f64, total_layers: usize) -> Result<Self, FusionConfigError> {
        let config = Self {
            eta,
            gamma,
            total_layers,
        };
        config.validate()?;
        Ok(config)
    }

    /// Create default configuration suitable for most use cases
    pub fn default() -> Self {
        Self {
            eta: 0.5,     // Equal weighting of EWMA and ScoutGate
            gamma: 0.1,   // Moderate decay for forward-causal weights
            total_layers: 24, // Default for GPT-OSS-20B
        }
    }

    /// Create configuration for GPT-OSS-20B model
    pub fn for_gptoss20b() -> Self {
        use crate::constants::models::GPT_OSS_20B;
        Self {
            eta: 0.5,
            gamma: 0.1,
            total_layers: GPT_OSS_20B.total_layers,
        }
    }

    /// Create configuration for GPT-OSS-120B model
    pub fn for_gptoss120b() -> Self {
        use crate::constants::models::GPT_OSS_120B;
        Self {
            eta: 0.5,
            gamma: 0.1,
            total_layers: GPT_OSS_120B.total_layers,
        }
    }

    /// Create configuration for Phi-Tiny-MoE model (for testing)
    pub fn for_phi_tiny_moe() -> Self {
        use crate::constants::models::PHI_TINY_MOE;
        Self {
            eta: 0.5,
            gamma: 0.1,
            total_layers: PHI_TINY_MOE.total_layers,
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), FusionConfigError> {
        if !(0.0..=1.0).contains(&self.eta) {
            return Err(FusionConfigError::InvalidEta(self.eta));
        }
        
        if self.gamma <= 0.0 {
            return Err(FusionConfigError::InvalidGamma(self.gamma));
        }
        
        if self.total_layers == 0 {
            return Err(FusionConfigError::InvalidLayers(self.total_layers));
        }
        
        Ok(())
    }
}

/// Error types for fusion configuration
#[derive(Debug, Clone, PartialEq)]
pub enum FusionConfigError {
    /// Invalid eta parameter (must be in [0,1])
    InvalidEta(f64),
    
    /// Invalid gamma parameter (must be > 0)
    InvalidGamma(f64),
    
    /// Invalid total layers (must be > 0)
    InvalidLayers(usize),
}

impl std::fmt::Display for FusionConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FusionConfigError::InvalidEta(eta) => {
                write!(f, "Invalid eta parameter {} (must be in [0,1])", eta)
            }
            FusionConfigError::InvalidGamma(gamma) => {
                write!(f, "Invalid gamma parameter {} (must be > 0)", gamma)
            }
            FusionConfigError::InvalidLayers(layers) => {
                write!(f, "Invalid total_layers {} (must be > 0)", layers)
            }
        }
    }
}

impl std::error::Error for FusionConfigError {}