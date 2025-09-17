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
    pub fn new(eta: f64, gamma: f64, total_layers: usize) -> Self {
        Self {
            eta,
            gamma,
            total_layers,
        }
    }

    /// Create configuration for GPT-OSS-20B model
    pub fn for_gptoss20b() -> Self {
        use crate::constants::GPT_OSS_20B;
        Self {
            eta: 0.5,
            gamma: 0.1,
            total_layers: GPT_OSS_20B.total_layers,
        }
    }

    /// Create configuration for GPT-OSS-120B model
    pub fn for_gptoss120b() -> Self {
        use crate::constants::GPT_OSS_120B;
        Self {
            eta: 0.5,
            gamma: 0.1,
            total_layers: GPT_OSS_120B.total_layers,
        }
    }

    /// Create configuration for Phi-Tiny-MoE model (for testing)
    pub fn for_phi_tiny_moe() -> Self {
        use crate::constants::PHI_TINY_MOE;
        Self {
            eta: 0.5,
            gamma: 0.1,
            total_layers: PHI_TINY_MOE.total_layers,
        }
    }
}

impl Default for FusionConfig {
    /// Create default configuration suitable for most use cases
    fn default() -> Self {
        Self {
            eta: 0.5,         // Equal weighting of EWMA and ScoutGate
            gamma: 0.1,       // Moderate decay for forward-causal weights
            total_layers: 24, // Default for GPT-OSS-20B
        }
    }
}
