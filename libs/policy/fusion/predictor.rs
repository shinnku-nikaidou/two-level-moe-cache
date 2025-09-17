//! Probability fusion predictor implementation
//!
//! This module implements the probability fusion algorithm from the documentation:
//!
//! 1. Base fusion: p^{base}_{e,ℓ}(t) := (1-η)·p̂^{EWMA}_{e,ℓ}(t) + η·p̂^{SG}_{e,ℓ}(t)
//! 2. Reuse distance: D(ℓ|ℓ(t)) calculation for forward-causal weights
//! 3. Forward-causal weights: W(ℓ|ℓ(t)) := e^{-γ·D(ℓ|ℓ(t))}
//! 4. Final fusion: p^{fuse}_{e,ℓ}(t) := p^{base}_{e,ℓ}(t) · W(ℓ|ℓ(t))

use std::collections::HashMap;

use super::error::FusionError;
use crate::AbstractExpert;

/// Probability fusion predictor
///
/// Combines EWMA and ScoutGate predictions with forward-causal weighting
/// based on layer reuse distances. This produces the final fused probabilities
/// that are used by the watermark algorithm for cache decisions.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbabilityFusion {
    /// η parameter for EWMA-ScoutGate blending (0.0 = pure EWMA, 1.0 = pure ScoutGate)
    pub eta: f64,

    /// γ parameter for reuse distance decay in forward-causal weights
    pub gamma: f64,

    /// Total number of layers in the model (for reuse distance calculation)
    pub total_layers: usize,
}

impl ProbabilityFusion {
    /// Create a new probability fusion predictor
    pub fn new(eta: f64, gamma: f64, total_layers: usize) -> Self {
        Self {
            eta,
            gamma,
            total_layers,
        }
    }

    /// Create fusion predictor for GPT-OSS-20B model
    pub fn for_gptoss20b() -> Self {
        use crate::constants::GPT_OSS_20B;
        Self::new(0.5, 0.1, GPT_OSS_20B.total_layers)
    }

    /// Create fusion predictor for GPT-OSS-120B model
    pub fn for_gptoss120b() -> Self {
        use crate::constants::GPT_OSS_120B;
        Self::new(0.5, 0.1, GPT_OSS_120B.total_layers)
    }

    /// Create fusion predictor for Phi-Tiny-MoE model (for testing)
    pub fn for_phi_tiny_moe() -> Self {
        use crate::constants::PHI_TINY_MOE;
        Self::new(0.5, 0.1, PHI_TINY_MOE.total_layers)
    }

    /// Fuse EWMA and ScoutGate predictions with forward-causal weights
    ///
    /// This implements the complete fusion pipeline:
    /// 1. Base fusion of EWMA and ScoutGate predictions
    /// 2. Forward-causal weighting based on reuse distances
    /// 3. Final fused probabilities output
    ///
    /// # Arguments
    /// * `ewma_predictions` - EWMA probability predictions for experts
    /// * `scoutgate_predictions` - ScoutGate probability predictions for experts
    /// * `current_layer` - Currently executing layer (0-based)
    ///
    /// # Returns
    /// * `HashMap<AbstractExpert, f64>` - Final fused probabilities p^{fuse}_{e,ℓ}(t)
    pub fn fuse_predictions(
        &self,
        ewma_predictions: &HashMap<AbstractExpert, f64>,
        scoutgate_predictions: &HashMap<AbstractExpert, f64>,
        current_layer: usize,
    ) -> Result<HashMap<AbstractExpert, f64>, FusionError> {
        // Validate current layer
        if current_layer >= self.total_layers {
            return Err(FusionError::InvalidCurrentLayer {
                current_layer,
                total_layers: self.total_layers,
            });
        }

        // Get all expert keys from both prediction maps
        let all_keys: std::collections::HashSet<_> = ewma_predictions
            .keys()
            .chain(scoutgate_predictions.keys())
            .collect();

        let mut fused_predictions = HashMap::new();

        for expert_key in all_keys {
            // Get individual predictions (default to 0.0 if missing)
            let ewma_prob = ewma_predictions.get(expert_key).copied().unwrap_or(0.0);
            let scoutgate_prob = scoutgate_predictions
                .get(expert_key)
                .copied()
                .unwrap_or(0.0);

            // Validate probability values
            self.validate_probability(*expert_key, ewma_prob)?;
            self.validate_probability(*expert_key, scoutgate_prob)?;

            // Step 1: Base fusion
            // p^{base}_{e,ℓ}(t) := (1-η)·p̂^{EWMA}_{e,ℓ}(t) + η·p̂^{SG}_{e,ℓ}(t)
            let base_prob = (1.0 - self.eta) * ewma_prob + self.eta * scoutgate_prob;

            // Step 2: Calculate forward-causal weight
            // W(ℓ|ℓ(t)) := e^{-γ·D(ℓ|ℓ(t))}
            let reuse_distance = self.calculate_reuse_distance(expert_key.layer_id, current_layer);
            let causal_weight = self.calculate_causal_weight(reuse_distance);

            // Step 3: Final fusion
            // p^{fuse}_{e,ℓ}(t) := p^{base}_{e,ℓ}(t) · W(ℓ|ℓ(t))
            let fused_prob = base_prob * causal_weight;

            fused_predictions.insert(*expert_key, fused_prob);
        }

        Ok(fused_predictions)
    }

    /// Calculate layer reuse distance for forward-causal weights
    ///
    /// Implements the reuse distance calculation from the documentation:
    /// D(ℓ|ℓ(t)) = {
    ///   ℓ - ℓ(t),                    if ℓ >= ℓ(t) (future layers in current token)
    ///   (L - ℓ(t)) + ℓ,              if ℓ < ℓ(t) (next token layers)  
    /// }
    ///
    /// # Arguments
    /// * `target_layer` - Target layer ℓ
    /// * `current_layer` - Current executing layer ℓ(t)
    ///
    /// # Returns
    /// * `usize` - Reuse distance D(ℓ|ℓ(t))
    pub fn calculate_reuse_distance(&self, target_layer: usize, current_layer: usize) -> usize {
        if target_layer >= current_layer {
            // Future layers in the current token
            target_layer - current_layer
        } else {
            // Layers in the next token (finish current token to L, then next token from 0 to target)
            (self.total_layers - current_layer) + target_layer
        }
    }

    /// Calculate forward-causal weight from reuse distance
    ///
    /// Implements: W(ℓ|ℓ(t)) := e^{-γ·D(ℓ|ℓ(t))}
    ///
    /// # Arguments
    /// * `reuse_distance` - Distance D(ℓ|ℓ(t))
    ///
    /// # Returns
    /// * `f64` - Forward-causal weight W(ℓ|ℓ(t))
    pub fn calculate_causal_weight(&self, reuse_distance: usize) -> f64 {
        (-self.gamma * reuse_distance as f64).exp()
    }

    /// Get eta parameter (EWMA-ScoutGate blending factor)
    pub fn eta(&self) -> f64 {
        self.eta
    }

    /// Get gamma parameter (reuse distance decay factor)
    pub fn gamma(&self) -> f64 {
        self.gamma
    }

    /// Get total layers
    pub fn total_layers(&self) -> usize {
        self.total_layers
    }

    // Private helper methods

    /// Validate that a probability value is in [0, 1]
    fn validate_probability(
        &self,
        expert: AbstractExpert,
        probability: f64,
    ) -> Result<(), FusionError> {
        if !(0.0..=1.0).contains(&probability) {
            return Err(FusionError::InvalidProbability {
                expert_key: format!("{:?}", expert),
                probability,
            });
        }
        Ok(())
    }
}

impl Default for ProbabilityFusion {
    /// Create default fusion predictor suitable for most use cases
    fn default() -> Self {
        Self {
            eta: 0.5,         // Equal weighting of EWMA and ScoutGate
            gamma: 0.1,       // Moderate decay for forward-causal weights
            total_layers: 24, // Default for GPT-OSS-20B
        }
    }
}
