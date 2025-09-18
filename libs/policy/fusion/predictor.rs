//! Probability fusion predictor implementation
//!
//! This module implements the probability fusion algorithm from the documentation:
//!
//! 1. Base fusion: p^{base}_{e,ℓ}(t) := (1-η)·p̂^{EWMA}_{e,ℓ}(t) + η·p̂^{SG}_{e,ℓ}(t)
//! 2. Reuse distance: D(ℓ|ℓ(t)) calculation for forward-causal weights
//! 3. Forward-causal weights: W(ℓ|ℓ(t)) := e^{-γ·D(ℓ|ℓ(t))}
//! 4. Final fusion: p^{fuse}_{e,ℓ}(t) := p^{base}_{e,ℓ}(t) · W(ℓ|ℓ(t))

use super::error::FusionError;
use crate::timer::Timer;
use std::sync::{Arc, RwLock};

/// Probability fusion predictor
///
/// Combines EWMA and ScoutGate predictions with forward-causal weighting
/// based on layer reuse distances. This produces the final fused probabilities
/// that are used by the watermark algorithm for cache decisions.
pub struct ProbabilityFusion {
    /// η parameter for EWMA-ScoutGate blending (0.0 = pure EWMA, 1.0 = pure ScoutGate)
    pub eta: f64,

    /// γ parameter for reuse distance decay in forward-causal weights
    pub gamma: f64,

    /// Timer reference for accessing layer information and time calculations
    timer: Arc<RwLock<Timer>>,
}

impl ProbabilityFusion {
    /// Create a new probability fusion predictor
    pub fn new(eta: f64, gamma: f64, timer: Arc<RwLock<Timer>>) -> Self {
        Self { eta, gamma, timer }
    }

    /// Create fusion predictor from model type
    pub fn from_model(_model_type: crate::constants::ModelType, timer: Arc<RwLock<Timer>>) -> Self {
        Self::new(0.0, 0.05, timer)
    }

    /// Fuse EWMA and ScoutGate predictions with forward-causal weights
    ///
    /// This implements the complete fusion pipeline:
    /// 1. Base fusion of EWMA and ScoutGate predictions
    /// 2. Forward-causal weighting based on reuse distances
    /// 3. Final fused probabilities output
    ///
    /// # Arguments
    /// * `ewma_predictions` - EWMA probability predictions using ExpertProbability
    /// * `scoutgate_predictions` - ScoutGate probability predictions using ExpertProbability
    ///
    /// # Returns
    /// * `crate::ExpertProbability` - Final fused probabilities p^{fuse}_{e,ℓ}(t)
    pub fn fuse(
        &self,
        ewma_predictions: &crate::ExpertProbability,
        scoutgate_predictions: &crate::ExpertProbability,
    ) -> Result<crate::ExpertProbability, FusionError> {
        // Create output ExpertProbability with same dimensions as inputs
        let mut result = crate::ExpertProbability::new(
            ewma_predictions.inner.len(),
            ewma_predictions.inner[0].len(),
        );

        // Cache input references for cleaner access
        let ewma_prob = &ewma_predictions.inner;
        let sg_prob = &scoutgate_predictions.inner;

        // Pre-calculate causal weights for all layers to optimize performance
        let causal_weights: Vec<f64> = (0..ewma_prob.len())
            .map(|layer_id| {
                let reuse_distance = self.calculate_reuse_distance(layer_id);
                self.calculate_causal_weight(reuse_distance)
            })
            .collect();

        // Elegant fusion: iterate through layers with pre-computed weights
        for (layer_id, causal_weight) in causal_weights.iter().enumerate() {
            let ewma_layer = &ewma_prob[layer_id];
            let sg_layer = &sg_prob[layer_id];

            // Process all experts in current layer with vectorized approach
            for expert_id in 0..ewma_layer.len() {
                let ewma_val = ewma_layer[expert_id].unwrap_or(0.0);
                let sg_val = sg_layer[expert_id].unwrap_or(0.0);

                // Single-step fusion: base fusion + causal weighting
                // p^{fuse}_{e,ℓ}(t) = [(1-η)·p̂^{EWMA} + η·p̂^{SG}] · W(ℓ|ℓ(t))
                let fused_prob = ((1.0 - self.eta) * ewma_val + self.eta * sg_val) * causal_weight;

                result.set(layer_id, expert_id, fused_prob);
            }
        }

        Ok(result)
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
    ///
    /// # Returns
    /// * `usize` - Reuse distance D(ℓ|ℓ(t))
    pub fn calculate_reuse_distance(&self, target_layer: usize) -> usize {
        let timer = self.timer.read().unwrap();
        let current_layer = timer.current_layer();
        if target_layer >= current_layer {
            // Future layers in the current token
            target_layer - current_layer
        } else {
            // Layers in the next token (finish current token to L, then next token from 0 to target)
            (timer.total_layers() - current_layer) + target_layer
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
}
