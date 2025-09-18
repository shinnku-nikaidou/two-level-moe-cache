use crate::constants::{ModelConfig, ModelType};
use crate::timer::Timer;
use crate::{AbstractExpert, ExpertProbability};
use std::sync::{Arc, RwLock};

use super::error::EwmaError;

/// EWMA (Exponentially Weighted Moving Average) predictor for expert activation probabilities
///
/// Implements the layer-local EWMA algorithm from the documentation using 0-based indexing.
/// Maintains activation probability estimates for each expert-layer pair using exponential
/// weighted moving averages with a shared Timer for layer-local clock management.
pub struct EwmaPredictor {
    /// Shared timer for layer-local time management
    timer: Arc<RwLock<Timer>>,

    /// EWMA smoothing parameter α ∈ (0,1]
    alpha: f64,

    /// Model configuration (layers, experts per layer, etc.)
    config: ModelConfig,

    /// Dense storage for EWMA values using ExpertProbability
    /// Stores probability estimates for all expert-layer pairs
    ewma_values: ExpertProbability,
}

impl EwmaPredictor {
    /// Create a new EWMA predictor with shared timer
    pub fn new(
        timer: Arc<RwLock<Timer>>,
        config: ModelConfig,
        alpha: f64,
    ) -> Result<Self, EwmaError> {
        // Validate alpha parameter
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(EwmaError::InvalidAlpha(alpha));
        }

        Ok(EwmaPredictor {
            timer,
            alpha,
            config: config.clone(),
            ewma_values: ExpertProbability::new(config.total_layers, config.experts_per_layer),
        })
    }

    /// Create EWMA predictor from model type
    pub fn from_model(timer: Arc<RwLock<Timer>>, model_type: ModelType) -> Self {
        let config: ModelConfig = model_type.into();
        Self::new(timer, config, crate::constants::ALPHA)
            .expect("Default EWMA configuration should be valid")
    }

    /// Update EWMA values for the current executing layer
    ///
    /// This implements the layer-local EWMA update:
    /// p̂_{e,ℓ}^{EWMA}(t) = (1-α)p̂_{e,ℓ}^{EWMA}(t⁻) + α·p̂_{e,ℓ}^{HIT}(t)
    ///
    /// Updates only occur when the timer indicates the corresponding layer is executing.
    pub fn update_layer_activations(
        &mut self,
        activated_experts: &[usize],
    ) -> Result<(), EwmaError> {
        // Get current layer from shared timer
        let current_layer = self.timer.read().unwrap().current_layer();

        // Get all possible experts for this layer
        let layer_experts =
            AbstractExpert::layer_experts(current_layer, self.config.experts_per_layer);

        // Update EWMA for each expert in this layer
        for expert in layer_experts {
            // Determine if this expert was activated (binary indicator)
            let is_activated = activated_experts.contains(&expert.expert_id);
            let hit_indicator = if is_activated { 1.0 } else { 0.0 };

            // Apply EWMA update: for first encounter, use the activation value directly
            // For subsequent updates, use the EWMA formula
            let current_ewma = self.ewma_values.get(expert.layer_id, expert.expert_id);
            let new_ewma = if let Some(current_value) = current_ewma {
                // Subsequent update: p̂_{e,ℓ}^{EWMA}(t) = (1-α)p̂_{e,ℓ}^{EWMA}(t⁻) + α·p̂_{e,ℓ}^{HIT}(t)
                (1.0 - self.alpha) * current_value + self.alpha * hit_indicator
            } else {
                // First encounter: use activation value directly (0 or 1)
                hit_indicator
            };

            // Store updated value
            self.ewma_values
                .set(expert.layer_id, expert.expert_id, new_ewma);
        }

        Ok(())
    }

    /// Get EWMA probability estimate for a specific expert-layer pair
    /// Returns 0.0 for experts that have never been encountered
    pub fn get_probability(&self, expert: AbstractExpert) -> f64 {
        self.ewma_values
            .get(expert.layer_id, expert.expert_id)
            .unwrap_or(0.0)
    }

    /// Get EWMA probability estimates for all experts in a specific layer
    pub fn get_layer_probabilities(&self, layer_id: usize) -> Vec<crate::Probability> {
        if layer_id >= self.config.total_layers {
            return vec![None; self.config.experts_per_layer];
        }

        let mut layer_probs = Vec::new();
        for expert_id in 0..self.config.experts_per_layer {
            let prob = self.ewma_values.get(layer_id, expert_id);
            layer_probs.push(prob);
        }

        layer_probs
    }

    /// Get all EWMA values as a reference to the internal ExpertProbability
    pub fn get_all_probabilities(&self) -> &ExpertProbability {
        &self.ewma_values
    }

    /// Calculate effective window size: N_eff ≈ (2-α)/α
    pub fn effective_window_size(&self) -> f64 {
        (2.0 - self.alpha) / self.alpha
    }

    /// Calculate theoretical variance bound for a given true probability p
    /// Var[p̂^{EWMA}] = p(1-p) * α/(2-α)
    pub fn variance_bound(&self, true_probability: f64) -> f64 {
        true_probability * (1.0 - true_probability) * self.alpha / (2.0 - self.alpha)
    }

    /// Get current alpha parameter
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
}
