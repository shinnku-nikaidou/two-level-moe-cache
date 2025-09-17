use std::collections::HashMap;
use std::rc::Rc;

use crate::AbstractExpert;
use crate::constants::ModelConfig;
use crate::timer::Timer;

use super::config::EwmaConfig;
use super::error::EwmaError;

/// EWMA (Exponentially Weighted Moving Average) predictor for expert activation probabilities
///
/// Implements the layer-local EWMA algorithm from the documentation using 0-based indexing.
/// Maintains activation probability estimates for each expert-layer pair using exponential
/// weighted moving averages with a shared Timer for layer-local clock management.
pub struct EwmaPredictor {
    /// Shared timer for layer-local time management
    timer: Rc<Timer>,

    /// EWMA smoothing parameter α ∈ (0,1]
    alpha: f64,

    /// Model configuration (layers, experts per layer, etc.)
    config: ModelConfig,

    /// Sparse storage for EWMA values: AbstractExpert -> probability
    /// Only stores experts that have been activated at least once
    ewma_values: HashMap<AbstractExpert, f64>,
}

impl EwmaPredictor {
    /// Create a new EWMA predictor with shared timer
    pub fn new(
        timer: Rc<Timer>,
        config: ModelConfig,
        ewma_config: EwmaConfig,
    ) -> Result<Self, EwmaError> {
        // Validate configuration
        ewma_config.validate()?;

        Ok(EwmaPredictor {
            timer,
            alpha: ewma_config.alpha,
            config,
            ewma_values: HashMap::new(),
        })
    }

    /// Create EWMA predictor for GPT-OSS-20B model
    pub fn for_gptoss20b(timer: Rc<Timer>) -> Result<Self, EwmaError> {
        use crate::constants::GPT_OSS_20B;
        Self::new(timer, GPT_OSS_20B.clone(), EwmaConfig::default())
    }

    /// Create EWMA predictor for GPT-OSS-120B model  
    pub fn for_gptoss120b(timer: Rc<Timer>) -> Result<Self, EwmaError> {
        use crate::constants::GPT_OSS_120B;
        Self::new(timer, GPT_OSS_120B.clone(), EwmaConfig::default())
    }

    /// Create EWMA predictor for Phi-Tiny-MoE model (for testing)
    pub fn for_phi_tiny_moe(timer: Rc<Timer>) -> Result<Self, EwmaError> {
        use crate::constants::PHI_TINY_MOE;
        Self::new(timer, PHI_TINY_MOE.clone(), EwmaConfig::default())
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
        let current_layer = self.timer.current_layer();

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
            let new_ewma = if let Some(current_ewma) = self.ewma_values.get(&expert) {
                // Subsequent update: p̂_{e,ℓ}^{EWMA}(t) = (1-α)p̂_{e,ℓ}^{EWMA}(t⁻) + α·p̂_{e,ℓ}^{HIT}(t)
                (1.0 - self.alpha) * current_ewma + self.alpha * hit_indicator
            } else {
                // First encounter: use activation value directly (0 or 1)
                hit_indicator
            };

            // Store updated value
            self.ewma_values.insert(expert, new_ewma);
        }

        Ok(())
    }

    /// Get EWMA probability estimate for a specific expert-layer pair
    /// Returns 0.0 for experts that have never been encountered
    pub fn get_probability(&self, expert: AbstractExpert) -> f64 {
        self.ewma_values.get(&expert).copied().unwrap_or(0.0) // Return 0.0 for never-encountered experts
    }

    /// Get EWMA probability estimates for all experts in a specific layer
    pub fn get_layer_probabilities(&self, layer_id: usize) -> HashMap<usize, f64> {
        let mut layer_probs = HashMap::new();

        // Get all abstract experts for this layer
        let layer_experts = AbstractExpert::layer_experts(layer_id, self.config.experts_per_layer);

        for expert in layer_experts {
            let probability = self.get_probability(expert);
            layer_probs.insert(expert.expert_id, probability);
        }

        layer_probs
    }

    /// Get all EWMA values as a reference to the internal HashMap
    pub fn get_all_probabilities(&self) -> &HashMap<AbstractExpert, f64> {
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

    /// Get number of expert-layer pairs currently tracked
    pub fn num_tracked_experts(&self) -> usize {
        self.ewma_values.len()
    }

    /// Clear all EWMA values (reset to initialization state)
    pub fn reset(&mut self) {
        self.ewma_values.clear();
    }
}
