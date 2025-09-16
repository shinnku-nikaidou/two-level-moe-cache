//! Mock implementation logic for cache decisions
//!
//! This module contains temporary mock implementations that simulate
//! the behavior of the full policy layer components until they are integrated.

use policy::{ExpertKey as PolicyExpertKey, ExpertParamType};
use std::collections::HashMap;

use super::manager::TwoTireWmExpertCacheManager;

/// Mock EWMA prediction and cache decision logic
pub fn get_cache_decision(
    mock_ewma_probs: &mut HashMap<PolicyExpertKey, f64>,
    policy_key: &PolicyExpertKey,
) -> (bool, f64) {
    // Mock EWMA prediction (real implementation would delegate to policy layer)
    let ewma_prob = mock_ewma_probs.get(policy_key).copied().unwrap_or(0.3);

    // Simple cache decision: cache if probability > 0.5
    let should_cache = ewma_prob > 0.5;

    (should_cache, ewma_prob)
}

/// Mock activation tracking and EWMA updates
pub fn update_activations(
    mock_ewma_probs: &mut HashMap<PolicyExpertKey, f64>,
    current_layer: usize,
    activated_experts: &[usize],
) {
    // Mock activation tracking (real implementation would delegate to policy layer)
    // Update mock EWMA probabilities for all experts in current layer
    for expert_id in 0..8 {
        // Assume 8 experts per layer
        let policy_key =
            PolicyExpertKey::new(expert_id, current_layer, ExpertParamType::MLP1Weight);

        let activated = activated_experts.contains(&expert_id);

        // Simple EWMA update: p_new = 0.9 * p_old + 0.1 * hit
        let old_prob = mock_ewma_probs.get(&policy_key).copied().unwrap_or(0.0);
        let hit = if activated { 1.0 } else { 0.0 };
        let new_prob = 0.9 * old_prob + 0.1 * hit;

        mock_ewma_probs.insert(policy_key, new_prob);
    }
}

/// Mock statistics generation
pub fn get_stats(manager: &TwoTireWmExpertCacheManager) -> HashMap<String, f64> {
    let mut stats = HashMap::new();

    // Mock watermark values (real implementation would get from policy layer)
    stats.insert("vram_watermark".to_string(), 0.75);
    stats.insert("ram_watermark".to_string(), 0.25);

    // Timing info
    stats.insert("current_time".to_string(), manager.current_time() as f64);
    stats.insert("current_layer".to_string(), manager.current_layer() as f64);
    stats.insert(
        "total_experts_tracked".to_string(),
        manager.mock_ewma_probs.len() as f64,
    );

    stats
}
