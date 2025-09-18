//! Watermark algorithm implementation
//!
//! This module implements the dual watermark algorithm from the documentation:
//!
//! 1. Benefit density calculation: b^G = p^fuse * C^G / S, b^R = p^fuse * C^R / S  
//! 2. Watermark updates: λ_G ← [λ_G + η_G(usage - K_G)]_+, λ_R ← [λ_R + η_R(usage - K_R)]_+
//! 3. Cache decisions: Keep in tier iff b >= λ

use crate::constants::ModelType;
use serde::{Deserialize, Serialize};

/// Cache decision for an expert
#[derive(Debug, Clone, PartialEq)]
pub enum CacheDecision {
    /// Keep expert in VRAM  
    KeepInVRAM,
    /// Demote expert from VRAM to RAM
    DemoteToRAM,
    /// Keep expert in RAM
    KeepInRAM,
    /// Evict expert from RAM to disk
    EvictToDisk,
    /// Load expert to RAM from disk
    LoadToRAM,
    /// Promote expert to VRAM from RAM
    PromoteToVRAM,
}

/// Memory tier for expert residence
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryTier {
    Vram = 0, // GPU memory - fastest, most limited
    Ram = 1,  // System memory - fast, moderate capacity
    Disk = 2, // NVMe/SSD storage - slower, largest capacity
}

pub struct ExpertState {
    pub inner: Vec<Vec<MemoryTier>>, // [layer][expert_idx] -> MemoryTier
}

impl ExpertState {
    /// Create a new ExpertState with given dimensions
    pub fn new(num_layers: usize, num_experts: usize) -> Self {
        let inner = vec![vec![MemoryTier::Disk; num_experts]; num_layers];
        Self { inner }
    }

    pub fn with_model(model_type: ModelType) -> Self {
        let config: crate::constants::ModelConfig = model_type.into();
        Self::new(config.total_layers, config.experts_per_layer)
    }

    /// Get memory tier for specific expert-layer pair
    pub fn get(&self, layer_id: usize, expert_id: usize) -> MemoryTier {
        self.inner[layer_id][expert_id]
    }

    /// Set memory tier for specific expert-layer pair
    pub fn set(&mut self, layer_id: usize, expert_id: usize, tier: MemoryTier) {
        if layer_id < self.inner.len() && expert_id < self.inner[layer_id].len() {
            self.inner[layer_id][expert_id] = tier;
        }
    }
}

/// Dual watermark algorithm implementation  
///
/// Manages two-tier expert caching using benefit density and adaptive watermarks.
/// Receives fused probabilities and produces cache decisions based on watermark thresholds.
pub struct WatermarkAlgorithm {
    /// VRAM capacity in mb (K_G)
    vram_capacity: usize,

    /// RAM capacity in mb (K_R)  
    ram_capacity: usize,

    /// VRAM watermark learning rate (η_G)
    vram_learning_rate: f64,

    /// RAM watermark learning rate (η_R)
    ram_learning_rate: f64,

    /// Cost of promoting expert from RAM to VRAM (C^G)
    cost_g: f64,

    /// Cost of loading expert from NVMe to RAM (C^R)  
    cost_r: f64,

    /// Expert size in mb (all experts assumed to have same size)
    expert_size: usize,

    /// Current watermark values (λ_G, λ_R)
    vram_watermark: f64,
    ram_watermark: f64,
}

impl WatermarkAlgorithm {
    /// Create a new watermark algorithm instance
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vram_capacity: usize,
        ram_capacity: usize,
        vram_learning_rate: f64,
        ram_learning_rate: f64,
        cost_g: f64,
        cost_r: f64,
        expert_size: usize,
    ) -> Self {
        Self {
            vram_capacity,
            ram_capacity,
            vram_learning_rate,
            ram_learning_rate,
            cost_g,
            cost_r,
            expert_size,
            vram_watermark: 0.0,
            ram_watermark: 0.0,
        }
    }

    /// Create watermark algorithm from model type
    pub fn from_model(
        model_type: ModelType,
        vram_capacity_mb: usize,
        ram_capacity_mb: usize,
    ) -> Self {
        let (vram_lr, ram_lr) = match model_type {
            ModelType::GptOss20B => (0.01, 0.01),
            ModelType::GptOss120B => (0.005, 0.005),
            ModelType::PhiTinyMoe => (0.01, 0.01),
        };

        Self::new(
            vram_capacity_mb * 1024 * 1024,
            ram_capacity_mb * 1024 * 1024,
            vram_lr,
            ram_lr,
            1.0,         // cost_g: RAM to VRAM cost
            10.0,        // cost_r: NVMe to RAM cost
            1024 * 1024, // expert_size: 1MB default
        )
    }

    /// Make cache decisions based on fused probabilities
    ///
    /// This implements the core watermark decision logic:
    /// - Calculate benefit densities for each expert
    /// - Compare against current watermarks  
    /// - Generate appropriate cache decisions
    pub fn make_cache_decisions(&mut self, fused_prob: &crate::ExpertProbability) -> ExpertState {
        // Directly construct ExpertState using functional style
        // Access inner data directly since it's now public
        let inner = fused_prob
            .inner
            .iter()
            .map(|layer_probs| {
                layer_probs
                    .iter()
                    .map(|prob| {
                        // Extract probability value, default to 0.0 if None
                        let prob_value = prob.unwrap_or(0.0);

                        // Calculate benefit densities using configuration
                        let size = self.expert_size as f64;
                        let vram_benefit = prob_value * self.cost_g / size;
                        let ram_benefit = prob_value * self.cost_r / size;

                        // Make cache decision based on watermarks
                        if vram_benefit >= self.vram_watermark {
                            MemoryTier::Vram
                        } else if ram_benefit >= self.ram_watermark {
                            MemoryTier::Ram
                        } else {
                            MemoryTier::Disk
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        ExpertState { inner }
    }

    /// Update watermarks using subgradient method
    ///
    /// Implements: λ_G ← [λ_G + η_G(usage - K_G)]_+, λ_R ← [λ_R + η_R(usage - K_R)]_+
    /// Note: Since we don't track actual memory usage, this is a simplified version
    /// that just applies learning rate decay
    fn update_watermarks(&mut self) {
        // Simplified watermark updates without actual usage tracking
        // In practice, this could be enhanced with feedback from the actual cache manager

        // For now, we keep the watermarks static or apply simple decay
        // Future enhancement: receive usage feedback from cache manager
        self.vram_watermark = (self.vram_watermark * 0.99).max(0.0); // Small decay
        self.ram_watermark = (self.ram_watermark * 0.99).max(0.0); // Small decay
    }
}
