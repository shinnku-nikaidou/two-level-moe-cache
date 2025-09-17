//! Watermark algorithm implementation
//!
//! This module implements the dual watermark algorithm from the documentation:
//!
//! 1. Benefit density calculation: b^G = p^fuse * C^G / S, b^R = p^fuse * C^R / S  
//! 2. Watermark updates: λ_G ← [λ_G + η_G(usage - K_G)]_+, λ_R ← [λ_R + η_R(usage - K_R)]_+
//! 3. Cache decisions: Keep in tier iff b >= λ

use super::config::WatermarkConfig;
use super::error::WatermarkError;
use crate::AbstractExpert;
use crate::constants::ModelType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

/// Dual watermark algorithm implementation  
///
/// Manages two-tier expert caching using benefit density and adaptive watermarks.
/// Receives fused probabilities and produces cache decisions based on watermark thresholds.
pub struct WatermarkAlgorithm {
    /// Configuration parameters
    config: WatermarkConfig,

    /// Current watermark values (λ_G, λ_R)
    vram_watermark: f64,
    ram_watermark: f64,

    /// Current time step
    current_time: u64,

    /// Expert memory tier tracking
    expert_states: HashMap<AbstractExpert, MemoryTier>,

    /// Current memory usage
    vram_used_bytes: usize,
    ram_used_bytes: usize,
}

impl WatermarkAlgorithm {
    /// Create a new watermark algorithm instance
    pub fn new(config: WatermarkConfig) -> Self {
        config.validate().expect("Invalid watermark configuration");

        Self {
            config,
            vram_watermark: 0.0,
            ram_watermark: 0.0,
            current_time: 0,
            expert_states: HashMap::new(),
            vram_used_bytes: 0,
            ram_used_bytes: 0,
        }
    }

    /// Create watermark algorithm from model type
    pub fn from_model(
        model_type: ModelType,
        vram_capacity_mb: usize,
        ram_capacity_mb: usize,
    ) -> Self {
        let config = WatermarkConfig::from_model(model_type, vram_capacity_mb, ram_capacity_mb);
        Self::new(config)
    }

    /// Advance to next time step and update watermarks
    pub fn step(&mut self) {
        self.current_time += 1;
        self.update_watermarks();
    }

    /// Make cache decisions based on fused probabilities
    ///
    /// This implements the core watermark decision logic:
    /// - Calculate benefit densities for each expert
    /// - Compare against current watermarks  
    /// - Generate appropriate cache decisions
    pub fn make_cache_decisions(
        &mut self,
        fused_probabilities: &HashMap<AbstractExpert, f64>,
    ) -> Result<HashMap<AbstractExpert, CacheDecision>, WatermarkError> {
        let mut decisions = HashMap::new();

        for (expert, &probability) in fused_probabilities {
            // Validate probability
            if !(0.0..=1.0).contains(&probability) {
                return Err(WatermarkError::InvalidProbability {
                    expert_key: format!("{:?}", expert),
                    probability,
                });
            }

            // Get or create expert state
            let current_tier = {
                let expert_tier = self.get_or_create_expert_tier(*expert);
                *expert_tier
            };

            // Calculate benefit densities using configuration
            let size = self.config.expert_size_bytes as f64;
            let vram_benefit = probability * self.config.ram_to_vram_cost / size;
            let ram_benefit = probability * self.config.nvme_to_ram_cost / size;

            // Make cache decision based on current tier and watermarks
            let decision = match current_tier {
                MemoryTier::Vram => {
                    if vram_benefit >= self.vram_watermark {
                        CacheDecision::KeepInVRAM
                    } else {
                        CacheDecision::DemoteToRAM
                    }
                }
                MemoryTier::Ram => {
                    if ram_benefit >= self.ram_watermark {
                        CacheDecision::KeepInRAM
                    } else {
                        CacheDecision::EvictToDisk
                    }
                }
                MemoryTier::Disk => {
                    if ram_benefit >= self.ram_watermark {
                        CacheDecision::LoadToRAM
                    } else {
                        // Stay on disk
                        continue;
                    }
                }
            };

            decisions.insert(*expert, decision);
        }

        Ok(decisions)
    }

    /// Update watermarks using subgradient method
    ///
    /// Implements: λ_G ← [λ_G + η_G(usage - K_G)]_+, λ_R ← [λ_R + η_R(usage - K_R)]_+
    fn update_watermarks(&mut self) {
        // VRAM watermark update
        let vram_constraint = self.vram_used_bytes as f64 - self.config.vram_capacity as f64;
        self.vram_watermark =
            (self.vram_watermark + self.config.vram_learning_rate * vram_constraint).max(0.0);

        // RAM watermark update
        let ram_constraint = self.ram_used_bytes as f64 - self.config.ram_capacity as f64;
        self.ram_watermark =
            (self.ram_watermark + self.config.ram_learning_rate * ram_constraint).max(0.0);
    }

    /// Get or create expert tier for tracking
    fn get_or_create_expert_tier(&mut self, expert: AbstractExpert) -> &mut MemoryTier {
        self.expert_states.entry(expert).or_insert(MemoryTier::Disk)
    }

    /// Apply cache decision (for simulation/testing)
    pub fn apply_decision(
        &mut self,
        expert: AbstractExpert,
        decision: CacheDecision,
    ) -> Result<(), WatermarkError> {
        let expert_tier = self
            .expert_states
            .get_mut(&expert)
            .ok_or_else(|| WatermarkError::ExpertNotFound(format!("{:?}", expert)))?;

        let expert_size = self.config.expert_size_bytes;

        match decision {
            CacheDecision::PromoteToVRAM => {
                if *expert_tier == MemoryTier::Ram {
                    self.vram_used_bytes += expert_size;
                    *expert_tier = MemoryTier::Vram;
                }
            }
            CacheDecision::DemoteToRAM => {
                if *expert_tier == MemoryTier::Vram {
                    self.vram_used_bytes -= expert_size;
                    *expert_tier = MemoryTier::Ram;
                }
            }
            CacheDecision::LoadToRAM => {
                if *expert_tier == MemoryTier::Disk {
                    self.ram_used_bytes += expert_size;
                    *expert_tier = MemoryTier::Ram;
                }
            }
            CacheDecision::EvictToDisk => {
                if *expert_tier == MemoryTier::Ram {
                    self.ram_used_bytes -= expert_size;
                    *expert_tier = MemoryTier::Disk;
                }
            }
            CacheDecision::KeepInVRAM | CacheDecision::KeepInRAM => {
                // No state change needed
            }
        }

        Ok(())
    }

    /// Get current watermark values
    pub fn get_watermarks(&self) -> (f64, f64) {
        (self.vram_watermark, self.ram_watermark)
    }

    /// Get current memory usage
    pub fn get_memory_usage(&self) -> (usize, usize) {
        (self.vram_used_bytes, self.ram_used_bytes)
    }

    /// Get algorithm configuration
    pub fn config(&self) -> &WatermarkConfig {
        &self.config
    }

    /// Get current time
    pub fn current_time(&self) -> u64 {
        self.current_time
    }

    /// Reset algorithm state
    pub fn reset(&mut self) {
        self.vram_watermark = 0.0;
        self.ram_watermark = 0.0;
        self.current_time = 0;
        self.expert_states.clear();
        self.vram_used_bytes = 0;
        self.ram_used_bytes = 0;
    }

    /// Get reference to expert tiers for status reporting
    pub fn expert_states(&self) -> &HashMap<AbstractExpert, MemoryTier> {
        &self.expert_states
    }
}
