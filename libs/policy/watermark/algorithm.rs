//! Watermark algorithm implementation
//!
//! This module implements the dual watermark algorithm from the documentation:
//!
//! 1. Benefit density calculation: b^G = p^fuse * C^G / S, b^R = p^fuse * C^R / S  
//! 2. Watermark updates: λ_G ← [λ_G + η_G(usage - K_G)]_+, λ_R ← [λ_R + η_R(usage - K_R)]_+
//! 3. Cache decisions: Keep in tier iff b >= λ

use std::collections::HashMap;

use crate::ExpertKey;
use super::config::WatermarkConfig;
use super::error::WatermarkError;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTier {
    VRAM,
    RAM, 
    Disk,
}

/// Expert state for watermark tracking
#[derive(Debug, Clone)]
pub struct ExpertState {
    pub expert_key: ExpertKey,
    pub current_tier: MemoryTier,
    pub size_bytes: usize,
    pub last_access_time: u64,
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
    
    /// Expert state tracking
    expert_states: HashMap<ExpertKey, ExpertState>,
    
    /// Current memory usage
    vram_used_bytes: usize,
    ram_used_bytes: usize,
}

impl WatermarkAlgorithm {
    /// Create a new watermark algorithm instance
    pub fn new(config: WatermarkConfig) -> Result<Self, WatermarkError> {
        config.validate()?;
        
        Ok(Self {
            config,
            vram_watermark: 0.0,
            ram_watermark: 0.0,
            current_time: 0,
            expert_states: HashMap::new(),
            vram_used_bytes: 0,
            ram_used_bytes: 0,
        })
    }

    /// Create watermark algorithm for GPT-OSS-20B model
    pub fn for_gptoss20b(vram_capacity_mb: usize, ram_capacity_mb: usize) -> Result<Self, WatermarkError> {
        let config = WatermarkConfig::for_gptoss20b(vram_capacity_mb, ram_capacity_mb)?;
        Self::new(config)
    }

    /// Create watermark algorithm for testing
    pub fn for_testing() -> Self {
        let config = WatermarkConfig::for_testing();
        Self::new(config).unwrap()
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
        fused_probabilities: &HashMap<ExpertKey, f64>,
    ) -> Result<HashMap<ExpertKey, CacheDecision>, WatermarkError> {
        let mut decisions = HashMap::new();

        for (expert_key, &probability) in fused_probabilities {
            // Validate probability
            if !(0.0..=1.0).contains(&probability) {
                return Err(WatermarkError::InvalidProbability {
                    expert_key: format!("{:?}", expert_key),
                    probability,
                });
            }

            // Get or create expert state 
            let current_tier = {
                let expert_state = self.get_or_create_expert_state(*expert_key);
                expert_state.current_tier
            };

            // Calculate benefit densities using configuration
            let size = self.config.expert_size_bytes as f64;
            let vram_benefit = probability * self.config.ram_to_vram_cost / size;
            let ram_benefit = probability * self.config.nvme_to_ram_cost / size;

            // Make cache decision based on current tier and watermarks
            let decision = match current_tier {
                MemoryTier::VRAM => {
                    if vram_benefit >= self.vram_watermark {
                        CacheDecision::KeepInVRAM
                    } else {
                        CacheDecision::DemoteToRAM
                    }
                }
                MemoryTier::RAM => {
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

            decisions.insert(*expert_key, decision);
        }

        Ok(decisions)
    }

    /// Update watermarks using subgradient method
    ///
    /// Implements: λ_G ← [λ_G + η_G(usage - K_G)]_+, λ_R ← [λ_R + η_R(usage - K_R)]_+
    fn update_watermarks(&mut self) {
        // VRAM watermark update
        let vram_constraint = self.vram_used_bytes as f64 - self.config.vram_capacity as f64;
        self.vram_watermark = (self.vram_watermark + self.config.vram_learning_rate * vram_constraint).max(0.0);

        // RAM watermark update  
        let ram_constraint = self.ram_used_bytes as f64 - self.config.ram_capacity as f64;
        self.ram_watermark = (self.ram_watermark + self.config.ram_learning_rate * ram_constraint).max(0.0);
    }

    /// Calculate benefit densities for an expert
    ///
    /// Implements: b^G = p^fuse * C^G / S, b^R = p^fuse * C^R / S
    fn calculate_benefit_densities(&self, probability: f64, expert_state: &ExpertState) -> (f64, f64) {
        let size = expert_state.size_bytes as f64;
        
        let vram_benefit = probability * self.config.ram_to_vram_cost / size;
        let ram_benefit = probability * self.config.nvme_to_ram_cost / size;
        
        (vram_benefit, ram_benefit)
    }

    /// Get or create expert state for tracking
    fn get_or_create_expert_state(&mut self, expert_key: ExpertKey) -> &mut ExpertState {
        self.expert_states.entry(expert_key).or_insert_with(|| {
            ExpertState {
                expert_key,
                current_tier: MemoryTier::Disk,
                size_bytes: self.config.expert_size_bytes,
                last_access_time: self.current_time,
            }
        })
    }

    /// Apply cache decision (for simulation/testing)
    pub fn apply_decision(&mut self, expert_key: ExpertKey, decision: CacheDecision) -> Result<(), WatermarkError> {
        let expert_state = self.expert_states.get_mut(&expert_key)
            .ok_or_else(|| WatermarkError::ExpertNotFound(format!("{:?}", expert_key)))?;

        match decision {
            CacheDecision::PromoteToVRAM => {
                if expert_state.current_tier == MemoryTier::RAM {
                    self.vram_used_bytes += expert_state.size_bytes;
                    expert_state.current_tier = MemoryTier::VRAM;
                }
            }
            CacheDecision::DemoteToRAM => {
                if expert_state.current_tier == MemoryTier::VRAM {
                    self.vram_used_bytes -= expert_state.size_bytes;
                    expert_state.current_tier = MemoryTier::RAM;
                }
            }
            CacheDecision::LoadToRAM => {
                if expert_state.current_tier == MemoryTier::Disk {
                    self.ram_used_bytes += expert_state.size_bytes;
                    expert_state.current_tier = MemoryTier::RAM;
                }
            }
            CacheDecision::EvictToDisk => {
                if expert_state.current_tier == MemoryTier::RAM {
                    self.ram_used_bytes -= expert_state.size_bytes;
                    expert_state.current_tier = MemoryTier::Disk;
                }
            }
            CacheDecision::KeepInVRAM | CacheDecision::KeepInRAM => {
                // No state change needed
            }
        }

        expert_state.last_access_time = self.current_time;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watermark_creation() {
        let algorithm = WatermarkAlgorithm::for_testing();
        assert_eq!(algorithm.get_watermarks(), (0.0, 0.0));
        assert_eq!(algorithm.get_memory_usage(), (0, 0));
    }

    #[test]
    fn test_benefit_density_calculation() {
        let algorithm = WatermarkAlgorithm::for_testing();
        let expert_state = ExpertState {
            expert_key: ExpertKey::new(0, 0),
            current_tier: MemoryTier::Disk,
            size_bytes: 1024 * 1024, // 1MB
            last_access_time: 0,
        };

        let (vram_benefit, ram_benefit) = algorithm.calculate_benefit_densities(0.5, &expert_state);
        
        // Expected: vram_benefit = 0.5 * 1.0 / (1024*1024), ram_benefit = 0.5 * 10.0 / (1024*1024)
        assert!((vram_benefit - 0.5 / (1024.0 * 1024.0)).abs() < 1e-10);
        assert!((ram_benefit - 5.0 / (1024.0 * 1024.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cache_decisions() {
        let mut algorithm = WatermarkAlgorithm::for_testing();
        
        let mut fused_probs = HashMap::new();
        let expert_key = ExpertKey::new(0, 0);
        fused_probs.insert(expert_key, 0.8);

        let decisions = algorithm.make_cache_decisions(&fused_probs).unwrap();
        
        // Expert starts on disk, should be loaded to RAM if benefit > watermark
        assert!(decisions.contains_key(&expert_key));
    }
}