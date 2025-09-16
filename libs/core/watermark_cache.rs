//! Two-tier watermark-based expert cache manager for Python integration
//!
//! This module implements the pure two-tier watermark algorithm as a Python class
//! that receives fused predictions from external policy components.

use pyo3::prelude::*;
use std::collections::HashMap;

use crate::python_types::*;

/// Two-tier watermark-based expert cache manager
///
/// This implements the pure dual watermark algorithm from the paper, focusing
/// solely on benefit density calculations and watermark-based cache decisions.
/// Prediction fusion is handled by external policy components.
#[pyclass]
pub struct TwoTireWmExpertCacheManager {
    /// Configuration parameters
    config: WatermarkConfig,

    /// Model type for this cache instance
    model_type: ModelType,

    /// Current watermark values (λ_G, λ_R)
    vram_watermark: f64,
    ram_watermark: f64,

    /// Current time step for algorithms
    current_time: u64,

    /// Current layer being executed (0-based)
    current_layer: usize,

    /// Total number of layers in the model
    total_layers: usize,

    /// Expert residency tracking
    expert_states: HashMap<ExpertKey, ExpertResidencyState>,

    /// Capacity usage tracking
    vram_used_bytes: usize,
    ram_used_bytes: usize,

    /// Fused predictions from external policy components (expert_key -> probability)
    fused_predictions: HashMap<ExpertKey, f64>,

    /// Expert access history for statistics
    access_history: Vec<Vec<ExpertKey>>, // Per time step
}

/// Internal expert state tracking
#[derive(Debug, Clone)]
struct ExpertResidencyState {
    expert_key: ExpertKey,
    current_tier: Option<MemoryTier>,
    size_bytes: usize,
    last_access_time: u64,
    access_count: u64,

    // Cost model (would be loaded from configuration)
    ram_to_vram_cost: f64, // C_{e,ℓ}^G
    nvme_to_ram_cost: f64, // C_{e,ℓ}^R
}

impl ExpertResidencyState {
    fn new(expert_key: ExpertKey, size_bytes: usize) -> Self {
        Self {
            expert_key,
            current_tier: None,
            size_bytes,
            last_access_time: 0,
            access_count: 0,
            // Default cost values (should be configured)
            ram_to_vram_cost: 1.0,
            nvme_to_ram_cost: 10.0,
        }
    }

    fn is_resident_in(&self, tier: MemoryTier) -> bool {
        match (self.current_tier, tier) {
            (Some(current), target) => {
                match target {
                    MemoryTier::VRAM => current == MemoryTier::VRAM,
                    MemoryTier::RAM => current == MemoryTier::RAM || current == MemoryTier::VRAM,
                    MemoryTier::DISK => true, // Always available on disk
                }
            }
            (None, MemoryTier::DISK) => true,
            (None, _) => false,
        }
    }

    fn miss_cost(&self) -> f64 {
        match self.current_tier {
            Some(MemoryTier::VRAM) => 0.0,
            Some(MemoryTier::RAM) => self.ram_to_vram_cost,
            Some(MemoryTier::DISK) | None => self.nvme_to_ram_cost + self.ram_to_vram_cost,
        }
    }

    fn record_access(&mut self, time: u64) {
        self.last_access_time = time;
        self.access_count += 1;
    }
}

#[pymethods]
impl TwoTireWmExpertCacheManager {
    #[new]
    pub fn new(
        model_type: ModelType,
        config: WatermarkConfig,
        total_layers: usize,
    ) -> PyResult<Self> {
        config.validate()?;

        Ok(Self {
            config,
            model_type,
            vram_watermark: 0.0,
            ram_watermark: 0.0,
            current_time: 0,
            current_layer: 0,
            total_layers,
            expert_states: HashMap::new(),
            vram_used_bytes: 0,
            ram_used_bytes: 0,
            fused_predictions: HashMap::new(),
            access_history: Vec::new(),
        })
    }

    /// Get a single expert by key (main interface method)
    pub fn get(&mut self, expert_key: ExpertKey) -> PyResult<ExpertRef> {
        // Record the access
        self._record_expert_access(&expert_key);

        // Ensure expert is available
        self._ensure_expert_available(&expert_key)?;

        // Return expert reference
        let state = self.expert_states.get(&expert_key).unwrap();
        let mut expert_ref = ExpertRef::new(expert_key.clone());
        expert_ref.set_tier(state.current_tier);
        expert_ref.set_size(state.size_bytes);

        Ok(expert_ref)
    }

    /// Get multiple experts in batch (main interface method)
    pub fn get_batch(&mut self, expert_keys: Vec<ExpertKey>) -> PyResult<Vec<ExpertRef>> {
        let mut results = Vec::new();

        // Record accesses for all experts
        for key in &expert_keys {
            self._record_expert_access(key);
        }

        // Ensure all experts are available
        for key in &expert_keys {
            self._ensure_expert_available(key)?;
        }

        // Build result
        for key in expert_keys {
            let state = self.expert_states.get(&key).unwrap();
            let mut expert_ref = ExpertRef::new(key);
            expert_ref.set_tier(state.current_tier);
            expert_ref.set_size(state.size_bytes);
            results.push(expert_ref);
        }

        Ok(results)
    }

    /// Clear all cache state
    pub fn clear(&mut self) {
        self.expert_states.clear();
        self.vram_used_bytes = 0;
        self.ram_used_bytes = 0;
        self.fused_predictions.clear();
        self.access_history.clear();
        self.vram_watermark = 0.0;
        self.ram_watermark = 0.0;
        self.current_time = 0;
        self.current_layer = 0;
    }

    /// Advance to next time step (main interface method)
    pub fn next(&mut self) {
        self.current_time += 1;
        self.current_layer = (self.current_layer + 1) % self.total_layers;

        // Update watermarks using subgradient method
        self._update_watermarks();

        // Apply cache decisions based on current fused predictions
        self._apply_cache_decisions();

        // Record access pattern for this time step
        self.access_history.push(Vec::new());
    }

    /// Update fused predictions from external policy components
    pub fn update_fused_predictions(&mut self, predictions: HashMap<String, f64>) {
        self.fused_predictions.clear();
        for (key_str, prob) in predictions {
            if let Ok(key) = self._parse_expert_key(&key_str) {
                self.fused_predictions.insert(key, prob);
            }
        }
    }

    /// Get current watermark values for debugging
    pub fn get_watermarks(&self) -> (f64, f64) {
        (self.vram_watermark, self.ram_watermark)
    }

    /// Get current capacity usage
    pub fn get_capacity_usage(&self) -> ((usize, usize), (usize, usize)) {
        (
            (self.vram_used_bytes, self.config.vram_capacity),
            (self.ram_used_bytes, self.config.ram_capacity),
        )
    }

    /// Get statistics for analysis
    pub fn get_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        stats.insert("vram_watermark".to_string(), self.vram_watermark);
        stats.insert("ram_watermark".to_string(), self.ram_watermark);
        stats.insert(
            "vram_utilization".to_string(),
            self.vram_used_bytes as f64 / self.config.vram_capacity as f64,
        );
        stats.insert(
            "ram_utilization".to_string(),
            self.ram_used_bytes as f64 / self.config.ram_capacity as f64,
        );
        stats.insert("total_experts".to_string(), self.expert_states.len() as f64);
        stats
    }
}

// Private implementation methods
impl TwoTireWmExpertCacheManager {
    fn _record_expert_access(&mut self, expert_key: &ExpertKey) {
        // Initialize expert state if not exists
        if !self.expert_states.contains_key(expert_key) {
            let state = ExpertResidencyState::new(expert_key.clone(), 1024); // Default size
            self.expert_states.insert(expert_key.clone(), state);
        }

        // Record access
        if let Some(state) = self.expert_states.get_mut(expert_key) {
            state.record_access(self.current_time);
        }

        // Add to current time step history
        if let Some(current_accesses) = self.access_history.last_mut() {
            current_accesses.push(expert_key.clone());
        }
    }

    fn _ensure_expert_available(&mut self, expert_key: &ExpertKey) -> PyResult<()> {
        let state = self.expert_states.get(expert_key).unwrap();

        // If expert is already in VRAM, nothing to do
        if state.is_resident_in(MemoryTier::VRAM) {
            return Ok(());
        }

        // Try to promote to VRAM if needed
        self._try_promote_to_vram(expert_key)?;

        Ok(())
    }

    fn _try_promote_to_vram(&mut self, expert_key: &ExpertKey) -> PyResult<()> {
        let state = self.expert_states.get(expert_key).unwrap();

        // Check if we have capacity
        if self.vram_used_bytes + state.size_bytes <= self.config.vram_capacity {
            // Direct promotion
            self._promote_expert_to_vram(expert_key);
        } else {
            // Need to evict something first
            self._make_vram_space(state.size_bytes)?;
            self._promote_expert_to_vram(expert_key);
        }

        Ok(())
    }

    fn _promote_expert_to_vram(&mut self, expert_key: &ExpertKey) {
        if let Some(state) = self.expert_states.get_mut(expert_key) {
            // Update tier tracking
            if state.current_tier == Some(MemoryTier::RAM) {
                // No change in RAM usage, just promote to VRAM
                self.vram_used_bytes += state.size_bytes;
            } else {
                // Load from disk to both RAM and VRAM
                self.ram_used_bytes += state.size_bytes;
                self.vram_used_bytes += state.size_bytes;
            }

            state.current_tier = Some(MemoryTier::VRAM);
        }
    }

    fn _make_vram_space(&mut self, needed_bytes: usize) -> PyResult<()> {
        // Find experts to evict based on watermark algorithm
        let mut candidates: Vec<_> = self
            .expert_states
            .iter()
            .filter(|(_, state)| state.is_resident_in(MemoryTier::VRAM))
            .map(|(key, state)| (key.clone(), state.size_bytes))
            .collect();

        // Sort by benefit density (ascending - evict lowest first)
        candidates.sort_by(|a, b| {
            let density_a = self._get_vram_benefit_density(&a.0);
            let density_b = self._get_vram_benefit_density(&b.0);
            density_a
                .partial_cmp(&density_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut freed_bytes = 0;
        for (key, size_bytes) in candidates {
            if freed_bytes >= needed_bytes {
                break;
            }

            // Check if we should evict based on watermark
            let density = self._get_vram_benefit_density(&key);
            if density < self.vram_watermark {
                self._demote_expert_from_vram(&key);
                freed_bytes += size_bytes;
            }
        }

        if freed_bytes < needed_bytes {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Could not free enough VRAM space",
            ));
        }

        Ok(())
    }

    fn _demote_expert_from_vram(&mut self, expert_key: &ExpertKey) {
        if let Some(state) = self.expert_states.get_mut(expert_key) {
            if state.current_tier == Some(MemoryTier::VRAM) {
                self.vram_used_bytes -= state.size_bytes;
                state.current_tier = Some(MemoryTier::RAM);
            }
        }
    }

    fn _update_watermarks(&mut self) {
        // Subgradient update: λ_G ← [λ_G + η_G(usage - capacity)]_+
        let vram_constraint = self.vram_used_bytes as f64 - self.config.vram_capacity as f64;
        self.vram_watermark =
            (self.vram_watermark + self.config.vram_learning_rate * vram_constraint).max(0.0);

        let ram_constraint = self.ram_used_bytes as f64 - self.config.ram_capacity as f64;
        self.ram_watermark =
            (self.ram_watermark + self.config.ram_learning_rate * ram_constraint).max(0.0);
    }

    fn _apply_cache_decisions(&mut self) {
        // Apply watermark-based decisions to all experts
        let keys: Vec<_> = self.expert_states.keys().cloned().collect();

        for key in keys {
            self._apply_expert_decision(&key);
        }
    }

    fn _apply_expert_decision(&mut self, expert_key: &ExpertKey) {
        let (vram_density, ram_density) = self._compute_benefit_densities(expert_key);
        let state = self.expert_states.get(expert_key).unwrap();

        match state.current_tier {
            Some(MemoryTier::VRAM) => {
                // Check if should demote from VRAM
                if vram_density < self.vram_watermark {
                    self._demote_expert_from_vram(expert_key);
                }
            }
            Some(MemoryTier::RAM) => {
                // Check if should evict from RAM
                if ram_density < self.ram_watermark {
                    self._evict_expert_from_ram(expert_key);
                }
            }
            _ => {} // On disk, no action needed
        }
    }

    fn _compute_benefit_densities(&self, expert_key: &ExpertKey) -> (f64, f64) {
        let fused_prob = self.fused_predictions.get(expert_key).copied().unwrap_or(0.0);
        let state = self.expert_states.get(expert_key).unwrap();

        // b^G = p^fuse * C^G / S
        let vram_density = fused_prob * state.ram_to_vram_cost / state.size_bytes as f64;
        // b^R = p^fuse * C^R / S
        let ram_density = fused_prob * state.nvme_to_ram_cost / state.size_bytes as f64;

        (vram_density, ram_density)
    }

    fn _get_vram_benefit_density(&self, expert_key: &ExpertKey) -> f64 {
        let (vram_density, _) = self._compute_benefit_densities(expert_key);
        vram_density
    }

    fn _evict_expert_from_ram(&mut self, expert_key: &ExpertKey) {
        if let Some(state) = self.expert_states.get_mut(expert_key) {
            if state.current_tier == Some(MemoryTier::RAM) {
                self.ram_used_bytes -= state.size_bytes;
                state.current_tier = Some(MemoryTier::DISK);
            }
        }
    }

    fn _parse_expert_key(&self, _key_str: &str) -> PyResult<ExpertKey> {
        // Placeholder parser - would need proper implementation
        // Expected format: "L0_E1_mlp1_weight"
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Expert key parsing not yet implemented",
        ))
    }
}
