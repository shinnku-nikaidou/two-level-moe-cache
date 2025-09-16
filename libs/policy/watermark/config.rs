//! Configuration for watermark algorithm
//!
//! This module provides configuration structures for the dual watermark
//! algorithm that manages two-tier expert caching based on benefit density.

/// Configuration for dual watermark algorithm
#[derive(Debug, Clone, PartialEq)]
pub struct WatermarkConfig {
    /// VRAM capacity in bytes (K_G)
    pub vram_capacity: usize,
    
    /// RAM capacity in bytes (K_R)  
    pub ram_capacity: usize,
    
    /// VRAM watermark learning rate (η_G)
    pub vram_learning_rate: f64,
    
    /// RAM watermark learning rate (η_R)
    pub ram_learning_rate: f64,
    
    /// Cost of promoting expert from RAM to VRAM (C^G)
    pub ram_to_vram_cost: f64,
    
    /// Cost of loading expert from NVMe to RAM (C^R)  
    pub nvme_to_ram_cost: f64,
    
    /// Expert size in bytes (simplified - in practice would vary per expert)
    pub expert_size_bytes: usize,
}

impl WatermarkConfig {
    /// Create a new watermark configuration
    pub fn new(
        vram_capacity: usize,
        ram_capacity: usize,
        vram_learning_rate: f64,
        ram_learning_rate: f64,
    ) -> Result<Self, WatermarkConfigError> {
        let config = Self {
            vram_capacity,
            ram_capacity,
            vram_learning_rate,
            ram_learning_rate,
            ram_to_vram_cost: 1.0,     // Default cost values
            nvme_to_ram_cost: 10.0,
            expert_size_bytes: 1024 * 1024, // 1MB default
        };
        config.validate()?;
        Ok(config)
    }

    /// Create configuration for GPT-OSS-20B model
    pub fn for_gptoss20b(vram_capacity_mb: usize, ram_capacity_mb: usize) -> Result<Self, WatermarkConfigError> {
        Self::new(
            vram_capacity_mb * 1024 * 1024,
            ram_capacity_mb * 1024 * 1024,
            0.01,   // Default learning rates
            0.01,
        )
    }

    /// Create configuration for GPT-OSS-120B model
    pub fn for_gptoss120b(vram_capacity_mb: usize, ram_capacity_mb: usize) -> Result<Self, WatermarkConfigError> {
        Self::new(
            vram_capacity_mb * 1024 * 1024,
            ram_capacity_mb * 1024 * 1024,
            0.005,  // Smaller learning rates for larger model
            0.005,
        )
    }

    /// Create configuration for testing
    pub fn for_testing() -> Self {
        Self {
            vram_capacity: 100 * 1024 * 1024,  // 100MB
            ram_capacity: 500 * 1024 * 1024,   // 500MB
            vram_learning_rate: 0.1,           // Higher learning rate for faster tests
            ram_learning_rate: 0.1,
            ram_to_vram_cost: 1.0,
            nvme_to_ram_cost: 10.0,
            expert_size_bytes: 1024 * 1024,    // 1MB per expert
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), WatermarkConfigError> {
        if self.vram_capacity == 0 {
            return Err(WatermarkConfigError::InvalidCapacity("VRAM capacity cannot be 0".to_string()));
        }
        
        if self.ram_capacity == 0 {
            return Err(WatermarkConfigError::InvalidCapacity("RAM capacity cannot be 0".to_string()));
        }
        
        if self.vram_capacity > self.ram_capacity {
            return Err(WatermarkConfigError::InvalidCapacity("VRAM capacity cannot exceed RAM capacity".to_string()));
        }
        
        if self.vram_learning_rate <= 0.0 {
            return Err(WatermarkConfigError::InvalidLearningRate("VRAM learning rate must be > 0".to_string()));
        }
        
        if self.ram_learning_rate <= 0.0 {
            return Err(WatermarkConfigError::InvalidLearningRate("RAM learning rate must be > 0".to_string()));
        }
        
        if self.ram_to_vram_cost <= 0.0 {
            return Err(WatermarkConfigError::InvalidCost("RAM to VRAM cost must be > 0".to_string()));
        }
        
        if self.nvme_to_ram_cost <= 0.0 {
            return Err(WatermarkConfigError::InvalidCost("NVMe to RAM cost must be > 0".to_string()));
        }
        
        if self.expert_size_bytes == 0 {
            return Err(WatermarkConfigError::InvalidSize("Expert size cannot be 0".to_string()));
        }
        
        Ok(())
    }
}

/// Error types for watermark configuration
#[derive(Debug, Clone, PartialEq)]
pub enum WatermarkConfigError {
    /// Invalid capacity configuration
    InvalidCapacity(String),
    
    /// Invalid learning rate
    InvalidLearningRate(String),
    
    /// Invalid cost parameter
    InvalidCost(String),
    
    /// Invalid size parameter
    InvalidSize(String),
}

impl std::fmt::Display for WatermarkConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WatermarkConfigError::InvalidCapacity(msg) => write!(f, "Invalid capacity: {}", msg),
            WatermarkConfigError::InvalidLearningRate(msg) => write!(f, "Invalid learning rate: {}", msg),
            WatermarkConfigError::InvalidCost(msg) => write!(f, "Invalid cost: {}", msg),
            WatermarkConfigError::InvalidSize(msg) => write!(f, "Invalid size: {}", msg),
        }
    }
}

impl std::error::Error for WatermarkConfigError {}