//! Configuration for watermark algorithm
//!
//! This module provides configuration structures for the dual watermark
//! algorithm that manages two-tier expert caching based on benefit density.

use crate::constants::ModelType;

/// Configuration for dual watermark algorithm
#[derive(Debug, Clone, PartialEq)]
pub struct WatermarkConfig {
    /// Model type for determining expert layout
    pub model_type: ModelType,
    
    /// VRAM capacity in bytes (K_G)
    pub vram_capacity: usize,

    /// RAM capacity in bytes (K_R)  
    pub ram_capacity: usize,

    /// VRAM watermark learning rate (η_G)
    pub vram_learning_rate: f64,

    /// RAM watermark learning rate (η_R)
    pub ram_learning_rate: f64,

    /// Cost of promoting expert from RAM to VRAM (C^G)
    pub cost_g: f64,

    /// Cost of loading expert from NVMe to RAM (C^R)  
    pub cost_r: f64,

    /// Expert size in bytes (all experts assumed to have same size)
    pub expert_size: usize,
}

impl WatermarkConfig {
    /// Create a new watermark configuration
    pub fn new(
        model_type: ModelType,
        vram_capacity: usize,
        ram_capacity: usize,
        vram_learning_rate: f64,
        ram_learning_rate: f64,
    ) -> Self {
        let config = Self {
            model_type,
            vram_capacity,
            ram_capacity,
            vram_learning_rate,
            ram_learning_rate,
            cost_g: 1.0, // Default cost values
            cost_r: 10.0,
            expert_size: 1024 * 1024, // 1MB default
        };
        config.validate().expect("Invalid watermark configuration");
        config
    }

    /// Create configuration for GPT-OSS-20B model
    pub fn for_gptoss20b(vram_capacity_mb: usize, ram_capacity_mb: usize) -> Self {
        Self::new(
            ModelType::GptOss20B,
            vram_capacity_mb * 1024 * 1024,
            ram_capacity_mb * 1024 * 1024,
            0.01, // Default learning rates
            0.01,
        )
    }

    /// Create configuration for GPT-OSS-120B model
    pub fn for_gptoss120b(vram_capacity_mb: usize, ram_capacity_mb: usize) -> Self {
        Self::new(
            ModelType::GptOss120B,
            vram_capacity_mb * 1024 * 1024,
            ram_capacity_mb * 1024 * 1024,
            0.005,
            0.005,
        )
    }

    /// Create configuration from model type with default memory capacities
    pub fn from_model(
        model_type: ModelType,
        vram_capacity_mb: usize,
        ram_capacity_mb: usize,
    ) -> Self {
        match model_type {
            ModelType::GptOss20B => Self::for_gptoss20b(vram_capacity_mb, ram_capacity_mb),
            ModelType::GptOss120B => Self::for_gptoss120b(vram_capacity_mb, ram_capacity_mb),
            ModelType::PhiTinyMoe => {
                // Use GPT-OSS-20B settings for now
                Self::for_gptoss20b(vram_capacity_mb, ram_capacity_mb)
            }
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), WatermarkConfigError> {
        if self.vram_capacity == 0 {
            return Err(WatermarkConfigError::InvalidCapacity(
                "VRAM capacity cannot be 0".to_string(),
            ));
        }

        if self.ram_capacity == 0 {
            return Err(WatermarkConfigError::InvalidCapacity(
                "RAM capacity cannot be 0".to_string(),
            ));
        }

        if self.vram_capacity > self.ram_capacity {
            return Err(WatermarkConfigError::InvalidCapacity(
                "VRAM capacity cannot exceed RAM capacity".to_string(),
            ));
        }

        if self.vram_learning_rate <= 0.0 {
            return Err(WatermarkConfigError::InvalidLearningRate(
                "VRAM learning rate must be > 0".to_string(),
            ));
        }

        if self.ram_learning_rate <= 0.0 {
            return Err(WatermarkConfigError::InvalidLearningRate(
                "RAM learning rate must be > 0".to_string(),
            ));
        }

        if self.cost_g <= 0.0 {
            return Err(WatermarkConfigError::InvalidCost(
                "RAM to VRAM cost must be > 0".to_string(),
            ));
        }

        if self.cost_r <= 0.0 {
            return Err(WatermarkConfigError::InvalidCost(
                "NVMe to RAM cost must be > 0".to_string(),
            ));
        }

        if self.expert_size == 0 {
            return Err(WatermarkConfigError::InvalidSize(
                "Expert size cannot be 0".to_string(),
            ));
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
            WatermarkConfigError::InvalidLearningRate(msg) => {
                write!(f, "Invalid learning rate: {}", msg)
            }
            WatermarkConfigError::InvalidCost(msg) => write!(f, "Invalid cost: {}", msg),
            WatermarkConfigError::InvalidSize(msg) => write!(f, "Invalid size: {}", msg),
        }
    }
}

impl std::error::Error for WatermarkConfigError {}
