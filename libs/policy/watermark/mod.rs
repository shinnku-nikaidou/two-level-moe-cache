//! Dual watermark algorithm module

//!
//! This module implements the dual watermark algorithm for two-tier expert caching.
//! It manages VRAM and RAM tiers using adaptive watermark thresholds based on
//! benefit density calculations.
//!
//! ## Algorithm Overview
//!
//! The watermark algorithm operates in the following steps:
//!
//! 1. **Benefit Density Calculation**: For each expert with fused probability p^{fuse}:
//!    - VRAM benefit: `b^G = p^{fuse} * C^G / S`
//!    - RAM benefit: `b^R = p^{fuse} * C^R / S`
//!      where C^G, C^R are tier access costs and S is expert size
//!
//! 2. **Watermark Updates**: Using subgradient method:
//!    - `λ_G ← [λ_G + η_G(vram_usage - K_G)]_+`
//!    - `λ_R ← [λ_R + η_R(ram_usage - K_R)]_+`
//!      where K_G, K_R are tier capacities and η_G, η_R are learning rates
//!
//! 3. **Cache Decisions**: For each expert:
//!    - Keep in tier if `benefit >= watermark`
//!    - Evict from tier if `benefit < watermark`

pub mod algorithm;
pub mod config;
pub mod error;

pub use algorithm::{CacheDecision, ExpertState, MemoryTier, WatermarkAlgorithm};
pub use config::{WatermarkConfig, WatermarkConfigError};
pub use error::WatermarkError;
