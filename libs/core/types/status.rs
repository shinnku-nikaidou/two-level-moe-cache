//! Expert status types for Python interface
//!
//! This module defines the ExpertStatus structure used to communicate
//! expert cache state from Rust to Python.

use super::expert::RustExpertKey;
use pyo3::prelude::*;

/// Simplified expert status information
///
/// Contains only the essential information needed by Python layer:
/// - expert_key: Unique identifier for the expert
/// - current_tier: Memory tier where expert currently resides (0=VRAM, 1=RAM, 2=DISK)
#[pyclass]
#[derive(Debug, Clone)]
pub struct RustExpertStatus {
    /// Expert unique identifier
    #[pyo3(get)]
    pub expert_key: RustExpertKey,

    /// Current memory tier as u8: VRAM=0, RAM=1, DISK=2
    #[pyo3(get)]
    pub current_tier: u8,
}

#[pymethods]
impl RustExpertStatus {
    /// Create a new expert status
    #[new]
    pub fn new(expert_key: RustExpertKey, current_tier: u8) -> Self {
        Self {
            expert_key,
            current_tier,
        }
    }

    /// String representation for debugging
    fn __repr__(&self) -> String {
        let tier_name = match self.current_tier {
            0 => "VRAM",
            1 => "RAM",
            2 => "DISK",
            _ => "UNKNOWN",
        };
        format!(
            "ExpertStatus(expert_key={:?}, current_tier={}({}))",
            self.expert_key, self.current_tier, tier_name
        )
    }
}
