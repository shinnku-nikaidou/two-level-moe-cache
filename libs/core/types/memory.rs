//! Memory tier types for Python interoperability
//!
//! This module defines the MemoryTier enumeration that represents different
//! storage tiers in the two-level cache hierarchy.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Memory tier enumeration matching Python's MemoryTier
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MemoryTier {
    value: MemoryTierEnum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum MemoryTierEnum {
    VRAM = 0, // GPU memory - fastest, most limited
    RAM = 1,  // System memory - fast, moderate capacity
    DISK = 2, // NVMe/SSD storage - slower, largest capacity
}

#[pymethods]
impl MemoryTier {
    #[classattr]
    pub const VRAM: Self = Self {
        value: MemoryTierEnum::VRAM,
    };

    #[classattr]
    pub const RAM: Self = Self {
        value: MemoryTierEnum::RAM,
    };

    #[classattr]
    pub const DISK: Self = Self {
        value: MemoryTierEnum::DISK,
    };

    #[new]
    fn new(value: i32) -> PyResult<Self> {
        match value {
            0 => Ok(Self {
                value: MemoryTierEnum::VRAM,
            }),
            1 => Ok(Self {
                value: MemoryTierEnum::RAM,
            }),
            2 => Ok(Self {
                value: MemoryTierEnum::DISK,
            }),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid MemoryTier value: {}",
                value
            ))),
        }
    }

    fn __str__(&self) -> &'static str {
        match self.value {
            MemoryTierEnum::VRAM => "VRAM",
            MemoryTierEnum::RAM => "RAM",
            MemoryTierEnum::DISK => "DISK",
        }
    }

    fn __repr__(&self) -> String {
        format!("MemoryTier.{}", self.__str__())
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.value == other.value
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.value.hash(&mut hasher);
        hasher.finish()
    }

    fn __int__(&self) -> i32 {
        self.value as i32
    }
}
