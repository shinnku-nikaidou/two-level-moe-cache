//! Memory tier types for Python interoperability
//!
//! This module defines the MemoryTier enumeration that represents different
//! storage tiers in the two-level cache hierarchy.

use policy::watermark::algorithm::MemoryTier;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Memory tier enumeration matching Python's MemoryTier
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RustMemoryTier {
    value: MemoryTier,
}

#[pymethods]
impl RustMemoryTier {
    #[classattr]
    pub const VRAM: Self = Self {
        value: MemoryTier::Vram,
    };

    #[classattr]
    pub const RAM: Self = Self {
        value: MemoryTier::Ram,
    };

    #[classattr]
    pub const DISK: Self = Self {
        value: MemoryTier::Disk,
    };

    #[new]
    fn new(value: i32) -> PyResult<Self> {
        match value {
            0 => Ok(Self {
                value: MemoryTier::Vram,
            }),
            1 => Ok(Self {
                value: MemoryTier::Ram,
            }),
            2 => Ok(Self {
                value: MemoryTier::Disk,
            }),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid MemoryTier value: {}",
                value
            ))),
        }
    }

    fn __str__(&self) -> &'static str {
        match self.value {
            MemoryTier::Vram => "VRAM",
            MemoryTier::Ram => "RAM",
            MemoryTier::Disk => "DISK",
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
