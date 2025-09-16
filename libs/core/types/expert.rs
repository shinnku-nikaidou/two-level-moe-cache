//! Expert-related types for Python interoperability
//!
//! This module defines expert parameter types, keys, and references used
//! in the two-level cache system.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use super::memory::MemoryTier;

/// Expert parameter type matching Python's ExpertParamType
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExpertParamType {
    value: ExpertParamTypeEnum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum ExpertParamTypeEnum {
    MLP1Weight,
    MLP1Bias,
    MLP2Weight,
    MLP2Bias,
}

#[pymethods]
impl ExpertParamType {
    #[classattr]
    pub const MLP1_WEIGHT: Self = Self {
        value: ExpertParamTypeEnum::MLP1Weight,
    };

    #[classattr]
    pub const MLP1_BIAS: Self = Self {
        value: ExpertParamTypeEnum::MLP1Bias,
    };

    #[classattr]
    pub const MLP2_WEIGHT: Self = Self {
        value: ExpertParamTypeEnum::MLP2Weight,
    };

    #[classattr]
    pub const MLP2_BIAS: Self = Self {
        value: ExpertParamTypeEnum::MLP2Bias,
    };

    #[new]
    fn new(value: &str) -> PyResult<Self> {
        match value {
            "mlp1_weight" => Ok(Self {
                value: ExpertParamTypeEnum::MLP1Weight,
            }),
            "mlp1_bias" => Ok(Self {
                value: ExpertParamTypeEnum::MLP1Bias,
            }),
            "mlp2_weight" => Ok(Self {
                value: ExpertParamTypeEnum::MLP2Weight,
            }),
            "mlp2_bias" => Ok(Self {
                value: ExpertParamTypeEnum::MLP2Bias,
            }),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid ExpertParamType value: {}",
                value
            ))),
        }
    }

    fn __str__(&self) -> &'static str {
        match self.value {
            ExpertParamTypeEnum::MLP1Weight => "mlp1_weight",
            ExpertParamTypeEnum::MLP1Bias => "mlp1_bias",
            ExpertParamTypeEnum::MLP2Weight => "mlp2_weight",
            ExpertParamTypeEnum::MLP2Bias => "mlp2_bias",
        }
    }

    fn __repr__(&self) -> String {
        format!("ExpertParamType('{}')", self.__str__())
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
}

/// Expert key matching Python's ExpertKey dataclass
#[pyclass]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExpertKey {
    #[pyo3(get, set)]
    pub layer_idx: usize,
    #[pyo3(get, set)]
    pub expert_id: usize,
    #[pyo3(get, set)]
    pub param_type: ExpertParamType,
}

#[pymethods]
impl ExpertKey {
    #[new]
    pub fn new(layer_idx: usize, expert_id: usize, param_type: ExpertParamType) -> Self {
        Self {
            layer_idx,
            expert_id,
            param_type,
        }
    }

    pub fn __str__(&self) -> String {
        format!(
            "L{}_E{}_{:?}",
            self.layer_idx, self.expert_id, self.param_type
        )
    }

    pub fn __repr__(&self) -> String {
        self.__str__()
    }

    pub fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    pub fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
}

/// Expert wrapper for Python interop (simplified version)
/// The actual Expert object management happens in Python side
#[pyclass]
#[derive(Debug, Clone)]
pub struct ExpertRef {
    #[pyo3(get)]
    pub expert_key: ExpertKey,
    #[pyo3(get)]
    pub current_tier: Option<MemoryTier>,
    #[pyo3(get)]
    pub is_loaded: bool,
    #[pyo3(get)]
    pub size_bytes: Option<usize>,
}

#[pymethods]
impl ExpertRef {
    #[new]
    pub fn new(expert_key: ExpertKey) -> Self {
        Self {
            expert_key,
            current_tier: None,
            is_loaded: false,
            size_bytes: None,
        }
    }

    pub fn set_tier(&mut self, tier: Option<MemoryTier>) {
        self.current_tier = tier;
        self.is_loaded = tier.is_some();
    }

    pub fn set_size(&mut self, size_bytes: usize) {
        self.size_bytes = Some(size_bytes);
    }

    pub fn is_in_vram(&self) -> bool {
        matches!(self.current_tier, Some(MemoryTier::VRAM))
    }

    pub fn is_in_ram(&self) -> bool {
        matches!(self.current_tier, Some(MemoryTier::RAM))
    }

    pub fn is_on_disk(&self) -> bool {
        matches!(self.current_tier, Some(MemoryTier::DISK)) || self.current_tier.is_none()
    }
}
