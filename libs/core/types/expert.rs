//! Expert-related types for Python interoperability
//!
//! This module defines expert parameter types, keys, and references used
//! in the two-level cache system.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Parameter type enumeration for PyO3
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RustExpertParamType {
    value: u8,
}

impl RustExpertParamType {
    pub const MLP1_WEIGHT: RustExpertParamType = RustExpertParamType { value: 0 };
    pub const MLP1_BIAS: RustExpertParamType = RustExpertParamType { value: 1 };
    pub const MLP2_WEIGHT: RustExpertParamType = RustExpertParamType { value: 2 };
    pub const MLP2_BIAS: RustExpertParamType = RustExpertParamType { value: 3 };

    /// All parameter types for each expert (all 4 MLP parameters)
    /// Used for iterating over all parameter types when generating expert status
    pub const ALL_PARAM_TYPES: [RustExpertParamType; 4] = [
        Self::MLP1_WEIGHT,
        Self::MLP1_BIAS,
        Self::MLP2_WEIGHT,
        Self::MLP2_BIAS,
    ];
}

#[pymethods]
impl RustExpertParamType {
    #[classattr]
    pub const PY_MLP1_WEIGHT: Self = Self::MLP1_WEIGHT;

    #[classattr]
    pub const PY_MLP1_BIAS: Self = Self::MLP1_BIAS;

    #[classattr]
    pub const PY_MLP2_WEIGHT: Self = Self::MLP2_WEIGHT;

    #[classattr]
    pub const PY_MLP2_BIAS: Self = Self::MLP2_BIAS;

    #[new]
    fn new(value: &str) -> PyResult<Self> {
        match value {
            "mlp1_weight" => Ok(Self::MLP1_WEIGHT),
            "mlp1_bias" => Ok(Self::MLP1_BIAS),
            "mlp2_weight" => Ok(Self::MLP2_WEIGHT),
            "mlp2_bias" => Ok(Self::MLP2_BIAS),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid ExpertParamType value: {}",
                value
            ))),
        }
    }

    fn __str__(&self) -> &'static str {
        match self.value {
            0 => "mlp1_weight",
            1 => "mlp1_bias",
            2 => "mlp2_weight",
            3 => "mlp2_bias",
            _ => "unknown",
        }
    }

    fn __repr__(&self) -> String {
        format!("RustExpertParamType('{}')", self.__str__())
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
pub struct RustExpertKey {
    #[pyo3(get, set)]
    pub layer_idx: usize,
    #[pyo3(get, set)]
    pub expert_id: usize,
    #[pyo3(get, set)]
    pub param_type: RustExpertParamType,
}

#[pymethods]
impl RustExpertKey {
    #[new]
    pub fn new(layer_idx: usize, expert_id: usize, param_type: RustExpertParamType) -> Self {
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
