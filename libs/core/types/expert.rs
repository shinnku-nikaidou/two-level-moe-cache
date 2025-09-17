//! Expert-related types for Python interoperability
//!
//! This module defines expert parameter types, keys, and references used
//! in the two-level cache system.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

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
