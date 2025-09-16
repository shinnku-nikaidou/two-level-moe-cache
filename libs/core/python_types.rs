//! Python type conversions for PyO3 bindings
//!
//! This module provides conversions between Python types used in the domain layer
//! and Rust types used in the core implementation.

use pyo3::prelude::*;
use pyo3::{Bound, PyAny};
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

/// Configuration for watermark algorithm
#[pyclass]
#[derive(Debug, Clone)]
pub struct WatermarkConfig {
    #[pyo3(get, set)]
    pub vram_capacity: usize,
    #[pyo3(get, set)]
    pub ram_capacity: usize,
    #[pyo3(get, set)]
    pub vram_learning_rate: f64, // η_G
    #[pyo3(get, set)]
    pub ram_learning_rate: f64, // η_R
    #[pyo3(get, set)]
    pub fusion_eta: f64, // η for EWMA-ScoutGate fusion
    #[pyo3(get, set)]
    pub reuse_decay_gamma: f64, // γ for reuse distance decay
    #[pyo3(get, set)]
    pub hysteresis_factor: f64, // Factor for admit/evict threshold separation
}

#[pymethods]
impl WatermarkConfig {
    #[new]
    pub fn new(
        vram_capacity: usize,
        ram_capacity: usize,
        vram_learning_rate: Option<f64>,
        ram_learning_rate: Option<f64>,
        fusion_eta: Option<f64>,
        reuse_decay_gamma: Option<f64>,
        hysteresis_factor: Option<f64>,
    ) -> Self {
        Self {
            vram_capacity,
            ram_capacity,
            vram_learning_rate: vram_learning_rate.unwrap_or(0.01),
            ram_learning_rate: ram_learning_rate.unwrap_or(0.01),
            fusion_eta: fusion_eta.unwrap_or(0.5),
            reuse_decay_gamma: reuse_decay_gamma.unwrap_or(0.1),
            hysteresis_factor: hysteresis_factor.unwrap_or(1.1),
        }
    }

    pub fn validate(&self) -> PyResult<()> {
        if self.vram_capacity == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "vram_capacity must be > 0",
            ));
        }
        if self.ram_capacity == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "ram_capacity must be > 0",
            ));
        }
        if self.vram_learning_rate <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "vram_learning_rate must be > 0",
            ));
        }
        if self.ram_learning_rate <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "ram_learning_rate must be > 0",
            ));
        }
        if !(0.0..=1.0).contains(&self.fusion_eta) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "fusion_eta must be in [0, 1]",
            ));
        }
        if self.reuse_decay_gamma <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "reuse_decay_gamma must be > 0",
            ));
        }
        if self.hysteresis_factor < 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "hysteresis_factor must be >= 1.0",
            ));
        }
        Ok(())
    }
}

/// Python-compatible model type
#[derive(Debug, Clone)]
pub enum ModelType {
    GptOss20B,
    GptOss120B,
    PhiTinyMoe,
}

impl FromPyObject<'_> for ModelType {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let value: &str = ob.extract()?;
        match value {
            "gpt-oss-20b" => Ok(ModelType::GptOss20B),
            "gpt-oss-120b" => Ok(ModelType::GptOss120B),
            "phi-tiny-moe" => Ok(ModelType::PhiTinyMoe),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid ModelType: {}",
                value
            ))),
        }
    }
}

impl<'py> IntoPyObject<'py> for ModelType {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let value = match self {
            ModelType::GptOss20B => "gpt-oss-20b",
            ModelType::GptOss120B => "gpt-oss-120b",
            ModelType::PhiTinyMoe => "phi-tiny-moe",
        };
        Ok(value.into_pyobject(py).unwrap().into_any())
    }
}
