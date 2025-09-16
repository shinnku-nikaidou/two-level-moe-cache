//! Model type definitions and Python interoperability
//!
//! This module defines the ModelType enumeration and its Python conversions
//! for different MoE model variants.

use pyo3::prelude::*;
use pyo3::{Bound, PyAny};

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
