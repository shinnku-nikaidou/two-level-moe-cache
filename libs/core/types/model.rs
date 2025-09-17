//! Model type definitions and Python interoperability
//!
//! This module defines the ModelType enumeration and its Python conversions
//! for different MoE model variants.

use pyo3::prelude::*;
use pyo3::{Bound, PyAny};

/// Python-compatible model type
#[derive(Debug, Clone)]
pub enum RustModelType {
    GptOss20B,
    GptOss120B,
    PhiTinyMoe,
}

impl FromPyObject<'_> for RustModelType {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let value: &str = ob.extract()?;
        match value {
            "gpt-oss-20b" => Ok(RustModelType::GptOss20B),
            "gpt-oss-120b" => Ok(RustModelType::GptOss120B),
            "phi-tiny-moe" => Ok(RustModelType::PhiTinyMoe),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid ModelType: {}",
                value
            ))),
        }
    }
}

impl<'py> IntoPyObject<'py> for RustModelType {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let value = match self {
            RustModelType::GptOss20B => "gpt-oss-20b",
            RustModelType::GptOss120B => "gpt-oss-120b",
            RustModelType::PhiTinyMoe => "phi-tiny-moe",
        };
        Ok(value.into_pyobject(py).unwrap().into_any())
    }
}
