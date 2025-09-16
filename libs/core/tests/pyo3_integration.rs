use core::CoreCache;
use pyo3::prelude::*;

#[test]
fn test_core_cache_creation() {
    let cache = CoreCache::new(100);
    assert_eq!(cache.capacity(), 100);
}

#[test]
fn test_core_cache_different_capacity() {
    let cache = CoreCache::new(500);
    assert_eq!(cache.capacity(), 500);
}

#[test]
fn test_pyo3_module_can_be_initialized() {
    Python::attach(|py| {
        let result = pyo3::wrap_pymodule!(core::core)(py);
        // Module creation should succeed - we just check that it doesn't panic
        let _ = result;
    });
}

#[test]
fn test_pyo3_add_function() {
    Python::attach(|py| {
        let module = pyo3::wrap_pymodule!(core::core)(py).into_bound(py);
        let add_func = module.getattr("add").unwrap();
        let result: i64 = add_func.call1((5, 3)).unwrap().extract().unwrap();
        assert_eq!(result, 8);
    });
}

#[test]
fn test_pyo3_get_version_function() {
    Python::attach(|py| {
        let module = pyo3::wrap_pymodule!(core::core)(py).into_bound(py);
        let version_func = module.getattr("get_version").unwrap();
        let result: String = version_func.call0().unwrap().extract().unwrap();
        assert_eq!(result, "0.1.0");
    });
}

#[test]
fn test_pyo3_core_cache_class() {
    Python::attach(|py| {
        let module = pyo3::wrap_pymodule!(core::core)(py).into_bound(py);
        let cache_class = module.getattr("PyCoreCache").unwrap();
        let cache_instance = cache_class.call1((100,)).unwrap();
        let capacity: usize = cache_instance
            .call_method0("capacity")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(capacity, 100);
    });
}
