use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=PYTHON_SYS_EXECUTABLE");
    println!("cargo:rerun-if-env-changed=VIRTUAL_ENV");
    println!("cargo:rerun-if-env-changed=UV_PYTHON");

    configure_python_linking();
}

fn configure_python_linking() {
    if let Some((lib_path, version)) = detect_python_env() {
        println!("cargo:rustc-link-search=native={}", lib_path);
        println!("cargo:rustc-link-lib=python{}", version);
        println!("cargo:warning=Using Python {} from {}", version, lib_path);
    } else {
        panic!(
            "Could not find a suitable Python installation. Please set PYTHON_SYS_EXECUTABLE environment variable."
        );
    }
}

fn detect_python_env() -> Option<(String, String)> {
    // Try different Python detection strategies in order of preference

    // 1. Explicit PYTHON_SYS_EXECUTABLE (highest priority)
    if let Ok(python_exe) = env::var("PYTHON_SYS_EXECUTABLE")
        && let Some(config) = get_python_config(&python_exe)
    {
        return Some(config);
    }

    // 2. Virtual environment
    if let Ok(venv_path) = env::var("VIRTUAL_ENV") {
        let python_exe = format!("{}/bin/python", venv_path);
        if Path::new(&python_exe).exists()
            && let Some(config) = get_python_config(&python_exe)
        {
            return Some(config);
        }
    }

    // 3. uv python (check multiple possible locations)
    if let Some(config) = detect_uv_python() {
        return Some(config);
    }

    // 4. System python3
    for python_cmd in ["python3", "python"] {
        if let Ok(output) = Command::new("which").arg(python_cmd).output()
            && output.status.success()
        {
            let python_path_output = String::from_utf8_lossy(&output.stdout);
            let python_path = python_path_output.trim();
            if !python_path.is_empty()
                && let Some(config) = get_python_config(python_path)
            {
                return Some(config);
            }
        }
    }

    None
}

fn detect_uv_python() -> Option<(String, String)> {
    // Try to find uv python installations
    let home_dir = env::var("HOME").ok()?;
    let uv_python_base = format!("{}/.local/share/uv/python", home_dir);

    if !Path::new(&uv_python_base).exists() {
        return None;
    }

    // Look for cpython installations
    if let Ok(entries) = std::fs::read_dir(&uv_python_base) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str()
                && name.starts_with("cpython-")
            {
                let python_exe = format!("{}/{}/bin/python", uv_python_base, name);
                if Path::new(&python_exe).exists()
                    && let Some(config) = get_python_config(&python_exe)
                {
                    return Some(config);
                }
            }
        }
    }

    None
}

fn get_python_config(python_exe: &str) -> Option<(String, String)> {
    let output = Command::new(python_exe)
        .args([
            "-c",
            "import sysconfig, sys; print(f'{sysconfig.get_config_var(\"LIBDIR\")},{sys.version_info.major}.{sys.version_info.minor}')"
        ])
        .output()
        .ok()?;

    if output.status.success() {
        let result_output = String::from_utf8_lossy(&output.stdout);
        let result = result_output.trim();
        let parts: Vec<&str> = result.split(',').collect();
        if parts.len() == 2 && !parts[0].is_empty() && !parts[1].is_empty() {
            // Verify the library directory exists
            if Path::new(parts[0]).exists() {
                return Some((parts[0].to_string(), parts[1].to_string()));
            }
        }
    }

    None
}
