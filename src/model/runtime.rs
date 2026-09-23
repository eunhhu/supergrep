//! Deterministic resolution of the bundled ONNX Runtime shared library.
//!
//! Search paths deliberately do not include the current working directory.
//! A packaged executable is expected to have `runtime/libonnxruntime.so.1.20.0`
//! immediately below its containing directory.  Developers may override that
//! location with an *absolute* `SUPERGREP_ORT_LIB` path.

use std::ffi::{CStr, OsStr};
use std::path::{Path, PathBuf};

use crate::{Result, SupergrepError};

/// Environment variable used to select an explicit development runtime.
pub const ORT_LIBRARY_ENV: &str = "SUPERGREP_ORT_LIB";

/// SONAME shipped by the v0.1 Linux ARM64 distribution bundle.
pub const BUNDLED_ORT_LIBRARY_FILE: &str = "libonnxruntime.so.1.20.0";

/// Exact runtime version measured and shipped by the v0.1 ARM64 bundle.
pub const REQUIRED_ORT_RUNTIME_VERSION: &str = "1.20.0";

/// How a runtime library was selected.  This is retained for diagnostics and
/// reproducible model-load reports; it does not imply that the library has
/// been dynamically loaded yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeLibrarySource {
    Environment,
    AdjacentBundle,
}

/// A regular-file runtime location selected without current-directory search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeLibrary {
    pub path: PathBuf,
    pub source: RuntimeLibrarySource,
    /// Every concrete candidate considered, in resolution order.
    pub searched_paths: Vec<PathBuf>,
}

/// Resolve a runtime for a known executable path using the real environment.
///
/// The caller should obtain the executable path explicitly (normally with
/// `std::env::current_exe`) so that package resolution is independent of the
/// caller's current directory.
pub fn resolve_runtime_library(executable_path: &Path) -> Result<RuntimeLibrary> {
    let environment_value = std::env::var_os(ORT_LIBRARY_ENV);
    resolve_runtime_library_with_env(executable_path, environment_value.as_deref())
}

/// Resolve a runtime for a known executable path with an injected environment
/// value.  The injected variant keeps tests deterministic and avoids mutating
/// process-global environment state.
pub fn resolve_runtime_library_with_env(
    executable_path: &Path,
    environment_value: Option<&OsStr>,
) -> Result<RuntimeLibrary> {
    if !executable_path.is_absolute() {
        return Err(SupergrepError::runtime(format!(
            "cannot resolve ONNX Runtime from non-absolute executable path `{}`; supply an absolute executable path",
            executable_path.display()
        )));
    }

    let executable_parent = executable_path.parent().ok_or_else(|| {
        SupergrepError::runtime(format!(
            "cannot resolve ONNX Runtime because executable path `{}` has no parent directory",
            executable_path.display()
        ))
    })?;
    let adjacent_path = executable_parent
        .join("runtime")
        .join(BUNDLED_ORT_LIBRARY_FILE);

    let mut searched_paths = Vec::with_capacity(2);
    let mut environment_problem = None;

    if let Some(value) = environment_value {
        let candidate = PathBuf::from(value);
        searched_paths.push(candidate.clone());
        if !candidate.is_absolute() {
            environment_problem = Some(format!(
                "{ORT_LIBRARY_ENV}=`{}` is not absolute",
                candidate.display()
            ));
        } else if candidate.is_file() {
            return Ok(RuntimeLibrary {
                path: candidate,
                source: RuntimeLibrarySource::Environment,
                searched_paths,
            });
        } else {
            environment_problem = Some(format!(
                "{ORT_LIBRARY_ENV}=`{}` is not an existing regular file",
                candidate.display()
            ));
        }
    }

    searched_paths.push(adjacent_path.clone());
    if adjacent_path.is_file() {
        return Ok(RuntimeLibrary {
            path: adjacent_path,
            source: RuntimeLibrarySource::AdjacentBundle,
            searched_paths,
        });
    }

    let environment_detail = environment_problem
        .map(|problem| format!("; {problem}"))
        .unwrap_or_default();
    Err(SupergrepError::runtime(format!(
        "could not resolve ONNX Runtime shared library for executable `{}`{environment_detail}; searched paths: {}",
        executable_path.display(),
        format_paths(&searched_paths)
    )))
}

/// Loads only the runtime's small C API base table to validate its version
/// and API level before `ort` initializes.  This prevents rc.9 from merely
/// warning and continuing against an unmeasured newer minor release.
pub fn validate_ort_runtime(path: &Path) -> Result<String> {
    if !path.is_absolute() {
        return Err(SupergrepError::runtime(format!(
            "ONNX Runtime library path must be absolute: {}",
            path.display()
        )));
    }
    if !path.is_file() {
        return Err(SupergrepError::runtime(format!(
            "ONNX Runtime library is not a regular file: {}",
            path.display()
        )));
    }

    let library = unsafe { libloading::Library::new(path) }.map_err(|error| {
        SupergrepError::runtime(format!(
            "could not inspect ONNX Runtime library {}: {error}",
            path.display()
        ))
    })?;
    let get_base = unsafe {
        library.get::<unsafe extern "C" fn() -> *const ort_sys::OrtApiBase>(b"OrtGetApiBase")
    }
    .map_err(|error| {
        SupergrepError::runtime(format!(
            "ONNX Runtime library {} is missing OrtGetApiBase: {error}",
            path.display()
        ))
    })?;
    let base = unsafe { get_base() };
    if base.is_null() {
        return Err(SupergrepError::runtime(format!(
            "ONNX Runtime library {} returned a null API base",
            path.display()
        )));
    }
    let get_version = unsafe { (*base).GetVersionString }.ok_or_else(|| {
        SupergrepError::runtime(format!(
            "ONNX Runtime library {} has no GetVersionString function",
            path.display()
        ))
    })?;
    let version_ptr = unsafe { get_version() };
    if version_ptr.is_null() {
        return Err(SupergrepError::runtime(format!(
            "ONNX Runtime library {} returned a null version string",
            path.display()
        )));
    }
    let version = unsafe { CStr::from_ptr(version_ptr) }
        .to_str()
        .map_err(|error| {
            SupergrepError::runtime(format!(
                "ONNX Runtime library {} returned a non-UTF-8 version string: {error}",
                path.display()
            ))
        })?;
    validate_ort_version_string(version)?;

    let get_api = unsafe { (*base).GetApi }.ok_or_else(|| {
        SupergrepError::runtime(format!(
            "ONNX Runtime library {} has no GetApi function",
            path.display()
        ))
    })?;
    let api = unsafe { get_api(ort_sys::ORT_API_VERSION) };
    if api.is_null() {
        return Err(SupergrepError::runtime(format!(
            "ONNX Runtime {version} does not provide required C API level {}",
            ort_sys::ORT_API_VERSION
        )));
    }
    Ok(version.to_owned())
}

/// Checks the semantic version independently of loading the platform dylib.
/// Product runtime builds remain pinned to one minor/API implementation.
pub fn validate_ort_version_string(version: &str) -> Result<()> {
    if version != REQUIRED_ORT_RUNTIME_VERSION {
        return Err(SupergrepError::runtime(format!(
            "ONNX Runtime version {version:?} is unsupported; supergrep v0.1 requires exactly {REQUIRED_ORT_RUNTIME_VERSION}"
        )));
    }
    Ok(())
}

fn format_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| format!("`{}`", path.display()))
        .collect::<Vec<_>>()
        .join(", ")
}
