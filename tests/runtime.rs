use std::ffi::OsStr;
use std::fs;
use std::path::Path;

use supergrep::model::{
    resolve_runtime_library_with_env, validate_ort_version_string, RuntimeLibrarySource,
    BUNDLED_ORT_LIBRARY_FILE, REQUIRED_ORT_RUNTIME_VERSION,
};
use tempfile::tempdir;

fn executable_path(root: &Path) -> std::path::PathBuf {
    root.join("bin").join("supergrep")
}

#[test]
fn absolute_environment_file_wins_over_adjacent_bundle() {
    let temporary = tempdir().unwrap();
    let executable = executable_path(temporary.path());
    let bundle = executable
        .parent()
        .unwrap()
        .join("runtime")
        .join(BUNDLED_ORT_LIBRARY_FILE);
    let override_path = temporary.path().join("development/libonnxruntime.so");
    fs::create_dir_all(bundle.parent().unwrap()).unwrap();
    fs::create_dir_all(override_path.parent().unwrap()).unwrap();
    fs::write(&bundle, b"bundle").unwrap();
    fs::write(&override_path, b"override").unwrap();

    let resolved =
        resolve_runtime_library_with_env(&executable, Some(override_path.as_os_str())).unwrap();

    assert_eq!(resolved.path, override_path);
    assert_eq!(resolved.source, RuntimeLibrarySource::Environment);
    assert_eq!(resolved.searched_paths, vec![resolved.path.clone()]);
}

#[test]
fn uses_only_path_adjacent_to_supplied_executable_when_environment_is_missing() {
    let temporary = tempdir().unwrap();
    let executable = executable_path(temporary.path());
    let bundle = executable
        .parent()
        .unwrap()
        .join("runtime")
        .join(BUNDLED_ORT_LIBRARY_FILE);
    fs::create_dir_all(bundle.parent().unwrap()).unwrap();
    fs::write(&bundle, b"bundle").unwrap();

    let resolved = resolve_runtime_library_with_env(&executable, None).unwrap();

    assert_eq!(resolved.path, bundle);
    assert_eq!(resolved.source, RuntimeLibrarySource::AdjacentBundle);
    assert_eq!(resolved.searched_paths, vec![resolved.path.clone()]);
}

#[test]
fn missing_environment_file_falls_back_to_adjacent_bundle_and_records_both() {
    let temporary = tempdir().unwrap();
    let executable = executable_path(temporary.path());
    let missing_override = temporary.path().join("missing/libonnxruntime.so");
    let bundle = executable
        .parent()
        .unwrap()
        .join("runtime")
        .join(BUNDLED_ORT_LIBRARY_FILE);
    fs::create_dir_all(bundle.parent().unwrap()).unwrap();
    fs::write(&bundle, b"bundle").unwrap();

    let resolved =
        resolve_runtime_library_with_env(&executable, Some(missing_override.as_os_str())).unwrap();

    assert_eq!(resolved.path, bundle);
    assert_eq!(resolved.source, RuntimeLibrarySource::AdjacentBundle);
    assert_eq!(
        resolved.searched_paths,
        vec![missing_override, resolved.path.clone()]
    );
}

#[test]
fn nonregular_environment_value_and_bundle_return_actionable_error() {
    let temporary = tempdir().unwrap();
    let executable = executable_path(temporary.path());
    let directory_override = temporary.path().join("directory-not-library");
    fs::create_dir_all(&directory_override).unwrap();

    let error = resolve_runtime_library_with_env(&executable, Some(directory_override.as_os_str()))
        .unwrap_err()
        .to_string();
    let expected_bundle = executable
        .parent()
        .unwrap()
        .join("runtime")
        .join(BUNDLED_ORT_LIBRARY_FILE);

    assert!(error.contains("not an existing regular file"), "{error}");
    assert!(
        error.contains(&directory_override.display().to_string()),
        "{error}"
    );
    assert!(
        error.contains(&expected_bundle.display().to_string()),
        "{error}"
    );
    assert!(error.contains("searched paths"), "{error}");
}

#[test]
fn relative_environment_value_is_rejected_without_current_directory_lookup() {
    let temporary = tempdir().unwrap();
    let executable = executable_path(temporary.path());

    let error = resolve_runtime_library_with_env(
        &executable,
        Some(OsStr::new("runtime/libonnxruntime.so")),
    )
    .unwrap_err()
    .to_string();

    assert!(error.contains("is not absolute"), "{error}");
    assert!(error.contains("runtime/libonnxruntime.so"), "{error}");
    assert!(error.contains("searched paths"), "{error}");
}

#[test]
fn relative_executable_path_is_rejected_without_current_directory_lookup() {
    let error = resolve_runtime_library_with_env(Path::new("bin/supergrep"), None)
        .unwrap_err()
        .to_string();

    assert!(error.contains("non-absolute executable path"), "{error}");
}

#[test]
fn runtime_version_must_match_the_exactly_measured_ort_release() {
    validate_ort_version_string(REQUIRED_ORT_RUNTIME_VERSION).unwrap();
    for unsupported in ["1.19.2", "1.20.1", "1.21.0", "1.20.0-rc.1"] {
        let error = validate_ort_version_string(unsupported)
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires exactly 1.20.0"), "{error}");
    }
}
