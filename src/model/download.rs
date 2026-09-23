//! Explicit, content-verified model installation and local verification.
//!
//! Registry lookup and verification never open a socket. Only a caller that
//! creates a downloader and invokes download can use the HTTP fetcher.

use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use directories::ProjectDirs;
use sha2::{Digest, Sha256};

use crate::{Result, SupergrepError};

use super::registry::{ModelArtifact, ModelProfile};

const COPY_BUFFER_BYTES: usize = 64 * 1024;
const MAX_STAGING_ATTEMPTS: u32 = 100;

/// A narrow streaming boundary so installer tests need neither a socket nor a
/// local HTTP server.
pub trait ArtifactFetcher {
    fn open(&self, url: &str) -> Result<Box<dyn Read + Send + Sync>>;
}

/// The production HTTPS-only artifact fetcher.
#[derive(Debug, Clone)]
pub struct HttpArtifactFetcher {
    connect_timeout: Duration,
    request_timeout: Duration,
}

impl Default for HttpArtifactFetcher {
    fn default() -> Self {
        Self {
            connect_timeout: Duration::from_secs(20),
            request_timeout: Duration::from_secs(15 * 60),
        }
    }
}

impl HttpArtifactFetcher {
    pub fn with_timeouts(connect_timeout: Duration, request_timeout: Duration) -> Self {
        Self {
            connect_timeout,
            request_timeout,
        }
    }
}

impl ArtifactFetcher for HttpArtifactFetcher {
    fn open(&self, url: &str) -> Result<Box<dyn Read + Send + Sync>> {
        if !url.starts_with("https://") {
            return Err(SupergrepError::model(format!(
                "refusing non-HTTPS model download URL {url:?}"
            )));
        }
        let response = ureq::AgentBuilder::new()
            .https_only(true)
            .timeout_connect(self.connect_timeout)
            .timeout(self.request_timeout)
            .redirects(5)
            .build()
            .get(url)
            .call()
            .map_err(|error| {
                SupergrepError::model(format!("model download request failed: {error}"))
            })?;
        if response.status() != 200 {
            return Err(SupergrepError::model(format!(
                "model download returned HTTP {}; expected 200",
                response.status()
            )));
        }
        Ok(response.into_reader())
    }
}

/// A deterministic root for revision-pinned installations.
///
/// The caller may supply an explicit root (for a CLI cache override or a test)
/// rather than changing process-wide environment variables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCache {
    root: PathBuf,
}

impl ModelCache {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        if root.as_os_str().is_empty() {
            return Err(SupergrepError::Input(
                "model cache path must not be empty".into(),
            ));
        }
        Ok(Self { root })
    }

    /// Resolve the platform cache root without creating it. This never falls
    /// back to the current directory when platform cache discovery fails.
    pub fn platform_default() -> Result<Self> {
        let directories = ProjectDirs::from("org", "openai", "supergrep").ok_or_else(|| {
            SupergrepError::model(
                "could not determine a platform cache directory; pass an explicit model cache path",
            )
        })?;
        Self::new(directories.cache_dir().join("models"))
    }

    pub fn from_override_or_platform_default(override_root: Option<PathBuf>) -> Result<Self> {
        match override_root {
            Some(root) => Self::new(root),
            None => Self::platform_default(),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn profile_parent(&self, profile: &ModelProfile) -> PathBuf {
        self.root.join(profile.id())
    }

    /// Layout: cache-root / profile-id / immutable-revision.
    pub fn profile_dir(&self, profile: &ModelProfile) -> PathBuf {
        self.profile_parent(profile).join(profile.revision())
    }

    pub fn artifact_path(&self, profile: &ModelProfile, artifact: &ModelArtifact) -> PathBuf {
        self.profile_dir(profile).join(&artifact.path)
    }
}

/// A file proven against the registry's exact size and SHA-256.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedArtifact {
    pub path: PathBuf,
    pub sha256: String,
    pub size_bytes: u64,
}

/// A locally verified revision directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedModel {
    pub profile_id: String,
    pub revision: String,
    pub directory: PathBuf,
    pub artifacts: Vec<VerifiedArtifact>,
}

impl VerifiedModel {
    pub fn artifact_path(&self, artifact: &ModelArtifact) -> PathBuf {
        self.directory.join(&artifact.path)
    }

    pub fn expected_tokenizer_path(&self, profile: &ModelProfile) -> PathBuf {
        self.directory.join(&profile.tokenizer().file)
    }
}

/// Verify a caller-supplied local directory. No network fallback is possible.
pub fn verify_directory(profile: &ModelProfile, directory: &Path) -> Result<VerifiedModel> {
    let metadata = match fs::symlink_metadata(directory) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(SupergrepError::MissingArtifact(directory.to_path_buf()));
        }
        Err(error) => return Err(error.into()),
    };
    if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
        return Err(SupergrepError::model(format!(
            "model directory is not a real directory: {}",
            directory.display()
        )));
    }

    let mut artifacts = Vec::with_capacity(profile.artifacts().len());
    for artifact in profile.artifacts() {
        let path = directory.join(&artifact.path);
        verify_artifact(artifact, &path)?;
        artifacts.push(VerifiedArtifact {
            path,
            sha256: artifact.sha256.clone(),
            size_bytes: artifact.size_bytes,
        });
    }
    Ok(VerifiedModel {
        profile_id: profile.id().to_owned(),
        revision: profile.revision().to_owned(),
        directory: directory.to_path_buf(),
        artifacts,
    })
}

/// Verify a profile at its resolved cache path. It reads only local files.
pub fn verify_cached(profile: &ModelProfile, cache: &ModelCache) -> Result<VerifiedModel> {
    verify_directory(profile, &cache.profile_dir(profile))
}

/// Installs all registry-declared, content-addressed artifacts under one
/// revision directory. The whole directory becomes visible only after each
/// artifact was streamed, size-checked, hashed, synced, and renamed.
pub struct ModelDownloader<F> {
    cache: ModelCache,
    fetcher: F,
}

impl<F> ModelDownloader<F> {
    pub fn with_fetcher(cache: ModelCache, fetcher: F) -> Self {
        Self { cache, fetcher }
    }

    pub fn cache(&self) -> &ModelCache {
        &self.cache
    }

    pub fn fetcher(&self) -> &F {
        &self.fetcher
    }
}

impl ModelDownloader<HttpArtifactFetcher> {
    pub fn with_http(cache: ModelCache) -> Self {
        Self::with_fetcher(cache, HttpArtifactFetcher::default())
    }
}

impl<F: ArtifactFetcher> ModelDownloader<F> {
    /// Explicitly fetch and atomically install a profile. Existing installations
    /// are never overwritten: they must first pass local verification.
    pub fn download(&self, profile: &ModelProfile) -> Result<VerifiedModel> {
        let parent = self.cache.profile_parent(profile);
        fs::create_dir_all(&parent)?;

        let lock_path = parent.join(format!(".{}.download.lock", profile.revision()));
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)?;
        fs2::FileExt::lock_exclusive(&lock).map_err(|error| {
            SupergrepError::model(format!(
                "could not acquire model installation lock {}: {error}",
                lock_path.display()
            ))
        })?;

        let outcome = self.download_locked(profile, &parent);
        let unlock_result = fs2::FileExt::unlock(&lock);
        match (outcome, unlock_result) {
            (Ok(model), Ok(())) => Ok(model),
            (Ok(_), Err(error)) => Err(SupergrepError::model(format!(
                "model installed but could not release installation lock {}: {error}",
                lock_path.display()
            ))),
            (Err(error), _) => Err(error),
        }
    }

    fn download_locked(&self, profile: &ModelProfile, parent: &Path) -> Result<VerifiedModel> {
        let final_dir = self.cache.profile_dir(profile);
        if final_dir.exists() {
            return verify_directory(profile, &final_dir);
        }

        let staging_dir = create_staging_dir(parent, profile.revision())?;
        let result = (|| {
            for (index, artifact) in profile.artifacts().iter().enumerate() {
                let artifact_path = staging_dir.join(&artifact.path);
                let artifact_parent = artifact_path.parent().ok_or_else(|| {
                    SupergrepError::Internal("validated artifact path has no parent".into())
                })?;
                fs::create_dir_all(artifact_parent)?;
                let reader = self.fetcher.open(&profile.artifact_url(artifact))?;
                download_artifact(reader, artifact, &artifact_path, index)?;
            }

            // Do not replace a directory another process created while the
            // advisory lock was held. A caller can inspect/repair it explicitly.
            if final_dir.exists() {
                return Err(SupergrepError::model(format!(
                    "refusing to replace an existing model directory {}",
                    final_dir.display()
                )));
            }
            fs::rename(&staging_dir, &final_dir)?;
            verify_directory(profile, &final_dir)
        })();

        if result.is_err() && staging_dir.exists() {
            let _ = fs::remove_dir_all(&staging_dir);
        }
        result
    }
}

fn create_staging_dir(parent: &Path, revision: &str) -> Result<PathBuf> {
    for sequence in 0..MAX_STAGING_ATTEMPTS {
        let candidate = parent.join(format!(
            ".{revision}.staging-{}-{sequence}",
            std::process::id()
        ));
        match fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(SupergrepError::model(format!(
        "could not allocate a unique staging directory beneath {}",
        parent.display()
    )))
}

fn download_artifact(
    mut reader: Box<dyn Read + Send + Sync>,
    artifact: &ModelArtifact,
    destination: &Path,
    sequence: usize,
) -> Result<()> {
    let filename = destination
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            SupergrepError::Internal("validated artifact destination has no UTF-8 filename".into())
        })?;
    let temporary = destination.with_file_name(format!(
        ".{filename}.part-{}-{sequence}",
        std::process::id()
    ));
    let result = (|| {
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        let (actual_size, actual_sha256) =
            copy_and_hash(&mut reader, &mut output, artifact.size_bytes)?;
        output.sync_all()?;
        drop(output);

        if actual_size != artifact.size_bytes {
            return Err(SupergrepError::model(format!(
                "downloaded {} bytes for {}; expected {}",
                actual_size,
                artifact.path.display(),
                artifact.size_bytes
            )));
        }
        if actual_sha256 != artifact.sha256 {
            return Err(SupergrepError::model(format!(
                "SHA-256 mismatch for {}; expected {}, got {}",
                artifact.path.display(),
                artifact.sha256,
                actual_sha256
            )));
        }
        fs::rename(&temporary, destination)?;
        Ok(())
    })();
    if result.is_err() && temporary.exists() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn copy_and_hash(
    reader: &mut dyn Read,
    writer: &mut dyn Write,
    expected_size: u64,
) -> Result<(u64, String)> {
    let mut bytes_written = 0u64;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; COPY_BUFFER_BYTES];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        bytes_written = bytes_written
            .checked_add(count as u64)
            .ok_or_else(|| SupergrepError::model("model download byte count overflowed"))?;
        if bytes_written > expected_size {
            return Err(SupergrepError::model(format!(
                "model download exceeded expected size {expected_size}"
            )));
        }
        writer.write_all(&buffer[..count])?;
        hasher.update(&buffer[..count]);
    }
    Ok((bytes_written, hex_digest(hasher.finalize().as_slice())))
}

fn verify_artifact(artifact: &ModelArtifact, path: &Path) -> Result<()> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(SupergrepError::MissingArtifact(path.to_path_buf()));
        }
        Err(error) => return Err(error.into()),
    };
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(SupergrepError::model(format!(
            "model artifact must be a regular file: {}",
            path.display()
        )));
    }
    if metadata.len() != artifact.size_bytes {
        return Err(SupergrepError::model(format!(
            "model artifact has {} bytes at {}; expected {}",
            metadata.len(),
            path.display(),
            artifact.size_bytes
        )));
    }

    let mut file = File::open(path)?;
    let mut sink = std::io::sink();
    let (actual_size, actual_sha256) = copy_and_hash(&mut file, &mut sink, artifact.size_bytes)?;
    if actual_size != artifact.size_bytes || actual_sha256 != artifact.sha256 {
        return Err(SupergrepError::model(format!(
            "model artifact digest mismatch at {}; expected {}, got {}",
            path.display(),
            artifact.sha256,
            actual_sha256
        )));
    }
    Ok(())
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(HEX[(byte >> 4) as usize] as char);
        result.push(HEX[(byte & 0x0f) as usize] as char);
    }
    result
}
