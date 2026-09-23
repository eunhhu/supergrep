use std::path::Path;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, SupergrepError>;

#[derive(Debug, Error)]
pub enum SupergrepError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid input: {0}")]
    Input(String),

    #[error("model error: {0}")]
    Model(String),

    #[error("model artifact is missing: {}", .0.display())]
    MissingArtifact(std::path::PathBuf),

    #[error("runtime error: {0}")]
    Runtime(String),

    #[error("internal error: {0}")]
    Internal(String),
}

impl SupergrepError {
    pub fn model(message: impl Into<String>) -> Self {
        Self::Model(message.into())
    }

    pub fn runtime(message: impl Into<String>) -> Self {
        Self::Runtime(message.into())
    }

    pub fn require_file(path: &Path) -> Result<()> {
        if path.is_file() {
            Ok(())
        } else {
            Err(Self::MissingArtifact(path.to_path_buf()))
        }
    }
}
