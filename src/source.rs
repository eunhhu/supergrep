//! Immutable, UTF-8 source snapshots and byte-to-line mapping.
//!
//! A [`Source`] is deliberately a snapshot rather than a lazy path reference:
//! search results must describe the bytes that were actually searched, even if
//! the file changes before output is rendered.

use std::{
    fs::{self, File, Metadata},
    io::{self, Read},
    ops::Range,
    path::{Path, PathBuf},
    time::SystemTime,
};

use thiserror::Error;

/// Stable identifier assigned by the caller for the lifetime of one scan.
///
/// It intentionally does not encode a path: paths can be renamed while a
/// search is running, whereas a source snapshot remains valid on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceId(pub usize);

impl SourceId {
    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    pub const fn get(self) -> usize {
        self.0
    }
}

/// Read-time facts used to detect the common forms of replacement or mutation
/// while a file is being snapshotted.
///
/// `modified` is optional because some filesystems cannot provide it. On Unix,
/// `identity` additionally records `(device, inode)` so replacement is usually
/// visible even when a timestamp has coarse precision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceMetadata {
    pub len: u64,
    pub modified: Option<SystemTime>,
    pub identity: Option<(u64, u64)>,
}

impl SourceMetadata {
    pub fn from_fs(metadata: &Metadata) -> Self {
        Self {
            len: metadata.len(),
            modified: metadata.modified().ok(),
            identity: file_identity(metadata),
        }
    }

    /// Metadata for an in-memory source. It is intentionally distinguishable
    /// from a successfully read file by its absent timestamp and identity.
    pub fn synthetic(len: u64) -> Self {
        Self {
            len,
            modified: None,
            identity: None,
        }
    }
}

#[cfg(unix)]
fn file_identity(metadata: &Metadata) -> Option<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;

    Some((metadata.dev(), metadata.ino()))
}

#[cfg(not(unix))]
fn file_identity(_metadata: &Metadata) -> Option<(u64, u64)> {
    None
}

/// A validated, immutable UTF-8 file snapshot.
#[derive(Debug, Clone)]
pub struct Source {
    id: SourceId,
    path: PathBuf,
    text: String,
    line_starts: Vec<usize>,
    metadata: SourceMetadata,
}

impl Source {
    /// Makes an in-memory source for deterministic tests and callers that
    /// already possess validated UTF-8 text.
    pub fn from_text(id: SourceId, path: impl Into<PathBuf>, text: impl Into<String>) -> Self {
        let text = text.into();
        let metadata = SourceMetadata::synthetic(text.len() as u64);
        Self::from_validated_text(id, path.into(), text, metadata)
    }

    /// Makes a source from bytes after applying the same NUL and UTF-8 policy
    /// as filesystem reads.
    pub fn from_bytes(
        id: SourceId,
        path: impl Into<PathBuf>,
        bytes: Vec<u8>,
        metadata: SourceMetadata,
    ) -> Result<Self, SourceReadError> {
        let bytes_read = bytes.len() as u64;
        let text = validate_bytes(bytes, bytes_read)?;
        Ok(Self::from_validated_text(id, path.into(), text, metadata))
    }

    /// Snapshots one regular, non-symlink file without ever reading more than
    /// `max_bytes`. The caller can account for every physical byte read via
    /// [`SourceReadError::bytes_read`] when a validation failure occurs.
    pub fn read(
        id: SourceId,
        path: impl AsRef<Path>,
        max_bytes: u64,
    ) -> Result<Self, SourceReadError> {
        let path = path.as_ref();
        let link_metadata = fs::symlink_metadata(path).map_err(|source| SourceReadError::Io {
            source,
            bytes_read: 0,
        })?;
        if link_metadata.file_type().is_symlink() {
            return Err(SourceReadError::Symlink { bytes_read: 0 });
        }
        if !link_metadata.file_type().is_file() {
            return Err(SourceReadError::NotRegular { bytes_read: 0 });
        }
        if link_metadata.len() > max_bytes {
            return Err(SourceReadError::TooLarge {
                limit: max_bytes,
                observed: link_metadata.len(),
                bytes_read: 0,
            });
        }

        let mut file = File::open(path).map_err(|source| SourceReadError::Io {
            source,
            bytes_read: 0,
        })?;
        let before_fs = file.metadata().map_err(|source| SourceReadError::Io {
            source,
            bytes_read: 0,
        })?;
        if !before_fs.file_type().is_file() {
            return Err(SourceReadError::NotRegular { bytes_read: 0 });
        }
        if before_fs.len() > max_bytes {
            return Err(SourceReadError::TooLarge {
                limit: max_bytes,
                observed: before_fs.len(),
                bytes_read: 0,
            });
        }
        let before = SourceMetadata::from_fs(&before_fs);

        // Opening a path can race with a replacement. Re-check the directory
        // entry before consuming any bytes so a newly introduced symlink or a
        // different regular file is discarded rather than searched. This is
        // best-effort (portable std I/O has no cross-platform O_NOFOLLOW), but
        // it closes the common check-then-open window and records a diagnostic
        // instead of accepting the wrong snapshot.
        let opened_path_fs = fs::symlink_metadata(path).map_err(|source| SourceReadError::Io {
            source,
            bytes_read: 0,
        })?;
        let opened_path = SourceMetadata::from_fs(&opened_path_fs);
        if opened_path_fs.file_type().is_symlink()
            || !opened_path_fs.file_type().is_file()
            || opened_path != before
        {
            return Err(SourceReadError::Changed {
                before,
                after: opened_path,
                bytes_read: 0,
            });
        }

        let mut bytes = Vec::new();
        let mut remaining = max_bytes;
        let mut buffer = [0_u8; 8192];
        while remaining > 0 {
            let requested = usize::try_from(remaining)
                .unwrap_or(usize::MAX)
                .min(buffer.len());
            let read = match file.read(&mut buffer[..requested]) {
                Ok(read) => read,
                Err(source) => {
                    return Err(SourceReadError::Io {
                        source,
                        bytes_read: bytes.len() as u64,
                    });
                }
            };
            if read == 0 {
                break;
            }
            bytes.extend_from_slice(&buffer[..read]);
            remaining -= read as u64;
        }

        let bytes_read = bytes.len() as u64;
        let after_fs = file
            .metadata()
            .map_err(|source| SourceReadError::Io { source, bytes_read })?;
        if after_fs.len() > max_bytes {
            return Err(SourceReadError::TooLarge {
                limit: max_bytes,
                observed: after_fs.len(),
                bytes_read,
            });
        }
        let after = SourceMetadata::from_fs(&after_fs);
        let path_after_fs = fs::symlink_metadata(path)
            .map_err(|source| SourceReadError::Io { source, bytes_read })?;
        let path_after = SourceMetadata::from_fs(&path_after_fs);
        if path_after_fs.file_type().is_symlink()
            || !path_after_fs.file_type().is_file()
            || before != after
            || before != path_after
            || bytes_read != after.len
        {
            return Err(SourceReadError::Changed {
                before,
                after: path_after,
                bytes_read,
            });
        }

        let text = validate_bytes(bytes, bytes_read)?;
        Ok(Self::from_validated_text(
            id,
            path.to_path_buf(),
            text,
            before,
        ))
    }

    fn from_validated_text(
        id: SourceId,
        path: PathBuf,
        text: String,
        metadata: SourceMetadata,
    ) -> Self {
        let line_starts = line_starts(&text);
        Self {
            id,
            path,
            text,
            line_starts,
            metadata,
        }
    }

    pub const fn id(&self) -> SourceId {
        self.id
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The exact text that was searched. It is never normalized or rewritten.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The original source bytes. Since a source is validated UTF-8, this is
    /// exactly `self.text().as_bytes()` without a second allocation.
    pub fn bytes(&self) -> &[u8] {
        self.text.as_bytes()
    }

    pub fn len_bytes(&self) -> usize {
        self.text.len()
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    pub fn metadata(&self) -> &SourceMetadata {
        &self.metadata
    }

    /// Byte offsets at which the corresponding 1-based line begins. The
    /// first element is always zero; a final newline creates an empty final
    /// logical line, as text editors conventionally do.
    pub fn line_starts(&self) -> &[usize] {
        &self.line_starts
    }

    pub fn line_count(&self) -> usize {
        self.line_starts.len()
    }

    /// Returns a 1-based line number for a byte offset, including `len()`.
    /// The caller is responsible for using character boundaries when it needs
    /// to slice text; byte-to-line lookup itself also has useful behavior for
    /// an interior UTF-8 byte.
    pub fn line_of_byte(&self, byte: usize) -> Option<usize> {
        if byte > self.text.len() {
            return None;
        }
        let index = self.line_starts.partition_point(|&start| start <= byte);
        Some(index.max(1))
    }

    pub fn line_start(&self, line: usize) -> Option<usize> {
        line.checked_sub(1)
            .and_then(|index| self.line_starts.get(index).copied())
    }

    /// Returns the exclusive byte end of a 1-based line. For all but the
    /// final line this includes its `\n`, preserving CRLF as `\r\n` bytes.
    pub fn line_end(&self, line: usize) -> Option<usize> {
        let index = line.checked_sub(1)?;
        if index >= self.line_starts.len() {
            return None;
        }
        Some(
            self.line_starts
                .get(index + 1)
                .copied()
                .unwrap_or(self.text.len()),
        )
    }

    /// Gets a UTF-8-valid original slice. Invalid byte ranges are rejected
    /// rather than adjusted, which keeps output evidence honest.
    pub fn slice(&self, range: Range<usize>) -> Option<&str> {
        self.text.get(range)
    }

    /// Maps a non-empty, valid byte range to 1-based inclusive line bounds.
    pub fn line_span(&self, range: Range<usize>) -> Option<(usize, usize)> {
        if range.start >= range.end
            || range.end > self.text.len()
            || !self.text.is_char_boundary(range.start)
            || !self.text.is_char_boundary(range.end)
        {
            return None;
        }
        Some((
            self.line_of_byte(range.start)?,
            self.line_of_byte(range.end - 1)?,
        ))
    }

    /// Best-effort check for a later on-disk replacement. Search rendering
    /// should use the snapshot regardless; this is useful to attach a warning
    /// without re-reading different content.
    pub fn changed_on_disk(&self) -> io::Result<bool> {
        let metadata = fs::symlink_metadata(&self.path)?;
        if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
            return Ok(true);
        }
        Ok(SourceMetadata::from_fs(&metadata) != self.metadata)
    }
}

fn line_starts(text: &str) -> Vec<usize> {
    let mut starts = Vec::with_capacity(text.bytes().filter(|&byte| byte == b'\n').count() + 1);
    starts.push(0);
    for (index, byte) in text.bytes().enumerate() {
        if byte == b'\n' {
            starts.push(index + 1);
        }
    }
    starts
}

fn validate_bytes(bytes: Vec<u8>, bytes_read: u64) -> Result<String, SourceReadError> {
    if bytes.contains(&0) {
        return Err(SourceReadError::Nul { bytes_read });
    }
    String::from_utf8(bytes).map_err(|error| SourceReadError::InvalidUtf8 {
        valid_up_to: error.utf8_error().valid_up_to(),
        bytes_read,
    })
}

/// A non-fatal source-read failure. Discovery turns these into diagnostics so
/// one unreadable file does not discard valid results from its siblings.
#[derive(Debug, Error)]
pub enum SourceReadError {
    #[error("could not read source: {source}")]
    Io {
        #[source]
        source: io::Error,
        bytes_read: u64,
    },

    #[error("source is a symbolic link")]
    Symlink { bytes_read: u64 },

    #[error("source is not a regular file")]
    NotRegular { bytes_read: u64 },

    #[error("source exceeds the {limit}-byte read limit (observed {observed} bytes)")]
    TooLarge {
        limit: u64,
        observed: u64,
        bytes_read: u64,
    },

    #[error("source contains a NUL byte")]
    Nul { bytes_read: u64 },

    #[error("source is not valid UTF-8 (valid through byte {valid_up_to})")]
    InvalidUtf8 { valid_up_to: usize, bytes_read: u64 },

    #[error("source changed while it was read")]
    Changed {
        before: SourceMetadata,
        after: SourceMetadata,
        bytes_read: u64,
    },
}

impl SourceReadError {
    pub const fn bytes_read(&self) -> u64 {
        match self {
            Self::Io { bytes_read, .. }
            | Self::Symlink { bytes_read }
            | Self::NotRegular { bytes_read }
            | Self::TooLarge { bytes_read, .. }
            | Self::Nul { bytes_read }
            | Self::InvalidUtf8 { bytes_read, .. }
            | Self::Changed { bytes_read, .. } => *bytes_read,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_index_preserves_crlf_and_final_newline() {
        let source = Source::from_text(SourceId::new(7), "memory", "first\r\n한글\n");
        assert_eq!(source.line_starts(), &[0, 7, 14]);
        assert_eq!(source.line_count(), 3);
        assert_eq!(source.line_of_byte(0), Some(1));
        assert_eq!(source.line_of_byte(6), Some(1));
        assert_eq!(source.line_of_byte(7), Some(2));
        assert_eq!(source.line_of_byte(14), Some(3));
        assert_eq!(source.line_span(7..13), Some((2, 2)));
    }

    #[test]
    fn rejects_nul_and_invalid_utf8_without_replacement() {
        let metadata = SourceMetadata::synthetic(2);
        assert!(matches!(
            Source::from_bytes(SourceId::new(0), "nul", vec![b'a', 0], metadata.clone()),
            Err(SourceReadError::Nul { .. })
        ));
        assert!(matches!(
            Source::from_bytes(SourceId::new(0), "bad", vec![0xff], metadata),
            Err(SourceReadError::InvalidUtf8 { .. })
        ));
    }
}
