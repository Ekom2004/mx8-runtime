#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

pub mod fs;

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use mx8_core::types::ManifestHash;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ManifestStoreError {
    #[error("invalid manifest_hash")]
    InvalidManifestHash,
    #[error("invalid intent_key")]
    InvalidIntentKey,
    #[error("manifest not found: {0}")]
    NotFound(String),
    #[error("manifest bytes do not match existing content for hash {hash}")]
    HashCollision { hash: String },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Minimal interface for v0 snapshot storage.
///
/// - Manifests are immutable by `manifest_hash`.
/// - `intent_key` is the stable identifier for “what the user pointed at”.
pub trait ManifestStore: Send + Sync + 'static {
    fn put_manifest_bytes(
        &self,
        hash: &ManifestHash,
        bytes: &[u8],
    ) -> Result<(), ManifestStoreError>;
    fn get_manifest_bytes(&self, hash: &ManifestHash) -> Result<Vec<u8>, ManifestStoreError>;

    fn get_current_snapshot(
        &self,
        intent_key: &str,
    ) -> Result<Option<ManifestHash>, ManifestStoreError>;
    fn set_current_snapshot(
        &self,
        intent_key: &str,
        hash: &ManifestHash,
    ) -> Result<(), ManifestStoreError>;
}

fn validate_key_component(value: &str) -> bool {
    if value.trim().is_empty() {
        return false;
    }
    if value.contains('/') || value.contains('\\') {
        return false;
    }
    if value.contains("..") {
        return false;
    }
    true
}

fn validate_manifest_hash(hash: &ManifestHash) -> bool {
    validate_key_component(hash.0.as_str())
}

fn validate_intent_key(intent_key: &str) -> bool {
    validate_key_component(intent_key)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), std::io::Error> {
    use std::io::Write;

    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "path must have parent")
    })?;
    std::fs::create_dir_all(parent)?;

    let mut tmp = path.to_path_buf();
    let suffix = format!("tmp.{}.{}", std::process::id(), unix_time_ms());
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "bad filename"))?;
    tmp.set_file_name(format!("{file_name}.{suffix}"));

    {
        let mut f = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }

    std::fs::rename(tmp, path)?;
    Ok(())
}

pub(crate) fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}
