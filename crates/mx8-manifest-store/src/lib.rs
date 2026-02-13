#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

pub mod fs;
#[cfg(feature = "s3")]
pub mod s3;
mod sha256;

use std::path::Path;
use std::time::Duration;
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
    #[error("unsupported manifest_store root: {0}")]
    UnsupportedRoot(String),
    #[error("runtime error: {0}")]
    Runtime(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Minimal interface for v0 snapshot storage.
///
/// - Manifests are immutable by `manifest_hash`.
/// - `intent_key` is the stable identifier for “what the user pointed at”.
pub trait ManifestStore: Send + Sync + 'static {
    fn try_acquire_index_lock(
        &self,
        intent_key: &str,
        stale_after: Duration,
        owner: LockOwner,
    ) -> Result<Option<Box<dyn IndexLockGuard>>, ManifestStoreError>;

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

#[derive(Debug, Clone)]
pub struct LockOwner {
    pub node_id: Option<String>,
}

pub trait IndexLockGuard: Send {}

pub fn intent_key_for_base(base: &str) -> String {
    let normalized = normalize_intent_base(base);
    let hash_hex = sha256_hex(normalized.as_bytes());
    format!("sha256_{hash_hex}")
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

pub fn sha256_hex(bytes: &[u8]) -> String {
    sha256::to_lower_hex(&sha256::sha256(bytes))
}

pub fn open_from_root(root: &str) -> Result<Box<dyn ManifestStore>, ManifestStoreError> {
    let trimmed = root.trim();
    if trimmed.is_empty() {
        return Err(ManifestStoreError::UnsupportedRoot(root.to_string()));
    }

    if let Some(rest) = trimmed.strip_prefix("s3://") {
        #[cfg(feature = "s3")]
        {
            return Ok(Box::new(s3::S3ManifestStore::from_env_url(rest)?));
        }
        #[cfg(not(feature = "s3"))]
        {
            let _ = rest;
            return Err(ManifestStoreError::UnsupportedRoot(format!(
                "s3://... requires feature 's3' (got {root:?})"
            )));
        }
    }

    Ok(Box::new(fs::FsManifestStore::new(trimmed)))
}

fn normalize_intent_base(base: &str) -> String {
    let trimmed = base.trim();
    trimmed.trim_end_matches('/').to_string()
}
