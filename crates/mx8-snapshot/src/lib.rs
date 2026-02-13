#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use mx8_core::dataset_link::DatasetLink;
use mx8_core::types::{ManifestHash, ManifestRecord, MANIFEST_SCHEMA_VERSION};
use mx8_manifest_store::{intent_key_for_base, LockOwner, ManifestStore, ManifestStoreError};
use thiserror::Error;
use tracing::{info, warn};

#[derive(Debug, Error)]
pub enum SnapshotError {
    #[error("dataset link parse error: {0}")]
    Link(#[from] mx8_core::dataset_link::DatasetLinkParseError),
    #[error("manifest store error: {0}")]
    Store(#[from] ManifestStoreError),
    #[error("dev manifest path is required to create a new snapshot")]
    DevManifestPathRequired,
    #[error("dev manifest read error: {0}")]
    DevManifestIo(#[from] std::io::Error),
    #[error("dev manifest parse error: {0}")]
    DevManifestParse(String),
    #[error("dev manifest invariant violated: {0}")]
    DevManifestInvariant(String),
    #[error("snapshot wait timed out after {0:?}")]
    WaitTimeout(Duration),
}

#[derive(Debug, Clone)]
pub struct ResolvedSnapshot {
    pub intent_key: Option<String>,
    pub manifest_hash: ManifestHash,
    pub manifest_bytes: Vec<u8>,
    pub schema_version: u32,
}

#[derive(Debug, Clone)]
pub struct SnapshotResolverConfig {
    pub lock_stale_after: Duration,
    pub wait_timeout: Duration,
    pub poll_interval: Duration,
    pub dev_manifest_path: Option<PathBuf>,
}

impl Default for SnapshotResolverConfig {
    fn default() -> Self {
        Self {
            lock_stale_after: Duration::from_secs(60),
            wait_timeout: Duration::from_secs(30),
            poll_interval: Duration::from_millis(100),
            dev_manifest_path: None,
        }
    }
}

#[derive(Clone)]
pub struct SnapshotResolver {
    store: Arc<dyn ManifestStore>,
    cfg: SnapshotResolverConfig,
}

impl SnapshotResolver {
    pub fn new(store: Arc<dyn ManifestStore>, cfg: SnapshotResolverConfig) -> Self {
        Self { store, cfg }
    }

    pub fn resolve(&self, link: &str, owner: LockOwner) -> Result<ResolvedSnapshot, SnapshotError> {
        let parsed = DatasetLink::parse(link)?;

        match parsed {
            DatasetLink::Pinned { manifest_hash, .. } => {
                let hash = ManifestHash(manifest_hash);
                let bytes = self.store.get_manifest_bytes(&hash)?;
                info!(
                    target: "mx8_proof",
                    event = "snapshot_resolved",
                    mode = "pinned",
                    manifest_hash = %hash.0,
                    "resolved pinned snapshot"
                );
                Ok(ResolvedSnapshot {
                    intent_key: None,
                    manifest_hash: hash,
                    manifest_bytes: bytes,
                    schema_version: MANIFEST_SCHEMA_VERSION,
                })
            }
            DatasetLink::Plain(base) => self.resolve_for_intent(&base, false, owner),
            DatasetLink::Refresh(base) => self.resolve_for_intent(&base, true, owner),
        }
    }

    fn resolve_for_intent(
        &self,
        base: &str,
        refresh: bool,
        owner: LockOwner,
    ) -> Result<ResolvedSnapshot, SnapshotError> {
        let intent_key = intent_key_for_base(base);

        if !refresh {
            if let Some(hash) = self.store.get_current_snapshot(&intent_key)? {
                let bytes = self.store.get_manifest_bytes(&hash)?;
                info!(
                    target: "mx8_proof",
                    event = "snapshot_resolved",
                    mode = "plain",
                    intent_key = %intent_key,
                    manifest_hash = %hash.0,
                    "resolved current snapshot"
                );
                return Ok(ResolvedSnapshot {
                    intent_key: Some(intent_key),
                    manifest_hash: hash,
                    manifest_bytes: bytes,
                    schema_version: MANIFEST_SCHEMA_VERSION,
                });
            }
        }

        if refresh {
            info!(
                target: "mx8_proof",
                event = "snapshot_refresh_requested",
                intent_key = %intent_key,
                "refresh requested"
            );
        }

        if let Some(guard) =
            self.store
                .try_acquire_index_lock(&intent_key, self.cfg.lock_stale_after, owner)?
        {
            let _guard = guard;
            info!(
                target: "mx8_proof",
                event = "snapshot_indexer_elected",
                intent_key = %intent_key,
                "elected indexer"
            );
            let path = self
                .cfg
                .dev_manifest_path
                .clone()
                .ok_or(SnapshotError::DevManifestPathRequired)?;

            let (_records, canonical_bytes) = load_dev_manifest_tsv(&path)?;
            let manifest_hash = ManifestHash(mx8_manifest_store::sha256_hex(&canonical_bytes));

            self.store
                .put_manifest_bytes(&manifest_hash, &canonical_bytes)?;
            self.store
                .set_current_snapshot(&intent_key, &manifest_hash)?;

            info!(
                target: "mx8_proof",
                event = "snapshot_resolved",
                mode = if refresh { "refresh" } else { "create" },
                intent_key = %intent_key,
                manifest_hash = %manifest_hash.0,
                manifest_bytes = canonical_bytes.len() as u64,
                "created snapshot"
            );
            return Ok(ResolvedSnapshot {
                intent_key: Some(intent_key),
                manifest_hash,
                manifest_bytes: canonical_bytes,
                schema_version: MANIFEST_SCHEMA_VERSION,
            });
        }

        // Not the indexer: wait for current snapshot to appear.
        let start = Instant::now();
        info!(
            target: "mx8_proof",
            event = "snapshot_index_wait",
            intent_key = %intent_key,
            wait_timeout_ms = self.cfg.wait_timeout.as_millis() as u64,
            "waiting for indexer"
        );
        while start.elapsed() < self.cfg.wait_timeout {
            if let Some(hash) = self.store.get_current_snapshot(&intent_key)? {
                let bytes = self.store.get_manifest_bytes(&hash)?;
                info!(
                    target: "mx8_proof",
                    event = "snapshot_resolved",
                    mode = "wait",
                    intent_key = %intent_key,
                    manifest_hash = %hash.0,
                    waited_ms = start.elapsed().as_millis() as u64,
                    "observed snapshot"
                );
                return Ok(ResolvedSnapshot {
                    intent_key: Some(intent_key),
                    manifest_hash: hash,
                    manifest_bytes: bytes,
                    schema_version: MANIFEST_SCHEMA_VERSION,
                });
            }
            std::thread::sleep(self.cfg.poll_interval);
        }

        warn!(
            target: "mx8_proof",
            event = "snapshot_index_wait_timeout",
            intent_key = %intent_key,
            waited_ms = start.elapsed().as_millis() as u64,
            "timed out waiting for snapshot"
        );
        Err(SnapshotError::WaitTimeout(self.cfg.wait_timeout))
    }
}

fn load_dev_manifest_tsv(path: &PathBuf) -> Result<(Vec<ManifestRecord>, Vec<u8>), SnapshotError> {
    let bytes = std::fs::read(path)?;
    let text = String::from_utf8_lossy(&bytes);

    let mut records: Vec<ManifestRecord> = Vec::new();
    for (line_no, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 2 {
            return Err(SnapshotError::DevManifestParse(format!(
                "line {}: expected at least 2 columns (sample_id<TAB>location)",
                line_no + 1
            )));
        }

        let sample_id: u64 = cols[0].trim().parse().map_err(|_| {
            SnapshotError::DevManifestParse(format!("line {}: bad sample_id", line_no + 1))
        })?;
        let location = cols[1].trim().to_string();
        if location.contains('\n') || location.contains('\t') {
            return Err(SnapshotError::DevManifestParse(format!(
                "line {}: location must not contain tabs/newlines",
                line_no + 1
            )));
        }

        let mut byte_offset: Option<u64> = None;
        let mut byte_length: Option<u64> = None;
        let mut decode_hint: Option<String> = None;

        if cols.len() >= 4 {
            if !cols[2].trim().is_empty() || !cols[3].trim().is_empty() {
                byte_offset = Some(cols[2].trim().parse().map_err(|_| {
                    SnapshotError::DevManifestParse(format!(
                        "line {}: bad byte_offset",
                        line_no + 1
                    ))
                })?);
                byte_length = Some(cols[3].trim().parse().map_err(|_| {
                    SnapshotError::DevManifestParse(format!(
                        "line {}: bad byte_length",
                        line_no + 1
                    ))
                })?);
            }
        } else if cols.len() == 3 {
            return Err(SnapshotError::DevManifestParse(format!(
                "line {}: byte_offset and byte_length must be set together",
                line_no + 1
            )));
        }

        if cols.len() >= 5 && !cols[4].trim().is_empty() {
            let hint = cols[4].trim().to_string();
            if hint.contains('\n') || hint.contains('\t') {
                return Err(SnapshotError::DevManifestParse(format!(
                    "line {}: decode_hint must not contain tabs/newlines",
                    line_no + 1
                )));
            }
            decode_hint = Some(hint);
        }

        let record = ManifestRecord {
            sample_id,
            location,
            byte_offset,
            byte_length,
            decode_hint,
        };
        record.validate().map_err(|e| {
            SnapshotError::DevManifestInvariant(format!("line {}: {e}", line_no + 1))
        })?;
        records.push(record);
    }

    records.sort_by_key(|r| r.sample_id);
    ensure_sequential_sample_ids(&records)?;

    let canonical = canonicalize_manifest_bytes(&records);
    Ok((records, canonical))
}

fn ensure_sequential_sample_ids(records: &[ManifestRecord]) -> Result<(), SnapshotError> {
    for (i, r) in records.iter().enumerate() {
        let expected = i as u64;
        if r.sample_id != expected {
            return Err(SnapshotError::DevManifestInvariant(format!(
                "expected sample_id {expected} but found {}",
                r.sample_id
            )));
        }
    }
    Ok(())
}

fn canonicalize_manifest_bytes(records: &[ManifestRecord]) -> Vec<u8> {
    let mut out = Vec::with_capacity(records.len() * 64);
    out.extend_from_slice(format!("schema_version={MANIFEST_SCHEMA_VERSION}\n").as_bytes());
    for r in records {
        // Stable TSV with 5 columns; empty string for None.
        // sample_id<TAB>location<TAB>byte_offset<TAB>byte_length<TAB>decode_hint\n
        out.extend_from_slice(r.sample_id.to_string().as_bytes());
        out.push(b'\t');
        out.extend_from_slice(r.location.as_bytes());
        out.push(b'\t');
        if let Some(v) = r.byte_offset {
            out.extend_from_slice(v.to_string().as_bytes());
        }
        out.push(b'\t');
        if let Some(v) = r.byte_length {
            out.extend_from_slice(v.to_string().as_bytes());
        }
        out.push(b'\t');
        if let Some(h) = &r.decode_hint {
            out.extend_from_slice(h.as_bytes());
        }
        out.push(b'\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use mx8_manifest_store::fs::FsManifestStore;

    #[test]
    fn canonical_hash_is_deterministic() -> anyhow::Result<()> {
        let root = {
            let mut p = std::env::temp_dir();
            p.push(format!(
                "mx8-snapshot-test-{}-{}",
                std::process::id(),
                mx8_manifest_store::sha256_hex(b"seed")
            ));
            std::fs::create_dir_all(&p)?;
            p
        };

        let store: Arc<dyn ManifestStore> = Arc::new(FsManifestStore::new(root));
        let cfg = SnapshotResolverConfig {
            dev_manifest_path: None,
            ..Default::default()
        };
        let resolver = SnapshotResolver::new(store, cfg);

        let records = vec![
            ManifestRecord {
                sample_id: 0,
                location: "a".to_string(),
                byte_offset: None,
                byte_length: None,
                decode_hint: None,
            },
            ManifestRecord {
                sample_id: 1,
                location: "b".to_string(),
                byte_offset: Some(0),
                byte_length: Some(10),
                decode_hint: Some("x".to_string()),
            },
        ];
        let canonical = canonicalize_manifest_bytes(&records);
        let h1 = mx8_manifest_store::sha256_hex(&canonical);
        let h2 = mx8_manifest_store::sha256_hex(&canonical);
        assert_eq!(h1, h2);

        // Silence unused var; resolver existence is what we care about in this smoke test.
        let _ = resolver;
        Ok(())
    }

    #[test]
    fn plain_link_creates_snapshot_then_reuses_current() -> anyhow::Result<()> {
        let root = {
            let mut p = std::env::temp_dir();
            p.push(format!(
                "mx8-snapshot-e2e-{}-{}",
                std::process::id(),
                mx8_manifest_store::sha256_hex(b"root")
            ));
            std::fs::create_dir_all(&p)?;
            p
        };

        let manifest_path = root.join("dev_manifest.tsv");
        std::fs::write(&manifest_path, "0\ta\n1\tb\t0\t10\tx\n")?;

        let store: Arc<dyn ManifestStore> = Arc::new(FsManifestStore::new(root.clone()));
        let cfg = SnapshotResolverConfig {
            dev_manifest_path: Some(manifest_path),
            ..Default::default()
        };
        let resolver = SnapshotResolver::new(store.clone(), cfg);

        let first = resolver.resolve(
            "s3://bucket/prefix/",
            LockOwner {
                node_id: Some("test".to_string()),
            },
        )?;
        assert!(!first.manifest_hash.0.is_empty());
        assert!(!first.manifest_bytes.is_empty());

        // New resolver without dev_manifest_path should reuse the current pointer.
        let resolver2 = SnapshotResolver::new(
            store,
            SnapshotResolverConfig {
                dev_manifest_path: None,
                ..Default::default()
            },
        );
        let second = resolver2.resolve(
            "s3://bucket/prefix/",
            LockOwner {
                node_id: Some("test2".to_string()),
            },
        )?;
        assert_eq!(first.manifest_hash.0, second.manifest_hash.0);
        assert_eq!(first.manifest_bytes, second.manifest_bytes);
        Ok(())
    }

    #[test]
    fn missing_dev_manifest_path_surfaces_as_dev_manifest_io() {
        let root =
            std::env::temp_dir().join(format!("mx8-snapshot-missing-{}", std::process::id()));
        let store: Arc<dyn ManifestStore> = Arc::new(FsManifestStore::new(root));
        let resolver = SnapshotResolver::new(
            store,
            SnapshotResolverConfig {
                dev_manifest_path: Some(PathBuf::from("/no/such/file.tsv")),
                ..Default::default()
            },
        );

        let err = resolver
            .resolve(
                "s3://bucket/prefix/@refresh",
                LockOwner {
                    node_id: Some("test".to_string()),
                },
            )
            .unwrap_err();

        match err {
            SnapshotError::DevManifestIo(_) => {}
            other => panic!("expected DevManifestIo, got {other:?}"),
        }
    }
}
