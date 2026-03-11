use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DurableNodeCaps {
    pub max_fetch_concurrency: u32,
    pub max_decode_concurrency: u32,
    pub max_inflight_bytes: u64,
    pub max_ram_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DurableNodeEntry {
    pub node_id: String,
    pub caps: Option<DurableNodeCaps>,
    pub last_heartbeat_unix_time_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DurableWorkRange {
    pub start_id: u64,
    pub end_id: u64,
    pub epoch: u32,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DurableLease {
    pub lease_id: String,
    pub node_id: String,
    pub range: DurableWorkRange,
    pub cursor: u64,
    pub expires_unix_time_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DurableProgress {
    pub node_id: String,
    pub lease_id: String,
    pub cursor: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DurableRangeKey {
    pub start_id: u64,
    pub end_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DurableCoordinatorMetrics {
    pub register_total: u64,
    pub heartbeat_total: u64,
    pub request_lease_total: u64,
    pub leases_granted_total: u64,
    pub leases_expired_total: u64,
    pub ranges_requeued_total: u64,
    pub progress_total: u64,
    pub resume_checkpoint_applied_total: u64,
    pub resume_checkpoint_rejected_total: u64,
    pub resume_ranges_applied_total: u64,
    pub active_leases: u64,
    pub registered_nodes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DurableCoordinatorSnapshot {
    pub schema_version: u32,
    pub manifest_hash: String,
    pub epoch: u32,
    pub write_term: u64,
    pub updated_unix_time_ms: u64,
    pub nodes: Vec<DurableNodeEntry>,
    pub available_ranges: Vec<DurableWorkRange>,
    pub rank_ranges: Option<Vec<Vec<DurableWorkRange>>>,
    pub frozen_membership: Option<Vec<String>>,
    pub departed_nodes: Vec<String>,
    pub leases: Vec<DurableLease>,
    pub progress: Vec<DurableProgress>,
    pub completed_ranges: Vec<DurableRangeKey>,
    pub resume_checkpoint_fingerprint: Option<u64>,
    pub next_lease_id: u64,
    pub drained_emitted: bool,
    pub metrics: DurableCoordinatorMetrics,
}

pub struct FileCoordinatorStore {
    path: PathBuf,
    write_lock: tokio::sync::Mutex<()>,
}

impl FileCoordinatorStore {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            write_lock: tokio::sync::Mutex::new(()),
        }
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    pub async fn load(&self) -> Result<Option<DurableCoordinatorSnapshot>> {
        if !self.path.exists() {
            return Ok(None);
        }
        let bytes = tokio::fs::read(&self.path)
            .await
            .with_context(|| format!("failed to read state store {}", self.path.display()))?;
        let snapshot: DurableCoordinatorSnapshot = serde_json::from_slice(&bytes)
            .with_context(|| format!("invalid state store json {}", self.path.display()))?;
        anyhow::ensure!(
            snapshot.schema_version == 1,
            "unsupported state store schema_version {}",
            snapshot.schema_version
        );
        Ok(Some(snapshot))
    }


    pub async fn save(&self, snapshot: &DurableCoordinatorSnapshot) -> Result<()> {
        let _guard = self.write_lock.lock().await;
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent).await?;
            }
        }

        let now_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let tmp = PathBuf::from(format!(
            "{}.{}.{}.tmp",
            self.path.display(),
            std::process::id(),
            now_nanos
        ));
        let bytes = serde_json::to_vec(snapshot)?;
        let mut file = tokio::fs::File::create(&tmp).await?;
        file.write_all(&bytes).await?;
        file.sync_data().await?;
        drop(file);
        tokio::fs::rename(&tmp, &self.path).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "mx8-coord-state-store-{}-{}-{}.json",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        ))
    }

    #[tokio::test]
    async fn save_then_load_roundtrip() -> Result<()> {
        let path = tmp_path("roundtrip");
        let _ = std::fs::remove_file(&path);
        let store = FileCoordinatorStore::new(path.clone());

        let snapshot = DurableCoordinatorSnapshot {
            schema_version: 1,
            manifest_hash: "abc".to_string(),
            epoch: 7,
            write_term: 3,
            updated_unix_time_ms: 42,
            nodes: vec![DurableNodeEntry {
                node_id: "node-a".to_string(),
                caps: Some(DurableNodeCaps {
                    max_fetch_concurrency: 8,
                    max_decode_concurrency: 4,
                    max_inflight_bytes: 1234,
                    max_ram_bytes: 5678,
                }),
                last_heartbeat_unix_time_ms: Some(99),
            }],
            available_ranges: vec![DurableWorkRange {
                start_id: 10,
                end_id: 20,
                epoch: 7,
                seed: 11,
            }],
            rank_ranges: Some(vec![vec![DurableWorkRange {
                start_id: 20,
                end_id: 30,
                epoch: 7,
                seed: 11,
            }]]),
            frozen_membership: Some(vec!["node-a".to_string()]),
            departed_nodes: vec![],
            leases: vec![DurableLease {
                lease_id: "lease-1".to_string(),
                node_id: "node-a".to_string(),
                range: DurableWorkRange {
                    start_id: 30,
                    end_id: 40,
                    epoch: 7,
                    seed: 11,
                },
                cursor: 35,
                expires_unix_time_ms: 123,
            }],
            progress: vec![DurableProgress {
                node_id: "node-a".to_string(),
                lease_id: "lease-1".to_string(),
                cursor: 35,
            }],
            completed_ranges: vec![DurableRangeKey {
                start_id: 0,
                end_id: 10,
            }],
            resume_checkpoint_fingerprint: Some(999),
            next_lease_id: 2,
            drained_emitted: false,
            metrics: DurableCoordinatorMetrics {
                register_total: 1,
                heartbeat_total: 2,
                request_lease_total: 3,
                leases_granted_total: 4,
                leases_expired_total: 5,
                ranges_requeued_total: 6,
                progress_total: 7,
                resume_checkpoint_applied_total: 8,
                resume_checkpoint_rejected_total: 9,
                resume_ranges_applied_total: 10,
                active_leases: 11,
                registered_nodes: 12,
            },
        };

        store.save(&snapshot).await?;
        let loaded = store
            .load()
            .await?
            .ok_or_else(|| anyhow::anyhow!("expected snapshot"))?;
        assert_eq!(loaded, snapshot);

        let _ = std::fs::remove_file(&path);
        Ok(())
    }
}
