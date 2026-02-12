use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JobId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LeaseId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ManifestHash(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkRange {
    pub start_id: u64,
    pub end_id: u64, // half-open [start_id, end_id)
    pub epoch: u32,
    pub seed: u64,
}

impl WorkRange {
    pub fn len(&self) -> u64 {
        self.end_id.saturating_sub(self.start_id)
    }

    pub fn is_empty(&self) -> bool {
        self.start_id >= self.end_id
    }

    pub fn contains(&self, sample_id: u64) -> bool {
        self.start_id <= sample_id && sample_id < self.end_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lease {
    pub lease_id: LeaseId,
    pub node_id: NodeId,
    pub range: WorkRange,
    pub cursor: u64,
    /// Lease expiration timestamp in Unix milliseconds.
    pub expires_unix_time_ms: u64,
}

/// Operator-configured per-node caps enforced by `mx8d-agent`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeCaps {
    pub max_fetch_concurrency: u32,
    pub max_decode_concurrency: u32,
    pub max_inflight_bytes: u64,
    pub max_ram_bytes: u64,
}

/// Periodic node stats reported to the coordinator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeStats {
    pub inflight_bytes: u64,
    pub ram_high_water_bytes: u64,
    pub fetch_queue_depth: u32,
    pub decode_queue_depth: u32,
    pub pack_queue_depth: u32,
}

/// Progress is reported in terms of the lease cursor.
///
/// v0 cursor semantics: cursor advances only after DELIVER (consumer receives the batch).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgressReport {
    pub job_id: JobId,
    pub node_id: NodeId,
    pub lease_id: LeaseId,
    pub cursor: u64,
    pub delivered_samples: u64,
    pub delivered_bytes: u64,
    pub unix_time_ms: u64,
}

/// v0 logical manifest schema version. Physical Parquet layout is defined later.
pub const MANIFEST_SCHEMA_VERSION: u32 = 0;

/// A single logical sample in the pinned snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestRecord {
    pub sample_id: u64,
    pub location: String,
    pub byte_offset: Option<u64>,
    pub byte_length: Option<u64>,
    pub decode_hint: Option<String>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ManifestRecordError {
    #[error("location must be non-empty")]
    EmptyLocation,
    #[error("byte_offset and byte_length must be set together (both Some or both None)")]
    PartialByteRange,
    #[error("byte_length must be > 0 when byte range is specified")]
    NonPositiveByteLength,
}

impl ManifestRecord {
    pub fn validate(&self) -> Result<(), ManifestRecordError> {
        if self.location.trim().is_empty() {
            return Err(ManifestRecordError::EmptyLocation);
        }

        match (self.byte_offset, self.byte_length) {
            (None, None) => Ok(()),
            (Some(_), Some(len)) => {
                if len == 0 {
                    return Err(ManifestRecordError::NonPositiveByteLength);
                }
                Ok(())
            }
            _ => Err(ManifestRecordError::PartialByteRange),
        }
    }
}
