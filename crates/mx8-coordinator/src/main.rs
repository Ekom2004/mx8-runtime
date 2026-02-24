#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

mod lease_log;

use std::collections::HashSet;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use clap::Parser;
use tokio::sync::{mpsc, Mutex, RwLock};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::Stream;
use tonic::{transport::Server, Request, Response, Status};
use tracing::{info, info_span, Instrument};

use mx8_core::types::{ManifestHash, MANIFEST_SCHEMA_VERSION};
use mx8_manifest_store::ManifestStore;
use mx8_observe::metrics::{Counter, Gauge};
use mx8_proto::v0::coordinator_server::{Coordinator, CoordinatorServer};
use mx8_proto::v0::*;
use mx8_snapshot::{SnapshotResolver, SnapshotResolverConfig};

const DEFAULT_GRPC_MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;
const MANIFEST_CHUNK_BYTES: usize = 1024 * 1024;

#[derive(Debug, Parser)]
#[command(name = "mx8-coordinator")]
struct Args {
    /// Address to bind the coordinator gRPC server.
    #[arg(long, env = "MX8_COORD_BIND_ADDR", default_value = "0.0.0.0:50051")]
    addr: SocketAddr,

    /// Expected number of nodes in this job (membership barrier).
    #[arg(long, env = "MX8_WORLD_SIZE", default_value_t = 1)]
    world_size: u32,

    /// Heartbeat interval returned to agents.
    #[arg(long, env = "MX8_HEARTBEAT_INTERVAL_MS", default_value_t = 1000)]
    heartbeat_interval_ms: u32,

    /// Lease TTL returned to agents.
    #[arg(long, env = "MX8_LEASE_TTL_MS", default_value_t = 10_000)]
    lease_ttl_ms: u32,

    /// Dataset link (plain / @refresh / @sha256:...).
    ///
    /// If set, the coordinator resolves a pinned snapshot using the manifest store.
    #[arg(long, env = "MX8_DATASET_LINK")]
    dataset_link: Option<String>,

    /// Root directory for the FS manifest_store.
    /// Defaults to `~/.mx8/manifests` when not set.
    #[arg(long, env = "MX8_MANIFEST_STORE_ROOT")]
    manifest_store_root: Option<String>,

    /// Development-only: path to a TSV manifest used to create a snapshot when needed.
    ///
    /// Format:
    ///   sample_id<TAB>location[<TAB>byte_offset<TAB>byte_length[<TAB>decode_hint]]
    #[arg(long, env = "MX8_DEV_MANIFEST_PATH")]
    dev_manifest_path: Option<PathBuf>,

    /// How long a stale intent lock is tolerated before reaping (FS store).
    #[arg(long, env = "MX8_SNAPSHOT_LOCK_STALE_MS", default_value_t = 60_000)]
    snapshot_lock_stale_ms: u64,

    /// Max time to wait for another indexer to publish the current snapshot pointer.
    #[arg(long, env = "MX8_SNAPSHOT_WAIT_TIMEOUT_MS", default_value_t = 30_000)]
    snapshot_wait_timeout_ms: u64,

    /// Legacy: pinned manifest hash for this job.
    ///
    /// Prefer `--dataset-link`; this is kept for compatibility during M3 integration.
    #[arg(long, env = "MX8_MANIFEST_HASH", default_value = "dev")]
    manifest_hash: String,

    /// Development-only: total number of samples to serve as contiguous WorkRanges.
    ///
    /// If this is 0, the coordinator will try to derive N from the pinned manifest bytes
    /// (canonical TSV v0) and issue ranges over `[0..N)`.
    #[arg(long, env = "MX8_DEV_TOTAL_SAMPLES", default_value_t = 0)]
    dev_total_samples: u64,

    /// Development-only: block size (samples) for lease WorkRanges.
    #[arg(long, env = "MX8_DEV_BLOCK_SIZE", default_value_t = 65_536)]
    dev_block_size: u64,

    /// Shuffle block order deterministically for this run.
    #[arg(
        long,
        env = "MX8_SHUFFLE",
        default_value_t = false,
        value_parser = clap::builder::BoolishValueParser::new()
    )]
    shuffle: bool,

    /// Shuffle seed for deterministic block order.
    #[arg(long, env = "MX8_SEED", default_value_t = 0)]
    seed: u64,

    /// Epoch for deterministic block order (changes the permutation).
    #[arg(long, env = "MX8_EPOCH", default_value_t = 0)]
    epoch: u32,

    /// Path for the lease write-ahead log (crash recovery).
    ///
    /// On restart the coordinator replays this file to skip already-completed ranges.
    /// Defaults to `<manifest_store_root>/../lease_logs/<manifest_hash>.log`.
    /// Set to an empty string or `none` to disable.
    #[arg(long, env = "MX8_LEASE_LOG_PATH")]
    lease_log_path: Option<String>,

    /// Minimum number of nodes required before the job starts issuing leases.
    ///
    /// Defaults to `world_size` (all nodes must register before work begins).
    /// Set lower to allow the job to start under a partial cluster and accept
    /// replacement nodes for any that later depart.
    #[arg(long, env = "MX8_MIN_WORLD_SIZE", default_value_t = 0)]
    min_world_size: u32,

    /// Optional: periodically emit a metrics snapshot to logs.
    #[arg(long, env = "MX8_METRICS_SNAPSHOT_INTERVAL_MS", default_value_t = 0)]
    metrics_snapshot_interval_ms: u64,

    /// gRPC max message size (both decode/encode) for manifest proxying and future APIs.
    #[arg(
        long,
        env = "MX8_GRPC_MAX_MESSAGE_BYTES",
        default_value_t = DEFAULT_GRPC_MAX_MESSAGE_BYTES
    )]
    grpc_max_message_bytes: usize,
}

fn count_samples_in_canonical_manifest_tsv(bytes: &[u8]) -> anyhow::Result<u64> {
    let s = std::str::from_utf8(bytes).map_err(|e| anyhow::anyhow!("manifest not utf-8: {e}"))?;
    let mut lines = s.lines();
    let first = lines
        .by_ref()
        .find(|l| !l.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("empty manifest"))?;
    anyhow::ensure!(
        first.trim_start().starts_with("schema_version="),
        "manifest header missing schema_version"
    );

    let mut n: u64 = 0;
    for raw in lines {
        if raw.trim().is_empty() {
            continue;
        }
        n = n
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("manifest row count overflow"))?;
    }
    Ok(n)
}

#[derive(Debug, Clone, Default)]
struct NodeEntry {
    caps: Option<NodeCaps>,
    last_heartbeat_unix_time_ms: Option<u64>,
    last_stats: Option<NodeStats>,
}

#[derive(Debug, Default)]
struct CoordinatorState {
    nodes: std::collections::BTreeMap<String, NodeEntry>,
    /// Global ranges available to any node (used for requeued remainders after failures).
    available_ranges: std::collections::VecDeque<WorkRange>,
    /// Deterministic per-rank schedule (failure-free determinism).
    ///
    /// Once membership is frozen at `min_world_size`, initial work is partitioned into these
    /// queues. Nodes draw from their own queue first; if empty, they may steal from others to
    /// complete the job under failure.
    rank_ranges: Option<Vec<std::collections::VecDeque<WorkRange>>>,
    /// Frozen membership ordering used for deterministic rank assignment (rank = Vec index).
    /// Set once when `min_world_size` nodes have registered; never grows or shrinks, but entries
    /// are swapped when a replacement node fills a departed rank slot.
    frozen_membership: Option<Vec<String>>,
    /// Node IDs that have timed out (heartbeat expired after freeze).
    /// Their rank slots are vacant and available for replacement nodes.
    departed_nodes: std::collections::BTreeSet<String>,
    leases: std::collections::BTreeMap<String, Lease>,
    progress: std::collections::BTreeMap<(String, String), u64>,
    next_lease_id: u64,
    drained_emitted: bool,
}

#[derive(Debug, Default)]
struct CoordinatorMetrics {
    register_total: Counter,
    heartbeat_total: Counter,
    request_lease_total: Counter,
    leases_granted_total: Counter,
    leases_expired_total: Counter,
    ranges_requeued_total: Counter,
    progress_total: Counter,
    active_leases: Gauge,
    registered_nodes: Gauge,
}

#[derive(Clone)]
struct CoordinatorSvc {
    state: Arc<RwLock<CoordinatorState>>,
    metrics: Arc<CoordinatorMetrics>,
    /// Configured maximum number of nodes (controls world_size advertised to agents and the
    /// capacity cap before freeze).
    world_size: u32,
    /// Minimum nodes required before the job starts issuing leases (barrier gate).
    /// Always <= world_size.
    min_world_size: u32,
    heartbeat_interval_ms: u32,
    lease_ttl_ms: u32,
    shuffle: bool,
    seed: u64,
    epoch: u32,
    manifest_hash: String,
    manifest_store: Option<Arc<dyn ManifestStore>>,
    /// Write-ahead log for completed lease ranges; `None` in dev/test mode.
    lease_log: Option<Arc<Mutex<lease_log::LeaseLog>>>,
}

fn existing_caps_changed(existing: &Option<NodeCaps>, new: &Option<NodeCaps>) -> bool {
    existing != new
}

impl CoordinatorSvc {
    fn validate_id(field: &'static str, value: &str) -> Option<Status> {
        if value.trim().is_empty() {
            return Some(Status::invalid_argument(format!(
                "{field} must be non-empty"
            )));
        }
        None
    }

    async fn ensure_registered(&self, node_id: &str) -> Option<Status> {
        let state = self.state.read().await;
        if !state.nodes.contains_key(node_id) {
            return Some(Status::failed_precondition(
                "node must RegisterNode before using coordinator APIs",
            ));
        }
        None
    }

    async fn is_job_ready(&self) -> bool {
        let state = self.state.read().await;
        // The job is ready as soon as the membership has been frozen (partition done).
        state.frozen_membership.is_some()
    }

    async fn build_snapshot(&self) -> GetJobSnapshotResponse {
        let state = self.state.read().await;
        let membership = state.frozen_membership.clone();
        let nodes = state
            .nodes
            .iter()
            .map(|(node_id, entry)| {
                let assigned_rank = membership
                    .as_ref()
                    .and_then(|m| m.iter().position(|id| id == node_id))
                    .or_else(|| state.nodes.keys().position(|id| id == node_id))
                    .unwrap_or(0) as u32;
                NodeSnapshot {
                    node_id: node_id.clone(),
                    assigned_rank,
                    last_heartbeat_unix_time_ms: entry.last_heartbeat_unix_time_ms.unwrap_or(0),
                    caps: entry.caps.clone(),
                    stats: entry.last_stats.clone(),
                }
            })
            .collect::<Vec<_>>();
        let live_leases = state.leases.values().cloned().collect::<Vec<_>>();
        let counters = Some(CoordinatorCounters {
            register_total: self.metrics.register_total.get(),
            heartbeat_total: self.metrics.heartbeat_total.get(),
            request_lease_total: self.metrics.request_lease_total.get(),
            leases_granted_total: self.metrics.leases_granted_total.get(),
            leases_expired_total: self.metrics.leases_expired_total.get(),
            ranges_requeued_total: self.metrics.ranges_requeued_total.get(),
            progress_total: self.metrics.progress_total.get(),
        });
        let capacity = state
            .frozen_membership
            .as_ref()
            .map(|m| m.len() as u32)
            .unwrap_or(self.world_size);
        let active_nodes = (state.nodes.len().saturating_sub(state.departed_nodes.len())) as u32;
        GetJobSnapshotResponse {
            server_unix_time_ms: Self::unix_time_ms(),
            manifest_hash: self.manifest_hash.clone(),
            world_size: capacity,
            registered_nodes: active_nodes,
            job_ready: state.frozen_membership.is_some(),
            job_drained: state.drained_emitted,
            active_leases: state.leases.len() as u64,
            available_ranges: state.available_ranges.len() as u64,
            nodes,
            live_leases,
            counters,
        }
    }

    fn mix64(mut x: u64) -> u64 {
        // Deterministic, stable mixing for shuffle keys (splitmix64 variant).
        x = x.wrapping_add(0x9E3779B97F4A7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
        x ^ (x >> 31)
    }

    fn shuffle_key(seed: u64, epoch: u32, start_id: u64) -> u64 {
        let epoch_u64 = epoch as u64;
        Self::mix64(seed ^ (epoch_u64.wrapping_mul(0x9E3779B97F4A7C15)) ^ start_id)
    }

    #[allow(clippy::result_large_err)]
    fn freeze_membership_and_partition_ranges(
        state: &mut CoordinatorState,
        min_world_size: u32,
        shuffle: bool,
        seed: u64,
        epoch: u32,
    ) -> Result<(), Status> {
        if state.frozen_membership.is_some() {
            return Ok(());
        }
        if (state.nodes.len() as u32) < min_world_size {
            return Err(Status::failed_precondition(
                "cannot freeze membership before min_world_size barrier is met",
            ));
        }

        let membership: Vec<String> = state.nodes.keys().cloned().collect();
        state.frozen_membership = Some(membership);

        let mut ranges: Vec<WorkRange> = Vec::new();
        while let Some(mut r) = state.available_ranges.pop_front() {
            r.seed = seed;
            r.epoch = epoch;
            ranges.push(r);
        }

        if shuffle {
            ranges.sort_by(|a, b| {
                let ka = Self::shuffle_key(seed, epoch, a.start_id);
                let kb = Self::shuffle_key(seed, epoch, b.start_id);
                ka.cmp(&kb).then(a.start_id.cmp(&b.start_id))
            });
        }

        let active_count = state.nodes.len();
        let mut rank_ranges: Vec<std::collections::VecDeque<WorkRange>> = (0..active_count)
            .map(|_| std::collections::VecDeque::new())
            .collect();

        for (i, r) in ranges.into_iter().enumerate() {
            let rank = (i as u32) % (active_count as u32);
            rank_ranges[rank as usize].push_back(r);
        }
        state.rank_ranges = Some(rank_ranges);
        Ok(())
    }

    #[allow(clippy::result_large_err)]
    fn repartition_unissued_ranges(
        state: &mut CoordinatorState,
        shuffle: bool,
        seed: u64,
        epoch: u32,
    ) -> Result<(), Status> {
        let members = state.frozen_membership.as_ref().ok_or_else(|| {
            Status::failed_precondition("cannot repartition before membership is frozen")
        })?;
        let active_count = members.len();
        if active_count == 0 {
            return Err(Status::failed_precondition(
                "cannot repartition with empty membership",
            ));
        }

        let mut ranges: Vec<WorkRange> = Vec::new();
        while let Some(mut r) = state.available_ranges.pop_front() {
            r.seed = seed;
            r.epoch = epoch;
            ranges.push(r);
        }
        if let Some(rank_ranges) = state.rank_ranges.as_mut() {
            for q in rank_ranges.iter_mut() {
                while let Some(mut r) = q.pop_front() {
                    r.seed = seed;
                    r.epoch = epoch;
                    ranges.push(r);
                }
            }
        }

        if shuffle {
            ranges.sort_by(|a, b| {
                let ka = Self::shuffle_key(seed, epoch, a.start_id);
                let kb = Self::shuffle_key(seed, epoch, b.start_id);
                ka.cmp(&kb).then(a.start_id.cmp(&b.start_id))
            });
        } else {
            ranges.sort_by_key(|r| r.start_id);
        }

        let mut rank_ranges: Vec<std::collections::VecDeque<WorkRange>> = (0..active_count)
            .map(|_| std::collections::VecDeque::new())
            .collect();
        for (i, r) in ranges.into_iter().enumerate() {
            let rank = i % active_count;
            rank_ranges[rank].push_back(r);
        }
        state.rank_ranges = Some(rank_ranges);
        Ok(())
    }

    fn rank_for_node(state: &CoordinatorState, node_id: &str) -> Option<u32> {
        let members = state.frozen_membership.as_ref()?;
        members
            .iter()
            .position(|id| id == node_id)
            .map(|i| i as u32)
    }

    fn pop_next_range_for_node(state: &mut CoordinatorState, node_id: &str) -> Option<WorkRange> {
        // 1) Requeued remainders are global priority.
        if let Some(r) = state.available_ranges.pop_front() {
            return Some(r);
        }

        let rank = Self::rank_for_node(state, node_id)?;
        let rank_ranges = state.rank_ranges.as_mut()?;

        // 2) Failure-free deterministic schedule: take from own queue.
        if let Some(r) = rank_ranges.get_mut(rank as usize)?.pop_front() {
            return Some(r);
        }

        None
    }

    fn unix_time_ms() -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        now.as_millis() as u64
    }

    async fn update_gauges(&self) {
        let state = self.state.read().await;
        self.metrics.active_leases.set(state.leases.len() as u64);
        self.metrics.registered_nodes.set(state.nodes.len() as u64);
    }

    async fn emit_metrics_snapshot(&self) {
        self.update_gauges().await;
        tracing::info!(
            target: "mx8_metrics",
            register_total = self.metrics.register_total.get(),
            heartbeat_total = self.metrics.heartbeat_total.get(),
            request_lease_total = self.metrics.request_lease_total.get(),
            leases_granted_total = self.metrics.leases_granted_total.get(),
            leases_expired_total = self.metrics.leases_expired_total.get(),
            ranges_requeued_total = self.metrics.ranges_requeued_total.get(),
            progress_total = self.metrics.progress_total.get(),
            active_leases = self.metrics.active_leases.get(),
            registered_nodes = self.metrics.registered_nodes.get(),
            world_size = self.world_size,
            manifest_hash = %self.manifest_hash,
            "metrics"
        );
    }

    async fn tick_once_at(&self, now_unix_time_ms: u64) {
        let mut expired: Vec<(Lease, u64)> = Vec::new();
        let mut requeued: Vec<(String, String, WorkRange, u64)> = Vec::new();
        let mut released: Vec<(String, u32, u64)> = Vec::new();
        let mut intervals_snapshot: Option<String> = None;
        let live_lease_count: usize;
        let mut drained_event: Option<(u64, u32)> = None;

        {
            let mut state = self.state.write().await;
            let mut expired_ids: Vec<String> = Vec::new();
            for (lease_id, lease) in &state.leases {
                if lease.expires_unix_time_ms <= now_unix_time_ms {
                    expired_ids.push(lease_id.clone());
                }
            }

            for lease_id in expired_ids {
                if let Some(lease) = state.leases.remove(&lease_id) {
                    let cursor = lease.cursor;
                    expired.push((lease, cursor));
                    state.progress.retain(|(_, id), _| id != &lease_id);
                }
            }

            for (lease, cursor) in &expired {
                let Some(range) = &lease.range else {
                    continue;
                };
                let cursor = *cursor;
                let remainder_start = cursor.clamp(range.start_id, range.end_id);
                if remainder_start >= range.end_id {
                    continue;
                }

                let remainder = WorkRange {
                    start_id: remainder_start,
                    end_id: range.end_id,
                    epoch: range.epoch,
                    seed: range.seed,
                };

                state.available_ranges.push_front(remainder.clone());
                requeued.push((
                    lease.lease_id.clone(),
                    lease.node_id.clone(),
                    remainder,
                    cursor,
                ));
            }

            // If a node is dead (no heartbeats) and not yet departed, release its remaining
            // scheduled ranges so the job can drain or be picked up by a replacement.
            // Skipping already-departed nodes avoids double-releasing and log spam.
            let dead_after_ms = self.lease_ttl_ms as u64;
            let newly_dead: Vec<(usize, String)> = match state.frozen_membership.as_ref() {
                None => Vec::new(),
                Some(members) => members
                    .iter()
                    .enumerate()
                    .filter_map(|(rank, node_id)| {
                        if state.departed_nodes.contains(node_id) {
                            return None; // already handled
                        }
                        let entry = state.nodes.get(node_id)?;
                        let last = entry.last_heartbeat_unix_time_ms?;
                        if now_unix_time_ms.saturating_sub(last) > dead_after_ms {
                            Some((rank, node_id.clone()))
                        } else {
                            None
                        }
                    })
                    .collect(),
            };

            if let Some(mut rank_ranges) = state.rank_ranges.take() {
                for (rank, node_id) in &newly_dead {
                    let q = &mut rank_ranges[*rank];
                    let mut count = 0u64;
                    while let Some(r) = q.pop_front() {
                        state.available_ranges.push_back(r);
                        count = count.saturating_add(1);
                    }
                    if count > 0 {
                        released.push((node_id.clone(), *rank as u32, count));
                    }
                    state.departed_nodes.insert(node_id.clone());
                }
                state.rank_ranges = Some(rank_ranges);
            }

            let mut intervals: Vec<(u64, u64, String, String)> = Vec::new();
            for lease in state.leases.values() {
                let Some(range) = &lease.range else {
                    continue;
                };
                intervals.push((
                    range.start_id,
                    range.end_id,
                    lease.lease_id.clone(),
                    lease.node_id.clone(),
                ));
            }
            intervals.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
            for w in intervals.windows(2) {
                let a = &w[0];
                let b = &w[1];
                if a.1 > b.0 {
                    tracing::error!(
                        target: "mx8_proof",
                        event = "overlap_detected",
                        manifest_hash = %self.manifest_hash,
                        a_lease_id = %a.2,
                        a_node_id = %a.3,
                        a_start_id = a.0,
                        a_end_id = a.1,
                        b_lease_id = %b.2,
                        b_node_id = %b.3,
                        b_start_id = b.0,
                        b_end_id = b.1,
                        "overlapping live lease ranges detected"
                    );
                    panic!("overlapping live lease ranges detected");
                }
            }

            live_lease_count = state.leases.len();
            if !intervals.is_empty() {
                let mut s = String::new();
                let max = intervals.len().min(16);
                for (i, (start, end, lease_id, node_id)) in
                    intervals.into_iter().take(max).enumerate()
                {
                    if i != 0 {
                        s.push_str(", ");
                    }
                    s.push_str(&format!("{start}-{end}:{lease_id}@{node_id}"));
                }
                if max < state.leases.len() {
                    s.push_str(", ...");
                }
                intervals_snapshot = Some(s);
            }

            let job_ready = state.nodes.len() as u32 >= self.world_size;
            let rank_ranges_empty = match &state.rank_ranges {
                None => true,
                Some(qs) => qs.iter().all(|q| q.is_empty()),
            };
            if job_ready
                && !state.drained_emitted
                && state.leases.is_empty()
                && state.available_ranges.is_empty()
                && rank_ranges_empty
            {
                state.drained_emitted = true;
                drained_event = Some((now_unix_time_ms, state.nodes.len() as u32));
            }
        }

        if !expired.is_empty() {
            self.metrics
                .leases_expired_total
                .inc_by(expired.len() as u64);
        }
        if !requeued.is_empty() {
            self.metrics
                .ranges_requeued_total
                .inc_by(requeued.len() as u64);
        }

        for (node_id, rank, count) in released {
            tracing::info!(
                target: "mx8_proof",
                event = "rank_released",
                node_id = %node_id,
                rank = rank,
                released_ranges = count,
                manifest_hash = %self.manifest_hash,
                "released ranges from dead rank"
            );
        }

        for (lease, cursor) in expired {
            if let Some(range) = &lease.range {
                tracing::info!(
                    target: "mx8_proof",
                    event = "lease_expired",
                    lease_id = %lease.lease_id,
                    node_id = %lease.node_id,
                    manifest_hash = %self.manifest_hash,
                    epoch = range.epoch,
                    start_id = range.start_id,
                    end_id = range.end_id,
                    cursor = cursor,
                    expires_unix_time_ms = lease.expires_unix_time_ms,
                    now_unix_time_ms = now_unix_time_ms,
                    "lease expired"
                );
            }
        }

        for (lease_id, node_id, remainder, cursor) in requeued {
            tracing::info!(
                target: "mx8_proof",
                event = "range_requeued",
                lease_id = %lease_id,
                node_id = %node_id,
                manifest_hash = %self.manifest_hash,
                epoch = remainder.epoch,
                start_id = remainder.start_id,
                end_id = remainder.end_id,
                cursor = cursor,
                "requeued remainder"
            );
        }

        tracing::info!(
            target: "mx8_proof",
            event = "no_overlap_ok",
            manifest_hash = %self.manifest_hash,
            live_leases = live_lease_count as u64,
            "no overlapping live leases"
        );

        if let Some(snapshot) = intervals_snapshot {
            tracing::info!(
                target: "mx8_proof",
                event = "live_leases",
                manifest_hash = %self.manifest_hash,
                intervals = %snapshot,
                "live lease intervals"
            );
        }

        if let Some((now, registered_nodes)) = drained_event {
            tracing::info!(
                target: "mx8_proof",
                event = "job_drained",
                manifest_hash = %self.manifest_hash,
                registered_nodes = registered_nodes,
                unix_time_ms = now,
                "job drained"
            );
        }

        self.update_gauges().await;
    }

    async fn tick_once(&self) {
        let now_unix_time_ms = Self::unix_time_ms();
        self.tick_once_at(now_unix_time_ms).await;
    }
}

#[tonic::async_trait]
impl Coordinator for CoordinatorSvc {
    type GetManifestStreamStream =
        Pin<Box<dyn Stream<Item = Result<ManifestChunk, Status>> + Send + 'static>>;

    async fn register_node(
        &self,
        request: Request<RegisterNodeRequest>,
    ) -> Result<Response<RegisterNodeResponse>, Status> {
        self.metrics.register_total.inc();
        let req = request.into_inner();
        if let Some(status) = Self::validate_id("job_id", &req.job_id) {
            return Err(status);
        }
        if let Some(status) = Self::validate_id("node_id", &req.node_id) {
            return Err(status);
        }

        if req.caps.is_none() {
            return Err(Status::invalid_argument("caps is required"));
        }

        let (assigned_rank, active_nodes, capacity, became_ready) = {
            let mut state = self.state.write().await;
            let now_ms = Self::unix_time_ms();

            if state.frozen_membership.is_some() {
                // ── Post-freeze path ──────────────────────────────────────────────────────
                if state.nodes.contains_key(&req.node_id) {
                    // Re-registration of a known node (e.g., brief disconnect and reconnect).
                    // Un-depart it if it was marked as departed.
                    let was_departed = state.departed_nodes.remove(&req.node_id);
                    if let Some(entry) = state.nodes.get_mut(&req.node_id) {
                        if existing_caps_changed(&entry.caps, &req.caps) {
                            tracing::warn!(
                                node_id = %req.node_id,
                                "RegisterNode updated caps for re-registering node"
                            );
                        }
                        entry.caps = req.caps.clone();
                        entry.last_heartbeat_unix_time_ms = Some(now_ms);
                        entry.last_stats = None;
                    }
                    if was_departed {
                        tracing::info!(
                            node_id = %req.node_id,
                            manifest_hash = %self.manifest_hash,
                            "departed node re-registered; slot reclaimed"
                        );
                    }
                } else {
                    // New node ID — first try to fill a vacant (departed) rank slot.
                    let departed_id = state.departed_nodes.iter().next().cloned();
                    if let Some(old_id) = departed_id {
                        // Swap new node_id into the slot previously held by old_id.
                        if let Some(membership) = state.frozen_membership.as_mut() {
                            if let Some(slot) = membership.iter_mut().find(|id| *id == &old_id) {
                                *slot = req.node_id.clone();
                            }
                        }
                        state.departed_nodes.remove(&old_id);
                        state.nodes.remove(&old_id);
                        state.nodes.insert(
                            req.node_id.clone(),
                            NodeEntry {
                                caps: req.caps.clone(),
                                last_heartbeat_unix_time_ms: Some(now_ms),
                                last_stats: None,
                            },
                        );
                        tracing::info!(
                            node_id = %req.node_id,
                            replaced_node_id = %old_id,
                            manifest_hash = %self.manifest_hash,
                            "replacement node filled departed rank slot"
                        );
                    } else {
                        // No departed slot: allow true scale-out until world_size capacity.
                        let current_capacity = state
                            .frozen_membership
                            .as_ref()
                            .map(|m| m.len() as u32)
                            .ok_or_else(|| {
                                Status::internal(
                                    "frozen membership missing in post-freeze register path",
                                )
                            })?;
                        if current_capacity >= self.world_size {
                            return Err(Status::failed_precondition(
                                "membership is frozen and at world_size capacity; \
                                 no vacant rank slots are available",
                            ));
                        }
                        if let Some(membership) = state.frozen_membership.as_mut() {
                            membership.push(req.node_id.clone());
                        } else {
                            return Err(Status::internal(
                                "frozen membership missing in post-freeze register path",
                            ));
                        }
                        state.nodes.insert(
                            req.node_id.clone(),
                            NodeEntry {
                                caps: req.caps.clone(),
                                last_heartbeat_unix_time_ms: Some(now_ms),
                                last_stats: None,
                            },
                        );
                        // Repartition all unissued ranges so the new rank can immediately consume.
                        Self::repartition_unissued_ranges(
                            &mut state,
                            self.shuffle,
                            self.seed,
                            self.epoch,
                        )?;
                        tracing::info!(
                            node_id = %req.node_id,
                            manifest_hash = %self.manifest_hash,
                            new_capacity = current_capacity + 1,
                            "scale-out node admitted into new rank slot"
                        );
                    }
                }
            } else {
                // ── Pre-freeze path ───────────────────────────────────────────────────────
                // Cap at world_size before the barrier fires.
                if state.nodes.len() as u32 >= self.world_size
                    && !state.nodes.contains_key(&req.node_id)
                {
                    return Err(Status::resource_exhausted(
                        "world_size capacity reached; wait for the job to start \
                         or increase --world-size",
                    ));
                }
                if let Some(existing) = state.nodes.get_mut(&req.node_id) {
                    if existing_caps_changed(&existing.caps, &req.caps) {
                        tracing::warn!("RegisterNode updated caps for existing node_id");
                    }
                    existing.caps = req.caps.clone();
                    existing.last_heartbeat_unix_time_ms = Some(now_ms);
                    existing.last_stats = None;
                } else {
                    state.nodes.insert(
                        req.node_id.clone(),
                        NodeEntry {
                            caps: req.caps.clone(),
                            last_heartbeat_unix_time_ms: Some(now_ms),
                            last_stats: None,
                        },
                    );
                }
            }

            // Derive rank: position in frozen_membership when available, BTreeMap order before.
            let assigned_rank = if let Some(membership) = state.frozen_membership.as_ref() {
                membership
                    .iter()
                    .position(|id| id == &req.node_id)
                    .ok_or_else(|| {
                        Status::internal("registered node not found in frozen membership")
                    })? as u32
            } else {
                state
                    .nodes
                    .keys()
                    .position(|id| id == &req.node_id)
                    .ok_or_else(|| Status::internal("registered node missing from state"))?
                    as u32
            };

            let capacity = state
                .frozen_membership
                .as_ref()
                .map(|m| m.len() as u32)
                .unwrap_or(self.world_size);

            let active_nodes =
                (state.nodes.len().saturating_sub(state.departed_nodes.len())) as u32;

            // Trigger freeze when we cross the min_world_size barrier for the first time.
            let became_ready =
                state.frozen_membership.is_none() && active_nodes >= self.min_world_size;

            (assigned_rank, active_nodes, capacity, became_ready)
        };

        if assigned_rank >= capacity {
            return Err(Status::failed_precondition(
                "node registered but rank exceeds capacity",
            ));
        }

        let resp = RegisterNodeResponse {
            assigned_rank,
            world_size: capacity,
            manifest_hash: self.manifest_hash.clone(),
            heartbeat_interval_ms: self.heartbeat_interval_ms,
            lease_ttl_ms: self.lease_ttl_ms,
            registered_nodes: active_nodes,
            job_ready: self.is_job_ready().await || became_ready,
        };

        if became_ready {
            let mut state = self.state.write().await;
            Self::freeze_membership_and_partition_ranges(
                &mut state,
                self.min_world_size,
                self.shuffle,
                self.seed,
                self.epoch,
            )?;
        }

        tracing::info!(
            job_id = %req.job_id,
            node_id = %req.node_id,
            manifest_hash = %resp.manifest_hash,
            assigned_rank = resp.assigned_rank,
            world_size = resp.world_size,
            registered_nodes = resp.registered_nodes,
            job_ready = resp.job_ready,
            "node registered"
        );

        self.update_gauges().await;
        Ok(Response::new(resp))
    }

    async fn heartbeat(
        &self,
        request: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        self.metrics.heartbeat_total.inc();
        let req = request.into_inner();
        if let Some(status) = Self::validate_id("job_id", &req.job_id) {
            return Err(status);
        }
        if let Some(status) = Self::validate_id("node_id", &req.node_id) {
            return Err(status);
        }
        if let Some(status) = self.ensure_registered(&req.node_id).await {
            return Err(status);
        }
        {
            let mut state = self.state.write().await;
            if let Some(entry) = state.nodes.get_mut(&req.node_id) {
                entry.last_heartbeat_unix_time_ms = Some(req.unix_time_ms);
                entry.last_stats = req.stats.clone();
            }
        }
        Ok(Response::new(HeartbeatResponse {}))
    }

    async fn request_lease(
        &self,
        request: Request<RequestLeaseRequest>,
    ) -> Result<Response<RequestLeaseResponse>, Status> {
        self.metrics.request_lease_total.inc();
        let req = request.into_inner();
        if let Some(status) = Self::validate_id("job_id", &req.job_id) {
            return Err(status);
        }
        if let Some(status) = Self::validate_id("node_id", &req.node_id) {
            return Err(status);
        }
        if req.want == 0 {
            return Err(Status::invalid_argument("want must be > 0"));
        }
        if let Some(status) = self.ensure_registered(&req.node_id).await {
            return Err(status);
        }

        if !self.is_job_ready().await {
            tracing::info!(
                target: "mx8_proof",
                event = "membership_wait",
                job_id = %req.job_id,
                node_id = %req.node_id,
                manifest_hash = %self.manifest_hash,
                world_size = self.world_size,
                "job not ready"
            );
            return Err(Status::failed_precondition(
                "job not ready (membership barrier)",
            ));
        }

        let mut leases = Vec::new();
        {
            let mut state = self.state.write().await;
            if state.frozen_membership.is_none() {
                // Should have been initialized when the barrier was met, but be defensive.
                Self::freeze_membership_and_partition_ranges(
                    &mut state,
                    self.min_world_size,
                    self.shuffle,
                    self.seed,
                    self.epoch,
                )?;
            }
            for _ in 0..req.want {
                let Some(range) = Self::pop_next_range_for_node(&mut state, &req.node_id) else {
                    break;
                };

                let lease_id = format!("lease-{}", state.next_lease_id);
                state.next_lease_id = state.next_lease_id.wrapping_add(1);

                let expires_unix_time_ms =
                    Self::unix_time_ms().saturating_add(self.lease_ttl_ms as u64);

                let cursor = range.start_id;
                let lease = Lease {
                    lease_id: lease_id.clone(),
                    node_id: req.node_id.clone(),
                    range: Some(range),
                    cursor,
                    expires_unix_time_ms,
                };
                state.leases.insert(lease_id, lease.clone());
                leases.push(lease);
            }
        }

        self.metrics
            .leases_granted_total
            .inc_by(leases.len() as u64);
        self.update_gauges().await;

        for lease in &leases {
            if let Some(range) = &lease.range {
                tracing::info!(
                    target: "mx8_proof",
                    event = "lease_granted",
                    job_id = %req.job_id,
                    node_id = %req.node_id,
                    lease_id = %lease.lease_id,
                    manifest_hash = %self.manifest_hash,
                    epoch = range.epoch,
                    start_id = range.start_id,
                    end_id = range.end_id,
                    expires_unix_time_ms = lease.expires_unix_time_ms,
                    "lease granted"
                );
            }
        }

        Ok(Response::new(RequestLeaseResponse {
            wait_ms: if leases.is_empty() { 500 } else { 0 },
            leases,
        }))
    }

    async fn report_progress(
        &self,
        request: Request<ReportProgressRequest>,
    ) -> Result<Response<ReportProgressResponse>, Status> {
        self.metrics.progress_total.inc();
        let req = request.into_inner();
        if let Some(status) = Self::validate_id("job_id", &req.job_id) {
            return Err(status);
        }
        if let Some(status) = Self::validate_id("node_id", &req.node_id) {
            return Err(status);
        }
        if let Some(status) = Self::validate_id("lease_id", &req.lease_id) {
            return Err(status);
        }
        if let Some(status) = self.ensure_registered(&req.node_id).await {
            return Err(status);
        }

        let mut state = self.state.write().await;
        let (existing_node_id, existing_cursor, existing_range) = {
            let Some(existing) = state.leases.get(&req.lease_id) else {
                return Err(Status::not_found("unknown lease_id"));
            };
            (
                existing.node_id.clone(),
                existing.cursor,
                existing.range.clone(),
            )
        };

        if existing_node_id != req.node_id {
            return Err(Status::failed_precondition("lease is not owned by node_id"));
        }

        let progress_key = (req.node_id.clone(), req.lease_id.clone());
        if let Some(prev) = state.progress.get(&progress_key) {
            if req.cursor < *prev {
                tracing::warn!(
                    target: "mx8_proof",
                    event = "cursor_regressed",
                    job_id = %req.job_id,
                    node_id = %req.node_id,
                    lease_id = %req.lease_id,
                    manifest_hash = %self.manifest_hash,
                    prev_cursor = *prev,
                    cursor = req.cursor,
                    "cursor moved backwards"
                );
                return Err(Status::failed_precondition("cursor moved backwards"));
            }
        }

        if let Some(range) = &existing_range {
            if req.cursor < range.start_id || req.cursor > range.end_id {
                return Err(Status::invalid_argument("cursor out of range"));
            }
        }

        if req.cursor < existing_cursor {
            tracing::warn!(
                target: "mx8_proof",
                event = "cursor_regressed",
                job_id = %req.job_id,
                node_id = %req.node_id,
                lease_id = %req.lease_id,
                manifest_hash = %self.manifest_hash,
                prev_cursor = existing_cursor,
                cursor = req.cursor,
                "cursor moved backwards (relative to lease cursor)"
            );
            return Err(Status::failed_precondition(
                "cursor moved backwards (relative to lease cursor)",
            ));
        }

        let completed = existing_range
            .as_ref()
            .is_some_and(|range| req.cursor >= range.end_id);

        state.progress.insert(progress_key, req.cursor);

        if completed {
            let removed = state.leases.remove(&req.lease_id);
            state.progress.retain(|(_, id), _| id != &req.lease_id);
            // Drop the write lock BEFORE async operations to avoid deadlock.
            // update_gauges() needs a read lock; holding write + requesting read = deadlock.
            drop(state);

            if let Some(removed) = &removed {
                if let Some(range) = &removed.range {
                    // Durably record the completion.  We write *after* removing from memory;
                    // if the coordinator crashes in this narrow window, the range will be
                    // re-issued on restart (safe: agents get NotFound and re-request).
                    if let Some(log) = &self.lease_log {
                        let mut log = log.lock().await;
                        if let Err(e) = log.append_completed(range.start_id, range.end_id).await {
                            tracing::error!(
                                error = %e,
                                path = %log.path().display(),
                                lease_id = %req.lease_id,
                                "lease log write failed; durability degraded for this range"
                            );
                        }
                    }

                    tracing::info!(
                        target: "mx8_proof",
                        event = "lease_completed",
                        job_id = %req.job_id,
                        node_id = %req.node_id,
                        lease_id = %req.lease_id,
                        manifest_hash = %self.manifest_hash,
                        epoch = range.epoch,
                        start_id = range.start_id,
                        end_id = range.end_id,
                        cursor = req.cursor,
                        delivered_samples = req.delivered_samples,
                        delivered_bytes = req.delivered_bytes,
                        unix_time_ms = req.unix_time_ms,
                        "lease completed"
                    );
                }
            }
            self.update_gauges().await;
            return Ok(Response::new(ReportProgressResponse {}));
        }

        if let Some(lease) = state.leases.get_mut(&req.lease_id) {
            lease.cursor = req.cursor;
            lease.expires_unix_time_ms =
                Self::unix_time_ms().saturating_add(self.lease_ttl_ms as u64);
        }

        tracing::info!(
            target: "mx8_proof",
            event = "progress",
            job_id = %req.job_id,
            node_id = %req.node_id,
            lease_id = %req.lease_id,
            manifest_hash = %self.manifest_hash,
            cursor = req.cursor,
            delivered_samples = req.delivered_samples,
            delivered_bytes = req.delivered_bytes,
            unix_time_ms = req.unix_time_ms,
            "progress accepted"
        );
        Ok(Response::new(ReportProgressResponse {}))
    }

    async fn get_manifest(
        &self,
        request: Request<GetManifestRequest>,
    ) -> Result<Response<GetManifestResponse>, Status> {
        let req = request.into_inner();
        if let Some(status) = Self::validate_id("job_id", &req.job_id) {
            return Err(status);
        }
        if let Some(status) = Self::validate_id("manifest_hash", &req.manifest_hash) {
            return Err(status);
        }
        let Some(store) = &self.manifest_store else {
            return Err(Status::failed_precondition(
                "manifest_store is not configured for this coordinator",
            ));
        };

        match store.get_manifest_bytes(&ManifestHash(req.manifest_hash.clone())) {
            Ok(bytes) => Ok(Response::new(GetManifestResponse {
                manifest_bytes: bytes,
                schema_version: MANIFEST_SCHEMA_VERSION,
            })),
            Err(mx8_manifest_store::ManifestStoreError::NotFound(_)) => {
                Err(Status::not_found("manifest not found"))
            }
            Err(mx8_manifest_store::ManifestStoreError::InvalidManifestHash) => {
                Err(Status::invalid_argument("invalid manifest_hash"))
            }
            Err(err) => Err(Status::internal(format!("manifest_store error: {err}"))),
        }
    }

    async fn get_manifest_stream(
        &self,
        request: Request<GetManifestRequest>,
    ) -> Result<Response<Self::GetManifestStreamStream>, Status> {
        let req = request.into_inner();
        if let Some(status) = Self::validate_id("job_id", &req.job_id) {
            return Err(status);
        }
        if let Some(status) = Self::validate_id("manifest_hash", &req.manifest_hash) {
            return Err(status);
        }
        let Some(store) = &self.manifest_store else {
            return Err(Status::failed_precondition(
                "manifest_store is not configured for this coordinator",
            ));
        };

        let bytes = match store.get_manifest_bytes(&ManifestHash(req.manifest_hash.clone())) {
            Ok(bytes) => bytes,
            Err(mx8_manifest_store::ManifestStoreError::NotFound(_)) => {
                return Err(Status::not_found("manifest not found"));
            }
            Err(mx8_manifest_store::ManifestStoreError::InvalidManifestHash) => {
                return Err(Status::invalid_argument("invalid manifest_hash"));
            }
            Err(err) => {
                return Err(Status::internal(format!("manifest_store error: {err}")));
            }
        };

        let (tx, rx) = mpsc::channel::<Result<ManifestChunk, Status>>(4);
        tokio::spawn(async move {
            for chunk in bytes.chunks(MANIFEST_CHUNK_BYTES) {
                let msg = ManifestChunk {
                    data: chunk.to_vec(),
                    schema_version: MANIFEST_SCHEMA_VERSION,
                };
                if tx.send(Ok(msg)).await.is_err() {
                    break;
                }
            }
        });

        Ok(Response::new(
            Box::pin(ReceiverStream::new(rx)) as Self::GetManifestStreamStream
        ))
    }

    async fn get_job_snapshot(
        &self,
        request: Request<GetJobSnapshotRequest>,
    ) -> Result<Response<GetJobSnapshotResponse>, Status> {
        let req = request.into_inner();
        if let Some(status) = Self::validate_id("job_id", &req.job_id) {
            return Err(status);
        }
        Ok(Response::new(self.build_snapshot().await))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    mx8_observe::logging::init_tracing();

    let args = Args::parse();

    let manifest_store_root = args
        .manifest_store_root
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(|h| std::path::PathBuf::from(h).join(".mx8/manifests"))
                .unwrap_or_else(|| std::path::PathBuf::from("/tmp/.mx8/manifests"))
        });
    let store: Arc<dyn ManifestStore> = Arc::from(mx8_manifest_store::open_from_root(
        &manifest_store_root.to_string_lossy(),
    )?);

    let resolved_manifest_hash = if let Some(link) = &args.dataset_link {
        let cfg = SnapshotResolverConfig {
            lock_stale_after: std::time::Duration::from_millis(args.snapshot_lock_stale_ms),
            wait_timeout: std::time::Duration::from_millis(args.snapshot_wait_timeout_ms),
            dev_manifest_path: args.dev_manifest_path.clone(),
            ..Default::default()
        };
        let resolver = SnapshotResolver::new(store.clone(), cfg);
        let resolved = resolver.resolve(
            link,
            mx8_manifest_store::LockOwner {
                node_id: Some("coordinator".to_string()),
            },
        )?;
        resolved.manifest_hash.0
    } else {
        args.manifest_hash.clone()
    };

    let span = info_span!(
        "mx8-coordinator",
        addr = %args.addr,
        world_size = args.world_size,
        manifest_hash = %resolved_manifest_hash
    );
    async move {
        info!("starting coordinator (v0 skeleton)");
        if args.dev_total_samples > 0 && args.dev_block_size == 0 {
            anyhow::bail!("MX8_DEV_BLOCK_SIZE must be > 0 when MX8_DEV_TOTAL_SAMPLES > 0");
        }

        // --- Lease write-ahead log -------------------------------------------
        // Derive the default log path from the manifest store root.
        let lease_log_path_default = manifest_store_root
            .parent()
            .unwrap_or(&manifest_store_root)
            .join("lease_logs")
            .join(format!("{resolved_manifest_hash}.log"));

        let raw_lease_log_path = args
            .lease_log_path
            .as_deref()
            .unwrap_or_else(|| lease_log_path_default.to_str().unwrap_or(""));

        // Allow the user to explicitly opt out ("none" or empty string).
        let lease_log_enabled = !raw_lease_log_path.is_empty()
            && raw_lease_log_path != "none"
            && resolved_manifest_hash != "dev";

        type LeaseLogState = (Option<Arc<Mutex<lease_log::LeaseLog>>>, HashSet<(u64, u64)>);
        let (lease_log_arc, completed_ranges): LeaseLogState = if lease_log_enabled {
            let log_path = PathBuf::from(raw_lease_log_path);
            match lease_log::LeaseLog::open(&log_path, &resolved_manifest_hash).await {
                Ok((log, completed)) => {
                    let recovered = completed.len();
                    info!(
                        path = %log_path.display(),
                        recovered_ranges = recovered,
                        "lease log opened"
                    );
                    (Some(Arc::new(Mutex::new(log))), completed)
                }
                Err(e) => {
                    return Err(anyhow::anyhow!(
                        "failed to open lease log at {}: {} (set --lease-log-path=none to disable crash recovery)",
                        raw_lease_log_path,
                        e
                    ));
                }
            }
        } else {
            (None, HashSet::new())
        };
        // ---------------------------------------------------------------------

        let mut available_ranges = std::collections::VecDeque::new();
        let total_samples = if args.dev_total_samples > 0 {
            args.dev_total_samples
        } else if resolved_manifest_hash != "dev" {
            match store.get_manifest_bytes(&ManifestHash(resolved_manifest_hash.clone())) {
                Ok(bytes) => count_samples_in_canonical_manifest_tsv(&bytes)?,
                Err(err) => {
                    tracing::warn!(error = %err, "could not derive total_samples from manifest");
                    0
                }
            }
        } else {
            0
        };

        let mut skipped_ranges: u64 = 0;
        if total_samples > 0 {
            let mut start_id = 0u64;
            while start_id < total_samples {
                let end_id = (start_id + args.dev_block_size).min(total_samples);
                if completed_ranges.contains(&(start_id, end_id)) {
                    skipped_ranges = skipped_ranges.saturating_add(1);
                } else {
                    available_ranges.push_back(WorkRange {
                        start_id,
                        end_id,
                        epoch: 0,
                        seed: 0,
                    });
                }
                start_id = end_id;
            }
        }

        if skipped_ranges > 0 {
            info!(
                skipped_ranges,
                remaining_ranges = available_ranges.len(),
                "coordinator resumed: skipped already-completed ranges"
            );
        }

        // min_world_size=0 means "same as world_size" (default).
        let min_world_size = if args.min_world_size == 0 {
            args.world_size
        } else {
            args.min_world_size.min(args.world_size)
        };

        let svc = CoordinatorSvc {
            state: Arc::new(RwLock::new(CoordinatorState {
                nodes: std::collections::BTreeMap::new(),
                available_ranges,
                rank_ranges: None,
                frozen_membership: None,
                departed_nodes: std::collections::BTreeSet::new(),
                leases: std::collections::BTreeMap::new(),
                progress: std::collections::BTreeMap::new(),
                next_lease_id: 0,
                // tick_once will emit job_drained on the first tick if all ranges were
                // already completed when the coordinator was resumed.
                drained_emitted: false,
            })),
            metrics: Arc::new(CoordinatorMetrics::default()),
            world_size: args.world_size,
            min_world_size,
            heartbeat_interval_ms: args.heartbeat_interval_ms,
            lease_ttl_ms: args.lease_ttl_ms,
            shuffle: args.shuffle,
            seed: args.seed,
            epoch: args.epoch,
            manifest_hash: resolved_manifest_hash,
            manifest_store: Some(store.clone()),
            lease_log: lease_log_arc,
        };

        let maintenance_svc = svc.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(std::time::Duration::from_millis(1000));
            loop {
                ticker.tick().await;
                maintenance_svc.tick_once().await;
            }
        });

        if args.metrics_snapshot_interval_ms > 0 {
            let svc_clone = svc.clone();
            let interval_ms = args.metrics_snapshot_interval_ms;
            tokio::spawn(async move {
                let mut ticker =
                    tokio::time::interval(std::time::Duration::from_millis(interval_ms));
                loop {
                    ticker.tick().await;
                    svc_clone.emit_metrics_snapshot().await;
                }
            });
        }
        Server::builder()
            .add_service(
                CoordinatorServer::new(svc)
                    .max_decoding_message_size(args.grpc_max_message_bytes)
                    .max_encoding_message_size(args.grpc_max_message_bytes),
            )
            .serve(args.addr)
            .await?;
        Ok(())
    }
    .instrument(span)
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store_root(test_name: &str) -> anyhow::Result<std::path::PathBuf> {
        let mut root = std::env::temp_dir();
        root.push(format!(
            "mx8-coordinator-test-{}-{}-{}",
            test_name,
            std::process::id(),
            CoordinatorSvc::unix_time_ms()
        ));
        std::fs::create_dir_all(&root)?;
        Ok(root)
    }

    #[tokio::test]
    async fn register_node_empty_job_id_is_invalid_argument() {
        let svc = CoordinatorSvc {
            state: Arc::new(RwLock::new(CoordinatorState::default())),
            metrics: Arc::new(CoordinatorMetrics::default()),
            world_size: 1,
            min_world_size: 1,
            heartbeat_interval_ms: 1000,
            lease_ttl_ms: 10_000,
            shuffle: false,
            seed: 0,
            epoch: 0,
            manifest_hash: "dev".to_string(),
            manifest_store: None,
            lease_log: None,
        };

        let err = svc
            .register_node(Request::new(RegisterNodeRequest {
                job_id: "".to_string(),
                node_id: "node".to_string(),
                caps: Some(NodeCaps::default()),
            }))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }

    #[tokio::test]
    async fn rank_assignment_is_stable_after_membership_barrier() -> anyhow::Result<()> {
        let svc = CoordinatorSvc {
            state: Arc::new(RwLock::new(CoordinatorState::default())),
            metrics: Arc::new(CoordinatorMetrics::default()),
            world_size: 2,
            min_world_size: 2,
            heartbeat_interval_ms: 1000,
            lease_ttl_ms: 10_000,
            shuffle: false,
            seed: 0,
            epoch: 0,
            manifest_hash: "dev".to_string(),
            manifest_store: None,
            lease_log: None,
        };

        let node2_first = svc
            .register_node(Request::new(RegisterNodeRequest {
                job_id: "job".to_string(),
                node_id: "node2".to_string(),
                caps: Some(NodeCaps::default()),
            }))
            .await?
            .into_inner();
        assert_eq!(node2_first.assigned_rank, 0);
        assert!(!node2_first.job_ready);

        let node1_second = svc
            .register_node(Request::new(RegisterNodeRequest {
                job_id: "job".to_string(),
                node_id: "node1".to_string(),
                caps: Some(NodeCaps::default()),
            }))
            .await?
            .into_inner();
        assert_eq!(node1_second.assigned_rank, 0);
        assert!(node1_second.job_ready);

        let node2_again = svc
            .register_node(Request::new(RegisterNodeRequest {
                job_id: "job".to_string(),
                node_id: "node2".to_string(),
                caps: Some(NodeCaps::default()),
            }))
            .await?
            .into_inner();
        assert_eq!(node2_again.assigned_rank, 1);
        assert!(node2_again.job_ready);
        Ok(())
    }

    #[tokio::test]
    async fn request_lease_want_zero_is_invalid_argument() {
        let svc = CoordinatorSvc {
            state: Arc::new(RwLock::new(CoordinatorState::default())),
            metrics: Arc::new(CoordinatorMetrics::default()),
            world_size: 1,
            min_world_size: 1,
            heartbeat_interval_ms: 1000,
            lease_ttl_ms: 10_000,
            shuffle: false,
            seed: 0,
            epoch: 0,
            manifest_hash: "dev".to_string(),
            manifest_store: None,
            lease_log: None,
        };

        svc.register_node(Request::new(RegisterNodeRequest {
            job_id: "job".to_string(),
            node_id: "node".to_string(),
            caps: Some(NodeCaps::default()),
        }))
        .await
        .unwrap();

        let err = svc
            .request_lease(Request::new(RequestLeaseRequest {
                job_id: "job".to_string(),
                node_id: "node".to_string(),
                want: 0,
            }))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }

    #[tokio::test]
    async fn get_manifest_serves_bytes_from_store() -> anyhow::Result<()> {
        let root = temp_store_root("get-manifest")?;
        let store: Arc<dyn ManifestStore> =
            Arc::new(mx8_manifest_store::fs::FsManifestStore::new(root));

        let hash = ManifestHash("h".to_string());
        store.put_manifest_bytes(&hash, b"manifest")?;

        let svc = CoordinatorSvc {
            state: Arc::new(RwLock::new(CoordinatorState::default())),
            metrics: Arc::new(CoordinatorMetrics::default()),
            world_size: 1,
            min_world_size: 1,
            heartbeat_interval_ms: 1000,
            lease_ttl_ms: 10_000,
            shuffle: false,
            seed: 0,
            epoch: 0,
            manifest_hash: "h".to_string(),
            manifest_store: Some(store),
            lease_log: None,
        };

        let resp = svc
            .get_manifest(Request::new(GetManifestRequest {
                job_id: "job".to_string(),
                manifest_hash: "h".to_string(),
            }))
            .await?
            .into_inner();

        assert_eq!(resp.manifest_bytes, b"manifest");
        assert_eq!(resp.schema_version, MANIFEST_SCHEMA_VERSION);
        Ok(())
    }

    #[tokio::test]
    async fn get_manifest_stream_reassembles_large_manifest() -> anyhow::Result<()> {
        use tokio_stream::StreamExt;

        let root = temp_store_root("get-manifest-stream")?;
        let store: Arc<dyn ManifestStore> =
            Arc::new(mx8_manifest_store::fs::FsManifestStore::new(root));

        let mut manifest = Vec::new();
        manifest.extend_from_slice(b"schema_version=0\n");
        manifest.extend(std::iter::repeat_n(b'a', 6 * 1024 * 1024));

        let hash = ManifestHash("h".to_string());
        store.put_manifest_bytes(&hash, manifest.as_slice())?;

        let svc = CoordinatorSvc {
            state: Arc::new(RwLock::new(CoordinatorState::default())),
            metrics: Arc::new(CoordinatorMetrics::default()),
            world_size: 1,
            min_world_size: 1,
            heartbeat_interval_ms: 1000,
            lease_ttl_ms: 10_000,
            shuffle: false,
            seed: 0,
            epoch: 0,
            manifest_hash: "h".to_string(),
            manifest_store: Some(store),
            lease_log: None,
        };

        let resp = svc
            .get_manifest_stream(Request::new(GetManifestRequest {
                job_id: "job".to_string(),
                manifest_hash: "h".to_string(),
            }))
            .await?
            .into_inner();

        let mut stream = resp;
        let mut out = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            anyhow::ensure!(
                chunk.data.len() <= MANIFEST_CHUNK_BYTES,
                "chunk larger than MANIFEST_CHUNK_BYTES"
            );
            anyhow::ensure!(
                chunk.schema_version == MANIFEST_SCHEMA_VERSION,
                "unexpected schema_version"
            );
            out.extend_from_slice(chunk.data.as_slice());
        }

        assert_eq!(out, manifest);
        Ok(())
    }

    #[tokio::test]
    async fn expired_lease_requeues_remainder_range() -> anyhow::Result<()> {
        let state = Arc::new(RwLock::new(CoordinatorState {
            nodes: std::collections::BTreeMap::new(),
            available_ranges: std::collections::VecDeque::from([WorkRange {
                start_id: 0,
                end_id: 100,
                epoch: 0,
                seed: 0,
            }]),
            rank_ranges: None,
            frozen_membership: None,
            departed_nodes: std::collections::BTreeSet::new(),
            leases: std::collections::BTreeMap::new(),
            progress: std::collections::BTreeMap::new(),
            next_lease_id: 0,
            drained_emitted: false,
        }));

        let svc = CoordinatorSvc {
            state: state.clone(),
            metrics: Arc::new(CoordinatorMetrics::default()),
            world_size: 1,
            min_world_size: 1,
            heartbeat_interval_ms: 1000,
            lease_ttl_ms: 10_000,
            shuffle: false,
            seed: 0,
            epoch: 0,
            manifest_hash: "dev".to_string(),
            manifest_store: None,
            lease_log: None,
        };

        svc.register_node(Request::new(RegisterNodeRequest {
            job_id: "job".to_string(),
            node_id: "node".to_string(),
            caps: Some(NodeCaps::default()),
        }))
        .await?;

        let lease = svc
            .request_lease(Request::new(RequestLeaseRequest {
                job_id: "job".to_string(),
                node_id: "node".to_string(),
                want: 1,
            }))
            .await?
            .into_inner()
            .leases
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("expected lease"))?;

        let Some(range) = &lease.range else {
            anyhow::bail!("expected range");
        };
        assert_eq!(range.start_id, 0);
        assert_eq!(range.end_id, 100);

        svc.report_progress(Request::new(ReportProgressRequest {
            job_id: "job".to_string(),
            node_id: "node".to_string(),
            lease_id: lease.lease_id.clone(),
            cursor: 40,
            delivered_samples: 40,
            delivered_bytes: 0,
            unix_time_ms: CoordinatorSvc::unix_time_ms(),
        }))
        .await?;

        let now = CoordinatorSvc::unix_time_ms();
        {
            let mut st = state.write().await;
            let entry = st
                .leases
                .get_mut(&lease.lease_id)
                .ok_or_else(|| anyhow::anyhow!("lease missing"))?;
            entry.expires_unix_time_ms = now.saturating_sub(1);
        }

        svc.tick_once_at(now).await;

        let st = state.read().await;
        assert!(st.leases.is_empty());
        let front = st
            .available_ranges
            .front()
            .ok_or_else(|| anyhow::anyhow!("expected requeued range"))?;
        assert_eq!(front.start_id, 40);
        assert_eq!(front.end_id, 100);
        Ok(())
    }

    #[tokio::test]
    async fn report_progress_rejects_cursor_regression() -> anyhow::Result<()> {
        let state = Arc::new(RwLock::new(CoordinatorState {
            nodes: std::collections::BTreeMap::new(),
            available_ranges: std::collections::VecDeque::from([WorkRange {
                start_id: 0,
                end_id: 100,
                epoch: 0,
                seed: 0,
            }]),
            rank_ranges: None,
            frozen_membership: None,
            departed_nodes: std::collections::BTreeSet::new(),
            leases: std::collections::BTreeMap::new(),
            progress: std::collections::BTreeMap::new(),
            next_lease_id: 0,
            drained_emitted: false,
        }));

        let svc = CoordinatorSvc {
            state: state.clone(),
            metrics: Arc::new(CoordinatorMetrics::default()),
            world_size: 1,
            min_world_size: 1,
            heartbeat_interval_ms: 1000,
            lease_ttl_ms: 10_000,
            shuffle: false,
            seed: 0,
            epoch: 0,
            manifest_hash: "dev".to_string(),
            manifest_store: None,
            lease_log: None,
        };

        svc.register_node(Request::new(RegisterNodeRequest {
            job_id: "job".to_string(),
            node_id: "node".to_string(),
            caps: Some(NodeCaps::default()),
        }))
        .await?;

        let lease = svc
            .request_lease(Request::new(RequestLeaseRequest {
                job_id: "job".to_string(),
                node_id: "node".to_string(),
                want: 1,
            }))
            .await?
            .into_inner()
            .leases
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("expected lease"))?;

        svc.report_progress(Request::new(ReportProgressRequest {
            job_id: "job".to_string(),
            node_id: "node".to_string(),
            lease_id: lease.lease_id.clone(),
            cursor: 10,
            delivered_samples: 0,
            delivered_bytes: 0,
            unix_time_ms: CoordinatorSvc::unix_time_ms(),
        }))
        .await?;

        let err = svc
            .report_progress(Request::new(ReportProgressRequest {
                job_id: "job".to_string(),
                node_id: "node".to_string(),
                lease_id: lease.lease_id.clone(),
                cursor: 9,
                delivered_samples: 0,
                delivered_bytes: 0,
                unix_time_ms: CoordinatorSvc::unix_time_ms(),
            }))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::FailedPrecondition);
        Ok(())
    }

    /// Run a single-node coordinator over a 3-block manifest, complete all leases,
    /// then restart with the same lease log and verify no new ranges are issued
    /// (the job starts already drained).
    #[tokio::test]
    async fn lease_log_restart_skips_completed_ranges() -> anyhow::Result<()> {
        let log_path = std::env::temp_dir().join(format!(
            "mx8-lease-log-restart-test-{}.log",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&log_path);

        // -- Run 1: complete all 3 ranges [0,100), [100,200), [200,300) --
        {
            let (log, completed) = lease_log::LeaseLog::open(&log_path, "testhash").await?;
            assert!(completed.is_empty());
            let log_arc = Arc::new(Mutex::new(log));

            let available_ranges = std::collections::VecDeque::from([
                WorkRange {
                    start_id: 0,
                    end_id: 100,
                    epoch: 0,
                    seed: 0,
                },
                WorkRange {
                    start_id: 100,
                    end_id: 200,
                    epoch: 0,
                    seed: 0,
                },
                WorkRange {
                    start_id: 200,
                    end_id: 300,
                    epoch: 0,
                    seed: 0,
                },
            ]);

            let svc = CoordinatorSvc {
                state: Arc::new(RwLock::new(CoordinatorState {
                    nodes: std::collections::BTreeMap::new(),
                    available_ranges,
                    rank_ranges: None,
                    frozen_membership: None,
                    departed_nodes: std::collections::BTreeSet::new(),
                    leases: std::collections::BTreeMap::new(),
                    progress: std::collections::BTreeMap::new(),
                    next_lease_id: 0,
                    drained_emitted: false,
                })),
                metrics: Arc::new(CoordinatorMetrics::default()),
                world_size: 1,
                min_world_size: 1,
                heartbeat_interval_ms: 1000,
                lease_ttl_ms: 10_000,
                shuffle: false,
                seed: 0,
                epoch: 0,
                manifest_hash: "testhash".to_string(),
                manifest_store: None,
                lease_log: Some(log_arc),
            };

            svc.register_node(Request::new(RegisterNodeRequest {
                job_id: "job".to_string(),
                node_id: "node".to_string(),
                caps: Some(NodeCaps::default()),
            }))
            .await?;

            // Complete all 3 leases one by one.
            for _ in 0..3 {
                let resp = svc
                    .request_lease(Request::new(RequestLeaseRequest {
                        job_id: "job".to_string(),
                        node_id: "node".to_string(),
                        want: 1,
                    }))
                    .await?
                    .into_inner();
                let lease = resp
                    .leases
                    .into_iter()
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("expected a lease"))?;
                let end = lease.range.as_ref().map(|r| r.end_id).unwrap_or(0);
                svc.report_progress(Request::new(ReportProgressRequest {
                    job_id: "job".to_string(),
                    node_id: "node".to_string(),
                    lease_id: lease.lease_id,
                    cursor: end,
                    delivered_samples: end,
                    delivered_bytes: 0,
                    unix_time_ms: CoordinatorSvc::unix_time_ms(),
                }))
                .await?;
            }
        }

        // -- Run 2: replay log, build available_ranges (should be empty) --
        {
            let (_log2, completed) = lease_log::LeaseLog::open(&log_path, "testhash").await?;
            assert_eq!(completed.len(), 3, "expected 3 completed ranges in log");
            assert!(completed.contains(&(0, 100)));
            assert!(completed.contains(&(100, 200)));
            assert!(completed.contains(&(200, 300)));

            // Simulate what main() does: filter completed ranges out.
            let block_size = 100u64;
            let total_samples = 300u64;
            let mut available: std::collections::VecDeque<WorkRange> =
                std::collections::VecDeque::new();
            let mut start_id = 0u64;
            while start_id < total_samples {
                let end_id = (start_id + block_size).min(total_samples);
                if !completed.contains(&(start_id, end_id)) {
                    available.push_back(WorkRange {
                        start_id,
                        end_id,
                        epoch: 0,
                        seed: 0,
                    });
                }
                start_id = end_id;
            }
            assert!(
                available.is_empty(),
                "all ranges should be filtered out after restart"
            );
        }

        let _ = std::fs::remove_file(&log_path);
        Ok(())
    }

    // ── Step 2: dynamic membership tests ─────────────────────────────────────

    /// A node that times out is marked departed; a replacement node fills its slot
    /// and gets the same rank.
    #[tokio::test]
    async fn replacement_node_fills_departed_slot() -> anyhow::Result<()> {
        let svc = CoordinatorSvc {
            state: Arc::new(RwLock::new(CoordinatorState {
                nodes: std::collections::BTreeMap::new(),
                available_ranges: std::collections::VecDeque::from([WorkRange {
                    start_id: 0,
                    end_id: 100,
                    epoch: 0,
                    seed: 0,
                }]),
                rank_ranges: None,
                frozen_membership: None,
                departed_nodes: std::collections::BTreeSet::new(),
                leases: std::collections::BTreeMap::new(),
                progress: std::collections::BTreeMap::new(),
                next_lease_id: 0,
                drained_emitted: false,
            })),
            metrics: Arc::new(CoordinatorMetrics::default()),
            world_size: 1,
            min_world_size: 1,
            heartbeat_interval_ms: 1000,
            lease_ttl_ms: 5_000,
            shuffle: false,
            seed: 0,
            epoch: 0,
            manifest_hash: "dev".to_string(),
            manifest_store: None,
            lease_log: None,
        };

        // Original node registers → barrier met → frozen at rank 0.
        let r1 = svc
            .register_node(Request::new(RegisterNodeRequest {
                job_id: "job".to_string(),
                node_id: "node-a".to_string(),
                caps: Some(NodeCaps::default()),
            }))
            .await?
            .into_inner();
        assert_eq!(r1.assigned_rank, 0);
        assert!(r1.job_ready);

        // Simulate node-a going dead: backdate heartbeat past TTL.
        {
            let now = CoordinatorSvc::unix_time_ms();
            let mut st = svc.state.write().await;
            if let Some(e) = st.nodes.get_mut("node-a") {
                e.last_heartbeat_unix_time_ms =
                    Some(now.saturating_sub(svc.lease_ttl_ms as u64 + 1));
            }
        }
        svc.tick_once().await;

        // node-a should now be in departed_nodes.
        {
            let st = svc.state.read().await;
            assert!(
                st.departed_nodes.contains("node-a"),
                "node-a should be departed"
            );
        }

        // Replacement node-b registers and fills node-a's slot at rank 0.
        let r2 = svc
            .register_node(Request::new(RegisterNodeRequest {
                job_id: "job".to_string(),
                node_id: "node-b".to_string(),
                caps: Some(NodeCaps::default()),
            }))
            .await?
            .into_inner();
        assert_eq!(r2.assigned_rank, 0, "replacement should inherit rank 0");
        assert!(r2.job_ready);

        // node-a should no longer be in departed_nodes.
        {
            let st = svc.state.read().await;
            assert!(
                !st.departed_nodes.contains("node-a"),
                "slot should be cleared after replacement"
            );
            assert!(
                !st.nodes.contains_key("node-a"),
                "old node should be evicted from nodes map"
            );
        }

        Ok(())
    }

    /// No vacant slot → new node is rejected.
    #[tokio::test]
    async fn no_vacant_slot_rejects_new_node_after_freeze() -> anyhow::Result<()> {
        let svc = CoordinatorSvc {
            state: Arc::new(RwLock::new(CoordinatorState::default())),
            metrics: Arc::new(CoordinatorMetrics::default()),
            world_size: 1,
            min_world_size: 1,
            heartbeat_interval_ms: 1000,
            lease_ttl_ms: 10_000,
            shuffle: false,
            seed: 0,
            epoch: 0,
            manifest_hash: "dev".to_string(),
            manifest_store: None,
            lease_log: None,
        };

        // node-a fills the only slot.
        svc.register_node(Request::new(RegisterNodeRequest {
            job_id: "job".to_string(),
            node_id: "node-a".to_string(),
            caps: Some(NodeCaps::default()),
        }))
        .await?;

        // node-b tries to register while node-a is still active → rejected.
        let err = svc
            .register_node(Request::new(RegisterNodeRequest {
                job_id: "job".to_string(),
                node_id: "node-b".to_string(),
                caps: Some(NodeCaps::default()),
            }))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::FailedPrecondition);
        Ok(())
    }

    /// With world_size headroom, a post-freeze node is admitted and receives work.
    #[tokio::test]
    async fn scale_out_node_is_admitted_after_freeze() -> anyhow::Result<()> {
        let svc = CoordinatorSvc {
            state: Arc::new(RwLock::new(CoordinatorState {
                nodes: std::collections::BTreeMap::new(),
                available_ranges: std::collections::VecDeque::from([
                    WorkRange {
                        start_id: 0,
                        end_id: 100,
                        epoch: 0,
                        seed: 0,
                    },
                    WorkRange {
                        start_id: 100,
                        end_id: 200,
                        epoch: 0,
                        seed: 0,
                    },
                ]),
                rank_ranges: None,
                frozen_membership: None,
                departed_nodes: std::collections::BTreeSet::new(),
                leases: std::collections::BTreeMap::new(),
                progress: std::collections::BTreeMap::new(),
                next_lease_id: 0,
                drained_emitted: false,
            })),
            metrics: Arc::new(CoordinatorMetrics::default()),
            world_size: 2,
            min_world_size: 1,
            heartbeat_interval_ms: 1000,
            lease_ttl_ms: 10_000,
            shuffle: false,
            seed: 0,
            epoch: 0,
            manifest_hash: "dev".to_string(),
            manifest_store: None,
            lease_log: None,
        };

        let first = svc
            .register_node(Request::new(RegisterNodeRequest {
                job_id: "job".to_string(),
                node_id: "node-a".to_string(),
                caps: Some(NodeCaps::default()),
            }))
            .await?
            .into_inner();
        assert_eq!(first.assigned_rank, 0);
        assert_eq!(first.world_size, 2);
        assert!(first.job_ready);

        let second = svc
            .register_node(Request::new(RegisterNodeRequest {
                job_id: "job".to_string(),
                node_id: "node-b".to_string(),
                caps: Some(NodeCaps::default()),
            }))
            .await?
            .into_inner();
        assert_eq!(second.assigned_rank, 1);
        assert_eq!(second.world_size, 2);
        assert!(second.job_ready);

        let lease = svc
            .request_lease(Request::new(RequestLeaseRequest {
                job_id: "job".to_string(),
                node_id: "node-b".to_string(),
                want: 1,
            }))
            .await?
            .into_inner()
            .leases
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("expected lease for scale-out node"))?;

        let range = lease
            .range
            .ok_or_else(|| anyhow::anyhow!("expected range in scale-out lease"))?;
        assert_eq!(range.start_id, 100);
        assert_eq!(range.end_id, 200);
        Ok(())
    }

    /// min_world_size < world_size: job starts when min_world_size nodes register.
    #[tokio::test]
    async fn min_world_size_allows_early_start() -> anyhow::Result<()> {
        let svc = CoordinatorSvc {
            state: Arc::new(RwLock::new(CoordinatorState {
                nodes: std::collections::BTreeMap::new(),
                available_ranges: std::collections::VecDeque::from([WorkRange {
                    start_id: 0,
                    end_id: 200,
                    epoch: 0,
                    seed: 0,
                }]),
                rank_ranges: None,
                frozen_membership: None,
                departed_nodes: std::collections::BTreeSet::new(),
                leases: std::collections::BTreeMap::new(),
                progress: std::collections::BTreeMap::new(),
                next_lease_id: 0,
                drained_emitted: false,
            })),
            metrics: Arc::new(CoordinatorMetrics::default()),
            world_size: 2,
            min_world_size: 1, // start with just 1 node
            heartbeat_interval_ms: 1000,
            lease_ttl_ms: 10_000,
            shuffle: false,
            seed: 0,
            epoch: 0,
            manifest_hash: "dev".to_string(),
            manifest_store: None,
            lease_log: None,
        };

        // Only 1 node registers (< world_size=2) but >= min_world_size=1.
        let r = svc
            .register_node(Request::new(RegisterNodeRequest {
                job_id: "job".to_string(),
                node_id: "node-a".to_string(),
                caps: Some(NodeCaps::default()),
            }))
            .await?
            .into_inner();
        assert!(
            r.job_ready,
            "job should be ready after min_world_size is met"
        );

        // node-a should be able to request a lease.
        let lease_resp = svc
            .request_lease(Request::new(RequestLeaseRequest {
                job_id: "job".to_string(),
                node_id: "node-a".to_string(),
                want: 1,
            }))
            .await?
            .into_inner();
        assert!(!lease_resp.leases.is_empty(), "node-a should get a lease");

        Ok(())
    }

    /// A departed node that recovers with the same node_id is un-departed on re-register.
    #[tokio::test]
    async fn departed_node_can_rejoin_with_same_identity() -> anyhow::Result<()> {
        let svc = CoordinatorSvc {
            state: Arc::new(RwLock::new(CoordinatorState::default())),
            metrics: Arc::new(CoordinatorMetrics::default()),
            world_size: 1,
            min_world_size: 1,
            heartbeat_interval_ms: 1000,
            lease_ttl_ms: 5_000,
            shuffle: false,
            seed: 0,
            epoch: 0,
            manifest_hash: "dev".to_string(),
            manifest_store: None,
            lease_log: None,
        };

        svc.register_node(Request::new(RegisterNodeRequest {
            job_id: "job".to_string(),
            node_id: "node-a".to_string(),
            caps: Some(NodeCaps::default()),
        }))
        .await?;

        // Backdate heartbeat so node-a is detected as dead.
        {
            let now = CoordinatorSvc::unix_time_ms();
            let mut st = svc.state.write().await;
            if let Some(e) = st.nodes.get_mut("node-a") {
                e.last_heartbeat_unix_time_ms =
                    Some(now.saturating_sub(svc.lease_ttl_ms as u64 + 1));
            }
        }
        svc.tick_once().await;
        {
            let st = svc.state.read().await;
            assert!(st.departed_nodes.contains("node-a"));
        }

        // Same node recovers and re-registers.
        let r = svc
            .register_node(Request::new(RegisterNodeRequest {
                job_id: "job".to_string(),
                node_id: "node-a".to_string(),
                caps: Some(NodeCaps::default()),
            }))
            .await?
            .into_inner();
        assert_eq!(r.assigned_rank, 0);

        // Should no longer be in departed_nodes.
        let st = svc.state.read().await;
        assert!(!st.departed_nodes.contains("node-a"));
        Ok(())
    }
}
