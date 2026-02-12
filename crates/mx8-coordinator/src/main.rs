#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use clap::Parser;
use tokio::sync::RwLock;
use tonic::{transport::Server, Request, Response, Status};
use tracing::{info, info_span, Instrument};

use mx8_core::types::{ManifestHash, MANIFEST_SCHEMA_VERSION};
use mx8_manifest_store::ManifestStore;
use mx8_observe::metrics::{Counter, Gauge};
use mx8_proto::v0::coordinator_server::{Coordinator, CoordinatorServer};
use mx8_proto::v0::*;
use mx8_snapshot::{SnapshotResolver, SnapshotResolverConfig};

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

    /// Root directory for the FS manifest_store (M3).
    #[arg(
        long,
        env = "MX8_MANIFEST_STORE_ROOT",
        default_value = "/var/lib/mx8/manifests"
    )]
    manifest_store_root: PathBuf,

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
    #[arg(long, env = "MX8_DEV_TOTAL_SAMPLES", default_value_t = 0)]
    dev_total_samples: u64,

    /// Development-only: block size (samples) for lease WorkRanges.
    #[arg(long, env = "MX8_DEV_BLOCK_SIZE", default_value_t = 65_536)]
    dev_block_size: u64,

    /// Optional: periodically emit a metrics snapshot to logs.
    #[arg(long, env = "MX8_METRICS_SNAPSHOT_INTERVAL_MS", default_value_t = 0)]
    metrics_snapshot_interval_ms: u64,
}

#[derive(Debug, Clone, Default)]
struct NodeEntry {
    caps: Option<NodeCaps>,
}

#[derive(Debug, Default)]
struct CoordinatorState {
    nodes: std::collections::BTreeMap<String, NodeEntry>,
    available_ranges: std::collections::VecDeque<WorkRange>,
    leases: std::collections::BTreeMap<String, Lease>,
    progress: std::collections::BTreeMap<(String, String), u64>,
    next_lease_id: u64,
}

#[derive(Debug, Default)]
struct CoordinatorMetrics {
    register_total: Counter,
    heartbeat_total: Counter,
    request_lease_total: Counter,
    leases_granted_total: Counter,
    progress_total: Counter,
    active_leases: Gauge,
    registered_nodes: Gauge,
}

#[derive(Debug, Clone)]
struct CoordinatorSvc {
    state: Arc<RwLock<CoordinatorState>>,
    metrics: Arc<CoordinatorMetrics>,
    world_size: u32,
    heartbeat_interval_ms: u32,
    lease_ttl_ms: u32,
    manifest_hash: String,
    manifest_store: Option<Arc<mx8_manifest_store::fs::FsManifestStore>>,
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
        state.nodes.len() as u32 >= self.world_size
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
            progress_total = self.metrics.progress_total.get(),
            active_leases = self.metrics.active_leases.get(),
            registered_nodes = self.metrics.registered_nodes.get(),
            world_size = self.world_size,
            manifest_hash = %self.manifest_hash,
            "metrics"
        );
    }
}

#[tonic::async_trait]
impl Coordinator for CoordinatorSvc {
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

        let registered_nodes = {
            let mut state = self.state.write().await;
            if let Some(existing) = state.nodes.get(&req.node_id) {
                if existing.caps != req.caps {
                    tracing::warn!("RegisterNode updated caps for existing node_id");
                }
            }
            state.nodes.insert(
                req.node_id.clone(),
                NodeEntry {
                    caps: req.caps.clone(),
                },
            );
            state.nodes.len() as u32
        };

        let assigned_rank = {
            let state = self.state.read().await;
            state
                .nodes
                .keys()
                .position(|id| id == &req.node_id)
                .unwrap_or(0) as u32
        };

        if assigned_rank >= self.world_size {
            return Err(Status::failed_precondition(
                "node registered, but exceeds configured world_size",
            ));
        }

        let resp = RegisterNodeResponse {
            assigned_rank,
            world_size: self.world_size,
            manifest_hash: self.manifest_hash.clone(),
            heartbeat_interval_ms: self.heartbeat_interval_ms,
            lease_ttl_ms: self.lease_ttl_ms,
            registered_nodes,
            job_ready: registered_nodes >= self.world_size,
        };

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
            for _ in 0..req.want {
                let Some(range) = state.available_ranges.pop_front() else {
                    break;
                };

                let lease_id = format!("lease-{}", state.next_lease_id);
                state.next_lease_id = state.next_lease_id.wrapping_add(1);

                let expires_unix_time_ms =
                    Self::unix_time_ms().saturating_add(self.lease_ttl_ms as u64);

                let lease = Lease {
                    lease_id: lease_id.clone(),
                    node_id: req.node_id.clone(),
                    range: Some(range),
                    cursor: 0,
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
        let Some(lease) = state.leases.get(&req.lease_id) else {
            return Err(Status::not_found("unknown lease_id"));
        };

        if lease.node_id != req.node_id {
            return Err(Status::failed_precondition("lease is not owned by node_id"));
        }

        let progress_key = (req.node_id.clone(), req.lease_id.clone());
        if let Some(prev) = state.progress.get(&progress_key) {
            if req.cursor < *prev {
                return Err(Status::invalid_argument("cursor moved backwards"));
            }
        }

        if let Some(range) = &lease.range {
            if req.cursor < range.start_id || req.cursor > range.end_id {
                return Err(Status::invalid_argument("cursor out of range"));
            }
        }
        state.progress.insert(progress_key, req.cursor);

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
}

#[tokio::main]
async fn main() -> Result<()> {
    mx8_observe::logging::init_tracing();

    let args = Args::parse();

    let store = Arc::new(mx8_manifest_store::fs::FsManifestStore::new(
        args.manifest_store_root.clone(),
    ));

    let resolved_manifest_hash = if let Some(link) = &args.dataset_link {
        let cfg = SnapshotResolverConfig {
            lock_stale_after: std::time::Duration::from_millis(args.snapshot_lock_stale_ms),
            wait_timeout: std::time::Duration::from_millis(args.snapshot_wait_timeout_ms),
            dev_manifest_path: args.dev_manifest_path.clone(),
            ..Default::default()
        };
        let resolver = SnapshotResolver::new((*store).clone(), cfg);
        let resolved = resolver.resolve(
            link,
            mx8_manifest_store::fs::LockOwner {
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

        let mut available_ranges = std::collections::VecDeque::new();
        if args.dev_total_samples > 0 {
            let mut start_id = 0u64;
            while start_id < args.dev_total_samples {
                let end_id = (start_id + args.dev_block_size).min(args.dev_total_samples);
                available_ranges.push_back(WorkRange {
                    start_id,
                    end_id,
                    epoch: 0,
                    seed: 0,
                });
                start_id = end_id;
            }
        }

        let svc = CoordinatorSvc {
            state: Arc::new(RwLock::new(CoordinatorState {
                nodes: std::collections::BTreeMap::new(),
                available_ranges,
                leases: std::collections::BTreeMap::new(),
                progress: std::collections::BTreeMap::new(),
                next_lease_id: 0,
            })),
            metrics: Arc::new(CoordinatorMetrics::default()),
            world_size: args.world_size,
            heartbeat_interval_ms: args.heartbeat_interval_ms,
            lease_ttl_ms: args.lease_ttl_ms,
            manifest_hash: resolved_manifest_hash,
            manifest_store: Some(store),
        };

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
            .add_service(CoordinatorServer::new(svc))
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

    #[tokio::test]
    async fn register_node_empty_job_id_is_invalid_argument() {
        let svc = CoordinatorSvc {
            state: Arc::new(RwLock::new(CoordinatorState::default())),
            metrics: Arc::new(CoordinatorMetrics::default()),
            world_size: 1,
            heartbeat_interval_ms: 1000,
            lease_ttl_ms: 10_000,
            manifest_hash: "dev".to_string(),
            manifest_store: None,
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
    async fn request_lease_want_zero_is_invalid_argument() {
        let svc = CoordinatorSvc {
            state: Arc::new(RwLock::new(CoordinatorState::default())),
            metrics: Arc::new(CoordinatorMetrics::default()),
            world_size: 1,
            heartbeat_interval_ms: 1000,
            lease_ttl_ms: 10_000,
            manifest_hash: "dev".to_string(),
            manifest_store: None,
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
}
