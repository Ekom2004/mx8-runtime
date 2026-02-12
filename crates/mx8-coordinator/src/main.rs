#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use clap::Parser;
use tokio::sync::RwLock;
use tonic::{transport::Server, Request, Response, Status};
use tracing::{info, info_span, Instrument};
use tracing_subscriber::EnvFilter;

use mx8_proto::v0::coordinator_server::{Coordinator, CoordinatorServer};
use mx8_proto::v0::*;

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

    /// Pinned manifest hash for this job (placeholder until M3 snapshot resolver is wired).
    #[arg(long, env = "MX8_MANIFEST_HASH", default_value = "dev")]
    manifest_hash: String,

    /// Development-only: total number of samples to serve as contiguous WorkRanges.
    #[arg(long, env = "MX8_DEV_TOTAL_SAMPLES", default_value_t = 0)]
    dev_total_samples: u64,

    /// Development-only: block size (samples) for lease WorkRanges.
    #[arg(long, env = "MX8_DEV_BLOCK_SIZE", default_value_t = 65_536)]
    dev_block_size: u64,
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

#[derive(Debug, Clone)]
struct CoordinatorSvc {
    state: Arc<RwLock<CoordinatorState>>,
    world_size: u32,
    heartbeat_interval_ms: u32,
    lease_ttl_ms: u32,
    manifest_hash: String,
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
}

#[tonic::async_trait]
impl Coordinator for CoordinatorSvc {
    async fn register_node(
        &self,
        request: Request<RegisterNodeRequest>,
    ) -> Result<Response<RegisterNodeResponse>, Status> {
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

        Ok(Response::new(RegisterNodeResponse {
            assigned_rank,
            world_size: self.world_size,
            manifest_hash: self.manifest_hash.clone(),
            heartbeat_interval_ms: self.heartbeat_interval_ms,
            lease_ttl_ms: self.lease_ttl_ms,
            registered_nodes,
            job_ready: registered_nodes >= self.world_size,
        }))
    }

    async fn heartbeat(
        &self,
        request: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
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

        Ok(Response::new(RequestLeaseResponse {
            wait_ms: if leases.is_empty() { 500 } else { 0 },
            leases,
        }))
    }

    async fn report_progress(
        &self,
        request: Request<ReportProgressRequest>,
    ) -> Result<Response<ReportProgressResponse>, Status> {
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
        Err(Status::not_found(
            "manifest not found (M3 will implement manifest_store)",
        ))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    let span = info_span!("mx8-coordinator", addr = %args.addr);
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
            world_size: args.world_size,
            heartbeat_interval_ms: args.heartbeat_interval_ms,
            lease_ttl_ms: args.lease_ttl_ms,
            manifest_hash: args.manifest_hash,
        };
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
            world_size: 1,
            heartbeat_interval_ms: 1000,
            lease_ttl_ms: 10_000,
            manifest_hash: "dev".to_string(),
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
            world_size: 1,
            heartbeat_interval_ms: 1000,
            lease_ttl_ms: 10_000,
            manifest_hash: "dev".to_string(),
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
