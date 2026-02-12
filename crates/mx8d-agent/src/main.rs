#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use tonic::transport::Channel;
use tracing::{info, info_span, Instrument};

use mx8_proto::v0::coordinator_client::CoordinatorClient;
use mx8_proto::v0::HeartbeatRequest;
use mx8_proto::v0::NodeCaps;
use mx8_proto::v0::NodeStats;
use mx8_proto::v0::RegisterNodeRequest;
use mx8_proto::v0::ReportProgressRequest;
use mx8_proto::v0::RequestLeaseRequest;

use mx8_observe::metrics::{Counter, Gauge};

#[derive(Debug, Parser)]
#[command(name = "mx8d-agent")]
struct Args {
    /// Coordinator address, e.g. http://127.0.0.1:50051
    #[arg(long, env = "MX8_COORD_URL", default_value = "http://127.0.0.1:50051")]
    coord_url: String,

    #[arg(long, env = "MX8_JOB_ID", default_value = "local-job")]
    job_id: String,

    #[arg(long, env = "MX8_NODE_ID", default_value = "local-node")]
    node_id: String,

    /// Optional: periodically emit a metrics snapshot to logs.
    #[arg(long, env = "MX8_METRICS_SNAPSHOT_INTERVAL_MS", default_value_t = 0)]
    metrics_snapshot_interval_ms: u64,

    /// Development-only: request leases continuously with this `want` value.
    /// Set to 0 to disable.
    #[arg(long, env = "MX8_DEV_LEASE_WANT", default_value_t = 0)]
    dev_lease_want: u32,
}

#[derive(Debug, Default)]
struct AgentMetrics {
    register_total: Counter,
    heartbeat_total: Counter,
    heartbeat_ok_total: Counter,
    heartbeat_err_total: Counter,
    inflight_bytes: Gauge,
    ram_high_water_bytes: Gauge,
}

impl AgentMetrics {
    fn snapshot(&self, job_id: &str, node_id: &str, manifest_hash: &str) {
        tracing::info!(
            target: "mx8_metrics",
            job_id = %job_id,
            node_id = %node_id,
            manifest_hash = %manifest_hash,
            register_total = self.register_total.get(),
            heartbeat_total = self.heartbeat_total.get(),
            heartbeat_ok_total = self.heartbeat_ok_total.get(),
            heartbeat_err_total = self.heartbeat_err_total.get(),
            inflight_bytes = self.inflight_bytes.get(),
            ram_high_water_bytes = self.ram_high_water_bytes.get(),
            "metrics"
        );
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    mx8_observe::logging::init_tracing();

    let args = Args::parse();
    let span = info_span!(
        "mx8d-agent",
        job_id = %args.job_id,
        node_id = %args.node_id,
        coord_url = %args.coord_url
    );

    async move {
        info!("starting agent (v0 skeleton)");
        let channel = Channel::from_shared(args.coord_url)?.connect().await?;
        let mut client = CoordinatorClient::new(channel);

        let metrics = std::sync::Arc::new(AgentMetrics::default());
        metrics.register_total.inc();
        let register_resp = client
            .register_node(RegisterNodeRequest {
                job_id: args.job_id.clone(),
                node_id: args.node_id.clone(),
                caps: Some(NodeCaps {
                    max_fetch_concurrency: 32,
                    max_decode_concurrency: 8,
                    max_inflight_bytes: 1 << 30,
                    max_ram_bytes: 4 << 30,
                }),
            })
            .await?
            .into_inner();

        let heartbeat_interval_ms = std::cmp::max(1, register_resp.heartbeat_interval_ms as u64);
        let manifest_hash = register_resp.manifest_hash.clone();

        info!(
            assigned_rank = register_resp.assigned_rank,
            world_size = register_resp.world_size,
            registered_nodes = register_resp.registered_nodes,
            job_ready = register_resp.job_ready,
            heartbeat_interval_ms = register_resp.heartbeat_interval_ms,
            lease_ttl_ms = register_resp.lease_ttl_ms,
            manifest_hash = %register_resp.manifest_hash,
            "registered with coordinator"
        );

        if args.metrics_snapshot_interval_ms > 0 {
            let metrics_clone = metrics.clone();
            let job_id = args.job_id.clone();
            let node_id = args.node_id.clone();
            let manifest_hash = manifest_hash.clone();
            let interval_ms = args.metrics_snapshot_interval_ms;
            tokio::spawn(async move {
                let mut ticker =
                    tokio::time::interval(std::time::Duration::from_millis(interval_ms));
                loop {
                    ticker.tick().await;
                    metrics_clone.snapshot(&job_id, &node_id, &manifest_hash);
                }
            });
        }

        loop {
            tokio::time::sleep(Duration::from_millis(heartbeat_interval_ms)).await;
            metrics.heartbeat_total.inc();

            let now_ms = mx8_observe::time::unix_time_ms();

            let stats = NodeStats {
                inflight_bytes: metrics.inflight_bytes.get(),
                ram_high_water_bytes: metrics.ram_high_water_bytes.get(),
                fetch_queue_depth: 0,
                decode_queue_depth: 0,
                pack_queue_depth: 0,
            };

            let res = client
                .heartbeat(HeartbeatRequest {
                    job_id: args.job_id.clone(),
                    node_id: args.node_id.clone(),
                    unix_time_ms: now_ms,
                    stats: Some(stats),
                })
                .await;

            match res {
                Ok(_) => {
                    metrics.heartbeat_ok_total.inc();
                    tracing::debug!("heartbeat ok");
                }
                Err(err) => {
                    metrics.heartbeat_err_total.inc();
                    tracing::warn!(error = %err, "heartbeat failed");
                }
            }

            if args.dev_lease_want > 0 {
                let res = client
                    .request_lease(RequestLeaseRequest {
                        job_id: args.job_id.clone(),
                        node_id: args.node_id.clone(),
                        want: args.dev_lease_want,
                    })
                    .await;

                match res {
                    Ok(resp) => {
                        let resp = resp.into_inner();
                        if resp.leases.is_empty() {
                            tracing::debug!(wait_ms = resp.wait_ms, "no lease available");
                        }

                        for lease in resp.leases {
                            if let Some(range) = &lease.range {
                                let _ = client
                                    .report_progress(ReportProgressRequest {
                                        job_id: args.job_id.clone(),
                                        node_id: args.node_id.clone(),
                                        lease_id: lease.lease_id.clone(),
                                        cursor: range.end_id,
                                        delivered_samples: range
                                            .end_id
                                            .saturating_sub(range.start_id),
                                        delivered_bytes: 0,
                                        unix_time_ms: mx8_observe::time::unix_time_ms(),
                                    })
                                    .await;
                            }
                        }
                    }
                    Err(err) => {
                        tracing::debug!(error = %err, "request_lease failed");
                    }
                }
            }
        }
    }
    .instrument(span)
    .await
}
