#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use tonic::transport::Channel;
use tracing::{info, info_span, warn, Instrument};

use mx8_proto::v0::coordinator_client::CoordinatorClient;
use mx8_proto::v0::GetManifestRequest;
use mx8_proto::v0::HeartbeatRequest;
use mx8_proto::v0::Lease;
use mx8_proto::v0::NodeCaps;
use mx8_proto::v0::NodeStats;
use mx8_proto::v0::RegisterNodeRequest;
use mx8_proto::v0::ReportProgressRequest;
use mx8_proto::v0::RequestLeaseRequest;

use mx8_observe::metrics::{Counter, Gauge};

use mx8_runtime::pipeline::{Pipeline, RuntimeCaps};
use mx8_runtime::sink::Sink;
use mx8_runtime::types::Batch;

#[derive(Debug, Parser, Clone)]
#[command(name = "mx8d-agent")]
struct Args {
    /// Coordinator address, e.g. http://127.0.0.1:50051
    #[arg(long, env = "MX8_COORD_URL", default_value = "http://127.0.0.1:50051")]
    coord_url: String,

    #[arg(long, env = "MX8_JOB_ID", default_value = "local-job")]
    job_id: String,

    #[arg(long, env = "MX8_NODE_ID", default_value = "local-node")]
    node_id: String,

    /// Where to cache `GetManifest` bytes locally (control-plane only).
    #[arg(
        long,
        env = "MX8_MANIFEST_CACHE_DIR",
        default_value = "/tmp/mx8/manifests"
    )]
    manifest_cache_dir: PathBuf,

    /// Optional: periodically emit a metrics snapshot to logs.
    #[arg(long, env = "MX8_METRICS_SNAPSHOT_INTERVAL_MS", default_value_t = 0)]
    metrics_snapshot_interval_ms: u64,

    /// Development-only: request leases continuously with this `want` value.
    /// Set to 0 to disable.
    #[arg(long, env = "MX8_DEV_LEASE_WANT", default_value_t = 0)]
    dev_lease_want: u32,

    #[arg(long, env = "MX8_BATCH_SIZE_SAMPLES", default_value_t = 512)]
    batch_size_samples: usize,

    #[arg(long, env = "MX8_MAX_QUEUE_BATCHES", default_value_t = 64)]
    max_queue_batches: usize,

    #[arg(long, env = "MX8_MAX_INFLIGHT_BYTES", default_value_t = 128 * 1024 * 1024)]
    max_inflight_bytes: u64,

    /// Artificially slow down delivery to prove backpressure + cursor semantics.
    #[arg(long, env = "MX8_SINK_SLEEP_MS", default_value_t = 0)]
    sink_sleep_ms: u64,

    /// How often to report progress while executing a lease.
    #[arg(long, env = "MX8_PROGRESS_INTERVAL_MS", default_value_t = 500)]
    progress_interval_ms: u64,
}

#[derive(Debug, Default)]
struct AgentMetrics {
    register_total: Counter,
    heartbeat_total: Counter,
    heartbeat_ok_total: Counter,
    heartbeat_err_total: Counter,
    inflight_bytes: Gauge,
    ram_high_water_bytes: Gauge,
    leases_started_total: Counter,
    leases_completed_total: Counter,
    delivered_samples_total: Counter,
    delivered_bytes_total: Counter,
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
            leases_started_total = self.leases_started_total.get(),
            leases_completed_total = self.leases_completed_total.get(),
            delivered_samples_total = self.delivered_samples_total.get(),
            delivered_bytes_total = self.delivered_bytes_total.get(),
            "metrics"
        );
    }
}

fn atomic_fetch_max_u64(dst: &AtomicU64, value: u64) {
    let mut cur = dst.load(Ordering::Relaxed);
    while cur < value {
        match dst.compare_exchange(cur, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(next) => cur = next,
        }
    }
}

fn write_atomic(path: &std::path::Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;

    let parent = path.parent().unwrap_or(path);
    std::fs::create_dir_all(parent)?;

    let mut tmp = path.to_path_buf();
    let suffix = format!(
        "tmp.{}.{}",
        std::process::id(),
        mx8_observe::time::unix_time_ms()
    );
    let file_name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("manifest");
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

#[derive(Debug)]
struct LeaseProgress {
    start_id: u64,
    end_id: u64,
    cursor: AtomicU64,
    delivered_samples: AtomicU64,
    delivered_bytes: AtomicU64,
}

impl LeaseProgress {
    fn new(start_id: u64, end_id: u64) -> Self {
        Self {
            start_id,
            end_id,
            cursor: AtomicU64::new(start_id),
            delivered_samples: AtomicU64::new(0),
            delivered_bytes: AtomicU64::new(0),
        }
    }

    fn cursor(&self) -> u64 {
        self.cursor.load(Ordering::Acquire)
    }

    fn delivered_samples(&self) -> u64 {
        self.delivered_samples.load(Ordering::Relaxed)
    }

    fn delivered_bytes(&self) -> u64 {
        self.delivered_bytes.load(Ordering::Relaxed)
    }
}

#[derive(Debug)]
struct LeaseProgressSink {
    progress: Arc<LeaseProgress>,
    sleep: Duration,
}

impl Sink for LeaseProgressSink {
    fn deliver(&self, batch: Batch) -> Result<()> {
        if !self.sleep.is_zero() {
            std::thread::sleep(self.sleep);
        }

        self.progress
            .delivered_samples
            .fetch_add(batch.sample_count() as u64, Ordering::Relaxed);
        self.progress
            .delivered_bytes
            .fetch_add(batch.payload_len() as u64, Ordering::Relaxed);

        let max_id = batch
            .sample_ids
            .iter()
            .copied()
            .max()
            .unwrap_or(self.progress.start_id);
        let next_cursor = max_id
            .saturating_add(1)
            .clamp(self.progress.start_id, self.progress.end_id);
        atomic_fetch_max_u64(&self.progress.cursor, next_cursor);
        Ok(())
    }
}

async fn report_progress_loop(
    channel: Channel,
    ctx: ProgressLoopCtx,
    progress: Arc<LeaseProgress>,
    mut done: tokio::sync::oneshot::Receiver<()>,
) {
    let mut client = CoordinatorClient::new(channel);
    let mut ticker = tokio::time::interval(std::cmp::max(Duration::from_millis(1), ctx.interval));
    loop {
        tokio::select! {
            _ = ticker.tick() => {
                let cursor = progress.cursor();
                let delivered_samples = progress.delivered_samples();
                let delivered_bytes = progress.delivered_bytes();
                let req = ReportProgressRequest {
                    job_id: ctx.job_id.clone(),
                    node_id: ctx.node_id.clone(),
                    lease_id: ctx.lease_id.clone(),
                    cursor,
                    delivered_samples,
                    delivered_bytes,
                    unix_time_ms: mx8_observe::time::unix_time_ms(),
                };
                if let Err(err) = client.report_progress(req).await {
                    warn!(error = %err, lease_id = %ctx.lease_id, cursor = cursor, "ReportProgress failed");
                } else {
                    tracing::debug!(lease_id = %ctx.lease_id, cursor = cursor, "ReportProgress ok");
                }
            }
            _ = &mut done => {
                tracing::info!(
                    target: "mx8_proof",
                    event = "lease_progress_loop_done",
                    job_id = %ctx.job_id,
                    node_id = %ctx.node_id,
                    manifest_hash = %ctx.manifest_hash,
                    lease_id = %ctx.lease_id,
                    cursor = progress.cursor(),
                    delivered_samples = progress.delivered_samples(),
                    delivered_bytes = progress.delivered_bytes(),
                    "lease progress loop done"
                );
                return;
            }
        }
    }
}

#[derive(Debug, Clone)]
struct ProgressLoopCtx {
    job_id: String,
    node_id: String,
    manifest_hash: String,
    lease_id: String,
    interval: Duration,
}

async fn run_lease(
    channel: Channel,
    args: &Args,
    manifest_hash: &str,
    metrics: &Arc<AgentMetrics>,
    lease: Lease,
) -> Result<()> {
    let Some(range) = &lease.range else {
        anyhow::bail!("lease missing range");
    };

    metrics.leases_started_total.inc();
    tracing::info!(
        target: "mx8_proof",
        event = "lease_started",
        job_id = %args.job_id,
        node_id = %args.node_id,
        manifest_hash = %manifest_hash,
        lease_id = %lease.lease_id,
        epoch = range.epoch,
        start_id = range.start_id,
        end_id = range.end_id,
        "lease started"
    );

    let progress = Arc::new(LeaseProgress::new(range.start_id, range.end_id));
    let sink = Arc::new(LeaseProgressSink {
        progress: progress.clone(),
        sleep: Duration::from_millis(args.sink_sleep_ms),
    });

    let (done_tx, done_rx) = tokio::sync::oneshot::channel();
    let reporter = tokio::spawn(report_progress_loop(
        channel.clone(),
        ProgressLoopCtx {
            job_id: args.job_id.clone(),
            node_id: args.node_id.clone(),
            manifest_hash: manifest_hash.to_string(),
            lease_id: lease.lease_id.clone(),
            interval: Duration::from_millis(args.progress_interval_ms),
        },
        progress.clone(),
        done_rx,
    ));

    let caps = RuntimeCaps {
        max_inflight_bytes: args.max_inflight_bytes,
        max_queue_batches: args.max_queue_batches,
        batch_size_samples: args.batch_size_samples,
    };
    let pipeline = Pipeline::new(caps);

    pipeline
        .run_manifest_cache_dir_range(
            sink,
            &args.manifest_cache_dir,
            manifest_hash,
            range.start_id,
            range.end_id,
        )
        .await?;

    let _ = done_tx.send(());
    let _ = reporter.await;

    let cursor = progress.cursor();
    let delivered_samples = progress.delivered_samples();
    let delivered_bytes = progress.delivered_bytes();

    metrics.leases_completed_total.inc();
    metrics.delivered_samples_total.inc_by(delivered_samples);
    metrics.delivered_bytes_total.inc_by(delivered_bytes);

    let mut client = CoordinatorClient::new(channel);
    let _ = client
        .report_progress(ReportProgressRequest {
            job_id: args.job_id.clone(),
            node_id: args.node_id.clone(),
            lease_id: lease.lease_id.clone(),
            cursor,
            delivered_samples,
            delivered_bytes,
            unix_time_ms: mx8_observe::time::unix_time_ms(),
        })
        .await;

    tracing::info!(
        target: "mx8_proof",
        event = "lease_finished",
        job_id = %args.job_id,
        node_id = %args.node_id,
        manifest_hash = %manifest_hash,
        lease_id = %lease.lease_id,
        cursor = cursor,
        delivered_samples = delivered_samples,
        delivered_bytes = delivered_bytes,
        "lease finished"
    );

    Ok(())
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
        let channel = Channel::from_shared(args.coord_url.clone())?
            .connect()
            .await?;
        let mut client = CoordinatorClient::new(channel.clone());

        let metrics = Arc::new(AgentMetrics::default());
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

        // M3: manifest proxy fallback (control-plane only).
        // Cache the pinned manifest locally so future runtime stages can load it without
        // needing to talk to the coordinator.
        match client
            .get_manifest(GetManifestRequest {
                job_id: args.job_id.clone(),
                manifest_hash: manifest_hash.clone(),
            })
            .await
        {
            Ok(resp) => {
                let resp = resp.into_inner();
                let path = args.manifest_cache_dir.join(&manifest_hash);
                if let Err(err) = write_atomic(&path, &resp.manifest_bytes) {
                    warn!(error = %err, path = %path.display(), "failed to cache manifest");
                } else {
                    info!(
                        target: "mx8_proof",
                        event = "manifest_cached",
                        job_id = %args.job_id,
                        node_id = %args.node_id,
                        manifest_hash = %manifest_hash,
                        schema_version = resp.schema_version,
                        manifest_bytes = resp.manifest_bytes.len() as u64,
                        path = %path.display(),
                        "cached manifest"
                    );
                }
            }
            Err(err) => {
                warn!(error = %err, "GetManifest failed (continuing)");
            }
        }

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

        let heartbeat_channel = channel.clone();
        let heartbeat_job_id = args.job_id.clone();
        let heartbeat_node_id = args.node_id.clone();
        let metrics_clone = metrics.clone();
        tokio::spawn(async move {
            let mut client = CoordinatorClient::new(heartbeat_channel);
            loop {
                tokio::time::sleep(Duration::from_millis(heartbeat_interval_ms)).await;
                metrics_clone.heartbeat_total.inc();
                let now_ms = mx8_observe::time::unix_time_ms();

                let stats = NodeStats {
                    inflight_bytes: metrics_clone.inflight_bytes.get(),
                    ram_high_water_bytes: metrics_clone.ram_high_water_bytes.get(),
                    fetch_queue_depth: 0,
                    decode_queue_depth: 0,
                    pack_queue_depth: 0,
                };

                let res = client
                    .heartbeat(HeartbeatRequest {
                        job_id: heartbeat_job_id.clone(),
                        node_id: heartbeat_node_id.clone(),
                        unix_time_ms: now_ms,
                        stats: Some(stats),
                    })
                    .await;

                match res {
                    Ok(_) => {
                        metrics_clone.heartbeat_ok_total.inc();
                        tracing::debug!("heartbeat ok");
                    }
                    Err(err) => {
                        metrics_clone.heartbeat_err_total.inc();
                        tracing::warn!(error = %err, "heartbeat failed");
                    }
                }
            }
        });

        if args.dev_lease_want > 0 {
            let lease_channel = channel.clone();
            let args_clone = args.clone();
            let metrics_clone = metrics.clone();
            let manifest_hash_clone = manifest_hash.clone();
            tokio::spawn(async move {
                let want = std::cmp::max(1, args_clone.dev_lease_want);
                loop {
                    let mut client = CoordinatorClient::new(lease_channel.clone());
                    let resp = client
                        .request_lease(RequestLeaseRequest {
                            job_id: args_clone.job_id.clone(),
                            node_id: args_clone.node_id.clone(),
                            want,
                        })
                        .await;

                    let resp = match resp {
                        Ok(resp) => resp.into_inner(),
                        Err(err) => {
                            tracing::info!(error = %err, "RequestLease failed; backing off");
                            tokio::time::sleep(Duration::from_millis(500)).await;
                            continue;
                        }
                    };

                    if resp.leases.is_empty() {
                        let wait_ms = std::cmp::max(1, resp.wait_ms);
                        tokio::time::sleep(Duration::from_millis(wait_ms as u64)).await;
                        continue;
                    }

                    for lease in resp.leases {
                        if let Err(err) = run_lease(
                            lease_channel.clone(),
                            &args_clone,
                            &manifest_hash_clone,
                            &metrics_clone,
                            lease,
                        )
                        .await
                        {
                            tracing::error!(error = %err, "lease execution failed");
                            tokio::time::sleep(Duration::from_millis(500)).await;
                        }
                    }
                }
            });
        }

        tokio::signal::ctrl_c().await?;
        Ok(())
    }
    .instrument(span)
    .await
}
