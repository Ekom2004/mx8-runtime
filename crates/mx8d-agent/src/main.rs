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

const DEFAULT_GRPC_MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

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

    #[arg(long, env = "MX8_PREFETCH_BATCHES", default_value_t = 1)]
    prefetch_batches: usize,

    #[arg(long, env = "MX8_MAX_QUEUE_BATCHES", default_value_t = 64)]
    max_queue_batches: usize,

    #[arg(long, env = "MX8_MAX_INFLIGHT_BYTES", default_value_t = 128 * 1024 * 1024)]
    max_inflight_bytes: u64,

    #[arg(long, env = "MX8_TARGET_BATCH_BYTES")]
    target_batch_bytes: Option<u64>,

    #[arg(long, env = "MX8_MAX_BATCH_BYTES")]
    max_batch_bytes: Option<u64>,

    #[arg(long, env = "MX8_MAX_PROCESS_RSS_BYTES", hide = true)]
    max_process_rss_bytes: Option<u64>,

    /// Artificially slow down delivery to prove backpressure + cursor semantics.
    #[arg(long, env = "MX8_SINK_SLEEP_MS", default_value_t = 0)]
    sink_sleep_ms: u64,

    /// How often to report progress while executing a lease.
    #[arg(long, env = "MX8_PROGRESS_INTERVAL_MS", default_value_t = 500)]
    progress_interval_ms: u64,

    /// gRPC max message size (both decode/encode) for manifest proxying and future APIs.
    #[arg(
        long,
        env = "MX8_GRPC_MAX_MESSAGE_BYTES",
        default_value_t = DEFAULT_GRPC_MAX_MESSAGE_BYTES
    )]
    grpc_max_message_bytes: usize,
}

#[derive(Debug, Default)]
struct AgentMetrics {
    register_total: Counter,
    heartbeat_total: Counter,
    heartbeat_ok_total: Counter,
    heartbeat_err_total: Counter,
    inflight_bytes: Gauge,
    ram_high_water_bytes: Gauge,
    active_lease_tasks: Gauge,
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
            active_lease_tasks = self.active_lease_tasks.get(),
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
    let mut client = CoordinatorClient::new(channel)
        .max_decoding_message_size(ctx.grpc_max_message_bytes)
        .max_encoding_message_size(ctx.grpc_max_message_bytes);
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
    grpc_max_message_bytes: usize,
}

#[derive(Debug, Clone)]
enum ManifestSource {
    CachedPath(Arc<PathBuf>),
    DirectStream,
}

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(raw) => matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => default,
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(raw) => raw.trim().parse::<usize>().ok().unwrap_or(default),
        Err(_) => default,
    }
}

async fn fetch_manifest_records_for_range(
    channel: Channel,
    args: &Args,
    manifest_hash: &str,
    start_id: u64,
    end_id: u64,
) -> Result<Vec<mx8_core::types::ManifestRecord>> {
    let max_line_bytes = env_usize("MX8_AGENT_MANIFEST_STREAM_MAX_LINE_BYTES", 8 * 1024 * 1024);
    let req = GetManifestRequest {
        job_id: args.job_id.clone(),
        manifest_hash: manifest_hash.to_string(),
    };
    let mut client = CoordinatorClient::new(channel)
        .max_decoding_message_size(args.grpc_max_message_bytes)
        .max_encoding_message_size(args.grpc_max_message_bytes);

    let stream_resp = client.get_manifest_stream(req.clone()).await;
    match stream_resp {
        Ok(resp) => {
            let mut stream = resp.into_inner();
            let mut parser = mx8_runtime::pipeline::ManifestRangeStreamParser::new(
                start_id,
                end_id,
                max_line_bytes,
            )?;
            let mut stream_schema: Option<u32> = None;
            while let Some(chunk) = stream.message().await? {
                if let Some(existing) = stream_schema {
                    if chunk.schema_version != existing {
                        tracing::error!(
                            target: "mx8_proof",
                            event = "manifest_schema_mismatch",
                            job_id = %args.job_id,
                            node_id = %args.node_id,
                            manifest_hash = %manifest_hash,
                            expected_schema_version = existing,
                            got_schema_version = chunk.schema_version,
                            "manifest stream schema mismatch"
                        );
                        anyhow::bail!("GetManifestStream returned inconsistent schema_version");
                    }
                } else {
                    stream_schema = Some(chunk.schema_version);
                }
                parser.push_chunk(chunk.data.as_slice())?;
            }
            anyhow::ensure!(
                stream_schema.is_some(),
                "GetManifestStream returned no chunks (empty manifest)"
            );
            match parser.finish() {
                Ok(records) => Ok(records),
                Err(err) => {
                    tracing::error!(
                        target: "mx8_proof",
                        event = "manifest_stream_truncated",
                        job_id = %args.job_id,
                        node_id = %args.node_id,
                        manifest_hash = %manifest_hash,
                        start_id = start_id,
                        end_id = end_id,
                        error = %err,
                        "manifest stream parse failed (fail-closed)"
                    );
                    Err(err)
                }
            }
        }
        Err(status) => {
            if status.code() != tonic::Code::Unimplemented {
                anyhow::bail!("GetManifestStream failed: {status}");
            }
            let resp = client.get_manifest(req).await?.into_inner();
            mx8_runtime::pipeline::load_manifest_records_range_from_read(
                std::io::Cursor::new(resp.manifest_bytes),
                start_id,
                end_id,
            )
        }
    }
}

async fn run_lease(
    channel: Channel,
    pipeline: Arc<Pipeline>,
    manifest_source: ManifestSource,
    args: &Args,
    manifest_hash: &str,
    metrics: &Arc<AgentMetrics>,
    lease: Lease,
) -> Result<()> {
    let Some(range) = &lease.range else {
        anyhow::bail!("lease missing range");
    };

    let direct_records = if matches!(manifest_source, ManifestSource::DirectStream) {
        Some(
            fetch_manifest_records_for_range(
                channel.clone(),
                args,
                manifest_hash,
                range.start_id,
                range.end_id,
            )
            .await?,
        )
    } else {
        None
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
        source = match manifest_source {
            ManifestSource::CachedPath(_) => "cached_path",
            ManifestSource::DirectStream => "direct_stream",
        },
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
            grpc_max_message_bytes: args.grpc_max_message_bytes,
        },
        progress.clone(),
        done_rx,
    ));

    match manifest_source {
        ManifestSource::CachedPath(path) => {
            pipeline
                .run_manifest_path_range(sink, path.as_path(), range.start_id, range.end_id)
                .await?;
        }
        ManifestSource::DirectStream => {
            let records = direct_records.ok_or_else(|| {
                anyhow::anyhow!("direct stream mode selected but records were not loaded")
            })?;
            pipeline.run_manifest_records(sink, records).await?;
        }
    }

    let _ = done_tx.send(());
    let _ = reporter.await;

    let cursor = progress.cursor();
    let delivered_samples = progress.delivered_samples();
    let delivered_bytes = progress.delivered_bytes();

    metrics.leases_completed_total.inc();
    metrics.delivered_samples_total.inc_by(delivered_samples);
    metrics.delivered_bytes_total.inc_by(delivered_bytes);

    let mut client = CoordinatorClient::new(channel)
        .max_decoding_message_size(args.grpc_max_message_bytes)
        .max_encoding_message_size(args.grpc_max_message_bytes);
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
        let mut client = CoordinatorClient::new(channel.clone())
            .max_decoding_message_size(args.grpc_max_message_bytes)
            .max_encoding_message_size(args.grpc_max_message_bytes);

        let metrics = Arc::new(AgentMetrics::default());
        metrics.register_total.inc();
        let caps = Some(NodeCaps {
            max_fetch_concurrency: 32,
            max_decode_concurrency: 8,
            max_inflight_bytes: 1 << 30,
            max_ram_bytes: 4 << 30,
        });

        let mut register_resp = client
            .register_node(RegisterNodeRequest {
                job_id: args.job_id.clone(),
                node_id: args.node_id.clone(),
                caps: caps.clone(),
            })
            .await?
            .into_inner();

        while !register_resp.job_ready {
            tokio::time::sleep(Duration::from_millis(200)).await;
            register_resp = client
                .register_node(RegisterNodeRequest {
                    job_id: args.job_id.clone(),
                    node_id: args.node_id.clone(),
                    caps: caps.clone(),
                })
                .await?
                .into_inner();
        }

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

        let manifest_stream_direct = env_bool("MX8_AGENT_MANIFEST_STREAM_DIRECT", false);
        let manifest_source = if manifest_stream_direct {
            info!(
                target: "mx8_proof",
                event = "manifest_ingest_mode_selected",
                mode = "direct_stream",
                "using direct stream manifest ingest path"
            );
            ManifestSource::DirectStream
        } else {
            // M3: manifest proxy fallback (control-plane only).
            // Cache the pinned manifest locally so future runtime stages can load it without
            // needing to talk to the coordinator.
            let mut manifest_path: Option<PathBuf> = None;
            match fetch_and_cache_manifest(&mut client, &args, &manifest_hash).await {
                Ok(Some(path)) => {
                    manifest_path = Some(path);
                }
                Ok(None) => {}
                Err(err) => {
                    if args.dev_lease_want > 0 {
                        return Err(anyhow::anyhow!("{err}"));
                    }
                    warn!(error = %err, "GetManifest failed (continuing)");
                }
            };

            let manifest_path = manifest_path.ok_or_else(|| {
                anyhow::anyhow!("manifest unavailable; cannot run leases without GetManifest")
            })?;
            ManifestSource::CachedPath(Arc::new(manifest_path))
        };

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
        let grpc_max = args.grpc_max_message_bytes;
        tokio::spawn(async move {
            let mut client = CoordinatorClient::new(heartbeat_channel)
                .max_decoding_message_size(grpc_max)
                .max_encoding_message_size(grpc_max);
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
            let manifest_source = manifest_source.clone();
            tokio::spawn(async move {
                let want = std::cmp::max(1, args_clone.dev_lease_want);
                let grpc_max = args_clone.grpc_max_message_bytes;
                let caps = RuntimeCaps {
                    max_inflight_bytes: args_clone.max_inflight_bytes,
                    max_queue_batches: args_clone.max_queue_batches,
                    batch_size_samples: args_clone.batch_size_samples,
                    prefetch_batches: args_clone.prefetch_batches,
                    target_batch_bytes: args_clone.target_batch_bytes,
                    max_batch_bytes: args_clone.max_batch_bytes,
                    max_process_rss_bytes: args_clone.max_process_rss_bytes,
                };
                let pipeline = Arc::new(Pipeline::new(caps));

                // `want` is interpreted as "max concurrent leases per node".
                let target_concurrency = std::cmp::max(1, want as usize);
                let mut joinset: tokio::task::JoinSet<(String, anyhow::Result<()>)> =
                    tokio::task::JoinSet::new();
                let mut active: usize = 0;
                let mut next_request_at = tokio::time::Instant::now();

                loop {
                    let now = tokio::time::Instant::now();
                    if active < target_concurrency && now >= next_request_at {
                        let deficit = target_concurrency - active;
                        let deficit_u32 = u32::try_from(deficit).unwrap_or(u32::MAX);

                        let mut client = CoordinatorClient::new(lease_channel.clone())
                            .max_decoding_message_size(grpc_max)
                            .max_encoding_message_size(grpc_max);
                        let resp = client
                            .request_lease(RequestLeaseRequest {
                                job_id: args_clone.job_id.clone(),
                                node_id: args_clone.node_id.clone(),
                                want: std::cmp::max(1, deficit_u32),
                            })
                            .await;

                        let resp = match resp {
                            Ok(resp) => resp.into_inner(),
                            Err(err) => {
                                tracing::info!(error = %err, "RequestLease failed; backing off");
                                next_request_at =
                                    tokio::time::Instant::now() + Duration::from_millis(500);
                                continue;
                            }
                        };

                        if resp.leases.is_empty() {
                            let wait_ms = std::cmp::max(1, resp.wait_ms);
                            next_request_at = tokio::time::Instant::now()
                                + Duration::from_millis(wait_ms as u64);
                            continue;
                        }

                        for lease in resp.leases {
                            let pipeline = pipeline.clone();
                            let args = args_clone.clone();
                            let manifest_hash = manifest_hash_clone.clone();
                            let manifest_source = manifest_source.clone();
                            let metrics = metrics_clone.clone();
                            let ch = lease_channel.clone();
                            let lease_id = lease.lease_id.clone();

                            tracing::info!(
                                target: "mx8_proof",
                                event = "lease_task_spawned",
                                job_id = %args.job_id,
                                node_id = %args.node_id,
                                manifest_hash = %manifest_hash,
                                lease_id = %lease_id,
                                "lease task spawned"
                            );

                            joinset.spawn(async move {
                                let res = run_lease(
                                    ch,
                                    pipeline,
                                    manifest_source,
                                    &args,
                                    &manifest_hash,
                                    &metrics,
                                    lease,
                                )
                                .await;
                                (lease_id, res)
                            });
                            active = active.saturating_add(1);
                            metrics_clone
                                .active_lease_tasks
                                .set(u64::try_from(active).unwrap_or(u64::MAX));
                            if active >= target_concurrency {
                                break;
                            }
                        }

                        next_request_at = tokio::time::Instant::now();
                        continue;
                    }

                    if active == 0 {
                        tokio::time::sleep_until(next_request_at).await;
                        continue;
                    }

                    tokio::select! {
                        res = joinset.join_next() => {
                            match res {
                                Some(Ok((lease_id, Ok(())))) => {
                                    tracing::info!(
                                        target: "mx8_proof",
                                        event = "lease_task_done",
                                        job_id = %args_clone.job_id,
                                        node_id = %args_clone.node_id,
                                        manifest_hash = %manifest_hash_clone,
                                        lease_id = %lease_id,
                                        "lease task done"
                                    );
                                }
                                Some(Ok((lease_id, Err(err)))) => {
                                    tracing::error!(
                                        target: "mx8_proof",
                                        event = "lease_task_done",
                                        job_id = %args_clone.job_id,
                                        node_id = %args_clone.node_id,
                                        manifest_hash = %manifest_hash_clone,
                                        lease_id = %lease_id,
                                        error = %err,
                                        "lease task done (error)"
                                    );
                                }
                                Some(Err(err)) => {
                                    tracing::error!(error = %err, "lease task join failed");
                                }
                                None => {}
                            }
                            active = active.saturating_sub(1);
                            metrics_clone
                                .active_lease_tasks
                                .set(u64::try_from(active).unwrap_or(u64::MAX));
                        }
                        _ = tokio::time::sleep_until(next_request_at), if active < target_concurrency => {
                            // Time to ask for more leases.
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

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::{run_lease, AgentMetrics, Args, ManifestSource};
    use mx8_proto::v0::coordinator_server::{Coordinator, CoordinatorServer};
    use mx8_proto::v0::{
        GetManifestRequest, GetManifestResponse, HeartbeatRequest, HeartbeatResponse, Lease,
        ManifestChunk, RegisterNodeRequest, RegisterNodeResponse, ReportProgressRequest,
        ReportProgressResponse, RequestLeaseRequest, RequestLeaseResponse, WorkRange,
    };
    use mx8_runtime::pipeline::{Pipeline, RuntimeCaps};
    use std::path::PathBuf;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};
    use tokio_stream::wrappers::TcpListenerStream;
    use tokio_stream::Stream;
    use tonic::transport::{Channel, Server};
    use tonic::{Request, Response, Status};

    #[derive(Clone)]
    struct MockCoordinator {
        manifest_bytes: Arc<Vec<u8>>,
        reports: Arc<Mutex<Vec<ReportProgressRequest>>>,
    }

    impl MockCoordinator {
        fn new(manifest_bytes: Vec<u8>) -> Self {
            Self {
                manifest_bytes: Arc::new(manifest_bytes),
                reports: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    #[tonic::async_trait]
    impl Coordinator for MockCoordinator {
        type GetManifestStreamStream =
            Pin<Box<dyn Stream<Item = Result<ManifestChunk, Status>> + Send + 'static>>;

        async fn register_node(
            &self,
            _request: Request<RegisterNodeRequest>,
        ) -> Result<Response<RegisterNodeResponse>, Status> {
            Err(Status::unimplemented("not used by this test"))
        }

        async fn heartbeat(
            &self,
            _request: Request<HeartbeatRequest>,
        ) -> Result<Response<HeartbeatResponse>, Status> {
            Err(Status::unimplemented("not used by this test"))
        }

        async fn request_lease(
            &self,
            _request: Request<RequestLeaseRequest>,
        ) -> Result<Response<RequestLeaseResponse>, Status> {
            Err(Status::unimplemented("not used by this test"))
        }

        async fn report_progress(
            &self,
            request: Request<ReportProgressRequest>,
        ) -> Result<Response<ReportProgressResponse>, Status> {
            let mut guard = self
                .reports
                .lock()
                .map_err(|_| Status::internal("report lock poisoned"))?;
            guard.push(request.into_inner());
            Ok(Response::new(ReportProgressResponse {}))
        }

        async fn get_manifest(
            &self,
            _request: Request<GetManifestRequest>,
        ) -> Result<Response<GetManifestResponse>, Status> {
            Ok(Response::new(GetManifestResponse {
                manifest_bytes: self.manifest_bytes.as_ref().clone(),
                schema_version: mx8_core::types::MANIFEST_SCHEMA_VERSION,
            }))
        }

        async fn get_manifest_stream(
            &self,
            _request: Request<GetManifestRequest>,
        ) -> Result<Response<Self::GetManifestStreamStream>, Status> {
            let chunks = self
                .manifest_bytes
                .chunks(7)
                .map(|c| {
                    Ok(ManifestChunk {
                        data: c.to_vec(),
                        schema_version: mx8_core::types::MANIFEST_SCHEMA_VERSION,
                    })
                })
                .collect::<Vec<_>>();
            Ok(Response::new(Box::pin(tokio_stream::iter(chunks))))
        }
    }

    fn test_args(manifest_cache_dir: PathBuf) -> Args {
        Args {
            coord_url: "http://127.0.0.1:0".to_string(),
            job_id: "job".to_string(),
            node_id: "node".to_string(),
            manifest_cache_dir,
            metrics_snapshot_interval_ms: 0,
            dev_lease_want: 0,
            batch_size_samples: 2,
            prefetch_batches: 1,
            max_queue_batches: 8,
            max_inflight_bytes: 8 * 1024 * 1024,
            target_batch_bytes: None,
            max_batch_bytes: None,
            max_process_rss_bytes: None,
            sink_sleep_ms: 0,
            progress_interval_ms: 60_000,
            grpc_max_message_bytes: 64 * 1024 * 1024,
        }
    }

    fn test_pipeline() -> Arc<Pipeline> {
        Arc::new(Pipeline::new(RuntimeCaps {
            max_inflight_bytes: 8 * 1024 * 1024,
            max_queue_batches: 8,
            batch_size_samples: 2,
            prefetch_batches: 1,
            target_batch_bytes: None,
            max_batch_bytes: None,
            max_process_rss_bytes: None,
        }))
    }

    fn report_summary(
        reports: &[ReportProgressRequest],
        lease_id: &str,
    ) -> Option<(u64, u64, u64)> {
        let mut best: Option<(u64, u64, u64)> = None;
        for r in reports.iter().filter(|r| r.lease_id == lease_id) {
            let cand = (r.cursor, r.delivered_samples, r.delivered_bytes);
            if best.map(|b| cand.0 >= b.0).unwrap_or(true) {
                best = Some(cand);
            }
        }
        best
    }

    #[test]
    fn manifest_stream_truncated_fails_closed() {
        let mut parser =
            mx8_runtime::pipeline::ManifestRangeStreamParser::new(0, 2, 1024).expect("parser init");
        parser
            .push_chunk(
                format!(
                    "schema_version={}\n0\tloc0\t\t\t\n1\t",
                    mx8_core::types::MANIFEST_SCHEMA_VERSION
                )
                .as_bytes(),
            )
            .expect("push");
        let err = parser.finish().unwrap_err();
        assert!(err.to_string().contains("expected at least 2 columns"));
    }

    #[test]
    fn manifest_stream_schema_mismatch_hard_fails() {
        let mut parser =
            mx8_runtime::pipeline::ManifestRangeStreamParser::new(0, 1, 1024).expect("parser init");
        let mut stream_schema: Option<u32> = None;

        let chunks = vec![
            (
                mx8_core::types::MANIFEST_SCHEMA_VERSION,
                format!(
                    "schema_version={}\n",
                    mx8_core::types::MANIFEST_SCHEMA_VERSION
                )
                .into_bytes(),
            ),
            (
                mx8_core::types::MANIFEST_SCHEMA_VERSION + 1,
                b"0\tloc0\t\t\t\n".to_vec(),
            ),
        ];

        let mut got_mismatch = false;
        for (schema, payload) in chunks {
            if let Some(existing) = stream_schema {
                if schema != existing {
                    got_mismatch = true;
                    break;
                }
            } else {
                stream_schema = Some(schema);
            }
            parser.push_chunk(payload.as_slice()).expect("push chunk");
        }
        assert!(got_mismatch, "schema mismatch should hard-fail");
    }

    #[test]
    fn manifest_stream_parity_with_cached_path() {
        let manifest = format!(
            "schema_version={}\n0\tloc0\t\t\t\n1\tloc1\t\t\t\n2\tloc2\t\t\t\n3\tloc3\t\t\t\n",
            mx8_core::types::MANIFEST_SCHEMA_VERSION
        )
        .into_bytes();

        let full = mx8_runtime::pipeline::load_manifest_records_from_read(std::io::Cursor::new(
            manifest.clone(),
        ))
        .expect("full parse");
        let expected = full[1..3].to_vec();

        let mut parser =
            mx8_runtime::pipeline::ManifestRangeStreamParser::new(1, 3, 1024).expect("parser init");
        for ch in manifest.chunks(3) {
            parser.push_chunk(ch).expect("push chunk");
        }
        let got = parser.finish().expect("finish parse");
        assert_eq!(got, expected);
    }

    #[test]
    fn manifest_stream_backpressure_bounded() {
        let mut parser =
            mx8_runtime::pipeline::ManifestRangeStreamParser::new(0, 1, 8).expect("parser init");
        let err = parser.push_chunk(b"schema_version=1").unwrap_err();
        assert!(err.to_string().contains("manifest line exceeds max bytes"));
    }

    #[tokio::test]
    async fn lease_manifest_stream_parity_with_cached_path() -> anyhow::Result<()> {
        let mut root = std::env::temp_dir();
        root.push(format!(
            "mx8-agent-lease-parity-{}-{}",
            std::process::id(),
            mx8_observe::time::unix_time_ms()
        ));
        std::fs::create_dir_all(&root)?;

        let mut manifest = String::new();
        manifest.push_str(&format!(
            "schema_version={}\n",
            mx8_core::types::MANIFEST_SCHEMA_VERSION
        ));
        for i in 0..5u64 {
            let p = root.join(format!("sample-{i}.bin"));
            std::fs::write(&p, [i as u8, (i.wrapping_mul(2)) as u8])?;
            manifest.push_str(&format!("{i}\t{}\t\t\t\n", p.display()));
        }
        let manifest_bytes = manifest.into_bytes();
        let manifest_path = root.join("manifest.tsv");
        std::fs::write(&manifest_path, &manifest_bytes)?;

        let mock = MockCoordinator::new(manifest_bytes.clone());
        let reports = mock.reports.clone();

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let incoming = TcpListenerStream::new(listener);
        let server = tokio::spawn(async move {
            Server::builder()
                .add_service(CoordinatorServer::new(mock))
                .serve_with_incoming(incoming)
                .await
        });

        let channel = Channel::from_shared(format!("http://{}", addr))?
            .connect()
            .await?;
        let args = test_args(root.clone());
        let metrics = Arc::new(AgentMetrics::default());
        let pipeline = test_pipeline();
        let range = WorkRange {
            start_id: 1,
            end_id: 4,
            epoch: 1,
            seed: 0,
        };
        let cached_lease = Lease {
            lease_id: "lease-cached".to_string(),
            node_id: args.node_id.clone(),
            range: Some(range.clone()),
            cursor: range.start_id,
            expires_unix_time_ms: 0,
        };
        let stream_lease = Lease {
            lease_id: "lease-stream".to_string(),
            node_id: args.node_id.clone(),
            range: Some(range.clone()),
            cursor: range.start_id,
            expires_unix_time_ms: 0,
        };
        let manifest_hash = "manifest-h";

        run_lease(
            channel.clone(),
            pipeline.clone(),
            ManifestSource::CachedPath(Arc::new(manifest_path)),
            &args,
            manifest_hash,
            &metrics,
            cached_lease,
        )
        .await?;

        run_lease(
            channel.clone(),
            pipeline,
            ManifestSource::DirectStream,
            &args,
            manifest_hash,
            &metrics,
            stream_lease,
        )
        .await?;

        let snapshot = reports
            .lock()
            .map_err(|_| anyhow::anyhow!("report lock poisoned"))?
            .clone();
        let cached = report_summary(&snapshot, "lease-cached")
            .ok_or_else(|| anyhow::anyhow!("missing cached-path progress reports"))?;
        let direct = report_summary(&snapshot, "lease-stream")
            .ok_or_else(|| anyhow::anyhow!("missing direct-stream progress reports"))?;

        assert_eq!(cached, direct);
        assert_eq!(cached.0, range.end_id);
        assert_eq!(cached.1, range.end_id - range.start_id);
        assert_eq!(cached.2, 6);

        server.abort();
        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }
}

async fn fetch_and_cache_manifest(
    client: &mut CoordinatorClient<Channel>,
    args: &Args,
    manifest_hash: &str,
) -> Result<Option<PathBuf>> {
    let req = GetManifestRequest {
        job_id: args.job_id.clone(),
        manifest_hash: manifest_hash.to_string(),
    };

    let path = args.manifest_cache_dir.join(manifest_hash);
    let stream_resp = client.get_manifest_stream(req.clone()).await;
    let (manifest_bytes, schema_version, manifest_bytes_logged) = match stream_resp {
        Ok(resp) => {
            let mut stream = resp.into_inner();
            let mut schema_version: Option<u32> = None;
            let mut streamed_bytes: u64 = 0;
            let parent = path.parent().unwrap_or(path.as_path());
            std::fs::create_dir_all(parent)?;
            let mut tmp = path.clone();
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
            let mut tmp_file = std::fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&tmp)?;
            while let Some(chunk) = stream.message().await? {
                if let Some(existing) = schema_version {
                    if chunk.schema_version != existing {
                        return Err(anyhow::anyhow!(
                            "GetManifestStream returned inconsistent schema_version"
                        ));
                    }
                } else {
                    schema_version = Some(chunk.schema_version);
                }
                use std::io::Write;
                tmp_file.write_all(chunk.data.as_slice())?;
                streamed_bytes = streamed_bytes.saturating_add(chunk.data.len() as u64);
            }
            let schema_version = schema_version.ok_or_else(|| {
                anyhow::anyhow!("GetManifestStream returned no chunks (empty manifest)")
            })?;
            use std::io::Write;
            tmp_file.flush()?;
            tmp_file.sync_all()?;
            std::fs::rename(&tmp, &path)?;
            (None, schema_version, streamed_bytes)
        }
        Err(status) => {
            if status.code() != tonic::Code::Unimplemented {
                return Err(anyhow::anyhow!("GetManifestStream failed: {status}"));
            }
            let resp = client.get_manifest(req).await?.into_inner();
            let logged = resp.manifest_bytes.len() as u64;
            (Some(resp.manifest_bytes), resp.schema_version, logged)
        }
    };

    if let Some(bytes) = manifest_bytes {
        if let Err(err) = write_atomic(&path, bytes.as_slice()) {
            warn!(error = %err, path = %path.display(), "failed to cache manifest");
        } else {
            info!(
                target: "mx8_proof",
                event = "manifest_cached",
                job_id = %args.job_id,
                node_id = %args.node_id,
                manifest_hash = %manifest_hash,
                schema_version = schema_version,
                manifest_bytes = manifest_bytes_logged,
                path = %path.display(),
                source = "unary_fallback",
                "cached manifest"
            );
        }
    } else {
        info!(
            target: "mx8_proof",
            event = "manifest_cached",
            job_id = %args.job_id,
            node_id = %args.node_id,
            manifest_hash = %manifest_hash,
            schema_version = schema_version,
            manifest_bytes = manifest_bytes_logged,
            path = %path.display(),
            source = "stream",
            "cached manifest"
        );
    }
    Ok(Some(path))
}
