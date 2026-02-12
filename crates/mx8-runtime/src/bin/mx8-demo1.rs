#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;
use tokio::signal;
use tracing::{info, info_span, warn, Instrument};

use mx8_observe::metrics::{Counter, Gauge};
use mx8_runtime::pipeline::{Pipeline, RuntimeCaps, RuntimeMetrics};
use mx8_runtime::sink::Sink;
use mx8_runtime::types::Batch;

#[derive(Debug, Parser)]
#[command(name = "mx8-demo1")]
struct Args {
    #[arg(long, env = "MX8_JOB_ID", default_value = "devjob")]
    job_id: String,

    #[arg(long, env = "MX8_NODE_ID", default_value = "local")]
    node_id: String,

    #[arg(long, env = "MX8_MANIFEST_HASH", default_value = "dev")]
    manifest_hash: String,

    /// Optional: use a cached manifest (from mx8d-agent) instead of synthetic data.
    ///
    /// If set, `MX8_MANIFEST_HASH` is used as the cache key (filename).
    #[arg(long, env = "MX8_MANIFEST_CACHE_DIR")]
    manifest_cache_dir: Option<std::path::PathBuf>,

    #[arg(long, env = "MX8_TOTAL_SAMPLES", default_value_t = 1_000_000)]
    total_samples: u64,

    #[arg(long, env = "MX8_BYTES_PER_SAMPLE", default_value_t = 1024)]
    bytes_per_sample: usize,

    #[arg(long, env = "MX8_BATCH_SIZE_SAMPLES", default_value_t = 512)]
    batch_size_samples: usize,

    #[arg(long, env = "MX8_MAX_QUEUE_BATCHES", default_value_t = 64)]
    max_queue_batches: usize,

    #[arg(long, env = "MX8_MAX_INFLIGHT_BYTES", default_value_t = 128 * 1024 * 1024)]
    max_inflight_bytes: u64,

    /// Artificially slow down delivery to prove backpressure + RAM caps.
    #[arg(long, env = "MX8_SINK_SLEEP_MS", default_value_t = 0)]
    sink_sleep_ms: u64,

    /// Periodically emit a metrics snapshot (0 disables).
    #[arg(long, env = "MX8_METRICS_SNAPSHOT_INTERVAL_MS", default_value_t = 1000)]
    metrics_snapshot_interval_ms: u64,
}

struct SlowSink {
    sleep: Duration,
    delivered_batches_total: Counter,
    delivered_samples_total: Counter,
    delivered_bytes_total: Counter,
    last_payload_bytes: Gauge,
}

impl SlowSink {
    fn new(sleep: Duration) -> Self {
        Self {
            sleep,
            delivered_batches_total: Counter::default(),
            delivered_samples_total: Counter::default(),
            delivered_bytes_total: Counter::default(),
            last_payload_bytes: Gauge::default(),
        }
    }
}

impl Sink for SlowSink {
    fn deliver(&self, batch: Batch) -> Result<()> {
        if !self.sleep.is_zero() {
            std::thread::sleep(self.sleep);
        }
        self.delivered_batches_total.inc();
        self.delivered_samples_total
            .inc_by(batch.sample_count() as u64);
        self.delivered_bytes_total
            .inc_by(batch.payload_len() as u64);
        self.last_payload_bytes.set(batch.payload_len() as u64);
        Ok(())
    }
}

fn emit_runtime_metrics_snapshot(
    metrics: &RuntimeMetrics,
    sink: &SlowSink,
    job_id: &str,
    node_id: &str,
    manifest_hash: &str,
) {
    tracing::info!(
        target: "mx8_metrics",
        job_id = %job_id,
        node_id = %node_id,
        manifest_hash = %manifest_hash,
        delivered_batches_total = metrics.delivered_batches_total.get(),
        delivered_samples_total = metrics.delivered_samples_total.get(),
        inflight_bytes = metrics.inflight_bytes.get(),
        inflight_bytes_high_water = metrics.inflight_bytes_high_water.get(),
        sink_delivered_batches_total = sink.delivered_batches_total.get(),
        sink_delivered_samples_total = sink.delivered_samples_total.get(),
        sink_delivered_bytes_total = sink.delivered_bytes_total.get(),
        sink_last_payload_bytes = sink.last_payload_bytes.get(),
        "metrics"
    );
}

#[tokio::main]
async fn main() -> Result<()> {
    mx8_observe::logging::init_tracing();
    let args = Args::parse();

    let span = info_span!(
        "mx8-demo1",
        job_id = %args.job_id,
        node_id = %args.node_id,
        manifest_hash = %args.manifest_hash,
        total_samples = args.total_samples,
        bytes_per_sample = args.bytes_per_sample,
        batch_size_samples = args.batch_size_samples,
        max_queue_batches = args.max_queue_batches,
        max_inflight_bytes = args.max_inflight_bytes,
        sink_sleep_ms = args.sink_sleep_ms,
    );

    async move {
        let caps = RuntimeCaps {
            max_inflight_bytes: args.max_inflight_bytes,
            max_queue_batches: args.max_queue_batches,
            batch_size_samples: args.batch_size_samples,
        };

        let pipeline = Pipeline::new(caps);
        let metrics = pipeline.metrics();

        let sink = Arc::new(SlowSink::new(Duration::from_millis(args.sink_sleep_ms)));

        let metrics_task = if args.metrics_snapshot_interval_ms > 0 {
            let interval_ms = std::cmp::max(1, args.metrics_snapshot_interval_ms);
            let metrics = metrics.clone();
            let sink = sink.clone();
            let job_id = args.job_id.clone();
            let node_id = args.node_id.clone();
            let manifest_hash = args.manifest_hash.clone();
            Some(tokio::spawn(async move {
                let mut ticker = tokio::time::interval(Duration::from_millis(interval_ms));
                loop {
                    ticker.tick().await;
                    emit_runtime_metrics_snapshot(
                        &metrics,
                        &sink,
                        &job_id,
                        &node_id,
                        &manifest_hash,
                    );
                }
            }))
        } else {
            None
        };

        let start = Instant::now();
        if args.manifest_cache_dir.is_some() {
            info!("starting bounded pipeline (manifest cache)");
        } else {
            info!("starting bounded pipeline (synthetic)");
        }

        tokio::select! {
            res = async {
                if let Some(dir) = &args.manifest_cache_dir {
                    pipeline
                        .run_manifest_cache_dir(sink.clone(), dir, &args.manifest_hash)
                        .await
                } else {
                    pipeline
                        .run_synthetic(sink.clone(), args.total_samples, args.bytes_per_sample)
                        .await
                }
            } => {
                res?;
            }
            _ = signal::ctrl_c() => {
                warn!("ctrl-c received; exiting");
            }
        }

        if let Some(task) = metrics_task {
            task.abort();
        }

        let elapsed = start.elapsed();
        emit_runtime_metrics_snapshot(
            &metrics,
            &sink,
            &args.job_id,
            &args.node_id,
            &args.manifest_hash,
        );

        let delivered_samples = metrics.delivered_samples_total.get();
        let throughput = if elapsed.as_secs_f64() > 0.0 {
            delivered_samples as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        info!(
            elapsed_ms = elapsed.as_millis() as u64,
            delivered_samples = delivered_samples,
            inflight_bytes_high_water = metrics.inflight_bytes_high_water.get(),
            samples_per_sec = throughput,
            "demo complete"
        );

        Ok(())
    }
    .instrument(span)
    .await
}
