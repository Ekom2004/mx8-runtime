#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

#[cfg(feature = "s3")]
use anyhow::Result;

#[cfg(feature = "s3")]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "s3")]
use std::sync::Arc;

#[cfg(feature = "s3")]
use clap::Parser;

#[cfg(feature = "s3")]
use mx8_runtime::pipeline::{Pipeline, RuntimeCaps};

#[cfg(feature = "s3")]
use mx8_runtime::sink::Sink;

#[cfg(feature = "s3")]
use mx8_runtime::types::Batch;

#[cfg(feature = "s3")]
use aws_sdk_s3::primitives::ByteStream;

#[cfg(not(feature = "s3"))]
fn main() {
    eprintln!("mx8-demo4-minio requires feature 's3' (run with: cargo run -p mx8-runtime --features s3 --bin mx8-demo4-minio)");
    std::process::exit(2);
}

#[cfg(feature = "s3")]
#[derive(Debug, Parser)]
#[command(name = "mx8-demo4-minio")]
struct Args {
    #[arg(long, env = "MX8_S3_ENDPOINT_URL")]
    s3_endpoint_url: String,

    #[arg(long, env = "MX8_MINIO_BUCKET", default_value = "mx8-demo")]
    bucket: String,

    #[arg(long, env = "MX8_MINIO_KEY", default_value = "data.bin")]
    key: String,

    #[arg(long, env = "MX8_TOTAL_SAMPLES", default_value_t = 4_096)]
    total_samples: u64,

    #[arg(long, env = "MX8_BYTES_PER_SAMPLE", default_value_t = 256)]
    bytes_per_sample: u64,

    #[arg(long, env = "MX8_BATCH_SIZE_SAMPLES", default_value_t = 512)]
    batch_size_samples: usize,

    #[arg(long, env = "MX8_MAX_INFLIGHT_BYTES", default_value_t = 64 * 1024 * 1024)]
    max_inflight_bytes: u64,

    #[arg(long, env = "MX8_PREFETCH_BATCHES", default_value_t = 8)]
    prefetch_batches: usize,
}

#[cfg(feature = "s3")]
#[tokio::main]
async fn main() -> Result<()> {
    mx8_observe::logging::init_tracing();
    let args = Args::parse();

    let client = mx8_runtime::s3::client_from_env().await?;

    let total_bytes: usize =
        usize::try_from(args.total_samples.saturating_mul(args.bytes_per_sample))
            .map_err(|_| anyhow::anyhow!("total bytes overflow"))?;

    let mut data: Vec<u8> = vec![0u8; total_bytes];
    for (i, b) in data.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(31).wrapping_add(7);
    }

    // Best effort bucket creation (ignore "already exists/owned" errors).
    if let Err(err) = client.create_bucket().bucket(&args.bucket).send().await {
        tracing::warn!(
            ?err,
            bucket = args.bucket.as_str(),
            "create_bucket failed (continuing)"
        );
    }

    client
        .put_object()
        .bucket(&args.bucket)
        .key(&args.key)
        .body(ByteStream::from(data.clone()))
        .send()
        .await?;

    let mut manifest = String::new();
    manifest.push_str(&format!(
        "schema_version={}\n",
        mx8_core::types::MANIFEST_SCHEMA_VERSION
    ));
    for sample_id in 0..args.total_samples {
        let off = sample_id.saturating_mul(args.bytes_per_sample);
        manifest.push_str(&format!(
            "{sample_id}\ts3://{}/{}\t{off}\t{}\n",
            args.bucket, args.key, args.bytes_per_sample
        ));
    }
    let manifest_bytes = manifest.into_bytes();

    let pipeline = Pipeline::new(RuntimeCaps {
        max_inflight_bytes: args.max_inflight_bytes,
        max_queue_batches: 64,
        batch_size_samples: args.batch_size_samples,
        prefetch_batches: args.prefetch_batches,
    });
    let metrics = pipeline.metrics();

    let sink = Arc::new(CountingSink::default());
    pipeline
        .run_manifest_bytes(sink.clone(), &manifest_bytes)
        .await?;

    let delivered = sink.delivered_samples.load(Ordering::Relaxed);
    let delivered_bytes = sink.delivered_bytes.load(Ordering::Relaxed);
    let inflight_high = metrics.inflight_bytes_high_water.get();

    tracing::info!(
        target: "mx8_metrics",
        event = "minio_gate_complete",
        endpoint = args.s3_endpoint_url.as_str(),
        bucket = args.bucket.as_str(),
        key = args.key.as_str(),
        delivered_samples = delivered,
        delivered_bytes = delivered_bytes,
        inflight_bytes_high_water = inflight_high,
        max_inflight_bytes = args.max_inflight_bytes,
        "minio gate complete"
    );

    anyhow::ensure!(
        delivered == args.total_samples,
        "delivered_samples {} != expected {}",
        delivered,
        args.total_samples
    );
    anyhow::ensure!(
        delivered_bytes == args.total_samples.saturating_mul(args.bytes_per_sample),
        "delivered_bytes {} != expected {}",
        delivered_bytes,
        args.total_samples.saturating_mul(args.bytes_per_sample)
    );
    anyhow::ensure!(
        inflight_high <= args.max_inflight_bytes,
        "inflight high water {} exceeded cap {}",
        inflight_high,
        args.max_inflight_bytes
    );

    Ok(())
}

#[cfg(feature = "s3")]
#[derive(Default)]
struct CountingSink {
    delivered_samples: AtomicU64,
    delivered_bytes: AtomicU64,
}

#[cfg(feature = "s3")]
impl Sink for CountingSink {
    fn deliver(&self, batch: Batch) -> Result<()> {
        self.delivered_samples
            .fetch_add(batch.sample_count() as u64, Ordering::Relaxed);
        self.delivered_bytes
            .fetch_add(batch.payload_len() as u64, Ordering::Relaxed);
        Ok(())
    }
}
