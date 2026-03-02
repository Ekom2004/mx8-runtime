#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]
#![allow(unexpected_cfgs)]

#[cfg(feature = "azure")]
use anyhow::Result;

#[cfg(feature = "azure")]
use std::sync::Arc;

#[cfg(feature = "azure")]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "azure")]
use azure_storage_blobs::prelude::PublicAccess;

#[cfg(feature = "azure")]
use clap::Parser;

#[cfg(feature = "azure")]
use mx8_runtime::pipeline::{Pipeline, RuntimeCaps};

#[cfg(feature = "azure")]
use mx8_runtime::sink::Sink;

#[cfg(feature = "azure")]
use mx8_runtime::types::Batch;

#[cfg(not(feature = "azure"))]
fn main() {
    eprintln!(
        "mx8-demo5-azurite requires feature 'azure' (run with: cargo run -p mx8-runtime --features azure --bin mx8-demo5-azurite)"
    );
    std::process::exit(2);
}

#[cfg(feature = "azure")]
#[derive(Debug, Parser)]
#[command(name = "mx8-demo5-azurite")]
struct Args {
    #[arg(
        long,
        env = "MX8_AZURE_ENDPOINT_URL",
        default_value = "http://127.0.0.1:10000/devstoreaccount1"
    )]
    azure_endpoint_url: String,

    #[arg(long, env = "MX8_AZURE_CONTAINER", default_value = "mx8-demo")]
    container: String,

    #[arg(long, env = "MX8_AZURE_BLOB", default_value = "data.bin")]
    blob: String,

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

#[cfg(feature = "azure")]
#[tokio::main]
async fn main() -> Result<()> {
    mx8_observe::logging::init_tracing();
    let args = Args::parse();

    let builder = mx8_runtime::azure::client_builder_from_env_with_endpoint_override(Some(
        &args.azure_endpoint_url,
    ))?;
    let container_client = builder.clone().container_client(&args.container);

    // Best effort container creation (ignore "already exists" and continue).
    if let Err(err) = container_client
        .create()
        .public_access(PublicAccess::None)
        .await
    {
        tracing::warn!(
            ?err,
            container = args.container.as_str(),
            "create_container failed (continuing)"
        );
    }

    let total_bytes: usize =
        usize::try_from(args.total_samples.saturating_mul(args.bytes_per_sample))
            .map_err(|_| anyhow::anyhow!("total bytes overflow"))?;

    let mut data: Vec<u8> = vec![0u8; total_bytes];
    for (i, b) in data.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(17).wrapping_add(11);
    }

    builder
        .clone()
        .blob_client(&args.container, &args.blob)
        .put_block_blob(data.clone())
        .await?;

    let mut manifest = String::new();
    manifest.push_str(&format!(
        "schema_version={}\n",
        mx8_core::types::MANIFEST_SCHEMA_VERSION
    ));
    for sample_id in 0..args.total_samples {
        let off = sample_id.saturating_mul(args.bytes_per_sample);
        manifest.push_str(&format!(
            "{sample_id}\taz://{}/{}\t{off}\t{}\n",
            args.container, args.blob, args.bytes_per_sample
        ));
    }
    let manifest_bytes = manifest.into_bytes();

    let pipeline = Pipeline::new(RuntimeCaps {
        max_inflight_bytes: args.max_inflight_bytes,
        max_queue_batches: 64,
        batch_size_samples: args.batch_size_samples,
        prefetch_batches: args.prefetch_batches,
        target_batch_bytes: None,
        max_batch_bytes: None,
        max_process_rss_bytes: None,
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
        event = "azurite_gate_complete",
        endpoint = args.azure_endpoint_url.as_str(),
        container = args.container.as_str(),
        blob = args.blob.as_str(),
        delivered_samples = delivered,
        delivered_bytes = delivered_bytes,
        inflight_bytes_high_water = inflight_high,
        max_inflight_bytes = args.max_inflight_bytes,
        "azurite gate complete"
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

#[cfg(feature = "azure")]
#[derive(Default)]
struct CountingSink {
    delivered_samples: AtomicU64,
    delivered_bytes: AtomicU64,
}

#[cfg(feature = "azure")]
impl Sink for CountingSink {
    fn deliver(&self, batch: Batch) -> Result<()> {
        self.delivered_samples
            .fetch_add(batch.sample_count() as u64, Ordering::Relaxed);
        self.delivered_bytes
            .fetch_add(batch.payload_len() as u64, Ordering::Relaxed);
        Ok(())
    }
}
