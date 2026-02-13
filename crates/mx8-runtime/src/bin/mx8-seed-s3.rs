#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

#[cfg(feature = "s3")]
use anyhow::Result;

#[cfg(feature = "s3")]
use clap::Parser;

#[cfg(feature = "s3")]
use aws_sdk_s3::primitives::ByteStream;

#[cfg(not(feature = "s3"))]
fn main() {
    eprintln!(
        "mx8-seed-s3 requires feature 's3' (run with: cargo run -p mx8-runtime --features s3 --bin mx8-seed-s3 -- ...)"
    );
    std::process::exit(2);
}

#[cfg(feature = "s3")]
#[derive(Debug, Parser)]
#[command(name = "mx8-seed-s3")]
struct Args {
    /// S3 bucket name.
    #[arg(long, env = "MX8_MINIO_BUCKET", default_value = "mx8-demo")]
    bucket: String,

    /// S3 key (object path).
    #[arg(long, env = "MX8_MINIO_KEY", default_value = "data.bin")]
    key: String,

    /// Local file to upload.
    #[arg(long, env = "MX8_SEED_FILE")]
    file: String,
}

#[cfg(feature = "s3")]
#[tokio::main]
async fn main() -> Result<()> {
    mx8_observe::logging::init_tracing();
    let args = Args::parse();

    let client = mx8_runtime::s3::client_from_env().await?;

    // Best effort bucket creation (ignore "already exists/owned" errors).
    if let Err(err) = client.create_bucket().bucket(&args.bucket).send().await {
        tracing::warn!(
            ?err,
            bucket = args.bucket.as_str(),
            "create_bucket failed (continuing)"
        );
    }

    let bytes = tokio::fs::read(&args.file).await?;
    let len = bytes.len();

    client
        .put_object()
        .bucket(&args.bucket)
        .key(&args.key)
        .body(ByteStream::from(bytes))
        .send()
        .await?;

    tracing::info!(
        target: "mx8_metrics",
        event = "s3_seed_complete",
        bucket = args.bucket.as_str(),
        key = args.key.as_str(),
        bytes = len,
        "seeded s3 object"
    );

    Ok(())
}
