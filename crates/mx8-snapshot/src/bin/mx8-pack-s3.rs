#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

#[cfg(feature = "s3")]
use anyhow::Result;

#[cfg(feature = "s3")]
use clap::Parser;

#[cfg(feature = "s3")]
use mx8_snapshot::pack_s3::{pack_s3, LabelMode, PackS3Config};

#[cfg(not(feature = "s3"))]
fn main() {
    eprintln!(
        "mx8-pack-s3 requires feature 's3' (run with: cargo run -p mx8-snapshot --features s3 --bin mx8-pack-s3 -- ...)"
    );
    std::process::exit(2);
}

#[cfg(feature = "s3")]
#[derive(Debug, Parser)]
#[command(name = "mx8-pack-s3")]
struct Args {
    /// Input dataset prefix (S3).
    ///
    /// Example: s3://my-bucket/raw/train/
    #[arg(long, env = "MX8_PACK_IN")]
    pack_in: String,

    /// Output dataset prefix (S3).
    ///
    /// Example: s3://my-bucket/mx8/train/
    #[arg(long, env = "MX8_PACK_OUT")]
    pack_out: String,

    /// Target shard size in MiB (uncompressed tar).
    #[arg(long, env = "MX8_PACK_SHARD_MB", default_value_t = 512)]
    shard_mb: u64,

    /// Label mode: auto|none|imagefolder.
    ///
    /// ImageFolder interprets keys as: prefix/<label>/<file...>
    #[arg(long, env = "MX8_S3_LABEL_MODE", default_value = "auto")]
    label_mode: String,

    /// If set, fail unless every object matches ImageFolder layout.
    #[arg(long, env = "MX8_PACK_REQUIRE_LABELS", default_value_t = false)]
    require_labels: bool,
}

#[cfg(feature = "s3")]
fn parse_label_mode(s: &str) -> Result<LabelMode> {
    let s = s.trim().to_ascii_lowercase();
    let mode = match s.as_str() {
        "none" | "off" | "false" | "0" => LabelMode::None,
        "imagefolder" | "image_folder" | "image-folder" => LabelMode::ImageFolder,
        "auto" | "" => LabelMode::Auto,
        _ => anyhow::bail!("invalid label mode {s:?} (expected: auto|none|imagefolder)"),
    };
    Ok(mode)
}

#[cfg(feature = "s3")]
#[tokio::main]
async fn main() -> Result<()> {
    mx8_observe::logging::init_tracing();

    let args = Args::parse();
    let label_mode = parse_label_mode(&args.label_mode)?;

    let res = pack_s3(PackS3Config {
        pack_in: args.pack_in,
        pack_out: args.pack_out,
        shard_mb: args.shard_mb,
        label_mode,
        require_labels: args.require_labels,
    })
    .await?;

    println!(
        "samples={} shards={} manifest_key={} manifest_hash={}",
        res.samples, res.shards, res.manifest_key, res.manifest_hash
    );
    Ok(())
}
