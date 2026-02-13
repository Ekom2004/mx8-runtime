#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;

use mx8_manifest_store::{LockOwner, ManifestStore};
use mx8_snapshot::{SnapshotResolver, SnapshotResolverConfig};

#[derive(Debug, Parser)]
#[command(name = "mx8-snapshot-resolve")]
struct Args {
    #[arg(long, env = "MX8_DATASET_LINK")]
    dataset_link: String,

    #[arg(
        long,
        env = "MX8_MANIFEST_STORE_ROOT",
        default_value = "/var/lib/mx8/manifests"
    )]
    manifest_store_root: String,

    #[arg(long, env = "MX8_DEV_MANIFEST_PATH")]
    dev_manifest_path: Option<PathBuf>,

    #[arg(long, env = "MX8_NODE_ID", default_value = "resolver")]
    node_id: String,

    #[arg(long, env = "MX8_SNAPSHOT_LOCK_STALE_MS", default_value_t = 60_000)]
    snapshot_lock_stale_ms: u64,

    #[arg(long, env = "MX8_SNAPSHOT_WAIT_TIMEOUT_MS", default_value_t = 30_000)]
    snapshot_wait_timeout_ms: u64,
}

fn main() -> Result<()> {
    mx8_observe::logging::init_tracing();
    let args = Args::parse();

    let store: Arc<dyn ManifestStore> = Arc::from(mx8_manifest_store::open_from_root(
        &args.manifest_store_root,
    )?);

    let cfg = SnapshotResolverConfig {
        lock_stale_after: Duration::from_millis(args.snapshot_lock_stale_ms),
        wait_timeout: Duration::from_millis(args.snapshot_wait_timeout_ms),
        dev_manifest_path: args.dev_manifest_path.clone(),
        ..Default::default()
    };
    let resolver = SnapshotResolver::new(store, cfg);
    let resolved = resolver.resolve(
        &args.dataset_link,
        LockOwner {
            node_id: Some(args.node_id.clone()),
        },
    )?;

    println!("manifest_hash: {}", resolved.manifest_hash.0);
    Ok(())
}
