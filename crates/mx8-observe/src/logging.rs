use tracing_subscriber::EnvFilter;

/// Initializes a `tracing_subscriber` using `MX8_LOG` first, then `RUST_LOG`, then a default.
///
/// Log field contract for MX8 daemons:
/// - Always include `job_id` and `node_id` when available.
/// - Include `manifest_hash` once resolved/pinned for the run.
/// - Include `lease_id` on any lease/progress-related event.
/// - Include `epoch` for any sharding/assignment-related event (even if 0 in early phases).
pub fn init_tracing() {
    let filter = env_filter();
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

pub fn env_filter() -> EnvFilter {
    EnvFilter::try_from_env("MX8_LOG")
        .or_else(|_| EnvFilter::try_from_default_env())
        .unwrap_or_else(|_| EnvFilter::new("info"))
}
