use super::*;

const NET_PRESSURE_SOURCE_ENV: &str = "MX8_AUTOTUNE_NET_SOURCE";

#[derive(Debug, Clone, Copy)]
pub(crate) struct NetPressureSample {
    pub(crate) ratio: f64,
    pub(crate) fresh: bool,
    pub(crate) age_ms: u64,
}

impl NetPressureSample {
    pub(crate) fn unavailable() -> Self {
        Self {
            ratio: 0.0,
            fresh: false,
            age_ms: 0,
        }
    }
}

pub(crate) trait NetPressureSource: Send + Sync {
    fn poll(&self) -> NetPressureSample;

    fn name(&self) -> &'static str {
        "noop"
    }
}

pub(crate) struct NetPressureRuntime {
    pub(crate) source: Arc<dyn NetPressureSource>,
    pub(crate) disabled_total_seed: u64,
}

#[derive(Debug, Default)]
struct NoopNetPressureSource;

impl NetPressureSource for NoopNetPressureSource {
    fn poll(&self) -> NetPressureSample {
        NetPressureSample::unavailable()
    }

    fn name(&self) -> &'static str {
        "noop"
    }
}

#[cfg(all(feature = "mx8_ebpf_net_pressure", target_os = "linux"))]
#[derive(Debug)]
struct EbpfNetPressureSource {
    started_at: Instant,
}

#[cfg(all(feature = "mx8_ebpf_net_pressure", target_os = "linux"))]
impl EbpfNetPressureSource {
    fn new() -> Self {
        Self {
            started_at: Instant::now(),
        }
    }
}

#[cfg(all(feature = "mx8_ebpf_net_pressure", target_os = "linux"))]
impl NetPressureSource for EbpfNetPressureSource {
    fn poll(&self) -> NetPressureSample {
        // Phase 1 skeleton: collector wiring is added in Step 3.
        NetPressureSample {
            ratio: 0.0,
            fresh: false,
            age_ms: self
                .started_at
                .elapsed()
                .as_millis()
                .min(u128::from(u64::MAX)) as u64,
        }
    }

    fn name(&self) -> &'static str {
        "ebpf_skeleton"
    }
}

#[cfg(all(feature = "mx8_ebpf_net_pressure", target_os = "linux"))]
fn build_ebpf_source() -> Result<Arc<dyn NetPressureSource>, &'static str> {
    Ok(Arc::new(EbpfNetPressureSource::new()))
}

#[cfg(not(all(feature = "mx8_ebpf_net_pressure", target_os = "linux")))]
fn build_ebpf_source() -> Result<Arc<dyn NetPressureSource>, &'static str> {
    Err("ebpf net pressure source unavailable (requires linux + mx8_ebpf_net_pressure feature)")
}

pub(crate) fn build_net_pressure_source() -> NetPressureRuntime {
    let source = env_string(NET_PRESSURE_SOURCE_ENV)
        .unwrap_or_else(|| "off".to_string())
        .trim()
        .to_ascii_lowercase();
    if source == "off" || source == "none" || source.is_empty() {
        return NetPressureRuntime {
            source: Arc::new(NoopNetPressureSource),
            disabled_total_seed: 0,
        };
    }
    if source == "ebpf" {
        match build_ebpf_source() {
            Ok(built) => {
                tracing::info!(
                    target: "mx8_proof",
                    event = "autotune_net_pressure_source_enabled",
                    source = built.name(),
                    "autotune network pressure source enabled"
                );
                return NetPressureRuntime {
                    source: built,
                    disabled_total_seed: 0,
                };
            }
            Err(msg) => {
                tracing::warn!(
                    target: "mx8_proof",
                    event = "autotune_net_pressure_source_unavailable",
                    source = "ebpf",
                    reason = msg,
                    "requested ebpf network pressure source unavailable; falling back to noop"
                );
                return NetPressureRuntime {
                    source: Arc::new(NoopNetPressureSource),
                    disabled_total_seed: 1,
                };
            }
        }
    }
    tracing::warn!(
        target: "mx8_proof",
        event = "autotune_net_pressure_source_unknown",
        source = source,
        "unknown network pressure source; falling back to noop"
    );
    NetPressureRuntime {
        source: Arc::new(NoopNetPressureSource),
        disabled_total_seed: 1,
    }
}
