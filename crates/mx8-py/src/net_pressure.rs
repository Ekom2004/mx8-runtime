use super::*;
#[cfg(all(feature = "mx8_ebpf_net_pressure", target_os = "linux"))]
use std::fs;

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

#[cfg(any(test, all(feature = "mx8_ebpf_net_pressure", target_os = "linux")))]
#[derive(Debug)]
struct LinuxTcpCounters {
    out_segs: u64,
    retrans_segs: u64,
    tcp_timeouts: u64,
}

#[cfg(all(feature = "mx8_ebpf_net_pressure", target_os = "linux"))]
#[derive(Debug)]
struct EbpfNetPressureState {
    last_poll_at: Instant,
    last_sample_at: Option<Instant>,
    last_ratio: f64,
    last_fresh: bool,
    last_counters: Option<(LinuxTcpCounters, Instant)>,
}

#[cfg(all(feature = "mx8_ebpf_net_pressure", target_os = "linux"))]
impl EbpfNetPressureState {
    fn new(now: Instant, counters: LinuxTcpCounters) -> Self {
        Self {
            last_poll_at: now,
            last_sample_at: Some(now),
            last_ratio: 0.0,
            last_fresh: false,
            last_counters: Some((counters, now)),
        }
    }

    fn sample(&self, now: Instant) -> NetPressureSample {
        let age_ms = self
            .last_sample_at
            .map(|t| now.duration_since(t).as_millis().min(u128::from(u64::MAX)) as u64)
            .unwrap_or(0);
        NetPressureSample {
            ratio: self.last_ratio.clamp(0.0, 1.0),
            fresh: self.last_fresh,
            age_ms,
        }
    }
}

#[cfg(all(feature = "mx8_ebpf_net_pressure", target_os = "linux"))]
#[derive(Debug)]
struct EbpfNetPressureSource {
    state: Mutex<EbpfNetPressureState>,
}

#[cfg(all(feature = "mx8_ebpf_net_pressure", target_os = "linux"))]
impl EbpfNetPressureSource {
    const SAMPLE_INTERVAL: Duration = Duration::from_millis(400);

    fn new() -> Result<Self, String> {
        let now = Instant::now();
        let counters = read_linux_tcp_counters()?;
        Ok(Self {
            state: Mutex::new(EbpfNetPressureState::new(now, counters)),
        })
    }
}

#[cfg(all(feature = "mx8_ebpf_net_pressure", target_os = "linux"))]
impl NetPressureSource for EbpfNetPressureSource {
    fn poll(&self) -> NetPressureSample {
        let now = Instant::now();
        let mut guard = match self.state.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };

        if now.duration_since(guard.last_poll_at) < Self::SAMPLE_INTERVAL {
            return guard.sample(now);
        }
        guard.last_poll_at = now;

        match read_linux_tcp_counters() {
            Ok(cur) => {
                if let Some((prev, prev_at)) = &guard.last_counters {
                    let elapsed_secs = now.duration_since(*prev_at).as_secs_f64().max(1e-6);
                    guard.last_ratio = estimate_net_pressure(prev, &cur, elapsed_secs);
                    guard.last_fresh = true;
                } else {
                    guard.last_ratio = 0.0;
                    guard.last_fresh = false;
                }
                guard.last_sample_at = Some(now);
                guard.last_counters = Some((cur, now));
            }
            Err(_) => {
                // Fail-open: keep last ratio but mark sample stale.
                guard.last_fresh = false;
            }
        }

        guard.sample(now)
    }

    fn name(&self) -> &'static str {
        "ebpf_linux_tcp"
    }
}

#[cfg(all(feature = "mx8_ebpf_net_pressure", target_os = "linux"))]
fn build_ebpf_source() -> Result<Arc<dyn NetPressureSource>, String> {
    Ok(Arc::new(EbpfNetPressureSource::new()?))
}

#[cfg(not(all(feature = "mx8_ebpf_net_pressure", target_os = "linux")))]
fn build_ebpf_source() -> Result<Arc<dyn NetPressureSource>, String> {
    Err("ebpf net pressure source unavailable (requires linux + mx8_ebpf_net_pressure feature)"
        .to_string())
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
                    reason = msg.as_str(),
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

#[cfg(all(feature = "mx8_ebpf_net_pressure", target_os = "linux"))]
fn read_linux_tcp_counters() -> Result<LinuxTcpCounters, String> {
    let snmp = fs::read_to_string("/proc/net/snmp")
        .map_err(|e| format!("failed to read /proc/net/snmp: {e}"))?;
    let netstat = fs::read_to_string("/proc/net/netstat")
        .map_err(|e| format!("failed to read /proc/net/netstat: {e}"))?;

    let out_segs = parse_proc_counter(&snmp, "Tcp:", "OutSegs")
        .ok_or_else(|| "missing Tcp OutSegs in /proc/net/snmp".to_string())?;
    let retrans_segs = parse_proc_counter(&snmp, "Tcp:", "RetransSegs")
        .ok_or_else(|| "missing Tcp RetransSegs in /proc/net/snmp".to_string())?;
    let tcp_timeouts = parse_proc_counter(&netstat, "TcpExt:", "TCPTimeouts").unwrap_or(0);

    Ok(LinuxTcpCounters {
        out_segs,
        retrans_segs,
        tcp_timeouts,
    })
}

#[cfg(any(test, all(feature = "mx8_ebpf_net_pressure", target_os = "linux")))]
fn parse_proc_counter(contents: &str, family_prefix: &str, field: &str) -> Option<u64> {
    let mut lines = contents.lines();
    while let Some(header) = lines.next() {
        let Some(values) = lines.next() else {
            break;
        };
        if !header.starts_with(family_prefix) || !values.starts_with(family_prefix) {
            continue;
        }
        let names = header.split_whitespace().skip(1);
        let vals = values.split_whitespace().skip(1);
        for (name, value) in names.zip(vals) {
            if name == field {
                return value.parse::<u64>().ok();
            }
        }
    }
    None
}

#[cfg(any(test, all(feature = "mx8_ebpf_net_pressure", target_os = "linux")))]
fn estimate_net_pressure(
    prev: &LinuxTcpCounters,
    cur: &LinuxTcpCounters,
    elapsed_secs: f64,
) -> f64 {
    let delta_out = cur.out_segs.saturating_sub(prev.out_segs);
    let delta_retrans = cur.retrans_segs.saturating_sub(prev.retrans_segs);
    let delta_timeouts = cur.tcp_timeouts.saturating_sub(prev.tcp_timeouts);

    let retrans_ratio = if delta_out == 0 {
        0.0
    } else {
        delta_retrans as f64 / delta_out as f64
    };
    let retrans_pressure = (retrans_ratio / 0.02).clamp(0.0, 1.0);

    let timeout_rate = delta_timeouts as f64 / elapsed_secs.max(1e-6);
    let timeout_pressure = (timeout_rate / 5.0).clamp(0.0, 1.0);

    (0.85 * retrans_pressure + 0.15 * timeout_pressure).clamp(0.0, 1.0)
}

#[cfg(test)]
mod net_pressure_tests {
    use super::*;

    #[test]
    fn parse_proc_counter_extracts_tcp_fields() {
        let snmp = "\
Tcp: RtoAlgorithm RtoMin RtoMax MaxConn ActiveOpens PassiveOpens AttemptFails EstabResets CurrEstab InSegs OutSegs RetransSegs
Tcp: 1 200 120000 -1 1970218 2230132 180889 56398 240 6144516041 6137193998 317556
";
        assert_eq!(
            parse_proc_counter(snmp, "Tcp:", "OutSegs"),
            Some(6_137_193_998)
        );
        assert_eq!(
            parse_proc_counter(snmp, "Tcp:", "RetransSegs"),
            Some(317_556)
        );
        assert_eq!(parse_proc_counter(snmp, "Tcp:", "Missing"), None);
    }

    #[test]
    fn estimate_net_pressure_increases_with_retrans_and_timeouts() {
        let prev = LinuxTcpCounters {
            out_segs: 10_000,
            retrans_segs: 10,
            tcp_timeouts: 1,
        };
        let low = LinuxTcpCounters {
            out_segs: 20_000,
            retrans_segs: 20,
            tcp_timeouts: 1,
        };
        let high = LinuxTcpCounters {
            out_segs: 20_000,
            retrans_segs: 400,
            tcp_timeouts: 25,
        };
        let low_p = estimate_net_pressure(&prev, &low, 2.0);
        let high_p = estimate_net_pressure(&prev, &high, 2.0);
        assert!(high_p > low_p);
        assert!((0.0..=1.0).contains(&low_p));
        assert!((0.0..=1.0).contains(&high_p));
    }
}
