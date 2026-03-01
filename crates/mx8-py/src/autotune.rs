use super::*;

pub(crate) fn rss_profile_fraction(profile: AutotuneProfile) -> f64 {
    match profile {
        AutotuneProfile::Safe => 0.60,
        AutotuneProfile::Balanced => 0.75,
        AutotuneProfile::Throughput => 0.85,
    }
}

pub(crate) fn derive_default_max_process_rss_bytes(
    profile: AutotuneProfile,
    max_inflight_bytes: u64,
) -> Option<u64> {
    let node_limit = detect_node_ram_limit_bytes();
    let reserve_bytes = 1u64 << 30;
    let base_rss = sample_process_rss_bytes_local().unwrap_or(0);
    let min_required = base_rss.saturating_add(max_inflight_bytes);
    let mut derived = match node_limit {
        Some(limit) => ((limit as f64) * rss_profile_fraction(profile)).max(1.0) as u64,
        None => min_required.saturating_add(reserve_bytes),
    };
    if node_limit.is_some() {
        derived = derived.saturating_sub(reserve_bytes);
    }
    if derived < min_required {
        derived = min_required;
    }
    Some(derived.max(max_inflight_bytes).max(1))
}

pub(crate) fn profile_name(profile: AutotuneProfile) -> &'static str {
    match profile {
        AutotuneProfile::Safe => "safe",
        AutotuneProfile::Balanced => "balanced",
        AutotuneProfile::Throughput => "throughput",
    }
}

pub(crate) fn derive_byte_batch_caps(
    max_inflight_bytes: u64,
    target_batch_bytes: Option<u64>,
    max_batch_bytes: Option<u64>,
) -> (Option<u64>, Option<u64>) {
    let default_max_batch = (max_inflight_bytes / 4)
        .clamp(4 * 1024 * 1024, 64 * 1024 * 1024)
        .min(max_inflight_bytes)
        .max(1);
    let effective_max_batch = max_batch_bytes
        .unwrap_or(default_max_batch)
        .min(max_inflight_bytes)
        .max(1);
    let default_target_batch = effective_max_batch.saturating_mul(9).div_ceil(10).max(1);
    let effective_target_batch = target_batch_bytes
        .unwrap_or(default_target_batch)
        .min(effective_max_batch)
        .max(1);
    (Some(effective_target_batch), Some(effective_max_batch))
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum AutotuneProfile {
    Safe,
    Balanced,
    Throughput,
}

impl AutotuneProfile {
    pub(crate) fn from_name(name: Option<&str>) -> Self {
        match name
            .unwrap_or("balanced")
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "safe" => Self::Safe,
            "throughput" => Self::Throughput,
            _ => Self::Balanced,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ProfileDefaults {
    pub(crate) max_inflight_bytes: u64,
    pub(crate) max_queue_batches: usize,
    pub(crate) prefetch_batches: usize,
}

impl ProfileDefaults {
    pub(crate) fn for_profile(profile: AutotuneProfile) -> Self {
        match profile {
            AutotuneProfile::Safe => Self {
                max_inflight_bytes: 128 * 1024 * 1024,
                max_queue_batches: 32,
                prefetch_batches: 1,
            },
            AutotuneProfile::Balanced => Self {
                max_inflight_bytes: 256 * 1024 * 1024,
                max_queue_batches: 64,
                prefetch_batches: 2,
            },
            AutotuneProfile::Throughput => Self {
                max_inflight_bytes: 512 * 1024 * 1024,
                max_queue_batches: 128,
                prefetch_batches: 4,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AutotuneRails {
    pub(crate) min_prefetch_batches: usize,
    pub(crate) max_prefetch_batches: usize,
    pub(crate) min_max_queue_batches: usize,
    pub(crate) max_max_queue_batches: usize,
    pub(crate) min_want: u32,
    pub(crate) max_want: u32,
}

impl AutotuneRails {
    pub(crate) fn for_profile(profile: AutotuneProfile) -> Self {
        match profile {
            AutotuneProfile::Safe => Self {
                min_prefetch_batches: 1,
                max_prefetch_batches: 4,
                min_max_queue_batches: 8,
                max_max_queue_batches: 64,
                min_want: 1,
                max_want: 2,
            },
            AutotuneProfile::Balanced => Self {
                min_prefetch_batches: 1,
                max_prefetch_batches: 8,
                min_max_queue_batches: 16,
                max_max_queue_batches: 128,
                min_want: 1,
                max_want: 4,
            },
            AutotuneProfile::Throughput => Self {
                min_prefetch_batches: 2,
                max_prefetch_batches: 16,
                min_max_queue_batches: 32,
                max_max_queue_batches: 256,
                min_want: 1,
                max_want: 8,
            },
        }
    }
}

#[derive(Debug)]
pub(crate) struct AutotuneShared {
    pub(crate) enabled: bool,
    pub(crate) want: AtomicU32,
    pub(crate) prefetch_batches: AtomicUsize,
    pub(crate) max_queue_batches: AtomicUsize,
    pub(crate) wait_ns_interval: AtomicU64,
    pub(crate) pressure_milli: AtomicU64,
    pub(crate) wait_ewma_milli: AtomicU64,
    pub(crate) rss_ewma_milli: AtomicU64,
    pub(crate) integral_rss_milli: AtomicI64,
    pub(crate) cooldown_ticks: AtomicU32,
}

impl AutotuneShared {
    pub(crate) fn new(
        enabled: bool,
        want: u32,
        prefetch_batches: usize,
        max_queue_batches: usize,
    ) -> Self {
        Self {
            enabled,
            want: AtomicU32::new(want.max(1)),
            prefetch_batches: AtomicUsize::new(prefetch_batches.max(1)),
            max_queue_batches: AtomicUsize::new(max_queue_batches.max(1)),
            wait_ns_interval: AtomicU64::new(0),
            pressure_milli: AtomicU64::new(0),
            wait_ewma_milli: AtomicU64::new(0),
            rss_ewma_milli: AtomicU64::new(0),
            integral_rss_milli: AtomicI64::new(0),
            cooldown_ticks: AtomicU32::new(0),
        }
    }

    pub(crate) fn on_wait(&self, elapsed: Duration) {
        if self.enabled {
            self.wait_ns_interval.fetch_add(
                elapsed.as_nanos().min(u128::from(u64::MAX)) as u64,
                Ordering::Relaxed,
            );
        }
    }
}

#[derive(Debug)]
pub(crate) struct AutotuneController {
    wait_ewma: f64,
    rss_ewma: f64,
    prev_rss_ewma: f64,
    integral_rss: f64,
    cooldown_ticks: u32,
    increase_ticks: u32,
}

impl AutotuneController {
    pub(crate) fn new() -> Self {
        Self {
            wait_ewma: 0.0,
            rss_ewma: 0.0,
            prev_rss_ewma: 0.0,
            integral_rss: 0.0,
            cooldown_ticks: 0,
            increase_ticks: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AutotuneUpdate {
    pub(crate) want: u32,
    pub(crate) prefetch_batches: usize,
    pub(crate) max_queue_batches: usize,
}

#[derive(Debug)]
pub(crate) struct AutotuneTickOutput {
    pub(crate) next: AutotuneUpdate,
    pub(crate) changed: bool,
    pub(crate) reason: &'static str,
    pub(crate) pressure: f64,
}

pub(crate) fn autotune_tick(
    state: &mut AutotuneController,
    current: AutotuneUpdate,
    rails: AutotuneRails,
    wait_ratio: f64,
    rss_ratio: f64,
    inflight_ratio: f64,
    interval_secs: f64,
) -> AutotuneTickOutput {
    const ALPHA: f64 = 0.2;
    const RSS_TARGET: f64 = 0.90;
    const WAIT_TARGET: f64 = 0.05;
    const KP: f64 = 1.5;
    const KI: f64 = 0.2;
    const KD: f64 = 0.1;

    state.wait_ewma = ALPHA * wait_ratio + (1.0 - ALPHA) * state.wait_ewma;
    state.rss_ewma = ALPHA * rss_ratio + (1.0 - ALPHA) * state.rss_ewma;

    let error = state.rss_ewma - RSS_TARGET;
    state.integral_rss = (state.integral_rss + error * interval_secs).clamp(-0.5, 0.5);
    let deriv = (state.rss_ewma - state.prev_rss_ewma) / interval_secs;
    state.prev_rss_ewma = state.rss_ewma;

    let pressure = (KP * error + KI * state.integral_rss + KD * deriv).clamp(0.0, 1.0);

    let mut next = current;
    let mut reason = "hold";

    let hard_cut = rss_ratio >= 0.97 || inflight_ratio >= 0.98;
    let soft_cut = pressure >= 0.60;
    let starvation = wait_ratio > 0.01 || state.wait_ewma > WAIT_TARGET;
    let can_increase = state.cooldown_ticks == 0
        && state.wait_ewma > WAIT_TARGET
        && pressure <= 0.30
        && inflight_ratio <= 0.85
        && starvation;

    if hard_cut {
        next.prefetch_batches = ((next.prefetch_batches as f64) * 0.5).floor() as usize;
        next.max_queue_batches = ((next.max_queue_batches as f64) * 0.7).floor() as usize;
        next.want = ((next.want as f64) * 0.5).floor() as u32;
        reason = "hard_cut";
        state.cooldown_ticks = 2;
    } else if soft_cut {
        next.prefetch_batches = next.prefetch_batches.saturating_sub(1);
        next.max_queue_batches = next.max_queue_batches.saturating_sub(2);
        reason = "soft_cut";
        state.increase_ticks = 0;
    } else if can_increase {
        next.prefetch_batches = next.prefetch_batches.saturating_add(1);
        next.max_queue_batches = next.max_queue_batches.saturating_add(2);
        if state.increase_ticks % 2 == 1 {
            next.want = next.want.saturating_add(1);
        }
        state.increase_ticks = state.increase_ticks.saturating_add(1);
        reason = "aimd_increase";
    } else {
        state.increase_ticks = 0;
        if state.cooldown_ticks > 0 {
            state.cooldown_ticks -= 1;
        }
    }

    next.prefetch_batches = next
        .prefetch_batches
        .clamp(rails.min_prefetch_batches, rails.max_prefetch_batches);
    next.max_queue_batches = next
        .max_queue_batches
        .clamp(rails.min_max_queue_batches, rails.max_max_queue_batches);
    next.want = next.want.clamp(rails.min_want, rails.max_want);

    AutotuneTickOutput {
        changed: next.prefetch_batches != current.prefetch_batches
            || next.max_queue_batches != current.max_queue_batches
            || next.want != current.want,
        next,
        reason,
        pressure,
    }
}

pub(crate) async fn autotune_loop(
    pipeline: Arc<Pipeline>,
    metrics: Arc<RuntimeMetrics>,
    shared: Arc<AutotuneShared>,
    max_inflight_bytes: u64,
    max_process_rss_bytes: Option<u64>,
    rails: AutotuneRails,
) {
    if !shared.enabled {
        return;
    }

    let mut state = AutotuneController::new();
    let interval = Duration::from_secs(2);
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        ticker.tick().await;

        let wait_ns = shared.wait_ns_interval.swap(0, Ordering::Relaxed);
        let wait_ratio = (wait_ns as f64 / interval.as_nanos() as f64).clamp(0.0, 1.0);
        let inflight_ratio = if max_inflight_bytes == 0 {
            0.0
        } else {
            (metrics.inflight_bytes.get() as f64 / max_inflight_bytes as f64).clamp(0.0, 1.5)
        };
        let rss_ratio = match max_process_rss_bytes {
            Some(rss_cap) if rss_cap > 0 => {
                (metrics.process_rss_bytes.get() as f64 / rss_cap as f64).clamp(0.0, 1.5)
            }
            _ => 0.0,
        };

        let current = AutotuneUpdate {
            want: shared.want.load(Ordering::Relaxed),
            prefetch_batches: shared.prefetch_batches.load(Ordering::Relaxed),
            max_queue_batches: shared.max_queue_batches.load(Ordering::Relaxed),
        };
        let tick = autotune_tick(
            &mut state,
            current,
            rails,
            wait_ratio,
            rss_ratio,
            inflight_ratio,
            interval.as_secs_f64(),
        );

        shared
            .pressure_milli
            .store((tick.pressure * 1000.0).round() as u64, Ordering::Relaxed);
        shared
            .wait_ewma_milli
            .store((state.wait_ewma * 1000.0).round() as u64, Ordering::Relaxed);
        shared
            .rss_ewma_milli
            .store((state.rss_ewma * 1000.0).round() as u64, Ordering::Relaxed);
        shared.integral_rss_milli.store(
            (state.integral_rss * 1000.0).round() as i64,
            Ordering::Relaxed,
        );
        shared
            .cooldown_ticks
            .store(state.cooldown_ticks, Ordering::Relaxed);

        if tick.changed {
            shared.want.store(tick.next.want, Ordering::Relaxed);
            shared
                .prefetch_batches
                .store(tick.next.prefetch_batches, Ordering::Relaxed);
            shared
                .max_queue_batches
                .store(tick.next.max_queue_batches, Ordering::Relaxed);
            pipeline.set_prefetch_batches(tick.next.prefetch_batches);
            pipeline.set_max_queue_batches(tick.next.max_queue_batches);
            tracing::info!(
                target: "mx8_proof",
                event = "autotune_runtime_adjustment",
                reason = tick.reason,
                wait_ratio = wait_ratio,
                wait_ewma = state.wait_ewma,
                pressure = tick.pressure,
                rss_ratio = rss_ratio,
                inflight_ratio = inflight_ratio,
                prefetch_batches = tick.next.prefetch_batches as u64,
                max_queue_batches = tick.next.max_queue_batches as u64,
                want = tick.next.want as u64,
                "autotune adjusted runtime knobs"
            );
        }
    }
}

#[cfg(test)]
mod autotune_tests {
    use super::{
        autotune_tick, AutotuneController, AutotuneRails, AutotuneTickOutput, AutotuneUpdate,
    };

    fn tick(
        state: &mut AutotuneController,
        current: AutotuneUpdate,
        rails: AutotuneRails,
        wait_ratio: f64,
        rss_ratio: f64,
        inflight_ratio: f64,
    ) -> AutotuneTickOutput {
        autotune_tick(
            state,
            current,
            rails,
            wait_ratio,
            rss_ratio,
            inflight_ratio,
            2.0,
        )
    }

    #[test]
    fn autotune_hard_cut_halves_knobs() {
        let mut state = AutotuneController::new();
        let rails = AutotuneRails {
            min_prefetch_batches: 1,
            max_prefetch_batches: 16,
            min_max_queue_batches: 8,
            max_max_queue_batches: 256,
            min_want: 1,
            max_want: 8,
        };
        let current = AutotuneUpdate {
            want: 4,
            prefetch_batches: 8,
            max_queue_batches: 100,
        };
        let out = tick(&mut state, current, rails, 0.0, 0.98, 0.2);
        assert!(out.changed);
        assert_eq!(out.reason, "hard_cut");
        assert_eq!(out.next.prefetch_batches, 4);
        assert_eq!(out.next.max_queue_batches, 70);
        assert_eq!(out.next.want, 2);
    }

    #[test]
    fn autotune_increase_respects_rails() {
        let mut state = AutotuneController::new();
        let rails = AutotuneRails {
            min_prefetch_batches: 1,
            max_prefetch_batches: 4,
            min_max_queue_batches: 8,
            max_max_queue_batches: 16,
            min_want: 1,
            max_want: 2,
        };
        let current = AutotuneUpdate {
            want: 2,
            prefetch_batches: 4,
            max_queue_batches: 16,
        };
        let out = tick(&mut state, current, rails, 0.30, 0.20, 0.10);
        assert!(!out.changed);
        assert_eq!(out.reason, "aimd_increase");
        assert_eq!(out.next.prefetch_batches, 4);
        assert_eq!(out.next.max_queue_batches, 16);
        assert_eq!(out.next.want, 2);
    }

    #[test]
    fn autotune_soft_cut_decrements() {
        let mut state = AutotuneController::new();
        let rails = AutotuneRails {
            min_prefetch_batches: 1,
            max_prefetch_batches: 8,
            min_max_queue_batches: 8,
            max_max_queue_batches: 64,
            min_want: 1,
            max_want: 4,
        };
        // Prime state to build pressure.
        let current = AutotuneUpdate {
            want: 3,
            prefetch_batches: 4,
            max_queue_batches: 32,
        };
        let _ = tick(&mut state, current, rails, 0.0, 0.95, 0.5);
        let out = tick(&mut state, current, rails, 0.0, 0.95, 0.5);
        assert_eq!(out.reason, "soft_cut");
        assert!(out.changed);
        assert_eq!(out.next.prefetch_batches, 3);
        assert_eq!(out.next.max_queue_batches, 30);
        assert_eq!(out.next.want, 3);
    }

    #[test]
    fn autotune_tick_sequence_is_deterministic() {
        let rails = AutotuneRails {
            min_prefetch_batches: 1,
            max_prefetch_batches: 8,
            min_max_queue_batches: 8,
            max_max_queue_batches: 64,
            min_want: 1,
            max_want: 4,
        };
        let seed = AutotuneUpdate {
            want: 1,
            prefetch_batches: 2,
            max_queue_batches: 16,
        };
        let signals = [
            (0.20, 0.50, 0.40),
            (0.22, 0.55, 0.45),
            (0.25, 0.60, 0.50),
            (0.05, 0.98, 0.99),
            (0.10, 0.70, 0.75),
            (0.18, 0.40, 0.30),
        ];

        let run = |mut state: AutotuneController| {
            let mut cur = seed;
            let mut out = Vec::new();
            for (wait, rss, inflight) in signals {
                let tick = tick(&mut state, cur, rails, wait, rss, inflight);
                out.push((
                    tick.next.want,
                    tick.next.prefetch_batches,
                    tick.next.max_queue_batches,
                    tick.reason,
                    tick.changed,
                ));
                cur = tick.next;
            }
            out
        };

        let first = run(AutotuneController::new());
        let second = run(AutotuneController::new());
        assert_eq!(first, second);
    }
}
