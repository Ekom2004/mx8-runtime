use super::*;

pub(crate) fn metrics_to_dict<'py>(
    py: Python<'py>,
    metrics: &RuntimeMetrics,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new_bound(py);
    out.set_item(
        "delivered_batches_total",
        metrics.delivered_batches_total.get(),
    )?;
    out.set_item(
        "delivered_samples_total",
        metrics.delivered_samples_total.get(),
    )?;
    out.set_item("inflight_bytes", metrics.inflight_bytes.get())?;
    out.set_item("process_rss_bytes", metrics.process_rss_bytes.get())?;
    out.set_item("ram_high_water_bytes", metrics.ram_high_water_bytes.get())?;
    out.set_item(
        "batch_payload_bytes_p50",
        metrics.batch_payload_bytes_p50.get(),
    )?;
    out.set_item(
        "batch_payload_bytes_p95",
        metrics.batch_payload_bytes_p95.get(),
    )?;
    out.set_item(
        "batch_payload_window_size",
        metrics.batch_payload_window_size.get(),
    )?;
    out.set_item(
        "batch_payload_bytes_p95_over_p50",
        metrics.batch_payload_bytes_p95_over_p50_milli.get() as f64 / 1000.0,
    )?;
    out.set_item(
        "batch_jitter_slo_breaches_total",
        metrics.batch_jitter_slo_breaches_total.get(),
    )?;
    out.set_item(
        "batch_jitter_band_adjustments_total",
        metrics.batch_jitter_band_adjustments_total.get(),
    )?;
    out.set_item(
        "batch_jitter_band_lower_pct",
        metrics.batch_jitter_band_lower_pct.get(),
    )?;
    out.set_item(
        "batch_jitter_band_upper_pct",
        metrics.batch_jitter_band_upper_pct.get(),
    )?;
    Ok(out)
}

pub(crate) fn dict_item<'py>(stats: &Bound<'py, PyDict>, key: &str) -> Option<Bound<'py, PyAny>> {
    stats.get_item(key).ok().flatten()
}

pub(crate) fn dict_u64(stats: &Bound<'_, PyDict>, key: &str) -> Option<u64> {
    dict_item(stats, key).and_then(|v| {
        v.extract::<u64>().ok().or_else(|| {
            v.extract::<i64>()
                .ok()
                .and_then(|n| u64::try_from(n.max(0)).ok())
        })
    })
}

pub(crate) fn dict_f64(stats: &Bound<'_, PyDict>, key: &str) -> Option<f64> {
    dict_item(stats, key).and_then(|v| v.extract::<f64>().ok())
}

pub(crate) fn dict_bool(stats: &Bound<'_, PyDict>, key: &str) -> Option<bool> {
    dict_item(stats, key).and_then(|v| v.extract::<bool>().ok())
}

pub(crate) fn dict_u64_list_sum(stats: &Bound<'_, PyDict>, key: &str) -> Option<u64> {
    let list = dict_item(stats, key)?.downcast_into::<PyList>().ok()?;
    let mut total = 0u64;
    for item in list.iter() {
        let value = item
            .extract::<u64>()
            .ok()
            .or_else(|| {
                item.extract::<i64>()
                    .ok()
                    .and_then(|n| u64::try_from(n).ok())
            })
            .unwrap_or(0);
        total = total.saturating_add(value);
    }
    Some(total)
}

pub(crate) fn bytes_to_gb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

pub(crate) fn classify_pressure(pressure: f64) -> &'static str {
    if pressure >= 0.85 {
        "HIGH"
    } else if pressure >= 0.60 {
        "WARM"
    } else {
        "STABLE"
    }
}

pub(crate) fn total_samples_from_stats(stats: &Bound<'_, PyDict>) -> Option<u64> {
    dict_u64(stats, "delivered_samples_total")
        .or_else(|| dict_u64(stats, "video_delivered_samples_total"))
        .or_else(|| dict_u64_list_sum(stats, "mix_source_delivered_samples_total"))
}

pub(crate) fn total_batches_from_stats(stats: &Bound<'_, PyDict>) -> Option<u64> {
    dict_u64(stats, "delivered_batches_total")
        .or_else(|| dict_u64(stats, "video_delivered_batches_total"))
        .or_else(|| dict_u64_list_sum(stats, "mix_source_delivered_batches_total"))
}

pub(crate) fn total_bytes_from_stats(stats: &Bound<'_, PyDict>) -> Option<u64> {
    dict_u64(stats, "video_delivered_bytes_total")
        .or_else(|| dict_u64_list_sum(stats, "mix_source_delivered_bytes_total"))
}

pub(crate) fn mix_stability(stats: &Bound<'_, PyDict>) -> Option<(f64, u64)> {
    let sources = dict_item(stats, "mix_sources")?
        .downcast_into::<PyList>()
        .ok()?;
    let mut max_ratio = 0.0f64;
    let mut saw_ratio = false;
    let mut jitter_total = 0u64;
    for item in sources.iter() {
        let source = match item.downcast::<PyDict>() {
            Ok(d) => d,
            Err(_) => continue,
        };
        let metrics = match source.get_item("metrics").ok().flatten() {
            Some(v) => match v.downcast_into::<PyDict>() {
                Ok(d) => d,
                Err(_) => continue,
            },
            None => continue,
        };
        if let Some(ratio) = dict_f64(&metrics, "batch_payload_bytes_p95_over_p50") {
            saw_ratio = true;
            if ratio > max_ratio {
                max_ratio = ratio;
            }
        }
        if let Some(breaches) = dict_u64(&metrics, "batch_jitter_slo_breaches_total") {
            jitter_total = jitter_total.saturating_add(breaches);
        }
    }
    if saw_ratio {
        Some((max_ratio, jitter_total))
    } else if jitter_total > 0 {
        Some((0.0, jitter_total))
    } else {
        None
    }
}

pub(crate) fn render_human_stats(stats: &Bound<'_, PyDict>) -> String {
    let is_distributed =
        dict_u64(stats, "world_size").is_some() || dict_u64(stats, "assigned_rank").is_some();
    let is_mixed = dict_item(stats, "mix_sources").is_some();
    let is_video = dict_item(stats, "video_layout").is_some();

    let mode = if is_distributed {
        let world = dict_u64(stats, "world_size").unwrap_or(1);
        let rank = dict_u64(stats, "assigned_rank").unwrap_or(0);
        format!("distributed (rank {}/{world})", rank.saturating_add(1))
    } else if is_mixed {
        "mixed".to_string()
    } else if is_video {
        "video".to_string()
    } else {
        "local".to_string()
    };

    let step = total_batches_from_stats(stats).or_else(|| dict_u64(stats, "mix_schedule_ticks"));
    let epoch = dict_u64(stats, "epoch");
    let decode_failed = dict_u64(stats, "video_decode_failed_total").unwrap_or(0);
    let status = if decode_failed > 0 {
        "DEGRADED"
    } else if step.unwrap_or(0) > 0 {
        "RUNNING"
    } else {
        "WARMING"
    };

    let mut lines = Vec::<String>::new();
    let mut summary = format!("Status: {status} | Mode: {mode}");
    if let Some(epoch) = epoch {
        summary.push_str(&format!(" | Epoch: {epoch}"));
    }
    if let Some(step) = step {
        summary.push_str(&format!(" | Step: {step}"));
    }
    lines.push(summary);

    if is_video {
        let clips_total = dict_u64(stats, "clips_total");
        let clips_remaining = dict_u64(stats, "clips_remaining");
        if let (Some(total), Some(rem)) = (clips_total, clips_remaining) {
            let done = total.saturating_sub(rem);
            let pct = if total > 0 {
                (done as f64 / total as f64) * 100.0
            } else {
                0.0
            };
            lines.push(format!("Progress: {pct:.2}% ({done}/{total} clips)"));
        }
    } else {
        let samples = total_samples_from_stats(stats).unwrap_or(0);
        let batches = total_batches_from_stats(stats).unwrap_or(0);
        lines.push(format!("Progress: {samples} samples | {batches} batches"));
    }

    if let Some(elapsed_s) = dict_f64(stats, "elapsed_seconds") {
        if elapsed_s > 0.0 {
            let samples = total_samples_from_stats(stats).unwrap_or(0);
            let samples_per_sec = samples as f64 / elapsed_s;
            let bytes = total_bytes_from_stats(stats);
            if let Some(bytes) = bytes {
                let bytes_per_sec_gb = (bytes as f64 / elapsed_s) / (1024.0 * 1024.0 * 1024.0);
                lines.push(format!(
                    "Throughput: {samples_per_sec:.1} samples/s | {bytes_per_sec_gb:.2} GB/s"
                ));
            } else {
                lines.push(format!("Throughput: {samples_per_sec:.1} samples/s"));
            }
        }
    }

    let rss = dict_u64(stats, "process_rss_bytes");
    let rss_cap = dict_u64(stats, "max_process_rss_bytes").filter(|v| *v > 0);
    let pressure = dict_f64(stats, "autotune_pressure")
        .or_else(|| dict_f64(stats, "mix_runtime_autotune_pressure"))
        .or_else(|| dict_f64(stats, "video_runtime_autotune_pressure"));
    if let Some(rss) = rss {
        if let Some(cap) = rss_cap {
            let pct = if cap > 0 {
                (rss as f64 / cap as f64) * 100.0
            } else {
                0.0
            };
            let mut line = format!(
                "Memory: {:.2} GB / {:.2} GB ({pct:.1}%)",
                bytes_to_gb(rss),
                bytes_to_gb(cap)
            );
            if let Some(p) = pressure {
                line.push_str(&format!(" | Pressure: {}", classify_pressure(p)));
            }
            lines.push(line);
        } else {
            let mut line = format!("Memory: {:.2} GB used", bytes_to_gb(rss));
            if let Some(p) = pressure {
                line.push_str(&format!(" | Pressure: {}", classify_pressure(p)));
            }
            lines.push(line);
        }
    }

    let mut ratio = dict_f64(stats, "batch_payload_bytes_p95_over_p50");
    let mut jitter = dict_u64(stats, "batch_jitter_slo_breaches_total");
    if ratio.is_none() && jitter.is_none() {
        if let Some((mix_ratio, mix_jitter)) = mix_stability(stats) {
            if mix_ratio > 0.0 {
                ratio = Some(mix_ratio);
            }
            if mix_jitter > 0 {
                jitter = Some(mix_jitter);
            }
        }
    }
    if ratio.is_some() || jitter.is_some() {
        let ratio_txt = ratio
            .map(|v| format!("{v:.2}"))
            .unwrap_or_else(|| "n/a".to_string());
        let jitter_txt = jitter
            .map(|v| v.to_string())
            .unwrap_or_else(|| "n/a".to_string());
        lines.push(format!(
            "Stability: p95/p50={ratio_txt} | jitter_slo_breaches={jitter_txt}"
        ));
    }

    let mut autotune_parts = Vec::<String>::new();
    if let Some(enabled) = dict_bool(stats, "autotune_enabled")
        .or_else(|| dict_bool(stats, "mix_runtime_autotune_enabled"))
        .or_else(|| dict_bool(stats, "video_runtime_autotune_enabled"))
    {
        autotune_parts.push(if enabled {
            "enabled".into()
        } else {
            "disabled".into()
        });
    }
    let prefetch = dict_u64(stats, "effective_prefetch_batches")
        .or_else(|| dict_u64(stats, "mix_effective_prefetch_batches"));
    if let Some(prefetch) = prefetch {
        autotune_parts.push(format!("prefetch={prefetch}"));
    }
    let queue = dict_u64(stats, "effective_max_queue_batches")
        .or_else(|| dict_u64(stats, "mix_effective_max_queue_batches"));
    if let Some(queue) = queue {
        autotune_parts.push(format!("queue={queue}"));
    }
    if let Some(want) = dict_u64(stats, "effective_want") {
        autotune_parts.push(format!("want={want}"));
    }
    if let Some(cooldown) = dict_u64(stats, "autotune_cooldown_ticks") {
        autotune_parts.push(format!("cooldown={cooldown}"));
    }
    let adjustments = dict_u64(stats, "mix_runtime_autotune_adjustments_total")
        .or_else(|| dict_u64(stats, "video_runtime_autotune_adjustments_total"));
    if let Some(adjustments) = adjustments {
        autotune_parts.push(format!("adjustments={adjustments}"));
    }
    if !autotune_parts.is_empty() {
        lines.push(format!("Autotune: {}", autotune_parts.join(" | ")));
    }

    if let Some(transitions_total) = dict_u64(stats, "elastic_transitions_total") {
        let pending = dict_bool(stats, "elastic_transition_pending").unwrap_or(false);
        let reason = dict_item(stats, "elastic_last_transition_reason")
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_else(|| "none".to_string());
        let current = dict_u64(stats, "elastic_current_world_size").unwrap_or(0);
        let target = dict_u64(stats, "elastic_target_world_size").unwrap_or(0);
        let state = if pending { "pending" } else { "stable" };
        lines.push(format!(
            "Elastic: {state} | transitions={transitions_total} | reason={reason} | world={current}->{target}"
        ));
    }

    if is_distributed {
        let world = dict_u64(stats, "world_size").unwrap_or(1);
        let rank = dict_u64(stats, "assigned_rank").unwrap_or(0);
        lines.push(format!(
            "Distributed: rank {} of {}",
            rank.saturating_add(1),
            world
        ));
    }

    lines.join("\n")
}
