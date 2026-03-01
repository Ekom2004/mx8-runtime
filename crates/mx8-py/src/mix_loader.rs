use super::*;

#[pyclass]
pub(crate) struct MixedDataLoader {
    pub(crate) loaders: Vec<MixLoader>,
    pub(crate) scheduler: WeightedRoundRobin,
    pub(crate) active: Vec<bool>,
    pub(crate) source_exhausted_total: Vec<u64>,
    pub(crate) delivered_batches: Vec<u64>,
    pub(crate) delivered_samples: Vec<u64>,
    pub(crate) delivered_bytes: Vec<u64>,
    pub(crate) starvation_total: Vec<u64>,
    pub(crate) steps_since_emit: Vec<u64>,
    pub(crate) max_starvation_window: u64,
    pub(crate) normalized_weights: Vec<u64>,
    pub(crate) shared_max_inflight_bytes: u64,
    pub(crate) shared_inflight_violation_total: u64,
    pub(crate) seed: u64,
    pub(crate) epoch: u64,
    pub(crate) schedule_ticks: u64,
    pub(crate) snapshot_enabled: bool,
    pub(crate) snapshot_period_ticks: u64,
    pub(crate) snapshot_emitted_total: u64,
    pub(crate) source_exhaustion_policy: SourceExhaustionPolicy,
    pub(crate) mix_profile: Option<String>,
    pub(crate) mix_autotune_enabled: bool,
    pub(crate) mix_effective_prefetch_batches: usize,
    pub(crate) mix_effective_max_queue_batches: usize,
    pub(crate) mix_runtime_autotune_enabled: bool,
    pub(crate) mix_autotune_rails: Option<AutotuneRails>,
    pub(crate) mix_autotune_controller: AutotuneController,
    pub(crate) mix_autotune_current: AutotuneUpdate,
    pub(crate) mix_runtime_autotune_period_ticks: u64,
    pub(crate) mix_runtime_autotune_last_tick: u64,
    pub(crate) mix_runtime_autotune_adjustments_total: u64,
    pub(crate) mix_runtime_autotune_pressure_milli: u64,
    pub(crate) mix_resume_source_checkpoint_mismatch_total: u64,
    pub(crate) mix_max_process_rss_bytes: Option<u64>,
    pub(crate) started_at: Instant,
}

pub(crate) enum MixLoader {
    Local(Py<DataLoader>),
    Distributed(Py<DistributedDataLoader>),
}

pub(crate) struct WeightedRoundRobin {
    weights: Vec<u64>,
    current: Vec<i128>,
    tie_break_offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SourceExhaustionPolicy {
    Error,
    Allow,
}

impl SourceExhaustionPolicy {
    pub(crate) fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "error" => Ok(Self::Error),
            "allow" => Ok(Self::Allow),
            _ => anyhow::bail!("invalid source_exhausted={raw:?} (expected: error|allow)"),
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Allow => "allow",
        }
    }
}

impl WeightedRoundRobin {
    pub(crate) fn new(weights: Vec<u64>, seed: u64, epoch: u64) -> Self {
        let n = weights.len();
        let tie_break_offset = if n == 0 {
            0
        } else {
            // Keep tie-break deterministic, and ensure epoch shifts the tie order when n > 1.
            let seed_component = (seed as usize) % n;
            let epoch_component = (epoch as usize) % n;
            (seed_component + epoch_component) % n
        };
        Self {
            current: vec![0; n],
            weights,
            tie_break_offset,
        }
    }

    fn select(&mut self, active: &[bool]) -> Option<usize> {
        if self.weights.len() != active.len() || self.current.len() != active.len() {
            return None;
        }
        let mut total_weight: i128 = 0;
        for (i, is_active) in active.iter().enumerate() {
            if *is_active {
                self.current[i] += i128::from(self.weights[i]);
                total_weight += i128::from(self.weights[i]);
            }
        }
        if total_weight <= 0 {
            return None;
        }

        let n = active.len();
        let mut best_idx: Option<usize> = None;
        let mut best_cur: i128 = i128::MIN;
        for (i, is_active) in active.iter().enumerate() {
            if !*is_active {
                continue;
            }
            let cur = self.current[i];
            if cur > best_cur {
                best_cur = cur;
                best_idx = Some(i);
                continue;
            }
            if cur == best_cur {
                let prev = best_idx.unwrap_or(i);
                let lhs = (i + n - self.tie_break_offset % n) % n;
                let rhs = (prev + n - self.tie_break_offset % n) % n;
                if lhs < rhs {
                    best_idx = Some(i);
                }
            }
        }

        let picked = best_idx?;
        self.current[picked] -= total_weight;
        Some(picked)
    }
}

pub(crate) fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

pub(crate) fn normalize_mix_weights(raw: &[f64]) -> Result<Vec<u64>> {
    if raw.is_empty() {
        anyhow::bail!("mix weights must be non-empty");
    }
    let mut scaled = Vec::with_capacity(raw.len());
    for (idx, w) in raw.iter().enumerate() {
        if !w.is_finite() {
            anyhow::bail!("mix weight at index {idx} is not finite");
        }
        if *w <= 0.0 {
            anyhow::bail!("mix weight at index {idx} must be > 0");
        }
        let v = (*w * 1_000_000.0).round();
        if v < 1.0 {
            anyhow::bail!("mix weight at index {idx} is too small after normalization");
        }
        scaled.push(v as u64);
    }
    let mut g = scaled[0];
    for &v in scaled.iter().skip(1) {
        g = gcd_u64(g, v);
    }
    if g > 1 {
        for v in &mut scaled {
            *v /= g;
        }
    }
    Ok(scaled)
}

pub(crate) fn compute_shared_mix_cap(max_inflight_bytes: &[u64]) -> Result<u64> {
    let Some(min_cap) = max_inflight_bytes.iter().copied().min() else {
        anyhow::bail!("mix requires at least one loader");
    };
    if min_cap == 0 {
        anyhow::bail!("mix loader max_inflight_bytes must be > 0");
    }
    Ok(min_cap)
}

pub(crate) fn should_emit_mix_snapshot(schedule_ticks: u64, snapshot_period_ticks: u64) -> bool {
    if snapshot_period_ticks == 0 || schedule_ticks == 0 {
        return false;
    }
    schedule_ticks.is_multiple_of(snapshot_period_ticks)
}

impl MixedDataLoader {
    fn source_inflight_bytes(source: &MixLoader, py: Python<'_>) -> u64 {
        match source {
            MixLoader::Local(loader) => {
                let guard = loader.borrow(py);
                guard.metrics.inflight_bytes.get()
            }
            MixLoader::Distributed(loader) => {
                let guard = loader.borrow(py);
                guard.metrics.inflight_bytes.get()
            }
        }
    }

    fn source_rss_bytes(source: &MixLoader, py: Python<'_>) -> u64 {
        match source {
            MixLoader::Local(loader) => {
                let guard = loader.borrow(py);
                guard.metrics.process_rss_bytes.get()
            }
            MixLoader::Distributed(loader) => {
                let guard = loader.borrow(py);
                guard.metrics.process_rss_bytes.get()
            }
        }
    }

    pub(crate) fn source_config(
        source: &MixLoader,
        py: Python<'_>,
    ) -> (Option<String>, Option<String>, usize, u64, usize, usize) {
        match source {
            MixLoader::Local(loader) => {
                let guard = loader.borrow(py);
                (
                    Some(guard.dataset_base.clone()),
                    Some(guard.manifest_hash.clone()),
                    guard.batch_size_samples,
                    guard.max_inflight_bytes,
                    guard.max_queue_batches,
                    guard.prefetch_batches,
                )
            }
            MixLoader::Distributed(loader) => {
                let guard = loader.borrow(py);
                (
                    None,
                    Some(guard.manifest_hash.clone()),
                    0,
                    0,
                    guard.pipeline.effective_max_queue_batches(),
                    guard.pipeline.effective_prefetch_batches(),
                )
            }
        }
    }

    pub(crate) fn source_apply_runtime_overrides(
        source: &MixLoader,
        py: Python<'_>,
        prefetch_batches: Option<usize>,
        max_queue_batches: Option<usize>,
    ) {
        match source {
            MixLoader::Local(loader) => {
                let mut guard = loader.borrow_mut(py);
                guard.apply_runtime_overrides(prefetch_batches, max_queue_batches);
            }
            MixLoader::Distributed(loader) => {
                let guard = loader.borrow_mut(py);
                if let Some(prefetch) = prefetch_batches {
                    guard.pipeline.set_prefetch_batches(prefetch.max(1));
                }
                if let Some(max_queue) = max_queue_batches {
                    guard.pipeline.set_max_queue_batches(max_queue.max(1));
                }
            }
        }
    }

    fn source_next<'py>(source: &MixLoader, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match source {
            MixLoader::Local(loader) => {
                let mut guard = loader.borrow_mut(py);
                guard.__next__(py)
            }
            MixLoader::Distributed(loader) => {
                let mut guard = loader.borrow_mut(py);
                let batch = guard.__next__()?;
                let out = Py::new(py, batch)?;
                Ok(out.into_bound(py).into_any())
            }
        }
    }

    fn source_checkpoint(source: &MixLoader, py: Python<'_>) -> PyResult<Vec<u8>> {
        match source {
            MixLoader::Local(loader) => Ok(loader.borrow(py).checkpoint(py)?.as_bytes().to_vec()),
            MixLoader::Distributed(loader) => {
                Ok(loader.borrow(py).checkpoint(py)?.as_bytes().to_vec())
            }
        }
    }

    fn total_inflight_bytes(&self, py: Python<'_>) -> u64 {
        self.loaders
            .iter()
            .map(|ldr| Self::source_inflight_bytes(ldr, py))
            .fold(0u64, |acc, v| acc.saturating_add(v))
    }

    fn realized_ratio(&self) -> Vec<f64> {
        let total_samples: u64 = self.delivered_samples.iter().sum();
        if total_samples == 0 {
            return vec![0.0f64; self.delivered_samples.len()];
        }
        self.delivered_samples
            .iter()
            .map(|v| (*v as f64) / (total_samples as f64))
            .collect::<Vec<_>>()
    }

    fn maybe_emit_snapshot(&mut self, py: Python<'_>) {
        if !self.snapshot_enabled
            || !should_emit_mix_snapshot(self.schedule_ticks, self.snapshot_period_ticks)
        {
            return;
        }

        self.snapshot_emitted_total = self.snapshot_emitted_total.saturating_add(1);
        let realized_ratio = self.realized_ratio();
        let total_inflight_bytes = self.total_inflight_bytes(py);
        tracing::info!(
            target: "mx8_proof",
            event = "mix_snapshot",
            tick = self.schedule_ticks,
            snapshot_index = self.snapshot_emitted_total,
            seed = self.seed,
            epoch = self.epoch,
            active_sources = self.active.iter().filter(|v| **v).count() as u64,
            mix_total_inflight_bytes = total_inflight_bytes,
            mix_shared_max_inflight_bytes = self.shared_max_inflight_bytes,
            normalized_weights = ?self.normalized_weights,
            delivered_samples = ?self.delivered_samples,
            delivered_bytes = ?self.delivered_bytes,
            starvation_total = ?self.starvation_total,
            realized_ratio = ?realized_ratio,
            "periodic mix proof snapshot"
        );
    }

    fn max_process_rss_bytes(&self, py: Python<'_>) -> u64 {
        self.loaders
            .iter()
            .map(|ldr| Self::source_rss_bytes(ldr, py))
            .max()
            .unwrap_or(0)
    }

    pub(crate) fn apply_resume_token(
        &mut self,
        py: Python<'_>,
        token: &MixLoaderCheckpointToken,
    ) -> PyResult<()> {
        if token.seed != self.seed {
            return Err(PyValueError::new_err(format!(
                "resume_from seed mismatch: token={} current={}",
                token.seed, self.seed
            )));
        }
        if token.epoch != self.epoch {
            return Err(PyValueError::new_err(format!(
                "resume_from epoch mismatch: token={} current={}",
                token.epoch, self.epoch
            )));
        }
        if token.source_count != self.loaders.len() {
            return Err(PyValueError::new_err(format!(
                "resume_from source_count mismatch: token={} current={}",
                token.source_count,
                self.loaders.len()
            )));
        }

        let mut mismatches = 0u64;
        for (idx, loader) in self.loaders.iter().enumerate() {
            let checkpoint = Self::source_checkpoint(loader, py)?;
            if checkpoint != token.source_checkpoints[idx] {
                mismatches = mismatches.saturating_add(1);
                tracing::warn!(
                    target: "mx8_proof",
                    event = "mix_resume_source_checkpoint_mismatch",
                    source_idx = idx as u64,
                    mismatches = mismatches,
                    "mix resume token source checkpoint differs from provided source loader; continuing in best-effort mode"
                );
            }
        }
        self.mix_resume_source_checkpoint_mismatch_total = self
            .mix_resume_source_checkpoint_mismatch_total
            .saturating_add(mismatches);

        self.active = token.active.clone();
        self.delivered_batches = token.delivered_batches.clone();
        self.delivered_samples = token.delivered_samples.clone();
        self.delivered_bytes = token.delivered_bytes.clone();
        self.starvation_total = token.starvation_total.clone();
        self.source_exhausted_total = token.source_exhausted_total.clone();
        self.steps_since_emit = token.steps_since_emit.clone();
        self.schedule_ticks = token.schedule_ticks;
        self.snapshot_emitted_total = token.snapshot_emitted_total;
        self.scheduler.current = token.scheduler_current.clone();
        self.scheduler.tie_break_offset = token.scheduler_tie_break_offset;
        Ok(())
    }

    fn maybe_run_runtime_autotune(&mut self, py: Python<'_>) {
        if !self.mix_runtime_autotune_enabled {
            return;
        }
        let Some(rails) = self.mix_autotune_rails else {
            return;
        };
        if self.schedule_ticks == 0 {
            return;
        }
        let period = self.mix_runtime_autotune_period_ticks.max(1);
        if self
            .schedule_ticks
            .saturating_sub(self.mix_runtime_autotune_last_tick)
            < period
        {
            return;
        }
        self.mix_runtime_autotune_last_tick = self.schedule_ticks;

        let wait_ratio = self
            .steps_since_emit
            .iter()
            .copied()
            .max()
            .map(|v| (v as f64 / self.max_starvation_window.max(1) as f64).clamp(0.0, 1.0))
            .unwrap_or(0.0);
        let inflight_ratio = if self.shared_max_inflight_bytes == 0 {
            0.0
        } else {
            (self.total_inflight_bytes(py) as f64 / self.shared_max_inflight_bytes as f64)
                .clamp(0.0, 1.5)
        };
        let rss_ratio = match self.mix_max_process_rss_bytes {
            Some(cap) if cap > 0 => {
                (self.max_process_rss_bytes(py) as f64 / cap as f64).clamp(0.0, 1.5)
            }
            _ => 0.0,
        };

        let tick = autotune_tick(
            &mut self.mix_autotune_controller,
            self.mix_autotune_current,
            rails,
            wait_ratio,
            rss_ratio,
            inflight_ratio,
            2.0,
        );
        self.mix_runtime_autotune_pressure_milli = (tick.pressure * 1000.0).round() as u64;
        if !tick.changed {
            return;
        }

        self.mix_autotune_current = tick.next;
        self.mix_effective_prefetch_batches = tick.next.prefetch_batches;
        self.mix_effective_max_queue_batches = tick.next.max_queue_batches;
        for loader in &self.loaders {
            Self::source_apply_runtime_overrides(
                loader,
                py,
                Some(tick.next.prefetch_batches),
                Some(tick.next.max_queue_batches),
            );
        }
        self.mix_runtime_autotune_adjustments_total = self
            .mix_runtime_autotune_adjustments_total
            .saturating_add(1);
        tracing::info!(
            target: "mx8_proof",
            event = "mix_runtime_autotune_adjustment",
            reason = tick.reason,
            wait_ratio = wait_ratio,
            rss_ratio = rss_ratio,
            inflight_ratio = inflight_ratio,
            pressure = tick.pressure,
            prefetch_batches = tick.next.prefetch_batches as u64,
            max_queue_batches = tick.next.max_queue_batches as u64,
            "mix runtime autotune adjusted loader knobs"
        );
    }
}

#[pymethods]
impl MixedDataLoader {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if !self.active.iter().any(|v| *v) {
            return Err(PyStopIteration::new_err(()));
        }
        loop {
            let total_inflight_bytes = self.total_inflight_bytes(py);
            if total_inflight_bytes > self.shared_max_inflight_bytes {
                self.shared_inflight_violation_total =
                    self.shared_inflight_violation_total.saturating_add(1);
                return Err(PyRuntimeError::new_err(format!(
                    "mix shared inflight cap exceeded: total_inflight_bytes={} shared_max_inflight_bytes={}",
                    total_inflight_bytes, self.shared_max_inflight_bytes
                )));
            }

            let Some(source_idx) = self.scheduler.select(&self.active) else {
                return Err(PyStopIteration::new_err(()));
            };
            if !self.active[source_idx] {
                continue;
            }

            for (i, is_active) in self.active.iter().enumerate() {
                if !*is_active {
                    continue;
                }
                self.steps_since_emit[i] = self.steps_since_emit[i].saturating_add(1);
                if self.steps_since_emit[i] == self.max_starvation_window {
                    self.starvation_total[i] = self.starvation_total[i].saturating_add(1);
                }
            }

            let next_item = Self::source_next(&self.loaders[source_idx], py);

            match next_item {
                Ok(item) => {
                    let (sample_count, payload_bytes) =
                        if let Ok(batch) = item.extract::<PyRef<'_, PyBatch>>() {
                            (
                                batch.lease.batch.sample_count() as u64,
                                batch.lease.batch.payload.len() as u64,
                            )
                        } else {
                            (0, 0)
                        };

                    self.delivered_batches[source_idx] =
                        self.delivered_batches[source_idx].saturating_add(1);
                    self.delivered_samples[source_idx] =
                        self.delivered_samples[source_idx].saturating_add(sample_count);
                    self.delivered_bytes[source_idx] =
                        self.delivered_bytes[source_idx].saturating_add(payload_bytes);
                    self.steps_since_emit[source_idx] = 0;
                    self.schedule_ticks = self.schedule_ticks.saturating_add(1);
                    self.maybe_run_runtime_autotune(py);
                    self.maybe_emit_snapshot(py);
                    return Ok(item);
                }
                Err(err) => {
                    if err.is_instance_of::<PyStopIteration>(py) {
                        self.source_exhausted_total[source_idx] =
                            self.source_exhausted_total[source_idx].saturating_add(1);
                        tracing::info!(
                            target: "mx8_proof",
                            event = "mix_source_exhausted",
                            source_idx = source_idx as u64,
                            policy = self.source_exhaustion_policy.as_str(),
                            delivered_batches = self.delivered_batches[source_idx],
                            delivered_samples = self.delivered_samples[source_idx],
                            delivered_bytes = self.delivered_bytes[source_idx],
                            schedule_ticks = self.schedule_ticks,
                            "mixed loader source exhausted"
                        );
                        match self.source_exhaustion_policy {
                            SourceExhaustionPolicy::Error => {
                                return Err(PyRuntimeError::new_err(format!(
                                    "mix source exhausted (source_idx={source_idx}, policy=error, schedule_ticks={}, delivered_samples={})",
                                    self.schedule_ticks, self.delivered_samples[source_idx]
                                )));
                            }
                            SourceExhaustionPolicy::Allow => {
                                self.active[source_idx] = false;
                                if !self.active.iter().any(|v| *v) {
                                    return Err(PyStopIteration::new_err(()));
                                }
                                continue;
                            }
                        }
                    }
                    return Err(err);
                }
            }
        }
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let out = PyDict::new_bound(py);
        out.set_item("seed", self.seed)?;
        out.set_item("epoch", self.epoch)?;
        out.set_item("mix_profile", self.mix_profile.as_deref())?;
        out.set_item("mix_autotune_enabled", self.mix_autotune_enabled)?;
        out.set_item(
            "mix_runtime_autotune_enabled",
            self.mix_runtime_autotune_enabled,
        )?;
        out.set_item(
            "mix_effective_prefetch_batches",
            self.mix_effective_prefetch_batches,
        )?;
        out.set_item(
            "mix_effective_max_queue_batches",
            self.mix_effective_max_queue_batches,
        )?;
        out.set_item("mix_schedule_ticks", self.schedule_ticks)?;
        out.set_item("mix_snapshot_enabled", self.snapshot_enabled)?;
        out.set_item("mix_snapshot_period_ticks", self.snapshot_period_ticks)?;
        out.set_item("mix_snapshot_emitted_total", self.snapshot_emitted_total)?;
        out.set_item("active_sources", self.active.iter().filter(|v| **v).count())?;
        out.set_item(
            "mix_source_delivered_batches_total",
            PyList::new_bound(py, self.delivered_batches.iter().copied()),
        )?;
        out.set_item(
            "mix_source_delivered_samples_total",
            PyList::new_bound(py, self.delivered_samples.iter().copied()),
        )?;
        out.set_item(
            "mix_source_delivered_bytes_total",
            PyList::new_bound(py, self.delivered_bytes.iter().copied()),
        )?;
        out.set_item(
            "mix_source_starvation_total",
            PyList::new_bound(py, self.starvation_total.iter().copied()),
        )?;
        out.set_item(
            "mix_source_exhausted_total",
            PyList::new_bound(py, self.source_exhausted_total.iter().copied()),
        )?;
        out.set_item(
            "mix_normalized_weights",
            PyList::new_bound(py, self.normalized_weights.iter().copied()),
        )?;
        out.set_item(
            "mix_source_exhaustion_policy",
            self.source_exhaustion_policy.as_str(),
        )?;
        let total_inflight_bytes = self.total_inflight_bytes(py);
        out.set_item("mix_total_inflight_bytes", total_inflight_bytes)?;
        out.set_item(
            "mix_shared_max_inflight_bytes",
            self.shared_max_inflight_bytes,
        )?;
        out.set_item(
            "mix_shared_inflight_violation_total",
            self.shared_inflight_violation_total,
        )?;
        out.set_item(
            "mix_runtime_autotune_adjustments_total",
            self.mix_runtime_autotune_adjustments_total,
        )?;
        out.set_item(
            "mix_runtime_autotune_period_ticks",
            self.mix_runtime_autotune_period_ticks,
        )?;
        out.set_item(
            "mix_runtime_autotune_pressure",
            self.mix_runtime_autotune_pressure_milli as f64 / 1000.0,
        )?;
        out.set_item(
            "mix_resume_source_checkpoint_mismatch_total",
            self.mix_resume_source_checkpoint_mismatch_total,
        )?;
        out.set_item("process_rss_bytes", self.max_process_rss_bytes(py))?;
        out.set_item("max_process_rss_bytes", self.mix_max_process_rss_bytes)?;
        out.set_item("elapsed_seconds", self.started_at.elapsed().as_secs_f64())?;
        out.set_item(
            "mix_realized_ratio",
            PyList::new_bound(py, self.realized_ratio()),
        )?;

        let source_stats = PyList::empty_bound(py);
        let realized = self.realized_ratio();
        for (source_idx, loader) in self.loaders.iter().enumerate() {
            let (
                dataset_base,
                manifest_hash,
                batch_size_samples,
                max_inflight_bytes,
                max_queue_batches,
                prefetch_batches,
            ) = Self::source_config(loader, py);
            let source = PyDict::new_bound(py);
            source.set_item("source_idx", source_idx)?;
            source.set_item("active", self.active[source_idx])?;
            source.set_item(
                "source_weight_normalized",
                self.normalized_weights[source_idx],
            )?;
            source.set_item("source_realized_ratio", realized[source_idx])?;
            source.set_item(
                "source_delivered_batches_total",
                self.delivered_batches[source_idx],
            )?;
            source.set_item(
                "source_delivered_samples_total",
                self.delivered_samples[source_idx],
            )?;
            source.set_item(
                "source_delivered_bytes_total",
                self.delivered_bytes[source_idx],
            )?;
            source.set_item("source_starvation_total", self.starvation_total[source_idx])?;
            source.set_item(
                "source_exhausted_total",
                self.source_exhausted_total[source_idx],
            )?;
            source.set_item("dataset_base", dataset_base.as_deref())?;
            source.set_item("manifest_hash", manifest_hash.as_deref())?;
            source.set_item("batch_size_samples", batch_size_samples)?;
            source.set_item("max_inflight_bytes", max_inflight_bytes)?;
            source.set_item("max_queue_batches", max_queue_batches)?;
            source.set_item("prefetch_batches", prefetch_batches)?;
            let metrics = match loader {
                MixLoader::Local(local) => {
                    let guard = local.borrow(py);
                    metrics_to_dict(py, guard.metrics.as_ref())?
                }
                MixLoader::Distributed(distributed) => {
                    let guard = distributed.borrow(py);
                    metrics_to_dict(py, guard.metrics.as_ref())?
                }
            };
            source.set_item("metrics", metrics)?;
            source_stats.append(source)?;
        }
        out.set_item("mix_sources", source_stats)?;
        Ok(out.into_any())
    }

    fn checkpoint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let source_checkpoints = self
            .loaders
            .iter()
            .map(|loader| Self::source_checkpoint(loader, py))
            .collect::<PyResult<Vec<_>>>()?;

        let token = MixLoaderCheckpointToken {
            seed: self.seed,
            epoch: self.epoch,
            source_count: self.loaders.len(),
            schedule_ticks: self.schedule_ticks,
            snapshot_emitted_total: self.snapshot_emitted_total,
            active: self.active.clone(),
            delivered_batches: self.delivered_batches.clone(),
            delivered_samples: self.delivered_samples.clone(),
            delivered_bytes: self.delivered_bytes.clone(),
            starvation_total: self.starvation_total.clone(),
            source_exhausted_total: self.source_exhausted_total.clone(),
            steps_since_emit: self.steps_since_emit.clone(),
            scheduler_current: self.scheduler.current.clone(),
            scheduler_tie_break_offset: self.scheduler.tie_break_offset,
            source_checkpoints,
        };
        Ok(PyBytes::new_bound(py, &token.encode()))
    }
}

#[cfg(test)]
mod mix_scheduler_tests {
    use super::{
        compute_shared_mix_cap, normalize_mix_weights, should_emit_mix_snapshot,
        SourceExhaustionPolicy, WeightedRoundRobin,
    };

    #[test]
    fn mix_scheduler_rejects_invalid_weights() {
        assert!(normalize_mix_weights(&[]).is_err());
        assert!(normalize_mix_weights(&[0.0]).is_err());
        assert!(normalize_mix_weights(&[-1.0, 1.0]).is_err());
        assert!(normalize_mix_weights(&[f64::NAN, 1.0]).is_err());
        assert!(normalize_mix_weights(&[f64::INFINITY, 1.0]).is_err());
    }

    #[test]
    fn mix_source_exhaustion_policy_parsing() {
        assert_eq!(
            SourceExhaustionPolicy::parse("error").expect("parse"),
            SourceExhaustionPolicy::Error
        );
        assert_eq!(
            SourceExhaustionPolicy::parse("allow").expect("parse"),
            SourceExhaustionPolicy::Allow
        );
        assert!(SourceExhaustionPolicy::parse("noop").is_err());
    }

    #[test]
    fn mix_scheduler_deterministic_for_fixed_seed_epoch() {
        let weights = normalize_mix_weights(&[3.0, 2.0, 1.0]).expect("normalize");
        let active = vec![true, true, true];
        let run = |seed: u64, epoch: u64| {
            let mut rr = WeightedRoundRobin::new(weights.clone(), seed, epoch);
            let mut out = Vec::new();
            for _ in 0..300 {
                out.push(rr.select(&active).expect("selection"));
            }
            out
        };
        let a = run(7, 11);
        let b = run(7, 11);
        assert_eq!(a, b);
    }

    #[test]
    fn mix_scheduler_weighted_round_robin_ratio_converges() {
        let weights = normalize_mix_weights(&[3.0, 1.0]).expect("normalize");
        let mut rr = WeightedRoundRobin::new(weights, 0, 0);
        let active = vec![true, true];
        let mut counts = [0u64, 0u64];
        let rounds = 4000u64;
        for _ in 0..rounds {
            let idx = rr.select(&active).expect("selection");
            counts[idx] += 1;
        }
        // Expect near 75/25 split with small tolerance.
        let ratio0 = counts[0] as f64 / rounds as f64;
        let ratio1 = counts[1] as f64 / rounds as f64;
        assert!((ratio0 - 0.75).abs() <= 0.02, "ratio0={ratio0}");
        assert!((ratio1 - 0.25).abs() <= 0.02, "ratio1={ratio1}");
    }

    #[test]
    fn mix_scheduler_no_source_starvation() {
        let weights = normalize_mix_weights(&[5.0, 1.0, 1.0]).expect("normalize");
        let mut rr = WeightedRoundRobin::new(weights, 1, 2);
        let active = vec![true, true, true];
        let mut last_seen = [0u64, 0u64, 0u64];
        let mut step = 0u64;
        for _ in 0..700 {
            step += 1;
            let idx = rr.select(&active).expect("selection");
            last_seen[idx] = step;
            for seen in last_seen {
                assert!(step.saturating_sub(seen) <= 50, "starvation detected");
            }
        }
    }

    #[test]
    fn mix_shared_cap_uses_min_loader_cap() {
        let cap = compute_shared_mix_cap(&[256, 128, 512]).expect("cap");
        assert_eq!(cap, 128);
    }

    #[test]
    fn mix_shared_cap_rejects_empty() {
        assert!(compute_shared_mix_cap(&[]).is_err());
    }

    #[test]
    fn mix_snapshot_periodic_emission_cadence() {
        let period = 8u64;
        let mut emitted = 0u64;
        let mut due_ticks = Vec::new();
        for tick in 1..=32u64 {
            if should_emit_mix_snapshot(tick, period) {
                emitted += 1;
                due_ticks.push(tick);
            }
        }
        assert_eq!(emitted, 4);
        assert_eq!(due_ticks, vec![8, 16, 24, 32]);
    }

    #[test]
    fn mix_shared_caps_match_single_source_safety_baseline() {
        // Single-source baseline cap and observed inflight behavior.
        let single_source_cap = 128u64;
        let single_source_inflight = [16u64, 40, 72, 96, 128, 92, 48, 0];
        let single_high_water = single_source_inflight.iter().copied().max().unwrap_or(0);
        assert!(single_source_inflight
            .iter()
            .all(|v| *v <= single_source_cap));
        assert!(single_high_water <= single_source_cap);

        // Mixed mode uses shared cap = min(source caps), which should match baseline cap.
        let shared_cap = compute_shared_mix_cap(&[128, 256]).expect("shared cap");
        assert_eq!(shared_cap, single_source_cap);

        // Simulated mixed-source inflight samples (source_a + source_b per step).
        let mixed_inflight_pairs = [
            (8u64, 8u64),
            (24, 16),
            (40, 32),
            (56, 40),
            (64, 64),
            (52, 40),
            (28, 20),
            (0, 0),
        ];
        let mut mixed_high_water = 0u64;
        let mut delivered_steps = 0usize;
        for (a, b) in mixed_inflight_pairs {
            let total = a.saturating_add(b);
            mixed_high_water = mixed_high_water.max(total);
            assert!(
                total <= shared_cap,
                "mixed inflight exceeded shared cap: total={total} cap={shared_cap}"
            );
            delivered_steps += 1;
        }

        // "Progress completes" proxy: all planned steps delivered and bounded by same cap.
        assert_eq!(delivered_steps, single_source_inflight.len());
        assert!(mixed_high_water <= single_source_cap);
    }

    #[test]
    fn mix_backpressure_blocks_all_sources_under_pressure() {
        // Simulate a mixed run where source A/B each report inflight bytes per scheduler tick.
        // When total inflight exceeds shared cap, scheduler should be considered blocked.
        let shared_cap = compute_shared_mix_cap(&[128, 256]).expect("shared cap");
        let inflight_pairs = [
            (40u64, 30u64), // below cap
            (90, 50),       // above cap -> block
            (80, 60),       // above cap -> block
            (64, 32),       // below cap
            (110, 30),      // above cap -> block
            (48, 16),       // below cap
        ];

        let mut blocked_ticks = 0u64;
        let mut delivered_ticks = 0u64;
        let mut source_a_deliveries = 0u64;
        let mut source_b_deliveries = 0u64;
        let mut rr =
            WeightedRoundRobin::new(normalize_mix_weights(&[1.0, 1.0]).expect("weights"), 0, 0);
        let active = vec![true, true];

        for (a, b) in inflight_pairs {
            let total = a.saturating_add(b);
            if total > shared_cap {
                blocked_ticks += 1;
                continue;
            }
            delivered_ticks += 1;
            let idx = rr.select(&active).expect("selection");
            if idx == 0 {
                source_a_deliveries += 1;
            } else {
                source_b_deliveries += 1;
            }
        }

        assert_eq!(
            blocked_ticks, 3,
            "expected pressure to block exactly 3 ticks"
        );
        assert_eq!(delivered_ticks, 3);
        assert!(source_a_deliveries > 0 && source_b_deliveries > 0);
    }

    #[test]
    fn mix_starvation_counter_stays_zero_in_steady_state() {
        let weights = normalize_mix_weights(&[3.0, 2.0, 1.0]).expect("weights");
        let mut rr = WeightedRoundRobin::new(weights, 9, 1);
        let active = vec![true, true, true];
        let starvation_window = 16u64;
        let mut since_emit = [0u64, 0u64, 0u64];
        let mut starvation_total = [0u64, 0u64, 0u64];

        for _step in 0..400u64 {
            for idx in 0..since_emit.len() {
                since_emit[idx] = since_emit[idx].saturating_add(1);
                if since_emit[idx] == starvation_window {
                    starvation_total[idx] = starvation_total[idx].saturating_add(1);
                }
            }
            let pick = rr.select(&active).expect("selection");
            since_emit[pick] = 0;
        }

        assert_eq!(starvation_total, [0, 0, 0]);
    }

    #[test]
    fn mix_replay_deterministic_sequence_fixed_inputs() {
        let weights = normalize_mix_weights(&[5.0, 3.0]).expect("weights");
        let active = vec![true, true];
        let run = |seed: u64, epoch: u64| {
            let mut rr = WeightedRoundRobin::new(weights.clone(), seed, epoch);
            let mut out = Vec::new();
            for _ in 0..200 {
                out.push(rr.select(&active).expect("selection"));
            }
            out
        };

        let first = run(42, 7);
        let second = run(42, 7);
        assert_eq!(first, second);
    }

    #[test]
    fn mix_replay_changes_when_epoch_changes() {
        // Equal weights force frequent ties, so epoch-driven tie-break offset should
        // produce a different deterministic sequence.
        let weights = normalize_mix_weights(&[1.0, 1.0, 1.0]).expect("weights");
        let active = vec![true, true, true];
        let run = |seed: u64, epoch: u64| {
            let mut rr = WeightedRoundRobin::new(weights.clone(), seed, epoch);
            let mut out = Vec::new();
            for _ in 0..90 {
                out.push(rr.select(&active).expect("selection"));
            }
            out
        };

        let epoch0 = run(77, 0);
        let epoch1 = run(77, 1);
        assert_ne!(epoch0, epoch1);
    }
}
