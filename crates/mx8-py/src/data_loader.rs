use super::*;

#[pyclass]
pub(crate) struct DataLoader {
    pub(crate) dataset_base: String,
    pub(crate) manifest_hash: String,
    pub(crate) batch_size_samples: usize,
    pub(crate) max_inflight_bytes: u64,
    pub(crate) max_process_rss_bytes: Option<u64>,
    pub(crate) max_queue_batches: usize,
    pub(crate) prefetch_batches: usize,
    pub(crate) pipeline: Pipeline,
    pub(crate) metrics: Arc<RuntimeMetrics>,
    pub(crate) autotune: Arc<AutotuneShared>,
    pub(crate) rx: tokio::sync::mpsc::Receiver<BatchLease>,
    pub(crate) task: Option<tokio::task::JoinHandle<Result<()>>>,
    pub(crate) autotune_task: Option<tokio::task::JoinHandle<()>>,
    pub(crate) rt: tokio::runtime::Runtime,
    pub(crate) checkpoint_supported: bool,
    pub(crate) checkpoint_schema_version: u32,
    pub(crate) checkpoint_epoch: u32,
    pub(crate) checkpoint_next_sample_id: u64,
    pub(crate) checkpoint_end_id: Option<u64>,
    pub(crate) started_at: Instant,
    pub(crate) transform: Option<CompiledTransform>,
    pub(crate) transform_samples_total: u64,
    pub(crate) transform_failures_total: u64,
    pub(crate) transform_timeout_total: u64,
    pub(crate) transform_output_ratio_ewma_milli: u64,
    pub(crate) transform_heavy_mode_active: bool,
}

#[pymethods]
impl DataLoader {
    #[new]
    #[pyo3(signature = (
        dataset_link,
        *,
        manifest_store=None,
        manifest_path=None,
        recursive=true,
        batch_size_samples=512,
        max_inflight_bytes=128*1024*1024,
        max_queue_batches=64,
        prefetch_batches=1,
        target_batch_bytes=None,
        max_batch_bytes=None,
        max_ram_bytes=None,
        start_id=None,
        end_id=None,
        resume_from=None,
        node_id=None,
        profile=None,
        autotune=None,
        transform=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        dataset_link: String,
        manifest_store: Option<PathBuf>,
        manifest_path: Option<PathBuf>,
        recursive: bool,
        batch_size_samples: usize,
        max_inflight_bytes: u64,
        max_queue_batches: usize,
        prefetch_batches: usize,
        target_batch_bytes: Option<u64>,
        max_batch_bytes: Option<u64>,
        max_ram_bytes: Option<u64>,
        start_id: Option<u64>,
        end_id: Option<u64>,
        resume_from: Option<Vec<u8>>,
        node_id: Option<String>,
        profile: Option<String>,
        autotune: Option<bool>,
        transform: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let parsed = mx8_core::dataset_link::DatasetLink::parse(&dataset_link)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        let resume_token = resume_from
            .as_deref()
            .map(DataLoaderCheckpointToken::decode)
            .transpose()
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        let dataset_base = match &parsed {
            mx8_core::dataset_link::DatasetLink::Plain(b) => b.clone(),
            mx8_core::dataset_link::DatasetLink::Refresh(b) => b.clone(),
            mx8_core::dataset_link::DatasetLink::Pinned { base, .. } => base.clone(),
        };

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("failed to start tokio runtime: {e}")))?;

        let root = manifest_store
            .or(env_path("MX8_MANIFEST_STORE_ROOT"))
            .unwrap_or_else(default_manifest_store);

        let dev_manifest_path = manifest_path.or(env_path("MX8_DEV_MANIFEST_PATH"));

        let selected_profile = AutotuneProfile::from_name(profile.as_deref());
        let mut max_process_rss_bytes_cap =
            max_ram_bytes.or_else(|| env_u64("MX8_MAX_PROCESS_RSS_BYTES"));
        if max_process_rss_bytes_cap.is_none() {
            max_process_rss_bytes_cap =
                derive_default_max_process_rss_bytes(selected_profile, max_inflight_bytes);
            if let Some(cap) = max_process_rss_bytes_cap {
                tracing::info!(
                    target: "mx8_proof",
                    event = "rss_cap_defaulted",
                    mode = "single_node_legacy",
                    profile = profile_name(selected_profile),
                    max_inflight_bytes = max_inflight_bytes,
                    max_process_rss_bytes = cap,
                    "defaulted process RSS cap from profile and node memory limit"
                );
            }
        }
        let (effective_target_batch_bytes, effective_max_batch_bytes) =
            derive_byte_batch_caps(max_inflight_bytes, target_batch_bytes, max_batch_bytes);
        let caps = RuntimeCaps {
            max_inflight_bytes,
            max_queue_batches,
            batch_size_samples,
            prefetch_batches,
            target_batch_bytes: effective_target_batch_bytes,
            max_batch_bytes: effective_max_batch_bytes,
            max_process_rss_bytes: max_process_rss_bytes_cap,
        };
        let pipeline = Pipeline::new(caps);
        let metrics = pipeline.metrics();
        let autotune_enabled = autotune.unwrap_or(true);
        let autotune_profile = if autotune_enabled {
            Some(selected_profile)
        } else {
            None
        };
        let rails = autotune_profile.map(AutotuneRails::for_profile);
        let initial_prefetch = rails
            .map(|r| {
                prefetch_batches
                    .max(r.min_prefetch_batches)
                    .min(r.max_prefetch_batches)
            })
            .unwrap_or(prefetch_batches.max(1));
        let initial_max_queue = rails
            .map(|r| {
                max_queue_batches
                    .max(r.min_max_queue_batches)
                    .min(r.max_max_queue_batches)
            })
            .unwrap_or(max_queue_batches.max(1));
        pipeline.set_prefetch_batches(initial_prefetch);
        pipeline.set_max_queue_batches(initial_max_queue);
        let autotune = Arc::new(AutotuneShared::new(
            autotune_enabled,
            1,
            initial_prefetch,
            initial_max_queue,
        ));
        if autotune_enabled {
            tracing::info!(
                target: "mx8_proof",
                event = "load_runtime_autotune_initialized",
                profile = match autotune_profile.unwrap_or(AutotuneProfile::Balanced) {
                    AutotuneProfile::Safe => "safe",
                    AutotuneProfile::Balanced => "balanced",
                    AutotuneProfile::Throughput => "throughput",
                },
                prefetch_batches = initial_prefetch as u64,
                max_queue_batches = initial_max_queue as u64,
                max_inflight_bytes = max_inflight_bytes,
                max_process_rss_bytes = max_process_rss_bytes_cap.unwrap_or(0),
                "load runtime autotune initialized"
            );
        }
        let autotune_task = if autotune_enabled {
            let net_runtime = build_net_pressure_source();
            autotune
                .net_disabled_total
                .fetch_add(net_runtime.disabled_total_seed, Ordering::Relaxed);
            Some(rt.spawn(autotune_loop(
                Arc::new(pipeline.clone()),
                metrics.clone(),
                autotune.clone(),
                max_inflight_bytes,
                max_process_rss_bytes_cap,
                rails.unwrap_or(AutotuneRails::for_profile(AutotuneProfile::Balanced)),
                net_runtime.source,
            )))
        } else {
            None
        };

        let compiled_transform = transform
            .as_ref()
            .map(|transform_fn| {
                Python::with_gil(|py| {
                    CompiledTransform::compile_from_py(
                        py,
                        &transform_fn.bind(py),
                        max_inflight_bytes,
                    )
                })
            })
            .transpose()?;

        if should_use_zero_manifest_scan(&parsed, &dataset_base) {
            if resume_token.is_some() {
                return Err(PyValueError::new_err(
                    "resume_from is not supported for zero-manifest scan datasets; pack/pin first",
                ));
            }
            let reservoir_size = env_usize("MX8_ZERO_MANIFEST_RESERVOIR", 100_000).max(1);
            let scan_result = if dataset_base.starts_with("gs://") {
                rt.block_on(async {
                    pipeline
                        .spawn_gcs_scan(&dataset_base, recursive, reservoir_size, start_id, end_id)
                        .await
                })
            } else {
                rt.block_on(async {
                    pipeline
                        .spawn_s3_scan(&dataset_base, recursive, reservoir_size, start_id, end_id)
                        .await
                })
            };
            match scan_result {
                Ok((rx, task)) => {
                    let backend = if dataset_base.starts_with("gs://") {
                        "gcs"
                    } else {
                        "s3"
                    };
                    tracing::info!(
                        target: "mx8_proof",
                        event = "zero_manifest_scan_enabled",
                        dataset_base = %dataset_base,
                        recursive = recursive,
                        reservoir_size = reservoir_size as u64,
                        backend = backend,
                        "using zero-manifest scan path"
                    );

                    let manifest_hash = format!(
                        "scan:{}",
                        mx8_manifest_store::sha256_hex(dataset_base.as_bytes())
                    );
                    return Ok(Self {
                        dataset_base,
                        manifest_hash,
                        batch_size_samples,
                        max_inflight_bytes,
                        max_process_rss_bytes: max_process_rss_bytes_cap,
                        max_queue_batches: initial_max_queue,
                        prefetch_batches: initial_prefetch,
                        pipeline,
                        metrics,
                        autotune,
                        rx,
                        task: Some(task),
                        autotune_task,
                        rt,
                        checkpoint_supported: false,
                        checkpoint_schema_version: mx8_core::types::MANIFEST_SCHEMA_VERSION,
                        checkpoint_epoch: DATA_CHECKPOINT_EPOCH,
                        checkpoint_next_sample_id: 0,
                        checkpoint_end_id: None,
                        started_at: Instant::now(),
                        transform: compiled_transform,
                        transform_samples_total: 0,
                        transform_failures_total: 0,
                        transform_timeout_total: 0,
                        transform_output_ratio_ewma_milli: 1000,
                        transform_heavy_mode_active: false,
                    });
                }
                Err(err) => {
                    let msg = err.to_string();
                    if msg.contains("precomputed_manifest_exists") {
                        tracing::info!(
                            target: "mx8_proof",
                            event = "zero_manifest_scan_fallback_precomputed_manifest",
                            dataset_base = %dataset_base,
                            "falling back to snapshot/manifest path"
                        );
                    } else {
                        return Err(PyValueError::new_err(format!("{err}")));
                    }
                }
            }
        }

        let store = std::sync::Arc::new(FsManifestStore::new(root));
        let resolver = SnapshotResolver::new(
            store.clone(),
            SnapshotResolverConfig {
                dev_manifest_path,
                recursive,
                materialize_manifest_bytes: true,
                ..SnapshotResolverConfig::default()
            },
        );

        let mut snapshot = resolver
            .resolve(
                &dataset_link,
                LockOwner {
                    node_id: node_id.clone(),
                },
            )
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;

        if !snapshot.manifest_bytes_materialized {
            snapshot.manifest_bytes =
                store
                    .get_manifest_bytes(&snapshot.manifest_hash)
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "failed to materialize manifest bytes for {}: {e}",
                            snapshot.manifest_hash.0
                        ))
                    })?;
            snapshot.manifest_bytes_materialized = true;
        }

        let (schema_version, total_samples) =
            manifest_schema_version_and_sample_count(&snapshot.manifest_bytes)
                .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        if schema_version != mx8_core::types::MANIFEST_SCHEMA_VERSION {
            return Err(PyValueError::new_err(format!(
                "unsupported schema_version {schema_version}"
            )));
        }

        if start_id.is_some() ^ end_id.is_some() {
            return Err(PyValueError::new_err(
                "start_id and end_id must be set together",
            ));
        }
        let requested_start = start_id.unwrap_or(0);
        let requested_end = end_id.unwrap_or(total_samples);
        if requested_start > requested_end {
            return Err(PyValueError::new_err("invalid range (start_id > end_id)"));
        }
        if requested_end > total_samples {
            return Err(PyValueError::new_err("end_id out of range"));
        }

        let mut effective_start = requested_start;
        if let Some(token) = &resume_token {
            if token.schema_version != schema_version {
                return Err(PyValueError::new_err(format!(
                    "resume_from schema_version mismatch: token={} current={}",
                    token.schema_version, schema_version
                )));
            }
            if token.epoch != DATA_CHECKPOINT_EPOCH {
                return Err(PyValueError::new_err(format!(
                    "resume_from epoch mismatch: token={} current={}",
                    token.epoch, DATA_CHECKPOINT_EPOCH
                )));
            }
            if token.manifest_hash != snapshot.manifest_hash.0 {
                return Err(PyValueError::new_err(format!(
                    "resume_from manifest_hash mismatch: token={} current={}",
                    token.manifest_hash, snapshot.manifest_hash.0
                )));
            }
            if token.end_id != requested_end {
                return Err(PyValueError::new_err(format!(
                    "resume_from end_id mismatch: token={} current={}",
                    token.end_id, requested_end
                )));
            }
            if token.next_sample_id < requested_start || token.next_sample_id > requested_end {
                return Err(PyValueError::new_err(format!(
                    "resume_from next_sample_id {} out of requested range [{}, {})",
                    token.next_sample_id, requested_start, requested_end
                )));
            }
            effective_start = token.next_sample_id;
        }

        let pipeline_for_stream = pipeline.clone();
        let manifest_bytes = snapshot.manifest_bytes;
        let (rx, task) = rt
            .block_on(async move {
                pipeline_for_stream
                    .spawn_manifest_bytes_range_stream(
                        manifest_bytes,
                        effective_start,
                        requested_end,
                    )
                    .await
            })
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;

        Ok(Self {
            dataset_base,
            manifest_hash: snapshot.manifest_hash.0,
            batch_size_samples,
            max_inflight_bytes,
            max_process_rss_bytes: max_process_rss_bytes_cap,
            max_queue_batches: initial_max_queue,
            prefetch_batches: initial_prefetch,
            pipeline,
            metrics,
            autotune,
            rx,
            task: Some(task),
            autotune_task,
            rt,
            checkpoint_supported: true,
            checkpoint_schema_version: schema_version,
            checkpoint_epoch: DATA_CHECKPOINT_EPOCH,
            checkpoint_next_sample_id: effective_start,
            checkpoint_end_id: Some(requested_end),
            started_at: Instant::now(),
            transform: compiled_transform,
            transform_samples_total: 0,
            transform_failures_total: 0,
            transform_timeout_total: 0,
            transform_output_ratio_ewma_milli: 1000,
            transform_heavy_mode_active: false,
        })
    }

    #[getter]
    fn manifest_hash(&self) -> &str {
        &self.manifest_hash
    }

    pub(crate) fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let out = metrics_to_dict(py, self.metrics.as_ref())?;
        out.set_item("batch_size_samples", self.batch_size_samples)?;
        out.set_item("max_inflight_bytes", self.max_inflight_bytes)?;
        out.set_item("max_process_rss_bytes", self.max_process_rss_bytes)?;
        out.set_item("max_queue_batches", self.max_queue_batches)?;
        out.set_item("prefetch_batches", self.prefetch_batches)?;
        out.set_item("elapsed_seconds", self.started_at.elapsed().as_secs_f64())?;
        out.set_item(
            "effective_prefetch_batches",
            self.pipeline.effective_prefetch_batches(),
        )?;
        out.set_item(
            "effective_max_queue_batches",
            self.pipeline.effective_max_queue_batches(),
        )?;
        out.set_item("autotune_enabled", self.autotune.enabled)?;
        out.set_item(
            "autotune_pressure",
            self.autotune.pressure_milli.load(Ordering::Relaxed) as f64 / 1000.0,
        )?;
        out.set_item(
            "autotune_wait_ewma",
            self.autotune.wait_ewma_milli.load(Ordering::Relaxed) as f64 / 1000.0,
        )?;
        out.set_item(
            "autotune_rss_ewma",
            self.autotune.rss_ewma_milli.load(Ordering::Relaxed) as f64 / 1000.0,
        )?;
        out.set_item(
            "autotune_integral_rss",
            self.autotune.integral_rss_milli.load(Ordering::Relaxed) as f64 / 1000.0,
        )?;
        out.set_item(
            "autotune_cooldown_ticks",
            self.autotune.cooldown_ticks.load(Ordering::Relaxed),
        )?;
        out.set_item(
            "autotune_net_pressure_ratio",
            self.autotune
                .net_pressure_ratio_milli
                .load(Ordering::Relaxed) as f64
                / 1000.0,
        )?;
        out.set_item(
            "autotune_net_signal_age_ms",
            self.autotune.net_signal_age_ms.load(Ordering::Relaxed),
        )?;
        out.set_item(
            "autotune_net_signal_stale_total",
            self.autotune.net_signal_stale_total.load(Ordering::Relaxed),
        )?;
        out.set_item(
            "autotune_net_assisted_backoff_total",
            self.autotune
                .net_assisted_backoff_total
                .load(Ordering::Relaxed),
        )?;
        out.set_item(
            "autotune_net_disabled_total",
            self.autotune.net_disabled_total.load(Ordering::Relaxed),
        )?;
        out.set_item("transform_enabled", self.transform.is_some())?;
        out.set_item("transform_samples_total", self.transform_samples_total)?;
        out.set_item("transform_failures_total", self.transform_failures_total)?;
        out.set_item("transform_timeout_total", self.transform_timeout_total)?;
        out.set_item(
            "transform_output_ratio_ewma",
            self.transform_output_ratio_ewma_milli as f64 / 1000.0,
        )?;
        out.set_item(
            "transform_heavy_mode_active",
            self.transform_heavy_mode_active,
        )?;
        Ok(out.into_any())
    }

    /// Print a single human-readable summary line to stderr.
    /// Call inside your training loop whenever you want a quick health check
    /// without building your own stat formatter.
    ///
    /// Example output:
    ///   [mx8] batches=128 samples=8192 inflight=45.2MB rss=234.1MB hwm=256.0MB
    pub(crate) fn print_stats(&self, py: Python<'_>) -> PyResult<()> {
        let s = self.stats(py)?;
        let s = s.downcast::<pyo3::types::PyDict>()?;
        let get_u64 = |key: &str| -> u64 {
            s.get_item(key)
                .ok()
                .flatten()
                .and_then(|v| v.extract::<u64>().ok())
                .unwrap_or(0)
        };
        let mb = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
        let batches = get_u64("delivered_batches_total");
        let samples = get_u64("delivered_samples_total");
        let inflight = get_u64("inflight_bytes");
        let rss = get_u64("process_rss_bytes");
        let hwm = get_u64("ram_high_water_bytes");
        eprintln!(
            "[mx8] batches={batches} samples={samples} inflight={:.1}MB rss={:.1}MB hwm={:.1}MB",
            mb(inflight),
            mb(rss),
            mb(hwm),
        );
        Ok(())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    pub(crate) fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let wait_started = Instant::now();
        let lease = py.allow_threads(|| self.rt.block_on(async { self.rx.recv().await }));
        let out = match lease {
            Some(mut lease) => {
                let extra_inflight_lease = self.apply_transform_if_enabled(&mut lease)?;
                self.on_batch_delivered(&lease);
                let out = Py::new(
                    py,
                    PyBatch {
                        lease,
                        _extra_inflight_lease: extra_inflight_lease,
                    },
                )?;
                Ok(out.into_bound(py).into_any())
            }
            None => {
                let Some(task) = self.task.take() else {
                    return Err(PyStopIteration::new_err(()));
                };
                let out = py.allow_threads(|| self.rt.block_on(task));
                match out {
                    Ok(Ok(())) => Err(PyStopIteration::new_err(())),
                    Ok(Err(err)) => Err(PyRuntimeError::new_err(format!("{err}"))),
                    Err(err) => Err(PyRuntimeError::new_err(format!(
                        "producer task failed: {err}"
                    ))),
                }
            }
        };
        self.autotune.on_wait(wait_started.elapsed());
        out
    }

    pub(crate) fn checkpoint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        if !self.checkpoint_supported {
            return Err(PyRuntimeError::new_err(
                "checkpoint is unavailable for zero-manifest scan loaders; use packed/pinned datasets",
            ));
        }
        let end_id = self
            .checkpoint_end_id
            .ok_or_else(|| PyRuntimeError::new_err("checkpoint end_id unavailable"))?;
        let token = DataLoaderCheckpointToken {
            manifest_hash: self.manifest_hash.clone(),
            schema_version: self.checkpoint_schema_version,
            epoch: self.checkpoint_epoch,
            next_sample_id: self.checkpoint_next_sample_id.min(end_id),
            end_id,
        };
        Ok(PyBytes::new_bound(py, &token.encode()))
    }

    pub(crate) fn close(&mut self) {
        if let Some(handle) = self.task.take() {
            handle.abort();
        }
        if let Some(handle) = self.autotune_task.take() {
            handle.abort();
        }
    }

    fn __del__(&mut self) {
        self.close();
    }
}

impl DataLoader {
    pub(crate) fn on_batch_delivered(&mut self, lease: &BatchLease) {
        if !self.checkpoint_supported {
            return;
        }
        if let Some(max_sample_id) = lease.batch.sample_ids.iter().copied().max() {
            let next = max_sample_id.saturating_add(1);
            if next > self.checkpoint_next_sample_id {
                self.checkpoint_next_sample_id = next;
            }
        }
    }

    pub(crate) fn apply_runtime_overrides(
        &mut self,
        prefetch_batches: Option<usize>,
        max_queue_batches: Option<usize>,
    ) {
        if let Some(prefetch) = prefetch_batches {
            let clamped = prefetch.max(1);
            self.prefetch_batches = clamped;
            self.pipeline.set_prefetch_batches(clamped);
        }
        if let Some(max_queue) = max_queue_batches {
            let clamped = max_queue.max(1);
            self.max_queue_batches = clamped;
            self.pipeline.set_max_queue_batches(clamped);
        }
    }

    fn apply_transform_if_enabled(
        &mut self,
        lease: &mut BatchLease,
    ) -> PyResult<Option<ExtraInFlightLease>> {
        let transform = match self.transform.as_ref() {
            Some(t) => t,
            _ => return Ok(None),
        };

        let out = transform.apply(&lease.batch).map_err(|e| {
            self.transform_failures_total = self.transform_failures_total.saturating_add(1);
            if e.timeout {
                self.transform_timeout_total = self.transform_timeout_total.saturating_add(1);
            }
            PyRuntimeError::new_err(format!("transform `{}` failed: {}", transform.name, e))
        })?;

        self.transform_samples_total = self
            .transform_samples_total
            .saturating_add(out.stats.transformed_samples);
        if self.transform_output_ratio_ewma_milli == 0 {
            self.transform_output_ratio_ewma_milli = out.stats.output_ratio_milli;
        } else {
            self.transform_output_ratio_ewma_milli =
                ((self.transform_output_ratio_ewma_milli.saturating_mul(80))
                    .saturating_add(out.stats.output_ratio_milli.saturating_mul(20)))
                    / 100;
        }
        self.transform_heavy_mode_active = out.stats.heavy_mode_active;
        if self.transform_heavy_mode_active && self.pipeline.effective_prefetch_batches() > 1 {
            self.pipeline.set_prefetch_batches(1);
        }

        let output_bytes = out.batch.payload.len() as u64;
        let extra_inflight_lease = if output_bytes > lease.bytes {
            let extra = output_bytes - lease.bytes;
            Some(
                self.rt
                    .block_on(self.pipeline.reserve_extra_inflight(extra))
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "transform inflight reservation failed for +{} bytes: {e}",
                            extra
                        ))
                    })?
                    .ok_or_else(|| {
                        PyRuntimeError::new_err("transform inflight reservation unexpectedly empty")
                    })?,
            )
        } else {
            None
        };

        lease.batch = out.batch;

        Ok(extra_inflight_lease)
    }
}

#[pyclass]
pub(crate) struct PyBatch {
    pub(crate) lease: BatchLease,
    pub(crate) _extra_inflight_lease: Option<ExtraInFlightLease>,
}

#[pymethods]
impl PyBatch {
    #[getter]
    fn sample_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let ids: Vec<u64> = self.lease.batch.sample_ids.iter().copied().collect();
        Ok(PyList::new_bound(py, ids))
    }

    #[getter]
    fn offsets<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let offsets: Vec<u64> = self.lease.batch.offsets.iter().copied().collect();
        Ok(PyList::new_bound(py, offsets))
    }

    #[getter]
    fn payload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        // v0: keep Python surface area simple; this copies bytes into Python-owned memory.
        //
        // Backpressure is still preserved (the underlying lease is held by this object),
        // but total process RSS may exceed `max_inflight_bytes` if the consumer also holds
        // onto many copied payloads.
        Ok(PyBytes::new_bound(py, &self.lease.batch.payload))
    }

    #[getter]
    fn label_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.lease.batch.label_ids {
            Some(ids) => {
                let out: Vec<u64> = ids.iter().copied().collect();
                Ok(PyList::new_bound(py, out).into_any())
            }
            None => Ok(py.None().into_bound(py).into_any()),
        }
    }

    fn to_torch<'py>(&self, py: Python<'py>) -> PyResult<TorchBatch3<'py>> {
        let torch = py.import_bound("torch").map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to import torch (install PyTorch to use to_torch): {e}"
            ))
        })?;

        let torch_uint8 = torch.getattr("uint8")?;
        let torch_int64 = torch.getattr("int64")?;

        // Payload: one contiguous uint8 buffer.
        //
        // v0: copy bytes into a Python-owned bytearray, then build a torch tensor view on it.
        // This is not zero-copy from the Rust runtime buffer, but it keeps the PyTorch API
        // frictionless while we validate product fit.
        let payload = PyByteArray::new_bound(py, &self.lease.batch.payload);
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("dtype", &torch_uint8)?;
        let payload_u8 = torch.call_method("frombuffer", (payload,), Some(&kwargs))?;

        // Offsets: convert u64 -> i64 for torch indexing.
        let mut offsets_i64 = Vec::with_capacity(self.lease.batch.offsets.len());
        for &off in self.lease.batch.offsets.iter() {
            let off_i64 = i64::try_from(off).map_err(|_| {
                PyValueError::new_err(format!(
                    "offset overflow converting u64 -> i64 (offset={off})"
                ))
            })?;
            offsets_i64.push(off_i64);
        }
        let offsets_list = PyList::new_bound(py, offsets_i64);
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("dtype", &torch_int64)?;
        let offsets_i64 = torch.call_method("tensor", (offsets_list,), Some(&kwargs))?;

        // Sample IDs: convert u64 -> i64 (common consumer type in Python).
        let mut sample_ids_i64 = Vec::with_capacity(self.lease.batch.sample_ids.len());
        for &sid in self.lease.batch.sample_ids.iter() {
            let sid_i64 = i64::try_from(sid).map_err(|_| {
                PyValueError::new_err(format!(
                    "sample_id overflow converting u64 -> i64 (sample_id={sid})"
                ))
            })?;
            sample_ids_i64.push(sid_i64);
        }
        let ids_list = PyList::new_bound(py, sample_ids_i64);
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("dtype", &torch_int64)?;
        let sample_ids_i64 = torch.call_method("tensor", (ids_list,), Some(&kwargs))?;

        Ok((payload_u8, offsets_i64, sample_ids_i64))
    }

    fn to_torch_with_labels<'py>(&self, py: Python<'py>) -> PyResult<TorchBatch4<'py>> {
        let (payload_u8, offsets_i64, sample_ids_i64) = self.to_torch(py)?;

        let ids = self.lease.batch.label_ids.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "batch has no label_ids (expected mx8:vision:imagefolder label hints in manifest)",
            )
        })?;

        let torch = py.import_bound("torch").map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to import torch (install PyTorch to use to_torch_with_labels): {e}"
            ))
        })?;
        let torch_int64 = torch.getattr("int64")?;

        let mut labels_i64 = Vec::with_capacity(ids.len());
        for &lab in ids.iter() {
            let lab_i64 = i64::try_from(lab).map_err(|_| {
                PyValueError::new_err(format!(
                    "label_id overflow converting u64 -> i64 (label_id={lab})"
                ))
            })?;
            labels_i64.push(lab_i64);
        }
        let labels_list = PyList::new_bound(py, labels_i64);
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("dtype", &torch_int64)?;
        let labels_i64 = torch.call_method("tensor", (labels_list,), Some(&kwargs))?;

        Ok((payload_u8, offsets_i64, sample_ids_i64, labels_i64))
    }
}
