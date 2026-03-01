use super::*;

pub(crate) struct LeaseProgress {
    start_id: u64,
    end_id: u64,
    cursor: AtomicU64,
    delivered_samples: AtomicU64,
    delivered_bytes: AtomicU64,
}

impl LeaseProgress {
    fn new(start_id: u64, end_id: u64) -> Self {
        Self {
            start_id,
            end_id,
            cursor: AtomicU64::new(start_id),
            delivered_samples: AtomicU64::new(0),
            delivered_bytes: AtomicU64::new(0),
        }
    }

    fn cursor(&self) -> u64 {
        self.cursor.load(Ordering::Acquire)
    }

    fn delivered_samples(&self) -> u64 {
        self.delivered_samples.load(Ordering::Relaxed)
    }

    fn delivered_bytes(&self) -> u64 {
        self.delivered_bytes.load(Ordering::Relaxed)
    }

    fn on_deliver(&self, batch: &mx8_runtime::types::Batch) {
        self.delivered_samples
            .fetch_add(batch.sample_count() as u64, Ordering::Relaxed);
        self.delivered_bytes
            .fetch_add(batch.payload_len() as u64, Ordering::Relaxed);

        let max_id = batch
            .sample_ids
            .iter()
            .copied()
            .max()
            .unwrap_or(self.start_id);
        let next_cursor = max_id.saturating_add(1).clamp(self.start_id, self.end_id);

        let mut cur = self.cursor.load(Ordering::Relaxed);
        while cur < next_cursor {
            match self.cursor.compare_exchange_weak(
                cur,
                next_cursor,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(actual) => cur = actual,
            }
        }
    }
}

pub(crate) async fn fetch_manifest_bytes(
    channel: Channel,
    grpc_max_message_bytes: usize,
    job_id: &str,
    manifest_hash: &str,
) -> Result<Vec<u8>> {
    let mut client = CoordinatorClient::new(channel)
        .max_decoding_message_size(grpc_max_message_bytes)
        .max_encoding_message_size(grpc_max_message_bytes);

    let req = GetManifestRequest {
        job_id: job_id.to_string(),
        manifest_hash: manifest_hash.to_string(),
    };

    let resp = client.get_manifest_stream(req.clone()).await;
    match resp {
        Ok(resp) => {
            let mut stream = resp.into_inner();
            let mut out: Vec<u8> = Vec::new();
            let mut schema_version: Option<u32> = None;
            while let Some(chunk) = stream.message().await? {
                if let Some(existing) = schema_version {
                    if chunk.schema_version != existing {
                        anyhow::bail!("GetManifestStream returned inconsistent schema_version");
                    }
                } else {
                    schema_version = Some(chunk.schema_version);
                }
                out.extend_from_slice(chunk.data.as_slice());
            }
            if schema_version.is_none() {
                anyhow::bail!("GetManifestStream returned no chunks (empty manifest)");
            }
            Ok(out)
        }
        Err(status) => {
            if status.code() != tonic::Code::Unimplemented {
                return Err(anyhow::anyhow!("GetManifestStream failed: {status}"));
            }
            let resp = client.get_manifest(req).await?.into_inner();
            Ok(resp.manifest_bytes)
        }
    }
}

pub(crate) async fn fetch_resume_checkpoint_bytes(
    channel: Channel,
    grpc_max_message_bytes: usize,
    job_id: &str,
) -> Result<Vec<u8>> {
    let mut client = CoordinatorClient::new(channel)
        .max_decoding_message_size(grpc_max_message_bytes)
        .max_encoding_message_size(grpc_max_message_bytes);
    let resp = client
        .get_resume_checkpoint(GetResumeCheckpointRequest {
            job_id: job_id.to_string(),
        })
        .await?;
    Ok(resp.into_inner().checkpoint)
}

pub(crate) async fn heartbeat_loop(
    channel: Channel,
    grpc_max_message_bytes: usize,
    interval: Duration,
    job_id: String,
    node_id: String,
    pipeline: Arc<Pipeline>,
    autotune: Arc<AutotuneShared>,
) {
    let mut client = CoordinatorClient::new(channel)
        .max_decoding_message_size(grpc_max_message_bytes)
        .max_encoding_message_size(grpc_max_message_bytes);

    loop {
        tokio::time::sleep(interval).await;
        let now_ms = unix_time_ms();

        let metrics = pipeline.metrics();
        let stats = NodeStats {
            inflight_bytes: metrics.inflight_bytes.get(),
            ram_high_water_bytes: metrics.ram_high_water_bytes.get(),
            fetch_queue_depth: 0,
            decode_queue_depth: 0,
            pack_queue_depth: 0,
            autotune_enabled: autotune.enabled,
            effective_prefetch_batches: clamp_u64_to_u32(
                pipeline.effective_prefetch_batches() as u64
            ),
            effective_max_queue_batches: clamp_u64_to_u32(
                pipeline.effective_max_queue_batches() as u64
            ),
            effective_want: autotune.want.load(Ordering::Relaxed),
            autotune_pressure_milli: clamp_u64_to_u32(
                autotune.pressure_milli.load(Ordering::Relaxed),
            ),
            autotune_cooldown_ticks: autotune.cooldown_ticks.load(Ordering::Relaxed),
            batch_payload_p95_over_p50_milli: clamp_u64_to_u32(
                metrics.batch_payload_bytes_p95_over_p50_milli.get(),
            ),
            batch_jitter_slo_breaches_total: metrics.batch_jitter_slo_breaches_total.get(),
        };

        let _ = client
            .heartbeat(HeartbeatRequest {
                job_id: job_id.clone(),
                node_id: node_id.clone(),
                unix_time_ms: now_ms,
                stats: Some(stats),
            })
            .await;
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn progress_loop(
    channel: Channel,
    grpc_max_message_bytes: usize,
    interval: Duration,
    job_id: String,
    node_id: String,
    lease_id: String,
    progress: Arc<LeaseProgress>,
    mut done: tokio::sync::oneshot::Receiver<()>,
) {
    let mut client = CoordinatorClient::new(channel)
        .max_decoding_message_size(grpc_max_message_bytes)
        .max_encoding_message_size(grpc_max_message_bytes);

    let mut ticker = tokio::time::interval(std::cmp::max(Duration::from_millis(1), interval));
    loop {
        tokio::select! {
            _ = ticker.tick() => {
                let _ = client
                    .report_progress(ReportProgressRequest {
                        job_id: job_id.clone(),
                        node_id: node_id.clone(),
                        lease_id: lease_id.clone(),
                        cursor: progress.cursor(),
                        delivered_samples: progress.delivered_samples(),
                        delivered_bytes: progress.delivered_bytes(),
                        unix_time_ms: unix_time_ms(),
                    })
                    .await;
            }
            _ = &mut done => {
                return;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_lease_and_stream_batches(
    channel: Channel,
    grpc_max_message_bytes: usize,
    pipeline: Arc<Pipeline>,
    manifest_bytes: Arc<Vec<u8>>,
    job_id: String,
    node_id: String,
    lease: mx8_core::types::Lease,
    tx: tokio::sync::mpsc::Sender<BatchLease>,
    progress_interval_ms: u64,
) -> Result<()> {
    let range = lease.range;

    let progress = Arc::new(LeaseProgress::new(range.start_id, range.end_id));
    let (done_tx, done_rx) = tokio::sync::oneshot::channel();
    let reporter = tokio::spawn(progress_loop(
        channel.clone(),
        grpc_max_message_bytes,
        Duration::from_millis(progress_interval_ms.max(1)),
        job_id.clone(),
        node_id.clone(),
        lease.lease_id.0.clone(),
        progress.clone(),
        done_rx,
    ));

    let (mut rx, task) = pipeline
        .spawn_manifest_bytes_range_stream((*manifest_bytes).clone(), range.start_id, range.end_id)
        .await?;

    while let Some(batch_lease) = rx.recv().await {
        progress.on_deliver(&batch_lease.batch);
        if tx.send(batch_lease).await.is_err() {
            break;
        }
    }

    task.await??;
    let _ = done_tx.send(());
    let _ = reporter.await;

    // Final progress report.
    let mut client = CoordinatorClient::new(channel)
        .max_decoding_message_size(grpc_max_message_bytes)
        .max_encoding_message_size(grpc_max_message_bytes);
    let _ = client
        .report_progress(ReportProgressRequest {
            job_id,
            node_id,
            lease_id: lease.lease_id.0,
            cursor: progress.cursor(),
            delivered_samples: progress.delivered_samples(),
            delivered_bytes: progress.delivered_bytes(),
            unix_time_ms: unix_time_ms(),
        })
        .await;

    Ok(())
}

#[pyclass]
pub(crate) struct DistributedDataLoader {
    pub(crate) coord_url: String,
    pub(crate) job_id: String,
    pub(crate) grpc_max_message_bytes: usize,
    pub(crate) manifest_hash: String,
    pub(crate) assigned_rank: u32,
    pub(crate) world_size: u32,
    pub(crate) elastic_state: Arc<ElasticRuntimeState>,
    pub(crate) max_process_rss_bytes: Option<u64>,
    pub(crate) metrics: Arc<RuntimeMetrics>,
    pub(crate) pipeline: Arc<Pipeline>,
    pub(crate) autotune: Arc<AutotuneShared>,
    pub(crate) rx: tokio::sync::mpsc::Receiver<BatchLease>,
    pub(crate) task: Option<tokio::task::JoinHandle<Result<()>>>,
    pub(crate) heartbeat_task: Option<tokio::task::JoinHandle<()>>,
    pub(crate) autotune_task: Option<tokio::task::JoinHandle<()>>,
    pub(crate) rt: tokio::runtime::Runtime,
    pub(crate) started_at: Instant,
}

#[derive(Debug)]
pub(crate) struct ElasticRuntimeState {
    pending: AtomicU32,
    transitions_total: AtomicU64,
    current_world_size: AtomicU32,
    target_world_size: AtomicU32,
    last_transition_unix_time_ms: AtomicU64,
    last_reason: Mutex<String>,
}

impl Default for ElasticRuntimeState {
    fn default() -> Self {
        Self {
            pending: AtomicU32::new(0),
            transitions_total: AtomicU64::new(0),
            current_world_size: AtomicU32::new(0),
            target_world_size: AtomicU32::new(0),
            last_transition_unix_time_ms: AtomicU64::new(0),
            last_reason: Mutex::new("none".to_string()),
        }
    }
}

impl ElasticRuntimeState {
    fn apply(
        &self,
        pending: bool,
        transitions_total: u64,
        last_reason: &str,
        current_world_size: u32,
        target_world_size: u32,
        last_transition_unix_time_ms: u64,
    ) {
        self.pending.store(u32::from(pending), Ordering::Relaxed);
        self.transitions_total
            .store(transitions_total, Ordering::Relaxed);
        self.current_world_size
            .store(current_world_size, Ordering::Relaxed);
        self.target_world_size
            .store(target_world_size, Ordering::Relaxed);
        self.last_transition_unix_time_ms
            .store(last_transition_unix_time_ms, Ordering::Relaxed);
        let mut guard = match self.last_reason.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        let reason = if last_reason.trim().is_empty() {
            "none"
        } else {
            last_reason
        };
        guard.clear();
        guard.push_str(reason);
    }

    fn apply_from_register(&self, resp: &RegisterNodeResponse) {
        self.apply(
            resp.elastic_transition_pending,
            resp.elastic_transitions_total,
            resp.elastic_last_transition_reason.as_str(),
            resp.elastic_current_world_size,
            resp.elastic_target_world_size,
            resp.elastic_last_transition_unix_time_ms,
        );
    }

    fn apply_from_request_lease(&self, resp: &RequestLeaseResponse) {
        self.apply(
            resp.elastic_transition_pending,
            resp.elastic_transitions_total,
            resp.elastic_last_transition_reason.as_str(),
            resp.elastic_current_world_size,
            resp.elastic_target_world_size,
            resp.elastic_last_transition_unix_time_ms,
        );
    }

    fn last_reason(&self) -> String {
        let guard = match self.last_reason.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.clone()
    }
}

#[pymethods]
impl DistributedDataLoader {
    #[new]
    #[pyo3(signature = (
        *,
        coord_url,
        job_id,
        node_id,
        batch_size_samples=512,
        max_inflight_bytes=128*1024*1024,
        max_queue_batches=64,
        prefetch_batches=1,
        target_batch_bytes=None,
        max_batch_bytes=None,
        want=1,
        max_ram_gb=None,
        profile=None,
        autotune=None,
        constraints=None,
        runtime=None,
        progress_interval_ms=500,
        grpc_max_message_bytes=DEFAULT_GRPC_MAX_MESSAGE_BYTES,
        resume_from=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        py: Python<'_>,
        coord_url: String,
        job_id: String,
        node_id: String,
        batch_size_samples: usize,
        max_inflight_bytes: u64,
        max_queue_batches: usize,
        prefetch_batches: usize,
        target_batch_bytes: Option<u64>,
        max_batch_bytes: Option<u64>,
        want: u32,
        max_ram_gb: Option<f64>,
        profile: Option<String>,
        autotune: Option<bool>,
        constraints: Option<Py<Constraints>>,
        runtime: Option<Py<RuntimeConfig>>,
        progress_interval_ms: u64,
        grpc_max_message_bytes: usize,
        resume_from: Option<Vec<u8>>,
    ) -> PyResult<Self> {
        let _ = (max_inflight_bytes, max_queue_batches, prefetch_batches);
        let constraints_cfg = constraints.as_ref().map(|c| c.bind(py).borrow().clone());
        let runtime_cfg = runtime.as_ref().map(|r| r.bind(py).borrow().clone());
        let max_ram_bytes_from_gb = max_ram_gb_to_bytes(max_ram_gb)?;
        let selected_profile = AutotuneProfile::from_name(profile.as_deref());
        let defaults = ProfileDefaults::for_profile(selected_profile);
        let mut effective_max_inflight_bytes = defaults.max_inflight_bytes;
        let mut effective_max_queue_batches = defaults.max_queue_batches;
        let mut effective_prefetch_batches = defaults.prefetch_batches;
        let mut effective_want = want;
        let mut effective_max_process_rss_bytes = env_u64("MX8_MAX_PROCESS_RSS_BYTES");
        let autotune_enabled = autotune.unwrap_or(true);
        if let Some(runtime_cfg) = &runtime_cfg {
            if let Some(prefetch) = runtime_cfg.prefetch_batches {
                effective_prefetch_batches = prefetch.max(1);
            }
            if let Some(max_queue) = runtime_cfg.max_queue_batches {
                effective_max_queue_batches = max_queue.max(1);
            }
            if let Some(want) = runtime_cfg.want {
                effective_want = want.max(1);
            }
        }
        if let Some(constraints_cfg) = &constraints_cfg {
            if let Some(max_inflight) = constraints_cfg.max_inflight_bytes {
                effective_max_inflight_bytes = max_inflight.max(1);
            }
            if let Some(max_process) = constraints_cfg.max_process_rss_bytes {
                effective_max_process_rss_bytes = Some(max_process.max(1));
            }
        }
        if let Some(max_ram_bytes) = max_ram_bytes_from_gb {
            if constraints_cfg
                .as_ref()
                .and_then(|c| c.max_process_rss_bytes)
                .is_some()
            {
                return Err(PyValueError::new_err(
                    "pass only one of max_ram_gb or constraints.max_ram_bytes",
                ));
            }
            effective_max_process_rss_bytes = Some(max_ram_bytes.max(1));
        }
        if effective_max_process_rss_bytes.is_none() {
            effective_max_process_rss_bytes = derive_default_max_process_rss_bytes(
                selected_profile,
                effective_max_inflight_bytes,
            );
            if let Some(cap) = effective_max_process_rss_bytes {
                tracing::info!(
                    target: "mx8_proof",
                    event = "rss_cap_defaulted",
                    mode = "distributed",
                    profile = profile_name(selected_profile),
                    max_inflight_bytes = effective_max_inflight_bytes,
                    max_process_rss_bytes = cap,
                    "defaulted process RSS cap from profile and node memory limit"
                );
            }
        }
        if let Some(max_process) = effective_max_process_rss_bytes {
            if effective_max_inflight_bytes > max_process {
                effective_max_inflight_bytes = max_process;
            }
        }

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("failed to start tokio runtime: {e}")))?;
        let elastic_state = Arc::new(ElasticRuntimeState::default());
        let elastic_state_for_register = elastic_state.clone();
        let coord_url_for_register = coord_url.clone();
        let job_id_for_register = job_id.clone();
        let node_id_for_register = node_id.clone();
        let resume_from_for_register = resume_from.clone();

        let (manifest_hash, assigned_rank, world_size, heartbeat_interval_ms, channel) = rt
            .block_on(async move {
                let channel = Channel::from_shared(coord_url_for_register.clone())?
                    .connect()
                    .await?;
                let mut client = CoordinatorClient::new(channel.clone())
                    .max_decoding_message_size(grpc_max_message_bytes)
                    .max_encoding_message_size(grpc_max_message_bytes);

                let caps = Some(NodeCaps {
                    max_fetch_concurrency: 32,
                    max_decode_concurrency: 8,
                    max_inflight_bytes: effective_max_inflight_bytes,
                    max_ram_bytes: effective_max_process_rss_bytes
                        .unwrap_or(effective_max_inflight_bytes),
                });

                let mut resp = client
                    .register_node(RegisterNodeRequest {
                        job_id: job_id_for_register.clone(),
                        node_id: node_id_for_register.clone(),
                        caps: caps.clone(),
                        resume_from: resume_from_for_register.clone().unwrap_or_default(),
                    })
                    .await?
                    .into_inner();
                elastic_state_for_register.apply_from_register(&resp);

                while !resp.job_ready {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    resp = client
                        .register_node(RegisterNodeRequest {
                            job_id: job_id_for_register.clone(),
                            node_id: node_id_for_register.clone(),
                            caps: caps.clone(),
                            resume_from: resume_from_for_register.clone().unwrap_or_default(),
                        })
                        .await?
                        .into_inner();
                    elastic_state_for_register.apply_from_register(&resp);
                }

                Ok::<_, anyhow::Error>((
                    resp.manifest_hash,
                    resp.assigned_rank,
                    resp.world_size,
                    resp.heartbeat_interval_ms,
                    channel,
                ))
            })
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;

        let manifest_bytes = rt
            .block_on(fetch_manifest_bytes(
                channel.clone(),
                grpc_max_message_bytes,
                &job_id,
                &manifest_hash,
            ))
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;

        let max_process_rss_bytes_cap = effective_max_process_rss_bytes;
        let (effective_target_batch_bytes, effective_max_batch_bytes) = derive_byte_batch_caps(
            effective_max_inflight_bytes,
            target_batch_bytes,
            max_batch_bytes,
        );
        let caps = RuntimeCaps {
            max_inflight_bytes: effective_max_inflight_bytes,
            max_queue_batches: effective_max_queue_batches,
            batch_size_samples,
            prefetch_batches: effective_prefetch_batches,
            target_batch_bytes: effective_target_batch_bytes,
            max_batch_bytes: effective_max_batch_bytes,
            max_process_rss_bytes: max_process_rss_bytes_cap,
        };
        let pipeline = Arc::new(Pipeline::new(caps));
        let metrics = pipeline.metrics();
        let autotune_profile = selected_profile;
        let rails = AutotuneRails::for_profile(autotune_profile);
        let initial_want = std::cmp::max(1, effective_want).clamp(rails.min_want, rails.max_want);
        let initial_prefetch = effective_prefetch_batches
            .max(rails.min_prefetch_batches)
            .min(rails.max_prefetch_batches);
        let initial_max_queue = effective_max_queue_batches
            .max(rails.min_max_queue_batches)
            .min(rails.max_max_queue_batches);
        pipeline.set_prefetch_batches(initial_prefetch);
        pipeline.set_max_queue_batches(initial_max_queue);
        let autotune = Arc::new(AutotuneShared::new(
            autotune_enabled,
            initial_want,
            initial_prefetch,
            initial_max_queue,
        ));

        if autotune_enabled {
            tracing::info!(
                target: "mx8_proof",
                event = "autotune_startup_caps_selected",
                profile = match autotune_profile {
                    AutotuneProfile::Safe => "safe",
                    AutotuneProfile::Balanced => "balanced",
                    AutotuneProfile::Throughput => "throughput",
                },
                prefetch_batches = initial_prefetch as u64,
                max_queue_batches = initial_max_queue as u64,
                want = initial_want as u64,
                max_inflight_bytes = effective_max_inflight_bytes,
                max_process_rss_bytes = max_process_rss_bytes_cap.unwrap_or(0),
                "autotune initialized"
            );
        }

        let heartbeat_task = {
            let interval = Duration::from_millis(std::cmp::max(1, heartbeat_interval_ms) as u64);
            let pipeline = pipeline.clone();
            let autotune = autotune.clone();
            let job_id = job_id.clone();
            let node_id = node_id.clone();
            let channel = channel.clone();
            Some(rt.spawn(heartbeat_loop(
                channel,
                grpc_max_message_bytes,
                interval,
                job_id,
                node_id,
                pipeline,
                autotune,
            )))
        };

        let autotune_task = if autotune_enabled {
            let pipeline = pipeline.clone();
            let metrics = metrics.clone();
            let shared = autotune.clone();
            Some(rt.spawn(autotune_loop(
                pipeline,
                metrics,
                shared,
                effective_max_inflight_bytes,
                max_process_rss_bytes_cap,
                rails,
            )))
        } else {
            None
        };

        let (tx, rx) = tokio::sync::mpsc::channel::<BatchLease>(initial_max_queue);
        let manifest_bytes = Arc::new(manifest_bytes);
        let autotune_for_requests = autotune.clone();
        let pipeline_for_requests = pipeline.clone();
        let elastic_for_requests = elastic_state.clone();
        let job_id_for_task = job_id.clone();
        let node_id_for_task = node_id.clone();
        let task = rt.spawn(async move {
            let mut next_request_at = tokio::time::Instant::now();
            let mut last_checkpointed_transition_total = 0u64;
            loop {
                let now = tokio::time::Instant::now();
                if now < next_request_at {
                    tokio::time::sleep_until(next_request_at).await;
                }
                let mut want = if autotune_for_requests.enabled {
                    autotune_for_requests.want.load(Ordering::Relaxed).max(1)
                } else {
                    std::cmp::max(1, effective_want)
                };
                if elastic_for_requests.pending.load(Ordering::Relaxed) != 0 {
                    // During elastic transitions, narrow request width to reduce
                    // boundary churn while membership converges.
                    want = 1;
                }

                let mut client = CoordinatorClient::new(channel.clone())
                    .max_decoding_message_size(grpc_max_message_bytes)
                    .max_encoding_message_size(grpc_max_message_bytes);
                let resp = client
                    .request_lease(RequestLeaseRequest {
                        job_id: job_id_for_task.clone(),
                        node_id: node_id_for_task.clone(),
                        want,
                    })
                    .await;

                let resp = match resp {
                    Ok(resp) => resp.into_inner(),
                    Err(err) => {
                        next_request_at = tokio::time::Instant::now() + Duration::from_millis(500);
                        let _ = err;
                        continue;
                    }
                };
                elastic_for_requests.apply_from_request_lease(&resp);
                if resp.elastic_transition_pending
                    && resp.elastic_transitions_total > last_checkpointed_transition_total
                {
                    match fetch_resume_checkpoint_bytes(
                        channel.clone(),
                        grpc_max_message_bytes,
                        &job_id_for_task,
                    )
                    .await
                    {
                        Ok(bytes) => {
                            tracing::info!(
                                target: "mx8_proof",
                                event = "elastic_transition_checkpoint_captured",
                                transitions_total = resp.elastic_transitions_total,
                                reason = %resp.elastic_last_transition_reason,
                                current_world_size = resp.elastic_current_world_size,
                                target_world_size = resp.elastic_target_world_size,
                                checkpoint_bytes = bytes.len() as u64,
                                "captured distributed resume checkpoint at elastic boundary"
                            );
                        }
                        Err(err) => {
                            tracing::warn!(
                                target: "mx8_proof",
                                event = "elastic_transition_checkpoint_capture_failed",
                                transitions_total = resp.elastic_transitions_total,
                                reason = %resp.elastic_last_transition_reason,
                                error = %err,
                                "failed to capture distributed resume checkpoint at elastic boundary"
                            );
                        }
                    }
                    last_checkpointed_transition_total = resp.elastic_transitions_total;
                }

                if resp.leases.is_empty() {
                    if resp.job_drained {
                        tracing::info!(
                            target: "mx8_proof",
                            event = "distributed_loader_job_drained",
                            job_id = %job_id_for_task,
                            node_id = %node_id_for_task,
                            "coordinator reported drained; stopping distributed loader request loop"
                        );
                        break;
                    }
                    let wait_ms = std::cmp::max(1, resp.wait_ms);
                    next_request_at =
                        tokio::time::Instant::now() + Duration::from_millis(wait_ms as u64);
                    continue;
                }

                for lease in resp.leases {
                    let core_lease = lease.try_to_core().map_err(anyhow::Error::from)?;
                    run_lease_and_stream_batches(
                        channel.clone(),
                        grpc_max_message_bytes,
                        pipeline_for_requests.clone(),
                        manifest_bytes.clone(),
                        job_id_for_task.clone(),
                        node_id_for_task.clone(),
                        core_lease,
                        tx.clone(),
                        progress_interval_ms,
                    )
                    .await?;
                }
            }
            Ok::<(), anyhow::Error>(())
        });

        Ok(Self {
            coord_url,
            job_id,
            grpc_max_message_bytes,
            manifest_hash,
            assigned_rank,
            world_size,
            elastic_state,
            max_process_rss_bytes: max_process_rss_bytes_cap,
            metrics,
            pipeline,
            autotune,
            rx,
            task: Some(task),
            heartbeat_task,
            autotune_task,
            rt,
            started_at: Instant::now(),
        })
    }

    #[getter]
    pub(crate) fn manifest_hash(&self) -> &str {
        &self.manifest_hash
    }

    #[getter]
    fn assigned_rank(&self) -> u32 {
        self.assigned_rank
    }

    #[getter]
    fn world_size(&self) -> u32 {
        self.world_size
    }

    pub(crate) fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let out = metrics_to_dict(py, self.metrics.as_ref())?;
        out.set_item(
            "effective_prefetch_batches",
            self.pipeline.effective_prefetch_batches(),
        )?;
        out.set_item(
            "effective_max_queue_batches",
            self.pipeline.effective_max_queue_batches(),
        )?;
        out.set_item("effective_want", self.autotune.want.load(Ordering::Relaxed))?;
        out.set_item("assigned_rank", self.assigned_rank)?;
        out.set_item("world_size", self.world_size)?;
        out.set_item(
            "elastic_transition_pending",
            self.elastic_state.pending.load(Ordering::Relaxed) != 0,
        )?;
        out.set_item(
            "elastic_transitions_total",
            self.elastic_state.transitions_total.load(Ordering::Relaxed),
        )?;
        out.set_item(
            "elastic_last_transition_reason",
            self.elastic_state.last_reason(),
        )?;
        out.set_item(
            "elastic_current_world_size",
            self.elastic_state
                .current_world_size
                .load(Ordering::Relaxed),
        )?;
        out.set_item(
            "elastic_target_world_size",
            self.elastic_state.target_world_size.load(Ordering::Relaxed),
        )?;
        out.set_item(
            "elastic_last_transition_unix_time_ms",
            self.elastic_state
                .last_transition_unix_time_ms
                .load(Ordering::Relaxed),
        )?;
        out.set_item("max_process_rss_bytes", self.max_process_rss_bytes)?;
        out.set_item("elapsed_seconds", self.started_at.elapsed().as_secs_f64())?;
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
        Ok(out.into_any())
    }

    pub(crate) fn checkpoint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py.allow_threads(|| {
            self.rt.block_on(async {
                let channel = Channel::from_shared(self.coord_url.clone())?
                    .connect()
                    .await?;
                let mut client = CoordinatorClient::new(channel)
                    .max_decoding_message_size(self.grpc_max_message_bytes)
                    .max_encoding_message_size(self.grpc_max_message_bytes);
                let resp = client
                    .get_resume_checkpoint(GetResumeCheckpointRequest {
                        job_id: self.job_id.clone(),
                    })
                    .await?;
                Ok::<Vec<u8>, anyhow::Error>(resp.into_inner().checkpoint)
            })
        });
        match bytes {
            Ok(bytes) => Ok(PyBytes::new_bound(py, &bytes)),
            Err(err) => Err(PyRuntimeError::new_err(format!("{err}"))),
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    pub(crate) fn __next__(&mut self) -> PyResult<PyBatch> {
        let wait_started = Instant::now();
        let out = self.rt.block_on(async {
            match self.rx.recv().await {
                Some(lease) => Ok(PyBatch { lease }),
                None => match self.task.take() {
                    Some(handle) => match handle.await {
                        Ok(Ok(())) => Err(PyStopIteration::new_err(())),
                        Ok(Err(err)) => Err(PyRuntimeError::new_err(format!("{err}"))),
                        Err(err) => Err(PyRuntimeError::new_err(format!(
                            "producer task failed: {err}"
                        ))),
                    },
                    None => Err(PyStopIteration::new_err(())),
                },
            }
        });
        self.autotune.on_wait(wait_started.elapsed());
        out
    }

    pub(crate) fn close(&mut self) {
        if let Some(handle) = self.task.take() {
            handle.abort();
        }
        if let Some(handle) = self.heartbeat_task.take() {
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
