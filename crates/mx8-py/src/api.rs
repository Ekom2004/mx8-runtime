use super::*;

#[pyfunction]
#[pyo3(signature = (dataset_link, *, manifest_store=None, manifest_path=None, recursive=true, node_id=None))]
pub(crate) fn resolve_manifest_hash(
    dataset_link: String,
    manifest_store: Option<PathBuf>,
    manifest_path: Option<PathBuf>,
    recursive: bool,
    node_id: Option<String>,
) -> PyResult<String> {
    let root = manifest_store
        .or(env_path("MX8_MANIFEST_STORE_ROOT"))
        .unwrap_or_else(default_manifest_store);
    let dev_manifest_path = manifest_path.or(env_path("MX8_DEV_MANIFEST_PATH"));

    let store = std::sync::Arc::new(FsManifestStore::new(root));
    let resolver = SnapshotResolver::new(
        store,
        SnapshotResolverConfig {
            dev_manifest_path,
            recursive,
            materialize_manifest_bytes: false,
            ..SnapshotResolverConfig::default()
        },
    );

    let snapshot = resolver
        .resolve(
            &dataset_link,
            LockOwner {
                node_id: node_id.clone(),
            },
        )
        .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;

    Ok(snapshot.manifest_hash.0)
}

#[pyfunction]
#[pyo3(signature = (loader, *, raw=false))]
pub(crate) fn stats<'py>(
    py: Python<'py>,
    loader: &Bound<'py, PyAny>,
    raw: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let raw_stats = loader.call_method0("stats").map_err(|_| {
        PyValueError::new_err("mx8.stats expects a loader object with a .stats() method")
    })?;

    if raw {
        return Ok(raw_stats);
    }

    let stats_dict = raw_stats
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("mx8.stats expected loader.stats() to return a dict"))?;

    let rendered = render_human_stats(stats_dict);
    Ok(PyString::new_bound(py, &rendered).into_any())
}

#[pyfunction(name = "video_index_build")]
#[pyo3(signature = (
    dataset_link,
    *,
    manifest_store=None,
    manifest_path=None,
    recursive=true,
    clip_len=16,
    stride=8,
    fps_policy="fixed_fps:8".to_string(),
    seed=0,
    epoch=0,
    max_clips_in_memory=1_000_000,
    node_id=None
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn internal_video_index_build<'py>(
    py: Python<'py>,
    dataset_link: String,
    manifest_store: Option<PathBuf>,
    manifest_path: Option<PathBuf>,
    recursive: bool,
    clip_len: u32,
    stride: u32,
    fps_policy: String,
    seed: u64,
    epoch: u64,
    max_clips_in_memory: usize,
    node_id: Option<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let root = manifest_store
        .or(env_path("MX8_MANIFEST_STORE_ROOT"))
        .unwrap_or_else(default_manifest_store);
    let dev_manifest_path = manifest_path.or(env_path("MX8_DEV_MANIFEST_PATH"));

    let store = std::sync::Arc::new(FsManifestStore::new(root));
    let resolver = SnapshotResolver::new(
        store,
        SnapshotResolverConfig {
            dev_manifest_path,
            recursive,
            materialize_manifest_bytes: true,
            ..SnapshotResolverConfig::default()
        },
    );
    let snapshot = resolver
        .resolve(
            &dataset_link,
            LockOwner {
                node_id: node_id.clone(),
            },
        )
        .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;

    let cfg = VideoStage1Config {
        clip_len,
        stride,
        fps_policy,
        seed,
        epoch,
        max_clips_in_memory,
    };
    let index = build_video_stage1_index_from_manifest_bytes(
        &snapshot.manifest_hash,
        &snapshot.manifest_bytes,
        &cfg,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
    let index_tsv = canonicalize_video_stage1_tsv(&index.clips);
    let clip_index_hash = mx8_manifest_store::sha256_hex(&index_tsv);

    let out = PyDict::new_bound(py);
    out.set_item("manifest_hash", snapshot.manifest_hash.0)?;
    out.set_item("video_schema_version", 1u64)?;
    out.set_item("clip_count", index.summary.clip_count)?;
    out.set_item(
        "tail_clips_dropped_total",
        index.summary.tail_clips_dropped_total,
    )?;
    out.set_item("clip_index_hash", clip_index_hash)?;
    out.set_item(
        "clip_ids_head",
        PyList::new_bound(
            py,
            index
                .clips
                .iter()
                .take(16)
                .map(|c| c.clip_id.clone())
                .collect::<Vec<_>>(),
        ),
    )?;
    let failure = PyDict::new_bound(py);
    failure.set_item("corrupt_media", index.summary.failure_counts.corrupt_media)?;
    failure.set_item("short_media", index.summary.failure_counts.short_media)?;
    failure.set_item(
        "unsupported_codec",
        index.summary.failure_counts.unsupported_codec,
    )?;
    failure.set_item(
        "missing_stream",
        index.summary.failure_counts.missing_stream,
    )?;
    out.set_item("failure_counts", failure)?;
    Ok(out.into_any())
}

#[pyfunction(name = "video_index_replay_check")]
#[pyo3(signature = (
    dataset_link,
    *,
    manifest_store=None,
    manifest_path=None,
    recursive=true,
    clip_len=16,
    stride=8,
    fps_policy="fixed_fps:8".to_string(),
    seed=0,
    epoch=0,
    max_clips_in_memory=1_000_000,
    node_id=None
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn internal_video_index_replay_check<'py>(
    py: Python<'py>,
    dataset_link: String,
    manifest_store: Option<PathBuf>,
    manifest_path: Option<PathBuf>,
    recursive: bool,
    clip_len: u32,
    stride: u32,
    fps_policy: String,
    seed: u64,
    epoch: u64,
    max_clips_in_memory: usize,
    node_id: Option<String>,
) -> PyResult<Bound<'py, PyAny>> {
    let first = internal_video_index_build(
        py,
        dataset_link.clone(),
        manifest_store.clone(),
        manifest_path.clone(),
        recursive,
        clip_len,
        stride,
        fps_policy.clone(),
        seed,
        epoch,
        max_clips_in_memory,
        node_id.clone(),
    )?;
    let second = internal_video_index_build(
        py,
        dataset_link,
        manifest_store,
        manifest_path,
        recursive,
        clip_len,
        stride,
        fps_policy,
        seed,
        epoch,
        max_clips_in_memory,
        node_id,
    )?;
    let first_d = first.downcast_into::<PyDict>()?;
    let second_d = second.downcast_into::<PyDict>()?;
    let h1: String = first_d
        .get_item("clip_index_hash")?
        .ok_or_else(|| PyRuntimeError::new_err("missing clip_index_hash from first run"))?
        .extract()?;
    let h2: String = second_d
        .get_item("clip_index_hash")?
        .ok_or_else(|| PyRuntimeError::new_err("missing clip_index_hash from second run"))?
        .extract()?;
    if h1 != h2 {
        return Err(PyRuntimeError::new_err(format!(
            "video index replay mismatch: first={h1} second={h2}"
        )));
    }
    let out = PyDict::new_bound(py);
    out.set_item("deterministic", true)?;
    out.set_item("clip_index_hash", h1)?;
    Ok(out.into_any())
}

#[pyfunction]
#[pyo3(signature = (pack_in, *, out, shard_mb=512, label_mode=None, require_labels=false, parallel_fetches=128))]
pub(crate) fn pack<'py>(
    py: Python<'py>,
    pack_in: String,
    out: String,
    shard_mb: u64,
    label_mode: Option<String>,
    require_labels: bool,
    parallel_fetches: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let label_mode = parse_pack_label_mode(label_mode.as_deref().unwrap_or("auto"))
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("failed to start tokio runtime: {e}")))?;

    let res = py
        .allow_threads(|| {
            rt.block_on(pack_s3(PackS3Config {
                pack_in,
                pack_out: out,
                shard_mb,
                label_mode,
                require_labels,
                parallel_fetches,
            }))
        })
        .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;

    let d = PyDict::new_bound(py);
    d.set_item("samples", res.samples)?;
    d.set_item("shards", res.shards)?;
    d.set_item("manifest_key", res.manifest_key)?;
    d.set_item("manifest_hash", res.manifest_hash)?;
    d.set_item("labels_key", res.labels_key)?;
    d.set_item("labels_hash", res.labels_hash)?;
    Ok(d.into_any())
}

#[pyfunction(name = "pack_dir")]
#[pyo3(signature = (in_dir, *, out, shard_mb=512, label_mode=None, require_labels=false))]
pub(crate) fn pack_dir<'py>(
    py: Python<'py>,
    in_dir: PathBuf,
    out: PathBuf,
    shard_mb: u64,
    label_mode: Option<String>,
    require_labels: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let label_mode = parse_pack_dir_label_mode(label_mode.as_deref().unwrap_or("auto"))
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;

    let res = py
        .allow_threads(|| {
            pack_dir_impl(PackDirConfig {
                in_dir,
                out_dir: out,
                shard_mb,
                label_mode,
                require_labels,
            })
        })
        .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;

    let d = PyDict::new_bound(py);
    d.set_item("samples", res.samples)?;
    d.set_item("shards", res.shards)?;
    d.set_item("manifest_path", res.manifest_path.display().to_string())?;
    d.set_item("manifest_hash", res.manifest_hash)?;
    d.set_item(
        "labels_path",
        res.labels_path.map(|p| p.display().to_string()),
    )?;
    d.set_item("labels_hash", res.labels_hash)?;
    Ok(d.into_any())
}

/// If `autopack=True` is set on a bare S3 prefix and no manifest exists yet,
/// pack the prefix in-place so subsequent loads use the fast precomputed path.
/// This is a one-time operation: once `_mx8/manifest.tsv` exists the pack step
/// is skipped on every subsequent call.
pub(crate) fn maybe_autopack(py: Python<'_>, s3_url: &str, shard_mb: u64) -> PyResult<()> {
    use mx8_snapshot::pack_s3::autopack_if_needed;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("autopack: failed to start runtime: {e}")))?;

    py.allow_threads(|| rt.block_on(autopack_if_needed(s3_url, shard_mb)))
        .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;

    Ok(())
}

#[pyfunction]
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
    start_id=None,
    end_id=None,
    resume_from=None,
    job_id=None,
    cluster_url=None,
    node_id=None,
    max_ram_gb=None,
    profile=None,
    autotune=None,
    constraints=None,
    runtime=None,
    autopack=false,
    autopack_shard_mb=512,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn load<'py>(
    py: Python<'py>,
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
    start_id: Option<u64>,
    end_id: Option<u64>,
    resume_from: Option<Vec<u8>>,
    job_id: Option<String>,
    cluster_url: Option<String>,
    node_id: Option<String>,
    max_ram_gb: Option<f64>,
    profile: Option<String>,
    autotune: Option<bool>,
    constraints: Option<Py<Constraints>>,
    runtime: Option<Py<RuntimeConfig>>,
    autopack: bool,
    autopack_shard_mb: u64,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = (max_inflight_bytes, max_queue_batches, prefetch_batches);
    let constraints_cfg = constraints.as_ref().map(|c| c.bind(py).borrow().clone());
    let runtime_cfg = runtime.as_ref().map(|r| r.bind(py).borrow().clone());
    let selected_profile = AutotuneProfile::from_name(profile.as_deref());
    let defaults = ProfileDefaults::for_profile(selected_profile);

    let mut effective_max_inflight_bytes = defaults.max_inflight_bytes;
    let mut effective_max_queue_batches = defaults.max_queue_batches;
    let mut effective_prefetch_batches = defaults.prefetch_batches;
    let mut effective_max_process_rss_bytes = env_u64("MX8_MAX_PROCESS_RSS_BYTES");
    let max_ram_bytes_from_gb = max_ram_gb_to_bytes(max_ram_gb)?;

    {
        let autotune_enabled = autotune.unwrap_or(true);

        if let Some(runtime_cfg) = &runtime_cfg {
            if let Some(prefetch) = runtime_cfg.prefetch_batches {
                effective_prefetch_batches = prefetch.max(1);
            }
            if let Some(max_queue) = runtime_cfg.max_queue_batches {
                effective_max_queue_batches = max_queue.max(1);
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
                    mode = "single_node",
                    profile = profile_name(selected_profile),
                    max_inflight_bytes = effective_max_inflight_bytes,
                    max_process_rss_bytes = cap,
                    "defaulted process RSS cap from profile and node memory limit"
                );
            }
        }

        if autotune_enabled {
            tracing::info!(
                target: "mx8_proof",
                event = "autotune_startup_caps_selected",
                mode = "single_node",
                profile = profile_name(selected_profile),
                max_inflight_bytes = effective_max_inflight_bytes,
                max_process_rss_bytes = effective_max_process_rss_bytes.unwrap_or(0),
                max_queue_batches = effective_max_queue_batches as u64,
                prefetch_batches = effective_prefetch_batches as u64,
                "v1 profile/autotune startup caps resolved"
            );
        }

        if let Some(max_process) = effective_max_process_rss_bytes {
            if effective_max_inflight_bytes > max_process {
                tracing::warn!(
                    target: "mx8_proof",
                    event = "autotune_cap_clamped",
                    requested_max_inflight_bytes = effective_max_inflight_bytes,
                    clamped_max_inflight_bytes = max_process,
                    max_process_rss_bytes = max_process,
                    "clamped max_inflight_bytes to max_process_rss_bytes"
                );
                effective_max_inflight_bytes = max_process;
            }
        }

        if !autotune_enabled && runtime_cfg.is_some() {
            tracing::info!(
                target: "mx8_proof",
                event = "autotune_disabled_manual_runtime",
                prefetch_batches = effective_prefetch_batches as u64,
                max_queue_batches = effective_max_queue_batches as u64,
                "manual runtime selected with autotune disabled"
            );
        }
    }

    if distributed_requested(cluster_url.as_deref()) {
        if start_id.is_some() || end_id.is_some() {
            return Err(PyValueError::new_err(
                "start_id/end_id are unsupported in distributed mode",
            ));
        }
        if autopack {
            return Err(PyValueError::new_err(
                "autopack is unsupported in distributed mode",
            ));
        }
        if manifest_store.is_some() || manifest_path.is_some() {
            return Err(PyValueError::new_err(
                "manifest_store/manifest_path are unsupported in distributed mode; coordinator owns snapshot resolution",
            ));
        }
        let coord_url = effective_cluster_url(cluster_url).ok_or_else(|| {
            PyValueError::new_err(
                "distributed mode requires cluster_url (or MX8_CLUSTER_URL / MX8_COORD_URL)",
            )
        })?;
        let job_id = job_id.or_else(|| env_string("MX8_JOB_ID")).ok_or_else(|| {
            PyValueError::new_err("distributed mode requires job_id (or MX8_JOB_ID)")
        })?;
        let rank = rank_from_env();
        let node_id = node_id.unwrap_or_else(|| format!("rank{rank}"));
        let out = DistributedDataLoader::new(
            py,
            coord_url,
            job_id,
            node_id,
            batch_size_samples,
            effective_max_inflight_bytes,
            effective_max_queue_batches,
            effective_prefetch_batches,
            target_batch_bytes,
            max_batch_bytes,
            1,
            max_ram_gb,
            profile,
            autotune,
            constraints,
            runtime,
            500,
            DEFAULT_GRPC_MAX_MESSAGE_BYTES,
            resume_from,
        )?;
        let out = Py::new(py, out)?;
        return Ok(out.into_bound(py).into_any());
    }

    if autopack && dataset_link.starts_with("s3://") && !dataset_link.contains('@') {
        maybe_autopack(py, &dataset_link, autopack_shard_mb)?;
    }

    let loader = DataLoader::new(
        dataset_link,
        manifest_store,
        manifest_path,
        recursive,
        batch_size_samples,
        effective_max_inflight_bytes,
        effective_max_queue_batches,
        effective_prefetch_batches,
        target_batch_bytes,
        max_batch_bytes,
        effective_max_process_rss_bytes,
        start_id,
        end_id,
        resume_from,
        node_id,
        profile.clone(),
        autotune,
    )?;
    let out = Py::new(py, loader)?;
    Ok(out.into_bound(py).into_any())
}

#[pyfunction]
#[pyo3(signature = (
    dataset,
    *,
    batch_size=512,
    memory_gb=8.0,
    profile="balanced".to_string(),
    job_id=None,
    resume_from=None,
    coordinator=None,
    node_id=None,
    recursive=true,
    manifest_store=None,
    manifest_path=None,
    autotune=None,
    constraints=None,
    runtime=None,
))]
#[allow(clippy::too_many_arguments)]
/// `mx8.run(...)` is the default job entry point.
///
/// - Single-node (`WORLD_SIZE` unset/1): delegates to `mx8.load(...)`.
/// - Distributed (`WORLD_SIZE > 1`): delegates to `mx8.DistributedDataLoader(...)`
///   and requires `job_id` + coordinator URL (`coordinator=` or `MX8_COORD_URL`).
pub(crate) fn run<'py>(
    py: Python<'py>,
    dataset: String,
    batch_size: usize,
    memory_gb: f64,
    profile: String,
    job_id: Option<String>,
    resume_from: Option<Vec<u8>>,
    coordinator: Option<String>,
    node_id: Option<String>,
    recursive: bool,
    manifest_store: Option<PathBuf>,
    manifest_path: Option<PathBuf>,
    autotune: Option<bool>,
    constraints: Option<Py<Constraints>>,
    runtime: Option<Py<RuntimeConfig>>,
) -> PyResult<Bound<'py, PyAny>> {
    let world_size = std::env::var("WORLD_SIZE")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .unwrap_or(1);

    if world_size > 1 {
        let coord_url = coordinator
            .or_else(|| std::env::var("MX8_COORD_URL").ok())
            .ok_or_else(|| {
                PyValueError::new_err(
                    "mx8.run distributed mode requires coordinator URL (pass coordinator=... or set MX8_COORD_URL)",
                )
            })?;
        let effective_job_id = job_id
            .or_else(|| std::env::var("MX8_JOB_ID").ok())
            .ok_or_else(|| {
                PyValueError::new_err(
                    "mx8.run distributed mode requires job_id (pass job_id=... or set MX8_JOB_ID)",
                )
            })?;
        let rank = std::env::var("RANK")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(0);
        let effective_node_id = node_id.unwrap_or_else(|| format!("rank{rank}"));
        let _ = (dataset, recursive, manifest_store, manifest_path);

        let loader = DistributedDataLoader::new(
            py,
            coord_url,
            effective_job_id,
            effective_node_id,
            batch_size,
            128 * 1024 * 1024,
            64,
            1,
            None,
            None,
            1,
            Some(memory_gb),
            Some(profile),
            autotune,
            constraints,
            runtime,
            500,
            DEFAULT_GRPC_MAX_MESSAGE_BYTES,
            resume_from,
        )?;
        let out = Py::new(py, loader)?;
        return Ok(out.into_bound(py).into_any());
    }

    let loader = load(
        py,
        dataset,
        manifest_store,
        manifest_path,
        recursive,
        batch_size,
        128 * 1024 * 1024,
        64,
        1,
        None,
        None,
        None,
        None,
        resume_from,
        job_id,
        coordinator,
        node_id,
        Some(memory_gb),
        Some(profile),
        autotune,
        constraints,
        runtime,
        false,
        512,
    )?;
    Ok(loader)
}

#[pyfunction]
#[pyo3(name = "text")]
#[pyo3(signature = (
    dataset_link,
    *,
    manifest_store=None,
    manifest_path=None,
    recursive=true,
    batch_size_samples=32,
    max_inflight_bytes=128*1024*1024,
    max_queue_batches=64,
    prefetch_batches=1,
    target_batch_bytes=None,
    max_batch_bytes=None,
    start_id=None,
    end_id=None,
    resume_from=None,
    job_id=None,
    cluster_url=None,
    node_id=None,
    max_ram_gb=None,
    profile=None,
    autotune=None,
    constraints=None,
    runtime=None,
    tokenizer="gpt2".to_string(),
    sequence_length=2048,
    stride=2048,
    add_bos=false,
    add_eos=true,
    truncate="right".to_string(),
    return_attention_mask=true,
    decode_error_policy="error".to_string(),
    autopack=false,
    autopack_shard_mb=512,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn py_text(
    py: Python<'_>,
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
    start_id: Option<u64>,
    end_id: Option<u64>,
    resume_from: Option<Vec<u8>>,
    job_id: Option<String>,
    cluster_url: Option<String>,
    node_id: Option<String>,
    max_ram_gb: Option<f64>,
    profile: Option<String>,
    autotune: Option<bool>,
    constraints: Option<Py<Constraints>>,
    runtime: Option<Py<RuntimeConfig>>,
    tokenizer: String,
    sequence_length: usize,
    stride: usize,
    add_bos: bool,
    add_eos: bool,
    truncate: String,
    return_attention_mask: bool,
    decode_error_policy: String,
    autopack: bool,
    autopack_shard_mb: u64,
) -> PyResult<Py<TextLoader>> {
    let _ = (max_inflight_bytes, max_queue_batches, prefetch_batches);
    if sequence_length == 0 {
        return Err(PyValueError::new_err("sequence_length must be > 0"));
    }
    if stride == 0 {
        return Err(PyValueError::new_err("stride must be > 0"));
    }
    let truncate_mode =
        TextTruncateMode::parse(&truncate).map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let decode_policy = TextDecodeErrorPolicy::parse(&decode_error_policy)
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;

    if autopack && dataset_link.starts_with("s3://") && !dataset_link.contains('@') {
        maybe_autopack(py, &dataset_link, autopack_shard_mb)?;
    }

    let tokenizer = Arc::new(load_text_tokenizer(&tokenizer)?);
    let bos_token_id = resolve_special_token_id(
        tokenizer.as_ref(),
        &[
            "<|bos|>",
            "<bos>",
            "<s>",
            "[BOS]",
            "<|startoftext|>",
            "<|endoftext|>",
        ],
    );
    let eos_token_id = resolve_special_token_id(
        tokenizer.as_ref(),
        &["<|eos|>", "</s>", "<eos>", "[EOS]", "<|endoftext|>"],
    );
    if add_bos && bos_token_id.is_none() {
        return Err(PyValueError::new_err(
            "add_bos=True but tokenizer has no known BOS token id",
        ));
    }
    if add_eos && eos_token_id.is_none() {
        return Err(PyValueError::new_err(
            "add_eos=True but tokenizer has no known EOS token id",
        ));
    }
    let pad_token_id = text_pad_token_id(tokenizer.as_ref(), eos_token_id);

    let constraints_cfg = constraints.as_ref().map(|c| c.bind(py).borrow().clone());
    let runtime_cfg = runtime.as_ref().map(|r| r.bind(py).borrow().clone());
    let selected_profile = AutotuneProfile::from_name(profile.as_deref());
    let defaults = ProfileDefaults::for_profile(selected_profile);

    let mut effective_max_inflight_bytes = defaults.max_inflight_bytes;
    let mut effective_max_queue_batches = defaults.max_queue_batches;
    let mut effective_prefetch_batches = defaults.prefetch_batches;
    let mut effective_max_process_rss_bytes = env_u64("MX8_MAX_PROCESS_RSS_BYTES");
    let max_ram_bytes_from_gb = max_ram_gb_to_bytes(max_ram_gb)?;

    {
        let _autotune_enabled = autotune.unwrap_or(true);

        if let Some(runtime_cfg) = &runtime_cfg {
            if let Some(prefetch) = runtime_cfg.prefetch_batches {
                effective_prefetch_batches = prefetch.max(1);
            }
            if let Some(max_queue) = runtime_cfg.max_queue_batches {
                effective_max_queue_batches = max_queue.max(1);
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
                    mode = "text",
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
    }

    if distributed_requested(cluster_url.as_deref()) {
        if start_id.is_some() || end_id.is_some() {
            return Err(PyValueError::new_err(
                "start_id/end_id are unsupported in distributed mode",
            ));
        }
        if autopack {
            return Err(PyValueError::new_err(
                "autopack is unsupported in distributed mode",
            ));
        }
        if manifest_store.is_some() || manifest_path.is_some() {
            return Err(PyValueError::new_err(
                "manifest_store/manifest_path are unsupported in distributed mode; coordinator owns snapshot resolution",
            ));
        }
        let coord_url = effective_cluster_url(cluster_url).ok_or_else(|| {
            PyValueError::new_err(
                "distributed mode requires cluster_url (or MX8_CLUSTER_URL / MX8_COORD_URL)",
            )
        })?;
        let job_id = job_id.or_else(|| env_string("MX8_JOB_ID")).ok_or_else(|| {
            PyValueError::new_err("distributed mode requires job_id (or MX8_JOB_ID)")
        })?;
        let rank = rank_from_env();
        let node_id = node_id.unwrap_or_else(|| format!("rank{rank}"));
        let distributed_loader = DistributedDataLoader::new(
            py,
            coord_url,
            job_id,
            node_id,
            batch_size_samples,
            effective_max_inflight_bytes,
            effective_max_queue_batches,
            effective_prefetch_batches,
            target_batch_bytes,
            max_batch_bytes,
            1,
            max_ram_gb,
            profile.clone(),
            autotune,
            constraints,
            runtime,
            500,
            DEFAULT_GRPC_MAX_MESSAGE_BYTES,
            resume_from,
        )?;
        let out = TextLoader {
            loader: TextLoaderInner::Distributed(distributed_loader),
            tokenizer,
            sequence_length,
            stride,
            add_bos,
            add_eos,
            bos_token_id,
            eos_token_id,
            pad_token_id,
            truncate: truncate_mode,
            return_attention_mask,
            decode_error_policy: decode_policy,
        };
        return Py::new(py, out);
    }

    let loader = DataLoader::new(
        dataset_link,
        manifest_store,
        manifest_path,
        recursive,
        batch_size_samples,
        effective_max_inflight_bytes,
        effective_max_queue_batches,
        effective_prefetch_batches,
        target_batch_bytes,
        max_batch_bytes,
        effective_max_process_rss_bytes,
        start_id,
        end_id,
        resume_from,
        node_id,
        profile,
        autotune,
    )?;
    let out = TextLoader {
        loader: TextLoaderInner::Local(loader),
        tokenizer,
        sequence_length,
        stride,
        add_bos,
        add_eos,
        bos_token_id,
        eos_token_id,
        pad_token_id,
        truncate: truncate_mode,
        return_attention_mask,
        decode_error_policy: decode_policy,
    };
    Py::new(py, out)
}

#[pyfunction]
#[pyo3(name = "audio")]
#[pyo3(signature = (
    dataset_link,
    *,
    manifest_store=None,
    manifest_path=None,
    recursive=true,
    batch_size_samples=32,
    max_inflight_bytes=128*1024*1024,
    max_queue_batches=64,
    prefetch_batches=1,
    target_batch_bytes=None,
    max_batch_bytes=None,
    start_id=None,
    end_id=None,
    resume_from=None,
    job_id=None,
    cluster_url=None,
    node_id=None,
    max_ram_gb=None,
    profile=None,
    autotune=None,
    constraints=None,
    runtime=None,
    sample_count=16000,
    channels=1,
    sample_rate_hz=None,
    decode_error_policy="error".to_string(),
    autopack=false,
    autopack_shard_mb=512,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn py_audio(
    py: Python<'_>,
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
    start_id: Option<u64>,
    end_id: Option<u64>,
    resume_from: Option<Vec<u8>>,
    job_id: Option<String>,
    cluster_url: Option<String>,
    node_id: Option<String>,
    max_ram_gb: Option<f64>,
    profile: Option<String>,
    autotune: Option<bool>,
    constraints: Option<Py<Constraints>>,
    runtime: Option<Py<RuntimeConfig>>,
    sample_count: usize,
    channels: usize,
    sample_rate_hz: Option<u32>,
    decode_error_policy: String,
    autopack: bool,
    autopack_shard_mb: u64,
) -> PyResult<Py<AudioLoader>> {
    let _ = (max_inflight_bytes, max_queue_batches, prefetch_batches);
    if sample_count == 0 {
        return Err(PyValueError::new_err("sample_count must be > 0"));
    }
    if channels != 1 {
        return Err(PyValueError::new_err(
            "mx8.audio currently supports channels=1 only (mono output)",
        ));
    }
    if let Some(hz) = sample_rate_hz {
        if hz == 0 {
            return Err(PyValueError::new_err("sample_rate_hz must be > 0 when set"));
        }
    }
    let decode_policy = AudioDecodeErrorPolicy::parse(&decode_error_policy)
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;

    if autopack && dataset_link.starts_with("s3://") && !dataset_link.contains('@') {
        maybe_autopack(py, &dataset_link, autopack_shard_mb)?;
    }

    let constraints_cfg = constraints.as_ref().map(|c| c.bind(py).borrow().clone());
    let runtime_cfg = runtime.as_ref().map(|r| r.bind(py).borrow().clone());
    let selected_profile = AutotuneProfile::from_name(profile.as_deref());
    let defaults = ProfileDefaults::for_profile(selected_profile);

    let mut effective_max_inflight_bytes = defaults.max_inflight_bytes;
    let mut effective_max_queue_batches = defaults.max_queue_batches;
    let mut effective_prefetch_batches = defaults.prefetch_batches;
    let mut effective_max_process_rss_bytes = env_u64("MX8_MAX_PROCESS_RSS_BYTES");
    let max_ram_bytes_from_gb = max_ram_gb_to_bytes(max_ram_gb)?;

    {
        let _autotune_enabled = autotune.unwrap_or(true);

        if let Some(runtime_cfg) = &runtime_cfg {
            if let Some(prefetch) = runtime_cfg.prefetch_batches {
                effective_prefetch_batches = prefetch.max(1);
            }
            if let Some(max_queue) = runtime_cfg.max_queue_batches {
                effective_max_queue_batches = max_queue.max(1);
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
                    mode = "audio",
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
    }

    let decoded_samples_total = Arc::new(AtomicU64::new(0));
    let decode_failures_total = Arc::new(AtomicU64::new(0));
    let decoded_frames_total = Arc::new(AtomicU64::new(0));

    if distributed_requested(cluster_url.as_deref()) {
        if start_id.is_some() || end_id.is_some() {
            return Err(PyValueError::new_err(
                "start_id/end_id are unsupported in distributed mode",
            ));
        }
        if autopack {
            return Err(PyValueError::new_err(
                "autopack is unsupported in distributed mode",
            ));
        }
        if manifest_store.is_some() || manifest_path.is_some() {
            return Err(PyValueError::new_err(
                "manifest_store/manifest_path are unsupported in distributed mode; coordinator owns snapshot resolution",
            ));
        }
        let coord_url = effective_cluster_url(cluster_url).ok_or_else(|| {
            PyValueError::new_err(
                "distributed mode requires cluster_url (or MX8_CLUSTER_URL / MX8_COORD_URL)",
            )
        })?;
        let job_id = job_id.or_else(|| env_string("MX8_JOB_ID")).ok_or_else(|| {
            PyValueError::new_err("distributed mode requires job_id (or MX8_JOB_ID)")
        })?;
        let rank = rank_from_env();
        let node_id = node_id.unwrap_or_else(|| format!("rank{rank}"));
        let distributed_loader = DistributedDataLoader::new(
            py,
            coord_url,
            job_id,
            node_id,
            batch_size_samples,
            effective_max_inflight_bytes,
            effective_max_queue_batches,
            effective_prefetch_batches,
            target_batch_bytes,
            max_batch_bytes,
            1,
            max_ram_gb,
            profile.clone(),
            autotune,
            constraints,
            runtime,
            500,
            DEFAULT_GRPC_MAX_MESSAGE_BYTES,
            resume_from,
        )?;
        let out = AudioLoader {
            loader: AudioLoaderInner::Distributed(distributed_loader),
            sample_count,
            channels,
            sample_rate_hz,
            decode_error_policy: decode_policy,
            decoded_samples_total,
            decode_failures_total,
            decoded_frames_total,
        };
        return Py::new(py, out);
    }

    let loader = DataLoader::new(
        dataset_link,
        manifest_store,
        manifest_path,
        recursive,
        batch_size_samples,
        effective_max_inflight_bytes,
        effective_max_queue_batches,
        effective_prefetch_batches,
        target_batch_bytes,
        max_batch_bytes,
        effective_max_process_rss_bytes,
        start_id,
        end_id,
        resume_from,
        node_id,
        profile,
        autotune,
    )?;
    let out = AudioLoader {
        loader: AudioLoaderInner::Local(loader),
        sample_count,
        channels,
        sample_rate_hz,
        decode_error_policy: decode_policy,
        decoded_samples_total,
        decode_failures_total,
        decoded_frames_total,
    };
    Py::new(py, out)
}

#[pyfunction]
#[pyo3(name = "image")]
#[pyo3(signature = (
    dataset_link,
    *,
    manifest_store=None,
    manifest_path=None,
    recursive=true,
    batch_size_samples=32,
    max_inflight_bytes=128*1024*1024,
    max_queue_batches=64,
    prefetch_batches=1,
    target_batch_bytes=None,
    max_batch_bytes=None,
    start_id=None,
    end_id=None,
    resume_from=None,
    job_id=None,
    cluster_url=None,
    node_id=None,
    max_ram_gb=None,
    profile=None,
    autotune=None,
    constraints=None,
    runtime=None,
    augment=None,
    resize_hw=None,
    crop_hw=None,
    horizontal_flip_p=None,
    color_jitter_brightness=None,
    color_jitter_contrast=None,
    color_jitter_saturation=None,
    color_jitter_hue=None,
    normalize_mean=None,
    normalize_std=None,
    seed=0,
    epoch=0,
    to_float=true,
    autopack=false,
    autopack_shard_mb=512,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn py_image(
    py: Python<'_>,
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
    start_id: Option<u64>,
    end_id: Option<u64>,
    resume_from: Option<Vec<u8>>,
    job_id: Option<String>,
    cluster_url: Option<String>,
    node_id: Option<String>,
    max_ram_gb: Option<f64>,
    profile: Option<String>,
    autotune: Option<bool>,
    constraints: Option<Py<Constraints>>,
    runtime: Option<Py<RuntimeConfig>>,
    augment: Option<String>,
    resize_hw: Option<(u32, u32)>,
    crop_hw: Option<(u32, u32)>,
    horizontal_flip_p: Option<f32>,
    color_jitter_brightness: Option<f32>,
    color_jitter_contrast: Option<f32>,
    color_jitter_saturation: Option<f32>,
    color_jitter_hue: Option<f32>,
    normalize_mean: Option<(f32, f32, f32)>,
    normalize_std: Option<(f32, f32, f32)>,
    seed: u64,
    epoch: u64,
    to_float: bool,
    autopack: bool,
    autopack_shard_mb: u64,
) -> PyResult<Py<ImageLoader>> {
    let _ = (max_inflight_bytes, max_queue_batches, prefetch_batches);

    if autopack && dataset_link.starts_with("s3://") && !dataset_link.contains('@') {
        maybe_autopack(py, &dataset_link, autopack_shard_mb)?;
    }

    let constraints_cfg = constraints.as_ref().map(|c| c.bind(py).borrow().clone());
    let runtime_cfg = runtime.as_ref().map(|r| r.bind(py).borrow().clone());
    let selected_profile = AutotuneProfile::from_name(profile.as_deref());
    let defaults = ProfileDefaults::for_profile(selected_profile);
    let mut resolved_resize_hw = resize_hw;
    let mut resolved_crop_hw = crop_hw;
    let mut resolved_horizontal_flip_p = horizontal_flip_p.unwrap_or(0.0);
    let mut resolved_color_jitter_brightness = color_jitter_brightness.unwrap_or(0.0);
    let mut resolved_color_jitter_contrast = color_jitter_contrast.unwrap_or(0.0);
    let mut resolved_color_jitter_saturation = color_jitter_saturation.unwrap_or(0.0);
    let mut resolved_color_jitter_hue = color_jitter_hue.unwrap_or(0.0);
    let mut resolved_normalize_mean = normalize_mean.map(|(r, g, b)| [r, g, b]);
    let mut resolved_normalize_std = normalize_std.map(|(r, g, b)| [r, g, b]);

    if let Some(augment_name_raw) = augment.as_deref() {
        let augment_name = augment_name_raw.trim().to_ascii_lowercase();
        match augment_name.as_str() {
            "imagenet" | "standard" => {
                if resolved_resize_hw.is_none() {
                    resolved_resize_hw = Some((256, 256));
                }
                if resolved_crop_hw.is_none() {
                    resolved_crop_hw = Some((224, 224));
                }
                if horizontal_flip_p.is_none() {
                    resolved_horizontal_flip_p = 0.5;
                }
                if color_jitter_brightness.is_none() {
                    resolved_color_jitter_brightness = 0.4;
                }
                if color_jitter_contrast.is_none() {
                    resolved_color_jitter_contrast = 0.4;
                }
                if color_jitter_saturation.is_none() {
                    resolved_color_jitter_saturation = 0.4;
                }
                if color_jitter_hue.is_none() {
                    resolved_color_jitter_hue = 0.0;
                }
                if resolved_normalize_mean.is_none() {
                    resolved_normalize_mean = Some([0.485, 0.456, 0.406]);
                }
                if resolved_normalize_std.is_none() {
                    resolved_normalize_std = Some([0.229, 0.224, 0.225]);
                }
            }
            "" => {
                return Err(PyValueError::new_err(
                    "augment must be non-empty when provided",
                ));
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported augment={augment_name_raw:?} (expected: imagenet|standard)"
                )));
            }
        }
    }

    if let Some((crop_h, crop_w)) = resolved_crop_hw {
        if crop_h == 0 || crop_w == 0 {
            return Err(PyValueError::new_err("crop_hw must be positive"));
        }
        if let Some((resize_h, resize_w)) = resolved_resize_hw {
            if crop_h > resize_h || crop_w > resize_w {
                return Err(PyValueError::new_err(format!(
                    "crop_hw {:?} exceeds resize_hw {:?}",
                    (crop_h, crop_w),
                    (resize_h, resize_w)
                )));
            }
        }
    }
    if !resolved_horizontal_flip_p.is_finite() || !(0.0..=1.0).contains(&resolved_horizontal_flip_p)
    {
        return Err(PyValueError::new_err(format!(
            "horizontal_flip_p must be in [0, 1], got {}",
            resolved_horizontal_flip_p
        )));
    }
    for (name, value) in [
        ("color_jitter_brightness", resolved_color_jitter_brightness),
        ("color_jitter_contrast", resolved_color_jitter_contrast),
        ("color_jitter_saturation", resolved_color_jitter_saturation),
        ("color_jitter_hue", resolved_color_jitter_hue),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(PyValueError::new_err(format!(
                "{name} must be finite and >= 0, got {value}"
            )));
        }
    }
    if resolved_color_jitter_hue > 0.0 {
        return Err(PyValueError::new_err(
            "color_jitter_hue > 0 is not supported yet (set color_jitter_hue=0.0)",
        ));
    }
    if resolved_normalize_mean.is_some() ^ resolved_normalize_std.is_some() {
        return Err(PyValueError::new_err(
            "normalize_mean and normalize_std must be provided together",
        ));
    }
    if let (Some(mean), Some(std)) = (resolved_normalize_mean, resolved_normalize_std) {
        for (idx, m) in mean.iter().enumerate() {
            if !m.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "normalize_mean[{idx}] must be finite"
                )));
            }
        }
        for (idx, s) in std.iter().enumerate() {
            if !s.is_finite() || *s <= 0.0 {
                return Err(PyValueError::new_err(format!(
                    "normalize_std[{idx}] must be finite and > 0"
                )));
            }
        }
    }

    let mut effective_max_inflight_bytes = defaults.max_inflight_bytes;
    let mut effective_max_queue_batches = defaults.max_queue_batches;
    let mut effective_prefetch_batches = defaults.prefetch_batches;
    let mut effective_max_process_rss_bytes = env_u64("MX8_MAX_PROCESS_RSS_BYTES");
    let max_ram_bytes_from_gb = max_ram_gb_to_bytes(max_ram_gb)?;

    {
        let _autotune_enabled = autotune.unwrap_or(true);

        if let Some(runtime_cfg) = &runtime_cfg {
            if let Some(prefetch) = runtime_cfg.prefetch_batches {
                effective_prefetch_batches = prefetch.max(1);
            }
            if let Some(max_queue) = runtime_cfg.max_queue_batches {
                effective_max_queue_batches = max_queue.max(1);
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
                    mode = "image",
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
    }

    if distributed_requested(cluster_url.as_deref()) {
        if start_id.is_some() || end_id.is_some() {
            return Err(PyValueError::new_err(
                "start_id/end_id are unsupported in distributed mode",
            ));
        }
        if autopack {
            return Err(PyValueError::new_err(
                "autopack is unsupported in distributed mode",
            ));
        }
        if manifest_store.is_some() || manifest_path.is_some() {
            return Err(PyValueError::new_err(
                "manifest_store/manifest_path are unsupported in distributed mode; coordinator owns snapshot resolution",
            ));
        }
        let coord_url = effective_cluster_url(cluster_url).ok_or_else(|| {
            PyValueError::new_err(
                "distributed mode requires cluster_url (or MX8_CLUSTER_URL / MX8_COORD_URL)",
            )
        })?;
        let job_id = job_id.or_else(|| env_string("MX8_JOB_ID")).ok_or_else(|| {
            PyValueError::new_err("distributed mode requires job_id (or MX8_JOB_ID)")
        })?;
        let rank = rank_from_env();
        let node_id = node_id.unwrap_or_else(|| format!("rank{rank}"));
        let distributed_loader = DistributedDataLoader::new(
            py,
            coord_url,
            job_id,
            node_id,
            batch_size_samples,
            effective_max_inflight_bytes,
            effective_max_queue_batches,
            effective_prefetch_batches,
            target_batch_bytes,
            max_batch_bytes,
            1,
            max_ram_gb,
            profile.clone(),
            autotune,
            constraints,
            runtime,
            500,
            DEFAULT_GRPC_MAX_MESSAGE_BYTES,
            resume_from,
        )?;
        let decode_backend = decode_backend_from_env()?;
        let rust_jpeg_codec = rust_jpeg_codec_from_env()?;
        let rust_resize_backend = rust_resize_backend_from_env()?;
        let decode_threads = decode_threads_from_env()?;
        let decode_pool = if matches!(decode_backend, DecodeBackend::Rust) && decode_threads > 1 {
            Some(Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(decode_threads)
                    .build()
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "failed to build rust decode thread pool: {e}"
                        ))
                    })?,
            ))
        } else {
            None
        };
        let manifest_hash_seed = stable_hash64(distributed_loader.manifest_hash().as_bytes());
        let out = ImageLoader {
            loader: ImageLoaderInner::Distributed(distributed_loader),
            resize_hw: resolved_resize_hw,
            crop_hw: resolved_crop_hw,
            horizontal_flip_p: resolved_horizontal_flip_p,
            color_jitter_brightness: resolved_color_jitter_brightness,
            color_jitter_contrast: resolved_color_jitter_contrast,
            color_jitter_saturation: resolved_color_jitter_saturation,
            color_jitter_hue: resolved_color_jitter_hue,
            normalize_mean: resolved_normalize_mean,
            normalize_std: resolved_normalize_std,
            seed,
            epoch,
            manifest_hash_seed,
            to_float,
            decode_backend,
            rust_jpeg_codec,
            rust_resize_backend,
            decode_threads,
            decode_pool,
            classes: None,
        };
        return Py::new(py, out);
    }

    let loader = ImageLoader::new(
        dataset_link,
        manifest_store,
        manifest_path,
        recursive,
        batch_size_samples,
        effective_max_inflight_bytes,
        effective_max_queue_batches,
        effective_prefetch_batches,
        target_batch_bytes,
        max_batch_bytes,
        effective_max_process_rss_bytes,
        start_id,
        end_id,
        resume_from,
        node_id,
        profile,
        autotune,
        resolved_resize_hw,
        resolved_crop_hw,
        resolved_horizontal_flip_p,
        resolved_color_jitter_brightness,
        resolved_color_jitter_contrast,
        resolved_color_jitter_saturation,
        resolved_color_jitter_hue,
        resolved_normalize_mean.map(|v| (v[0], v[1], v[2])),
        resolved_normalize_std.map(|v| (v[0], v[1], v[2])),
        seed,
        epoch,
        to_float,
    )?;
    Py::new(py, loader)
}

#[pyfunction]
#[pyo3(signature = (
    dataset_link,
    *,
    manifest_store=None,
    manifest_path=None,
    recursive=true,
    clip_len=16,
    stride=8,
    fps=8,
    batch_size_samples=32,
    seed=0,
    epoch=0,
    resume_from=None,
    job_id=None,
    cluster_url=None,
    max_ram_gb=None,
    profile=None,
    autotune=None,
    constraints=None,
    runtime=None,
    node_id=None
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn video(
    py: Python<'_>,
    dataset_link: String,
    manifest_store: Option<PathBuf>,
    manifest_path: Option<PathBuf>,
    recursive: bool,
    clip_len: u32,
    stride: u32,
    fps: u32,
    batch_size_samples: usize,
    seed: u64,
    epoch: u64,
    resume_from: Option<Vec<u8>>,
    job_id: Option<String>,
    cluster_url: Option<String>,
    max_ram_gb: Option<f64>,
    profile: Option<String>,
    autotune: Option<bool>,
    constraints: Option<Py<Constraints>>,
    runtime: Option<Py<RuntimeConfig>>,
    node_id: Option<String>,
) -> PyResult<Py<VideoDataLoader>> {
    if fps == 0 {
        return Err(PyValueError::new_err("video fps must be > 0"));
    }
    let root = manifest_store
        .or(env_path("MX8_MANIFEST_STORE_ROOT"))
        .unwrap_or_else(default_manifest_store);
    let dev_manifest_path = manifest_path.or(env_path("MX8_DEV_MANIFEST_PATH"));
    let store = std::sync::Arc::new(FsManifestStore::new(root));
    let resolver = SnapshotResolver::new(
        store,
        SnapshotResolverConfig {
            dev_manifest_path,
            recursive,
            materialize_manifest_bytes: true,
            ..SnapshotResolverConfig::default()
        },
    );
    let snapshot = resolver
        .resolve(
            &dataset_link,
            LockOwner {
                node_id: node_id.clone(),
            },
        )
        .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
    let resume_token = resume_from
        .as_deref()
        .map(VideoLoaderCheckpointToken::decode)
        .transpose()
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;

    let fps_policy = format!("fixed_fps:{fps}");
    let cfg = VideoStage1Config {
        clip_len,
        stride,
        fps_policy: fps_policy.clone(),
        seed,
        epoch,
        max_clips_in_memory: env_usize("MX8_VIDEO_STAGE2_MAX_CLIPS_IN_MEMORY", 2_000_000),
    };
    let index = build_video_stage1_index_from_manifest_bytes(
        &snapshot.manifest_hash,
        &snapshot.manifest_bytes,
        &cfg,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
    let stage2d_sidecars = build_stage2d_sidecar_map(&snapshot.manifest_bytes)
        .map_err(|e| PyRuntimeError::new_err(format!("build stage2d sidecar map failed: {e}")))?;
    let mut clips = index.clips;
    let mut assigned_rank = 0u32;
    let mut world_size = 1u32;
    if distributed_requested(cluster_url.as_deref()) {
        world_size = world_size_from_env().max(1);
        assigned_rank = rank_from_env().min(world_size.saturating_sub(1));
        if world_size > 1 {
            clips = clips
                .into_iter()
                .enumerate()
                .filter_map(|(idx, clip)| {
                    let idx = idx as u32;
                    if idx % world_size == assigned_rank {
                        Some(clip)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
        }
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("failed to start tokio runtime: {e}")))?;

    let constraints_cfg = constraints.as_ref().map(|c| c.bind(py).borrow().clone());
    let runtime_cfg = runtime.as_ref().map(|r| r.bind(py).borrow().clone());

    if let Some(runtime_cfg) = &runtime_cfg {
        if runtime_cfg.prefetch_batches.is_some()
            || runtime_cfg.max_queue_batches.is_some()
            || runtime_cfg.want.is_some()
        {
            return Err(PyValueError::new_err(
                "mx8.video runtime overrides are unsupported in this stage",
            ));
        }
    }

    let selected_profile = AutotuneProfile::from_name(profile.as_deref());
    let defaults = ProfileDefaults::for_profile(selected_profile);
    let autotune_enabled = autotune.unwrap_or(true);
    let mut effective_max_inflight_bytes = defaults.max_inflight_bytes;
    let mut effective_max_process_rss_bytes = env_u64("MX8_MAX_PROCESS_RSS_BYTES");
    let max_ram_bytes_from_gb = max_ram_gb_to_bytes(max_ram_gb)?;

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
        effective_max_process_rss_bytes =
            derive_default_max_process_rss_bytes(selected_profile, effective_max_inflight_bytes);
        if let Some(cap) = effective_max_process_rss_bytes {
            tracing::info!(
                target: "mx8_proof",
                event = "rss_cap_defaulted",
                mode = "video",
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
    let max_inflight_bytes = effective_max_inflight_bytes.max(1);
    let decode_backend = video_decode_backend_from_env()?;
    let bytes_per_clip = env_usize("MX8_VIDEO_STAGE2_BYTES_PER_CLIP", 4096).max(1);
    let decode_contract = VideoDataLoader::derive_decode_contract(clip_len, bytes_per_clip)?;
    if batch_size_samples == 0 {
        return Err(PyValueError::new_err(
            "video batch_size_samples must be > 0",
        ));
    }
    if (batch_size_samples as u64).saturating_mul(bytes_per_clip as u64) > max_inflight_bytes {
        return Err(PyValueError::new_err(format!(
            "video batch_size_samples ({batch_size_samples}) * bytes_per_clip ({bytes_per_clip}) exceeds max_inflight_bytes ({max_inflight_bytes}); lower batch size or raise cap"
        )));
    }
    let video_runtime_autotune_period_batches = env_u64("MX8_VIDEO_AUTOTUNE_PERIOD_BATCHES")
        .unwrap_or(16)
        .max(1);

    let clips_total = clips.len() as u64;
    let mut next_idx = 0usize;
    if let Some(token) = &resume_token {
        if token.manifest_hash != snapshot.manifest_hash.0 {
            return Err(PyValueError::new_err(format!(
                "resume_from manifest_hash mismatch: token={} current={}",
                token.manifest_hash, snapshot.manifest_hash.0
            )));
        }
        if token.seed != seed {
            return Err(PyValueError::new_err(format!(
                "resume_from seed mismatch: token={} current={}",
                token.seed, seed
            )));
        }
        if token.epoch != epoch {
            return Err(PyValueError::new_err(format!(
                "resume_from epoch mismatch: token={} current={}",
                token.epoch, epoch
            )));
        }
        if token.clip_len != clip_len || token.stride != stride || token.fps != fps {
            return Err(PyValueError::new_err(
                "resume_from video clip configuration mismatch",
            ));
        }
        if token.assigned_rank != assigned_rank || token.world_size != world_size {
            return Err(PyValueError::new_err(format!(
                "resume_from distributed shape mismatch: token rank/world={}/{} current={}/{}",
                token.assigned_rank, token.world_size, assigned_rank, world_size
            )));
        }
        if token.clips_total != clips_total {
            return Err(PyValueError::new_err(format!(
                "resume_from clips_total mismatch: token={} current={}",
                token.clips_total, clips_total
            )));
        }
        next_idx = usize::try_from(token.next_idx)
            .map_err(|_| PyValueError::new_err("resume_from next_idx overflow"))?;
        if next_idx > clips.len() {
            return Err(PyValueError::new_err("resume_from next_idx out of range"));
        }
    }

    tracing::info!(
        target: "mx8_proof",
        event = "video_loader_initialized",
        manifest_hash = %snapshot.manifest_hash.0,
        clip_len = clip_len as u64,
        stride = stride as u64,
        fps = fps as u64,
        batch_size_samples = batch_size_samples as u64,
        clips_total = clips_total,
        stage2d_sidecars = stage2d_sidecars.len() as u64,
        decode_backend = video_decode_backend_name(decode_backend),
        max_inflight_bytes = max_inflight_bytes,
        max_process_rss_bytes = effective_max_process_rss_bytes.unwrap_or(0),
        profile = match selected_profile {
            AutotuneProfile::Safe => "safe",
            AutotuneProfile::Balanced => "balanced",
            AutotuneProfile::Throughput => "throughput",
        },
        autotune_enabled = autotune_enabled,
        bytes_per_clip = bytes_per_clip as u64,
        seed = seed,
        epoch = epoch,
        assigned_rank = assigned_rank,
        world_size = world_size,
        "initialized stage2b video loader"
    );

    Py::new(
        py,
        VideoDataLoader {
            manifest_hash: snapshot.manifest_hash.0,
            clips,
            stage2d_sidecars,
            rt,
            decode_backend,
            next_idx,
            batch_size_samples,
            max_inflight_bytes,
            bytes_per_clip,
            decode_contract,
            seed,
            epoch,
            clip_len,
            stride,
            decode_fps: fps,
            fps_policy,
            delivered_batches: 0,
            delivered_samples: 0,
            delivered_bytes: 0,
            decode_attempted_clips: 0,
            decode_succeeded_clips: 0,
            decode_failed_io_read_failed: 0,
            decode_failed_corrupt_media: 0,
            decode_failed_short_media: 0,
            decode_failed_unsupported_codec: 0,
            decode_failed_missing_stream: 0,
            decode_failed_backend_unavailable: 0,
            decode_failed_decode_failed: 0,
            decode_backend_fallback_total: 0,
            decode_ms_total: 0,
            s3_range_requests_total: 0,
            s3_range_bytes_fetched_total: 0,
            s3_stage2d_plan_used_total: 0,
            s3_stage2d_plan_fallback_total: 0,
            s3_full_object_range_fallback_total: 0,
            video_runtime_autotune_enabled: autotune_enabled,
            video_runtime_autotune_period_batches,
            video_runtime_autotune_last_batch: 0,
            video_runtime_autotune_adjustments_total: 0,
            video_runtime_autotune_gpu_clamps_total: 0,
            video_runtime_autotune_pressure_milli: 0,
            video_gpu_pressure_milli: 0,
            video_gpu_pressure_available: false,
            video_gpu_pressure_unavailable_total: 0,
            video_gpu_recovery_streak: 0,
            video_last_gpu_sample_at: None,
            video_max_process_rss_bytes: effective_max_process_rss_bytes,
            assigned_rank,
            world_size,
            job_id,
            cluster_url,
            started_at: Instant::now(),
        },
    )
}

#[pyfunction]
#[pyo3(signature = (
    loaders,
    *,
    weights,
    seed=0,
    epoch=0,
    starvation_window=10_000,
    source_exhausted="error",
    resume_from=None,
    job_id=None,
    cluster_url=None,
    max_ram_gb=None,
    profile=None,
    autotune=None,
    constraints=None,
    runtime=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn mix(
    py: Python<'_>,
    loaders: Vec<Py<PyAny>>,
    weights: Vec<f64>,
    seed: u64,
    epoch: u64,
    starvation_window: u64,
    source_exhausted: &str,
    resume_from: Option<Vec<u8>>,
    job_id: Option<String>,
    cluster_url: Option<String>,
    max_ram_gb: Option<f64>,
    profile: Option<String>,
    autotune: Option<bool>,
    constraints: Option<Py<Constraints>>,
    runtime: Option<Py<RuntimeConfig>>,
) -> PyResult<Py<MixedDataLoader>> {
    let _ = (job_id, cluster_url);
    if loaders.is_empty() {
        return Err(PyValueError::new_err(
            "mix requires at least one loader instance",
        ));
    }
    if weights.len() != loaders.len() {
        return Err(PyValueError::new_err(format!(
            "mix weights length ({}) must match loader count ({})",
            weights.len(),
            loaders.len()
        )));
    }
    let normalized =
        normalize_mix_weights(&weights).map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let resume_token = resume_from
        .as_deref()
        .map(MixLoaderCheckpointToken::decode)
        .transpose()
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;

    let constraints_cfg = constraints.as_ref().map(|c| c.bind(py).borrow().clone());
    let runtime_cfg = runtime.as_ref().map(|r| r.bind(py).borrow().clone());
    let max_ram_bytes_from_gb = max_ram_gb_to_bytes(max_ram_gb)?;
    let mut parsed_loaders = Vec::<MixLoader>::with_capacity(loaders.len());
    for loader in loaders {
        let bound = loader.bind(py);
        if let Ok(local) = bound.extract::<Py<DataLoader>>() {
            parsed_loaders.push(MixLoader::Local(local));
            continue;
        }
        if let Ok(distributed) = bound.extract::<Py<DistributedDataLoader>>() {
            parsed_loaders.push(MixLoader::Distributed(distributed));
            continue;
        }
        return Err(PyValueError::new_err(
            "mix requires loaders created by mx8.load(...) (local or distributed)",
        ));
    }

    let mut inflight_caps = Vec::with_capacity(parsed_loaders.len());
    let mut batch_sizes = Vec::with_capacity(parsed_loaders.len());
    for loader in &parsed_loaders {
        let (_, _, batch_size_samples, max_inflight_bytes, _, _) =
            MixedDataLoader::source_config(loader, py);
        if max_inflight_bytes > 0 {
            inflight_caps.push(max_inflight_bytes);
        }
        if batch_size_samples > 0 {
            batch_sizes.push(batch_size_samples);
        }
    }
    let loader_shared_max_inflight_bytes = if inflight_caps.is_empty() {
        128 * 1024 * 1024
    } else {
        compute_shared_mix_cap(&inflight_caps).map_err(|e| PyValueError::new_err(format!("{e}")))?
    };

    if let Some(&first_batch_size) = batch_sizes.first() {
        if batch_sizes.iter().any(|v| *v != first_batch_size) {
            return Err(PyValueError::new_err(
                "mix requires all loaders to have identical batch_size_samples",
            ));
        }
    }

    let selected_profile = AutotuneProfile::from_name(profile.as_deref());
    let mix_profile = Some(
        match selected_profile {
            AutotuneProfile::Safe => "safe",
            AutotuneProfile::Balanced => "balanced",
            AutotuneProfile::Throughput => "throughput",
        }
        .to_string(),
    );
    let mix_autotune_enabled = autotune.unwrap_or(true);
    let mut runtime_prefetch_override: Option<usize> = None;
    let mut runtime_max_queue_override: Option<usize> = None;
    let mut shared_max_inflight_override: Option<u64> = None;
    let mut mix_autotune_rails: Option<AutotuneRails> = None;
    let mut mix_max_process_rss_bytes: Option<u64> = None;

    {
        if mix_autotune_enabled {
            let defaults = ProfileDefaults::for_profile(selected_profile);
            mix_autotune_rails = Some(AutotuneRails::for_profile(selected_profile));
            runtime_prefetch_override = Some(defaults.prefetch_batches.max(1));
            runtime_max_queue_override = Some(defaults.max_queue_batches.max(1));
            shared_max_inflight_override = Some(defaults.max_inflight_bytes.max(1));
        }

        if let Some(runtime_cfg) = &runtime_cfg {
            if runtime_cfg.want.is_some() {
                return Err(PyValueError::new_err(
                    "mx8.mix runtime.want is unsupported (use DataLoader/DistributedDataLoader for lease parallelism)",
                ));
            }
            if let Some(prefetch) = runtime_cfg.prefetch_batches {
                runtime_prefetch_override = Some(prefetch.max(1));
            }
            if let Some(max_queue) = runtime_cfg.max_queue_batches {
                runtime_max_queue_override = Some(max_queue.max(1));
            }
        }

        if let Some(constraints_cfg) = &constraints_cfg {
            if let Some(max_inflight) = constraints_cfg.max_inflight_bytes {
                shared_max_inflight_override = Some(max_inflight.max(1));
            }
            if let Some(max_process) = constraints_cfg.max_process_rss_bytes {
                mix_max_process_rss_bytes = Some(max_process.max(1));
            }
            if let (Some(max_process_rss), Some(candidate_cap)) = (
                constraints_cfg.max_process_rss_bytes,
                shared_max_inflight_override,
            ) {
                if candidate_cap > max_process_rss {
                    tracing::warn!(
                        target: "mx8_proof",
                        event = "mix_cap_clamped",
                        requested_max_inflight_bytes = candidate_cap,
                        clamped_max_inflight_bytes = max_process_rss,
                        max_process_rss_bytes = max_process_rss,
                        "clamped mix max_inflight_bytes to max_process_rss_bytes"
                    );
                    shared_max_inflight_override = Some(max_process_rss);
                }
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
            mix_max_process_rss_bytes = Some(max_ram_bytes.max(1));
            if let Some(candidate_cap) = shared_max_inflight_override {
                if candidate_cap > max_ram_bytes {
                    shared_max_inflight_override = Some(max_ram_bytes);
                }
            }
        }
    }

    for loader in &parsed_loaders {
        MixedDataLoader::source_apply_runtime_overrides(
            loader,
            py,
            runtime_prefetch_override,
            runtime_max_queue_override,
        );
    }

    let mut queue_caps = Vec::with_capacity(parsed_loaders.len());
    let mut prefetch_caps = Vec::with_capacity(parsed_loaders.len());
    for loader in &parsed_loaders {
        let (_, _, _, _, max_queue_batches, prefetch_batches) =
            MixedDataLoader::source_config(loader, py);
        if max_queue_batches > 0 {
            queue_caps.push(max_queue_batches);
        }
        if prefetch_batches > 0 {
            prefetch_caps.push(prefetch_batches);
        }
    }

    let mut shared_max_inflight_bytes = shared_max_inflight_override
        .map(|v| v.min(loader_shared_max_inflight_bytes))
        .unwrap_or(loader_shared_max_inflight_bytes);

    if mix_max_process_rss_bytes.is_none() {
        mix_max_process_rss_bytes =
            derive_default_max_process_rss_bytes(selected_profile, shared_max_inflight_bytes);
        if let Some(cap) = mix_max_process_rss_bytes {
            tracing::info!(
                target: "mx8_proof",
                event = "rss_cap_defaulted",
                mode = "mix",
                profile = profile_name(selected_profile),
                max_inflight_bytes = shared_max_inflight_bytes,
                max_process_rss_bytes = cap,
                "defaulted process RSS cap from profile and node memory limit"
            );
        }
    }
    if let Some(max_process) = mix_max_process_rss_bytes {
        if shared_max_inflight_bytes > max_process {
            shared_max_inflight_bytes = max_process;
        }
    }

    let mix_effective_prefetch_batches = runtime_prefetch_override
        .unwrap_or_else(|| prefetch_caps.iter().copied().min().unwrap_or(1usize).max(1));
    let mix_effective_max_queue_batches = runtime_max_queue_override
        .unwrap_or_else(|| queue_caps.iter().copied().min().unwrap_or(1usize).max(1));
    let mix_runtime_autotune_enabled = mix_autotune_enabled;
    let mix_runtime_autotune_period_ticks = env_u64("MX8_MIX_AUTOTUNE_PERIOD_TICKS")
        .unwrap_or(32)
        .max(1);

    let scheduler = WeightedRoundRobin::new(normalized.clone(), seed, epoch);
    let source_exhaustion_policy = SourceExhaustionPolicy::parse(source_exhausted)
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let n = parsed_loaders.len();
    let snapshot_enabled = env_bool("MX8_MIX_SNAPSHOT", false);
    let snapshot_period_ticks = env_u64("MX8_MIX_SNAPSHOT_PERIOD_TICKS")
        .unwrap_or(64)
        .max(1);
    tracing::info!(
        target: "mx8_proof",
        event = "mix_initialized",
        sources = n as u64,
        seed = seed,
        epoch = epoch,
        shared_max_inflight_bytes = shared_max_inflight_bytes,
        normalized_weights = ?normalized,
        source_exhaustion_policy = source_exhaustion_policy.as_str(),
        profile = mix_profile.as_deref().unwrap_or("legacy"),
        autotune_enabled = mix_autotune_enabled,
        runtime_autotune_enabled = mix_runtime_autotune_enabled,
        mix_effective_prefetch_batches = mix_effective_prefetch_batches as u64,
        mix_effective_max_queue_batches = mix_effective_max_queue_batches as u64,
        mix_runtime_autotune_period_ticks = mix_runtime_autotune_period_ticks,
        snapshot_enabled = snapshot_enabled,
        snapshot_period_ticks = snapshot_period_ticks,
        max_queue_batches = ?queue_caps,
        prefetch_batches = ?prefetch_caps,
        "initialized mixed loader"
    );
    let mut out = MixedDataLoader {
        loaders: parsed_loaders,
        scheduler,
        active: vec![true; n],
        source_exhausted_total: vec![0; n],
        delivered_batches: vec![0; n],
        delivered_samples: vec![0; n],
        delivered_bytes: vec![0; n],
        starvation_total: vec![0; n],
        steps_since_emit: vec![0; n],
        max_starvation_window: starvation_window.max(1),
        normalized_weights: normalized,
        shared_max_inflight_bytes,
        shared_inflight_violation_total: 0,
        seed,
        epoch,
        schedule_ticks: 0,
        snapshot_enabled,
        snapshot_period_ticks,
        snapshot_emitted_total: 0,
        source_exhaustion_policy,
        mix_profile,
        mix_autotune_enabled,
        mix_effective_prefetch_batches,
        mix_effective_max_queue_batches,
        mix_runtime_autotune_enabled,
        mix_autotune_rails,
        mix_autotune_controller: AutotuneController::new(),
        mix_autotune_current: AutotuneUpdate {
            want: 1,
            prefetch_batches: mix_effective_prefetch_batches,
            max_queue_batches: mix_effective_max_queue_batches,
        },
        mix_runtime_autotune_period_ticks,
        mix_runtime_autotune_last_tick: 0,
        mix_runtime_autotune_adjustments_total: 0,
        mix_runtime_autotune_pressure_milli: 0,
        mix_resume_source_checkpoint_mismatch_total: 0,
        mix_max_process_rss_bytes,
        started_at: Instant::now(),
    };
    if let Some(token) = resume_token {
        out.apply_resume_token(py, &token)?;
    }
    Py::new(py, out)
}

// ── mx8.coordinator() ──────────────────────────────────────────────────────

/// Handle returned by `mx8.coordinator()`. Use as a context manager:
///
/// ```python
/// with mx8.coordinator() as coord:
///     loader = mx8.DistributedDataLoader(coord_url=coord.url, ...)
/// ```
///
/// The coordinator subprocess is terminated when the context exits (or when
/// `coord.stop()` is called explicitly).
#[pyclass]
pub(crate) struct CoordinatorHandle {
    url: String,
    child: Option<std::process::Child>,
}

#[pymethods]
impl CoordinatorHandle {
    /// The HTTP URL the coordinator is listening on (e.g. `http://127.0.0.1:50051`).
    #[getter]
    fn url(&self) -> &str {
        &self.url
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: Option<PyObject>,
        _exc_val: Option<PyObject>,
        _exc_tb: Option<PyObject>,
    ) -> bool {
        self.stop();
        false
    }

    /// Terminate the coordinator subprocess immediately.
    /// Called automatically on context-manager exit or garbage collection.
    fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    fn __del__(&mut self) {
        self.stop();
    }

    fn __repr__(&self) -> String {
        format!("mx8.CoordinatorHandle(url={:?})", self.url)
    }
}

/// Find the `mx8d-coordinator` binary.
/// Checks PATH directories, then the directory containing the current
/// executable (useful when the binary is bundled alongside the wheel).
pub(crate) fn find_coordinator_binary() -> Result<std::path::PathBuf, String> {
    if let Ok(path_var) = std::env::var("PATH") {
        let sep = if cfg!(windows) { ';' } else { ':' };
        for dir in path_var.split(sep) {
            let candidate = std::path::PathBuf::from(dir).join("mx8d-coordinator");
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let candidate = parent.join("mx8d-coordinator");
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    Err("mx8d-coordinator not found in PATH. \
         Build it with: cargo build -p mx8-coordinator \
         then add the binary to your PATH."
        .to_string())
}

/// Bind to port 0 to let the OS pick a free port, then release it.
/// There is a small TOCTOU window but it is acceptable for local use.
pub(crate) fn pick_free_port() -> Option<u16> {
    std::net::TcpListener::bind("127.0.0.1:0")
        .ok()
        .and_then(|l| l.local_addr().ok())
        .map(|a| a.port())
}

/// Start a local coordinator subprocess and return a handle to it.
///
/// Parameters
/// ----------
/// rank : int, optional
///     Current process rank. When set, only rank 0 starts the coordinator
///     subprocess; all other ranks simply wait for it to become ready.
///     This is the correct pattern for torchrun / multi-GPU jobs where every
///     rank runs the same script.  When omitted the function behaves as a
///     single-process helper and always starts the coordinator.
/// world_size : int
///     Number of ranks in this job (default 1).
/// dataset_link : str, optional
///     Dataset link passed to the coordinator for snapshot resolution.
/// port : int, optional
///     Port to bind the coordinator on.
///     When `rank` is set defaults to 50051 (so all ranks agree on the port).
///     When `rank` is not set a free port is chosen automatically.
/// bind_host : str
///     Host/IP the coordinator binds on (default ``"127.0.0.1"``).
///     Set to ``"0.0.0.0"`` for multi-node jobs where other machines need
///     to reach the coordinator.
/// master_addr : str, optional
///     The address other ranks use to reach the coordinator.  Defaults to
///     ``bind_host`` (fine for single-machine).  For multi-node, set this
///     to the coordinator machine's reachable hostname or IP so the returned
///     ``coord.url`` is correct for all ranks on all machines.
/// timeout_secs : int
///     How long every rank waits for the coordinator to become ready
///     (default 30).
/// log : bool
///     Forward the coordinator's stdout/stderr to the parent process's
///     stderr.  Default False (output is suppressed).
///
/// Single-machine multi-GPU (torchrun)
/// ------------------------------------
/// ```python
/// # train.py — every rank runs this
/// import os, mx8
///
/// rank       = int(os.environ["RANK"])
/// world_size = int(os.environ["WORLD_SIZE"])
///
/// with mx8.coordinator(rank=rank, world_size=world_size) as coord:
///     loader = mx8.DistributedDataLoader(
///         coord_url=coord.url,
///         job_id="train",
///         node_id=f"rank{rank}",
///         batch_size_samples=512,
///         max_ram_gb=24,
///         profile="balanced",
///     )
/// ```
///
/// Multi-node
/// ----------
/// ```python
/// # On every node — set bind_host + master_addr so the URL is reachable
/// with mx8.coordinator(
///     rank=rank,
///     world_size=world_size,
///     bind_host="0.0.0.0",
///     master_addr="node0.cluster.local",
///     port=50051,
/// ) as coord:
///     loader = mx8.DistributedDataLoader(coord_url=coord.url, ...)
/// ```
#[pyfunction]
#[pyo3(signature = (*, rank=None, world_size=1, dataset_link=None, port=None, bind_host="127.0.0.1".to_string(), master_addr=None, timeout_secs=30, log=false))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn coordinator(
    rank: Option<u32>,
    world_size: u32,
    dataset_link: Option<String>,
    port: Option<u16>,
    bind_host: String,
    master_addr: Option<String>,
    timeout_secs: u64,
    log: bool,
) -> PyResult<CoordinatorHandle> {
    let is_rank_zero = rank.is_none_or(|r| r == 0);

    // Port selection:
    // - Explicit port always wins.
    // - With rank set: use 50051 so every rank agrees without coordination.
    // - Without rank: auto-pick a free port (single-process helper).
    let port = match port {
        Some(p) => p,
        None => {
            if rank.is_some() {
                50051
            } else {
                pick_free_port().unwrap_or(50051)
            }
        }
    };

    // The address ranks use to connect. For single-machine use 127.0.0.1.
    // For multi-node the caller provides master_addr (the coordinator's
    // public hostname / IP).
    let connect_host = master_addr
        .as_deref()
        .unwrap_or(if bind_host == "0.0.0.0" {
            "127.0.0.1"
        } else {
            &bind_host
        })
        .to_string();

    let bind_addr = format!("{bind_host}:{port}");
    let connect_addr: std::net::SocketAddr = format!("{connect_host}:{port}")
        .parse()
        .map_err(|e| PyRuntimeError::new_err(format!("mx8.coordinator: bad address: {e}")))?;
    let url = format!("http://{connect_host}:{port}");

    // Only rank 0 (or a rank-less single-process call) starts the subprocess.
    let mut child: Option<std::process::Child> = if is_rank_zero {
        let binary = find_coordinator_binary()
            .map_err(|e| PyRuntimeError::new_err(format!("mx8.coordinator: {e}")))?;

        let mut cmd = std::process::Command::new(&binary);
        cmd.arg("--addr").arg(&bind_addr);
        cmd.arg("--world-size").arg(world_size.to_string());
        if let Some(link) = &dataset_link {
            cmd.arg("--dataset-link").arg(link);
        }
        if log {
            cmd.stdout(std::process::Stdio::inherit());
            cmd.stderr(std::process::Stdio::inherit());
        } else {
            cmd.stdout(std::process::Stdio::null());
            cmd.stderr(std::process::Stdio::null());
        }

        let c = cmd.spawn().map_err(|e| {
            PyRuntimeError::new_err(format!(
                "mx8.coordinator: failed to start {}: {e}",
                binary.display()
            ))
        })?;
        Some(c)
    } else {
        // Non-zero ranks have no subprocess — they just wait for rank 0's.
        None
    };

    // Every rank polls until the coordinator TCP port accepts connections.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    let mut ready = false;
    while std::time::Instant::now() < deadline {
        if std::net::TcpStream::connect_timeout(
            &connect_addr,
            std::time::Duration::from_millis(100),
        )
        .is_ok()
        {
            ready = true;
            break;
        }
        // Rank 0 can detect early process exit and give a useful error.
        if let Some(ref mut c) = child {
            if let Ok(Some(status)) = c.try_wait() {
                return Err(PyRuntimeError::new_err(format!(
                    "mx8.coordinator: coordinator process exited early \
                     (status={status}). Re-run with log=True to see output."
                )));
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    if !ready {
        if let Some(mut c) = child {
            let _ = c.kill();
            let _ = c.wait();
        }
        let rank_hint = rank.map_or(String::new(), |r| format!(" (rank {r})"));
        return Err(PyRuntimeError::new_err(format!(
            "mx8.coordinator: {connect_host}:{port} did not become ready \
             within {timeout_secs}s{rank_hint}"
        )));
    }

    Ok(CoordinatorHandle { url, child })
}
