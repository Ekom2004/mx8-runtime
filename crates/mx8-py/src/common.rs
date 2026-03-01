use super::*;

pub(crate) type TorchBatch3<'py> = (Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>);
pub(crate) type TorchBatch4<'py> = (
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
);

pub(crate) const VIDEO_LAYOUT: &str = "thwc";
pub(crate) const VIDEO_DTYPE: &str = "u8";
pub(crate) const VIDEO_COLORSPACE: &str = "rgb24";
pub(crate) const DATA_CHECKPOINT_MAGIC: &str = "mx8_checkpoint_v1";
pub(crate) const DATA_CHECKPOINT_KIND: &str = "data_loader";
pub(crate) const VIDEO_CHECKPOINT_KIND: &str = "video_loader";
pub(crate) const MIX_CHECKPOINT_KIND: &str = "mix_loader";
pub(crate) const DATA_CHECKPOINT_EPOCH: u32 = 0;

#[derive(Debug, Clone, Copy)]
pub(crate) enum VideoDecodeBackend {
    Cli,
    Ffi,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum DecodeBackend {
    Rust,
    Python,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum RustJpegCodec {
    Zune,
    Image,
    Turbo,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum RustResizeBackend {
    FastImageResize,
    Image,
}

#[pyclass]
#[derive(Debug, Clone, Default)]
pub(crate) struct Constraints {
    #[pyo3(get, set)]
    pub(crate) max_inflight_bytes: Option<u64>,
    #[pyo3(get, set)]
    pub(crate) max_process_rss_bytes: Option<u64>,
}

#[pymethods]
impl Constraints {
    #[new]
    #[pyo3(signature = (max_inflight_bytes=None, max_ram_bytes=None))]
    fn new(max_inflight_bytes: Option<u64>, max_ram_bytes: Option<u64>) -> Self {
        Self {
            max_inflight_bytes,
            max_process_rss_bytes: max_ram_bytes,
        }
    }

    #[getter]
    fn max_ram_bytes(&self) -> Option<u64> {
        self.max_process_rss_bytes
    }

    #[setter]
    fn set_max_ram_bytes(&mut self, value: Option<u64>) {
        self.max_process_rss_bytes = value;
    }
}

#[pyclass]
#[derive(Debug, Clone, Default)]
pub(crate) struct RuntimeConfig {
    #[pyo3(get, set)]
    pub(crate) prefetch_batches: Option<usize>,
    #[pyo3(get, set)]
    pub(crate) max_queue_batches: Option<usize>,
    #[pyo3(get, set)]
    pub(crate) want: Option<u32>,
}

#[pymethods]
impl RuntimeConfig {
    #[new]
    #[pyo3(signature = (prefetch_batches=None, max_queue_batches=None, want=None))]
    fn new(
        prefetch_batches: Option<usize>,
        max_queue_batches: Option<usize>,
        want: Option<u32>,
    ) -> Self {
        Self {
            prefetch_batches,
            max_queue_batches,
            want,
        }
    }
}

pub(crate) fn decode_backend_from_env() -> PyResult<DecodeBackend> {
    let raw = std::env::var("MX8_DECODE_BACKEND").unwrap_or_else(|_| "python".to_string());
    let v = raw.trim().to_ascii_lowercase();
    match v.as_str() {
        "" | "python" => Ok(DecodeBackend::Python),
        "rust" => Ok(DecodeBackend::Rust),
        _ => Err(PyValueError::new_err(format!(
            "invalid MX8_DECODE_BACKEND={raw:?} (expected: python|rust)"
        ))),
    }
}

pub(crate) fn decode_backend_name(backend: DecodeBackend) -> &'static str {
    match backend {
        DecodeBackend::Rust => "rust",
        DecodeBackend::Python => "python",
    }
}

pub(crate) fn rust_jpeg_codec_from_env() -> PyResult<RustJpegCodec> {
    let raw = std::env::var("MX8_RUST_JPEG_CODEC").unwrap_or_else(|_| "zune".to_string());
    let v = raw.trim().to_ascii_lowercase();
    match v.as_str() {
        "" | "zune" => Ok(RustJpegCodec::Zune),
        "image" => Ok(RustJpegCodec::Image),
        "turbo" | "turbojpeg" => Ok(RustJpegCodec::Turbo),
        _ => Err(PyValueError::new_err(format!(
            "invalid MX8_RUST_JPEG_CODEC={raw:?} (expected: zune|image|turbo)"
        ))),
    }
}

pub(crate) fn rust_jpeg_codec_name(codec: RustJpegCodec) -> &'static str {
    match codec {
        RustJpegCodec::Zune => "zune",
        RustJpegCodec::Image => "image",
        RustJpegCodec::Turbo => "turbo",
    }
}

pub(crate) fn rust_resize_backend_from_env() -> PyResult<RustResizeBackend> {
    let raw = std::env::var("MX8_RUST_RESIZE_BACKEND").unwrap_or_else(|_| "fast".to_string());
    let v = raw.trim().to_ascii_lowercase();
    match v.as_str() {
        "" | "fast" | "fir" | "fast_image_resize" => Ok(RustResizeBackend::FastImageResize),
        "image" => Ok(RustResizeBackend::Image),
        _ => Err(PyValueError::new_err(format!(
            "invalid MX8_RUST_RESIZE_BACKEND={raw:?} (expected: fast|image)"
        ))),
    }
}

pub(crate) fn rust_resize_backend_name(backend: RustResizeBackend) -> &'static str {
    match backend {
        RustResizeBackend::FastImageResize => "fast",
        RustResizeBackend::Image => "image",
    }
}

pub(crate) fn decode_threads_from_env() -> PyResult<usize> {
    let default_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let Some(raw) = std::env::var("MX8_DECODE_THREADS").ok() else {
        return Ok(default_threads);
    };
    let parsed = raw.trim().parse::<usize>().map_err(|_| {
        PyValueError::new_err(format!(
            "invalid MX8_DECODE_THREADS={raw:?} (expected positive integer)"
        ))
    })?;
    if parsed == 0 {
        return Err(PyValueError::new_err(
            "invalid MX8_DECODE_THREADS=0 (expected >= 1)",
        ));
    }
    Ok(parsed)
}

pub(crate) fn video_decode_backend_from_env() -> PyResult<VideoDecodeBackend> {
    let raw = std::env::var("MX8_VIDEO_DECODE_BACKEND").unwrap_or_else(|_| "cli".to_string());
    let v = raw.trim().to_ascii_lowercase();
    match v.as_str() {
        "" | "cli" | "ffmpeg" => Ok(VideoDecodeBackend::Cli),
        "ffi" | "ffmpeg_next" => Ok(VideoDecodeBackend::Ffi),
        _ => Err(PyValueError::new_err(format!(
            "invalid MX8_VIDEO_DECODE_BACKEND={raw:?} (expected: cli|ffi)"
        ))),
    }
}

pub(crate) fn video_decode_backend_name(backend: VideoDecodeBackend) -> &'static str {
    match backend {
        VideoDecodeBackend::Cli => "cli",
        VideoDecodeBackend::Ffi => "ffi",
    }
}

pub(crate) fn labels_to_torch_i64<'py>(
    py: Python<'py>,
    labels: &[u64],
) -> PyResult<Bound<'py, PyAny>> {
    let torch = py.import_bound("torch").map_err(|e| {
        PyRuntimeError::new_err(format!(
            "failed to import torch (install PyTorch to use ImageLoader): {e}"
        ))
    })?;
    let torch_int64 = torch.getattr("int64")?;
    let mut labels_i64 = Vec::with_capacity(labels.len());
    for &lab in labels {
        labels_i64.push(i64::try_from(lab).map_err(|_| {
            PyValueError::new_err(format!(
                "label_id overflow converting u64 -> i64 (label_id={lab})"
            ))
        })?);
    }
    let labels_list = PyList::new_bound(py, labels_i64);
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("dtype", &torch_int64)?;
    torch.call_method("tensor", (labels_list,), Some(&kwargs))
}

pub(crate) fn parse_pack_label_mode(s: &str) -> Result<PackLabelMode> {
    let s = s.trim().to_ascii_lowercase();
    let mode = match s.as_str() {
        "none" | "off" | "false" | "0" => PackLabelMode::None,
        "imagefolder" | "image_folder" | "image-folder" => PackLabelMode::ImageFolder,
        "auto" | "" => PackLabelMode::Auto,
        _ => anyhow::bail!("invalid label mode {s:?} (expected: auto|none|imagefolder)"),
    };
    Ok(mode)
}

pub(crate) fn parse_pack_dir_label_mode(s: &str) -> Result<PackDirLabelMode> {
    let s = s.trim().to_ascii_lowercase();
    let mode = match s.as_str() {
        "none" | "off" | "false" | "0" => PackDirLabelMode::None,
        "imagefolder" | "image_folder" | "image-folder" => PackDirLabelMode::ImageFolder,
        "auto" | "" => PackDirLabelMode::Auto,
        _ => anyhow::bail!("invalid label mode {s:?} (expected: auto|none|imagefolder)"),
    };
    Ok(mode)
}

pub(crate) fn max_ram_gb_to_bytes(max_ram_gb: Option<f64>) -> PyResult<Option<u64>> {
    let Some(max_ram_gb) = max_ram_gb else {
        return Ok(None);
    };
    if !max_ram_gb.is_finite() || max_ram_gb <= 0.0 {
        return Err(PyValueError::new_err(
            "max_ram_gb must be a finite value > 0",
        ));
    }
    let bytes = max_ram_gb * (1024f64 * 1024f64 * 1024f64);
    if bytes > u64::MAX as f64 {
        return Err(PyValueError::new_err("max_ram_gb is too large"));
    }
    Ok(Some(bytes as u64))
}

pub(crate) fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
}

/// Returns the user-local manifest store root: `$HOME/.mx8/manifests`.
/// Falls back to `/tmp/.mx8/manifests` on systems where HOME is unset
/// (e.g. some container environments). Never uses `/var/lib/...` which
/// requires root and fails silently on developer machines.
pub(crate) fn default_manifest_store() -> PathBuf {
    std::env::var_os("HOME")
        .map(|h| PathBuf::from(h).join(".mx8/manifests"))
        .unwrap_or_else(|| PathBuf::from("/tmp/.mx8/manifests"))
}

pub(crate) fn env_string(name: &str) -> Option<String> {
    std::env::var(name).ok().and_then(|v| {
        let trimmed = v.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

pub(crate) fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.trim().parse::<u64>().ok()
}

pub(crate) fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

pub(crate) fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(raw) => matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => default,
    }
}

pub(crate) fn world_size_from_env() -> u32 {
    env_string("WORLD_SIZE")
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(1)
}

pub(crate) fn rank_from_env() -> u32 {
    env_string("RANK")
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(0)
}

pub(crate) fn distributed_requested(cluster_url: Option<&str>) -> bool {
    cluster_url.is_some() || world_size_from_env() > 1
}

pub(crate) fn effective_cluster_url(cluster_url: Option<String>) -> Option<String> {
    cluster_url
        .or_else(|| env_string("MX8_CLUSTER_URL"))
        .or_else(|| env_string("MX8_COORD_URL"))
}

pub(crate) fn should_use_zero_manifest_scan(
    parsed: &mx8_core::dataset_link::DatasetLink,
    base: &str,
) -> bool {
    if !env_bool("MX8_ZERO_MANIFEST_ENABLED", true) {
        return false;
    }

    if !matches!(
        parsed,
        mx8_core::dataset_link::DatasetLink::Plain(_)
            | mx8_core::dataset_link::DatasetLink::Refresh(_)
    ) {
        return false;
    }

    let trimmed = base.trim();
    if !trimmed.starts_with("s3://") && !trimmed.starts_with("gs://") {
        return false;
    }
    let lowered = trimmed.to_ascii_lowercase();
    !lowered.ends_with(".tsv")
}

pub(crate) fn parse_u64_ascii(raw: &str) -> Option<u64> {
    raw.trim().parse::<u64>().ok()
}

pub(crate) fn parse_s3_bucket_key_local(location: &str) -> Result<(String, String)> {
    let Some(rest) = location.trim().strip_prefix("s3://") else {
        anyhow::bail!("not an s3 uri: {location}");
    };
    let rest = rest.trim().trim_start_matches('/');
    let Some((bucket, key)) = rest.split_once('/') else {
        anyhow::bail!("invalid s3 uri (missing key): {location}");
    };
    let bucket = bucket.trim();
    let key = key.trim();
    anyhow::ensure!(
        !bucket.is_empty(),
        "invalid s3 uri (empty bucket): {location}"
    );
    anyhow::ensure!(!key.is_empty(), "invalid s3 uri (empty key): {location}");
    Ok((bucket.to_string(), key.to_string()))
}

pub(crate) fn parse_gcs_bucket_key_local(location: &str) -> Result<(String, String)> {
    let Some(rest) = location.trim().strip_prefix("gs://") else {
        anyhow::bail!("not a gs uri: {location}");
    };
    let rest = rest.trim().trim_start_matches('/');
    let Some((bucket, key)) = rest.split_once('/') else {
        anyhow::bail!("invalid gs uri (missing key): {location}");
    };
    let bucket = bucket.trim();
    let key = key.trim();
    anyhow::ensure!(
        !bucket.is_empty(),
        "invalid gs uri (empty bucket): {location}"
    );
    anyhow::ensure!(!key.is_empty(), "invalid gs uri (empty key): {location}");
    Ok((bucket.to_string(), key.to_string()))
}

pub(crate) fn parse_decode_hint_fields_local(raw_hint: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    if !raw_hint.starts_with("mx8:video") {
        return out;
    }
    for token in raw_hint.split(';').skip(1) {
        let Some((k, v)) = token.split_once('=') else {
            continue;
        };
        let key = k.trim();
        let value = v.trim();
        if key.is_empty() || value.is_empty() {
            continue;
        }
        out.insert(key.to_string(), value.to_string());
    }
    out
}

pub(crate) fn parse_stage2d_sidecar_from_hint(
    record: &mx8_core::types::ManifestRecord,
) -> Result<Option<VideoRangeSidecar>> {
    let Some(raw_hint) = record.decode_hint.as_deref() else {
        return Ok(None);
    };
    let fields = parse_decode_hint_fields_local(raw_hint);
    let Some(raw_chunks) = fields.get("stage2d_chunks") else {
        return Ok(None);
    };
    let stream_id = fields
        .get("stream_id")
        .and_then(|v| v.trim().parse::<u32>().ok())
        .unwrap_or(0);
    let codec = fields
        .get("codec")
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());
    let schema_version = fields
        .get("stage2d_schema")
        .and_then(|v| v.trim().parse::<u32>().ok())
        .unwrap_or(VIDEO_RANGE_SCHEMA_VERSION);
    let mut chunks = Vec::<VideoRangeChunk>::new();
    for (idx, chunk_raw) in raw_chunks.split('|').enumerate() {
        let part = chunk_raw.trim();
        if part.is_empty() {
            continue;
        }
        let cols: Vec<&str> = part.split(',').collect();
        if cols.len() != 6 {
            anyhow::bail!(
                "invalid stage2d_chunks entry for sample_id {} at index {}",
                record.sample_id,
                idx
            );
        }
        let chunk = VideoRangeChunk {
            chunk_index: cols[0]
                .trim()
                .parse::<u32>()
                .map_err(|_| anyhow::anyhow!("invalid chunk_index"))?,
            start_ms: cols[1]
                .trim()
                .parse::<u64>()
                .map_err(|_| anyhow::anyhow!("invalid start_ms"))?,
            end_ms: cols[2]
                .trim()
                .parse::<u64>()
                .map_err(|_| anyhow::anyhow!("invalid end_ms"))?,
            start_byte: cols[3]
                .trim()
                .parse::<u64>()
                .map_err(|_| anyhow::anyhow!("invalid start_byte"))?,
            end_byte: cols[4]
                .trim()
                .parse::<u64>()
                .map_err(|_| anyhow::anyhow!("invalid end_byte"))?,
            keyframe: matches!(cols[5].trim(), "1" | "true" | "yes" | "on"),
        };
        chunks.push(chunk);
    }
    if chunks.is_empty() {
        return Ok(None);
    }
    let sidecar = VideoRangeSidecar {
        sample_id: record.sample_id,
        media_uri: record.location.clone(),
        stream_id,
        codec,
        schema_version,
        chunks,
    };
    sidecar
        .validate()
        .map_err(|e| anyhow::anyhow!("stage2d sidecar validation failed: {e}"))?;
    Ok(Some(sidecar))
}

pub(crate) fn build_stage2d_sidecar_map(
    manifest_bytes: &[u8],
) -> Result<HashMap<String, VideoRangeSidecar>> {
    let records = mx8_runtime::pipeline::load_manifest_records_from_read(std::io::Cursor::new(
        manifest_bytes,
    ))?;
    let mut out = HashMap::<String, VideoRangeSidecar>::new();
    for record in &records {
        match parse_stage2d_sidecar_from_hint(record) {
            Ok(Some(sidecar)) => {
                let k = VideoDataLoader::stage2d_sidecar_key(&sidecar.media_uri, sidecar.stream_id);
                out.insert(k, sidecar);
            }
            Ok(None) => {}
            Err(err) => {
                tracing::warn!(
                    target: "mx8_proof",
                    event = "video_stage2d_sidecar_parse_failed",
                    sample_id = record.sample_id,
                    media_uri = %record.location,
                    detail = %err,
                    "failed to parse stage2d sidecar hint"
                );
            }
        }
    }
    Ok(out)
}

pub(crate) fn detect_cgroup_memory_limit_bytes() -> Option<u64> {
    let path = std::path::Path::new("/sys/fs/cgroup/memory.max");
    let txt = std::fs::read_to_string(path).ok()?;
    let v = txt.trim();
    if v.eq_ignore_ascii_case("max") {
        return None;
    }
    let out = parse_u64_ascii(v)?;
    if out == 0 || out >= (u64::MAX / 2) {
        None
    } else {
        Some(out)
    }
}

pub(crate) fn detect_proc_memtotal_bytes() -> Option<u64> {
    let txt = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in txt.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kib = rest.split_whitespace().next()?.parse::<u64>().ok()?;
            return kib.checked_mul(1024);
        }
    }
    None
}

pub(crate) fn detect_sysctl_memsize_bytes() -> Option<u64> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_u64_ascii(std::str::from_utf8(&out.stdout).ok()?)
}

pub(crate) fn detect_node_ram_limit_bytes() -> Option<u64> {
    let cgroup = detect_cgroup_memory_limit_bytes();
    let host = detect_proc_memtotal_bytes().or_else(detect_sysctl_memsize_bytes);
    match (cgroup, host) {
        (Some(c), Some(h)) => Some(c.min(h)),
        (Some(c), None) => Some(c),
        (None, Some(h)) => Some(h),
        (None, None) => None,
    }
}

pub(crate) fn sample_process_rss_bytes_local() -> Option<u64> {
    let pid = std::process::id().to_string();
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", pid.as_str()])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let txt = std::str::from_utf8(&out.stdout).ok()?;
    let kib = txt.split_whitespace().next()?.parse::<u64>().ok()?;
    kib.checked_mul(1024)
}

pub(crate) const DEFAULT_GRPC_MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

pub(crate) fn unix_time_ms() -> u64 {
    let now = std::time::SystemTime::now();
    now.duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis()
        .min(u64::MAX as u128) as u64
}

pub(crate) fn clamp_u64_to_u32(v: u64) -> u32 {
    if v > u32::MAX as u64 {
        u32::MAX
    } else {
        v as u32
    }
}
