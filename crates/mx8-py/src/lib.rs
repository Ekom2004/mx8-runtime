#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ::image::imageops::FilterType as ImageFilterType;
use anyhow::Result;
use fast_image_resize::images::Image as FirImage;
use fast_image_resize::{
    FilterType as FirFilterType, PixelType as FirPixelType, ResizeAlg as FirResizeAlg,
    ResizeOptions as FirResizeOptions, Resizer as FirResizer,
};
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList, PyTuple};
use rayon::prelude::*;
use turbojpeg::PixelFormat as TurboPixelFormat;
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::JpegDecoder;

use mx8_manifest_store::fs::FsManifestStore;
use mx8_manifest_store::LockOwner;
use mx8_manifest_store::ManifestStore;
use mx8_proto::v0::coordinator_client::CoordinatorClient;
use mx8_proto::v0::GetManifestRequest;
use mx8_proto::v0::HeartbeatRequest;
use mx8_proto::v0::NodeCaps;
use mx8_proto::v0::NodeStats;
use mx8_proto::v0::RegisterNodeRequest;
use mx8_proto::v0::ReportProgressRequest;
use mx8_proto::v0::RequestLeaseRequest;
use mx8_runtime::pipeline::{BatchLease, Pipeline, RuntimeCaps, RuntimeMetrics};
use mx8_snapshot::labels::load_labels_for_base;
use mx8_snapshot::pack_dir::{
    pack_dir as pack_dir_impl, LabelMode as PackDirLabelMode, PackDirConfig,
};
use mx8_snapshot::pack_s3::{pack_s3, LabelMode as PackLabelMode, PackS3Config};
use mx8_snapshot::video_stage1::{
    build_video_stage1_index_from_manifest_bytes, canonicalize_video_stage1_tsv, VideoClipRecord,
    VideoStage1Config,
};
use mx8_snapshot::video_stage2d::{
    plan_video_ranges, ByteRange, RangePlannerConfig, VideoRangeChunk, VideoRangeSidecar,
    VIDEO_RANGE_SCHEMA_VERSION,
};
use mx8_snapshot::{SnapshotResolver, SnapshotResolverConfig};
use mx8_wire::TryToCore;
use tonic::transport::Channel;

#[cfg(mx8_video_ffi)]
use ffmpeg_next as ffmpeg;
#[cfg(mx8_video_ffi)]
use std::sync::OnceLock;

type TorchBatch3<'py> = (Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>);
type TorchBatch4<'py> = (
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
);

const VIDEO_LAYOUT: &str = "thwc";
const VIDEO_DTYPE: &str = "u8";
const VIDEO_COLORSPACE: &str = "rgb24";

#[derive(Debug, Clone, Copy)]
enum VideoDecodeBackend {
    Cli,
    Ffi,
}

#[derive(Debug, Clone, Copy)]
enum DecodeBackend {
    Rust,
    Python,
}

#[derive(Debug, Clone, Copy)]
enum RustJpegCodec {
    Zune,
    Image,
    Turbo,
}

#[derive(Debug, Clone, Copy)]
enum RustResizeBackend {
    FastImageResize,
    Image,
}

#[derive(Debug)]
struct DecodeBatchResult {
    nchw_u8: Vec<u8>,
    h: usize,
    w: usize,
    decode_ms: u64,
    resize_ms: u64,
    pack_ms: u64,
}

#[pyclass]
struct ImageLoader {
    loader: DataLoader,
    resize_hw: Option<(u32, u32)>,
    to_float: bool,
    decode_backend: DecodeBackend,
    rust_jpeg_codec: RustJpegCodec,
    rust_resize_backend: RustResizeBackend,
    decode_threads: usize,
    decode_pool: Option<Arc<rayon::ThreadPool>>,
    classes: Option<Vec<String>>,
}

#[pyclass]
struct DataLoader {
    dataset_base: String,
    manifest_hash: String,
    batch_size_samples: usize,
    max_inflight_bytes: u64,
    max_queue_batches: usize,
    prefetch_batches: usize,
    pipeline: Pipeline,
    metrics: Arc<RuntimeMetrics>,
    autotune: Arc<AutotuneShared>,
    rx: tokio::sync::mpsc::Receiver<BatchLease>,
    task: Option<tokio::task::JoinHandle<Result<()>>>,
    autotune_task: Option<tokio::task::JoinHandle<()>>,
    rt: tokio::runtime::Runtime,
}

#[pyclass]
struct MixedDataLoader {
    loaders: Vec<Py<DataLoader>>,
    scheduler: WeightedRoundRobin,
    active: Vec<bool>,
    source_exhausted_total: Vec<u64>,
    delivered_batches: Vec<u64>,
    delivered_samples: Vec<u64>,
    delivered_bytes: Vec<u64>,
    starvation_total: Vec<u64>,
    steps_since_emit: Vec<u64>,
    max_starvation_window: u64,
    normalized_weights: Vec<u64>,
    shared_max_inflight_bytes: u64,
    shared_inflight_violation_total: u64,
    seed: u64,
    epoch: u64,
    schedule_ticks: u64,
    snapshot_enabled: bool,
    snapshot_period_ticks: u64,
    snapshot_emitted_total: u64,
    source_exhaustion_policy: SourceExhaustionPolicy,
    mix_profile: Option<String>,
    mix_autotune_enabled: bool,
    mix_effective_prefetch_batches: usize,
    mix_effective_max_queue_batches: usize,
    mix_runtime_autotune_enabled: bool,
    mix_autotune_rails: Option<AutotuneRails>,
    mix_autotune_controller: AutotuneController,
    mix_autotune_current: AutotuneUpdate,
    mix_runtime_autotune_period_ticks: u64,
    mix_runtime_autotune_last_tick: u64,
    mix_runtime_autotune_adjustments_total: u64,
    mix_runtime_autotune_pressure_milli: u64,
    mix_max_process_rss_bytes: Option<u64>,
}

#[pyclass]
struct VideoBatch {
    sample_ids: Vec<u64>,
    clip_ids: Vec<String>,
    media_uris: Vec<String>,
    clip_starts: Vec<u64>,
    offsets: Vec<u64>,
    payload: Vec<u8>,
    frames_per_clip: u32,
    frame_height: u32,
    frame_width: u32,
    channels: u32,
    stride_t: u64,
    stride_h: u64,
    stride_w: u64,
    stride_c: u64,
}

#[pyclass]
struct VideoDataLoader {
    manifest_hash: String,
    clips: Vec<VideoClipRecord>,
    stage2d_sidecars: HashMap<String, VideoRangeSidecar>,
    rt: tokio::runtime::Runtime,
    decode_backend: VideoDecodeBackend,
    next_idx: usize,
    batch_size_samples: usize,
    max_inflight_bytes: u64,
    bytes_per_clip: usize,
    decode_contract: VideoDecodeContract,
    seed: u64,
    epoch: u64,
    clip_len: u32,
    stride: u32,
    decode_fps: u32,
    fps_policy: String,
    delivered_batches: u64,
    delivered_samples: u64,
    delivered_bytes: u64,
    decode_attempted_clips: u64,
    decode_succeeded_clips: u64,
    decode_failed_io_read_failed: u64,
    decode_failed_corrupt_media: u64,
    decode_failed_short_media: u64,
    decode_failed_unsupported_codec: u64,
    decode_failed_missing_stream: u64,
    decode_failed_backend_unavailable: u64,
    decode_failed_decode_failed: u64,
    decode_ms_total: u64,
    s3_range_requests_total: u64,
    s3_range_bytes_fetched_total: u64,
    s3_stage2d_plan_used_total: u64,
    s3_stage2d_plan_fallback_total: u64,
    s3_full_object_range_fallback_total: u64,
    video_runtime_autotune_enabled: bool,
    video_runtime_autotune_period_batches: u64,
    video_runtime_autotune_last_batch: u64,
    video_runtime_autotune_adjustments_total: u64,
    video_runtime_autotune_pressure_milli: u64,
    video_max_process_rss_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
struct VideoDecodeContract {
    frames_per_clip: u32,
    frame_height: u32,
    frame_width: u32,
    channels: u32,
    stride_t: u64,
    stride_h: u64,
    stride_w: u64,
    stride_c: u64,
    clip_bytes: u64,
}

#[derive(Debug)]
struct VideoDecodeError {
    class: &'static str,
    path: PathBuf,
    detail: String,
}

#[derive(Debug, Clone)]
struct WeightedRoundRobin {
    weights: Vec<u64>,
    current: Vec<i128>,
    tie_break_offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceExhaustionPolicy {
    Error,
    Allow,
}

impl SourceExhaustionPolicy {
    fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "error" => Ok(Self::Error),
            "allow" => Ok(Self::Allow),
            _ => anyhow::bail!("invalid source_exhausted={raw:?} (expected: error|allow)"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Allow => "allow",
        }
    }
}

impl WeightedRoundRobin {
    fn new(weights: Vec<u64>, seed: u64, epoch: u64) -> Self {
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

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn normalize_mix_weights(raw: &[f64]) -> Result<Vec<u64>> {
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

fn compute_shared_mix_cap(max_inflight_bytes: &[u64]) -> Result<u64> {
    let Some(min_cap) = max_inflight_bytes.iter().copied().min() else {
        anyhow::bail!("mix requires at least one loader");
    };
    if min_cap == 0 {
        anyhow::bail!("mix loader max_inflight_bytes must be > 0");
    }
    Ok(min_cap)
}

fn should_emit_mix_snapshot(schedule_ticks: u64, snapshot_period_ticks: u64) -> bool {
    if snapshot_period_ticks == 0 || schedule_ticks == 0 {
        return false;
    }
    schedule_ticks.is_multiple_of(snapshot_period_ticks)
}

#[pyclass]
#[derive(Debug, Clone, Default)]
struct Constraints {
    #[pyo3(get, set)]
    max_inflight_bytes: Option<u64>,
    #[pyo3(get, set)]
    max_process_rss_bytes: Option<u64>,
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
struct RuntimeConfig {
    #[pyo3(get, set)]
    prefetch_batches: Option<usize>,
    #[pyo3(get, set)]
    max_queue_batches: Option<usize>,
    #[pyo3(get, set)]
    want: Option<u32>,
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

fn metrics_to_dict<'py>(py: Python<'py>, metrics: &RuntimeMetrics) -> PyResult<Bound<'py, PyDict>> {
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

fn decode_backend_from_env() -> PyResult<DecodeBackend> {
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

fn decode_backend_name(backend: DecodeBackend) -> &'static str {
    match backend {
        DecodeBackend::Rust => "rust",
        DecodeBackend::Python => "python",
    }
}

fn rust_jpeg_codec_from_env() -> PyResult<RustJpegCodec> {
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

fn rust_jpeg_codec_name(codec: RustJpegCodec) -> &'static str {
    match codec {
        RustJpegCodec::Zune => "zune",
        RustJpegCodec::Image => "image",
        RustJpegCodec::Turbo => "turbo",
    }
}

fn rust_resize_backend_from_env() -> PyResult<RustResizeBackend> {
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

fn rust_resize_backend_name(backend: RustResizeBackend) -> &'static str {
    match backend {
        RustResizeBackend::FastImageResize => "fast",
        RustResizeBackend::Image => "image",
    }
}

fn decode_threads_from_env() -> PyResult<usize> {
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

fn video_decode_backend_from_env() -> PyResult<VideoDecodeBackend> {
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

fn video_decode_backend_name(backend: VideoDecodeBackend) -> &'static str {
    match backend {
        VideoDecodeBackend::Cli => "cli",
        VideoDecodeBackend::Ffi => "ffi",
    }
}

fn labels_to_torch_i64<'py>(py: Python<'py>, labels: &[u64]) -> PyResult<Bound<'py, PyAny>> {
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

fn decode_images_nchw_u8(
    lease: &BatchLease,
    resize_hw: Option<(u32, u32)>,
    rust_jpeg_codec: RustJpegCodec,
    rust_resize_backend: RustResizeBackend,
    decode_threads: usize,
    decode_pool: Option<&rayon::ThreadPool>,
) -> PyResult<DecodeBatchResult> {
    struct DecodedRgb {
        sample_id: u64,
        h: usize,
        w: usize,
        rgb_u8: Vec<u8>,
    }

    struct DecodeOneResult {
        image: DecodedRgb,
        decode_us: u64,
        resize_us: u64,
    }

    fn elapsed_us_u64(started: Instant) -> u64 {
        let micros = started.elapsed().as_micros();
        if micros > u128::from(u64::MAX) {
            u64::MAX
        } else {
            micros as u64
        }
    }

    fn decode_one_image(
        sample_id: u64,
        bytes: &[u8],
        resize_hw: Option<(u32, u32)>,
        rust_jpeg_codec: RustJpegCodec,
        rust_resize_backend: RustResizeBackend,
    ) -> Result<DecodeOneResult, String> {
        fn looks_like_jpeg(bytes: &[u8]) -> bool {
            bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xD8
        }

        fn decode_rgb_with_image(
            bytes: &[u8],
            sample_id: u64,
        ) -> Result<(Vec<u8>, u32, u32), String> {
            let decoded = ::image::load_from_memory(bytes)
                .map_err(|e| format!("decode failed for sample_id {sample_id}: {e}"))?;
            let rgb = decoded.to_rgb8();
            let width = rgb.width();
            let height = rgb.height();
            Ok((rgb.into_raw(), width, height))
        }

        fn decode_jpeg_rgb_with_zune(
            bytes: &[u8],
            sample_id: u64,
        ) -> Result<(Vec<u8>, u32, u32), String> {
            let cursor = ZCursor::new(bytes);
            let mut decoder = JpegDecoder::new(cursor);
            let pixels = decoder
                .decode()
                .map_err(|e| format!("zune decode failed for sample_id {sample_id}: {e}"))?;
            let info = decoder.info().ok_or_else(|| {
                format!("zune decode missing image info for sample_id {sample_id}")
            })?;
            let width = u32::from(info.width);
            let height = u32::from(info.height);
            let expected_len = usize::try_from(width)
                .ok()
                .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
                .and_then(|px| px.checked_mul(3))
                .ok_or_else(|| format!("zune decoded shape overflow for sample_id {sample_id}"))?;
            if pixels.len() != expected_len {
                return Err(format!(
                    "zune decoded unexpected channel count for sample_id {sample_id} (len={}, expected_rgb_len={expected_len})",
                    pixels.len()
                ));
            }
            Ok((pixels, width, height))
        }

        fn decode_jpeg_rgb_with_turbo(
            bytes: &[u8],
            sample_id: u64,
        ) -> Result<(Vec<u8>, u32, u32), String> {
            let decoded = turbojpeg::decompress(bytes, TurboPixelFormat::RGB)
                .map_err(|e| format!("turbojpeg decode failed for sample_id {sample_id}: {e}"))?;
            let width_u32 = u32::try_from(decoded.width)
                .map_err(|_| format!("turbojpeg width overflow for sample_id {sample_id}"))?;
            let height_u32 = u32::try_from(decoded.height)
                .map_err(|_| format!("turbojpeg height overflow for sample_id {sample_id}"))?;
            let row_bytes = decoded
                .width
                .checked_mul(3)
                .ok_or_else(|| format!("turbojpeg row size overflow for sample_id {sample_id}"))?;
            let expected_len = decoded
                .width
                .checked_mul(decoded.height)
                .and_then(|pixels| pixels.checked_mul(3))
                .ok_or_else(|| {
                    format!("turbojpeg decoded shape overflow for sample_id {sample_id}")
                })?;

            if decoded.pitch == row_bytes && decoded.pixels.len() == expected_len {
                return Ok((decoded.pixels, width_u32, height_u32));
            }

            if decoded.pitch < row_bytes {
                return Err(format!(
                    "turbojpeg invalid pitch for sample_id {sample_id} (pitch={} row_bytes={row_bytes})",
                    decoded.pitch
                ));
            }
            let required_len = decoded.height.checked_mul(decoded.pitch).ok_or_else(|| {
                format!("turbojpeg pitch buffer overflow for sample_id {sample_id}")
            })?;
            if decoded.pixels.len() < required_len {
                return Err(format!(
                    "turbojpeg decoded buffer too small for sample_id {sample_id} (len={}, required={required_len})",
                    decoded.pixels.len()
                ));
            }

            let mut compact = vec![0u8; expected_len];
            for y in 0..decoded.height {
                let src_start = y.checked_mul(decoded.pitch).ok_or_else(|| {
                    format!("turbojpeg src offset overflow for sample_id {sample_id}")
                })?;
                let src_end = src_start.checked_add(row_bytes).ok_or_else(|| {
                    format!("turbojpeg src end overflow for sample_id {sample_id}")
                })?;
                let dst_start = y.checked_mul(row_bytes).ok_or_else(|| {
                    format!("turbojpeg dst offset overflow for sample_id {sample_id}")
                })?;
                let dst_end = dst_start.checked_add(row_bytes).ok_or_else(|| {
                    format!("turbojpeg dst end overflow for sample_id {sample_id}")
                })?;
                compact[dst_start..dst_end].copy_from_slice(&decoded.pixels[src_start..src_end]);
            }
            Ok((compact, width_u32, height_u32))
        }

        let decode_started = Instant::now();
        let (raw_rgb, width, height) = match rust_jpeg_codec {
            RustJpegCodec::Zune if looks_like_jpeg(bytes) => {
                decode_jpeg_rgb_with_zune(bytes, sample_id)?
            }
            RustJpegCodec::Turbo if looks_like_jpeg(bytes) => {
                decode_jpeg_rgb_with_turbo(bytes, sample_id)?
            }
            _ => decode_rgb_with_image(bytes, sample_id)?,
        };
        let decode_us = elapsed_us_u64(decode_started);

        let (raw_rgb, width, height, resize_us) = if let Some((h, w)) = resize_hw {
            let resize_started = Instant::now();
            let resized = match rust_resize_backend {
                RustResizeBackend::FastImageResize => {
                    let src_image =
                        FirImage::from_vec_u8(width, height, raw_rgb, FirPixelType::U8x3).map_err(
                            |e| {
                                format!(
                                    "fast resize source init failed for sample_id {sample_id}: {e}"
                                )
                            },
                        )?;
                    let mut dst_image = FirImage::new(w, h, FirPixelType::U8x3);
                    let mut resizer = FirResizer::new();
                    let resize_options = FirResizeOptions::new()
                        .resize_alg(FirResizeAlg::Convolution(FirFilterType::Bilinear));
                    resizer
                        .resize(&src_image, &mut dst_image, &resize_options)
                        .map_err(|e| {
                            format!("fast resize failed for sample_id {sample_id}: {e}")
                        })?;
                    dst_image.into_vec()
                }
                RustResizeBackend::Image => {
                    let rgb =
                        ::image::RgbImage::from_raw(width, height, raw_rgb).ok_or_else(|| {
                            format!("decoded rgb buffer shape mismatch for sample_id {sample_id}")
                        })?;
                    let resized = ::image::imageops::resize(&rgb, w, h, ImageFilterType::Triangle);
                    resized.into_raw()
                }
            };
            (resized, w, h, elapsed_us_u64(resize_started))
        } else {
            (raw_rgb, width, height, 0)
        };

        let h =
            usize::try_from(height).map_err(|_| "image height does not fit usize".to_string())?;
        let w = usize::try_from(width).map_err(|_| "image width does not fit usize".to_string())?;
        let rgb_len = h
            .checked_mul(w)
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or_else(|| "decoded rgb size overflow".to_string())?;
        if raw_rgb.len() != rgb_len {
            return Err(format!(
                "decoded rgb length mismatch for sample_id {sample_id} (len={}, expected={rgb_len})",
                raw_rgb.len()
            ));
        }
        Ok(DecodeOneResult {
            image: DecodedRgb {
                sample_id,
                h,
                w,
                rgb_u8: raw_rgb,
            },
            decode_us,
            resize_us,
        })
    }

    let sample_ids = lease.batch.sample_ids.clone();
    let offsets = lease.batch.offsets.clone();
    let payload = lease.batch.payload.clone();
    let sample_count = sample_ids.len();
    if sample_count == 0 {
        return Err(PyRuntimeError::new_err("empty image batch"));
    }

    let decode_us_total = Arc::new(AtomicU64::new(0));
    let resize_us_total = Arc::new(AtomicU64::new(0));

    let decode_at = |i: usize| -> Result<DecodedRgb, String> {
        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;
        if end < start || end > payload.len() {
            return Err(format!(
                "bad offsets for sample_id {} (start={} end={} payload_len={})",
                sample_ids[i],
                start,
                end,
                payload.len()
            ));
        }
        let one = decode_one_image(
            sample_ids[i],
            &payload[start..end],
            resize_hw,
            rust_jpeg_codec,
            rust_resize_backend,
        )?;
        decode_us_total.fetch_add(one.decode_us, Ordering::Relaxed);
        resize_us_total.fetch_add(one.resize_us, Ordering::Relaxed);
        Ok(one.image)
    };

    let decoded: Vec<Result<DecodedRgb, String>> =
        if decode_threads > 1 && sample_count > 1 && decode_pool.is_some() {
            decode_pool
                .ok_or_else(|| PyRuntimeError::new_err("decode thread pool unavailable"))?
                .install(|| (0..sample_count).into_par_iter().map(decode_at).collect())
        } else {
            (0..sample_count).map(decode_at).collect()
        };

    let mut images = Vec::with_capacity(sample_count);
    let mut first_h: Option<usize> = None;
    let mut first_w: Option<usize> = None;
    for maybe_image in decoded {
        let image = maybe_image.map_err(PyRuntimeError::new_err)?;

        match (first_h, first_w) {
            (None, None) => {
                first_h = Some(image.h);
                first_w = Some(image.w);
            }
            (Some(h0), Some(w0)) if h0 == image.h && w0 == image.w => {}
            (Some(h0), Some(w0)) => {
                return Err(PyValueError::new_err(format!(
                    "decoded image shape mismatch in batch (sample_id={} got={}x{}, expected={}x{}); set resize_hw for variable-size inputs",
                    image.sample_id, image.h, image.w, h0, w0
                )));
            }
            _ => {}
        }
        images.push(image);
    }

    let h = first_h.ok_or_else(|| PyRuntimeError::new_err("empty image batch"))?;
    let w = first_w.ok_or_else(|| PyRuntimeError::new_err("empty image batch"))?;
    let pixels_per_image = h
        .checked_mul(w)
        .ok_or_else(|| PyRuntimeError::new_err("pixels_per_image overflow"))?;
    let bytes_per_image = pixels_per_image
        .checked_mul(3)
        .ok_or_else(|| PyRuntimeError::new_err("bytes_per_image overflow"))?;
    let total_bytes = sample_count
        .checked_mul(bytes_per_image)
        .ok_or_else(|| PyRuntimeError::new_err("decoded batch byte size overflow"))?;
    let mut out = vec![0u8; total_bytes];

    let pack_started = Instant::now();
    for (image_index, image) in images.iter().enumerate() {
        let image_base = image_index
            .checked_mul(bytes_per_image)
            .ok_or_else(|| PyRuntimeError::new_err("decoded batch index overflow"))?;
        let (c0, tail) =
            out[image_base..image_base + bytes_per_image].split_at_mut(pixels_per_image);
        let (c1, c2) = tail.split_at_mut(pixels_per_image);
        for pixel_idx in 0..pixels_per_image {
            let src = pixel_idx
                .checked_mul(3)
                .ok_or_else(|| PyRuntimeError::new_err("decoded pixel index overflow"))?;
            c0[pixel_idx] = image.rgb_u8[src];
            c1[pixel_idx] = image.rgb_u8[src + 1];
            c2[pixel_idx] = image.rgb_u8[src + 2];
        }
    }

    Ok(DecodeBatchResult {
        nchw_u8: out,
        h,
        w,
        decode_ms: decode_us_total.load(Ordering::Relaxed) / 1000,
        resize_ms: resize_us_total.load(Ordering::Relaxed) / 1000,
        pack_ms: elapsed_us_u64(pack_started) / 1000,
    })
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
        node_id=None,
        profile=None,
        autotune=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
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
        node_id: Option<String>,
        profile: Option<String>,
        autotune: Option<bool>,
    ) -> PyResult<Self> {
        let parsed = mx8_core::dataset_link::DatasetLink::parse(&dataset_link)
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
            .unwrap_or_else(|| PathBuf::from("/var/lib/mx8/manifests"));

        let dev_manifest_path = manifest_path.or(env_path("MX8_DEV_MANIFEST_PATH"));

        let max_process_rss_bytes_cap =
            max_ram_bytes.or_else(|| env_u64("MX8_MAX_PROCESS_RSS_BYTES"));
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
            Some(AutotuneProfile::from_name(profile.as_deref()))
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
            Some(rt.spawn(autotune_loop(
                Arc::new(pipeline.clone()),
                metrics.clone(),
                autotune.clone(),
                max_inflight_bytes,
                max_process_rss_bytes_cap,
                rails.unwrap_or(AutotuneRails::for_profile(AutotuneProfile::Balanced)),
            )))
        } else {
            None
        };

        if should_use_zero_manifest_scan(&parsed, &dataset_base) {
            let reservoir_size = env_usize("MX8_ZERO_MANIFEST_RESERVOIR", 100_000).max(1);
            match rt.block_on(async {
                pipeline
                    .spawn_s3_scan(&dataset_base, recursive, reservoir_size, start_id, end_id)
                    .await
            }) {
                Ok((rx, task)) => {
                    tracing::info!(
                        target: "mx8_proof",
                        event = "zero_manifest_scan_enabled",
                        dataset_base = %dataset_base,
                        recursive = recursive,
                        reservoir_size = reservoir_size as u64,
                        "using zero-manifest s3 scan path"
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
                        max_queue_batches: initial_max_queue,
                        prefetch_batches: initial_prefetch,
                        pipeline,
                        metrics,
                        autotune,
                        rx,
                        task: Some(task),
                        autotune_task,
                        rt,
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

        let pipeline_for_stream = pipeline.clone();
        let manifest_bytes = snapshot.manifest_bytes;
        let (rx, task) = rt
            .block_on(async move {
                match (start_id, end_id) {
                    (Some(s), Some(e)) => {
                        pipeline_for_stream
                            .spawn_manifest_bytes_range_stream(manifest_bytes, s, e)
                            .await
                    }
                    (None, None) => {
                        pipeline_for_stream
                            .spawn_manifest_bytes_stream(manifest_bytes)
                            .await
                    }
                    _ => anyhow::bail!("start_id and end_id must be set together"),
                }
            })
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;

        Ok(Self {
            dataset_base,
            manifest_hash: snapshot.manifest_hash.0,
            batch_size_samples,
            max_inflight_bytes,
            max_queue_batches: initial_max_queue,
            prefetch_batches: initial_prefetch,
            pipeline,
            metrics,
            autotune,
            rx,
            task: Some(task),
            autotune_task,
            rt,
        })
    }

    #[getter]
    fn manifest_hash(&self) -> &str {
        &self.manifest_hash
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let out = metrics_to_dict(py, self.metrics.as_ref())?;
        out.set_item("batch_size_samples", self.batch_size_samples)?;
        out.set_item("max_inflight_bytes", self.max_inflight_bytes)?;
        out.set_item("max_queue_batches", self.max_queue_batches)?;
        out.set_item("prefetch_batches", self.prefetch_batches)?;
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
        Ok(out.into_any())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let wait_started = Instant::now();
        let lease = py.allow_threads(|| self.rt.block_on(async { self.rx.recv().await }));
        let out = match lease {
            Some(lease) => {
                let out = Py::new(py, PyBatch { lease })?;
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

    fn close(&mut self) {
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

impl MixedDataLoader {
    fn total_inflight_bytes(&self, py: Python<'_>) -> u64 {
        self.loaders
            .iter()
            .map(|ldr| ldr.borrow(py).metrics.inflight_bytes.get())
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
            .map(|ldr| ldr.borrow(py).metrics.process_rss_bytes.get())
            .max()
            .unwrap_or(0)
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
            let mut guard = loader.borrow_mut(py);
            guard.apply_runtime_overrides(
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

impl DataLoader {
    fn apply_runtime_overrides(
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

            let next_item = {
                let mut loader = self.loaders[source_idx].borrow_mut(py);
                loader.__next__(py)
            };

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
            "mix_realized_ratio",
            PyList::new_bound(py, self.realized_ratio()),
        )?;

        let source_stats = PyList::empty_bound(py);
        let realized = self.realized_ratio();
        for (source_idx, loader) in self.loaders.iter().enumerate() {
            let guard = loader.borrow(py);
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
            source.set_item("dataset_base", guard.dataset_base.as_str())?;
            source.set_item("manifest_hash", guard.manifest_hash.as_str())?;
            source.set_item("batch_size_samples", guard.batch_size_samples)?;
            source.set_item("max_inflight_bytes", guard.max_inflight_bytes)?;
            source.set_item("max_queue_batches", guard.max_queue_batches)?;
            source.set_item("prefetch_batches", guard.prefetch_batches)?;
            source.set_item("metrics", metrics_to_dict(py, guard.metrics.as_ref())?)?;
            source_stats.append(source)?;
        }
        out.set_item("mix_sources", source_stats)?;
        Ok(out.into_any())
    }
}

#[pymethods]
impl VideoBatch {
    #[getter]
    fn sample_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        Ok(PyList::new_bound(py, self.sample_ids.iter().copied()))
    }

    #[getter]
    fn clip_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        Ok(PyList::new_bound(py, self.clip_ids.iter().cloned()))
    }

    #[getter]
    fn media_uris<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        Ok(PyList::new_bound(py, self.media_uris.iter().cloned()))
    }

    #[getter]
    fn clip_starts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        Ok(PyList::new_bound(py, self.clip_starts.iter().copied()))
    }

    #[getter]
    fn offsets<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        Ok(PyList::new_bound(py, self.offsets.iter().copied()))
    }

    #[getter]
    fn payload<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new_bound(py, &self.payload))
    }

    #[getter]
    fn frames_per_clip(&self) -> u32 {
        self.frames_per_clip
    }

    #[getter]
    fn frame_height(&self) -> u32 {
        self.frame_height
    }

    #[getter]
    fn frame_width(&self) -> u32 {
        self.frame_width
    }

    #[getter]
    fn channels(&self) -> u32 {
        self.channels
    }

    #[getter]
    fn layout(&self) -> &'static str {
        VIDEO_LAYOUT
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        VIDEO_DTYPE
    }

    #[getter]
    fn colorspace(&self) -> &'static str {
        VIDEO_COLORSPACE
    }

    #[getter]
    fn strides<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        Ok(PyList::new_bound(
            py,
            [self.stride_t, self.stride_h, self.stride_w, self.stride_c],
        ))
    }
}

impl VideoDataLoader {
    fn decode_failed_total(&self) -> u64 {
        self.decode_failed_io_read_failed
            .saturating_add(self.decode_failed_corrupt_media)
            .saturating_add(self.decode_failed_short_media)
            .saturating_add(self.decode_failed_unsupported_codec)
            .saturating_add(self.decode_failed_missing_stream)
            .saturating_add(self.decode_failed_backend_unavailable)
            .saturating_add(self.decode_failed_decode_failed)
    }

    fn maybe_run_runtime_autotune(&mut self) {
        if !self.video_runtime_autotune_enabled {
            return;
        }
        let period = self.video_runtime_autotune_period_batches.max(1);
        if self
            .delivered_batches
            .saturating_sub(self.video_runtime_autotune_last_batch)
            < period
        {
            return;
        }
        self.video_runtime_autotune_last_batch = self.delivered_batches;

        let Some(max_process_rss_bytes) = self.video_max_process_rss_bytes else {
            return;
        };
        if max_process_rss_bytes == 0 {
            return;
        }
        let rss_bytes = sample_process_rss_bytes_local().unwrap_or(0);
        let rss_ratio = (rss_bytes as f64 / max_process_rss_bytes as f64).clamp(0.0, 2.0);
        let inflight_ratio =
            (self.max_inflight_bytes as f64 / max_process_rss_bytes as f64).clamp(0.0, 2.0);
        let pressure = rss_ratio.max(inflight_ratio);
        self.video_runtime_autotune_pressure_milli = (pressure * 1000.0).round() as u64;

        let min_inflight = (self.batch_size_samples as u64)
            .saturating_mul(self.decode_contract.clip_bytes)
            .max(1);
        let max_inflight = max_process_rss_bytes.max(min_inflight);
        let old_inflight = self.max_inflight_bytes;
        let mut next_inflight = old_inflight;
        if rss_ratio > 0.92 {
            next_inflight = ((old_inflight as f64) * 0.85).round() as u64;
        } else if rss_ratio < 0.60 {
            next_inflight = ((old_inflight as f64) * 1.05).round() as u64;
        }
        next_inflight = next_inflight.clamp(min_inflight, max_inflight);
        if next_inflight != old_inflight {
            self.max_inflight_bytes = next_inflight;
            self.video_runtime_autotune_adjustments_total = self
                .video_runtime_autotune_adjustments_total
                .saturating_add(1);
            tracing::info!(
                target: "mx8_proof",
                event = "video_runtime_autotune_adjustment",
                pressure = pressure,
                rss_ratio = rss_ratio,
                max_process_rss_bytes = max_process_rss_bytes,
                old_max_inflight_bytes = old_inflight,
                new_max_inflight_bytes = next_inflight,
                adjustments_total = self.video_runtime_autotune_adjustments_total,
                "video runtime autotune adjusted max inflight"
            );
        }
    }

    fn derive_decode_contract(
        clip_len: u32,
        bytes_per_clip: usize,
    ) -> PyResult<VideoDecodeContract> {
        let clip_len_usize = usize::try_from(clip_len).map_err(|_| {
            PyRuntimeError::new_err(format!("clip_len conversion overflow: {clip_len}"))
        })?;
        if clip_len_usize == 0 {
            return Err(PyRuntimeError::new_err(
                "clip_len must be > 0 for decode sizing",
            ));
        }
        let bytes_per_frame = std::cmp::max(1, bytes_per_clip / clip_len_usize);
        let pixels_per_frame = std::cmp::max(1, bytes_per_frame / 3);
        let side = std::cmp::max(1, (pixels_per_frame as f64).sqrt().floor() as usize);
        let side_u32 = u32::try_from(side)
            .map_err(|_| PyRuntimeError::new_err(format!("video decode side too large: {side}")))?;
        let frame_bytes = u64::from(side_u32)
            .checked_mul(u64::from(side_u32))
            .and_then(|v| v.checked_mul(3))
            .ok_or_else(|| PyRuntimeError::new_err("video decode frame byte size overflow"))?;
        let clip_bytes = frame_bytes
            .checked_mul(u64::from(clip_len))
            .ok_or_else(|| PyRuntimeError::new_err("video decode clip byte size overflow"))?;
        let stride_h = u64::from(side_u32)
            .checked_mul(3)
            .ok_or_else(|| PyRuntimeError::new_err("video decode stride_h overflow"))?;
        Ok(VideoDecodeContract {
            frames_per_clip: clip_len,
            frame_height: side_u32,
            frame_width: side_u32,
            channels: 3,
            stride_t: frame_bytes,
            stride_h,
            stride_w: 3,
            stride_c: 1,
            clip_bytes,
        })
    }

    fn decode_error(path: &std::path::Path, class: &'static str, detail: &str) -> VideoDecodeError {
        VideoDecodeError {
            class,
            path: path.to_path_buf(),
            detail: detail.to_string(),
        }
    }

    fn classify_ffmpeg_failure(stderr: &str) -> &'static str {
        let lower = stderr.to_ascii_lowercase();
        if lower.contains("no such file or directory") {
            return "io_read_failed";
        }
        if lower.contains("invalid data found when processing input")
            || lower.contains("moov atom not found")
            || lower.contains("error reading header")
        {
            return "corrupt_media";
        }
        if lower.contains("matches no streams")
            || lower.contains("stream specifier")
            || lower.contains("output file #0 does not contain any stream")
        {
            return "missing_stream";
        }
        if lower.contains("unsupported codec")
            || lower.contains("unknown codec")
            || lower.contains("decoder")
        {
            return "unsupported_codec";
        }
        if lower.contains("end of file") {
            return "short_media";
        }
        "decode_failed"
    }

    fn bump_decode_failure(&mut self, class: &str) {
        match class {
            "io_read_failed" => {
                self.decode_failed_io_read_failed =
                    self.decode_failed_io_read_failed.saturating_add(1);
            }
            "corrupt_media" => {
                self.decode_failed_corrupt_media =
                    self.decode_failed_corrupt_media.saturating_add(1);
            }
            "short_media" => {
                self.decode_failed_short_media = self.decode_failed_short_media.saturating_add(1);
            }
            "unsupported_codec" => {
                self.decode_failed_unsupported_codec =
                    self.decode_failed_unsupported_codec.saturating_add(1);
            }
            "missing_stream" => {
                self.decode_failed_missing_stream =
                    self.decode_failed_missing_stream.saturating_add(1);
            }
            "decode_backend_unavailable" => {
                self.decode_failed_backend_unavailable =
                    self.decode_failed_backend_unavailable.saturating_add(1);
            }
            _ => {
                self.decode_failed_decode_failed =
                    self.decode_failed_decode_failed.saturating_add(1);
            }
        }
    }

    fn decode_clip_from_path_cli(
        &self,
        path: &std::path::Path,
        start_seconds: f64,
    ) -> Result<Vec<u8>, VideoDecodeError> {
        let side = self.decode_contract.frame_width;
        let clip_len_usize =
            usize::try_from(self.decode_contract.frames_per_clip).map_err(|_| {
                Self::decode_error(
                    path,
                    "decode_failed",
                    "clip_len conversion overflow for decode contract",
                )
            })?;
        let expected_clip_bytes =
            usize::try_from(self.decode_contract.clip_bytes).map_err(|_| {
                Self::decode_error(path, "decode_failed", "clip_bytes conversion overflow")
            })?;
        let seek_arg = format!("{start_seconds:.6}");
        let scale_arg = format!("scale={side}:{side}:flags=bilinear");
        let ffmpeg_bin = std::env::var("MX8_FFMPEG_BIN").unwrap_or_else(|_| "ffmpeg".to_string());

        let output = Command::new(&ffmpeg_bin)
            .arg("-hide_banner")
            .arg("-loglevel")
            .arg("error")
            .arg("-nostdin")
            .arg("-ss")
            .arg(seek_arg)
            .arg("-i")
            .arg(path)
            .arg("-an")
            .arg("-sn")
            .arg("-dn")
            .arg("-frames:v")
            .arg(clip_len_usize.to_string())
            .arg("-vf")
            .arg(scale_arg)
            .arg("-pix_fmt")
            .arg("rgb24")
            .arg("-f")
            .arg("rawvideo")
            .arg("pipe:1")
            .output()
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => Self::decode_error(
                    path,
                    "decode_backend_unavailable",
                    &format!("ffmpeg binary not found: {ffmpeg_bin}"),
                ),
                _ => Self::decode_error(path, "io_read_failed", &e.to_string()),
            })?;

        if !output.status.success() {
            let stderr_text = String::from_utf8_lossy(&output.stderr);
            let class = Self::classify_ffmpeg_failure(&stderr_text);
            let detail = stderr_text
                .lines()
                .find(|line| !line.trim().is_empty())
                .map(str::trim)
                .unwrap_or("ffmpeg decode failed");
            return Err(Self::decode_error(path, class, detail));
        }

        let mut decoded = output.stdout;
        if decoded.len() < expected_clip_bytes {
            return Err(Self::decode_error(
                path,
                "short_media",
                &format!(
                    "decoded bytes {} below expected {}",
                    decoded.len(),
                    expected_clip_bytes
                ),
            ));
        }
        if decoded.len() > expected_clip_bytes {
            decoded.truncate(expected_clip_bytes);
        }
        Ok(decoded)
    }

    #[cfg(mx8_video_ffi)]
    fn decode_clip_from_path_ffi(
        &self,
        path: &std::path::Path,
        start_seconds: f64,
    ) -> Result<Vec<u8>, VideoDecodeError> {
        static FFMPEG_INIT: OnceLock<std::result::Result<(), String>> = OnceLock::new();
        let init = FFMPEG_INIT.get_or_init(|| ffmpeg::init().map_err(|e| e.to_string()));
        if let Err(err) = init {
            return Err(Self::decode_error(
                path,
                "decode_backend_unavailable",
                &format!("ffmpeg-next init failed: {err}"),
            ));
        }

        let mut input_ctx = ffmpeg::format::input(path).map_err(|e| {
            Self::decode_error(path, "io_read_failed", &format!("open input failed: {e}"))
        })?;
        let stream = input_ctx
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| Self::decode_error(path, "missing_stream", "no video stream found"))?;
        let stream_index = stream.index();
        let codec_ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| {
                Self::decode_error(path, "decode_failed", &format!("codec context failed: {e}"))
            })?;
        let mut decoder = codec_ctx.decoder().video().map_err(|e| {
            Self::decode_error(
                path,
                "unsupported_codec",
                &format!("video decoder init failed: {e}"),
            )
        })?;

        let side = self.decode_contract.frame_width;
        let side_usize = usize::try_from(side).map_err(|_| {
            Self::decode_error(path, "decode_failed", "frame width conversion overflow")
        })?;
        let clip_len_usize =
            usize::try_from(self.decode_contract.frames_per_clip).map_err(|_| {
                Self::decode_error(
                    path,
                    "decode_failed",
                    "clip_len conversion overflow for decode contract",
                )
            })?;
        let expected_clip_bytes =
            usize::try_from(self.decode_contract.clip_bytes).map_err(|_| {
                Self::decode_error(path, "decode_failed", "clip_bytes conversion overflow")
            })?;
        let start_frame_index = if start_seconds <= 0.0 {
            0usize
        } else {
            (start_seconds * f64::from(self.decode_fps.max(1))).floor() as usize
        };

        let mut scaler = ffmpeg::software::scaling::context::Context::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            ffmpeg::format::Pixel::RGB24,
            side,
            side,
            ffmpeg::software::scaling::flag::Flags::BILINEAR,
        )
        .map_err(|e| {
            Self::decode_error(
                path,
                "decode_failed",
                &format!("scale context init failed: {e}"),
            )
        })?;

        let mut out = Vec::<u8>::with_capacity(expected_clip_bytes);
        let mut decoded = ffmpeg::util::frame::video::Video::empty();
        let mut rgb = ffmpeg::util::frame::video::Video::empty();
        let mut seen_frames: usize = 0;
        let mut kept_frames: usize = 0;

        let mut push_frame = |src: &ffmpeg::util::frame::video::Video,
                              seen_frames: &mut usize,
                              kept_frames: &mut usize,
                              out: &mut Vec<u8>|
         -> Result<()> {
            if *seen_frames < start_frame_index {
                *seen_frames = seen_frames.saturating_add(1);
                return Ok(());
            }
            if *kept_frames >= clip_len_usize {
                return Ok(());
            }
            scaler
                .run(src, &mut rgb)
                .map_err(|e| anyhow::anyhow!("scale frame failed: {e}"))?;
            let stride = rgb.stride(0);
            let data = rgb.data(0);
            let row_bytes = side_usize.saturating_mul(3);
            for row in 0..side_usize {
                let start = row
                    .checked_mul(stride)
                    .ok_or_else(|| anyhow::anyhow!("rgb row offset overflow"))?;
                let end = start
                    .checked_add(row_bytes)
                    .ok_or_else(|| anyhow::anyhow!("rgb row end overflow"))?;
                if end > data.len() {
                    anyhow::bail!("scaled frame buffer too small");
                }
                out.extend_from_slice(&data[start..end]);
            }
            *seen_frames = seen_frames.saturating_add(1);
            *kept_frames = kept_frames.saturating_add(1);
            Ok(())
        };

        for (pkt_stream, packet) in input_ctx.packets() {
            if pkt_stream.index() != stream_index {
                continue;
            }
            decoder.send_packet(&packet).map_err(|e| {
                Self::decode_error(path, "decode_failed", &format!("send packet failed: {e}"))
            })?;
            while decoder.receive_frame(&mut decoded).is_ok() {
                push_frame(&decoded, &mut seen_frames, &mut kept_frames, &mut out)
                    .map_err(|e| Self::decode_error(path, "decode_failed", &e.to_string()))?;
                if kept_frames >= clip_len_usize {
                    break;
                }
            }
            if kept_frames >= clip_len_usize {
                break;
            }
        }

        decoder.send_eof().map_err(|e| {
            Self::decode_error(path, "decode_failed", &format!("send eof failed: {e}"))
        })?;
        while kept_frames < clip_len_usize && decoder.receive_frame(&mut decoded).is_ok() {
            push_frame(&decoded, &mut seen_frames, &mut kept_frames, &mut out)
                .map_err(|e| Self::decode_error(path, "decode_failed", &e.to_string()))?;
        }

        if out.len() < expected_clip_bytes {
            return Err(Self::decode_error(
                path,
                "short_media",
                &format!(
                    "decoded bytes {} below expected {}",
                    out.len(),
                    expected_clip_bytes
                ),
            ));
        }
        if out.len() > expected_clip_bytes {
            out.truncate(expected_clip_bytes);
        }
        Ok(out)
    }

    #[cfg(not(mx8_video_ffi))]
    fn decode_clip_from_path_ffi(
        &self,
        path: &std::path::Path,
        _start_seconds: f64,
    ) -> Result<Vec<u8>, VideoDecodeError> {
        Err(Self::decode_error(
            path,
            "decode_backend_unavailable",
            "ffi backend not compiled (rebuild with RUSTFLAGS='--cfg mx8_video_ffi')",
        ))
    }

    fn decode_clip_from_path(
        &self,
        path: &std::path::Path,
        start_seconds: f64,
    ) -> Result<Vec<u8>, VideoDecodeError> {
        match self.decode_backend {
            VideoDecodeBackend::Cli => self.decode_clip_from_path_cli(path, start_seconds),
            VideoDecodeBackend::Ffi => match self.decode_clip_from_path_ffi(path, start_seconds) {
                Ok(v) => Ok(v),
                Err(err) => {
                    tracing::warn!(
                        target: "mx8_proof",
                        event = "video_decode_backend_fallback",
                        from_backend = "ffi",
                        to_backend = "cli",
                        class = err.class,
                        path = %path.display(),
                        detail = %err.detail,
                        "ffi decode failed; falling back to cli backend"
                    );
                    self.decode_clip_from_path_cli(path, start_seconds)
                }
            },
        }
    }

    fn stage2d_sidecar_key(media_uri: &str, stream_id: u32) -> String {
        format!("{media_uri}#{stream_id}")
    }

    fn s3_fetch_ranges_bytes(
        &mut self,
        clip: &VideoClipRecord,
        bucket: &str,
        key: &str,
        ranges: &[ByteRange],
    ) -> Result<Vec<u8>, VideoDecodeError> {
        let client = self
            .rt
            .block_on(mx8_runtime::s3::client_from_env())
            .map_err(|e| {
                Self::decode_error(
                    std::path::Path::new(&clip.media_uri),
                    "io_read_failed",
                    &format!("init s3 client failed: {e}"),
                )
            })?;

        let mut payload = Vec::<u8>::new();
        for range in ranges {
            let end_inclusive = range.end_byte.checked_sub(1).ok_or_else(|| {
                Self::decode_error(
                    std::path::Path::new(&clip.media_uri),
                    "decode_failed",
                    "invalid empty stage2d byte range",
                )
            })?;
            let range_header = format!("bytes={}-{}", range.start_byte, end_inclusive);
            let out = self
                .rt
                .block_on(async {
                    client
                        .get_object()
                        .bucket(bucket)
                        .key(key)
                        .range(range_header)
                        .send()
                        .await
                })
                .map_err(|e| {
                    Self::decode_error(
                        std::path::Path::new(&clip.media_uri),
                        "io_read_failed",
                        &format!("s3 range get failed: {e:?}"),
                    )
                })?;
            let bytes = self
                .rt
                .block_on(async {
                    out.body
                        .collect()
                        .await
                        .map(|c| c.into_bytes().to_vec())
                        .map_err(|e| e.to_string())
                })
                .map_err(|e| {
                    Self::decode_error(
                        std::path::Path::new(&clip.media_uri),
                        "io_read_failed",
                        &format!("s3 range body collect failed: {e}"),
                    )
                })?;
            let expected = usize::try_from(range.end_byte.saturating_sub(range.start_byte))
                .map_err(|_| {
                    Self::decode_error(
                        std::path::Path::new(&clip.media_uri),
                        "decode_failed",
                        "stage2d range size conversion overflow",
                    )
                })?;
            if bytes.len() != expected {
                return Err(Self::decode_error(
                    std::path::Path::new(&clip.media_uri),
                    "io_read_failed",
                    &format!(
                        "s3 range returned {} bytes, expected {}",
                        bytes.len(),
                        expected
                    ),
                ));
            }
            payload.extend_from_slice(&bytes);
            self.s3_range_requests_total = self.s3_range_requests_total.saturating_add(1);
            self.s3_range_bytes_fetched_total = self
                .s3_range_bytes_fetched_total
                .saturating_add(bytes.len() as u64);
        }
        tracing::info!(
            target: "mx8_proof",
            event = "video_range_fetch",
            media_uri = %clip.media_uri,
            clip_id = %clip.clip_id,
            request_count = ranges.len() as u64,
            fetched_bytes = payload.len() as u64,
            "fetched s3 ranges for video clip"
        );
        Ok(payload)
    }

    fn clip_payload_bytes_s3(
        &mut self,
        clip: &VideoClipRecord,
    ) -> Result<Vec<u8>, VideoDecodeError> {
        let (bucket, key) = parse_s3_bucket_key_local(&clip.media_uri).map_err(|e| {
            Self::decode_error(
                std::path::Path::new(&clip.media_uri),
                "decode_failed",
                &format!("invalid s3 uri: {e}"),
            )
        })?;
        let sidecar_key = Self::stage2d_sidecar_key(&clip.media_uri, clip.stream_id);
        let clip_start_ms = clip
            .clip_start
            .saturating_mul(1000)
            .saturating_div(u64::from(self.decode_fps.max(1)));
        let clip_len_ms = u64::from(clip.clip_len.max(1))
            .saturating_mul(1000)
            .saturating_div(u64::from(self.decode_fps.max(1)));

        let planner_cfg = RangePlannerConfig {
            max_ranges: env_usize("MX8_VIDEO_STAGE2D_MAX_RANGES", 8).max(1),
            merge_gap_bytes: env_u64("MX8_VIDEO_STAGE2D_MERGE_GAP_BYTES").unwrap_or(0),
        };

        if let Some(sidecar) = self.stage2d_sidecars.get(&sidecar_key) {
            match plan_video_ranges(sidecar, clip_start_ms, clip_len_ms.max(1), planner_cfg) {
                Ok(plan) => {
                    self.s3_stage2d_plan_used_total =
                        self.s3_stage2d_plan_used_total.saturating_add(1);
                    tracing::info!(
                        target: "mx8_proof",
                        event = "video_range_plan",
                        media_uri = %clip.media_uri,
                        clip_id = %clip.clip_id,
                        range_count = plan.ranges.len() as u64,
                        planned_bytes = plan.planned_bytes,
                        anchor_ms = plan.anchor_ms,
                        clip_start_ms = clip_start_ms,
                        clip_len_ms = clip_len_ms,
                        "planned stage2d s3 ranges for video clip"
                    );
                    if let Ok(bytes) = self.s3_fetch_ranges_bytes(clip, &bucket, &key, &plan.ranges)
                    {
                        let tmp_path = std::env::temp_dir().join(format!(
                            "mx8-video-stage2d-{}-{}-{}.mp4",
                            std::process::id(),
                            unix_time_ms(),
                            clip.sample_id
                        ));
                        std::fs::write(&tmp_path, &bytes).map_err(|e| {
                            Self::decode_error(
                                std::path::Path::new(&clip.media_uri),
                                "io_read_failed",
                                &format!("write stage2d temp file failed: {e}"),
                            )
                        })?;
                        let seek_seconds =
                            clip_start_ms.saturating_sub(plan.anchor_ms) as f64 / 1000.0;
                        let decoded = self.decode_clip_from_path(&tmp_path, seek_seconds);
                        let _ = std::fs::remove_file(&tmp_path);
                        if decoded.is_ok() {
                            return decoded;
                        }
                        self.s3_stage2d_plan_fallback_total =
                            self.s3_stage2d_plan_fallback_total.saturating_add(1);
                        tracing::warn!(
                            target: "mx8_proof",
                            event = "video_range_fallback",
                            reason = "stage2d_decode_failed",
                            media_uri = %clip.media_uri,
                            clip_id = %clip.clip_id,
                            "stage2d range decode failed; falling back to full-object range"
                        );
                    } else {
                        self.s3_stage2d_plan_fallback_total =
                            self.s3_stage2d_plan_fallback_total.saturating_add(1);
                        tracing::warn!(
                            target: "mx8_proof",
                            event = "video_range_fallback",
                            reason = "stage2d_fetch_failed",
                            media_uri = %clip.media_uri,
                            clip_id = %clip.clip_id,
                            "stage2d range fetch failed; falling back to full-object range"
                        );
                    }
                }
                Err(err) => {
                    self.s3_stage2d_plan_fallback_total =
                        self.s3_stage2d_plan_fallback_total.saturating_add(1);
                    tracing::warn!(
                        target: "mx8_proof",
                        event = "video_range_fallback",
                        reason = "stage2d_plan_failed",
                        media_uri = %clip.media_uri,
                        clip_id = %clip.clip_id,
                        detail = %err,
                        "stage2d range planning failed; falling back to full-object range"
                    );
                }
            }
        }

        self.s3_full_object_range_fallback_total =
            self.s3_full_object_range_fallback_total.saturating_add(1);
        let client = self
            .rt
            .block_on(mx8_runtime::s3::client_from_env())
            .map_err(|e| {
                Self::decode_error(
                    std::path::Path::new(&clip.media_uri),
                    "io_read_failed",
                    &format!("init s3 client failed: {e}"),
                )
            })?;
        let object_len = self
            .rt
            .block_on(async { client.head_object().bucket(&bucket).key(&key).send().await })
            .map_err(|e| {
                Self::decode_error(
                    std::path::Path::new(&clip.media_uri),
                    "io_read_failed",
                    &format!("s3 head_object failed: {e:?}"),
                )
            })?
            .content_length()
            .unwrap_or(0);
        if object_len <= 0 {
            return Err(Self::decode_error(
                std::path::Path::new(&clip.media_uri),
                "io_read_failed",
                "s3 object has unknown/zero size",
            ));
        }
        let full_range = vec![ByteRange {
            start_byte: 0,
            end_byte: object_len as u64,
        }];
        tracing::info!(
            target: "mx8_proof",
            event = "video_range_plan",
            media_uri = %clip.media_uri,
            clip_id = %clip.clip_id,
            range_count = 1u64,
            planned_bytes = object_len as u64,
            reason = "full_object_range_fallback",
            "using full-object s3 range fallback for video clip"
        );
        let bytes = self.s3_fetch_ranges_bytes(clip, &bucket, &key, &full_range)?;
        let tmp_path = std::env::temp_dir().join(format!(
            "mx8-video-s3-full-{}-{}-{}.mp4",
            std::process::id(),
            unix_time_ms(),
            clip.sample_id
        ));
        std::fs::write(&tmp_path, &bytes).map_err(|e| {
            Self::decode_error(
                std::path::Path::new(&clip.media_uri),
                "io_read_failed",
                &format!("write s3 temp file failed: {e}"),
            )
        })?;
        let seek_seconds = (clip.clip_start as f64) / f64::from(self.decode_fps.max(1));
        let decoded = self.decode_clip_from_path(&tmp_path, seek_seconds);
        let _ = std::fs::remove_file(&tmp_path);
        decoded
    }

    fn clip_payload_bytes(&mut self, clip: &VideoClipRecord) -> Result<Vec<u8>, VideoDecodeError> {
        if clip.media_uri.starts_with("s3://") {
            return self.clip_payload_bytes_s3(clip);
        }
        let path = std::path::PathBuf::from(&clip.media_uri);
        if !path.exists() {
            return Err(Self::decode_error(
                &path,
                "io_read_failed",
                "input path does not exist",
            ));
        }
        let seek_seconds = (clip.clip_start as f64) / f64::from(self.decode_fps.max(1));
        self.decode_clip_from_path(&path, seek_seconds)
    }
}

impl VideoDecodeError {
    fn into_pyerr(self) -> PyErr {
        PyRuntimeError::new_err(format!(
            "video decode {} for {}: {}",
            self.class,
            self.path.display(),
            self.detail
        ))
    }
}

#[pymethods]
impl VideoDataLoader {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.next_idx >= self.clips.len() {
            return Err(PyStopIteration::new_err(()));
        }
        let batch_started = Instant::now();
        let end_idx = self
            .next_idx
            .saturating_add(self.batch_size_samples)
            .min(self.clips.len());
        let mut sample_ids = Vec::with_capacity(end_idx.saturating_sub(self.next_idx));
        let mut clip_ids = Vec::with_capacity(end_idx.saturating_sub(self.next_idx));
        let mut media_uris = Vec::with_capacity(end_idx.saturating_sub(self.next_idx));
        let mut clip_starts = Vec::with_capacity(end_idx.saturating_sub(self.next_idx));
        let mut offsets =
            Vec::with_capacity(end_idx.saturating_sub(self.next_idx).saturating_add(1));
        offsets.push(0);
        let mut payload = Vec::<u8>::new();

        for idx in self.next_idx..end_idx {
            let clip_record = self.clips[idx].clone();
            let sample_id = clip_record.sample_id;
            let clip_id_for_log = clip_record.clip_id.clone();
            let media_uri_for_log = clip_record.media_uri.clone();
            let clip_start_for_log = clip_record.clip_start;
            self.decode_attempted_clips = self.decode_attempted_clips.saturating_add(1);
            let decode_started = Instant::now();
            let clip_payload = match self.clip_payload_bytes(&clip_record) {
                Ok(v) => v,
                Err(err) => {
                    self.bump_decode_failure(err.class);
                    tracing::warn!(
                        target: "mx8_proof",
                        event = "video_decode_failed",
                        class = err.class,
                        media_uri = %media_uri_for_log,
                        clip_id = %clip_id_for_log,
                        clip_start = clip_start_for_log,
                        decode_attempted_clips_total = self.decode_attempted_clips,
                        decode_failed_total = self.decode_failed_total(),
                        detail = %err.detail,
                        "video decode failed"
                    );
                    return Err(err.into_pyerr());
                }
            };
            let decode_ms = decode_started
                .elapsed()
                .as_millis()
                .min(u128::from(u64::MAX)) as u64;
            self.decode_ms_total = self.decode_ms_total.saturating_add(decode_ms);
            self.decode_succeeded_clips = self.decode_succeeded_clips.saturating_add(1);
            payload.extend_from_slice(&clip_payload);
            offsets.push(payload.len() as u64);
            sample_ids.push(sample_id);
            clip_ids.push(clip_id_for_log);
            media_uris.push(media_uri_for_log);
            clip_starts.push(clip_start_for_log);
        }

        let payload_bytes = payload.len() as u64;
        if payload_bytes > self.max_inflight_bytes {
            return Err(PyRuntimeError::new_err(format!(
                "video batch payload {} exceeds max_inflight_bytes {}",
                payload_bytes, self.max_inflight_bytes
            )));
        }

        self.next_idx = end_idx;
        self.delivered_batches = self.delivered_batches.saturating_add(1);
        self.delivered_samples = self
            .delivered_samples
            .saturating_add(sample_ids.len() as u64);
        self.delivered_bytes = self.delivered_bytes.saturating_add(payload_bytes);
        self.maybe_run_runtime_autotune();
        let batch_decode_ms = batch_started
            .elapsed()
            .as_millis()
            .min(u128::from(u64::MAX)) as u64;
        tracing::info!(
            target: "mx8_proof",
            event = "video_decode_batch",
            decode_backend = video_decode_backend_name(self.decode_backend),
            batch_samples = sample_ids.len() as u64,
            payload_bytes = payload_bytes,
            batch_decode_ms = batch_decode_ms,
            decode_attempted_clips_total = self.decode_attempted_clips,
            decode_succeeded_clips_total = self.decode_succeeded_clips,
            decode_failed_total = self.decode_failed_total(),
            "video decode batch delivered"
        );

        let out = Py::new(
            py,
            VideoBatch {
                sample_ids,
                clip_ids,
                media_uris,
                clip_starts,
                offsets,
                payload,
                frames_per_clip: self.decode_contract.frames_per_clip,
                frame_height: self.decode_contract.frame_height,
                frame_width: self.decode_contract.frame_width,
                channels: self.decode_contract.channels,
                stride_t: self.decode_contract.stride_t,
                stride_h: self.decode_contract.stride_h,
                stride_w: self.decode_contract.stride_w,
                stride_c: self.decode_contract.stride_c,
            },
        )?;
        Ok(out.into_bound(py).into_any())
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let out = PyDict::new_bound(py);
        out.set_item("manifest_hash", &self.manifest_hash)?;
        out.set_item("seed", self.seed)?;
        out.set_item("epoch", self.epoch)?;
        out.set_item("clip_len", self.clip_len)?;
        out.set_item("stride", self.stride)?;
        out.set_item("fps_policy", &self.fps_policy)?;
        out.set_item("video_layout", VIDEO_LAYOUT)?;
        out.set_item("video_dtype", VIDEO_DTYPE)?;
        out.set_item("video_colorspace", VIDEO_COLORSPACE)?;
        out.set_item(
            "video_decode_backend",
            video_decode_backend_name(self.decode_backend),
        )?;
        out.set_item(
            "video_frames_per_clip",
            self.decode_contract.frames_per_clip,
        )?;
        out.set_item("video_frame_height", self.decode_contract.frame_height)?;
        out.set_item("video_frame_width", self.decode_contract.frame_width)?;
        out.set_item("video_channels", self.decode_contract.channels)?;
        out.set_item("video_stride_t", self.decode_contract.stride_t)?;
        out.set_item("video_stride_h", self.decode_contract.stride_h)?;
        out.set_item("video_stride_w", self.decode_contract.stride_w)?;
        out.set_item("video_stride_c", self.decode_contract.stride_c)?;
        out.set_item("video_clip_bytes", self.decode_contract.clip_bytes)?;
        out.set_item("bytes_per_clip", self.bytes_per_clip as u64)?;
        out.set_item("max_inflight_bytes", self.max_inflight_bytes)?;
        out.set_item(
            "video_runtime_autotune_enabled",
            self.video_runtime_autotune_enabled,
        )?;
        out.set_item(
            "video_runtime_autotune_pressure",
            self.video_runtime_autotune_pressure_milli as f64 / 1000.0,
        )?;
        out.set_item(
            "video_runtime_autotune_adjustments_total",
            self.video_runtime_autotune_adjustments_total,
        )?;
        out.set_item("clips_total", self.clips.len() as u64)?;
        out.set_item(
            "clips_remaining",
            (self.clips.len().saturating_sub(self.next_idx)) as u64,
        )?;
        out.set_item("video_delivered_batches_total", self.delivered_batches)?;
        out.set_item("video_delivered_samples_total", self.delivered_samples)?;
        out.set_item("video_delivered_bytes_total", self.delivered_bytes)?;
        out.set_item(
            "video_decode_attempted_clips_total",
            self.decode_attempted_clips,
        )?;
        out.set_item(
            "video_decode_succeeded_clips_total",
            self.decode_succeeded_clips,
        )?;
        out.set_item(
            "video_decode_failed_io_read_failed_total",
            self.decode_failed_io_read_failed,
        )?;
        out.set_item(
            "video_decode_failed_corrupt_media_total",
            self.decode_failed_corrupt_media,
        )?;
        out.set_item(
            "video_decode_failed_short_media_total",
            self.decode_failed_short_media,
        )?;
        out.set_item(
            "video_decode_failed_unsupported_codec_total",
            self.decode_failed_unsupported_codec,
        )?;
        out.set_item(
            "video_decode_failed_missing_stream_total",
            self.decode_failed_missing_stream,
        )?;
        out.set_item(
            "video_decode_failed_backend_unavailable_total",
            self.decode_failed_backend_unavailable,
        )?;
        out.set_item(
            "video_decode_failed_decode_failed_total",
            self.decode_failed_decode_failed,
        )?;
        let decode_failed_total = self.decode_failed_total();
        out.set_item("video_decode_failed_total", decode_failed_total)?;
        out.set_item("video_decode_ms_total", self.decode_ms_total)?;
        out.set_item(
            "video_stage2d_sidecars_total",
            self.stage2d_sidecars.len() as u64,
        )?;
        out.set_item(
            "video_s3_range_requests_total",
            self.s3_range_requests_total,
        )?;
        out.set_item(
            "video_s3_range_bytes_fetched_total",
            self.s3_range_bytes_fetched_total,
        )?;
        out.set_item(
            "video_s3_stage2d_plan_used_total",
            self.s3_stage2d_plan_used_total,
        )?;
        out.set_item(
            "video_s3_stage2d_plan_fallback_total",
            self.s3_stage2d_plan_fallback_total,
        )?;
        out.set_item(
            "video_s3_full_object_range_fallback_total",
            self.s3_full_object_range_fallback_total,
        )?;
        Ok(out.into_any())
    }
}

impl ImageLoader {
    fn next_python_decode<'py>(
        &self,
        py: Python<'py>,
        lease: BatchLease,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sample_count = lease.batch.sample_count();
        let started = std::time::Instant::now();
        let labels = lease.batch.label_ids.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "batch has no label_ids (expected mx8:vision:imagefolder label hints in manifest)",
            )
        })?;

        let torch = py.import_bound("torch").map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to import torch (install PyTorch to use ImageLoader): {e}"
            ))
        })?;
        let np = py.import_bound("numpy").map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to import numpy (install numpy to use ImageLoader): {e}"
            ))
        })?;
        let pil = py.import_bound("PIL.Image").map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to import PIL.Image (install Pillow to use ImageLoader): {e}"
            ))
        })?;
        let io = py.import_bound("io")?;

        let bytes_io = io.getattr("BytesIO")?;
        let image_open = pil.getattr("open")?;

        let mut xs: Vec<Bound<'py, PyAny>> = Vec::with_capacity(lease.batch.sample_count());

        for i in 0..lease.batch.sample_count() {
            let start = lease.batch.offsets[i] as usize;
            let end = lease.batch.offsets[i + 1] as usize;
            if end < start || end > lease.batch.payload.len() {
                return Err(PyRuntimeError::new_err(format!(
                    "bad offsets for sample_id {} (start={} end={} payload_len={})",
                    lease.batch.sample_ids[i],
                    start,
                    end,
                    lease.batch.payload.len()
                )));
            }

            let b = PyBytes::new_bound(py, &lease.batch.payload[start..end]);
            let bio = bytes_io.call1((b,))?;

            let mut img = image_open
                .call1((bio,))?
                .call_method1("convert", ("RGB",))?;
            if let Some((h, w)) = self.resize_hw {
                img = img.call_method1("resize", ((w, h),))?;
            }

            let kwargs = PyDict::new_bound(py);
            kwargs.set_item("dtype", np.getattr("uint8")?)?;
            kwargs.set_item("copy", true)?;
            let arr = np.call_method("array", (img,), Some(&kwargs))?;

            let x = torch.getattr("from_numpy")?.call1((arr,))?;
            let x = x
                .call_method1("permute", (2i64, 0i64, 1i64))?
                .call_method0("contiguous")?;
            let x = if self.to_float {
                x.call_method0("float")?.call_method1("div", (255.0f32,))?
            } else {
                x
            };
            xs.push(x);
        }

        let xs_list = PyList::new_bound(py, xs);
        let images = torch.getattr("stack")?.call1((xs_list, 0i64))?;
        let labels = labels_to_torch_i64(py, labels)?;
        let out = PyTuple::new_bound(py, [images.to_object(py), labels.to_object(py)]);
        let elapsed_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        tracing::debug!(
            target: "mx8_proof",
            event = "vision_decode_batch",
            backend = "python",
            samples = sample_count as u64,
            elapsed_ms = elapsed_ms,
            "vision decode batch"
        );
        Ok(out.into_any())
    }

    fn next_rust_decode<'py>(
        &self,
        py: Python<'py>,
        lease: BatchLease,
    ) -> PyResult<Bound<'py, PyAny>> {
        let sample_count = lease.batch.sample_count();
        let started = std::time::Instant::now();
        let labels = lease.batch.label_ids.as_ref().ok_or_else(|| {
            PyValueError::new_err(
                "batch has no label_ids (expected mx8:vision:imagefolder label hints in manifest)",
            )
        })?;

        let decode_result = py.allow_threads(|| {
            decode_images_nchw_u8(
                &lease,
                self.resize_hw,
                self.rust_jpeg_codec,
                self.rust_resize_backend,
                self.decode_threads,
                self.decode_pool.as_deref(),
            )
        })?;

        let torch = py.import_bound("torch").map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to import torch (install PyTorch to use ImageLoader): {e}"
            ))
        })?;
        let torch_uint8 = torch.getattr("uint8")?;

        let payload = PyByteArray::new_bound(py, &decode_result.nchw_u8);
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("dtype", &torch_uint8)?;
        let images = torch.call_method("frombuffer", (payload,), Some(&kwargs))?;

        let b_i64 = i64::try_from(lease.batch.sample_count())
            .map_err(|_| PyValueError::new_err("batch size does not fit i64"))?;
        let h_i64 = i64::try_from(decode_result.h)
            .map_err(|_| PyValueError::new_err("height does not fit i64"))?;
        let w_i64 = i64::try_from(decode_result.w)
            .map_err(|_| PyValueError::new_err("width does not fit i64"))?;

        let images = images
            .call_method1("view", (b_i64, 3i64, h_i64, w_i64))?
            .call_method0("contiguous")?;
        let images = if self.to_float {
            images
                .call_method0("float")?
                .call_method1("div", (255.0f32,))?
        } else {
            images
        };

        let labels = labels_to_torch_i64(py, labels)?;
        let out = PyTuple::new_bound(py, [images.to_object(py), labels.to_object(py)]);
        let elapsed_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        tracing::debug!(
            target: "mx8_proof",
            event = "vision_decode_batch",
            backend = "rust",
            samples = sample_count as u64,
            elapsed_ms = elapsed_ms,
            decode_ms = decode_result.decode_ms,
            resize_ms = decode_result.resize_ms,
            pack_ms = decode_result.pack_ms,
            decode_threads = self.decode_threads as u64,
            rust_jpeg_codec = rust_jpeg_codec_name(self.rust_jpeg_codec),
            rust_resize_backend = rust_resize_backend_name(self.rust_resize_backend),
            "vision decode batch"
        );
        Ok(out.into_any())
    }
}

#[pymethods]
impl ImageLoader {
    #[new]
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
        max_ram_bytes=None,
        start_id=None,
        end_id=None,
        node_id=None,
        profile=None,
        autotune=None,
        resize_hw=None,
        to_float=true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
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
        node_id: Option<String>,
        profile: Option<String>,
        autotune: Option<bool>,
        resize_hw: Option<(u32, u32)>,
        to_float: bool,
    ) -> PyResult<Self> {
        let loader = DataLoader::new(
            dataset_link,
            manifest_store,
            manifest_path,
            recursive,
            batch_size_samples,
            max_inflight_bytes,
            max_queue_batches,
            prefetch_batches,
            target_batch_bytes,
            max_batch_bytes,
            max_ram_bytes,
            start_id,
            end_id,
            node_id,
            profile,
            autotune,
        )?;
        let base = loader.dataset_base.clone();
        let classes = loader
            .rt
            .block_on(async { load_labels_for_base(&base).await })
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
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
        tracing::info!(
            target: "mx8_proof",
            event = "vision_decode_backend_selected",
            backend = decode_backend_name(decode_backend),
            decode_threads = decode_threads,
            rust_jpeg_codec = rust_jpeg_codec_name(rust_jpeg_codec),
            rust_resize_backend = rust_resize_backend_name(rust_resize_backend),
            "vision decode backend selected"
        );
        Ok(Self {
            loader,
            resize_hw,
            to_float,
            decode_backend,
            rust_jpeg_codec,
            rust_resize_backend,
            decode_threads,
            decode_pool,
            classes,
        })
    }

    #[getter]
    fn classes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.classes {
            Some(v) => Ok(PyList::new_bound(py, v).into_any()),
            None => Ok(py.None().into_bound(py).into_any()),
        }
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.loader.stats(py)
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let lease = py.allow_threads(|| {
            self.loader
                .rt
                .block_on(async { self.loader.rx.recv().await })
        });

        let Some(lease) = lease else {
            let Some(task) = self.loader.task.take() else {
                return Err(PyStopIteration::new_err(()));
            };
            let out = py.allow_threads(|| self.loader.rt.block_on(task));
            return match out {
                Ok(Ok(())) => Err(PyStopIteration::new_err(())),
                Ok(Err(err)) => Err(PyRuntimeError::new_err(format!("{err}"))),
                Err(err) => Err(PyRuntimeError::new_err(format!(
                    "producer task failed: {err}"
                ))),
            };
        };

        match self.decode_backend {
            DecodeBackend::Rust => self.next_rust_decode(py, lease),
            DecodeBackend::Python => self.next_python_decode(py, lease),
        }
    }
}

#[pyclass]
struct PyBatch {
    lease: BatchLease,
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

#[pyfunction]
#[pyo3(signature = (dataset_link, *, manifest_store=None, manifest_path=None, recursive=true, node_id=None))]
fn resolve_manifest_hash(
    dataset_link: String,
    manifest_store: Option<PathBuf>,
    manifest_path: Option<PathBuf>,
    recursive: bool,
    node_id: Option<String>,
) -> PyResult<String> {
    let root = manifest_store
        .or(env_path("MX8_MANIFEST_STORE_ROOT"))
        .unwrap_or_else(|| PathBuf::from("/var/lib/mx8/manifests"));
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
fn internal_video_index_build<'py>(
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
        .unwrap_or_else(|| PathBuf::from("/var/lib/mx8/manifests"));
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
fn internal_video_index_replay_check<'py>(
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
#[pyo3(signature = (pack_in, *, out, shard_mb=512, label_mode=None, require_labels=false))]
fn pack<'py>(
    py: Python<'py>,
    pack_in: String,
    out: String,
    shard_mb: u64,
    label_mode: Option<String>,
    require_labels: bool,
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
fn pack_dir<'py>(
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

#[pyfunction]
#[pyo3(name = "image")]
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
    node_id=None,
    max_ram_gb=None,
    profile=None,
    autotune=None,
    constraints=None,
    runtime=None,
))]
#[allow(clippy::too_many_arguments)]
fn load(
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
    node_id: Option<String>,
    max_ram_gb: Option<f64>,
    profile: Option<String>,
    autotune: Option<bool>,
    constraints: Option<Py<Constraints>>,
    runtime: Option<Py<RuntimeConfig>>,
) -> PyResult<Py<DataLoader>> {
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

        if autotune_enabled && effective_max_process_rss_bytes.is_none() {
            if let Some(node_limit) = detect_node_ram_limit_bytes() {
                let profile_fraction = match selected_profile {
                    AutotuneProfile::Safe => 0.60f64,
                    AutotuneProfile::Balanced => 0.75f64,
                    AutotuneProfile::Throughput => 0.85f64,
                };
                let reserve_bytes = 1u64 << 30;
                let base_rss = sample_process_rss_bytes_local().unwrap_or(0);
                let mut derived = ((node_limit as f64) * profile_fraction) as u64;
                derived = derived.saturating_sub(reserve_bytes);
                let min_required = base_rss.saturating_add(effective_max_inflight_bytes);
                if derived < min_required {
                    derived = min_required;
                }
                effective_max_process_rss_bytes = Some(derived.max(effective_max_inflight_bytes));
                tracing::info!(
                    target: "mx8_proof",
                    event = "autotune_startup_caps_selected",
                    mode = "single_node",
                    profile = match selected_profile {
                        AutotuneProfile::Safe => "safe",
                        AutotuneProfile::Balanced => "balanced",
                        AutotuneProfile::Throughput => "throughput",
                    },
                    max_inflight_bytes = effective_max_inflight_bytes,
                    max_process_rss_bytes = effective_max_process_rss_bytes.unwrap_or(0),
                    max_queue_batches = effective_max_queue_batches as u64,
                    prefetch_batches = effective_prefetch_batches as u64,
                    "v1 profile/autotune startup caps resolved"
                );
            }
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
        node_id,
        profile.clone(),
        autotune,
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
    batch_size_samples=32,
    max_inflight_bytes=128*1024*1024,
    max_queue_batches=64,
    prefetch_batches=1,
    target_batch_bytes=None,
    max_batch_bytes=None,
    start_id=None,
    end_id=None,
    node_id=None,
    max_ram_gb=None,
    profile=None,
    autotune=None,
    constraints=None,
    runtime=None,
    resize_hw=None,
    to_float=true,
))]
#[allow(clippy::too_many_arguments)]
fn py_image(
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
    node_id: Option<String>,
    max_ram_gb: Option<f64>,
    profile: Option<String>,
    autotune: Option<bool>,
    constraints: Option<Py<Constraints>>,
    runtime: Option<Py<RuntimeConfig>>,
    resize_hw: Option<(u32, u32)>,
    to_float: bool,
) -> PyResult<Py<ImageLoader>> {
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

        if let Some(max_process) = effective_max_process_rss_bytes {
            if effective_max_inflight_bytes > max_process {
                effective_max_inflight_bytes = max_process;
            }
        }
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
        node_id,
        profile,
        autotune,
        resize_hw,
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
    max_ram_gb=None,
    profile=None,
    autotune=None,
    constraints=None,
    runtime=None,
    node_id=None
))]
#[allow(clippy::too_many_arguments)]
fn video(
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
        .unwrap_or_else(|| PathBuf::from("/var/lib/mx8/manifests"));
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
    if autotune_enabled && effective_max_process_rss_bytes.is_none() {
        if let Some(node_limit) = detect_node_ram_limit_bytes() {
            let profile_fraction = match selected_profile {
                AutotuneProfile::Safe => 0.60f64,
                AutotuneProfile::Balanced => 0.75f64,
                AutotuneProfile::Throughput => 0.85f64,
            };
            let reserve_bytes = 1u64 << 30;
            let mut derived = ((node_limit as f64) * profile_fraction) as u64;
            derived = derived.saturating_sub(reserve_bytes);
            let min_required = effective_max_inflight_bytes;
            if derived < min_required {
                derived = min_required;
            }
            effective_max_process_rss_bytes = Some(derived.max(effective_max_inflight_bytes));
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

    tracing::info!(
        target: "mx8_proof",
        event = "video_loader_initialized",
        manifest_hash = %snapshot.manifest_hash.0,
        clip_len = clip_len as u64,
        stride = stride as u64,
        fps = fps as u64,
        batch_size_samples = batch_size_samples as u64,
        clips_total = index.summary.clip_count,
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
        "initialized stage2b video loader"
    );

    Py::new(
        py,
        VideoDataLoader {
            manifest_hash: snapshot.manifest_hash.0,
            clips: index.clips,
            stage2d_sidecars,
            rt,
            decode_backend,
            next_idx: 0,
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
            video_runtime_autotune_pressure_milli: 0,
            video_max_process_rss_bytes: effective_max_process_rss_bytes,
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
    max_ram_gb=None,
    profile=None,
    autotune=None,
    constraints=None,
    runtime=None,
))]
#[allow(clippy::too_many_arguments)]
fn mix(
    py: Python<'_>,
    loaders: Vec<Py<DataLoader>>,
    weights: Vec<f64>,
    seed: u64,
    epoch: u64,
    starvation_window: u64,
    source_exhausted: &str,
    max_ram_gb: Option<f64>,
    profile: Option<String>,
    autotune: Option<bool>,
    constraints: Option<Py<Constraints>>,
    runtime: Option<Py<RuntimeConfig>>,
) -> PyResult<Py<MixedDataLoader>> {
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

    let constraints_cfg = constraints.as_ref().map(|c| c.bind(py).borrow().clone());
    let runtime_cfg = runtime.as_ref().map(|r| r.bind(py).borrow().clone());
    let max_ram_bytes_from_gb = max_ram_gb_to_bytes(max_ram_gb)?;

    let mut inflight_caps = Vec::with_capacity(loaders.len());
    let mut batch_sizes = Vec::with_capacity(loaders.len());
    for loader in &loaders {
        let guard = loader.borrow(py);
        inflight_caps.push(guard.max_inflight_bytes);
        batch_sizes.push(guard.batch_size_samples);
    }
    let loader_shared_max_inflight_bytes = compute_shared_mix_cap(&inflight_caps)
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;

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

    for loader in &loaders {
        let mut guard = loader.borrow_mut(py);
        guard.apply_runtime_overrides(runtime_prefetch_override, runtime_max_queue_override);
    }

    let mut queue_caps = Vec::with_capacity(loaders.len());
    let mut prefetch_caps = Vec::with_capacity(loaders.len());
    for loader in &loaders {
        let guard = loader.borrow(py);
        queue_caps.push(guard.max_queue_batches);
        prefetch_caps.push(guard.prefetch_batches);
    }

    let shared_max_inflight_bytes = shared_max_inflight_override
        .map(|v| v.min(loader_shared_max_inflight_bytes))
        .unwrap_or(loader_shared_max_inflight_bytes);

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
    let n = loaders.len();
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
    let out = MixedDataLoader {
        loaders,
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
        mix_max_process_rss_bytes,
    };
    Py::new(py, out)
}

#[pymodule]
fn mx8(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let internal = PyModule::new_bound(py, "_internal")?;
    internal.add_function(wrap_pyfunction!(internal_video_index_build, &internal)?)?;
    internal.add_function(wrap_pyfunction!(
        internal_video_index_replay_check,
        &internal
    )?)?;
    m.add_submodule(&internal)?;
    m.setattr("_internal", &internal)?;

    m.add_class::<DataLoader>()?;
    m.add_class::<ImageLoader>()?;
    m.add_class::<MixedDataLoader>()?;
    m.add_class::<VideoDataLoader>()?;
    m.add_class::<VideoBatch>()?;
    m.add_class::<DistributedDataLoader>()?;
    m.add_class::<Constraints>()?;
    m.add_class::<RuntimeConfig>()?;
    m.add_class::<PyBatch>()?;
    m.add_function(wrap_pyfunction!(pack, m)?)?;
    m.add_function(wrap_pyfunction!(pack_dir, m)?)?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(py_image, m)?)?;
    m.add_function(wrap_pyfunction!(video, m)?)?;
    m.add_function(wrap_pyfunction!(mix, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_manifest_hash, m)?)?;
    Ok(())
}

fn parse_pack_label_mode(s: &str) -> Result<PackLabelMode> {
    let s = s.trim().to_ascii_lowercase();
    let mode = match s.as_str() {
        "none" | "off" | "false" | "0" => PackLabelMode::None,
        "imagefolder" | "image_folder" | "image-folder" => PackLabelMode::ImageFolder,
        "auto" | "" => PackLabelMode::Auto,
        _ => anyhow::bail!("invalid label mode {s:?} (expected: auto|none|imagefolder)"),
    };
    Ok(mode)
}

fn parse_pack_dir_label_mode(s: &str) -> Result<PackDirLabelMode> {
    let s = s.trim().to_ascii_lowercase();
    let mode = match s.as_str() {
        "none" | "off" | "false" | "0" => PackDirLabelMode::None,
        "imagefolder" | "image_folder" | "image-folder" => PackDirLabelMode::ImageFolder,
        "auto" | "" => PackDirLabelMode::Auto,
        _ => anyhow::bail!("invalid label mode {s:?} (expected: auto|none|imagefolder)"),
    };
    Ok(mode)
}

fn max_ram_gb_to_bytes(max_ram_gb: Option<f64>) -> PyResult<Option<u64>> {
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

fn derive_byte_batch_caps(
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
fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.trim().parse::<u64>().ok()
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(raw) => matches!(
            raw.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => default,
    }
}

fn should_use_zero_manifest_scan(parsed: &mx8_core::dataset_link::DatasetLink, base: &str) -> bool {
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
    if !trimmed.starts_with("s3://") {
        return false;
    }
    let lowered = trimmed.to_ascii_lowercase();
    !lowered.ends_with(".tsv")
}

fn parse_u64_ascii(raw: &str) -> Option<u64> {
    raw.trim().parse::<u64>().ok()
}

fn parse_s3_bucket_key_local(location: &str) -> Result<(String, String)> {
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

fn parse_decode_hint_fields_local(raw_hint: &str) -> HashMap<String, String> {
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

fn parse_stage2d_sidecar_from_hint(
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

fn build_stage2d_sidecar_map(manifest_bytes: &[u8]) -> Result<HashMap<String, VideoRangeSidecar>> {
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

fn detect_cgroup_memory_limit_bytes() -> Option<u64> {
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

fn detect_proc_memtotal_bytes() -> Option<u64> {
    let txt = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in txt.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kib = rest.split_whitespace().next()?.parse::<u64>().ok()?;
            return kib.checked_mul(1024);
        }
    }
    None
}

fn detect_sysctl_memsize_bytes() -> Option<u64> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_u64_ascii(std::str::from_utf8(&out.stdout).ok()?)
}

fn detect_node_ram_limit_bytes() -> Option<u64> {
    let cgroup = detect_cgroup_memory_limit_bytes();
    let host = detect_proc_memtotal_bytes().or_else(detect_sysctl_memsize_bytes);
    match (cgroup, host) {
        (Some(c), Some(h)) => Some(c.min(h)),
        (Some(c), None) => Some(c),
        (None, Some(h)) => Some(h),
        (None, None) => None,
    }
}

fn sample_process_rss_bytes_local() -> Option<u64> {
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

const DEFAULT_GRPC_MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

fn unix_time_ms() -> u64 {
    let now = std::time::SystemTime::now();
    now.duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis()
        .min(u64::MAX as u128) as u64
}

#[derive(Debug)]
struct LeaseProgress {
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

#[derive(Debug, Clone, Copy)]
enum AutotuneProfile {
    Safe,
    Balanced,
    Throughput,
}

impl AutotuneProfile {
    fn from_name(name: Option<&str>) -> Self {
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
struct ProfileDefaults {
    max_inflight_bytes: u64,
    max_queue_batches: usize,
    prefetch_batches: usize,
}

impl ProfileDefaults {
    fn for_profile(profile: AutotuneProfile) -> Self {
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
struct AutotuneRails {
    min_prefetch_batches: usize,
    max_prefetch_batches: usize,
    min_max_queue_batches: usize,
    max_max_queue_batches: usize,
    min_want: u32,
    max_want: u32,
}

impl AutotuneRails {
    fn for_profile(profile: AutotuneProfile) -> Self {
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
struct AutotuneShared {
    enabled: bool,
    want: AtomicU32,
    prefetch_batches: AtomicUsize,
    max_queue_batches: AtomicUsize,
    wait_ns_interval: AtomicU64,
    pressure_milli: AtomicU64,
    wait_ewma_milli: AtomicU64,
    rss_ewma_milli: AtomicU64,
    integral_rss_milli: AtomicI64,
    cooldown_ticks: AtomicU32,
}

impl AutotuneShared {
    fn new(enabled: bool, want: u32, prefetch_batches: usize, max_queue_batches: usize) -> Self {
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

    fn on_wait(&self, elapsed: Duration) {
        if self.enabled {
            self.wait_ns_interval.fetch_add(
                elapsed.as_nanos().min(u128::from(u64::MAX)) as u64,
                Ordering::Relaxed,
            );
        }
    }
}

#[derive(Debug)]
struct AutotuneController {
    wait_ewma: f64,
    rss_ewma: f64,
    prev_rss_ewma: f64,
    integral_rss: f64,
    cooldown_ticks: u32,
    increase_ticks: u32,
}

impl AutotuneController {
    fn new() -> Self {
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
struct AutotuneUpdate {
    want: u32,
    prefetch_batches: usize,
    max_queue_batches: usize,
}

#[derive(Debug)]
struct AutotuneTickOutput {
    next: AutotuneUpdate,
    changed: bool,
    reason: &'static str,
    pressure: f64,
}

fn autotune_tick(
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

async fn autotune_loop(
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

async fn fetch_manifest_bytes(
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

async fn heartbeat_loop(
    channel: Channel,
    grpc_max_message_bytes: usize,
    interval: Duration,
    job_id: String,
    node_id: String,
    pipeline: Arc<Pipeline>,
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
async fn progress_loop(
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
async fn run_lease_and_stream_batches(
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
struct DistributedDataLoader {
    manifest_hash: String,
    assigned_rank: u32,
    world_size: u32,
    metrics: Arc<RuntimeMetrics>,
    pipeline: Arc<Pipeline>,
    autotune: Arc<AutotuneShared>,
    rx: tokio::sync::mpsc::Receiver<BatchLease>,
    task: Option<tokio::task::JoinHandle<Result<()>>>,
    heartbeat_task: Option<tokio::task::JoinHandle<()>>,
    autotune_task: Option<tokio::task::JoinHandle<()>>,
    rt: tokio::runtime::Runtime,
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
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
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
        if let Some(max_process) = effective_max_process_rss_bytes {
            if effective_max_inflight_bytes > max_process {
                effective_max_inflight_bytes = max_process;
            }
        }

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("failed to start tokio runtime: {e}")))?;

        let (manifest_hash, assigned_rank, world_size, heartbeat_interval_ms, channel) = rt
            .block_on(async {
                let channel = Channel::from_shared(coord_url.clone())?.connect().await?;
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
                        job_id: job_id.clone(),
                        node_id: node_id.clone(),
                        caps: caps.clone(),
                    })
                    .await?
                    .into_inner();

                while !resp.job_ready {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    resp = client
                        .register_node(RegisterNodeRequest {
                            job_id: job_id.clone(),
                            node_id: node_id.clone(),
                            caps: caps.clone(),
                        })
                        .await?
                        .into_inner();
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
        let task = rt.spawn(async move {
            let mut next_request_at = tokio::time::Instant::now();
            loop {
                let now = tokio::time::Instant::now();
                if now < next_request_at {
                    tokio::time::sleep_until(next_request_at).await;
                }
                let want = if autotune_for_requests.enabled {
                    autotune_for_requests.want.load(Ordering::Relaxed).max(1)
                } else {
                    std::cmp::max(1, effective_want)
                };

                let mut client = CoordinatorClient::new(channel.clone())
                    .max_decoding_message_size(grpc_max_message_bytes)
                    .max_encoding_message_size(grpc_max_message_bytes);
                let resp = client
                    .request_lease(RequestLeaseRequest {
                        job_id: job_id.clone(),
                        node_id: node_id.clone(),
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

                if resp.leases.is_empty() {
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
                        job_id.clone(),
                        node_id.clone(),
                        core_lease,
                        tx.clone(),
                        progress_interval_ms,
                    )
                    .await?;
                }
            }
        });

        Ok(Self {
            manifest_hash,
            assigned_rank,
            world_size,
            metrics,
            pipeline,
            autotune,
            rx,
            task: Some(task),
            heartbeat_task,
            autotune_task,
            rt,
        })
    }

    #[getter]
    fn manifest_hash(&self) -> &str {
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

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
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

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<PyBatch> {
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

    fn close(&mut self) {
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
