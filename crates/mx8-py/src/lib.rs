#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use std::path::PathBuf;
use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use fast_image_resize::images::Image as FirImage;
use fast_image_resize::{
    FilterType as FirFilterType, PixelType as FirPixelType, ResizeAlg as FirResizeAlg,
    ResizeOptions as FirResizeOptions, Resizer as FirResizer,
};
use image::imageops::FilterType as ImageFilterType;
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList, PyTuple};
use rayon::prelude::*;
use turbojpeg::PixelFormat as TurboPixelFormat;
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::JpegDecoder;

use mx8_manifest_store::fs::FsManifestStore;
use mx8_manifest_store::LockOwner;
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
use mx8_snapshot::{SnapshotResolver, SnapshotResolverConfig};
use mx8_wire::TryToCore;
use tonic::transport::Channel;

type TorchBatch3<'py> = (Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>);
type TorchBatch4<'py> = (
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
);

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
struct ImageFolderLoader {
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
    metrics: Arc<RuntimeMetrics>,
    rx: tokio::sync::mpsc::Receiver<BatchLease>,
    task: Option<tokio::task::JoinHandle<Result<()>>>,
    rt: tokio::runtime::Runtime,
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
    #[pyo3(signature = (max_inflight_bytes=None, max_process_rss_bytes=None))]
    fn new(max_inflight_bytes: Option<u64>, max_process_rss_bytes: Option<u64>) -> Self {
        Self {
            max_inflight_bytes,
            max_process_rss_bytes,
        }
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

fn labels_to_torch_i64<'py>(py: Python<'py>, labels: &[u64]) -> PyResult<Bound<'py, PyAny>> {
    let torch = py.import_bound("torch").map_err(|e| {
        PyRuntimeError::new_err(format!(
            "failed to import torch (install PyTorch to use ImageFolderLoader): {e}"
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
            let decoded = image::load_from_memory(bytes)
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
                        image::RgbImage::from_raw(width, height, raw_rgb).ok_or_else(|| {
                            format!("decoded rgb buffer shape mismatch for sample_id {sample_id}")
                        })?;
                    let resized = image::imageops::resize(&rgb, w, h, ImageFilterType::Triangle);
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
        manifest_store_root=None,
        dev_manifest_path=None,
        batch_size_samples=512,
        max_inflight_bytes=128*1024*1024,
        max_queue_batches=64,
        prefetch_batches=1,
        target_batch_bytes=None,
        max_batch_bytes=None,
        max_process_rss_bytes=None,
        start_id=None,
        end_id=None,
        node_id=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dataset_link: String,
        manifest_store_root: Option<PathBuf>,
        dev_manifest_path: Option<PathBuf>,
        batch_size_samples: usize,
        max_inflight_bytes: u64,
        max_queue_batches: usize,
        prefetch_batches: usize,
        target_batch_bytes: Option<u64>,
        max_batch_bytes: Option<u64>,
        max_process_rss_bytes: Option<u64>,
        start_id: Option<u64>,
        end_id: Option<u64>,
        node_id: Option<String>,
    ) -> PyResult<Self> {
        let parsed = mx8_core::dataset_link::DatasetLink::parse(&dataset_link)
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        let dataset_base = match parsed {
            mx8_core::dataset_link::DatasetLink::Plain(b) => b,
            mx8_core::dataset_link::DatasetLink::Refresh(b) => b,
            mx8_core::dataset_link::DatasetLink::Pinned { base, .. } => base,
        };

        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("failed to start tokio runtime: {e}")))?;

        let root = manifest_store_root
            .or(env_path("MX8_MANIFEST_STORE_ROOT"))
            .unwrap_or_else(|| PathBuf::from("/var/lib/mx8/manifests"));

        let dev_manifest_path = dev_manifest_path.or(env_path("MX8_DEV_MANIFEST_PATH"));

        let store = std::sync::Arc::new(FsManifestStore::new(root));
        let resolver = SnapshotResolver::new(
            store,
            SnapshotResolverConfig {
                dev_manifest_path,
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

        let max_process_rss_bytes_cap =
            max_process_rss_bytes.or_else(|| env_u64("MX8_MAX_PROCESS_RSS_BYTES"));
        let caps = RuntimeCaps {
            max_inflight_bytes,
            max_queue_batches,
            batch_size_samples,
            prefetch_batches,
            target_batch_bytes,
            max_batch_bytes,
            max_process_rss_bytes: max_process_rss_bytes_cap,
        };
        let pipeline = Pipeline::new(caps);
        let metrics = pipeline.metrics();

        let (rx, task) = rt
            .block_on(async move {
                match (start_id, end_id) {
                    (Some(s), Some(e)) => {
                        pipeline
                            .spawn_manifest_bytes_range_stream(snapshot.manifest_bytes, s, e)
                            .await
                    }
                    (None, None) => {
                        pipeline
                            .spawn_manifest_bytes_stream(snapshot.manifest_bytes)
                            .await
                    }
                    _ => anyhow::bail!("start_id and end_id must be set together"),
                }
            })
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;

        Ok(Self {
            dataset_base,
            manifest_hash: snapshot.manifest_hash.0,
            metrics,
            rx,
            task: Some(task),
            rt,
        })
    }

    #[getter]
    fn manifest_hash(&self) -> &str {
        &self.manifest_hash
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(metrics_to_dict(py, self.metrics.as_ref())?.into_any())
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let lease = py.allow_threads(|| self.rt.block_on(async { self.rx.recv().await }));
        match lease {
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
        }
    }
}

impl ImageFolderLoader {
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
                "failed to import torch (install PyTorch to use ImageFolderLoader): {e}"
            ))
        })?;
        let np = py.import_bound("numpy").map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to import numpy (install numpy to use ImageFolderLoader): {e}"
            ))
        })?;
        let pil = py.import_bound("PIL.Image").map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to import PIL.Image (install Pillow to use ImageFolderLoader): {e}"
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
                "failed to import torch (install PyTorch to use ImageFolderLoader): {e}"
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
impl ImageFolderLoader {
    #[new]
    #[pyo3(signature = (
        dataset_link,
        *,
        manifest_store_root=None,
        dev_manifest_path=None,
        batch_size_samples=32,
        max_inflight_bytes=128*1024*1024,
        max_queue_batches=64,
        prefetch_batches=1,
        target_batch_bytes=None,
        max_batch_bytes=None,
        start_id=None,
        end_id=None,
        node_id=None,
        resize_hw=None,
        to_float=true,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dataset_link: String,
        manifest_store_root: Option<PathBuf>,
        dev_manifest_path: Option<PathBuf>,
        batch_size_samples: usize,
        max_inflight_bytes: u64,
        max_queue_batches: usize,
        prefetch_batches: usize,
        target_batch_bytes: Option<u64>,
        max_batch_bytes: Option<u64>,
        start_id: Option<u64>,
        end_id: Option<u64>,
        node_id: Option<String>,
        resize_hw: Option<(u32, u32)>,
        to_float: bool,
    ) -> PyResult<Self> {
        let loader = DataLoader::new(
            dataset_link,
            manifest_store_root,
            dev_manifest_path,
            batch_size_samples,
            max_inflight_bytes,
            max_queue_batches,
            prefetch_batches,
            target_batch_bytes,
            max_batch_bytes,
            None,
            start_id,
            end_id,
            node_id,
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
#[pyo3(signature = (dataset_link, *, manifest_store_root=None, dev_manifest_path=None, node_id=None))]
fn resolve_manifest_hash(
    dataset_link: String,
    manifest_store_root: Option<PathBuf>,
    dev_manifest_path: Option<PathBuf>,
    node_id: Option<String>,
) -> PyResult<String> {
    let root = manifest_store_root
        .or(env_path("MX8_MANIFEST_STORE_ROOT"))
        .unwrap_or_else(|| PathBuf::from("/var/lib/mx8/manifests"));
    let dev_manifest_path = dev_manifest_path.or(env_path("MX8_DEV_MANIFEST_PATH"));

    let store = std::sync::Arc::new(FsManifestStore::new(root));
    let resolver = SnapshotResolver::new(
        store,
        SnapshotResolverConfig {
            dev_manifest_path,
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
#[pyo3(signature = (
    dataset_link,
    *,
    manifest_store_root=None,
    dev_manifest_path=None,
    batch_size_samples=512,
    max_inflight_bytes=128*1024*1024,
    max_queue_batches=64,
    prefetch_batches=1,
    target_batch_bytes=None,
    max_batch_bytes=None,
    start_id=None,
    end_id=None,
    node_id=None,
    profile=None,
    autotune=None,
    constraints=None,
    runtime=None,
))]
#[allow(clippy::too_many_arguments)]
fn load(
    py: Python<'_>,
    dataset_link: String,
    manifest_store_root: Option<PathBuf>,
    dev_manifest_path: Option<PathBuf>,
    batch_size_samples: usize,
    max_inflight_bytes: u64,
    max_queue_batches: usize,
    prefetch_batches: usize,
    target_batch_bytes: Option<u64>,
    max_batch_bytes: Option<u64>,
    start_id: Option<u64>,
    end_id: Option<u64>,
    node_id: Option<String>,
    profile: Option<String>,
    autotune: Option<bool>,
    constraints: Option<Py<Constraints>>,
    runtime: Option<Py<RuntimeConfig>>,
) -> PyResult<Py<DataLoader>> {
    let has_v1_args =
        profile.is_some() || autotune.is_some() || constraints.is_some() || runtime.is_some();

    let constraints_cfg = constraints.as_ref().map(|c| c.bind(py).borrow().clone());
    let runtime_cfg = runtime.as_ref().map(|r| r.bind(py).borrow().clone());

    let mut effective_max_inflight_bytes = max_inflight_bytes;
    let mut effective_max_queue_batches = max_queue_batches;
    let mut effective_prefetch_batches = prefetch_batches;
    let mut effective_max_process_rss_bytes = env_u64("MX8_MAX_PROCESS_RSS_BYTES");

    if has_v1_args {
        let selected_profile = AutotuneProfile::from_name(profile.as_deref());
        let defaults = ProfileDefaults::for_profile(selected_profile);
        let autotune_enabled = autotune.unwrap_or(true);
        effective_max_inflight_bytes = defaults.max_inflight_bytes;
        effective_max_queue_batches = defaults.max_queue_batches;
        effective_prefetch_batches = defaults.prefetch_batches;

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
        manifest_store_root,
        dev_manifest_path,
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
    )?;
    Py::new(py, loader)
}

#[pymodule]
fn mx8(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let vision = PyModule::new_bound(py, "vision")?;
    vision.add_class::<ImageFolderLoader>()?;
    m.add_submodule(&vision)?;
    m.setattr("vision", &vision)?;

    m.add_class::<DataLoader>()?;
    m.add_class::<DistributedDataLoader>()?;
    m.add_class::<Constraints>()?;
    m.add_class::<RuntimeConfig>()?;
    m.add_class::<PyBatch>()?;
    m.add_function(wrap_pyfunction!(pack, m)?)?;
    m.add_function(wrap_pyfunction!(pack_dir, m)?)?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
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
fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.trim().parse::<u64>().ok()
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

fn parse_u64_ascii(raw: &str) -> Option<u64> {
    raw.trim().parse::<u64>().ok()
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

    fn from_env() -> Self {
        Self::from_name(std::env::var("MX8_AUTOTUNE_PROFILE").ok().as_deref())
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
        progress_interval_ms=500,
        grpc_max_message_bytes=DEFAULT_GRPC_MAX_MESSAGE_BYTES,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
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
        progress_interval_ms: u64,
        grpc_max_message_bytes: usize,
    ) -> PyResult<Self> {
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
                    max_inflight_bytes,
                    max_ram_bytes: max_inflight_bytes,
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

        let max_process_rss_bytes_cap = env_u64("MX8_MAX_PROCESS_RSS_BYTES");
        let caps = RuntimeCaps {
            max_inflight_bytes,
            max_queue_batches,
            batch_size_samples,
            prefetch_batches,
            target_batch_bytes,
            max_batch_bytes,
            max_process_rss_bytes: max_process_rss_bytes_cap,
        };
        let pipeline = Arc::new(Pipeline::new(caps));
        let metrics = pipeline.metrics();
        let autotune_enabled = env_bool("MX8_AUTOTUNE", false);
        let autotune_profile = AutotuneProfile::from_env();
        let rails = AutotuneRails::for_profile(autotune_profile);
        let initial_want = std::cmp::max(1, want).clamp(rails.min_want, rails.max_want);
        let initial_prefetch = prefetch_batches
            .max(rails.min_prefetch_batches)
            .min(rails.max_prefetch_batches);
        let initial_max_queue = max_queue_batches
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
                max_inflight_bytes = max_inflight_bytes,
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
                max_inflight_bytes,
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
                    std::cmp::max(1, want)
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
