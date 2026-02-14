#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList, PyTuple};

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
use mx8_runtime::pipeline::{BatchLease, Pipeline, RuntimeCaps};
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

#[pyclass]
struct ImageFolderLoader {
    loader: DataLoader,
    resize_hw: Option<(u32, u32)>,
    to_float: bool,
}

#[pyclass]
struct DataLoader {
    manifest_hash: String,
    rx: tokio::sync::mpsc::Receiver<BatchLease>,
    task: Option<tokio::task::JoinHandle<Result<()>>>,
    rt: tokio::runtime::Runtime,
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
        start_id: Option<u64>,
        end_id: Option<u64>,
        node_id: Option<String>,
    ) -> PyResult<Self> {
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

        let caps = RuntimeCaps {
            max_inflight_bytes,
            max_queue_batches,
            batch_size_samples,
            prefetch_batches,
        };
        let pipeline = Pipeline::new(caps);

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
            manifest_hash: snapshot.manifest_hash.0,
            rx,
            task: Some(task),
            rt,
        })
    }

    #[getter]
    fn manifest_hash(&self) -> &str {
        &self.manifest_hash
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
            start_id,
            end_id,
            node_id,
        )?;
        Ok(Self {
            loader,
            resize_hw,
            to_float,
        })
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
                // PIL expects size=(width,height)
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

        let torch_int64 = torch.getattr("int64")?;
        let mut labels_i64 = Vec::with_capacity(labels.len());
        for &lab in labels.iter() {
            labels_i64.push(i64::try_from(lab).map_err(|_| {
                PyValueError::new_err(format!(
                    "label_id overflow converting u64 -> i64 (label_id={lab})"
                ))
            })?);
        }
        let labels_list = PyList::new_bound(py, labels_i64);
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("dtype", &torch_int64)?;
        let labels = torch.call_method("tensor", (labels_list,), Some(&kwargs))?;

        let out = PyTuple::new_bound(py, [images.to_object(py), labels.to_object(py)]);
        Ok(out.into_any())
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
    start_id=None,
    end_id=None,
    node_id=None,
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
    start_id: Option<u64>,
    end_id: Option<u64>,
    node_id: Option<String>,
) -> PyResult<Py<DataLoader>> {
    let loader = DataLoader::new(
        dataset_link,
        manifest_store_root,
        dev_manifest_path,
        batch_size_samples,
        max_inflight_bytes,
        max_queue_batches,
        prefetch_batches,
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
    m.add_class::<PyBatch>()?;
    m.add_function(wrap_pyfunction!(pack, m)?)?;
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

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
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
            ram_high_water_bytes: metrics.inflight_bytes_high_water.get(),
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
    rx: tokio::sync::mpsc::Receiver<BatchLease>,
    task: Option<tokio::task::JoinHandle<Result<()>>>,
    heartbeat_task: Option<tokio::task::JoinHandle<()>>,
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

        let caps = RuntimeCaps {
            max_inflight_bytes,
            max_queue_batches,
            batch_size_samples,
            prefetch_batches,
        };
        let pipeline = Arc::new(Pipeline::new(caps));

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

        let (tx, rx) = tokio::sync::mpsc::channel::<BatchLease>(max_queue_batches);
        let manifest_bytes = Arc::new(manifest_bytes);
        let task = rt.spawn(async move {
            let mut next_request_at = tokio::time::Instant::now();
            let want = std::cmp::max(1, want);
            loop {
                let now = tokio::time::Instant::now();
                if now < next_request_at {
                    tokio::time::sleep_until(next_request_at).await;
                }

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
                        pipeline.clone(),
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
            rx,
            task: Some(task),
            heartbeat_task,
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

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<PyBatch> {
        self.rt.block_on(async {
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
        })
    }

    fn close(&mut self) {
        if let Some(handle) = self.task.take() {
            handle.abort();
        }
        if let Some(handle) = self.heartbeat_task.take() {
            handle.abort();
        }
    }

    fn __del__(&mut self) {
        self.close();
    }
}
