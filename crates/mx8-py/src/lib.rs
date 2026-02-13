#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use std::path::PathBuf;

use anyhow::Result;
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList};

use mx8_manifest_store::fs::FsManifestStore;
use mx8_manifest_store::LockOwner;
use mx8_runtime::pipeline::{BatchLease, Pipeline, RuntimeCaps};
use mx8_snapshot::{SnapshotResolver, SnapshotResolverConfig};

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

    fn to_torch<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
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

#[pymodule]
fn mx8(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DataLoader>()?;
    m.add_class::<PyBatch>()?;
    m.add_function(wrap_pyfunction!(resolve_manifest_hash, m)?)?;
    Ok(())
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
}
