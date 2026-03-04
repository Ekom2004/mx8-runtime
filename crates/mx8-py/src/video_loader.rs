use super::*;

#[pyclass]
pub(crate) struct VideoBatch {
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
    experimental_device_output_active: bool,
    experimental_device_direct_write_active: bool,
}

#[pyclass]
pub(crate) struct VideoDataLoader {
    pub(crate) manifest_hash: String,
    pub(crate) clips: Vec<VideoClipRecord>,
    pub(crate) stage2d_sidecars: HashMap<String, VideoRangeSidecar>,
    pub(crate) rt: tokio::runtime::Runtime,
    pub(crate) decode_backend: VideoDecodeBackend,
    pub(crate) next_idx: usize,
    pub(crate) batch_size_samples: usize,
    pub(crate) max_inflight_bytes: u64,
    pub(crate) bytes_per_clip: usize,
    pub(crate) decode_contract: VideoDecodeContract,
    pub(crate) seed: u64,
    pub(crate) epoch: u64,
    pub(crate) clip_len: u32,
    pub(crate) stride: u32,
    pub(crate) decode_fps: u32,
    pub(crate) fps_policy: String,
    pub(crate) delivered_batches: u64,
    pub(crate) delivered_samples: u64,
    pub(crate) delivered_bytes: u64,
    pub(crate) decode_attempted_clips: u64,
    pub(crate) decode_succeeded_clips: u64,
    pub(crate) decode_failed_io_read_failed: u64,
    pub(crate) decode_failed_corrupt_media: u64,
    pub(crate) decode_failed_short_media: u64,
    pub(crate) decode_failed_unsupported_codec: u64,
    pub(crate) decode_failed_missing_stream: u64,
    pub(crate) decode_failed_backend_unavailable: u64,
    pub(crate) decode_failed_decode_failed: u64,
    pub(crate) decode_backend_fallback_total: u64,
    pub(crate) decode_ms_total: u64,
    pub(crate) s3_range_requests_total: u64,
    pub(crate) s3_range_bytes_fetched_total: u64,
    pub(crate) s3_stage2d_plan_used_total: u64,
    pub(crate) s3_stage2d_plan_fallback_total: u64,
    pub(crate) s3_full_object_range_fallback_total: u64,
    pub(crate) video_runtime_autotune_enabled: bool,
    pub(crate) video_runtime_autotune_period_batches: u64,
    pub(crate) video_runtime_autotune_last_batch: u64,
    pub(crate) video_runtime_autotune_adjustments_total: u64,
    pub(crate) video_runtime_autotune_gpu_clamps_total: u64,
    pub(crate) video_runtime_autotune_pressure_milli: u64,
    pub(crate) video_experimental_device_output_requested: bool,
    pub(crate) video_experimental_device_output_active: bool,
    pub(crate) video_experimental_device_output_fallback_total: u64,
    pub(crate) video_experimental_device_direct_write_requested: bool,
    pub(crate) video_experimental_device_direct_write_active: bool,
    pub(crate) video_experimental_device_direct_write_fallback_total: u64,
    pub(crate) video_experimental_device_direct_write_batches_total: u64,
    pub(crate) video_gpu_pressure_milli: u64,
    pub(crate) video_gpu_pressure_available: bool,
    pub(crate) video_gpu_pressure_unavailable_total: u64,
    pub(crate) video_gpu_recovery_streak: u64,
    pub(crate) video_last_gpu_sample_at: Option<Instant>,
    pub(crate) video_max_process_rss_bytes: Option<u64>,
    pub(crate) assigned_rank: u32,
    pub(crate) world_size: u32,
    pub(crate) job_id: Option<String>,
    pub(crate) cluster_url: Option<String>,
    pub(crate) started_at: Instant,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct VideoDecodeContract {
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
pub(crate) struct VideoDecodeError {
    class: &'static str,
    path: PathBuf,
    detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VideoDeviceDestinationContract {
    shape: [i64; 5],
    strides: [i64; 5],
    data_ptr: u64,
    stream_id: u64,
}

impl VideoDeviceDestinationContract {
    fn from_cuda_tensor<'py>(cuda_payload: &Bound<'py, PyAny>, stream_id: u64) -> PyResult<Self> {
        let shape = py_i64_sequence_5(&cuda_payload.getattr("shape")?, "shape")?;
        let strides = py_i64_sequence_5(&cuda_payload.call_method0("stride")?, "stride")?;
        let data_ptr = cuda_payload.call_method0("data_ptr")?.extract::<u64>()?;
        Ok(Self {
            shape,
            strides,
            data_ptr,
            stream_id,
        })
    }

    fn validate_for_shape(&self, expected_shape: [i64; 5]) -> Result<(), String> {
        if self.data_ptr == 0 {
            return Err("data_ptr is zero".to_string());
        }
        if self.shape != expected_shape {
            return Err(format!(
                "shape mismatch: actual={:?} expected={:?}",
                self.shape, expected_shape
            ));
        }
        let expected_strides = Self::contiguous_strides_for_shape(expected_shape)?;
        if self.strides != expected_strides {
            return Err(format!(
                "stride mismatch: actual={:?} expected={:?}",
                self.strides, expected_strides
            ));
        }
        Ok(())
    }

    fn contiguous_strides_for_shape(shape: [i64; 5]) -> Result<[i64; 5], String> {
        if shape.iter().any(|dim| *dim <= 0) {
            return Err(format!("shape must be positive for all dims: {shape:?}"));
        }
        let c = shape[4];
        let w = shape[3];
        let h = shape[2];
        let t = shape[1];
        let stride_c = 1_i64;
        let stride_w = checked_mul_i64(c, stride_c)?;
        let stride_h = checked_mul_i64(w, stride_w)?;
        let stride_t = checked_mul_i64(h, stride_h)?;
        let stride_b = checked_mul_i64(t, stride_t)?;
        Ok([stride_b, stride_t, stride_h, stride_w, stride_c])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VideoDeviceWritePath {
    TorchCopyScaffold,
    DirectWriteNativeV1,
    DirectWriteStagedCopyFallback,
}

impl VideoDeviceWritePath {
    fn mode_name(self) -> &'static str {
        match self {
            Self::TorchCopyScaffold => "torch_copy_scaffold",
            Self::DirectWriteNativeV1 => "direct_write_native_v1",
            Self::DirectWriteStagedCopyFallback => "direct_write_staged_copy_fallback_v1",
        }
    }
}

fn checked_mul_i64(lhs: i64, rhs: i64) -> Result<i64, String> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| format!("i64 multiply overflow: lhs={lhs} rhs={rhs}"))
}

fn py_i64_sequence_5(value: &Bound<'_, PyAny>, field_name: &str) -> PyResult<[i64; 5]> {
    let raw: Vec<i64> = value.extract().map_err(|e| {
        PyValueError::new_err(format!(
            "video device destination {field_name} is not an integer sequence: {e}"
        ))
    })?;
    if raw.len() != 5 {
        return Err(PyValueError::new_err(format!(
            "video device destination {field_name} must have len=5, got len={}",
            raw.len()
        )));
    }
    Ok([raw[0], raw[1], raw[2], raw[3], raw[4]])
}

fn maybe_load_direct_write_op_library(torch: &Bound<'_, PyAny>) -> Result<(), String> {
    let Some(path) = env_string("MX8_VIDEO_DIRECT_WRITE_OP_LIBRARY") else {
        return Ok(());
    };
    static DIRECT_WRITE_OP_LIBRARY_LOAD: std::sync::OnceLock<Result<(), String>> =
        std::sync::OnceLock::new();
    let result = DIRECT_WRITE_OP_LIBRARY_LOAD.get_or_init(|| {
        let ops = torch
            .getattr("ops")
            .map_err(|err| format!("torch.ops unavailable for direct-write op load: {err}"))?;
        ops.call_method1("load_library", (path.clone(),))
            .map_err(|err| format!("failed loading direct-write op library {path}: {err}"))?;
        Ok(())
    });
    result.clone()
}

impl VideoBatch {
    fn write_payload_into_cuda_destination<'py>(
        &self,
        cuda_payload: &Bound<'py, PyAny>,
        payload_view: &Bound<'py, PyAny>,
        stream: &Bound<'py, PyAny>,
        destination: &VideoDeviceDestinationContract,
        write_path: VideoDeviceWritePath,
    ) -> PyResult<()> {
        // Phase 1 boundary: all destination writes flow through one internal contract-checked path.
        let copy_kwargs = PyDict::new_bound(payload_view.py());
        copy_kwargs.set_item("non_blocking", true)?;
        cuda_payload.call_method("copy_", (payload_view.clone(),), Some(&copy_kwargs))?;
        if let Err(err) = cuda_payload.call_method1("record_stream", (stream.clone(),)) {
            tracing::warn!(
                target: "mx8_proof",
                event = "video_experimental_device_output_record_stream_failed",
                detail = %err,
                "record_stream failed for video experimental device output"
            );
        }
        tracing::info!(
            target: "mx8_proof",
            event = "video_experimental_device_destination_writer",
            mode = write_path.mode_name(),
            stream_id = destination.stream_id,
            data_ptr = destination.data_ptr,
            bytes = self.payload.len() as u64,
            "video destination writer boundary completed payload write"
        );
        Ok(())
    }

    fn try_native_direct_write<'py>(
        &self,
        cuda_payload: &Bound<'py, PyAny>,
        payload_view: &Bound<'py, PyAny>,
        stream: &Bound<'py, PyAny>,
        destination: &VideoDeviceDestinationContract,
    ) -> Result<VideoDeviceWritePath, String> {
        // Native direct-write integration point:
        // if a custom torch op is available, route writes through it.
        let py = payload_view.py();
        let torch = py
            .import_bound("torch")
            .map_err(|err| format!("failed to import torch: {err}"))?;
        maybe_load_direct_write_op_library(&torch)?;
        let ops = torch
            .getattr("ops")
            .map_err(|err| format!("torch.ops unavailable: {err}"))?;
        let ns = ops
            .getattr("mx8_video")
            .map_err(|err| format!("torch.ops.mx8_video unavailable: {err}"))?;
        let direct_write_op = ns
            .getattr("direct_write_u8")
            .map_err(|err| format!("torch.ops.mx8_video.direct_write_u8 unavailable: {err}"))?;
        direct_write_op
            .call1((
                cuda_payload.clone(),
                payload_view.clone(),
                destination.stream_id,
            ))
            .map_err(|err| format!("torch.ops.mx8_video.direct_write_u8 failed: {err}"))?;
        if let Err(err) = cuda_payload.call_method1("record_stream", (stream.clone(),)) {
            tracing::warn!(
                target: "mx8_proof",
                event = "video_experimental_device_output_record_stream_failed",
                detail = %err,
                "record_stream failed for native direct-write path"
            );
        }
        Ok(VideoDeviceWritePath::DirectWriteNativeV1)
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

    fn to_torch<'py>(&self, py: Python<'py>) -> PyResult<TorchBatch3<'py>> {
        let torch = py.import_bound("torch").map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to import torch (install PyTorch to use video to_torch): {e}"
            ))
        })?;
        let torch_uint8 = torch.getattr("uint8")?;
        let torch_int64 = torch.getattr("int64")?;

        let payload = PyByteArray::new_bound(py, &self.payload);
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("dtype", &torch_uint8)?;
        let payload_u8 = torch.call_method("frombuffer", (payload,), Some(&kwargs))?;

        let rows = i64::try_from(self.sample_ids.len())
            .map_err(|_| PyValueError::new_err("video batch rows do not fit i64"))?;
        let frames = i64::from(self.frames_per_clip);
        let height = i64::from(self.frame_height);
        let width = i64::from(self.frame_width);
        let channels = i64::from(self.channels);
        let expected_shape = [rows, frames, height, width, channels];
        let payload_view =
            payload_u8.call_method1("view", (rows, frames, height, width, channels))?;

        let payload_out = if self.experimental_device_output_active {
            let cuda_attempt = (|| -> PyResult<Bound<'py, PyAny>> {
                let cuda = torch.getattr("cuda")?;
                let stream = cuda.call_method0("current_stream")?;
                let stream_id = stream.getattr("cuda_stream")?.extract::<u64>()?;

                let kwargs = PyDict::new_bound(py);
                kwargs.set_item("dtype", &torch_uint8)?;
                kwargs.set_item("device", "cuda")?;
                let cuda_payload = torch.call_method(
                    "empty",
                    ((rows, frames, height, width, channels),),
                    Some(&kwargs),
                )?;
                let destination =
                    VideoDeviceDestinationContract::from_cuda_tensor(&cuda_payload, stream_id)?;
                destination
                    .validate_for_shape(expected_shape)
                    .map_err(|detail| {
                        PyRuntimeError::new_err(format!(
                            "video device destination contract validation failed: {detail}"
                        ))
                    })?;
                tracing::info!(
                    target: "mx8_proof",
                    event = "video_experimental_device_destination_contract",
                    mode = if self.experimental_device_direct_write_active {
                        "direct_write_destination_contract_v1"
                    } else {
                        "device_output"
                    },
                    stream_id = destination.stream_id,
                    data_ptr = destination.data_ptr,
                    shape = ?destination.shape,
                    strides = ?destination.strides,
                    bytes = self.payload.len() as u64,
                    "prepared torch-owned cuda destination tensor contract for video payload"
                );

                let write_path = if self.experimental_device_direct_write_active {
                    tracing::info!(
                        target: "mx8_proof",
                        event = "video_experimental_device_direct_write_attempt",
                        mode = "native_direct_write_v1",
                        stream_id = destination.stream_id,
                        bytes = self.payload.len() as u64,
                        "attempting experimental native direct-write into destination tensor"
                    );
                    match self.try_native_direct_write(
                        &cuda_payload,
                        &payload_view,
                        &stream,
                        &destination,
                    ) {
                        Ok(VideoDeviceWritePath::DirectWriteNativeV1) => {
                            VideoDeviceWritePath::DirectWriteNativeV1
                        }
                        Ok(other) => other,
                        Err(detail) => {
                            tracing::warn!(
                                target: "mx8_proof",
                                event = "video_experimental_device_direct_write_native_fallback",
                                stream_id = destination.stream_id,
                                data_ptr = destination.data_ptr,
                                detail = %detail,
                                "native direct-write unavailable; falling back to staged-copy destination writer"
                            );
                            VideoDeviceWritePath::DirectWriteStagedCopyFallback
                        }
                    }
                } else {
                    VideoDeviceWritePath::TorchCopyScaffold
                };

                if write_path != VideoDeviceWritePath::DirectWriteNativeV1 {
                    self.write_payload_into_cuda_destination(
                        &cuda_payload,
                        &payload_view,
                        &stream,
                        &destination,
                        write_path,
                    )?;
                }

                if env_bool("MX8_VIDEO_EXPERIMENTAL_DEVICE_OUTPUT_ENFORCE_STREAM", true) {
                    let post_stream_id = cuda
                        .call_method0("current_stream")?
                        .getattr("cuda_stream")?
                        .extract::<u64>()?;
                    if post_stream_id != stream_id {
                        return Err(PyRuntimeError::new_err(format!(
                            "video device output stream changed during write (before={stream_id} after={post_stream_id})"
                        )));
                    }
                }

                tracing::info!(
                    target: "mx8_proof",
                    event = "video_experimental_device_output_write",
                    mode = write_path.mode_name(),
                    stream_id = destination.stream_id,
                    data_ptr = destination.data_ptr,
                    bytes = self.payload.len() as u64,
                    "video experimental device output wrote payload into torch-owned cuda tensor"
                );
                Ok(cuda_payload)
            })();
            match cuda_attempt {
                Ok(v) => v,
                Err(err) => {
                    tracing::warn!(
                        target: "mx8_proof",
                        event = "video_experimental_device_output_runtime_fallback",
                        detail = %err,
                        "video experimental device output runtime write failed; falling back to cpu tensor"
                    );
                    payload_view
                }
            }
        } else {
            payload_view
        };

        let mut offsets_i64 = Vec::with_capacity(self.offsets.len());
        for &off in self.offsets.iter() {
            offsets_i64.push(i64::try_from(off).map_err(|_| {
                PyValueError::new_err(format!(
                    "offset overflow converting u64 -> i64 (offset={off})"
                ))
            })?);
        }
        let offsets_list = PyList::new_bound(py, offsets_i64);
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("dtype", &torch_int64)?;
        let offsets_i64 = torch.call_method("tensor", (offsets_list,), Some(&kwargs))?;

        let mut sample_ids_i64 = Vec::with_capacity(self.sample_ids.len());
        for &sid in self.sample_ids.iter() {
            sample_ids_i64.push(i64::try_from(sid).map_err(|_| {
                PyValueError::new_err(format!(
                    "sample_id overflow converting u64 -> i64 (sample_id={sid})"
                ))
            })?);
        }
        let ids_list = PyList::new_bound(py, sample_ids_i64);
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("dtype", &torch_int64)?;
        let sample_ids_i64 = torch.call_method("tensor", (ids_list,), Some(&kwargs))?;

        Ok((payload_out, offsets_i64, sample_ids_i64))
    }
}

impl VideoDataLoader {
    const VIDEO_GPU_PRESSURE_MIN_SAMPLE_INTERVAL: std::time::Duration =
        std::time::Duration::from_secs(2);

    fn decode_failed_total(&self) -> u64 {
        self.decode_failed_io_read_failed
            .saturating_add(self.decode_failed_corrupt_media)
            .saturating_add(self.decode_failed_short_media)
            .saturating_add(self.decode_failed_unsupported_codec)
            .saturating_add(self.decode_failed_missing_stream)
            .saturating_add(self.decode_failed_backend_unavailable)
            .saturating_add(self.decode_failed_decode_failed)
    }

    fn sample_gpu_pressure_ratio() -> Result<f64, String> {
        if let Some(raw) = env_string("MX8_VIDEO_GPU_PRESSURE_RATIO") {
            let parsed = raw
                .trim()
                .parse::<f64>()
                .map_err(|_| format!("invalid MX8_VIDEO_GPU_PRESSURE_RATIO={raw:?}"))?;
            if !parsed.is_finite() || parsed < 0.0 {
                return Err(format!(
                    "invalid MX8_VIDEO_GPU_PRESSURE_RATIO={raw:?} (expected finite >= 0)"
                ));
            }
            return Ok(parsed.clamp(0.0, 2.0));
        }
        let smi_bin = std::env::var("MX8_NVIDIA_SMI_BIN").unwrap_or_else(|_| "nvidia-smi".into());
        let output = Command::new(&smi_bin)
            .arg("--query-gpu=memory.used,memory.total")
            .arg("--format=csv,noheader,nounits")
            .output()
            .map_err(|e| format!("failed to execute {smi_bin}: {e}"))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!(
                "{smi_bin} failed with status {}: {}",
                output.status,
                stderr.trim()
            ));
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut max_ratio: Option<f64> = None;
        for line in stdout
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
        {
            let mut parts = line.split(',').map(str::trim);
            let used = parts
                .next()
                .ok_or_else(|| format!("malformed {smi_bin} output line: {line}"))?
                .parse::<f64>()
                .map_err(|_| format!("invalid used-memory token in {smi_bin} output: {line}"))?;
            let total = parts
                .next()
                .ok_or_else(|| format!("malformed {smi_bin} output line: {line}"))?
                .parse::<f64>()
                .map_err(|_| format!("invalid total-memory token in {smi_bin} output: {line}"))?;
            if total <= 0.0 {
                continue;
            }
            let ratio = (used / total).clamp(0.0, 2.0);
            max_ratio = Some(max_ratio.map_or(ratio, |cur| cur.max(ratio)));
        }
        max_ratio.ok_or_else(|| format!("{smi_bin} returned no GPU memory rows"))
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
        let gpu_backend_selected = matches!(
            self.decode_backend,
            VideoDecodeBackend::Nvdec | VideoDecodeBackend::Auto
        );
        let mut gpu_ratio = 0.0f64;
        let mut gpu_ratio_available = false;
        if gpu_backend_selected {
            let now = Instant::now();
            let should_sample_gpu = match self.video_last_gpu_sample_at {
                None => true,
                Some(last) => {
                    now.saturating_duration_since(last)
                        >= Self::VIDEO_GPU_PRESSURE_MIN_SAMPLE_INTERVAL
                }
            };
            if should_sample_gpu {
                self.video_last_gpu_sample_at = Some(now);
                match Self::sample_gpu_pressure_ratio() {
                    Ok(ratio) => {
                        gpu_ratio = ratio;
                        gpu_ratio_available = true;
                        self.video_gpu_pressure_milli = (ratio * 1000.0).round() as u64;
                        self.video_gpu_pressure_available = true;
                        if ratio < 0.80 {
                            self.video_gpu_recovery_streak =
                                self.video_gpu_recovery_streak.saturating_add(1);
                        } else {
                            self.video_gpu_recovery_streak = 0;
                        }
                    }
                    Err(err) => {
                        self.video_gpu_pressure_unavailable_total =
                            self.video_gpu_pressure_unavailable_total.saturating_add(1);
                        self.video_gpu_pressure_milli = 0;
                        self.video_gpu_pressure_available = false;
                        self.video_gpu_recovery_streak = 0;
                        tracing::warn!(
                            target: "mx8_proof",
                            event = "video_gpu_pressure_unavailable",
                            decode_backend = video_decode_backend_name(self.decode_backend),
                            unavailable_total = self.video_gpu_pressure_unavailable_total,
                            detail = %err,
                            "gpu pressure signal unavailable for video autotune"
                        );
                    }
                }
            } else {
                gpu_ratio_available = self.video_gpu_pressure_available;
                gpu_ratio = self.video_gpu_pressure_milli as f64 / 1000.0;
            }
        } else {
            self.video_gpu_pressure_milli = 0;
            self.video_gpu_pressure_available = false;
            self.video_gpu_recovery_streak = 0;
            self.video_last_gpu_sample_at = None;
        }
        let mut pressure = rss_ratio.max(inflight_ratio);
        if gpu_ratio_available {
            pressure = pressure.max(gpu_ratio);
        }
        self.video_runtime_autotune_pressure_milli = (pressure * 1000.0).round() as u64;

        let min_inflight = (self.batch_size_samples as u64)
            .saturating_mul(self.decode_contract.clip_bytes)
            .max(1);
        let max_inflight = max_process_rss_bytes.max(min_inflight);
        let old_inflight = self.max_inflight_bytes;
        let mut next_inflight = old_inflight;
        let mut trigger = "none";
        let mut gpu_clamp = false;
        if gpu_backend_selected && !gpu_ratio_available {
            next_inflight = ((old_inflight as f64) * 0.75).round() as u64;
            trigger = "gpu_telemetry_unavailable";
            gpu_clamp = true;
        } else if gpu_ratio_available && gpu_ratio >= 0.97 {
            next_inflight = min_inflight;
            trigger = "gpu_hard_cut";
            gpu_clamp = true;
        } else if gpu_ratio_available && gpu_ratio >= 0.92 {
            next_inflight = ((old_inflight as f64) * 0.85).round() as u64;
            trigger = "gpu_soft_cut";
            gpu_clamp = true;
        } else if rss_ratio > 0.92 {
            next_inflight = ((old_inflight as f64) * 0.85).round() as u64;
            trigger = "cpu_soft_cut";
        } else {
            let can_scale_up = if gpu_backend_selected {
                gpu_ratio_available && self.video_gpu_recovery_streak >= 3
            } else {
                true
            };
            if can_scale_up && pressure < 0.60 {
                next_inflight = ((old_inflight as f64) * 1.05).round() as u64;
                trigger = if gpu_backend_selected {
                    "gpu_recovery_scale_up"
                } else {
                    "cpu_scale_up"
                };
            }
        }
        next_inflight = next_inflight.clamp(min_inflight, max_inflight);
        if gpu_clamp {
            self.video_runtime_autotune_gpu_clamps_total = self
                .video_runtime_autotune_gpu_clamps_total
                .saturating_add(1);
        }
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
                inflight_ratio = inflight_ratio,
                gpu_ratio = if gpu_ratio_available { gpu_ratio } else { -1.0 },
                gpu_recovery_streak = self.video_gpu_recovery_streak,
                trigger = trigger,
                max_process_rss_bytes = max_process_rss_bytes,
                old_max_inflight_bytes = old_inflight,
                new_max_inflight_bytes = next_inflight,
                adjustments_total = self.video_runtime_autotune_adjustments_total,
                gpu_clamps_total = self.video_runtime_autotune_gpu_clamps_total,
                "video runtime autotune adjusted max inflight"
            );
        }
    }

    pub(crate) fn derive_decode_contract(
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

    fn decode_clip_from_path_with_ffmpeg(
        &self,
        path: &std::path::Path,
        start_seconds: f64,
        ffmpeg_input_args: &[&str],
        vf_arg: &str,
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
        let vf_arg = if vf_arg.is_empty() {
            format!("scale={side}:{side}:flags=bilinear")
        } else {
            vf_arg.to_string()
        };
        let ffmpeg_bin = std::env::var("MX8_FFMPEG_BIN").unwrap_or_else(|_| "ffmpeg".to_string());

        let mut cmd = Command::new(&ffmpeg_bin);
        cmd.arg("-hide_banner")
            .arg("-loglevel")
            .arg("error")
            .arg("-nostdin");
        for arg in ffmpeg_input_args {
            cmd.arg(arg);
        }
        let output = cmd
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
            .arg(vf_arg)
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

    fn decode_clip_from_path_cli(
        &self,
        path: &std::path::Path,
        start_seconds: f64,
    ) -> Result<Vec<u8>, VideoDecodeError> {
        self.decode_clip_from_path_with_ffmpeg(path, start_seconds, &[], "")
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

    #[cfg(mx8_video_nvdec)]
    fn decode_clip_from_path_nvdec(
        &self,
        path: &std::path::Path,
        start_seconds: f64,
    ) -> Result<Vec<u8>, VideoDecodeError> {
        // Decode on CUDA when available, then download to host and normalize to rgb24.
        let side = self.decode_contract.frame_width;
        let vf_arg =
            format!("hwdownload,format=nv12,scale={side}:{side}:flags=bilinear,format=rgb24");
        let err = match self.decode_clip_from_path_with_ffmpeg(
            path,
            start_seconds,
            &["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"],
            &vf_arg,
        ) {
            Ok(v) => return Ok(v),
            Err(err) => err,
        };
        let lower = err.detail.to_ascii_lowercase();
        if lower.contains("cuda")
            || lower.contains("cuvid")
            || lower.contains("hwaccel")
            || lower.contains("cannot load libcuda")
            || lower.contains("no device")
            || lower.contains("device type cuda needed for codec")
            || lower.contains("operation not permitted")
            || lower.contains("not compiled")
        {
            return Err(Self::decode_error(
                path,
                "decode_backend_unavailable",
                &err.detail,
            ));
        }
        Err(err)
    }

    #[cfg(not(mx8_video_nvdec))]
    fn decode_clip_from_path_nvdec(
        &self,
        path: &std::path::Path,
        _start_seconds: f64,
    ) -> Result<Vec<u8>, VideoDecodeError> {
        Err(Self::decode_error(
            path,
            "decode_backend_unavailable",
            "nvdec backend not compiled (rebuild with RUSTFLAGS='--cfg mx8_video_nvdec')",
        ))
    }

    fn decode_clip_with_backend(
        &self,
        backend: VideoDecodeBackend,
        path: &std::path::Path,
        start_seconds: f64,
    ) -> Result<Vec<u8>, VideoDecodeError> {
        match backend {
            VideoDecodeBackend::Cli => self.decode_clip_from_path_cli(path, start_seconds),
            VideoDecodeBackend::Ffi => self.decode_clip_from_path_ffi(path, start_seconds),
            VideoDecodeBackend::Nvdec => self.decode_clip_from_path_nvdec(path, start_seconds),
            VideoDecodeBackend::Auto => Err(Self::decode_error(
                path,
                "decode_failed",
                "auto backend must be resolved via fallback chain",
            )),
        }
    }

    fn decode_clip_with_fallback_chain(
        &mut self,
        path: &std::path::Path,
        start_seconds: f64,
        backends: &[VideoDecodeBackend],
    ) -> Result<Vec<u8>, VideoDecodeError> {
        for (idx, backend) in backends.iter().enumerate() {
            match self.decode_clip_with_backend(*backend, path, start_seconds) {
                Ok(v) => return Ok(v),
                Err(err) => {
                    if let Some(next_backend) = backends.get(idx + 1).copied() {
                        self.decode_backend_fallback_total =
                            self.decode_backend_fallback_total.saturating_add(1);
                        tracing::warn!(
                            target: "mx8_proof",
                            event = "video_decode_backend_fallback",
                            from_backend = video_decode_backend_name(*backend),
                            to_backend = video_decode_backend_name(next_backend),
                            fallback_total = self.decode_backend_fallback_total,
                            class = err.class,
                            path = %path.display(),
                            detail = %err.detail,
                            "video decode backend failed; falling back"
                        );
                        continue;
                    }
                    return Err(err);
                }
            }
        }
        Err(Self::decode_error(
            path,
            "decode_failed",
            "empty decode backend fallback chain",
        ))
    }

    fn decode_clip_from_path(
        &mut self,
        path: &std::path::Path,
        start_seconds: f64,
    ) -> Result<Vec<u8>, VideoDecodeError> {
        match self.decode_backend {
            VideoDecodeBackend::Cli => self.decode_clip_from_path_cli(path, start_seconds),
            VideoDecodeBackend::Ffi => self.decode_clip_with_fallback_chain(
                path,
                start_seconds,
                &[VideoDecodeBackend::Ffi, VideoDecodeBackend::Cli],
            ),
            VideoDecodeBackend::Nvdec => self.decode_clip_with_fallback_chain(
                path,
                start_seconds,
                &[
                    VideoDecodeBackend::Nvdec,
                    VideoDecodeBackend::Ffi,
                    VideoDecodeBackend::Cli,
                ],
            ),
            VideoDecodeBackend::Auto => self.decode_clip_with_fallback_chain(
                path,
                start_seconds,
                &[
                    VideoDecodeBackend::Nvdec,
                    VideoDecodeBackend::Ffi,
                    VideoDecodeBackend::Cli,
                ],
            ),
        }
    }

    pub(crate) fn stage2d_sidecar_key(media_uri: &str, stream_id: u32) -> String {
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
        if clip.media_uri.starts_with("gs://") {
            return self.clip_payload_bytes_gcs(clip);
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

    fn gcs_fetch_ranges_bytes(
        &mut self,
        clip: &VideoClipRecord,
        bucket: &str,
        key: &str,
        ranges: &[ByteRange],
    ) -> Result<Vec<u8>, VideoDecodeError> {
        let client = self
            .rt
            .block_on(mx8_runtime::gcs::client_from_env())
            .map_err(|e| {
                Self::decode_error(
                    std::path::Path::new(&clip.media_uri),
                    "io_read_failed",
                    &format!("init gcs client failed: {e}"),
                )
            })?;

        use google_cloud_storage::http::objects::download::Range as GcsRange;
        use google_cloud_storage::http::objects::get::GetObjectRequest as GcsGetReq;

        let mut payload = Vec::<u8>::new();
        for range in ranges {
            let end_inclusive = range.end_byte.checked_sub(1).ok_or_else(|| {
                Self::decode_error(
                    std::path::Path::new(&clip.media_uri),
                    "decode_failed",
                    "invalid empty stage2d byte range",
                )
            })?;
            let bytes = self
                .rt
                .block_on(async {
                    client
                        .download_object(
                            &GcsGetReq {
                                bucket: bucket.to_string(),
                                object: key.to_string(),
                                ..Default::default()
                            },
                            &GcsRange(Some(range.start_byte), Some(end_inclusive)),
                        )
                        .await
                        .map_err(|e| e.to_string())
                })
                .map_err(|e| {
                    Self::decode_error(
                        std::path::Path::new(&clip.media_uri),
                        "io_read_failed",
                        &format!("gcs range download failed: {e}"),
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
                        "gcs range returned {} bytes, expected {}",
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
            "fetched gcs ranges for video clip"
        );
        Ok(payload)
    }

    fn clip_payload_bytes_gcs(
        &mut self,
        clip: &VideoClipRecord,
    ) -> Result<Vec<u8>, VideoDecodeError> {
        let (bucket, key) = parse_gcs_bucket_key_local(&clip.media_uri).map_err(|e| {
            Self::decode_error(
                std::path::Path::new(&clip.media_uri),
                "decode_failed",
                &format!("invalid gs uri: {e}"),
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
                        "planned stage2d gcs ranges for video clip"
                    );
                    if let Ok(bytes) =
                        self.gcs_fetch_ranges_bytes(clip, &bucket, &key, &plan.ranges)
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
            .block_on(mx8_runtime::gcs::client_from_env())
            .map_err(|e| {
                Self::decode_error(
                    std::path::Path::new(&clip.media_uri),
                    "io_read_failed",
                    &format!("init gcs client failed: {e}"),
                )
            })?;
        use google_cloud_storage::http::objects::get::GetObjectRequest as GcsGetReq;
        let meta = self
            .rt
            .block_on(async {
                client
                    .get_object(&GcsGetReq {
                        bucket: bucket.clone(),
                        object: key.clone(),
                        ..Default::default()
                    })
                    .await
            })
            .map_err(|e| {
                Self::decode_error(
                    std::path::Path::new(&clip.media_uri),
                    "io_read_failed",
                    &format!("gcs get_object failed: {e:?}"),
                )
            })?;
        let object_len = meta.size;
        if object_len <= 0 {
            return Err(Self::decode_error(
                std::path::Path::new(&clip.media_uri),
                "io_read_failed",
                "gcs object has unknown/zero size",
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
            "using full-object gcs range fallback for video clip"
        );
        let bytes = self.gcs_fetch_ranges_bytes(clip, &bucket, &key, &full_range)?;
        let tmp_path = std::env::temp_dir().join(format!(
            "mx8-video-gcs-full-{}-{}-{}.mp4",
            std::process::id(),
            unix_time_ms(),
            clip.sample_id
        ));
        std::fs::write(&tmp_path, &bytes).map_err(|e| {
            Self::decode_error(
                std::path::Path::new(&clip.media_uri),
                "io_read_failed",
                &format!("write gcs temp file failed: {e}"),
            )
        })?;
        let seek_seconds = (clip.clip_start as f64) / f64::from(self.decode_fps.max(1));
        let decoded = self.decode_clip_from_path(&tmp_path, seek_seconds);
        let _ = std::fs::remove_file(&tmp_path);
        decoded
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
        if self.video_experimental_device_output_requested
            && !self.video_experimental_device_output_active
        {
            self.video_experimental_device_output_fallback_total = self
                .video_experimental_device_output_fallback_total
                .saturating_add(1);
        }
        if self.video_experimental_device_direct_write_requested
            && !self.video_experimental_device_direct_write_active
        {
            self.video_experimental_device_direct_write_fallback_total = self
                .video_experimental_device_direct_write_fallback_total
                .saturating_add(1);
        }
        if self.video_experimental_device_direct_write_active {
            self.video_experimental_device_direct_write_batches_total = self
                .video_experimental_device_direct_write_batches_total
                .saturating_add(1);
        }
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
                experimental_device_output_active: self.video_experimental_device_output_active,
                experimental_device_direct_write_active: self
                    .video_experimental_device_direct_write_active,
            },
        )?;
        Ok(out.into_bound(py).into_any())
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let out = PyDict::new_bound(py);
        out.set_item("manifest_hash", &self.manifest_hash)?;
        out.set_item("assigned_rank", self.assigned_rank)?;
        out.set_item("world_size", self.world_size)?;
        out.set_item("job_id", self.job_id.as_deref())?;
        out.set_item("cluster_url", self.cluster_url.as_deref())?;
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
            "process_rss_bytes",
            sample_process_rss_bytes_local().unwrap_or(0),
        )?;
        out.set_item("max_process_rss_bytes", self.video_max_process_rss_bytes)?;
        out.set_item("elapsed_seconds", self.started_at.elapsed().as_secs_f64())?;
        out.set_item(
            "video_runtime_autotune_enabled",
            self.video_runtime_autotune_enabled,
        )?;
        out.set_item(
            "video_runtime_autotune_pressure",
            self.video_runtime_autotune_pressure_milli as f64 / 1000.0,
        )?;
        out.set_item(
            "video_experimental_device_output_requested",
            self.video_experimental_device_output_requested,
        )?;
        out.set_item(
            "video_experimental_device_output_active",
            self.video_experimental_device_output_active,
        )?;
        out.set_item(
            "video_experimental_device_output_fallback_total",
            self.video_experimental_device_output_fallback_total,
        )?;
        out.set_item(
            "video_experimental_device_direct_write_requested",
            self.video_experimental_device_direct_write_requested,
        )?;
        out.set_item(
            "video_experimental_device_direct_write_active",
            self.video_experimental_device_direct_write_active,
        )?;
        out.set_item(
            "video_experimental_device_direct_write_fallback_total",
            self.video_experimental_device_direct_write_fallback_total,
        )?;
        out.set_item(
            "video_experimental_device_direct_write_batches_total",
            self.video_experimental_device_direct_write_batches_total,
        )?;
        out.set_item(
            "video_runtime_autotune_adjustments_total",
            self.video_runtime_autotune_adjustments_total,
        )?;
        out.set_item(
            "video_runtime_autotune_gpu_clamps_total",
            self.video_runtime_autotune_gpu_clamps_total,
        )?;
        out.set_item(
            "video_gpu_pressure",
            self.video_gpu_pressure_milli as f64 / 1000.0,
        )?;
        out.set_item(
            "video_gpu_pressure_unavailable_total",
            self.video_gpu_pressure_unavailable_total,
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
        out.set_item(
            "video_decode_backend_fallback_total",
            self.decode_backend_fallback_total,
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

    fn checkpoint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let token = VideoLoaderCheckpointToken {
            manifest_hash: self.manifest_hash.clone(),
            seed: self.seed,
            epoch: self.epoch,
            clip_len: self.clip_len,
            stride: self.stride,
            fps: self.decode_fps,
            next_idx: self.next_idx as u64,
            clips_total: self.clips.len() as u64,
            assigned_rank: self.assigned_rank,
            world_size: self.world_size,
        };
        Ok(PyBytes::new_bound(py, &token.encode()))
    }
}
