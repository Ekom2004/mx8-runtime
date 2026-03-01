use super::*;

pub(crate) enum AudioLoaderInner {
    Local(DataLoader),
    Distributed(DistributedDataLoader),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AudioDecodeErrorPolicy {
    Error,
    Skip,
}

impl AudioDecodeErrorPolicy {
    pub(crate) fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "error" => Ok(Self::Error),
            "skip" => Ok(Self::Skip),
            _ => anyhow::bail!("invalid decode_error_policy={raw:?} (expected: error|skip)"),
        }
    }
}

#[pyclass]
pub(crate) struct AudioLoader {
    pub(crate) loader: AudioLoaderInner,
    pub(crate) sample_count: usize,
    pub(crate) channels: usize,
    pub(crate) sample_rate_hz: Option<u32>,
    pub(crate) decode_error_policy: AudioDecodeErrorPolicy,
    pub(crate) decoded_samples_total: Arc<AtomicU64>,
    pub(crate) decode_failures_total: Arc<AtomicU64>,
    pub(crate) decoded_frames_total: Arc<AtomicU64>,
}

#[pyclass]
pub(crate) struct AudioBatch {
    samples: Vec<f32>,
    sample_rates_hz: Vec<i64>,
    sample_ids: Vec<i64>,
    batch_rows: usize,
    sample_count: usize,
}

pub(crate) fn pcm_int_to_f32(sample: i64, bits_per_sample: u32) -> Result<f32> {
    if bits_per_sample == 0 || bits_per_sample > 32 {
        anyhow::bail!("unsupported PCM bits_per_sample={bits_per_sample}");
    }
    let max_mag = (1i64 << (bits_per_sample - 1)) as f32;
    if max_mag <= 0.0 {
        anyhow::bail!("invalid PCM bits_per_sample={bits_per_sample}");
    }
    Ok(((sample as f32) / max_mag).clamp(-1.0, 1.0))
}

pub(crate) fn downmix_interleaved_to_mono(
    interleaved: &[f32],
    channels: usize,
) -> Result<Vec<f32>> {
    if channels == 0 {
        anyhow::bail!("audio reports zero channels");
    }
    if !interleaved.len().is_multiple_of(channels) {
        anyhow::bail!(
            "interleaved sample length {} is not divisible by channels {}",
            interleaved.len(),
            channels
        );
    }
    if channels == 1 {
        return Ok(interleaved.to_vec());
    }

    let frames = interleaved.len() / channels;
    let mut mono = Vec::<f32>::with_capacity(frames);
    for frame in 0..frames {
        let base = frame * channels;
        let mut acc = 0.0f32;
        for c in 0..channels {
            acc += interleaved[base + c];
        }
        mono.push((acc / (channels as f32)).clamp(-1.0, 1.0));
    }
    Ok(mono)
}

pub(crate) fn decode_wav_audio(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    let cursor = std::io::Cursor::new(bytes);
    let mut reader = hound::WavReader::new(cursor)?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels);
    if channels == 0 {
        anyhow::bail!("wav audio reports zero channels");
    }
    let bits = u32::from(spec.bits_per_sample);
    let sample_rate_hz = spec.sample_rate;

    let interleaved = match spec.sample_format {
        hound::SampleFormat::Int => {
            if bits == 0 || bits > 32 {
                anyhow::bail!("unsupported wav PCM bits_per_sample={bits}");
            }
            let mut samples = Vec::<f32>::new();
            if bits <= 8 {
                for sample in reader.samples::<i8>() {
                    samples.push(pcm_int_to_f32(i64::from(sample?), bits)?);
                }
            } else if bits <= 16 {
                for sample in reader.samples::<i16>() {
                    samples.push(pcm_int_to_f32(i64::from(sample?), bits)?);
                }
            } else {
                for sample in reader.samples::<i32>() {
                    samples.push(pcm_int_to_f32(i64::from(sample?), bits)?);
                }
            }
            samples
        }
        hound::SampleFormat::Float => {
            if spec.bits_per_sample != 32 {
                anyhow::bail!(
                    "unsupported wav float bits_per_sample={} (expected 32)",
                    spec.bits_per_sample
                );
            }
            let mut samples = Vec::<f32>::new();
            for sample in reader.samples::<f32>() {
                samples.push(sample?.clamp(-1.0, 1.0));
            }
            samples
        }
    };
    let mono = downmix_interleaved_to_mono(&interleaved, channels)?;
    Ok((mono, sample_rate_hz))
}

pub(crate) fn decode_flac_audio(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    let cursor = std::io::Cursor::new(bytes);
    let mut reader = claxon::FlacReader::new(cursor)?;
    let streaminfo = reader.streaminfo();
    let channels = usize::try_from(streaminfo.channels).map_err(|_| {
        anyhow::anyhow!(
            "flac channel count does not fit usize: {}",
            streaminfo.channels
        )
    })?;
    if channels == 0 {
        anyhow::bail!("flac audio reports zero channels");
    }
    let bits = streaminfo.bits_per_sample;
    let sample_rate_hz = streaminfo.sample_rate;
    let mut interleaved = Vec::<f32>::new();
    for sample in reader.samples() {
        interleaved.push(pcm_int_to_f32(i64::from(sample?), bits)?);
    }
    let mono = downmix_interleaved_to_mono(&interleaved, channels)?;
    Ok((mono, sample_rate_hz))
}

pub(crate) fn decode_audio_mono_f32(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WAVE" {
        return decode_wav_audio(bytes);
    }
    if bytes.starts_with(b"fLaC") {
        return decode_flac_audio(bytes);
    }
    anyhow::bail!("unsupported audio format (expected WAV or FLAC)")
}

impl AudioLoader {
    fn decode_lease(&self, lease: BatchLease) -> PyResult<AudioBatch> {
        let mut samples = Vec::<f32>::new();
        let mut sample_rates_hz = Vec::<i64>::new();
        let mut sample_ids = Vec::<i64>::new();
        let mut rows = 0usize;

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
            let sample_id = lease.batch.sample_ids[i];
            let sample_id_i64 = i64::try_from(sample_id).map_err(|_| {
                PyValueError::new_err(format!(
                    "sample_id overflow converting u64 -> i64 (sample_id={sample_id})"
                ))
            })?;
            let bytes = &lease.batch.payload[start..end];
            let (mut mono, sample_rate_hz) = match decode_audio_mono_f32(bytes) {
                Ok(decoded) => decoded,
                Err(err) => {
                    self.decode_failures_total.fetch_add(1, Ordering::Relaxed);
                    match self.decode_error_policy {
                        AudioDecodeErrorPolicy::Error => {
                            return Err(PyRuntimeError::new_err(format!(
                                "audio decode failed for sample_id={sample_id}: {err}"
                            )));
                        }
                        AudioDecodeErrorPolicy::Skip => continue,
                    }
                }
            };

            if let Some(expected_hz) = self.sample_rate_hz {
                if sample_rate_hz != expected_hz {
                    self.decode_failures_total.fetch_add(1, Ordering::Relaxed);
                    match self.decode_error_policy {
                        AudioDecodeErrorPolicy::Error => {
                            return Err(PyRuntimeError::new_err(format!(
                                "audio sample_rate mismatch for sample_id={sample_id} (decoded={} expected={})",
                                sample_rate_hz, expected_hz
                            )));
                        }
                        AudioDecodeErrorPolicy::Skip => continue,
                    }
                }
            }

            self.decoded_samples_total.fetch_add(1, Ordering::Relaxed);
            self.decoded_frames_total
                .fetch_add(mono.len() as u64, Ordering::Relaxed);

            if mono.len() > self.sample_count {
                mono.truncate(self.sample_count);
            } else if mono.len() < self.sample_count {
                mono.resize(self.sample_count, 0.0);
            }

            samples.extend_from_slice(&mono);
            sample_rates_hz.push(i64::from(sample_rate_hz));
            sample_ids.push(sample_id_i64);
            rows = rows.saturating_add(1);
        }

        Ok(AudioBatch {
            samples,
            sample_rates_hz,
            sample_ids,
            batch_rows: rows,
            sample_count: self.sample_count,
        })
    }
}

#[pymethods]
impl AudioLoader {
    #[getter]
    fn sample_count(&self) -> usize {
        self.sample_count
    }

    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    #[getter]
    fn sample_rate_hz(&self) -> Option<u32> {
        self.sample_rate_hz
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stats = match &self.loader {
            AudioLoaderInner::Local(loader) => loader.stats(py)?,
            AudioLoaderInner::Distributed(loader) => loader.stats(py)?,
        };
        let dict = stats.downcast::<PyDict>()?;
        dict.set_item("audio_sample_count", self.sample_count)?;
        dict.set_item("audio_channels", self.channels)?;
        dict.set_item("audio_expected_sample_rate_hz", self.sample_rate_hz)?;
        dict.set_item(
            "audio_decode_samples_total",
            self.decoded_samples_total.load(Ordering::Relaxed),
        )?;
        dict.set_item(
            "audio_decode_failures_total",
            self.decode_failures_total.load(Ordering::Relaxed),
        )?;
        dict.set_item(
            "audio_decoded_frames_total",
            self.decoded_frames_total.load(Ordering::Relaxed),
        )?;
        Ok(stats)
    }

    fn checkpoint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        match &self.loader {
            AudioLoaderInner::Local(loader) => loader.checkpoint(py),
            AudioLoaderInner::Distributed(loader) => loader.checkpoint(py),
        }
    }

    fn print_stats(&self, py: Python<'_>) -> PyResult<()> {
        match &self.loader {
            AudioLoaderInner::Local(loader) => loader.print_stats(py),
            AudioLoaderInner::Distributed(loader) => {
                let stats = loader.stats(py)?;
                let stats = stats.downcast::<PyDict>()?;
                let text = render_human_stats(stats).replace('\n', " | ");
                eprintln!("[mx8] {text}");
                Ok(())
            }
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        loop {
            let lease = match &mut self.loader {
                AudioLoaderInner::Local(loader) => {
                    let lease =
                        py.allow_threads(|| loader.rt.block_on(async { loader.rx.recv().await }));
                    let Some(lease) = lease else {
                        let Some(task) = loader.task.take() else {
                            return Err(PyStopIteration::new_err(()));
                        };
                        let out = py.allow_threads(|| loader.rt.block_on(task));
                        return match out {
                            Ok(Ok(())) => Err(PyStopIteration::new_err(())),
                            Ok(Err(err)) => Err(PyRuntimeError::new_err(format!("{err}"))),
                            Err(err) => Err(PyRuntimeError::new_err(format!(
                                "producer task failed: {err}"
                            ))),
                        };
                    };
                    loader.on_batch_delivered(&lease);
                    lease
                }
                AudioLoaderInner::Distributed(loader) => {
                    let wait_started = Instant::now();
                    let lease =
                        py.allow_threads(|| loader.rt.block_on(async { loader.rx.recv().await }));
                    let Some(lease) = lease else {
                        let Some(task) = loader.task.take() else {
                            return Err(PyStopIteration::new_err(()));
                        };
                        let out = py.allow_threads(|| loader.rt.block_on(task));
                        return match out {
                            Ok(Ok(())) => Err(PyStopIteration::new_err(())),
                            Ok(Err(err)) => Err(PyRuntimeError::new_err(format!("{err}"))),
                            Err(err) => Err(PyRuntimeError::new_err(format!(
                                "producer task failed: {err}"
                            ))),
                        };
                    };
                    loader.autotune.on_wait(wait_started.elapsed());
                    lease
                }
            };
            let batch = self.decode_lease(lease)?;
            if batch.batch_rows == 0 {
                continue;
            }
            let out = Py::new(py, batch)?;
            return Ok(out.into_bound(py).into_any());
        }
    }

    fn close(&mut self) {
        match &mut self.loader {
            AudioLoaderInner::Local(loader) => loader.close(),
            AudioLoaderInner::Distributed(loader) => loader.close(),
        }
    }

    fn __del__(&mut self) {
        self.close();
    }
}

#[pymethods]
impl AudioBatch {
    #[getter]
    fn sample_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let torch = py.import_bound("torch").map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to import torch (install PyTorch to use AudioBatch): {e}"
            ))
        })?;
        let torch_int64 = torch.getattr("int64")?;
        let values = PyList::new_bound(py, self.sample_ids.iter().copied());
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("dtype", &torch_int64)?;
        torch.call_method("tensor", (values,), Some(&kwargs))
    }

    #[getter]
    fn sample_rates_hz<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let torch = py.import_bound("torch").map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to import torch (install PyTorch to use AudioBatch): {e}"
            ))
        })?;
        let torch_int64 = torch.getattr("int64")?;
        let values = PyList::new_bound(py, self.sample_rates_hz.iter().copied());
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("dtype", &torch_int64)?;
        torch.call_method("tensor", (values,), Some(&kwargs))
    }

    #[getter]
    fn samples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let torch = py.import_bound("torch").map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to import torch (install PyTorch to use AudioBatch): {e}"
            ))
        })?;
        let torch_float32 = torch.getattr("float32")?;
        let values = PyList::new_bound(py, self.samples.iter().copied());
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("dtype", &torch_float32)?;
        let flat = torch.call_method("tensor", (values,), Some(&kwargs))?;
        let rows = i64::try_from(self.batch_rows)
            .map_err(|_| PyValueError::new_err("audio batch rows do not fit i64"))?;
        let sample_count = i64::try_from(self.sample_count)
            .map_err(|_| PyValueError::new_err("audio sample_count does not fit i64"))?;
        flat.call_method1("view", (rows, sample_count))
    }

    fn to_torch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let samples = self.samples(py)?;
        let sample_rates_hz = self.sample_rates_hz(py)?;
        let sample_ids = self.sample_ids(py)?;
        let out = PyTuple::new_bound(
            py,
            [
                samples.to_object(py),
                sample_rates_hz.to_object(py),
                sample_ids.to_object(py),
            ],
        );
        Ok(out.into_any())
    }
}
