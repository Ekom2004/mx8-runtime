use super::*;

pub(crate) enum TextLoaderInner {
    Local(DataLoader),
    Distributed(DistributedDataLoader),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TextTruncateMode {
    Right,
    Error,
}

impl TextTruncateMode {
    pub(crate) fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "right" => Ok(Self::Right),
            "error" => Ok(Self::Error),
            _ => anyhow::bail!("invalid truncate={raw:?} (expected: right|error)"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TextDecodeErrorPolicy {
    Error,
    Skip,
}

impl TextDecodeErrorPolicy {
    pub(crate) fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "error" => Ok(Self::Error),
            "skip" => Ok(Self::Skip),
            _ => anyhow::bail!("invalid decode_error_policy={raw:?} (expected: error|skip)"),
        }
    }
}

#[pyclass]
pub(crate) struct TextLoader {
    pub(crate) loader: TextLoaderInner,
    pub(crate) tokenizer: Arc<Tokenizer>,
    pub(crate) sequence_length: usize,
    pub(crate) stride: usize,
    pub(crate) add_bos: bool,
    pub(crate) add_eos: bool,
    pub(crate) bos_token_id: Option<u32>,
    pub(crate) eos_token_id: Option<u32>,
    pub(crate) pad_token_id: i64,
    pub(crate) truncate: TextTruncateMode,
    pub(crate) return_attention_mask: bool,
    pub(crate) decode_error_policy: TextDecodeErrorPolicy,
}

#[pyclass]
pub(crate) struct TextBatch {
    token_ids: Vec<i64>,
    attention_mask: Option<Vec<u8>>,
    sample_ids: Vec<i64>,
    batch_rows: usize,
    sequence_length: usize,
}

pub(crate) fn resolve_special_token_id(tokenizer: &Tokenizer, names: &[&str]) -> Option<u32> {
    names.iter().find_map(|name| tokenizer.token_to_id(name))
}

pub(crate) fn text_pad_token_id(tokenizer: &Tokenizer, eos_token_id: Option<u32>) -> i64 {
    if let Some(padding) = tokenizer.get_padding() {
        return i64::from(padding.pad_id);
    }
    eos_token_id.map(i64::from).unwrap_or(0)
}

pub(crate) fn load_text_tokenizer(tokenizer_ref: &str) -> PyResult<Tokenizer> {
    let trimmed = tokenizer_ref.trim();
    if trimmed.is_empty() {
        return Err(PyValueError::new_err(
            "tokenizer must be non-empty (expected tokenizer preset like \"gpt2\" or tokenizer.json path)",
        ));
    }
    if std::path::Path::new(trimmed).exists() {
        return Tokenizer::from_file(trimmed).map_err(|e| {
            PyValueError::new_err(format!("failed to load tokenizer file {trimmed:?}: {e}"))
        });
    }
    match trimmed.to_ascii_lowercase().as_str() {
        "gpt2" => Tokenizer::from_pretrained("gpt2", None)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to load gpt2 tokenizer: {e}"))),
        _ => Err(PyValueError::new_err(format!(
            "unsupported tokenizer {trimmed:?}; pass \"gpt2\" or a tokenizer.json path"
        ))),
    }
}

impl TextLoader {
    fn tokenize_lease(&self, lease: BatchLease) -> PyResult<TextBatch> {
        let mut token_ids = Vec::<i64>::new();
        let mut attention_mask = if self.return_attention_mask {
            Some(Vec::<u8>::new())
        } else {
            None
        };
        let mut sample_ids = Vec::<i64>::new();
        let mut rows = 0usize;
        let step = self.stride.max(1);

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
            let text_bytes = &lease.batch.payload[start..end];
            let text = match std::str::from_utf8(text_bytes) {
                Ok(v) => v,
                Err(err) => match self.decode_error_policy {
                    TextDecodeErrorPolicy::Error => {
                        return Err(PyValueError::new_err(format!(
                            "utf8 decode failed for sample_id={sample_id}: {err}"
                        )));
                    }
                    TextDecodeErrorPolicy::Skip => continue,
                },
            };

            let encoding = self.tokenizer.encode(text, false).map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "tokenization failed for sample_id={sample_id}: {e}"
                ))
            })?;
            let mut ids: Vec<i64> = encoding.get_ids().iter().map(|v| i64::from(*v)).collect();
            if self.add_bos {
                let bos = self.bos_token_id.ok_or_else(|| {
                    PyValueError::new_err("add_bos=True but tokenizer has no bos token id")
                })?;
                ids.insert(0, i64::from(bos));
            }
            if self.add_eos {
                let eos = self.eos_token_id.ok_or_else(|| {
                    PyValueError::new_err("add_eos=True but tokenizer has no eos token id")
                })?;
                ids.push(i64::from(eos));
            }

            if self.truncate == TextTruncateMode::Error && ids.len() > self.sequence_length {
                return Err(PyValueError::new_err(format!(
                    "token sequence too long for sample_id={sample_id} (tokens={} > sequence_length={})",
                    ids.len(),
                    self.sequence_length
                )));
            }

            if ids.is_empty() {
                sample_ids.push(sample_id_i64);
                token_ids.extend(std::iter::repeat_n(self.pad_token_id, self.sequence_length));
                if let Some(mask) = attention_mask.as_mut() {
                    mask.extend(std::iter::repeat_n(0u8, self.sequence_length));
                }
                rows = rows.saturating_add(1);
                continue;
            }

            let mut cursor = 0usize;
            loop {
                let end_idx = cursor.saturating_add(self.sequence_length).min(ids.len());
                let window = &ids[cursor..end_idx];
                sample_ids.push(sample_id_i64);
                token_ids.extend_from_slice(window);
                if window.len() < self.sequence_length {
                    token_ids.extend(std::iter::repeat_n(
                        self.pad_token_id,
                        self.sequence_length - window.len(),
                    ));
                }
                if let Some(mask) = attention_mask.as_mut() {
                    mask.extend(std::iter::repeat_n(1u8, window.len()));
                    if window.len() < self.sequence_length {
                        mask.extend(std::iter::repeat_n(
                            0u8,
                            self.sequence_length - window.len(),
                        ));
                    }
                }
                rows = rows.saturating_add(1);

                if end_idx >= ids.len() || self.truncate == TextTruncateMode::Error {
                    break;
                }
                cursor = cursor.saturating_add(step);
                if cursor >= ids.len() {
                    break;
                }
            }
        }

        Ok(TextBatch {
            token_ids,
            attention_mask,
            sample_ids,
            batch_rows: rows,
            sequence_length: self.sequence_length,
        })
    }
}

#[pymethods]
impl TextLoader {
    #[getter]
    fn sequence_length(&self) -> usize {
        self.sequence_length
    }

    #[getter]
    fn stride(&self) -> usize {
        self.stride
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stats = match &self.loader {
            TextLoaderInner::Local(loader) => loader.stats(py)?,
            TextLoaderInner::Distributed(loader) => loader.stats(py)?,
        };
        let dict = stats.downcast::<PyDict>()?;
        dict.set_item("text_sequence_length", self.sequence_length)?;
        dict.set_item("text_stride", self.stride)?;
        dict.set_item("text_return_attention_mask", self.return_attention_mask)?;
        Ok(stats)
    }

    fn checkpoint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        match &self.loader {
            TextLoaderInner::Local(loader) => loader.checkpoint(py),
            TextLoaderInner::Distributed(loader) => loader.checkpoint(py),
        }
    }

    fn print_stats(&self, py: Python<'_>) -> PyResult<()> {
        match &self.loader {
            TextLoaderInner::Local(loader) => loader.print_stats(py),
            TextLoaderInner::Distributed(loader) => {
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
                TextLoaderInner::Local(loader) => {
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
                TextLoaderInner::Distributed(loader) => {
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
            let batch = self.tokenize_lease(lease)?;
            if batch.batch_rows == 0 {
                continue;
            }
            let out = Py::new(py, batch)?;
            return Ok(out.into_bound(py).into_any());
        }
    }

    fn close(&mut self) {
        match &mut self.loader {
            TextLoaderInner::Local(loader) => loader.close(),
            TextLoaderInner::Distributed(loader) => loader.close(),
        }
    }

    fn __del__(&mut self) {
        self.close();
    }
}

#[pymethods]
impl TextBatch {
    #[getter]
    fn sample_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let torch = py.import_bound("torch").map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to import torch (install PyTorch to use TextBatch): {e}"
            ))
        })?;
        let torch_int64 = torch.getattr("int64")?;
        let values = PyList::new_bound(py, self.sample_ids.iter().copied());
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("dtype", &torch_int64)?;
        torch.call_method("tensor", (values,), Some(&kwargs))
    }

    #[getter]
    fn token_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let torch = py.import_bound("torch").map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to import torch (install PyTorch to use TextBatch): {e}"
            ))
        })?;
        let torch_int64 = torch.getattr("int64")?;
        let values = PyList::new_bound(py, self.token_ids.iter().copied());
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("dtype", &torch_int64)?;
        let flat = torch.call_method("tensor", (values,), Some(&kwargs))?;
        let rows = i64::try_from(self.batch_rows)
            .map_err(|_| PyValueError::new_err("text batch rows do not fit i64"))?;
        let seq = i64::try_from(self.sequence_length)
            .map_err(|_| PyValueError::new_err("sequence_length does not fit i64"))?;
        flat.call_method1("view", (rows, seq))
    }

    #[getter]
    fn attention_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let Some(mask_raw) = &self.attention_mask else {
            return Ok(py.None().into_bound(py).into_any());
        };
        let torch = py.import_bound("torch").map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to import torch (install PyTorch to use TextBatch): {e}"
            ))
        })?;
        let torch_bool = torch.getattr("bool")?;
        let values = PyList::new_bound(py, mask_raw.iter().map(|v| *v != 0));
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("dtype", &torch_bool)?;
        let flat = torch.call_method("tensor", (values,), Some(&kwargs))?;
        let rows = i64::try_from(self.batch_rows)
            .map_err(|_| PyValueError::new_err("text batch rows do not fit i64"))?;
        let seq = i64::try_from(self.sequence_length)
            .map_err(|_| PyValueError::new_err("sequence_length does not fit i64"))?;
        flat.call_method1("view", (rows, seq))
    }

    fn to_torch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let token_ids = self.token_ids(py)?;
        let sample_ids = self.sample_ids(py)?;
        if self.attention_mask.is_some() {
            let attention_mask = self.attention_mask(py)?;
            let out = PyTuple::new_bound(
                py,
                [
                    token_ids.to_object(py),
                    attention_mask.to_object(py),
                    sample_ids.to_object(py),
                ],
            );
            return Ok(out.into_any());
        }
        let out = PyTuple::new_bound(py, [token_ids.to_object(py), sample_ids.to_object(py)]);
        Ok(out.into_any())
    }
}
