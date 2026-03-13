use super::*;

const DEFAULT_TRANSFORM_MAX_MS_PER_SAMPLE: u64 = 250;
const DEFAULT_TRANSFORM_MAX_MS_PER_BATCH: u64 = 10_000;
const DEFAULT_TRANSFORM_MAX_OUTPUT_BYTES_PER_SAMPLE: u64 = 16 * 1024 * 1024;
const WASM_PAGE_BYTES: u64 = 64 * 1024;

#[derive(Clone, Copy, Debug)]
pub(crate) struct TransformPolicy {
    pub(crate) max_ms_per_sample: u64,
    pub(crate) max_ms_per_batch: u64,
    pub(crate) max_output_bytes_per_sample: u64,
    pub(crate) max_output_bytes_per_batch: u64,
    pub(crate) heavy_ms_per_sample: u64,
}

impl TransformPolicy {
    pub(crate) fn from_env(max_inflight_bytes: u64) -> Self {
        let max_output_bytes_per_batch = env_u64("MX8_TRANSFORM_MAX_OUTPUT_BYTES_PER_BATCH")
            .unwrap_or(max_inflight_bytes)
            .min(max_inflight_bytes)
            .max(1);
        Self {
            max_ms_per_sample: env_u64("MX8_TRANSFORM_MAX_MS_PER_SAMPLE")
                .unwrap_or(DEFAULT_TRANSFORM_MAX_MS_PER_SAMPLE)
                .max(1),
            max_ms_per_batch: env_u64("MX8_TRANSFORM_MAX_MS_PER_BATCH")
                .unwrap_or(DEFAULT_TRANSFORM_MAX_MS_PER_BATCH)
                .max(1),
            max_output_bytes_per_sample: env_u64("MX8_TRANSFORM_MAX_OUTPUT_BYTES_PER_SAMPLE")
                .unwrap_or(DEFAULT_TRANSFORM_MAX_OUTPUT_BYTES_PER_SAMPLE)
                .max(1),
            max_output_bytes_per_batch,
            heavy_ms_per_sample: env_u64("MX8_TRANSFORM_HEAVY_MS_PER_SAMPLE")
                .unwrap_or(8)
                .max(1),
        }
    }
}

pub(crate) struct CompiledTransform {
    pub(crate) name: String,
    pub(crate) policy: TransformPolicy,
    engine: wasmi::Engine,
    module: wasmi::Module,
}

pub(crate) struct TransformApplyStats {
    pub(crate) transformed_samples: u64,
    pub(crate) output_ratio_milli: u64,
    pub(crate) heavy_mode_active: bool,
}

pub(crate) struct TransformApplyOutput {
    pub(crate) batch: Batch,
    pub(crate) stats: TransformApplyStats,
}

#[derive(Debug)]
pub(crate) struct TransformRuntimeError {
    pub(crate) message: String,
    pub(crate) timeout: bool,
}

impl std::fmt::Display for TransformRuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for TransformRuntimeError {}

impl CompiledTransform {
    pub(crate) fn compile_from_py(
        py: Python<'_>,
        transform_fn: &Bound<'_, PyAny>,
        max_inflight_bytes: u64,
    ) -> PyResult<Self> {
        if !transform_fn.is_callable() {
            return Err(PyValueError::new_err(
                "transform must be callable (use @mx8.transform)",
            ));
        }

        let marker: Option<bool> = transform_fn
            .getattr("__mx8_transform__")
            .ok()
            .and_then(|v| v.extract::<bool>().ok());
        if marker != Some(true) {
            return Err(PyValueError::new_err(
                "transform must be decorated with @mx8.transform",
            ));
        }

        let inspect = py.import_bound("inspect")?;
        let sig = inspect.getattr("signature")?.call1((transform_fn,))?;
        let params = sig.getattr("parameters")?;
        let param_len = params.call_method0("__len__")?.extract::<usize>()?;
        if param_len != 1 {
            return Err(PyValueError::new_err(format!(
                "@mx8.transform function must accept exactly 1 argument (got {param_len})"
            )));
        }

        let compile_fn = transform_fn.getattr("__mx8_compile__").map_err(|_| {
            PyValueError::new_err("@mx8.transform compile hook missing (__mx8_compile__)")
        })?;
        if !compile_fn.is_callable() {
            return Err(PyValueError::new_err(
                "@mx8.transform compile hook (__mx8_compile__) is not callable",
            ));
        }

        let compiled_obj = compile_fn.call0()?;
        let compile_bytes: Vec<u8> = if let Ok(v) = compiled_obj.extract::<Vec<u8>>() {
            v
        } else if let Ok(s) = compiled_obj.extract::<String>() {
            s.into_bytes()
        } else {
            return Err(PyValueError::new_err(
                "@mx8.transform compile hook must return bytes or str (WASM or WAT)",
            ));
        };

        let wasm_bytes = normalize_wasm_bytes(&compile_bytes).map_err(PyValueError::new_err)?;

        let engine = wasmi::Engine::default();
        let module = wasmi::Module::new(&engine, &wasm_bytes)
            .map_err(|e| PyValueError::new_err(format!("transform wasm compile failed: {e}")))?;

        {
            let mut store = wasmi::Store::new(&engine, ());
            let linker = wasmi::Linker::new(&engine);
            let instance = linker
                .instantiate(&mut store, &module)
                .and_then(|pre| pre.start(&mut store))
                .map_err(|e| {
                    PyValueError::new_err(format!("transform wasm instantiate failed: {e}"))
                })?;
            instance
                .get_export(&store, "memory")
                .and_then(|ext| ext.into_memory())
                .ok_or_else(|| PyValueError::new_err("transform wasm must export memory"))?;
            let _ = instance
                .get_typed_func::<(i32, i32, i32), i64>(&store, "transform")
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "transform wasm must export `transform(i32,i32,i32)->i64`: {e}"
                    ))
                })?;
        }

        let name = transform_fn
            .getattr("__mx8_transform_name__")
            .ok()
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_else(|| "transform".to_string());

        let policy = TransformPolicy::from_env(max_inflight_bytes);
        tracing::info!(
            target: "mx8_proof",
            event = "transform_validated",
            transform_name = %name,
            max_ms_per_sample = policy.max_ms_per_sample,
            max_ms_per_batch = policy.max_ms_per_batch,
            max_output_bytes_per_sample = policy.max_output_bytes_per_sample,
            max_output_bytes_per_batch = policy.max_output_bytes_per_batch,
            "validated transform wasm contract at loader startup"
        );

        Ok(Self {
            name,
            policy,
            engine,
            module,
        })
    }

    pub(crate) fn apply(
        &self,
        batch: &Batch,
    ) -> std::result::Result<TransformApplyOutput, TransformRuntimeError> {
        let mut store = wasmi::Store::new(&self.engine, ());
        let linker = wasmi::Linker::new(&self.engine);
        let instance = linker
            .instantiate(&mut store, &self.module)
            .and_then(|pre| pre.start(&mut store))
            .map_err(|e| TransformRuntimeError {
                message: format!("transform wasm instantiate failed: {e}"),
                timeout: false,
            })?;
        let memory = instance
            .get_export(&store, "memory")
            .and_then(|ext| ext.into_memory())
            .ok_or_else(|| TransformRuntimeError {
                message: "transform wasm must export memory".to_string(),
                timeout: false,
            })?;
        let transform = instance
            .get_typed_func::<(i32, i32, i32), i64>(&store, "transform")
            .map_err(|e| TransformRuntimeError {
                message: format!("transform wasm must export `transform(i32,i32,i32)->i64`: {e}"),
                timeout: false,
            })?;

        if batch.offsets.len() != batch.sample_ids.len().saturating_add(1) {
            return Err(TransformRuntimeError {
                message: "transform input batch invariant failed: offsets/sample_ids mismatch"
                    .to_string(),
                timeout: false,
            });
        }

        let mut output_payload: Vec<u8> = Vec::with_capacity(batch.payload.len());
        let mut output_offsets: Vec<u64> =
            Vec::with_capacity(batch.sample_ids.len().saturating_add(1));
        output_offsets.push(0);
        let mut output_ids: Vec<u64> = Vec::with_capacity(batch.sample_ids.len());
        let mut output_labels: Option<Vec<u64>> = batch
            .label_ids
            .as_ref()
            .map(|labels| Vec::with_capacity(labels.len()));

        let input_bytes = batch.payload.len() as u64;
        let started = Instant::now();
        for idx in 0..batch.sample_ids.len() {
            let elapsed_batch_ms = started.elapsed().as_millis() as u64;
            if elapsed_batch_ms > self.policy.max_ms_per_batch {
                return Err(TransformRuntimeError {
                    message: format!(
                        "transform batch time budget exceeded: {}ms > {}ms",
                        elapsed_batch_ms, self.policy.max_ms_per_batch
                    ),
                    timeout: true,
                });
            }

            let start = usize::try_from(batch.offsets[idx]).map_err(|_| TransformRuntimeError {
                message: "transform offset conversion overflow (start)".to_string(),
                timeout: false,
            })?;
            let end =
                usize::try_from(batch.offsets[idx + 1]).map_err(|_| TransformRuntimeError {
                    message: "transform offset conversion overflow (end)".to_string(),
                    timeout: false,
                })?;
            if end < start || end > batch.payload.len() {
                return Err(TransformRuntimeError {
                    message: format!(
                        "transform input slice bounds invalid for sample {}: start={} end={} payload_len={}",
                        batch.sample_ids[idx],
                        start,
                        end,
                        batch.payload.len()
                    ),
                    timeout: false,
                });
            }

            let input = &batch.payload[start..end];
            let input_len = input.len();
            let remaining_batch_cap = self
                .policy
                .max_output_bytes_per_batch
                .saturating_sub(output_payload.len() as u64);
            let out_cap_u64 = self
                .policy
                .max_output_bytes_per_sample
                .min(remaining_batch_cap)
                .max(1);
            let out_cap = usize::try_from(out_cap_u64).map_err(|_| TransformRuntimeError {
                message: "transform output cap conversion overflow".to_string(),
                timeout: false,
            })?;

            let in_ptr: usize = 0;
            let out_ptr: usize = input_len.saturating_add(WASM_PAGE_BYTES as usize);
            let needed = out_ptr
                .saturating_add(out_cap)
                .saturating_add(WASM_PAGE_BYTES as usize);
            ensure_memory_capacity(&memory, &mut store, needed)?;
            memory
                .write(&mut store, in_ptr, input)
                .map_err(|e| TransformRuntimeError {
                    message: format!("transform wasm memory write failed: {e}"),
                    timeout: false,
                })?;

            let sample_started = Instant::now();
            let out_len_i64 = transform
                .call(
                    &mut store,
                    (
                        i32::try_from(in_ptr).unwrap_or(0),
                        i32::try_from(input_len).map_err(|_| TransformRuntimeError {
                            message: format!(
                                "transform input too large for wasm i32 len: {}",
                                input_len
                            ),
                            timeout: false,
                        })?,
                        i32::try_from(out_ptr).map_err(|_| TransformRuntimeError {
                            message: format!(
                                "transform output pointer too large for wasm i32: {}",
                                out_ptr
                            ),
                            timeout: false,
                        })?,
                    ),
                )
                .map_err(|e| TransformRuntimeError {
                    message: format!("transform wasm call failed: {e}"),
                    timeout: false,
                })?;
            let elapsed_sample_ms = sample_started.elapsed().as_millis() as u64;
            if elapsed_sample_ms > self.policy.max_ms_per_sample {
                return Err(TransformRuntimeError {
                    message: format!(
                        "transform sample time budget exceeded for sample {}: {}ms > {}ms",
                        batch.sample_ids[idx], elapsed_sample_ms, self.policy.max_ms_per_sample
                    ),
                    timeout: true,
                });
            }

            if out_len_i64 == -1 {
                continue;
            }
            if out_len_i64 < 0 {
                return Err(TransformRuntimeError {
                    message: format!(
                        "transform wasm returned invalid negative length {} for sample {}",
                        out_len_i64, batch.sample_ids[idx]
                    ),
                    timeout: false,
                });
            }
            let out_len = usize::try_from(out_len_i64).map_err(|_| TransformRuntimeError {
                message: format!(
                    "transform wasm output length conversion overflow: {}",
                    out_len_i64
                ),
                timeout: false,
            })?;
            if out_len as u64 > self.policy.max_output_bytes_per_sample {
                return Err(TransformRuntimeError {
                    message: format!(
                        "transform output exceeds per-sample cap for sample {}: {} > {} bytes",
                        batch.sample_ids[idx], out_len, self.policy.max_output_bytes_per_sample
                    ),
                    timeout: false,
                });
            }
            if out_len > out_cap {
                return Err(TransformRuntimeError {
                    message: format!(
                        "transform output exceeds allocated cap for sample {}: {} > {} bytes",
                        batch.sample_ids[idx], out_len, out_cap
                    ),
                    timeout: false,
                });
            }

            let mut out = vec![0u8; out_len];
            memory
                .read(&store, out_ptr, &mut out)
                .map_err(|e| TransformRuntimeError {
                    message: format!("transform wasm memory read failed: {e}"),
                    timeout: false,
                })?;

            output_payload.extend_from_slice(&out);
            output_offsets.push(output_payload.len() as u64);
            output_ids.push(batch.sample_ids[idx]);
            if let (Some(src), Some(dst)) = (batch.label_ids.as_ref(), output_labels.as_mut()) {
                if let Some(label) = src.get(idx).copied() {
                    dst.push(label);
                }
            }
        }

        if output_offsets.len() != output_ids.len().saturating_add(1) {
            return Err(TransformRuntimeError {
                message: "transform output invariant failed: offsets/sample_ids mismatch"
                    .to_string(),
                timeout: false,
            });
        }

        let output_bytes = output_payload.len() as u64;
        let ratio_milli = if input_bytes == 0 {
            1000
        } else {
            ((output_bytes as f64 / input_bytes as f64) * 1000.0).round() as u64
        };
        let transformed_samples = output_ids.len() as u64;
        let batch_ms = started.elapsed().as_millis() as u64;
        let per_sample_ms = if transformed_samples == 0 {
            batch_ms
        } else {
            batch_ms.div_ceil(transformed_samples)
        };

        Ok(TransformApplyOutput {
            batch: Batch {
                sample_ids: Arc::from(output_ids.as_slice()),
                label_ids: output_labels.map(|v| Arc::from(v.as_slice())),
                offsets: Arc::from(output_offsets.as_slice()),
                payload: Arc::from(output_payload.as_slice()),
            },
            stats: TransformApplyStats {
                transformed_samples,
                output_ratio_milli: ratio_milli,
                heavy_mode_active: per_sample_ms >= self.policy.heavy_ms_per_sample,
            },
        })
    }
}

fn normalize_wasm_bytes(raw: &[u8]) -> std::result::Result<Vec<u8>, String> {
    const WASM_MAGIC: &[u8; 4] = b"\0asm";
    if raw.starts_with(WASM_MAGIC) {
        return Ok(raw.to_vec());
    }

    match wat::parse_bytes(raw) {
        Ok(parsed) => Ok(parsed.into_owned()),
        Err(err) => {
            if let Ok(text) = std::str::from_utf8(raw) {
                wat::parse_str(text).map_err(|e| {
                    format!("transform compile output is not valid WAT/WASM: {err}; {e}")
                })
            } else {
                Err(format!(
                    "transform compile output is not valid WASM binary and not UTF-8 WAT: {err}"
                ))
            }
        }
    }
}

fn ensure_memory_capacity(
    memory: &wasmi::Memory,
    store: &mut wasmi::Store<()>,
    needed_bytes: usize,
) -> std::result::Result<(), TransformRuntimeError> {
    let needed_pages = (needed_bytes as u64).div_ceil(WASM_PAGE_BYTES);
    let current_pages = memory.size(&*store) as u64;
    if current_pages >= needed_pages {
        return Ok(());
    }
    let delta = needed_pages.saturating_sub(current_pages);
    let delta_u32 = u32::try_from(delta).map_err(|_| TransformRuntimeError {
        message: format!("transform wasm memory grow delta overflow: {delta} pages"),
        timeout: false,
    })?;
    memory
        .grow(store, delta_u32)
        .map_err(|e| TransformRuntimeError {
            message: format!("transform wasm memory grow failed: {e}"),
            timeout: false,
        })?;
    Ok(())
}
