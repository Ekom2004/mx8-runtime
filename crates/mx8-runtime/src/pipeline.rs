use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::{mpsc, OwnedSemaphorePermit, Semaphore};

use mx8_observe::metrics::{Counter, Gauge};

use crate::sink::Sink;
use crate::types::Batch;

const PERMIT_UNIT_BYTES: u64 = 1024;

#[derive(Debug, Clone, Copy)]
pub struct RuntimeCaps {
    pub max_inflight_bytes: u64,
    pub max_queue_batches: usize,
    pub batch_size_samples: usize,
}

#[derive(Debug, Default)]
pub struct RuntimeMetrics {
    pub delivered_batches_total: Counter,
    pub delivered_samples_total: Counter,
    pub inflight_bytes: Gauge,
    pub inflight_bytes_high_water: Gauge,
}

impl RuntimeMetrics {
    fn on_inflight_add(&self, delta: u64) {
        let now = self.inflight_bytes.add(delta);
        self.inflight_bytes_high_water.max(now);
    }

    fn on_inflight_sub(&self, delta: u64) {
        self.inflight_bytes.sub(delta);
    }
}

struct InflightBatch {
    batch: Batch,
    bytes: u64,
    _permit: OwnedSemaphorePermit,
}

pub struct Pipeline {
    caps: RuntimeCaps,
    metrics: Arc<RuntimeMetrics>,
    inflight_sem: Arc<Semaphore>,
}

impl Pipeline {
    pub fn new(caps: RuntimeCaps) -> Self {
        let max_units = caps.max_inflight_bytes.div_ceil(PERMIT_UNIT_BYTES).max(1);
        let max = usize::try_from(max_units).unwrap_or(usize::MAX);
        Self {
            caps,
            metrics: Arc::new(RuntimeMetrics::default()),
            inflight_sem: Arc::new(Semaphore::new(max)),
        }
    }

    pub fn metrics(&self) -> Arc<RuntimeMetrics> {
        self.metrics.clone()
    }

    pub async fn run_synthetic<S: Sink>(
        &self,
        sink: Arc<S>,
        total_samples: u64,
        bytes_per_sample: usize,
    ) -> Result<()> {
        let (tx, sink_task) = self.spawn_sink(sink)?;
        self.produce_synthetic_batches(tx.clone(), total_samples, bytes_per_sample)
            .await?;
        drop(tx);
        sink_task.await??;
        Ok(())
    }

    pub async fn run_manifest_cache_dir<S: Sink>(
        &self,
        sink: Arc<S>,
        manifest_cache_dir: impl AsRef<Path>,
        manifest_hash: &str,
    ) -> Result<()> {
        let path = manifest_cache_dir.as_ref().join(manifest_hash);
        let bytes = std::fs::read(&path)?;
        self.run_manifest_bytes(sink, &bytes).await
    }

    pub async fn run_manifest_bytes<S: Sink>(
        &self,
        sink: Arc<S>,
        manifest_bytes: &[u8],
    ) -> Result<()> {
        let records = parse_canonical_manifest_tsv(manifest_bytes)?;

        let (tx, sink_task) = self.spawn_sink(sink)?;
        self.produce_manifest_batches(tx.clone(), records).await?;
        drop(tx);
        sink_task.await??;
        Ok(())
    }

    fn spawn_sink<S: Sink>(
        &self,
        sink: Arc<S>,
    ) -> Result<(
        mpsc::Sender<InflightBatch>,
        tokio::task::JoinHandle<Result<()>>,
    )> {
        let (tx, mut rx) = mpsc::channel::<InflightBatch>(self.caps.max_queue_batches);
        let metrics = self.metrics.clone();

        let sink_task = {
            let sink = sink.clone();
            let metrics = metrics.clone();
            tokio::spawn(async move {
                while let Some(inflight) = rx.recv().await {
                    let bytes = inflight.bytes;
                    let batch = inflight.batch;
                    let sample_count = u64::try_from(batch.sample_count()).unwrap_or_default();
                    // Run delivery in a blocking thread so a slow sink exerts backpressure
                    // without stalling the entire tokio runtime.
                    let sink = sink.clone();
                    tokio::task::spawn_blocking(move || sink.deliver(batch))
                        .await
                        .map_err(anyhow::Error::from)??;

                    metrics.delivered_batches_total.inc();
                    metrics.delivered_samples_total.inc_by(sample_count);
                    metrics.on_inflight_sub(bytes);
                    tracing::info!(
                        target: "mx8_proof",
                        event = "delivered",
                        batch_bytes = bytes,
                        inflight_bytes = metrics.inflight_bytes.get(),
                        "delivered batch"
                    );
                }
                Ok::<(), anyhow::Error>(())
            })
        };

        Ok((tx, sink_task))
    }

    async fn produce_synthetic_batches(
        &self,
        tx: mpsc::Sender<InflightBatch>,
        total_samples: u64,
        bytes_per_sample: usize,
    ) -> Result<()> {
        let mut current_ids: Vec<u64> = Vec::with_capacity(self.caps.batch_size_samples);
        let mut current_payload: Vec<u8> = Vec::with_capacity(
            self.caps
                .batch_size_samples
                .saturating_mul(bytes_per_sample),
        );

        let mut sender = tx;
        for sample_id in 0..total_samples {
            current_ids.push(sample_id);
            current_payload.extend(std::iter::repeat_n(0u8, bytes_per_sample));

            if current_ids.len() >= self.caps.batch_size_samples {
                self.send_batch(&mut sender, &mut current_ids, &mut current_payload)
                    .await?;
            }
        }

        if !current_ids.is_empty() {
            self.send_batch(&mut sender, &mut current_ids, &mut current_payload)
                .await?;
        }

        Ok(())
    }

    async fn produce_manifest_batches(
        &self,
        tx: mpsc::Sender<InflightBatch>,
        records: Vec<mx8_core::types::ManifestRecord>,
    ) -> Result<()> {
        let mut sender = tx;
        let mut start = 0usize;
        while start < records.len() {
            let end = (start + self.caps.batch_size_samples).min(records.len());
            self.send_manifest_batch(&mut sender, &records[start..end])
                .await?;
            start = end;
        }
        Ok(())
    }

    async fn send_manifest_batch(
        &self,
        tx: &mut mpsc::Sender<InflightBatch>,
        records: &[mx8_core::types::ManifestRecord],
    ) -> Result<()> {
        let plan = tokio::task::spawn_blocking({
            let records = records.to_vec();
            move || plan_read_specs(&records)
        })
        .await
        .map_err(anyhow::Error::from)??;

        let bytes = plan.total_bytes;
        let permit_units = if bytes == 0 {
            0u64
        } else {
            bytes.div_ceil(PERMIT_UNIT_BYTES).max(1)
        };

        anyhow::ensure!(
            permit_units <= u32::MAX as u64,
            "batch too large for permit accounting ({} units)",
            permit_units
        );

        let permit = self
            .inflight_sem
            .clone()
            .acquire_many_owned(permit_units as u32)
            .await?;

        self.metrics.on_inflight_add(bytes);

        let read = tokio::task::spawn_blocking(move || read_specs_payload(&plan.specs))
            .await
            .map_err(anyhow::Error::from)?;

        let payload = match read {
            Ok(p) => p,
            Err(err) => {
                // Ensure gauge matches permit release on early exit.
                // Permit will drop as we return.
                self.metrics.on_inflight_sub(bytes);
                return Err(err);
            }
        };

        let batch = Batch {
            sample_ids: Arc::from(plan.sample_ids.as_slice()),
            payload: Arc::from(payload.as_slice()),
        };

        tx.send(InflightBatch {
            batch,
            bytes,
            _permit: permit,
        })
        .await?;

        Ok(())
    }

    async fn send_batch(
        &self,
        tx: &mut mpsc::Sender<InflightBatch>,
        ids: &mut Vec<u64>,
        payload: &mut Vec<u8>,
    ) -> Result<()> {
        let bytes = u64::try_from(payload.len()).unwrap_or(u64::MAX);
        let permit_units = if bytes == 0 {
            0u64
        } else {
            bytes.div_ceil(PERMIT_UNIT_BYTES).max(1)
        };

        anyhow::ensure!(
            permit_units <= u32::MAX as u64,
            "batch too large for permit accounting ({} units)",
            permit_units
        );

        let permit = self
            .inflight_sem
            .clone()
            .acquire_many_owned(permit_units as u32)
            .await?;

        let batch = Batch {
            sample_ids: Arc::from(ids.as_slice()),
            payload: Arc::from(payload.as_slice()),
        };

        ids.clear();
        payload.clear();

        self.metrics.on_inflight_add(bytes);

        tx.send(InflightBatch {
            batch,
            bytes,
            _permit: permit,
        })
        .await?;

        Ok(())
    }
}

#[derive(Debug, Clone)]
struct ReadSpec {
    sample_id: u64,
    path: PathBuf,
    offset: u64,
    len: u64,
    expect_eof: bool,
}

#[derive(Debug, Clone)]
struct BatchPlan {
    sample_ids: Vec<u64>,
    specs: Vec<ReadSpec>,
    total_bytes: u64,
}

fn plan_read_specs(records: &[mx8_core::types::ManifestRecord]) -> Result<BatchPlan> {
    let mut sample_ids = Vec::with_capacity(records.len());
    let mut specs = Vec::with_capacity(records.len());
    let mut total_bytes: u64 = 0;

    for r in records {
        if r.location.starts_with("s3://") {
            anyhow::bail!(
                "s3 locations are not supported in this runtime slice: {}",
                r.location
            );
        }

        let path = PathBuf::from(&r.location);

        let (offset, len, expect_eof) = match (r.byte_offset, r.byte_length) {
            (Some(off), Some(len)) => (off, len, false),
            (None, None) => {
                let md = std::fs::metadata(&path)?;
                let size = md.len();
                (0u64, size, true)
            }
            _ => anyhow::bail!("partial byte range for sample_id {}", r.sample_id),
        };

        sample_ids.push(r.sample_id);
        specs.push(ReadSpec {
            sample_id: r.sample_id,
            path,
            offset,
            len,
            expect_eof,
        });

        total_bytes = total_bytes
            .checked_add(len)
            .ok_or_else(|| anyhow::anyhow!("byte size overflow"))?;
    }

    Ok(BatchPlan {
        sample_ids,
        specs,
        total_bytes,
    })
}

fn read_specs_payload(specs: &[ReadSpec]) -> Result<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};

    let mut total: u64 = 0;
    for s in specs {
        total = total
            .checked_add(s.len)
            .ok_or_else(|| anyhow::anyhow!("byte size overflow"))?;
    }

    let cap = usize::try_from(total).map_err(|_| anyhow::anyhow!("payload too large"))?;
    let mut payload = Vec::<u8>::with_capacity(cap);

    for s in specs {
        let len = usize::try_from(s.len).map_err(|_| anyhow::anyhow!("payload too large"))?;

        let mut f = std::fs::File::open(&s.path)?;
        f.seek(SeekFrom::Start(s.offset))?;

        let start = payload.len();
        payload.resize(start.saturating_add(len), 0u8);
        f.read_exact(&mut payload[start..])?;

        if s.expect_eof {
            let mut tmp = [0u8; 1];
            let extra = f.read(&mut tmp)?;
            if extra != 0 {
                anyhow::bail!(
                    "file grew after sizing (expected eof): sample_id={} path={}",
                    s.sample_id,
                    s.path.display()
                );
            }
        }
    }

    Ok(payload)
}

fn parse_canonical_manifest_tsv(bytes: &[u8]) -> Result<Vec<mx8_core::types::ManifestRecord>> {
    let s = std::str::from_utf8(bytes).map_err(|e| anyhow::anyhow!("manifest not utf-8: {e}"))?;

    let mut lines = s.lines();
    let first = lines
        .by_ref()
        .find(|l| !l.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("empty manifest"))?;

    let Some((k, v)) = first.split_once('=') else {
        anyhow::bail!("manifest header missing schema_version");
    };
    if k.trim() != "schema_version" {
        anyhow::bail!("manifest header must be schema_version=<n>");
    }
    let schema_version: u32 = v
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid schema_version"))?;
    anyhow::ensure!(
        schema_version == mx8_core::types::MANIFEST_SCHEMA_VERSION,
        "unsupported schema_version {}",
        schema_version
    );

    let mut records: Vec<mx8_core::types::ManifestRecord> = Vec::new();
    for (i, raw) in lines.enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 2 {
            anyhow::bail!("line {}: expected at least 2 columns", i + 2);
        }
        let sample_id: u64 = cols[0]
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("line {}: bad sample_id", i + 2))?;
        let location = cols[1].trim().to_string();

        let mut byte_offset: Option<u64> = None;
        let mut byte_length: Option<u64> = None;
        let mut decode_hint: Option<String> = None;

        if cols.len() >= 4 {
            if !cols[2].trim().is_empty() || !cols[3].trim().is_empty() {
                byte_offset = Some(
                    cols[2]
                        .trim()
                        .parse()
                        .map_err(|_| anyhow::anyhow!("line {}: bad byte_offset", i + 2))?,
                );
                byte_length = Some(
                    cols[3]
                        .trim()
                        .parse()
                        .map_err(|_| anyhow::anyhow!("line {}: bad byte_length", i + 2))?,
                );
            }
        } else if cols.len() == 3 {
            anyhow::bail!("line {}: partial byte range", i + 2);
        }

        if cols.len() >= 5 && !cols[4].trim().is_empty() {
            decode_hint = Some(cols[4].trim().to_string());
        }

        let record = mx8_core::types::ManifestRecord {
            sample_id,
            location,
            byte_offset,
            byte_length,
            decode_hint,
        };
        record
            .validate()
            .map_err(|e| anyhow::anyhow!("line {}: {e}", i + 2))?;
        records.push(record);
    }

    // Enforce sequential IDs for the early dev manifest format.
    for (idx, r) in records.iter().enumerate() {
        let expected = idx as u64;
        anyhow::ensure!(
            r.sample_id == expected,
            "expected sample_id {} but found {}",
            expected,
            r.sample_id
        );
    }

    Ok(records)
}
