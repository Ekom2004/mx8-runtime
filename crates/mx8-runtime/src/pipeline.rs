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

        self.produce_batches(tx.clone(), total_samples, bytes_per_sample)
            .await?;
        drop(tx);

        sink_task.await??;
        Ok(())
    }

    async fn produce_batches(
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
