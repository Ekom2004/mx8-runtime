use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;

use mx8_runtime::pipeline::{Pipeline, RuntimeCaps};
use mx8_runtime::sink::Sink;
use mx8_runtime::types::Batch;

struct SlowSink {
    sleep: Duration,
    delivered_batches: AtomicU64,
    delivered_bytes: AtomicU64,
}

impl SlowSink {
    fn new(sleep: Duration) -> Self {
        Self {
            sleep,
            delivered_batches: AtomicU64::new(0),
            delivered_bytes: AtomicU64::new(0),
        }
    }
}

impl Sink for SlowSink {
    fn deliver(&self, batch: Batch) -> Result<()> {
        std::thread::sleep(self.sleep);
        self.delivered_batches.fetch_add(1, Ordering::Relaxed);
        self.delivered_bytes
            .fetch_add(batch.payload_len() as u64, Ordering::Relaxed);
        Ok(())
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn slow_sink_enforces_inflight_ram_cap() {
    let caps = RuntimeCaps {
        max_inflight_bytes: 32 * 1024,
        max_queue_batches: 8,
        batch_size_samples: 16,
        prefetch_batches: 1,
    };
    let pipeline = Pipeline::new(caps);
    let metrics = pipeline.metrics();

    let sink = Arc::new(SlowSink::new(Duration::from_millis(20)));
    pipeline
        .run_synthetic(sink.clone(), 256, 1024)
        .await
        .unwrap();

    assert_eq!(sink.delivered_batches.load(Ordering::Relaxed), 16);
    assert_eq!(sink.delivered_bytes.load(Ordering::Relaxed), 256 * 1024);

    let high_water = metrics.inflight_bytes_high_water.get();
    assert!(
        high_water <= caps.max_inflight_bytes,
        "inflight high-water {} > cap {}",
        high_water,
        caps.max_inflight_bytes
    );
}
