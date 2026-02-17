use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use anyhow::Result;

use mx8_runtime::pipeline::{Pipeline, RuntimeCaps};
use mx8_runtime::sink::Sink;
use mx8_runtime::types::Batch;

struct SlowSink {
    sleep: Duration,
    delivered_samples: AtomicU64,
    delivered_bytes: AtomicU64,
}

impl SlowSink {
    fn new(sleep: Duration) -> Self {
        Self {
            sleep,
            delivered_samples: AtomicU64::new(0),
            delivered_bytes: AtomicU64::new(0),
        }
    }
}

#[derive(Default)]
struct RecordingSink {
    batch_bytes: Mutex<Vec<usize>>,
    delivered_samples: AtomicU64,
}

impl Sink for RecordingSink {
    fn deliver(&self, batch: Batch) -> Result<()> {
        self.delivered_samples
            .fetch_add(batch.sample_count() as u64, Ordering::Relaxed);
        let mut guard = self.batch_bytes.lock().expect("batch_bytes mutex poisoned");
        guard.push(batch.payload_len());
        Ok(())
    }
}

impl Sink for SlowSink {
    fn deliver(&self, batch: Batch) -> Result<()> {
        std::thread::sleep(self.sleep);
        self.delivered_samples
            .fetch_add(batch.sample_count() as u64, Ordering::Relaxed);
        self.delivered_bytes
            .fetch_add(batch.payload_len() as u64, Ordering::Relaxed);
        Ok(())
    }
}

fn temp_dir(test_name: &str) -> Result<std::path::PathBuf> {
    let mut root = std::env::temp_dir();
    root.push(format!(
        "mx8-runtime-{test_name}-{}-{}",
        std::process::id(),
        mx8_observe::time::unix_time_ms()
    ));
    std::fs::create_dir_all(&root)?;
    Ok(root)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn manifest_pipeline_reads_files_and_is_bounded() -> Result<()> {
    let root = temp_dir("manifest-pipeline")?;

    // Make two files (one full, one ranged).
    let f0 = root.join("a.bin");
    let f1 = root.join("b.bin");
    std::fs::write(&f0, vec![1u8; 1024])?;
    std::fs::write(&f1, vec![2u8; 2048])?;

    let manifest = format!(
        "schema_version=0\n\
0\t{}\t\t\t\n\
1\t{}\t0\t10\tx\n",
        f0.display(),
        f1.display()
    );

    let caps = RuntimeCaps {
        max_inflight_bytes: 32 * 1024,
        max_queue_batches: 8,
        batch_size_samples: 2,
        prefetch_batches: 1,
        target_batch_bytes: None,
        max_batch_bytes: None,
        max_process_rss_bytes: None,
    };
    let pipeline = Pipeline::new(caps);
    let metrics = pipeline.metrics();

    let sink = Arc::new(SlowSink::new(Duration::from_millis(20)));
    pipeline
        .run_manifest_bytes(sink.clone(), manifest.as_bytes())
        .await?;

    assert_eq!(sink.delivered_samples.load(Ordering::Relaxed), 2);
    assert_eq!(sink.delivered_bytes.load(Ordering::Relaxed), 1024 + 10);

    let high_water = metrics.inflight_bytes_high_water.get();
    assert!(
        high_water <= caps.max_inflight_bytes,
        "inflight high-water {} > cap {}",
        high_water,
        caps.max_inflight_bytes
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn manifest_pipeline_can_run_subset_range() -> Result<()> {
    let root = temp_dir("manifest-pipeline-range")?;

    let f0 = root.join("a.bin");
    let f1 = root.join("b.bin");
    let f2 = root.join("c.bin");
    std::fs::write(&f0, vec![1u8; 100])?;
    std::fs::write(&f1, vec![2u8; 200])?;
    std::fs::write(&f2, vec![3u8; 300])?;

    let manifest = format!(
        "schema_version=0\n\
0\t{}\t\t\t\n\
1\t{}\t\t\t\n\
2\t{}\t\t\t\n",
        f0.display(),
        f1.display(),
        f2.display()
    );

    let caps = RuntimeCaps {
        max_inflight_bytes: 8 * 1024,
        max_queue_batches: 4,
        batch_size_samples: 8,
        prefetch_batches: 1,
        target_batch_bytes: None,
        max_batch_bytes: None,
        max_process_rss_bytes: None,
    };
    let pipeline = Pipeline::new(caps);

    let sink = Arc::new(SlowSink::new(Duration::from_millis(1)));
    pipeline
        .run_manifest_bytes_range(sink.clone(), manifest.as_bytes(), 1, 3)
        .await?;

    assert_eq!(sink.delivered_samples.load(Ordering::Relaxed), 2);
    assert_eq!(sink.delivered_bytes.load(Ordering::Relaxed), 200 + 300);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn manifest_pipeline_bytes_aware_batches_are_bounded() -> Result<()> {
    let root = temp_dir("manifest-pipeline-bytes-aware")?;
    let f0 = root.join("data.bin");
    std::fs::write(&f0, vec![7u8; 1400])?;

    let manifest = format!(
        "schema_version=0\n\
0\t{}\t0\t600\tx\n\
1\t{}\t600\t600\tx\n\
2\t{}\t1200\t200\tx\n",
        f0.display(),
        f0.display(),
        f0.display()
    );

    let caps = RuntimeCaps {
        max_inflight_bytes: 8 * 1024,
        max_queue_batches: 8,
        batch_size_samples: 32,
        prefetch_batches: 1,
        target_batch_bytes: Some(700),
        max_batch_bytes: Some(1024),
        max_process_rss_bytes: None,
    };
    let pipeline = Pipeline::new(caps);
    let sink = Arc::new(RecordingSink::default());
    pipeline
        .run_manifest_bytes(sink.clone(), manifest.as_bytes())
        .await?;

    assert_eq!(sink.delivered_samples.load(Ordering::Relaxed), 3);
    let got = sink.batch_bytes.lock().expect("batch_bytes mutex poisoned");
    assert_eq!(&*got, &[600usize, 800usize]);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn manifest_pipeline_bytes_aware_requires_byte_lengths() -> Result<()> {
    let root = temp_dir("manifest-pipeline-bytes-aware-missing-size")?;
    let f0 = root.join("data.bin");
    std::fs::write(&f0, vec![1u8; 512])?;
    let manifest = format!(
        "schema_version=0\n\
0\t{}\t\t\t\n",
        f0.display()
    );

    let caps = RuntimeCaps {
        max_inflight_bytes: 8 * 1024,
        max_queue_batches: 8,
        batch_size_samples: 32,
        prefetch_batches: 1,
        target_batch_bytes: Some(256),
        max_batch_bytes: Some(1024),
        max_process_rss_bytes: None,
    };
    let pipeline = Pipeline::new(caps);
    let sink = Arc::new(SlowSink::new(Duration::from_millis(1)));
    let err = pipeline
        .run_manifest_bytes(sink, manifest.as_bytes())
        .await
        .expect_err("expected byte-aware mode to require explicit byte_length");

    assert!(
        err.to_string()
            .contains("byte-aware batching requires byte_length"),
        "unexpected error: {err}"
    );
    Ok(())
}
