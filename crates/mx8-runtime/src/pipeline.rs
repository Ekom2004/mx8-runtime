use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::{mpsc, OwnedSemaphorePermit, Semaphore};

use mx8_observe::metrics::{Counter, Gauge};

use crate::sink::Sink;
use crate::types::Batch;

#[cfg(feature = "s3")]
use aws_sdk_s3::primitives::{AggregatedBytes, ByteStream};

#[cfg(feature = "s3")]
type S3Client = aws_sdk_s3::Client;

#[cfg(feature = "s3")]
type S3SdkError<E> = aws_sdk_s3::error::SdkError<E>;

#[cfg(not(feature = "s3"))]
type S3Client = ();

type HttpClient = reqwest::Client;

const PERMIT_UNIT_BYTES: u64 = 1024;

#[derive(Debug, Clone, Copy)]
pub struct RuntimeCaps {
    pub max_inflight_bytes: u64,
    pub max_queue_batches: usize,
    pub batch_size_samples: usize,
    pub prefetch_batches: usize,
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

pub struct BatchLease {
    pub batch: Batch,
    pub bytes: u64,
    metrics: Arc<RuntimeMetrics>,
    _permit: OwnedSemaphorePermit,
}

impl Drop for BatchLease {
    fn drop(&mut self) {
        self.metrics.on_inflight_sub(self.bytes);
    }
}

#[derive(Clone)]
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

    /// Spawn a manifest-driven producer which streams `BatchLease`s to the caller.
    ///
    /// The returned lease holds the inflight RAM permit until it is dropped, so a consumer
    /// (e.g., a Python iterator) can apply backpressure by holding onto batches.
    pub async fn spawn_manifest_bytes_stream(
        &self,
        manifest_bytes: Vec<u8>,
    ) -> Result<(
        mpsc::Receiver<BatchLease>,
        tokio::task::JoinHandle<Result<()>>,
    )> {
        let records = parse_canonical_manifest_tsv(&manifest_bytes)?;
        let has_s3 = records.iter().any(|r| r.location.starts_with("s3://"));
        let has_http = records
            .iter()
            .any(|r| r.location.starts_with("http://") || r.location.starts_with("https://"));
        let s3_client: Option<S3Client> = if has_s3 {
            #[cfg(feature = "s3")]
            {
                Some(crate::s3::client_from_env().await?)
            }
            #[cfg(not(feature = "s3"))]
            {
                anyhow::bail!(
                    "manifest contains s3:// locations but mx8-runtime was built without feature 's3'"
                );
            }
        } else {
            None
        };

        let http_client: Option<HttpClient> = if has_http {
            Some(
                reqwest::Client::builder()
                    .connect_timeout(std::time::Duration::from_secs(2))
                    .timeout(std::time::Duration::from_secs(15))
                    .build()?,
            )
        } else {
            None
        };

        let (tx, rx) = mpsc::channel::<BatchLease>(self.caps.max_queue_batches);
        let pipeline = self.clone();
        let task = tokio::spawn(async move {
            pipeline
                .produce_manifest_batches(tx, records, s3_client, http_client)
                .await
        });
        Ok((rx, task))
    }

    pub async fn spawn_manifest_bytes_range_stream(
        &self,
        manifest_bytes: Vec<u8>,
        start_id: u64,
        end_id: u64,
    ) -> Result<(
        mpsc::Receiver<BatchLease>,
        tokio::task::JoinHandle<Result<()>>,
    )> {
        let records = parse_canonical_manifest_tsv(&manifest_bytes)?;
        let records = select_record_range(records, start_id, end_id)?;

        let has_s3 = records.iter().any(|r| r.location.starts_with("s3://"));
        let has_http = records
            .iter()
            .any(|r| r.location.starts_with("http://") || r.location.starts_with("https://"));
        let s3_client: Option<S3Client> = if has_s3 {
            #[cfg(feature = "s3")]
            {
                Some(crate::s3::client_from_env().await?)
            }
            #[cfg(not(feature = "s3"))]
            {
                anyhow::bail!(
                    "manifest contains s3:// locations but mx8-runtime was built without feature 's3'"
                );
            }
        } else {
            None
        };

        let http_client: Option<HttpClient> = if has_http {
            Some(
                reqwest::Client::builder()
                    .connect_timeout(std::time::Duration::from_secs(2))
                    .timeout(std::time::Duration::from_secs(15))
                    .build()?,
            )
        } else {
            None
        };

        tracing::info!(
            target: "mx8_proof",
            event = "range_selected",
            start_id = start_id,
            end_id = end_id,
            samples = records.len() as u64,
            "selected manifest range"
        );

        let (tx, rx) = mpsc::channel::<BatchLease>(self.caps.max_queue_batches);
        let pipeline = self.clone();
        let task = tokio::spawn(async move {
            pipeline
                .produce_manifest_batches(tx, records, s3_client, http_client)
                .await
        });
        Ok((rx, task))
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

    pub async fn run_manifest_cache_dir_range<S: Sink>(
        &self,
        sink: Arc<S>,
        manifest_cache_dir: impl AsRef<Path>,
        manifest_hash: &str,
        start_id: u64,
        end_id: u64,
    ) -> Result<()> {
        let path = manifest_cache_dir.as_ref().join(manifest_hash);
        let bytes = std::fs::read(&path)?;
        self.run_manifest_bytes_range(sink, &bytes, start_id, end_id)
            .await
    }

    pub async fn run_manifest_bytes<S: Sink>(
        &self,
        sink: Arc<S>,
        manifest_bytes: &[u8],
    ) -> Result<()> {
        let records = parse_canonical_manifest_tsv(manifest_bytes)?;
        let has_s3 = records.iter().any(|r| r.location.starts_with("s3://"));
        let has_http = records
            .iter()
            .any(|r| r.location.starts_with("http://") || r.location.starts_with("https://"));
        let s3_client: Option<S3Client> = if has_s3 {
            #[cfg(feature = "s3")]
            {
                Some(crate::s3::client_from_env().await?)
            }
            #[cfg(not(feature = "s3"))]
            {
                anyhow::bail!(
                    "manifest contains s3:// locations but mx8-runtime was built without feature 's3'"
                );
            }
        } else {
            None
        };

        let http_client: Option<HttpClient> = if has_http {
            Some(
                reqwest::Client::builder()
                    .connect_timeout(std::time::Duration::from_secs(2))
                    .timeout(std::time::Duration::from_secs(15))
                    .build()?,
            )
        } else {
            None
        };

        let (tx, sink_task) = self.spawn_sink(sink)?;
        self.produce_manifest_batches(tx.clone(), records, s3_client, http_client)
            .await?;
        drop(tx);
        sink_task.await??;
        Ok(())
    }

    pub async fn run_manifest_bytes_range<S: Sink>(
        &self,
        sink: Arc<S>,
        manifest_bytes: &[u8],
        start_id: u64,
        end_id: u64,
    ) -> Result<()> {
        let records = parse_canonical_manifest_tsv(manifest_bytes)?;
        let records = select_record_range(records, start_id, end_id)?;

        let has_s3 = records.iter().any(|r| r.location.starts_with("s3://"));
        let has_http = records
            .iter()
            .any(|r| r.location.starts_with("http://") || r.location.starts_with("https://"));
        let s3_client: Option<S3Client> = if has_s3 {
            #[cfg(feature = "s3")]
            {
                Some(crate::s3::client_from_env().await?)
            }
            #[cfg(not(feature = "s3"))]
            {
                anyhow::bail!(
                    "manifest contains s3:// locations but mx8-runtime was built without feature 's3'"
                );
            }
        } else {
            None
        };

        let http_client: Option<HttpClient> = if has_http {
            Some(
                reqwest::Client::builder()
                    .connect_timeout(std::time::Duration::from_secs(2))
                    .timeout(std::time::Duration::from_secs(15))
                    .build()?,
            )
        } else {
            None
        };

        tracing::info!(
            target: "mx8_proof",
            event = "range_selected",
            start_id = start_id,
            end_id = end_id,
            samples = records.len() as u64,
            "selected manifest range"
        );

        let (tx, sink_task) = self.spawn_sink(sink)?;
        self.produce_manifest_batches(tx.clone(), records, s3_client, http_client)
            .await?;
        drop(tx);
        sink_task.await??;
        Ok(())
    }

    fn spawn_sink<S: Sink>(
        &self,
        sink: Arc<S>,
    ) -> Result<(
        mpsc::Sender<BatchLease>,
        tokio::task::JoinHandle<Result<()>>,
    )> {
        let (tx, mut rx) = mpsc::channel::<BatchLease>(self.caps.max_queue_batches);
        let metrics = self.metrics.clone();

        let sink_task = {
            let sink = sink.clone();
            let metrics = metrics.clone();
            tokio::spawn(async move {
                while let Some(inflight) = rx.recv().await {
                    let bytes = inflight.bytes;
                    let batch = inflight.batch.clone();
                    let sample_count = u64::try_from(batch.sample_count()).unwrap_or_default();
                    // Run delivery in a blocking thread so a slow sink exerts backpressure
                    // without stalling the entire tokio runtime.
                    let sink = sink.clone();
                    tokio::task::spawn_blocking(move || sink.deliver(batch))
                        .await
                        .map_err(anyhow::Error::from)??;

                    metrics.delivered_batches_total.inc();
                    metrics.delivered_samples_total.inc_by(sample_count);
                    drop(inflight);
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
        tx: mpsc::Sender<BatchLease>,
        total_samples: u64,
        bytes_per_sample: usize,
    ) -> Result<()> {
        let mut current_ids: Vec<u64> = Vec::with_capacity(self.caps.batch_size_samples);
        let mut current_offsets: Vec<u64> =
            Vec::with_capacity(self.caps.batch_size_samples.saturating_add(1));
        current_offsets.push(0);
        let mut current_payload: Vec<u8> = Vec::with_capacity(
            self.caps
                .batch_size_samples
                .saturating_mul(bytes_per_sample),
        );

        let mut sender = tx;
        for sample_id in 0..total_samples {
            current_ids.push(sample_id);
            current_payload.extend(std::iter::repeat_n(0u8, bytes_per_sample));
            current_offsets.push(u64::try_from(current_payload.len()).unwrap_or(u64::MAX));

            if current_ids.len() >= self.caps.batch_size_samples {
                self.send_batch(
                    &mut sender,
                    &mut current_ids,
                    &mut current_offsets,
                    &mut current_payload,
                )
                .await?;
            }
        }

        if !current_ids.is_empty() {
            self.send_batch(
                &mut sender,
                &mut current_ids,
                &mut current_offsets,
                &mut current_payload,
            )
            .await?;
        }

        Ok(())
    }

    async fn produce_manifest_batches(
        &self,
        tx: mpsc::Sender<BatchLease>,
        records: Vec<mx8_core::types::ManifestRecord>,
        s3_client: Option<S3Client>,
        http_client: Option<HttpClient>,
    ) -> Result<()> {
        let prefetch = std::cmp::max(1, self.caps.prefetch_batches);
        if prefetch <= 1 {
            let sender = tx;
            let mut start = 0usize;
            while start < records.len() {
                let end = (start + self.caps.batch_size_samples).min(records.len());
                let inflight = build_manifest_inflight_batch(
                    self.caps,
                    self.metrics.clone(),
                    self.inflight_sem.clone(),
                    &records[start..end],
                    s3_client.as_ref(),
                    http_client.as_ref(),
                )
                .await?;
                sender.send(inflight).await?;
                start = end;
            }
            return Ok(());
        }

        let records = Arc::new(records);
        let mut joinset = tokio::task::JoinSet::new();
        let mut buffer: std::collections::BTreeMap<usize, BatchLease> =
            std::collections::BTreeMap::new();
        let mut next_to_send: usize = 0;
        let mut next_batch_id: usize = 0;
        let mut start = 0usize;

        while start < records.len() || !joinset.is_empty() {
            while start < records.len() && joinset.len() < prefetch {
                let end = (start + self.caps.batch_size_samples).min(records.len());
                let batch_id = next_batch_id;
                next_batch_id = next_batch_id.saturating_add(1);

                let caps = self.caps;
                let metrics = self.metrics.clone();
                let sem = self.inflight_sem.clone();
                let records = records.clone();
                let s3_client = s3_client.clone();
                let http_client = http_client.clone();

                joinset.spawn(async move {
                    let inflight = build_manifest_inflight_batch(
                        caps,
                        metrics,
                        sem,
                        &records[start..end],
                        s3_client.as_ref(),
                        http_client.as_ref(),
                    )
                    .await;
                    (batch_id, inflight)
                });

                start = end;
            }

            let Some(res) = joinset.join_next().await else {
                break;
            };
            let (batch_id, inflight) = res.map_err(anyhow::Error::from)?;
            let inflight = inflight?;
            buffer.insert(batch_id, inflight);

            while let Some(inflight) = buffer.remove(&next_to_send) {
                tx.send(inflight).await?;
                next_to_send = next_to_send.saturating_add(1);
            }
        }

        Ok(())
    }

    async fn send_batch(
        &self,
        tx: &mut mpsc::Sender<BatchLease>,
        ids: &mut Vec<u64>,
        offsets: &mut Vec<u64>,
        payload: &mut Vec<u8>,
    ) -> Result<()> {
        let bytes = u64::try_from(payload.len()).unwrap_or(u64::MAX);
        anyhow::ensure!(
            bytes <= self.caps.max_inflight_bytes,
            "batch payload bytes {} exceeds max_inflight_bytes {}",
            bytes,
            self.caps.max_inflight_bytes
        );
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

        anyhow::ensure!(
            offsets.len() == ids.len().saturating_add(1),
            "offsets length {} must equal sample_ids length + 1 ({})",
            offsets.len(),
            ids.len().saturating_add(1),
        );
        anyhow::ensure!(
            offsets.first().copied().unwrap_or_default() == 0,
            "offsets must start at 0"
        );
        anyhow::ensure!(
            offsets.last().copied().unwrap_or_default() == bytes,
            "offsets must end at payload length ({}), got {}",
            bytes,
            offsets.last().copied().unwrap_or_default()
        );

        let batch = Batch {
            sample_ids: Arc::from(ids.as_slice()),
            offsets: Arc::from(offsets.as_slice()),
            payload: Arc::from(payload.as_slice()),
        };

        ids.clear();
        offsets.clear();
        offsets.push(0);
        payload.clear();

        self.metrics.on_inflight_add(bytes);

        tx.send(BatchLease {
            batch,
            bytes,
            metrics: self.metrics.clone(),
            _permit: permit,
        })
        .await?;

        Ok(())
    }
}

#[derive(Debug, Clone)]
struct ReadSpec {
    sample_id: u64,
    location: SpecLocation,
    offset: u64,
    len: u64,
    expect_eof: bool,
}

#[derive(Debug, Clone)]
enum SpecLocation {
    Local(PathBuf),
    #[cfg(feature = "s3")]
    S3 {
        bucket: String,
        key: String,
    },
    Http {
        url: String,
    },
}

#[derive(Debug, Clone)]
struct BatchPlan {
    sample_ids: Vec<u64>,
    specs: Vec<ReadSpec>,
    total_bytes: u64,
}

async fn build_batch_plan(
    records: &[mx8_core::types::ManifestRecord],
    s3_client: Option<&S3Client>,
    http_client: Option<&HttpClient>,
) -> Result<BatchPlan> {
    #[cfg(not(feature = "s3"))]
    let _ = s3_client;

    let mut sample_ids = Vec::with_capacity(records.len());
    let mut specs = Vec::with_capacity(records.len());
    let mut total_bytes: u64 = 0;

    for r in records {
        let location = parse_location(&r.location)?;

        let (offset, len, expect_eof) = match (r.byte_offset, r.byte_length) {
            (Some(off), Some(len)) => (off, len, false),
            (None, None) => match &location {
                SpecLocation::Local(path) => {
                    let path = path.clone();
                    let size = tokio::task::spawn_blocking(move || -> Result<u64> {
                        Ok(std::fs::metadata(&path)?.len())
                    })
                    .await
                    .map_err(anyhow::Error::from)??;
                    (0u64, size, true)
                }
                #[cfg(feature = "s3")]
                SpecLocation::S3 { bucket, key } => {
                    let client = s3_client.ok_or_else(|| {
                        anyhow::anyhow!("s3 client not configured but s3 location present")
                    })?;
                    let bucket = bucket.clone();
                    let key = key.clone();
                    let out = s3_with_retry(r.sample_id, || {
                        let bucket = bucket.clone();
                        let key = key.clone();
                        async move { client.head_object().bucket(bucket).key(key).send().await }
                    })
                    .await?;
                    let size = out.content_length().unwrap_or(0) as u64;
                    anyhow::ensure!(
                        size > 0,
                        "s3 object has unknown/zero size: s3://{bucket}/{key}"
                    );
                    (0u64, size, false)
                }
                SpecLocation::Http { url } => {
                    let client = http_client.ok_or_else(|| {
                        anyhow::anyhow!("http client not configured but http location present")
                    })?;
                    let size = http_head_len(client, url, r.sample_id).await?;
                    (0u64, size, false)
                }
            },
            _ => anyhow::bail!("partial byte range for sample_id {}", r.sample_id),
        };

        sample_ids.push(r.sample_id);
        specs.push(ReadSpec {
            sample_id: r.sample_id,
            location,
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

async fn fetch_specs_payload(
    specs: &[ReadSpec],
    s3_client: Option<&S3Client>,
    http_client: Option<&HttpClient>,
) -> Result<Vec<u8>> {
    #[cfg(not(feature = "s3"))]
    let _ = s3_client;

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
        match &s.location {
            SpecLocation::Local(path) => {
                let path = path.clone();
                let offset = s.offset;
                let expect_eof = s.expect_eof;
                let sample_id = s.sample_id;
                let bytes = tokio::task::spawn_blocking(move || -> Result<Vec<u8>> {
                    use std::io::{Read, Seek, SeekFrom};
                    let mut f = std::fs::File::open(&path)?;
                    f.seek(SeekFrom::Start(offset))?;
                    let mut buf = vec![0u8; len];
                    f.read_exact(&mut buf)?;

                    if expect_eof {
                        let mut tmp = [0u8; 1];
                        let extra = f.read(&mut tmp)?;
                        if extra != 0 {
                            anyhow::bail!(
                                "file grew after sizing (expected eof): sample_id={} path={}",
                                sample_id,
                                path.display()
                            );
                        }
                    }

                    Ok(buf)
                })
                .await
                .map_err(anyhow::Error::from)??;
                payload.extend_from_slice(&bytes);
            }
            #[cfg(feature = "s3")]
            SpecLocation::S3 { bucket, key } => {
                let client = s3_client.ok_or_else(|| {
                    anyhow::anyhow!("s3 client not configured but s3 location present")
                })?;
                let start = s.offset;
                let end = start
                    .checked_add(s.len)
                    .and_then(|v| v.checked_sub(1))
                    .ok_or_else(|| anyhow::anyhow!("invalid range length"))?;
                let range = format!("bytes={start}-{end}");

                let bucket = bucket.clone();
                let key = key.clone();
                let out = s3_with_retry(s.sample_id, || {
                    let bucket = bucket.clone();
                    let key = key.clone();
                    let range = range.clone();
                    async move {
                        client
                            .get_object()
                            .bucket(bucket)
                            .key(key)
                            .range(range)
                            .send()
                            .await
                    }
                })
                .await?;
                let b = collect_byte_stream(out.body).await?;
                anyhow::ensure!(
                    b.len() == len,
                    "s3 returned {} bytes, expected {} (sample_id={})",
                    b.len(),
                    len,
                    s.sample_id
                );
                payload.extend_from_slice(&b);
            }
            SpecLocation::Http { url } => {
                let client = http_client.ok_or_else(|| {
                    anyhow::anyhow!("http client not configured but http location present")
                })?;
                let b = http_range_get(client, url, s.offset, s.len, s.sample_id).await?;
                anyhow::ensure!(
                    b.len() == len,
                    "http returned {} bytes, expected {} (sample_id={})",
                    b.len(),
                    len,
                    s.sample_id
                );
                payload.extend_from_slice(&b);
            }
        }
    }

    Ok(payload)
}

#[cfg(feature = "s3")]
async fn collect_byte_stream(stream: ByteStream) -> Result<Vec<u8>> {
    let collected: AggregatedBytes = stream.collect().await?;
    Ok(collected.into_bytes().to_vec())
}

#[cfg(feature = "s3")]
fn s3_is_transient<E>(err: &S3SdkError<E>) -> bool {
    match err {
        S3SdkError::TimeoutError(_) => true,
        S3SdkError::DispatchFailure(_) => true,
        S3SdkError::ResponseError(_) => true,
        S3SdkError::ConstructionFailure(_) => false,
        S3SdkError::ServiceError(_) => err
            .raw_response()
            .map(|raw| {
                let status_u16: u16 = raw.status().into();
                status_u16 == 429 || status_u16 >= 500
            })
            .unwrap_or(false),
        _ => false,
    }
}

#[cfg(feature = "s3")]
async fn s3_with_retry<T, E, F, Fut>(sample_id: u64, mut f: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = std::result::Result<T, S3SdkError<E>>>,
    E: std::error::Error + Send + Sync + 'static,
{
    const MAX_ATTEMPTS: usize = 3;
    const BASE_DELAY_MS: u64 = 50;
    const MAX_DELAY_MS: u64 = 1000;

    let mut attempt: usize = 0;
    let mut delay_ms: u64 = BASE_DELAY_MS;
    loop {
        attempt = attempt.saturating_add(1);
        match f().await {
            Ok(v) => return Ok(v),
            Err(err) => {
                let transient = s3_is_transient(&err);
                if transient && attempt < MAX_ATTEMPTS {
                    let jitter = mx8_observe::time::unix_time_ms().wrapping_add(sample_id) % 37;
                    tokio::time::sleep(std::time::Duration::from_millis(
                        delay_ms.saturating_add(jitter),
                    ))
                    .await;
                    delay_ms = (delay_ms.saturating_mul(2)).min(MAX_DELAY_MS);
                    continue;
                }
                return Err(anyhow::Error::new(err));
            }
        }
    }
}

fn parse_location(location: &str) -> Result<SpecLocation> {
    if let Some(rest) = location.strip_prefix("s3://") {
        #[cfg(not(feature = "s3"))]
        {
            let _ = rest;
            anyhow::bail!("s3 support not enabled (rebuild mx8-runtime with --features s3)");
        }
        #[cfg(feature = "s3")]
        {
            let (bucket, key) = parse_s3_bucket_key(rest)?;
            return Ok(SpecLocation::S3 { bucket, key });
        }
    }
    if location.starts_with("http://") || location.starts_with("https://") {
        return Ok(SpecLocation::Http {
            url: location.to_string(),
        });
    }
    Ok(SpecLocation::Local(PathBuf::from(location)))
}

async fn http_head_len(client: &HttpClient, url: &str, sample_id: u64) -> Result<u64> {
    let resp = http_with_retry(sample_id, || async move {
        client
            .head(url)
            .header(reqwest::header::CONNECTION, "close")
            .send()
            .await
    })
    .await?;

    if !resp.status().is_success() {
        anyhow::bail!("http HEAD failed: status={} url={}", resp.status(), url);
    }

    let len = resp
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .ok_or_else(|| anyhow::anyhow!("http HEAD missing/invalid content-length: url={url}"))?;
    anyhow::ensure!(len > 0, "http HEAD returned zero length: url={url}");
    Ok(len)
}

async fn http_range_get(
    client: &HttpClient,
    url: &str,
    offset: u64,
    len: u64,
    sample_id: u64,
) -> Result<Vec<u8>> {
    let end = offset
        .checked_add(len)
        .and_then(|v| v.checked_sub(1))
        .ok_or_else(|| anyhow::anyhow!("invalid range length"))?;
    let range = format!("bytes={offset}-{end}");

    let resp = http_with_retry(sample_id, || {
        let range = range.clone();
        async move {
            client
                .get(url)
                .header(reqwest::header::CONNECTION, "close")
                .header(reqwest::header::RANGE, range)
                .send()
                .await
        }
    })
    .await?;

    if resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        anyhow::bail!(
            "http range GET failed: status={} url={} (expected 206)",
            resp.status(),
            url
        );
    }

    let bytes = resp.bytes().await?;
    Ok(bytes.to_vec())
}

async fn http_with_retry<F, Fut>(sample_id: u64, mut f: F) -> Result<reqwest::Response>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<reqwest::Response, reqwest::Error>>,
{
    const MAX_ATTEMPTS: usize = 5;
    const BASE_DELAY_MS: u64 = 50;
    const MAX_DELAY_MS: u64 = 1000;

    let mut attempt: usize = 0;
    let mut delay_ms: u64 = BASE_DELAY_MS;
    loop {
        attempt = attempt.saturating_add(1);
        match f().await {
            Ok(resp) => {
                let status = resp.status();
                let transient = status == reqwest::StatusCode::TOO_MANY_REQUESTS
                    || status == reqwest::StatusCode::REQUEST_TIMEOUT
                    || status.is_server_error();
                if transient && attempt < MAX_ATTEMPTS {
                    let jitter = mx8_observe::time::unix_time_ms().wrapping_add(sample_id) % 37;
                    tokio::time::sleep(std::time::Duration::from_millis(
                        delay_ms.saturating_add(jitter),
                    ))
                    .await;
                    delay_ms = (delay_ms.saturating_mul(2)).min(MAX_DELAY_MS);
                    continue;
                }
                return Ok(resp);
            }
            Err(err) => {
                let transient = err.is_timeout() || err.is_connect();
                if transient && attempt < MAX_ATTEMPTS {
                    let jitter = mx8_observe::time::unix_time_ms().wrapping_add(sample_id) % 37;
                    tokio::time::sleep(std::time::Duration::from_millis(
                        delay_ms.saturating_add(jitter),
                    ))
                    .await;
                    delay_ms = (delay_ms.saturating_mul(2)).min(MAX_DELAY_MS);
                    continue;
                }
                return Err(anyhow::Error::new(err));
            }
        }
    }
}

async fn build_manifest_inflight_batch(
    caps: RuntimeCaps,
    metrics: Arc<RuntimeMetrics>,
    inflight_sem: Arc<Semaphore>,
    records: &[mx8_core::types::ManifestRecord],
    s3_client: Option<&S3Client>,
    http_client: Option<&HttpClient>,
) -> Result<BatchLease> {
    let plan = build_batch_plan(records, s3_client, http_client).await?;
    let bytes = plan.total_bytes;
    anyhow::ensure!(
        bytes <= caps.max_inflight_bytes,
        "batch payload bytes {} exceeds max_inflight_bytes {}",
        bytes,
        caps.max_inflight_bytes
    );
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

    let permit = inflight_sem
        .clone()
        .acquire_many_owned(permit_units as u32)
        .await?;

    metrics.on_inflight_add(bytes);

    let payload = match fetch_specs_payload(&plan.specs, s3_client, http_client).await {
        Ok(p) => p,
        Err(err) => {
            metrics.on_inflight_sub(bytes);
            return Err(err);
        }
    };

    let mut offsets: Vec<u64> = Vec::with_capacity(plan.specs.len().saturating_add(1));
    offsets.push(0);
    let mut running: u64 = 0;
    for spec in &plan.specs {
        running = running
            .checked_add(spec.len)
            .ok_or_else(|| anyhow::anyhow!("offset overflow"))?;
        offsets.push(running);
    }
    anyhow::ensure!(
        offsets.len() == plan.sample_ids.len().saturating_add(1),
        "offsets length mismatch ({} vs ids {})",
        offsets.len(),
        plan.sample_ids.len()
    );
    anyhow::ensure!(
        offsets.last().copied().unwrap_or_default() == bytes,
        "offsets must end at payload length ({}), got {}",
        bytes,
        offsets.last().copied().unwrap_or_default()
    );

    let batch = Batch {
        sample_ids: Arc::from(plan.sample_ids.as_slice()),
        offsets: Arc::from(offsets.as_slice()),
        payload: Arc::from(payload.as_slice()),
    };

    Ok(BatchLease {
        batch,
        bytes,
        metrics,
        _permit: permit,
    })
}

#[cfg(feature = "s3")]
fn parse_s3_bucket_key(rest: &str) -> Result<(String, String)> {
    // rest = "<bucket>/<key...>"
    let rest = rest.trim().trim_start_matches('/');
    let Some((bucket, key)) = rest.split_once('/') else {
        anyhow::bail!("invalid s3 url (missing key): s3://{rest}");
    };
    let bucket = bucket.trim();
    let key = key.trim();
    anyhow::ensure!(!bucket.is_empty(), "invalid s3 url (empty bucket)");
    anyhow::ensure!(!key.is_empty(), "invalid s3 url (empty key)");
    Ok((bucket.to_string(), key.to_string()))
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

fn select_record_range(
    records: Vec<mx8_core::types::ManifestRecord>,
    start_id: u64,
    end_id: u64,
) -> Result<Vec<mx8_core::types::ManifestRecord>> {
    anyhow::ensure!(start_id <= end_id, "invalid range (start_id > end_id)");
    let len = records.len();
    let start = usize::try_from(start_id).map_err(|_| anyhow::anyhow!("start_id too large"))?;
    let end = usize::try_from(end_id).map_err(|_| anyhow::anyhow!("end_id too large"))?;
    anyhow::ensure!(start <= len, "start_id out of range");
    anyhow::ensure!(end <= len, "end_id out of range");
    Ok(records[start..end].to_vec())
}

#[cfg(all(test, feature = "s3"))]
mod s3_parse_tests {
    use super::*;

    #[test]
    fn parse_s3_bucket_key_ok() -> Result<()> {
        let (b, k) = parse_s3_bucket_key("mybucket/path/to.obj")?;
        assert_eq!(b, "mybucket");
        assert_eq!(k, "path/to.obj");
        Ok(())
    }

    #[test]
    fn parse_s3_bucket_key_rejects_missing_key() {
        let err = parse_s3_bucket_key("mybucket").unwrap_err();
        assert!(err.to_string().contains("missing key"));
    }
}
