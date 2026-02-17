use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

use mx8_runtime::pipeline::{Pipeline, RuntimeCaps};
use mx8_runtime::sink::Sink;
use mx8_runtime::types::Batch;

struct RecordingSink {
    sample_ids: Mutex<Vec<u64>>,
}

impl RecordingSink {
    fn new() -> Self {
        Self {
            sample_ids: Mutex::new(Vec::new()),
        }
    }

    fn take(&self) -> Vec<u64> {
        self.sample_ids
            .lock()
            .map(|mut v| std::mem::take(&mut *v))
            .unwrap_or_default()
    }
}

impl Sink for RecordingSink {
    fn deliver(&self, batch: Batch) -> Result<()> {
        let mut guard = self
            .sample_ids
            .lock()
            .map_err(|_| anyhow::anyhow!("recording sink mutex poisoned"))?;
        guard.extend(batch.sample_ids.iter().copied());
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

fn create_sparse_data_file(path: &Path, size_bytes: u64) -> Result<()> {
    let f = std::fs::File::create(path)?;
    f.set_len(size_bytes)?;
    Ok(())
}

fn build_http_manifest(
    addr: SocketAddr,
    total_samples: u64,
    bytes_per_sample: u64,
) -> Result<Vec<u8>> {
    let url = format!("http://{addr}/data.bin");
    let mut out = String::new();
    out.push_str("schema_version=0\n");
    for i in 0..total_samples {
        let off = i
            .checked_mul(bytes_per_sample)
            .ok_or_else(|| anyhow::anyhow!("byte offset overflow"))?;
        out.push_str(&format!("{i}\t{url}\t{off}\t{bytes_per_sample}\n"));
    }
    Ok(out.into_bytes())
}

#[derive(Clone)]
struct ServerConfig {
    file_path: PathBuf,
    bytes_per_sample: u64,
    batch_size_samples: usize,
    request_counter: Arc<AtomicU64>,
}

async fn serve_one_connection(mut sock: tokio::net::TcpStream, cfg: ServerConfig) -> Result<()> {
    let mut buf = vec![0u8; 64 * 1024];
    let mut n: usize = 0;
    loop {
        let read = sock.read(&mut buf[n..]).await?;
        if read == 0 {
            anyhow::bail!("client disconnected before request complete");
        }
        n = n.saturating_add(read);
        if n >= 4 && buf[..n].windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
        anyhow::ensure!(n < buf.len(), "request headers too large");
    }

    let req = std::str::from_utf8(&buf[..n]).map_err(|e| anyhow::anyhow!("bad utf8: {e}"))?;
    let mut lines = req.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing request line"))?;
    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing method"))?;
    let path = parts
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing path"))?;

    let mut range: Option<(u64, u64)> = None;
    for line in lines {
        if line.is_empty() {
            break;
        }
        let Some((k, v)) = line.split_once(':') else {
            continue;
        };
        if k.trim().eq_ignore_ascii_case("range") {
            let v = v.trim();
            if let Some(r) = v.strip_prefix("bytes=") {
                if let Some((a, b)) = r.split_once('-') {
                    if let (Ok(start), Ok(end)) = (a.trim().parse::<u64>(), b.trim().parse::<u64>())
                    {
                        range = Some((start, end));
                    }
                }
            }
        }
    }

    anyhow::ensure!(
        method.eq_ignore_ascii_case("GET"),
        "unsupported method: {method}"
    );
    anyhow::ensure!(path == "/data.bin", "unknown path: {path}");
    let Some((start, end)) = range else {
        anyhow::bail!("range header required");
    };
    anyhow::ensure!(end >= start, "invalid range");
    let body_len = end - start + 1;

    // Inject some jitter so later batches often finish before earlier ones, forcing the
    // prefetch in-order buffer to be exercised.
    let request_no = cfg.request_counter.fetch_add(1, Ordering::Relaxed);
    let batch_bytes = cfg
        .bytes_per_sample
        .saturating_mul(cfg.batch_size_samples as u64);
    let slow_batch = start < batch_bytes;
    let base_ms: u64 = if slow_batch { 10 } else { 1 };
    let jitter_ms = request_no % 3;
    tokio::time::sleep(Duration::from_millis(base_ms.saturating_add(jitter_ms))).await;

    let bytes = tokio::task::spawn_blocking({
        let p = cfg.file_path.clone();
        move || -> Result<Vec<u8>> {
            use std::io::{Read, Seek, SeekFrom};
            let mut f = std::fs::File::open(p)?;
            f.seek(SeekFrom::Start(start))?;
            let mut out =
                vec![0u8; usize::try_from(body_len).map_err(|_| anyhow::anyhow!("too large"))?];
            f.read_exact(&mut out)?;
            Ok(out)
        }
    })
    .await
    .map_err(anyhow::Error::from)??;

    let headers = format!(
        "HTTP/1.1 206 Partial Content\r\nContent-Length: {body_len}\r\nContent-Range: bytes {start}-{end}/0\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n"
    );
    sock.write_all(headers.as_bytes()).await?;
    sock.write_all(&bytes).await?;
    sock.shutdown().await?;
    Ok(())
}

async fn spawn_range_server(cfg: ServerConfig) -> Result<(SocketAddr, oneshot::Sender<()>)> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).await?;
    let addr = listener.local_addr()?;
    let (shutdown_tx, mut shutdown_rx) = oneshot::channel::<()>();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = &mut shutdown_rx => { break; }
                res = listener.accept() => {
                    let Ok((sock, _peer)) = res else { break; };
                    let cfg = cfg.clone();
                    tokio::spawn(async move {
                        let _ = serve_one_connection(sock, cfg).await;
                    });
                }
            }
        }
    });
    Ok((addr, shutdown_tx))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prefetch_preserves_delivery_order() -> Result<()> {
    let root = temp_dir("prefetch-order")?;
    let data_file = root.join("data.bin");

    let total_samples: u64 = 128;
    let bytes_per_sample: u64 = 256;
    let batch_size_samples: usize = 8;
    let total_bytes = total_samples
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| anyhow::anyhow!("overflow"))?;
    create_sparse_data_file(&data_file, total_bytes)?;

    let (addr, shutdown) = spawn_range_server(ServerConfig {
        file_path: data_file,
        bytes_per_sample,
        batch_size_samples,
        request_counter: Arc::new(AtomicU64::new(0)),
    })
    .await?;

    let manifest = build_http_manifest(addr, total_samples, bytes_per_sample)?;

    let caps = RuntimeCaps {
        max_inflight_bytes: 64 * 1024,
        max_queue_batches: 16,
        batch_size_samples,
        prefetch_batches: 8,
        target_batch_bytes: None,
        max_batch_bytes: None,
        max_process_rss_bytes: None,
    };
    let pipeline = Pipeline::new(caps);
    let metrics = pipeline.metrics();

    let sink = Arc::new(RecordingSink::new());
    pipeline.run_manifest_bytes(sink.clone(), &manifest).await?;
    let _ = shutdown.send(());

    let got = sink.take();
    let want: Vec<u64> = (0..total_samples).collect();
    assert_eq!(got, want, "prefetch must preserve deterministic order");

    let high_water = metrics.inflight_bytes_high_water.get();
    assert!(
        high_water <= caps.max_inflight_bytes,
        "inflight high-water {} > cap {}",
        high_water,
        caps.max_inflight_bytes
    );

    Ok(())
}
