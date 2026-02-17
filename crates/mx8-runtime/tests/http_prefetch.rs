use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

use mx8_runtime::pipeline::{Pipeline, RuntimeCaps};
use mx8_runtime::sink::Sink;
use mx8_runtime::types::Batch;

struct CountingSink {
    delivered_samples: AtomicU64,
}

impl CountingSink {
    fn new() -> Self {
        Self {
            delivered_samples: AtomicU64::new(0),
        }
    }
}

impl Sink for CountingSink {
    fn deliver(&self, batch: Batch) -> Result<()> {
        self.delivered_samples
            .fetch_add(batch.sample_count() as u64, Ordering::Relaxed);
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
    latency: Duration,
    fail_every_n: u64,
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

    let req_no = cfg.request_counter.fetch_add(1, Ordering::Relaxed) + 1;
    if cfg.fail_every_n > 0 && req_no.is_multiple_of(cfg.fail_every_n) {
        tokio::time::sleep(cfg.latency).await;
        sock.write_all(
            b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        )
        .await?;
        sock.shutdown().await?;
        return Ok(());
    }

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

    anyhow::ensure!(path == "/data.bin", "unknown path: {path}");
    let file_len = tokio::task::spawn_blocking({
        let p = cfg.file_path.clone();
        move || -> Result<u64> { Ok(std::fs::metadata(p)?.len()) }
    })
    .await
    .map_err(anyhow::Error::from)??;

    tokio::time::sleep(cfg.latency).await;

    if method.eq_ignore_ascii_case("HEAD") {
        let headers = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {file_len}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n"
        );
        sock.write_all(headers.as_bytes()).await?;
        sock.shutdown().await?;
        return Ok(());
    }

    anyhow::ensure!(
        method.eq_ignore_ascii_case("GET"),
        "unsupported method: {method}"
    );
    let Some((start, end)) = range else {
        anyhow::bail!("range header required for test server");
    };
    anyhow::ensure!(start < file_len, "range start out of bounds");
    let end = end.min(file_len.saturating_sub(1));
    anyhow::ensure!(end >= start, "invalid range");
    let body_len = end - start + 1;

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
        "HTTP/1.1 206 Partial Content\r\nContent-Length: {body_len}\r\nContent-Range: bytes {start}-{end}/{file_len}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n"
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
async fn http_manifest_prefetch_completes_and_respects_caps() -> Result<()> {
    let root = temp_dir("http-prefetch")?;
    let data_file = root.join("data.bin");

    let total_samples: u64 = 512;
    let bytes_per_sample: u64 = 256;
    let total_bytes = total_samples
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| anyhow::anyhow!("overflow"))?;
    create_sparse_data_file(&data_file, total_bytes)?;

    let (addr, shutdown) = spawn_range_server(ServerConfig {
        file_path: data_file,
        latency: Duration::from_millis(5),
        fail_every_n: 0,
        request_counter: Arc::new(AtomicU64::new(0)),
    })
    .await?;

    let manifest = build_http_manifest(addr, total_samples, bytes_per_sample)?;

    let caps = RuntimeCaps {
        max_inflight_bytes: 64 * 1024,
        max_queue_batches: 16,
        batch_size_samples: 8,
        prefetch_batches: 4,
        target_batch_bytes: None,
        max_batch_bytes: None,
        max_process_rss_bytes: None,
    };
    let pipeline = Pipeline::new(caps);
    let metrics = pipeline.metrics();

    let sink = Arc::new(CountingSink::new());
    pipeline.run_manifest_bytes(sink.clone(), &manifest).await?;
    let _ = shutdown.send(());

    assert_eq!(
        sink.delivered_samples.load(Ordering::Relaxed),
        total_samples
    );
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
async fn http_manifest_retries_transient_503() -> Result<()> {
    let root = temp_dir("http-prefetch-retry")?;
    let data_file = root.join("data.bin");

    let total_samples: u64 = 128;
    let bytes_per_sample: u64 = 256;
    let total_bytes = total_samples
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| anyhow::anyhow!("overflow"))?;
    create_sparse_data_file(&data_file, total_bytes)?;

    let (addr, shutdown) = spawn_range_server(ServerConfig {
        file_path: data_file,
        latency: Duration::from_millis(1),
        fail_every_n: 7,
        request_counter: Arc::new(AtomicU64::new(0)),
    })
    .await?;

    let manifest = build_http_manifest(addr, total_samples, bytes_per_sample)?;

    let caps = RuntimeCaps {
        max_inflight_bytes: 64 * 1024,
        max_queue_batches: 16,
        batch_size_samples: 8,
        prefetch_batches: 4,
        target_batch_bytes: None,
        max_batch_bytes: None,
        max_process_rss_bytes: None,
    };
    let pipeline = Pipeline::new(caps);
    let metrics = pipeline.metrics();

    let sink = Arc::new(CountingSink::new());
    pipeline.run_manifest_bytes(sink.clone(), &manifest).await?;
    let _ = shutdown.send(());

    assert_eq!(
        sink.delivered_samples.load(Ordering::Relaxed),
        total_samples
    );
    let high_water = metrics.inflight_bytes_high_water.get();
    assert!(
        high_water <= caps.max_inflight_bytes,
        "inflight high-water {} > cap {}",
        high_water,
        caps.max_inflight_bytes
    );
    Ok(())
}
