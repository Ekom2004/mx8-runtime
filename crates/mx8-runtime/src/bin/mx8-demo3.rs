#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use std::io::Write;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

use mx8_runtime::pipeline::{Pipeline, RuntimeCaps};
use mx8_runtime::sink::Sink;
use mx8_runtime::types::Batch;

#[derive(Debug, Parser)]
#[command(name = "mx8-demo3")]
struct Args {
    #[arg(long, env = "MX8_TOTAL_SAMPLES", default_value_t = 1_024)]
    total_samples: u64,

    #[arg(long, env = "MX8_BYTES_PER_SAMPLE", default_value_t = 256)]
    bytes_per_sample: u64,

    #[arg(long, env = "MX8_BATCH_SIZE_SAMPLES", default_value_t = 16)]
    batch_size_samples: usize,

    #[arg(long, env = "MX8_MAX_QUEUE_BATCHES", default_value_t = 64)]
    max_queue_batches: usize,

    #[arg(long, env = "MX8_MAX_INFLIGHT_BYTES", default_value_t = 32 * 1024 * 1024)]
    max_inflight_bytes: u64,

    #[arg(long, env = "MX8_PREFETCH_COMPARE", default_value_t = 8)]
    prefetch_compare: usize,

    #[arg(long, env = "MX8_HTTP_LATENCY_MS", default_value_t = 25)]
    http_latency_ms: u64,

    #[arg(long, env = "MX8_HTTP_BANDWIDTH_BPS", default_value_t = 0)]
    http_bandwidth_bps: u64,

    /// Inject a transient HTTP failure every N requests (0 disables).
    #[arg(long, env = "MX8_HTTP_FAIL_EVERY_N", default_value_t = 0)]
    http_fail_every_n: u64,

    #[arg(long, env = "MX8_KEEP_ARTIFACTS", default_value_t = false)]
    keep_artifacts: bool,
}

struct CountingSink {
    delivered_samples: AtomicU64,
    delivered_bytes: AtomicU64,
}

impl CountingSink {
    fn new() -> Self {
        Self {
            delivered_samples: AtomicU64::new(0),
            delivered_bytes: AtomicU64::new(0),
        }
    }
}

impl Sink for CountingSink {
    fn deliver(&self, batch: Batch) -> Result<()> {
        self.delivered_samples
            .fetch_add(batch.sample_count() as u64, Ordering::Relaxed);
        self.delivered_bytes
            .fetch_add(batch.payload_len() as u64, Ordering::Relaxed);
        Ok(())
    }
}

fn demo_temp_root() -> PathBuf {
    let mut root = std::env::temp_dir();
    root.push(format!(
        "mx8-demo3-{}-{}",
        std::process::id(),
        mx8_observe::time::unix_time_ms()
    ));
    root
}

fn create_sparse_data_file(path: &Path, size_bytes: u64) -> anyhow::Result<()> {
    let f = std::fs::File::create(path)?;
    f.set_len(size_bytes)?;
    Ok(())
}

fn build_http_manifest(
    addr: SocketAddr,
    data_path: &str,
    total_samples: u64,
    bytes_per_sample: u64,
) -> Result<Vec<u8>> {
    let url = format!("http://{addr}{data_path}");
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

async fn run_once(manifest_bytes: &[u8], caps: RuntimeCaps) -> Result<(u64, u64, u64, u128)> {
    let pipeline = Pipeline::new(caps);
    let metrics = pipeline.metrics();
    let sink = Arc::new(CountingSink::new());

    let start = Instant::now();
    pipeline
        .run_manifest_bytes(sink.clone(), manifest_bytes)
        .await?;
    let elapsed = start.elapsed().as_millis();

    Ok((
        sink.delivered_samples.load(Ordering::Relaxed),
        sink.delivered_bytes.load(Ordering::Relaxed),
        metrics.inflight_bytes_high_water.get(),
        elapsed,
    ))
}

#[derive(Clone)]
struct ServerConfig {
    file_path: PathBuf,
    latency: Duration,
    bandwidth_bps: u64,
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

    let (start, _end_inclusive, status_line, extra_headers, body_len) = match range {
        Some((start, end)) => {
            anyhow::ensure!(start < file_len, "range start out of bounds");
            let end = end.min(file_len.saturating_sub(1));
            anyhow::ensure!(end >= start, "invalid range");
            let len = end - start + 1;
            let headers = format!(
                "Content-Range: bytes {start}-{end}/{file_len}\r\nAccept-Ranges: bytes\r\n"
            );
            (start, end, "HTTP/1.1 206 Partial Content", headers, len)
        }
        None => (
            0,
            file_len.saturating_sub(1),
            "HTTP/1.1 200 OK",
            "Accept-Ranges: bytes\r\n".to_string(),
            file_len,
        ),
    };

    if cfg.bandwidth_bps > 0 {
        let extra_ms = body_len
            .saturating_mul(1000)
            .saturating_div(cfg.bandwidth_bps);
        if extra_ms > 0 {
            tokio::time::sleep(Duration::from_millis(extra_ms)).await;
        }
    }

    let bytes = tokio::task::spawn_blocking({
        let p = cfg.file_path.clone();
        move || -> Result<Vec<u8>> {
            use std::io::{Read, Seek, SeekFrom};
            let mut f = std::fs::File::open(p)?;
            f.seek(SeekFrom::Start(start))?;
            let mut out = vec![
                0u8;
                usize::try_from(body_len)
                    .map_err(|_| anyhow::anyhow!("body too large"))?
            ];
            f.read_exact(&mut out)?;
            Ok(out)
        }
    })
    .await
    .map_err(anyhow::Error::from)??;

    let headers = format!(
        "{status_line}\r\nContent-Length: {body_len}\r\n{extra_headers}Connection: close\r\n\r\n"
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
                _ = &mut shutdown_rx => {
                    break;
                }
                res = listener.accept() => {
                    let Ok((sock, _peer)) = res else {
                        break;
                    };
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

#[tokio::main]
async fn main() -> Result<()> {
    mx8_observe::logging::init_tracing();
    let args = Args::parse();

    anyhow::ensure!(args.bytes_per_sample > 0, "bytes_per_sample must be > 0");
    anyhow::ensure!(args.total_samples > 0, "total_samples must be > 0");
    anyhow::ensure!(
        args.batch_size_samples > 0,
        "batch_size_samples must be > 0"
    );
    anyhow::ensure!(args.prefetch_compare > 0, "prefetch_compare must be > 0");

    let root = demo_temp_root();
    std::fs::create_dir_all(&root)?;
    let data_file = root.join("data.bin");

    let total_bytes = args
        .total_samples
        .checked_mul(args.bytes_per_sample)
        .ok_or_else(|| anyhow::anyhow!("total bytes overflow"))?;
    create_sparse_data_file(&data_file, total_bytes)?;

    let server_cfg = ServerConfig {
        file_path: data_file.clone(),
        latency: Duration::from_millis(args.http_latency_ms),
        bandwidth_bps: args.http_bandwidth_bps,
        fail_every_n: args.http_fail_every_n,
        request_counter: Arc::new(AtomicU64::new(0)),
    };
    let (addr, shutdown) = spawn_range_server(server_cfg).await?;
    let manifest_bytes =
        build_http_manifest(addr, "/data.bin", args.total_samples, args.bytes_per_sample)?;

    let base_caps = RuntimeCaps {
        max_inflight_bytes: args.max_inflight_bytes,
        max_queue_batches: args.max_queue_batches,
        batch_size_samples: args.batch_size_samples,
        prefetch_batches: 1,
        target_batch_bytes: None,
        max_batch_bytes: None,
        max_process_rss_bytes: None,
    };

    let (s1, b1, hw1, ms1) = run_once(&manifest_bytes, base_caps).await?;
    let sps1 = if ms1 == 0 {
        0.0
    } else {
        (s1 as f64) * 1000.0 / (ms1 as f64)
    };
    println!(
        "[demo3] prefetch=1 elapsed_ms={} samples={} bytes={} inflight_high_water={} samples_per_sec={}",
        ms1, s1, b1, hw1, sps1
    );

    let compare_caps = RuntimeCaps {
        prefetch_batches: args.prefetch_compare,
        ..base_caps
    };
    let (s2, b2, hw2, ms2) = run_once(&manifest_bytes, compare_caps).await?;
    let sps2 = if ms2 == 0 {
        0.0
    } else {
        (s2 as f64) * 1000.0 / (ms2 as f64)
    };
    println!(
        "[demo3] prefetch={} elapsed_ms={} samples={} bytes={} inflight_high_water={} samples_per_sec={}",
        args.prefetch_compare, ms2, s2, b2, hw2, sps2
    );

    let _ = shutdown.send(());

    if args.keep_artifacts {
        let mut meta = std::fs::File::create(root.join("meta.txt"))?;
        writeln!(meta, "addr={addr}")?;
        writeln!(meta, "data_file={}", data_file.display())?;
        println!("[demo3] kept artifacts: {}", root.display());
    } else {
        let _ = std::fs::remove_dir_all(&root);
    }

    Ok(())
}
