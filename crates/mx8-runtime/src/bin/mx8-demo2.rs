#![forbid(unsafe_code)]
#![cfg_attr(not(test), deny(clippy::expect_used, clippy::unwrap_used))]

use std::fs::File;
use std::io::{BufWriter, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::watch;

#[derive(Debug, Parser)]
#[command(name = "mx8-demo2")]
struct Args {
    #[arg(long, env = "MX8_WORLD_SIZE", default_value_t = 2)]
    world_size: u32,

    #[arg(long, env = "MX8_TOTAL_SAMPLES", default_value_t = 80_000)]
    total_samples: u64,

    #[arg(long, env = "MX8_BYTES_PER_SAMPLE", default_value_t = 256)]
    bytes_per_sample: u64,

    #[arg(long, env = "MX8_DEV_BLOCK_SIZE", default_value_t = 10_000)]
    block_size: u64,

    #[arg(long, env = "MX8_BATCH_SIZE_SAMPLES", default_value_t = 512)]
    batch_size_samples: usize,

    #[arg(long, env = "MX8_MAX_QUEUE_BATCHES", default_value_t = 64)]
    max_queue_batches: usize,

    #[arg(long, env = "MX8_MAX_INFLIGHT_BYTES", default_value_t = 32 * 1024 * 1024)]
    max_inflight_bytes: u64,

    #[arg(long, env = "MX8_SINK_SLEEP_MS", default_value_t = 25)]
    sink_sleep_ms: u64,

    #[arg(long, env = "MX8_LEASE_TTL_MS", default_value_t = 2000)]
    lease_ttl_ms: u32,

    #[arg(long, env = "MX8_HEARTBEAT_INTERVAL_MS", default_value_t = 200)]
    heartbeat_interval_ms: u32,

    #[arg(long, env = "MX8_PROGRESS_INTERVAL_MS", default_value_t = 200)]
    progress_interval_ms: u64,

    #[arg(long, env = "MX8_JOB_ID", default_value = "demo2")]
    job_id: String,

    #[arg(long, env = "MX8_DATASET_LINK", default_value = "demo://demo2/")]
    dataset_link: String,

    /// Which node to kill after startup (1-based).
    ///
    /// Use 0 to attempt auto-selecting the first observed lease owner (best-effort).
    #[arg(long, env = "MX8_KILL_NODE_INDEX", default_value_t = 1)]
    kill_node_index: u32,

    #[arg(long, env = "MX8_KILL_AFTER_MS", default_value_t = 750)]
    kill_after_ms: u64,

    #[arg(long, env = "MX8_WAIT_REQUEUE_TIMEOUT_MS", default_value_t = 15_000)]
    wait_requeue_timeout_ms: u64,

    #[arg(long, env = "MX8_WAIT_DRAIN_TIMEOUT_MS", default_value_t = 60_000)]
    wait_drain_timeout_ms: u64,

    #[arg(long, env = "MX8_COORD_BIND_ADDR", default_value = "127.0.0.1:50051")]
    coord_bind_addr: String,

    #[arg(long, env = "MX8_COORD_URL", default_value = "http://127.0.0.1:50051")]
    coord_url: String,

    #[arg(long, env = "MX8_KEEP_ARTIFACTS", default_value_t = false)]
    keep_artifacts: bool,
}

#[derive(Debug, Clone, Default)]
struct DemoEvents {
    saw_requeued: bool,
    saw_drained: bool,
    first_lease_node_id: Option<String>,
}

fn find_free_local_port(start: u16) -> anyhow::Result<u16> {
    for port in start..=start.saturating_add(200) {
        if let Ok(listener) = TcpListener::bind(("127.0.0.1", port)) {
            drop(listener);
            return Ok(port);
        }
    }
    Err(anyhow::anyhow!("no free port found starting at {start}"))
}

fn workspace_root() -> anyhow::Result<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .ok_or_else(|| anyhow::anyhow!("cannot derive workspace root from CARGO_MANIFEST_DIR"))?;
    Ok(root.to_path_buf())
}

fn create_sparse_data_file(path: &Path, size_bytes: u64) -> anyhow::Result<()> {
    let f = File::create(path)?;
    f.set_len(size_bytes)?;
    Ok(())
}

fn write_dev_manifest(
    path: &Path,
    data_file: &Path,
    total_samples: u64,
    bytes_per_sample: u64,
) -> anyhow::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    let data_path = data_file.display();
    for i in 0..total_samples {
        let off = i
            .checked_mul(bytes_per_sample)
            .ok_or_else(|| anyhow::anyhow!("byte offset overflow"))?;
        writeln!(w, "{i}\t{data_path}\t{off}\t{bytes_per_sample}")?;
    }
    w.flush()?;
    Ok(())
}

fn demo_temp_root() -> PathBuf {
    let mut root = std::env::temp_dir();
    root.push(format!(
        "mx8-demo2-{}-{}",
        std::process::id(),
        mx8_observe::time::unix_time_ms()
    ));
    root
}

async fn read_coordinator_logs(
    mut stdout: tokio::process::ChildStdout,
    events_tx: watch::Sender<DemoEvents>,
) -> anyhow::Result<()> {
    let mut reader = BufReader::new(&mut stdout).lines();
    while let Some(line) = reader.next_line().await? {
        println!("coord| {line}");
        let mut cur = (*events_tx.borrow()).clone();
        if line.contains("event=\"range_requeued\"") {
            cur.saw_requeued = true;
        }
        if line.contains("event=\"job_drained\"") {
            cur.saw_drained = true;
        }
        if cur.first_lease_node_id.is_none() && line.contains("event=\"lease_granted\"") {
            if let Some(node_id) = parse_field_value(&line, "node_id=") {
                cur.first_lease_node_id = Some(node_id);
            }
        }
        let _ = events_tx.send(cur);
    }
    Ok(())
}

fn parse_field_value(line: &str, key: &str) -> Option<String> {
    let idx = line.find(key)?;
    let rest = &line[idx + key.len()..];
    let end = rest
        .find(|c: char| c.is_whitespace() || c == ',' || c == '}')
        .unwrap_or(rest.len());
    let value = rest[..end].trim();
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

async fn wait_for<F>(
    mut rx: watch::Receiver<DemoEvents>,
    timeout: Duration,
    f: F,
) -> anyhow::Result<()>
where
    F: Fn(DemoEvents) -> bool,
{
    let fut = async move {
        loop {
            if f(rx.borrow().clone()) {
                return Ok(());
            }
            rx.changed().await.map_err(anyhow::Error::from)?;
        }
    };
    tokio::time::timeout(timeout, fut)
        .await
        .map_err(|_| anyhow::anyhow!("timeout"))?
}

#[tokio::main]
async fn main() -> Result<()> {
    mx8_observe::logging::init_tracing();
    let args = Args::parse();

    anyhow::ensure!(args.world_size >= 2, "world_size must be >= 2 for demo2");
    anyhow::ensure!(args.bytes_per_sample > 0, "bytes_per_sample must be > 0");
    anyhow::ensure!(args.block_size > 0, "block_size must be > 0");
    if args.kill_node_index != 0 {
        anyhow::ensure!(
            args.kill_node_index >= 1 && args.kill_node_index <= args.world_size,
            "kill_node_index must be 0 (auto) or in [1..world_size]"
        );
    }

    let root = demo_temp_root();
    let store_root = root.join("store");
    let cache_dir = root.join("cache");
    let data_file = root.join("data.bin");
    let dev_manifest = root.join("dev_manifest.tsv");

    std::fs::create_dir_all(&store_root)?;
    std::fs::create_dir_all(&cache_dir)?;

    let total_bytes = args
        .total_samples
        .checked_mul(args.bytes_per_sample)
        .ok_or_else(|| anyhow::anyhow!("total bytes overflow"))?;

    println!("[demo2] tmp_root={}", root.display());
    println!(
        "[demo2] generating data file ({} bytes) + manifest ({} samples)",
        total_bytes, args.total_samples
    );
    create_sparse_data_file(&data_file, total_bytes)?;
    write_dev_manifest(
        &dev_manifest,
        &data_file,
        args.total_samples,
        args.bytes_per_sample,
    )?;

    let ws_root = workspace_root()?;

    println!("[demo2] building mx8-coordinator + mx8d-agent");
    let status = std::process::Command::new("cargo")
        .current_dir(&ws_root)
        .args(["build", "-p", "mx8-coordinator", "-p", "mx8d-agent"])
        .status()?;
    anyhow::ensure!(status.success(), "cargo build failed");

    let coord_bin = ws_root.join("target").join("debug").join("mx8-coordinator");
    let agent_bin = ws_root.join("target").join("debug").join("mx8d-agent");
    anyhow::ensure!(coord_bin.exists(), "missing {}", coord_bin.display());
    anyhow::ensure!(agent_bin.exists(), "missing {}", agent_bin.display());

    let (coord_bind_addr, coord_url) = {
        let default_bind = "127.0.0.1:50051";
        let default_url = "http://127.0.0.1:50051";
        if args.coord_bind_addr == default_bind && args.coord_url == default_url {
            let port = find_free_local_port(50051)?;
            (
                format!("127.0.0.1:{port}"),
                format!("http://127.0.0.1:{port}"),
            )
        } else {
            (args.coord_bind_addr.clone(), args.coord_url.clone())
        }
    };

    println!(
        "[demo2] starting coordinator (bind={}, url={})",
        coord_bind_addr, coord_url
    );
    let mut coord = Command::new(&coord_bin)
        .arg("--world-size")
        .arg(args.world_size.to_string())
        .env("MX8_COORD_BIND_ADDR", &coord_bind_addr)
        .env("MX8_WORLD_SIZE", args.world_size.to_string())
        .env(
            "MX8_HEARTBEAT_INTERVAL_MS",
            args.heartbeat_interval_ms.to_string(),
        )
        .env("MX8_LEASE_TTL_MS", args.lease_ttl_ms.to_string())
        .env("MX8_DATASET_LINK", format!("{}@refresh", args.dataset_link))
        .env("MX8_MANIFEST_STORE_ROOT", &store_root)
        .env("MX8_DEV_MANIFEST_PATH", &dev_manifest)
        .env("MX8_DEV_BLOCK_SIZE", args.block_size.to_string())
        .env("MX8_METRICS_SNAPSHOT_INTERVAL_MS", "0")
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;

    let coord_stdout = coord
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("failed to capture coordinator stdout"))?;

    let (events_tx, events_rx) = watch::channel(DemoEvents::default());
    let coord_logs_task = tokio::spawn(read_coordinator_logs(coord_stdout, events_tx));

    tokio::time::sleep(Duration::from_millis(750)).await;

    println!("[demo2] starting agents (world_size={})", args.world_size);
    let mut agents = Vec::new();
    let mut agent_node_ids = Vec::new();
    for i in 1..=args.world_size {
        let node_id = format!("node{i}");
        agent_node_ids.push(node_id.clone());
        let child = Command::new(&agent_bin)
            .env("MX8_COORD_URL", &coord_url)
            .env("MX8_JOB_ID", &args.job_id)
            .env("MX8_NODE_ID", node_id)
            .env("MX8_MANIFEST_CACHE_DIR", &cache_dir)
            .env("MX8_DEV_LEASE_WANT", "1")
            .env(
                "MX8_BATCH_SIZE_SAMPLES",
                args.batch_size_samples.to_string(),
            )
            .env("MX8_MAX_QUEUE_BATCHES", args.max_queue_batches.to_string())
            .env(
                "MX8_MAX_INFLIGHT_BYTES",
                args.max_inflight_bytes.to_string(),
            )
            .env("MX8_SINK_SLEEP_MS", args.sink_sleep_ms.to_string())
            .env(
                "MX8_PROGRESS_INTERVAL_MS",
                args.progress_interval_ms.to_string(),
            )
            .env("MX8_METRICS_SNAPSHOT_INTERVAL_MS", "0")
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()?;
        agents.push(child);
    }

    let kill_idx = if args.kill_node_index == 0 {
        println!("[demo2] waiting for first lease_granted (to pick kill target)");
        let _ = wait_for(events_rx.clone(), Duration::from_millis(5000), |e| {
            e.first_lease_node_id.is_some()
        })
        .await;

        let lease_owner = events_rx.borrow().first_lease_node_id.clone();
        match lease_owner {
            Some(lease_owner) => {
                let idx = agent_node_ids
                    .iter()
                    .position(|n| n == &lease_owner)
                    .unwrap_or(0);
                println!("[demo2] auto kill lease owner node_id={lease_owner}");
                idx
            }
            None => {
                println!("[demo2] auto-select failed; falling back to kill node index=1");
                0
            }
        }
    } else {
        usize::try_from(args.kill_node_index - 1)?
    };

    tokio::time::sleep(Duration::from_millis(args.kill_after_ms)).await;
    println!(
        "[demo2] killing agent index={} node_id={}",
        kill_idx + 1,
        agent_node_ids[kill_idx]
    );
    if let Some(child) = agents.get_mut(kill_idx) {
        let _ = child.kill().await;
    }

    println!("[demo2] waiting for range_requeued");
    wait_for(
        events_rx.clone(),
        Duration::from_millis(args.wait_requeue_timeout_ms),
        |e| e.saw_requeued,
    )
    .await
    .map_err(|_| anyhow::anyhow!("timeout waiting for range_requeued"))?;

    println!("[demo2] waiting for job_drained");
    wait_for(
        events_rx,
        Duration::from_millis(args.wait_drain_timeout_ms),
        |e| e.saw_drained,
    )
    .await
    .map_err(|_| anyhow::anyhow!("timeout waiting for job_drained"))?;

    println!("[demo2] cleanup");
    for child in &mut agents {
        let _ = child.kill().await;
    }
    let _ = coord.kill().await;
    let _ = coord_logs_task.await;

    if args.keep_artifacts {
        println!("[demo2] kept artifacts: {}", root.display());
    } else {
        let _ = std::fs::remove_dir_all(&root);
    }
    Ok(())
}
