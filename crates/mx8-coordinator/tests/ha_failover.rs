#![forbid(unsafe_code)]

use std::net::{SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use mx8_proto::v0::coordinator_client::CoordinatorClient;
use mx8_proto::v0::{
    GetJobSnapshotRequest, HeartbeatRequest, NodeCaps, RegisterNodeRequest, ReportProgressRequest,
    RequestLeaseRequest,
};
use tonic::Code;

const JOB_ID: &str = "ha-failover-gate";
const NODE_ID: &str = "node-1";
static RUN_COUNTER: AtomicU64 = AtomicU64::new(0);

struct CoordinatorProc {
    child: Option<Child>,
}

impl CoordinatorProc {
    fn spawn(
        bin: &Path,
        addr: SocketAddr,
        manifest_store_root: &Path,
        leader_lease_path: &Path,
        state_store_path: &Path,
        leader_id: &str,
    ) -> Result<Self> {
        let child = Command::new(bin)
            .arg("--addr")
            .arg(addr.to_string())
            .arg("--world-size")
            .arg("1")
            .arg("--min-world-size")
            .arg("1")
            .arg("--dev-total-samples")
            .arg("100")
            .arg("--dev-block-size")
            .arg("10")
            .arg("--manifest-store-root")
            .arg(manifest_store_root)
            .arg("--lease-log-path")
            .arg("none")
            .arg("--ha-enable")
            .arg("--ha-lease-path")
            .arg(leader_lease_path)
            .arg("--ha-leader-id")
            .arg(leader_id)
            .arg("--ha-lease-ttl-ms")
            .arg("600")
            .arg("--ha-renew-interval-ms")
            .arg("100")
            .arg("--state-store-enable")
            .arg("--state-store-path")
            .arg(state_store_path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("failed to spawn coordinator {leader_id}"))?;
        Ok(Self { child: Some(child) })
    }

    fn kill_and_wait(&mut self) -> Result<()> {
        let Some(mut child) = self.child.take() else {
            return Ok(());
        };
        let _ = child.kill();
        let _ = child.wait()?;
        Ok(())
    }
}

impl Drop for CoordinatorProc {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn unique_dir(name: &str) -> Result<PathBuf> {
    let run_id = RUN_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "mx8-ha-failover-{name}-{}-{}-{run_id}",
        std::process::id(),
        unix_time_ms()
    ));
    std::fs::create_dir_all(&path)?;
    Ok(path)
}

fn resolve_coordinator_bin() -> Result<String> {
    if let Ok(p) = std::env::var("CARGO_BIN_EXE_mx8-coordinator") {
        return Ok(p);
    }
    if let Ok(p) = std::env::var("CARGO_BIN_EXE_mx8_coordinator") {
        return Ok(p);
    }
    let exe = std::env::current_exe().context("failed to resolve current test binary path")?;
    let debug_dir = exe
        .parent()
        .and_then(|p| p.parent())
        .ok_or_else(|| anyhow::anyhow!("failed to derive target/debug from test binary"))?;
    Ok(debug_dir
        .join("mx8-coordinator")
        .to_string_lossy()
        .to_string())
}

fn pick_free_addr() -> Result<SocketAddr> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let addr = listener.local_addr()?;
    drop(listener);
    Ok(addr)
}

async fn connect_with_retry(
    addr: SocketAddr,
    timeout: Duration,
) -> Result<CoordinatorClient<tonic::transport::Channel>> {
    let deadline = Instant::now() + timeout;
    let url = format!("http://{addr}");
    loop {
        match CoordinatorClient::connect(url.clone()).await {
            Ok(client) => return Ok(client),
            Err(err) => {
                if Instant::now() >= deadline {
                    return Err(anyhow::anyhow!(
                        "timed out connecting to coordinator {}: {err}",
                        addr
                    ));
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }
    }
}

async fn register_once(addr: SocketAddr) -> Result<Result<(), tonic::Status>> {
    let mut client = connect_with_retry(addr, Duration::from_secs(3)).await?;
    let req = RegisterNodeRequest {
        job_id: JOB_ID.to_string(),
        node_id: NODE_ID.to_string(),
        caps: Some(NodeCaps {
            max_fetch_concurrency: 4,
            max_decode_concurrency: 2,
            max_inflight_bytes: 64 * 1024 * 1024,
            max_ram_bytes: 8 * 1024 * 1024 * 1024,
        }),
        resume_from: Vec::new(),
    };
    match client.register_node(req).await {
        Ok(_) => Ok(Ok(())),
        Err(status) => Ok(Err(status)),
    }
}

async fn register_until_ok(addr: SocketAddr, timeout: Duration) -> Result<()> {
    let deadline = Instant::now() + timeout;
    loop {
        match register_once(addr).await? {
            Ok(()) => return Ok(()),
            Err(status) => {
                if Instant::now() >= deadline {
                    anyhow::bail!(
                        "timed out registering node on {} (last error: {})",
                        addr,
                        format_args!("{}: {}", status.code(), status.message())
                    );
                }
                tokio::time::sleep(Duration::from_millis(75)).await;
            }
        }
    }
}

async fn wait_for_promoted_client(
    addr: SocketAddr,
    timeout: Duration,
) -> Result<CoordinatorClient<tonic::transport::Channel>> {
    let deadline = Instant::now() + timeout;
    loop {
        let mut client = connect_with_retry(addr, Duration::from_secs(2)).await?;
        match client
            .heartbeat(HeartbeatRequest {
                job_id: JOB_ID.to_string(),
                node_id: NODE_ID.to_string(),
                unix_time_ms: unix_time_ms(),
                stats: None,
            })
            .await
        {
            Ok(_) => return Ok(client),
            Err(status) if status.code() == Code::FailedPrecondition => {
                if Instant::now() >= deadline {
                    anyhow::bail!("timed out waiting for follower {} to promote", addr);
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            Err(status) => {
                return Err(anyhow::anyhow!(
                    "heartbeat failed while waiting for promotion: {status}"
                ));
            }
        }
    }
}

fn ranges_overlap(a: (u64, u64), b: (u64, u64)) -> bool {
    a.0 < b.1 && b.0 < a.1
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn leader_failover_reloads_state_store_and_continues() -> Result<()> {
    let bin = resolve_coordinator_bin()?;
    let root = unique_dir("gate")?;
    let manifest_store_root = root.join("manifests");
    std::fs::create_dir_all(&manifest_store_root)?;
    let leader_lease_path = root.join("ha").join("lease.txt");
    let state_store_path = root.join("state").join("snapshot.json");

    let a_addr = pick_free_addr()?;
    let b_addr = pick_free_addr()?;
    let mut proc_a = CoordinatorProc::spawn(
        Path::new(&bin),
        a_addr,
        &manifest_store_root,
        &leader_lease_path,
        &state_store_path,
        "coord-a",
    )?;
    register_until_ok(a_addr, Duration::from_secs(6)).await?;
    let mut leader_client = connect_with_retry(a_addr, Duration::from_secs(4)).await?;
    let lease_resp = leader_client
        .request_lease(RequestLeaseRequest {
            job_id: JOB_ID.to_string(),
            node_id: NODE_ID.to_string(),
            want: 1,
        })
        .await?
        .into_inner();
    let lease = lease_resp
        .leases
        .first()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("expected leader to grant lease"))?;
    let range = lease
        .range
        .clone()
        .ok_or_else(|| anyhow::anyhow!("lease missing range"))?;
    let cursor_1 = range.start_id.saturating_add(3);
    leader_client
        .report_progress(ReportProgressRequest {
            job_id: JOB_ID.to_string(),
            node_id: NODE_ID.to_string(),
            lease_id: lease.lease_id.clone(),
            cursor: cursor_1,
            delivered_samples: 3,
            delivered_bytes: 3 * 1024,
            unix_time_ms: unix_time_ms(),
        })
        .await?;

    let _proc_b = CoordinatorProc::spawn(
        Path::new(&bin),
        b_addr,
        &manifest_store_root,
        &leader_lease_path,
        &state_store_path,
        "coord-b",
    )?;

    // Follower should be up before we kill active leader.
    let _ = connect_with_retry(b_addr, Duration::from_secs(4)).await?;
    proc_a.kill_and_wait()?;

    // Wait for follower promotion and force state-store replay via a mutating call.
    let mut follower_client = wait_for_promoted_client(b_addr, Duration::from_secs(8)).await?;

    let snap = follower_client
        .get_job_snapshot(GetJobSnapshotRequest {
            job_id: JOB_ID.to_string(),
        })
        .await?
        .into_inner();
    let replayed = snap
        .live_leases
        .iter()
        .find(|l| l.lease_id == lease.lease_id)
        .ok_or_else(|| anyhow::anyhow!("expected lease to be present after failover replay"))?;
    assert!(
        replayed.cursor >= cursor_1,
        "expected replayed cursor >= {cursor_1}, got {}",
        replayed.cursor
    );

    let cursor_2 = cursor_1.saturating_add(2);
    follower_client
        .report_progress(ReportProgressRequest {
            job_id: JOB_ID.to_string(),
            node_id: NODE_ID.to_string(),
            lease_id: lease.lease_id.clone(),
            cursor: cursor_2,
            delivered_samples: 2,
            delivered_bytes: 2 * 1024,
            unix_time_ms: unix_time_ms(),
        })
        .await?;
    let snap2 = follower_client
        .get_job_snapshot(GetJobSnapshotRequest {
            job_id: JOB_ID.to_string(),
        })
        .await?
        .into_inner();
    let advanced = snap2
        .live_leases
        .iter()
        .find(|l| l.lease_id == lease.lease_id)
        .ok_or_else(|| anyhow::anyhow!("expected lease to remain live after second progress"))?;
    assert!(
        advanced.cursor >= cursor_2,
        "expected cursor to advance to >= {cursor_2}, got {}",
        advanced.cursor
    );

    let _ = std::fs::remove_dir_all(&root);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn leader_failover_preserves_no_overlap_for_new_leases() -> Result<()> {
    let bin = resolve_coordinator_bin()?;
    let root = unique_dir("no-overlap")?;
    let manifest_store_root = root.join("manifests");
    std::fs::create_dir_all(&manifest_store_root)?;
    let leader_lease_path = root.join("ha").join("lease.txt");
    let state_store_path = root.join("state").join("snapshot.json");

    let a_addr = pick_free_addr()?;
    let b_addr = pick_free_addr()?;
    let mut proc_a = CoordinatorProc::spawn(
        Path::new(&bin),
        a_addr,
        &manifest_store_root,
        &leader_lease_path,
        &state_store_path,
        "coord-a",
    )?;
    register_until_ok(a_addr, Duration::from_secs(6)).await?;
    let mut leader_client = connect_with_retry(a_addr, Duration::from_secs(4)).await?;

    let initial = leader_client
        .request_lease(RequestLeaseRequest {
            job_id: JOB_ID.to_string(),
            node_id: NODE_ID.to_string(),
            want: 2,
        })
        .await?
        .into_inner()
        .leases;
    assert_eq!(initial.len(), 2, "expected two initial leases");
    let initial_ranges: Vec<(u64, u64)> = initial
        .iter()
        .map(|l| {
            let r = l
                .range
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("initial lease missing range"))?;
            Ok((r.start_id, r.end_id))
        })
        .collect::<Result<_>>()?;

    let _proc_b = CoordinatorProc::spawn(
        Path::new(&bin),
        b_addr,
        &manifest_store_root,
        &leader_lease_path,
        &state_store_path,
        "coord-b",
    )?;
    let _ = connect_with_retry(b_addr, Duration::from_secs(4)).await?;
    proc_a.kill_and_wait()?;
    let mut follower = wait_for_promoted_client(b_addr, Duration::from_secs(8)).await?;

    let next = follower
        .request_lease(RequestLeaseRequest {
            job_id: JOB_ID.to_string(),
            node_id: NODE_ID.to_string(),
            want: 3,
        })
        .await?
        .into_inner()
        .leases;
    assert!(
        !next.is_empty(),
        "expected follower to grant additional non-overlapping work"
    );

    for lease in &next {
        let r = lease
            .range
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("new lease missing range"))?;
        let new_range = (r.start_id, r.end_id);
        for existing in &initial_ranges {
            assert!(
                !ranges_overlap(*existing, new_range),
                "overlap detected: existing {:?} new {:?}",
                existing,
                new_range
            );
        }
    }

    let _ = std::fs::remove_dir_all(&root);
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn leader_failover_accepts_duplicate_progress_replay() -> Result<()> {
    let bin = resolve_coordinator_bin()?;
    let root = unique_dir("dup-progress")?;
    let manifest_store_root = root.join("manifests");
    std::fs::create_dir_all(&manifest_store_root)?;
    let leader_lease_path = root.join("ha").join("lease.txt");
    let state_store_path = root.join("state").join("snapshot.json");

    let a_addr = pick_free_addr()?;
    let b_addr = pick_free_addr()?;
    let mut proc_a = CoordinatorProc::spawn(
        Path::new(&bin),
        a_addr,
        &manifest_store_root,
        &leader_lease_path,
        &state_store_path,
        "coord-a",
    )?;
    register_until_ok(a_addr, Duration::from_secs(6)).await?;
    let mut leader_client = connect_with_retry(a_addr, Duration::from_secs(4)).await?;

    let lease = leader_client
        .request_lease(RequestLeaseRequest {
            job_id: JOB_ID.to_string(),
            node_id: NODE_ID.to_string(),
            want: 1,
        })
        .await?
        .into_inner()
        .leases
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("expected initial lease"))?;
    let range = lease
        .range
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("lease missing range"))?;
    let cursor = range.start_id.saturating_add(4);
    leader_client
        .report_progress(ReportProgressRequest {
            job_id: JOB_ID.to_string(),
            node_id: NODE_ID.to_string(),
            lease_id: lease.lease_id.clone(),
            cursor,
            delivered_samples: 4,
            delivered_bytes: 4096,
            unix_time_ms: unix_time_ms(),
        })
        .await?;

    let _proc_b = CoordinatorProc::spawn(
        Path::new(&bin),
        b_addr,
        &manifest_store_root,
        &leader_lease_path,
        &state_store_path,
        "coord-b",
    )?;
    let _ = connect_with_retry(b_addr, Duration::from_secs(4)).await?;
    proc_a.kill_and_wait()?;
    let mut follower = wait_for_promoted_client(b_addr, Duration::from_secs(8)).await?;

    // Replay the same progress payload twice across the failover boundary.
    follower
        .report_progress(ReportProgressRequest {
            job_id: JOB_ID.to_string(),
            node_id: NODE_ID.to_string(),
            lease_id: lease.lease_id.clone(),
            cursor,
            delivered_samples: 4,
            delivered_bytes: 4096,
            unix_time_ms: unix_time_ms(),
        })
        .await?;
    follower
        .report_progress(ReportProgressRequest {
            job_id: JOB_ID.to_string(),
            node_id: NODE_ID.to_string(),
            lease_id: lease.lease_id.clone(),
            cursor,
            delivered_samples: 4,
            delivered_bytes: 4096,
            unix_time_ms: unix_time_ms(),
        })
        .await?;

    let snap = follower
        .get_job_snapshot(GetJobSnapshotRequest {
            job_id: JOB_ID.to_string(),
        })
        .await?
        .into_inner();
    let lease_after = snap
        .live_leases
        .iter()
        .find(|l| l.lease_id == lease.lease_id)
        .ok_or_else(|| anyhow::anyhow!("expected lease to remain live after duplicate replay"))?;
    assert_eq!(
        lease_after.cursor, cursor,
        "duplicate replay should keep cursor stable"
    );

    let _ = std::fs::remove_dir_all(&root);
    Ok(())
}
