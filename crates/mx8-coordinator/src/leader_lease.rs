use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use tokio::time::{sleep, Duration};

#[derive(Debug, Clone)]
pub struct LeaderLeaseConfig {
    pub lease_path: PathBuf,
    pub leader_id: String,
    pub lease_ttl_ms: u64,
    pub renew_interval_ms: u64,
}

#[derive(Debug, Clone)]
pub struct ObservedLeader {
    pub term: u64,
    pub leader_id: String,
    pub expires_unix_time_ms: u64,
}

impl ObservedLeader {
    pub fn is_expired(&self, now_unix_time_ms: u64) -> bool {
        self.expires_unix_time_ms <= now_unix_time_ms
    }
}

#[derive(Debug, Clone)]
pub enum LeaderTick {
    Leader { term: u64 },
    Follower { observed: Option<ObservedLeader> },
}

#[derive(Debug)]
pub struct LeaderGate {
    self_id: String,
    is_leader: AtomicBool,
    term: AtomicU64,
    observed_term: AtomicU64,
}

impl LeaderGate {
    pub fn new(self_id: String) -> Self {
        Self {
            self_id,
            is_leader: AtomicBool::new(false),
            term: AtomicU64::new(0),
            observed_term: AtomicU64::new(0),
        }
    }

    pub fn apply_tick(&self, tick: LeaderTick) {
        match tick {
            LeaderTick::Leader { term } => {
                self.term.store(term, Ordering::Relaxed);
                self.observed_term.store(term, Ordering::Relaxed);
                self.is_leader.store(true, Ordering::Relaxed);
            }
            LeaderTick::Follower { observed } => {
                self.is_leader.store(false, Ordering::Relaxed);
                if let Some(obs) = observed {
                    self.observed_term.store(obs.term, Ordering::Relaxed);
                }
            }
        }
    }

    pub fn fence_message(&self) -> String {
        format!(
            "not leader for mutating operation (self_id={}, local_term={}, observed_term={})",
            self.self_id,
            self.term.load(Ordering::Relaxed),
            self.observed_term.load(Ordering::Relaxed)
        )
    }

    pub fn is_leader(&self) -> bool {
        self.is_leader.load(Ordering::Relaxed)
    }

    pub fn term(&self) -> u64 {
        self.term.load(Ordering::Relaxed)
    }
}

#[derive(Debug, Clone)]
struct LeaderLeaseRecord {
    term: u64,
    leader_id: String,
    expires_unix_time_ms: u64,
}

impl LeaderLeaseRecord {
    fn encode(&self) -> String {
        format!(
            "v=1\nterm={}\nleader_id={}\nexpires_unix_time_ms={}\n",
            self.term, self.leader_id, self.expires_unix_time_ms
        )
    }

    fn decode(raw: &str) -> Result<Self> {
        let mut version: Option<u32> = None;
        let mut term: Option<u64> = None;
        let mut leader_id: Option<String> = None;
        let mut expires_unix_time_ms: Option<u64> = None;
        for line in raw.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Some((k, v)) = line.split_once('=') else {
                anyhow::bail!("invalid leader lease line: {line}");
            };
            match k.trim() {
                "v" => {
                    version = Some(
                        v.trim()
                            .parse::<u32>()
                            .map_err(|_| anyhow::anyhow!("invalid leader lease version"))?,
                    )
                }
                "term" => {
                    term = Some(
                        v.trim()
                            .parse::<u64>()
                            .map_err(|_| anyhow::anyhow!("invalid leader lease term"))?,
                    )
                }
                "leader_id" => leader_id = Some(v.trim().to_string()),
                "expires_unix_time_ms" => {
                    expires_unix_time_ms = Some(v.trim().parse::<u64>().map_err(|_| {
                        anyhow::anyhow!("invalid leader lease expires_unix_time_ms")
                    })?)
                }
                _ => anyhow::bail!("unknown leader lease field: {}", k.trim()),
            }
        }
        anyhow::ensure!(version == Some(1), "unsupported leader lease version");
        let term = term.ok_or_else(|| anyhow::anyhow!("missing leader lease term"))?;
        let leader_id =
            leader_id.ok_or_else(|| anyhow::anyhow!("missing leader lease leader_id"))?;
        let expires_unix_time_ms = expires_unix_time_ms
            .ok_or_else(|| anyhow::anyhow!("missing leader lease expires_unix_time_ms"))?;
        Ok(Self {
            term,
            leader_id,
            expires_unix_time_ms,
        })
    }
}

struct UpdateLockGuard {
    path: PathBuf,
}

impl Drop for UpdateLockGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn unix_time_ms() -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    now.as_millis() as u64
}

fn lock_path_for(lease_path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.update_lock", lease_path.display()))
}

fn try_acquire_update_lock(lease_path: &Path) -> Result<Option<UpdateLockGuard>> {
    if let Some(parent) = lease_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let lock_path = lock_path_for(lease_path);
    match std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&lock_path)
    {
        Ok(_f) => Ok(Some(UpdateLockGuard { path: lock_path })),
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => Ok(None),
        Err(err) => Err(err.into()),
    }
}

fn read_record(lease_path: &Path) -> Result<Option<LeaderLeaseRecord>> {
    if !lease_path.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(lease_path)
        .with_context(|| format!("failed reading {}", lease_path.display()))?;
    let record = LeaderLeaseRecord::decode(&raw)?;
    Ok(Some(record))
}

fn write_record(lease_path: &Path, rec: &LeaderLeaseRecord) -> Result<()> {
    if let Some(parent) = lease_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let tmp = PathBuf::from(format!("{}.tmp", lease_path.display()));
    std::fs::write(&tmp, rec.encode())?;
    std::fs::rename(&tmp, lease_path)?;
    Ok(())
}

pub fn tick_once(cfg: &LeaderLeaseConfig) -> Result<LeaderTick> {
    let now = unix_time_ms();
    let expires = now.saturating_add(cfg.lease_ttl_ms.max(1));

    let Some(_guard) = try_acquire_update_lock(&cfg.lease_path)? else {
        let observed = read_record(&cfg.lease_path)?.map(|r| ObservedLeader {
            term: r.term,
            leader_id: r.leader_id,
            expires_unix_time_ms: r.expires_unix_time_ms,
        });
        return Ok(LeaderTick::Follower { observed });
    };

    let current = read_record(&cfg.lease_path)?;
    match current {
        Some(rec) => {
            if !(ObservedLeader {
                term: rec.term,
                leader_id: rec.leader_id.clone(),
                expires_unix_time_ms: rec.expires_unix_time_ms,
            })
            .is_expired(now)
            {
                if rec.leader_id == cfg.leader_id {
                    let renewed = LeaderLeaseRecord {
                        term: rec.term,
                        leader_id: cfg.leader_id.clone(),
                        expires_unix_time_ms: expires,
                    };
                    write_record(&cfg.lease_path, &renewed)?;
                    return Ok(LeaderTick::Leader { term: renewed.term });
                }
                return Ok(LeaderTick::Follower {
                    observed: Some(ObservedLeader {
                        term: rec.term,
                        leader_id: rec.leader_id,
                        expires_unix_time_ms: rec.expires_unix_time_ms,
                    }),
                });
            }

            let promoted = LeaderLeaseRecord {
                term: rec.term.saturating_add(1),
                leader_id: cfg.leader_id.clone(),
                expires_unix_time_ms: expires,
            };
            write_record(&cfg.lease_path, &promoted)?;
            Ok(LeaderTick::Leader {
                term: promoted.term,
            })
        }
        None => {
            let created = LeaderLeaseRecord {
                term: 1,
                leader_id: cfg.leader_id.clone(),
                expires_unix_time_ms: expires,
            };
            write_record(&cfg.lease_path, &created)?;
            Ok(LeaderTick::Leader { term: created.term })
        }
    }
}

pub async fn run_loop(cfg: LeaderLeaseConfig, gate: Arc<LeaderGate>) {
    let interval = Duration::from_millis(cfg.renew_interval_ms.max(1));
    loop {
        match tick_once(&cfg) {
            Ok(tick) => gate.apply_tick(tick),
            Err(err) => {
                tracing::error!(
                    error = %err,
                    lease_path = %cfg.lease_path.display(),
                    "leader lease tick failed; fencing mutating APIs until leadership recovers"
                );
                gate.apply_tick(LeaderTick::Follower { observed: None });
            }
        }
        sleep(interval).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_lease_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "mx8-coord-leader-lease-{}-{}-{}.txt",
            name,
            std::process::id(),
            unix_time_ms()
        ))
    }

    #[test]
    fn leader_lease_election_and_fencing() -> Result<()> {
        let path = tmp_lease_path("election");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(lock_path_for(&path));

        let a = LeaderLeaseConfig {
            lease_path: path.clone(),
            leader_id: "a".to_string(),
            lease_ttl_ms: 200,
            renew_interval_ms: 50,
        };
        let b = LeaderLeaseConfig {
            lease_path: path.clone(),
            leader_id: "b".to_string(),
            lease_ttl_ms: 200,
            renew_interval_ms: 50,
        };

        match tick_once(&a)? {
            LeaderTick::Leader { term } => assert_eq!(term, 1),
            _ => anyhow::bail!("a should be leader"),
        }
        match tick_once(&b)? {
            LeaderTick::Follower { observed } => {
                let obs = observed.ok_or_else(|| anyhow::anyhow!("missing observed leader"))?;
                assert_eq!(obs.term, 1);
                assert_eq!(obs.leader_id, "a");
            }
            _ => anyhow::bail!("b should be follower"),
        }

        // Force lease expiry, then b should promote with term+1.
        let expired = LeaderLeaseRecord {
            term: 1,
            leader_id: "a".to_string(),
            expires_unix_time_ms: unix_time_ms().saturating_sub(1),
        };
        write_record(&path, &expired)?;

        match tick_once(&b)? {
            LeaderTick::Leader { term } => assert_eq!(term, 2),
            _ => anyhow::bail!("b should become leader"),
        }
        match tick_once(&a)? {
            LeaderTick::Follower { observed } => {
                let obs = observed.ok_or_else(|| anyhow::anyhow!("missing observed leader"))?;
                assert_eq!(obs.term, 2);
                assert_eq!(obs.leader_id, "b");
            }
            _ => anyhow::bail!("a should be fenced follower"),
        }

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(lock_path_for(&path));
        Ok(())
    }
}
