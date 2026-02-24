//! Append-only write-ahead log for completed lease ranges.
//!
//! # Format (newline-delimited text)
//! ```text
//! H v1 <manifest_hash>   -- header, written once when a new file is created
//! C <start_id> <end_id>  -- range [start_id, end_id) fully completed
//! ```
//!
//! # Crash recovery semantics
//! On coordinator restart:
//! 1. All `C` lines are replayed → build a `HashSet<(u64, u64)>` of completed ranges.
//! 2. The initial block partition skips any range whose exact `(start_id, end_id)` pair
//!    appears in that set.
//! 3. In-flight leases at crash time are *never* logged, so they are re-issued after restart
//!    (safe: agents receive `NotFound` for stale lease IDs and re-request new leases).
//!
//! This gives **at-least-once semantics within a single epoch**: every range is delivered at
//! least once, and ranges that were cleanly completed before the crash are not re-issued.
//!
//! # Durability note
//! Each append calls `sync_data()` to flush to the storage device before returning.
//! Completions are infrequent (O(total_blocks / world_size) per epoch), so the overhead
//! is negligible.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use tokio::io::AsyncWriteExt;

/// Write-ahead log for completed lease ranges.
pub struct LeaseLog {
    file: tokio::fs::File,
    path: PathBuf,
}

impl LeaseLog {
    /// Open or create a lease log at `path`.
    ///
    /// - If the file exists, replays all `C` lines to rebuild the completed-range set.
    /// - If the file is new, writes a header line with `manifest_hash`.
    ///
    /// Returns `(log, completed_ranges)`.
    pub async fn open(path: &Path, manifest_hash: &str) -> Result<(Self, HashSet<(u64, u64)>)> {
        let replay = Self::replay(path)?;

        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent).await?;
            }
        }

        let is_new = !path.exists();
        let file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await?;

        let mut log = Self {
            file,
            path: path.to_owned(),
        };

        if is_new || replay.is_empty_file {
            log.write_raw(&format!("H v1 {manifest_hash}\n")).await?;
        } else {
            match replay.header_manifest_hash.as_deref() {
                Some(existing) if existing == manifest_hash => {}
                Some(existing) => {
                    bail!(
                        "lease log header hash mismatch: expected '{}', found '{}'",
                        manifest_hash,
                        existing
                    );
                }
                None => {
                    bail!("lease log is missing required header line: H v1 <manifest_hash>");
                }
            }
        }

        Ok((log, replay.completed))
    }

    /// Read an existing log file, returning completed ranges and the parsed header hash.
    /// Returns empty state if the file does not exist.
    fn replay(path: &Path) -> Result<ReplayState> {
        if !path.exists() {
            return Ok(ReplayState::default());
        }
        let content = std::fs::read_to_string(path)?;
        let mut state = ReplayState {
            is_empty_file: content.trim().is_empty(),
            ..ReplayState::default()
        };

        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("H v1 ") {
                let header_hash = rest.trim().to_string();
                if let Some(existing) = &state.header_manifest_hash {
                    if existing != &header_hash {
                        bail!(
                            "lease log contains conflicting headers: '{}' vs '{}'",
                            existing,
                            header_hash
                        );
                    }
                } else {
                    state.header_manifest_hash = Some(header_hash);
                }
                continue;
            }
            if let Some(rest) = line.strip_prefix("C ") {
                let mut parts = rest.splitn(2, ' ');
                if let (Some(s), Some(e)) = (parts.next(), parts.next()) {
                    if let (Ok(start), Ok(end)) = (s.trim().parse::<u64>(), e.trim().parse::<u64>())
                    {
                        state.completed.insert((start, end));
                    }
                }
            }
        }
        Ok(state)
    }

    /// Durably record that range `[start_id, end_id)` has been fully completed.
    ///
    /// Calls `sync_data()` before returning so the record survives a coordinator crash.
    pub async fn append_completed(&mut self, start_id: u64, end_id: u64) -> Result<()> {
        self.write_raw(&format!("C {start_id} {end_id}\n")).await
    }

    /// Path of the log file on disk.
    pub fn path(&self) -> &Path {
        &self.path
    }

    async fn write_raw(&mut self, s: &str) -> Result<()> {
        self.file.write_all(s.as_bytes()).await?;
        self.file.sync_data().await?;
        Ok(())
    }
}

#[derive(Default)]
struct ReplayState {
    completed: HashSet<(u64, u64)>,
    header_manifest_hash: Option<String>,
    is_empty_file: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("mx8-lease-log-{name}-{}.log", std::process::id()))
    }

    #[tokio::test]
    async fn new_file_returns_empty_completed_set() {
        let path = tmp_path("new");
        let _ = std::fs::remove_file(&path);

        let (_log, completed) = LeaseLog::open(&path, "abc123").await.unwrap();
        assert!(completed.is_empty());

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn replay_reads_completed_ranges() {
        let path = tmp_path("replay");
        let _ = std::fs::remove_file(&path);

        // Write some completions then reopen.
        {
            let (mut log, _) = LeaseLog::open(&path, "abc123").await.unwrap();
            log.append_completed(0, 65536).await.unwrap();
            log.append_completed(65536, 131072).await.unwrap();
        }

        let (_log2, completed) = LeaseLog::open(&path, "abc123").await.unwrap();
        assert!(completed.contains(&(0, 65536)));
        assert!(completed.contains(&(65536, 131072)));
        assert!(!completed.contains(&(131072, 196608)));

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn replay_ignores_malformed_lines() {
        let path = tmp_path("malformed");
        let _ = std::fs::remove_file(&path);

        // Manually write a file with some bad lines.
        std::fs::write(
            &path,
            "H v1 abc\nC 0 65536\nC bad line\nC 65536 131072\ngarbage\n",
        )
        .unwrap();

        let (_log, completed) = LeaseLog::open(&path, "abc").await.unwrap();
        assert!(completed.contains(&(0, 65536)));
        assert!(completed.contains(&(65536, 131072)));
        assert_eq!(completed.len(), 2);

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn open_fails_on_mismatched_manifest_hash() {
        let path = tmp_path("mismatch");
        let _ = std::fs::remove_file(&path);

        std::fs::write(&path, "H v1 hash-a\nC 0 10\n").unwrap();
        let msg = match LeaseLog::open(&path, "hash-b").await {
            Ok(_) => panic!("expected mismatch error"),
            Err(err) => format!("{err:#}"),
        };
        assert!(msg.contains("header hash mismatch"));

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn open_fails_when_header_missing() {
        let path = tmp_path("missing-header");
        let _ = std::fs::remove_file(&path);

        std::fs::write(&path, "C 0 10\n").unwrap();
        let msg = match LeaseLog::open(&path, "hash-a").await {
            Ok(_) => panic!("expected missing-header error"),
            Err(err) => format!("{err:#}"),
        };
        assert!(msg.contains("missing required header"));

        let _ = std::fs::remove_file(&path);
    }
}
