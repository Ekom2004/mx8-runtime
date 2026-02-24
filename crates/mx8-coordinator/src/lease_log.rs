//! Append-only write-ahead log for completed lease ranges.
//!
//! # Format (newline-delimited text)
//! ```text
//! H v1 <manifest_hash>   -- header, written once when a new file is created
//! C <start_id> <end_id>          -- range [start_id, end_id) fully completed
//! P <start_id> <end_id> <cursor> -- accepted cursor for a live/requeued range
//! ```
//!
//! # Crash recovery semantics
//! On coordinator restart:
//! 1. All `C` lines are replayed into completed ranges.
//! 2. All `P` lines are replayed into a per-range-end max cursor index.
//! 3. Startup range construction skips completed ranges and resumes partial ranges from the
//!    max durable cursor when present.
//!
//! This gives **deterministic crash recovery within a single epoch** with bounded replay:
//! completed ranges are not re-issued, and partially progressed ranges resume from the latest
//! durable cursor.
//!
//! # Durability note
//! Each append calls `sync_data()` to flush to the storage device before returning.
//! Completions are infrequent (O(total_blocks / world_size) per epoch), so the overhead
//! is negligible.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use tokio::io::AsyncWriteExt;

/// Write-ahead log for completed lease ranges.
pub struct LeaseLog {
    file: tokio::fs::File,
    path: PathBuf,
}

#[derive(Debug, Default)]
pub struct RecoveredLeaseState {
    pub completed_ranges: HashSet<(u64, u64)>,
    pub progress_by_end: HashMap<u64, u64>,
}

impl LeaseLog {
    /// Open or create a lease log at `path`.
    ///
    /// - If the file exists, replays all `C` lines to rebuild the completed-range set.
    /// - If the file is new, writes a header line with `manifest_hash`.
    ///
    /// Returns `(log, recovered_state)`.
    pub async fn open(path: &Path, manifest_hash: &str) -> Result<(Self, RecoveredLeaseState)> {
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

        Ok((
            log,
            RecoveredLeaseState {
                completed_ranges: replay.completed,
                progress_by_end: replay.progress_by_end,
            },
        ))
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
            if let Some(rest) = line.strip_prefix("P ") {
                let parts = rest.split_whitespace().collect::<Vec<_>>();
                if parts.len() == 3 {
                    let parsed = (
                        parts[0].trim().parse::<u64>(),
                        parts[1].trim().parse::<u64>(),
                        parts[2].trim().parse::<u64>(),
                    );
                    if let (Ok(start), Ok(end), Ok(cursor)) = parsed {
                        if start < end && cursor >= start && cursor <= end {
                            state
                                .progress_by_end
                                .entry(end)
                                .and_modify(|prev| *prev = (*prev).max(cursor))
                                .or_insert(cursor);
                        }
                    }
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

    /// Durably record accepted cursor progress for range `[start_id, end_id)`.
    pub async fn append_progress(&mut self, start_id: u64, end_id: u64, cursor: u64) -> Result<()> {
        self.write_raw(&format!("P {start_id} {end_id} {cursor}\n"))
            .await
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
    progress_by_end: HashMap<u64, u64>,
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

        let (_log, recovered) = LeaseLog::open(&path, "abc123").await.unwrap();
        assert!(recovered.completed_ranges.is_empty());
        assert!(recovered.progress_by_end.is_empty());

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

        let (_log2, recovered) = LeaseLog::open(&path, "abc123").await.unwrap();
        assert!(recovered.completed_ranges.contains(&(0, 65536)));
        assert!(recovered.completed_ranges.contains(&(65536, 131072)));
        assert!(!recovered.completed_ranges.contains(&(131072, 196608)));

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

        let (_log, recovered) = LeaseLog::open(&path, "abc").await.unwrap();
        assert!(recovered.completed_ranges.contains(&(0, 65536)));
        assert!(recovered.completed_ranges.contains(&(65536, 131072)));
        assert_eq!(recovered.completed_ranges.len(), 2);

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

    #[tokio::test]
    async fn replay_progress_tracks_max_cursor_per_range_end() {
        let path = tmp_path("progress");
        let _ = std::fs::remove_file(&path);

        std::fs::write(
            &path,
            "H v1 abc\nP 0 100 20\nP 20 100 60\nP 60 100 40\nP 100 200 180\n",
        )
        .unwrap();

        let (_log, recovered) = LeaseLog::open(&path, "abc").await.unwrap();
        assert_eq!(recovered.progress_by_end.get(&100).copied(), Some(60));
        assert_eq!(recovered.progress_by_end.get(&200).copied(), Some(180));

        let _ = std::fs::remove_file(&path);
    }
}
